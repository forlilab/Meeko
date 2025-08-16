#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
eol="\n"
import logging
import sys
import warnings
from typing import Union
from typing import Optional, Any

import numpy as np
import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdMolInterchange

from .utils.jsonutils import rdkit_mol_from_json, tuple_to_string, string_to_tuple
from .utils.jsonutils import convert_to_tuple_keyed_dict
from .utils.jsonutils import convert_to_int_keyed_dict
from .utils.jsonutils import BaseJSONParsable
from .utils import rdkitutils
from .utils import utils
from .utils.geomutils import calcDihedral
from .utils.pdbutils import PDBAtomInfo

try:
    from misctools import StereoIsomorphism
except ImportError as _import_misctools_error:
    _has_misctools = False
else:
    _has_misctools = True


from .utils import rdkitutils

logger = logging.getLogger(__name__)

# region DEFAULT VALUES
DEFAULT_PDBINFO = None
DEFAULT_CHARGE = 0.0
DEFAULT_COORD = np.array([0.0, 0.0, 0.0], dtype="float")
DEFAULT_ATOMIC_NUM = None
DEFAULT_ATOM_TYPE = None
DEFAULT_IS_IGNORE = False
DEFAULT_GRAPH = []

DEFAULT_BOND_ROTATABLE = False
DEFAULT_BOND_BREAKABLE = False

DEFAULT_RING_CLOSURE_BONDS_REMOVED = []
DEFAULT_RING_CLOSURE_PSEUDOS_BY_ATOM = defaultdict
# endregion


# region Helper Data Organization Classes
class UniqAtomParams:
    """
    A helper class used to keep parameters organized in a particular way that lets them be more usable.

    Attributes
    ----------
    params: list[]
        can be thought of as rows
    param_names: list[]
        can be thought of as columns
    """

    def __init__(self):
        self.params = []  # aka rows
        self.param_names = []  # aka column names

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates an UniqAtomParams object, populates it with information from the input dictionary, then returns
        the new object.

        Parameters
        ----------
        dictionary: dict()
            A dictionary containing the keys "params" and "param_names", where the value for "params" is parseable as
            rows and the value for "param_names" contains the corresponding column data.

        Returns
        -------
        A populated UniqAtomParams object
        """
        uap = UniqAtomParams()
        uap.params = [row.copy() for row in dictionary["params"]]
        uap.param_names = dictionary["param_names"].copy()
        return uap

    def get_indices_from_atom_params(self, atom_params):
        """
        Retrieves the indices of specific atom parameters in the UniqAtomParams object.

        Parameters
        ----------
        atom_params: dict()
            A dict with keys that correspond to the param names already in the UniqAtomParams object. The values are
            lists that should all be the same size, and

        Returns
        -------
        A list of indices corresponding to the order of parameters in the atom_params value lists that indicates the
        index of that "row" of parameters in UniqAtomParams params.
        """
        nr_items = set([len(values) for key, values in atom_params.items()])
        if len(nr_items) != 1:
            raise RuntimeError(
                f"all lists in atom_params must have same length, got {nr_items}"
            )
        if set(atom_params) != set(self.param_names):
            msg = f"parameter names in atom_params differ from internal ones\n"
            msg += f"  - in atom_params: {set(atom_params)}"
            msg += f"  - internal: {set(self.param_names)}"
            raise RuntimeError(msg)
        nr_items = nr_items.pop()
        param_idxs = []
        for i in range(nr_items):
            row = [atom_params[key][i] for key in self.param_names]
            param_index = None
            for j, existing_row in enumerate(self.params):
                if row == existing_row:
                    param_index = j
                    break
            param_idxs.append(param_index)
        return param_idxs

    def add_parameter(self, new_param_dict):
        # remove None values to avoid a column with only Nones
        new_param_dict = {k: v for k, v in new_param_dict.items() if v is not None}
        incoming_keys = set(new_param_dict.keys())
        existing_keys = set(self.param_names)
        new_keys = incoming_keys.difference(existing_keys)
        for new_key in new_keys:
            self.param_names.append(new_key)
            for row in self.params:
                row.append(None)  # fill in empty "cell" in new "column"

        new_row = []
        for key in self.param_names:
            value = new_param_dict.get(key, None)
            new_row.append(value)

        if len(new_keys) == 0:  # try to match with existing row
            for index, row in enumerate(self.params):
                if row == new_row:
                    return index

        # if we are here, we didn't match
        new_row_index = len(self.params)
        self.params.append(new_row)
        return new_row_index

    def add_molsetup(
        self, molsetup, atom_params=None, add_atomic_nr=False, add_atom_type=False
    ):
        if "charge" in molsetup.atom_params or "atom_type" in molsetup.atom_params:
            msg = '"charge" and "atom_type" found in molsetup.atom_params'
            msg += " but are hard-coded to store molsetup.charge and"
            msg += " molsetup.atom_type in the internal data structure"
            raise RuntimeError(msg)
        if atom_params is None:
            atom_params = molsetup.atom_params
        param_idxs = []
        for atom in molsetup.atoms:
            if atom.is_ignore:
                param_idx = None
            else:
                p = {k: v[atom.index] for (k, v) in molsetup.atom_params.items()}
                if add_atomic_nr:
                    if "atomic_nr" in p:
                        raise RuntimeError(
                            "trying to add atomic_nr but it's already in atom_params"
                        )
                    p["atomic_nr"] = atom.atomic_num
                if add_atom_type:
                    if "atom_type" in p:
                        raise RuntimeError(
                            "trying to add atom_type but it's already in atom_params"
                        )
                    p["atom_type"] = atom.atom_type
                param_idx = self.add_parameter(p)
            param_idxs.append(param_idx)
        return param_idxs


@dataclass
class Atom(BaseJSONParsable):
    index: int
    pdbinfo: Union[str, PDBAtomInfo] = DEFAULT_PDBINFO
    charge: float = DEFAULT_CHARGE
    coord: np.ndarray = field(default_factory=lambda: np.zeros(3))
    atomic_num: int = DEFAULT_ATOMIC_NUM
    atom_type: str = DEFAULT_ATOM_TYPE
    is_ignore: bool = DEFAULT_IS_IGNORE
    graph: list[int] = field(default_factory=list)

    is_dummy: bool = False
    is_pseudo_atom: bool = False
    
    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "Atom") -> Optional[dict[str, Any]]:

        output_dict = {
            "index": obj.index,
            "pdbinfo": obj.pdbinfo,
            "charge": obj.charge,
            "coord": obj.coord.tolist(),  # converts coord from numpy array to lists
            "atomic_num": obj.atomic_num,
            "atom_type": obj.atom_type,
            "is_ignore": obj.is_ignore,
            "graph": obj.graph,
            "is_dummy": obj.is_dummy,
            "is_pseudo_atom": obj.is_pseudo_atom,
        }
        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = {
            "index",
            "pdbinfo",
            "charge",
            "coord",
            "atomic_num",
            "atom_type",
            "is_ignore",
            "graph",
            "is_dummy",
            "is_pseudo_atom",
        }

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]): 

        # Constructs an atom object from the provided keys.
        index = obj["index"]
        pdbinfo = PDBAtomInfo(*obj["pdbinfo"])
        charge = obj["charge"]
        coord = np.asarray(obj["coord"])
        atomic_num = obj["atomic_num"]
        atom_type = obj["atom_type"]
        is_ignore = obj["is_ignore"]
        graph = obj["graph"]
        is_dummy = obj["is_dummy"]
        is_pseudo_atom = obj["is_pseudo_atom"]
        output_atom = cls(
            index,
            pdbinfo,
            charge,
            coord,
            atomic_num,
            atom_type,
            is_ignore,
            graph,
            is_dummy,
            is_pseudo_atom,
        )
        return output_atom
    # endregion


@dataclass
class Bond(BaseJSONParsable):
    canon_id: tuple[int, int] = field(init=False)  # Excluded from __init__
    index1: int
    index2: int
    rotatable: bool = DEFAULT_BOND_ROTATABLE
    breakable: bool = DEFAULT_BOND_BREAKABLE

    def __post_init__(self):
        self.canon_id = self.get_bond_id(self.index1, self.index2)
    
    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "Bond") -> Optional[dict[str, Any]]:
        
        output_dict = {
                "canon_id": tuple_to_string(obj.canon_id),
                "index1": obj.index1,
                "index2": obj.index2,
                "rotatable": obj.rotatable,
                "breakable": obj.breakable,
        }
        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = {"canon_id", "index1", "index2", "rotatable"}

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]): 

        # Constructs a bond object from the provided keys.
        index1 = obj["index1"]
        index2 = obj["index2"]
        rotatable = obj["rotatable"]
        breakable = obj.get("breakable", DEFAULT_BOND_BREAKABLE)
        output_bond = cls(index1, index2, rotatable, breakable)
        return output_bond
    # endregion

    @staticmethod
    def get_bond_id(idx1: int, idx2: int):
        """
        Generates a consistent, "canonical", bond id from a pair of atom indices in the graph.

        Parameters
        ----------
        idx1: int
            atom index of one of the atoms in the bond
        idx2: int
            atom index of the other atom in the bond

        Returns
        -------
        canon_id: tuple
            a tuple of the two indices in their canonical order.
        """
        idx_min = min(idx1, idx2)
        idx_max = max(idx1, idx2)
        return idx_min, idx_max

@dataclass
class Ring(BaseJSONParsable):
    ring_id: tuple
    
    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "Ring") -> Optional[dict[str, Any]]:

        output_dict = {
            "ring_id": tuple_to_string(obj.ring_id),
        }
        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = {"ring_id"}
    
    @classmethod
    def _decode_object(cls, obj: dict[str, Any]): 

        # Constructs a Ring object from the provided keys.
        ring_id = string_to_tuple(obj["ring_id"], int)
        output_ring = cls(ring_id)
        return output_ring
    # endregion


@dataclass
class RingClosureInfo:
    bonds_removed: list = field(default_factory=list)
    pseudos_by_atom: dict = DEFAULT_RING_CLOSURE_PSEUDOS_BY_ATOM


@dataclass
class Restraint(BaseJSONParsable):
    atom_index: int
    target_coords: tuple[float, float, float]
    kcal_per_angstrom_square: float
    delay_angstroms: float

    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "Restraint") -> Optional[dict[str, Any]]:

        output_dict = {
            "atom_index": obj.atom_index,
            "target_coords": tuple_to_string(obj.target_coords),
            "kcal_per_angstrom_square": obj.kcal_per_angstrom_square,
            "delay_angstroms": obj.delay_angstroms,
        }
        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = {
            "atom_index",
            "target_coords",
            "kcal_per_angstrom_square",
            "delay_angstroms",
        }

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]): 

        # Constructs a Restraint object from the provided keys.
        atom_index = obj["atom_index"]
        target_coords = tuple(obj["target_coords"])
        kcal_per_angstrom_square = obj["kcal_per_angstrom_square"]
        delay_angstroms = obj["delay_angstroms"]
        output_restraint = cls(
            atom_index, target_coords, kcal_per_angstrom_square, delay_angstroms
        )
        return output_restraint
    # endregion
    
    def copy(self):
        new_target_coords = (
            self.target_coords[0],
            self.target_coords[1],
            self.target_coords[2],
        )
        new_restraint = Restraint(
            self.atom_index,
            new_target_coords,
            self.kcal_per_angstrom_square,
            self.delay_angstroms,
        )
        return new_restraint
# endregion


class MoleculeSetup(BaseJSONParsable):
    """
    Base MoleculeSetup Class, provides a way to store information about molecules for a number of purposes.

    Attributes
    ----------
    name: str
    pseudoatom_count: int

    atoms: list[Atom]
    bond_info: dict[tuple, Bond]
    rings: dict
    ring_closure_info: RingClosureInfo
    rotamers: list[dict]

    atom_params: dict
    restraints: list[Restraint]
    flexibility_model: dict
    """

    # region CLASS CONSTANTS
    PSEUDOATOM_ATOMIC_NUM = 0
    # endregion

    def __init__(self, name: str = None):

        # Initializer attributes 
        self.name: str = name

        # (JSON-bound) computed attributes
        self.pseudoatom_count: int = 0
        self.atoms: list[Atom] = []
        self.bond_info: dict[tuple, Bond] = {}
        self.rings: dict[tuple, Ring] = {}
        self.ring_closure_info = RingClosureInfo([], {})
        self.rotamers: list[dict] = []  # TODO: revisit rotamer implementation
        self.atom_params: dict = {}
        self.restraints: list = (
            []
        )  # TODO: determine whether restraints are being used anymore

        # TODO: redesign flexibility model to resolve some of the circular imports and to make it more structured
        self.flexibility_model = None  # from flexibility_model - from flexibility.py
    
    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "MoleculeSetup") -> Optional[dict[str, Any]]:
            
        output_dict = {
            "name": obj.name,
            "pseudoatom_count": obj.pseudoatom_count,
            "atoms": [Atom.json_encoder(x) for x in obj.atoms],
            "bond_info": {
                tuple_to_string(k): Bond.json_encoder(v)
                for k, v in obj.bond_info.items()
            },
            "rings": {
                tuple_to_string(k): Ring.json_encoder(v)
                for k, v in obj.rings.items()
            },
            "ring_closure_info": obj.ring_closure_info.__dict__,
            "rotamers": [{tuple_to_string(k): v for k, v in rotamer.items()} for rotamer in obj.rotamers],
            "atom_params": obj.atom_params,
            "restraints": [
                Restraint.json_encoder(x) for x in obj.restraints
            ],
            "flexibility_model": obj.flexibility_model,
        }
        # Addressing some flexibility model-specific structures.
        if "rigid_body_connectivity" in obj.flexibility_model:
            new_rigid_body_conn_dict = {
                tuple_to_string(k): v
                for k, v in obj.flexibility_model["rigid_body_connectivity"].items()
            }
            output_dict["flexibility_model"] = {
                k: (
                    v
                    if k != "rigid_body_connectivity"
                    else new_rigid_body_conn_dict
                )
                for k, v in obj.flexibility_model.items()
            }
        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = {
            "name",
            "pseudoatom_count",
            "atoms",
            "bond_info",
            "rings",
            "ring_closure_info",
            "rotamers",
            "atom_params",
            "restraints",
            "flexibility_model",
        }
    
    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):

        # Constructs a MoleculeSetup object and restores the expected attributes
        name = obj["name"]
        molsetup = cls(name)

        molsetup.pseudoatom_count = obj["pseudoatom_count"]
        molsetup.atoms = [Atom.from_dict(x) for x in obj["atoms"]]
        molsetup.bond_info = {
            string_to_tuple(k, int): Bond.from_dict(v)
            for k, v in obj["bond_info"].items()
        }
        molsetup.rings = {
            string_to_tuple(k, int): Ring.from_dict(v) for k, v in obj["rings"].items()
        }
        molsetup.ring_closure_info = RingClosureInfo(
            obj["ring_closure_info"]["bonds_removed"],
            obj["ring_closure_info"]["pseudos_by_atom"],
        )
        molsetup.rotamers = [convert_to_tuple_keyed_dict(rotamer, int) for rotamer in obj["rotamers"]]
        molsetup.atom_params = obj["atom_params"]
        molsetup.restraints = [Restraint.from_dict(x) for x in obj["restraints"]]
        molsetup.flexibility_model = obj["flexibility_model"]
        if "rigid_body_connectivity" in molsetup.flexibility_model:
            tuples_rigid_body_connectivity = {
                string_to_tuple(k, int): string_to_tuple(v)
                for k, v in molsetup.flexibility_model[
                    "rigid_body_connectivity"
                ].items()
            }
            molsetup.flexibility_model["rigid_body_connectivity"] = (
                tuples_rigid_body_connectivity
            )

        for attribute in ["rigid_body_graph", "rigid_body_members", "rigid_index_by_atom"]: 
            if attribute in molsetup.flexibility_model:
                molsetup.flexibility_model[attribute] = convert_to_int_keyed_dict(
                    molsetup.flexibility_model[attribute]
                )

        return molsetup
    # endregion

    # region Manually Building A MoleculeSetup
    def add_atom(
        self,
        atom_index: int = None,
        overwrite: bool = False,
        pdbinfo: Union[str, PDBAtomInfo] = DEFAULT_PDBINFO,
        charge: float = DEFAULT_CHARGE,
        coord: np.ndarray = None,
        atomic_num: int = DEFAULT_ATOMIC_NUM,
        atom_type: str = DEFAULT_ATOM_TYPE,
        is_ignore: bool = DEFAULT_IS_IGNORE,
        graph: list[int] = None,
    ):
        """
        Adds an atom with all the specified attributes to the MoleculeSetup, either at the specified atom index, or by
        appending it to the internal list of atoms. Default values will be used for any attributes with unspecified
        values.

        Parameters
        ----------
        atom_index: int
            atom index in the MoleculeSetup
        overwrite: bool
            can we overwrite other atoms may be in the same atom index as this one
        pdbinfo: str
            pdb string for the atom
        coord: np.ndarray
            the atom's coordinates
        charge: float
            partial charge to be loaded for the atom
        atomic_num: int
            the atomic number of the atom
        atom_type: str
            TODO: needs info
        is_ignore: bool
            ignore flag for the atom
        graph: List[List[int]]

        Returns
        -------
        None

        Raises
        ------
        RuntimeException
            If the user tries to overwrite an existing atom without explicitly allowing overwrites.
        """
        # If atom index is specified and it would be trying to overwrite an existing atom in the atom list, raises a
        # Runtime Exception
        insert_disallowed = len(self.atoms) > atom_index and not overwrite
        if (
            atom_index is not None
            and insert_disallowed
            and not self.atoms[atom_index].is_dummy
        ):
            raise RuntimeError(
                "ADD_ATOM Error: the atom_index [%d] is already occupied (use 'overwrite' to force)"
            )

        # If atom index is not specified, appends the new atom to the end of the current atom list
        if atom_index is None:
            atom_index = len(self.atoms)

        # Inserts dummy atoms if a specified atom index is greater than the current length of the atom list
        while atom_index > len(self.atoms):
            self.atoms.append(Atom(len(self.atoms), is_dummy=True))

        # Creates and adds new atom to the atom list
        if coord is None:
            coord = deepcopy(DEFAULT_COORD)
        if graph is None:
            graph = deepcopy(DEFAULT_GRAPH)
        new_atom = Atom(
            atom_index,
            pdbinfo,
            charge,
            coord,
            atomic_num,
            atom_type,
            is_ignore,
            graph,
        )
        if atom_index < len(self.atoms):
            self.atoms[atom_index] = new_atom
            return
        self.atoms.append(new_atom)
        return

    def add_pseudoatom(
        self,
        pdbinfo: Union[str, PDBAtomInfo] = DEFAULT_PDBINFO,
        charge: float = DEFAULT_CHARGE,
        coord: np.ndarray = None,
        atom_type: str = DEFAULT_ATOM_TYPE,
        is_ignore: bool = DEFAULT_IS_IGNORE,
        anchor_list: list[int] = None,
        rotatable: bool = False,
    ):
        """
        Adds a pseudoatom with all the specified attributes to the MoleculeSetup. Default values will be used for any
        attributes with unspecified values. Multiple bonds can be specified to support the centroids of aromatic rings.
        If rotatable, makes the anchor atom rotatable to allow the pseudoatom movement.

        Parameters
        ----------
        pdbinfo: str
            PDB string for the pseudoatom.
        charge: float
            partial charge for the pseudoatom
        coord: np.ndarray
            the pseudoatom's coordinates
        atom_type: str
            TODO: needs info
        is_ignore: bool
            ignore flag for the pseudoatom
        anchor_list: list[int]
            a list of ints indicating the multiple bonds that can be specified as input
        rotatable: bool
            flag indicating if the anchor atom should be marked as rotatable to allow the pseudoatom movement.

        Returns
        -------
        pseudoatom_index: int
            The atom_index of the added pseudoatom

        Raises
        ------
        RuntimeError:
            When the incorrect number of anchors of pseudoatoms are found in rigid_groups in the flexibility model

        """
        # Places the atom at the end of the atom list.
        pseudoatom_index = len(self.atoms)
        # Creates the atom and marks it as a pseudoatom
        if coord is None:
            coord = deepcopy(DEFAULT_COORD)
        new_pseudoatom = Atom(
            pseudoatom_index,
            pdbinfo=pdbinfo,
            charge=charge,
            coord=coord,
            atomic_num=self.PSEUDOATOM_ATOMIC_NUM,
            atom_type=atom_type,
            is_ignore=is_ignore,
            is_pseudo_atom=True,
        )
        self.atoms.append(new_pseudoatom)
        # Adds bonds for all of the provided anchor atoms
        if anchor_list is not None:
            for anchor in anchor_list:
                self.add_bond(pseudoatom_index, anchor, rotatable=rotatable)
        # If there are no specified anchor atoms,
        if not self.flexibility_model or not anchor_list:
            return pseudoatom_index
        # TODO: revise this logic
        # If there is a flexibility model in the MoleculeSetup, adds the psuedoatom to the flexibility model's rigid
        # group tracking
        rigid_groups_indices = []
        for anchor in anchor_list:
            for rigid_index, members in self.flexibility_model[
                "rigid_body_members"
            ].items():
                if anchor in members:
                    rigid_groups_indices.append(rigid_index)
        if len(rigid_groups_indices) != 1:
            raise RuntimeError(
                f"anchors of pseudo atom found in {len(rigid_groups_indices)} rigid_groups (must be 1)"
            )
        rigid_index = rigid_groups_indices[0]
        self.flexibility_model["rigid_body_members"][rigid_index].append(
            pseudoatom_index
        )
        # returns the pseudoatom index
        return pseudoatom_index

    def delete_atom(self, atom_index: int):
        """
        Clears the atom data at a specified atom index and replaces the atom with a dummy atom.

        Parameters
        ----------
        atom_index: int
            atom index to replace with a dummy atom

        Returns
        -------
        None
        """
        blank_atom = Atom(atom_index, is_dummy=True)
        self.atoms[atom_index] = blank_atom
        return

    def add_bond(
        self,
        atom_index_1: int,
        atom_index_2: int,
        rotatable: bool = DEFAULT_BOND_ROTATABLE,
    ) -> None:
        """
        Creates a bond and adds it to all the internal data structures where atom bonds are being tracked.

        Parameters
        ----------
        atom_index_1: int
            Atom index of one of the atoms in the bond
        atom_index_2: int
            Atom index of the other atom in the bond
        rotatable: bool
            Indicates whether the bond is rotatable

        Returns
        -------
        None

        Raises
        ------
        IndexError:
            When one or more the given bond atom indices do not exist in the MoleculeSetup
        """
        # Checks that both of the atom indices provided are valid indices, otherwise throws an error
        if len(self.atoms) <= atom_index_1 or len(self.atoms) <= atom_index_2:
            raise IndexError(
                "ADD_BOND: provided atom indices outside the range of atoms currently in MoleculeSetup"
            )
        # Adds each atom to the other's bond graph
        if atom_index_2 not in self.atoms[atom_index_1].graph:
            self.atoms[atom_index_1].graph.append(atom_index_2)
        if atom_index_1 not in self.atoms[atom_index_2].graph:
            self.atoms[atom_index_2].graph.append(atom_index_1)
        # Creates new bond object and uses its internal canonical bond id to add it to MoleculeSetup bond tracking.
        new_bond = Bond(atom_index_1, atom_index_2, rotatable)
        self.bond_info[new_bond.canon_id] = new_bond
        return

    def delete_bond(self, atom_index_1: int, atom_index_2: int):
        """
        Deletes a bond from the molecule setup.

        Parameters
        ----------
        atom_index_1: int
            The atom index of one of the atoms in the bond to delete
        atom_index_2: int
            The atom index of the other atom in the bond to delete

        Returns
        -------
        None
        """
        # Gets canon bond id for the bond to delete
        canon_bond_id = Bond.get_bond_id(atom_index_1, atom_index_2)
        # Deletes the bond from the internal bond table
        del self.bond_info[canon_bond_id]
        # Removes the bond from each atom's graph
        self.atoms[atom_index_1].graph.remove(atom_index_2)
        self.atoms[atom_index_2].graph.remove(atom_index_1)
        return

    def add_rotamers(
        self, index_list: list[(int, int, int, int)], angle_list: np.ndarray
    ):
        """
        Adds rotamers to the internal record of rotamers.

        Parameters
        ----------
        index_list: list[(int, int, int, int)]
        angle_list: np.ndarray

        Returns
        -------
        None
        """
        # It's unclear how this will work without the coordinates in the moleculesetup. food for thought.
        # TODO: address issues with the lack of coords and add detail in function comment
        rotamers = {}
        for (idx1, idx2, idx3, idx4), angle in zip(index_list, angle_list):
            bond_id = Bond.get_bond_id(idx2, idx3)
            if bond_id in rotamers:
                raise RuntimeError("repeated bond %d-$d" % bond_id)
            if not self.bond_info[bond_id].rotatable:
                raise RuntimeError(
                    "trying to add rotamer for non rotatable bond %d-%d" % bond_id
                )
            # d0 = calcDihedral(xyz[i1], xyz[i2], xyz[i3], xyz[i4])
            dihedral = 0  # TODO: fix this
            rotamers[bond_id] = angle - dihedral
        self.rotamers.append(rotamers)
        return

    def delete_rotamers(
        self,
        bond_id_list: list[tuple] = None,
        index_list: list[(int, int, int, int)] = None,
    ):
        """
        Deletes rotamers from the internal list of rotamers, either by using bond ids or by generating bond ids from a
        list of input indices.

        Parameters
        ----------
        bond_id_list: list[tuple]
        index_list: list[(int, int, int, int)]

        Returns
        -------
        None
        """
        # loops through the index list, generates bond ids from the provided indices and adds them to the bond_id_list
        if index_list is not None:
            for idx1, idx2, idx3, idx4 in index_list:
                bond_id = Bond.get_bond_id(idx2, idx3)
                bond_id_list.append(bond_id)
        # deletes all bond_ids in bond_id_list from self.rotamers
        if bond_id_list is not None:
            for bond_id in bond_id_list:
                if bond_id in self.rotamers:
                    del self.rotamers[bond_id]
        return


    @property
    def true_atom_count(self):
        """
        Counts the number of atoms in the MoleculeSetup that are not pseudo_atoms or marked as dummy atoms

        Returns
        -------
        count: int
            The number of atoms currently in the MoleculeSetup that are not dummy atoms or pseudo_atoms.
        """
        count = 0
        for atom in self.atoms:
            if not atom.is_pseudo_atom and not atom.is_dummy:
                count += 1
        return count

    # this might void the graph connections and everything in bonds, might need to add a dict of the changes we're
    # making and then use that save this for a future push.
    def clean_atoms(self, remove_pseudoatoms: bool = False):
        """
        Cleans dummy and potentially also pseudoatoms from the MoleculeSetup so only true atoms remain. Note that this
        is pretty slow and should not be done often.

        Parameters
        ----------
        remove_pseudoatoms: bool
            Indicates if we want to remove all the pseudoatoms from the MoleculeSetup.

        Returns
        -------
        The number of atoms removed from the MoleculeSetup.
        """
        new_atoms = []
        removed_atom_count = 0
        atom_index_mapping = {}

        for atom in self.atoms:
            if remove_pseudoatoms and atom.is_pseudo_atom:
                removed_atom_count += 1
                continue
            if atom.is_dummy:
                removed_atom_count += 1
                continue
            atom.index = atom.index - removed_atom_count
            new_atoms.append(atom)
        self.atoms = new_atoms
        if remove_pseudoatoms:
            self.pseudoatom_count = 0
        return removed_atom_count

    # endregion

    # region Getters and Setters

    def get_pdbinfo(self, atom_index: int):
        """
        Retrieves the PDB Info string for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        pdbinfo: str
            A string containing the pdb information for the atom

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain
            data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_PDBINFO: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].pdbinfo

    def get_charge(self, atom_index: int):
        """
        Retrieves the partial charge for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        charge: float
            The charge associated with the atom

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain
            data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_CHARGE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].charge

    def get_coord(self, atom_index: int):
        """
        Retrieves the coordinates for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        coord: np.ndarray
            The coordinates associated with the atom.

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain
            data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_CHARGE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].coord

    def get_atomic_num(self, atom_index: int):
        """
        Retrieves the atomic number for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        atomic_num: int
            The atomic number associated with an atom.

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain
            data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_ATOMIC_NUM: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].atomic_num

    def get_atom_type(self, atom_index: int):
        """
        Retrieves the atom type for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        charge: str
            The atom index associated with the atom

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_ATOM_TYPE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].atom_type

    def set_atom_type(self, atom_index: int, atom_type: str) -> None:
        """
        Sets the atom type for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to set atom_type for.
        atom_type
            Atom type string to set.

        Returns
        -------
        None

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "SET_ATOM_TYPE: provided atom index is out of range or is a dummy atom"
            )
        self.atoms[atom_index].atom_type = atom_type
        return

    def set_atom_type_from_uniq_atom_params(
        self, uniq_atom_params: UniqAtomParams, prefix: str
    ):
        """
        Uses a UniqAtomParams object to set the atom_type attribute for atoms in the Molecule Setup object. Adds the specified prefix
        to each of the atom_type attributes pulled from UniqAtomParams.

        Parameters
        ----------
        uniq_atom_params: UniqAtomParams
            A uniq atom params object to extract atom_type from
        prefix: string
            A prefix to be appended to all the atom_type attributes

        Returns
        -------
        None
        """
        # Gets a mapping from parameter indices in atom_params to those in uniq_atom_params
        parameter_indices = uniq_atom_params.get_indices_from_atom_params(
            self.atom_params
        )
        # Checks that we have the correct number of retrieved indices.
        if len(parameter_indices) != len(self.atoms):
            raise RuntimeError(
                "Number of parameters ({len(parameter_indices)}) not equal to number of atoms in Molecule Setup ({len(self.atom_type)})"
            )
        # Loops through the indices in parameter indices and sets atom types with the input prefix
        for i, j in enumerate(parameter_indices):
            self.atom_type[i] = f"{prefix}{j}"
        return None

    def get_is_ignore(self, atom_index: int):
        """
        Retrieves the is_ignore boolean for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        is_ignore: bool
            Indicates whether a particular atom should be ignored

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain
            data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_IS_IGNORE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].is_ignore

    def get_neighbors(self, atom_index: int):
        """
        Retrieves the partial charge for the atom with the specified atom index.

        Parameters
        ----------
        atom_index: int
            Atom index to retrieve data for.

        Returns
        -------
        graph: list[int]
            The graph of the atoms connections to other atoms.

        Raises
        ------
        IndexError:
            When the provided atom index does not exist in the MoleculeSetup or the atom index does not contain
            data.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_GRAPH: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].graph

    # endregion

    def merge_terminal_atoms(self, indices) -> None:
        """
        Primarily for merging hydrogens, but will merge the data for any atom or pseudoatom that is bonded to only one
        other atom.

        Parameters
        ----------
        indices: list
            A list of indices to merge

        Returns
        -------
        None
        """
        for index in indices:
            if len(self.get_neighbors(index)) != 1:
                msg = "Atempted to merge atom %d with %d neighbors. "
                msg += "Only atoms with one neighbor can be merged."
                msg = msg % (index + 1, self.get_neighbors(index))
                raise RuntimeError(msg)
            neighbor_index = self.get_neighbors(index)[0]
            self.atoms[neighbor_index].charge += self.get_charge(index)
            self.atoms[index].charge = 0.0
            self.atoms[index].is_ignore = True
        return

    # NOTE: This is a candidate for moving to utils
    @staticmethod
    def get_bonds_in_ring(ring: tuple) -> list[tuple]:
        """
        Takes as input a tuple of atom indices corresponding to atoms in a ring and returns a list of all the bonds ids
        in the ring.

        Parameters
        ----------
        ring: tuple
            A list of atom indices of the atoms in a ring.

        Returns
        -------
        A list of canonical bond id tuples for the bonds in the ring.
        """
        bonds = []
        num_indices = len(ring)
        for i in range(num_indices):
            bond = (ring[i], ring[(i + 1) % num_indices])
            bond = Bond.get_bond_id(bond[0], bond[1])
            bonds.append(bond)
        return bonds

    def _recursive_graph_walk(
        self, idx: int, collected: list[int] = None, exclude: list[int] = None
    ):
        """
        Recursively walks through a molecular graph and returns bond-connected subgroups.

        Parameters
        ----------
        idx: int
            atom index to start the recursive walk from
        collected: list[int]
            a list of connected subgroups
        exclude: list[int]
            a list of atom indices to exclude from the final walk.

        Returns
        -------
        A list of ints indicating the subgroups that are bond-connected.
        """
        if collected is None:
            collected = []
        if exclude is None:
            exclude = []
        for neighbor in self.get_neighbors(idx):
            if neighbor in collected or neighbor in exclude:
                continue
            collected.append(neighbor)
            self._recursive_graph_walk(neighbor, collected, exclude)
        return collected

    def write_coord_string(self) -> str:
        """
        Constructs and returns a string of all atom and pseudoatom elements and coordinates.

        Returns
        -------
        A string of all atom and pseudoatom elements and coordinates.
        """
        n = len(self.atoms)
        output_string = "%d\n\n" % n
        for index in range(n):
            element = "Ne"
            if self.atoms[index].is_dummy:
                continue
            if not self.atoms[index].is_pseudo_atom:
                element = utils.mini_periodic_table[self.atoms[index].atomic_num]
            x, y, z = self.atoms[index].coord
            output_string += "%3s %12.6f %12.6f %12.6f\n" % (element, x, y, z)
        return output_string

    def show(self) -> None:
        """
        Legacy function to print the contents of a MoleculeSetup in a human-readable format.

        Returns
        -------
        None
        """
        total_charge = 0

        print("Molecule Setup\n")
        print(
            "==============[ ATOMS ]==================================================="
        )
        print("idx  |          coords            | charge |ign| atype    | connections")
        print(
            "-----+----------------------------+--------+---+----------+--------------- . . . "
        )
        for atom in self.atoms:
            print(
                "% 4d | % 8.3f % 8.3f % 8.3f | % 1.3f | %d"
                % (
                    atom.index,
                    atom.coord[0],
                    atom.coord[1],
                    atom.coord[2],
                    atom.charge,
                    atom.is_ignore,
                ),
                "| % -8s |" % atom.atom_type,
                atom.graph,
            )
            total_charge += atom.charge
        print(
            "-----+----------------------------+--------+---+----------+--------------- . . . "
        )
        print("  TOT CHARGE: %3.3f" % total_charge)

        print("\n==============[ BONDS ]================")
        # For sanity users, we won't show those keys for now
        keys_to_not_show = ["type"]
        for bond_id, bond in list(self.bond_info.items()):
            t = ", ".join(
                "%s: %s" % (i, j)
                for i, j in bond.__dict__.items()
                if i not in keys_to_not_show
            )
            print("% 8s - " % str(bond_id), t)
        print("")
        return


# region External Toolkit Support
class MoleculeSetupExternalToolkit(ABC):
    """
    Additional functions and requirements to extend the MoleculeSetup class in order to use it with  external toolkits
    such as RDKit and OpenBabel.

    Required Attributes
    -------------------
    dihedral_interactions: list
        A list of fourier series [add detail]
    """

    @staticmethod
    def are_fourier_series_identical(series1: list, series2: list) -> bool:
        """
        Compares two fourier series represented as lists of dictionaries.

        Parameters
        ----------
        series1: list[dict]
            The first fourier series to compare.
        series2: list[dict]
            The second fourier series to compare.

        Returns
        -------
        A bool indicicating whether the fourier series are equal.
        """
        # Gets the indices of both series by periodicity and checks for equality
        index_by_periodicity1 = {
            series1[index]["periodicity"]: index for index in range(len(series1))
        }
        index_by_periodicity2 = {
            series2[index]["periodicity"]: index for index in range(len(series2))
        }
        if index_by_periodicity1 != index_by_periodicity2:
            return False
        # After establishing equality of the indices, loops through periodicity abd checks that the values stored in
        # each fourier series dictionary are equal.
        for periodicity in index_by_periodicity1:
            index1 = index_by_periodicity1[periodicity]
            index2 = index_by_periodicity2[periodicity]
            for key in ["k", "phase", "periodicity"]:
                if series1[index1][key] != series2[index2][key]:
                    return False
        return True

    def add_dihedral_interaction(self, fourier_series):
        """
        Adds a safe copy of the input fourier series to the dihedral_interactions list if the fourier series is not
        already in the list.

        Parameters
        ----------
        fourier_series: list[dict]

        Returns
        -------
        index: int
            The index of the input fourier series in the dihedral interactions list.
        """
        index = 0
        for existing_fs in self.dihedral_interactions:
            if self.are_fourier_series_identical(existing_fs, fourier_series):
                return index
            index += 1
        safe_copy = json.loads(json.dumps(fourier_series))
        self.dihedral_interactions.append(safe_copy)
        return index

    @abstractmethod
    def init_atom(self, compute_gasteiger_charges, read_charges_from_prop, coords):
        pass

    @abstractmethod
    def init_bond(self):
        pass

    @abstractmethod
    def get_mol_name(self):
        pass

    @abstractmethod
    def find_pattern(self, smarts: str):
        pass

    @abstractmethod
    def get_smiles_and_order(self):
        pass

    pass


class RDKitMoleculeSetup(MoleculeSetup, MoleculeSetupExternalToolkit, BaseJSONParsable):
    """
    Subclass of MoleculeSetup, used to represent MoleculeSetup objects working with RDKit objects

    Attributes
    ----------
    mol : rdkit.Chem.rdchem.Mol
        An RDKit Mol object to base the Molecule Setup on.
    modified_atom_positions: list
        List of dictionaries where keys are atom indices, Used to store sets of coordinates, e.g. docked poses, as
        dictionaries indexed by the atom index, because not all atoms need to have new coordinates specified.
        Unspecified hydrogen positions bonded to modified heavy atom positions are to be calculated "on-the-fly".
    dihedral_interactions: list[]
        A list of unique fourier_series, each of which are represented as a list of dictionaries.
    dihedral_partaking_atoms: dict()
        a mapping from tuples of atom indices to the indices in dihedral_interactions
    dihedral_labels: dict()
        a mapping from tuples of atom indices to dihedral labels
    atom_to_ring_id: dict()
        mapping of atom index to ring id of each atom belonging to the ring
    rmsd_symmetry_indices: tuple
        Tuples of the indices of the molecule's atoms that match a substructure query. needs info.

    Methods
    -------
    from_mol()
        constructor for the RDKitMoleculeSetup object (consider adapting to init?)
    """

    def __init__(self, name: str = None,
                 source: "MoleculeSetup" = None):
        
        # Initializer attributes 
        super().__init__(name)

        if source:
            if isinstance(source, MoleculeSetup):
                for key, value in source.__dict__.items():
                    setattr(self, key, deepcopy(value))
            else:
                raise TypeError("Expected 'source' to be an instance of MoleculeSetup, got type: {}".format(type(source)))

        # (JSON-bound) computed attributes
        self.mol = None
        self.modified_atom_positions = []
        self.dihedral_interactions: list[dict] = []
        self.dihedral_partaking_atoms: dict = {}
        self.dihedral_labels: dict = {}
        self.atom_to_ring_id = {}
        self.rmsd_symmetry_indices = ()

    # region JSON-interchange functions
    @classmethod
    def json_encoder(cls, obj: "RDKitMoleculeSetup") -> Optional[dict[str, Any]]:

        output_dict = MoleculeSetup.json_encoder(obj)

        output_dict["mol"] = rdMolInterchange.MolToJSON(obj.mol)
        output_dict["modified_atom_positions"] = obj.modified_atom_positions
        output_dict["dihedral_interactions"] = obj.dihedral_interactions
        output_dict["dihedral_partaking_atoms"] = {tuple_to_string(k): v for k,v in obj.dihedral_partaking_atoms.items()}
        output_dict["dihedral_labels"] = {tuple_to_string(k): v for k,v in obj.dihedral_labels.items()}
        output_dict["atom_to_ring_id"] = obj.atom_to_ring_id
        output_dict["rmsd_symmetry_indices"] = obj.rmsd_symmetry_indices

        return output_dict
    
    # Keys to check for deserialized JSON 
    expected_json_keys = frozenset(
        MoleculeSetup.expected_json_keys.union({
            "mol",
            "modified_atom_positions",
            "dihedral_interactions",
            "dihedral_partaking_atoms",
            "dihedral_labels",
            "atom_to_ring_id",
            "rmsd_symmetry_indices",
        })
    )
    
    @classmethod
    def _decode_object(cls, obj: dict[str, Any]): 
        
        base_molsetup = MoleculeSetup.from_dict(obj)
        rdkit_molsetup = cls(source = base_molsetup)

        # Restores RDKitMoleculeSetup-specific attributes from the json dict
        rdkit_molsetup.mol = rdkit_mol_from_json(obj["mol"])
        rdkit_molsetup.modified_atom_positions = list(map(int, obj["modified_atom_positions"]))
        rdkit_molsetup.dihedral_interactions = obj["dihedral_interactions"]
        rdkit_molsetup.dihedral_partaking_atoms = convert_to_tuple_keyed_dict(obj["dihedral_partaking_atoms"], int)
        rdkit_molsetup.dihedral_labels = convert_to_tuple_keyed_dict(obj["dihedral_labels"], int)
        rdkit_molsetup.atom_to_ring_id = {
            int(k): [string_to_tuple(t) for t in v]
            for k, v in obj["atom_to_ring_id"].items()
        }
        rdkit_molsetup.rmsd_symmetry_indices = list(map(string_to_tuple, obj["rmsd_symmetry_indices"]))
        return rdkit_molsetup
    # endregion

    def copy(self):
        """
        Returns a copy of the current RDKitMoleculeSetup.
        """
        newsetup = RDKitMoleculeSetup()
        
        for key, value in self.__dict__.items():
            if key == "mol":
                # Create a new RDKit molecule object
                newsetup.mol = Chem.Mol(self.mol) if self.mol else None
            else:
                # Deep copy other attributes
                setattr(newsetup, key, deepcopy(value))
        
        return newsetup

    @classmethod
    def from_mol(
        cls,
        mol: Chem.Mol,
        keep_chorded_rings: bool = False,
        keep_equivalent_rings: bool = False,
        compute_gasteiger_charges: bool = True,
        read_charges_from_prop: str = None,
        conformer_id: int = -1,
    ):
        """

        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
            RDKit Mol object to build the RDKitMoleculeSetup from.
        keep_chorded_rings: bool
        keep_equivalent_rings: bool
        compute_gasteiger_charges: bool
        read_charges_from_prop: str
        conformer_id: int

        Returns
        -------
        molsetup: RDKitMoleculeSetup
            A populated RDKitMoleculeSetup object

        Raises
        ------
        ValueError:
            If the RDKit Mol has implicit Hydrogens or if there are no conformers for the given RDKit Mol
        """
        # Checks if the input molecule is valid
        if cls.has_implicit_hydrogens(mol):
            raise ValueError("RDKit molecule has implicit Hs. Need explicit Hs.")
        if mol.GetNumConformers() == 0:
            raise ValueError(
                "RDKit molecule does not have a conformer. Need 3D coordinates."
            )

        # Gets the RDKit Conformer that we are going to load into the molecule setup
        rdkit_conformer = mol.GetConformer(conformer_id)
        if not rdkit_conformer.Is3D():
            warnings.warn(
                "RDKit molecule not labeled as 3D. This warning won't show again.", RuntimeWarning
            )
            RDKitMoleculeSetup.warned_not3D = True
        if mol.GetNumConformers() > 1 and conformer_id == -1:
            msg = "RDKit molecule has multiple conformers. Considering only the first one."
            warnings.warn(msg, RuntimeWarning)
        if len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError(f"RDKit molecule has {len(Chem.GetMolFrags(mol))} fragments. Must have 1.")
        if mol.HasQuery():
            raise ValueError(f"RDKit molecule has query. Check exotic fields (atom or bond) in SDF.")

        # Creating and populating the molecule setup with properties from RDKit as well as calculated values from our
        # functions
        molsetup = cls()
        molsetup.mol = mol
        molsetup.atom_true_count = molsetup.get_num_mol_atoms()
        molsetup.name = molsetup.get_mol_name()
        coords = rdkit_conformer.GetPositions()
        molsetup.init_atom(compute_gasteiger_charges, read_charges_from_prop, coords)
        molsetup.init_bond()
        molsetup.perceive_rings(keep_chorded_rings, keep_equivalent_rings)
        # molsetup.rmsd_symmetry_indices = cls.get_symmetries_for_rmsd(mol)

        # to store sets of coordinates, e.g. docked poses, as dictionaries indexed by
        # the atom index, because not all atoms need to have new coordinates specified
        # Unspecified hydrogen positions bonded to modified heavy atom positions
        # are to be calculated "on-the-fly".
        molsetup.modified_atom_positions = (
            []
        )  # list of dictionaries where keys are atom indices

        return molsetup

    def init_atom(self, compute_gasteiger_charges: bool, read_charges_from_prop: str, coords: list[np.ndarray]):
        """
        Generates information about the atoms in an RDKit Mol and adds them to an RDKitMoleculeSetup.

        Parameters
        ----------
        compute_gasteiger_charges: bool
            Indicates whether we should compute gasteiger charges.
        coords: list[np.ndarray]
            Atom coordinates for the RDKit Mol.

        Returns
        -------
        None
        """
        # extract/generate charges
        if compute_gasteiger_charges: 
            if read_charges_from_prop is not None: 
                raise ValueError(
                    "Conflicting options: compute_gasteiger_charges and read_charges_from_prop cannot both be set. "
                )
            charges = rdkitutils.compute_gasteiger_charges(self.mol)
        elif read_charges_from_prop is not None: 
            if not isinstance(read_charges_from_prop, str) or not read_charges_from_prop: 
                raise ValueError(
                    f"Invalid atom property name for read_charges_from_prop: expected a nonempty string (str), but got {type(read_charges_from_prop).__name__} instead. "
                )
            charges = [
                        float(atom.GetProp(read_charges_from_prop)) 
                        if atom.HasProp(read_charges_from_prop) else None
                        for atom in self.mol.GetAtoms()
                    ]
            if None in charges: 
                for idx, charge in enumerate(charges):
                    if charge is None:
                        logger.error(f"Charge at index {idx} is None.")
                raise ValueError(
                    f"The list of charges based on atom property name {read_charges_from_prop} contains None. "
                )  
        else:
            charges = [0.0] * self.mol.GetNumAtoms()
        # register atom
        for a in self.mol.GetAtoms():
            idx = a.GetIdx()
            self.add_atom(
                atom_index=idx,
                pdbinfo=rdkitutils.getPdbInfoNoNull(a),
                charge=charges[idx],
                coord=coords[idx],
                atomic_num=a.GetAtomicNum(),
                is_ignore=False,
            )

    def init_bond(self):
        """
        Uses the RDKit mol to initialize bond info for the RDKitMoleculeSetup

        Returns
        -------
        None
        """
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            rotatable = int(b.GetBondType()) == 1
            self.add_bond(idx1, idx2, rotatable=rotatable)

    def find_pattern(self, smarts: str):
        """
        Given a SMARTS pattern, finds substruct matches in the molecule.

        Parameters
        ----------
        smarts:
            A SMARTS string to find in the RDKit Mol object

        Returns
        -------
        The substruct matches in the RDKit Mol for the given SMARTS.
        """
        p = Chem.MolFromSmarts(smarts)
        nr_atoms = self.mol.GetNumAtoms()
        return self.mol.GetSubstructMatches(p, maxMatches=nr_atoms)

    def get_mol_name(self):
        """
        Gets the RDKit Mol's name from self.mol.

        Returns
        -------
        If the mol has a name, returns the name property.
        """
        if self.mol.HasProp("_Name"):
            return self.mol.GetProp("_Name")
        else:
            return None

    # TODO: Add more inline comments and clean up this function
    def get_smiles_and_order(self):
        """
        Returns the SMILES string and the mapping between atom indices in the SMILES and self.molof an atom after
        running RDKit's RemoveHs function.

        Returns
        -------
        smiles:
        order:
        """
        mol_no_ignore = self.mol

        # 3D SDF files written by other toolkits (OEChem, ChemAxon)
        # seem to not include the chiral flag in the bonds block, only in
        # the atoms block. RDKit ignores the atoms chiral flag as per the
        # spec. When reading SDF (e.g. from PubChem/PDB),
        # we may need to have RDKit assign stereo from coordinates, see:
        # https://sourceforge.net/p/rdkit/mailman/message/34399371/
        ps = Chem.RemoveHsParameters()
        # a user reported PDBbind Mol Blocks to have hcount=1 for Hs,
        # which adds a query to the RDKit H atom and then Chem.RemoveHs
        # does not remove Hs with queries by default
        # https://github.com/forlilab/Meeko/issues/62
        # https://github.com/rdkit/rdkit/issues/6615
        ps.removeWithQuery = True
        mol_noH = Chem.RemoveHs(mol_no_ignore, ps)  # imines (=NH) may become chiral
        # stereo imines [H]/N=C keep [H] after RemoveHs()
        # H isotopes also kept after RemoveHs()
        atomic_num_mol_noH = [atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
        noH_to_H = []
        parents_of_hs = {}
        for index, atom in enumerate(mol_no_ignore.GetAtoms()):
            if atom.GetAtomicNum() == 1:
                continue
            for i in range(len(noH_to_H), len(atomic_num_mol_noH)):
                if atomic_num_mol_noH[i] > 1:
                    break
                h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
                assert h_atom.GetAtomicNum() == 1
                neighbors = h_atom.GetNeighbors()
                assert len(neighbors) == 1
                parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
                noH_to_H.append("H")
            noH_to_H.append(index)
        extra_hydrogens = len(atomic_num_mol_noH) - len(noH_to_H)
        if extra_hydrogens > 0:
            assert set(atomic_num_mol_noH[len(noH_to_H) :]) == {1}
        for i in range(extra_hydrogens):
            h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
            assert h_atom.GetAtomicNum() == 1
            neighbors = h_atom.GetNeighbors()
            assert len(neighbors) == 1
            parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
            noH_to_H.append("H")

        # noH_to_H has the same length as the number of atoms in mol_noH
        # and each value is:
        #    - the index of the corresponding atom in mol, if value is integer
        #    - an hydrogen, if value is "H"
        # now, we need to replace those "H" with integers
        # "H" occur with stereo imine (e.g. [H]/N=C) and heavy Hs (e.g. [2H])
        hs_by_parent = {}
        for hidx, pidx in parents_of_hs.items():
            hs_by_parent.setdefault(pidx, [])
            hs_by_parent[pidx].append(hidx)
        for pidx, hidxs in hs_by_parent.items():
            siblings_of_h = [
                atom
                for atom in mol_no_ignore.GetAtomWithIdx(noH_to_H[pidx]).GetNeighbors()
                if atom.GetAtomicNum() == 1
            ]
            sortidx = [
                i
                for i, j in sorted(
                    list(enumerate(siblings_of_h)), key=lambda x: x[1].GetIdx()
                )
            ]
            if len(hidxs) == len(siblings_of_h):
                # This is the easy case, just map H to each other in the order they appear
                for i, hidx in enumerate(hidxs):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            elif len(hidxs) < len(siblings_of_h):
                # check hydrogen isotopes
                sibling_isotopes = [
                    siblings_of_h[sortidx[i]].GetIsotope()
                    for i in range(len(siblings_of_h))
                ]
                molnoH_isotopes = [mol_noH.GetAtomWithIdx(hidx) for hidx in hidxs]
                matches = []
                for i, sibling_isotope in enumerate(sibling_isotopes):
                    for hidx in hidxs[len(matches) :]:
                        if mol_noH.GetAtomWithIdx(hidx).GetIsotope() == sibling_isotope:
                            matches.append(i)
                            break
                if len(matches) != len(hidxs):
                    raise RuntimeError(
                        "Number of matched isotopes %d differs from query Hs: %d"
                        % (len(matches), len(hidxs))
                    )
                for hidx, i in zip(hidxs, matches):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            else:
                raise RuntimeError(
                    "nr of Hs in mol_noH bonded to an atom exceeds nr of Hs in mol_no_ignore"
                )

        smiles = Chem.MolToSmiles(mol_noH)
        order_string = mol_noH.GetProp("_smilesAtomOutputOrder")
        order_string = order_string.replace(",]", "]")  # remove trailing comma
        order = json.loads(order_string)  # mol_noH to smiles
        order = list(np.argsort(order))
        order = {noH_to_H[i]: order[i] + 1 for i in range(len(order))}  # 1-index
        
        # remove polar hydrogen isotopes from order 
        # this prevents them to appear in SMILES IDX but rather in H PARENT
        for atom in mol_noH.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetIsotope() > 0:
                order.pop(atom.GetIdx())
        
        return smiles, order

    # region Ring Construction
    def perceive_rings(self, keep_chorded_rings: bool, keep_equivalent_rings: bool):
        """
        Uses Hanser-Jauffret-Kaufmann exhaustive ring detection to find the rings in the molecule

        Parameters
        ----------
        keep_chorded_rings: bool
            Indicates whether we want to keep chorded rings
        keep_equivalent_rings: bool
            Indicates whether we want to keep equivalent rings

        Returns
        -------
        None
        """

        old_graph = {atom.index: atom.graph for atom in self.atoms}
        hjk_ring_detection = utils.HJKRingDetection(old_graph)
        rings = hjk_ring_detection.scan(keep_chorded_rings, keep_equivalent_rings)
        for ring_atom_indices in rings:
            ring_to_add = Ring(ring_atom_indices)
            self.rings[ring_atom_indices] = ring_to_add
        return

    # endregion

    def get_conformer_with_modified_positions(self, new_atom_positions):
        """
        Gets a conformer with the specified new atom positions.
        We operate on one conformer at a time because SetTerminalAtomPositions acts on all conformers of a molecule,
        and we do not want to guarantee that all conformers require the same set of terminal atoms to be updated.

        Parameters
        ----------
        new_atom_positions:
            The new atom positions we want to use.

        Returns
        -------
        new_conformer:
            A new conformer with the input new atom positions.
        """
        new_mol = Chem.Mol(self.mol)
        new_conformer = Chem.Conformer(self.mol.GetConformer())
        is_set_list = [False] * self.mol.GetNumAtoms()
        for atom_index, new_position in new_atom_positions.items():
            new_conformer.SetAtomPosition(atom_index, new_position)
            is_set_list[atom_index] = True
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(new_conformer, assignId=True)
        for atom_index, is_set in enumerate(is_set_list):
            if not is_set and new_mol.GetAtomWithIdx(atom_index).GetAtomicNum() == 1:
                neighbors = new_mol.GetAtomWithIdx(atom_index).GetNeighbors()
                if len(neighbors) != 1:
                    raise RuntimeError("Expected H to have one neighbors")
                Chem.SetTerminalAtomCoords(new_mol, atom_index, neighbors[0].GetIdx())
        return new_conformer

    def get_mol_with_modified_positions(self, new_atom_positions_list=None):
        """
        Modifies the stored RDKit Mol to a new set of atom positions, either those provided or the ones stored in
        self.modified_atom_positions, and returns the modified Mol object.

        Parameters
        ----------
        new_atom_positions_list:
            New atom positions to add to the RDKit Mol object.

        Returns
        -------
        new_mol: rdkit.Chem.rdchem.Mol
            A new RDKit Mol object with conformers that have the desired new atom positions.
        """
        if new_atom_positions_list is None:
            new_atom_positions_list = self.modified_atom_positions
        new_mol = Chem.Mol(self.mol)
        new_mol.RemoveAllConformers()
        for new_atom_positions in new_atom_positions_list:
            conformer = self.get_conformer_with_modified_positions(new_atom_positions)
            new_mol.AddConformer(conformer, assignId=True)
        return new_mol

    def get_num_mol_atoms(self):
        """
        Gets the number of atoms in the RDKit Mol object.

        Returns
        -------
        Number of atoms in the RDKit Mol object.
        """
        return self.mol.GetNumAtoms()

    def get_equivalent_atoms(self):
        """

        Returns
        -------

        """
        return list(Chem.CanonicalRankAtoms(self.mol, breakTies=False))

    @staticmethod
    def get_symmetries_for_rmsd(mol, max_matches=17):
        mol_noHs = Chem.RemoveHs(mol)
        matches = mol.GetSubstructMatches(
            mol_noHs, uniquify=False, maxMatches=max_matches
        )
        if len(matches) == max_matches:
            if mol.HasProp("_Name"):
                molname = mol.GetProp("_Name")
            else:
                molname = ""
            warnings.warn(
                "Found the maximum nr of matches (%d) in RDKitMolSetup.get_symmetries_for_rmsd"
                % max_matches, RuntimeWarning
            )
            warnings.warn(
                'Maybe this molecule is "too" symmetric? %s %s' % (molname, Chem.MolToSmiles(mol_noHs)),
                RuntimeWarning
            )
        return matches

    @staticmethod
    def has_implicit_hydrogens(mol):
        # based on needsHs from RDKit's AddHs.cpp
        for atom in mol.GetAtoms():
            nr_H_neighbors = 0
            for neighbor in atom.GetNeighbors():
                nr_H_neighbors += int(neighbor.GetAtomicNum() == 1)
            if atom.GetTotalNumHs(includeNeighbors=False) > nr_H_neighbors:
                return True
        return False

    def restrain_to(
        self, target_mol, kcal_per_angstrom_square=1.0, delay_angstroms=2.0
    ):
        """

        Parameters
        ----------
        target_mol
        kcal_per_angstrom_square
        delay_angstroms

        Returns
        -------

        """
        if not _has_misctools:
            raise ImportError(_import_misctools_error)
        stereo_isomorphism = StereoIsomorphism()
        mapping, idx = stereo_isomorphism(target_mol, self.mol)
        lig_to_drive = {b: a for (a, b) in mapping}
        num_real_atoms = target_mol.GetNumAtoms()
        target_positions = target_mol.GetConformer().GetPositions()
        for atom_index in range(len(mapping)):
            target_xyz = target_positions[lig_to_drive[atom_index]]
            restraint = Restraint(
                atom_index, target_xyz, kcal_per_angstrom_square, delay_angstroms
            )
            self.restraints.append(restraint)
        return

# endregion
