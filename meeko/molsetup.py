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
import sys
import warnings

import numpy as np
import rdkit.Chem
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolInterchange

from .utils import rdkitutils
from .utils import utils
from .utils.geomutils import calcDihedral
from .utils.pdbutils import PDBAtomInfo
from .receptor_pdbqt import PDBQTReceptor

try:
    from openbabel import openbabel as ob
    from .utils import obutils
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True

try:
    from misctools import StereoIsomorphism
except ImportError as _import_misctools_error:
    _has_misctools = False
else:
    _has_misctools = True


from .utils import rdkitutils

# region DEFAULT VALUES
DEFAULT_PDBINFO = None
DEFAULT_CHARGE = 0.0
DEFAULT_ATOMIC_NUM = None
DEFAULT_ATOM_TYPE = None
DEFAULT_IS_IGNORE = False
DEFAULT_IS_CHIRAL = False

DEFAULT_BOND_ORDER = 0
DEFAULT_BOND_ROTATABLE = False

DEFAULT_RING_CORNER_FLIP = False
DEFAULT_RING_GRAPH = defaultdict
DEFAULT_RING_IS_AROMATIC = False
# endregion


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
        if "q" in molsetup.atom_params or "atom_type" in molsetup.atom_params:
            msg = '"q" and "atom_type" found in molsetup.atom_params'
            msg += " but are hard-coded to store molsetup.charge and"
            msg += " molsetup.atom_type in the internal data structure"
            raise RuntimeError(msg)
        if atom_params is None:
            atom_params = molsetup.atom_params
        param_idxs = []
        for atom_index, ignore in enumerate(molsetup.atom_ignore):
            if ignore:
                param_idx = None
            else:
                p = {k: v[atom_index] for (k, v) in molsetup.atom_params.items()}
                if add_atomic_nr:
                    if "atomic_nr" in p:
                        raise RuntimeError(
                            "trying to add atomic_nr but it's already in atom_params"
                        )
                    p["atomic_nr"] = molsetup.atomic_num[atom_index]
                if add_atom_type:
                    if "atom_type" in p:
                        raise RuntimeError(
                            "trying to add atom_type but it's already in atom_params"
                        )
                    p["atom_type"] = molsetup.atom_type[atom_index]
                param_idx = self.add_parameter(p)
            param_idxs.append(param_idx)
        return param_idxs


class MoleculeSetup:

    # region CLASS CONSTANTS
    PSEUDOATOM_ATOMIC_NUM = 0
    # endregion

    def __init__(self, name: str = None, is_sidechain: bool = False):
        # Molecule Setup Identity
        self.name = name
        self.is_sidechain = is_sidechain
        self.true_atom_count = 0
        self.pseudo_atom_count = 0

        # Tracking atoms and bonds
        self.atoms: list[Atom] = []
        self.bond_info = {}
        self.rings = {}
        self.rotamers = []
        pass

    @classmethod
    def from_prmtop_inpcrd(cls, prmtop, crd_filename: str):
        # TODO: pull functionality over and clean up
        pass

    # region Manually Building A MoleculeSetup
    def add_atom(
        self,
        atom_index: int = None,
        overwrite: bool = False,
        pdbinfo: str = DEFAULT_PDBINFO,
        charge: float = DEFAULT_CHARGE,
        atomic_num: int = DEFAULT_ATOMIC_NUM,
        atom_type: str = DEFAULT_ATOM_TYPE,
        is_ignore: bool = DEFAULT_IS_IGNORE,
        is_chiral: bool = DEFAULT_IS_CHIRAL,
        graph: list[int] = field(default_factory=list),
    ):
        """
        Adds an atom with all the specified attributes to the MoleculeSetup, either at the specified atom index, or by
        appending it to the internal list of atoms. Default values will be used for any attributes with unspecified
        values.

        Parameters
        ----------
        atom_index: int
        overwrite: bool
        pdbinfo: str
        charge: float
        atomic_num: int
        atom_type: str
        is_ignore: bool
        is_chiral: bool
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
        dummy_atom_present = self.atoms[atom_index].is_dummy
        if atom_index is not None and insert_disallowed and not dummy_atom_present:
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
        new_atom = Atom(
            atom_index, pdbinfo, charge, atomic_num, atom_type, is_ignore, is_chiral, graph
        )
        if atom_index < len(self.atoms):
            self.atoms[atom_index] = new_atom
            return
        self.atoms.append(new_atom)
        return

    def add_pseudo_atom(
        self,
        pdbinfo: str = DEFAULT_PDBINFO,
        charge: float = DEFAULT_CHARGE,
        atom_type: str = DEFAULT_ATOM_TYPE,
        is_ignore: bool = DEFAULT_IS_IGNORE,
        anchor_list: list[int] = None,
        rotatable: bool = False,
        directional_vectors: list[int] = None,
    ):
        """

        Parameters
        ----------
        pdbinfo
        charge
        atom_type
        is_ignore
        anchor_list
        rotatable
        directional_vectors

        Returns
        -------

        """
        # Places the atom at the end of the atom list.
        pseudoatom_index = len(self.atoms)
        # Creates the atom and marks it as a pseudoatom
        new_pseudoatom = Atom(
            pseudoatom_index,
            pdbinfo=pdbinfo,
            charge=charge,
            atomic_num=self.PSEUDOATOM_ATOMIC_NUM,
            atom_type=atom_type,
            is_ignore=is_ignore,
            is_pseudo_atom=True,
        )
        # Adds bonds for all of the provided anchor atoms
        if anchor_list is not None:
            for anchor in anchor_list:
                self.add_bond(pseudoatom_index, anchor, rotatable=rotatable)
        # Adds directional vectors [Check what this is used for/if this is used]
        if directional_vectors is not None:
            self._add_interaction_vectors(pseudoatom_index, directional_vectors)
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
        order: int = DEFAULT_BOND_ORDER,
        rotatable: bool = DEFAULT_BOND_ROTATABLE,
    ):
        """

        Parameters
        ----------
        atom_index_1
        atom_index_2
        order
        rotatable

        Returns
        -------

        Raises
        ------

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
        new_bond = Bond(atom_index_1, atom_index_2, order, rotatable)
        self.bond_info[new_bond.canon_id] = new_bond
        return

    def delete_bond(self, atom_index_1: int, atom_index_2: int):
        """

        Parameters
        ----------
        atom_index_1
        atom_index_2

        Returns
        -------

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
        self, index_list: list[(int, int, int, int)], angles_list: np.ndarray
    ):
        # TODO: pull implementation over
        pass

    def count_true_atoms(self):
        for atom in self.atoms:
            if not atom.is_pseudo_atom and not atom.is_dummy:
                self.true_atom_count += 1
        return

    def _add_interaction_vectors(self, atom_index: int, vector_list: list[np.array]):
        """
        Adds input vector list to the list of directional ineraction vectors for the specified atom.

        Parameters
        ----------
        atom_index: int
            index of the atom to add the vectors to
        vector_list: list[np.array]
            a list of directional interaction vectors

        Returns
        -------
        None

        Raises
        ------
        IndexError
            if the specified atom index does not exist or is a dummy atom.
        """
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "INTERACTION_VECTORS: provided atom index is out of range or is a dummy atom"
            )
        for vector in vector_list:
            self.atoms[atom_index].interaction_vectors.append(vector)
        return

    # endregion

    # region Getters and Setters

    def get_pdbinfo(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_PDBINFO: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].pdbinfo

    def get_charge(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_CHARGE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].charge

    def get_atomic_num(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_ATOMIC_NUM: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].atomic_num

    def get_atom_type(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_ATOM_TYPE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].atom_type

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
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_IS_IGNORE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].is_ignore

    def get_is_chiral(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_IS_CHIRAL: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].is_chiral

    def get_neighbors(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_GRAPH: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].graph

    def get_interaction_vectors(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_INTERACTION_VECTORS: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].interaction_vectors

    # endregion

    # def get_bonds_in_ring
    # def walk_recursive
    # def perceive_rings

    # def merge_terminal_atoms(self, indices):
    #     """
    #     Primarily for merging hydrogens, but will merge the data for any atom or pseudoatom that is bonded to only one
    #     other atom.
    #
    #     Parameters
    #     ----------
    #     indices: list
    #         A list of indices to merge
    #
    #     Returns
    #     -------
    #     None
    #     """
    #     for index in indices:
    #         if len(self.get_neighbors(index)) != 1:
    #             msg = "Atempted to merge atom %d with %d neighbors. "
    #             msg += "Only atoms with one neighbor can be merged."
    #             msg = msg % (index + 1, self.get_neighbors(index))
    #             raise RuntimeError(msg)
    #         neighbor_index = self.get_neighbors(index)[0]
    #         self.atoms[neighbor_index].charge += self.get_charge(index)
    #         self.atoms[index].charge = 0.0
    #         self.atoms[index].is_ignore = True
    #     return


# TODO: RENAME THIS TO NOT BE WORD VOMIT, CLEAN UP - Sets all the requirements for if you're building
# A moleculesetup with an external toolkit like RDKit or OB
class MoleculeSetupExternalToolBuild(ABC):

    @staticmethod
    def are_fourier_series_identical(fs1, fs2):
        """
        NOT MODIFIED YET
        """
        index_by_periodicity1 = {
            fs1[index]["periodicity"]: index for index in range(len(fs1))
        }
        index_by_periodicity2 = {
            fs2[index]["periodicity"]: index for index in range(len(fs2))
        }
        if index_by_periodicity1 != index_by_periodicity2:
            return False
        for periodicity in index_by_periodicity1:
            index1 = index_by_periodicity1[periodicity]
            index2 = index_by_periodicity2[periodicity]
            for key in ["k", "phase", "periodicity"]:
                if fs1[index1][key] != fs2[index2][key]:
                    return False
        return True

    def add_dihedral_interaction(self, fourier_series):
        """

        Parameters
        ----------
        fourier_series

        Returns
        -------

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
    def init_atom(self):
        pass

    @abstractmethod
    def init_bond(self):
        pass

    @abstractmethod
    def get_mol_name(self):
        pass

    @abstractmethod
    def find_pattern(self):
        pass

    @abstractmethod
    def get_smiles_and_order(self):
        pass

class RDKitMoleculeSetup(MoleculeSetup, MoleculeSetupExternalToolBuild):
    """
    Subclass of MoleculeSetup, used to represent MoleculeSetup objects working with RDKit objects

    Attributes
    ----------
    mol : rdkit.Chem.rdchem.Mol
        an RDKit Mol object to base the Molecule Setup on
    modified_atom_positions :
        list of dictionaries where keys are atom indices

    Methods
    -------
    from_mol()
        constructor for the RDKitMoleculeSetup object (consider adapting to init?)
    """

    @classmethod
    def from_mol(
        cls,
        mol,
        keep_chorded_rings=False,
        keep_equivalent_rings=False,
        assign_charges=True,
        conformer_id=-1,
    ):
        if cls.has_implicit_hydrogens(mol):
            raise ValueError("RDKit molecule has implicit Hs. Need explicit Hs.")
        if mol.GetNumConformers() == 0:
            raise ValueError(
                "RDKit molecule does not have a conformer. Need 3D coordinates."
            )
        rdkit_conformer = mol.GetConformer(conformer_id)
        if not rdkit_conformer.Is3D():
            warnings.warn(
                "RDKit molecule not labeled as 3D. This warning won't show again."
            )
            RDKitMoleculeSetup.warned_not3D = True
        if mol.GetNumConformers() > 1 and conformer_id == -1:
            msg = "RDKit molecule has multiple conformers. Considering only the first one."
            print(msg, file=sys.stderr)
        molsetup = cls()
        molsetup.mol = mol
        molsetup.atom_true_count = molsetup.get_num_mol_atoms()
        molsetup.name = molsetup.get_mol_name()
        coords = rdkit_conformer.GetPositions()
        molsetup.init_atom(assign_charges, coords)
        molsetup.init_bond()
        molsetup.perceive_rings(keep_chorded_rings, keep_equivalent_rings)
        molsetup.rmsd_symmetry_indices = cls.get_symmetries_for_rmsd(mol)

        # to store sets of coordinates, e.g. docked poses, as dictionaries indexed by
        # the atom index, because not all atoms need to have new coordinates specified
        # Unspecified hydrogen positions bonded to modified heavy atom positions
        # are to be calculated "on-the-fly".
        molsetup.modified_atom_positions = (
            []
        )  # list of dictionaries where keys are atom indices

        return molsetup

    # @classmethod
    # def from_mol(
    #     cls,
    #     mol: rdkit.Chem.Mol,
    #     keep_chorded_rings: bool = False,
    #     keep_equivalent_rings: bool = False,
    #     assign_charges: bool = True,
    #     conformer_id: int = 1,
    # ):
    #     # Checks if the input molecule is valid
    #     if has_implicit_hydrogens(mol):
    #         raise ValueError("RDKit molecule has implicit Hs. Need explicit Hs.")
    #     if mol.GetNumConformers(mol) == 0:
    #         raise ValueError(
    #             "RDKit molecule does not have a conformer. Need 3D coordinates."
    #         )
    #
    #     # Gets the RDKit Conformer that we are going to load into the molecule setup
    #     rdkit_conformer = mol.GetConformer(conformer_id)
    #
    #     # Creating and populating the molecule setup with properties from RDKit
    #
    #     pass

    def get_conformer_with_modified_positions(self, new_atom_positions):
        # we operate on one conformer at a time because SetTerminalAtomPositions
        # acts on all conformers of a molecule, and we don't want to guarantee
        # that all conformers require the same set of terminal atoms to be updated
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
        if new_atom_positions_list is None:
            new_atom_positions_list = self.modified_atom_positions
        new_mol = Chem.Mol(self.mol)
        new_mol.RemoveAllConformers()
        for new_atom_positions in new_atom_positions_list:
            conformer = self.get_conformer_with_modified_positions(new_atom_positions)
            new_mol.AddConformer(conformer, assignId=True)
        return new_mol

    def get_smiles_and_order(self):
        """
        return the SMILES after Chem.RemoveHs()
        and the mapping between atom indices in smiles and self.mol
        """

        ### # support sidechains in which not all real atoms are included in PDBQT
        ### ignored = []
        ### for atom in self.mol.GetAtoms():
        ###     index = atom.GetIdx()
        ###     if self.atom_ignore[index]:
        ###         ignored.append(index)
        ### if len(ignored) > 0: # e.g. sidechain padded with amides
        ###     mol_no_ignore = Chem.EditableMol(self.mol)
        ###     for index in sorted(ignored, reverse=True):
        ###         mol_no_ignore.RemoveAtom(index)
        ###     # remove dangling Hs
        ###     dangling_hs = []
        ###     for atom in mol_no_ignore.GetMol().GetAtoms():
        ###         if atom.GetAtomicNum() == 1:
        ###             if len(atom.GetNeighbors()) == 0:
        ###                 dangling_hs.append(atom.GetIdx())
        ###     for index in sorted(dangling_hs, reverse=True):
        ###         mol_no_ignore.RemoveAtom(index)
        ###     mol_no_ignore = mol_no_ignore.GetMol()
        ###     Chem.SanitizeMol(mol_no_ignore)
        ### else:
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
        return smiles, order

    def find_pattern(self, smarts):
        p = Chem.MolFromSmarts(smarts)
        return self.mol.GetSubstructMatches(p)

    def get_mol_name(self):
        if self.mol.HasProp("_Name"):
            return self.mol.GetProp("_Name")
        else:
            return None

    def get_num_mol_atoms(self):
        return self.mol.GetNumAtoms()

    def get_equivalent_atoms(self):
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
            print(
                "warning: found the maximum nr of matches (%d) in RDKitMolSetup.get_symmetries_for_rmsd"
                % max_matches
            )
            print(
                'Maybe this molecule is "too" symmetric? %s' % molname,
                Chem.MolToSmiles(mol_noHs),
            )
        return matches

    def init_atom(self, assign_charges, coords):
        """initialize the atom table information"""
        # extract/generate charges
        if assign_charges:
            copy_mol = Chem.Mol(self.mol)
            for atom in copy_mol.GetAtoms():
                if atom.GetAtomicNum() == 34:
                    atom.SetAtomicNum(16)
            rdPartialCharges.ComputeGasteigerCharges(copy_mol)
            charges = [a.GetDoubleProp("_GasteigerCharge") for a in copy_mol.GetAtoms()]
        else:
            charges = [0.0] * self.mol.GetNumAtoms()
        # perceive chirality
        chiral_info = {}
        for data in Chem.FindMolChiralCenters(self.mol, includeUnassigned=True):
            chiral_info[data[0]] = data[1]
        # register atom
        for a in self.mol.GetAtoms():
            idx = a.GetIdx()
            chiral = False
            if idx in chiral_info:
                chiral = chiral_info[idx]
            self.add_atom(
                idx,
                coord=coords[idx],
                atomic_num=a.GetAtomicNum(),
                charge=charges[idx],
                atom_type=None,
                pdbinfo=rdkitutils.getPdbInfoNoNull(a),
                chiral=False,
                ignore=False,
            )

    def init_bond(self):
        """initialize bond information"""
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = int(b.GetBondType())
            # fix the RDKit aromatic type (FIXME)
            if bond_order == 12:  # aromatic
                bond_order = 5
            if bond_order == 1:
                rotatable = True
            else:
                rotatable = False
            self.add_bond(idx1, idx2, order=bond_order, rotatable=rotatable)

    def copy(self):
        """return a copy of the current setup"""
        newsetup = RDKitMoleculeSetup()
        for key, value in self.__dict__.items():
            if key != "mol":
                newsetup.__dict__[key] = deepcopy(value)
        newsetup.mol = Chem.Mol(self.mol)  # not sure how deep of a copy this is
        return newsetup

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


class OBMoleculeSetup(MoleculeSetup):

    def get_mol_name(self):
        return self.mol.GetTitle()

    def find_pattern(self, smarts):
        obsmarts = ob.OBSmartsPattern()
        obsmarts.Init(smarts)
        found = obsmarts.Match(self.mol)
        output = []
        if found:
            for x in obsmarts.GetUMapList():
                output.append([y - 1 for y in x])
        return output

    def get_num_mol_atoms(self):
        return self.mol.NumAtoms()

    def get_equivalent_atoms(self):
        raise NotImplementedError

    def init_atom(self, assign_charges):
        """initialize atom data table"""
        for a in ob.OBMolAtomIter(self.mol):
            partial_charge = a.GetPartialCharge() * float(assign_charges)
            self.add_atom(
                a.GetIdx() - 1,
                coord=np.asarray(obutils.getAtomCoords(a), dtype="float"),
                atomic_num=a.GetAtomicNum(),
                charge=partial_charge,
                atom_type=None,
                pdbinfo=obutils.getPdbInfoNoNull(a),
                ignore=False,
                chiral=a.IsChiral(),
            )
            # TODO check consistency for chiral model between OB and RDKit

    def init_bond(self):
        """initialize bond data table"""
        for b in ob.OBMolBondIter(self.mol):
            idx1 = b.GetBeginAtomIdx() - 1
            idx2 = b.GetEndAtomIdx() - 1
            bond_order = b.GetBondOrder()
            if b.IsAromatic():
                bond_order = 5
            self.add_bond(idx1, idx2, order=bond_order)

    def copy(self):
        """return a copy of the current setup"""
        return OBMoleculeSetup(template=self)

@dataclass
class Atom:
    index: int
    pdbinfo: str = DEFAULT_PDBINFO
    charge: float = DEFAULT_CHARGE
    atomic_num: int = DEFAULT_ATOMIC_NUM
    atom_type: str = DEFAULT_ATOM_TYPE
    is_ignore: bool = DEFAULT_IS_IGNORE
    is_chiral: bool = DEFAULT_IS_CHIRAL
    graph: list[int] = field(default_factory=list)
    interaction_vectors: list[np.array] = field(default_factory=list)

    is_dummy: bool = False
    is_pseudo_atom: bool = False


@dataclass
class Bond:
    canon_id: (int, int)
    index1: int
    index2: int
    order: int = DEFAULT_BOND_ORDER
    rotatable: bool = DEFAULT_BOND_ROTATABLE

    def __init__(
        self,
        index1: int,
        index2: int,
        order: int = DEFAULT_BOND_ORDER,
        rotatable: bool = DEFAULT_BOND_ROTATABLE,
    ):
        self.canon_id = self.get_bond_id(index1, index2)
        self.index1 = index1
        self.index2 = index2
        self.order = order
        self.rotatable = rotatable
        return

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
class Ring:
    ring_id: int
    corner_flip: bool = DEFAULT_RING_CORNER_FLIP
    graph: dict = DEFAULT_RING_GRAPH
    is_aromatic: bool = DEFAULT_RING_IS_AROMATIC


@dataclass
class Restraint:
    atom_index: int
    target_xyz: (float, float, float)
    kcal_per_angstrom_square: float
    delay_angstroms: float

    def copy(self):
        new_target_xyz = (self.target_xyz[0], self.target_xyz[1], self.target_xyz[2])
        new_restraint = Restraint(
            self.atom_index,
            new_target_xyz,
            self.kcal_per_angstrom_square,
            self.delay_angstroms,
        )
        return new_restraint


# TODO: refactor molsetup class then refactor this and consider making it more readable.
class MoleculeSetupEncoder(json.JSONEncoder):
    """
    JSON Encoder class for molecule setup objects. Makes decisions about how to convert types to JSON serializable types
    so they can be reliably decoded should a user want to pull the JSON back into an object.
    """

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for Molecule Setup objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the MoleculeSetup JSON format for MoleculeSetup objects.
            For all other objects will return the default JSON encoding.

        Returns
        -------
        A JSON serializable object that represents the MoleculeSetup class or the default JSONEncoder output for an
        object.
        """
        # Checks if the input object is a MoleculeSetup as desired
        if isinstance(obj, MoleculeSetup):
            separator_char = ","  # TODO: consider setting this somewhere else so it is the same for decode and encode
            # Sets the elements of an output dict of attributes based off of all the attributes that are guaranteed to
            # be present and
            output_dict = {
                # Attributes that are dictionaries mapping from atom index to some other property
                "coord": {
                    k: v.tolist() for k, v in obj.coord.items()
                },  # converts coords from numpy arrays to lists
                "charge": obj.charge,
                "pdbinfo": obj.pdbinfo,
                "atom_type": obj.atom_type,
                "atom_ignore": obj.atom_ignore,
                "chiral": obj.chiral,
                "graph": obj.graph,
                "element": obj.element,
                "interaction_vector": obj.interaction_vector,
                "atom_to_ring_id": obj.atom_to_ring_id,
                # Dihedral
                "dihedral_interactions": obj.dihedral_interactions,
                "dihedral_partaking_atoms": obj.dihedral_partaking_atoms,
                "dihedral_labels": obj.dihedral_labels,
                # Dictionaries with tuple keys have the tuples converted to strings
                "bond": {
                    separator_char.join([str(i) for i in k]): v
                    for k, v in obj.bond.items()
                },
                "rings": {
                    separator_char.join([str(i) for i in k]): v
                    for k, v in obj.rings.items()
                },
                # Dictionaries of dictionaries
                "atom_params": obj.atom_params,
                "flexibility_model": obj.flexibility_model,
                "ring_closure_info": obj.ring_closure_info,
                "ring_corners": obj.ring_corners,
                # Lists
                "atom_pseudo": obj.atom_pseudo,
                "restraints": [asdict(restraint) for restraint in obj.restraints],
                "rings_aromatic": obj.rings_aromatic,
                "rotamers": obj.rotamers,
                # Simple variables
                "atom_true_count": obj.atom_true_count,
                "is_sidechain": obj.is_sidechain,
                "name": obj.name,
                "rmsd_symmetry_indices": obj.rmsd_symmetry_indices,
            }
            # Since the flexibility model attribute contains dictionaries with tuples as keys, it needs to be treated
            # more specifically.
            if "rigid_body_connectivity" in obj.flexibility_model:
                new_rigid_body_conn_dict = {
                    separator_char.join([str(i) for i in k]): v
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

            # Adds mol attribute if the input MoleculeSetup is an RDKitMoleculeSetup
            if isinstance(obj, RDKitMoleculeSetup):
                output_dict["mol"] = rdMolInterchange.MolToJSON(obj.mol)
            return output_dict
        # If the input object is not a MoleculeSetup, returns the default JSON encoding for that object
        return json.JSONEncoder.default(self, obj)
