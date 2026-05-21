"""MoleculeSetup and RDKitMoleculeSetup core classes.

Data classes (Atom, Bond, Ring, RingClosureInfo, Restraint, UniqAtomParams) live
in sibling modules. This module contains the orchestration classes that hold
them together and the RDKit-coupled subclass.
"""

from copy import deepcopy
import logging
from typing import Any, Optional, Union

import numpy as np
from rdkit import Chem

from ..utils.jsonutils import BaseJSONParsable
from ..utils.pdbutils import PDBAtomInfo
from ..utils import utils

from .atom import (
    Atom,
    DEFAULT_ATOM_TYPE,
    DEFAULT_ATOMIC_NUM,
    DEFAULT_CHARGE,
    DEFAULT_COORD,
    DEFAULT_GRAPH,
    DEFAULT_IS_IGNORE,
    DEFAULT_PDBINFO,
)
from .bond import Bond, DEFAULT_BOND_ROTATABLE
from .flex_model import FlexibilityModel
from .ring import Ring, RingClosureInfo
from .restraint import Restraint
from .uniq_atom_params import UniqAtomParams

eol = "\n"

logger = logging.getLogger(__name__)


class MoleculeSetup(BaseJSONParsable):
    """Container for molecule data used downstream of `MoleculePreparation`.

    Attributes
    ----------
    name: str
    pseudoatom_count: int
    atoms: list[Atom]
    bond_info: dict[tuple, Bond]
    rings: dict[tuple, Ring]
    ring_closure_info: RingClosureInfo
    rotamers: list[dict]
    atom_params: dict
    restraints: list[Restraint]
    flexibility_model: FlexibilityModel
    """

    PSEUDOATOM_ATOMIC_NUM = 0

    def __init__(self, name: str = None):
        self.name: str = name
        self.pseudoatom_count: int = 0
        self.atoms: list[Atom] = []
        self.bond_info: dict[tuple, Bond] = {}
        self.rings: dict[tuple, Ring] = {}
        self.ring_closure_info = RingClosureInfo([], {})
        self.rotamers: list[dict] = []
        self.atom_params: dict = {}
        self.restraints: list = []
        self.flexibility_model = FlexibilityModel()

    @classmethod
    def json_encoder(cls, obj: "MoleculeSetup") -> Optional[dict[str, Any]]:
        from . import io
        return io.encode_molecule_setup(obj)

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
        from . import io
        return io.decode_molecule_setup(cls, obj)

    # ----- invariant-preserving primitives -----

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
        insert_disallowed = len(self.atoms) > atom_index and not overwrite
        if (
            atom_index is not None
            and insert_disallowed
            and not self.atoms[atom_index].is_dummy
        ):
            raise RuntimeError(
                "ADD_ATOM Error: the atom_index [%d] is already occupied (use 'overwrite' to force)"
            )
        if atom_index is None:
            atom_index = len(self.atoms)
        while atom_index > len(self.atoms):
            self.atoms.append(Atom(len(self.atoms), is_dummy=True))
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
        pseudoatom_index = len(self.atoms)
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
        if anchor_list is not None:
            for anchor in anchor_list:
                self.add_bond(pseudoatom_index, anchor, rotatable=rotatable)
        if not self.flexibility_model or not anchor_list:
            return pseudoatom_index
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
        return pseudoatom_index

    def delete_atom(self, atom_index: int):
        self.atoms[atom_index] = Atom(atom_index, is_dummy=True)

    def add_bond(
        self,
        atom_index_1: int,
        atom_index_2: int,
        rotatable: bool = DEFAULT_BOND_ROTATABLE,
    ) -> None:
        if len(self.atoms) <= atom_index_1 or len(self.atoms) <= atom_index_2:
            raise IndexError(
                "ADD_BOND: provided atom indices outside the range of atoms currently in MoleculeSetup"
            )
        if atom_index_2 not in self.atoms[atom_index_1].graph:
            self.atoms[atom_index_1].graph.append(atom_index_2)
        if atom_index_1 not in self.atoms[atom_index_2].graph:
            self.atoms[atom_index_2].graph.append(atom_index_1)
        new_bond = Bond(atom_index_1, atom_index_2, rotatable)
        self.bond_info[new_bond.canon_id] = new_bond

    def delete_bond(self, atom_index_1: int, atom_index_2: int):
        canon_bond_id = Bond.get_bond_id(atom_index_1, atom_index_2)
        del self.bond_info[canon_bond_id]
        self.atoms[atom_index_1].graph.remove(atom_index_2)
        self.atoms[atom_index_2].graph.remove(atom_index_1)

    def add_rotamers(
        self, index_list: list[tuple[int, int, int, int]], angle_list: np.ndarray
    ):
        rotamers = {}
        for (idx1, idx2, idx3, idx4), angle in zip(index_list, angle_list):
            bond_id = Bond.get_bond_id(idx2, idx3)
            if bond_id in rotamers:
                raise RuntimeError("repeated bond %d-$d" % bond_id)
            if not self.bond_info[bond_id].rotatable:
                raise RuntimeError(
                    "trying to add rotamer for non rotatable bond %d-%d" % bond_id
                )
            dihedral = 0  # TODO: fix this
            rotamers[bond_id] = angle - dihedral
        self.rotamers.append(rotamers)

    def delete_rotamers(
        self,
        bond_id_list: list[tuple] = None,
        index_list: list[tuple[int, int, int, int]] = None,
    ):
        if index_list is not None:
            for idx1, idx2, idx3, idx4 in index_list:
                bond_id = Bond.get_bond_id(idx2, idx3)
                bond_id_list.append(bond_id)
        if bond_id_list is not None:
            for bond_id in bond_id_list:
                if bond_id in self.rotamers:
                    del self.rotamers[bond_id]

    @property
    def true_atom_count(self):
        count = 0
        for atom in self.atoms:
            if not atom.is_pseudo_atom and not atom.is_dummy:
                count += 1
        return count

    def clean_atoms(self, remove_pseudoatoms: bool = False):
        from . import editing
        return editing.clean_atoms(self, remove_pseudoatoms)

    # ----- accessors (bounds-checked) -----

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

    def get_coord(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_CHARGE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].coord

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

    def set_atom_type(self, atom_index: int, atom_type: str) -> None:
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "SET_ATOM_TYPE: provided atom index is out of range or is a dummy atom"
            )
        self.atoms[atom_index].atom_type = atom_type

    def set_atom_type_from_uniq_atom_params(
        self, uniq_atom_params: UniqAtomParams, prefix: str
    ):
        from . import editing
        return editing.set_atom_type_from_uniq_atom_params(self, uniq_atom_params, prefix)

    def get_is_ignore(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_IS_IGNORE: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].is_ignore

    def get_neighbors(self, atom_index: int):
        if atom_index > len(self.atoms) or self.atoms[atom_index].is_dummy:
            raise IndexError(
                "GET_GRAPH: provided atom index is out of range or is a dummy atom"
            )
        return self.atoms[atom_index].graph

    # ----- chemistry-aware edits & graph algorithms (delegate to editing.py / graph.py) -----

    def merge_terminal_atoms(self, indices, merge_rmin_half: bool = False) -> None:
        from . import editing
        return editing.merge_terminal_atoms(self, indices, merge_rmin_half)

    @staticmethod
    def get_bonds_in_ring(ring: tuple) -> list[tuple]:
        from . import graph
        return graph.get_bonds_in_ring(ring)

    def _recursive_graph_walk(
        self, idx: int, collected: list[int] = None, exclude: list[int] = None
    ):
        from . import graph
        return graph.recursive_graph_walk(self, idx, collected, exclude)

    def write_coord_string(self) -> str:
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
        keys_to_not_show = ["type"]
        for bond_id, bond in list(self.bond_info.items()):
            t = ", ".join(
                "%s: %s" % (i, j)
                for i, j in bond.__dict__.items()
                if i not in keys_to_not_show
            )
            print("% 8s - " % str(bond_id), t)
        print("")


class RDKitMoleculeSetup(MoleculeSetup, BaseJSONParsable):
    """MoleculeSetup paired with an RDKit Chem.Mol.

    Extra attributes
    ----------------
    mol : rdkit.Chem.rdchem.Mol
    modified_atom_positions: list
    dihedral_interactions: list[dict]
    dihedral_partaking_atoms: dict
    dihedral_labels: dict
    atom_to_ring_id: dict
    rmsd_symmetry_indices: tuple
    """

    def __init__(self, name: str = None, source: "MoleculeSetup" = None):
        super().__init__(name)
        if source:
            if isinstance(source, MoleculeSetup):
                for key, value in source.__dict__.items():
                    setattr(self, key, deepcopy(value))
            else:
                raise TypeError(
                    "Expected 'source' to be an instance of MoleculeSetup, got type: {}".format(
                        type(source)
                    )
                )
        self.mol = None
        self.modified_atom_positions = []
        self.dihedral_interactions: list[dict] = []
        self.dihedral_partaking_atoms: dict = {}
        self.dihedral_labels: dict = {}
        self.atom_to_ring_id = {}
        self.rmsd_symmetry_indices = ()
        self.compute_charges = False

    @classmethod
    def json_encoder(cls, obj: "RDKitMoleculeSetup") -> Optional[dict[str, Any]]:
        from . import io
        return io.encode_rdkit_molecule_setup(obj)

    expected_json_keys = frozenset(
        MoleculeSetup.expected_json_keys.union(
            {
                "mol",
                "modified_atom_positions",
                "dihedral_interactions",
                "dihedral_partaking_atoms",
                "dihedral_labels",
                "atom_to_ring_id",
                "rmsd_symmetry_indices",
            }
        )
    )

    @classmethod
    def _decode_object(cls, obj: dict[str, Any]):
        from . import io
        return io.decode_rdkit_molecule_setup(cls, obj)

    def copy(self):
        newsetup = RDKitMoleculeSetup()
        for key, value in self.__dict__.items():
            if key == "mol":
                newsetup.mol = Chem.Mol(self.mol) if self.mol else None
            else:
                setattr(newsetup, key, deepcopy(value))
        return newsetup

    # ----- RDKit-coupled methods: thin wrappers that delegate to rdkit_adapter -----

    @classmethod
    def from_mol(cls, mol, **kwargs):
        from . import rdkit_adapter
        return rdkit_adapter.from_rdkit_mol(cls, mol, **kwargs)

    def init_atom(self, *args, **kwargs):
        from . import rdkit_adapter
        return rdkit_adapter.init_atom(self, *args, **kwargs)

    def init_bond(self):
        from . import rdkit_adapter
        return rdkit_adapter.init_bond(self)

    def calculate_charges(self, charge_model, read_charges_from_prop):
        from . import rdkit_adapter
        return rdkit_adapter.calculate_charges(self, charge_model, read_charges_from_prop)

    def get_charges_from_template(self, charge_model, template_charge):
        from . import rdkit_adapter
        return rdkit_adapter.get_charges_from_template(self, charge_model, template_charge)

    def find_pattern(self, smarts, uniquify=False, max_matches=int(1e7)):
        from . import rdkit_adapter
        return rdkit_adapter.find_pattern(self, smarts, uniquify=uniquify, max_matches=max_matches)

    def get_mol_name(self):
        from . import rdkit_adapter
        return rdkit_adapter.get_mol_name(self)

    def get_smiles_and_order(self):
        from . import rdkit_adapter
        return rdkit_adapter.get_smiles_and_order(self)

    def perceive_rings(self, keep_chorded_rings: bool, keep_equivalent_rings: bool):
        from . import rdkit_adapter
        return rdkit_adapter.perceive_rings(self, keep_chorded_rings, keep_equivalent_rings)

    def get_conformer_with_modified_positions(self, new_atom_positions):
        from . import rdkit_adapter
        return rdkit_adapter.get_conformer_with_modified_positions(self, new_atom_positions)

    def get_mol_with_modified_positions(self, new_atom_positions_list=None):
        from . import rdkit_adapter
        return rdkit_adapter.get_mol_with_modified_positions(self, new_atom_positions_list)

    def get_num_mol_atoms(self):
        from . import rdkit_adapter
        return rdkit_adapter.get_num_mol_atoms(self)

    def get_equivalent_atoms(self):
        from . import rdkit_adapter
        return rdkit_adapter.get_equivalent_atoms(self)

    def restrain_to(self, target_mol, kcal_per_angstrom_square=1.0, delay_angstroms=2.0):
        from . import rdkit_adapter
        return rdkit_adapter.restrain_to(
            self, target_mol, kcal_per_angstrom_square, delay_angstroms
        )

    def add_dihedral_interaction(self, fourier_series):
        from . import rdkit_adapter
        return rdkit_adapter.add_dihedral_interaction(self, fourier_series)

    @staticmethod
    def are_fourier_series_identical(series1, series2):
        from . import rdkit_adapter
        return rdkit_adapter.are_fourier_series_identical(series1, series2)

    @staticmethod
    def get_symmetries_for_rmsd(mol, max_matches=17):
        from . import rdkit_adapter
        return rdkit_adapter.get_symmetries_for_rmsd(mol, max_matches)

    @staticmethod
    def has_implicit_hydrogens(mol):
        from . import rdkit_adapter
        return rdkit_adapter.has_implicit_hydrogens(mol)
