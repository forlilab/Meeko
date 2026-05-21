"""meeko.polymer package.

Re-exports the names that lived in the former single-file ``meeko/polymer.py``
so existing imports keep working::

    from meeko.polymer import (
        Polymer, Monomer, ResiduePadder, ResidueTemplate,
        ResidueChemTemplates, PolymerCreationError,
        add_rotamers_to_polymer_molsetups,
    )

Module contents have been split into focused submodules.
"""

from .errors import PolymerCreationError
from .monomer import Monomer
from .padder import (
    NoAtomMapWarning,
    ResiduePadder,
    apply_atom_mappings,
    get_molAtomMapNumbers,
    remove_atoms_with_mapping,
    remove_unmapped_atoms_from_mol,
)
from .polymer import Polymer
from .rotamers import add_rotamers_to_polymer_molsetups, residues_rotamers
from .templates import ResidueChemTemplates, ResidueTemplate
from .utils import (
    _delete_residues,
    _snap_to_int,
    divide_int_gracefully,
    find_graph_paths,
    find_inter_mols_bonds,
    find_inter_mols_bonds_kdtree,
    find_inter_mols_bonds_kdtree_fast,
    find_inter_mols_bonds_old,
    get_updated_positions,
    handle_parsing_situations,
    mapping_by_mcs,
    rectify_charges,
    update_H_positions,
)

__all__ = [
    # data / orchestration classes
    "Polymer",
    "Monomer",
    "ResiduePadder",
    "ResidueTemplate",
    "ResidueChemTemplates",
    "NoAtomMapWarning",
    # errors
    "PolymerCreationError",
    # rotamers
    "residues_rotamers",
    "add_rotamers_to_polymer_molsetups",
    # utilities
    "find_graph_paths",
    "find_inter_mols_bonds",
    "find_inter_mols_bonds_old",
    "find_inter_mols_bonds_kdtree",
    "find_inter_mols_bonds_kdtree_fast",
    "mapping_by_mcs",
    "rectify_charges",
    "get_updated_positions",
    "update_H_positions",
    "divide_int_gracefully",
    "handle_parsing_situations",
    # atom-map helpers
    "get_molAtomMapNumbers",
    "remove_unmapped_atoms_from_mol",
    "apply_atom_mappings",
    "remove_atoms_with_mapping",
]
