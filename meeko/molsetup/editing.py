"""Chemistry-aware mutations on a populated MoleculeSetup.

These operations went beyond simple invariant-preserving primitives like
``add_atom`` / ``add_bond`` — they tweak charges, atom types, or topology
based on chemical context — so they live outside the data class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from .setup import MoleculeSetup
    from .uniq_atom_params import UniqAtomParams


def merge_terminal_atoms(
    setup: "MoleculeSetup", indices: Iterable[int], merge_rmin_half: bool = False
) -> None:
    """Fold each terminal atom's charge (and optionally rmin_half) into its
    single neighbor, then mark the terminal atom ``is_ignore``.

    Primarily used to absorb hydrogens for united-atom AD4 scoring.
    """
    if merge_rmin_half and "rmin_half" not in setup.atom_params:
        raise ValueError("can't merge rmin_half because it's not in atom_params")
    for index in indices:
        if len(setup.get_neighbors(index)) != 1:
            msg = "Atempted to merge atom %d with %d neighbors. "
            msg += "Only atoms with one neighbor can be merged."
            msg = msg % (index + 1, setup.get_neighbors(index))
            raise RuntimeError(msg)
        neighbor_index = setup.get_neighbors(index)[0]
        setup.atoms[neighbor_index].charge += setup.get_charge(index)
        setup.atoms[index].charge = 0.0
        setup.atoms[index].is_ignore = True
        if not merge_rmin_half:
            continue
        r_neigh = setup.atom_params["rmin_half"][neighbor_index]
        r_source = setup.atom_params["rmin_half"][index]
        new_r = np.cbrt(r_neigh**3 + r_source**3)
        setup.atom_params["rmin_half"][neighbor_index] = new_r
        setup.atom_params["rmin_half"][index] = 0.0


def clean_atoms(setup: "MoleculeSetup", remove_pseudoatoms: bool = False) -> int:
    """Drop dummy (and optionally pseudo) atoms and re-index the rest.

    Returns the number of atoms removed.
    """
    new_atoms = []
    removed_atom_count = 0
    for atom in setup.atoms:
        if remove_pseudoatoms and atom.is_pseudo_atom:
            removed_atom_count += 1
            continue
        if atom.is_dummy:
            removed_atom_count += 1
            continue
        atom.index = atom.index - removed_atom_count
        new_atoms.append(atom)
    setup.atoms = new_atoms
    if remove_pseudoatoms:
        setup.pseudoatom_count = 0
    return removed_atom_count


def set_atom_type_from_uniq_atom_params(
    setup: "MoleculeSetup", uniq_atom_params: "UniqAtomParams", prefix: str
) -> None:
    """Rewrite each atom's ``atom_type`` to ``f"{prefix}{j}"`` where ``j`` is
    the row index of the atom's params inside ``uniq_atom_params``.
    """
    parameter_indices = uniq_atom_params.get_indices_from_atom_params(
        setup.atom_params
    )
    if len(parameter_indices) != len(setup.atoms):
        raise RuntimeError(
            "Number of parameters ({len(parameter_indices)}) not equal to number of atoms in Molecule Setup ({len(setup.atom_type)})"
        )
    for i, j in enumerate(parameter_indices):
        # Preserves the original (latent) bug: writes to ``setup.atom_type``,
        # not ``setup.atoms[i].atom_type``. setup.atom_type doesn't exist; the
        # caller path triggering this is currently dead code. Keeping behavior
        # identical to pre-refactor so any future fix is a separate change.
        setup.atom_type[i] = f"{prefix}{j}"
