"""Pure graph algorithms over the atom/bond topology of a MoleculeSetup.

These functions do not need any toolkit (RDKit, OpenBabel) — they walk
the integer-index graph stored on ``Atom.graph`` / ``MoleculeSetup.bond_info``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from .bond import Bond

if TYPE_CHECKING:
    from .setup import MoleculeSetup


def get_bonds_in_ring(ring: tuple) -> list[tuple]:
    """Canonical bond ids walking around ``ring`` (a tuple of atom indices)."""
    bonds: list[tuple] = []
    num_indices = len(ring)
    for i in range(num_indices):
        bonds.append(Bond.get_bond_id(ring[i], ring[(i + 1) % num_indices]))
    return bonds


def recursive_graph_walk(
    setup: "MoleculeSetup",
    idx: int,
    collected: list[int] | None = None,
    exclude: Iterable[int] | None = None,
) -> list[int]:
    """Collect all atoms reachable from ``idx`` via bond hops, excluding any
    atoms in ``exclude``. Returns the accumulated list (also mutated in place).
    """
    if collected is None:
        collected = []
    if exclude is None:
        exclude = []
    for neighbor in setup.get_neighbors(idx):
        if neighbor in collected or neighbor in exclude:
            continue
        collected.append(neighbor)
        recursive_graph_walk(setup, neighbor, collected, exclude)
    return collected
