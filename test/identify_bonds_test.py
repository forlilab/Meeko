import numpy as np

from meeko.molecule_pdbqt import _identify_bonds as identify_bonds_ligand
from meeko.receptor_pdbqt import _identify_bonds as identify_bonds_receptor
from meeko.utils.covalent_radius_table import covalent_radius
from meeko.utils.autodock4_atom_types_elements import autodock4_atom_types_elements

implementations = (identify_bonds_ligand, identify_bonds_receptor)


def run_both(atom_idx, positions, atom_types):
    """both modules carry the same function, so check them together"""
    results = []
    for identify_bonds in implementations:
        bonds = identify_bonds(atom_idx, np.array(positions, dtype=np.float32),
                               np.array(atom_types))
        results.append({int(k): sorted(int(i) for i in v) for k, v in bonds.items()})
    assert results[0] == results[1]
    return results[0]


def test_two_bonded_carbons():
    # 1.5 A apart, well inside 1.1 * (0.76 + 0.76)
    bonds = run_both([0, 1], [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], ["C", "C"])
    assert bonds == {0: [1], 1: [0]}


def test_two_distant_carbons():
    # 2.5 A apart, outside the covalent cutoff
    bonds = run_both([0, 1], [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], ["C", "C"])
    assert bonds == {0: [], 1: []}


def test_methane_like():
    # central carbon with four hydrogens at 1.09 A
    d = 1.09 / np.sqrt(3.0)
    positions = [[0.0, 0.0, 0.0], [d, d, d], [-d, -d, d], [d, -d, -d], [-d, d, -d]]
    bonds = run_both([0, 1, 2, 3, 4], positions, ["C", "HD", "HD", "HD", "HD"])
    assert bonds[0] == [1, 2, 3, 4]
    for h in (1, 2, 3, 4):
        assert bonds[h] == [0]


def reference_bonds(atom_idx, positions, atom_types):
    """brute force version of the same rule, for comparison"""
    positions = np.asarray(positions, dtype=np.float64)
    n = len(atom_idx)
    k = 5 if n > 5 else n
    radii = [covalent_radius[autodock4_atom_types_elements[t]] for t in atom_types]
    bonds = {}
    for i in range(n):
        d = np.linalg.norm(positions - positions[i], axis=1)
        nearest = np.argsort(d, kind="stable")[1:k]
        bonds[int(atom_idx[i])] = sorted(
            int(atom_idx[j]) for j in nearest
            if d[j] < 1.1 * (radii[i] + radii[j])
        )
    return bonds


def test_matches_brute_force_reference():
    rng = np.random.RandomState(0)
    for trial in range(20):
        n = int(rng.randint(2, 120))
        positions = (rng.rand(n, 3) * rng.uniform(3.0, 14.0)).astype(np.float32)
        atom_types = np.array([["C", "N", "OA", "HD", "SA"][i]
                               for i in rng.randint(0, 5, n)])
        got = run_both(list(range(n)), positions, atom_types)
        want = reference_bonds(list(range(n)), positions, atom_types)
        assert got == want


def test_non_contiguous_indices():
    # annotation lists are subsets, so keys must be the given indices
    positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [10.0, 0.0, 0.0]]
    bonds = run_both([4, 7, 9], positions, ["C", "C", "C"])
    assert bonds == {4: [7], 7: [4], 9: []}


def test_single_atom():
    bonds = run_both([0], [[0.0, 0.0, 0.0]], ["C"])
    assert bonds == {}


def test_no_atoms():
    bonds = run_both([], np.empty((0, 3)), np.array([], dtype="U2"))
    assert bonds == {}
