from meeko import PDBQTMolecule
import numpy as np
import pathlib

workdir = pathlib.Path(__file__)
datadir = workdir.parents[0] / "rdkitmol_from_docking_data"

fpath = datadir / "vina-result-ethanol.pdbqt"


def test_atoms_all():
    pdbqtmol = PDBQTMolecule.from_file(fpath)
    assert len(pdbqtmol.atoms()) == pdbqtmol._atoms.shape[0]


def test_atoms_scalar_index():
    pdbqtmol = PDBQTMolecule.from_file(fpath)
    atoms = pdbqtmol.atoms(0)
    assert len(atoms) == 1
    assert atoms[0]["idx"] == 0


def test_atoms_numpy_scalar_index():
    pdbqtmol = PDBQTMolecule.from_file(fpath)
    atoms = pdbqtmol.atoms(np.int64(1))
    assert len(atoms) == 1
    assert atoms[0]["idx"] == 1


def test_atoms_sequence_index():
    pdbqtmol = PDBQTMolecule.from_file(fpath)
    for idx in ([0, 2], (0, 2), np.array([0, 2])):
        atoms = pdbqtmol.atoms(idx)
        assert [a["idx"] for a in atoms] == [0, 2]


def test_positions_scalar_index():
    pdbqtmol = PDBQTMolecule.from_file(fpath)
    xyz = pdbqtmol.positions(0)
    assert xyz.shape == (1, 3)
