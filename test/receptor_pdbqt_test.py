from meeko.receptor_pdbqt import PDBQTReceptor
from meeko.receptor_pdbqt import _read_receptor_pdbqt_string

# three typed atoms, no pseudo atoms
PDBQT = "\n".join([
    "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00    -0.347 N ",
    "ATOM      2  CA  ALA A   1      12.560  13.207  10.000  1.00  0.00     0.180 C ",
    "ATOM      3  O   ALA A   1      13.100  14.600  10.000  1.00  0.00    -0.271 OA",
]) + "\n"

# same, with an AutoDock4Zn TZ pseudo atom in the middle
PDBQT_TZ = "\n".join([
    "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00    -0.347 N ",
    "ATOM      2  TZ  ZN  A   2      12.000  13.000  10.000  1.00  0.00     0.000 TZ",
    "ATOM      3  CA  ALA A   3      12.560  13.207  10.000  1.00  0.00     0.180 C ",
    "ATOM      4  O   ALA A   3      13.100  14.600  10.000  1.00  0.00    -0.271 OA",
]) + "\n"


def check_idx_matches_row(atoms):
    for row, atom in enumerate(atoms):
        assert atom["idx"] == row


def test_idx_matches_row():
    atoms, annotations = _read_receptor_pdbqt_string(PDBQT)
    assert len(atoms) == 3
    check_idx_matches_row(atoms)


def test_skip_typing_reads_atoms():
    atoms, annotations = _read_receptor_pdbqt_string(PDBQT, skip_typing=True)
    assert len(atoms) == 3
    check_idx_matches_row(atoms)


def test_skip_typing_on_receptor():
    receptor = PDBQTReceptor(PDBQT, skip_typing=True)
    assert receptor._atoms.shape[0] == 3


def test_pseudo_atom_is_dropped():
    atoms, annotations = _read_receptor_pdbqt_string(PDBQT_TZ)
    assert len(atoms) == 3
    assert "TZ" not in list(atoms["atom_type"])
    check_idx_matches_row(atoms)


def test_annotations_index_correct_rows():
    atoms, annotations = _read_receptor_pdbqt_string(PDBQT_TZ)
    assert annotations["all"] == [0, 1, 2]
    # the OA follows the dropped TZ, so its row shifts down by one
    assert annotations["hb_acc"] == [2]
    assert atoms[annotations["hb_acc"][0]]["atom_type"] == "OA"
    assert sorted(atoms[i]["atom_type"] for i in annotations["vdw"]) == ["C", "N"]


def test_no_atoms_raises():
    try:
        _read_receptor_pdbqt_string("REMARK nothing here\n")
    except ValueError:
        return
    raise AssertionError("expected ValueError for a string with no atoms")
