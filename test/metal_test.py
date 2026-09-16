from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem

import pathlib
import pytest

import meeko

# Test Data
pkgdir = pathlib.Path(meeko.__file__).parents[1]
heme_dative_sdf = pkgdir / "test/metal_complex_data/heme_dative.sdf"
heme_dative_pdbqt = pkgdir / "test/metal_complex_data/heme_dative.pdbqt"

# region Fixtures
# endregion

# region Helper Functions
def molsetups_from_sdf(sdf_fn):
    mk_prep = MoleculePreparation()
    mol = Chem.MolFromMolFile(str(sdf_fn), removeHs = False)
    molsetups = mk_prep.prepare(mol)
    return molsetups

def molsetup_to_pdbqt_string(molsetup): 
    write_pdbqt = PDBQTWriterLegacy.write_string
    pdbqt_string, is_ok, error_msg = write_pdbqt(molsetup)
    return pdbqt_string

def check_equal(string_from_test, ref_fn): 
    with open(ref_fn, "r") as f:
        string_expected = f.read()
    assert string_from_test == string_expected

# region Test Cases
def test_heme_dative(): 
    molsetup = molsetups_from_sdf(heme_dative_sdf)[0]
    pdbqt_string = molsetup_to_pdbqt_string(molsetup)
    check_equal(pdbqt_string, heme_dative_pdbqt)
# endregion
