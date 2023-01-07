from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import warnings
import pytest


def test():
    p = Chem.SmilesParserParams()
    mol = Chem.MolFromSmiles("Oc1ccc(cc1)[N+](=O)[O-]\tnitrophenol", p)
    etkdg_params = Chem.rdDistGeom.ETKDGv3()
    Chem.rdDistGeom.EmbedMolecule(mol, etkdg_params)
    mk_prep = MoleculePreparation()
    with warnings.catch_warnings(record=True) as cought:
        mk_prep.prepare(mol)
        for w in cought:
            print(w)
    assert(mk_prep.is_ok == False)

mk_prep = MoleculePreparation()

def test_no_conformer():
    mol = Chem.MolFromSmiles("C1CCCOC1")
    mol = Chem.AddHs(mol)
    with pytest.raises(ValueError) as e:
        mk_prep.prepare(mol)
    assert(str(e.value).endswith("Need 3D coordinates."))

def test_multi_conf():
    mol = Chem.MolFromSmiles("C1CNC(=O)OC1")
    mol = Chem.AddHs(mol)
    nr_conf = 2
    param = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMultipleConfs(mol, nr_conf, param)
    with pytest.warns(UserWarning, match=r"RDKit molecule has multiple conformers"):
        mk_prep.prepare(mol)
    

