from meeko import MoleculePreparation
from rdkit import Chem
import warnings

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
