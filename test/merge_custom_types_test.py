from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import rdDistGeom
import warnings
import pytest


def test1():
    typer = {
        "ATOM_PARAMS": {
            "reduced set": [
                {"smarts": "[#1]", "atype": "H",},
                {"smarts": "[#1][#6]([#7,#8])[#7,#8]", "atype": "HD"}, # one H in oxazole
                {"smarts": "[C]", "atype": "C"},
                {"smarts": "[c]", "atype": "A"},
                {"smarts": "[#7]", "atype": "NA"},
                {"smarts": "[#8]", "atype": "OA"},
            ]
        }
    }
    
    preparator = MoleculePreparation(atom_type_smarts=typer)
    
    mol = Chem.MolFromSmiles("c1ncco1") # oxazole
    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMolecule(mol)
    setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    
    count_HD = 0
    for line in pdbqt_string.split("\n"):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            count_HD += int(line[77:79] == "HD")

    assert(count_HD == 1)

def test2():
    typer = {
        "ATOM_PARAMS": {
            "reduced set": [
                {"smarts": "[#1]", "atype": "H_MERGE",},
                {"smarts": "[#1][#6]([#7,#8])[#7,#8]", "atype": "HC"}, # one H in oxazole
                {"smarts": "[#1][#8]", "atype": "HD"},
                {"smarts": "[C]", "atype": "C"},
                {"smarts": "[c]", "atype": "A"},
                {"smarts": "[#7]", "atype": "NA"},
                {"smarts": "[#8]", "atype": "OA"},
            ]
        }
    }
    
    preparator = MoleculePreparation(
        atom_type_smarts=typer,
        merge_these_atom_types=("H", "H_MERGE"),
    )
    
    mol = Chem.MolFromSmiles("c1nc(CO)co1") # oxazole with an -OH
    mol = Chem.AddHs(mol)
    rdDistGeom.EmbedMolecule(mol)
    setups = preparator.prepare(mol)
    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(setups[0])
    
    count_atoms = 0
    count_HD = 0
    count_HC = 0
    for line in pdbqt_string.split("\n"):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            count_atoms += 1
            count_HD += int(line[77:79] == "HD")
            count_HC += int(line[77:79] == "HC")

    assert(count_HD == 1)
    assert(count_HC == 1)
    assert(count_atoms == 9) # 5 (oxazole ring) + 1 (HC type) + 3 (COH) = 9
