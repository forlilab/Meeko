from rdkit import Chem
from meeko import MoleculePreparation
from meeko import oids_block_from_setup
from meeko import parse_offxml
from meeko import AtomTyper
import openforcefields
import pathlib
import json

p = pathlib.Path(openforcefields.__file__) # find openff-forcefields xml files
offxml = p.parents[0] / "offxml" / "openff-2.0.0.offxml"
offxml = offxml.resolve()
vdw_list, dihedral_list, vdw_by_type = parse_offxml(offxml)
meeko_config = {"keep_nonpolar_hydrogens": True, "flexible_amides": True}
meeko_config["atom_type_smarts"] = json.loads(AtomTyper.defaults_json)
meeko_config["atom_type_smarts"]["ATOM_PARAMS"] = {"openff-2.0.0": vdw_list}
meeko_config["atom_type_smarts"]["DIHEDRALS"] = dihedral_list
mk_prep = MoleculePreparation.from_config(meeko_config)

def mol_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    etkdg_param = Chem.rdDistGeom.ETKDGv3()
    Chem.rdDistGeom.EmbedMolecule(mol, etkdg_param)
    return mol
    
def test():
    mol = mol_from_smiles("c1c(OC)ccnc1NC(=O)C")
    mk_prep.prepare(mol)
    molsetup = mk_prep.setup
    oids = oids_block_from_setup(molsetup)
    print(oids)

if __name__ == "__main__":
    test()
