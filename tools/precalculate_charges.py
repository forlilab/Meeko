# tool to precalculate charges based on template

import json 

json_file = "meeko/data/residue_chem_templates.json"

try:
    # Open the file in read mode ('r') using a context manager
    with open(json_file, 'r') as file:
        # Deserialize the JSON data into a Python dictionary
        templates = json.load(file)
except FileNotFoundError:
    print("Error: The file 'data.json' was not found.") #
except json.JSONDecodeError as e:
    print(f"Error: Failed to decode JSON from the file: {e}") #

from rdkit import Chem

residue_templates = templates["residue_templates"]
padders = templates['padders']

template_names = list(residue_templates.keys())
template_smiles = [v['smiles'] for k,v in residue_templates.items()]
template_mols = [Chem.MolFromSmiles(s) for s in template_smiles]
template_links = [v['link_labels'] for k, v in residue_templates.items()]

# Run reactions

from rdkit.Chem import rdChemReactions
from rdkit import Chem

# 3. Configure parser to preserve hydrogens
ps = Chem.SmilesParserParams()
ps.removeHs = False

for temp, val in residue_templates.items():
    print(temp)
    mol = Chem.MolFromSmiles(val['smiles'], ps)
    print(Chem.MolToMolBlock(mol))
    for k, v in val['link_labels'].items():
        print(temp, k, v)
        rxn_smarts = padders[v]['rxn_smarts']
        rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
        result = rxn.RunReactants((mol,))
        
        if len(result) > 0: 
            print(Chem.MolToMolBlock(result[0][0]))
            mol = result[0][0]
            Chem.SanitizeMol(mol)

    mol = Chem.AddHs(mol)
    residue_templates[temp]["padded_mol"] = mol
    residue_templates[temp]["molblock"] = Chem.MolToMolBlock(mol)
            

# Calculate gasteiger charges
from rdkit.Chem import AllChem

def calculate_gasteiger_charges(mol):
    """
    Calculates Gasteiger partial charges for a molecule from a SMILES string
    and returns them as a list of (atom_symbol, charge) tuples.
    """
    # 1. Create a molecule object from the SMILES string
    if mol is None:
        print(f"Error: No Mol")
        return None
    
    # Ensure explicit hydrogens are added, as Gasteiger calculation assumes this
    mol = Chem.AddHs(mol)
    
    # 2. Compute Gasteiger charges
    AllChem.ComputeGasteigerCharges(mol)
    
    # 3. Extract and store the charges in a list
    charges_list = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        # Charges are stored as a double property in the atom object
        try:
            charge = atom.GetDoubleProp('_GasteigerCharge')
            charges_list.append(charge)
        except KeyError:
            # Handle cases where a charge might not be assigned (e.g. for certain elements/issues)
            charges_list.append("NaN/Error")
            
    return charges_list

for t, val in residue_templates.items():
    charges = calculate_gasteiger_charges(val["padded_mol"])
    residue_templates[t]["gasteiger_charges"] = charges

# calculate espaloma charges
import espaloma as esp
from openff.toolkit.topology import Molecule

def espaloma_charges(mol):
    espaloma_model = esp.get_model("latest")
    openffmol = Molecule.from_rdkit(mol, hydrogens_are_explicit=True, allow_undefined_stereo=True)
    molgraph = esp.Graph(openffmol)
    espaloma_model(molgraph.heterograph)
    charges = [float(q) for q in molgraph.nodes["n1"].data["q"]]

    return charges

print("calculating espaloma charges....")
for t, val in residue_templates.items():
    print("calculating espaloma charge for ", t)
    try:
        charges = espaloma_charges(val["padded_mol"])
    except:
        charges = []

    residue_templates[t]["espaloma_charges"] = charges



# calculate nagl charges
from openff.toolkit import Molecule
def calculate_nagl_charge(mol):
    mol_off = Molecule.from_rdkit(mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)

    try:
        mol_off.assign_partial_charges(
            partial_charge_method="openff-gnn-am1bcc-1.0.0.pt"
            )

        charges = mol_off.partial_charges.magnitude.tolist()
    except Exception as e:
        print("NAGL charge computation failed with with exception:")
        print(e)
        print("Make sure you've installed the latest version of openff")
        charges = None

    return charges

print(" ")
print("calculating nagl charges....")
for t, val in residue_templates.items():
    print("calculating nagl charge for ", t)
    charges = calculate_nagl_charge(val["padded_mol"])
    residue_templates[t]["nagl_charges"] = charges


# save to file
padded_smiles = {}
for t, val in residue_templates.items():
    padded_smiles[t] = {}
    padded_smiles[t]["padded_smiles"] = Chem.MolToSmiles(val["padded_mol"])
    padded_smiles[t]["molblock"] = val["molblock"]
    padded_smiles[t]["gasteiger_charges"] = val["gasteiger_charges"]
    padded_smiles[t]["espaloma_charges"] = val["espaloma_charges"]
    padded_smiles[t]["nagl_charges"] = val["nagl_charges"]

with open("residue_padded_smiles.json", "w") as json_file:
    json.dump(padded_smiles, json_file, indent=4)