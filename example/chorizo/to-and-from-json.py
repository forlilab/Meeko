#!/usr/bin/env python

import json

from meeko import LinkedRDKitChorizo
from meeko import ChorizoResidue
from meeko import MoleculePreparation
from meeko import RDKitMoleculeSetup
from meeko.molsetup import MoleculeSetupEncoder

with open("AHHY.pdb") as f:
    pdb_str = f.read()

chorizo = LinkedRDKitChorizo(pdb_str)
mk_prep = MoleculePreparation()
chorizo.flexibilize_protein_sidechain("A:TYR:4", mk_prep) # to add rdkit mol to molsetup
residue = chorizo.residues["A:TYR:4"]

# molsetup JSON conversion
molsetup = residue.molsetup
encoder = MoleculeSetupEncoder()
d = encoder.default(molsetup)
s = json.dumps(d)
molsetup2 = RDKitMoleculeSetup.from_json(s)

# chorizo residue JSON conversion
jsonstr = residue.to_json()
residue2 = ChorizoResidue.from_json(jsonstr) 
