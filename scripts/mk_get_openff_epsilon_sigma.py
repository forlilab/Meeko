#!/usr/bin/env python

import json
import pathlib
import sys

from rdkit import Chem
import openforcefields

from meeko import MoleculePreparation
from meeko import get_openff_epsilon_sigma
from meeko import load_openff

vdw_list, _, vdw_by_type = load_openff()
sdf_filename = sys.argv[1]
rdkit_mol = Chem.MolFromMolFile(sdf_filename, removeHs=False)
data = get_openff_epsilon_sigma(
    rdkit_mol,
    vdw_list,
    vdw_by_type,
    output_index_start=0,
)
print(json.dumps(data, indent=4))
    

