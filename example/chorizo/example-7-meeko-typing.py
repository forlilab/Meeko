#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy
from meeko import MoleculePreparation
import numpy as np

chorizo = LinkedRDKitChorizo("AHHY.pdb")

atom_params, coords = chorizo.export_static_atom_params()
default_types = atom_params["atom_type"]

# modify atom type of tyrosine HD to HX
new_atom_types = [{"smarts": "[H]Oc", "atype": "HX"}]
mk_prep = MoleculePreparation(add_atom_types=new_atom_types)
chorizo.mk_parameterize_residue("A:TYR:4", mk_prep)
atom_params, coords = chorizo.export_static_atom_params()
modified_types = atom_params["atom_type"]

for deftype, modtype in zip(default_types, modified_types):
    line = f"{deftype:5} {modtype:5}"
    if deftype != modtype:
        line += " <--"
    print(line)

