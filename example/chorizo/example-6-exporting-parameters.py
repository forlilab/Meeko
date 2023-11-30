#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy
import numpy as np

chorizo = LinkedRDKitChorizo("just-one-ALA.pdb")

atom_params, coords = chorizo.export_static_atom_params()
coords = np.array(coords)

print("coordinates")
print(coords)
print("")
print("atom parameters")
for key, values in atom_params.items():
    print(key, values)
