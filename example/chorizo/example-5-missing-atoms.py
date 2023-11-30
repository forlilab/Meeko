#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy


# This works fine (ALA stands for alanine amino acid)
chorizo = LinkedRDKitChorizo("just-one-ALA.pdb")

# The following will raise an error because ALA has missing atoms
# the line corresponding to atom named CB is missing
try:
    chorizo = LinkedRDKitChorizo("just-one-ALA-missing-CB.pdb")
except RuntimeError as err:
    print("Failed to read just-one-ALA-missing-CB.pdb")
    print(err)

# now we avoid the error and put the failed ALA in removed_residues
chorizo = LinkedRDKitChorizo("just-one-ALA-missing-CB.pdb", allow_bad_res=True)

print("nr of residues:", len(chorizo.residues))
print("nr removed residues:", len(chorizo.removed_residues)) # not deleted residues

# We would like to interact with the chorizo object in a way that
# makes it natural to prompt the user to take deliberate action
# to edit, build, or modify problematic residues, either from
# a GUI or from a command line tool.
