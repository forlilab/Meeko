#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy

# AHHY is named after the four residues A: alanine, H: histidine, Y: tyrosine
# the protonation state of the histidines is determined based on the
# existing H atoms (hydrogens). There are three templates for histidine
# (HIE/HID/HIP) and the best match of all three is retained
chorizo = LinkedRDKitChorizo("AHHY.pdb")
print("nr of residues:", len(chorizo.residues))
print("nr removed residues:", len(chorizo.removed_residues))

# Make PDBQT for AutoDock-Vina or AutoDock-GPU
pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
rigid_part, movable_part = pdbqt_strings
print("length of rigid pdbqt:", len(rigid_part))
print("length of flexible residues pdbqt:", len(movable_part))
