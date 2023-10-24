#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy

# We want both histidines to be "HIP" and to delete the tyrosine
mutations = {
        "A:HIS:2": "A:HIP:2",
        "A:HIS:3": "A:HIP:3",
        }
delete_residues = ("A:TYR:4",)

chorizo = LinkedRDKitChorizo(
        "AHHY.pdb",
        del_res=delete_residues,
        mutate_res_dict=mutations)

print("nr of residues:", len(chorizo.residues))
print("nr removed residues:", len(chorizo.removed_residues)) # not deleted residues

# Make PDBQT for AutoDock-Vina or AutoDock-GPU
pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
rigid_part, movable_part = pdbqt_strings
print("length of rigid pdbqt:", len(rigid_part))
print("length of flexible residues pdbqt:", len(movable_part))
