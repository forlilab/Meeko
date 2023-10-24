#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy
from meeko import MoleculePreparation

chorizo = LinkedRDKitChorizo("AHHY.pdb")

mk_prep = MoleculePreparation() # makes RDKitMoleculeSetup (aka molsetup) instances from RDKit molecules. This is what is used for small molecule (ligand) preparation

residue_id = "A:HIS:2" # <chain>:<resname>"<resnum>

chorizo.res_to_molsetup(residue_id, mk_prep)

# Make PDBQT for AutoDock-Vina or AutoDock-GPU
pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
rigid_part, movable_part = pdbqt_strings
print("length of rigid pdbqt:", len(rigid_part))
print("length of flexible residues pdbqt:", len(movable_part))


# Residues that are parameterized with the stored data ("meeko/data/prot\_res\_params.json") don't currently get an RDKitMoleculeSetup (molsetup).
for res in chorizo.res_list:
    has_molsetup = "molsetup" in chorizo.residues[res]
    print(res, "has molsetup:", has_molsetup)

