#!/usr/bin/env python

from meeko import LinkedRDKitChorizo
from meeko import PDBQTWriterLegacy

from rdkit import Chem


pdb_fn = "just-three-residues.pdb"
print(f"Loading {pdb_fn}")

# Argument "termini" modifies the capping of terminal residues.
# what happens internally is that we will prefix the residue name
# with "N", so we will use the data stored for "NMET" instead of "MET".
# it may be cleaner to pass something like {":MET:15": "NMET"},
# i.e. treat it like a "mutation"
termini = {":MET:15": "N"}

chorizo = LinkedRDKitChorizo(pdb_fn, termini=termini)
print("nr of residues:", len(chorizo.residues))
print("nr removed residues:", len(chorizo.removed_residues))

# Make PDBQT for AutoDock-Vina or AutoDock-GPU
pdbqt_strings = PDBQTWriterLegacy.write_string_from_linked_rdkit_chorizo(chorizo)
rigid_part, movable_part = pdbqt_strings
print("length of rigid pdbqt:", len(rigid_part))
print("length of flexible residues pdbqt:", len(movable_part))

met15_padded, is_actual_res, atom_index_map = chorizo.get_padded_mol(":MET:15")
met15_resmol = chorizo.residues[":MET:15"]["resmol"]

# met15_padded is an instance of an RDMol
# is_acutal_res is a list of bools to indicate whether each atom is MET15 or padding
# atom_index_map maps the atom indices of the padded molecules (keys) to the
# atom indices of the residue rdkit molecule (chorizo.residues[":MET:15"]["resmol"]
Chem.MolToMolFile(met15_padded, "met15_padded.mol")
Chem.MolToMolFile(met15_resmol, "met15_resmol.mol")
