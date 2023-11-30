#!/usr/bin/env python

import json
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from meeko import load_openff
from meeko import AtomTyper
from meeko import MoleculePreparation

vdw_list, dihedral_list, vdw_by_atype = load_openff()

meeko_config = {
    "merge_these_atom_types": (),
    "flexible_amides": True,
    "atom_params": {
        "vdw": vdw_list
     },
    "charge_model": "espaloma",
    "dihedral_params": dihedral_list,
}

meeko_prep = MoleculePreparation.from_config(meeko_config)

#mol = Chem.MolFromMolFile("baloxa-core-noether-noOH-okchiral-noMethyl.sdf", removeHs=False)

mol = Chem.MolFromSmiles("CN1CN(C)n2ccc(=O)cc2C1=O")
mol = Chem.AddHs(mol)
rdDistGeom.EmbedMolecule(mol)

meeko_prep.prepare(mol)
molsetup = meeko_prep.setup

print("Unique dihedral potentials in this molecule:")
for i, term in enumerate(molsetup.dihedral_interactions):
    print(i, term)

print()
print("List of contributions:")
for atom_idxs in molsetup.dihedral_partaking_atoms:
    term_id = molsetup.dihedral_partaking_atoms[atom_idxs]
    openff_label = molsetup.dihedral_labels[atom_idxs]
    print("atoms involved: %12s, openff_label: %s, term_id: %d" % (" ".join(["%2d" % i for i in atom_idxs]), openff_label, term_id))







