#!/usr/bin/env python

from rdkit import Chem
from rdkit.Chem import rdDistGeom
from meeko import MoleculePreparation

def test():
    smiles = "C1CCCC1"
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    etkdgv3 = rdDistGeom.srETKDGv3()
    rdDistGeom.EmbedMolecule(mol, etkdgv3)
    mk_prep = MoleculePreparation()
    molsetup = mk_prep.prepare(mol)[0]
    nr_rot_bonds = 0
    for idxs, bond_info in molsetup.bond.items():
        nr_rot_bonds += int(bond_info["rotatable"])
    assert(nr_rot_bonds == 0)
