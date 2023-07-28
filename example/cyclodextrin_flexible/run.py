#!/usr/bin/env python

from os import linesep
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
from vina import Vina

# first anchor atom of flex will be superimposed to first anchor atom of cyclodextrin
# same for second atom
# indexing from 1 for easy visualization in molecular viewers
anchor_atoms_flex = (9, 8)
anchor_atoms_cd = (8, 10)
first_flex_atom = 9 # first sidechain atom (from cyclodextrin side)

# the following atoms will be excluded from the rigid receptor
# because they are part of the sidechain
cd_ignore = anchor_atoms_cd + (147, 11, 85, 86)

cd_core = Chem.MolFromMolFile("cyclodextrin-core.mol", removeHs=False)
flex = Chem.MolFromMolFile("sidechain.mol", removeHs=False)
lig = Chem.MolFromMolFile("ligand.mol", removeHs=False)

anchor_map = (
    (anchor_atoms_flex[0] - 1, anchor_atoms_cd[0] - 1), # -1 to convert to 0-index
    (anchor_atoms_flex[1] - 1, anchor_atoms_cd[1] - 1)
)
Chem.rdMolAlign.AlignMol(flex, cd_core, atomMap=anchor_map)

mk_prep = MoleculePreparation()
molsetup_list = mk_prep.prepare(
    flex,
    root_atom_index=first_flex_atom - 1,
    not_terminal_atoms=(first_flex_atom - 1,) # to make the bond between anchor atoms rotatable
)

def check(ok, err):
    if not ok:
        print(err, file=sys.stderr)
        sys.exit()

# at the moment we only get more than one setup if mk_prep was configured with reactive_smarts
molsetup = molsetup_list[0]
flex_pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(molsetup)
check(ok, err)
flex_pdbqt_str = PDBQTWriterLegacy.adapt_pdbqt_for_autodock4_flexres(flex_pdbqt_str, res="", chain="", num="")

# make a PDBQT for the ligand
molsetup_list = mk_prep.prepare(lig)
molsetup = molsetup_list[0]
lig_pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(molsetup)
check(ok, err)

# make a pdbqt for the cyclodextrin
mk_prep = MoleculePreparation(
    rigidify_bonds_smarts=("[*][*]",), # no rotatable bonds
    rigidify_bonds_indices=((0, 1),),
)
molsetup_list = mk_prep.prepare(cd_core)
molsetup = molsetup_list[0]

# since we are removing atoms charges are not guaranteed to add up to an integer
# vina scoring function does not use charges, so that's OK
for i in cd_ignore:
    molsetup.atom_ignore[i - 1] = True # this atom won't be written to PDBQT

rec_pdbqt, ok, err = PDBQTWriterLegacy.write_string(molsetup)
check(ok, err)

# ROOT, ENDROOT, and TORSDOF should not be in the receptor
clean_rec_pdbqt = ""
for line in rec_pdbqt.split(linesep)[:-1]:
    if line.startswith("ROOT") or line.startswith("ENDROOT") or line.startswith("TORSDOF"):
        continue
    clean_rec_pdbqt += line + linesep

with open("receptor_rigid.pdbqt", "w") as f:
    f.write(clean_rec_pdbqt)

# flexible sidechain clashing with the cyclodextrin is OK, docking will resolve
with open("receptor_flex.pdbqt", "w") as f:
    f.write(flex_pdbqt_str)

v = Vina()
v.set_receptor(rigid_pdbqt_filename="receptor_rigid.pdbqt",
               flex_pdbqt_filename="receptor_flex.pdbqt")
v.compute_vina_maps(center=(20, 20, 35), box_size=(30, 30, 30))
v.set_ligand_from_string(lig_pdbqt_str)
v.dock()
results_pdbqt = v.poses()
pdbqt_mol = PDBQTMolecule(results_pdbqt)
docked_mols = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
with Chem.SDWriter("docked_ligand.sdf") as f:
    for i in range(docked_mols[0].GetNumConformers()):
        f.write(docked_mols[0], confId=i)

with Chem.SDWriter("docked_flex.sdf") as f:
    for i in range(docked_mols[1].GetNumConformers()):
        f.write(docked_mols[1], confId=i)
