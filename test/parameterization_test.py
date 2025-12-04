#!/usr/bin/env python

import pathlib
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from meeko import MoleculePreparation
import meeko
import pytest

try:
    import openforcefields
    _got_openff = True
except ImportError as err:
    _got_openff = False


pkgdir = pathlib.Path(meeko.__file__).parents[1]

lig_3zlq_a = pkgdir / "test/parameterization_data/3zlq_lig_a.sdf"
lig_3zlq_b = pkgdir / "test/parameterization_data/3zlq_lig_b.sdf"


def canonicalize_four_indices(four_indices):
    a, b, c, d = four_indices
    if c > b:
        return a, b, c, d
    else:
        return d, c, b, a
def get_nr_paths_per_param(molsetup):
    param_to_nr_paths = {}
    index_to_atoms = {}
    for atoms, i in molsetup.dihedral_partaking_atoms.items():
        index_to_atoms.setdefault(i, [])
        index_to_atoms[i].append(atoms)
    for index, param in enumerate(molsetup.dihedral_interactions):
        param = str(param)
        assert param not in param_to_nr_paths
        if index not in index_to_atoms:
            continue  # SMARTS assignment overrides previous, dihedrals might not be used
        param_to_nr_paths[param] = len(index_to_atoms[index])
    return param_to_nr_paths

@pytest.mark.skipif(not _got_openff, reason="requires openff-forcefields")
def test_dihedral_indexing():
    mol_a = Chem.MolFromMolFile(str(lig_3zlq_a), removeHs=False)
    mol_b = Chem.MolFromMolFile(str(lig_3zlq_b), removeHs=False)
    mk_prep = MoleculePreparation(dihedral_model="openff")
    molsetup_a = mk_prep(mol_a)[0]
    molsetup_b = mk_prep(mol_b)[0]
    canonical_a = set([canonicalize_four_indices(idxs) for idxs in molsetup_a.dihedral_partaking_atoms])
    canonical_b = set([canonicalize_four_indices(idxs) for idxs in molsetup_b.dihedral_partaking_atoms])
    assert canonical_a == set(molsetup_a.dihedral_partaking_atoms) 
    assert canonical_b == set(molsetup_b.dihedral_partaking_atoms) 
    a_nr = get_nr_paths_per_param(molsetup_a)
    b_nr = get_nr_paths_per_param(molsetup_b)
    assert a_nr == b_nr
    return

@pytest.mark.skipif(not _got_openff, reason="requires openff-forcefields")
def test_openff_water():
    etkdg_v3 = Chem.rdDistGeom.ETKDGv3()
    water = Chem.MolFromSmiles("O")
    water = Chem.AddHs(water)
    Chem.rdDistGeom.EmbedMolecule(water, etkdg_v3)
    mk_prep = MoleculePreparation(load_atom_params=["openff"]) 
    molsetup = mk_prep(water)[0]
    rmin_half = molsetup.atom_params["rmin_half"]
    epsilon = molsetup.atom_params["epsilon"]
    assert rmin_half[1] == rmin_half[2] 
    assert epsilon[1] == epsilon[2] 
    return

def test_merge_rmin_half():
    methanol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    etkdg_v3 = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(methanol, etkdg_v3)
    mk_prep = MoleculePreparation(
        merge_these_atom_types=["H"],
        load_atom_params=["ad4_types", "ad4_vdw"],
        merge_rmin_half=False,
    )
    molsetup = mk_prep(methanol)[0]
    mk_prep_merge = MoleculePreparation(
        merge_these_atom_types=["H"],
        load_atom_params=["ad4_types", "ad4_vdw"],
        merge_rmin_half=True,
    )
    molsetup_merge = mk_prep_merge(methanol)[0]
    c_normal = molsetup.atom_params["rmin_half"][0]
    c_merged = molsetup_merge.atom_params["rmin_half"][0]
    assert c_merged > c_normal
    return
