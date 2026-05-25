#!/usr/bin/env python

import pathlib
from rdkit import Chem
from rdkit.Chem import rdDistGeom
from meeko import MoleculePreparation
import numpy as np
import pytest

try:
    import openforcefields
    _got_openff = True
except ImportError as err:
    _got_openff = False


workdir = pathlib.Path(__file__)
datadir = workdir.parents[0] / "parameterization_data"
lig_3zlq_a = datadir / "3zlq_lig_a.sdf"
lig_3zlq_b = datadir / "3zlq_lig_b.sdf"


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

@pytest.mark.skipif(not _got_openff, reason="requires openff-forcefields")
def test_openff_acetate():
    etkdg_v3 = Chem.rdDistGeom.ETKDGv3()
    acetate = Chem.MolFromSmiles("CC(=O)[O-]")
    acetate = Chem.AddHs(acetate)
    Chem.rdDistGeom.EmbedMolecule(acetate, etkdg_v3)
    mk_prep_200 = MoleculePreparation(load_atom_params=["openff"]) 
    mk_prep_230 = MoleculePreparation(load_atom_params=["openff-2.3.0"]) 
    molsetup_200 = mk_prep_200(acetate)[0]
    molsetup_230 = mk_prep_230(acetate)[0]
    rmin_half_200 = molsetup_200.atom_params["rmin_half"]
    rmin_half_230 = molsetup_230.atom_params["rmin_half"]
    assert rmin_half_200 != rmin_half_230
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

def test_mbondi():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1nc[nH]c1CO"))
    etkdg_v3 = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol, etkdg_v3)
    mk_prep = MoleculePreparation(load_atom_params="gb_mbondi")
    molsetup = mk_prep(mol)[0]
    assert 1.3 in molsetup.atom_params["mbondi_radius"]  # H bound to N or C
    assert 0.8 in molsetup.atom_params["mbondi_radius"]  # H bound to O or S
    assert 1.2 not in molsetup.atom_params["mbondi_radius"]  # other H
    assert molsetup.atom_params["gb_screen"].count(0.72) == 4  # C
    assert molsetup.atom_params["gb_screen"].count(0.79) == 2  # N

def test_merge_crippen():
    mol = Chem.AddHs(Chem.MolFromSmiles("OC(F)F"))
    etkdg_v3 = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol, etkdg_v3)
    mk_prep = MoleculePreparation(crippen=True, load_atom_params=["ad4_types"])
    molsetup = mk_prep(mol)[0]
    mk_prep = MoleculePreparation(crippen=True, load_atom_params=["ad4_types"], merge_these_atom_params=["crippen"])
    molsetup_merge = mk_prep(mol)[0]
    h_mask = np.array([atom.atom_type == "H" for atom in molsetup.atoms])
    crippen_orig = np.array(molsetup.atom_params["crippen"])
    crippen_merge = np.array(molsetup_merge.atom_params["crippen"])
    abs_diff = np.abs(crippen_merge - crippen_orig)
    assert abs(sum(crippen_orig) - sum(crippen_merge)) < 1e-8
    assert np.max(crippen_orig[h_mask]) > 1e-2
    assert np.max(crippen_merge[h_mask]) < 1e-8
    assert np.min(abs_diff[h_mask]) > 1e-2
    assert np.max(abs_diff[~h_mask]) > 1e-2

def test_override_ad4sol_par_including_q():
    qasp = 0.01097
    mol = Chem.AddHs(Chem.MolFromSmiles("c1nc[nH]c1CO"))
    etkdg_v3 = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol, etkdg_v3)
    mk_prep = MoleculePreparation(load_atom_params=["ad4_types", "ad4_desolv_volume", "ad4_desolv_param"])
    molsetup_default = mk_prep(mol)[0]
    ref_par = np.array(molsetup_default.atom_params["ad4_sol_par"])
    ref_par += qasp * np.abs([atom.charge for atom in molsetup_default.atoms])
    mk_prep_mod = MoleculePreparation(
        charge_model="zero", 
        merge_these_atom_types=[],
        override_ad4sol_par_including_q=True,
        override_ad4sol_par_including_q_qasp=qasp,
    )
    molsetup_mod = mk_prep_mod(mol)[0]
    mod_par = np.array(molsetup_mod.atom_params["ad4_sol_par"]) 
    assert np.max(np.abs([atom.charge for atom in molsetup_default.atoms])) > np.finfo(float).eps
    assert np.max(np.abs([atom.charge for atom in molsetup_mod.atoms])) < np.finfo(float).eps
    assert np.max(np.abs(ref_par - mod_par)) < np.finfo(float).eps
    assert np.max(np.abs(ref_par)) > np.finfo(float).eps

def test_add_atom_types_with_multiple_typing_groups():
    mol = Chem.AddHs(Chem.MolFromSmiles("c1nc[nH]c1CO"))
    etkdg_v3 = rdDistGeom.ETKDGv3()
    rdDistGeom.EmbedMolecule(mol, etkdg_v3)
    mk_prep = MoleculePreparation(
        load_atom_params=["ad4_types", "gb_mbondi"],
        add_atom_types=[{"smarts": "[#1]", "atype": "H"}],
    )
    molsetup = mk_prep(mol)[0]
    return
