#!/usr/bin/env python

import pathlib
from rdkit import Chem
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
