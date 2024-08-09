#!/usr/bin/env python

import pathlib
from rdkit import Chem
from meeko import MoleculePreparation


workdir = pathlib.Path(__file__).parents[0]


def run(
    folder,
    fn,
    expected_rot_bonds,
    expected_sizes_of_rigid_bodies,
    expected_members=None,
):

    path = workdir / pathlib.Path(folder) / fn
    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    mk_prep = MoleculePreparation()
    molsetup = mk_prep(mol)[0]

    # sizes of rigid groups/bodies
    rigid_body_members = molsetup.flexibility_model["rigid_body_members"]
    sizes = []
    for _, members in rigid_body_members.items():
        sizes.append(len(members))
    assert sorted(sizes) == sorted(expected_sizes_of_rigid_bodies)

    # check rotatable bonds
    rot_bonds = set()
    for bond_id, bond in molsetup.bond_info.items():
        if bond.rotatable:
            rot_bonds.add(bond_id)
    assert rot_bonds == expected_rot_bonds

    # optionally check actual atom indices in each rigid body
    if expected_members is not None:
        expected_members = set([tuple(sorted(members)) for members in expected_members])
        actual_members = set()
        for _, members in rigid_body_members.items():
            actual_members.add(tuple(sorted(members)))
        assert actual_members == expected_members
    return


def test_non_sequential_atom_ordering_01():
    run(
        fn="non_sequential_atom_ordering_01.mol",
        folder="flexibility_data",
        expected_rot_bonds={(0, 1)},
        expected_sizes_of_rigid_bodies=[2, 8],
    )


def test_non_sequential_atom_ordering_02():
    run(
        fn="non_sequential_atom_ordering_02.mol",
        folder="flexibility_data",
        expected_rot_bonds={(0, 1), (1, 2), (4, 6), (6, 7)},
        expected_sizes_of_rigid_bodies=[1, 1, 2, 6, 7],
        expected_members=[
            [1],
            [6],
            [2, 3],
            [7, 8, 9, 10, 11, 12],
            [0, 4, 5, 13, 14, 15, 16],
        ],
    )


def test_non_sequential_atom_ordering_03():
    run(
        fn="non_sequential_atom_ordering_03.mol",
        folder="macrocycle_data",
        expected_rot_bonds={(0, 1), (1, 2), (2, 3), (3, 8), (4, 6), (6, 7)},
        expected_sizes_of_rigid_bodies=[1, 1, 2, 2, 6, 7],
        expected_members=[
            [1],
            [2, 32],
            [3, 33],
            [6],
            [0, 4, 5, 16, 14, 13, 15],
            [7, 8, 9, 10, 11, 12],
        ],
    )


def test_non_sequential_atom_ordering_04():
    """this one lacks aliphatic C-C bond and can't break under CG/G typing"""
    run(
        fn="non_sequential_atom_ordering_04.mol",
        folder="macrocycle_data",
        expected_rot_bonds=set(),
        expected_sizes_of_rigid_bodies=[19],
    )


def test_non_sequential_atom_ordering_05():
    """two rings, one breakable, and one lacking an aliphatic C-C bond
    that can't break under CG/G typing"""
    run(
        fn="non_sequential_atom_ordering_05.mol",
        folder="macrocycle_data",
        expected_rot_bonds={(14, 16), (16, 17), (17, 22), (10, 11), (11, 19)},
        expected_sizes_of_rigid_bodies=[17, 1, 5, 2, 2],
    )
