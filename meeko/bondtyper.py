#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko bond typer
#

from rdkit.Chem.rdchem import BondType


class BondTyperLegacy:

    def __call__(
        self, setup, flexible_amides, rigidify_bonds_smarts, rigidify_bonds_indices
    ):
        """Typing atom bonds in the legacy way

        Args:
            setup: MoleculeSetup object

            rigidify_bond_smarts (list): patterns to freeze bonds, e.g. conjugated carbons
        """

        canonicalize_bond = lambda i, j: (i, j) if i < j else (j, i)  # same as Bond.get_bond_id
        amide_bonds = [
            canonicalize_bond(x[0], x[1]) for x in setup.find_pattern("[NX3]-[CX3]=[O,N,S]")
        ]  # includes amidines

        # tertiary amides with non-identical substituents will be allowed to rotate
        tertiary_amides = [
            x for x in setup.find_pattern("[NX3]([!#1])([!#1])-[CX3]=[O,N,S]")
        ]
        equivalent_atoms = setup.get_equivalent_atoms()
        num_amides_removed = 0
        num_amides_originally = len(amide_bonds)
        for x in tertiary_amides:
            r1, r2 = x[1], x[2]
            if equivalent_atoms[r1] != equivalent_atoms[r2]:
                amide_bonds.remove(canonicalize_bond(x[0], x[3]))
                num_amides_removed += 1
        assert num_amides_originally == num_amides_removed + len(amide_bonds)

        single_triple_single = [
            (x[0], x[1], x[2], x[3]) for x in setup.find_pattern("[*]-[*]#[*]-[*]")
        ]
        single_to_rigidify = []
        triple_to_rotate = []
        for i, j, k, m in single_triple_single:
            triple_to_rotate.append(canonicalize_bond(j, k))
            single_to_rigidify.append(canonicalize_bond(i, j))
            single_to_rigidify.append(canonicalize_bond(k, m))
        # fully rigidify nitrile and alike
        single_to_rigidify.extend(
            [canonicalize_bond(x[0], x[1]) for x in setup.find_pattern("[*]-[*]#[*X1]")]
        )
        to_rigidify = set()
        n_smarts = len(rigidify_bonds_smarts)
        assert n_smarts == len(rigidify_bonds_indices)
        for i in range(n_smarts):
            a, b = rigidify_bonds_indices[i]
            smarts = rigidify_bonds_smarts[i]
            indices_list = setup.find_pattern(smarts)
            for indices in indices_list:
                atom_a = indices[a]
                atom_b = indices[b]
                to_rigidify.add(canonicalize_bond(atom_a, atom_b))

        for bond_id, bond in setup.bond_info.items():
            if (
                bond_id[0] >= setup.mol.GetNumAtoms()
                or bond_id[1] >= setup.mol.GetNumAtoms()
            ):
                continue  # at least one of the atoms is pseudo or dummy
            rdkit_bond = setup.mol.GetBondBetweenAtoms(bond_id[0], bond_id[1])
            rotatable = rdkit_bond.GetBondType() == BondType.SINGLE
            if bond_id in to_rigidify:
                rotatable = False
            # check if bond is amide
            if bond_id in amide_bonds and not flexible_amides:
                rotatable = False
            if bond_id in single_to_rigidify:
                rotatable = False
            if bond_id in triple_to_rotate:
                rotatable = True
            bond.rotatable = rotatable
