#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko bond typer
#

import os
import sys
from collections import defaultdict
from operator import itemgetter


class BondTyperLegacy:

    def __call__(self, setup, flexible_amides, rigidify_bonds_smarts, rigidify_bonds_indices):
        """Typing atom bonds in the legacy way

        Args:
            setup: MoleculeSetup object

            rigidify_bond_smarts (list): patterns to freeze bonds, e.g. conjugated carbons
        """
        def _is_terminal(idx):
            """ check if the atom has more than one connection with non-ignored atoms"""
            if setup.get_element(idx) == 1:
                return True
            return len([x for x in setup.get_neigh(idx) if not setup.get_ignore(x)]) == 1
        amide_bonds = [(x[0], x[1]) for x in setup.find_pattern('[NX3]-[CX3]=[O,N]')] # includes amidines

        # tertiary amides with non-identical substituents will be allowed to rotate
        tertiary_amides = [x for x in setup.find_pattern('[NX3]([!#1])([!#1])-[CX3]=[O,N]')]
        equivalent_atoms = setup.get_equivalent_atoms()
        num_amides_removed = 0
        num_amides_originally = len(amide_bonds)
        for x in tertiary_amides:
            r1, r2 = x[1], x[2]
            if equivalent_atoms[r1] != equivalent_atoms[r2]:
                amide_bonds.remove((x[0], x[3]))
                num_amides_removed += 1
        assert(num_amides_originally == num_amides_removed + len(amide_bonds))

        to_rigidify = set()
        n_smarts = len(rigidify_bonds_smarts)
        assert(n_smarts == len(rigidify_bonds_indices))
        for i in range(n_smarts):
            a, b = rigidify_bonds_indices[i]
            smarts = rigidify_bonds_smarts[i]
            indices_list = setup.find_pattern(smarts)
            for indices in indices_list:
                atom_a = indices[a]
                atom_b = indices[b]
                to_rigidify.add((atom_a, atom_b))
                to_rigidify.add((atom_b, atom_a))

        for bond_id, bond_info in setup.bond.items():
            rotatable = True
            bond_order = setup.bond[bond_id]['bond_order']
            # bond requested to be rigid
            if bond_id in to_rigidify:
                bond_order = 1.1 # macrocycle class breaks bonds if bond_order == 1
                rotatable = False
            # non-rotatable bond
            if bond_info['bond_order'] > 1:
                rotatable = False
            # in-ring bond
            if len(bond_info['in_rings']):
                rotatable = False
            # it's a terminal atom (methyl, halogen, hydrogen...)
            if _is_terminal(bond_id[0]) or _is_terminal(bond_id[1]):
                rotatable = False
            # check if bond is amide
            # NOTE this should have been done during the setup, right?
            if (bond_id in amide_bonds or (bond_id[1], bond_id[0]) in amide_bonds) and not flexible_amides:
                rotatable = False
                bond_order = 99
            setup.bond[bond_id]['rotatable'] = rotatable
            setup.bond[bond_id]['bond_order'] = bond_order

