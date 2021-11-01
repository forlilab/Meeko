#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko bond typer
#

import os
import sys
from collections import defaultdict
from operator import itemgetter

from openbabel import openbabel as ob


class BondTyperLegacy:

    def __call__(self, mol, flexible_amides, rigidify_bonds_smarts, rigidify_bonds_indices):
        """Typing atom bonds in the legacy way

        Args:
            mol: input OBMol/RDKit molecule object

            rigidify_bond_smarts (list): patterns to freeze bonds, e.g. conjugated carbons
        """
        def _is_terminal(idx):
            """ check if the atom has more than one connection with non-hydrogen atoms"""
            if mol.setup.get_element(idx) == 1:
                return True
            return len([x for x in mol.setup.get_neigh(idx) if not mol.setup.get_element(idx) == 1]) == 1
        # cache aromatic bonds
        aromatic_bonds = [set(x) for x in mol.setup.smarts.find_pattern('[a]~[a]') ]
        # TODO figure out if that's what we want?
        aromatic_bonds = []
        # cache amidine bonds
        amide_bonds = [ set(x) for x in mol.setup.smarts.find_pattern('C(=O)-[$([#7X2])]') ]
        amidine_bonds = [ set(x) for x in  mol.setup.smarts.find_pattern("[$([#6]~[#7])]~[#7]")]

        to_rigidify = set()
        n_smarts = len(rigidify_bonds_smarts)
        assert(n_smarts == len(rigidify_bonds_indices))
        for i in range(n_smarts):
            a, b = rigidify_bonds_indices[i]
            smarts = rigidify_bonds_smarts[i]
            indices_list = mol.setup.smarts.find_pattern(smarts)
            for indices in indices_list:
                atom_a = indices[a]
                atom_b = indices[b]
                to_rigidify.add((atom_a, atom_b))
                to_rigidify.add((atom_b, atom_a))

        for bond_id, bond_info in mol.setup.bond.items():
            rotatable = True
            bond_order = mol.setup.bond[bond_id]['bond_order']
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
            # bond between aromatics # TODO careful with this?
            if bond_id in aromatic_bonds:
                bond_order = 5
                rotatable = False
            # it's a terminal atom (methyl, halogen, hydrogen...)
            if _is_terminal(bond_id[0]) or _is_terminal(bond_id[1]):
                rotatable = False
            # check if bond is amide
            # NOTE this should have been done during the setup, right?
            if bond_id in amide_bonds and not flexible_amides:
                rotatable = False
                bond_order = 99
            # amidine
            if bond_id in amidine_bonds:
                bond_order = 99
                rotatable = False
            mol.setup.bond[bond_id]['rotatable'] = rotatable
            mol.setup.bond[bond_id]['bond_order'] = bond_order

