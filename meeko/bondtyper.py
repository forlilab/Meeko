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
            mol (OBMol): input OBMol molecule object

            rigidify_bond_smarts (list): patterns to freeze bonds, e.g. conjugated carbons
        """

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

        for ob_bond in ob.OBMolBondIter(mol):
            rotatable = True
            begin = ob_bond.GetBeginAtomIdx()
            end = ob_bond.GetEndAtomIdx()
            bond_order = ob_bond.GetBondOrder()

            if (begin, end) in to_rigidify:
                bond_order = 1.1 # macrocycle class breaks bonds if bond_order == 1
                rotatable = False

            if bond_order > 1:
                rotatable = False

            if ob_bond.IsInRing():
                rotatable = False

            if ob_bond.GetBeginAtom().GetExplicitDegree() == 1 or ob_bond.GetEndAtom().GetExplicitDegree() == 1: # terminal
                rotatable = False

            if ob_bond.IsAromatic():
                bond_order = 5
                rotatable = False

            if mol.setup.is_methyl(begin) or mol.setup.is_methyl(end):
                rotatable = False

            if ob_bond.IsAmide() and not flexible_amides:
                bond_order = 99
                rotatable = False

            if self._is_amidine(ob_bond):
                bond_order = 99
                rotatable = False

            bond_id = mol.setup.get_bond_id(begin, end)
            mol.setup.bond[bond_id]['rotatable'] = rotatable
            mol.setup.bond[bond_id]['bond_order'] = bond_order

    def _is_imide(self, ob_bond):
        """ python version of openbabel pdbqtformat.cpp/IsImide(OBBond* querybond)"""
        if ob_bond.GetBondOrder() != 2:
            return False
        bgn = ob_bond.GetBeginAtom().GetAtomicNum()
        end = ob_bond.GetEndAtom().GetAtomicNum()
        if (bgn == 6 and end == 7) or (bgn == 7 and end == 6):
            return True
        return False
    
    def _is_amidine(self, ob_bond):
        """ python version of openbabel pdbqtformat.cpp/IsImide(OBBond* querybond)"""
        if ob_bond.GetBondOrder() != 1:
            return False
        bgn = ob_bond.GetBeginAtom().GetAtomicNum()
        end = ob_bond.GetEndAtom().GetAtomicNum()
        if bgn == 6 and end == 7:
            c = ob_bond.GetBeginAtom()
            n = ob_bond.GetEndAtom()
        elif bgn == 7 and end == 6:
            n = ob_bond.GetBeginAtom()
            c = ob_bond.GetEndAtom()
        else:
            return False
        if n.GetExplicitDegree() != 3:
            return False
        # make sure C is attached to =N
        for b in ob.OBAtomBondIter(c):
            if self._is_imide(b):
                return True
        return False
