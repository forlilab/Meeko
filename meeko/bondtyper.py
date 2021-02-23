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

    def __call__(self, mol, keep_amide_rigid):
        """Typing atom bonds in the legacy way
        
        Args:
            mol (OBMol): input OBMol molecule object

        """

        for ob_bond in ob.OBMolBondIter(mol):
            rotatable = True
            # @Stefano, might want to define something else for the default type (-1 ?)
            default_type = 42

            begin = ob_bond.GetBeginAtomIdx()
            end = ob_bond.GetEndAtomIdx()

            bond_order = ob_bond.GetBondOrder()

            if bond_order > 1:
                rotatable = False

            if ob_bond.IsInRing():
                rotatable = False

            if ob_bond.GetBeginAtom().GetExplicitDegree() == 1 or ob_bond.GetEndAtom().GetExplicitDegree(): # terminal
                rotatable = False

            if ob_bond.IsAromatic():
                bond_order = 5
                rotatable = False

            # Check for methyl group
            if mol.setup.is_methyl(begin) or mol.setup.is_methyl(end):
                terminal = True

            # @Stefano, Is it also for the amide bonds?
            if ob_bond.IsAmide() and keep_amide_rigid:
                bond_order = 99
                rotatable = False

            if self._is_imide(ob_bond):
                rotatable = False

            bond_id = mol.setup.get_bond_id(begin, end)
            mol.setup.bond[bond_id]['rotatable'] = rotatable
            mol.setup.bond[bond_id]['type'] = default_type
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
        if n.GetImplicitValence() != 3:
            return False
        # make sure C is attached to =N
        for b in ob.OBAtomBondIter(c):
            if isImide(b):
                return True
        return False
