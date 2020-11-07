#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
#

import os
import sys
from collections import defaultdict
from operator import itemgetter

from openbabel import openbabel as ob


# TODO
# NOTES
# - bond type values > 0 must mean the bond could rotate (if enabled)
class BondTyperLegacy:
    """
        SMARTS patterns for rotatable bonds should *not* contain
        ring specifications: ring bonds will be pruned after
        assignment is performed, to guarantee that macrocycle
        bonds are correctly typed, if they are active.

    """
    def __init__(self): #, flex_macrocycle=False):
        """ """
        #self.macrocycle = None
        #if flex_macrocycle:
        #    # parameters here
        #    self.macrocycle = FlexibleMacroCycle()

        #
        # the proper bond typing will be done:
        # 1. assing rotatable typing with smarts (ignoring ring)
        # 2. keep track of which rings rigidify it (if >1; ignore, set None)
        # 3. enabling a macrocycle  will trigger all these bonds to be fred

    def set_types(self, mol):
        """ """
        """ here SMARTS patterns should take place"""
        self.mol = mol
        setup = mol.setup
        macrocycle_bonds = []
        self._cache_amide(mol)

        for b in ob.OBMolBondIter(mol):
            begin = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            bond_id = setup.get_bond_id(begin, end)
            bond = setup.get_bond(begin,end)
            rotatable = True
            terminal = (len(setup.graph[begin]) == 1 or len(setup.graph[end]) == 1)
            in_ring = self.mol.GetBond(begin,end).IsInRing()
            # smarts magic should happen here
            if in_ring or terminal:
                rotatable = False
            # TODO SMART THE HECK OUT OF THE MOLECULE HERE
            _type = 42
            setup.bond[bond_id]['rotatable'] = rotatable
            setup.bond[bond_id]['type'] = _type

    def _cache_amide(self, mol):
        """ store info about amides to be made non-rotatable"""
        pattern = "[NH1]-[CX3]=[OX1]"
        self._non_rotatable_amides = []
        found = mol.setup.smarts.find_pattern(pattern)
        if found is None:
            return
        for p in found:
            couple = mol.setup.get_bond_id(p[0], p[1] )
            couple = mol.setup.get_bond_id(*couple)
            self._non_rotatable_amides.append(couple)
        self._non_rotatable_amides = set(self._non_rotatable_amides)
        print("FOUND THESE MANY AMIDES", self._non_rotatable_amides)

    def set_types_legacy(self, mol):
        """ here SMARTS patterns should take place"""
        # self.mol = mol
        def _is_methyl(atom_idx):
            atom = mol.GetAtom(atom_idx)
            h_count = len([x for x in ob.OBAtomAtomIter(atom) if x.GetAtomicNum() == 1])
            # if h_count == 3:
                # print "METHYL", atom_idx
            return h_count == 3

        setup = mol.setup
        # self._cache_amide(mol)
        for b in ob.OBMolBondIter(mol):
            begin = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            bond_id = setup.get_bond_id(begin, end)
            bond = setup.get_bond(begin,end)
            bond_obj = mol.GetBond(begin,end)
            bond_order = bond_obj.GetBondOrder()
            if bond_obj.IsAromatic():
                bond_order = 5
            rotatable = True
            terminal = (len(setup.graph[begin]) == 1 or len(setup.graph[end]) == 1)
            # CHECK FOR METYL GROUPS!
            if _is_methyl(begin) or _is_methyl(end):
                terminal = True
            in_ring = bond_obj.IsInRing()
            is_amide = False
            if bond_obj.IsAmide() and not bond_obj.IsTertiaryAmide() and False:
            # if bond_id in self._non_rotatable_amides:
                is_amide = True
                bond_order = 99
            # smarts magic should happen here
            # if in_ring or terminal or bond_obj.GetBondOrder()>1 or bond_obj.IsAromatic():
            if in_ring or terminal or bond_order > 1 or is_amide:
                rotatable = False
            # TODO SMART THE HECK OUT OF THE MOLECULE HERE
            _type = 42
            setup.bond[bond_id]['rotatable'] = rotatable
            setup.bond[bond_id]['type'] = _type
            setup.bond[bond_id]['bond_order'] = bond_order

        #return bond_types
