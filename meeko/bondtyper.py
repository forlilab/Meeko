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
    def __init__(self, amide_rigid=True):
        """Initialize the legacy atom typer for AutoDock 4.2.x
        
        Args:
            amide_rigid (bool): consider amide bond as rigid (default: True)

        """
        self._keep_amide_rigid = amide_rigid
        self._amide_bonds = None

    def _identify_amide_bonds(self, mol):
        """Store info about amide bonds to be made non-rotatable.
        
        Args:
            mol (OBMol): input OBMol object
        
        """
        self._amide_bonds = []
        pattern = "[NX3][CX3](=[OX1])[#6]"

        found = mol.setup.smarts.find_pattern(pattern)

        for p in found:
            couple = mol.setup.get_bond_id(p[0], p[1])
            couple = mol.setup.get_bond_id(*couple)
            self._amide_bonds.append(couple)

        self._amide_bonds = set(self._amide_bonds)

    def set_types_legacy(self, mol):
        """Typing atom bonds in the legacy way
        
        Args:
            mol (OBMol): input OBMol molecule object

        """
        setup = mol.setup

        if self._keep_amide_rigid:
            self._identify_amide_bonds(mol)

        for b in ob.OBMolBondIter(mol):
            rotatable = True
            is_amide_bond = False
            # @Stefano, might want to define something else for the default type (-1 ?)
            default_type = 42

            begin = b.GetBeginAtomIdx()
            end = b.GetEndAtomIdx()
            bond_id = setup.get_bond_id(begin, end)
            bond = setup.get_bond(begin,end)
            bond_obj = mol.GetBond(begin,end)
            bond_order = bond_obj.GetBondOrder()
            in_ring = bond_obj.IsInRing()
            terminal = (len(setup.graph[begin]) == 1 or len(setup.graph[end]) == 1)

            if bond_obj.IsAromatic():
                bond_order = 5

            # Check for methyl group
            if setup.is_methyl(begin) or setup.is_methyl(end):
                terminal = True

            # Is this bond an amide bond?
            if (begin, end) in self._amide_bonds:
                is_amide_bond = True
                bond_order = 99

            # @Stefano, Is it also for the amide bonds?
            if bond_obj.IsAmide() and not bond_obj.IsTertiaryAmide() and False:
                is_amide_bond = True
                bond_order = 99

            # If in_ring or terminal or bond_obj.GetBondOrder() > 1 or bond_obj.IsAromatic():
            if in_ring or terminal or bond_order > 1 or is_amide_bond:
                rotatable = False

            # TODO SMART THE HECK OUT OF THE MOLECULE HERE
            setup.bond[bond_id]['rotatable'] = rotatable
            setup.bond[bond_id]['type'] = default_type
            setup.bond[bond_id]['bond_order'] = bond_order
