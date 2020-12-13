#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
#

import os
from collections import OrderedDict

from openbabel import openbabel as ob

from .utils import utils


class AtomTyperLegacy:
    def __init__(self, strict=True):
        self._strict = strict
    
    def _get_hydro_type(self, hydrogen):
        """identify polar HB (HD type)"""
        for n in ob.OBAtomAtomIter(hydrogen):
            if n.GetAtomicNum() in (7, 8, 16):
                return "HD"
            return "H"


    def set_param_legacy(self, mol):
        """ define atom parameters for AD 4.2.x
            strict: if True (default) adhere to the strict
            definition of the AD atom types v4.2.1

            otherwise, disable HB capabilities for aromatic
            oxygen and sulfur, and for R-NO2
        """
        setup = mol.setup
        if not self._strict:
            # cache non-polar oxygens
            oxygen_cache = []
            patterns = [("[#7](~O)(~O)(~a)",1), # aromatic-nitro
                        ("[a]~[#8]~[a]", 1), # aromatic ether
                        ("[#6](-[#8])(=[#8])", 1), # carboxy hydroxyl
                        ('[o]',0), # aromatic oxygen
                        ]

            for smarts, idx in patterns:
                found = mol.setup.smarts.find_pattern(smarts)
                if found:
                    for group in found:
                        oxygen_cache.append(group[idx])

            oxygen_cache = set(oxygen_cache)
        # nitrogen acceptor:
        # - aromatic with 2 connections
        # - any aliphatic without 4 connections
        nitro_acceptor_pattern = '[$([n;X2]),$([N;!X4])]' 
        nitro_acceptors = []
        found = mol.setup.smarts.find_pattern(nitro_acceptor_pattern)
        if found:
            for group in found:
                nitro_acceptors.append(group[0])
        

        for atom in ob.OBMolAtomIter(mol):
            atom_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            element = ob.GetSymbol(atomic_num)
            atom_type = element
            # hydrogen
            if element == "H":
                atom_type = self._get_hydro_type(atom)
            # carbon
            elif element == "C":
                if atom.IsAromatic():
                    atom_type = "A"
            # oxygen
            elif element == 'O':
                atom_type = 'OA'
                if not self._strict and atom_idx in oxygen_cache:
                    atom_type = 'O'
            # nitrogen
            elif element == 'N':
                if atom_idx in nitro_acceptors:
                    atom_type = 'NA'
            # sulfur
            elif element == 'S':
                atom_type = 'SA'
                if not self._strict:
                    if atom.IsAromatic():
                        atom_type = 'S'
            #print("Setting atom types", atom_idx, atom_type)
            setup.set_atom_type(atom_idx, atom_type)
