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
    def __init__(self):
        self.parameters = self._init_parms()

    def _init_parms(self):
        """ parse parameters from data file"""
        self.parms = OrderedDict()
        d = utils.path_module("meeko")
        f = os.path.join(d, "data/waterfield.txt")
        with open(f, 'r') as fp:
            raw = fp.readlines()
        for i, r in enumerate(raw):
            if r.startswith('ATOM'):
                r = r.split(None,8)[1:]
                self.parms[r[6]] = {'atom_type': r[0], 'parms': r[1:6]}

    def set_types(self, mol):
        """ """
        atom_set = {}
        setup = mol.setup
        finder = setup.smarts
        for pattern, data in list(self.parms.items()):
            found = finder.find_pattern(pattern)
            if not found:
                continue
            atom_type = data['atom_type']
            for f in found:
                atom_idx = f[0]
                setup.set_atom_type(atom_idx, atom_type)
                atom_set[atom_idx] = atom_type
        # hydrogen bond type
        # TODO this shuold be made more robust DIOGO/JEROME
        """
        for atom_idx, atom_type in list(atom_set.items()):
            vectors = self._is_hb(atom_idx, atom_type)
            if vectors is not None:
                setup.add_interaction_vector(atom_idx, vectors)
        """

    def set_param_legacy(self, mol, strict=True):
        """ define atom parameters for AD 4.2.x
            strict: if True (default) adhere to the strict
            definition of the AD atom types v4.2.1

            otherwise, disable HB capabilities for aromatic
            oxygen and sulfur, and for R-NO2
        """
        setup = mol.setup
        if not strict:
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

        def _get_hydro_type(hydrogen):
            """identify polar HB (HD type)"""
            for n in ob.OBAtomAtomIter(hydrogen):
                if n.GetAtomicNum() in (7, 8, 16):
                    return "HD"
                return "H"

        for atom in ob.OBMolAtomIter(mol):
            atom_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            element = ob.GetSymbol(atomic_num)
            atom_type = element
            # hydrogen
            if element == "H":
                atom_type = _get_hydro_type(atom)
            # carbon
            elif element == "C":
                if atom.IsAromatic():
                    atom_type = "A"
            # oxygen
            elif element == 'O':
                atom_type = 'OA'
                if not strict and atom_idx in oxygen_cache:
                    atom_type = 'O'
            # sulfur
            elif element == 'S':
                atom_type = 'SA'
                if not strict:
                    if atom.IsAromatic():
                        atom_type = 'S'
            #print("Setting atom types", atom_idx, atom_type)
            setup.set_atom_type(atom_idx, atom_type)

    def _is_hb(self, idx, atom_type):
        """ XXX TMP PLACEHOLDER TO DO"""
        if 'H_O' in atom_type:
            # donor
            # return a dummy vector
            return [[1.0,2.0,3.0]]

        elif atom_type.startswith('O_'):
            # acceptor
            # return a dummy vector
            return [[3.0,2.0,1.0]]

    def calculate_hb_vectors(self):
        pass
