#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko atom typer
#

import os
from collections import OrderedDict

from openbabel import openbabel as ob

from .utils import utils


class AtomTyperLegacy:
    def __init__(self):
        """Initialize the legacy atom typer for AutoDock 4.2.x

        """
        self._nitrogen_nonpolar_patterns = {#'[#7]': ('NA', 0),
                                            #'[#7X2v3]': ('NA', 0), # immine, pyridine
                                            #'[#7;X1]': ('NA', 0), # HCN
                                            #'[#7X3v3]': ('NA', 0), # amine, pyrrole, aniline, amide
                                            '[#7X3v3][a]': ('N', 0), # pyrrole, aniline
                                            '[#7X3v3][#6X3v4]': ('N', 0), # amide
                                            '[#7X4]': ('N', 0)
                                            }

        self._oxygen_nonpolar_patterns = {#"[#8]": ('OA', 0),
                                          '[#7](~O)(~O)(~a)': ('O', 1), # aromatic-nitro
                                          '[a]~[#8]~[a]': ('O', 1), # aromatic ether
                                          '[#6](-[#8])(=[#8])': ('O', 1), # carboxy hydroxyl
                                          '[o]': ('O', 0), # aromatic oxygen
                                          }
    
    def _hydrogen_atom_type(self, hydrogen):
        """identify polar HB (HD type)"""
        for n in ob.OBAtomAtomIter(hydrogen):
            if n.GetAtomicNum() in (7, 8, 16):
                return "HD"
            return "H"

    def set_param_legacy(self, mol):
        """Setting the atom types

        Args:
            mol (OBMol): input OBMol molecule object

        """
        oxygen_nonpolars = []
        nitrogen_nonpolars = []
        setup = mol.setup

        # Find non-polar oxygen atoms
        for smarts, config in self._oxygen_nonpolar_patterns.items():
            found = setup.smarts.find_pattern(smarts)
            for group in found:
                oxygen_nonpolars.append(group[config[1]])

        oxygen_nonpolars = set(oxygen_nonpolars)

        # Find non-polar nitrogen atoms
        for smarts, config in self._nitrogen_nonpolar_patterns.items():
            found = setup.smarts.find_pattern(smarts)
            for group in found:
                nitrogen_nonpolars.append(group[config[1]])

        nitrogen_nonpolars = set(nitrogen_nonpolars)

        # Go through all the atoms now
        for atom in ob.OBMolAtomIter(mol):
            atom_idx = atom.GetIdx()
            atomic_num = atom.GetAtomicNum()
            element = ob.GetSymbol(atomic_num)
            atom_type = element

            if element == "H":
                atom_type = self._hydrogen_atom_type(atom)
            elif element == "C":
                if atom.IsAromatic():
                    atom_type = "A"
            elif element == 'O':
                atom_type = 'O' if atom_idx in oxygen_nonpolars else 'OA'
            elif element == 'N':
                atom_type = 'N' if atom_idx in nitrogen_nonpolars else 'NA'
            elif element == 'S':
                atom_type = 'S' if atom.IsAromatic() else 'SA'

            setup.set_atom_type(atom_idx, atom_type)
