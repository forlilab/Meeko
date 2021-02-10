#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko preparation
#

import os
import sys
from collections import OrderedDict

from openbabel import openbabel as ob

from .setup import MoleculeSetup
from .atomtyper import AtomTyperLegacy
from .bondtyper import BondTyperLegacy
from .hydrate import HydrateMoleculeLegacy
from .macrocycle import FlexMacrocycle
from .flexibility import FlexibilityBuilder
from .writer import PDBQTWriterLegacy
from .utils import obutils


class MoleculePreparation:
    def __init__(self, merge_hydrogens=True, hydrate=False, macrocycle=False, amide_rigid=True):
        self._merge_hydrogens = merge_hydrogens
        self._add_water = hydrate
        self._break_macrocycle = macrocycle
        self._keep_amide_rigid = amide_rigid
        self._mol = None

        self._atom_typer = AtomTyperLegacy()
        self._bond_typer = BondTyperLegacy(self._keep_amide_rigid)
        self._macrocycle_typer = FlexMacrocycle(min_ring_size=7, max_ring_size=33, min_score=50) #max_ring_size=26, min_ring_size=8)
        self._flex_builder = FlexibilityBuilder()
        self._water_builder = HydrateMoleculeLegacy()
        self._writer = PDBQTWriterLegacy()

    def prepare(self, mol):
        """ """
        if mol.NumAtoms() == 0:
            raise ValueError('Error: no atoms present in the molecule')

        self._mol = mol
        MoleculeSetup(mol)

        # 1.  assign atom types (including HB types, vectors and stuff)
        # DISABLED TODO self.atom_typer.set_parm(mol)
        self._atom_typer.set_param_legacy(mol)

        # 2a. add pi-model + merge_h_pi (THIS CHANGE SOME ATOM TYPES)

        # 2b. merge_h_classic
        if self._merge_hydrogens:
            mol.setup.merge_hydrogen()

        # 3.  assign bond types by using SMARTS...
        #     - bonds should be typed even in rings (but set as non-rotatable)
        #     - if macrocycle is selected, they will be enabled (so they must be typed already!)
        self._bond_typer.set_types_legacy(mol)

        # 4 . hydrate molecule
        if self._add_water:
            self._water_builder.hydrate(mol)

        # 5.  scan macrocycles
        if self._break_macrocycle:
            # calculate possible breakable bonds
            self._macrocycle_typer.search_macrocycle(mol)

        # 6.  build flexibility...
        # 6.1 if macrocycles typed:
        #     - walk the setup graph by skipping proposed closures
        #       and score resulting flex_trees basing on the lenght
        #       of the branches generated
        #     - actually break the best closure bond (THIS CHANGES SOME ATOM TYPES)
        # 6.2  - walk the graph and build the flextree
        # 7.  but disable all bonds that are in rings and not
        #     in flexible macrocycles
        # TODO restore legacy AD types for PDBQT
        #self._atom_typer.set_param_legacy(mol)
        self._flex_builder.process_mol(mol)
        # TODO re-run typing after breaking bonds
        # self.bond_typer.set_types_legacy(mol, exclude=[macrocycle_bonds])
    
    def show_setup(self):
        if self._mol is not None:
            tot_charge = 0

            print("Molecule setup\n")
            print("==============[ ATOMS ]===================================================")
            print("idx  |          coords            | charge |ign| atype    | connections")
            print("-----+----------------------------+--------+---+----------+--------------- . . . ")
            for k, v in list(self._mol.setup.coord.items()):
                print("% 4d | % 8.3f % 8.3f % 8.3f | % 1.3f | %d" % (k, v[0], v[1], v[2],
                      self._mol.setup.charge[k], self._mol.setup.atom_ignore[k]),
                      "| % -8s |" % self._mol.setup.atom_type[k],
                      self._mol.setup.graph[k])
                tot_charge += self._mol.setup.charge[k]
            print("-----+----------------------------+--------+---+----------+--------------- . . . ")
            print("  TOT CHARGE: %3.3f" % tot_charge)

            print("\n======[ DIRECTIONAL VECTORS ]==========")
            for k, v in list(self._mol.setup.coord.items()):
                if k in self._mol.setup.interaction_vector:
                    print("% 4d " % k, self._mol.setup.atom_type[k], end=' ')

            print("\n==============[ BONDS ]================")
            # For sanity users, we won't show those keys for now
            keys_to_not_show = ['bond_order', 'type']
            for k, v in list(self._mol.setup.bond.items()):
                t = ', '.join('%s: %s' % (i, j) for i, j in v.items() if not i in keys_to_not_show)
                print("% 8s - " % str(k), t)

            self._macrocycle_typer.show_macrocycle_scores()

            print('')
    
    def write_pdbqt_string(self):
        if self._mol is not None:
            return self._writer.write_string(self._mol)
        else:
            raise RuntimeError('Cannot generate PDBQT file, the molecule is not prepared.')

    def write_pdbqt_file(self, pdbqt_filename):
        try:
            with open(pdbqt_filename,'w') as w:
                w.write(self.write_pdbqt_string())
        except:
            raise RuntimeError('Cannot write PDBQT file %s.' % pdbqt_filename)
