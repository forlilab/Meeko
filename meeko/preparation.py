#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko preparation
#

import os
import sys
from collections import OrderedDict

from rdkit import Chem

from .setup import OBMoleculeSetup
from .setup import RDKitMoleculeSetup
from .atomtyper import AtomTyper
from .bondtyper import BondTyperLegacy
from .hydrate import HydrateMoleculeLegacy
from .macrocycle import FlexMacrocycle
from .flexibility import FlexibilityBuilder
from .writer import PDBQTWriterLegacy

try:
    from openbabel import openbabel as ob
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True


class MoleculePreparation:
    def __init__(self, keep_nonpolar_hydrogens=False,
            hydrate=False, flexible_amides=False,
            rigid_macrocycles=False, min_ring_size=7, max_ring_size=33,
            rigidify_bonds_smarts=[], rigidify_bonds_indices=[],
            double_bond_penalty=50, atom_type_smarts={},
            is_protein_sidechain=False, add_index_map=False,
            stop_at_defaults=False, remove_smiles=False):

        self.keep_nonpolar_hydrogens = keep_nonpolar_hydrogens
        self.hydrate = hydrate
        self.flexible_amides = flexible_amides
        self.rigid_macrocycles = rigid_macrocycles
        self.min_ring_size = min_ring_size
        self.max_ring_size = max_ring_size
        self.rigidify_bonds_smarts = rigidify_bonds_smarts
        self.rigidify_bonds_indices = rigidify_bonds_indices
        self.double_bond_penalty = double_bond_penalty
        self.atom_type_smarts = atom_type_smarts
        self.is_protein_sidechain = is_protein_sidechain
        self.add_index_map = add_index_map
        self.remove_smiles = remove_smiles

        if stop_at_defaults: return # create an object to show just the defaults (e.g. to argparse)

        self.setup = None
        self._atom_typer = AtomTyper(self.atom_type_smarts)
        self._bond_typer = BondTyperLegacy()
        self._macrocycle_typer = FlexMacrocycle(
                self.min_ring_size, self.max_ring_size, self.double_bond_penalty)
        self._flex_builder = FlexibilityBuilder()
        self._water_builder = HydrateMoleculeLegacy()
        self._writer = PDBQTWriterLegacy()
        self.is_ok = None
        self.log = None
        self._classes_setup = {Chem.rdchem.Mol: RDKitMoleculeSetup}
        if _has_openbabel:
            self._classes_setup[ob.OBMol] = OBMoleculeSetup

    @classmethod
    def init_just_defaults(cls):
        return cls(stop_at_defaults=True)

    @ classmethod
    def from_config(cls, config):
        expected_keys = cls.init_just_defaults().__dict__.keys()
        bad_keys = [k for k in config if k not in expected_keys]
        for key in bad_keys:
            print("ERROR: unexpected key \"%s\" in MoleculePreparation.from_config()" % key, file=sys.stderr)
        if len(bad_keys) > 0:
            raise ValueError
        p = cls(**config)
        return p

    def prepare(self, mol, is_protein_sidechain=None):
        """ if protein_sidechain, C H N O will be removed,
            root will be CA, and BEGIN/END_RES will be added.
        """
        if is_protein_sidechain is None:
            is_protein_sidechain = self.is_protein_sidechain
        mol_type = type(mol)
        if not mol_type in self._classes_setup:
            raise TypeError("Molecule is not an instance of supported types: %s" % type(mol))
        setup_class = self._classes_setup[mol_type]
        setup = setup_class(mol, is_protein_sidechain)
        self.setup = setup
        # 1.  assign atom types (including HB types, vectors and stuff)
        # DISABLED TODO self.atom_typer.set_parm(mol)
        self._atom_typer(setup)
        # 2a. add pi-model + merge_h_pi (THIS CHANGE SOME ATOM TYPES)
        # disabled
        # 2b. merge_h_classic
        if not self.keep_nonpolar_hydrogens:
            setup.merge_hydrogen()
        # 3.  assign bond types by using SMARTS...
        #     - bonds should be typed even in rings (but set as non-rotatable)
        #     - if macrocycle is selected, they will be enabled (so they must be typed already!)
        self._bond_typer(setup, self.flexible_amides, self.rigidify_bonds_smarts, self.rigidify_bonds_indices)
        # 4 . hydrate molecule
        if self.hydrate:
            self._water_builder.hydrate(setup)
        # 5.  break macrocycles into open/linear form
        if not self.rigid_macrocycles:
            # calculate possible breakable bonds
            self._macrocycle_typer.search_macrocycle(setup)
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
        if is_protein_sidechain:
            calpha_atom_index = self.get_calpha_atom_index() # 1-index
            new_setup = self._flex_builder(mol, root_atom_index=calpha_atom_index)
            new_setup.is_protein_sidechain = True
        else:
            new_setup = self._flex_builder(setup, root_atom_index=None)
        self.setup = new_setup
        # TODO re-run typing after breaking bonds
        # self.bond_typer.set_types_legacy(mol, exclude=[macrocycle_bonds])
        self.is_ok = self._check()


    def _check(self):
        # verify that all atoms have been typed
        is_ok = True
        msg = ""
        for idx in self.setup.atom_type:
            atom_type = self.setup.atom_type[idx]
            if atom_type is None:
                msg += 'atom number %d has None type, mol name: %s' % (idx, self.get_mol_name())
                is_ok = False
        self.log = msg
        return is_ok 


    def get_calpha_atom_index(self):
        """ used for preparing flexible sidechains
            requires exactly 1 atom named "CA"
            returns 1-index of CA atom
        """

        ca_atoms = [] # we want exactly 1
        for atom_idx in self.setup.get_atom_indices(true_atoms_only=True):
            pdbinfo = self.setup.get_pdbinfo(atom_idx)
            if pdbinfo.name.strip() == 'CA':
                ca_atoms.append(atom_idx)
        if len(ca_atoms) != 1:
            sys.stderr.write("ERROR: flexible residue: need exactly one 'CA' atom.\n")
            sys.stderr.write("       found %d 'CA' atoms\n" % len(ca_atoms))
            sys.stderr.write("       molecule name: %s\n" % self.setup.name)
            sys.exit(42)
        return ca_atoms[0]

    def show_setup(self):
        if self.setup is not None:
            tot_charge = 0

            print("Molecule setup\n")
            print("==============[ ATOMS ]===================================================")
            print("idx  |          coords            | charge |ign| atype    | connections")
            print("-----+----------------------------+--------+---+----------+--------------- . . . ")
            for k, v in list(self.setup.coord.items()):
                print("% 4d | % 8.3f % 8.3f % 8.3f | % 1.3f | %d" % (k, v[0], v[1], v[2],
                      self.setup.charge[k], self.setup.atom_ignore[k]),
                      "| % -8s |" % self.setup.atom_type[k],
                      self.setup.graph[k])
                tot_charge += self.setup.charge[k]
            print("-----+----------------------------+--------+---+----------+--------------- . . . ")
            print("  TOT CHARGE: %3.3f" % tot_charge)

            print("\n======[ DIRECTIONAL VECTORS ]==========")
            for k, v in list(self.setup.coord.items()):
                if k in self.setup.interaction_vector:
                    print("% 4d " % k, self.setup.atom_type[k], end=' ')

            print("\n==============[ BONDS ]================")
            # For sanity users, we won't show those keys for now
            keys_to_not_show = ['bond_order', 'type']
            for k, v in list(self.setup.bond.items()):
                t = ', '.join('%s: %s' % (i, j) for i, j in v.items() if not i in keys_to_not_show)
                print("% 8s - " % str(k), t)

            self._macrocycle_typer.show_macrocycle_scores()

            print('')

    def write_pdbqt_string(self, add_index_map=None, remove_smiles=None):
        if self.is_ok == False:
            raise RuntimeError("Molecule not OK, refusing to write PDBQT\n\nLOG:\n%s" % self.log)
        if add_index_map is None: add_index_map = self.add_index_map
        if remove_smiles is None: remove_smiles = self.remove_smiles
        if self.setup is not None:
            return self._writer.write_string(self.setup, add_index_map, remove_smiles)
        else:
            raise RuntimeError('Cannot generate PDBQT file, the molecule is not prepared.')

    def write_pdbqt_file(self, pdbqt_filename, add_index_map=None, remove_smiles=None):
        with open(pdbqt_filename,'w') as w:
            w.write(self.write_pdbqt_string(add_index_map, remove_smiles))
