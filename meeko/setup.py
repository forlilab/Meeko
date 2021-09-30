#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np
from openbabel import openbabel as ob

from .utils import obutils


# based on the assumption we are using OpenBabel

# TODO modify so that there are no more dictionaries and only list/arrays (fix me)
# methods like add_x,del_x,get_x will deal with indexing (fix me)
# the goal is to remove dictionaries (fix me)

# (fix me)) provide infrastructure to calculate "differential" modifications,
# like tautomers: the class calculating tautomers will copy and modify the setup
# to swap protons and change the atom types (and possibly rotations?), then
# there will e a function to extract only differences (i.e., atom types and coordinates,
# in the case of tautomers) and store them to make it a switch-state

# TODO use only 1D arrays (C)

class MoleculeSetup(object):
    """ mol: molecule structurally prepared:
                    - hydrogens
                    - partial charges

        the setup provides:
            - data storage
            - SMARTS matcher for all atom typers
    """
    # possibly useless here
    attributes_to_copy = [
        "atom_pseudo",
        "atom_ignore",
        "atom_type",
        "atom_true_count",
        "atom_pseudo",
        "element",
        "coord",
        "charge",
        "pdbinfo",
        "graph",
        "bond",
        "interaction_vector",
        "rings",
        "rings_aromatic",
        "ring_atom_to_ring_id",
        "ring_bond_breakable",
        "flexibility_model",
        'history',
        'is_protein_sidechain',
        ]

    def __init__(self, obmol, template=None, flexible_amides=False, is_protein_sidechain=False):
        """initialize a molecule template, either from scratch (template is None)
            or by using an existing setup (template is an instance of MoleculeSetup
        """
        if not isinstance(obmol, ob.OBMol):
            raise TypeError('Input molecule must be an OBMol but is %s' % type(obmol))

        # TODO might be useless, a counter (int) might be enough
        self.atom_pseudo = []
        self.coord = OrderedDict()  # FIXME all OrderedDict shuold be converted to lists?
        self.charge = OrderedDict()
        self.pdbinfo = OrderedDict()
        self.atom_type = OrderedDict()
        self.atom_ignore = OrderedDict()
        self.atom_true_count = 0
        self.graph = OrderedDict()
        self.bond = OrderedDict()
        self.element = OrderedDict()
        self.interaction_vector = OrderedDict()
        self.flexibility_model = {}
        # ring information
        self.rings = {}
        self.rings_aromatic = []
        self.ring_atom_to_ring_id = defaultdict(list)
        self.ring_bond_breakable = defaultdict(dict)  # used by flexible macrocycles
        self.ring_corners = {}  # used to store corner flexibility
        # this could be used to keep track of transformations? (corner flipping)
        self.history = []
        self.is_protein_sidechain = False

        if template is None:
            self.init_from_obmol(obmol, flexible_amides, is_protein_sidechain)
        elif isinstance(template, MoleculeSetup):
            self.init_from_template(template)
        else:
            raise TypeError('FATAL: template must be an instance of MoleculeSetup')

    def copy(self):
        """ return a copy of the current setup"""
        return MoleculeSetup(self.mol, template=self)

    # atom types
    def set_atom_type(self, idx, atom_type):
        """ set the atom type for atom index

        atom_type : int or str?

        return: void
        """
        self.atom_type[idx] = atom_type

    def get_atom_type(self, idx):
        """ return the atom type for atom index in the lookup table

        idx : int

        return: str
        """
        return self.atom_type[idx]

    # ignore flag
    def set_ignore(self, idx, state):
        """ set the ignore flag (bool) for the atom
        (used formerged hydrogen)

        idx: int
        state: bool

        """
        self.atom_ignore[idx] = state

    # charge
    def get_charge(self, idx):
        """ return partial charge for atom index
        idx: int

        """
        return self.charge[idx]

    def set_charge(self, idx, charge):
        """ set partial charge"""
        self.charge[idx] = charge

    def get_coord(self, idx):
        """ return coordinates of atom index"""
        return self.coord[idx]

    def set_coord(self, idx, coord):
        """ define coordinates of atom index"""
        self.coord[idx] = coord

    def get_neigh(self, idx):
        """ return atoms connected to atom index"""
        return self.graph[idx]

    def get_ignore(self, idx):
        """ return if the atom is ignored"""
        return bool(self.atom_ignore[idx])

    def set_element(self, idx, elem_num):
        """ set the atomic number of the atom idx"""
        self.element[idx] = elem_num

    def get_element(self, idx):
        """ return the atomic number of the atom idx"""
        return self.element[idx]

    def get_atom_ring_count(self, idx):
        """ return the number of rings to which this atom belongs"""
        return len(self.ring_atom_to_ring_id[idx])

    # interaction vectors
    def add_interaction_vector(self, idx, vector_list):
        """ add vector list to list of directional interaction vectors for atom idx"""
        if not idx in self.interaction_vector:
            self.interaction_vector[idx] = []
        for vec in vector_list:
            self.interaction_vector[idx].append(vec)

    def get_interaction_vector(self, idx):
        """ get list of directional interaction vectors for atom idx"""
        return self.interaction_vector[idx]

    def del_interaction_vector(self, idx):
        """ delete list of directional interaction vectors for atom idx"""
        del self.interaction_vector[idx]

    # Pseudo-atoms
    def add_pseudo(self, coord, charge, anchor_list, atom_type, bond_type, rotatable, pdbinfo=None, directional_vectors=None, ignore=False):
        """ add data about pseudoatom
            multiple bonds can be specified in "anchor_list" to support the centroids of aromatic rings

            if rotatable, makes the anchor atom rotatable to allow the pseudoatom movement
        """
        idx = self.atom_true_count + len(self.atom_pseudo) + 1
        self.atom_pseudo.append(idx)
        self.coord[idx] = coord
        self.charge[idx] = charge
        self.pdbinfo[idx] = pdbinfo
        self.atom_type[idx] = atom_type
        self.graph[idx] = []
        self.element[idx] = 0
        for anchor in anchor_list:
            self.add_bond(idx, anchor, 0, rotatable, bond_type=bond_type)
        if not directional_vectors is None:
            self.add_interaction_vector(idx, directional_vectors)
        self.set_ignore(idx, ignore)
        return idx
    
    # Bonds
    def add_bond(self, idx1, idx2, order=0, rotatable=False, in_rings=None, bond_type=None):
        """ bond_type default: 0 (non rotatable) """
        # NOTE: in_ring is used during bond typing to keep endo-cyclic rotatable bonds (e.g., sp3)
        #       as non-rotatable. Possibly, this might be handled by the bond typer itself?
        #       the type would allow it
        if in_rings is None:
            in_rings = []
        if not idx2 in self.graph[idx1]:
            self.graph[idx1].append(idx2)
        if not idx1 in self.graph[idx2]:
            self.graph[idx2].append(idx1)
        # bond_id = self.get_bond_id(idx1, idx2)
        # self.bond[bond_id] = {}
        self.set_bond(idx1, idx2, order, rotatable, in_rings, bond_type)

    def set_bond(self, idx1, idx2, order=None, rotatable=None, in_rings=None, bond_type=None):
        """ populate bond lookup table with properties
            bonds are identified by any tuple of atom indices
            the function generates the canonical bond id

            order      : int
            rotatable  : bool
            in_rings   : list (rings to which the bond belongs)
            bond_type  : int

        """
        bond_id = self.get_bond_id(idx1, idx2)
        if order is None:
            order = 0
        if rotatable is None:
            rotatable = False
        if in_rings is None:
            in_rings = []
        # if bond_type==None:
        #     bond_type=None
        self.bond[bond_id] = {'bond_order': order,
                              'type': bond_type,
                              'rotatable': rotatable,
                              'in_rings': in_rings}

    def del_bond(self, idx1, idx2):
        """ remove a bond from the lookup table """
        bond_id = self.get_bond_id(idx1, idx2)
        del self.bond[bond_id]
        self.graph[idx1].remove(idx2)
        if not self.graph[idx1]:
            del self.graph[idx1]
        self.graph[idx2].remove(idx1)
        if not self.graph[idx2]:
            del self.graph[idx2]

    def get_bond(self, idx1, idx2):
        """ return properties of a bond in the lookup table
            if the bond does not exist, None is returned

            idx1, idx2 : int

            return: dict or voidko
        """
        bond_idx = self.get_bond_id(idx1, idx2)
        try:
            return self.bond[bond_idx]
        except IndexError:
            return None

    @staticmethod
    def get_bond_id(idx1, idx2):
        """ used to generate canonical bond id from a pair of nodes in the graph"""
        idx_min = min(idx1, idx2)
        idx_max = max(idx1, idx2)
        return (idx_min, idx_max)

    def is_methyl(self, atom_idx):
        """ identify methyl groups (to be done with SMARTS)"""
        atom = self.mol.GetAtom(atom_idx)
        if not (atom.GetAtomicNum() == 6):
            return False
        h_count = len([x for x in ob.OBAtomAtomIter(atom) if x.GetAtomicNum() == 1])
        return h_count == 3

    def init_atom(self):
        """initialize atom data table"""
        for a in ob.OBMolAtomIter(self.mol):
            idx = a.GetIdx()
            self.coord[idx] = np.asarray(obutils.getAtomCoords(a), dtype='float')
            self.charge[idx] = a.GetPartialCharge()
            self.pdbinfo[idx] = obutils.getPdbInfoNoNull(a)
            self.atom_type[idx] = None
            self.graph[idx] = [x.GetIdx() for x in ob.OBAtomAtomIter(a)]
            self.element[idx] = a.GetAtomicNum()
            # by default all atoms are considered
            self.set_ignore(idx, False)

    def ring_atom_to_ring(self, arg):
        #print("RACCOON core/docking/setup.py  UPDATE YOUR CODE")
        #print("CALLING", self.ring_atom_to_ring_id, self.ring_atom_to_ring_id[arg])
        return self.ring_atom_to_ring_id[arg]

    def walk_recursive(self, idx, collected=None, exclude=None):
        """ walk molecular graph and return subgraphs that are bond-connected"""
        if collected is None:
            collected = []
        if exclude is None:
            exclude = []
        for neigh in self.get_neigh(idx):
            if neigh in exclude:
                continue
            collected.append(neigh)
            exclude.append(neigh)
            self.walk_recursive(neigh, collected, exclude)
        return collected

    def perceive_rings(self):
        """ collect information about rings"""
        perceived = self.mol.GetSSSR()
        for r in perceived:
            ring_id = tuple(r._path)
            if r.IsAromatic():
                self.rings_aromatic.append(ring_id)
            self.rings[ring_id] = {'corner_flip': False}
            graph = {}
            for member in ring_id:
                # atom to ring lookup
                self.ring_atom_to_ring_id[member].append(ring_id)
                # graph of atoms affected by potentia ring movements
                graph[member] = self.walk_recursive(member, collected=[], exclude=list(ring_id))
            self.rings[ring_id]['graph'] = graph
    
    def init_bond(self, flexible_amides):
        """initialize bond data table"""
        for b in ob.OBMolBondIter(self.mol):
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = b.GetBondOrder()
            if b.IsAromatic():
                bond_order = 5
            if b.IsAmide() and not b.IsTertiaryAmide():
                if not flexible_amides:
                    bond_order = 999
            # check if bond is a ring bond, i.e., both atoms belongs to the same ring
            # TODO make this a single int value
            in_rings = []
            if idx1 in self.ring_atom_to_ring_id:
                in_rings.append(self.ring_atom_to_ring_id[idx1])
            # if not both atoms are in a ring, or not in the same ring, not a ring bond
            test1 = (not idx2 in self.ring_atom_to_ring_id)
            test2 = (not self.ring_atom_to_ring_id[idx2] in in_rings)
            if test1 or test2:
                in_rings = []
            self.add_bond(idx1, idx2, order=bond_order, in_rings=in_rings)

    def init_from_obmol(self, obmol, flexible_amides=False, is_protein_sidechain=False):
        """generate a new molecule setup

            NOTE: OpenBabel uses 1-based index
        """
        # storing info
        self.mol = obmol
        self.smarts = obutils.SMARTSmatcher(self.mol)
        self.atom_true_count = self.mol.NumAtoms()
        self.init_atom()
        self.perceive_rings()
        self.init_bond(flexible_amides)
        self.mol.setup = self
        if is_protein_sidechain:
            self.ignore_backbone()
    
    def init_from_template(self, template):
        """ copy attributes to duplicate the template setup
        NOTE: the molecule will always keep the original setup (i.e., template)"""
        # TODO enable some kind of plugin system here too, to allow other objects to
        # add attributes?
        # TODO although, this would make the setup more fragile-> better have attributes
        # explicitely added here, and that's it
        for attr in self.attributes_to_copy:
            attr_copy = deepcopy(getattr(template, attr))
            setattr(self, attr, attr_copy)
        # TODO possible BUG? the molecule is shared by the different setups
        #      if one of them alters the molecule, properties will not be the same
        self.mol = template.mol
        self.smarts = template.smarts

    def merge_hydrogen(self):
        """ standard method to merge hydrogens bound to carbons"""
        merged = 0

        for a, neigh_list in list(self.graph.items()):
            # look for hydrogens
            if not self.get_element(a) == 1:
                continue

            hydrogen_charge = self.get_charge(a)

            for n in neigh_list:
                # look for carbons
                if self.get_element(n) == 6:
                    merged += 1
                    carbon_charge = self.get_charge(n)
                    # carbon adsorbs the final charge
                    self.set_charge(n, carbon_charge + hydrogen_charge)
                    # flag hydrogen to be ignored and set its charge to 0
                    self.set_charge(a, 0)
                    self.set_ignore(a, True)


    def ignore_backbone(self):
        """ set ignore for PDB atom names 'C', 'N', 'H', and 'O'
            these atoms are kept in the rigid PDBQT by ReactiveReceptor"""

        exclude_pdbname = {'C': 0, 'N': 0, 'H': 0, 'O': 0} # store counts of found atoms
        for atom in ob.OBMolAtomIter(self.mol):
            pdbinfo = obutils.getPdbInfo(atom)
            if pdbinfo.name.strip() in exclude_pdbname:
                idx = atom.GetIdx()
                self.set_ignore(idx, True)
                exclude_pdbname[pdbinfo.name.strip()] += 1
        for name in exclude_pdbname:
            n_found = exclude_pdbname[name]
            if n_found != 1:
                print("Warning: expected 1 atom with PDB name '%s' but found %d" % (name, n_found))
