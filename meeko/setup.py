#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from copy import deepcopy
from collections import defaultdict, OrderedDict

import numpy as np

try:
    from .utils import obutils
except:
    from meeko.utils import obutils


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

    def __init__(self, mol, template=None):
        """initialize a molecule template, either from scratch (template is None)
            or by using an existing setup (template is an instance of MoleculeSetup
        """
        # TODO might be useless, a counter (int) might be enough
        self.mol = mol
        self.mol.setup = self
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
        if isinstance(template, MoleculeSetup):
            self.init_from_template(template)
        # else:
        #     raise TypeError('FATAL: template must be an instance of MoleculeSetup')

    def copy(self):
        """ return a copy of the current setup"""
        return MoleculeSetup(self.mol, template=self)


    def add_atom(self, idx=None, coord=np.array([0.0, 0.0,0.0], dtype='float'), element=None, charge=0.0, atom_type=None, pdbinfo=None, neighbors=None, ignore=False, overwrite=False):
        """ function to add all atom information at once """
        if idx is None:
            idx = len(self.coord)
        if idx in self.coord and not overwrite:
            print("ADD_ATOM> Error: the idx [%d] is already occupied (use 'overwrite' to force)")
            return False
        self.set_coord(idx, coord)
        self.set_charge(idx, charge)
        self.set_element(idx, element)
        self.set_atom_type(idx, atom_type)
        if neighbors is None:
            neighbors = []
        self.set_neigh(idx, neighbors)
        self.set_ignore(idx, ignore)
        return idx

    # pseudo-atoms
    def add_pseudo(self, coord=np.array([0.0,0.0,0.0], dtype='float'), charge=0.0, anchor_list=None, atom_type=None, bond_type=None, rotatable=False, pdbinfo=None, directional_vectors=None, ignore=False, overwrite=False):
        """ add a new pseudoatom
            multiple bonds can be specified in "anchor_list" to support the centroids of aromatic rings

            if rotatable, makes the anchor atom rotatable to allow the pseudoatom movement
        """
        idx = self.atom_true_count + len(self.atom_pseudo) + 1
        if idx in self.coord and not overwrite:
            print("ADD_PSEUDO> Error: the idx [%d] is already occupied (use 'overwrite' to force)")
            return False
        self.atom_pseudo.append(idx)
        # add the pseudoatom information to the atoms
        self.add_atom(idx=idx, coord=coord, 
                element=0,
                charge=charge, 
                atom_type=atom_type,
                pdbinfo=pdbinfo, 
                neighbors=neighbors,
                ignore=ignore,
                overwrite=overwrite)
        # self.coord[idx] = coord
        # self.charge[idx] = charge
        # self.pdbinfo[idx] = pdbinfo
        # self.atom_type[idx] = atom_type
        # self.graph[idx] = []
        # self.element[idx] = 0
        # self.set_ignore(idx, ignore)
        if not anchor_list is None:
            for anchor in anchor_list:
                self.add_bond(idx, anchor, 0, rotatable, bond_type=bond_type)
        if not directional_vectors is None:
            self.add_interaction_vector(idx, directional_vectors)
        return idx
    
    # Bonds
    def add_bond(self, idx1, idx2, order=0, rotatable=False, in_rings=None, bond_type=None):
        """ bond_type default: 0 (non rotatable) """
        # NOTE: in_ring is used during bond typing to keep endo-cyclic rotatable bonds (e.g., sp3)
        #       as non-rotatable. Possibly, this might be handled by the bond typer itself?
        #       the type would allow it
        # TODO check if in_rings should be checked by this function?
        if in_rings is None:
            in_rings = []
        if not idx2 in self.graph[idx1]:
            self.graph[idx1].append(idx2)
        if not idx1 in self.graph[idx2]:
            self.graph[idx2].append(idx1)
        self.set_bond(idx1, idx2, order, rotatable, in_rings, bond_type)

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

    def set_neigh(self, idx, neigh_list):
        """ update the molecular graph with the neighbor indices provided """
        if not idx in self.graph:
            self.graph[idx] = []
        for n in neigh_list:
            if not n in self.graph[idx]:
                self.graph[idx].append(n)
            if not n in self.graph:
                self.graph[n] = []
            if not idx in self.graph[n]:
                self.graph[n].append(idx)

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
        # FIXME this should be replaced by self.get_atom_rings()
        return len(self.ring_atom_to_ring_id[idx])

    def get_atom_rings(self, idx):
        # FIXME this should replace self.get_atom_ring_count()
        """ return the list of rings to which the atom idx belongs"""
        if idx in self.ring_atom_to_ring_id:
            return self.ring_atom_to_ring_id[idx]
        return []

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

    def set_pdbinfo(self, idx, data):
        """ add PDB data (resname/num, atom name, etc.) to the atom """
        self.pdbinfo[idx] = data


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
        # TODO check if we want to delete nodes that have no connections (we might want to keep them)
        # if not self.graph[idx1]:
        #     del self.graph[idx1]
        self.graph[idx2].remove(idx1)
        # if not self.graph[idx2]:
        #     del self.graph[idx2]

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



