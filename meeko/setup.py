#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from copy import deepcopy
from collections import defaultdict, OrderedDict
import inspect
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from .utils import rdkitutils

try:
    from openbabel import openbabel as ob
    from .utils import obutils
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True

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


# TODO update add_atom to not to specify neighbors, which should be defined only when a bond is created?
# TODO change all attributes_to_copy to have underscore ?

class MoleculeSetup:
    """ mol: molecule structurally prepared with explicit hydrogens

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
        "chiral",
        "graph",
        "bond",
        "interaction_vector",
        "rings",
        "rings_aromatic",
        "atom_to_ring_id",
        "ring_bond_breakable",
        "flexibility_model",
        'history',
        'is_protein_sidechain',
        'name',
        ]

    def __init__(self, mol, flexible_amides=False, is_protein_sidechain=False, assign_charges=True, template=None):
        """initialize a molecule template, either from scratch (template is None)
            or by using an existing setup (template is an instance of MoleculeSetup
        """
        stack = inspect.stack()
        the_class = stack[1][0].f_locals["self"].__class__.__name__
        the_method = stack[1][0].f_code.co_name
        #print("I was called by {}.{}()".format(the_class, the_method))
        #print("Setup initialized with:", mol, "template:", template)

        self.mol = mol
        self.atom_pseudo = []
        self.coord = OrderedDict()  # FIXME all OrderedDict shuold be converted to lists?
        self.charge = OrderedDict()
        self.pdbinfo = OrderedDict()
        self.atom_type = OrderedDict()
        self.atom_ignore = OrderedDict()
        self.chiral = OrderedDict()
        self.atom_true_count = 0
        self.graph = OrderedDict()
        self.bond = OrderedDict()
        self.element = OrderedDict()
        self.interaction_vector = OrderedDict()
        self.flexibility_model = {}
        # ring information
        self.rings = {}
        self.rings_aromatic = []
        self.atom_to_ring_id = defaultdict(list)
        self.ring_bond_breakable = defaultdict(dict)  # used by flexible macrocycles
        self.ring_corners = {}  # used to store corner flexibility
        self.name = None
        # this could be used to keep track of transformations? (corner flipping)
        self.history = []
        self.is_protein_sidechain = False
        if template is None:
            self.process_mol(flexible_amides, is_protein_sidechain, assign_charges)
        else:
            if not isinstance(template, MoleculeSetup):
                raise TypeError('FATAL: template must be an instance of MoleculeSetup')
            self.copy_attributes_from(template)

    def process_mol(self, flexible_amides, is_protein_sidechain, assign_charges):
        self.atom_true_count = self.get_num_mol_atoms()
        self.name = self.get_mol_name()
        self.init_atom(assign_charges)
        self.perceive_rings()
        self.init_bond(flexible_amides)
        if is_protein_sidechain:
            self.ignore_backbone()
        return

    def add_atom(self, idx=None, coord=np.array([0.0, 0.0,0.0], dtype='float'),
            element=None, charge=0.0, atom_type=None, pdbinfo=None, neighbors=None,
            ignore=False, chiral=False, overwrite=False):
        """ function to add all atom information at once;
            every property is optional
        """
        if idx is None:
            idx = len(self.coord)
        if idx in self.coord and not overwrite:
            print("ADD_ATOM> Error: the idx [%d] is already occupied (use 'overwrite' to force)")
            return False
        self.set_coord(idx, coord)
        self.set_charge(idx, charge)
        self.set_element(idx, element)
        self.set_atom_type(idx, atom_type)
        self.set_pdbinfo(idx, pdbinfo)
        if neighbors is None:
            neighbors = []
        self.set_neigh(idx, neighbors)
        self.set_chiral(idx, chiral)
        self.set_ignore(idx, ignore)
        return idx

    def del_atom(self, idx):
        """ remove an atom and update all data associate with it """
        pass
        # coords
        # charge
        # element
        # type
        # neighbor graph
        # chiral
        # ignore
        # update bonds bonds (using the neighbor graph)
        # If pseudo-atom, update other information, too


    # pseudo-atoms
    def add_pseudo(self, coord=np.array([0.0,0.0,0.0], dtype='float'), charge=0.0,
            anchor_list=None, atom_type=None, bond_type=None, rotatable=False,
            pdbinfo=None, directional_vectors=None, ignore=False, chira0=False, overwrite=False):
        """ add a new pseudoatom
            multiple bonds can be specified in "anchor_list" to support the centroids of aromatic rings

            if rotatable, makes the anchor atom rotatable to allow the pseudoatom movement
        """
        idx = self.atom_true_count + len(self.atom_pseudo)
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
                neighbors=[],
                ignore=ignore,
                overwrite=overwrite)
        # anchor atoms
        if not anchor_list is None:
            for anchor in anchor_list:
                self.add_bond(idx, anchor, 0, rotatable, bond_type=bond_type)
        # directional vectors
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

    def set_chiral(self, idx, chiral):
        """ set chiral flag for atom """
        self.chiral[idx] = chiral

    def get_chiral(self, idx):
        """ get chiral flag for atom """
        return self.chiral[idx]

    def get_ignore(self, idx):
        """ return if the atom is ignored"""
        return bool(self.atom_ignore[idx])

    def is_aromatic(self, idx):
        """ check if atom is aromatic """
        for r in self.rings_aromatic:
            if idx in r:
                return True
        return False

    def set_element(self, idx, elem_num):
        """ set the atomic number of the atom idx"""
        self.element[idx] = elem_num

    def get_element(self, idx):
        """ return the atomic number of the atom idx"""
        return self.element[idx]

    # def get_atom_ring_count(self, idx):
    #     """ return the number of rings to which this atom belongs"""
    #     # FIXME this should be replaced by self.get_atom_rings()
    #     return len(self.atom_to_ring_id[idx])

    def get_atom_rings(self, idx):
        # FIXME this should replace self.get_atom_ring_count()
        """ return the list of rings to which the atom idx belongs"""
        if idx in self.atom_to_ring_id:
            return self.atom_to_ring_id[idx]
        return []

    def get_atom_indices(self, true_atoms_only=False):
        """ return the indices of the atoms registered in the setup
            if 'true_atoms_only' are requested, then pseudoatoms are ignored
        """
        indices = list(self.coord.keys())
        if true_atoms_only:
            return [ x for x in indices if not x in self.atom_pseudo ]
        return indices

    # interaction vectors
    def add_interaction_vector(self, idx, vector_list):
        """ add vector list to list of directional interaction vectors for atom idx"""
        if not idx in self.interaction_vector:
            self.interaction_vector[idx] = []
        for vec in vector_list:
            self.interaction_vector[idx].append(vec)

    # TODO evaluate if useful
    def _get_attrib(self, idx, attrib, default=None):
        """ generic function to provide a default for retrieving properties and returning standard values """
        return getattr(self, attrib).get(idx, default)

    def get_interaction_vector(self, idx):
        """ get list of directional interaction vectors for atom idx"""
        return self.interaction_vector[idx]

    def del_interaction_vector(self, idx):
        """ delete list of directional interaction vectors for atom idx"""
        del self.interaction_vector[idx]

    def set_pdbinfo(self, idx, data):
        """ add PDB data (resname/num, atom name, etc.) to the atom """
        self.pdbinfo[idx] = data

    def get_pdbinfo(self, idx):
        """ retrieve PDB data (resname/num, atom name, etc.) to the atom """
        return self.pdbinfo[idx]

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

    # replaced by
    # def ring_atom_to_ring(self, arg):
    #     return self.atom_to_ring_id[arg]

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

    def copy_attributes_from(self, template):
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

    def copy(self):
        """ return a copy of the current setup"""
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def init_atom(self):
        """ iterate through molecule atoms and build the atoms table """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def perceive_rings(self):
        """ populate the rings and aromatic rings tableshe atoms table:
        self.rings_aromatics : list
            contains the list of ring_id items that are aromatic
        self.rings: dict
            store information about rings, like if they have corners that can
            be flipped and the graph of atoms that belong to them:

                self.rings[ring_id] = {
                                'corner_flip': False
                                'graph': {}
                                }

            The atom is built using the `walk_recursive` method
        self.ring_atom_to_ring_id: dict
            mapping of each atom belonginig to the ring: atom_idx -> ring_id
        """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def init_bond(self, flexible_amides):
        """ iterate through molecule bonds and build the bond table (id, table)
            INPUT
                flexible_amides: bool
                    this flag is used to define if primary and secondary amides
                    need to be considered flexible or not (bond_order= 999)
            CALCULATE
                bond_order: int
                    0       : pseudo-bond (for pseudoatoms)
                    1,2,3   : single-triple bond
                    5       : aromatic
                    999     : rigid

                if bond is in ring (both start and end atom indices in the bond are in the same ring)

            SETUP OPERATION
                Setup.add_bond(idx1, idx2, order, in_rings=[])
        """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def ignore_backbone(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_mol_name(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def find_pattern(self, smarts):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_smiles_and_order(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")


class RDKitMoleculeSetup(MoleculeSetup):

    def get_smiles_and_order(self):

        # 3D SDF files written by other toolkits (OEChem, ChemAxon)
        # seem to not include the chiral flag in the bonds block, only in
        # the atoms block. RDKit ignores the atoms chiral flag as per the
        # spec. When reading SDF (e.g. from PubChem/PDB),
        # we may need to have RDKit assign stereo from coordinates, see:
        # https://sourceforge.net/p/rdkit/mailman/message/34399371/
        mol_noH = Chem.RemoveHs(self.mol) # imines (=NH) may become chiral
        atomic_num_mol_noH = [atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
        noH_to_H = []
        num_H_in_noH = 0 # e.g. stereo imines [H]/N=C keep [H] after RemoveHs()
        for (index, atom) in enumerate(self.mol.GetAtoms()):
            if atom.GetAtomicNum() == 1: continue
            for i in range(len(noH_to_H), len(atomic_num_mol_noH)):
                if atomic_num_mol_noH[i] > 1: break
                noH_to_H.append('H')
            noH_to_H.append(index)
        extra_hydrogens = len(atomic_num_mol_noH) - len(noH_to_H)
        if extra_hydrogens > 0:
            assert(set(atomic_num_mol_noH[len(noH_to_H):]) == {1})
            noH_to_H.extend(['H'] * extra_hydrogens)

        # map indices of explicit hydrogens, e.g. stereo imine [H]/N=C
        for index in range(len(noH_to_H)):
            if noH_to_H[index] != 'H': continue
            h_atom = mol_noH.GetAtomWithIdx(index)
            assert(h_atom.GetAtomicNum() == 1)
            parents = h_atom.GetNeighbors()
            assert(len(parents) == 1)
            num_h_in_parent = len([a for a in parents[0].GetNeighbors() if a.GetAtomicNum() == 1])
            if num_h_in_parent != 1:
                msg = "Can't handle %d explicit H for each heavy atomin noH mol.\n" % num_h_in_parent
                msg += "Was expecting only imines [H]N=\n"
                raise RuntimeError(msg)
            parent_index_in_mol_with_H = noH_to_H[parents[0].GetIdx()]
            parent_in_mol_with_H = self.mol.GetAtomWithIdx(parent_index_in_mol_with_H)
            h_in_mol_with_H = [a for a in parent_in_mol_with_H.GetNeighbors() if a.GetAtomicNum() == 1]
            if len(h_in_mol_with_H) != 1:
                msg = "Can't handle %d explicit H for each heavy atomin noH mol.\n" % len(h_in_mol_with_H)
                msg += "Was expecting only imines [H]N=\n"
                raise RuntimeError(msg)
            noH_to_H[index] = h_in_mol_with_H[0].GetIdx()

        smiles = Chem.MolToSmiles(mol_noH)
        order_string = mol_noH.GetProp("_smilesAtomOutputOrder")
        order_string = order_string.replace(',]', ']') # remove trailing comma
        order = json.loads(order_string) # mol_noH to smiles
        order = list(np.argsort(order))
        order = {noH_to_H[i]: order[i]+1 for i in range(len(order))} # 1-index
        return smiles, order

    def find_pattern(self, smarts):
        p = Chem.MolFromSmarts(smarts)
        return self.mol.GetSubstructMatches(p)

    def get_mol_name(self):
        if self.mol.HasProp("_Name"):
            return self.mol.GetProp("_Name")
        else:
            return None

    def get_num_mol_atoms(self):
        return self.mol.GetNumAtoms()

    def get_equivalent_atoms(self):
       return list(Chem.CanonicalRankAtoms(self.mol, breakTies=False))

    def init_atom(self, assign_charges):
        """ initialize the atom table information """
        # extract the coordinates
        c = self.mol.GetConformers()[0]
        coords = c.GetPositions()
        # extract/generate charges
        if assign_charges:
            rdPartialCharges.ComputeGasteigerCharges(self.mol)
            charges = [a.GetDoubleProp('_GasteigerCharge') for a in self.mol.GetAtoms()]
        else:
            charges = [0.0] * self.mol.GetNumAtoms()
        # perceive chirality
        # TODO check consistency for chiral model between OB and RDKit
        chiral_info = {}
        for data in Chem.FindMolChiralCenters(self.mol, includeUnassigned=True):
            chiral_info[data[0]] = data[1]
        # register atom
        for a in self.mol.GetAtoms():
            idx = a.GetIdx()
            chiral = False
            if idx in chiral_info:
                chiral = chiral_info[idx]
            self.add_atom(idx,
                    coord=coords[idx],
                    element=a.GetAtomicNum(),
                    charge=charges[idx],
                    atom_type=None,
                    pdbinfo = rdkitutils.getPdbInfoNoNull(a),
                    neighbors = [n.GetIdx() for n in a.GetNeighbors()],
                    chiral=False,
                    ignore=False)

    def init_bond(self, flexible_amides):
        """ initialize bond information """
        amide_smarts = "[$([NX3][#1])][$([CX3](=[OX1])[#6])]" # does not match tertiary amides
        amide_bonds = [set(f) for f in self.find_pattern(amide_smarts)]
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = int(b.GetBondType())
            # fix the RDKit aromatic type (FIXME)
            if bond_order == 12: # aromatic
                bond_order = 5
            if bond_order == 1:
                rotatable = True
                if flexible_amides:
                    continue
                # check if bond is a tertiary amide bond
                if set((idx1, idx2)) in amide_bonds:
                    # TODO check what is the best to keep here (rotatable or bond_order)
                    rotatable=False
                    bond_order = 999
            else:
                rotatable = False
            idx1_rings = set(self.get_atom_rings(idx1))
            idx2_rings = set(self.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.add_bond(idx1, idx2, order=bond_order, rotatable=rotatable, in_rings=in_rings)


    def perceive_rings(self):
        """ perceive ring information """

        def isRingAromatic(bonds_in_ring):
            """ Index ID#: RDKitCB_8
            RDKit recipe for identifying if ring is aromatic
            """
            for bond_idx in bonds_in_ring:
                if not self.mol.GetBondWithIdx(bond_idx).GetIsAromatic():
                    return False
            return True

        ring_info = self.mol.GetRingInfo()
        perceived = ring_info.AtomRings()
        bond_rings = ring_info.BondRings()
        for idx, ring_id in enumerate(perceived):
            if isRingAromatic(bond_rings[idx]):
                self.rings_aromatic.append(ring_id)
            self.rings[ring_id] = {'corner_flip':False}
            graph = {}
            for member in ring_id:
                # atom to ring lookup
                self.atom_to_ring_id[member].append(ring_id)
                # graph of atoms affected by potential ring movements
                graph[member] = self.walk_recursive(member, collected=[], exclude=list(ring_id))
            self.rings[ring_id]['graph'] = graph

    def copy(self):
        """ return a copy of the current setup"""
        return RDKitMoleculeSetup(self.mol, template=self)


class OBMoleculeSetup(MoleculeSetup):

    def get_mol_name(self):
        return self.mol.GetTitle()

    def find_pattern(self, smarts):
        obsmarts = ob.OBSmartsPattern()
        obsmarts.Init(smarts)
        found = obsmarts.Match(self.mol)
        output = []
        if found:
            for x in obsmarts.GetUMapList():
                output.append([y-1 for y in x])
        return output

    def get_num_mol_atoms(self):
        return self.mol.NumAtoms()

    def get_equivalent_atoms(self):
        raise NotImplementedError 
    
    def init_atom(self, assign_charges):
        """initialize atom data table"""
        for a in ob.OBMolAtomIter(self.mol):
            # TODO fix this to be zero-based?
            partial_charge = a.GetPartialCharge() * float(assign_charges)
            self.add_atom(a.GetIdx() - 1,
                    coord=np.asarray(obutils.getAtomCoords(a), dtype='float'),
                    element=a.GetAtomicNum(),
                    charge=partial_charge,
                    atom_type=None,
                    pdbinfo = obutils.getPdbInfoNoNull(a),
                    neighbors=[x.GetIdx() - 1 for x in ob.OBAtomAtomIter(a)],
                    ignore=False, chiral=a.IsChiral())
            # TODO check consistency for chiral model between OB and RDKit

    def perceive_rings(self):
        """ collect information about rings"""
        perceived = self.mol.GetSSSR()
        for r in perceived:
            ring_id = tuple(i-1 for i in tuple(r._path))
            if r.IsAromatic():
                self.rings_aromatic.append(ring_id)
            self.rings[ring_id] = {'corner_flip': False}
            graph = {}
            for member in ring_id:
                # atom to ring lookup
                self.atom_to_ring_id[member].append(ring_id)
                # graph of atoms affected by potential ring movements
                graph[member] = self.walk_recursive(member, collected=[], exclude=list(ring_id))
            self.rings[ring_id]['graph'] = graph

    def init_bond(self, flexible_amides):
        """initialize bond data table"""
        for b in ob.OBMolBondIter(self.mol):
            idx1 = b.GetBeginAtomIdx() - 1
            idx2 = b.GetEndAtomIdx() - 1
            bond_order = b.GetBondOrder()
            if b.IsAromatic():
                bond_order = 5
            if b.IsAmide() and not b.IsTertiaryAmide():
                if not flexible_amides:
                    bond_order = 999
            # check if bond is a ring bond, i.e., both atoms belongs to the same ring
            idx1_rings = set(self.get_atom_rings(idx1))
            idx2_rings = set(self.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.add_bond(idx1, idx2, order=bond_order, in_rings=in_rings)

    def ignore_backbone(self):
        """ set ignore for PDB atom names 'C', 'N', 'H', and 'O'
            these atoms are kept in the rigid PDBQT by ReactiveReceptor"""
        # TODO this function is very fragile, to be replaced by SMARTS
        # also, not sure where it is used...
        exclude_pdbname = {'C': 0, 'N': 0, 'H': 0, 'O': 0} # store counts of found atoms
        for atom in ob.OBMolAtomIter(self.mol):
            pdbinfo = obutils.getPdbInfo(atom)
            if pdbinfo.name.strip() in exclude_pdbname:
                idx = atom.GetIdx() - 1
                self.set_ignore(idx, True)
                exclude_pdbname[pdbinfo.name.strip()] += 1
        for name in exclude_pdbname:
            n_found = exclude_pdbname[name]
            if n_found != 1:
                print("Warning: expected 1 atom with PDB name '%s' but found %d" % (name, n_found))

    def copy(self):
        """ return a copy of the current setup"""
        return OBMoleculeSetup(self.mol, template=self)
