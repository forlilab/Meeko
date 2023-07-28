#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from copy import deepcopy
from collections import defaultdict, OrderedDict
import json
import warnings
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from .utils import rdkitutils
from .utils import utils

try:
    from openbabel import openbabel as ob
    from .utils import obutils
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True


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

    def __init__(self):

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
        self.ring_closure_info = {
            "bonds_removed": [],
            "pseudos_by_atom": {},
        }
        # ring information
        self.rings = {}
        self.rings_aromatic = []
        self.atom_to_ring_id = defaultdict(list)
        self.ring_corners = {}  # used to store corner flexibility
        self.name = None

    def copy(self):
        newsetup = MoleculeSetup()
        newsetup.__dict__ = deepcopy(self.__dict__)
        return newsetup

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

    def get_bonds_in_ring(self, ring):
        """ input: 'ring' (list of atom indices)
            returns list of bonds in ring, each bond is a pair of atom indices
        """
        n = len(ring)
        bonds = []
        for i in range(n):
            bond = (ring[i], ring[(i+1) % n])
            bond = (min(bond), max(bond))
            bonds.append(bond)
        return bonds

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

    def merge_terminal_atoms(self, indices):
        """for merging hydrogens, but will merge any atom or pseudo atom
            that is bonded to only one other atom"""

        for index in indices:
            if len(self.graph[index]) != 1:
                msg = "Atempted to merge atom %d with %d neighbors. "
                msg += "Only atoms with one neighbor can be merged."
                msg = msg % (index + 1, len(self.graph[index]))
                raise RuntimeError(msg)
            neighbor_index = self.graph[index][0]
            self.charge[neighbor_index] += self.get_charge(index)
            self.charge[index] = 0.0
            self.set_ignore(index, True)
            
    def has_implicit_hydrogens(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def init_atom(self):
        """ iterate through molecule atoms and build the atoms table """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def perceive_rings(self, keep_chorded_rings, keep_equivalent_rings):
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

        def isRingAromatic(ring_atom_indices):
            for atom_idx1, atom_idx2 in self.get_bonds_in_ring(ring_atom_indices):
                bond = self.mol.GetBondBetweenAtoms(atom_idx1, atom_idx2)
                if not bond.GetIsAromatic():
                    return False
            return True

        hjk_ring_detection = utils.HJKRingDetection(self.graph) 
        rings = hjk_ring_detection.scan(keep_chorded_rings, keep_equivalent_rings) # list of tuples of atom indices
        for ring_atom_idxs in rings:
            if isRingAromatic(ring_atom_idxs):
                self.rings_aromatic.append(ring_atom_idxs)
            self.rings[ring_atom_idxs] = {'corner_flip':False}
            graph = {}
            for atom_idx in ring_atom_idxs:
                self.atom_to_ring_id[atom_idx].append(ring_atom_idxs)
                # graph of atoms affected by potential ring movements
                graph[atom_idx] = self.walk_recursive(atom_idx, collected=[], exclude=list(ring_atom_idxs))
            self.rings[ring_atom_idxs]['graph'] = graph

    def init_bond(self):
        """ iterate through molecule bonds and build the bond table (id, table)
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

    def get_mol_name(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def find_pattern(self, smarts):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_smiles_and_order(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def show(self):
        tot_charge = 0

        print("Molecule setup\n")
        print("==============[ ATOMS ]===================================================")
        print("idx  |          coords            | charge |ign| atype    | connections")
        print("-----+----------------------------+--------+---+----------+--------------- . . . ")
        for k, v in list(self.coord.items()):
            print("% 4d | % 8.3f % 8.3f % 8.3f | % 1.3f | %d" % (k, v[0], v[1], v[2],
                  self.charge[k], self.atom_ignore[k]),
                  "| % -8s |" % self.atom_type[k],
                  self.graph[k])
            tot_charge += self.charge[k]
        print("-----+----------------------------+--------+---+----------+--------------- . . . ")
        print("  TOT CHARGE: %3.3f" % tot_charge)

        print("\n======[ DIRECTIONAL VECTORS ]==========")
        for k, v in list(self.coord.items()):
            if k in self.interaction_vector:
                print("% 4d " % k, self.atom_type[k], end=' ')

        print("\n==============[ BONDS ]================")
        # For sanity users, we won't show those keys for now
        keys_to_not_show = ['bond_order', 'type']
        for k, v in list(self.bond.items()):
            t = ', '.join('%s: %s' % (i, j) for i, j in v.items() if not i in keys_to_not_show)
            print("% 8s - " % str(k), t)

        # _macrocycle_typer.show_macrocycle_scores(self)

        print('')



class RDKitMoleculeSetup(MoleculeSetup):

    @classmethod
    def from_mol(cls, mol, keep_chorded_rings=False, keep_equivalent_rings=False,
                 assign_charges=True, conformer_id=-1):
        if mol.GetNumConformers() == 0: 
            raise ValueError("RDKit molecule does not have a conformer. Need 3D coordinates.")
        rdkit_conformer = mol.GetConformer(conformer_id) 
        if not rdkit_conformer.Is3D():
            warnings.warn("RDKit molecule not labeled as 3D. This warning won't show again.")
            RDKitMoleculeSetup.warned_not3D = True
        if mol.GetNumConformers() > 1 and conformer_id == -1:
            msg = "RDKit molecule has multiple conformers. Considering only the first one." 
            print(msg, file=sys.stderr)
        molsetup = cls()
        molsetup.mol = mol
        molsetup.atom_true_count = molsetup.get_num_mol_atoms()
        molsetup.name = molsetup.get_mol_name()
        coords = rdkit_conformer.GetPositions()
        molsetup.init_atom(assign_charges, coords)
        molsetup.perceive_rings(keep_chorded_rings, keep_equivalent_rings)
        molsetup.init_bond()
        return molsetup


    def get_smiles_and_order(self):
        """
            return the SMILES after Chem.RemoveHs()
            and the mapping between atom indices in smiles and self.mol
        """

        # 3D SDF files written by other toolkits (OEChem, ChemAxon)
        # seem to not include the chiral flag in the bonds block, only in
        # the atoms block. RDKit ignores the atoms chiral flag as per the
        # spec. When reading SDF (e.g. from PubChem/PDB),
        # we may need to have RDKit assign stereo from coordinates, see:
        # https://sourceforge.net/p/rdkit/mailman/message/34399371/
        mol_noH = Chem.RemoveHs(self.mol) # imines (=NH) may become chiral
        # stereo imines [H]/N=C keep [H] after RemoveHs()
        # H isotopes also kept after RemoveHs()
        atomic_num_mol_noH = [atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
        noH_to_H = []
        parents_of_hs = {}
        for (index, atom) in enumerate(self.mol.GetAtoms()):
            if atom.GetAtomicNum() == 1: continue
            for i in range(len(noH_to_H), len(atomic_num_mol_noH)):
                if atomic_num_mol_noH[i] > 1:
                    break
                h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
                assert(h_atom.GetAtomicNum() == 1)
                neighbors = h_atom.GetNeighbors()
                assert(len(neighbors) == 1)
                parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
                noH_to_H.append('H')
            noH_to_H.append(index)
        extra_hydrogens = len(atomic_num_mol_noH) - len(noH_to_H)
        if extra_hydrogens > 0:
            assert(set(atomic_num_mol_noH[len(noH_to_H):]) == {1})
        for i in range(extra_hydrogens):
            h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
            assert(h_atom.GetAtomicNum() == 1)
            neighbors = h_atom.GetNeighbors()
            assert(len(neighbors) == 1)
            parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
            noH_to_H.append('H')

        # noH_to_H has the same length as the number of atoms in mol_noH
        # and each value is:
        #    - the index of the corresponding atom in mol, if value is integer
        #    - an hydrogen, if value is "H"
        # now, we need to replace those "H" with integers
        # "H" occur with stereo imine (e.g. [H]/N=C) and heavy Hs (e.g. [2H])
        hs_by_parent = {}
        for hidx, pidx in parents_of_hs.items():
            hs_by_parent.setdefault(pidx, [])
            hs_by_parent[pidx].append(hidx)
        for pidx, hidxs in hs_by_parent.items():
            siblings_of_h = [atom for atom in self.mol.GetAtomWithIdx(noH_to_H[pidx]).GetNeighbors() if atom.GetAtomicNum() == 1]
            sortidx = [i for i, j in sorted(list(enumerate(siblings_of_h)), key=lambda x: x[1].GetIdx())]
            if len(hidxs) == len(siblings_of_h):  
                # This is the easy case, just map H to each other in the order they appear
                for i, hidx in enumerate(hidxs):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            elif len(hidxs) < len(siblings_of_h):
                # check hydrogen isotopes
                sibling_isotopes = [siblings_of_h[sortidx[i]].GetIsotope() for i in range(len(siblings_of_h))]
                molnoH_isotopes = [mol_noH.GetAtomWithIdx(hidx) for hidx in hidxs]
                matches = []
                for i, sibling_isotope in enumerate(sibling_isotopes):
                    for hidx in hidxs[len(matches):]:
                        if mol_noH.GetAtomWithIdx(hidx).GetIsotope() == sibling_isotope:
                            matches.append(i)
                            break
                if len(matches) != len(hidxs):
                    raise RuntimeError("Number of matched isotopes %d differs from query Hs: %d" % (len(matched), len(hidxs)))
                for hidx, i in zip(hidxs, matches):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            else:
                raise RuntimeError("nr of Hs in mol_noH bonded to an atom exceeds nr of Hs in self.mol")

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

    def init_atom(self, assign_charges, coords):
        """ initialize the atom table information """
        # extract/generate charges
        if assign_charges:
            copy_mol = Chem.Mol(self.mol)
            for atom in copy_mol.GetAtoms():
                if atom.GetAtomicNum() == 34:
                    atom.SetAtomicNum(16)
            rdPartialCharges.ComputeGasteigerCharges(copy_mol)
            charges = [a.GetDoubleProp('_GasteigerCharge') for a in copy_mol.GetAtoms()]
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

    def init_bond(self):
        """ initialize bond information """
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = int(b.GetBondType())
            # fix the RDKit aromatic type (FIXME)
            if bond_order == 12: # aromatic
                bond_order = 5
            if bond_order == 1:
                rotatable = True
            else:
                rotatable = False
            idx1_rings = set(self.get_atom_rings(idx1))
            idx2_rings = set(self.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.add_bond(idx1, idx2, order=bond_order, rotatable=rotatable, in_rings=in_rings)

    def copy(self):
        """ return a copy of the current setup"""
        newsetup = RDKitMoleculeSetup()
        for key, value in self.__dict__.items():
            if key != "mol":
                newsetup.__dict__[key] = deepcopy(value)
        newsetup.mol = Chem.Mol(self.mol) # not sure how deep of a copy this is
        return newsetup

    def has_implicit_hydrogens(self):
        # based on needsHs from RDKit's AddHs.cpp
        for atom in self.mol.GetAtoms():
            nr_H_neighbors = 0
            for neighbor in atom.GetNeighbors():
                nr_H_neighbors += int(neighbor.GetAtomicNum() == 1)
            if atom.GetTotalNumHs(includeNeighbors=False) > nr_H_neighbors:
                return True
        return False


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

    def init_bond(self):
        """initialize bond data table"""
        for b in ob.OBMolBondIter(self.mol):
            idx1 = b.GetBeginAtomIdx() - 1
            idx2 = b.GetEndAtomIdx() - 1
            bond_order = b.GetBondOrder()
            if b.IsAromatic():
                bond_order = 5
            # check if bond is a ring bond, i.e., both atoms belongs to the same ring
            idx1_rings = set(self.get_atom_rings(idx1))
            idx2_rings = set(self.get_atom_rings(idx2))
            in_rings = list(set.intersection(idx1_rings, idx2_rings))
            self.add_bond(idx1, idx2, order=bond_order, in_rings=in_rings)

    def copy(self):
        """ return a copy of the current setup"""
        return OBMoleculeSetup(self.mol, template=self)
