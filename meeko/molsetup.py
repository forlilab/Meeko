#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from copy import deepcopy
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import json
import warnings
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolInterchange

from .utils import rdkitutils
from .utils import utils
from .utils.geomutils import calcDihedral
from .utils.pdbutils import PDBAtomInfo
from .receptor_pdbqt import PDBQTReceptor

try:
    from openbabel import openbabel as ob
    from .utils import obutils
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True

try:
    from misctools import StereoIsomorphism
except ImportError as _import_misctools_error:
    _has_misctools = False
else:
    _has_misctools = True


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
    # TODO: fill out additional detail for docstring
    """ mol: molecule structurally prepared with explicit hydrogens

        the setup provides:
            - data storage
            - SMARTS matcher for all atom typers

    Base molecule setup class, providing a way to store information about molecules
    for a number of purposes.

    Attributes
    ----------
    mol: rdkit.Chem.rdchem.Mol
        not guaranteed to be present in the object, contains the RDKit mol object that the molecule setup was based on
        or corresponds to.
    atom_pseudo: List[]
        a list of indices of pseudo-atoms?

    coord: OrderedDict()
        a mapping between atom index and coordinates, where atom indices are stored as ints and coordinates are
        numpy arrays of three floats.
    charge: OrderedDict()
        a mapping between atom index (int) and charge (float)
    pdbinfo: OrderedDict()
        a mapping between atom index (int) and pdb data (PDBAtomInfo) for that atom
    atom_type: OrderedDict()
        TODO: if these values are meant to be consistent, then this should be an enum?
        a mapping from some sort of index (presumably an int) to atom type (string)
    atom_params: dict()
        a mapping from the name of a parameter (string) to a parameter

    dihedral_interactions: list[]
        a list of strings maybe?
    dihedral_partaking_atoms: dict()
        a mapping from atom index (int?) to dihedral index (int?)
    dihedral_labels: dict()
        a mapping from atom index to a string dihedral label

    atom_ignore: OrderedDict()
        a mapping from atom index to the bool that indicates whether it should be ignored
    chiral: OrderedDict()
        a mapping from atom index to a bool that indicates whether an atom is chiral
    atom_true_count: int
        the number of atoms being represented in this molsetup
    graph: OrderedDict()
        a mapping from atom index (int) to a list of neighboring atom indices?
    bond: OrderedDict()
        a mapping from bond id (int?) to a dictionary containing two elements. The elements are
        bond_order (int) and rotatable (bool).
    element: OrderedDict()
        represents the atomic number of the atom index. A mapping from atom index to atomic number
    interaction_vector: OrderedDict()
        a mapping from atom index to a list representing directional interaction vectors for atom idx
    flexibility_model: dict()
        unclear what exactly is consistently contained by this dict, but it seems to be a keyword to value dictionary
    ring_closure_info: dict() containing two elements.
        desperately in need of refactoring.
    restraints: list[]
        list of restraint dataclasses
    is_sidechain: bool
        indicates whether this molsetup pertains to a sidechain
    rmsd_symmetry_indices: empty tuple
        only seems to be set in RDKit version of molsetup, at which point it set to be equal to tuples of the indices
        of the molecule's atoms that match a substructure query.

    rings: dict()
        store information about rings, like if they have corners that can be flipped and the graph of atoms that belong
        to them. Mapping from ring id (string?) to a dictionary with two elements:
            'corner_flip': bool
            'graph': dict()
    rings_aromatic: list[]
        contains the list of ring_id items that are aromatic
    atom_to_ring_id: dict()
        mapping of atom index to ring id of each atom belonging to the ring
    ring_corners: dict()
        unclear what this is a mapping of, but is used to store corner flexibility for the rings

    name: string or None
        should this be here or does it only need to be in the RDKit version of this class.
        either contains the name of the molecule as provided in RDKit or is None.
    rotamers: list[]
        A list of rotamers, where each rotamer is represented as a dictionary mapping from bond_id (presumably a string
        or an int) to an int or a float representing something
    """

    def __init__(self):

        self.atom_pseudo = []
        self.coord = OrderedDict()  # FIXME all OrderedDict should be converted to lists?
        self.charge = OrderedDict()
        self.pdbinfo = OrderedDict()
        self.atom_type = OrderedDict()
        self.atom_params = {}
        self.dihedral_interactions = list()
        self.dihedral_partaking_atoms = dict()
        self.dihedral_labels = dict()
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
        self.restraints = []
        self.is_sidechain = False
        self.rmsd_symmetry_indices = ()
        self.rings = {}
        self.rings_aromatic = []
        self.atom_to_ring_id = defaultdict(list)
        self.ring_corners = {}  # used to store corner flexibility
        self.name = None
        self.rotamers = []

    def copy(self):
        newsetup = MoleculeSetup()
        newsetup.__dict__ = deepcopy(self.__dict__)
        return newsetup

    @classmethod
    def from_mol(cls, mol, keep_chorded_rings=False, keep_equivalent_rings=False,
                 assign_charges=True):
        """
        Creates an instance of a molsetup object and loads data into it from an RDKit mol object?

        Parameters
        ----------
        mol: rdkit.Chem.rdchem.Mol
            RDKit Mol to load data from.
        keep_chorded_rings: bool
            needs info
        keep_equivalent_rings: bool
            needs info
        assign_charges: bool
            needs info

        Returns
        -------
        A new molsetup object with data from the provided mol.
        The following fields would be populated: mol, atom_true_count_, name,

        Note
        ----
        Why is this implemented here in this way when all of the methods it calls are methods we want the
        inheriting classes to override? Its call signatures also seem to be consistent with the way methods are
        defined in a very specific subclass of this class, so it would behoove us to consider the sublass/superclass
        structure here.
        """
        molsetup = cls()
        molsetup.mol = mol
        molsetup.atom_true_count = molsetup.get_num_mol_atoms()
        molsetup.name = molsetup.get_mol_name()
        molsetup.init_atom(assign_charges)  # find a way to standardize signature across classes
        molsetup.init_bond()
        molsetup.perceive_rings(keep_chorded_rings, keep_equivalent_rings)
        return molsetup

    @classmethod
    def from_prmtop_inpcrd(cls, prmtop, crd_filename):
        """
        Creates an instance of a molsetup object and loads data into it from prmtop and a crd file.

        Parameters
        ----------
        prmtop: unclear, should be renamed
        crd_filename: presumable the name of a file containing coordinates

        Returns
        -------
        A molsetup populated with data from the input prmtop and crd file.

        Note
        ____
        Is this used anywhere or should it be deprecated. Or can it be deprecated.
        """
        molsetup = cls()
        x, y, z = prmtop._coords_from_inpcrd(crd_filename)
        all_atom_indices = prmtop.vmdsel("")
        if len(all_atom_indices) != len(x):
            raise RuntimeError("number of prmtop atoms differs from x coordinates")
        prmtop.upper_atom_types = prmtop._gen_gaff_uppercase()
        charges = [chrg / 18.2223 for chrg in prmtop.read_flag("CHARGE")]
        atomic_nrs = prmtop.read_flag("ATOMIC_NUMBER")
        pdbqt_string = ""
        molsetup.atom_params["rmin_half"] = []
        molsetup.atom_params["epsilon"] = []
        vdw_by_atype = {}
        for vdw in prmtop._retrieve_vdw():
            if vdw.atom in vdw_by_atype:
                raise RuntimeError("repeated atom type from prmtop._retrieve_vdw()")
            vdw_by_atype[vdw.atom] = {"rmin_half": vdw.r, "epsilon": vdw.e}
        for i in prmtop.atom_sel_idx:
            pdbinfo = rdkitutils.PDBAtomInfo(
                name=prmtop.pdb_atom_name[i],
                resName=prmtop.pdb_resname[i],
                resNum=int(prmtop.pdb_resid[i]),
                chain=" ",
            )
            atype = prmtop.upper_atom_types[i]
            molsetup.add_atom(
                i,
                coord=(x[i], y[i], z[i]),
                charge=charges[i],
                atom_type=atype,
                element=atomic_nrs[i],
                pdbinfo=pdbinfo,
                chiral=False,
                ignore=False,
                add_atom_params={
                    "rmin_half": vdw_by_atype[atype]["rmin_half"],
                    "epsilon": vdw_by_atype[atype]["epsilon"],
                }
            )
            # molsetup.atom_params["rmin_half"].append(vdw_by_atype[atype]["rmin_half"])
            # molsetup.atom_params["epsilon"].append(vdw_by_atype[atype]["epsilon"])
        bond_order = 1  # the prmtop does not have bond orders, so we set all to 1
        bonds_inc_h = prmtop.read_flag("BONDS_INC_HYDROGEN")
        bonds_not_h = prmtop.read_flag("BONDS_WITHOUT_HYDROGEN")
        bonds = bonds_inc_h + bonds_not_h
        nr_bonds = int(len(bonds) / 3)
        for index_bond in range(nr_bonds):
            i_atom = int(bonds[index_bond * 3 + 0] / 3)
            j_atom = int(bonds[index_bond * 3 + 1] / 3)
            if (i_atom in prmtop.atom_sel_idx) and (j_atom in prmtop.atom_sel_idx):
                molsetup.add_bond(i_atom, j_atom, order=bond_order, rotatable=False)
        return molsetup

    @classmethod
    def from_pdb_and_residue_params(cls, pdb_fname, residue_params_fname=None):
        """
        Creates an instance of a molsetup object and loads data into it from a pdb file and a parameter file.

        Parameters
        ----------
        pdb_fname: string
            the filepath for a pdb file to load data from
        residue_params_fname: string or None
            the filepath for residue parameters to be used in the molsetup
        Returns
        -------
        A molsetup populated with data based off of the pdb, and if it was provided, the input parameters.

        Note
        ----
        Are the residue parameters being used at all in this? Also, this does not seem to be used anywhere,
        is this something we want to consider deprecating. The coordinate part of the atom being added seems to be
        inconsistent,
        """
        if residue_params_fname is not None:
            with open(residue_params_fname) as f:
                residue_params = json.load(f)
        with open(pdb_fname) as f:
            lines = f.readlines()
        x = []
        y = []
        z = []
        res_list = []
        atom_names_list = []
        last_res = None
        pdbinfos = []
        for line in lines:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resn = line[17:20]
            resi = int(line[22:26])
            chain = line[21:22]
            res = (resn, resi, chain)
            if res != last_res:
                res_list.append(resn)
                atom_names_list.append([])
            last_res = res
            atom_name = line[12:16].strip()
            atom_names_list[-1].append(atom_name)
            x.append(float(line[30:38]))
            y.append(float(line[38:46]))
            z.append(float(line[46:54]))
            pdbinfo = rdkitutils.PDBAtomInfo(
                name=atom_name,
                resName=resn,
                resNum=resi,
                chain=chain
            )
            pdbinfos.append(pdbinfo)

        atom_params = {}
        atom_counter = 0
        all_ok = True
        all_err = ""
        for (res, atom_names) in zip(res_list, atom_names_list):
            p, ok, err = PDBQTReceptor.get_params_for_residue(res, atom_names)
            nr_params_to_add = set([len(values) for _, values in p.items()])
            if len(nr_params_to_add) != 1:
                raise RuntimeError("inconsistent number of parameters in %s" % p)
            nr_params_to_add = nr_params_to_add.pop()
            all_ok &= ok
            all_err += err
            for key in p:
                atom_params.setdefault(key, [None] * atom_counter)
                atom_params[key].extend(p[key])
            atom_counter += nr_params_to_add
            for key in atom_params:
                if key not in p:
                    atom_params[key].extend([None] * nr_params_to_add)

        if not all_ok:
            raise RuntimeError(all_err)

        molsetup = cls()
        charges = atom_params.pop("gasteiger")
        atypes = atom_params.pop("atom_types")
        for i in range(len(x)):
            molsetup.add_atom(
                i,
                coord=(x[i], y[i], z[i]),
                charge=charges[i],
                atom_type=atypes[i],
                element=None,
                pdbinfo=pdbinfo,
                chiral=False,
                ignore=False,
            )

        molsetup.atom_params = atom_params

        return molsetup

    def set_types_from_uniq_atom_params(self, uniq_atom_params, prefix):
        """
        Uses a UniqAtomParams object to set the atom_type attribute in the molsetup object. Adds the specified prefix
        to each of the atom types.

        Parameters
        ----------
        uniq_atom_params: UniqAtomParams
            A uniq atom params object to extract the atom_types from
        prefix: string
            A prefix to be appended to all the atom_types.

        Returns
        -------
        None

        Note
        ----
        Deprecation candidate? What is this being used for/kept around for?

        """
        param_idxs = uniq_atom_params.get_indices_from_atom_params(self.atom_params)
        if len(param_idxs) != len(self.atom_type):
            raise RuntimeError("{len(param_idxs)} parameters but {len(self.atom_type)} in molsetup")
        for i, j in enumerate(param_idxs):
            self.atom_type[i] = f"{prefix}{j}"
        return None

    def add_atom(self, idx=None, coord=np.array([0.0, 0.0, 0.0], dtype='float'),
                 element=None, charge=0.0, atom_type=None, pdbinfo=None,
                 ignore=False, chiral=False, overwrite=False, add_atom_params=None):
        """
        Adds an atom with all of the atom information a user wants to specify to the molsetup. Every property is
        optional, but possible to set.

        Parameters
        ----------
        idx: int or None
            atom index
        coord: np float array of length 3
            coordinates f the atom
        element: int or None
            the atomic number of the
        charge: float
            partial charges to be loaded for the atom
        atom_type: string or int or None
            needs info on exactly what these are
        pdbinfo: string or None:
            pdb string for the atom
        ignore: bool
            ignore flag for the atom
        chiral: bool
            indicates whether the atom should be marked as chiral
        overwrite: bool
            are we overwriting other atoms that may be in the same coordinate position as this one
        add_atom_params: or None


        Returns
        -------
        False if the index is already occupied and we are not overwriting existing atoms. If the method succeeds,
        returns the index of the added atom.
        """
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
        self.set_chiral(idx, chiral)
        self.set_ignore(idx, ignore)
        self.graph.setdefault(idx, [])
        for key in self.atom_params:
            if type(add_atom_params) == dict and key in add_atom_params:
                value = add_atom_params[key]
            else:
                value = None
            self.atom_params[key].append(value)
        return idx

    # TODO: are we deprecating/deleting the following or is it functionality we still want to add
    def del_atom(self, idx):
        """ remove an atom and update all data associate with it """
        pass
        # coords
        # charge
        # element
        # type
        # neighbor graph
        # ignore
        # update bonds bonds (using the neighbor graph)
        # If pseudo-atom, update other information, too

    # pseudo-atoms
    def add_pseudo(self, coord=np.array([0.0, 0.0, 0.0], dtype='float'), charge=0.0,
                   anchor_list=None, atom_type=None, rotatable=False,
                   pdbinfo=None, directional_vectors=None, ignore=False, overwrite=False):
        """
        Adds a new pseudoatom to the molsetup. Multiple bonds can be specified in "anchor_list" to support the
        centroids of aromatic rings. If rotatable, makes the anchor atom rotatable to allow the pseudoatom movement

        Parameters
        ----------
        coord: numpy float array of length 3
            pseudoatom coordinates
        charge: float
            partial charge for the pseudoatom
        anchor_list: list[] or None
            a list of ints indicating multiple bonds that can be specified
        atom_type: or None

        rotatable: bool
        pdbinfo: string or None
        directional_vectors: or None
        ignore: bool
        overwrite: bool
            indicates whether the
        Returns
        -------

        """
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
                      ignore=ignore,
                      overwrite=overwrite)
        # anchor atoms
        if anchor_list is not None:
            for anchor in anchor_list:
                self.add_bond(idx, anchor, order=0, rotatable=rotatable)
        # directional vectors
        if directional_vectors is not None:
            self.add_interaction_vector(idx, directional_vectors)
        return idx

    def add_bond(self, idx1, idx2, order=0, rotatable=False):
        """
        Adds a bond to the molsetup graph between the two indices indicated.

        Parameters
        ----------
        idx1: int
            first atom's index
        idx2: int
            second atom's index
        order: int
            bond order
        rotatable: bool
            whether the bond is rotatable

        """

        if not idx2 in self.graph[idx1]:
            self.graph[idx1].append(idx2)
        if not idx1 in self.graph[idx2]:
            self.graph[idx2].append(idx1)
        self.set_bond(idx1, idx2, order, rotatable)

    def add_rotamer(self, indices_list, angles_list):
        xyz = self.coord
        rotamer = {}
        for (i1, i2, i3, i4), angle in zip(indices_list, angles_list):
            bond_id = self.get_bond_id(i2, i3)
            if bond_id in rotamer:
                raise RuntimeError("repeated bond %d-%d" % bond_id)
            if not self.bond[bond_id]["rotatable"]:
                raise RuntimeError("trying to add rotamer for non rotatable bond %d-%d" % bond_id)
            d0 = calcDihedral(xyz[i1], xyz[i2], xyz[i3], xyz[i4])
            rotamer[bond_id] = angle - d0
        self.rotamers.append(rotamer)

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
        """ return atoms connected to atom index
        Note
        ----
        This should be get_neighbors, just neigh is nowhere near clear enough.
        """
        return self.graph[idx]

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
            return [x for x in indices if not x in self.atom_pseudo]
        return indices

    # interaction vectors
    def add_interaction_vector(self, idx, vector_list):
        """ add vector list to list of directional interaction vectors for atom idx"""
        if idx not in self.interaction_vector:
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

    def set_bond(self, idx1, idx2, order=0, rotatable=False):
        """ populate bond lookup table with properties
            bonds are identified by any tuple of atom indices
            the function generates the canonical bond id

            order      : int
            rotatable  : bool
        """
        bond_id = self.get_bond_id(idx1, idx2)
        self.bond[bond_id] = {
            'bond_order': order,
            'rotatable': rotatable,
        }

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
            bond = (ring[i], ring[(i + 1) % n])
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
        rings = hjk_ring_detection.scan(keep_chorded_rings, keep_equivalent_rings)  # list of tuples of atom indices
        for ring_atom_idxs in rings:
            if isRingAromatic(ring_atom_idxs):
                self.rings_aromatic.append(ring_atom_idxs)
            self.rings[ring_atom_idxs] = {'corner_flip': False}
            graph = {}
            for atom_idx in ring_atom_idxs:
                self.atom_to_ring_id[atom_idx].append(ring_atom_idxs)
                # graph of atoms affected by potential ring movements
                graph[atom_idx] = self.walk_recursive(atom_idx, collected=[], exclude=list(ring_atom_idxs))
            self.rings[ring_atom_idxs]['graph'] = graph

    def add_dihedral_interaction(self, fourier_series):
        """ """
        index = 0
        for existing_fs in self.dihedral_interactions:
            if self.are_fourier_series_identical(existing_fs, fourier_series):
                return index
            index += 1
        safe_copy = json.loads(json.dumps(fourier_series))
        self.dihedral_interactions.append(safe_copy)
        return index

    @staticmethod
    def are_fourier_series_identical(fs1, fs2):
        index_by_periodicity1 = {fs1[index]["periodicity"]: index for index in range(len(fs1))}
        index_by_periodicity2 = {fs2[index]["periodicity"]: index for index in range(len(fs2))}
        if index_by_periodicity1 != index_by_periodicity2:
            return False
        for periodicity in index_by_periodicity1:
            index1 = index_by_periodicity1[periodicity]
            index2 = index_by_periodicity2[periodicity]
            for key in ["k", "phase", "periodicity"]:
                if fs1[index1][key] != fs2[index2][key]:
                    return False
        return True

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
                Setup.add_bond(idx1, idx2, order, rotatable)
        """
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_mol_name(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def find_pattern(self, smarts):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def get_smiles_and_order(self):
        raise NotImplementedError("This method must be overloaded by inheriting class")

    def write_xyz_string(self):
        n = len(self.element)
        string = "%d\n\n" % n
        for index in range(n):
            if index < self.atom_true_count:
                element = utils.mini_periodic_table[self.element[index]]
            else:
                element = "Ne"
            x, y, z = self.coord[index]
            string += "%3s %12.6f %12.6f %12.6f\n" % (element, x, y, z)
        return string

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

    def to_json(self):
        return json.dumps(self, cls=MoleculeSetupEncoder)

    @classmethod
    def from_json(cls, json_string):
        molsetup = json.loads(json_string, object_hook=cls.molsetup_json_decoder)
        return molsetup

    @staticmethod
    def molsetup_json_decoder(obj):
        """
        Takes an object and attempts to decode it into a molecule setup object. Handles type conversions that it knows
        about, otherwise leaves attributes as the output that JSON leaves them as

        Parameters
        ----------
        obj: Object
            This can be any object, but it should be a dictionary from deserializing a JSON of a molecule setup object.

        Returns
        -------
        If the input is a dictionary corresponding to a molecule setup, will return a molecule setup with data
        populated from the dictionary. Otherwise, returns the input object.

        """
        # if the input object is not a dict, we know that it will not be parsable and is unlikely to be usable or
        # safe data, so we should ignore it.
        if type(obj) is not dict:
            return obj

        # if there's a "mol" key in the input dictionary, then it's an RDKitMoleculeSetup
        # as opposed to being the base class. OpenBabel MoleculeSetups are deprecated,
        # assuming mol is an RDKit molecule, not an OpenBabel one.
        if "mol" in obj:
            mol = obj.pop("mol")
            molsetup = RDKitMoleculeSetup()
        else:
            mol = None
            molsetup = MoleculeSetup()

        # check that all the keys we expect are in the object dictionary as a safety measure
        expected_molsetup_keys = {"atom_pseudo", "coord", "charge", "pdbinfo", "atom_type", "atom_params",
                                  "dihedral_interactions", "dihedral_partaking_atoms", "dihedral_labels", "atom_ignore",
                                  "chiral", "atom_true_count", "graph", "bond", "element", "interaction_vector",
                                  "flexibility_model", "ring_closure_info", "restraints", "is_sidechain",
                                  "rmsd_symmetry_indices", "rings", "rings_aromatic", "atom_to_ring_id", "ring_corners",
                                  "name", "rotamers",
                                  }
        if set(obj.keys()) != expected_molsetup_keys:
            return obj

        separator_char = ","
        # creates a molecule setup and sets all the expected molsetup fields. Converts to desired type for molsetups.
        molsetup.atom_pseudo = [int(v) for v in obj["atom_pseudo"]]
        molsetup.coord = OrderedDict({int(k): np.asarray(v) for k, v in obj["coord"].items()})
        molsetup.charge = OrderedDict({int(k): float(v) for k, v in obj["charge"].items()})
        molsetup.pdbinfo = OrderedDict({int(k): PDBAtomInfo(*v) for k, v in obj["pdbinfo"].items()})
        molsetup.atom_type = OrderedDict({int(k): v for k, v in obj["atom_type"].items()})
        molsetup.atom_params = obj["atom_params"]
        molsetup.dihedral_interactions = obj["dihedral_interactions"]
        molsetup.dihedral_partaking_atoms = obj["dihedral_partaking_atoms"]
        molsetup.dihedral_labels = obj["dihedral_labels"]
        molsetup.atom_ignore = OrderedDict({int(k): v for k, v in obj["atom_ignore"].items()})
        molsetup.chiral = OrderedDict({int(k): v for k, v in obj["chiral"].items()})
        molsetup.atom_true_count = obj["atom_true_count"]
        molsetup.graph = OrderedDict({int(k): v for k, v in obj["graph"].items()})
        molsetup.bond = OrderedDict({tuple([int(i) for i in k.split(separator_char)]): v
                                     for k, v in obj["bond"].items()})
        molsetup.element = OrderedDict({int(k): v for k, v in obj["element"].items()})
        molsetup.interaction_vector = OrderedDict(obj["interaction_vector"])
        # Handling flexibility model dictionary of dictionaries
        molsetup.flexibility_model = obj["flexibility_model"]
        if 'rigid_body_connectivity' in molsetup.flexibility_model:
            tuples_rigid_body_connectivity = \
                {tuple([int(i) for i in k.split(separator_char)]): tuple(v)
                 for k, v in molsetup.flexibility_model['rigid_body_connectivity'].items()}
            molsetup.flexibility_model['rigid_body_connectivity'] = tuples_rigid_body_connectivity
        if 'rigid_body_graph' in molsetup.flexibility_model:
            molsetup.flexibility_model['rigid_body_graph'] = \
                {int(k): v for k, v in molsetup.flexibility_model['rigid_body_graph'].items()}
        if 'rigid_body_members' in molsetup.flexibility_model:
            molsetup.flexibility_model['rigid_body_members'] = \
                {int(k): v for k, v in molsetup.flexibility_model['rigid_body_members'].items()}

        molsetup.ring_closure_info = obj["ring_closure_info"]
        molsetup.restraints = [Restraint(*v) for k, v in obj["restraints"]]
        molsetup.is_sidechain = obj["is_sidechain"]
        molsetup.rmsd_symmetry_indices = tuple([tuple(v) for v in obj["rmsd_symmetry_indices"]])
        molsetup.rings = {tuple([int(i) for i in k.split(separator_char)]): v
                          for k, v in obj["rings"].items()}
        for key in molsetup.rings:
            if 'graph' in molsetup.rings[key]:
                molsetup.rings[key]['graph'] = {int(k): v for k, v in molsetup.rings[key]['graph'].items()}
        molsetup.rings_aromatic = [tuple(v) for v in obj["rings_aromatic"]]
        molsetup.atom_to_ring_id = obj["atom_to_ring_id"]
        molsetup.atom_to_ring_id = {int(k): [tuple(t) for t in v] for k, v in obj['atom_to_ring_id'].items()}
        molsetup.ring_corners = obj["ring_corners"]
        molsetup.name = obj["name"]
        molsetup.rotamers = obj["rotamers"]
        if mol is not None:
            rdkit_mols = rdMolInterchange.JSONToMols(mol)
            if len(rdkit_mols) != 1:
                raise ValueError(f"Expected 1 rdkit mol from json string but got {len(rdkit_mols)}")
            molsetup.mol = rdkit_mols[0]
        return molsetup


class RDKitMoleculeSetup(MoleculeSetup):

    @classmethod
    def from_mol(cls, mol, keep_chorded_rings=False, keep_equivalent_rings=False,
                 assign_charges=True, conformer_id=-1):
        if cls.has_implicit_hydrogens(mol):
            raise ValueError("RDKit molecule has implicit Hs. Need explicit Hs.")
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
        molsetup.init_bond()
        molsetup.perceive_rings(keep_chorded_rings, keep_equivalent_rings)
        molsetup.rmsd_symmetry_indices = cls.get_symmetries_for_rmsd(mol)

        # to store sets of coordinates, e.g. docked poses, as dictionaries indexed by
        # the atom index, because not all atoms need to have new coordinates specified
        # Unspecified hydrogen positions bonded to modified heavy atom positions
        # are to be calculated "on-the-fly".
        molsetup.modified_atom_positions = []  # list of dictionaries where keys are atom indices

        return molsetup

    def get_conformer_with_modified_positions(self, new_atom_positions):
        # we operate on one conformer at a time because SetTerminalAtomPositions
        # acts on all conformers of a molecule, and we don't want to guarantee
        # that all conformers require the same set of terminal atoms to be updated
        new_mol = Chem.Mol(self.mol)
        new_conformer = Chem.Conformer(self.mol.GetConformer())
        is_set_list = [False] * self.mol.GetNumAtoms()
        for atom_index, new_position in new_atom_positions.items():
            new_conformer.SetAtomPosition(atom_index, new_position)
            is_set_list[atom_index] = True
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(new_conformer, assignId=True)
        for atom_index, is_set in enumerate(is_set_list):
            if not is_set and new_mol.GetAtomWithIdx(atom_index).GetAtomicNum() == 1:
                neighbors = new_mol.GetAtomWithIdx(atom_index).GetNeighbors()
                if len(neighbors) != 1:
                    raise RuntimeError("Expected H to have one neighbors")
                Chem.SetTerminalAtomCoords(new_mol, atom_index, neighbors[0].GetIdx())
        return new_conformer

    def get_mol_with_modified_positions(self, new_atom_positions_list=None):
        if new_atom_positions_list is None:
            new_atom_positions_list = self.modified_atom_positions
        new_mol = Chem.Mol(self.mol)
        new_mol.RemoveAllConformers()
        for new_atom_positions in new_atom_positions_list:
            conformer = self.get_conformer_with_modified_positions(new_atom_positions)
            new_mol.AddConformer(conformer, assignId=True)
        return new_mol

    def get_smiles_and_order(self):
        """
            return the SMILES after Chem.RemoveHs()
            and the mapping between atom indices in smiles and self.mol
        """

        ### # support sidechains in which not all real atoms are included in PDBQT
        ### ignored = []
        ### for atom in self.mol.GetAtoms():
        ###     index = atom.GetIdx()
        ###     if self.atom_ignore[index]:
        ###         ignored.append(index)
        ### if len(ignored) > 0: # e.g. sidechain padded with amides
        ###     mol_no_ignore = Chem.EditableMol(self.mol)
        ###     for index in sorted(ignored, reverse=True):
        ###         mol_no_ignore.RemoveAtom(index)
        ###     # remove dangling Hs
        ###     dangling_hs = []
        ###     for atom in mol_no_ignore.GetMol().GetAtoms():
        ###         if atom.GetAtomicNum() == 1:
        ###             if len(atom.GetNeighbors()) == 0:
        ###                 dangling_hs.append(atom.GetIdx())
        ###     for index in sorted(dangling_hs, reverse=True):
        ###         mol_no_ignore.RemoveAtom(index)
        ###     mol_no_ignore = mol_no_ignore.GetMol()
        ###     Chem.SanitizeMol(mol_no_ignore) 
        ### else:
        mol_no_ignore = self.mol

        # 3D SDF files written by other toolkits (OEChem, ChemAxon)
        # seem to not include the chiral fla  g in the bonds block, only in
        # the atoms block. RDKit ignores the atoms chiral flag as per the
        # spec. When reading SDF (e.g. from PubChem/PDB),
        # we may need to have RDKit assign stereo from coordinates, see:
        # https://sourceforge.net/p/rdkit/mailman/message/34399371/
        ps = Chem.RemoveHsParameters()
        # a user reported PDBbind Mol Blocks to have hcount=1 for Hs,
        # which adds a query to the RDKit H atom and then Chem.RemoveHs
        # does not remove Hs with queries by default
        # https://github.com/forlilab/Meeko/issues/62
        # https://github.com/rdkit/rdkit/issues/6615
        ps.removeWithQuery = True
        mol_noH = Chem.RemoveHs(mol_no_ignore, ps)  # imines (=NH) may become chiral
        # stereo imines [H]/N=C keep [H] after RemoveHs()
        # H isotopes also kept after RemoveHs()
        atomic_num_mol_noH = [atom.GetAtomicNum() for atom in mol_noH.GetAtoms()]
        noH_to_H = []
        parents_of_hs = {}
        for (index, atom) in enumerate(mol_no_ignore.GetAtoms()):
            if atom.GetAtomicNum() == 1: continue
            for i in range(len(noH_to_H), len(atomic_num_mol_noH)):
                if atomic_num_mol_noH[i] > 1:
                    break
                h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
                assert (h_atom.GetAtomicNum() == 1)
                neighbors = h_atom.GetNeighbors()
                assert (len(neighbors) == 1)
                parents_of_hs[len(noH_to_H)] = neighbors[0].GetIdx()
                noH_to_H.append('H')
            noH_to_H.append(index)
        extra_hydrogens = len(atomic_num_mol_noH) - len(noH_to_H)
        if extra_hydrogens > 0:
            assert (set(atomic_num_mol_noH[len(noH_to_H):]) == {1})
        for i in range(extra_hydrogens):
            h_atom = mol_noH.GetAtomWithIdx(len(noH_to_H))
            assert (h_atom.GetAtomicNum() == 1)
            neighbors = h_atom.GetNeighbors()
            assert (len(neighbors) == 1)
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
            siblings_of_h = [atom for atom in mol_no_ignore.GetAtomWithIdx(noH_to_H[pidx]).GetNeighbors() if
                             atom.GetAtomicNum() == 1]
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
                    raise RuntimeError(
                        "Number of matched isotopes %d differs from query Hs: %d" % (len(matches), len(hidxs)))
                for hidx, i in zip(hidxs, matches):
                    noH_to_H[hidx] = siblings_of_h[sortidx[i]].GetIdx()
            else:
                raise RuntimeError("nr of Hs in mol_noH bonded to an atom exceeds nr of Hs in mol_no_ignore")

        smiles = Chem.MolToSmiles(mol_noH)
        order_string = mol_noH.GetProp("_smilesAtomOutputOrder")
        order_string = order_string.replace(',]', ']')  # remove trailing comma
        order = json.loads(order_string)  # mol_noH to smiles
        order = list(np.argsort(order))
        order = {noH_to_H[i]: order[i] + 1 for i in range(len(order))}  # 1-index
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

    @staticmethod
    def get_symmetries_for_rmsd(mol, max_matches=17):
        mol_noHs = Chem.RemoveHs(mol)
        matches = mol.GetSubstructMatches(mol_noHs, uniquify=False, maxMatches=max_matches)
        if len(matches) == max_matches:
            if mol.HasProp("_Name"):
                molname = mol.GetProp("_Name")
            else:
                molname = ""
            print(
                "warning: found the maximum nr of matches (%d) in RDKitMolSetup.get_symmetries_for_rmsd" % max_matches)
            print("Maybe this molecule is \"too\" symmetric? %s" % molname, Chem.MolToSmiles(mol_noHs))
        return matches

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
                          pdbinfo=rdkitutils.getPdbInfoNoNull(a),
                          chiral=False,
                          ignore=False)

    def init_bond(self):
        """ initialize bond information """
        for b in self.mol.GetBonds():
            idx1 = b.GetBeginAtomIdx()
            idx2 = b.GetEndAtomIdx()
            bond_order = int(b.GetBondType())
            # fix the RDKit aromatic type (FIXME)
            if bond_order == 12:  # aromatic
                bond_order = 5
            if bond_order == 1:
                rotatable = True
            else:
                rotatable = False
            self.add_bond(idx1, idx2, order=bond_order, rotatable=rotatable)

    def copy(self):
        """ return a copy of the current setup"""
        newsetup = RDKitMoleculeSetup()
        for key, value in self.__dict__.items():
            if key != "mol":
                newsetup.__dict__[key] = deepcopy(value)
        newsetup.mol = Chem.Mol(self.mol)  # not sure how deep of a copy this is
        return newsetup

    @staticmethod
    def has_implicit_hydrogens(mol):
        # based on needsHs from RDKit's AddHs.cpp
        for atom in mol.GetAtoms():
            nr_H_neighbors = 0
            for neighbor in atom.GetNeighbors():
                nr_H_neighbors += int(neighbor.GetAtomicNum() == 1)
            if atom.GetTotalNumHs(includeNeighbors=False) > nr_H_neighbors:
                return True
        return False

    def restrain_to(self, target_mol, kcal_per_angstrom_square=1.0, delay_angstroms=2.0):
        if not _has_misctools:
            raise ImportError(_import_misctools_error)
        stereo_isomorphism = StereoIsomorphism()
        mapping, idx = stereo_isomorphism(target_mol, self.mol)
        lig_to_drive = {b: a for (a, b) in mapping}
        num_real_atoms = target_mol.GetNumAtoms()
        target_positions = target_mol.GetConformer().GetPositions()
        for atom_index in range(len(mapping)):
            target_xyz = target_positions[lig_to_drive[atom_index]]
            restraint = Restraint(atom_index, target_xyz, kcal_per_angstrom_square, delay_angstroms)
            self.restraints.append(restraint)
        return


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
                output.append([y - 1 for y in x])
        return output

    def get_num_mol_atoms(self):
        return self.mol.NumAtoms()

    def get_equivalent_atoms(self):
        raise NotImplementedError

    def init_atom(self, assign_charges):
        """initialize atom data table"""
        for a in ob.OBMolAtomIter(self.mol):
            partial_charge = a.GetPartialCharge() * float(assign_charges)
            self.add_atom(a.GetIdx() - 1,
                          coord=np.asarray(obutils.getAtomCoords(a), dtype='float'),
                          element=a.GetAtomicNum(),
                          charge=partial_charge,
                          atom_type=None,
                          pdbinfo=obutils.getPdbInfoNoNull(a),
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
            self.add_bond(idx1, idx2, order=bond_order)

    def copy(self):
        """ return a copy of the current setup"""
        return OBMoleculeSetup(template=self)


class UniqAtomParams:
    """
    This seems to be a helper class used to keep parameters organized in a particular way that lets them be more usable.
    We should consider whether this is being used/if it is a candidate for deprecation.

    Attributes
    ----------
    params: list[]
        can be thought of as rows
    param_names: list[]
        can be thought of as columns
    """

    def __init__(self):
        self.params = []  # aka rows
        self.param_names = []  # aka column names

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates an UniqAtomParams object, populates it with information from the input dictionary, then returns
        the new object.

        Parameters
        ----------
        dictionary: dict()
            A dictionary containing the keys "params" and "param_names", where the value for "params" is parseable as
            rows and the value for "param_names" contains the corresponding column data.

        Returns
        -------
        A populated UniqAtomParams object
        """
        uap = UniqAtomParams()
        uap.params = [row.copy() for row in dictionary["params"]]
        uap.param_names = dictionary["param_names"].copy()
        return uap

    def get_indices_from_atom_params(self, atom_params):
        """
        To retrieve the indices of specific atom parameters in the UniqAtomParams object.

        Parameters
        ----------
        atom_params: dict()
            A dict with keys that correspond to the param names already in the UniqAtomParams object. The values are
            lists that should all be the same size, and

        Returns
        -------
        A list of indices corresponding to the order of parameters in the atom_params value lists that indicates the
        index of that "row" of parameters in UniqAtomParams params.
        """
        nr_items = set([len(values) for key, values in atom_params.items()])
        if len(nr_items) != 1:
            raise RuntimeError(f"all lists in atom_params must have same length, got {nr_items}")
        if set(atom_params) != set(self.param_names):
            msg = f"parameter names in atom_params differ from internal ones\n"
            msg += f"  - in atom_params: {set(atom_params)}"
            msg += f"  - internal: {set(self.param_names)}"
            raise RuntimeError(msg)
        nr_items = nr_items.pop()
        param_idxs = []
        for i in range(nr_items):
            row = [atom_params[key][i] for key in self.param_names]
            param_index = None
            for j, existing_row in enumerate(self.params):
                if row == existing_row:
                    param_index = j
                    break
            param_idxs.append(param_index)
        return param_idxs

    def add_parameter(self, new_param_dict):
        # remove None values to avoid a column with only Nones
        new_param_dict = {k: v for k, v in new_param_dict.items() if v is not None}
        incoming_keys = set(new_param_dict.keys())
        existing_keys = set(self.param_names)
        new_keys = incoming_keys.difference(existing_keys)
        for new_key in new_keys:
            self.param_names.append(new_key)
            for row in self.params:
                row.append(None)  # fill in empty "cell" in new "column"

        new_row = []
        for key in self.param_names:
            value = new_param_dict.get(key, None)
            new_row.append(value)

        if len(new_keys) == 0:  # try to match with existing row
            for index, row in enumerate(self.params):
                if row == new_row:
                    return index

        # if we are here, we didn't match
        new_row_index = len(self.params)
        self.params.append(new_row)
        return new_row_index

    def add_molsetup(self, molsetup, atom_params=None, add_atomic_nr=False, add_atom_type=False):
        if "q" in molsetup.atom_params or "atom_type" in molsetup.atom_params:
            msg = '"q" and "atom_type" found in molsetup.atom_params'
            msg += ' but are hard-coded to store molsetup.charge and'
            msg += ' molsetup.atom_type in the internal data structure'
            raise RuntimeError(msg)
        if atom_params is None:
            atom_params = molsetup.atom_params
        param_idxs = []
        for atom_index, ignore in molsetup.atom_ignore.items():
            if ignore:
                param_idx = None
            else:
                p = {k: v[atom_index] for (k, v) in molsetup.atom_params.items()}
                if add_atomic_nr:
                    if "atomic_nr" in p:
                        raise RuntimeError("trying to add atomic_nr but it's already in atom_params")
                    p["atomic_nr"] = molsetup.element[atom_index]
                if add_atom_type:
                    if "atom_type" in p:
                        raise RuntimeError("trying to add atom_type but it's already in atom_params")
                    p["atom_type"] = molsetup.atom_type[atom_index]
                param_idx = self.add_parameter(p)
            param_idxs.append(param_idx)
        return param_idxs


@dataclass
class Restraint:
    atom_index: int
    target_xyz: (float, float, float)
    kcal_per_angstrom_square: float
    delay_angstroms: float

    def copy(self):
        new_target_xyz = (
            self.target_xyz[0],
            self.target_xyz[1],
            self.target_xyz[2])
        new_restraint = Restraint(
            self.atom_index,
            new_target_xyz,
            self.kcal_per_angstrom_square,
            self.delay_angstroms)
        return new_restraint


# TODO: refactor molsetup class then refactor this and consider making it more readable.
class MoleculeSetupEncoder(json.JSONEncoder):
    """
    JSON Encoder class for molecule setup objects.
    """

    def default(self, obj):
        """
        Overrides the default JSON encoder for data structures for Molecule Setup objects.

        Parameters
        ----------
        obj: object
            Can take any object as input, but will only create the Molsetup JSON format for Molsetup objects. For all
            other objects will return the default json encoding.

        Returns
        -------
        A JSON serializable object that represents the MoleculeSetup class or the default JSONEncoder output for an
        object.
        """
        if isinstance(obj, MoleculeSetup):
            separator_char = ","  # TODO: consider setting this somewhere else so it is the same for decode and encode
            output_dict = {
                "atom_pseudo": obj.atom_pseudo,
                "coord": {k: v.tolist() for k, v in obj.coord.items()},
                "charge": obj.charge,
                "pdbinfo": obj.pdbinfo,
                "atom_type": obj.atom_type,
                "atom_params": obj.atom_params,
                "dihedral_interactions": obj.dihedral_interactions,
                "dihedral_partaking_atoms": obj.dihedral_partaking_atoms,
                "dihedral_labels": obj.dihedral_labels,
                "atom_ignore": obj.atom_ignore,
                "chiral": obj.chiral,
                "atom_true_count": obj.atom_true_count,
                "graph": obj.graph,
                "bond": {separator_char.join([str(i) for i in k]): v for k, v in obj.bond.items()},
                "element": obj.element,
                "interaction_vector": obj.interaction_vector,
                "flexibility_model": obj.flexibility_model,
                "ring_closure_info": obj.ring_closure_info,
                "restraints": obj.restraints,
                "is_sidechain": obj.is_sidechain,
                "rmsd_symmetry_indices": obj.rmsd_symmetry_indices,
                "rings": {separator_char.join([str(i) for i in k]): v for k, v in obj.rings.items()},
                "rings_aromatic": obj.rings_aromatic,
                "atom_to_ring_id": obj.atom_to_ring_id,
                "ring_corners": obj.ring_corners,
                "name": obj.name,
                "rotamers": obj.rotamers
            }
            # Since the flexibility model attribute contains dictionaries with tuples as keys, it needs to be treated
            # more specifically.
            if 'rigid_body_connectivity' in obj.flexibility_model:
                new_rigid_body_conn_dict = {separator_char.join([str(i) for i in k]): v
                                            for k, v in obj.flexibility_model['rigid_body_connectivity'].items()}
                output_dict["flexibility_model"] = \
                    {k: (v if k != 'rigid_body_connectivity' else new_rigid_body_conn_dict)
                     for k, v in obj.flexibility_model.items()}

            # Adds mol attribute if the input molecule setup is an RDKitMoleculeSetup
            if isinstance(obj, RDKitMoleculeSetup):
                output_dict["mol"] = rdMolInterchange.MolToJSON(obj.mol)
            return output_dict
        return json.JSONEncoder.default(self, obj)
