#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os
from collections import defaultdict

import numpy as np
from scipy import spatial

from .utils.covalent_radius_table import covalent_radius
from .utils.autodock4_atom_types_elements import autodock4_atom_types_elements


atom_property_definitions = {'H': 'vdw', 'C': 'vdw', 'A': 'vdw', 'N': 'vdw', 'P': 'vdw', 'S': 'vdw',
                             'Br': 'vdw', 'I': 'vdw', 'F': 'vdw', 'Cl': 'vdw',
                             'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc',
                             'HD': 'hb_don', 'HS': 'hb_don',
                             'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal',
                             'MG': 'metal', 'CA': 'metal', 'FE': 'metal', 'ZN': 'metal', 'MN': 'metal',
                             'W': 'water',
                             'G0': 'glue', 'G1': 'glue', 'G2': 'glue', 'G3': 'glue',
                             'CG0': 'glue', 'CG1': 'glue', 'CG2': 'glue', 'CG3': 'glue'}


def _read_receptor_pdbqt_file(pdbqt_filename):
    i = 0
    atoms = []
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ("xyz", "f4", (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U2')]
    atom_annotations = {'hb_acc': [], 'hb_don': [],
                        'all': [], 'vdw': [],
                        'metal': []}
    # TZ is a pseudo atom for AutoDock4Zn FF
    pseudo_atom_types = ['TZ']

    with open(pdbqt_filename) as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('ATOM') or line.startswith("HETATM"):
                idx = i
                serial = int(line[6:11].strip())
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chainid = line[21].strip()
                resid = int(line[22:26].strip())
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=np.float32)
                partial_charges = float(line[71:77].strip())
                atom_type = line[77:79].strip()

                if not atom_type in pseudo_atom_types:
                    atom_annotations['all'].append(i)
                    atom_annotations[atom_property_definitions[atom_type]].append(i)
                    atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type))

                i += 1

    atoms = np.array(atoms, dtype=atoms_dtype)

    return atoms, atom_annotations


def _identify_bonds(atom_idx, positions, atom_types):
    bonds = defaultdict(list)
    KDTree = spatial.cKDTree(positions)
    bond_allowance_factor = 1.1
    # If we ask more than the number of coordinates/element
    # in the BHTree, we will end up with some inf values
    k = 5 if len(atom_idx) > 5 else len(atom_idx)
    atom_idx = np.array(atom_idx)

    for atom_i, position, atom_type in zip(atom_idx, positions, atom_types):
        distances, indices = KDTree.query(position, k=k)
        r_cov = covalent_radius[autodock4_atom_types_elements[atom_type]]

        optimal_distances = [bond_allowance_factor * (r_cov + covalent_radius[autodock4_atom_types_elements[atom_types[i]]]) for i in indices[1:]]
        bonds[atom_i] = atom_idx[indices[1:][np.where(distances[1:] < optimal_distances)]].tolist()

    return bonds


class PDBQTReceptor:

    def __init__(self, pdbqt_filename):
        self._pdbqt_filename = pdbqt_filename
        self._atoms = None
        self._atom_annotations = None
        self._KDTree = None

        self._atoms, self._atom_annotations = _read_receptor_pdbqt_file(self._pdbqt_filename)
        # We add to the KDTree only the rigid part of the receptor
        self._KDTree = spatial.cKDTree(self._atoms['xyz'])
        self._bonds = _identify_bonds(self._atom_annotations['all'], self._atoms['xyz'], self._atoms['atom_type'])

    def __repr__(self):
        return ('<Receptor from PDBQT file %s containing %d atoms>' % (self._pdbqt_filename, self._atoms.shape[0]))

    def atoms(self, atom_idx=None):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None and self._atoms.size > 1:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
            atoms = self._atoms[atom_idx]
        else:
            atoms = self._atoms

        return atoms.copy()

    def positions(self, atom_idx=None):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        return np.atleast_2d(self.atoms(atom_idx)['xyz'])

    def closest_atoms_from_positions(self, xyz, radius, atom_properties=None, ignore=None):
        """Retrieve indices of the closest atoms around a positions/coordinates 
        at a certain radius.

        Args:
            xyz (np.ndarray): array of 3D coordinates
            raidus (float): radius
            atom_properties (str): property of the atoms to retrieve 
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)
            ignore (int or list): ignore atom for the search using atom id (0-based)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        index = self._KDTree.query_ball_point(xyz, radius, p=2, return_sorted=True)

        # When nothing was found around...
        if not index:
            return np.array([])

        # Handle the case when positions for of only one atom was passed in the input
        try:
            index = {i for j in index for i in j}
        except:
            index = set(index)

        if atom_properties is not None:
            if not isinstance(atom_properties, (list, tuple)):
                atom_properties = [atom_properties]

            try:
                for atom_property in atom_properties:
                    index.intersection_update(self._atom_annotations[atom_property])
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_annotations.keys()))

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            index = index.difference([i for i in ignore])

        index = list(index)
        atoms = self._atoms[index].copy()

        return atoms

    def closest_atoms(self, atom_idx, radius, atom_properties=None):
        """Retrieve indices of the closest atoms around a positions/coordinates 
        at a certain radius.

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            raidus (float): radius
            atom_properties (str or list): property of the atoms to retrieve 
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        return self.closest_atoms_from_positions(self._atoms[atom_idx]['xyz'], radius, atom_properties, atom_idx)

    def neighbor_atoms(self, atom_idx):
        """Return neighbor (bonded) atoms

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            list_of_list: list of lists containing the neighbor (bonded) atoms (0-based)

        """
        if not isinstance(atom_idx, (list, tuple, np.ndarray)):
            atom_idx = [atom_idx]

        return [self._bonds[i] for i in atom_idx]
