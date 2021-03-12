#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os
from collections import defaultdict

import numpy as np
from scipy import spatial

from .utils.van_der_waals_radius_table import van_der_waals_radius
from .utils.covalent_radius_table import covalent_radius
from .utils.autodock4_atom_types_elements import autodock4_atom_types_elements


atom_property_definitions = {'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc', 
                             'HD': 'hb_don', 'HS': 'hb_don',
                             'Cl': 'non-metal', 
                             'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal'}


def _read_receptor_pdbqt_file(pdbqt_filename):
    i = 0
    atoms = []
    atoms_dtype = [('id', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ("xyz", "f4", (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U2')]
    atom_properties = {'all': [], 'vdw': [], 'hb_acc': [], 'hb_don': [], 'non-metal': [], 'metal': []}

    with open(pdbqt_filename) as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('ATOM'):
                atom_id = int(line[6:11].strip())
                atom_name = line[12:15].strip()
                resname = line[17:20].strip()
                resid = int(line[22:26].strip())
                chainid = line[21].strip()
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=np.float32)
                partial_charges = float(line[71:77].strip())
                atom_type = line[77:79].strip()
                
                atom_properties['all'].append(i)
                try:
                    atom_properties[atom_property_definitions[atom_type]].append(i)
                except:
                    atom_properties['vdw'].append(i)
                atoms.append((atom_id, atom_name, resid, resname, chainid, xyz, partial_charges, atom_type))

                i += 1

    atoms = np.array(atoms, dtype=atoms_dtype)

    return atoms, atom_properties


def _identify_bonds(positions, atom_types):
    count = 0
    bonds = defaultdict(list)
    KDTree = spatial.cKDTree(positions)
    bond_length_allowance_factor = 1.1

    for atom_type, position in zip(atom_types, positions):
        distances, indices = KDTree.query(position, k=5)
        r_cov1 = covalent_radius[autodock4_atom_types_elements[atom_type]]

        optimal_distances = [bond_length_allowance_factor * (r_cov1 + covalent_radius[autodock4_atom_types_elements[atom_types[i]]]) for i in indices[1:]]
        bonds[count] = indices[1:][np.where(distances[1:] < optimal_distances)].tolist()

        count += 1

    return bonds


class PDBQTReceptor:

    def __init__(self, pdbqt_filename):
        self._pdbqt_filename = pdbqt_filename
        self._atoms = None
        self._atom_properties = None
        self._KDTree = None

        self._atoms, self._atom_properties = _read_receptor_pdbqt_file(self._pdbqt_filename)
        # We add to the KDTree only the rigid part of the receptor
        self._KDTree = spatial.cKDTree(self._atoms['xyz'])
        self._bonds = _identify_bonds(self._atoms['xyz'], self._atoms['atom_type'])

    def __repr__(self):
        return ('<Receptor from PDBQT file %s containing %d atoms>' % (self._pdbqt_filename, self._atoms.shape[0]))

    def positions(self, atom_ids=None):
        """
        Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_ids (int, list): index of one or multiple atoms

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        if atom_ids is not None and self._atoms.size > 1:
            if not isinstance(atom_ids, (list, tuple, np.ndarray)):
                atom_ids = np.array(atom_ids, dtype=np.int)
            # -1 because numpy array is 0-based
            positions = self._atoms['xyz'][atom_ids]
        else:
            positions = self._atoms['xyz']

        return np.atleast_2d(positions).copy()

    def atoms(self, atom_ids=None):
        """Return the atom i

        Args:
            atom_ids (int, list): index of one or multiple atoms

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_ids is not None and self._atoms.size > 1:
            if not isinstance(atom_ids, (list, tuple, np.ndarray)):
                atom_ids = np.array(atom_ids, dtype=np.int)
            atoms = self._atoms[atom_ids]
        else:
            atoms = self._atoms

        return atoms.copy()

    def closest_atoms(self, xyz, radius, atom_properties=None, ignore=None):
        """Retrieve indices of the closest atoms around x 
        at a certain radius.

        Args:
            xyz (array_like): array of 3D coordinates
            raidus (float): radius
            atom_property (str): property of the atoms to retrieve (properties: vdw, hb_don, hb_acc, metal)
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
            if not isinstance(atom_properties, (list, tuple, np.ndarray)):
                atom_properties = [atom_properties]

            try:
                for atom_property in atom_properties:
                    index.intersection_update(self._atom_properties[atom_property])
            except:
                raise KeyError('Atom property %s is not valid. Valid atom properties are: %s' % (atom_property, self._atom_properties.keys()))

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            index = index.difference([i for i in ignore])

        index = list(index)

        return self._atoms[index]

    def neighbor_atoms(self, atom_ids):
        """Return neighbor (bonded) atoms

        Args:
            atom_ids (int, list): index of one or multiple atoms (0-based)

        Returns:
            list_of_list: list of lists containing the neighbor (bonded) atoms (0-based)

        """
        if not isinstance(atom_ids, (list, tuple, np.ndarray)):
            atom_ids = [atom_ids]

        return [self._bonds[i] for i in atom_ids]
