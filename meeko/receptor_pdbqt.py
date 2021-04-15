#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os

import numpy as np
from scipy import spatial

from .utils.utils import _identify_bonds


def _read_receptor_pdbqt_file(pdbqt_filename):
    i = 0
    atoms = []
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ("xyz", "f4", (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U2')]

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

                atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type))

                i += 1

    atoms = np.array(atoms, dtype=atoms_dtype)

    return atoms


class PDBQTReceptor:

    def __init__(self, pdbqt_filename, extra_atom_types=None):
        """PDBQTReceptor class for reading PDBQT files for AutoDock4, AutoDock-GPU or AutoDock-Vina

        Args:
            pdbqt_filename (str): pdbqt filename
            extra_atom_types (dict): dictionary for defining new non-standard AutoDock atom types. (default: None)
                Used for identifying bonds in the molecule.
                Dictionary: {<new_atom_type1 (str)>: <element (str)>, ..., <new_atom_typeN (str)>: <element (str)>}

        """
        self._pdbqt_filename = pdbqt_filename
        self._atoms = None
        self._atom_annotations = None
        self._KDTree = None

        self._atoms = _read_receptor_pdbqt_file(self._pdbqt_filename)
        # We add to the KDTree only the rigid part of the receptor
        self._KDTree = spatial.cKDTree(self._atoms['xyz'])
        self._bonds = _identify_bonds(self._atoms['idx'], self._atoms['xyz'],
                                      self._atoms['atom_type'], extra_atom_types)

    def __repr__(self):
        return ('<Receptor from PDBQT file %s containing %d atoms>' % (self._pdbqt_filename, self._atoms.shape[0]))

    def atoms(self, atom_idx=None, atom_types=None):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
            atoms = self._atoms[atom_idx]
        else:
            atoms = self._atoms

        # Select atoms only with these atom types
        if atom_types is not None:
            mask = np.isin(atoms['atom_type'], atom_types)
            atoms = atoms[mask]

        return atoms.copy()

    def positions(self, atom_idx=None, atom_types=None):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        return np.atleast_2d(self.atoms(atom_idx, atom_types)['xyz'])

    def closest_atoms_from_positions(self, xyz, radius, atom_types=None, ignore=None):
        """Retrieve indices of the closest atoms around a positions/coordinates 
        at a certain radius.

        Args:
            xyz (np.ndarray): array of 3D coordinates
            radius (float): radius in Angstrom
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)
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

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            index = index.difference([i for i in ignore])

        index = list(index)
        atoms = self._atoms[index]

        # Select atoms only with these atom types
        if atom_types is not None:
            mask = np.isin(atoms['atom_type'], atom_types)
            atoms = atoms[mask]

        return atoms.copy()

    def closest_atoms(self, atom_idx, radius, atom_types=None):
        """Retrieve indices of the closest atoms around a positions/coordinates 
        at a certain radius.

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            radius (float): radius in Angstrom
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        return self.closest_atoms_from_positions(self._atoms[atom_idx]['xyz'], radius, atom_types, atom_idx)

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
