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


atom_property_definitions = {'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc', 
                             'HD': 'hb_don', 'HS': 'hb_don',
                             'Cl': 'non-metal', 
                             'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal',
                             'W': 'water',
                             'G0': 'glue', 'G1': 'glue', 'G2': 'glue', 'G3': 'glue', 
                             'CG0': 'glue', 'CG1': 'glue', 'CG2': 'glue', 'CG3': 'glue'}


def _read_ligand_pdbqt_file(pdbqt_filename, poses_to_read=None):
    i = 0
    atoms = None
    positions = []
    n_poses = 0
    location = 'ligand'
    store_atom_properties = True
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U2')]
    atom_properties = {'ligand': [], 'flexible_residue': [],
                       'all': [], 'vdw': [], 'hb_acc': [], 'hb_don': [], 
                       'water': [], 'glue': [], 'reactive': []}

    with open(pdbqt_filename) as f:
        lines = f.readlines()

        for line in lines:
            if line.startswith('MODEL'):
                i = 0
                n_poses += 1

                # Check if the molecule topology is the same for each pose
                if atoms is not None:
                    columns = ['idx', 'serial', 'name', 'resid', 'resname', 'chain', 'partial_charges', 'atom_type']
                    if not np.array_equal(atoms[columns], tmp_atoms[columns]):
                        error_msg = 'PDBQT file %s does contain molecules with different topologies'
                        raise RuntimeError(error_msg % pdbqt_filename)

                tmp_positions = []
                tmp_atoms = []
            elif line.startswith('ATOM') or line.startswith("HETATM"):
                idx = i
                serial = int(line[6:11].strip())
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chainid = line[21].strip()
                resid = int(line[22:26].strip())
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=float)
                partial_charges = float(line[71:77].strip())
                atom_type = line[77:79].strip()

                if store_atom_properties:
                    atom_properties[location].append(i)
                    atom_properties['all'].append(i)
                    try:
                        atom_properties[atom_property_definitions[atom_type]].append(i)
                    except:
                        atom_properties['vdw'].append(i)

                tmp_atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type))
                tmp_positions.append(xyz)
                i += 1
            elif line.startswith('BEGIN_RES'):
                location = 'flexible_residue'
            elif line.startswith('END_RES'):
                # We never know if there is a molecule just after the flexible residue...
                location = 'ligand'
            elif line.startswith('ENDMDL'):
                # After reading the first pose no need to store atom properties 
                # anymore, it is the same for every pose
                store_atom_properties = False

                """We store the atoms (topology) only once, since 
                it is supposed to be the same for all the molecules in the PDBQT file.
                But we will continue to compare the topology of the current pose with the 
                first one seen in the PDBQT file, to be sure only the atom positions are
                changing."""
                tmp_atoms = np.array(tmp_atoms, dtype=atoms_dtype)
                if atoms is None:
                    atoms = tmp_atoms.copy()

                positions.append(tmp_positions)

                if n_poses >= poses_to_read and poses_to_read != -1:
                    break

    positions = np.array(positions).reshape((n_poses, atoms.shape[0], 3))

    return atoms, atom_properties, positions


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


class PDBQTMolecule:

    def __init__(self, pdbqt_filename, poses_to_read=None):
        """PDBQTMolecule object

        Contains both __getitem__ and __iter__ methods, someone might lose his hair because of this.

        Args:
            pdbqt_filename (str): pdbqt filename
            poses_to_read (int): total number of poses to read (default: None, read all)

        """
        self._current_pose = 0
        self._poses_to_read = poses_to_read if poses_to_read is not None else -1
        self._pdbqt_filename = pdbqt_filename
        self._atoms = None
        self._atom_properties = None
        self._positions = None
        self._name = os.path.splitext(os.path.basename(self._pdbqt_filename))[0]

        # Juice all the information from that PDBQT file
        self._atoms, self._atom_properties, self._positions = _read_ligand_pdbqt_file(self._pdbqt_filename, self._poses_to_read)

        # Build KDTrees for each pose
        self._KDTrees = [spatial.cKDTree(positions) for positions in self._positions]

        # Identify bonds in the ligands
        mol_atoms = self._atoms[self._atom_properties['ligand']]
        self._bonds = _identify_bonds(self._atom_properties['ligand'], mol_atoms['xyz'], mol_atoms['atom_type'])

        """... then in the flexible residues 
        Since we are extracting bonds from docked poses, we might be in the situation
        where the ligand reacted with one the flexible residues and we don't want to 
        consider them as normally bonded..."""
        if self.has_flexible_residues():
            flex_atoms = self._atoms[self._atom_properties['flexible_residue']]
            self._bonds.update(_identify_bonds(self._atom_properties['flexible_residue'], flex_atoms['xyz'], flex_atoms['atom_type']))

    def __getitem__(self, value):
        if isinstance(value, int):
            if value < 0 or value >= self._positions.shape[0]:
                raise IndexError('The index (%d) is out of range.' % value)
        elif isinstance(value, slice):
            raise TypeError('Slicing is not implemented for PDBQTMolecule object.')
        else:
            raise TypeError('Invalid argument type.')

        self._current_pose = value
        return self

    def __iter__(self):
        self._current_pose -= 1
        return self

    def __next__(self):
        if self._current_pose + 1 >= self._positions.shape[0]:
            raise StopIteration

        self._current_pose += 1

        return self

    def __repr__(self):
        repr_str = '<Molecule from PDBQT file %s containing %d poses of %d atoms>'
        return (repr_str % (self._pdbqt_filename, self._positions.shape[0], self._atoms.shape[0]))

    @property    
    def name(self):
        return self._name

    @property
    def pose_id(self):
        return self._current_pose

    def available_atom_properties(self):
        return [k for k, v in self._atom_properties.items() if not k in ['ligand', 'flexible_residue'] and len(v) > 0]

    def has_flexible_residues(self):
        if self._atom_properties['flexible_residue']:
            return True
        return False

    def positions(self, atom_idx=None):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        if atom_idx is not None and self._positions.size > 1:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
            positions = self._positions[self._current_pose, atom_idx,:]
        else:
            positions = self._positions[self._current_pose,:,:]

        return np.atleast_2d(positions).copy()

    def atoms(self, atom_idx=None):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None and self._positions.size > 1:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
            atoms = self._atoms[atom_idx].copy()
            atoms['xyz'] = self._positions[self._current_pose, atom_idx,:]
        else:
            atoms = self._atoms.copy()
            atoms['xyz'] = self._positions[self._current_pose,:,:]

        return atoms

    def atoms_by_properties(self, atom_properties):
        """Return atom based on their properties

        Args:
            atom_properties (str or list): property of the atoms to retrieve 
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)

        """
        if not isinstance(atom_properties, (list, tuple)):
            atom_properties = [atom_properties]

        if len(atom_properties) > 1:
            try:
                index = set(self._atom_properties[atom_properties[0]])

                for atom_property in atom_properties[1:]:
                    index.intersection_update(self._atom_properties[atom_property])
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_properties.keys()))

            index = list(index)
        else:
            try:
                index = self._atom_properties[atom_properties[0]]
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_properties.keys()))

        selected_atoms = self._atoms[index].copy()
        if self._current_pose != 0:
            selected_atoms['xyz'] = self._positions[self._current_pose, index,:]

        return selected_atoms

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
        index = self._KDTrees[self._current_pose].query_ball_point(xyz, radius, p=2, return_sorted=True)

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
                    index.intersection_update(self._atom_properties[atom_property])
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_properties.keys()))

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            index = index.difference([i for i in ignore])

        index = list(index)
        atoms = self._atoms[index].copy()
        atoms['xyz'] = self._positions[self._current_pose, index,:]

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
