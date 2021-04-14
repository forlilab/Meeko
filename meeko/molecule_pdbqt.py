#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os
from collections import defaultdict

import numpy as np
from scipy import spatial
from openbabel import openbabel as ob

from .utils.covalent_radius_table import covalent_radius
from .utils.autodock4_atom_types_elements import autodock4_atom_types_elements


def _read_ligand_pdbqt_file(pdbqt_filename, poses_to_read=-1, energy_range=-1, is_dlg=False):
    i = 0
    n_poses = 0
    previous_serial = 0
    tmp_positions = []
    tmp_atoms = []
    tmp_actives = []
    tmp_pdbqt_string = ''
    water_indices = {*()}
    water_atom_type = 'W'
    location = 'ligand'
    energy_best_pose = None
    is_first_pose = True
    is_model = False
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U3')]

    atoms = None
    positions = []
    pose_data = {'ligand': [], 'flexible_residue': [], 'water': [],
                 'n_poses': None, 'active_atoms': [], 'free_energies': [], 
                 'index_map': {}, 'pdbqt_string': []}

    with open(pdbqt_filename) as f:
        lines = f.readlines()

        for line in lines:
            if is_dlg:
                if line.startswith('DOCKED'):
                    line = line[8:]
                else:
                    # If the line does not contain DOCKED, we ignore it
                    continue

            if not line.startswith(('MODEL', 'ENDMDL')):
                """This is very lazy I know...
                But would you rather spend time on rebuilding the whole torsion tree and stuff
                for writing PDBQT files or drinking margarita? Energy was already spend to build
                that, so let's re-use it!"""
                tmp_pdbqt_string += line

            if line.startswith('MODEL'):
                # Reinitialize variables
                i = 0
                previous_serial = 0
                tmp_positions = []
                tmp_atoms = []
                tmp_actives = []
                tmp_pdbqt_string = ''
                is_model = True
            elif line.startswith('ATOM') or line.startswith("HETATM"):
                serial = int(line[6:11].strip())
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chainid = line[21].strip()
                resid = int(line[22:26].strip())
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=float)
                try:
                    # PDBQT files from dry.py script are stripped from their partial charges. sigh...
                    partial_charges = float(line[71:77].strip())
                except:
                    partial_charges = 0.0
                atom_type = line[77:-1].strip()

                """ We are looking for gap in the serial atom numbers. Usually if they
                are not following it means that atoms are missing. This will happen with
                water molecules after using dry.py, only non-overlapping water molecules
                are kept. Also if the current serial becomes suddenly inferior than the
                previous and equal to 1, it means that we are now in another molecule/flexible 
                residue. So here we are adding dummy atoms
                """
                if (previous_serial + 1 != serial) and not (serial < previous_serial and serial == 1):
                    diff = serial - previous_serial - 1
                    for _ in range(diff):
                        xyz_nan = [999.999, 999.999, 999.999]
                        tmp_atoms.append((i, 9999, 'XXXX', 9999, 'XXX', 'X', xyz_nan, 999.999, 'XX'))
                        tmp_positions.append(xyz_nan)
                        i += 1

                # Once it is done, we can return to a normal life... and add existing atoms
                tmp_atoms.append((i, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type))
                tmp_positions.append(xyz)
                tmp_actives.append(i)

                if is_first_pose and atom_type != water_atom_type:
                    pose_data[location].append(i)
                elif atom_type == water_atom_type:
                    # We store water idx separately from the rest since their number can be variable
                    water_indices.update([i])

                previous_serial = serial
                i += 1
            elif line.startswith('REMARK'):
                if line.startswith('REMARK INDEX MAP') and is_first_pose:
                    integers = [int(integer) for integer in line.split()[3:]]

                    if len(integers) % 2 == 1:
                        raise RuntimeError("Number of indices in INDEX MAP is odd")

                    for j in range(int(len(integers) / 2)): 
                        pose_data['index_map'][integers[j*2]] = integers[j*2 + 1]
                elif line.startswith('REMARK VINA RESULT') or line.startswith('USER    Estimated Free Energy of Binding'):
                    # Read free energy from output PDBQT files
                    try:
                        # Vina
                        energy = float(line.split()[3])
                    except:
                        # AD4
                        energy = float(line.split()[7])

                    if energy_best_pose is None:
                        energy_best_pose = energy
                    energy_current_pose = energy

                    diff_energy = energy_current_pose - energy_best_pose
                    if (energy_range <= diff_energy and energy_range != -1):
                        break

                    pose_data['free_energies'].append(energy)
            elif line.startswith('BEGIN_RES'):
                location = 'flexible_residue'
            elif line.startswith('END_RES'):
                # We never know if there is a molecule just after the flexible residue...
                location = 'ligand'
            elif line.startswith('ENDMDL'):
                n_poses += 1
                # After reading the first pose no need to store atom properties
                # anymore, it is the same for every pose
                is_first_pose = False

                tmp_atoms = np.array(tmp_atoms, dtype=atoms_dtype)

                if atoms is None:
                    """We store the atoms (topology) only once, since it is supposed to be
                    the same for all the molecules in the PDBQT file (except when water molecules
                    are involved... classic). But we will continue to compare the topology of
                    the current pose with the first one seen in the PDBQT file, to be sure only
                    the atom positions are changing."""
                    atoms = tmp_atoms.copy()
                else:
                    # Check if the molecule topology is the same for each pose
                    # We ignore water molecules (W) and atom type XX
                    columns = ['idx', 'serial', 'name', 'resid', 'resname', 'chain', 'partial_charges', 'atom_type']
                    top1 = atoms[np.isin(atoms['atom_type'], [water_atom_type, 'XX'], invert=True)][columns]
                    top2 = tmp_atoms[np.isin(atoms['atom_type'], [water_atom_type, 'XX'], invert=True)][columns]

                    if not np.array_equal(top1, top2):
                        error_msg = 'PDBQT file %s does contain molecules with different topologies'
                        raise RuntimeError(error_msg % pdbqt_filename)

                    # Update information about water molecules (W) as soon as we find new ones
                    tmp_water_molecules_idx = tmp_atoms[tmp_atoms['atom_type'] == water_atom_type]['idx']
                    water_molecules_idx = atoms[atoms['atom_type'] == 'XX']['idx']
                    new_water_molecules_idx = list(set(tmp_water_molecules_idx).intersection(water_molecules_idx))
                    atoms[new_water_molecules_idx] = tmp_atoms[new_water_molecules_idx]

                positions.append(tmp_positions)
                pose_data['active_atoms'].append(tmp_actives)
                pose_data['pdbqt_string'].append(tmp_pdbqt_string)

                if (n_poses >= poses_to_read and poses_to_read != -1):
                    break

        """ if there is no model, it means that there is only one molecule
        so when we reach the end of the file, we store the atoms, 
        positions and actives stuff. """
        if not is_model:
            n_poses += 1
            atoms = np.array(tmp_atoms, dtype=atoms_dtype)
            positions.append(tmp_positions)
            pose_data['active_atoms'].append(tmp_actives)
            pose_data['pdbqt_string'].append(tmp_pdbqt_string)

    positions = np.array(positions).reshape((n_poses, atoms.shape[0], 3))

    pose_data['n_poses'] = n_poses

    # We add indices of all the water molecules we saw
    if water_indices:
        pose_data['water'] = list(water_indices)

    return atoms, positions, pose_data


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

    def __init__(self, pdbqt_filename, name=None, poses_to_read=None, energy_range=None, is_dlg=False):
        """PDBQTMolecule class for reading PDBQT (or dlg) files from AutoDock4, AutoDock-GPU or AutoDock-Vina

        Contains both __getitem__ and __iter__ methods, someone might lose his mind because of this.

        Args:
            pdbqt_filename (str): pdbqt filename
            name (str): name of the molecule (default: None, use filename without pdbqt suffix)
            poses_to_read (int): total number of poses to read (default: None, read all)
            energy_range (float): read docked poses until the maximum energy difference 
                from best pose is reach, for example 2.5 kcal/mol (default: Non, read all)
            is_dlg (bool): input file is in dlg (AutoDock docking log) format (default: False)

        """
        self._current_pose = 0
        self._pdbqt_filename = pdbqt_filename
        self._atoms = None
        self._positions = None
        self._bonds = None
        self._pose_data = None
        if name is None:
            self._name = os.path.splitext(os.path.basename(self._pdbqt_filename))[0]
        else:
            self._name = name

        # Juice all the information from that PDBQT file
        poses_to_read = poses_to_read if poses_to_read is not None else -1
        energy_range = energy_range if energy_range is not None else -1
        results = _read_ligand_pdbqt_file(self._pdbqt_filename, poses_to_read, energy_range, is_dlg)
        self._atoms, self._positions, self._pose_data = results

        # Build KDTrees for each pose (search closest atoms by distance)
        self._KDTrees = [spatial.cKDTree(positions) for positions in self._positions]

        # Identify bonds in the ligands
        ligand_atoms = self.ligands()
        self._bonds = _identify_bonds(ligand_atoms['idx'], ligand_atoms['xyz'], ligand_atoms['atom_type'])

        """... then in the flexible residues 
        Since we are extracting bonds from docked poses, we might be in the situation
        where the ligand reacted with one of the flexible residues and we don't want to 
        consider them as normally bonded..."""
        if self.has_flexible_residues():
            flex_atoms = self.flexible_residues()
            self._bonds.update(_identify_bonds(flex_atoms['idx'], flex_atoms['xyz'], flex_atoms['atom_type']))

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

        return self._current_pose

    def __repr__(self):
        repr_str = '<Molecule from PDBQT file %s containing %d poses of %d atoms>'
        return (repr_str % (self._pdbqt_filename, self._pose_data['n_poses'], self._atoms.shape[0]))

    @property    
    def name(self):
        """Return the name of the molecule."""
        return self._name

    @property
    def pose_id(self):
        """Return the index of the current pose."""
        return self._current_pose

    @property
    def number_of_poses(self):
        return self._pose_data['n_poses']
    
    @property
    def score(self):
        """Return the score (kcal/mol) of the current pose."""
        return self._pose_data['free_energies'][self._current_pose]

    def has_ligands(self):
        """Tell if the molecule contains a ligand or not

        People might dock only flexible sidechains and water molecules. We
        are not here to judge, it's a free country.

        """
        if self._pose_data['ligand']:
            return True
        else:
            return False

    def has_flexible_residues(self):
        """Tell if the molecule contains a flexible residue or not.

        Returns:
            bool: True if contains flexible residues, otherwise False

        """
        if self._pose_data['flexible_residue']:
            return True
        else:
            return False

    def has_water_molecules(self):
        """Tell if the molecules contains water molecules or not in the current pose.

        Returns:
            bool: True if contains water molecules in the current pose, otherwise False

        """
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        if set(self._pose_data['water']).intersection(active_atoms_idx):
            return True
        else:
            return False

    def atoms(self, atom_idx=None, atom_types=None, only_active=True):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
        else:
            atom_idx = np.arange(0, self._atoms.shape[0])

        if atom_types is not None:
            if isinstance(atom_types, str):
                atom_types = np.array([atom_types])

        # Get index of only the active atoms
        if only_active:
            active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
            atom_idx = np.array(list(set(atom_idx).intersection(active_atoms_idx)), dtype=int)

        atoms = self._atoms[atom_idx].copy()

        # Select atoms only with these atom types
        if atom_types is not None:
            mask = np.isin(atoms['atom_type'], atom_types)
            atoms = atoms[mask]
            atom_idx = atom_idx[mask]

        atoms['xyz'] = self._positions[self._current_pose, atom_idx,:]

        return atoms

    def positions(self, atom_idx=None, atom_types=None, only_active=True):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        return np.atleast_2d(self.atoms(atom_idx, atom_types, only_active)['xyz'])

    def ligands(self, atom_types=None, positions_only=False):
        """Return ligand atoms
        
        Args:
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None, return all atoms)
            position_only (bool): return only atom positions (default: False)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t) 
                or 2d ndarray of coordinates (xyz) if position_only = True

        """
        if not self.has_ligands():
            dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                     ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                     ('partial_charges', 'f4'), ('atom_type', 'U3')]
            return np.array([], dtype=dtype)

        ligand_atoms_idx = self._pose_data['ligand']
        if positions_only:
            return self.positions(ligand_atoms_idx, atom_types)
        else:
            return self.atoms(ligand_atoms_idx, atom_types)

    def flexible_residues(self, atom_types=None, positions_only=False):
        """Return flexible residues atoms
        
        Args:
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None, return all atom type atoms)
            position_only (bool): return only atom positions (default: False)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t) 
                or 2d ndarray of coordinates (xyz) if position_only = True

        """
        if not self.has_flexible_residues():
            dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                     ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                     ('partial_charges', 'f4'), ('atom_type', 'U3')]
            return np.array([], dtype=dtype)

        ligand_atoms_idx = self._pose_data['flexible_residue']
        if positions_only:
            return self.positions(ligand_atoms_idx, atom_types)
        else:
            return self.atoms(ligand_atoms_idx, atom_types)

    def water_molecules(self, positions_only=False, only_active=True):
        """Return water molecules atoms
        
        Args:
            position_only (bool): return only atom positions (default: False)
            only_active (bool): return only active water molecules (default: True, return only active water molecules)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t) 
                or 2d ndarray of coordinates (xyz) if position_only = True

        """
        if only_active:
            if not self.has_water_molecules():
                dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                         ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                         ('partial_charges', 'f4'), ('atom_type', 'U3')]
                return np.array([], dtype=dtype)

        water_atoms_idx = self._pose_data['water']
        if positions_only:
            return self.positions(water_atoms_idx, only_active=only_active)
        else:
            return self.atoms(water_atoms_idx, only_active=only_active)

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
        dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                 ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                 ('partial_charges', 'f4'), ('atom_type', 'U3')]
        atom_idx = self._KDTrees[self._current_pose].query_ball_point(xyz, radius, p=2, return_sorted=True)

        # When nothing was found around...
        if not atom_idx:
            return np.array([], dtype=dtype)

        # Handle the case when positions for of only one atom was passed in the input
        try:
            atom_idx = {i for j in atom_idx for i in j}
        except:
            atom_idx = set(atom_idx)

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            atom_idx = atom_idx.difference([i for i in ignore])

        # Get index of only the active atoms
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        atom_idx = np.array(list(set(atom_idx).intersection(active_atoms_idx)), dtype=int)

        if atom_idx.size > 0:
            atoms = self._atoms[atom_idx].copy()

            # Select atoms only with these atom types
            if atom_types is not None:
                mask = np.isin(atoms['atom_type'], atom_types)
                atoms = atoms[mask]
                atom_idx = atom_idx[mask]

            atoms['xyz'] = self._positions[self._current_pose, atom_idx,:]
            return atoms
        else:
            return np.array([], dtype=dtype)

    def closest_atoms(self, atom_idx, radius, atom_types=None):
        """Retrieve indices of the closest atoms around a positions/coordinates 
        at a certain radius.

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            radius (float): radius in Angstrom
            atom_types (str, list of str): AutoDock atom type or list or atom types (default: None)

        Returns:
            ndarray: ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if not isinstance(atom_idx, (list, tuple)):
                atom_idx = [atom_idx]

        # Get index of only the active atoms
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        atom_idx = np.array(list(set(atom_idx).intersection(active_atoms_idx)), dtype=int)

        if atom_idx.size > 0:
            positions = self._positions[self._current_pose, atom_idx,:]
            return self.closest_atoms_from_positions(positions, radius, atom_types, atom_idx)
        else:
            return np.array([], dtype=self._atoms_dtype)

    def neighbor_atoms(self, atom_idx):
        """Return neighbor (bonded) atoms

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            list_of_list: list of lists containing the neighbor (bonded) atoms (0-based)

        """
        if not isinstance(atom_idx, (list, tuple, np.ndarray)):
            atom_idx = [atom_idx]

        # Get index of only the active atoms
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        atom_idx = np.array(list(set(atom_idx).intersection(active_atoms_idx)), dtype=int)

        return [self._bonds[i] for i in atom_idx]

    def write_pdbqt_string(self, as_model=True):
        """Write PDBQT output string of the current pose
        
        Args:
            as_model (bool): Qdd MODEL/ENDMDL keywords to the output PDBQT string (default: True)

        """
        if as_model:
            pdbqt_string = 'MODEL    %5d\n' % (self._current_pose + 1)
            pdbqt_string += self._pose_data['pdbqt_string'][self._current_pose] 
            pdbqt_string += 'ENDMDL\n'
            return pdbqt_string
        else: 
            return self._pose_data['pdbqt_string'][self._current_pose]

    def write_pdbqt_file(self, output_pdbqtfilename, overwrite=False, as_model=False):
        """Write PDBQT file of the current pose

        Args:
            output_pdbqtfilename (str): filename of the output PDBQT file
            overwrite (bool): overwrite on existing pdbqt file (default: False)
            as_model (bool): Qdd MODEL/ENDMDL keywords to the output PDBQT string (default: False)

        """
        if not overwrite and os.path.isfile(output_pdbqtfilename):
            raise RuntimeError('Output PDBQT file %s already exists' % output_pdbqtfilename)

        if as_model:
            pdbqt_string = 'MODEL    %5d\n' % (self._current_pose + 1)
            pdbqt_string += self._pose_data['pdbqt_string'][self._current_pose] 
            pdbqt_string += 'ENDMDL\n'
        else:
            pdbqt_string = self._pose_data['pdbqt_string'][self._current_pose]

        with open(output_pdbqtfilename, 'w') as w:
            w.write(pdbqt_string)

    def copy_coordinates_to_obmol(self, obmol, index_map=None):
        """Copy coordinates of the current pose to an obmol object 

        Args:
            obmol (OBMol): coordinates will be changed in this object
            index_map (dict): map of atom indices from obmol (keys) to coords (values) (Default: None)

        """
        if index_map is None:
            index_map = self._pose_data['index_map']

        n_atoms = obmol.NumAtoms()
        n_matched_atoms = 0
        hydrogens_to_delete = []
        heavy_parents = []

        for atom in ob.OBMolAtomIter(obmol):
            ob_index = atom.GetIdx() # 1-index

            if ob_index in index_map:
                pdbqt_index = index_map[ob_index] - 1
                x, y, z = self._positions[self._current_pose][pdbqt_index, :]
                atom.SetVector(x, y, z)
                n_matched_atoms += 1
            elif atom.GetAtomicNum() != 1:
                raise RuntimeError('Heavy atom in OBMol is missing, only hydrogens can be missing')
            else:
                hydrogens_to_delete.append(atom)
                bond_counter = 0

                for bond in ob.OBAtomBondIter(atom):
                    bond_counter += 1
                if bond_counter != 1:
                    raise RuntimeError('Hydrogen atom has more than one bonds (%d bonds)' % bond_counter)

                begin_atom = bond.GetBeginAtom()
                end_atom = bond.GetEndAtom()

                if atom == begin_atom:
                    heavy_parents.append(end_atom)
                elif atom == end_atom:
                    heavy_parents.append(begin_atom)
                else:
                    raise RuntimeError('Hydrogen isn\'t either Begin or End atom of its own bond')

        if n_matched_atoms != len(index_map):
            raise RuntimeError('Not all the atoms were considered')

        # delete explicit hydrogens
        for hydrogen in hydrogens_to_delete:
            obmol.DeleteHydrogen(hydrogen)

        # increment implicit H count of heavy atom parents
        for heavy_parent in heavy_parents:
            n_implicit = heavy_parent.GetImplicitHCount()
            heavy_parent.SetImplicitHCount(n_implicit + 1)

        # add back explicit hydrogens
        obmol.AddHydrogens()
        if obmol.NumAtoms() != n_atoms:
            raise RuntimeError('Number of atoms changed after deleting and adding hydrogens')

