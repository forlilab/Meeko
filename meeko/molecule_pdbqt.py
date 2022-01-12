#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os
from collections import defaultdict

import numpy as np
from scipy import spatial
from rdkit import Chem
from rdkit.Geometry import Point3D

from .utils.covalent_radius_table import covalent_radius
from .utils.autodock4_atom_types_elements import autodock4_atom_types_elements

try:
    from openbabel import openbabel as ob
except ImportError:
    _has_openbabel = False
else:
    _has_openbabel = True


atom_property_definitions = {'H': 'vdw', 'C': 'vdw', 'A': 'vdw', 'N': 'vdw', 'P': 'vdw', 'S': 'vdw',
                             'Br': 'vdw', 'I': 'vdw', 'F': 'vdw', 'Cl': 'vdw',
                             'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc',
                             'HD': 'hb_don', 'HS': 'hb_don',
                             'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal',
                             'MG': 'metal', 'CA': 'metal', 'FE': 'metal', 'ZN': 'metal', 'MN': 'metal',
                             'W': 'water',
                             'G0': 'glue', 'G1': 'glue', 'G2': 'glue', 'G3': 'glue',
                             'CG0': 'glue', 'CG1': 'glue', 'CG2': 'glue', 'CG3': 'glue'}


def _read_ligand_pdbqt_file(pdbqt_string, poses_to_read=-1, energy_range=-1, is_dlg=False, skip_typing=False):
    i = 0
    n_poses = 0
    previous_serial = 0
    tmp_positions = []
    tmp_atoms = []
    tmp_actives = []
    tmp_pdbqt_string = ''
    water_indices = {*()}  
    location = 'ligand'
    energy_best_pose = None
    is_first_pose = True
    is_model = False
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ('xyz', 'f4', (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U3')]

    atoms = None
    positions = []
    atom_annotations = {'ligand': [], 'flexible_residue': [], 'water': [],
                        'hb_acc': [], 'hb_don': [],
                        'all': [], 'vdw': [],
                        'glue': [], 'reactive': [], 'metal': []}
    pose_data = {'n_poses': None, 'active_atoms': [], 'free_energies': [], 
                 'index_map': {}, 'pdbqt_string': [],
                 'smiles': None, 'smiles_index_map': {}, 'smiles_h_parent': []}

    lines = pdbqt_string.split('\n')
    if len(lines[-1]) == 0: lines = lines[:-1]
    lines = [line + '\n' for line in lines]
    for line in lines:
        if is_dlg:
            if line.startswith('DOCKED'):
                line = line[8:]
            else:
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

            # We store water idx separately from the rest since their number can be variable
            if is_first_pose and atom_type != 'W':
                atom_annotations[location].append(i)
                atom_annotations['all'].append(i)
                if not skip_typing:
                    atom_annotations[atom_property_definitions[atom_type]].append(i)

            if atom_type == 'W':
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
            elif line.startswith('REMARK SMILES IDX') and is_first_pose:
                integers = [int(integer) for integer in line.split()[3:]]
                if len(integers) % 2 == 1:
                    raise RuntimeError("Number of indices in SMILES IDX is odd")
                for j in range(int(len(integers) / 2)): 
                    pose_data['smiles_index_map'][integers[j*2]] = integers[j*2 + 1]
            elif line.startswith('REMARK H PARENT') and is_first_pose:
                integers = [int(integer) for integer in line.split()[3:]]
                if len(integers) % 2 == 1:
                    raise RuntimeError("Number of indices in H PARENT is odd")
                for j in range(int(len(integers) / 2)): 
                    pose_data['smiles_h_parent'].append((integers[j*2], integers[j*2 + 1]))
            elif line.startswith('REMARK SMILES') and is_first_pose: # must check after SMILES IDX
                pose_data['smiles'] = line.split()[2]
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
                top1 = atoms[np.isin(atoms['atom_type'], ['W', 'XX'], invert=True)][columns]
                top2 = tmp_atoms[np.isin(atoms['atom_type'], ['W', 'XX'], invert=True)][columns]

                if not np.array_equal(top1, top2):
                    error_msg = 'molecules have different topologies'
                    raise RuntimeError(error_msg)

                # Update information about water molecules (W) as soon as we find new ones
                tmp_water_molecules_idx = tmp_atoms[tmp_atoms['atom_type'] == 'W']['idx']
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
        atom_annotations['water'] = list(water_indices)

    return atoms, positions, atom_annotations, pose_data


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

    def __init__(self, pdbqt_string, name=None, poses_to_read=None, energy_range=None, is_dlg=False, skip_typing=False):
        """PDBQTMolecule class for reading PDBQT (or dlg) files from AutoDock4, AutoDock-GPU or AutoDock-Vina

        Contains both __getitem__ and __iter__ methods, someone might lose his mind because of this.

        Args:
            pdbqt_string (str): pdbqt string
            name (str): name of the molecule (default: None, use filename without pdbqt suffix)
            poses_to_read (int): total number of poses to read (default: None, read all)
            energy_range (float): read docked poses until the maximum energy difference 
                from best pose is reach, for example 2.5 kcal/mol (default: Non, read all)
            is_dlg (bool): input file is in dlg (AutoDock docking log) format (default: False)

        """
        self._current_pose = 0
        self._pdbqt_filename = None
        self._atoms = None
        self._positions = None
        self._bonds = None
        self._atom_annotations = None
        self._pose_data = None
        self._name = name

        # Juice all the information from that PDBQT file
        poses_to_read = poses_to_read if poses_to_read is not None else -1
        energy_range = energy_range if energy_range is not None else -1
        results = _read_ligand_pdbqt_file(pdbqt_string, poses_to_read, energy_range, is_dlg, skip_typing)
        self._atoms, self._positions, self._atom_annotations, self._pose_data = results

        if self._atoms.shape[0] == 0:
            raise RuntimeError('read 0 atoms. Consider PDBQTMolecule.from_file(fname)')

        # Build KDTrees for each pose (search closest atoms by distance)
        self._KDTrees = [spatial.cKDTree(positions) for positions in self._positions]

        # Identify bonds in the ligands
        if not skip_typing:
            mol_atoms = self._atoms[self._atom_annotations['ligand']]
            self._bonds = _identify_bonds(self._atom_annotations['ligand'], mol_atoms['xyz'], mol_atoms['atom_type'])

            """... then in the flexible residues 
            Since we are extracting bonds from docked poses, we might be in the situation
            where the ligand reacted with one the flexible residues and we don't want to 
            consider them as normally bonded..."""
            if self.has_flexible_residues():
                flex_atoms = self._atoms[self._atom_annotations['flexible_residue']]
                self._bonds.update(_identify_bonds(self._atom_annotations['flexible_residue'], flex_atoms['xyz'], flex_atoms['atom_type']))

    @classmethod
    def from_file(cls, pdbqt_filename, name=None, poses_to_read=None, energy_range=None, is_dlg=False, skip_typing=False): 
        if name is None:
            name = os.path.splitext(os.path.basename(pdbqt_filename))[0]
        with open(pdbqt_filename) as f:
            pdbqt_string = f.read()
        instance = cls(pdbqt_string, name, poses_to_read, energy_range, is_dlg, skip_typing) 
        instance._pdbqt_filename = pdbqt_filename
        return instance

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
        repr_str = '<Molecule named %s containing %d poses of %d atoms>'
        return (repr_str % (self._name, self._pose_data['n_poses'], self._atoms.shape[0]))

    @property    
    def name(self):
        """Return the name of the molecule."""
        return self._name

    @property
    def pose_id(self):
        """Return the index of the current pose."""
        return self._current_pose

    @property
    def score(self):
        """Return the score (kcal/mol) of the current pose."""
        return self._pose_data['free_energies'][self._current_pose]

    def available_atom_properties(self, ignore_properties=None):
        """Return all the available atom properties for that molecule.
        
        The following properties are ignored: ligand and flexible_residue

        """
        if ignore_properties is None:
            ignore_properties = []

        if not isinstance(ignore_properties, (list, tuple)):
            ignore_properties = [ignore_properties]

        ignore_properties += ['ligand', 'flexible_residue', 'water']

        return [k for k, v in self._atom_annotations.items() 
                if not k in ignore_properties and len(v) > 0]

    def has_flexible_residues(self):
        """Tell if the molecule contains a flexible residue or not.

        Returns:
            bool: True if contains flexible residues, otherwise False

        """
        if self._atom_annotations['flexible_residue']:
            return True
        else:
            return False

    def has_water_molecules(self):
        """Tell if the molecules contains water molecules or not in the current pose.

        Returns:
            bool: True if contains water molecules in the current pose, otherwise False

        """
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        if set(self._atom_annotations['water']).intersection(active_atoms_idx):
            return True
        else:
            return False

    def atoms(self, atom_idx=None, only_active=True):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=np.int)
        else:
            atom_idx = np.arange(0, self._atoms.shape[0])

        # Get index of only the active atoms
        if only_active:
            active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
            atom_idx = list(set(atom_idx).intersection(active_atoms_idx))

        atoms = self._atoms[atom_idx].copy()
        atoms['xyz'] = self._positions[self._current_pose, atom_idx,:]

        return atoms

    def positions(self, atom_idx=None, only_active=True):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        return np.atleast_2d(self.atoms(atom_idx, only_active)['xyz'])

    def atoms_by_properties(self, atom_properties, only_active=True):
        """Return atom based on their properties

        Args:
            atom_properties (str or list): property of the atoms to retrieve 
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)
            only_active (bool): return only active atoms (default: True, return only active atoms)

        """
        if not isinstance(atom_properties, (list, tuple)):
            atom_properties = [atom_properties]

        if len(atom_properties) > 1:
            try:
                atom_idx = set(self._atom_annotations[atom_properties[0]])

                for atom_property in atom_properties[1:]:
                    atom_idx.intersection_update(self._atom_annotations[atom_property])
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_annotations.keys()))

            atom_idx = list(atom_idx)
        else:
            try:
                atom_idx = self._atom_annotations[atom_properties[0]]
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_properties[0], self._atom_annotations.keys()))

        if atom_idx:
            return self.atoms(atom_idx, only_active)
        else:
            return np.array([])

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
        atom_idx = self._KDTrees[self._current_pose].query_ball_point(xyz, radius, p=2, return_sorted=True)

        # When nothing was found around...
        if not atom_idx:
            return np.array([])

        # Handle the case when positions for of only one atom was passed in the input
        try:
            atom_idx = {i for j in atom_idx for i in j}
        except:
            atom_idx = set(atom_idx)

        if atom_properties is not None:
            if not isinstance(atom_properties, (list, tuple)):
                atom_properties = [atom_properties]

            try:
                for atom_property in atom_properties:
                    atom_idx.intersection_update(self._atom_annotations[atom_property])
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_annotations.keys()))

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            atom_idx = atom_idx.difference([i for i in ignore])

        # Get index of only the active atoms
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        atom_idx = list(set(atom_idx).intersection(active_atoms_idx))

        if atom_idx:
            atoms = self._atoms[atom_idx].copy()
            atoms['xyz'] = self._positions[self._current_pose, atom_idx,:]
            return atoms
        else:
            return np.array([])

    def closest_atoms(self, atom_idx, radius, atom_properties=None):
        """Retrieve indices of the closest atoms around a positions/coordinates 
        at a certain radius.

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            raidus (float): radius
            atom_properties (str or list): property of the atoms to retrieve 
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)

        Returns:
            ndarray: ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if not isinstance(atom_idx, (list, tuple)):
                atom_idx = [atom_idx]

        # Get index of only the active atoms
        active_atoms_idx = self._pose_data['active_atoms'][self._current_pose]
        atom_idx = list(set(atom_idx).intersection(active_atoms_idx))

        if atom_idx:
            positions = self._positions[self._current_pose, atom_idx,:]
            return self.closest_atoms_from_positions(positions, radius, atom_properties, atom_idx)
        else:
            return np.array([])

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
        atom_idx = list(set(atom_idx).intersection(active_atoms_idx))

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
        print(overwrite and os.path.isfile(output_pdbqtfilename))
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
        if not _has_openbabel:
            raise ImportError('openbabel is required')

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

    def export_rdkit_mol(self, exclude_hydrogens=False):
        smiles = self._pose_data['smiles'] # REMARK SMILES
        if smiles is None:
            raise RuntimeError("need REMARK SMILES in PDBQT file")
        index_map = self._pose_data['smiles_index_map'] # REMARK SMILES IDS
        h_parents = self._pose_data['smiles_h_parent'] # REMARK H PARENT
        mol = Chem.MolFromSmiles(smiles)
        n_atoms = mol.GetNumAtoms()
        # https://sourceforge.net/p/rdkit/mailman/message/36474923/
        conf = Chem.Conformer(n_atoms)
        n_matched_atoms = 0
        for i in range(n_atoms):
            pdbqt_index = index_map[i+1] - 1
            x, y, z = self._positions[self._current_pose][pdbqt_index, :]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        conf_id = mol.AddConformer(conf)
        if exclude_hydrogens:
            return mol
        mol = Chem.AddHs(mol, addCoords=True)
        conf = mol.GetConformer()
        used_h = []
        for (parent_rdkit_index, h_pdbqt_index) in h_parents:
            h_pdbqt_index -= 1
            x, y, z = self._positions[self._current_pose][h_pdbqt_index, :]
            parent_atom = mol.GetAtomWithIdx(parent_rdkit_index - 1)
            candidate_hydrogens = [atom.GetIdx() for atom in parent_atom.GetNeighbors() if atom.GetAtomicNum() == 1]
            for h_rdkit_index in candidate_hydrogens:
                if h_rdkit_index not in used_h:
                    break
            used_h.append(h_rdkit_index)
            conf.SetAtomPosition(h_rdkit_index, Point3D(x, y, z))
        return mol
