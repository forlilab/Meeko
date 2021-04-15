#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko - Interactions
#

import os
from abc import ABC, abstractmethod

import numpy as np


def _compute_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


class _Interaction(ABC):
    """Abstract class for molecular interactions"""
    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def find(self, **kwargs):
        pass


class _DistanceBased(_Interaction):
    """Generic class for interactions (hydrohpbic, metal, reactive, etc...) based on distance only"""
    def __init__(self, distance, lig_atom_types, rec_atom_types):
        assert distance > 0, 'Distance must be superior than 0 Angstrom'
        assert len(lig_atom_types) > 0, 'ligand atom types must be defined'
        assert len(rec_atom_types) > 0, 'receptor atom types must be defined'

        self._distance = distance
        self._lig_atom_types = lig_atom_types
        self._rec_atom_types = rec_atom_types

    def __repr__(self):
        repr_str = '- Name: %s\n' % self.name
        repr_str += '    type: Distance based interaction\n'
        repr_str += '    distance: %3.1f\n' % self._distance
        repr_str += '    ligand atom types  : %s\n' % self._lig_atom_types
        repr_str += '    receptor atom types: %s\n' % self._rec_atom_types
        return repr_str

    @property
    def name(self):
        return type(self).__name__

    def find(self, molecule, receptor):
        """Find distance-based interactions between ligand and receptor.

        Args:
            molecule (PDBQTMolecule): molecule
            receptor (PDBQTReceptor): receptor

        Returns:
            np.ndarray: array of (ligand_atom_index, receptor_atom_index, distance)
                for each interaction between ligand and rigid receptor
            np.ndarray: array of (ligand_atom_index, receptor_atom_index, distance)
                for each interaction between ligand and flexible sidechain

        """
        dtype = [('ligand_idx', int), ('receptor_idx', int), ('distance', float)]
        rigid_interactions = [] 
        flex_interactions = []
        has_flexible_residues = molecule.has_flexible_residues()
        lig_atoms = molecule.ligands(self._lig_atom_types)

        """We are going to ignore ligand atoms and water molecules
        when searching for ligand -- flexible receptor interactions"""
        to_ignore = list(lig_atoms['idx'])
        if molecule.has_water_molecules():
            to_ignore += list(molecule.water_molecules()['idx'])

        for lig_atom in lig_atoms:
            # Get interactions with the rigid part of the receptor
            rec_atoms = receptor.closest_atoms_from_positions(lig_atom['xyz'], self._distance, self._rec_atom_types)
            rigid_interactions.extend((lig_atom['idx'], rec_atom['idx'], np.linalg.norm(lig_atom['xyz'] - rec_atom['xyz'])) for rec_atom in rec_atoms)

            # If present, get interactions with the flexible sidechain atoms (part of the receptor)
            # But we ignore the ligand itself
            if has_flexible_residues:
                rec_atoms = molecule.closest_atoms_from_positions(lig_atom['xyz'], self._distance, self._rec_atom_types, to_ignore)
                flex_interactions.extend((lig_atom['idx'], rec_atom['idx'], np.linalg.norm(lig_atom['xyz'] - rec_atom['xyz'])) for rec_atom in rec_atoms)

        return np.array(rigid_interactions, dtype=dtype), np.array(flex_interactions, dtype=dtype)


class Hydrophobic(_DistanceBased):
    def __init__(self, distance=4.0,
                 lig_atom_types=['NA', 'OA', 'SA', 'OS', 'NS', 'C', 'A', 'N', 'P', 'S', 'Br', 'I', 'F', 'Cl'],
                 rec_atom_types=['NA', 'OA', 'SA', 'OS', 'NS', 'C', 'A', 'N', 'P', 'S', 'Br', 'I', 'F', 'Cl']):
        """Hydrophobic interactions

        Args:
            distance (float): distance cutoff between ligand and receptor/flexible sidechain atoms (default: 4.0)
            lig_atom_types (list of str): list of ligand hydrophobic AutoDock atom types
                (default: ['NA', 'OA', 'SA', 'OS', 'NS', 'C', 'A', 'N', 'P', 'S', 'Br', 'I', 'F', 'Cl'])
            rec_atom_types (list of str): list of receptor hydrophobic AutoDock atom types
                (default: ['NA', 'OA', 'SA', 'OS', 'NS', 'C', 'A', 'N', 'P', 'S', 'Br', 'I', 'F', 'Cl'])

        """
        super().__init__(distance, lig_atom_types, rec_atom_types)


class Reactive(_DistanceBased):
    def __init__(self, distance=2.0,
                 lig_atom_types=['C1'],
                 rec_atom_types=['S4']):
        """Reactive interaction between ligand and receptor (flexible sidechain).

        Args:
            distance (float): distance cutoff between ligand and reactive (flexible) sidechain atoms (default: 2.0)
            lig_atom_types (list of str): list of ligand reactive AutoDock atom types
                (default: ['C1'])
            rec_atom_types (list of str): list of receptor reactive AutoDock atom types
                (default: ['S4'])

        """
        super().__init__(distance, lig_atom_types, rec_atom_types)


class Metal(_DistanceBased):
    def __init__(self, distance=3.0,
                 lig_atom_types=['NA', 'OA', 'SA', 'OS', 'NS'],
                 rec_atom_types=['Mg', 'Ca', 'Fe', 'Zn', 'Mn', 'MG', 'CA', 'FE', 'ZN', 'MN']):
        """Metal interaction between the ligand and metals in the receptor.
        
        Args:
            distance (float): distance cutoff between ligand and metal receptor atoms (default: 3.0)
            lig_atom_types (list of str): list of ligand AutoDock atom types
                (default: ['NA', 'OA', 'SA', 'OS', 'NS'])
            rec_atom_types (list of str): list of receptor metal AutoDock atom types
                (default: ['Mg', 'Ca', 'Fe', 'Zn', 'Mn'])

        """
        super().__init__(distance, lig_atom_types, rec_atom_types)


class _HBBased(_Interaction):
    """Generic class for hydrogen bond-like interactions"""
    def __init__(self, distance, angles, lig_atom_types, rec_atom_types, hb_type):
        assert distance > 0, 'Distance must be superior than 0 Angstrom'
        assert len(angles) == 2, 'An array-like of two angles must be defined'
        assert len(lig_atom_types) > 0, 'ligand atom types must be defined'
        assert len(rec_atom_types) > 0, 'receptor atom types must be defined'
        assert hb_type in ['donor', 'acceptor', 'both'], 'lig HBond can be only donor, acceptor or both'

        self._distance = distance
        self._angles = [np.radians(angle) for angle in angles]
        self._lig_atom_types = lig_atom_types
        self._rec_atom_types = rec_atom_types
        self._hb_type = hb_type

    def __repr__(self):
        repr_str = '- Name: %s\n' % self.name
        repr_str += '    type: HBond based interaction\n'
        repr_str += '    distance: %3.1f\n' % self._distance
        repr_str += '    angle:\n'
        if self._hb_type in ['donor', 'both']:
            angle_1, angle_2 = np.degrees(self._angles)
        else:
            angle_2, angle_1 = np.degrees(self._angles)
        repr_str += '        - Donor-H -- acceptor angle       : %5.2f\n' % angle_1
        repr_str += '        - Pre_acceptor-acceptor -- H angle: %5.2f\n' % angle_2
        repr_str += '    ligand atom types  : %s\n' % self._lig_atom_types
        repr_str += '    receptor atom types: %s\n' % self._rec_atom_types
        return repr_str

    @property
    def name(self):
        return type(self).__name__

    def find(self, molecule, receptor):
        """Find Hydrogen bond-like interactions between ligand and receptor.

        Args:
            molecule (PDBQTMolecule): molecule
            receptor (PDBQTReceptor): receptor

        Returns:
            np.ndarray: array of (ligand_atom_index, receptor_atom_index, distance, donor_angle, acceptor_angle)
                for each interaction between ligand and rigid receptor
            np.ndarray: array of (ligand_atom_index, receptor_atom_index, distance, donor_angle, acceptor_angle)
                for each interaction between ligand and flexible sidechain

        """
        dtype = [('ligand_idx', int), ('receptor_idx', int), 
                 ('distance', float), ('angle_don', float), ('angle_acc', float)]
        rigid_interactions = []
        flex_interactions = []
        lig_hb_pre_position = None
        rec_hb_pre_position = None
        has_flexible_residues = molecule.has_flexible_residues()

        lig_atoms = molecule.ligands(self._lig_atom_types)
        if molecule.has_water_molecules():
            lig_atoms = np.concatenate((lig_atoms, molecule.water_molecules()))

        """We are going to ignore ligand atoms
        when searching for ligand -- flexible/water interactions"""
        to_ignore = list(lig_atoms['idx'])

        for lig_atom in lig_atoms:
            # Dirty trick to avoid looking at theattached heavy atom
            # ... and we assume that donor atoms are all going to be hydrogen atoms
            if lig_atom['atom_type'][0].upper() == 'H':
                max_distance = self._distance - 1.0
            else:
                max_distance = self._distance

            # Get pre-acceptor position (if acceptor) or pre-hydrogen position (if donor) for that atom in the ligand
            lig_bound_atoms_index = molecule.neighbor_atoms(lig_atom['idx'])
            if any(lig_bound_atoms_index):
                lig_bound_atoms = molecule.atoms(lig_bound_atoms_index[0])
                # This is not accurate when bonds don't have the same length
                lig_hb_pre_position = np.mean(lig_bound_atoms['xyz'], axis=0)
            else:
                # If no atom bound, likely a water molecule attached to the ligand
                lig_hb_pre_position = None

            # Get interactions with the rigid part of the receptor
            rec_rigid_atoms = receptor.closest_atoms_from_positions(lig_atom['xyz'], max_distance, self._rec_atom_types)
            rec_rigid_flex = [receptor]
            rec_rigid_flex_atoms = [rec_rigid_atoms]
            rec_rigid_flex_types = ['rigid']

            # If present, get interactions with the flexible sidechain atoms (part of the receptor)
            # But we ignore the ligand itself
            if has_flexible_residues:
                rec_flex_atoms = molecule.closest_atoms_from_positions(lig_atom['xyz'], max_distance, self._rec_atom_types, to_ignore)
                rec_rigid_flex.append(molecule)
                rec_rigid_flex_atoms.append(rec_flex_atoms)
                rec_rigid_flex_types.append('flex')

            # Add interactions
            for rec, rec_atoms, rec_type in zip(rec_rigid_flex, rec_rigid_flex_atoms, rec_rigid_flex_types):
                for rec_atom in rec_atoms:
                    # Dirty trick to avoid looking at the attached heavy atom
                    # ... and we assume that donor atoms are all going to be hydrogen atoms
                    if rec_atom['atom_type'][0].upper() == 'H':
                        max_distance = self._distance - 1.0
                    else:
                        max_distance = self._distance

                    distance = np.linalg.norm(lig_atom['xyz'] - rec_atom['xyz'])

                    if distance <= max_distance:
                        # Get pre-acceptor position (if acceptor) or pre-hydrogen position (if donor) for that atom in the ligand
                        rec_bound_atoms_index = rec.neighbor_atoms(rec_atom['idx'])
                        if any(rec_bound_atoms_index):
                            rec_bound_atoms = rec.atoms(rec_bound_atoms_index[0])
                            # This is not accurate when bonds don't have the same length
                            rec_hb_pre_position = np.mean(rec_bound_atoms['xyz'], axis=0)
                        else:
                            # If no atom bound, likely a water molecule in the receptor
                            rec_hb_pre_position = None

                        if lig_hb_pre_position is not None:
                            angle_1 = _compute_angle(lig_atom['xyz'] - lig_hb_pre_position, lig_atom['xyz'] - rec_atom['xyz'])
                        else:
                            angle_1 = np.radians(180)

                        if rec_hb_pre_position is not None:
                            angle_2 = _compute_angle(rec_atom['xyz'] - rec_hb_pre_position, rec_atom['xyz'] - lig_atom['xyz'])
                        else:
                            angle_2 = np.radians(180)

                        if (angle_1 >= self._angles[0]) & (angle_2 >= self._angles[1]):
                            tmp = [lig_atom['idx'], rec_atom['idx'], distance]

                            if self._hb_type in ['donor', 'both']:
                                tmp += [np.degrees(angle_1), np.degrees(angle_2)]
                            else:
                                tmp += [np.degrees(angle_2), np.degrees(angle_1)]

                            if rec_type == 'rigid':
                                rigid_interactions.append(tuple(tmp))
                            else:
                                flex_interactions.append(tuple(tmp))

        return np.array(rigid_interactions, dtype=dtype), np.array(flex_interactions, dtype=dtype)


class HBDonor(_HBBased):
    def __init__(self, distance=3.5, angles=(120, 90), 
                 lig_atom_types=['HD', 'HS'],
                 rec_atom_types=['NA', 'OA', 'SA', 'OS', 'NS']):
        """Hydrogen bond interaction between the ligand (donor) and the receptor (acceptor)
        
        Args:
            distance (float): distance cutoff between donor and acceptor atoms (default: 3.5)
            angles (array-like): angles between donor and acceptor atoms (default: (120, 90))
                Donor-H -- acceptor angle              (angle_1)
                Pre_acceptor-acceptor -- H angle       (angle_2)
            lig_atom_types (list of str): list of ligand HB donor AutoDock atom types
                (default: ['HD', 'HS'])
            rec_atom_types (list of str): list of receptor HB acceptor AutoDock atom types
                (default: ['NA', 'OA', 'SA', 'OS', 'NS'])

        """
        # Because the distance will be between the acceptor and the hydrogen atoms
        super().__init__(distance, angles, lig_atom_types, rec_atom_types, 'donor')


class HBAcceptor(_HBBased):
    def __init__(self, distance=3.5, angles=(120, 90),
                 lig_atom_types=['NA', 'OA', 'SA', 'OS', 'NS'],
                 rec_atom_types=['HD', 'HS']):
        """Hydrogen bond interaction between the ligand (acceptor) and the receptor (donor)
        
        Args:
            distance (float): distance cutoff between donor and acceptor atoms (default: 3.5)
            angles (array-like): angles between donor and acceptor atoms (default: (120, 90))
                Donor-H -- acceptor angle              (angle_1)
                Pre_acceptor-acceptor -- H angle       (angle_2)
            lig_atom_types (list of str): list of ligand HB donor AutoDock atom types
                (default: ['NA', 'OA', 'SA', 'OS', 'NS'] )
            rec_atom_types (list of str): list of receptor HB acceptor AutoDock atom types
                (default: ['HD', 'HS'])

        """
        # Because the distance will be between the acceptor and the hydrogen atoms
        super().__init__(distance, angles[::-1], lig_atom_types, rec_atom_types, 'acceptor')


class Water(_HBBased):
    def __init__(self, distance=3.5, angles=(120, 90),
                 lig_atom_types=['W'],
                 rec_atom_types=['HD', 'HS', 'NA', 'OA', 'SA', 'OS', 'NS']):
        """Interaction between donor/acceptor water molecules (attached to the ligand) and receptor (donor/acceptor)
        
        Args:
            distance (float): distance cutoff between donor water molecule and acceptor atoms (default: 3.2)
            angle (float): angle between donor water molecule and acceptor atoms (default: (120, 90))
                Donor-H -- W_acceptor angle              (angle_1)
                Pre_acceptor-acceptor -- W_donor angle   (angle_2)
            lig_atom_types (list of str): list of water AutoDock atom types
                (default: ['W'])
            rec_atom_types (list of str): list of receptor HB acceptor AutoDock atom types
                (default: ['HD', 'HS', 'NA', 'OA', 'SA', 'OS', 'NS'])

        """
        super().__init__(distance, angles, lig_atom_types, rec_atom_types, 'both')


class WaterDonor(_HBBased):
    def __init__(self, distance=3.5, angle=90,
                 lig_atom_types=['W'],
                 rec_atom_types=['NA', 'OA', 'SA', 'OS', 'NS']):
        """Interaction between donor water molecules (attached to the ligand) and receptor (acceptor)
        
        Args:
            distance (float): distance cutoff between donor water molecule and acceptor atoms (default: 3.2)
            angle (float): angle between donor water molecule and acceptor atoms (default: 90)
                W_donor -- Pre_acceptor-acceptor angle
            lig_atom_types (list of str): list of water AutoDock atom types
                (default: ['W'])
            rec_atom_types (list of str): list of receptor HB acceptor AutoDock atom types
                (default: ['NA', 'OA', 'SA', 'OS', 'NS'])

        """
        super().__init__(distance, (0, angle), lig_atom_types, rec_atom_types, 'donor')


class WaterAcceptor(_HBBased):
    def __init__(self, distance=3.5, angle=120,
                 lig_atom_types=['W'],
                 rec_atom_types=['HD', 'HS']):
        """Interaction between acceptor water molecules (attached to the ligand) and receptor (donor)
        
        Args:
            distance (float): distance cutoff between donor water molecule and acceptor atoms (default: 3.2)
            angle (float): angle between donor water molecule and acceptor atoms (default: 120)
                W_acceptor -- Donor-H angle
            lig_atom_types (list of str): list of water AutoDock atom types
                (default: ['W'])
            rec_atom_types (list of str): list of receptor HB donor AutoDock atom types
                (default: ['HD', 'HS'])

        """
        # Because the distance will be between the acceptor and the hydrogen atoms
        super().__init__(distance, (0, angle), lig_atom_types, rec_atom_types, 'acceptor')
