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
    def find(self, **kwargs):
        pass

    @abstractmethod
    def name(self):
        pass


class _DistanceBased(_Interaction):
    """Generic class for interactions (hydrohpbic, metal, reactive, etc...) based on distance only"""
    def __init__(self, distance, atom_properties):
        assert distance > 0, 'Distance must be superior than 0 Angstrom'
        assert len(atom_properties) == 2, 'An array-like of two atom properties must be defined'

        self._distance = distance
        self._atom_properties = atom_properties

    @property
    def name(self):
        return type(self).__name__

    def find(self, ligand, receptor):
        """Find distance-based interactions between ligand and receptor.

        Args:
            ligand (PDBQTMolecularr): ligand
            receptor (PDBQTReceptor): receptor

        Returns:
            np.ndarray: array of (ligand_atom_index, receptor_atom_index, distance)
                for each interaction between ligand and rigid receptor
            np.ndarray: array of (ligand_atom_index, receptor_atom_index, distance)
                for each interaction between ligand and flexible sidechain

        """
        dtype = [('ligand_idx', int), ('receptor_idx', int), 
                 ('distance', float)]
        rigid_interactions = [] 
        flex_interactions = []
        has_flexible_residues = ligand.has_flexible_residues()
        lig_atoms = ligand.atoms_by_properties(['ligand', self._atom_properties[0]])

        for lig_atom in lig_atoms:
            # Get interactions with the rigid part of the receptor
            rec_atoms = receptor.closest_atoms_from_positions(lig_atom['xyz'], self._distance, self._atom_properties[1])
            rigid_interactions.extend((lig_atom['idx'], rec_atom['idx'], np.linalg.norm(lig_atom['xyz'] - rec_atom['xyz'])) for rec_atom in rec_atoms)

            # Get interactions with the flexible part of the receptor (if present)
            if has_flexible_residues:
                rec_atoms = ligand.closest_atoms_from_positions(lig_atom['xyz'], self._distance, ['flexible_residue', self._atom_properties[1]])
                flex_interactions.extend((lig_atom['idx'], rec_atom['idx'], np.linalg.norm(lig_atom['xyz'] - rec_atom['xyz'])) for rec_atom in rec_atoms)

        return np.array(rigid_interactions, dtype=dtype), np.array(flex_interactions, dtype=dtype)


class Hydrophobic(_DistanceBased):
    def __init__(self, distance=4.5):
        """Hydrophobic interactions

        Args:
            distance (float): distance cutoff between ligand and receptor/flexible sidechain atoms (default: 4.5)

        """
        super().__init__(distance, ['vdw', 'all'])


class Reactive(_DistanceBased):
    def __init__(self, distance=2.0):
        """Reactive interaction between ligand and receptor (flexible sidechain).

        Args:
            distance (float): distance cutoff between ligand and reactive (flexible) sidechain atoms (default: 2.0)

        """
        super().__init__(distance, ['reactive', 'reactive'])


class Metal(_DistanceBased):
    def __init__(self, distance=3.0):
        """Metal interaction between the ligand and metals in the receptor.
        
        Args:
            distance (float): distance cutoff between ligand and metal receptor atoms (default: 3.0)

        """
        super().__init__(distance, ['hb_acc', 'metal'])


class _HBBased(_Interaction):
    """Generic class for hydrogen bond-like interactions"""
    def __init__(self, distance, angles, atom_properties):
        assert distance > 0, 'Distance must be superior than 0 Angstrom'
        assert len(angles) == 2, 'An array-like of two angles must be defined'
        assert len(atom_properties) == 2, 'An array-like of two atom properties must be defined'

        self._distance = distance
        self._angles = [np.radians(angle) for angle in angles]
        self._atom_properties = atom_properties

    @property
    def name(self):
        return type(self).__name__

    def find(self, ligand, receptor):
        """Find Hydrogen bond-like interactions between ligand and receptor.

        Args:
            ligand (PDBQTMolecularr): ligand
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
        lig_atom_property, rec_atom_property = self._atom_properties

        has_flexible_residues = ligand.has_flexible_residues()

        if lig_atom_property == 'water':
            lig_atoms = ligand.atoms_by_properties([lig_atom_property])
        else:
            lig_atoms = ligand.atoms_by_properties(['ligand', lig_atom_property])

        for lig_atom in lig_atoms:
            # Get pre-acceptor position (if acceptor) or pre-hydrogen position (if donor) for that atom in the ligand
            lig_bound_atoms_index = ligand.neighbor_atoms(lig_atom['idx'])
            if any(lig_bound_atoms_index):
                lig_bound_atoms = ligand.atoms(lig_bound_atoms_index[0])
                # This is not accurate when bonds don't have the same length
                lig_hb_pre_position = np.mean(lig_bound_atoms['xyz'], axis=0)
            else:
                # If no atom bound, likely a water molecule attached to the ligand
                lig_hb_pre_position = None

            # Get rigid part of the receptor
            rec_rigid_atoms = receptor.closest_atoms_from_positions(lig_atom['xyz'], self._distance, rec_atom_property)
            rec_rigid_flex = [receptor]
            rec_rigid_flex_atoms = [rec_rigid_atoms]
            rec_rigid_flex_types = ['rigid']

            # Get the flexible part of the receptor (if present)
            if has_flexible_residues:
                rec_flex_atoms = ligand.closest_atoms_from_positions(lig_atom['xyz'], self._distance, ['flexible_residue', rec_atom_property])
                rec_rigid_flex.append(ligand)
                rec_rigid_flex_atoms.append(rec_flex_atoms)
                rec_rigid_flex_types.append('flex')

            # Add interactions
            for rec, rec_atoms, rec_type in zip(rec_rigid_flex, rec_rigid_flex_atoms, rec_rigid_flex_types):
                for rec_atom in rec_atoms:
                    distance = np.linalg.norm(lig_atom['xyz'] - rec_atom['xyz'])

                    if distance <= self._distance:
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
                            if lig_atom_property == 'hb_don':
                                tmp += [np.degrees(angle_1), np.degrees(angle_2)]
                            else:
                                tmp += [np.degrees(angle_2), np.degrees(angle_1)]

                            if rec_type == 'rigid':
                                rigid_interactions.append(tuple(tmp))
                            else:
                                flex_interactions.append(tuple(tmp))

        return np.array(rigid_interactions, dtype=dtype), np.array(flex_interactions, dtype=dtype)


class HBDonor(_HBBased):
    def __init__(self, distance=3.5, angles=(120, 90)):
        """Hydrogen bond interaction between the ligand (donor) and the receptor (acceptor)
        
        Args:
            distance (float): distance cutoff between donor and acceptor atoms (default: 3.5)
            angles (array-like): angles between donor and acceptor atoms (default: (120, 90))
                Donor-H -- acceptor angle              (angle_1)
                Pre_acceptor-acceptor -- H angle       (angle_2)

        """
        # Because the distance will be between the acceptor and the hydrogen atoms
        super().__init__(distance - 1., angles, ['hb_don', 'hb_acc'])


class HBAcceptor(_HBBased):
    def __init__(self, distance=3.5, angles=(120, 90)):
        """Hydrogen bond interaction between the ligand (acceptor) and the receptor (donor)
        
        Args:
            distance (float): distance cutoff between donor and acceptor atoms (default: 3.5)
            angles (array-like): angles between donor and acceptor atoms (default: (120, 90))
                Donor-H -- acceptor angle              (angle_1)
                Pre_acceptor-acceptor -- H angle       (angle_2)

        """
        # Because the distance will be between the acceptor and the hydrogen atoms
        super().__init__(distance - 1., angles[::-1], ['hb_acc', 'hb_don'])


class WaterDonor(_HBBased):
    def __init__(self, distance=3.2, angle=90):
        """Interaction between donor water molecules (attached to the ligand) and receptor (acceptor)
        
        Args:
            distance (float): distance cutoff between donor water molecule and acceptor atoms (default: 3.2)
            angle (float): angle between donor water molecule and acceptor atoms (default: 90)
                W_donor -- Pre_acceptor-acceptor angle

        """
        super().__init__(distance, (0, angle), ['water', 'hb_acc'])


class WaterAcceptor(_HBBased):
    def __init__(self, distance=3.2, angle=120):
        """Interaction between acceptor water molecules (attached to the ligand) and receptor (donor)
        
        Args:
            distance (float): distance cutoff between donor water molecule and acceptor atoms (default: 3.2)
            angle (float): angle between donor water molecule and acceptor atoms (default: 120)
                W_acceptor -- Donor-H angle

        """
        # Because the distance will be between the acceptor and the hydrogen atoms
        super().__init__(distance - 1., (0, angle), ['water', 'hb_don'])
