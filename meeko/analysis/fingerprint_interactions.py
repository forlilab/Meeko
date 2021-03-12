#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os

import numpy as np
import pandas as pd


def _compute_angle(v1, v2):
    unit_vector_1 = v1 / np.linalg.norm(v1)
    unit_vector_2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle


def _is_valid_hydrogen_bond(atom_a, atom_b, atom_c, atom_d, criteria):
    """Check if the hydrogen bond is valid based on the angles

    Donor-H -- acceptor angle        : atom_c-atom_b -- atom_a
    Pre_acceptor-acceptor -- H angle : atom_d-atom_a -- atom_b

    Source: https://psa-lab.github.io/Hbind/user_guide/

    """
    angle_1 = np.degrees(_compute_angle(atom_b - atom_c, atom_b - atom_a))
    angle_2 = np.degrees(_compute_angle(atom_a - atom_d, atom_a - atom_b))
    return (angle_1 >= criteria[0]) & (angle_2 >= criteria[1])


class FingerprintInteractions:

    def __init__(self):
        self._data = None
        self._unique_interactions = None
        self._criteria = {'hb_acc': [3.2, 120, 90], 'hb_don': [3.2, 120, 90],
                          'all': [4.2], 'vdw': [4.2],
                          'reactive': [2.0]}
        self._valid_interactions = {'hb_acc': 'hb_don', 'hb_don': 'hb_acc',
                                    'all': 'all', 'vdw': 'vdw',
                                    'reactive': 'reactive'}

    def run(self, receptor, molecules):
        data = []
        self._unique_interactions = {'hb': {*()}, 'vdw': {*()}, 'reactive': {*()}}

        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        for molecule in molecules:
            for pose in molecule:
                tmp_hb = []
                tmp_vdw = []

                for interaction_type in ['hb_acc', 'hb_don', 'all']:
                    max_distance = self._criteria[interaction_type][0]

                    lig_atoms = pose.atoms_by_properties(['ligand', interaction_type])

                    for lig_atom in lig_atoms:
                        rec_atoms = receptor.closest_atoms(lig_atom['xyz'], max_distance, self._valid_interactions[interaction_type])

                        if rec_atoms.size > 0:
                            if interaction_type in ['all']:
                                interactions = ['%s:%d' % (x[0], x[1]) for x in np.unique(rec_atoms[['chain', 'resid']], axis=0)]
                                # We trick 'all' interaction type as 'vdw' at the end
                                tmp_vdw.extend(interactions)
                            elif interaction_type in ['hb_acc', 'hb_don']:
                                interactions = []

                                # Get the atoms attached
                                # We will use their positions to calculate the HB vector
                                lig_bound_atoms_index = pose.neighbor_atoms(lig_atom['id'] - 1)
                                lig_bound_atoms = pose.atoms(lig_bound_atoms_index[0])
                                lig_hb_vector = np.mean(lig_bound_atoms['xyz'], axis=0)

                                # Get unique receptor residues around the lig_atom
                                resids, index = np.unique(rec_atoms[['chain', 'resid', 'name']], axis=0, return_index=True)

                                for i in index:
                                    # And we do the same thing with the receptor
                                    rec_bound_atoms_index = receptor.neighbor_atoms(rec_atoms[i]['id'] - 1)
                                    rec_bound_atoms = receptor.atoms(rec_bound_atoms_index[0])
                                    rec_hb_vector = np.mean(rec_bound_atoms['xyz'], axis=0)

                                    if interaction_type == 'hb_acc':
                                        good_hb = _is_valid_hydrogen_bond(lig_atom['xyz'], rec_atoms[i]['xyz'],
                                                                          rec_hb_vector, lig_hb_vector,
                                                                          self._criteria[interaction_type][1:])
                                    else:
                                        good_hb = _is_valid_hydrogen_bond(rec_atoms[i]['xyz'], lig_atom['xyz'],
                                                                          lig_hb_vector, rec_hb_vector,
                                                                          self._criteria[interaction_type][1:])

                                    if good_hb:
                                        chain, resid, name = rec_atoms[i][['chain', 'resid', 'name']]
                                        interactions.append('%s:%d:%s' % (chain, resid, name))

                                tmp_hb.extend(interactions)

                tmp_hb = set(tmp_hb)
                tmp_vdw = set(tmp_vdw)
                # Store all the unique interactions we seen
                self._unique_interactions['hb'].update(tmp_hb)
                self._unique_interactions['vdw'].update(tmp_vdw)

                data.append((pose.name, pose.pose_id, list(tmp_hb), list(tmp_vdw)))

        self._data = data

    def to_dataframe(self):
        count = 0
        resid_to_idx_encoder = {}
        columns = [[], []]
        names = []
        poses = []

        # Generate one-hot encoder-like (and the columns for the dataframe)
        for inte_type, resids in self._unique_interactions.items():
            columns[0].extend([inte_type] * len(resids))
            columns[1].extend(resids)

            for resid in resids:
                resid_to_idx_encoder[resid] = count
                count += 1

        # Create multicolumns for the dataframe
        c_tuples = list(zip(*columns))
        multi_columns = pd.MultiIndex.from_tuples(c_tuples)

        # Convert resids in one hot fingerprint interactions
        fpi = np.zeros(shape=(len(self._data), count), dtype=int)

        for i, pose_molecule in enumerate(self._data):
            idx = [resid_to_idx_encoder[x] for x in pose_molecule[2] + pose_molecule[3]]
            fpi[i][idx] = 1
            names.append(pose_molecule[0])
            poses.append(pose_molecule[1] + 1)

        # Create final dataframe
        df = pd.DataFrame(fpi, index=np.arange(0, len(self._data)), columns=multi_columns)
        df['name'] = names
        df['pose'] = poses
        df.set_index(['name', 'pose'], inplace=True)

        return df
