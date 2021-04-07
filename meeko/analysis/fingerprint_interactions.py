#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

import os

import numpy as np
import pandas as pd

from .interactions import Hydrophobic, Reactive, Metal
from .interactions import HBDonor, HBAcceptor, WaterDonor, WaterAcceptor


class FingerprintInteractions:

    def __init__(self, receptor):
        """FingerprintInteractions object

        Args:
            receptor (PDBQTReceptor): receptor 

        """
        self._data = []
        self._interactions = [Hydrophobic(),
                              Reactive(),
                              Metal(), 
                              HBDonor(), HBAcceptor(),
                              WaterDonor(), WaterAcceptor()]
        self._unique_interactions = {interaction.name: {*()} for interaction in self._interactions}
        self._receptor = receptor

    def run(self, molecules):
        """Run the fingerprint interactions.
        
        Args:
            molecules (PDBQTMolecule, list of PDBQTMolecule): molecule or list of molecules

        """
        data = []

        if not isinstance(molecules, (list, tuple)):
            molecules = [molecules]

        for molecule in molecules:
            for pose in molecule:
                tmp = {}

                for interaction in self._interactions:
                    resids = ''
                    columns = ['chain', 'resid']
                    if interaction.name in ['HBDonor', 'HBAcceptor', 'WaterDonor', 'WaterAcceptor']:
                        columns += ['name']

                    rigid_interactions, flex_interactions = interaction.find(pose, self._receptor)

                    rec_rigid_atoms = self._receptor.atoms(rigid_interactions['receptor_idx'])
                    rec_flex_atoms = pose.atoms(flex_interactions['receptor_idx'])

                    if rec_rigid_atoms.size > 0:
                        unique_resids = np.unique(rec_rigid_atoms[columns])
                        resids = [':'.join([interaction.name] + [str(v) for v in u]) for u in unique_resids]

                    if rec_flex_atoms.size > 0:
                        unique_resids = np.unique(rec_flex_atoms[columns])
                        resids += [':'.join([interaction.name] + [str(v) for v in u]) for u in unique_resids]

                    tmp[interaction.name] = resids

                tmp_data = [pose.name, pose.pose_id]
                for inte_type, resids in tmp.items():
                    unique_resids = set(resids)
                    self._unique_interactions[inte_type].update(unique_resids)
                    tmp_data.append(list(unique_resids))
                data.append(tmp_data)

        self._data.extend(data)

    def to_dataframe(self, remove_common=False):
        """Generate a panda DataFrame with all the interactions
        
        Args:
            remove_common (bool): remove all the interactions (columns) that
                are common to all the molecules (default: False)

        Returns:
            pd.DataFrame: pandas DataFrame containing all the interactions
                found between the molecules and the receptor

        """
        count = 0
        resid_to_idx_encoder = {}
        columns = [[], []]
        names = []
        poses = []

        # Generate one-hot encoder-like (and the columns for the dataframe)
        for inte_type, resids in self._unique_interactions.items():
            columns[0].extend([inte_type] * len(resids))
            # Remove the v/w/h_ tag for the column names
            columns[1].extend([resid.split(':', 1)[-1] for resid in resids])

            for resid in resids:
                resid_to_idx_encoder[resid] = count
                count += 1

        # Create multicolumns for the dataframe
        c_tuples = list(zip(*columns))
        multi_columns = pd.MultiIndex.from_tuples(c_tuples)

        # Convert resids in one hot fingerprint interactions
        fpi = np.zeros(shape=(len(self._data), count), dtype=int)

        for i, pose_molecule in enumerate(self._data):
            resids = [j for i in pose_molecule[2:] for j in i]
            idx = [resid_to_idx_encoder[resid] for resid in resids]
            fpi[i][idx] = 1
            names.append(pose_molecule[0])
            poses.append(pose_molecule[1] + 1)

        # Create final dataframe
        df = pd.DataFrame(fpi, index=np.arange(0, len(self._data)), columns=multi_columns)
        
        df['name'] = names
        df['pose'] = poses
        df.set_index(['name', 'pose'], inplace=True)

        if remove_common:
            # Remove interactions (columns) that are common to all molecules
            df = df.loc[:, (df.sum(axis=0) != df.shape[0])]

        return df
