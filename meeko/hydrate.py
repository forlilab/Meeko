#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko hydrate molecule
#

import numpy as np

from .utils import geomutils
from .utils import pdbutils


class HydrateMoleculeLegacy:
    def __init__(self, distance=3.0, charge=0, atom_type="W"):
        """Initialize the legacy hydrate typer for AutoDock 4.2.x

        Args:
            distance (float): distance between water molecules and ligand heavy atoms. (default: 3.0)
            charge (float): partial charge of the water molecule. Not use for the hydrated docking. (default: 0)
            atom_type (str): atom type of the water molecule. (default: W)

        """
        self._distance = distance
        self._charge = charge
        self._atom_type = atom_type
        self._bond_type = 1
        self._rotatable = False
        self._hb_config = {'HD': {1: (1, 1)},    # neigh: 1, wat: 1, sp1
                           'OA': {
                                      1: (2, 2), # neigh: 1, wat: 2, sp2
                                      2: (2, 3)  # neigh: 2, wat: 2, sp3
                                  },
                           'SA': {
                                      1: (2, 2), # neigh: 1, wat: 2, sp2
                                      2: (2, 3)  # neigh: 2, wat: 2, sp3
                                  },
                           'NA': {
                                      1: (1, 1), # neigh: 1, wat: 3, sp1
                                      2: (1, 2), # neigh: 2, wat: 1, sp2
                                      3: (1, 3)  # neigh: 3, wat: 1, sp3
                                  }
                           }

    def _place_sp1_one_water(self, anchor_xyz, neighbor_xyz, hb_length=3.0):
        position = anchor_xyz + geomutils.vector(neighbor_xyz, anchor_xyz)
        position = geomutils.resize_vector(position, hb_length, anchor_xyz)
        positions = np.array([position])

        return positions

    def _place_sp2_one_water(self, anchor_xyz, neighbor1_xyz, neighbor2_xyz, hb_length=3.0):
        position = geomutils.atom_to_move(anchor_xyz, [neighbor1_xyz, neighbor2_xyz])
        position = geomutils.resize_vector(position, hb_length, anchor_xyz)
        positions = np.array([position])

        return positions

    def _place_sp2_two_waters(self, anchor_xyz, neighbor1_xyz, neighbor2_xyz, hb_lengths, angles):
        if len(hb_lengths) != 2:
            raise ValueError()
        if len(angles) != 2:
            raise ValueError()

        positions = []

        r = geomutils.rotation_axis(neighbor1_xyz, anchor_xyz, neighbor2_xyz, origin=anchor_xyz)
        p = neighbor1_xyz

        # We rotate p to get each vectors if necessary
        for hb_length, angle in zip(hb_lengths, angles):
            vector = p
            if angle != 0.:
                position = geomutils.rotate_point(vector, anchor_xyz, r, angle)
            position = geomutils.resize_vector(position, hb_length, anchor_xyz)
            positions.append(position)

        positions = np.array(positions)

        return positions

    def _place_sp3_one_water(self, anchor_xyz, neighbor1_xyz, neighbor2_xyz, neighbor3_xyz, hb_length):
        # We have to normalize bonds, otherwise the water molecule is not well placed
        v1 = anchor_xyz + geomutils.normalize(geomutils.vector(anchor_xyz, neighbor1_xyz))
        v2 = anchor_xyz + geomutils.normalize(geomutils.vector(anchor_xyz, neighbor2_xyz))
        v3 = anchor_xyz + geomutils.normalize(geomutils.vector(anchor_xyz, neighbor3_xyz))

        position = geomutils.atom_to_move(anchor_xyz, [v1, v2, v3])
        position = geomutils.resize_vector(position, hb_length, anchor_xyz)
        positions = np.array([position])

        return positions

    def _place_sp3_two_waters(self, anchor_xyz, neighbor1_xyz, neighbor2_xyz, hb_lengths, angles):
        if len(hb_lengths) != 2:
            raise ValueError()
        if len(angles) != 2:
            raise ValueError()

        positions = []

        v1 = anchor_xyz + geomutils.normalize(geomutils.vector(anchor_xyz, neighbor1_xyz))
        v2 = anchor_xyz + geomutils.normalize(geomutils.vector(anchor_xyz, neighbor2_xyz))

        r = anchor_xyz + geomutils.normalize(geomutils.vector(v1, v2))
        p = geomutils.atom_to_move(anchor_xyz, [v1, v2])

        # We rotate p to get each vectors if necessary
        for hb_length, angle in zip(hb_lengths, angles):
            vector = p
            if angle != 0.:
                position = geomutils.rotate_point(vector, anchor_xyz, r, angle)
            position = geomutils.resize_vector(position, hb_length, anchor_xyz)
            positions.append(position)

        positions = np.array(positions)

        return positions

    def hydrate(self, setup):
        """Add water molecules to the ligand

        Args:
            setup: MoleculeSetup object

        """
        water_anchors = []
        water_positions = []
        # It will be the same distance for all of the water molecules
        hb_length = self._distance

        for a, neighbors in setup.graph.items():
            atom_type = setup.get_atom_type(a)
            anchor_xyz = setup.get_coord(a)
            neighbor1_xyz = setup.get_coord(neighbors[0])
            positions = np.array([])
            n_wat = None
            hyb = None

            if atom_type in self._hb_config:
                try:
                    n_wat, hyb = self._hb_config[atom_type][len(neighbors)]
                except KeyError:
                    raise RuntimeError('Cannot place water molecules on atom %d of type %s with %d neighbors.' % (a, atom_type, len(neighbors)))

                water_anchors.append(a)

            if hyb == 1:
                if n_wat == 1:
                    # Example: X-HD
                    positions = self._place_sp1_one_water(anchor_xyz,
                                                          neighbor1_xyz,
                                                          hb_length - 1.0)
            elif hyb == 2:
                if n_wat == 1:
                    # Example: X-Nitrogen-X
                    neighbor2_xyz = setup.get_coord(neighbors[1])
                    positions = self._place_sp2_one_water(anchor_xyz,
                                                          neighbor1_xyz, neighbor2_xyz,
                                                          hb_length)
                elif n_wat == 2:
                    # Example: C=0 (backbone oxygen)
                    tmp_neighbors = [x for x in setup.get_neigh(neighbors[0]) if not x == a]
                    neighbor2_xyz =  setup.get_coord(tmp_neighbors[0])
                    positions = self._place_sp2_two_waters(anchor_xyz,
                                                           neighbor1_xyz, neighbor2_xyz,
                                                           [hb_length, hb_length],
                                                           [-np.radians(120), np.radians(120)])
                elif n_wat == 3:
                    hyb = 3
            elif hyb == 3:
                if n_wat == 1:
                    # Example: Ammonia
                    neighbor2_xyz = setup.get_coord(neighbors[1])
                    neighbor3_xyz = setup.get_coord(neighbors[2])
                    positions = self._place_sp3_one_water(anchor_xyz,
                                                          neighbor1_xyz, neighbor2_xyz, neighbor3_xyz,
                                                          hb_length)
                elif n_wat == 2:
                    # Example: O-HD (Oxygen in hydroxyl group)
                    neighbor2_xyz = setup.get_coord(neighbors[1])
                    positions = self._place_sp3_two_waters(anchor_xyz,
                                                           neighbor1_xyz, neighbor2_xyz,
                                                           [hb_length, hb_length],
                                                           [-np.radians(60), np.radians(60)])
                elif n_wat == 3:
                    positions = np.array([])

            if positions.size:
                water_positions.append(positions)

        for water_anchor, waters_on_anchor in zip(water_anchors, water_positions):
            for water_on_anchor in waters_on_anchor:
                tmp = setup.pdbinfo[water_anchor]
                pdbinfo = pdbutils.PDBAtomInfo('WAT', tmp.resName, tmp.resNum, tmp.chain)
                setup.add_pseudo(water_on_anchor, self._charge, [water_anchor], self._atom_type,
                                     self._bond_type, self._rotatable, pdbinfo)
