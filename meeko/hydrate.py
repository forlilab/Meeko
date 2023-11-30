#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko hydrate molecule
#

import json
import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
import warnings

from .atomtyper import AtomicGeometry
from .molsetup import RDKitMoleculeSetup
from .utils import geomutils
from .utils import pdbutils

class Waters:

    @staticmethod
    def make_molsetup_OPC():
        """build a molsetup for an OPC water"""
    
        raise NotImplementedError
        rdmol = Chem.MolFromSmiles("O")
        conformer = Chem.Conformer(rdmol.GetNumAtoms())
        conformer.SetAtomPosition(0, Point3D(0, 0, 0))
        rdmol.AddConformer(conformer)
        molsetup = RDKitMoleculeSetup(rdmol)
        return molsetup
    
    @staticmethod
    def make_molsetup_TIP3P_AA():
        """build a molsetup for an all atom (AA) TIP3P water, i.e. explicit Hs"""
    
        rdmol = Chem.MolFromSmiles("O")
        rdmol = Chem.AddHs(rdmol)
        conformer = Chem.Conformer(rdmol.GetNumAtoms())
        dist_oh = 0.9572
        ang_hoh = np.radians(104.52)
        conformer.SetAtomPosition(0, Point3D(0, 0, 0))
        conformer.SetAtomPosition(1, Point3D(dist_oh, 0, 0))
        conformer.SetAtomPosition(2, Point3D(np.cos(ang_hoh)*dist_oh, np.sin(ang_hoh)*dist_oh, 0))
        rdmol.AddConformer(conformer)
        molsetup = RDKitMoleculeSetup(rdmol)
        molsetup.bond[(0, 1)]["rotatable"] = False
        molsetup.bond[(0, 2)]["rotatable"] = False
        molsetup.atom_type[0] = "n-tip3p-O"
        molsetup.atom_type[1] = "n-tip3p-H"
        molsetup.atom_type[2] = "n-tip3p-H"
        molsetup.charge[0] = -0.834
        molsetup.charge[1] =  0.417
        molsetup.charge[2] =  0.417
        return molsetup


class Hydrate:
    defaults = [
        {"smarts": "[#7X2;v3;!+](=,:[*])[*]", "IDX": 1, "z": [2, 3], "is_donor": False, "geometries": [
                {"phi": 0.0, "distance": 3.0},
            ]},
        {"smarts": "[#1][#7,#8,#9]", "IDX": 1, "z": [2], "is_donor": True, "geometries": [
                {"phi": 0.0, "distance": 2.0},
            ]},
        {"smarts": "[#8X1]=[X3][*]", "IDX": 1, "z": [2], "x": [3], "is_donor": False, "geometries": [
                {"phi": 60.0, "theta":   0, "distance": 3.0},
                {"phi": 60.0, "theta": 180, "distance": 3.0},
            ]},
    ]

    def __init__(self, water_model="tip3p", planar_tol=0.05):
        self.rule_list = json.loads(json.dumps(self.defaults))
        self.planar_tol = planar_tol
        if water_model == "tip3p":
            self.make_water = Waters.make_molsetup_TIP3P_AA
        else:
            raise RuntimeError("unknown water model: %s" % water_model)

    def __call__(self, molsetup):
        coordinates = [molsetup.coord[i] for i in range(molsetup.atom_true_count)]
        water_molsetup_list = []
        for rule in self.rule_list:
            hits = molsetup.find_pattern(rule["smarts"])
            if len(hits) == 0:
                continue
            matched_idxs = set()
            smarts_idx = rule.get("IDX", 1) - 1 # default to first atom in SMARTS
            for hit in hits:
                parent_index = hit[smarts_idx]
                parent_center = coordinates[parent_index]
                if parent_index in matched_idxs: 
                    warnings.warn("SMARTS <%s> matches same target atom more than once, ignoring" % rule["smarts"])
                    continue
                matched_idxs.add(parent_index)
                # required settings
                z = [hit[i-1] for i in rule["z"]]
                is_donor = rule["is_donor"]
                # optional settings
                x = rule.get("x", [])
                x = [hit[i-1] for i in x]
                x90 = rule.get("x90", False)
                atomgeom = AtomicGeometry(parent_index, z, x, x90, self.planar_tol)
                for geometry in rule["geometries"]:
                    # required values
                    distance = geometry["distance"]
                    phi = np.radians(geometry["phi"])
                    # optional values
                    theta = geometry.get("theta", 0)
                    theta = np.radians(theta)
                    # place water
                    water_center = atomgeom.calc_point(distance, theta, phi, coordinates)
                    watersetup = self.make_water()
                    watercoords = [watersetup.coord[i] for i in watersetup.coord]
                    self.orient_water(watercoords, water_center, parent_center, is_donor)
                    for i in range(len(watercoords)):
                        watersetup.coord[i] = watercoords[i]
                    water_molsetup_list.append(watersetup)
        return water_molsetup_list

    
    @staticmethod
    def orient_water(coords, target_xyz, anchor_xyz, anchor_is_donor):
        """ coords will be changed in place
            target_xyz is where the water oxygen will be
            anchor_xyz is the atom to which this molecule belongs
            anchor_is_donor is True for H, and False for O, N, S
    
            expects starting O to be at (0, 0, 0) and an H along x-axis
        """
    
        if anchor_is_donor:
            ang_hoh = np.radians(104.52)
            ang_lp = np.radians(109.5) # probably good enough for lone pairs
            x =  np.cos(np.pi-ang_hoh/2) * np.cos(ang_lp/2)
            y = -np.sin(np.pi-ang_hoh/2) * np.cos(ang_lp/2)
            z = np.sin(ang_lp/2)
            axis = (x, y, z)
        else:
            axis = (1, 0, 0)
        # rotate
        v = np.array(anchor_xyz - target_xyz)
        v /= np.sqrt(np.dot(v, v))
        rotaxis = np.cross(axis, v)
        magnitude = np.sqrt(np.dot(rotaxis, rotaxis))
        if magnitude > 1e-6:
            #rotangle = np.arccos(np.dot(axis, v))
            rotangle = np.arcsin(magnitude)
            for i in range(len(coords)):
                coords_tuple = AtomicGeometry.rot3D(coords[i], rotaxis, rotangle) # normalizes rotaxis
                coords[i] = list(coords_tuple)
                
        # translate
        for i in range(len(coords)):
            for j in range(3):
                coords[i][j] = coords[i][j] + target_xyz[j]


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
                setup.add_pseudo(
                        coord=water_on_anchor,
                        charge=self._charge,
                        anchor_list=[water_anchor],
                        atom_type=self._atom_type,
                        rotatable=self._rotatable,
                        pdbinfo=pdbinfo)
