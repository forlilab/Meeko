#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#
# Class to manage autodock maps
#

import collections
import os
import re
import copy
import warnings

import numpy as np
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator


def _guess_format(filename, file_format=None):
    if file_format is None:
        splitted = os.path.splitext(filename)
        file_format = splitted[1][1:]

    file_format = file_format.upper()

    return file_format


def _loader(grid, edges=None, origin=None, delta=None):

    return grid, edges, origin, delta


def _load_autdock_map(filename):
    center = None
    grid = None
    shape = None
    delta = None

    with open(map_file) as f:
        lines = f.readlines()

        for line in lines:
            if re.search("^SPACING", line):
                delta = np.float(line.split(" ")[1])
            elif re.search("^NELEMENTS", line):
                shape = np.array(line.split(" ")[1:4], dtype=np.int)
                # Transform even numbers to the nearest odd integer
                shape = shape // 2 * 2 + 1
            elif re.search("CENTER", line):
                center = np.array(line.split(" ")[1:4], dtype=np.float)
            elif re.search("^[0-9]", line):
                # If the line starts with a number, we stop
                break

        # Get the energy for each grid element
        grid = [np.float(line) for line in lines[6:]]
        # Some sorceries happen here --> swap x and z axes
        grid = np.swapaxes(np.reshape(grid, shape[::-1]), 0, 2)

    return grid, center, delta


class Map():
    def __init__(self, name=None, grid=None, edges=None, center=None, origin=None, delta=None):
        """Create an AutoDock map object"""

        self._center = None
        self._delta = None
        self._filename = None
        self._name = None
        self._npts = None
        self._grid = None
        self._grid_interpn = None
        self._origin = None

        self._exporters = {'MAP': _export_autodock_map}
        self._loaders = {'MAP': _load_autdock_map}

        if grid is not None:
            if isinstance(grid, str):
                filename = grid

                try:
                    with open(filename):
                        pass
                except (OSError, IOError):
                    filename = None
            else:


        if filename is not None:
            grid, shape, center, delta = self._read_autdock_map(filename)
        
        self._load_grid(grid, edges, center, origin, delta)

    def __repr__(self):
        """Print basic information about the maps"""
        info = "Box center   : %s\n" % " ".join(self._center.astype(str))
        info += "Box size     : %s\n" % " ".join(self._npts.astype(str))
        info += "Box spacing  : %s\n" % self._spacing
        info += "Affinity maps: %s\n" % " ".join(self._maps.keys())

        return info

    def __eq__(self, other):

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):

    def __sub__(self, other):

    def __mul__(self, other):

    def __truediv__(self, other):

    def __floordiv__(self, other):

    def __pow__(self, other):

    def __radd__(self, other):

    def __rsub__(self, other):

    def __rmul__(self, other):

    def __rtruediv__(self, other):

    def __rfloordiv__(self, other):

    def __rpow__(self, other):

    def load(self, grid_filename):
        """Load AutoDock Map"""
        file_format = _guess_format(grid_filename)

        try:
            loader = self._loaders[file_format]
        except:
            error_msg = "Cannot read %s file format. Available format: %s" % (file_format.lower(), self._loaders.keys())
            raise ValueError(error_msg)

        self.grid, self.edges, self.origin, self.delta = loader(grid_filename)

    def is_in_map(self, xyz):
        """Check if coordinates are in the map.

        Args:
            xyz (array_like): Array of 3d coordinates

        Returns:
            ndarray: 1d Numpy array of boolean

        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_in = np.logical_and(self._xmin <= x, x <= self._xmax)
        y_in = np.logical_and(self._ymin <= y, y <= self._ymax)
        z_in = np.logical_and(self._zmin <= z, z <= self._zmax)
        all_in = np.all((x_in, y_in, z_in), axis=0)
        # all_in = np.logical_and(np.logical_and(x_in, y_in), z_in)

        return all_in

    def energy(self, xyz, method='linear'):
        """Grid energy of each coordinates xyz.

        Args:
            xyz (array_like): Array of 3d coordinates
            method (str): Interpolate method (default: linear)

        Returns:
            ndarray: 1d Numpy array of the energy values

        """
        return self._grid_interpn(xyz, method=method)

    def add_bias(self, name, coordinates, bias_value, radius):
        """Add energy bias to map using Juan's method.

        Args:
            name (str): name of the new or existing map
            coordinates (array_like): 2d array of 3d coordinates
            bias_value (float): energy bias value to add (in kcal/mol)
            radius (float): radius of the bias (in Angtrom)

        """
        coordinates = np.atleast_2d(coordinates)

        if name in self._maps:
            new_map = self._maps[name]
        else:
            new_map = np.zeros(self._npts)

        # We add all the bias one by one in the new map
        for coordinate in coordinates:
            sphere_xyz = self.neighbor_points(coordinate, radius)
            indexes = self._cartesian_to_index(sphere_xyz)

            distances = spatial.distance.cdist([coordinate], sphere_xyz, "euclidean")[0]
            bias_energy = bias_value * np.exp(-1. * (distances ** 2) / (radius ** 2))

            new_map[indexes[:,0], indexes[:,1], indexes[:,2]] += bias_energy

        # And we replace the original one only at the end, it is faster
        self._maps[name] = new_map
        self._maps_interpn[name] = self._generate_affinity_map_interpn(new_map)
    
    def export(self, filename, prefix=None, grid_parameter_file="grid.gpf",
               grid_data_file="maps.fld", macromolecule="molecule.pdbqt")
        """Export AutoDock maps.

        Args:
            map_types (list): list of atom types to export
            prefix (str): prefix name file (default: None)
            grid_parameter_file (str): name of the gpf file (default: grid.gpf)
            grid_data_file (str): name of the fld file (default: maps.fld)
            macromolecule (str): name of the receptor (default: molecule.pdbqt)

        Returns:
            None

        """
