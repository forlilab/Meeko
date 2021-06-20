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

    file_format = file_format.lower()

    return file_format


class Map():
    def __init__(self, name, grid=None, edges=None, center=None, origin=None, delta=None):
        """Create an AutoDock map object"""

        self.center = None
        self.delta = None
        self.filename = None
        self.name = name
        self.grid = None
        self._grid_interpn = None
        self.origin = None

        self._exporters = {'map': self._export_autodock_map}
        self._loaders = {'map': self._load_autdock_map}

        if grid is not None:
            if isinstance(grid, str):
                filename = grid

                # Try if we can open the file
                try:
                    with open(filename):
                        pass
                except (OSError, IOError):
                    filename = None

                self.load(filename)
                self._filename = filename
            else:
                self._load(grid, edges, center, origin, delta)

    def __repr__(self):
        """Print basic information about the maps"""
        info = '------ Grid information ------\n'
        info += 'Grid name      : %s\n' % self.name
        info += 'Grid origin    : %s\n' % ' '.join(['%.3f' % c for c in self.origin])
        info += 'Grid center    : %s\n' % ' '.join(['%.3f' % c for c in self.center])
        info += 'Grid size      : %s\n' % ' '.join([str(d) for d in self.grid.shape])
        info += 'Grid spacing   : %s\n' % self.delta
        info += '------------------------------'

        return info

    """
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
    """

    def _load(self, grid, edges=None, center=None, origin=None, delta=None):
        """Load and check AutoDock Map"""
        if center is not None and origin is not None:
            error_msg = 'Cannot define both origin and center at the same time.'
            raise ValueError(error_msg)

        self.grid = np.asanyarray(grid)

        if edges is not None:
            self.edges = np.asanyarray(edges)
        elif delta is not None and (origin is not None or center is not None):
            self.delta = float(delta)

            if origin is not None:
                self.origin = np.asanyarray(origin)

                if len(origin) != grid.ndim:
                    error_msg = 'Dimension of origin is not the same as grid dimension.'
                    raise TypeError(error_msg)

                # Get edges
                edges = [self.origin[dim] + np.arange(m) * self.delta for dim, m in enumerate(self.grid.shape)]

                self.edges = np.asanyarray(edges)
                self.center = np.asanyarray([np.mean(edge) for edge in self.edges])
            else:
                self.center = np.asanyarray(center)

                if len(center) != grid.ndim:
                    error_msg = 'Dimension of center is not the same as grid dimension.'
                    raise TypeError(error_msg)

                # Get Min and Max coordinates
                half_length = (self.delta * self.grid.shape) / 2.
                xmin, ymin, zmin = self.center - half_length
                xmax, ymax, zmax = self.center + half_length
                # Get edges
                edges = [np.linspace(xmin, xmax, self.grid.shape[0]),
                         np.linspace(ymin, ymax, self.grid.shape[1]),
                         np.linspace(zmin, zmax, self.grid.shape[2])]

                self.edges = np.asanyarray([x, y, z])
                self.origin = np.asanyarray([xmin[0], ymin[0], zmax[0]])
        else:
            error_msg = 'Wrong/missing data to set up the AutoDock Map. Use'
            error_msg += 'Map(grid=<array>, edges=<list>) or '
            error_msg += 'Map(grid=<array>, origin=(x0, y0, z0), delta=d):\n'
            error_msg += 'Map(grid=<array>, center=(x0, y0, z0), delta=d):\n'
            error_msg += 'grid=%s edges=%s center=%s origin=%s delta=%s'
            error_msg = error_msg % (grid, edges, center, origin, delta)
            raise ValueError(error_msg)

    def _load_autdock_map(self, filename):
        center = None
        grid = None
        npts = None
        delta = None

        with open(filename) as f:
            lines = f.readlines()

            for line in lines:
                if re.search('^SPACING', line):
                    delta = np.float(line.split(' ')[1])
                elif re.search('^NELEMENTS', line):
                    nvoxels = np.array(line.split(' ')[1:4], dtype=np.int)
                    # Transform even numbers to the nearest odd integer
                    npts = nvoxels // 2 * 2 + 1
                elif re.search('CENTER', line):
                    center = np.array(line.split(' ')[1:4], dtype=np.float)
                elif re.search('^[0-9]', line):
                    # If the line starts with a number, we stop
                    break

            # Check that the number of grid points was defined
            assert npts is not None, 'NELEMENTS of the grid is not defined.'

            # Get the energy for each grid element
            grid = [np.float(line) for line in lines[6:]]
            # Some sorceries happen here --> swap x and z axes
            grid = np.swapaxes(np.reshape(grid, npts[::-1]), 0, 2)

        # Check that the center and spacing points were defined
        assert center is not None, 'CENTER of the grid is not defined.'
        assert delta is not None, 'SPACING of the grid is not defined.'

        # Compute the origin of the grid
        xmin, ymin, zmin = center - ((delta * nvoxels) / 2.)
        origin = np.asanyarray([xmin, ymin, zmin])

        self._load(grid, origin=origin, delta=delta)

    def load(self, grid_filename):
        """Load AutoDock Map"""
        file_format = _guess_format(grid_filename)

        try:
            loader = self._loaders[file_format]
        except:
            error_msg = 'Cannot read %s file format. Available format: %s' % (file_format, self._loaders.keys())
            raise ValueError(error_msg)

        loader(grid_filename)

    def _export_autodock_map(self, filename, **kwargs):
        """
    
        Args:
            map_types (list): list of atom types to export
            grid_parameter_file (str): name of the gpf file (default: NULL)
            grid_data_file (str): name of the fld file (default: NULL)
            macromolecule (str): name of the receptor (default: NULL)
        """
        grid_shape = np.array(self.grid.shape)

         # Check that the number of grid points in all dimension is odd
        if not all(grid_shape % 2):
            error_msg = 'Cannot write AutoDock map in map format.'
            error_msg += ' The number of voxel must be even.\n'
            error_msg += 'Grid size : %s' % ' '.join([str(d) for d in self.grid.shape])
            raise ValueError(error_msg)

        nvoxels = grid_shape - 1

        grid_parameter_file = kwargs.get('grid_parameter_file', 'NULL')
        grid_data_file = kwargs.get('grid_data_file', 'NULL')
        macromolecule = kwargs.get('macromolecule', 'NULL')

        with open(filename, 'w') as w:
            # Write header
            w.write('GRID_PARAMETER_FILE %s\n' % grid_parameter_file)
            w.write('GRID_DATA_FILE %s\n' % grid_data_file)
            w.write('MACROMOLECULE %s\n' % macromolecule)
            w.write('SPACING %s\n' % self.delta)
            w.write('NELEMENTS %s\n' % ' '.join(nvoxels.astype(str)))
            w.write('CENTER %s\n' % ' '.join(['%.3f' % c for c in self.center]))
            # Write grid (swap x and z axis before)
            m = np.swapaxes(self.grid, 0, 2).flatten()
            w.write('\n'.join(m.astype(str)))
            w.write('\n')
    
    def export(self, filename, overwrite=False, **kwargs):
        """Export AutoDock maps."""
        file_format = _guess_format(filename)

        if not overwrite and os.path.exists(filename):
            raise IOError('File %s already exists. Use overwrite=True to overwrite the file.' % filename)

        try:
            exporter = self._exporters[file_format]
        except:
            error_msg = 'Cannot export %s file format.'
            error_msg += ' Available format: %s' 
            error_msg = error_msg % (file_format, self._exporters.keys())
            raise ValueError(error_msg)

        exporter(filename, **kwargs)

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

    def add_bias(self, coordinates, bias_value, radius):
        """Add energy bias to map using Juan's method.

        Args:
            coordinates (array_like): array of 3d coordinates
            bias_value (float): energy bias value to add (in kcal/mol)
            radius (float): radius of the bias (in Angtrom)

        """
        coordinates = np.atleast_2d(coordinates)

        # We add all the bias one by one in the new map
        for coordinate in coordinates:
            sphere_xyz = self.neighbor_points(coordinate, radius)
            indexes = self._cartesian_to_index(sphere_xyz)

            distances = spatial.distance.cdist([coordinate], sphere_xyz, 'euclidean')[0]
            bias_energy = bias_value * np.exp(-1. * (distances ** 2) / (radius ** 2))

            new_map[indexes[:,0], indexes[:,1], indexes[:,2]] += bias_energy

        # And we replace the original one only at the end, it is faster
        self._maps[name] = new_map
        self._maps_interpn[name] = self._generate_affinity_map_interpn(new_map)

    def add_mask(self, coordinates, mask_value, radius):
        """Add energy mask to map using Diogo's method.
        
        Args:
            coordinates (array_like): array of 3d coordinates
            mask_value (float): energy mask value to add (in kcal/mol)
            radius (float): radius of the mask (in Angtrom)

        """
        coordinates = np.atleast_2d(coordinates)

