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


def _grid(x):
    """Access the underlying ndarray of a Grid object or return the object itself"""
    try:
        return x.grid
    except AttributeError:
        return x

def _guess_format(filename, file_format=None):
    if file_format is None:
        splitted = os.path.splitext(filename)
        file_format = splitted[1][1:]

    file_format = file_format.lower()

    return file_format


def _generate_grid_interpn(points, values):
    """
    Return a interpolate function from the grid and the affinity map.
    This helps to interpolate the energy of coordinates off the grid.
    """
    return RegularGridInterpolator(points, values, bounds_error=False, fill_value=np.inf)


class Map():
    def __init__(self, grid=None, points=None, center=None, delta=None):
        """Create an AutoDock map object"""

        self.grid = None
        self._grid_interpn = None
        self.points = None
        self.center = None
        self.delta = None
        self.origin = None
        self.edges = None

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
            else:
                self._load(grid, points, center=center, delta=delta)

    def _check_compatible(self, other):
        """Check if *other* can be used in an arithmetic operation.
        1) *other* is a scalar
        2) *other* is a grid defined on the same edges
        :Raises: :exc:`TypeError` if not compatible.
        """
        if not (np.isreal(other) or self == other):
            raise TypeError(
                "The argument can not be arithmetically combined with the grid. "
                "It must be a scalar or a grid with identical edges. "
                "Use Grid.resample(other.edges) to make a new grid that is "
                "compatible with other.")
        return True

    def __repr__(self):
        """Print basic information about the maps"""
        info = '--------- Grid information ---------\n'
        info += 'Origin  : %s\n' % ' '.join(['%.3f' % c for c in self.origin])
        info += 'Center  : %s\n' % ' '.join(['%.3f' % c for c in self.center])
        info += 'Points  : %s\n' % ' '.join([str(d) for d in self.grid.shape])
        info += 'Spacing : %s\n' % self.delta
        info += '------------------------------------'

        return info

    def __eq__(self, other):
        if not isinstance(other, Map):
            return False

        return np.all(other.grid == self.grid) \
               and np.all(other.origin == self.origin) \
               and np.all([np.all(other_points == self_points) for other_points, \
                                 self_points in zip(other.points, self.points)])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        self._check_compatible(other)
        return self.grid <= _grid(other)

    def __lt__(self, other):
        self._check_compatible(other)
        return self.grid < _grid(other)

    def __ge__(self, other):
        self._check_compatible(other)
        return self.grid >= _grid(other)

    def __gt__(self, other):
        self._check_compatible(other)
        return self.grid > _grid(other)

    def __add__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid + _grid(other), points=self.points)

    def __sub__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid - _grid(other), points=self.points)

    def __mul__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid * _grid(other), points=self.points)

    def __truediv__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid / _grid(other), points=self.points)

    def __floordiv__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid // _grid(other), points=self.points)

    def __pow__(self, other):
        self._check_compatible(other)
        return self.__class__(np.power(self.grid, _grid(other)), points=self.points)

    def __radd__(self, other):
        self._check_compatible(other)
        return self.__class__(_grid(other) + self.grid, points=self.points)

    def __rsub__(self, other):
        self._check_compatible(other)
        return self.__class__(_grid(other) - self.grid, points=self.points)

    def __rmul__(self, other):
        self._check_compatible(other)
        return self.__class__(_grid(other) * self.grid, points=self.points)

    def __rtruediv__(self, other):
        self._check_compatible(other)
        return self.__class__(_grid(other) / self.grid, points=self.points)

    def __rfloordiv__(self, other):
        self._check_compatible(other)
        return self.__class__(_grid(other) // self.grid, points=self.points)

    def __rpow__(self, other):
        self._check_compatible(other)
        return self.__class__(np.power(_grid(other), self.grid), points=self.points)

    def _load(self, grid, points=None, center=None, origin=None, delta=None):
        """Load and check AutoDock Map"""
        if center is not None and origin is not None:
            error_msg = 'Cannot define both origin and center at the same time.'
            raise ValueError(error_msg)

        self.grid = np.asanyarray(grid)
        shape = np.array(self.grid.shape)

        if points is not None:
            # Check the dimension of the points with the grid
            if len(points) != grid.ndim:
                error_msg = 'Dimension of points is not the same as grid dimension.'
                raise TypeError(error_msg)
            elif not all([len(points[0]) == s for dim, s in enumerate(shape)]):
                error_msg = 'Number of points is not the same as the grid in (at least) one dimension.'
                raise TypeError(error_msg)

            # Check that the spacing is constant
            delta = np.unique(np.diff(points))
            if delta.size > 1:
                raise TypeError('The grid spacing (delta) must be the same in all dimension.')

            self.points = np.asanyarray(points, dtype=object)
            self.center = np.asanyarray([np.mean(point) for point in self.points])
            self.origin = np.asanyarray([point[0] for point in self.points])
            self.delta = float(delta[0])
        elif delta is not None and (origin is not None or center is not None):
            self.delta = float(delta)

            if origin is not None:
                # Check the dimension of the origin and the grid
                if len(origin) != grid.ndim:
                    error_msg = 'Dimension of origin is not the same as grid dimension.'
                    raise TypeError(error_msg)

                # Get grid points using the origin information
                points = [origin[dim] + (np.arange(s)) * self.delta for dim, s in enumerate(shape)]

                self.points = np.asanyarray(points, dtype=object)
                self.origin = np.asanyarray(origin)
                self.center = np.asanyarray([np.mean(point) for point in points])
            else:
                # Check the dimension of the center and the grid
                if len(center) != grid.ndim:
                    error_msg = 'Dimension of center is not the same as grid dimension.'
                    raise TypeError(error_msg)

                # Get grid points using the center information
                half_length = (self.delta * (shape - 1)) / 2.
                xmin, ymin, zmin = center - half_length
                xmax, ymax, zmax = center + half_length
                points = [np.linspace(xmin, xmax, shape[0]),
                          np.linspace(ymin, ymax, shape[1]),
                          np.linspace(zmin, zmax, shape[2])]

                self.points = np.asanyarray(points, dtype=object)
                self.origin = np.asanyarray([xmin, ymin, zmin])
                self.center = np.asanyarray(center)
        else:
            error_msg = 'Wrong/missing data to set up the AutoDock Map. Use'
            error_msg += 'Map(grid=<array>, edges=<list>) or '
            error_msg += 'Map(grid=<array>, origin=(x0, y0, z0), delta=d):\n'
            error_msg += 'Map(grid=<array>, center=(x0, y0, z0), delta=d):\n'
            error_msg += 'grid=%s edges=%s center=%s origin=%s delta=%s'
            error_msg = error_msg % (grid, edges, center, origin, delta)
            raise ValueError(error_msg)

        # Get grid edges
        edges = [self.points[dim][0] + (np.arange(s + 1) - 0.5) * self.delta for dim, s in enumerate(shape)]
        self.edges = np.asanyarray(edges, dtype=object)

        # Generate interprelator
        self._grid_interpn = _generate_grid_interpn(self.points, self.grid)

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
                    nelements = np.array(line.split(' ')[1:4], dtype=np.int)
                    # Transform even numbers to the nearest odd integer
                    npoints = nelements // 2 * 2 + 1
                elif re.search('CENTER', line):
                    center = np.array(line.split(' ')[1:4], dtype=np.float)
                elif re.search('^[0-9]', line):
                    # If the line starts with a number, we stop
                    break

            # Check that the number of grid points was defined
            assert npoints is not None, 'NELEMENTS of the grid is not defined.'

            # Get the energy for each grid element
            grid = [np.float(line) for line in lines[6:]]
            # Some sorceries happen here --> swap x and z axes
            grid = np.swapaxes(np.reshape(grid, npoints[::-1]), 0, 2)

        # Check that the center and spacing points were defined
        assert center is not None, 'CENTER of the grid is not defined.'
        assert delta is not None, 'SPACING of the grid is not defined.'

        self._load(grid, center=center, delta=delta)

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
        shape = np.array(self.grid.shape)

         # Check that the number of grid points in all dimension is odd
        if not all(shape % 2):
            error_msg = 'Cannot write grid in AutoDock map format.'
            error_msg += ' The number of points must be odd.\n'
            error_msg += 'Grid points : %s' % ' '.join([str(d) for d in self.grid.shape])
            raise ValueError(error_msg)

        nelements = shape - 1

        grid_parameter_file = kwargs.get('grid_parameter_file', 'NULL')
        grid_data_file = kwargs.get('grid_data_file', 'NULL')
        macromolecule = kwargs.get('macromolecule', 'NULL')

        with open(filename, 'w') as w:
            # Write header
            w.write('GRID_PARAMETER_FILE %s\n' % grid_parameter_file)
            w.write('GRID_DATA_FILE %s\n' % grid_data_file)
            w.write('MACROMOLECULE %s\n' % macromolecule)
            w.write('SPACING %s\n' % self.delta)
            w.write('NELEMENTS %s\n' % ' '.join(nelements.astype(str)))
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

