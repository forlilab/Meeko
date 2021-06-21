#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#
# Class to manage autodock maps
#

import os
import re

import numpy as np
from scipy import spatial
from scipy.interpolate import RegularGridInterpolator


def _map(x):
    """Access the underlying ndarray of a Map object or return the object itself"""
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


def _build_kdtree_from_points(points):
    """Return the kdtree using points (x, y, z)."""
    X, Y, Z = np.meshgrid(points[0], points[1], points[2])
    xyz = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    kdtree = spatial.cKDTree(xyz)

    return kdtree


class Map():
    """Class to manage energy map for AutoDock."""
    def __init__(self, grid=None, points=None, center=None, delta=None):
        """Create an AutoDock map object from data.
        
        The Map can be manipulated as a standard numpy array or by directly
        accessing to the internal grid (Map.grid).

        There are multiple ways to create a Map object.
        
        From a numpy array::

            grid = np.array(shape=(55, 55, 55))
            m = Map(grid, center=(0, 0, 0), delta=0.375)


        From an existing AutoDock map file::

            m = Map(filename)

        or::

            m = Map()
            m.load(filename)


        Args:
            grid (ndarray or str): 3D numpy array or filename
            points (list of arrays): The points along each X Y Z dimensions
            center (array-like): Center of the grid define by X Y Z coordinates
            delta (float): Grid spacing in each dimension

        """

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
                except:
                    raise

                self.load(filename)
            else:
                self._load(grid, points, center=center, delta=delta)

    def _check_compatible(self, other):
        if not (np.all(np.isreal(other)) or self == other):
            raise TypeError(
                "The argument can not be arithmetically combined with the grid. "
                "It must be a scalar or a grid with identical edges. "
                "Use Grid.resample(other.edges) to make a new grid that is "
                "compatible with other.")

        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(origin={self.origin}, center={self.center}, shape={self.grid.shape}, spacing={self.delta})"
    
    def __array__(self):
        return self.grid
    
    def __iter__(self):
        for elem in self.grid:
            yield elem
    
    def __getitem__(self, key):
        return self.grid[key]
    
    @property
    def size(self):
        return np.prod(self.grid.shape)
    
    @property
    def shape(self):
        return self.grid.shape

    def __eq__(self, other):
        if not isinstance(other, Map):
            return False

        return np.all(other.grid == self.grid) \
               and np.all([np.all(other_points == self_points) for other_points, \
                                  self_points in zip(other.points, self.points)])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        self._check_compatible(other)
        return self.grid <= _map(other)

    def __lt__(self, other):
        self._check_compatible(other)
        return self.grid < _map(other)

    def __ge__(self, other):
        self._check_compatible(other)
        return self.grid >= _map(other)

    def __gt__(self, other):
        self._check_compatible(other)
        return self.grid > _map(other)

    def __add__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid + _map(other), points=self.points)

    def __sub__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid - _map(other), points=self.points)

    def __mul__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid * _map(other), points=self.points)

    def __truediv__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid / _map(other), points=self.points)

    def __floordiv__(self, other):
        self._check_compatible(other)
        return self.__class__(self.grid // _map(other), points=self.points)

    def __pow__(self, other):
        self._check_compatible(other)
        return self.__class__(np.power(self.grid, _map(other)), points=self.points)

    def __radd__(self, other):
        self._check_compatible(other)
        return self.__class__(_map(other) + self.grid, points=self.points)

    def __rsub__(self, other):
        self._check_compatible(other)
        return self.__class__(_map(other) - self.grid, points=self.points)

    def __rmul__(self, other):
        self._check_compatible(other)
        return self.__class__(_map(other) * self.grid, points=self.points)

    def __rtruediv__(self, other):
        self._check_compatible(other)
        return self.__class__(_map(other) / self.grid, points=self.points)

    def __rfloordiv__(self, other):
        self._check_compatible(other)
        return self.__class__(_map(other) // self.grid, points=self.points)

    def __rpow__(self, other):
        self._check_compatible(other)
        return self.__class__(np.power(_map(other), self.grid), points=self.points)

    def _update_grid_interpn(self, method='linear', bounds_error=False, fill_value=np.inf):
        """Update the internal grid interpolator (Map._grid_interpn).
        
        Args:
            method (str): Method of interpolation to perform ("linear" and "nearest"). Default is linear.
            bounds_error (bool): if True, when interpolated values are requested outside the grid, a ValueError
                is raised. If False, then the fill_value is used. Default is False.
            fill_value (float): If provided, the value to use for points outside of the interpolation domain.
                Default is np.inf.

        """
        self._grid_interpn = RegularGridInterpolator(self.points, self.grid, method, bounds_error, fill_value)

    def _load(self, grid, points=None, center=None, origin=None, delta=None):
        """Load and check energy grid."""
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
            delta = np.unique(np.around(np.diff(points), 3))
            if delta.size > 1:
                raise TypeError('The grid spacing (delta) must be the same in all dimension.')
            
            # Store info
            self.points = tuple(points)
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

                # Store info
                self.points = tuple(points)
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

                # Store info
                self.points = tuple(points)
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
        self.edges = tuple(edges)

        # Build KDTree
        self._kdtree = _build_kdtree_from_points(self.points)

        # Generate interprelator
        self._grid_interpn =  self._update_grid_interpn()

    def _load_autdock_map(self, filename):
        """Load energy map in AutoDock map format."""
        center = None
        grid = None
        npoints = None
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

    def load(self, filename):
        """Load existing energy map.
        
        The format of the input file will be deduced from the suffix of the filename

        Implemented formats: AutoDock map

        Args:
            filename (str): filename of the output file

        """
        file_format = _guess_format(filename)

        try:
            loader = self._loaders[file_format]
        except:
            error_msg = 'Cannot read %s file format. Available format: %s' % (file_format, self._loaders.keys())
            raise ValueError(error_msg)

        loader(filename)

    def _export_autodock_map(self, filename, **kwargs):
        """Export energy map in AutoDock map format."""
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
        """Export energy map to file.

        The format of the output file will be deduced from the suffix of the filename.

        Implemented formats: AutoDock map

        Args:
            filename (str): filename of the output file
            overwrite (bool): to allow overwriting over existing file (default: False)

        Kwargs:
            grid_parameter_file (str): For AutoDock map, name of the gpf file (default: NULL)
            grid_data_file (str): For AutoDock map, name of the fld file (default: NULL)
            macromolecule (str): For AutoDock map, name of the receptor (default: NULL)

        """
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

    def interpolate_energy(self, xyz, method='linear'):
        """Interpolate the energy from the grid for each coordinates xyz.

        Args:
            xyz (array_like): Array of 3D coordinates
            method (str): Interpolate method (default: linear)

        Returns:
            ndarray: 1d Numpy array of the energy values

        """
        return self._grid_interpn(xyz, method=method)

    def is_in_map(self, xyz):
        """Check if coordinates are in the map.

        Args:
            xyz (array_like): 3d coordinates (x, y, z) of a point

        Returns:
            ndarray: 1d Numpy array of boolean

        """
        xyz = np.atleast_2d(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

        x_in = np.logical_and(self.points[0][0] <= x, x <= self.points[0][-1])
        y_in = np.logical_and(self.points[1][0] <= y, y <= self.points[1][-1])
        z_in = np.logical_and(self.points[2][0] <= z, z <= self.points[2][-1])
        all_in = np.all((x_in, y_in, z_in), axis=0)

        return all_in

    def neighbor_points(self, xyz, radius, min_radius=0):
        """Grid coordinates around a point at a certain distance.

        Args:
            xyz (array_like): 3d coordinates (x, y, z) of a point
            radius (float): max radius in Angstrom
            min_radius (float): min radius in Angstrom (default: 0)

        Returns:
            ndarray: coordinates

        """
        coordinates = self._kdtree.data[self._kdtree.query_ball_point(xyz, radius)]
        
        if min_radius > 0:
            distances = spatial.distance.cdist([xyz], coordinates, "euclidean")[0]
            coordinates = coordinates[distances >= min_radius]

        return coordinates

    def _cartesian_to_index(self, xyz):
        """Return the closest grid index of the cartesian grid coordinates."""
        idx = np.rint((xyz - self._kdtree.mins) / self.delta).astype(np.int)
        # All the index values outside the grid are clipped (limited) to the nearest index
        np.clip(idx, [0, 0, 0], self.grid.shape, idx)

        return idx

    def add_bias(self, xyz, bias_value, radius):
        """Add energy bias to map using Juan's method.

        Args:
            xyz (array_like): array of 3d coordinates (x, y, z)
            bias_value (float): energy bias value to add (in kcal/mol)
            radius (float): radius of the bias (in Angtrom)

        """
        coordinates = np.atleast_2d(xyz)
        biased_grid = np.copy(self.grid)

        # We add all the bias one by one in the new map
        for coordinate in coordinates:
            sphere_xyz = self.neighbor_points(coordinate, radius)
            indexes = self._cartesian_to_index(sphere_xyz)

            distances = spatial.distance.cdist([coordinate], sphere_xyz, 'euclidean')[0]
            bias_energy = bias_value * np.exp(-1. * (distances ** 2) / (radius ** 2))

            biased_grid[indexes[:,0], indexes[:,1], indexes[:,2]] += bias_energy

        # And we replace the original grid with the biased version
        self.grid = biased_grid

        self._update_grid_interpn()

    def add_mask(self, xyz, mask_value, radius):
        """Add energy mask to map using Diogo's method.
        
        Args:
            xyz (array_like): array of 3d coordinates (x, y, z)
            mask_value (float): energy mask value to add (in kcal/mol)
            radius (float): radius of the mask (in Angtrom)

        """
        coordinates = np.atleast_2d(xyz)
        masked_grid = np.ones(shape=self.grid.shape) * mask_value

        # We add all the bias one by one in the new map
        for coordinate in coordinates:
            sphere_xyz = self.neighbor_points(coordinate, radius)
            indexes = self._cartesian_to_index(sphere_xyz)

            masked_grid[indexes[:,0], indexes[:,1], indexes[:,2]] = self.grid[indexes[:,0], indexes[:,1], indexes[:,2]]

        # And we replace the original grid with the masked version
        self.grid = masked_grid

        self._update_grid_interpn()

