from os import linesep as os_linesep
import pathlib
import numpy as np

def get_gpf_string(center, size_angstrom, rec_fname, rec_types, lig_types, map_prefix=None, dielectric=-42, smooth=0.5, spacing=0.375, ff_param_fname=None):

    size_x, size_y, size_z = size_angstrom
    # the following guarantees an EVEN number of grid points
    npts_x = 2 * int(size_x / (2 * spacing))
    npts_y = 2 * int(size_y / (2 * spacing))
    npts_z = 2 * int(size_z / (2 * spacing))

    if map_prefix is None:
        map_prefix = pathlib.Path(rec_fname).with_suffix("").name

    gpf = ( "npts {gpf_npts_x:d} {gpf_npts_y:d} {gpf_npts_z:d}\n"
            "gridfld {map_prefix:s}.maps.fld\n"
            "spacing {gpf_spacing:1.3f}\n"
            "receptor_types {rec_types:s}\n"
            "ligand_types {lig_types_std:s}\n"
            "receptor {rec_fname:s}\n"
            "gridcenter {gpf_gridcenter_x:3.3f} {gpf_gridcenter_y:3.3f} {gpf_gridcenter_z:3.3f}\n"
            "smooth {gpf_smooth:3.3f}\n"
            "{mapfiles_std:s}"
            "elecmap {map_prefix:s}.e.map\n"
            "dsolvmap {map_prefix:s}.d.map\n"
            "dielectric {gpf_dielectric:3.3f}\n")

    if ff_param_fname is not None:
        gpf = "parameter_file %s\n" % ff_param_fname + gpf

    info = {}
    info["gpf_spacing"] = spacing
    info["gpf_smooth"] = smooth
    info["gpf_dielectric"] = dielectric
    info['map_prefix'] = map_prefix
    info['ff_param_fname'] = ff_param_fname
    info['rec_types'] = ' '.join(rec_types)
    info['rec_fname'] = rec_fname
    info['gpf_npts_x'] = npts_x
    info['gpf_npts_y'] = npts_y
    info['gpf_npts_z'] = npts_z
    info['gpf_gridcenter_x'] = center[0]
    info['gpf_gridcenter_y'] = center[1]
    info['gpf_gridcenter_z'] = center[2]

    mapfiles = ""
    for atype in lig_types:
        mapfiles+= "map %s.%s.map\n" % (map_prefix, atype)

    info['lig_types_std'] = " ".join(lig_types)
    info['mapfiles_std'] = mapfiles
    return gpf.format(**info), (npts_x, npts_y, npts_z)

def box_to_pdb_string(box_center, npts, spacing=0.375):
    """
        8 ______ 7
         /.    /|
      4 /_.___/3|
        | . X | |  <-- 9
        |5....|./6
        |.____|/
       1      2

    """

    step_x = int(npts[0] / 2.0) * spacing
    step_y = int(npts[1] / 2.0) * spacing
    step_z = int(npts[2] / 2.0) * spacing
    center_x, center_y, center_z = box_center
    corners = []
    corners.append([center_x - step_x, center_y - step_y, center_z - step_z] ) # 1
    corners.append([center_x + step_x, center_y - step_y, center_z - step_z] ) # 2
    corners.append([center_x + step_x, center_y + step_y, center_z - step_z] ) # 3
    corners.append([center_x - step_x, center_y + step_y, center_z - step_z] ) # 4
    corners.append([center_x - step_x, center_y - step_y, center_z + step_z] ) # 5
    corners.append([center_x + step_x, center_y - step_y, center_z + step_z] ) # 6
    corners.append([center_x + step_x, center_y + step_y, center_z + step_z] ) # 7
    corners.append([center_x - step_x, center_y + step_y, center_z + step_z] ) # 8

    count = 1
    res = "BOX"
    chain = "X"
    pdb_out = ""
    line = "ATOM  %5d %4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00 10.00          %1s" + os_linesep
    for idx in range(len(corners)):
        x = corners[idx][0]
        y = corners[idx][1]
        z = corners[idx][2]
        pdb_out += line % (count, "Ne", res, chain, idx+1, x, y, z, "Ne")
        count += 1

    # center
    pdb_out += line % (count+1, "Xe", res, chain, idx+1, center_x, center_y, center_z, "Xe")

    pdb_out += "CONECT    1    2" + os_linesep
    pdb_out += "CONECT    1    4" + os_linesep
    pdb_out += "CONECT    1    5" + os_linesep
    pdb_out += "CONECT    2    3" + os_linesep
    pdb_out += "CONECT    2    6" + os_linesep
    pdb_out += "CONECT    3    4" + os_linesep
    pdb_out += "CONECT    3    7" + os_linesep
    pdb_out += "CONECT    4    8" + os_linesep
    pdb_out += "CONECT    5    6" + os_linesep
    pdb_out += "CONECT    5    8" + os_linesep
    pdb_out += "CONECT    6    7" + os_linesep
    pdb_out += "CONECT    7    8" + os_linesep
    return pdb_out

def is_point_outside_box(point, center, npts, spacing=0.375):

    # Autogrid always outputs an odd number of grid points, roundup if even.
    # An odd number of grid poins is an even number of voxels.
    halfvoxels = np.array(npts) / 2
    step = spacing * halfvoxels.astype(int)
    mincorner = center - step
    maxcorner = center + step
    x, y, z = point
    is_outside = False
    is_outside |= x >= maxcorner[0] or x <= mincorner[0]
    is_outside |= y >= maxcorner[1] or y <= mincorner[1]
    is_outside |= z >= maxcorner[2] or z <= mincorner[2]
    return is_outside

def calc_box(coord_array, padding):
    """Calulate gridbox around given coordinate array with given padding

    Args:
        coord_array (array-like): array of XYZ coordinates as strings or floats
        padding (string/float): padding (in angstroms) for box around given coordinates

    Returns:
        tuple: tuple of tuples with center coordinates (angstroms) and dimension sizes (angstroms) for box
    """    
    padding = float(padding)
    xa = [float(c[0]) for c in coord_array]
    ya = [float(c[1]) for c in coord_array]
    za = [float(c[2]) for c in coord_array]
    x_min = min(xa)
    x_max = max(xa)
    y_min = min(ya)
    y_max = max(ya)
    z_min = min(za)
    z_max = max(za)
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    center_z = (z_min + z_max) / 2.0
    size_x = x_max - x_min + 2 * padding
    size_y = y_max - y_min + 2 * padding
    size_z = z_max - z_min + 2 * padding
    return (center_x, center_y, center_z), (size_x, size_y, size_z)

boron_silicon_atompar  = "atom_par Si     4.10  0.200  35.8235  -0.00143  0.0  0.0  0  -1  -1  6" + os_linesep
boron_silicon_atompar += "atom_par B      3.84  0.155  29.6478  -0.00152  0.0  0.0  0  -1  -1  0" + os_linesep
