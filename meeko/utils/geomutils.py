#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Geometry utilities (vectors, rotations, dihedrals).
#

import math
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Vectors
# ---------------------------------------------------------------------------

def vector(p1, p2=None, norm=False):
    """Vector from ``p1`` to ``p2``.

    Single-arg form ``vector(p)`` returns ``p`` itself as a numpy array.
    Two-arg form ``vector(p1, p2)`` returns ``p2 - p1``. Set ``norm=True``
    to normalize the result.
    """
    p1 = np.asarray(p1, dtype=float)
    if p2 is None:
        vec = p1
    else:
        vec = np.asarray(p2, dtype=float) - p1
    if norm:
        return normalize(vec)
    return vec


def normalize(v):
    """Return a unit-length copy of ``v``."""
    return v / np.sqrt(np.dot(v, v))


def norm(v):
    """Return the Euclidean norm (length) of ``v``."""
    return np.sqrt(np.dot(v, v))


def resize_vector(v, length, origin=None):
    """Rescale ``v`` to have ``length``. If ``origin`` is given, the
    rescaling is done relative to it (the returned vector starts at
    ``origin`` and has the requested length toward ``v``).
    """
    if origin is not None:
        return (normalize(v - origin) * length) + origin
    return normalize(v) * length


def dot(v1, v2):
    """Dot product of two vectors."""
    return float(np.dot(v1, v2))


def vecAngle(v1, v2, rad=True):
    """Angle (radians by default) between two vectors. Returns 0 for
    identical vectors. ``rad=False`` returns degrees.
    """
    if np.array_equal(v1, v2):
        return 0
    angle = np.dot(normalize(v1), normalize(v2))
    try:
        result = math.acos(angle)
    except ValueError:
        print("#vecAngle> CHECK TrottNormalization", v1, v2, sys.exc_info()[1])
        return 0
    return result if rad else math.degrees(result)


def absoluteAngleDifference(angle1, angle2, rad=True):
    """Smallest absolute angular distance between two angles.

    https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference
    """
    ref = math.radians(180) if rad else 180
    return ref - abs(abs(angle1 - angle2) - ref)


def averageCoords(coordList):
    """Centroid of a list of coordinate triples."""
    avg = np.zeros(3)
    for c in coordList:
        avg += c
    return avg / len(coordList)


def averageVector(vectorList, norm=True):
    """Average of a list of vectors, optionally normalized."""
    out = np.zeros(3, dtype=float)
    for v in vectorList:
        out += v
    out /= len(vectorList)
    if norm:
        out = normalize(out)
    return out


def quickdist(f, s, sq=False):
    """Distance (or squared distance) between two coordinate triples."""
    try:
        d = (f[0] - s[0]) ** 2 + (f[1] - s[1]) ** 2 + (f[2] - s[2]) ** 2
        return math.sqrt(d) if sq else d
    except Exception:
        print("First", f)
        print("Second", s)
        print("WARNING! missing coordinates", sys.exc_info()[1])
        raise


# ---------------------------------------------------------------------------
# Planes
# ---------------------------------------------------------------------------

def calcPlane(p1, p2, p3):
    """Normal of the plane through three points."""
    v12 = vector(p1, p2)
    v13 = vector(p3, p2)
    return normalize(np.cross(v12, v13))


def calcPlaneVect(v1, v2, normalized=True):
    """Plane defined by two vectors (cross product), normalized by default."""
    plane = np.cross(v1, v2)
    if not normalized:
        return plane
    return normalize(plane)


def coplanar(plane, coord_list=(), reference=(0.0, 0.0, 0.0), tolerance=0.2):
    """Return coords in ``coord_list`` whose offset from ``reference`` lies
    within ``tolerance`` of being coplanar with ``plane``.
    """
    coplane_list = []
    for c in coord_list:
        pos = vector(reference, c)
        if dot(plane, pos) <= tolerance:
            coplane_list.append(c)
    return coplane_list


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def rotation_matrix(axis, theta):
    """Rotation matrix for counterclockwise rotation about ``axis`` by
    ``theta`` radians.

    Source: https://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2)
    b, c, d = -axis * math.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotate_around_axis(vec, rot_axis, apply_point=(0.0, 0.0, 0.0)):
    """Rotate ``vec`` around ``rot_axis`` (a 4-tuple ``(u, v, w, angle)``)
    and translate by ``apply_point``.
    """
    x, y, z = vec[0], vec[1], vec[2]
    u = rot_axis[0]
    v = rot_axis[1]
    w = rot_axis[2]
    ux = u * x
    uy = u * y
    uz = u * z
    vx = v * x
    vy = v * y
    vz = v * z
    wx = w * x
    wy = w * y
    wz = w * z
    sa = math.sin(rot_axis[3])
    ca = math.cos(rot_axis[3])
    p0 = (
        u * (ux + vy + wz)
        + (x * (v * v + w * w) - u * (vy + wz)) * ca
        + (-wy + vz) * sa
    ) + apply_point[0]
    p1 = (
        v * (ux + vy + wz)
        + (y * (u * u + w * w) - v * (ux + wz)) * ca
        + (wx - uz) * sa
    ) + apply_point[1]
    p2 = (
        w * (ux + vy + wz)
        + (z * (u * u + v * v) - w * (ux + vy)) * ca
        + (-vx + uy) * sa
    ) + apply_point[2]
    return np.array([p0, p1, p2])


def rotation_axis(p0, p1, p2, origin=None):
    """Axis perpendicular to the plane through (p0, p1, p2), centered at
    ``origin`` if given (else at p0)."""
    r = normalize(np.cross(vector(p1, p0), vector(p2, p0)))
    if origin is not None:
        return origin + r
    return p0 + r


def atom_to_move(o, p):
    """Return a coordinate one bond above the centroid of ``p`` along the
    direction opposite to ``o → mean(p)``. Used to find lone-pair positions.
    """
    p = np.atleast_2d(p)
    return o + normalize(-1.0 * vector(o, np.mean(p, axis=0)))


def rotate_point(p, p1, p2, angle):
    """Rotate point ``p`` by ``angle`` (rad) around the axis ``p1`` → ``p2``.

    Source: http://paulbourke.net/geometry/rotate/PointRotate.py
    """
    pn = p - p1
    n = normalize(p2 - p1)

    c = np.cos(angle)
    t = 1.0 - np.cos(angle)
    s = np.sin(angle)
    x, y, z = n[0], n[1], n[2]
    R = np.array(
        [
            [t * x ** 2 + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y ** 2 + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z ** 2 + c],
        ]
    )
    return np.dot(pn, R) + p1


# ---------------------------------------------------------------------------
# Dihedrals
# ---------------------------------------------------------------------------

def calcDihedral(A, B, C, D):
    """Dihedral angle (rad) for the four points A, B, C, D."""
    A, B, C, D = [np.array(x) for x in (A, B, C, D)]
    b1 = B - A
    b2 = C - B
    b3 = D - C
    temp = np.linalg.norm(b2) * b1
    y = np.dot(temp, np.cross(b2, b3))
    x = np.dot(np.cross(b1, b2), np.cross(b2, b3))
    return np.arctan2(y, x)


def calcDihedral_old(a1, a2, a3, a4):
    """Legacy dihedral angle from four coordinates."""
    v1 = vector(a1, a2)
    v2 = vector(a3, a2)
    v3 = vector(a3, a4)
    v4 = np.cross(v1, v2)
    v5 = np.cross(v2, v4)
    try:
        return math.atan2(np.dot(v3, v4), np.dot(v3, v5) * math.sqrt(np.dot(v2, v2)))
    except ZeroDivisionError:
        return 0.0


# ---------------------------------------------------------------------------
# Aromatic-ring geometry
# ---------------------------------------------------------------------------

def calcRingCentroidNormal(atomCoords):
    """Centroid + plane normal info for a ring's atomic coordinates."""
    a1, a2 = atomCoords[0], atomCoords[1]
    centroid = averageCoords(atomCoords)
    plane = calcPlane(a1, a2, atomCoords[2])
    v1 = vector(centroid, a1)
    v2 = vector(centroid, a2)
    normal1 = normalize(np.cross(v1, v2))
    normal2 = normalize(np.cross(v2, v1))
    centroid_norm1 = normalize(vector(centroid, normal1))
    centroid_norm2 = normalize(vector(centroid, normal2))
    return {
        "centroid": centroid,
        "plane": plane,
        "normals": [normal1, normal2],
        "centroid_normals": [centroid_norm1, centroid_norm2],
    }


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def normValue(v, vmin, vmax, normrange=(0, 10)):
    """Linearly map ``v`` from ``[vmin, vmax]`` into ``normrange``."""
    return normrange[0] + (v - vmin) * (normrange[1]) / (vmax - vmin)


def normProduct(a, b, mode="simple"):
    if mode == "simple":
        return a * b
    if mode == "scaled":
        return (a * b) * (a + b)
    raise ValueError(f"unknown mode {mode!r}")
