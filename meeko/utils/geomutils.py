#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
#

import math
import sys

import numpy as np


def get_vector(coor1, coord2):
    """ calculate normalized vector between atoms"""
    vec = np.array([coord2[0] - coord1[0], coord2[1] - coord1[1], coord2[2] - coord1[2]], 'f')
    return normalize(vec)


def vector(a, b):
    """
    Return the vector between a and b
    """
    return b - a

def resize_vector(v, length, origin=None):
    """ Resize a vector v to a new length in regard to a origin """
    if origin is not None:
        return (normalize(v - origin) * length) + origin
    else:
        return normalize(v) * length


def get_vector_normal(vector):
    """return the first vector normal to the input vector"""
    return np.array(vector[1], vector[0], vector[2])


def normalize(v):
    """ numpy normalize"""
    return v / np.sqrt(np.dot(v, v))

###def centroid(self, atomlist):
###    """ calculate centroid """
###    centroid =  np.array([0., 0., 0.], 'f')
###    for i in atomlist:
###        centroid += obutils.getAtomCoord(p.getAtomCoord(i)
###    return centroid/len(atomlist)
###

def averageCoords(coordList):
    """ http://stackoverflow.com/questions/23020659/fastest-way-to-calculate-the-centroid-of-a-set-of-coordinate-tuples-in-python-wi"""
    avg = np.zeros(3)
    for c in coordList:
        avg += c
    return avg / len(coordList)


def calcPlaneVect(v1, v2, norm=True):
    """ calculate plane defined by two numpy.vectors"""
    # print "PLANE", v1, v2
    plane = np.cross(v1, v2)
    if not norm:
        return plane
    return normalize(plane)


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2)
    b, c, d = -axis*math.sin(theta/2)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])


def rotate_around_axis(vector, rot_axis, apply_point=[0., 0., 0.]):
    """
    Rotate a vector around an axis (rot_axis) applied to the point apply_point (NEW)

    Rotate a vector applied in apply_point around a pivot rot_axis ?
    vector = vector that is rotated
    rot_axis = vector around wich rotation is performed

        ?????? CHANGING THE INPUT VALUE?
    """
    # From Ludo
    # vector
    x = vector[0]
    y = vector[1]
    z = vector[2]

    # rotation pivot
    u = rot_axis[0]
    v = rot_axis[1]
    w = rot_axis[2]
    ux = u*x
    uy = u*y
    uz = u*z
    vx = v*x
    vy = v*y
    vz = v*z
    wx = w*x
    wy = w*y
    wz = w*z
    sa = math.sin(rot_axis[3])
    ca = math.cos(rot_axis[3])
    #vector[0]=(u*(ux+vy+wz)+(x*(v*v+w*w)-u*(vy+wz))*ca+(-wy+vz)*sa)+ apply_point[0]
    #vector[1]=(v*(ux+vy+wz)+(y*(u*u+w*w)-v*(ux+wz))*ca+(wx-uz)*sa)+ apply_point[1]
    #vector[2]=(w*(ux+vy+wz)+(z*(u*u+v*v)-w*(ux+vy))*ca+(-vx+uy)*sa)+ apply_point[2]
    p0 = (u*(ux+vy+wz)+(x*(v*v+w*w)-u*(vy+wz))*ca+(-wy+vz)*sa) + apply_point[0]
    p1 = (v*(ux+vy+wz)+(y*(u*u+w*w)-v*(ux+wz))*ca+(wx-uz)*sa) + apply_point[1]
    p2 = (w*(ux+vy+wz)+(z*(u*u+v*v)-w*(ux+vy))*ca+(-vx+uy)*sa) + apply_point[2]
    #b = [vector, m, rot_axis]

    return np.array([p0, p1, p2])


def rotation_axis(p0, p1, p2, origin=None):
    """
    Compute rotation axis centered at the origin if not None
    """
    r = normalize(np.cross(vector(p1, p0), vector(p2, p0)))

    if origin is not None:
        return origin + r

    return p0 + r


def atom_to_move(o, p):
    """
    Return the coordinates xyz of an atom just above acceptor/donor atom o
    """
    # It will not work if there is just one dimension
    p = np.atleast_2d(p)
    return o + normalize(-1. * vector(o, np.mean(p, axis=0)))


def rotate_point(p, p1, p2, angle):
    """ Rotate the point p around the axis p1-p2
    Source: http://paulbourke.net/geometry/rotate/PointRotate.py"""
    # Translate the point we want to rotate to the origin
    pn = p - p1

    # Get the unit vector from the axis p1-p2
    n = p2 - p1
    n = normalize(n)

    # Setup the rotation matrix
    c = np.cos(angle)
    t = 1. - np.cos(angle)
    s = np.sin(angle)
    x, y, z = n[0], n[1], n[2]

    R = np.array([[t*x**2 + c, t*x*y - s*z, t*x*z + s*y],
                 [t*x*y + s*z, t*y**2 + c, t*y*z - s*x],
                 [t*x*z - s*y, t*y*z + s*x, t*z**2 + c]])

    # ... and apply it
    ptr = np.dot(pn, R)

    # And to finish, we put it back
    p = ptr + p1

    return p


def getVecNormalToVec(vec):
    """
    calculate a vector that is normal to the numpy.array vector input

    Source: http://forums.create.msdn.com/forums/p/9551/50048.aspx
    A coworker pointed out a trick to get a vector perpendicular to the normal vector:
    simply swap two of the values, negate one of those, and zero the third.
    So, if I have a normal vector of form Vector3(a, b, c), then one such vector that
    is perpendicular to it is Vector3(b, -a, 0).  Thus, there are six possible vectors
    that are attainable by this method.  The only trouble case is when the normal vector
    contains elements whose values are zero, in which case you have to be a bit careful
    which values you swap and negate.  You just never want to end up with the zero vector.

    n n 0
    n 0 n
    0 n n
    n 0 0
    0 n 0
    n n n
    0 0 0

    if np.array([0., 0., 0.], 'f') == vec:
        print "Warning: zero vector, no normal possible"
        return vec
    x,y,z = vec
    zero = [ a == 0 for a in vec ]
    idx = range(3)
    swap1, swap2, zero = None, None, None
    for i in range(3):
        if not vec[i] == 0:
            negate = -vec[i]
        else:
            swap1 =
    """
    if (not vec[1] == 0) or (not vec[2] == 0):
        c = np.array([1.,0.,0.], 'f')
    else:
        c = np.array([0.,1.,0.], 'f')
    return calcPlane(vec, c, norm=True)


def calcDihedral(a1, a2, a3, a4):
    """ given 4 set of coordinates return the dihedral
        angle between them
    """
    v1 = vector(a1, a2)
    v2 = vector(a3, a2)
    v3 = vector(a3, a4)

    v4 = np.cross(v1, v2)
    v5 = np.cross(v2, v4)
    try:
        dihe = math.atan2(np.dot(v3,v4), np.dot(v3,v5) * math.sqrt(np.dot(v2,v2)))
    except ZeroDivisionError:
        dihe = 0.
    return dihe


def makeCircleOnPlane(center, r, normal, points = 8):
    """
    Calculate the points of a circle lying on an arbitrary plane
    defined by the normal vector.
    center : coords of center of the circle
    r      : radius
    normal : normal of the plane where the circle lies
    points : number of points for the circle

    # http://www.physicsforums.com/showthread.php?t=123168
    # P = Rcos(theta))U + Rsin(theta)N x U +c
    # Where u is a unit vector from the centre of the circle
    # to any point on the circumference; R is the radius;
    # n is a unit vector perpendicular to the plane and c is the centre of the circle.

    http://forums.create.msdn.com/forums/p/9551/50048.aspx
    A coworker pointed out a trick to get a vector perpendicular to the normal vector:
    simply swap two of the values, negate one of those, and zero the third.
    So, if I have a normal vector of form Vector3(a, b, c), then one such vector that
    is perpendicular to it is Vector3(b, -a, 0).  Thus, there are six possible vectors
    that are attainable by this method.  The only trouble case is when the normal vector
    contains elements whose values are zero, in which case you have to be a bit careful
    which values you swap and negate.  You just never want to end up with the zero vector.
    """
    N = normal
    U = array([N[1], -N[0], 0], 'f')
    step = PI2/points
    circle = []
    for i in range(points):
        theta = PI2-(step*i)
        P = (r*cos(theta)*U)+(r*sin(theta))*(cross(N,U))+center
        P = normalize(vector(center,P))*r
        P = vecSum(P,center)
        circle.append(P)
    return circle


def quickdist(f,s,sq = False):
    """ works with coordinates/vectors"""
    try:
        d=(f[0]-s[0])**2 + (f[1]-s[1])**2 + (f[2]-s[2])**2
        if sq: return math.sqrt(d)
        else:  return d
    except:
        print("First", f)
        print("Second", s)
        print("WARNING! missing coordinates", sys.exc_info()[1])
        raise Exception
        #return None


def atomsToVector(at1, at2=None, norm=0):

    at1 = atomCoord(at1)
    if at2: at2 = atomCoord(at2)
    return vector(at1, at2, norm=norm)


def vector(p1 , p2 = None, norm = 0): # TODO use Numpy?
    if not p2 is None:
        vec = np.array([p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]],'f')
    else:
        vec = np.array([p1[0], p1[1], p1[2] ], 'f' )
    if norm:
        return normalize(vec)
    else:
        return vec


def norm(A): # TODO use Numpy
    "Return vector norm"
    return np.sqrt(sum(A*A))


def normalize(A): # TODO use Numpy
    "Normalize the Vector"
    return A/norm(A)


def calcPlane(p1, p2, p3):
    # returns the plane containing the 3 input points
    v12 = vector(p1,p2)
    v13 = vector(p3,p2)
    return normalize(np.cross(v12, v13))


def dot(vector1, vector2):  # TODO remove and use Numpy
    dot_product = 0.
    for i in range(0, len(vector1)):
        dot_product += (vector1[i] * vector2[i])
    return dot_product


def vecAngle(v1, v2, rad=1): # TODO remove and use Numpy?
    angle = dot(normalize(v1), normalize(v2))
    if np.array_equal(v1, v2):
        return 0
    try:
        if rad:
            return math.acos(angle)
        else:
            return math.degrees(math.acos(angle))
    except ValueError:
        print("#vecAngle> CHECK TrottNormalization", v1, v2, sys.exc_info()[1])
        return 0


def absoluteAngleDifference(angle1, angle2, rad=1):
    """ https://gamedev.stackexchange.com/questions/4467/comparing-angles-and-working-out-the-difference"""
    ref = 180
    if rad:
        ref = math.radians(180)
    diff = ref - abs( abs(angle1 - angle2) - ref)
    return diff


def vecSum(vec1, vec2): # TODO remove and use Numpy # TODO to be used in the PDBQT+ data!
    return array([vec1[0]+vec2[0], vec1[1]+vec2[1], vec1[2]+vec2[2] ], 'f')


def normValue(v, vmin, vmax, normrange=[0,10]):
    # http://mathforum.org/library/drmath/view/60433.html
    # min = A
    # max = B
    # v   = x
    # y = 1 + (x-A)*(10-1)/(B-A)
    #return  1 + (v-vmin)*(10-1)/(vmax-vmin)
    return  normrange[0] + (v-vmin)*( normrange[1] )/(vmax-vmin)
    #top = (v-vmin)(10-1)
    #down = (vmax-vmin)
    #x =  1 + top/down
    #return x


def normProduct(a, b, mode = 'simple'):
    if mode =='simple': return a*b
    elif mode =='scaled': return (a*b)*(a+b)


def avgVector_untested(vec_list, normalize=False):
    # XXX NOT WORKING!!!
    # http://devmaster.net/forums/topic/5443-average-direction-vector/
    #weight = 1;
    #average = vec[0];
    #for (i = 1; i < n; ++i)
    #{
    #    find angle between average and vec[i];
    #    angle *= weight / (weight + 1);
    #    average = rotate vec[i] towards average by angle;
    #    ++weight;
    #}
    print("avgVector> NOT WORKING!!!! NEVER TESTED")

    weight = 1
    average = vec_list[0]
    for i in range(len(vec_list) - 1):
        angle = vecAngle(average, vec_list[i + 1])
        angle *= weight / (weight + 1)
        average = rotate_around_axis(vec_list[i + 1], m, ax)
        # XXX m?
        # XXX ax?
        weight += 1
    return average


def averageVector(vectorList, norm=True):
    """ """
    vector = np.array([0.,0.,0.], 'f')
    for v in vectorList:
        vector += v
    vector = vector/len(vectorList)
    if norm:
        vector = normalize(vector)
    return vector


def coplanar(plane, coord_list = [], reference = [0., 0., 0.], tolerance = 0.2):
    """ return list of coordinates that are within <tolerance>
        from the plane. If the reference is provided, vectors will be
        calculated with <reference> as origin.

    """
    coplane_list = []
    for c in coord_list:
        pos = vector(reference, c)
        if dot(plane, pos) <= tolerance:
            coplane_list.append(c)
    return coplane_list


def calcRingCentroidNormal(atomCoords):
    """ extract aromatic ring geometric info from a numpy array """
    a1 = atomCoords[0]
    a2 = atomCoords[1]
    a3 = atomCoords[2]

    centroid = averageCoords(atomCoords)
    plane = calcPlane(a1, a2, a3)
    v1 = vector(centroid, a1)
    v2 = vector(centroid, a2)
    normal1 = normalize(np.cross(v1, v2))
    normal2 = normalize(np.cross(v2, v1))
    centroid_norm1 = normalize(vector(centroid, normal1))
    centroid_norm2 = normalize(vector(centroid, normal2))
    return {'centroid':centroid, 'plane':plane, 'normals':[normal1, normal2],
            'centroid_normals':[centroid_norm1, centroid_norm2]}


def gaussian(x, ymax = 1., center=0., spread=0.7, invert=False):
    """ simple gaussian function"""
    if invert:
        invert = -1
    else:
        invert = 1
    return ymax * e **( -((x-center)**2/ (2*spread**2) ) ) * invert


def ellipticGaussian(coord, pseudo, planeVec, centroid=None, dist=None, ellipticity = 1.0,
    g_ymax=1.0, g_center=0.0, g_spread=1.2, g_invert=True):
    """ calculate elliptical gaussian potential based on the angle with the plane
          ____
       .--        __.M
      <       P--'
       `''----|
              |
        ------O-----  plane
    """
    vec = vector(coord, pseudo)
    theta = vecAngle( vec, planeVec)
    if dist is None:
        d = quickdist(coord, pseudo, sq=1)
    else:
        d = dist
    d += (d * ellipticity * (math.fabs(math.cos(theta))))
    ## this code creates a concave cone-shaped ring
    ## d += -(d * ellipticity * (math.fabs(math.cos(theta))))
    gvalue = gaussian(d, ymax = g_ymax, center = g_center,
            spread=g_spread, invert=g_invert)
    return gvalue
