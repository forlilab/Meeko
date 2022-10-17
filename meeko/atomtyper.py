#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko atom typer
#

import json
import os
import pathlib

import numpy as np

from .utils import utils
from .utils import pdbutils

try:
    import espaloma
except ImportError:
    _has_espaloma = False
else:
    _has_espaloma = True

try:
    import torch
except ImportError:
    _has_torch = False
else:
    _has_torch = True


class AtomTyper:

    defaults_json = """{
        "ATOM_PARAMS": {
            "alkyl glue": [
                {"smarts": "[#1]",                  "atype": "H", "comment": "invisible"},
                {"smarts": "[#1][#7,#8,#9,#15,#16]","atype": "HD"},
                {"smarts": "[#5]",              "atype": "B"},
                {"smarts": "[C]",               "atype": "C"},
                {"smarts": "[c]",               "atype": "A"},
                {"smarts": "[#7]",              "atype": "NA"},
                {"smarts": "[#8]",              "atype": "OA"},
                {"smarts": "[#9]",              "atype": "F"},
                {"smarts": "[#12]",             "atype": "Mg"},
                {"smarts": "[#14]",             "atype": "Si"},
                {"smarts": "[#15]",             "atype": "P"},
                {"smarts": "[#16]",             "atype": "S"},
                {"smarts": "[#17]",             "atype": "Cl"},
                {"smarts": "[#20]",             "atype": "Ca"},
                {"smarts": "[#25]",             "atype": "Mn"},
                {"smarts": "[#26]",             "atype": "Fe"},
                {"smarts": "[#30]",             "atype": "Zn"},
                {"smarts": "[#35]",             "atype": "Br"},
                {"smarts": "[#53]",             "atype": "I"},
                {"smarts": "[#7X3v3][a]",       "atype": "N",  "comment": "pyrrole, aniline"},
                {"smarts": "[#7X3v3][#6X3v4]",  "atype": "N",  "comment": "amide"},
                {"smarts": "[#7+1]",            "atype": "N",  "comment": "ammonium, pyridinium"},
                {"smarts": "[SX2]",             "atype": "SA", "comment": "sulfur acceptor"}
            ],
            "ad4_desolv_volume": [
                {"smarts": "[#1]",               "ad4_sol_vol":  0.0},
                {"smarts": "[#1][#8,#7,#16,#9]", "ad4_sol_vol":  0.0},
                {"smarts": "[#6]",               "ad4_sol_vol": 33.5103},
                {"smarts": "[#7]",               "ad4_sol_vol": 22.4493},
                {"smarts": "[#8]",               "ad4_sol_vol": 17.1573},
                {"smarts": "[#9]",               "ad4_sol_vol": 15.448 },
                {"smarts": "[#12]",              "ad4_sol_vol":  1.56  },
                {"smarts": "[#15]",              "ad4_sol_vol": 38.7924},
                {"smarts": "[#16]",              "ad4_sol_vol": 33.5103},
                {"smarts": "[#17]",              "ad4_sol_vol": 35.8235},
                {"smarts": "[#20]",              "ad4_sol_vol":  2.77  },
                {"smarts": "[#25]",              "ad4_sol_vol":  2.14  },
                {"smarts": "[#26]",              "ad4_sol_vol":  1.84  },
                {"smarts": "[#30]",              "ad4_sol_vol":  1.7   },
                {"smarts": "[#35]",              "ad4_sol_vol": 42.5661},
                {"smarts": "[#53]",              "ad4_sol_vol": 55.0585}
            ],
            "ad4_desolv_param": [
                {"smarts": "[*]",                "ad4_sol_par":  0.00110},
                {"smarts": "[#1]",               "ad4_sol_par":  0.00000},
                {"smarts": "[#1][#8,#7,#16,#9]", "ad4_sol_par":  0.00051},
                {"smarts": "[C]",                "ad4_sol_par": -0.00143},
                {"smarts": "[c]",                "ad4_sol_par": -0.00052},
                {"smarts": "[#7]",               "ad4_sol_par": -0.00162},
                {"smarts": "[#8]",               "ad4_sol_par": -0.00251},
                {"smarts": "[#16]",              "ad4_sol_par": -0.00214}
            ]
        },
        "OFFATOMS": {
            "example_offsite_charges": [
                {"smarts": "[#7X2;v3;!+](=,:[*])[*]", "IDX": [1], "OFFATOMS": [
                    {"z": [2, 3], "phi": 0, "distance": 0.2,
                    "atype": "OFFCHRG", "pull_charge_fraction": 1.08}
                ]}
            ]
        },
        "CHARGE_MODEL": "gasteiger"
    }
    """
    def __init__(self, parameters={}):
        self.parameters = json.loads(self.defaults_json)
        for key in parameters:
            self.parameters[key] = json.loads(json.dumps(parameters[key])) # a safe copy

    def __call__(self, setup):
        self._type_atoms(setup)
        self._type_dihedrals(setup)
        # CHARGE_MODEL must preceed OFFATOMS because of offsite-charges 
        if "CHARGE_MODEL" in self.parameters:
            if self.parameters["CHARGE_MODEL"] == "espaloma":
                set_espaloma_charges(setup) 
            elif self.parameters["CHARGE_MODEL"] == "gasteiger":
                pass # gasteiger automatically set in molsetup
            else:
                raise RuntimeError("only espaloma/gasteiger charges accepted")
        if 'OFFATOMS' in self.parameters:
            cached_offatoms = self._cache_offatoms(setup)
            coords = [x for x in setup.coord.values()]
            self._set_offatoms(setup, cached_offatoms, coords)
        return

    def _type_atoms(self, setup):
        parsmar = self.parameters['ATOM_PARAMS']
        # ensure every "atompar" is defined in a single "smartsgroup"
        ensure = {}
        # go over all "smartsgroup"s
        for smartsgroup in parsmar:
            if smartsgroup == 'comment': continue
            for line in parsmar[smartsgroup]: # line is a dict, e.g. {"smarts": "[#1][#7,#8,#9,#15,#16]","atype": "HD"}
                smarts = str(line['smarts'])
                # get indices of the atoms in the smarts to which the parameters will be assigned
                idxs = [0] # by default, the first atom in the smarts gets parameterized
                if 'IDX' in line:
                    idxs = [i - 1 for i in line['IDX']] # convert from 1- to 0-indexing
                # match SMARTS
                hits = setup.find_pattern(smarts)
                for atompar in line:
                    if atompar in ["smarts", "comment", "IDX"]: continue
                    if atompar not in setup.atom_params:
                        setup.atom_params[atompar] = [None] * len(setup.coord) 
                    value = line[atompar]
                    # keep track of every "smartsgroup" that modified "atompar"
                    ensure.setdefault(atompar, [])
                    ensure[atompar].append(smartsgroup)
                    # Each "hit" is a tuple of atom indices that matched the smarts
                    # The length of each "hit" is the number of atoms in the smarts
                    for hit in hits:
                        # Multiple atoms may be targeted by a single smarts:
                        # For example: both oxygens in NO2 are parameterized by a single smarts pattern.
                        # "idxs" are 1-indeces of atoms in the smarts to which parameters are to be assigned.
                        for idx in idxs:
                            if atompar == "atype":
                                setup.set_atom_type(hit[idx], value) # overrides previous calls
                            setup.atom_params[atompar][hit[idx]] = value
        # guarantee that each atompar is exclusive of a single group
        for atompar in ensure:
            if len(set(ensure[atompar])) > 1:
                msg = 'WARNING: %s is modified in multiple smartsgroups: %s' % (atompar, set(ensure[atompar]))
                print(msg)
        return


    def _cache_offatoms(self, setup):
        """ precalculate off-site atoms """
        parsmar = self.parameters['OFFATOMS']
        cached_offatoms = {}
        n_offatoms = 0
        atoms_with_offchrg = set()
        # each parent atom can only be matched once in each smartsgroup
        for smartsgroup in parsmar:
            if smartsgroup == "comment": continue
            tmp = {}
            for line in parsmar[smartsgroup]:
                # SMARTS
                smarts = str(line['smarts'])
                hits = setup.find_pattern(smarts)
                # atom indexes in smarts string
                smarts_idxs = [0]
                if 'IDX' in line:
                    smarts_idxs = [i - 1 for i in line['IDX']]
                for smarts_idx in smarts_idxs:
                    for hit in hits:
                        parent_idx = hit[smarts_idx]
                        tmp.setdefault(parent_idx, []) # TODO tmp[parent_idx] = [], yeah?
                        for offatom in line['OFFATOMS']:
                            # set defaults
                            tmp[parent_idx].append(
                                {'offatom': {'distance': 1.0,
                                             'x90': False,
                                             'phi': 0.0,
                                             'theta': 0.0,
                                             'z': [],
                                             'x': []},
                                 'atom_params': {}
                                })
                            for key in offatom:
                                if key in ['distance', 'x90']:
                                    tmp[parent_idx][-1]['offatom'][key] = offatom[key]
                                # replace SMARTS indexes by the atomic index
                                elif key in ['z', 'x']:
                                    for i in offatom[key]:
                                        idx = hit[i - 1]
                                        tmp[parent_idx][-1]['offatom'][key].append(idx)
                                # convert degrees to radians
                                elif key in ['theta', 'phi']:
                                    tmp[parent_idx][-1]['offatom'][key] = np.radians(offatom[key])
                                # ignore comments
                                elif key in ['comment']:
                                    pass
                                elif key == 'atype':
                                    tmp[parent_idx][-1]['atom_params'][key] = offatom[key]
                                elif key == 'pull_charge_fraction':
                                    if parent_idx in atoms_with_offchrg:
                                        raise RuntimeError("atom %d has charge pulled more than once" % parent_idx)
                                    atoms_with_offchrg.add(parent_idx)
                                    tmp[parent_idx][-1]['atom_params'][key] = offatom[key]
                                else:
                                    pass
            for parent_idx in tmp:
                for offatom_dict in tmp[parent_idx]:
                    #print '1-> ', self.atom_params['q'], len(self.coords)
                    atom_params = offatom_dict['atom_params']
                    offatom = offatom_dict['offatom']
                    atomgeom = AtomicGeometry(parent_idx,
                                              neigh=offatom['z'],
                                              xneigh=offatom['x'],
                                              x90=offatom['x90'])
                    if "pull_charge_fraction" in atom_params:
                        pull_charge_fraction = atom_params["pull_charge_fraction"]
                    else:
                        pull_charge_fraction = 0.0
                    args = (atom_params['atype'],
                            offatom['distance'],
                            offatom['theta'],
                            offatom['phi'],
                            pull_charge_fraction)
                    # number of coordinates (before adding new offatom)
                    cached_offatoms[n_offatoms] = (atomgeom, args)
                    n_offatoms += 1
        return cached_offatoms

    def _set_offatoms(self, setup, cached_offatoms, coords):
        """add cached offatoms"""
        for k, (atomgeom, args) in cached_offatoms.items():
            (atom_type, dist, theta, phi, pull_charge_fraction) = args
            offatom_coords = atomgeom.calc_point(dist, theta, phi, coords)
            tmp = setup.get_pdbinfo(atomgeom.parent+1)
            pdbinfo = pdbutils.PDBAtomInfo('G', tmp.resName, tmp.resNum, tmp.chain)
            q_parent = (1 - pull_charge_fraction) * setup.charge[atomgeom.parent] 
            q_offsite = pull_charge_fraction * setup.charge[atomgeom.parent]
            pseudo_atom = {
                    'coord': offatom_coords,
                    'anchor_list': [atomgeom.parent],
                    'charge': q_offsite,
                    'pdbinfo': pdbinfo,
                    'atom_type': atom_type,
                    'rotatable': False
                    }
            setup.charge[atomgeom.parent] = q_parent
            setup.add_pseudo(**pseudo_atom)
        return

    def _type_dihedrals(self, molsetup):

        if "DIHEDRALS" not in self.parameters:
            return
        parsmar = self.parameters['DIHEDRALS']
        dihedrals = {}

        for line in parsmar:
            smarts = str(line['smarts'])
            hits = molsetup.find_pattern(smarts)
            if len(hits) == 0:
                continue # non-rotatable bonds get dihedrals
            idxs = [i - 1 for i in line['IDX']]
            tid = line["id"] if "id" in line else None
            fourier_series = []
            term_indices = {}
            for key in line:
                for keyword in ['phase', 'k', 'periodicity', 'idivf']:
                    if key.startswith(keyword):
                        t = int(key.replace(keyword, '')) # e.g. "phase2" -> int(2)
                        if t not in term_indices:
                            term_indices[t] = len(fourier_series)
                            fourier_series.append({})
                        index = term_indices[t]
                        fourier_series[index][keyword] = line[key]
                        break

            for index in range(len(fourier_series)):
                if "idivf" in fourier_series[index]:
                    idivf = fourier_series[index].pop("idivf")
                    fourier_series[index]["k"] /= idivf

            dihedral_index = molsetup.add_dihedral_interaction(fourier_series)

            for hit in hits:
                atom_idxs = tuple([hit[j] for j in idxs])
                molsetup.dihedral_partaking_atoms[atom_idxs] = dihedral_index
                molsetup.dihedral_labels[atom_idxs] = tid



class AtomicGeometry():
    """generate reference frames and add extra sites"""

    def __init__(self, parent, neigh, xneigh=[], x90=False, planar_tol=0.1):
        """arguments are indices of atoms"""

        self.planar_tol = planar_tol # angstroms, length of neighbor vecs for z-axis

        # real atom hosting extra sites
        if type(parent) != int:
            raise RuntimeError('parent must be int')
        self.parent = parent

        # list of bonded atoms (used to define z-axis)
        self.neigh = []
        for i in neigh:
            if type(i) != int:
                raise RuntimeError('neigh indices must be int')
            self.neigh.append(i)

        # list of atoms that
        self.xneigh = []
        for i in xneigh:
            if type(i) != int:
                raise RuntimeError('xneigh indices must be int')
            self.xneigh.append(i)

        self.calc_x = len(self.xneigh) > 0
        self.x90 = x90 # y axis becomes x axis (useful to rotate in-plane by 90 deg)

    def calc_point(self, distance, theta, phi, coords):
        """return coordinates of point specified in spherical coordinates"""

        z = self._calc_z(coords)

        # return pt aligned with z-axis
        if phi == 0.:
            return z * distance + np.array(coords[self.parent])

        # need x-vec if phi != 0
        elif self.calc_x == False:
            raise RuntimeError('phi must be zero if X undefined')

        else:
            x = self._calc_x(coords)
            if self.x90:
                x = np.cross(self.z, x)
            y = np.cross(z, x)
            pt = z * distance
            pt = self.rot3D(pt, y, phi)
            pt = self.rot3D(pt, z, theta)
            pt += np.array(coords[self.parent])
            return pt

    def _calc_z(self, coords):
        """ maximize distance from neigh """
        z = np.zeros(3)
        cumsum = np.zeros(3)
        for i in self.neigh:
            v = np.array(coords[self.parent]) - np.array(coords[i])
            cumsum += v
            z += self.normalized(v)
        z = self.normalized(z)
        if np.sum(cumsum**2) < self.planar_tol**2:
            raise RuntimeError('Refusing to place Z axis on planar atom')
        return z

    def _calc_x(self, coords):
        x = np.zeros(3)
        for i in self.xneigh:
            v = np.array(coords[self.parent]) - np.array(coords[i])
            x += self.normalized(v)
        x = self.normalized(x)
        return x

    @staticmethod
    def rot3D(pt, ax, rad):
        """
            Rotate point:
            pt = (x,y,z) coordinates to be rotated
            ax = vector around wich rotation is performed
            rad = rotate by "rad" radians
        """
        # If axis has len=0, rotate by 0.0 rad on any axis
        # Make sure ax has unitary length
        len_ax = (ax[0]**2 + ax[1]**2 + ax[2]**2)**0.5
        if len_ax == 0.:
            u, v, w = (1, 0, 0)
            rad = 0.0
        else:
            u, v, w = [i/len_ax for i in ax]
        x, y, z = pt
        ux, uy, uz = u*x, u*y, u*z
        vx, vy, vz = v*x, v*y, v*z
        wx, wy, wz = w*x, w*y, w*z
        sa=np.sin(rad)
        ca=np.cos(rad)
        p0 =(u*(ux+vy+wz)+(x*(v*v+w*w)-u*(vy+wz))*ca+(-wy+vz)*sa)
        p1=(v*(ux+vy+wz)+(y*(u*u+w*w)-v*(ux+wz))*ca+(wx-uz)*sa)
        p2=(w*(ux+vy+wz)+(z*(u*u+v*v)-w*(ux+vy))*ca+(-vx+uy)*sa)
        return (p0, p1, p2)


    def normalized(self, vec):
        l = sum([x**2 for x in vec])**0.5
        if type(vec) == list:
            return [x/l for x in vec]
        else:
            # should be np.array
            return vec / l

def set_espaloma_charges(molsetup):
    if not _has_espaloma or not _has_torch:
        raise ImportError("espaloma and pytorch are required")
    pretrained_model = pathlib.Path(espaloma.__file__).parents[1] / "espaloma_model.pt"
    if not pretrained_model.exists():
        msg = "Could not find %s" % pretrained_model
        msg += "Consider:"
        msg += "    $ cd %s" % pretrained_model.parents[0]
        msg += "    $ wget http://data.wangyq.net/espaloma_model.pt"
        raise RuntimeError(msg)
    from .molsetup import RDKitMoleculeSetup
    if not isinstance(molsetup, RDKitMoleculeSetup):
        raise NotImplementedError("need rdkit molecule for espaloma charges")
    from openff.toolkit.topology import Molecule
    rdmol = molsetup.mol
    openffmol = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)
    molgraph = espaloma.Graph(openffmol)
    espaloma_model = torch.load(pretrained_model)
    espaloma_model(molgraph.heterograph)
    charges = [float(q) for q in molgraph.nodes["n1"].data["q"]]
    total_charge = 0.0
    for i in range(len(charges)):
        #print("%12.4f %12.4f" % (molsetup.charge[i], charges[i]))
        molsetup.charge[i] = charges[i] 
        total_charge += charges[i]
    for j in range(i+1, len(molsetup.charge)):
        if molsetup.charge[j] != 0.:
            raise RuntimeError("expected zero charge beyond real atoms, at this point") 
