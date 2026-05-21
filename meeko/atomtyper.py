#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import pathlib
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .utils import pdbutils
from .utils import rdkitutils

pkg_dir = pathlib.Path(__file__).parents[0]
params_dir = pkg_dir / "data" / "params"


# ---------------------------------------------------------------------------
# Module-level functions (the actual implementations)
# ---------------------------------------------------------------------------

def type_everything(
    molsetup,
    atom_params,
    charge_model,
    offatom_params=None,
    dihedral_params=None,
):
    type_atoms(molsetup, atom_params)

    # offatoms must be typed after charges, because offsites pull charge
    if offatom_params is not None:
        cached_offatoms = cache_offatoms(molsetup, offatom_params)
        coords = {atom.index: atom.coord for atom in molsetup.atoms if not atom.is_dummy}
        set_offatoms(molsetup, cached_offatoms, coords)

    if dihedral_params not in (None, "espaloma"):
        type_dihedrals(molsetup, dihedral_params)


def type_atoms(molsetup, atom_params):
    # ensure every "atompar" is defined in a single "smartsgroup"
    ensure = {}
    for smartsgroup in atom_params:
        if smartsgroup == "comment":
            continue
        for line in atom_params[smartsgroup]:
            smarts = str(line["smarts"])
            idxs = [0]
            if "IDX" in line:
                idxs = [i for i in line["IDX"]]
            hits = molsetup.find_pattern(smarts)
            for atompar in line:
                if atompar in ["smarts", "comment", "IDX"]:
                    continue
                if atompar not in molsetup.atom_params:
                    molsetup.atom_params[atompar] = [None] * len(molsetup.atoms)
                value = line[atompar]
                ensure.setdefault(atompar, [])
                ensure[atompar].append(smartsgroup)
                for hit in hits:
                    for idx in idxs:
                        if atompar == "atype":
                            molsetup.set_atom_type(hit[idx], value)
                        molsetup.atom_params[atompar][hit[idx]] = value

    for atompar in ensure:
        if len(set(ensure[atompar])) > 1:
            msg = "%s is modified in multiple smartsgroups: %s" % (
                atompar,
                set(ensure[atompar]),
            )
            warnings.warn(msg)


def cache_offatoms(molsetup, offatom_params):
    """precalculate off-site atoms"""
    cached_offatoms = {}
    n_offatoms = 0
    atoms_with_offchrg = set()
    for smartsgroup in offatom_params:
        if smartsgroup == "comment":
            continue
        tmp = {}
        for line in offatom_params[smartsgroup]:
            smarts = str(line["smarts"])
            hits = molsetup.find_pattern(smarts, uniquify=True)
            smarts_idxs = [0]
            if "IDX" in line:
                smarts_idxs = [i for i in line["IDX"]]
            for smarts_idx in smarts_idxs:
                for hit in hits:
                    parent_idx = hit[smarts_idx]
                    tmp.setdefault(parent_idx, [])
                    for offatom in line["OFFATOMS"]:
                        tmp[parent_idx].append(
                            {
                                "offatom": {
                                    "distance": 1.0,
                                    "x90": False,
                                    "phi": 0.0,
                                    "theta": 0.0,
                                    "z": [],
                                    "x": [],
                                },
                                "atom_params": {},
                            }
                        )
                        for key in offatom:
                            if key in ["distance", "x90"]:
                                tmp[parent_idx][-1]["offatom"][key] = offatom[key]
                            elif key in ["z", "x"]:
                                for i in offatom[key]:
                                    idx = hit[i]
                                    tmp[parent_idx][-1]["offatom"][key].append(idx)
                            elif key in ["theta", "phi"]:
                                tmp[parent_idx][-1]["offatom"][key] = np.radians(
                                    offatom[key]
                                )
                            elif key in ["comment"]:
                                pass
                            elif key == "atype":
                                tmp[parent_idx][-1]["atom_params"][key] = offatom[key]
                            elif key == "pull_charge_fraction":
                                if parent_idx in atoms_with_offchrg:
                                    raise RuntimeError(
                                        "atom %d has charge pulled more than once"
                                        % parent_idx
                                    )
                                atoms_with_offchrg.add(parent_idx)
                                tmp[parent_idx][-1]["atom_params"][key] = offatom[key]
                            else:
                                pass
        for parent_idx in tmp:
            for offatom_dict in tmp[parent_idx]:
                atom_params = offatom_dict["atom_params"]
                offatom = offatom_dict["offatom"]
                atomgeom = AtomicGeometry(
                    parent_idx,
                    neigh=offatom["z"],
                    xneigh=offatom["x"],
                    x90=offatom["x90"],
                )
                if "pull_charge_fraction" in atom_params:
                    pull_charge_fraction = atom_params["pull_charge_fraction"]
                else:
                    pull_charge_fraction = 0.0
                args = (
                    atom_params["atype"],
                    offatom["distance"],
                    offatom["theta"],
                    offatom["phi"],
                    pull_charge_fraction,
                )
                cached_offatoms[n_offatoms] = (atomgeom, args)
                n_offatoms += 1
    return cached_offatoms


def set_offatoms(molsetup, cached_offatoms, coords):
    """add cached offatoms"""
    for k, (atomgeom, args) in cached_offatoms.items():
        (atom_type, dist, theta, phi, pull_charge_fraction) = args
        offatom_coords = atomgeom.calc_point(dist, theta, phi, coords)
        tmp = molsetup.get_pdbinfo(atomgeom.parent + 1)
        pdbinfo = pdbutils.PDBAtomInfo(
            "G", tmp.resName, tmp.resNum, tmp.icode, tmp.chain
        )
        q_parent = (1 - pull_charge_fraction) * molsetup.get_charge(atomgeom.parent)
        q_offsite = pull_charge_fraction * molsetup.get_charge(atomgeom.parent)
        pseudo_atom = {
            "coord": offatom_coords,
            "anchor_list": [atomgeom.parent],
            "charge": q_offsite,
            "pdbinfo": pdbinfo,
            "atom_type": atom_type,
            "rotatable": False,
        }
        molsetup.atoms[atomgeom.parent].charge = q_parent
        molsetup.add_pseudoatom(**pseudo_atom)


def type_dihedrals(molsetup, dihedral_params):
    canon = lambda x: x if x[2] > x[1] else (x[3], x[2], x[1], x[0])

    for line in dihedral_params:
        smarts = str(line["smarts"])
        hits = molsetup.find_pattern(smarts)
        if len(hits) == 0:
            continue
        idxs = [i for i in line["IDX"]]
        tid = line["id"] if "id" in line else None
        fourier_series = []
        term_indices = {}
        for key in line:
            for keyword in ["phase", "k", "periodicity", "idivf"]:
                if key.startswith(keyword):
                    t = int(key.replace(keyword, ""))
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
            atom_idxs = canon(atom_idxs)
            molsetup.dihedral_partaking_atoms[atom_idxs] = dihedral_index
            molsetup.dihedral_labels[atom_idxs] = tid


# ---------------------------------------------------------------------------
# Thin shim: preserves external `AtomTyper.x(...)` callers
# ---------------------------------------------------------------------------

class AtomTyper:
    """Backward-compat shim. Prefer the module-level functions for new code."""

    type_everything = staticmethod(type_everything)
    _type_atoms = staticmethod(type_atoms)
    _cache_offatoms = staticmethod(cache_offatoms)
    _set_offatoms = staticmethod(set_offatoms)
    _type_dihedrals = staticmethod(type_dihedrals)


# ---------------------------------------------------------------------------
# AtomicGeometry: a real stateful class (kept as-is)
# ---------------------------------------------------------------------------

class AtomicGeometry:
    """generate reference frames and add extra sites"""

    def __init__(self, parent, neigh, xneigh=[], x90=False, planar_tol=0.1):
        self.planar_tol = planar_tol

        if type(parent) != int:
            raise RuntimeError("parent must be int")
        self.parent = parent

        self.neigh = []
        for i in neigh:
            if type(i) != int:
                raise RuntimeError("neigh indices must be int")
            self.neigh.append(i)

        self.xneigh = []
        for i in xneigh:
            if type(i) != int:
                raise RuntimeError("xneigh indices must be int")
            self.xneigh.append(i)

        self.calc_x = len(self.xneigh) > 0
        self.x90 = x90

    def calc_point(self, distance, theta, phi, coords):
        """return coordinates of point specified in spherical coordinates"""
        z = self._calc_z(coords)

        if phi == 0.0:
            return z * distance + np.array(coords[self.parent])
        elif self.calc_x == False:
            raise RuntimeError("phi must be zero if X undefined")
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
        """maximize distance from neigh"""
        z = np.zeros(3)
        cumsum = np.zeros(3)
        for i in self.neigh:
            v = np.array(coords[self.parent]) - np.array(coords[i])
            cumsum += v
            z += self.normalized(v)
        z = self.normalized(z)
        if np.sum(cumsum**2) < self.planar_tol**2:
            raise RuntimeError("Refusing to place Z axis on planar atom")
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
        len_ax = (ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) ** 0.5
        if len_ax == 0.0:
            u, v, w = (1, 0, 0)
            rad = 0.0
        else:
            u, v, w = [i / len_ax for i in ax]
        x, y, z = pt
        ux, uy, uz = u * x, u * y, u * z
        vx, vy, vz = v * x, v * y, v * z
        wx, wy, wz = w * x, w * y, w * z
        sa = np.sin(rad)
        ca = np.cos(rad)
        p0 = (
            u * (ux + vy + wz)
            + (x * (v * v + w * w) - u * (vy + wz)) * ca
            + (-wy + vz) * sa
        )
        p1 = (
            v * (ux + vy + wz)
            + (y * (u * u + w * w) - v * (ux + wz)) * ca
            + (wx - uz) * sa
        )
        p2 = (
            w * (ux + vy + wz)
            + (z * (u * u + v * v) - w * (ux + vy)) * ca
            + (-vx + uy) * sa
        )
        return (p0, p1, p2)

    def normalized(self, vec):
        l = sum([x**2 for x in vec]) ** 0.5
        if type(vec) == list:
            return [x / l for x in vec]
        else:
            return vec / l


def add_crippen_to_molsetup(molsetup):
    atom_contribs = rdMolDescriptors._CalcCrippenContribs(molsetup.mol)
    crippen = [atom[0] for atom in atom_contribs]
    nr_pseudo_atoms = len(molsetup.atoms) - molsetup.mol.GetNumAtoms()
    crippen += [0.0] * nr_pseudo_atoms
    molsetup.atom_params["crippen"] = crippen
    return None


def set_ad4sol_par_including_q(molsetup, qasp):
    # does not set ad4sol volume
    par_fn = params_dir / "ad4_desolv_param.json"
    with open(par_fn) as f:
        dsolv_params = json.load(f)
    type_atoms(molsetup, dsolv_params)
    charges = rdkitutils.compute_gasteiger_charges(molsetup.mol)
    nonpolar_h = Chem.MolFromSmarts("[#1][!#7;!#8;!#9;!#16]")
    for h_idx, parent_idx in molsetup.mol.GetSubstructMatches(nonpolar_h):
        charges[parent_idx] += charges[h_idx]
        charges[h_idx] = 0.0
    for index, charge in enumerate(charges):
        molsetup.atom_params["ad4_sol_par"][index] += qasp * abs(charge)
    return None
