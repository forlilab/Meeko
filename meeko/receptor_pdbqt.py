#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko
#

from collections import defaultdict
import json
from os import linesep as os_linesep
import pathlib
import sys

import numpy as np
from scipy import spatial

from .utils.covalent_radius_table import covalent_radius
from .utils.autodock4_atom_types_elements import autodock4_atom_types_elements
from .reactive import get_reactive_atype

pkg_dir = pathlib.Path(__file__).parents[0]
with open(pkg_dir / "data" / "residue_params.json") as f:
    residue_params = json.load(f)
with open(pkg_dir / "data" / "flexres_templates.json") as f:
    flexres_templates = json.load(f)
# the above is controversial, see
# https://stackoverflow.com/questions/6028000/how-to-read-a-static-file-from-inside-a-python-package

def _write_pdbqt_line(atomidx, x, y, z, charge, atom_name, res_name, res_num, atom_type, chain,
                      alt_id=" ", in_code="", occupancy=1.0, temp_factor=0.0, record_type="ATOM"):
    if len(atom_name) > 4:
        raise ValueError("max length of atom_name is 4 but atom name is %s" % atom_name)
    atom_name = "%-3s" % atom_name
    line = "{:6s}{:5d} {:4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"
    line += os_linesep
    return line.format(record_type, atomidx, atom_name, alt_id, res_name, chain,
                   res_num, in_code, x, y, z,
                   occupancy, temp_factor, charge, atom_type)


atom_property_definitions = {'H': 'vdw', 'C': 'vdw', 'A': 'vdw', 'N': 'vdw', 'P': 'vdw', 'S': 'vdw',
                             'Br': 'vdw', 'I': 'vdw', 'F': 'vdw', 'Cl': 'vdw',
                             'NA': 'hb_acc', 'OA': 'hb_acc', 'SA': 'hb_acc', 'OS': 'hb_acc', 'NS': 'hb_acc',
                             'HD': 'hb_don', 'HS': 'hb_don',
                             'Mg': 'metal', 'Ca': 'metal', 'Fe': 'metal', 'Zn': 'metal', 'Mn': 'metal',
                             'MG': 'metal', 'CA': 'metal', 'FE': 'metal', 'ZN': 'metal', 'MN': 'metal',
                             'W': 'water',
                             'G0': 'glue', 'G1': 'glue', 'G2': 'glue', 'G3': 'glue',
                             'CG0': 'glue', 'CG1': 'glue', 'CG2': 'glue', 'CG3': 'glue'}


def _read_receptor_pdbqt_string(pdbqt_string, skip_typing=False):
    atoms = []
    atoms_dtype = [('idx', 'i4'), ('serial', 'i4'), ('name', 'U4'), ('resid', 'i4'),
                   ('resname', 'U3'), ('chain', 'U1'), ("xyz", "f4", (3)),
                   ('partial_charges', 'f4'), ('atom_type', 'U2'),
                   ('alt_id', 'U1'), ('in_code', 'U1'),
                   ('occupancy', 'f4'), ('temp_factor', 'f4'), ('record_type', 'U6')
                  ]
    atom_annotations = {'hb_acc': [], 'hb_don': [],
                        'all': [], 'vdw': [],
                        'metal': []}
    # TZ is a pseudo atom for AutoDock4Zn FF
    pseudo_atom_types = ['TZ']

    idx = 0
    for line in pdbqt_string.split(os_linesep):
        if line.startswith('ATOM') or line.startswith("HETATM"):
            serial = int(line[6:11].strip())
            name = line[12:16].strip()
            resname = line[17:20].strip()
            chainid = line[21].strip()
            resid = int(line[22:26].strip())
            xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()], dtype=np.float32)
            try:
                partial_charges = float(line[71:77].strip())
            except:
                partial_charges = None # probably reading a PDB, not PDBQT
            atom_type = line[77:79].strip()
            alt_id = line[16:17].strip()
            in_code = line[26:27].strip()
            try:
                occupancy = float(line[54:60])
            except:
                occupancy = None
            try:
                temp_factor = float(line[60:68])
            except:
                temp_factor = None
            record_type = line[0:6].strip()

            if skip_typing:
                atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type,
                              alt_id, in_code, occupancy, temp_factor, record_type))
                continue
            if not atom_type in pseudo_atom_types:
                atom_annotations['all'].append(idx)
                atom_annotations[atom_property_definitions[atom_type]].append(idx)
                atoms.append((idx, serial, name, resid, resname, chainid, xyz, partial_charges, atom_type,
                              alt_id, in_code, occupancy, temp_factor, record_type))

            idx += 1

    atoms = np.array(atoms, dtype=atoms_dtype)

    return atoms, atom_annotations


def _identify_bonds(atom_idx, positions, atom_types):
    bonds = defaultdict(list)
    KDTree = spatial.cKDTree(positions)
    bond_allowance_factor = 1.1
    # If we ask more than the number of coordinates/element
    # in the BHTree, we will end up with some inf values
    k = 5 if len(atom_idx) > 5 else len(atom_idx)
    atom_idx = np.array(atom_idx)

    for atom_i, position, atom_type in zip(atom_idx, positions, atom_types):
        distances, indices = KDTree.query(position, k=k)
        r_cov = covalent_radius[autodock4_atom_types_elements[atom_type]]

        optimal_distances = [bond_allowance_factor * (r_cov + covalent_radius[autodock4_atom_types_elements[atom_types[i]]]) for i in indices[1:]]
        bonds[atom_i] = atom_idx[indices[1:][np.where(distances[1:] < optimal_distances)]].tolist()

    return bonds


class PDBQTReceptor:

    flexres_templates = flexres_templates
    skip_types=("H",)

    def __init__(self, pdbqt_filename, skip_typing=False):
        self._pdbqt_filename = pdbqt_filename
        self._atoms = None
        self._atom_annotations = None
        self._KDTree = None

        with open(pdbqt_filename) as f:
            pdbqt_string = f.read()

        self._atoms, self._atom_annotations = _read_receptor_pdbqt_string(pdbqt_string, skip_typing)
        # We add to the KDTree only the rigid part of the receptor
        self._KDTree = spatial.cKDTree(self._atoms['xyz'])
        self._bonds = _identify_bonds(self._atom_annotations['all'], self._atoms['xyz'], self._atoms['atom_type'])
        self.atom_idxs_by_res = self.get_atom_indices_by_residue(self._atoms)

    def __repr__(self):
        return ('<Receptor from PDBQT file %s containing %d atoms>' % (self._pdbqt_filename, self._atoms.shape[0]))

    @staticmethod
    def get_atom_indices_by_residue(atoms):
        """ return a dictionary where residues are keys and
             values are lists of atom indices

            >>> atom_idx_by_res = {("A", "LYS", 417): [0, 1, 2, 3, ..., 8]}
        """

        atom_idx_by_res = {}
        for atom_index, atom in enumerate(atoms):
            res_id = (atom["chain"], atom["resname"], atom["resid"])
            atom_idx_by_res.setdefault(res_id, [])
            atom_idx_by_res[res_id].append(atom_index)
        return atom_idx_by_res

    @staticmethod
    def get_params_for_residue(resname, atom_names, residue_params=residue_params):
        excluded_params = ("atom_names", "bond_cut_atoms", "bonds")
        atom_params = {}
        atom_counter = 0
        err = ""
        ok = True
        is_matched = False
        for terminus in ["", "N", "C"]: # e.g. "CTYR" for C-term TYR, hard-coded in residue_params
            r_id = "%s%s" % (terminus, resname)
            if r_id not in residue_params:
                err = "residue %s not in residue_params" % r_id + os_linesep
                ok = False
                return atom_params, ok, err
            ref_names = set(residue_params[r_id]["atom_names"])
            query_names = set(atom_names)
            if ref_names == query_names:
                is_matched = True
                break

        if not is_matched:
            ok = False
            err = "residue %s did not match residue_params" % r_id + os_linesep
            err += "ref_names: %s" % ref_names + os_linesep 
            err += "query_names: %s" % query_names + os_linesep 
            return atom_params, ok, err

        for atom_name in atom_names:
            name_index = residue_params[r_id]["atom_names"].index(atom_name)
            for param in residue_params[r_id].keys():
                if param in excluded_params:
                    continue
                if param not in atom_params:
                    atom_params[param] = [None] * atom_counter
                value = residue_params[r_id][param][name_index]
                atom_params[param].append(value)
            atom_counter += 1

        return atom_params, ok, err

    def assign_types_charges(self, residue_params=residue_params):
        wanted_params = ("atom_types", "gasteiger")
        atom_params = {key: [] for key in wanted_params}
        ok = True
        err = ""
        for r_id, atom_indices in self.atom_idxs_by_res.items():
            atom_names = tuple(self.atoms(atom_indices)["name"])
            resname = r_id[1]
            params_this_res, ok_, err_ = self.get_params_for_residue(resname, atom_names, residue_params)
            ok &= ok_
            err += err_
            if not ok_:
                print("did not match %s with template" % str(r_id), file=sys.stderr)
                continue
            for key in wanted_params:
                atom_params[key].extend(params_this_res[key])
        if ok:
            self._atoms["partial_charges"] = atom_params["gasteiger"]
            self._atoms["atom_type"] = atom_params["atom_types"]
        return ok, err

    def write_flexres_from_template(self, res_id, atom_index=0):
        success = True
        error_msg = ""
        branch_offset = atom_index # templates assume first atom is 1
        output = {"pdbqt": "", "flex_indices": [], "atom_index": atom_index}
        resname = res_id[1]
        if resname not in self.flexres_templates:
            success = False
            error_msg = "no flexible residue template for resname %s, sorry" % resname
            return output, success, error_msg
        if res_id not in self.atom_idxs_by_res:
            success = False
            chains = set(self._atoms["chain"])
            error_msg += "could not find residue with chain='%s', resname=%s, resnum=%d" % res_id + os_linesep
            error_msg += "chains in this receptor: %s" % ", ".join("'%s'" % c for c in chains) + os_linesep
            if " " in chains: # should not happen because we use strip() when parsing the chain
                error_msg += "use ' ' (a space character) for empty chain" + os_linesep
            if "" in chains:
                error_msg += "use '' (empty string) for empty chain" + os_linesep
            return output, success, error_msg

        # collect lines of res_id
        atoms_by_name = {}
        for i in self.atom_idxs_by_res[res_id]:
            name = self._atoms[i]["name"]
            if name in ['C', 'N', 'O', 'H', 'H1', 'H2', 'H3', 'OXT']: # skip backbone atoms
                continue
            atype = self._atoms[i]["atom_type"]
            if atype in self.skip_types:
                continue
            output["flex_indices"].append(i)
            atoms_by_name[name] = self.atoms(i)

        # check it was a full match
        template = self.flexres_templates[resname]
        got_atoms = set(atoms_by_name)
        ref_atoms = set()
        for i in range(len(template["is_atom"])):
            if template["is_atom"][i]:
                ref_atoms.add(template["atom_name"][i])
        if got_atoms != ref_atoms:
            success = False
            error_msg += "mismatch in atom names for residue %s" % str(res_id) + os_linesep
            error_msg += "names found but not in template: %s" % str(got_atoms.difference(ref_atoms)) + os_linesep
            error_msg += "missing names: %s" % str(ref_atoms.difference(got_atoms)) + os_linesep
            return output, success, error_msg

        # create output string
        n_lines = len(template['is_atom'])
        for i in range(n_lines):
            if template['is_atom'][i]:
                atom_index += 1
                name = template['atom_name'][i]
                atom = atoms_by_name[name]
                if atom["atom_type"] not in self.skip_types:
                    atom["serial"] = atom_index
                    output["pdbqt"] += self.write_pdbqt_line(atom)
            else:
                line = template['original_line'][i]
                if branch_offset > 0 and (line.startswith("BRANCH") or line.startswith("ENDBRANCH")):
                    keyword, i, j = line.split()
                    i = int(i) + branch_offset
                    j = int(j) + branch_offset
                    line = "%s %3d %3d" % (keyword, i, j)
                output["pdbqt"] += line + os_linesep # e.g. BRANCH keywords

        output["atom_index"] = atom_index
        return output, success, error_msg

    @staticmethod
    def write_pdbqt_line(atom):
        return _write_pdbqt_line(atom["serial"], atom["xyz"][0], atom["xyz"][1], atom["xyz"][2],
                                 atom["partial_charges"], atom["name"], atom["resname"],
                                 atom["resid"], atom["atom_type"], atom["chain"],
                                 atom["alt_id"], atom["in_code"], atom["occupancy"],
                                 atom["temp_factor"], atom["record_type"])


    def write_pdbqt_string(self, flexres=()):
        ok = True
        err = ""
        pdbqt = {"rigid": "",
                 "flex":  {},
                 "flex_indices": []}
        atom_index = 0
        for res_id in set(flexres):
            output, ok_, err_ = self.write_flexres_from_template(res_id, atom_index)
            atom_index = output["atom_index"] # next residue starts here
            ok &= ok_
            err += err_
            pdbqt["flex_indices"].extend(output["flex_indices"])
            pdbqt["flex"][res_id] = ""
            pdbqt["flex"][res_id] += "BEGIN_RES %3s %1s%4d" % (res_id) + os_linesep
            pdbqt["flex"][res_id] += output["pdbqt"]
            pdbqt["flex"][res_id] += "END_RES %3s %1s%4d" % (res_id) + os_linesep

        # use non-flex lines for rigid part
        for i, atom in enumerate(self._atoms):
            if i not in pdbqt["flex_indices"] and atom["atom_type"] not in self.skip_types:
                pdbqt["rigid"] += self.write_pdbqt_line(atom)

        return pdbqt, ok, err

    @staticmethod
    def make_flexres_reactive(pdbqtstr, reactive_name, resname, prefix_atype="", residue_params=residue_params):
        atom_names = residue_params[resname]["atom_names"]
        bonds = residue_params[resname]["bonds"]
        def get_neigh(idx, bonds):
            neigh = set()
            for (i, j) in bonds:
                if i == idx:
                    neigh.add(j)
                elif j == idx:
                    neigh.add(i)
            return neigh
        react_idx = atom_names.index(reactive_name)
        one_bond_away = get_neigh(react_idx, bonds)
        two_bond_away = set()
        for i in one_bond_away:
            for j in get_neigh(i, bonds):
                if (j != react_idx) and (j not in one_bond_away):
                    two_bond_away.add(j)
        names_1bond = [atom_names[i] for i in one_bond_away]
        names_2bond = [atom_names[i] for i in two_bond_away]
        new_pdbqt_str = ""
        for i, line in enumerate(pdbqtstr.split(os_linesep)[:-1]):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                name = line[12:16].strip()
                print(name, "<<<") # TODO need proper atom names from chorizo
                atype = line[77:].strip()
                if name == reactive_name:
                    new_type = prefix_atype + get_reactive_atype(atype, 1)
                elif name in names_1bond:
                    new_type = prefix_atype + get_reactive_atype(atype, 2)
                elif name in names_2bond:
                    new_type = prefix_atype + get_reactive_atype(atype, 3)
                else:
                    new_type = atype
                new_pdbqt_str += line[:77] + new_type + os_linesep
            else:
                new_pdbqt_str += line + os_linesep
        return new_pdbqt_str


    def atoms(self, atom_idx=None):
        """Return the atom i

        Args:
            atom_idx (int, list): index of one or multiple atoms

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        if atom_idx is not None and self._atoms.size > 1:
            if not isinstance(atom_idx, (list, tuple, np.ndarray)):
                atom_idx = np.array(atom_idx, dtype=int)
            atoms = self._atoms[atom_idx]
        else:
            atoms = self._atoms

        return atoms.copy()

    def positions(self, atom_idx=None):
        """Return coordinates (xyz) of all atoms or a certain atom

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            ndarray: 2d ndarray of coordinates (xyz)

        """
        return np.atleast_2d(self.atoms(atom_idx)['xyz'])

    def closest_atoms_from_positions(self, xyz, radius, atom_properties=None, ignore=None):
        """Retrieve indices of the closest atoms around a positions/coordinates
        at a certain radius.

        Args:
            xyz (np.ndarray): array of 3D coordinates
            raidus (float): radius
            atom_properties (str): property of the atoms to retrieve
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)
            ignore (int or list): ignore atom for the search using atom id (0-based)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        index = self._KDTree.query_ball_point(xyz, radius, p=2, return_sorted=True)

        # When nothing was found around...
        if not index:
            return np.array([])

        # Handle the case when positions for of only one atom was passed in the input
        try:
            index = {i for j in index for i in j}
        except:
            index = set(index)

        if atom_properties is not None:
            if not isinstance(atom_properties, (list, tuple)):
                atom_properties = [atom_properties]

            try:
                for atom_property in atom_properties:
                    index.intersection_update(self._atom_annotations[atom_property])
            except:
                error_msg = 'Atom property %s is not valid. Valid atom properties are: %s'
                raise KeyError(error_msg % (atom_property, self._atom_annotations.keys()))

        if ignore is not None:
            if not isinstance(ignore, (list, tuple, np.ndarray)):
                ignore = [ignore]
            index = index.difference([i for i in ignore])

        index = list(index)
        atoms = self._atoms[index].copy()

        return atoms

    def closest_atoms(self, atom_idx, radius, atom_properties=None):
        """Retrieve indices of the closest atoms around a positions/coordinates
        at a certain radius.

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)
            raidus (float): radius
            atom_properties (str or list): property of the atoms to retrieve
                (properties: ligand, flexible_residue, vdw, hb_don, hb_acc, metal, water, reactive, glue)

        Returns:
            ndarray: 2d ndarray (atom_id, atom_name, resname, resid, chainid, xyz, q, t)

        """
        return self.closest_atoms_from_positions(self._atoms[atom_idx]['xyz'], radius, atom_properties, atom_idx)

    def neighbor_atoms(self, atom_idx):
        """Return neighbor (bonded) atoms

        Args:
            atom_idx (int, list): index of one or multiple atoms (0-based)

        Returns:
            list_of_list: list of lists containing the neighbor (bonded) atoms (0-based)

        """
        if not isinstance(atom_idx, (list, tuple, np.ndarray)):
            atom_idx = [atom_idx]

        return [self._bonds[i] for i in atom_idx]
