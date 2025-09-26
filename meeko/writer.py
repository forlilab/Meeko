#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko PDBQT writer
#

import logging
import sys
import json
import math
eol="\n"

import numpy as np
from rdkit import Chem
from .utils import pdbutils
from .utils.rdkitutils import mini_periodic_table

from .molsetup import Bond


logger = logging.getLogger(__name__)

def oids_json_from_setup(molsetup, name="LigandFromMeeko"):
    if len(molsetup.restraints):
        raise NotImplementedError(
            "molsetup has restraints but these aren't written to oids block yet"
        )
    offchrg_type = "OFFCHRG"
    offchrg_by_parent = {}
    for atom in molsetup.atoms:
        if atom.is_pseudo_atom and atom.atom_type == offchrg_type:
            neigh = molsetup.get_neighbors(atom.index)
            if len(neigh) != 1:
                raise RuntimeError(
                    "offsite charge %s is bonded to: %s which has len() != 1"
                    % (atom.index, json.dumps(neigh))
                )
            if neigh[0] in offchrg_by_parent:
                raise RuntimeError(
                    "atom %d has more than one offsite charge" % neigh[0]
                )
            offchrg_by_parent[neigh[0]] = atom.index
    output_indices_start_at_one = True
    index_start = int(output_indices_start_at_one)
    positions_block = ""
    charges = []
    offchrg_by_oid_parent = {}
    elements = []
    n_real_atoms = molsetup.true_atom_count
    n_fake_atoms = molsetup.pseudoatom_count
    indexmap = {}  # molsetup: oid
    count_oids = 0
    for atom in molsetup.atoms:
        index = atom.index
        if atom.is_dummy or atom.is_pseudo_atom or atom.is_ignore:
            continue
        if atom.atom_type == offchrg_type:
            continue # handled by offchrg_by_parent
        oid_id = count_oids + index_start
        indexmap[index] = count_oids
        x, y, z = atom.coord
        positions_block += "position.%d = (%f,%f,%f)\n" % (oid_id, x, y, z)
        charges.append(atom.charge)
        if index in offchrg_by_parent:
            index_pseudo = offchrg_by_parent[index]
            xq_abs, yq_abs, zq_abs = molsetup.atoms[index_pseudo].coord
            xq_rel = xq_abs - x
            yq_rel = yq_abs - y
            zq_rel = zq_abs - z
            offchrg_by_oid_parent[count_oids] = {
                "q": molsetup.atoms[index_pseudo].charge,
                "xyz": (xq_rel, yq_rel, zq_rel),
            }
        count_oids += 1
        element = "%s %s %d" % (name, atom.atom_type, oid_id)
        elements.append(element)

    tmp = []
    for index in range(len(charges)):
        if index in offchrg_by_oid_parent:
            tmplist = [
                "%f" % charges[index],
                "0.0",
                "0.0",
                "0.0",
            ]  # xyz relative to current elemtn
            tmplist.append("%f" % offchrg_by_oid_parent[index]["q"])
            tmplist.append("%f,%f,%f" % offchrg_by_oid_parent[index]["xyz"])
            tmp.append(",".join(tmplist))
        else:
            tmp.append("%f" % charges[index])
    charges_line = "import_charges = {%s}\n" % ("|".join(tmp))
    elements_line = "elements = %s\n" % (",".join(elements))

    bonds = [[] for _ in range(count_oids)]
    # bond_orders = [[] for _ in range(count_oids)]
    static_links = []
    for i, j in molsetup.bond_info.keys():
        if molsetup.get_is_ignore(i) or molsetup.get_is_ignore(j):
            continue
        if (
            molsetup.get_atom_type(i) == offchrg_type
            or molsetup.get_atom_type(j) == offchrg_type
        ):
            continue
        oid_i = indexmap[i]
        oid_j = indexmap[j]
        bonds[oid_i].append("%d" % (oid_j + index_start))
        # bond_orders[oid_i].append("%d" % molsetup.bond_info[(i, j)].order)
        if not molsetup.bond_info[(i, j)].rotatable:
            static_links.append("%d,%d" % (oid_i + index_start, oid_j + index_start))
    bonds = [",".join(j_list) for j_list in bonds]
    bonds_line = "connectivity = {%s}\n" % ("|".join(bonds))
    # bond_orders = [",".join(orders) for orders in bond_orders]
    # bondorder_line = "bond_order = {%s}\n" % ("|".join(bond_orders))
    staticlinks_line = "static_links = {%s}\n" % ("|".join(static_links))

    output = ""
    output += "[Group: %s]\n" % name
    output += positions_block
    output += charges_line
    output += elements_line
    output += bonds_line
    #output += bondorder_line
    output += staticlinks_line
    output += "number = 1\t\t// can only be 1 for the sandbox currently (but any number for classical MC)\n"
    output += "group_dipole = 1\t// not relevant for sandbox but classical MC\n"
    output += "rand_independent=0\t// not relevant for sandbox but classical MC\n"
    output += "bond_range = 4\t\t// bond range AD default\n"
    output += "\n"
    output += get_dihedrals_block(molsetup, indexmap, name)

    return output, indexmap


def oids_block_from_setup(molsetup, name="LigandFromMeeko"):
    if len(molsetup.restraints):
        raise NotImplementedError(
            "molsetup has restraints but these aren't written to oids block yet"
        )
    offchrg_type = "OFFCHRG"
    offchrg_by_parent = {}
    for atom in molsetup.atoms:
        if atom.is_pseudo_atom and atom.atom_type == offchrg_type:
            neigh = molsetup.get_neighbors(atom.index)
            if len(neigh) != 1:
                raise RuntimeError(
                    "offsite charge %s is bonded to: %s which has len() != 1"
                    % (atom.index, json.dumps(neigh))
                )
            if neigh[0] in offchrg_by_parent:
                raise RuntimeError(
                    "atom %d has more than one offsite charge" % neigh[0]
                )
            offchrg_by_parent[neigh[0]] = atom.index
    output_indices_start_at_one = True
    index_start = int(output_indices_start_at_one)
    positions_block = ""
    charges = []
    offchrg_by_oid_parent = {}
    elements = []
    n_real_atoms = molsetup.true_atom_count
    n_fake_atoms = molsetup.pseudoatom_count
    indexmap = {}  # molsetup: oid
    count_oids = 0
    for atom in molsetup.atoms:
        index = atom.index
        if atom.is_dummy or atom.is_pseudo_atom or atom.is_ignore:
            continue
        if atom.atom_type == offchrg_type:
            continue  # handled by offchrg_by_parent
        oid_id = count_oids + index_start
        indexmap[index] = count_oids
        x, y, z = atom.coord
        positions_block += "position.%d = (%f,%f,%f)\n" % (oid_id, x, y, z)
        charges.append(atom.charge)
        if index in offchrg_by_parent:
            index_pseudo = offchrg_by_parent[index]
            xq_abs, yq_abs, zq_abs = molsetup.atoms[index_pseudo].coord
            xq_rel = xq_abs - x
            yq_rel = yq_abs - y
            zq_rel = zq_abs - z
            offchrg_by_oid_parent[count_oids] = {
                "q": molsetup.get_charge(index_pseudo),
                "xyz": (xq_rel, yq_rel, zq_rel),
            }
        count_oids += 1
        element = "%s %s %d" % (name, molsetup.get_atom_type(index), oid_id)
        elements.append(element)

    tmp = []
    for index in range(len(charges)):
        if index in offchrg_by_oid_parent:
            tmplist = [
                "%f" % charges[index],
                "0.0",
                "0.0",
                "0.0",
            ]  # xyz relative to current elemtn
            tmplist.append("%f" % offchrg_by_oid_parent[index]["q"])
            tmplist.append("%f,%f,%f" % offchrg_by_oid_parent[index]["xyz"])
            tmp.append(",".join(tmplist))
        else:
            tmp.append("%f" % charges[index])
    charges_line = "import_charges = {%s}\n" % ("|".join(tmp))
    elements_line = "elements = %s\n" % (",".join(elements))

    bonds = [[] for _ in range(count_oids)]
    # bond_orders = [[] for _ in range(count_oids)]
    static_links = []
    for i, j in molsetup.bond_info.keys():
        if molsetup.get_is_ignore(i) or molsetup.get_is_ignore(j):
            continue
        if (
            molsetup.get_atom_type(i) == offchrg_type
            or molsetup.get_atom_type(j) == offchrg_type
        ):
            continue
        oid_i = indexmap[i]
        oid_j = indexmap[j]
        bonds[oid_i].append("%d" % (oid_j + index_start))
        # bond_orders[oid_i].append("%d" % molsetup.bond_info[(i, j)].order)
        if not molsetup.bond_info[(i, j)].rotatable:
            static_links.append("%d,%d" % (oid_i + index_start, oid_j + index_start))
    bonds = [",".join(j_list) for j_list in bonds]
    bonds_line = "connectivity = {%s}\n" % ("|".join(bonds))
    # bond_orders = [",".join(orders) for orders in bond_orders]
    # bondorder_line = "bond_order = {%s}\n" % ("|".join(bond_orders))
    staticlinks_line = "static_links = {%s}\n" % ("|".join(static_links))

    output = ""
    output += "[Group: %s]\n" % name
    output += positions_block
    output += charges_line
    output += elements_line
    output += bonds_line
    # output += bondorder_line
    output += staticlinks_line
    output += "number = 1\t\t// can only be 1 for the sandbox currently (but any number for classical MC)\n"
    output += "group_dipole = 1\t// not relevant for sandbox but classical MC\n"
    output += "rand_independent=0\t// not relevant for sandbox but classical MC\n"
    output += "bond_range = 4\t\t// bond range AD default\n"
    output += "\n"
    output += get_dihedrals_block(molsetup, indexmap, name)

    return output, indexmap


def get_dihedrals_block(molsetup, indexmap, name):
    # molsetup.dihedral_interactions    is a list of unique fourier_series
    # molsetup.dihedral_partaking_atoms has tuples of atom indices as keys, and the values
    #                                   are the indices in molsetup.dihedral_interactions
    # molsetup.dihedral_labels          also has tuples of atom indices as keys, but the
    #                                   values are not guaranteed to be unique

    # Let's carefully use dihedral_labels to name the interactions
    label_by_index = {}
    atomidx_by_index = {}
    for atomidx in molsetup.dihedral_partaking_atoms:
        a, b, c, d = atomidx
        if (
            molsetup.get_is_ignore(a)
            or molsetup.get_is_ignore(b)
            or molsetup.get_is_ignore(c)
            or molsetup.get_is_ignore(d)
        ):
            continue
        bond_id = Bond.get_bond_id(b, c)
        if not molsetup.bond_info[bond_id].rotatable:
            continue
        index = molsetup.dihedral_partaking_atoms[atomidx]
        atomidx_by_index.setdefault(index, set())
        atomidx_by_index[index].add(atomidx)
        label = (
            molsetup.dihedral_labels[atomidx]
            if atomidx in molsetup.dihedral_labels
            else None
        )
        if label is None:
            label = "from_meeko_%d" % index
        label_by_index.setdefault(index, set())
        label_by_index[index].add(label)
    spent_labels = set()
    for index in label_by_index:
        label = "_".join(label_by_index[index])
        number = 0
        while label in spent_labels:
            number += 1
            label = "_".join(label_by_index[index]) + "_v%d" % number
        label_by_index[index] = label
        spent_labels.add(label)

    text = ""
    for index in label_by_index:
        text += "[Interaction: %s, %s]\n" % (name, label_by_index[index])
        text += "type = dihedral\n"
        atomidx_strings = []
        for atomidx in atomidx_by_index[index]:
            string = ",".join(["%d" % (indexmap[i] + 1) for i in atomidx])
            atomidx_strings.append(string)
        text += "elements = {%s}\n" % ("|".join(atomidx_strings))
        text += "parameters = %s\n" % _aux_fourier_conversion(
            molsetup.dihedral_interactions[index]
        )
        text += "\n"
    return text


def _aux_fourier_conversion(fourier_series):
    # convert from:
    #   k*(1+cos(n*theta-phase))
    # to:
    #   (k/2)*(1+cos(n*(theta+phase)))
    # where n = periodicity
    max_periodicity = max([fs["periodicity"] for fs in fourier_series])
    tmp = [(0, 0)] * max_periodicity
    for fs in fourier_series:
        i = fs["periodicity"] - 1
        k = 2.0 * fs["k"]
        phase = -1 * fs["phase"]
        tmp[i] = (k, phase)
    strings = []
    periodicity = 0
    for k, phase in tmp:
        periodicity += 1
        k_str = "0"
        if phase == 0:
            phase_str = "0"
        else:
            phase_str = ("%f" % (phase / np.pi)).rstrip("0").rstrip(".") + "*pi"
            if phase_str == "1*pi":
                phase_str = "pi"
            if phase_str == "-1*pi":
                phase_str = "-pi"
            if periodicity != 1:
                phase_str += "/%d" % periodicity
        if k != 0:
            k_str = "%f*4.184/60.221" % (k)
        strings.append("%s,%s" % (k_str, phase_str))
    return "(" + ";".join(strings) + ")"


class PDBQTWriterLegacy:

    @staticmethod
    def _get_pdbinfo_fitting_pdb_chars(pdbinfo):
        """return strings and integers that are guaranteed
        to fit within the designated chars of the PDB format"""

        atom_name = pdbinfo.name
        res_name = pdbinfo.resName
        res_num = pdbinfo.resNum
        chain = pdbinfo.chain
        if len(atom_name) > 4:
            atom_name = atom_name[0:4]
        if len(res_name) > 3:
            res_name = res_name[0:3]
        if res_num > 9999:
            res_num = res_num % 10000
        if len(chain) > 1:
            chain = chain[0:1]
        return atom_name, res_name, res_num, chain

    @classmethod
    def _make_pdbqt_line_from_molsetup(cls, setup, atom_idx, count):
        """ """
        pdbinfo = setup.get_pdbinfo(atom_idx)
        if pdbinfo is None:
            pdbinfo = pdbutils.PDBAtomInfo("", "", 0, "")
        atom_name, res_name, res_num, chain = cls._get_pdbinfo_fitting_pdb_chars(
            pdbinfo
        )  # TODO icode
        coord = setup.get_coord(atom_idx)
        atom_type = setup.get_atom_type(atom_idx)
        charge = setup.get_charge(atom_idx)
        pdbqt_line = cls._make_pdbqt_line(
            count, atom_name, res_name, chain, res_num, coord, charge, atom_type
        )
        return pdbqt_line

    @staticmethod
    def _make_pdbqt_line(
        count, atom_name, res_name, chain, res_num, coord, charge, atom_type, icode=""
    ):
        if atom_type is None:
            msg = "Can't write PDBQT because atom_type is None.\n"
            msg += "Probably, MoleculePreparation was instantiated with a configuration that does not assign atom types.\n"
            msg += "For example, mk_prep = MoleculePreparation(load_atom_params='vina_params') assigns atomic radii and\n"
            msg += "other properties, but not atom types. The default for load_atom_params is 'ad4_types'.\n"
            msg += "From command line scripts, this option is usually set in JSON files passed to option --mk_config.\n"
            raise ValueError(msg)
        record_type = "ATOM"
        alt_id = " "
        occupancy = 1.0
        temp_factor = 0.0
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"
        pdbqt_line = atom.format(
            record_type,
            count,
            atom_name,
            alt_id,
            res_name,
            chain,
            res_num,
            icode,
            float(coord[0]),
            float(coord[1]),
            float(coord[2]),
            occupancy,
            temp_factor,
            charge,
            atom_type,
        )
        return pdbqt_line

    @classmethod
    def _walk_graph_recursive(cls, setup, node, data, edge_start=0, first=False):
        """recursive walk of rigid bodies"""

        if first:
            data["pdbqt_buffer"].append("ROOT")
            member_pool = sorted(setup.flexibility_model["rigid_body_members"][node])
        else:
            member_pool = setup.flexibility_model["rigid_body_members"][node][:]
            member_pool.remove(edge_start)
            member_pool = [edge_start] + member_pool

        for member in member_pool:
            if setup.get_is_ignore(member) == 1:
                continue
            pdbqt_line = cls._make_pdbqt_line_from_molsetup(
                setup, member, data["count"]
            )
            data["pdbqt_buffer"].append(pdbqt_line)
            data["numbering"][member] = data["count"]  # count starts at 1
            data["count"] += 1

        if first:
            data["pdbqt_buffer"].append("ENDROOT")

        data["visited"].append(node)

        for neigh in setup.flexibility_model["rigid_body_graph"][node]:
            if neigh in data["visited"]:
                continue

            # Write the branch
            begin, next_index = setup.flexibility_model["rigid_body_connectivity"][
                node, neigh
            ]

            # do not write branch (or anything downstream) if any of the two atoms
            # defining the rotatable bond are ignored
            if setup.get_is_ignore(begin) or setup.get_is_ignore(next_index):
                continue

            begin = data["numbering"][begin]
            end = data["count"]

            data["pdbqt_buffer"].append("BRANCH %3d %3d" % (begin, end))
            data = cls._walk_graph_recursive(setup, neigh, data, edge_start=next_index)
            data["pdbqt_buffer"].append("ENDBRANCH %3d %3d" % (begin, end))

        return data

    @staticmethod
    def _is_molsetup_ok(setup, bad_charge_ok):

        success = True
        error_msg = ""

        if len(setup.restraints):
            error_msg = "molsetup has restraints but these can't be written to PDBQT"
            success = False

        for atom in setup.atoms:
            if atom.is_ignore:
                continue
            if atom.atom_type is None:
                error_msg += "atom number %d has None type, mol name: %s\n" % (
                    atom.index,
                    setup.get_mol_name(),
                )
                success = False
        for atom in setup.atoms:
            if atom.is_ignore:
                continue
            if atom.atom_type is None:
                error_msg += "atom number %d has None type, mol name: %s\n" % (
                    atom.index,
                    setup.get_mol_name(),
                )
                success = False
            c = atom.charge
            if not bad_charge_ok and (
                type(c) != float and type(c) != int or math.isnan(c) or math.isinf(c)
            ):
                error_msg += (
                    "atom number %d has non finite charge, mol name: %s, charge: %s\n"
                    % (atom.index, setup.get_mol_name(), str(c))
                )
                success = False

        return success, error_msg

    @classmethod
    def write_string_from_polymer(cls, polymer):
        rigid_pdbqt_string, flex_pdbqt_dict = cls.write_from_polymer(
            polymer
        )
        flex_pdbqt_string = ""
        for res_id, pdbqt_string in flex_pdbqt_dict.items():
            flex_pdbqt_string += pdbqt_string
        return rigid_pdbqt_string, flex_pdbqt_string

    @classmethod
    def write_from_polymer(cls, polymer):
        rigid_pdbqt_string = ""
        flex_pdbqt_dict = {}
        atom_count = 0
        flex_atom_count = 0
        for res_id, monomer in polymer.get_valid_monomers().items():
            chain, resnum = res_id.split(":")
            if resnum[-1].isalpha():
                icode = resnum[-1]
                resnum = int(resnum[:-1])
            else:
                icode = ""
                resnum = int(resnum)
            molsetup = monomer.molsetup
            resname = monomer.input_resname
            if monomer.is_movable:
                original_ignore = {atom.index: atom.is_ignore for atom in molsetup.atoms}
                graph = molsetup.flexibility_model["rigid_body_graph"]
                root = molsetup.flexibility_model["root"]
                if len(graph[root]) > 1:
                    raise RuntimeError(
                        f"flexible residue {res_id} has {len(graph[root])}"
                        " rotatable bonds from root, but PDBQT is limited to 1"
                    )
                elif len(graph[root]) == 0:
                    logger.warning(
                        f"flexible residue {res_id} has no movable atoms, omitting from flexres PDBQT"
                    )
                else:
                    # set ignore to True for static atoms of flexible sidechains
                    # to exclude them from the PDBQT string
                    for atom_idx, is_flex in enumerate(monomer.is_flexres_atom):
                            molsetup.atoms[atom_idx].is_ignore = not is_flex
                    this_flex_pdbqt, ok, err = PDBQTWriterLegacy.write_string(
                        molsetup, remove_smiles=True, add_index_map=True
                    )
                    for atom in molsetup.atoms:
                        atom.is_ignore = original_ignore[atom.index]
                    if not ok:
                        raise RuntimeError(err)
                    this_flex_pdbqt, flex_atom_count = (
                        cls.adapt_pdbqt_for_autodock4_flexres(
                            this_flex_pdbqt,
                            resname,
                            chain,
                            str(resnum) + icode,
                            skip_rename_ca_cb=True,
                            atom_count=flex_atom_count,
                        )
                    )
                    flex_pdbqt_dict[res_id] = this_flex_pdbqt

            for atom_idx, atom in enumerate(molsetup.atoms):
                if atom.is_ignore or monomer.is_flexres_atom[atom_idx]:
                    continue
                atom_type = atom.atom_type
                coord = atom.coord
                atom_name = atom.pdbinfo.name
                charge = atom.charge
                atom_count += 1
                rigid_pdbqt_string += (
                    cls._make_pdbqt_line(
                        atom_count,
                        atom_name,
                        resname,
                        chain,
                        resnum,
                        coord,
                        charge,
                        atom_type,
                        icode,
                    )
                    + eol
                )
        return rigid_pdbqt_string, flex_pdbqt_dict

    @classmethod
    def write_string(
        cls, setup, add_index_map=False, remove_smiles=False, bad_charge_ok=False
    ):
        """Output a PDBQT file as a string.

        Args:
            setup: RDKitMoleculeSetup

        Returns:
            str:  PDBQT string of the molecule
            bool: success
            str:  error message
        """

        success, error_msg = cls._is_molsetup_ok(setup, bad_charge_ok)
        if not success:
            pdbqt_string = ""
            return pdbqt_string, success, error_msg

        data = {
            "visited": [],
            "numbering": {},
            "pdbqt_buffer": [],
            "count": 1,
        }
        atom_counter = {}

        torsdof = len(setup.flexibility_model["rigid_body_graph"]) - 1

        if "torsions_org" in setup.flexibility_model:
            torsdof_org = setup.flexibility_model["torsions_org"]
            data["pdbqt_buffer"].append(
                "REMARK Flexibility Score: %8.3f" % setup.flexibility_model["score"]
            )
            active_tors = torsdof_org
        else:
            active_tors = torsdof

        data = cls._walk_graph_recursive(
            setup, setup.flexibility_model["root"], data, first=True
        )

        if add_index_map:
            for i, remark_line in enumerate(
                cls.remark_index_map(setup, data["numbering"])
            ):
                # Need to use 'insert' because data["numbering"]
                # is populated in self._walk_graph_recursive.
                data["pdbqt_buffer"].insert(i, remark_line)

        if not remove_smiles:
            smiles, order = setup.get_smiles_and_order()
            missing_h = []  # hydrogens which are not in the smiles
            strings_h_parent = []
            for key in data["numbering"]:
                if setup.atoms[key].is_pseudo_atom:
                    continue
                if key not in order:
                    if setup.get_atomic_num(key) != 1:
                        error_msg += (
                            "non-Hydrogen atom unexpectedely missing from smiles!?"
                        )
                        error_msg += " (mol name: %s)\n" % setup.get_mol_name()
                        pdbqt_string = ""
                        success = False
                        return pdbqt_string, success, error_msg
                    missing_h.append(key)
                    parents = setup.get_neighbors(key)
                    parents = [
                        i for i in parents if i < setup.true_atom_count
                    ]  # exclude pseudos
                    if len(parents) != 1:
                        error_msg += (
                            f"expected hydrogen {key} to be bonded to exactly one atom"
                            f" but it's bonded to {parents}"
                        )
                        error_msg += " (mol name: %s)\n" % setup.get_mol_name()
                        pdbqt_string = ""
                        success = False
                        return pdbqt_string, success, error_msg
                    parent_idx = order[parents[0]]  # already 1-indexed
                    string = " %d %d" % (
                        parent_idx,
                        data["numbering"][key],
                    )  # key 0-indexed; _numbering[key] 1-indexed
                    strings_h_parent.append(string)
            remarks_h_parent = cls.break_long_remark_lines(
                strings_h_parent, "REMARK H PARENT"
            )
            remark_prefix = "REMARK SMILES IDX"
            remark_idxmap = cls.remark_index_map(
                setup, data["numbering"], order, remark_prefix, missing_h
            )
            remarks = []
            remarks.append("REMARK SMILES %s" % smiles)  # break line at 79 chars?
            remarks.extend(remark_idxmap)
            remarks.extend(remarks_h_parent)

            for i, remark_line in enumerate(remarks):
                # Need to use 'insert' because data["numbering"]
                # is populated in self._walk_graph_recursive.
                data["pdbqt_buffer"].insert(i, remark_line)

        # torsdof is always going to be the one of the rigid, non-macrocyclic one
        data["pdbqt_buffer"].append("TORSDOF %d" % active_tors)

        pdbqt_string = eol.join(data["pdbqt_buffer"]) + eol
        return pdbqt_string, success, error_msg

    @classmethod
    def remark_index_map(
        cls, setup, numbering, order=None, prefix="REMARK INDEX MAP", missing_h=()
    ):
        """write mapping of atom indices from input molecule to output PDBQT
        order[ob_index(i.e. 'key')] = smiles_index
        """

        if order is None:
            order = {key: key + 1 for key in numbering}  # key+1 breaks OB
        # max_line_length = 79
        # remark_lines = []
        # line = prefix
        strings = []
        for key in numbering:
            if setup.atoms[key].is_pseudo_atom:
                continue
            if key in missing_h:
                continue
            string = " %d %d" % (order[key], numbering[key])
            strings.append(string)
        return cls.break_long_remark_lines(strings, prefix)
        #    candidate_text = " %d %d" % (order[key], self._numbering[key])
        #    if (len(line) + len(candidate_text)) < max_line_length:
        #        line += candidate_text
        #    else:
        #        remark_lines.append(line)
        #        line = 'REMARK INDEX MAP' + candidate_text
        # remark_lines.append(line)
        # return remark_lines

    @staticmethod
    def break_long_remark_lines(strings, prefix, max_line_length=79):
        remarks = [prefix]
        for string in strings:
            if (len(remarks[-1]) + len(string)) < max_line_length:
                remarks[-1] += string
            else:
                remarks.append(prefix + string)
        return remarks

    @staticmethod
    def adapt_pdbqt_for_autodock4_flexres(
        pdbqt_string, res, chain, num, skip_rename_ca_cb=False, atom_count=None
    ):
        """adapt pdbqt_string to be compatible with AutoDock4 requirements:
         - first and second atoms named CA and CB
         - write BEGIN_RES / END_RES
         - remove TORSDOF
        this is for covalent docking (tethered)
        """
        new_string = "BEGIN_RES %s %s %s" % (res, chain, num) + eol
        atom_number = 0
        offset = atom_count
        for line in pdbqt_string.split(eol):
            if line == "":
                continue
            if line.startswith("TORSDOF"):
                continue
            if line.startswith("ATOM"):
                if not skip_rename_ca_cb:
                    atom_number += 1
                    if atom_number == 1:
                        line = line[:13] + "CA" + line[15:]
                    elif atom_number == 2:
                        line = line[:13] + "CB" + line[15:]
                if atom_count is not None:
                    atom_count += 1
                    n = "%5d" % atom_count
                    n = n[:5]
                    line = line[:6] + n + line[11:]
                new_string += line + eol
                continue
            elif offset is not None and (
                line.startswith("BRANCH") or line.startswith("ENDBRANCH")
            ):
                keyword, i, j = line.split()
                new_string += (
                    f"{keyword} {int(i)+offset:3d} {int(j)+offset:3d}" + eol
                )
                continue
            new_string += line + eol
        new_string += "END_RES %s %s %s" % (res, chain, num) + eol
        if atom_count is None:
            return new_string  # just keeping backwards compatibility
        else:
            return new_string, atom_count
