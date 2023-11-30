#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko PDBQT writer
#

import sys
import json
import math
import pathlib

import numpy as np
from rdkit import Chem
from .utils import pdbutils
from .utils.rdkitutils import mini_periodic_table

linesep = pathlib.os.linesep

def oids_json_from_setup(molsetup, name="LigandFromMeeko"):
    if len(molsetup.restraints):
        raise NotImplementedError("molsetup has restraints but these aren't written to oids block yet")
    offchrg_type = "OFFCHRG"
    offchrg_by_parent = {}
    for i in molsetup.atom_pseudo:
        if molsetup.atom_type[i] == offchrg_type:
            neigh = molsetup.get_neigh(i)
            if len(neigh) != 1:
                raise RuntimeError("offsite charge %s is bonded to: %s which has len() != 1" % (
                    i, json.dumps(neigh)))
            if neigh[0] in offchrg_by_parent:
                raise RuntimeError("atom %d has more than one offsite charge" % neigh[0])
            offchrg_by_parent[neigh[0]] = i
    output_indices_start_at_one = True
    index_start = int(output_indices_start_at_one)
    positions_block = ""
    charges = []
    offchrg_by_oid_parent = {}
    elements = []
    n_real_atoms = molsetup.atom_true_count
    n_fake_atoms = len(molsetup.atom_pseudo)
    indexmap = {} # molsetup: oid
    count_oids = 0
    for index in range(n_real_atoms):
        if molsetup.atom_ignore[index]:
            continue
        if molsetup.atom_type[index] == offchrg_type:
            continue # handled by offchrg_by_parent
        oid_id = count_oids + index_start
        indexmap[index] = count_oids
        x, y, z = molsetup.coord[index]
        positions_block += "position.%d = (%f,%f,%f)\n" % (oid_id, x, y, z)
        charges.append(molsetup.charge[index])
        if index in offchrg_by_parent:
            index_pseudo = offchrg_by_parent[index]
            xq_abs, yq_abs, zq_abs = molsetup.coord[index_pseudo]
            xq_rel = xq_abs - x
            yq_rel = yq_abs - y
            zq_rel = zq_abs - z
            offchrg_by_oid_parent[count_oids] = {
                "q": molsetup.charge[index_pseudo],
                "xyz": (xq_rel, yq_rel, zq_rel),
            }
        count_oids += 1
        element = "%s %s %d" % (name, molsetup.atom_type[index], oid_id)
        elements.append(element)
    
    tmp = []
    for index in range(len(charges)):
        if index in offchrg_by_oid_parent:
            tmplist = ["%f" % charges[index], "0.0", "0.0" ,"0.0"] # xyz relative to current elemtn
            tmplist.append("%f" % offchrg_by_oid_parent[index]["q"])
            tmplist.append("%f,%f,%f" % offchrg_by_oid_parent[index]["xyz"])
            tmp.append(",".join(tmplist))
        else:
            tmp.append("%f" % charges[index])
    charges_line = "import_charges = {%s}\n" % ("|".join(tmp))
    elements_line = "elements = %s\n" % (",".join(elements))

    bonds = [[] for _ in range(count_oids)]
    bond_orders = [[] for _ in range(count_oids)]
    static_links = []
    for i, j in molsetup.bond.keys():
        if molsetup.atom_ignore[i] or molsetup.atom_ignore[j]:
            continue
        if molsetup.atom_type[i] == offchrg_type or molsetup.atom_type[j] == offchrg_type:
            continue
        oid_i = indexmap[i]
        oid_j = indexmap[j]
        bonds[oid_i].append("%d" % (oid_j+index_start))
        bond_orders[oid_i].append("%d" % molsetup.bond[(i, j)]["bond_order"])
        if not molsetup.bond[(i, j)]["rotatable"]:
            static_links.append("%d,%d" % (oid_i + index_start, oid_j + index_start))
    bonds = [",".join(j_list) for j_list in bonds]
    bonds_line = "connectivity = {%s}\n" % ("|".join(bonds))
    bond_orders = [",".join(orders) for orders in bond_orders]
    bondorder_line = "bond_order = {%s}\n" % ("|".join(bond_orders))
    staticlinks_line = "static_links = {%s}\n" % ("|".join(static_links))


    output = ""
    output += "[Group: %s]\n" % name
    output += positions_block
    output += charges_line
    output += elements_line
    output += bonds_line
    output += bondorder_line
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
        raise NotImplementedError("molsetup has restraints but these aren't written to oids block yet")
    offchrg_type = "OFFCHRG"
    offchrg_by_parent = {}
    for i in molsetup.atom_pseudo:
        if molsetup.atom_type[i] == offchrg_type:
            neigh = molsetup.get_neigh(i)
            if len(neigh) != 1:
                raise RuntimeError("offsite charge %s is bonded to: %s which has len() != 1" % (
                    i, json.dumps(neigh)))
            if neigh[0] in offchrg_by_parent:
                raise RuntimeError("atom %d has more than one offsite charge" % neigh[0])
            offchrg_by_parent[neigh[0]] = i
    output_indices_start_at_one = True
    index_start = int(output_indices_start_at_one)
    positions_block = ""
    charges = []
    offchrg_by_oid_parent = {}
    elements = []
    n_real_atoms = molsetup.atom_true_count
    n_fake_atoms = len(molsetup.atom_pseudo)
    indexmap = {} # molsetup: oid
    count_oids = 0
    for index in range(n_real_atoms):
        if molsetup.atom_ignore[index]:
            continue
        if molsetup.atom_type[index] == offchrg_type:
            continue # handled by offchrg_by_parent
        oid_id = count_oids + index_start
        indexmap[index] = count_oids
        x, y, z = molsetup.coord[index]
        positions_block += "position.%d = (%f,%f,%f)\n" % (oid_id, x, y, z)
        charges.append(molsetup.charge[index])
        if index in offchrg_by_parent:
            index_pseudo = offchrg_by_parent[index]
            xq_abs, yq_abs, zq_abs = molsetup.coord[index_pseudo]
            xq_rel = xq_abs - x
            yq_rel = yq_abs - y
            zq_rel = zq_abs - z
            offchrg_by_oid_parent[count_oids] = {
                "q": molsetup.charge[index_pseudo],
                "xyz": (xq_rel, yq_rel, zq_rel),
            }
        count_oids += 1
        element = "%s %s %d" % (name, molsetup.atom_type[index], oid_id)
        elements.append(element)
    
    tmp = []
    for index in range(len(charges)):
        if index in offchrg_by_oid_parent:
            tmplist = ["%f" % charges[index], "0.0", "0.0" ,"0.0"] # xyz relative to current elemtn
            tmplist.append("%f" % offchrg_by_oid_parent[index]["q"])
            tmplist.append("%f,%f,%f" % offchrg_by_oid_parent[index]["xyz"])
            tmp.append(",".join(tmplist))
        else:
            tmp.append("%f" % charges[index])
    charges_line = "import_charges = {%s}\n" % ("|".join(tmp))
    elements_line = "elements = %s\n" % (",".join(elements))

    bonds = [[] for _ in range(count_oids)]
    bond_orders = [[] for _ in range(count_oids)]
    static_links = []
    for i, j in molsetup.bond.keys():
        if molsetup.atom_ignore[i] or molsetup.atom_ignore[j]:
            continue
        if molsetup.atom_type[i] == offchrg_type or molsetup.atom_type[j] == offchrg_type:
            continue
        oid_i = indexmap[i]
        oid_j = indexmap[j]
        bonds[oid_i].append("%d" % (oid_j+index_start))
        bond_orders[oid_i].append("%d" % molsetup.bond[(i, j)]["bond_order"])
        if not molsetup.bond[(i, j)]["rotatable"]:
            static_links.append("%d,%d" % (oid_i + index_start, oid_j + index_start))
    bonds = [",".join(j_list) for j_list in bonds]
    bonds_line = "connectivity = {%s}\n" % ("|".join(bonds))
    bond_orders = [",".join(orders) for orders in bond_orders]
    bondorder_line = "bond_order = {%s}\n" % ("|".join(bond_orders))
    staticlinks_line = "static_links = {%s}\n" % ("|".join(static_links))


    output = ""
    output += "[Group: %s]\n" % name
    output += positions_block
    output += charges_line
    output += elements_line
    output += bonds_line
    output += bondorder_line
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
        if (molsetup.atom_ignore[a] or
            molsetup.atom_ignore[b] or
            molsetup.atom_ignore[c] or
            molsetup.atom_ignore[d]):
            continue
        bond_id = molsetup.get_bond_id(b, c)
        if not molsetup.bond[bond_id]["rotatable"]:
            continue
        index = molsetup.dihedral_partaking_atoms[atomidx]
        atomidx_by_index.setdefault(index, set())
        atomidx_by_index[index].add(atomidx)
        label = molsetup.dihedral_labels[atomidx] if atomidx in molsetup.dihedral_labels else None
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
            string = ",".join(["%d" % (indexmap[i]+1) for i in atomidx])
            atomidx_strings.append(string)
        text += "elements = {%s}\n" % ("|".join(atomidx_strings))
        text += "parameters = %s\n" % _aux_fourier_conversion(molsetup.dihedral_interactions[index])
        text += '\n'
    return text

def _aux_fourier_conversion(fourier_series):
    # convert from:
    #   k*(1+cos(n*theta-phase))
    # to:
    #   (k/2)*(1+cos(n*(theta+phase)))
    # where n = periodicity
    max_periodicity = max([fs['periodicity'] for fs in fourier_series])
    tmp = [(0, 0)] * max_periodicity
    for fs in fourier_series:
        i = fs['periodicity'] - 1
        k = 2.0 * fs['k']
        phase = -1 * fs['phase']
        tmp[i] = (k, phase)
    strings = []
    periodicity = 0
    for (k, phase) in tmp:
        periodicity += 1
        k_str = '0'
        if phase == 0:
            phase_str = '0'
        else:
            phase_str = ('%f' % (phase/np.pi)).rstrip('0').rstrip('.') + '*pi'
            if phase_str == '1*pi':
                phase_str = 'pi'
            if phase_str == '-1*pi':
                phase_str = '-pi'
            if periodicity != 1:
                phase_str += "/%d" % periodicity
        if k != 0: k_str = '%f*4.184/60.221' % (k)
        strings.append("%s,%s" % (k_str, phase_str))
    return "(" + ";".join(strings) + ")"



class PDBQTWriterLegacy():

    @staticmethod
    def _get_pdbinfo_fitting_pdb_chars(pdbinfo):
        """ return strings and integers that are guaranteed
            to fit within the designated chars of the PDB format """

        atom_name = pdbinfo.name
        res_name = pdbinfo.resName
        res_num = pdbinfo.resNum
        chain = pdbinfo.chain
        if len(atom_name) > 4: atom_name = atom_name[0:4]
        if len(res_name) > 3: res_name = res_name[0:3]
        if res_num > 9999: res_num = res_num % 10000
        if len(chain) > 1: chain = chain[0:1]
        return atom_name, res_name, res_num, chain

    @classmethod
    def _make_pdbqt_line_from_molsetup(cls, setup, atom_idx, count):
        """ """
        pdbinfo = setup.pdbinfo[atom_idx]
        if pdbinfo is None:
            pdbinfo = pdbutils.PDBAtomInfo('', '', 0, '')
        atom_name, res_name, res_num, chain = cls._get_pdbinfo_fitting_pdb_chars(pdbinfo)
        coord = setup.coord[atom_idx]
        atom_type = setup.get_atom_type(atom_idx)
        charge = setup.charge[atom_idx]
        pdbqt_line = cls._make_pdbqt_line(
                count, atom_name, res_name, chain, res_num, coord, charge, atom_type)
        return pdbqt_line

    @staticmethod
    def _make_pdbqt_line(count, atom_name, res_name, chain, res_num, coord, charge, atom_type): 
        record_type = "ATOM"
        alt_id = " "
        in_code = ""
        occupancy = 1.0
        temp_factor = 0.0
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"
        pdbqt_line = atom.format(record_type, count, atom_name, alt_id, res_name, chain,
                           res_num, in_code, float(coord[0]), float(coord[1]), float(coord[2]),
                           occupancy, temp_factor, charge, atom_type)
        return pdbqt_line

    @classmethod
    def _walk_graph_recursive(cls, setup, node, data, edge_start=0, first=False):
        """ recursive walk of rigid bodies"""
        
        if first:
            data["pdbqt_buffer"].append('ROOT')
            member_pool = sorted(setup.flexibility_model['rigid_body_members'][node])
        else:
            member_pool = setup.flexibility_model['rigid_body_members'][node][:]
            member_pool.remove(edge_start)
            member_pool = [edge_start] + member_pool

        for member in member_pool:
            if setup.atom_ignore[member] == 1:
                continue
            pdbqt_line = cls._make_pdbqt_line_from_molsetup(setup, member, data["count"])
            data["pdbqt_buffer"].append(pdbqt_line)
            data["numbering"][member] = data["count"] # count starts at 1
            data["count"] += 1

        if first:
            data["pdbqt_buffer"].append('ENDROOT')

        data["visited"].append(node)

        for neigh in setup.flexibility_model['rigid_body_graph'][node]:
            if neigh in data["visited"]:
                continue

            # Write the branch
            begin, next_index = setup.flexibility_model['rigid_body_connectivity'][node, neigh]

            # do not write branch (or anything downstream) if any of the two atoms
            # defining the rotatable bond are ignored
            if setup.atom_ignore[begin] or setup.atom_ignore[next_index]:
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

        for idx, atom_type in setup.atom_type.items():
            if setup.atom_ignore[idx]:
                continue
            if atom_type is None:
                error_msg += 'atom number %d has None type, mol name: %s\n' % (idx, setup.get_mol_name())
                success = False
            c = setup.charge[idx]
            if not bad_charge_ok and (type(c) != float and type(c) != int or math.isnan(c) or math.isinf(c)):
                error_msg += 'atom number %d has non finite charge, mol name: %s, charge: %s\n' % (idx, setup.get_mol_name(), str(c))
                success = False

        return success, error_msg


    @classmethod
    def write_string_from_linked_rdkit_chorizo(cls, chorizo):
        rigid_pdbqt_string, flex_pdbqt_dict = cls.write_string_flexdict_from_linked_rdkit_chorizo(chorizo)
        flex_pdbqt_string = ""
        for res_id, pdbqt_string in flex_pdbqt_dict.items():
            flex_pdbqt_string += pdbqt_string
        return rigid_pdbqt_string, flex_pdbqt_string


    @classmethod
    def write_from_linked_rdkit_chorizo(cls, chorizo):
        rigid_pdbqt_string = ""
        flex_pdbqt_dict = {}
        atom_count = 0
        flex_atom_count = 0
        for res_id in chorizo.getValidResidues():
            resmol = chorizo.residues[res_id].rdkit_mol
            positions = resmol.GetConformer().GetPositions()
            chain, resname, resnum = res_id.split(":")
            resnum = int(resnum)
            molsetup_mapidx = {} if chorizo.residues[res_id].molsetup_mapidx is None else chorizo.residues[res_id].molsetup_mapidx
            molsetup_ignored = () if chorizo.residues[res_id].molsetup_ignored is None else chorizo.residues[res_id].molsetup_ignored
            flexres_idxs = [j for (i, j) in molsetup_mapidx.items() if j not in molsetup_ignored]
            for atom in resmol.GetAtoms():
                if atom.GetIdx() in flexres_idxs:
                    continue
                props = atom.GetPropsAsDict()
                atom_type = props["atom_type"]
                if atom_type == "H": # TODO read ignore flag in molsetup
                    continue
                coord = positions[atom.GetIdx()]
                atom_name = props.get("atom_name", "")
                charge = props["gasteiger"]
                atom_count += 1
                rigid_pdbqt_string += cls._make_pdbqt_line(
                    atom_count, atom_name, resname, chain, resnum, coord, charge, atom_type) + linesep
            if chorizo.residues[res_id].molsetup:
                chain, resname, resnum = res_id.split(":")
                molsetup = chorizo.residues[res_id].molsetup
                this_flex_pdbqt, ok, err = PDBQTWriterLegacy.write_string(molsetup, remove_smiles=True)
                if not ok:
                    raise RuntimeError(err)
                this_flex_pdbqt, flex_atom_count = cls.adapt_pdbqt_for_autodock4_flexres(
                    this_flex_pdbqt, 
                    resname,
                    chain,
                    resnum,
                    skip_rename_ca_cb=True,
                    atom_count=flex_atom_count)
                flex_pdbqt_dict[res_id] = this_flex_pdbqt
        return rigid_pdbqt_string, flex_pdbqt_dict 

    @classmethod
    def write_string(cls, setup, add_index_map=False, remove_smiles=False, bad_charge_ok=False):
        """Output a PDBQT file as a string.

        Args:
            setup: MoleculeSetup

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

        torsdof = len(setup.flexibility_model['rigid_body_graph']) - 1

        if 'torsions_org' in setup.flexibility_model:
            torsdof_org = setup.flexibility_model['torsions_org']
            data["pdbqt_buffer"].append('REMARK Flexibility Score: %2.2f' % setup.flexibility_model['score'] )
            active_tors = torsdof_org
        else:
            active_tors = torsdof

        data = cls._walk_graph_recursive(setup, setup.flexibility_model["root"], data, first=True)

        if add_index_map:
            for i, remark_line in enumerate(cls.remark_index_map(setup, data["numbering"])):
                # Need to use 'insert' because data["numbering"]
                # is populated in self._walk_graph_recursive.
                data["pdbqt_buffer"].insert(i, remark_line)

        if not remove_smiles:
            smiles, order = setup.get_smiles_and_order()
            missing_h = [] # hydrogens which are not in the smiles
            strings_h_parent = []
            for key in data["numbering"]:
                if key in setup.atom_pseudo: continue
                if key not in order:
                    if setup.get_element(key) != 1:
                        error_msg += "non-Hydrogen atom unexpectedely missing from smiles!?"
                        error_msg += " (mol name: %s)\n" % setup.get_mol_name()
                        pdbqt_string = ""
                        success = False
                        return pdbqt_string, success, error_msg
                    missing_h.append(key)
                    parents = setup.get_neigh(key)
                    parents = [i for i in parents if i < setup.atom_true_count] # exclude pseudos
                    if len(parents) != 1:
                        error_msg += "expected hydrogen to be bonded to exactly one atom"
                        error_msg += " (mol name: %s)\n" % setup.get_mol_name()
                        pdbqt_string = ""
                        success = False
                        return pdbqt_string, success, error_msg
                    parent_idx = order[parents[0]] # already 1-indexed
                    string = ' %d %d' % (parent_idx, data["numbering"][key]) # key 0-indexed; _numbering[key] 1-indexed
                    strings_h_parent.append(string)
            remarks_h_parent = cls.break_long_remark_lines(strings_h_parent, "REMARK H PARENT")
            remark_prefix = "REMARK SMILES IDX"
            remark_idxmap = cls.remark_index_map(setup, data["numbering"], order, remark_prefix, missing_h)
            remarks = []
            remarks.append("REMARK SMILES %s" % smiles) # break line at 79 chars?
            remarks.extend(remark_idxmap)
            remarks.extend(remarks_h_parent)

            for i, remark_line in enumerate(remarks):
                # Need to use 'insert' because data["numbering"]
                # is populated in self._walk_graph_recursive.
                data["pdbqt_buffer"].insert(i, remark_line)

        # torsdof is always going to be the one of the rigid, non-macrocyclic one
        data["pdbqt_buffer"].append('TORSDOF %d' % active_tors)

        pdbqt_string =  linesep.join(data["pdbqt_buffer"]) + linesep
        return pdbqt_string, success, error_msg

    @classmethod
    def remark_index_map(cls, setup, numbering, order=None, prefix="REMARK INDEX MAP", missing_h=()):
        """ write mapping of atom indices from input molecule to output PDBQT
            order[ob_index(i.e. 'key')] = smiles_index
        """

        if order is None: order = {key: key+1 for key in numbering} # key+1 breaks OB
        #max_line_length = 79
        #remark_lines = []
        #line = prefix
        strings = []
        for key in numbering:
            if key in setup.atom_pseudo: continue
            if key in missing_h: continue
            string = " %d %d" % (order[key], numbering[key])
            strings.append(string)
        return cls.break_long_remark_lines(strings, prefix)
        #    candidate_text = " %d %d" % (order[key], self._numbering[key])
        #    if (len(line) + len(candidate_text)) < max_line_length:
        #        line += candidate_text
        #    else:
        #        remark_lines.append(line)
        #        line = 'REMARK INDEX MAP' + candidate_text
        #remark_lines.append(line)
        #return remark_lines

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
    def adapt_pdbqt_for_autodock4_flexres(pdbqt_string, res, chain, num, skip_rename_ca_cb=False, atom_count=None):
        """ adapt pdbqt_string to be compatible with AutoDock4 requirements:
             - first and second atoms named CA and CB
             - write BEGIN_RES / END_RES
             - remove TORSDOF
            this is for covalent docking (tethered)
        """
        new_string = "BEGIN_RES %s %s %s" % (res, chain, num) + linesep
        atom_number = 0
        for line in pdbqt_string.split(linesep):
            if line == "":
                continue
            if line.startswith("TORSDOF"):
                continue
            if line.startswith("ATOM"):
                if not skip_rename_ca_cb:
                    atom_number+=1
                    if atom_number == 1:
                        line = line[:13] + 'CA' + line[15:]
                    elif atom_number == 2:
                        line = line[:13] + 'CB' + line[15:]
                if atom_count is not None:
                    atom_count += 1
                    n = "%5d" % atom_count
                    n = n[:5]
                    line = line[:6] + n + line[11:]
                new_string += line + linesep
                continue
            new_string += line + linesep
        new_string += "END_RES %s %s %s" % (res, chain, num) + linesep
        if atom_count is None:
            return new_string # just keeping backwards compatibility
        else:
            return new_string, atom_count
