#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko PDBQT writer
#

import sys
import json
import math

import numpy as np
from rdkit import Chem
from .utils import pdbutils
from .utils.rdkitutils import mini_periodic_table


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
    def _make_pdbqt_line(cls, setup, atom_idx, resinfo_set, count):
        """ """
        record_type = "ATOM"
        alt_id = " "
        pdbinfo = setup.pdbinfo[atom_idx]
        if pdbinfo is None:
            pdbinfo = pdbutils.PDBAtomInfo('', '', 0, '')
        resinfo = pdbutils.PDBResInfo(pdbinfo.resName, pdbinfo.resNum, pdbinfo.chain)
        resinfo_set.add(resinfo)
        atom_name, res_name, res_num, chain = cls._get_pdbinfo_fitting_pdb_chars(pdbinfo)
        in_code = ""
        occupancy = 1.0
        temp_factor = 0.0
        coord = setup.coord[atom_idx]
        atom_type = setup.get_atom_type(atom_idx)
        charge = setup.charge[atom_idx]
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"

        pdbqt_line = atom.format(record_type, count, pdbinfo.name, alt_id, res_name, chain,
                           res_num, in_code, float(coord[0]), float(coord[1]), float(coord[2]),
                           occupancy, temp_factor, charge, atom_type)
        return pdbqt_line, resinfo_set

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
            pdbqt_line, resinfo_set = cls._make_pdbqt_line(setup, member, data["resinfo_set"], data["count"])
            data["resinfo_set"] = resinfo_set # written as if _make_pdbqt_line() doesn't modify its args (for readability)
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

        success = True
        error_msg = ""
        
        if setup.has_implicit_hydrogens():
            error_msg += "molecule has implicit hydrogens (name=%s)\n" % setup.get_mol_name()
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

        if not success:
            pdbqt_string = ""
            return pdbqt_string, success, error_msg

        data = {
            "visited": [],
            "numbering": {},
            "pdbqt_buffer": [],
            "count": 1,
            "resinfo_set": set(),
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

        if False: #self.setup.is_protein_sidechain:
            if len(data["resinfo_set"]) > 1:
                print("Warning: more than a single resName, resNum, chain in flexres", file=sys.stderr)
                print(data["resinfo_set"], file=sys.stderr)
            resinfo = list(data["resinfo_set"])[0]
            pdbinfo = pdbutils.PDBAtomInfo('', resinfo.resName, resinfo.resNum, resinfo.chain)
            _, res_name, res_num, chain = cls._get_pdbinfo_fitting_pdb_chars(pdbinfo)
            resinfo_string = "{:3s} {:1s}{:4d}".format(res_name, chain, res_num)
            data["pdbqt_buffer"].insert(0, 'BEGIN_RES %s' % resinfo_string)
            data["pdbqt_buffer"].append('END_RES %s' % resinfo_string)
        else: # no TORSDOF in flexres
            # torsdof is always going to be the one of the rigid, non-macrocyclic one
            data["pdbqt_buffer"].append('TORSDOF %d' % active_tors)

        pdbqt_string =  '\n'.join(data["pdbqt_buffer"]) + '\n'
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
    def adapt_pdbqt_for_autodock4_flexres(pdbqt_string, res, chain, num):
        """ adapt pdbqt_string to be compatible with AutoDock4 requirements:
             - first and second atoms named CA and CB
             - write BEGIN_RES / END_RES
             - remove TORSDOF
            this is for covalent docking (tethered)
        """
        new_string = "BEGIN_RES %s %s %s\n" % (res, chain, num)
        atom_number = 0
        for line in pdbqt_string.split("\n"):
            if line == "":
                continue
            if line.startswith("TORSDOF"):
                continue
            if line.startswith("ATOM"):
                atom_number+=1
                if atom_number == 1:
                    line = line[:13] + 'CA' + line[15:]
                elif atom_number == 2:
                    line = line[:13] + 'CB' + line[15:]
                new_string += line + '\n'
                continue
            new_string += line + '\n'
        new_string += "END_RES %s %s %s\n" % (res, chain, num)
        return new_string
