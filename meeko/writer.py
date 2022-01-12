#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko PDBQT writer
#

import sys
import json

import numpy as np
from rdkit import Chem
from .utils import pdbutils
from .utils.rdkitutils import mini_periodic_table


class PDBQTWriterLegacy():
    def __init__(self):
        """Initialize the PDBQT writer."""
        self._count = 1
        self._visited = []
        self._numbering = {}
        self._pdbqt_buffer = []
        self._resinfo_set = set() # for flexres keywords BEGIN_RES / END_RES

    def _get_pdbinfo_fitting_pdb_chars(self, pdbinfo):
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

    def _make_pdbqt_line(self, atom_idx):
        """ """
        record_type = "ATOM"
        alt_id = " "
        pdbinfo = self.setup.pdbinfo[atom_idx]
        if pdbinfo is None:
            pdbinfo = pdbutils.PDBAtomInfo('', '', 0, '')
        resinfo = pdbutils.PDBResInfo(pdbinfo.resName, pdbinfo.resNum, pdbinfo.chain)
        self._resinfo_set.add(resinfo)
        atom_name, res_name, res_num, chain = self._get_pdbinfo_fitting_pdb_chars(pdbinfo)
        in_code = ""
        occupancy = 1.0
        temp_factor = 0.0
        coord = self.setup.coord[atom_idx]
        atom_type = self.setup.get_atom_type(atom_idx)
        charge = self.setup.charge[atom_idx]
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"

        return atom.format(record_type, self._count, pdbinfo.name, alt_id, res_name, chain,
                           res_num, in_code, float(coord[0]), float(coord[1]), float(coord[2]),
                           occupancy, temp_factor, charge, atom_type)

    def _walk_graph_recursive(self, node, edge_start=0, first=False): #, rigid_body_id=None):
        """ recursive walk of rigid bodies"""
        if first:
            self._pdbqt_buffer.append('ROOT')
            member_pool = sorted(self.model['rigid_body_members'][node])
        else:
            member_pool = self.model['rigid_body_members'][node][:]
            member_pool.remove(edge_start)
            member_pool = [edge_start] + member_pool

        for member in member_pool:
            if self.setup.atom_ignore[member] == 1:
                continue

            self._pdbqt_buffer.append(self._make_pdbqt_line(member))
            self._numbering[member] = self._count # _count starts at 1
            self._count += 1

        if first:
            self._pdbqt_buffer.append('ENDROOT')

        self._visited.append(node)

        for neigh in self.model['rigid_body_graph'][node]:
            if neigh in self._visited:
                continue

            # Write the branch
            begin, next_index = self.model['rigid_body_connectivity'][node, neigh]

            # do not write branch (or anything downstream) if any of the two atoms
            # defining the rotatable bond are ignored
            if self.setup.atom_ignore[begin] or self.setup.atom_ignore[next_index]:
                continue

            begin = self._numbering[begin]
            end = self._count

            self._pdbqt_buffer.append("BRANCH %3d %3d" % (begin, end))
            self._walk_graph_recursive(neigh, edge_start=next_index)
            self._pdbqt_buffer.append("ENDBRANCH %3d %3d" % (begin, end))

    def write_string(self, setup, add_index_map=False, remove_smiles=False):
        """Output a PDBQT file as a string.

        Args:
            setup: MoleculeSetup

        Returns:
            str: PDBQT string of the molecule

        """
        self._count = 1
        self._visited = []
        self._numbering = {}
        self._pdbqt_buffer = []
        self._atom_counter = {}
        self._resinfo_set = set()

        self.setup = setup
        self.model = setup.flexibility_model
        # get a copy of the current setup, since it's going to be messed up by the hacks for legacy, D3R, etc...
        self.setup = setup.copy()

        root = self.model['root']
        torsdof = len(self.model['rigid_body_graph']) - 1

        if 'torsions_org' in self.model:
            torsdof_org = self.model['torsions_org']
            self._pdbqt_buffer.append('REMARK Flexibility Score: %2.2f' % self.model['score'] )
            active_tors = torsdof_org
        else:
            active_tors = torsdof

        self._walk_graph_recursive(root, first=True)

        if add_index_map:
            for i, remark_line in enumerate(self.remark_index_map()):
                # need to use 'insert' because self._numbering is calculated
                # only after self._walk_graph_recursive
                self._pdbqt_buffer.insert(i, remark_line)

        if not remove_smiles:
            smiles, order = self.setup.get_smiles_and_order()
            missing_h = [] # hydrogens which are not in the smiles
            strings_h_parent = []
            for key in self._numbering:
                if key in self.setup.atom_pseudo: continue
                if key not in order:
                    element = self.setup.get_element(key)
                    #atom = self.setup.mol.GetAtomWithIdx(key)
                    if element != 1:
                        raise RuntimeError("non-Hydrogen atom unexpectedely missing from smiles!?")
                    missing_h.append(key)
                    parents = self.setup.get_neigh(key)
                    parents = [i for i in parents if i < self.setup.atom_true_count] # exclude pseudos
                    if len(parents) != 1:
                        raise RuntimeError("expected hydrogen to have exactly one parent")
                    parent_idx = order[parents[0]] # already 1-indexed
                    string = ' %d %d' % (parent_idx, self._numbering[key]) # key 0-indexed; _numbering[key] 1-indexed
                    strings_h_parent.append(string)
            remarks_h_parent = self.break_long_remark_lines(strings_h_parent, "REMARK H PARENT")
            remark_prefix = "REMARK SMILES IDX"
            remark_idxmap = self.remark_index_map(order, remark_prefix, missing_h)
            remarks = []
            remarks.append("REMARK SMILES %s" % smiles) # break line at 79 chars?
            remarks.extend(remark_idxmap)
            remarks.extend(remarks_h_parent)

            for i, remark_line in enumerate(remarks):
                # need to use 'insert' because self._numbering is calculated
                # only after self._walk_graph_recursive
                self._pdbqt_buffer.insert(i, remark_line)

        if self.setup.is_protein_sidechain:
            if len(self._resinfo_set) > 1:
                print("Warning: more than a single resName, resNum, chain in flexres", file=sys.stderr)
                print(self._resinfo_set, file=sys.stderr)
            resinfo = list(self._resinfo_set)[0]
            pdbinfo = pdbutils.PDBAtomInfo('', resinfo.resName, resinfo.resNum, resinfo.chain)
            _, res_name, res_num, chain = self._get_pdbinfo_fitting_pdb_chars(pdbinfo)
            resinfo_string = "{:3s} {:1s}{:4d}".format(res_name, chain, res_num)
            self._pdbqt_buffer.insert(0, 'BEGIN_RES %s' % resinfo_string)
            self._pdbqt_buffer.append('END_RES %s' % resinfo_string)
        else: # no TORSDOF in flexres
            # torsdof is always going to be the one of the rigid, non-macrocyclic one
            self._pdbqt_buffer.append('TORSDOF %d' % active_tors)


        return '\n'.join(self._pdbqt_buffer) + '\n'


    def remark_index_map(self, order=None, prefix="REMARK INDEX MAP", missing_h=[]):
        """ write mapping of atom indices from input molecule to output PDBQT
            order[ob_index(i.e. 'key')] = smiles_index
        """

        if order is None: order = {key: key+1 for key in self._numbering} # FIXME key+1 breaks OB
        #max_line_length = 79
        #remark_lines = []
        #line = prefix
        strings = []
        for key in self._numbering:
            if key in self.setup.atom_pseudo: continue
            if key in missing_h: continue
            string = " %d %d" % (order[key], self._numbering[key])
            strings.append(string)
        return self.break_long_remark_lines(strings, prefix)
        #    candidate_text = " %d %d" % (order[key], self._numbering[key])
        #    if (len(line) + len(candidate_text)) < max_line_length:
        #        line += candidate_text
        #    else:
        #        remark_lines.append(line)
        #        line = 'REMARK INDEX MAP' + candidate_text
        #remark_lines.append(line)
        #return remark_lines

    def break_long_remark_lines(self, strings, prefix, max_line_length=79):
        remarks = [prefix]
        for string in strings:
            if (len(remarks[-1]) + len(string)) < max_line_length:
                remarks[-1] += string
            else:
                remarks.append(prefix + string)
        return remarks
