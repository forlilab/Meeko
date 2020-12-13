#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
#

import sys

from openbabel import openbabel as ob

from .atomtyper import AtomTyperLegacy as AtomTyper


class MolWriterLegacyPDBQT():
    def __init__(self):
        #self.element_table = ob.OBElementTable()
        self.debug = []

    def write(self, mol, filename=None): #, flex_model=0):
        """ """
        self.__FILENAME = filename
        self.mol = mol

        self.model = mol.setup.flexibility_model
        # get a copy of the current setup, since it's going to be messed up by the hacks for legacy, D3R, etc...
        self.setup = mol.setup.copy()
        # print "\n\n\n------------------------------------------\n\n WRITNIG MODEL", flex_model, "score", self.model['score']
        # if flex_model==0:
        #     # first flexible model does not have alternative setups
        #     self.setup=self.mol.setup
        # else:
        #     # multiple flexible models will have their own setup
        #     self.setup = self.model['setup']
        root = self.model['root']
        self.fix_atom_types()
        # self.BACE()
        self._atom_counter = {}

        self._buff = []
        # reference = mol.setup.flex_configurations[0]
        torsdof = len(self.model['rigid_body_graph']) - 1
        if 'torsions_org' in  self.model:
            torsdof_org = self.model['torsions_org']
            self._buff.append( 'REMARK  Flexibility Score: %2.2f' % self.model['score'] )
            for bond_id, data in list(self.setup.ring_bond_breakable.items()):
                if data['active'] == True:
                    self._buff.append( 'REMARK  Glue-bond: [% 2d ] :: [% 2d ]' % (bond_id[0], bond_id[1]) )
            self._buff.append( 'REMARK  Active torsions [% 2d ] -> [% 2d ]' % (torsdof_org, torsdof) )
            active_tors = torsdof_org
        else:
            active_tors = torsdof
        self._count = 1
        self._visited = []
        self._numbering = {}
        self._walk_graph_recursive(root, first=True)
        # torsdof is always going to be the one of the rigid, non-macrocyclic one
        # self._buff.append('TORSDOF %d\n' % (len(self.model['rigid_body_graph'])-1))
        self._buff.append('TORSDOF %d\n' % active_tors)
        del self._atom_counter
        del self.model
        del self.mol
        del self.setup
        # del self._current_setup
        if filename is None:
            return self._buff
        try:
            with open(filename,'w') as fp:
                fp.write('\n'.join(self._buff))
            return True
        except:
            print("Error [%s]" % sys.exc_info()[1])
            return False

    def fix_atom_types(self, print_full_types=False):
        """ set legacy atom types and update closure atoms"""
        # print "SKIPPA FIXING ATOM TYPES"
        # return
        self.atom_typer = AtomTyper()
        print("RESTORE ATOM TYPES")
        self.atom_typer.set_param_legacy(self.mol)

        self.dpf_lines = []
        dpf_line = 'intnbp_r_eps 0.0 0.0 %d %d % s % s'
        # intnbp_r_eps  1.51 10.0000042 12 2 G0 CG
        parm_13 = []
        parm_14 = []
        all_types = []
        glue = []

        target_table = {'A' :'AG',
                        'OA':'OG',
                        'NA':'NG',
                        'N' :'Ng',
                        'SA':'SG',
                        'S' :'Sg',
                        'C' :'CG',
                        #'P' :'PG', #
                        }
        neighbor13_14_tab = {'OA':'O1',
                             'NA':'N1',
                             'N' :'N2',
                             'SA':'S1',
                             'Cl':'Cx',
                             'Br':'B1',
                             'HD':'H1',
                             'P':'P1'}

        # setup = self.model['setup']
        for i, bond_id in enumerate(self.model['broken_bonds']):
            # update targe  t atoms
            for target_idx in bond_id:
                # update C-g atoms
                curr_at_type = self.setup.get_atom_type(target_idx)
                at_type = target_table.get(curr_at_type, None)
                if at_type is None:
                    at_type = "%sG" % curr_at_type
                # print "updating  target   [% 3d] |% 2s| -> |% 2s|" %(target_idx, curr_at_type, at_type)
                self.setup.set_atom_type(target_idx, at_type)
                # print "TARGET IS", target_idx, at_type
                all_types.append(at_type)

                # updating 1,3 and 1,4
                for kind, atom_idx_list in list(self.model['neigh_13_14'][i][target_idx].items()):
                    # print " - updating [%s] neighbors of %d " % (kind, target_idx)
                    # print "type: [%s]" % kind
                    for idx in atom_idx_list:
                        if self.setup.get_ignore(idx):
                            continue
                        curr_at_type = self.setup.get_atom_type(idx)
                        at_type = neighbor13_14_tab.get(curr_at_type, None)
                        if at_type is None:
                            at_type = "%s1" % curr_at_type
                        # print "   >> atom type [% 3d] |% 2s| -> |% 2s|" %(idx, curr_at_type, at_type)
                        self.setup.set_atom_type(idx, at_type)
                        # print "XXX", self.setup.get_atom_type(idx)
                        all_types.append(at_type)

        if not print_full_types:
            return
        all_types = set(all_types)
        unique_pairs = []
        for a1 in all_types:
            for a2 in all_types:
                unique_pairs.append((a1, a2))
        for p in set(unique_pairs):
            line = dpf_line % (1, 2, p[0], p[1])
            self.dpf_lines.append(line)
            print("LINE>", line)

        full=True
        if full:
            all_types=set(list(neighbor13_14_tab.values()) + list(target_table.values()))
            unique_pairs = []
            for a1 in all_types:
                for a2 in all_types:
                    unique_pairs.append((a1, a2))
            for p in set(unique_pairs):
                line = dpf_line % (1, 2, p[0], p[1])
                self.dpf_lines.append(line)
                print("FULL>", line)

    def BACE(self):
        """
        [C](=[O:1])[N:2][C][C:3][O]'
        where
        :1 is for OX
        :2 is for NX
        :3 is for CX
        """
        pattern = '[C](=[O:1])[N:2][C][C:3][O]'
        atom_types = {1:'OX', 2:'NX', 3:'CX'}

        found = self.setup.smarts.find_pattern(pattern)
        if not found:
            return
        if len(found) > 1:
            print("****** WARNING! Multiple pattern matching for BACE constraint! *********")
        # print "FOUNDX", found
        found = found[0]
        for idx, atype in list(atom_types.items()):
            atom_idx = found[idx]
            self.setup.set_atom_type(atom_idx, atype)

    def _walk_graph_recursive(self, node, edge_start=0, first=False): #, rigid_body_id=None):
        """ recursive walk of rigid bodies"""
        """

        print("\n============================WALK")
        print("CALLED WITH", node, edge_start)
        print("RB MEMNBERS", self.model['rigid_body_members'])
        print("RB CONNECTIVIYY", self.model['rigid_body_connectivity'])
        print("------------------------")
        """
        if first:
            self._buff.append('ROOT')
        # add atoms in this rigid body
        self.debug.append("MODEL")
        current_rb = node
        # for member in sorted(self.model['rigid_body_members'][node]):
        if first:
            member_pool = sorted(self.model['rigid_body_members'][node])
        else:
            member_pool = self.model['rigid_body_members'][node][:]
            # print("MEMBERPOOL NOW", member_pool)
            # print("REMOVING", edge_start)
            member_pool.remove(edge_start)
            member_pool = [edge_start] + member_pool
        
        # print(member_pool)
        
        for member in member_pool:
            if self.setup.atom_ignore[member] == 1:
                continue
            line = self._make_pdbqt_line(member)
            self._buff.append(line)
            self.debug.append(line)
            self._numbering[member] = self._count
            self._count += 1
        if first:
            self._buff.append('ENDROOT')
        self.debug.append('ENDMDL')
        self._visited.append(node)
        for neigh in self.model['rigid_body_graph'][node]:
            if neigh in self._visited:
                continue
            # open branch
            begin, next_index = self.model['rigid_body_connectivity'][node, neigh]
            begin = self._numbering[begin]
            end = self._count
            # print("END IS NOW", end)
            line = "BRANCH % 4d % 4d" % (begin, end)
            self._buff.append(line)
            self._walk_graph_recursive(neigh, edge_start=next_index)
            line = "ENDBRANCH % 4d % 4d" % (begin, end)
            self._buff.append(line)

    def _make_pdbqt_line(self, atom_idx):
        """ """
        _type = "ATOM"
        alt_id = " "
        res_name = 'LIG'
        chain = "L"
        res_seq = 1
        in_code = ""
        occupancy = 1.0
        temp_factor = 0.0
        atomic_num = self.setup.element[atom_idx]
        atom_symbol = ob.GetSymbol(atomic_num)
        if not atom_symbol in self._atom_counter:
            self._atom_counter[atom_symbol] = 0
        self._atom_counter[atom_symbol] += 1
        atom_count = self._atom_counter[atom_symbol]
        atom_name = "%s%d" % (atom_symbol, atom_count)
        coord = self.setup.coord[atom_idx]
        element = self.setup.get_atom_type(atom_idx)
        #print("ELEMENT", element)
        charge = self.setup.charge[atom_idx]
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"
        return atom.format(_type, self._count, atom_name, alt_id, res_name, chain,
                    res_seq, in_code, float(coord[0]), float(coord[1]), float(coord[2]),
                    occupancy, temp_factor, charge, element)
