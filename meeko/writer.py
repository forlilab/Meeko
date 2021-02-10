#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko PDBQT writer
#

import sys

from openbabel import openbabel as ob

from .atomtyper import AtomTyperLegacy


class PDBQTWriterLegacy():
    def __init__(self):
        """Initialize the PDBQT writer."""
        self._atom_typer = AtomTyperLegacy()
        self._count = 1
        self._visited = []
        self._numbering = {}
        self._pdbqt_buffer = []
        self._atom_counter = {}

    def _fix_atom_types(self):
        """ set legacy atom types and update closure atoms"""
        self._atom_typer.set_param_legacy(self.mol)

        target_table = {'A' :'AG',
                        'OA':'OG',
                        'NA':'NG',
                        'N' :'Ng',
                        'SA':'SG',
                        'S' :'Sg',
                        'C' :'CG0',
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

        for i, bond_id in enumerate(self.model['broken_bonds']):
            # update target atoms
            for target_idx in bond_id:
                # update C-g atoms
                curr_at_type = self.setup.get_atom_type(target_idx)
                at_type = target_table.get(curr_at_type, None)
                if at_type is None:
                    at_type = "%sG" % curr_at_type

                self.setup.set_atom_type(target_idx, at_type)

    def _make_pdbqt_line(self, atom_idx):
        """ """
        record_type = "ATOM"
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
        atom_type = self.setup.get_atom_type(atom_idx)
        charge = self.setup.charge[atom_idx]
        atom = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}    {:6.3f} {:<2s}"

        return atom.format(record_type, self._count, atom_name, alt_id, res_name, chain,
                           res_seq, in_code, float(coord[0]), float(coord[1]), float(coord[2]),
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
            self._numbering[member] = self._count
            self._count += 1

        if first:
            self._pdbqt_buffer.append('ENDROOT')

        self._visited.append(node)

        for neigh in self.model['rigid_body_graph'][node]:
            if neigh in self._visited:
                continue

            # Write the branch
            begin, next_index = self.model['rigid_body_connectivity'][node, neigh]
            begin = self._numbering[begin]
            end = self._count

            self._pdbqt_buffer.append("BRANCH %4d %4d" % (begin, end))
            self._walk_graph_recursive(neigh, edge_start=next_index)
            self._pdbqt_buffer.append("ENDBRANCH %4d %4d" % (begin, end))

    def write_string(self, mol):
        """Output a PDBQT file as a string.
        
        Args:
            mol (OBMol): OBMol that was prepared with Meeko

        Returns:
            str: PDBQT string of the molecule

        """
        self._count = 1
        self._visited = []
        self._numbering = {}
        self._pdbqt_buffer = []
        self._atom_counter = {}

        self.mol = mol
        self.model = mol.setup.flexibility_model
        # get a copy of the current setup, since it's going to be messed up by the hacks for legacy, D3R, etc...
        self.setup = mol.setup.copy()

        root = self.model['root']
        torsdof = len(self.model['rigid_body_graph']) - 1

        # Make sure the atom types are correct
        self._fix_atom_types()

        if 'torsions_org' in self.model:
            torsdof_org = self.model['torsions_org']
            self._pdbqt_buffer.append('REMARK Flexibility Score: %2.2f' % self.model['score'] )
            for bond_id, data in list(self.setup.ring_bond_breakable.items()):
                if data['active'] == True:
                    self._pdbqt_buffer.append('REMARK Glue-bond: [% 2d ] :: [% 2d ]' % (bond_id[0], bond_id[1]) )
            self._pdbqt_buffer.append('REMARK Active torsions [% 2d ] -> [% 2d ]' % (torsdof_org, torsdof) )
            active_tors = torsdof_org
        else:
            active_tors = torsdof

        self._walk_graph_recursive(root, first=True)

        # torsdof is always going to be the one of the rigid, non-macrocyclic one
        self._pdbqt_buffer.append('TORSDOF %d\n' % active_tors)

        return '\n'.join(self._pdbqt_buffer)
