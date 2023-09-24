#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko macrocycle builder
#

import os
import sys
from collections import defaultdict
from operator import itemgetter


class FlexMacrocycle:
    def __init__(self, min_ring_size=7, max_ring_size=33, double_bond_penalty=50, max_breaks=4):
        """Initialize macrocycle typer.

        Args:
            min_ring_size (int): minimum size of the ring (default: 7)
            max_ring_size (int): maximum size of the ring (default: 33)
            double_bond_penalty (float)

        """
        self._min_ring_size = min_ring_size
        self._max_ring_size = max_ring_size
        # accept also double bonds (if nothing better is found)
        self._double_bond_penalty = double_bond_penalty
        self.max_breaks = max_breaks

        self.setup = None
        self.breakable_rings = None
        self._conj_bond_list = None

    def collect_rings(self, setup):
        """ get non-aromatic rings of desired size and
            list bonds that are part of unbreakable rings

            Bonds belonging to rigid cycles can't be deleted or
            made rotatable even if they are part of a breakable ring
        """
        breakable_rings = []
        rigid_rings = []
        for ring_id in list(setup.rings.keys()): # ring_id are the atom indices in each ring
            size = len(ring_id)
            if ring_id in setup.rings_aromatic:
                rigid_rings.append(ring_id)
            elif size < self._min_ring_size:
                rigid_rings.append(ring_id)
                # do not add rings > _max_ring_size to rigid_rings
                # because bonds in rigid rings will not be breakable 
                # and these bonds may also belong to breakable rings
            elif size <= self._max_ring_size:
                breakable_rings.append(ring_id)

        bonds_in_rigid_cycles = set()
        for ring_atom_indices in rigid_rings:
            for bond in setup.get_bonds_in_ring(ring_atom_indices):
                bonds_in_rigid_cycles.add(bond)

        return breakable_rings, bonds_in_rigid_cycles 

    def _detect_conj_bonds(self):
        """ detect bonds in conjugated systems
        """
        # TODO this should be removed once atom typing will be done
        conj_bond_list = []
        # pattern = "[R0]=[R0]-[R0]=[R0]" # Does not match conjugated bonds inside  the macrocycle?
        pattern = '*=*[*]=,#,:[*]' # from SMARTS_InteLigand.txt
        found = self.setup.find_pattern(pattern)
        for f in found:
            bond = (f[1], f[2])
            bond = (min(bond), max(bond))
            conj_bond_list.append(bond)
        return conj_bond_list

    def _score_bond(self, bond):
        """ provide a score for the likeness of the bond to be broken"""
        bond = self.setup.get_bond_id(bond[0], bond[1])
        atom_idx1, atom_idx2 = bond
        score = 100

        bond_order = self.setup.bond[bond]['bond_order']
        if bond_order not in [1, 2, 3]: # aromatic, double, made rigid explicitly (order=1.1 from --rigidify)
            return -1
        if self.setup.atom_type[atom_idx1] != "C":
            return -1
        if self.setup.atom_type[atom_idx2] != "C":
            return -1
        # triple bond tolerated but not preferred (TODO true?)
        if bond_order == 3:
            score -= 30
        elif (bond_order == 2):
            score -= self._double_bond_penalty
        if bond in self._conj_bond_list:
            score -= 30
        # discourage chiral atoms
        if self.setup.get_chiral(atom_idx1) or self.setup.get_chiral(atom_idx2):
            score -= 20
        return score

    def get_breakable_bonds(self, bonds_in_rigid_rings):
        """ find breaking points for rings
            following guidelines defined in [1]
            The optimal bond has the following properties:
            - does not involve a chiral atom
            - is not double/triple (?)
            - is between two carbons
            (preferably? we can now generate pseudoAtoms on the fly!)
            - is a bond present only in one ring

             [1] Forli, Botta, J. Chem. Inf. Model., 2007, 47 (4)
              DOI: 10.1021/ci700036j
        """
        breakable = {}
        for ring_atom_indices in self.breakable_rings:
            for bond in self.setup.get_bonds_in_ring(ring_atom_indices):
                score = self._score_bond(bond)
                if score > 0 and bond not in bonds_in_rigid_rings:
                    breakable[bond] = {'score': score}
        return breakable

    def search_macrocycle(self, setup, delete_these_bonds=[]):
        """Search for macrocycle in the molecule

        Args:
            setup : MoleculeSetup object

        """
        self.setup = setup

        self.breakable_rings, bonds_in_rigid_rings = self.collect_rings(setup)
        self._conj_bond_list = self._detect_conj_bonds()
        if len(delete_these_bonds) == 0:
            breakable_bonds = self.get_breakable_bonds(bonds_in_rigid_rings)
        else:
            breakable_bonds = {}
            for bond in delete_these_bonds:
                bond = self.setup.get_bond_id(bond[0], bond[1])
                breakable_bonds[bond] = {"score": self._score_bond(bond)}
        break_combo_data = self.combinatorial_break_search(breakable_bonds)
        return break_combo_data, bonds_in_rigid_rings

    def combinatorial_break_search(self, breakable_bonds):
        """ enumerate all combinations of broken bonds
            once a bond is broken, it will break one or more rings
            subsequent bonds will be pulled from intact (unbroken) rings
            
            the number of broken bonds may be variable
            returns only combinations of broken bonds that break the max number of broken bonds
        """

        max_breaks = self.max_breaks
        break_combos = self._recursive_break(self.breakable_rings, max_breaks, breakable_bonds, set(), [])
        break_combos = list(break_combos) # convert from set
        max_broken_bonds = 0
        output_break_combos = [] # found new max, discard prior data
        output_bond_scores = []
        output_broken_rings = []
        for broken_bonds in break_combos:
            n_broken_bonds = len(broken_bonds)
            bond_score = sum([breakable_bonds[bond]['score'] for bond in broken_bonds])
            broken_rings = self.get_broken_rings(self.breakable_rings, broken_bonds)
            if n_broken_bonds > max_broken_bonds:
                max_broken_bonds = n_broken_bonds
                output_break_combos = [] # found new max, discard prior data
                output_bond_scores = []
                output_broken_rings = []
            if n_broken_bonds == max_broken_bonds:
                output_break_combos.append(broken_bonds)
                output_bond_scores.append(bond_score)
                output_broken_rings.append(broken_rings)
        break_combo_data = {"bond_break_combos": output_break_combos,
                            "bond_break_scores": output_bond_scores,
                            "broken_rings": output_broken_rings}
        return break_combo_data


    def _recursive_break(self, rings, max_breaks, breakable_bonds, output=set(), broken_bonds=[]):
        if max_breaks == 0:
            return output
        unbroken_rings = self.get_unbroken_rings(rings, broken_bonds)
        atoms_in_broken_bonds = atoms_in_broken_bonds = [atom_idx for bond in broken_bonds for atom_idx in bond]
        for bond in breakable_bonds:
            if bond[0] in atoms_in_broken_bonds or bond[1] in atoms_in_broken_bonds:
                continue # each atom can be in only one broken bond
            is_bond_in_ring = False
            for ring in unbroken_rings:
                if bond in self.setup.get_bonds_in_ring(ring):
                    is_bond_in_ring = True
                    break
            if is_bond_in_ring:
                current_broken_bonds = [(a, b) for (a, b) in broken_bonds + [bond]]
                num_unbroken_rings = len(self.get_unbroken_rings(rings, current_broken_bonds))
                data_row = tuple(sorted([(a, b) for (a, b) in current_broken_bonds]))
                output.add(data_row)
                if num_unbroken_rings > 0:
                    output = self._recursive_break(rings, max_breaks-1, breakable_bonds,
                                                   output, current_broken_bonds)
        return output

    
    def get_unbroken_rings(self, rings, broken_bonds):
        unbroken = []
        for ring in rings:
            is_unbroken = True
            for bond in broken_bonds:
                if bond in self.setup.get_bonds_in_ring(ring): # consider precalculating bonds
                    is_unbroken = False
                    break # pun intended
            if is_unbroken:
                unbroken.append(ring)        
        return unbroken

    def get_broken_rings(self, rings, broken_bonds):
        broken_rings = []
        for ring in rings:
            is_broken = False
            for bond in broken_bonds:
                if bond in self.setup.get_bonds_in_ring(ring): # consider precalculating bonds
                    is_broken = True
                    break # pun intended
            if is_broken:
                broken_rings.append(ring)        
        return broken_rings


    def show_macrocycle_scores(self, setup):
        print("Warning: not showing macrocycle scores, check implementation.")
        return
        if setup is not None:
            print("\n==============[ MACROCYCLE SCORES ]================")
            bond_by_ring = defaultdict(list)

            for bond_id, data in list(setup.ring_bond_breakable.items()):
                ring_id = data['ring_id']
                bond_by_ring[ring_id].append(bond_id)

            for ring_id, bonds in list(bond_by_ring.items()):
                data = []
                print("-----------[ ring id: %s | size: %2d ]-----------" % (",".join([str(x) for x in ring_id]), len(ring_id)))

                for b in bonds:
                    score = setup.ring_bond_breakable[b]['score']
                    data.append((b, score))

                data = sorted(data, key=itemgetter(1), reverse=True)

                for b_count, b in enumerate(data):
                    begin = b[0][0]
                    end = b[0][1]
                    # bond = self.mol.setup.get_bond(b[0][0], b[0][1])
                    # begin = bond.GetBeginAtomIdx()
                    # end = bond.GetEndAtomIdx()
                    info = (b_count, begin, end, b[1], "#" * int(b[1] / 5), "-" * int(20 - b[1] / 5))
                    print("[ %2d] Bond [%3d --%3d] s:%3d [%s%s]" % info)

