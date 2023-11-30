#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko flexibility typer
#

from copy import deepcopy
from collections import defaultdict
from itertools import product
from operator import itemgetter
from .utils import pdbutils


class FlexibilityBuilder:

    def __call__(self, setup, freeze_bonds=None, root_atom_index=None, break_combo_data=None, bonds_in_rigid_rings=None, glue_pseudo_atoms=None):
        """ """
        self.setup = setup
        self.flexibility_models = {}
        self._frozen_bonds = []
        if not freeze_bonds is None:
            self._frozen_bonds = freeze_bonds[:]
        # build graph for standard molecule (no open macrocycle rings)
        model = self.build_rigid_body_connectivity()
        model = self.set_graph_root(model, root_atom_index) # finds root if root_atom_index==None
        self.add_flex_model(model, score=False)

        # evaluate possible graphs for various ring openings
        if break_combo_data is not None:
            bond_break_combos = break_combo_data['bond_break_combos']
            bond_break_scores = break_combo_data['bond_break_scores']
            broken_rings_list = break_combo_data['broken_rings']
            for index in range(len(bond_break_combos)):
                bond_break_combo = bond_break_combos[index]
                bond_break_score = bond_break_scores[index]
                broken_rings =     broken_rings_list[index]
                model = self.build_rigid_body_connectivity(bond_break_combo, broken_rings, bonds_in_rigid_rings, glue_pseudo_atoms)
                self.set_graph_root(model, root_atom_index) # finds root if root_atom_index==None
                self.add_flex_model(model, score=True, initial_score=bond_break_score)

        self.select_best_model()

        # clean up
        del self._frozen_bonds
        del self.flexibility_models
        return self.setup


    def select_best_model(self):
        """
        select flexibility model with best complexity score
        """
        if len(self.flexibility_models) == 1: # no macrocyle open rings
            best_model = list(self.flexibility_models.values())[0]
        else:
            score_sorted_models = []
            for m_id, model in list(self.flexibility_models.items()):
                score_sorted_models.append((m_id, model['score']))
            #print("SORTED", score_sorted_models)
            score_sorted = sorted(score_sorted_models, key=itemgetter(1), reverse=True)
            #for model_id, score in score_sorted:
            #    print("ModelId[% 3d] score: %2.2f" % (model_id, score))
            # the 0-model is the rigid model used as reference
            best_model_id, best_model_score = score_sorted[1]
            best_model = self.flexibility_models[best_model_id]
        
        setup = best_model['setup']
        del best_model['setup']
        self.setup = setup
        best_model['torsions_org'] = self.flexibility_models[0]['torsions']

        self.setup.flexibility_model = best_model

    def add_flex_model(self, model, score=False, initial_score=0):
        """ add a flexible model to the list of configurations,
            and optionally score it, basing on the connectivity properties
        """

        model_id = len(self.flexibility_models)
        if score == False:
            model['score'] = float('inf')
        else:
            penalty = self.score_flex_model(model)
            model['score'] = initial_score + penalty
        self.flexibility_models[model_id] = model

    def build_rigid_body_connectivity(self, bonds_to_break=None, broken_rings=None, bonds_in_rigid_rings=None, glue_pseudo_atoms=None):
        """
        rigid_body_graph is the graph of rigid bodies
        ( rigid_body_id->[rigid_body_id,...] )

        rigid_body_members contains the atom indices in each rigid_body_id,
        ( rigid_body_id->[atom,...] )

        rigid_body_connectivity contains connectivity information between
        rigid bodies, mapping a two rigid bodies to the two atoms that connect
        them
        ( (rigid_body1, rigid_body2) -> (atom1,atom2)
        """
        # make a copy of the current mol graph, updated with the broken bond
        if bonds_to_break is None:
            self._current_setup = self.setup
        else:
            self._current_setup = self.copy_setup(bonds_to_break, broken_rings, bonds_in_rigid_rings)
            self.update_closure_atoms(bonds_to_break, glue_pseudo_atoms)

        # walk the mol graph to build the rigid body maps
        self._visited = defaultdict(lambda:False)
        self._rigid_body_members = {}
        self._rigid_body_connectivity = {}
        self._rigid_body_graph = defaultdict(list)
        self._rigid_index_by_atom = {}
        # START VALUE HERE SHOULD BE MADE MODIFIABLE FOR FLEX CHAIN
        self._rigid_body_count = 0
        self.walk_rigid_body_graph(start=0)
        model = {'rigid_body_graph' : deepcopy(self._rigid_body_graph),
                'rigid_body_connectivity' : deepcopy(self._rigid_body_connectivity),
                'rigid_body_members' : deepcopy(self._rigid_body_members),
                'setup' : self._current_setup}
        return model

    def copy_setup(self, bond_list, broken_rings, bonds_in_rigid_rings):
        """ copy connectivity information (graph and bonds) from the setup,
            optionally delete bond_id listed in bonds_to_break,
            updating connectivty information
        """
        setup = self.setup.copy()
        for bond in bond_list:
            setup.del_bond(*bond)

        for ring in broken_rings:
            for bond in setup.get_bonds_in_ring(ring):
                if bond not in bonds_in_rigid_rings: # e.g. bonds in small rings do not rotata
                    if bond in bond_list:
                        continue # this bond has been deleted
                    bond_item = setup.get_bond(*bond)
                    if bond_item['bond_order'] == 1:
                        setup.bond[bond]['rotatable'] = True
        return setup


    def calc_max_depth(self, graph, seed_node, visited=[], depth=0):
        maxdepth = depth 
        visited.append(seed_node)
        for node in graph[seed_node]:
            if node not in visited:
                visited.append(node)
                newdepth = self.calc_max_depth(graph, node, visited, depth + 1)
                maxdepth = max(maxdepth, newdepth)
        return maxdepth


    def set_graph_root(self, model, root_atom_index=None):
        """ TODO this has to be made aware of the weight of the groups left
         (see 1jff:TA1)
        """

        if root_atom_index is None: # find rigid group that minimizes max_depth
            graph = deepcopy(model['rigid_body_graph'])
            while len(graph) > 2: # remove leafs until 1 or 2 rigid groups remain
                leaves = []
                for vertex, edges in list(graph.items()):
                    if len(edges) == 1:
                        leaves.append(vertex)
                for l in leaves:
                    for vertex, edges in list(graph.items()):
                        if l in edges:
                            edges.remove(l)
                            graph[vertex] = edges
                    del graph[l]

            if len(graph) == 0:
                root_body_index = 0
            elif len(graph) == 1:
                root_body_index = list(graph.keys())[0]
            else:
                r1, r2 = list(graph.keys())
                r1_size = len(model['rigid_body_members'][r1])
                r2_size = len(model['rigid_body_members'][r2])
                if r1_size >= r2_size:
                    root_body_index = r1
                else:
                    root_body_index = r2

        else: # find index of rigid group
            for body_index in model['rigid_body_members']:
                if root_atom_index in model['rigid_body_members'][body_index]: # 1-index atoms
                    root_body_index = body_index

        model['root'] = root_body_index
        model['torsions'] = len(model['rigid_body_members']) - 1
        model['graph_depth'] = self.calc_max_depth(model['rigid_body_graph'], root_body_index, visited=[], depth=0)

        return model


    def score_flex_model(self, model):
        """ score a flexibility model basing on the graph properties"""
        base = self.flexibility_models[0]['graph_depth']
        score = 10 * (base-model['graph_depth'])
        return score


    def _generate_closure_pseudo(self, setup, bond_id, coords_dict={}):
        """ calculate position and parameters of the pseudoatoms for the closure"""
        closure_pseudo = []
        for idx in (0,1):
            target = bond_id[1 - idx]
            anchor = bond_id[0 - idx]
            if coords_dict is None or len(coords_dict) == 0:
                coord = setup.get_coord(target)
            else:
                coord = coords_dict[anchor]
            anchor_info = setup.pdbinfo[anchor]
            pdbinfo = pdbutils.PDBAtomInfo('G', anchor_info.resName, anchor_info.resNum, anchor_info.chain)
            closure_pseudo.append({
                'coord': coord,
                'anchor_list': [anchor],
                'charge': 0.0,
                'pdbinfo': pdbinfo,
                'atom_type': 'G',
                'rotatable': False})
        return closure_pseudo


    def update_closure_atoms(self, bonds_to_break, coords_dict):
        """ create pseudoatoms required by the flexible model with broken bonds"""

        setup = self._current_setup
        for i, bond in enumerate(bonds_to_break):
            setup.ring_closure_info["bonds_removed"].append(bond) # bond is pair of atom indices
            pseudos = self._generate_closure_pseudo(setup, bond, coords_dict)
            for pseudo in pseudos:
                pseudo['atom_type'] = "%s%d" % (pseudo['atom_type'], i)
                pseudo_index = setup.add_pseudo(**pseudo)
                atom_index = pseudo['anchor_list'][0]
                if atom_index in setup.ring_closure_info:
                    raise RuntimeError("did not expect more than one G per atom")
                setup.ring_closure_info["pseudos_by_atom"][atom_index] = pseudo_index
            setup.set_atom_type(bond[0], "CG%d" % i)
            setup.set_atom_type(bond[1], "CG%d" % i)


    def walk_rigid_body_graph(self, start):
        """ recursive walk to build the graph of rigid bodies"""
        idx = 0
        rigid = [start]
        self._visited[start] = True
        current_rigid_body_count = self._rigid_body_count
        self._rigid_index_by_atom[start] = current_rigid_body_count
        sprouts_buffer = []
        while idx < len(rigid):
            current = rigid[idx]
            for neigh in self._current_setup.get_neigh(current):
                bond_id = self._current_setup.get_bond_id(current, neigh)
                bond_info = self._current_setup.get_bond(current, neigh)
                if self._visited[neigh]:
                    is_rigid_bond = (bond_info['rotatable'] == False) or (bond_id in self._frozen_bonds)
                    neigh_in_other_rigid_body = current_rigid_body_count != self._rigid_index_by_atom[neigh]
                    if is_rigid_bond and neigh_in_other_rigid_body:
                        raise RuntimeError('Flexible bonds within rigid group. We have a problem.')
                    continue
                if bond_info['rotatable'] and (bond_id not in self._frozen_bonds):
                    sprouts_buffer.append((current, neigh))
                else:
                    rigid.append(neigh)
                    self._rigid_index_by_atom[neigh] = current_rigid_body_count
                    self._visited[neigh] = True
            idx += 1
        self._rigid_body_members[current_rigid_body_count] = rigid
        for current, neigh in sprouts_buffer:
            if self._visited[neigh]: continue
            self._rigid_body_count+=1
            self._rigid_body_connectivity[current_rigid_body_count, self._rigid_body_count] = current, neigh
            self._rigid_body_connectivity[self._rigid_body_count, current_rigid_body_count] = neigh, current
            self._rigid_body_graph[current_rigid_body_count].append(self._rigid_body_count)
            self._rigid_body_graph[self._rigid_body_count].append(current_rigid_body_count)
            self.walk_rigid_body_graph(neigh)

