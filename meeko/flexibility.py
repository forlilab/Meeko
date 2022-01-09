#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Meeko flexibility typer
#

from copy import deepcopy
from collections import defaultdict
from itertools import product
from operator import itemgetter


class FlexibilityBuilder:

    def __call__(self, setup, freeze_bonds=None, root_atom_index=None):
        """ """
        # self.flexibility_models = []
        self.setup = setup
        self.flexibility_models = {}
        self._frozen_bonds = []
        if not freeze_bonds is None:
            self._frozen_bonds = freeze_bonds[:]
        # build rigid body graph
        #self.build_flexible_model_standard() # always needed for relative score in score_model
        # build graph for standard molecule (no open macrocycle rings)
        model = self.build_rigid_body_connectivity()
        model = self.set_graph_root(model, root_atom_index) # finds root if root_atom_index==None
        self.add_flex_model(model, score=False)

        # evaluate possible graphs for various ring openings
        #self.build_flexible_model_open_rings()
        self.build_breakable_bond_matrix()
        for breakable_bond_vector in self.breakable_bond_matrix:
            bond_tuple, rings, score = breakable_bond_vector #[2]
            model = self.build_rigid_body_connectivity(bonds_to_break=breakable_bond_vector)
            self.set_graph_root(model, root_atom_index) # finds root if root_atom_index==None
            self.add_flex_model(model, score=True, initial_score=score)

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

    def build_rigid_body_connectivity(self, bonds_to_break=None):
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
            neigh_13_14 = []
            broken_bonds = []
        else:
            # print "BREAKING BONDS!", bonds_to_break
            self._current_setup = self.copy_setup(bonds_to_break)
            # collect 1,3 and 1,4 interactions and generate pseudo-atoms if necessary
            # TODO should we have a function here to populate 1,3 and 1,4 interactions
            # even for regular dockings?
            neigh_13_14,broken_bonds = self.update_closure_atoms(bonds_to_break)
        # walk the mol graph to build the rigid body maps
        self._visited = defaultdict(lambda:False)
        self._rigid_body_members = {}
        self._rigid_body_connectivity = {}
        self._rigid_body_graph = defaultdict(list)
        self._rigid_index_by_atom = {}
        # START VALUE HERE SHOULD BE MADE MODIFIABLE FOR FLEX CHAIN
        self._rigid_body_count = 0
        self.walk_rigid_body_graph(start=1)
        # if only a rigid body is found
        if len(self._rigid_body_members) == 1:
            self._rigid_body_connectivity[0] = [0]
            self._rigid_body_graph[0] = [0]
        model = {'rigid_body_graph' : deepcopy(self._rigid_body_graph),
                'rigid_body_connectivity' : deepcopy(self._rigid_body_connectivity),
                'rigid_body_members' : deepcopy(self._rigid_body_members),
                #'neigh_13_14' : neigh_13_14,
                'setup' : self._current_setup,
                'broken_bonds' : broken_bonds,
                }
        return model

    def copy_setup(self, bonds_to_break):
        """ copy connectivity information (graph and bonds) from the setup,
            optionally delete bond_id listed in bonds_to_break,
            updating connectivty information
        """
        # TODO check to re-enable torsions that might be fred by
        #      bond breaking
        # TODO This should be well-coordinated to prevent
        # that bonds connected to closure atoms are made rotatable
        # when no hydrogens are attached to them (one rotation matrix less...)
        setup = self.setup.copy()
        deleted_bonds = []
        bond_id_list = bonds_to_break[0]
        rings_to_enable = bonds_to_break[1]
        score = bonds_to_break[2]
        for bond_id in bond_id_list:
            setup.ring_bond_breakable[bond_id]['active'] = True
            setup.del_bond(*bond_id)
            deleted_bonds.append(bond_id)
        # free potentially rotatable bonds that are in the ring
        for ring_members in rings_to_enable:
            ring_size = len(ring_members)
            for idx in range(ring_size):
                atom_idx1 = ring_members[idx % ring_size]
                atom_idx2 = ring_members[(idx + 1) % ring_size]
                bond_id = setup.get_bond_id(atom_idx1, atom_idx2)
                if bond_id in bond_id_list:
                    # trying to delete the bond that has been broken
                    continue
                # TODO coordinate this with the bond type rules
                bond_item = setup.get_bond(*bond_id)
                if len(bond_item['in_rings']) > 1:
                    continue
                if bond_item['bond_order'] == 1: # not amide, amidine, aromatic
                    setup.bond[bond_id]['rotatable'] = True

        return setup


    def calc_max_depth(self, graph, seed_node, visited=[], depth=0):
        maxdepth = depth 
        newdepth = 0
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
            #max_depth = 0
            while len(graph) > 2: # remove leafs until 1 or 2 rigid groups remain
                #max_depth += 1
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

            if len(graph) == 1:
                root_body_index = list(graph.keys())[0]
            else:
                r1, r2 = list(graph.keys())
                r1_size = len(model['rigid_body_members'][r1])
                r2_size = len(model['rigid_body_members'][r2])
                if r1_size >= r2_size:
                    root_body_index = r1
                else:
                    root_body_index = r2
                #max_depth += 1

        else: # find index of rigid group
            for body_index in model['rigid_body_members']:
                if root_atom_index in model['rigid_body_members'][body_index]: # 1-index atoms
                    root_body_index = body_index

        model['root'] = root_body_index
        model['torsions'] = len(model['rigid_body_members']) - 1
        model['graph_depth'] = self.calc_max_depth(model['rigid_body_graph'], root_body_index, visited=[], depth=0)

        return model

    def build_breakable_bond_matrix(self):
        """
            ring_1 ={ 0:10, 1:9, 2:10, 3:9}
            ring_2={ 4:9, 5:8, 6:4}

        built the tuple of all possible combinations of breakable bonds for all the rings
        with their breakable score:
            (r1[0], r2[0], ... , rN[0]) : score1
            (r1[1], r2[0], ... , rN[0]) : score2
             ...     ...   ...   ...
            (r1[99], r2[99], ... , rN[99) : scoreMM
        """
        self.breakable_bond_matrix = []
        breakable_bonds = self.setup.ring_bond_breakable
        if len(breakable_bonds) == 0:
            return
        ring_bonds = defaultdict(list)
        for bond_id, data in list(breakable_bonds.items()):
            ring_id = data['ring_id']
            score = data['score']
            ring_bonds[ring_id].append((bond_id,ring_id, score))
        # enumerate all possible combinations of breakable rings for each ring
        breakable_bond_raw = list(product(*list(ring_bonds.values())))
        for vect in breakable_bond_raw:
            bond_list = []
            ring_list = []
            total_score = 0
            for bond in vect:
                bond_id = bond[0]
                ring_id = bond[1]
                score = bond[2]
                bond_list.append(bond_id)
                ring_list.append(ring_id)
                total_score += score
            total_score = float(total_score) / len(bond_list)
            ring_list = list(set(ring_list))
            self.breakable_bond_matrix.append((bond_list, ring_list,total_score))

    def score_flex_model(self, model):
        """ score a flexibility model basing on the graph properties"""
        base = self.flexibility_models[0]['graph_depth']
        score = 10 * (base-model['graph_depth'])
        return score

    def update_closure_atoms(self, bonds_to_break):
        """ create pseudoatoms required by the flexible model with broken bonds"""
        if bonds_to_break is None:
            return [], []

        neigh_13_14 = []
        broken_bonds = []
        # counter to number glue atoms, G1-G1, G2-G2, ...
        bond_id_list, _, _ = bonds_to_break

        for i, bond_id in enumerate(bond_id_list):
            bond_data = self._current_setup.ring_bond_breakable[bond_id]
            neigh_13_14.append(bond_data['neigh_13_14'])
            # TODO this part should simply transfer the closure information in the
            # mol.setuo.closure list, and when generating the atom types for the closure
            # the appropriate potentials will be updated (internally)
            # TODO also, the atom type should not be set here
            broken_bonds.append(bond_id)
            # Add pseudo glue atoms
            for pseudo in bond_data['closure_pseudo']:
                pseudo['atom_type'] = "%s%d" % (pseudo['atom_type'], i)
                p_idx = self._current_setup.add_pseudo(**pseudo)
            # Change atom type of the closure atoms to CGX
            self._current_setup.set_atom_type(bond_id[0], "CG%d" % i)
            self._current_setup.set_atom_type(bond_id[1], "CG%d" % i)

        return neigh_13_14, broken_bonds

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

