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
    def __init__(self):
        """ """
        pass

    def process_mol(self, mol, freeze_bonds=None):
        """ """
        # self.flexibility_models = []
        self.mol = mol
        self.flexibility_models = {}
        self._frozen_bonds = []
        if not freeze_bonds is None:
            self._frozen_bonds = freeze_bonds[:]
        # build rigid body graph
        self.build_flexible_models()
        # clean up
        del self.mol
        del self._frozen_bonds
        del self.flexibility_models

    def build_flexible_models(self):
        """
        build flexibility models for all configurations of the ligand (standard,
        and all macrocycle opening configurations)
        """
        self.build_flexible_model_standard()
        self.build_flexible_model_open_rings()
        self.select_best_model()

    def build_flexible_model_standard(self):
        """
        build flexibility model for unmodified (no macrocycle) standard molecular model
        """
        model = self.build_rigid_body_connectivity()
        model = self.find_graph_root(model)
        self.add_flex_model(model, score=False)

    def build_flexible_model_open_rings(self):
        """
        generate the matrix of all breakable bonds combinations, then
        build the flexibility models of the open ring molecules
        and score them
        """
        # build breakable bonds matrix
        self.build_breakable_bond_matrix()
        for breakable_bond_vector in self.breakable_bond_matrix:
            bond_tuple, rings, score = breakable_bond_vector #[2]
            model = self.build_rigid_body_connectivity(bonds_to_break=breakable_bond_vector)
            model = self.find_graph_root(model)
            self.add_flex_model(model, score=True, initial_score=score)

    def select_best_model(self):
        """
        select flexibility model with best complexity score
        """
        if len(self.flexibility_models) == 1:
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
        self.mol.setup = setup
        best_model['torsions_org'] = self.flexibility_models[0]['torsions']

        self.mol.setup.flexibility_model = best_model

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
            self._current_setup = self.mol.setup
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
                'neigh_13_14' : neigh_13_14,
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
        setup = self.mol.setup.copy()
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
                if bond_item['type'] > 0 and bond_item['bond_order'] == 1:
                    setup.bond[bond_id]['rotatable'] = True

        return setup

    def find_graph_root(self, model):
        """ TODO this has to be made aware of the weight of the groups left
         (see 1jff:TA1)
        """
        graph = deepcopy(model['rigid_body_graph'])
        max_depth = 0
        torsions = 0
        while len(graph) > 2:
            max_depth += 1
            leaves = []
            for vertex, edges in list(graph.items()):
                if len(edges) == 1:
                    leaves.append(vertex)
                    torsions += 1
            for l in leaves:
                for vertex, edges in list(graph.items()):
                    if l in edges:
                        edges.remove(l)
                        graph[vertex] = edges
                del graph[l]

        if len(graph) == 1:
            model['root'] = list(graph.keys())[0]
        else:
            r1, r2 = list(graph.keys())
            r1_size = len(model['rigid_body_members'][r1])
            r2_size = len(model['rigid_body_members'][r2])
            if r1_size >= r2_size:
                model['root'] = r1
            else:
                model['root'] = r2
            torsions += 1
            max_depth += 1
        model['graph_depth'] = max_depth
        model['torsions'] = torsions

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
        breakable_bonds = self.mol.setup.ring_bond_breakable
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
        bond_id_list, ring_id_list, score_list = bonds_to_break
        closure_count = 0

        for bond_id in bond_id_list:
            bond_data = self._current_setup.ring_bond_breakable[bond_id]
            neigh_13_14.append(bond_data['neigh_13_14'])
            # TODO this part should simply transfer the closure information in the
            # mol.setuo.closure list, and when generating the atom types for the closure
            # the appropriate potentials will be updated (internally)
            # TODO also, the atom type should not be set here
            broken_bonds.append(bond_id)
            for pseudo in bond_data['closure_pseudo']:
                # print "GGG", pseudo
                pseudo['atom_type'] = "%s%d" % (pseudo['atom_type'], closure_count)
                p_idx = self._current_setup.add_pseudo(**pseudo)
            closure_count += 1

        return neigh_13_14, broken_bonds

    def walk_rigid_body_graph(self, start):
        """ recursive walk to build the graph of rigid bodies"""
        idx = 0
        rigid = [start]
        self._visited[start] = True
        current_rigid_body_count = self._rigid_body_count
        while idx < len(rigid):
            current = rigid[idx]
            for neigh in self._current_setup.get_neigh(current):
                if self._visited[neigh]:
                    continue
                bond_id = self._current_setup.get_bond_id(current, neigh)
                bond_info = self._current_setup.get_bond(current,neigh)
                if bond_info['rotatable'] and (not bond_id in self._frozen_bonds):
                    self._rigid_body_count+=1
                    self._rigid_body_connectivity[current_rigid_body_count, self._rigid_body_count] = current, neigh
                    self._rigid_body_connectivity[self._rigid_body_count, current_rigid_body_count] = neigh, current
                    self._rigid_body_graph[current_rigid_body_count].append(self._rigid_body_count)
                    self._rigid_body_graph[self._rigid_body_count].append(current_rigid_body_count)
                    self.walk_rigid_body_graph(neigh)
                else:
                    rigid.append(neigh)
                    self._visited[neigh] = True
            idx += 1
        self._rigid_body_members[current_rigid_body_count] = rigid
