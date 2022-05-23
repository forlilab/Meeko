#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
# collection of useful snippets of code that's used frequently
#

import os
import importlib
from operator import itemgetter


mini_periodic_table = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
}


def path_module(module_name):
    try:
        from importlib import util

        specs = util.find_spec(module_name)
        if specs is not None:
            return specs.submodule_search_locations[0]
    except:
        try:
            _, path, _ = imp.find_module(module_name)
            abspath = os.path.abspath(path)
            return abspath
        except ImportError:
            return None
    return None


def getNameExt(fname):
    """extract name and extension from the input file, removing the dot
    filename.ext -> [filename, ext]
    """
    name, ext = os.path.splitext(fname)
    return name, ext[1:]  # .lower()


class HJKRingDetection:
    """Implementation of the Hanser-Jauffret-Kaufmann exhaustive ring detection
    algorithm:
        ref:
        Th. Hanser, Ph. Jauffret, and G. Kaufmann
        J. Chem. Inf. Comput. Sci. 1996, 36, 1146-1152
    """

    def __init__(self, mgraph, max_iterations=8000000):
        self.mgraph = {key: [x for x in values] for (key, values) in mgraph.items()}
        self.rings = []
        self._iterations = 0
        self._max_iterations = max_iterations
        self._is_failed = False

    def scan(self, keep_chorded_rings=False, keep_equivalent_rings=False):
        """run the full protocol for exhaustive ring detection
        by default, only chordless rings are kept, and equivalent rings removed.
        (equivalent rings are rings that have the same size and share the same
        neighbors)
        """
        self.prune()
        self.build_pgraph()
        self.vertices = self._get_sorted_vertices()
        while self.vertices:
            self._remove_vertex(self.vertices[0])
        if not keep_chorded_rings:
            self.find_chordless_rings(keep_equivalent_rings)
        output_rings = []
        for ring in self.rings:
            output_rings.append(tuple(ring[:-1]))
        return output_rings

    def _get_sorted_vertices(self):
        """function to return the vertices to be removed, sorted by increasing
        connectivity order (see paper)"""
        vertices = ((k, len(v)) for k, v in self.mgraph.items())
        return [x[0] for x in sorted(vertices, key=itemgetter(1))]

    def prune(self):
        """iteratively prune graph until there are no leafs left (nodes with only
        one connection)"""
        while True:
            prune = []
            for node, neighbors in self.mgraph.items():
                if len(neighbors) == 1:
                    prune.append((node, neighbors))
            if len(prune) == 0:
                break
            for node, neighbors in prune:
                self.mgraph.pop(node)
                for n in neighbors:
                    self.mgraph[n].remove(node)

    def build_pgraph(self, prune=True):
        """convert the M-graph (molecular graph) into the P-graph (path/bond graph)"""
        self.pgraph = []
        for node, neigh in self.mgraph.items():
            for n in neigh:
                # use sets for unique id
                edge = set((node, n))
                if not edge in self.pgraph:
                    self.pgraph.append(edge)
        # re-convert the edges to lists because order matters in cycle detection
        self.pgraph = [list(x) for x in self.pgraph]

    def _remove_vertex(self, vertex):
        """remove a vertex and join all edges connected by that vertex (this is
        the REMOVE function from the paper)
        """
        visited = {}
        remove = []
        pool = []
        for path in self.pgraph:
            if self._has_vertex(vertex, path):
                pool.append(path)
        for i, path1 in enumerate(pool):
            for j, path2 in enumerate(pool):
                if i == j:
                    continue
                self._iterations += 1
                if self._iterations > self._max_iterations:
                    self._is_failed = True
                    break
                pair_id = tuple(set((i, j)))
                if pair_id in visited:
                    continue
                visited[pair_id] = None
                common = list(set(path1) & set(path2))
                common_count = len(common)
                # check if two paths have only this vertex in common or (or
                # two, if they're a cycle)
                if not 1 <= common_count <= 2:
                    continue
                # generate the joint path
                joint_path = self._concatenate_path(path1, path2, vertex)
                is_ring = joint_path[0] == joint_path[-1]
                # if paths share more than two vertices but they're not a ring, then skip
                if (common_count == 2) and not is_ring:
                    continue
                # store the ring...
                if is_ring:
                    self._add_ring(joint_path)
                # ...or the common path
                elif not joint_path in self.pgraph:
                    self.pgraph.append(joint_path)
        # remove used paths
        for p in pool:
            self.pgraph.remove(p)
        # remove the used vertex
        self.vertices.remove(vertex)

    def _add_ring(self, ring):
        """add newly found rings to the list (if not already there)"""
        r = set(ring)
        for candidate in self.rings:
            if r == set(candidate):
                return
        self.rings.append(ring)

    def _has_vertex(self, vertex, edge):
        """check if the vertex is part of this edge, and if true, return the
        sorted edge so that the vertex is the first in the list"""
        if edge[0] == vertex:
            return edge
        if edge[-1] == vertex:
            return edge[::-1]
        return None

    def _concatenate_path(self, path1, path2, v):
        """concatenate two paths sharing a common vertex
        a-b, c-b => a-b-c : idx1=1, idx2=1
        b-a, c-b => a-b-c : idx1=0, idx2=1
        a-b, b-c => a-b-c : idx1=1, idx2=0
        b-a, b-c => a-b-c : idx1=0, idx2=0
        """
        if not path1[-1] == v:
            path1.reverse()
        if not path2[0] == v:
            path2.reverse()
        return path1 + path2[1:]

    def _edge_in_pgraph(self, edge):
        """check if edge is already in pgraph"""
        e = set(edge)
        for p in self.pgraph:
            if e == set(p) and len(p) == len(edge):
                return True
        return False

    def find_chordless_rings(self, keep_equivalent_rings):
        """find chordless rings: cycles in which two vertices are not connected
        by an edge that does not itself belong to the cycle (Source:
        https://en.wikipedia.org/wiki/Cycle_%28graph_theory%29#Chordless_cycle)

        - iterate through rings starting from the smallest ones: A,B,C,D...
        - for each ring (A), find a candidate (e.g.: B) that is smaller and shares at least an edge
        - for this pair, calculate the two differences (A-B and B-A) in the list of edges of each
        - if  ( (A-B) + (B-A) ) a smaller ring (e.g.: C), then the current ring has a chord
        """
        # sort rings by the smallest to largest
        self.rings.sort(key=len, reverse=False)
        chordless_rings = []
        ring_edges = []
        rings_set = [set(x) for x in self.rings]
        for r in self.rings:
            edges = []
            for i in range(len(r) - 1):
                edges.append(
                    tuple(
                        set((r[i], r[(i + 1) % len(r)])),
                    )
                )
            edges = sorted(edges, key=itemgetter(0))
            ring_edges.append(edges)
        ring_contacts = {}
        for i, r1 in enumerate(self.rings):
            chordless = True
            r1_edges = ring_edges[i]
            ring_contacts[i] = []
            for j, r2 in enumerate(self.rings):
                if i == j:
                    continue
                if len(r2) >= len(r1):
                    # the candidate ring is larger than or the same size of the candidate
                    continue
                # avoid rings that don't share at least an edge
                # shared = set(r1) & set(r2)
                r2_edges = ring_edges[j]
                shared = set(r1_edges) & set(r2_edges)
                if len(shared) < 1:
                    continue
                ring_contacts[i].append(j)
                # get edges difference (r2_edges - r1_edges)
                core_edges = [x for x in r2_edges if not x in r1_edges]
                chord = [x for x in r1_edges if not x in r2_edges]
                # combined = chord + core_edges
                ring_new = []
                for edge in chord + core_edges:
                    ring_new.append(edge[0])
                    ring_new.append(edge[1])
                ring_new = set(ring_new)
                if (ring_new in rings_set) and (len(ring_new) < len(r1) - 1):
                    chordless = False
                    break
            if chordless:
                chordless_rings.append(i)
                ring_contacts[i] = set(ring_contacts[i])
        if not keep_equivalent_rings:
            chordless_rings = self._remove_equivalent_rings(
                chordless_rings, ring_contacts
            )
        self.rings = [self.rings[x] for x in chordless_rings]
        return

    def _remove_equivalent_rings(self, chordless_rings, ring_contacts):
        """remove equivalent rings by clustering by size, then by ring neighbors.
        Two rings A and B are equivalent if satisfy the following conditions:
            - same size
            - same neighbor ring(s) [C,D, ...]
            - (A - C) == (B -C)
        """
        size_clusters = {}
        # cluster rings by their size
        for ring_id in chordless_rings:
            if len(ring_contacts[ring_id]) == 0:
                continue
            size = len(self.rings[ring_id]) - 1
            if not size in size_clusters:
                size_clusters[size] = []
            size_clusters[size].append(ring_id)
        remove = []
        # process rings of the same size
        for size, ring_pool in size_clusters.items():
            for ri in ring_pool:
                if ri in remove:
                    continue
                for rj in ring_pool:
                    if ri == rj:
                        continue
                    common_neigh = ring_contacts[ri] & ring_contacts[rj]
                    for c in common_neigh:
                        d1 = set(self.rings[ri]) - set(self.rings[c])
                        d2 = set(self.rings[rj]) - set(self.rings[c])
                        if d1 == d2:
                            remove.append(rj)
        chordless_rings = [i for i in chordless_rings if not i in set(remove)]
        # for r in set(remove):
        #    chordless_rings.remove(r)
        return chordless_rings


# def writeList(filename, inlist, mode = 'w', addNewLine = False):
#     if addNewLine: nl = "\n"
#     else: nl = ""
#     fp = open(filename, mode)
#     for i in inlist:
#         fp.write(str(i)+nl)
#     fp.close()


# def getResInfo(string):
#    """ CHAIN:RESnum -> [ "CHAIN", "RES", num ]"""
#    if ':' in string:
#        chain, resraw = string.split(':')
#    else:
#        chain = ''
#        resraw = string
#    try:
#        res = resraw[0:3]
#        num = int(resraw[3:])
#    except:
#        # heuristic for nucleic acids
#        regex = r'[UACGT]+\d'
#        match = re.search(regex, resraw)
#        if match is None:
#            print("WARNING! Unknown residue naming scheme")
#            return chain, "X", "X"
#        res = resraw[0]
#        num = int(resraw[1:])
#        #print "NUCLEIC:",  chain, res, num
#    return chain, res, num


# def get_data_file(file_handle, dir_name, data_file):
#     module_dir, module_fname = os.path.split(file_handle)
#     DATAPATH = os.path.join(module_dir, dir_name, data_file)
#     return DATAPATH
