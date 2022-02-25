#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Raccoon
# collection of useful snippets of code that's used frequently
#

import os
import re
import sys
import importlib
from operator import itemgetter


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
    """ extract name and extension from the input file, removing the dot
        filename.ext -> [filename, ext]
    """
    name, ext = os.path.splitext(fname)
    return name, ext[1:] #.lower()


class HJKRingDetection(object):
    """Implementation of the Hanser-Jauffret-Kaufmann exhaustive ring detection
    algorithm:
        ref:
        Th. Hanser, Ph. Jauffret, and G. Kaufmann
        J. Chem. Inf. Comput. Sci. 1996, 36, 1146-1152
    """

    def __init__(self, mgraph):
        self.mgraph = {key: [x for x in values] for (key, values) in mgraph.items()}
        self.rings = []
        self._iterations = 0

    def scan(self):
        """run the full protocol for exhaustive ring detection"""
        self.prune()
        self.build_pgraph()
        self.vertices = self._get_sorted_vertices()
        while self.vertices:
            self._remove_vertex(self.vertices[0])
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
        """iteratively prune graph until there are no nodes with only one
        connection"""
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


# def writeList(filename, inlist, mode = 'w', addNewLine = False):
#     if addNewLine: nl = "\n"
#     else: nl = ""
#     fp = open(filename, mode)
#     for i in inlist:
#         fp.write(str(i)+nl)
#     fp.close()


#def getResInfo(string):
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

