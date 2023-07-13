#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Reactive
#

from math import sqrt
from os import linesep as os_linesep


class ReactiveAtomTyper:

    def __init__(self):

        self.ff = {
            "HD": {"rii": 2.00, "epsii": 0.020},
            "C":  {"rii": 4.00, "epsii": 0.150},
            "A":  {"rii": 4.00, "epsii": 0.150},
            "N":  {"rii": 3.50, "epsii": 0.160},
            "NA": {"rii": 3.50, "epsii": 0.160},
            "OA": {"rii": 3.20, "epsii": 0.200},
            "OS": {"rii": 3.20, "epsii": 0.200},
            "F":  {"rii": 3.09, "epsii": 0.080},
            "P":  {"rii": 4.20, "epsii": 0.200},
            "SA": {"rii": 4.00, "epsii": 0.200},
            "S":  {"rii": 4.00, "epsii": 0.200},
            "Cl": {"rii": 4.09, "epsii": 0.276},
            "CL": {"rii": 4.09, "epsii": 0.276},
            "Br": {"rii": 4.33, "epsii": 0.389},
            "BR": {"rii": 4.33, "epsii": 0.389},
            "I":  {"rii": 4.72, "epsii": 0.550},
            "Si": {"rii": 4.10, "epsii": 0.200},
            "B":  {"rii": 3.84, "epsii": 0.155},
            "W":  {"rii": 0.00, "epsii": 0.000},
        }
        std_atypes = list(self.ff.keys())
        rt, r2s, r2o = self.enumerate_reactive_types(std_atypes)
        self.reactive_type = rt
        self.reactive_to_std_atype_mapping = r2s
        self.reactive_to_order = r2o

    @staticmethod
    def enumerate_reactive_types(atypes):
        reactive_type = {1:{}, 2:{}, 3:{}}
        reactive_to_std_atype_mapping = {}
        reactive_to_order = {}
        for order in (1,2,3):
            for atype in atypes:
                if len(atype) == 1:
                    new_atype = "%s%d" % (atype, order)
                else:
                    new_atype = "%s%d" % (atype[0], order+3)
                    if new_atype in reactive_to_std_atype_mapping:
                        new_atype = "%s%d" % (atype[0], order+6)
                        if new_atype in reactive_to_std_atype_mapping:
                            raise RuntimeError("ran out of numbers for reactive types :(")
                reactive_to_std_atype_mapping[new_atype] = atype
                reactive_to_order[new_atype] = order
                reactive_type[order][atype] = new_atype
                ### # avoid atom type clashes with multiple reactive residues by
                ### # prefixing with the index of the residue, e.g. C3 -> 1C3.
                ### for i in range(8): # hopefully 8 reactive residues is sufficient
                ###     prefixed_new_atype = '%d%s' % ((i+1), new_atype)
                ###     reactive_to_std_atype_mapping[prefixed_new_atype] = atype
        return reactive_type, reactive_to_std_atype_mapping, reactive_to_order


    def get_scaled_parm(self, atype1, atype2):
        """ generate scaled parameters for a pairwise interaction between two atoms involved in a
            reactive interaction

            Rij and epsij are calculated according to the AD force field:
                # - To obtain the Rij value for non H-bonding atoms, calculate the
                #        arithmetic mean of the Rii values for the two atom types.
                #        Rij = (Rii + Rjj) / 2
                #
                # - To obtain the epsij value for non H-bonding atoms, calculate the
                #        geometric mean of the epsii values for the two atom types.
                #        epsij = sqrt( epsii * epsjj )
        """

        atype1_org, _ = self.get_basetype_and_order(atype1)
        atype2_org, _ = self.get_basetype_and_order(atype2)
        atype1_rii = self.ff[atype1_org]['rii']
        atype1_epsii = self.ff[atype1_org]['epsii']
        atype2_rii = self.ff[atype2_org]['rii']
        atype2_epsii = self.ff[atype2_org]['epsii']
        atype1_rii = atype1_rii
        atype2_rii = atype2_rii
        rij = (atype1_rii + atype2_rii) / 2
        epsij = sqrt(atype1_epsii * atype2_epsii)
        return rij, epsij


    def get_reactive_atype(self, atype, reactive_order):
        """ create or retrive new reactive atom type label name"""
        macrocycle_glue_types = ["CG%d" % i for i in range(9)]
        macrocycle_glue_types += ["G%d" % i for i in range(9)]
        if atype in macrocycle_glue_types:
            return None
        return self.reactive_type[reactive_order][atype]


    def get_basetype_and_order(self, atype):
        if len(atype) > 1:
            if atype[0].isdecimal():
                atype = atype[1:] # reactive residues are prefixed with a digit
        if atype not in self.reactive_to_std_atype_mapping:
            return None, None
        basetype = self.reactive_to_std_atype_mapping[atype]
        order = self.reactive_to_order[atype]
        return basetype, order


reactive_typer = ReactiveAtomTyper()
get_reactive_atype = reactive_typer.get_reactive_atype

def assign_reactive_types(molsetup, smarts, smarts_idx, get_reactive_atype=get_reactive_atype):

    atype_dicts = []
    for atom_indices in molsetup.find_pattern(smarts):
        atypes = molsetup.atom_type.copy()
        reactive_atom_index = atom_indices[smarts_idx]

        # type reactive atom
        original_type = molsetup.atom_type[reactive_atom_index]
        reactive_type = get_reactive_atype(original_type, reactive_order=1) 
        atypes[reactive_atom_index] = reactive_type

        # type atoms 1 bond away from reactive atom
        for index1 in molsetup.graph[reactive_atom_index]:
            if not molsetup.atom_ignore[index1]:
                original_type = molsetup.atom_type[index1]
                reactive_type = get_reactive_atype(original_type, reactive_order=2)
                atypes[index1] = reactive_type

            # type atoms 2 bonds away from reactive
            for index2 in molsetup.graph[index1]:
                if index2 == reactive_atom_index:
                    continue
                if not molsetup.atom_ignore[index2]:
                    original_type = molsetup.atom_type[index2]
                    reactive_type = get_reactive_atype(original_type, reactive_order=3)
                    atypes[index2] = reactive_type

        atype_dicts.append(atypes)

    return atype_dicts


# enumerate atom type pair combinations for reactive docking configuration file

def get_reactive_config(types_1, types_2, eps12, r12, r13_scaling, r14_scaling, ignore=['HD', 'F'], coeff_vdw=.1662):
    """
    Args:
        types_1 (list): 1st set of atom types
        types_2 (list): 2nd set of atom types

    Returns:
        derivtypes (dict):
        modpairs (list):
    """

    # for reactive pair (1-2 interaction) use 13-7 potential
    n12 = 13
    m12 = 7

    # for 1-3 and 1-4 interactions stick to 12-6 potential
    n = 12
    m = 6

    derivtypes = {}
    for reactype in set(types_1).union(set(types_2)):
        basetype, order = reactive_typer.get_basetype_and_order(reactype)
        derivtypes.setdefault(basetype, [])
        derivtypes[basetype].append(reactype)

    scaling = {3: r13_scaling, 4: r14_scaling}

    modpairs = {}
    for t1 in set(types_1):
        basetype_1, order_1 = reactive_typer.get_basetype_and_order(t1)
        for t2 in set(types_2):
            basetype_2, order_2 = reactive_typer.get_basetype_and_order(t2)
            order = order_1 + order_2
            pair_id = tuple(sorted([t1, t2]))
            if order == 2: # 1-2 interaction: these are the reactive atoms.
                modpairs[pair_id] = {"eps": eps12, "r_eq": r12, "n": n12, "m": m12}
            elif order == 3 or order == 4:
                if basetype_1 in ignore or basetype_2 in ignore:
                    rij = 0.01
                    epsij = 0.001
                else:
                    rij, epsij = reactive_typer.get_scaled_parm(t1, t2)
                    rij *= scaling[order]
                    epsij *= coeff_vdw
                modpairs[pair_id] = {"eps": epsij, "r_eq": rij, "n": n, "m": m}

    # pairs of types across sets that also happen within each set
    def enum_pairs(types):
        pairs = set()
        for i in range(len(types_1)):
            for j in range(i, len(types_1)):
                pair_id = tuple(sorted([types_1[i], types_1[j]]))
                pairs.add(pair_id)
        return pairs
    collisions = []
    collisions.extend([p for p in enum_pairs(types_1) if p in modpairs])
    collisions.extend([p for p in enum_pairs(types_2) if p in modpairs])
    collisions = set(collisions)

    return derivtypes, modpairs, collisions
