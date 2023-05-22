#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Reactive
#

from math import sqrt


class ReactiveAtomTyper:

    def __init__(self):

        self.FE_coeff_vdW = 0.1662
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
        rt, r2s = self.enumerate_reactive_types(std_atypes)
        self.reactive_type = rt
        self.reactive_to_std_atype_mapping = r2s

    @staticmethod
    def enumerate_reactive_types(atypes):
        reactive_type = {1:{}, 2:{}, 3:{}}
        reactive_to_std_atype_mapping = {}
        for order in (1,2,3):
            for atype in atypes:
                if len(atype) == 1:
                    new_atype = "%s%d" % (atype, order)
                else:
                    new_atype = "%s%d" % (atype[0], order+3)
                    if new_atype in reactive_to_std_atype_mapping:
                        new_atype = "%s%d" % (atype[0], order+6)
                reactive_to_std_atype_mapping[new_atype] = atype
                reactive_type[order][atype] = new_atype
                ### # avoid atom type clashes with multiple reactive residues by
                ### # prefixing with the index of the residue, e.g. C3 -> 1C3.
                ### for i in range(8): # hopefully 8 reactive residues is sufficient
                ###     prefixed_new_atype = '%d%s' % ((i+1), new_atype)
                ###     reactive_to_std_atype_mapping[prefixed_new_atype] = atype
        return reactive_type, reactive_to_std_atype_mapping


    def get_scaled_parm(self, atype1, atype2, r_scaling=1.0, ignore=['HD', 'F'], apply_vdw_coeff=True):
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
        if apply_vdw_coeff is True:
            vdw_coeff = self.FE_coeff_vdW
        else:
            vdw_coeff = 1.0
        atype1_org = self.reactive_to_std_atype_mapping[atype1]
        atype2_org = self.reactive_to_std_atype_mapping[atype2]
        if (atype1_org in ignore) or (atype2_org in ignore):
            rij = 0.01
            epsij = 0.001
        else:
            atype1_rii = self.ff[atype1_org]['rii']
            atype1_epsii = self.ff[atype1_org]['epsii']
            atype2_rii = self.ff[atype2_org]['rii']
            atype2_epsii = self.ff[atype2_org]['epsii']
            atype1_rii = atype1_rii
            atype2_rii = atype2_rii
            rij = r_scaling * ( atype1_rii + atype2_rii) / 2
            epsij = sqrt( atype1_epsii * atype2_epsii) * vdw_coeff
        return rij, epsij


    def get_reactive_atype(self, atype, reactive_order):
        """ create or retrive new reactive atom type label name"""
        if atype in ['CG0', 'CG1', 'CG2', 'CG3', "CG4", "CG5", "CG6", "CG7", "CG8"]:
            return None
        return self.reactive_type[reactive_order][atype]


reactive_typer = ReactiveAtomTyper()
get_reactive_atype = reactive_typer.get_reactive_atype

def assign_reactive_types(molsetup, smarts, smarts_idx, get_reactive_atype=get_reactive_atype):

    atype_dicts = []
    for atom_indices in molsetup.find_pattern(smarts):
        atypes = molsetup.atom_type.copy()
        reactive_atom_index = atom_indices[smarts_idx]

        # type reactive atom
        original_type = atypes[reactive_atom_index]
        reactive_type = get_reactive_atype(original_type, reactive_order=1) 
        atypes[reactive_atom_index] = reactive_type

        # type atoms 1 bond away from reactive atom
        for index1 in molsetup.graph[reactive_atom_index]:
            original_type = atypes[index1]
            reactive_type = get_reactive_atype(original_type, reactive_order=2)
            atypes[index1] = reactive_type

            # type atoms 2 bonds away from reactive
            for index2 in molsetup.graph[index1]:
                if index2 == reactive_atom_index:
                    continue
                original_type = atypes[index2]
                reactive_type = get_reactive_atype(original_type, reactive_order=3)
                atypes[index2] = reactive_type

        atype_dicts.append(atypes)

    return atype_dicts
    
