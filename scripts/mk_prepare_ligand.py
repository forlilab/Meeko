#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys

from openbabel import openbabel as ob

from meeko import MoleculePreparation
from meeko import obutils


def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Meeko")
    parser.add_argument("-i", "--mol", dest="input_molecule_file", required=True,
                        action="store", help="molecule file (MOL2, SDF,...)")
    parser.add_argument("-m", "--macrocycle", dest="build_macrocycle", default=False,
                        action="store_true", help="break macrocycle for docking")
    parser.add_argument("-w", "--hydrate", dest="add_water", default=False,
                        action="store_true", help="add water molecules for hydrated docking")
    parser.add_argument("--no_merge_hydrogen", dest="no_merge_hydrogen", default=True,
                        action="store_false", help="do not merge nonpolar hydrogen atoms")
    parser.add_argument("--add_hydrogen", dest="add_hydrogen", default=False,
                        action="store_true", help="add hydrogen atoms")
    parser.add_argument("--pH", dest="pH_value", default=None,
                        action="store", help="correct protonation for pH (default: No correction)")
    parser.add_argument("-f", "--flex", dest="is_protein_sidechain", default=False,
                        action="store_true", help="prepare as flexible protein residue")
    parser.add_argument("-r", "--rigidify_bonds_smarts", dest="rigidify_bonds_smarts", default=[],
                        action="append", help="SMARTS patterns to rigidify bonds")
    parser.add_argument("-b", "--rigidify_bonds_indices", dest="rigidify_bonds_indices", default=[],
                        action="append", help="indices of two atoms that define a bond (start at 1)", nargs='+', type=int)
    parser.add_argument("--no_index_map", dest="save_index_map", default=True,
                        action="store_false", help="do not write map of atom indices from input to pdbqt")
    parser.add_argument("-o", "--out", dest="output_pdbqt_file", default=None,
                        action="store", help="output pdbqt file")
    parser.add_argument("-v", "--verbose", dest="verbose", default=False,
                        action="store_true", help="print information about molecule setup")
    return parser.parse_args()


def main():
    args = cmd_lineparser()
    input_molecule_file = args.input_molecule_file
    output_pdbqt_file = args.output_pdbqt_file
    verbose = args.verbose
    build_macrocycle = args.build_macrocycle
    add_water = args.add_water
    no_merge_hydrogen = args.no_merge_hydrogen
    add_hydrogen = args.add_hydrogen
    pH_value = args.pH_value
    is_protein_sidechain = args.is_protein_sidechain
    save_index_map = args.save_index_map

    # SMARTS patterns to make bonds rigid
    rigidify_bonds_smarts = args.rigidify_bonds_smarts
    rigidify_bonds_indices = args.rigidify_bonds_indices
    if len(rigidify_bonds_indices) != len(rigidify_bonds_smarts):
        raise RuntimeError('length of --rigidify_bonds_indices differs from length of --rigidify_bonds_smarts')
    for indices in rigidify_bonds_indices:
        if len(indices) != 2:
            raise RuntimeError('--rigidify_bonds_indices must specify pairs, e.g. -b 1 2 -b 3 4')
        indices[0] = indices[0] - 1 # convert from 1- to 0-index
        indices[1] = indices[1] - 1

    mol = obutils.load_molecule_from_file(input_molecule_file)

    if pH_value is not None:
        mol.CorrectForPH(float(pH_value))

    if add_hydrogen:
        mol.AddHydrogens()
        charge_model = ob.OBChargeModel.FindType("Gasteiger")
        charge_model.ComputeCharges(mol)

    preparator = MoleculePreparation(merge_hydrogens=no_merge_hydrogen, macrocycle=build_macrocycle, 
                                     hydrate=add_water, amide_rigid=True,
                                     rigidify_bonds_smarts=rigidify_bonds_smarts,
                                     rigidify_bonds_indices=rigidify_bonds_indices)
    preparator.prepare(mol, is_protein_sidechain)

    # maybe verbose could be an option and it will show the various bond scores and breakdowns?
    if verbose:
        preparator.show_setup()

    if output_pdbqt_file is None:
        text = preparator.write_pdbqt_string(save_index_map)
        print(text)
    else:
        preparator.write_pdbqt_file(output_pdbqt_file, save_index_map)


if __name__ == '__main__':
    main()
