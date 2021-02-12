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

    mol = obutils.load_molecule_from_file(input_molecule_file)

    if pH_value is not None:
        mol.CorrectForPH(float(pH_value))

    if add_hydrogen:
        mol.AddHydrogens()
        charge_model = ob.OBChargeModel.FindType("Gasteiger")
        charge_model.ComputeCharges(mol)

    preparator = MoleculePreparation(merge_hydrogens=no_merge_hydrogen, macrocycle=build_macrocycle, 
                                     hydrate=add_water, amide_rigid=True)
    preparator.prepare(mol)

    # maybe verbose could be an option and it will show the various bond scores and breakdowns?
    if verbose:
        preparator.show_setup()

    if output_pdbqt_file is None:
        # if output format not defined, use infile to infer outname
        name, ext = os.path.splitext(input_molecule_file)
        output_pdbqt_file = "%s.pdbqt" % (name)

    preparator.write_pdbqt_file(output_pdbqt_file)


if __name__ == '__main__':
    main()