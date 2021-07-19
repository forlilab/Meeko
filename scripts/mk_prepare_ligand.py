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
    parser.add_argument("-i", "--mol", dest="input_molecule_filename", required=True,
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
                        action="append", help="SMARTS patterns to rigidify bonds",
                        metavar='SMARTS')
    parser.add_argument("-b", "--rigidify_bonds_indices", dest="rigidify_bonds_indices", default=[],
                        action="append", help="indices of two atoms (in the SMARTS) that define a bond (start at 1)",
                        nargs='+', type=int, metavar='i j')
    parser.add_argument("--double_bond_penalty", default=50, help="penalty > 100 prevents breaking double bonds", type=int)
    parser.add_argument("--no_index_map", dest="save_index_map", default=True,
                        action="store_false", help="do not write map of atom indices from input to pdbqt")
    parser.add_argument("-o", "--out", dest="output_pdbqt_filename", default=None,
                        action="store", help="output pdbqt filename. Single molecule input only.")
    parser.add_argument("--multimol_outdir", dest="multimol_output_directory", default=None,
                        action="store", help="folder to write output pdbqt for multi-mol inputs. Incompatible with -o/--out and -/--.")
    parser.add_argument("--multimol_prefix", dest="multimol_prefix", default=None,
                        action="store", help="replace internal molecule name in multi-molecule input by specified prefix. Incompatible with -o/--out and -/--.")
    parser.add_argument("-v", "--verbose", dest="verbose", default=False,
                        action="store_true", help="print information about molecule setup")
    parser.add_argument('-', '--',  dest='redirect_stdout', action='store_true',
                        help='do not write file, redirect output to STDOUT. Argument -o/--out is ignored. Single molecule input only.')
    args = parser.parse_args()
    if args.multimol_output_directory is not None or args.multimol_prefix is not None:
        if args.output_pdbqt_filename is not None:
            print("Argument -o/--out incompatible with --multimol_outdir and --multimol_prefix", file=sys.stderr)
            sys.exit(2)
        if args.redirect_stdout:
            print("Argument -/-- incompatible with --multimol_outdir and --multimol_prefix", file=sys.stderr)
            sys.exit(2)

    return args


def main():
    args = cmd_lineparser()
    input_molecule_filename = args.input_molecule_filename
    output_pdbqt_filename = args.output_pdbqt_filename
    verbose = args.verbose
    build_macrocycle = args.build_macrocycle
    double_bond_penalty = args.double_bond_penalty
    add_water = args.add_water
    no_merge_hydrogen = args.no_merge_hydrogen
    add_hydrogen = args.add_hydrogen
    pH_value = args.pH_value
    is_protein_sidechain = args.is_protein_sidechain
    save_index_map = args.save_index_map
    redirect_stdout = args.redirect_stdout
    multimol_output_directory = args.multimol_output_directory
    multimol_prefix = args.multimol_prefix

    do_process_multimol = (multimol_prefix is not None) or (multimol_output_directory is not None)
    if do_process_multimol:
        pdbqt_byname = {}
        duplicates = []

    if multimol_output_directory is not None:
        if not os.path.exists(multimol_output_directory):
            os.mkdir(multimol_output_directory)

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

    frmt = os.path.splitext(input_molecule_filename)[1][1:]
    with open(input_molecule_filename) as f:
        input_string = f.read()
    obmol_supplier = obutils.OBMolSupplier(input_string, frmt)

    mol_counter = 0

    for mol in obmol_supplier:

        mol_counter += 1

        if mol_counter > 1 and do_process_multimol == False:
            print("Processed only the first molecule of multiple molecule input.")
            print("Use --multimol_prefix and/or --multimol_outdir to process all molecules in %s." % (
                input_molecule_filename))
            break

        if pH_value is not None:
            mol.CorrectForPH(float(pH_value))

        if add_hydrogen:
            mol.AddHydrogens()
            charge_model = ob.OBChargeModel.FindType("Gasteiger")
            charge_model.ComputeCharges(mol)

        preparator = MoleculePreparation(merge_hydrogens=no_merge_hydrogen, macrocycle=build_macrocycle, 
                                         hydrate=add_water, amide_rigid=True,
                                         rigidify_bonds_smarts=rigidify_bonds_smarts,
                                         rigidify_bonds_indices=rigidify_bonds_indices,
                                         double_bond_penalty=double_bond_penalty)
        preparator.prepare(mol, is_protein_sidechain)

        # maybe verbose could be an option and it will show the various bond scores and breakdowns?
        if verbose:
            preparator.show_setup()

        ligand_prepared = preparator.write_pdbqt_string(save_index_map)

        # single molecule mode (no --multimol_* arguments were provided)
        if not do_process_multimol:
            if not redirect_stdout:
                if output_pdbqt_filename is None:
                    output_pdbqt_filename = '%s.pdbqt' % os.path.splitext(input_molecule_filename)[0]

                print(ligand_prepared, file=open(output_pdbqt_filename, 'w'))
            else:
                print(ligand_prepared)

        # multiple molecule mode        
        else:
            name = mol.GetTitle()
            if name in pdbqt_byname:
                duplicates.append(name)
            if multimol_prefix is not None:
                name = '%s-%d' % (multimol_prefix, mol_counter)
                pdbqt_byname[name] = ligand_prepared
            elif name not in duplicates:
                pdbqt_byname[name] = ligand_prepared

    if do_process_multimol:
        if len(duplicates):
            if multimol_prefix:
                print("Warning: %d molecules had duplicated names, e.g. %s" % (len(duplicates), duplicates[0]))
            else:
                print("Warning: %d molecules with duplicated names were NOT processed, e.g. %s" % (len(duplicates), duplicates[0]))

        if multimol_output_directory is None: multimol_output_directory = '.'
        for name in pdbqt_byname:
            fname = os.path.join(multimol_output_directory, name + '.pdbqt')
            with open(fname, 'w') as f:
                f.write(pdbqt_byname[name])

if __name__ == '__main__':
    main()
