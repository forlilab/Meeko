#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys
import json

from openbabel import openbabel as ob

from meeko import MeekoConfig
from meeko import MoleculePreparation
from meeko import obutils

def cmd_lineparser(argv=None):
    mk_config = MeekoConfig()
    if argv is None:
        argv = sys.argv

    conf_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    conf_parser.add_argument("-c", "--config_file")

    args, remaining_argv = conf_parser.parse_known_args()

    if args.config_file:
        mk_config.config_filename = args.config_file
        mk_config.update_from_json()

    parser = argparse.ArgumentParser(description="Meeko")
    parser.set_defaults(**mk_config.__dict__)
    parser.add_argument("-i", "--mol", dest="input_molecule_filename", required=True,
                        action="store", help="molecule file (MOL2, SDF,...)")
    parser.add_argument("-m", "--macrocycle",dest="break_macrocycle",
                        action="store_true", help="break macrocycle for docking")
    parser.add_argument("-w", "--hydrate", dest="hydrate",
                        action="store_true", help="add water molecules for hydrated docking")
    parser.add_argument("--no_merge_hydrogen", dest="merge_hydrogens",
                        action="store_false", help="do not merge nonpolar hydrogen atoms")
    parser.add_argument("--add_hydrogen", dest="add_hydrogen",
                        action="store_true", help="add hydrogen atoms")
    parser.add_argument("--pH", dest="pH_value",
                        action="store", help="correct protonation for pH (default: No correction)")
    parser.add_argument("-f", "--flex", dest="is_protein_sidechain",
                        action="store_true", help="prepare as flexible protein residue")
    parser.add_argument("-r", "--rigidify_bonds_smarts", dest="rigidify_bonds_smarts",
                        action="append", help="SMARTS patterns to rigidify bonds",
                        metavar='SMARTS')
    parser.add_argument("-b", "--rigidify_bonds_indices", dest="rigidify_bonds_indices",
                        action="append", help="indices of two atoms (in the SMARTS) that define a bond (start at 1)",
                        nargs='+', type=int, metavar='i j')
    parser.add_argument("-p", "--param", dest="params_filename",
                        action="store", help="SMARTS based atom typing (JSON format)")
    parser.add_argument("--double_bond_penalty", help="penalty > 100 prevents breaking double bonds", type=int)
    parser.add_argument("--no_index_map", dest="save_index_map",
                        action="store_false", help="do not write map of atom indices from input to pdbqt")
    parser.add_argument("-o", "--out", dest="output_pdbqt_filename",
                        action="store", help="output pdbqt filename. Single molecule input only.")
    parser.add_argument("--multimol_outdir", dest="multimol_output_directory",
                        action="store", help="folder to write output pdbqt for multi-mol inputs. Incompatible with -o/--out and -/--.")
    parser.add_argument("--multimol_prefix", dest="multimol_prefix",
                        action="store", help="replace internal molecule name in multi-molecule input by specified prefix. Incompatible with -o/--out and -/--.")
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action="store_true", help="print information about molecule setup")
    parser.add_argument('-', '--',  dest='redirect_stdout', action='store_true',
                        help='do not write file, redirect output to STDOUT. Argument -o/--out is ignored. Single molecule input only.')

    args = parser.parse_args(remaining_argv)
    
    mk_config.__dict__.update(args.__dict__)

    if mk_config.multimol_output_directory is not None or mk_config.multimol_prefix is not None:
        if mk_config.output_pdbqt_filename is not None:
            print("Argument -o/--out incompatible with --multimol_outdir and --multimol_prefix", file=sys.stderr)
            sys.exit(2)
        if mk_config.redirect_stdout:
            print("Argument -/-- incompatible with --multimol_outdir and --multimol_prefix", file=sys.stderr)
            sys.exit(2)

    return mk_config


def main():
    mk_config = cmd_lineparser()
    multimol_output_directory = mk_config.multimol_output_directory
    multimol_prefix = mk_config.multimol_prefix
    input_molecule_filename = mk_config.input_molecule_filename

    do_process_multimol = (mk_config.multimol_prefix is not None) or (multimol_output_directory is not None)
    if do_process_multimol:
        pdbqt_byname = {}
        duplicates = []

    if multimol_output_directory is not None:
        if not os.path.exists(multimol_output_directory):
            os.mkdir(multimol_output_directory)

    # SMARTS patterns to make bonds rigid
    rigidify_bonds_smarts = mk_config.rigidify_bonds_smarts
    rigidify_bonds_indices = mk_config.rigidify_bonds_indices
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

        if mk_config.pH_value is not None:
            mol.CorrectForPH(float(mk_config.pH_value))

        if mk_config.add_hydrogen:
            mol.AddHydrogens()
            charge_model = ob.OBChargeModel.FindType("Gasteiger")
            charge_model.ComputeCharges(mol)

        preparator = MoleculePreparation(mk_config)
        preparator.prepare(mol, mk_config.is_protein_sidechain)

        # maybe verbose could be an option and it will show the various bond scores and breakdowns?
        if mk_config.verbose:
            preparator.show_setup()

        ligand_prepared = preparator.write_pdbqt_string(mk_config.save_index_map)

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
