#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys
import json

from rdkit import Chem

from meeko import MoleculePreparation
from meeko import rdkitutils

try:
    from meeko import obutils # fails if openbabel not available
except:
    _has_openbabel = False
else:
    _has_openbabel = True

def cmd_lineparser():
    backend = 'rdkit'
    if '--ob_backend' in sys.argv:
        if not _has_openbabel:
            raise ImportError('--ob_backend requires openbabel which is not available')
        backend = 'ob'
        sys.argv.remove('--ob_backend')
    conf_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    conf_parser.add_argument('-c', '--config_file',
            help='configure MoleculePreparation from JSON file. Overriden by command line args.')
    confargs, remaining_argv = conf_parser.parse_known_args()

    # initalize config dict with defaults from MoleculePreparation object
    preparator_defaults = MoleculePreparation.init_just_defaults()
    config = json.loads(json.dumps(preparator_defaults.__dict__)) # using dict -> str -> dict as a safe copy method

    if confargs.config_file is not None:
        with open(confargs.config_file) as f:
            c = json.load(f)
            config.update(c)

    parser = argparse.ArgumentParser(parents=[conf_parser]) # using parents to show --config_file in help msg
    parser.set_defaults(**config)
    parser.add_argument("-i", "--mol", dest="input_molecule_filename", required=True,
                        action="store", help="molecule file (MOL2, SDF,...)")
    parser.add_argument("--rigid_macrocycles",dest="rigid_macrocycles",
                        action="store_true", help="keep macrocycles rigid in input conformation")
    parser.add_argument("-w", "--hydrate", dest="hydrate",
                        action="store_true", help="add water molecules for hydrated docking")
    parser.add_argument("--keep_nonpolar_hydrogens", dest="keep_nonpolar_hydrogens",
                        action="store_true", help="keep non-polar hydrogens (default: merge onto heavy atom)")
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
    parser.add_argument("-a", "--flexible_amides", dest="flexible_amides",
                        action="store_true", help="allow amide bonds to rotate and be non-planar, which is bad")
    parser.add_argument("-p", "--atom_type_smarts", dest="atom_type_smarts_json",
                        action="store", help="SMARTS based atom typing (JSON format)")
    parser.add_argument("--double_bond_penalty", help="penalty > 100 prevents breaking double bonds", type=int)
    parser.add_argument("--add_index_map", dest="add_index_map",
                        action="store_true", help="write map of atom indices from input to pdbqt")
    parser.add_argument("--remove_smiles", dest="remove_smiles",
                        action="store_true", help="do not write smiles as remark to pdbqt")
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
    # just for display option, if the arg exists, it will be parsed by 'conf_parser' above

    args = parser.parse_args(remaining_argv)

    # command line arguments override config
    for key in config:
        if key in args.__dict__:
            config[key] = args.__dict__[key]

    if args.atom_type_smarts_json is not None:
        with open(args.atom_type_smarts_json) as f:
            config['atom_type_smarts'] = json.load(f)

    if args.multimol_output_directory is not None or args.multimol_prefix is not None:
        if args.output_pdbqt_filename is not None:
            print("Argument -o/--out incompatible with --multimol_outdir and --multimol_prefix", file=sys.stderr)
            sys.exit(2)
        if args.redirect_stdout:
            print("Argument -/-- incompatible with --multimol_outdir and --multimol_prefix", file=sys.stderr)
            sys.exit(2)

    return args, config, backend


if __name__ == '__main__':

    args, config, backend = cmd_lineparser()
    multimol_output_directory = args.multimol_output_directory
    multimol_prefix = args.multimol_prefix
    input_molecule_filename = args.input_molecule_filename

    do_process_multimol = (multimol_prefix is not None) or (multimol_output_directory is not None)
    if do_process_multimol:
        pdbqt_byname = {}
        duplicates = []

    if multimol_output_directory is not None:
        if not os.path.exists(multimol_output_directory):
            os.mkdir(multimol_output_directory)

    # SMARTS patterns to make bonds rigid
    rigidify_bonds_smarts = config['rigidify_bonds_smarts']
    rigidify_bonds_indices = config['rigidify_bonds_indices']
    if len(rigidify_bonds_indices) != len(rigidify_bonds_smarts):
        raise RuntimeError('length of --rigidify_bonds_indices differs from length of --rigidify_bonds_smarts')
    for indices in rigidify_bonds_indices:
        if len(indices) != 2:
            raise RuntimeError('--rigidify_bonds_indices must specify pairs, e.g. -b 1 2 -b 3 4')
        indices[0] = indices[0] - 1 # convert from 1- to 0-index
        indices[1] = indices[1] - 1

    fname, ext = os.path.splitext(input_molecule_filename)
    ext = ext[1:].lower()
    if backend == 'rdkit':
        parsers = {'sdf': Chem.SDMolSupplier, 'mol2': rdkitutils.Mol2MolSupplier}
        if not ext in parsers:
            print("*ERROR* Format [%s] not supported." % ext)
            sys.exit(1)
        mol_supplier = parsers[ext](input_molecule_filename, removeHs=False) # input must have explicit H
    elif backend == 'ob':
        print("Using openbabel instead of rdkit")
        mol_supplier = obutils.OBMolSupplier(input_molecule_filename, ext)

    mol_counter = 0
    num_skipped = 0
    is_after_first = False
    for mol in mol_supplier:
        if is_after_first and do_process_multimol == False:
            print("Processed only the first molecule of multiple molecule input.")
            print("Use --multimol_prefix and/or --multimol_outdir to process all molecules in %s." % (
                input_molecule_filename))
            break
        is_after_first = True

        # check that molecule was successfully loaded
        if backend == 'rdkit':
            is_valid = mol is not None
        elif backend == 'ob':
            is_valid = mol.NumAtoms() > 0
        mol_counter += int(is_valid==True)
        num_skipped += int(is_valid==False)
        if not is_valid: continue

        preparator = MoleculePreparation.from_config(config)
        preparator.prepare(mol)

        # maybe verbose could be an option and it will show the various bond scores and breakdowns?
        if args.verbose:
            preparator.show_setup()

        ligand_prepared = preparator.write_pdbqt_string()

        # single molecule mode (no --multimol_* arguments were provided)
        if not do_process_multimol:
            if not args.redirect_stdout:
                if args.output_pdbqt_filename is None:
                    output_pdbqt_filename = '%s.pdbqt' % fname
                else:
                    output_pdbqt_filename = args.output_pdbqt_filename

                print(ligand_prepared, file=open(output_pdbqt_filename, 'w'))
            else:
                print(ligand_prepared, end='')

        # multiple molecule mode
        else:
            name = preparator.setup.name # setup.name may be None
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

        print("Processed molecules: %d" % mol_counter)
        print("Skipped molecules: %d" % num_skipped)
