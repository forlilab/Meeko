#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import os
import sys
import json
import warnings

from rdkit import Chem

from meeko import MoleculePreparation
from meeko import rdkitutils
from meeko import PDBQTWriterLegacy

try:
    import prody
    from meeko import CovalentBuilder
    _prody_parsers = {'pdb': prody.parsePDB, 'mmcif': prody.parseMMCIF}
    warnings.warn("Prody not available, covalent docking won't work", ImportWarning)
except:
    _has_prody = False
    _prody_parsers = {}
else:
    _has_prody = True

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

    config = MoleculePreparation.get_defaults_dict()

    if confargs.config_file is not None:
        with open(confargs.config_file) as f:
            c = json.load(f)
            config.update(c)

    parser = argparse.ArgumentParser()#parents=[conf_parser]) # parents shows --config_file in help msg
    parser.set_defaults(**config)
    parser.add_argument("-v", "--verbose", dest="verbose",
                        action="store_true", help="print information about molecule setup")

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("-i", "--mol", dest="input_molecule_filename", required=True,
                        action="store", help="molecule file (MOL2, SDF,...)")
    io_group.add_argument("-o", "--out", dest="output_pdbqt_filename",
                        action="store", help="output pdbqt filename. Single molecule input only.")
    io_group.add_argument("--multimol_outdir", dest="multimol_output_dir",
                        action="store", help="folder to write output pdbqt for multi-mol inputs. Incompatible with -o/--out and -/--.")
    io_group.add_argument("--multimol_prefix", dest="multimol_prefix",
                        action="store", help="replace internal molecule name in multi-molecule input by specified prefix. Incompatible with -o/--out and -/--.")
    io_group.add_argument('-', '--',  dest='redirect_stdout', action='store_true',
                        help='do not write file, redirect output to STDOUT. Argument -o/--out is ignored. Single molecule input only.')

    config_group = parser.add_argument_group("Molecule preparation")
    config_group.add_argument('-c', '--config_file',
            help='configure MoleculePreparation from JSON file. Overriden by command line args.') # parsed above by conf_parser, here for help msg
    config_group.add_argument("--rigid_macrocycles",dest="rigid_macrocycles",
                        action="store_true", help="keep macrocycles rigid in input conformation")
    config_group.add_argument("--keep_chorded_rings",dest="keep_chorded_rings",
                        action="store_true", help="return all rings from exhaustive perception")
    config_group.add_argument("--keep_equivalent_rings",dest="keep_equivalent_rings",
                        action="store_true", help="equivalent rings have the same size and neighbors")
    config_group.add_argument("--min_ring_size", dest="min_ring_size",
                        type=int, help="min nr of atoms in ring for opening")
    config_group.add_argument("-w", "--hydrate", dest="hydrate",
                        action="store_true", help="add water molecules for hydrated docking")
    config_group.add_argument("--merge_these_atom_types", dest="merge_these_atom_types", nargs="*",
                        help="list of atom types to merge, default is \"H\"", default=["H"])
    config_group.add_argument("-r", "--rigidify_bonds_smarts", dest="rigidify_bonds_smarts",
                        action="append", help="SMARTS patterns to rigidify bonds",
                        metavar='SMARTS')
    config_group.add_argument("-b", "--rigidify_bonds_indices", dest="rigidify_bonds_indices",
                        action="append", help="indices of two atoms (in the SMARTS) that define a bond (start at 1)",
                        nargs='+', type=int, metavar='i j')
    config_group.add_argument("-a", "--flexible_amides", dest="flexible_amides",
                        action="store_true", help="allow amide bonds to rotate and be non-planar, which is bad")
    config_group.add_argument("-p", "--atom_type_smarts", dest="atom_type_smarts_json",
                        action="store", help="SMARTS based atom typing (JSON format)")
    config_group.add_argument("-aa", "--add_atom_types", dest="add_atom_types_json",
                        action="append", help="Additional atom types to assign (JSON formated)", metavar="[{'smarts': '<smarts pattern>', 'atype': ',atomtype name>'}, {'smarts': '<smarts pattern>', 'atype': ',atomtype name>'}]")
    config_group.add_argument("--double_bond_penalty", help="penalty > 100 prevents breaking double bonds", type=int)
    config_group.add_argument("--bad_charge_ok", help="NaN and Inf charges allowed in PDBQT", action="store_true")
    config_group.add_argument("--add_index_map", dest="add_index_map",
                        action="store_true", help="write map of atom indices from input to pdbqt")
    config_group.add_argument("--remove_smiles", dest="remove_smiles",
                        action="store_true", help="do not write smiles as remark to pdbqt")
    reactive_group = parser.add_argument_group("Reactive docking")
    reactive_group.add_argument('--reactive_smarts',  help="SMARTS pattern for reactive group")
    reactive_group.add_argument('--reactive_smarts_idx', help='index (1-based) of the reactive atom in --reactive_smarts', type=int)

    need_prody_msg = ''
    if not _has_prody: need_prody_msg = ". Needs Prody which is unavailable"
    covalent_group = parser.add_argument_group("Covalent docking (tethered)%s" % (need_prody_msg))
    covalent_group.add_argument('--receptor', help='receptor filename. Supported formats: [%s]%s' % (
        '/'.join(list(_prody_parsers.keys())),
        need_prody_msg))
    covalent_group.add_argument('--rec_residue',
                                help='examples: "A:LYS:204", "A:HIS:", ":LYS:"')
    covalent_group.add_argument('--tether_smarts',
                                help='SMARTS pattern to define ligand atoms for receptor attachment')
    covalent_group.add_argument('--tether_smarts_indices', type=int, nargs=2, required=False,
                                metavar='IDX', default=[1, 2],
                                help='indices (1-based) of the SMARTS atoms that will be attached (default: 1 2)')

    args = parser.parse_args(remaining_argv)

    # check reactive arguments
    if (args.reactive_smarts is None) != (args.reactive_smarts_idx is None):
        print("Arguments --reactive_smarts and --reactive_smarts_idx require each other", file=sys.stderr)
        sys.exit(2)
    elif args.reactive_smarts_idx is not None:
        if args.reactive_smarts_idx < 1:
            print("--reactive_smarts_idx is 1-indexed, but got %d" % args.reactive_smarts_idx, file=sys.stderr)
            sys.exit(2)
        args.reactive_smarts_idx -= 1 # convert from 1- to 0-index

    # command line arguments override config
    for key in config:
        if key in args.__dict__:
            config[key] = args.__dict__[key]

    if args.atom_type_smarts_json is not None:
        with open(args.atom_type_smarts_json) as f:
            config['atom_type_smarts'] = json.load(f)
    
    if args.add_atom_types_json is not None:
        additional_ats = []
        for at in args.add_atom_types_json:
            at = json.loads(at)
            if type(at) == dict:
                additional_ats.append(at)
            elif type(at) == list:
                additional_ats.extend(at)
        config['add_atom_types'] = additional_ats

    if args.multimol_output_dir is not None or args.multimol_prefix is not None:
        if args.output_pdbqt_filename is not None:
            print("Warning: -o/--out ignored with --multimol_outdir or --multimol_prefix", file=sys.stderr)
        if args.redirect_stdout:
            print("Warning: -/-- ignored with --multimol_outdir or --multimol_prefix", file=sys.stderr)

    # verify sanity of covalent docking input
    num_required_covalent_args = 0
    num_required_covalent_args += int(args.receptor is not None)
    num_required_covalent_args += int(args.rec_residue is not None)
    num_required_covalent_args += int(args.tether_smarts is not None)
    if num_required_covalent_args not in [0, 3]:
        print("Error: --receptor, --rec_residue, and --tether_smarts are all required for covalent docking.")
        sys.exit(2)
    is_covalent = num_required_covalent_args == 3
    if is_covalent and not _has_prody:
        raise ImportError("Covalent docking requires Prody which is not available")
    if min(args.tether_smarts_indices) < 1:
        print("--tether_smarts_indices is 1-indexed, all values must be greater than zero", file=sys.stderr) 
        sys.exit(2)
    args.tether_smarts_indices = [i-1 for i in args.tether_smarts_indices] # convert to 0-index

    # verify sanity of SMARTS patterns to make bonds rigid and convert to 0-based indices
    rigidify_bonds_smarts = config['rigidify_bonds_smarts']
    rigidify_bonds_indices = config['rigidify_bonds_indices']
    if len(rigidify_bonds_indices) != len(rigidify_bonds_smarts):
        raise RuntimeError('length of --rigidify_bonds_indices differs from length of --rigidify_bonds_smarts')
    for indices in rigidify_bonds_indices:
        if len(indices) != 2:
            raise RuntimeError('--rigidify_bonds_indices must specify pairs, e.g. -b 1 2 -b 3 4')
        indices[0] = indices[0] - 1 # convert from 1- to 0-index
        indices[1] = indices[1] - 1

    return args, config, backend, is_covalent


class Output:
    def __init__(self, multimol_output_dir, multimol_prefix, redirect_stdout, output_filename):
        is_multimol = (multimol_prefix is not None) or (multimol_output_dir is not None)
        self._mkdir(multimol_output_dir)

        if multimol_output_dir is None:
            multimol_output_dir = '.'
        self.multimol_output_dir = multimol_output_dir
        self.multimol_prefix = multimol_prefix
        self.redirect_stdout = redirect_stdout
        self.output_filename = output_filename
        self.is_multimol = is_multimol
        self.visited_filenames = set()
        self.duplicate_filenames = set()
        self.visited_names = set()
        self.duplicate_names = set()
        self.num_files_written = 0
        self.counter = 0

    def __call__(self, pdbqt_string, name, suffixes=()):
        self.counter += 1
        if name in self.visited_names:
            self.duplicate_names.add(name)
        self.visited_names.add(name)
        if self.multimol_prefix is not None:
            name = '%s-%d' % (args.multimol_prefix, self.counter)
        for suffix in suffixes:
            if suffix is not None and len(suffix) > 0:
                name += "_" + suffix
        if self.is_multimol:
            if name in self.visited_filenames:
                self.duplicate_filenames.add(name)
                repeat_id = 1
                newname = name + "-again%d" % repeat_id
                while newname in self.visited_filenames:
                    repeat_id += 1
                    newname = name + "-again%d" % repeat_id
                print("Renaming %s to %s to disambiguate filename" % (name, newname), file=sys.stderr)
                name = newname
            self.visited_filenames.add(name)
            fpath = os.path.join(self.multimol_output_dir, name + '.pdbqt')
            print(pdbqt_string, end='', file=open(fpath, 'w'))
            self.num_files_written += 1
        elif self.redirect_stdout:
            print(pdbqt_string, end='')
        else:
            if self.output_filename is None:
                filename = '%s.pdbqt' % name
            else:
                filename = self.output_filename
            print(pdbqt_string, end='', file=open(filename, 'w'))
            self.num_files_written += 1

    def _mkdir(self, multimol_output_dir):
        """make directory if it doesn't exist yet """
        if multimol_output_dir is not None:
            if not os.path.exists(multimol_output_dir):
                os.mkdir(multimol_output_dir)

    def get_duplicates_info_string(self):
        if not self.is_multimol:
            return None
        if len(self.duplicate_names):
            # with multimol_prefix with can have duplicate molecule names,
            # but not duplicate filenames. This warning is for such cases.
            string = "Warning: %d molecules have duplicated names" % len(self.duplicate_names)
        else:
            string = "No duplicate molecule molecule names were found"
        # if we have duplicate_filenames, we also have duplicate molecule names,
        # but it suffices to return the string below
        if len(self.duplicate_filenames):
            string = "Warning: %d molecules with repeated names were suffixed with -again<n>" % len(self.duplicate_filenames)
        return string

    @staticmethod
    def get_suffixes(molsetups):
        if len(molsetups) == 1:
            return ("",) # no suffix needed
        else:
            return tuple("mk%d" % (i + 1) for i in range(len(molsetups)))


if __name__ == '__main__':

    args, config, backend, is_covalent = cmd_lineparser()
    input_molecule_filename = args.input_molecule_filename

    # read input
    input_fname, ext = os.path.splitext(input_molecule_filename)
    ext = ext[1:].lower()
    if backend == 'rdkit':
        parsers = {'sdf': Chem.SDMolSupplier, 'mol2': rdkitutils.Mol2MolSupplier, 'mol': Chem.SDMolSupplier}
        if not ext in parsers:
            print("*ERROR* Format [%s] not in supported formats [%s]" % (ext, '/'.join(list(parsers.keys()))))
            sys.exit(1)
        mol_supplier = parsers[ext](input_molecule_filename, removeHs=False) # input must have explicit H
    elif backend == 'ob':
        print("Using openbabel instead of rdkit")
        mol_supplier = obutils.OBMolSupplier(input_molecule_filename, ext)

    # configure output writer
    if args.output_pdbqt_filename is None:
        output_filename = input_fname + '.pdbqt'
    else:
        output_filename = args.output_pdbqt_filename
    output = Output(
            args.multimol_output_dir,
            args.multimol_prefix,
            args.redirect_stdout,
            output_filename,
            )

    # initialize covalent object for receptor
    if is_covalent:
        rec_filename = args.receptor
        _, rec_extension = os.path.splitext(rec_filename)
        rec_extension = rec_extension[1:].lower()
        prody_parser = _prody_parsers[rec_extension]
        rec_prody_mol = prody_parser(rec_filename)
        covalent_builder = CovalentBuilder(rec_prody_mol, args.rec_residue)

    input_mol_counter = 0
    input_mol_skipped = 0
    input_mol_with_failure = 0 # if reactive or covalent, each mol can yield multiple PDBQT
    nr_failures = 0
    is_after_first = False
    preparator = MoleculePreparation.from_config(config)
    for mol in mol_supplier:
        if is_after_first and output.is_multimol == False:
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
        input_mol_skipped += int(is_valid==False)
        if not is_valid: continue

        this_mol_had_failure = False

        if is_covalent:
            for cov_lig in covalent_builder.process(mol, args.tether_smarts, args.tether_smarts_indices):
                root_atom_index = cov_lig.indices[0]
                molsetups = preparator.prepare(cov_lig.mol, root_atom_index=root_atom_index, not_terminal_atoms=[root_atom_index])
                res, chain, num = cov_lig.res_id
                suffixes = output.get_suffixes(molsetups)
                for molsetup, suffix in zip(molsetups, suffixes):
                    pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup, bad_charge_ok=args.bad_charge_ok)
                    if success:
                        pdbqt_string = PDBQTWriterLegacy.adapt_pdbqt_for_autodock4_flexres(pdbqt_string, res, chain, num)
                        name = molsetup.name
                        output(pdbqt_string, name, (cov_lig.label, suffix))
                    else:
                        nr_failures += 1
                        this_mol_had_failure = True
                        print(error_msg, file=sys.stderr)
                        
        else:
            molsetups = preparator.prepare(mol)
            suffixes = output.get_suffixes(molsetups) 
            for molsetup, suffix in zip(molsetups, suffixes):
                pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(molsetup, bad_charge_ok=args.bad_charge_ok)
                if success:
                    name = molsetup.name
                    output(pdbqt_string, name, (suffix,))
                    if args.verbose: 
                        molsetup.show()
                else:
                    nr_failures += 1
                    this_mol_had_failure = True
                    print(error_msg, file=sys.stderr)

        input_mol_with_failure += int(this_mol_had_failure)

    if output.is_multimol:
        print("Input molecules processed: %d, skipped: %d" % (output.counter, input_mol_skipped))
        print("PDBQT files written: %d" % (output.num_files_written))
        print("PDBQT files not written due to error: %d" % (nr_failures))
        print("Input molecules with errors: %d" % (input_mol_with_failure))
        print(output.get_duplicates_info_string())
