#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#

import argparse
import gzip
import pathlib
import sys
import warnings

from rdkit import Chem

from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
from meeko import LinkedRDKitChorizo
from meeko import export_pdb_updated_flexres


def cmd_lineparser():
    parser = argparse.ArgumentParser(
        description='Export docked ligand to SDF, and receptor to PDB',
    )
    parser.add_argument(dest='docking_results_filename', nargs = "+",
                        help='Docking output file(s), either PDBQT \
                        file from Vina or DLG file from AD-GPU.')
    parser.add_argument('-s', '--write_sdf', metavar='filename',
                        help="defaults to input filename with suffix from --sufix")
    parser.add_argument(
        '-p',
        '--write_pdb',
        metavar='filename',
        help="defaults to input filename with suffix from --suffix",
    )
    parser.add_argument("--suffix", default="_docked",
                        help="suffix for output filesnames that are not explicitly set")
    parser.add_argument('-j', '--read_json', metavar='filename',
                        help="receptor written by mk_prepare_receptor -j/--write_json")
    parser.add_argument('--all_dlg_poses', action='store_true',
                        help="write all AutoDock-GPU poses, not just cluster leads.")
    parser.add_argument('-k', '--keep_flexres_sdf', action='store_true',
                        help="add flexres, if any, to SDF ouput")
    parser.add_argument('-', '--',  dest='redirect_stdout', action='store_true',
                        help="do not write SDF file, just print it to STDOUT")
    return parser.parse_args()

args = cmd_lineparser()

docking_results_filenames = args.docking_results_filename
write_sdf = args.write_sdf
write_pdb = args.write_pdb
read_json = args.read_json
suffix = args.suffix
all_dlg_poses = args.all_dlg_poses
keep_flexres_sdf = args.keep_flexres_sdf
redirect_stdout = args.redirect_stdout

if (
    (write_sdf is not None or write_pdb is not None) and
    len(docking_results_filenames) > 1
):
    msg = "With multiple input files, the output filenames are based on the"
    msg += "input filename. The suffix can be controlled with option --suffix."
    msg += "--write options are incompatible with multiple input files"
    print("--write options incompatible with multiple input files", file=sys.stderr)
    sys.exit(2)

if redirect_stdout and len(docking_results_filenames) > 1:
    print("option -/-- incompatible with multiple input files", file=sys.stderr)
    sys.exit(2)

if read_json is not None:
    with open(read_json) as f:
        json_string = f.read()
    polymer = LinkedRDKitChorizo.from_json(json_string)
else:
    polymer = None
    if write_pdb is not None:
        print("option -p (write pdb) requires -j (receptor receptor file)", file=sys.stderr)
        sys.exit(2)


for filename in docking_results_filenames:
    is_dlg = filename.endswith('.dlg') or filename.endswith(".dlg.gz")
    if filename.endswith(".gz"):
        with gzip.open(filename) as f:
            string = f.read().decode()
    else:
        with open(filename) as f:
            string = f.read()
    mol_name = pathlib.Path(filename).with_suffix("").name
    pdbqt_mol = PDBQTMolecule(
        string,
        name=mol_name,
        is_dlg=is_dlg,
        skip_typing=True,
    )
    only_cluster_leads = is_dlg and not all_dlg_poses
    sdf_string, failures = RDKitMolCreate.write_sd_string(
            pdbqt_mol,
            only_cluster_leads=only_cluster_leads,
            keep_flexres=keep_flexres_sdf,
    )
    for i in failures:
        warnings.warn("molecule %d not converted to RDKit/SD File" % i)
    if len(failures) == len(pdbqt_mol._atom_annotations["mol_index"]):
        msg = "\nCould not convert to RDKit. Maybe meeko was not used for preparing\n"
        msg += "the input PDBQT for docking, and the SMILES string is missing?\n"
        msg += "Except for standard protein sidechains, all ligands and flexible residues\n"
        msg += "require a REMARK SMILES line in the PDBQT, which is added automatically by meeko."
        raise RuntimeError(msg)
    if not redirect_stdout:
        if write_sdf is None:
            fn = pathlib.Path(filename).with_suffix("").name + f"{suffix}.sdf"
        else:
            fn = write_sdf
        with open(fn, "w") as f:
            f.write(sdf_string)
    else:
        print(sdf_string)

    # write receptor with updated flexres
    if read_json is not None:
        pdb_string = ""
        for pose in pdbqt_mol:
            model_nr = pose.pose_id + 1
            pdb_string += "MODEL " + f"{model_nr:8}" + pathlib.os.linesep
            pdb_string += export_pdb_updated_flexres(polymer, pose)
            pdb_string += "ENDMDL" + pathlib.os.linesep
        if write_pdb is None:
            fn = pathlib.Path(filename).with_suffix("").name + f"{suffix}.pdb"
        else:
            fn = write_pdb
        with open(fn, "w") as f:
            f.write(pdb_string)