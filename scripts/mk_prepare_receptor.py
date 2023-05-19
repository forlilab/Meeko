#!/usr/bin/env python

import argparse
import json
from os import linesep as os_linesep
import pathlib
import sys

from meeko import Receptor

path_to_this_script = pathlib.Path(__file__).resolve()

def parse_residue_string(string):
    """
    Args:
        string (str): Residue identifier in the format 'chain:resname:resnum'.

    Returns:
        tuple: Residue identifier components: (chain, resname, resnum).
        bool: Whether the parsing was successful or not.
        str: An error message describing any encountered parsing errors.

    Examples:
        >>> parse_residue_string("A:ALA:42")
        (('A', 'ALA', 42), True, '')
    """

    ok = True
    err = ""
    res_id = (None, None, None)
    if string.count(":") != 2:
        ok = False
        err = "Need exacly two ':' but found %d in '%s'" % (string.count(":"), string) + os_linesep
        return res_id, ok, err
    chain, resname, resnum = string.split(":")
    if len(chain) != 1:
        ok = False
        err += "chain must be 1 char but it is '%s' (%d chars) in '%s'" % (chain, len(chain), string) + os_linesep
    if len(resname) > 3:
        ok = False
        err += "resname must be max 3 characters long, but is '%s' in '%s'" % (resname, string) + os_linesep
    try:
        resnum = int(resnum)
    except:
        ok = False
        err += "resnum could not be converted to integer, it was '%s' in '%s'" % (resnum, string) + os_linesep
    if ok:
        res_id = (chain, resname, resnum)
    return res_id, ok, err 


class TalkativeParser(argparse.ArgumentParser):
    def error(self, message):
        """overload to print_help for every error"""
        self.print_help()
        this_script = path_to_this_script.name
        print('\n%s: error: %s' % (this_script, message), file=sys.stderr)
        sys.exit(2)

def get_args():
    parser = TalkativeParser()
    parser.add_argument('--pdb', help="input can be PDBQT but charges and types will be reassigned")
    parser.add_argument('--pdbqt', help="keeps existing charges and types")
    parser.add_argument('-o', '--output_filename', required=True, help="will suffix _rigid/_flex with --flexres")
    parser.add_argument('-f', '--flexres', action="append",
                        help="repeat flag for each residue, e.g: -f \" :LYS:42\" -f \"B:TYR:23\" and keep space for empty chain")
    parser.add_argument('-r', '--reactive_flexres', action="append",
                        help="same as --flexres but for reactive residues (max 8)")
    args = parser.parse_args()
    if (args.pdb is None) == (args.pdbqt is None):
        msg = "need either --pdb or --pdbqt"
        print("Command line error: " + msg, file=sys.stderr)
        sys.exit(2)
    if (args.pdbqt is not None) and args.flexres is None:
        msg = "nothing to do when reading --pdbqt and no --flexres."
        print("Command line error: " + msg, file=sys.stderr)
        sys.exit(2)
    if args.reactive_flexres and len(args.reactive_flexres) > 8: 
        msg = "got %d reactive_flexres but maximum is 8." % (len(args.reactive_flexres))
        print("Command line error: " + msg, file=sys.stderr)
        sys.exit(2)
    return args

args = get_args()

if args.pdb is not None:
    with open(args.pdb) as f:
        pdb_string = f.read()
    data, success, error_msg = Receptor.parse_residue_data_from_pdb(pdb_string)
    if not success:
        print("Error: " + error_msg, file=sys.stderr, end="")
        sys.exit(2)
    atom_params = Receptor.assign_residue_params(data["resnames"], data["atom_names"])
    atom_types = atom_params["atom_types"]
    charges    = atom_params["gasteiger"]
    pdbqtstring = Receptor.write_pdbqt_from_residue_data(data, charges, atom_types)
else:
    with open(args.pdbqt) as f:
        pdbqtstring = f.read()
    
if args.flexres is None:
    with open(args.output_filename, "w") as f:
        f.write(pdbqtstring)
else:
    res_ids = []
    all_ok = True
    all_err = ""
    for flexres in args.flexres:
        res_id, ok, err = parse_residue_string(flexres) # expecting "A:TYR:42"
        all_ok &= ok
        all_err += err
        res_ids.append(res_id)
    if not all_ok:
        print(all_err, file=sys.stderr)
        sys.exit(2)
    receptor = Receptor(pdbqtstring)
    pdbqt, success, error_msg = receptor.write_pdbqt_string(res_ids)
    if not success:
        print("Error: " + error_msg, file=sys.stderr)
        sys.exit(2)

    outpath = pathlib.Path(args.output_filename)
    rigid_fn = str(outpath.with_suffix("")) + "_rigid" + outpath.suffix
    flex_fn = str(outpath.with_suffix("")) + "_flex" + outpath.suffix
    with open(rigid_fn, "w") as f:
        f.write(pdbqt["rigid"]) 
    with open(flex_fn, "w") as f:
        f.write(pdbqt["flex"])
