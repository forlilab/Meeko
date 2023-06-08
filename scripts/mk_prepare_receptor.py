#!/usr/bin/env python

import argparse
import json
from os import linesep as os_linesep
import pathlib
import sys

from meeko import Receptor
from meeko import reactive_typer
from meeko import get_reactive_config

path_to_this_script = pathlib.Path(__file__).resolve()

def parse_residue_string(string):
    """
    Args:
        string (str): Residue identifier in the format 'chain:resname:resnum'.

    Returns:
        tuple: Residue identifier components: (chain, resname, resnum).
        bool: Whether the parsing was successful or not.
        str: An error message describing any encountered parsing errors.

    Example:
        >>> parse_residue_string("A:ALA:42")
        (('A', 'ALA', 42), True, '')
    """

    ok = True
    err = ""
    res_id = (None, None, None)
    if string.count(":") != 2:
        ok = False
        err += "Need exacly two ':' but found %d in '%s'" % (string.count(":"), string) + os_linesep
        err += "Example: 'A:HIE:42'" + os_linesep
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

def parse_residue_string_and_name(string):
    """
    Example:
        >>> parse_residue_string_and_name("A:HIE:42:CE1")
        ({"res_id": ('A', 'ALA', 42), "name": "CE1"}, True, '')
    """
    ok = True
    err = ""
    output = {"res_id": (None, None, None),
              "name": None}
    if string.count(":") != 3:
        ok = False
        err += "Need three ':' but found %d in %s" % (string.count(":"), string) + os_linesep
        err += "Example: 'A:HIE:42:CE1'" + os_linesep
        return output, ok, err

    resid_string, name = string.rsplit(":", 1)
    output["name"] = name
    res_id, ok_, err_ = parse_residue_string(resid_string)
    output["res_id"] = res_id
    ok &= ok_
    err += err_
    return output, ok, err

def parse_resname_and_name(string):
    """
    Example:
        >>> parse_resname_and_name("HIE:CE1")
        (("HIE", "CE1"), True, "")
    """
    ok = True
    err = ""
    output = (None, None)
    if string.count(":") != 1:
        ok = False
        err += "Expected one ':' but found %d in %s" % (string.count(":"), string) + os_linesep
        err += "Example: 'HIE:CE1'" + os_linesep
        return output, ok, err
    resname, name = string.split(":")
    return (resname, name), ok, err




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
    parser.add_argument('-o', '--output_filename', required=True, help="will suffix _rigid/_flex with flexible or reactive residues")
    parser.add_argument('-f', '--flexres', action="append", default=[],
                        help="repeat flag for each residue, e.g: -f \" :LYS:42\" -f \"B:TYR:23\" and keep space for empty chain")
    parser.add_argument('-r', '--reactive_flexres', action="append", default=[],
                        help="same as --flexres but for reactive residues (max 8)")
    parser.add_argument('-g', '--reactive_name', action="append", default=[],
                        help="set name of reactive atom of a residue type, e.g: -g 'TRP:NE1'. Overridden by --reactive_name_specific")
    parser.add_argument('-s', '--reactive_name_specific', action="append", default=[],
                        help="set name of reactive atom for an individual residue, e.g: -s 'A:HIE:42:NE2'. Residue will be reactive.")
    parser.add_argument('--r_eq_12', default=2.5, type=float, help="r_eq for reactive atoms (1-2 interaction)")
    parser.add_argument('--eps_12', default=1.8, type=float, help="epsilon for reactive atoms (1-2 interaction)")
    parser.add_argument('--r_eq_13_scaling', default=0.5, type=float, help="r_eq scaling for 1-3 interaction across reactive atoms")
    parser.add_argument('--r_eq_14_scaling', default=0.5, type=float, help="r_eq scaling for 1-4 interaction across reactive atoms")
    args = parser.parse_args()

    if (args.pdb is None) == (args.pdbqt is None):
        msg = "need either --pdb or --pdbqt"
        print("Command line error: " + msg, file=sys.stderr)
        sys.exit(2)
    if (args.pdbqt is not None) and (len(args.flexres) == 0):
        msg = "nothing to do when reading --pdbqt and no --flexres."
        print("Command line error: " + msg, file=sys.stderr)
        sys.exit(2)
    return args

args = get_args()

reactive_atom = {
    "SER":  "OG", "LYS":  "NZ", "TYR":  "OH", "CYS":  "SG", "HIE": "NE2",
    "HID": "ND1", "GLU": "OE2", "THR": "OG1", "MET":  "SD",
}
modified_resnames = set()
for react_name_str in args.reactive_name:
    (resname, name), ok, err = parse_resname_and_name(react_name_str)
    if ok:
        if resname in modified_resnames:
            print("Command line error: repeated resname %s passed to --reactive_resname" % resname + os_linesep, file=sys.stderr)
            sys.exit(2)
        modified_resnames.add(resname)
        reactive_atom[resname] = name
    else:
        print("Error in parsing --reactive_name argument" + os_linesep, file=sys.stderr)
        print(err, file=sys.stderr)
        sys.exit(2)

reactive_flexres = {}
all_ok = True
all_err = ""
for resid_string in args.reactive_flexres:
    res_id, ok, err = parse_residue_string(resid_string) 
    if ok:
        resname = res_id[1]
        if resname in reactive_atom:
            reactive_flexres[res_id] = reactive_atom[resname]
        else:
            all_ok = False
            all_err += "no default reactive name for %s, " % resname 
            all_err += "use --reactive_name or --reactive_name_specific" + os_linesep
    all_ok &= ok
    all_err += err

for string in args.reactive_name_specific:
    out, ok, err = parse_residue_string_and_name(string) 
    if ok:
        # override name if res_id was also passed to --reactive_flexres
        reactive_flexres[out["res_id"]] = out["name"]
    all_ok &= ok
    all_err += err

if len(reactive_flexres) > 8: 
    msg = "got %d reactive_flexres but maximum is 8." % (len(args.reactive_flexres))
    print("Command line error: " + msg, file=sys.stderr)
    sys.exit(2)

all_flexres = set()
for resid_string in args.flexres:
    res_id, ok, err = parse_residue_string(resid_string) 
    all_ok &= ok
    all_err += err
    if ok:
        if res_id in reactive_flexres:
            all_err += "Command line error: can't determine if %s is reactive or not." % str(res_id) + os_linesep
            all_err += "Do not pass %s to --flexres if it is reactive." % str(res_id) + os_linesep
            all_ok = False
        all_flexres.add(res_id)

if all_ok:
    all_flexres = all_flexres.union(reactive_flexres)
else:
    print("Error:", file=sys.stderr)
    print(all_err, file=sys.stderr)
    sys.exit(2)

if len(all_flexres) > 0:
    print()
    print("Flexible residues:")
    print("chain resname resnum is_reactive reactive_atom")
    string = "%5s%8s%7d%12s%14s"
    for res_id in all_flexres:
        chain, resname, resnum = res_id
        is_react = res_id in reactive_flexres
        if is_react:
            react_atom = reactive_flexres[res_id]
        else:
            react_atom = ""
        print(string % (chain, resname, resnum, is_react, react_atom))
    
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

if len(all_flexres) == 0:
    with open(args.output_filename, "w") as f:
        f.write(pdbqtstring)
else:
    receptor = Receptor(pdbqtstring)
    pdbqt, success, error_msg = receptor.write_pdbqt_string(flex_res=all_flexres)
    if not success:
        print("Error: " + error_msg, file=sys.stderr)
        sys.exit(2)
    all_flex_pdbqt = ""
    reactive_flexres_count = 0
    for res_id, flexres_pdbqt in pdbqt["flex"].items():
        if res_id in reactive_flexres:
            reactive_flexres_count += 1
            prefix_atype = "%d" % reactive_flexres_count
            resname = res_id[1]
            reactive_atom = reactive_flexres[res_id]
            flexres_pdbqt = Receptor.make_flexres_reactive(flexres_pdbqt, reactive_atom, resname, prefix_atype)
        all_flex_pdbqt += flexres_pdbqt

    outpath = pathlib.Path(args.output_filename)
    rigid_fn = str(outpath.with_suffix("")) + "_rigid" + outpath.suffix
    flex_fn = str(outpath.with_suffix("")) + "_flex" + outpath.suffix
    with open(rigid_fn, "w") as f:
        f.write(pdbqt["rigid"]) 
    with open(flex_fn, "w") as f:
        f.write(all_flex_pdbqt)

    # configuration info for AutoDock-GPU reactive docking
    if len(reactive_flexres) > 0:
        any_lig_base_types = ["HD", "C", "A", "N", "NA", "OA", "F", "P", "SA",
                              "S", "Cl", "CL", "Br", "BR", "I", "Si", "B"]
        any_lig_reac_types = []
        for order in (1, 2, 3):
            for t in any_lig_base_types:
                any_lig_reac_types.append(reactive_typer.get_reactive_atype(t, order))

        rec_reac_types = []
        for line in all_flex_pdbqt.split(os_linesep):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                atype = line[77:].strip()
                basetype, _ = reactive_typer.get_basetype_and_order(atype) 
                if basetype is not None: # is None if not reactive
                    rec_reac_types.append(line[77:].strip())

        derivtypes, modpairs, collisions = get_reactive_config(
                                        any_lig_reac_types,
                                        rec_reac_types,
                                        args.r_eq_12,
                                        args.eps_12,
                                        args.r_eq_13_scaling,
                                        args.r_eq_14_scaling)

        if len(collisions) > 0:
            collision_fn = str(outpath.with_suffix(".atype_collisions"))
            collision_str = ""
            for t1, t2 in collisions:
                collision_str += "%3s %3s" % (t1, t2) + os_linesep
            with open(collision_fn, "w") as f:
                f.write(collision_str)
            print()
            print("%d type pairs may lead to intra-molecular reactions. These were written to '%s'" % (
                len(collisions), collision_fn))
            print()

        # in modpairs (dict): types are keys, parameters are values
        # now we will write a configuration file with nbp keywords
        # that AD-GPU reads using the --import_dpf flag
        # nbp stands for "non-bonded potential" or "non-bonded pairwise"
        line = "intnbp_r_eps %8.6f %8.6f %3d %3d %4s %4s" + os_linesep
        config = ""
        nbp_count = 0
        for (t1, t2), param in modpairs.items():
            config += line % (param["r_eq"], param["eps"], param["n"], param["m"], t1, t2)
            nbp_count += 1
        config_fn = str(outpath.with_suffix(".reactive_nbp"))
        with open(config_fn, "w") as f:
            f.write(config)
        print()
        print("Wrote %d non-bonded reactive pairs to file '%s'." % (nbp_count, config_fn))
        print("Use the following option with AutoDock-GPU:")
        print("    --import_dpf %s" % (config_fn))
        print()

        derivtype_list = []
        new_type_count = 0
        for basetype, reactypes in derivtypes.items():
            s = ",".join(reactypes) + "=" + basetype
            derivtype_list.append(s)
            new_type_count += len(reactypes)
        if len(derivtype_list) > 0:
            derivtype_fn = str(outpath.with_suffix(".derivtype"))
            config_str = "--derivtype " + "/".join(derivtype_list)
            with open(derivtype_fn, "w") as f:
                f.write(config_str + os_linesep)
            print("AutoDock-GPU will need to derive %d reactive types from standard atom types." % new_type_count)
            print("The required --derivtype command has been written to '%s'. " % derivtype_fn)
            print()
