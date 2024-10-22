#!/usr/bin/env python

import argparse
import json
import math
from os import linesep as os_linesep
import pathlib
import sys

import numpy as np

from meeko.reactive import atom_name_to_molsetup_index, assign_reactive_types_by_index
from meeko import PDBQTMolecule
from meeko import RDKitMolCreate
from meeko import MoleculePreparation
from meeko import MoleculeSetup
from meeko import ResidueChemTemplates
from meeko import PDBQTWriterLegacy
from meeko import LinkedRDKitChorizo
from meeko import LinkedRDKitChorizoEncoder
from meeko import ChorizoCreationError
from meeko import reactive_typer
from meeko import get_reactive_config
from meeko import gridbox
from meeko import __file__ as pkg_init_path
from rdkit import Chem

try:
    import prody
except ImportError as import_error:
    _prody_import_error = import_error
    _got_prody = False
else:
    SUPPORTED_PRODY_FORMATS = {"pdb": prody.parsePDB, "cif": prody.parseMMCIF}
    _got_prody = True

path_to_this_script = pathlib.Path(__file__).resolve()

pkg_dir = pathlib.Path(pkg_init_path).parents[0]
templates_fn = str(pkg_dir / "data" / "residue_chem_templates.json")
with open(templates_fn) as f:
    res_chem_templates = json.load(f)


def parse_cmdline_res(string):
    """ "A:5,7,BB:12C  ->  "A:5", "A:7", "BB:12C" """
    blocks = ("," + string).split(":")
    nr_blocks = len(blocks) - 1
    keys = []
    for i in range(nr_blocks):
        chain = blocks[i].split(",")[-1]
        if i + 1 == nr_blocks:
            resnums = blocks[i + 1].split(",")
        else:
            resnums = blocks[i + 1].split(",")[:-1]
        if len(resnums) == 0:
            raise ValueError(f"missing residue in {resnums}")
        for resnum in resnums:
            keys.append(f"{chain}:{resnum}")
    return keys


def parse_cmdline_res_assign(string):
    """convert "A:5,7=CYX,A:19A,B:17=HID" to {"A:5": "CYX", "A:7": "CYX", ":19A": "HID"}"""

    output = {}
    nr_assignments = string.count("=")
    string = "," + string  # enables `residues =` below to work in first iteraton
    tmp = string.split("=")
    for i in range(nr_assignments):
        residues = tmp[i].split(",")[1:]
        assigned_name = tmp[i + 1].split(",")[0]
        chain = ""
        for residue in residues:
            fields = residue.split(":")
            if len(fields) == 1:
                resnum = fields[0]
            elif len(fields) == 2:
                chain = fields[0]
                resnum = fields[1]
            else:
                raise ValueError(f"too many : in {residue}")
            if len(resnum) == 0:
                raise ValueError(f"missing residue in {residues}")
            key = f"{chain}:{resnum}"
            if key in output:
                raise ValueError(f"repeated {key} in {residue}")
            output[key] = assigned_name
    return output


class TalkativeParser(argparse.ArgumentParser):
    def error(self, message):
        """overload to print_help for every error"""
        self.print_help()
        this_script = path_to_this_script.name
        print("\n%s: error: %s" % (this_script, message), file=sys.stderr)
        sys.exit(2)


def check(success, error_msg):
    if not success:
        print("Error: " + error_msg, file=sys.stderr)
        sys.exit(2)

def required_length(nmin, nmax):
    class RequiredLength(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = "fargument {self.dest} requires between"
                msg += " {nmin} and {nmax} arguments"
                raise argparse.ArgumentTypeError(msg)
            setattr(namespace, self.dest, values)
    return RequiredLength

def get_args():
    parser = TalkativeParser()

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--read_pdb",
        metavar="PDB_FILENAME",
        help="reads PDB, not PDBQT, and does not use ProDy",
    )
    need_prody_msg = ""
    # if prody is not installed, the help message is extended to tell
    # the user how to install prody
    if not _got_prody:
        need_prody_msg = " which can be installed from PyPI or conda-forge."
    io_group.add_argument(
        "-i",
        "--read_with_prody",
        metavar="MACROMOL_FILENAME",
        help=f"reads PDB/mmCIF file with Prody{need_prody_msg}")
    io_group.add_argument(
        "-o",
        "--output_basename",
        help="default basename for --write options used without filename",
    )
    io_group.add_argument(
        "-p", "--write_pdbqt",
        metavar="PDBQT_FILENAME",
        nargs="*",
        help="adds _rigid/_flex with flexible residues (filename defaults to --output_basename when not specified)",
    )
    io_group.add_argument(
        "-j", "--write_json",
        metavar="JSON_FILENAME",
        help="parameterized receptor (filename defaults to --output_basename when not specified)",
        nargs="*",
        action=required_length(0, 1))

    io_group.add_argument(
        "--write_pdb",
        help="prepared receptor (must specify filename)",
        metavar="PDB_FILENAME",
    )
    io_group.add_argument(
        "-g",
        "--write_gpf",
        metavar="GPF_FILENAME",
        help="autogrid input file (filename defaults to --output_basename when not specified)",
        nargs="*",
        action=required_length(0, 1))
    io_group.add_argument(
        "-v", "--write_vina_box",
        metavar="VINA_BOX_FILENAME",
        help="config file for Vina with box dimensions (filename defaults to --output_basename when not specified_",
        nargs="*",
        action=required_length(0, 1))


    config_group = parser.add_argument_group("Receptor perception")
    config_group.add_argument("-n", "--set_template", help="e.g. A:5,7=CYX,B:17=HID")
    config_group.add_argument("-d", "--delete_residues", help="e.g. A:350,B:15,16,17")
    config_group.add_argument("-b", "--blunt_ends", help="e.g. A:123,200=2,A:1=0")
    config_group.add_argument("--add_templates", help="[.json]", metavar="JSON_FILENAME")
    config_group.add_argument("--mk_config", help="[.json]", metavar="JSON_FILENAME")
    config_group.add_argument(
        "-a", "--allow_bad_res",
        action="store_true",
        help="delete residues with missing atoms instead of raising error",
    )
    config_group.add_argument("--default_altloc", help="default alternate location (overridden by --wanted_altloc)")
    config_group.add_argument("--wanted_altloc", help="require altloc for specific residues, e.g. :5=B,B:17=A")
    config_group.add_argument(
        "-f",
        "--flexres",
        action="append",
        default=[],
        help='specify the flexible residues by the chain ID and residue number, e.g. -f ":42,B:23" is equivalent to -f ":42" -f "B:23" (skip chain ID if it is unspecified)',
    )

    box_group = parser.add_argument_group("Size and center of grid box")

    box_group.add_argument(
        "--box_size", help="size of grid box (x, y, z) in Angstrom", nargs=3, type=float,
        metavar=("X", "Y", "Z"),
    )
    box_group.add_argument(
        "--box_center",
        help="center of grid box (x, y, z) in Angstrom",
        nargs=3,
        metavar=("X", "Y", "Z"),
        type=float,
    )
    box_group.add_argument(
        "--box_center_off_reactive_res",
        help="project grid box center along CA-CB bond 5 A away from CB (only applicable when there is exactly one reactive flexible residue)",
        action="store_true",
    )
    box_group.add_argument(
        "--box_enveloping",
        metavar="FILENAME",
        help="Box will envelop atoms in this file [.sdf .mol .mol2 .pdb .pdbqt]",
    )
    box_group.add_argument(
        "--padding", help="padding around atoms passed to --box_enveloping [A]", type=float
    )

    reactive_group = parser.add_argument_group("Reactive")
    reactive_group.add_argument(
        "-r",
        "--reactive_flexres",
        action="append",
        default=[],
        help='same as --flexres but for reactive residues (max 8)',
    )
    reactive_group.add_argument(
        "--reactive_name",
        action="append",
        default=[],
        help="set name of reactive atom of a residue type, e.g: --reactive_name 'TRP:NE1'. Repeat flag for multiple assignments. Overridden by --reactive_name_specific",
    )
    reactive_group.add_argument(
        "-s",
        "--reactive_name_specific",
        action="append",
        default=[],
        help="set name of reactive atom for an individual residue by the residue ID, e.g: -s 'A:42=NE2'. Residue will be reactive.",
    )

    reactive_group.add_argument(
        "--r_eq_12",
        default=1.8,
        type=float,
        help="r_eq for reactive atoms (1-2 interaction)",
    )
    reactive_group.add_argument(
        "--eps_12",
        default=2.5,
        type=float,
        help="epsilon for reactive atoms (1-2 interaction)",
    )
    reactive_group.add_argument(
        "--r_eq_13_scaling",
        default=0.5,
        type=float,
        help="r_eq scaling for 1-3 interaction across reactive atoms",
    )
    reactive_group.add_argument(
        "--r_eq_14_scaling",
        default=0.5,
        type=float,
        help="r_eq scaling for 1-4 interaction across reactive atoms",
    )
    args = parser.parse_args()
    
    if args.read_pdb is None and args.read_with_prody is None:
        parser.print_help()
        msg = "Need input filename: use either -i/--read_with_prody or --read_pdb"
        print(os_linesep + msg)
        sys.exit(2)

    if args.read_pdb is not None and args.read_with_prody is not None:
        msg = "Can't use both -i/--read_with_prody and --read_pdb"
        print(os_linesep + msg, file=sys.stderr)
        sys.exit(2)

    if args.write_gpf is not None and args.write_pdbqt is None:
        # there's a few of places that assume this condition has been checked
        msg = "--write_gpf requires --write_pdbqt because autogrid expects"
        msg += " the GPF file to point to the PDBQT file." 
        print(os_linesep + msg)
        sys.exit(2)

    skip_gpf = args.write_gpf is None and args.write_vina_box is None
    if not skip_gpf:

        box_help = f"""
    writing a grid parameter file (--write_gpf) or a config file with the
    box dimensions for vina (-v/--write_vina_box) requires setting the box
    center and size with one of the following three combinations:
    1) --box_center and --box_size
    2) --box_center_off_reactive_res and --box_size
    3) --box_enveloping and --padding"""

        # Ensure correct number of box specs
        nr_boxcenter_specs = sum(
            [
                (args.box_center is not None),
                (args.box_center_off_reactive_res),
                (args.box_enveloping is not None),
            ]
        )
        nr_boxsize_specs = sum(
            [(args.box_size is not None), (args.padding is not None)]
        )

        box_specs = [(nr_boxcenter_specs, "box center"), (nr_boxsize_specs, "box size")]

        for spec_count, spec_type in box_specs:
            if spec_count > 1:
                msg = f"{spec_type} can't be specified in more than once. {box_help}"
                print("Command line error: " + msg, file=sys.stderr)
                sys.exit(2)
            elif spec_count < 1:
                msg = (
                    f"missing {spec_type} to write .gpf file for autogrid4. {box_help}"
                )
                print("Command line error: " + msg, file=sys.stderr)
                sys.exit(2)

        # Ensure correct combinations of box specs
        if args.box_size is None:
            if args.box_center_off_reactive_res:
                msg = f"--box_center_off_reactive_res requires --box_size. {box_help}"
                print("Command line error: " + msg, file=sys.stderr)
                sys.exit(2)
            elif args.box_center is not None:
                msg = f"--box_center requires --box_size. {box_help}"
                print("Command line error: " + msg, file=sys.stderr)
                sys.exit(2)

        if (args.padding is None) != (args.box_enveloping is None):
            msg = f"--padding and --box_enveloping must be used together. {box_help}"
            print("Command line error: " + msg, file=sys.stderr)
            sys.exit(2)

    return args


args = get_args()

if args.wanted_altloc is None:
    wanted_altloc = None
else:
    wanted_altloc = parse_cmdline_res_assign(args.wanted_altloc)
    # Ensure meaningful wanted_altloc
    for key, value in wanted_altloc.items():
        if isinstance(value, str) and value.strip() == "":
            msg = "Wanted atloc cannot be an empty string or a string with just space"
            print("Command line error: " + msg, file=sys.stderr)
            sys.exit(2)


# Ensure meaningful default_altloc
if args.default_altloc is not None and args.default_altloc.strip()=="":
    msg = "Allowed atloc cannot be an empty string or a string with just space"
    print("Command line error: " + msg, file=sys.stderr)
    sys.exit(2)

# check write options have default if used without argument
write_flags = [
    args.write_pdbqt,
    args.write_json,
    args.write_gpf,
    args.write_vina_box,
]
needed_default = False
for flag in write_flags:
    # flag is none if not used, and is empty list when used without arg
    if flag is not None and len(flag) == 0:
        needed_default = True
        break
if needed_default and args.output_basename is None:
    msg = "--write flags require either a filename argument or"
    msg += " --output_basename to set a default"
    print(msg)
    sys.exit(2)

# Default mapping of residue name and reactive atom name
reactive_atom = {
    "SER": "OG",
    "LYS": "NZ",
    "TYR": "OH",
    "CYS": "SG",
    "HIE": "NE2",
    "HID": "ND1",
    "GLU": "OE2",
    "THR": "OG1",
    "MET": "SD",
}

# Process custom mapping of residue name and reactive atom name
modified = set()
for react_name_str in args.reactive_name:
    resname, name = react_name_str.split(":")
    if resname in modified:
        print(
            "Command line error: repeated resname %s passed to --reactive_resname"
            % resname
            + os_linesep,
            file=sys.stderr,
        )
        sys.exit(2)
    modified.add(resname)
    reactive_atom[resname] = name

# Process specified mapping of residue ID and reactive atom name
modified = set()
reactive_flexres_name = {}
for string in args.reactive_name_specific:
    res_assign = parse_cmdline_res_assign(string)
    for res_id in res_assign:
        if res_id in modified:
            print(
                "Command line error: repeated resid %s passed to --reactive_name_specific"
                % res_id
                + os_linesep,
                file=sys.stderr,
            )
            sys.exit(2)
        modified.add(res_id)
        reactive_flexres_name[res_id] = res_assign[res_id]

# Process residue ID of reactive flexible residues without specified reactive atom
reactive_flexres = set(reactive_flexres_name)
for resid_string in args.reactive_flexres:
    res_list = parse_cmdline_res(resid_string)
    for res_id in res_list:
        if res_id not in reactive_flexres:
            reactive_flexres.add(res_id)
            reactive_flexres_name[res_id] = ""

# Evaluate number of reactive flexible residues
if len(reactive_flexres) > 8:
    msg = "got %d reactive_flexres but maximum is 8." % (len(args.reactive_flexres))
    print("Command line error: " + msg, file=sys.stderr)
    sys.exit(2)

# Evaluate compatibility with other options
if len(reactive_flexres) != 1 and args.box_center_off_reactive_res:
    msg = (
        "--box_center_off_reactive_res can be used only with one reactive" + os_linesep
    )
    msg += "residue, but %d reactive residues are set" % len(reactive_flexres_name)
    print("Command line error:" + msg, file=sys.stderr)
    sys.exit(2)

# Process residue ID of nonreactive flexible residues
nonreactive_flexres = set()
for string in args.flexres:
    for res_id in parse_cmdline_res(string):
        if res_id not in reactive_flexres:
            nonreactive_flexres.add(res_id)


set_template = {}
if args.set_template is not None:
    set_template = parse_cmdline_res_assign(args.set_template)

blunt_ends = []
if args.blunt_ends is not None:
    j = parse_cmdline_res_assign(args.blunt_ends)
    # TODO parse also input/raw atom names, easier than indices
    j = [(k, int(v)) for k, v in j.items()]
    blunt_ends.extend(j)

delete_residues = []
if args.delete_residues is not None:
    delete_residues = parse_cmdline_res(args.delete_residues)

if args.mk_config is not None:
    with open(args.mk_config) as f:
        mk_config = json.load(f)
    mk_prep = MoleculePreparation.from_config(mk_config)
else:
    mk_prep = MoleculePreparation()

# load templates for mapping
if args.add_templates is not None:
    with open(args.add_templates) as f:
        more_templates = json.load(f)
    bad_keys = set()
    for key, values in more_templates.items():
        if key not in res_chem_templates:
            bad_keys.add(key)
        else:
            res_chem_templates[key].update(values)
    if len(bad_keys):
        msg = "Unrecognized keys provided in add_templates:" + os_linesep
        msg += f"{bad_keys=}" + os_linesep
        msg += f"allowed keys: {res_chem_templates.keys()}" + os_linesep
        msg += f"see template file: {templates_fn}" + os_linesep
        print(msg, file=sys.stderr)
        sys.exit(2)
templates = ResidueChemTemplates.from_dict(res_chem_templates)

# create chorizos
if args.read_with_prody is not None:
    if not _got_prody:
        print(_prody_import_error, file=sys.stderr)
        print("option --read_with_prody requires Prody, which is not installed.")
        print("Installable from PyPI (pip install prody) or conda-forge (micromamba install prody)")
        sys.exit(2)
    ext = pathlib.Path(args.read_with_prody).suffix[1:].lower()
    if ext in SUPPORTED_PRODY_FORMATS:
        parser = SUPPORTED_PRODY_FORMATS[ext]
        input_obj = parser(args.read_with_prody, altloc="all")
        try:
            chorizo = LinkedRDKitChorizo.from_prody(
                input_obj,
                templates,
                mk_prep,
                set_template,
                delete_residues,
                args.allow_bad_res,
                blunt_ends=blunt_ends,
                wanted_altloc=wanted_altloc,
                default_altloc=args.default_altloc,
            )
        except ChorizoCreationError as e:
            print(e)
            sys.exit(1)
else:
    with open(args.read_pdb) as f:
        pdb_string = f.read()
    try:
        chorizo = LinkedRDKitChorizo.from_pdb_string(
            pdb_string,
            templates,  # residue_templates, padders, ambiguous,
            mk_prep,
            set_template,
            delete_residues,
            args.allow_bad_res,
            blunt_ends=blunt_ends,
            wanted_altloc=wanted_altloc,
            default_altloc=args.default_altloc,
        )
    except ChorizoCreationError as e:
        print(e)
        sys.exit(1)


# Use residue name in the input structure file to find reactive atom name
# According to the mapping of residue name and reactive atom name
for res_id in reactive_flexres:
    if res_id not in chorizo.residues:
        print("resid %s not found in input receptor file" % res_id)
        sys.exit(2)
    res_atom = reactive_flexres_name[res_id]
    if not res_atom:
        input_resname = chorizo.residues[res_id].input_resname
        if input_resname in reactive_atom:
            reactive_flexres_name[res_id] = reactive_atom[input_resname]
        else:
            print("no default reactive name for %s, " % input_resname)
            print("use --reactive_name or --reactive_name_specific" + os_linesep)
            sys.exit(2)

# Print nonreactive and reactive flexible residues specs
if len(nonreactive_flexres) + len(reactive_flexres) > 0:
    print()
    print("Flexible residues:")
    print("chain resnum is_reactive reactive_atom")
    string = "%5s%7s%12s%14s"

    if len(nonreactive_flexres) > 0:
        for res_id in nonreactive_flexres:
            chain, resnum = res_id.split(":")
            react_atom = ""
            print(string % (chain, resnum, False, react_atom))

    if len(reactive_flexres) > 0:
        for res_id in reactive_flexres_name:
            chain, resnum = res_id.split(":")
            react_atom = reactive_flexres_name[res_id]
            print(string % (chain, resnum, True, react_atom))

# Assign reactive atom types for atoms in reactive flexible residues
reactive_prefix = 1
for res_id in reactive_flexres:
    # get reactive atom types
    reactive_aname = reactive_flexres_name[res_id]
    reactive_atomi = atom_name_to_molsetup_index(
        chorizo.residues[res_id], reactive_aname
    )
    if reactive_atomi is None:
        print(f"cannot find reactive atom name {reactive_aname} from residue {res_id} in input receptor file")
        sys.exit(2)
    reactive_atypes = assign_reactive_types_by_index(chorizo.residues[res_id].molsetup, reactive_atomi)
    # set reactive atom types
    nr_atom = len(chorizo.residues[res_id].molsetup.atoms)
    for atom_index in range(nr_atom):
        if (
            chorizo.residues[res_id].molsetup.atoms[atom_index].atom_type
            != reactive_atypes[atom_index]
        ):
            chorizo.residues[res_id].molsetup.atoms[
                atom_index
            ].atom_type = f"{reactive_prefix}{reactive_atypes[atom_index]}"
    reactive_prefix += 1

# Combine nonreactive and reactive flexible residues into one set
all_flexres = nonreactive_flexres.union(reactive_flexres)

for res_id in all_flexres:
    chorizo.flexibilize_sidechain(res_id, mk_prep)


any_lig_base_types = [
    "HD",
    "C",
    "A",
    "N",
    "NA",
    "OA",
    "F",
    "P",
    "SA",
    "S",
    "Cl",
    "CL",
    "Br",
    "BR",
    "I",
    "Si",
    "B",
]

if args.output_basename is not None:
    outpath = pathlib.Path(args.output_basename)

written_files_log = {"filename": [], "description": []}

if args.write_json is not None:
    if args.write_json:
        fn = args.write_json[0]
    else:  # args.write_json is empty list (was used without arg)
        fn = str(outpath) + ".json"
    with open(fn, "w") as f:
        json.dump(chorizo, f, cls=LinkedRDKitChorizoEncoder)
    written_files_log["filename"].append(fn)
    written_files_log["description"].append("parameterized receptor")

if args.write_pdb is not None:
    fn = args.write_pdb[0]
    with open(fn, "w") as f:
        f.write(chorizo.to_pdb())
    written_files_log["filename"].append(fn)
    written_files_log["description"].append("receptor")

if args.write_pdbqt is not None:
    if args.write_pdbqt:
        if args.write_pdbqt[0].endswith(".pdbqt"):
            # may need to suffix _rigid/_flex with flexres
            fn_base = str(pathlib.Path(args.write_pdbqt[0]).with_suffix(""))
        else:
            fn_base = args.write_pdbqt[0]
    else:
        fn_base = str(outpath)

    pdbqt_tuple = PDBQTWriterLegacy.write_from_linked_rdkit_chorizo(chorizo)
    rigid_pdbqt, flex_pdbqt_dict = pdbqt_tuple

    if len(all_flexres) == 0:
        box_center = args.box_center
        rigid_fn = fn_base + ".pdbqt"
        flex_fn = None
    else:
        print(f"{reactive_flexres=}")
        all_flex_pdbqt = ""
        reactive_flexres_count = 0
        for res_id, flexres_pdbqt in flex_pdbqt_dict.items():
            all_flex_pdbqt += flexres_pdbqt
    
        rigid_fn = fn_base + "_rigid.pdbqt"
        flex_fn = fn_base + "_flex.pdbqt"
    
        written_files_log["filename"].append(flex_fn)
        written_files_log["description"].append("flexible receptor input file")
        with open(flex_fn, "w") as f:
            f.write(all_flex_pdbqt)
    
    written_files_log["filename"].append(rigid_fn)
    written_files_log["description"].append("static (i.e., rigid) receptor input file")
    with open(rigid_fn, "w") as f:
        f.write(rigid_pdbqt)

def warn_flexres_outside_box(chorizo, box_center, box_size):
    for res_id, res in chorizo.residues.items():
        if not res.is_movable:
            continue
        for atom in res.molsetup.atoms:
            if not res.is_flexres_atom[atom.index]:
                continue
            if gridbox.is_point_outside_box(atom.coord, box_center, box_size, spacing=1.0):
                print(
                    "WARNING: Flexible residue outside box." + os_linesep,
                    file=sys.stderr,
                )
                print(
                    "WARNING: Strongly recommended to use a box that encompasses flexible residues."
                    + os_linesep,
                    file=sys.stderr,
                )
                break  # only need to warn once

skip_gpf = args.write_gpf is None and args.write_vina_box is None
if not skip_gpf:
    if args.box_center is not None:
        box_center = args.box_center
        box_size = args.box_size
    elif args.box_center_off_reactive_res:
        # we have only one reactive residue and will set the box center
        # to be 5 Angstromg away from CB along the CA->CB vector
        box_centers = []
        for res_id in reactive_flexres:
            molsetup = chorizo.residues[res_id].molsetup
            calpha_idx = [
                atom.index for atom in molsetup.atoms if atom.pdbinfo.name == "CA"
            ]
            cbeta_idx = [
                atom.index for atom in molsetup.atoms if atom.pdbinfo.name == "CB"
            ]
            if len(calpha_idx) != 1:
                check(
                    success=False,
                    error_msg=f"found {len(calpha_idx)} CA in {res_id} but expected 1",
                )
            if len(cbeta_idx) != 1:
                check(
                    success=False,
                    error_msg=f"found {len(cbeta_idx)} CB in {res_id} but expected 1",
                )
            calpha_idx = calpha_idx[0]
            cbeta_idx = cbeta_idx[0]
            ca = molsetup.get_coord(calpha_idx)
            cb = molsetup.get_coord(cbeta_idx)
            v = cb - ca
            v /= math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-8
            box_center = ca + 5 * v
            box_centers.append(box_center)
        box_center = np.mean(box_centers, 0)
        box_size = args.box_size
    elif args.box_enveloping is not None:
        ft = pathlib.Path(args.box_enveloping).suffix
        suppliers = {
            ".pdb": Chem.MolFromPDBFile,
            ".mol": Chem.MolFromMolFile,
            ".mol2": Chem.MolFromMol2File,
            ".sdf": Chem.SDMolSupplier,
            ".pdbqt": None,
        }
        if ft not in suppliers.keys():
            check(
                success=False,
                error_msg=f"Given --box_enveloping file type {ft} not readable!"
            )
        elif ft != ".sdf" and ft != ".pdbqt":
            ligmol = suppliers[ft](args.box_enveloping, removeHs=False, sanitize=False)
        elif ft == ".sdf":
            ligmol = suppliers[ft](args.box_enveloping, removeHs=False, sanitize=False)[
                0
            ]  # assume we only want first molecule in file
        else:  # .pdbqt
            ligmol = RDKitMolCreate.from_pdbqt_mol(
                PDBQTMolecule.from_file(args.box_enveloping)
            )[
                0
            ]  # assume we only want first molecule in file

        box_center, box_size = gridbox.calc_box(
            ligmol.GetConformer().GetPositions(), args.padding
        )
    else:
        print("Error: No box center specified.", file=sys.stderr)
        sys.exit(2)


    if args.write_gpf is not None:
        if args.write_gpf:
            gpf_fn = args.write_gpf[0]
        else:
            gpf_fn = pathlib.Path(rigid_fn).with_suffix(".gpf")
        # write .dat parameter file for B and Si
        ff_fn = pathlib.Path(gpf_fn).parents[0] / pathlib.Path(
            "boron-silicon-atom_par.dat"
        )
        written_files_log["filename"].append(str(ff_fn))
        written_files_log["description"].append(
            "atomic parameters for B and Si (for autogrid)"
        )
        with open(ff_fn, "w") as f:
            f.write(gridbox.boron_silicon_atompar)

        rec_types = [
            "HD",
            "C",
            "A",
            "N",
            "NA",
            "OA",
            "F",
            "P",
            "SA",
            "S",
            "Cl",
            "Br",
            "I",
            "Mg",
            "Ca",
            "Mn",
            "Fe",
            "Zn",
        ]
        gpf_string, npts = gridbox.get_gpf_string(
            box_center,
            box_size,
            pathlib.Path(rigid_fn).name,  # requires --write_pdbqt
            rec_types,
            any_lig_base_types,
            ff_param_fname=ff_fn.name,
        )

        written_files_log["filename"].append(str(gpf_fn))
        written_files_log["description"].append("autogrid input file")
        with open(gpf_fn, "w") as f:
            f.write(gpf_string)

    # write gridbox vina format
    if args.write_vina_box is not None:
        if args.write_vina_box:
            box_vina_fn = args.write_vina_box[0]
        else:
            box_vina_fn = str(outpath) + ".box.txt"

        written_files_log["filename"].append(box_vina_fn)
        written_files_log["description"].append("Vina-style box dimension file")
        with open(box_vina_fn, "w") as f:
            f.write(gridbox.box_to_vina_string(box_center, box_size))

    # write a PDB for the box
    if args.write_vina_box is not None or args.write_gpf is not None:
        if args.output_basename is not None:
            box_fn = str(outpath) + ".box.pdb" 
        elif args.write_gpf is not None:
            # relies on --write_gpf forcing --write_pdbqt which sets rigid_fn
            box_fn = str(pathlib.Path(rigid_fn).with_suffix(".box.pdb"))
        else:
            # suffix .pdb even if box_vina_fn does not end with ".txt"
            box_fn = box_vina_fn.replace(".txt", "") + ".pdb"
        written_files_log["filename"].append(box_fn)
        written_files_log["description"].append("PDB file to visualize the grid box")
        with open(box_fn, "w") as f:
            f.write(gridbox.box_to_pdb_string(box_center, box_size, spacing=1.0))

    warn_flexres_outside_box(chorizo, box_center, box_size)


# configuration info for AutoDock-GPU reactive docking
if len(reactive_flexres) > 0 and args.write_pdbqt is not None:
    any_lig_reac_types = []
    for order in (1, 2, 3):
        for t in any_lig_base_types:
            any_lig_reac_types.append(reactive_typer.get_reactive_atype(t, order))

    rec_reac_types = []
    for line in all_flex_pdbqt.split(os_linesep):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atype = line[77:].strip()
            basetype, _ = reactive_typer.get_basetype_and_order(atype)
            if basetype is not None:  # is None if not reactive
                rec_reac_types.append(line[77:].strip())

    derivtypes, modpairs, collisions = get_reactive_config(
        any_lig_reac_types,
        rec_reac_types,
        args.eps_12,
        args.r_eq_12,
        args.r_eq_13_scaling,
        args.r_eq_14_scaling,
    )

    if len(collisions) > 0:
        collision_str = ""
        for t1, t2 in collisions:
            collision_str += "%3s %3s" % (t1, t2) + os_linesep
        collision_fn = str(outpath.with_suffix(".atype_collisions"))
        written_files_log["filename"].append(collision_fn)
        written_files_log["description"].append(
            "type pairs (n=%d) that may lead to intra-molecular reactions"
            % len(collisions)
        )
        with open(collision_fn, "w") as f:
            f.write(collision_str)

    # The maps block is to tell AutoDock-GPU the base types for the reactive types.
    # This could be done with -T/--derivtypes, but putting derivtypes and intnbp
    # lines in a single configuration file simplifies the command line call.
    map_block = ""
    map_prefix = pathlib.Path(rigid_fn).with_suffix("").name
    all_types = []
    for basetype, reactypes in derivtypes.items():
        all_types.append(basetype)
        map_block += "map %s.%s.map" % (map_prefix, basetype) + os_linesep
        for reactype in reactypes:
            all_types.append(reactype)
            map_block += "map %s.%s.map" % (map_prefix, basetype) + os_linesep
    config = "ligand_types " + " ".join(all_types) + os_linesep
    config += "fld %s.maps.fld" % map_prefix + os_linesep
    config += map_block

    # in modpairs (dict): types are keys, parameters are values
    # now we will write a configuration file with nbp keywords
    # that AD-GPU reads using the --import_dpf flag
    # nbp stands for "non-bonded potential" or "non-bonded pairwise"
    line = "intnbp_r_eps %8.6f %8.6f %3d %3d %4s %4s" + os_linesep
    nbp_count = 0
    for (t1, t2), param in modpairs.items():
        config += line % (param["r_eq"], param["eps"], param["n"], param["m"], t1, t2)
        nbp_count += 1
    config_fn = str(outpath.with_suffix(".reactive_config"))
    written_files_log["filename"].append(config_fn)
    written_files_log["description"].append("reactive parameters for AutoDock-GPU")
    with open(config_fn, "w") as f:
        f.write(config)
    print()
    print("For reactive docking, pass the configuration file to AutoDock-GPU:")
    print(
        "    autodock_gpu -C 1 --import_dpf %s --flexres %s -L <ligand_filename>"
        % (config_fn, flex_fn)
    )
    print()

if written_files_log["filename"]:
    print()
    print("Files written:")
    longest_fn = max([len(fn) for fn in written_files_log["filename"]])
    line = "%%%ds <-- " % longest_fn + "%s"
    for fn, desc in zip(written_files_log["filename"], written_files_log["description"]):
        print(line % (fn, desc))
    if (
        args.output_basename is not None and
        args.output_basename.endswith(".pdbqt") and
        args.write_pdbqt is None
    ):
        print()
        print("PDBQT files were NOT written. Use -p/--write_pdbqt for that.")
        print("Note that -o/--output_basename just sets a default for --write flags")
        print()
else:
    print()
    print()
    print("Receptor was prepared, but no files were written.")
    print("")
    print("Consider the following --write options:")
    print("  -p/--write_pdbqt")
    print("  -j/--write_json")
    print("  -g/--write_gpf")
    print("  -v/--write_vina_box")
    print("")
    print("Use -o/--output_basename, or set a filename after each --write flag")
    print("")
    print("Recommended for AutoDock-GPU:")
    print("  -o my_receptor -p -j -g")
    print("")
    print("Recommended for AutoDock-Vina:")
    print("  -o my_receptor -p -j -v")
