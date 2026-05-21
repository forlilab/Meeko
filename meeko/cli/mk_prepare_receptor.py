#!/usr/bin/env python

import argparse
import logging
import json
eol = "\n"
import pathlib
import sys

from meeko.reactive import atom_name_to_molsetup_index, assign_reactive_types_by_index
from meeko.utils.utils import parse_cmdline_res
from meeko.utils.utils import parse_cmdline_res_assign
from meeko import MoleculePreparation
from meeko import ResidueChemTemplates
from rdkit import Chem
import meeko

try:
    import prody
except ImportError as import_error:
    _prody_import_error = import_error
    _got_prody = False
else:
    SUPPORTED_PRODY_FORMATS = {"pdb": prody.parsePDB, "cif": prody.parseMMCIF}
    _got_prody = True

path_to_this_script = pathlib.Path(__file__).resolve()
pkg_dir = pathlib.Path(meeko.__file__).parents[0]
mk_config_dir = pkg_dir / "data" / "mk_config"


def sdf_to_json(sdf_path: str, resname: str) -> dict:
    """Convert an SDF file into a residue template JSON."""

    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mol = Chem.AddHs(mol)  # ensure explicit Hs
    smiles = Chem.MolToSmiles(mol)
    atom_names = [str(i) for i in range(mol.GetNumAtoms())]

    return {
        "ambiguous": {resname: [resname]},
        "residue_templates": {
            resname: {
                "smiles": smiles,
                "atom_name": atom_names,
                "link_labels": {}
            }
        }
    }


from ._common import check, make_talkative_parser, required_length
from ._receptor_helpers import (
    ANY_LIG_BASE_TYPES,
    build_mk_config,
    build_polymer,
    print_write_summary,
    resolve_box,
    resolve_residue_selections,
    validate_altloc_and_write_flags,
    warn_flexres_outside_box,
    write_gpf_and_vina_outputs,
    write_json_output,
    write_pdb_output,
    write_pdbqt_output,
    write_reactive_config,
)

# Backward-compat: third-party code may import TalkativeParser from this module.
TalkativeParser = make_talkative_parser(path_to_this_script)

def get_args():
    parser = TalkativeParser()

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        "--read_pdb",
        metavar="PDB_FILENAME",
        help="reads PDB, not PDBQT, and does not use ProDy",
    )
    io_group.add_argument(
        "--read_json",
        metavar="JSON_FILENAME",
        help="reads json receptor, probably prepared by meeko. Existing parameters and flexres are lost.",
    )
    io_group.add_argument(
        "--read_pqr",
        metavar="PQR_FILENAME",
        help="reads PQR and does not use ProDy",
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
        nargs="*",
        metavar="PDB_FILENAME",
    )
    io_group.add_argument(
        "--ignore_https_cert",
        action="store_true",
        help="Ignore https certificate errors when downloading from PDB database (potentially dangerous if rscb.org were spoofed, please only use as a last resort) ",
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
    io_group.add_argument(
        "--debug_fn",
        help="log debug level to filename",
    )

    config_group = parser.add_argument_group("Receptor perception")
    config_group.add_argument("-n", "--set_template", help="e.g. A:5,7=CYX,B:17=HID")
    config_group.add_argument("-d", "--delete_residues", help="e.g. A:350,B:15,16,17")
    config_group.add_argument("-b", "--blunt_ends", help="e.g. A:123,200=2,A:1=0")
    config_group.add_argument("--add_templates", help="Additional residue templates. Can be a JSON file path or 'resname:file.sdf'. Repeat --add_templates to add multiple files.", action="append", default=[])
    config_group.add_argument("--cache_templates", 
                              help=(
                                  "Turns on caching of ResidueChemTemplates (default is OFF) by this option and "
                                  "(optionally) a provided JSON filename. " 
                                  "Default cache filename is: $HOME/.meeko_residue_chem_templates_cached.json) "
                                  "When the caching is ON, the templates for polymer construction will be read from "
                                  "the specified cache file and updates may be made to the same file in a cumulative manner. " 
                              ), 
                              nargs = "?", 
                              default=False,
    )
    config_group.add_argument("--config_file", help="local json configuration file. Overrides --config_preset option-wise. Overriden by command line options.")
    config_group.add_argument("--config_preset", help="name of packaged configuration (choices: scofu1). Overriden by --config_file and by command line options.")
    config_group.add_argument(
        "-x", "--delete_bad_res",
        action="store_true",
        help="delete residues that don't match templates instead of raising error",
    )

    # keep -a/--allow_bad_res for backwards compatibility, superseeded by -x/--delete_bad_res
    config_group.add_argument("-a", "--allow_bad_res", action="store_true", help=argparse.SUPPRESS)

    config_group.add_argument("--default_altloc", help="default alternate location (overridden by --wanted_altloc)")
    config_group.add_argument("--wanted_altloc", help="require altloc for specific residues, e.g. :5=B,B:17=A")
    config_group.add_argument("--forgive_extra_bonds",
        action="store_true",
        help="allows processing clashed structures because templates match even with excess bonds to other residues at the expense of causing unpredictable problems and potentially matching incorrect templates")
    config_group.add_argument(
        "-f",
        "--flexres",
        action="append",
        default=[],
        help='specify the flexible residues by the chain ID and residue number, e.g. -f ":42,B:23" is equivalent to -f ":42" -f "B:23" (leave chain ID empty if omitted in input PDB or mmCIF)',
    )
    config_group.add_argument(
        "-t",
        "--rot_terminal_group",
        action="append",
        default=[],
        help='specify the residues for which to make terminal functional group rotatable by the chain ID and residue number, e.g. -t ":42,B:23" is equivalent to -t ":42" -t "B:23" (leave chain ID empty if omitted in input PDB or mmCIF)',
    )

    
    config_group.add_argument(
        "--compute_charges",
        help="compute charges from scratch with the given charge model instead of reading from template (note: this option is slower)",
        action="store_true",
    )

    config_group.add_argument(
        "--charge_model",
        choices=("gasteiger", "espaloma", "nagl", "zero", "read"),
        help="default is gasteiger, 'zero' sets all zeros, 'read' requires --read_pqr",
        default=None,
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

    if args.debug_fn:
        logger = logging.getLogger()
        logger.setLevel("DEBUG")
        formatter = logging.Formatter("%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s [%(name)s@%(filename)s:%(lineno)d]", datefmt='%Y-%m-%d %H:%M:%S')
        handler = logging.FileHandler(args.debug_fn)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.debug("Starting to log")
    
    num_input_flags = sum([flag is not None for flag in (args.read_pdb, args.read_pqr, args.read_with_prody, args.read_json)])

    if num_input_flags == 0:
        parser.print_help()
        msg = "Need input filename: use either -i/--read_with_prody, --read_pdb, --read_json, or --read_pqr"
        print(eol + msg)
        sys.exit(2)

    if num_input_flags > 1:
        msg = "Can't use more than one at a time from -i/--read_with_prody, --read_pdb, --read_json, and --read_pqr"
        print(eol + msg, file=sys.stderr)
        sys.exit(2)

    if args.cache_templates is not False:
        if args.cache_templates is None:
            print(f"--cache_templates is turned on, but a name is not provided. The default filename ($HOME/.meeko_residue_chem_templates_cached.json) will be used. ", 
                file=sys.stderr)
            default_cache_fn = ".meeko_residue_chem_templates_cached.json"
            args.cache_templates = str(pathlib.Path.home() / default_cache_fn)

    if args.write_gpf is not None and args.write_pdbqt is None:
        # there's a few of places that assume this condition has been checked
        msg = "--write_gpf requires --write_pdbqt because autogrid expects"
        msg += " the GPF file to point to the PDBQT file." 
        print(eol + msg)
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


def main():
    args = get_args()
    delete_bad_res = args.allow_bad_res or args.delete_bad_res

    validate_altloc_and_write_flags(args)

    wanted_altloc = (
        None
        if args.wanted_altloc is None
        else parse_cmdline_res_assign(args.wanted_altloc)
    )


    residues = resolve_residue_selections(args)
    reactive_atom = residues.reactive_atom_by_resname
    reactive_flexres = residues.reactive_flexres
    reactive_flexres_name = residues.reactive_flexres_name
    nonreactive_flexres = residues.nonreactive_flexres
    rot_term_res = residues.rot_term_res
    set_template = residues.set_template
    blunt_ends = residues.blunt_ends
    delete_residues = residues.delete_residues

    mk_config = build_mk_config(args, mk_config_dir)

    # initialize MoleculePreparation with config
    mk_prep = MoleculePreparation.from_config(mk_config)

    if mk_config["compute_charges"]:
        # use green text
        print(f"\033[32m {mk_prep.charge_model} harges will be computed from scratch\n \033[0m")
    else:
        print(f"\033[32m {mk_prep.charge_model} charges will be read from template file \n \033[0m")
   
    # load templates for mapping
    if args.cache_templates:
        cache_file = args.cache_templates

        try:
            with open(cache_file, "r") as f:
                json_str = f.read()
            templates = ResidueChemTemplates.from_json(json_str)
        except FileNotFoundError:
            print(f"WARNING: specified cache file for residue chem templates not found. " + eol +
                  f"The initial templates will be default, and a new cache will be created at {cache_file}. ", 
                  file=sys.stderr, 
                  )
            templates = ResidueChemTemplates.create_from_defaults()
        except Exception as e:
            print(f"An error occurred with --cache_templates: {e}")
            sys.exit(1)
    else: 
        templates = ResidueChemTemplates.create_from_defaults()


    for item in args.add_templates:
        if item.endswith(".json"):
            templates.add_json_file(item)
        elif ":" in item: #expect format resname:sdf
            resname, sdf_file = item.split(":", 1)
            template_json = sdf_to_json(sdf_file, resname)
            templates.add_dict(template_json)
        else:
            print("--add_templates must be either a JSON file or resname:file.sdf")
            sys.exit(2)
    
    polymer = build_polymer(
        args,
        templates,
        mk_prep,
        set_template=set_template,
        delete_residues=delete_residues,
        delete_bad_res=delete_bad_res,
        blunt_ends=blunt_ends,
        wanted_altloc=wanted_altloc,
        prody_parsers=SUPPORTED_PRODY_FORMATS if _got_prody else {},
        got_prody=_got_prody,
        prody_import_error=_prody_import_error if not _got_prody else None,
    )
    
    
    # Update residue chem template cache
    if args.cache_templates: 
        updated_templates_json_strs = templates.to_json()
        with open(cache_file, 'w') as f:
            f.write(updated_templates_json_strs)
    
    # Use residue name in the input structure file to find reactive atom name
    # According to the mapping of residue name and reactive atom name
    for res_id in reactive_flexres:
        if res_id not in polymer.monomers:
            print("resid %s not found in input receptor file" % res_id)
            sys.exit(2)
        res_atom = reactive_flexres_name[res_id]
        if not res_atom:
            input_resname = polymer.monomers[res_id].input_resname
            if input_resname in reactive_atom:
                reactive_flexres_name[res_id] = reactive_atom[input_resname]
            else:
                print("no default reactive name for %s, " % input_resname)
                print("use --reactive_name or --reactive_name_specific" + eol)
                sys.exit(2)

    # Use residue name in input file to confirm
    # requested rotatable terminal group residues are eligible
    rotatable_termgrp_residues_allowed = [
        "SER",
        "LYS",
        "TYR",
        "CYS",
        "HIS",
        "HIE",
        "HID",
        "HIP",
        "ASN",
        "GLN",
        "THR",
        "MET",
    ]
    for res_id in rot_term_res:
        if res_id not in polymer.monomers:
            print("resid %s not found in input receptor file" % res_id)
            sys.exit(2)
        input_resname = polymer.monomers[res_id].input_resname
        if input_resname not in rotatable_termgrp_residues_allowed:
            print(f"{input_resname} (resid {res_id}) is not a valid residue for use with --rot_terminal_group."+ eol)
            print("Available residues are: ")
            print(", ".join(rotatable_termgrp_residues_allowed))
            sys.exit(2)
    
    # Print nonreactive and reactive flexible residues specs
    if len(nonreactive_flexres) + len(reactive_flexres) + len(rot_term_res) > 0:
        print()
        print("Flexible residues:")
        print("chain resnum is_reactive reactive_atom")
        string = "%5s%7s%12s%14s"
    
        if len(nonreactive_flexres) > 0:
            for res_id in nonreactive_flexres:
                chain, resnum = res_id.split(":")
                react_atom = ""
                print(string % (chain, resnum, False, react_atom))

        if len(rot_term_res) > 0:
            for res_id in rot_term_res:
                chain, resnum = res_id.split(":")
                react_atom = ""
                print(string % (chain, resnum, False, react_atom), "(rotatable terminal group)")
    
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
            polymer.monomers[res_id], reactive_aname
        )
        if reactive_atomi is None:
            print(f"cannot find reactive atom name {reactive_aname} from residue {res_id} in input receptor file")
            sys.exit(2)
        reactive_atypes = assign_reactive_types_by_index(polymer.monomers[res_id].molsetup, reactive_atomi)
        # set reactive atom types
        nr_atom = len(polymer.monomers[res_id].molsetup.atoms)
        for atom_index in range(nr_atom):
            if (
                polymer.monomers[res_id].molsetup.atoms[atom_index].atom_type
                != reactive_atypes[atom_index]
            ):
                polymer.monomers[res_id].molsetup.atoms[
                    atom_index
                ].atom_type = f"{reactive_prefix}{reactive_atypes[atom_index]}"
        reactive_prefix += 1
    
    # Combine nonreactive and reactive flexible residues into one set
    all_flexres = nonreactive_flexres.union(reactive_flexres)
    
    for res_id in all_flexres:
        polymer.flexibilize_sidechain(res_id, mk_prep)

    # Make terminal groups rotatable by rigidifying everything except the
    # terminal group and then making the residue flexible. The definition of
    # sidechain is dynamic: whatever is allowed to rotate constitutes the
    # sidechain (for PDBQT writing purposes).
    rot_term_smarts = "[#6;!$(C(=O)N);!$([#6;R1]~[#7;R1])]-[#6;!$(C(=O)N);!$([#6;R1]~[#7;R1])]"
    rot_term_indices = (0, 1)
    mk_config_rot_term = mk_config.copy()
    mk_config_rot_term.setdefault("rigidify_bonds_smarts", [])
    mk_config_rot_term.setdefault("rigidify_bonds_indices", [])
    mk_config_rot_term["rigidify_bonds_smarts"].append(rot_term_smarts)
    mk_config_rot_term["rigidify_bonds_indices"].append(rot_term_indices)
    mk_prep_rot_term = MoleculePreparation.from_config(mk_config_rot_term)
    for res_id in rot_term_res:
        polymer.monomers[res_id].parameterize(mk_prep_rot_term, res_id)
        polymer.flexibilize_sidechain(res_id, mk_prep_rot_term)
    
    any_lig_base_types = ANY_LIG_BASE_TYPES

    outpath = (
        pathlib.Path(args.output_basename) if args.output_basename is not None else None
    )
    written_files_log = {"filename": [], "description": []}

    write_json_output(args, polymer, outpath, written_files_log)
    write_pdb_output(args, polymer, written_files_log)
    write_state = write_pdbqt_output(
        args, polymer, outpath, all_flexres, rot_term_res, written_files_log
    )

    if args.write_gpf is not None or args.write_vina_box is not None:
        box_center, box_size = resolve_box(args, polymer, reactive_flexres)
        write_gpf_and_vina_outputs(
            args,
            write_state,
            box_center,
            box_size,
            outpath,
            any_lig_base_types,
            written_files_log,
        )
        warn_flexres_outside_box(polymer, box_center, box_size)

    if len(reactive_flexres) > 0 and args.write_pdbqt is not None:
        write_reactive_config(
            args, write_state, outpath, any_lig_base_types, written_files_log
        )

    print_write_summary(args, written_files_log)

if __name__ == "__main__":
    sys.exit(main())
