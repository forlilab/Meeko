"""Helpers extracted from ``mk_prepare_receptor.main()``.

Keeping these out of the entry-point module lets each phase (validation,
config assembly, residue selection, polymer construction, reactive
typing) be tested in isolation without spinning up argparse.
"""

import json
import pathlib
import sys
from dataclasses import dataclass, field

from meeko.utils.utils import parse_cmdline_res, parse_cmdline_res_assign


_DEFAULT_REACTIVE_ATOM_BY_RESNAME = {
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


@dataclass
class ResidueSelections:
    """Structured residue selections parsed from the command line."""

    reactive_atom_by_resname: dict = field(default_factory=dict)
    reactive_flexres: set = field(default_factory=set)
    reactive_flexres_name: dict = field(default_factory=dict)
    nonreactive_flexres: set = field(default_factory=set)
    rot_term_res: set = field(default_factory=set)
    set_template: dict = field(default_factory=dict)
    blunt_ends: list = field(default_factory=list)
    delete_residues: list = field(default_factory=list)


def validate_altloc_and_write_flags(args) -> None:
    """Exit with code 2 if any of the altloc / write-flag combinations
    are invalid. Returns ``None``; ``args`` is not mutated.
    """
    if args.wanted_altloc is not None:
        wanted_altloc = parse_cmdline_res_assign(args.wanted_altloc)
        for value in wanted_altloc.values():
            if isinstance(value, str) and value.strip() == "":
                print(
                    "Command line error: Wanted atloc cannot be an empty string or a string with just space",
                    file=sys.stderr,
                )
                sys.exit(2)

    if args.default_altloc is not None and args.default_altloc.strip() == "":
        print(
            "Command line error: Allowed atloc cannot be an empty string or a string with just space",
            file=sys.stderr,
        )
        sys.exit(2)

    write_flags = [
        args.write_pdbqt,
        args.write_json,
        args.write_gpf,
        args.write_vina_box,
    ]
    needed_default = any(flag is not None and len(flag) == 0 for flag in write_flags)
    if needed_default and args.output_basename is None:
        print(
            "--write flags require either a filename argument or"
            " --output_basename to set a default"
        )
        sys.exit(2)


def resolve_residue_selections(args) -> ResidueSelections:
    """Parse all the residue-related CLI flags into a single object.

    Handles --reactive_name, --reactive_name_specific, --reactive_flexres,
    --flexres, --rot_terminal_group, --set_template, --blunt_ends,
    --delete_residues. Exits with code 2 on conflicting or excess inputs.
    """
    sel = ResidueSelections(
        reactive_atom_by_resname=dict(_DEFAULT_REACTIVE_ATOM_BY_RESNAME)
    )

    modified: set = set()
    for react_name_str in args.reactive_name:
        resname, name = react_name_str.split(":")
        if resname in modified:
            print(
                "Command line error: repeated resname %s passed to --reactive_resname\n"
                % resname,
                file=sys.stderr,
            )
            sys.exit(2)
        modified.add(resname)
        sel.reactive_atom_by_resname[resname] = name

    modified = set()
    for string in args.reactive_name_specific:
        res_assign = parse_cmdline_res_assign(string)
        for res_id in res_assign:
            if res_id in modified:
                print(
                    "Command line error: repeated resid %s passed to --reactive_name_specific\n"
                    % res_id,
                    file=sys.stderr,
                )
                sys.exit(2)
            modified.add(res_id)
            sel.reactive_flexres_name[res_id] = res_assign[res_id]

    sel.reactive_flexres = set(sel.reactive_flexres_name)
    for resid_string in args.reactive_flexres:
        for res_id in parse_cmdline_res(resid_string):
            if res_id not in sel.reactive_flexres:
                sel.reactive_flexres.add(res_id)
                sel.reactive_flexres_name[res_id] = ""

    if len(sel.reactive_flexres) > 8:
        print(
            "Command line error: got %d reactive_flexres but maximum is 8."
            % len(args.reactive_flexres),
            file=sys.stderr,
        )
        sys.exit(2)

    if len(sel.reactive_flexres) != 1 and args.box_center_off_reactive_res:
        print(
            "Command line error:--box_center_off_reactive_res can be used only with one"
            " reactive\nresidue, but %d reactive residues are set"
            % len(sel.reactive_flexres_name),
            file=sys.stderr,
        )
        sys.exit(2)

    for string in args.flexres:
        for res_id in parse_cmdline_res(string):
            if res_id not in sel.reactive_flexres:
                sel.nonreactive_flexres.add(res_id)

    for string in args.rot_terminal_group:
        for res_id in parse_cmdline_res(string):
            if res_id not in sel.reactive_flexres and res_id not in sel.nonreactive_flexres:
                sel.rot_term_res.add(res_id)

    if args.set_template is not None:
        sel.set_template = parse_cmdline_res_assign(args.set_template)

    if args.blunt_ends is not None:
        j = parse_cmdline_res_assign(args.blunt_ends)
        sel.blunt_ends = [(k, int(v)) for k, v in j.items()]

    if args.delete_residues is not None:
        sel.delete_residues = parse_cmdline_res(args.delete_residues)

    return sel


def build_mk_config(args, mk_config_dir: pathlib.Path) -> dict:
    """Assemble the ``MoleculePreparation`` config dict from ``--config_preset``
    JSON, ``--config_file`` JSON, and any explicit command-line overrides.

    Exits with code 2 if ``--charge_model read`` is requested without
    ``--read_pqr``.
    """
    mk_config: dict = {}

    if args.config_preset is not None:
        with open(mk_config_dir / f"{args.config_preset}.json") as f:
            mk_config.update(json.load(f))

    if args.config_file is not None:
        with open(args.config_file) as f:
            mk_config.update(json.load(f))

    mk_config["compute_charges"] = args.compute_charges

    if args.charge_model is not None:
        mk_config["charge_model"] = args.charge_model

    if mk_config.get("charge_model") == "read":
        if args.read_pqr is None:
            print("Error: --charge_model read requires --read_pqr")
            sys.exit(2)
        mk_config["charge_atom_prop"] = "PQRCharge"

    return mk_config
