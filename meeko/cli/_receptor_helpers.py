"""Helpers extracted from ``mk_prepare_receptor.main()``.

Keeping these out of the entry-point module lets each phase (validation,
config assembly, residue selection, polymer construction, reactive
typing) be tested in isolation without spinning up argparse.
"""

import json
import pathlib
import sys

from meeko.utils.utils import parse_cmdline_res_assign


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
