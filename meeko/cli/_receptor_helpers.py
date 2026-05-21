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

from ._common import check


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


def build_polymer(
    args,
    templates,
    mk_prep,
    set_template: dict,
    delete_residues: list,
    delete_bad_res: bool,
    blunt_ends: list,
    wanted_altloc,
    prody_parsers: dict,
    got_prody: bool,
    prody_import_error: Exception,
):
    """Dispatch one of the four ``--read_*`` polymer constructors.

    ``--read_with_prody`` (ProDy) → ``Polymer.from_prody``
    ``--read_pdb`` (PDB string) → ``Polymer.from_pdb_string``
    ``--read_json`` (Meeko JSON) → reload + go through ``from_pdb_string``
    ``--read_pqr`` (PQR string) → ``Polymer.from_pqr_string``

    Exits with code 1 on ``PolymerCreationError`` (after printing the
    error). Exits with code 2 on missing-prody (when ``--read_with_prody``).
    """
    from meeko import Polymer, PolymerCreationError

    if args.read_with_prody is not None:
        if not got_prody:
            print(prody_import_error, file=sys.stderr)
            print("option --read_with_prody requires Prody, which is not installed.")
            print(
                "Installable from PyPI (pip install prody) or conda-forge"
                " (micromamba install prody)"
            )
            sys.exit(2)
        ext = pathlib.Path(args.read_with_prody).suffix[1:].lower()
        if ext not in prody_parsers:
            print(
                f"--read_with_prody: unsupported extension {ext!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        parser = prody_parsers[ext]
        input_obj = parser(args.read_with_prody, altloc="all")
        try:
            return Polymer.from_prody(
                input_obj,
                templates,
                mk_prep,
                set_template,
                delete_residues,
                args.ignore_https_cert,
                delete_bad_res,
                blunt_ends=blunt_ends,
                wanted_altloc=wanted_altloc,
                default_altloc=args.default_altloc,
                forgive_extra_bonds=args.forgive_extra_bonds,
            )
        except PolymerCreationError as e:
            print(e)
            sys.exit(1)

    if args.read_pdb is not None:
        with open(args.read_pdb) as f:
            pdb_string = f.read()
        try:
            return Polymer.from_pdb_string(
                pdb_string,
                templates,
                mk_prep,
                set_template,
                delete_residues,
                args.ignore_https_cert,
                delete_bad_res,
                blunt_ends=blunt_ends,
                wanted_altloc=wanted_altloc,
                default_altloc=args.default_altloc,
            )
        except PolymerCreationError as e:
            print(e)
            sys.exit(1)

    if args.read_json is not None:
        # Load the saved polymer, dump it to PDB, then re-load through
        # from_pdb_string so user options (set_template, blunt_ends,
        # delete_residues, altloc, …) are applied.
        with open(args.read_json) as f:
            json_string = f.read()
        try:
            polymer = Polymer.from_json(json_string)
            pdb_string = polymer.to_pdb()
            return Polymer.from_pdb_string(
                pdb_string,
                templates,
                mk_prep,
                set_template,
                delete_residues,
                args.ignore_https_cert,
                delete_bad_res,
                blunt_ends=blunt_ends,
                wanted_altloc=wanted_altloc,
                default_altloc=args.default_altloc,
                forgive_extra_bonds=args.forgive_extra_bonds,
            )
        except PolymerCreationError as e:
            print(e)
            sys.exit(1)

    # args.read_pqr is not None
    with open(args.read_pqr) as f:
        pdb_string = f.read()
    try:
        print(
            "Reading a PQR file. The following options or configurations will be ignored: "
        )
        print("  - default_altloc")
        print("  - wanted_altloc")
        if mk_prep.charge_model != "read":
            print("Only reading structures from PQR. ")
            print(f"Charge model of choice: {mk_prep.charge_model}")
        else:
            print("Reading structures and partial charges from PQR. ")
        return Polymer.from_pqr_string(
            pdb_string,
            templates,
            mk_prep,
            set_template,
            delete_residues,
            args.ignore_https_cert,
            delete_bad_res,
            blunt_ends=blunt_ends,
            forgive_extra_bonds=args.forgive_extra_bonds,
        )
    except PolymerCreationError as e:
        print(e)
        sys.exit(1)


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


# ---------------------------------------------------------------------------
# Output-side helpers (write_json / write_pdb / write_pdbqt / GPF / Vina box /
# reactive config / final status report).
# ---------------------------------------------------------------------------

ANY_LIG_BASE_TYPES = [
    "HD", "C", "A", "N", "NA", "OA", "F", "P",
    "SA", "S", "Cl", "Br", "I", "Si", "B",
]

GPF_REC_TYPES = [
    "HD", "C", "A", "N", "NA", "OA", "F", "P",
    "SA", "S", "Cl", "Br", "I", "Mg", "Ca", "Mn", "Fe", "Zn",
]


@dataclass
class WriteState:
    """Outputs from ``write_pdbqt_output`` that downstream phases (GPF,
    reactive config) need to reference."""

    rigid_fn: str = None
    flex_fn: str = None
    all_flex_pdbqt: str = ""


def _append_log(written_files_log, fn, description):
    written_files_log["filename"].append(str(fn))
    written_files_log["description"].append(description)


def write_json_output(args, polymer, outpath, written_files_log) -> None:
    """Handle ``--write_json``."""
    if args.write_json is None:
        return
    if args.write_json:
        fn = args.write_json[0]
    else:
        fn = str(outpath) + ".json"
    with open(fn, "w") as f:
        f.write(polymer.to_json())
    _append_log(written_files_log, fn, "parameterized receptor")


def write_pdb_output(args, polymer, written_files_log) -> None:
    """Handle ``--write_pdb``."""
    if args.write_pdb is None:
        return
    if not args.write_pdb:
        raise ValueError("--write_pdb requires a filename")
    fn = args.write_pdb[0]
    with open(fn, "w") as f:
        f.write(polymer.to_pdb())
    _append_log(written_files_log, fn, "processed receptor PDB")


def write_pdbqt_output(
    args, polymer, outpath, all_flexres, rot_term_res, written_files_log
) -> WriteState:
    """Handle ``--write_pdbqt`` and the rigid/flex split.

    Returns ``WriteState`` carrying ``rigid_fn``, ``flex_fn``, and the
    accumulated ``all_flex_pdbqt`` string that the GPF and reactive-config
    phases need.
    """
    from meeko import PDBQTWriterLegacy

    state = WriteState()
    if args.write_pdbqt is None:
        return state

    if args.write_pdbqt:
        if args.write_pdbqt[0].endswith(".pdbqt"):
            fn_base = str(pathlib.Path(args.write_pdbqt[0]).with_suffix(""))
        else:
            fn_base = args.write_pdbqt[0]
    else:
        fn_base = str(outpath)

    rigid_pdbqt, flex_pdbqt_dict = PDBQTWriterLegacy.write_from_polymer(polymer)

    if len(all_flexres) + len(rot_term_res) == 0:
        state.rigid_fn = fn_base + ".pdbqt"
    else:
        for flexres_pdbqt in flex_pdbqt_dict.values():
            state.all_flex_pdbqt += flexres_pdbqt
        state.rigid_fn = fn_base + "_rigid.pdbqt"
        state.flex_fn = fn_base + "_flex.pdbqt"
        if state.all_flex_pdbqt:
            _append_log(written_files_log, state.flex_fn, "flexible receptor input file")
            with open(state.flex_fn, "w") as f:
                f.write(state.all_flex_pdbqt)

    _append_log(
        written_files_log, state.rigid_fn, "static (i.e., rigid) receptor input file"
    )
    with open(state.rigid_fn, "w") as f:
        f.write(rigid_pdbqt)
    return state


def warn_flexres_outside_box(polymer, box_center, box_size) -> None:
    """Print a stderr warning if any flexible residue's atom lies outside
    the docking box."""
    from meeko import gridbox

    eol = "\n"
    for res in polymer.monomers.values():
        if not res.is_movable:
            continue
        for atom in res.molsetup.atoms:
            if not res.is_flexres_atom[atom.index]:
                continue
            if gridbox.is_point_outside_box(
                atom.coord, box_center, box_size, spacing=1.0
            ):
                print(
                    "WARNING: Flexible residue outside box." + eol,
                    file=sys.stderr,
                )
                print(
                    "WARNING: Strongly recommended to use a box that encompasses"
                    " flexible residues." + eol,
                    file=sys.stderr,
                )
                return  # only need to warn once


def resolve_box(args, polymer, reactive_flexres):
    """Compute ``(box_center, box_size)`` from one of the supported sources:
    ``--box_center``, ``--box_center_off_reactive_res``, or
    ``--box_enveloping`` (PDB/MOL/MOL2/SDF/PDBQT).

    Exits with code 2 if none specify a box.
    """
    import math
    import numpy as np
    from rdkit import Chem
    from meeko import PDBQTMolecule, RDKitMolCreate, gridbox, pdbutils

    if args.box_center is not None:
        return args.box_center, args.box_size

    if args.box_center_off_reactive_res:
        box_centers = []
        for res_id in reactive_flexres:
            molsetup = polymer.monomers[res_id].molsetup
            calpha_idx = [
                atom.index for atom in molsetup.atoms if atom.pdbinfo.name == "CA"
            ]
            cbeta_idx = [
                atom.index for atom in molsetup.atoms if atom.pdbinfo.name == "CB"
            ]
            check(
                len(calpha_idx) == 1,
                f"found {len(calpha_idx)} CA in {res_id} but expected 1",
            )
            check(
                len(cbeta_idx) == 1,
                f"found {len(cbeta_idx)} CB in {res_id} but expected 1",
            )
            ca = molsetup.get_coord(calpha_idx[0])
            cb = molsetup.get_coord(cbeta_idx[0])
            v = cb - ca
            v /= math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) + 1e-8
            box_centers.append(ca + 5 * v)
        return np.mean(box_centers, 0), args.box_size

    if args.box_enveloping is not None:
        ft = pathlib.Path(args.box_enveloping).suffix
        suppliers = {
            ".pdb": None,
            ".mol": Chem.MolFromMolFile,
            ".mol2": Chem.MolFromMol2File,
            ".sdf": Chem.SDMolSupplier,
            ".pdbqt": None,
        }
        if ft not in suppliers:
            check(False, f"Given --box_enveloping file type {ft} not readable!")
        if ft == ".pdb":
            pdbstr = pdbutils.strip_altloc_from_pdb_file(args.box_enveloping)
            ligmol = Chem.MolFromPDBBlock(pdbstr, removeHs=False, sanitize=False)
        elif ft == ".pdbqt":
            ligmol = RDKitMolCreate.from_pdbqt_mol(
                PDBQTMolecule.from_file(args.box_enveloping)
            )[0]
        elif ft == ".sdf":
            ligmol = suppliers[ft](
                args.box_enveloping, removeHs=False, sanitize=False
            )[0]
        else:
            ligmol = suppliers[ft](
                args.box_enveloping, removeHs=False, sanitize=False
            )
        return gridbox.calc_box(
            ligmol.GetConformer().GetPositions(), args.padding
        )

    print("Error: No box center specified.", file=sys.stderr)
    sys.exit(2)


def write_gpf_and_vina_outputs(
    args,
    write_state: WriteState,
    box_center,
    box_size,
    outpath,
    any_lig_base_types,
    written_files_log,
) -> None:
    """Write the AutoGrid GPF file, the Vina-format box file, and the
    visualization PDB. All gated on ``--write_gpf`` / ``--write_vina_box``.
    """
    from meeko import gridbox

    if args.write_gpf is not None:
        if args.write_gpf:
            gpf_fn = args.write_gpf[0]
        else:
            gpf_fn = pathlib.Path(write_state.rigid_fn).with_suffix(".gpf")
        ff_fn = pathlib.Path(gpf_fn).parents[0] / pathlib.Path(
            "boron-silicon-atom_par.dat"
        )
        _append_log(
            written_files_log,
            ff_fn,
            "atomic parameters for B and Si (for autogrid)",
        )
        with open(ff_fn, "w") as f:
            f.write(gridbox.boron_silicon_atompar)

        gpf_string, _ = gridbox.get_gpf_string(
            box_center,
            box_size,
            pathlib.Path(write_state.rigid_fn).name,
            GPF_REC_TYPES,
            any_lig_base_types,
            ff_param_fname=ff_fn.name,
        )
        _append_log(written_files_log, gpf_fn, "autogrid input file")
        with open(gpf_fn, "w") as f:
            f.write(gpf_string)

    box_vina_fn = None
    if args.write_vina_box is not None:
        if args.write_vina_box:
            box_vina_fn = args.write_vina_box[0]
        else:
            box_vina_fn = str(outpath) + ".box.txt"
        _append_log(written_files_log, box_vina_fn, "Vina-style box dimension file")
        with open(box_vina_fn, "w") as f:
            f.write(gridbox.box_to_vina_string(box_center, box_size))

    if args.write_vina_box is not None or args.write_gpf is not None:
        if args.output_basename is not None:
            box_fn = str(outpath) + ".box.pdb"
        elif args.write_gpf is not None:
            box_fn = str(pathlib.Path(write_state.rigid_fn).with_suffix(".box.pdb"))
        else:
            box_fn = box_vina_fn.replace(".txt", "") + ".pdb"
        _append_log(written_files_log, box_fn, "PDB file to visualize the grid box")
        with open(box_fn, "w") as f:
            f.write(gridbox.box_to_pdb_string(box_center, box_size, spacing=1.0))


def write_reactive_config(
    args,
    write_state: WriteState,
    outpath,
    any_lig_base_types,
    written_files_log,
) -> None:
    """Write the AutoDock-GPU reactive-docking configuration file."""
    from meeko import get_reactive_config, reactive_typer

    eol = "\n"

    any_lig_reac_types = []
    for order in (1, 2, 3):
        for t in any_lig_base_types:
            any_lig_reac_types.append(reactive_typer.get_reactive_atype(t, order))

    rec_reac_types = []
    for line in write_state.all_flex_pdbqt.split(eol):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atype = line[77:].strip()
            basetype, _ = reactive_typer.get_basetype_and_order(atype)
            if basetype is not None:
                rec_reac_types.append(line[77:].strip())

    derivtypes, modpairs, collisions = get_reactive_config(
        any_lig_reac_types,
        rec_reac_types,
        args.eps_12,
        args.r_eq_12,
        args.r_eq_13_scaling,
        args.r_eq_14_scaling,
    )

    if collisions:
        collision_str = ""
        for t1, t2 in collisions:
            collision_str += "%3s %3s" % (t1, t2) + eol
        collision_fn = str(outpath.with_suffix(".atype_collisions"))
        _append_log(
            written_files_log,
            collision_fn,
            "type pairs (n=%d) that may lead to intra-molecular reactions"
            % len(collisions),
        )
        with open(collision_fn, "w") as f:
            f.write(collision_str)

    map_block = ""
    map_prefix = pathlib.Path(write_state.rigid_fn).with_suffix("").name
    all_types = []
    for basetype, reactypes in derivtypes.items():
        all_types.append(basetype)
        map_block += "map %s.%s.map" % (map_prefix, basetype) + eol
        for reactype in reactypes:
            all_types.append(reactype)
            map_block += "map %s.%s.map" % (map_prefix, basetype) + eol
    config = "ligand_types " + " ".join(all_types) + eol
    config += "fld %s.maps.fld" % map_prefix + eol
    config += map_block

    line_fmt = "intnbp_r_eps %8.6f %8.6f %3d %3d %4s %4s" + eol
    for (t1, t2), param in modpairs.items():
        config += line_fmt % (
            param["r_eq"], param["eps"], param["n"], param["m"], t1, t2
        )
    config_fn = str(outpath.with_suffix(".reactive_config"))
    _append_log(written_files_log, config_fn, "reactive parameters for AutoDock-GPU")
    with open(config_fn, "w") as f:
        f.write(config)
    print()
    print("For reactive docking, pass the configuration file to AutoDock-GPU:")
    print(
        "    autodock_gpu -C 1 --import_dpf %s --flexres %s -L <ligand_filename>"
        % (config_fn, write_state.flex_fn)
    )
    print()


def print_write_summary(args, written_files_log) -> None:
    """Final status section of the receptor CLI."""
    if written_files_log["filename"]:
        print()
        print("Files written:")
        longest_fn = max(len(fn) for fn in written_files_log["filename"])
        line = "%%%ds <-- " % longest_fn + "%s"
        for fn, desc in zip(
            written_files_log["filename"], written_files_log["description"]
        ):
            print(line % (fn, desc))
        if (
            args.output_basename is not None
            and args.output_basename.endswith(".pdbqt")
            and args.write_pdbqt is None
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
