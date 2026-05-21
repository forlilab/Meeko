"""Helpers extracted from ``mk_prepare_ligand.main()``.

The per-molecule processing loop is the body that's worth pulling out:
both the covalent and non-covalent paths share the same inner step of
"write PDBQT, count failures, log errors". The helpers below expose
that shape as named functions.
"""

import sys


def _write_one_pdbqt(
    molsetup,
    name,
    suffixes,
    output,
    bad_charge_ok: bool,
    add_index_map: bool,
    verbose: bool,
    adapt_flexres: tuple = None,
) -> int:
    """Render one ``molsetup`` to PDBQT and hand it to ``output``.

    ``adapt_flexres``, if given, is ``(res, chain, num)`` — triggers the
    AutoDock4 flexres adaptation on the resulting string (used for the
    covalent-ligand path).

    Returns ``1`` on failure, ``0`` on success (so callers can sum the
    return value into a running failure count).
    """
    from meeko import PDBQTWriterLegacy

    pdbqt_string, success, error_msg = PDBQTWriterLegacy.write_string(
        molsetup,
        bad_charge_ok=bad_charge_ok,
        add_index_map=add_index_map,
    )
    if not success:
        print(error_msg, file=sys.stderr)
        return 1

    if adapt_flexres is not None:
        res, chain, num = adapt_flexres
        pdbqt_string = PDBQTWriterLegacy.adapt_pdbqt_for_autodock4_flexres(
            pdbqt_string, res, chain, num
        )

    output(pdbqt_string, name, suffixes)
    if verbose:
        molsetup.show()
    return 0


def process_covalent_mol(
    mol, args, preparator, covalent_builder, output
) -> int:
    """Run the covalent-ligand path for one input ``mol``.

    Iterates ``covalent_builder.process(...)`` to enumerate one or more
    covalent ligands per input mol; for each, prepares + writes the
    PDBQT (with AutoDock4 flexres adaptation). Returns the failure count.
    """
    nr_failures = 0
    for cov_lig in covalent_builder.process(
        mol, args.tether_smarts, args.tether_smarts_indices
    ):
        root_atom_index = cov_lig.indices[0]
        molsetups = preparator.prepare(
            cov_lig.mol,
            root_atom_index=root_atom_index,
            not_terminal_atoms=[root_atom_index],
            rename_atoms=args.rename_atoms,
        )
        chain, res, num = cov_lig.res_id
        suffixes = output.get_suffixes(molsetups)
        for molsetup, suffix in zip(molsetups, suffixes):
            nr_failures += _write_one_pdbqt(
                molsetup,
                name=molsetup.name,
                suffixes=(cov_lig.label, suffix),
                output=output,
                bad_charge_ok=args.bad_charge_ok,
                add_index_map=args.add_index_map,
                verbose=False,
                adapt_flexres=(res, chain, num),
            )
    return nr_failures


def process_noncovalent_mol(mol, name, args, preparator, output):
    """Run the standard non-covalent path for one input ``mol``.

    Returns ``(nr_failures, raised_during_prepare)``. The raised-flag
    lets the caller skip the rest of the loop body when prepare() itself
    blew up (a behavior the original main() relied on).
    """
    try:
        molsetups = preparator.prepare(mol, rename_atoms=args.rename_atoms)
    except Exception as error_msg:
        print(error_msg, file=sys.stderr)
        return 1, True

    if len(molsetups) > 1:
        output.is_multimol = True
    suffixes = output.get_suffixes(molsetups)
    nr_failures = 0
    for molsetup, suffix in zip(molsetups, suffixes):
        nr_failures += _write_one_pdbqt(
            molsetup,
            name=name,
            suffixes=(suffix,),
            output=output,
            bad_charge_ok=args.bad_charge_ok,
            add_index_map=args.add_index_map,
            verbose=args.verbose,
        )
    return nr_failures, False
