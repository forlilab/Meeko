# Log 06 — Receptor CLI output dispatch (final stage of Task E)

Continues from log 05. The remaining 340 LOC of output-side logic in `mk_prepare_receptor.main()` (`--write_json`, `--write_pdb`, `--write_pdbqt`, GPF/Vina box/box-PDB generation, AD-GPU reactive config, final "Files written:" report) are now seven named helpers in `_receptor_helpers.py`. **92 tests pass, 4 skipped.**

## Commit

| SHA | What | LOC change |
|---|---|---|
| `34cb553` | Stage 6 — extract receptor output dispatch | +425 / −374 |

## Final receptor CLI sizes

| Metric | Original | After log_05 | After log_06 |
|---|---:|---:|---:|
| `mk_prepare_receptor.py` (file) | 1184 | 946 | **610** |
| `main()` body | ~790 | ~566 | **~233** |

That's the full arc:
- **−574 LOC** removed from the entry-point file (48% reduction).
- **−557 LOC** removed from `main()` (70% reduction).

## What lives in the helpers now

| Helper | LOC | Role |
|---|---:|---|
| `write_json_output` | ~10 | `--write_json` file write |
| `write_pdb_output` | ~10 | `--write_pdb` file write |
| `write_pdbqt_output` | ~40 | `--write_pdbqt` with rigid/flex split; returns `WriteState` |
| `resolve_box` | ~60 | resolves box center+size from `--box_center` / `--box_center_off_reactive_res` / `--box_enveloping` |
| `write_gpf_and_vina_outputs` | ~55 | GPF file, Vina-format box, box-visualization PDB |
| `write_reactive_config` | ~70 | AD-GPU `*.reactive_config` + collision report |
| `print_write_summary` | ~35 | Final status stanza |
| `warn_flexres_outside_box` | ~20 | Previously nested inside `main()` |

Plus a small `WriteState` dataclass carrying `rigid_fn`, `flex_fn`, `all_flex_pdbqt` between phases, and two new module-level constants (`ANY_LIG_BASE_TYPES`, `GPF_REC_TYPES`) replacing inline literals.

## `main()` shape now

The output section reads in three blocks of 1–7 lines each:

```python
write_json_output(args, polymer, outpath, written_files_log)
write_pdb_output(args, polymer, written_files_log)
write_state = write_pdbqt_output(
    args, polymer, outpath, all_flexres, rot_term_res, written_files_log
)

if args.write_gpf is not None or args.write_vina_box is not None:
    box_center, box_size = resolve_box(args, polymer, reactive_flexres)
    write_gpf_and_vina_outputs(
        args, write_state, box_center, box_size, outpath,
        any_lig_base_types, written_files_log,
    )
    warn_flexres_outside_box(polymer, box_center, box_size)

if len(reactive_flexres) > 0 and args.write_pdbqt is not None:
    write_reactive_config(
        args, write_state, outpath, any_lig_base_types, written_files_log
    )

print_write_summary(args, written_files_log)
```

## Side cleanup

Top-level imports in `mk_prepare_receptor.py` collapsed from 17 to 7 names: `PDBQTMolecule`, `RDKitMolCreate`, `PDBQTWriterLegacy`, `Polymer`, `PolymerCreationError`, `MoleculeSetup`, `reactive_typer`, `get_reactive_config`, `gridbox`, `pdbutils`, `math`, `numpy` are now only referenced inside helpers and don't pollute the entry-point module's namespace.

## Status of `design_analysis_v1.md` priorities

| # | Task | Status |
|---|---|---|
| 1 | Flatten static-only classes | ✅ log 01 |
| 2 | Polymer split | ✅ log 02 |
| 3 | `MoleculePreparation` → `PrepConfig` | ✅ log 03 |
| 4 | Type the flexibility model | ✅ log 00 |
| 5 | Split `molsetup.py` | ✅ log 00 |
| 6 | Dedupe `geomutils.py` | ✅ log 04 |
| 7 | Extract CLI logic | ✅ log 05 (input side) + log 06 (output side) |

**All 7 priorities from the original critical assessment are now done.** The `refactoring` branch contains 28 commits since branching off `develop`, each one keeping the 92/4 test baseline.

## Open follow-ups (not in scope of the original audit)

- **Ligand CLI** still has the `Output` helper class and ~140-LOC `main()`. Less urgent than the receptor was.
- **`meeko/tmp/`** directory ships as part of the package despite never being imported (flagged in log 04). One `git rm` away from cleanup.
- **`PDBQTWriterLegacy` / `HydrateMoleculeLegacy`** names suggest planned successors that never landed. Worth documenting or renaming.
- The `mk_prepare_receptor.py` `main()` still has the **reactive-typing block** (lines ~530–558) that wasn't extracted — it operates on `polymer.monomers[res_id].molsetup` directly and would slot into the existing helpers naturally.

Branch is in a clean, tested, well-segmented state — ready for review or merge whenever stakeholders are.
