# Log 05 — CLI extraction (Task E)

The CLI god-`main()` problem from `design_analysis_v1.md` is meaningfully addressed. Per the plan in `06_refactoring_cli.md`. **92 tests pass, 4 skipped** at every stage.

## Commits

| # | SHA | Stage | LOC change |
|---|---|---|---|
| 1 | `ecb0bd2` | Shared `_common.py` (`TalkativeParser`, `check`, `required_length`) | +59 / −22 |
| 2 | `e8c481e` | `validate_altloc_and_write_flags` + `build_mk_config` | +90 / −58 |
| 3 | `51edd81` | `resolve_residue_selections` + `ResidueSelections` dataclass | +128 / −100 |
| 4 | `fcc766c` | `build_polymer` dispatch | +151 / −101 |
| 5 | `b7f8169` | Ligand: `process_covalent_mol` + `process_noncovalent_mol` | +130 / −55 |

## Size impact

| File | Before | After | Δ |
|---|---:|---:|---:|
| `mk_prepare_receptor.py` | 1184 | **946** | −238 |
| `mk_prepare_ligand.py` | 697 | 655 | −42 |
| `mk_prepare_receptor.main()` (lines from `def main` to `if __name__`) | ~790 | **~566** | −224 |
| `mk_prepare_ligand.main()` | ~180 | ~139 | −41 |

New files holding the extracted logic:

| File | LOC | What |
|---|---:|---|
| `meeko/cli/_common.py` | 56 | `make_talkative_parser`, `check`, `required_length` |
| `meeko/cli/_receptor_helpers.py` | 329 | `validate_altloc_and_write_flags`, `build_mk_config`, `resolve_residue_selections`, `ResidueSelections`, `build_polymer` |
| `meeko/cli/_ligand_helpers.py` | 117 | `process_covalent_mol`, `process_noncovalent_mol`, `_write_one_pdbqt` |

## What main() looks like now

The receptor `main()` still does too much (566 LOC), but it's now mostly orchestration — the messy phases (input parsing, residue selection, polymer construction) live in named helpers. The remaining bulk is the **output writing** dispatch (lines ~700–1100 in the new file: `--write_pdb`, `--write_pdbqt` with flexres logic, `--write_json`, GPF generation). That's the next-largest cohesive block worth extracting.

The ligand `main()` is now close to the original design analysis target of ~30 lines for the orchestration body — the per-molecule loop body is two helper calls and a tally update.

## API surface preserved

Every CLI flag still works. `TalkativeParser` is still importable from `meeko.cli.mk_prepare_receptor` (it's now built by `make_talkative_parser(path_to_this_script)` and assigned at module load). No console_scripts entrypoints changed. 92 tests pass throughout.

## Loose ends / next moves

- **Receptor output dispatch** is the biggest remaining block in `main()` (~400 LOC). Splitting it into `write_pdb_output`, `write_pdbqt_output`, `write_json_output`, `write_gpf_output` would bring `main()` below 200 LOC. Deferred to a follow-up.
- **`meeko/tmp/`** (flagged in log 04) still ships. Not in scope here but worth a separate `git rm` discussion.

## Status of `design_analysis_v1.md` priorities

| # | Task | Status |
|---|---|---|
| 1 | Flatten static-only classes | ✅ log 01 |
| 2 | Polymer split | ✅ log 02 |
| 3 | `MoleculePreparation` → `PrepConfig` | ✅ log 03 |
| 4 | Type the flexibility model | ✅ log 00 |
| 5 | Split `molsetup.py` | ✅ log 00 |
| 6 | Dedupe `geomutils.py` | ✅ log 04 |
| 7 | Extract CLI logic | ✅ log 05 (partial — output dispatch remains) |

**All 7 priorities are now addressed.** The receptor CLI output dispatch is the last meaningful god-block; everything else from the original critical assessment has either been refactored or deemed out of scope.
