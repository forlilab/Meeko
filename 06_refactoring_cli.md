# Refactoring plan — CLI extraction (Task E)

## What's wrong with the current shape

The two preparation CLIs each have a god `main()`:

| File | LOC | `main()` size | Notes |
|---|---:|---:|---|
| `meeko/cli/mk_prepare_receptor.py` | 1184 | **~790 LOC** (lines 393–1182) | 16 distinct logical phases, no helpers |
| `meeko/cli/mk_prepare_ligand.py` | 697 | ~180 LOC (lines 515–694) | has `Output` helper class; `main` itself is straight-line |

`mk_export.py` (164 LOC) is small enough to leave alone.

The original directive in `design_analysis_v1.md`:

> `mk_prepare_receptor.py:main` is ~500 lines of nested conditionals: arg parsing, IO, dispatch, validation, error reporting … should be split: argument schema → config → core call → reporting.

## Strategy

Each CLI `main()` should be a short orchestrator: it parses args, calls a small set of helper functions that do the real work, prints status, exits with the right code. The helpers go in new sibling modules so they're importable and testable in isolation.

Bias toward **incremental, behavior-preserving** moves: extract one logical block at a time, run tests after each, commit.

## Receptor CLI — five extractions

Five logical phases of `main()` that are self-contained and can move out:

| Phase | Lines (approx) | Target helper |
|---|---:|---|
| A. Validate altloc / write flags | 397–432 | `_validate_cli_args(args)` |
| B. Reactive / flexres / rot-term residue parsing | 434–516 | `_resolve_residue_selections(args)` |
| C. Assemble `mk_config` from preset + JSON + args | 533–562 | `_build_mk_config(args)` |
| D. Polymer construction dispatch (4 readers) | 596–696 | `_build_polymer(args, templates, mk_prep, ...)` |
| E. Reactive type assignment + flexres preparation | 705–820 | `_apply_reactive_typing(args, polymer, ...)` |

Output dispatch (lines 837–1072: `--write_pdb`, `--write_pdbqt`, `--write_json`, GPF output) is the next-largest block. Worth a separate stage if time allows; deferred to the end.

## Ligand CLI — one extraction

The 180-line `main()` body is mostly one loop that iterates molecules and prepares each. Pull the body of the loop into:

| Phase | Target |
|---|---|
| Per-molecule prep+write | `_prepare_and_write(mol, args, config, preparator, output, ...)` |

Plus split the covalent / non-covalent branches into named helpers so the loop is short.

## Shared utilities

`TalkativeParser`, `check`, `required_length` (receptor CLI) are reusable. Move them into `meeko/cli/_common.py` so both CLIs can use them. Don't break the existing `cli.mk_prepare_receptor.TalkativeParser` import surface (re-export from the receptor module).

## Stages, in order

1. Add `meeko/cli/_common.py` with shared helpers (`TalkativeParser`, `check`, `required_length`). Re-export from receptor module for backward compat.
2. Receptor: extract `_validate_cli_args` + `_build_mk_config`. Smallest first.
3. Receptor: extract `_resolve_residue_selections`.
4. Receptor: extract `_build_polymer`.
5. Receptor: extract `_apply_reactive_typing`.
6. Ligand: extract `_prepare_and_write`.
7. (Optional) Receptor: extract the output-writing dispatch.

Each stage = one commit. Tests pass between each.

## Risk

Medium. The CLIs are tested by the user-facing entrypoints; there's no unit test that exercises `main()` directly. The `polymer_creation_test.py` and `parameterization_test.py` exercise the underlying classes the CLIs call, but not the CLI control flow itself.

Mitigation:
- Behavior-preserving moves only — extract code verbatim into a function, replace the original block with a single call.
- After each extraction, do a smoke-run: `python -c "import meeko.cli.mk_prepare_receptor; print(meeko.cli.mk_prepare_receptor.main.__doc__)"` to confirm imports + module attribute survive.
- Run the full 92-test suite to catch any regressions in code paths the helpers touch transitively.

## Success criteria

- 92 tests pass, 4 skipped (baseline) after every commit.
- `mk_prepare_receptor` `main()` shrinks from ~790 to under 300 LOC (~60% reduction).
- `mk_prepare_ligand` `main()` shrinks from ~180 to under 100 LOC.
- Both CLIs invocable from the command line as before.

## Out of scope

- **Not** redesigning the CLI argument surface; every existing flag keeps its meaning.
- **Not** introducing a `click`/`typer`/`fire`-style replacement for `argparse`.
- **Not** moving the entrypoints out of the package or changing console_scripts in `setup.py`.
