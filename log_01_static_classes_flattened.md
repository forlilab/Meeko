# Log 01 — Static-only classes flattened (Task A)

The three static-only classes flagged in `design_analysis_v1.md` are now module-level functions with thin compatibility shims. Plan was in `02_refactoring_static_classes.md`. All 92 tests pass (4 skipped) after every commit.

## Commits

| # | SHA | File | LOC delta |
|---|---|---|---|
| 1 | `a65111a` | `meeko/atomtyper.py` | −249 +217 (32 lines gone) |
| 2 | `76c4a08` | `meeko/rdkit_mol_create.py` | −419 +426 (~0; mostly indentation) |
| 3 | `dae62fa` | `meeko/writer.py` | −425 +368 (57 lines gone) |

Net: **~90 LOC removed** across three modules, mostly the `cls`/`@staticmethod`/`@classmethod` decorator boilerplate and method-body indentation.

## What each commit did

**`AtomTyper`** — 6 methods promoted to `type_everything`, `type_atoms`, `cache_offatoms`, `set_offatoms`, `type_dihedrals` at module level. The new public names drop the underscore prefix (the methods were `_type_atoms` etc., but they weren't really private — they were called from outside the class). The shim class re-exposes the underscore names so callers don't break.

**`RDKitMolCreate`** — 7 methods promoted, plus two class-level data dicts (`flexres`, `ambiguous_flexres_choices`) promoted to `FLEXRES` and `AMBIGUOUS_FLEXRES_CHOICES` module constants. They were tables of constants, never instance state.

**`PDBQTWriterLegacy`** — 10 methods promoted. The recursive `_walk_graph_recursive` stays recursive — it just no longer needs `cls` to recurse. `write_string_from_polymer` is now a 2-line helper around `write_from_polymer`.

## Pattern (carried forward from the MoleculeSetup refactor)

Each module ends with a shim class:

```python
class AtomTyper:
    """Backward-compat shim. Prefer the module-level functions for new code."""
    type_everything = staticmethod(type_everything)
    _type_atoms     = staticmethod(type_atoms)
    ...
```

So both styles work — `AtomTyper.type_everything(...)` (old) and `atomtyper.type_everything(...)` (new) — and downstream code can migrate at its own pace.

## Status of `design_analysis_v1.md` priorities

| # | Task | Status |
|---|---|---|
| 1 | Flatten static-only classes | ✅ done in log 01 |
| 2 | Polymer split | **TODO** (next) |
| 3 | `MoleculePreparation` → `PrepConfig` | TODO |
| 4 | Type the flexibility model | ✅ done in log 00 |
| 5 | Split `molsetup.py` | ✅ done in log 00 |
| 6 | Dedupe `geomutils.py` | TODO |
| 7 | Extract CLI logic | TODO |

3 of 7 done. Next target: the 3597-LOC `polymer.py` god module.
