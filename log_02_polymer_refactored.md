# Log 02 — Polymer god module refactored (Task B)

The 3597-LOC `polymer.py` god module has been split into the `meeko/polymer/` package per the plan in `03_refactoring_polymer.md`. **92 tests pass, 4 skipped** after every commit — identical to the baseline.

## Commits

| # | SHA | Stage | LOC change |
|---|---|---|---|
| 1 | `d83eaf7` | Split polymer.py into a package | 8 new modules from 1 file |
| 2 | `5b45439` | Extract + dedupe `_pdb`/`_pqr`/`_prody`-to-residue-mols | polymer.py 1729→1436 (-293) |
| 3 | `aa2dbe2` | Extract CCD network fetch, add `allow_template_fetch` | polymer.py 1436→1507 (+71 from helper) |

## Final `meeko/polymer/` layout

| File | LOC | Purpose |
|---|---:|---|
| `polymer.py` | 1507 | `Polymer` class — orchestrator (still the largest piece) |
| `utils.py` | 589 | graph paths, inter-residue bond detection, MCS, charges, H positions |
| `padder.py` | 364 | `ResiduePadder`, `NoAtomMapWarning`, 4 atom-map helpers |
| `parsers.py` | 329 | PDB / PQR / ProDy parsers with deduped common stage |
| `monomer.py` | 295 | `Monomer` |
| `templates.py` | 273 | `ResidueTemplate`, `ResidueChemTemplates` |
| `rotamers.py` | 116 | `residues_rotamers` table + `add_rotamers_to_polymer_molsetups` |
| `__init__.py` | 74 | backward-compat re-exports |
| `errors.py` | 37 | `PolymerCreationError` |

Total: **3584 LOC across 9 files** vs. 3597 LOC in one file — close to break-even on raw count, but the largest module (`polymer.py`) is now 58% the size of the original.

## What changed beyond moving code

**Stage 2 dedupe.** The PDB and PQR parsers share most of their pipeline (stream lines → accumulate `AtomField` blocks per residue → verify reskey→resname uniqueness → build one RDKit mol per residue via `_aux_altloc_mol_build`). The final stage is now a single `_build_residue_mols_from_blocks` helper with a `per_residue_postprocess` hook that the PQR variant uses to attach `PQRCharge` / `PQRRadius` props. The PQR line tokenizer and atom-record builder, previously nested inside the parser body, became `_get_pqr_atom_items` and `_atom_from_pqr_items` module-level helpers. The ProDy parser stays distinct (it operates on a hierarchical ProDy object instead of streamed text) but lives in the same `parsers.py` for cohesion.

**Stage 3 network fetch.** The 50-line CCD-fetching block that used to live inside `Polymer.__init__` is now `_resolve_unknown_residues_via_ccd`, a module-level helper. `__init__` gained `allow_template_fetch: bool = True`. The default preserves existing behavior; passing `False` raises `PolymerCreationError` with explicit guidance instead of silently reaching out to `rcsb.org`. Construction code in tests or notebooks where network access is undesired can now opt out without monkey-patching.

## API stability

Every previous import still resolves:

```python
from meeko import (
    Polymer, Monomer, ResiduePadder, ResidueTemplate, ResidueChemTemplates,
    PolymerCreationError, add_rotamers_to_polymer_molsetups,
)
```

Inside `meeko/polymer/polymer.py`, the four parser-related `@staticmethod`s (`_add_if_new`, `_pdb_to_residue_mols`, `_pqr_to_residue_mols`, `_prody_to_residue_mols`) are now one-line wrappers around the `parsers.py` functions; no in-tree caller had to change.

## Status of `design_analysis_v1.md` priorities

| # | Task | Status |
|---|---|---|
| 1 | Flatten static-only classes | ✅ done in log 01 |
| 2 | Polymer split | ✅ done in log 02 |
| 3 | `MoleculePreparation` → `PrepConfig` | **TODO** (next) |
| 4 | Type the flexibility model | ✅ done in log 00 |
| 5 | Split `molsetup.py` | ✅ done in log 00 |
| 6 | Dedupe `geomutils.py` | TODO |
| 7 | Extract CLI logic | TODO |

**4 of 7 done.** Remaining: `MoleculePreparation` config object, `geomutils.py` dedupe, and CLI extraction.

## Suggested next move

Either:

- **C.** `MoleculePreparation` → `PrepConfig` dataclass + `prepare(mol, cfg)` function. Removes the 30-kwarg constructor and the `temp_compute_charges` self-mutation hack inside `prepare()`. Touches every caller of `MoleculePreparation(...)`, including both CLIs.
- **D.** Dedupe `geomutils.py` (483 LOC with duplicated `vector`/`normalize`, hand-written linear algebra carrying TODO comments to switch to NumPy). Pure cleanup, no API surprises.

C is higher value but touches more files. D is mechanical and quick.
