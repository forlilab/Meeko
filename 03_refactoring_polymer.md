# Refactoring plan — `polymer.py` god module (Task B)

## Scope

`meeko/polymer.py` is 3597 LOC in one file, holding seven classes and ~14 module-level utility functions. The `Polymer` class alone is ~1700 LOC. The mission is to:

1. **Split the file into a package** so each class lives in its own module (mirrors the `meeko/molsetup/` strategy from log 00).
2. **Dedupe** the three near-identical `_pdb_to_residue_mols` / `_pqr_to_residue_mols` / `_prody_to_residue_mols` static methods on `Polymer`.
3. **Pull network side effects out of `Polymer.__init__`**: it currently does CCD-fetching (`build_noncovalent_CC`, `build_linked_CCs`) inside the constructor.

External API preserved throughout — `from meeko import Polymer, Monomer, ResiduePadder, ResidueTemplate, ResidueChemTemplates, PolymerCreationError, add_rotamers_to_polymer_molsetups` keeps working.

## What lives where

Inventory of `polymer.py` (LOC ranges approximate):

| Symbol | Lines | Kind |
|---|---|---|
| Top-level utility functions | 103–806 | 14 free functions (graph paths, bond finding, MCS mapping, charge math, H position updates, residue deletion) |
| `PolymerCreationError` | 807–836 | Exception subclass |
| `handle_parsing_situations` | 837–885 | Free function |
| `ResidueChemTemplates` | 886–1073 | 187 LOC class |
| `Polymer` | 1074–2725 | 1700 LOC **god class** |
| `add_rotamers_to_polymer_molsetups` | 2727–2800 | Free function (re-exported) |
| `Monomer` | 2801–3116 | 315 LOC class |
| `NoAtomMapWarning` | 3117–3124 | 7 LOC logging filter |
| `ResiduePadder` | 3125–3465 | 340 LOC class |
| Padder helpers | 3408–3465 | 4 free functions |
| `ResidueTemplate` | 3466–end | ~130 LOC class |

Module-level data: `residues_rotamers` (lines 60–100, 40 LOC sidechain SMARTS table).

## Stage 1 — split the file into a package (low risk, mechanical)

Convert `meeko/polymer.py` → `meeko/polymer/` package. Same atomic-rename strategy used for `molsetup`:

```
meeko/polymer/
├── __init__.py        re-exports for backward compat
├── errors.py          PolymerCreationError
├── utils.py           find_graph_paths, find_inter_mols_bonds*,
│                      mapping_by_mcs, _snap_to_int,
│                      divide_int_gracefully, rectify_charges,
│                      get_updated_positions, update_H_positions,
│                      _delete_residues, handle_parsing_situations
├── padder.py          ResiduePadder, NoAtomMapWarning,
│                      get_molAtomMapNumbers, remove_unmapped_atoms_from_mol,
│                      apply_atom_mappings, remove_atoms_with_mapping
├── templates.py       ResidueTemplate, ResidueChemTemplates
├── monomer.py         Monomer
├── rotamers.py        residues_rotamers, add_rotamers_to_polymer_molsetups
└── polymer.py         Polymer
```

Each file imports siblings as needed. `__init__.py` re-exports every previously top-level name so `from meeko.polymer import X` still resolves.

Single commit. Tests pass. No logic changes.

## Stage 2 — dedupe `_*_to_residue_mols` parsers (medium risk)

The three static methods on `Polymer` (PDB / PQR / ProDy) share a common shape: iterate residues from an upstream source, build per-residue RDKit mols, accumulate `bonds` and `raw_input_mols`. Extract a single parser-agnostic builder that takes a per-source "atom-record iterator" adapter.

Tentatively:
```
meeko/polymer/parsers.py
    def build_residue_mols(atom_records, ...): ...   # common body

meeko/polymer/parsers_pdb.py
    def pdb_string_to_residue_mols(pdb_str): ...    # was Polymer._pdb_to_residue_mols
meeko/polymer/parsers_pqr.py
    def pqr_string_to_residue_mols(pqr_str): ...
meeko/polymer/parsers_prody.py
    def prody_to_residue_mols(ag): ...
```

The three `Polymer.from_pdb_string` / `from_pqr_string` / `from_prody` classmethods then become very short.

This is medium risk because the three current functions are not byte-identical and likely diverged over time. Cross-test each parser against the existing test fixtures before declaring success.

Commit after each parser is migrated (3 commits).

## Stage 3 — pull network side effects out of `__init__` (medium-high risk)

`Polymer.__init__` currently:
- Validates input.
- Runs `build_noncovalent_CC` / `build_linked_CCs` — these hit `rcsb.org` to fetch CCD templates for unknown residues.
- Calls `self._get_monomers(...)`.
- Calls `self._build_padded_mols(...)`.
- Optionally calls `self.parameterize(mk_prep)`.

The right shape: a constructor that constructs, plus an explicit builder.

Sketch:
```python
class Polymer:
    def __init__(self, monomers, bonds, residue_chem_templates, ...):
        # just assign fields
        ...

    @classmethod
    def build(cls, raw_input_mols, bonds, residue_chem_templates, *,
              mk_prep=None, set_template=None, blunt_ends=None,
              get_atomprop_from_raw=None, ignore_https_cert=False,
              forgive_extra_bonds=False, allow_template_fetch=True):
        # current __init__ body lives here; explicit allow_template_fetch
        ...

# Backward compat: keep the old __init__ signature, route it through build()
```

`from_pdb_file` / `from_pdb_string` / `from_pqr_string` / `from_prody` already act like factories — they will route through `build()` too.

The explicit `allow_template_fetch` flag makes the network behavior visible at the call site, instead of being a hidden surprise.

One commit per logical change.

## Success criteria (after each commit, no exceptions)

- 92 tests pass, 4 skipped — the baseline.
- `from meeko import Polymer, Monomer, ResiduePadder, ResidueTemplate, ResidueChemTemplates, PolymerCreationError, add_rotamers_to_polymer_molsetups` continues to resolve.
- No regression in CLI behavior (`mk_prepare_receptor` is the heaviest consumer).

## Non-goals for this pass

- **Not** changing the JSON serialization format.
- **Not** touching `chemtempgen.py` (the CCD template generator). That's its own 1132 LOC file and deserves a separate pass.
- **Not** moving `Monomer.parameterize` / `flexibilize` / `rigidify` semantics — only their location.
