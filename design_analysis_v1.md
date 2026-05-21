# Meeko — First-Pass Critical Assessment

## 1. High-level shape

```
meeko/  ~13 kLOC core + ~5 kLOC subpackages
├── preparation.py      787   MoleculePreparation  (the "entry point")
├── molsetup.py        2144   MoleculeSetup, RDKitMoleculeSetup, Atom, Bond, Ring, ...
├── polymer.py         3597   Polymer, Monomer, ResidueChemTemplates, ResiduePadder, ResidueTemplate
├── chemtempgen.py     1132   ChemicalComponent + ~15 module-level helpers
├── writer.py           789   PDBQTWriterLegacy + module-level OIDS writers
├── molecule_pdbqt.py   692   PDBQTMolecule
├── receptor_pdbqt.py   261   PDBQTReceptor
├── atomtyper.py        403   AtomTyper + AtomicGeometry
├── hydrate.py          461   Waters, Hydrate, HydrateMoleculeLegacy
├── macrocycle.py       376   FlexMacrocycle
├── flexibility.py      434   (pure functions — the one place that resists OO bloat)
├── cli/mk_prepare_*.py 1184+697   monolithic main() (~700 LOC each)
└── utils/              ~2 kLOC mixed pure utilities
```

A pipeline shape is implied — `mol → MoleculePreparation → MoleculeSetup → AtomTyper → BondTyper → FlexMacrocycle → flexibility → PDBQTWriterLegacy` — but it is not made explicit anywhere; you reconstruct it by reading `MoleculePreparation.prepare` (preparation.py:522).

## 2. God classes / oversized modules

### `MoleculeSetup` / `RDKitMoleculeSetup` (molsetup.py)
- 2144 LOC, ~35 methods across 3 stacked classes (`MoleculeSetup`, abstract `MoleculeSetupExternalToolkit`, `RDKitMoleculeSetup`).
- Holds: atoms, bonds, rings, ring-closure info, rotamers, atom params, restraints, flexibility model, charges, dihedral interactions, PDB info, OFF dihedral series, symmetry, RDKit conformer manipulation, JSON encode/decode, plus `show()` debug printing.
- It is the de facto in-memory format and *everything* depends on its attribute names — that is why `flexibility.py` mutates dicts on it and why `atomtyper.py` reaches in to set `.atom_params`. Mixing a data container, a builder, a serializer, and a chemistry-aware editor in one class is the largest single source of coupling.

### `Polymer` (polymer.py)
- 3597 LOC in one module, with the class itself at ~1700 LOC, 30+ methods, plus four `from_*` constructors (`from_pdb_file`, `from_pdb_string`, `from_pqr_string`, `from_prody`) and three nearly duplicated `_*_to_residue_mols` static methods (pdb / pqr / prody) — classic three-fold duplication that should be one parser-agnostic builder fed by adapters.
- `__init__` is ~200 lines (polymer.py:1095–1290) and silently does template-fetching from the PDB CCD inside the constructor (`build_noncovalent_CC`, `build_linked_CCs`), which is a network side effect hidden in object construction.
- Holds polymer-state, business logic (parameterize/flexibilize/rigidify), parsing, padding, and serialization. Splitting into `PolymerData` / `PolymerBuilder` / `PolymerWriter` (and moving `Monomer`, `ResiduePadder`, `ResidueTemplate`, `ResidueChemTemplates` to their own files) would already shrink it dramatically.

### `MoleculePreparation` (preparation.py)
- ~30 constructor kwargs (preparation.py:84–117) — a textbook *parameter object smell*. `from_config` / `from_json_file` / `get_defaults_dict` confirm it: configuration is really its own concept and should be a dataclass/`PrepConfig`.
- `prepare()` is a 190-line orchestration script (preparation.py:522–711) that pokes at `self.compute_charges` temporarily (`temp_compute_charges` hack at 575–578, 627–629) — a side-effecting mutation of the object during a single call. That's a bug magnet.
- Conceptually it is a *function* with state for configuration only; it should be `prepare(mol, cfg)` rather than `MoleculePreparation(...).prepare(mol)`.

### `PDBQTWriterLegacy` (writer.py)
- 100% `@classmethod` / `@staticmethod` (writer.py:343–end). No state. This is **a module masquerading as a class** — five static methods plus `write_string`, `write_string_from_polymer`, etc. should be free functions. The class adds nothing except a namespace, and `oids_block_from_setup` lives outside it as a free function anyway, so the convention is already inconsistent.

### `AtomTyper` (atomtyper.py)
- Same pattern: every method is `@classmethod` / `@staticmethod`. There is no `self` anywhere. It is six pure functions wrapped in a class for namespacing. Same critique as `PDBQTWriterLegacy`.

### `RDKitMolCreate` (rdkit_mol_create.py)
- Same: only static/class methods, no state. Pure functions in a class costume.

### CLI scripts (cli/mk_prepare_ligand.py, mk_prepare_receptor.py)
- `mk_prepare_receptor.py:main` is ~500 lines (393–~890) of nested conditionals: arg parsing, IO, dispatch, validation, error reporting, and even a nested function `warn_flexres_outside_box` defined inside `main` at line 899. The CLIs are essentially a second, undocumented API surface and should be split: argument schema → config → core call → reporting.

## 3. Pure functions hiding inside classes

The recurring antipattern: **a class with no instance state, just `@staticmethod`/`@classmethod`** (count of `@staticmethod` alone: `polymer.py` 15, `atomtyper.py` 5, `writer.py` 5, `molsetup.py` 5, `hydrate.py` 3, `rdkit_mol_create.py` 3). Most of these are pure transformations of (molsetup, params) → result and would be clearer as module-level functions — exactly the way `flexibility.py` is already written. `flexibility.py` is the cleanest module in the codebase precisely because it resisted that temptation.

Concrete candidates to flatten:
- `AtomTyper.*` → `atomtyper.type_atoms(...)`, `atomtyper.cache_offatoms(...)`, etc.
- `PDBQTWriterLegacy.*` → free functions in `writer.py`.
- `RDKitMolCreate.*` → free functions in `rdkit_mol_create.py`.
- `Polymer._build_padded_mols`, `_pdb_to_residue_mols`, `_pqr_to_residue_mols`, `_prody_to_residue_mols`, `_get_monomers`, `_build_rdkit_mol`, `_get_best_missing_Hs` — all static, all algorithmic, none need `Polymer` state. Move to a `polymer/builders.py`.
- `MoleculeSetup.get_bonds_in_ring` (static), `get_symmetries_for_rmsd`, `has_implicit_hydrogens` (RDKitMoleculeSetup) → module-level.

## 4. Inverse problem: free functions that *should* be methods or grouped

- `flexibility.py` exposes `get_flexibility_model`, `merge_terminal_atoms`, `update_closure_atoms`, `walk_rigid_body_graph` as bare functions that all mutate a "flex_model dict" living on `MoleculeSetup.flexibility_model`. The `flexibility_model` deserves to be an actual class (or dataclass) rather than a dict-of-strings keyed magically — currently both serialization (`molsetup.py:489–502, 542–557`) and consumers need to know its private keys (`rigid_body_connectivity`, `rigid_body_graph`, `rigid_body_members`, `rigid_index_by_atom`).
- `chemtempgen.py` has 15 module-level functions (`embed`, `cap`, `deprotonate`, `recharge`, `get_smiles_with_atom_names`, `get_pretty_smiles`, ...) that almost all operate on a `Chem.Mol` and would naturally hang off `ChemicalComponent` or a `ResidueTemplateBuilder`.
- `utils/geomutils.py` mixes NumPy-ready code with hand-written linear algebra that has TODO comments saying "use NumPy" (geomutils.py:303, 314, 319, 331, 338, 361). Several functions are duplicated (`vector` defined at lines 19 and 303; `normalize` at 38 and 319). This file needs deduplication before refactoring touches it.

## 5. Other smells worth flagging

- **Circular-import workarounds**: `flexibility_model` left as a dict to "resolve circular imports" (molsetup.py:461–462) — that is a structural smell, not a runtime constraint; the cycle is between `molsetup` and `flexibility`, and breaks if `FlexibilityModel` lives in its own module.
- **Mixed serialization style**: `BaseJSONParsable` with `json_encoder` / `_decode_object` is custom rather than using `dataclasses` + `json`/`pydantic`. Every class re-implements field-by-field encode/decode with `tuple_to_string`/`string_to_tuple` adapters because dict keys aren't JSON-native (molsetup.py:472–502). The data layer should use dataclasses with explicit (de)serializers — current code drifts every time a field is added.
- **`HydrateMoleculeLegacy`** still exists alongside `Hydrate` (hydrate.py:74 vs 214). The "Legacy" suffix is also on `PDBQTWriterLegacy`. Either there's a planned successor that never landed, or these are kept for compatibility that should be documented.
- **Side effects in `__init__`**: `Polymer.__init__` hits the network (CCD fetch) and runs `self.parameterize` if `mk_prep` is given. Constructors should construct.
- **`tmp/` directory** sits inside the package — likely shouldn't ship.

## 6. Suggested refactoring priorities (cheapest → highest value)

1. **Flatten static-only classes** (`AtomTyper`, `PDBQTWriterLegacy`, `RDKitMolCreate`) to modules. Mechanical, low-risk, removes ~200 LOC of `cls`-self ceremony.
2. **Split `Polymer` into module + builders**: separate parsing (pdb/pqr/prody) from data, dedupe the three `_*_to_residue_mols`.
3. **Promote `MoleculePreparation` config to a dataclass `PrepConfig`** and make `prepare(mol, cfg)` a function. Removes the `temp_compute_charges` self-mutation hack.
4. **Type the flexibility model**: `FlexibilityModel` dataclass, move dict-key knowledge out of `molsetup.json_encoder`.
5. **Split `molsetup.py`**: `atom.py`, `bond.py`, `ring.py`, `restraint.py`, `setup.py`, `rdkit_setup.py`. Each <300 LOC.
6. **Dedupe `geomutils.py`** and migrate to NumPy as the existing TODO comments instruct.
7. **Extract CLI logic** into testable functions; `main()` should be ~30 lines of arg-parsing + dispatch.
