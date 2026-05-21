# Refactoring plan — `MoleculePreparation` (Task C)

## What's wrong with the current shape

`meeko/preparation.py` (787 LOC) defines one class, `MoleculePreparation`, with:

- A 30-kwarg constructor (preparation.py:84–117) — a textbook parameter-object smell.
- A 190-line `prepare()` method (preparation.py:522–711) that mutates `self.compute_charges` mid-call:

  ```python
  temp_compute_charges = None
  if template_charge is None and self.compute_charges is False:
      temp_compute_charges = self.compute_charges
      self.compute_charges = True
  ...
  if temp_compute_charges is not None:
      self.compute_charges = temp_compute_charges
  ```

  That pattern means a single `prepare()` call temporarily flips a piece of "configuration" — concurrent calls or interrupted calls leave the instance in an inconsistent state.
- `get_defaults_dict()` recovers the constructor signature via `inspect.signature`. That's a clue that the kwargs *want* to be a config object.

## Goals

1. Introduce a `PrepConfig` dataclass that owns the 30 simple configuration knobs.
2. Make `MoleculePreparation` build one in `__init__` and keep using it as the source of truth, while preserving every `self.foo` attribute external callers already touch (e.g. `mk_prep.charge_model`).
3. Eliminate the `temp_compute_charges` self-mutation hack by passing `compute_charges` as an explicit local in `prepare()`.
4. Add a public factory entry-point that takes a `PrepConfig` directly:
   `MoleculePreparation.from_prep_config(cfg)`.

## Non-goals (yet)

- **Not** converting `prepare()` into a free function. The class still owns derived state (`atom_params` built from files, `dihedral_params` from OpenFF, `_bond_typer`, `_macrocycle_typer`, `_water_builder`, `espaloma_model`) — turning the orchestration into a stand-alone function requires also moving that derived state somewhere, which is its own pass.
- **Not** changing CLI flags. `mk_prepare_ligand.py` and `mk_prepare_receptor.py` keep parsing args into a dict and calling `MoleculePreparation.from_config(dict)`.
- **Not** removing `from_config` / `from_json_file` / `get_defaults_dict` — they remain backward-compat shims.

## External API contract preserved

These keep working unchanged:

| Pattern | Where |
|---|---|
| `MoleculePreparation(**kwargs)` | all tests, CLIs |
| `MoleculePreparation.from_config(dict)` | CLI builds config from argparse + JSON |
| `MoleculePreparation.from_json_file(path)` | CLI alt path |
| `MoleculePreparation.get_defaults_dict()` | CLI builds defaults |
| `mk_prep(mol)` / `mk_prep.prepare(mol)` | every prep call site |
| `mk_prep.calc_flex(...)` | called by `Monomer.flexibilize` (polymer/monomer.py:258) |
| `mk_prep.charge_model` (attr read) | CLI introspection |
| `mk_prep.setup` (deprecated) | legacy users |

## Stages

### Stage 1 — introduce `PrepConfig`

- Create `meeko/prep_config.py` with a `@dataclass`-based `PrepConfig`.
- Fields are the 30 constructor kwargs of `MoleculePreparation` (defaults preserved exactly).
- `PrepConfig.from_dict`, `PrepConfig.to_dict`, `PrepConfig.from_json_file`.
- No changes to `MoleculePreparation` yet — purely additive.

### Stage 2 — use it inside `MoleculePreparation`

- `MoleculePreparation.__init__` now builds `self._config = PrepConfig(**kwargs)` first, then assigns `self.foo = self._config.foo` for every config field (backward compat).
- Validation moves into `PrepConfig.__post_init__` where it doesn't depend on derived state; the cross-field validations stay in `__init__`.
- `get_defaults_dict()` now reads field defaults from the dataclass instead of using `inspect.signature(cls)`.
- Add `MoleculePreparation.from_prep_config(cfg)` factory.

### Stage 3 — kill the `temp_compute_charges` hack

- In `prepare()`, replace the four-line save/restore dance with a local `compute_charges` variable derived from `self.compute_charges` and the `template_charge` argument. No more writing to `self`.
- Audit `RDKitMoleculeSetup.from_mol` / `init_atom` to confirm the `temp_compute_charges` reset *inside* the molsetup is also self-contained (already pulled into the rdkit_adapter in log 00; should be fine).

### Stage 4 — tests pass, commit each stage

92 tests pass, 4 skipped (baseline) after every stage.

## Risk assessment

- **Low risk** for stages 1–2: additive + 1-line constructor changes; tests already exercise every constructor path through the CLIs.
- **Medium risk** for stage 3: the hack was added for a reason — `template_charge` is set in `Monomer.parameterize`, which calls `mk_prep(mol=..., template_charge=...)`. Need to confirm no other code path reads `self.compute_charges` while `prepare()` is running. (It would have to be on the same thread, since Python's GIL precludes preemption — and `prepare()` is straight-line code, no callbacks. The risk is conceptual rather than concrete.)
