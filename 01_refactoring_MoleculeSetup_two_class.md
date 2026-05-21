# Refactoring `MoleculeSetup`: two-class split vs. multi-file design

Yes — the two-class split is viable, simpler to land, and a strict improvement over today. But it doesn't solve every problem the multi-file design solves. Think of them as **two stops on the same road**, not alternatives.

## The two-class shape

```
MoleculeData          # pure dataclass — fields only, no behavior
├── atoms: list[Atom]
├── bond_info: dict[tuple, Bond]
├── rings, rotamers, atom_params, restraints
└── flexibility_model: FlexibilityModel   # ← also a dataclass, not a dict

MoleculeSetup         # operates on MoleculeData
├── self.data: MoleculeData
├── add_atom / delete_atom / add_bond / delete_bond      # invariant preservers
├── merge_terminal_atoms / clean_atoms                   # editing
├── set_atom_type_from_uniq_atom_params                  # chemistry-aware mutation
├── get_neighbors / get_bonds_in_ring                    # graph queries
├── show / write_coord_string                            # debug
└── to_json / from_json                                  # serialization (delegates to data)
```

External callers still write `setup.add_atom(...)` — API surface barely changes. Downstream code keeps working.

## What it fixes vs. what it doesn't

**Fixes:**
- Serialization becomes trivial (`MoleculeData` is a dataclass → `dataclasses.asdict` + tuple-key adapters → ~20 lines instead of 80).
- `FlexibilityModel` as a proper class eliminates the dict-key magic in encode/decode.
- You can pass `data` to pure functions without dragging the ops layer along — testability win.
- Forces explicit invariants: the ops class is the *only* thing that can mutate `data` safely.

**Doesn't fix:**
- **The ops class is still a 30-method god class** — just sitting on top of a thinner state. The categorical mess (editing vs. algorithms vs. RDKit-interop vs. debug) is still there.
- **RDKit coupling** stays. You still need somewhere for `from_mol`, `find_pattern`, `perceive_rings`, `calculate_charges`, `get_charges_from_template`. Your options:
  - (a) Put them on `MoleculeSetup` — RDKit is now a hard dependency of the ops class.
  - (b) Subclass `RDKitMoleculeSetup(MoleculeSetup)` — you've reinvented the current ABC mixin pattern, just on a thinner base.
  - (c) Keep an `rdkit_adapter` module alongside the two classes — now it's a *2.5-class* design, which is fine, but it's the multi-file design starting to leak in.
- **`atom_params` dict opacity** is unchanged — that's a typing problem, not a class-count problem.

## Honest take

**Start with the two-class split.** It's:
- ~1–2 days of work versus ~1–2 weeks for the full split.
- Almost no downstream API breakage.
- Easy to back out if it feels wrong.
- Captures the highest-value win (clean serialization, typed flexibility model).

If after living with it for a while you find `MoleculeSetup` is still doing too much — particularly if the RDKit-interop methods feel out of place — *then* push to the multi-file design as a v2 refactor. The two-class split is on the path to the multi-file design; it's not a dead-end.

The one piece I'd still pull out **even in the two-class version**: the RDKit-specific construction (`from_mol`, `find_pattern`, `perceive_rings`). Make those module-level functions in a `rdkit_adapter.py` from day one — that alone kills the `MoleculeSetupExternalToolkit` ABC and the `RDKitMoleculeSetup` subclass without forcing a wider split. So the realistic minimum is **two classes + one adapter module**, which is a much cleaner stable point than pure two-class.

## Recommended starting point

```
meeko/molsetup.py        # contains MoleculeData + MoleculeSetup
meeko/rdkit_adapter.py   # contains from_rdkit_mol, find_pattern, perceive_rings,
                         # get_charges_from_template, etc.
```

Two new things to land:
1. **`MoleculeData`** — dataclass with all the fields currently on `MoleculeSetup`. Includes a typed `FlexibilityModel` dataclass for the `flexibility_model` slot.
2. **`rdkit_adapter`** module — RDKit-coupled free functions that take/return `MoleculeData` (or build a `MoleculeSetup`).

`MoleculeSetup` becomes a thin wrapper: it owns a `MoleculeData`, exposes invariant-preserving mutators (`add_atom`/`add_bond`/...) and high-level operations (`merge_terminal_atoms`, `set_atom_type_from_uniq_atom_params`, etc.), but delegates all serialization to the dataclass and all RDKit interop to the adapter.

## Migration sequence

1. Define `MoleculeData` and `FlexibilityModel` dataclasses; move fields off `MoleculeSetup`. All `self.foo` accesses become `self.data.foo`. Mechanical, can be done with regex + tests.
2. Replace `json_encoder` / `_decode_object` with dataclass-based serialization.
3. Extract `from_mol` / `find_pattern` / `perceive_rings` / `get_charges_from_template` into `rdkit_adapter.py`. Delete `MoleculeSetupExternalToolkit` and `RDKitMoleculeSetup`.
4. Stop here. Re-evaluate before pushing further.
