# Refactoring `MoleculeSetup`: separating data from operations

Short answer: **yes, but not all the way** — separate operations into layers rather than going fully anemic.

## What `MoleculeSetup` currently mixes

Looking at the 35 methods, they fall into ~7 distinct concerns:

| Concern | Examples | Belongs where |
|---|---|---|
| **State** | `atoms`, `bonds`, `rings`, `rotamers`, `atom_params`, `flexibility_model` | data class |
| **Invariant-preserving primitives** | `add_atom`, `delete_atom`, `add_bond`, `delete_bond` (must keep `Atom.graph` consistent) | **methods on data class** |
| **Plain accessors** | `get_charge`, `get_coord`, `get_atom_type`, `get_neighbors` | properties on `Atom`/`Bond`, drop wrappers |
| **Chemistry-aware mutation** | `merge_terminal_atoms`, `set_atom_type_from_uniq_atom_params`, `clean_atoms` | external functions in `editing.py` |
| **Algorithms** | `_recursive_graph_walk`, `get_bonds_in_ring` (already static), `perceive_rings` | external functions in `graph.py` |
| **External-toolkit bridge** | `from_mol`, `find_pattern`, `init_atom`, `calculate_charges`, `get_charges_from_template`, `get_conformer_with_modified_positions` | free functions in `rdkit_adapter.py` — **kills the `MoleculeSetupExternalToolkit` ABC entirely** |
| **Serialization, debug** | `json_encoder`, `_decode_object`, `show`, `write_coord_string` | dataclass auto-serialization + a `dump.py` module |

## Why the middle ground

- **Going fully data-only** (every method becomes a free function): you'll break invariants. `add_bond` doesn't just append to `bond_info` — it also updates `Atom.graph` on both endpoints. That's a primitive that *deserves* to be a method, because external callers will forget to update graphs.
- **Going fully methods-on-class** (status quo): everything becomes coupled. `atomtyper.py` already pokes at `setup.atom_params` directly; macrocycle/flexibility/writer all reach in. The class can't say no.

The line that works: **mutators that maintain structural invariants stay; everything else leaves.**

## Concrete proposed shape

```
meeko/molsetup/
├── atom.py            Atom dataclass + properties (drop get_charge wrappers)
├── bond.py            Bond, ring_id helper
├── ring.py            Ring, RingClosureInfo
├── restraint.py       Restraint
├── flex_model.py      FlexibilityModel dataclass (replaces the opaque dict)
├── setup.py           MoleculeSetup
│                        - fields only (dataclass)
│                        - add_atom / delete_atom / add_bond / delete_bond
│                          (the invariant-preservers, ~6 methods total)
├── editing.py         merge_terminal_atoms, clean_atoms, set_atom_types_from_*
├── graph.py           bonds_in_ring, recursive_graph_walk, neighbors_of
├── rdkit_adapter.py   from_rdkit_mol, find_pattern, perceive_rings,
│                      get_conformer_with_modified_positions, calculate_charges
└── io.py              JSON encode/decode (using dataclass fields)
```

## The big wins

1. **`MoleculeSetupExternalToolkit` ABC disappears** — there's no need for an abstract toolkit mixin if RDKit interop is a free function `from_rdkit_mol(mol) -> MoleculeSetup`. The "external toolkit" pattern was forced by inheritance; flatten it and the problem vanishes.
2. **No more inheritance** — `RDKitMoleculeSetup` is the same type as `MoleculeSetup`, just constructed by an RDKit-aware factory function. Subclass goes away.
3. **Serialization becomes ~30 lines** — dataclasses + a few converters for tuple-keyed dicts, instead of 80 lines of `json_encoder`/`_decode_object` per class.
4. **`atom_params` opacity** — currently `dict[str, list]` with implicit positional alignment to `atoms`. Either typed properly (`AtomParams` dataclass with per-atom records) or made an explicit lookup; either way the contract becomes inspectable.

## Tradeoffs to be honest about

- **Migration is wide**: every `setup.foo()` call in atomtyper/flexibility/macrocycle/writer/preparation/polymer needs auditing. Doable, but it's not a one-day job.
- **Discoverability**: `setup.<tab>` shows fewer things; you have to know that `editing.merge_terminal_atoms(setup, ...)` exists. A clean `meeko.molsetup` `__init__.py` re-export mitigates this.
- **You may discover the dict-typed `atom_params` is depended on positionally** by external code (third-party scripts). Worth a quick grep through `cli/`, `tools/`, and any docs/examples before locking in a typed replacement.

## Recommendation

Yes — start with the cleanest cut (RDKit adapter extraction + flexibility model dataclass). Those two alone eliminate the inheritance ABC and the dict-key-magic in serialization, with minimal blast radius. The full module split can follow once you've validated the call-site count.
