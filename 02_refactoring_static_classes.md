# Refactoring plan — flatten static-only classes

## Target classes

Three classes in the codebase are namespaces in disguise: every method is decorated `@staticmethod` or `@classmethod`, none use `self`, and the class itself carries no state. They exist purely to group functions.

| Class | File | LOC | Method count |
|---|---|---:|---:|
| `AtomTyper` | `meeko/atomtyper.py` | 403 | 6 (5 static + 1 classmethod) |
| `PDBQTWriterLegacy` | `meeko/writer.py` | 789 | 10 (5 static + 5 classmethod) |
| `RDKitMolCreate` | `meeko/rdkit_mol_create.py` | 455 | 7 (3 static + 4 classmethod) |

## What changes

For each class:
1. Promote each method body to a module-level function.
2. Keep the class as a thin shim whose methods delegate to the module-level functions, so external callers (`AtomTyper.type_everything(...)`, `PDBQTWriterLegacy.write_string(...)`, `RDKitMolCreate.from_pdbqt_mol(...)`) continue to work unchanged.
3. New code can call the module-level functions directly (`atomtyper.type_everything(...)`).

The shim layer mirrors the `RDKitMoleculeSetup` strategy from stage 3 of the MoleculeSetup refactor: behavior identical, surface preserved, internals flattened.

## Order

1. **`AtomTyper`** (smallest, simplest) — proves the pattern.
2. **`RDKitMolCreate`** — similar shape, similar size.
3. **`PDBQTWriterLegacy`** — largest, but same mechanics.

Each lands as one commit. Each step runs the full test suite before commit; refuse to commit on regression.

## Non-goals

- **Not** changing function signatures — we're only flattening, not redesigning.
- **Not** removing the shim classes in this pass. They can be deprecated and eventually removed in a follow-up once external (downstream) callers have been audited.
- **Not** touching the `_walk_graph_recursive` recursive helper in writer.py beyond moving it (it stays a free function — its recursion pattern doesn't need flattening logic).

## Success criteria

- Each commit: 92 tests pass, 4 skipped (baseline).
- After all three commits: zero external API change. Imports like `from meeko import AtomTyper, PDBQTWriterLegacy, RDKitMolCreate` continue to resolve to working symbols.
- Each module's top-level grows; class body shrinks to one-liners.
