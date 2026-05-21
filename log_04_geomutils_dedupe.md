# Log 04 — `geomutils.py` dedupe (Task D)

The 483-LOC `geomutils.py` is now 313 LOC. The duplicate `vector` and `normalize` definitions are gone, the stale "TODO use NumPy" comments on already-NumPy code are gone, and seven provably-broken functions (would crash on call due to undefined names) are gone too. Per the plan in `05_refactoring_geomutils.md`. **92 tests pass, 4 skipped.**

## Commit

| SHA | What | LOC delta |
|---|---|---|
| `6d1b1ab` | Dedupe + drop broken dead code | +216 / −386 (net −170) |

## What was removed

### Duplicates collapsed

| Symbol | Previous shape | Now |
|---|---|---|
| `vector` | two defs (lines 19 + 303); second won; first dead source | single def `vector(p1, p2=None, norm=False)`; `np.asarray(..., dtype=float)` instead of `np.array(..., 'f')` to preserve input dtype |
| `normalize` | two defs (lines 38 + 319), equivalent bodies | single def `v / np.sqrt(np.dot(v, v))` |

### Broken-on-call dead code

Seven functions that would crash immediately if anyone tried to call them, with zero in-tree callers:

- `get_vector` — typo `coor1` / `coord1`
- `get_vector_normal` — wrong `np.array(...)` signature
- `getVecNormalToVec` — calls `calcPlane(norm=True)` but `calcPlane` has no `norm` kwarg
- `makeCircleOnPlane` — uses undefined `array`, `PI2`, `cos`, `sin`, `cross`, `vecSum`
- `atomsToVector` — calls undefined `atomCoord`
- `avgVector_untested` — literally prints `"NOT WORKING!!!! NEVER TESTED"` and references unbound `m`, `ax`
- `gaussian` / `ellipticGaussian` — both use undefined `e` (Euler's number)

Plus `vecSum` (used `array(...)` via undefined name; zero live callers).

The only references to any of these were inside `meeko/tmp/` (which is not imported by any live code; flagged separately in `design_analysis_v1.md` as "shouldn't ship").

### Stale comments

`# TODO use Numpy` removed from `norm`, `normalize`, `vector`, `dot`, `vecAngle`, `vecSum` — every one already used NumPy. Legacy commented-out blocks and dead branches also cleaned.

## What stayed

Functional but currently zero-caller helpers are preserved — they look like public utility surface re-exported via `from meeko import geomutils`:

`vector`, `normalize`, `norm`, `resize_vector`, `dot`, `vecAngle`, `absoluteAngleDifference`, `averageCoords`, `averageVector`, `quickdist`, `calcPlane`, `calcPlaneVect`, `coplanar`, `rotation_matrix`, `rotate_around_axis`, `rotation_axis`, `atom_to_move`, `rotate_point`, `calcDihedral`, `calcDihedral_old`, `calcRingCentroidNormal`, `normValue`, `normProduct`.

## Live consumer (canary)

`meeko/hydrate.py` is the only live consumer (uses `vector`, `resize_vector`, `normalize`, `rotation_axis`, `atom_to_move`, `rotate_point`). It imports and instantiates cleanly post-refactor; all `Hydrate`-touching tests pass.

## Status of `design_analysis_v1.md` priorities

| # | Task | Status |
|---|---|---|
| 1 | Flatten static-only classes | ✅ log 01 |
| 2 | Polymer split | ✅ log 02 |
| 3 | `MoleculePreparation` → `PrepConfig` | ✅ log 03 |
| 4 | Type the flexibility model | ✅ log 00 |
| 5 | Split `molsetup.py` | ✅ log 00 |
| 6 | Dedupe `geomutils.py` | ✅ log 04 |
| 7 | Extract CLI logic | TODO |

**6 of 7 done.** Last remaining: pull the ~500-line `main()` out of `cli/mk_prepare_receptor.py` into testable functions. (`mk_prepare_ligand.py` has the same shape but smaller, ~700 LOC total.)

## Loose ends worth a follow-up

- **`meeko/tmp/` directory** flagged in the original design analysis. It's never imported, hosts the only references to the dead functions removed today (`vecSum`, `makeCircleOnPlane`, etc.), and shipping it inside the package is what the analysis called out. Worth a separate `git rm` once stakeholders agree it's safe to drop.
