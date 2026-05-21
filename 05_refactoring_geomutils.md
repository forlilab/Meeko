# Refactoring plan — `geomutils.py` dedupe (Task D)

## What's wrong with the current shape

`meeko/utils/geomutils.py` is 483 LOC with three categories of problems:

### Duplicate definitions
- `vector` defined at line 19 (`def vector(a, b): return b - a`) and again at line 303 (`def vector(p1, p2=None, norm=0)`). The second wins; the first is dead source.
- `normalize` defined at line 38 and again at line 319. Both bodies are equivalent (`v / np.sqrt(np.dot(v, v))` vs `A / np.sqrt(sum(A*A))`).

### Stale TODO comments
The file has TODO comments saying "use Numpy" on functions that already use NumPy (`norm`, `normalize`, `vector`, `dot`, `vecAngle`, `vecSum`). Old guidance, no action needed.

### Dead code that would crash on call
Six functions reference undefined names — they don't fail at import, but any caller dies immediately. None have callers anywhere in the live codebase:

| Function | What's wrong |
|---|---|
| `get_vector_normal` | `np.array(vector[1], vector[0], vector[2])` — wrong np.array signature |
| `getVecNormalToVec` | docstring has un-indented `if/else` Python in it; calls `calcPlane(vec, c, norm=True)` but `calcPlane` doesn't take a `norm` kwarg |
| `makeCircleOnPlane` | uses undefined `array`, `PI2`, `cos`, `sin`, `cross`, `vecSum` |
| `atomsToVector` | calls undefined `atomCoord` |
| `avgVector_untested` | function literally prints `"NOT WORKING!!!! NEVER TESTED"` and references unbound `m`/`ax` |
| `gaussian` / `ellipticGaussian` | use undefined `e` (Euler's number) |

There is also `get_vector` with a typo (parameter `coor1` but body uses `coord1`), with zero callers.

## Plan

### Scope: narrow

The original directive in `design_analysis_v1.md` was "dedupe `geomutils.py` and migrate to NumPy as the existing TODO comments instruct." The migration is already done — every function already uses NumPy; the TODOs are stale. The remaining task is dedupe + remove the obvious dead/broken code.

### Out of scope

- **Not** deleting `meeko/tmp/` (separate concern; that's the only "consumer" of the broken `vecSum`/`calcPlane` etc., and `meeko/tmp/` is not imported by any live code anyway).
- **Not** removing zero-caller-but-functional helpers (`calcDihedral`, `dot`, `norm`, `calcPlane`, `coplanar`, `vecSum`, `vecAngle`, `quickdist`, `normValue`, `normProduct`, `averageVector`, `calcRingCentroidNormal`, `averageCoords`, `calcPlaneVect`, `rotation_matrix`, `rotate_around_axis`, `absoluteAngleDifference`). These look like public utility surface — keep them.

### Concrete changes (one commit)

1. **Collapse the two `vector` defs into one.** Use the line-303 signature (positional `p1`, optional `p2`, optional `norm`) since downstream calls use the `vector(a, b)` form which works under both. Replace the `np.array(..., 'f')` float32 cast with `np.asarray(p2) - np.asarray(p1)` to preserve input dtype.
2. **Collapse the two `normalize` defs into one** — the line-38 form (`v / np.sqrt(np.dot(v, v))`).
3. **Drop the stale `# TODO use Numpy` comments** on functions that already use NumPy.
4. **Fix `get_vector`** — typo `coord1` → `coor1` (or just remove since it has zero callers).
5. **Remove the six provably-broken / pure-crash functions** listed above. No live caller exists in the codebase (the only references are inside `meeko/tmp/`, which is dead).

### Behavior preserved

Every live consumer (`meeko/hydrate.py` only) keeps working: `vector`, `resize_vector`, `normalize`, `rotation_axis`, `atom_to_move`, `rotate_point` are all preserved with identical observable behavior.

### Risk

Low. The duplicates already have one variant shadowing the other; collapsing to a single def doesn't change runtime behavior except for the dtype-cast issue (which we handle deliberately). The six removed functions can't be in active use — they crash on call.

## Success criteria

- 92 tests pass, 4 skipped (baseline).
- `meeko/hydrate.py` is the canary — if the file imports and runs its tests, the dedupe is sound.
- File shrinks from 483 → roughly 300 LOC.
