# model-kit: instructions for coding agents

Follow these steps for any 3D piece you build or check with Blender: a prop, a part of a stage, a
character or a tree. A piece is one object a game loads. Never pick a size, a viewing distance or
a limit so that a check passes. Measure it, cite it, or ask the person.

`model-kit guide --where` prints where this install keeps its bars and its kind rules.

## 1. Name the piece and its kind

Write down what the piece is, how big it is in real metres, and how it is used. Then name its kind:

- an assembly: a hard-surface piece built from parts that sit on each other (a lamp, a guitar, a
  truss). Read `kinds/assembly/rules.md`.
- a branching piece: a plant whose limbs fork from a parent and carry leaves. Read
  `kinds/branching/rules.md`. Trees are grown, not modelled: see "The tree grower" in
  [REFERENCE.md](REFERENCE.md).
- a rigged piece: a body with bones, a skin and clips. It follows the assembly rules and the body
  and clip checks.

## 2. Find the nearest bar

A bar is a finished reference piece that passes the checks. `bars/INDEX.md` lists each bar with
what it fits and what it is wrong for. Pick the bar that fits the piece and read its `README.md`.

- **Copy:** the bar's folder shape fits. Copy its `build.py`, its Blender script and its
  `style.json` into one folder of the project and change what the piece needs. `build.py` runs
  in plain Python and starts Blender itself; the Blender script builds the parts and writes the
  pack, the .glb and a scene folder.
- **Build:** no bar fits. Use only the kind's rules, and tell the person that no bar covers this
  piece.

## 3. Say how the piece is seen

Before building, write the piece's `view` in its pack: `nearest_m` and `farthest_m`, each with a
why from how the piece is used (`why` and `farthest_why`). A held instrument is seen from about
half a metre; a tree from its path. If you do not know, ask the person. Never pick a distance so
that `detail` or `coplanar` passes.

## 4. Build, then run the gate

Set `MODEL_KIT_BLENDER` to the Blender executable and run the piece's `build.py`. Then run the
gate on the scene folder it wrote:

```bash
model-kit gate <scene>
```

It prints `passed`, `partial` or `failed: n`. A partial run is not a pass: a check could not
measure something, and the lines above it say what the pack leaves out. Fix each failure, build
again and run the gate again. An allow line (`--allow FILE`) excuses one finding and needs a reason
a stranger would accept, never "it passes this way".

## 5. Look at the piece

The gate checks numbers. It cannot tell whether the piece looks right. After it passes, draw the
piece beside its bar at one scale:

```bash
model-kit sheet <name>.pack.json --piece <Piece> --style style.json --out sheet.png
```

Read `sheet.png`. Compare the piece with the bar: its size against the 1 m post, its outline, its
parts. Fix what looks wrong, then run the gate and the sheet again.

## Changing the kit itself

These rules are for work on this repository, not on a project that uses it.

- `uv run pytest` runs the tests. Tests marked `blender` run only when `MODEL_KIT_BLENDER` is set.
- `uv run model-kit profile <build.py> [--blend B] [--top 30] [-- args ...]` profiles a build
  inside Blender and writes `<build>.prof` in the current folder.
- A long solve writes one flushed line per solved key to stderr (clip, key, chain, seconds). Run
  the build in the background with stderr in a log and read the log's tail.
- `uv run --group qsm python -I tools/qsm_to_skeleton.py <model.mat> <out.skeleton.npz>` converts
  one TreeQSM scan into a measured-tree skeleton. It needs scipy.
- After editing a `bar.json`, run `uv run model-kit bars` to rewrite `bars/INDEX.md`; `--check`
  exits 1 when the file is stale.

### Adding a check

A new check meets the standing rule:

- it rests on a general principle, with a cited source;
- its plants (faults planted on purpose) are caught on at least 2 different pieces in pytest;
- it is quiet on the known-good set;
- no limit, range, window or station set is chosen so that a check passes. A range or limit taken
  from what a model's own clips already do is tuning.

No triangle count is a pass or fail line anywhere in kit code. A count may be printed or recorded,
never compared with a limit.
