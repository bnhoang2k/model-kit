# model-kit reference

Everything the kit does, for a person or an agent that needs the detail. Start with the
[README](README.md); an agent follows [AGENTS.md](AGENTS.md).

The `core` package is plain Python and numpy and never imports Blender. The `blender` package
holds the code that runs inside Blender; `model-kit` starts Blender itself, from
`MODEL_KIT_BLENDER`.

## Commands

`model-kit <command>`, or `uv run model-kit <command>` inside a clone of this repo. Exit 0 is done,
1 is a finding, 2 is a run that could not start or was not ok.

- `init [--dir D]` installs this release for a project and adds one line to its `CLAUDE.md` (made
  if missing) and to its `AGENTS.md` when it has one. Inside a Python project (a `pyproject.toml`)
  it adds the kit as a uv dependency; elsewhere it installs it as a uv tool. A second run changes
  nothing.
- `guide` prints the steps a coding agent follows ([AGENTS.md](AGENTS.md)).
- `gate <scene> [--checks a,b] [--allow FILE]` runs the checks on a saved scene folder. With no
  `--checks`, every registered check runs. Prints `passed`, `partial` or `failed: n`.
- `sheet <pack.pack.json> --piece P --style S [--bar auto|<slug>|none] [--bars DIR]
  [--out sheet.png] [--views page,front,...]` renders a piece beside its bar. It needs Blender
  (`MODEL_KIT_BLENDER`) and the sheet extra (Pillow).
- `profile <build.py> [--blend B] [--top 30] [-- args ...]` runs a build script under cProfile
  inside Blender, writes `<build>.prof` in the current folder and prints the top functions by
  cumulative time. The build calls `runner.done()` as its last step.
- `bars [--root bars] [--check]` writes `<root>/INDEX.md`. With `--check` it exits 1 when the
  file is stale and writes nothing.

## Checks

Each check rests on a general principle and a cited source, not on one model. The full URLs sit in
each check's `source` attribute. `src/model_kit/kinds/assembly/rules.md` says what each one
measures and what its plant is: the planted fault each check must catch. A new check must meet
the standing rule in `AGENTS.md`.

All checks, by what they judge:

| Group | Check | It fails when |
| --- | --- | --- |
| Parts and assemblies | `crossings` | two solid parts share space, or one is sunk more than halfway into another |
| | `take-apart` | a group cannot come out of the assembly in any direction without cutting another |
| | `floaters` | a part does not reach something that stands |
| | `bearing` | a part is not held the way ordinary assembly holds it |
| | `receiver`, `connectors` | a peg seats where no hole's wall surrounds it, or a guest does not rest on its host |
| | `keep-off` | a mount lands on a spot its piece says to keep clear |
| | `clearance` | two pieces side by side are neither joined nor a finger's width apart |
| | `seat-fit` | a seat is deeper than the sitter's thigh |
| | `mirror-exact` | a declared mirror part is not the exact reflection of its twin |
| | `done` | a piece does not say what done looks like or how big it is |
| Surfaces and lines | `topology` | a part is not one closed, outward surface, so its outline would break |
| | `markings`, `markings-flush`, `inset-seat` | a print, paint or inlay is built as a solid part, stands off its host or weaves through it |
| | `near-crease` | a drawn line stops mid-surface or flickers as an angle moves |
| | `detail` | a curve is blockier, or finer, than the eye sees at the nearest viewing distance |
| | `coplanar` | two faces lie closer than the depth buffer resolves, so they would flicker |
| Bodies and clips | `must-touch` | a hand, foot or held prop does not touch what it holds |
| | `reach`, `solver-error`, `trace` | a clip target is out of reach, missed, or points at a stale spot |
| | `overlap`, `stress-poses` | a posed body sinks into itself or a prop, in a clip or at a joint's range end |
| | `bend-quality` | a limb loses girth or collapses as it bends |
| | `clip-hygiene` | a clip slides, pops, floats off the floor or drops a prop |
| | `frames`, `sides`, `mirror-pair` | a point is read in the wrong frame, a hand is on the wrong side, or a mirrored clip is not mirrored |
| | `no-fallback` | a build falls back to a guess when an anchor is missing |
| Grown plants | `limb-seat`, `floor-organic` | a limb or leaf does not sit on its parent, or dips into the soil |
| | `crossing-organic`, `near-miss-organic` | two limbs grow through or graze each other |
| | `taper`, `leaf-angle` | a limb thickens toward its tip, or leaves lie at angles real crowns do not show |
| | `reference-range` | a grown tree's size and form fall outside its species' cited ranges |
| | `skeleton-match` | the shipped mesh is not the mesh the skeleton makes |
| Export | `export-hygiene` | the .glb's node table disagrees with the pack that made it |
| | `coverage` | an exported part or clip is measured by no check in the run |

The rules with the longest reasoning:

- `stress-poses`: a body never passes through itself in its range of motion. Skin meeting skin at
  a range end is touch, not passing through. Source: the range-of-motion table each rig cites in
  `rig.joints` (AAOS via Hendriks et al. 2020; NASA Human Integration Design Handbook Rev 1,
  App. B Table 2; 38 CFR 4.71a).
- `mirror-exact`: a part declared the mirror of another is its reflection, with the same
  triangles and corners. Source: the Blender Manual's Mirror modifier. A left shin that splits its
  quads along other diagonals than the right one fails.
- `seat-fit`: a seat is shallower than the sitter's buttock-to-knee-crease length, so the knees
  pass the front edge. Source: CityU online anthropometry (5th percentile buttock-popliteal
  length sets the seat depth). The margin past the edge is this kit's own call.
- `clearance`: two separate pieces side by side keep a gap a finger fits through, or they are
  joined. Source: ISO 13854:2017, 25 mm for a finger. The kit borrows only the body-part size.
- `markings`: a print, paint film, inlay or screen image is a colour change on a surface, so it
  is a decal and not a solid part. Source: film thickness, 4 microns of pad-print ink (Inkcups),
  1 to 20 microns of printing ink (Podhajny 2002), 25 to 70 microns of paint (two data sheets).
- `markings-flush`: a solid part that stands off its host by less than the eye resolves, at the
  nearest distance the piece is seen from, shows only as a colour change, so it belongs on the
  host as a decal. Source: the eye resolves about 1 arcminute (EyeWiki, secondary), and
  the kit judges by eye.
- `coplanar`: two faces that face the same way, overlap and lie closer than one depth step of
  the style's reference camera, at the farthest distance the piece is seen from, flicker. Source:
  Khronos, Depth Buffer Precision (OpenGL Wiki, read from its archived copy). A face pair that a
  closed part of the piece hides is not judged.
- `near-crease`: a drawn line follows a real edge. It runs corner to corner where two faces
  meet, never stops mid-surface, and never comes and goes as rounding or bending moves an angle.
  Source: drafting practice (Ford, Engineering Graphics and Design); the
  band is the gradient of the dihedral angle (Bridson, Marino and Fedkiw 2003). A piece that
  declares `outlined: false` is skipped, with a printed line giving the reason.
- `bend-quality`: a limb keeps its girth and its section as it bends. Source: soft tissue is
  nearly incompressible (arXiv 2312.04108); the girth limit is the largest change measured on
  real limbs (Ringenbach, Yu and Zhang 2024; Klepser and Morlock 2020). The area floor is a stated
  design call. Girth is judged at clip poses only.
- `inset-seat`: a part printed on a surface or set into it meets that surface along one closed
  line, with its rim wholly on or inside it, or wholly on or outside it. The rim is the part's
  outline seen from outside the host along the host's area-weighted normal, and each loop is
  judged on its own at 1 arcminute. Source: the 4 micron ink film; the eye's 1 arcminute. No source was read for the thickness of an inlay.
- `bearing`: a part is held the way ordinary assembly holds it. Its weight lies over its
  support, its fastener sits in its host for about its own diameter, its seat bears on the host
  over the whole seat, and a retainer touches what it retains. Source: Nord-Lock (2015) and NASA
  RP-1228 for thread engagement; statics for the rest, with no outside source cited.
- `detail`: a curve has the facets the eye can see and no more. It is blocky when the gap between
  a facet and its curve is over what the eye resolves at the nearest viewing distance, and finer
  than it needs to be when half the segments would look the same. The gap for two unequal faces
  is the merged-chord gap `a·b·sinθ/√(a²+b²+2ab·cosθ)`. Source: the eye's 1 arcminute. No triangle count is compared with a limit.
- `clip-hygiene`, stand rule: gravity loads a standing body's supports, so in a `stand` clip a
  support stays on the floor, within the overlap check's skin budget, except in its `lifts`
  windows (a step or a hop, each with a why). Source: statics; the windows are a design call of
  this kit.

- `crossings`: two solid parts cannot share space, and a part sunk more than halfway into its
  neighbour is embedded. Depth is the smallest extent of the shared space over 13 fixed
  directions, against 0.5 times the thinner part's shape diameter (Shapira et al. 2008, a 17-ray
  cone).
- `take-apart`: each group comes away from the others without cutting them. A group is tried
  along its pull from the piece's centre, then along 26 fixed ways, and fails only when every way
  cuts. A pair cuts along a way when its depth, measured as `crossings` measures it, is more than
  its depth at rest plus the `crossings` limit. The finding names the least deep way. A cable
  neither moves nor blocks.

The seen floor. A piece with a `view` has findings smaller than 1 arcminute at `view.nearest_m`
dropped and counted in a coverage line. This applies to near-crease line ends, the visible half of
floaters and clip hygiene. The floor never adds a finding or tightens a limit. Faces on a
`view.hidden` side are skipped only by near-crease's line-end rule; a float on a hidden side is
still a float.

## Manifest fields

These fields are optional where they say so. An older pack without them still loads; where a field has parts of its
own (`view` needs `nearest_m` and `why`; `farthest_m` needs `farthest_why`) or a case needs it (a rig with several roots needs
`root`), the field says so below. A check that needs a
field the pack leaves out makes the run `partial` (`Unmeasured`), or prints a not-applicable line
when the rule does not apply by design.

In `pack.json`, on a piece:

- `view`: how close an eye comes to the piece. `nearest_m` is the nearest distance in real
  metres, and `why` says how the piece is used (a held instrument, a tree seen from its path).
  `hidden` lists sides never seen (`front`, `back`, `left`, `right`, `top`, `bottom`, in the
  piece frame: front is +z, top is +y), and `hidden_why` says why. `hidden` defaults to none, and
  a `hidden_why` needs a side to explain.
  `farthest_m` is the farthest distance in real metres, with a `farthest_why` saying how the piece
  is used; it is never nearer than `nearest_m`. It is optional, but `coplanar` reads it: the depth
  buffer is coarsest at the far end, so a piece without it reads `Unmeasured` there.
- `outlined`: whether the piece's shipping look draws outlines. It defaults to `true`, so the
  kit stays strict. `false` needs an `outlined_why` saying how the look draws the piece without
  lines, and the near-crease check then skips the piece and prints a not-applicable line. An
  `outlined_why` with `outlined` true or missing is refused. Only near-crease reads it.

On a listed row:

- `finish`: `solid` (the default), `print`, `paint`, `inlay` or `display`. Every part of a row
  that is not `solid` has shape `decal`, and the reverse (`markings`).
- `held`: `rests` (the default) or `bonded`. `bonded` is glued, welded or moulded in, so the
  gravity rule of `bearing` does not apply to it.

On a part:

- `fastener`: true for a bolt, screw or pin. Its `on` names the part it fastens into (`bearing`).
- `retains`: parts of the same piece this part holds down or in, such as a string tree, a strap
  or a jaw. Each must touch it (`bearing`).

In a piece's `rig`:

- `supports`: the contact groups that bear the body's weight when it stands (two feet, four
  paws). Clip hygiene reads them instead of guessing the feet.
- `joints`: each bending joint's range of motion, by bone. `range_deg` gives a low and a high in
  degrees per axis, as euler angles in the bone's rest frame, and `why` cites the table or the
  design. A chain limit must lie inside its joint's range. `stress-poses` takes the joints to
  the ends of these ranges.
- `root`: the bone a seat and the side rules measure from. It defaults to the one parentless
  bone, and a rig with several parentless bones must name it.
- `sit.past_edge`: the bones whose heads must lie past a seat's front edge, the knees of a
  person (`seat-fit`).

In `clips.json`, on a clip:

- `posture`: `stand`, `sit`, `lie` or `free`. A clip with a `seat` is `sit`, and a `sit` clip
  names a seat. A clip with no `posture` is one written before it existed.
- `lifts`: supports that may leave the floor in a `stand` clip, each with `support` (a contact
  group of `rig.supports`), `when` (frame windows inside the clip) and `why`. Outside them a
  loaded support rests on the floor (clip hygiene).

In a layout, on a placement:

- `rests_on`: the id of another placement of the same layout that this one sits on with no mount
  (gear on a deck, a box on a table). A placement has `on` or `rests_on`, not both. `clearance`
  measures that it touches its host.

In the style file:

- `dials.edges.crease_deg`, with a `crease_why`: the angle over which two faces shade as a hard
  edge. The runtime outline draws an edge over it. `near-crease` and `detail` read it.
- `dials.edges.borders_only: true`, with a `borders_why`: the look draws outlines and no fold
  lines. Use it instead of `crease_deg`; a style that declares both is refused. `near-crease`
  prints a not-applicable line with the reason. A style that declares neither reads `Unmeasured`.
- `page_camera` depth fields, read by `coplanar` for the depth step: `depth_projection` (only
  `"perspective"` is supported), `near_m` and `far_m` (the camera's near and far planes, in real
  metres), `depth_bits` (the integer depth buffer's width) and `depth_why` (where those numbers
  come from). All five are declared together or none; a style with none reads `Unmeasured`.
- `dials.leaves` (`normal_lean`, `centre_drop`) needs a `why`. A style checks its dials when it is
  built, so one made in code is as legal as one read from a file.

## The tree grower

`model_kit.core.grow.grow_buds` grows one tree from a species rules file and a seed. It is the
only grower. Each year the tree adds nodes and buds, a bud wakes or sleeps by its light and its
leader, and the result is a skeleton: limbs, radii and leaves. `load_rules(path)` reads a species
file and merges the kit file `kinds/branching/botany.rules.json`; every number in either file
names the source rows behind it. Growing never runs in Blender: the skeleton is saved as
`.skeleton.npz`, and `model_kit.blender.grow` builds the mesh from it. Leaves sit on the twig
level, or on the highest level a young tree grew.

The checks measure the result against four lidar-scanned trees (Demol, Gielen and Verbeeck 2021,
CC-BY 4.0; credits in `test/fixtures/qsm/CREDITS.md`). A tree is judged by the facts the rules
cite, never by a number fitted until the trees pass.
