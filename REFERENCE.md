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
- `gate <scene> [--checks a,b] [--allow FILE] [--timing FILE] [--pieces KEYS]` runs the checks on
  a saved scene folder. With no `--checks`, every registered check runs. Stdout is one status
  line: `passed`, `partial` or `failed: n`. Stderr comes in this order: coverage lines, one
  `time: <check> | real <s> s | coverage <s> s | plants <s> s` line per check, allowed lines,
  then the failure lines last. `--timing FILE` appends one JSON line per check and phase to FILE
  (fields: scene, kit, check, phase, seconds, loadavg_1m, pieces; `pieces` is the sorted keys of
  a scoped run, or null). The file's folder must exist, or the run cannot start (exit 2). Times
  are recorded, never judged: no time is a pass or a fail. A blind check exits 2 with nothing on
  stdout, but the run first finishes every check and prints its lines, then one
  `cannot start: BlindCheck: <check>: <item>; <check>: <item>` line naming every blind item. A
  blind run appends its timing rows too. Only a run on the whole scene is a pass. To iterate on
  one piece, add `--pieces <pack>:<piece>` (comma-separated keys; see [AGENTS.md](AGENTS.md)
  step 4): the part checks measure only those pieces, and the run always prints `partial` (exit
  1). An empty, repeated or unknown key cannot start (exit 2).
- `sheet <pack.pack.json> --piece P --style S [--bar auto|<slug>|none] [--bars DIR]
  [--out sheet.png] [--views page,front,...]` renders a piece beside its bar. It needs Blender
  (`MODEL_KIT_BLENDER`) and the sheet extra (Pillow).
- `profile <build.py> [--blend B] [--top 30] [-- args ...]` runs a build script under cProfile
  inside Blender, writes `<build>.prof` in the current folder and prints the top functions by
  cumulative time. The build calls `runner.done()` as its last step.
- `bake <scene> --stage FILE [--replace] [--compare [--lights FILE] [--compare-out DIR]]` bakes a
  stage's light onto a lightmap UV of every still piece and writes `<scene>/bake/`. It refuses a
  scene that already has one unless `--replace` is given. Blender writes into
  `<scene>/bake.partial/`, which takes the place of `bake/` only when the run succeeds, so a
  failed or stopped run leaves the old bake as it was. See "Light bake" below. It needs Blender
  (`MODEL_KIT_BLENDER`).
- `bars [--root bars] [--check]` writes `<root>/INDEX.md`. With `--check` it exits 1 when the
  file is stale and writes nothing.

## Checks

Each check rests on a general principle and a cited source, not on one model. The full URLs sit in
each check's `source` attribute. Each kind's `rules.md` (`src/model_kit/kinds/assembly/rules.md`,
`src/model_kit/kinds/branching/rules.md`) says what each of that kind's checks measures and what
its plant is: the planted fault the check must catch. The body, clip, layout and scene checks
declare no kind, so no `rules.md` has a heading for them; their principle and source sit in the
check's own module. A new check must meet the standing rule in `AGENTS.md`.

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
- Surface samples. `crossings`, `take-apart` and `bearing` sample a part's surface a quarter of
  its thickness apart. When one surface would need more than 10 million points, no point is made.
  The check writes an `Unmeasured` line that names the part and where it reads thin (a thin spot,
  or a shell sunk inside another shell of the same part), and the run is `partial`.

The seen floor. A piece with a `view` has findings smaller than 1 arcminute at `view.nearest_m`
dropped and counted in a coverage line. This applies to near-crease line ends, the visible half of
floaters and clip hygiene. The floor never adds a finding or tightens a limit. Faces on a
`view.hidden` side are skipped only by near-crease's line-end rule; a float on a hidden side is
still a float.

## Light bake

A bake writes the light of one layout into pictures, so a game reads the light from a texture
instead of computing it. Think of it as photographing the lamplight once and pasting the photo on
the objects. A part gets a lightmap UV (a second set of texture coordinates, one island per
smooth region of the part), and each light layer becomes one picture.

### The stage file

The stage file, of any name, passed as `--stage FILE` (bar 05 keeps `stage.json` beside its scene
folder; parsed by `core/manifest/stage.py`; the glTF frame). Positions, `size` and `radius_m` are in
the layout's units, as the layout's own positions are; only the densities are per real metre. A
style whose `units_per_metre` is not 1 is not tested yet.

- `stage`: the file version, 1. `layout`: the layout the bake lights.
- `mode`: `indirect` bakes only light that bounced at least once. `full` also bakes the direct
  light from the lamp itself. `full` needs soft lights (an area light, a big point radius) or far
  more pixels to pass `lightmap-density`: the hard shadow edge of a small light changes with every
  doubling of the density, so the bake never converges.
- `texels_per_m` sets the density of texels on a surface, for a piece seen from 1 m: a piece
  gets `texels_per_m / view.nearest_m` texels per metre in the shipped picture (twice that on the
  doubled grid the bake renders), so a piece seen from 0.5 m gets twice the density. The optional
  `placement_texels_per_m` maps a still placement id to its own density, for a placement that
  `lightmap-density` names while the others converge; a placement not listed uses `texels_per_m`. `samples` and
  `seed` set the Cycles noise. A picture's side is not set in the stage: the bake packs the islands and
  records the side as `size` in `bake.json`.
- `layers`: one picture per name; each holds at least one light. `live`: the placement ids that
  move, which the bake leaves out (at least one placement must stay still).
- `lights`: an id to `kind` (`sun`, `point`, `spot`, `area`, `hemisphere`), `position`, `layer`,
  and the fields the kind needs: `direction` (sun, spot, area), `size` [w, h] (area), `cone_deg`
  and `blend` (spot), `radius_m` (point, spot, optional), `sky_ground` (hemisphere).
- `cameras`: an id to `position`, `target`, `fov_deg`, `aspect`. A face no camera sees gets no
  lightmap UV and is listed `unseen`.

A layer bakes at white and strength 1. The colour and strength a light has live belong to the
page that mixes the layers (and to the `--lights` file of `--compare`).

### `bake.json`

Written next to `<layout>.glb` and one half-float `<layer>.exr` per layer. It holds:

- `bake` (the version, 1), `layout`, `mode`, `size` (the pictures' side in pixels), `uv_channel`
  (which UV set holds each part's lightmap), `unseen`, `live`, `samples` (the samples baked: two
  bakes of half each), `seed`, `blender` (the version that baked).
- `format`: `exr-half-zip`, half-float EXR with ZIP compression.
- `units`: `irradiance/pi`. A texel holds the irradiance its layer gives at strength 1, divided by
  pi; the mixer multiplies by `UNITS` (pi) to give three.js its `lightMap` irradiance.
- `layers`: per layer its `picture` file, `lights` (the light ids it holds), `seed` (the first of
  its two bakes' seeds), `baked_lights` (per light the `kind`, `color` and `strength` Blender baked
  it at, and `half` for a hemisphere), and, per placement, the `noise` and `error` the density
  check reads. `noise` is the RMS noise of the mean of the two bakes at the doubled grid; `error`
  is the RMS, over the 4 phases of the 2 x 2 block grid, of the picture shrunk at that phase and
  grown back against that mean (phase 0 is the shipped picture), so it does not depend on where
  the packer lands an island.

### Light units in the mixer

`js/bake-mixer.js` takes each layer's colour (linear RGB) and `strength`. The strength is
Blender's for that light kind, not three.js's intensity: a sun's irradiance in W/m^2, a point's or
spot's power in W, a hemisphere's world strength. A page that holds three.js intensities converts
each with `strengthFromThree(kind, intensity)`; the kind is in `baked_lights`. The factors
(`STRENGTH_PER_THREE`, Blender strength per unit of three.js intensity):

| Kind | Factor | How it is known |
|---|---|---|
| sun | 1 | Measured: a white plane, `test/blender/test_bake_units.py`. |
| point | 4 pi | Physical watt-to-candela factor; `--compare` matched a live slab to 1%. |
| spot | 4 pi | Derived: a Blender spot keeps the point's intensity inside its cone. |
| hemisphere | 1 / pi | Derived: a uniform sky of radiance S gives irradiance pi x S. |

An area light has no factor: three.js's `RectAreaLight` takes luminance, and nothing measures the
match. `compare.html` lights its live stand-ins by these same factors.

### The four checks

All four skip a scene with no `bake/` folder and read only what is on disk.

- `bake-files`: `bake.json` and the pictures agree. It fails on a listed picture that is missing,
  a picture no layer lists, a picture that is not `size` x `size`, and a seen placement with no
  `noise` entry.
- `lightmap-overlap`: no two parts claim one texel of the picture.
- `seam-jump`: the light does not step across a seam between two islands of one part by more
  than the bend there explains. Every seam is judged. The step is read just inside each island
  and compared with the bake's own noise and resolution error plus a Lambert allowance: the
  distance between the two sides' unit normals times the brighter side's light. A flat seam gets
  almost no allowance; a corner of 60 degrees or more can never step past its allowance, so it
  needs no read. Two normals are one surface (a flat seam, no allowance) when they differ by no
  more than the file's own precision, and the unwrap joins faces by the same rule
  (`seam_jump.same_normal`). That precision has two parts. The glTF exporter rounds each normal
  component to 4 decimals, so two exported normals of one normal differ by up to
  sqrt(3) x 1e-4. Blender computes a face's normal from float32 corners: each coordinate is off
  by up to eps/2 x its size (eps = 2^-23), a corner by up to sqrt(3)/2 x eps x m (m the face's
  largest coordinate in its mesh's frame), and that tilts the normal by up to k x that / L, for
  the face's k corners and L the smallest distance between two of them (`normal_noise`). A
  0.2 mm cell 0.8 m from its mesh's origin tilts by about 1e-3; a face near its origin, by almost
  nothing. The allowance is exact for light along
  the normal; light more than 45 degrees off it can step a bent seam past the line, and the
  person judges that finding. A seam with a bend under 1 on a part too thin to read twice is
  named in a coverage line, and the run is `partial`.
- `lightmap-density`: the light does not change when the density doubles. The loss is the part of
  `error` that is signal, not noise: `sqrt(max(0, error^2 - 3/4 noise^2))`. A placement fails when
  the loss is over its noise in any layer. The fix is a higher `texels_per_m` and a new bake,
  never a lower factor. How far to raise it is measured, not derived: on bar 05 the per-texel
  noise did not change when the density doubled, and a loss held at a hard contact edge (where a
  piece stands on the desk) fell only about as density^-1/2 (the desk's loss over noise went from
  1.53 at 256 to 1.15 at 512). Bake once at the next doubling and read the new ratio.

### `bake --compare`

`model-kit bake <scene> --stage FILE --compare` renders each stage camera twice, in Blender and in
headless Chromium through the shipped mixer, and prints one line per camera: its id, the mean
absolute difference in sRGB (0 to 1), the share of pixels the scene covers in both pictures and the
share it covers in only one. Neither side draws a background (Blender's film is transparent, the
browser clears to alpha 0), and the mean counts only pixels with alpha 255 in both pictures, so the
background and the anti-aliased silhouette edges drop out. A large "in only one" share means the
two sides do not frame the same scene. It also prints `light maps: N of M baked parts`: fewer than
all means the page did not find some baked part by name, and that part was drawn with no bake.
On a scene that already has a bake it compares that bake and
does not rewrite it; `--replace` bakes again first. The two pictures go to `compare/` beside the
scene folder, or to `--compare-out`. `--lights FILE` gives a layer's colour and strength (a JSON
object of layer to `{color, strength}`); the default is every layer white at 1.

The number says where to look. It is never a pass and no check reads it. The Blender side is
Cycles at 64 samples with no denoise, so its own noise sets a floor under the number. The live light in the
browser is a stand-in, and its match to Blender differs by kind: the sun is measured; a point
light uses the physical watt-to-candela factor 1/(4 pi); a spot reuses the point factor and a
hemisphere is derived, both unmeasured; an area light has no live stand-in, and `compare` prints
a note that names it.

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
- `rests_on`: two or more parts of the same piece this part rests across, such as a pallet's top
  board on its three stringers. It replaces `on` for that part (never both). It must touch each
  (`floaters`), balance over all of them together, and not sink into any (`bearing`).

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

**The card copy.** A grown tree has a game-weight copy: a piece with `cards_of` naming the grown
piece (`blender.grow.add_cards`, `core.grow.cards`). It has no skeleton of its own and shares its
source's. It ships the source's wood levels plus one `Cards_<shade>` part of leaf cards that carries
a painted leaf image (packed into the .glb as an alpha-cut material). The leaves are gathered into
round balls (the style's `dials.cards.ball_m`), each ball is covered with cards that sit on real
leaf centres and face the sky or the ground, and a card buried in a neighbouring ball and dark is
dropped. The dials are `ball_m`, `normal_mix` and `sky`, each with one `why`. A card piece is
`outlined: false` with an `outlined_why`. The bar `bars/02-redbud/` ships `Redbud_Cards`, and
`tools/tree_cards_page/index.html` draws it beside `Redbud` in three.js.

A viewer must not flip a card's back-face normal: both faces use the clump normal. glTF
`doubleSided` makes a stock viewer flip it before lighting, so the shared normal holds only in a
viewer that turns the flip off. In three.js, set `material.onBeforeCompile` on a card material to:

```js
o.material.onBeforeCompile = (s) => {
  s.fragmentShader = s.fragmentShader.replace('#include <normal_fragment_begin>',
    'float faceDirection = 1.0; vec3 normal = normalize( vNormal ); vec3 nonPerturbedNormal = normal;')
}
```

`tools/tree_cards_page/index.html` does this, and so does Razel's viewer.

A viewer should sample the leaf texture with anisotropic filtering (three.js:
`texture.anisotropy = renderer.capabilities.getMaxAnisotropy()`); glTF has no field for it. A
slanted card then picks its mip level from the short axis of its footprint, not the long one,
which would land on a level where no texel passes the alpha cut. The atlas bleeds leaf colour
under clear texels, so far mips keep the leaf colour. The page `tools/tree_cards_page/index.html`
sets the anisotropy too.

The card rules are ported from Razel's "Just a simple tree",
github.com/MikeFernandez-Pro/just-a-simple-tree, MIT License, Copyright (c) 2026 Mike Fernandez.
The MIT permission notice is at the top of `core/grow/cards.py`. Razel's constants (card tilt,
roundness, burial, hidden share, rays, ambient-occlusion mix, fade-in, clump light) are kept as
named constants there.
