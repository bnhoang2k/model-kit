# Bar 02: the redbud

The reference for branching (grown) pieces: an eastern redbud grown bud by bud, year by year,
from botany facts. A new tree is compared against it at the page camera, so you can see at a
glance whether the crown, trunk, limbs and leaves read right at the same scale.

## What is in this folder

- `redbud.rules.json`: the species file. 23 rows (each a rule, a value and its source), 10
  sources, one cited range and 17 params, each tagged. The grower merges it with the kit file
  (`src/model_kit/kinds/branching/botany.rules.json`), which holds the facts true of every tree
  and the constants of the growth model. A species file may not set a kit key.
- `build.py`: grows the two trees with `model_kit.core.grow.buds.grow_buds`, runs Blender
  through `model_kit/blender/grow_build.py`, and writes `redbud.pack.json`, `redbud.glb` and
  `scene/`. Run it with `MODEL_KIT_BLENDER` set: `uv run python bars/02-redbud/build.py`.
- `style.json`: 0.5 units per metre; the page camera is ortho at turn 14, tilt 52;
  `min_thickness` is 0.018 m; `dials.leaves` is the leaf-normal lean (a render dial).
  It also declares the reference camera for depth: a three.js perspective camera, near 0.1 m,
  far 2000 m, 24-bit depth (the 24 bits to confirm; `depth_why` has the sources).
- `redbud.pack.json` and `redbud.glb` (Draco, 2.5 MB): the pack with its two pieces.
- `scene/`: the saved scene the gate reads: each piece's triangles and skeleton, the node table
  of the .glb, and `rules/redbud.rules.json` (the copy `reference-range` reads). It does not
  ship in the wheel.
- `sheet.png`: `Redbud` alone at the page camera and four more views.

## The pieces

| Piece | Grown as | Grow time | Triangles | Nearest view (intended) |
| --- | --- | --- | --- | --- |
| `Redbud` (the bar) | 5.0 m, seed 6, full | about 35 s | 349,392 | 5 m |
| `Redbud_Lite` | 5.0 m, seed 7, lite | about 35 s | 128,322 | 20 m |

Grow time is `perf_counter` around one `grow_buds` call, as `build.py` prints it, one grow at a
time on one machine. The budget is 60 s for `Redbud`.

Triangles are report-only: no target, no pass or fail. Before the tubes took their resolution
from the view (the fixed 12, 9, 7, 5, 4 sides per level; full / lite): 257,746 / 104,648. After:
349,548 / 128,274. After every bend that shows was rounded or kept a sharp crease:
349,392 / 128,322. An earlier version of the grower gave pieces of 51,726 (full) and 39,282 (lite). By part, now (before the
view in brackets), `Redbud`: trunk and roots 7,262 (1,814), leaders 11,470 (9,370), branches
41,332 (34,594), twigs 62,326 (38,294), shoots 107,674 (54,346), leaves 119,328 (119,328).
`Redbud_Lite`: 1,692 (1,558), 5,200 (5,256), 18,416 (18,354), 33,296 (22,480), 43,600 (30,882)
and 26,118 (26,118).

Nearest view is the distance the detail check reads (`view.nearest_m`):
- `Redbud`, 5 m: the bar scene is the page camera framing the whole 5 m tree, and a person in a
  scene stands back about one tree height to see a small tree whole.
- `Redbud_Lite`, 20 m: lite is the copy for scenes that repeat many trees, seen as a backdrop.
  It takes over from the full tree at about four tree heights, where single leaves no longer
  read.

Farthest view (`view.farthest_m`): `Redbud`, 20 m, where `Redbud_Lite` takes over. `Redbud_Lite`,
50 m: an estimate from how it is used, to be confirmed (the far end of the backdrop; the text
above says only where the backdrop begins).

## How it was grown

Each year every bud reads its light from a shadow grid, resource flows from the base by the
extended Borchert-Honda model, and shoots bend toward light and their gravity set-point angle.
The redbud has no terminal bud, so each year its top side bud carries the shoot on (the relay),
and its leaves are two-ranked, 180 degrees apart. Together they make the zigzag twigs; no
zigzag angle is written. Shaded limbs are shed. After growth the pipe model gives the radii, and
a limb stops short of a thicker limb it would grow through. A limb in the skeleton is a
continuation chain: at each fork it runs on through the thickest child.

Seeds make a tree repeatable: the bar's test regrows each saved skeleton from the rules and the
kit file and compares them array for array.

Each limb is a tube and each leaf a folded blade on a short stalk. The Blender step makes one
mesh per level (trunk and roots, leaders, branches, twigs, shoots) and one leaf mesh, and leans
each leaf normal outward along the line from the crown centre (the `dials.leaves` render dial in
`style.json`), so the crown shades as one mass.

**The dark leaf patch** in some views is not a defect in the tree. The lean is baked into the
leaf normals of `redbud.glb`, and the sheet's light turns with the camera and comes from the
upper left, so leaves that lean away from it darken on one side.

## The inputs and their tags

`measured` cites a source or a photo. `fitted` was fitted to a cited range on seeds 1 to 4.
`budget` is a page cost, or a choice with no species fact behind it.

| Param | Value | Tag | Source |
| --- | --- | --- | --- |
| `leaves.divergence_deg` | 180 | measured | [OE] orthodistichous |
| `leaves.per_node` | 1 | measured | [W][UF] "alternate" |
| `buds.terminal` | absent | measured | [NRC] "Terminal bud absent" |
| `internode.length_m` | 0.04 | budget | an assumed value; [WK] measured 2.0 cm on one cultivar, which grows only 2.5 m in 10 years with the kit's constants |
| `branch.angle_deg` | 44 to 54 | measured | [DM], beech and ash: a stand-in, no redbud value found |
| `gsa.by_order_deg` | 180, 135, 125 | measured | [DM], beech and ash: a stand-in (assumed values) |
| `age.height_points` | 2.3 m at 4 years, 5.0 m at 10 | measured | [WK] (a cultivar); [W] |
| `twig.tip_radius_m` | 0.001 | budget | no shoot-tip radius found; it gives a trunk about 15 cm across, as photo [P3] shows |
| `levels.names` | 5 part names | budget | page budget |
| `apical.control` | 0.49 | fitted | fitted to `spread_of_height`, seeds 1 to 4 |
| `leaves.angle_beta` | a 2.1984, b 2.2862 | measured | [CH] Cercis siliquastrum, a same-genus stand-in |
| `leaves.outline` | a 10-point heart | measured | [W] "heart shaped" |
| `leaves.length_m` | 0.07 to 0.12 | measured | [W] 7-12 cm |
| `leaves.petiole_of_blade` | 0.35 | measured | photo [P5] |
| `roots.count` | 5 | measured | photo [P3] |
| `lite.leaf_thin` | 2 | budget | page budget |
| `lite.leaf_outline` | a 6-point heart | budget | page budget |

The tube sides are not inputs. The rules file has no `tubes.sides` or `lite.tube_sides`: the mesh
chooses a tube's sides and axis rings when it is cut, from the piece's `view.nearest_m` (the
fewest sides whose gap to the curve the eye cannot see at that distance, and enough to be round
at the style's crease angle), so a skeleton serves a full piece and its lite piece. A bend that
would show is rounded by an arc wider than the tube; the arc may take the whole of a band its
neighbour does not need (the two 50 to 63 degree bends on a `Redbud` leader limb, which half a
band could not round). A bend whose bands leave no room for such an arc stays the skeleton's
sharp crease, as a limb can turn sharply where it grew so: `Redbud_Lite`'s trunk keeps one, a
41 degree bend beside a seat. The detail check judges its edges like any other edge, and the
whole gate passes, so the bar has no pinned output.

The species tags have no "guess" tag. A stand-in from other species is tagged `measured`, and
its row says it is a stand-in.

## The range and the fit

`spread_of_height` is 0.67 to 1.67. The low end is the one young tree measured: about 5 ft wide
and 7.5 ft tall at about 4 years [WK] (a semi-compact cultivar). The high end is the widest
mature ratio, 10 m over 6 m [W]. A 10-year tree lies between the two ages; no wild 10-year tree
was found. The mature band alone (0.89 to 1.67) was the first choice, but no `apical.control`
from 0.44 to 0.56 put four seeds inside it: the bud grower's spread varies with the seed by more
than that band allows. There is no `crown_base_of_height` range: the photo read (0.30 to
0.45) is unverified, so the gate prints a not-judged line for it.

`apical.control` was fitted on seeds 1 to 4 (spread 0.75, 0.86, 0.94 and 0.90 at 0.49). Every
held-out seed 5 to 8 passes the gate's eight branching checks, full and lite:

| Seed | Spread / height, full | Lite | Crown base / height, full (report-only) |
| --- | --- | --- | --- |
| 5 | 0.759 | 0.761 | 0.276 |
| 6 | 0.782 | 0.788 | 0.308 |
| 7 | 0.721 | 0.719 | 0.072 |
| 8 | 0.732 | 0.737 | 0.177 |

Seed 6 (full) and seed 7 (lite) are the bar's pieces, so the two are different trees.

## Parked items, rechecked

- Buried root ends below y = 0: roots are buried limbs whose tips must be in the soil, and
  `floor-organic` passes on both pieces.
- The lite listed text and `on:` links: each lite part lists its own rows plus the `lite` rows,
  and the lite leaf part lists `leaves.blade` (the bud grower reads the blade length in lite mode
  too). Every tube part sits `on:` the level below it, and the leaves on `Twigs`.

## What the page camera shows

Looking down 52 degrees, turned 14 degrees: a loose, irregular dome of leaf clusters, a little
taller than wide, on a short trunk that forks low and flares into the soil. Gaps between the
clusters show dark limbs. The twig and shoot levels, and the leaves, are marked to skip the
outline.

## Instancing

The bar is one mesh per level. A page that draws many trees should instance those meshes (one
draw per level and shade, many transforms). That is the page's job, not the pack's.

## Compare a new branching piece against it

```
uv run model-kit sheet new.pack.json --piece Tree --style style.json --bar auto --out sheet.png
```

`--bar auto` picks this bar for any branching piece. The sheet puts the new piece in row one and
`Redbud` in row two, at one scale, with a 1 m post beside each. Then run
`uv run model-kit gate <scene>`: with no `--checks` it runs every registered check, the eight
branching ones included (limb seats, floor, crossings, near misses, the skeleton match,
reference ranges, taper and leaf angle).

On the bar's own scene the gate prints `passed`. Its coverage notes: leaf cards have no inside,
so the crossing and near-miss checks leave them out; the crown-base range is not judged (there
is none); `taper` leaves out the few limbs with fewer than 4 points.

## Sources

- [W] Wikipedia, Cercis canadensis: https://en.wikipedia.org/wiki/Cercis_canadensis
- [NC] NC State Extension: https://plants.ces.ncsu.edu/plants/cercis-canadensis/
- [UF] UF/IFAS ST145: https://ask.ifas.ufl.edu/publication/ST145
- [NRC] Natural Resources Canada, Trees of Canada: https://tidcf.nrcan.gc.ca/en/trees/factsheet/415
- [OE] Owens & Ewers 1991, Can. J. Bot. 69: 1956-1963: https://doi.org/10.1139/b91-245
- [CH] Chianucci et al. 2018, leaf inclination data: https://doi.org/10.1007/s13595-018-0730-x
- [WK] Werner & Kenealy, USPP35080: https://patents.google.com/patent/USPP35080P2/en
- [DM] Demol et al., lidar QSMs of beech and ash (CC-BY 4.0): https://doi.org/10.5281/zenodo.4557401
- Photos [P3] and [P5], Wikimedia Commons (links in `redbud.rules.json`).
- The kit file's sources: Eloy 2011 (Leonardo's rule), Palubicki et al. 2009 (the growth model)
  and Barthelemy & Caraglio 2007 (plagiotropic shoots).

## Taste read

On the sheet the pieces read as good enough for scenery. Two things are noted, neither broken: the redbud crown sits a little off the trunk in two clumps, and the sweetgum forks low into several stems. Outline: none (`dials.edges.outline` stays null), because an outline on every leaf turns a crown solid ink.

## Look fixes

- Limbs bend smoothly: relay shoots left at the full branch angle, so limbs kinked every year and looped. A relay now leaves at `RELAY_SHARE_OF_ANGLE` (0.25) of it, for every species (`core/grow/shape.py`).
- No antlers at the top: the beech and ash set-points (135, 125) stood the top shoots up as spikes. The redbud's set-points are now 115 and 100 (`gsa.by-order`), its spreading habit; spread / height is now about 1.2 to 1.3.
- Seed 5 dips a low limb 1.25 cm into the soil, so the lite piece uses seed 7. Triangles are roughly three times the earlier count (report-only).
