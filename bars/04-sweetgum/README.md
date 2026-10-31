# Bar 04: the sweetgum

A second branching bar: a young sweetgum (*Liquidambar styraciflua*) grown bud by bud from the
same kit-wide rules as bar 02, with its own species file. It shows the grower on a spiral,
terminal-bud species. It is scenery, not a botanical portrait. Bar 02 stays the class default
(`default: false` here).

## What is in this folder

- `sweetgum.rules.json`: the species file, merged with the kit file
  (`model_kit/kinds/branching/botany.rules.json`) when it is loaded. Every param has a tag and
  cites a row; every row cites a source or says it has none.
- `build.py`: grows the two trees with the bud grower, runs Blender through
  `model_kit/blender/grow_build.py`, and writes `sweetgum.pack.json`, `sweetgum.glb` and
  `scene/`. It also copies the species file into `scene/rules/`, where `reference-range` reads
  it. Run it with `MODEL_KIT_BLENDER` set: `uv run python bars/04-sweetgum/build.py`.
- `style.json`: bar 02's style (0.5 units per metre, the page camera at ortho, turn 14, tilt 52,
  the `dials.leaves` lean), plus `dials.edges.crease_deg` 35 with its `crease_why`, for the edge checks.
  It also declares the reference camera for depth: a three.js perspective camera, near 0.1 m,
  far 2000 m, 24-bit depth (the 24 bits to confirm; `depth_why` has the sources).
- `sweetgum.pack.json` and `sweetgum.glb` (Draco): the pack with its two pieces.
- `scene/`: the saved scene the gate reads. It does not ship in the wheel.
- `sheet.png`: the piece alone at the page camera and four more views.

## The pieces

Triangle counts and grow times are report-only: no target, no pass or fail.

| Piece | Grown as | Years | Triangles | Grow time | Leaves | Spread / height |
| --- | --- | --- | --- | --- | --- | --- |
| `Sweetgum` (the bar) | 3.87 m, seed 7, full | 6 | 48,794 | 8.4 s | 1,296 | 0.778 |
| `Sweetgum_Lite` | 3.17 m, seed 6, lite | 5 | 14,580 | 3.6 s | 531 | 0.669 |

Before the tubes took their resolution from the view (the fixed 12, 9, 7, 5, 4 sides per level;
full / lite): 31,534 / 10,584 triangles. After: 48,490 / 14,580. After every kink that shows was
rounded: 48,794 / 14,580. By part, now (before the view in brackets), `Sweetgum`:
trunk and roots 5,844 (1,120), leaders 5,538 (2,832), branches 11,064 (6,770), twigs 9,142
(4,582), shoots 1,654 (678), leaves 15,552 (15,552). `Sweetgum_Lite`: 916 (632), 1,068 (626),
3,372 (2,890), 4,120 (2,210), 1,918 (1,040) and 3,186 (3,186).

Grow time is the wall time of `grow_buds` in the build (system Python, one run). `sweetgum.glb`
is 409 kB with Draco. `Sweetgum` has 5 trunk and root limbs, 15 leaders, 52 branches, 49 twigs
and 9 shoots; its crown base is at 0.27 of the height (reported; no cited range).

Viewing distance, for the detail check:

- `Sweetgum`: intended `nearest_m` 4.0. In the bar scene a viewer stands about one tree height
  away, the nearest distance at which the whole 3.9 m tree fits in view.
- `Sweetgum_Lite`: intended `nearest_m` 15.0. It takes over where many copies stand in a
  backdrop row, about four tree heights away, where a coarser leaf no longer shows.

Farthest view (`view.farthest_m`): `Sweetgum`, 15 m, where `Sweetgum_Lite` takes over.
`Sweetgum_Lite`, 40 m: an estimate from how it is used, to be confirmed (the far end of the
backdrop row; the text above says only where the row begins).

## Inputs and their tags

| Param | Value | Tag | Source |
| --- | --- | --- | --- |
| `age.height_points` | 0.46, 1.77, 3.87 m at 1, 3, 6 years | measured | Krinard & Johnson 1985, DEF site (planted, 10 x 10 ft) |
| `buds.terminal` | present | measured | Virginia Tech dendrology |
| `leaves.divergence_deg` | 137.5 | measured | Orwa et al. 2009 (spiral); Okabe 2016 (golden angle) |
| `leaves.per_node` | 1 | measured | spiral arrangement |
| `leaves.angle_beta` | 6.1932, 1.8352 | measured | Chianucci et al. 2018 data, method of moments |
| `leaves.length_m` | 0.10 to 0.15 | measured | Virginia Tech (width; length taken equal) |
| `leaves.outline` | five-pointed star | measured | Virginia Tech (star-like lobes); points drawn for this bar |
| `apical.control` | 0.50 | fitted | spread / height, seeds 1 to 4 |
| `branch.angle_deg` | 40 to 50 | measured | an assumed value: beech and ash QSM medians, other species |
| `gsa.by_order_deg` | 180, 135, 125 | measured | an assumed value: beech and ash, other species |
| `internode.length_m` | 0.05 | model | an assumed value; no source opened |
| `leaves.petiole_of_blade` | 0.6 | model | an assumed value; no source opened |
| `twig.tip_radius_m` | 0.004 | budget | the thinnest tube the page draws |
| `roots.count` | 4 | budget | root flares the page draws |
| `levels.names`, `lite.*` | as bar 02 | budget | page cost |

A species file allows the tags `measured`, `fitted`, `budget` and `model`, and a `measured` param
must cite a source on every row it names. `budget` is a cost limit (what the page draws), not a
fact. Guesses with no source (internode, petiole) carry `model`. The branch angle and set-point angles stay `measured`: they cite the
beech and ash QSM medians [DM], other species, and their rows say so.

The tube sides are not inputs. The rules file has no `tubes.sides` or `lite.tube_sides`: the mesh
chooses a tube's sides and axis rings when it is cut, from the piece's `view.nearest_m`. A kink
that would show is rounded by an arc that may take the whole of a band its neighbour does not
need (the 50 degree bend on a `Sweetgum` branch limb, which half a band could not round), so the
detail check is quiet and the bar has no pinned output. A bend whose bands leave no room for an
arc wider than the tube would stay the skeleton's sharp crease, as a limb can turn sharply, and
detail would judge it like any other edge; neither sweetgum piece has one.

## Ranges and the fit

One range: `spread_of_height` 0.47 to 0.83, from NC State Extension (40 to 50 ft over 60 to 100
ft) and UF/IFAS ST358 (35 to 50 ft over 60 to 75 ft). These are mature trees; young trees grow
in a narrow pyramid, so a young tree may sit low in the band. No source gives a crown base or a
codominant fork for a young sweetgum, so those ranges are left out and `reference-range`
reports them as not judged.

`apical.control` is the one fitted input. At 0.54 (fitted with alpha 18), the kit's
alpha 16.75 grows a 9.5 to 9.7 m tree in 6 years, outside the grower's scale guard. On seeds 1
to 4 at 3.87 m:

| apical.control | seed 1 | seed 2 | seed 3 | seed 4 |
| --- | --- | --- | --- | --- |
| 0.48 | 0.722 | 0.546 | 0.593 | 0.853 (out) |
| 0.50 | 0.790 | 0.722 | 0.786 | 0.677 |
| 0.52 | 0.726 | 0.649 | 0.683 | 0.656 |

0.50 passes on all four seeds with a scale of 0.76; 0.52 also passes but scales the tree by
0.56, which shrinks each leaf to 56% of its cited size. So 0.50.

Held-out seeds 5 to 8 at 0.50 (spread / height):

| Seed | Full, 3.87 m | Lite, 3.17 m |
| --- | --- | --- |
| 5 | 0.754 | 0.802 |
| 6 | 0.802 | 0.669 |
| 7 | 0.778 | 0.766 |
| 8 | 0.903 (out) | 0.819 |

Seed 8 full misses the band's top. Seeds 5 and 8 also grow a trunk whose ring dips 0.011 to
0.029 units below the soil past its base, which `floor-organic` fails. That is a grower matter,
not a species one, so the bar uses seeds 7 (full) and 6 (lite), which pass every check.

## What the page camera shows

An open young tree, taller than wide, with star leaves on upright limbs. The trunk forks low into
several upright leaders rather than one straight central leader, and four root flares spread at
the base. A real young sweetgum keeps one leader in a narrow pyramid; this grower does not reach
that form at this age.

## Checks

`uv run model-kit gate bars/04-sweetgum/scene` prints `passed`. The plant checks
`reference-range`, `taper` and `leaf-angle` are registered with bar 02's rebuild; on this
scene they were run directly, with their plants, and passed.

## Sources

- Krinard, R. M. and Johnson, R. L. 1985. Eighteen-year development of sweetgum plantings on two
  sites. Tree Planters' Notes 36(3): 6-8.
- NC State Extension, Liquidambar styraciflua.
- Gilman, E. F. et al. UF/IFAS ST358, Liquidambar styraciflua.
- Virginia Tech dendrology fact sheet, sweetgum.
- Orwa, C. et al. 2009. Agroforestree Database 4.0.
- Okabe, T. 2016. Acta Societatis Botanicorum Poloniae, doi 10.5586/asbp.3527.
- Chianucci, F. et al. 2018. Annals of Forest Science, doi 10.1007/s13595-018-0730-x.
- Demol, M. et al. QSM trees, doi 10.5281/zenodo.4557401 (CC-BY 4.0).

## Taste read

On the sheet the pieces read as good enough for scenery. Two things are noted, neither broken: the redbud crown sits a little off the trunk in two clumps, and the sweetgum forks low into several stems. Outline: none (`dials.edges.outline` stays null), because an outline on every leaf turns a crown solid ink.
