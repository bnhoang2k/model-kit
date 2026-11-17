# Bar 06: a letter sorter, toy style

The machine reference for the toy style. A parcel sorting machine, written once as a parts list at
real proportions and built as a toy of itself: squashed, rounded hard, its controls oversized and
its fine fittings left out. It is the known-good model for a machine drawn as a toy, and its
folder is the machine kit that built it, so a new machine starts by copying this folder.

The same parts list also builds the machine at real-product detail (`--style real` in the kit);
that version is not a bar. The toy was picked over it as the detail level for machines
that stand beside the chibi (bar 03-chibi), and the kit's toy numbers are this
machine's.

## Reads as

A machine 3.57 m long, 1.29 m deep and 1.42 m tall to its beacon, in 107 closed parts. A plinth on
feet carries a spine cabinet with a belt between guide rails. Along the belt: a feeder magazine with
three parcels, a C-frame scanner with a beacon, six pushers, six labelled pocket trays (four
holding a parcel), and a takeaway belt under the pockets. At the feeder end an infeed cabinet holds
a door, an operator panel (screen, three oversized buttons, an emergency stop) and a stack light.
Every part's rim rounds a full 90 degrees, so the runtime outline draws no crease on it, only its
silhouette, as on the chibi.

## Fits

- Hard-surface machines and props drawn as toys, seen from a scene camera 4 m or more away.
- Any machine built from a parts list with the kit in this folder: copy the folder, replace
  `parts.py`, and keep the toy numbers in `toy.py` unless the machine has a reason to change one.

## Wrong for

- Real-product detail. The toy drops the screws, rivets, hinges, louvres, legends, the key switch,
  the dividers, the lamp bars and the conduits. Bar 05-lamp is the hard-surface bar at real detail.
- Rigged characters and branching plants.
- Pieces seen from nearer than 4 m: every curve's facet count is set by what the eye resolves at
  that distance.
- A measured product. No product or drawing is followed; every size is a modelling choice.

## What "toy" means

The toy derives from the same parts list as the real machine, in `resolve.py`, with the numbers
in `toy.py`:

| Name | Value | What it does |
| --- | --- | --- |
| `TOY_SCALE` | 0.88, 0.76, 1.0 | Squash along x (the machine's length), y (up) and z (depth). Squatter, so the operator panel sits at the chibi's chest (0.70 m) instead of its face (0.98 m). |
| `TOY_CTL` | 1.8 | Controls (parts marked `ctl`: buttons, the emergency stop, lights) grow by this instead of squashing, the chibi's oversized look. |
| `TOY_ROUND`, `TOY_RIM` | 0.30, 0.30 | Hard rounding: the footprint's corner radius and the top rim's radius, each as a share of the part's sides. The rim sweeps the full 90 degrees, in facets of at most 30, under the style's 35 degree crease angle. |
| `TOY_MIN_SIDE`, `TOY_MIN_H` | 0.03 m, 0.012 m | Thin parts thicken to these, so a rail or a label still reads at 4 m. |
| `NEAREST_M` | 4 m | The nearest a camera gets; it sets how many facets a curve needs. |

Fine fittings are dropped: a part marked `tier: "real"` is left out of the toy, and a part marked
`keep` moves onto its dropped host's own host (the push buttons onto the panel). Each part keeps its
offset on its host face, scaled. A host's radii then ease back toward the real ones only as far as
the parts it carries need to sit wholly on its flat face. A few parts carry a `toy` override next
to their real numbers, where the squashed layout would otherwise collide.

## The machine kit

The five scripts in this folder are the machine kit, the build that made this sorter and
that a new machine reuses:

- `parts.py`: the sorter's parts as data, plus `ROLES`, `ROWS` and `META`. Its docstring says how
  a part is written (`kind`, `host`, `face`, `at`, `size`, the radii, `tier`, `ctl`, `keep`,
  `held`).
- `resolve.py`: places every part on its host face and derives the toy. Plain Python, no Blender:
  `python3 bars/06-sorter/resolve.py bars/06-sorter toy` prints every seat or collision problem and
  exits 1 if there is one. It also knows two kinds the sorter does not use: a `tilt` (a part turned
  in a side face's plane, for a sloped lane or a slanted chute) and an `arc` (an annular sector
  swept round a vertical axis, for a curved belt).
- `toy.py`: the toy numbers above. A machine overrides one by defining the same name in its
  `parts.py`, with its reason next to it.
- `machine_build.py`: turns each placed part into a closed solid in Blender and writes the pack,
  the .glb and the gate's scene folder.
- `build.py`: runs it in the toy style and installs the result (`MODEL_KIT_BLENDER=/path/to/blender
  uv run python bars/06-sorter/build.py`, from the repository root).

To start a new machine, copy the folder, write its `parts.py` at real proportions (mark fine
fittings `tier: "real"` and controls `ctl: True`), set `META`, and build. Axes are glTF: y up, the
origin at the centre of the footprint on the floor.

## What is in the folder

- `bar.json`: the bar's record. Its one reference points to the modelling choices below.
- The five scripts above, and `style.json`: 1 unit per metre; the scene camera is ortho at turn 45,
  tilt 30; crease angle 35; the same three.js reference camera for depth as the other bars.
- `sorter_toy.pack.json` and `sorter_toy.glb`: the pack (one piece, `Sorter`) and its packed file.
- `scene/`: the saved scene the gate reads (`uv run model-kit gate bars/06-sorter/scene`). It does
  not ship in the wheel. Its `layouts/floor.json` places the sorter alone at the origin.
- `sheet.png`: the sorter beside the chibi (bar 03-chibi) at one scale, in five views.

## The modelling choices

No source gives these; each is a choice. The sizes are the real-scale ones in `parts.py`; the toy
derives from them.

| Choice | Value | Reason |
| --- | --- | --- |
| Footprint | 4.0 x 1.16 m plinth; the drive motor and the end conduit reach 4.14 m overall | A long machine that reads as a line of pockets from a scene camera. |
| Height | 1.71 m to the beacon | The belt runs at 1.20 m, a working feed height; the scanner and magazine stand about 0.5 m over it. The toy's belt top is 0.955 m. |
| Parcel | 0.30 x 0.12 x 0.22 m; the magazine stack shrinks to 0.27 and 0.24 m | A small shipping box: three in the magazine, four in pockets, one on the takeaway. |
| Pockets | 6, 0.42 m wide, 0.44 m apart, shallow trays with tall dividers | Shallow fronts keep the parcels visible from the scene camera. |
| Diverters | Six pneumatic pushers behind the belt, guide rails open at each pocket | A straight paddle across the belt cannot be drawn without rotation; a pusher reads as the sort. |
| Takeaway | A stadium-ended belt under the pockets, ending at the spine's end | It feeds the main belt there; a nose past the spine would have no flat face to seat on. |
| Toy numbers | the table under "What toy means" | Squatter and softer than the real machine, so it stands beside the chibi. |
| Join | every part sinks 0.018 mm into its host | Under a 0.03 mm build bar and a 0.05 mm join limit, so a part touches its host and nothing passes deeper. |
| Flat faces | caps and walls cut to 0.1 m | A face tessellated from its corners alone left slivers metres long, which a join check misread as a deep seat. |
| `view.nearest_m` | 4 m | A viewer frames one step at a time from the front or side of a machine; 4 m is the same choice for every machine in this style, to be measured in use. |
| `view.farthest_m` | 30 m | The wide shot of the whole floor. |
| Root part name | `base` | The coplanar check's self-test plants a fault on the first two parts by name; with `beacon_base` and `beacon_cap` first, both planted faces were hidden seats and the gate stopped. |

## Gate

`uv run model-kit gate bars/06-sorter/scene` reads `passed`, with no allow list. Every row is held
`bonded` except the parcels, which rest, so `bearing` prints its rest and sink rules as not
applicable for the bonded parts. The piece has no mounts and no connectors, and stands alone, so
`connectors`, `receiver`, `keep-off`, `clearance` and `coplanar` print not applicable.
