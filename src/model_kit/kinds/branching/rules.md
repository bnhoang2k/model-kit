# Branching pieces: what each check measures

A branching piece is a tree grown from cited rules and a seed, with a skeleton (each limb's
axis and radii, each leaf's frame) kept beside its mesh. A check lists here only if its `kinds`
names `branching`. Each heading below is the name of a check, with what it measures, its rules,
the tolerance factors it uses and the plants that prove it can fail.

A limb is one branch, twig or root. A leaf is one card with a node on its host limb and a blade.
The skeleton has two grow modes: a full skeleton draws each leaf's petiole (the leaf stalk) as a
ribbon, and a lite skeleton draws none.

Tolerance factors live in `core/tolerances.py`. Limb checks scale a factor by the limb radius
where they measure. The floor and skeleton-match scale a factor by the piece's tube thickness
base (the thinnest listed wall of its tube parts, so leaf rows do not count, or the style's
`min_thickness` dial, whichever is larger). The planted faults
use `CALIBRATION_FAIL` (2.0) and `CALIBRATION_PASS` (0.5), a two-way calibration of this kit's own design.

Two checks, crossing-organic and near-miss-organic, leave out some pieces of a limb pair. The
leave-outs (`core/grow/segments`) are the same ones the grower's stop-short rule uses, so the
grower and the checks see one crotch. A limb's seat part is its base, up to where its gap to its
parent first exceeds `SEAT_ZONE_OPENS_OF_RADIUS` (1.2) x the thinner radius, or where that gap
stops growing, and never past a cap set by the two radii. Two siblings that
grow from one place also share a zone. A limb's own neighbouring segments (within 2) are left
out too.

## limb-seat

Measures whether every limb sits on its parent and every leaf sits on its host. It reads the
skeleton only.

Rules, per limb (item `<pack>:<piece> <limb id>`):
- A limb with no parent that is not rooted floats. It fails.
- A limb's base must lie within `SEAT_OF_PARENT_RADIUS` (1.0) x the parent's radius of the
  parent's axis.
- A limb's base ring must not poke out of the parent. The base's distance from the axis, plus
  the limb's base radius, must stay within `SEAT_CHILD_OF_PARENT_RADIUS` (1.05) x the parent's
  radius there. This also covers a base radius over that share of the parent's radius.

Rules, per leaf (item `<pack>:<piece> leaf.<n>`):
- The leaf's node must lie within `SEAT_OF_PARENT_RADIUS` x the host's radius of the host's axis.
- The blade's base must lie within the petiole allowance, plus `SEAT_OF_PARENT_RADIUS` x the
  host's radius, of the node. On a full skeleton the allowance is the leaf's petiole length,
  because the mesh draws the ribbon. On a lite skeleton the allowance is 0, so the blade must
  start within the host's radius of its node.

Plants, one unit each, built from the skeleton's points, not by calling the rules above (the
leaf plant copies the petiole allowance on purpose):
- On a piece with a seated limb, one limb with a grandparent (else the first limb with a parent)
  is moved off its parent's axis, across the parent's tangent and away from the axis, by
  `CALIBRATION_FAIL` x the parent's radius.
- On a piece with a leaf, leaf 0's blade base and blade are moved out along its petiole by
  `CALIBRATION_FAIL` x the host's radius, plus any shortfall of its base distance below its
  allowance. The plant clears the line whatever the start.

## floor-organic

Measures whether the piece stands in the soil. The soil is y = 0 in the piece frame. It reads
the skeleton and the leaves' triangles. `tol` is `FLOOR_OF_THICKNESS` (1.0) x the tube thickness
base.

Rules, per limb (item `<pack>:<piece> <limb id>`):
- A rooted limb must have no parent. Its base must not sit above the soil. It must not be sunk
  deeper than `ROOTED_DEPTH_OF_RADIUS` (1.2) x its own base radius (the rooted depth limit).
  Past the first `ROOTED_DEPTH_OF_RADIUS` x its base radius of its arc length (the stub that may
  stand in the soil), no point's ring may dip below `-tol`. The stub is read by arc length, so a
  short first segment does not make a legal stub read as a dip.
- A buried limb (a root flare) must grow from a rooted limb. It must be seated no higher than
  `BURIED_SEAT_OF_RADIUS` (3.5) x that limb's radius there above the soil (the buried seat
  height). Its tip must be in the soil, not above it.
- Any other limb, past its seat part (the part buried in its parent's wood), must not dip below
  `-tol` once its radius is taken off.

Rules, per leaf (item `<pack>:<piece> leaf.<n>`): no vertex of the leaf's triangles may dip
below `-tol`. A leaf's lowest possible vertex is bounded from its node, base and size first, so
only the leaves that could dip have their triangles built.

Plants, one unit each on every grown piece, built from the skeleton's points and the leaves'
triangles alone:
- Lift: the whole skeleton is lifted until the deepest rooted base sits `CALIBRATION_FAIL` x
  `tol` above the soil.
- Sunk: the whole skeleton is sunk until that base sits at `CALIBRATION_FAIL` x its depth limit
  below the soil. Lift and sunk share one item, the deepest rooted limb.
- Sink leaf: the leaf whose node is lowest is sunk so its lowest vertex sits at
  `-CALIBRATION_FAIL` x `tol`. A piece with no leaves sinks the free limb with the lowest point
  (less radius) the same way; if the limb's seat part hides that point, a second candidate sinks
  the limb until its highest point (plus radius) sits there.

## crossing-organic

Measures whether two limbs grow through each other. Every limb pair is read (a parent and its
child too), and each limb is read against itself, so a limb that curls back through itself is
seen. For each pair the check finds the piece pair with the smallest gap share: the gap between
the two axes less both radii, over the thinner radius there. The leave-outs above (seat parts,
shared zones, a limb's own joints) are not read.

Rule (item `<pack>:<piece> <a> x <b>`): the pair fails when the limbs overlap by more than
`CROSS_OF_RADIUS` (0.5) x the thinner radius there. The zones open at
`SEAT_ZONE_OPENS_OF_RADIUS` (1.2).

Leaf cards have no inside, so a crossing does not apply to them. They are left out. A coverage
line names each piece that has leaves, once per piece.

Plant, one unit per grown piece: one twig is moved rigidly so it overlaps an unrelated limb, past
that limb's seat part, by `CALIBRATION_FAIL` x `CROSS_OF_RADIUS` = one thinner radius. A tree with
no twig on an unrelated limb (a seedling: every limb grows from the trunk) moves another limb,
outermost level first, against an unrelated limb, then against any other limb. The unit holds up
to three placements; a piece whose limbs take none (a lone trunk) stops the gate.

## near-miss-organic

Measures whether two limbs graze or almost touch. It reads the same limb pairs, with the same
leave-outs, as crossing-organic. The two checks split the gap share between them, so a pair is
never reported by both.

Rule (item `<pack>:<piece> <a> x <b>`): with `g` the gap and `r` the thinner radius, the pair
fails when `-CROSS_OF_RADIUS` x r <= g <= `NEAR_OF_RADIUS` (0.5) x r. That is closer than reads
apart at the page camera, but not deep enough to be a crossing. The zones open at
`SEAT_ZONE_OPENS_OF_RADIUS` (1.2), which sits above `CALIBRATION_FAIL` x `NEAR_OF_RADIUS`, so
a crotch that has opened never reads as a near miss.

Leaves near a limb they do not grow on pass, because real crowns do this. Leaf cards are left
out, with one coverage line per piece that has leaves.

Plant, one unit per grown piece: one twig is moved rigidly to a gap of `CALIBRATION_PASS` x
`NEAR_OF_RADIUS` x r (a quarter of the thinner radius) from an unrelated limb. A seedling, and a
piece whose limbs take no placement, are planted and stop the gate as crossing-organic's are.

## skeleton-match

Measures whether the mesh that ships is the mesh the skeleton makes. The other branching checks
read the skeleton, so this check proves they judge what ships. The scene's mesh is compared part
by part with the mesh the skeleton grows (item `<pack>:<piece> <part>`). `tol` is
`SKELETON_MATCH_OF_THICKNESS` (0.05) x the tube thickness base, about a tenth of the thinnest
limb radius.

Rules, per part:
- A shipped part the skeleton does not grow fails.
- The triangle counts must be equal. A part the skeleton grows and the scene lacks counts as 0.
- Every shipped triangle must match a triangle of the skeleton's part, and the reverse. Two
  triangles match when each corner of one lies within `tol` of a corner of the other, both ways,
  and their normals point the same way. A re-wired, flipped or zero-area triangle over the
  right vertices fails.
- Every shipped vertex must lie within `tol` of a vertex of the skeleton's part, and the
  reverse, so a vertex missing from the shipped mesh fails.

A branching piece with a skeleton but no geometry in the scene is not measured. It is listed
in coverage, and the run is partial.

Plant, one unit per grown piece: one shipped vertex is moved along +Y by `CALIBRATION_FAIL` x `tol`.
The plant picks a vertex whose nearest other vertex of the skeleton's part lies over
`PLANT_CLEAR_OF_TOL` (3.0) x `tol` away, so the move cannot land near another vertex. Every
copy of that vertex in the part moves. If no vertex of the part stands that clear, a second
candidate moves the part's first vertex along +Y by the part's bounding box diagonal.

## near-crease

A drawn line follows a real edge. The runtime outline draws an edge when the turn between its two
faces is over the style's one crease angle (`dials.edges.crease_deg`). Two faults follow, on
every part but those of shape `cable` or `decal` and of the roles that draw no outline (`eyes`,
`skip_outline`):

- **Rule A, an ambiguous edge.** An edge whose turn lies within its band of the crease angle can
  flip between builds. The band is how far the position rounding of the shipped file
  (`measured.export.position_bits`: a Draco step over the part's own box, or float32 spacing when
  it is null) can move the turn, from the dihedral angle's gradient. On a part bound to more than
  one bone the turn is also read at every clip sample of its rig: an edge fails when its drawn
  state differs between two samples (or the rest pose), or a sample brings it within its band. A
  band wider than `CREASE_BAND_CAP_OF_ANGLE` x the crease angle belongs to a sliver face: the part
  gets one `sliver` finding and the edge is not called a near crease. A piece with no
  `measured.export` is `Unmeasured` for this rule.
- **Rule B, a dangling dash.** The drawn edges of a part join into lines. A vertex with exactly
  one drawn edge, not on an open boundary (a leaf card's rim has one), is a line end, and it
  fails at any angle. It is read at rest and, on a part bound to more than one bone, at every
  clip sample. It is judged only as it will be seen: a line end whose stub (the drawn edges from
  the end to the next junction or the other end) is shorter than one arcminute at the piece's
  `view.nearest_m` (`EYE_RESOLUTION_ARCMIN`) is dropped and counted in a coverage line, and so is
  one whose stub lies wholly on a side the view lists as `hidden`. A piece with no `view` is
  judged as before.

A piece that declares `outlined: false` (with an `outlined_why`) is skipped with a
not-applicable line: its shipping look draws no outline on it, so no line can dash or flip.

Limit and sources: the crease angle is the style's; the band is the dihedral gradient of
Bridson, Marino and Fedkiw 2003, Section 4 (https://www.cs.ubc.ca/~rbridson/docs/cloth2003.pdf),
with each corner moved by sqrt(3) / 2 of a step; the cap is `CREASE_BAND_CAP_OF_ANGLE` (a stated
design share, set before any bar was measured). Drafting leaves tangent edges out of orthographic
views (Ford, Engineering Graphics and Design; SolidWorks help) as supporting evidence.

Items: `<pack>:<piece> <part> edge <i> near`, `... edge <i> bends across`, `... sliver` and
`... line end <v>`. A style that draws no outline prints a not-applicable line; a style with no
`crease_deg` is `Unmeasured`. Plants, one per rule per piece: an edge folded to half its band
over the crease angle; a skinned edge bent across the angle at one clip sample; a vertex of a
smooth part raised until the edges around it draw and their neighbours do not. The check must
flag each.

## detail

A curve has the facets the eye can see at the distance the piece is seen from, and no more. An
edge under the style's crease angle shades smooth, so it stands for a curve; the two flat faces
that meet there differ from that curve by a gap, the sagitta `(w / 2) tan(turn / 4)`, with `w`
the narrower face's height over the edge. Each piece declares `view.nearest_m`, the nearest
distance in real metres its viewer's eye comes from, and why. Every smooth edge of every part but
those of shape `decal` and of the roles that draw no outline is judged at rest in real metres
(`units_per_metre`); no camera is assumed, and an edge whose two faces both look toward a side
the `view` lists as `hidden` is left out.

- **Rule 1, blocky.** The worst edge's sagitta is over `nearest_m x tan(EYE_RESOLUTION_ARCMIN)`.
- **Rule 2, finer than the eye.** Take an edge out and its two faces become one chord; the old
  corner then stands off it by the edge's merged gap. For a ring of equal faces that is
  `4 cos^2(turn / 4)` times the sagitta (a ring with half its segments); for unequal faces, such
  as a string kinked over a nut, it is `a b sin(turn) / sqrt(a^2 + b^2 + 2 a b cos(turn))` with
  `a` and `b` the faces' heights over the edge, which is more. A part fails when its largest
  merged gap still passes rule 1 and its largest smooth turn doubled stays under the crease angle
  (past it, halving would add a drawn line). It is measured by gap, never by triangle size, so
  the small triangles at a sphere's pole are not waste.

An edge is smooth when its turn is under the crease angle and over the rounding a float32 mesh
carries (the rounding band at the float32 step), so a flat quad's diagonal is not smooth and a
finely cut curve is, whatever the piece's size.

A limb can turn sharply where it grew so; a tree is not a hose bent smooth. Where a bend's
bands leave no room for an arc wider than the tube, the mesher does not round it (the arc would
fold the wall) and keeps the skeleton's sharp crease. A crease whose band is shared with another
bend may still fold its mitre's inner wall a little. The build does not refuse either, and adds
no finding of its own: this check judges their edges like any other edge.

Limit and source: `EYE_RESOLUTION_ARCMIN` (1.0), the minimum angle of resolution of normal sight
(EyeWiki, https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, secondary; a design call of
this kit); about 0.29 mm a metre. No triangle count is compared with a limit. A part with no smooth
edge (a box) is not applicable; a piece with no `view` or a style with no `crease_deg` is
`Unmeasured`. Items: `<pack>:<piece> <part> blocky at <D> m` and
`... finer than the eye at <D> m`. Plants, one per rule per piece: a part scaled about its
centroid until its worst sagitta is twice the limit, and one until its largest merged gap is half
of it; when scaling cannot make the waste, the piece is declared from far enough away instead.

## reference-range

Measures whether a grown tree's size and form lie inside its species' cited ranges. It is a
result check, not a tree-file check: it judges what the bud grower made, and it reads the
skeleton and the species' rules file. It is the fit target: the species file's `fitted` inputs
(apical control) are fitted to its ranges on seeds 1 to 4. It is the one check that may be fixed
by refitting a fitted input. Any other failing plant check is fixed by changing a general rule,
for every species.

Each quantity is a share of the tree's height (the highest limb point, which is how the grower
scales the tree to `height_m`), judged against the range of the same key in the rules file
(`ranges.<key>`, with its cited rows). Items are `<pack>:<piece> <range key>`.

- `spread_of_height`. The spread is the forest-inventory crown diameter: the mean of the crown's
  widest width between the driplines and its width at right angles to it (Schomaker et al. 2007,
  USDA FS GTR SRS-102, sec 2.3.8.1). It is measured in plan view (glTF axes: x and z) over the
  whole crown: every limb point that is not a root and not below the crown base, and every
  vertex of every leaf. The widest width is the longest distance across the convex hull of those
  points; the width at right angles is the hull's extent in that direction. The inventory leaves
  out abnormally long one-sided branches; this check does not.
- `crown_base_of_height`. The crown base is the height of the lowest live leaf (the inventory's
  "lowest live foliage of the obvious live crown", sec 2.3.2.2). A limb's seat is not used: a
  second stem seated at the soil with leaves only high up would read a crown base of 0. A tree
  with no leaves has no crown base: the range is not applicable, and the spread is measured on
  the wood.
- `fork_of_height`: taste-only. No cited numeric definition of a codominant fork exists (Gilman
  2003, AUF 29(5): 291-294, gives none). A species file may carry the range as a note; the check
  prints a not-judged line for it and judges nothing.

The rules file is the copy in the scene's `rules/` folder, found by the file name in the
skeleton's `info["rules"]`. The skeleton also records the hashes of the species file and of the
kit file it was grown from. A copy that differs fails, naming both hashes (item
`<pack>:<piece> rules` or `<pack>:<piece> kit`), and the ranges are then not judged, because they
belong to another file. A rules file missing from the scene is `Unmeasured`, so the run is
`partial`. A skeleton with no hashes (one saved before the bud grower, or a measured tree) is not applicable: a
coverage line says so, and it never fails.

Tolerance factor: `RANGE_PLANT_OVER` (1.1), used only by the plants. The ranges are the
tolerance.

Plants, one unit for each range key a judged piece has, built from the crown wood's widest width and
the leaves' heights alone, never from the check's own spread:
- Too wide: the crown is scaled across x and z about the trunk by the factor that puts at least
  `RANGE_PLANT_OVER` x the upper end of the spread range into the spread (the spread is at least
  half the wood's widest width, so twice the target over that width does it). Height and crown
  base do not change.
- Bare below: every leaf whose node sits below `RANGE_PLANT_OVER` x the upper end of the
  crown-base range of the height is removed. A tree with no leaf above that line has all its
  leaves lifted until the lowest sits on it.

## Tree-file checks

There are two kinds of tree check. A result check judges what the grower made: its measure is not an
input. `reference-range` and the five checks that judge any skeleton (`limb-seat`, `floor-organic`, `crossing-organic`, `near-miss-organic`, `skeleton-match`) are result checks. A tree-file check binds any
skeleton file to a fact about all trees. It judges a grower's tree, a hand-made tree, a scanned
tree, or a tree grown from a species file with an impossible input. On the kit's own grower it
catches only code and data mistakes (a bad radius pass, a blade angle of 85 degrees in a species
file). It proves nothing about how well the grower grows. Its quiet case is the measured trees
(`test/fixtures/qsm`, four lidar scans of beech and ash, CC-BY 4.0) and the measured data. `taper`
and `leaf-angle` are tree-file checks. Each module's docstring opens with that sentence.

## taper

Measures whether a limb's radius falls from its base to its tip. It reads the skeleton only, in
real metres (model units over the scene's `units_per_metre`).

Principle: the pipe model. A stem's cross-section follows the leaf area it carries, so radius
falls toward the tips ([Lehnebach et al. 2018, Ann. Bot. 121: 773-795](https://pmc.ncbi.nlm.nih.gov/articles/PMC5906963),
reviewing Shinozaki et al. 1964).

Form: per limb, the least-squares slope of radius over arc length. The arc length is taken along
the whole limb, then read at the points used. The slope is fitted on the radius less the limb's
first used radius, so a limb of equal radii (a radius floor) has a slope of exactly 0.

Threshold (item `<pack>:<piece> <limb id>`): a limb fails when its slope is over 0. A per-step
limit is out: measured trees grow by more than 10% in 8 to 9% of their cylinder steps, which is
noise.

Which limbs are judged depends on the skeleton:
- A measured skeleton (its `info` names a `source_doi`) uses only the points with a radius of at
  least `RELIABLE_RADIUS_M` (0.05 m). A limb is judged when its base radius is at least that and
  it has `TAPER_MIN_POINTS` (4) such points. Source: [Demol et al. 2022, Silva Fennica 56(1) 10550](https://doi.org/10.14214/sf.10550):
  leaf-off ash scan diameters under 10 cm are overestimated, by 4% above 10 cm and 73% under
  2.5 cm. On the four fixtures this judges 1, 4, 1 and 4 limbs, and none rises. The quiet case is
  real but thin.
- A grown skeleton has no scan error. Every limb with `TAPER_MIN_POINTS` points is judged. (The
  reliability radius would leave a 5 m tree almost nothing to judge.)
- A coverage line counts the limbs not judged on each piece.

Plant, one unit on every piece with a judged limb: the judged limb with the most points is given a
radius that grows from its base radius by `TAPER_PLANT_RISE` (30%) toward its tip, in step with arc
length. A piece with no judged limb has no unit; its coverage line counts the limbs not judged.

## leaf-angle

Measures whether a tree's blades lean where measured broadleaf crowns put them, and spread their
angles in every part of the crown. It reads each leaf's `normal` and the height of its `node`.

Principle: measured broadleaf crowns hold blade angles with a stable range of means and a spread
in every crown part ([Chianucci et al. 2018, Mendeley Data 10.17632/4rmc7r8zvy.2](https://data.mendeley.com/datasets/4rmc7r8zvy/2),
23,882 leaves of 138 species; [Pisek et al. 2013](https://experts.nau.edu/en/publications/is-the-spherical-leaf-inclination-angle-distribution-a-valid-assu/),
read as an abstract only). The blade angle is the angle of the blade from horizontal (0 flat, 90
vertical): the angle between its face normal and the vertical, the de Wit convention the dataset
uses. The grower stores the face normal in the skeleton's `normal`.

Rules:
- `<pack>:<piece> mean`. The mean blade angle of the whole tree must lie between
  `LEAF_ANGLE_MEAN_MIN_DEG` (10.96) and `LEAF_ANGLE_MEAN_MAX_DEG` (72.06). These are the lowest
  species mean (Carpinus betulus, 12.90, SD 9.69, n 100) and the highest (Liquidambar
  styraciflua, 69.43, SD 12.58, n 91), each widened by 2 x SD / sqrt(n) of its own species. Without
  the widening, a sweetgum fitted to 69.43 would fail on about half its seeds.
- `<pack>:<piece> third.<0|1|2>`. The standard deviation of the blade angles (n - 1, as the
  dataset's R script takes it) in the bottom, middle and top third of the crown must be at least
  `LEAF_ANGLE_SD_MIN_DEG` (9.69). The thirds are equal-height slices of the span of the leaves'
  node heights, as the dataset's lower, middle and upper canopy sets. 9.69 is the smallest SD over
  all 262 species-sector-date sets, whole canopy included (Carpinus betulus). The smallest SD of a
  sector alone is 9.80, but a crown's variance is at least the mean of its thirds' variances, so
  9.80 would fail a measured species whose only set is a whole canopy.
- A mean or a third with fewer than `LEAF_ANGLE_MIN_LEAVES` (26) leaves is not judged, and a
  coverage line says so. The dataset's sector sets hold 26 to 133 leaves, so the band and the
  floor were not measured on fewer.
- A skeleton with no leaves (a scan, a winter tree) gets a not-applicable coverage line, no
  finding, and no error.

What it cannot see: only the angle from horizontal, not which way a blade faces. A crown whose
blades vary in angle but all face one way passes. Such facing plates are taste-only (judged by
eye): no cited facing-spread measure was found, and flat two-ranked sprays are real
(Barthelemy & Caraglio 2007), so a visual review judges them. A seed-1 redbud fixture in the
grow tests would be expected to fail the floor in every third.
It does not: its thirds spread 12.6, 12.8 and 12.9 degrees, above 9.69. The fixture is kept, and
a test in the grow tests records the numbers. What the dataset file proves:
`leaf_angle_refs.json` holds one mean, SD and n per species, so `test_measured_species_pass`
shows every species mean in the band and every species SD at or above the floor. It holds no
per-sector sets, so it does not show that every third of a species' crown reaches the floor.

Plants, one unit each on every piece with enough leaves; a plant turns each blade about its own base
to the new angle, and keeps the way it faces:
- Plates: every blade set to the middle of the range of species means (so the mean stands
  apart). Every judged third's SD falls to 0. One unit for each judged third.
- Steep: every blade set to `LEAF_ANGLE_PLANT_STEEP_DEG` (85), a species whose beta has a mean of
  85. The item is `mean`.
- Flat: every blade laid flat (0). The item is `mean`. Steep and flat are the two ends of the one
  mean rule, so they share an item.

## leonardo (taste-only)

No `leonardo` check ships. Leonardo's rule says the branches above a fork carry the same total
area as the stem below it, as r_parent^D = sum of r_child^D, with 1.8 < D < 2.3 ([Eloy 2011,
arXiv:1105.2591](https://arxiv.org/abs/1105.2591), as the botany note cites it). A sweep fitted D on
the four measured trees, over forks whose parent radius is at least 0.05 m, with 10 to 31 forks
each. The original form gives D = 1.235, 3.295, 1.750 and 2.160 (one tree in the band), the
log-radius form gives 2.44 to 6.0, and the per-fork median gives 1.68 to 2.72. At a scan fork the
continuation is nearly as thick as the parent, so D is poorly identified. Choosing a radius or
widening the band until the four fit would be tuning to pass. Fork shape is judged by eye on the
rendered tree.

## even-cover (taste-only)

No `even-cover` check ships. A search found no clumping-index source (He et al. 2012 is not
open access). Even with no clumping, the strictest cited floor is 0.160 cover over the
redbud's leaf depth, below the 0.24 its bald top measured, so no cited floor can catch the
defect this check was built for. Thin or bald crowns are judged by eye on the
rendered tree.

## crown-balance and branching-stats (taste-only)

Neither check ships. A search found weak support for both. `crown-balance` asks that the leaf
area's centre sits over the trunk base, but leaning crowns are real, and a
band needs a cited source first. `branching-stats` asks for a bifurcation ratio by Strahler order.
Leopold 1971 gives 4.7 to 6.5 (as reported in secondary literature), and Steingraeber & Waller 1986 argue the ratio
changes with order, so a band would have to be computed on the measured trees first.
A script in the branching tests prints both measures for information. Crown lean and branch
counts are judged by eye on the rendered tree.
