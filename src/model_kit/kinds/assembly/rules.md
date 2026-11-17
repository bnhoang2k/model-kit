# Assembly pieces: what each check measures

An assembly piece is put together from parts by a build script. A check lists here only if its
`kinds` names `assembly`. Each heading below is the name of a registered check, with what it
measures and the plant that proves it can fail.

The body, clip, layout and scene checks (`must-touch`, `reach`, `trace`, `sides`, `mirror-pair`,
`no-fallback`, `overlap`, `solver-error`, `clip-hygiene`, `stress-poses`, `bend-quality`,
`seat-fit`, `receiver`, `connectors`, `export-hygiene`, `bake-files`, `lightmap-overlap`,
`seam-jump`, `lightmap-density`, `coverage`) measure a posed body, a clip, a layout, a pack's packed file, a light bake or the
whole scene, not a recipe, so they declare no kind and have
no heading here.

## crossings

Two parts of one piece must not sink into each other. Each pair of parts whose boxes overlap and
whose surfaces meet is measured for how far they share space (the shortest move that separates
them, the penetration depth of collision detection, read from samples of both surfaces) and for
the thickness of the thinner part there (a cone of rays, the shape diameter). A pair fails when the shared depth is over half the thinner
part's thickness: a part sunk more than halfway into another is embedded, not resting. The limit
follows the parts and not the pack. Flush contact shares nothing. A part that is open cannot be
measured and leaves the run `partial`. A designed join, such as a pin in its socket, a tenon in
its mortise or a bolt through a plate, is an allowlist line with a reason. Plant: a copy of one
part sunk twice the limit into its neighbour. The check must flag the pair.

## take-apart

Each group of a piece must come away from the others without cutting them. The groups are
taken apart one at a time, the last one built first, and a group taken off earlier is gone, as in
a real teardown: the moving group is measured only against the groups still on the piece, as
built. A group is tried first along its pull, the line from the piece's centre to its own centre.
If that cuts, it is tried along each of 26 fixed ways (the 13 directions crossings reads, both
signs), the ways nearest its pull first, and it comes away when any one is clear. A group at the
piece's centre has no pull and is tried along the 26. Along a way the group moves out in 21 equal
shares of the piece size from 0 to 1. Each pair is measured as `crossings` does, with the same
limit (half the thinner part's thickness), and cuts when the space it shares is deeper than it was
at rest by more than that limit (a seat at rest is `crossings`' to allow). So the check is never
stricter than the pull alone, for a group that has one. When every way cuts, the finding names the
way that cuts least deep as a unit vector. A cable (a part that bends) neither moves nor blocks:
only its rigid plugs are tried, and it says so in a `not applicable` line. A piece in one group
has nothing to take apart. Source: Wilson and Latombe 1994 (read, section 5.3 and p. 372), a part comes
out along a direction nothing blocks. Plant: six slabs around one group, a gap clear of it and as
thick as the piece is large, in a group of their own that is built first and so comes off last. No
way out is free, so the check must flag the pair of groups.

## floaters

Each part reaches something that stands. A part with `on` must touch its `on` part, and a part
with no `on` must rest on the piece's lowest point or be named an anchor in `measured.hosts`;
touching means within the piece's contact length: one arcminute (`EYE_RESOLUTION_ARCMIN`) at the
piece's `view.nearest_m`, in project units through `units_per_metre` (`sight.contact_length`).
A gap a viewer can tell apart is a float, and a smaller one touches. A side the view lists as
`hidden` changes nothing here: a float is a physical fault. A piece with no `view` has no contact
length and is `Unmeasured`. Plant: lift one small part twice the contact length off its host, or
off the floor. The check must flag the part. A part with `rests_on` must touch each support it
names (`<part> off <support>`). Plants: tilt it off the support it shares least contact with, and
name a support it does not touch.

## topology

Each part is one closed surface the runtime outline can follow. Corners within a small share of
the piece size (the diagonal of the box of all its parts) are welded. Then a part must have no
zero-area face (a face with a repeated corner, or with an area under a small share of its own
longest edge squared), no open edge, no edge on 3 or more faces, a consistent winding and a
positive volume. Only a face with a repeated corner is dropped before the edges are counted, so
a thin sliver never opens an edge. Parts of shape `leaf`, `cable` or `decal`, and of role
`skip_outline`, are left out. Plant: flip one face of one part. The check must flag the part.

## frames

Each point and direction in a piece's manifest is read in the frame it is written in, and lies
where it says. Every clip and layout that places the piece must name its `frame.axes`; every hold,
seat and mount point must lie within the piece's contact length of its surface parts (a mount: of
any part; a piece with no `view` has no contact length, so this rule is `Unmeasured`); and a
hold's `off` must lie within an angle of the outward normal at the nearest surface point. Plant: a copy of the piece in another frame, a copy with y and z swapped,
and a copy with one `off` turned past the angle. The check must flag each.

## done

Each piece says what done looks like and how big it is: `done` is text with something in it and
`size_class` is one of the allowed classes. The pack loader already refuses both, so this guards
a pack built in code. Plant: a copy with `done` emptied, and one with `size_class` blanked. The
check must flag each.

## keep-off

A mount stays clear of the features its piece says to keep off: a truss coupler, a brace node,
any spot where a mount would land on something it must not sit on. A piece lists `keep_off`
points, each with a radius. Every mount of the piece is measured against every point, and a mount
nearer than the radius fails. The check reads the manifest, so a piece with no geometry in the
scene is measured too. Plant: a copy with one mount moved onto a keep-off point. The check must
flag the pair.

## mirror-exact

A mirrored part is its source's exact image, triangle for triangle. A reflection keeps every
vertex, so a part and its mirror hold the same triangles; a quad split along its other diagonal on
one side draws a different crease there. A pair is a part whose `form` is `mirror_of` another, or
two parts whose bones swap under the rig's mirror map. The source's triangles, reflected and with
their corners welded, must be the target's triangles. When each side holds the same number of
parts on those bones, each source pairs with the target it matches best; when the counts differ,
the group is `Unmeasured`. Plant: one quad's diagonal flipped on the target of each pair. The
check must flag the pair.

## clearance

Two separate pieces placed side by side keep a gap a finger fits through, or they are joined. In
each layout, every pair of placements with no `on` or `rests_on` between them is measured for the
nearest distance between their solid parts at the placed transforms, and a pair that touches,
overlaps or sits under 25 mm (`MIN_GAP_M`, ISO 13854's finger gap, in real metres times
`units_per_metre`) fails. A placement that `rests_on` another must touch it instead, within the
resting piece's contact length. Parts of
shape `cable` are left out, and so are branching pieces. Plant: one placement moved toward its
nearest unjoined neighbour to half the minimum gap, and one resting placement lifted twice the
contact length off its host. The check must flag each pair.

A plug seated in a socket is the one contact between unjoined pieces that is meant, and a power
mount refuses a placed piece (`receiver`), so a connector end joins it. A connector end that names
a layout and a host placement seats the cable piece's plug part (the end's `plug` record and every
part resting `on` it, such as its blades) in the part of the host the mount names (the mount's
`part` record). Only those pairs of parts are not judged; the plug against the host's other parts,
and every other part of the two pieces, still are. A cable piece with more than one placement in
the layout seats nothing, as in `connectors`; so does an end with no `plug` or a mount with no
`part`, and a plain coverage line says so (the parts are never guessed from the geometry). A
coverage line names each pair it excuses. An excused pair that shares more space than the
shorter of the two pieces' contact lengths (a plug sunk into, or driven through, the host part)
is still a finding; contact is excused, penetration is not.

## markings

A print, a paint film, an inlay or a screen image is a change of colour on a surface, not an
object: ink and paint films are microns thick. A listed row says how it is made in `finish`
(`solid` by default, or `print`, `paint`, `inlay`, `display`), and a marked row's parts are
decals. Two rules. Every part of a marked row has shape `decal`. Every part of shape `decal`
belongs to a marked row. A third rule, for solid rows that are really markings (a part so low
over its `on` host that it shows only as a colour change), is the markings-flush check, which reads the piece's `view`:
it is judged by the eye's resolution (1 arcmin) at the piece's nearest viewing distance. Plant: a decal
rebuilt as a solid box, and a decal's row made solid. The check must flag each.

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

## inset-seat

A part printed on a surface or set into it meets that surface along one closed line. It reads
every part of role `eyes` or `decal` and its `on` host, by the part's fill shape (eyes draw no
outline).

- **Printed.** An open patch, or a closed part with no corner or face centre deeper inside its
  host than the piece's contact length (`sight.contact_length`), is printed. Every corner must
  lie within that distance of the host's surface, else `lifts off`.
- **Inset.** A closed part deeper inside a closed host than that is set into it. It must cut the
  host in exactly one closed loop (`geom.cut_loops`), else `meets <host> in <n> loops`. Its rim
  is its outline seen from outside the host: the edges between a face that looks out (a positive
  dot product with the host's outward normal at the part) and one that does not. Each rim loop is
  judged on its own, and `weaves across` when it has corners inside the host by more than the
  contact length and others outside by more than that. The contact length is the seen floor:
  `EYE_RESOLUTION_ARCMIN` (one arcminute) at the piece's `view.nearest_m`, in project units
  through `units_per_metre`. A box or disc set deep with its top proud is quiet. No crease angle
  is read.

Limit and sources: the contact length is the piece's seen floor, shared with every contact
check (`sight.contact_length`). A print is microns thick
(pad-print ink dries to about 4 microns, Inkcups ICN-B150 manual,
https://tagless.inkcups.com/wp-content/uploads/2017/08/ICN-B150-Manual-Compressed.pdf). One
arcminute is the resolution of normal sight (EyeWiki, Visual Acuity Testing in Adults,
https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, secondary).

Items: `<pack>:<piece> <part> lifts off <host>`, `... meets <host> in <n> loops`, `... weaves
across <host>`. A part with no `on` host, a closed part on an open host, any part of a piece with
no `view` (no contact length), one whose host faces cancel, one with no rim edge, and a rule with
no plant that can fail are `Unmeasured`. Plants, one per rule per piece: half
a printed part's rim pushed into its host (or lifted off a thin one); an inset part sunk wholly
inside its host; half an inset part's rim moved out of the host and half into it.

## bearing

A part is held the way an ordinary assembly holds it, in the piece's frame (y up). Five rules.
A part with an `on` host that is not a fastener rests on, or hangs from, the host's contact patch,
read as a whole by its area-weighted normal: if the patch faces up it must keep its centre of mass
over the outline of its upward faces, and if it faces down it must hang under it (a patch that
faces sideways, or up and down in equal measure, prints a not-applicable line). Rule 1 is not
applicable to a part of a row listed `held: bonded` (a glued joint), or
to any part of a rigged piece. A `fastener` part must sit inside its `on` host along its own axis
for at least its shank diameter (1 x, Nord-Lock's rule of thumb for steel; NASA RP-1228 gives the
pull-out physics).

The contact of a part and its host follows Baraff's contact model: points of the part, each with
the normal of the face it meets (the host's, or the part's own where the host meets it with an
edge or corner) and its depth along it. The part's contact direction is the mean of those normals,
each point an equal share of the part's surface, so a host face counts only where the part
touches it. Rule 3 and rule 5 read this one direction.

A part with an `on` host must have its seat face wholly over the host's surface it bears on: the
faces it meets that turn from the seat by less than the style's crease angle, and every face that
joins them across an edge turned less than that angle (a smooth curve, such as a lathed base or a
radiused board, faceted). A bevel or an edge turned more than the crease angle ends that surface.
Holes inside it are counted in. The seat is read at the level of that surface, to the contact
length, over its extent padded by the contact length; only float noise is let go. A seat read
nowhere is no pass. A style with no crease angle cannot judge a seat. Rule 3 is not applicable to a
part of a rigged piece whose bones differ from its host's. A part with `retains` must touch each
part it retains, to float noise.

A part with an `on` host that is not a fastener, in a row held `rests`, must lie on its host's
surface: no point of it is deeper inside the host than the contact length, else `<part> sinks
into <host> <mm> mm`. The depth is the shortest move that takes every point of it out of the host
(the penetration depth), over the contact direction and crossings' 13 directions, so a fret that
hangs over a board's end reads its depth under the top and not its run along the board. Its
contact window is the contact length plus crossings' limit (half the thinner part): a part pushed
past the contact length keeps its contact there, and the two limits meet with no gap. A part sunk
past crossings' limit is reported by both checks; rule 5 never passes it. It is judged by depth
and not by the up or down test of the first rule, so a part on a patch that faces sideways is
judged too. A row held `bonded` is not judged (a bond may be set into its host), and a designed
embed is an allowlist line with a reason. The sink rule is not applicable to a part of a rigged
piece whose bones differ from its host's, nor to a pair where half the thinner part's thickness
is not over the contact length (no depth lies between a seat and crossings' limit). Parts printed
on a surface are left out.

A part with `rests_on` (two or more supports, a tabletop on its legs) is held by all of them
together. Rule 1 reads its centre of mass over the support polygon, the convex hull of its upward
contact points on every support
(https://en.wikipedia.org/wiki/Support_polygon); else `<part> tips off <s1>, <s2>, ...`, naming
every support in `rests_on` order, touched or not. Rule 3 is not applicable (no single surface
holds its whole seat). Rule 5 runs once per support. A row held `bonded` skips rules 1 and 5. The
rest and sink plants run on the smallest such part.

Plant: slide a resting or hanging part off its patch, pull a fastener out to half the engagement
it needs, slide a mounted part until its seat face overhangs, lift a retainer off what it retains
by twice the contact length, and push a resting part into its host along its contact direction to
twice the contact length deep (less, when the part or its host is too thin to take it and keep the
contact; a seat with a small gap goes that gap further). The check must flag each.

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

Limit and source: `EYE_RESOLUTION_ARCMIN` (1.0), the minimum angle of resolution of normal sight
(EyeWiki, https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, secondary; a design call of
this kit); about 0.29 mm a metre. No triangle count is compared with a limit. A part with no smooth
edge (a box) is not applicable; a piece with no `view` or a style with no `crease_deg` is
`Unmeasured`. Items: `<pack>:<piece> <part> blocky at <D> m` and
`... finer than the eye at <D> m`. Plants, one per rule per piece: a part scaled about its
centroid until its worst sagitta is twice the limit, and one until its largest merged gap is half
of it; when scaling cannot make the waste, the piece is declared from far enough away instead.

## markings-flush

Rule 3 of the markings check, judged by eye. A solid part whose largest outward
stand-off from its `on` host is under what the eye resolves at the piece's nearest viewing
distance (`view.nearest_m`, real metres through `units_per_metre`) shows only as a change of
colour, so it is a marking built as an object. The limit is `nearest_m x
tan(EYE_RESOLUTION_ARCMIN)`: about 0.29 mm at 1 m, 2.9 mm at 10 m. A 1.5 mm float on an
instrument held at 1 m is over it and stays a part; hardware 0.5 to 1.5 mm proud is quiet there.

The stand-off is the largest outward distance of the part's vertices from the host's surface. A
closed host tells inside from outside by ray parity; an open host has no inside, so the distance
is unsigned. A part buried deeper than the limit inside a closed host is hidden, not a marking,
and is left to the crossing checks. A part of a row whose `finish` is not `solid`, a decal, a part
with no `on` host and a part whose outermost faces all look toward a side the `view` lists as
`hidden` are left out. A piece with a hosted part and no `view` is `Unmeasured`.

Limit and sources: `EYE_RESOLUTION_ARCMIN` (a design call of this kit; EyeWiki, secondary) and ink and
paint films under 0.1 mm. Item: `<pack>:<piece> <part> flush with <host>: a
marking?`. A real flush part is answered by its row's `finish` or an allow line with a reason.
Plant: one solid part pushed straight toward its host, as a rigid body, until its outermost vertex
stands half the limit off; a part no rigid push brings flush is pressed onto the host instead.

## coplanar

Two faces that face the same way, overlap and lie closer than one depth step draw in the same
stored depth, and a perspective depth buffer picks between them by noise: they flicker. The step
at a distance `z` is `z^2 (f - n) / (f n 2^bits)`, from the style's reference camera (`near_m`,
`far_m`, `depth_bits` in `page_camera`, with `units.depth_step`). It is coarsest at the farthest
view, so the gap is one step at the piece's `view.farthest_m`, in the pack's units. No number is
chosen.

Two items. `<pack>:<piece> <a> x <b> flicker`: a face of part `a` and a face of part `b` whose
normals agree to `COPLANAR_NORMAL_COS` (0.9999, so a steep crossing is not a shared
plane), overlap, and stay within the gap of one plane over the whole overlap. `<layout> <a>
x <b> flicker`: the same between a part of one placement and a part of another, at their placed
transforms; the pair takes the larger of the two pieces' `farthest_m`. A joined pair (`on`,
`rests_on`) is judged too. The overlap, summed over the pair, must cover more than one resolvable
spot (the contact length squared); a smaller patch is not a flicker anyone sees.
A face pair whose whole overlap one closed solid part other than the two hides on both faces (inside
it, or flush on its surface with the face turned into it) is left out: a face behind a nearer surface is not
drawn. A patch partly in the open stays a finding.

A decal is lifted off its host by the piece's contact length (`sight.decal_lift`), the largest
lift no viewer can see, and the build raises when that is under the gap: the piece's view range is
then too wide for a lifted decal. A piece with no `farthest_m`, or a style with no depth camera,
reads `Unmeasured`. A piece of another recipe kind is not judged.

Limit and sources: Khronos, "Depth Buffer Precision" (OpenGL Wiki; see
`core/checks/coplanar.py`). Plants: a face of one part is copied, in the plane it lies in,
into another part of the piece; one placement of a layout is copied in place under a new id.
