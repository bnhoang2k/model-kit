"""Shape constants of the grower that are not species facts.

Species, size and page-budget numbers live in a rules file, each citing a rule row. What stays
here is geometry and numerics: turns in degrees, epsilons, and the crown ray march's step and
caps. Lengths are in metres; the grower converts them with the style's units per metre.
"""

import numpy as np

# Turns in degrees.
FULL_TURN_DEG = 360.0
HALF_TURN_DEG = 180.0
QUARTER_TURN_DEG = 90.0  # elevation 90 points straight up

# The up axis (glTF Y) and the side axis a vertical direction falls back to.
UP = np.array([0.0, 1.0, 0.0])
SIDE = np.array([1.0, 0.0, 0.0])

# Epsilons.
PERP_EPS = 1e-4  # a direction this close to vertical has no side of its own: use SIDE
FLAT_EPS = 1e-3  # a horizontal part (or a cross product) shorter than this is treated as none
SEGMENT_EPS = 1e-12  # floor on a segment's squared length, so a repeated point divides safely
LEAF_FRAME_EPS = 1e-6  # a leaf frame axis this close to length 1 is unit (float noise only)

# The random bend of a growing limb: per segment, a step drawn from this box (x, y, z), so a
# limb wanders sideways more than up or down. Part of the growth algorithm, not of a species.
WOBBLE_BOX = (1.0, 0.6, 1.0)

# The petiole starts this share of the host's radius out from its axis, inside the tube.
PETIOLE_START_OF_RADIUS = 0.5

# The crown ray march (`Crown.reach`): step along the ray, give up past the cap, and never
# return less than the minimum, so every limb has some length. REACH_FAR_F seeds the search
# for the ray's deepest point when the ray never enters the crown.
REACH_STEP_M = 0.02
REACH_CAP_M = 16.0
REACH_MIN_M = 0.1
REACH_FAR_F = 9.0

# Tube tips. A cone ends in one point this share of the end radius past
# the last ring; a dome (the fork crotch) steps through two shrinking rings, each (ring radius
# share, advance along the tangent as a share of the end radius), then ends in a point a whole
# end radius past the last ring.
CONE_POINT_OF_RADIUS = 0.8
DOME_RINGS = ((0.75, 0.45), (0.35, 0.85))
DOME_POINT_OF_RADIUS = 1.0

# --- the bud-grown mesh (`mesh.py`, skeletons whose info["mesh"] is "buds-v1"). Numerics of the
# mesh, not botanical facts: the root flare is a render shape that closes the crease an earlier
# version of the grower showed where a thin root leaves a round trunk (about 79 deg, up to 126 deg underneath).

# Merging collinear points: a point may be dropped when it lies within this share of its own radius
# of the chord from the last kept point, and its radius is within this share of the radius the
# chord gives there. 0.1 is the sagitta of a 7-sided tube facet (1 - cos(pi / 7) = 0.099): the
# merge moves the surface no more than the tube's own faceting already does.
MERGE_OFFSET_OF_RADIUS = 0.1
MERGE_RADIUS_STEP = 0.1
# The buds-v2 mesh (`mesh.py`) merges by what the eye resolves (`Sight.gap`) instead, so the two
# shares above belong to the buds-v1 branch only. It rounds a bend of the axis with an arc in space
# that wants this many tube radii of radius (it takes less when its bands leave less room). An arc
# of one tube radius or less folds the inner wall of the bend onto itself (the rings meet there and
# the faces between them have no area), so the mesh keeps such a bend a sharp crease rather than
# cut one; at two, the inner wall still bends with a radius of one tube radius. The principle
# proves only that it must be over 1; 2 is a choice. Geometry of the mesh, not a verdict limit.
FILLET_RADIUS_OF_TUBE_RADIUS = 2.0
# A child seated at arc fraction t keeps the point at that fraction when it is within this of it
# (float noise from rescaling the limb); a seat between two points keeps both neighbours.
SEAT_T_EPS = 1e-9

# The root flare. Each buried child pulls a lobe out of its parent's lowest rings. The lobe reaches
# out by this many of the root's own base radii, so the root leaves a surface about as wide as it is.
LOBE_OF_ROOT_RADIUS = 1.0
# Its half-width about the axis is this many times the angle the root's base radius spans seen from
# the axis (atan2(root radius, trunk radius)), at most LOBE_MAX_HALF_WIDTH_DEG, so the lobe is
# wider than the root and neighbouring lobes join. The profile is cos squared, so it eases to the
# round tube with no kink.
LOBE_SPREAD = 2.0
LOBE_MAX_HALF_WIDTH_DEG = 90.0
# The flare is full up to the root's seat and eases (smoothstep) to nothing over this many trunk
# radii above it: the flare is about a trunk diameter tall. Rings are added across that height.
FLARE_HEIGHT_OF_RADIUS = 2.0
FLARE_RINGS = 4
# A flare ring closer than this share of the radius to a ring already there is not added.
FLARE_MIN_GAP_OF_RADIUS = 0.25

# A petiole ribbon whose cross product with UP is shorter than this is vertical: it takes the
# leaf's own across axis for its width.
PETIOLE_FLAT_EPS = 1e-5

# The stop-short rule tests each segment of a new limb in this many equal pieces and cuts at
# the start of the first piece that comes too close: the cut lands within an eighth of a segment
# of where the gap falls below the clearance. Numerics, not a species fact.
CLEAR_SUBSTEPS = 8

# Stop-short compares a limb with its own earlier pieces this many pairs at a time: a memory bound
# for a long limb of short pieces, not a tolerance. It changes no result.
PAIR_CHUNK = 1 << 18

# Zones (`segments`). The narrowest angle at which two limbs from one place may part: a seat
# part or shared zone never runs past the length at which limbs parting this wide have opened
# the clearance, and it ends early where the gap (d less both radii) grows slower than sin of
# this angle per unit length. Taper adds to the gap's growth, so the line this holds is the
# angle itself for limbs of constant radius (shoots at the radius floor), and about 1.0 to 1.25
# degrees for tapered limbs: tapered twin twigs at 1.25 to 1.9 degrees pass, constant-radius
# twins fail up to 1.9. Limbs fused narrower than that line are measured and fail.
# Basis: the measured 3.68 degrees:
# - 3.68 is the narrowest TreeQSM `branch.angle` (a branch against its parent at the branching
#   point) over all 31 ash and beech trees of Demol, Gielen & Verbeeck 2021 (Zenodo 4557401),
#   branches with a base diameter of 10 cm or more (n 54; thinner diameters are not reliable,
#   Demol et al. 2022), from `test/core/checks/branching/measure_qsm_angles.py`. It rests on one
#   fork, FEXC6 (10.5 cm): the next narrowest are 5.78 (FSYL11) and 5.85 (FSYL7), and ash's next
#   is 10.45 (FEXC13).
# - It is a child-to-parent angle. Here it is applied to both stall rules, a child against its
#   parent and siblings from one place: TreeQSM's `branch.angle` gives no sibling-to-sibling
#   angle, so the parent-child value stands in for the sibling rule.
# - The four fixture trees alone (`measure.py`, radii of 5 cm or more) give 13 such forks, too
#   few to set it: their narrowest are 17.94 (trunk and child), 16.83 (order 1 and child) and
#   17.35 (sibling order 1 limbs).
SIBLING_MIN_ANGLE_DEG = 3.68

# Gaps for a zone end are read at points this share of the limb's base radius apart along its
# axis, from its base. Numerics, not a tolerance.
ZONE_STEP_OF_RADIUS = 0.25

# A limb is compared with itself only between pieces whose arc gap is more than this many times
# the sum of their radii (grown by the clearance). Numerics, not a tolerance: 1.0 is the arc over
# which two pieces of one straight tube touch by construction, whatever the internode length.
SELF_WINDOW_OF_RADII = 1.0
# The checks cut a limb into pieces no longer than this share of its self-window, so the arc gap
# between two pieces' nearest ends stays near the arc between their closest points: a pair whose
# points lie more than 1.5 windows apart along the limb is always measured, however long its
# segments. Numerics: a hairpin of two long straight segments joined by a short turn is the case.
SELF_PIECE_OF_WINDOW = 0.25

# --- the bud grower (`buds.py`, `light.py`). Numerics of the algorithm, not species facts.

# The shadow grid's cell edge, in internodes. Palubicki et al. 2009 sec 6: "each cell edge
# approximately the same length as an internode".
SHADOW_CELL_OF_INTERNODE = 1.0

# The band the grow's scale must lie in: the height the rules cite for the grown (whole) age over
# the raw grown height. A grow that misses it by more is a model failure, not a fit miss, so
# `_emit` raises instead of stretching the tree. The band holds both tiny fixture species at every
# height their ages cite (0.5 to 1.5 m, test_no_raise_across_the_cited_ages) and the redbud
# (about 1.03) and sweetgum (0.84 to 0.86) reference trees, with room; a 1-metamer maple that a
# 30x stretch would hide is far outside it. A model choice, not a botanical fact.
SCALE_MIN = 0.5
SCALE_MAX = 2.0

# Parallel transport of a leaf-rank reference along a bending axis: an axis turn shorter than this
# (the cross product of two unit directions) moves the reference by nothing.
TRANSPORT_EPS = 1e-9

# The trunk's first point sits this share of the trunk's radius below the soil, so the rooted base
# is inside it (the floor check lets 1.2 of the radius).
TRUNK_SINK_OF_RADIUS = 0.5

# The soil's height in the grow's frame: the seedling starts on it (the piece frame's y = 0, which
# the uniform scale to the asked height keeps). A shoot cannot grow into the ground, so it is an
# obstacle to every shoot, limb and leaf above it (`buds`); only the trunk's foot and the roots go
# below it. A frame convention, not a tolerance.
SOIL_Y = 0.0

# Stop-short on the finished tree (`buds._stop_short`) looks at what it made again, on the kept
# parts, until a pass cuts nothing more. A tree that is still being cut after this many passes
# is a model failure and raises. Numerics: in practice a second pass confirms the first.
STOP_SHORT_PASSES = 8
# A node whose arc length is within this share of its limb's kept length is kept (float noise from
# the piece cut landing on a node).
KEEP_EPS = 1e-9

# Roots (`buds._roots`): `roots.count` is the one species number. The rest is how a root is drawn,
# not a botanical fact: roots leave the trunk's base point (at the soil) spread evenly round the
# trunk with a jitter, start this far below horizontal, and curve down into the soil; each is a few
# trunk radii long, and thinner than the trunk. The values are those fitted to the redbud
# (angle -20, jitter 18, droop 0.5, 4 segments, radius 0.6 of the trunk, tip 0.35 of the
# base), with the radius at 0.5 of the trunk: the reference trees measured 0.51.
ROOT_BELOW_HORIZONTAL_DEG = 20.0
ROOT_JITTER_DEG = 18.0
ROOT_DROOP = 0.5
ROOT_SEGMENTS = 4
ROOT_LENGTH_OF_RADIUS = (4.0, 6.0)
ROOT_RADIUS_OF_TRUNK = 0.5
ROOT_TIP_OF_BASE = 0.35

# The leaf mesh's look, for the leaves the bud grower makes (`buds._leaves`). Not botanical facts:
# how the folded blade and its petiole ribbon are drawn. They are fitted to the redbud (fold
# 0.16, blade centre (0, 0.42, 0.04), petiole tip 0.07; the petiole ribbon's half-width 0.0044 m
# against a 0.004 m twig tip is 1.1 tip radii), and live here, not in the rules file. The shade list has one shade; a species with several can ask for them later.
LEAF_SHADES = ("Leaf",)
LEAF_FOLD = 0.16
LEAF_BLADE_CENTRE = (0.0, 0.42, 0.04)
LEAF_PETIOLE_HALF_WIDTH_OF_TIP_RADIUS = 1.1
LEAF_PETIOLE_TIP_OF_BLADE = 0.07
# The outward lean of each leaf normal is a render setting, in style.json `dials.leaves`
# (`mesh.leaf_lean`). The grower bakes none of it into a skeleton.

# A relay shoot leaves at this share of the branch angle (a look choice, no cited value): small
# enough that a relay axis zig-zags instead of curling, large enough that it is not a pole.
RELAY_SHARE_OF_ANGLE = 0.25
