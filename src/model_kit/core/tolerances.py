"""The one home for tolerance factors.

Most factors are pure ratios. A check multiplies one by a named length: body checks use the
chain's `effector_radius` (the `*_OF_EFFECTOR` factors), piece and manifest checks use the
thickness base (`units.thickness_base`, the `*_OF_THICKNESS` factors), and a few use the
piece size (`*_OF_PIECE_SIZE`). Touching is no factor: it is the piece's contact length
(`sight.contact_length`). Other factors take the length their name ends in, such as a
branch radius (`*_OF_RADIUS`) or the solver's reference reach (`*_OF_REACH`). The degree limits
(`*_DEG`) are angles. Other kinds are not ratios: `COMPARE_TOL` is the one absolute length, in
model units; a real-world length from a standard ends in `_M`, in metres, and a check multiplies
it by `units_per_metre`; an angle the eye resolves is a `*_ARCMIN`, in minutes of arc, and a
check turns it into a length with the distance a piece is seen from (`View.nearest_m`), that
distance times the tangent of the angle. No other module holds a bare tolerance number as a
module-level constant. `test/test_constant_home.py` enforces that for an upper-case name bound
to a number, or to a tuple, call or sum of numbers: each is registered there with a category, or
it moves here. The test does not see a number in a list, a dict, an array call on a list, a
tuple unpacking, a module-level `if`, a conditional value, or an attribute operand, and it does
not see a number inside a function or a class. A reviewer reads those.
"""

import math

# Contact distance between an effector and a surface, as a share of the effector radius.
# Source: a fixed touch of 0.005 against a reference effector radius of 0.0586.
TOUCH_OF_EFFECTOR = 0.085

# Furthest an effector may sit behind a surface, as a share of the effector radius.
# Source: a fixed behind-max of 0.03 against the same reference radius.
BEHIND_MAX_OF_EFFECTOR = 0.5

# Safety margin subtracted from a chain's reach, as a share of the effector radius. A target
# must lie within reach minus this, so the solver is never asked to fully straighten a limb.
# Source: none; a design call of this kit.
REACH_MARGIN_OF_EFFECTOR = 0.1

# Extra slack a planted fault gets on top of a check's budget, as a multiple of that budget.
# Source: none; a design call of this kit.
PLANT_EXTRA_OF_BUDGET = 2.0

# Two-way calibration: a measure under CALIBRATION_PASS times its budget must pass and one
# over CALIBRATION_FAIL times its budget must fail. Source: none; a design call of this kit.
CALIBRATION_PASS = 0.5
CALIBRATION_FAIL = 2.0

# Trace's off-surface plant: the planted in-plane offset is this multiple of the distance from
# the hold point to the surface edge (never less than that edge plus CALIBRATION_FAIL times the
# trace tolerance). Source: a design call of this kit, 1.5 x the in-plane extent.
OFF_SURFACE_PLANT_OF_EDGE = 1.5

# --- Piece-geometry checks (crossings, take-apart, floaters, topology) ----------------------
# Scenes built with absolute lengths, in model units, do not carry over to other sizes. Touching
# is no such length: every contact rule reads the piece's contact length (`sight.contact_length`,
# one arcminute at `view.nearest_m`), so no fixed touch, end tolerance or contact budget factor
# remains. A decal's lift is that contact length too (`sight.decal_lift`), checked against one
# depth step at the piece's farthest view, so no gear base or lift factor remains. The piece-size
# factors (`WELD_OF_PIECE_SIZE`) follow the piece's extent instead,
# `ZERO_AREA_OF_LONGEST_EDGE_SQUARED` is a shape ratio, and `TAKE_APART_STEPS` is a count.
# The numbers below were measured on a reference scene of five pieces built with the kit.

# The size of the largest reference piece, for the topology weld: the diagonal of the box of all
# a piece's parts, in model units. The five reference pieces measure 0.2912, 0.3171, 0.4982,
# 0.8731 and 1.2458. This is the largest: over it, no reference piece welds wider than 1e-5, the
# distance every one of them passed at. Provenance only, so no check can import it as a size.
_SOURCE_GEAR_PIECE_SIZE = 1.2458

# Topology welds corners within this distance of a seed corner (never chaining through other
# corners), as a share of the piece size: the diagonal of the box of all the piece's parts. The
# weld follows float noise, which grows with a piece's extent, and not the wall thickness (a
# thin-walled limb with a 0.1 mm edge is a real edge, not a double). Source: a reference weld
# distance of 1e-5 (a `remove_doubles(dist=1e-5)` pass) over _SOURCE_GEAR_PIECE_SIZE: 8.0e-6. So
# no reference piece welds wider than the 1e-5 every one of them passed at.
# Measured, a weld sweep at multiples of the factor: the reference scene is clean at
# 0.01 to 1 times and first fails at 2 times (one part gets an edge on 3 faces);
# the chibi bar (`bars/03-chibi/scene`) is clean up to 8 times and fails at 16. At the factor
# the chibi welds at 1.1e-5, and its shortest real edge measured 9.4e-5, 8.6 times
# longer.
WELD_OF_PIECE_SIZE = 1e-5 / _SOURCE_GEAR_PIECE_SIZE

# A face is a zero-area face (a sliver that has collapsed to a line) when its area is under this
# factor times the square of its own longest edge. The test is scale-free: a long thin real face
# and a small pole-fan triangle both keep their shape ratio at any size. Source: no published
# rule (a fixed absolute area such as `f.calc_area() < 1e-9` counts real faces of a small ring
# mesh). Set from measurement, between float noise (float32 epsilon is 1.2e-7) and the
# thinnest real face (a ratio near half the thickness base over the piece size, 8.6e-5 for the
# thinnest reference piece).
# Measured, area over the longest edge squared, smallest face of any part, at the weld above:
# the reference scene 8.0e-5 (a hem strip; the other pieces 4.5e-4 and up), the chibi bar
# 1.1e-3 (arm_L; its 62 pole-fan slivers, area 2.5e-8, are 1.1e-3 and up). The factor is 8 times
# under the thinnest face of the reference scene and 80 times over float32 epsilon.
ZERO_AREA_OF_LONGEST_EDGE_SQUARED = 1e-5

# Take-apart moves each group out along each way it is tried in this many equal
# shares of the piece size, the first at 0 and the last at 1. Source: a design call of
# this kit, "21 shares from 0 to 1".
TAKE_APART_STEPS = 21

# How take-apart's plant boxes a group in: a cage of six slabs around the group's box. The wall
# is this share of the piece size thick, so a group that moves out in any of the 26 ways is
# driven well into a wall whatever the sweep's step is. The gap between the group's box and the
# cage is this share of the group box's least extent, so the cage is clean at rest. A plant's
# size, which no check compares anything with. Source: none; both are sized from the piece's own
# geometry the way `SINK_PLANT_OF_CROSS` is. A decision.
CAGE_WALL_OF_PIECE = 1.0
CAGE_GAP_OF_GROUP = 0.1

# The cone of `geom.local_thickness`: rays at these angles from the inward normal (the 120 degree
# cone of the cited paper), `CONE_AROUND` rays on each ring but the first (the normal itself). The
# thickness it reads sets how deep two parts may cross (crossings judges depth by the thinner
# part). Source: Shapira, Shamir and Cohen-Or 2008, "Consistent mesh partitioning and
# skeletonisation using the shape diameter function" (the shape diameter, `geom.local_thickness`).
CONE_DEG = (0.0, 30.0, 60.0)
CONE_AROUND = 8

# A ray's weight falls as 1 / (1 + angle / CONE_WEIGHT_DEG): the rays nearest the normal count most.
CONE_WEIGHT_DEG = 20.0

# --- Manifest checks (frames, sides, mirror-pair) --------------------------------------------
# These read the pack's points and directions against the geometry and the clips. The distance a
# point may sit from its surface is the piece's contact length (`sight.contact_length`), not a
# factor. No published rule measured the angle factors, so each is fixed by what a reference
# scene built with the kit shows. A value keeps headroom over the worst clean one; the holds that
# fail a rule on that scene are findings, not reasons to widen a factor.

# How far a hold's `off` may turn from the outward normal at the nearest surface point, in
# degrees. Source: a design call of this kit (a tilted-`off` case).
# Measured on the reference scene, over its 10 holds (a point inside a round part counts as near
# to every face of the ring, as `geom.face_distances` gives them): 0.0 for three holds, 7.2 for
# one, 15.55 for a cymbal's own tilt, then 129.8 and 170.1: those two have a face of another part
# nearer than the one the hand meets (a hand under a cymbal; a pedal link below its footboard),
# which is the "hand passes under a thin plate" case, not a tilt. Worst clean value: 15.55, 52% of
# the limit. The calibration plant tilts to CALIBRATION_FAIL times the limit, 60, and
# CALIBRATION_PASS times it, 15, passes.
OFF_ANGLE_DEG = 30.0

# How far a user's facing may turn from the direction to the holds they use, in degrees. Source:
# a design call of this kit, the use-side rule ("within an angle").
# Measured on the reference scene, over the 3 placed pieces that have a use station with holds
# (the angle between the rig root's horizontal line to the placed holds' centroid and the placed
# facing): 14.7, 27.0 and 30.8 (two pieces whose players stand either side of a central user, so
# the centroid is off the facing). Worst clean value: 30.8, 68% of the limit. A half-turn plant
# gives 180.
USE_ANGLE_DEG = 45.0

# How far past the station's centre plane a hold or a hand may sit on the wrong side before the
# `sides` check fails it, in metres: half a shoulder width. A hand works across the body's midline
# up to the other shoulder, so only a hold past that shoulder is on the wrong side. Source: Gordon,
# C. C. et al. (2014), 2012 Anthropometric Survey of U.S. Army Personnel: Methods and Summary
# Statistics, Technical Report NATICK/TR-15/007 (DTIC ADA611869), dimension (8) Biacromial
# Breadth, percentile table, page 59: women, 5th percentile, 33.50 cm (n = 1986). Biacromial
# breadth is measured between the two acromion landmarks (the bony shoulder points). The 5th
# percentile woman is chosen so the band is the narrowest real shoulder, and the check fails
# the most. Half of 0.335 m. The check scales it by `units_per_metre`.
HALF_SHOULDER_M = 0.1675

# --- Assembly checks (receiver, connectors, export hygiene) ------------------------------------
# These measure how pieces meet: a guest part on a host mount, a cable end on a socket, a node of
# the packed file on its slot. Receiver and connectors judge an end by the piece's contact length;
# export hygiene uses NODE_OF_PIECE_SIZE, since a node position is exact.

# How far a node of the packed .glb may sit from the place the pack gives it (export hygiene: a
# piece root on its grid slot, at its layout position, or at the origin), as a share of the
# piece's size: the diagonal of the box of all its parts. A position is exact up to float32 noise,
# which grows with the piece's extent, so it is the weld's tolerance and not a share of the wall
# thickness: the thickness limit above is 0.478 on the held pack, over three times a drumstick's
# tip length. Set equal to WELD_OF_PIECE_SIZE (1e-5 over the largest reference piece).
# Measured on the two bar scenes' real node tables (the redbud's and the chibi's .glb,
# as the exporter wrote them): every root is exactly on its slot, a clean error of 0.
NODE_OF_PIECE_SIZE = WELD_OF_PIECE_SIZE

# A node's translation is stored as float32, so a root far from the origin carries rounding the
# piece-size limit above does not: one rounding moves each axis by up to FLOAT32_EPS / 2 times
# its size. The node limit adds NODE_FLOAT32_ULPS x FLOAT32_EPS x |position|, so it stays as
# tight as above at the origin (a carried prop) and grows only with the distance. Source: the IEEE
# 754 single-precision machine epsilon, and 4 ulps for the exporter's few matrix steps (on the
# smallest reference piece one rounding alone reaches the bare limit at |p| = 39).
FLOAT32_EPS = 2.0**-23
NODE_FLOAT32_ULPS = 4

# --- Branching checks (core/checks/branching) ------------------------------------------------
# Limb tolerances scale with the limb radius where they measure; the floor and skeleton-match
# scale with the piece's tube thickness base (its listed rows that have a tube part).

# A limb's base may sit this share of its parent's radius from the parent's axis, and a leaf's
# node this share of its host's radius from the host's axis; a leaf's base may sit its petiole
# plus this share of the host's radius from its node. Source: a limb's base lies inside the
# parent's tube.
SEAT_OF_PARENT_RADIUS = 1.0

# A limb's base radius may be this share of its parent's radius there. Source: a base radius over
# 1.05 times the parent's radius there is a fault.
SEAT_CHILD_OF_PARENT_RADIUS = 1.05

# Where two limbs that grow from one place stop being one crotch: a seat zone ends where the
# gap first exceeds this share of the thinner radius. Source: the kit file's
# `limbs.clearance_of_radius` (kinds/branching/botany.rules.json, row `limbs.clearance`, tagged
# model: "a modelling margin, not measured"; the principle, that solid wood cannot pass through
# solid wood, is a fact), so the grower's stop-short rule and the checks draw the crotch at one
# line. No measurement sets it: a child and its parent, and two siblings from one place, are
# read only past the zone this line ends, so their measured gaps cannot test it. Moving it alone
# would split the grower and the checks.
# It sits above CALIBRATION_FAIL x NEAR_OF_RADIUS, so a crotch that has opened never reads as a
# near miss.
SEAT_ZONE_OPENS_OF_RADIUS = 1.2

# The overlap, as a share of the thinner radius, past which a gap is a crossing and not a near
# miss. No source and no measured basis: crossing-organic fails g < -CROSS x r and
# near-miss-organic fails -CROSS x r <= g <= NEAR_OF_RADIUS x r (one comparison,
# `branching/_pairs.classify_gap`), so together they fail every gap up
# to NEAR_OF_RADIUS x r whatever this is. It moves no pass-or-fail line, only which check reports
# the gap and how deep the crossing plant goes (CALIBRATION_FAIL x 0.5 = one thinner radius, which
# still fits inside the thinner limb), so no measurement can set it and moving it would be taste.
# Measured on the four trees (thinner radius of 5 cm or more, the
# reliable laser-scan radius of Demol et al. 2022): one honest pair overlaps past its zones,
# -0.229 r (FEXC_14 order1.2 x trunk.0, where the trunk's radius rises along the crotch), and no
# value of this factor would pass it.
CROSS_OF_RADIUS = 0.5

# Below this share of the thinner radius, a gap between two limbs fails as touching. No source:
# the value is a call of this kit that a gap this small reads as touching at the page camera
# (taste). The measured trees only bound it from above: on the four trees, the smallest honest
# gap of a pair that is not parent and child or siblings is +2.397 r (FSYL_15, trunk.0 against
# itself; thinner radius of 5 cm or more, Demol et al. 2022). A child and its parent, and two
# siblings from one place, are read only past their zone, which ends where the gap opens past
# SEAT_ZONE_OPENS_OF_RADIUS or stops growing.
# It must stay under SEAT_ZONE_OPENS_OF_RADIUS / CALIBRATION_FAIL = 0.6. Moving it inside that
# bound would be taste.
NEAR_OF_RADIUS = 0.5

# A limb that is neither rooted nor buried, or a leaf, may dip this share of the tube thickness
# base below the soil. Source: a design call of this kit (one tube thickness).
FLOOR_OF_THICKNESS = 1.0

# A rooted limb's base may sit this share of its own base radius below the soil. Source: a
# sweep: honest trunk bases sit at most 0.568 of their base radius down (the
# redbud's trunk.base_depth_m), under half of 1.2. The tube
# thickness tolerance was tried here, but every honest trunk sits 3.3 tolerances down.
ROOTED_DEPTH_OF_RADIUS = 1.2

# A buried limb may be seated this share of its rooted parent's radius there above the soil.
# Source: the same sweep: honest roots seat at 1.165 to 1.721 of the trunk's radius up, under
# half of 3.5. The floor tolerance was tried here, but every honest root seats 5 to 12
# tolerances up.
BURIED_SEAT_OF_RADIUS = 3.5

# A shipped vertex or triangle corner may sit this share of the tube thickness base from the
# skeleton's own mesh, and back. The export reproduces the skeleton's triangles up to float32
# and welding, so the tolerance is a tenth of the thinnest limb radius (0.0045 u against a base
# of 0.009 u for the redbud): a limb shrunk to a hairline or swollen fails. Source: a
# sweep (the earlier 0.5, trace's factor, let 93% of limbs halve or double). The honest
# worst: the float32 round trip of `grown_parts` moves a corner at most 1.2e-7 u over the sweep,
# and the adapter's weld (`WELD_DECIMALS` 6) at most 5e-7 u, both far under half the 4.5e-4 u.
SKELETON_MATCH_OF_THICKNESS = 0.05

# skeleton-match's plant moves a vertex whose nearest other vertex lies over this many
# tolerances away, so the move cannot land near one. Source: a design call of this kit, "over 3 x the
# tolerance away".
PLANT_CLEAR_OF_TOL = 3.0

# --- Solver weights (core/solve.py) ---------------------------------------------------------
# The solver's cost has no units: the miss is divided by the chain's reach (`rig.reach`, model
# units), and the other terms are in radians. The weights reproduce the optimum of a reference
# solver, on a reference rig: that solver's cost (its miss in model units) divided
# by SOLVER_REFERENCE_REACH squared.
# Source: `rig.reach` of the reference rig's right arm chain (arm, forearm and hand; hand
# effector), measured from the first bone's head: 0.21267 model units (0.2126661...), in a
# style of 0.5 units per metre. The reach does not count the `body` to `arm` link.
SOLVER_REFERENCE_REACH = 0.2127

# Pull toward the last key (radians squared), so a solve does not flip. The reference
# solver's weight: 0.0004.
W_STAY = 0.0004 / SOLVER_REFERENCE_REACH**2

# A joint angle past its chain limit (radians squared). The reference solver's weight: 1.
W_LIMIT = 1.0 / SOLVER_REFERENCE_REACH**2

# The effector's turn away from `level` (squared matrix entries): W_LEVEL = level_weight times
# this. The level weight is LEVEL_WEIGHT by default and 1.0 for a held stick, as in the
# reference solver.
W_LEVEL_PER_WEIGHT = 1.0 / SOLVER_REFERENCE_REACH**2
LEVEL_WEIGHT = 0.02

# --- Solver terms and their factors (core/solve.py) -----------------------------------------
# Every number the reference solver used is named here.
# - A weight is the reference value over SOLVER_REFERENCE_REACH squared.
# - A length is a factor: the reference solver's length over a reference length, either
#   SOLVER_REFERENCE_REACH or SOLVER_REFERENCE_EFFECTOR_RADIUS. The solver multiplies it by the
#   chain's own reach or `effector_radius`, so it stays scale-free and, on the reference rig,
#   gives the reference length back.
# - A count or a share is kept as it was.
# `test_tolerances.py` pins each derived value to the original number.

# The reference rig's effector radius: the mitt's, in model units.
SOLVER_REFERENCE_EFFECTOR_RADIUS = 0.0586

# Keep the elbow, wrist and effector centre out of an avoid shape. The term is this weight times
# the squared shortfall in clearance (a length), so the weight is per squared model unit and the
# term has no units. Form: `25.0 * max(0.0, avoid - clearance) ** 2`.
W_AVOID = 25.0 / SOLVER_REFERENCE_REACH**2

# Keep every skin point of the effector on a plane's front. Form:
# `4.0 * max(0.0, m - n.dot(w - p0)) ** 2`.
W_PLANE = 4.0 / SOLVER_REFERENCE_REACH**2

# A skin point inside a solid, or nearer than its margin. Form: `100.0 * pen * pen` (at a weight of
# 25 the solver traded 5 mm of elbow in the ribs for 2.5 cm of reach).
# 2.5 cm of reach).
W_SOLID = 100.0 / SOLVER_REFERENCE_REACH**2

# A solid's own back points inside an arm part's surface. Form:
# `25.0 * (dist + max(mg, 0.0)) ** 2`.
W_SOLID_BACK = 25.0 / SOLVER_REFERENCE_REACH**2

# A solid's box, grown by this, is the prefilter: skin points outside it skip the surface
# search. Factor of SOLVER_REFERENCE_REACH. Form: `lo.x - 0.07 < q.x < hi.x + 0.07`.
SOLID_PREFILTER_OF_REACH = 0.07 / SOLVER_REFERENCE_REACH

# How far an arm part's skin keeps from a body or head solid. Factor of the effector radius.
# Source: a hand margin of 0.003 for body and head solids.
SOLID_MARGIN_OF_EFFECTOR = 0.003 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# How far an arm part's skin keeps from a piece of gear. Factor of the effector radius.
# Source: a hand margin of 0.006 for gear solids (wider than
# the gate's skin: keys only, the frames between may dip).
GEAR_MARGIN_OF_EFFECTOR = 0.006 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# An arm part's surface box, grown by this, is the prefilter for a solid's back points. Factor
# of the effector radius. Form: the box minus 0.003 below and plus 0.003 above.
BACK_PREFILTER_OF_EFFECTOR = 0.003 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# A hand on a press zone sets its margin to minus this share of the zone's budget. Form:
# `min(margin, -0.5 * budget)`.
PRESS_MARGIN_SHARE = 0.5

# The radii the avoid shapes allow for the elbow, the wrist and the effector centre. Factors
# of the effector radius. Source: the radii 0.05 (forearm), 0.055 (hand) and 0.059 (mid). The
# 0.059 is the mitt's radius rounded, and
# the factor keeps it as written.
AVOID_ELBOW_OF_EFFECTOR = 0.05 / SOLVER_REFERENCE_EFFECTOR_RADIUS
AVOID_WRIST_OF_EFFECTOR = 0.055 / SOLVER_REFERENCE_EFFECTOR_RADIUS
AVOID_CENTRE_OF_EFFECTOR = 0.059 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# An ellipsoid's distance in radii, times this, is a length: how far a point is from the
# shape. Factor of SOLVER_REFERENCE_REACH. Form: `(k - 1.0) * 0.30 - r`.
AVOID_HEAD_SCALE_OF_REACH = 0.30 / SOLVER_REFERENCE_REACH

# Skin points kept on an effector chunk, picked furthest first so every bulge is in. Source:
# a pick of 90 points.
SKIN_SAMPLES_PER_CHUNK = 90

# Skin points the solid phase tests on each arm part, taken from the part's sampled skin.
# Source: 36 points on the hand, 48 on the forearm and 16 on the arm.
SOLID_SKIN_POINTS_HAND = 36
SOLID_SKIN_POINTS_FOREARM = 48
SOLID_SKIN_POINTS_ARM = 16

# Points kept on each part of a solid, so a small screw counts as much as a big shell. Source:
# every `max(1, n // 40)`th vertex.
SOLID_SAMPLES_PER_PART = 40

# How far a skin point's blend weights may sum from 1 and still read as one point (float noise
# in weights that were normalised, not a measuring tolerance).
SKIN_WEIGHT_SLACK = 1e-6

# How far a bone's 3x3 (or `rig.world`'s) may be from a rotation, as the largest entry of
# R R^T - I. Avoid shapes, solids and chunks read a bone frame as world lengths, so a scale or
# shear there would bend the gaps. Float32 rest matrices from Blender sit near 1e-7; 1e-5 is
# a hundred times that and far below any real scale (a scale of 1.00001 is 2e-5).
BONE_RIGID_SLACK = 1e-5

# A tool whose tip misses by more than this is moved by the miss and re-solved, up to
# TIP_RETRIES times. Factor of the effector radius. Source: stop when the miss is under 0.004,
# in at most 3 rounds. The two retry thresholds have
# different bases on purpose, as in the reference solver: this one is a precision at the fist
# (the effector radius), the next is a miss of the whole arm (the reach).
TIP_RETRY_MISS_OF_EFFECTOR = 0.004 / SOLVER_REFERENCE_EFFECTOR_RADIUS
TIP_RETRIES = 3

# A quick solve (one start, from the last key) that leaves a tool this far from its target is
# solved again from every start. Factor of SOLVER_REFERENCE_REACH. Source: a retry when a tool's
# error is over 0.01.
QUICK_RETRY_MISS_OF_REACH = 0.01 / SOLVER_REFERENCE_REACH

# How far from 1 a unit vector's length may be and still read as one (a stand's facing, a
# decal's cast direction).
UNIT_LENGTH_SLACK = 1e-6

# --- Pose overlap (core/checks/overlap.py) ----------------------------------------------------
# A pose-contact gate with absolute depths, in model units, is set against a mitt
# (SOLVER_REFERENCE_EFFECTOR_RADIUS). Each factor is that depth over the
# mitt radius, so the check multiplies it by the rig's own effector radius and a different-sized
# rig keeps the proportions. The values are contact budgets fitted to a reference rig.

# The deepest a body vertex or a prop vertex may sit inside another surface anywhere, as a share
# of the effector radius. Source: a depth of 0.002 (facets and float noise, no more).
SKIN_OF_EFFECTOR = 0.002 / SOLVER_REFERENCE_EFFECTOR_RADIUS
# Clip hygiene's stand rule reads it too: a loaded sole touches the
# floor, and the only slack is mesh noise, this skin budget of the support's own radius. The
# planted-foot band, FLOOR_OF_EFFECTOR, is 5 times wider and would miss a 0.006-unit lift
# (about 8 mm at 0.747 units per metre).

# The deepest a hand may sink into the hold it presses, or a stick tip into a head, as a share of
# the effector radius. Source: a depth of 0.010 (a hand on its own hold: a fist on a
# guitar face, fingers on a key, a hand on a jog wheel).
PRESS_OF_EFFECTOR = 0.010 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# The deepest the body may sink into a seat's surface parts, as a share of the effector radius.
# Source: a depth of 0.025 (the seat and thighs on a cushion, a soft seat).
SEAT_OF_EFFECTOR = 0.025 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# How far past a hold's effector radius the solver may leave the effector, as a share of the
# effector radius: the worst honest solve miss. A grip zone is (1 + this) times the radius round
# the hold point. Source: the worst grip error, 0.024 units, and the worst fret error, 0.036
# units, measured on a reference scene. This is the worse
# one, 0.036, over the mitt radius: the zone it gives (0.0946) lies between the grip zone radius
# (0.09) and the fret zone radius (0.10), each "the mitt (0.059) plus the solver's worst
# error". The solver-error check reads it too.
SOLVE_ERR_OF_EFFECTOR = 0.036 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# --- Clip hygiene (core/checks/clip_hygiene.py) -----------------------------------------------
# Three clip tolerances were set in a reference scene's lengths (a foot tolerance, a slide
# tolerance and a pop tolerance). Each factor is that length over the mitt radius
# (SOLVER_REFERENCE_EFFECTOR_RADIUS), as the other depths are, so the check multiplies it by the
# rig's own effector radius and a different-sized rig keeps the proportion. No sweep of the values
# was run: a finding is a finding, not a reason to widen one.

# How far under the floor (y 0) a body vertex may sit, and the height within which a foot counts as
# planted, as a share of the effector radius. Source: a foot tolerance of 0.01 (for both the
# below-the-floor rule and the planted-foot test).
FLOOR_OF_EFFECTOR = 0.01 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# How far a planted foot may move across the floor between two samples, as a share of the
# effector radius. Source: a slide tolerance of 0.02.
SLIDE_OF_EFFECTOR = 0.02 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# How far a loop's last sample may sit from its start sample, at any body vertex, as a share of
# the effector radius. Source: a pop tolerance of 0.01.
POP_OF_EFFECTOR = 0.01 / SOLVER_REFERENCE_EFFECTOR_RADIUS

# --- Coplanar (core/checks/coplanar.py) -------------------------------------------------------
# Two faces share a plane only if their unit normals agree to this dot product. Without it the
# plane test lets a face that crosses another steeply count as "the same plane": the overlap of
# the two is a sliver along the line where they cross, and every corner of a sliver lies within
# the depth gap of both planes (bar 05's two arm tubes cross at 10 degrees, along a 0.24 long
# seam, and read a hit). Source: a pairwise z-fighting test that skips a pair whose normals'
# dot product is under 0.9999 ("the real planes must agree"), about 0.8 degrees, a value that 60
# reference gate scenes passed with. A two-way plane test alone did
# not keep that seam quiet (the sliver's corners are near both planes), which is why the angle
# stays a limit here.
COPLANAR_NORMAL_COS = 0.9999

# --- Export compare (core/export_compare.py) -------------------------------------------------
# How far two piece-frame triangle arrays may differ, per coordinate, and still be the same
# part. Absolute, in model units: it is not a share of a piece length, because the arrays hold
# float32 Blender coordinates carried through a float64 matrix, and that rounding error grows
# with how far a coordinate sits from zero, not with how big its piece is. A rebuild of the
# same script gives the same float32 values, so the error is only the matrix arithmetic.
# Source: the float32 spacing at 8 model units is 2**-20, about 9.5e-7; 1e-6 sits just above
# that rounding. The 8 is an assumed stage extent, not a measured one: if a piece
# reaches further from its root, raise this to the float32 spacing at that coordinate.
# It is also the float allowance on a surface-to-surface distance: `inset_seat` and `floaters`
# add it to the contact length, so a decal built to stand off its host by exactly that length
# reads as touching. There it bounds the same float32 rounding of a distance between two saved
# surfaces (measured up to 2.5e-7 on bar 03's eyes, 4e-8 on bar 05's logo).
COMPARE_TOL = 1e-6

# --- Stress poses (core/checks/stress_poses.py) -----------------------------------------------
# How far a posed face's normal may turn from the rest normal carried by the face's main bone
# before the face counts as turned inside out. Past a right angle the posed normal points against
# the carried one (their dot product is negative), so the face looks the other way. Source:
# geometry (a posed face normal that turns more than 90 degrees from
# its rest normal carried by the face's main bone). It is a definition and not a measured limit.
INVERTED_FACE_DEG = 90.0

# --- Layout clearance (core/checks/clearance.py) --------------------------------------------------
# The least gap two unjoined placed pieces keep, in metres: a finger's width. A check multiplies
# it by `units_per_metre`. Source: ISO 13854:2017, the minimum gap for a finger, 25 mm. The
# standard's sample pages show its scope and Table 1's first rows (body 500, head 300, leg 180 mm)
# and stop before the finger row (https://cdn.standards.iteh.ai/samples/66459/
# 46ce3b47790e4526802912dbbd921354/ISO-13854-2017.pdf); the finger row, "Finger >= 25", is from BG
# ETEM's EN ISO 13854 sheet (https://medien.bgetem.de/medienportal/artikel/UzA0NC1F/@@download/
# download, p. 1). The standard's scope is crushing hazards between moving machine parts, and a
# layout has none: this kit borrows only the body-part size, not the crushing scope.
MIN_GAP_M = 0.025

# --- Near-crease (core/edges.py, core/checks/near_crease.py) ----------------------------------
# The widest rounding band an edge may carry, as a share of the style's crease angle. The band is
# the first-order bound on how far the shipped file's position rounding can move an edge's turn
# (the dihedral angle's gradient, Bridson, Marino and Fedkiw 2003, Section 4,
# https://www.cs.ubc.ca/~rbridson/docs/cloth2003.pdf). The gradient at a corner is one over the
# height of the corner's face above the edge, so a band this wide belongs to a face only about ten
# position steps high: a sliver, not a rounded face. Topology's zero-area rule owns
# slivers, so the check reports such an edge as a sliver and not as a near crease. Half is a
# stated design share: no study sets it. It was fixed before any bar was measured.
CREASE_BAND_CAP_OF_ANGLE = 0.5

# --- Bend quality (core/checks/bend_quality.py) -----------------------------------------------
# How much a slice's perimeter (girth) may change against rest, as a share of the rest perimeter:
# the largest girth change the cited scan studies measured on real limbs through bending (no study of
# slice AREA was found, only of girth).
#   Ringenbach, Yu and Zhang 2024 (Textile Research Journal 94(23-24), n = 50), abstract: bending
#     the knee from 0 to 90 degrees "increased the knee circumference by 6.4%"; thigh and calf
#     belly girth "showed no significant changes".
#     https://research.polyu.edu.hk/en/publications/analysis-of-knee-bending-motion-through-three-dimensional-scannin/
#   Klepser and Morlock 2020 (CDATP 1(1):30-38, n = 6), the full text read: upper arm,
#     straight against elbow flexed 90 degrees: girth differences of -1.3 cm to +2.8 cm, "average
#     deviation 3.61%"; thigh, leg flex: -4.1 cm to +1.8 cm, average -2.7%; squat: -9.5 cm to
#     +6.5 cm, average 0%. https://cdatp.publia.org/cdatp/article/download/6/7/131
# The arithmetic. A share needs a girth to divide by. Ringenbach gives the share itself, 6.4%.
# Klepser gives shares only as averages (3.61%, 2.7%, 0%), and its extremes only in centimetres
# with no stated girth of the same limb, so no share can be made from them without inventing a
# girth. The largest share the studies themselves state is max(6.4, 3.61, 2.7, 0) = 6.4%, and
# 6.4 / 100 = 0.064. Two cautions, kept here so the number is not trusted past what it is: the
# cm extremes (+2.8, -9.5 cm) may be larger shares than 6.4% on a limb of ordinary girth, and
# Klepser's subjects held 1 kg, so muscle was loaded. Girth is a proxy for the slice (the study
# itself notes "width and depths of the body parts alter" where girth holds).
BEND_GIRTH_CHANGE = 0.064

# The least a slice's area may be, as a share of its rest area. A STATED DESIGN CALL, NOT a
# measurement: no study measures limb cross-section area against joint angle. It is a floor on
# how far a ring may collapse, nothing more.
BEND_AREA_FLOOR = 0.5

# A skin weight at or under this is left out by the bind step (`blender/bind.py` imports it), and
# one over 1 minus this is wholly on one side. Source: the chibi build, which adds a vertex to
# a group only when its weight is above 0.001. It decides where the blend band of a joint starts
# and ends, never what passes.
BIND_WEIGHT_FLOOR = 0.001

# --- Detail at the declared viewing distance (core/checks/detail.py, markings_flush.py) -----
# The smallest angle the eye resolves, in minutes of arc: the minimum angle of resolution of
# normal sight, acuity 1.0 (20/20). A facet's gap from the curve it stands for, or a part's
# stand-off from its host, shows when it subtends more than this at the nearest viewing distance.
# At one metre the angle is about 0.29 mm. A design call of this kit sets it; the best-case limit for
# spotting a kink (hyperacuity, 2 to 5 arcseconds) was the other option, and the kit's other
# limits use a normal person, not a best case. Source: EyeWiki ("Visual Acuity
# Testing in Adults", https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, a secondary
# source that cites the International Council of Ophthalmology's 1984 Visual Acuity Measurement
# Standard for the decimal acuity scale).
EYE_RESOLUTION_ARCMIN = 1.0

# --- Bearing (core/checks/bearing.py) ---------------------------------------------------------
# How far a fastener must sit inside the part it fastens, as a share of its own shank diameter.
# Source: Nord-Lock, "The Experts: Thread engagement in a tapped hole" (2015): "approximately 1
# times the nominal diameter in steel and 2 times the nominal diameter in aluminum", a stated rule
# of thumb that cites no origin
# (https://www.nord-lock.com/insights/bolting-tips/2015/the-experts-thread-engagement-in-a-tapped-hole/).
# The physics is the pull-out load in NASA RP-1228 (Barrett, Fastener Design Manual, 1990,
# "Calculating Pullout Load for Threaded Hole", where the load grows with the length of thread
# engagement; https://ntrs.nasa.gov/api/citations/19900009424/downloads/19900009424.pdf). The kit
# uses the steel multiple for every host and has no material field (a design call of this kit).
ENGAGE_OF_DIAMETER = 1.0

# How many samples lie along a fastener's axis when the check measures how much of it is inside
# its host (inside the solid, or ringed by `ENGAGE_RAYS` rays, for each). A count, not a
# tolerance: the engagement is read to 1/64 of the fastener's length. Source: the take-apart
# count above, a design choice for resolution.
ENGAGE_SAMPLES = 64

# How many rays leave an axis point, spread round the axis at right angles to it, when the check
# asks whether the host surrounds that point. A count, not a tolerance: the point is surrounded
# when every ray meets the host, so a bore closed all round counts and an open side does not.
# Eight is every 45 degrees, a design choice for resolution.
ENGAGE_RAYS = 8

# A face looks along an axis more than across it past this cosine: the cosine of 45 degrees. The
# bearing check reads it three ways: a patch whose mean normal is under it faces neither up nor
# down, a face whose normal is under it along a shank's axis is on the shank's wall, and a face
# over it is square to a seat's normal. A definition of "looks one way more than another" (an
# equal split of the right angle), not a measured limit.
BEARING_WALL_COSINE = math.sqrt(0.5)


# --- Crossing depth by part thickness (core/checks/crossings.py, take_apart.py, bearing.py) ----
# How far two parts may share space, as a share of the thinner part's local thickness
# (`geom.thinner_thickness`). Bearing allows a part to sit this deep into its host and keep its
# contact, so a part crossings accepts is a part bearing still finds seated. Shared
# space is the shortest move that separates them (`geom.shortest_exit`, the penetration depth of
# collision detection, over 26 fixed directions); thickness is the shape diameter at the
# middle of that region (`geom.local_thickness`). Past it the one part is embedded and not
# resting, and an embedded part is a join the builder declares (an allowlist line with a reason).
# Source: a design call of this kit: whether a dent looks wrong depends on the size of the thing
# dented, so "half the thinner part": a part sunk more than halfway into
# another is embedded. It replaces the old pack-wide 0.5 mm limit on the deepest cut between two
# faces, which a part with narrow faces could never fail and which could not see a rod through a
# thin cap. The shape diameter is Shapira, Shamir and Cohen-Or (2008). A decision, not a
# measured value.
CROSS_OF_PART_THICKNESS = 0.5

# How deep bearing's sink plant (rule 5, "sinks into") pushes a part into its host, at most, as a
# share of the depth crossings fails at (`CROSS_OF_PART_THICKNESS` times the thinner of the two
# parts' least extents). The shape diameter reads at least a part's least extent (slanted rays
# run longer), so a push under this share of half that extent stays under half the thinner part's
# thickness, where the contact is still found. The depth between the contact length and that
# limit can be narrow, so the share is high; a tenth of the limit is the margin. A plant's size,
# which no check compares anything with. Source: none; the plant is sized from its own geometry
# the way `OFF_SURFACE_PLANT_OF_EDGE` is. A decision.
SINK_PLANT_OF_CROSS = 0.9

# How far apart the samples of a part's surface sit when the space it shares with another part is
# measured, as a share of the part's thickness at its centre. Source: a design call of this kit
# ("spaced at most a quarter of the sampled part's local thickness"). A resolution and not a
# limit: a finer step can only find a deeper point, and the limit is set by
# CROSS_OF_PART_THICKNESS alone.
CROSS_SAMPLE_OF_THICKNESS = 0.25

# The most points `geom.surface_samples` may make for one surface: past it the lattice is never
# built and `errors.TooManySamples` is raised. A memory bound, not a limit any piece is judged by:
# a lattice point costs about 82 bytes at its peak (measured on the lamp bar's shade, 1.19e6
# points in 97 MB) and the count compared is an upper bound about twice the points made, so the
# cap keeps one surface's points near 400 MB, a twentieth of an 8 GB laptop. Source: a design call
# of this kit. Without a cap, a part whose thickness at its centre reads 2e-5 (a shell sunk
# 0.018 mm inside another shell of the same part) asks for about 2e10 points, over 100 GB.
# The bars' largest surface asks for at most 2.8e6 (the lamp's shade).
MAX_SURFACE_SAMPLES = 10_000_000


# --- plant checks ------------------------------------------------------------------------------

# How far past its cited range's upper end the reference-range plants push a tree: the crown is
# scaled until its spread is this share of the range's upper end, and the leaves are cut away
# below this share of the crown-base range's upper end. Source: a design call of this kit, a plant
# at "1.1 x the range's upper end", a margin of ten per cent that the check sees whatever
# the start.
RANGE_PLANT_OVER = 1.1

# Taper and leaf-angle (core/checks/branching/taper.py, leaf_angle.py).
# The forms and numbers come from a reference sweep of the four measured trees.
# Leonardo's rule has no constant here: no stated fit puts the four measured trees inside Eloy's
# band, so it is taste-only (see `kinds/branching/rules.md`).

# The smallest limb radius, in real metres, that a measured (lidar) tree's radius is trusted at.
# Source: Demol et al. 2022, Silva Fennica 56(1) 10550 (doi:10.14214/sf.10550): leaf-off ash QSM
# diameters under 10 cm are overestimated, by 4% above 10 cm and by 73% under 2.5 cm. It is chosen
# from that paper before any slope was read. `taper` applies it to measured skeletons only; a
# grown skeleton has no lidar error, so every limb of it is judged.
RELIABLE_RADIUS_M = 0.05

# The fewest points a limb needs for `taper` to fit a radius slope. Source: a form
# fixed before the run (a line through three points or fewer says little about a trend).
TAPER_MIN_POINTS = 4

# How far the taper plant makes a limb thicken toward its tip: the radius at the tip is this much
# over the radius at the base. Source: the plant's test,
# `test_taper_catches_a_thickening_limb` ("a limb whose radius grows 30% toward its tip").
TAPER_PLANT_RISE = 0.3

# The range of the mean blade angle (degrees from horizontal; 0 flat, 90 vertical) a whole tree may
# have: the lowest and highest species mean in the leaf-angle dataset, each widened by twice its
# own standard error (2 SD / sqrt(n)). Source: Chianucci et al. 2018 (Mendeley Data
# 10.17632/4rmc7r8zvy.2, `dataset_full.csv`), 138 species. Low end: Carpinus betulus, mean 12.90,
# SD 9.69, n 100, so 12.90 - 1.94 = 10.96. High end: Liquidambar styraciflua, mean 69.43, SD 12.58,
# n 91, so 69.43 + 2.64 = 72.06. The widening is the species' own measurement error: without it the
# sweetgum, whose beta is fitted to that very 69.43, would fail on about half its seeds.
LEAF_ANGLE_MEAN_MIN_DEG = 10.96
LEAF_ANGLE_MEAN_MAX_DEG = 72.06

# The least spread (standard deviation, degrees) of blade angles a crown third may have. Source:
# the same dataset: the smallest SD over all 262 species-sector-date sets, whole canopy included, is
# 9.69 (Carpinus betulus, whole canopy, n 100). The smallest sector-only SD is 9.80 (Betula
# alleghaniensis, middle canopy, n 75), but a crown's variance is at least the mean of its thirds'
# variances, so a measured species with only a whole-canopy set has a third at or under its whole
# SD; 9.80 would fail a measured species. 9.69 is the highest floor no measured set contradicts.
LEAF_ANGLE_SD_MIN_DEG = 9.69

# The fewest leaves a mean or a crown third needs before `leaf-angle` judges it. Source: the same
# dataset's sector sets hold 26 to 133 leaves (median 90); the band and the floor were measured on
# sets of that size, so a smaller set is not judged.
LEAF_ANGLE_MIN_LEAVES = 26

# The blade angle the steep plant gives every leaf. Source: a design call of this kit (a
# species file whose blade-angle beta has a mean of 85 degrees); it sits above the band's top.
LEAF_ANGLE_PLANT_STEEP_DEG = 85.0

# --- Manifest loaders (core/manifest/rig.py) -------------------------------------------------
# A rest pose's head-to-tail distance and its bone's `length` are one number written twice: the
# loader refuses a rig whose two differ by more than this, in model units. Float rounding of the
# file's own numbers, so it sits far under any real bone.
REST_LENGTH_TOLERANCE = 1e-9

# --- Light bake (core/checks/seam_jump.py, core/checks/lightmap_density.py) --------------------
# How many standard deviations a step across a lightmap seam must exceed before `seam-jump`
# reports it. The deviation adds the two sides' shipped noise (as extrapolated to the edge) and
# each side's recorded resolution `error` in quadrature, `sqrt(5 (ns_a^2 + ns_b^2) + e_a^2 +
# e_b^2)`: the picture cannot resolve a step finer than its own resolution error (derived in
# `seam_jump.py`). Source: the general three-sigma rule of a normal error (99.7% of honest noise
# lies inside it).
SEAM_JUMP_SIGMAS = 3.0

# The share of the bake's `noise` squared that a 2 x 2 average leaves in the grown-back picture of
# pure noise: 1 - 1/4. `lightmap-density` takes it off the error before it judges what is left.
# Source: the variance of a mean of four independent texels (derived in `lightmap_density.py`).
DENSITY_PURE_NOISE_SHARE = 0.75
