"""Check bearing: a part is held the way an ordinary assembly holds it.

Five rules, over each piece of a kind in `kinds` (assembly), in the piece's frame (y up). A
part printed on a surface (role `eyes` or `decal`) is left out of all five. A part or host with
no triangles is left out too (floaters reports it).

Contact. The faces of two parts are in contact (`geom.contact_faces`) when they lie within the
touch distance (the piece's contact length, `sight.contact_length`), look at each other, and
each lies in front of the other, allowing a part to sit into its host by `CROSS_OF_PART_THICKNESS`
times the thinner of the two parts' thickness (`geom.thinner_thickness`): the depth crossings
fails at, so a part crossings accepts keeps its contact. The thickness is read once per part-host
pair, at the centroid of the part's faces nearest the host (those within the touch distance of the
nearest one). Where it cannot be read (an open or unsound mesh) the part's rest and seat rules
are `Unmeasured` and the run goes on. Faces that look away from each other, such as the opposite
sides of a flush cap and post, are not in contact.

1. **Rest or hang** (statics). A part with an `on` host that is not a fastener rests on, or
   hangs from, the host's contact patch. The patch is read as a whole: its area-weighted normal.
   Looking up more than 45 degrees from sideways is a rest, down a hang; anything else (a patch
   that faces sideways, or that faces up and down in equal measure, as a pocket or a clamp does)
   holds the part by something other than gravity and prints a not-applicable line. A rest is
   stable when the part's centre of mass (`geom.centre_of_mass`), projected to y 0, lies inside
   the convex hull of the patch's upward faces, past float noise; a hang, when it also lies under
   the patch's lowest point. Else the item `<pack>:<piece> <part> tips off <host>` (rest) or
   `... swings off <host>` (hang). Not applicable to a part of a bonded row (`Listed.held`) and to every part of a rigged
   piece, where a joint carries each part.
2. **Fastener engagement.** A `fastener` part with an `on` host. Its axis is the principal axis
   of its vertices: the one whose eigenvalue lies furthest from the other two (the long axis of
   a bolt, the short axis of a disc head). Its diameter is twice the median wall radius about the
   axis, taken over the faces that look more outward than along the axis, each at the mean
   distance of its corners from the axis, weighted by area, so a head, few faces, does not move
   it. (The corners of a faceted round shank lie on its circumcircle, the radius of the round
   bolt it stands for; the distance of a face's plane from the axis would read low by a factor
   of cos(pi/N) for N sides. A shank of box section reads at its corners too: the round hole it
   must clear, so about 1.41 times its width for a square one. That is a stricter reading, stated
   and not tuned.) The engagement is the length of
   the axis, between the part's extremes, that the host surrounds: of `ENGAGE_SAMPLES` points
   along it, those inside the host's solid, or from which `ENGAGE_RAYS` rays at right angles to
   the axis all meet the host within the shank radius plus the touch distance (a bore closed
   round the shank, which `geom.inside` misses, and not a cavity wider than that). Under
   `ENGAGE_OF_DIAMETER` times the diameter, the item is `<pack>:<piece> <part> engages <host> <mm>
   mm` (the engagement). A fastener with no `on` host cannot be measured and is `Unmeasured`.
3. **Full-face seating.** A part with an `on` host. Its contact points (Baraff's model, below)
   give the contact direction; the part's contact faces within 45 degrees of it
   (`BEARING_WALL_COSINE`) bear, and the direction is refined to their own mean normal by area,
   then to the plane of the part's seat corners (its contact corners nearest the host: four or
   more, spread over a plane, all in it to float noise) when that plane agrees with it within 45
   degrees. So a host edge or bevel in reach cannot tilt a flat seat, and a side contact (an end
   face against a pocket wall) cannot stand in for the seat. Contact faces that look opposite ways
   are held between surfaces. The contact plane is the mean plane of the part's seat faces that
   look that way. The host's outline is the surface the seat bears on: the faces the seat meets
   (Baraff's contact, the separation 0 to float noise, or the nearest within float noise of the
   least gap) that turn from the seat by less than the style's crease angle (`Style.crease_deg`),
   and every face joined to them across an edge turned less than that angle, so that the facets
   of a curve (a lathed base, a radiused board) are one surface and a bevel or edge turned more
   ends it; holes inside it are counted in (a point is inside when rays from it in the plane meet
   the outline all round). The seat is cut to the outline's extent in the plane, padded by the
   touch distance, and read where it lies at the outline's level (along the outline's own normal,
   its extent padded by the touch distance and float noise): a point there more than float noise
   (`WELD_OF_PIECE_SIZE` times the piece size) outside the outline is an overhang, the item
   `<pack>:<piece> <part> overhangs <host>`; a point climbing away from the host rests on nothing
   and is not read. A seat with no point read is `Unmeasured`, never a pass; so is every seat in a
   style with no crease angle. Not applicable to a part on a rigged piece whose bones differ from
   its host's (a joint carries it).
4. **Retainer touches.** A part with `retains` lies within float noise of each part it retains
   (`geom.surface_gap`): real contact. Else the item
   `<pack>:<piece> <part> off <retained>`.

5. **Sinks into** (non-penetration). A part with an `on` host that is not a fastener, in a row held
   `rests` (not `bonded`; a bond may be set into its host, and a designed embed is an allowlist
   line with a reason), lies on its host's surface: no point of it is inside the host deeper than
   the contact length. Its contact points are read from a window of the touch distance plus the
   sink, crossings' limit (a part pushed past the touch distance has no face within it of the
   host, so the narrow contact loses it, and rules 1 and 3 then read "no face bears"): the two
   limits meet with no gap. Its depth is the penetration depth of its points inside the host, the
   shortest move that takes them all out, over the contact direction and crossings' 13
   directions both ways. Over the contact length the item is `<pack>:<piece> <part> sinks into
   <host> <mm> mm`. It is judged by depth and not by the up/down test of rule 1, so a part on a
   patch that faces sideways is judged too. A part sunk past crossings' limit is crossings'
   finding too, and this rule never passes it. Not applicable to a part on a rigged piece whose
   bones differ from its host's.

Contact points (rules 3 and 5). Baraff's contact model: a contact is a point of one body on a face
of the other, and its normal is that face's outward normal; the bodies are apart there by the
normal's dot with the gap (eq. 9-1), negative when they interpenetrate. The part's faces in the
wide contact (the touch distance plus the sink) and those with a corner inside the host are
sampled `CROSS_SAMPLE_OF_THICKNESS` times the thinner part's thickness apart, each sample an equal
share of the part's surface. A sample's normal is that of the host face it lies square off, or is
inside; where the host meets it with an edge or corner, the part's own. A sample inside the host
or within the touch distance of it is in contact. The contact direction is the mean of the
contact normals, refined once to those within 45 degrees of it: a host face counts at the area of
the part that touches it, never at its whole area, and a tall side face of the part counts only
where it touches.

Lengths are in model units and convert to millimetres through the style's `units_per_metre`.
An `on` host or retained part that has no geometry in the scene raises, as floaters does. A part
whose centre of mass or axis cannot be measured (not a closed solid) makes the run `partial`
(`Unmeasured`). A rule that does not apply prints a not-applicable line. A piece with no `view` has
no contact length: every rule reads `Unmeasured` for it and it is judged on nothing.

Plants, one unit per rule per piece where the rule applies, built from the geometry alone (a
part's contact patch, centre of mass, axis, seat and outline; never a rule's verdict). Each unit
takes the smallest part the rule can judge and moves it: slide a resting or hanging part over the
edge of its patch until its centre of mass is `CALIBRATION_FAIL` touches past it (at most half the
part's own extent, so that a narrow part keeps some contact); pull a fastener clear of its host
along its axis, both ways, so that it holds 0.0 mm (the item names that engagement); slide a
mounted part (a fastener's head too) along its seat until the seat face overhangs the host's
outline by `CALIBRATION_FAIL` touches, or by half, a tenth or a fiftieth of the seat's own extent
when that is less (a long slide takes a bolt's head clear of the seat); move a retainer clear of
what it retains by `CALIBRATION_FAIL` touches; push a resting part into its host along its
contact direction to `CALIBRATION_FAIL` touches deep, or to `SINK_PLANT_OF_CROSS` times the depth
crossings fails at of the thinner extent of the part and its host when that is less (the item
names the depth it puts the part at; a seat with a small gap goes that gap further). The gate
skips a unit whose part already fails and stops the run when no move is caught. A rule that no
part is judged by has no unit.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, field

import numpy as np

from model_kit.core import edges, geom
from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    BEARING_WALL_COSINE,
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    CROSS_OF_PART_THICKNESS,
    CROSS_SAMPLE_OF_THICKNESS,
    ENGAGE_OF_DIAMETER,
    ENGAGE_RAYS,
    ENGAGE_SAMPLES,
    SINK_PLANT_OF_CROSS,
    WELD_OF_PIECE_SIZE,
)

NAME = "bearing"
Y = 1  # the up axis in the piece frame (glTF)

PRINCIPLE = ("A part stays where it is put by ordinary assembly mechanics: its weight lies over "
             "its support, its fastener sits in what it fastens for about its own diameter, its "
             "seat bears on its host over the whole seat, a retainer touches what it retains, and "
             "a resting part lies on its host's surface and does not sink into it.")
SOURCE = (
    "Rule 1: a body rests stably when its centre of mass projects inside the support polygon, the "
    "convex hull of its contact (statics); Prevost, Whiting, Lefebvre and Sorkine-Hornung, 'Make "
    "It Stand: Balancing Shapes for 3D Fabrication', ACM Trans. Graph. 32(4), SIGGRAPH 2013, "
    "https://doi.org/10.1145/2461912.2461957 (the support polygon as the hull of contact is "
    "stated in Prevost et al. 2016, "
    "https://cs.dartmouth.edu/wjarosz/publications/prevost16balancing.pdf, which this "
    "rule follows; the 2013 text is not quoted here). "
    "Rule 2: Nord-Lock, 'The Experts: Thread engagement in a tapped hole' (2015), "
    "approximately 1 times the nominal diameter in steel, "
    "https://www.nord-lock.com/insights/bolting-tips/2015/the-experts-thread-engagement-in-a-tapped-hole/, "
    "with the physics in NASA RP-1228 Fastener Design Manual (Barrett, 1990), 'Calculating "
    "Pullout Load for Threaded Hole', https://ntrs.nasa.gov/api/citations/19900009424/downloads/19900009424.pdf "
    "(a design call of this kit, which has no material field). "
    "Rule 3: a loaded face bears wholly on the surface under it, as a bearing plate does on its "
    "support; ACI 318-19 section 22.8, Bearing, treats the loaded area as lying within the "
    "supporting area, but is not quoted here. No other source is known for the full-face "
    "reading. "
    "Rule 4: a retainer holds a part only through contact, as form closure needs contacts "
    "(Bicchi, 'On the closure properties of robotic grasping', Int. J. Robotics Research 14(4), "
    "1995, https://www.centropiaggio.unipi.it/sites/default/files/grasp-IJRR95.pdf; the paper "
    "is not quoted here). "
    "Rule 5, and the contact points of rules 3 and 5: rigid bodies in contact do not "
    "interpenetrate, and a resting body bears on the surface under it: Baraff, 'An Introduction "
    "to Physically Based Modeling: Rigid Body Simulation II, Nonpenetration Constraints', "
    "SIGGRAPH '97 Course Notes, section 6 (problems of nonpenetration constraints), section 8 "
    "(Colliding Contact: for a vertex/face contact the contact normal is the outward "
    "unit normal of the contact face (page D40), and for an edge/edge contact it is the cross "
    "product of the two edges (page D41)) and section 9 (Resting Contact, page D50, eq. 9-1: "
    "the separation d = n . (pa - pb) at each contact point, negative where the bodies "
    "interpenetrate), "
    "https://www.cs.cmu.edu/~baraff/sigcourse/notesd2.pdf (the 1997 and 2001 editions of the "
    "course notes 'Physically Based Modeling: Rigid Body Simulation'). "
    "The depth of a part in its host is the penetration depth of collision detection, the "
    "shortest move that separates them, read over a stated set of directions as crossings reads "
    "shared space (`geom.DEPTH_DIRECTIONS`). Rule 3's smooth surface is the style's own crease "
    "angle (`dials.edges.crease_deg`, with its `crease_why`): under it two faces shade as one "
    "surface, so the faceting of a curve is not an edge of the host. "
    "The 45 degree split between up, down and sideways (`BEARING_WALL_COSINE`, rules 1 to 3) has "
    "no source; it is a design call of this kit.")

REST, FASTENER, SEAT, RETAINER, SINK = "rest", "fastener", "seat", "retainer", "sink"
RULES = (REST, FASTENER, SEAT, RETAINER, SINK)

# --- small geometry -------------------------------------------------------------------------


def areas(tris: np.ndarray) -> np.ndarray:
    """The area of each triangle, (N,)."""
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
                                axis=1)


def normals(tris: np.ndarray) -> np.ndarray:
    """Unit face normals by winding, (N, 3); a zero-area face has none (zeros)."""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(length > 0.0, n / np.where(length > 0.0, length, 1.0), 0.0)


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def hull_2d(points: np.ndarray) -> np.ndarray:
    """The convex hull of (N, 2) points, counter-clockwise, (H, 2). Fewer than three corners, or
    all on a line, give the distinct extreme points (H < 3)."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(pts) < 3:
        return pts

    def half(seq):
        out: list[np.ndarray] = []
        for q in seq:
            while len(out) >= 2 and _cross2(out[-1] - out[-2], q - out[-2]) <= 0.0:
                out.pop()
            out.append(q)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def depth_in(hull: np.ndarray, point: np.ndarray) -> float:
    """How far inside a counter-clockwise hull a point lies: its distance to the nearest edge,
    negative when outside. A hull of fewer than three corners holds nothing (-inf)."""
    if len(hull) < 3:
        return -math.inf
    worst = math.inf
    for i in range(len(hull)):
        a, b = hull[i], hull[(i + 1) % len(hull)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length > 0.0:
            worst = min(worst, _cross2(edge, point - a) / length)
    return worst


def _sample_points(tris: np.ndarray, step: float) -> np.ndarray:
    """Points along every edge of the triangles, (M, 3), no further than `step` apart.

    A region that sticks out of an outline has a boundary point out of it too (a path from the
    outside point to far away crosses the boundary outside the outline), so the edges that only
    one triangle uses are enough, and cost the perimeter and not the area.
    """
    if len(tris) == 0:
        return np.zeros((0, 3))
    start = tris.reshape(-1, 3)
    end = np.roll(tris, -1, axis=1).reshape(-1, 3)
    grid = np.round(np.stack([start, end], axis=1) / geom.NO_DIRECTION).astype(np.int64).tolist()
    keys = [tuple(sorted((tuple(a), tuple(b)))) for a, b in grid]
    count = Counter(keys)
    keep = np.array([count[k] == 1 for k in keys], dtype=bool)
    start, end = start[keep], end[keep]
    if len(start) == 0:
        return np.zeros((0, 3))
    cuts = np.maximum(1, np.ceil(np.linalg.norm(end - start, axis=1) / step)).astype(int)
    edge = np.repeat(np.arange(len(start)), cuts + 1)
    first = np.cumsum(cuts + 1) - (cuts + 1)
    t = (np.arange(len(edge)) - first[edge]) / cuts[edge]
    return start[edge] + t[:, None] * (end[edge] - start[edge])


# --- one piece, measured --------------------------------------------------------------------


@dataclass
class Entry:
    """What one rule found for one part: a finding, a pass, or why it could not be judged."""

    rule: str
    part: str
    status: str  # "pass", "fail", "na" or "unmeasured"; "ready" before the rule is judged
    finding: Finding | None = None
    note: str = ""  # the reason, for "na" and "unmeasured"
    plan: dict = field(default_factory=dict)  # the geometry the rule reads, and a plant moves


class _Piece:
    """A piece ready to measure: the view, its touch distance and its length scale."""

    def __init__(self, ctx: SceneCtx, view: PieceView) -> None:
        if view.contact is None:
            raise ValueError(f"{view.key}: no contact length without a view")
        self.view = view
        self.touch = view.contact
        self.noise = WELD_OF_PIECE_SIZE * view.size
        self.rigged = view.piece.rig is not None
        self.held = {row.id: row.held for row in view.piece.listed}
        self._indexes: dict[str, geom.SurfaceIndex] = {}
        self.sinks: dict[tuple[str, str], float] = {}  # `_sink` per (part, host) pair
        self.touches: dict[tuple[str, str], _Touch] = {}  # `_touch` per (part, host) pair
        self.crease = ctx.style.crease_deg  # the style's crease angle, or None
        self._joins: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self.mm = 1000.0 / ctx.style.units_per_metre
        self.records = {p.name: p for p in view.piece.parts}
        self.by_name = {p.name: p for p in view.parts}

    def host_of(self, part: PartMesh) -> PartMesh:
        host = self.by_name.get(part.on)
        if host is None or len(host.tris) == 0:
            raise ManifestError(f"{self.view.key} {part.name}: on {part.on!r}, which has no "
                                f"geometry in the scene; the scene has {sorted(self.by_name)}")
        return host

    def index(self, host: PartMesh) -> geom.SurfaceIndex:
        if host.name not in self._indexes:
            self._indexes[host.name] = geom.SurfaceIndex(host.tris)
        return self._indexes[host.name]

    def joins(self, host: PartMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The host's faces' neighbours across its interior edges (`edges.interior_edges`, welded
        at float noise), as (start, neighbour, turn in degrees): face k's neighbours are
        `neighbour[start[k]:start[k + 1]]`."""
        if host.name not in self._joins:
            found = edges.interior_edges(host.tris, self.noise)
            a = np.concatenate([found.faces[:, 0], found.faces[:, 1]])
            b = np.concatenate([found.faces[:, 1], found.faces[:, 0]])
            turn = np.concatenate([found.turn_deg, found.turn_deg])
            order = np.argsort(a, kind="stable")
            start = np.searchsorted(a[order], np.arange(len(host.tris) + 1))
            self._joins[host.name] = (start, b[order], turn[order])
        return self._joins[host.name]

    def bonded(self, part: PartMesh) -> bool:
        record = self.records.get(part.name)
        return record is not None and self.held.get(record.listed) == "bonded"

    def bones(self, part: PartMesh) -> frozenset[str]:
        record = self.records.get(part.name)
        return frozenset(record.bones) if record is not None else frozenset()

    def item(self, part: PartMesh, how: str, other: str) -> str:
        return f"{item_of(self.view, part.name)} {how} {other}"


def contact_patch(part: np.ndarray, host: np.ndarray, touch: float, sunk: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The contact of two parts' triangles (the module note's "Contact"): masks over the part's
    and the host's faces, `geom.contact_faces` with the touch distance and the allowed sink."""
    return geom.contact_faces(part, host, touch, sunk)


def _seat_point(piece: _Piece, part: PartMesh, host: PartMesh) -> np.ndarray:
    """The centroid of the part's faces nearest its host: those whose centres lie within the touch
    distance of the nearest face centre's distance to the host."""
    dist = piece.index(host).closest(part.tris.mean(axis=1)).dist
    return pieces.centroid(part.tris[dist <= dist.min() + piece.touch])


def _sink(piece: _Piece, part: PartMesh, host: PartMesh) -> float:
    """How far the part may sit in its host: `CROSS_OF_PART_THICKNESS` times the thinner part's
    thickness at the seat (`geom.thinner_thickness`, read once per part-host pair and kept).
    Raises `Unmeasurable` where that thickness cannot be read."""
    key = (part.name, host.name)
    if key not in piece.sinks:
        piece.sinks[key] = CROSS_OF_PART_THICKNESS * geom.thinner_thickness(
            part.tris, host.tris, _seat_point(piece, part, host))
    return piece.sinks[key]


def _contact(piece: _Piece, part: PartMesh, host: PartMesh) -> tuple[np.ndarray, np.ndarray]:
    """Raises `Unmeasurable` when the allowed sink cannot be read (`_sink`)."""
    return contact_patch(part.tris, host.tris, piece.touch, _sink(piece, part, host))


def _rest_plan(piece: _Piece, part: PartMesh, host: PartMesh, patch: np.ndarray) -> Entry:
    """Rule 1's geometry for one part. The patch is read as a whole: its area-weighted normal.
    The entry is `ready` with the patch's hull, the part's centre of mass and whether it rests
    or hangs, or says why the rule cannot judge the part."""
    ht = host.tris[patch]
    n, area = normals(ht), areas(ht)
    net = float((area * n[:, Y]).sum() / area.sum())  # the patch's mean normal, y part
    if abs(net) <= BEARING_WALL_COSINE:
        return Entry(REST, part.name, "na", note="its patch faces neither up nor down as a whole")
    resting = net > 0.0
    pts = ht[n[:, Y] > 0 if resting else n[:, Y] < 0].reshape(-1, 3)
    try:
        com = geom.centre_of_mass(part.tris)
    except ValueError:
        return Entry(REST, part.name, "unmeasured", note="not a closed solid, no centre of mass")
    return Entry(REST, part.name, "ready", plan={
        "host": host, "pts": pts, "hull": hull_2d(pts[:, [0, 2]]), "com": com,
        "resting": resting})


def _rest(piece: _Piece, part: PartMesh, entry: Entry) -> Entry:
    """Rule 1 for one part: its centre of mass over its patch."""
    plan = entry.plan
    host, pts, hull, com, resting = (plan[k] for k in ("host", "pts", "hull", "com", "resting"))
    depth = depth_in(hull, com[[0, 2]])
    verb, how = ("rests", "tips off") if resting else ("hangs", "swings off")
    why = None
    if depth < piece.noise:
        shown = "outside" if depth < 0 else "on the edge of"
        gap = abs(depth) * piece.mm if math.isfinite(depth) else 0.0
        why = (f"{verb} on {host.name}, and its centre of mass is {shown} the patch's outline "
               f"{gap:.2f} mm (it must lie inside)")
    elif not resting and com[Y] > float(pts[:, Y].min()):
        why = (f"hangs from {host.name}, and its centre of mass is above the patch's lowest point "
               f"by {(com[Y] - pts[:, Y].min()) * piece.mm:.2f} mm")
    if why is None:
        return Entry(REST, part.name, "pass", plan=plan)
    return Entry(REST, part.name, "fail", Finding(NAME, piece.item(part, how, host.name), why),
                 plan=plan)


def _axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(a point on the axis, the unit axis) of a set of vertices: the principal axis whose
    eigenvalue lies furthest from the other two."""
    mean = vertices.mean(axis=0)
    values, vectors = np.linalg.eigh(np.cov((vertices - mean).T))
    long_axis = (values[2] - values[1]) >= (values[1] - values[0])
    return mean, vectors[:, 2 if long_axis else 0]


def _wall_radius(tris: np.ndarray, point: np.ndarray, axis: np.ndarray) -> float | None:
    """The area-weighted median, over the faces that look outward from the axis, of the mean
    distance of each face's corners from the axis (the circumradius of a faceted round shank).
    None with no such face."""
    n, area = normals(tris), areas(tris)
    along = n @ axis
    out = n - np.outer(along, axis)
    length = np.linalg.norm(out, axis=1)
    wall = (np.abs(along) < BEARING_WALL_COSINE) & (length > 0.0) & (area > 0.0)
    if not wall.any():
        return None
    rel = tris[wall] - point  # (W, 3, 3): each corner from the axis point
    across = rel - (rel @ axis)[..., None] * axis  # its part at right angles to the axis
    radius = np.linalg.norm(across, axis=2).mean(axis=1)
    order = np.argsort(radius)
    cum = np.cumsum(area[wall][order])
    return float(radius[order][np.searchsorted(cum, cum[-1] / 2.0)])


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def ringed(host: np.ndarray, points: np.ndarray, axis: np.ndarray, reach: float) -> np.ndarray:
    """Which points the host's wall rings: every one of `ENGAGE_RAYS` rays, spread round the axis
    at right angles to it, meets the host within `reach`. That counts a bore the host closes
    all round, which `geom.inside` misses, and not a cavity wider than the shank's reach."""
    u, v = _plane_basis(axis)
    angle = 2.0 * math.pi * np.arange(ENGAGE_RAYS) / ENGAGE_RAYS
    rays = np.outer(np.cos(angle), u) + np.outer(np.sin(angle), v)
    return (geom.ray_counts(host, points, rays, reach) > 0).all(axis=1)


def _enclosed(host: np.ndarray, points: np.ndarray, axis: np.ndarray, reach: float
              ) -> np.ndarray:
    """Which axis points the host surrounds: inside its solid, or ringed by its wall (`ringed`)."""
    held = ringed(host, points, axis, reach)
    if geom.is_closed(host):
        held |= geom.inside(host, points)
    return held


def _fastener_plan(piece: _Piece, part: PartMesh, host: PartMesh) -> Entry:
    """Rule 2's geometry for one fastener: its axis, its shank radius and its extent along the
    axis, or why the rule cannot judge it."""
    vertices = np.unique(part.tris.reshape(-1, 3), axis=0)
    if len(vertices) < 4:
        return Entry(FASTENER, part.name, "unmeasured", note="too few vertices for an axis")
    point, axis = _axis(vertices)
    radius = _wall_radius(part.tris, point, axis)
    if radius is None or not radius > 0.0:
        return Entry(FASTENER, part.name, "unmeasured", note="no wall faces, no shank diameter")
    t = (vertices - point) @ axis
    return Entry(FASTENER, part.name, "ready", plan={
        "host": host, "point": point, "axis": axis, "radius": radius,
        "lo": float(t.min()), "hi": float(t.max())})


def _engage(piece: _Piece, part: PartMesh, entry: Entry) -> Entry:
    """Rule 2 for one fastener."""
    plan = entry.plan
    host, point, axis, radius = (plan[k] for k in ("host", "point", "axis", "radius"))
    lo, hi = plan["lo"], plan["hi"]
    length = hi - lo
    at = lo + (np.arange(ENGAGE_SAMPLES) + 0.5) * length / ENGAGE_SAMPLES
    held = _enclosed(host.tris, point + np.outer(at, axis), axis, radius + piece.touch)
    engaged = float(held.mean()) * length
    need = ENGAGE_OF_DIAMETER * 2.0 * radius
    plan = {**plan, "engaged": engaged, "need": need}
    if engaged >= need:
        return Entry(FASTENER, part.name, "pass", plan=plan)
    item = piece.item(part, "engages", f"{host.name} {engaged * piece.mm:.1f} mm")
    why = (f"sits {engaged * piece.mm:.2f} mm along its axis inside {host.name}; its shank "
           f"diameter is {2 * radius * piece.mm:.2f} mm, so it needs {need * piece.mm:.2f} mm")
    return Entry(FASTENER, part.name, "fail", Finding(NAME, item, why), plan=plan)


def _rays_meet(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """In a plane: whether each of `ENGAGE_RAYS` rays from each point meets the (N, 3, 2)
    triangles' edges, (M, ENGAGE_RAYS). A point inside a triangle is the caller's to settle."""
    angle = 2.0 * math.pi * np.arange(ENGAGE_RAYS) / ENGAGE_RAYS
    dirs = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    start = tris.reshape(-1, 2)
    along = (np.roll(tris, -1, axis=1) - tris).reshape(-1, 2)
    out = np.zeros((len(points), ENGAGE_RAYS), dtype=bool)
    step = max(1, geom.PAIR_BLOCK // max(1, len(start)))
    for k, d in enumerate(dirs):
        denom = d[0] * along[:, 1] - d[1] * along[:, 0]
        live = np.abs(denom) > geom.NO_DIRECTION * np.linalg.norm(along, axis=1)
        for lo in range(0, len(points), step):
            w = start[None, :, :] - points[lo : lo + step, None, :]
            t = (w[..., 0] * along[:, 1] - w[..., 1] * along[:, 0]) / np.where(live, denom, 1.0)
            u = (w[..., 0] * d[1] - w[..., 1] * d[0]) / np.where(live, denom, 1.0)
            out[lo : lo + step, k] = (live & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)).any(axis=1)
    return out


def _seat_corners_plane(piece: _Piece, faces: np.ndarray, host: PartMesh,
                        toward: np.ndarray) -> np.ndarray | None:
    """The unit normal of the plane through the part's seat corners, turned to face `toward` (the
    host patch's direction), or None. The seat corners are the corners of the part's contact faces
    nearest its host (within float noise of the nearest one). When there are four or more (three
    always span a plane, so only a fourth tests flatness), they spread along two directions (their
    RMS spread across the second is over float noise: they are not on a line) and all lie within
    float noise of one plane, the part sits flat on its host, and that plane is its seat's,
    whatever slanted host faces the slack let into the patch.

    The plane overrides the patch only where the two agree within 45 degrees
    (`BEARING_WALL_COSINE`, the cosine at which a face looks one way more than another). The
    patch's mean normal is the coarse direction of the whole contact; the corners refine it. When
    the nearest corners lie on a side contact (a heel's end face against a pocket wall, a fret's
    end against a board's end), their plane is a different contact from the one the patch
    describes, and it does not stand in for the seat. Else None, and the host patch decides."""
    corners = np.unique(faces.reshape(-1, 3), axis=0)
    dist = piece.index(host).closest(corners).dist
    seat = corners[dist <= dist.min() + piece.noise]
    if len(seat) < 4:
        return None
    centred = seat - seat.mean(axis=0)
    _, values, vh = np.linalg.svd(centred, full_matrices=False)
    if not values[1] / math.sqrt(len(seat)) > piece.noise:  # RMS spread across the second axis
        return None
    if float(np.abs(centred @ vh[2]).max()) > piece.noise:
        return None
    if abs(vh[2] @ toward) <= BEARING_WALL_COSINE:
        return None
    return vh[2] if vh[2] @ toward > 0.0 else -vh[2]


@dataclass
class _Touch:
    """The contact points of one part on its host, after Baraff's contact model: points of the
    part's surface, each with the contact normal there (out of the part, into the host) and its
    signed depth (positive inside the host's solid, negative off it)."""

    pts: np.ndarray  # (M, 3) points on the part's faces
    into: np.ndarray  # (M, 3) the unit contact normal at each, pointing into the host
    depth: np.ndarray  # (M,) how far inside the host (negative: how far off it)
    held: np.ndarray  # (M,) inside the host's solid
    face: np.ndarray  # (M,) the host face nearest each
    touching: np.ndarray  # (M,) in contact: inside the host, or within the touch distance of it
    reached: bool  # some face of the host lies in the wide contact (`_touch`)


def _touch(piece: _Piece, part: PartMesh, host: PartMesh) -> _Touch:
    """The contact points of the part on its host (`_Touch`), read once per part-host pair and
    kept: rule 3 and rule 5 read the same ones. Raises `Unmeasurable` where the sink cannot be
    read (`_sink`).

    Baraff's contact model (the module's `SOURCE`, rule 5): a contact is a point of one body on a
    face of the other, and its contact normal is that face's outward normal; the bodies are apart
    there by the normal's dot with the gap between the two points (eq. 9-1), negative when they
    interpenetrate. The faces read are those in the wide contact, which reaches the touch distance
    plus the sink (crossings' limit: a part pushed past the touch distance has no face within it of
    the host, so the narrow contact loses it), and those with a corner inside the host (the end of
    a rod set into a ball has none of its faces in a contact), cut to the host's box padded by that
    reach (a long face, a string's, reaches far past its contact). They are sampled
    `CROSS_SAMPLE_OF_THICKNESS` times the thinner part's thickness apart (crossings' step; their
    corners are always samples), so each sample stands for an equal share of the part's surface.
    At each sample the nearest point of the host decides the contact: inside the host's solid, or
    on the inside of a host face (the sample lies square off it, to float noise), the normal is
    the host face's, reversed; on a host edge or corner the face is the part's, and the normal is
    the part's own. A sample is in contact when it lies inside the host or within the touch
    distance of it (a sample by a host edge that lies behind its own face is not). Its depth is
    its distance to the host's surface: Baraff's separation along the contact normal of the face
    it lies off or has gone through.
    """
    key = (part.name, host.name)
    if key in piece.touches:
        return piece.touches[key]
    sink = _sink(piece, part, host)
    window, patch = contact_patch(part.tris, host.tris, piece.touch + sink, sink)
    index = piece.index(host)
    window |= index.inside(part.tris.reshape(-1, 3)).reshape(-1, 3).any(axis=1)
    reach = piece.touch + sink  # no point further than this from the host's box is in reach
    faces = geom.clip_to_box(part.tris[window], (host.lo - reach, host.hi + reach))
    if not len(faces):
        none = np.zeros((0, 3))
        flags = np.zeros(0, dtype=bool)
        out = _Touch(none, none, np.zeros(0), flags, np.zeros(0, dtype=np.intp), flags,
                     bool(patch.any()))
        piece.touches[key] = out
        return out
    pts = geom.surface_samples(faces, CROSS_SAMPLE_OF_THICKNESS * sink / CROSS_OF_PART_THICKNESS)
    near = index.closest(pts)
    held = index.inside(pts)
    gap = pts - near.point  # from the host's nearest point out to the sample
    across = np.linalg.norm(gap - (gap * near.normal).sum(axis=1)[:, None] * near.normal, axis=1)
    on_face = held | (across <= piece.noise)
    into = -near.normal
    ahead = on_face.copy()
    if not on_face.all():  # the part's own face normal, where the host offers an edge or corner
        own = geom.SurfaceIndex(faces).closest(pts[~on_face]).normal
        into[~on_face] = own
        ahead[~on_face] = (-gap[~on_face] * own).sum(axis=1) >= -piece.noise
    touching = held | ((near.dist <= piece.touch + piece.noise) & ahead)
    out = _Touch(pts, into, np.where(held, near.dist, -near.dist), held, near.face, touching,
                 bool(patch.any()))
    piece.touches[key] = out
    return out


def _contact_direction(touch: _Touch) -> np.ndarray | None:
    """The part's contact direction, out of it into its host: the mean of the contact normals of
    its contact points (`_touch`), each point an equal share of the part's surface, so that a host
    face counts at the area of the part that touches it and not at its whole area, and a part's
    face only where it touches. Refined once to the points whose normal lies within 45 degrees of
    that mean (`BEARING_WALL_COSINE`), so that a side contact does not tilt it. None when no point
    is in contact, or when the normals cancel (a part held between surfaces)."""
    into = touch.into[touch.touching]
    if not len(into):
        return None
    total = into.sum(axis=0)
    if not np.linalg.norm(total) > geom.NO_DIRECTION * len(into):
        return None
    n = total / np.linalg.norm(total)
    near = into @ n > BEARING_WALL_COSINE
    if near.any():
        refined = into[near].sum(axis=0)
        n = refined / np.linalg.norm(refined)
    return n


def _seat_face(piece: _Piece, part: PartMesh, host: PartMesh, seat_faces: np.ndarray,
               rule: str) -> tuple[np.ndarray, float, np.ndarray] | Entry:
    """The seat of one part on its host, for rule 3: (the contact direction, the seat's offset
    along it, the seat faces), or the `rule` entry that says why the part has no seat.

    The contact direction is `_contact_direction`'s, from the part's contact points (`_touch`).
    It says which of the part's contact faces bear: those within 45 degrees of it
    (`BEARING_WALL_COSINE`). The seat is the part's, so the direction is refined to their own
    mean normal, by area (a face of the part bears on the host, as in Baraff's vertex and face
    contact with the face on the part: a host bevel or edge in reach of a flat seat then tilts
    nothing), and then to the plane of the part's seat corners when it sits flat and that plane
    agrees (`_seat_corners_plane`). The contact plane is the mean plane of the part's contact faces
    that look that way, and the seat is the faces of it that lie in that plane (within the touch
    distance).
    """
    faces = part.tris[seat_faces]
    touch = _touch(piece, part, host)
    if not touch.touching.any():
        gap = float(-touch.depth.max()) * piece.mm
        return Entry(rule, part.name, "unmeasured", note=(
            f"no point of its contact faces lies within the contact length of its host (the "
            f"nearest is {gap:.2f} mm off), so where its seat bears cannot be read"))
    fn, fa = normals(faces), areas(faces)
    n = _contact_direction(touch)
    total = (fa[:, None] * fn).sum(axis=0)  # its contact faces, by area: none where they cancel
    if n is None or not np.linalg.norm(total) > geom.NO_DIRECTION * fa.sum():
        return Entry(rule, part.name, "na", note="its contact faces look opposite ways, it is "
                                                 "held between surfaces")
    bearing_faces = fn @ n > BEARING_WALL_COSINE
    if bearing_faces.any():
        own = (fa[bearing_faces, None] * fn[bearing_faces]).sum(axis=0)
        n = own / np.linalg.norm(own)
    from_corners = _seat_corners_plane(piece, faces, host, n)
    if from_corners is not None:
        n = from_corners
    square = fn @ n > BEARING_WALL_COSINE
    if not square.any():
        return Entry(rule, part.name, "na", note="no face of it looks squarely at its host")
    offset = float((faces[square].mean(axis=1) @ n * fa[square]).sum() / fa[square].sum())
    flat = square & (np.abs(faces @ n - offset) <= piece.touch).all(axis=1)
    if not flat.any():
        return Entry(rule, part.name, "na", note="no face of it lies in the contact plane")
    return n, offset, faces[flat]


def _bearing_surface(piece: _Piece, host: PartMesh, touch: _Touch, n: np.ndarray) -> np.ndarray:
    """The host's surface a seat bears on, as a mask over the host's faces: the faces that look at
    the part (within 45 degrees of the contact direction `n`, `BEARING_WALL_COSINE`) and that join
    a face the seat meets without a crease.

    The faces the seat meets are those nearest its contact points along `n` that meet the host
    (Baraff's contact, the separation 0 to float noise, or inside it; a seat that rests a gap off
    meets it first at its nearest points, within float noise of the least gap), and that turn from
    the seat's plane by less than the style's crease angle (`Style.crease_deg`, the angle past
    which the look draws a fold line: under it two faces shade as one smooth surface). From them
    the surface runs on across every edge that turns less than the crease angle: such an edge is
    the faceting of a curve, a lathed base or a radiused board, not an edge of the host. So a seat
    on a curved host bears on the curve, and a seat past a host's edge (a sharp one, or a bevel
    turned more than the crease angle) does not bear on the far side of it: the seat meets a bevel
    only along its edge, as a face meets a corner."""
    hn = normals(host.tris)
    level = hn @ n < -max(BEARING_WALL_COSINE, math.cos(math.radians(piece.crease)))
    along = touch.touching & (touch.into @ n > BEARING_WALL_COSINE)
    gap = -touch.depth
    meets = along & (touch.held | (gap <= piece.noise))
    if along.any() and not meets.any():
        meets = along & (gap <= gap[along].min() + piece.noise)
    region = np.zeros(len(host.tris), dtype=bool)
    region[touch.face[meets]] = True
    region &= level
    if not region.any():
        return region
    start, neighbour, turn = piece.joins(host)
    frontier = np.flatnonzero(region)
    while len(frontier):  # out from the faces it meets, across every smooth edge
        count = start[frontier + 1] - start[frontier]
        at = np.repeat(start[frontier] - np.cumsum(count) + count, count) + np.arange(count.sum())
        step = neighbour[at][turn[at] < piece.crease]
        step = np.unique(step[level[step] & ~region[step]])
        region[step] = True
        frontier = step
    return region


def _seat_plan(piece: _Piece, part: PartMesh, host: PartMesh, seat_faces: np.ndarray) -> Entry:
    """Rule 3's geometry for one part with a host, or why the rule cannot judge it.

    The seat and its contact direction are `_seat_face`'s. The host's outline is the surface the
    seat bears on (`_bearing_surface`), holes inside it counted in (a point is inside when rays
    from it in the plane meet the host all round). Only the part of the seat within the touch
    distance of the host counts: the seat faces are cut to the outline's extent in the plane,
    padded by the touch distance, and a sample further than that from the host is no contact.
    A style with no crease angle cannot say where the host's surface turns into an edge, so the
    rule is `Unmeasured` there.
    """
    if piece.crease is None:
        return Entry(SEAT, part.name, "unmeasured", note="the style has no crease_deg in "
                     "dials.edges, so where its host's surface turns into an edge cannot be read")
    got = _seat_face(piece, part, host, seat_faces, SEAT)
    if isinstance(got, Entry):
        return got
    n, _, seat = got
    lit = _bearing_surface(piece, host, _touch(piece, part, host), n)
    if not lit.any():
        return Entry(SEAT, part.name, "na", note="its host has no face in the contact plane")
    outline = host.tris[lit]
    u, v = _plane_basis(n)
    frame = np.stack([u, v, n])
    plane = outline @ frame.T
    lo, hi = plane.reshape(-1, 3).min(axis=0), plane.reshape(-1, 3).max(axis=0)
    deep = float(np.abs(seat @ n).max() + np.abs(lo[2]) + np.abs(hi[2])) + 1.0
    up = (areas(outline)[:, None] * normals(outline)).sum(axis=0)  # the surface's own normal
    up /= np.linalg.norm(up)
    level = outline.reshape(-1, 3) @ up
    reach = piece.touch + piece.noise
    return Entry(SEAT, part.name, "ready", plan={
        "host": host, "normal": n, "u": u, "v": v, "seat": seat, "patch": outline, "frame": frame,
        "lo": lo, "hi": hi, "deep": deep, "up": up,
        "band": (float(level.min()) - reach, float(level.max()) + reach)})


def _seat(piece: _Piece, part: PartMesh, entry: Entry) -> Entry:
    """Rule 3 for one part: its seat face inside the host's outline."""
    plan = entry.plan
    host, n, u, v = plan["host"], plan["normal"], plan["u"], plan["v"]
    seat, outline, frame = plan["seat"], plan["patch"], plan["frame"]
    lo, hi, deep = plan["lo"], plan["hi"], plan["deep"]
    box = ([lo[0] - piece.touch, lo[1] - piece.touch, -deep],
           [hi[0] + piece.touch, hi[1] + piece.touch, deep])
    near = geom.clip_to_box(seat @ frame.T, box) @ frame if len(seat) else seat
    pts = _sample_points(near, CALIBRATION_PASS * piece.touch) if len(near) else seat[:0, 0]
    # The seat is read where it lies at the level of the surface it bears on, to the contact length
    # (along that surface's own normal): a point at that level past the surface's edge hangs past
    # it, and a point that climbs away from the host (a bar sheared up off its block) rests on
    # nothing there and is not read.
    at = pts @ plan["up"]
    counted = pts[(at >= plan["band"][0]) & (at <= plan["band"][1])]
    if not len(counted):
        # A seat with no point read has nothing to read an overhang from: no pass.
        return Entry(SEAT, part.name, "unmeasured", plan=plan, note=(
            "no point of its seat face lies at the level of the surface it bears on, within the "
            "contact length, so where it bears cannot be read"))
    over = 0.0
    dist = geom.footprint_distance(outline, n, counted)
    off = dist > piece.noise
    if off.any():
        flat2 = np.stack([u, v], axis=1)
        hole = _rays_meet(counted[off] @ flat2, outline @ flat2).all(axis=1)
        over = float(dist[off][~hole].max()) if (~hole).any() else 0.0
    if not over > piece.noise:
        return Entry(SEAT, part.name, "pass", plan=plan)
    why = (f"its seat face on {host.name} reaches at least {over * piece.mm:.2f} mm past the "
           f"surface it bears on, in the contact plane (read over that surface's extent, padded by "
           f"the contact length)")
    return Entry(SEAT, part.name, "fail", Finding(NAME, piece.item(part, "overhangs", host.name),
                                                  why), plan=plan)


def _penetration(piece: _Piece, host: PartMesh, touch: _Touch, n: np.ndarray) -> float:
    """How far a part has gone into its host, from its points inside it: the shortest move that
    takes them all out (the penetration depth of collision detection), over the moves along the
    contact direction `n` and along crossings' 13 directions both ways (`geom.DEPTH_DIRECTIONS`,
    the same stand-in for every direction). Along each, the way out is the furthest any point must
    go to leave the host (`geom.exits`). So a fret sunk into a board's top and hanging over its
    end reads its depth below the top, not its run along the board."""
    ways = np.vstack([-n, geom.DEPTH_WAYS])
    pts, into = touch.pts[touch.held], touch.into[touch.held]
    # A point inside the host by no more than float noise lies on its surface, and it leaves at
    # once along a way out through the face it lies on (one that looks out of the host there).
    leaves = (touch.depth[touch.held] <= piece.noise)[:, None] & (-into @ ways.T > 0.0)
    reach = max(2.0 * float(touch.depth[touch.held].max()), piece.touch)
    return geom.shortest_exit([geom.Exit(host.tris, pts, ways, leaves)], reach)


def _sink_plan(piece: _Piece, part: PartMesh, host: PartMesh) -> Entry:
    """Rule 5's geometry for one part of a `rests` row, or why the rule cannot judge it.

    The part's contact points (`_touch`, rule 3's) say where it has gone into its host: those
    inside its solid. Its depth is their penetration depth
    (`_penetration`): Baraff's separation (eq. 9-1) is read along a contact normal, and the
    shortest way out is the one along the normal of the face the part went in by, so a fret on a
    board's top that hangs over the board's end is as deep as it lies under the top, and a heel
    pushed into a pocket's end wall as deep as it lies past that wall. 0 when no point is inside.
    The contact direction (`_contact_direction`, rule 3's) is the way a plant pushes; `level` is
    how deep the seat lies along it (the depth when the part is in its host, else minus the gap
    of the nearest point whose normal lies within 45 degrees of it), and the plan keeps the least
    extent of the part and of the host in that frame (a rod's thickness is its width, not its
    length), which the plant sizes its push by. The host is a closed solid here (the sink was read
    from it).
    """
    touch = _touch(piece, part, host)
    if not touch.touching.any():
        return Entry(SINK, part.name, "na", note="no point of it is in contact with its host "
                     "(floaters reports a part off its host)")
    n = _contact_direction(touch)
    if n is None:
        return Entry(SINK, part.name, "na", note="its contact points look opposite ways, it is "
                     "held between surfaces")
    if touch.held.any():
        depth = level = _penetration(piece, host, touch, n)
    else:
        depth = 0.0
        level = float(touch.depth[touch.into @ n > BEARING_WALL_COSINE].max(initial=-np.inf))
    u, v = _plane_basis(n)
    frame = np.stack([u, v, n])
    extent = min(float(np.ptp(m.tris.reshape(-1, 3) @ frame.T, axis=0).min())
                 for m in (part, host))
    return Entry(SINK, part.name, "ready", plan={
        "host": host, "normal": n, "level": level, "depth": depth, "extent": extent})


def _sinks(piece: _Piece, part: PartMesh, entry: Entry) -> Entry:
    """Rule 5 for one part: its seat no deeper in the host than the contact length."""
    plan = entry.plan
    host, depth = plan["host"], plan["depth"]
    if not depth > piece.touch:
        return Entry(SINK, part.name, "pass", plan=plan)
    item = piece.item(part, "sinks into", f"{host.name} {depth * piece.mm:.1f} mm")
    why = (f"it sits {depth * piece.mm:.2f} mm inside {host.name} (the shortest move out of it); a "
           f"resting part lies on the host's surface, and a gap under "
           f"{piece.touch * piece.mm:.2f} mm (the contact length) is all it may sink")
    return Entry(SINK, part.name, "fail", Finding(NAME, item, why), plan=plan)


def _retainer_plans(piece: _Piece, part: PartMesh) -> list[Entry]:
    """Rule 4's geometry for one part that retains others: one `ready` entry per retained part."""
    out = []
    for name in piece.records[part.name].retains:
        held = piece.by_name.get(name)
        if held is None or len(held.tris) == 0:
            raise ManifestError(f"{piece.view.key} {part.name}: retains {name!r}, which has no "
                                f"geometry in the scene; the scene has {sorted(piece.by_name)}")
        out.append(Entry(RETAINER, part.name, "ready", plan={"held": name}))
    return out


def _retain(piece: _Piece, part: PartMesh, entry: Entry) -> Entry:
    """Rule 4 for one retained part: the retainer touches it."""
    name = entry.plan["held"]
    held = piece.by_name[name]
    a = geom.SurfaceIndex(part.tris)
    b = geom.SurfaceIndex(held.tris)
    gap = geom.surface_gap(a, b, piece.noise)
    if gap <= piece.noise:
        return Entry(RETAINER, part.name, "pass", plan=entry.plan)
    size = float(np.linalg.norm(np.maximum(part.hi, held.hi) - np.minimum(part.lo, held.lo)))
    gap = geom.surface_gap(a, b, max(piece.touch, size))
    why = f"{gap * piece.mm:.2f} mm from {name}, which it retains (it must touch)"
    return Entry(RETAINER, part.name, "fail", Finding(NAME, piece.item(part, "off", name), why),
                 plan=entry.plan)


def _sink_entry(piece: _Piece, part: PartMesh, host: PartMesh, lost: str | None) -> Entry:
    """Rule 5's entry for one part with a host that is not a fastener (a fastener's depth is rule
    2's): `ready`, or why the rule does not judge it."""
    if piece.bonded(part):
        return Entry(SINK, part.name, "na", note="its parts-list row is held bonded: the bond, "
                     "not the surface it rests on, joins it (a bond may be set into its host)")
    if piece.rigged and piece.bones(part) != piece.bones(host):
        return Entry(SINK, part.name, "na", note="a joint carries it: its bones differ from its "
                     "host's")
    if lost:
        return Entry(SINK, part.name, "unmeasured", note=lost)
    sink = _sink(piece, part, host)
    if not sink > piece.touch:
        return Entry(SINK, part.name, "na", note="half the thinner part's thickness is not over "
                     "the contact length, so no depth lies between a seat and crossings' limit")
    touch = _touch(piece, part, host)
    if not touch.reached and not touch.touching.any():
        return Entry(SINK, part.name, "na", note="no face of it lies on or in its host (floaters "
                     "reports a part off its host)")
    return _sink_plan(piece, part, host)


def _plan_part(piece: _Piece, part: PartMesh) -> list[Entry]:
    """Every rule's entry for one solid part with geometry, before it is judged: `ready` with the
    geometry the rule reads, or `na` or `unmeasured` with the reason."""
    record = piece.records.get(part.name)
    entries: list[Entry] = []
    fastener = record is not None and record.fastener
    if part.on is not None:
        host = piece.host_of(part)
        lost = None  # why the contact cannot be read, when it cannot
        try:
            seat_faces, patch = _contact(piece, part, host)
        except Unmeasurable as err:
            seat_faces = patch = None
            lost = (f"no seat thickness (the first part is the part, the second its host "
                    f"{host.name}): {err}")
        if not fastener:
            if piece.bonded(part):
                entries.append(Entry(REST, part.name, "na", note="its parts-list row is held "
                                     "bonded: the bond, not gravity, holds it"))
            elif piece.rigged:
                entries.append(Entry(REST, part.name, "na", note="its piece is rigged: a joint "
                                     "carries each part, not gravity"))
            elif lost:
                entries.append(Entry(REST, part.name, "unmeasured", note=lost))
            elif patch.any():
                entries.append(_rest_plan(piece, part, host, patch))
        if piece.rigged and piece.bones(part) != piece.bones(host):
            entries.append(Entry(SEAT, part.name, "na", note="a joint carries it: its bones "
                                 "differ from its host's"))
        elif lost:
            entries.append(Entry(SEAT, part.name, "unmeasured", note=lost))
        elif patch.any():
            entries.append(_seat_plan(piece, part, host, seat_faces))
        if lost is None and not patch.any():
            note = "no face of it bears on its host (floaters reports a part off its host)"
            if not fastener and not piece.rigged and not piece.bonded(part):
                entries.append(Entry(REST, part.name, "na", note=note))
            if not (piece.rigged and piece.bones(part) != piece.bones(host)):
                entries.append(Entry(SEAT, part.name, "na", note=note))
        if fastener:
            entries.append(_fastener_plan(piece, part, host))
        else:
            entries.append(_sink_entry(piece, part, host, lost))
    elif fastener:
        entries.append(Entry(FASTENER, part.name, "unmeasured", note="fastener with no host"))
    if record is not None and record.retains:
        entries += _retainer_plans(piece, part)
    return entries


_JUDGE = {REST: _rest, FASTENER: _engage, SEAT: _seat, RETAINER: _retain, SINK: _sinks}


def _measure_part(piece: _Piece, part: PartMesh) -> list[Entry]:
    """Every rule's entries for one solid part with geometry."""
    return [_JUDGE[e.rule](piece, part, e) if e.status == "ready" else e
            for e in _plan_part(piece, part)]


def measure(ctx: SceneCtx, view: PieceView, parts: Collection[str] | None = None
            ) -> list[Entry]:
    """Every entry of a piece, or of just the named parts."""
    piece = _Piece(ctx, view)
    entries: list[Entry] = []
    for part in view.parts:
        if (parts is not None and part.name not in parts) or not part.solid or not len(part.tris):
            continue
        entries += _measure_part(piece, part)
    return entries


# --- plants ---------------------------------------------------------------------------------


def _unit_vector(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _margin(touch: float, room: float, share: float = 0.5) -> float:
    """How far past an edge a plant slides a part: `CALIBRATION_FAIL` touches, but at most `share`
    of the part's own `room` along the slide, so that a part narrower than that stays partly over
    the edge and keeps its contact."""
    return min(CALIBRATION_FAIL * touch, share * room)


def _moves(piece: _Piece, part: PartMesh, entry: Entry) -> list[np.ndarray]:
    """Displacements to try on a part, most likely first, read from the entry's geometry."""
    touch = piece.touch
    plan = entry.plan
    out: list[np.ndarray] = []
    if entry.rule == REST:
        hull, com = plan["hull"], plan["com"]
        for d2 in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]):
            d = np.array([d2[0], 0.0, d2[1]])
            reach = float((hull @ np.array(d2)).max() - com[[0, 2]] @ np.array(d2))
            half = 0.5 * float(np.ptp(part.tris.reshape(-1, 3) @ d))
            out.append(d * (reach + _margin(touch, half)))
    elif entry.rule == FASTENER:
        # clear of the host along the axis, so that nothing of the shank is surrounded
        axis, point = plan["axis"], plan["point"]
        along = (plan["host"].tris.reshape(-1, 3) - point) @ axis
        out = [axis * (float(along.max()) - plan["lo"] + touch),
               -axis * (plan["hi"] - float(along.min()) + touch)]
    elif entry.rule == SEAT:
        patch = plan["patch"].reshape(-1, 3)
        seat = plan["seat"].reshape(-1, 3)
        # a fastener's shank crosses the host, so a long slide takes its head clear of the seat
        # and the contact is lost; the shorter slides keep it, and the same overhang is read
        for share in (0.5, 0.1, 0.02):  # of the seat's own extent, longest slide first
            for d in (plan["u"], -plan["u"], plan["v"], -plan["v"]):
                reach = float((patch @ d).max() - (seat @ d).max())
                out.append(d * (reach + _margin(touch, float(np.ptp(seat @ d)), share)))
    elif entry.rule == SINK:
        # into the host along the contact direction, to the depth `_sink_push` names, from where
        # the seat lies now (a seat off the surface by a gap goes the gap further)
        out = [plan["normal"] * max(_sink_push(piece, plan) - plan["level"], 0.0)]
    else:
        held = piece.by_name[plan["held"]]
        for d in np.vstack([np.eye(3), -np.eye(3)]):
            reach = float((held.tris.reshape(-1, 3) @ d).max() - (part.tris.reshape(-1, 3) @ d).min())
            out.append(d * (max(reach, 0.0) + CALIBRATION_FAIL * touch))
    return out


def _fault_item(piece: _Piece, part: PartMesh, entry: Entry) -> str:
    """The item the check reports once the part is moved, from its names and its geometry."""
    if entry.rule == REST:
        return piece.item(part, "tips off" if entry.plan["resting"] else "swings off",
                          entry.plan["host"].name)
    if entry.rule == FASTENER:  # pulled clear, a fastener holds nothing
        return piece.item(part, "engages", f"{entry.plan['host'].name} 0.0 mm")
    if entry.rule == SEAT:
        return piece.item(part, "overhangs", entry.plan["host"].name)
    if entry.rule == SINK:  # the depth the push puts the seat at (or the one it has, if deeper)
        deep = max(_sink_push(piece, entry.plan), entry.plan["level"])
        return piece.item(part, "sinks into", f"{entry.plan['host'].name} {deep * piece.mm:.1f} mm")
    return piece.item(part, "off", entry.plan["held"])


def _size(part: PartMesh) -> float:
    return float(np.linalg.norm(part.hi - part.lo))


def _sink_push(piece: _Piece, plan: dict) -> float:
    """How deep the sink plant puts the seat: `CALIBRATION_FAIL` contact lengths, but at most
    `SINK_PLANT_OF_CROSS` of the depth crossings fails at (`CROSS_OF_PART_THICKNESS`) of the
    thinner extent of the part and its host, so the seat keeps its contact. Geometry alone: no
    check compares it."""
    return min(CALIBRATION_FAIL * piece.touch,
               SINK_PLANT_OF_CROSS * CROSS_OF_PART_THICKNESS * plan["extent"])


def _unit(ctx: SceneCtx, view: PieceView, piece: _Piece, rule: str,
          planned: list[tuple[PartMesh, list[Entry]]]) -> PlantUnit | None:
    """The unit for one rule: on the smallest part the rule judges, every move that could make
    the check report that rule's item on it. None when the rule judges no part of the piece.
    Read from `_plan_part`'s geometry (`planned`, smallest part first); no rule is judged.

    The sink rule is planted on the smallest part whose push can be deeper than the contact length
    (`_sink_push`); failing that, on the part with the deepest push, which the gate then reports
    blind. A part whose plan has no finite `level` (no contact normal lies within 45 degrees of
    its contact direction) is not planted: its push would be infinite."""
    ready = [(part, entry) for part, entries in planned for entry in entries
             if entry.rule == rule and entry.status == "ready"
             and (rule != SINK or np.isfinite(entry.plan["level"]))]
    if rule == SINK and ready:
        deep = [pe for pe in ready if _sink_push(piece, pe[1].plan) > piece.touch]
        ready = deep or [max(ready, key=lambda pe: _sink_push(piece, pe[1].plan))]
    if not ready:
        return None
    part, entry = ready[0]
    item = _fault_item(piece, part, entry)
    geometry = {p.name: p.tris for p in view.parts}
    return PlantUnit(item, tuple(
        Plant(f"{rule}: {item}", item,
              pieces.with_geometry(ctx, view, geometry | {part.name: part.tris + move}))
        for move in _moves(piece, part, entry)))


# --- the check ------------------------------------------------------------------------------

NOT_APPLICABLE = {
    FASTENER: "no part is a fastener",
    RETAINER: "no part retains another",
}


class Bearing:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._seen: tuple[SceneCtx, dict[str, list[Entry]]] | None = None

    def _entries(self, ctx: SceneCtx) -> dict[str, list[Entry]]:
        """Every entry of every piece, kept for the scene it was made for."""
        if self._seen is None or self._seen[0] is not ctx:
            self._seen = (ctx, {v.key: measure(ctx, v)
                                for v in pieces.contact_pieces(ctx, self.kinds)})
        return self._seen[1]

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units = []
        for view in pieces.contact_pieces(ctx, self.kinds):
            piece = _Piece(ctx, view)
            planned = [(p, _plan_part(piece, p)) for p in sorted(
                (p for p in view.parts if p.solid and len(p.tris)), key=_size)]
            units += [unit for rule in RULES
                      if (unit := _unit(ctx, view, piece, rule, planned))]
        return units

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part with a verdict under some rule: a rule that is not applicable or could
        not be measured judges nothing."""
        entries = self._entries(ctx)
        return {item_of(view, e.part) for view in pieces.contact_pieces(ctx, self.kinds)
                for e in entries[view.key] if e.status in ("pass", "fail")}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found: list[Finding] = []
        if only is None:
            for entries in self._entries(ctx).values():
                found += [e.finding for e in entries if e.finding is not None]
            return found
        for view in pieces.contact_pieces(ctx, self.kinds, only):
            wanted = {p.name for p in view.parts
                      if any(i.startswith(f"{item_of(view, p.name)} ") for i in only)}
            found += [e.finding for e in measure(ctx, view, wanted)
                      if e.finding is not None and e.finding.item in only]
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = pieces.coverage(ctx, NAME, self.kinds, printed=True)
        lines += pieces.unviewed(ctx, NAME, self.kinds)
        for view in pieces.contact_pieces(ctx, self.kinds):
            entries = self._entries(ctx)[view.key]
            by_note: dict[tuple[str, str, bool], list[str]] = {}
            for e in entries:
                if e.status in ("na", "unmeasured"):
                    by_note.setdefault((e.rule, e.note, e.status == "unmeasured"), []
                                       ).append(e.part)
            for (rule, note, lost), names in sorted(by_note.items()):
                text = (f"{NAME}: {'skipped' if lost else 'not applicable'} ({note}; {rule} rule): "
                        f"{view.key} {', '.join(sorted(names))}")
                lines.append(Unmeasured(text) if lost else text)
            for rule, why in NOT_APPLICABLE.items():
                if not any(e.rule == rule for e in entries):
                    lines.append(f"{NAME}: not applicable ({why}; {rule} rule): {view.key}")
        return lines
