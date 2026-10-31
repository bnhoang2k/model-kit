"""Triangles from a skeleton: a tube for every limb, a folded blade and petiole for every leaf.

Each part is an (N, 3, 3) array of triangles in the skeleton's own axes and units. Winding gives
the face: tubes face outward, blades face along the leaf normal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from model_kit.core import edges
from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION
from model_kit.core.grow.buds import MESH_V1, MESH_VERSION
from model_kit.core.grow.shape import (
    CONE_POINT_OF_RADIUS,
    DOME_POINT_OF_RADIUS,
    DOME_RINGS,
    FILLET_RADIUS_OF_TUBE_RADIUS,
    FLARE_HEIGHT_OF_RADIUS,
    FLARE_MIN_GAP_OF_RADIUS,
    FLARE_RINGS,
    FLAT_EPS,
    LOBE_MAX_HALF_WIDTH_DEG,
    LOBE_OF_ROOT_RADIUS,
    LOBE_SPREAD,
    MERGE_OFFSET_OF_RADIUS,
    MERGE_RADIUS_STEP,
    PERP_EPS,
    PETIOLE_FLAT_EPS,
    SEAT_T_EPS,
    SEGMENT_EPS,
    SIDE,
    UP,
)
from model_kit.core.grow.skeleton import DOME, Skeleton, arc_lengths, unit
from model_kit.core.sight import Sight
from model_kit.core.units import Style

EMPTY = np.zeros((0, 3, 3))


def leaf_part_name(shade: str) -> str:
    return f"Leaves_{shade}"


# ------------------------------------------------------------------------------------- tubes


def _perp(d: np.ndarray) -> np.ndarray:
    s = np.cross(d, UP)
    return unit(s) if float(np.sqrt(s @ s)) >= PERP_EPS else SIDE.copy()


def _tangents(pts: np.ndarray) -> np.ndarray:
    """(n, 3) unit tangents, a central difference. A repeated point repeats the previous ring's
    tangent; leading ones take the first real tangent; a limb of one spot points up."""
    n = len(pts)
    raw = pts[np.minimum(np.arange(n) + 1, n - 1)] - pts[np.maximum(np.arange(n) - 1, 0)]
    out = np.zeros_like(raw)
    prev = None
    for i in range(n):
        d = unit(raw[i])
        if d @ d > 0.0:
            prev = d
        out[i] = prev if prev is not None else 0.0
    real = np.flatnonzero((out * out).sum(axis=1) > 0.0)
    if not len(real):
        out[:] = UP
    else:
        out[: real[0]] = out[real[0]]
    return out


def _frames(tans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Parallel-transported (normal, binormal) per ring."""
    n = len(tans)
    nrm = _perp(tans[0])
    ns, bs = np.empty((n, 3)), np.empty((n, 3))
    for i in range(n):
        t = tans[i]
        v = nrm - t * (nrm @ t)
        nrm = unit(v) if v @ v > 0.0 else _perp(t)
        ns[i], bs[i] = nrm, unit(np.cross(t, nrm))
    return ns, bs


def _ring(centre, nrm, bi, radius, cos, sin) -> np.ndarray:
    """(s, 3) corners of a ring about `centre`."""
    return centre + (nrm * cos[:, None] + bi * sin[:, None]) * radius


def _band(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Triangles of the quad strip between rings a and b (both (s, 3)): 2 s of them."""
    a1, b1 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
    return np.concatenate([np.stack([a, a1, b1], axis=1), np.stack([a, b1, b], axis=1)])


def _cap(ring: np.ndarray, tip: np.ndarray) -> np.ndarray:
    """A fan from a point to a ring: s triangles."""
    return np.stack([ring, np.roll(ring, -1, axis=0), np.broadcast_to(tip, ring.shape)], axis=1)


@dataclass(frozen=True)
class Lobe:
    """A root flare lobe on a limb's lowest rings, in the limb's own units."""

    direction: np.ndarray  # unit vector, the way the root leaves
    seat_s: float  # arc length of the root's seat from the limb's base
    height: float  # the lobe is full up to seat_s and gone at seat_s + height
    extra: float  # how far past the round tube it reaches, at its fullest
    half_width: float  # radians about the axis


def _lobed_radii(pts, radii, ns, bs, sides: int, lobes) -> np.ndarray:
    """(rings, sides) corner radii: the ring's radius, pushed out by each lobe's cos squared bump
    about its root's azimuth in the ring's own frame, scaled by the smoothstep fall with height."""
    s = arc_lengths(pts)
    ang = 2.0 * np.pi * np.arange(sides) / sides
    out = np.repeat(radii[:, None], sides, axis=1)
    for lobe in lobes:
        x = np.clip((s - lobe.seat_s) / lobe.height, 0.0, 1.0)
        fall = 1.0 - x * x * (3.0 - 2.0 * x)
        u, v = ns @ lobe.direction, bs @ lobe.direction
        round_here = np.hypot(u, v) > FLAT_EPS  # a ring seen along the root has no side to bulge
        gap = np.abs((ang[None, :] - np.arctan2(v, u)[:, None] + np.pi) % (2.0 * np.pi) - np.pi)
        bump = np.cos(0.5 * np.pi * np.minimum(gap / lobe.half_width, 1.0)) ** 2
        out = np.maximum(out, radii[:, None] + lobe.extra * fall[:, None] * bump
                         * round_here[:, None])
    return out


def _rings(pts: np.ndarray, radii: np.ndarray, sides: int, lobes=None):
    """(the rings of `sides` corners along the points, their tangents, normals and binormals, and
    the corners' cos and sin): the wall `tube_tris` closes and `_over_gap_bands` reads."""
    ang = 2.0 * np.pi * np.arange(sides) / sides
    cos, sin = np.cos(ang), np.sin(ang)
    tans = _tangents(pts)
    ns, bs = _frames(tans)
    if lobes:
        reach = _lobed_radii(pts, radii, ns, bs, sides, lobes)
        rings = [_ring(pts[i], ns[i], bs[i], reach[i][:, None], cos, sin) for i in range(len(pts))]
    else:
        rings = [_ring(pts[i], ns[i], bs[i], radii[i], cos, sin) for i in range(len(pts))]
    return rings, tans, ns, bs, cos, sin


def _over_gap_bands(pts: np.ndarray, radii: np.ndarray, sides: int, sight: Sight) -> set[int]:
    """The bands (band j runs from ring j to ring j + 1) of a round tube's wall, with no caps, on
    which `detail` would read an edge as over the gap: `edges.sagittas` over `sight.gap`, among the
    edges that turn by less than the crease angle. The rings share their corners exactly, so they
    are welded at the float guard alone."""
    rings = _rings(pts, radii, sides)[0]
    tris = np.concatenate([_band(a, b) for a, b in itertools.pairwise(rings)])
    found = edges.interior_edges(tris, NO_DIRECTION)
    over = (found.turn_deg < sight.crease_deg) & (edges.sagittas(tris, found) > sight.gap)
    return {int(i) for i in np.unique(found.faces[over] // (2 * sides))}


def tube_tris(pts, radii, sides: int, tip: int, lobes=None) -> np.ndarray:
    """Tube mesh: rings of `sides` corners along the points, a base cap fanned from
    its first corner, and a cone or dome tip. Faces point outward. With `lobes` (root flare) the
    rings bulge toward each lobe and the base cap fans from the ring's centre, because a lobed
    ring is no longer convex."""
    pts = np.asarray(pts, dtype=float)
    radii = np.asarray(radii, dtype=float)
    rings, tans, ns, bs, cos, sin = _rings(pts, radii, sides, lobes)
    parts = [_band(a, b) for a, b in itertools.pairwise(rings)]
    base = rings[0][::-1]  # reversed: the cap faces back along the limb
    if lobes:
        # The same fan (apex first), with the apex at the ring's centre.
        parts.append(_cap(base, pts[0])[:, [2, 0, 1]])
    else:
        parts.append(np.stack([np.broadcast_to(base[0], (sides - 2, 3)), base[1:-1], base[2:]],
                              axis=1))
    last, t, r = rings[-1], tans[-1], radii[-1]
    if tip == DOME:
        for ring_r, advance in DOME_RINGS:
            ring = _ring(pts[-1] + t * r * advance, ns[-1], bs[-1], r * ring_r, cos, sin)
            parts.append(_band(last, ring))
            last = ring
        point = pts[-1] + t * r * DOME_POINT_OF_RADIUS
    else:
        point = pts[-1] + t * r * CONE_POINT_OF_RADIUS
    parts.append(_cap(last, point))
    return np.concatenate(parts)


# ------------------------------------------------------------------------- the bud-grown mesh


def _seated_points(pts: np.ndarray, seats) -> np.ndarray:
    """(n,) bool: the points a child seated at an arc fraction of `seats` stands on. A seat on a
    point marks that point; a seat between two points marks both."""
    cum = arc_lengths(pts)
    frac = cum / cum[-1] if cum[-1] > 0.0 else np.zeros(len(pts))
    mark = np.zeros(len(pts), dtype=bool)
    for t in np.asarray(seats, dtype=float).reshape(-1):
        nearest = int(np.argmin(np.abs(frac - t)))
        if abs(frac[nearest] - t) <= SEAT_T_EPS:
            mark[nearest] = True
        else:
            above = int(np.searchsorted(frac, t))
            mark[max(above - 1, 0)] = mark[min(above, len(pts) - 1)] = True
    return mark


def _merge_collinear(pts: np.ndarray, radii: np.ndarray, seated: np.ndarray,
                     within: float | None = None) -> np.ndarray:
    """Indexes of the points to keep, ascending. From a kept anchor, the run is extended while
    every skipped point lies within MERGE_OFFSET_OF_RADIUS of its radius from the anchor-to-end
    chord, its radius is within MERGE_RADIUS_STEP of the chord's, and none is seated. Measuring
    from the anchor (not turn by turn) stops a gentle curve of many small turns from being flattened.
    The first and last points stay. Fixed order, no random choice.

    With `within` (the buds-v2 mesh: the piece's seen floor in the skeleton's units) both
    tolerances are that one length instead of a share of the radius: a point goes when dropping it
    moves the surface by less than the eye resolves."""
    n = len(pts)
    kept = [0]
    a = 0
    while a < n - 1:
        e = a + 1
        while e + 1 < n and _spans(pts, radii, seated, a, e + 1, within):
            e += 1
        kept.append(e)
        a = e
    return np.array(kept, dtype=int)


def _spans(pts, radii, seated, a: int, e: int, within: float | None = None) -> bool:
    """Whether the points strictly between a and e can all go."""
    mid = slice(a + 1, e)
    if seated[mid].any():
        return False
    ab = pts[e] - pts[a]
    den = max(float(ab @ ab), SEGMENT_EPS)
    f = np.clip(((pts[mid] - pts[a]) @ ab) / den, 0.0, 1.0)
    off = np.sqrt(((pts[a] + f[:, None] * ab - pts[mid]) ** 2).sum(axis=1))
    r = radii[mid]
    step = np.abs(radii[a] + f * (radii[e] - radii[a]) - r)
    if within is not None:
        return bool((off <= within).all() and (step <= within).all())
    return bool((off <= MERGE_OFFSET_OF_RADIUS * r).all() and (step <= MERGE_RADIUS_STEP * r).all())


def _fillet(d_in: np.ndarray, d_out: np.ndarray, node: np.ndarray, rho: float, tau: float,
            count: int) -> np.ndarray:
    """(count + 1, n) points of the arc of radius `rho` that rounds the kink at `node`, where the
    path turns by `tau` radians from the unit direction `d_in` to `d_out`: it leaves the incoming
    segment `rho * tan(tau / 2)` before the node, turns through `tau` in `count` equal chords and
    joins the outgoing segment the same distance after it. The vectors are of any one length n."""
    reach = rho * np.tan(tau / 2.0)
    inward = unit(d_out - d_in * (d_in @ d_out))  # across the incoming segment, toward the bend
    first = node - d_in * reach
    centre = first + inward * rho
    theta = tau * np.arange(count + 1) / count
    return centre + rho * (np.outer(-np.cos(theta), inward) + np.outer(np.sin(theta), d_in))


def _round_bends(pts: np.ndarray, radii: np.ndarray, sight: Sight) -> tuple[np.ndarray, np.ndarray]:
    """The axis and radii with each kink that would show rounded where it can be (the buds-v2
    mesh), and kept as a sharp crease where it cannot.

    A tube's wall turns at a ring where the axis bends (`bend`) and where its radius changes slope
    (`step`, the change of the wall's lean; the step of the pipe model at a seat is one): the edge
    there turns by about the two together. Two bands of length `w` that meet at a turn `t` leave a
    joint whose gap to the curve is `edges.chord_sagitta(w, t)`, which the eye sees when it is over
    `sight.gap`; a turn of `sight.crease_deg` or more draws a line. Either is rounded (`_blend`):
    the axis by an arc in space that leaves each band the same reach from the node, and the radius
    by a curve along that arc that meets both bands' radius and lean.

    A tube swept along a curve folds its inner wall onto itself where the curve bends tighter
    than the tube is thick: the rings' planes cross at the centre of curvature, so a ring wider
    than that radius crosses its neighbour. The radius is not a direction in space, so only the
    axis's bend can fold the wall: an arc of reach `x` along bands that turn by `bend` has a radius
    of `x / tan(bend / 2)`, and it folds unless that is over the thickest ring it carries. A kink
    kept as it is folds when a ring stands within `r sin(bend / 2)` of it (its own ring lies in the
    plane between the bands). Two arcs on one band may not overlap, so the bands are shared
    (`_share_room`): first every kink gets the reach below which it folds, then the room left grows
    every arc toward the `FILLET_RADIUS_OF_TUBE_RADIUS` it wants.

    A bend whose bands cannot give its arc that reach is not rounded: it stays a sharp crease, as a
    tree's limb can turn sharply where it grew so (a tree is not a hose bent smooth), and the room
    goes back to its neighbours. When several arcs fold at once, only the one furthest short is
    creased before the bands are shared again, so a bend that can be rounded once its neighbour
    is kept is rounded. A crease's edges are judged by `detail` like any other edge.

    The wall is then read as `detail` reads it (`_over_gap_bands`, on the tube's own sides). An
    edge over the gap adds a chord to the arc it lies on, or rounds the kink it lies beside (a kink
    judged quiet on whole bands may not be once an arc has taken one), until none shows; an edge
    beside a crease is the crease's, and is left as it is."""
    if len(pts) < 3:
        return pts, radii
    seg = pts[1:] - pts[:-1]
    length = np.sqrt((seg * seg).sum(axis=1))
    rise = radii[1:] - radii[:-1]
    lean = np.arctan2(rise, length)  # the wall's lean against the axis, per band
    sides = tube_sides(float(radii.max()), sight)
    # node: (bend, turn in degrees, thickest ring)
    kinks: dict[int, tuple[float, float, float]] = {}
    rounded: set[int] = set()
    for k in range(1, len(pts) - 1):
        a, b = length[k - 1], length[k]
        if not (a > 0.0 and b > 0.0):
            continue
        bend = _turn(seg[k - 1], seg[k])
        turn = float(np.degrees(bend + abs(float(lean[k] - lean[k - 1]))))
        if not turn > 0.0:
            continue
        kinks[k] = (bend, turn, float(radii[k - 1:k + 2].max()))
        if turn >= sight.crease_deg or edges.chord_sagitta(min(a, b), turn) > sight.gap:
            rounded.add(k)
    creases: set[int] = set()
    counts: dict[int, int] = {}
    while True:
        path, owner, folds = _rounded(pts, radii, seg, length, rise, kinks, rounded, counts, sight)
        if folds:
            # Crease one fold, the one furthest short of the reach it needs, and share the bands
            # again: the room it gives back may let the others round (two bends that crowd one
            # band both fall short together, though one alone fits). Each such pass moves one
            # node out of `rounded` for good, so these passes end.
            worst = max(folds, key=lambda k: (folds[k], kinks[k][0], -k))
            rounded.discard(worst)
            creases.add(worst)
            continue
        finer, more = set(), set()  # arcs that take a chord more, kinks to round
        for j in _over_gap_bands(*path, sides, sight) if kinks else ():
            here = {int(owner[j]), int(owner[j + 1])}
            if not here & creases:
                finer |= here & rounded
                more |= (here & set(kinks)) - rounded
        if not (finer or more):
            return path
        for k in finer:
            counts[k] += 1
        rounded |= more


def _rounded(pts, radii, seg, length, rise, kinks, rounded, counts, sight):
    """((points, radii), the node each row belongs to, and each kink in `rounded` whose arc would
    fold with how far its reach falls short of `r tan(bend / 2)`, r its thickest ring) of a limb
    with the kinks in `rounded` blended and the others kept (`_round_bends`). `counts` holds each
    blend's chords, and gets the starting count of a new one."""
    n = len(pts)
    want, least = np.zeros(n), np.zeros(n)
    for k, (bend, turn, thick) in kinks.items():
        a, b = length[k - 1], length[k]
        if k in rounded:
            least[k] = min(thick * np.tan(bend / 2.0), a, b)
            # an arc never reaches past either band, so a turn near 180 degrees asks for all of both
            half = 0.5 * min(np.radians(turn), np.pi)
            want[k] = min(max(FILLET_RADIUS_OF_TUBE_RADIUS * radii[k] * np.tan(half), least[k]),
                          a, b)
        else:
            least[k] = want[k] = min(thick * np.sin(bend / 2.0), a, b)
    reach = _share_room(length, least, np.zeros(n))
    reach = _share_room(length, np.where(reach < least, reach, want), reach)
    arc = np.array([reach[k] if k in rounded else 0.0 for k in range(n)])  # kept kinks reach none
    out_p, out_r, owner, folds = [pts[:1]], [radii[:1]], [0], {}
    for k in range(1, n - 1):
        if k not in rounded:
            out_p.append(pts[k:k + 1])
            out_r.append(radii[k:k + 1])
            owner.append(k)
            continue
        bend, turn, _ = kinks[k]
        a, b = length[k - 1], length[k]
        rho = arc[k] / np.tan(bend / 2.0) if bend > 0.0 else np.inf
        if k not in counts:
            span = rho * bend if bend > 0.0 else 2.0 * arc[k]
            counts[k] = max(edges.segments_for((span + kinks[k][2] * bend) / np.radians(turn),
                                               turn, sight.gap),
                            int(turn // sight.crease_deg) + 1)
        p, r = _blend(pts[k], seg[k - 1] / a, seg[k] / b, bend, float(arc[k]), float(radii[k]),
                      float(rise[k - 1] / a), float(rise[k] / b), counts[k])
        if not (rho > r.max() and r.min() > 0.0):
            folds[k] = max(float(r.max()) * np.tan(bend / 2.0) - float(arc[k]), 0.0)
        out_p.append(p)
        out_r.append(r)
        owner.extend([k] * len(p))
    out_p.append(pts[-1:])
    out_r.append(radii[-1:])
    owner.append(n - 1)
    path, radius, owner = np.concatenate(out_p), np.concatenate(out_r), np.array(owner)
    # Where the arcs on a band take all of it (or an arc takes a whole band up to a kept node),
    # the ring that ends one is the ring that starts the next: what is left of the band has no
    # direction, so it is one ring, and the node's own ring is the one kept.
    full = length - arc[:-1] - arc[1:] <= NO_DIRECTION
    first = np.concatenate([[True], owner[1:] != owner[:-1]])  # each item's first row
    same = np.zeros(len(path), dtype=bool)
    for i in np.flatnonzero(first)[1:]:
        if full[owner[i] - 1]:
            same[i if owner[i] in rounded else i - 1] = True
    return (path[~same], radius[~same]), owner[~same], folds


def _share_room(length: np.ndarray, want: np.ndarray, base: np.ndarray) -> np.ndarray:
    """(n,) how far along its two bands each node's arc reaches: from `base` toward `want` (both 0
    for a node with no arc, and for both ends). Band j runs from node j to node j + 1, and the two
    arcs on it may not reach further than its `length` together. Every arc grows by the same share
    of what it still wants until a band is full: the arcs on that band stop there, and the others
    grow on. So an arc takes the whole band when its neighbour needs none of it, and half of it (as
    two equal arcs would) when both need more than half; no arc stops while its neighbour leaves
    room unused."""
    reach = base.astype(float)
    growing = want > base
    while growing.any():
        share, full = 1.0, None
        for j, room in enumerate(length):
            ends = [k for k in (j, j + 1) if growing[k]]
            if ends:
                taken = sum(reach[k] for k in (j, j + 1))  # a growing end still holds its base
                here = (room - taken) / sum(want[k] - base[k] for k in ends)
                if here < share:
                    share, full = here, ends
        if full is None:
            reach[growing] = want[growing]
            break
        for k in full:
            reach[k], growing[k] = base[k] + share * (want[k] - base[k]), False
    return reach


def _blend(node: np.ndarray, d_in: np.ndarray, d_out: np.ndarray, bend: float, reach: float,
           radius: float, slope_in: float, slope_out: float, count: int
           ) -> tuple[np.ndarray, np.ndarray]:
    """(count + 1 points, their radii) of the rings that round one kink. The axis leaves the
    incoming band `reach` before `node` and runs on an arc (`_fillet`; a straight run when the axis
    does not bend) to the outgoing band, `reach` after it. The radius along it is the cubic that
    starts and ends on the bands' radii with their slopes (`slope_in`, `slope_out`, per unit of
    length), so the wall has no corner where the rings meet the bands."""
    t = np.arange(count + 1) / count
    if bend > 0.0:
        rho = reach / np.tan(bend / 2.0)
        p, span = _fillet(d_in, d_out, node, rho, bend, count), rho * bend
    else:
        p, span = node + np.outer(2.0 * reach * t - reach, d_in), 2.0 * reach
    r0, r1 = radius - slope_in * reach, radius + slope_out * reach
    r = ((2 * t**3 - 3 * t**2 + 1) * r0 + (t**3 - 2 * t**2 + t) * span * slope_in
         + (-2 * t**3 + 3 * t**2) * r1 + (t**3 - t**2) * span * slope_out)
    return p, r


def _turn(u: np.ndarray, v: np.ndarray) -> float:
    """The angle between two vectors in radians; 0 when either has no length."""
    nu, nv = float(np.sqrt(u @ u)), float(np.sqrt(v @ v))
    return float(np.arccos(np.clip(u @ v / (nu * nv), -1.0, 1.0))) if nu > 0.0 and nv > 0.0 else 0.0


def _project(pts: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray, float]:
    """(arc length of the polyline point nearest q, the unit tangent there, and that point's
    position as segment index plus fraction). The first closest segment wins a tie."""
    a, ab = pts[:-1], pts[1:] - pts[:-1]
    den = np.maximum((ab * ab).sum(axis=1), SEGMENT_EPS)
    f = np.clip(((q - a) * ab).sum(axis=1) / den, 0.0, 1.0)
    dist = ((a + ab * f[:, None] - q) ** 2).sum(axis=1)
    k = int(np.argmin(dist))
    cum = arc_lengths(pts)
    return float(cum[k] + f[k] * np.sqrt(den[k])), unit(ab[k]), float(f[k] + k)


def _lobes(sk: Skeleton, pts, radii, roots) -> list[Lobe]:
    """One lobe per buried child in `roots`, sized by the child's base radius against the limb's
    radius at the seat. A root leaving along the axis has no side to bulge toward: no lobe."""
    out = []
    for c in roots:
        c_pts, c_radii = sk.limb_points(int(c)), sk.limb_radii(int(c))
        seat_s, tangent, where = _project(pts, c_pts[0])
        k = min(int(where), len(pts) - 2)
        r_limb = float(radii[k] + (radii[k + 1] - radii[k]) * (where - k))
        out_of = np.sqrt(((c_pts - c_pts[0]) ** 2).sum(axis=1))
        if not (out_of > 0.0).any():
            continue
        d = unit(c_pts[int(np.argmax(out_of > 0.0))] - c_pts[0])  # the first step that goes anywhere
        if float(np.sqrt(((d - tangent * (d @ tangent)) ** 2).sum())) <= FLAT_EPS:
            continue
        r_root = float(c_radii[0])
        out.append(Lobe(
            direction=d, seat_s=seat_s, height=FLARE_HEIGHT_OF_RADIUS * r_limb,
            extra=LOBE_OF_ROOT_RADIUS * r_root,
            half_width=min(LOBE_SPREAD * float(np.arctan2(r_root, r_limb)),
                           float(np.radians(LOBE_MAX_HALF_WIDTH_DEG)))))
    return out


def _with_flare_rings(pts, radii, lobes, steps: int = FLARE_RINGS, apart: float | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """The polyline with rings added across each lobe's height (`steps` of them from its seat to
    its top, FLARE_RINGS in buds-v1), so the flare has a profile even where the limb has few
    points. A ring goes in unless it lies within FLARE_MIN_GAP_OF_RADIUS of the radius there of
    another (in buds-v2, within `apart`, the length the eye resolves)."""
    cum = arc_lengths(pts)
    have = list(cum)
    for lobe in lobes:
        for j in range(steps + 1):
            s = lobe.seat_s + lobe.height * j / steps
            gap = apart if apart is not None else (
                FLARE_MIN_GAP_OF_RADIUS * float(np.interp(s, cum, radii)))
            if 0.0 < s < cum[-1] and all(abs(s - h) > gap for h in have):
                have.append(s)
    have.sort()
    arcs = np.array(have)
    new_pts = np.stack([np.interp(arcs, cum, pts[:, k]) for k in range(3)], axis=1)
    return new_pts, np.interp(arcs, cum, radii)


# ------------------------------------------------------------------------------------- leaves


def leaf_triangles(node, base, mid, across, normal, length, *, outline, fold, blade_centre,
                   petiole_tip, petiole_width, lite: bool) -> np.ndarray:
    """Triangles of m leaves given as arrays (`node`, `base`, `mid`, `across`, `normal`: (m, 3);
    `length`: (m,)) and the skeleton's mesh settings, blades first then petioles:
    (m * per_leaf, 3, 3). `_leaf_batch` reads the arrays from a skeleton; the grower calls this
    before its skeleton exists, to keep leaves out of the soil (`buds._leaves`)."""
    node, base = np.asarray(node, dtype=float), np.asarray(base, dtype=float)
    mid, across = np.asarray(mid, dtype=float), np.asarray(across, dtype=float)
    normal = np.asarray(normal, dtype=float)
    outline, blade_centre = np.asarray(outline, dtype=float), np.asarray(blade_centre, dtype=float)
    k = len(outline)
    length = np.asarray(length, dtype=float)[:, None]  # (m, 1)
    x, y = outline[:, 0], outline[:, 1]
    rim_local = np.stack([x, y, fold * np.abs(x)], axis=1)  # (k, 3): across, along, up
    local = np.concatenate([rim_local, blade_centre[None]])  # centre last

    def place(local_pts):
        # (m, p, 3) = base + across * x L + mid * y L + normal * z L
        return (
            base[:, None, :]
            + across[:, None, :] * local_pts[None, :, 0, None] * length[:, :, None]
            + mid[:, None, :] * local_pts[None, :, 1, None] * length[:, :, None]
            + normal[:, None, :] * local_pts[None, :, 2, None] * length[:, :, None]
        )

    verts = place(local)  # (m, k + 1, 3)
    rim, centre = verts[:, :k], verts[:, k]
    blade = np.stack(
        [np.broadcast_to(centre[:, None], rim.shape), rim, np.roll(rim, -1, axis=1)], axis=2
    )  # (m, k, 3, 3)
    if lite:
        return blade.reshape(-1, 3, 3)
    a = node
    b = base + mid * (petiole_tip * length)
    d = b - a
    side = np.cross(d, UP)
    flat = np.sqrt((side * side).sum(axis=1)) > PETIOLE_FLAT_EPS
    side_len = np.maximum(np.linalg.norm(side, axis=1), PETIOLE_FLAT_EPS)
    side = np.where(flat[:, None], side / side_len[:, None], across)
    side = side * petiole_width
    q0, q1, q2, q3 = a - side, a + side, b + side, b - side
    quad = np.stack([np.stack([q0, q1, q2], axis=1), np.stack([q0, q2, q3], axis=1)], axis=1)
    # Face the same way as the blade.
    face = np.cross(q1 - q0, q2 - q1)
    flip = (face * normal).sum(axis=1) < 0.0
    quad[flip] = quad[flip][:, :, ::-1]
    return np.concatenate([blade, quad], axis=1).reshape(-1, 3, 3)


def _leaf_batch(sk: Skeleton, idx: np.ndarray) -> np.ndarray:
    """Triangles of the leaves `idx`, blades first then petioles: (len(idx) * per_leaf, 3, 3)."""
    idx = np.asarray(idx, dtype=int)
    return leaf_triangles(sk.node[idx], sk.base[idx], sk.mid[idx], sk.across[idx],
                          sk.normal[idx], sk.length[idx], outline=sk.outline, fold=sk.fold,
                          blade_centre=sk.blade_centre, petiole_tip=sk.petiole_tip,
                          petiole_width=sk.petiole_width, lite=sk.lite)


def leaf_tris(sk: Skeleton, i: int) -> np.ndarray:
    """Leaf mesh: a fan from `blade_centre` to the outline, the halves folded
    `fold * |x|` up the normal, plus the petiole ribbon as 2 triangles unless `sk.lite`."""
    return _leaf_batch(sk, np.array([i]))


def all_leaf_tris(sk: Skeleton) -> np.ndarray:
    """The triangles of every leaf, in leaf order, as `leaf_tris` makes them: (n, 3, 3)."""
    return _leaf_batch(sk, np.arange(len(sk.host)))


# ------------------------------------------------------------------------------------- parts


def _children(sk: Skeleton) -> list[list[int]]:
    kids: list[list[int]] = [[] for _ in range(len(sk.limb_id))]
    for c, p in enumerate(sk.parent):
        if p >= 0:
            kids[int(p)].append(c)
    return kids


def needs_sight(sk: Skeleton) -> bool:
    """Whether `grown_parts` reads the piece's sight for `sk`: only the buds-v2 mesh does."""
    return sk.info.get("mesh") == MESH_VERSION


def tube_sides(radius: float, sight: Sight) -> int:
    """The corners per ring of a buds-v2 tube whose thickest ring is `radius`: the fewest whose gap
    to the circle is under `sight.gap` (`edges.segments_for`), and never fewer than make it round
    (`round_sides`)."""
    return max(round_sides(sight.crease_deg), edges.segments_for(radius, 360.0, sight.gap))


def round_sides(crease_deg: float) -> int:
    """The fewest sides a round tube has: more than `360 / crease_deg`, so the turn between two of
    its faces is under the crease angle and no line is drawn along it."""
    return int(360.0 // crease_deg) + 1


def _lobed_gaps(pts, radii, lobes, sides: int, crease_deg: float) -> tuple[float, float]:
    """The largest gap, at any smooth edge of a tube of `sides` corners with `lobes`, between the
    two flat faces that meet there and the curve they stand for (`edges.chord_sagitta` of the
    narrower face and the edge's turn, as `detail` reads them): first the girth (the edges that run
    from ring to ring, which more sides lower), then the profile (the ring edges and the quads'
    diagonals, which more rings lower). The
    rings are cut as `tube_tris` cuts them, so a lobe's outline, the wall's profile and the axis's
    own bend are all in the numbers. Edges that turn by the crease angle or more draw a line and
    are not judged."""
    ns, bs = _frames(_tangents(pts))
    ang = 2.0 * np.pi * np.arange(sides) / sides
    reach = _lobed_radii(pts, radii, ns, bs, sides, lobes)  # (rings, sides)
    ring = (ns[:, None, :] * np.cos(ang)[None, :, None]
            + bs[:, None, :] * np.sin(ang)[None, :, None]) * reach[..., None]
    rings = len(pts)
    points = (pts[:, None, :] + ring).reshape(-1, 3)
    at = np.arange(rings * sides).reshape(rings, sides)

    def gap(v: np.ndarray, wings: np.ndarray) -> float:
        """The worst smooth gap of edges `v` (E, 2) with a face wing on each side (E, 2)."""
        turn = np.nan_to_num(edges.turns_deg(points, v, wings))
        edge = points[v[:, 1]] - points[v[:, 0]]
        length = np.linalg.norm(edge, axis=1)
        high = np.stack([np.linalg.norm(np.cross(edge, points[w] - points[v[:, 0]]), axis=1)
                         for w in wings.T], axis=1) / np.maximum(length, NO_DIRECTION)[:, None]
        sag = edges.chord_sagitta(high.min(axis=1), turn)
        return float(sag[turn < crease_deg].max(initial=0.0))

    nxt, prev = np.roll(at, -1, axis=1), np.roll(at, 1, axis=1)
    girth = gap(np.stack([at[:-1].ravel(), at[1:].ravel()], axis=1),
                np.stack([prev[:-1].ravel(), nxt[1:].ravel()], axis=1))
    across = gap(np.stack([at[1:-1].ravel(), nxt[1:-1].ravel()], axis=1),
                 np.stack([nxt[2:].ravel(), at[:-2].ravel()], axis=1))
    diagonal = gap(np.stack([at[:-1].ravel(), nxt[1:].ravel()], axis=1),
                   np.stack([at[1:].ravel(), nxt[:-1].ravel()], axis=1))
    return girth, max(across, diagonal)


def _lobed_rings(pts, radii, lobes, sight: Sight, who: str
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    """The rings and the corners per ring of a buds-v2 tube with root lobes: the flare is cut in
    `FLARE_RINGS` steps and the ring in the sides a plain tube of its thickest ring needs, and
    each is raised by one while a lobe's outline (more sides) or profile (more steps) still shows
    a gap. No ring goes in within the gap of another, so once every lobe's steps are that short
    more steps cannot help; the profile is then still over the gap, and `who` (the piece) is named
    in the error: a tube over what the eye resolves is never returned."""
    sides = tube_sides(float(radii.max()), sight)
    steps = FLARE_RINGS
    while True:
        rings = _with_flare_rings(pts, radii, lobes, steps, sight.gap)
        girth, profile = _lobed_gaps(*rings, lobes, sides, sight.crease_deg)
        if girth > sight.gap:
            sides += 1
        elif profile <= sight.gap:
            return *rings, sides
        elif all(lobe.height / steps <= sight.gap for lobe in lobes):
            raise ManifestError(
                f"{who}: a root lobe's flare still shows a gap of {profile:.3g} between its faces "
                f"and the curve (over the {sight.gap:.3g} the eye resolves) with its rings "
                "already no further apart than that; the piece cannot be meshed to its view")
        else:
            steps += 1


def _limb_tris(sk: Skeleton, i: int, kids: list[int], sight: Sight | None) -> np.ndarray:
    mesh = sk.info.get("mesh")
    if mesh is None:  # the original mesh: one tube, every point a ring
        return tube_tris(sk.limb_points(i), sk.limb_radii(i), int(sk.sides[sk.level[i]]),
                         int(sk.tip[i]))
    if mesh == MESH_V1:
        return _limb_tris_v1(sk, i, kids)
    if mesh == MESH_VERSION:
        return _limb_tris_v2(sk, i, kids, sight)
    raise ManifestError(f"skeleton {sk.info.get('piece')!r}: unknown mesh version {mesh!r}; this "
                        f"kit meshes {MESH_V1!r}, {MESH_VERSION!r} and the original mesh")


def _merged(sk: Skeleton, i: int, kids: list[int], within: float | None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Limb i's points and radii with the collinear points that seat nothing merged away."""
    pts, radii = sk.limb_points(i), sk.limb_radii(i)
    seated = _seated_points(pts, sk.t[kids]) if kids else np.zeros(len(pts), dtype=bool)
    keep = _merge_collinear(pts, radii, seated, within)
    return pts[keep], radii[keep]


def _limb_tris_v1(sk: Skeleton, i: int, kids: list[int]) -> np.ndarray:
    """Buds-v1: collinear points that seat nothing are merged (by a share of the radius), each
    buried child (a root) pulls a lobe out of this limb's lowest rings, and the sides are
    `Skeleton.sides`."""
    pts, radii = _merged(sk, i, kids, None)
    lobes = _lobes(sk, pts, radii, [c for c in kids if sk.buried[c]])
    if lobes:
        pts, radii = _with_flare_rings(pts, radii, lobes)
    return tube_tris(pts, radii, int(sk.sides[sk.level[i]]), int(sk.tip[i]), lobes)


def _limb_tris_v2(sk: Skeleton, i: int, kids: list[int], sight: Sight | None) -> np.ndarray:
    """Buds-v2: the same tube, but every length that sets how finely it is cut is what the eye
    resolves (`sight.gap`): collinear points go when dropping them moves the wall by less, a kink
    that would show is rounded where its bands leave room for an arc wider than the tube, and kept
    as the skeleton's sharp crease where they do not (`_round_bends`), the corners per ring are
    the fewest whose gap to the curve is under it (`edges.segments_for`, or `_lobed_rings` where
    roots pull lobes), and never fewer than make it round (`round_sides`)."""
    if sight is None:
        raise ManifestError(f"skeleton {sk.info.get('piece')!r}: the {MESH_VERSION} mesh takes "
                            "its tube sides from the piece's view, and was given no sight (the "
                            "piece has no view, or the style no crease_deg)")
    pts, radii = _round_bends(*_merged(sk, i, kids, sight.gap), sight)
    lobes = _lobes(sk, pts, radii, [c for c in kids if sk.buried[c]])
    if lobes:
        who = f"skeleton {sk.info.get('piece')!r}, limb {sk.limb_id[i]}"
        pts, radii, sides = _lobed_rings(pts, radii, lobes, sight, who)
    else:
        sides = tube_sides(float(radii.max()), sight)
    return tube_tris(pts, radii, sides, int(sk.tip[i]), lobes)


def limb_tris(sk: Skeleton, i: int, sight: Sight | None) -> np.ndarray:
    """The triangles of limb i's tube, as `grown_parts` makes them."""
    return _limb_tris(sk, i, [int(c) for c in np.flatnonzero(sk.parent == i)], sight)


def part_names(sk: Skeleton) -> list[str]:
    """The names of the parts `grown_parts` makes, in its order, without making them."""
    return [*(name for li, name in enumerate(sk.level_names) if (sk.level == li).any()),
            *(leaf_part_name(shade) for si, shade in enumerate(sk.shades)
              if (sk.shade == si).any())]


def grown_parts(sk: Skeleton, sight: Sight | None) -> dict[str, np.ndarray]:
    """One part per level name and then per leaf shade; a level or shade with nothing is left out.

    A skeleton whose `info["mesh"]` is "buds-v2" (the bud grower's, from a rules file without
    `tubes.sides`) takes its tube sides and its axis rings from `sight`, the piece's seen floor
    and the style's crease angle: a tube has the fewest segments whose gap to the curve the eye
    cannot see (`edges.segments_for`), and never fewer than make it round (`round_sides`). It
    needs a `sight`; a missing one raises. A "buds-v1" skeleton (a rules file with `tubes.sides`)
    gets the lobed root flare and the collinear point merge, with `Skeleton.sides`. Every other
    skeleton keeps the original mesh exactly. Neither older mesh reads `sight`."""
    parts: dict[str, np.ndarray] = {}
    kids = _children(sk)
    for li, name in enumerate(sk.level_names):
        tris = [_limb_tris(sk, int(i), kids[int(i)], sight) for i in np.flatnonzero(sk.level == li)]
        if tris:
            parts[name] = np.concatenate(tris)
    for si, shade in enumerate(sk.shades):
        idx = np.flatnonzero(sk.shade == si)
        if len(idx):
            parts[leaf_part_name(shade)] = _leaf_batch(sk, idx)
    return parts


@dataclass(frozen=True)
class LeafLean:
    """How leaf normals are shaded. A render choice, read from style.json `dials.leaves`.

    `toward_out`: the share of each normal that points outward from the crown centre (0 keeps the
    face normal, 1 is fully radial). `centre_drop`: the crown centre sits this share of the leaf
    box height below the box middle.
    """

    toward_out: float
    centre_drop: float


def leaf_lean(style: Style) -> LeafLean:
    """The leaf-normal lean of a style. A style with no `dials.leaves` raises, naming it. A dial
    that is there was checked when the `Style` was built (`Style.validate`)."""
    dial = (style.dials or {}).get("leaves")
    if dial is None:
        raise ManifestError("style: dials.leaves is missing; it must hold normal_lean and "
                            "centre_drop (the leaf-normal lean is a render dial)")
    return LeafLean(toward_out=float(dial["normal_lean"]), centre_drop=float(dial["centre_drop"]))


def crown_normals(leaf_parts: dict[str, np.ndarray], lean: LeafLean) -> dict[str, np.ndarray]:
    """Per-corner normals for the leaf parts, (N, 3, 3) each. The crown centre is the middle of
    the leaf box, lowered `lean.centre_drop` of its height. Each normal leans `lean.toward_out`
    outward along the line from that centre (away from it), the rest from the face."""
    if not leaf_parts:
        return {}
    allv = np.concatenate([t.reshape(-1, 3) for t in leaf_parts.values()])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    centre = (lo + hi) / 2.0
    centre[1] -= lean.centre_drop * (hi[1] - lo[1])
    c = lean.toward_out
    out = {}
    for name, tris in leaf_parts.items():
        face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        face = _unit_rows(face)
        radial = _unit_rows(tris - centre)  # (N, 3, 3)
        out[name] = _unit_rows(c * radial + (1.0 - c) * face[:, None, :])
    return out


def _unit_rows(v: np.ndarray) -> np.ndarray:
    """v scaled to length 1 along the last axis; a zero vector stays zero."""
    n = np.sqrt((v * v).sum(axis=-1, keepdims=True))
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0.0)
