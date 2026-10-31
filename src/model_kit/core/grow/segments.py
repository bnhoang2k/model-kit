"""Gaps between limbs, measured on their axes: the math the grower's stop-short rule and the
branching checks share, so a crotch reads the same to both.

A limb is a chain of axis segments with a radius at each point. Two segments are compared at
the closest points of their axes: the distance `d` there, less the two radii there, is the gap
(negative when the tubes overlap). Gap shares divide it by the thinner of the two radii.

Zones are lengths along a limb from its base (arc lengths, never segment counts); a limb is
split into pieces at every zone end, and a piece belongs to a zone when its middle lies in it.
A zone ends where the gap first opens past the clearance times the thinner radius, or where it
stalls: the gap (d less both radii) grows slower per unit length than sin
`SIBLING_MIN_ANGLE_DEG`. The test reads the gap, not the axis angle, so taper adds to the
growth: the line is that angle for limbs of constant radius, and about 1.0 to 1.25 degrees for
tapered ones (tapered twin twigs at 1.25 to 1.9 degrees pass; constant-radius twins fail up to
1.9). Gaps are read at points `ZONE_STEP_OF_RADIUS` times the limb's base radius apart from its
base. The stall end keeps a limb that runs alongside another from sheltering under the cap.
- The seat part. Each limb's base, up to where its gap to its parent ends the zone, sits in the
  parent's wood by construction (limb-seat judges it). It is left out of every pair, against
  any limb. It never runs past `fork_cap` of the two radii there, so a limb that hugs its parent
  is still measured past that length.
- A shared zone. Two children of one parent whose seat spheres overlap (`seat_sphere`: the
  parent's radius plus the child's radius times one plus the clearance, about each base) grow
  from one place. Each one's shared zone against the other runs to where their gap ends the
  zone, capped by `fork_cap` of their base radii, so siblings fused at a narrow angle still fail.
  Siblings seated apart share no zone.
- A limb against itself: pieces whose arc gap is within `SELF_WINDOW_OF_RADII` times the sum of
  their radii (grown by the clearance) are left out, so a thick limb with short internodes is
  not read as touching itself, and a limb that curls back through itself is still seen.
A parent and its child are compared like any pair past the child's seat part, so a twig that
grows back through its branch, or hugs it, is seen.

The stop-short rule itself (`Limb`, `Clearance`, `make_limbs`) lives at the end of this module,
because the bud grower (`buds.py`) uses it: it passes `max_piece_of_window` so its pieces are cut
as small as the checks cut them.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from model_kit.core.grow.shape import (
    PAIR_CHUNK,
    SEGMENT_EPS,
    SELF_WINDOW_OF_RADII,
    SIBLING_MIN_ANGLE_DEG,
    ZONE_STEP_OF_RADIUS,
)
from model_kit.core.grow.skeleton import CONE, arc_lengths, limb_at


def seg_seg(p1, q1, p2, q2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows of segments p1-q1 and p2-q2: the closest points' fractions (s, t) and distance.
    Ericson's closest points of two segments, vectorised, clamped to each segment."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.maximum((d1 * d1).sum(axis=1), SEGMENT_EPS)
    e = np.maximum((d2 * d2).sum(axis=1), SEGMENT_EPS)
    f = (d2 * r).sum(axis=1)
    c = (d1 * r).sum(axis=1)
    b = (d1 * d2).sum(axis=1)
    den = a * e - b * b
    ok = den > SEGMENT_EPS * a * e
    s = np.where(ok, np.clip((b * f - c * e) / np.where(ok, den, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0),
                 np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    gap = (p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None])
    return s, t, np.sqrt((gap * gap).sum(axis=1))


def gap_shares(p1, q1, r1a, r1b, p2, q2, r2a, r2b) -> tuple[np.ndarray, np.ndarray]:
    """For rows of segment pairs, each end with a radius: (gap, thinner radius) at the closest
    points of the two axes."""
    s, t, d = seg_seg(p1, q1, p2, q2)
    r1 = r1a + (r1b - r1a) * s
    r2 = r2a + (r2b - r2a) * t
    return d - r1 - r2, np.minimum(r1, r2)


def point_gaps(pts: np.ndarray, radii: np.ndarray, other_pts: np.ndarray,
               other_radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point of a limb (with its radius): (gap, thinner radius) to another limb's axis."""
    a, ab = other_pts[:-1], other_pts[1:] - other_pts[:-1]
    den = np.maximum((ab * ab).sum(axis=1), SEGMENT_EPS)
    f = np.clip(((pts[:, None, :] - a[None]) * ab[None]).sum(axis=2) / den, 0.0, 1.0)
    dist = np.sqrt(((a[None] + ab[None] * f[..., None] - pts[:, None, :]) ** 2).sum(axis=2))
    k = np.argmin(dist, axis=1)
    rows = np.arange(len(pts))
    r_other = other_radii[k] + (other_radii[k + 1] - other_radii[k]) * f[rows, k]
    return dist[rows, k] - r_other - radii, np.minimum(r_other, radii)


def first_open(pts: np.ndarray, radii: np.ndarray, other_pts: np.ndarray,
               other_radii: np.ndarray, clearance: float) -> float:
    """The arc length along a limb where its zone against another limb ends: where their gap
    first exceeds `clearance` times the thinner radius (it has opened), or where the gap first
    grows slower than limbs parting at `SIBLING_MIN_ANGLE_DEG` (it stalls: the limb hugs the
    other or turns back toward it); inf when neither happens."""
    cum = arc_lengths(pts)
    step = ZONE_STEP_OF_RADIUS * float(radii[0])
    s = np.append(np.arange(0.0, cum[-1], step), cum[-1])
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(pts) - 2)
    seg = np.maximum(cum[k + 1] - cum[k], SEGMENT_EPS)
    f = np.clip((s - cum[k]) / seg, 0.0, 1.0)
    at = pts[k] + (pts[k + 1] - pts[k]) * f[:, None]
    r = radii[k] + (radii[k + 1] - radii[k]) * f
    gap, r_min = point_gaps(at, r, other_pts, other_radii)
    opened = gap > clearance * r_min
    stalled = np.r_[False, (np.diff(gap) < np.diff(s) * math.sin(math.radians(
        SIBLING_MIN_ANGLE_DEG)))]
    end = np.flatnonzero(opened | stalled)
    return float(s[end[0]]) if len(end) else math.inf


def fork_cap(r1: float, r2: float, clearance: float) -> float:
    """The longest a seat part or shared zone may run: two limbs of radii r1 and r2 that part
    at `SIBLING_MIN_ANGLE_DEG` have opened the clearance by this length from their seat."""
    return (r1 + r2) * (1.0 + clearance) / math.sin(math.radians(SIBLING_MIN_ANGLE_DEG))


def parent_radius(pts: np.ndarray, parent_pts: np.ndarray, parent_radii: np.ndarray) -> float:
    """The parent's radius at the point of its axis nearest the limb's base."""
    _, r = point_gaps(pts[:1], np.full(1, np.inf), parent_pts, parent_radii)
    return float(r[0])


def seat_length(pts: np.ndarray, radii: np.ndarray, parent_pts: np.ndarray,
                parent_radii: np.ndarray, clearance: float) -> float:
    """How far along a limb its seat part runs: to where its gap to its parent opens, capped by
    `fork_cap` and by the limb's length."""
    cap = fork_cap(parent_radius(pts, parent_pts, parent_radii), float(radii[0]), clearance)
    length = float(arc_lengths(pts)[-1])
    return min(first_open(pts, radii, parent_pts, parent_radii, clearance), cap, length)


def seat_sphere(parent_r: float, r: float, clearance: float) -> float:
    """The radius about a limb's base inside which a sibling grows from the same place."""
    return parent_r + r * (1.0 + clearance)


def from_one_place(base_a: np.ndarray, sphere_a: float, base_b: np.ndarray,
                   sphere_b: float) -> bool:
    """Whether two children of one parent grow from one place: their seat spheres overlap
    (which holds whenever their bases are closer than the sum of their base radii)."""
    return float(np.sqrt(((base_a - base_b) ** 2).sum())) < sphere_a + sphere_b


def shared_zones(a_pts: np.ndarray, a_radii: np.ndarray, b_pts: np.ndarray,
                 b_radii: np.ndarray, clearance: float) -> tuple[float, float]:
    """The shared zone of two siblings that grow from one place, as a length along each: to
    where their gap opens, capped by `fork_cap` of their base radii and by each one's length."""
    cap = fork_cap(float(a_radii[0]), float(b_radii[0]), clearance)
    za = min(first_open(a_pts, a_radii, b_pts, b_radii, clearance), cap,
             float(arc_lengths(a_pts)[-1]))
    zb = min(first_open(b_pts, b_radii, a_pts, a_radii, clearance), cap,
             float(arc_lengths(b_pts)[-1]))
    return za, zb


def pieces(pts: np.ndarray, radii: np.ndarray, cuts=(), substeps: int = 1,
           max_len: np.ndarray | None = None) -> dict:
    """A limb split into pieces: each segment in `substeps` equal parts (more where `max_len`,
    one length per segment, asks for shorter pieces), and again at every arc length in `cuts`.
    Arrays `p`, `q` (ends), `ra`, `rb` (radii there), `s0`, `s1` (arc lengths there) and `seg`
    (the segment each piece lies on)."""
    cum = arc_lengths(pts)
    rows: list[tuple[int, float, float]] = []
    for k in range(len(pts) - 1):
        length = cum[k + 1] - cum[k]
        parts = substeps
        if max_len is not None and max_len[k] > 0.0:
            parts = max(parts, math.ceil(length / max_len[k]))
        fs = {j / parts for j in range(parts)}
        if length > 0.0:
            fs |= {(c - cum[k]) / length for c in cuts
                   if cum[k] + SEGMENT_EPS < c < cum[k + 1] - SEGMENT_EPS}  # no slivers
        fs = sorted(fs) + [1.0]
        rows += [(k, a, b) for a, b in itertools.pairwise(fs) if b > a]
    seg = np.array([r[0] for r in rows], dtype=int)
    fa = np.array([r[1] for r in rows])
    fb = np.array([r[2] for r in rows])
    d, dr, ds = pts[seg + 1] - pts[seg], radii[seg + 1] - radii[seg], cum[seg + 1] - cum[seg]
    return {"p": pts[seg] + d * fa[:, None], "q": pts[seg] + d * fb[:, None],
            "ra": radii[seg] + dr * fa, "rb": radii[seg] + dr * fb,
            "s0": cum[seg] + ds * fa, "s1": cum[seg] + ds * fb, "seg": seg}


def left_out(same: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray, s0_a: np.ndarray,
             s1_a: np.ndarray, s0_b: np.ndarray, s1_b: np.ndarray, r_a: np.ndarray,
             r_b: np.ndarray, seat_a: np.ndarray, seat_b: np.ndarray, zone_a: np.ndarray,
             zone_b: np.ndarray, clearance: float) -> np.ndarray:
    """Which piece pairs the zones leave out. Per pair: whether the two pieces are on one limb,
    their segments, each piece's arc-length interval (`s0`, `s1`) and radius, each limb's seat
    part, and each one's shared zone against the other (0 when they share none). Seat parts and
    zones are read at a piece's middle.

    Two pieces of one limb are left out when the arc between their intervals,
    `max(0, s0_b - s1_a, s0_a - s1_b)`, is at most `SELF_WINDOW_OF_RADII` times
    `(r_a + r_b) * (1 + clearance)`. Adjacent segments share a node, so their arc gap is 0 and
    they are always left out. `seg_a` and `seg_b` are kept for callers that name segments; the
    arc gap decides."""
    arc = np.maximum(0.0, np.maximum(s0_b - s1_a, s0_a - s1_b))
    window = (r_a + r_b) * (1.0 + clearance) * SELF_WINDOW_OF_RADII
    mid_a, mid_b = (s0_a + s1_a) / 2.0, (s0_b + s1_b) / 2.0
    return ((same & (arc <= window))
            | (mid_a < seat_a) | (mid_b < seat_b)
            | ((mid_a < zone_a) & (mid_b < zone_b)))


def boxes_meet(lo: np.ndarray, hi: np.ndarray, box_lo: np.ndarray, box_hi: np.ndarray):
    """(n,) bool: which of n boxes (lo, hi rows) meet one box."""
    return np.all((lo <= box_hi) & (box_lo <= hi), axis=1)


# --- limbs and the stop-short rule: the part of the grower both growers share

class Limb:
    """A limb as the draws planned it (`pts`, `radii`, `length`; `at` walks this plan) and as it
    is kept: `out_pts` and `out_radii` end where it stops short, `keep` arc length along it. A
    limb is `alive` when it is made: its parent is made and holds its seat on the kept part."""

    __slots__ = (
        "alive",
        "buried",
        "children",
        "cum",
        "index",
        "keep",
        "kind",
        "length",
        "level",
        "on",
        "out_pts",
        "out_radii",
        "pts",
        "radii",
        "rooted",
        "seat",
        "sphere",
        "t",
        "tip",
    )

    def __init__(self, kind: str, level: int, pts, radii, on: Limb | None, t: float,
                 tip: int = CONE, rooted: bool = False, buried: bool = False) -> None:
        self.kind, self.level = kind, level
        self.pts = np.asarray(pts, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.on, self.t, self.tip = on, t, tip
        self.rooted, self.buried = rooted, buried
        self.cum = arc_lengths(self.pts)
        self.length = float(self.cum[-1])
        self.index = -1
        self.alive = True  # until `_make` says otherwise
        self.out_pts, self.out_radii, self.keep = self.pts, self.radii, self.length
        self.children: list[Limb] = []
        self.seat = 0.0  # how far along it its seat part runs (`segments.seat_length`)
        self.sphere = 0.0  # its seat sphere's radius (`segments.seat_sphere`)

    def at(self, t: float):
        return limb_at(self.pts, self.radii, t, self.cum)

    def holds(self, t: float) -> bool:
        """Whether arc fraction t of the plan lies on the kept part."""
        return t * self.length <= self.keep

    def kept_t(self, t: float) -> float:
        """Arc fraction t of the plan as a fraction of the kept part."""
        return min(1.0, t * self.length / self.keep) if self.keep > 0.0 else 0.0

    def replan(self) -> None:
        """Take the kept part as the whole plan, for another pass of stop-short. Arc lengths from
        the base are unchanged (the kept part is a prefix). Seats on this limb are fractions of the
        plan, so the caller rescales its children's `t`; made children are listed afresh."""
        self.pts, self.radii = self.out_pts, self.out_radii
        self.cum = arc_lengths(self.pts)
        self.length = float(self.cum[-1])
        self.keep = self.length
        self.alive = True
        self.children = []

    @property
    def parent(self) -> int:
        return -1 if self.on is None else self.on.index

    @property
    def seat_t(self) -> float:
        return self.t if self.on is None else self.on.kept_t(self.t)


def prefix(pts: np.ndarray, radii: np.ndarray, s: float) -> tuple[np.ndarray, np.ndarray]:
    """The first s of a limb's arc length: its points before s, and the point at s."""
    cum = arc_lengths(pts)
    k = int(np.searchsorted(cum, s, side="left"))  # points 0..k-1 lie before s
    out_pts, out_radii = list(pts[:k]), list(radii[:k])
    if k < len(pts) and cum[k] == s:
        out_pts.append(pts[k])
        out_radii.append(float(radii[k]))
    elif 0 < k < len(pts):
        f = (s - cum[k - 1]) / (cum[k] - cum[k - 1])
        out_pts.append(pts[k - 1] + (pts[k] - pts[k - 1]) * f)
        out_radii.append(float(radii[k - 1] + (radii[k] - radii[k - 1]) * f))
    return np.array(out_pts).reshape(-1, 3), np.array(out_radii)


def _earlier_pairs(live: np.ndarray):
    """Every pair (i, j) of `live`, i before j, in blocks of at most `PAIR_CHUNK` pairs: the
    same pairs `np.triu_indices` gives, without holding all of them at once (a long limb of
    short pieces has millions)."""
    n = len(live)
    block = max(1, PAIR_CHUNK // max(n, 1))
    for a in range(0, n, block):
        rows = np.arange(a, min(a + block, n))
        i = np.repeat(rows, n)
        j = np.tile(np.arange(n), len(rows))
        ahead = j > i
        if ahead.any():
            yield live[i[ahead]], live[j[ahead]]


class Clearance:
    """The stop-short rule. A new limb ends at the start of the first piece whose gap to a limb
    already made, or to its own earlier part, falls below `share` times the thinner radius
    there. The zones (`segments`: seat parts, shared zones of siblings from one place, a limb's
    own joints) leave pieces out, the zones opening at the same share. Each segment of the new
    limb is tested in `substeps` equal pieces, split again at every zone end."""

    def __init__(self, share: float, substeps: int, max_piece_of_window: float | None = None
                 ) -> None:
        self.share, self.substeps = share, substeps
        self.max_piece = max_piece_of_window
        self.p = np.zeros((0, 3))
        self.q = np.zeros((0, 3))
        self.ra = np.zeros(0)
        self.rb = np.zeros(0)
        self.mid = np.zeros(0)  # each piece's middle, as arc length along its limb
        self.seat = np.zeros(0)  # its limb's seat part
        self.owner: list[Limb] = []
        self.lo = np.zeros((0, 3))
        self.hi = np.zeros((0, 3))

    def _max_len(self, radii: np.ndarray) -> np.ndarray | None:
        """The longest piece of each segment: `max_piece_of_window` of the limb's self-window there,
        as the checks cut a limb (`checks.branching._pairs.limb_pairs`: the window is twice the
        thicker end's radius, grown by the clearance, times `SELF_WINDOW_OF_RADII`). None when the
        caller gave no share: pieces are then cut by `substeps` alone. A
        hairpin of long segments is missed otherwise: two pieces of one limb that are
        a whole long piece apart are measured only when the pieces are short."""
        if self.max_piece is None:
            return None
        window = 2.0 * np.maximum(radii[:-1], radii[1:]) * (1.0 + self.share) * SELF_WINDOW_OF_RADII
        return window * self.max_piece

    def made(self, limb: Limb) -> None:
        """Add a made limb's kept pieces, split at the end of its seat part."""
        pc = pieces(limb.out_pts, limb.out_radii, (limb.seat,),
                    max_len=self._max_len(limb.out_radii))
        pad = (np.maximum(pc["ra"], pc["rb"]) * (1.0 + self.share))[:, None]
        self.p, self.q = np.concatenate([self.p, pc["p"]]), np.concatenate([self.q, pc["q"]])
        self.ra = np.concatenate([self.ra, pc["ra"]])
        self.rb = np.concatenate([self.rb, pc["rb"]])
        self.mid = np.concatenate([self.mid, (pc["s0"] + pc["s1"]) / 2.0])
        self.seat = np.concatenate([self.seat, np.full(len(pc["seg"]), limb.seat)])
        self.owner += [limb] * len(pc["seg"])
        self.lo = np.concatenate([self.lo, np.minimum(pc["p"], pc["q"]) - pad])
        self.hi = np.concatenate([self.hi, np.maximum(pc["p"], pc["q"]) + pad])

    def _seat(self, limb: Limb) -> list[Limb]:
        """Set the limb's seat part and seat sphere; return its siblings from one place."""
        on = limb.on
        if on is None:
            return []
        pts, radii = limb.pts, limb.radii
        limb.seat = seat_length(pts, radii, on.out_pts, on.out_radii, self.share)
        limb.sphere = seat_sphere(parent_radius(pts, on.out_pts, on.out_radii),
                                  float(radii[0]), self.share)
        return [sib for sib in on.children
                if from_one_place(sib.out_pts[0], sib.sphere, pts[0], limb.sphere)]

    def _others(self, near: np.ndarray, sibs: list[Limb], zones) -> list[tuple]:
        """The other limbs' pieces, in groups: (p, q, ra, rb, mid, seat, their zone against the
        new limb, the new limb's zone against them). Stored pieces share no zone; each sibling
        from one place is split afresh at its seat and zone ends."""
        n = len(near)
        groups = [(self.p[near], self.q[near], self.ra[near], self.rb[near], self.mid[near],
                   self.seat[near], np.zeros(n), 0.0)]
        for sib, (z_new, z_sib) in zip(sibs, zones, strict=True):
            pc = pieces(sib.out_pts, sib.out_radii, (sib.seat, z_sib),
                        max_len=self._max_len(sib.out_radii))
            m = len(pc["seg"])
            groups.append((pc["p"], pc["q"], pc["ra"], pc["rb"], (pc["s0"] + pc["s1"]) / 2.0,
                           np.full(m, sib.seat), np.full(m, z_sib), z_new))
        return groups

    def cut(self, limb: Limb) -> None:
        """Set the limb's kept part. A limb kept with no length is not made.

        Shared zones are read against the limb as kept so far, so the cut repeats until it
        stays put: a shorter limb opens its siblings' zones sooner, never later."""
        sibs = self._seat(limb)
        pts, radii = limb.pts, limb.radii
        pad = float(radii.max()) * (1.0 + self.share)
        near = np.flatnonzero(boxes_meet(self.lo, self.hi, pts.min(axis=0) - pad,
                                         pts.max(axis=0) + pad))
        skip = {id(sib) for sib in sibs}
        near = np.array([k for k in near if id(self.owner[k]) not in skip], dtype=int)
        keep = limb.length
        while True:
            zones = [shared_zones(limb.out_pts, limb.out_radii, sib.out_pts, sib.out_radii,
                                  self.share) for sib in sibs]
            mine = pieces(pts, radii, [limb.seat] + [z[0] for z in zones], self.substeps,
                          max_len=self._max_len(radii))
            live = np.flatnonzero(mine["s1"] <= keep)
            mine_r = np.maximum(mine["ra"], mine["rb"])  # each piece's thicker end
            hit = np.zeros(len(mine["s0"]), dtype=bool)
            for p, q, ra, rb, o_mid, o_seat, o_zone, z_new in self._others(near, sibs, zones):
                if not len(p) or not len(live):
                    continue
                i, j = np.repeat(live, len(p)), np.tile(np.arange(len(p)), len(live))
                gap, r = gap_shares(mine["p"][i], mine["q"][i], mine["ra"][i], mine["rb"][i],
                                    p[j], q[j], ra[j], rb[j])
                # Another limb's pieces: the self-window is masked (`same` is False), so its
                # arc interval and radius are read as the piece's middle and thicker end.
                out = left_out(np.zeros(len(i), dtype=bool), mine["seg"][i], mine["seg"][i],
                               mine["s0"][i], mine["s1"][i], o_mid[j], o_mid[j],
                               mine_r[i], np.maximum(ra[j], rb[j]), limb.seat, o_seat[j],
                               z_new, o_zone[j], self.share)
                hit[i[(gap < self.share * r) & ~out]] = True
            for i, j in _earlier_pairs(live):  # the limb against its own earlier pieces
                gap, r = gap_shares(mine["p"][i], mine["q"][i], mine["ra"][i], mine["rb"][i],
                                    mine["p"][j], mine["q"][j], mine["ra"][j], mine["rb"][j])
                out = left_out(np.ones(len(i), dtype=bool), mine["seg"][i], mine["seg"][j],
                               mine["s0"][i], mine["s1"][i], mine["s0"][j], mine["s1"][j],
                               mine_r[i], mine_r[j], limb.seat, limb.seat, 0.0, 0.0, self.share)
                hit[j[(gap < self.share * r) & ~out]] = True
            if not hit.any():
                return
            cut = float(mine["s0"][np.argmax(hit)])
            if cut >= keep:
                return
            keep = cut
            limb.out_pts, limb.out_radii = prefix(pts, radii, keep)
            limb.keep = keep
            if keep <= 0.0 or len(limb.out_pts) < 2:
                limb.alive = False
                return


def make_limbs(grown: list[Limb], clear: Clearance) -> list[Limb]:
    """Make the planned limbs, each stopped short of the limbs made before it, and return the
    made ones in growing order, indexed. The draws planned every limb already, so the order of
    making changes no draw: limbs are made level by level (trunk and roots, leaders, branches,
    twigs, shoots), in growing order within a level, so a thin limb stops short of a thick one
    and never the other way round. A limb is made only when its parent is made and holds its
    seat on the part it kept."""
    for limb in sorted(grown, key=lambda limb: limb.level):  # a stable sort keeps growing order
        on = limb.on
        limb.alive = on is None or (on.alive and on.holds(limb.t))
        if limb.alive:
            clear.cut(limb)
        if limb.alive:
            clear.made(limb)
            if on is not None:
                on.children.append(limb)
    limbs = [limb for limb in grown if limb.alive]
    for i, limb in enumerate(limbs):
        limb.index = i
    return limbs
