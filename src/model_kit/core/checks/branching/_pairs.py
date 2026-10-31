"""The limb-pair search the crossing and near-miss checks share.

Each limb is split into pieces along its axis: its segments, split again at every zone end.
Two pieces are compared at the closest points of their axes (`grow.segments.gap_shares`): the
gap there and the thinner radius there. A limb pair reports its piece pair with the smallest
share `gap / r`. A limb is also paired with itself (item `<a> x <a>`), so a limb that curls
back through itself is seen.

The zones (`grow.segments`) leave pieces out: each limb's seat part against every limb, the
shared zone of two siblings from one place, and a limb's own joints. The grower's stop-short
rule uses the same zones, so a crotch reads the same to both.

The search runs in two steps: a box test on every piece's box (padded by its radius times
`1 + reach`), swept along x, then the exact distance on the pairs that pass, in chunks of
`CHUNK` piece pairs. A pair whose gap is at most `reach` times the thinner radius always passes
the box test.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Collection, Iterator

import numpy as np

from model_kit.core.checks.branching._pieces import Grown
from model_kit.core.grow.segments import (
    from_one_place,
    gap_shares,
    left_out,
    parent_radius,
    pieces,
    seat_length,
    seat_sphere,
    shared_zones,
)
from model_kit.core.grow.shape import (
    SEGMENT_EPS,
    SELF_PIECE_OF_WINDOW,
    SELF_WINDOW_OF_RADII,
    SIDE,
)
from model_kit.core.grow.skeleton import Skeleton, arc_lengths
from model_kit.core.metagate import Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import CROSS_OF_RADIUS, NEAR_OF_RADIUS

# Piece pairs measured at once: a memory bound, not a tolerance.
CHUNK = 4096

CROSSING, NEAR_MISS = "crossing", "near miss"


def classify_gap(gap: float, r: float) -> str | None:
    """What a limb pair's gap `gap` is at the thinner radius `r`: `CROSSING` when the limbs
    overlap by more than `CROSS_OF_RADIUS` x r, `NEAR_MISS` from there up to a gap of
    `NEAR_OF_RADIUS` x r, else None. One comparison for both checks, so the line between them
    belongs to exactly one: a limb that reaches exactly `CROSS_OF_RADIUS` x r is at the limit and
    not over it, as in `crossings` (a part sunk exactly half the thinner part is not embedded)."""
    if gap < -CROSS_OF_RADIUS * r:
        return CROSSING
    if gap <= NEAR_OF_RADIUS * r:
        return NEAR_MISS
    return None


class Zones:
    """A skeleton's zones (`grow.segments`), worked out as they are asked for."""

    def __init__(self, sk: Skeleton, clearance: float) -> None:
        self.sk, self.clearance = sk, clearance
        self._seat: dict[int, float] = {}
        self._sphere: dict[int, float] = {}
        self._shared: dict[tuple[int, int], tuple[float, float] | None] = {}

    def seat(self, i: int) -> float:
        """How far along limb i its seat part runs (0 for a limb with no parent)."""
        if i not in self._seat:
            sk, p = self.sk, int(self.sk.parent[i])
            self._seat[i] = 0.0 if p < 0 else seat_length(
                sk.limb_points(i), sk.limb_radii(i), sk.limb_points(p), sk.limb_radii(p),
                self.clearance)
        return self._seat[i]

    def sphere(self, i: int) -> float:
        if i not in self._sphere:
            sk, p = self.sk, int(self.sk.parent[i])
            r_parent = parent_radius(sk.limb_points(i), sk.limb_points(p), sk.limb_radii(p))
            self._sphere[i] = seat_sphere(r_parent, float(sk.limb_radii(i)[0]), self.clearance)
        return self._sphere[i]

    def shared(self, i: int, j: int) -> tuple[float, float] | None:
        """(i's shared zone against j, j's against i) for two siblings from one place, else
        None."""
        key = (i, j)
        if key not in self._shared:
            sk = self.sk
            pi, pj = int(sk.parent[i]), int(sk.parent[j])
            out = None
            if i != j and pi >= 0 and pi == pj and from_one_place(
                    sk.limb_points(i)[0], self.sphere(i), sk.limb_points(j)[0], self.sphere(j)):
                out = shared_zones(sk.limb_points(i), sk.limb_radii(i), sk.limb_points(j),
                                   sk.limb_radii(j), self.clearance)
            self._shared[key] = out
            self._shared[(j, i)] = None if out is None else (out[1], out[0])
        return self._shared[key]


def _sibling_pairs(sk: Skeleton, limbs: set[int] | None) -> list[tuple[int, int]]:
    kids: dict[int, list[int]] = {}
    for i, p in enumerate(sk.parent):
        if p >= 0 and (limbs is None or i in limbs):
            kids.setdefault(int(p), []).append(i)
    return [(b, c) for group in kids.values() for n, b in enumerate(group) for c in group[n + 1:]]


def _box_pairs(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every (i, j), i < j, whose boxes meet: sorted by box start along x, swept."""
    order = np.argsort(lo[:, 0], kind="stable")
    lo_s, hi_s = lo[order], hi[order]
    end = np.searchsorted(lo_s[:, 0], hi_s[:, 0], side="right")  # boxes starting before i ends
    count = np.maximum(end - np.arange(len(order)) - 1, 0)
    i = np.repeat(np.arange(len(order)), count)
    if not len(i):
        return i, i
    j = i + 1 + np.arange(len(i)) - np.repeat(np.cumsum(count) - count, count)
    meet = np.all((lo_s[j, 1:] <= hi_s[i, 1:]) & (lo_s[i, 1:] <= hi_s[j, 1:]), axis=1)
    i, j = order[i[meet]], order[j[meet]]
    return np.minimum(i, j), np.maximum(i, j)


def limb_pairs(sk: Skeleton, only: Collection[tuple[int, int]] | None = None, *,
               reach: float, opens: float) -> list[tuple[int, int, float, float]]:
    """(i, j, gap, r_min), i <= j, for every limb pair (a limb with itself too) whose padded
    piece boxes meet, outside the zones: the gap and thinner radius at the piece pair with the
    smallest gap share.

    `reach` pads each piece's box by its radius times `1 + reach`, so every pair with a gap of
    at most `reach` times the thinner radius is found. `opens` is the clearance the zones open
    at. With `only`, just those limb pairs are measured.
    """
    n_limbs = len(sk.limb_id)
    want = limbs = None
    if only is not None:
        want = {(min(i, j), max(i, j)) for i, j in only}
        limbs = {k for pair in want for k in pair}
    zones = Zones(sk, opens)
    cuts: dict[int, list[float]] = {i: [] for i in (range(n_limbs) if limbs is None else limbs)}
    shared: dict[tuple[int, int], tuple[float, float]] = {}
    for b, c in _sibling_pairs(sk, limbs):
        if want is not None and (b, c) not in want:
            continue
        z = zones.shared(b, c)
        if z is not None:
            shared[(b, c)] = z
            cuts[b].append(z[0])
            cuts[c].append(z[1])
    parts = []
    for i, extra in cuts.items():
        radii = sk.limb_radii(i)
        window = 2.0 * np.maximum(radii[:-1], radii[1:]) * (1.0 + opens) * SELF_WINDOW_OF_RADII
        pc = pieces(sk.limb_points(i), radii, [zones.seat(i), *extra],
                    max_len=window * SELF_PIECE_OF_WINDOW)
        pc["limb"] = np.full(len(pc["seg"]), i)
        parts.append(pc)
    if not parts:
        return []
    pc = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    p, q, ra, rb = pc["p"], pc["q"], pc["ra"], pc["rb"]
    limb, seg, s0, s1 = pc["limb"], pc["seg"], pc["s0"], pc["s1"]
    pad = (np.maximum(ra, rb) * (1.0 + reach))[:, None]
    si, sj = _box_pairs(np.minimum(p, q) - pad, np.maximum(p, q) + pad)
    swap = limb[si] > limb[sj]
    si, sj = np.where(swap, sj, si), np.where(swap, si, sj)
    li, lj = limb[si], limb[sj]
    key = li.astype(np.int64) * n_limbs + lj
    if want is not None:
        hit = np.isin(key, [i * n_limbs + j for i, j in want])
        si, sj, li, lj, key = si[hit], sj[hit], li[hit], lj[hit], key[hit]
    if not len(si):
        return []
    seat = np.array([zones.seat(int(i)) if i in cuts else 0.0 for i in range(n_limbs)])
    pairs, inverse = np.unique(key, return_inverse=True)
    z = np.array([shared.get((int(k // n_limbs), int(k % n_limbs)), (0.0, 0.0))
                  for k in pairs]).reshape(-1, 2)
    thick = np.maximum(ra, rb)  # each piece's thicker end
    out = left_out(li == lj, seg[si], seg[sj], s0[si], s1[si], s0[sj], s1[sj], thick[si],
                   thick[sj], seat[li], seat[lj], z[inverse, 0], z[inverse, 1], opens)
    si, sj, key = si[~out], sj[~out], key[~out]
    if not len(si):
        return []
    gaps, rmin = np.empty(len(si)), np.empty(len(si))
    for k in range(0, len(si), CHUNK):
        x, y = si[k:k + CHUNK], sj[k:k + CHUNK]
        gaps[k:k + CHUNK], rmin[k:k + CHUNK] = gap_shares(p[x], q[x], ra[x], rb[x],
                                                          p[y], q[y], ra[y], rb[y])
    share = gaps / np.maximum(rmin, SEGMENT_EPS)
    order = np.lexsort((share, key))
    first = order[np.flatnonzero(np.r_[True, key[order][1:] != key[order][:-1]])]
    return [(int(key[k] // n_limbs), int(key[k] % n_limbs), float(gaps[k]), float(rmin[k]))
            for k in first]


def related(sk: Skeleton) -> set[tuple[int, int]]:
    """Every parent-and-child or sibling limb pair (i, j), i < j (for the measure script)."""
    out = {(int(p), i) for i, p in enumerate(sk.parent) if p >= 0}
    return out | set(_sibling_pairs(sk, None))


# --- placing a limb against another (plants and calibration) -----------------------------------

# Each segment is read at this many evenly spaced points when a placement is verified. A
# sampling density, not a tolerance: it sets how finely the verification looks.
PLACE_SAMPLES = 16


def _dense(pts: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.arange(PLACE_SAMPLES) / PLACE_SAMPLES
    seg = np.repeat(np.arange(len(pts) - 1), PLACE_SAMPLES)
    ff = np.tile(f, len(pts) - 1)
    at = pts[seg] + (pts[seg + 1] - pts[seg]) * ff[:, None]
    r = radii[seg] + (radii[seg + 1] - radii[seg]) * ff
    return np.concatenate([at, pts[-1:]]), np.concatenate([r, radii[-1:]])


def sampled_share(a_pts, a_radii, b_pts, b_radii) -> float:
    """The smallest gap share between two limbs, read point against point along both axes:
    a check on a placement that does not use the segment math the checks use."""
    ap, ar = _dense(a_pts, a_radii)
    bp, br = _dense(b_pts, b_radii)
    d = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
    return float(((d - ar[:, None] - br[None, :]) / np.minimum(ar[:, None], br[None, :])).min())


def placements(sk: Skeleton, x: int, y: int, share: float, after: float = 0.0
               ) -> Iterator[Skeleton]:
    """Each copy of sk with limb x moved rigidly so its closest approach to limb y has the gap
    share `share`, best placement first. A vertex of x is set beside a vertex of y (one more than
    `after` along y's arc, so past y's seat part), across y's axis, and a move is kept only when
    no other part of x comes closer (read by `sampled_share`), so the share is the pair's
    smallest. Nothing when no vertex of y gives such a placement."""
    xp, xr = sk.limb_points(x), sk.limb_radii(x)
    yp, yr = sk.limb_points(y), sk.limb_radii(y)
    cum = arc_lengths(yp)
    middle = (len(yp) - 1) / 2
    for q in sorted(range(len(yp)), key=lambda k: (abs(k - middle), k)):
        if cum[q] <= after:
            continue
        tangent = yp[min(q + 1, len(yp) - 1)] - yp[max(q - 1, 0)]
        tangent = tangent / max(float(np.linalg.norm(tangent)), SEGMENT_EPS)
        towards = xp.mean(axis=0) - yp[q]
        towards = towards - tangent * float(towards @ tangent)
        if float(np.linalg.norm(towards)) <= SEGMENT_EPS:
            towards = np.cross(tangent, SIDE)
        n0 = towards / float(np.linalg.norm(towards))
        b0 = np.cross(tangent, n0)
        for n in (n0, -n0, b0, -b0):
            k = int(np.argmin(xp @ n))
            r_min = min(float(yr[q]), float(xr[k]))
            h = float(yr[q]) + float(xr[k]) + share * r_min
            moved = xp + (yp[q] + n * h - xp[k])
            if sampled_share(moved, xr, yp, yr) < share - (abs(share) + 1.0) * SEGMENT_EPS:
                continue  # another part of x comes closer
            pts = np.array(sk.pts)
            pts[sk.start[x]:sk.start[x + 1]] = moved
            yield dataclasses.replace(sk, pts=pts)


def place_pair(sk: Skeleton, x: int, y: int, share: float, after: float = 0.0
               ) -> Skeleton | None:
    """The best of `placements`, or None when there is none."""
    return next(placements(sk, x, y, share, after), None)


def unrelated(sk: Skeleton, x: int) -> list[int]:
    """The limbs that do not grow from the same place as x, thickest base first."""
    px = int(sk.parent[x])
    out = [j for j in range(len(sk.limb_id))
           if j != x and int(sk.parent[j]) != x and j != px
           and not (px >= 0 and int(sk.parent[j]) == px)]
    return sorted(out, key=lambda j: (-float(sk.limb_radii(j)[0]), j))


def twigs(sk: Skeleton) -> list[int]:
    """The limbs nothing grows on that are neither rooted nor buried, outermost level first."""
    has_child = {int(p) for p in sk.parent if p >= 0}
    out = [i for i in range(len(sk.limb_id)) if i not in has_child and sk.parent[i] >= 0
           and not sk.rooted[i] and not sk.buried[i]]
    return sorted(out, key=lambda i: (-int(sk.level[i]), i))


def pair_order(sk: Skeleton) -> Iterator[tuple[int, int]]:
    """Each (x, y) to try moving limb x against limb y, best first, each pair once. A twig against
    an unrelated limb reads as a real fault. A tree with none (a seedling of a trunk and its
    roots, every twig on the trunk) then moves any limb, outermost level first, against an
    unrelated limb, and last against any other limb, thickest base first."""
    movers = sorted(range(len(sk.limb_id)), key=lambda i: (-int(sk.level[i]), i))
    seen: set[tuple[int, int]] = set()
    sources = (
        ((x, y) for x in twigs(sk) for y in unrelated(sk, x)),
        ((x, y) for x in movers for y in unrelated(sk, x)),
        ((x, y) for x in movers
         for y in sorted(range(len(sk.limb_id)), key=lambda j: (-float(sk.limb_radii(j)[0]), j))
         if y != x),
    )
    for pair in itertools.chain.from_iterable(sources):
        if pair not in seen:
            seen.add(pair)
            yield pair


def pair_units(ctx: SceneCtx, g: Grown, share: float, opens: float, label: str,
               candidates: int = 3) -> list[PlantUnit]:
    """The plant unit of one grown piece for a check that reads limb pairs: the first (x, y) of
    `pair_order` that takes a placement at `share`, with up to `candidates` placements as its
    candidates, best first (past y's seat part, zones opening at `opens`). The gate keeps the
    first the check catches. Geometry only: the placement is read by `sampled_share`, never by
    `limb_pairs`. A piece the checks judge (it has a limb) whose limbs take no placement gets a
    unit with no candidates, named after the first pair `pair_order` offers (a lone limb: itself),
    so the gate stops on it. A skeleton with no limb has no unit."""
    sk = g.skeleton
    if not len(sk.limb_id):
        return []
    zones = Zones(sk, opens)
    first = None
    for x, y in pair_order(sk):
        first = first or (x, y)
        found = list(itertools.islice(placements(sk, x, y, share, after=zones.seat(y)),
                                      candidates))
        if found:
            item = g.pair(x, y)
            return [PlantUnit(item, tuple(Plant(f"{label}: {item}", item, g.with_skeleton(ctx, m))
                                          for m in found))]
    return [PlantUnit(g.pair(*(first or (0, 0))), ())]
