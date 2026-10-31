"""Triangle geometry in numpy: nearest point, ray parity, closed-mesh tests, box cuts.

This replaces Blender's BVHTree in a core that never imports bpy or scipy. Triangles are an
(N, 3, 3) array: N triangles, 3 corners, 3 coordinates. Winding gives the face normal.
"""

import itertools
import math
from functools import cached_property
from typing import NamedTuple

import numpy as np

from model_kit.core.errors import Unmeasurable
from model_kit.core.tolerances import CONE_AROUND, CONE_DEG, CONE_WEIGHT_DEG

# Most (point, triangle) pairs one dense step may hold. Bigger inputs run in point chunks.
PAIR_BUDGET = 5_000_000
# Most candidate pairs the exact closest-point routine takes at once.
PAIR_BLOCK = 500_000
# Ray hits closer than this along a ray do not count (the origin's own surface).
_RAY_T_MIN = 1e-9
# Slack on the barycentric edges so a ray through an exact edge is not lost to rounding.
_RAY_EDGE_SLACK = 1e-12
# A triangle whose cross product is below this share of its longest edge squared is a segment:
# it has no area. A float guard, not a measuring tolerance.
DEGENERATE_SHARE = 1e-12
# A vector shorter than this has no direction (float noise, not a measuring tolerance).
NO_DIRECTION = 1e-9

# Faces within this distance (model units) of the nearest one tie with it. Float noise, like
# NO_DIRECTION, not a measuring tolerance: the faces of a round part tie at its axis to about 1e-10.
TIE_DISTANCE = 1e-9

# Triangles per leaf of a SurfaceIndex. A structural constant: it sets how coarse the boxes are,
# never what a query answers.
LEAF_SIZE = 32
# Bits per axis of a Morton code (3 x 21 = 63 bits fit a uint64).
_MORTON_BITS = 21
# `core.bvh` grows a box by this share of the mesh's size in two places, both float guards that
# make a prune safe, neither a measuring tolerance: the outline boxes a ray is tested against, so
# a hit the exact test counts (within `_RAY_EDGE_SLACK` of an edge) is never lost to rounding; and
# the nearest-point bound, so a bound of 0 (a point on a shared edge) still keeps a tied face.
_SLAB_PAD = 1e-9
# The nearest-point bound keeps boxes within this share beyond the best upper bound.
_BOUND_SLACK = 1e-9

# Float guards of `tri_tri_cut` and `surface_gap`, the numbers of a standard
# triangle-pair cut test. None is a measuring tolerance.
# A corner this near a triangle's plane is on it.
_PLANE_SLACK = 1e-9
# A triangle whose cross product is shorter than this has no face.
_NO_AREA = 1e-14
# Two unit normals whose cross product is shorter than this are parallel: the faces are flush.
_PARALLEL = 1e-6
# An intersection segment shorter than this is a point.
_NO_SPAN = 1e-12
# An edge whose squared length is below this is a point.
_NO_EDGE = 1e-18
# Points along the intersection segment, ends included, where `tri_tri_cut` measures the cut
# (5 by default).
CUT_SAMPLES = 5

# Three fixed, off-axis directions. A mesh edge or corner rarely lies on all of them.
RAYS = np.array([[1.0, 0.31, 0.17], [-0.23, 1.0, 0.41], [0.37, -0.29, 1.0]])
RAYS /= np.linalg.norm(RAYS, axis=1, keepdims=True)


class Nearest(NamedTuple):
    dist: np.ndarray  # (M,) distance to the surface
    point: np.ndarray  # (M, 3) nearest point on the surface
    face: np.ndarray  # (M,) index of the nearest triangle
    normal: np.ndarray  # (M, 3) unit normal of that triangle (zeros for a degenerate one)


def _as_tris(tris) -> np.ndarray:
    tris = np.asarray(tris, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape (N, 3, 3), got {tris.shape}")
    if not np.isfinite(tris).all():
        raise ValueError("triangles hold a non-finite value (NaN or infinity)")
    return tris


def _as_points(pts, name: str = "points") -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (M, 3), got {pts.shape}")
    if not np.isfinite(pts).all():
        raise ValueError(f"{name} hold a non-finite value (NaN or infinity)")
    return pts


def _dot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", u, v)


def _cross(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise cross product of two (n, 3) arrays: the same arithmetic as `np.cross` for 3-vectors
    (u1*v2 - u2*v1, ...), bit for bit, without its per-call axis bookkeeping, which cost more than
    the products on the small batches a solver sends."""
    out = np.empty(np.broadcast(u, v).shape)
    out[:, 0] = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    out[:, 1] = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    out[:, 2] = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    return out


def _closest_on_segments(p, a, b) -> np.ndarray:
    ab = b - a
    len2 = _dot(ab, ab)
    t = np.where(len2 > 0, _dot(p - a, ab) / np.where(len2 > 0, len2, 1.0), 0.0)
    return a + np.clip(t, 0.0, 1.0)[:, None] * ab


def _closest_on_triangles(p, a, b, c) -> np.ndarray:
    """Closest point on triangle abc to p, row by row (Ericson, RTCD 5.1.5).

    Zero-area triangles are handled as the segments they collapse to.
    """
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        face = a + (ab * (vb / (va + vb + vc))[:, None]) + (ac * (vc / (va + vb + vc))[:, None])
        on_ab = a + ab * (d1 / (d1 - d3))[:, None]
        on_ac = a + ac * (d2 / (d2 - d6))[:, None]
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        on_bc = b + (c - b) * w[:, None]
    # Ericson's region tests, first match wins.
    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
    ]
    # The first region that matches wins: `np.select` without its per-call broadcasting.
    q = face
    for k, choice in zip(reversed(conds), reversed([a, b, on_ab, c, on_ac, on_bc]), strict=True):
        q = np.where(k[:, None], choice, q)
    # Zero-area (or numerically flat) triangles divide by nothing: use the nearest edge.
    longest2 = np.maximum(np.maximum(_dot(ab, ab), _dot(ac, ac)), _dot(c - b, c - b))
    cross = _cross(ab, ac)
    flat = (np.sqrt(_dot(cross, cross)) <= DEGENERATE_SHARE * longest2) | ~np.isfinite(q).all(
        axis=1
    )
    if flat.any():
        pf, af, bf, cf = p[flat], a[flat], b[flat], c[flat]
        cands = [
            _closest_on_segments(pf, af, bf),
            _closest_on_segments(pf, bf, cf),
            _closest_on_segments(pf, cf, af),
        ]
        d2s = np.stack([_dot(pf - k, pf - k) for k in cands])
        best = np.argmin(d2s, axis=0)
        q[flat] = np.stack(cands)[best, np.arange(len(pf))]
    return q


def _unit_normals(tris: np.ndarray) -> np.ndarray:
    n = _cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(length > 0, n / np.where(length > 0, length, 1.0), 0.0)


def _box_dist2(lo, hi, pts) -> np.ndarray:
    """Squared distance from each point (m, 3) to each box (N, 3) lo/hi: (m, N)."""
    total = 0.0
    for k in range(3):
        gap = np.maximum(
            np.maximum(lo[None, :, k] - pts[:, None, k], pts[:, None, k] - hi[None, :, k]), 0.0
        )
        total = total + gap * gap
    return total


def _closest_dense(tris, pts) -> Nearest:
    """Nearest point on the triangle set for each point, with no index (the reference path).

    A cheap bound prunes triangles: the distance to a triangle's box never exceeds the true
    distance, and the nearest vertex gives an upper bound. Only triangles whose box lies
    within that upper bound get the exact test. Memory stays bounded by chunking the points.

    Caveats for callers. At a shared edge or vertex, two or more triangles can be equally
    near; which one wins (and so `face` and `normal`) is not defined. When the nearest face
    is degenerate (zero area), `normal` is (0, 0, 0): never divide by it or take a sign from it.
    """
    tris = _as_tris(tris)
    pts = _as_points(pts)
    n_tris, n_pts = len(tris), len(pts)
    if n_tris == 0:
        raise ValueError("closest_points needs at least one triangle")
    best_d2 = np.full(n_pts, np.inf)
    best_pt = np.zeros((n_pts, 3))
    best_face = np.zeros(n_pts, dtype=np.intp)
    lo, hi = tris.min(axis=1), tris.max(axis=1)
    step = max(1, PAIR_BUDGET // n_tris)
    for start in range(0, n_pts, step):
        chunk = pts[start : start + step]
        lb2 = _box_dist2(lo, hi, chunk)
        ub2 = np.full(len(chunk), np.inf)
        for corner in range(3):
            diff = chunk[:, None, :] - tris[None, :, corner, :]
            ub2 = np.minimum(ub2, np.einsum("mnk,mnk->mn", diff, diff).min(axis=1))
        pi, fi = np.nonzero(lb2 <= ub2[:, None] * (1 + 1e-9))
        for s in range(0, len(pi), PAIR_BLOCK):
            bp, bf = pi[s : s + PAIR_BLOCK], fi[s : s + PAIR_BLOCK]
            p = chunk[bp]
            q = _closest_on_triangles(p, tris[bf, 0], tris[bf, 1], tris[bf, 2])
            d2 = _dot(p - q, p - q)
            gp = bp + start
            np.minimum.at(best_d2, gp, d2)
            won = d2 == best_d2[gp]
            best_pt[gp[won]] = q[won]
            best_face[gp[won]] = bf[won]
    return Nearest(
        dist=np.sqrt(best_d2),
        point=best_pt,
        face=best_face,
        normal=_unit_normals(tris)[best_face] if n_pts else np.zeros((0, 3)),
    )


def ray_counts(tris, origins, dirs, reach: float | None = None) -> np.ndarray:
    """Hits along each ray: an (M, K) int array for M origins and K directions.

    Moller-Trumbore. Every hit with t > 1e-9 counts: there is no early exit, since parity
    needs the whole count. Directions need not be unit length. With `reach`, a hit further than
    that distance from its origin (t times the direction's length) does not count.
    """
    tris = _as_tris(tris)
    origins = _as_points(origins, "origins")
    dirs = _as_points(dirs, "dirs")
    counts = np.zeros((len(origins), len(dirs)), dtype=int)
    if len(tris) == 0 or len(origins) == 0:
        return counts
    a = tris[:, 0]
    e1, e2 = tris[:, 1] - a, tris[:, 2] - a
    normal = np.cross(e1, e2)
    step = max(1, PAIR_BUDGET // len(tris))
    for k, d in enumerate(dirs):
        pvec = np.cross(d, e2)
        det = np.einsum("nk,nk->n", e1, pvec)
        scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d)
        live = np.abs(det) > 1e-12 * scale
        if not live.any():
            continue
        w = np.cross(e1, d)
        inv = 1.0 / np.where(live, det, 1.0)
        # tvec = o - a; u = tvec.pvec / det, v = tvec.(e1 x d) / det, t = tvec.(e1 x e2) / det.
        offs = [(m, (a * m).sum(axis=1)) for m in (pvec, w, normal)]
        for start in range(0, len(origins), step):
            o = origins[start : start + step]
            u, v, t = ((o @ m.T - off[None, :]) * inv for m, off in offs)
            hit = (
                live
                & (u >= -_RAY_EDGE_SLACK)
                & (v >= -_RAY_EDGE_SLACK)
                & (u + v <= 1 + _RAY_EDGE_SLACK)
                & (t > _RAY_T_MIN)
            )
            if reach is not None:
                hit &= t * np.linalg.norm(d) <= reach
            counts[start : start + step, k] = hit.sum(axis=1)
    return counts


def exits(tris, origins, dirs) -> np.ndarray:
    """How far each origin goes along each direction before it meets the triangles: an (M, K)
    float array for M origins and K unit directions, inf where a ray meets nothing. From inside a
    closed mesh it is the way out: how far the point must move along that direction to leave.

    Moller-Trumbore, as `ray_counts`: a hit counts with t > 1e-9 (a point on the surface does not
    meet the face it lies on).
    """
    tris = _as_tris(tris)
    origins = _as_points(origins, "origins")
    dirs = _as_points(dirs, "dirs")
    out = np.full((len(origins), len(dirs)), np.inf)
    if len(tris) == 0 or len(origins) == 0:
        return out
    a = tris[:, 0]
    e1, e2 = tris[:, 1] - a, tris[:, 2] - a
    normal = np.cross(e1, e2)
    step = max(1, PAIR_BUDGET // len(tris))
    for k, d in enumerate(dirs):
        pvec = np.cross(d, e2)
        det = np.einsum("nk,nk->n", e1, pvec)
        scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        live = np.abs(det) > 1e-12 * scale
        if not live.any():
            continue
        w = np.cross(e1, d)
        inv = 1.0 / np.where(live, det, 1.0)
        offs = [(m, (a * m).sum(axis=1)) for m in (pvec, w, normal)]
        for start in range(0, len(origins), step):
            o = origins[start : start + step]
            u, v, t = ((o @ m.T - off[None, :]) * inv for m, off in offs)
            hit = (live & (u >= -_RAY_EDGE_SLACK) & (v >= -_RAY_EDGE_SLACK)
                   & (u + v <= 1 + _RAY_EDGE_SLACK) & (t > _RAY_T_MIN))
            out[start : start + step, k] = np.where(hit, t, np.inf).min(axis=1)
    return out


class Exit(NamedTuple):
    """One part's points inside a closed mesh, and the ways they may leave it (`shortest_exit`)."""

    tris: np.ndarray  # (T, 3, 3) the closed mesh the points lie in
    points: np.ndarray  # (M, 3) points inside it
    dirs: np.ndarray  # (K, 3) unit directions: way k is row k of every side's `dirs`
    leaves: np.ndarray | None = None  # (M, K) True where a point leaves at once (0 to go)


def shortest_exit(sides, reach: float) -> float:
    """The shortest move that takes every point of every side out of its mesh: the penetration
    depth of collision detection. For each way k the move is the furthest any point must go along
    that side's direction k to leave its mesh (`exits`, so a point of a side that lies on a
    face and `leaves` goes 0); the answer is the least of those over the ways. Two parts that
    share space are one side each, the second with the reversed directions (moving the one is
    moving the other the opposite way). 0 for no points.

    Only the faces within `reach` of a side's points can end a way out that short, so the way out
    is read on those, and `reach` doubles until it holds the answer (or every face is read). The
    answer is inf when no way leaves the mesh.
    """
    sides = [s for s in sides if len(s.points)]
    if not sides:
        return 0.0
    boxes = [(s.points.min(axis=0), s.points.max(axis=0), s.tris.min(axis=1), s.tris.max(axis=1))
             for s in sides]
    while True:
        worst = np.zeros(len(sides[0].dirs))
        every = True
        for s, (lo, hi, t_lo, t_hi) in zip(sides, boxes, strict=True):
            near = ((t_hi >= lo - reach) & (t_lo <= hi + reach)).all(axis=1)
            out = exits(s.tris[near], s.points, s.dirs)
            if s.leaves is not None:
                out[s.leaves] = 0.0
            worst = np.maximum(worst, out.max(axis=0))
            every &= bool(near.all())
        best = float(worst.min())
        if best <= reach or every:
            return best
        reach *= 2.0


def _spread_bits(x: np.ndarray) -> np.ndarray:
    """Spread the low 21 bits of each value so two zero bits sit between neighbours."""
    x = x & np.uint64(0x1FFFFF)
    for shift, mask in (
        (32, 0x1F00000000FFFF),
        (16, 0x1F0000FF0000FF),
        (8, 0x100F00F00F00F00F),
        (4, 0x10C30C30C30C30C3),
        (2, 0x1249249249249249),
    ):
        x = (x | (x << np.uint64(shift))) & np.uint64(mask)
    return x


def _morton(points: np.ndarray) -> np.ndarray:
    """Morton code of each point inside the box of all the points: nearby points, nearby codes."""
    lo = points.min(axis=0)
    size = float((points.max(axis=0) - lo).max())
    unit = (points - lo) / (size if size > 0 else 1.0)
    cells = np.minimum(unit * (1 << _MORTON_BITS), (1 << _MORTON_BITS) - 1).astype(np.uint64)
    return (
        _spread_bits(cells[:, 0])
        | (_spread_bits(cells[:, 1]) << np.uint64(1))
        | (_spread_bits(cells[:, 2]) << np.uint64(2))
    )


def _corner_mean(tris: np.ndarray) -> np.ndarray:
    """The centre of each triangle: its three corners' mean, (N, 3). Faster than `mean(axis=1)`,
    which reduces over a short middle axis slowly."""
    return (tris[:, 0] + tris[:, 1] + tris[:, 2]) / 3


def _corner_min(tris: np.ndarray) -> np.ndarray:
    """The low corner of each triangle's box, (N, 3)."""
    return np.minimum(np.minimum(tris[:, 0], tris[:, 1]), tris[:, 2])


def _corner_max(tris: np.ndarray) -> np.ndarray:
    """The high corner of each triangle's box, (N, 3)."""
    return np.maximum(np.maximum(tris[:, 0], tris[:, 1]), tris[:, 2])


class SurfaceIndex:
    """A triangle set sorted into leaves of `LEAF_SIZE` triangles, for fast queries.

    The triangles sort by the Morton code of their centroids, so each leaf holds triangles that
    sit near each other, and each leaf keeps its box. Build it once and query many times.
    `closest` and `inside` walk a bounding-volume hierarchy built on this same order (see
    `core.bvh`), which drops a whole branch of triangles when its box cannot matter, so the
    answers match the dense routines (`closest_points`, `ray_counts`). The leaf boxes serve the
    pair queries of two indexes (`_pair_rows`).

    `tris` keeps the triangles in their original order, and every face index a query returns
    is an original index. `last_pairs` counts the exact (point, triangle) tests the latest
    query ran: the work a test pins to show the index prunes.
    """

    def __init__(self, tris) -> None:
        tris = _as_tris(tris)
        if len(tris) == 0:
            raise ValueError("a SurfaceIndex needs at least one triangle")
        self.tris = tris
        self._leaf = LEAF_SIZE
        self._order = np.argsort(_morton(_corner_mean(tris)), kind="stable")
        self._sorted = tris[self._order]
        self._tlo, self._thi = _corner_min(self._sorted), _corner_max(self._sorted)
        starts = np.arange(0, len(tris), self._leaf)
        self._lo = np.minimum.reduceat(self._tlo, starts)
        self._hi = np.maximum.reduceat(self._thi, starts)
        self._tree = None
        self.last_pairs = 0

    def __len__(self) -> int:
        return len(self.tris)

    @cached_property
    def _normals(self) -> np.ndarray:
        """Unit normal of each triangle in its original order, formed when first asked."""
        return _unit_normals(self.tris)

    def _bvh(self):
        """The hierarchy over this index, built when a query first asks. (`bvh` reads this module
        when it loads, so the import waits until here.)"""
        if self._tree is None:
            from model_kit.core.bvh import Bvh

            self._tree = Bvh(self)
        return self._tree

    def closest(self, pts) -> Nearest:
        """Nearest point on the surface for each point; the same answer as `closest_points`.

        Equal distances go to the lowest original face index.
        """
        tree = self._bvh()
        near = tree.closest(pts)
        self.last_pairs = tree.last_pairs
        return near

    def inside(self, pts) -> np.ndarray:
        """True where a point is inside the closed mesh: the same answer as `inside`.

        A point outside the index's box is outside. Parity needs a closed mesh, as in `inside`.
        """
        tree = self._bvh()
        out = tree.inside(pts)
        self.last_pairs = tree.last_pairs
        return out

    def deepest(self, pts, groups, floor: float = 0.0):
        """The deepest inside point of each group of `groups` deeper than `floor`: the same answer
        as `inside` and `closest` on every point and a maximum per group, without testing a point
        that cannot win (see `Bvh.deepest`)."""
        tree = self._bvh()
        out = tree.deepest(pts, groups, floor)
        self.last_pairs = tree.last_pairs
        return out


def inside(tris, pts) -> np.ndarray:
    """True where a point is inside a closed mesh: an odd hit count on at least 2 of 3 rays.

    A one-off `SurfaceIndex` query. A point outside the mesh's box is outside.
    """
    tris = _as_tris(tris)
    pts = _as_points(pts)
    if len(tris) == 0:
        return np.zeros(len(pts), dtype=bool)
    return SurfaceIndex(tris).inside(pts)


def closest_points(tris, pts) -> Nearest:
    """Nearest point on the triangle set for each point: a one-off `SurfaceIndex` query.

    Build a `SurfaceIndex` once when many queries hit one surface.

    Caveats for callers. At a shared edge or vertex, two or more triangles can be equally
    near; the lowest original face index wins an exact tie, but rounding can make one of them
    nearer by a hair. When the nearest face is degenerate (zero area), `normal` is (0, 0, 0):
    never divide by it or take a sign from it.
    """
    tris = _as_tris(tris)
    pts = _as_points(pts)
    if len(tris) == 0:
        raise ValueError("closest_points needs at least one triangle")
    return SurfaceIndex(tris).closest(pts)


def face_distances(tris, point) -> tuple[np.ndarray, np.ndarray]:
    """For one point: the distance to each triangle, (N,), and each triangle's unit normal,
    (N, 3), by its winding (zeros for a degenerate one).

    The faces tied for nearest are those within `TIE_DISTANCE` of the least distance. The index
    names one nearest face, the lowest at a tie; a check that needs every tied normal (a point
    inside a round part is as near to a whole ring of faces) reads them here.
    """
    tris = _as_tris(tris)
    p = np.asarray(point, dtype=float).reshape(1, 3)
    q = _closest_on_triangles(np.broadcast_to(p, (len(tris), 3)), tris[:, 0], tris[:, 1],
                              tris[:, 2])
    return np.linalg.norm(p - q, axis=1), _unit_normals(tris)


def signed_volume(tris) -> float:
    """Volume of a closed mesh; positive when the winding faces outward."""
    tris = _as_tris(tris)
    return float(np.einsum("nk,nk->", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0)


def is_closed(tris) -> bool:
    """True when every edge is shared by exactly two triangles.

    Corners match by exact coordinates, so a triangle soup that repeats each vertex is fine.
    """
    tris = _as_tris(tris)
    if len(tris) == 0:
        return False
    _, ids = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    ids = ids.reshape(-1, 3)
    edges = np.concatenate([ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool((counts == 2).all())


def transform(points, m4x4) -> np.ndarray:
    """Apply a 4x4 matrix to points of shape (..., 3)."""
    points = np.asarray(points, dtype=float)
    m = np.asarray(m4x4, dtype=float)
    return points @ m[:3, :3].T + m[:3, 3]


def _plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors that span the plane at right angles to unit vector n."""
    helper = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def _cross2(o, x, y) -> np.ndarray:
    return (x[..., 0] - o[..., 0]) * (y[..., 1] - o[..., 1]) - (
        x[..., 1] - o[..., 1]) * (y[..., 0] - o[..., 0])


def _segment_dist2_2d(p, a, b) -> np.ndarray:
    ab = b - a
    len2 = (ab * ab).sum(axis=-1)
    t = np.where(len2 > 0, ((p - a) * ab).sum(axis=-1) / np.where(len2 > 0, len2, 1.0), 0.0)
    q = a + np.clip(t, 0.0, 1.0)[..., None] * ab
    return ((p - q) ** 2).sum(axis=-1)


def footprint_distance(tris, n, pts) -> np.ndarray:
    """In-plane distance from each point to the footprint of a triangle set: (M,).

    The triangles and the points are projected onto the plane at right angles to `n`.
    The result is the 2-D distance from each projected point to the union of the projected
    triangles: 0 inside any one of them (edges included), else the distance to the nearest
    projected edge. A triangle seen edge-on projects to a segment and counts as that segment
    only: its projected area must exceed a tiny share of its longest projected edge squared
    (`DEGENERATE_SHARE`) before a point can be inside it.
    """
    tris = _as_tris(tris)
    pts = _as_points(pts)
    n = np.asarray(n, dtype=float)
    if n.shape != (3,) or not np.linalg.norm(n) > 0:
        raise ValueError(f"n must be a non-zero 3-vector, got {n!r}")
    if len(tris) == 0:
        raise ValueError("footprint_distance needs at least one triangle")
    u, v = _plane_basis(n / np.linalg.norm(n))
    flat = np.stack([tris @ u, tris @ v], axis=-1)  # (N, 3, 2)
    at = np.stack([pts @ u, pts @ v], axis=-1)  # (M, 2)
    a, b, c = flat[None, :, 0], flat[None, :, 1], flat[None, :, 2]
    area = _cross2(a, b, c)
    # Seen edge-on, a triangle's projected area is float noise: it counts as its segment only.
    longest2 = np.max([((y - x) ** 2).sum(axis=-1) for x, y in ((a, b), (b, c), (c, a))], axis=0)
    has_area = np.abs(area) > DEGENERATE_SHARE * longest2
    out = np.empty(len(pts))
    step = max(1, PAIR_BLOCK // len(tris))
    for start in range(0, len(pts), step):
        p = at[start : start + step, None, :]  # (m, 1, 2)
        d1, d2, d3 = _cross2(a, b, p), _cross2(b, c, p), _cross2(c, a, p)
        within = has_area & (((d1 >= 0) & (d2 >= 0) & (d3 >= 0))
                                | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0)))
        edge2 = np.minimum(np.minimum(_segment_dist2_2d(p, a, b), _segment_dist2_2d(p, b, c)),
                           _segment_dist2_2d(p, c, a))
        out[start : start + step] = np.where(within.any(axis=1), 0.0,
                                             np.sqrt(edge2.min(axis=1)))
    return out


# --- cutting a surface to a box ---------------------------------------------------------


def drop_slivers(tris: np.ndarray) -> np.ndarray:
    """Remove zero-area triangles: they add no surface and have no normal."""
    if len(tris) == 0:
        return tris
    edges = [tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1], tris[:, 0] - tris[:, 2]]
    longest2 = np.max([np.einsum("ij,ij->i", e, e) for e in edges], axis=0)
    area2 = np.linalg.norm(np.cross(edges[0], -edges[2]), axis=1)
    return tris[area2 > DEGENERATE_SHARE * longest2]


def _clip_polygon(poly: list[np.ndarray], axis: int, bound: float, above: bool):
    out = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        da = a[axis] - bound if above else bound - a[axis]
        db = b[axis] - bound if above else bound - b[axis]
        if da >= 0:
            out.append(a)
        if (da >= 0) != (db >= 0):
            out.append(a + (da / (da - db)) * (b - a))
    return out


def clip_to_box(tris: np.ndarray, box) -> np.ndarray:
    """The part of each triangle inside an axis box, as triangles with the same winding."""
    lo, hi = np.asarray(box[0], dtype=float), np.asarray(box[1], dtype=float)
    t_lo, t_hi = tris.min(axis=1), tris.max(axis=1)
    whole = ((t_lo >= lo) & (t_hi <= hi)).all(axis=1)
    touches = ((t_hi >= lo) & (t_lo <= hi)).all(axis=1)
    out = [tris[whole]]
    for tri in tris[touches & ~whole]:
        poly = list(tri)
        for axis in range(3):
            poly = _clip_polygon(poly, axis, lo[axis], True)
            if poly:
                poly = _clip_polygon(poly, axis, hi[axis], False)
            if not poly:
                break
        # A corner on a bound is kept twice by the clipper; drop the repeat.
        poly = [p for i, p in enumerate(poly) if not np.array_equal(p, poly[i - 1])]
        if len(poly) >= 3:
            out.append(np.array([[poly[0], poly[k], poly[k + 1]] for k in range(1, len(poly) - 1)]))
    return drop_slivers(np.concatenate(out))


# --- cuts and gaps between two parts, and welding --------------------------------------------


def _as_index(tris) -> SurfaceIndex:
    return tris if isinstance(tris, SurfaceIndex) else SurfaceIndex(tris)


def _pair_rows(ia: SurfaceIndex, ib: SurfaceIndex, pad: float):
    """Chunks `(pa, pb)` of paired rows into the two indexes' sorted triangle arrays: every pair,
    one from each index, whose boxes (grown by `pad`) overlap. Leaf boxes prune first, then
    triangle boxes. `index._order[row]` is the triangle's original index."""
    near = ((ia._lo[:, None, :] - pad <= ib._hi[None, :, :])
            & (ib._lo[None, :, :] - pad <= ia._hi[:, None, :])).all(axis=2)
    leaf_a, leaf_b = np.nonzero(near)
    step = max(1, PAIR_BLOCK // (ia._leaf * ib._leaf))
    for start in range(0, len(leaf_a), step):
        rows_a = leaf_a[start : start + step, None] * ia._leaf + np.arange(ia._leaf)
        rows_b = leaf_b[start : start + step, None] * ib._leaf + np.arange(ib._leaf)
        shape = (len(rows_a), ia._leaf, ib._leaf)
        pa = np.broadcast_to(rows_a[:, :, None], shape)
        pb = np.broadcast_to(rows_b[:, None, :], shape)
        live = (pa < len(ia.tris)) & (pb < len(ib.tris))
        pa, pb = pa[live], pb[live]
        keep = ((ia._tlo[pa] - pad <= ib._thi[pb]) & (ib._tlo[pb] - pad <= ia._thi[pa])).all(1)
        pa, pb = pa[keep], pb[keep]
        if len(pa):
            yield pa, pb


def _tri_pairs(ia: SurfaceIndex, ib: SurfaceIndex, pad: float):
    """Chunks `(a, b)` of paired triangles, (K, 3, 3) each: every pair, one from each index,
    whose boxes (grown by `pad`) overlap."""
    for pa, pb in _pair_rows(ia, ib, pad):
        yield ia._sorted[pa], ib._sorted[pb]


def _plane_span(tris, n, p0, d):
    """Where each triangle meets the plane through `p0` with unit normal `n`, as a span along
    the unit direction `d`: (met, low t, high t, point at low t, point at high t)."""
    s = np.einsum("kvj,kj->kv", tris - p0[:, None, :], n)
    above, below = s > _PLANE_SLACK, s < -_PLANE_SLACK
    nxt = [1, 2, 0]
    straddle = (above & below[:, nxt]) | (below & above[:, nxt])  # edge i to i+1 crosses
    along = (tris[:, nxt] - tris) * (s / np.where(straddle, s - s[:, nxt], 1.0))[:, :, None]
    pts = np.concatenate([tris, tris + along], axis=1)  # the corners, then the edge crossings
    valid = np.concatenate([~above & ~below, straddle], axis=1)
    t = np.einsum("kpj,kj->kp", pts, d)
    low, high = np.where(valid, t, np.inf), np.where(valid, t, -np.inf)
    i_low, i_high = low.argmin(axis=1), high.argmax(axis=1)
    rows = np.arange(len(tris))
    return (valid.any(axis=1), low[rows, i_low], high[rows, i_high], pts[rows, i_low],
            pts[rows, i_high])


def _pair_spans(a, b):
    """For paired triangles: unit normals, whether their planes cross at a line that both meet,
    and each triangle's span on that line (low t, high t, point at each), from `_plane_span`."""
    na, nb = (np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]) for t in (a, b))
    la, lb = np.linalg.norm(na, axis=1), np.linalg.norm(nb, axis=1)
    live = (la >= _NO_AREA) & (lb >= _NO_AREA)
    na, nb = na / np.where(live, la, 1.0)[:, None], nb / np.where(live, lb, 1.0)[:, None]
    d = np.cross(na, nb)
    ld = np.linalg.norm(d, axis=1)
    live &= ld >= _PARALLEL
    d = d / np.where(live, ld, 1.0)[:, None]
    met_a, *span_a = _plane_span(a, nb, b[:, 0], d)
    met_b, *span_b = _plane_span(b, na, a[:, 0], d)
    return na, nb, live & met_a & met_b, span_a, span_b


def _point_segment_dist(p, a, b):
    ab = b - a
    l2 = np.einsum("kj,kj->k", ab, ab)
    t = np.einsum("kj,kj->k", p - a, ab) / np.where(l2 < _NO_EDGE, 1.0, l2)
    t = np.where(l2 < _NO_EDGE, 0.0, np.clip(t, 0.0, 1.0))
    return np.linalg.norm(a + ab * t[:, None] - p, axis=1)


def _edge_dist(p, tris):
    """Distance from each point to the boundary edges of its triangle."""
    return np.minimum.reduce([_point_segment_dist(p, tris[:, i], tris[:, (i + 1) % 3])
                              for i in range(3)])


def _cuts(a, b, graze: float) -> np.ndarray:
    """The cut test for paired triangles: how deep each pair cuts into each other, 0
    for a pair that does not cut."""
    na, nb, live, (a_lo, a_hi, a_p0, a_p1), (b_lo, b_hi, _, _) = _pair_spans(a, b)
    # each triangle must reach past the other's face, or the two only graze
    reach_a = np.maximum(0.0, -np.einsum("kvj,kj->kv", a - b[:, :1], nb).min(axis=1))
    reach_b = np.maximum(0.0, -np.einsum("kvj,kj->kv", b - a[:, :1], na).min(axis=1))
    live &= np.minimum(reach_a, reach_b) > graze
    t0, t1 = np.maximum(a_lo, b_lo), np.minimum(a_hi, b_hi)
    live &= t1 >= t0
    out = np.zeros(len(a))
    rows = np.nonzero(live)[0]
    if len(rows) == 0:
        return out
    a, b, a_lo, a_hi, a_p0, a_p1, t0, t1 = (x[rows] for x in (a, b, a_lo, a_hi, a_p0, a_p1, t0, t1))
    span = a_hi - a_lo
    flat = span <= _NO_SPAN
    best = np.zeros(len(rows))
    for k in range(CUT_SAMPLES):
        t = t0 + (t1 - t0) * (k / (CUT_SAMPLES - 1))
        frac = np.where(flat, 0.0, (t - a_lo) / np.where(flat, 1.0, span))
        p = a_p0 + (a_p1 - a_p0) * frac[:, None]
        best = np.maximum(best, np.minimum(_edge_dist(p, a), _edge_dist(p, b)))
    out[rows] = best
    return out


def tri_tri_cut(a, b, graze: float = 0.0) -> float:
    """The deepest cut between two triangle soups, 0 when they do not cut.

    A pair of triangles cuts when each reaches more than `graze` past the other's face (along
    that face's normal) and their planes' intersection segment runs inside both. The pair's
    depth is the segment's deepest point, measured at `CUT_SAMPLES` points as the smaller of
    its distances to each triangle's boundary edges. Triangles that lie flush, or touch only
    along an edge, give 0, so a foot standing on a deck does not cut it, and a post sunk 2 mm
    into a deck cuts about 2 mm. This is the cut test over every pair whose boxes
    overlap, taken from two `SurfaceIndex` leaf sets (a soup given as an array is indexed here;
    give an index to reuse one). `graze` is the shallowest reach that counts as a cut.
    """
    ia, ib = _as_index(a), _as_index(b)
    best = 0.0
    for ta, tb in _tri_pairs(ia, ib, 0.0):
        best = max(best, float(_cuts(ta, tb, graze).max()))
    return best


def self_cuts(tris, graze: float = 0.0, ids=None, together=None
              ) -> tuple[np.ndarray, np.ndarray]:
    """The faces of one soup that cut each other: (pairs, depths).

    A pair is two faces that share no vertex and cut each other as `tri_tri_cut` defines a cut
    (each reaches more than `graze` past the other's face, and the cut is deeper than `graze`).
    `pairs` is (K, 2) original face indexes with the lower first, each pair once, in order; `depths`
    is (K,) the depth of each cut. `ids` (N, 3) names each face's corners; two faces share a
    vertex when they have an id in common. Left out, the ids come from equal coordinates, so a
    caller whose mesh splits a vertex (a glTF seam) gives welded ids (`weld`).

    `together` (N,) is for a soup that has been posed: faces with the same id of 0 or more
    have moved as one rigid body since the soup was last looked at, so they stand as they did and
    a pair of them is left out. A face with an id under 0 moved on its own and is tested with all.
    Only the faces that may have changed their stand to each other are paired, which is most of
    the saving when a limb turns and the rest of the body does not.
    """
    tris = _as_tris(tris)
    empty = np.zeros((0, 2), dtype=np.intp), np.zeros(0)
    if len(tris) < 2:
        return empty
    if ids is None:
        ids = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)[1].reshape(-1, 3)
    ids = np.asarray(ids)
    if ids.shape != (len(tris), 3):
        raise ValueError(f"ids must be ({len(tris)}, 3), got {ids.shape}")
    if together is None:
        subsets = [np.arange(len(tris))]
        jobs = [(0, 0)]
    else:
        together = np.asarray(together)
        if together.shape != (len(tris),):
            raise ValueError(f"together must be ({len(tris)},), got {together.shape}")
        subsets = [np.nonzero(together == k)[0] for k in np.unique(together[together >= 0])]
        subsets.append(np.nonzero(together < 0)[0])  # the faces that moved alone, last
        jobs = [(a, b) for a in range(len(subsets)) for b in range(a + 1, len(subsets))]
        jobs.append((len(subsets) - 1, len(subsets) - 1))
    made: dict[int, SurfaceIndex] = {}

    def index_of(k: int) -> SurfaceIndex | None:
        if len(subsets[k]) and k not in made:
            made[k] = SurfaceIndex(tris[subsets[k]])
        return made.get(k)

    pairs: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    for a, b in jobs:
        ia, ib = index_of(a), index_of(b)
        if ia is None or ib is None:
            continue
        for pa, pb in _pair_rows(ia, ib, 0.0):
            i, j = subsets[a][ia._order[pa]], subsets[b][ib._order[pb]]
            keep = i < j if a == b else np.ones(len(i), dtype=bool)
            keep &= ~(ids[i][:, :, None] == ids[j][:, None, :]).any(axis=(1, 2))
            if not keep.any():
                continue
            i, j = i[keep], j[keep]
            cut = _cuts(tris[i], tris[j], graze)
            over = cut > graze
            pairs.append(np.stack([np.minimum(i, j)[over], np.maximum(i, j)[over]], axis=1))
            depths.append(cut[over])
    if not pairs:
        return empty
    pairs_all, depths_all = np.concatenate(pairs), np.concatenate(depths)
    order = np.lexsort((pairs_all[:, 1], pairs_all[:, 0]))
    return pairs_all[order], depths_all[order]


def self_crossings(tris, graze: float = 0.0, ids=None, together=None) -> np.ndarray:
    """Index pairs (K, 2) of the faces of one soup that share no vertex and cut each other; see
    `self_cuts`, which also gives the depths."""
    return self_cuts(tris, graze, ids, together)[0]


def _segment_dist(p1, q1, p2, q2) -> np.ndarray:
    """Distance between segments p1q1 and p2q2, row by row (Ericson, RTCD 5.1.9). A segment
    shorter than `_NO_EDGE` squared is a point."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = np.einsum("kj,kj->k", d1, d1), np.einsum("kj,kj->k", d2, d2)
    b, c = np.einsum("kj,kj->k", d1, d2), np.einsum("kj,kj->k", d1, r)
    f = np.einsum("kj,kj->k", d2, r)
    a_pt, e_pt = a < _NO_EDGE, e < _NO_EDGE
    a_s, e_s = np.where(a_pt, 1.0, a), np.where(e_pt, 1.0, e)
    denom = a * e - b * b
    s = np.where(denom > 0.0, np.clip((b * f - c * e) / np.where(denom > 0.0, denom, 1.0),
                                      0.0, 1.0), 0.0)
    t = (b * s + f) / e_s
    s = np.where(t < 0.0, np.clip(-c / a_s, 0.0, 1.0),
                 np.where(t > 1.0, np.clip((b - c) / a_s, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    s, t = np.where(a_pt, 0.0, s), np.where(e_pt, 0.0, t)
    return np.linalg.norm((p1 + d1 * s[:, None]) - (p2 + d2 * t[:, None]), axis=1)


def _gaps(a, b) -> np.ndarray:
    """The distance between paired triangles: 0 when they cross, else the least of the six
    corner-to-face distances and the nine edge-to-edge distances."""
    _, _, live, (a_lo, a_hi, _, _), (b_lo, b_hi, _, _) = _pair_spans(a, b)
    out = np.where(live & (np.minimum(a_hi, b_hi) >= np.maximum(a_lo, b_lo)), 0.0, np.inf)
    for x, y in ((a, b), (b, a)):
        for i in range(3):
            near = _closest_on_triangles(x[:, i], y[:, 0], y[:, 1], y[:, 2])
            out = np.minimum(out, np.linalg.norm(x[:, i] - near, axis=1))
    for i in range(3):
        for j in range(3):
            out = np.minimum(out, _segment_dist(a[:, i], a[:, (i + 1) % 3],
                                                b[:, j], b[:, (j + 1) % 3]))
    return out


def surface_gap(a, b, within: float) -> float:
    """The distance between two triangle soups' surfaces, 0 where they cross or touch.

    Exact when it is at most `within`. A larger answer is only some value over `within`, or
    infinity when no triangles of the two lie within `within` of each other: pairs further apart
    than that are never tested. A soup may be given as an array or as a `SurfaceIndex`.
    """
    ia, ib = _as_index(a), _as_index(b)
    best = float("inf")
    for ta, tb in _tri_pairs(ia, ib, within):
        best = min(best, float(_gaps(ta, tb).min()))
    return best


def _near_labels(pts: np.ndarray, tol: float) -> np.ndarray:
    """For distinct points, the index of the seed each one merges to. Going through the points
    in order, a point not yet merged becomes a seed and takes every unmerged point within `tol`
    of it; a point never joins through another point, so a row of corners cannot chain. Points
    are binned into cells of side `tol`, so two points within `tol` sit in the same cell or in
    cells next to each other."""
    cells = np.floor(pts / tol).astype(np.int64)
    table, cell_of = np.unique(cells, axis=0, return_inverse=True)
    cell_of = cell_of.reshape(-1)
    order = np.argsort(cell_of, kind="stable")
    start = np.searchsorted(cell_of[order], np.arange(len(table)))
    count = np.bincount(cell_of, minlength=len(table))
    ends_a: list[np.ndarray] = []
    ends_b: list[np.ndarray] = []
    for step in itertools.product((-1, 0, 1), repeat=3):
        if step < (0, 0, 0):  # each pair of neighbouring cells once
            continue
        _, ids = np.unique(np.vstack([table, table + step]), axis=0, return_inverse=True)
        ids = ids.reshape(-1)
        where = np.full(ids.max() + 1, -1)
        where[ids[: len(table)]] = np.arange(len(table))
        neighbour = where[ids[len(table):]]
        cell_a = np.nonzero(neighbour >= 0)[0]
        cell_b = neighbour[cell_a]
        pairs = count[cell_a] * count[cell_b]
        which = np.repeat(np.arange(len(cell_a)), pairs)
        local = np.arange(pairs.sum()) - np.repeat(np.cumsum(pairs) - pairs, pairs)
        a = order[start[cell_a][which] + local // count[cell_b][which]]
        b = order[start[cell_b][which] + local % count[cell_b][which]]
        keep = (a < b if step == (0, 0, 0) else slice(None))
        a, b = a[keep], b[keep]
        close = np.linalg.norm(pts[a] - pts[b], axis=1) <= tol
        ends_a.append(a[close])
        ends_b.append(b[close])
    a, b = np.concatenate(ends_a), np.concatenate(ends_b)
    swap = a > b
    a, b = np.where(swap, b, a), np.where(swap, a, b)
    order = np.argsort(a, kind="stable")
    a, b = a[order], b[order]
    first = np.searchsorted(a, np.arange(len(pts) + 1))
    seed = np.full(len(pts), -1)
    for i in range(len(pts)):  # in sort order, the first corner of a cluster is its seed
        if seed[i] < 0:
            seed[i] = i
            near = b[first[i] : first[i + 1]]
            seed[near[seed[near] < 0]] = i
    return seed


def weld(tris, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Weld a triangle soup: merge corners that lie within `tol` of a seed corner, and share
    equal corners.

    Going through the distinct corners in sort order (x, then y, then z), a corner not yet merged
    becomes a seed and takes every unmerged corner within `tol` of it. A corner joins a seed
    only, never another corner that joined it, so a row of corners each `tol` from the next
    makes several vertices and not one. This is Blender's `remove_doubles(dist)` with a fixed
    order. Returns `(verts, faces)`: `verts` (V, 3) are the seeds, and `faces` (N, 3) index
    them, one row per input triangle in order. A triangle whose corners all merge keeps its row
    and repeats an index.
    """
    tris = _as_tris(tris)
    if not tol > 0:
        raise ValueError(f"weld tolerance must be above 0, got {tol!r}")
    pts, inverse = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    heads, cluster = np.unique(_near_labels(pts, tol), return_inverse=True)
    return pts[heads], cluster.reshape(-1)[inverse.reshape(-1)].reshape(-1, 3)


def shells(faces, labels) -> list[tuple[int, np.ndarray]]:
    """The connected shells of one mesh: `[(label, face indexes)]`, in order of each shell's
    lowest face.

    `faces` (T, 3) index vertices, and `labels` (V,) hold one integer per vertex (a body's
    strongest vertex group). Two faces belong to one shell when a chain of faces joins them
    through shared vertex indices. A shell's label is the label most of its vertices hold (the
    lowest label at a tie). A vertex that no face uses belongs to no shell. A mesh of several
    closed boxes in one vertex list gives one shell per box, so each gets its own
    `SurfaceIndex`: parity `inside()` is wrong over shells that overlap in one index.
    """
    faces = np.asarray(faces)
    labels = np.asarray(labels)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (T, 3), got {faces.shape}")
    if len(faces) == 0:
        return []
    if faces.min() < 0 or faces.max() >= len(labels):
        raise ValueError(f"faces name vertices outside 0..{len(labels) - 1}")
    parent = np.arange(len(labels))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    for a, b, c in faces.tolist():
        for u, v in ((a, b), (a, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    root = np.array([find(int(v)) for v in faces[:, 0]])
    out: list[tuple[int, np.ndarray]] = []
    for r in dict.fromkeys(root.tolist()):  # first-seen order is lowest-face order
        rows = np.nonzero(root == r)[0]
        verts = np.unique(faces[rows])
        out.append((int(np.bincount(labels[verts]).argmax()), rows))
    return out


# Cut endpoints within this share of the two meshes' box diagonal are one point when no weld is
# given. The endpoints of neighbouring cut segments agree to float rounding (about 1e-16 of the
# size); this sits far above that and far below any real edge. A float guard, not a measuring
# tolerance: a check passes its own weld.
_CUT_WELD_OF_SIZE = 1e-9


def _cut_segments(ia: SurfaceIndex, ib: SurfaceIndex) -> np.ndarray:
    """The segment where each cutting pair of triangles meets: (K, 2, 3) endpoints, the part of
    the planes' intersection line that runs inside both triangles. A pair that only touches at a
    point, or lies flush, gives none."""
    out: list[np.ndarray] = []
    for ta, tb in _tri_pairs(ia, ib, 0.0):
        _, _, live, (a_lo, a_hi, a_p0, a_p1), (b_lo, b_hi, b_p0, b_p1) = _pair_spans(ta, tb)
        live &= np.minimum(a_hi, b_hi) - np.maximum(a_lo, b_lo) > _NO_SPAN
        start = np.where((a_lo >= b_lo)[:, None], a_p0, b_p0)  # the later of the two lows
        end = np.where((a_hi <= b_hi)[:, None], a_p1, b_p1)  # the earlier of the two highs
        out.append(np.stack([start[live], end[live]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2, 3))


def _chain(ends: np.ndarray) -> list[list[int]]:
    """Chain edges (E, 2) of vertex ids into lines, as lists of ids. A connected set of edges
    with at most two odd-degree vertices is one line, whichever order its edges come in: from an
    odd vertex (an open line), else from its lowest vertex (a closed loop, first id repeated at
    the end). A tour that closes on the way is spliced into, where it passes a vertex that still
    has edges, so a figure of eight (a vertex of degree 4) is one loop and not two. A set with
    more odd vertices cannot be one line; it splits greedily into several. Open lines come
    first, then loops, each in order of its first vertex."""
    around: dict[int, list[int]] = {}
    for e, (u, v) in enumerate(ends.tolist()):
        around.setdefault(u, []).append(e)
        around.setdefault(v, []).append(e)
    unused = set(range(len(ends)))

    def step(node: int) -> int | None:
        edge = next((e for e in around[node] if e in unused), None)
        if edge is not None:
            unused.discard(edge)
        return edge

    def tour(start: int) -> list[int]:
        """Hierholzer's walk from `start`: a line over every edge of its component."""
        stack, path = [start], []
        while stack:
            edge = step(stack[-1])
            if edge is None:
                path.append(stack.pop())
            else:
                stack.append(int(ends[edge].sum() - stack[-1]))
        return path[::-1]

    def greedy(start: int) -> list[int]:
        path = [start]
        while (edge := step(path[-1])) is not None:
            path.append(int(ends[edge].sum() - path[-1]))
        return path

    seen: set[int] = set()
    lines: list[list[int]] = []
    for first in sorted(around):
        if first in seen:
            continue
        component, todo = [], [first]
        seen.add(first)
        while todo:
            node = todo.pop()
            component.append(node)
            for e in around[node]:
                other = int(ends[e].sum() - node)
                if other not in seen:
                    seen.add(other)
                    todo.append(other)
        odd = sorted(n for n in component if len(around[n]) % 2)
        if len(odd) <= 2:
            lines.append(tour(odd[0] if odd else min(component)))
        else:
            lines += [greedy(n) for n in odd if any(e in unused for e in around[n])]
            while any(e in unused for n in component for e in around[n]):
                lines.append(greedy(next(n for n in sorted(component)
                                         if any(e in unused for e in around[n]))))
    return sorted(lines, key=lambda line: (line[0] == line[-1], line[0]))


def cut_loops(a, b, weld: float | None = None) -> list[np.ndarray]:
    """Where two meshes cut each other, as lines: each a (K, 3) array of points along the cut.

    Every pair of triangles, one from each mesh, that cuts gives a segment (`_cut_segments`).
    Segment ends within `weld` of each other are one point (the default is a float guard,
    `_CUT_WELD_OF_SIZE` times the meshes' box diagonal), and a segment that two triangles give
    twice, where the cut runs along a shared edge, counts once. The segments chain into lines: an
    open line runs from one end to another, and a closed loop returns to its start, its last
    point equal to its first. A cut that touches itself, where more than two segments meet, is one
    loop (`_chain`), however its segments are ordered. A cut with more than two odd-degree points
    (four or more line ends) cannot be one line, and `_chain` splits it greedily: the line count
    is that split's own, the same on every run, and another split could give a different count.
    Any item text that prints the count (`inset_seat`'s "in <n> loops") carries that caveat. Open
    lines come first, then loops, each in order of its first point. A mesh pair that does not
    cut gives `[]`. Either mesh may be given as an array or a `SurfaceIndex`.
    """
    ia, ib = _as_index(a), _as_index(b)
    segments = _cut_segments(ia, ib)
    if len(segments) == 0:
        return []
    if weld is None:
        every = np.concatenate([ia.tris.reshape(-1, 3), ib.tris.reshape(-1, 3)])
        weld = _CUT_WELD_OF_SIZE * float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    if not weld > 0:
        raise ValueError(f"weld tolerance must be above 0, got {weld!r}")
    pts, inverse = np.unique(segments.reshape(-1, 3), axis=0, return_inverse=True)
    heads, cluster = np.unique(_near_labels(pts, weld), return_inverse=True)
    ends = cluster.reshape(-1)[inverse.reshape(-1)].reshape(-1, 2)
    ends = np.unique(np.sort(ends[ends[:, 0] != ends[:, 1]], axis=1), axis=0)
    if len(ends) == 0:
        return []
    points = pts[heads]
    lines = _chain(ends)
    return [points[line] for line in lines]

# --- mass and contact (the bearing check) ----------------------------------------------------


def centre_of_mass(tris) -> np.ndarray:
    """The centroid of a closed mesh's volume, (3,): uniform density.

    Signed tetrahedra from the origin, as `signed_volume` forms them: each triangle with the
    origin makes one, its centroid is the mean of its corners and the origin, and its volume is
    the triangle's share of `signed_volume`. A mesh that is open, empty or wound inward (volume
    at or under zero) has no centre of mass and raises `ValueError`.
    """
    tris = _as_tris(tris)
    if not is_closed(tris):
        raise ValueError("the centre of mass needs a closed mesh")
    vol = np.einsum("nk,nk->n", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    total = float(vol.sum())
    if not total > 0.0:
        raise ValueError(f"the centre of mass needs an outward-wound mesh, volume is {total!r}")
    return (vol[:, None] * tris.sum(axis=1) / 4.0).sum(axis=0) / total


def contact_faces(a, b, within: float, sunk: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """The faces of two soups that bear on each other: (faces of `a`, faces of `b`), each a
    boolean mask over that soup's triangles in their given order.

    A pair of faces is in contact when they lie within `within` of each other (`_gaps`: 0 where
    they cross), their unit normals have a negative dot product, and each lies in front of the
    other: some corner of it is on the outward side of the other's plane, or no more than `sunk`
    behind it (a part may be sunk a little into its host). A face that meets the other soup only
    along an edge or lies beside it is not in contact. Neither are two faces that look away from
    each other: the far side of a thin part, or the opposite sides of two flush parts, whose
    normals oppose but whose planes are back to back.
    """
    ia, ib = _as_index(a), _as_index(b)
    mask_a = np.zeros(len(ia.tris), dtype=bool)
    mask_b = np.zeros(len(ib.tris), dtype=bool)
    for pa, pb in _pair_rows(ia, ib, within):
        ta, tb = ia._sorted[pa], ib._sorted[pb]
        na, nb = ia._normals[ia._order[pa]], ib._normals[ib._order[pb]]
        near = _gaps(ta, tb) <= within
        facing = np.einsum("kj,kj->k", na, nb)
        in_front_b = np.einsum("kvj,kj->kv", tb - ta[:, :1], na).max(axis=1) >= -sunk - _PLANE_SLACK
        in_front_a = np.einsum("kvj,kj->kv", ta - tb[:, :1], nb).max(axis=1) >= -sunk - _PLANE_SLACK
        hit = near & (facing < -NO_DIRECTION) & in_front_a & in_front_b
        mask_a[ia._order[pa[hit]]] = True
        mask_b[ib._order[pb[hit]]] = True
    return mask_a, mask_b


# --- how far two parts share space, and how thick a part is -----------------------------------

# The directions `overlap_depth` measures the shared space along: the 3 axes, the 6 face
# diagonals and the 4 body diagonals of a cube, 13 unit vectors. They are a stated, fixed set (a
# stand-in for the shortest move that separates two parts), in whatever frame the points are in.
DEPTH_DIRECTIONS = np.array(
    [d for d in itertools.product((-1.0, 0.0, 1.0), repeat=3)
     if any(d) and next(c for c in d if c) > 0.0], dtype=float)
DEPTH_DIRECTIONS /= np.linalg.norm(DEPTH_DIRECTIONS, axis=1, keepdims=True)
# The same set with both signs: 26 unit vectors, the ways a part may be moved or a point may leave.
# Crossings, bearing and take-apart all read this one array.
DEPTH_WAYS = np.vstack([DEPTH_DIRECTIONS, -DEPTH_DIRECTIONS])



def _lattice(tris: np.ndarray, spacing: float) -> np.ndarray:
    """Points on the triangles, neighbours at most `spacing` apart along the lattice lines.

    Each triangle gets a lattice from the corner opposite its longest edge, stepping at most
    `spacing` along the two edges there, so a thin sliver costs a line of points and not a
    plane of them. The longest edge is sampled too, and the corners are always points.
    """
    corners = np.roll(tris, -1, axis=1) - tris  # edge i runs from corner i to corner i + 1
    length = np.linalg.norm(corners, axis=2)
    last = length.argmax(axis=1)  # the longest edge; the corner opposite it is `last + 2`
    rows = np.arange(len(tris))
    apex = (last + 2) % 3
    origin = tris[rows, apex]
    e1 = tris[rows, (apex + 1) % 3] - origin
    e2 = tris[rows, (apex + 2) % 3] - origin
    n1 = np.maximum(1, np.ceil(np.linalg.norm(e1, axis=1) / spacing)).astype(int)
    n2 = np.maximum(1, np.ceil(np.linalg.norm(e2, axis=1) / spacing)).astype(int)
    n3 = np.maximum(1, np.ceil(length[rows, last] / spacing)).astype(int)
    out = []
    shapes = np.stack([n1, n2, n3], axis=1)
    for shape in np.unique(shapes, axis=0):
        pick = (shapes == shape).all(axis=1)
        k1, k2, k3 = (int(s) for s in shape)
        u, v = np.meshgrid(np.arange(k1 + 1) / k1, np.arange(k2 + 1) / k2, indexing="ij")
        keep = (u + v) <= 1.0 + 1e-12
        uv = np.stack([u[keep], v[keep]], axis=1)
        o, a, b = origin[pick], e1[pick], e2[pick]
        out.append((o[:, None] + uv[None, :, 0, None] * a[:, None]
                    + uv[None, :, 1, None] * b[:, None]).reshape(-1, 3))
        s = (np.arange(k3 + 1) / k3)[None, :, None]
        out.append((o[:, None] + a[:, None] + s * (b - a)[:, None]).reshape(-1, 3))
    return np.concatenate(out)


def surface_samples(tris, spacing: float) -> np.ndarray:
    """Points on the surface, neighbouring points at most `spacing` apart, each point once.

    A corner or an edge point that several triangles share is one point, so a fine mesh does not
    count its vertices five times over. The order is sorted, so the answer repeats.
    """
    tris = _as_tris(tris)
    if not spacing > 0:
        raise ValueError(f"spacing must be above 0, got {spacing!r}")
    if len(tris) == 0:
        return np.zeros((0, 3))
    return np.unique(_lattice(tris, spacing), axis=0)


def overlap_depth(points) -> float:
    """How far two parts share space, from the sample points that lie inside both of them.

    The smallest extent of the points over `DEPTH_DIRECTIONS`: the width of the shared region
    along its thinnest direction, a stand-in for the shortest move that separates the parts
    (penetration depth, as collision detection defines it). Two parts that overlap by 90% share
    nine tenths of a width, and a rod through a plate shares the plate's thickness. 0 for no
    points. The points are surface samples of each part that the other contains.
    """
    pts = _as_points(points)
    if len(pts) == 0:
        return 0.0
    along = pts @ DEPTH_DIRECTIONS.T
    return float((along.max(axis=0) - along.min(axis=0)).min())


def _ray_hits(tris: np.ndarray, origins: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """For each ray, the distance to its nearest hit past the origin's own surface
    (`_RAY_T_MIN`), inf for none. Moller-Trumbore, one unit direction per row."""
    a = tris[:, 0]
    e1, e2 = tris[:, 1] - a, tris[:, 2] - a
    near = np.full(len(origins), np.inf)
    step = max(1, PAIR_BLOCK // len(tris))
    for start in range(0, len(origins), step):
        o, d = origins[start : start + step], dirs[start : start + step]
        pvec = np.cross(d[:, None, :], e2[None, :, :])  # (rays, tris, 3)
        det = np.einsum("nk,rnk->rn", e1, pvec)
        scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        live = np.abs(det) > 1e-12 * scale[None, :]
        inv = 1.0 / np.where(live, det, 1.0)
        tvec = o[:, None, :] - a[None, :, :]
        qvec = np.cross(tvec, e1[None, :, :])
        u = np.einsum("rnk,rnk->rn", tvec, pvec) * inv
        v = np.einsum("rk,rnk->rn", d, qvec) * inv
        t = np.einsum("nk,rnk->rn", e2, qvec) * inv
        hit = (live & (u >= -_RAY_EDGE_SLACK) & (v >= -_RAY_EDGE_SLACK)
               & (u + v <= 1 + _RAY_EDGE_SLACK) & (t > _RAY_T_MIN))
        near[start : start + step] = np.where(hit, t, np.inf).min(axis=1)
    return near


def local_thickness(tris, point) -> float:
    """How thick a closed mesh is at a point: the shape diameter of Shapira, Shamir and Cohen-Or
    (2008, "Consistent mesh partitioning and skeletonisation using the shape diameter function").

    From the nearest surface point, rays go into the part inside a cone around the inward
    normal (`CONE_DEG`, `CONE_AROUND` around each ring). Rays whose length is more than one
    standard deviation from the median are dropped (a rod's end face sends its straight ray the
    whole length of the rod; the rest of the cone reads the rod's width), and the rest average,
    weighted toward the normal (`CONE_WEIGHT_DEG`). A slab reads a little over its thickness,
    since slanted rays run longer. Works whichever way the faces wind. Raises `ValueError` on an
    open mesh, which has no inside, and when no ray finds an opposite surface.
    """
    tris = _as_tris(tris)
    p = _as_points(np.asarray(point, dtype=float).reshape(1, 3), "point")[0]
    if not is_closed(tris):
        raise ValueError("local_thickness needs a closed mesh: an open one has no inside")
    dist, normals = face_distances(tris, p)
    usable = np.nonzero(np.linalg.norm(normals, axis=1) > 0.0)[0]
    if len(usable) == 0:
        raise ValueError("local_thickness found no face with an area")
    face = int(usable[dist[usable].argmin()])
    t = tris[face]
    foot = _closest_on_triangles(p[None], t[None, 0], t[None, 1], t[None, 2])[0]
    toward = normals[face] * (-1.0 if signed_volume(tris) >= 0.0 else 1.0)  # into the part
    u, v = _plane_basis(toward)
    angles, dirs = [0.0], [toward]
    for deg in CONE_DEG[1:]:
        a = math.radians(deg)
        for k in range(CONE_AROUND):
            phi = 2.0 * math.pi * k / CONE_AROUND
            dirs.append(math.cos(a) * toward
                        + math.sin(a) * (math.cos(phi) * u + math.sin(phi) * v))
            angles.append(deg)
    dirs = np.array(dirs)
    reach = _ray_hits(tris, np.broadcast_to(foot, dirs.shape), dirs)
    found = np.isfinite(reach)
    if not found.any():
        raise ValueError("local_thickness: no ray hit an opposite surface")
    reach, deg = reach[found], np.array(angles)[found]
    keep = np.abs(reach - np.median(reach)) <= reach.std()
    if not keep.any():
        keep[:] = True
    weight = 1.0 / (1.0 + deg[keep] / CONE_WEIGHT_DEG)
    return float((weight * reach[keep]).sum() / weight.sum())


def thinner_thickness(a_tris, b_tris, point) -> float:
    """How thick the thinner of two parts is at a point: the smaller of their `local_thickness`
    there. Crossings, take-apart and bearing read the depth two parts may share from it (half
    this, `CROSS_OF_PART_THICKNESS`).

    Raises `Unmeasurable` (a check turns it into an `Unmeasured` line for the item) when either
    part's `local_thickness` raises: the part is open, or sound nowhere near the point. It never
    falls back to a thickness read somewhere else on the part.
    """
    reads = []
    for which, tris in (("first", a_tris), ("second", b_tris)):
        try:
            reads.append(local_thickness(tris, point))
        except ValueError as err:
            raise Unmeasurable(f"the {which} part has no thickness there: {err}") from err
    return min(reads)
