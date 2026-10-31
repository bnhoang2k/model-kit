"""A bounding-volume hierarchy (BVH) over a mesh's triangles, for many-point queries.

Think of a phone book with tabs: a query reads a few tabs to find the page, not every page.
`geom.SurfaceIndex` sorts the triangles along a Morton curve (nearby triangles, nearby places in
the list) and keeps one box per run of `LEAF_SIZE` triangles. A query still scanned every one of
those boxes and then exact-tested every triangle of every leaf that might matter. This module
builds a tree on top of the same sorted order, so a query walks down from the root and drops a
whole branch as soon as its box cannot matter. Core imports only the standard library and numpy.

There is one index, not two. A `SurfaceIndex` owns the triangles and their order; a `Bvh` reads
them and adds the tree. `SurfaceIndex.closest` and `SurfaceIndex.inside` call the `Bvh`, so every
check that holds a `SurfaceIndex` gets the speed with no edit. The brute-force functions stay as the
reference (`geom._closest_dense`, `geom.ray_counts`) and the tests hold both to them.

What a query answers is fixed by the exact per-triangle arithmetic. `closest` runs
`geom._closest_on_triangles` itself. `inside` forms the same Moller-Trumbore terms as
`geom.ray_counts` but sums them in another order (a per-row dot product, not a matrix product), so a
value can differ from `ray_counts` in its last bits; the float slack on the barycentric edges is
what keeps a ray through an edge counted, and the tests hold the two equal. The tree only decides
which triangles get that arithmetic, and it drops one only when a box bound proves it cannot change
the answer.

Other part-against-part checks reuse these two queries:
- `closest(points)`: the nearest point, distance, face and normal. Clearance probes every vertex of
  a part against the neighbour with it; bearing reads the face and its normal.
- `inside(points)`: ray parity. Overlap and crossings use it for the vertices of one shell in
  another.
- `deepest(points, groups, floor)`: the deepest inside point of each group, above a floor. Test C of
  overlap needs one maximum per (vertex group, shell), not a depth for every vertex. A crossing
  depth (the deepest point of one part inside another) is the same question with one group.
"""

from typing import NamedTuple

import numpy as np

from model_kit.core import geom
from model_kit.core.geom import (
    _BOUND_SLACK,
    _RAY_EDGE_SLACK,
    _RAY_T_MIN,
    _SLAB_PAD,
    RAYS,
    Nearest,
    _unit_normals,
)

# Triangles per leaf of the tree. A structural constant: it sets how deep the tree is, never what a
# query answers (the tests run it from 1 to more than the mesh holds).
BVH_LEAF = 4

# `deepest` first measures every SEED_STRIDE-th point of each group. Structural: it sets how many
# points seed the bounds, never what the query answers (the tests run it from 1 up).
SEED_STRIDE = 8

# A `closest` walk starts at the lowest tree level that holds at most this many (point, node)
# pairs. Structural, a speed knob: it sets where the walk starts, never what a query answers.
START_PAIRS = 2048

_NO_FACE = np.iinfo(np.intp).max


class _Best:
    """The running nearest hit per point, merged block by block.

    A smaller squared distance wins; on an exact tie the lower original face index wins.
    """

    def __init__(self, n: int) -> None:
        self.d2 = np.full(n, np.inf)
        self.point = np.zeros((n, 3))
        self.face = np.full(n, _NO_FACE, dtype=np.intp)

    def merge(self, rows, d2, q, face) -> None:
        new = self.d2.copy()
        np.minimum.at(new, rows, d2)
        self.face[new < self.d2] = _NO_FACE  # a nearer hit voids the old face
        won = d2 == new[rows]
        np.minimum.at(self.face, rows[won], face[won])
        keep = won & (face == self.face[rows])
        self.point[rows[keep]] = q[keep]
        self.d2 = new


def _levels(lo: np.ndarray, hi: np.ndarray, leaf: int):
    """Boxes of a tree over per-triangle boxes (N, k): level 0 joins `leaf` triangles, each level
    above joins neighbours in pairs, up to one root. Each level is a (k, nodes) pair of rows, one
    contiguous row per axis."""
    starts = np.arange(0, len(lo), leaf)
    lo, hi = np.minimum.reduceat(lo, starts), np.maximum.reduceat(hi, starts)
    los, his = [np.ascontiguousarray(lo.T)], [np.ascontiguousarray(hi.T)]
    while len(lo) > 1:
        starts = np.arange(0, len(lo), 2)
        lo, hi = np.minimum.reduceat(lo, starts), np.maximum.reduceat(hi, starts)
        los.append(np.ascontiguousarray(lo.T))
        his.append(np.ascontiguousarray(hi.T))
    return los, his


def _far2(lo, hi, p) -> np.ndarray:
    """For each (point, box) pair, per axis rows `lo`, `hi`, `p` (3, K): an upper bound on the
    squared distance from the point to the nearest triangle in the box.

    The box is the tight box of its triangles, so each of its six faces touches a triangle. A
    triangle touching the near face along one axis is at most as far as that face plus, along the
    other two axes, the far side of the box (Roussopoulos' MINMAXDIST). The least over the three
    axes bounds the nearest triangle. The three axes go through one (3, K) array at a time, since a
    small batch costs more in numpy calls than in arithmetic; each element and the order of the
    sums are the per-axis loop's."""
    mid = (lo + hi) / 2
    near = np.where(p <= mid, lo, hi)
    far = np.where(p >= mid, lo, hi)
    d_far = (p - far) ** 2
    swap = (p - near) ** 2 - d_far
    return (d_far[0] + d_far[1] + d_far[2]) + np.minimum(np.minimum(swap[0], swap[1]), swap[2])


def _bounds2(lo, hi, p):
    """For each (point, box) pair (rows as in `_far2`): the squared distance from the point to the
    box, and the `_far2` bound."""
    gap = np.maximum(np.maximum(lo - p, p - hi), 0.0)
    gap *= gap
    return (gap[0] + gap[1] + gap[2]), _far2(lo, hi, p)


class Deepest(NamedTuple):
    """The answer of `Bvh.deepest`, one row per group that has a point deeper than the floor."""

    group: np.ndarray  # (K,) group ids, ascending
    depth: np.ndarray  # (K,) the deepest inside point's distance to the surface
    row: np.ndarray  # (K,) its row in the query (the lowest row among equal depths)


class _Ray(NamedTuple):
    """What a ray direction needs of a mesh, formed once per ray and mesh."""

    live: np.ndarray  # (N,) triangle faces the ray (not parallel to it)
    inv: np.ndarray  # (N,) 1 / determinant
    mats: tuple  # three (N, 3) Moller-Trumbore row vectors
    offs: list  # three (N,) offsets, one per row vector
    across: np.ndarray  # (2, 3) two unit vectors across the ray
    tri_lo: np.ndarray  # (2, N) outline of each triangle seen along the ray
    tri_hi: np.ndarray
    node_lo: list  # per tree level, (2, nodes) outline boxes
    node_hi: list


class Bvh:
    """A tree of boxes over the triangles of a `SurfaceIndex`, answering `closest` and `inside`.

    Build one from a `SurfaceIndex` (or triangles, which become one). Level 0 holds one box per
    `BVH_LEAF` triangles of the index's Morton order; each level above joins neighbours in pairs, up
    to one root. Node j of a level has children 2j and 2j + 1 below it.

    Face indexes a query returns are original indexes. `last_pairs` counts the exact
    (point, triangle) tests the latest query ran: the work a test pins to show the tree prunes.
    """

    def __init__(self, source) -> None:
        index = geom._as_index(source)
        if BVH_LEAF < 1:
            raise ValueError("BVH_LEAF must be at least 1")
        self._tris = index.tris
        self._count = len(index)
        self._leaf = BVH_LEAF
        self._sorted, self._order = index._sorted, index._order
        self._tlo, self._thi = index._tlo, index._thi
        self._lox, self._hix = _levels(self._tlo, self._thi, self._leaf)
        size = float((self._hix[-1][:, 0] - self._lox[-1][:, 0]).max())
        self._pad = _SLAB_PAD * (size if size > 0 else 1.0)
        self._rays: dict[int, _Ray] = {}
        self.last_pairs = 0

    def __len__(self) -> int:
        return self._count

    # --- walking the tree --------------------------------------------------------------------

    def _roots(self, m: int, level: int | None = None):
        """The first (point, node) pairs for m points: every point at every node of `level`
        (the root when `level` is not given)."""
        level = len(self._lox) - 1 if level is None else level
        top = self._lox[level].shape[1]
        return np.repeat(np.arange(m), top), np.tile(np.arange(top), m)

    def _start_level(self, m: int) -> int:
        """The level a `closest` walk of m points starts at: the lowest one that holds at most
        `START_PAIRS` (point, node) pairs. A few points spend more in numpy calls per level than in
        arithmetic, so they skip the top levels and test all their nodes at once; a level above is
        only a prune, so skipping it never changes the answer."""
        for level in range(len(self._lox)):
            if m * self._lox[level].shape[1] <= START_PAIRS:
                return level
        return len(self._lox) - 1

    def _children(self, level: int, pi, ni):
        """The (point, child) pairs below (point, node) pairs of `level`, in point order."""
        ni2 = np.empty(2 * len(ni), dtype=ni.dtype)
        ni2[0::2] = 2 * ni
        ni2[1::2] = 2 * ni + 1
        keep = ni2 < self._lox[level - 1].shape[1]
        return np.repeat(pi, 2)[keep], ni2[keep]

    def _expand(self, pi, li):
        """Every (point, triangle) pair of the (point, leaf) pairs: point rows, sorted rows."""
        rows = li[:, None] * self._leaf + np.arange(self._leaf)[None, :]
        keep = rows < self._count
        return np.broadcast_to(pi[:, None], rows.shape)[keep], rows[keep]

    def _limit2(self, ub2) -> np.ndarray:
        """The squared distance within which a box still counts: the upper bound, grown by a share
        of itself and by the index's float pad (a bound of 0 must not drop a tied face)."""
        return (np.sqrt(ub2) * (1 + _BOUND_SLACK) + self._pad) ** 2

    # --- nearest surface ---------------------------------------------------------------------

    def _test_leaves(self, chunk, base, pi, li, best, bounded=True) -> None:
        """Exact tests of chunk points `pi` against whole leaves `li`, merged into `best`.

        With `bounded`, a triangle whose own box is farther than the point's best distance so far
        is dropped before the exact test: its box distance never exceeds its true distance."""
        block = max(1, geom.PAIR_BLOCK // self._leaf)
        for s in range(0, len(pi), block):
            rp, rt = self._expand(pi[s : s + block], li[s : s + block])
            p = chunk[rp]
            if bounded:
                gap = np.maximum(np.maximum(self._tlo[rt] - p, p - self._thi[rt]), 0.0)
                near = geom._dot(gap, gap) <= self._limit2(best.d2[rp + base])
                rp, rt, p = rp[near], rt[near], p[near]
            self.last_pairs += len(rt)
            t = self._sorted[rt]
            q = geom._closest_on_triangles(p, t[:, 0], t[:, 1], t[:, 2])
            best.merge(rp + base, geom._dot(p - q, p - q), q, self._order[rt])

    def _nearest_chunk(self, chunk, base, best) -> None:
        """Nearest triangles for the points of `chunk`, which start at row `base` of the query.

        Down the tree, level by level: a (point, node) pair is dropped when the node's box is
        farther than a distance some triangle is proven to be within (`_far2`), and that bound
        only shrinks. The leaf with the least such bound is tested first, which gives the exact
        bound; the other leaves within it are tested after, each triangle behind its own box.
        The walk starts at `_start_level`, not always at the root.
        """
        m = len(chunk)
        axes = np.ascontiguousarray(chunk.T)
        ub2 = np.full(m, np.inf)
        top = self._start_level(m)
        pi, ni = self._roots(m, top)
        for level in range(top, -1, -1):
            lb2, mm2 = _bounds2(self._lox[level][:, ni], self._hix[level][:, ni], axes[:, pi])
            np.minimum.at(ub2, pi, mm2)
            keep = lb2 <= self._limit2(ub2)[pi]
            pi, ni, lb2, mm2 = pi[keep], ni[keep], lb2[keep], mm2[keep]
            if level:
                pi, ni = self._children(level, pi, ni)
        # The leaf with the least bound for each point goes first.
        least = np.full(m, np.inf)
        np.minimum.at(least, pi, mm2)
        chosen = np.flatnonzero(mm2 == least[pi])
        _, first = np.unique(pi[chosen], return_index=True)
        chosen = chosen[first]
        self._test_leaves(chunk, base, pi[chosen], ni[chosen], best, bounded=False)
        rest = np.ones(len(pi), dtype=bool)
        rest[chosen] = False
        rest &= lb2 <= self._limit2(best.d2[base : base + m])[pi]
        self._test_leaves(chunk, base, pi[rest], ni[rest], best)

    def closest(self, pts) -> Nearest:
        """Nearest point on the surface for each point; the same answer as `geom.closest_points`.

        Equal distances go to the lowest original face index. At a shared edge or vertex, rounding
        can make one of the faces nearer by a hair, and the exact arithmetic decides.
        """
        pts = geom._as_points(pts)
        best = _Best(len(pts))
        self.last_pairs = 0
        step = max(1, geom.PAIR_BUDGET // self._lox[0].shape[1])
        for start in range(0, len(pts), step):
            self._nearest_chunk(pts[start : start + step], start, best)
        return Nearest(
            dist=np.sqrt(best.d2),
            point=best.point,
            face=best.face,
            normal=_unit_normals(self._tris[best.face]),
        )

    # --- inside ------------------------------------------------------------------------------

    def _ray(self, k: int) -> _Ray:
        """The terms of ray k: Moller-Trumbore per triangle (the terms of `geom.ray_counts`, summed
        in another order), and the outline boxes.

        Seen along the ray, a hit is the origin falling inside the triangle's outline. So the
        outline's box, in two axes across the ray, is a cheap test, and the boxes join up the tree
        as the 3D boxes do. A hit is never lost to it: the boxes are grown by the float pad."""
        if k not in self._rays:
            d = RAYS[k]
            a = self._sorted[:, 0]
            e1, e2 = self._sorted[:, 1] - a, self._sorted[:, 2] - a
            pvec = np.cross(d, e2)
            det = np.einsum("nk,nk->n", e1, pvec)
            scale = np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) * np.linalg.norm(d)
            live = np.abs(det) > 1e-12 * scale
            inv = 1.0 / np.where(live, det, 1.0)
            mats = (pvec, np.cross(e1, d), np.cross(e1, e2))
            helper = np.eye(3)[int(np.argmin(np.abs(d)))]
            first = np.cross(d, helper)
            first /= np.linalg.norm(first)
            across = np.stack([first, np.cross(d, first)])
            seen = np.einsum("ntj,aj->nta", self._sorted, across)
            lo = np.minimum(np.minimum(seen[:, 0], seen[:, 1]), seen[:, 2]) - self._pad
            hi = np.maximum(np.maximum(seen[:, 0], seen[:, 1]), seen[:, 2]) + self._pad
            node_lo, node_hi = _levels(lo, hi, self._leaf)
            self._rays[k] = _Ray(live, inv, mats, [(a * m).sum(axis=1) for m in mats], across,
                                 np.ascontiguousarray(lo.T), np.ascontiguousarray(hi.T), node_lo,
                                 node_hi)
        return self._rays[k]

    def _odd(self, origins, k: int) -> np.ndarray:
        """Whether ray k from each origin hits the mesh an odd number of times."""
        ray = self._ray(k)
        odd = np.zeros(len(origins), dtype=bool)
        step = max(1, geom.PAIR_BUDGET // self._lox[0].shape[1])
        block = max(1, geom.PAIR_BLOCK // self._leaf)
        for start in range(0, len(origins), step):
            chunk = origins[start : start + step]
            seen = np.ascontiguousarray((chunk @ ray.across.T).T)  # (2, m)
            pi, ni = self._roots(len(chunk))
            for level in range(len(self._lox) - 1, -1, -1):
                lo, hi = ray.node_lo[level], ray.node_hi[level]
                keep = np.ones(len(pi), dtype=bool)
                for a in range(2):
                    at = seen[a][pi]
                    keep &= (at >= lo[a][ni]) & (at <= hi[a][ni])
                pi, ni = pi[keep], ni[keep]
                if level:
                    pi, ni = self._children(level, pi, ni)
            counts = np.zeros(len(chunk), dtype=int)
            for s in range(0, len(pi), block):
                rp, rt = self._expand(pi[s : s + block], ni[s : s + block])
                keep = np.ones(len(rp), dtype=bool)
                for a in range(2):
                    at = seen[a][rp]
                    keep &= (at >= ray.tri_lo[a][rt]) & (at <= ray.tri_hi[a][rt])
                rp, rt = rp[keep], rt[keep]
                self.last_pairs += len(rt)
                o = chunk[rp]
                u, v, t = (
                    (np.einsum("ij,ij->i", o, m[rt]) - off[rt]) * ray.inv[rt]
                    for m, off in zip(ray.mats, ray.offs, strict=True)
                )
                hit = (
                    ray.live[rt]
                    & (u >= -_RAY_EDGE_SLACK)
                    & (v >= -_RAY_EDGE_SLACK)
                    & (u + v <= 1 + _RAY_EDGE_SLACK)
                    & (t > _RAY_T_MIN)
                )
                counts += np.bincount(rp[hit], minlength=len(chunk))
            odd[start : start + len(chunk)] = counts % 2 == 1
        return odd

    def inside(self, pts) -> np.ndarray:
        """True where a point is inside the closed mesh: the same answer as `geom.inside`, an odd
        hit count on at least 2 of 3 rays.

        A point outside the mesh's box is outside. The third ray runs only for the points where the
        first two disagree: when they agree, the third cannot change the majority.
        """
        pts = geom._as_points(pts)
        out = np.zeros(len(pts), dtype=bool)
        self.last_pairs = 0
        root_lo, root_hi = self._lox[-1][:, 0], self._hix[-1][:, 0]
        within = ((pts >= root_lo - self._pad) & (pts <= root_hi + self._pad)).all(axis=1)
        rows = np.nonzero(within)[0]
        if len(rows) == 0:
            return out
        origins = pts[rows]
        votes = self._odd(origins, 0).astype(int) + self._odd(origins, 1)
        split = np.flatnonzero(votes == 1)
        if len(split):
            votes[split] += self._odd(origins[split], 2)
        out[rows] = votes >= 2
        return out

    # --- the deepest inside point per group --------------------------------------------------

    def _exit(self, pts) -> np.ndarray:
        """For each point, a distance that an inside point's depth cannot exceed.

        An inside point has an odd hit count on at least two of the three rays, and a hit lies in
        the mesh's box, so on those two rays the first surface is no farther than the ray's exit
        from the box. The depth is at most the middle of the three exit distances."""
        lo, hi = self._lox[-1][:, 0] - self._pad, self._hix[-1][:, 0] + self._pad
        exits = np.full((len(pts), len(RAYS)), np.inf)
        for k, d in enumerate(RAYS):
            for a in range(3):
                if d[a] != 0:
                    far = (hi[a] if d[a] > 0 else lo[a])
                    exits[:, k] = np.minimum(exits[:, k], (far - pts[:, a]) / d[a])
        return np.sort(exits, axis=1)[:, 1] + self._pad

    def _near_known(self, pts, known, depth) -> np.ndarray:
        """For each point, an upper bound on its distance to the surface from points whose
        distance is known: distance to the surface changes by at most the distance moved, so a point
        at `r` from one whose depth is `d` is no deeper than `d + r`. (Grown by the float pad.)"""
        out = np.full(len(pts), np.inf)
        if not len(known):
            return out
        step = max(1, geom.PAIR_BUDGET // (4 * len(known)))
        for start in range(0, len(pts), step):
            gap = pts[start : start + step, None, :] - known[None, :, :]
            reach = np.sqrt(np.einsum("pqk,pqk->pq", gap, gap)) + depth[None, :]
            out[start : start + step] = reach.min(axis=1)
        return out + self._pad

    def deepest(self, pts, groups, floor: float = 0.0) -> Deepest:
        """The deepest inside point of each group, for the groups that have one deeper than `floor`.

        `groups` (M,) gives each point an integer group. A point's depth is its distance to the
        surface (`closest`); only points `inside` count. The answer is the same as taking `inside`
        and `closest` for every point and a maximum per group, but it stops early. A point that
        cannot be deeper than the deepest found so far in its group is never tested. How deep a point
        can be comes from what is known: the mesh's box (`_exit`) and the points already measured
        (`_near_known`: the surface is 1-Lipschitz). Every point is considered; a point is skipped
        only when a bound proves it cannot win.

        It first measures a spread of points (every eighth of each group), then repeatedly the
        points with the highest bound, until none can win.
        """
        pts = geom._as_points(pts)
        groups = np.asarray(groups)
        if groups.shape != (len(pts),):
            raise ValueError(f"groups must have shape ({len(pts)},), got {groups.shape}")
        root_lo, root_hi = self._lox[-1][:, 0], self._hix[-1][:, 0]
        live = ((pts >= root_lo - self._pad) & (pts <= root_hi + self._pad)).all(axis=1)
        rows = np.flatnonzero(live)
        names, gid = np.unique(groups[rows], return_inverse=True)
        bound = self._exit(pts[rows])
        best = np.full(len(names), -np.inf)
        best_row = np.zeros(len(names), dtype=np.intp)
        todo = np.ones(len(rows), dtype=bool)
        fresh = np.zeros((0, 3))  # measured since the bounds last took them in
        fresh_depth = np.zeros(0)
        tests = 0

        def measure(chosen) -> None:
            """Test the points `chosen` (indexes into `rows`) and fold the inside ones in."""
            nonlocal fresh, fresh_depth, tests
            todo[chosen] = False
            pick = rows[chosen]
            held = self.inside(pts[pick])
            tests += self.last_pairs
            pick, chosen = pick[held], chosen[held]
            if not len(pick):
                return
            near = self.closest(pts[pick])
            tests += self.last_pairs
            fresh = np.concatenate([fresh, pts[pick]])
            fresh_depth = np.concatenate([fresh_depth, near.dist])
            # The deepest of each group in this batch, the lowest row on a tie.
            order = np.lexsort((pick, -near.dist, gid[chosen]))
            mine = gid[chosen][order]
            top = order[np.r_[True, mine[1:] != mine[:-1]]]
            g, d, r = gid[chosen][top], near.dist[top], pick[top]
            win = (d > best[g]) | ((d == best[g]) & (r < best_row[g]))
            best[g[win]], best_row[g[win]] = d[win], r[win]

        by_group = np.lexsort((np.arange(len(rows)), gid))
        start = np.searchsorted(gid[by_group], gid[by_group], side="left")
        measure(by_group[(np.arange(len(rows)) - start) % SEED_STRIDE == 0])
        take = 8
        while True:
            left = np.flatnonzero(todo)
            if not len(left):
                break
            bound[left] = np.minimum(bound[left],
                                     self._near_known(pts[rows[left]], fresh, fresh_depth))
            fresh, fresh_depth = fresh[:0], fresh_depth[:0]
            todo &= (bound > floor) & (bound >= best[gid])
            left = np.flatnonzero(todo)
            if not len(left):
                break
            # The `take` highest-bound points of each group still to try, ties by lowest row.
            order = left[np.lexsort((rows[left], -bound[left], gid[left]))]
            first = np.searchsorted(gid[order], gid[order], side="left")
            measure(order[np.arange(len(order)) - first < take])
            take *= 4
        self.last_pairs = tests
        won = best > floor
        return Deepest(names[won], best[won], best_row[won])
