"""Slices through a bending joint: where a plane cuts a part's skin, and the outline it leaves.

Think of slicing a salami along a bent sausage: the cut shows how thick the sausage is there. A
limb that bends keeps its thickness, so the slices of a bent limb should look like the slices of
the straight one. `core.checks.bend_quality` compares them.

A joint has two sides: the bones above it (the parent side) and the bones from it down (the child
side). Each vertex of a part carries a weight, the share of its motion that comes from the child
side: 0 on the parent side, 1 on the child side, between them across the blend band, where linear
blend skinning mixes the two motions. The slices are spread across that band.

- `band_offsets` finds the planes at rest: `n` offsets along the joint's bisector, between the
  parent-side vertex nearest the joint and the child-side vertex nearest it.
- `cut_slices` cuts a (posed) part at given offsets and measures each outline.
- `joint_slices` does both for one pose.

Plane k (counting from 1) sits at the fraction `k / (n + 1)` of the band. By default every plane is
parallel to the joint's bisecting plane. A tube bent at an angle `t` and cut that way is cut
obliquely, and an oblique cut of a straight tube is longer than a square one by `1 / cos`. So the
posed slices take `fan`: the direction of the limb before the joint, `a`, and after it, `b`. Plane
k then turns from `a` to `b` by the same fraction (the plane at the middle is the bisecting plane),
the way the limb itself turns across the band, and a bend that keeps the limb's section keeps the
slice.

A plane can cut a part more than once (a deep fold cuts both segments). The slice keeps the
outline whose middle is nearest the joint's own point on the plane.

Rings. A plane is the wrong cut for a sharp bend or an L-shaped joint: at a 135 degree knee the
planes cut the limb along its length, and at an ankle they cut the shin lengthwise. So the check
reads rings (`ring_slices`) instead: the line on the skin where the child-side weight is `k /
(n + 1)`. A ring is the same band of skin at rest and posed, whatever the pose does to the limb's
direction, and a pinch shrinks it. Its area is that of the loop's shadow on its own mean plane
(Newell's vector area), which is the area of a flat ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import Unmeasurable

# A vector shorter than this has no direction (float noise, not a measuring tolerance).
NO_DIRECTION = 1e-9


@dataclass(frozen=True)
class Slice:
    """One outline: its plane's `offset` from the joint along the bisector, the `area` it holds
    and its `perimeter`."""

    offset: float
    area: float
    perimeter: float


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    length = float(np.linalg.norm(v))
    if not length > NO_DIRECTION:
        raise Unmeasurable("a joint's two sides run opposite ways, so no plane bisects them")
    return v / length


def band_offsets(tris, weights, bone_head, bisector, band: tuple[float, float], n: int
                 ) -> np.ndarray:
    """The `n` offsets of the slice planes, along `bisector` from `bone_head`.

    `tris` is (T, 3, 3) and `weights` (T, 3) holds the child-side weight of every corner; a NaN
    corner takes no part (it belongs to a bone tree the joint is not in). `band` is (low, high): a
    corner at or under `low` is wholly parent side, one at or over `high` wholly child side. The
    planes are spread evenly between the last parent-side corner and the first child-side corner,
    k / (n + 1) of the way for k = 1..n. Raises `Unmeasurable` when a side has no corner or the
    child side lies behind the parent side."""
    tris = np.asarray(tris, dtype=float)
    weights = np.asarray(weights, dtype=float)
    along = (tris - np.asarray(bone_head, dtype=float)) @ _unit(bisector)
    with np.errstate(invalid="ignore"):
        parent = along[weights <= band[0]]
        child = along[weights >= band[1]]
    if not len(parent) or not len(child):
        raise Unmeasurable("the part has no vertex wholly on one side of the joint")
    near_parent, near_child = float(parent.max()), float(child.min())
    if not near_child > near_parent:
        raise Unmeasurable("the joint's child side does not lie ahead of its parent side along "
                           "the bisector")
    return near_parent + (near_child - near_parent) * np.arange(1, n + 1) / (n + 1)


def _edge_points(a: np.ndarray, b: np.ndarray, sa: np.ndarray, sb: np.ndarray) -> np.ndarray:
    """Where edges a-b cross the plane, given the signed distances. Each edge is taken from its
    lexicographically smaller end, so two triangles that share an edge get the same bits."""
    diff = a - b
    first = (diff != 0).argmax(axis=1)
    swap = diff[np.arange(len(a)), first] > 0
    lo = np.where(swap[:, None], b, a)
    hi = np.where(swap[:, None], a, b)
    s_lo = np.where(swap, sb, sa)
    s_hi = np.where(swap, sa, sb)
    point = lo + (s_lo / (s_lo - s_hi))[:, None] * (hi - lo)
    # an end on the plane is the crossing itself, to the bit, whichever triangle asks
    point = np.where((s_lo == 0)[:, None], lo, point)
    return np.where((s_hi == 0)[:, None], hi, point)


def _segments(tris: np.ndarray, point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """(M, 2, 3): the cut of every triangle the plane crosses. A corner on the plane counts as
    above it, so a plane through a vertex cuts consistently on both sides."""
    s = (tris - point) @ normal
    above = s >= 0
    crossing = np.nonzero(above.any(axis=1) & ~above.all(axis=1))[0]
    if not len(crossing):
        return np.empty((0, 2, 3))
    t, up, sd = tris[crossing], above[crossing], s[crossing]
    lone = np.where((up.sum(axis=1) == 1)[:, None], up, ~up).argmax(axis=1)
    rows = np.arange(len(crossing))
    p = t[rows, lone]
    sp = sd[rows, lone]
    out = np.empty((len(crossing), 2, 3))
    for k, step in enumerate((1, 2)):
        other = (lone + step) % 3
        out[:, k] = _edge_points(p, t[rows, other], sp, sd[rows, other])
    return out


def _loops(segments: np.ndarray) -> list[np.ndarray]:
    """The closed polylines the segments chain into, each an (K, 3) array. A chain that does not
    close is left out."""
    if not len(segments):
        return []
    pairs = [((tuple(a.tolist()), tuple(b.tolist())), (a, b)) for a, b in segments]
    # a triangle with one corner on the plane cuts to a point, not a segment
    pairs = [(keys, ab) for keys, ab in pairs if keys[0] != keys[1]]
    points: dict[tuple, np.ndarray] = {}
    ends: dict[tuple, list[tuple]] = {}
    for (ka, kb), (a, b) in pairs:
        points[ka], points[kb] = a, b
        ends.setdefault(ka, []).append(kb)
        ends.setdefault(kb, []).append(ka)
    left = {k: list(v) for k, v in ends.items()}
    loops = []
    for start in ends:
        while left[start]:
            chain = [start]
            here = start
            closed = False
            while left[here]:
                nxt = left[here].pop()
                left[nxt].remove(here)
                if nxt == start:
                    closed = True
                    break
                chain.append(nxt)
                here = nxt
            if closed and len(chain) >= 3:
                loops.append(np.array([points[k] for k in chain]))
    return loops


def _measure(loop: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    """(area, perimeter) of a closed polyline lying in a plane with the given normal."""
    ahead = np.roll(loop, -1, axis=0)
    area = 0.5 * abs(float(np.cross(loop, ahead).sum(axis=0) @ normal))
    return area, float(np.linalg.norm(ahead - loop, axis=1).sum())


def cut_slices(tris, bone_head, bisector, offsets, fan: tuple | None = None) -> list[Slice]:
    """The slice at each offset: the outline nearest the joint, measured.

    Plane k passes the point `bone_head + offset * normal` with the normal `bisector` or, with
    `fan = (a, b)`, the unit blend of `a` and `b` at k / (n + 1) of the way. Raises `Unmeasurable`
    when a plane gives no closed outline."""
    tris = np.asarray(tris, dtype=float)
    head = np.asarray(bone_head, dtype=float)
    bisector = _unit(bisector)
    offsets = np.asarray(offsets, dtype=float)
    out = []
    for k, offset in enumerate(offsets, start=1):
        if fan is None:
            normal = bisector
        else:
            share = k / (len(offsets) + 1)
            normal = _unit((1 - share) * _unit(fan[0]) + share * _unit(fan[1]))
        point = head + float(offset) * normal
        loops = _loops(_segments(tris, point, normal))
        if not loops:
            raise Unmeasurable(f"slice {k} at offset {float(offset):.4f} cuts no closed outline")
        nearest = min(loops, key=lambda lp: float(np.linalg.norm(lp.mean(axis=0) - point)))
        area, perimeter = _measure(nearest, normal)
        out.append(Slice(float(offset), area, perimeter))
    return out


def joint_slices(tris, weights, bone_head, bisector, band: tuple[float, float], n: int,
                 fan: tuple | None = None) -> list[Slice]:
    """`n` slices across the joint's blend band: `band_offsets`, then `cut_slices`."""
    offsets = band_offsets(tris, weights, bone_head, bisector, band, n)
    return cut_slices(tris, bone_head, bisector, offsets, fan)


# --- rings: the line of equal child-side weight --------------------------------------------------


def _level_segments(tris: np.ndarray, weights: np.ndarray, level: float) -> np.ndarray:
    """(M, 2, 3): where the weight crosses `level` on each triangle. A corner at the level counts
    as over it. A triangle with a NaN corner takes no part."""
    s = weights - level
    whole = ~np.isnan(weights).any(axis=1)
    above = s >= 0
    crossing = np.nonzero(whole & above.any(axis=1) & ~above.all(axis=1))[0]
    if not len(crossing):
        return np.empty((0, 2, 3))
    t, up, sd = tris[crossing], above[crossing], s[crossing]
    lone = np.where((up.sum(axis=1) == 1)[:, None], up, ~up).argmax(axis=1)
    rows = np.arange(len(crossing))
    out = np.empty((len(crossing), 2, 3))
    for k, step in enumerate((1, 2)):
        other = (lone + step) % 3
        out[:, k] = _edge_points(t[rows, lone], t[rows, other], sd[rows, lone], sd[rows, other])
    return out


def crossing_triangles(weights, level: float) -> np.ndarray:
    """(T,) whether the weight crosses `level` on each triangle (`weights` is (T, 3))."""
    weights = np.asarray(weights, dtype=float)
    above = weights - level >= 0
    return ~np.isnan(weights).any(axis=1) & above.any(axis=1) & ~above.all(axis=1)


def ring_loop(tris, weights, bone_head, level: float) -> np.ndarray:
    """The ring at `level`: the closed outline of equal child-side weight nearest the joint, an
    (K, 3) array. `weights` is (T, 3), as for `band_offsets`. Raises `Unmeasurable` when the weight
    never crosses the level in a closed line."""
    tris = np.asarray(tris, dtype=float)
    loops = _loops(_level_segments(tris, np.asarray(weights, dtype=float), level))
    if not loops:
        raise Unmeasurable(f"no closed outline where the weight is {level:.3f}")
    head = np.asarray(bone_head, dtype=float)
    return min(loops, key=lambda lp: float(np.linalg.norm(lp.mean(axis=0) - head)))


def _flat(loop: np.ndarray) -> tuple[float, float]:
    """(area, perimeter) of a closed polyline: the area is Newell's, that of its shadow on its
    own mean plane."""
    ahead = np.roll(loop, -1, axis=0)
    return (0.5 * float(np.linalg.norm(np.cross(loop, ahead).sum(axis=0))),
            float(np.linalg.norm(ahead - loop, axis=1).sum()))


def ring_slices(tris, weights, bone_head, n: int) -> list[Slice]:
    """The `n` rings at the weights k / (n + 1), k = 1..n, measured. A ring's `offset` is its
    weight level."""
    out = []
    for k in range(1, n + 1):
        level = k / (n + 1)
        area, perimeter = _flat(ring_loop(tris, weights, bone_head, level))
        out.append(Slice(level, area, perimeter))
    return out
