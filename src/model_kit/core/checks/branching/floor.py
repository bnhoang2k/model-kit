"""Branching check: floor-organic. Rooted limbs stand in the soil, buried limbs grow from them
into it, and nothing else dips below it by more than a tube thickness.

The soil is y = 0 in the piece frame. `tol` is `FLOOR_OF_THICKNESS` x the piece's tube
thickness base. Read from the skeleton and the leaves' triangles (item `<pack>:<piece> <limb id>`
or `<pack>:<piece> leaf.<n>`):
- a rooted limb must have no parent; its base must not be above the soil, nor sunk more than
  `ROOTED_DEPTH_OF_RADIUS` x its own base radius below it (the `depth`); the ring at each of its
  points past the first `depth` of its arc length (the stub that may stand in the soil) must not
  dip below `-tol`. A ring of radius r perpendicular to a unit direction d reaches
  r x sqrt(1 - d_y^2) below its point, so an upright trunk point at y = 0 lies at 0, not at -r.
  Each point's ring is read on its adjacent segments' directions and, at a joint, on the
  central difference the mesh draws it on; the lowest is taken (`grow.skeleton.ring_lows`);
- a buried limb must grow from a rooted limb, seated no higher than `BURIED_SEAT_OF_RADIUS` x
  that limb's radius there, and its tip must be in the soil;
- any other limb, past its seat part (`grow.segments`: the part buried in its parent's wood),
  less its radius, must not dip below `-tol`;
- no vertex of a leaf's triangles may dip below `-tol`.

Plants, one unit each on every grown piece, built from the skeleton's points and the leaves'
triangles alone (never `_limb_findings`, `_low_past` or `_leaf_low`): the whole skeleton lifted
until the deepest rooted base sits `CALIBRATION_FAIL` x the tolerance above the soil; the whole
skeleton sunk until that base sits at `-CALIBRATION_FAIL` x its depth limit (both on that rooted
limb); and the leaf whose node is lowest sunk so its lowest vertex sits at `-CALIBRATION_FAIL` x
the tolerance (with no leaves, the free limb whose lowest point is lowest sunk the same way, then
sunk until its highest point sits there).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

import numpy as np

from model_kit.core.checks.branching._pairs import Zones
from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown, tube_base
from model_kit.core.grow.mesh import leaf_tris
from model_kit.core.grow.shape import SEGMENT_EPS
from model_kit.core.grow.skeleton import Skeleton, arc_lengths, ring_lows
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    BURIED_SEAT_OF_RADIUS,
    CALIBRATION_FAIL,
    FLOOR_OF_THICKNESS,
    ROOTED_DEPTH_OF_RADIUS,
    SEAT_ZONE_OPENS_OF_RADIUS,
)

NAME = "floor-organic"


def _free(sk: Skeleton) -> np.ndarray:
    return np.flatnonzero(~sk.rooted & ~sk.buried)


def _leaf_reach(sk: Skeleton) -> float:
    """How far any leaf vertex lies from the leaf's base, in blade lengths: the outline folded
    up, the blade centre, and the petiole's end."""
    x, y = sk.outline[:, 0], sk.outline[:, 1]
    rim = np.sqrt(x * x + y * y + (sk.fold * x) ** 2)
    return float(max(rim.max(), np.linalg.norm(sk.blade_centre), sk.petiole_tip))


def _leaf_bounds(sk: Skeleton) -> np.ndarray:
    """(F,) a floor under each leaf's vertices, from its node, base and size alone."""
    low = np.minimum(sk.node[:, 1], sk.base[:, 1])
    return low - sk.length * _leaf_reach(sk) - sk.petiole_width


def _leaf_low(sk: Skeleton, f: int) -> float:
    return float(leaf_tris(sk, f)[..., 1].min())


def _low_past(pts: np.ndarray, radii: np.ndarray, start: float) -> float:
    """The lowest y less radius along a limb past arc length `start` (the point at `start`
    included)."""
    cum = arc_lengths(pts)
    if start >= cum[-1]:
        return np.inf
    k = int(np.searchsorted(cum, start, side="right"))  # the first point past start
    f = (start - cum[k - 1]) / max(cum[k] - cum[k - 1], SEGMENT_EPS)
    at = pts[k - 1, 1] + (pts[k, 1] - pts[k - 1, 1]) * f
    r = radii[k - 1] + (radii[k] - radii[k - 1]) * f
    return float(min(at - r, (pts[k:, 1] - radii[k:]).min()))


def _limb_findings(g: Grown, i: int, tol: float, zones: Zones) -> list[Finding]:
    sk, item, found = g.skeleton, g.limb(i), []
    pts, radii, p = sk.limb_points(i), sk.limb_radii(i), int(sk.parent[i])
    if sk.rooted[i]:
        if p >= 0:
            found.append(Finding(NAME, item, f"rooted, but it grows from {sk.limb_id[p]}"))
        depth = ROOTED_DEPTH_OF_RADIUS * float(radii[0])
        if pts[0, 1] > 0.0:
            found.append(Finding(NAME, item, f"rooted, but its base is {pts[0, 1]:.4f} above "
                                             "the soil"))
        elif pts[0, 1] < -depth:
            found.append(Finding(NAME, item, f"rooted base sunk to {pts[0, 1]:.4f}, below "
                                             f"{-depth:.4f}"))
        past = arc_lengths(pts) > depth  # the stub that may stand in the soil is not read
        low = float(ring_lows(pts, radii)[past].min()) if past.any() else np.inf
        if low < -tol:
            found.append(Finding(NAME, item, f"past its base it dips to {low:.4f}, below "
                                             f"{-tol:.4f}"))
        return found
    if sk.buried[i]:
        if p < 0 or not sk.rooted[p]:
            found.append(Finding(NAME, item, "buried, but it does not grow from a rooted limb"))
        else:
            _, r = sk.nearest(p, pts[:1])
            high = BURIED_SEAT_OF_RADIUS * float(r[0])
            if pts[0, 1] > high:
                found.append(Finding(NAME, item, f"buried, but seated {pts[0, 1]:.4f} above the "
                                                 f"soil, over {high:.4f}"))
        if pts[-1, 1] > 0.0:
            found.append(Finding(NAME, item, f"buried, but its tip is {pts[-1, 1]:.4f} above "
                                             "the soil"))
        return found
    start = zones.seat(i) if p >= 0 else 0.0
    low = _low_past(pts, radii, start)
    if low < -tol:
        found.append(Finding(NAME, item, f"dips to {low:.4f}, below {-tol:.4f}"))
    return found


def _findings(ctx: SceneCtx, g: Grown, only) -> list[Finding]:
    sk = g.skeleton
    tol = FLOOR_OF_THICKNESS * tube_base(ctx, g.piece)
    zones = Zones(sk, SEAT_ZONE_OPENS_OF_RADIUS)
    found = [f for i in range(len(sk.limb_id)) if g.wants(only, g.limb(i))
             for f in _limb_findings(g, i, tol, zones)]
    candidates = np.flatnonzero(_leaf_bounds(sk) < -tol) if len(sk.host) else []
    for f in candidates:
        item = g.leaf(int(f))
        if g.wants(only, item) and (low := _leaf_low(sk, int(f))) < -tol:
            found.append(Finding(NAME, item, f"a leaf vertex dips to {low:.4f}, below "
                                             f"{-tol:.4f}"))
    return found


def _lift(sk: Skeleton, by: np.ndarray) -> Skeleton:
    return dataclasses.replace(sk, pts=sk.pts + by, node=sk.node + by, base=sk.base + by)


def _rooted_units(ctx: SceneCtx, g: Grown, tol: float) -> list[PlantUnit]:
    """The lift and the sink of the whole skeleton, each on the deepest rooted limb."""
    sk = g.skeleton
    rooted = np.flatnonzero(sk.rooted)
    if not len(rooted):
        return []
    bases = sk.pts[sk.start[rooted], 1]
    deepest = int(rooted[np.argmin(bases)])
    item = g.limb(deepest)
    limit = ROOTED_DEPTH_OF_RADIUS * float(sk.radii[sk.start[deepest]])
    lifted = _lift(sk, np.array([0.0, -float(bases.min()) + CALIBRATION_FAIL * tol, 0.0]))
    sunk = _lift(sk, np.array([0.0, -float(bases.min()) - CALIBRATION_FAIL * limit, 0.0]))
    return [PlantUnit(item, (Plant(f"lift: {item}", item, g.with_skeleton(ctx, lifted)),)),
            PlantUnit(item, (Plant(f"sunk: {item}", item, g.with_skeleton(ctx, sunk)),))]


def _leaf_unit(ctx: SceneCtx, g: Grown, tol: float) -> PlantUnit:
    """The leaf whose node is lowest, sunk until its lowest triangle corner sits at
    `-CALIBRATION_FAIL` x tol."""
    sk = g.skeleton
    f = int(np.argmin(sk.node[:, 1]))
    drop = np.array([0.0, float(leaf_tris(sk, f)[..., 1].min()) + CALIBRATION_FAIL * tol, 0.0])
    node, base = np.array(sk.node), np.array(sk.base)
    node[f] -= drop
    base[f] -= drop
    item = g.leaf(f)
    plant = Plant(f"sink leaf: {item}", item,
                  g.with_skeleton(ctx, dataclasses.replace(sk, node=node, base=base)))
    return PlantUnit(item, (plant,))


def _limb_unit(ctx: SceneCtx, g: Grown, tol: float, free: np.ndarray) -> PlantUnit:
    """The free limb whose lowest point (less its radius) is lowest, sunk until that point sits
    at `-CALIBRATION_FAIL` x tol; the second candidate sinks it until its highest point (plus its
    radius) sits there, so no part of it is left above the soil."""
    sk = g.skeleton
    lows = [float((sk.limb_points(i)[:, 1] - sk.limb_radii(i)).min()) for i in free]
    i = int(free[int(np.argmin(lows))])
    highest = float((sk.limb_points(i)[:, 1] + sk.limb_radii(i)).max())
    item = g.limb(i)
    plants = []
    for lowering in (min(lows) + CALIBRATION_FAIL * tol, highest + CALIBRATION_FAIL * tol):
        pts = np.array(sk.pts)
        pts[sk.start[i]:sk.start[i + 1], 1] -= lowering
        plants.append(Plant(f"sink limb: {item}", item,
                            g.with_skeleton(ctx, dataclasses.replace(sk, pts=pts))))
    return PlantUnit(item, tuple(plants))


def _units(ctx: SceneCtx, g: Grown) -> list[PlantUnit]:
    tol = FLOOR_OF_THICKNESS * tube_base(ctx, g.piece)
    units = _rooted_units(ctx, g, tol)
    if len(g.skeleton.host):
        units.append(_leaf_unit(ctx, g, tol))
    elif len(free := _free(g.skeleton)):
        units.append(_limb_unit(ctx, g, tol, free))
    return units


class FloorOrganic:
    name = NAME
    kinds: frozenset[str] = KINDS

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [u for g in grown(ctx) for u in _units(ctx, g)]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The tube and leaf parts of each grown piece: its limbs and its leaves are read."""
        return {part for g in grown(ctx) for part in g.limb_parts() | g.leaf_parts()}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for g in grown(ctx, only) for f in _findings(ctx, g, only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return coverage(ctx, NAME)
