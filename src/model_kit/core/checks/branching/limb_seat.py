"""Branching check: limb-seat. Every limb sits on its parent, every leaf on its host.

Read from the skeleton, with the radius where it measures as the length:
- a limb with no parent that is not rooted floats (item `<pack>:<piece> <limb id>`);
- a limb's base is more than `SEAT_OF_PARENT_RADIUS` x the parent's radius from the parent's
  axis, or its base ring pokes out of the parent: its distance from the axis plus its base
  radius is over `SEAT_CHILD_OF_PARENT_RADIUS` x the parent's radius there (which covers a base
  radius over that share);
- a leaf's node is more than `SEAT_OF_PARENT_RADIUS` x the host's radius from the host's axis,
  or its blade's base is farther from its node than that much plus its petiole (item
  `<pack>:<piece> leaf.<n>`). The petiole counts only on a full skeleton, whose mesh draws the
  petiole ribbon (skeleton-match ties the two); a lite skeleton draws none, so its blade must
  start within the host's radius of its node.

Plants, one unit each, built from the skeleton's points alone (never `_limb_findings`,
`_leaf_findings` or `_allowance`): on every piece with a seated limb, one limb that has a
grandparent (else the first limb with a parent) moved off its parent's axis by `CALIBRATION_FAIL`
x the parent's radius, perpendicular to the parent's tangent and away from the axis; and on every
piece with a leaf, leaf 0's base and blade moved out along its petiole by `CALIBRATION_FAIL` x the
host's radius, plus any shortfall of its base distance below its allowance, so the plant clears the
line whatever the start.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

import numpy as np

from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown
from model_kit.core.grow.shape import PERP_EPS, SIDE, UP
from model_kit.core.grow.skeleton import unit
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    SEAT_CHILD_OF_PARENT_RADIUS,
    SEAT_OF_PARENT_RADIUS,
)

NAME = "limb-seat"


def _perp(v: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """v's part across the tangent, made unit; a side direction when v runs along it."""
    across = v - tangent * float(v @ tangent)
    if float(np.sqrt(across @ across)) > PERP_EPS:
        return unit(across)
    side = np.cross(tangent, SIDE)
    return unit(side if float(np.sqrt(side @ side)) > PERP_EPS else np.cross(tangent, UP))


def _limb_findings(g: Grown, only) -> list[Finding]:
    sk, found = g.skeleton, []
    for i in range(len(sk.limb_id)):
        if not g.wants(only, g.limb(i)):
            continue
        p = int(sk.parent[i])
        if p < 0:
            if not sk.rooted[i]:
                found.append(Finding(NAME, g.limb(i), "no parent and not rooted: it floats"))
            continue
        dist, r = sk.nearest(p, sk.limb_points(i)[:1])
        dist, r = float(dist[0]), float(r[0])
        parent = sk.limb_id[p]
        if dist > SEAT_OF_PARENT_RADIUS * r:
            found.append(Finding(NAME, g.limb(i), f"base {dist:.4f} from {parent}'s axis, over "
                                                  f"{SEAT_OF_PARENT_RADIUS * r:.4f} (its radius "
                                                  f"there {r:.4f})"))
        r0 = float(sk.limb_radii(i)[0])
        if dist + r0 > SEAT_CHILD_OF_PARENT_RADIUS * r:
            found.append(Finding(NAME, g.limb(i), f"base ring reaches {dist + r0:.4f} from "
                                                  f"{parent}'s axis (base radius {r0:.4f}), over "
                                                  f"{SEAT_CHILD_OF_PARENT_RADIUS * r:.4f}: it "
                                                  "pokes out of the bark"))
    return found


def _allowance(sk, f: int) -> float:
    """How far past the host's radius a leaf's blade may start from its node: its petiole on a
    full skeleton (the mesh draws it), nothing on a lite one."""
    return 0.0 if sk.lite else float(sk.petiole[f])


def _leaf_findings(g: Grown, only) -> list[Finding]:
    sk, found = g.skeleton, []
    for f in range(len(sk.host)):
        if not g.wants(only, g.leaf(f)):
            continue
        host = int(sk.host[f])
        dist, r = sk.nearest(host, sk.node[f:f + 1])
        dist, r = float(dist[0]), float(r[0])
        tol = SEAT_OF_PARENT_RADIUS * r
        if dist > tol:
            found.append(Finding(NAME, g.leaf(f), f"node {dist:.4f} from {sk.limb_id[host]}'s "
                                                  f"axis, over {tol:.4f}"))
        reach = float(np.linalg.norm(sk.base[f] - sk.node[f]))
        allowed = _allowance(sk, f)
        if reach > allowed + tol:
            what = "its petiole" if not sk.lite else "no petiole (lite)"
            found.append(Finding(NAME, g.leaf(f), f"blade base {reach:.4f} from its node, over "
                                                  f"{what} {allowed:.4f} plus {tol:.4f}"))
    return found


def _limb_units(ctx: SceneCtx, g: Grown) -> list[PlantUnit]:
    sk = g.skeleton
    seated = [int(i) for i in np.flatnonzero(sk.parent >= 0)]
    if not seated:
        return []  # no limb sits on a parent: this rule judges none
    i = next((i for i in seated if sk.parent[int(sk.parent[i])] >= 0), seated[0])
    p = int(sk.parent[i])
    base = sk.limb_points(i)[0]
    _, r = sk.nearest(p, base[None])
    point, _, tangent = sk.at(p, float(sk.t[i]))
    away = _perp(base - point, tangent)
    pts = np.array(sk.pts)
    pts[sk.start[i]:sk.start[i + 1]] += away * CALIBRATION_FAIL * float(r[0])
    plant = Plant(f"off axis: {g.limb(i)}", g.limb(i),
                  g.with_skeleton(ctx, dataclasses.replace(sk, pts=pts)))
    return [PlantUnit(g.limb(i), (plant,))]


def _leaf_units(ctx: SceneCtx, g: Grown) -> list[PlantUnit]:
    sk = g.skeleton
    if not len(sk.host):
        return []  # no leaf: this rule judges none
    out = sk.base[0] - sk.node[0]
    reach = float(np.sqrt(out @ out))
    direction = unit(out) if reach > 0.0 else unit(sk.mid[0])
    _, r = sk.nearest(int(sk.host[0]), sk.node[:1])
    # What the mesh draws past the node: a deliberate copy of `_allowance`, so a blinded
    # `_allowance` cannot move the plant (the guard refuses the original).
    petiole = 0.0 if sk.lite else float(sk.petiole[0])
    base = np.array(sk.base)
    base[0] += direction * (CALIBRATION_FAIL * float(r[0]) + max(0.0, petiole - reach))
    plant = Plant(f"blade far: {g.leaf(0)}", g.leaf(0),
                  g.with_skeleton(ctx, dataclasses.replace(sk, base=base)))
    return [PlantUnit(g.leaf(0), (plant,))]


class LimbSeat:
    name = NAME
    kinds: frozenset[str] = KINDS

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return ([u for g in grown(ctx) for u in _limb_units(ctx, g)]
                + [u for g in grown(ctx) for u in _leaf_units(ctx, g)])

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The tube and leaf parts of each grown piece: its limbs and its leaves are read."""
        return {part for g in grown(ctx) for part in g.limb_parts() | g.leaf_parts()}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for g in grown(ctx, only)
                for f in _limb_findings(g, only) + _leaf_findings(g, only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return coverage(ctx, NAME)
