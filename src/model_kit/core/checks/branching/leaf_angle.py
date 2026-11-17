"""A tree-file check: it binds any skeleton; on the kit's own grower it catches only code and data
mistakes.

Branching check: leaf-angle. Blade angles lie where measured broadleaf crowns put them, and spread
in every part of the crown.

The blade angle is the angle of the blade from horizontal (0 flat, 90 vertical): the angle between
the blade's face normal and the vertical, the de Wit convention of the leaf-inclination dataset
(Chianucci et al. 2018, Mendeley Data 10.17632/4rmc7r8zvy.2; Pisek et al. 2013). The skeleton's
`normal` is that face normal. Two judgements, both from that dataset:
- `<pack>:<piece> mean`: the mean blade angle of the whole tree lies between
  `LEAF_ANGLE_MEAN_MIN_DEG` and `LEAF_ANGLE_MEAN_MAX_DEG`, the range of the species means (each end
  widened by twice its species' standard error). A species file whose blade-angle beta has a mean
  of 85 degrees fails here.
- `<pack>:<piece> third.<0|1|2>`: the standard deviation (n - 1, as the dataset's R script takes
  it) of the blade angles in each third of the crown is at least `LEAF_ANGLE_SD_MIN_DEG`. The thirds
  are the bottom, middle and top equal-height thirds of the span of the leaves' nodes, as the
  dataset's lower, middle and upper canopy sets. A flat plate of identical blades fails every third.

A mean or a third with fewer than `LEAF_ANGLE_MIN_LEAVES` leaves is not judged (a coverage line
says so): the dataset's sets hold at least that many, so neither the range nor the floor was
measured on fewer. A skeleton with no leaves (a measured tree, a winter tree) has a not-applicable
coverage line and no finding.

The check sees only the angle from horizontal, not which way the blade faces: a crown of blades at
varied angles that all face one way passes.

Plants, one unit each on every piece with a judged mean or third, built from the blades' count and
heights alone (never `_findings`, `blade_angles` or `spreads`): every blade set to the middle of the
measured range of species means ("plates", one unit for each judged third, item `third.<k>`);
every blade set to `LEAF_ANGLE_PLANT_STEEP_DEG` ("steep", item `mean`); every blade laid flat,
0 degrees ("flat", item `mean`). A plant turns each blade about its own base, keeping the way it
faces, and leaves the leaf's other numbers alone.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown
from model_kit.core.grow.skeleton import Skeleton
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    LEAF_ANGLE_MEAN_MAX_DEG,
    LEAF_ANGLE_MEAN_MIN_DEG,
    LEAF_ANGLE_MIN_LEAVES,
    LEAF_ANGLE_PLANT_STEEP_DEG,
    LEAF_ANGLE_SD_MIN_DEG,
)

NAME = "leaf-angle"
MEAN = "mean"
THIRDS = ("bottom", "middle", "top")  # item `third.<index>`
UP = np.array([0.0, 1.0, 0.0])
SIDE = np.array([1.0, 0.0, 0.0])  # the way a vertical blade is turned from: it has no outward side


# --- the measures ------------------------------------------------------------------------------


def _unit_rows(v: np.ndarray) -> np.ndarray:
    length = np.sqrt((v * v).sum(axis=1))
    return v / np.where(length > 0.0, length, 1.0)[:, None]


def blade_angles(sk: Skeleton) -> np.ndarray:
    """(F,) each blade's angle from horizontal, in degrees: 0 flat, 90 vertical."""
    if not len(sk.host):
        return np.zeros(0)
    return np.degrees(np.arccos(np.clip(np.abs(_unit_rows(sk.normal)[:, 1]), 0.0, 1.0)))


def third_of(sk: Skeleton) -> np.ndarray:
    """(F,) 0, 1 or 2: the equal-height third of the leaves' node heights each leaf is in."""
    if not len(sk.host):
        return np.zeros(0, dtype=int)
    y = sk.node[:, 1]
    low, span = float(y.min()), float(y.max() - y.min())
    if not span > 0.0:
        return np.zeros(len(y), dtype=int)
    return np.minimum(((y - low) / span * len(THIRDS)).astype(int), len(THIRDS) - 1)


@dataclass(frozen=True)
class Spread:
    """The mean and the standard deviation of a set of blade angles, and how many there are."""

    n: int
    mean: float
    sd: float


def spread_of(angles: np.ndarray) -> Spread:
    n = len(angles)
    return Spread(n, float(angles.mean()) if n else 0.0,
                  float(angles.std(ddof=1)) if n > 1 else 0.0)


def spreads(sk: Skeleton) -> tuple[Spread, list[Spread]]:
    """The whole tree's blade angles, and each third's."""
    angles, which = blade_angles(sk), third_of(sk)
    return spread_of(angles), [spread_of(angles[which == k]) for k in range(len(THIRDS))]


def enough(s: Spread) -> bool:
    return s.n >= LEAF_ANGLE_MIN_LEAVES


# --- the findings ------------------------------------------------------------------------------


def _findings(g: Grown, only) -> list[Finding]:
    whole, thirds = spreads(g.skeleton)
    found = []
    item = g.item(MEAN)
    if enough(whole) and g.wants(only, item) and not (
            LEAF_ANGLE_MEAN_MIN_DEG <= whole.mean <= LEAF_ANGLE_MEAN_MAX_DEG):
        found.append(Finding(NAME, item, f"mean blade angle {whole.mean:.2f} deg over {whole.n} "
                                         f"leaves, outside the measured range "
                                         f"{LEAF_ANGLE_MEAN_MIN_DEG} to {LEAF_ANGLE_MEAN_MAX_DEG} "
                                         "deg of species means"))
    for k, s in enumerate(thirds):
        item = g.item(f"third.{k}")
        if enough(s) and g.wants(only, item) and s.sd < LEAF_ANGLE_SD_MIN_DEG:
            found.append(Finding(NAME, item, f"blade angles of the {THIRDS[k]} third spread "
                                             f"{s.sd:.2f} deg (SD over {s.n} leaves), under the "
                                             f"smallest measured {LEAF_ANGLE_SD_MIN_DEG} deg"))
    return found


# --- the plants --------------------------------------------------------------------------------


def turned(sk: Skeleton, degrees: float | np.ndarray) -> Skeleton:
    """The skeleton with every blade turned about its own base to `degrees` from horizontal,
    keeping the way it faces (the horizontal part of its normal, and which side is up)."""
    n = _unit_rows(sk.normal)
    side = np.where(n[:, 1] >= 0.0, 1.0, -1.0)
    flat = n * np.array([1.0, 0.0, 1.0])
    out = np.where((flat * flat).sum(axis=1)[:, None] > 0.0, _unit_rows(flat), SIDE)
    theta = np.radians(degrees) * np.ones(len(n))
    target = out * np.sin(theta)[:, None] + UP * (side * np.cos(theta))[:, None]
    axis = np.cross(n, target)
    sin = np.sqrt((axis * axis).sum(axis=1))
    cos = (n * target).sum(axis=1)
    k = axis / np.where(sin > 0.0, sin, 1.0)[:, None]

    def rotated(v: np.ndarray) -> np.ndarray:  # Rodrigues' rotation of v about k
        return (v * cos[:, None] + np.cross(k, v) * sin[:, None]
                + k * ((k * v).sum(axis=1) * (1.0 - cos))[:, None])

    return dataclasses.replace(sk, mid=rotated(sk.mid), across=rotated(sk.across), normal=target)


def _judged_counts(sk: Skeleton) -> tuple[int, list[int]]:
    """How many leaves the whole tree and each third hold: what decides whether they are judged."""
    return len(sk.host), np.bincount(third_of(sk), minlength=len(THIRDS)).tolist()


def _units(g: Grown, ctx: SceneCtx) -> list[PlantUnit]:
    sk = g.skeleton
    whole, thirds = _judged_counts(sk)
    units = []
    if any(n >= LEAF_ANGLE_MIN_LEAVES for n in thirds):
        # Every blade at one angle, inside the range of species means, so the mean stands apart.
        plates = g.with_skeleton(ctx, turned(sk, 0.5 * (LEAF_ANGLE_MEAN_MIN_DEG
                                                       + LEAF_ANGLE_MEAN_MAX_DEG)))
        for k, n in enumerate(thirds):
            if n >= LEAF_ANGLE_MIN_LEAVES:
                item = g.item(f"third.{k}")
                units.append(PlantUnit(item, (Plant(f"plates: {item}", item, plates),)))
    if whole >= LEAF_ANGLE_MIN_LEAVES:
        for label, degrees in (("steep", LEAF_ANGLE_PLANT_STEEP_DEG), ("flat", 0.0)):
            item = g.item(MEAN)
            units.append(PlantUnit(item, (Plant(f"{label}: {item}", item,
                                                g.with_skeleton(ctx, turned(sk, degrees))),)))
    return units


class LeafAngle:
    name = NAME
    kinds: frozenset[str] = KINDS
    scope = "piece"

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [u for g in grown(ctx) for u in _units(g, ctx)]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The leaf parts of each grown piece: its blades are read."""
        return {part for g in grown(ctx) for part in g.leaf_parts()}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for g in grown(ctx, only) for f in _findings(g, only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = coverage(ctx, NAME)
        for g in grown(ctx):
            whole, thirds = spreads(g.skeleton)
            if not whole.n:
                lines.append(f"{NAME}: not applicable (no leaves): {g.key}")
                continue
            for item, s in [(g.item(MEAN), whole)] + [
                    (g.item(f"third.{k}"), t) for k, t in enumerate(thirds)]:
                if not enough(s):
                    lines.append(f"{NAME}: not judged ({s.n} leaves, fewer than "
                                 f"{LEAF_ANGLE_MIN_LEAVES}): {item}")
        return lines
