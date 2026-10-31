"""A tree-file check: it binds any skeleton; on the kit's own grower it catches only code and data
mistakes.

Branching check: taper. A limb's radius does not rise toward its tip.

The pipe model: a stem's cross-section follows the leaf area it carries, so radius falls from a
limb's base to its tip (Shinozaki et al. 1964, reviewed by Lehnebach et al. 2018, Ann. Bot. 121:
773-795). Per limb, the check fits a least-squares line to the radius over the arc length, both in
real metres (model units over the scene's `units_per_metre`). It fails a limb whose slope is over
0 (item `<pack>:<piece> <limb id>`). A single step that grows is not judged: measured trees grow
by more than 10% in 8 to 9% of their cylinder steps, which is noise.

Which limbs are judged depends on where the skeleton came from. A measured skeleton (its `info`
names a `source_doi`) has lidar radii that are not reliable on thin wood, so only points whose
radius is at least `RELIABLE_RADIUS_M` (Demol et al. 2022) are used, and a limb is judged only when
its base radius is at least that and it has `TAPER_MIN_POINTS` such points. A grown skeleton has no
lidar error: every limb with `TAPER_MIN_POINTS` points is judged. The arc length is taken along the
whole limb, then read at the points used. The slope is fitted on the radius less the limb's first
used radius, so a limb of equal radii has a slope of exactly 0 and never fails on rounding.

Plant, one unit on every piece that has a judged limb, built from the limbs' points and radii
alone (never `fit` or `_findings`): the judged limb with the most points is given a radius that grows
from its base radius by `TAPER_PLANT_RISE` (30%) toward its tip, in step with arc length. A piece
with no judged limb has none: its coverage line says how many limbs are not judged.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown
from model_kit.core.grow.skeleton import Skeleton, arc_lengths
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import RELIABLE_RADIUS_M, TAPER_MIN_POINTS, TAPER_PLANT_RISE

NAME = "taper"


@dataclass(frozen=True)
class Fit:
    """A judged limb's radius trend, in metres: the slope (radius per arc length), the number of
    points used, and the radius at the first and the last point used."""

    slope: float
    points: int
    first: float
    last: float


def is_measured(sk: Skeleton) -> bool:
    """A skeleton read from a scan names the source it was converted from."""
    return "source_doi" in sk.info


def used(sk: Skeleton, i: int, units_per_metre: float) -> tuple[np.ndarray, np.ndarray] | None:
    """The (arc length, radius) points of limb i the trend is read on, in metres, or None when the
    limb is not judged (see the module docstring): which limbs are judged, and no slope."""
    arc = arc_lengths(sk.limb_points(i)) / units_per_metre
    radius = sk.limb_radii(i) / units_per_metre
    if is_measured(sk):
        if radius[0] < RELIABLE_RADIUS_M:
            return None
        keep = radius >= RELIABLE_RADIUS_M
        arc, radius = arc[keep], radius[keep]
    if len(arc) < TAPER_MIN_POINTS:
        return None
    along = arc - arc.mean()
    if not float(along @ along) > 0.0:
        return None
    return arc, radius


def fit(sk: Skeleton, i: int, units_per_metre: float) -> Fit | None:
    """Limb i's radius slope, or None when the limb is not judged (see the module docstring)."""
    points = used(sk, i, units_per_metre)
    if points is None:
        return None
    arc, radius = points
    along = arc - arc.mean()
    slope = float(along @ (radius - radius[0])) / float(along @ along)
    return Fit(slope, len(arc), float(radius[0]), float(radius[-1]))


def fits(sk: Skeleton, units_per_metre: float) -> dict[int, Fit]:
    """The judged limbs of a skeleton, by limb index."""
    return {i: f for i in range(len(sk.limb_id)) if (f := fit(sk, i, units_per_metre)) is not None}


def _findings(g: Grown, ctx: SceneCtx, only) -> list[Finding]:
    found = []
    for i, f in fits(g.skeleton, ctx.style.units_per_metre).items():
        item = g.limb(i)
        if f.slope > 0.0 and g.wants(only, item):
            found.append(Finding(NAME, item, f"radius rises {f.slope:.4f} m per metre of limb "
                                             f"({f.points} points, {f.first:.4f} m to "
                                             f"{f.last:.4f} m); the pipe model says it falls "
                                             "toward the tip"))
    return found


def _units(g: Grown, ctx: SceneCtx) -> list[PlantUnit]:
    """The judged limb with the most points, its radius made to rise. No unit when no limb is
    judged: the check judges no limb of the piece, and coverage says so."""
    sk = g.skeleton
    judged = {i: len(points[0]) for i in range(len(sk.limb_id))
              if (points := used(sk, i, ctx.style.units_per_metre)) is not None}
    if not judged:
        return []
    i = max(judged, key=lambda k: (judged[k], -k))
    arc = arc_lengths(sk.limb_points(i))
    radii = np.array(sk.radii)
    radii[sk.start[i]:sk.start[i + 1]] = sk.limb_radii(i)[0] * (
        1.0 + TAPER_PLANT_RISE * arc / arc[-1])
    plant = Plant(f"thickening: {g.limb(i)}", g.limb(i),
                  g.with_skeleton(ctx, dataclasses.replace(sk, radii=radii)))
    return [PlantUnit(g.limb(i), (plant,))]


class Taper:
    name = NAME
    kinds: frozenset[str] = KINDS

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [u for g in grown(ctx) for u in _units(g, ctx)]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The tube parts of each grown piece: its limbs are read."""
        return {part for g in grown(ctx) for part in g.limb_parts()}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for g in grown(ctx, only) for f in _findings(g, ctx, only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = coverage(ctx, NAME)
        for g in grown(ctx):
            sk = g.skeleton
            left = len(sk.limb_id) - len(fits(sk, ctx.style.units_per_metre))
            if left:
                why = (f"base radius under {RELIABLE_RADIUS_M} m, or fewer than "
                       f"{TAPER_MIN_POINTS} points of at least that radius" if is_measured(sk)
                       else f"fewer than {TAPER_MIN_POINTS} points")
                lines.append(f"{NAME}: not judged ({left} of {len(sk.limb_id)} limbs: {why}): "
                             f"{g.key}")
        return lines
