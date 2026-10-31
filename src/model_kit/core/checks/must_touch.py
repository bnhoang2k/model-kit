"""Check must-touch: a hand, foot or held prop really touches the hold it is said to hold.

For each clip hold and each due sample, the `who` points map into the hold piece's frame.
The gap is the nearest distance from those points to the hold's surface (its parts, cut to
its `sub_box`). A press counts as 0: a point inside a closed surface part, near the surface,
and over the surface's footprint seen along the hold's `off`. A gap over the hold's budget
is a finding.

Units. The budget, `behind_max` and the effector radius are world lengths (model units). The
measure runs in the piece frame, which a carried prop's pose may scale (it takes the
armature's scale). So at each sample the world lengths convert to piece units, and the gap
back to world units, through `targets.length_in_world` along the hold's `off`: the one
conversion trace uses too. It is exact under a uniform scale.

Two plant units per hold, at its first due sample: lift the `who` points off along the hold's
`off`, and slide them off the surface's edge. Both clear the surface by twice the budget
whatever the start, so a deep press or a large dangle budget cannot hide them. They share the
hold's item but test two things (the nearest-distance measure, and the footprint test of a
press), so each is its own unit, one candidate each; the gate skips both when the hold already
fails. They are built from the hold's geometry and samples, never from the measure.

A hold whose clip has no samples, or no sample inside its `when` windows, is listed in
coverage and makes the run partial (`metagate.Unmeasured`): it could not be measured.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.geom import (
    NO_DIRECTION,
    SurfaceIndex,
    clip_to_box,
    drop_slivers,
    footprint_distance,
    is_closed,
    transform,
)
from model_kit.core.manifest import Clip, ClipHold, anchor
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import (
    ClipSamples,
    SceneCtx,
    check_windows,
    due,
    effector_radius,
    piece_geom,
)
from model_kit.core.targets import length_in_world, piece_frames
from model_kit.core.tolerances import (
    BEHIND_MAX_OF_EFFECTOR,
    PLANT_EXTRA_OF_BUDGET,
    TOUCH_OF_EFFECTOR,
)

NAME = "must-touch"


# --- the measure ----------------------------------------------------------------------------


def _surface_index(surface) -> SurfaceIndex:
    """The index of a hold surface, without its zero-area triangles (they add no surface)."""
    tris = drop_slivers(np.asarray(surface, dtype=float))
    if len(tris) == 0:
        raise ManifestError("the hold surface has no triangle with area")
    return SurfaceIndex(tris)


def hold_gap(used_parts: dict[str, np.ndarray | SurfaceIndex], surface_tris, pts,
             behind_max: float, *, off, budget: float) -> tuple[float, float]:
    """(gap, depth) from points to a hold surface, all in the piece frame.

    A point presses when all three hold: it is inside one of the closed `used_parts`; its
    nearest distance to the surface is within `behind_max`; and, seen along the hold's `off`,
    it lies within `budget` of the surface's footprint (`geom.footprint_distance`). Any press
    makes the gap 0, and `depth` is the deepest pressed point's distance behind the surface
    along `off`. Else the gap is the smallest nearest distance and `depth` is 0. Zero-area
    surface triangles add no surface, so they are left out of the measure.

    The surface and the parts may be `SurfaceIndex` objects built once by the caller (the
    surface one without zero-area triangles); arrays get a one-off index. The nearest distance
    runs first, on every point. The inside and footprint tests then run only on the points
    within `behind_max` of the surface, since only those can press.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 3)
    surface = surface_tris if isinstance(surface_tris, SurfaceIndex) else _surface_index(
        surface_tris)
    n = np.asarray(off, dtype=float)
    n = n / np.linalg.norm(n)
    near = surface.closest(pts)
    close = np.nonzero(near.dist <= behind_max)[0]
    pressed = np.zeros(len(pts), dtype=bool)
    if len(close):
        in_part = np.zeros(len(close), dtype=bool)
        for part in used_parts.values():
            index = part if isinstance(part, SurfaceIndex) else SurfaceIndex(part)
            in_part |= index.inside(pts[close])
        cand = close[in_part]
        if len(cand):
            pressed[cand] = footprint_distance(surface.tris, n, pts[cand]) <= budget
    if pressed.any():
        below = np.maximum((near.point - pts) @ n, 0.0)
        return 0.0, float(below[pressed].max())
    return float(near.dist.min()), 0.0


# --- one hold, resolved ---------------------------------------------------------------------


@dataclass
class _Job:
    """One clip hold. `_resolve` fills the fields every hold has; `_bind` adds its samples."""

    clip: Clip
    hold: ClipHold
    item: str
    placed_by: str
    used: dict[str, SurfaceIndex]  # the whole closed surface parts, piece frame
    surface: np.ndarray  # surface triangles cut to the sub_box, piece frame
    surface_index: SurfaceIndex  # the same, without zero-area triangles
    point: np.ndarray  # the hold point, piece frame
    off: np.ndarray  # unit off direction, piece frame
    budget: float
    radius: float
    behind_max: float
    prop_points: np.ndarray | None  # a held prop's counted points, in the prop's frame
    samples: ClipSamples | None = None
    due: np.ndarray | None = None  # sample indices
    frames: np.ndarray | None = None  # (S, 4, 4) world-from-piece matrix at each sample

    def who_world(self, s: int, samples: ClipSamples | None = None) -> np.ndarray:
        samples = samples or self.samples
        if self.prop_points is None:
            return samples.groups[self.hold.who][s]
        return transform(self.prop_points, samples.props[self.hold.who][s])

    def who_piece(self, s: int, samples: ClipSamples | None = None):
        return transform(self.who_world(s, samples), np.linalg.inv(self.frames[s]))

    def unit(self, s: int) -> float:
        """World length of one piece unit along `off` at sample `s` (1 with no scale)."""
        return length_in_world(self.frames[s, :3, :3], self.off)


def _item(clip: Clip, index: int, hold: ClipHold) -> str:
    return f"{clip.name} holds[{index}]: {hold.who} on {hold.ref}"


def _resolve(ctx: SceneCtx, clip: Clip, index: int, hold: ClipHold) -> _Job:
    """Everything about a hold that needs no samples. A bad hold raises here."""
    item = _item(clip, index, hold)
    check_windows(clip, hold.when, item)
    found = anchor(ctx.packs, hold.ref, clip=clip)
    if found.kind != "hold":
        raise ManifestError(f"{item}: {hold.ref} names a {found.kind}, not a hold")
    spot = found.item
    geom = piece_geom(ctx, hold.ref.pack, found.piece.name)
    used = {}
    for part in spot.surface:
        if part not in geom.parts:
            raise ManifestError(f"{item}: the scene has no geometry for surface part {part!r}")
        tris = np.asarray(geom.parts[part], dtype=float)
        if not is_closed(tris):
            raise ManifestError(f"{item}: surface part {part!r} is not closed, so the press "
                                "test cannot tell inside from outside")
        used[part] = SurfaceIndex(tris)
    surface = np.concatenate([index.tris for index in used.values()])
    if spot.sub_box is not None:
        surface = clip_to_box(surface, spot.sub_box)
        if len(surface) == 0:
            raise ManifestError(f"{item}: the hold's sub_box holds none of its surface")

    prop = clip.props.get(hold.who)
    prop_points = None
    if prop is None:
        if hold.who_box is not None:
            raise ManifestError(f"{item}: who_box applies to a held prop, and {hold.who!r} "
                                "is a contact group")
    else:
        pgeom = piece_geom(ctx, prop.pack, prop.piece)
        prop_points = np.unique(
            np.concatenate([np.asarray(t, dtype=float).reshape(-1, 3)
                            for t in pgeom.parts.values()]), axis=0)
        if hold.who_box is not None:
            lo, hi = np.asarray(hold.who_box[0]), np.asarray(hold.who_box[1])
            prop_points = prop_points[((prop_points >= lo) & (prop_points <= hi)).all(axis=1)]
        if len(prop_points) == 0:
            raise ManifestError(f"{item}: the who_box holds none of prop {hold.who!r}'s points")

    radius = effector_radius(ctx, clip.name, hold.who)
    if hold.dangle is not None:
        budget = hold.dangle.budget
    elif hold.budget is not None:
        budget = hold.budget
    else:
        budget = TOUCH_OF_EFFECTOR * radius
    off = np.asarray(spot.off, dtype=float)
    return _Job(
        clip=clip, hold=hold, item=item, placed_by=found.placed_by, used=used,
        surface=surface, surface_index=_surface_index(surface),
        point=np.asarray(spot.point, dtype=float),
        off=off / np.linalg.norm(off), budget=budget, radius=radius,
        behind_max=BEHIND_MAX_OF_EFFECTOR * radius, prop_points=prop_points,
    )


def _bind(job: _Job, samples: ClipSamples) -> _Job:
    """The hold with its clip's samples: due sample indices and the piece's frames."""
    who = job.hold.who
    if job.prop_points is None and who not in samples.groups:
        raise ManifestError(f"{job.item}: the samples hold no contact group {who!r}")
    if job.prop_points is not None and who not in samples.props:
        raise ManifestError(f"{job.item}: the samples hold no pose for prop {who!r}")
    frames = piece_frames(job.clip, samples, job.placed_by, job.item)
    return dataclasses.replace(job, samples=samples, due=due(samples.times, job.hold.when),
                               frames=frames)


def _jobs(ctx: SceneCtx, only: Collection[str] | None = None) -> Iterator[_Job | str]:
    """Each hold as a job, or a coverage line naming a hold that cannot be measured.

    A clip with no holds, solved or keyed, has nothing to touch: one not-applicable line names it.

    Every hold resolves first, sampled or not, so a bad hold stops the run either way. With
    `only`, just the holds whose items it names are resolved (a plant's scoped run).
    """
    for clip in ctx.clips.values():
        if not clip.holds:
            yield f"{NAME}: not applicable (no holds): {clip.name}"
            continue
        resolved = [_resolve(ctx, clip, i, hold) for i, hold in enumerate(clip.holds)
                    if only is None or _item(clip, i, hold) in only]
        samples = ctx.samples.get(clip.name)
        for job in resolved:
            if samples is None:
                yield Unmeasured(f"{NAME}: skipped {job.item}: the scene has no samples for "
                                 "the clip")
                continue
            job = _bind(job, samples)
            if len(job.due) == 0:
                yield Unmeasured(f"{NAME}: skipped {job.item}: no sample inside its when "
                                 "windows")
                continue
            yield job


# --- plants ---------------------------------------------------------------------------------


def _longest_in_plane(surface_pts: np.ndarray, n: np.ndarray) -> np.ndarray:
    flat = surface_pts - np.outer(surface_pts @ n, n)
    flat = flat - flat.mean(axis=0)
    _, vecs = np.linalg.eigh(flat.T @ flat)
    for k in range(2, -1, -1):  # largest spread first; skip the plane normal itself
        d = vecs[:, k] - (vecs[:, k] @ n) * n
        if np.linalg.norm(d) >= NO_DIRECTION:
            return d / np.linalg.norm(d)
    raise ManifestError("the hold surface has no in-plane extent")


def _moved(ctx: SceneCtx, job: _Job, s: int, delta_piece: np.ndarray) -> SceneCtx:
    """A copy of the scene with the hold's `who` moved by `delta_piece` at sample `s`."""
    delta = job.frames[s, :3, :3] @ delta_piece
    samples = job.samples
    who = job.hold.who
    if who in job.clip.props:
        poses = np.array(samples.props[who], dtype=float)
        poses[s, :3, 3] += delta
        samples = dataclasses.replace(samples, props={**samples.props, who: poses})
    else:
        pts = np.array(samples.groups[who], dtype=float)
        pts[s] += delta
        samples = dataclasses.replace(samples, groups={**samples.groups, who: pts})
    return dataclasses.replace(ctx, samples={**ctx.samples, job.clip.name: samples})


def _plants(ctx: SceneCtx, job: _Job) -> list[PlantUnit]:
    s = int(job.due[0])
    w = job.who_piece(s)
    surface_pts = job.surface.reshape(-1, 3)
    n, unit = job.off, job.unit(s)
    extra = PLANT_EXTRA_OF_BUDGET * job.budget / unit  # piece units
    lift = n * ((surface_pts @ n).max() - (w @ n).min() + extra)

    d = w.mean(axis=0) - job.point
    d = d - (d @ n) * n
    if np.linalg.norm(d) < NO_DIRECTION:
        d = _longest_in_plane(surface_pts, n)
    else:
        d = d / np.linalg.norm(d)
    slide = d * ((surface_pts @ d).max() - (w @ d).min() + job.radius / unit + extra)
    return [PlantUnit(job.item, (Plant(f"{how}: {job.item}", job.item, _moved(ctx, job, s, by),
                                        samples=frozenset({s})),))
            for how, by in (("lift", lift), ("slide", slide))]


# --- the check ------------------------------------------------------------------------------


class MustTouch:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [p for job in _jobs(ctx) if isinstance(job, _Job) for p in _plants(ctx, job)]

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        found = []
        for job in _jobs(ctx, only):
            if not isinstance(job, _Job):
                continue
            worst = None
            for s in job.due:
                if samples and int(s) not in samples:
                    continue
                unit = job.unit(int(s))
                gap, _ = hold_gap(job.used, job.surface_index, job.who_piece(int(s)),
                                  job.behind_max / unit, off=job.off, budget=job.budget / unit)
                gap *= unit  # world units, like the budget
                if gap > job.budget and (worst is None or gap > worst[0]):
                    worst = (gap, float(job.samples.times[s]))
            if worst is not None:
                found.append(Finding(NAME, job.item, f"gap {worst[0]:.4f} over budget "
                                                     f"{job.budget:.4f} at frame {worst[1]:g}"))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return [line for line in _jobs(ctx) if isinstance(line, str)]
