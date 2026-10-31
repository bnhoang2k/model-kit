"""Check trace: every clip target still points at its hold, bone or seat.

A solved clip bakes each key target's world point (`baked`) and its seated root. After a
model edit those baked numbers can go stale. This check resolves each ref again from the
current scene and compares. Three failures, each measured against `tol`, the contact length
(`sight.contact_length`: one arcminute at `view.nearest_m`) of the piece the item belongs to (a
`rig:` target uses the rig piece, a hold, seat, mount or use target its own piece, the seat rule
the seat's piece). A piece with no `view` has no contact length: its items are `Unmeasured`.

- stale: the target's `ref` plus `offset` is more than `tol` from `baked`.
- off-surface: the point the hold's `point` plus the in-plane part of the target's `offset`
  reaches lies outside the hold's surface. A raised offset (along the hold's `off`) is not in
  plane and passes.
- seat root: the root derived from the clip's seat is more than `tol` from the sampled root
  bone at some sample inside the clip's seat windows. This runs on every sampled clip that
  names a seat, solved or keyed: the seat is a fact of the clip, and the rule reads no baked
  numbers.

Where an offset lives. A target's offset is in the frame of what its ref names, and
`model_kit.core.targets` resolves it (the one place a ref and offset become a world point).
- A hold, seat, mount or use ref: the piece frame, the same frame as the hold's `off` and its
  surface triangles. The world point is the item's `point` carried through the piece's
  placement, or, for a carried prop, through the prop's pose at the key's sample, plus the
  offset turned by that frame.
- A `rig:<bone>` ref: the bone's frame, at the bone head, from the sampled bone matrix at the
  key's sample.
Scale in a sampled bone or prop matrix is dropped from the offset, so an offset keeps its
length in the world. A carried prop's pose may scale its geometry, so the off-surface test
converts the offset to the piece's own units (`Resolved.offset_in_piece`) and the tolerance and
plants are sized in the world, then converted to piece units. The key's
sample is the one whose time is nearest the key. `sample_at` raises when the sample times are
not finite and strictly increasing, or when the key lies outside the sampled range by more than
half the gap to the end sample. Stale compares the world point with `baked`. The off-surface test works wholly in
the piece frame.

The off-surface test is containment, worked in the piece frame. Let `n` be the hold's unit
`off`, `p` its `point`, `o` the target's offset, `f = o - (o . n) n` its in-plane part, `L = |f|`
and `d = f / L`. Project the hold's surface triangles (its parts, cut to the `sub_box`) onto the
plane perpendicular to `n`, and lay the plane out with `p` at the origin, `d` as the x axis.
The offset's in-plane point is `(L, 0)`. It is inside when some projected triangle meets the
line `y = 0` in an interval `[t0, t1]` with `t0 - tol <= L <= t1 + tol`. The `tol` is the
trace tolerance, slack for float noise at an edge. The edge distance `E` is the largest `t1`:
how far the surface reaches from `p` along `d`. An offset with no in-plane part has nothing to
test, so a hold point near an edge with a raised or zero in-plane offset passes.

The seat root. For a clip with `seat`, the seat top is the seat's `point` carried to the world
by its placement, and `facing` is the seat's facing turned by the same placement, flattened
to the ground plane and made unit length. The expected root is
`seat_top - below_seat x Y + forward x facing`, with `Y` the glTF up axis and `below_seat`
and `forward` from the rig's `sit`, in project units: `targets.seat_root`, the one seat rule,
which `blender.build.seated_root` also uses. It is compared with the head of the rig's root
bone (`rig.root`, else the one bone with no parent) at every sample inside the clip's seat windows
(`manifest.seat_windows`: the clip's `seat_when`, or the whole clip when it has none). The
seat is resolved again at each sample's own time, so a seat on a carried prop follows the prop.
One finding reports the largest gap and the frame it falls on (the earliest on a tie). A clip
that sits down part way through names the frames it sits in; before them the root is free.
A bad seat window, or `seat_when` without a `seat`, cannot reach the check: `Clip.validate`
refuses such a clip when it is built.

Skips are listed in coverage as `trace: skipped (<reason>): <clip, target or seat>`: a keyed
clip's targets, a clip with no samples, a keyed seated clip's seat when it has no samples, a
seated clip whose seat windows hold no sample (`skipped (no sample in seat_when)`), and a
target with no `baked` point (nothing to go stale; it still gets the surface test). The skips
that could not be measured (no samples, no sample in the windows) make the run partial
(`metagate.Unmeasured`).

Plants, each a copy of the scene that must read as a finding on its item. They are units built
from the manifest, the samples and the surface alone, never from the gap, the reach test or the
seat root the check measures. A target's stale plant and its off-surface plant are two units of
the same item. A stale unit tries the move one way and then the other, so a point that already
sits off its baked number cannot cancel it. The gate skips a unit whose item already fails.
- stale, hold target: one hold point moved along its `off` so the move is
  `CALIBRATION_FAIL x tol` long in the world.
- stale, `rig:` target: the sampled bone moved by `CALIBRATION_FAIL x tol` at the key's sample.
- stale, seat, mount or use target: its own point moved by `CALIBRATION_FAIL x tol` (world).
- off-surface: one hold target's in-plane offset set to
  `max(OFF_SURFACE_PLANT_OF_EDGE x E, E + CALIBRATION_FAIL x tol)` along its own direction
  (the target is re-baked, so only the surface test can see it). The second term keeps the
  plant outside the `tol` edge slack when the hold point sits within a tolerance of the rim.
- seat: the seat point moved by `CALIBRATION_FAIL x tol`, seen from the first sample inside
  the seat windows.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.entries import unit
from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION, clip_to_box
from model_kit.core.manifest import (
    Anchor,
    Clip,
    Hold,
    Pack,
    Piece,
    Seat,
    Target,
    anchor,
    seat_windows,
)
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.rig import rig_root
from model_kit.core.scene import (
    ClipSamples,
    SceneCtx,
    due,
    piece_geom,
    rig_spec,
)
from model_kit.core.sight import contact_length
from model_kit.core.targets import UP, Resolved, resolve_ref, resolve_target, seat_root
from model_kit.core.tolerances import CALIBRATION_FAIL, OFF_SURFACE_PLANT_OF_EDGE

NAME = "trace"
# Significant digits a length prints with: a decal-thin tolerance such as 0.000075 prints whole.
_PRINT_DIGITS = 6


def _length(x: float) -> str:
    """A length for a finding line, in positional notation with `_PRINT_DIGITS` significant digits."""
    return np.format_float_positional(x, precision=_PRINT_DIGITS, unique=True, fractional=False,
                                      trim="-")


def _tolerance(ctx: SceneCtx, pack: str, piece: Piece) -> float | None:
    """The contact length of `piece` of `pack`, in the pack's units; None when the piece declares
    no `view`."""
    if pack not in ctx.packs:
        raise ManifestError(f"trace: no pack {pack!r} for a tolerance")
    return contact_length(piece, ctx.packs[pack].units_per_metre)


def _piece_of(ctx: SceneCtx, clip: Clip, target: Target) -> tuple[str, Piece]:
    """The pack and piece a target is traced with: its ref's piece, or for a `rig:` ref the clip's
    rigged piece."""
    if not target.ref.is_bone:
        return target.ref.pack, anchor(ctx.packs, target.ref, clip=clip).piece
    if clip.name not in ctx.rig_of_clip:
        raise ManifestError(f"clip {clip.name!r}: the scene names no rig for it (rig_of_clip)")
    rig_spec(ctx, clip.name)  # raises the named ManifestError for a pack or piece the scene lacks
    pack, _, name = ctx.rig_of_clip[clip.name].partition(":")
    return pack, ctx.packs[pack].pieces[name]


# --- resolving a target ---------------------------------------------------------------------


def _item(clip: Clip, index: int, target: Target) -> str:
    return f"{clip.name} targets[{index}]: key {target.key} on {target.ref}"


@dataclass
class _Job:
    clip: Clip
    index: int
    target: Target
    item: str
    res: Resolved  # the ref at the key's sample, before the offset
    found: Anchor | None  # None for a rig: ref
    tol: float  # the contact length of the target's piece

    @property
    def point(self) -> np.ndarray:
        """World point of the ref plus the offset."""
        return self.res.with_offset(self.target.offset)

    @property
    def hold(self) -> Hold | None:
        if self.found is not None and self.found.kind == "hold":
            return self.found.item
        return None


def _seat_item(clip: Clip) -> str:
    return f"{clip.name} seat: {clip.seat}"


def _jobs(ctx: SceneCtx, only: Collection[str] | None = None) -> Iterator[_Job | str]:
    """Each target of a solved clip as a job, or a coverage line for what is skipped.

    With `only`, just the targets whose items it names (a plant's scoped run).
    """
    for clip in ctx.clips.values():
        if clip.authoring == "keyed":
            yield f"{NAME}: not applicable (keyed targets): {clip.name}"
            continue
        if clip.name not in ctx.samples:
            yield Unmeasured(f"{NAME}: skipped (no samples): {clip.name}")
            continue
        for index, target in enumerate(clip.targets):
            item = _item(clip, index, target)
            if only is not None and item not in only:
                continue
            tol = _tolerance(ctx, *_piece_of(ctx, clip, target))
            if tol is None:
                yield Unmeasured(f"{NAME}: skipped (the piece declares no view, so it has no "
                                 f"contact length): {item}")
                continue
            found = None if target.ref.is_bone else anchor(ctx.packs, target.ref, clip=clip)
            job = _Job(clip, index, target, item, resolve_target(ctx, clip.name, target),
                       found, tol)
            if target.baked is None:
                yield f"{NAME}: skipped (no baked point): {item}"
            yield job


# --- the three measures ---------------------------------------------------------------------


def _surface(ctx: SceneCtx, job: _Job) -> np.ndarray:
    """The hold's surface triangles in the piece frame, cut to its `sub_box`."""
    spot = job.hold
    geom = piece_geom(ctx, job.target.ref.pack, job.found.piece.name)
    parts = []
    for part in spot.surface:
        if part not in geom.parts:
            raise ManifestError(f"{job.item}: the scene has no geometry for surface part "
                                f"{part!r}")
        parts.append(np.asarray(geom.parts[part], dtype=float))
    surface = np.concatenate(parts)
    if spot.sub_box is not None:
        surface = clip_to_box(surface, spot.sub_box)
        if len(surface) == 0:
            raise ManifestError(f"{job.item}: the hold's sub_box holds none of its surface")
    return surface


def _split(offset: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The offset's part along `n` and its in-plane part."""
    along = (offset @ n) * n
    return along, offset - along


def _line_reach(surface: np.ndarray, point: np.ndarray, n: np.ndarray, d: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Where the line from `point` along in-plane `d` meets the surface, seen along `n`.

    Returns `(t0, t1)`, each `(M,)`: one interval per triangle the line meets, `t` the
    distance along `d`. A triangle meets the line at its vertices on the line and at the
    edges that cross it; the interval runs between the extremes.
    """
    rel = surface - point
    x, y = rel @ d, rel @ np.cross(n, d)  # (N, 3) each: the projected triangles
    nxt = np.roll(np.arange(3), -1)
    x2, y2 = x[:, nxt], y[:, nxt]
    with np.errstate(divide="ignore", invalid="ignore"):
        crossing = np.where(y * y2 < 0.0, x + (x2 - x) * y / (y - y2), np.nan)
    on_line = np.where(y == 0.0, x, np.nan)
    cuts = np.concatenate([crossing, on_line], axis=1)  # (N, 6), nan where none
    met = ~np.isnan(cuts).all(axis=1)
    cuts = cuts[met]
    return np.nanmin(cuts, axis=1), np.nanmax(cuts, axis=1)


def _unit_off(job: _Job) -> np.ndarray:
    off = np.asarray(job.hold.off, dtype=float)
    return off / np.linalg.norm(off)


def _stale_gap(job: _Job) -> float:
    return float(np.linalg.norm(job.point - np.asarray(job.target.baked, dtype=float)))


def _edge(t1: np.ndarray) -> float:
    """How far the surface reaches along the line: the largest interval end, 0 if it misses."""
    return float(t1.max()) if t1.size else 0.0


def _flat_piece(job: _Job, n: np.ndarray) -> np.ndarray:
    """The in-plane part of the target's offset, in the piece's own units.

    The offset is in model units. A scaled carried prop's pose stretches its geometry, so the
    offset is converted to piece units before it meets the triangles.
    """
    _, flat = _split(job.res.offset_in_piece(job.target.offset), n)
    return flat


def _off_surface(ctx: SceneCtx, job: _Job) -> tuple[float, float] | None:
    """(in-plane length, edge distance) of a hold target, or None when it lands on the surface.

    Both lengths are in piece units. The edge distance is how far the surface reaches along
    the offset's direction; it is 0 when the line misses the surface. The edge slack is the
    trace tolerance, converted to piece units along that direction.
    """
    n = _unit_off(job)
    flat = _flat_piece(job, n)
    length = float(np.linalg.norm(flat))
    if length < NO_DIRECTION:
        return None
    d = flat / length
    slack = job.tol / job.res.piece_length_in_world(d)
    t0, t1 = _line_reach(_surface(ctx, job), np.asarray(job.hold.point, dtype=float), n, d)
    if np.any((t0 - slack <= length) & (length <= t1 + slack)):
        return None
    return length, _edge(t1)


def _seat_due(clip: Clip, samples: ClipSamples) -> np.ndarray:
    """Indices of the samples inside the clip's seat windows (already checked)."""
    return due(samples.times, seat_windows(clip))


def _seat_root(ctx: SceneCtx, clip: Clip, samples: ClipSamples
               ) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """(sample, expected root, sampled root) at each sample inside the seat windows. See the
    module docstring. The seat is resolved at each sample's own time, so a seat on a carried
    prop follows the prop."""
    item = _seat_item(clip)
    found = anchor(ctx.packs, clip.seat, clip=clip)
    spec = rig_spec(ctx, clip.name)
    if spec.sit is None:
        raise ManifestError(f"{item}: clip {clip.name!r} names a seat, but its rig has no sit")
    try:
        root = rig_root(spec)
    except ManifestError as err:
        raise ManifestError(f"{item}: {err}") from err
    if root not in samples.bones:
        raise ManifestError(f"{item}: the samples hold no root bone {root!r}")
    seat: Seat = found.item
    facing = np.asarray(seat.facing, dtype=float)
    rows = []
    for s in _seat_due(clip, samples):
        res = resolve_ref(ctx, clip.name, clip.seat, float(samples.times[s]), what=item)
        expected = seat_root(res.point, res.frame @ facing, spec.sit, item)
        rows.append((int(s), expected,
                     np.asarray(samples.bones[root][s], dtype=float)[:3, 3]))
    return rows


def _seat_tolerance(ctx: SceneCtx, clip: Clip) -> float | None:
    """The contact length of the seat's piece, None when it declares no view."""
    return _tolerance(ctx, clip.seat.pack, anchor(ctx.packs, clip.seat, clip=clip).piece)


def _seated(ctx: SceneCtx, only: Collection[str] | None = None
            ) -> Iterator[tuple[Clip, ClipSamples]]:
    """Each sampled clip that names a seat, solved or keyed, with a sample inside its seat
    windows and whose seat's piece has a contact length; with `only`, just those whose seat items
    it names. The seat is a fact of the clip, not of a target."""
    for clip in ctx.clips.values():
        if clip.seat is not None and clip.name in ctx.samples \
                and (only is None or _seat_item(clip) in only) \
                and len(_seat_due(clip, ctx.samples[clip.name])) \
                and _seat_tolerance(ctx, clip) is not None:
            yield clip, ctx.samples[clip.name]


def _unmeasured_seats(ctx: SceneCtx) -> Iterator[str]:
    """A coverage line for each clip that names a seat and cannot be measured: a keyed clip
    with no samples (a solved one already has its own line), or a sampled clip with no sample
    inside its seat windows."""
    for clip in ctx.clips.values():
        if clip.seat is None:
            continue
        if clip.name not in ctx.samples:
            if clip.authoring == "keyed":
                yield Unmeasured(f"{NAME}: skipped (no samples): {_seat_item(clip)}")
        elif len(_seat_due(clip, ctx.samples[clip.name])) == 0:
            yield Unmeasured(f"{NAME}: skipped (no sample in seat_when): {_seat_item(clip)}")
        if _seat_tolerance(ctx, clip) is None:
            yield Unmeasured(f"{NAME}: skipped (the seat's piece declares no view, so it has no "
                             f"contact length): {_seat_item(clip)}")


# --- plants ---------------------------------------------------------------------------------


def _with_pack(ctx: SceneCtx, pack: str, piece: str, edit: Callable) -> SceneCtx:
    old: Pack = ctx.packs[pack]
    pieces = {**old.pieces, piece: edit(old.pieces[piece])}
    return dataclasses.replace(
        ctx, packs={**ctx.packs, pack: dataclasses.replace(old, pieces=pieces)})


def _move_point(ctx: SceneCtx, pack: str, piece: str, kind: str, item_id: str,
                delta: np.ndarray) -> SceneCtx:
    """A copy of the scene with one hold or seat point moved by `delta` (piece frame)."""
    def edit(p):
        items = tuple(dataclasses.replace(x, point=tuple(np.add(x.point, delta)))
                      if x.id == item_id else x for x in getattr(p, kind))
        return dataclasses.replace(p, **{kind: items})
    return _with_pack(ctx, pack, piece, edit)


def _move_bone(ctx: SceneCtx, job: _Job, delta: np.ndarray) -> SceneCtx:
    """A copy of the scene with the target's bone moved by `delta` (world) at the key's sample."""
    samples = ctx.samples[job.clip.name]
    bone = job.target.ref.id
    poses = np.array(samples.bones[bone], dtype=float)
    poses[job.res.sample, :3, 3] += delta
    moved = dataclasses.replace(samples, bones={**samples.bones, bone: poses})
    return dataclasses.replace(ctx, samples={**ctx.samples, job.clip.name: moved})


def _with_target(ctx: SceneCtx, job: _Job, target: Target) -> SceneCtx:
    targets = tuple(target if i == job.index else t for i, t in enumerate(job.clip.targets))
    clip = dataclasses.replace(job.clip, targets=targets)
    return dataclasses.replace(ctx, clips={**ctx.clips, clip.name: clip})


def _perpendicular(n: np.ndarray) -> np.ndarray:
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    d = np.cross(n, axis)
    return d / np.linalg.norm(d)


def _off_surface_plant(ctx: SceneCtx, job: _Job) -> SceneCtx:
    n = _unit_off(job)
    along, flat = _split(job.res.offset_in_piece(job.target.offset), n)
    d = flat / np.linalg.norm(flat) if np.linalg.norm(flat) >= NO_DIRECTION else _perpendicular(n)
    _, t1 = _line_reach(_surface(ctx, job), np.asarray(job.hold.point, dtype=float), n, d)
    edge = _edge(t1)
    margin = CALIBRATION_FAIL * job.tol / job.res.piece_length_in_world(d)  # piece units
    planted_piece = along + d * max(OFF_SURFACE_PLANT_OF_EDGE * edge, edge + margin)
    # Back to model units: the displacement in the world, in the ref's own frame.
    planted = job.res.frame.T @ (job.res.full @ planted_piece)
    target = dataclasses.replace(job.target, offset=tuple(float(v) for v in planted))
    # Re-bake: the solver would aim at the planted point, so only the surface test can fire.
    baked = job.res.with_offset(planted)
    target = dataclasses.replace(target, baked=None if job.target.baked is None
                                 else tuple(float(v) for v in baked))
    return _with_target(ctx, job, target)


def _piece_axis_move(job: _Job, axis: np.ndarray) -> np.ndarray:
    """A piece-frame move along `axis` that is `CALIBRATION_FAIL x tol` long in the world."""
    axis = axis / np.linalg.norm(axis)
    return axis * CALIBRATION_FAIL * job.tol / job.res.piece_length_in_world(axis)


def _stale_plants(ctx: SceneCtx, job: _Job) -> list[tuple[str, SceneCtx]]:
    """The target's own number moved one way and then the other (see the docstring)."""
    out = []
    for sign in (1.0, -1.0):
        if job.hold is not None:
            moved = _move_point(ctx, job.target.ref.pack, job.found.piece.name, "holds",
                                job.hold.id, sign * _piece_axis_move(job, _unit_off(job)))
        elif job.target.ref.is_bone:
            moved = _move_bone(ctx, job, sign * np.array([CALIBRATION_FAIL * job.tol, 0.0, 0.0]))
        else:  # a seat, mount or use: move its own point
            moved = _move_point(ctx, job.target.ref.pack, job.found.piece.name,
                                f"{job.found.kind}s", job.found.item.id,
                                sign * _piece_axis_move(job, UP))
        out.append(("stale", moved))
    return out


def _job_units(ctx: SceneCtx, job: _Job) -> list[PlantUnit]:
    units = []
    if job.target.baked is not None:
        units.append(unit(job.item, _stale_plants(ctx, job)))
    if job.hold is not None:
        units.append(unit(job.item, [("off-surface", _off_surface_plant(ctx, job))]))
    return units


def _seat_unit(ctx: SceneCtx, clip: Clip) -> PlantUnit:
    found = anchor(ctx.packs, clip.seat, clip=clip)
    samples = ctx.samples[clip.name]
    first = _seat_due(clip, samples)[0]
    res = resolve_ref(ctx, clip.name, clip.seat, float(samples.times[first]))
    tol = _seat_tolerance(ctx, clip)
    delta = UP * CALIBRATION_FAIL * tol / res.piece_length_in_world(UP)
    return unit(_seat_item(clip), [
        ("seat", _move_point(ctx, clip.seat.pack, found.piece.name, "seats", found.item.id,
                             sign * delta)) for sign in (1.0, -1.0)])


# --- the check ------------------------------------------------------------------------------


class Trace:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units = [u for job in _jobs(ctx) if isinstance(job, _Job) for u in _job_units(ctx, job)]
        units += [_seat_unit(ctx, clip) for clip, _ in _seated(ctx)]
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for job in _jobs(ctx, only):
            if not isinstance(job, _Job):
                continue
            tol = job.tol
            if job.target.baked is not None and (gap := _stale_gap(job)) > tol:
                found.append(Finding(NAME, job.item, f"stale: the ref plus offset resolves "
                                                     f"{_length(gap)} from its baked point, "
                                                     f"over {_length(tol)}"))
            if job.hold is not None and (missed := _off_surface(ctx, job)) is not None:
                length, edge = missed
                found.append(Finding(NAME, job.item, f"off the surface: the in-plane offset "
                                                     f"{length:.4f} lands outside it; the "
                                                     f"surface reaches {edge:.4f} that way"))
        for clip, clip_samples in _seated(ctx, only):
            gaps = [(float(np.linalg.norm(expected - actual)), s)
                    for s, expected, actual in _seat_root(ctx, clip, clip_samples)]
            gap, at = max(gaps, key=lambda g: g[0])  # max keeps the earliest of a tie
            tol = _seat_tolerance(ctx, clip)
            if gap > tol:
                frame = float(clip_samples.times[at])
                found.append(Finding(NAME, _seat_item(clip),
                                     f"root is {_length(gap)} from the seat-derived root at "
                                     f"frame {frame:g}, over {_length(tol)}"))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return [line for line in _jobs(ctx) if isinstance(line, str)] + list(
            _unmeasured_seats(ctx))
