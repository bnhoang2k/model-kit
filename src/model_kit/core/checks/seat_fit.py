"""Check seat-fit: a sitter's knees pass the seat's front edge.

A seat is shallower than the sitter's buttock-to-knee-crease length, so the knees pass the front
edge and the lower legs hang free. The rule is a modelling choice of this kit.

For every clip whose posture is `sit` and that names a seat, at every sample inside the clip's seat
windows (`manifest.seat_windows`), each bone of the rig's `sit.past_edge` has its head past the
seat's front edge by more than `SKIN_OF_EFFECTOR` times `r`. `r` is the rig's smallest chain or
support radius (`core.supports.body_radius`).

- **The front edge** is the furthest point of the seat's `surface` parts along the seat's facing,
  measured from the seat point, at the placed transform (`front_edge`). The facing is flattened to
  the ground plane and made unit length, as `targets.seat_root` does for the sitter's root. The
  edge is a line across the facing, so how far a knee lies to the side does not matter.
- **The margin** is a design call. A knee joint's centre sits forward of the knee crease, so a
  centre on the edge already has the crease behind it. The skin factor then only absorbs float
  noise. A leg that sinks into the seat stays overlap's finding; this check adds only the edge.

Items: `<clip> <bone> behind the edge`. One finding carries the smallest margin over the clip's
seat samples. One allow line covers one clip and one bone.

Coverage. A rule that does not apply by design prints a `not applicable` line: a rig with no `sit`,
a clip whose posture is not `sit` (a `sit` clip always names a seat: `Clip.validate` refuses one
that does not). Missing data is `Unmeasured`: a
clip with no posture, a rig whose `sit` names no `past_edge` bone, a clip with no samples or no
rig, a `past_edge` bone the samples do not hold, no sample inside the seat windows, a seat with no
surface geometry, a seat that faces straight up or down, and a rig with no radius to measure by.

Plants, one unit per measured item, built from the sitter's samples and the seat alone: move the
whole sitter back along the seat's facing until that bone's head lies `CALIBRATION_FAIL` tolerances
behind the edge, seen at the first, the middle and the last sample inside the seat windows. The
gate skips an item that already fails, because `run` reports it. Each scene's seat setup is built
once and shared by the run, the plants and the coverage.

The check measures clips, not a recipe, so `kinds` is empty.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.entries import first_middle_last, unit
from model_kit.core.errors import Unmeasurable
from model_kit.core.geom import NO_DIRECTION
from model_kit.core.manifest import Clip, Ref, RigSpec, Seat, anchor, seat_windows
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import ClipSamples, SceneCtx, due, piece_key, rig_spec
from model_kit.core.supports import body_radius
from model_kit.core.targets import UP, resolve_ref
from model_kit.core.tolerances import CALIBRATION_FAIL, SKIN_OF_EFFECTOR

NAME = "seat-fit"
PRINCIPLE = ("Seat depth is shorter than the sitter's buttock-to-knee-crease length, so the "
             "knees pass the seat's front edge and the lower legs hang free.")
SOURCE = (
    "CityU online anthropometry, seat depth: the 5th percentile buttock-popliteal length sets the "
    "maximum seat depth (https://personal.cityu.edu.hk/meachan/online%20anthropometry/Chapter2/"
    "Ch2-14.htm). The margin past the edge is this kit's call: the knee joint's centre sits "
    "forward of the crease, so a centre on the edge already has the crease behind it.")


def _forward(facing, what: str) -> np.ndarray:
    """The facing flattened to the ground plane and made unit length. A facing straight up or
    down has no forward and raises `Unmeasurable`."""
    facing = np.asarray(facing, dtype=float)
    flat = facing - (facing @ UP) * UP
    length = float(np.linalg.norm(flat))
    if length < NO_DIRECTION:
        raise Unmeasurable(f"{what}: facing {tuple(float(v) for v in facing)} faces straight up "
                           "or down, so it has no forward")
    return flat / length


def _surface(ctx: SceneCtx, ref: Ref, clip: Clip | None) -> np.ndarray:
    """The triangle corners of a seat's surface parts, in the piece's own frame: (N, 3). Raises
    `Unmeasurable` for a seat with no surface part or a piece with no geometry."""
    found = anchor(ctx.packs, ref, clip=clip)
    geom = ctx.pieces.get(piece_key(ref.pack, found.piece.name))
    if geom is None:
        raise Unmeasurable(f"no geometry for piece {ref.pack}:{found.piece.name}")
    names = found.item.surface
    if not names:
        raise Unmeasurable(f"seat {ref} names no surface part")
    missing = [n for n in names if n not in geom.parts]
    if missing:
        raise Unmeasurable(f"seat {ref}: no geometry for surface part {missing[0]!r}")
    return np.concatenate([np.asarray(geom.parts[n], dtype=float).reshape(-1, 3) for n in names])


def _reach(points: np.ndarray, seat: Seat, full: np.ndarray, facing, ref: Ref) -> float:
    """The largest distance of `points` (piece frame) from the seat point along the flattened
    facing, after the map `full` to the world."""
    forward = _forward(facing, f"seat {ref}")
    return float(((points - np.asarray(seat.point, dtype=float)) @ full.T @ forward).max())


def front_edge(ctx: SceneCtx, seat_ref: Ref, clip: Clip | None = None, s: int = 0) -> float:
    """How far the seat's surface parts reach along the seat's facing, from the seat point: the
    largest distance along the flattened facing, in world units.

    With a `clip`, the seat is read as the clip places it at sample `s` (the placement of a floor
    piece, the sampled pose of a carried prop). With none, it is read in the piece's own frame."""
    found = anchor(ctx.packs, seat_ref, clip=clip)
    if found.kind != "seat":
        raise Unmeasurable(f"{seat_ref} names a {found.kind}, not a seat")
    seat = found.item
    points = _surface(ctx, seat_ref, clip)
    if clip is None:
        return _reach(points, seat, np.eye(3), seat.facing, seat_ref)
    samples = ctx.samples.get(clip.name)
    if samples is None:
        raise Unmeasurable(f"clip {clip.name!r} has no samples to place the seat by")
    res = resolve_ref(ctx, clip.name, seat_ref, float(samples.times[s]))
    return _reach(points, seat, res.full, res.frame @ np.asarray(seat.facing, dtype=float),
                  seat_ref)


# --- one prepared clip ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Seat:
    """The seat at one sample: its point and forward in the world, and its front edge from the
    point."""

    point: np.ndarray
    forward: np.ndarray
    edge: float


@dataclass(eq=False)
class _Clip:
    ctx: SceneCtx
    clip: Clip
    samples: ClipSamples
    spec: RigSpec
    tol: float
    ids: list[int]  # the samples inside the seat windows
    bones: list[str]  # the past_edge bones the samples hold
    seat: dict[int, _Seat]  # per sample in `ids`, where the seat is


def _item(clip: Clip, bone: str) -> str:
    return f"{clip.name} {bone} behind the edge"


@dataclass
class _Work:
    """What one clip has to measure, and the coverage lines for the rest."""

    ready: _Clip | None = None
    lines: list[str] = dataclasses.field(default_factory=list)


def _work(ctx: SceneCtx, clip: Clip) -> _Work:
    w = _Work()
    name = clip.name
    whole = f"{name} behind the edge"

    def n_a(why: str) -> _Work:
        w.lines.append(f"{NAME}: not applicable ({why}): {whole}")
        return w

    def skip(why: str, item: str = whole) -> _Work:
        w.lines.append(Unmeasured(f"{NAME}: skipped ({why}): {item}"))
        return w

    if name not in ctx.rig_of_clip:
        return skip("no rig")
    spec = rig_spec(ctx, name)
    if spec.sit is None:
        return n_a("the rig has no sit")
    if clip.posture is None:
        return skip("the clip has no posture")
    if clip.posture != "sit":
        return n_a(f"{clip.posture} clip: only a sit clip rests on a seat")
    # a `sit` clip names a seat: `Clip.validate` refuses one that does not, so there is no branch
    if not spec.sit.past_edge:
        return skip("the rig's sit names no past_edge bone")
    samples = ctx.samples.get(name)
    if samples is None:
        return skip("no samples")
    bones = []
    for bone in spec.sit.past_edge:
        if bone in samples.bones:
            bones.append(bone)
        else:
            skip(f"the samples hold no bone {bone}", _item(clip, bone))
    ids = [int(s) for s in due(samples.times, seat_windows(clip))]
    if not ids:
        return skip("no sample inside the seat windows")
    radius = body_radius(spec, ctx.rests.get(ctx.rig_of_clip[name]))
    if radius is None:
        return skip("no radius to measure by")
    try:
        seat = _seats(ctx, clip, samples, ids)
    except Unmeasurable as err:
        return skip(str(err))
    w.ready = _Clip(ctx, clip, samples, spec, SKIN_OF_EFFECTOR * radius, ids, bones, seat)
    return w


def _seats(ctx: SceneCtx, clip: Clip, samples: ClipSamples, ids: list[int]) -> dict[int, _Seat]:
    """The seat at each sample of `ids`, resolved at that sample's own time (a seat on a carried
    prop moves with the prop). Raises `Unmeasurable` when any sample cannot be read."""
    found = anchor(ctx.packs, clip.seat, clip=clip)
    if found.kind != "seat":
        raise Unmeasurable(f"{clip.seat} names a {found.kind}, not a seat")
    points = _surface(ctx, clip.seat, clip)
    out = {}
    for s in ids:
        res = resolve_ref(ctx, clip.name, clip.seat, float(samples.times[s]))
        facing = res.frame @ np.asarray(found.item.facing, dtype=float)
        forward = _forward(facing, f"seat {clip.seat}")
        out[s] = _Seat(res.point, forward,
                       _reach(points, found.item, res.full, facing, clip.seat))
    return out


# --- the measure -----------------------------------------------------------------------------------


@dataclass
class _Reach:
    """Where one bone's head lies at one sample, past the front edge (negative: behind it)."""

    sample: int
    past: float


def _past(c: _Clip, bone: str, s: int) -> float:
    """How far the bone's head lies past the seat's front edge at sample `s`."""
    seat = c.seat[s]
    head = np.asarray(c.samples.bones[bone][s], dtype=float)[:3, 3]
    return float((head - seat.point) @ seat.forward) - seat.edge


def _worst(c: _Clip, bone: str, ids: list[int]) -> _Reach:
    return min((_Reach(s, _past(c, bone, s)) for s in ids), key=lambda r: r.past)


def _finding(c: _Clip, bone: str, ids: list[int]) -> Finding | None:
    worst = _worst(c, bone, ids)
    if worst.past > c.tol:
        return None
    where = (f"{-worst.past:.4f} behind the edge" if worst.past < 0
             else f"only {worst.past:.4f} past the edge")
    return Finding(NAME, _item(c.clip, bone),
                   f"the head is {where} at frame {c.samples.times[worst.sample]:g}, short of "
                   f"the {c.tol:.4f} it must be past")


def _findings(works: dict[str, _Work], only: Collection[str] | None, samples: frozenset[int]
              ) -> list[Finding]:
    found: list[Finding] = []
    for work in works.values():
        c = work.ready
        if c is None:
            continue
        ids = [s for s in c.ids if not samples or s in samples]
        for bone in c.bones:
            if only is not None and _item(c.clip, bone) not in only:
                continue
            if ids and (f := _finding(c, bone, ids)) is not None:
                found.append(f)
    return found


# --- plants ------------------------------------------------------------------------------------------


def _moved_back(c: _Clip, bone: str, s: int) -> tuple[str, SceneCtx, set[int]]:
    """The whole sitter moved back along the seat's facing until the bone's head lies
    `CALIBRATION_FAIL` tolerances behind the edge at sample `s`: (label, the copy, the sample)."""
    back = (_past(c, bone, s) + CALIBRATION_FAIL * c.tol) * c.seat[s].forward
    bones = {}
    for name, matrices in c.samples.bones.items():
        moved = np.array(matrices, dtype=float)
        moved[:, :3, 3] -= back
        bones[name] = moved
    groups = {g: np.asarray(p, dtype=float) - back for g, p in c.samples.groups.items()}
    shells = {p: (np.asarray(v, dtype=float) - back).astype(np.asarray(v).dtype)
              for p, v in c.samples.shells.items()}
    moved_samples = dataclasses.replace(c.samples, bones=bones, groups=groups, shells=shells)
    ctx = dataclasses.replace(c.ctx, samples={**c.ctx.samples, c.clip.name: moved_samples})
    return "behind the edge", ctx, {s}


class SeatFit:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe
    principle = PRINCIPLE
    source = SOURCE
    _works: tuple[SceneCtx, dict[str, _Work]] | None = None

    def _all(self, ctx: SceneCtx) -> dict[str, _Work]:
        """Every clip's seat setup, built once per scene: the run, the plants and the
        coverage of one scene share it."""
        if self._works is None or self._works[0] is not ctx:
            self._works = (ctx, {name: _work(ctx, clip) for name, clip in ctx.clips.items()})
        return self._works[1]

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        """One unit per measured item, from the sitter's samples and the seat alone: move the
        sitter back at the first, the middle and the last sample inside the seat windows."""
        units = []
        for work in self._all(ctx).values():
            c = work.ready
            if c is None:
                continue
            moves = first_middle_last(c.ids)
            units += [unit(_item(c.clip, bone), [_moved_back(c, bone, s) for s in moves])
                      for bone in c.bones]
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        return _findings(self._all(ctx), only, samples)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        for work in self._all(ctx).values():
            lines += work.lines
        return list(dict.fromkeys(lines))

