"""Hold side and use side: the right hand is on the right.

A station's `facing` is the direction its user looks, toward the piece. The user's right at the
station is `right = facing x up` (glTF up is +y). Three rules:

- **Hold side, on the piece.** A hold is on the wrong side only when it lies past half a shoulder
  width on the other side: with `d = (point - station point) . right`, an `R` hold fails when
  `d` is under `-band` and an `L` hold when `d` is over `band`. The `band` is `HALF_SHOULDER_M`
  (half the biacromial breadth of a 5th-percentile woman, ANSUR II) times `units_per_metre`. A
  hand works across the body's midline up to the other shoulder, so a hold on the centre plane,
  or short of the other shoulder, has a side. `both` passes (a plain coverage line). The item is
  `<pack>:<piece> hold <id> side`.
- **Hold side, in a clip.** The hand (or the held prop) of a clip hold must not be past half a
  shoulder width on the wrong side of the rig. The side is measured, never read from a name:
  `d = (the who points' centroid - the rig root) . right` at the hold's first due sample, with
  `right` turned by the placed piece's pose, judged against the same band. A hold with a
  `cross` is excused (a plain coverage line), as is a `both` hold. A clip with no samples, or a
  hold with no due sample, is `Unmeasured`. The item is `<clip> holds[<i>]: <who> on <ref> side`.
- **Use side, in a clip.** For each placed piece with a use station that has holds, the horizontal
  direction from the rig root (the root bone's head at the clip's first sample) to the placed
  centroid of the station's hold points must lie within `USE_ANGLE_DEG` of the placed station's
  facing. A station with no holds is a plain coverage line. The item is
  `<clip> use <placement>.<use>`.

A scene with a rigged clip and nothing to judge (no hold on a piece, none in a clip, no station
with holds) prints one not-applicable line, so a body that holds nothing is not a silent pass. A
`both` hold prints `not applicable (both-side hold)`.

The plants are one unit per judged item, built from the manifest and, for a clip, its samples
(never from the side or the angle the check measures): a hold's side swapped, and, when a hold or
hand sits inside the band so the swap leaves it passing, the hold's point (or the hand's points at
the hold's first due sample) moved across the centre plane to `CALIBRATION_FAIL` bands on the
wrong side (the first two rules); the placement turned half a turn in place and, if that leaves the angle in range, the
station's facing reversed instead (the third). The gate tries the candidates in order, skips a
unit whose item already fails, and stops the run when the check catches none.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.entries import Candidate, Entry, unit
from model_kit.core.checks.pieces import with_piece
from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION, transform
from model_kit.core.manifest import (
    Anchor,
    Clip,
    ClipHold,
    Hold,
    Piece,
    Placement,
    Use,
    anchor,
)
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.rig import rig_root
from model_kit.core.scene import (
    ClipSamples,
    SceneCtx,
    check_windows,
    due,
    piece_geom,
    piece_key,
    rig_spec,
)
from model_kit.core.targets import UP, piece_frame
from model_kit.core.tolerances import CALIBRATION_FAIL, HALF_SHOULDER_M, USE_ANGLE_DEG

NAME = "sides"
OTHER = {"R": "L", "L": "R"}


# --- the building blocks ----------------------------------------------------------------------


def use_of(piece: Piece, id: str, what: str) -> Use:
    use = next((u for u in piece.uses if u.id == id), None)
    if use is None:
        raise ManifestError(f"{what}: names use station {id!r}, which {piece.name!r} lacks")
    return use


def right_of(use: Use, what: str) -> np.ndarray:
    """The user's right at a station: `facing x up`, as a unit vector (horizontal, whatever the
    length or the tilt of `facing`), so a distance along it is a length. A station that faces
    straight up or down has none, so it raises."""
    right = np.cross(np.asarray(use.facing, dtype=float), UP)
    length = float(np.linalg.norm(right))
    if length < NO_DIRECTION:
        raise ManifestError(f"{what}: use station {use.id!r} faces straight up or down, so it "
                            "has no right")
    return right / length


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    """`v` as a unit vector. A zero vector (a degenerate pose) raises, never passes."""
    length = float(np.linalg.norm(v))
    if length < NO_DIRECTION:
        raise ManifestError(f"{what}: the placed station's right has no length")
    return v / length


def _offset(point: np.ndarray, origin: np.ndarray, right: np.ndarray) -> float:
    """How far `point` lies to the user's right of `origin`, along the unit vector `right`."""
    return float((np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)) @ right)


def band_of(ctx: SceneCtx) -> float:
    """Half a shoulder width in the scene's units: how far past the centre plane a hold or a
    hand may sit on the wrong side before it fails."""
    return HALF_SHOULDER_M * ctx.style.units_per_metre


def _wrong(side: str, d: float, band: float) -> bool:
    """Whether `d` (the offset to the user's right) is past the band on the side opposite
    `side`: `R` fails under `-band`, `L` over `band`."""
    return d < -band if side == "R" else d > band


def _across(side: str, band: float) -> float:
    """The offset to the user's right that puts a hold of `side` `CALIBRATION_FAIL` bands on the
    wrong side."""
    return (-1.0 if side == "R" else 1.0) * CALIBRATION_FAIL * band


def _with_hold(ctx: SceneCtx, pack: str, piece: Piece, hold: Hold, **fields) -> SceneCtx:
    holds = tuple(dataclasses.replace(h, **fields) if h.id == hold.id else h for h in piece.holds)
    return with_piece(ctx, pack, piece.name, dataclasses.replace(piece, holds=holds))


def _flipped(ctx: SceneCtx, pack: str, piece: Piece, hold: Hold) -> SceneCtx:
    return _with_hold(ctx, pack, piece, hold, side=OTHER[hold.side])


def _with_use(ctx: SceneCtx, pack: str, piece: Piece, use: Use) -> SceneCtx:
    uses = tuple(use if u.id == use.id else u for u in piece.uses)
    return with_piece(ctx, pack, piece.name, dataclasses.replace(piece, uses=uses))


def _root(ctx: SceneCtx, clip: Clip, samples: ClipSamples, s: int) -> np.ndarray:
    """The rig root bone's head at sample `s`. The root is `rig.root`, else the one bone with no
    parent."""
    try:
        root = rig_root(rig_spec(ctx, clip.name))
    except ManifestError as err:
        raise ManifestError(f"clip {clip.name!r}: {err}") from err
    if root not in samples.bones:
        raise ManifestError(f"clip {clip.name!r}: the samples hold no root bone {root!r}")
    return np.asarray(samples.bones[root][s], dtype=float)[:3, 3]


def _who_centroid(ctx: SceneCtx, clip: Clip, hold: ClipHold, samples: ClipSamples, s: int,
                  item: str) -> np.ndarray:
    """The centroid of the hold's `who` points at sample `s`: the contact group's points, or
    the points of the held prop (inside its `who_box`, if it has one) carried by its pose."""
    prop = clip.props.get(hold.who)
    if prop is None:
        if hold.who not in samples.groups:
            raise ManifestError(f"{item}: the samples hold no contact group {hold.who!r}")
        return np.asarray(samples.groups[hold.who][s], dtype=float).reshape(-1, 3).mean(axis=0)
    if hold.who not in samples.props:
        raise ManifestError(f"{item}: the samples hold no pose for prop {hold.who!r}")
    geom = piece_geom(ctx, prop.pack, prop.piece)
    points = np.unique(np.concatenate([np.asarray(t, dtype=float).reshape(-1, 3)
                                       for t in geom.parts.values()]), axis=0)
    if hold.who_box is not None:
        lo, hi = np.asarray(hold.who_box[0]), np.asarray(hold.who_box[1])
        points = points[((points >= lo) & (points <= hi)).all(axis=1)]
    if len(points) == 0:
        raise ManifestError(f"{item}: the who_box holds none of prop {hold.who!r}'s points")
    return transform(points, np.asarray(samples.props[hold.who][s], dtype=float)).mean(axis=0)


# --- the judged items -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Item:
    """An item the check judges, with how to measure it and how to plant it, each run only when
    asked for: a plant never runs `measure`, and a measure never runs `candidates`."""

    item: str
    measure: Callable[[], Entry]
    candidates: Callable[[], Iterator[Candidate]]


def _not_applicable(why: str, item: str) -> str:
    return f"{NAME}: not applicable ({why}): {item}"


# --- hold side, on the piece ------------------------------------------------------------------


def _piece_entry(item: str, hold: Hold, piece: Piece, band: float) -> Entry:
    use = use_of(piece, hold.use, item)
    d = _offset(hold.point, use.point, right_of(use, item))
    detail = (f"side {hold.side}, but the point lies {d:+.4f} to the right of station "
              f"{use.id!r}, past half a shoulder width ({band:.4f}) on the other side")
    return Entry(item, not _wrong(hold.side, d, band), detail)


def _piece_candidates(ctx: SceneCtx, pack: str, piece: Piece, hold: Hold
                      ) -> Iterator[tuple[str, SceneCtx]]:
    """The hold's side swapped; then, for a hold inside the band where the swap leaves it
    passing, its point moved across the station's centre plane to `CALIBRATION_FAIL` bands on
    the wrong side (the point's other two coordinates stay)."""
    yield "swapped", _flipped(ctx, pack, piece, hold)
    use = use_of(piece, hold.use, f"{piece_key(pack, piece.name)} hold {hold.id}")
    right = right_of(use, f"{piece_key(pack, piece.name)} hold {hold.id}")
    point = np.asarray(hold.point, dtype=float)
    moved = point + (_across(hold.side, band_of(ctx)) - _offset(point, use.point, right)) * right
    yield "moved across", _with_hold(ctx, pack, piece, hold, point=tuple(float(v) for v in moved))


def _piece_items(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[_Item | str]:
    for pack_name, pack in sorted(ctx.packs.items()):
        for piece in pack.pieces.values():
            for hold in piece.holds:
                item = f"{piece_key(pack_name, piece.name)} hold {hold.id} side"
                if only is not None and item not in only:
                    continue
                if hold.side == "both":
                    yield _not_applicable("both-side hold",
                                          f"{piece_key(pack_name, piece.name)} hold {hold.id}")
                    continue
                yield _Item(item,
                            lambda i=item, h=hold, p=piece: _piece_entry(i, h, p, band_of(ctx)),
                            lambda n=pack_name, p=piece, h=hold: _piece_candidates(ctx, n, p, h))


# --- hold side, in a clip ---------------------------------------------------------------------


def _hold_entry(ctx: SceneCtx, clip: Clip, hold: ClipHold, found: Anchor, samples: ClipSamples,
                s: int, item: str) -> Entry:
    spot = found.item
    use = use_of(found.piece, spot.use, item)
    frame = piece_frame(clip, samples, found.placed_by, s, item)
    right = _unit(frame[:3, :3] @ right_of(use, item), item)
    d = _offset(_who_centroid(ctx, clip, hold, samples, s, item), _root(ctx, clip, samples, s),
                right)
    band = band_of(ctx)
    detail = (f"the hold is side {spot.side}, but {hold.who} lies {d:+.4f} to the user's right "
              f"of the rig root at frame {samples.times[s]:g}, past half a shoulder width "
              f"({band:.4f}) on the other side")
    return Entry(item, not _wrong(spot.side, d, band), detail)


def _hand_candidates(ctx: SceneCtx, clip: Clip, hold: ClipHold, found: Anchor,
                     samples: ClipSamples, s: int, item: str
                     ) -> Iterator[tuple[str, SceneCtx, Collection[int]]]:
    """The placed piece's hold side swapped; then, for a hand inside the band where the swap
    leaves it passing, the hand's points (or the held prop's pose) at sample `s` moved across the
    rig root's centre plane to `CALIBRATION_FAIL` bands on the wrong side."""
    spot = found.item
    yield "swapped", _flipped(ctx, hold.ref.pack, found.piece, spot), ()
    use = use_of(found.piece, spot.use, item)
    frame = piece_frame(clip, samples, found.placed_by, s, item)
    right = _unit(frame[:3, :3] @ right_of(use, item), item)
    offset = _offset(_who_centroid(ctx, clip, hold, samples, s, item),
                     _root(ctx, clip, samples, s), right)
    delta = (_across(spot.side, band_of(ctx)) - offset) * right
    if hold.who in clip.props:
        props = {**samples.props, hold.who: np.array(samples.props[hold.who], dtype=float)}
        props[hold.who][s][:3, 3] += delta
        moved = dataclasses.replace(samples, props=props)
    else:
        groups = {**samples.groups, hold.who: np.array(samples.groups[hold.who], dtype=float)}
        groups[hold.who][s] += delta
        moved = dataclasses.replace(samples, groups=groups)
    yield ("moved across", dataclasses.replace(ctx, samples={**ctx.samples, clip.name: moved}),
           (s,))


def _hold_items(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[_Item | str]:
    for clip in ctx.clips.values():
        for i, hold in enumerate(clip.holds):
            item = f"{clip.name} holds[{i}]: {hold.who} on {hold.ref} side"
            if only is not None and item not in only:
                continue
            found = anchor(ctx.packs, hold.ref, clip=clip)
            if found.kind != "hold":
                raise ManifestError(f"{item}: {hold.ref} names a {found.kind}, not a hold")
            spot = found.item
            if hold.cross is not None:
                yield f"{NAME}: crossing by design: {item}"
                continue
            if spot.side == "both":
                yield _not_applicable("both-side hold", item)
                continue
            samples = ctx.samples.get(clip.name)
            if samples is None:
                yield Unmeasured(f"{NAME}: skipped (no samples): {item}")
                continue
            check_windows(clip, hold.when, item)
            first = due(samples.times, hold.when)
            if len(first) == 0:
                yield Unmeasured(f"{NAME}: skipped (no sample inside its when windows): {item}")
                continue
            yield _Item(item,
                        lambda c=clip, h=hold, f=found, sm=samples, s=int(first[0]), it=item:
                        _hold_entry(ctx, c, h, f, sm, s, it),
                        lambda c=clip, h=hold, f=found, sm=samples, s=int(first[0]), it=item:
                        _hand_candidates(ctx, c, h, f, sm, s, it))


# --- use side, in a clip ----------------------------------------------------------------------


def _turned(placement: Placement) -> Placement:
    """The placement turned half a turn about the up axis, in place: q' = (0, 1, 0, 0) * q."""
    x, y, z, w = placement.quaternion
    return dataclasses.replace(placement, quaternion=(z, w, -x, -y))


def _flat(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], 0.0, v[2]])


def _use_items(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[_Item | str]:
    for clip in ctx.clips.values():
        for key, placement in clip.floor_gear.pieces.items():
            pack = ctx.packs.get(placement.pack)
            piece = None if pack is None else pack.pieces.get(placement.piece)
            if piece is None:
                raise ManifestError(f"clip {clip.name!r}: placement {key!r} names "
                                    f"{placement.pack}:{placement.piece}, which the scene lacks")
            for use in piece.uses:
                item = f"{clip.name} use {key}.{use.id}"
                if only is not None and item not in only:
                    continue
                holds = [h for h in piece.holds if h.use == use.id]
                if not holds:
                    yield f"{NAME}: no holds on station: {item}"
                    continue
                samples = ctx.samples.get(clip.name)
                if samples is None:
                    yield Unmeasured(f"{NAME}: skipped (no samples): {item}")
                    continue
                yield _Item(item,
                            lambda c=clip, k=key, pl=placement, pc=piece, u=use, hs=holds,
                            sm=samples, it=item: _use_entry(ctx, c, k, pl, pc, u, hs, sm, it),
                            lambda c=clip, k=key, pl=placement, pc=piece, u=use:
                            _use_candidates(ctx, c, k, pl, pc, u))


def _use_candidates(ctx: SceneCtx, clip: Clip, key: str, placement: Placement, piece: Piece,
                    use: Use) -> Iterator[tuple[str, SceneCtx]]:
    """The placement turned half a turn in place, then the station's facing reversed."""
    gear = dataclasses.replace(clip.floor_gear,
                               pieces={**clip.floor_gear.pieces, key: _turned(placement)})
    yield "turned", dataclasses.replace(
        ctx, clips={**ctx.clips, clip.name: dataclasses.replace(clip, floor_gear=gear)})
    reversed_use = dataclasses.replace(use, facing=tuple(-float(v) for v in use.facing))
    yield "faced away", _with_use(ctx, placement.pack, piece, reversed_use)


def _use_entry(ctx: SceneCtx, clip: Clip, key: str, placement: Placement, piece: Piece,
               use: Use, holds: list[Hold], samples: ClipSamples, item: str) -> Entry:
    centre = placement.to_world(np.mean([h.point for h in holds], axis=0))
    toward = _flat(centre - _root(ctx, clip, samples, 0))
    facing = _flat(placement.matrix()[:3, :3] @ np.asarray(use.facing, dtype=float))
    if float(np.linalg.norm(facing)) < NO_DIRECTION:
        raise ManifestError(f"{item}: use station {use.id!r} faces straight up or down, so it "
                            "has no facing on the ground")
    if float(np.linalg.norm(toward)) < NO_DIRECTION:
        ok = False
        detail = "the station's holds lie straight above the rig root, so they have no direction"
    else:
        cosine = float(toward @ facing / np.linalg.norm(toward) / np.linalg.norm(facing))
        angle = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
        ok = angle <= USE_ANGLE_DEG
        detail = (f"the holds lie {angle:.1f} degrees off the station's facing (limit "
                  f"{USE_ANGLE_DEG:g})")
    return Entry(item, ok, detail)


# --- the check --------------------------------------------------------------------------------


def _scan(ctx: SceneCtx, only: Collection[str] | None = None) -> tuple[list[_Item], list[str]]:
    items, lines = [], []
    for part in (_piece_items, _hold_items, _use_items):
        for out in part(ctx, only):
            (items if isinstance(out, _Item) else lines).append(out)
    return items, lines


def _findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    entries = (it.measure() for it in _scan(ctx, only)[0])
    return [Finding(NAME, e.item, e.detail) for e in entries if not e.ok]


def _units(ctx: SceneCtx) -> list[PlantUnit]:
    return [unit(it.item, it.candidates()) for it in _scan(ctx)[0]]


class Sides:
    name = NAME
    kinds: frozenset[str] = frozenset()  # piece manifests of every kind and clips: no one recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return _units(ctx)

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # The hold side reads each hold's first due sample and the use side the clip's first:
        # `samples` is accepted and ignored.
        return _findings(ctx, only)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        items, lines = _scan(ctx)
        rigged = any(name in ctx.rig_of_clip for name in ctx.clips)
        if rigged and not items:  # a body that holds nothing has no side to judge
            nothing = "no piece hold, clip hold or use station with holds to judge"
            lines = [*lines, f"{NAME}: not applicable ({nothing})"]
        return lines
