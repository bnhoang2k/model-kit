"""Where a clip target points: the one place a ref and its offset become a world point.

A target names a ref (a hold, seat, mount or use on a placed or carried piece, or a `rig:<bone>`)
and an `offset`. The offset is in model units, in the frame of what the ref names, and that frame
is a rotation plus a translation only: any scale in a sampled bone or prop matrix is dropped
(the matrix's rotation part is orthonormalised), so an offset keeps its length. The frame is:

- a hold on a placed piece: the placement's rotation and position, fixed over the clip;
- a hold on a carried prop's piece: the prop's sampled pose at the key's sample, where the
  hold point goes through the full pose (it is a point on the prop's geometry, which the pose
  scales) and only the offset uses the scale-free frame;
- a `rig:<bone>` ref: the sampled bone matrix at the key's sample, point at the bone head.

The key's sample is the one whose time is nearest the key (times and keys are clip frames).
A miss, or a key too far from every sample, raises. Sample times must be finite and strictly
increasing. A matrix with no usable rotation raises `ManifestError`: singular, not finite, or a
reflection (negative determinant).

`Resolved.linear` is the ref's full 3x3 (scale included) from the ref's own units to world. It
equals `frame` unless a carried prop's pose scales its geometry. A check that compares an offset
with the piece's geometry converts the offset to piece units with `offset_in_piece`.

This module is also the one home of the frames the checks measure in: `piece_frame` and
`piece_frames` give a placed or carried piece's world-from-piece matrix at a sample (a
placement's matrix, or the carried prop's sampled pose, scale included), and `length_in_world`
converts a length between piece and world units along a direction, the one way every check
converts a world budget to a scaled piece. `tool_tip` is the one lookup of a target's tool and
holds the +Z convention of its tip. `seat_root` is the one seat rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION
from model_kit.core.manifest import Chain, Clip, ClipHold, Ref, SitSpec, Target, anchor
from model_kit.core.scene import ClipSamples, SceneCtx, check_times

# The glTF up axis.
UP = np.array([0.0, 1.0, 0.0])
# A tool's tip lies along its piece's +Z from its origin (a drumstick's does).
TIP_AXIS = np.array([0.0, 0.0, 1.0])


def length_in_world(linear, direction) -> float:
    """World length of one piece unit along `direction` (a piece-frame vector), under the
    3x3 `linear` from piece to world. Divide a world length by it to get piece units."""
    d = np.asarray(direction, dtype=float)
    return float(np.linalg.norm(np.asarray(linear, dtype=float) @ d) / np.linalg.norm(d))


@dataclass(frozen=True, eq=False)
class Resolved:
    """A target's ref in the world at the key's sample, before the offset."""

    sample: int  # index of the key's sample
    point: np.ndarray  # (3,) the ref's world point
    frame: np.ndarray  # (3, 3) rotation from the ref's frame to world, orthonormal
    linear: np.ndarray | None = None  # (3, 3) full map, scale included; None means `frame`

    @property
    def full(self) -> np.ndarray:
        """The ref's own units to world, scale included."""
        return self.frame if self.linear is None else self.linear

    def offset_in_piece(self, offset) -> np.ndarray:
        """The offset (model units, in the ref's frame) as a displacement in the piece's own
        units, the units of its hold points and triangles."""
        return np.linalg.solve(self.full, self.frame @ np.asarray(offset, dtype=float))

    def piece_length_in_world(self, direction) -> float:
        """World length of one piece unit along `direction` (a piece-frame vector)."""
        return length_in_world(self.full, direction)

    def with_offset(self, offset) -> np.ndarray:
        """The world point of `offset` (in the ref's frame) from the ref's point."""
        return self.point + self.frame @ np.asarray(offset, dtype=float)

    def offset_to(self, world_point) -> np.ndarray:
        """The offset (in the ref's frame) that lands at `world_point`."""
        return self.frame.T @ (np.asarray(world_point, dtype=float) - self.point)


def sample_at(samples: ClipSamples, t: float) -> int:
    """The index of the sample whose time is nearest `t` (a clip frame number).

    Times must be finite and strictly increasing, or it raises. It also raises when `t` lies
    outside the sampled range by more than half the gap to the end sample (one frame for a
    single sample): a key the samples do not reach would read another frame's pose in silence.
    """
    check_times(samples.times, f"clip {samples.clip!r}")
    times = np.asarray(samples.times, dtype=float)
    nearest = int(np.argmin(np.abs(times - t)))
    if times[0] <= t <= times[-1]:
        return nearest
    end = 0 if t < times[0] else times.size - 1
    gap = abs(times[1] - times[0]) if times.size > 1 and end == 0 else (
        abs(times[-1] - times[-2]) if times.size > 1 else 1.0)
    if abs(times[end] - t) > gap / 2:
        raise ManifestError(
            f"clip {samples.clip!r}: no sample near frame {t:g}: the samples run "
            f"{times[0]:g} to {times[-1]:g}, and {t:g} is more than half a sample step "
            f"({gap:g}) outside them")
    return nearest


def _rotation(matrix, what: str) -> np.ndarray:
    """The rotation part of a 4x4 matrix with its scale removed (polar decomposition)."""
    part = np.asarray(matrix, dtype=float)[:3, :3]
    if not np.all(np.isfinite(part)):
        raise ManifestError(f"{what}: its matrix is not finite")
    u, s, vt = np.linalg.svd(part)
    if not np.all(s > 0.0):
        raise ManifestError(f"{what}: its matrix is singular, so it has no frame")
    if np.linalg.det(part) < 0.0:
        raise ManifestError(f"{what}: its matrix is a reflection (negative determinant), so "
                            "it has no right-handed frame")
    return u @ vt


def piece_frame(clip: Clip, samples: ClipSamples, placed_by: str, s: int, what: str
                ) -> np.ndarray:
    """The (4, 4) world-from-piece matrix of a placed or carried piece at sample `s`.

    `placed_by` is an `Anchor.placed_by`: a placement key of the clip's `floor_gear` (fixed
    over the clip, no scale), or `prop:<prop id>` (the prop's sampled pose, scale included).
    """
    return piece_frames(clip, samples, placed_by, what)[s]


def piece_frames(clip: Clip, samples: ClipSamples, placed_by: str, what: str) -> np.ndarray:
    """`piece_frame` at every sample: (S, 4, 4). A miss raises."""
    if placed_by.startswith("prop:"):
        prop = placed_by.removeprefix("prop:")
        if prop not in samples.props:
            raise ManifestError(f"{what}: the samples hold no pose for prop {prop!r}")
        return np.asarray(samples.props[prop], dtype=float)
    placement = clip.floor_gear.pieces.get(placed_by)
    if placement is None:
        raise ManifestError(f"{what}: clip {clip.name!r} places no piece {placed_by!r}")
    return np.tile(placement.matrix(), (len(samples.times), 1, 1))


def resolve_target(ctx: SceneCtx, clip_name: str, target: Target) -> Resolved:
    """The target's ref in the world at its key's sample. A miss raises."""
    return resolve_ref(ctx, clip_name, target.ref, target.key,
                       what=f"{clip_name} target on {target.ref}")


def resolve_ref(ctx: SceneCtx, clip_name: str, ref: Ref, key: float, *,
                what: str | None = None) -> Resolved:
    """A ref (hold, seat, mount, use or `rig:<bone>`) in the world at the sample nearest
    `key`, before any offset. A miss raises."""
    clip = ctx.clips.get(clip_name)
    if clip is None:
        raise ManifestError(f"clip {clip_name!r}: not in the scene's clips")
    samples = ctx.samples.get(clip_name)
    if samples is None:
        raise ManifestError(f"clip {clip_name!r}: the scene has no samples for it")
    what = what or f"{clip_name} ref {ref}"
    s = sample_at(samples, key)
    if ref.is_bone:
        if ref.id not in samples.bones:
            raise ManifestError(f"{what}: the samples hold no bone {ref.id!r}")
        matrix = np.asarray(samples.bones[ref.id][s], dtype=float)
        return Resolved(s, matrix[:3, 3].copy(), _rotation(matrix, what))
    found = anchor(ctx.packs, ref, clip=clip)
    m = piece_frame(clip, samples, found.placed_by, s, what)
    # The item's point sits on the piece's geometry, so the full matrix (a carried prop's
    # scale too) carries it; only an offset uses the scale-free frame.
    point = m[:3, :3] @ np.asarray(found.item.point, dtype=float) + m[:3, 3]
    carried = found.placed_by.startswith("prop:")
    return Resolved(s, point, _rotation(m, what), m[:3, :3].copy() if carried else None)


@dataclass(frozen=True, eq=False)
class ToolTip:
    """A target's tool at one sample: its piece's world-from-piece pose and its `tip_length`."""

    pose: np.ndarray  # (4, 4) world from the tool's piece, scale included
    length: float  # the piece's measured tip_length, in the piece's own units

    @property
    def point(self) -> np.ndarray:
        """The tip in the world: `length` along the piece's +Z, through the pose (and its scale)."""
        return self.pose[:3, :3] @ (self.length * TIP_AXIS) + self.pose[:3, 3]


def tool_tip(ctx: SceneCtx, clip: Clip, item: str, tool: str, s: int) -> ToolTip | str:
    """The tool of a target at sample `s`, or the reason it cannot be measured (the caller makes
    that an `Unmeasured`). A tool that names no prop of the clip, or a prop whose piece no loaded
    pack holds, is a manifest error and raises."""
    prop = clip.props.get(tool)
    if prop is None:
        raise ManifestError(f"{item}: tool {tool!r} names no prop of the clip; it has "
                            f"{sorted(clip.props)}")
    pack = ctx.packs.get(prop.pack)
    piece = None if pack is None else pack.pieces.get(prop.piece)
    if piece is None:
        raise ManifestError(f"{item}: tool {tool!r} is piece {prop.pack}:{prop.piece}, "
                            "which no loaded pack holds")
    samples = ctx.samples[clip.name]
    if tool not in samples.props:
        return f"no sampled pose for tool {tool}"
    if "tip_length" not in piece.measured:
        return f"no measured tip_length for tool {tool}"
    return ToolTip(piece_frame(clip, samples, f"prop:{tool}", s, item),
                   float(piece.measured["tip_length"]))


def target_item(clip: Clip, index: int, target: Target) -> str:
    """The item name of a clip's target in the solver checks."""
    return f"{clip.name} targets[{index}]: {target.chain} -> {target.ref}"


def seat_root(seat_point, facing, sit: SitSpec, what: str = "seat") -> np.ndarray:
    """A seated rig's root bone head: `sit.forward` along the seat's facing from the seat top
    point, and `sit.below_seat` below it along glTF up. The facing is flattened to the ground
    plane and made unit length. Point and facing share one frame (the world, or a piece's own
    frame), and the result is in it. A facing straight up or down raises."""
    facing = np.asarray(facing, dtype=float)
    flat = facing - (facing @ UP) * UP
    length = float(np.linalg.norm(flat))
    if length < NO_DIRECTION:
        raise ManifestError(f"{what}: facing {tuple(float(v) for v in facing)} faces straight "
                            "up or down, so it has no forward")
    return (np.asarray(seat_point, dtype=float) - sit.below_seat * UP
            + sit.forward * flat / length)


def target_world(ctx: SceneCtx, clip_name: str, target: Target) -> np.ndarray:
    """The (3,) world point of the target's ref plus its offset, at the key's sample."""
    return resolve_target(ctx, clip_name, target).with_offset(target.offset)


def dangle_hold(clip: Clip, target: Target, chain: Chain) -> ClipHold | None:
    """The dangle hold on the target's ref for this chain's effector that covers the key, or
    None. Reach excuses such a target; solver error measures it against the dangle's budget."""
    return next((
        h for h in clip.holds
        if h.ref == target.ref and h.dangle is not None and h.who == chain.effector_group
        and any(t0 <= target.key <= t1 for t0, t1 in h.when)), None)
