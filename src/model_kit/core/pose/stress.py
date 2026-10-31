"""The stress poses of a body: each bending joint, and each joint with its parent joint, taken to
the ends of its range of motion, as synthetic samples a check can read.

Think of a doctor testing a joint: bend it as far as it goes, then bend the joint above it as
far as it goes too. A body that is whole at those poses is whole in between.

The set:

- For each bending joint with a bending parent joint (the nearest bending bone above it): every
  mix of both joints' axis ends, everything else at rest. A 3-axis shoulder and a 1-axis elbow
  give 2^3 x 2^1 = 16.
- Then, for each bending joint: each of its axes to each end alone.

A range is `rig.joints[bone].range_deg`, degrees about the bone's rest frame in the order
`rig.pose_world` uses. The order is the rig's bone order for joints, `x`, `y`, `z` for axes and the
low end before the high end, so a rig always gives the same list. A pose is named
`<child>+<parent> <child ends>; <parent ends>` (`elbow+shoulder x lo; x hi y lo z hi`) or
`<joint> <axis> <end>` (`knee x hi`). The names hold no `|` (an allowlist line splits on it) and no
`*?[` (it matches by `fnmatch`).

`stress_scene` poses the body at every one of them, with `core.pose.skin`, and returns the scene
with one synthetic clip, `STRESS_CLIP`, whose samples are the poses. A real clip's name cannot
hold a space, so the name never clashes with one.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.manifest import Clip, FloorGear, RigSpec
from model_kit.core.manifest.rig import bending_joints
from model_kit.core.pose.skin import skin
from model_kit.core.rig import Pose, pose_world
from model_kit.core.scene import ClipSamples, SceneCtx

STRESS_CLIP = "stress poses"
AXES = "xyz"
ENDS = ("lo", "hi")  # the low and the high end of a range, in that order
FPS = 24.0  # a synthetic clip needs one; no check reads it


@dataclass(frozen=True, eq=False)
class StressPose:
    """One pose of the set: its `name`, the `joints` it turns (child first, then parent) and the
    `pose` itself (`rig.pose_world` reads it; a bone not in it sits at rest)."""

    name: str
    joints: tuple[str, ...]
    pose: Pose


def missing_ranges(spec: RigSpec) -> tuple[str, ...]:
    """The bending joints of the rig that have no `joints` entry, in bone order."""
    return tuple(b for b in bending_joints(spec) if b not in spec.joints)


def _axes(spec: RigSpec, joint: str) -> list[tuple[str, tuple[float, float]]]:
    """A joint's axes in x, y, z order, each with its range in radians."""
    ranges = spec.joints[joint].range_deg
    return [(a, (math.radians(ranges[a][0]), math.radians(ranges[a][1])))
            for a in AXES if a in ranges]


def _turn(chosen: dict[str, dict[str, float]]) -> Pose:
    pose: Pose = {}
    for joint, angles in chosen.items():
        euler = np.zeros(3)
        for axis, angle in angles.items():
            euler[AXES.index(axis)] = angle
        pose[joint] = (euler, np.zeros(3))
    return pose


def _mixes(spec: RigSpec, joints: Sequence[str]):
    """Every mix of the axis ends of `joints`: (the name of the mix, the angles per joint)."""
    slots = [(j, a, r) for j in joints for a, r in _axes(spec, j)]
    for ends in itertools.product(range(2), repeat=len(slots)):
        chosen: dict[str, dict[str, float]] = {j: {} for j in joints}
        parts: dict[str, list[str]] = {j: [] for j in joints}
        for (joint, axis, rng), end in zip(slots, ends, strict=True):
            chosen[joint][axis] = rng[end]
            parts[joint] += [axis, ENDS[end]]
        yield "; ".join(" ".join(parts[j]) for j in joints), chosen


def stress_poses(spec: RigSpec, *, partial: bool = False) -> list[StressPose]:
    """The stress poses of a rig (the module's docstring has the set and the names).

    A bending joint with no `joints` entry raises `Unmeasurable`, naming each. With `partial`, a
    joint with no entry is left out instead, with every pair that holds it, and the rest are
    posed (a caller then reports the missing ones as unmeasured)."""
    missing = missing_ranges(spec)
    if missing and not partial:
        raise Unmeasurable(f"no joint range for the bending joint(s) {', '.join(missing)}, so "
                           "their stress poses cannot be taken")
    every = set(bending_joints(spec))
    bending = [b for b in bending_joints(spec) if b in spec.joints]
    parent = {b.name: b.parent for b in spec.bones}
    poses: list[StressPose] = []
    for joint in bending:
        up = parent[joint]
        while up is not None and up not in every:
            up = parent[up]
        if up is None or up not in spec.joints:
            continue
        for mix, chosen in _mixes(spec, (joint, up)):
            poses.append(StressPose(f"{joint}+{up} {mix}", (joint, up), _turn(chosen)))
    for joint in bending:
        for axis, rng in _axes(spec, joint):
            for end in range(2):
                poses.append(StressPose(f"{joint} {axis} {ENDS[end]}", (joint,),
                                        _turn({joint: {axis: rng[end]}})))
    return poses


def _groups(spec: RigSpec, rest, shells: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each contact group's points per sample, as a clip's `g:` arrays hold them: the vertices
    whose strongest bone is in the group, parts in name order."""
    out = {}
    for group, members in spec.contact_groups.items():
        per = []
        for part in sorted(rest.shells):
            names = np.array(rest.vertex_groups[part])
            keep = np.isin(names[rest.labels[part]], list(members))
            per.append(shells[part][:, keep])
        out[group] = np.concatenate(per, axis=1)
    return out


def stress_scene(ctx: SceneCtx, rig_key: str, *, partial: bool = False
                 ) -> tuple[SceneCtx, list[StressPose]]:
    """The scene with its clips replaced by one synthetic clip, `STRESS_CLIP`, whose samples are
    the rig's stress poses, and the poses (sample i is pose i).

    Each sample holds the body's shells (`core.pose.skin`), the posed bone matrices (`b:`, from
    `pose_world`) and the contact groups (`g:`). The clip's posture is `free`. `partial` is
    `stress_poses`'. A rig the scene holds no rest file for, a part with no skin weights, or a
    bending joint with no range raises `Unmeasurable`; no pose to take raises it too."""
    pack, _, name = rig_key.partition(":")
    piece = ctx.packs[pack].pieces[name] if pack in ctx.packs else None
    if piece is None or piece.rig is None or rig_key not in ctx.rigs:
        raise ManifestError(f"rig {rig_key!r} is no rigged piece of the loaded scene")
    spec, rig = piece.rig, ctx.rigs[rig_key]
    rest = ctx.rests.get(rig_key)
    if rest is None:
        raise Unmeasurable(f"the scene holds no rest file for rig {rig_key!r}, so it cannot be "
                           "posed")
    poses = stress_poses(spec, partial=partial)
    if not poses:
        raise Unmeasurable(f"no bending joint of rig {rig_key!r} has a range, so there is no "
                           "pose to take")
    worlds = [pose_world(rig, p.pose) for p in poses]
    shells = {part: np.empty((len(poses), pts.shape[1], 3), np.float32)
              for part, pts in rest.shells.items()}
    for i, world in enumerate(worlds):
        for part, points in skin(rest, rig, world).items():
            shells[part][i] = points
    samples = ClipSamples(
        clip=STRESS_CLIP, times=np.arange(1.0, len(poses) + 1.0),
        groups=_groups(spec, rest, shells),
        bones={b: np.stack([w[b] for w in worlds]) for b in rig.order}, props={},
        shells=shells, faces=dict(rest.faces), labels=dict(rest.labels))
    clip = Clip(STRESS_CLIP, (1, len(poses)), FPS, "keyed", None, (), (), FloorGear({}), {}, None,
                posture="free")
    return (dataclasses.replace(ctx, clips={STRESS_CLIP: clip}, samples={STRESS_CLIP: samples},
                                rig_of_clip={STRESS_CLIP: rig_key}), poses)
