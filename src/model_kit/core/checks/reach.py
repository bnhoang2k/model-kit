"""Check reach, pre-solve reach: every solved clip target lies within its chain's reach.

For each target of a `solved` clip, the reach origin is the head of the chain's first bone
(the shoulder or hip), read from the posed bones at the sample nearest the target's `key`. The
target resolves to a world point through `targets.py` (its ref plus its `offset`, the offset in
the ref's frame, rotation only). The point may lie no further from that head than the chain's
reach, plus the tool's `tip_length` when the target names a tool, minus
`REACH_MARGIN_OF_EFFECTOR` of the effector radius. `tip_length` is in the tool's own units, and
the tool's sampled pose may scale it (a carried prop takes the armature's scale), so it is
converted to world units through `targets.length_in_world` at the key's sample, along the line
from the tool's origin to the target (exact under a uniform scale).

A target is excused, and listed in coverage, when a clip hold on the same ref is a dangle whose
`who` is the target chain's effector group and whose `when` covers the target's key. A keyed
clip and a clip with no samples are skipped and listed. A keyed clip is not applicable by design;
a clip with no samples, or a target whose tool has no sampled pose or no `tip_length`
(`targets.tool_tip`, shared with the solver-error check), could not be measured, so it makes the run partial
(`metagate.Unmeasured`).

One plant unit per checked target: the target moves along the first-bone-to-target line to
`CALIBRATION_FAIL` margins past the allowed reach. The unit is built from the target's reach and
position, never from whether it passes; the gate skips a target that already fails.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION
from model_kit.core.manifest import Chain, Clip, Target
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.rig import chain_root, reach
from model_kit.core.scene import SceneCtx, rig_spec
from model_kit.core.targets import (
    UP,
    Resolved,
    ToolTip,
    dangle_hold,
    length_in_world,
    resolve_target,
    target_item,
    tool_tip,
)
from model_kit.core.tolerances import CALIBRATION_FAIL, REACH_MARGIN_OF_EFFECTOR

NAME = "reach"


@dataclass
class _Job:
    clip: Clip
    index: int
    target: Target
    item: str
    first_bone: np.ndarray  # head of the chain's first bone, world, at the sample nearest the key
    ref: Resolved  # the target's ref in the world, before the offset
    allowed: float  # reach plus tool tip
    margin: float  # REACH_MARGIN_OF_EFFECTOR times the effector radius

    @property
    def point(self) -> np.ndarray:
        return self.ref.with_offset(self.target.offset)

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.point - self.first_bone))

    @property
    def limit(self) -> float:
        return self.allowed - self.margin


def _tip_length(tip: ToolTip, ref: Resolved, offset) -> float:
    """The tool's `tip_length` in world units at the key's sample, along the line from the
    tool's origin to the target."""
    pose = tip.pose
    toward = np.linalg.solve(pose[:3, :3], ref.with_offset(offset) - pose[:3, 3])
    if np.linalg.norm(toward) < NO_DIRECTION:
        toward = UP
    return tip.length * length_in_world(pose[:3, :3], toward)


def _dangles(clip: Clip, target: Target, chain: Chain) -> bool:
    """A dangle hold on the target's ref, for this chain's effector, covering the key."""
    return dangle_hold(clip, target, chain) is not None


def _jobs(ctx: SceneCtx, only: Collection[str] | None = None) -> Iterator[_Job | str]:
    """Each checkable target as a job, or a coverage line naming one that is skipped.

    With `only`, just the targets whose items it names (a plant's scoped run).
    """
    for clip in ctx.clips.values():
        if clip.authoring == "keyed":
            yield f"{NAME}: not applicable (keyed): {clip.name}"
            continue
        samples = ctx.samples.get(clip.name)
        if samples is None:
            yield Unmeasured(f"{NAME}: skipped (no samples): {clip.name}")
            continue
        spec = rig_spec(ctx, clip.name)
        rig = ctx.rigs.get(ctx.rig_of_clip[clip.name])
        if rig is None:
            raise ManifestError(f"clip {clip.name!r}: rig {ctx.rig_of_clip[clip.name]!r} is "
                                "not in the scene's rigs")
        for index, target in enumerate(clip.targets):
            item = target_item(clip, index, target)
            if only is not None and item not in only:
                continue
            chain = spec.chains.get(target.chain)
            if chain is None:
                raise ManifestError(f"{item}: the rig has no chain {target.chain!r}; "
                                    f"it has {sorted(spec.chains)}")
            if _dangles(clip, target, chain):
                yield f"{NAME}: not applicable (dangle): {item}"
                continue
            ref = resolve_target(ctx, clip.name, target)
            extra = 0.0
            if target.tool is not None:
                tip = tool_tip(ctx, clip, item, target.tool, ref.sample)
                if isinstance(tip, str):
                    yield Unmeasured(f"{NAME}: skipped ({tip}): {item}")
                    continue
                extra = _tip_length(tip, ref, target.offset)
            yield _Job(
                clip=clip, index=index, target=target, item=item,
                first_bone=chain_root({b: m[ref.sample] for b, m in samples.bones.items()}, chain),
                ref=ref,
                allowed=reach(rig, chain) + extra,
                margin=REACH_MARGIN_OF_EFFECTOR * chain.effector_radius,
            )


def _plant(ctx: SceneCtx, job: _Job) -> PlantUnit:
    """Move the target `CALIBRATION_FAIL` margins past the allowed reach, along the
    first-bone-to-target line."""
    line = job.point - job.first_bone
    length = float(np.linalg.norm(line))
    # straight up when the target sits on the first bone's head
    direction = line / length if length > 0 else UP
    where = job.first_bone + direction * (job.allowed + CALIBRATION_FAIL * job.margin)
    offset = tuple(float(x) for x in job.ref.offset_to(where))
    moved = dataclasses.replace(job.target, offset=offset)
    targets = tuple(moved if i == job.index else t for i, t in enumerate(job.clip.targets))
    clip = dataclasses.replace(job.clip, targets=targets)
    planted = dataclasses.replace(ctx, clips={**ctx.clips, clip.name: clip})
    return PlantUnit(job.item, (Plant(f"too far: {job.item}", job.item, planted),))


class Reach:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [_plant(ctx, job) for job in _jobs(ctx) if isinstance(job, _Job)]

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [Finding(NAME, job.item,
                        f"{job.distance:.4f} from its chain's first bone, over the allowed "
                        f"{job.limit:.4f} (reach {job.allowed:.4f} less margin {job.margin:.4f})")
                for job in _jobs(ctx, only) if isinstance(job, _Job) and job.distance > job.limit]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return [line for line in _jobs(ctx) if isinstance(line, str)]
