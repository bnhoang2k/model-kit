"""Check solver-error: at each key of a solved clip, the effector (or the tool tip) lies within
`SOLVE_ERR_OF_EFFECTOR` times the effector radius of its target.

For each target of a `solved` clip, the target resolves to a world point through `targets.py`
(its ref plus its `offset`). The measured point is read from the samples at the sample nearest the
target's `key`, the same pose the clip plays:

- the effector bone's matrix carries `effector_mid` (the same point the solver measures);
- with a `tool`, the measured point is the tool's tip (`targets.tool_tip`, the one lookup the reach check
  shares): the prop's sampled pose carries the point `tip_length` (the piece's measured field)
  along the piece's +Z from its origin, the convention stick pieces follow. The
  pose's scale converts the length, as the reach check converts it.

A target that a clip hold marks `dangle` (same ref, same effector group, covering the key) is
measured against that dangle's `budget` instead of the solver tolerance: the budget is how far
the effector may hang from the target.

The scene folder holds no solve inputs, so the plant moves a sampled effector bone (and a tool's
pose with it) at the key's sample, away from the target, by `CALIBRATION_FAIL` times the limit. A
target that already fails is skipped by the gate (`run` reports it). The known fault, a solve capped at
one iteration, is proven caught by `test_one_iteration_solve_fails_the_measure`.

A keyed clip has no key targets: a plain not-applicable line. Missing data is `Unmeasured`: a clip
with no samples or no rig, a target whose effector bone is not sampled, a tool with no sampled
pose or no `tip_length`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.entries import unit
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Clip, Target
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx, rig_spec
from model_kit.core.targets import UP, dangle_hold, resolve_target, target_item, tool_tip
from model_kit.core.tolerances import CALIBRATION_FAIL, SOLVE_ERR_OF_EFFECTOR

NAME = "solver-error"


@dataclass
class _Job:
    clip: Clip
    target: Target
    item: str
    sample: int  # index of the key's sample
    aim: np.ndarray  # (3,) the target in the world
    here: np.ndarray  # (3,) the measured point in the world
    limit: float
    dangle: bool  # the limit is a dangle's budget

    @property
    def miss(self) -> float:
        return float(np.linalg.norm(self.here - self.aim))


def _jobs(ctx: SceneCtx, only: Collection[str] | None = None) -> Iterator[_Job | str]:
    """Each measurable target as a job, or a coverage line naming one that is skipped. With
    `only`, just the targets whose items it names."""
    for clip in ctx.clips.values():
        if clip.authoring == "keyed":
            yield f"{NAME}: not applicable (keyed clip, no key targets): {clip.name}"
            continue
        samples = ctx.samples.get(clip.name)
        if samples is None:
            yield Unmeasured(f"{NAME}: skipped (no samples): {clip.name}")
            continue
        key = ctx.rig_of_clip.get(clip.name)
        rig = None if key is None else ctx.rigs.get(key)
        if rig is None:
            yield Unmeasured(f"{NAME}: skipped (no rig): {clip.name}")
            continue
        spec = rig_spec(ctx, clip.name)
        for index, target in enumerate(clip.targets):
            item = target_item(clip, index, target)
            if only is not None and item not in only:
                continue
            chain = spec.chains.get(target.chain)
            if chain is None:
                raise ManifestError(f"{item}: the rig has no chain {target.chain!r}; "
                                    f"it has {sorted(spec.chains)}")
            ref = resolve_target(ctx, clip.name, target)
            bone = samples.bones.get(chain.effector)
            mid = rig.effector_mid.get(chain.effector)
            if bone is None or mid is None:
                yield Unmeasured(f"{NAME}: skipped (no sampled effector {chain.effector}): "
                                 f"{item}")
                continue
            if target.tool is None:
                matrix = np.asarray(bone[ref.sample], dtype=float)
                here = matrix[:3, :3] @ np.asarray(mid, dtype=float) + matrix[:3, 3]
            else:
                tip = tool_tip(ctx, clip, item, target.tool, ref.sample)
                if isinstance(tip, str):
                    yield Unmeasured(f"{NAME}: skipped ({tip}): {item}")
                    continue
                here = tip.point
            hold = dangle_hold(clip, target, chain)
            yield _Job(clip, target, item, ref.sample, ref.with_offset(target.offset), here,
                       hold.dangle.budget if hold is not None
                       else SOLVE_ERR_OF_EFFECTOR * chain.effector_radius,
                       hold is not None)


def _plant(ctx: SceneCtx, job: _Job) -> PlantUnit:
    """Move the effector bone (and a tool's pose) at the key's sample `CALIBRATION_FAIL` limits
    further from the target, along the line from the target through where it is."""
    away = job.here - job.aim
    size = float(np.linalg.norm(away))
    shift = (away / size if size > 0 else UP) * CALIBRATION_FAIL * job.limit
    chain = rig_spec(ctx, job.clip.name).chains[job.target.chain]
    samples = ctx.samples[job.clip.name]
    bones = np.array(samples.bones[chain.effector], dtype=float)
    bones[job.sample, :3, 3] += shift
    fields = {"bones": {**samples.bones, chain.effector: bones}}
    if job.target.tool is not None:
        pose = np.array(samples.props[job.target.tool], dtype=float)
        pose[job.sample, :3, 3] += shift
        fields["props"] = {**samples.props, job.target.tool: pose}
    moved = dataclasses.replace(samples, **fields)
    return unit(job.item, [("off its target", dataclasses.replace(
        ctx, samples={**ctx.samples, job.clip.name: moved}), {job.sample})])


class SolverError:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        # One unit per measurable target, whatever its miss: the gate skips a target that `run`
        # already reports, and the shift moves the effector further from the aim either way.
        return [_plant(ctx, job) for job in _jobs(ctx) if isinstance(job, _Job)]

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # One sample per target, the key's: `samples` is accepted and ignored.
        return [Finding(NAME, job.item,
                        f"{job.miss:.4f} from its target, over the allowed {job.limit:.4f}"
                        + (" (the dangle's budget)" if job.dangle else "")
                        + f" at key {job.target.key}")
                for job in _jobs(ctx, only) if isinstance(job, _Job) and job.miss > job.limit]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return [line for line in _jobs(ctx) if isinstance(line, str)]
