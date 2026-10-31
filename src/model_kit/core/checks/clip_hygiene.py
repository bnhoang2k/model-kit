"""Check clip-hygiene: five rules on a clip's samples.

Items are `<clip> floor`, `<clip> slide <support>`, `<clip> stand <support>`, `<clip> pop` and
`<clip> prop <prop id>`, so one allow line covers a whole clip and rule. A finding carries the worst
excess over the clip. The body's feet are the rig's `supports` (`core.supports`), never a guess from
the top bone. A support has a radius `r` (a chain's effector radius, or its own footprint: see
`core.supports`). The floor rule and the pop rule use `base`, the smallest radius over the rig's
chains and supports (the strictest, as pose overlap takes it for its skin); slide and stand use the
support's own. The length factors are reference lengths over the mitt radius, so on the
reference rig they give those lengths back.

- **floor**: no body vertex of any sampled part under y 0 of the clip's world, less
  `FLOOR_OF_EFFECTOR` times `base`.
- **slide**: a clip's supports do not slide while it stands. A standing sample is one outside the
  clip's seat windows (`seat_windows`: the `seat_when` windows, the whole clip when there are
  none); a clip with no `seat` stands at every sample. Only pairs of consecutive samples that both
  stand are measured. Per sample the support's lowest point (by y) is followed; a support is
  planted at a standing sample when that point is within `FLOOR_OF_EFFECTOR` times its `r`.
  Between two consecutive planted samples the lowest point may move at most `SLIDE_OF_EFFECTOR`
  times `r` across the floor (x and z).
- **stand**: gravity loads a standing body's supports, so one off the ground is held up by effort
  and lifts only on purpose. In a `stand` clip, at each sample
  outside the support's `lifts` windows (a step, a hop, with a why), the support's lowest point is
  at most `SKIN_OF_EFFECTOR` times its `r` above y 0: a loaded sole touches the floor, and the only
  slack is mesh noise, the overlap check's skin budget. The planted-foot band of the slide rule
  would miss a 0.006-unit lift (about 8 mm at 0.747 units per metre).
- **pop**: a `loop` clip's last sample matches the sample at `loop_from` (the first, with none)
  within `POP_OF_EFFECTOR` times `base`, at every body vertex.
- **prop**: the clip uses a carried prop only while it is shown, inside its `show_frames`
  windows. The samples hold no show flag (a player may hide a prop by visibility, and the kit
  writes every prop pose at every sample), so the rule reads what the clip asks of the prop. A
  sample uses the prop when a clip hold names it (its `ref` is on the prop, or its `who` is the
  prop), a target names it (its `ref` is on the prop, or it is the target's `tool`): a hold at
  the samples of its `when` windows, a target at the sample nearest its `key`. Every using sample
  must lie in a show window. Being shown while unused is not a finding.

The floor, slide and pop rules are judged only as they will be seen (a design call of this kit),
by the `view` of the rig's piece. A finding whose measured size (the depth under y 0, the travel
of the planted foot, the jump of the loop) is smaller than one arcminute at `view.nearest_m`
(`sight.seen_floor`) is dropped and counted in a coverage line per piece. The limits do not
move, and the stand and prop rules are not touched: stand is gravity and prop is a flag, neither a
length the eye measures. The rules read no faces, so a side the view lists as `hidden` changes
nothing here. A piece with no `view` is judged as before, with no floor and no line.

Plants, one unit per measured item, built from the clip's samples and manifest alone (never from a
rule's verdict): sink one body vertex `CALIBRATION_FAIL` floor tolerances under y 0; slide one
planted support sample by `CALIBRATION_FAIL` slide tolerances (away from the slide it already has);
lift one support by `CALIBRATION_FAIL` times its stand band at its first judged sample (a support
sunk too far for that is set that high above the floor instead); move one vertex of the last sample
`CALIBRATION_FAIL` pop tolerances from its start (away from where it is); narrow the prop's show
windows so one using sample falls outside (or drop them all). A unit holds the faults to try, at
the middle, first and last sample, and the gate keeps the first the rule catches. Under a view the
floor, slide and pop plants move by `CALIBRATION_FAIL` times the larger of the tolerance and the
seen floor, so they stay visible. An item that already fails is skipped by the gate. A fault no
candidate makes fail stops the run (`BlindCheck`).

Coverage. A rule with nothing to measure in a clip is a plain `not applicable` line: no loop, no
props, a clip seated at every sample for the slide rule, a rig that names no supports, a support
never planted in two consecutive samples, a clip whose posture is not `stand` (stand), a support
lifted at every sample, a prop no hold, target or tool uses, a loop that starts at its last
sample (its seam has no jump). Missing data is `Unmeasured`: a clip
with no samples or no rig, no body shells (floor, pop), no radius to measure by (no chain and no
support to size it), a support with no sampled points, and, for the stand rule, a clip with no
posture or a `stand` clip on a rig that names no supports.

The check measures clips, not a recipe, so `kinds` is empty.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from model_kit.core.checks.entries import first_middle_last, unit
from model_kit.core.checks.pieces import Sight, sight_of, under_floor_line
from model_kit.core.errors import Unmeasurable
from model_kit.core.manifest import Clip, Ref, RigSpec, anchor, seat_windows
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import ClipSamples, RigRest, SceneCtx, due, rig_spec
from model_kit.core.supports import body_radius, support_radius
from model_kit.core.targets import sample_at
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    FLOOR_OF_EFFECTOR,
    POP_OF_EFFECTOR,
    SKIN_OF_EFFECTOR,
    SLIDE_OF_EFFECTOR,
)

NAME = "clip-hygiene"
RIGHT = np.array([1.0, 0.0, 0.0])  # the way a plant moves when there is no better direction


@dataclass(eq=False)
class _Clip:
    ctx: SceneCtx
    clip: Clip
    samples: ClipSamples
    spec: RigSpec
    # The smallest radius over the rig's chains and supports, None for a rig with neither.
    base: float | None
    radii: dict[str, float]  # support -> its own radius
    unsized: dict[str, str]  # support -> why it has no radius (a support with no footprint)
    # How the rig's piece will be seen, or None when it declares no `view`. It is shared by every
    # clip of the rig, and keeps the tally of what the floor dropped.
    sight: Sight | None = None

    @property
    def floor(self) -> float:
        return FLOOR_OF_EFFECTOR * self.base

    def planted_within(self, group: str) -> float:
        """How high a support's lowest point may be and count as planted: its own floor limit."""
        return FLOOR_OF_EFFECTOR * self.radii[group]

    def band(self, group: str) -> float:
        """How high a standing support's lowest point may be: the skin budget of its own radius."""
        return SKIN_OF_EFFECTOR * self.radii[group]

    def lifted(self, group: str) -> np.ndarray:
        """Per sample: True inside a window where the clip lets this support leave the floor."""
        mask = np.zeros(len(self.samples.times), dtype=bool)
        for lift in self.clip.lifts:
            if lift.support == group:
                mask[due(self.samples.times, lift.when)] = True
        return mask

    def standing(self) -> np.ndarray:
        """Per sample: True when the clip stands there, outside its seat windows."""
        mask = np.ones(len(self.samples.times), dtype=bool)
        if self.clip.seat is not None:
            mask[due(self.samples.times, seat_windows(self.clip))] = False
        return mask

    def body(self) -> np.ndarray:
        """Every body vertex of every sample: (S, V, 3)."""
        return np.concatenate([np.asarray(a, dtype=float) for a in self.samples.shells.values()],
                              axis=1)


def _sight(ctx: SceneCtx, key: str) -> Sight | None:
    """The sight of the piece that a rig key names: its floor, from its `view`."""
    pack_name, _, name = key.partition(":")
    pack = ctx.packs[pack_name]
    return sight_of(pack.pieces[name], pack.units_per_metre)


def _prepare(ctx: SceneCtx, clip: Clip, sights: dict[str, Sight | None]) -> _Clip | Unmeasured:
    samples = ctx.samples.get(clip.name)
    if samples is None:
        return Unmeasured(f"{NAME}: skipped (no samples): {clip.name}")
    if clip.name not in ctx.rig_of_clip:
        return Unmeasured(f"{NAME}: skipped (no rig): {clip.name}")
    spec = rig_spec(ctx, clip.name)
    rest: RigRest | None = ctx.rests.get(ctx.rig_of_clip[clip.name])
    radii: dict[str, float] = {}
    unsized: dict[str, str] = {}
    for group in spec.supports:
        try:
            radii[group] = support_radius(spec, rest, group)
        except Unmeasurable as err:
            unsized[group] = str(err)
    key = ctx.rig_of_clip[clip.name]
    if key not in sights:
        sights[key] = _sight(ctx, key)
    return _Clip(ctx, clip, samples, spec, body_radius(spec, rest, radii), radii, unsized,
                 sights[key])


def _ids(c: _Clip, samples: frozenset[int]) -> list[int]:
    return [s for s in range(len(c.samples.times)) if not samples or s in samples]


def _seen(c: _Clip, item: str, size: float) -> bool:
    """Whether a finding of this measured size is one a viewer could see. Under a `view`, a size
    smaller than one arcminute at its nearest distance is dropped and tallied; with no `view`,
    every finding is kept."""
    if c.sight is None:
        return True
    if size < c.sight.floor:
        c.sight.under.add(item)
        return False
    c.sight.shown.add(item)
    return True


def _push(c: _Clip, tolerance: float) -> float:
    """How far a plant moves a body to break a rule: `CALIBRATION_FAIL` times the larger of the
    rule's tolerance and the seen floor, so the fault is a finding under a view too."""
    return CALIBRATION_FAIL * max(tolerance, 0.0 if c.sight is None else c.sight.floor)


# --- the rules: what each measures -----------------------------------------------------------------
# A rule reads a prepared clip and returns its worst finding for one item (or None). Its `status`
# says first whether the item can be measured at all: a coverage line, or None when it can.


def _floor_status(c: _Clip) -> str | None:
    item = f"{c.clip.name} floor"
    if not c.samples.shells:
        return Unmeasured(f"{NAME}: skipped (no body shells): {item}")
    if c.base is None:
        return Unmeasured(f"{NAME}: skipped (no radius to measure by): {item}")
    return None


def _floor(c: _Clip, ids: list[int]) -> Finding | None:
    body = c.body()
    low = body[:, :, 1].min(axis=1)
    worst = None
    for s in ids:
        if -low[s] > c.floor and (worst is None or low[s] < low[worst]):
            worst = s
    if worst is None or not _seen(c, f"{c.clip.name} floor", float(-low[worst])):
        return None
    return Finding(NAME, f"{c.clip.name} floor",
                   f"lowest body vertex y {low[worst]:.4f}, under the floor limit "
                   f"{-c.floor:.4f} at frame {c.samples.times[worst]:g}")


def _planted(c: _Clip, group: str) -> tuple[np.ndarray, np.ndarray]:
    """(the lowest point of the group at each sample, whether it is planted and standing there)."""
    pts = np.asarray(c.samples.groups[group], dtype=float)
    low = pts[np.arange(len(pts)), np.argmin(pts[:, :, 1], axis=1)]
    return low, (low[:, 1] <= c.planted_within(group)) & c.standing()


def _pairs(planted: np.ndarray) -> list[int]:
    """The samples s whose pair (s - 1, s) has the foot planted (and standing) at both."""
    return [s for s in range(1, len(planted)) if planted[s] and planted[s - 1]]


def _slide(c: _Clip, group: str, ids: list[int]) -> Finding | None:
    low, planted = _planted(c, group)
    tol = SLIDE_OF_EFFECTOR * c.radii[group]
    keep = set(ids)
    worst = None
    for s in _pairs(planted):
        if s not in keep and s - 1 not in keep:
            continue
        d = float(np.hypot(*(low[s] - low[s - 1])[[0, 2]]))
        if d > tol and (worst is None or d > worst[0]):
            worst = (d, s)
    if worst is None or not _seen(c, f"{c.clip.name} slide {group}", worst[0]):
        return None
    d, s = worst
    t = c.samples.times
    return Finding(NAME, f"{c.clip.name} slide {group}",
                   f"planted support moved {d:.4f} between frames {t[s - 1]:g} and {t[s]:g}, "
                   f"over the allowed {tol:.4f}")


def _judged(c: _Clip, group: str) -> list[int]:
    """The samples the stand rule judges for a support: those outside its lift windows."""
    return np.nonzero(~c.lifted(group))[0].tolist()


def _stand(c: _Clip, group: str, ids: list[int]) -> Finding | None:
    low = np.asarray(c.samples.groups[group], dtype=float)[:, :, 1].min(axis=1)
    band = c.band(group)
    keep = set(ids)
    worst = None
    for s in _judged(c, group):
        if s in keep and low[s] > band and (worst is None or low[s] > low[worst]):
            worst = s
    if worst is None:
        return None
    return Finding(NAME, f"{c.clip.name} stand {group}",
                   f"lowest point of {group} is {low[worst]:.4f} above the floor at frame "
                   f"{c.samples.times[worst]:g}, over the allowed {band:.4f}, and no lift names "
                   "that frame")


def _loop_sample(c: _Clip) -> int:
    return sample_at(c.samples, c.clip.frames[0] if c.clip.loop_from is None else c.clip.loop_from)


def _pop(c: _Clip, ids: list[int]) -> Finding | None:
    last, first = len(c.samples.times) - 1, _loop_sample(c)
    if last not in ids and first not in ids:
        return None
    body = c.body()
    d = float(np.linalg.norm(body[last] - body[first], axis=1).max())
    tol = POP_OF_EFFECTOR * c.base
    if d <= tol or not _seen(c, f"{c.clip.name} pop", d):
        return None
    t = c.samples.times
    start = "its first sample" if c.clip.loop_from is None else f"loop_from {c.clip.loop_from}"
    return Finding(NAME, f"{c.clip.name} pop",
                   f"the last sample (frame {t[last]:g}) is {d:.4f} from {start} (frame "
                   f"{t[first]:g}), over the allowed {tol:.4f}")


def _on_prop(c: _Clip, ref: Ref, prop_id: str) -> bool:
    return (not ref.is_bone
            and anchor(c.ctx.packs, ref, clip=c.clip).placed_by == f"prop:{prop_id}")


def _uses(c: _Clip, prop_id: str) -> list[int]:
    """The samples where a hold, a target or a tool of the clip names the prop."""
    used: set[int] = set()
    for hold in c.clip.holds:
        if hold.who == prop_id or _on_prop(c, hold.ref, prop_id):
            used |= set(due(c.samples.times, hold.when).tolist())
    for target in c.clip.targets:
        if target.tool == prop_id or _on_prop(c, target.ref, prop_id):
            used.add(sample_at(c.samples, target.key))
    return sorted(used)


def _outside(c: _Clip, prop_id: str, ids: list[int] | None = None) -> list[int]:
    """The using samples (among `ids`, when given) outside every one of the prop's windows."""
    inside = set(due(c.samples.times, c.clip.props[prop_id].show_frames).tolist())
    return [s for s in _uses(c, prop_id) if s not in inside and (ids is None or s in ids)]


def _prop(c: _Clip, prop_id: str, ids: list[int]) -> Finding | None:
    bad = _outside(c, prop_id, ids)
    if not bad:
        return None
    windows = [list(w) for w in c.clip.props[prop_id].show_frames]
    return Finding(NAME, f"{c.clip.name} prop {prop_id}",
                   f"the clip uses it at frame {c.samples.times[bad[0]]:g}, outside its show "
                   f"windows {windows}: {len(bad)} sample{'s' if len(bad) > 1 else ''} outside")


# --- what a clip has to measure ---------------------------------------------------------------------


@dataclass
class _Work:
    """The items measurable in one clip and the coverage lines for the rest."""

    floor: bool = False
    slides: list[str] = dataclasses.field(default_factory=list)  # supports
    stands: list[str] = dataclasses.field(default_factory=list)  # supports
    pop: bool = False
    props: list[str] = dataclasses.field(default_factory=list)
    lines: list[str] = dataclasses.field(default_factory=list)


def _support_status(c: _Clip, group: str, item: str) -> Unmeasured | None:
    """The coverage line for a support that cannot be measured (no radius, no points), or None."""
    if group in c.unsized:
        return Unmeasured(f"{NAME}: skipped (no radius for support {group}: "
                          f"{c.unsized[group]}): {item}")
    if group not in c.samples.groups:
        return Unmeasured(f"{NAME}: skipped (no points for group {group}): {item}")
    return None


def _work(c: _Clip) -> _Work:
    w = _Work()
    name = c.clip.name

    def n_a(why: str, item: str) -> None:
        w.lines.append(f"{NAME}: not applicable ({why}): {item}")

    if (line := _floor_status(c)) is not None:
        w.lines.append(line)
    else:
        w.floor = True
    # slide
    supports = c.spec.supports
    if not c.standing().any():
        n_a("seated clip: it is seated at every sample", f"{name} slide")
    elif not supports:
        n_a("the rig names no supports", f"{name} slide")
    else:
        for group in supports:
            item = f"{name} slide {group}"
            if (line := _support_status(c, group, item)) is not None:
                w.lines.append(line)
            elif not _pairs(_planted(c, group)[1]):
                n_a("the support is never planted in two samples in a row", item)
            else:
                w.slides.append(group)
    # stand
    posture = c.clip.posture
    if posture is None:
        w.lines.append(Unmeasured(f"{NAME}: skipped (the clip has no posture): {name} stand"))
    elif posture != "stand":
        n_a(f"{posture} clip: only a standing body's supports bear its weight", f"{name} stand")
    elif not supports:
        w.lines.append(Unmeasured(f"{NAME}: skipped (the rig names no supports): {name} stand"))
    else:
        for group in supports:
            item = f"{name} stand {group}"
            if (line := _support_status(c, group, item)) is not None:
                w.lines.append(line)
            elif not _judged(c, group):
                n_a("the support is inside a lift window at every sample", item)
            else:
                w.stands.append(group)
    # pop
    if not c.clip.loop:
        n_a("no loop", f"{name} pop")
    elif not c.samples.shells:
        w.lines.append(Unmeasured(f"{NAME}: skipped (no body shells): {name} pop"))
    elif c.base is None:
        w.lines.append(Unmeasured(f"{NAME}: skipped (no radius to measure by): {name} pop"))
    elif _loop_sample(c) == len(c.samples.times) - 1:
        n_a("the loop starts at the last sample, so its seam has no jump", f"{name} pop")
    else:
        w.pop = True
    # props
    if not c.clip.props:
        n_a("no props", f"{name} prop")
    for prop_id in c.clip.props:
        if _uses(c, prop_id):
            w.props.append(prop_id)
        else:
            n_a("not used by a hold, target or tool", f"{name} prop {prop_id}")
    return w


def _findings(ctx: SceneCtx, only: Collection[str] | None, samples: frozenset[int],
              sights: dict[str, Sight | None] | None = None) -> list[Finding]:
    """The findings of every clip. `sights` collects, per rig piece, the sight that tallied what
    the seen floor dropped (a caller that wants the tally passes an empty dict)."""
    found: list[Finding] = []
    sights = {} if sights is None else sights
    for clip in ctx.clips.values():
        c = _prepare(ctx, clip, sights)
        if isinstance(c, Unmeasured):
            continue
        w = _work(c)
        ids = _ids(c, samples)

        def wanted(item: str) -> bool:
            return only is None or item in only

        results = []
        if w.floor and wanted(f"{clip.name} floor"):
            results.append(_floor(c, ids))
        for group in w.slides:
            if wanted(f"{clip.name} slide {group}"):
                results.append(_slide(c, group, ids))
        for group in w.stands:
            if wanted(f"{clip.name} stand {group}"):
                results.append(_stand(c, group, ids))
        if w.pop and wanted(f"{clip.name} pop"):
            results.append(_pop(c, ids))
        for prop_id in w.props:
            if wanted(f"{clip.name} prop {prop_id}"):
                results.append(_prop(c, prop_id, ids))
        found += [f for f in results if f is not None]
    return found


# --- plants -----------------------------------------------------------------------------------------
# A plant is built from the clip's samples and manifest alone, never from a rule's verdict: each
# unit names its item and the faults to try, and the gate keeps the first one the rule catches.


def _with(ctx: SceneCtx, clip: str, **fields) -> SceneCtx:
    moved = dataclasses.replace(ctx.samples[clip], **fields)
    return dataclasses.replace(ctx, samples={**ctx.samples, clip: moved})


def _direction(away: np.ndarray) -> np.ndarray:
    size = float(np.linalg.norm(away))
    return away / size if size > 0 else RIGHT


def _offset(c: _Clip, away: np.ndarray, factor: float, radius: float) -> np.ndarray:
    """How far a plant moves a body along the unit `away` to break a rule whose tolerance is
    `factor` times `radius`: `CALIBRATION_FAIL` tolerances, or `CALIBRATION_FAIL` seen floors when
    the floor is the larger, so the fault is a finding under a view too. With no view, or a
    tolerance past the floor, the product keeps its original order (`away * C * factor * radius`),
    so the bits of the plant do not depend on the view code."""
    if c.sight is None or factor * radius >= c.sight.floor:
        return away * CALIBRATION_FAIL * factor * radius
    return away * (CALIBRATION_FAIL * c.sight.floor)



def _body_part(c: _Clip) -> str:
    """The first body part that holds a vertex."""
    return next(part for part, shell in c.samples.shells.items() if np.asarray(shell).shape[1])


def _floor_unit(c: _Clip) -> PlantUnit:
    """Sink the first vertex of the body `CALIBRATION_FAIL` floor tolerances under y 0, at the
    middle sample, then the first, then the last."""
    part = _body_part(c)
    item = f"{c.clip.name} floor"
    n = len(c.samples.times)
    cands = []
    for s in dict.fromkeys([n // 2, 0, n - 1]):
        sunk = np.array(c.samples.shells[part])
        sunk[s, 0, 1] = -_push(c, c.floor)
        cands.append(("under the floor",
                      _with(c.ctx, c.clip.name, shells={**c.samples.shells, part: sunk}), {s}))
    return unit(item, cands)


def _slide_unit(c: _Clip, group: str) -> PlantUnit:
    """Move the planted support away from the slide it already has, at the first, the middle and
    the last sample that completes a planted pair."""
    low, planted = _planted(c, group)
    cands = []
    for s in first_middle_last(_pairs(planted)):
        step = low[s] - low[s - 1]
        away = _direction(np.array([step[0], 0.0, step[2]]))
        pts = np.array(c.samples.groups[group], dtype=float)
        pts[s] += _offset(c, away, SLIDE_OF_EFFECTOR, c.radii[group])
        cands.append(("slides", _with(c.ctx, c.clip.name, groups={**c.samples.groups, group: pts}),
                      {s}))
    return unit(f"{c.clip.name} slide {group}", cands)


def _stand_unit(c: _Clip, group: str) -> PlantUnit:
    """Lift the support `CALIBRATION_FAIL` bands at the first sample the rule judges; then, for a
    support sunk so far there that this leaves it under the band, set its lowest point
    `CALIBRATION_FAIL` bands above the floor, at the first, middle and last judged sample."""
    judged = _judged(c, group)
    band = c.band(group)
    cands = []
    for s, absolute in [(judged[0], False)] + [(s, True) for s in first_middle_last(judged)]:
        pts = np.array(c.samples.groups[group], dtype=float)
        pts[s, :, 1] += (CALIBRATION_FAIL * band - pts[s, :, 1].min()) if absolute \
            else CALIBRATION_FAIL * band
        cands.append(("hovers",
                      _with(c.ctx, c.clip.name, groups={**c.samples.groups, group: pts}), {s}))
    return unit(f"{c.clip.name} stand {group}", cands)


def _pop_unit(c: _Clip) -> PlantUnit:
    """Move the first vertex of the body at the last sample `CALIBRATION_FAIL` pop tolerances away
    from where it is at the loop's start."""
    last, first = len(c.samples.times) - 1, _loop_sample(c)
    part = _body_part(c)
    moved = np.array(c.samples.shells[part])
    away = _direction(moved[last, 0].astype(float) - moved[first, 0].astype(float))
    moved[last, 0] = moved[last, 0] + _offset(c, away, POP_OF_EFFECTOR, c.base)
    return unit(f"{c.clip.name} pop",
                [("pops", _with(c.ctx, c.clip.name, shells={**c.samples.shells, part: moved}),
                  {last})])


def _unshown_instant(c: _Clip) -> float | None:
    """A frame the clip's manifest can name (inside its frames) at which no sample is due, or
    None: the middle of two neighbouring samples, else an end of the clip. A one-frame window
    there hides the prop at every sample."""
    times = c.samples.times
    gaps = [float((a + b) / 2) for a, b in pairwise(times)]
    for t in [*gaps, float(c.clip.frames[0]), float(c.clip.frames[1])]:
        if not len(due(times, ((t, t),))):
            return t
    return None


def _prop_unit(c: _Clip, prop_id: str) -> PlantUnit:
    """Narrow the show windows to the samples before the first using one (or, when that is the
    first sample, to the samples after it), so that using sample falls outside; then show the prop
    only at a frame no sample reaches, which no slack in the sample times can bring back. Each is
    a state the manifest loader accepts: a window is forward, inside the clip's frames and never
    empty."""
    first = _uses(c, prop_id)[0]
    times, frames = c.samples.times, c.clip.frames
    shows = []
    if first > 0:
        shows.append(((frames[0], float(times[first - 1])),))
    elif len(times) > 1:
        shows.append(((float(times[1]), frames[1]),))
    if (hidden := _unshown_instant(c)) is not None:
        shows.append(((hidden, hidden),))
    cands = []
    for show in shows:
        prop = dataclasses.replace(c.clip.props[prop_id], show_frames=show)
        clip = dataclasses.replace(c.clip, props={**c.clip.props, prop_id: prop})
        cands.append(("used while hidden",
                      dataclasses.replace(c.ctx, clips={**c.ctx.clips, clip.name: clip})))
    return unit(f"{c.clip.name} prop {prop_id}", cands)


class ClipHygiene:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units: list[PlantUnit] = []
        sights: dict[str, Sight | None] = {}
        for clip in ctx.clips.values():
            c = _prepare(ctx, clip, sights)
            if isinstance(c, Unmeasured):
                continue
            w = _work(c)
            if w.floor:
                units.append(_floor_unit(c))
            units += [_slide_unit(c, g) for g in w.slides]
            units += [_stand_unit(c, g) for g in w.stands]
            if w.pop:
                units.append(_pop_unit(c))
            units += [_prop_unit(c, p) for p in w.props]
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        return _findings(ctx, only, samples)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        sights: dict[str, Sight | None] = {}
        for clip in ctx.clips.values():
            c = _prepare(ctx, clip, sights)
            lines += [c] if isinstance(c, Unmeasured) else _work(c).lines
        return list(dict.fromkeys(lines + self._floor_lines(ctx)))

    def _floor_lines(self, ctx: SceneCtx) -> list[str]:
        """The coverage line of the seen floor for each rig piece with a `view`: how many floor,
        slide and pop findings it dropped for being smaller than the eye resolves. One unscoped
        pass over every clip. None for a piece with no view."""
        sights: dict[str, Sight | None] = {}
        _findings(ctx, None, frozenset(), sights)
        lines: list[str] = []
        for key, sight in sights.items():
            if sight is not None:
                pack_name, _, name = key.partition(":")
                piece = ctx.packs[pack_name].pieces[name]
                lines.append(under_floor_line(NAME, sight, piece, key))
        return lines
