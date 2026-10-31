"""Check bend-quality: a limb keeps its thickness when it bends.

Think of a garden hose. Bend it and the hose stays round and as wide as before; a bent limb does
the same, because flesh is mostly water and cannot be squeezed small. Linear blend skinning does
squeeze: an elbow pinches, and a twist shrinks one ring like the neck of a candy wrapper.

The check reads parts that are bound to more than one bone. For each bending joint that lies
inside such a part, `core.slices` finds `SLICES` rings on the skin, the lines where the weight of
the joint's child side is 1/8, 2/8 ... 7/8, at rest and at every pose, and measures each ring's
outline. The poses are every sample of every clip of the rig, and no others: a piece is judged
only as it will be seen, so the ends of a joint's range, which no clip reaches, are not judged.
A ring is the same band of skin at rest and posed, so it follows a limb round a sharp bend and
round an L-shaped joint, where a plane cut would run along the limb and measure the cut, not the
skin. Per ring, against its rest ring:

- **girth**: the perimeter changes by no more than `BEND_GIRTH_CHANGE`, the largest girth change
  the cited scan studies measured on real limbs through bending;
- **collapse**: the area stays at or over `BEND_AREA_FLOOR` of the rest area. The floor is a stated
  design call, not a measurement. It catches the candy-wrapper twist.

The shape ratio `4 pi A / P^2` (1 for a circle, toward 0 for a flat sliver) is printed in coverage
and never judged: no source bounds it.

Items: `<pack>:<piece> <part> <joint> slice <k> girth at <where>` and `... collapse at <where>`,
where slice k is the ring at weight k/8 and `<where>` is `<clip> frame <n>`.

What a finding means. A single-bone hinge under linear blend skinning loses girth as a fixed
function of its angle, whatever the weights: every point of a ring at child weight w moves by the
same blend (1 - w) I + w R, so the ring's girth depends on w and the angle alone. The halfway ring
of a hinge passes the 6.4% limit at about 60 degrees and loses about 11.3% at 80 degrees
(more bands or vertices change nothing). So a finding at a clip sample past
that angle is not cured by better weights; only a split joint is (helper or twist bones, the turn
shared by several sub-joints). The 6.4% limit comes from one knee study, Ringenbach, Yu and Zhang
2024. Split joints are the open option.

A part bound to one bone does not bend and prints a not-applicable line. A part with no skin
weights and a ring that closes nowhere are `Unmeasured`.

A part blended across a joint with no vertex wholly on either side is measured too; a ring
it cannot close is `Unmeasured`. A part with a vertex that carries no skin weight is `Unmeasured`
(its share across a joint is undefined).

Plants, per bending joint inside a part and kind, so the elbow and the knee are planted as much as
the chest: a band of the skin shrunk toward the bone axis by half at one clip sample (girth); a band
turned end for end about the axis, or squeezed by half when the weights leave the ring little
turned (collapse). A plant is a copy of the scene with one sample's
shell edited, at the middle ring (weight 1/2) of the first clip sample that closes the joint's
rings, built from the rest rings and the posed shells alone. The gate skips an item that already
fails, and a joint that no clip sample of the rig closes is a plain not-applicable unit.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field

import numpy as np

from model_kit.core.checks.entries import first_middle_last, unit
from model_kit.core.checks.overlap import body_parts
from model_kit.core.errors import Unmeasurable
from model_kit.core.manifest.rig import bending_joints
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.pose.stress import STRESS_CLIP
from model_kit.core.rig import pose_world
from model_kit.core.scene import RigRest, SceneCtx
from model_kit.core.slices import Slice, crossing_triangles, ring_loop, ring_slices
from model_kit.core.tolerances import BEND_AREA_FLOOR, BEND_GIRTH_CHANGE, BIND_WEIGHT_FLOOR

NAME = "bend-quality"
KINDS = ("girth", "collapse")
PRINCIPLE = ("Soft tissue is mostly water and nearly incompressible, so a limb keeps its girth and "
             "its section as it bends, and a twist spreads along the limb instead of crushing one "
             "ring.")
SOURCE = (
    "Soft tissue near-incompressibility (https://arxiv.org/pdf/2312.04108). The girth limit is the "
    "largest girth change measured on real limbs through bending: Ringenbach, Yu and Zhang 2024, "
    "knee girth +6.4% at 90 degrees (https://research.polyu.edu.hk/en/publications/"
    "analysis-of-knee-bending-motion-through-three-dimensional-scannin/); Klepser and Morlock "
    "2020, upper arm and thigh (https://cdatp.publia.org/cdatp/article/download/6/7/131). No study "
    "measures slice area, so the collapse floor is a stated design call.")

# Rings per joint, at the child-side weights k / (SLICES + 1). Seven puts one ring at 1/2, where
# linear blend skinning pinches most, and three on each side.
SLICES = 7
# A plant turns the far side of its ring end for end about the axis. A 90 degree turn cannot do:
# a point blended with itself turned 90 degrees keeps 0.71 of its radius at the worst, so the area
# falls to 0.5 at the least and never under the floor. End for end takes the ring to a point.
TWIST_PLANT_RAD = math.pi
# The share of the radius a pinch plant takes off the ring.
PINCH_PLANT = 0.5
ITEM = re.compile(r"^(?P<rig>\S+) (?P<part>\S+) (?P<joint>\S+) slice (?P<k>\d+) "
                  r"(?P<kind>girth|collapse) at (?P<where>.+)$")


# --- one rig's fixed facts -------------------------------------------------------------------


@dataclass(eq=False)
class _Joint:
    """A bending joint inside a skinned part, with its rest rings."""

    part: str
    joint: str
    faces: np.ndarray  # (T, 3) vertex ids of the part
    weight: np.ndarray  # (V,) child-side weight of each vertex; NaN for another bone tree's
    rest: list[Slice]


@dataclass(eq=False)
class _Prepared:
    """A rig ready to measure: its joints, and the coverage lines preparing gave."""

    key: str
    joints: list[_Joint]
    lines: list[str] = field(default_factory=list)


def _below(parent: dict[str, str | None], bone: str) -> set[str]:
    """`bone` and every bone under it."""
    out = {bone}
    for b in parent:
        up = b
        while up is not None:
            if up == bone:
                out.add(b)
            up = parent[up]
    return out


def _tree(parent: dict[str, str | None], bone: str) -> set[str]:
    """The bones of the tree that holds `bone`: all the bones that share its root."""
    def root(b: str) -> str:
        while parent[b] is not None:
            b = parent[b]
        return b
    top = root(bone)
    return {b for b in parent if root(b) == top}


def _child_weight(rest: RigRest, part: str, inside: set[str], tree: set[str]) -> np.ndarray:
    """(V,) the share of each vertex's motion that comes from the bones in `inside`, among the
    bones of `tree`. A vertex that the tree does not wholly move is NaN: it belongs to another
    tree of the rig and takes no part in this joint's rings."""
    w = np.asarray(rest.weights[part], dtype=float)
    index = np.asarray(rest.weight_groups[part], dtype=np.intp)
    w = w / w.sum(axis=1, keepdims=True)
    names = rest.vertex_groups[part]
    in_sub = np.array([n in inside for n in names])[index]
    in_tree = np.array([n in tree for n in names])[index]
    sub = (w * in_sub).sum(axis=1)
    whole = (w * in_tree).sum(axis=1)
    out = np.full(len(w), np.nan)
    ours = whole >= 1.0 - BIND_WEIGHT_FLOOR
    out[ours] = sub[ours] / whole[ours]
    return out


def _reaches_a_ring(weight: np.ndarray) -> bool:
    """Whether a part carries the child-side weight of some ring (1/8 up to 7/8): the joint lies
    inside the part. A part with a vertex wholly on each side does, and so does a part blended
    across the joint with none wholly on a side; a part wholly on one side does not. A
    ring the part cannot close is then `Unmeasured`, not skipped as not applicable."""
    own = weight[~np.isnan(weight)]
    return bool(own.size and own.min() <= SLICES / (SLICES + 1) and own.max() >= 1 / (SLICES + 1))


def _prepare(ctx: SceneCtx, key: str) -> _Prepared:
    pack, _, name = key.partition(":")
    spec = ctx.packs[pack].pieces[name].rig
    rig, rest = ctx.rigs[key], ctx.rests.get(key)
    prepared = _Prepared(key, [])
    if rest is None:
        prepared.lines.append(Unmeasured(f"{NAME}: skipped (no rest file): {key}"))
        return prepared
    parent = {b.name: b.parent for b in spec.bones}
    bending = bending_joints(spec)
    world = pose_world(rig, {})
    for part in body_parts(ctx, key):
        if len(rest.vertex_groups[part]) < 2:
            prepared.lines.append(f"{NAME}: not applicable (the part is bound to one bone and "
                                  f"does not bend): {key} {part}")
            continue
        if part not in rest.weights or part not in rest.weight_groups:
            prepared.lines.append(Unmeasured(f"{NAME}: skipped (no skin weights in the rest "
                                             f"file): {key} {part}"))
            continue
        if not (np.asarray(rest.weights[part], dtype=float).sum(axis=1) > 0).all():
            prepared.lines.append(Unmeasured(f"{NAME}: skipped (a vertex carries no skin weight, "
                                             f"so its share across a joint is undefined): "
                                             f"{key} {part}"))
            continue
        faces = rest.faces[part].astype(int)
        tris = rest.shells[part][0].astype(float)[faces]
        found = False
        for joint in bending:
            weight = _child_weight(rest, part, _below(parent, joint), _tree(parent, joint))
            if not _reaches_a_ring(weight):
                continue  # the joint is not inside this part
            found = True
            try:
                rest_rings = ring_slices(tris, weight[faces], world[joint][:3, 3], SLICES)
                if any(not (r.area > 0 and r.perimeter > 0) for r in rest_rings):
                    raise Unmeasurable("a rest ring has no area")
            except Unmeasurable as err:
                prepared.lines.append(Unmeasured(f"{NAME}: skipped ({err}): {key} {part} "
                                                 f"{joint}"))
                continue
            prepared.joints.append(_Joint(part, joint, faces, weight, rest_rings))
        if not found:
            prepared.lines.append(f"{NAME}: not applicable (no bending joint lies inside the "
                                  f"part): {key} {part}")
    return prepared


# --- the measure -----------------------------------------------------------------------------


@dataclass(eq=False)
class _Where:
    """One pose to read: its label, and the shells and bone matrices at it."""

    label: str
    shells: dict[str, np.ndarray]  # part -> (V, 3)
    bones: dict[str, np.ndarray]  # bone -> (4, 4)


@dataclass(eq=False)
class _Scan:
    """What a scan found: findings, the lines it leaves, and the range of the shape ratio."""

    findings: list[Finding] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    slices: int = 0
    shape: list[float] = field(default_factory=list)  # lowest and highest 4 pi A / P^2, or []


def _measure_joint(prep: _Prepared, joint: _Joint, where: _Where, scan: _Scan,
                   wanted: Collection[str] | None) -> None:
    shell = where.shells.get(joint.part)
    if shell is None:
        return
    try:
        got = ring_slices(shell.astype(float)[joint.faces], joint.weight[joint.faces],
                          where.bones[joint.joint][:3, 3], SLICES)
    except Unmeasurable as err:
        scan.lines.append(Unmeasured(f"{NAME}: skipped ({err}) at {where.label}: {prep.key} "
                                     f"{joint.part} {joint.joint}"))
        return
    for k, (now, then) in enumerate(zip(got, joint.rest, strict=True), start=1):
        scan.slices += 1
        q = 4 * math.pi * now.area / now.perimeter**2
        scan.shape = [min(scan.shape[0], q), max(scan.shape[1], q)] if scan.shape else [q, q]
        head_item = f"{prep.key} {joint.part} {joint.joint} slice {k}"
        girth = now.perimeter / then.perimeter - 1.0
        if abs(girth) > BEND_GIRTH_CHANGE:
            item = f"{head_item} girth at {where.label}"
            if wanted is None or item in wanted:
                scan.findings.append(Finding(NAME, item, (
                    f"perimeter {now.perimeter:.4f} against {then.perimeter:.4f} at rest "
                    f"({girth:+.1%}), over the allowed {BEND_GIRTH_CHANGE:.1%}")))
        share = now.area / then.area
        if share < BEND_AREA_FLOOR:
            item = f"{head_item} collapse at {where.label}"
            if wanted is None or item in wanted:
                scan.findings.append(Finding(NAME, item, (
                    f"area {now.area:.6f} is {share:.1%} of {then.area:.6f} at rest, under the "
                    f"floor {BEND_AREA_FLOOR:.0%}")))


def _clip_wheres(ctx: SceneCtx, key: str, samples: frozenset[int],
                 labels: Collection[str] | None) -> Iterator[_Where]:
    for clip, rig_key in ctx.rig_of_clip.items():
        data = ctx.samples.get(clip)
        if rig_key != key or data is None or clip == STRESS_CLIP:  # the stress clip is no clip
            continue
        for s, t in enumerate(data.times):
            label = f"{clip} frame {float(t):g}"
            if (samples and s not in samples) or (labels is not None and label not in labels):
                continue
            yield _Where(label, {p: v[s] for p, v in data.shells.items()},
                         {b: m[s] for b, m in data.bones.items()})


# --- plants ----------------------------------------------------------------------------------


def _planted(ctx: SceneCtx, clip: str, part: str, s: int, ids: np.ndarray,
             points: np.ndarray) -> SceneCtx:
    data = ctx.samples[clip]
    moved = np.array(data.shells[part])
    moved[s, ids] = points
    return dataclasses.replace(ctx, samples={**ctx.samples, clip: dataclasses.replace(
        data, shells={**data.shells, part: moved})})


def _axis(loop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The middle of a ring and its unit normal (Newell's)."""
    normal = np.cross(loop, np.roll(loop, -1, axis=0)).sum(axis=0)
    return loop.mean(axis=0), normal / float(np.linalg.norm(normal))


def _turn(points: np.ndarray, centre: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """The points turned by `angle` about the line through `centre` along the unit `axis`."""
    d = points - centre
    cos, sin = math.cos(angle), math.sin(angle)
    along = (d @ axis)[:, None] * axis
    return centre + along + (d - along) * cos + np.cross(axis, d) * sin


def _pinch(points, over, centre, axis) -> np.ndarray:
    """The points drawn toward the line through `centre` along `axis` by `PINCH_PLANT`."""
    d = points - centre
    return points - PINCH_PLANT * (d - (d @ axis)[:, None] * axis)


def _twist(points, over, centre, axis) -> np.ndarray:
    """The points that lie over the ring's weight turned end for end about the line."""
    out = np.array(points)
    out[over] = _turn(points[over], centre, axis, TWIST_PLANT_RAD)
    return out


def _spot(ctx: SceneCtx, key: str, joint: _Joint) -> tuple[str, int, str] | None:
    """Where a joint's plants go: (the clip, the sample, the label `<clip> frame <n>`) of the
    first clip sample, at the middle, the start and the end of each clip of the rig in turn, whose
    posed part closes all `SLICES` rings. None when no sample of the rig does: the check measures
    nothing of this joint, and says so in its coverage."""
    for clip, rig_key in ctx.rig_of_clip.items():
        data = ctx.samples.get(clip)
        if rig_key != key or data is None or clip == STRESS_CLIP or joint.part not in data.shells:
            continue
        for s in first_middle_last(range(len(data.times))):
            try:
                ring_slices(data.shells[joint.part][s].astype(float)[joint.faces],
                            joint.weight[joint.faces], data.bones[joint.joint][s][:3, 3], SLICES)
            except Unmeasurable:
                continue
            return clip, s, f"{clip} frame {float(data.times[s]):g}"
    return None


def _joint_units(ctx: SceneCtx, key: str, joint: _Joint) -> list[PlantUnit]:
    """The girth unit and the collapse unit of one joint. Each edits the ring at the middle level
    (weight 1/2, where a hinge pinches most) of the first clip sample that closes its rings."""
    spot = _spot(ctx, key, joint)
    head_item = f"{key} {joint.part} {joint.joint}"
    if spot is None:
        return [PlantUnit(head_item, (), not_applicable=(
            "no clip sample of the rig closes all the rings of this joint"))]
    clip, s, where = spot
    data = ctx.samples[clip]
    shell = data.shells[joint.part][s].astype(float)
    corner = joint.weight[joint.faces]
    level = (SLICES // 2 + 1) / (SLICES + 1)
    ids = np.unique(joint.faces[crossing_triangles(corner, level)])
    centre, axis = _axis(ring_loop(shell[joint.faces], corner, data.bones[joint.joint][s][:3, 3],
                                   level))
    over = joint.weight[ids] >= level

    def planted(edit) -> SceneCtx:
        return _planted(ctx, clip, joint.part, s, ids, edit(shell[ids], over, centre, axis))

    # A twist leaves the ring's area to the weights the band holds on each side of the level, and
    # a band with a vertex close to the level turns the ring little. Squeezing the band by half
    # takes the area to a quarter, under the floor, whatever the weights.
    return [unit(f"{head_item} slice {SLICES // 2 + 1} {kind} at {where}",
                 [(label, planted(edit), {s}) for label, edit in edits])
            for kind, edits in (("girth", (("pinched", _pinch),)),
                                ("collapse", (("twisted", _twist), ("squeezed", _pinch))))]


# --- the check -------------------------------------------------------------------------------


class BendQuality:
    name = NAME
    kinds: frozenset[str] = frozenset()  # it measures a rig's body, not a piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._prepared: dict[str, tuple[object, object, _Prepared]] = {}
        self._full: tuple[SceneCtx, dict[str, _Scan]] | None = None

    # -- preparing a rig

    def _prep(self, ctx: SceneCtx, key: str) -> _Prepared:
        spec = ctx.packs[key.partition(":")[0]].pieces[key.partition(":")[2]].rig
        kept = self._prepared.get(key)
        if kept is None or kept[0] is not spec or kept[1] is not ctx.rests.get(key):
            kept = self._prepared[key] = (spec, ctx.rests.get(key), _prepare(ctx, key))
        return kept[2]

    # -- scanning

    def _scan(self, ctx: SceneCtx, key: str, only: Collection[str] | None,
              samples: frozenset[int]) -> _Scan:
        """The findings and lines of one rig. With `only`, just those items' poses are read; with
        `samples`, just those clip samples."""
        scan = _Scan()
        prep = self._prep(ctx, key)
        scan.lines += prep.lines
        if not prep.joints:
            return scan
        labels = None
        if only is not None:
            labels = {m["where"] for item in only if (m := ITEM.match(item)) and m["rig"] == key}
        for where in _clip_wheres(ctx, key, samples, labels):
            for joint in prep.joints:
                _measure_joint(prep, joint, where, scan, only)
        return scan

    def _keys(self, ctx: SceneCtx, only: Collection[str] | None) -> list[str]:
        return [k for k in ctx.rigs
                if only is None or any(i.startswith(f"{k} ") for i in only)]

    def _all(self, ctx: SceneCtx) -> dict[str, _Scan]:
        if self._full is None or self._full[0] is not ctx:
            self._full = (ctx, {k: self._scan(ctx, k, None, frozenset()) for k in ctx.rigs})
        return self._full[1]

    # -- the check

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        if only is None and not samples:
            return [f for scan in self._all(ctx).values() for f in scan.findings]
        found: list[Finding] = []
        for key in self._keys(ctx, only):
            found += self._scan(ctx, key, only, samples).findings
        return [f for f in found if only is None or f.item in only]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        for key, scan in self._all(ctx).items():
            lines += scan.lines
            joints = self._prep(ctx, key).joints
            if scan.slices:
                lines.append(
                    f"{NAME}: measured {len(joints)} joint(s), {scan.slices} rings; shape ratio "
                    f"4 pi A / P^2 from {scan.shape[0]:.2f} to {scan.shape[1]:.2f}, reported and "
                    f"not judged: {key}")
        return list(dict.fromkeys(lines))

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        """One unit per bending joint inside a part, and kind: the elbow and the knee as much as
        the chest. Built from the rest rings and the posed shells alone, never from the check's
        own scan of a ring against its rest ring."""
        units: list[PlantUnit] = []
        for key in ctx.rigs:
            for joint in self._prep(ctx, key).joints:
                units += _joint_units(ctx, key, joint)
        return units
