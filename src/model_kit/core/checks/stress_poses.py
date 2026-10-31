"""Check stress-poses: a body is whole at the ends of its joints' ranges.

The poses are `core.pose.stress`: each bending joint, with its parent joint, taken to the ends of
their ranges (`rig.joints`), the rest of the body at rest. At each pose, three things must hold:

- **C** no body shell sinks into another deeper than the skin: overlap's test C, with its budget
  (`SKIN_OF_EFFECTOR` times the rig's radius) and its excused pairs. Contact at the end of a range
  is touch, which the skin allows; passing through is not.
- **self** no skinned part passes through itself: two faces of one part that share no vertex and
  cut each other deeper than the skin (`geom.self_cuts`). C only finds a vertex in a different
  shell, so a limb that is one mesh would fold through itself unseen. A pair of faces that
  already cut in the rest pose is the mesh's own and is left out, as C leaves out the pairs of
  shells that cross at rest. A face shares a vertex with another when their corners weld
  (`WELD_OF_PIECE_SIZE`), because a glTF seam splits one vertex in two.

  **A bent joint's skin folds into itself at the crease; the check judges the rest of the body,
  not the crease.** So a crossing between a face on one side of a posed joint and a face on the
  other is excused, as overlap's test C excuses a bone and its deforming ancestor. A face is on
  the child side when its main bone (the one that weighs most over its corners) is the joint's
  bone or a bone below it. It is on the parent side when its main bone is the joint's parent
  bone, or another bone of the parent's chain that does not lie below the joint. The parent bone
  is the nearest ancestor that skins a part, as in test C. In a pair pose a crossing across
  either posed joint is excused. This adds no tolerance.
- **inverted** no face turns inside out: a skinned face whose posed normal is more than
  `INVERTED_FACE_DEG` from its rest normal turned by the face's main bone (the bone that weighs
  most over its three corners). Posing never changes a face's winding, so topology's flipped rule
  cannot see this.

Items: `<pack>:<piece> <pose> C <a> x <b>`, `<pack>:<piece> <pose> self <part>`,
`<pack>:<piece> <pose> inverted <part>`. A pose is named by `core.pose.stress`, so one allow line
covers one pose. Real ranges are coupled (full shoulder flexion limits rotation), so a corner can
be a pose no body reaches; a finding there is not cured by narrowing a range without a source. It
takes an allow line that names the pose and the coupling's citation.

A rig with several roots (bone trees) has no stress pose for a face weighted across two trees:
a stress pose turns the joints of one tree and leaves the others at rest, so such a face tears.
Those faces are left out of self and inverted, and a not-applicable line says how many. A body
shell that holds such a face is still read by C as it stands.

A rig with no bending joint prints a not-applicable line. A bending joint with no `joints` entry,
a part with no skin weights and a rig with no rest file are `Unmeasured`: the joints that have a
range are still measured. So is a body test that overlap cannot read (`Unmeasurable` from
`body_contacts`): C is `Unmeasured` and the other kinds still run.

Plants, per rig and kind, built from the stressed body's geometry alone (never from the check's
own tests), at the first, the middle and the last pose: one body vertex sunk into a shell it is not
excused from, at fractions of the room the shell has (C, `overlap.ROOM_FRACTIONS`); one vertex of a
part pushed through the part's far wall (self); one posed face turned inside out (inverted). A
plant is a copy of the stressed scene with one pose's shells moved, and `run` reads a scene that
already holds the stress clip as it is. A pose whose item the real run already reports is skipped
by the gate. A kind that judges no item (no pair of closed shells, no part of two faces, no face)
is a plain not-applicable line. A kind that judges an item and has no plant the check reports
stops the gate (`BlindCheck`). A body test that overlap cannot read is `Unmeasured`.

The self rule has a known blind spot: a cut is no deeper than its faces are wide, so it cannot see
a fold through faces narrower than the skin. The self plant goes on the part with the widest
faces, so it proves the rule there only (`test_stress_poses.py` pins the gap with a strict xfail).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field

import numpy as np

from model_kit.core.checks.entries import unit
from model_kit.core.checks.overlap import (
    body_contact_plants,
    body_contacts,
    body_parts,
    body_unreadable,
    has_body_test,
    room_ladder,
)
from model_kit.core.errors import Unmeasurable
from model_kit.core.geom import SurfaceIndex, self_cuts, weld
from model_kit.core.manifest import RigSpec
from model_kit.core.manifest.rig import bending_joints
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.pose import skin
from model_kit.core.pose.stress import (
    STRESS_CLIP,
    StressPose,
    missing_ranges,
    stress_poses,
    stress_scene,
)
from model_kit.core.rig import RigData, pose_world
from model_kit.core.scene import ClipSamples, RigRest, SceneCtx
from model_kit.core.supports import body_radius
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    INVERTED_FACE_DEG,
    SKIN_OF_EFFECTOR,
    WELD_OF_PIECE_SIZE,
)

NAME = "stress-poses"
KINDS = ("C", "self", "inverted")
PRINCIPLE = ("A body never passes through itself anywhere in its range of motion: skin meets skin "
             "at the end of a range, and meeting is not passing through.")
SOURCE = (
    "Joint ranges come from published range-of-motion tables, and each rig cites the one it used "
    "in rig.joints: AAOS normal range of motion via Hendriks et al. 2020, supplementary table "
    "(https://cdn-links.lww.com/permalink/prsgo/b/"
    "prsgo_8_6_2020_04_17_hendriks_gox-d-20-00155r2_sdc1.pdf); NASA Human Integration Design "
    "Handbook Rev 1, Appendix B Table 2 (https://www.nasa.gov/sites/default/files/atoms/files/"
    "human_integration_design_handbook_revision_1.pdf); 38 CFR 4.71a Note (2) "
    "(https://www.law.cornell.edu/cfr/text/38/4.71a). Sources disagree by up to 20 degrees on "
    "some axes. Contact at a range end is touch, which overlap's skin budget allows.")
# Plants try these vertices of a part at most, and push each this share of the way to the part's
# far wall beyond that wall (at least `CALIBRATION_FAIL` skins): a face that only just pierces a
# wall cuts it a sliver's width, and no deeper than the skin.
PLANT_VERTICES = 8
PLANT_BEYOND = 0.5


# --- one rig's fixed facts -------------------------------------------------------------------


@dataclass(eq=False)
class _Part:
    """What a skinned part keeps between poses."""

    faces: np.ndarray  # (F, 3) vertex ids
    ids: np.ndarray  # (F, 3) welded vertex ids, for "share a vertex"
    normals: np.ndarray  # (F, 3) unit rest normals, 0 for a face with no area
    bone: np.ndarray  # (F,) index into `bones` of the face's main bone
    bones: tuple[str, ...]
    slots: np.ndarray  # (V, K) index into `bones` of each weight slot
    live: np.ndarray  # (V, K) whether the slot holds a weight
    seam: np.ndarray  # (F,) the face is weighted across two bone trees
    rest_pairs: frozenset[int] | None = None  # faces i < j that cut at rest, as i * F + j


@dataclass(eq=False)
class _Fixed:
    """A rig's facts that do not change with the pose: its spec, the poses, the budget and the
    parts."""

    key: str
    spec: RigSpec
    rest: RigRest
    poses: list[StressPose]
    parts: dict[str, _Part]
    skin: float
    subtree: dict[str, tuple[str, ...]]  # each joint with the bones below it
    above: dict[str, frozenset[str]]  # each joint's parent side: see `_above`
    lines: list[str] = field(default_factory=list)  # coverage lines from preparing


def _unit(v: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, length, out=np.zeros_like(v), where=length > 0)


def _normals(tris: np.ndarray) -> np.ndarray:
    return _unit(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]))


def _main_bones(rest: RigRest, part: str) -> np.ndarray:
    """Per face, the index into the part's vertex groups of the group that weighs most over the
    face's three corners."""
    faces = rest.faces[part].astype(int)
    weights = np.asarray(rest.weights[part], dtype=float)
    slots = np.asarray(rest.weight_groups[part], dtype=np.intp)
    dense = np.zeros((weights.shape[0], len(rest.vertex_groups[part])))
    np.add.at(dense, (np.arange(len(weights))[:, None], slots), weights)
    return dense[faces].sum(axis=1).argmax(axis=1)


def _seam(slots: np.ndarray, live: np.ndarray, faces: np.ndarray, bones: tuple[str, ...],
          tree: dict[str, int]) -> np.ndarray:
    """(F,) whether a face is weighted across two bone trees: the bones that weigh its three
    corners do not all lie in one tree."""
    seen = np.array([tree.get(b, -1) for b in bones])[slots]
    low = np.where(live, seen, len(tree) + 1).min(axis=1)
    high = np.where(live, seen, -1).max(axis=1)
    face_low, face_high = low[faces].min(axis=1), high[faces].max(axis=1)
    return (face_high >= 0) & (face_low != face_high)


def _above(spec: RigSpec, skins: set[str], subtree: dict[str, tuple[str, ...]]
           ) -> dict[str, frozenset[str]]:
    """Each joint's parent side: its parent bone (the nearest ancestor that skins a part, as test
    C takes it) and the other bones of that bone's chains, but no bone below the joint."""
    parent = {bone.name: bone.parent for bone in spec.bones}
    out = {}
    for joint in spec.joints:
        up = parent[joint]
        while up is not None and up not in skins:
            up = parent[up]
        mates = {b for chain in spec.chains.values() if up in chain.bones for b in chain.bones}
        out[joint] = frozenset(({up} | mates) if up is not None else ()) - set(subtree[joint])
    return out


def _fixed(ctx: SceneCtx, key: str, spec: RigSpec, poses: list[StressPose]) -> _Fixed | str:
    """The rig's fixed facts, or the reason it has none (no radius to measure by)."""
    rest = ctx.rests[key]
    base = body_radius(spec, rest)
    if base is None:
        return "no radius to measure by"
    every = np.concatenate([rest.shells[p][0] for p in rest.shells]).astype(float)
    size = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    roots = [bone.name for bone in spec.bones if bone.parent is None]
    owner: dict[str, str] = {}
    for bone in spec.bones:  # the root each bone grows from (parents come first)
        owner[bone.name] = bone.name if bone.parent is None else owner[bone.parent]
    tree = {name: roots.index(root) for name, root in owner.items()}
    parts = {}
    for part in body_parts(ctx, key):
        faces = rest.faces[part].astype(int)
        tris = rest.shells[part][0].astype(float)[faces]
        ids = weld(tris, WELD_OF_PIECE_SIZE * size)[1] if len(tris) else faces
        slots = np.asarray(rest.weight_groups[part], dtype=np.intp)
        live = np.asarray(rest.weights[part]) > 0
        seam = (_seam(slots, live, faces, rest.vertex_groups[part], tree)
                if len(roots) > 1 and len(faces) else np.zeros(len(faces), dtype=bool))
        parts[part] = _Part(faces, ids, _normals(tris) if len(tris) else tris,
                            _main_bones(rest, part), rest.vertex_groups[part], slots, live, seam)
    children: dict[str, list[str]] = {}
    for bone in spec.bones:
        children.setdefault(bone.parent, []).append(bone.name)

    def below(bone: str) -> tuple[str, ...]:
        return (bone, *(b for c in children.get(bone, ()) for b in below(c)))
    subtree = {joint: below(joint) for joint in spec.joints}
    skins = {g for part in parts.values() for g in part.bones}
    return _Fixed(key, spec, rest, poses, parts, SKIN_OF_EFFECTOR * base, subtree,
                  _above(spec, skins, subtree))


def _rest_pairs(fixed: _Fixed, part: str) -> frozenset[int]:
    """The faces of the part that cut each other at rest, kept once per rig."""
    p = fixed.parts[part]
    if p.rest_pairs is None:
        tris = fixed.rest.shells[part][0].astype(float)[p.faces]
        pairs, _ = self_cuts(tris, fixed.skin, p.ids)
        p.rest_pairs = frozenset(int(i) * len(p.faces) + int(j) for i, j in pairs)
    return p.rest_pairs


# --- what moved as one body ------------------------------------------------------------------
# At a stress pose only one or two joints turn. Every bone under a turned joint moves by that
# joint's change, so a vertex that only those bones weigh is carried rigidly with them: it stands
# to any other vertex of its group as it did at rest. A pair that was not crossing at rest cannot
# cross now, and one that was is excused, so the tests leave such pairs out and look at the
# blend zones and at what lies between bodies that moved apart. Same ids mean same body.


def _bone_bodies(fixed: _Fixed, pose: StressPose) -> dict[str, int]:
    """The body each bone moves with at this pose: 0 for a bone that stays at rest. `pose.joints`
    lists the child joint first, then its parent, and the loop takes them in reverse, so the later
    rank overwrites the earlier. The bones under the parent joint get 1, and the bones under the
    child joint, which lie under the parent too, get 2. A pose that turns one joint gives its
    bones 1."""
    body = {bone.name: 0 for bone in fixed.spec.bones}
    for rank, joint in enumerate(reversed(pose.joints), start=1):
        for bone in fixed.subtree[joint]:
            body[bone] = rank
    return body


def _rigid(fixed: _Fixed, scene: SceneCtx, rig: RigData, s: int, pristine: bool
           ) -> dict[str, np.ndarray]:
    """Per part, (V,) the body each vertex moves with at pose `s`, or -1 for a vertex that is
    blended between bodies or (in a scene whose samples were edited) is not where the pose puts
    it."""
    body = _bone_bodies(fixed, fixed.poses[s])
    out = {}
    for part, p in fixed.parts.items():
        of_group = np.array([body[b] for b in p.bones])
        seen = of_group[p.slots]
        low = np.where(p.live, seen, len(p.bones) + 3).min(axis=1)
        high = np.where(p.live, seen, -1).max(axis=1)
        out[part] = np.where(low == high, low, -1)
    if not pristine:
        data = scene.samples[STRESS_CLIP]
        posed = skin(fixed.rest, rig, pose_world(rig, fixed.poses[s].pose))
        for part, ids in out.items():
            ids[(posed[part].astype(np.float32) != data.shells[part][s]).any(axis=1)] = -1
    return out


def _faces_together(p: _Part, vertex_body: np.ndarray) -> np.ndarray:
    """(F,) the body a face moves with: its corners' body when they agree, else -1."""
    a, b, c = vertex_body[p.faces].T
    return np.where((a == b) & (b == c), a, -1)


# --- the crease of a posed joint -------------------------------------------------------------


def _sides(fixed: _Fixed, s: int, part: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """For each joint pose `s` turns (the child joint first), (F,) masks of the part's faces on
    its child side and on its parent side. A face sits on the side of its main bone."""
    p = fixed.parts[part]
    out = []
    for joint in fixed.poses[s].joints:
        below, above = set(fixed.subtree[joint]), fixed.above[joint]
        out.append((np.array([b in below for b in p.bones])[p.bone],
                    np.array([b in above for b in p.bones])[p.bone]))
    return out


def _folded(sides: list[tuple[np.ndarray, np.ndarray]], i: np.ndarray, j: np.ndarray
            ) -> np.ndarray:
    """Whether faces `i` and `j` (index arrays of one length) lie on the two sides of one of the
    posed joints: the crease, where a bent joint's skin folds into itself."""
    out = np.zeros(len(i), dtype=bool)
    for child, parent in sides:
        out |= (child[i] & parent[j]) | (parent[i] & child[j])
    return out


# --- the measure -----------------------------------------------------------------------------


def _self(fixed: _Fixed, samples: ClipSamples, s: int, part: str, vertex_body: np.ndarray
          ) -> tuple[float, int, int] | None:
    """The deepest cut between two faces of a part at pose `s` that did not cut at rest and does
    not lie across a posed joint, leaving out faces weighted across two bone trees: (depth, face,
    face), or None. `vertex_body` is `_rigid`'s, for the part."""
    p = fixed.parts[part]
    if len(p.faces) < 2:
        return None
    tris = samples.shells[part][s].astype(float)[p.faces]
    pairs, depth = self_cuts(tris, fixed.skin, p.ids, _faces_together(p, vertex_body))
    if not len(pairs):
        return None
    rest = _rest_pairs(fixed, part)
    first, second = pairs[:, 0], pairs[:, 1]
    keep = np.array([int(i) * len(p.faces) + int(j) not in rest for i, j in pairs])
    keep &= ~_folded(_sides(fixed, s, part), first, second)
    keep &= ~(p.seam[first] | p.seam[second])
    new = np.nonzero(keep)[0]
    if not len(new):
        return None
    k = int(new[np.argmax(depth[new])])
    return float(depth[k]), int(pairs[k][0]), int(pairs[k][1])


def _carried(fixed: _Fixed, samples: ClipSamples, rig: RigData, s: int, part: str
             ) -> np.ndarray:
    """The rest normal of every face of a part turned by its main bone's change at pose `s`: the
    bone's posed matrix times the inverse of its rest matrix, as `core.pose.skin` takes it."""
    p = fixed.parts[part]
    change = np.empty((len(p.bones), 3, 3))
    for g, bone in enumerate(p.bones):
        full = samples.bones[bone][s] @ np.linalg.inv(rig.world @ rig.rest[bone])
        change[g] = full[:3, :3]
    return _unit(np.einsum("fij,fj->fi", change[p.bone], p.normals))


def _inverted(fixed: _Fixed, samples: ClipSamples, rig: RigData, s: int, part: str
              ) -> tuple[float, int] | None:
    """The face of a part at pose `s` turned furthest past `INVERTED_FACE_DEG`: (the angle in
    degrees, the face), or None. A face weighted across two bone trees is left out."""
    p = fixed.parts[part]
    if len(p.faces) == 0:
        return None
    posed = _normals(samples.shells[part][s].astype(float)[p.faces])
    carried = _carried(fixed, samples, rig, s, part)
    both = ((np.linalg.norm(posed, axis=1) > 0) & (np.linalg.norm(carried, axis=1) > 0)
            & ~p.seam)
    angle = np.degrees(np.arccos(np.clip(np.einsum("fi,fi->f", posed, carried), -1.0, 1.0)))
    angle = np.where(both, angle, 0.0)
    f = int(angle.argmax())
    return (float(angle[f]), f) if angle[f] > INVERTED_FACE_DEG else None


def _wanted(fixed: _Fixed, only: Collection[str] | None, samples: frozenset[int]
            ) -> dict[int, set[str]]:
    """Per pose index, the kinds to measure: all of them, or those that `only` names."""
    out: dict[int, set[str]] = {}
    prefix = f"{fixed.key} "
    for s, pose in enumerate(fixed.poses):
        if samples and s not in samples:
            continue
        head = f"{prefix}{pose.name} "
        if only is None:
            out[s] = set(KINDS)
            continue
        kinds = {item[len(head):].split(" ", 1)[0] for item in only if item.startswith(head)}
        if kinds:
            out[s] = kinds & set(KINDS)
    return out


def _measure(fixed: _Fixed, scene: SceneCtx, rig: RigData, only: Collection[str] | None,
             samples: frozenset[int], pristine: bool = True, skipped: list[str] | None = None
             ) -> list[Finding]:
    """The findings of one rig over the stress scene `scene`. `pristine` says the samples are the
    poses as `stress_scene` made them; a scene with edited samples (a plant) is not, and then
    every vertex that is not where its pose puts it is measured alone. When overlap cannot read
    the body test C (`Unmeasurable`), C gives no finding and the reason goes to `skipped`."""
    wanted = _wanted(fixed, only, samples)
    if not wanted:
        return []
    data = scene.samples[STRESS_CLIP]
    clip = scene.clips[STRESS_CLIP]
    found: list[Finding] = []
    head = {s: f"{fixed.key} {fixed.poses[s].name}" for s in wanted}
    bodies = {s: _rigid(fixed, scene, rig, s, pristine) for s in wanted}
    ids = frozenset(s for s, kinds in wanted.items() if "C" in kinds)
    if ids:
        every = len(ids) == len(fixed.poses)
        together = np.full((len(fixed.poses), sum(len(v) for v in bodies[min(ids)].values())), -1)
        for s in ids:
            together[s] = np.concatenate([bodies[s][part] for part in fixed.parts])
        try:
            contacts = body_contacts(scene, clip, frozenset() if every else ids, together)
        except Unmeasurable as err:
            contacts = []
            if skipped is not None:
                skipped.append(str(err))
        for c in contacts:
            if "C" in wanted.get(c.sample, ()):
                found.append(Finding(NAME, f"{head[c.sample]} C {c.a} x {c.b}",
                                     f"depth {c.depth:.4f} over budget {c.budget:.4f}"))
    for s, kinds in wanted.items():
        for part in fixed.parts:
            if "self" in kinds:
                cut = _self(fixed, data, s, part, bodies[s][part])
                if cut is not None:
                    found.append(Finding(
                        NAME, f"{head[s]} self {part}",
                        f"faces {cut[1]} and {cut[2]} cut {cut[0]:.4f} deep, over budget "
                        f"{fixed.skin:.4f}"))
            if "inverted" in kinds:
                turned = _inverted(fixed, data, rig, s, part)
                if turned is not None:
                    found.append(Finding(
                        NAME, f"{head[s]} inverted {part}",
                        f"face {turned[1]} turned {turned[0]:.1f} deg from its bone's rest "
                        f"normal, over {INVERTED_FACE_DEG:g}"))
    return [f for f in found if only is None or f.item in only]


# --- plants ----------------------------------------------------------------------------------


def _with_vertex(scene: SceneCtx, part: str, s: int, vertex: int, point: np.ndarray
                 ) -> SceneCtx:
    """The stress scene with one vertex of a part moved to `point` at pose `s`."""
    data = scene.samples[STRESS_CLIP]
    moved = np.array(data.shells[part])
    moved[s, vertex] = point
    return dataclasses.replace(scene, samples={STRESS_CLIP: dataclasses.replace(
        data, shells={**data.shells, part: moved})})


def _spread(n: int) -> list[int]:
    """Up to `PLANT_VERTICES` indexes spread over 0..n-1."""
    return sorted({int(i) for i in np.linspace(0, n - 1, min(n, PLANT_VERTICES))})


def _far_wall(tris: np.ndarray, origin: np.ndarray, direction: np.ndarray, own: np.ndarray
              ) -> tuple[float, int] | None:
    """How far along the unit `direction` from `origin` the ray first meets a face of `tris`
    (F, 3, 3) other than the faces in the mask `own` (the faces of the vertex it leaves), and
    which face, or None. Moeller and Trumbore's ray and triangle test."""
    edge1, edge2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    live = ~own & (np.abs(det) > 1e-14 * max(1.0, float(np.abs(det).max())))
    inv = np.zeros(len(det))
    inv[live] = 1.0 / det[live]
    tvec = origin - tris[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, edge1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", edge2, qvec) * inv
    hit = live & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if not hit.any():
        return None
    face = int(np.nonzero(hit)[0][np.argmin(t[hit])])
    return float(t[face]), face


def _face_reach(tris: np.ndarray) -> np.ndarray:
    """(F,) the radius of the circle inside each face. The depth of a cut is measured to the edges
    of both faces (`geom.tri_tri_cut`), so no cut is deeper than the faces that make it are wide."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    edges = (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - b, axis=1)
             + np.linalg.norm(a - c, axis=1))
    return 2 * area / np.where(edges > 0, edges, 1.0)


def _self_parts(fixed: _Fixed, scene: SceneCtx, s: int
                ) -> list[tuple[str, _Part, np.ndarray, np.ndarray]]:
    """The parts the self rule judges at pose `s` (those of two faces or more), as (the part, its
    faces, its points, the width of each face), the part with the widest faces first. A cut is as
    deep as its faces are wide (`geom.tri_tri_cut`), so a part of faces narrower than the skin can
    show no self-cut at all, and the widest-faced part is the one a plant can prove most on."""
    data = scene.samples[STRESS_CLIP]
    out = []
    for part, p in fixed.parts.items():
        if len(p.faces) >= 2:
            points = data.shells[part][s].astype(float)
            out.append((part, p, points, _face_reach(points[p.faces])))
    return sorted(out, key=lambda q: -float(q[3].max()))


def _self_candidates(fixed: _Fixed, scene: SceneCtx, s: int
                     ) -> Iterator[tuple[str, SceneCtx]]:
    """Copies of the scene with one vertex of a part pushed through the part's far wall: (the
    item it should fail, the copy). A vertex goes straight toward the deepest point inside the
    part (the centre of the largest ball in it), past the first wall it meets there, by
    `CALIBRATION_FAIL` skins. A part with no inside gives none.

    The parts with the widest faces come first (`_self_parts`), and in a part the vertices of
    its widest faces."""
    depth = CALIBRATION_FAIL * fixed.skin
    for part, p, points, reach in _self_parts(fixed, scene, s):
        ladder = room_ladder(SurfaceIndex(points[p.faces]))
        if not ladder:
            continue
        deep = ladder[-1]
        wide = np.argsort(-reach, kind="stable")
        made, seen = 0, set()
        for f in wide:
            for v in p.faces[f]:
                if made == PLANT_VERTICES:
                    break
                if int(v) in seen:
                    continue
                seen.add(int(v))
                way = deep - points[v]
                size = float(np.linalg.norm(way))
                if not size > 0:
                    continue
                own = (p.faces == v).any(axis=1)
                hit = _far_wall(points[p.faces], points[v], way / size, own)
                if hit is None:
                    continue
                wall, across = hit
                # a pair the rule excuses proves nothing: a crease across a posed joint, a seam
                faces = np.nonzero(own)[0]
                if (_folded(_sides(fixed, s, part), faces, np.full(len(faces), across)).any()
                        or p.seam[faces].any() or p.seam[across]):
                    continue
                moved = _with_vertex(
                    scene, part, s, int(v),
                    points[v] + way / size * (wall + max(depth, PLANT_BEYOND * wall)))
                made += 1
                yield f"{fixed.key} {fixed.poses[s].name} self {part}", moved
            if made == PLANT_VERTICES:
                break


def _inverted_candidates(fixed: _Fixed, scene: SceneCtx, s: int
                         ) -> Iterator[tuple[str, SceneCtx]]:
    """Copies of the scene with one face turned inside out: the third corner of a face goes to
    its reflection across the line through the other two, which reverses the face's winding."""
    data = scene.samples[STRESS_CLIP]
    for part in sorted(fixed.parts, key=lambda q: -len(fixed.parts[q].faces)):
        p = fixed.parts[part]
        points = data.shells[part][s].astype(float)
        for f in _spread(len(p.faces)):
            a, b, c = (points[v] for v in p.faces[f])
            edge = b - a
            length2 = float(edge @ edge)
            if not length2 > 0:
                continue
            foot = a + edge * (float((c - a) @ edge) / length2)
            moved = _with_vertex(scene, part, s, int(p.faces[f][2]), 2 * foot - c)
            yield f"{fixed.key} {fixed.poses[s].name} inverted {part}", moved


def _plant_samples(n: int) -> list[int]:
    """The poses a plant may go at: the first, the middle and the last."""
    return list(dict.fromkeys([0, n // 2, n - 1]))


# What a kind says when it judges no item at all.
_NO_ITEM = {
    "C": "no pair of closed body shells is judged",
    "self": "no part has two faces for one to cross the other",
    "inverted": "no part has a face to turn inside out",
}


def _first_item(pairs: Iterator[tuple[str, SceneCtx]]) -> tuple[str | None, list[SceneCtx]]:
    """The first item the pairs name, and its copies (the pairs that follow with the same item)."""
    item, copies = None, []
    for named, moved in pairs:
        if item is None:
            item = named
        elif named != item:
            break
        copies.append(moved)
    return item, copies


def _judged_item(fixed: _Fixed, scene: SceneCtx, kind: str, s: int) -> str | None:
    """The first item the kind judges at pose `s` (the part its plants try first), or None when
    no part is of the judged kind. Used only to name a plant that could not be made."""
    head = f"{fixed.key} {fixed.poses[s].name}"
    if kind == "self":
        parts = _self_parts(fixed, scene, s)
        return f"{head} self {parts[0][0]}" if parts else None
    parts = sorted((q for q in fixed.parts if len(fixed.parts[q].faces)),
                   key=lambda q: -len(fixed.parts[q].faces))
    return f"{head} inverted {parts[0]}" if parts else None


def _units(rig: _Rig) -> list[PlantUnit]:
    """The plant units of one rig, from the stressed body's geometry alone (never from the
    check's own tests). Per kind, at the first, the middle and the last pose: the first item the
    kind's candidates name, with that item's candidates (a vertex or a face of the same part).
    The gate skips a pose whose item the real run already reports, as the corners of a range
    often do.

    A kind with a judged item and no plant at any pose is one unit with no candidates and no
    reason, which the gate stops on. A kind that judges no item says so."""
    fixed, scene = rig.fixed, rig.scene
    clip = scene.clips[STRESS_CLIP]
    poses = _plant_samples(len(fixed.poses))
    out: list[PlantUnit] = []

    def first(s: int, kind: str) -> tuple[str | None, list[SceneCtx]]:
        if kind == "C":
            head = f"{fixed.key} {fixed.poses[s].name}"
            for a, b, copies in body_contact_plants(scene, clip, s):
                return f"{head} C {a} x {b}", copies
            return None, []
        return _first_item((_self_candidates if kind == "self" else _inverted_candidates)(
            fixed, scene, s))

    for kind in KINDS:
        if kind == "C":
            if body_unreadable(scene, clip):
                continue  # coverage says why: a line for missing data
            if not has_body_test(scene, clip):
                out.append(PlantUnit(f"{fixed.key} C", (), not_applicable=(
                    "fewer than two closed body shells")))
                continue
        made: list[PlantUnit] = []
        for s in poses:
            item, copies = first(s, kind)
            if item is not None and copies:
                made.append(unit(item, [(kind, moved, {s}) for moved in copies]))
        if made:
            out += made
            continue
        if kind == "C":
            judged = next((f"{fixed.key} {fixed.poses[s].name} C {a} x {b}" for s in poses
                           for a, b, _ in body_contact_plants(scene, clip, s)), None)
        else:
            judged = next((q for s in poses if (q := _judged_item(fixed, scene, kind, s))), None)
        out.append(unit(judged, []) if judged is not None else PlantUnit(
            f"{fixed.key} {kind}", (), not_applicable=_NO_ITEM[kind]))
    return out


# --- the check -------------------------------------------------------------------------------


@dataclass(eq=False)
class _Rig:
    """A rig ready to measure: its fixed facts and its stress scene."""

    fixed: _Fixed
    scene: SceneCtx
    rig: RigData
    pristine: bool  # the scene's samples are the poses as `stress_scene` made them


def _is_stress_scene(ctx: SceneCtx, key: str, poses: list[StressPose]) -> bool:
    """Whether `ctx` already holds the stress clip of this rig (a plant's scene does)."""
    data = ctx.samples.get(STRESS_CLIP)
    return (ctx.rig_of_clip.get(STRESS_CLIP) == key and data is not None
            and len(data.times) == len(poses))


class StressPoses:
    name = NAME
    kinds: frozenset[str] = frozenset()  # it measures a rig's body, not a piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._fixed: dict[str, tuple[RigSpec, RigRest, _Fixed | str]] = {}
        self._stress: dict[str, tuple[SceneCtx, SceneCtx | Unmeasurable]] = {}

    # -- preparing a rig

    def _scene(self, ctx: SceneCtx, key: str, poses: list[StressPose]) -> SceneCtx:
        """The stress scene of a rig: `ctx` itself when it holds the clip already, else built
        once per scene. Raises `Unmeasurable` when the rig cannot be posed."""
        if _is_stress_scene(ctx, key, poses):
            return ctx
        kept = self._stress.get(key)
        if kept is None or kept[0] is not ctx:
            try:
                made: SceneCtx | Unmeasurable = stress_scene(ctx, key, partial=True)[0]
            except Unmeasurable as err:
                made = err
            kept = self._stress[key] = (ctx, made)
        if isinstance(kept[1], Unmeasurable):
            raise kept[1]
        return kept[1]

    def _prepare(self, ctx: SceneCtx, key: str) -> tuple[_Rig | None, list[str]]:
        """The rig ready to measure, or None with the coverage lines that say why not. Lines come
        back also for a rig that is measured in part."""
        pack, _, name = key.partition(":")
        spec = ctx.packs[pack].pieces[name].rig
        if spec is None or not bending_joints(spec):
            return None, [f"{NAME}: not applicable (the rig has no bending joint): {key}"]
        lines: list[str] = []
        missing = missing_ranges(spec)
        if missing:
            lines.append(Unmeasured(f"{NAME}: skipped (no range for the joint(s) "
                                    f"{', '.join(missing)}): {key}"))
        rest = ctx.rests.get(key)
        if rest is None:
            return None, lines + [Unmeasured(f"{NAME}: skipped (no rest file): {key}")]
        poses = stress_poses(spec, partial=True)
        try:
            scene = self._scene(ctx, key, poses) if poses else None
        except Unmeasurable as err:
            return None, lines + [Unmeasured(f"{NAME}: skipped ({err}): {key}")]
        if scene is None:
            return None, lines + [Unmeasured(
                f"{NAME}: skipped (no joint has a range, so there is no pose to take): {key}")]
        cached = self._fixed.get(key)
        if cached is None or cached[0] is not spec or cached[1] is not rest:
            cached = self._fixed[key] = (spec, rest, _fixed(ctx, key, spec, poses))
        fixed = cached[2]
        if isinstance(fixed, str):
            return None, lines + [Unmeasured(f"{NAME}: skipped ({fixed}): {key}")]
        torn = sum(int(p.seam.sum()) for p in fixed.parts.values())
        if torn:
            lines.append(f"{NAME}: not applicable ({torn} faces are weighted across two bone "
                         f"trees, which no stress pose defines; they are skipped): {key}")
        return _Rig(fixed, scene, ctx.rigs[key], scene is not ctx), lines

    def _keys(self, ctx: SceneCtx, only: Collection[str] | None) -> list[str]:
        """The rigs a run reads: all, or those that an item of `only` names."""
        return [k for k in ctx.rigs
                if only is None or any(i.startswith(f"{k} ") for i in only)]

    # -- the check

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        found: list[Finding] = []
        for key in self._keys(ctx, only):
            rig, _ = self._prepare(ctx, key)
            if rig is not None:
                found += _measure(rig.fixed, rig.scene, rig.rig, only, samples, rig.pristine)
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        for key in ctx.rigs:
            rig, said = self._prepare(ctx, key)
            lines += said
            if rig is not None:
                lines.append(f"{NAME}: measured {len(rig.fixed.poses)} poses: {key}")
                why = body_unreadable(rig.scene, rig.scene.clips[STRESS_CLIP])
                if why:
                    lines.append(Unmeasured(f"{NAME}: skipped ({why}): {key} C"))
        return list(dict.fromkeys(lines))

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units: list[PlantUnit] = []
        for key in ctx.rigs:
            rig, _ = self._prepare(ctx, key)
            if rig is not None:
                units += _units(rig)
        return units
