"""Exporters that run inside Blender and write the scene format `core.scene.load_scene` reads.

They write three parts of a scene folder:

    rigs/<pack>/<piece>.json     export_rig
    rigs/<pack>/<piece>.rest.npz export_rig (the body's rest shells)
    pieces/<pack>/<piece>.npz    export_pieces
    samples/<clip>.npz           export_samples

The facts come from the pack, never from loose arguments: a piece's rig (`Piece.rig`) names the
armature, the bones, the contact groups (a group name and the vertex groups whose vertices make
its points), the chains and the outline modifiers; a piece's parts name its meshes; a `Clip`
names the frames and the props. The caller writes `style.json`, `packs/` and `clips.json`.
`save_scene` refuses a folder with files in it, and these exporters refuse to overwrite a file.
So export into a fresh folder: the caller clears it before each run.

Every array is in glTF axes (`axes.BLENDER_TO_GLTF`). A bone or prop matrix maps the bone's own
Blender frame to the glTF world, the same as `core.rig.pose_world` returns it. Mesh points come
from the evaluated mesh (modifiers on). A rig's `outline_modifiers` (and `export_pieces`'s
`hide_modifiers`) name modifiers, such as an outline shell, that are switched off for the export
and switched back on after it.

A vertex belongs to its strongest deform group: a vertex group named after a deform bone of
the armature. Mask and helper groups never take a vertex. Needs Blender 4.4 or later (action
slots).

A rigged piece's body is read part by part: each mesh of the piece (see `_piece_objects`) is a
part (`part_name`), and every part of the piece needs its mesh. The rig file lists, per part,
the deform groups its mesh carries (`vertex_groups`), and `rigs/<pack>/<piece>.rest.npz` holds
the rest-pose shells. Every sample file holds, per part, the posed points (`v:`), the
triangulated faces (`f:`) and each vertex's strongest group as an index into that list (`l:`).
The rest file also holds each part's raw skin weights (`w:`, `j:`), and the rig must be plain
linear blend skinning.
A vertex with no deform weight is refused.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import bpy
import numpy as np

from model_kit.blender import build
from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Clip, Pack, Piece, RigSpec
from model_kit.core.rig import RigData
from model_kit.core.scene import REST_SUFFIX, shell_arrays

_SAFE = re.compile(r"[A-Za-z0-9_.-]+")

# Guards on the adapter's contract with the numpy forward kinematics, not check tolerances.
# Blender stores transforms in float32, so a value counts as equal within these.
UNIT_SCALE_TOL = 1e-6  # a pose scale this close to 1 is unit scale
ZERO_LOCATION_TOL = 1e-9  # a connected bone's location this close to 0 is unmoved
WORLD_DRIFT_TOL = 1e-6  # the armature object's matrix may not move more during a clip
STEP_TOL = 1e-9  # how far `end` may sit off the `step` grid from `start`
MIN_BLENDER = (4, 4, 0)  # action slots


def _file_name(name: str, what: str) -> str:
    if not _SAFE.fullmatch(name) or name in {".", ".."}:
        raise ManifestError(f"{what} {name!r} cannot be a file name")
    return name


def _new_file(path: Path) -> Path:
    if path.exists():
        raise ManifestError(f"{path}: already written; export into a fresh folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _mat(matrix) -> np.ndarray:
    """A mathutils matrix as a float64 array (rows stay rows)."""
    return np.array(matrix, dtype=float)


def _to_gltf(points: np.ndarray) -> np.ndarray:
    """Blender points (..., 3) to glTF axes."""
    return points @ BLENDER_TO_GLTF[:3, :3].T


def _quat_matrix(q: Sequence[float]) -> np.ndarray:
    """The 3x3 turn of a quaternion given as [x, y, z, w]."""
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if q.shape != (4,) or not np.isfinite(norm) or norm == 0:
        raise ManifestError(f"a quaternion is 4 finite numbers, not all 0, got {list(q)}")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _evaluate() -> bpy.types.Depsgraph:
    bpy.context.view_layer.update()
    return bpy.context.evaluated_depsgraph_get()


def _check_hidden(objects: Iterable[bpy.types.Object], names: Collection[str]) -> None:
    """Each name in `hide_modifiers` must match a modifier on at least one exported mesh of
    the whole call (a typo raises; a piece without an outline is fine)."""
    seen = {mod.name for obj in objects for mod in obj.modifiers}
    missing = sorted(set(names) - seen)
    if missing:
        raise ManifestError(f"hide_modifiers names no modifier on the exported meshes: {missing}")


@contextmanager
def _modifiers_off(objects: Iterable[bpy.types.Object], names: Collection[str]) -> Iterator[None]:
    """Switch off the named modifiers for the export and back on after it."""
    names = set(names)
    switched = [mod for obj in objects for mod in obj.modifiers
                if mod.name in names and mod.show_viewport]
    try:
        for mod in switched:
            mod.show_viewport = False
        yield
    finally:
        for mod in switched:
            mod.show_viewport = True


def _armature(obj) -> bpy.types.Object:
    if getattr(obj, "type", None) != "ARMATURE":
        raise ManifestError(f"{getattr(obj, 'name', obj)!r} is not an armature object")
    return obj


def _skinned(armature: bpy.types.Object) -> list[bpy.types.Object]:
    """The meshes an Armature modifier binds to this armature, sorted by name."""
    found = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"
             and any(m.type == "ARMATURE" and m.object == armature for m in obj.modifiers)]
    return sorted(found, key=lambda obj: obj.name)


def _deform_groups(armature: bpy.types.Object) -> set[str]:
    """The vertex group names that can move a vertex: the armature's deform bones."""
    return {bone.name for bone in armature.data.bones if bone.use_deform}


def _strongest(obj: bpy.types.Object, deform: Collection[str]) -> list[str | None]:
    """Each vertex's strongest deform group by name, or None when it has no deform weight.

    Only groups named in `deform` are ranked. A tie in weight goes to the group that comes
    first in `obj.vertex_groups`.
    """
    names = {g.index: g.name for g in obj.vertex_groups if g.name in deform}
    out: list[str | None] = []
    for vert in obj.data.vertices:
        entries = [e for e in vert.groups if e.group in names and e.weight > 0]
        best = max(entries, key=lambda e: (e.weight, -e.group), default=None)
        out.append(None if best is None else names[best.group])
    return out


def _world_points(obj: bpy.types.Object, depsgraph) -> np.ndarray:
    """The evaluated mesh's vertices in Blender world space, in the base mesh's order."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        n = len(mesh.vertices)
        if n != len(obj.data.vertices):
            raise ManifestError(
                f"mesh {obj.name!r}: its modifiers change the vertex count "
                f"({len(obj.data.vertices)} to {n}); name the shell in hide_modifiers")
        co = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
    finally:
        evaluated.to_mesh_clear()
    m = _mat(evaluated.matrix_world)
    return co.reshape(-1, 3).astype(float) @ m[:3, :3].T + m[:3, 3]


def _members(meshes: list[bpy.types.Object], vertex_groups: Collection[str],
             deform: Collection[str]) -> list[np.ndarray]:
    """Per mesh, the indices of vertices whose strongest deform group is in `vertex_groups`."""
    wanted = set(vertex_groups)
    return [np.flatnonzero([name in wanted for name in _strongest(obj, deform)])
            for obj in meshes]


# --- body shells ----------------------------------------------------------------------------


def _name_meshes(piece: Piece, meshes: Iterable[bpy.types.Object]
                 ) -> tuple[dict[str, bpy.types.Object], list[str]]:
    """Map a piece's meshes to its part names (`part_name`), and list the problems: two objects
    of one part, a mesh that is no part, a part with no mesh."""
    named: dict[str, bpy.types.Object] = {}
    problems = []
    for obj in meshes:
        part = part_name(piece.name, obj.name)
        if part in named:
            problems.append(f"piece {piece.name!r}: objects {named[part].name!r} and "
                            f"{obj.name!r} are both part {part!r}")
        named[part] = obj
    parts = {p.name for p in piece.parts}
    problems += [f"piece {piece.name!r}: mesh {obj.name!r} is part {part!r}, which the pack "
                 "does not list" for part, obj in named.items() if part not in parts]
    problems += [f"piece {piece.name!r}: part {part!r} has no mesh"
                 for part in sorted(parts - set(named))]
    return named, problems


def _body_meshes(piece: Piece) -> dict[str, bpy.types.Object]:
    """The mesh of every part of a rigged piece, by part name. A mesh that is no part, or a
    part with no mesh, raises."""
    _, meshes = _piece_objects(piece)
    named, problems = _name_meshes(piece, meshes)
    if problems:
        raise ManifestError("; ".join(problems))
    return dict(sorted(named.items()))


def _label_groups(obj: bpy.types.Object, deform: Collection[str]) -> tuple[str, ...]:
    """The mesh's deform vertex groups, in group order: what its labels index into."""
    return tuple(g.name for g in obj.vertex_groups if g.name in deform)


def _labels(part: str, obj: bpy.types.Object, deform: Collection[str],
            groups: Sequence[str], who: str) -> np.ndarray:
    """Each vertex's strongest deform group as an index into `groups`. A vertex with no deform
    weight, or whose strongest group is not in `groups`, raises."""
    names = _strongest(obj, deform)
    bare = sum(name is None for name in names)
    if bare:
        raise ManifestError(f"{who}: part {part!r}: {bare} of {len(names)} vertices have no "
                            "deform weight; every vertex of a body part needs a deform group")
    index = {name: i for i, name in enumerate(groups)}
    lost = sorted({name for name in names if name not in index})
    if lost:
        raise ManifestError(f"{who}: part {part!r}: the strongest groups {lost} are not in the "
                            f"rig file's vertex_groups {list(groups)}; export_rig again")
    return np.array([index[name] for name in names], dtype=np.int32)


def _skin(named: Mapping[str, bpy.types.Object], groups: Mapping[str, Sequence[str]]
          ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Each part's raw skin weights, as Blender stores them: (V, K) float32 weights and (V, K)
    int32 indices into `groups[part]`, K the most groups any vertex of the part weighs. A vertex
    with fewer is padded with weight 0 and index 0. A weight at or under 0 is no influence and
    is left out. The weights are as stored in the vertex groups: the bind step already floored
    what it wrote (`BIND_WEIGHT_FLOOR`), and nothing is floored or renormalised here. The skinner
    renormalises each vertex to sum 1, as Blender's Armature modifier does."""
    weights: dict[str, np.ndarray] = {}
    indices: dict[str, np.ndarray] = {}
    for part, obj in named.items():
        table = {name: i for i, name in enumerate(groups[part])}
        slot = {g.index: table[g.name] for g in obj.vertex_groups if g.name in table}
        rows = [sorted((slot[e.group], e.weight) for e in vert.groups
                       if e.group in slot and e.weight > 0) for vert in obj.data.vertices]
        k = max(1, max((len(row) for row in rows), default=1))
        w = np.zeros((len(rows), k), dtype=np.float32)
        j = np.zeros((len(rows), k), dtype=np.int32)
        for v, row in enumerate(rows):
            for n, (group, weight) in enumerate(row):
                j[v, n], w[v, n] = group, weight
        weights[part], indices[part] = w, j
    return weights, indices


def _faces(part: str, obj: bpy.types.Object) -> np.ndarray:
    """The base mesh's triangles, once, as (T, 3) int32 vertex indices."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tri)
    if not len(tri):
        raise ManifestError(f"part {part!r}: the mesh has no triangles")
    return tri.reshape(-1, 3)


def _body(piece: Piece, named: Mapping[str, bpy.types.Object], deform: Collection[str],
          who: str, groups: Mapping[str, Sequence[str]] | None = None):
    """(meshes, groups, labels, faces) of a rigged piece's body, each keyed by part name.
    `named` is `_body_meshes(piece)`. `groups` are the rig file's vertex groups per part;
    without them each mesh's own are used."""
    if groups is None:
        groups = {part: _label_groups(obj, deform) for part, obj in named.items()}
    elif set(groups) != set(named):
        raise ManifestError(f"{who}: the rig file's vertex_groups are for parts {sorted(groups)}"
                            f", but the piece's meshes are parts {sorted(named)}; export_rig "
                            "again")
    labels = {part: _labels(part, obj, deform, groups[part], who) for part, obj in named.items()}
    faces = {part: _faces(part, obj) for part, obj in named.items()}
    return named, dict(groups), labels, faces


# --- rig ------------------------------------------------------------------------------------


def _weighed(armature: bpy.types.Object, bone: str) -> bool:
    """Whether any skinned vertex has a weight above 0 in the vertex group named `bone`."""
    for obj in _skinned(armature):
        group = obj.vertex_groups.get(bone)
        if group is not None and any(e.group == group.index and e.weight > 0
                                     for vert in obj.data.vertices for e in vert.groups):
            return True
    return False


def _not_inert(armature: bpy.types.Object, bone: bpy.types.Bone,
               ridden: Mapping[str, str]) -> str | None:
    """Why a scaled bone is not inert, or None when it is: a scaled bone is allowed only with
    no child bone, no skinned vertex that weighs it and no clip prop that rides it."""
    if bone.children:
        return f"it has child bones {[c.name for c in bone.children]}"
    if _weighed(armature, bone.name):
        return "a skinned weight is on it"
    if bone.name in ridden:
        return f"the clip's prop {ridden[bone.name]!r} rides it"
    return None


def _check_bones(armature: bpy.types.Object, when: str = "", *, pose: bool = True,
                 ridden: Mapping[str, str] | None = None,
                 inert: dict[str, str | None] | None = None) -> None:
    """The numpy forward kinematics (`core.rig.pose_world`) holds only for these bones.

    The contract has two parts. The rig part reads no pose value: rotation_mode 'XYZ', no active
    constraint, full inheritance, and a local location. A muted constraint, or one at influence
    0, does not move a bone and is allowed. `export_rig` checks only this part (`pose=False`),
    because the rig file does not depend on the pose the armature happens to be in. The rig
    part still reads constraint mute and influence, and rotation_mode, at the current frame:
    an action that keys one of them can make `export_rig` refuse on one frame and pass on
    another. `export_samples` checks them again at every sampled frame.

    The pose part reads the current pose values (`pose=True`, the default, which checks both
    parts): a bone's scale must be 1 and a connected bone must not be moved. A scaled bone is
    allowed when it is inert: no child bone, no skinned weight on it, and no prop of the clip
    rides it (`ridden` maps a bone to the id of a prop on it). Each such bone prints one line,
    once per `inert`, the verdicts that the caller keeps for the whole export. `export_samples`
    runs the whole check at every sampled frame.
    """
    ridden = ridden or {}
    verdicts = inert if inert is not None else {}
    problems = []
    for pb in armature.pose.bones:
        bone = pb.bone
        if pb.rotation_mode != "XYZ":
            problems.append(f"{pb.name!r}: rotation_mode is {pb.rotation_mode!r}, not 'XYZ'")
        if pose and any(abs(s - 1.0) > UNIT_SCALE_TOL for s in pb.scale):
            if pb.name not in verdicts:
                verdicts[pb.name] = _not_inert(armature, bone, ridden)
                if verdicts[pb.name] is None:
                    print(f"inert scaled bone {pb.name!r}: pose scale {tuple(pb.scale)}; no "
                          "child bone, no skinned weight and no clip prop on it")
            if verdicts[pb.name] is not None:
                problems.append(f"{pb.name!r}: pose scale is {tuple(pb.scale)}, not 1, and "
                                f"the bone is not inert: {verdicts[pb.name]}")
        active = [c.name for c in pb.constraints if not c.mute and c.influence > 0]
        if active:
            problems.append(f"{pb.name!r}: has active constraints {active}")
        if not bone.use_inherit_rotation or bone.inherit_scale != "FULL":
            problems.append(f"{pb.name!r}: must inherit its parent's rotation and full scale")
        if not bone.use_local_location:
            problems.append(f"{pb.name!r}: location must be local (use_local_location)")
        if pose and bone.use_connect and any(abs(v) > ZERO_LOCATION_TOL for v in pb.location):
            problems.append(f"{pb.name!r}: is connected, so Blender ignores its location "
                            f"{tuple(pb.location)}")
    if problems:
        raise ManifestError(f"armature {armature.name!r}{when} breaks the forward-kinematics "
                            "contract: " + "; ".join(problems))


def _bone_order(armature: bpy.types.Object) -> list[str]:
    """Bone names, parents before children."""
    order: list[str] = []
    stack = [b for b in reversed(armature.data.bones) if b.parent is None]
    while stack:
        bone = stack.pop()
        order.append(bone.name)
        stack.extend(reversed(bone.children))
    return order


# The Armature modifier settings of plain linear blend skinning, as Blender names them.
_PLAIN_BLEND = {"use_deform_preserve_volume": False, "use_multi_modifier": False,
                "use_vertex_groups": True, "use_bone_envelopes": False}


def _check_shape_keys(obj: bpy.types.Object, who: str) -> None:
    """A mesh may carry shape keys only when every one of them rests at value 0. At value 0 a
    key moves nothing, so the rest shell and the stored weights are the base mesh's, as the kit
    skins them. A key above or below 0 at rest moves vertices outside the weights and raises.
    `use_relative` off raises too: the keys then blend by `eval_time` and ignore their values.
    `export_samples` plays the clip's KEY slot on top of this rest (see `_rest_shape_keys`)."""
    key = obj.data.shape_keys
    if key is None:
        return
    if not key.use_relative:
        raise ManifestError(f"{who}: mesh {obj.name!r} has shape keys with use_relative off; "
                            "they blend by eval_time, not by value, so rest is not the base mesh")
    for block in key.key_blocks:
        if block.name != key.reference_key.name and block.value != 0.0:
            raise ManifestError(
                f"{who}: mesh {obj.name!r} has shape key {block.name!r} at rest value "
                f"{block.value:g}; a shape key must rest at 0, or it moves vertices outside the "
                "weights")


def _check_skinning(armature: bpy.types.Object, named: Mapping[str, bpy.types.Object],
                    outline: Collection[str], who: str) -> None:
    """Refuse a rig that is not plain linear blend skinning: the kit skins each vertex by its
    bones' posed matrices and its stored weights, so anything that bends a vertex another way
    would pose the shells differently from Blender.

    Refused: a deforming bone with B-Bone segments (`bbone_segments` above 1); a mesh with a
    shape key at a rest value other than 0 (`_check_shape_keys`); an Armature modifier with
    `use_deform_preserve_volume` or `use_multi_modifier` on, `use_vertex_groups` off,
    `use_bone_envelopes` on or a `vertex_group` mask; and any modifier after the Armature
    modifier other than the rig's `outline` modifiers (they are switched off for the export). A
    body mesh with no Armature modifier to this armature raises too, since its weights would
    never be applied."""
    bent = sorted(b.name for b in armature.data.bones if b.use_deform and b.bbone_segments > 1)
    if bent:
        raise ManifestError(f"{who}: deforming bones {bent} have bbone_segments above 1; "
                            "a B-Bone bends a vertex by more than one bone matrix")
    for obj in named.values():
        _check_shape_keys(obj, who)
        mods = [m for m in obj.modifiers if m.type == "ARMATURE" and m.object == armature]
        if not mods:
            raise ManifestError(f"{who}: mesh {obj.name!r} has no Armature modifier to "
                                f"{armature.name!r}")
        for mod in mods:
            for setting, plain in _PLAIN_BLEND.items():
                if getattr(mod, setting) != plain:
                    raise ManifestError(
                        f"{who}: mesh {obj.name!r}: Armature modifier {mod.name!r} has "
                        f"{setting} {not plain}; the kit poses with plain linear blend skinning")
            if mod.vertex_group:
                raise ManifestError(
                    f"{who}: mesh {obj.name!r}: Armature modifier {mod.name!r} is masked by "
                    f"vertex group {mod.vertex_group!r}; the kit poses with plain linear blend "
                    "skinning")
        stack = list(obj.modifiers)
        after = [m.name for m in stack[stack.index(mods[0]) + 1:] if m.name not in outline]
        if after:
            raise ManifestError(
                f"{who}: mesh {obj.name!r}: modifiers {after} come after the Armature modifier "
                f"{mods[0].name!r}; only the rig's outline_modifiers {sorted(outline)} may")


def _rig_spec(pack: Pack, piece: str) -> RigSpec:
    """The rig of a rigged piece of `pack`. A missing piece, or one with no rig, raises."""
    found = pack.pieces.get(piece)
    if found is None:
        raise ManifestError(f"pack {pack.name!r} has no piece {piece!r}; its pieces are "
                            f"{sorted(pack.pieces)}")
    if found.rig is None:
        raise ManifestError(f"piece {pack.name}:{piece} has no rig in the pack")
    return found.rig


def _check_armature(armature: bpy.types.Object, rig: RigSpec, who: str) -> None:
    """The armature's name must be the rig's `armature`."""
    if armature.name != rig.armature:
        raise ManifestError(f"{who}: the armature object is {armature.name!r}, but the pack's "
                            f"rig names {rig.armature!r}")


def _check_rig_bones(armature: bpy.types.Object, rig: RigSpec, who: str) -> None:
    """The armature's bone names and parents must be the rig's `bones`."""
    got = {b.name: (b.parent.name if b.parent else None) for b in armature.data.bones}
    want = {b.name: b.parent for b in rig.bones}
    if got == want:
        return
    extra, lacking = sorted(set(got) - set(want)), sorted(set(want) - set(got))
    moved = sorted(b for b in set(got) & set(want) if got[b] != want[b])
    raise ManifestError(f"{who}: the armature's bones differ from the pack's rig: only in the "
                        f"armature {extra}, only in the pack {lacking}, with another parent "
                        f"{moved}")


def _check_contact_groups(armature: bpy.types.Object, rig: RigSpec,
                          meshes: list[bpy.types.Object], who: str) -> None:
    """Each entry of each contact group must be a deform bone that is the strongest group of at
    least one skinned vertex. Otherwise the group's points would silently leave it out."""
    deform = _deform_groups(armature)
    owned = {name for obj in meshes for name in _strongest(obj, deform) if name is not None}
    problems = []
    for group, entries in rig.contact_groups.items():
        for entry in entries:
            if entry not in deform:
                problems.append(f"contact group {group!r}: {entry!r} is not a deform bone of "
                                "the armature")
            elif entry not in owned:
                problems.append(f"contact group {group!r}: no vertices have {entry!r} as their "
                                "strongest deform group")
    if problems:
        raise ManifestError(f"{who}: " + "; ".join(problems))


def _rig_data(armature, effector_vertex_groups: Mapping[str, Sequence[str]], *,
              hide_modifiers: Collection[str]) -> RigData:
    """The rig of `armature` in the scene format: the shared body of `export_rig`.

    `effector_vertex_groups` maps an effector bone to the vertex groups whose vertices make its
    `effector_mid`: the rest-pose centre of the skinned vertices whose strongest deform group is
    one of them, in the bone's rest frame. `hide_modifiers` are switched off while it measures.

    Every pose bone must use rotation_mode 'XYZ', have no active constraints, full inheritance
    and a local location. Otherwise `pose_world` would not match Blender, so this raises. It
    reads no pose value (`_check_bones` with `pose=False`): the rest matrices and the effector
    mids come from the rest pose, so the pose the armature is in does not matter. Scale and
    connected-bone checks belong to `export_samples`, which makes them at every sampled frame.
    """
    armature = _armature(armature)
    _check_bones(armature, pose=False)
    meshes = _skinned(armature)
    _check_hidden(meshes, hide_modifiers)
    order = _bone_order(armature)
    bones = armature.data.bones
    rest = {name: _mat(bones[name].matrix_local) for name in order}
    arm_world = _mat(armature.matrix_world)
    mids: dict[str, np.ndarray] = {}
    if effector_vertex_groups:
        deform = _deform_groups(armature)
        saved = armature.data.pose_position
        try:
            armature.data.pose_position = "REST"
            with _modifiers_off(meshes, hide_modifiers):
                depsgraph = _evaluate()
                for bone, groups in effector_vertex_groups.items():
                    if bone not in rest:
                        raise ManifestError(f"effector bone {bone!r} is not in the armature")
                    picked = [_world_points(obj, depsgraph)[idx]
                              for obj, idx in zip(meshes, _members(meshes, groups, deform),
                                                  strict=True)
                              if len(idx)]
                    if not picked:
                        raise ManifestError(f"effector {bone!r}: no skinned vertex has one of "
                                            f"{list(groups)} as its strongest deform group")
                    centre = np.append(np.concatenate(picked).mean(axis=0), 1.0)
                    mids[bone] = (np.linalg.inv(rest[bone]) @ np.linalg.inv(arm_world)
                                  @ centre)[:3]
        finally:
            armature.data.pose_position = saved
            _evaluate()
    return RigData(order=tuple(order),
                   parent={b: (bones[b].parent.name if bones[b].parent else None)
                           for b in order},
                   rest=rest, world=BLENDER_TO_GLTF @ arm_world, effector_mid=mids)


def _write_rig(rig: RigData, path: Path, clips: Sequence[str],
               vertex_groups: Mapping[str, Sequence[str]] | None = None) -> Path:
    """Write a rig file at `path` (a fresh file) and return `path`. `clips` are the clips that
    pose this rig; `vertex_groups` are the groups per part that the sample labels index into."""
    if len(set(clips)) != len(clips):
        raise ManifestError(f"{path}: clips repeat a name: {list(clips)}")
    _new_file(path).write_text(json.dumps({
        "order": list(rig.order),
        "parent": rig.parent,
        "rest": {b: m.tolist() for b, m in rig.rest.items()},
        "world": rig.world.tolist(),
        "effector_mid": {b: [float(x) for x in v] for b, v in rig.effector_mid.items()},
        "clips": sorted(clips),
        "vertex_groups": {part: list(g) for part, g in (vertex_groups or {}).items()},
    }, indent=2) + "\n")
    return path


def _rest_shells(armature: bpy.types.Object, named: Mapping[str, bpy.types.Object],
                 hide_modifiers: Collection[str]) -> dict[str, np.ndarray]:
    """Each part's points at the rest pose, as (1, V, 3) float32 in glTF world."""
    saved = armature.data.pose_position
    try:
        armature.data.pose_position = "REST"
        with _modifiers_off(named.values(), hide_modifiers):
            depsgraph = _evaluate()
            return {part: _to_gltf(_world_points(obj, depsgraph))[None].astype(np.float32)
                    for part, obj in named.items()}
    finally:
        armature.data.pose_position = saved
        _evaluate()


def export_rig(armature, pack: Pack, piece: str, out: Path, *, clips: Sequence[str]) -> Path:
    """Write `rigs/<pack>/<piece>.json` under the scene folder `out`. Returns the file's path.

    The rig is `pack.pieces[piece].rig`. The piece must have one, the armature object must be
    named `rig.armature`, and its bone names and parents must be `rig.bones`; otherwise this
    raises and writes nothing. `clips` are the clips that pose this rig.

    Each chain's `effector_mid` is the rest-pose centre of the skinned vertices whose strongest
    deform group is one of `rig.contact_groups[chain.effector_group]`, in the effector bone's
    rest frame. `rig.outline_modifiers` are switched off while it measures. Every contact group
    entry must be a deform bone that is the strongest group of at least one skinned vertex.

    The rig part of the bone contract of `_rig_data` holds too (XYZ rotation, no active
    constraints, full inheritance, local location). The armature's current pose is not read: a
    scaled or moved bone does not stop this. `export_samples` checks the pose values at every
    sampled frame (see `_check_bones`).

    The piece's body is read at the rest pose: every part of the piece needs its mesh (a mesh
    that is no part raises), and every vertex a deform weight. The rig file lists the deform
    groups of each part (`vertex_groups`), and `rigs/<pack>/<piece>.rest.npz` holds, per part,
    `v:` (1, V, 3) float32 rest points in glTF world, `f:` (T, 3) int32 faces, `l:` (V,)
    int32 labels, each vertex's strongest group as an index into that list, and the raw skin:
    `w:` (V, K) float32 weights as Blender stores them and `j:` (V, K) int32 indices into the
    same list (see `_skin`). The rig must be plain linear blend skinning (`_check_skinning`).
    """
    rig = _rig_spec(pack, piece)
    who = f"{pack.name}:{piece}"
    effectors: dict[str, tuple[str, ...]] = {}
    for name, chain in rig.chains.items():
        if chain.effector_group not in rig.contact_groups:
            raise ManifestError(f"{who}: chain {name!r} has effector group "
                                f"{chain.effector_group!r}, which is no contact group of the rig")
        groups = rig.contact_groups[chain.effector_group]
        if effectors.setdefault(chain.effector, groups) != groups:
            raise ManifestError(f"{who}: chain {name!r} and another chain both end at "
                                f"{chain.effector!r} with different contact groups")
    armature = _armature(armature)
    _check_armature(armature, rig, who)
    _check_rig_bones(armature, rig, who)
    _check_contact_groups(armature, rig, _skinned(armature), who)
    meshes = _body_meshes(pack.pieces[piece])
    _check_skinning(armature, meshes, rig.outline_modifiers, who)
    named, groups, labels, faces = _body(pack.pieces[piece], meshes, _deform_groups(armature),
                                         who)
    weights, weight_groups = _skin(named, groups)
    folder = Path(out) / "rigs" / _file_name(pack.name, "pack")
    path = folder / f"{_file_name(piece, 'piece')}.json"
    rest_path = folder / f"{_file_name(piece, 'piece')}{REST_SUFFIX}"
    data = _rig_data(armature, effectors, hide_modifiers=rig.outline_modifiers)
    shells = _rest_shells(armature, named, rig.outline_modifiers)
    if rest_path.exists():
        raise ManifestError(f"{rest_path}: already written; export into a fresh folder")
    _write_rig(data, path, clips, groups)
    np.savez_compressed(_new_file(rest_path),
                        **shell_arrays(shells, faces, labels, weights, weight_groups))
    return path


# --- pieces ---------------------------------------------------------------------------------


def _triangles(obj: bpy.types.Object, depsgraph, to_piece: np.ndarray) -> np.ndarray:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        tri = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int64)
        mesh.loop_triangles.foreach_get("vertices", tri)
    finally:
        evaluated.to_mesh_clear()
    if not len(tri):
        raise ManifestError(f"part {obj.name!r}: the mesh has no triangles")
    m = to_piece @ _mat(evaluated.matrix_world)
    points = co.reshape(-1, 3).astype(float) @ m[:3, :3].T + m[:3, 3]
    return points[tri.reshape(-1, 3)]


def part_name(piece: str, object_name: str) -> str:
    """The pack.json part name of a piece's object: `object_name` without the piece's
    `build.object_name` prefix (`<piece>/`), when it has one."""
    prefix = build.object_name(piece, "")
    return object_name.removeprefix(prefix)


def _piece_objects(piece: Piece) -> tuple[bpy.types.Object, list[bpy.types.Object]]:
    """A piece's root object and its meshes, sorted by name.

    An unrigged piece's root is the object named for the piece, and its meshes are the root
    itself when it is a mesh, plus its mesh descendants. A rigged piece's root is the armature
    object `rig.armature` (it need not share the piece's name), and its meshes are the meshes
    skinned to that armature plus the root's mesh descendants.
    """
    name = piece.rig.armature if piece.rig is not None else piece.name
    root = bpy.data.objects.get(name)
    if root is None:
        what = "armature" if piece.rig is not None else "root"
        raise ManifestError(f"piece {piece.name!r}: no {what} object named {name!r} in the scene")
    if piece.rig is not None:
        _armature(root)
    meshes = {obj.name: obj for obj in (root, *root.children_recursive) if obj.type == "MESH"}
    if piece.rig is not None:
        meshes.update({obj.name: obj for obj in _skinned(root)})
    return root, [meshes[n] for n in sorted(meshes)]


def export_pieces(pack: Pack, out_dir: Path, *, pieces: Sequence[str],
                  hide_modifiers: Collection[str] = ()) -> list[Path]:
    """Write `pieces/<pack>/<piece>.npz` under the scene folder `out_dir`, one per name in
    `pieces`. Each name must be a piece of `pack`.

    A piece's parts are its meshes (see `_piece_objects` for where they are), as (N, 3, 3)
    triangles in the root's frame in glTF axes. Each mesh is keyed by its part name, the name
    pack.json uses: a `PackBuilder` object `<piece>/<part>` (`build.object_name`) loses its
    `<piece>/` prefix, and any other object keeps its name.

    The meshes and the pack's parts must agree. Before anything is written this raises, naming
    each, on a mesh that is no part of its piece, on a part that has no mesh, and on two
    objects of one piece that come to one part name. A rigged piece's `rig.outline_modifiers`
    are switched off with `hide_modifiers`, and switched back on after.
    """
    if len(set(pieces)) != len(pieces):
        raise ManifestError(f"pieces repeat a name: {list(pieces)}")
    folder = Path(out_dir) / "pieces" / _file_name(pack.name, "pack")
    found = []
    problems = []
    for name in pieces:
        if name not in pack.pieces:
            raise ManifestError(f"pack {pack.name!r} has no piece {name!r}; its pieces are "
                                f"{sorted(pack.pieces)}")
        piece = pack.pieces[name]
        root, meshes = _piece_objects(piece)
        named, found_problems = _name_meshes(piece, meshes)
        problems += found_problems
        found.append((piece, root, named))
    if problems:
        raise ManifestError("; ".join(problems))
    _check_hidden([obj for _, _, named in found for obj in named.values()], hide_modifiers)
    for piece, _, named in found:
        if piece.rig is not None and piece.rig.outline_modifiers:
            _check_hidden(named.values(), piece.rig.outline_modifiers)
    made = []
    for piece, root, named in found:
        path = folder / f"{_file_name(piece.name, 'piece')}.npz"
        if path.exists():
            raise ManifestError(f"{path}: already written; export into a fresh folder")
        hidden = {*hide_modifiers, *(piece.rig.outline_modifiers if piece.rig else ())}
        with _modifiers_off(named.values(), hidden):
            depsgraph = _evaluate()
            to_piece = BLENDER_TO_GLTF @ np.linalg.inv(_mat(root.evaluated_get(depsgraph)
                                                            .matrix_world))
            arrays = {part: _triangles(obj, depsgraph, to_piece) for part, obj in named.items()}
        _evaluate()
        made.append((path, arrays))
    for path, arrays in made:  # every piece is read before the first file is written
        np.savez_compressed(_new_file(path), **arrays)
    return [path for path, _ in made]


# --- samples --------------------------------------------------------------------------------


def _sample_times(start: float, end: float, step: float) -> np.ndarray:
    if not step > 0 or end < start:
        raise ManifestError(f"samples need step > 0 and end >= start, got {start}..{end} "
                            f"by {step}")
    n = round((end - start) / step)
    if abs(start + n * step - end) > STEP_TOL:
        raise ManifestError(f"samples: {start}..{end} is not a whole number of {step} steps")
    return start + step * np.arange(n + 1)


def _use_action(armature: bpy.types.Object, action: bpy.types.Action) -> None:
    if tuple(bpy.app.version) < MIN_BLENDER:
        raise ManifestError(f"export_samples needs Blender {'.'.join(map(str, MIN_BLENDER))} or "
                            f"later (action slots), this is {bpy.app.version_string}")
    data = armature.animation_data_create()
    data.action = action
    if data.action_slot is None:
        slots = [s for s in action.slots if s.target_id_type in {"OBJECT", "UNSPECIFIED"}]
        if len(slots) != 1:
            raise ManifestError(f"action {action.name!r}: cannot pick a slot for "
                                f"{armature.name!r} from {[s.identifier for s in slots]}")
        data.action_slot = slots[0]


def _pose_state(armature: bpy.types.Object) -> dict[str, tuple]:
    return {pb.name: (tuple(pb.location), tuple(pb.rotation_quaternion),
                      tuple(pb.rotation_euler), tuple(pb.rotation_axis_angle),
                      tuple(pb.scale)) for pb in armature.pose.bones}


@contextmanager
def _rest_pose_no_nla(armature: bpy.types.Object) -> Iterator[None]:
    """Reset every pose bone to rest and switch the armature's NLA off, so a channel the
    sampled clip does not key reads rest and no track or strip plays. The switch is the
    object's `animation_data.use_nla`; the tracks' own mute and solo flags are never touched.
    On exit (a raise included) the switch and every pose bone's values are put back."""
    pose = _pose_state(armature)
    data = armature.animation_data
    use_nla = data.use_nla if data else None
    try:
        for pb in armature.pose.bones:
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
        if data:
            data.use_nla = False
        yield
    finally:
        if data and use_nla is not None:
            data.use_nla = use_nla
        for pb in armature.pose.bones:
            (pb.location, pb.rotation_quaternion, pb.rotation_euler, pb.rotation_axis_angle,
             pb.scale) = pose[pb.name]
        bpy.context.view_layer.update()


def _shape_keys(objects: Iterable[bpy.types.Object]) -> list[bpy.types.Key]:
    """The shape-key datablocks of `objects`, each once (meshes may share one)."""
    found = {}
    for obj in objects:
        key = obj.data.shape_keys
        if key is not None:
            found[key.name] = key
    return list(found.values())


@contextmanager
def _rest_shape_keys(keys: Sequence[bpy.types.Key]) -> Iterator[None]:
    """Put every shape key at rest, as `_rest_pose_no_nla` does the bones: each block (the
    reference key aside) at value 0, each key's own action taken off and its NLA switched off,
    so a block the sampled clip does not key reads rest. On exit (a raise included) the values,
    the NLA switches and each key's action and slot are put back; a key that had no animation
    data has none again."""
    saved = []
    for key in keys:
        data = key.animation_data
        saved.append((key, data is not None, data.action if data else None,
                      data.action_slot if data else None, data.use_nla if data else None,
                      {b.name: b.value for b in key.key_blocks}))
    try:
        for key in keys:
            for block in key.key_blocks:
                if block.name != key.reference_key.name:
                    block.value = 0.0
            if key.animation_data:
                key.animation_data.use_nla = False
                key.animation_data.action = None  # its own motion is not the clip's
        yield
    finally:
        for key, had_data, action, slot, use_nla, values in saved:
            if had_data:
                key.animation_data.action = action
                if action is not None and slot is not None:
                    key.animation_data.action_slot = slot
                key.animation_data.use_nla = use_nla
            elif key.animation_data:
                key.animation_data_clear()
            for name, value in values.items():
                key.key_blocks[name].value = value


_KEY_BLOCK = re.compile(r'key_blocks\["([^"]+)"\]')


def _slot_blocks(action: bpy.types.Action, slot) -> set[str]:
    """The shape-key blocks that the curves of KEY slot `slot` of `action` animate."""
    blocks = set()
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot) if strip.type == "KEYFRAME" else None
            for fc in bag.fcurves if bag else ():
                found = _KEY_BLOCK.match(fc.data_path)
                if found:
                    blocks.add(found[1])
    return blocks


def _pair_key_slots(keys: Sequence[bpy.types.Key], action: bpy.types.Action
                    ) -> dict[str, bpy.types.ActionSlot]:
    """Which KEY slot of `action` animates which shape key, as {key name: slot}. A key with no
    slot is left out and stays at rest. Read this before `_rest_shape_keys`, which takes the
    keys' own actions off.

    A slot is paired by the first of three kinds of evidence that names a key:
    1. Blender's own record: the key's animation data holds this action and slot, or the slot's
       `users()` lists the key.
    2. The slot's identifier is "KE" and the key's name (how Blender names it when it is made).
    3. The slot's curves animate blocks that the key has, and no other sampled key has them all.
       A key renamed after its slot was made is paired this way. A slot named "KE" and the name
       of a key that exists in the file but is not sampled belongs to that key, so it is never
       paired this way.
    Evidence of a weaker kind is used only for slots and keys that a stronger kind left over. A
    slot that names a key of the sampled body by no kind of evidence keys a mesh outside the
    body and is ignored. Two slots for one key, one slot that fits two keys, and a slot that
    animates some blocks of a key that has no slot but not all of them, are not told apart from
    a mismatch, so they raise."""
    slots = [s for s in action.slots if s.target_id_type == "KEY"]
    blocks = {s.identifier: _slot_blocks(action, s) for s in slots}
    have = {k.name: {b.name for b in k.key_blocks} for k in keys}
    sampled = {k.name for k in keys}
    elsewhere = {f"KE{k.name}" for k in bpy.data.shape_keys if k.name not in sampled}

    def recorded(key, slot) -> bool:
        data = key.animation_data
        if data and data.action == action and data.action_slot == slot:
            return True
        return any(u.as_pointer() == key.as_pointer() for u in slot.users())

    kinds = (
        recorded,
        lambda key, slot: slot.identifier == f"KE{key.name}",
        lambda key, slot: slot.identifier not in elsewhere and bool(blocks[slot.identifier])
        and blocks[slot.identifier] <= have[key.name],
    )
    paired: dict[str, bpy.types.ActionSlot] = {}
    left = list(slots)
    for kind in kinds:
        fits = {s.identifier: [k for k in keys if k.name not in paired and kind(k, s)]
                for s in left}
        for ident, found in fits.items():
            if len(found) > 1:
                raise ManifestError(
                    f"action {action.name!r}: KEY slot {ident!r} fits shape keys "
                    f"{sorted(k.name for k in found)}; record which one it animates by "
                    "assigning the slot to that key")
        owners: dict[str, list[str]] = {}
        for ident, found in fits.items():
            for k in found:
                owners.setdefault(k.name, []).append(ident)
        for name, idents in owners.items():
            if len(idents) > 1:
                raise ManifestError(
                    f"action {action.name!r}: KEY slots {sorted(idents)} both fit shape key "
                    f"{name!r}; record which one it animates by assigning the slot to it")
            paired[name] = next(s for s in left if s.identifier == idents[0])
        left = [s for s in left if s not in paired.values()]
    for slot in (s for s in left if s.identifier not in elsewhere):
        partial = sorted(k.name for k in keys
                         if k.name not in paired and blocks[slot.identifier] & have[k.name])
        if partial:
            raise ManifestError(
                f"action {action.name!r}: KEY slot {slot.identifier!r} animates blocks "
                f"{sorted(blocks[slot.identifier])}, and shape keys {partial} have some of them "
                "but no slot; name each KEY slot 'KE' and its key's name, or assign it to its key")
    return paired


def _use_action_on_keys(keys: Sequence[bpy.types.Key], action: bpy.types.Action,
                        paired: Mapping[str, bpy.types.ActionSlot]) -> None:
    """Play `action` on each key that has a slot in `paired` (see `_pair_key_slots`)."""
    for key in keys:
        slot = paired.get(key.name)
        if slot is not None:
            data = key.animation_data_create()
            data.action = action
            data.action_slot = slot


def _rig_world(out: Path, pack: str, piece: str, clip: str) -> tuple[str, np.ndarray]:
    """(`rigs/<pack>/<piece>.json`, its armature world in Blender axes) of the rig file that
    `export_rig` wrote for this piece. It must exist and its `clips` must name `clip`."""
    rig_file = f"rigs/{pack}/{piece}.json"
    path = out / rig_file
    if not path.is_file():
        raise ManifestError(
            f"clip {clip!r}: no rig file {path}; export_rig with the clip in `clips` first, once")
    try:
        raw = json.loads(path.read_text())
        clips = raw["clips"]
        world = np.asarray(raw["world"], dtype=float)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"{path}: cannot read the rig file: {exc!r}") from exc
    if clip not in clips:
        raise ManifestError(f"clip {clip!r}: the rig file {path} poses {list(clips)}, not this "
                            "clip; export_rig with the clip in `clips` first, once")
    if world.shape != (4, 4):
        raise ManifestError(f"{rig_file}: world must be 4x4, got {world.shape}")
    return rig_file, np.linalg.inv(BLENDER_TO_GLTF) @ world


def _rig_groups(out: Path, pack: str, piece: str, clip: str) -> dict[str, tuple[str, ...]]:
    """The `vertex_groups` of the rig file that `export_rig` wrote: per part, the groups that the
    sample labels index into. A file without them raises."""
    rig_file = f"rigs/{pack}/{piece}.json"
    try:
        found = json.loads((out / rig_file).read_text())["vertex_groups"]
        return {str(part): tuple(names) for part, names in found.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ManifestError(f"clip {clip!r}: {rig_file} holds no vertex_groups ({exc!r}); "
                            "export_rig again") from exc


def export_samples(armature, action, pack: Pack, piece: str, clip: Clip, out: Path, *,
                   step: float = 0.5) -> Path:
    """Write `samples/<clip.name>.npz` under the scene folder `out`. Returns the file's path.

    The rig is `pack.pieces[piece].rig` (as in `export_rig`: the piece must have one and the
    armature object must be named `rig.armature`). Samples run over `clip.frames` (frame
    numbers, both ends included) every `step` frames. Each sample holds:

    - `g:<group>`: for every contact group of the rig, the points of every skinned vertex
      whose strongest vertex group is one of the group's vertex groups. Each of those must be
      a deform bone that is the strongest group of at least one skinned vertex, or this raises.
    - `b:<bone>`: every posed bone's world matrix.
    - `p:<prop>`: for each of `clip.props`, the prop's frame: its bone's matrix moved by
      `offset_position` and turned by `offset_quaternion`, in that bone's frame.
    - for every part of the piece (see `export_rig`): `v:<part>` (S, V, 3) float32 posed
      points in glTF world, `f:<part>` (T, 3) int32 faces and `l:<part>` (V,) int32 labels, an
      index into the rig file's `vertex_groups[part]`. A mesh that is no part, a part with no
      mesh, or a vertex with no deform weight raises. The rig's outline modifiers are off, so
      V is the base mesh's vertex count.

    An inert scaled bone (no child bone, no skinned weight, and no prop of `clip` on it) passes
    the bone check and prints one line; any other scaled bone raises.

    At every sample the pose must keep the whole forward-kinematics contract, the pose checks
    included (see `_check_bones`), and the armature object must stand where the rig file says:
    export the rig first (`export_rig` with `clip.name` in `clips`), and the armature's world
    matrix must equal that file's `world` at every sample. A rig file holds one `world` matrix,
    which `pose_world` and the solver read, so a clip that places the armature elsewhere would
    be checked and solved from the wrong placement. Either raises.

    Each clip starts from the rest pose (every pose bone at location 0, identity rotation and
    scale 1) with the armature's NLA switched off (`animation_data.use_nla`), so a channel the
    clip does not key reads rest, whatever action ran before. Shape keys rest the same way
    (every block at value 0, NLA off), and the clip's KEY slots play on them (see
    `_use_action_on_keys`), so the sampled points follow a keyed shape key. The armature's
    action, pose, NLA switch, the shape keys' values, actions and NLA switches, and the scene
    frame are restored afterwards, a raise included. NLA tweak mode is not
    supported: leave it before the call.
    """
    rig = _rig_spec(pack, piece)
    armature = _armature(armature)
    who = f"{pack.name}:{piece}"
    _check_armature(armature, rig, who)
    name = _file_name(clip.name, "clip")
    path = Path(out) / "samples" / f"{name}.npz"
    times = _sample_times(clip.frames[0], clip.frames[1], step)
    bone_names = [b.name for b in armature.pose.bones]
    offsets = {}
    for pid, prop in clip.props.items():
        if prop.bone not in bone_names:
            raise ManifestError(f"prop {pid!r}: bone {prop.bone!r} is not in the armature")
        offset = np.eye(4)
        offset[:3, :3] = _quat_matrix(prop.offset_quaternion)
        offset[:3, 3] = np.asarray(prop.offset_position, dtype=float)
        offsets[pid] = offset
    hide_modifiers = rig.outline_modifiers
    meshes = _skinned(armature)
    _check_hidden(meshes, hide_modifiers)
    _check_contact_groups(armature, rig, meshes, who)
    deform = _deform_groups(armature)
    members = {g: _members(meshes, names, deform) for g, names in rig.contact_groups.items()}
    named = _body_meshes(pack.pieces[piece])  # a mesh and part that disagree: fix that first
    rig_file, rig_world = _rig_world(Path(out), pack.name, piece, name)
    named, _, labels, faces = _body(pack.pieces[piece], named, deform, who,
                                    _rig_groups(Path(out), pack.name, piece, name))
    objects = list({obj.name: obj for obj in (*meshes, *named.values())}.values())
    ridden = {prop.bone: pid for pid, prop in clip.props.items()}
    inert: dict[str, str | None] = {}

    scene = bpy.context.scene
    data = armature.animation_data
    saved_action = data.action if data else None
    saved_slot = data.action_slot if data else None
    saved_frame = (scene.frame_current, scene.frame_subframe)
    s = len(times)
    bones = {b: np.empty((s, 4, 4)) for b in bone_names}
    points = {g: np.empty((s, sum(len(i) for i in per_mesh), 3)) for g, per_mesh in members.items()}
    shells = {part: np.empty((s, len(obj.data.vertices), 3), dtype=np.float32)
              for part, obj in named.items()}
    keys = _shape_keys(objects)
    paired = _pair_key_slots(keys, action)  # before the rest step takes the keys' actions off
    with _rest_pose_no_nla(armature), _rest_shape_keys(keys):
        try:
            _use_action(armature, action)
            _use_action_on_keys(keys, action, paired)
            with _modifiers_off(objects, hide_modifiers):
                for i, t in enumerate(times):
                    whole = math.floor(t)
                    scene.frame_set(int(whole), subframe=float(t - whole))
                    depsgraph = _evaluate()
                    _check_bones(armature, f" at frame {t:g}", pose=True,
                                 ridden=ridden, inert=inert)
                    posed = armature.evaluated_get(depsgraph)
                    arm_world = _mat(posed.matrix_world)
                    drift = np.abs(arm_world - rig_world).max()
                    if drift > WORLD_DRIFT_TOL:
                        raise ManifestError(
                            f"armature {armature.name!r} at frame {t:g}: the object moved by "
                            f"{drift:.3g} from the world {rig_file} holds; place the armature "
                            "once, before export_rig, and animate bones, not the armature object")
                    world = BLENDER_TO_GLTF @ arm_world
                    for pb in posed.pose.bones:
                        bones[pb.name][i] = world @ _mat(pb.matrix)
                    body = {obj.name: _to_gltf(_world_points(obj, depsgraph)) for obj in objects}
                    mesh_points = [body[obj.name] for obj in meshes]
                    for part, obj in named.items():
                        shells[part][i] = body[obj.name]
                    for g, per_mesh in members.items():
                        points[g][i] = np.concatenate([p[idx] for p, idx in
                                                       zip(mesh_points, per_mesh, strict=True)])
        finally:
            data = armature.animation_data
            data.action = saved_action
            if saved_action is not None and saved_slot is not None:
                data.action_slot = saved_slot
            scene.frame_set(saved_frame[0], subframe=saved_frame[1])
    arrays = {"times": times}
    arrays.update({f"g:{g}": p for g, p in points.items()})
    arrays.update({f"b:{b}": m for b, m in bones.items()})
    arrays.update({f"p:{pid}": bones[clip.props[pid].bone] @ offset
                   for pid, offset in offsets.items()})
    arrays.update(shell_arrays(shells, faces, labels))
    np.savez_compressed(_new_file(path), **arrays)
    return path
