"""Armatures from a pack's rest bones, and binds of part meshes to them.

`make_armature` turns `Bone.rest` (head, tail, roll, deform, in the piece's glTF frame) into a
Blender armature under the piece root. `bind_part` gives a mesh one vertex group per bound bone
and an Armature modifier: a `rigid` bind weights every vertex 1.0 to its one bone; a `chain`
bind spreads weights along the chain with `forms.chain_weights`, a smoothstep falloff.
`effector_radius` measures how far a chain's end-bone contact patch reaches along the bone.

Frames. Positions in the manifest are glTF; they go into Blender through `build.B`, and chain
weights are computed on each vertex in the piece frame, in glTF, through `build.G3`. The
armature sits under the piece root at identity, so its space is the piece frame.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import bpy
import numpy as np
from mathutils import Vector

from model_kit.blender.build import G3, B
from model_kit.core.errors import ManifestError
from model_kit.core.forms import chain_weights
from model_kit.core.manifest import Bind, Bone, Chain
from model_kit.core.tolerances import BIND_WEIGHT_FLOOR


def make_armature(root: bpy.types.Object, name: str, bones: Sequence[Bone]) -> bpy.types.Object:
    """An armature named `name` (bare) under `root`, one bone per entry of `bones`.

    Every bone must carry `rest`, and a parent must come before its children. Bones are STICK,
    unconnected, rotation mode XYZ, with full inheritance and local location, which is what
    `export._check_bones` needs. A taken name, a bone with no rest, a parent that is not in
    `bones` or a bone of no length raises.
    """
    for bone in bones:
        if bone.rest is None:
            raise ManifestError(f"armature {name!r}: bone {bone.name!r} has no rest pose")
    if name in bpy.data.objects:
        raise ManifestError(f"armature name {name!r} is taken by an object in the file")
    seen: set[str] = set()
    for bone in bones:
        if bone.name in seen:
            raise ManifestError(f"armature {name!r}: bone {bone.name!r} is listed twice")
        if bone.parent is not None and bone.parent not in seen:
            raise ManifestError(f"armature {name!r}: bone {bone.name!r} has parent "
                                f"{bone.parent!r}, which is not listed before it")
        seen.add(bone.name)

    data = bpy.data.armatures.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    if obj.name != name:
        raise ManifestError(f"armature name {name!r} is taken; Blender named it {obj.name!r}")
    obj.parent = root
    data.display_type = "STICK"

    layer = bpy.context.view_layer
    previous = layer.objects.active
    layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        edit = data.edit_bones
        for bone in bones:
            rest = bone.rest
            head, tail = Vector(B(*rest.head)), Vector(B(*rest.tail))
            if (tail - head).length < 1e-6:
                raise ManifestError(f"armature {name!r}: bone {bone.name!r} has no length")
            eb = edit.new(bone.name)
            eb.head, eb.tail, eb.roll = head, tail, rest.roll
            eb.use_deform = rest.deform
            eb.use_connect = False
            eb.use_inherit_rotation = True
            eb.inherit_scale = "FULL"
            eb.use_local_location = True
            if bone.parent is not None:
                eb.parent = edit[bone.parent]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        layer.objects.active = previous
    for pb in obj.pose.bones:
        pb.rotation_mode = "XYZ"
    return obj


def _skinned_to(armature: bpy.types.Object) -> list[bpy.types.Object]:
    return [ob for ob in bpy.data.objects if ob.type == "MESH"
            and any(m.type == "ARMATURE" and m.object == armature for m in ob.modifiers)]


def effector_radius(armature: bpy.types.Object, chain: Chain,
                    contact_groups: Mapping[str, Sequence[str]]) -> float:
    """How far the effector's contact patch reaches along its bone, from the patch centre.

    The patch is every vertex, at rest, whose strongest deform group is in
    `contact_groups[chain.effector_group]`. The result is the largest (v - c) . axis, with c the
    patch centroid and axis the effector bone's unit head-to-tail. No such vertex raises.
    """
    if chain.effector_group not in contact_groups:
        raise ManifestError(f"effector group {chain.effector_group!r} is not in contact_groups")
    wanted = set(contact_groups[chain.effector_group])
    bone = armature.data.bones.get(chain.effector)
    if bone is None:
        raise ManifestError(f"effector bone {chain.effector!r} is not in armature "
                            f"{armature.name!r}")
    deform = {b.name for b in armature.data.bones if b.use_deform}
    to_armature = armature.matrix_world.inverted()
    picked: list[Vector] = []
    for ob in _skinned_to(armature):
        names = {g.index: g.name for g in ob.vertex_groups if g.name in deform}
        for v in ob.data.vertices:
            weights = [(g.weight, names[g.group]) for g in v.groups if g.group in names]
            if weights and max(weights)[1] in wanted:
                picked.append(to_armature @ ob.matrix_world @ v.co)
    if not picked:
        raise ManifestError(f"effector {chain.effector!r}: no skinned vertex has one of "
                            f"{sorted(wanted)} as its strongest deform group")
    pts = np.array([tuple(p) for p in picked])
    axis = np.array(tuple(bone.tail_local - bone.head_local))
    axis /= np.linalg.norm(axis)
    return float(((pts - pts.mean(axis=0)) @ axis).max())


def bind_part(ob: bpy.types.Object, armature: bpy.types.Object, bind: Bind) -> None:
    """Weight mesh `ob` to `bind.bones` of `armature` and add the Armature modifier.

    One vertex group per bound bone. `rigid`: every vertex weighs 1.0 to the one bone. `chain`:
    each vertex weighs `chain_weights` on its `bind.falloff.along` coordinate in the piece
    frame, and a weight at or under `BIND_WEIGHT_FLOOR` is left out. A vertex that ends in no group
    raises, as does a chain bind with no falloff or a bound bone that does not deform.
    """
    if ob.type != "MESH":
        raise ManifestError(f"part {ob.name!r}: only a mesh can be bound")
    if not bind.bones:
        raise ManifestError(f"part {ob.name!r}: bind lists no bones")
    for name in bind.bones:
        if name not in armature.data.bones:
            raise ManifestError(f"part {ob.name!r}: bone {name!r} is not in armature "
                                f"{armature.name!r}")
        if not armature.data.bones[name].use_deform:
            raise ManifestError(f"part {ob.name!r}: bone {name!r} does not deform, so its "
                                "vertices would never move")
        if name in ob.vertex_groups:
            raise ManifestError(f"part {ob.name!r}: already has a vertex group {name!r}")
    verts = ob.data.vertices
    n = len(verts)
    if bind.mode == "rigid":
        if len(bind.bones) != 1:
            raise ManifestError(f"part {ob.name!r}: a rigid bind takes one bone, "
                                f"got {list(bind.bones)}")
        weights = np.ones((n, 1))
    elif bind.mode == "chain":
        if bind.falloff is None:
            raise ManifestError(f"part {ob.name!r}: a chain bind needs a falloff")
        to_armature = armature.matrix_world.inverted() @ ob.matrix_world
        points = np.array([G3(to_armature @ v.co) for v in verts]).reshape(n, 3)
        try:
            weights = chain_weights(points, bind.falloff, len(bind.bones))
        except ValueError as exc:
            raise ManifestError(f"part {ob.name!r}: {exc}") from exc
    else:
        raise ManifestError(f"part {ob.name!r}: unknown bind mode {bind.mode!r}")

    groups = [ob.vertex_groups.new(name=name) for name in bind.bones]
    for k, group in enumerate(groups):
        for i in np.nonzero(weights[:, k] > BIND_WEIGHT_FLOOR)[0]:
            group.add([int(i)], float(weights[i, k]), "REPLACE")
    bare = [i for i, v in enumerate(ob.data.vertices) if not v.groups]
    if bare:
        raise ManifestError(f"part {ob.name!r}: {len(bare)} vertices have no deform weight "
                            f"(first: {bare[:5]})")
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = armature
    mod.use_vertex_groups = True
    mod.use_bone_envelopes = False
