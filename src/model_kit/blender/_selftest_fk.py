"""Self-test: the numpy forward kinematics against Blender's own pose matrices.

Run it in Blender through `runner.run_blender(<this file>, [out_dir])`. It builds a 3-bone
armature in an empty scene, writes its rig with `export_rig` to `<out_dir>/rigs/shop/fk.json`
(the pack is `rig_pack("shop", "fk")`), poses it 5 ways, and writes each pose with Blender's
pose-bone world matrices (glTF axes) to `<out_dir>/fk_poses.json`. The test feeds the same
poses to `core.rig.pose_world`.

`build_rig`, `cube` and the record builders (`rig_pack`, `plain_pack`, `hand_chain`,
`keyed_clip`) are also used by the Blender tests to build their scenes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import bpy
import numpy as np

from model_kit.blender import runner
from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.blender.export import export_rig
from model_kit.core.manifest import (
    Bind,
    Bone,
    Chain,
    Clip,
    FloorGear,
    Frame,
    Listed,
    MirrorSpec,
    Pack,
    Part,
    Piece,
    Prop,
    Recipe,
    RigSpec,
    SitSpec,
)

# name, head, tail, roll, parent, connected. Odd rolls and a slanted bone so no axis is easy.
BONES = (
    ("root", (0.0, 0.0, 0.0), (0.0, 0.0, 0.4), 0.0, None, False),
    ("body", (0.0, 0.0, 0.4), (0.0, 0.1, 1.0), 0.3, "root", True),
    ("hand.R", (0.2, 0.1, 1.0), (0.5, 0.1, 1.1), -0.5, "body", False),
)

# Euler XYZ (radians) and location per bone. The connected bone never gets a location.
POSES = (
    {},
    {"root": ((0.3, 0.0, 0.0), (0.0, 0.0, 0.0))},
    {"root": ((0.1, -0.4, 0.7), (0.05, 0.1, -0.2)), "body": ((-0.5, 0.2, 0.3), (0, 0, 0)),
     "hand.R": ((0.9, -0.3, 1.2), (0.01, -0.02, 0.03))},
    {"body": ((1.2, 0.0, -0.8), (0, 0, 0)), "hand.R": ((0.0, 1.4, 0.0), (0, 0, 0))},
    {"root": ((-2.0, 1.0, 2.5), (0.3, -0.2, 0.1)), "body": ((0.7, -1.1, 0.4), (0, 0, 0)),
     "hand.R": ((-1.5, 0.6, -2.2), (-0.04, 0.05, 0.02))},
)


def build_rig(name: str = "Rig", extra: Sequence[tuple] = ()) -> bpy.types.Object:
    """A 3-bone armature (`BONES`, then any `extra` rows of the same shape), moved, turned and
    scaled, with 'XYZ' pose bones."""
    data = bpy.data.armatures.new(name)
    arm = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(arm)
    arm.location = (0.3, -0.2, 0.5)
    arm.rotation_euler = (0.2, -0.1, 0.4)
    arm.scale = (1.5, 1.5, 1.5)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for bone, head, tail, roll, parent, connected in (*BONES, *extra):
        eb = data.edit_bones.new(bone)
        eb.head, eb.tail, eb.roll = head, tail, roll
        if parent is not None:
            eb.parent = data.edit_bones[parent]
            eb.use_connect = connected
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in arm.pose.bones:
        pb.rotation_mode = "XYZ"
    bpy.context.view_layer.update()
    return arm


def cube(name: str, centre, half) -> bpy.types.Object:
    """A box mesh object with 8 vertices and 6 quads, linked to the scene."""
    (cx, cy, cz), (hx, hy, hz) = centre, half
    verts = [(cx + sx * hx, cy + sy * hy, cz + sz * hz)
             for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


# --- record builders ------------------------------------------------------------------------


def _piece(name: str, parts: Sequence[str], rig: RigSpec | None) -> Piece:
    """A piece of one listed row and the given parts, with nothing else on it."""
    return Piece(
        name=name, frame=Frame("gltf", "the origin"), recipe=Recipe("assembly", "test"),
        listed=(Listed("stock", "Test stock", "test", 0.002, None),),
        parts=tuple(Part(name=p, listed="stock", shape="box", on=None, group="g", role="solid",
                         bones=rig.bind[p].bones if rig else (),
                         bind=rig.bind[p].mode if rig else None) for p in parts),
        uses=(), holds=(), seats=(), mounts=(), done="A test piece.", size_class="piece",
        measured={}, rig=rig)


def plain_pack(name: str, pieces: Mapping[str, Sequence[str]], *,
               base: Pack | None = None) -> Pack:
    """A pack of unrigged pieces, `{piece: [part names]}`. With `base`, those pieces join the
    pieces of `base` (same pack name)."""
    made = {piece: _piece(piece, parts, None) for piece, parts in pieces.items()}
    if base is None:
        return Pack(name=name, units_per_metre=1.0, pieces=made)
    return Pack(name=base.name, units_per_metre=base.units_per_metre,
                pieces={**base.pieces, **made})


def hand_chain(group: str = "hand.R", *, effector: str = "hand.R", root: str = "body") -> Chain:
    """A one-bone chain on `effector`, whose effector points come from contact group `group`."""
    return Chain(root=root, bones=(effector,), effector=effector, effector_radius=0.05,
                 dofs={effector: "xyz"}, limits={}, effector_group=group)


def rig_pack(name: str = "shop", piece: str = "Doll", *, armature: str = "Rig",
             extra: Sequence[tuple] = (), parts: Mapping[str, str] | None = None,
             contact_groups: Mapping[str, Sequence[str]] | None = None,
             chains: Mapping[str, Chain] | None = None,
             outline_modifiers: Sequence[str] = ()) -> Pack:
    """A pack of one rigged piece whose rig is `BONES` and any `extra` rows, the bones of
    `build_rig(armature, extra)`. `parts` maps a part name to the one bone it is bound to. The
    rig has no contact groups and no chains unless given."""
    rows = (*BONES, *extra)
    bones = tuple(Bone(n, parent, float(np.linalg.norm(np.subtract(tail, head))))
                  for n, head, tail, _roll, parent, _connected in rows)
    parts = dict(parts or {})
    rig = RigSpec(
        armature=armature, bones=bones,
        bind={p: Bind((bone,), "rigid") for p, bone in parts.items()},
        contact_groups={g: tuple(v) for g, v in (contact_groups or {}).items()},
        chains=dict(chains or {}), sit=SitSpec(0.08, 0.03), mirror=MirrorSpec((), "x"),
        outline_modifiers=tuple(outline_modifiers))
    return Pack(name=name, units_per_metre=1.0, pieces={piece: _piece(piece, list(parts), rig)})


def keyed_clip(name: str, *, frames: tuple[int, int] = (1, 2),
               props: Mapping[str, Prop] | None = None) -> Clip:
    """A keyed clip with no holds and no targets, over `frames`, carrying `props`."""
    return Clip(name=name, frames=frames, fps=24.0, authoring="keyed", seat=None, holds=(),
                targets=(), floor_gear=FloorGear({}), props=dict(props or {}),
                mirror_of=None)


def main() -> None:
    out = Path(runner.script_args()[0])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = build_rig()
    export_rig(arm, rig_pack("shop", "fk"), "fk", out, clips=[])
    cases = []
    for pose in POSES:
        for pb in arm.pose.bones:
            euler, loc = pose.get(pb.name, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
            pb.rotation_euler, pb.location = euler, loc
        bpy.context.view_layer.update()
        world = BLENDER_TO_GLTF @ np.array(arm.matrix_world, dtype=float)
        cases.append({
            "pose": {pb.name: [list(pb.rotation_euler), list(pb.location)]
                     for pb in arm.pose.bones},
            "world": {pb.name: (world @ np.array(pb.matrix, dtype=float)).tolist()
                      for pb in arm.pose.bones},
        })
    (out / "fk_poses.json").write_text(json.dumps({"poses": cases}, indent=2) + "\n")
    runner.done()


if __name__ == "__main__":
    main()
