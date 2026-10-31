"""The body of a rig as points and triangles, labelled by deform group: the input to a compare.

`pose_arrays` reads every mesh that an Armature modifier binds to one armature, as the evaluated
mesh stands at rest or at one frame of the armature's current action. It needs no pack, so it
reads a model that no pack describes. Points and triangles are in glTF axes (X right, Y up,
Z front), the axes `core.compare` reads. Needs Blender 4.4 or later.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping

import bpy
import numpy as np

from model_kit.blender import export
from model_kit.core.errors import ManifestError


def pose_arrays(armature: bpy.types.Object, *, frame: float | None = None,
                relabel: Mapping[str, str] | None = None,
                leave_groups: Collection[str] = (),
                hide_modifiers: Collection[str] = ()) -> dict[str, np.ndarray]:
    """Every mesh skinned to `armature`, at rest (`frame=None`) or posed at `frame`.

    The pose is whatever the armature's current action gives at that frame; the armature and
    the scene frame are put back afterwards. A modifier named in `hide_modifiers` is off while
    it reads (a typo raises), and a modifier that still changes the vertex count raises.

    Returns:
      - `points` (N, 3) float64 and `labels` (N,): each vertex's strongest deform group, or ""
        when it has none.
      - `tris` (M, 3, 3) and `tri_labels` (M,): the label when all three corners share it,
        else "".

    `relabel` maps a mesh object's name to one label for all its vertices (a name that is no
    skinned mesh raises). `leave_groups` then drops every vertex with one of those labels, and
    every triangle that touches one.
    """
    armature = export._armature(armature)
    relabel = dict(relabel or {})
    meshes = export._skinned(armature)
    export._check_hidden(meshes, hide_modifiers)
    unknown = sorted(set(relabel) - {obj.name for obj in meshes})
    if unknown:
        raise ManifestError(f"relabel names no mesh skinned to {armature.name!r}: {unknown}")
    deform = export._deform_groups(armature)
    scene = bpy.context.scene
    saved_pose = armature.data.pose_position
    saved_frame = (scene.frame_current, scene.frame_subframe)
    points, labels, faces = [], [], []
    try:
        armature.data.pose_position = "REST" if frame is None else "POSE"
        with export._modifiers_off(meshes, hide_modifiers):
            if frame is not None:
                whole = math.floor(frame)
                scene.frame_set(int(whole), subframe=float(frame - whole))
            depsgraph = export._evaluate()
            offset = 0
            for obj in meshes:
                pts = export._to_gltf(export._world_points(obj, depsgraph))
                if obj.name in relabel:
                    names = [relabel[obj.name]] * len(pts)
                else:
                    names = [name or "" for name in export._strongest(obj, deform)]
                points.append(pts)
                labels += names
                faces.append(export._faces(obj.name, obj) + offset)
                offset += len(pts)
    finally:
        armature.data.pose_position = saved_pose
        scene.frame_set(saved_frame[0], subframe=saved_frame[1])
        export._evaluate()

    all_points = np.concatenate(points) if points else np.zeros((0, 3))
    all_labels = np.array(labels, dtype=str)
    all_faces = np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
    keep = ~np.isin(all_labels, list(leave_groups))
    kept_faces = all_faces[keep[all_faces].all(axis=1)]
    tri_ends = all_labels[kept_faces]  # (M, 3): the label at each corner
    same = (tri_ends[:, 0] == tri_ends[:, 1]) & (tri_ends[:, 0] == tri_ends[:, 2])
    tri_labels = np.where(same, tri_ends[:, 0], "").astype(str)
    return {
        "points": all_points[keep],
        "labels": all_labels[keep],
        "tris": all_points[kept_faces],
        "tri_labels": tri_labels,
    }
