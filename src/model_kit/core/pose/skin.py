"""Linear blend skinning: a body moved by its bones' posed matrices, mixed by its weights.

This is what Blender's Armature modifier does with vertex groups and no envelopes. A vertex
moves by each bone that weighs on it: the bone's posed world matrix times the inverse of its
rest world matrix, applied to the vertex's rest point. The results are mixed by the weights,
which are first scaled to sum to 1 (Blender does the same and drops no small weight).

One difference: Blender leaves a vertex whose total weight is at or under 1e-4 at rest, while
`skin` raises only for a total of exactly 0. A kit export cannot make such a vertex (the bind
step floors each weight at 0.001 and the export refuses a vertex with no deform weight), but a
rig taken from a .blend file could.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.rig import RigData
from model_kit.core.scene import RigRest


def _bone_changes(rig: RigData, world: Mapping[str, np.ndarray], part: str,
                  groups: tuple[str, ...]) -> np.ndarray:
    """(G, 4, 4): for each deform group of the part, its bone's posed world matrix times the
    inverse of the bone's rest world matrix. A bone the rig or `world` lacks raises."""
    out = np.empty((len(groups), 4, 4))
    for g, bone in enumerate(groups):
        if bone not in rig.rest:
            raise ManifestError(f"part {part!r}: deform group {bone!r} is no bone of the rig")
        if bone not in world:
            raise ManifestError(f"part {part!r}: no posed matrix for bone {bone!r}")
        out[g] = np.asarray(world[bone], dtype=float) @ np.linalg.inv(rig.world @ rig.rest[bone])
    return out


def skin(rest: RigRest, rig: RigData, world: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The posed points of every part of `rest`, as a (V, 3) float64 array per part in glTF
    world.

    `world` is each bone's posed world matrix in glTF axes, as `rig.pose_world` returns it or a
    saved sample's `b:` arrays hold it for one sample. At the rest pose the result is the rest
    shell. A part with no weights raises `Unmeasurable`; a vertex whose weights sum to 0, or a
    bone with no posed matrix, raises `ManifestError`.
    """
    out: dict[str, np.ndarray] = {}
    for part, shell in rest.shells.items():
        if part not in rest.weights or part not in rest.weight_groups:
            raise Unmeasurable(f"part {part!r} has no skin weights in its rest file, so it "
                               "cannot be posed")
        points = np.asarray(shell[0], dtype=float)
        weights = np.asarray(rest.weights[part], dtype=float)
        index = np.asarray(rest.weight_groups[part], dtype=np.intp)
        total = weights.sum(axis=1)
        bare = np.nonzero(~(total > 0))[0]
        if len(bare):
            raise ManifestError(f"part {part!r}: vertex {int(bare[0])} has weights that sum to "
                                f"0 ({len(bare)} such vertices), so no bone moves it")
        change = _bone_changes(rig, world, part, rest.vertex_groups[part])
        # Each group moves every vertex; each vertex then takes the images its slots name.
        moved = np.einsum("gij,vj->gvi", change[:, :3, :3], points) + change[:, None, :3, 3]
        images = moved[index, np.arange(len(points))[:, None]]  # (V, K, 3)
        out[part] = np.einsum("vk,vki->vi", weights / total[:, None], images)
    return out
