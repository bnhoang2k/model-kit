"""Forward kinematics and chain reach, in numpy.

`RigData` is the rig as the checks and the solver need it: bone rest matrices, the parent
links and the axis change to glTF. A pose is `{bone: (euler_xyz, loc)}`. A bone missing from a
pose sits at rest. All matrices are 4x4 and act on column vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Chain
from model_kit.core.manifest.rig import bending_joints, rig_root

# `bending_joints` and `rig_root` live in the manifest package (the loader needs them and cannot
# import this module); they are re-exported here, where the checks look for rig functions.
__all__ = [
    "Pose",
    "RigData",
    "bending_joints",
    "chain_root",
    "effector",
    "pose_world",
    "reach",
    "rig_root",
]


@dataclass(eq=False)
class RigData:
    order: tuple[str, ...]  # parents before children
    parent: dict[str, str | None]
    rest: dict[str, np.ndarray]  # (4, 4) armature-space rest matrix per bone
    world: np.ndarray  # (4, 4), with the Blender-to-glTF axis change
    effector_mid: dict[str, np.ndarray]  # (3,) per effector bone, in that bone's rest frame

    def __post_init__(self) -> None:
        """A bad rig fails once, when it is built: every bone in `order` has a parent entry and
        a rest matrix, and each parent is a bone listed earlier."""
        seen: set[str] = set()
        for bone in self.order:
            if bone not in self.parent:
                raise ManifestError(f"rig bone {bone!r} has no parent entry")
            if bone not in self.rest:
                raise ManifestError(f"rig bone {bone!r} has no rest matrix")
            parent = self.parent[bone]
            if parent is not None and parent not in seen:
                if parent in self.order:
                    raise ManifestError(
                        f"rig order lists {bone!r} before its parent {parent!r}; parents come first")
                raise ManifestError(f"rig bone {bone!r}: its parent {parent!r} is not in the rig")
            seen.add(bone)


# A bone's euler angles (radians, X, Y, Z) and its local translation.
Pose = dict[str, tuple[np.ndarray, np.ndarray]]


def _rotation_xyz(euler) -> np.ndarray:
    """The 3x3 matrix of Blender's 'XYZ' euler mode: `Rz @ Ry @ Rx` on column vectors.

    The X turn is applied first, then Y, then Z, each about a fixed axis.
    """
    ax, ay, az = (float(a) for a in euler)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _basis(euler, loc) -> np.ndarray:
    """`T(loc) @ R_xyz(euler)`: the pose change in a bone's own frame."""
    m = np.eye(4)
    m[:3, :3] = _rotation_xyz(euler)
    m[:3, 3] = np.asarray(loc, dtype=float)
    return m


def pose_world(rig: RigData, pose: Pose) -> dict[str, np.ndarray]:
    """The posed world matrix of every bone, in glTF axes.

    A root gets `rest @ basis`. A child gets `W[parent] @ inv(rest[parent]) @ rest[bone] @
    basis`. Every matrix is then pre-multiplied by `rig.world`. A bone listed before its
    parent, a parent outside the rig, or a pose for an unknown bone raises.
    """
    unknown = sorted(set(pose) - set(rig.order))
    if unknown:
        raise ManifestError(f"pose names bones the rig lacks: {unknown}")
    armature: dict[str, np.ndarray] = {}
    for bone in rig.order:
        euler, loc = pose.get(bone, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
        basis = _basis(euler, loc)
        parent = rig.parent.get(bone)
        if bone not in rig.rest:
            raise ManifestError(f"rig bone {bone!r} has no rest matrix")
        if parent is None:
            armature[bone] = rig.rest[bone] @ basis
        elif parent not in armature:
            raise ManifestError(
                f"rig order lists {bone!r} before its parent {parent!r}; parents come first")
        else:
            armature[bone] = (armature[parent] @ np.linalg.inv(rig.rest[parent])
                              @ rig.rest[bone] @ basis)
    return {bone: rig.world @ m for bone, m in armature.items()}


def effector(rig: RigData, pose: Pose, chain: Chain) -> np.ndarray:
    """The chain's effector point in world space: `effector_mid` carried by the posed bone."""
    mid = rig.effector_mid.get(chain.effector)
    if mid is None:
        raise ManifestError(f"rig has no effector_mid for effector bone {chain.effector!r}")
    matrix = pose_world(rig, pose)[chain.effector]
    return matrix[:3, :3] @ np.asarray(mid, dtype=float) + matrix[:3, 3]


def chain_root(bones_world: dict[str, np.ndarray], chain: Chain) -> np.ndarray:
    """The chain's first bone's head in world space, read from posed bone matrices.

    That head is the shoulder (or hip): the point the chain turns about. `chain.root` is an
    ancestor that does not turn in a solve, so it is not the reach origin.
    """
    first = chain.bones[0]
    if first not in bones_world:
        raise ManifestError(f"no posed matrix for the chain's first bone {first!r}")
    return np.asarray(bones_world[first])[:3, 3].copy()


def reach(rig: RigData, chain: Chain) -> float:
    """How far the chain can stretch, in world units (glTF axes, `rig.world` applied).

    It is the sum of the distances between consecutive bone heads, from the head of the
    chain's first bone down to the effector bone, plus the length of `effector_mid`. The first
    bone's head is the shoulder (or hip), the true centre of the sphere the effector can reach;
    the chain root above it does not turn in a solve and adds nothing. Heads and the mid are
    measured after `rig.world`, so an armature object scale is honored, the same as in
    `chain_root` and `effector`. The pose does not matter: bone lengths do not change.
    """
    mid = rig.effector_mid.get(chain.effector)
    if mid is None:
        raise ManifestError(f"rig has no effector_mid for effector bone {chain.effector!r}")
    path = list(chain.bones)
    for bone in path:
        if bone not in rig.rest:
            raise ManifestError(f"rig bone {bone!r} has no rest matrix")
    linear, shift = rig.world[:3, :3], rig.world[:3, 3]
    heads = np.array([linear @ rig.rest[b][:3, 3] + shift for b in path], dtype=float)
    mid_world = linear @ rig.rest[chain.effector][:3, :3] @ np.asarray(mid, dtype=float)
    return float(np.linalg.norm(np.diff(heads, axis=0), axis=1).sum()
                 + np.linalg.norm(mid_world))
