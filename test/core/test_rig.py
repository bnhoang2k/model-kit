"""Forward kinematics and reach on a synthetic three-bone arm."""

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Chain
from model_kit.core.rig import RigData, chain_root, effector, pose_world, reach

MID = np.array([0.0, 0.05, 0.02])


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _arm(order=("b1", "b2", "b3")) -> RigData:
    """Three bones of length 0.1 along +Y: heads at y = 0, 0.1, 0.2. Bone 3 is the effector."""
    heads = {"b1": 0.0, "b2": 0.1, "b3": 0.2}
    return RigData(
        order=tuple(order),
        parent={"b1": None, "b2": "b1", "b3": "b2"},
        rest={b: _translate(0.0, y, 0.0) for b, y in heads.items()},
        world=np.eye(4),
        effector_mid={"b3": MID},
    )


# b1 is the chain's root (a strict ancestor of the first bone); b2 and b3 are the chain bones.
# Reach and `chain_root` start at b2, the first bone: the root does not turn in a solve.
CHAIN = Chain(root="b1", bones=("b2", "b3"), effector="b3", effector_radius=0.02,
              dofs={"b2": "x", "b3": "x"}, limits={}, effector_group="hand")


def test_rest_pose_effector():
    point = effector(_arm(), {}, CHAIN)
    assert np.allclose(point, np.array([0.0, 0.2, 0.0]) + MID)


def test_elbow_90():
    # Bone 2 turns 90 degrees about X. Bone 3 hangs off bone 2's head at (0, 0.1, 0).
    # In bone 2's frame, bone 3's head is (0, 0.1, 0) and the effector is that plus MID.
    # Rx(90) sends +Y to +Z and +Z to -Y.
    pose = {"b2": ((np.pi / 2, 0.0, 0.0), (0.0, 0.0, 0.0))}
    expected = np.array([0.0, 0.1, 0.0]) + _rx(np.pi / 2) @ (np.array([0.0, 0.1, 0.0]) + MID)
    assert np.allclose(expected, [0.0, 0.1 - 0.02, 0.1 + 0.05])
    assert np.allclose(effector(_arm(), pose, CHAIN), expected)


def test_euler_xyz_is_rz_ry_rx():
    angles = (0.3, -0.5, 0.7)
    pose = {"b1": (angles, (0.0, 0.0, 0.0))}
    w = pose_world(_arm(), pose)
    r_expected = _rz(angles[2]) @ _ry(angles[1]) @ _rx(angles[0])
    assert np.allclose(w["b1"][:3, :3], r_expected)
    assert not np.allclose(r_expected, _rx(angles[0]) @ _ry(angles[1]) @ _rz(angles[2]))


def test_loc_is_not_turned_by_own_rotation():
    pose = {"b1": ((0.0, 0.0, np.pi / 2), (0.1, 0.0, 0.0))}
    w = pose_world(_arm(), pose)
    assert np.allclose(w["b1"][:3, 3], (0.1, 0.0, 0.0))
    # b2 head is 0.1 up the bone; Rz(90) sends +Y to -X.
    assert np.allclose(w["b2"][:3, 3], (0.1 - 0.1, 0.0, 0.0))


def test_world_matrix_is_applied_last():
    rig = _arm()
    rig.world = _translate(1.0, 2.0, 3.0)
    assert np.allclose(effector(rig, {}, CHAIN), np.array([1.0, 2.2, 3.0]) + MID)


def test_chain_root_reads_the_posed_head_of_the_first_bone():
    # b1 moves 0.5 along X; b2 hangs 0.1 up it, so b2's head, the shoulder, is at (0.5, 0.1, 0).
    pose = {"b1": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))}
    w = pose_world(_arm(), pose)
    assert np.allclose(chain_root(w, CHAIN), (0.5, 0.1, 0.0))


def test_chain_root_needs_the_first_bone_posed():
    with pytest.raises(ManifestError, match="b2"):
        chain_root({"b1": np.eye(4)}, CHAIN)


def test_reach_from_first_bone():
    # b1 is the chain root, 0.1 above b2. Only b2 to b3 (0.1) plus the mid counts.
    assert reach(_arm(), CHAIN) == pytest.approx(0.1 + np.linalg.norm(MID))


def test_target_0_2135_from_the_posed_shoulder_is_out_of_a_0_2127_arm():
    # b3 sits 0.1127 past b2 and the mid adds 0.1, so the arm reaches 0.2127 from its shoulder.
    rig = _arm()
    rig.rest = {**rig.rest, "b3": _translate(0.0, 0.1 + 0.1127, 0.0)}
    rig.effector_mid = {"b3": np.array([0.0, 0.1, 0.0])}
    assert reach(rig, CHAIN) == pytest.approx(0.2127)
    w = pose_world(rig, {"b1": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.0))})
    target = chain_root(w, CHAIN) + np.array([0.0, -0.2135, 0.0])
    assert np.linalg.norm(target - chain_root(w, CHAIN)) > reach(rig, CHAIN)
    # measured from the root b1 it would be inside: the root is 0.1 further back
    assert np.linalg.norm(target - w["b1"][:3, 3]) < reach(rig, CHAIN) + 0.1


def test_reach_scales_with_the_world_matrix():
    # An armature object scaled by 0.01 shrinks every length by 0.01.
    rig = _arm()
    rig.world = np.diag([0.01, 0.01, 0.01, 1.0])
    assert reach(rig, CHAIN) == pytest.approx(0.01 * (0.1 + np.linalg.norm(MID)))


def test_reach_equals_the_straight_world_distance_when_mid_points_along_the_arm():
    rig = _arm()
    rig.world = np.diag([0.01, 0.01, 0.01, 1.0])
    rig.effector_mid = {"b3": np.array([0.0, 0.05, 0.0])}
    straight = np.linalg.norm(effector(rig, {}, CHAIN) - chain_root(pose_world(rig, {}), CHAIN))
    assert straight == pytest.approx(0.01 * 0.15)
    assert reach(rig, CHAIN) == pytest.approx(straight)


def test_parent_order_enforced_when_the_rig_is_built():
    with pytest.raises(ManifestError, match="b2"):
        _arm(order=("b2", "b1", "b3"))


def test_pose_world_rechecks_the_order():
    rig = _arm()
    rig.order = ("b2", "b1", "b3")
    with pytest.raises(ManifestError, match="b2"):
        pose_world(rig, {})


def test_parent_outside_the_rig_raises_by_name():
    with pytest.raises(ManifestError, match="ghost"):
        RigData(order=("b1",), parent={"b1": "ghost"}, rest={"b1": np.eye(4)},
                world=np.eye(4), effector_mid={})


def test_bone_with_no_rest_matrix_raises_by_name():
    rig = _arm()
    del rig.rest["b2"]
    with pytest.raises(ManifestError, match="b2"):
        reach(rig, CHAIN)
    with pytest.raises(ManifestError, match="b2"):
        pose_world(rig, {})


def test_pose_naming_an_unknown_bone_raises():
    with pytest.raises(ManifestError, match="nope"):
        pose_world(_arm(), {"nope": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))})


def test_effector_with_no_mid_raises():
    rig = _arm()
    rig.effector_mid = {}
    with pytest.raises(ManifestError, match="b3"):
        effector(rig, {}, CHAIN)


def test_rig_contract_helpers_are_reachable_from_core_rig():
    from model_kit.core import rig as core_rig
    from model_kit.core.manifest import rig as manifest_rig

    assert core_rig.bending_joints is manifest_rig.bending_joints
    assert core_rig.rig_root is manifest_rig.rig_root
