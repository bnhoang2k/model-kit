"""The stress pose set: which poses a rig gets, how they are named, and the scene that holds them."""

import dataclasses
import math
from functools import cache

import numpy as np
import pytest

from model_kit.core.errors import Unmeasurable
from model_kit.core.manifest import Bind, Bone, Joint, RigSpec
from model_kit.core.pose.stress import (
    STRESS_CLIP,
    missing_ranges,
    stress_poses,
    stress_scene,
)
from model_kit.core.rig import pose_world

from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene


def spec_of(bones: dict, bound: dict, joints: dict) -> RigSpec:
    """A rig spec that is only bones, what is bound to them and their joint ranges (degrees)."""
    return RigSpec(
        armature="t", bones=tuple(Bone(n, p, 1.0) for n, p in bones.items()),
        bind={part: Bind(tuple(b), "rigid") for part, b in bound.items()},
        contact_groups={}, chains={},
        joints={j: Joint(dict(r), "a test range") for j, r in joints.items()})


def arm() -> RigSpec:
    """A trunk bound to the root, and an arm: a 3-axis shoulder and a 1-axis elbow."""
    return spec_of({"root": None, "shoulder": "root", "elbow": "shoulder"},
                   {"trunk": ["root"], "arm": ["shoulder", "elbow"]},
                   {"shoulder": {"x": (-10, 20), "y": (-30, 40), "z": (-50, 60)},
                    "elbow": {"x": (-150, 0)}})


@cache
def scene(body: str):
    return {"humanoid": humanoid_scene, "quadruped": quadruped_scene, "worm": worm_scene}[body]()


def spec_of_scene(ctx):
    key = next(iter(ctx.rigs))
    pack, _, name = key.partition(":")
    return key, ctx.packs[pack].pieces[name].rig


def test_pair_corners_count():
    """A 3-axis parent and a 1-axis child: 2^3 x 2^1 = 16 pair poses, then 4 axes x 2 ends = 8."""
    poses = stress_poses(arm())
    pairs = [p for p in poses if len(p.joints) == 2]
    singles = [p for p in poses if len(p.joints) == 1]
    assert (len(pairs), len(singles)) == (16, 8)
    assert len(poses) == 24
    assert all(p.joints == ("elbow", "shoulder") for p in pairs)


def test_tail_grows_by_pairs():
    """Six bones of two axes each: 5 neighbouring pairs of 16, then 6 x 2 axes x 2 ends."""
    bones = {"root": None, **{f"t{i}": ("root" if i == 1 else f"t{i - 1}") for i in range(1, 7)}}
    ranges = {f"t{i}": {"x": (-20, 20), "z": (-30, 30)} for i in range(1, 7)}
    spec = spec_of(bones, {"body": ["root"], "tail": [f"t{i}" for i in range(1, 7)]}, ranges)
    assert len(stress_poses(spec)) == 5 * 16 + 24


def test_a_joint_whose_parent_is_not_a_joint_has_only_its_own_poses():
    poses = stress_poses(arm())
    assert [p.name for p in poses if p.joints == ("shoulder",)] == [
        f"shoulder {a} {e}" for a in "xyz" for e in ("lo", "hi")]


def test_a_pose_holds_only_its_joints_each_axis_at_a_range_end():
    pair = next(p for p in stress_poses(arm()) if p.name == "elbow+shoulder x lo; x hi y lo z hi")
    assert set(pair.pose) == {"elbow", "shoulder"}
    elbow, shoulder = (np.degrees(pair.pose[j][0]) for j in ("elbow", "shoulder"))
    assert elbow == pytest.approx([-150, 0, 0])
    assert shoulder == pytest.approx([20, -30, 60])  # the unused axes stay at 0
    assert not pair.pose["elbow"][1].any() and not pair.pose["shoulder"][1].any()  # no move
    single = next(p for p in stress_poses(arm()) if p.name == "shoulder z lo")
    assert set(single.pose) == {"shoulder"}
    assert np.degrees(single.pose["shoulder"][0]) == pytest.approx([0, 0, -50])


def test_a_range_is_taken_in_radians_by_pose_world():
    assert math.isclose(stress_poses(arm())[0].pose["elbow"][0][0], math.radians(-150))


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_names_are_unique_stable_and_safe_in_an_allow_line(body):
    _, spec = spec_of_scene(scene(body))
    names = [p.name for p in stress_poses(spec)]
    assert len(set(names)) == len(names) and names
    assert names == [p.name for p in stress_poses(spec)]  # the order does not change
    assert not any(c in name for name in names for c in "|*?[]")  # allow-file and fnmatch


def test_the_order_is_pairs_by_bone_then_singles_by_bone():
    names = [p.name for p in stress_poses(arm())]
    assert names[0] == "elbow+shoulder x lo; x lo y lo z lo"
    assert names[15] == "elbow+shoulder x hi; x hi y hi z hi"
    assert names[16:20] == ["shoulder x lo", "shoulder x hi", "shoulder y lo", "shoulder y hi"]
    assert names[-2:] == ["elbow x lo", "elbow x hi"]


def test_missing_joint_range_raises_naming_each_joint():
    spec = dataclasses.replace(arm(), joints={})
    assert missing_ranges(spec) == ("shoulder", "elbow")
    with pytest.raises(Unmeasurable, match="shoulder, elbow"):
        stress_poses(spec)
    only_elbow = dataclasses.replace(arm(), joints={"elbow": arm().joints["elbow"]})
    assert missing_ranges(only_elbow) == ("shoulder",)
    # partial: the elbow alone, with no pair (its parent joint has no range)
    assert [p.name for p in stress_poses(only_elbow, partial=True)] == ["elbow x lo", "elbow x hi"]


def test_a_rig_with_no_bending_joint_has_no_poses():
    spec = spec_of({"root": None, "a": "root"}, {"all": ["root"]}, {})
    assert stress_poses(spec) == [] and missing_ranges(spec) == ()


# --- the scene ------------------------------------------------------------------------------


def test_stress_scene_holds_one_synthetic_clip_with_a_sample_per_pose():
    ctx = scene("humanoid")
    key, spec = spec_of_scene(ctx)
    stressed, poses = stress_scene(ctx, key)
    assert [p.name for p in poses] == [p.name for p in stress_poses(spec)]
    assert list(stressed.clips) == [STRESS_CLIP] == list(stressed.samples)
    assert stressed.clips[STRESS_CLIP].posture == "free"
    assert stressed.rig_of_clip == {STRESS_CLIP: key}
    data = stressed.samples[STRESS_CLIP]
    assert len(data.times) == len(poses)
    rest = ctx.rests[key]
    assert set(data.shells) == set(rest.shells)
    assert all(a.shape == (len(poses), rest.shells[p].shape[1], 3) and a.dtype == np.float32
               for p, a in data.shells.items())
    assert set(data.groups) == set(spec.contact_groups)
    assert ctx.clips != stressed.clips and STRESS_CLIP not in ctx.clips  # the source is intact


def test_a_stress_sample_is_the_rig_posed_by_its_pose():
    ctx = scene("humanoid")
    key, _ = spec_of_scene(ctx)
    stressed, poses = stress_scene(ctx, key)
    data = stressed.samples[STRESS_CLIP]
    at = {p.name: i for i, p in enumerate(poses)}
    i = at["elbow.L x hi"]  # the elbow's high end is 0: the body is at rest
    rest = ctx.rests[key]
    for part, shell in data.shells.items():
        assert np.allclose(shell[i], rest.shells[part][0], atol=1e-6), part
    j = at["elbow.L x lo"]
    world = pose_world(ctx.rigs[key], poses[j].pose)
    for bone, matrices in data.bones.items():
        assert np.allclose(matrices[j], world[bone]), bone
    assert not np.allclose(data.shells["arm_L"][j], rest.shells["arm_L"][0], atol=1e-3)
    assert np.allclose(data.shells["arm_R"][j], rest.shells["arm_R"][0], atol=1e-6)  # untouched


def test_stress_scene_needs_a_rest_file_with_weights():
    ctx = scene("worm")
    key, _ = spec_of_scene(ctx)
    bare = dataclasses.replace(ctx, rests={})
    with pytest.raises(Unmeasurable, match="no rest file"):
        stress_scene(bare, key)
    rest = ctx.rests[key]
    unweighted = dataclasses.replace(rest, weights={}, weight_groups={})
    with pytest.raises(Unmeasurable, match="no skin weights"):
        stress_scene(dataclasses.replace(ctx, rests={key: unweighted}), key)


def test_stress_scene_with_no_pose_to_take_is_unmeasurable():
    ctx = scene("worm")
    key, spec = spec_of_scene(ctx)
    pack, _, name = key.partition(":")
    bare = dataclasses.replace(ctx.packs[pack].pieces[name],
                               rig=dataclasses.replace(spec, joints={}))
    packs = {pack: dataclasses.replace(ctx.packs[pack], pieces={
        **ctx.packs[pack].pieces, name: bare})}
    with pytest.raises(Unmeasurable, match="no joint range"):
        stress_scene(dataclasses.replace(ctx, packs=packs), key)
    with pytest.raises(Unmeasurable, match="no bending joint of rig .* has a range"):
        stress_scene(dataclasses.replace(ctx, packs=packs), key, partial=True)
