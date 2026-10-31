"""The clip solver on small synthetic chains, and the left/right mirror."""

import re
from itertools import pairwise

import numpy as np
import pytest

from model_kit.core import solve as solve_module
from model_kit.core.errors import ManifestError
from model_kit.core.geom import SurfaceIndex
from model_kit.core.manifest import (
    Avoid,
    Bone,
    Chain,
    Clip,
    FloorGear,
    MirrorSpec,
    Ref,
    RigSpec,
    SitSpec,
    Target,
)
from model_kit.core.rig import Pose, RigData, effector, pose_world, reach
from model_kit.core.solve import (
    SOLID_STEPS,
    STEPS,
    ChainArgs,
    Chunk,
    Plane,
    SkinPoints,
    Solid,
    Tool,
    mirror_pose,
    nelder,
    other_side,
    solve_chain,
    solve_clip,
)
from model_kit.core.tolerances import (
    AVOID_ELBOW_OF_EFFECTOR,
    QUICK_RETRY_MISS_OF_REACH,
    SOLID_MARGIN_OF_EFFECTOR,
    SOLVER_REFERENCE_REACH,
    TIP_RETRIES,
    TIP_RETRY_MISS_OF_EFFECTOR,
    W_SOLID,
)

from .shapes import box

BONE = 0.1


def _translate(x: float, y: float, z: float) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _planar(scale: float = 1.0, along: int = 0) -> RigData:
    """A base bone at the origin, then two bones of 0.1 along +X (or the axis `along`) and a
    hand at their tip.

    Every rest frame has the world's axes, so a turn about bone Z turns in the XY plane.
    """
    heads = {"base": 0.0, "upper": 0.0, "lower": BONE, "hand": 2 * BONE}
    return RigData(
        order=("base", "upper", "lower", "hand"),
        parent={"base": None, "upper": "base", "lower": "upper", "hand": "lower"},
        rest={b: _translate(*np.eye(3)[along] * x * scale) for b, x in heads.items()},
        world=np.eye(4),
        effector_mid={"hand": np.zeros(3)},
    )


# One effector radius for every test: a mitt's share of an arm, on this 0.2 arm.
RADIUS = 0.0586 * (2 * BONE) / SOLVER_REFERENCE_REACH


def _chain(dofs=None, limits=None, radius=RADIUS) -> Chain:
    return Chain(root="base", bones=("upper", "lower", "hand"), effector="hand",
                 effector_radius=radius,
                 dofs=dofs or {"upper": "z", "lower": "z", "hand": "z"},
                 limits=limits or {}, effector_group="hand")


PLAIN = ChainArgs()  # no level, tool, plane or avoid: the plain reach


def _angles(pose, bones=("upper", "lower", "hand")) -> np.ndarray:
    return np.array([np.asarray(pose.get(b, ((0, 0, 0), (0, 0, 0)))[0], dtype=float)
                     for b in bones])


def test_nelder_finds_a_bowl_minimum():
    x, fx = nelder(lambda v: (v[0] - 1.0) ** 2 + 2.0 * (v[1] + 0.5) ** 2, [0.0, 0.0], 0.25)
    assert x == pytest.approx([1.0, -0.5], abs=1e-4)
    assert fx < 1e-8


# Exact two-link answer for REACH_TARGET: elbow 1.7264 rad, shoulder -0.4684 rad.
REACH_TARGET = np.array([0.12, 0.05, 0.0])


def test_two_link_reachable():
    # As in a clip, the last key is near the answer (0.05 rad off on each joint), so the stay
    # term's pull costs almost nothing.
    last = {"upper": ((0.0, 0.0, -0.42), (0.0, 0.0, 0.0)),
            "lower": ((0.0, 0.0, 1.68), (0.0, 0.0, 0.0))}
    target = REACH_TARGET
    result = solve_chain(_planar(), {}, _chain(), target, args=PLAIN, x0=last)
    assert result.err < 1e-3  # 0.33 mm: the stay pull of the 0.05 rad start
    assert np.linalg.norm(effector(_planar(), result.pose, _chain()) - target) == pytest.approx(
        result.err, abs=1e-12)


def test_stay_costs_a_small_miss_from_rest():
    # The stay term pulls toward the last key. From rest, 1.8 rad away, today's weights trade
    # 8.9 mm of this 0.2 chain's reach (4.5 %) for it. This chain is close to the reference
    # arm (reach 0.2127), so the weights bite as they do on the reference arm. This pins that
    # property so a weight change shows here.
    result = solve_chain(_planar(), {}, _chain(), REACH_TARGET, args=PLAIN)
    assert 5e-3 < result.err < 1.2e-2


def test_unreachable_reports_shortfall():
    target = np.array([0.0, 0.3, 0.0])
    result = solve_chain(_planar(), {}, _chain(), target, args=PLAIN)
    assert result.err == pytest.approx(np.linalg.norm(target) - 2 * BONE, abs=1e-3)


def test_hinge_dof_only_turns_x():
    # Bones along +Y, so the elbow's X hinge bends the forearm.
    chain = _chain(dofs={"upper": "xyz", "lower": "x", "hand": "xyz"})
    result = solve_chain(_planar(along=1), {}, chain, np.array([0.05, 0.08, 0.1]), args=PLAIN)
    lower = np.asarray(result.pose["lower"][0], dtype=float)
    assert lower[1] == 0.0 and lower[2] == 0.0
    assert lower[0] != 0.0
    assert result.err < 1.2e-2  # the stay miss from rest, as above


def test_effector_turns_only_for_level():
    chain = _chain(dofs={"upper": "z", "lower": "z", "hand": "xyz"})
    target = np.array([0.12, 0.05, 0.0])
    free = solve_chain(_planar(), {}, chain, target, args=PLAIN)
    assert np.all(np.asarray(free.pose.get("hand", ((0, 0, 0), ()))[0]) == 0.0)
    level = np.eye(3)
    held = solve_chain(_planar(), {}, chain, target,
                       args=ChainArgs(level=level, level_weight=1.0))
    hand = pose_world(_planar(), held.pose)["hand"][:3, :3]
    assert np.abs(hand - level).max() < 1e-2  # the hand turns back to level
    assert np.any(np.asarray(held.pose["hand"][0]) != 0.0)
    assert held.err < 2e-2  # the stay miss from rest, as above


def test_limit_margin():
    # The target sits near the shoulder: reaching it needs the elbow folded to 2.2 rad. The
    # limit is a penalty, not a wall, so the elbow ends a little past it: within 0.03 rad.
    chain = _chain(limits={"lower:z": (-1.0, 1.0)})
    result = solve_chain(_planar(), {}, chain, np.array([0.06, 0.07, 0.0]), args=PLAIN)
    elbow = float(np.asarray(result.pose["lower"][0])[2])
    assert 1.0 <= elbow <= 1.0 + 0.03
    assert result.err > 0.03  # the limit, not the target, decided the bend


MIRROR = MirrorSpec(pairs=(("arm.R", "arm.L"), ("hand.R", "hand.L")), axis="x")


def test_other_side_swaps_pairs_and_keeps_centre_bones():
    assert other_side("arm.R", MIRROR) == "arm.L"
    assert other_side("hand.L", MIRROR) == "hand.R"
    assert other_side("body", MIRROR) == "body"


def test_mirror_keeps_the_mirror_axis_turn_and_flips_its_shift():
    pose = {"arm.R": ((0.1, 0.2, 0.3), (0.01, 0.02, 0.03))}
    out = mirror_pose(pose, MIRROR)
    assert set(out) == {"arm.L"}
    euler, loc = out["arm.L"]
    assert np.allclose(euler, (0.1, -0.2, -0.3))
    assert np.allclose(loc, (-0.01, 0.02, 0.03))
    euler, loc = mirror_pose(pose, MirrorSpec(pairs=MIRROR.pairs, axis="y"))["arm.L"]
    assert np.allclose(euler, (-0.1, 0.2, -0.3))
    assert np.allclose(loc, (0.01, -0.02, 0.03))


def test_mirror_is_involution():
    pose = {"arm.R": ((0.1, 0.2, 0.3), (0.01, 0.02, 0.03)),
            "hand.L": ((-0.4, 0.5, -0.6), (0.0, 0.0, 0.0)),
            "body": ((0.7, -0.8, 0.9), (0.04, -0.05, 0.06))}
    back = mirror_pose(mirror_pose(pose, MIRROR), MIRROR)
    assert set(back) == set(pose)
    for bone, (euler, loc) in pose.items():
        assert np.allclose(back[bone][0], euler) and np.allclose(back[bone][1], loc)


def test_scale_free():
    target = np.array([0.12, 0.05, 0.0])
    small = solve_chain(_planar(), {}, _chain(), target, args=PLAIN)
    big = solve_chain(_planar(10.0), {}, _chain(radius=10 * RADIUS), target * 10.0,
                      args=PLAIN)
    assert np.abs(_angles(small.pose) - _angles(big.pose)).max() < 1e-6
    assert big.err == pytest.approx(10.0 * small.err, abs=1e-9)


def test_shape_free_in_effector_radius():
    # Same chain, a fingertip against a big mitt: the radius does not enter the cost.
    target = np.array([0.12, 0.05, 0.0])
    limits = {"lower:z": (-1.0, 1.0)}
    for goal in (target, np.array([0.03, 0.02, 0.0])):
        tip = solve_chain(_planar(), {}, _chain(limits=limits, radius=0.003), goal, args=PLAIN)
        mitt = solve_chain(_planar(), {}, _chain(limits=limits, radius=0.1), goal, args=PLAIN)
        assert np.abs(_angles(tip.pose) - _angles(mitt.pose)).max() < 1e-12


def test_start_pose_picks_the_elbow_branch():
    # x0 (the last key) is where the search starts and what the stay term pulls toward. The
    # target has two elbow solutions; each start keeps its own side.
    target = np.array([0.12, 0.05, 0.0])
    for bend in (0.6, -0.6):
        last = {"lower": ((0.0, 0.0, bend), (0.0, 0.0, 0.0))}
        result = solve_chain(_planar(), {}, _chain(), target, args=PLAIN, x0=last)
        assert result.err < 1e-2  # the stay miss: the start is over 1 rad off
        assert np.sign(float(np.asarray(result.pose["lower"][0])[2])) == np.sign(bend)


def test_stay_miss_is_reported():
    # From rest the stay term holds the arm back, so the effector ends short of the target.
    # That shortfall is not hidden: `err` is the real distance, and it is not zero.
    rig, chain = _planar(), _chain()
    result = solve_chain(rig, {}, chain, REACH_TARGET, args=PLAIN)
    real = float(np.linalg.norm(effector(rig, result.pose, chain) - REACH_TARGET))
    assert result.err == pytest.approx(real, abs=1e-12)
    assert result.err > 1e-3
    assert np.allclose(result.point, effector(rig, result.pose, chain), atol=1e-12)


# --- the new terms ----------------------------------------------------------------------------

# A hand-held tool: its tip is 0.1 along the hand bone's X.
TOOL = Tool(tip=np.array([0.1, 0.0, 0.0]), slide=0.0)
# A last key close to the answer for REACH_TARGET (elbow below the X axis).
NEAR = {"upper": ((0.0, 0.0, -0.42), (0.0, 0.0, 0.0)),
        "lower": ((0.0, 0.0, 1.68), (0.0, 0.0, 0.0))}


def _posed(rig, pose, bone, local=(0.0, 0.0, 0.0)) -> np.ndarray:
    m = pose_world(rig, pose)[bone]
    return m[:3, :3] @ np.asarray(local, dtype=float) + m[:3, 3]


def _skin(**bone_points) -> dict[str, SkinPoints]:
    """Skin points that follow one bone each: `hand=[(x, y, z), ...]` in that bone's frame."""
    out = {}
    for bone, points in bone_points.items():
        pts = np.asarray(points, dtype=float)
        out[bone] = SkinPoints(bones=np.array([[bone]] * len(pts)), weights=np.ones((len(pts), 1)),
                               local=pts[:, None, :])
    return out


def _skin_world(rig, pose, skin, bone) -> np.ndarray:
    return skin[bone].posed(pose_world(rig, pose))


def test_tool_tip_reaches_target():
    rig, chain = _planar(), _chain()
    target = np.array([0.2, 0.12, 0.0])
    tipped = solve_chain(rig, {}, chain, target, args=ChainArgs(tool=TOOL))
    tip = _posed(rig, tipped.pose, "hand", TOOL.tip)
    assert tipped.err == pytest.approx(float(np.linalg.norm(tip - target)), abs=1e-12)
    assert np.allclose(tipped.point, tip, atol=1e-12)
    assert tipped.err < 1.5e-2  # the stay miss from rest
    # The hand centre alone cannot reach this point (0.233 away, reach 0.2): the tip does.
    assert np.linalg.norm(_posed(rig, tipped.pose, "hand") - target) > 0.05
    assert solve_chain(rig, {}, chain, target, args=PLAIN).err > 0.03


def _sdf_box(p, avoid) -> float:
    d = np.abs(np.asarray(p) - np.asarray(avoid.centre)) - np.asarray(avoid.half)
    return float(np.linalg.norm(np.maximum(d, 0.0))) if np.any(d > 0) else float(d.max())


def test_avoid_keeps_elbow_out_of_box():
    rig, chain = _planar(), _chain()
    shape = Avoid(id="torso", bone="base", kind="box", centre=(0.09, -0.04, 0.0),
                  half=(0.01, 0.01, 0.05), why="test")
    elbow_r = AVOID_ELBOW_OF_EFFECTOR * RADIUS  # 0.047: the elbow is a big ball on this arm
    # avoid_clearance 0 turns the term off, so the shape alone changes nothing.
    free = solve_chain(rig, {}, chain, REACH_TARGET, args=PLAIN, x0=NEAR, avoid=(shape,))
    alone = solve_chain(rig, {}, chain, REACH_TARGET, args=PLAIN, x0=NEAR)
    assert np.array_equal(_angles(free.pose), _angles(alone.pose))
    assert _sdf_box(_posed(rig, free.pose, "lower"), shape) - elbow_r < -0.05  # deep inside
    # A negative clearance lets the elbow's ball overlap the box by 0.04 and no more.
    kept = solve_chain(rig, {}, chain, REACH_TARGET, args=ChainArgs(avoid_clearance=-0.04),
                       x0=NEAR, avoid=(shape,))
    assert _sdf_box(_posed(rig, kept.pose, "lower"), shape) - elbow_r > -0.04 - 0.004
    assert kept.err < 3e-2  # the reach gave a little


def test_avoid_ellipsoid_keeps_the_hand_centre_out():
    rig, chain = _planar(), _chain()
    # The target sits inside an ellipsoid of semi-axes 0.04: the hand centre must stay out.
    head = Avoid(id="head", bone="base", kind="ellipsoid", centre=(0.12, 0.05, 0.0),
                 half=(0.04, 0.04, 0.04), why="test")
    near = {"upper": ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0)), "lower": ((0.0, 0.0, -1.7), (0.0, 0.0, 0.0))}
    off = solve_chain(rig, {}, chain, REACH_TARGET, args=PLAIN, x0=near, avoid=(head,))
    on = solve_chain(rig, {}, chain, REACH_TARGET, args=ChainArgs(avoid_clearance=0.0),
                     x0=near, avoid=(head,))
    assert np.allclose(_angles(off.pose), _angles(on.pose), atol=1e-12)
    kept = solve_chain(rig, {}, chain, REACH_TARGET, args=ChainArgs(avoid_clearance=0.004),
                       x0=near, avoid=(head,))
    hand = _posed(rig, kept.pose, "hand")
    k = np.linalg.norm((hand - np.asarray(head.centre)) / np.asarray(head.half))
    assert k > 1.0  # outside the ellipsoid
    assert kept.err > off.err  # the clearance cost some reach


HAND_PLATE = [(0.04, 0.01, 0.0), (-0.04, 0.01, 0.0), (0.04, -0.01, 0.0), (-0.04, -0.01, 0.0)]


def test_plane_keeps_skin_in_front():
    rig = _planar()
    chain = _chain()
    skin = _skin(hand=HAND_PLATE)
    target = np.array([0.12, 0.02, 0.0])
    # The plane y = 0.03 faces up: every skin point should end at y >= 0.03 + margin.
    plane = Plane(point=np.array([0.0, 0.03, 0.0]), normal=np.array([0.0, 1.0, 0.0]), margin=0.0)
    turned = {"hand": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))}  # the plate starts on edge
    free = solve_chain(rig, turned, chain, target, args=PLAIN, skin=skin)
    held = solve_chain(rig, turned, chain, target, args=ChainArgs(plane=plane), skin=skin)
    low_free = _skin_world(rig, free.pose, skin, "hand")[:, 1].min()
    low_held = _skin_world(rig, held.pose, skin, "hand")[:, 1].min()
    assert low_free < 0.03 - 0.02
    assert low_held > 0.03 - 0.005
    # The plane frees the hand's own dofs: it turned the plate, which `free` could not.
    assert np.asarray(held.pose["hand"][0])[2] != np.asarray(turned["hand"][0])[2]
    assert np.asarray(free.pose["hand"][0])[2] == np.asarray(turned["hand"][0])[2]


def test_plane_needs_skin():
    plane = Plane(point=np.zeros(3), normal=np.array([0.0, 1.0, 0.0]), margin=0.0)
    with pytest.raises(ManifestError, match="skin"):
        solve_chain(_planar(), {}, _chain(), REACH_TARGET, args=ChainArgs(plane=plane))


def _slab(scale=1.0, margin=0.005, **extra) -> Solid:
    """A slab under y = -0.01, wide in X and Z, with a margin on the hand's skin."""
    tris = box((-0.1, -0.2, -0.1), (0.3, -0.01, 0.1)) * scale
    return Solid(name="slab", index=SurfaceIndex(tris),
                 margins={"hand": margin * scale}, **extra)


SOLID_TARGET = np.array([0.15, 0.0, 0.0])
UNDER_HAND = [(0.0, -0.02, 0.0)]  # a skin point 2 cm under the hand centre


def test_solid_phase_moves_skin_out():
    rig, chain = _planar(), _chain()
    skin = _skin(hand=UNDER_HAND)
    last = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN)  # a start near the target
    free = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, x0=last.pose, skin=skin)
    pushed = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, x0=last.pose, skin=skin,
                         solids=(_slab(),))
    under_free = _skin_world(rig, free.pose, skin, "hand")[0, 1]
    under_pushed = _skin_world(rig, pushed.pose, skin, "hand")[0, 1]
    assert under_free < -0.01  # inside the slab without the solid
    assert under_pushed > -0.01 + 0.005 - 1.5e-3  # out by its margin
    assert pushed.err < 0.03  # the hand rose by about the skin's depth, no more


def test_solid_margin_missing_means_untested():
    rig, chain = _planar(), _chain()
    skin = _skin(hand=UNDER_HAND, lower=[(0.0, -0.03, 0.0)])
    none = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin)
    # The slab has a margin for "hand" only: the lower arm's point is not tested against it.
    only_hand = _slab()
    assert "lower" not in only_hand.margins
    free_hand = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=_skin(lower=[(0, -0.03, 0)]),
                            solids=(only_hand,))
    assert np.array_equal(_angles(free_hand.pose), _angles(none.pose))
    # With a margin for "lower" the same point is pushed.
    both = _slab()
    both = Solid(name="slab", index=both.index, margins={"lower": 0.005})
    pushed = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=_skin(lower=[(0, -0.03, 0)]),
                         solids=(both,))
    assert not np.allclose(_angles(pushed.pose), _angles(none.pose), atol=1e-6)


def test_held_by_skips_own_prop():
    rig, chain = _planar(), _chain()  # effector_group "hand"
    skin = _skin(hand=UNDER_HAND)
    none = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin)
    mine = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin,
                       solids=(_slab(held_by="hand"),))
    assert np.array_equal(_angles(mine.pose), _angles(none.pose))  # the fist's own prop
    other = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin,
                        solids=(_slab(held_by="hand.L"),))
    assert not np.allclose(_angles(other.pose), _angles(none.pose), atol=1e-6)  # not mine


def test_grip_zone_skips_the_solid():
    rig, chain = _planar(), _chain()
    skin = _skin(hand=UNDER_HAND)
    none = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin)
    # The grip zone is a ball in the solid's frame around where the hand's point sits.
    gripped = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin,
                          solids=(_slab(skip={"hand": ((np.array([0.15, -0.02, 0.0]), 0.05),)}),))
    # The point is inside the ball, so the slab does not push it. (Phase two still runs from
    # phase one's best, so the angles agree to the search's precision, not bit for bit.)
    assert np.allclose(_angles(gripped.pose), _angles(none.pose), atol=1e-3)
    pushed = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin, solids=(_slab(),))
    assert np.abs(_angles(pushed.pose) - _angles(none.pose)).max() > 1e-2
    # A grip ball is per side: one keyed to the other hand's group skips nothing here.
    other = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin, solids=(
        _slab(skip={"hand.L": ((np.array([0.15, -0.02, 0.0]), 0.05),)}),))
    assert np.allclose(_angles(other.pose), _angles(pushed.pose), atol=1e-6)


def test_skin_points_blend_two_bones():
    rig = _planar()
    pose = {"lower": ((0.0, 0.0, 0.5), (0.0, 0.0, 0.0)), "hand": ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0))}
    world = pose_world(rig, pose)
    cuff = SkinPoints(bones=np.array([["lower", "hand"]]), weights=np.array([[0.5, 0.5]]),
                      local=np.array([[[0.0, 0.05, 0.0], [0.0, -0.05, 0.0]]]))
    got = cuff.posed(world)[0]
    from_lower = world["lower"][:3, :3] @ [0.0, 0.05, 0.0] + world["lower"][:3, 3]
    from_hand = world["hand"][:3, :3] @ [0.0, -0.05, 0.0] + world["hand"][:3, 3]
    assert np.allclose(got, 0.5 * from_lower + 0.5 * from_hand, atol=1e-12)
    assert np.linalg.norm(from_lower - from_hand) > 0.01  # the two bones really disagree


def test_skin_points_reject_bad_weights():
    with pytest.raises(ManifestError, match="weights"):
        SkinPoints(bones=np.array([["lower", "hand"]]), weights=np.array([[0.5, 0.4]]),
                   local=np.zeros((1, 2, 3)))
    with pytest.raises(ManifestError, match="shape"):
        SkinPoints(bones=np.array([["lower"]]), weights=np.array([[1.0]]), local=np.zeros((1, 2, 3)))


def test_back_points_push_out_of_chunk():
    rig, chain = _planar(), _chain()
    # The solid's own mesh is far away: only its back points matter. One sits where the hand
    # will be; the hand's skin chunk is a 4 cm box around the hand bone's head.
    far = Solid(name="gear", index=SurfaceIndex(box((5, 5, 5), (6, 6, 6))), margins={"hand": 0.002},
                back_points=np.array([[0.16, 0.0, 0.0], [0.2, 0.05, 0.0]]))
    chunk_tris = box((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02))
    chunks = (Chunk(bone="hand", index=SurfaceIndex(chunk_tris)),)
    target = np.array([0.15, 0.0, 0.0])

    def inside_chunk(pose) -> np.ndarray:
        m = np.linalg.inv(pose_world(rig, pose)["hand"])
        local = (m[:3, :3] @ far.back_points.T).T + m[:3, 3]
        return np.all(np.abs(local) < 0.02, axis=1)

    free = solve_chain(rig, {}, chain, target, args=PLAIN)
    assert inside_chunk(free.pose)[0]  # the gear point is in the hand without the solid
    pushed = solve_chain(rig, {}, chain, target, args=PLAIN, solids=(far,), chunks=chunks)
    assert not inside_chunk(pushed.pose).any()


def _press_gear(margin: float) -> tuple[Solid, tuple[Chunk, ...]]:
    """A far-off solid with one back point where the hand will be, and the hand's 4 cm box."""
    gear = Solid(name="gear", index=SurfaceIndex(box((5, 5, 5), (6, 6, 6))),
                 margins={"hand": margin}, back_points=np.array([[0.16, 0.0, 0.0]]))
    chunk = Chunk(bone="hand", index=SurfaceIndex(box((-0.02, -0.02, -0.02), (0.02, 0.02, 0.02))))
    return gear, (chunk,)


def _depth_in_hand_box(rig, pose, point) -> float:
    """How far inside the hand's 2 cm half-width box the point sits (negative: outside)."""
    m = np.linalg.inv(pose_world(rig, pose)["hand"])
    local = m[:3, :3] @ point + m[:3, 3]
    return 0.02 - float(np.abs(local).max())


def test_press_margin_below_zero_still_pushes():
    # A press zone sets the hand's margin below zero: the back point may sit that deep in the
    # hand. One deeper than that is still pushed, out to the margin and no further.
    rig, chain = _planar(), _chain()
    gear, chunks = _press_gear(-0.008)
    free = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN)
    pushed = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, solids=(gear,), chunks=chunks)
    point = gear.back_points[0]
    before = _depth_in_hand_box(rig, free.pose, point)
    after = _depth_in_hand_box(rig, pushed.pose, point)
    assert before > 0.015  # the point is deep in the hand without the solid
    assert after < before - 0.005  # a non-zero push
    assert after > 0.008 - 2e-3  # it stops at the press depth, not at the surface


def test_press_margin_deeper_than_the_point_does_not_push():
    rig, chain = _planar(), _chain()
    gear, chunks = _press_gear(-0.03)  # the point is inside by less than this
    free = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN)
    held = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, solids=(gear,), chunks=chunks)
    assert np.allclose(_angles(held.pose), _angles(free.pose), atol=1e-3)


def test_phase_two_starts_from_phase_one(monkeypatch):
    rig, chain = _planar(), _chain()
    skin = _skin(hand=UNDER_HAND)
    real = solve_module.nelder
    calls = []

    def spy(f, x0, step, *a, **kw):
        best = real(f, x0, step, *a, **kw)
        calls.append((list(x0), step, best[0], f))
        return best

    monkeypatch.setattr(solve_module, "nelder", spy)
    start = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN)
    calls.clear()
    solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, x0=start.pose, skin=skin,
                solids=(_slab(),))
    assert [c[1] for c in calls[:2]] == list(STEPS)
    assert [c[1] for c in calls[2:4]] == list(SOLID_STEPS)
    assert calls[2][0] == calls[1][2]  # phase two starts at phase one's best
    assert calls[4][0] == calls[0][0] and calls[4][1] == SOLID_STEPS[0]  # and again from x0
    # Phase one costs no solid term; phase two does. The same point costs more in phase two.
    x = np.array(calls[1][2])
    assert calls[2][3](x) > calls[1][3](x) + 1e-6


def _scaled(vec, k):
    return np.asarray(vec, dtype=float) * k


def _case_avoid(k, on):
    shape = Avoid(id="torso", bone="base", kind="box",
                  centre=tuple(_scaled((0.09, -0.04, 0.0), k)),
                  half=tuple(_scaled((0.01, 0.01, 0.05), k)), why="test")
    return {"target": _scaled(REACH_TARGET, k), "x0": NEAR, "avoid": (shape,),
            "args": ChainArgs(avoid_clearance=-0.04 * k if on else 0.0)}


def _case_ellipsoid(k, on):
    head = Avoid(id="head", bone="base", kind="ellipsoid",
                 centre=tuple(_scaled((0.12, 0.05, 0.0), k)),
                 half=tuple(_scaled((0.04, 0.04, 0.04), k)), why="test")
    near = {"upper": ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0)),
            "lower": ((0.0, 0.0, -1.7), (0.0, 0.0, 0.0))}
    return {"target": _scaled(REACH_TARGET, k), "x0": near, "avoid": (head,),
            "args": ChainArgs(avoid_clearance=0.004 * k if on else 0.0)}


def _case_plane(k, on):
    skin = _skin(hand=HAND_PLATE)
    skin = {b: SkinPoints(p.bones, p.weights, p.local * k) for b, p in skin.items()}
    plane = Plane(point=_scaled((0.0, 0.03, 0.0), k), normal=np.array([0.0, 1.0, 0.0]),
                  margin=0.002 * k)
    return {"target": _scaled((0.12, 0.02, 0.0), k), "skin": skin,
            "pose": {"hand": ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))},
            "args": ChainArgs(plane=plane if on else None)}


def _case_slab(k, on, skip=True):
    # The second hand point is deeper in the slab than the first. The grip ball covers it at the
    # solve, so the slab skips it and pushes only the first; without the ball it pushes both.
    skin = _skin(hand=UNDER_HAND + [(-0.04, -0.02, 0.0)], lower=[(0.05, -0.02, 0.0)])
    skin = {b: SkinPoints(p.bones, p.weights, p.local * k) for b, p in skin.items()}
    tris = box((-0.1, -0.2, -0.1), (0.3, -0.01, 0.1)) * k
    ball = (_scaled((0.13, -0.035, 0.0), k), 0.03 * k)
    slab = Solid(name="slab", index=SurfaceIndex(tris), margins={"hand": 0.005 * k},
                 held_by="hand.L", skip={"hand": (ball,)} if skip else {})
    return {"target": _scaled(SOLID_TARGET, k), "skin": skin,
            "solids": (slab,) if on else (), "args": PLAIN}


def test_the_slab_cases_grip_ball_frees_its_point():
    def run(skip):
        spec = _case_slab(1.0, True, skip)
        return solve_chain(_planar(), {}, _chain(), spec.pop("target"), **spec)

    gripped, bare = run(True), run(False)
    assert np.abs(_angles(gripped.pose) - _angles(bare.pose)).max() > 1e-3


def _case_back(k, on):
    gear = Solid(name="gear", index=SurfaceIndex(box((5, 5, 5), (6, 6, 6)) * k),
                 margins={"hand": 0.002 * k}, back_points=_scaled([[0.16, 0.0, 0.0]], k))
    chunk = Chunk(bone="hand", index=SurfaceIndex(box((-0.02,) * 3, (0.02,) * 3) * k))
    return {"target": _scaled((0.15, 0.0, 0.0), k), "solids": (gear,) if on else (),
            "chunks": (chunk,), "args": PLAIN}


@pytest.mark.parametrize("case", [_case_avoid, _case_ellipsoid, _case_plane, _case_slab,
                                  _case_back],
                         ids=["avoid box", "avoid ellipsoid", "plane", "solid skin with a grip ball",
                              "solid back points"])
def test_scale_free_with_solids(case):
    # Rig, solids, skin, shapes and target all scaled by 10 give the same angles: each gap
    # enters the cost in reference-rig units, not as a raw world length.
    def run(k, on=True):
        spec = case(k, on)
        return solve_chain(_planar(k), spec.pop("pose", {}), _chain(radius=k * RADIUS),
                           spec.pop("target"), **spec)

    small, big = run(1.0), run(10.0)
    assert np.abs(_angles(small.pose) - _angles(big.pose)).max() < 1e-6
    assert big.err == pytest.approx(10.0 * small.err, abs=1e-9)
    # The term is live: with it off the answer is elsewhere.
    assert np.abs(_angles(small.pose) - _angles(run(1.0, on=False).pose)).max() > 1e-3


def test_shapes_and_solids_may_ride_bones_off_the_chain():
    rig, chain = _two_arms(), _arm_chain("R")
    shape = Avoid(id="torso", bone="upL", kind="box", centre=(0.09, -0.04, 0.0),
                  half=(0.01, 0.01, 0.05), why="test")
    args = ChainArgs(avoid_clearance=-0.04)
    on_left = solve_chain(rig, {}, chain, REACH_TARGET, args=args, x0=NEAR_R, avoid=(shape,))
    on_base = solve_chain(rig, {}, chain, REACH_TARGET, args=args, x0=NEAR_R,
                          avoid=(Avoid("torso", "base", "box", shape.centre, shape.half, "test"),))
    assert np.array_equal(_angles(on_left.pose, ("upR", "loR", "haR")),
                          _angles(on_base.pose, ("upR", "loR", "haR")))
    # A solid on the left hand moves with the left hand's pose, which the right arm's solve
    # reads but does not turn. Its slab sits 0.35 along X of haL, so it sits under the right hand.
    skin = _skin(haR=UNDER_HAND)
    slab = Solid(name="slab", index=SurfaceIndex(box((0.25, -0.2, -0.1), (0.45, -0.01, 0.1))),
                 margins={"haR": 0.005}, bone="haL")
    none = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin)
    lifted = {"haL": ((0.0, 0.0, 0.0), (0.0, -0.1, 0.0))}  # the slab drops out of the way
    away = solve_chain(rig, lifted, chain, SOLID_TARGET, args=PLAIN, skin=skin, solids=(slab,))
    here = solve_chain(rig, {}, chain, SOLID_TARGET, args=PLAIN, skin=skin, solids=(slab,))
    names = ("upR", "loR", "haR")
    assert np.abs(_angles(away.pose, names) - _angles(none.pose, names)).max() < 1e-3
    assert np.abs(_angles(here.pose, names) - _angles(none.pose, names)).max() > 1e-2


@pytest.mark.parametrize("make", [
    lambda: Tool(tip=np.zeros(2), slide=0.0),
    lambda: Plane(point=np.zeros(3), normal=np.zeros(3), margin=0.0),
    lambda: ChainArgs(level=np.eye(2)),
    lambda: Solid(name="s", index=SurfaceIndex(box((0, 0, 0), (1, 1, 1))), margins={"a": 0.0},
                  back_points=np.zeros((2, 2))),
], ids=["tool tip", "plane normal", "level shape", "back points"])
def test_a_bad_record_raises(make):
    with pytest.raises(ManifestError):
        make()


def test_avoid_clearance_without_shapes_raises():
    with pytest.raises(ManifestError, match="no avoid shapes"):
        solve_chain(_planar(), {}, _chain(), REACH_TARGET, args=ChainArgs(avoid_clearance=0.004))


def test_skin_on_a_bone_off_the_chain_raises():
    skin = _skin(base=[(0.0, 0.0, 0.0)])
    slab = _slab()
    slab = Solid(name="slab", index=slab.index, margins={"base": 0.0})
    with pytest.raises(ManifestError, match="not on the chain"):
        solve_chain(_planar(), {}, _chain(), SOLID_TARGET, args=PLAIN, skin=skin, solids=(slab,))


# --- whole clips ------------------------------------------------------------------------------

AVOID_NONE: tuple = ()


NEAR_R = {"upR": NEAR["upper"], "loR": NEAR["lower"]}


def _two_arms() -> RigData:
    """A base with a right arm along +X and a left arm along -X, bones of 0.1 each."""
    heads = {"base": 0.0, "upR": 0.0, "loR": BONE, "haR": 2 * BONE,
             "upL": 0.0, "loL": -BONE, "haL": -2 * BONE}
    parent = {"base": None, "upR": "base", "loR": "upR", "haR": "loR",
              "upL": "base", "loL": "upL", "haL": "loL"}
    return RigData(order=tuple(heads), parent=parent,
                   rest={b: _translate(x, 0.0, 0.0) for b, x in heads.items()}, world=np.eye(4),
                   effector_mid={"haR": np.zeros(3), "haL": np.zeros(3)})


def _arm_chain(side: str) -> Chain:
    bones = (f"up{side}", f"lo{side}", f"ha{side}")
    return Chain(root="base", bones=bones, effector=bones[-1], effector_radius=RADIUS,
                 dofs={b: "z" for b in bones}, limits={}, effector_group="hand")


def _spec(chains: dict[str, Chain], avoid=()) -> RigSpec:
    return RigSpec(armature="Test", bones=(Bone("base", None, 0.1),), bind={},
                   contact_groups={}, chains=chains, sit=SitSpec(0.1, 0.1),
                   mirror=MirrorSpec(pairs=(), axis="x"), avoid=tuple(avoid))


def _target(key: int, chain: str, tool: str | None = None) -> Target:
    return Target(key=key, chain=chain, ref=Ref.parse("rig:base"), offset=(0.0, 0.0, 0.0),
                  why=None, baked=None, tool=tool)


def _clip(targets, loop: bool = False) -> Clip:
    return Clip(name="c", frames=(1, 9), fps=24.0, authoring="solved", seat=None, holds=(),
                targets=tuple(targets), floor_gear=FloorGear({}), props={}, mirror_of=None,
                loop=loop)


def _run(rig, chains, targets, points, *, loop=False, args=None, base=None, **kw):
    """`points` maps (key, chain) to the world target; `args` maps the same to a ChainArgs."""
    spec = _spec(chains)
    return solve_clip(
        rig, spec, _clip(targets, loop), lambda t, pose: np.asarray(points[(t.key, t.chain)]),
        base=base or (lambda frame: {}), args=lambda t, pose: (args or {}).get((t.key, t.chain), PLAIN),
        solids=kw.get("solids", ()), skin=kw.get("skin"), chunks=kw.get("chunks", ()))


ARM_R = {"R": _arm_chain("R")}
PTS_R = {(1, "R"): [0.12, 0.05, 0.0], (5, "R"): [0.1, 0.1, 0.0], (9, "R"): [0.12, 0.05, 0.0]}


def test_solve_clip_chains_keys(monkeypatch):
    rig = _two_arms()
    starts = []
    real = solve_module.solve_chain

    def spy(rig_, pose, chain, target, **kw):
        starts.append(kw.get("x0"))
        return real(rig_, pose, chain, target, **kw)

    monkeypatch.setattr(solve_module, "solve_chain", spy)
    out = _run(rig, ARM_R, [_target(k, "R") for k in (1, 5, 9)], PTS_R)
    assert set(out) == {(1, "R"), (5, "R"), (9, "R")}
    assert starts[0] is None
    assert starts[1] is out[(1, "R")].pose  # each key starts from the chain's last solve
    assert starts[2] is out[(5, "R")].pose
    for result in out.values():
        assert result.err < 1.5e-2


def test_solve_clip_sorts_targets_by_key(monkeypatch):
    rig = _two_arms()
    order = []
    real = solve_module.solve_chain
    monkeypatch.setattr(solve_module, "solve_chain", lambda r, p, c, t, **kw: (
        order.append(tuple(np.round(t, 3))), real(r, p, c, t, **kw))[1])
    _run(rig, ARM_R, [_target(k, "R") for k in (9, 1, 5)], PTS_R)
    assert order == [(0.12, 0.05, 0.0), (0.1, 0.1, 0.0), (0.12, 0.05, 0.0)]


def test_solve_clip_two_chains_one_key():
    rig = _two_arms()
    chains = {"R": _arm_chain("R"), "L": _arm_chain("L")}
    points = {(1, "R"): [0.12, 0.05, 0.0], (1, "L"): [-0.12, 0.05, 0.0],
              (5, "R"): [0.1, 0.1, 0.0], (5, "L"): [-0.1, 0.1, 0.0]}
    targets = [_target(1, "R"), _target(1, "L"), _target(5, "R"), _target(5, "L")]
    out = _run(rig, chains, targets, points)
    assert set(out) == {(1, "R"), (1, "L"), (5, "R"), (5, "L")}
    # Each chain solves on its own bones; the later one also carries the earlier one's.
    assert {"upR", "loR"} <= set(out[(1, "R")].pose)
    assert not set(out[(1, "R")].pose) & {"upL", "loL", "haL"}
    assert {"upL", "loL"} <= set(out[(1, "L")].pose)
    for result in out.values():
        assert result.err < 1.5e-2


def test_solve_clip_unknown_chain_raises():
    rig = _two_arms()
    with pytest.raises(ManifestError, match="chain 'X'"):
        _run(rig, ARM_R, [_target(1, "X")], {(1, "X"): [0.1, 0.1, 0.0]})


def _two_arm_scene():
    """Both arms at one key, skin and a chunk for each hand, and a slab under both."""
    chains = {"R": _arm_chain("R"), "L": _arm_chain("L")}
    points = {(1, "R"): [0.15, 0.0, 0.0], (1, "L"): [-0.15, 0.0, 0.0]}
    targets = [_target(1, "R"), _target(1, "L")]
    skin = _skin(haR=UNDER_HAND, haL=UNDER_HAND)
    slab = Solid(name="slab", index=SurfaceIndex(box((-0.3, -0.2, -0.1), (0.3, -0.01, 0.1))),
                 margins={"haR": 0.005, "haL": 0.005})
    gear = Solid(name="gear", index=SurfaceIndex(box((5, 5, 5), (6, 6, 6))),
                 margins={"haR": 0.002, "haL": 0.002}, back_points=np.array([[0.16, 0.0, 0.0]]))
    cube = box((-0.02,) * 3, (0.02,) * 3)
    chunks = tuple(Chunk(bone=b, index=SurfaceIndex(cube)) for b in ("haR", "haL"))
    return _two_arms(), chains, targets, points, skin, (slab, gear), chunks


def test_solve_clip_two_arms_with_solids():
    rig, chains, targets, points, skin, solids, chunks = _two_arm_scene()
    out = _run(rig, chains, targets, points, solids=solids, skin=skin, chunks=chunks)
    assert set(out) == {(1, "R"), (1, "L")}
    for result in out.values():
        assert result.err < 0.03
    # Each hand's skin point is out of the slab by its margin, up to the solve's precision.
    for side, bone in (("R", "haR"), ("L", "haL")):
        under = skin[bone].posed(pose_world(rig, out[(1, side)].pose))[0, 1]
        assert under > -0.01 + 0.005 - 2e-3


def test_each_arm_solves_against_its_own_skin_and_chunks(monkeypatch):
    rig, chains, targets, points, skin, solids, chunks = _two_arm_scene()
    seen = []
    real = solve_module._solid_term

    def spy(chain, solids_, skin_, chunks_, length):
        seen.append((chain.effector, sorted(skin_), sorted(c.bone for c in chunks_)))
        return real(chain, solids_, skin_, chunks_, length)

    monkeypatch.setattr(solve_module, "_solid_term", spy)
    _run(rig, chains, targets, points, solids=solids, skin=skin, chunks=chunks)
    assert sorted(seen) == [("haL", ["haL"], ["haL"]), ("haR", ["haR"], ["haR"])]


def test_a_later_chain_at_a_key_sees_the_earlier_chains_solve(monkeypatch):
    rig, chains, targets, points, *_ = _two_arm_scene()
    poses = {}
    real = solve_module.solve_chain

    def spy(r, pose, chain, target, **kw):
        poses[chain.effector] = pose
        return real(r, pose, chain, target, **kw)

    monkeypatch.setattr(solve_module, "solve_chain", spy)
    asked = {}

    def resolve(t, pose):
        asked[t.chain] = pose
        return np.asarray(points[(t.key, t.chain)])

    def args(t, pose):
        asked.setdefault("args " + t.chain, pose)
        return PLAIN

    out = solve_clip(rig, _spec(chains), _clip(targets), resolve, base=lambda frame: {},
                     args=args)
    right = out[(1, "R")].pose
    for bone in ("upR", "loR"):  # the hand is the effector: not turned without a level
        assert bone not in poses["haR"]  # the first chain starts from the base pose
        for got in (poses["haL"], asked["L"], asked["args L"]):
            assert np.array_equal(got[bone][0], right[bone][0])  # the second sees it solved
    assert not {"upL", "loL", "haL"} & set(asked["R"])


def test_skin_and_chunks_on_no_chain_raise():
    rig, chains, targets, points, skin, solids, chunks = _two_arm_scene()
    # build_clips-style keys name no bone of the rig's chains.
    with pytest.raises(ManifestError, match="skin.*'hand'.*no chain"):
        _run(rig, chains, targets, points, solids=solids, skin=_skin(hand=UNDER_HAND))
    stray = (*chunks, Chunk(bone="forearm", index=chunks[0].index))
    with pytest.raises(ManifestError, match="chunk.*'forearm'.*no chain"):
        _run(rig, chains, targets, points, solids=solids, skin=skin, chunks=stray)


def test_the_old_tuple_form_of_skip_raises():
    with pytest.raises(ManifestError, match="skip"):
        Solid(name="s", index=SurfaceIndex(box((0, 0, 0), (1, 1, 1))), margins={"a": 0.0},
              skip=((np.zeros(3), 0.1),))


def test_a_later_key_does_not_inherit_an_earlier_keys_overlay():
    rig, chains, _, _, *_ = _two_arm_scene()
    seen = {}

    def resolve(t, pose):
        seen[(t.key, t.chain)] = sorted(pose)
        return np.array([0.12, 0.05, 0.0]) * (1 if t.chain == "R" else -1)

    targets = [_target(1, "R"), _target(5, "R")]
    solve_clip(rig, _spec(chains), _clip(targets), resolve, base=lambda frame: {},
               args=lambda t, pose: PLAIN)
    assert seen[(5, "R")] == []  # key 5's pose is the base, not key 1's solve


def _stretched(world_scale=1.0, shear=False) -> RigData:
    rig = _planar()
    world = np.diag([world_scale, world_scale, world_scale, 1.0])
    if shear:
        rig.rest["base"] = rig.rest["base"].copy()
        rig.rest["base"][0, 1] = 0.2
    rig.world = world
    return rig


def _scaled_ancestor() -> RigData:
    """The planar chain under a new root `torso` whose rest is stretched 2x along X. Its own
    rest and `base`'s rest are rigid, but a turned `torso` shears everything under it."""
    rig = _planar()
    rest = {**rig.rest, "torso": np.diag([2.0, 1.0, 1.0, 1.0])}
    parent = {"torso": None, **{b: (p or "torso") for b, p in rig.parent.items()}}
    return RigData(order=("torso", *rig.order), parent=parent, rest=rest, world=np.eye(4),
                   effector_mid=rig.effector_mid)


@pytest.mark.parametrize("make", [lambda: _stretched(2.0), lambda: _stretched(shear=True),
                                  _scaled_ancestor],
                         ids=["scaled world", "sheared shape bone", "scaled ancestor"])
def test_shapes_need_rigid_bone_matrices(make):
    shape = Avoid(id="torso", bone="base", kind="box", centre=(0.09, -0.04, 0.0),
                  half=(0.01, 0.01, 0.05), why="test")
    args = ChainArgs(avoid_clearance=-0.04)
    with pytest.raises(ManifestError, match="rigid"):
        solve_chain(make(), {}, _chain(), REACH_TARGET, args=args, avoid=(shape,))
    # With nothing posed by a bone, a scaled world is fine: the plain reach is unchanged.
    solve_chain(make(), {}, _chain(), REACH_TARGET * 2.0, args=PLAIN)


def test_keep_last(monkeypatch):
    rig = _two_arms()
    calls = []
    real = solve_module.solve_chain
    monkeypatch.setattr(solve_module, "solve_chain", lambda r, p, c, t, **kw: (
        calls.append(1), real(r, p, c, t, **kw))[1])
    args = {(5, "R"): ChainArgs(keep_last=True)}
    out = _run(rig, ARM_R, [_target(1, "R"), _target(5, "R")], PTS_R, args=args)
    assert len(calls) == 1  # the held key did not solve
    assert np.array_equal(_angles(out[(5, "R")].pose, ("upR", "loR", "haR")),
                          _angles(out[(1, "R")].pose, ("upR", "loR", "haR")))
    # The miss is real: it is measured at the held arm against the new target.
    hand = effector(rig, out[(5, "R")].pose, ARM_R["R"])
    assert out[(5, "R")].err == pytest.approx(float(np.linalg.norm(hand - PTS_R[(5, "R")])),
                                              abs=1e-12)
    # With no earlier solve, keep_last has nothing to keep: the key solves.
    calls.clear()
    _run(rig, ARM_R, [_target(1, "R")], PTS_R, args={(1, "R"): ChainArgs(keep_last=True)})
    assert len(calls) == 1


def test_loop_end_reuses_key_zero():
    rig = _two_arms()
    targets = [_target(k, "R") for k in (1, 5, 9)]
    looped = _run(rig, ARM_R, targets, PTS_R, loop=True)
    bones = ("upR", "loR", "haR")
    assert np.array_equal(_angles(looped[(9, "R")].pose, bones), _angles(looped[(1, "R")].pose, bones))
    # Without `loop`, key 9 solves from key 5's answer and lands a little elsewhere: the pop.
    straight = _run(rig, ARM_R, targets, PTS_R, loop=False)
    assert not np.array_equal(_angles(straight[(9, "R")].pose, bones),
                              _angles(straight[(1, "R")].pose, bones))
    # A last key that differs from key 0 is solved, loop or not.
    other = dict(PTS_R)
    other[(9, "R")] = [0.11, 0.06, 0.0]
    moved = _run(rig, ARM_R, targets, {**other}, loop=True)
    assert not np.array_equal(_angles(moved[(9, "R")].pose, bones), _angles(moved[(1, "R")].pose, bones))


def test_tip_retry(monkeypatch):
    rig = _two_arms()
    chain = {"R": _arm_chain("R")}
    tool = Tool(tip=np.array([0.1, 0.0, 0.0]), slide=0.0)
    args = {(1, "R"): ChainArgs(level=np.eye(3), level_weight=1.0, tool=tool)}
    points = {(1, "R"): [0.25, 0.12, 0.0]}
    seen = []
    real = solve_module.solve_chain

    def spy(r, p, c, t, **kw):
        out = real(r, p, c, t, **kw)
        seen.append((np.array(t), kw.get("quick"), out))
        return out

    monkeypatch.setattr(solve_module, "solve_chain", spy)
    out = _run(rig, chain, [_target(1, "R", tool="stick")], points, args=args)
    first_miss = seen[0][2].err
    assert first_miss > TIP_RETRY_MISS_OF_EFFECTOR * RADIUS  # the first solve misses the tip
    assert 2 <= len(seen) <= 1 + TIP_RETRIES
    assert all(quick for _, quick, _ in seen[1:])  # retries start from the last answer
    final = out[(1, "R")]
    assert final.err < first_miss  # and the retries brought the tip closer
    # The reported miss is against the real target, not the shifted one the retry aimed at.
    tip = _posed(rig, final.pose, "haR", tool.tip)
    assert final.err == pytest.approx(float(np.linalg.norm(tip - points[(1, "R")])), abs=1e-12)
    # A tip that already hits retries nothing.
    seen.clear()
    easy = {(1, "R"): [0.3, 0.0, 0.0]}  # the arm straight: already there
    _run(rig, chain, [_target(1, "R", tool="stick")], easy,
         args={(1, "R"): ChainArgs(tool=tool)})
    assert len(seen) == 1


def test_quick_solve_that_misses_runs_again_in_full(monkeypatch):
    rig = _two_arms()
    tool = Tool(tip=np.array([0.1, 0.0, 0.0]), slide=0.0)
    turned = np.array([[np.cos(2.356), -np.sin(2.356), 0.0], [np.sin(2.356), np.cos(2.356), 0.0],
                       [0.0, 0.0, 1.0]])  # the tool turned 135 degrees: a long way for the wrist
    args = {(1, "R"): ChainArgs(level=np.eye(3), level_weight=1.0, tool=tool),
            (5, "R"): ChainArgs(level=turned, level_weight=1.0, tool=tool)}
    points = {(1, "R"): [0.3, 0.0, 0.0], (5, "R"): [0.1, 0.2, 0.0]}
    seen = []
    real = solve_module.solve_chain

    def spy(r, p, c, t, **kw):
        out = real(r, p, c, t, **kw)
        seen.append((bool(kw.get("quick")), out.err))
        return out

    monkeypatch.setattr(solve_module, "solve_chain", spy)
    _run(rig, ARM_R, [_target(1, "R", "stick"), _target(5, "R", "stick")], points, args=args)
    assert seen[0][0] is False  # key 1 has no last solve to be quick from
    first_quick = next(i for i, (quick, _) in enumerate(seen) if quick)
    assert seen[first_quick][1] > QUICK_RETRY_MISS_OF_REACH * 0.2  # missed by over 1% of reach
    assert seen[first_quick + 1][0] is False  # so it ran again from every start


def test_quick_solve_that_hits_runs_once(monkeypatch):
    rig = _two_arms()
    tool = Tool(tip=np.array([0.1, 0.0, 0.0]), slide=0.0)
    args = {(k, "R"): ChainArgs(level=np.eye(3), level_weight=1.0, tool=tool) for k in (1, 5)}
    points = {(1, "R"): [0.3, 0.0, 0.0], (5, "R"): [0.3, 0.0, 0.0]}
    seen = []
    real = solve_module.solve_chain
    monkeypatch.setattr(solve_module, "solve_chain", lambda r, p, c, t, **kw: (
        seen.append(bool(kw.get("quick"))), real(r, p, c, t, **kw))[1])
    _run(rig, ARM_R, [_target(1, "R", "stick"), _target(5, "R", "stick")], points, args=args)
    assert seen == [False, True]


# --- the poses between keys --------------------------------------------------------------------

TORSO_LO, TORSO_HI = (-0.06, -0.03, -0.1), (0.06, 0.03, 0.1)  # across the hand's straight path
TORSO = box(TORSO_LO, TORSO_HI)
SWING = {(1, "R"): [0.0, 0.15, 0.0], (9, "R"): [0.0, -0.15, 0.0]}  # over the torso, then under


def _torso(offset: float = 0.0) -> Solid:
    margin = SOLID_MARGIN_OF_EFFECTOR * RADIUS  # the kit's keep-out for a body solid
    return Solid(name="torso", index=SurfaceIndex(TORSO + [offset, 0.0, 0.0]),
                 margins={"haR": margin})


def _inside_torso(rig, pose) -> bool:
    """Whether the hand's skin point is inside the torso box. Written from the box's corners,
    so it does not lean on the solver's own measure."""
    p = _posed(rig, pose, "haR")
    return bool(np.all(p > TORSO_LO) and np.all(p < TORSO_HI))


def _between(a: Pose, b: Pose, t: float) -> Pose:
    return {bone: ((1 - t) * np.asarray(a[bone][0]) + t * np.asarray(b[bone][0]),
                   (1 - t) * np.asarray(a[bone][1]) + t * np.asarray(b[bone][1]))
            for bone in a}


def _swing(frames=(1, 9), offset: float = 0.0):
    """The right hand swings from above the torso to below it, with the torso in the way."""
    rig = _two_arms()
    points = {(f, "R"): p for f, p in zip(frames, SWING.values())}
    out = _run(rig, ARM_R, [_target(f, "R") for f in frames], points, solids=(_torso(offset),),
               skin=_skin(haR=[(0, 0, 0)]))
    return rig, out


def _frames(out, chain="R") -> list[int]:
    return sorted(frame for frame, name in out if name == chain)


def _hand_in_torso_between(rig, out, frames, t_steps=None) -> list[tuple[int, int, float]]:
    """Each (key, next key, blend fraction) whose blended pose has the hand inside the torso.
    Every whole frame the player shows, or `t_steps` samples between each pair of keys."""
    found = []
    for fa, fb in pairwise(frames):
        a, b = out[(fa, "R")].pose, out[(fb, "R")].pose
        ts = ([(f - fa) / (fb - fa) for f in range(fa, fb + 1)] if t_steps is None
              else np.linspace(0.0, 1.0, t_steps))
        found += [(fa, fb, float(t)) for t in ts if _inside_torso(rig, _between(a, b, t))]
    return found


def test_in_between_sweep_through_body_gets_a_mid_key():
    rig, out = _swing()
    # The swing is a real one: straight between the two authored keys, the hand is in the torso.
    first, last = out[(1, "R")].pose, out[(9, "R")].pose
    assert any(_inside_torso(rig, _between(first, last, t)) for t in np.linspace(0, 1, 201))
    frames = _frames(out)
    assert frames[0] == 1 and frames[-1] == 9 and len(frames) > 2
    # The player shows whole frames only: the hand is out of the torso at every one of them.
    assert _hand_in_torso_between(rig, out, frames) == []
    # The pass condition: for every two consecutive keys in the result, either no dense sample
    # between them is inside the torso, or the earlier key's `sweep_to_next` reports a depth > 0.
    # (A mid-key sits on a whole frame, so a sweep it cannot clear is reported, never silent.)
    for fa, fb in pairwise(frames):
        dense = _hand_in_torso_between(rig, out, [fa, fb], t_steps=401)
        assert not dense or out[(fa, "R")].sweep_to_next > 0.0, (fa, fb, len(dense))
    assert out[(frames[-1], "R")].sweep_to_next == 0.0
    # On this swing the pair of keys 2 and 3 is the one the whole-frame mid-keys cannot clear.
    assert frames[1:3] == [2, 3]
    assert _hand_in_torso_between(rig, out, [2, 3], t_steps=401)
    assert out[(2, "R")].sweep_to_next > 0.0


def test_mid_keys_do_not_move_the_authored_keys(monkeypatch):
    _, with_mid = _swing()
    monkeypatch.setattr(solve_module, "_fill_between", lambda *a, **k: ({}, []))
    _, without = _swing()
    assert _frames(without) == [1, 9] and len(_frames(with_mid)) > 2
    for slot in without:
        for bone, (euler, loc) in without[slot].pose.items():
            assert np.array_equal(with_mid[slot].pose[bone][0], euler)
            assert np.array_equal(with_mid[slot].pose[bone][1], loc)
        assert with_mid[slot].err == without[slot].err


def test_mid_keys_clear_the_sweeps_they_can():
    # The mid-keys are not decoration: most pairs the swing is cut into are clean when sampled
    # densely, and only the reported ones are not.
    rig, out = _swing()
    frames = _frames(out)
    dense = {fa: _hand_in_torso_between(rig, out, [fa, fb], t_steps=401)
             for fa, fb in pairwise(frames)}
    assert sum(not d for d in dense.values()) >= len(dense) - 1
    assert all(out[(fa, "R")].sweep_to_next > 0.0 for fa, d in dense.items() if d)


def test_a_clean_in_between_gets_no_mid_key():
    _, out = _swing(offset=1.0)  # the torso stands well clear of the swing
    assert _frames(out) == [1, 9]
    assert all(r.sweep_to_next == 0.0 for r in out.values())


def test_keys_one_frame_apart_get_no_mid_key_and_report_the_sweep():
    # The player shows no frame between them, so there is no whole frame to put a key on. The
    # sweep is not dropped: the earlier key says how deep the blend goes.
    rig, out = _swing(frames=(1, 2))
    assert _frames(out) == [1, 2]
    a, b = out[(1, "R")].pose, out[(2, "R")].pose  # the blend does cross the torso in between
    assert any(_inside_torso(rig, _between(a, b, t)) for t in np.linspace(0, 1, 401))
    assert out[(1, "R")].sweep_to_next > 0.0
    assert out[(2, "R")].sweep_to_next == 0.0


def test_solids_without_skin_solve_and_check_nothing_between():
    # `skin` defaults to None: no skin point to sweep, so the pass adds nothing and does not crash.
    rig = _two_arms()
    out = _run(rig, ARM_R, [_target(1, "R"), _target(9, "R")], SWING, solids=(_torso(),))
    assert _frames(out) == [1, 9]


# --- progress of a long solve -----------------------------------------------------------------


def test_solve_prints_one_progress_line_per_key(capsys):
    # A solve can run for hours; a caller polling a background run reads one line per key.
    _run(_two_arms(), ARM_R, [_target(k, "R") for k in (9, 1, 5)], PTS_R)
    seen = capsys.readouterr()
    lines = seen.err.splitlines()
    assert seen.out == ""  # stdout stays the caller's own
    assert len(lines) == 3
    for line, (done, key) in zip(lines, enumerate((1, 5, 9), 1), strict=True):
        assert re.fullmatch(
            rf"solve c: key {key} R \({done}/3\) took \d+\.\d s, \d+\.\d s in all", line), line


def test_solve_prints_a_line_when_an_in_between_check_is_done(capsys):
    _swing()  # two keys and a torso in the way: two key lines, then the pair between them
    lines = capsys.readouterr().err.splitlines()
    assert [line.split(": ")[1].split(" R")[0] for line in lines] == [
        "key 1", "key 9", "between keys 1 and 9"]
    assert re.search(r"R, \d+ mid-keys, took \d+\.\d s, \d+\.\d s in all$", lines[-1])


# --- one nearest-surface query per solid ------------------------------------------------------


class _CountingIndex(SurfaceIndex):
    """A SurfaceIndex that counts its `closest` calls."""

    calls = 0

    def closest(self, pts):
        type(self).calls += 1
        return super().closest(pts)


def _rod_gauge(index):
    """A solid gauge for a thin rod whose one end is 0.0059 from a hand skin point (inside the
    margin, not inside the rod), and the posed world."""
    rig, chain = _planar(), _chain()
    skin = _skin(hand=[(0.0, 0.0, 0.0)])
    rod = solve_module.Solid(name="rod", index=index, margins={"hand": 0.006})
    gauge = solve_module._solid_gauge(chain, (rod,), skin, [], reach(rig, chain))
    return rod, gauge, pose_world(rig, {})


def test_exits_from_the_margin_beyond_a_thin_rod_clear_every_way():
    # The start sits in the margin past the rod's end. Out the far way it must cross the rod and
    # its margin on both sides; a march of only diag + margin stopped short and raised IndexError.
    rod, gauge, world = _rod_gauge(
        SurfaceIndex(box((0.2059, -0.004, -0.004), (1.2059, 0.004, 0.004))))
    ways = gauge.exits(world, np.array([0.0, 1.0, 0.0]))
    assert len(ways) == 4  # y is the travel axis, so +-x and +-z remain
    start = _skin_world(_planar(), {}, _skin(hand=[(0.0, 0.0, 0.0)]), "hand")[0]
    for way in ways:
        near = rod.index.closest((start + way)[None])
        outside = np.einsum("ij,ij->i", start + way - near.point, near.normal) >= 0.0
        assert outside[0] and near.dist[0] >= 0.006


class _NeverClear(SurfaceIndex):
    """A SurfaceIndex that puts every point on its surface, so no march ever leaves it."""

    def closest(self, pts):
        near = super().closest(pts)
        return near._replace(dist=np.zeros(len(pts)), point=np.asarray(pts, dtype=float),
                             normal=np.tile([0.0, 0.0, 1.0], (len(pts), 1)))


def test_a_way_that_never_clears_is_skipped_not_indexed():
    _, gauge, world = _rod_gauge(
        _NeverClear(box((0.2059, -0.004, -0.004), (1.2059, 0.004, 0.004))))
    assert gauge.exits(world, np.array([0.0, 1.0, 0.0])) == []


def test_a_solids_skin_parts_share_one_query_and_each_keeps_its_own_answer():
    # Two skin parts with different margins against one slab. The gauge asks the slab once for
    # both; each part's cost must equal the one from asking the slab about that part alone.
    rig, chain = _planar(), _chain()
    skin = _skin(hand=[(0.0, -0.02, 0.0), (0.01, 0.0, 0.0), (-0.03, -0.015, 0.02)],
                 lower=[(0.0, -0.03, 0.0), (0.02, -0.012, -0.01)])
    tris = box((-0.1, -0.2, -0.1), (0.3, -0.01, 0.1))
    margins = {"hand": 0.005, "lower": 0.011}
    counting = Solid(name="slab", index=_CountingIndex(tris), margins=margins)
    plain = SurfaceIndex(tris)
    length = reach(rig, chain)
    gauge = solve_module._solid_gauge(chain, (counting,), skin, [], length)
    unit = SOLVER_REFERENCE_REACH / length
    for angle in (0.0, 0.3, -0.5):
        pose = {"upper": ((0.0, 0.0, angle), (0.0, 0.0, 0.0))}
        world = pose_world(rig, pose)
        _CountingIndex.calls = 0
        got = gauge.term(None, world, None)
        assert _CountingIndex.calls == 1
        want = 0.0
        for part, margin in margins.items():
            q = skin[part].posed(world)
            near = plain.closest(q)
            inside = np.einsum("ij,ij->i", q - near.point, near.normal) < 0.0
            pen = np.maximum(np.where(inside, near.dist + margin, margin - near.dist), 0.0)
            want += W_SOLID * float((pen * unit) @ (pen * unit))
        assert want > 0.0
        assert got == want  # the same bits, not just close
