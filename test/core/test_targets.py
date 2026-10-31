import dataclasses

import numpy as np
import pytest

from model_kit.core import targets
from model_kit.core.errors import AnchorMissing, ManifestError
from model_kit.core.manifest import SitSpec
from model_kit.core.targets import resolve_target, sample_at, target_world

from .scenes import make_ctx, quat_y, rest, turn_y, turn_z


def _target(ctx):
    return ctx.clips["reach"].targets[0]


def _with_target(ctx, **changes):
    clip = ctx.clips["reach"]
    moved = dataclasses.replace(clip, targets=(dataclasses.replace(clip.targets[0], **changes),))
    return dataclasses.replace(ctx, clips={"reach": moved})


def test_sample_at_is_the_nearest_time():
    samples = make_ctx().samples["reach"]  # times 1..10
    assert sample_at(samples, 1) == 0
    assert sample_at(samples, 5) == 4
    assert sample_at(samples, 5.4) == 4
    assert sample_at(samples, 5.6) == 5
    assert sample_at(samples, 0.6) == 0  # within half a step before the first sample
    assert sample_at(samples, 10.4) == 9


def test_sample_at_a_key_outside_the_samples_raises():
    samples = make_ctx().samples["reach"]  # times 1..10
    for t in (-3, 0.4, 10.6, 99):
        with pytest.raises(ManifestError, match="no sample near"):
            sample_at(samples, t)


def test_sample_at_uses_the_samples_own_step():
    coarse = dataclasses.replace(make_ctx().samples["reach"], times=np.arange(1.0, 20.0, 2.0))
    assert sample_at(coarse, 4) in (1, 2)  # 3 or 5, a full frame away but half a step
    with pytest.raises(ManifestError, match="no sample near"):
        sample_at(coarse, 21.5)


def test_sample_at_with_no_times_raises():
    samples = dataclasses.replace(make_ctx().samples["reach"], times=np.zeros(0))
    with pytest.raises(ManifestError, match="no times"):
        sample_at(samples, 1)


def test_placed_hold_point_plus_offset():
    ctx = make_ctx(at=(0.5, 0.0, 0.0), offset=(0.1, 0.0, 0.2))
    # hold point (0, 0.002, 0) at the placement, plus the offset
    np.testing.assert_allclose(target_world(ctx, "reach", _target(ctx)), (0.6, 0.002, 0.2))


def test_placement_turn_rotates_the_offset():
    # 90 degrees about Y: +x -> (0, 0, -1), +z -> (1, 0, 0)
    ctx = make_ctx(at=(0.5, 0.0, 0.0), quat=quat_y(90), offset=(0.3, 0.0, 0.5))
    np.testing.assert_allclose(target_world(ctx, "reach", _target(ctx)),
                               (0.5 + 0.5, 0.002, -0.3), atol=1e-12)


def test_carried_prop_uses_the_pose_at_the_key_sample():
    ctx = make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=turn_y(90), offset=(0.0, 0.0, 0.5))
    np.testing.assert_allclose(target_world(ctx, "reach", _target(ctx)), (1.0, 0.002, 0.0),
                               atol=1e-12)
    poses = np.array(ctx.samples["reach"].props["plate"])
    poses[4, :3, 3] = (2.0, 0.0, 0.0)  # the sample at frame 5 only
    samples = dataclasses.replace(ctx.samples["reach"], props={"plate": poses})
    moved = dataclasses.replace(ctx, samples={"reach": samples})
    np.testing.assert_allclose(target_world(moved, "reach", _target(moved)), (2.5, 0.002, 0.0),
                               atol=1e-12)
    later = _with_target(moved, key=6)
    np.testing.assert_allclose(target_world(later, "reach", _target(later)), (1.0, 0.002, 0.0),
                               atol=1e-12)


def test_bone_ref_uses_the_bone_head_and_frame():
    bones = {"chest": np.tile(rest(0.4, 0.0, 0.0, turn_z(90)), (10, 1, 1))}
    ctx = make_ctx(ref="rig:chest", bones=bones, offset=(0.5, 0.0, 0.0))
    np.testing.assert_allclose(target_world(ctx, "reach", _target(ctx)), (0.4, 0.5, 0.0),
                               atol=1e-12)


def test_scale_in_a_matrix_does_not_scale_the_offset():
    # an armature world scaled by 0.01: the bone matrix is a turn times 0.01, at (0.4, 0, 0)
    bones = {"chest": np.tile(rest(0.4, 0.0, 0.0, 0.01 * turn_z(90)), (10, 1, 1))}
    ctx = make_ctx(ref="rig:chest", bones=bones, offset=(0.5, 0.0, 0.0))
    point = target_world(ctx, "reach", _target(ctx))
    np.testing.assert_allclose(point, (0.4, 0.5, 0.0), atol=1e-12)
    assert np.linalg.norm(point - (0.4, 0.0, 0.0)) == pytest.approx(0.5)
    scaled = make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=0.01 * turn_y(90),
                      offset=(0.0, 0.0, 0.5))
    # The hold point (0, 0.002, 0) is on the prop's geometry, so the 0.01 pose scales it to
    # (0, 0.00002, 0). The offset keeps its length: 0.5 along the turned z, which is +x.
    point = target_world(scaled, "reach", _target(scaled))
    np.testing.assert_allclose(point, (1.0, 0.00002, 0.0), atol=1e-12)
    hold = resolve_target(scaled, "reach", _target(scaled)).point
    np.testing.assert_allclose(hold, (0.5, 0.00002, 0.0), atol=1e-12)
    assert np.linalg.norm(point - hold) == pytest.approx(0.5)


def test_a_reflection_raises():
    mirrored = np.diag([-1.0, 1.0, 1.0])
    bones = {"chest": np.tile(rest(0.4, 0.0, 0.0, mirrored), (10, 1, 1))}
    ctx = make_ctx(ref="rig:chest", bones=bones)
    with pytest.raises(ManifestError, match="reflection.*chest|chest.*reflection"):
        target_world(ctx, "reach", _target(ctx))
    prop = make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=mirrored)
    with pytest.raises(ManifestError, match="reflection"):
        target_world(prop, "reach", _target(prop))


def test_offset_to_inverts_with_offset():
    ctx = make_ctx(at=(0.5, 0.0, 0.0), quat=quat_y(90))
    resolved = resolve_target(ctx, "reach", _target(ctx))
    world = np.array([0.7, 0.3, -0.4])
    np.testing.assert_allclose(resolved.with_offset(resolved.offset_to(world)), world,
                               atol=1e-12)


def test_singular_frame_raises():
    bones = {"chest": np.tile(rest(0.4, 0.0, 0.0, np.zeros((3, 3))), (10, 1, 1))}
    ctx = make_ctx(ref="rig:chest", bones=bones)
    with pytest.raises(ManifestError, match="singular"):
        target_world(ctx, "reach", _target(ctx))


def test_misses_raise():
    ctx = make_ctx(ref="rig:chest")
    with pytest.raises(ManifestError, match="chest"):
        target_world(ctx, "reach", _target(ctx))
    bad = make_ctx(ref="kit:Plate.nowhere")
    with pytest.raises(AnchorMissing):
        target_world(bad, "reach", _target(bad))
    with pytest.raises(ManifestError, match="no samples"):
        target_world(make_ctx(samples=False), "reach", _target(make_ctx()))
    with pytest.raises(ManifestError, match="nope"):
        target_world(make_ctx(), "nope", _target(make_ctx()))


def test_sample_times_must_be_finite_and_strictly_increasing():
    samples = make_ctx().samples["reach"]
    for bad in ([1, 1, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
                [1, 3, 2, 4, 5, 6, 7, 8, 9, 10], [1, 2, np.inf, 4, 5, 6, 7, 8, 9, 10]):
        broken = dataclasses.replace(samples, times=np.array(bad, dtype=float))
        with pytest.raises(ManifestError, match="strictly increasing"):
            sample_at(broken, 1.4)


def test_uneven_sample_gaps_do_not_refuse_a_key_between_samples():
    uneven = dataclasses.replace(make_ctx().samples["reach"],
                                 times=np.array([1.0, 2, 3, 4, 5, 6, 7, 20, 40, 80]))
    assert sample_at(uneven, 31) == 8  # 40 is nearer than 20
    assert sample_at(uneven, 61) == 9  # 80 is nearer than 40
    with pytest.raises(ManifestError, match="no sample near"):
        sample_at(uneven, 130)


def test_a_scaled_carried_prop_resolved_in_piece_units():
    scaled = make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=0.1 * turn_y(90),
                      offset=(0.0, 0.0, 0.5))
    res = resolve_target(scaled, "reach", _target(scaled))
    np.testing.assert_allclose(res.offset_in_piece((0.0, 0.0, 0.5)), (0.0, 0.0, 5.0), atol=1e-12)
    assert res.piece_length_in_world((0.0, 0.0, 1.0)) == pytest.approx(0.1)
    plain = make_ctx(at=(0.5, 0.0, 0.0))
    res = resolve_target(plain, "reach", _target(plain))
    np.testing.assert_allclose(res.offset_in_piece((0.1, 0.2, 0.3)), (0.1, 0.2, 0.3))


def test_piece_frames_for_a_placement_and_a_carried_prop():
    placed = make_ctx(at=(0.5, 0.0, 0.0), quat=quat_y(90))
    clip, samples = placed.clips["reach"], placed.samples["reach"]
    frames = targets.piece_frames(clip, samples, "Plate", "test")
    assert frames.shape == (10, 4, 4)
    want = clip.floor_gear.pieces["Plate"].matrix()
    assert all(np.array_equal(f, want) for f in frames)
    scaled = make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=1.25 * turn_y(90))
    clip, samples = scaled.clips["reach"], scaled.samples["reach"]
    frames = targets.piece_frames(clip, samples, "prop:plate", "test")
    np.testing.assert_array_equal(frames, samples.props["plate"])
    assert targets.piece_frame(clip, samples, "prop:plate", 3, "test") == pytest.approx(
        samples.props["plate"][3])
    with pytest.raises(ManifestError, match="no pose for prop 'lamp'"):
        targets.piece_frames(clip, samples, "prop:lamp", "test")


def test_length_in_world_reads_the_scale_along_a_direction():
    stretch = np.diag([2.0, 1.0, 0.5]) @ turn_y(90)
    for d, want in (((1.0, 0.0, 0.0), 0.5), ((0.0, 3.0, 0.0), 1.0), ((0.0, 0.0, 1.0), 2.0)):
        assert targets.length_in_world(stretch, d) == pytest.approx(want)


def test_resolve_ref_is_resolve_target_without_the_offset():
    ctx = make_ctx(at=(0.5, 0.0, 0.0), quat=quat_y(90), offset=(0.3, 0.0, 0.5), key=4)
    target = _target(ctx)
    res = targets.resolve_ref(ctx, "reach", target.ref, target.key)
    same = resolve_target(ctx, "reach", target)
    assert res.sample == same.sample == 3
    np.testing.assert_array_equal(res.point, same.point)
    np.testing.assert_array_equal(res.frame, same.frame)


def test_seat_root_is_the_one_seat_rule():
    sit = SitSpec(below_seat=0.1, forward=0.05)
    np.testing.assert_allclose(targets.seat_root((1.0, 0.5, 2.0), (-1.0, 0.0, 0.0), sit),
                               (0.95, 0.4, 2.0))
    # The facing is flattened to the ground and made unit length.
    np.testing.assert_allclose(targets.seat_root((0.0, 0.46, 0.0), (0.0, 0.5, 2.0),
                                                 SitSpec(0.083, 0.03)), (0.0, 0.377, 0.03))
    with pytest.raises(ManifestError, match="straight up or down"):
        targets.seat_root((0.0, 0.0, 0.0), (1e-13, 1.0, 0.0), sit)
