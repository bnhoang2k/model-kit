import dataclasses

import numpy as np
import pytest

from model_kit.core.checks import reach
from model_kit.core.checks.reach import Reach
from model_kit.core.errors import AnchorMissing, ManifestError
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import CALIBRATION_FAIL

from ..scenes import (
    ARM_REACH,
    MARGIN,
    SHOULDER_X,
    TIMES,
    TIP,
    make_ctx,
    quat_y,
    rest,
    turn_y,
    turn_z,
)
from .piece_scenes import plants_caught as piece_plants_caught
from .piece_scenes import tried_and_caught

SHOULDER = np.array([SHOULDER_X, 0.0, 0.0])
END = SHOULDER_X + ARM_REACH  # the x where a straight arm ends, from the posed shoulder


def findings(ctx):
    return Reach().run(ctx)


def plants_caught(ctx) -> bool:
    return piece_plants_caught(Reach(), ctx)


def the_plant(ctx):
    """The one candidate of the one plant unit of a scene with one target."""
    (unit,) = Reach().plants(ctx)
    (plant,) = unit.candidates
    return plant


def test_reachable_passes():
    ctx = make_ctx(at=(0.5, 0.0, 0.0))
    assert findings(ctx) == []
    assert run_gate([Reach()], ctx, None).status == "passed"


def test_unreachable_fails():
    found = findings(make_ctx(at=(1.2, 0.0, 0.0)))
    assert len(found) == 1
    assert found[0].check == "reach"
    assert "arm.R" in found[0].item and "kit:Plate.top_R" in found[0].item


def test_the_margin_is_subtracted_from_reach():
    limit = END - MARGIN
    # the plate point is (x, 0, 0) when the plate sits at y = -0.002
    assert findings(make_ctx(at=(limit - MARGIN / 2, -0.002, 0.0))) == []
    # inside reach, but in the margin band: only a check that subtracts the margin fails it
    assert len(findings(make_ctx(at=(limit + MARGIN / 2, -0.002, 0.0)))) == 1
    assert len(findings(make_ctx(at=(END + MARGIN, -0.002, 0.0)))) == 1


def _short_arm(at):
    """The scene with an arm of reach 0.2127: hand.R's head 0.1127 past arm.R's, mid 0.1."""
    ctx = make_ctx(at=at)
    rig = ctx.rigs["kit:Doll"]
    short = dataclasses.replace(
        rig, rest={**rig.rest, "hand.R": rest(SHOULDER_X + 0.1127, 0, 0)},
        effector_mid={**rig.effector_mid, "hand.R": np.array([0.1, 0.0, 0.0])})
    return dataclasses.replace(ctx, rigs={"kit:Doll": short})


def test_reach_from_first_bone():
    # The shoulder (arm.R's head) is at x = 0.3. A target 0.2135 from it, on the side toward
    # the body, lies 0.0865 from the body: inside the reach measured from the body, outside
    # the reach of this 0.2127 arm measured from its shoulder.
    assert len(findings(_short_arm(at=(SHOULDER_X - 0.2135, -0.002, 0.0)))) == 1
    assert findings(_short_arm(at=(SHOULDER_X - 0.1, -0.002, 0.0))) == []


def test_reach_is_measured_from_the_posed_shoulder_not_the_body():
    # 0.25 behind the body is 0.55 from the shoulder: past the arm's 0.5, though near the body.
    found = findings(make_ctx(at=(-0.25, -0.002, 0.0)))
    assert len(found) == 1 and "first bone" in found[0].detail
    assert findings(make_ctx(at=(0.1, -0.002, 0.0))) == []
    # the shoulder follows the posed first bone: moved to x = 0.9 it brings x = 1.2 in reach
    assert findings(make_ctx(at=(1.2, -0.002, 0.0), root=(0.6, 0.0, 0.0))) == []
    assert len(findings(make_ctx(at=(1.2, -0.002, 0.0), root=(0.0, 0.0, 0.0)))) == 1


def test_unreachable_dangle_passes():
    ctx = make_ctx(at=(1.2, 0.0, 0.0), dangle=0.3)
    assert findings(ctx) == []
    assert any("dangle" in line for line in Reach().coverage(ctx))


def test_dangle_of_another_effector_does_not_excuse():
    ctx = make_ctx(at=(1.2, 0.0, 0.0), dangle=0.3, dangle_who="foot.R")
    assert len(findings(ctx)) == 1
    assert Reach().coverage(ctx) == []


def test_dangle_outside_the_key_does_not_excuse():
    assert findings(make_ctx(at=(1.2, 0.0, 0.0), dangle=0.3, dangle_when=((1.0, 4.0),))) != []
    assert findings(make_ctx(at=(1.2, 0.0, 0.0), dangle=0.3, dangle_when=((5.0, 9.0),))) == []
    keyed = ((1.0, 2.0), (5.0, 5.0))
    assert findings(make_ctx(at=(1.2, 0.0, 0.0), dangle=0.3, dangle_when=keyed)) == []


def test_tool_extends_reach():
    past = END + 0.1
    assert len(findings(make_ctx(at=(past, -0.002, 0.0)))) == 1
    assert findings(make_ctx(at=(past, -0.002, 0.0), tool=True)) == []
    assert len(findings(make_ctx(at=(END + TIP + 0.01, -0.002, 0.0), tool=True))) == 1


@pytest.mark.parametrize("scale", [1.25, 0.8])
def test_tool_tip_is_converted_to_world_units(scale):
    # tip_length is in the stick's own units; the stick's pose carries the armature's scale.
    tip = TIP * scale
    inside = make_ctx(at=(END + tip - MARGIN - 0.005, -0.002, 0.0), tool=True,
                      tool_scale=scale)
    assert findings(inside) == []
    assert plants_caught(inside)
    outside = make_ctx(at=(END + tip - MARGIN + 0.005, -0.002, 0.0), tool=True,
                       tool_scale=scale)
    assert len(findings(outside)) == 1


def test_scale_1_25_lets_a_far_tool_target_pass_and_0_8_fails_a_near_one():
    far = END + 0.16  # past 0.8 + 0.14 - margin, inside 0.8 + 0.175 - margin
    assert len(findings(make_ctx(at=(far, -0.002, 0.0), tool=True))) == 1
    assert findings(make_ctx(at=(far, -0.002, 0.0), tool=True, tool_scale=1.25)) == []
    near = END + 0.13  # inside 0.8 + 0.14 - margin, past 0.8 + 0.112 - margin
    assert findings(make_ctx(at=(near, -0.002, 0.0), tool=True)) == []
    assert len(findings(make_ctx(at=(near, -0.002, 0.0), tool=True, tool_scale=0.8))) == 1


def test_a_tool_with_no_sampled_pose_is_unmeasured():
    ctx = make_ctx(tool=True)
    samples = dataclasses.replace(ctx.samples["reach"], props={})
    ctx = dataclasses.replace(ctx, samples={"reach": samples})
    assert findings(ctx) == []
    (line,) = Reach().coverage(ctx)
    assert isinstance(line, Unmeasured) and "no sampled pose for tool stick" in line


def test_unknown_tool_raises_naming_it():
    ctx = make_ctx()
    clip = ctx.clips["reach"]
    bad = dataclasses.replace(clip, targets=(dataclasses.replace(clip.targets[0], tool="Hammer"),))
    with pytest.raises(ManifestError, match="Hammer"):
        findings(dataclasses.replace(ctx, clips={"reach": bad}))


def test_tool_without_tip_length_is_unmeasured():
    ctx = make_ctx(tool=True)
    stick = dataclasses.replace(ctx.packs["kit"].pieces["Stick"], measured={})
    pack = dataclasses.replace(ctx.packs["kit"], pieces={**ctx.packs["kit"].pieces,
                                                         "Stick": stick})
    ctx = dataclasses.replace(ctx, packs={"kit": pack})
    assert findings(ctx) == []
    (line,) = Reach().coverage(ctx)
    assert isinstance(line, Unmeasured) and "tip_length" in line and "skipped" in line


def test_leg_chain_checked():
    assert findings(make_ctx(chain="leg.R", at=(0.0, -0.85, 0.0))) == []
    found = findings(make_ctx(chain="leg.R", at=(0.0, -1.0, 0.0)))
    assert len(found) == 1 and "leg.R" in found[0].item


def test_rig_ref_target_checked():
    near = {"chest": np.tile(rest(0.4, 0.0, 0.0), (len(TIMES), 1, 1))}
    far = {"chest": np.tile(rest(1.4, 0.0, 0.0), (len(TIMES), 1, 1))}
    assert findings(make_ctx(ref="rig:chest", bones=near)) == []
    found = findings(make_ctx(ref="rig:chest", bones=far))
    assert len(found) == 1 and "rig:chest" in found[0].item


def test_offset_is_added_to_the_resolved_point():
    assert findings(make_ctx(at=(0.5, 0.0, 0.0), offset=(0.0, 0.0, 0.0))) == []
    assert len(findings(make_ctx(at=(0.5, 0.0, 0.0), offset=(0.6, 0.0, 0.0)))) == 1


def test_first_bone_comes_from_the_sample_at_the_key():
    assert findings(make_ctx(at=(1.2, 0.0, 0.0), root=(0.6, 0.0, 0.0))) == []


def test_first_bone_uses_the_nearest_sample_to_the_key():
    ctx = make_ctx(at=(1.2, 0.0, 0.0), key=5)
    shoulders = np.array(ctx.samples["reach"].bones["arm.R"])
    shoulders[4, 0, 3] = 0.9  # the sample at frame 5 only
    samples = dataclasses.replace(ctx.samples["reach"], bones={
        **ctx.samples["reach"].bones, "arm.R": shoulders})
    moved = dataclasses.replace(ctx, samples={"reach": samples})
    assert findings(moved) == []
    clip = moved.clips["reach"]
    later = dataclasses.replace(clip, targets=(dataclasses.replace(clip.targets[0], key=6),))
    assert len(findings(dataclasses.replace(moved, clips={"reach": later}))) == 1


def test_carried_prop_target_uses_the_sampled_pose():
    assert findings(make_ctx(carried=(0.5, 0.0, 0.0))) == []
    assert len(findings(make_ctx(carried=(1.3, 0.0, 0.0)))) == 1


def test_keyed_clip_listed_in_coverage():
    ctx = make_ctx(at=(5.0, 0.0, 0.0), authoring="keyed")
    assert findings(ctx) == []
    assert Reach().coverage(ctx) == ["reach: not applicable (keyed): reach"]
    assert list(Reach().plants(ctx)) == []


def test_clip_without_samples_listed_in_coverage():
    ctx = make_ctx(samples=False)
    assert findings(ctx) == []
    assert Reach().coverage(ctx) == ["reach: skipped (no samples): reach"]


def test_unresolved_ref_raises():
    with pytest.raises(AnchorMissing):
        findings(make_ctx(ref="kit:Plate.nowhere"))


def test_unknown_chain_raises():
    with pytest.raises(ManifestError, match="tail"):
        findings(make_ctx(chain="tail"))


def test_plant_caught():
    for ctx in (make_ctx(at=(0.5, 0.0, 0.0)), make_ctx(chain="leg.R", at=(0.0, -0.5, 0.0)),
                make_ctx(at=(0.9, -0.002, 0.0), tool=True), make_ctx(carried=(0.4, 0.0, 0.0))):
        assert findings(ctx) == []
        assert plants_caught(ctx)


def test_plant_lands_two_margins_past_reach_on_the_first_bone_line():
    ctx = make_ctx(at=(0.3, 0.0, 0.3), root=(0.1, 0.2, 0.0))
    plant = the_plant(ctx)
    moved = plant.ctx.clips["reach"].targets[0]
    target = np.array([0.3, 0.002, 0.3]) + moved.offset
    shoulder = np.array([0.1 + SHOULDER_X, 0.2, 0.0])  # the posed shoulder
    assert np.linalg.norm(target - shoulder) == pytest.approx(
        ARM_REACH + CALIBRATION_FAIL * MARGIN)
    start = np.array([0.3, 0.002, 0.3]) - shoulder
    cos = (target - shoulder) @ start / (
        np.linalg.norm(target - shoulder) * np.linalg.norm(start))
    assert cos == pytest.approx(1.0)


def test_a_failing_target_still_gets_a_unit_and_the_gate_skips_it():
    ctx = make_ctx(at=(1.2, 0.0, 0.0))
    (unit,) = Reach().plants(ctx)  # built from the geometry, not from the verdict
    assert unit.item == findings(ctx)[0].item
    assert run_gate([Reach()], ctx, None).status == "failed"


def test_dangle_target_gets_no_plant():
    assert list(Reach().plants(make_ctx(at=(0.5, 0.0, 0.0), dangle=0.3))) == []


# --- the offset is in the frame of what the ref names ---------------------------------------
# Hand-worked: a 90 degree turn about Y sends +x to (0, 0, -1) and +z to (1, 0, 0). A 90 degree
# turn about Z sends +x to +y. The hold point is (0, 0.002, 0), which neither turn moves.


def test_placement_turn_carries_the_offset_to_world():
    quat = quat_y(90)
    # piece-frame (0.3, 0, 0) lands at world (0.5, 0.002, -0.3): 0.36 from the shoulder, inside 0.5
    assert findings(make_ctx(at=(0.5, 0.0, 0.0), quat=quat, offset=(0.3, 0.0, 0.0))) == []
    # unturned, the same offset would land at x = 0.8 and fail; so a turn that is ignored shows
    assert len(findings(make_ctx(at=(0.5, 0.0, 0.0), offset=(0.5, 0.0, 0.0)))) == 1
    # piece-frame (0, 0, 0.35) lands at world (0.85, 0.002, 0.0): out of reach
    assert len(findings(make_ctx(at=(0.5, 0.0, 0.0), quat=quat, offset=(0.0, 0.0, 0.35)))) == 1
    assert findings(make_ctx(at=(0.5, 0.0, 0.0), offset=(0.0, 0.0, 0.35))) == []


def test_carried_prop_turn_carries_the_offset_to_world():
    spin = turn_y(90)
    assert findings(make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=spin,
                             offset=(0.3, 0.0, 0.0))) == []
    assert len(findings(make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=spin,
                                 offset=(0.0, 0.0, 0.35)))) == 1
    assert findings(make_ctx(carried=(0.5, 0.0, 0.0), offset=(0.0, 0.0, 0.35))) == []


def test_bone_turn_carries_the_offset_to_world():
    turned = {"chest": np.tile(rest(0.4, 0.0, 0.0, turn_z(90)), (len(TIMES), 1, 1))}
    plain = {"chest": np.tile(rest(0.4, 0.0, 0.0), (len(TIMES), 1, 1))}
    # bone-frame (0.45, 0, 0) lands at world (0.4, 0.45, 0): 0.46 from the shoulder
    assert findings(make_ctx(ref="rig:chest", bones=turned, offset=(0.45, 0.0, 0.0))) == []
    assert len(findings(make_ctx(ref="rig:chest", bones=plain, offset=(0.45, 0.0, 0.0)))) == 1
    # bone-frame (0, 0.5, 0) lands at world (-0.1, 0, 0) after the turn: 0.4 from the shoulder
    assert findings(make_ctx(ref="rig:chest", bones=turned, offset=(0.0, 0.5, 0.0))) == []


def test_plants_caught_under_a_turned_frame():
    quat = quat_y(90)
    for ctx in (make_ctx(at=(0.5, 0.0, 0.0), quat=quat, offset=(0.3, 0.0, 0.0)),
                make_ctx(carried=(0.5, 0.0, 0.0), carried_spin=turn_y(90),
                         offset=(0.3, 0.0, 0.0)),
                make_ctx(ref="rig:chest", offset=(0.45, 0.0, 0.0),
                         bones={"chest": np.tile(rest(0.4, 0.0, 0.0, turn_z(90)),
                                                 (len(TIMES), 1, 1))})):
        assert findings(ctx) == []
        assert plants_caught(ctx)


def test_plant_distance_is_exact_under_a_turn():
    ctx = make_ctx(at=(0.5, 0.0, 0.0), quat=quat_y(90), offset=(0.3, 0.0, 0.0))
    plant = the_plant(ctx)
    moved = np.array(plant.ctx.clips["reach"].targets[0].offset)
    world = np.array([0.5, 0.002, 0.0]) + turn_y(90) @ moved  # independent of the check's code
    assert np.linalg.norm(world - SHOULDER) == pytest.approx(ARM_REACH + CALIBRATION_FAIL * MARGIN)


def test_a_plant_measures_only_the_target_it_planted(monkeypatch):
    ctx = make_ctx(at=(0.5, 0.0, 0.0))
    clip = ctx.clips["reach"]
    ctx = dataclasses.replace(ctx, clips={"reach": dataclasses.replace(
        clip, targets=clip.targets * 3)})
    calls = []
    real = reach.resolve_target

    def spy(ctx, clip_name, target):
        calls.append(target)
        return real(ctx, clip_name, target)

    monkeypatch.setattr(reach, "resolve_target", spy)
    units = Reach().plants(ctx)
    assert len(units) == 3
    for unit in units:
        (plant,) = unit.candidates
        assert plant.scope == frozenset({plant.item})
        calls.clear()
        assert [f.item for f in Reach().run(plant.ctx, only=plant.scope)] == [plant.item]
        assert len(calls) == 1
    calls.clear()
    assert Reach().run(ctx) == [] and len(calls) == 3


def test_a_solved_clip_with_no_samples_makes_the_run_partial():
    assert run_gate([Reach()], make_ctx(samples=False), None).status == "partial"
    assert run_gate([Reach()], make_ctx(authoring="keyed"), None).status == "passed"
    assert run_gate([Reach()], make_ctx(at=(1.2, 0.0, 0.0), dangle=0.5), None).status == "passed"


def test_the_plant_distance_is_the_calibration_factor(monkeypatch):
    monkeypatch.setattr(reach, "CALIBRATION_FAIL", 3.0, raising=False)
    ctx = make_ctx(at=(0.5, 0.0, 0.0))
    plant = the_plant(ctx)
    target = plant.ctx.clips["reach"].targets[0]
    world = np.array([0.5, 0.002, 0.0]) + np.asarray(target.offset)
    assert np.linalg.norm(world - SHOULDER) == pytest.approx(ARM_REACH + 3.0 * MARGIN)


def test_the_plants_never_ask_whether_a_target_passes(monkeypatch):
    def refuse(self):
        raise AssertionError("a plant asked whether the target passes")

    monkeypatch.setattr(reach._Job, "distance", property(refuse))
    monkeypatch.setattr(reach._Job, "limit", property(refuse))
    assert [u.item for u in Reach().plants(make_ctx(at=(0.5, 0.0, 0.0)))]


def test_the_plants_are_caught_on_two_chains():
    arm = make_ctx(at=(0.5, 0.0, 0.0))
    leg = make_ctx(chain="leg.R", at=(0.0, -0.5, 0.0))
    for ctx in (arm, leg):
        assert len(tried_and_caught(Reach(), ctx)) == 1
