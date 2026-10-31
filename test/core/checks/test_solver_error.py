"""Check solver-error: at each key of a solved clip, the effector (or the tool tip) lies within
`SOLVE_ERR_OF_EFFECTOR` times the effector radius of its target, or of a dangle's budget.

The scene is the reach scene (`core/scenes.py`): one solved clip, one target on the plate's
`top_R` hold, a doll whose `hand.R` carries the effector 0.2 beyond its head. The hand sits at
x 0.6, so the effector is at x 0.8; the plate is put where the test wants the miss.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core import solve as solve_module
from model_kit.core.checks.solver_error import SolverError
from model_kit.core.manifest import Ref
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.rig import pose_world
from model_kit.core.scene import PieceGeom
from model_kit.core.solve import ChainArgs, solve_chain
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS, SOLVE_ERR_OF_EFFECTOR

from ..scenes import RADIUS, RIG, TIMES, TIP, make_ctx, rest
from .piece_scenes import chosen, tried_and_caught

LIMIT = SOLVE_ERR_OF_EFFECTOR * RADIUS  # 0.036
EFFECTOR_X = 0.8  # hand.R's head 0.6 plus effector_mid 0.2
ITEM = "reach targets[0]: arm.R -> kit:Plate.top_R"


def hand(x=0.6, y=0.0, z=0.0):
    """The sampled hand.R bone at (x, y, z), the same at every sample."""
    return {"hand.R": np.tile(rest(x, y, z), (len(TIMES), 1, 1))}


def scene(miss=0.0, *, direction=(0.0, 0.0, 1.0), **kw):
    """The effector sits `miss` from the target along `direction` (a unit vector)."""
    target = np.array([EFFECTOR_X, 0.0, 0.0]) - miss * np.asarray(direction, dtype=float)
    return make_ctx(at=tuple(target - (0.0, 0.002, 0.0)), bones=hand(), **kw)


def findings(ctx, only=None):
    return SolverError().run(ctx, only=only)


def plants_caught(check, ctx) -> list:
    return chosen(check, ctx)


# --- registration -------------------------------------------------------------------------------


def test_the_check_is_registered_and_is_not_a_piece_check():
    assert gate.CHECKS["solver-error"] is SolverError
    assert SolverError.name == "solver-error"
    assert SolverError.kinds == frozenset()


# --- the measure ----------------------------------------------------------------------------------


def test_an_effector_on_its_target_passes_and_its_plant_is_caught():
    ctx = scene()
    assert findings(ctx) == []
    assert len(plants_caught(SolverError(), ctx)) == 1
    verdict = run_gate([SolverError()], ctx, None)
    assert verdict.status == "passed" and verdict.coverage == []


@pytest.mark.parametrize("direction", [(0, 0, 1), (1, 0, 0), (0, -1, 0), (0.6, 0.0, 0.8)])
def test_the_miss_is_a_distance_in_any_direction(direction):
    assert findings(scene(CALIBRATION_PASS * LIMIT, direction=direction)) == []
    found = findings(scene(CALIBRATION_FAIL * LIMIT, direction=direction))
    assert [f.item for f in found] == [ITEM]
    assert found[0].check == "solver-error"


def test_the_limit_is_the_solve_error_times_the_effector_radius():
    assert LIMIT == pytest.approx(0.036)
    assert findings(scene(LIMIT * 0.99)) == []
    assert len(findings(scene(LIMIT * 1.01))) == 1


def test_the_finding_names_the_miss_and_the_limit():
    (found,) = findings(scene(0.1))
    assert "0.1000" in found.detail and f"{LIMIT:.4f}" in found.detail


def test_only_the_sample_at_the_key_counts():
    ctx = scene()
    moved = np.array(ctx.samples["reach"].bones["hand.R"])
    moved[0, :3, 3] += (0.0, 0.0, 0.5)  # frame 1; the key is frame 5
    samples = dataclasses.replace(ctx.samples["reach"], bones={
        **ctx.samples["reach"].bones, "hand.R": moved})
    assert findings(dataclasses.replace(ctx, samples={"reach": samples})) == []
    moved[4, :3, 3] += (0.0, 0.0, 0.5)  # frame 5
    assert len(findings(dataclasses.replace(ctx, samples={"reach": samples}))) == 1


def test_the_effector_follows_the_bone_matrix_turn():
    # hand.R turned a quarter about z carries effector_mid (0.2, 0, 0) to (0, 0.2, 0) from its head
    ctx = make_ctx(at=(0.6, 0.2 - 0.002, 0.0),
                   bones={"hand.R": np.tile(rest(0.6, 0.0, 0.0, [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
                                            (len(TIMES), 1, 1))})
    assert findings(ctx) == []


def test_a_leg_target_is_measured_on_the_foot():
    foot = {"foot.R": np.tile(rest(0.0, -0.8, 0.0), (len(TIMES), 1, 1))}  # mid (0, 0, 0.1)
    ok = make_ctx(chain="leg.R", at=(0.0, -0.802, 0.1), bones=foot)
    assert findings(ok) == []
    off = make_ctx(chain="leg.R", at=(0.0, -0.802, 0.3), bones=foot)
    (found,) = findings(off)
    assert "leg.R" in found.item


def test_a_rig_bone_ref_is_a_target_too():
    near = {**hand(), "chest": np.tile(rest(EFFECTOR_X, 0.0, 0.0), (len(TIMES), 1, 1))}
    assert findings(make_ctx(ref="rig:chest", bones=near)) == []
    far = {**hand(), "chest": np.tile(rest(EFFECTOR_X, 0.5, 0.0), (len(TIMES), 1, 1))}
    assert len(findings(make_ctx(ref="rig:chest", bones=far))) == 1


def test_the_offset_moves_the_target():
    ctx = make_ctx(at=(EFFECTOR_X, -0.002, 0.0), bones=hand(), offset=(0.0, 0.0, 0.0))
    assert findings(ctx) == []
    assert len(findings(make_ctx(at=(EFFECTOR_X, -0.002, 0.0), bones=hand(),
                                 offset=(0.0, 0.0, 0.2)))) == 1


# --- dangle ---------------------------------------------------------------------------------------


def test_a_dangle_replaces_the_tolerance_with_its_budget():
    miss = 3 * LIMIT
    assert len(findings(scene(miss))) == 1
    assert findings(scene(miss, dangle=0.3)) == []
    (found,) = findings(scene(0.4, dangle=0.3))
    assert "dangle" in found.detail and "0.3000" in found.detail


def test_a_dangle_budget_may_be_tighter_than_the_solve_tolerance():
    assert len(findings(scene(LIMIT / 2, dangle=LIMIT / 4))) == 1


def test_a_dangle_of_another_effector_does_not_replace_it():
    assert len(findings(scene(3 * LIMIT, dangle=0.3, dangle_who="foot.R"))) == 1


def test_a_dangle_outside_the_key_does_not_replace_it():
    assert len(findings(scene(3 * LIMIT, dangle=0.3, dangle_when=((1.0, 4.0),)))) == 1
    assert findings(scene(3 * LIMIT, dangle=0.3, dangle_when=((5.0, 9.0),))) == []


# --- a tool -----------------------------------------------------------------------------------------


def stick_geometry():
    """A drumstick: tip_length TIP along +Z, and a butt end that reaches further the other way
    (0.3 along -Z), as a stick held near its tip does."""
    return PieceGeom("Stick", {"top": np.array([[[0.0, 0.0, 0.0], [0.01, 0.0, 0.05],
                                                 [0.0, 0.0, TIP]],
                                                [[0.0, 0.0, 0.0], [0.01, 0.0, -0.1],
                                                 [0.0, 0.0, -0.3]]])})


def with_stick(ctx):
    return dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Stick": stick_geometry()})


def test_a_tool_is_measured_at_its_tip_not_at_the_hand():
    # The hand's effector is at x 0.8, far from the target; the stick's tip (0, 0, TIP) is on it.
    ctx = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True))
    assert findings(ctx) == []
    assert len(plants_caught(SolverError(), ctx)) == 1
    miss = with_stick(make_ctx(at=(0.0, -0.002, TIP + 2 * LIMIT), bones=hand(), tool=True))
    assert len(findings(miss)) == 1


@pytest.mark.parametrize("scale", [2.0, 0.5])
def test_the_tip_follows_the_props_scaled_pose(scale):
    # tip_length is in the stick's own units; the stick's pose carries the armature's scale.
    ctx = with_stick(make_ctx(at=(0.0, -0.002, TIP * scale), bones=hand(), tool=True,
                              tool_scale=scale))
    assert findings(ctx) == []
    unscaled = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True,
                                   tool_scale=scale))
    assert len(findings(unscaled)) == 1


def test_the_tip_is_along_the_pieces_plus_z_not_toward_its_furthest_vertex():
    # The drumstick's tip_length (TIP) runs along +Z; its butt end reaches 0.3 along -Z. The tip is
    # at (0, 0, TIP), not at the butt end.
    ctx = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True))
    assert findings(ctx) == []
    butt = with_stick(make_ctx(at=(0.0, -0.002, -TIP), bones=hand(), tool=True))
    assert len(findings(butt)) == 1


def test_a_tool_needs_no_geometry_to_be_measured():
    ctx = make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True)
    assert findings(ctx) == [] and SolverError().coverage(ctx) == []
    assert len(findings(make_ctx(at=(0.0, -0.002, TIP + 2 * LIMIT), bones=hand(), tool=True))) == 1


def test_a_tool_with_no_tip_length_is_unmeasured():
    ctx = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True))
    stick = dataclasses.replace(ctx.packs["kit"].pieces["Stick"], measured={})
    pack = dataclasses.replace(ctx.packs["kit"], pieces={**ctx.packs["kit"].pieces, "Stick": stick})
    lines = SolverError().coverage(dataclasses.replace(ctx, packs={"kit": pack}))
    assert len(lines) == 1 and isinstance(lines[0], Unmeasured) and "tip_length" in lines[0]


def test_a_tool_with_no_sampled_pose_is_unmeasured():
    ctx = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True))
    samples = dataclasses.replace(ctx.samples["reach"], props={})
    lines = SolverError().coverage(dataclasses.replace(ctx, samples={"reach": samples}))
    assert len(lines) == 1 and isinstance(lines[0], Unmeasured) and "pose" in lines[0]


# --- plants -----------------------------------------------------------------------------------------


def test_the_plant_moves_the_effector_bone_at_the_keys_sample_by_the_failing_multiple():
    ctx = scene()
    (plant,) = chosen(SolverError(), ctx)
    assert plant.item == ITEM and plant.samples == frozenset({4})
    before = ctx.samples["reach"].bones["hand.R"]
    after = plant.ctx.samples["reach"].bones["hand.R"]
    moved = np.linalg.norm(after[:, :3, 3] - before[:, :3, 3], axis=1)
    assert moved[4] == pytest.approx(CALIBRATION_FAIL * LIMIT)
    assert np.all(np.delete(moved, 4) == 0.0)


def test_a_tool_plant_moves_the_prop_pose_with_the_hand():
    ctx = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True))
    (plant,) = chosen(SolverError(), ctx)
    before = ctx.samples["reach"].props["stick"][4, :3, 3]
    after = plant.ctx.samples["reach"].props["stick"][4, :3, 3]
    assert np.linalg.norm(after - before) == pytest.approx(CALIBRATION_FAIL * LIMIT)


def test_a_dangle_plant_exceeds_the_dangle_budget():
    ctx = scene(0.0, dangle=0.3)
    (plant,) = chosen(SolverError(), ctx)
    assert any(f.item == ITEM for f in SolverError().run(plant.ctx, only=plant.scope))
    assert findings(ctx) == []


def test_a_target_that_already_fails_is_skipped_by_the_gate_not_filtered_out():
    # `run` reports it, so the gate skips its unit; the unit is still built, blind to the miss
    ctx = scene(3 * LIMIT)
    check = SolverError()
    assert [u.item for u in check.plants(ctx)] == [ITEM]
    assert chosen(check, ctx) == []


def on_shelf(ctx):
    """The same reach with its target a hold on a second piece: `kit:Shelf`, a copy of the plate."""
    pack = ctx.packs["kit"]
    shelf = dataclasses.replace(pack.pieces["Plate"], name="Shelf")
    clip = ctx.clips["reach"]
    (plate,) = clip.floor_gear.pieces.values()
    placed = dataclasses.replace(plate, id="Shelf", piece="Shelf")
    target = dataclasses.replace(clip.targets[0], ref=Ref.parse("kit:Shelf.top_R"))
    moved = dataclasses.replace(clip, targets=(target,), floor_gear=dataclasses.replace(
        clip.floor_gear, pieces={"Shelf": placed}))
    return dataclasses.replace(
        ctx, packs={"kit": dataclasses.replace(pack, pieces={**pack.pieces, "Shelf": shelf})},
        pieces={**ctx.pieces, "kit:Shelf": ctx.pieces["kit:Plate"]},
        clips={"reach": moved})


def test_the_units_are_tried_and_caught_on_two_pieces():
    """Two different target pieces (the plate, the shelf) and so two different items; a tool
    scene besides, which measures at the tip."""
    plate = scene()
    shelf = on_shelf(scene())
    (shelf_item,) = [u.item for u in SolverError().plants(shelf)]
    assert shelf_item == "reach targets[0]: arm.R -> kit:Shelf.top_R" != ITEM
    assert tried_and_caught(SolverError(), plate) == [ITEM]
    assert tried_and_caught(SolverError(), shelf) == [shelf_item]
    tool = with_stick(make_ctx(at=(0.0, -0.002, TIP), bones=hand(), tool=True))
    assert tried_and_caught(SolverError(), tool) == [ITEM]


def test_a_scoped_run_measures_only_the_named_items():
    ctx = scene(3 * LIMIT)
    assert findings(ctx, only={"reach targets[1]: arm.R -> kit:Plate.top_R"}) == []
    assert [f.item for f in findings(ctx, only={ITEM})] == [ITEM]


def test_a_real_miss_fails_the_run():
    verdict = run_gate([SolverError()], scene(3 * LIMIT), None)
    assert verdict.status == "failed" and verdict.failed == 1
    assert verdict.lines[0].startswith(f"solver-error | {ITEM} | ")


# --- coverage ---------------------------------------------------------------------------------------


def test_a_keyed_clip_is_a_plain_not_applicable_line():
    ctx = scene(authoring="keyed")
    lines = SolverError().coverage(ctx)
    assert len(lines) == 1 and not isinstance(lines[0], Unmeasured)
    assert "not applicable" in lines[0] and "keyed" in lines[0] and "reach" in lines[0]
    assert findings(ctx) == [] and list(SolverError().plants(ctx)) == []
    assert run_gate([SolverError()], ctx, None).status == "passed"


def test_a_clip_with_no_samples_is_unmeasured_and_partial():
    ctx = make_ctx(samples=False)
    (line,) = SolverError().coverage(ctx)
    assert isinstance(line, Unmeasured) and "no samples" in line
    assert run_gate([SolverError()], ctx, None).status == "partial"


def test_a_clip_with_no_rig_is_unmeasured():
    ctx = dataclasses.replace(scene(), rigs={})
    (line,) = SolverError().coverage(ctx)
    assert isinstance(line, Unmeasured) and "no rig" in line
    assert findings(ctx) == []
    ctx = dataclasses.replace(scene(), rig_of_clip={})
    (line,) = SolverError().coverage(ctx)
    assert isinstance(line, Unmeasured) and "no rig" in line


def test_a_target_with_no_sampled_effector_bone_is_unmeasured():
    ctx = make_ctx(at=(EFFECTOR_X, -0.002, 0.0))  # the scene samples no hand.R
    (line,) = SolverError().coverage(ctx)
    assert isinstance(line, Unmeasured) and "hand.R" in line and ITEM in line
    assert run_gate([SolverError()], ctx, None).status == "partial"


def test_a_clean_scene_has_no_coverage_lines():
    assert SolverError().coverage(scene()) == []


# --- the known fault ---------------------------------------------------------------------------------


def one_iteration(monkeypatch):
    """The solver capped at one Nelder-Mead round per search, the 1-iteration solve."""
    real = solve_module.nelder
    monkeypatch.setattr(solve_module, "nelder", lambda f, x0, step, **kw: real(f, x0, step, iters=1))


def solved_ctx(target):
    """A scene whose hand.R and arm.R are posed by `solve_chain` toward `target`."""
    ctx = make_ctx(at=tuple(np.asarray(target) - (0.0, 0.002, 0.0)))
    chain = ctx.packs["kit"].pieces["Doll"].rig.chains["arm.R"]
    result = solve_chain(RIG, {}, chain, target, args=ChainArgs())
    world = pose_world(RIG, result.pose)
    bones = {b: np.tile(world[b], (len(TIMES), 1, 1)) for b in ("arm.R", "hand.R")}
    samples = dataclasses.replace(ctx.samples["reach"], bones={**ctx.samples["reach"].bones, **bones})
    return dataclasses.replace(ctx, samples={"reach": samples}), result


def test_one_iteration_solve_fails_the_measure(monkeypatch):
    # A target the arm must turn a long way for: the full solve lands on it, one round does not.
    target = np.array([0.3, 0.3, 0.4])  # 0.5 from the shoulder: the arm turned a quarter
    ctx, full = solved_ctx(target)
    assert full.err < LIMIT  # the stay term leaves about a centimetre; inside the tolerance
    assert findings(ctx) == []
    one_iteration(monkeypatch)
    ctx, capped = solved_ctx(target)
    assert capped.err > CALIBRATION_FAIL * LIMIT  # the capped solve really missed
    (found,) = findings(ctx)
    assert found.item == ITEM
    assert f"{capped.err:.4f}" in found.detail  # the measure reads the solver's own miss
    assert run_gate([SolverError()], ctx, None).status == "failed"
