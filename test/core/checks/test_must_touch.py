import dataclasses
import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import geom
from model_kit.core.checks import must_touch
from model_kit.core.checks.must_touch import MustTouch, hold_gap
from model_kit.core.errors import AnchorMissing, ManifestError
from model_kit.core.geom import (
    RAYS,
    SurfaceIndex,
    clip_to_box,
    drop_slivers,
    footprint_distance,
    is_closed,
    ray_counts,
)
from model_kit.core.manifest import (
    Clip,
    ClipHold,
    Dangle,
    FloorGear,
    Pack,
    Placement,
    Prop,
    Ref,
    load_pack,
)
from model_kit.core.metagate import run_gate
from model_kit.core.scene import (
    ClipSamples,
    PieceGeom,
    RigData,
    SceneCtx,
    effector_radius,
    piece_key,
)
from model_kit.core.tolerances import (
    BEHIND_MAX_OF_EFFECTOR,
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    TOUCH_OF_EFFECTOR,
)
from model_kit.core.units import Style

from ..shapes import blob, block, box, open_plate, plate
from .piece_scenes import tried_and_caught

RADIUS = 0.0586
BUDGET = TOUCH_OF_EFFECTOR * RADIUS  # about 0.005
BEHIND = BEHIND_MAX_OF_EFFECTOR * RADIUS
STYLE = Style(units_per_metre=1.0, page_camera={}, dials={})
TIMES = np.arange(1.0, 11.0)
PLATE_TOP = 0.002
BLOCK_TOP = 0.08
PATCH = [[-0.03, 0.07, -0.03], [0.03, 0.09, 0.03]]  # the middle of the block's top
UP = (0.0, 1.0, 0.0)
VALLEY_DIP = 0.01  # the V top's crease sits this far under its rims
VALLEY_PATCH = [[-0.09, 0.06, -0.09], [0.09, 0.09, 0.09]]  # both valley faces, no walls


def _hold(id_, surface, point):
    return {"id": id_, "use": "front", "side": "R", "surface": surface, "point": point,
            "off": [0.0, 1.0, 0.0], "grip": False}


def _plain_piece(parts, holds):
    return {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": 0.002}],
        "parts": [{"name": p, "listed": "stock", "shape": "box", "on": None, "group": "g",
                   "role": "solid"} for p in parts],
        "uses": [{"id": "front", "point": [0.0, 0.0, 0.3], "facing": [0.0, 0.0, -1.0],
                  "posture": "stand"}],
        "holds": holds, "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {},
    }


@cache
def kit_pack() -> Pack:
    doll = _plain_piece(["torso", "arm_R"], [])
    doll["uses"] = []
    doll["rig"] = {
        "armature": "TestRig",
        "bones": [
            {"name": "body", "parent": None, "length": 0.3},
            {"name": "arm.R", "parent": "body", "length": 0.1},
            {"name": "hand.R", "parent": "arm.R", "length": 0.05},
        ],
        "bind": {"torso": {"bones": ["body"], "mode": "rigid"},
                 "arm_R": {"bones": ["arm.R", "hand.R"], "mode": "chain"}},
        "contact_groups": {"hand.R": ["hand.R"]},
        "chains": {"arm.R": {
            "root": "body", "bones": ["arm.R", "hand.R"], "effector": "hand.R",
            "effector_radius": RADIUS, "dofs": {"arm.R": "xyz", "hand.R": "xyz"},
            "limits": {}, "effector_group": "hand.R",
        }},
        "sit": {"below_seat": 0.08, "forward": 0.03},
        "mirror": {"pairs": [], "axis": "x"},
    }
    data = {"pack": "kit", "units_per_metre": 1.0, "pieces": {
        "Plate": _plain_piece(["top"], [_hold("top_R", ["top"], [0.0, PLATE_TOP, 0.0])]),
        "Block": _plain_piece(["body"], [
            _hold("top_R", ["body"], [0.0, BLOCK_TOP, 0.0]),
            {**_hold("patch_R", ["body"], [0.0, BLOCK_TOP, 0.0]), "sub_box": PATCH},
        ]),
        "Sheet": _plain_piece(["skin"], [_hold("top_R", ["skin"], [0.0, PLATE_TOP, 0.0])]),
        "Stick": _plain_piece(["shaft", "tip"], []),
        "Valley": _plain_piece(["body"], [
            {**_hold("top_R", ["body"], [0.0, BLOCK_TOP - VALLEY_DIP, 0.0]),
             "sub_box": VALLEY_PATCH},
        ]),
        "Doll": doll,
    }}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


STICK = {  # in the stick's own frame: a long shaft and a raised tip at its far end
    "shaft": box((-0.01, 0.0, 0.0), (0.01, 0.01, 0.2)),
    "tip": box((-0.01, 0.05, 0.2), (0.01, 0.06, 0.22)),
}
TIP_BOX = ((-0.02, 0.04, 0.19), (0.02, 0.07, 0.23))

GEOM = {
    piece_key("kit", "Plate"): PieceGeom("Plate", {"top": plate()}),
    piece_key("kit", "Block"): PieceGeom("Block", {"body": block()}),
    piece_key("kit", "Sheet"): PieceGeom("Sheet", {"skin": open_plate()}),
    piece_key("kit", "Stick"): PieceGeom("Stick", STICK),
}

def _valley() -> np.ndarray:
    """A closed prism 0.2 wide and deep whose top is a V: rims at 0.08, crease at 0.07."""
    ring = [(-0.1, 0.0), (0.1, 0.0), (0.1, BLOCK_TOP), (0.0, BLOCK_TOP - VALLEY_DIP),
            (-0.1, BLOCK_TOP)]
    front = [(x, y, 0.1) for x, y in ring]
    back = [(x, y, -0.1) for x, y in ring]
    tris = []
    for k in (1, 2, 3):  # both caps, fanned from the bottom-left corner the notch can see
        tris.append((front[0], front[k], front[k + 1]))
        tris.append((back[0], back[k + 1], back[k]))
    for k in range(5):
        a, b = front[k], front[(k + 1) % 5]
        c, d = back[(k + 1) % 5], back[k]
        tris += [(a, d, c), (a, c, b)]
    return np.array(tris, dtype=float)


GEOM[piece_key("kit", "Valley")] = PieceGeom("Valley", {"body": _valley()})

RIG = RigData(order=("body",), parent={"body": None}, rest={"body": np.eye(4)}, world=np.eye(4),
              effector_mid={"hand.R": np.zeros(3)})


def make_ctx(point=(0.0, 0.004, 0.0), *, piece="Plate", ref=None, budget=None, dangle=None,
             when=((1.0, 10.0),), who="hand.R", who_box=None, prop_matrix=None,
             placement=None) -> SceneCtx:
    """One clip, `touch`, frames 1..10, one hold on `kit:<piece>.top_R`."""
    hold = ClipHold(
        ref=Ref.parse(ref or f"kit:{piece}.top_R"), who=who, who_box=who_box, when=when,
        budget=budget, dangle=None if dangle is None else Dangle(dangle, "the feet hang"),
        cross=None,
    )
    props = {}
    prop_samples = {}
    if prop_matrix is not None:
        props["stick"] = Prop("stick", "Stick", "kit", "hand.R", (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0, 1.0), ((1.0, 10.0),))
        prop_samples["stick"] = np.tile(np.asarray(prop_matrix, dtype=float), (len(TIMES), 1, 1))
    if placement is None:  # the clip places the piece at the origin, under its own name
        placement = Placement(piece, piece, "kit", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    gear = FloorGear({placement.id: placement})
    clip = Clip(name="touch", frames=(1, 10), fps=24.0, authoring="keyed", seat=None,
                holds=(hold,), targets=(), floor_gear=gear, props=props, mirror_of=None)
    pts = np.asarray(point, dtype=float).reshape(-1, 3)
    samples = ClipSamples(clip="touch", times=TIMES.copy(),
                          groups={"hand.R": np.tile(pts, (len(TIMES), 1, 1))},
                          bones={}, props=prop_samples)
    return SceneCtx(style=STYLE, packs={"kit": kit_pack()}, clips={"touch": clip},
                    pieces=dict(GEOM), samples={"touch": samples}, rigs={"kit:Doll": RIG},
                    rig_of_clip={"touch": "kit:Doll"})


def findings(ctx):
    return MustTouch().run(ctx)


def planted(ctx):
    """The candidate plants of every unit (a hold has two units, lift and slide, one each)."""
    return [p for unit in MustTouch().plants(ctx) for p in unit.candidates]


def plants_caught(ctx) -> bool:
    check = MustTouch()
    plants = planted(ctx)
    assert [p.label.split(":")[0] for p in plants] == ["lift", "slide"]
    return all(any(f.item == p.item for f in check.run(p.ctx, only=p.scope)) for p in plants)


def test_effector_radius_and_default_budget():
    ctx = make_ctx()
    assert effector_radius(ctx, "touch", "hand.R") == RADIUS
    assert BUDGET == pytest.approx(0.005, abs=2e-5)


def test_rest_passes():
    ctx = make_ctx((0.0, 0.004, 0.0))
    assert findings(ctx) == []
    assert run_gate([MustTouch()], ctx, None).status == "passed"


def test_press_passes():
    ctx = make_ctx((0.0, 0.001, 0.0))
    assert findings(ctx) == []
    assert run_gate([MustTouch()], ctx, None).status == "passed"


def test_lift_off_fails():
    found = findings(make_ctx((0.0, 0.012, 0.0)))
    assert len(found) == 1
    assert found[0].check == "must-touch"
    assert "hand.R" in found[0].item and "kit:Plate.top_R" in found[0].item


def test_slide_off_edge_fails():
    assert len(findings(make_ctx((0.13, 0.004, 0.0)))) == 1


def test_wrong_side_of_thin_plate_fails():
    assert len(findings(make_ctx((0.0, -0.02, 0.0)))) == 1


def test_plants_caught_for_rest_and_press():
    for point in ((0.0, 0.004, 0.0), (0.0, 0.001, 0.0)):
        ctx = make_ctx(point)
        assert findings(ctx) == []
        assert plants_caught(ctx)


def test_half_budget_lift_passes():
    assert findings(make_ctx((0.0, PLATE_TOP + CALIBRATION_PASS * BUDGET, 0.0))) == []


def test_double_budget_lift_fails():
    assert len(findings(make_ctx((0.0, PLATE_TOP + CALIBRATION_FAIL * BUDGET, 0.0)))) == 1


def test_deep_press_plant_not_blind():
    pts = np.array([[0.0, BLOCK_TOP - 0.025, 0.0]])
    ctx = make_ctx(pts[0], piece="Block")
    assert findings(ctx) == []
    used = {"body": block()}
    gap, depth = hold_gap(used, block(), pts, BEHIND, off=UP, budget=BUDGET)
    assert gap == 0.0 and depth == pytest.approx(0.025)
    # A fixed 0.03 lift would leave the point about one budget off: a coin toss.
    old_gap, _ = hold_gap(used, block(), pts + [0.0, 0.03, 0.0], BEHIND, off=UP,
                          budget=BUDGET)
    assert old_gap == pytest.approx(0.005)
    # The kit's lift clears the highest surface point by twice the budget.
    lift = next(p for p in planted(ctx) if p.label.startswith("lift"))
    moved = lift.ctx.samples["touch"].groups["hand.R"][0]
    assert moved[:, 1].min() >= BLOCK_TOP + 2 * BUDGET - 1e-12
    assert plants_caught(ctx)


def test_dangle_budget_plants_not_blind():
    ctx = make_ctx((0.0, 0.1, 0.0), dangle=0.2)
    assert findings(ctx) == []  # 0.098 off, inside the dangle budget
    assert plants_caught(ctx)
    assert len(findings(make_ctx((0.0, 0.3, 0.0), dangle=0.2))) == 1


def test_points_over_hold_point_slide_has_direction():
    ctx = make_ctx((0.0, 0.004, 0.0))  # straight over the hold point: no in-plane direction
    slide = next(p for p in planted(ctx) if p.label.startswith("slide"))
    moved = slide.ctx.samples["touch"].groups["hand.R"][0, 0]
    assert np.all(np.isfinite(moved))
    assert np.hypot(moved[0], moved[2]) >= 0.1 + RADIUS + 2 * BUDGET - 1e-12
    assert moved[1] == pytest.approx(0.004)
    assert any(f.item == slide.item for f in MustTouch().run(slide.ctx))


def test_open_part_stops():
    ctx = make_ctx(piece="Sheet")
    with pytest.raises(ManifestError, match="skin"):
        findings(ctx)
    with pytest.raises(ManifestError, match="skin"):
        run_gate([MustTouch()], ctx, None)


def test_when_outside_clip_stops():
    with pytest.raises(ManifestError, match="outside"):
        findings(make_ctx(when=((5.0, 12.0),)))
    with pytest.raises(ManifestError, match="outside"):
        run_gate([MustTouch()], make_ctx(when=((0.0, 4.0),)), None)


def test_empty_when_stops():
    with pytest.raises(ManifestError, match="when"):
        findings(make_ctx(when=()))
    with pytest.raises(ManifestError, match="when"):
        run_gate([MustTouch()], make_ctx(when=()), None)


def _stick_at(y):
    m = np.eye(4)
    m[:3, 3] = (0.0, y, -0.1)  # the shaft spans z -0.1..0.1, the whole plate
    return m


def test_held_prop_who_box():
    # The shaft lies 0.001 over the plate; the tip is 0.05 up and past the edge.
    m = _stick_at(PLATE_TOP + 0.001)
    whole = make_ctx(who="stick", prop_matrix=m)
    assert effector_radius(whole, "touch", "stick") == RADIUS
    assert findings(whole) == []  # every stick point counts: the shaft touches
    tip_only = make_ctx(who="stick", prop_matrix=m, who_box=TIP_BOX)
    assert len(findings(tip_only)) == 1  # only the tip counts, and it floats
    assert plants_caught(tip_only)
    # A tip that rests on the plate passes, though the shaft hangs in the air below the box.
    m2 = np.eye(4)
    m2[:3, 3] = (0.0, PLATE_TOP + 0.002 - 0.05, -0.21)  # tip 0.002 over the plate centre
    resting = make_ctx(who="stick", prop_matrix=m2, who_box=TIP_BOX)
    assert findings(resting) == []
    assert plants_caught(resting)


def test_placed_piece_maps_points_into_piece_frame():
    turn = Placement("Plate_A", "Plate", "kit", (1.0, 0.5, -2.0),
                     (0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)))
    rest = turn.to_world([[0.05, 0.004, 0.0]])[0]
    ctx = make_ctx(rest, ref="kit:Plate_A.top_R", placement=turn)
    assert findings(ctx) == []
    assert plants_caught(ctx)
    lifted = turn.to_world([[0.05, 0.012, 0.0]])[0]
    assert len(findings(make_ctx(lifted, ref="kit:Plate_A.top_R", placement=turn))) == 1


def test_hold_with_no_due_sample_is_listed_in_coverage():
    ctx = make_ctx()
    ctx = dataclasses.replace(ctx, samples={"touch": dataclasses.replace(
        ctx.samples["touch"], times=np.array([1.0, 2.0]),
        groups={"hand.R": ctx.samples["touch"].groups["hand.R"][:2]})})
    ctx = dataclasses.replace(ctx, clips={"touch": dataclasses.replace(
        ctx.clips["touch"],
        holds=(dataclasses.replace(ctx.clips["touch"].holds[0], when=((5.0, 6.0),)),))})
    assert findings(ctx) == []
    assert list(MustTouch().plants(ctx)) == []
    lines = MustTouch().coverage(ctx)
    assert len(lines) == 1 and "no sample" in lines[0] and "kit:Plate.top_R" in lines[0]


def test_clip_with_no_holds_is_listed():
    ctx = make_ctx()
    clip = dataclasses.replace(ctx.clips["touch"], holds=())
    bare = dataclasses.replace(ctx, clips={"touch": clip})
    assert findings(bare) == [] and list(MustTouch().plants(bare)) == []
    assert MustTouch().coverage(bare) == ["must-touch: not applicable (no holds): touch"]
    assert MustTouch().coverage(ctx) == []
    verdict = run_gate([MustTouch()], bare, None)
    assert verdict.status == "passed"  # a plain line: it does not make the run partial
    assert verdict.coverage == ["must-touch: not applicable (no holds): touch"]


def test_hold_on_a_carried_prop_uses_its_pose_per_sample():
    # The plate is carried as prop `tray`; its pose moves each sample, and the hand follows.
    ctx = make_ctx()
    clip = ctx.clips["touch"]
    tray = Prop("tray", "Plate", "kit", "hand.R", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                ((1.0, 10.0),))
    clip = dataclasses.replace(clip, floor_gear=FloorGear({}), props={"tray": tray})
    poses = np.tile(np.eye(4), (len(TIMES), 1, 1))
    poses[:, :3, 3] = np.stack([TIMES * 0.1, TIMES * 0.05, -TIMES * 0.02], axis=1)

    def with_hand(lift):
        hand = poses[:, None, :3, 3] + [0.0, 0.004 + lift, 0.0]
        samples = dataclasses.replace(ctx.samples["touch"], groups={"hand.R": hand},
                                      props={"tray": poses})
        return dataclasses.replace(ctx, clips={"touch": clip}, samples={"touch": samples})

    resting = with_hand(0.0)
    assert findings(resting) == []
    assert plants_caught(resting)
    assert len(findings(with_hand(0.01))) == 1


def _carried_plate(scale, lift):
    """The plate carried as prop `tray`, its pose scaled by `scale` (the armature's scale);
    the hand rests `lift` world units over its top face."""
    ctx = make_ctx()
    tray = Prop("tray", "Plate", "kit", "hand.R", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                ((1.0, 10.0),))
    clip = dataclasses.replace(ctx.clips["touch"], floor_gear=FloorGear({}), props={"tray": tray})
    pose = np.eye(4)
    pose[:3, :3] = scale * np.eye(3)
    pose[:3, 3] = (0.3, 0.9, -0.2)
    top = pose[:3, :3] @ [0.0, PLATE_TOP, 0.0] + pose[:3, 3]
    hand = np.tile(top + [0.0, lift, 0.0], (len(TIMES), 1, 1))
    samples = dataclasses.replace(ctx.samples["touch"], groups={"hand.R": hand},
                                  props={"tray": np.tile(pose, (len(TIMES), 1, 1))})
    return dataclasses.replace(ctx, clips={"touch": clip}, samples={"touch": samples})


def test_the_budget_is_in_world_units_on_a_scaled_carried_prop():
    # At armature scale 1.25 a hand 0.006 off reads 0.0048 in the prop's units, under the
    # 0.005 budget: measured in the world it must fail.
    assert BUDGET < 0.006 < 1.25 * BUDGET
    (found,) = findings(_carried_plate(1.25, 0.006))
    assert "gap 0.0060 over budget 0.0050" in found.detail
    # At scale 0.8 an honest 0.0045 rest reads 0.0056 in the prop's units: it must pass.
    assert 0.8 * BUDGET < 0.0045 < BUDGET
    assert findings(_carried_plate(0.8, 0.0045)) == []


@pytest.mark.parametrize("scale", [1.25, 0.8, 10.0, 0.1])
def test_scaled_carried_prop_calibrates_both_ways(scale):
    near = _carried_plate(scale, CALIBRATION_PASS * BUDGET)
    assert findings(near) == []
    assert plants_caught(near)
    assert len(findings(_carried_plate(scale, CALIBRATION_FAIL * BUDGET))) == 1


def test_sub_box_press_inside_the_patch_passes_and_plants_are_caught():
    ctx = make_ctx((0.01, BLOCK_TOP - 0.005, 0.01), piece="Block", ref="kit:Block.patch_R")
    assert findings(ctx) == []
    assert plants_caught(ctx)
    assert run_gate([MustTouch()], ctx, None).status == "passed"  # no BlindCheck


def test_sub_box_press_outside_the_patch_fails():
    # Pressed into the same part, 0.06 beside the patch: beside the face, not behind it.
    ctx = make_ctx((0.09, BLOCK_TOP - 0.005, 0.0), piece="Block", ref="kit:Block.patch_R")
    found = findings(ctx)
    assert len(found) == 1 and "patch_R" in found[0].item
    gap, depth = hold_gap({"body": block()}, clip_to_box(block(), PATCH),
                          [[0.09, BLOCK_TOP - 0.005, 0.0]], BEHIND, off=UP, budget=BUDGET)
    assert gap == pytest.approx(np.hypot(0.06, 0.005)) and depth == 0.0


def test_zero_area_face_never_makes_a_press():
    # A point 0.035 deep (past behind_max) beside a zero-area sliver: the sliver adds no
    # surface, so the point is measured to the real top face and does not press.
    sliver = np.array([[[0.0, 0.044, 0.0], [0.01, 0.044, 0.0], [0.02, 0.044, 0.0]]])
    pts = [[0.0, 0.045, 0.0]]
    gap, depth = hold_gap({"body": block()}, np.concatenate([block(), sliver]), pts, BEHIND,
                          off=UP, budget=BUDGET)
    assert gap == pytest.approx(0.035) and depth == 0.0


def _lifted_last(times):
    """Samples at `times`, resting except the last one, lifted well off the plate."""
    ctx = make_ctx(when=((1.0, 5.0),))
    pts = np.tile([[[0.0, 0.004, 0.0]]], (len(times), 1, 1))
    pts[-1, 0, 1] = 0.05
    samples = dataclasses.replace(ctx.samples["touch"], times=np.asarray(times),
                                  groups={"hand.R": pts})
    return dataclasses.replace(ctx, samples={"touch": samples})


def test_due_slack_is_one_millionth_of_a_frame():
    assert len(findings(_lifted_last([1.0, 3.0, 5.0 + 5e-7]))) == 1  # due: inside the slack
    assert findings(_lifted_last([1.0, 3.0, 5.0 + 2e-6])) == []  # past the slack: not due
    early = _lifted_last([1.0 - 5e-7, 3.0, 4.0])
    pts = early.samples["touch"].groups["hand.R"][::-1].copy()  # lift the first sample instead
    early = dataclasses.replace(early, samples={"touch": dataclasses.replace(
        early.samples["touch"], groups={"hand.R": pts})})
    assert len(findings(early)) == 1


def test_unsampled_clip_still_resolves_its_holds():
    ctx = dataclasses.replace(make_ctx(ref="kit:Plate.nowhere"), samples={})
    with pytest.raises(AnchorMissing, match="nowhere"):
        MustTouch().coverage(ctx)
    with pytest.raises(AnchorMissing, match="nowhere"):
        run_gate([MustTouch()], ctx, None)
    good = dataclasses.replace(make_ctx(), samples={})
    assert MustTouch().coverage(good) == [(
        "must-touch: skipped touch holds[0]: hand.R on kit:Plate.top_R: "
        "the scene has no samples for the clip")]


def test_press_a_hair_past_the_patch_edge_passes():
    # 1 um outside the patch's edge, 5 mm into the block: an honest press, not a gap.
    ctx = make_ctx((0.03 + 1e-6, BLOCK_TOP - 0.005, 0.0), piece="Block", ref="kit:Block.patch_R")
    assert findings(ctx) == []
    assert plants_caught(ctx)
    assert run_gate([MustTouch()], ctx, None).status == "passed"


def test_press_under_a_concave_crease_passes():
    assert is_closed(_valley())
    crease = BLOCK_TOP - VALLEY_DIP
    for deep in (0.003, 0.01, 0.02):
        ctx = make_ctx((0.0, crease - deep, 0.0), piece="Valley")
        assert findings(ctx) == [], deep
        assert plants_caught(ctx)
        assert run_gate([MustTouch()], ctx, None).status == "passed"
    gap, depth = hold_gap({"body": _valley()}, clip_to_box(_valley(), VALLEY_PATCH),
                          [[0.0, crease - 0.01, 0.0]], BEHIND, off=UP, budget=BUDGET)
    assert gap == 0.0 and depth == pytest.approx(0.01)
    # Lifted off the crease by twice the budget, it fails.
    lifted = make_ctx((0.0, crease + CALIBRATION_FAIL * BUDGET, 0.0), piece="Valley")
    assert len(findings(lifted)) == 1


def test_a_plant_measures_only_the_hold_it_planted(monkeypatch):
    ctx = make_ctx()
    clip = ctx.clips["touch"]
    ctx = dataclasses.replace(ctx, clips={"touch": dataclasses.replace(clip,
                                                                        holds=clip.holds * 3)})
    resolved = []
    real_resolve = must_touch._resolve

    def spy(ctx, clip, index, hold):
        resolved.append(index)
        return real_resolve(ctx, clip, index, hold)

    monkeypatch.setattr(must_touch, "_resolve", spy)
    plants = planted(ctx)
    assert len(plants) == 6
    for plant in plants:
        assert plant.scope == frozenset({plant.item})
        resolved.clear()
        found = MustTouch().run(plant.ctx, only=plant.scope)
        assert [f.item for f in found] == [plant.item]
        assert len(resolved) == 1  # the planted hold only, not all three
    resolved.clear()
    assert MustTouch().run(ctx) == [] and resolved == [0, 1, 2]  # the real run: every hold


def test_a_hold_with_no_samples_or_no_due_sample_makes_the_run_partial():
    ctx = make_ctx()
    assert run_gate([MustTouch()], dataclasses.replace(ctx, samples={}), None).status == \
        "partial"
    clip = ctx.clips["touch"]
    late = dataclasses.replace(clip, holds=(dataclasses.replace(clip.holds[0],
                                                                when=((5.0, 6.0),)),))
    early = dataclasses.replace(ctx.samples["touch"], times=np.array([1.0, 2.0]),
                                groups={"hand.R": ctx.samples["touch"].groups["hand.R"][:2]})
    unseen = dataclasses.replace(ctx, clips={"touch": late}, samples={"touch": early})
    assert run_gate([MustTouch()], unseen, None).status == "partial"


def test_plant_measures_one_sample(monkeypatch):
    ctx = make_ctx()
    calls = []
    real = must_touch.hold_gap

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(must_touch, "hold_gap", spy)
    check = MustTouch()
    plants = planted(ctx)
    assert len(plants) == 2
    assert len(next(job for job in must_touch._jobs(ctx)).due) == len(TIMES)  # many samples due
    for plant in plants:
        assert len(plant.samples) == 1  # a plant names its one sample, apart from its item
        calls.clear()
        found = check.run(plant.ctx, only=plant.scope, samples=plant.samples)
        assert [f.item for f in found] == [plant.item]
        assert len(calls) == 1
    calls.clear()
    check.run(ctx)
    assert len(calls) == len(TIMES)  # no samples named: every due sample


def test_a_gate_run_measures_each_plant_once_and_the_real_run_every_sample(monkeypatch):
    calls = []
    real = must_touch.hold_gap
    monkeypatch.setattr(must_touch, "hold_gap",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    assert run_gate([MustTouch()], make_ctx(), None).status == "passed"
    assert len(calls) == 2 + len(TIMES)  # lift, slide, then the real run


def _dense_gap(used, surface, pts, behind_max, off, budget):
    """The measure, spelled out on the dense routines."""
    surface = drop_slivers(surface)
    n = np.asarray(off, dtype=float) / np.linalg.norm(off)
    near = geom._closest_dense(surface, pts)
    in_part = np.zeros(len(pts), dtype=bool)
    for tris in used.values():
        in_part |= (ray_counts(tris, pts, RAYS) % 2 == 1).sum(axis=1) >= 2
    pressed = (in_part & (near.dist <= behind_max)
               & (footprint_distance(surface, n, pts) <= budget))
    if pressed.any():
        below = np.maximum((near.point - pts) @ n, 0.0)
        return 0.0, float(below[pressed].max())
    return float(near.dist.min()), 0.0


def test_a_surface_of_thousands_of_triangles_gives_the_unpruned_gaps_with_pruned_work():
    ball = blob(50, 62) * 0.1  # 6076 triangles, closed
    assert len(ball) > 6000 and is_closed(ball)
    rng = np.random.default_rng(8)
    sets = {
        # a shell around the surface, deep enough to press and far enough to hover
        "pressing": ball[rng.integers(0, len(ball), 300)].mean(axis=1)
        * rng.uniform(0.9, 1.05, size=(300, 1)),
        "above": rng.uniform(-0.12, 0.12, size=(64, 3)) + [0.0, 0.5, 0.0],
        "hovering": ball[rng.integers(0, len(ball), 64)].mean(axis=1) * 1.3,
    }
    for label, pts in sets.items():
        want = _dense_gap({"body": ball}, ball, pts, BEHIND, UP, BUDGET)
        surface, part = SurfaceIndex(ball), SurfaceIndex(ball)
        got = hold_gap({"body": part}, surface, pts, BEHIND, off=UP, budget=BUDGET)
        assert got == pytest.approx(want, abs=1e-12), label
        assert (want[0] == 0.0) == (label == "pressing"), label  # both outcomes are covered
        assert surface.last_pairs < 0.10 * len(pts) * len(ball), label
        # Raw arrays give the same answer through a one-off index.
        assert hold_gap({"body": ball}, ball, pts, BEHIND, off=UP,
                        budget=BUDGET) == pytest.approx(want, abs=1e-12), label


def test_inside_and_footprint_run_only_on_points_near_the_surface(monkeypatch):
    ball = blob() * 0.1
    near = ball[:2].mean(axis=1) * 0.97  # just under the surface
    far = np.random.default_rng(3).uniform(1.0, 2.0, size=(40, 3))
    pts = np.concatenate([far[:20], near, far[20:]])
    seen = {"inside": [], "footprint": []}
    real_inside, real_footprint = SurfaceIndex.inside, must_touch.footprint_distance

    def spy_inside(self, p):
        seen["inside"].append(len(p))
        return real_inside(self, p)

    def spy_footprint(tris, n, p):
        seen["footprint"].append(len(p))
        return real_footprint(tris, n, p)

    monkeypatch.setattr(SurfaceIndex, "inside", spy_inside)
    monkeypatch.setattr(must_touch, "footprint_distance", spy_footprint)
    hold_gap({"body": ball}, ball, pts, BEHIND, off=UP, budget=BUDGET)
    assert seen["inside"] == [2] and seen["footprint"] == [2]


def test_the_plants_never_call_the_measure(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a plant called the measure")

    monkeypatch.setattr(must_touch, "hold_gap", refuse)
    assert [u.item for u in MustTouch().plants(make_ctx())] == [
        "touch holds[0]: hand.R on kit:Plate.top_R"] * 2


def test_the_plants_are_caught_on_two_pieces():
    valley = make_ctx((0.0, BLOCK_TOP - VALLEY_DIP - 0.01, 0.0), piece="Valley")  # pressed in
    for ctx in (make_ctx(), valley):
        assert findings(ctx) == []
        assert len(tried_and_caught(MustTouch(), ctx)) == 2


def test_a_hold_that_fails_already_is_skipped_by_the_gate_not_filtered_out():
    ctx = make_ctx((0.0, 0.3, 0.0))
    assert len(findings(ctx)) == 1
    assert len(MustTouch().plants(ctx)) == 2
    assert run_gate([MustTouch()], ctx, None).status == "failed"
