"""Check overlap, pose overlap: a body, a placed piece and a carried prop never sink into each other.

The scene is the pack `kit`. A `Doll` (rigged: a torso, an arm whose far half is the hand, and
an eye printed on the torso) is posed by the clip `play`. A `Block` stands in the clip (placed),
a `Rod` rides the doll's hand (carried) and a `Throne` is the seat. Everything starts far apart,
so the clean scene passes; each test brings one vertex, or the rod, into another surface by a
chosen depth. The depth budgets are the factors 0.002 (skin), 0.010 (press) and 0.025 (seat)
times the mitt radius.
"""

import dataclasses
import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.overlap import Overlap
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.geom import is_closed
from model_kit.core.manifest import Clip, ClipHold, FloorGear, Pack, Placement, Prop, Ref, load_pack
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.rig import RigData
from model_kit.core.scene import ClipSamples, PieceGeom, RigRest, SceneCtx
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    PRESS_OF_EFFECTOR,
    SEAT_OF_EFFECTOR,
    SKIN_OF_EFFECTOR,
    SOLVE_ERR_OF_EFFECTOR,
)
from model_kit.core.units import Style

from ..shapes import box
from .piece_scenes import TOUCH, chosen, tried_and_caught
from .sight_scenes import view_json

RADIUS = 0.0586
SKIN = SKIN_OF_EFFECTOR * RADIUS
PRESS = PRESS_OF_EFFECTOR * RADIUS
SEAT = SEAT_OF_EFFECTOR * RADIUS
ZONE = (1 + SOLVE_ERR_OF_EFFECTOR) * RADIUS
STYLE = Style(units_per_metre=1.0, page_camera={}, dials={})
TIMES = np.arange(1.0, 5.0)
BLOCK_AT = (2.0, 0.0, 0.0)
ROD_AT = (4.0, 0.0, 0.0)
THRONE_AT = (-2.0, 0.0, 0.0)
BLOCK_TOP = 0.08
CUSHION_TOP = 0.1
FRONT = 0.05  # the torso's front face

ITEM_A = "play A rig:arm_R/hand.R x kit:Block body"
ITEM_B = "play B kit:Rod shaft x rig:torso/body"
ITEM_C = "play C rig:arm_R/hand.R x rig:torso/body"
ITEM_D = "play D kit:Rod shaft x kit:Block body"
ITEM_SEAT = "play A rig:arm_R/hand.R x kit:Throne cushion"

# --- the pack -------------------------------------------------------------------------------


def _piece(parts, **fields):
    piece = {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": 0.002}],
        "parts": [{"name": p, "listed": "stock", "shape": "box", "on": None, "group": "g",
                   "role": "solid"} for p in parts],
        "uses": [{"id": "front", "point": [0.0, 0.0, 0.3], "facing": [0.0, 0.0, -1.0],
                  "posture": "stand"}],
        "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {}, "view": view_json(TOUCH),
    }
    return {**piece, **fields}


def _hold(id_, surface, point, grip):
    return {"id": id_, "use": "front", "side": "R", "surface": surface, "point": point,
            "off": [0.0, 1.0, 0.0], "grip": grip}


@cache
def kit_pack() -> Pack:
    doll = _piece(["torso", "arm_R", "eye_L"], uses=[])
    doll["parts"][2]["role"] = "eyes"
    doll["rig"] = {
        "armature": "TestRig",
        "bones": [
            {"name": "root", "parent": None, "length": 0.1},
            {"name": "body", "parent": "root", "length": 0.3},
            {"name": "arm.R", "parent": "body", "length": 0.1},
            {"name": "hand.R", "parent": "arm.R", "length": 0.05},
            {"name": "eyeball", "parent": "root", "length": 0.02},
        ],
        "bind": {"torso": {"bones": ["body"], "mode": "rigid"},
                 "arm_R": {"bones": ["arm.R", "hand.R"], "mode": "chain"},
                 "eye_L": {"bones": ["eyeball"], "mode": "rigid"}},
        "contact_groups": {"hand.R": ["hand.R"]},
        "chains": {"arm.R": {
            "root": "body", "bones": ["arm.R", "hand.R"], "effector": "hand.R",
            "effector_radius": RADIUS, "dofs": {"arm.R": "xyz", "hand.R": "xyz"},
            "limits": {}, "effector_group": "hand.R",
        }},
        "sit": {"below_seat": 0.08, "forward": 0.03},
        "mirror": {"pairs": [], "axis": "x"},
    }
    throne = _piece(["cushion"], uses=[{"id": "front", "point": [0.0, 0.0, 0.3],
                                        "facing": [0.0, 0.0, -1.0], "posture": "sit"}],
                    seats=[{"id": "sit", "use": "front", "point": [0.0, CUSHION_TOP, 0.0],
                            "facing": [0.0, 0.0, -1.0], "surface": ["cushion"]}])
    data = {"pack": "kit", "units_per_metre": 1.0, "pieces": {
        "Doll": doll,
        "Block": _piece(["body"], holds=[_hold("top_R", ["body"], [-0.1, BLOCK_TOP, 0.0], False),
                                         _hold("wrap_R", ["body"], [-0.1, BLOCK_TOP, 0.0], True)]),
        "Rod": _piece(["shaft"], holds=[_hold("grip_R", ["shaft"], [0.0, 0.0, 0.02], True)]),
        "Throne": throne,
    }}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


# --- the body --------------------------------------------------------------------------------


def indexed(tris):
    """A triangle soup as shared vertices (V, 3) and faces (T, 3)."""
    verts, inverse = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3).astype(np.int32)


def _body():
    """The rest shells: part -> (verts, faces, labels). The arm's far half is the hand."""
    torso_v, torso_f = indexed(box((-0.1, 0.0, -FRONT), (0.1, 0.3, FRONT)))
    arm_v, arm_f = indexed(box((0.12, 0.2, -0.03), (0.3, 0.26, 0.03)))
    eye_v, eye_f = indexed(box((-0.02, 0.1, 0.06), (0.02, 0.14, 0.1)))  # printed on the torso front
    return {
        "torso": (torso_v, torso_f, np.zeros(len(torso_v), np.int32)),
        "arm_R": (arm_v, arm_f, (arm_v[:, 0] > 0.21).astype(np.int32)),
        "eye_L": (eye_v, eye_f, np.zeros(len(eye_v), np.int32)),
    }


GROUPS = {"torso": ("body",), "arm_R": ("arm.R", "hand.R"), "eye_L": ("eyeball",)}
BODY = _body()
HAND, HAND2 = (int(v) for v in np.nonzero(BODY["arm_R"][2] == 1)[0][:2])  # two hand vertices
ARM = int(np.nonzero(BODY["arm_R"][2] == 0)[0][0])  # a vertex of the arm group
EYE = 0


def _rest() -> RigRest:
    return RigRest(
        shells={p: v[None].astype(np.float32) for p, (v, _, _) in BODY.items()},
        faces={p: f for p, (_, f, _) in BODY.items()},
        labels={p: lab for p, (_, _, lab) in BODY.items()},
        vertex_groups=dict(GROUPS))


RIG = RigData(
    order=("root", "body", "arm.R", "hand.R", "eyeball"),
    parent={"root": None, "body": "root", "arm.R": "body", "hand.R": "arm.R", "eyeball": "root"},
    rest={b: np.eye(4) for b in ("root", "body", "arm.R", "hand.R", "eyeball")}, world=np.eye(4),
    effector_mid={"hand.R": np.zeros(3)})

ROD = box((-0.01, -0.01, 0.0), (0.01, 0.01, 0.2))
BLOCK = box((-0.2, 0.0, -0.1), (0.2, BLOCK_TOP, 0.1))
CUSHION = box((-0.15, 0.0, -0.15), (0.15, CUSHION_TOP, 0.15))


def pose(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def make_ctx(*, holds=(), seat=False, rod=True, block=True, rod_pose=None,
             show=((1.0, 4.0),)) -> SceneCtx:
    """The clip `play`, frames 1 to 4, four samples, the doll at rest at every one."""
    placed = {}
    if block:
        placed["Block"] = Placement("Block", "Block", "kit", BLOCK_AT, (0.0, 0.0, 0.0, 1.0))
    if seat:
        placed["Throne"] = Placement("Throne", "Throne", "kit", THRONE_AT, (0.0, 0.0, 0.0, 1.0))
    props = {}
    prop_poses = {}
    if rod:
        props["rod"] = Prop("rod", "Rod", "kit", "hand.R", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                            show)
        prop_poses["rod"] = np.tile(pose(*ROD_AT) if rod_pose is None else rod_pose,
                                    (len(TIMES), 1, 1))
    clip = Clip(name="play", frames=(1, 4), fps=24.0, authoring="keyed",
                seat=Ref.parse("kit:Throne.sit") if seat else None, holds=tuple(holds),
                targets=(), floor_gear=FloorGear(placed), props=props, mirror_of=None)
    n = len(TIMES)
    samples = ClipSamples(
        clip="play", times=TIMES.copy(), groups={}, bones={}, props=prop_poses,
        shells={p: np.tile(v.astype(np.float32), (n, 1, 1)) for p, (v, _, _) in BODY.items()},
        faces={p: f for p, (_, f, _) in BODY.items()},
        labels={p: lab for p, (_, _, lab) in BODY.items()})
    pieces = {"kit:Block": PieceGeom("Block", {"body": BLOCK}),
              "kit:Rod": PieceGeom("Rod", {"shaft": ROD}),
              "kit:Throne": PieceGeom("Throne", {"cushion": CUSHION})}
    return SceneCtx(style=STYLE, packs={"kit": kit_pack()}, clips={"play": clip}, pieces=pieces,
                    samples={"play": samples}, rigs={"kit:Doll": RIG},
                    rig_of_clip={"play": "kit:Doll"}, rests={"kit:Doll": _rest()})


def put(ctx, part, vertex, s, point):
    """The scene with one body vertex of sample `s` moved to `point`."""
    samples = ctx.samples["play"]
    moved = np.array(samples.shells[part])
    moved[s, vertex] = point
    return dataclasses.replace(ctx, samples={"play": dataclasses.replace(
        samples, shells={**samples.shells, part: moved})})


def at_block(dx=0.0, depth=0.0):
    """A point `depth` under the block's top face, `dx` along x from the block's middle."""
    return (BLOCK_AT[0] + dx, BLOCK_TOP - depth, BLOCK_AT[2])


def with_clip(ctx, **fields):
    clip = dataclasses.replace(ctx.clips["play"], **fields)
    return dataclasses.replace(ctx, clips={"play": clip})


def hold(ref, who="hand.R", when=((1.0, 4.0),)):
    return ClipHold(ref=Ref.parse(ref), who=who, who_box=None, when=when, budget=None,
                    dangle=None, cross=None)


def findings(ctx, only=None, samples=frozenset()):
    return Overlap().run(ctx, only=only, samples=samples)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


# One scene per kind, a vertex or the rod `depth` into the other surface.


def scene_a(depth):
    return put(make_ctx(), "arm_R", HAND, 1, at_block(depth=depth))


def scene_b(depth):
    return make_ctx(rod_pose=pose(0.0, 0.15, FRONT - depth))


def scene_c(depth):
    return put(make_ctx(), "arm_R", HAND, 1, (0.0, 0.15, FRONT - depth))


def scene_d(depth):
    return make_ctx(rod_pose=pose(BLOCK_AT[0], BLOCK_TOP + 0.01 - depth, -0.05))


KINDS = [("A", scene_a, ITEM_A), ("B", scene_b, ITEM_B), ("C", scene_c, ITEM_C),
         ("D", scene_d, ITEM_D)]


def plants_caught(check, ctx) -> list[str]:
    """The kinds whose unit the gate would try, each with a candidate the check catches."""
    return [p.label.split(":")[0] for p in chosen(check, ctx)]


# --- registration -------------------------------------------------------------------------------


def test_the_check_is_registered_and_is_not_a_piece_check():
    assert gate.CHECKS["overlap"] is Overlap
    assert Overlap.name == "overlap"
    assert Overlap.kinds == frozenset()


# --- the four tests -----------------------------------------------------------------------------


def test_a_clean_scene_passes_every_kind_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(Overlap(), ctx) == ["A", "B", "C", "D"]
    assert Overlap().coverage(ctx) == []
    assert run_gate([Overlap()], ctx, None).status == "passed"


@pytest.mark.parametrize("kind, scene, item", KINDS)
def test_each_kind_passes_clean_and_fails_pushed_in_twice_the_skin(kind, scene, item):
    assert findings(scene(0.0)) == []
    (found,) = findings(scene(CALIBRATION_FAIL * SKIN))
    assert found.check == "overlap"
    assert found.item == item
    assert "depth" in found.detail and "budget" in found.detail


@pytest.mark.parametrize("kind, scene, item", KINDS)
def test_calibration_half_the_skin_passes_and_twice_the_skin_fails(kind, scene, item):
    assert findings(scene(CALIBRATION_PASS * SKIN)) == []
    assert items(scene(CALIBRATION_FAIL * SKIN)) == [item]


def test_a_finding_reports_the_worst_depth_over_the_clip_at_its_frame():
    ctx = put(scene_a(2 * SKIN), "arm_R", HAND, 2, at_block(depth=5 * SKIN))
    (found,) = findings(ctx)
    assert f"depth {5 * SKIN:.4f}" in found.detail
    assert f"budget {SKIN:.4f}" in found.detail
    assert "frame 3" in found.detail


def test_every_sample_of_the_clip_is_measured():
    for s in range(len(TIMES)):
        assert items(put(make_ctx(), "arm_R", HAND, s, at_block(depth=2 * SKIN))) == [ITEM_A]


def test_a_scoped_run_reports_only_the_items_it_names():
    ctx = put(scene_a(2 * SKIN), "arm_R", HAND2, 1, (0.0, 0.15, FRONT - 2 * SKIN))
    assert items(ctx) == [ITEM_A, ITEM_C]
    assert [f.item for f in findings(ctx, only={ITEM_C})] == [ITEM_C]
    assert [f.item for f in findings(ctx, only={ITEM_A})] == [ITEM_A]
    assert findings(ctx, only={"play A rig:torso/body x kit:Block body"}) == []


def test_a_scoped_run_reads_only_the_samples_it_names():
    ctx = put(make_ctx(), "arm_R", HAND, 2, at_block(depth=2 * SKIN))
    assert findings(ctx, samples=frozenset({0, 1})) == []
    assert items_of(findings(ctx, samples=frozenset({2}))) == [ITEM_A]


def items_of(found):
    return sorted(f.item for f in found)


def test_a_carried_prop_is_measured_only_while_it_is_shown():
    sunk = pose(0.0, 0.15, FRONT - 2 * SKIN)
    assert items(make_ctx(rod_pose=sunk, show=((1.0, 4.0),))) == [ITEM_B]
    assert items(make_ctx(rod_pose=sunk, show=((9.0, 12.0),))) == []


def test_a_show_window_reads_the_scenes_due_slack():
    # a window that starts a hair after the last sample still shows the prop there
    sunk = pose(0.0, 0.15, FRONT - 2 * SKIN)
    assert items(make_ctx(rod_pose=sunk, show=((4.0 + 5e-7, 6.0),))) == [ITEM_B]
    assert items(make_ctx(rod_pose=sunk, show=((4.0 + 5e-6, 6.0),))) == []


def test_a_hold_who_names_no_contact_group_says_so_not_that_no_chain_matched():
    ctx = make_ctx(holds=(hold("kit:Block.top_R", who="nobody"),))
    with pytest.raises(ManifestError, match="no contact group 'nobody'"):
        findings(ctx)


def test_a_prop_pose_with_a_scale_measures_depth_in_world_units():
    # the rod's pose scales it by 2: its end face sits one skin deep in the world, not in the rod
    depth = CALIBRATION_FAIL * SKIN
    scaled = pose(0.0, 0.15, FRONT - depth)
    scaled[:3, :3] *= 2.0
    (found,) = findings(make_ctx(rod_pose=scaled))
    assert found.item == ITEM_B
    assert f"depth {depth:.4f}" in found.detail


# --- zones --------------------------------------------------------------------------------------


def test_grip_zone_excuses_hand():
    # the hand sinks 0.02 into the block, ten skins, within the zone round a grip hold's point
    grip = (hold("kit:Block.wrap_R"),)
    deep = put(make_ctx(holds=grip), "arm_R", HAND, 1, at_block(dx=-0.1, depth=0.02))
    assert findings(deep) == []
    # past the zone (the hold point is 0.2 away) the same depth fails
    far = put(make_ctx(holds=grip), "arm_R", HAND, 1, at_block(dx=0.1, depth=0.02))
    assert items(far) == [ITEM_A]
    # and with no hold at all the same depth fails inside the zone's place
    assert items(put(make_ctx(), "arm_R", HAND, 1, at_block(dx=-0.1, depth=0.02))) == [ITEM_A]


def test_the_grip_zone_is_the_radius_plus_the_solve_error():
    grip = (hold("kit:Block.wrap_R"),)
    depth = 0.02

    def sunk(share):  # a vertex `depth` under the hold point's face, `share` of the zone away
        dx = float(np.sqrt((share * ZONE) ** 2 - depth**2))
        return put(make_ctx(holds=grip), "arm_R", HAND, 1, at_block(dx=-0.1 + dx, depth=depth))

    assert findings(sunk(0.98)) == []
    assert items(sunk(1.02)) == [ITEM_A]


def test_the_grip_zone_excuses_only_the_holding_contact_group():
    grip = (hold("kit:Block.wrap_R"),)
    # the arm vertex is not in contact group hand.R, so the grip does not excuse it
    arm = put(make_ctx(holds=grip), "arm_R", ARM, 1, at_block(dx=-0.1, depth=0.02))
    assert items(arm) == ["play A rig:arm_R/arm.R x kit:Block body"]


def test_a_grip_hold_excuses_a_piece_vertex_inside_the_holding_limb():
    # B: the gripped rod's end sits inside the arm_R mesh, which is one shell holding the arm
    # group and the hand group. The grip's contact group is the hand, so the shell counts.
    rod_pose = pose(0.21, 0.23, -0.02)
    ungripped = make_ctx(rod_pose=rod_pose)
    assert items(ungripped) == ["play B kit:Rod shaft x rig:arm_R/arm.R"]
    gripped = make_ctx(rod_pose=rod_pose, holds=(hold("kit:Rod.grip_R"),))
    assert findings(gripped) == []


def test_a_stray_holding_group_vertex_away_from_the_contact_excuses_nothing():
    # one torso vertex is weighted to hand.R. The gripped rod sunk 15 skins into the chest
    # touches no face of that vertex, so the grip excuses nothing there.
    sunk = pose(0.0, 0.15, FRONT - 15 * SKIN)
    grip = (hold("kit:Rod.grip_R"),)

    def with_stray(ctx):
        labels = BODY["torso"][2].copy()
        labels[0] = 1
        rest = ctx.rests["kit:Doll"]
        rest = dataclasses.replace(
            rest, labels={**rest.labels, "torso": labels},
            vertex_groups={**rest.vertex_groups, "torso": ("body", "hand.R")})
        samples = dataclasses.replace(ctx.samples["play"],
                                      labels={**ctx.samples["play"].labels, "torso": labels})
        return dataclasses.replace(ctx, rests={"kit:Doll": rest}, samples={"play": samples})

    plain = make_ctx(rod_pose=sunk, holds=grip)
    assert items(plain) == [ITEM_B]
    assert items(with_stray(plain)) == [ITEM_B]


def test_a_face_that_touches_the_holding_group_is_excused():
    # the wrist faces of the arm mesh have corners in both the arm and the hand group
    rod_pose = pose(0.21, 0.23, -0.02)
    assert findings(make_ctx(rod_pose=rod_pose, holds=(hold("kit:Rod.grip_R"),))) == []


def test_a_press_hold_allows_the_press_budget_not_more():
    press = (hold("kit:Block.top_R"),)
    ok = put(make_ctx(holds=press), "arm_R", HAND, 1, at_block(dx=-0.1, depth=0.6 * PRESS))
    assert findings(ok) == []
    too_deep = put(make_ctx(holds=press), "arm_R", HAND, 1,
                   at_block(dx=-0.1, depth=CALIBRATION_FAIL * PRESS))
    assert items(too_deep) == [ITEM_A]
    # without the hold the press depth is over the skin budget
    assert items(put(make_ctx(), "arm_R", HAND, 1, at_block(dx=-0.1, depth=0.6 * PRESS))) == [ITEM_A]


def test_a_hold_excuses_only_inside_its_when_windows():
    press = (hold("kit:Block.top_R", when=((1.0, 2.0),)),)
    inside = put(make_ctx(holds=press), "arm_R", HAND, 0, at_block(dx=-0.1, depth=0.6 * PRESS))
    outside = put(make_ctx(holds=press), "arm_R", HAND, 3, at_block(dx=-0.1, depth=0.6 * PRESS))
    assert findings(inside) == []
    assert items(outside) == [ITEM_A]


def test_a_prop_tip_on_a_head_gets_the_press_budget():
    # D: the rod holds the block (a stick on a head); its tip may sink the press depth
    tip = (hold("kit:Block.top_R", who="rod"),)
    ok = make_ctx(rod_pose=pose(BLOCK_AT[0] - 0.1, BLOCK_TOP + 0.01 - 0.6 * PRESS, -0.05),
                  holds=tip)
    assert findings(ok) == []
    deep = make_ctx(rod_pose=pose(BLOCK_AT[0] - 0.1, BLOCK_TOP + 0.01 - CALIBRATION_FAIL * PRESS,
                                  -0.05), holds=tip)
    assert items(deep) == [ITEM_D]


def test_the_seat_allows_the_seat_budget_on_its_surface_parts_at_the_seat_windows():
    throne = lambda d, dx=0.0: put(
        make_ctx(seat=True), "arm_R", HAND, 1, (THRONE_AT[0] + dx, CUSHION_TOP - d, 0.0))
    assert findings(throne(0.6 * SEAT)) == []
    assert items(throne(CALIBRATION_FAIL * SEAT)) == [ITEM_SEAT]
    # no seat in the clip: the same depth over the skin fails
    plain = put(make_ctx(), "arm_R", HAND, 1, at_block(depth=0.6 * SEAT))
    assert items(plain) == [ITEM_A]


def test_the_seat_zone_follows_seat_when():
    ctx = make_ctx(seat=True)
    sunk = put(ctx, "arm_R", HAND, 1, (THRONE_AT[0], CUSHION_TOP - 0.6 * SEAT, 0.0))
    assert findings(sunk) == []  # no seat_when: the whole clip sits
    windowed = with_clip(sunk, seat_when=((3.0, 4.0),))
    assert items(windowed) == [ITEM_SEAT]  # sample 1 is at frame 2, outside the window


# --- the body's own joints ----------------------------------------------------------------------


def test_rest_crossing_is_excused():
    # at rest the hand already sits in the torso: that pair is a joint of the rig
    rest = _rest()
    shells = {**rest.shells, "arm_R": rest.shells["arm_R"].copy()}
    shells["arm_R"][0, HAND] = (0.0, 0.15, FRONT - 0.02)
    crossed = dataclasses.replace(make_ctx(), rests={"kit:Doll": dataclasses.replace(
        rest, shells=shells)})
    deeper = put(crossed, "arm_R", HAND, 1, (0.0, 0.15, FRONT - 5 * SKIN))
    assert findings(deeper) == []
    # the same sample on a doll with no such crossing at rest fails
    assert items(put(make_ctx(), "arm_R", HAND, 1, (0.0, 0.15, FRONT - 5 * SKIN))) == [ITEM_C]


def test_a_bone_and_its_parent_cross_by_design():
    # arm.R's parent is body: the upper arm in the torso is the shoulder
    arm = put(make_ctx(), "arm_R", ARM, 1, (0.0, 0.15, FRONT - 5 * SKIN))
    assert findings(arm) == []


def test_eyes_left_out():
    # an eye vertex sunk in the torso or in the block is no finding: a printed part is no source
    in_torso = put(make_ctx(), "eye_L", EYE, 1, (0.0, 0.15, FRONT - 5 * SKIN))
    assert findings(in_torso) == []
    assert findings(put(make_ctx(), "eye_L", EYE, 1, at_block(depth=5 * SKIN))) == []
    # and an eye shell is no target: a hand vertex inside the eye box fails nothing
    assert findings(put(make_ctx(), "arm_R", HAND, 1, (0.0, 0.12, 0.08))) == []
    # the control: the same moves by a torso-group vertex do fail, so the eye is what is skipped
    assert items(put(make_ctx(), "arm_R", HAND, 1, (0.0, 0.15, FRONT - 5 * SKIN))) == [ITEM_C]


def test_the_kept_body_follows_the_piece_record_not_only_the_rest_file():
    ctx = scene_a(CALIBRATION_FAIL * SKIN)
    assert items(ctx) == [ITEM_A]  # builds and keeps the body for this rest file
    doll = ctx.packs["kit"].pieces["Doll"]
    printed = dataclasses.replace(doll, parts=tuple(
        dataclasses.replace(p, role="decal") if p.name == "arm_R" else p for p in doll.parts))
    pack = dataclasses.replace(ctx.packs["kit"], pieces={**ctx.packs["kit"].pieces,
                                                         "Doll": printed})
    assert items(dataclasses.replace(ctx, packs={"kit": pack})) == []  # the same rest file


def test_a_decal_part_of_a_placed_piece_is_left_out():
    ctx = scene_a(CALIBRATION_FAIL * SKIN)
    block = ctx.packs["kit"].pieces["Block"]
    printed = dataclasses.replace(block, parts=tuple(
        dataclasses.replace(p, role="decal") for p in block.parts))
    pack = dataclasses.replace(ctx.packs["kit"], pieces={**ctx.packs["kit"].pieces,
                                                         "Block": printed})
    assert findings(dataclasses.replace(ctx, packs={"kit": pack})) == []


# --- what cannot be measured --------------------------------------------------------------------


def test_a_rig_with_no_rest_file_is_unmeasured_never_a_pass():
    ctx = dataclasses.replace(make_ctx(), rests={})
    assert findings(ctx) == []
    lines = Overlap().coverage(ctx)
    assert lines == ["overlap: skipped (no body shells): kit:Doll"]
    assert isinstance(lines[0], Unmeasured)
    assert run_gate([Overlap()], ctx, None).status == "partial"


def test_a_clip_whose_samples_carry_no_shells_is_unmeasured():
    ctx = make_ctx()
    bare = dataclasses.replace(ctx.samples["play"], shells={}, faces={}, labels={})
    ctx = dataclasses.replace(ctx, samples={"play": bare})
    lines = Overlap().coverage(ctx)
    assert lines == ["overlap: skipped (no body shells): play"]
    assert isinstance(lines[0], Unmeasured)
    assert run_gate([Overlap()], ctx, None).status == "partial"


def test_a_clip_with_no_samples_is_unmeasured():
    ctx = dataclasses.replace(make_ctx(), samples={})
    lines = Overlap().coverage(ctx)
    assert lines == ["overlap: skipped (no samples): play"]
    assert isinstance(lines[0], Unmeasured)


def test_a_kind_with_nothing_to_measure_is_a_plain_line_not_an_unmeasured_one():
    ctx = make_ctx(rod=False, block=False)
    lines = Overlap().coverage(ctx)
    assert lines == [
        "overlap: not applicable (no placed or carried piece): play A",
        "overlap: not applicable (no placed or carried piece): play B",
        "overlap: not applicable (needs a carried prop and a placed piece): play D",
    ]
    assert not any(isinstance(line, Unmeasured) for line in lines)
    assert plants_caught(Overlap(), ctx) == ["C"]
    assert run_gate([Overlap()], ctx, None).status == "passed"


def test_no_carried_prop_leaves_only_d_not_applicable():
    lines = Overlap().coverage(make_ctx(rod=False))
    assert lines == ["overlap: not applicable (needs a carried prop and a placed piece): play D"]


def thick_block(ctx, thickness):
    """The scene with the placed block `thickness` thick (and as wide and deep as before)."""
    return dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Block": PieceGeom(
        "Block", {"body": box((-0.2, 0.0, -0.1), (0.2, thickness, 0.1))})})


def test_a_part_with_a_skin_and_a_half_of_room_is_planted_and_caught_not_not_applicable():
    # a block 3 skins thick has 1.5 skins of room. A body vertex there is a
    # real finding, so test A must plant it. A not-applicable unit would pass the gate without one
    ctx = thick_block(make_ctx(rod=False), 3 * SKIN)
    (unit_a,) = [u for u in Overlap().plants(ctx) if u.item.startswith("play A ")]
    assert not unit_a.not_applicable and unit_a.candidates
    assert "A" in {i.split(" ")[1] for i in tried_and_caught(Overlap(), ctx)}
    assert run_gate([Overlap()], ctx, None).status == "passed"


def test_a_part_too_thin_for_any_fault_stops_the_gate_not_a_not_applicable_line():
    # a block 0.2 skin thick: the part is judged and no vertex inside it can pass the skin. That is
    # a part the check cannot prove, not a part that does not apply, so the gate raises
    ctx = thick_block(make_ctx(rod=False), 0.2 * SKIN)
    (unit_a,) = [u for u in Overlap().plants(ctx) if u.item.startswith("play A ")]
    assert not unit_a.not_applicable and unit_a.candidates
    with pytest.raises(BlindCheck, match="play A "):
        run_gate([Overlap()], ctx, None)


def test_a_big_thin_part_does_not_stand_in_for_a_thicker_one():
    # the slab has the bigger box and is thinner than the skin; the cube has the room. The units
    # are ordered by room, not by box, so A is planted on the cube and caught, not stopped on the slab
    from model_kit.core.checks.overlap import Skipped, prepare_clip

    ctx = make_ctx(rod=False)
    slab = box((-0.5, 0.0, -0.5), (0.5, 0.2 * SKIN, 0.5))
    cube = box((0.6, 0.0, -0.02), (0.64, 0.04, 0.02))
    both = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Block": PieceGeom(
        "Block", {"slab": slab, "cube": cube})})
    prepared = prepare_clip(both, both.clips["play"])
    assert not isinstance(prepared, Skipped)
    (block,) = [i for i in prepared.insts if i.label == "kit:Block"]
    assert block.closed_parts()[0].name == "slab"  # the biggest box is the slab
    (unit_a,) = [u for u in Overlap().plants(both) if u.item.startswith("play A ")]
    assert unit_a.item.endswith("kit:Block cube")
    assert "A" in {i.split(" ")[1] for i in tried_and_caught(Overlap(), both)}
    assert run_gate([Overlap()], both, None).status == "passed"


def test_a_test_with_no_closed_part_to_judge_is_a_plain_not_applicable_unit():
    ctx = make_ctx(rod=False)
    opened = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Block": PieceGeom(
        "Block", {"body": BLOCK[:2]})})  # the top face only: no inside
    (unit_a,) = [u for u in Overlap().plants(opened) if u.item == "play A"]
    assert unit_a.not_applicable and not unit_a.candidates
    verdict = run_gate([Overlap()], opened, None)
    assert f"overlap: not applicable ({unit_a.not_applicable}): play A" in verdict.coverage


def test_an_open_container_part_is_unmeasured():
    ctx = make_ctx()
    open_block = BLOCK[:2]  # the top face only
    opened = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Block": PieceGeom(
        "Block", {"body": open_block})})
    lines = Overlap().coverage(opened)
    assert "overlap: skipped (open part, no inside): kit:Block body" in lines
    assert isinstance(next(x for x in lines if "open part" in x), Unmeasured)


def test_a_part_split_at_a_seam_is_closed_after_the_weld():
    # glTF splits a vertex at a UV seam; after the float32 round trip the two copies differ by
    # a hair. Topology welds them, so this part is closed here too and never unmeasured.
    ctx = make_ctx()
    seamed = np.array(BLOCK)
    seamed[np.all(seamed[:, :, 0] == 0.2, axis=1)] += 1e-7  # the +x face: its corners drift
    split = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Block": PieceGeom(
        "Block", {"body": seamed})})
    assert not is_closed(seamed)
    assert Overlap().coverage(split) == []
    deep = put(split, "arm_R", HAND, 1, at_block(depth=CALIBRATION_FAIL * SKIN))
    assert items(deep) == [ITEM_A]


def test_an_open_body_shell_is_unmeasured_and_no_target():
    ctx = make_ctx()
    rest = _rest()
    torso = BODY["torso"][1][:-1]  # one face short: the torso is open
    opened = dataclasses.replace(ctx, rests={"kit:Doll": dataclasses.replace(
        rest, faces={**rest.faces, "torso": torso})},
        samples={"play": dataclasses.replace(
            ctx.samples["play"], faces={**ctx.samples["play"].faces, "torso": torso})})
    lines = Overlap().coverage(opened)
    assert "overlap: skipped (open shell, no inside): rig:torso/body" in lines
    assert isinstance(next(x for x in lines if "open shell" in x), Unmeasured)
    assert findings(put(opened, "arm_R", HAND, 1, (0.0, 0.15, FRONT - 5 * SKIN))) == []


# --- the gate -----------------------------------------------------------------------------------


def test_the_plant_is_caught_by_run_gate_and_a_real_overlap_fails_the_run():
    verdict = run_gate([Overlap()], scene_a(CALIBRATION_FAIL * SKIN), None)
    assert verdict.status == "failed"
    assert verdict.failed == 1
    assert verdict.lines[0].startswith(f"overlap | {ITEM_A} | depth ")


def test_a_skin_no_plant_can_beat_stops_the_gate(monkeypatch):
    from model_kit.core.checks import overlap

    monkeypatch.setattr(overlap, "SKIN_OF_EFFECTOR", 1e6)  # a budget no plant reaches
    ctx = make_ctx()
    units = Overlap().plants(ctx)
    assert [u.item.split(" ")[1] for u in units] == list("ABCD")
    assert not any(u.not_applicable for u in units)  # every kind is judged: none is dropped
    with pytest.raises(BlindCheck):
        run_gate([Overlap()], ctx, None)


# --- plants as units --------------------------------------------------------------------

# The scene, and the tests whose contact the hold or the seat excuses.
HELD = {
    "a press hold on the block": (lambda: make_ctx(holds=(hold("kit:Block.top_R"),)), "AB"),
    "a grip hold on the block": (lambda: make_ctx(holds=(hold("kit:Block.wrap_R"),)), "AB"),
    "a grip hold on the rod": (lambda: make_ctx(holds=(hold("kit:Rod.grip_R"),)), "AB"),
    "the rod holding the block": (
        lambda: make_ctx(holds=(hold("kit:Block.top_R", who="rod"),)), "D"),
    "a seat": (lambda: make_ctx(seat=True), "AB"),
}


@pytest.mark.parametrize("name", sorted(HELD))
def test_a_scene_with_a_hold_or_the_seat_has_a_caught_unit_for_each_test_the_excuse_touches(name):
    """In a scene with a hold or the seat, each test that excuse touches (A and B, or D) has a unit
    the check catches, and the gate does not stop. It does not claim the planted item is the
    excused one: items with no hold or seat on them are planted first, so the excuse path itself
    is not proven here (this test does not close that gap)."""
    make, held = HELD[name]
    ctx = make()
    tried = {i.split(" ")[1] for i in tried_and_caught(Overlap(), ctx)}
    assert set(held) <= tried
    assert run_gate([Overlap()], ctx, None).status in {"passed", "partial"}


def test_the_units_are_tried_and_caught_on_two_pieces():
    from ..bodies.humanoid import humanoid_scene
    assert {i.split(" ")[1] for i in tried_and_caught(Overlap(), make_ctx())} == set("ABCD")
    assert tried_and_caught(Overlap(), humanoid_scene())


def test_a_unit_holds_its_item_at_the_first_middle_and_last_sample():
    """At each of those samples it tries the fractions of the room, shallowest first."""
    from model_kit.core.checks.overlap import ROOM_FRACTIONS

    units = {u.item.split(" ")[1]: u for u in Overlap().plants(make_ctx())}
    depths = len(ROOM_FRACTIONS)
    assert [sorted(p.samples) for p in units["C"].candidates] == [
        [s] for s in (0, 2, 3) for _ in range(depths)]
    assert all(len({p.item for p in u.candidates}) == 1 for u in units.values())
