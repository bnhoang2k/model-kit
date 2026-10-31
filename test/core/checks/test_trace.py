import dataclasses
import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks import trace
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.checks.pieces import with_piece
from model_kit.core.checks.reach import Reach
from model_kit.core.checks.trace import Trace
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import (
    Clip,
    FloorGear,
    Pack,
    Placement,
    Prop,
    Ref,
    Target,
    View,
    load_pack,
)
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import ClipSamples, PieceGeom, RigData, SceneCtx, piece_key
from model_kit.core.sight import seen_distance
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS
from model_kit.core.units import Style

from ..scenes import matrix, quat_y, turn_z
from ..shapes import plate
from .piece_scenes import accepted, refuses_the_core, tried_and_caught
from .sight_scenes import unviewed, view_json, viewed

THICKNESS = 0.002
TOL = 0.001  # the contact length of every piece here (its view's seen floor)
STYLE = Style(units_per_metre=1.0, page_camera={}, dials={})
TIMES = np.arange(1.0, 11.0)
HOLD_POINT = (0.0, 0.002, 0.0)
HALF = 0.1  # the 0.2 wide plate reaches 0.1 from its centre along x and z
EDGE_POINT = (0.08, 0.002, 0.0)  # an off-centre hold, 0.02 from the plate's +x edge
SEAT_POINT = (0.0, 0.4, 0.0)
RIM_POINT = (0.1 - 1.5 * TOL, 0.002, 0.0)  # 1.5 tol inside the +x rim
HOLD_POINTS = {"top_R": HOLD_POINT, "edge_R": EDGE_POINT, "rim_R": RIM_POINT,
               "top": SEAT_POINT}
BELOW, FORWARD = 0.08, 0.03
IDENTITY_Q = (0.0, 0.0, 0.0, 1.0)


def _piece(parts, holds=(), seats=()):
    return {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test",
                    "min_thickness": THICKNESS}],
        "parts": [{"name": p, "listed": "stock", "shape": "box", "on": None, "group": "g",
                   "role": "solid"} for p in parts],
        "uses": [{"id": "front", "point": [0.0, 0.0, 0.3], "facing": [0.0, 0.0, -1.0],
                  "posture": "sit" if seats else "stand"}],
        "holds": list(holds), "seats": list(seats), "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {}, "view": view_json(TOL),
    }


def _hold(id_, surface, point, off=(0.0, 1.0, 0.0), sub_box=None):
    out = {"id": id_, "use": "front", "side": "R", "surface": surface, "point": list(point),
           "off": list(off), "grip": False}
    if sub_box is not None:
        out["sub_box"] = sub_box
    return out


@cache
def kit_pack() -> Pack:
    doll = _piece(["torso"])
    doll["uses"] = []
    doll["rig"] = {
        "armature": "TestRig",
        "bones": [
            {"name": "body", "parent": None, "length": 0.3},
            {"name": "arm.R", "parent": "body", "length": 0.1},
            {"name": "hand.R", "parent": "arm.R", "length": 0.05},
        ],
        "bind": {"torso": {"bones": ["body", "arm.R", "hand.R"], "mode": "chain"}},
        "contact_groups": {"hand.R": ["hand.R"]},
        "chains": {"arm.R": {
            "root": "body", "bones": ["arm.R", "hand.R"], "effector": "hand.R",
            "effector_radius": 0.0586, "dofs": {"arm.R": "xyz", "hand.R": "xyz"},
            "limits": {}, "effector_group": "hand.R",
        }},
        "sit": {"below_seat": BELOW, "forward": FORWARD},
        "mirror": {"pairs": [], "axis": "x"},
    }
    seat = {"id": "top", "use": "front", "point": list(SEAT_POINT), "facing": [0.0, 0.0, 1.0],
            "surface": ["top"]}
    data = {"pack": "kit", "units_per_metre": 1.0, "pieces": {
        "Plate": _piece(["top"], [_hold("top_R", ["top"], HOLD_POINT),
                                  _hold("edge_R", ["top"], EDGE_POINT),
                                  _hold("rim_R", ["top"], RIM_POINT)]),
        "Throne": _piece(["top"], seats=[seat]),
        "Doll": doll,
    }}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


RIG = RigData(order=("body",), parent={"body": None}, rest={"body": np.eye(4)}, world=np.eye(4),
              effector_mid={"hand.R": np.zeros(3)})
GEOM = {
    piece_key("kit", "Plate"): PieceGeom("Plate", {"top": plate()}),
    piece_key("kit", "Throne"): PieceGeom("Throne", {"top": plate()}),
}


def make_ctx(offset=(0.0, 0.0, 0.0), *, baked="fresh", ref="kit:Plate.top_R", position=(0, 0, 0),
             quaternion=IDENTITY_Q, authoring="solved", seat=None, root=None, body=None,
             key=3, carried=False, samples=True, prop_scale=1.0, seat_when=None,
             roots=None) -> SceneCtx:
    """One clip, `hit`, frames 1..10. By default one target on `kit:Plate.top_R`.

    `baked="fresh"` bakes the point the scene really resolves to; pass a vector to override
    it or None for no baked point. A seated clip puts a throne at `position` and a root bone
    at `root` (default: where the seat rule puts it), or at `roots`, one root point per sample
    in `TIMES`. `seat_when` is the clip's seat window list (None: the whole clip).
    """
    plate_p = Placement("Plate", "Plate", "kit", tuple(float(v) for v in position), quaternion)
    throne = Placement("Throne", "Throne", "kit", tuple(float(v) for v in position), quaternion)
    gear = {"Throne": throne}
    props = {}
    prop_pose = None
    if carried:
        props["plate"] = Prop("plate", "Plate", "kit", "hand.R", (0.0, 0.0, 0.0), IDENTITY_Q,
                              ((1.0, 10.0),))
        prop_pose = matrix((0.5, 0.5, 0.0), prop_scale * turn_z(90))
    else:
        gear["Plate"] = plate_p
    targets = ()
    if authoring == "solved":
        targets = (Target(key=key, chain="arm.R", ref=Ref.parse(ref), offset=tuple(offset),
                          why="test", baked=None, tool=None),)
    clip = Clip(name="hit", frames=(1, 10), fps=24.0, authoring=authoring,
                seat=None if seat is None else Ref.parse(seat),
                holds=(), targets=targets, floor_gear=FloorGear(gear), props=props,
                mirror_of=None, seat_when=None if seat_when is None else tuple(seat_when))
    s = len(TIMES)
    body_m = np.tile(matrix() if body is None else body, (s, 1, 1))
    bones = {"body": body_m}
    samples_obj = ClipSamples(clip="hit", times=TIMES.copy(), groups={}, bones=bones,
                              props={} if prop_pose is None else
                              {"plate": np.tile(prop_pose, (s, 1, 1))})
    ctx = SceneCtx(style=STYLE, packs={"kit": kit_pack()}, clips={"hit": clip},
                   pieces=dict(GEOM), samples={"hit": samples_obj} if samples else {},
                   rigs={"kit:Doll": RIG}, rig_of_clip={"hit": "kit:Doll"})
    if authoring == "solved":
        t = targets[0]
        if baked == "fresh":
            baked = _resolved(ctx, t, key)
        # Re-bake with the offset: a baked point is what the solver aimed at.
        clip = dataclasses.replace(clip, targets=(dataclasses.replace(
            t, baked=None if baked is None else tuple(float(v) for v in baked)),))
        ctx = dataclasses.replace(ctx, clips={"hit": clip})
    if seat is not None:
        want = _expected_root(position, quaternion)
        if roots is not None:
            bones = {"body": np.stack([matrix(r) for r in roots])}
        else:
            bones = {"body": np.tile(matrix(want if root is None else root), (s, 1, 1))}
        samples_obj = dataclasses.replace(samples_obj, bones=bones)
        ctx = dataclasses.replace(ctx, samples={"hit": samples_obj} if samples else {})
    return ctx


def _resolved(ctx, target, key) -> np.ndarray:
    """The world point the target's ref plus offset names, worked out here with no check code."""
    clip = ctx.clips["hit"]
    ref = target.ref
    if ref.is_bone:
        m = ctx.samples["hit"].bones[ref.id][key - 1]
        return m[:3, :3] @ np.asarray(target.offset) + m[:3, 3]
    local = np.asarray(HOLD_POINTS[ref.id]) + np.asarray(target.offset)
    if clip.props:
        m = ctx.samples["hit"].props["plate"][key - 1]
        spin = m[:3, :3] / np.linalg.norm(m[:3, 0])  # the pose without its scale
        # the hold point rides the scaled geometry; the offset keeps its world length
        return m[:3, :3] @ np.asarray(HOLD_POINTS[ref.id]) + m[:3, 3] + spin @ np.asarray(
            target.offset)
    return clip.floor_gear.pieces[ref.piece].to_world(local)


def _expected_root(position, quaternion) -> np.ndarray:
    throne = Placement("Throne", "Throne", "kit", tuple(float(v) for v in position), quaternion)
    top = throne.to_world(SEAT_POINT)
    facing = throne.to_world((0.0, 0.0, 1.0)) - throne.to_world((0.0, 0.0, 0.0))
    return top - np.array([0.0, BELOW, 0.0]) + FORWARD * facing


def findings(ctx):
    return Trace().run(ctx)


def accepted_plants(ctx) -> list:
    """The candidate the gate accepts from each unit (the first the check catches)."""
    check = Trace()
    return [accepted(check, u) for u in check.plants(ctx)]


def plants_caught(ctx, labels) -> bool:
    """Every unit has a candidate the check catches, and the accepted ones are `labels`."""
    got = accepted_plants(ctx)
    if any(p is None for p in got):
        return False
    assert sorted(p.label.split(":")[0] for p in got) == sorted(labels)
    return True


def retarget(ctx, **changes):
    clip = ctx.clips["hit"]
    target = dataclasses.replace(clip.targets[0], **changes)
    clip = dataclasses.replace(clip, targets=(target,))
    return dataclasses.replace(ctx, clips={"hit": clip})


def baked_of(ctx) -> np.ndarray:
    return np.asarray(ctx.clips["hit"].targets[0].baked)


def move_hold(ctx, delta) -> SceneCtx:
    pack = kit_pack()
    piece = pack.pieces["Plate"]
    hold = dataclasses.replace(piece.holds[0], point=tuple(np.add(piece.holds[0].point, delta)))
    piece = dataclasses.replace(piece, holds=(hold,))
    pack = dataclasses.replace(pack, pieces={**pack.pieces, "Plate": piece})
    return dataclasses.replace(ctx, packs={"kit": pack})


# --- stale ----------------------------------------------------------------------------------


def test_fresh_target_passes():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert run_gate([Trace()], ctx, None).status == "passed"


def test_half_tol_drift_passes():
    ctx = make_ctx()
    drifted = baked_of(ctx) + [CALIBRATION_PASS * TOL, 0.0, 0.0]
    assert findings(retarget(ctx, baked=tuple(drifted))) == []


def test_double_tol_drift_reads_stale():
    ctx = make_ctx()
    drifted = baked_of(ctx) + [0.0, CALIBRATION_FAIL * TOL, 0.0]
    found = findings(retarget(ctx, baked=tuple(drifted)))
    assert len(found) == 1
    assert found[0].check == "trace" and "stale" in found[0].detail
    assert "hit" in found[0].item and "kit:Plate.top_R" in found[0].item


def test_moved_hold_reads_stale():
    ctx = make_ctx()
    moved = move_hold(ctx, (0.0, CALIBRATION_FAIL * TOL, 0.0))
    found = findings(moved)
    assert len(found) == 1 and "stale" in found[0].detail


def test_offset_is_in_the_pieces_frame():
    # The piece is turned 90 degrees about Y, so a piece-frame +x offset points along world -z.
    ctx = make_ctx((0.05, 0.0, 0.0), quaternion=quat_y(90))
    expected = Placement("Plate", "Plate", "kit", (0, 0, 0), quat_y(90)).to_world(
        np.add(HOLD_POINT, (0.05, 0.0, 0.0)))
    assert np.allclose(baked_of(ctx), expected)
    assert expected[2] == pytest.approx(-0.05)
    assert findings(ctx) == []


def test_rig_ref_reads_the_bone_at_the_key():
    body = matrix((1.0, 2.0, 3.0), turn_z(90))
    ctx = make_ctx((0.1, 0.0, 0.0), ref="rig:body", body=body)
    # In the bone frame +x is world +y after a 90 degree turn about Z.
    assert np.allclose(baked_of(ctx), (1.0, 2.1, 3.0))
    assert findings(ctx) == []
    off = findings(retarget(ctx, baked=(1.1, 2.0, 3.0)))
    assert len(off) == 1 and "stale" in off[0].detail


def test_carried_piece_uses_the_prop_pose():
    ctx = make_ctx((0.0, 0.01, 0.0), carried=True)
    assert np.allclose(baked_of(ctx), (0.5 - 0.012, 0.5, 0.0))
    assert findings(ctx) == []
    assert len(findings(retarget(ctx, baked=(0.5, 0.5, 0.0)))) == 1


def test_key_reads_the_nearest_sample():
    body = np.stack([matrix((float(t), 0.0, 0.0)) for t in TIMES])
    ctx = make_ctx(ref="rig:body", key=4)
    ctx = dataclasses.replace(ctx, samples={"hit": dataclasses.replace(
        ctx.samples["hit"], bones={"body": body})})
    ctx = retarget(ctx, baked=(4.0, 0.0, 0.0))
    assert findings(ctx) == []


# --- off surface ----------------------------------------------------------------------------


def test_raised_offset_passes():
    ctx = make_ctx((0.0, 0.3, 0.0))  # a stick lifted well above the plate, along `off`
    assert findings(ctx) == []


def test_in_plane_offset_inside_passes():
    assert findings(make_ctx((0.5 * HALF, 0.02, 0.0))) == []


def test_in_plane_offset_off_surface_fails():
    ctx = make_ctx((1.5 * HALF, 0.0, 0.0))
    found = findings(ctx)  # baked is fresh, so only the surface test can fire
    assert len(found) == 1 and "off the surface" in found[0].detail


def test_off_surface_checks_the_in_plane_part_only():
    # A diagonal offset: 0.08 in plane along x, plus a raise. Inside the plate either way.
    assert findings(make_ctx((0.08, 0.5, 0.0))) == []
    assert len(findings(make_ctx((0.08, 0.5, 0.2)))) == 1  # 0.2 along z is past the 0.1 half


def test_off_centre_hold_toward_the_near_edge_fails():
    # The hold sits 0.02 from the +x edge. 0.05 toward it lands off the plate, though 0.05 is
    # well under the plate's half-width.
    found = findings(make_ctx((0.05, 0.0, 0.0), ref="kit:Plate.edge_R"))
    assert len(found) == 1 and "off the surface" in found[0].detail
    assert "0.0200" in found[0].detail  # the surface reaches 0.02 that way


def test_off_centre_hold_toward_the_far_side_passes():
    assert findings(make_ctx((-0.05, 0.0, 0.0), ref="kit:Plate.edge_R")) == []
    assert findings(make_ctx((-0.15, 0.0, 0.0), ref="kit:Plate.edge_R")) == []  # to the far edge
    assert len(findings(make_ctx((-0.2, 0.0, 0.0), ref="kit:Plate.edge_R"))) == 1


def test_edge_offset_within_tol_of_the_edge_passes():
    assert findings(make_ctx((0.02 + CALIBRATION_PASS * TOL, 0.0, 0.0),
                             ref="kit:Plate.edge_R")) == []
    assert len(findings(make_ctx((0.02 + CALIBRATION_FAIL * TOL, 0.0, 0.0),
                                 ref="kit:Plate.edge_R"))) == 1


def test_other_refs_have_no_surface_test():
    ctx = make_ctx((5.0, 0.0, 0.0), ref="rig:body")  # far from anything, but baked fresh
    assert findings(ctx) == []


# --- seat root ------------------------------------------------------------------------------


def test_seat_root_checked():
    ctx = make_ctx(seat="kit:Throne.top", position=(0.5, 0.0, 0.2), quaternion=quat_y(90))
    assert findings(ctx) == []
    want = _expected_root((0.5, 0.0, 0.2), quat_y(90))
    assert want[1] == pytest.approx(SEAT_POINT[1] - BELOW)
    near = make_ctx(seat="kit:Throne.top", position=(0.5, 0.0, 0.2), quaternion=quat_y(90),
                    root=want + [CALIBRATION_PASS * TOL, 0.0, 0.0])
    assert findings(near) == []
    far = make_ctx(seat="kit:Throne.top", position=(0.5, 0.0, 0.2), quaternion=quat_y(90),
                   root=want + [0.0, CALIBRATION_FAIL * TOL, 0.0])
    found = findings(far)
    assert len(found) == 1 and "seat" in found[0].item and "root" in found[0].detail


def _with_rig(ctx, **fields):
    pack = ctx.packs["kit"]
    doll = pack.pieces["Doll"]
    rig = dataclasses.replace(doll.rig, **fields)
    pieces = {**pack.pieces, "Doll": dataclasses.replace(doll, rig=rig)}
    return dataclasses.replace(ctx, packs={"kit": dataclasses.replace(pack, pieces=pieces)})


@pytest.mark.parametrize("key", ["gone:Doll", "kit:Gone"])
def test_a_rig_ref_on_a_pack_or_piece_the_scene_lacks_raises_the_named_error(key):
    """The loader refuses a rig file for such a piece, but a scene built in memory can name one:
    the check raises `ManifestError` naming the clip, not a bare `KeyError`."""
    ctx = make_ctx(ref="rig:body")
    ctx = dataclasses.replace(ctx, rigs={**ctx.rigs, key: RIG}, rig_of_clip={"hit": key})
    with pytest.raises(ManifestError, match="clip 'hit'.*rigged piece"):
        findings(ctx)


def test_seat_root_reads_the_named_root():
    from model_kit.core.manifest import Bone

    ctx = make_ctx(seat="kit:Throne.top")
    rig = ctx.packs["kit"].pieces["Doll"].rig
    two = _with_rig(ctx, bones=(*rig.bones, Bone("tail", None, 0.1)))
    with pytest.raises(ManifestError, match=r"parentless bones.*tail"):
        findings(two)
    assert findings(_with_rig(two, root="body")) == []


def test_a_seat_on_a_rig_with_no_sit_raises_naming_the_clip():
    ctx = _with_rig(make_ctx(seat="kit:Throne.top"), sit=None)
    with pytest.raises(ManifestError, match="clip 'hit'.*no sit"):
        findings(ctx)


def test_seat_root_follows_the_forward_offset():
    # Forward is along the seat's facing: a root put FORWARD behind instead fails.
    pos = (0.0, 0.0, 0.0)
    want = _expected_root(pos, IDENTITY_Q)
    wrong = want - [0.0, 0.0, 2 * FORWARD]
    assert len(findings(make_ctx(seat="kit:Throne.top", position=pos, root=wrong))) == 1


def _rising_roots(start, step=0.5):
    """Roots on `start` at the first sample and rising by `step` per sample after it."""
    return [np.asarray(start) + [0.0, step * i, 0.0] for i in range(len(TIMES))]


def test_seat_root_runs_on_every_sample_by_default():
    want = _expected_root((0, 0, 0), IDENTITY_Q)
    ctx = make_ctx(seat="kit:Throne.top", roots=_rising_roots(want))
    (found,) = findings(ctx)
    assert found.item == "hit seat: kit:Throne.top"
    # The largest gap is at the last sample, frame 10: 0.5 x 9 above the seat-derived root.
    assert found.detail == "root is 4.5 from the seat-derived root at frame 10, over 0.001"


def test_honest_sit_down_needs_its_window():
    want = _expected_root((0, 0, 0), IDENTITY_Q)
    roots = [want + [0.0, 0.1, 0.0] if t <= 7 else want for t in TIMES]
    sat = make_ctx(seat="kit:Throne.top", roots=roots, seat_when=((8, 10),))
    assert findings(sat) == []
    (found,) = findings(make_ctx(seat="kit:Throne.top", roots=roots))
    assert "at frame 1," in found.detail  # a tie takes the earliest sample


def test_root_drift_inside_window_fails():
    want = _expected_root((0, 0, 0), IDENTITY_Q)
    roots = [want + [0.0, CALIBRATION_FAIL * TOL, 0.0] if t == 9 else want for t in TIMES]
    (found,) = findings(make_ctx(seat="kit:Throne.top", roots=roots,
                                 seat_when=((8, 10),)))
    assert "at frame 9," in found.detail


def test_turned_seat_inside_window_passes():
    pos = (0.5, 0.0, 0.2)
    want = _expected_root(pos, quat_y(90))
    roots = [want + [0.0, 0.3, 0.0] if t < 8 else want for t in TIMES]
    ctx = make_ctx(seat="kit:Throne.top", position=pos, quaternion=quat_y(90), roots=roots,
                   seat_when=((8, 10),))
    assert findings(ctx) == []


def test_seat_window_with_no_sample_is_unmeasured():
    ctx = make_ctx(seat="kit:Throne.top", seat_when=((2.2, 2.3),))
    assert findings(ctx) == []
    line = "trace: skipped (no sample in seat_when): hit seat: kit:Throne.top"
    assert line in Trace().coverage(ctx)
    assert run_gate([Trace()], ctx, None).status == "partial"


def test_seat_plant_caught_inside_window():
    ctx = make_ctx(authoring="keyed", seat="kit:Throne.top", seat_when=((8, 10),))
    assert findings(ctx) == []
    assert plants_caught(ctx, ["seat"])


def test_code_built_bad_seat_window_raises_when_built():
    """The check never meets these clips: a clip with such a window refuses to be built."""
    ctx = make_ctx(seat="kit:Throne.top")
    clip = ctx.clips["hit"]
    for windows, match in [(((5, 3),), r"seat_when\[0\]: window \[5, 3\] runs backwards"),
                           (((0, 4),), r"seat_when\[0\]: window \[0, 4\] is outside"),
                           (((2, 40),), r"seat_when\[0\]: window \[2, 40\] is outside"),
                           ((), "seat_when: must not be empty"),
                           (((1,),), r"seat_when\[0\]: must be \[start, end\]")]:
        with pytest.raises(ManifestError, match=f"clip 'hit': {match}"):
            dataclasses.replace(clip, seat_when=windows)


def test_code_built_seat_window_without_a_seat_cannot_be_built():
    """The check never meets this clip: a clip with a seat window and no seat refuses to be built."""
    ctx = make_ctx()
    with pytest.raises(ManifestError, match="clip 'hit': seat_when: needs a seat"):
        dataclasses.replace(ctx.clips["hit"], seat_when=((2, 5),))


def test_the_seat_rule_runs_on_a_keyed_seated_clip():
    # The seat is a clip fact, not a target: a keyed clip that sits is checked too.
    ctx = make_ctx(authoring="keyed", seat="kit:Throne.top", position=(0.5, 0.0, 0.2),
                   quaternion=quat_y(90))
    assert findings(ctx) == []
    assert plants_caught(ctx, ["seat"])
    assert run_gate([Trace()], ctx, None).status == "passed"
    want = _expected_root((0.5, 0.0, 0.2), quat_y(90))
    far = make_ctx(authoring="keyed", seat="kit:Throne.top", position=(0.5, 0.0, 0.2),
                   quaternion=quat_y(90), root=want + [0.0, CALIBRATION_FAIL * TOL, 0.0])
    (found,) = findings(far)
    assert found.item == "hit seat: kit:Throne.top" and "root" in found.detail


def test_keyed_seated_clip_through_all_three_checks():
    # A keyed clip has no holds and no solved targets; only the seat rule measures it.
    want = _expected_root((0, 0, 0), IDENTITY_Q)
    roots = [want + [0.0, 0.1, 0.0] if t < 8 else want for t in TIMES]
    ctx = make_ctx(authoring="keyed", seat="kit:Throne.top", seat_when=((8, 10),), roots=roots)
    verdict = run_gate([MustTouch(), Reach(), Trace()], ctx, None)
    assert verdict.status == "passed"
    assert set(verdict.coverage) == {"must-touch: not applicable (no holds): hit",
                                     "reach: not applicable (keyed): hit",
                                     "trace: not applicable (keyed targets): hit"}
    assert plants_caught(ctx, ["seat"])
    off = [want + [0.0, CALIBRATION_FAIL * TOL, 0.0] if t == 9 else r
           for t, r in zip(TIMES, roots)]
    bad = make_ctx(authoring="keyed", seat="kit:Throne.top", seat_when=((8, 10),), roots=off)
    failed = run_gate([MustTouch(), Reach(), Trace()], bad, None)
    assert failed.status == "failed" and failed.failed == 1
    assert "hit seat: kit:Throne.top" in failed.lines[0]


def test_a_keyed_seated_clip_with_no_samples_is_listed():
    ctx = make_ctx(authoring="keyed", seat="kit:Throne.top", samples=False)
    assert findings(ctx) == []
    assert Trace().coverage(ctx) == ["trace: not applicable (keyed targets): hit",
                                     "trace: skipped (no samples): hit seat: kit:Throne.top"]


# --- coverage and plants --------------------------------------------------------------------


def test_keyed_listed_in_coverage():
    ctx = make_ctx(authoring="keyed")
    assert findings(ctx) == []
    assert Trace().coverage(ctx) == ["trace: not applicable (keyed targets): hit"]
    assert list(Trace().plants(ctx)) == []


def test_missing_samples_listed_in_coverage():
    ctx = make_ctx(samples=False)
    assert findings(ctx) == []
    assert any(line.startswith("trace: skipped") and "hit" in line
               for line in Trace().coverage(ctx))


def test_target_without_baked_listed_and_not_stale():
    ctx = make_ctx(baked=None)
    assert findings(ctx) == []
    lines = Trace().coverage(ctx)
    assert len(lines) == 1 and "baked" in lines[0] and "hit targets[0]" in lines[0]


def test_unreadable_baseline_raises():
    ctx = make_ctx(ref="kit:Plate.top_R")
    ctx = dataclasses.replace(ctx, samples={"hit": dataclasses.replace(
        ctx.samples["hit"], bones={})})
    ctx = retarget(ctx, ref=Ref.parse("rig:body"), baked=(0, 0, 0))
    with pytest.raises(ManifestError):
        findings(ctx)


def test_plants_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])
    seated = make_ctx(seat="kit:Throne.top", position=(0.5, 0.0, 0.2))
    assert findings(seated) == []
    assert plants_caught(seated, ["stale", "off-surface", "seat"])


def test_plants_caught_for_an_off_centre_hold():
    ctx = make_ctx((-0.03, 0.04, 0.0), ref="kit:Plate.edge_R")
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])


def test_plants_caught_with_an_existing_offset_and_turned_piece():
    ctx = make_ctx((0.02, 0.1, 0.03), quaternion=quat_y(30), position=(1.0, 0.0, 0.0))
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])


def test_off_surface_plant_fails_only_on_the_surface_test():
    ctx = make_ctx()
    plant = next(p for p in accepted_plants(ctx) if p.label.startswith("off-surface"))
    found = Trace().run(plant.ctx)
    assert len(found) == 1 and "off the surface" in found[0].detail


# --- plants near a rim, seat refs, sample times, rig plants -------------------------


def test_plant_caught_when_the_hold_point_is_within_a_tolerance_of_the_rim():
    ctx = make_ctx((1e-4, 0.0, 0.0), ref="kit:Plate.rim_R")
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])
    assert run_gate([Trace()], ctx, None).status == "passed"  # no BlindCheck


def test_a_seat_ref_target_is_traced():
    ctx = make_ctx((0.0, 0.05, 0.0), ref="kit:Throne.top", position=(0.5, 0.0, 0.2))
    assert np.allclose(baked_of(ctx), (0.5, 0.45, 0.2))
    assert findings(ctx) == []
    stale = findings(retarget(ctx, baked=tuple(baked_of(ctx) + [0.0, CALIBRATION_FAIL * TOL, 0])))
    assert len(stale) == 1 and "stale" in stale[0].detail and "kit:Throne.top" in stale[0].item
    units = list(Trace().plants(ctx))  # its own point moves: the stale test has a plant
    assert [u.item for u in units] == [stale[0].item]
    assert plants_caught(ctx, ["stale"])


def test_nearest_sample_is_not_the_exact_one():
    times = TIMES + 0.3  # no sample sits on a whole frame
    body = np.stack([matrix((float(i), 0.0, 0.0)) for i in range(len(TIMES))])
    ctx = make_ctx(ref="rig:body", key=4)
    ctx = dataclasses.replace(ctx, samples={"hit": dataclasses.replace(
        ctx.samples["hit"], times=times, bones={"body": body})})
    assert findings(retarget(ctx, baked=(3.0, 0.0, 0.0))) == []  # the sample at 4.3
    assert len(findings(retarget(ctx, baked=(4.0, 0.0, 0.0)))) == 1  # the sample at 5.3


def test_a_key_the_samples_do_not_reach_raises():
    ctx = make_ctx(ref="rig:body", key=9)
    short = dataclasses.replace(ctx.samples["hit"], times=TIMES[:5].copy(),
                                bones={"body": ctx.samples["hit"].bones["body"][:5]})
    ctx = dataclasses.replace(ctx, samples={"hit": short})
    with pytest.raises(ManifestError, match="no sample near"):
        findings(ctx)


def test_a_scene_of_rig_targets_still_has_a_plant():
    ctx = make_ctx((0.1, 0.0, 0.0), ref="rig:body", body=matrix((1.0, 2.0, 3.0), turn_z(90)))
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale"])
    assert run_gate([Trace()], ctx, None).status == "passed"


def test_coverage_lines_name_the_reason():
    assert Trace().coverage(make_ctx(authoring="keyed")) == ["trace: not applicable (keyed targets): hit"]
    assert Trace().coverage(make_ctx(samples=False)) == ["trace: skipped (no samples): hit"]
    (line,) = Trace().coverage(make_ctx(baked=None))
    assert line.startswith("trace: skipped (no baked point): hit targets[0]")


def test_scaled_prop_s_tenth_plants_are_not_blind():
    ctx = make_ctx(carried=True, prop_scale=0.1)
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])
    assert run_gate([Trace()], ctx, None).status == "passed"  # no BlindCheck


def test_scaled_prop_s_tenth_surface_reaches_a_tenth_as_far():
    # The plate reaches 0.1 in piece units, which is 0.01 in the world at s = 0.1.
    assert findings(make_ctx((0.005, 0.0, 0.0), carried=True, prop_scale=0.1)) == []
    found = findings(make_ctx((0.05, 0.0, 0.0), carried=True, prop_scale=0.1))
    assert len(found) == 1 and "off the surface" in found[0].detail


def test_scaled_prop_s_ten_surface_reaches_ten_times_as_far():
    # The plate reaches 0.1 in piece units, which is 1.0 in the world at s = 10.
    ctx = make_ctx((0.5, 0.0, 0.0), carried=True, prop_scale=10.0)
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])
    found = findings(make_ctx((1.5, 0.0, 0.0), carried=True, prop_scale=10.0))
    assert len(found) == 1 and "off the surface" in found[0].detail


def test_scaled_prop_stale_is_measured_in_world_length():
    ctx = make_ctx(carried=True, prop_scale=10.0)
    near = baked_of(ctx) + [CALIBRATION_PASS * TOL, 0.0, 0.0]
    assert findings(retarget(ctx, baked=tuple(near))) == []
    far = baked_of(ctx) + [CALIBRATION_FAIL * TOL, 0.0, 0.0]
    assert len(findings(retarget(ctx, baked=tuple(far)))) == 1


def test_stale_plants_for_a_seat_ref_scaled_and_turned():
    ctx = make_ctx((0.0, 0.05, 0.0), ref="kit:Throne.top", position=(0.5, 0.0, 0.2),
                   quaternion=quat_y(40))
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale"])


@pytest.mark.parametrize("scale", [10.0, 0.1])
def test_edge_slack_is_the_world_tolerance_at_any_scale(scale):
    edge = HALF * scale  # the plate reaches 0.1 in piece units
    inside = make_ctx((edge + CALIBRATION_PASS * TOL, 0.0, 0.0), carried=True, prop_scale=scale)
    assert findings(inside) == []
    outside = make_ctx((edge + CALIBRATION_FAIL * TOL, 0.0, 0.0), carried=True, prop_scale=scale)
    found = findings(outside)
    assert len(found) == 1 and "off the surface" in found[0].detail


def test_rim_plant_is_not_blind_at_a_small_scale():
    # The hold sits 1.5 tol (piece units) inside the rim and the prop is drawn at 0.1, so the
    # slack in piece units is ten times the world tolerance. The plant's margin must be
    # converted the same way or it lands inside the slack.
    ctx = make_ctx((1e-6, 0.0, 0.0), ref="kit:Plate.rim_R", carried=True, prop_scale=0.1)
    assert findings(ctx) == []
    assert plants_caught(ctx, ["stale", "off-surface"])
    assert run_gate([Trace()], ctx, None).status == "passed"  # no BlindCheck


def test_a_plant_measures_only_the_target_or_seat_it_planted(monkeypatch):
    ctx = make_ctx(seat="kit:Throne.top")
    clip = ctx.clips["hit"]
    ctx = dataclasses.replace(ctx, clips={"hit": dataclasses.replace(
        clip, targets=clip.targets * 3)})
    calls = []
    real_target, real_ref = trace.resolve_target, trace.resolve_ref

    def spy_target(ctx, clip_name, target):
        calls.append(target.chain)
        return real_target(ctx, clip_name, target)

    def spy_ref(ctx, clip_name, ref, key, **kw):
        calls.append("")  # the seat rule resolves the seat ref, with no chain
        return real_ref(ctx, clip_name, ref, key, **kw)

    monkeypatch.setattr(trace, "resolve_target", spy_target)
    monkeypatch.setattr(trace, "resolve_ref", spy_ref)
    plants = accepted_plants(ctx)
    assert sorted(p.label.split(":")[0] for p in plants) == ["off-surface"] * 3 + ["seat"] + [
        "stale"] * 3
    for plant in plants:
        assert plant.scope == frozenset({plant.item})
        calls.clear()
        found = Trace().run(plant.ctx, only=plant.scope)
        assert found and {f.item for f in found} == {plant.item}
        # One target resolve for a target plant; one seat resolve for the seat plant.
        # The seat rule resolves the seat at each of the 10 due samples.
        assert calls == ([""] * len(TIMES) if plant.label.startswith("seat") else ["arm.R"])
    calls.clear()
    assert Trace().run(ctx) == [] and sorted(calls) == [""] * len(TIMES) + ["arm.R"] * 3


def test_the_tolerance_prints_unrounded(monkeypatch):
    tol = 0.000075  # a near view's contact length
    monkeypatch.setattr(trace, "_tolerance", lambda ctx, pack, piece: tol)
    ctx = make_ctx()
    drifted = baked_of(ctx) + [0.0, CALIBRATION_FAIL * tol, 0.0]
    (found,) = findings(retarget(ctx, baked=tuple(drifted)))
    assert found.detail.endswith("over 0.000075"), found.detail
    assert "resolves 0.00015 from" in found.detail
    far = make_ctx(seat="kit:Throne.top", root=_expected_root((0, 0, 0), IDENTITY_Q)
                   + [0.0, CALIBRATION_FAIL * tol, 0.0])
    (seat,) = findings(far)
    assert seat.detail == ("root is 0.00015 from the seat-derived root at frame 1, over "
                           "0.000075")


def test_missing_samples_make_the_run_partial_and_by_design_skips_do_not():
    assert run_gate([Trace()], make_ctx(samples=False), None).status == "partial"
    seated = make_ctx(authoring="keyed", seat="kit:Throne.top", samples=False)
    assert run_gate([Trace()], seated, None).status == "partial"
    assert run_gate([Trace()], make_ctx(authoring="keyed"), None).status == "passed"
    assert run_gate([Trace()], make_ctx(baked=None), None).status == "passed"


def _with_decal_pack(ctx):
    """The scene plus a pack whose one piece is seen from so near that its contact length is
    0.00015."""
    decal = _piece(["sticker"])
    decal["view"] = view_json(0.00015)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decals.pack.json"
        path.write_text(json.dumps({"pack": "decals", "units_per_metre": 1.0,
                                    "pieces": {"Sticker": decal}}))
        pack = load_pack(path, STYLE)
    return dataclasses.replace(ctx, packs={**ctx.packs, "decals": pack})


def test_the_tolerance_is_the_target_s_own_pieces():
    # A piece seen from near in another pack does not tighten this piece's trace tolerance.
    ctx = _with_decal_pack(make_ctx())
    near = baked_of(ctx) + [CALIBRATION_PASS * TOL, 0.0, 0.0]
    assert findings(retarget(ctx, baked=tuple(near))) == []
    far = baked_of(ctx) + [CALIBRATION_FAIL * TOL, 0.0, 0.0]
    assert len(findings(retarget(ctx, baked=tuple(far)))) == 1
    seated = _with_decal_pack(make_ctx(seat="kit:Throne.top", root=_expected_root(
        (0, 0, 0), IDENTITY_Q) + [0.0, CALIBRATION_PASS * TOL, 0.0]))
    assert findings(seated) == []
    assert run_gate([Trace()], seated, None).status == "passed"


def test_build_and_trace_use_the_one_seat_rule(monkeypatch):
    from model_kit.core import targets

    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return targets.seat_root(*args, **kwargs)

    monkeypatch.setattr(trace, "seat_root", spy)
    assert trace.Trace().run(make_ctx(seat="kit:Throne.top")) == []
    assert len(calls) == len(TIMES)  # one call per due sample
    build = (Path(targets.__file__).parents[1] / "blender" / "build.py").read_text()
    body = build[build.index("def seated_root("):]
    body = body[:body.index("\ndef ")]
    assert "seat_root(" in body and "below_seat" not in body.split('"""')[-1]
    assert "checks.trace" not in build


# --- the plants are built blind to the check -------------------------------------------------


def test_trace_plants_never_call_the_core(monkeypatch):
    ctx = make_ctx(seat="kit:Throne.top")
    want = sorted(u.item for u in Trace().plants(ctx))
    refuses_the_core(monkeypatch, trace, "_stale_gap", "_off_surface", "_seat_root")
    monkeypatch.setattr(Trace, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in Trace().plants(ctx)) == want
    assert len(want) == 3  # the target's stale and off-surface plants, and the seat's


def test_trace_plants_caught_on_two_pieces():
    """The Plate's hold target (stale and off-surface), the Throne's seat, and the humanoid
    scene's Chair and Bench seats: every unit the gate tries has a candidate the check catches."""
    from ..bodies.humanoid import humanoid_scene

    plate = tried_and_caught(Trace(), make_ctx(seat="kit:Throne.top"))
    assert plate == ["hit targets[0]: key 3 on kit:Plate.top_R"] * 2 + ["hit seat: kit:Throne.top"]
    people = tried_and_caught(Trace(), viewed(humanoid_scene(), TOL))
    assert sorted(people) == ["sit_bench seat: humanoid:Bench.seat_1",
                              "sit_chair seat: humanoid:Chair.seat_1"]
    for ctx in (make_ctx(seat="kit:Throne.top"), humanoid_scene()):
        run_gate([Trace()], ctx, None)  # no BlindCheck


def test_a_stale_unit_tries_the_move_one_way_and_then_the_other():
    ctx = make_ctx()
    (stale, _) = Trace().plants(ctx)
    first, second = stale.candidates
    was = np.asarray(ctx.packs["kit"].pieces["Plate"].holds[0].point)

    def moved(plant):
        return np.asarray(plant.ctx.packs["kit"].pieces["Plate"].holds[0].point) - was

    assert np.allclose(moved(first), -moved(second)) and np.linalg.norm(moved(first)) > 0.0


# --- the contact length is the traced piece's own, and a piece with no view has none -------------


def viewed_piece(ctx, piece, floor):
    """The scene with one piece's view set so its contact length is `floor`."""
    record = dataclasses.replace(ctx.packs["kit"].pieces[piece],
                                 view=View(seen_distance(floor, 1.0), "a test distance"))
    return with_piece(ctx, "kit", piece, record)


def test_a_target_on_a_piece_with_no_view_is_unmeasured_and_makes_the_run_partial():
    ctx = unviewed(make_ctx())
    drifted = baked_of(ctx) + [0.0, 10 * TOL, 0.0]  # far past the tolerance, were there one
    ctx = retarget(ctx, baked=tuple(drifted))
    assert findings(ctx) == []
    assert Trace().plants(ctx) == []
    (line,) = [x for x in Trace().coverage(ctx) if "no view" in x]
    assert isinstance(line, Unmeasured) and line.endswith("on kit:Plate.top_R")
    assert run_gate([Trace()], ctx, None).status == "partial"


def test_a_seat_on_a_piece_with_no_view_is_unmeasured():
    ctx = unviewed(make_ctx(seat="kit:Throne.top", root=_expected_root((0, 0, 0), IDENTITY_Q)
                            + [0.0, 10 * TOL, 0.0]))
    assert [f for f in findings(ctx) if f.item.startswith("hit seat")] == []
    assert [u for u in Trace().plants(ctx) if u.item.startswith("hit seat")] == []
    assert any(isinstance(x, Unmeasured) and "hit seat" in x and "no view" in x
               for x in Trace().coverage(ctx))


def test_each_piece_is_traced_by_its_own_contact_length():
    """The throne is seen from far (a long contact length) or from near: the same drift of the
    seat root passes on one and fails on the other."""
    ctx = make_ctx(seat="kit:Throne.top", root=_expected_root((0, 0, 0), IDENTITY_Q)
                   + [0.0, 3 * TOL, 0.0])
    far = viewed_piece(ctx, "Throne", 10 * TOL)
    assert [f for f in findings(far) if f.item.startswith("hit seat")] == []
    near = viewed_piece(ctx, "Throne", TOL)
    assert [f.item for f in findings(near) if f.item.startswith("hit seat")] == [
        "hit seat: kit:Throne.top"]
