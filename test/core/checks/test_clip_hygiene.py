"""Check clip-hygiene: five rules on a clip's samples.

- floor: no body vertex under y 0, less `FLOOR_OF_EFFECTOR` times the effector radius;
- slide: a standing clip's planted support does not move across the floor between samples;
- stand: a standing clip's supports stay down (`SKIN_OF_EFFECTOR` times the support's radius)
  outside the windows where the clip lifts them;
- pop: a loop's last sample matches its start sample;
- prop: the clip uses a carried prop only while it is shown, inside its `show_frames` windows.

The scene is the pack `kit`: a `Doll` rigged with a root, a body, a right leg whose chain starts at
the root (its group `foot.R` is the rig's one support) and a right arm whose chain starts at the
body (so `hand.R` is no support), and a `Rod` the doll carries. The clip `walk` is a `stand` clip
with six samples, frames 1 to 6. The body is a box well above the floor; the support rests at y 0
and never moves; the rod is shown inside frames 2 to 4 and a hold on it is used over the same
frames.
"""

import dataclasses
import json
import re
import tempfile
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate, sight
from model_kit.core.allow import Allowlist
from model_kit.core.checks.clip_hygiene import ClipHygiene
from model_kit.core.manifest import (
    Clip,
    ClipHold,
    FloorGear,
    Lift,
    Pack,
    Prop,
    Ref,
    Target,
    load_pack,
)
from model_kit.core.manifest.records import window_problem
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.rig import RigData
from model_kit.core.scene import ClipSamples, SceneCtx, due, rig_spec
from model_kit.core.supports import support_radius
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    FLOOR_OF_EFFECTOR,
    POP_OF_EFFECTOR,
    SKIN_OF_EFFECTOR,
    SLIDE_OF_EFFECTOR,
)
from model_kit.core.units import Style

from ..bodies.bar03 import bar03_scene
from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..shapes import box
from .piece_scenes import caught, chosen, tried_and_caught
from .sight_scenes import viewed

REGRESSIONS = Path(__file__).resolve().parents[1] / "fixtures" / "regressions"
RADIUS = 0.0586
FLOOR = FLOOR_OF_EFFECTOR * RADIUS
SLIDE = SLIDE_OF_EFFECTOR * RADIUS
POP = POP_OF_EFFECTOR * RADIUS
BAND = SKIN_OF_EFFECTOR * RADIUS  # how high a standing support's lowest point may be
STYLE = Style(units_per_metre=1.0, page_camera={}, dials={})
TIMES = np.arange(1.0, 7.0)
N = len(TIMES)
SHOWN = ((2.0, 4.0),)  # the rod's windows
WHOLE_CLIP = [("foot.R", ((1.0, 6.0),))]  # a lift for the support over every sample
FOOT_AT = np.array([0.3, 0.0, 0.0])

# --- the pack -------------------------------------------------------------------------------


def _piece(parts, **fields):
    piece = {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": 0.002}],
        "parts": [{"name": p, "listed": "stock", "shape": "box", "on": None, "group": "g",
                   "role": "solid"} for p in parts],
        "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {},
    }
    return {**piece, **fields}


def _chain(root, bones, effector, group, radius=RADIUS):
    return {"root": root, "bones": bones, "effector": effector, "effector_radius": radius,
            "dofs": {b: "xyz" for b in bones}, "limits": {}, "effector_group": group}


@cache
def kit_pack(foot_radius: float = RADIUS) -> Pack:
    doll = _piece(["torso"])
    doll["rig"] = {
        "armature": "TestRig",
        "bones": [{"name": "root", "parent": None, "length": 0.1},
                  {"name": "body", "parent": "root", "length": 0.3},
                  {"name": "leg.R", "parent": "root", "length": 0.3},
                  {"name": "foot.R", "parent": "leg.R", "length": 0.1},
                  {"name": "arm.R", "parent": "body", "length": 0.2},
                  {"name": "hand.R", "parent": "arm.R", "length": 0.1}],
        "bind": {"torso": {"bones": ["body"], "mode": "rigid"}},
        "contact_groups": {"foot.R": ["foot.R"], "hand.R": ["hand.R"]},
        "chains": {"leg.R": _chain("root", ["leg.R", "foot.R"], "foot.R", "foot.R", foot_radius),
                   "arm.R": _chain("body", ["arm.R", "hand.R"], "hand.R", "hand.R")},
        "sit": {"below_seat": 0.08, "forward": 0.03},
        "mirror": {"pairs": [], "axis": "x"},
        "supports": ["foot.R"],
    }
    rod = _piece(["shaft"], uses=[{"id": "front", "point": [0.0, 0.0, 0.3],
                                   "facing": [0.0, 0.0, -1.0], "posture": "stand"}],
                 holds=[{"id": "grip_R", "use": "front", "side": "R", "surface": ["shaft"],
                         "point": [0.0, 0.0, 0.02], "off": [0.0, 1.0, 0.0], "grip": True}])
    data = {"pack": "kit", "units_per_metre": 1.0, "pieces": {
        "Doll": doll, "Rod": rod, "Throne": _piece(["cushion"])}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


def rest(x, y, z):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


RIG = RigData(
    order=("root", "body", "leg.R", "foot.R", "arm.R", "hand.R"),
    parent={"root": None, "body": "root", "leg.R": "root", "foot.R": "leg.R", "arm.R": "body",
            "hand.R": "arm.R"},
    rest={b: np.eye(4) for b in ("root", "body", "leg.R", "foot.R", "arm.R", "hand.R")},
    world=np.eye(4), effector_mid={"foot.R": np.zeros(3), "hand.R": np.zeros(3)})

TORSO = box((-0.1, 0.1, -0.05), (0.1, 0.5, 0.05)).reshape(-1, 3)  # 36 corners, all above y 0.1
FOOT = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.05, 0.05, 0.05]]) + FOOT_AT
HAND = np.array([[0.0, 0.4, 0.0], [0.05, 0.4, 0.0]]) + (0.3, 0.0, 0.0)


def rod_pose(s):
    """The rod's pose: unit scale, moving a little with the sample."""
    return rest(0.5, 0.2 + 0.01 * s, 0.0)


def make_ctx(*, loop=False, loop_from=None, seat=False, rod=True, shells=True, groups=True,
             rig=True, samples=True, seat_when=None, show=SHOWN, use=((2.0, 4.0),), holder="ref",
             targets=(), foot_radius=RADIUS, posture="auto", lifts=()) -> SceneCtx:
    """The clip `walk`. The rod, when carried, is shown over `show` and used by a hold over `use`
    (`holder` says how the hold names it: its `ref` is on the rod, or the rod is its `who`).
    `targets` are (key, how) pairs, how `ref` (a target on the rod) or `tool` (the rod is the tool).
    `posture` is "sit" for a seated clip and "stand" otherwise, unless given (None for a clip
    written before posture); `lifts` are (support, windows) pairs."""
    props = {}
    poses = {}
    holds = ()
    if rod:
        props["rod"] = Prop("rod", "Rod", "kit", "hand.R", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                            show)
        poses["rod"] = np.array([rod_pose(s) for s in range(N)])
        if use:
            holds = (ClipHold(
                ref=Ref.parse("kit:Rod.grip_R" if holder == "ref" else "rig:hand.R"),
                who="hand.R" if holder == "ref" else "rod", who_box=None, when=use, budget=None,
                dangle=None, cross=None),)
    aims = tuple(Target(
        key=key, chain="arm.R", ref=Ref.parse("kit:Rod.grip_R" if how == "ref" else "rig:hand.R"),
        offset=(0.0, 0.0, 0.0), why=None, baked=None, tool="rod" if how == "tool" else None)
        for key, how in targets)
    clip = Clip(name="walk", frames=(1, 6), fps=24.0, authoring="solved" if aims else "keyed",
                seat=Ref.parse("kit:Throne.sit") if seat else None, holds=holds, targets=aims,
                floor_gear=FloorGear({}), props=props, mirror_of=None, loop=loop,
                loop_from=loop_from, seat_when=seat_when,
                posture=("sit" if seat else "stand") if posture == "auto" else posture,
                lifts=tuple(Lift(support, windows, "a test lift") for support, windows in lifts))
    body = {"torso": np.tile(TORSO.astype(np.float32), (N, 1, 1))} if shells else {}
    grouped = ({"foot.R": np.tile(FOOT, (N, 1, 1)), "hand.R": np.tile(HAND, (N, 1, 1))}
               if groups else {})
    clip_samples = ClipSamples(clip="walk", times=TIMES.copy(), groups=grouped, bones={},
                               props=poses, shells=body)
    pack = kit_pack(foot_radius)
    return SceneCtx(style=STYLE, packs={"kit": pack}, clips={"walk": clip}, pieces={},
                    samples={"walk": clip_samples} if samples else {},
                    rigs={"kit:Doll": RIG} if rig else {},
                    rig_of_clip={"walk": "kit:Doll"} if rig else {})


def edit(ctx, **arrays):
    """The scene with `samples` fields replaced by `arrays` (name -> new value)."""
    return dataclasses.replace(ctx, samples={"walk": dataclasses.replace(ctx.samples["walk"],
                                                                         **arrays)})


def sink(ctx, y, s=2, vertex=0):
    """The scene with one torso vertex of sample `s` moved to height `y`."""
    torso = np.array(ctx.samples["walk"].shells["torso"])
    torso[s, vertex, 1] = y
    return edit(ctx, shells={"torso": torso})


def slide_foot(ctx, dx, s=3, y=None):
    """The scene with the foot group of sample `s` moved `dx` along x (and to height `y`)."""
    foot = np.array(ctx.samples["walk"].groups["foot.R"])
    foot[s, :, 0] += dx
    if y is not None:
        foot[s, :, 1] += y
    return edit(ctx, groups={**ctx.samples["walk"].groups, "foot.R": foot})


def pop_last(ctx, dx, s=-1, vertex=0):
    """The scene with one torso vertex of sample `s` moved `dx` along x."""
    torso = np.array(ctx.samples["walk"].shells["torso"])
    torso[s, vertex, 0] += dx
    return edit(ctx, shells={"torso": torso})


def findings(ctx, only=None):
    return ClipHygiene().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def plants_caught(check, ctx) -> list[str]:
    return sorted(p.item for p in chosen(check, ctx))


# --- registration -------------------------------------------------------------------------------


def test_the_check_is_registered_and_is_not_a_piece_check():
    assert gate.CHECKS["clip-hygiene"] is ClipHygiene
    assert ClipHygiene.name == "clip-hygiene"
    assert ClipHygiene.kinds == frozenset()


def test_a_clean_clip_passes_and_every_rules_plant_is_caught():
    ctx = make_ctx(loop=True)
    assert findings(ctx) == []
    assert plants_caught(ClipHygiene(), ctx) == [
        "walk floor", "walk pop", "walk prop rod", "walk slide foot.R", "walk stand foot.R"]
    verdict = run_gate([ClipHygiene()], ctx, None)
    assert verdict.status == "passed" and verdict.coverage == []


# --- floor ----------------------------------------------------------------------------------------


def test_a_vertex_just_under_the_floor_passes_and_one_well_under_fails():
    assert items(sink(make_ctx(), -CALIBRATION_PASS * FLOOR)) == []
    found = findings(sink(make_ctx(), -CALIBRATION_FAIL * FLOOR))
    assert [f.item for f in found] == ["walk floor"]
    assert found[0].check == "clip-hygiene"


def test_the_floor_limit_is_the_factor_times_the_radius():
    assert FLOOR == pytest.approx(0.01)
    assert items(sink(make_ctx(), -FLOOR * 0.99)) == []
    assert items(sink(make_ctx(), -FLOOR * 1.01)) == ["walk floor"]


def test_a_vertex_above_the_floor_never_counts():
    assert items(sink(make_ctx(), 0.0)) == []
    assert items(sink(make_ctx(), 5.0)) == []


def test_the_floor_finding_names_the_worst_vertex_and_its_frame():
    ctx = sink(sink(make_ctx(), -0.05, s=1, vertex=3), -0.2, s=4, vertex=7)
    (found,) = findings(ctx)
    assert "-0.2000" in found.detail and "frame 5" in found.detail


def test_the_floor_limit_follows_the_smallest_effector_radius():
    # a rig with a smaller hand radius has the strictest floor, as overlap's skin does
    pack = kit_pack()
    doll = pack.pieces["Doll"]
    chains = {**doll.rig.chains, "arm.R": dataclasses.replace(doll.rig.chains["arm.R"],
                                                              effector_radius=RADIUS / 2)}
    small = dataclasses.replace(pack, pieces={**pack.pieces, "Doll": dataclasses.replace(
        doll, rig=dataclasses.replace(doll.rig, chains=chains))})
    ctx = dataclasses.replace(sink(make_ctx(), -0.75 * FLOOR), packs={"kit": small})
    assert items(ctx) == ["walk floor"]
    assert items(sink(make_ctx(), -0.75 * FLOOR)) == []


def test_a_sample_filter_reads_only_those_samples():
    ctx = sink(make_ctx(), -0.2, s=4)
    assert ClipHygiene().run(ctx, samples=frozenset({0, 1})) == []
    assert len(ClipHygiene().run(ctx, samples=frozenset({4}))) == 1


# --- slide ----------------------------------------------------------------------------------------


def test_a_planted_foot_that_creeps_passes_and_one_that_slides_fails():
    assert items(slide_foot(make_ctx(), CALIBRATION_PASS * SLIDE)) == []
    found = findings(slide_foot(make_ctx(), CALIBRATION_FAIL * SLIDE))
    assert [f.item for f in found] == ["walk slide foot.R"]


def test_the_slide_limit_is_the_factor_times_the_radius():
    assert SLIDE == pytest.approx(0.02)  # a length measured on the reference rig
    assert items(slide_foot(make_ctx(), SLIDE * 0.99, s=5)) == []
    assert items(slide_foot(make_ctx(), SLIDE * 1.01, s=5)) == ["walk slide foot.R"]


def test_the_slide_limit_is_the_supports_own_radius():
    # The support's radius is twice the hand's: its slide limit is twice the reference 0.02, and
    # a slide the reference radius would fail passes. (Floor and pop keep the smallest radius.)
    big = make_ctx(foot_radius=2 * RADIUS)
    assert items(slide_foot(big, 2 * SLIDE * 0.99, s=5)) == []
    assert items(slide_foot(big, 2 * SLIDE * 1.01, s=5)) == ["walk slide foot.R"]
    assert items(slide_foot(make_ctx(), 1.5 * SLIDE, s=5)) == ["walk slide foot.R"]
    # a support smaller than the hand has the strictest slide limit, its own
    small = make_ctx(foot_radius=RADIUS / 2)
    assert items(slide_foot(small, SLIDE * 0.49, s=5)) == []
    assert items(slide_foot(small, SLIDE * 0.51, s=5)) == ["walk slide foot.R"]
    assert "walk slide foot.R" in plants_caught(ClipHygiene(), big)
    assert "walk slide foot.R" in plants_caught(ClipHygiene(), small)


def test_slide_is_the_floor_plane_distance_not_the_height():
    # a foot that rises (still planted) by 0.8 floor tolerances has not slid; the clip lifts it all
    # along, so the stand rule has nothing to say about the height
    assert items(slide_foot(make_ctx(lifts=WHOLE_CLIP), 0.0, y=0.8 * FLOOR)) == []
    # and a slide along z counts as much as one along x
    ctx = make_ctx()
    foot = np.array(ctx.samples["walk"].groups["foot.R"])
    foot[3, :, 2] += CALIBRATION_FAIL * SLIDE
    assert items(edit(ctx, groups={**ctx.samples["walk"].groups, "foot.R": foot})) == [
        "walk slide foot.R"]


def test_a_foot_lifted_off_the_floor_is_not_planted():
    # raised past the floor tolerance for one sample, it may move: neither pair is planted
    assert items(slide_foot(make_ctx(lifts=WHOLE_CLIP), 10 * SLIDE, y=2 * FLOOR)) == []


def test_the_lowest_point_of_the_group_is_the_one_followed():
    ctx = make_ctx()
    foot = np.array(ctx.samples["walk"].groups["foot.R"])
    foot[3, 3, 0] += 5 * SLIDE  # the highest point of the group, not the planted one
    assert items(edit(ctx, groups={**ctx.samples["walk"].groups, "foot.R": foot})) == []


def with_supports(ctx, supports):
    """The scene with the doll's rig naming `supports`."""
    pack = ctx.packs["kit"]
    doll = pack.pieces["Doll"]
    rig = dataclasses.replace(doll.rig, supports=tuple(supports))
    doll = dataclasses.replace(doll, rig=rig)
    return dataclasses.replace(ctx, packs={"kit": dataclasses.replace(
        pack, pieces={**pack.pieces, "Doll": doll})})


def test_supports_replace_top_bone_chains():
    """The rig's `supports` are the slide rule's groups. A chain that starts at the top bone is
    no foot unless the rig lists it; a group that no top-bone chain ends in is one if it does."""
    ctx = make_ctx()
    hand = np.array(ctx.samples["walk"].groups["hand.R"])
    hand[:, :, 1] = 0.0  # on the floor
    hand[3, :, 0] += 10 * SLIDE
    hand_slides = edit(ctx, groups={**ctx.samples["walk"].groups, "hand.R": hand})
    assert items(hand_slides) == []  # the hand is no support
    # name the hand a support and its slide counts; the leg chain (top bone) is no foot now
    by_hand = with_supports(hand_slides, ["hand.R"])
    assert items(by_hand) == ["walk slide hand.R"]
    foot_slides = with_supports(slide_foot(ctx, 10 * SLIDE), ["hand.R"])
    assert "walk slide foot.R" not in items(foot_slides)
    assert not any(x.startswith("walk slide foot") for x in ClipHygiene().coverage(foot_slides))
    # a rig that names no supports has no slide rule and says so without naming a top bone
    bare = with_supports(slide_foot(ctx, 10 * SLIDE), [])
    assert items(bare) == []
    (line,) = [x for x in ClipHygiene().coverage(bare) if "walk slide" in x]
    assert "not applicable" in line and "names no supports" in line
    assert not isinstance(line, Unmeasured) and "top bone" not in line


def test_quadruped_has_four_supports_and_no_tail_foot():
    ctx = quadruped_scene()
    spec = rig_spec(ctx, "stand")
    assert len(spec.supports) == 4 and "tail_tip" not in spec.supports
    # A guess would take the effector group of every chain that starts at a parentless bone, and
    # the tail's chain starts at the pelvis, the rig's only parentless bone: a tail tip "foot".
    top = {b.name for b in spec.bones if b.parent is None}
    guessed = {c.effector_group for c in spec.chains.values() if c.root in top}
    assert "tail_tip" in guessed and len(guessed) == 5
    check = ClipHygiene()
    assert check.run(ctx) == []
    slides = sorted(p.item for p in chosen(check, ctx) if " slide " in p.item)
    assert slides == sorted(f"stand slide {g}" for g in spec.supports)
    assert not any("tail" in x for x in [p.item for p in chosen(check, ctx)] + check.coverage(ctx))


def test_each_support_has_its_own_radius():
    ctx = make_ctx(foot_radius=2 * RADIUS)
    spec = ctx.packs["kit"].pieces["Doll"].rig
    assert support_radius(spec, None, "foot.R") == pytest.approx(2 * RADIUS)  # its chain's
    both = with_supports(ctx, ["foot.R", "hand.R"])
    assert support_radius(both.packs["kit"].pieces["Doll"].rig, None, "hand.R") == RADIUS


def test_a_seated_clip_is_not_a_standing_clip():
    ctx = slide_foot(make_ctx(seat=True), 10 * SLIDE)
    assert items(ctx) == []
    lines = ClipHygiene().coverage(ctx)
    assert any("not applicable" in line and "seat" in line and "walk slide" in line
               for line in lines)
    assert not any(isinstance(line, Unmeasured) for line in lines)


def test_a_clip_seated_only_in_its_seat_windows_stands_outside_them():
    # frames 1..6; the clip is seated at frame 6 only, so frame 4 (sample 3) is a standing sample
    seated_late = make_ctx(seat=True, seat_when=((6.0, 6.0),))
    slid = slide_foot(seated_late, 10 * SLIDE, s=3)
    assert items(slid) == ["walk slide foot.R"]
    assert items(seated_late) == []
    assert not any("seated clip" in line for line in ClipHygiene().coverage(seated_late))
    assert "walk slide foot.R" in plants_caught(ClipHygiene(), seated_late)


def test_a_slide_into_a_seated_sample_is_not_a_standing_pair():
    # the slide is at sample 5 (frame 6), the seated one: its pair (4, 5) is not both standing
    ctx = slide_foot(make_ctx(seat=True, seat_when=((6.0, 6.0),)), 10 * SLIDE, s=5)
    assert items(ctx) == []


def test_a_foot_that_never_plants_is_a_plain_not_applicable_line():
    ctx = make_ctx(lifts=WHOLE_CLIP)
    foot = np.array(ctx.samples["walk"].groups["foot.R"])
    foot[:, :, 1] += 1.0
    ctx = edit(ctx, groups={**ctx.samples["walk"].groups, "foot.R": foot})
    assert items(ctx) == []
    (line,) = [x for x in ClipHygiene().coverage(ctx) if "slide" in x]
    assert "not applicable" in line and "planted" in line and not isinstance(line, Unmeasured)


def test_the_slide_plant_moves_one_planted_sample_by_the_failing_multiple():
    ctx = make_ctx()
    plants = [p for p in chosen(ClipHygiene(), ctx) if p.item == "walk slide foot.R"]
    assert len(plants) == 1
    before = ctx.samples["walk"].groups["foot.R"]
    after = plants[0].ctx.samples["walk"].groups["foot.R"]
    moved = np.linalg.norm((after - before)[..., [0, 2]], axis=2)
    (s,) = plants[0].samples
    assert np.all(moved[s] == pytest.approx(CALIBRATION_FAIL * SLIDE))
    assert np.all(np.delete(moved, s, axis=0) == 0.0)
    assert np.array_equal(after[..., 1], before[..., 1])  # still on the floor


def test_a_plant_moves_away_from_a_slide_already_there():
    # the foot already crept one way; the plant must not cancel it
    ctx = slide_foot(make_ctx(), 0.9 * SLIDE, s=3)
    assert "walk slide foot.R" in plants_caught(ClipHygiene(), ctx)


# --- stand ----------------------------------------------------------------------------------------


def lift_support(ctx, clip, group, dy, s):
    """The scene with the points of `group` in sample `s` of `clip` raised by `dy`."""
    points = np.array(ctx.samples[clip].groups[group], dtype=float)
    points[s, :, 1] += dy
    moved = dataclasses.replace(ctx.samples[clip], groups={**ctx.samples[clip].groups,
                                                           group: points})
    return dataclasses.replace(ctx, samples={**ctx.samples, clip: moved})


def stand_items(ctx):
    return sorted(f.item for f in ClipHygiene().run(ctx) if " stand " in f.item)


def test_a_standing_support_off_the_floor_by_more_than_the_skin_band_fails():
    assert BAND == pytest.approx(SKIN_OF_EFFECTOR * RADIUS)
    assert BAND == pytest.approx(0.002)  # mesh noise, in the reference rig's lengths
    ctx = make_ctx()
    assert stand_items(lift_support(ctx, "walk", "foot.R", BAND * 0.99, 2)) == []
    (found,) = findings(lift_support(ctx, "walk", "foot.R", BAND * 1.01, 2))
    assert found.item == "walk stand foot.R" and "frame 3" in found.detail
    assert f"{BAND:.4f}" in found.detail
    # the planted-foot band (FLOOR_OF_EFFECTOR) is far wider: a lift it would call planted fails
    assert FLOOR_OF_EFFECTOR > 4 * SKIN_OF_EFFECTOR
    assert stand_items(lift_support(ctx, "walk", "foot.R", 0.5 * FLOOR, 2)) == ["walk stand foot.R"]


def test_the_band_is_the_supports_own_radius():
    big = make_ctx(foot_radius=2 * RADIUS)  # the support is twice the hand: twice the band
    assert stand_items(lift_support(big, "walk", "foot.R", 2 * BAND * 0.99, 2)) == []
    assert stand_items(lift_support(big, "walk", "foot.R", 2 * BAND * 1.01, 2)) == [
        "walk stand foot.R"]
    small = make_ctx(foot_radius=RADIUS / 2)
    assert stand_items(lift_support(small, "walk", "foot.R", BAND * 0.49, 2)) == []
    assert stand_items(lift_support(small, "walk", "foot.R", BAND * 0.51, 2)) == [
        "walk stand foot.R"]


def test_the_worst_sample_decides_and_a_sunk_support_is_left_to_the_floor_rule():
    ctx = lift_support(lift_support(make_ctx(), "walk", "foot.R", 3 * BAND, 1),
                       "walk", "foot.R", 5 * BAND, 4)
    (found,) = findings(ctx)
    assert "frame 5" in found.detail and f"{5 * BAND:.4f}" in found.detail
    assert stand_items(lift_support(make_ctx(), "walk", "foot.R", -5 * BAND, 2)) == []


@pytest.mark.parametrize("body, clip", [(humanoid_scene, "stand"), (quadruped_scene, "stand")])
def test_stand_rule_catches_a_lifted_support(body, clip):
    ctx = body()
    spec = rig_spec(ctx, clip)
    assert spec.supports and ClipHygiene().run(ctx) == []
    for group in spec.supports:
        radius = support_radius(spec, ctx.rests[ctx.rig_of_clip[clip]], group)
        band = SKIN_OF_EFFECTOR * radius
        item = f"{clip} stand {group}"
        assert stand_items(lift_support(ctx, clip, group, band * 0.9, 3)) == []
        lifted = lift_support(ctx, clip, group, band * CALIBRATION_FAIL, 3)
        assert stand_items(lifted) == [item]
        # the plant is exactly this fault, at the first judged sample
        (plant,) = [p for p in chosen(ClipHygiene(), ctx) if p.item == item]
        assert plant.samples == frozenset({0})
        assert [f.item for f in ClipHygiene().run(plant.ctx, only=plant.scope,
                                                  samples=plant.samples)] == [item]


def test_lift_window_excuses_a_step():
    ctx = humanoid_scene()
    clip = ctx.clips["step"]
    assert ClipHygiene().run(ctx) == []
    (lift,) = clip.lifts
    assert lift.support == "foot.L" and lift.why
    # the same samples with the window taken away: the left foot hovers
    bare = dataclasses.replace(ctx, clips={**ctx.clips, "step": dataclasses.replace(
        clip, lifts=())})
    found = [f for f in ClipHygiene().run(bare) if " stand " in f.item]
    assert [f.item for f in found] == ["step stand foot.L"]
    assert "frame 8" in found[0].detail  # the first of the held samples
    # a window that stops short leaves the later lifted samples judged
    (t0, t1), = lift.when
    short = dataclasses.replace(clip, lifts=(dataclasses.replace(lift, when=((t0, t1 - 4),)),))
    cut = dataclasses.replace(ctx, clips={**ctx.clips, "step": short})
    assert stand_items(cut) == ["step stand foot.L"]
    # the other foot gets no excuse from the left foot's window
    right = lift_support(ctx, "step", "foot.R", 5 * BAND, 10)
    assert stand_items(right) == ["step stand foot.R"]


def test_a_lift_excuses_only_its_own_support():
    ctx = make_ctx(lifts=[("foot.R", ((2.0, 4.0),))])
    assert stand_items(lift_support(ctx, "walk", "foot.R", 5 * BAND, 2)) == []  # frame 3, inside
    assert stand_items(lift_support(ctx, "walk", "foot.R", 5 * BAND, 4)) == ["walk stand foot.R"]
    # the plant picks the first sample outside the window (frame 1), never an excused one
    (plant,) = [p for p in chosen(ClipHygiene(), ctx) if p.item == "walk stand foot.R"]
    assert plant.samples == frozenset({0})
    late = make_ctx(lifts=[("foot.R", ((1.0, 2.0),))])
    (plant,) = [p for p in chosen(ClipHygiene(), late) if p.item == "walk stand foot.R"]
    assert plant.samples == frozenset({2})  # frame 3
    assert [f.item for f in ClipHygiene().run(plant.ctx, only=plant.scope,
                                              samples=plant.samples)] == ["walk stand foot.R"]


def test_a_support_lifted_at_every_sample_has_no_stand_rule_to_run():
    ctx = make_ctx(lifts=WHOLE_CLIP)
    (line,) = [x for x in ClipHygiene().coverage(ctx) if "walk stand" in x]
    assert "not applicable" in line and "lift window" in line
    assert not isinstance(line, Unmeasured)
    assert not any(p.item == "walk stand foot.R" for p in chosen(ClipHygiene(), ctx))


def test_d9_idle_before_fix_fails():
    """A chibi idle in which the feet lift: the breath rises the whole body, and each foot's
    lowest point with it, 0.0060 scene units at frame 37 (the chibi is 0.747 units per metre)."""
    z = np.load(REGRESSIONS / "idle_feet.npz", allow_pickle=False)
    ctx = bar03_scene()
    spec = ctx.packs["chibi"].pieces["chibi"].rig
    assert spec.supports == ("ankle.L", "ankle.R")
    old = dataclasses.replace(
        ctx.samples["idle"], times=z["times"].astype(float), shells={}, faces={}, labels={},
        groups={g: z[f"g:{g}"].astype(float) for g in spec.supports},
        bones={k[2:]: z[k] for k in z.files if k.startswith("b:")})
    before = dataclasses.replace(ctx, samples={"idle": old})
    band = SKIN_OF_EFFECTOR * support_radius(spec, ctx.rests["chibi:chibi"], "ankle.L")
    found = ClipHygiene().run(before)
    assert sorted(f.item for f in found if " stand " in f.item) == [
        "idle stand ankle.L", "idle stand ankle.R"]
    for f in found:
        assert "frame 37" in f.detail  # the top of the breath
    low = old.groups["ankle.L"][:, :, 1].min(axis=1)
    assert low.max() - low[0] == pytest.approx(0.0060, abs=1e-5)  # the lift the defect measured
    assert low[0] < band < low.max()  # at rest the foot is inside the band; the lift is out of it
    assert (low.max() - low[0]) > band  # the lift alone is over the band
    # the fixed idle (the bar's scene now) holds each foot at one height, inside the band
    assert ClipHygiene().run(ctx) == []
    now = ctx.samples["idle"].groups["ankle.L"][:, :, 1].min(axis=1)
    assert now.max() < band


def test_the_stand_rule_plants_are_caught_on_the_chibi_idle():
    ctx = bar03_scene()
    plants = {p.item: p for p in chosen(ClipHygiene(), ctx)}
    assert {"idle stand ankle.L", "idle stand ankle.R"} <= set(plants)
    plant = plants["idle stand ankle.L"]
    assert [f.item for f in ClipHygiene().run(plant.ctx, only=plant.scope, samples=plant.samples)
            ] == ["idle stand ankle.L"]


def test_sit_clip_skips_stand():
    ctx = humanoid_scene()
    lines = [x for x in ClipHygiene().coverage(ctx) if "sit_chair stand" in x]
    assert len(lines) == 1 and "not applicable" in lines[0] and "sit clip" in lines[0]
    assert not isinstance(lines[0], Unmeasured)
    assert not any(p.item.startswith("sit_chair stand") for p in chosen(ClipHygiene(), ctx))
    # a sitter's feet may hang: lift them far and the stand rule still says nothing
    seated = lift_support(ctx, "sit_chair", "foot.L", 1.0, 3)
    assert not [f for f in ClipHygiene().run(seated) if " stand " in f.item]


@pytest.mark.parametrize("posture", ["lie", "free"])
def test_lie_and_free_clips_skip_stand(posture):
    ctx = make_ctx(posture=posture)
    (line,) = [x for x in ClipHygiene().coverage(ctx) if "walk stand" in x]
    assert "not applicable" in line and f"{posture} clip" in line
    assert not isinstance(line, Unmeasured)
    assert stand_items(lift_support(ctx, "walk", "foot.R", 1.0, 2)) == []


def test_stand_clip_without_supports_is_partial():
    ctx = with_supports(make_ctx(), [])
    (line,) = [x for x in unmeasured(ctx) if "stand" in x]
    assert isinstance(line, Unmeasured) and "walk stand" in line and "names no supports" in line
    verdict = run_gate([ClipHygiene()], ctx, None)
    assert verdict.status == "partial" and verdict.failed == 0


def test_a_clip_with_no_posture_is_unmeasured_for_stand_and_partial():
    ctx = make_ctx(posture=None)
    (line,) = unmeasured(ctx)
    assert "walk stand" in line and "no posture" in line
    assert run_gate([ClipHygiene()], ctx, None).status == "partial"
    assert [p.item for p in chosen(ClipHygiene(), ctx) if " stand " in p.item] == []
    assert items(slide_foot(ctx, 10 * SLIDE)) == ["walk slide foot.R"]  # slide still runs


# --- pop ------------------------------------------------------------------------------------------


def test_a_loop_whose_last_sample_matches_its_first_passes():
    assert items(make_ctx(loop=True)) == []


def test_a_loop_that_pops_fails():
    assert items(pop_last(make_ctx(loop=True), CALIBRATION_PASS * POP)) == []
    (found,) = findings(pop_last(make_ctx(loop=True), CALIBRATION_FAIL * POP))
    assert found.item == "walk pop"
    assert f"{CALIBRATION_FAIL * POP:.4f}" in found.detail and f"{POP:.4f}" in found.detail


def test_the_pop_limit_is_the_factor_times_the_radius():
    assert POP == pytest.approx(0.01)
    assert items(pop_last(make_ctx(loop=True), POP * 0.99)) == []
    assert items(pop_last(make_ctx(loop=True), POP * 1.01)) == ["walk pop"]


def test_the_worst_vertex_decides_the_pop():
    ctx = pop_last(pop_last(make_ctx(loop=True), 0.9 * POP, vertex=1), 0.9 * POP, vertex=5)
    assert items(ctx) == []  # two vertices near the limit, none over it
    assert items(pop_last(ctx, 0.3 * POP, vertex=5)) == ["walk pop"]


def test_a_loop_is_compared_with_its_loop_from_sample():
    ctx = make_ctx(loop=True, loop_from=3)  # frame 3 is sample 2
    moved = np.array(ctx.samples["walk"].shells["torso"])
    moved[2, :, 0] += 0.3  # the start is somewhere else than the first sample
    moved[5] = moved[2]  # the last sample matches the start
    matched = edit(ctx, shells={"torso": moved})
    assert items(matched) == []
    moved[5] = moved[0]  # and now it matches the first sample instead
    assert items(edit(ctx, shells={"torso": moved})) == ["walk pop"]


def test_a_clip_that_does_not_loop_has_no_pop_rule():
    ctx = pop_last(make_ctx(loop=False), 10 * POP)
    assert items(ctx) == []
    lines = ClipHygiene().coverage(ctx)
    assert any("not applicable" in x and "no loop" in x and "walk pop" in x for x in lines)
    assert not any(isinstance(x, Unmeasured) for x in lines)


def test_the_pop_plant_breaks_the_last_sample_by_the_failing_multiple():
    ctx = make_ctx(loop=True)
    (plant,) = [p for p in chosen(ClipHygiene(), ctx) if p.item == "walk pop"]
    assert plant.samples == frozenset({N - 1})
    before = ctx.samples["walk"].shells["torso"]
    after = plant.ctx.samples["walk"].shells["torso"]
    moved = np.linalg.norm((after - before).astype(float), axis=2)
    assert moved[-1].max() == pytest.approx(CALIBRATION_FAIL * POP, rel=1e-4)
    assert np.all(moved[:-1] == 0.0)


# --- prop windows ---------------------------------------------------------------------------------


def narrowed(ctx, show):
    """The scene with the rod's show windows replaced."""
    clip = ctx.clips["walk"]
    rod = dataclasses.replace(clip.props["rod"], show_frames=show)
    return dataclasses.replace(ctx, clips={"walk": dataclasses.replace(clip, props={"rod": rod})})


def test_a_prop_used_only_while_shown_passes():
    assert items(make_ctx()) == []


def test_a_prop_used_one_sample_outside_its_windows_fails():
    found = findings(make_ctx(use=((2.0, 5.0),)))
    assert [f.item for f in found] == ["walk prop rod"]
    assert found[0].check == "clip-hygiene"
    assert "frame 5" in found[0].detail and "1 sample" in found[0].detail


def test_the_detail_names_the_first_sample_outside_and_the_count():
    (found,) = findings(make_ctx(use=((1.0, 6.0),)))
    assert "frame 1" in found.detail and "3 samples" in found.detail  # 1, 5 and 6 are outside
    (found,) = findings(narrowed(make_ctx(), ((2.0, 2.0),)))
    assert "frame 3" in found.detail and "2 samples" in found.detail


def test_a_prop_that_is_shown_but_not_used_is_not_a_finding():
    # the rule is "used only while shown": being shown early or late is not a use
    assert items(make_ctx(show=((1.0, 6.0),))) == []
    assert items(make_ctx(use=((3.0, 3.0),))) == []


def test_a_window_edge_sample_counts_as_inside():
    assert items(make_ctx(show=((2.0, 4.0),))) == []
    assert items(narrowed(make_ctx(), ((2.5, 4.0),))) == ["walk prop rod"]  # frame 2 < 2.5
    assert items(narrowed(make_ctx(), ((2.0, 3.5),))) == ["walk prop rod"]  # frame 4 > 3.5


def test_a_prop_with_two_windows():
    ctx = make_ctx(show=((1.0, 2.0), (5.0, 6.0)), use=((1.0, 2.0), (5.0, 6.0)))
    assert items(ctx) == []
    assert items(narrowed(ctx, ((1.0, 2.0),))) == ["walk prop rod"]
    assert items(make_ctx(show=((1.0, 2.0), (5.0, 6.0)), use=((1.0, 3.0),))) == ["walk prop rod"]


@pytest.mark.parametrize("holder", ["ref", "who"])
def test_a_hold_names_the_prop_by_its_ref_or_its_who(holder):
    assert items(make_ctx(holder=holder)) == []
    assert items(make_ctx(holder=holder, use=((2.0, 5.0),))) == ["walk prop rod"]


def test_a_hold_on_another_piece_does_not_use_the_prop():
    ctx = make_ctx(use=((1.0, 6.0),), holder="who")
    clip = ctx.clips["walk"]
    other = dataclasses.replace(clip.holds[0], who="hand.R")  # a hand on the body, not the rod
    ctx = dataclasses.replace(ctx, clips={"walk": dataclasses.replace(clip, holds=(other,))})
    assert items(ctx) == []


def test_a_target_on_the_prop_uses_it_at_its_key():
    assert items(make_ctx(use=(), targets=[(3, "ref")])) == []
    found = findings(make_ctx(use=(), targets=[(3, "ref"), (6, "ref")]))
    assert [f.item for f in found] == ["walk prop rod"] and "frame 6" in found[0].detail


def test_a_tool_target_uses_the_prop_it_carries():
    assert items(make_ctx(use=(), targets=[(4, "tool")])) == []
    found = findings(make_ctx(use=(), targets=[(5, "tool")]))
    assert [f.item for f in found] == ["walk prop rod"] and "frame 5" in found[0].detail


def test_a_prop_the_clip_never_uses_is_a_plain_not_applicable_line():
    ctx = make_ctx(use=())
    assert items(ctx) == []
    lines = ClipHygiene().coverage(ctx)
    assert any("not applicable" in x and "not used by a hold, target or tool" in x
               and "walk prop rod" in x for x in lines)
    assert not any(isinstance(x, Unmeasured) for x in lines)
    assert not any(p.item == "walk prop rod" for p in chosen(ClipHygiene(), ctx))
    assert run_gate([ClipHygiene()], ctx, None).status == "passed"


def test_a_clip_with_no_props_has_no_prop_rule():
    lines = ClipHygiene().coverage(make_ctx(rod=False))
    assert any("not applicable" in x and "no props" in x for x in lines)


def test_the_prop_plant_narrows_the_window_so_a_using_sample_falls_outside():
    ctx = make_ctx()
    (plant,) = [p for p in chosen(ClipHygiene(), ctx) if p.item == "walk prop rod"]
    window = plant.ctx.clips["walk"].props["rod"].show_frames
    assert window != SHOWN
    assert any(not any(t0 <= TIMES[s] <= t1 for t0, t1 in window) for s in (1, 2, 3))  # 2, 3, 4
    assert ctx.clips["walk"].props["rod"].show_frames == SHOWN  # the real scene is untouched
    assert [f.item for f in findings(plant.ctx, only=plant.scope)] == ["walk prop rod"]


def test_the_prop_plant_works_when_the_first_sample_is_the_use():
    ctx = make_ctx(show=((1.0, 6.0),), use=((1.0, 1.0),))
    assert "walk prop rod" in plants_caught(ClipHygiene(), ctx)


# --- coverage: missing data is Unmeasured ----------------------------------------------------------------


def unmeasured(ctx):
    return [x for x in ClipHygiene().coverage(ctx) if isinstance(x, Unmeasured)]


def test_a_clip_with_no_samples_is_unmeasured_and_partial():
    ctx = make_ctx(samples=False)
    (line,) = ClipHygiene().coverage(ctx)
    assert isinstance(line, Unmeasured) and "no samples" in line and "walk" in line
    assert run_gate([ClipHygiene()], ctx, None).status == "partial"


def test_a_clip_with_no_rig_is_unmeasured():
    ctx = make_ctx(rig=False)
    (line,) = ClipHygiene().coverage(ctx)
    assert isinstance(line, Unmeasured) and "no rig" in line
    assert findings(ctx) == []


def test_a_clip_with_no_body_shells_leaves_floor_and_pop_unmeasured():
    ctx = make_ctx(loop=True, shells=False)
    lines = unmeasured(ctx)
    assert {x.split(": ")[-1] for x in lines} == {"walk floor", "walk pop"}
    assert all("no body shells" in x for x in lines)
    assert run_gate([ClipHygiene()], ctx, None).status == "partial"
    # slide and prop still run
    assert items(slide_foot(ctx, 10 * SLIDE)) == ["walk slide foot.R"]


def test_a_loop_clip_needs_shells_for_pop_but_a_plain_clip_does_not():
    assert not any("pop" in x for x in unmeasured(make_ctx(shells=False)))


def test_a_support_with_no_sampled_points_is_unmeasured():
    ctx = make_ctx(groups=False)
    slide, stand = unmeasured(ctx)
    assert "walk slide foot.R" in slide and "no points" in slide
    assert "walk stand foot.R" in stand and "no points" in stand


def test_a_clean_loop_scene_has_no_coverage_lines():
    assert ClipHygiene().coverage(make_ctx(loop=True)) == []


# --- the gate ------------------------------------------------------------------------------------------------


def test_a_real_fault_fails_the_run_and_one_allow_line_covers_the_clip_and_rule(tmp_path):
    ctx = sink(make_ctx(), -0.2)
    verdict = run_gate([ClipHygiene()], ctx, None)
    assert verdict.status == "failed" and verdict.failed == 1
    assert verdict.lines[0].startswith("clip-hygiene | walk floor | ")
    path = tmp_path / "allow.txt"
    path.write_text("clip-hygiene | walk floor | the doll dips on a bow\n")
    allow = Allowlist.load(path)
    assert run_gate([ClipHygiene()], ctx, allow).status == "passed"


@pytest.mark.parametrize("make, item", [
    (lambda: slide_foot(make_ctx(), 10 * SLIDE), "walk slide foot.R"),
    (lambda: pop_last(make_ctx(loop=True), 10 * POP), "walk pop"),
    (lambda: make_ctx(use=((2.0, 5.0),)), "walk prop rod"),
])
def test_a_real_slide_pop_or_prop_fault_fails_the_run(make, item):
    verdict = run_gate([ClipHygiene()], make(), None)
    assert verdict.status == "failed" and verdict.failed == 1
    assert verdict.lines[0].startswith(f"clip-hygiene | {item} | ")


def test_a_scoped_run_measures_only_the_named_items():
    ctx = pop_last(sink(make_ctx(loop=True), -0.2), 10 * POP)
    assert sorted(f.item for f in findings(ctx)) == ["walk floor", "walk pop"]
    assert [f.item for f in findings(ctx, only={"walk pop"})] == ["walk pop"]
    assert findings(ctx, only={"walk prop rod"}) == []


# --- the seen floor ---------------------------------------------------------------------------
# Floor, slide and pop are judged only as they will be seen: a finding whose measured size (the
# depth under y 0, the travel of the planted foot, the jump of the loop) is smaller than one
# arcminute at the rig piece's nearest viewing distance is dropped and counted. The limits do not
# move. SEEN is such a length in the test pack's units: more than the floor, slide and pop limits.

SEEN = 0.05

FAULTS = {
    "floor": (lambda size: sink(make_ctx(), -size), "walk floor"),
    "slide": (lambda size: slide_foot(make_ctx(), size, s=5), "walk slide foot.R"),
    "pop": (lambda size: pop_last(make_ctx(loop=True), size), "walk pop"),
}


def floor_counts(ctx):
    return {m.group(3): int(m.group(1)) for line in ClipHygiene().coverage(ctx)
            if (m := re.fullmatch(r"clip-hygiene: (\d+) under 1 arcmin at (\S+) m: (\S+)",
                                  str(line)))}


@pytest.mark.parametrize("rule", sorted(FAULTS))
def test_a_motion_just_over_the_floor_fires_and_one_just_under_is_dropped_and_counted(rule):
    make, item = FAULTS[rule]
    assert max(FLOOR, SLIDE, POP) < 0.5 * SEEN
    over = viewed(make(1.01 * SEEN), SEEN)
    assert items(over) == [item]
    assert floor_counts(over) == {"kit:Doll": 0}
    under = viewed(make(0.99 * SEEN), SEEN)
    assert items(under) == []
    assert floor_counts(under) == {"kit:Doll": 1}
    # the same motion is a finding with no view, and with a view that sees more than it
    assert items(make(0.99 * SEEN)) == [item]
    assert items(viewed(make(0.99 * SEEN), 0.5 * SEEN)) == [item]


def test_the_floor_never_loosens_a_motion_that_the_limit_alone_would_pass():
    ctx = viewed(sink(make_ctx(), -0.5 * FLOOR), SEEN)
    assert items(ctx) == []
    assert floor_counts(ctx) == {"kit:Doll": 0}  # it was no finding, so nothing was dropped


def test_stand_and_prop_are_not_judged_by_the_floor():
    hover = viewed(slide_foot(make_ctx(), 0.0, y=2 * BAND, s=2), 10 * BAND)
    assert items(hover) == ["walk stand foot.R"]  # a lift of 2 bands, under the seen floor
    prop = viewed(make_ctx(show=((4.0, 5.0),)), 10.0)
    assert items(prop) == ["walk prop rod"]


def test_a_piece_with_no_view_gets_no_floor_no_line_and_no_gap():
    for ctx in (sink(make_ctx(), -2 * FLOOR), slide_foot(make_ctx(), 2 * SLIDE, s=5),
                pop_last(make_ctx(loop=True), 2 * POP)):
        tiny = viewed(ctx, 1e-9)  # a floor under every motion: the view changes nothing
        assert items(ctx) == items(tiny) != []
        assert not any("arcmin" in x for x in ClipHygiene().coverage(ctx))
        assert [x for x in ClipHygiene().coverage(ctx) if isinstance(x, Unmeasured)] == [
            x for x in ClipHygiene().coverage(tiny) if isinstance(x, Unmeasured)]


def test_a_viewed_clean_clip_still_has_no_gap_in_its_coverage():
    ctx = viewed(make_ctx(loop=True), SEEN)
    distance = sight.seen_distance(SEEN, 1.0)
    line = f"clip-hygiene: 0 under 1 arcmin at {distance:g} m: kit:Doll"
    assert ClipHygiene().coverage(ctx) == [line]
    assert run_gate([ClipHygiene()], ctx, None).status == "passed"


def size_of(finding):
    return float(re.search(r"(?:y -|moved |is )(\d+\.\d+)", finding.detail).group(1))


def test_the_bar_03_body_drops_and_keeps_by_size_on_a_second_piece():
    """Bar 03's body: each rule's plant is a motion of twice its tolerance. A floor over that
    motion drops it, and a floor under it leaves it."""
    ctx = bar03_scene()
    check = ClipHygiene()
    visited = set()
    for plant in chosen(check, ctx):
        if not plant.item.endswith((" floor", " pop")) and " slide " not in plant.item:
            continue
        visited.add(plant.item)
        (found,) = [f for f in check.run(plant.ctx, only=plant.scope, samples=plant.samples)
                    if f.item == plant.item]
        size = size_of(found)
        over = viewed(plant.ctx, 0.9 * size)
        assert plant.item in {f.item for f in check.run(over, only=plant.scope)}, plant.item
        under = viewed(plant.ctx, 1.1 * size)
        assert plant.item not in {f.item for f in check.run(under, only=plant.scope)}, plant.item
        assert floor_counts(under)[next(iter(ctx.rig_of_clip.values()))] >= 1, plant.item
    # the loop above must not pass by visiting nothing
    assert any(i.endswith(" floor") for i in visited) and any(i.endswith(" pop") for i in visited)


def test_plants_stay_caught_under_a_floor_on_two_pieces():
    far = {"fixture": viewed(make_ctx(loop=True), SEEN), "bar 03": viewed(bar03_scene(), 0.2)}
    for label, ctx in far.items():
        planted = chosen(ClipHygiene(), ctx)
        kinds = {p.item.split(" ", 1)[1].split(" ")[0] for p in planted}
        assert {"floor", "slide", "pop"} <= kinds, label
        assert plants_caught(ClipHygiene(), ctx), label


# --- plants are units built blind to the rules ------------------------------------------


def test_the_units_are_tried_and_caught_on_two_pieces():
    for label, ctx in {"fixture": make_ctx(loop=True), "bar 03": bar03_scene()}.items():
        tried = tried_and_caught(ClipHygiene(), ctx)
        assert {"floor", "pop"} <= {i.rpartition(" ")[2] for i in tried}, label


def test_a_looped_clip_that_jumps_at_the_seam_fails_at_the_seam_sample():
    """The pop is judged between the last sample and the loop's start sample, and a
    finding names both frames. A sample in the middle of the clip is no seam."""
    for loop_from, start in ((None, 1.0), (3, 3.0)):
        ctx = pop_last(make_ctx(loop=True, loop_from=loop_from), 2 * POP)
        (found,) = [f for f in findings(ctx) if f.item == "walk pop"]
        assert f"the last sample (frame {TIMES[-1]:g}) is" in found.detail
        assert f"(frame {start:g})" in found.detail
        assert [f.item for f in ClipHygiene().run(ctx, samples=frozenset({N - 1}))] == ["walk pop"]
        assert ClipHygiene().run(ctx, samples=frozenset({1})) == []


def test_a_loop_that_starts_at_its_last_sample_has_no_seam_to_judge():
    """A one-sample clip: its last sample is its start, so the pop compares it with itself."""
    whole = make_ctx(loop=True)
    data = whole.samples["walk"]
    ctx = edit(whole, times=TIMES[:1], shells={k: v[:1] for k, v in data.shells.items()},
               groups={k: v[:1] for k, v in data.groups.items()},
               props={k: v[:1] for k, v in data.props.items()})
    assert not any(u.item == "walk pop" for u in ClipHygiene().plants(ctx))
    (line,) = [x for x in ClipHygiene().coverage(ctx) if "walk pop" in x]
    assert "not applicable" in line and "last sample" in line and not isinstance(line, Unmeasured)
    assert run_gate([ClipHygiene()], ctx, None).status == "passed"


def test_a_support_sunk_below_the_lift_plant_still_gets_a_caught_plant():
    """The first fault lifts by two bands. A support sunk deeper than that is still under the
    band, so the unit's next fault sets its lowest point two bands above the floor."""
    ctx = lift_support(make_ctx(), "walk", "foot.R", -5 * BAND, 0)
    check = ClipHygiene()
    (unit,) = [u for u in check.plants(ctx) if u.item == "walk stand foot.R"]
    assert not caught(check, unit.candidates[0])
    hit = next(p for p in unit.candidates if caught(check, p))
    assert hit.samples == frozenset({0})


def test_the_prop_unit_has_a_fault_no_slack_in_the_sample_times_can_undo():
    ctx = make_ctx()
    (unit,) = [u for u in ClipHygiene().plants(ctx) if u.item == "walk prop rod"]
    (window,) = unit.candidates[-1].ctx.clips["walk"].props["rod"].show_frames
    assert window[0] == window[1]  # one frame, between two samples
    assert window_problem(window, ctx.clips["walk"].frames) is None  # the loader accepts it
    assert not len(due(ctx.samples["walk"].times, (window,)))
    assert caught(ClipHygiene(), unit.candidates[-1])


def test_a_no_view_slide_plant_keeps_the_original_order_of_its_product():
    """With no view the offset is `away * C * factor * r`, left to right.
    Grouping `C * (factor * r)` first moves some bits."""
    ctx = make_ctx()
    foot = np.array(ctx.samples["walk"].groups["foot.R"])
    foot[0] += (0.011, 0.0, 0.023)  # the creep the plant must move away from
    ctx = edit(ctx, groups={**ctx.samples["walk"].groups, "foot.R": foot})
    (unit,) = [u for u in ClipHygiene().plants(ctx) if u.item == "walk slide foot.R"]
    step = foot[1][np.argmin(foot[1][:, 1])] - foot[0][np.argmin(foot[0][:, 1])]
    away = np.array([step[0], 0.0, step[2]])
    away = away / float(np.linalg.norm(away))
    planted = unit.candidates[0].ctx.samples["walk"].groups["foot.R"]
    want = foot[1] + away * CALIBRATION_FAIL * SLIDE_OF_EFFECTOR * RADIUS
    assert np.array_equal(planted[1], want)
