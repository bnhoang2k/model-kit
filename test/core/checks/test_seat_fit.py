"""Check seat-fit: a sitter's knees pass the seat's front edge.

The bodies are the test humanoid on its chair (`sit_chair`) and its bench (`sit_bench`): two seat
pieces, one sitter whose `sit.past_edge` names the two knees. Both seats face +z, and the sitter's
knees land well past the front edge. The plants move the sitter back along the facing.
"""

import dataclasses
import math
from functools import cache

import numpy as np
import pytest

from model_kit.core.checks.seat_fit import NAME, PRINCIPLE, SOURCE, SeatFit, front_edge
from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.manifest import FloorGear, Prop, Ref
from model_kit.core.metagate import Finding, Unmeasured, run_gate
from model_kit.core.scene import SceneCtx, rig_spec
from model_kit.core.supports import body_radius
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS, SKIN_OF_EFFECTOR

from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene
from ..test_scene import shop_ctx
from .piece_scenes import chosen, tried_and_caught

SEATS = {"sit_chair": 0.39, "sit_bench": 0.33}  # the front edge from the seat point, in metres
SEAT_POINT_Z = -0.03  # both seats: the seat point is 0.03 behind the floor point under the seat
KNEES = ("knee.L", "knee.R")


@cache
def scene(body: str = "humanoid") -> SceneCtx:
    return {"humanoid": humanoid_scene, "quadruped": quadruped_scene, "worm": worm_scene}[body]()


def tolerance(ctx: SceneCtx, clip: str) -> float:
    key = ctx.rig_of_clip[clip]
    return SKIN_OF_EFFECTOR * body_radius(rig_spec(ctx, clip), ctx.rests[key])


def caught(check, plant) -> bool:
    found = check.run(plant.ctx, only=plant.scope, samples=plant.samples)
    return any(f.item == plant.item for f in found)


def item(clip: str, bone: str = "knee.L") -> str:
    return f"{clip} {bone} behind the edge"


def with_clip(ctx: SceneCtx, clip: str, **fields) -> SceneCtx:
    changed = dataclasses.replace(ctx.clips[clip], **fields)
    return dataclasses.replace(ctx, clips={**ctx.clips, clip: changed})


def with_samples(ctx: SceneCtx, clip: str, **fields) -> SceneCtx:
    changed = dataclasses.replace(ctx.samples[clip], **fields)
    return dataclasses.replace(ctx, samples={**ctx.samples, clip: changed})


def with_sit(ctx: SceneCtx, **fields) -> SceneCtx:
    """The scene with the rigged piece's `sit` changed."""
    key = next(iter(ctx.rigs))
    pack, _, name = key.partition(":")
    piece = ctx.packs[pack].pieces[name]
    rig = dataclasses.replace(piece.rig, sit=dataclasses.replace(piece.rig.sit, **fields))
    pieces = {**ctx.packs[pack].pieces, name: dataclasses.replace(piece, rig=rig)}
    return dataclasses.replace(ctx, packs={**ctx.packs, pack: dataclasses.replace(
        ctx.packs[pack], pieces=pieces)})


def knee_z(ctx: SceneCtx, clip: str, bone: str = "knee.L") -> np.ndarray:
    return np.asarray(ctx.samples[clip].bones[bone])[:, 2, 3]


def sitter_at(ctx: SceneCtx, clip: str, bone: str, past: float) -> SceneCtx:
    """The scene with the sitter moved along z, by hand, so `bone`'s head lies `past` past the
    front edge at the first sample. The seats are at the origin and face +z, so the edge is at
    z = seat point + front edge. This reads no check code."""
    edge = SEAT_POINT_Z + SEATS[clip]
    move = np.array([0.0, 0.0, edge + past - knee_z(ctx, clip, bone)[0]])
    data = ctx.samples[clip]
    bones = {}
    for name, matrices in data.bones.items():
        moved = np.array(matrices, dtype=float)
        moved[:, :3, 3] += move
        bones[name] = moved
    return with_samples(ctx, clip, bones=bones,
                        groups={g: np.asarray(p) + move for g, p in data.groups.items()},
                        shells={p: np.asarray(v) + move.astype(np.float32)
                                for p, v in data.shells.items()})


def turned(ctx: SceneCtx, clip: str, degrees: float, shift) -> SceneCtx:
    """The scene with the seat piece and the sitter turned about the vertical axis and moved: the
    whole clip as one rigid body."""
    a = math.radians(degrees)
    turn = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0],
                     [-math.sin(a), 0.0, math.cos(a)]])
    shift = np.asarray(shift, dtype=float)
    data = ctx.samples[clip]
    bones = {}
    for name, matrices in data.bones.items():
        moved = np.array(matrices, dtype=float)
        moved[:, :3, :3] = turn @ moved[:, :3, :3]
        moved[:, :3, 3] = moved[:, :3, 3] @ turn.T + shift
        bones[name] = moved
    placement = ctx.clips[clip].floor_gear.pieces[ctx.clips[clip].seat.piece]
    q = [0.0, math.sin(a / 2), 0.0, math.cos(a / 2)]
    pieces = {**ctx.clips[clip].floor_gear.pieces, placement.id: dataclasses.replace(
        placement, position=tuple(float(v) for v in shift), quaternion=tuple(q))}
    out = with_clip(ctx, clip, floor_gear=dataclasses.replace(ctx.clips[clip].floor_gear,
                                                              pieces=pieces))
    return with_samples(out, clip, bones=bones,
                        groups={g: np.asarray(p) @ turn.T + shift for g, p in data.groups.items()},
                        shells={p: (np.asarray(v, dtype=float) @ turn.T + shift).astype(np.float32)
                                for p, v in data.shells.items()})


# --- the front edge ---------------------------------------------------------------------------------


@pytest.mark.parametrize("clip", list(SEATS))
def test_front_edge_of_a_box_seat(clip):
    """The seat's surface is one box: the edge is its front face, from the seat point."""
    ctx = scene()
    ref = ctx.clips[clip].seat
    assert front_edge(ctx, ref) == pytest.approx(SEATS[clip])  # in the piece's own frame
    assert front_edge(ctx, ref, ctx.clips[clip]) == pytest.approx(SEATS[clip])  # as placed


def test_the_front_edge_follows_the_placement():
    """A seat turned a quarter round and moved still reaches the same distance along its facing."""
    ctx = turned(scene(), "sit_chair", 90.0, (3.0, 0.0, -2.0))
    clip = ctx.clips["sit_chair"]
    assert front_edge(ctx, clip.seat, clip) == pytest.approx(SEATS["sit_chair"])


def test_the_edge_is_taken_from_the_surface_parts_alone():
    """The chair's back rises behind the seat, the legs stand under it: neither is the surface."""
    ctx = scene()
    ref = ctx.clips["sit_chair"].seat
    piece = ctx.pieces["humanoid:Chair"]
    assert set(piece.parts) > {"seat"}
    deeper = dataclasses.replace(piece, parts={**piece.parts, "back": piece.parts["back"] + [0, 0, 1.0]})
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, "humanoid:Chair": deeper})
    assert front_edge(ctx, ref) == pytest.approx(SEATS["sit_chair"])


def test_front_edge_of_something_that_is_no_seat_raises():
    ctx = scene()
    with pytest.raises(Unmeasurable, match="not a seat"):
        front_edge(ctx, Ref.parse("humanoid:Chair.sit"))


# --- knees past the edge pass --------------------------------------------------------------------------


@pytest.mark.parametrize("clip", list(SEATS))
def test_knees_past_edge_pass(clip):
    ctx = scene()
    assert SeatFit().run(ctx) == []
    assert (knee_z(ctx, clip) > SEAT_POINT_Z + SEATS[clip]).all()


@pytest.mark.parametrize("clip", list(SEATS))
@pytest.mark.parametrize("bone", KNEES)
def test_the_tolerance_is_a_skin_of_the_smallest_radius(clip, bone):
    """Two tolerances past the edge pass; at the edge, and inside the tolerance, fail; behind
    the edge fails. The sitter is moved by hand, so the numbers do not come from the check."""
    ctx = scene()
    tol = tolerance(ctx, clip)
    spec = rig_spec(ctx, clip)
    smallest = min(c.effector_radius for c in spec.chains.values())  # the arms: 0.034, not the legs
    assert tol == pytest.approx(SKIN_OF_EFFECTOR * smallest)
    check = SeatFit()
    for past, ok in ((CALIBRATION_FAIL * tol, True), (CALIBRATION_PASS * tol, False),
                     (0.0, False), (-CALIBRATION_FAIL * tol, False), (-0.2, False)):
        found = check.run(sitter_at(ctx, clip, bone, past), only={item(clip, bone)})
        assert (found == []) is ok, (past, found)


def test_a_finding_names_the_clip_the_bone_and_the_gap():
    ctx = sitter_at(scene(), "sit_chair", "knee.R", -0.05)
    [found] = SeatFit().run(ctx, only={item("sit_chair", "knee.R")})
    assert found.check == NAME and found.item == "sit_chair knee.R behind the edge"
    assert "0.0500 behind the edge" in found.detail and "frame 1" in found.detail
    # the whole sitter moved, so the other knee is behind too, but only the named item is read
    assert {f.item for f in SeatFit().run(ctx)} == {item("sit_chair"), item("sit_chair", "knee.R")}


def test_a_turned_and_moved_seat_gives_the_same_verdicts():
    ctx = turned(scene(), "sit_bench", 135.0, (-1.5, 0.0, 4.0))
    check = SeatFit()
    assert check.run(ctx) == []
    bench_items = {item("sit_bench", bone) for bone in KNEES}
    plants = [p for p in chosen(check, ctx) if p.item in bench_items]
    assert len(plants) == 2 and all(caught(check, p) for p in plants)


def test_only_the_samples_in_the_seat_windows_are_judged():
    """A clip that sits down part way through: before its seat window the knees are free."""
    ctx = scene()
    behind = sitter_at(ctx, "sit_chair", "knee.L", -0.1)
    check = SeatFit()
    assert [f.item for f in check.run(behind)] == [item("sit_chair"), item("sit_chair", "knee.R")]
    windowed = with_clip(behind, "sit_chair", seat_when=((1.2, 1.8),))  # between two samples
    assert check.run(windowed) == []
    assert any(isinstance(x, Unmeasured) and "no sample inside the seat windows" in x
               for x in check.coverage(windowed))
    early = with_clip(behind, "sit_chair", seat_when=((1.0, 5.0),))
    assert item("sit_chair") in {f.item for f in check.run(early)}
    # a sample the window leaves out is not judged, even though the knee is behind there
    data = ctx.samples["sit_chair"]
    moved = {b: np.array(m) for b, m in data.bones.items()}
    for m in moved.values():
        m[8:, 2, 3] -= 0.5  # samples 8 on: the sitter slides back, outside the window
    late = with_clip(with_samples(ctx, "sit_chair", bones=moved), "sit_chair",
                     seat_when=((1.0, 5.0),))
    assert check.run(late) == []
    assert item("sit_chair") in {f.item for f in check.run(
        with_clip(with_samples(ctx, "sit_chair", bones=moved), "sit_chair"))}


def test_only_and_samples_narrow_a_run():
    ctx = sitter_at(scene(), "sit_bench", "knee.L", -0.1)
    check = SeatFit()
    assert {f.item for f in check.run(ctx, only={item("sit_bench")})} == {item("sit_bench")}
    assert check.run(ctx, only={item("sit_chair")}) == []
    assert check.run(ctx, samples=frozenset({3}))  # the move holds at every sample
    assert check.run(ctx, only={item("sit_bench")}, samples=frozenset({99})) == []


# --- a seat the clip carries -----------------------------------------------------------------------------------


def carried(ctx: SceneCtx, clip: str, poses: np.ndarray) -> SceneCtx:
    """The clip with its chair carried as a prop that sits at `poses` (S, 4, 4) in the world
    instead of being placed on the floor."""
    placed = ctx.clips[clip].floor_gear.pieces[ctx.clips[clip].seat.piece]
    prop = Prop("seat", placed.piece, placed.pack, "pelvis", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                ((1.0, 13.0),))
    seat = Ref(placed.pack, placed.piece, ctx.clips[clip].seat.id)
    out = with_clip(ctx, clip, floor_gear=FloorGear({}), props={"seat": prop}, seat=seat)
    return with_samples(out, clip, props={"seat": poses})


def test_a_carried_seat_is_read_at_each_sample():
    ctx = scene()
    n = len(ctx.samples["sit_chair"].times)
    check = SeatFit()
    still = carried(ctx, "sit_chair", np.tile(np.eye(4), (n, 1, 1)))
    assert check.run(still) == []
    assert all(caught(check, p) for p in chosen(check, still))
    # the seat slides forward 0.5 from sample 6 on, so the knees are behind its edge there
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[6:, 2, 3] = 0.5
    [found, *_] = check.run(carried(ctx, "sit_chair", poses))
    assert found.item == item("sit_chair") and "frame 7" in found.detail


def test_a_carried_seat_that_turns_up_is_unmeasured_not_a_crash():
    ctx = scene()
    n = len(ctx.samples["sit_chair"].times)
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[-1, :3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]  # the last sample faces straight up
    ctx = carried(ctx, "sit_chair", poses)
    check = SeatFit()
    assert check.run(ctx) == []
    assert {p.item for p in chosen(check, ctx)} == {item("sit_bench"), item("sit_bench", "knee.R")}
    assert any(isinstance(x, Unmeasured) and "faces straight up or down" in x
               for x in check.coverage(ctx))
    assert run_gate([check], ctx, None).status == "partial"


# --- plants ----------------------------------------------------------------------------------------------


def test_plant_caught_on_chair_and_bench():
    """Each knee of each seat gets a plant: the sitter moved back until that knee sits two
    tolerances behind the front edge."""
    ctx = scene()
    check = SeatFit()
    plants = chosen(check, ctx)
    by_item = {item(c, b): (c, b) for c in SEATS for b in KNEES}
    assert {p.item for p in plants} == set(by_item)
    pieces = {ctx.clips[c].seat.piece for c in SEATS}
    assert pieces == {"Chair", "Bench"}
    for plant in plants:
        assert caught(check, plant), plant.label
        clip, bone = by_item[plant.item]
        tol = tolerance(ctx, clip)
        gap = knee_z(plant.ctx, clip, bone)[0] - (SEAT_POINT_Z + SEATS[clip])
        assert gap == pytest.approx(-CALIBRATION_FAIL * tol, abs=1e-6)
        assert plant.samples == frozenset({0})


def test_a_plant_is_a_finding_only_because_it_was_planted():
    ctx = scene()
    check = SeatFit()
    for plant in chosen(check, ctx):
        assert check.run(ctx, only={plant.item}, samples=plant.samples) == [], plant.label


def test_the_plant_moves_the_whole_sitter_and_leaves_the_scene_alone():
    ctx = scene()
    before = {b: np.array(m) for b, m in ctx.samples["sit_chair"].bones.items()}
    plant = next(p for p in chosen(SeatFit(), ctx) if p.item == item("sit_chair"))
    for bone, matrices in plant.ctx.samples["sit_chair"].bones.items():
        np.testing.assert_allclose(matrices[:, :3, :3], before[bone][:, :3, :3])
        step = matrices[:, :3, 3] - before[bone][:, :3, 3]
        np.testing.assert_allclose(step, np.tile(step[0], (len(step), 1)), atol=1e-12)
        assert step[0][2] < 0 and abs(step[0][0]) < 1e-12 and abs(step[0][1]) < 1e-12
    for bone, matrices in ctx.samples["sit_chair"].bones.items():
        np.testing.assert_array_equal(matrices, before[bone])


def test_an_item_that_already_fails_has_no_plant():
    behind = sitter_at(scene(), "sit_chair", "knee.L", -0.1)
    check = SeatFit()
    assert {p.item for p in chosen(check, behind)} == {item("sit_bench"), item("sit_bench", "knee.R")}
    assert {f.item for f in check.run(behind)} == {item("sit_chair"), item("sit_chair", "knee.R")}


def test_the_meta_gate_accepts_every_plant_and_the_run_passes():
    verdict = run_gate([SeatFit()], scene(), None)
    assert verdict.status == "passed"


def test_the_meta_gate_fails_a_sitter_behind_the_edge():
    verdict = run_gate([SeatFit()], sitter_at(scene(), "sit_bench", "knee.R", -0.04), None)
    assert verdict.status == "failed" and verdict.failed == 2
    assert all(f"seat-fit | sit_bench knee.{side} behind the edge" in " ".join(verdict.lines)
               for side in "LR")


# --- coverage ------------------------------------------------------------------------------------------------


def test_no_past_edge_is_partial():
    """A rig whose sit names no past_edge bone cannot be measured: the run is partial."""
    ctx = with_sit(scene(), past_edge=())
    lines = SeatFit().coverage(ctx)
    blind = [x for x in lines if isinstance(x, Unmeasured)]
    assert {x.split(": ")[-1] for x in blind} == {"sit_chair behind the edge",
                                                  "sit_bench behind the edge"}
    assert all("past_edge" in x for x in blind)
    assert SeatFit().plants(ctx) == [] and SeatFit().run(ctx) == []
    assert run_gate([SeatFit()], ctx, None).status == "partial"


def test_the_shop_doll_has_no_past_edge_so_it_is_partial():
    """The fixture Doll sits on the fixture Bench and has no knee, so no past_edge."""
    ctx = shop_ctx()
    assert rig_spec(ctx, "sit_reach").sit.past_edge == ()
    lines = SeatFit().coverage(ctx)
    assert any(isinstance(x, Unmeasured) and "past_edge" in x and "sit_reach" in x for x in lines)
    assert run_gate([SeatFit()], ctx, None).status == "partial"


def test_stand_clip_not_applicable():
    lines = SeatFit().coverage(scene())
    assert lines == [
        f"{NAME}: not applicable (stand clip: only a sit clip rests on a seat): stand behind the edge",
        f"{NAME}: not applicable (stand clip: only a sit clip rests on a seat): step behind the edge"]
    assert not any(isinstance(x, Unmeasured) for x in lines)


@pytest.mark.parametrize("body", ["quadruped", "worm"])
def test_a_rig_with_no_sit_is_not_applicable(body):
    ctx = scene(body)
    assert all(rig_spec(ctx, clip).sit is None for clip in ctx.clips)
    lines = SeatFit().coverage(ctx)
    assert lines and all("not applicable (the rig has no sit)" in x for x in lines)
    assert SeatFit().plants(ctx) == [] and SeatFit().run(ctx) == []


def test_a_clip_with_no_posture_is_unmeasured():
    ctx = with_clip(scene(), "sit_chair", posture=None)
    lines = SeatFit().coverage(ctx)
    assert any(isinstance(x, Unmeasured) and "no posture" in x and "sit_chair" in x for x in lines)
    assert run_gate([SeatFit()], ctx, None).status == "partial"


def test_a_sit_clip_with_no_seat_cannot_exist():
    """The check has no not-applicable line for this clip. The clip refuses to be built,
    so the check never meets it."""
    with pytest.raises(ManifestError, match=r"clip 'sit_chair': posture: a 'sit' clip must name"):
        with_clip(scene(), "sit_chair", seat=None)


def test_a_clip_with_no_samples_is_unmeasured():
    ctx = scene()
    ctx = dataclasses.replace(ctx, samples={k: v for k, v in ctx.samples.items()
                                            if k != "sit_bench"})
    lines = SeatFit().coverage(ctx)
    assert any(isinstance(x, Unmeasured) and "no samples" in x and "sit_bench" in x for x in lines)
    assert SeatFit().run(ctx) == []


def test_a_past_edge_bone_the_samples_lack_is_unmeasured():
    ctx = scene()
    data = ctx.samples["sit_chair"]
    ctx = with_samples(ctx, "sit_chair", bones={b: m for b, m in data.bones.items()
                                                if b != "knee.R"})
    lines = SeatFit().coverage(ctx)
    assert any(isinstance(x, Unmeasured) and "no bone knee.R" in x for x in lines)
    check = SeatFit()
    assert check.run(ctx) == []
    assert item("sit_chair") in {p.item for p in chosen(check, ctx)}  # the other knee is measured
    assert item("sit_chair", "knee.R") not in {p.item for p in chosen(check, ctx)}


def test_a_seat_with_no_geometry_is_unmeasured():
    ctx = scene()
    gone = {k: v for k, v in ctx.pieces.items() if k != "humanoid:Chair"}
    lines = SeatFit().coverage(dataclasses.replace(ctx, pieces=gone))
    assert any(isinstance(x, Unmeasured) and "no geometry for piece humanoid:Chair" in x
               for x in lines)


def test_a_seat_facing_straight_up_is_unmeasured():
    ctx = scene()
    pack = ctx.packs["humanoid"]
    piece = pack.pieces["Chair"]
    seat = dataclasses.replace(piece.seats[0], facing=(0.0, 1.0, 0.0))
    chair = dataclasses.replace(piece, seats=(seat,))
    ctx = dataclasses.replace(ctx, packs={"humanoid": dataclasses.replace(
        pack, pieces={**pack.pieces, "Chair": chair})})
    lines = SeatFit().coverage(ctx)
    assert any(isinstance(x, Unmeasured) and "faces straight up or down" in x for x in lines)


# --- the check's face -----------------------------------------------------------------------------------------


def test_the_check_states_its_principle_and_its_source():
    check = SeatFit()
    assert check.name == "seat-fit" and check.kinds == frozenset()
    assert check.principle == PRINCIPLE and check.source == SOURCE
    assert PRINCIPLE.count(". ") == 0 and PRINCIPLE.endswith(".")  # one sentence
    assert "personal.cityu.edu.hk/meachan/online%20anthropometry/Chapter2/Ch2-14.htm" in SOURCE
    assert "buttock-popliteal" in SOURCE


def test_findings_read_as_the_gate_prints_them():
    ctx = sitter_at(scene(), "sit_chair", "knee.L", -0.05)
    [found, *_] = SeatFit().run(ctx)
    assert isinstance(found, Finding) and found.item.endswith(" behind the edge")


# --- plants as units ---------------------------------------------------------------------


def test_the_units_are_tried_and_caught_on_two_pieces():
    ctx = scene()
    assert sorted(tried_and_caught(SeatFit(), ctx)) == sorted(
        item(c, b) for c in SEATS for b in KNEES)
    assert {ctx.clips[c].seat.piece for c in SEATS} == {"Chair", "Bench"}


def test_an_item_that_already_fails_still_has_a_unit_and_the_gate_skips_it():
    behind = sitter_at(scene(), "sit_chair", "knee.L", -0.1)
    check = SeatFit()
    units = {u.item: u for u in check.plants(behind)}
    assert item("sit_chair") in units and units[item("sit_chair")].candidates
    assert item("sit_chair") not in {p.item for p in chosen(check, behind)}


def test_a_unit_holds_a_fault_at_the_first_middle_and_last_seated_sample():
    ctx = scene()
    (unit,) = [u for u in SeatFit().plants(ctx) if u.item == item("sit_chair")]
    seated = sorted({next(iter(p.samples)) for p in unit.candidates})
    assert len(unit.candidates) == len(seated) >= 2
    assert seated[0] == 0 and seated == sorted(seated)


def test_the_seat_setup_is_built_once_per_scene(monkeypatch):
    """The run, the plants and the coverage of one scene share each clip's setup."""
    from model_kit.core.checks import seat_fit
    calls = []
    real = seat_fit._work
    monkeypatch.setattr(seat_fit, "_work", lambda ctx, clip: calls.append(clip.name) or real(ctx, clip))
    ctx = scene()
    check = SeatFit()
    check.run(ctx)
    check.coverage(ctx)
    check.plants(ctx)
    check.run(ctx, only={item("sit_chair")})
    assert sorted(calls) == sorted(ctx.clips)
