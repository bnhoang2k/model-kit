import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.sides import Sides
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import Cross, Use
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    HALF_SHOULDER_M,
    USE_ANGLE_DEG,
)

from .manifest_scenes import (
    HOLD_Z,
    TOP,
    deck_geometry,
    make_ctx,
    plants_caught,
    samples,
    with_bench,
    with_clip,
    with_clip_hold,
    with_hold,
    with_piece,
)
from .piece_scenes import accepted, refuses_the_core, tried_and_caught

R_HOLD = "play holds[0]: hand.R on kit:Deck.hold_R side"
L_HOLD = "play holds[1]: hand.L on kit:Deck.hold_L side"
USE = "play use Deck.front"
HALF = HALF_SHOULDER_M  # the test scene is in metres (units_per_metre 1)
OUT = HALF + 0.02  # a point or hand past the band
IN = HALF - 0.02  # just inside it


def hold_at(ctx, piece, hold_id, x, side):
    """The scene with hold `hold_id` of `kit:<piece>` at x `x`, on `side`."""
    holds = tuple(dataclasses.replace(h, point=(x, TOP, HOLD_Z), side=side) if h.id == hold_id
                  else h for h in ctx.packs["kit"].pieces[piece].holds)
    return with_piece(ctx, piece, holds=holds)


def findings(ctx, only=None):
    return Sides().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def test_the_check_is_registered_and_is_not_a_piece_check():
    assert gate.CHECKS["sides"] is Sides
    assert Sides.name == "sides"
    assert Sides.kinds == frozenset()


def test_a_clean_scene_passes_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(Sides(), ctx)
    assert run_gate([Sides()], ctx, None).status == "passed"


# --- hold side, on the piece ------------------------------------------------------------------


def test_a_hold_on_the_wrong_side_of_its_station_fails():
    ctx = with_hold(make_ctx(), "hold_R", side="L", point=(OUT, TOP, HOLD_Z))
    (found,) = [f for f in findings(ctx) if f.item.startswith("kit:Deck")]
    assert found.item == "kit:Deck hold hold_R side"


def test_right_is_facing_cross_up():
    # the user looks toward +z instead: their right is -x, so the R hold at +x is on their left
    deck = make_ctx().packs["kit"].pieces["Deck"]
    use = dataclasses.replace(deck.uses[0], facing=(0.0, 0.0, 1.0))
    ctx = with_piece(make_ctx(), "Deck", uses=(use,))
    ctx = hold_at(hold_at(ctx, "Deck", "hold_R", OUT, "R"), "Deck", "hold_L", -OUT, "L")
    assert {f.item for f in findings(ctx) if f.item.startswith("kit:Deck")} == {
        "kit:Deck hold hold_R side", "kit:Deck hold hold_L side"}


def test_a_hold_on_the_centre_plane_is_inside_the_band_and_passes():
    for side in ("R", "L"):
        ctx = with_hold(make_ctx(), "hold_R", point=(0.0, TOP, HOLD_Z), side=side)
        assert "kit:Deck hold hold_R side" not in items(ctx)


@pytest.mark.parametrize("x, side, fails", [(0.001, "R", False), (-0.001, "R", False),
                                            (-IN, "R", False), (-OUT, "R", True),
                                            (OUT, "R", False), (IN, "L", False),
                                            (OUT, "L", True), (-OUT, "L", False),
                                            (0.7, "both", False), (-0.7, "both", False)])
def test_only_a_hold_past_the_band_is_on_the_wrong_side_and_both_passes(x, side, fails):
    ctx = with_hold(make_ctx(), "hold_R", point=(x, TOP, HOLD_Z), side=side)
    assert ("kit:Deck hold hold_R side" in items(ctx)) is fails


def test_the_band_is_half_a_shoulder_in_pack_units():
    # half of 0.335 m (Gordon et al. 2014, women, 5th percentile), times units_per_metre
    assert HALF_SHOULDER_M == pytest.approx(0.335 / 2)
    ctx = with_hold(make_ctx(), "hold_R", point=(-0.1, TOP, HOLD_Z))
    assert "kit:Deck hold hold_R side" not in items(ctx)
    # the same point in a scene of half-metre units lies 0.2 m from the plane: past the band
    ctx = dataclasses.replace(ctx, style=dataclasses.replace(ctx.style, units_per_metre=0.5))
    (found,) = [f for f in findings(ctx) if f.item == "kit:Deck hold hold_R side"]
    assert "0.0838" in found.detail


def test_r_hold_past_half_a_shoulder_fails_on_two_pieces():
    ctx = with_bench()
    for piece in ("Deck", "Bench"):
        wrong_r = hold_at(ctx, piece, "hold_R", -OUT, "R")
        assert f"kit:{piece} hold hold_R side" in items(wrong_r)
        wrong_l = hold_at(ctx, piece, "hold_L", OUT, "L")  # L mirrors R
        assert f"kit:{piece} hold hold_L side" in items(wrong_l)


def test_r_hold_just_inside_the_band_passes_on_two_pieces():
    ctx = with_bench()
    for piece in ("Deck", "Bench"):
        near_r = hold_at(ctx, piece, "hold_R", -IN, "R")
        assert f"kit:{piece} hold hold_R side" not in items(near_r)
        near_l = hold_at(ctx, piece, "hold_L", IN, "L")
        assert f"kit:{piece} hold hold_L side" not in items(near_l)


def test_a_both_sides_hold_is_a_plain_not_applicable_line():
    lines = [c for c in Sides().coverage(make_ctx()) if "hold_B" in c]
    assert sorted(lines) == [
        "sides: not applicable (both-side hold): kit:Deck hold hold_B",
        "sides: not applicable (both-side hold): play holds[2]: hand.R on kit:Deck.hold_B side",
        "sides: not applicable (both-side hold): play_L holds[2]: hand.L on kit:Deck.hold_B side"]
    assert not any(isinstance(c, Unmeasured) for c in lines)


def test_a_station_that_faces_straight_up_raises():
    deck = make_ctx().packs["kit"].pieces["Deck"]
    use = Use(deck.uses[0].id, deck.uses[0].point, (0.0, 1.0, 0.0), "stand")
    with pytest.raises(ManifestError, match="front"):
        findings(with_piece(make_ctx(), "Deck", uses=(use,)))


# --- hold side, in a clip ---------------------------------------------------------------------


def hands_at(ctx, x):
    """Every clip's right hand at x `x` from the rig root, the left hand opposite."""
    return dataclasses.replace(ctx, samples={n: samples(n, x) for n in ctx.samples})


def wide(ctx=None):
    """The scene with each hand past the band on its own side (the hands read right)."""
    return hands_at(make_ctx() if ctx is None else ctx, OUT)


def swapped_hands(ctx, clip="play"):
    """Both hands of `clip` on the other side. The caller sets them past the band first
    (`wide`), or the swap stays inside it."""
    s = ctx.samples[clip]
    groups = {"hand.R": s.groups["hand.L"], "hand.L": s.groups["hand.R"]}
    return dataclasses.replace(ctx, samples={**ctx.samples,
                                             clip: dataclasses.replace(s, groups=groups)})


def test_a_hand_on_the_other_side_of_the_rig_fails():
    assert {f.item for f in findings(swapped_hands(wide()))} == {R_HOLD, L_HOLD}


def test_a_hand_past_half_a_shoulder_fails_and_just_inside_passes_on_two_clips():
    ctx = with_bench()
    for clip, hold in (("play", R_HOLD), ("rest", "rest holds[0]: hand.R on kit:Bench.hold_R side")):
        out = hands_at(ctx, OUT)
        out = dataclasses.replace(out, samples={**out.samples, clip: swapped_hands(out, clip)
                                                .samples[clip]})
        assert hold in {f.item for f in findings(out)}
        near = hands_at(ctx, IN)
        near = dataclasses.replace(near, samples={**near.samples, clip: swapped_hands(near, clip)
                                                  .samples[clip]})
        assert hold not in {f.item for f in findings(near)}


def test_a_hand_on_the_centre_plane_is_inside_the_band_and_passes():
    ctx = make_ctx()
    ctx = dataclasses.replace(ctx, samples={**ctx.samples, "play": samples("play", right_x=0.0)})
    assert {R_HOLD, L_HOLD}.isdisjoint(f.item for f in findings(ctx))


def test_the_side_is_measured_from_the_rig_root_not_from_a_name():
    ctx = make_ctx()
    s = ctx.samples["play"]
    root = np.tile(np.eye(4), (len(s.times), 1, 1))
    root[:, :3, 3] = (0.5, 0.0, 0.3)  # the doll stands right of the deck: both hands are left of it
    bones = {**s.bones, "root": root}
    ctx = dataclasses.replace(ctx, samples={**ctx.samples,
                                            "play": dataclasses.replace(s, bones=bones)})
    assert R_HOLD in {f.item for f in findings(ctx)}
    assert L_HOLD not in {f.item for f in findings(ctx)}


def test_sides_measures_from_named_root():
    from model_kit.core.manifest import Bone

    ctx = make_ctx()
    rig = ctx.packs["kit"].pieces["Doll"].rig
    # A second parentless bone: with no `root` the rig has no one root to measure from.
    two = dataclasses.replace(rig, bones=(*rig.bones, Bone("tail", None, 0.1)))
    with pytest.raises(ManifestError, match=r"parentless bones.*tail"):
        findings(with_piece(ctx, "Doll", rig=two))
    named = with_piece(ctx, "Doll", rig=dataclasses.replace(two, root="root"))
    assert findings(named) == findings(ctx)
    # The named root decides the side: name the body bone, which sits at the same x here, and
    # then move it right of the deck, so the right hold falls on the wrong side.
    s = ctx.samples["play"]
    body = np.tile(np.eye(4), (len(s.times), 1, 1))
    body[:, :3, 3] = (0.5, 0.0, 0.3)
    bones = {**s.bones, "body": body}
    moved = dataclasses.replace(ctx, samples={**ctx.samples, "play": dataclasses.replace(
        s, bones=bones)})
    assert R_HOLD not in {f.item for f in findings(moved)}
    by_body = with_piece(moved, "Doll", rig=dataclasses.replace(rig, root="body"))
    assert R_HOLD in {f.item for f in findings(by_body)}


def test_the_side_follows_the_placed_station():
    ctx = make_ctx()
    gear = ctx.clips["play"].floor_gear
    turned = dataclasses.replace(gear.pieces["Deck"], quaternion=(0.0, 1.0, 0.0, 0.0))
    ctx = with_clip(wide(ctx), "play", floor_gear=dataclasses.replace(gear, pieces={"Deck": turned}))
    assert {R_HOLD, L_HOLD} <= {f.item for f in findings(ctx)}


def test_a_deliberate_cross_is_excused():
    ctx = with_clip_hold(swapped_hands(wide()), "play", 0, cross=Cross("the hand reaches over"))
    assert {f.item for f in findings(ctx)} == {L_HOLD}
    assert any("hold_R" in c and not isinstance(c, Unmeasured) for c in Sides().coverage(ctx))


def test_a_both_sides_hold_passes_whichever_hand():
    ctx = swapped_hands(wide())
    assert not [f for f in findings(ctx) if "hold_B" in f.item]


def test_only_the_first_due_sample_counts():
    ctx = swapped_hands(wide())
    s = ctx.samples["play"]
    good = wide().samples["play"]
    groups = {k: np.concatenate([good.groups[k][:4], s.groups[k][4:]]) for k in s.groups}
    ctx = dataclasses.replace(ctx, samples={"play": dataclasses.replace(s, groups=groups),
                                            "play_L": ctx.samples["play_L"]})
    assert findings(ctx) == []  # the hands cross after the holds' first due sample


def test_a_clip_with_no_samples_or_no_due_sample_is_unmeasured():
    ctx = make_ctx()
    none = dataclasses.replace(ctx, samples={})
    lines = [c for c in Sides().coverage(none) if "skipped" in c]
    assert lines and all(isinstance(c, Unmeasured) for c in lines)
    assert run_gate([Sides()], none, None).status == "partial"
    late = with_clip_hold(ctx, "play", 0, when=((10.0, 10.0),))
    ctx_s = ctx.samples["play"]
    cut = dataclasses.replace(ctx_s, times=ctx_s.times[:5], groups={
        k: v[:5] for k, v in ctx_s.groups.items()}, bones={k: v[:5] for k, v in ctx_s.bones.items()})
    late = dataclasses.replace(late, samples={**ctx.samples, "play": cut})
    assert any(isinstance(c, Unmeasured) and "holds[0]" in c for c in Sides().coverage(late))


# --- use side ---------------------------------------------------------------------------------


def root_for(angle_degrees, length=0.5):
    """A root position from which the deck's holds read `angle_degrees` off the station's
    facing (the holds' centroid is (0, TOP, HOLD_Z))."""
    a = np.radians(angle_degrees)
    return (-length * np.sin(a), 0.0, HOLD_Z + length * np.cos(a))


def rooted(ctx, position):
    out = dict(ctx.samples)
    for name, s in ctx.samples.items():
        root = np.tile(np.eye(4), (len(s.times), 1, 1))
        root[:, :3, 3] = position
        out[name] = dataclasses.replace(s, bones={**s.bones, "root": root})
    return dataclasses.replace(ctx, samples=out)


def test_calibration_half_the_angle_passes_and_double_fails():
    base = make_ctx()
    assert USE not in {f.item for f in findings(rooted(base, root_for(0.0)))}
    assert USE not in {f.item for f in findings(rooted(base, root_for(CALIBRATION_PASS
                                                                       * USE_ANGLE_DEG)))}
    found = [f for f in findings(rooted(base, root_for(CALIBRATION_FAIL * USE_ANGLE_DEG)))
             if f.item == USE]
    assert len(found) == 1 and f"{CALIBRATION_FAIL * USE_ANGLE_DEG:.1f}" in found[0].detail


def test_a_piece_turned_a_half_turn_faces_away_from_its_user():
    ctx = make_ctx()
    gear = ctx.clips["play"].floor_gear
    turned = dataclasses.replace(gear.pieces["Deck"], quaternion=(0.0, 1.0, 0.0, 0.0))
    ctx = with_clip(ctx, "play", floor_gear=dataclasses.replace(gear, pieces={"Deck": turned}))
    (found,) = [f for f in findings(ctx) if f.item == USE]
    assert "180" in found.detail


def test_the_facing_is_flattened_to_the_ground():
    deck = make_ctx().packs["kit"].pieces["Deck"]
    use = dataclasses.replace(deck.uses[0], facing=(0.0, -0.5, -1.0))
    assert USE not in {f.item for f in findings(with_piece(make_ctx(), "Deck", uses=(use,)))}


def test_a_station_with_no_holds_is_a_plain_line():
    ctx = with_piece(make_ctx(), "Deck", holds=())
    ctx = with_clip(with_clip(ctx, "play", holds=()), "play_L", holds=())
    lines = [c for c in Sides().coverage(ctx) if "no holds" in c]
    assert sorted(lines) == ["sides: no holds on station: play use Deck.front",
                             "sides: no holds on station: play_L use Deck.front"]
    assert not any(isinstance(c, Unmeasured) for c in lines)


def test_a_scene_with_nothing_to_judge_is_a_not_applicable_line_not_silence():
    ctx = with_piece(make_ctx(), "Deck", holds=(), uses=())
    ctx = with_clip(with_clip(ctx, "play", holds=()), "play_L", holds=())
    assert findings(ctx) == [] and list(Sides().plants(ctx)) == []
    (line,) = Sides().coverage(ctx)
    assert "not applicable" in line and "no piece hold" in line
    assert not isinstance(line, Unmeasured)
    assert not any("no piece hold" in x for x in Sides().coverage(make_ctx()))  # it has holds


def test_the_use_rule_reads_the_first_sample_of_the_clip():
    ctx = make_ctx()
    s = ctx.samples["play"]
    root = np.array(s.bones["root"])
    root[1:, :3, 3] = (5.0, 0.0, 5.0)  # the doll walks away after the first sample
    ctx = dataclasses.replace(ctx, samples={**ctx.samples, "play": dataclasses.replace(
        s, bones={**s.bones, "root": root})})
    assert USE not in {f.item for f in findings(ctx)}


def test_a_clip_with_no_samples_leaves_its_stations_unmeasured():
    ctx = dataclasses.replace(make_ctx(), samples={})
    assert any(isinstance(c, Unmeasured) and USE in c for c in Sides().coverage(ctx))


# --- plants and scope -------------------------------------------------------------------------


def test_the_plants_swap_a_side_and_turn_a_placement():
    # the test scene's holds and hands sit inside the band, so a swapped side leaves them
    # passing: those units are caught by the second candidate, moving across the band
    check, ctx = Sides(), make_ctx()
    labels = {accepted(check, u).label.split(":")[0] for u in check.plants(ctx)}
    assert labels == {"moved across", "turned"}
    far = wide(hold_at(hold_at(ctx, "Deck", "hold_R", OUT, "R"), "Deck", "hold_L", -OUT, "L"))
    labels = {accepted(check, u).label.split(":")[0] for u in check.plants(far)}
    assert labels == {"swapped", "turned"}  # past the band the swap is caught first


def test_a_swapped_side_is_not_caught_inside_the_band_but_a_move_across_is():
    check, ctx = Sides(), make_ctx()
    for item in ("kit:Deck hold hold_R side", R_HOLD):
        (u,) = [u for u in check.plants(ctx) if u.item == item]
        got = {p.label.split(":")[0]: p for p in u.candidates}
        assert list(got) == ["swapped", "moved across"]
        caught = {k: item in {f.item for f in check.run(p.ctx, only=p.scope)}
                  for k, p in got.items()}
        assert caught == {"swapped": False, "moved across": True}


def test_a_move_across_lands_past_the_band_whatever_the_start_and_the_side():
    check = Sides()
    for x, side in ((0.0, "R"), (OUT, "R"), (-OUT, "R"), (0.0, "L"), (-OUT, "L"), (OUT, "L")):
        ctx = with_hold(make_ctx(), "hold_R", point=(x, TOP, HOLD_Z), side=side)
        (u,) = [u for u in check.plants(ctx) if u.item == "kit:Deck hold hold_R side"]
        across = next(p for p in u.candidates if p.label.startswith("moved across"))
        assert "kit:Deck hold hold_R side" in {f.item for f in check.run(across.ctx)}, (x, side)


def test_every_item_that_can_fail_has_a_plant():
    got = {p.item for p in Sides().plants(make_ctx())}
    assert {"kit:Deck hold hold_R side", "kit:Deck hold hold_L side", R_HOLD, L_HOLD,
            USE, "play_L use Deck.front"} <= got


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = swapped_hands(wide())
    assert [f.item for f in findings(ctx, only={L_HOLD})] == [L_HOLD]


def test_a_blind_check_stops_the_run():
    class Blind(Sides):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="sides"):
        run_gate([Blind()], make_ctx(), None)


# --- the plants are built blind to the check -------------------------------------------------


def test_sides_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import sides

    ctx = make_ctx()
    want = sorted(u.item for u in Sides().plants(ctx))
    refuses_the_core(monkeypatch, sides, "_findings", "_piece_entry", "_hold_entry", "_use_entry",
                     "_wrong")
    monkeypatch.setattr(Sides, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in Sides().plants(ctx)) == want and len(want) == 8


def test_sides_plants_caught_on_two_stations():
    """The Deck and the Bench (its twin): each hold's side and each station's use is tried."""
    ctx = with_bench()
    assert findings(ctx) == []
    got = tried_and_caught(Sides(), ctx)
    assert {"kit:Deck hold hold_R side", "kit:Bench hold hold_L side", "play use Deck.front",
            "rest_L use Bench.front", "rest holds[0]: hand.R on kit:Bench.hold_R side"} <= set(got)
    assert len(got) == 16
    assert run_gate([Sides()], ctx, None).status == "passed"


# --- the move-across plant in a clip ----------------------------------------------------------


def with_cup(ctx, x):
    """The clip `play` holds `hold_R` with a prop, `cup` (a piece with the deck's box for a body),
    whose pose puts it at x `x`; its other hold is dropped."""
    from model_kit.core.manifest import Prop
    from model_kit.core.scene import PieceGeom

    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces,
                                           "kit:Doll": PieceGeom("Doll", deck_geometry())})
    cup = Prop("cup", "Doll", "kit", "hand.R", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), ())
    clip = ctx.clips["play"]
    hold = dataclasses.replace(clip.holds[0], who="cup")
    ctx = with_clip(ctx, "play", holds=(hold,), props={"cup": cup})
    s = ctx.samples["play"]
    pose = np.tile(np.eye(4), (len(s.times), 1, 1))
    pose[:, :3, 3] = (x, 0.0, ROOT_Z)
    return dataclasses.replace(ctx, samples={**ctx.samples, "play": dataclasses.replace(
        s, props={"cup": pose})})


ROOT_Z = 0.3
CUP = "play holds[0]: cup on kit:Deck.hold_R side"


def test_a_held_prop_past_the_band_fails_and_inside_it_passes():
    assert CUP in {f.item for f in findings(with_cup(make_ctx(), -OUT))}
    assert CUP not in {f.item for f in findings(with_cup(make_ctx(), -IN))}


def test_a_prop_inside_the_band_is_caught_by_moving_it_across_and_only_at_the_due_sample():
    check, ctx = Sides(), with_cup(make_ctx(), -0.1)
    (u,) = [u for u in check.plants(ctx) if u.item == CUP]
    got = {p.label.split(":")[0]: p for p in u.candidates}
    assert list(got) == ["swapped", "moved across"]
    assert CUP not in {f.item for f in check.run(got["swapped"].ctx, only=got["swapped"].scope)}
    moved = got["moved across"]
    assert CUP in {f.item for f in check.run(moved.ctx, only=moved.scope)}
    before, after = ctx.samples["play"].props["cup"], moved.ctx.samples["play"].props["cup"]
    assert moved.samples == {0} and (after[1:] == before[1:]).all()  # the first sample only
    assert (ctx.samples["play"].props["cup"] == before).all()  # the input scene is untouched


def test_a_hand_inside_the_band_is_caught_by_moving_it_across_on_two_clips():
    ctx = with_bench()
    for item in (R_HOLD, "rest holds[0]: hand.R on kit:Bench.hold_R side"):
        check = Sides()
        (u,) = [u for u in check.plants(ctx) if u.item == item]
        got = {p.label.split(":")[0]: p for p in u.candidates}
        assert list(got) == ["swapped", "moved across"]
        assert item not in {f.item for f in check.run(got["swapped"].ctx)}
        assert item in {f.item for f in check.run(got["moved across"].ctx)}
        assert got["moved across"].samples == {0}


# --- the band is a length: the verdict does not depend on how the facing is written ----------

# Same station, the user looking toward -z, written three ways: unit, longer, shorter, and tilted
# down at the table. The user's right is the horizontal unit vector (1, 0, 0) in all of them.
FACINGS = [(0.0, 0.0, -1.0), (0.0, 0.0, -2.0), (0.0, 0.0, -0.5), (0.0, -0.8, -0.6)]
BENCH_R = "rest holds[0]: hand.R on kit:Bench.hold_R side"


def facing_of(ctx, piece, facing):
    deck = ctx.packs["kit"].pieces[piece]
    use = dataclasses.replace(deck.uses[0], facing=facing)
    return with_piece(ctx, piece, uses=(use,))


@pytest.mark.parametrize("facing", FACINGS)
def test_a_piece_hold_is_judged_by_the_same_length_whatever_the_facing_is_written(facing):
    ctx = with_bench()
    for piece in ("Deck", "Bench"):
        base = facing_of(ctx, piece, facing)
        item = f"kit:{piece} hold hold_R side"
        assert item not in items(hold_at(base, piece, "hold_R", -IN, "R"))
        (found,) = [f for f in findings(hold_at(base, piece, "hold_R", -OUT, "R"))
                    if f.item == item]
        assert f"{-OUT:+.4f}" in found.detail  # the distance printed is in metres


@pytest.mark.parametrize("facing", FACINGS)
def test_a_clip_hold_is_judged_by_the_same_length_whatever_the_facing_is_written(facing):
    ctx = with_bench()
    for piece, item in (("Deck", R_HOLD), ("Bench", BENCH_R)):
        clip = "play" if piece == "Deck" else "rest"
        base = facing_of(ctx, piece, facing)
        near = swapped_hands(hands_at(base, IN), clip)
        assert item not in {f.item for f in findings(near)}
        far = swapped_hands(hands_at(base, OUT), clip)
        (found,) = [f for f in findings(far) if f.item == item]
        assert f"{-OUT:+.4f}" in found.detail


@pytest.mark.parametrize("facing", FACINGS)
def test_the_plants_still_catch_whatever_the_facing_is_written(facing):
    ctx = facing_of(facing_of(with_bench(), "Deck", facing), "Bench", facing)
    assert {"kit:Deck hold hold_R side", "kit:Bench hold hold_L side", R_HOLD, BENCH_R} <= set(
        tried_and_caught(Sides(), ctx))
