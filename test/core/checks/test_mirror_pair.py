import dataclasses

import pytest

from model_kit.core import gate
from model_kit.core.checks.mirror_pair import MirrorPair
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import FloorGear, Placement
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS

from .manifest_scenes import (
    HOLD_Z,
    LIMIT,
    TOP,
    make_ctx,
    plants_caught,
    with_bench,
    with_clip,
    with_clip_hold,
    with_hold,
    with_piece,
)
from .piece_scenes import accepted, refuses_the_core, tried_and_caught
from .sight_scenes import unviewed, viewed

HEAD = "play_L mirror-of play"


def findings(ctx, only=None):
    return MirrorPair().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def test_the_check_is_registered_and_is_not_a_piece_check():
    assert gate.CHECKS["mirror-pair"] is MirrorPair
    assert MirrorPair.name == "mirror-pair"
    assert MirrorPair.kinds == frozenset()


def test_a_clean_mirror_passes_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(MirrorPair(), ctx)
    assert run_gate([MirrorPair()], ctx, None).status == "passed"


def test_a_both_sides_hold_maps_to_itself_with_the_swapped_who():
    # source hold 2 is hold_B (side both) held by hand.R; the mirror holds it with hand.L
    ctx = make_ctx()
    assert ctx.clips["play"].holds[2].ref == ctx.clips["play_L"].holds[2].ref
    assert findings(ctx) == []
    dropped = with_clip_hold(ctx, "play_L", 2, who="hand.R")
    assert items(dropped) == [f"{HEAD} holds[2]"]


def test_dropping_the_who_swap_fails_that_hold():
    ctx = with_clip_hold(make_ctx(), "play_L", 0, who="hand.R")
    (found,) = findings(ctx)
    assert found.item == f"{HEAD} holds[0]"
    assert "hand.L" in found.detail


def test_dropping_the_side_swap_fails_that_hold():
    ctx = make_ctx()
    ref = ctx.clips["play"].holds[0].ref  # the unswapped hold: hold_R again
    ctx = with_clip_hold(ctx, "play_L", 0, ref=ref)
    assert items(ctx) == [f"{HEAD} holds[0]"]


def test_a_different_when_fails():
    ctx = with_clip_hold(make_ctx(), "play_L", 1, when=((1.0, 5.0),))
    assert items(ctx) == [f"{HEAD} holds[1]"]


def test_a_hold_on_another_placement_fails():
    ctx = make_ctx()
    gear = FloorGear({**ctx.clips["play_L"].floor_gear.pieces,
                      "Deck2": Placement("Deck2", "Deck", "kit", (1.0, 0.0, 0.0),
                                         (0.0, 0.0, 0.0, 1.0))})
    ctx = with_clip(ctx, "play_L", floor_gear=gear)
    ref = dataclasses.replace(ctx.clips["play_L"].holds[0].ref, piece="Deck2")
    ctx = with_clip_hold(ctx, "play_L", 0, ref=ref)
    assert items(ctx) == [f"{HEAD} holds[0]"]


def moved_left_hold(ctx, gap):
    return with_hold(ctx, "hold_L", point=(-0.1 + gap, TOP, HOLD_Z))


def test_calibration_half_the_limit_passes_and_double_fails():
    ctx = make_ctx()
    assert findings(moved_left_hold(ctx, CALIBRATION_PASS * LIMIT)) == []
    # the left hold is the partner of the right one, and the right one of the left
    assert items(moved_left_hold(ctx, CALIBRATION_FAIL * LIMIT)) == [f"{HEAD} holds[0]",
                                                                      f"{HEAD} holds[1]"]


def test_a_source_hold_with_no_partner_on_the_other_side_fails():
    ctx = make_ctx()
    deck = ctx.packs["kit"].pieces["Deck"]
    extra = dataclasses.replace(deck.holds[0], id="hold_R2", point=(0.15, TOP, HOLD_Z))
    ctx = with_piece(ctx, "Deck", holds=(*deck.holds, extra))
    ref = dataclasses.replace(ctx.clips["play"].holds[0].ref, id="hold_R2")
    source = ctx.clips["play"]
    added = dataclasses.replace(source.holds[0], ref=ref)
    ctx = with_clip(ctx, "play", holds=(*source.holds, added))
    assert items(ctx) == [f"{HEAD} holds[3]"]


def test_the_partner_must_have_the_swapped_side():
    ctx = with_hold(make_ctx(), "hold_L", side="R")
    assert f"{HEAD} holds[0]" in items(ctx)


# --- targets ----------------------------------------------------------------------------------


def test_targets_map_chain_to_mirrored_chain():
    ctx = make_ctx()
    assert [t.chain for t in ctx.clips["play"].targets] == ["arm.R", "arm.L"]
    assert [t.chain for t in ctx.clips["play_L"].targets] == ["arm.L", "arm.R"]
    assert findings(ctx) == []


def test_a_target_on_the_same_chain_fails():
    clip = make_ctx().clips["play_L"]
    targets = (dataclasses.replace(clip.targets[0], chain="arm.R"), clip.targets[1])
    assert items(with_clip(make_ctx(), "play_L", targets=targets)) == [f"{HEAD} targets[0]"]


def test_a_target_at_another_key_fails():
    clip = make_ctx().clips["play_L"]
    targets = (dataclasses.replace(clip.targets[0], key=6), clip.targets[1])
    assert items(with_clip(make_ctx(), "play_L", targets=targets)) == [f"{HEAD} targets[0]"]


# --- the rest ---------------------------------------------------------------------------------


def test_a_scene_with_no_mirrored_clip_has_nothing_to_measure():
    ctx = with_clip(make_ctx(), "play_L", mirror_of=None)
    assert findings(ctx) == []
    assert list(MirrorPair().plants(ctx)) == []
    # not silent: the check says it has no mirrored clip to read (a plain line, not Unmeasured)
    (line,) = MirrorPair().coverage(ctx)
    assert "not applicable" in line and "no clip has mirror_of" in line
    assert not isinstance(line, Unmeasured)


def test_a_mirrored_clip_with_no_rig_raises():
    ctx = dataclasses.replace(make_ctx(), rig_of_clip={"play": "kit:Doll"})
    with pytest.raises(ManifestError, match="play_L"):
        findings(ctx)


def test_every_item_has_a_plant():
    got = {p.item for p in MirrorPair().plants(make_ctx())}
    assert got == {f"{HEAD} holds[0]", f"{HEAD} holds[1]", f"{HEAD} holds[2]",
                   f"{HEAD} targets[0]", f"{HEAD} targets[1]"}


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = with_clip_hold(with_clip_hold(make_ctx(), "play_L", 0, who="hand.R"),
                         "play_L", 1, who="hand.L")
    assert len(findings(ctx)) == 2
    assert [f.item for f in findings(ctx, only={f"{HEAD} holds[1]"})] == [f"{HEAD} holds[1]"]


def test_a_blind_check_stops_the_run():
    class Blind(MirrorPair):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="mirror-pair"):
        run_gate([Blind()], make_ctx(), None)


def test_mirror_pair_without_mirror_is_not_applicable():
    ctx = make_ctx()
    rig = ctx.packs["kit"].pieces["Doll"].rig
    assert not any("not applicable" in c for c in MirrorPair().coverage(ctx))
    bare = with_piece(ctx, "Doll", rig=dataclasses.replace(rig, mirror=None))
    lines = [c for c in MirrorPair().coverage(bare) if "not applicable" in c]
    assert lines == ["mirror-pair: not applicable (the rig has no mirror map): kit:Doll"]
    assert not isinstance(lines[0], Unmeasured)
    # The mirrored clip has nothing to map, so it neither fails nor raises.
    assert findings(bare) == []
    assert run_gate([MirrorPair()], bare, None).status == "passed"


# --- the plants are built blind to the check -------------------------------------------------


def test_mirror_pair_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import mirror_pair

    ctx = make_ctx()
    want = sorted(u.item for u in MirrorPair().plants(ctx))
    refuses_the_core(monkeypatch, mirror_pair, "_findings", "_entries", "_hold_entry",
                     "_target_entry")
    monkeypatch.setattr(MirrorPair, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in MirrorPair().plants(ctx)) == want and len(want) == 5


def test_mirror_pair_plants_caught_on_two_stations():
    """The Deck's mirrored clip and the Bench's: each hold and target is tried and caught."""
    ctx = with_bench()
    assert findings(ctx) == []
    got = tried_and_caught(MirrorPair(), ctx)
    assert sorted(got) == sorted(f"{clip}_L mirror-of {clip} {kind}[{i}]"
                                 for clip in ("play", "rest")
                                 for kind, n in (("holds", 3), ("targets", 2)) for i in range(n))
    assert run_gate([MirrorPair()], ctx, None).status == "passed"


def test_a_plant_breaks_every_hold_so_the_check_need_not_say_which_one_it_matched():
    """The mirrored clip's holds all lose their swap, whichever one matched the source hold."""
    ctx = make_ctx()
    (unit,) = [u for u in MirrorPair().plants(ctx) if u.item == f"{HEAD} holds[0]"]
    plant = accepted(MirrorPair(), unit)
    assert plant.label.startswith("unswapped who")
    assert {h.who for h in plant.ctx.clips["play_L"].holds} == {ctx.clips["play"].holds[0].who}


# --- the contact length is the piece's own ----------------------------------------------------


def test_a_piece_with_no_view_leaves_a_side_hold_unmeasured_and_a_both_hold_judged():
    ctx = unviewed(moved_left_hold(make_ctx(), CALIBRATION_FAIL * LIMIT))
    assert findings(ctx) == []  # no contact length: nothing to compare the reflected point with
    lines = [x for x in MirrorPair().coverage(ctx) if "no view" in x]
    assert {str(x).rsplit(": ", 1)[1] for x in lines} == {f"{HEAD} holds[0]",
                                                          f"{HEAD} holds[1]"}
    assert all(isinstance(x, Unmeasured) for x in lines)
    assert run_gate([MirrorPair()], ctx, None).status == "partial"
    # source hold 2 is the both-sides hold: it needs no length and is still judged
    dropped = with_clip_hold(ctx, "play_L", 2, who="hand.R")
    assert items(dropped) == [f"{HEAD} holds[2]"]
    assert not [u for u in MirrorPair().plants(ctx) if u.item.endswith("holds[0]")]


def test_the_reach_follows_the_pieces_view():
    ctx = moved_left_hold(make_ctx(), 3 * LIMIT)
    assert items(ctx) == [f"{HEAD} holds[0]", f"{HEAD} holds[1]"]
    assert items(viewed(ctx, 5 * LIMIT)) == []
