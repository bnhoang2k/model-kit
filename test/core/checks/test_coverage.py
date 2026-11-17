"""Check coverage: nothing exported goes unmeasured.

Two rules. Every exported part (every part with geometry in the scene) must be measured by at
least one of the checks named in the run, as their `measured` hooks say (items
`<pack>:<piece> <part>`). And a body that sinks into a placed or carried piece (a contact that
overlap's test A finds) needs a clip hold or the seat on that piece at that moment (items
`<clip> <pack>:<piece> no-hold`).

`Coverage` takes the other checks of the run. The gate builds it last, from those.
"""

import dataclasses
import sys

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import coverage as coverage_module
from model_kit.core.checks import overlap as overlap_module
from model_kit.core.checks.coverage import Coverage
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.checks.topology import Topology
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scope import Scope

from ..shapes import box
from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base, refuses_the_core, tried_and_caught, two_pieces, with_parts
from .test_overlap import (
    CUSHION_TOP,
    HAND,
    ROD_AT,
    SKIN,
    THRONE_AT,
    at_block,
    hold,
    put,
    with_clip,
)
from .test_overlap import make_ctx as clip_ctx

BLOCK_ITEM = "play kit:Block no-hold"
THRONE_ITEM = "play kit:Throne no-hold"
ROD_ITEM = "play kit:Rod no-hold"
IN_BLOCK = 0.001  # a contact is any depth: this is far under the skin
HOLD_ON_BLOCK = "kit:Block.top_R"


def findings(ctx, others=(), only=None, samples=frozenset()):
    return Coverage(others).run(ctx, only=only, samples=samples)


def items(ctx, others=None):
    return sorted(f.item for f in findings(ctx, [Floaters()] if others is None else others))


def contact_scene(depth=IN_BLOCK, **fields):
    """The hand inside the block at sample 1 (frame 2), with these clip fields changed."""
    ctx = put(clip_ctx(), "arm_R", HAND, 1, at_block(depth=depth))
    return with_clip(ctx, **fields) if fields else ctx


def test_the_check_is_registered_and_is_a_scene_check():
    assert gate.CHECKS["coverage"] is Coverage
    assert Coverage.name == "coverage"
    assert Coverage.kinds == frozenset()
    assert Coverage().others == ()


# --- exported parts ----------------------------------------------------------------------------


def test_a_part_that_some_check_measured_passes():
    ctx = piece_ctx(post_on_base())
    assert findings(ctx, [Floaters(), Topology()]) == []
    assert findings(ctx, [Floaters()]) == []


def test_a_part_that_no_check_measured_fails():
    ctx = piece_ctx(post_on_base())
    found = findings(ctx, [])
    assert sorted(f.item for f in found) == ["kit:Thing base", "kit:Thing post"]
    assert all(f.check == "coverage" and "no check measured" in f.detail for f in found)


def test_the_parts_each_check_measured_are_pooled():
    # crossings measures a piece with two solids only, floaters every part, so the printed
    # mark is measured by floaters alone
    parts = {**post_on_base(), "mark": box((0.0, 0.0, 0.0), (0.01, 0.01, 0.01))}
    ctx = piece_ctx(parts, {"mark": {"role": "decal"}})
    assert sorted(f.item for f in findings(ctx, [Crossings()])) == ["kit:Thing mark"]
    assert findings(ctx, [Crossings(), Floaters()]) == []


def test_a_lone_part_that_crossings_cannot_test_is_unmeasured_without_other_checks():
    ctx = piece_ctx({"base": post_on_base()["base"]})
    assert items(ctx, [Crossings()]) == ["kit:Thing base"]
    assert items(ctx, [Crossings(), Topology()]) == []


def test_a_check_with_no_hook_adds_nothing():
    ctx = piece_ctx(post_on_base())
    assert len(findings(ctx, [MustTouch()])) == 2


def test_a_branching_piece_no_check_reads_is_unmeasured():
    ctx = piece_ctx(post_on_base(), also={"Tree": post_on_base()})
    got = sorted(f.item for f in findings(ctx, [Floaters()]))
    assert got == ["kit:Tree base", "kit:Tree post"]  # floaters leaves a branching piece out


def test_a_part_with_no_triangles_is_still_an_exported_part():
    ctx = with_parts(piece_ctx(post_on_base()), ghost=np.zeros((0, 3, 3)))
    assert items(ctx) == ["kit:Thing ghost"]


def test_the_part_plant_is_a_mesh_no_manifest_lists_and_is_caught():
    ctx = piece_ctx(post_on_base())
    check = Coverage([Floaters(), Topology()])
    (unit,) = check.plants(ctx)
    (plant,) = unit.candidates
    assert plant.item.startswith("kit:")
    assert plant.ctx.pieces.keys() > ctx.pieces.keys()  # one more exported mesh
    assert [f.item for f in check.run(plant.ctx, only=plant.scope)] == [plant.item]
    assert run_gate([check], ctx, None).status == "passed"


def test_a_check_that_claims_every_part_leaves_coverage_blind():
    class Liar(Floaters):
        def measured(self, ctx):
            return {f"{key} {part}" for key, geom in ctx.pieces.items() for part in geom.parts}

    with pytest.raises(BlindCheck, match="coverage"):
        run_gate([Coverage([Liar()])], piece_ctx(post_on_base()), None)


def test_no_geometry_means_no_exported_part():
    ctx = piece_ctx(post_on_base())
    bare = dataclasses.replace(ctx, pieces={})
    assert findings(bare, []) == []
    (unit,) = Coverage([]).plants(bare)  # the check can still be shown to see a mesh
    (plant,) = unit.candidates
    assert [f.item for f in Coverage([]).run(plant.ctx, only=plant.scope)] == [plant.item]


def test_a_part_no_mount_is_near_stays_unmeasured_under_frames():
    from model_kit.core.checks.frames import Frames
    from model_kit.core.scene import PieceGeom

    from .manifest_scenes import make_ctx as deck_ctx

    ctx = deck_ctx()
    geom = ctx.pieces["kit:Deck"]
    knob = box((-0.01, 0.03, -0.01), (0.01, 0.05, 0.01))
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Deck": PieceGeom(
        "Deck", {**geom.parts, "knob": knob})})
    assert sorted(f.item for f in findings(ctx, [Frames()])) == ["kit:Deck knob"]


# --- contacts without a hold --------------------------------------------------------------------


def test_a_clean_scene_has_no_contact():
    ctx = clip_ctx()
    assert items(ctx) == []
    assert run_gate([Coverage([Floaters()])], ctx, None).status == "passed"


def test_a_body_inside_a_placed_piece_with_no_hold_fails():
    (found,) = findings(contact_scene(), [Floaters()])
    assert (found.check, found.item) == ("coverage", BLOCK_ITEM)
    assert "frame 2" in found.detail and "hold" in found.detail


def test_a_contact_one_vertex_deep_is_a_contact_at_any_depth():
    assert items(contact_scene(depth=SKIN / 100)) == [BLOCK_ITEM]
    assert items(contact_scene(depth=-0.01)) == []  # above the block: no contact


def test_a_hold_on_the_piece_at_that_moment_excuses_the_contact():
    assert items(contact_scene(holds=(hold(HOLD_ON_BLOCK),))) == []
    at_frame_2 = hold(HOLD_ON_BLOCK, when=((2.0, 2.0),))
    assert items(contact_scene(holds=(at_frame_2,))) == []


def test_a_hold_in_another_window_does_not_excuse_it():
    late = hold(HOLD_ON_BLOCK, when=((3.0, 4.0),))
    assert items(contact_scene(holds=(late,))) == [BLOCK_ITEM]


def test_a_hold_on_another_piece_does_not_excuse_it():
    on_rod = hold("kit:Rod.grip_R")
    assert items(contact_scene(holds=(on_rod,))) == [BLOCK_ITEM]


def test_the_seat_counts_as_a_hold_on_the_throne():
    ctx = put(clip_ctx(seat=True), "arm_R", HAND, 1,
              (THRONE_AT[0], CUSHION_TOP - IN_BLOCK, THRONE_AT[2]))
    assert items(ctx) == []
    unseated = with_clip(ctx, seat=None)
    assert items(unseated) == [THRONE_ITEM]


def test_a_carried_piece_is_a_piece_too():
    ctx = put(clip_ctx(), "arm_R", HAND, 1, (ROD_AT[0], ROD_AT[1], ROD_AT[2] + 0.1))
    assert items(ctx) == [ROD_ITEM]
    assert items(with_clip(ctx, holds=(hold("kit:Rod.grip_R"),))) == []


def test_a_carried_piece_that_is_not_shown_is_not_touched():
    ctx = put(clip_ctx(show=((3.0, 4.0),)), "arm_R", HAND, 1,
              (ROD_AT[0], ROD_AT[1], ROD_AT[2] + 0.1))
    assert items(ctx) == []  # the rod is hidden at frame 2


def test_a_scoped_run_reads_only_the_clip_and_samples_it_names():
    ctx = contact_scene()
    assert [f.item for f in findings(ctx, [Floaters()], only={BLOCK_ITEM})] == [BLOCK_ITEM]
    assert findings(ctx, [Floaters()], only={"play kit:Rod no-hold"}) == []
    assert findings(ctx, [Floaters()], only={"other kit:Block no-hold"}) == []
    assert findings(ctx, [Floaters()], samples=frozenset({0, 2, 3})) == []
    assert len(findings(ctx, [Floaters()], samples=frozenset({1}))) == 1


def test_the_contact_plant_sinks_a_vertex_into_a_piece_and_takes_its_holds_away():
    held = contact_scene(holds=(hold(HOLD_ON_BLOCK),))
    check = Coverage([Floaters()])
    units = [u for u in check.plants(held) if u.item.endswith("no-hold")]
    assert sorted(u.item for u in units) == [BLOCK_ITEM, ROD_ITEM]  # one per piece, not one per clip
    plant = units[0].candidates[0]
    assert plant.samples and plant.item.startswith("play kit:")
    assert [f.item for f in check.run(plant.ctx, only=plant.scope, samples=plant.samples)] == [
        plant.item]
    assert run_gate([check], held, None).status == "passed"


def test_a_scene_whose_clip_places_no_piece_has_no_contact_to_test():
    ctx = clip_ctx(block=False, rod=False)
    assert items(ctx) == []
    lines = Coverage([Floaters()]).coverage(ctx)
    assert any("play" in line and "not applicable" in line for line in lines), lines
    assert not any(isinstance(line, Unmeasured) for line in lines)


def test_a_clip_with_no_samples_is_unmeasured_and_the_run_partial():
    ctx = dataclasses.replace(clip_ctx(), samples={})
    lines = Coverage([Floaters()]).coverage(ctx)
    assert [line for line in lines if isinstance(line, Unmeasured)] == [
        "coverage: skipped (no samples): play"]
    assert run_gate([Coverage([Floaters()])], ctx, None).status == "partial"


def test_a_clip_with_no_body_shells_is_unmeasured():
    ctx = clip_ctx()
    bare = dataclasses.replace(ctx, rests={})
    lines = Coverage([Floaters()]).coverage(bare)
    assert any(isinstance(line, Unmeasured) and "no body shells" in line for line in lines), lines


def test_a_piece_with_no_geometry_in_the_scene_is_unmeasured_for_contacts():
    ctx = clip_ctx()
    pieces = {k: v for k, v in ctx.pieces.items() if k != "kit:Block"}
    lines = Coverage([Floaters()]).coverage(dataclasses.replace(ctx, pieces=pieces))
    assert any(isinstance(line, Unmeasured) and "no geometry" in line and "kit:Block" in line
               for line in lines), lines


def test_a_contact_scene_is_still_clean_for_the_clip_checks_below_the_budget():
    # coverage and overlap are different rules: a graze under the skin passes overlap
    from model_kit.core.checks.overlap import Overlap

    ctx = contact_scene(depth=SKIN / 2)
    assert Overlap().run(ctx) == []
    assert items(ctx) == [BLOCK_ITEM]


# --- how the gate builds it ---------------------------------------------------------------------


def test_the_gate_builds_coverage_last_from_the_other_checks_of_the_run():
    checks = gate.build_checks(["coverage", "floaters", "reach"])
    assert [c.name for c in checks] == ["floaters", "reach", "coverage"]
    assert checks[-1].others == tuple(checks[:-1])
    assert gate.build_checks(["coverage"])[0].others == ()
    assert [c.name for c in gate.build_checks(["reach"])] == ["reach"]


def test_the_default_run_names_coverage_last_with_every_other_check():
    checks = gate.build_checks(list(gate.CHECKS))
    assert checks[-1].name == "coverage"
    assert [c.name for c in checks[:-1]] == [n for n in gate.CHECKS if n != "coverage"]
    assert len(checks[-1].others) == len(gate.CHECKS) - 1


def test_the_cable_and_the_takes_part_count_as_measured_when_their_checks_ran():
    from model_kit.core.checks.connectors import Connectors
    from model_kit.core.checks.receiver import Receiver

    from .assembly_scenes import make_ctx as lead_ctx

    ctx = lead_ctx()
    bare = {f.item for f in Coverage([]).run(ctx)}
    assert {"kit:Lead cable", "kit:Lamp pin"} <= bare
    read = {f.item for f in Coverage([Connectors(), Receiver()]).run(ctx)}
    assert not {"kit:Lead cable", "kit:Lamp pin"} & read
    assert "kit:Lamp shade" in read
    assert not {"kit:Lead plug", "kit:Lead pin", "kit:Rack frame"} & read  # the joints read them


def test_the_no_hold_plant_of_a_seated_sit_clip_is_a_legal_clip_and_is_caught():
    """The plant takes the seat away. A `sit` clip with no seat cannot be built, so the plant's
    clip is made postureless as well; the run must not raise, and the plant must still be caught."""
    from ..bodies.humanoid import humanoid_scene

    ctx = humanoid_scene()
    assert any(c.posture == "sit" and c.seat is not None for c in ctx.clips.values())
    result = run_gate([Coverage([Floaters()])], ctx, None)
    assert result.status in ("passed", "partial")
    check = Coverage([Floaters()])
    units = [u for u in check.plants(ctx) if u.item.endswith("no-hold") and "sit_" in u.item]
    assert units
    plants = [p for u in units for p in u.candidates]
    for plant in plants:
        assert [f.item for f in check.run(plant.ctx, only=plant.scope, samples=plant.samples)] == [
            plant.item]


SCAN_NAMES = ("unheld_contacts", "part_contacts")

# --- the plants are built without the scan ------------------------------------------------


def test_the_contact_plants_never_call_the_scan(monkeypatch):
    check = Coverage([Floaters()])
    ctx = contact_scene()
    # the scan, and the scan of a body vertex in a part that overlap's test A shares with it, are
    # refused in every module of the checks that holds either name, so no import path is missed
    holders = [m for name, m in list(sys.modules.items())
               if name.startswith("model_kit.core.checks.") and m is not None
               and any(hasattr(m, scan) for scan in SCAN_NAMES)]
    assert coverage_module in holders and overlap_module in holders
    for module in holders:
        refuses_the_core(monkeypatch, module, *[n for n in SCAN_NAMES if hasattr(module, n)])
    assert {u.item for u in check.plants(ctx)} >= {BLOCK_ITEM, ROD_ITEM}


def test_the_contact_scan_has_one_home():
    """`unheld_contacts` is coverage's; overlap and coverage share the one scan of a body
    vertex in a piece part, `overlap.part_contacts`, and nothing else keeps a copy."""
    assert not hasattr(overlap_module, "unheld_contacts")
    assert not hasattr(overlap_module, "unheld_contact_plant")
    assert not hasattr(overlap_module, "has_contact_test")
    assert coverage_module.part_contacts is overlap_module.part_contacts


def test_the_contact_plants_are_caught_on_two_clips():
    from ..bodies.humanoid import humanoid_scene

    for ctx in (clip_ctx(), humanoid_scene()):
        items_tried = tried_and_caught(Coverage([Floaters()]), ctx)
        assert any(i.endswith(" no-hold") for i in items_tried), items_tried


def test_a_piece_the_clip_never_shows_is_not_applicable_and_a_shown_one_is_planted():
    ctx = clip_ctx(show=((9.0, 10.0),))  # the rod is carried, and hidden at every sample
    units = {u.item: u for u in Coverage([Floaters()]).plants(ctx)}
    assert units[ROD_ITEM].not_applicable == "the piece is never shown in the clip"
    assert units[ROD_ITEM].candidates == ()
    assert units[BLOCK_ITEM].candidates
    verdict = run_gate([Coverage([Floaters()])], ctx, None)
    assert "coverage: not applicable (the piece is never shown in the clip): " + ROD_ITEM in (
        verdict.coverage)


def test_a_shown_piece_with_no_room_for_a_plant_stops_the_gate_not_a_not_applicable_line(
        monkeypatch):
    monkeypatch.setattr(overlap_module.PiecePart, "room", lambda self: [])
    ctx = clip_ctx()
    units = {u.item: u for u in Coverage([Floaters()]).plants(ctx)}
    assert not units[BLOCK_ITEM].candidates and not units[BLOCK_ITEM].not_applicable
    with pytest.raises(BlindCheck, match="no-hold"):
        run_gate([Coverage([Floaters()])], ctx, None)


def test_a_piece_with_no_closed_part_has_no_unit_and_a_line_says_why():
    ctx = clip_ctx()
    open_block = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Block": dataclasses.replace(
        ctx.pieces["kit:Block"], parts={"body": ctx.pieces["kit:Block"].parts["body"][:-1]})})
    assert BLOCK_ITEM not in {u.item for u in Coverage([Floaters()]).plants(open_block)}
    assert any(isinstance(line, Unmeasured) and "open part" in line
               for line in Coverage([Floaters()]).coverage(open_block))


def test_a_body_with_no_closed_shell_still_gets_a_contact_plant():
    """The scan judges every body vertex, so a body whose shells are all open is judged and is
    planted (vertex 0): the unit never says `no closed shell` for a judged body."""
    ctx = clip_ctx()
    rest = ctx.rests["kit:Doll"]
    open_faces = {part: faces[:-1] for part, faces in rest.faces.items()}
    opened = dataclasses.replace(
        ctx, rests={"kit:Doll": dataclasses.replace(rest, faces=open_faces)},
        samples={"play": dataclasses.replace(ctx.samples["play"], faces=open_faces)})
    check = Coverage([Floaters()])
    units = {u.item: u for u in check.plants(opened)}
    assert units[BLOCK_ITEM].candidates and not units[BLOCK_ITEM].not_applicable
    assert "closed shell" not in " ".join(u.not_applicable for u in units.values())
    assert BLOCK_ITEM in tried_and_caught(check, opened)


def test_a_piece_the_clip_places_twice_has_one_unit_of_its_first_copy_s_scenes():
    from model_kit.core.manifest import Placement

    from .test_overlap import BLOCK_AT
    ctx = clip_ctx()
    clip = ctx.clips["play"]
    placed = {**clip.floor_gear.pieces,
              "Block2": Placement("Block2", "Block", "kit", (BLOCK_AT[0] + 0.5, *BLOCK_AT[1:]),
                                  (0.0, 0.0, 0.0, 1.0))}
    twice = dataclasses.replace(ctx, clips={"play": dataclasses.replace(
        clip, floor_gear=dataclasses.replace(clip.floor_gear, pieces=placed))})
    prepared = overlap_module.prepare_clip(twice, twice.clips["play"])
    assert [i.label for i in prepared.insts].count("kit:Block") == 2
    units = [u for u in Coverage([Floaters()]).plants(twice) if u.item == BLOCK_ITEM]
    # the first, middle and last sample, each at the fractions of the room: one copy's scenes
    assert len(units) == 1
    assert 0 < len(units[0].candidates) <= 3 * len(overlap_module.ROOM_FRACTIONS)
    assert len({u.item for u in Coverage([Floaters()]).plants(twice)}) == len(
        Coverage([Floaters()]).plants(twice))


def test_a_scoped_coverage_skips_dropped_pieces_and_still_catches_its_ghost():
    ctx = two_pieces(decal=True)
    scope = Scope(frozenset({"kit:Thing"}))
    checks = [Crossings(), Coverage([Crossings()], scope)]
    verdict = run_gate(checks, ctx, None, scope=scope)  # a blind ghost plant would raise
    text = "\n".join(verdict.lines)
    assert "kit:Thing label" in text
    assert "kit:Other label" not in text
