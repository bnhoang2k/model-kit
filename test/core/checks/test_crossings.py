import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from model_kit.core import gate, geom
from model_kit.core.allow import Allowlist
from model_kit.core.checks import crossings, pieces
from model_kit.core.checks.crossings import Crossings
from model_kit.core.errors import BlindCheck, Unmeasurable
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import piece_key
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS, CROSS_OF_PART_THICKNESS

from ..bodies.furniture import chair, table, tenon
from ..shapes import box, cylinder
from .bearing_scenes import SUNK_PIECES, sunk_post
from .piece_scenes import (
    CROSS,
    chunk_and_block,
    make_ctx,
    plants_caught,
    post_on_base,
    refuses_the_core,
    sheet_and_knob,
    strip_and_post,
    tried_and_caught,
)

ITEM = "kit:Thing base x post"


def joined():
    """The post sunk two limits into the base (a designed join), with a cap flush on the post's
    top, so a clean pair is left for a plant."""
    return make_ctx({**post_on_base(sink=CALIBRATION_FAIL * CROSS),
                     "cap": box((-0.06, 0.10, -0.06), (0.06, 0.12, 0.06))})


def findings(ctx, only=None):
    return Crossings().run(ctx, only=only)


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["crossings"] is Crossings
    assert Crossings.kinds == frozenset({"assembly"})


def test_a_clean_piece_passes_and_its_plants_are_caught():
    ctx = make_ctx(post_on_base())
    assert findings(ctx) == []
    assert plants_caught(Crossings(), ctx)
    assert run_gate([Crossings()], ctx, None).status == "passed"


def test_a_sunk_part_fails_naming_the_pair_with_its_names_sorted():
    found = findings(make_ctx(post_on_base(sink=CALIBRATION_FAIL * CROSS)))
    assert [(f.check, f.item) for f in found] == [("crossings", ITEM)]
    assert "share" in found[0].detail


def test_calibration_half_the_limit_passes_and_double_fails():
    assert findings(make_ctx(post_on_base(sink=CALIBRATION_PASS * CROSS))) == []
    assert len(findings(make_ctx(post_on_base(sink=CALIBRATION_FAIL * CROSS)))) == 1


def test_a_gap_is_not_a_crossing():
    assert findings(make_ctx(post_on_base(lift=0.05))) == []


def test_a_plant_is_the_clean_parts_copy_sunk_two_limits_into_its_neighbour():
    units = list(Crossings().plants(make_ctx(post_on_base())))
    assert len(units) == 1
    (unit,) = units
    assert unit.item == "kit:Thing base x post_plant"  # the post is the smaller part
    plant = unit.candidates[0]
    parts = plant.ctx.pieces[piece_key("kit", "Thing")].parts
    assert len(parts) == 3  # a copy is added: the original parts stay
    assert [f.item for f in findings(plant.ctx, only=plant.scope)] == [unit.item]


@pytest.mark.parametrize("scene,copied", [(sheet_and_knob, "knob_plant"),
                                          (strip_and_post, "post_plant")])
def test_the_plant_copies_the_cheapest_part_not_the_fewest_triangles(scene, copied):
    ctx = make_ctx(scene())
    (unit,) = Crossings().plants(ctx)
    assert copied in unit.item
    assert "sheet_plant" not in unit.item and "strip_plant" not in unit.item
    assert plants_caught(Crossings(), ctx)


@pytest.mark.parametrize("scene,item", [
    (sheet_and_knob, "kit:Thing base x knob_plant"),
    (chunk_and_block, "kit:Thing block_plant x slab"),  # the thin film makes the chunk dear
])
def test_the_cost_counts_the_hosts_samples_inside_the_copy(scene, item):
    (unit,) = Crossings().plants(make_ctx(scene()))
    assert unit.item == item


def test_equal_parts_tie_to_the_first_in_part_order():
    ctx = make_ctx({"left": box((-0.5, 0.0, -0.1), (-0.2, 0.004, 0.1)),
                    "right": box((0.2, 0.0, -0.1), (0.5, 0.004, 0.1)),
                    "base": box((-0.6, -0.05, -0.2), (0.6, 0.0, 0.2))})
    (first,) = Crossings().plants(ctx)
    (second,) = Crossings().plants(ctx)
    assert "left_plant" in first.item
    assert first.item == second.item


def test_a_plant_cuts_by_the_planted_depth_so_the_limit_decides():
    (unit,) = Crossings().plants(make_ctx(post_on_base()))
    plant = unit.candidates[0]
    (found,) = findings(plant.ctx, only=plant.scope)
    depth = float(found.detail.split("share ")[1].split(" ")[0])
    assert depth > CROSS


def test_a_pair_that_fails_already_still_gets_a_plant_on_its_own_copy():
    ctx = make_ctx(post_on_base(sink=CALIBRATION_FAIL * CROSS))
    assert [f.item for f in findings(ctx)] == [ITEM]
    (unit,) = Crossings().plants(ctx)
    assert unit.item == "kit:Thing base x post_plant" != ITEM  # a new part, so a new item
    assert plants_caught(Crossings(), ctx)


def test_a_scoped_run_measures_only_the_items_it_is_given():
    parts = {**post_on_base(sink=CALIBRATION_FAIL * CROSS),
             "pin": box((0.03, 0.0, 0.03), (0.07, 0.04, 0.07))}  # a pin through the base
    ctx = make_ctx(parts)
    assert {f.item for f in findings(ctx)} == {ITEM, "kit:Thing base x pin"}
    assert [f.item for f in findings(ctx, only={ITEM})] == [ITEM]
    assert findings(ctx, only={"kit:Thing base x nothing"}) == []


def test_printed_parts_are_not_measured():
    parts = post_on_base(sink=CALIBRATION_FAIL * CROSS)
    for role in ("eyes", "decal"):
        ctx = make_ctx(parts, {"post": {"role": role}})
        assert findings(ctx) == []
        assert any("not applicable" in line and "post" in line
                   for line in Crossings().coverage(ctx))


def test_a_designed_join_is_an_allowlist_line_with_a_reason(tmp_path):
    ctx = joined()
    path = tmp_path / "a.allow"
    path.write_text(f"crossings | {ITEM} | the post is pinned into its socket\n")
    verdict = run_gate([Crossings()], ctx, Allowlist.load(path))
    assert verdict.status == "passed"
    assert "the post is pinned into its socket" in verdict.allowed[0]


def test_a_join_that_matches_nothing_is_stale_and_fails(tmp_path):
    path = tmp_path / "a.allow"
    path.write_text(f"crossings | {ITEM} | the post is pinned into its socket\n")
    verdict = run_gate([Crossings()], make_ctx(post_on_base()), Allowlist.load(path))
    assert verdict.status == "failed" and verdict.failed == 1
    assert "stale" in verdict.lines[0]


def test_a_join_with_no_reason_allows_nothing(tmp_path):
    ctx = joined()
    path = tmp_path / "a.allow"
    path.write_text(f"crossings | {ITEM} |\n")
    verdict = run_gate([Crossings()], ctx, Allowlist.load(path))
    assert verdict.status == "failed" and verdict.failed == 2  # the crossing, and the bad line


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = make_ctx(post_on_base(), also={"Tree": post_on_base(sink=0.05)})
    check = Crossings()
    assert findings(ctx) == []  # its crossings are not this check's to find
    (line,) = check.coverage(ctx)
    assert line == "crossings: not applicable (branching piece): kit:Tree"
    assert not isinstance(line, Unmeasured)  # the branching checks measure it
    assert run_gate([check], ctx, None).status == "passed"


def test_a_piece_with_no_geometry_in_the_scene_is_unmeasured():
    ctx = dataclasses.replace(make_ctx(post_on_base()), pieces={})
    (line,) = Crossings().coverage(ctx)
    assert line == "crossings: skipped (no geometry): kit:Thing"
    assert isinstance(line, Unmeasured)
    assert run_gate([Crossings()], ctx, None).status == "partial"


def test_a_pair_far_apart_is_planted_by_the_move_onto_its_centre_and_caught():
    # two parts far apart: the short sinks miss, the last move puts the copy onto the other part
    ctx = make_ctx({"base": box((0, 0, 0), (0.1, 0.1, 0.1)), "far": box((5, 0, 0), (5.1, 0.1, 0.1))})
    (unit,) = Crossings().plants(ctx)
    assert unit.item == "kit:Thing base x far_plant" and unit.not_applicable == ""
    caught = [i for i, p in enumerate(unit.candidates)
              if [f.item for f in findings(p.ctx, only=p.scope)] == [unit.item]]
    assert caught == [len(unit.candidates) - 1]
    assert tried_and_caught(Crossings(), ctx) == [unit.item]
    assert run_gate([Crossings()], ctx, None).status == "passed"


def test_a_piece_of_one_solid_part_needs_no_plant():
    ctx = make_ctx({"base": box((0, 0, 0), (0.1, 0.1, 0.1))})
    (line,) = Crossings().coverage(ctx)
    assert line == "crossings: not applicable (fewer than two solid parts): kit:Thing"
    assert not isinstance(line, Unmeasured)
    assert run_gate([Crossings()], ctx, None).status == "passed"


def test_a_pin_driven_into_a_plate_shares_how_far_it_is_driven():
    # a 0.05 pin driven 0.0633 into a plate: the shortest move that frees it is straight back out,
    # by 0.0633, and that is over half the pin's width
    plate = box((-0.5, -0.1, -0.5), (0.5, 0.1, 0.5))
    pin = box((-0.025, 0.1 - 0.0133 - 0.05, -0.025), (0.025, 0.6 - 0.0133 - 0.05, 0.025))
    (found,) = findings(make_ctx({"plate": plate, "pin": pin}))
    assert float(found.detail.split("share ")[1].split(" ")[0]) == pytest.approx(0.0633, rel=0.01)


def test_a_blind_check_stops_the_run():
    class Blind(Crossings):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="crossings"):
        run_gate([Blind()], make_ctx(post_on_base()), None)


def test_parts_with_no_manifest_row_are_still_measured():
    ctx = make_ctx(post_on_base(sink=CALIBRATION_FAIL * CROSS))
    piece = ctx.packs["kit"].pieces["Thing"]
    bare = dataclasses.replace(piece, parts=piece.parts[:1])  # the post has no row
    pack = dataclasses.replace(ctx.packs["kit"], pieces={"Thing": bare})
    ctx = dataclasses.replace(ctx, packs={"kit": pack})
    assert [f.item for f in findings(ctx)] == [ITEM]


def test_the_thinner_part_sets_the_limit():
    # The same 0.015 sink fails into a 20 mm base (limit 10 mm) and passes into an 80 mm one
    # (the post is 40 mm wide, so its half is 20 mm): the limit follows the parts, not the pack.
    thin = post_on_base(sink=0.015)
    thick = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.08, 0.1)),
             "post": box((-0.02, 0.065, -0.02), (0.02, 0.165, 0.02))}
    assert len(findings(make_ctx(thin))) == 1
    assert findings(make_ctx(thick)) == []


def test_a_bolt_straight_through_a_plate_fails_until_it_is_declared(tmp_path):
    # a 12 mm bolt through an 11 mm plate with no hole shares the plate's whole thickness
    plate = box((-0.15, 0.0, 0.0), (0.15, 0.3, 0.011))
    bolt = box((-0.006, 0.1, -0.005), (0.006, 0.112, 0.016))
    ctx = make_ctx({"plate": plate, "bolt": bolt})
    (found,) = findings(ctx)
    assert found.item == "kit:Thing bolt x plate"
    path = tmp_path / "a.allow"
    path.write_text(f"crossings | {found.item} | the bolt is driven through the plate it holds\n")
    assert run_gate([Crossings()], ctx, Allowlist.load(path)).failed == 0


def test_a_rod_pushed_end_first_is_as_thick_as_it_is_wide():
    # a 10 mm rod, 300 mm long, driven 12 mm into a block: the shared space is the rod's width,
    # 10 mm, and that is twice the half of the thickness a rod's rim reads
    block = box((-0.5, -0.5, -0.5), (0.5, 0.0, 0.5))
    rod = cylinder((0.0, 0.0), 0.005, -0.012, 0.288)
    (found,) = findings(make_ctx({"block": block, "rod": rod}))
    assert found.item == "kit:Thing block x rod"
    assert float(found.detail.split("thickness ")[1].split(")")[0]) < 0.02  # not the 0.3 length


# --- shared space: the cases a distance to the nearest surface cannot tell apart -----------------


def cubes(offset):
    return {"a": box((0, 0, 0), (1, 1, 1)), "b": box((offset, 0, 0), (1 + offset, 1, 1))}


def test_cubes_that_overlap_by_90_percent_fail():
    (found,) = findings(make_ctx(cubes(0.1)))
    # the sample lattice reads the shared width up to one spacing (0.26 here) short
    assert 0.8 < float(found.detail.split("share ")[1].split(" ")[0]) <= 0.9


def test_a_graze_of_a_millimetre_passes_and_a_cube_apart_is_clear():
    assert findings(make_ctx(cubes(0.999))) == []  # 1 mm shared of 1 m: far under half
    assert findings(make_ctx(cubes(1.0))) == []  # face to face
    assert findings(make_ctx(cubes(1.5))) == []  # apart


def test_a_40_mm_leg_pushed_through_a_40_mm_seat_fails():
    seat = box((-0.2, 0.40, -0.2), (0.2, 0.44, 0.2))
    leg = box((-0.02, 0.0, -0.02), (0.02, 0.46, 0.02))  # 20 mm proud of the seat's top
    (found,) = findings(make_ctx({"seat": seat, "leg": leg}))
    assert found.item == "kit:Thing leg x seat"


def test_a_leg_10_mm_into_a_20_mm_seat_is_at_the_limit():
    seat = box((-0.2, 0.41, -0.2), (0.2, 0.43, 0.2))
    leg = box((-0.02, 0.0, -0.02), (0.02, 0.42, 0.02))  # flush under the seat, 10 mm into it
    assert findings(make_ctx({"seat": seat, "leg": leg})) == []


@pytest.mark.parametrize("radius", [0.008, 0.01])
def test_a_rod_through_a_plate_with_no_hole_fails_whatever_its_width(radius):
    # a 16 mm rod and a 20 mm rod, through a 20 mm plate
    plate = cylinder((0.0, 0.0), 0.15, -0.01, 0.01)
    rod = cylinder((0.0, 0.0), radius, -0.2, 0.2)
    (found,) = findings(make_ctx({"plate": plate, "rod": rod}))
    assert found.item == "kit:Thing plate x rod"


def test_two_near_equal_parts_are_plantable():
    # a stick of a shaft and a tip of about the same width, as a held stick is built
    parts = {"shaft": cylinder((0.0, 0.0), 0.010, 0.0, 0.19),
             "tip": cylinder((0.0, 0.0), 0.012, 0.19, 0.215)}
    ctx = make_ctx(parts)
    assert findings(ctx) == []
    assert plants_caught(Crossings(), ctx)
    assert not any("not applicable (no two" in line for line in Crossings().coverage(ctx))


# --- true depth: the pierced cap, the drumstick, the tenon -------------------------------------


def cap_and_rod(hole=0.0):
    """A full disc 4 mm thick with a rod straight through its middle. With `hole`, the disc is a
    ring of that inner radius around the rod, as a coupler is once it is fixed."""
    return {"cap": cylinder((0.0, 0.0), 0.1, -0.002, 0.002, hole=hole),
            "rod": cylinder((0.0, 0.0), 0.01, -0.3, 0.3)}


def test_pierced_cap_fails():
    # the rod stands inside the cap to its own radius, 10 mm, far over half the cap's 4 mm
    (found,) = findings(make_ctx(cap_and_rod()))
    assert found.item == "kit:Thing cap x rod"
    assert float(found.detail.split("thickness ")[1].split(")")[0]) == pytest.approx(0.004, rel=0.2)
    # the rod shares at least the cap's whole thickness (no move shorter than that frees it)
    assert float(found.detail.split("share ")[1].split(" ")[0]) >= 0.004


def test_the_same_rod_through_an_annulus_passes():
    ctx = make_ctx(cap_and_rod(hole=0.02))
    assert findings(ctx) == []


def drum_head_and_stick():
    """A drum head, a disc 3 mm thick, with a held stick standing on it by its tip."""
    return {"head": cylinder((0.0, 0.0), 0.15, 0.0, 0.003),
            "stick": cylinder((0.0, 0.0), 0.007, 0.003, 0.303)}


def test_drumstick_plant_caught():
    # a 14 mm stick on a 3 mm head: the fixed pack-wide limit could never fail it (its faces are
    # too narrow for a cut to run deep), and the true depth does
    ctx = make_ctx(drum_head_and_stick())
    assert findings(ctx) == []
    (unit,) = Crossings().plants(ctx)
    assert unit.item.startswith("kit:Thing ") and "head" in unit.item
    assert plants_caught(Crossings(), ctx)


def test_declared_tenon_is_quiet(tmp_path):
    ctx = make_ctx(tenon().geometry(0.0))
    path = tmp_path / "a.allow"
    path.write_text("crossings | kit:Thing post x rail | the post's tenon is driven into the "
                    "rail's mortise, which the rail's mesh does not cut\n")
    verdict = run_gate([Crossings()], ctx, Allowlist.load(path))
    assert verdict.status == "passed" and verdict.failed == 0  # the cap is left to plant on
    assert "mortise" in verdict.allowed[0]


def test_undeclared_tenon_fails():
    (found,) = findings(make_ctx(tenon().geometry(0.0)))
    assert found.item == "kit:Thing post x rail"
    verdict = run_gate([Crossings()], make_ctx(tenon().geometry(0.0)), None)
    assert verdict.status == "failed" and verdict.failed == 1 and verdict.allowed == []
    assert verdict.coverage == []  # the cap gives the plant, so nothing is left unmeasured


def test_crossings_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import crossings

    ctx = make_ctx(post_on_base())
    refuses_the_core(monkeypatch, crossings, "measure", "_cuts", "_touch")
    monkeypatch.setattr(crossings.Crossings, "run", lambda *a, **k: 1 / 0)
    assert [u.item for u in Crossings().plants(ctx)] == ["kit:Thing base x post_plant"]


def test_crossings_plants_caught_on_two_pieces():
    for name, ctx in (("post on a base", make_ctx(post_on_base())),
                      ("drum head and stick", make_ctx(drum_head_and_stick()))):
        assert tried_and_caught(Crossings(), ctx), name  # tried, not skipped, and caught
        run_gate([Crossings()], ctx, None)


def test_plant_caught_on_two_pieces():
    # different pieces: a post on a base, a drum head with its stick, a chair and a table
    scenes = [make_ctx(post_on_base()), make_ctx(drum_head_and_stick()),
              make_ctx(chair().geometry(0.0)), make_ctx(table().geometry(0.0))]
    assert sum(plants_caught(Crossings(), ctx) for ctx in scenes) == len(scenes)


def test_an_open_part_is_unmeasured_and_its_pairs_are_left_out():
    open_post = post_on_base(sink=CALIBRATION_FAIL * CROSS)["post"][:-2]  # a box with a hole
    ctx = make_ctx({"base": post_on_base()["base"], "post": open_post})
    assert findings(ctx) == []
    assert Crossings().measured(ctx) == set()
    lines = Crossings().coverage(ctx)
    assert "crossings: skipped (open part): kit:Thing post" in lines
    assert all(isinstance(line, Unmeasured) for line in lines if "open part" in line)
    assert run_gate([Crossings()], ctx, None).status == "partial"


def _unsound_off_centre(monkeypatch, victim):
    """Make `geom.local_thickness` raise for one part anywhere but at its own centre: a closed
    mesh that is sound where its centre thickness is read and not where it meets the other part
    (a closed mesh cannot be made so by shape alone, so the read is patched)."""
    real = geom.local_thickness
    centre = pieces.centroid(victim)

    def patched(tris, point):
        if np.array_equal(tris, victim) and not np.allclose(point, centre):
            raise ValueError("local_thickness: no ray hit an opposite surface")
        return real(tris, point)

    monkeypatch.setattr(geom, "local_thickness", patched)


def test_open_part_reads_unmeasured_and_run_continues(monkeypatch):
    parts = post_on_base(sink=CALIBRATION_FAIL * CROSS)
    ctx = make_ctx(parts)
    _unsound_off_centre(monkeypatch, parts["post"])
    assert findings(ctx) == []  # no depth, no number, and no exit 2
    (line,) = [x for x in Crossings().coverage(ctx) if ITEM in x]
    assert isinstance(line, Unmeasured) and line.startswith("crossings: skipped (the ")
    assert "no thickness there" in line
    verdict = run_gate([Crossings()], ctx, None)
    assert verdict.status == "partial"
    assert verdict.failed == 0


def test_a_skip_line_reads_as_one_phrase_for_a_part_with_no_thickness_at_its_centre(monkeypatch):
    parts = post_on_base()
    real = crossings.local_thickness

    def patched(tris, point):
        if np.array_equal(tris, parts["post"]):
            raise ValueError("local_thickness: no ray hit an opposite surface")
        return real(tris, point)

    monkeypatch.setattr(crossings, "local_thickness", patched)
    lines = Crossings().coverage(make_ctx(parts))
    assert "crossings: skipped (no thickness at its centre): kit:Thing post" in lines


def test_crossings_has_no_centre_thickness_fallback(monkeypatch):
    parts = post_on_base(sink=CALIBRATION_FAIL * CROSS)
    base, post = crossings.solid_of(parts["base"]), crossings.solid_of(parts["post"])
    assert crossings.measure(base, post).over  # sound: a number
    _unsound_off_centre(monkeypatch, parts["post"])
    with pytest.raises(Unmeasurable):  # the part's centre thickness does not stand in
        crossings.measure(base, post)


@pytest.mark.parametrize("piece", SUNK_PIECES)
def test_sunk_just_past_half_thinner_fails_crossings_on_two_pieces(piece):
    (found,) = findings(sunk_post(piece, 1.04))
    assert found.item == ITEM
    assert findings(sunk_post(piece, 0.96)) == []  # and just under half it stands


def test_the_limit_in_a_finding_is_the_factor_times_the_thinner_thickness():
    (found,) = findings(make_ctx(post_on_base(sink=CALIBRATION_FAIL * CROSS)))
    limit = float(found.detail.split("limit ")[1].split(",")[0])
    thickness = float(found.detail.split("thickness ")[1].split(")")[0])
    assert limit == pytest.approx(CROSS_OF_PART_THICKNESS * thickness, abs=1e-4)
    assert thickness == pytest.approx(0.02, rel=0.15)


# --- depth is the shortest move that separates the parts ----------------------------------------
# A strip lying on a board that is tilted off every axis and diagonal: the region the two share is
# a long thin wedge, so its extent along a fixed direction is far more than how far the strip sank.
# The move that clears the strip is straight out of the board, by exactly how far it sank.


def _turned(tris, degrees, axis):
    """The triangles turned `degrees` about `axis` through the origin (Rodrigues)."""
    k = np.asarray(axis, dtype=float)
    k /= np.linalg.norm(k)
    a = np.radians(degrees)
    cross = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rot = np.eye(3) + np.sin(a) * cross + (1 - np.cos(a)) * cross @ cross
    return (tris.reshape(-1, 3) @ rot.T).reshape(tris.shape)


def _board_with_strip(strip, board, sunk, tilt):
    """A board whose top lies at y = 0 and a strip lying on it, its lowest point `sunk` below the
    top; both turned by `tilt` = (degrees, axis). The strip is any mesh with its length along x."""
    low = strip.reshape(-1, 3)[:, 1].min()
    strip = strip + np.array([0.0, -sunk - low, 0.0])
    return {"board": _turned(board, *tilt), "strip": _turned(strip, *tilt)}


# Two different pieces: a flat fret-like strip on a plank, and a round rod on a thick slab, each
# tilted a different way. The limit is `CROSS_OF_PART_THICKNESS` x the thinner part's width.
STRIP_WIDTH, ROD_WIDTH = 0.008, 0.012
SUNK_SCENES = {
    "strip": (box((-0.25, 0.0, -0.004), (0.25, 0.01, 0.004)), STRIP_WIDTH,
              box((-0.3, -0.04, -0.05), (0.3, 0.0, 0.05)), (22.0, (0.1, 0.0, 1.0))),
    "rod": (_turned(cylinder((0.0, 0.0), ROD_WIDTH / 2, -0.25, 0.25, sides=48), 90.0, (0, 0, 1)),
            ROD_WIDTH, box((-0.3, -0.1, -0.1), (0.3, 0.0, 0.1)), (31.0, (1.0, 0.0, 0.4))),
}


def _sunk_ctx(name, share):
    strip, width, board, tilt = SUNK_SCENES[name]
    return make_ctx(_board_with_strip(strip, board, share * CROSS_OF_PART_THICKNESS * width, tilt))


@pytest.mark.parametrize("name", SUNK_SCENES)
def test_a_strip_sunk_under_the_limit_into_a_tilted_board_passes(name):
    ctx = _sunk_ctx(name, 0.6)
    assert findings(ctx) == []
    assert plants_caught(Crossings(), ctx)


@pytest.mark.parametrize("name", SUNK_SCENES)
def test_a_strip_sunk_past_the_limit_into_a_tilted_board_fails(name):
    (found,) = findings(_sunk_ctx(name, 1.4))
    assert found.item == "kit:Thing board x strip"


def test_the_shared_depth_of_a_tilted_strip_is_how_far_it_sank():
    # No move shorter than straight out of the board frees the strip. The 26 directions hold none
    # that is straight out of a tilted board, so the one nearest it reads a little more (the cosine
    # of its angle off the normal), not the 0.0117 that the extent of the shared region read.
    sank = 2.0 * CROSS_OF_PART_THICKNESS * SUNK_SCENES["strip"][1]
    (found,) = findings(_sunk_ctx("strip", 2.0))
    shared = float(found.detail.split("share ")[1].split(" ")[0])
    assert sank * 0.99 <= shared <= sank * 1.15


# --- a pair whose points all leave along one way is flush ----------------------------------------


def flush_pair():
    """A thin sheet on a slab, its bottom face inside the slab's top face by float noise only."""
    slab = crossings.solid_of(box((-1.0, -0.2, -1.0), (1.0, 0.0, 1.0)))
    sheet = crossings.solid_of(box((-0.8, -geom.TIE_DISTANCE / 2, -0.8),
                                   (0.8, 0.01, 0.8)))
    return sheet, slab


def every_fixture_pair():
    """Each pair of solid parts in the scenes of this file."""
    scenes = [joined(), make_ctx(post_on_base(lift=0.05)),
              *(sunk_post(piece, share) for piece in SUNK_PIECES for share in (0.6, 1.4)),
              make_ctx(cubes(0.1)), make_ctx(cubes(0.999)), make_ctx(cubes(1.0)),
              *(_sunk_ctx(name, share) for name in SUNK_SCENES for share in (0.6, 1.4))]
    for ctx in scenes:
        for view in pieces.measured_pieces(ctx, Crossings.kinds):
            made = crossings.made_of(view)
            for i, j in pieces.box_overlaps(made.parts):
                yield made.solids[made.parts[i].name], made.solids[made.parts[j].name]


def test_all_leave_needs_one_way_shared_by_every_side():
    a = SimpleNamespace(leaves=np.array([[True, False], [True, True]]))
    b = SimpleNamespace(leaves=np.array([[True, True]]))
    c = SimpleNamespace(leaves=np.array([[False, True]]))
    assert crossings._all_leave([a, b])  # way 0 on both sides
    assert not crossings._all_leave([a, c])  # a leaves by way 0 only, c by way 1 only
    assert crossings._all_leave([b])  # one side holds every point


def test_a_flush_pair_is_read_without_casting_a_ray(monkeypatch):
    a, b = flush_pair()
    monkeypatch.setattr(crossings, "shortest_exit", lambda *_: pytest.fail("a ray was cast"))
    assert crossings.measure(a, b) == crossings.Cross(0.0, 0.0, 0.0)


def test_the_flush_pair_would_cast_a_ray_without_the_shortcut(monkeypatch):
    # Proof the fixture reaches the shortcut and not an early return of `measure`.
    a, b = flush_pair()
    monkeypatch.setattr(crossings, "_all_leave", lambda sides: False)
    called = []
    real = crossings.shortest_exit
    monkeypatch.setattr(crossings, "shortest_exit",
                        lambda *args: called.append(1) or real(*args))
    assert crossings.measure(a, b) == crossings.Cross(0.0, 0.0, 0.0)
    assert called


def test_the_shortcut_never_changes_a_reading(monkeypatch):
    pairs = [flush_pair(), *every_fixture_pair()]
    fast = [crossings.measure(a, b) for a, b in pairs]
    monkeypatch.setattr(crossings, "_all_leave", lambda sides: False)
    assert fast == [crossings.measure(a, b) for a, b in pairs]
