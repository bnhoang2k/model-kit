"""Check markings-flush: a solid part flush with its host, to the eye at the piece's nearest viewing
distance, is a marking built as an object (rule 3 of the markings check, judged by eye).

The limit is what the eye resolves at the declared distance: 0.29 mm at 1 m. Plants are one solid
part pushed into its host, on the test Panel's plate and bolts and on the Strat regression fixture.
"""

import dataclasses
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from model_kit.core import gate, sight
from model_kit.core.checks import pieces
from model_kit.core.checks.markings_flush import MarkingsFlush, _host_of, _marked
from model_kit.core.checks.pieces import PartMesh
from model_kit.core.manifest import View
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom, load_scene, piece_key
from model_kit.core.tolerances import EYE_RESOLUTION_ARCMIN

from ..bodies.bar03 import bar03_scene
from ..bodies.humanoid import humanoid_scene
from ..shapes import box
from .piece_scenes import make_ctx, refuses_the_core
from .sight_scenes import unviewed

CHECK = MarkingsFlush()
EYE_PER_METRE = math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60.0))  # 0.29 mm a metre
MM = 1e-3
STRAT = Path(__file__).resolve().parents[1] / "fixtures" / "regressions" / "strat_before_neck_plate"
ITEM = "kit:Thing cap flush with base: a marking?"


def viewed(ctx, pack, name, distance=1.0):
    piece = ctx.packs[pack].pieces[name]
    view = View(distance, "a test piece, seen from this far")
    return pieces.with_piece(ctx, pack, name, dataclasses.replace(piece, view=view))


def base_and_cap(proud, distance=1.0, depth=0.01):
    """A closed slab (top at y = 0.1) with a cap `on` it whose top stands `proud` metres above the
    slab's top and whose foot reaches `depth` below it."""
    parts = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1)),
             "cap": box((-0.02, 0.1 - depth, -0.02), (0.02, 0.1 + proud, 0.02))}
    return viewed(make_ctx(parts, {"cap": {"on": "base"}}), "kit", "Thing", distance)


def items(ctx, only=None):
    return [f.item for f in CHECK.run(ctx, only=only)]


def test_the_check_is_registered_and_states_its_principle():
    assert gate.CHECKS["markings-flush"] is MarkingsFlush
    assert MarkingsFlush.kinds == frozenset({"assembly"})
    assert CHECK.principle.endswith(".") and "http" in CHECK.source


# --- the limit: what the eye resolves at the declared distance --------------------------------


def test_the_limit_is_the_eye_at_the_distance_in_pack_units():
    (view,) = pieces.measured_pieces(base_and_cap(0.001, 2.0), CHECK.kinds)
    assert sight.seen_floor(view.piece, 1.0) == pytest.approx(2.0 * EYE_PER_METRE, rel=1e-12)
    assert sight.seen_floor(view.piece, 3.0) == pytest.approx(6.0 * EYE_PER_METRE, rel=1e-12)


def test_a_float_of_1_5_mm_at_about_1_m_is_visible_so_not_flush():
    """A 1.5 mm float on an instrument at about a metre can be seen."""
    for proud in (0.5 * MM, 1.0 * MM, 1.5 * MM):
        assert items(base_and_cap(proud, distance=1.0)) == [], proud


def test_the_same_float_is_flush_from_far_enough_away():
    """1.5 mm subtends under a minute of arc beyond 5.2 m, and is then a change of colour."""
    far = 1.5 * MM / EYE_PER_METRE  # the distance where it is exactly the limit
    assert items(base_and_cap(1.5 * MM, distance=far * 0.99)) == []
    assert items(base_and_cap(1.5 * MM, distance=far * 1.01)) == [ITEM]


def test_a_seam_under_the_eye_at_a_metre_is_flush_and_a_hair_over_is_not():
    limit = EYE_PER_METRE  # a metre away
    assert items(base_and_cap(limit * 0.99)) == [ITEM]
    assert items(base_and_cap(limit * 1.01)) == []


def test_a_plate_flush_with_its_host_is_flagged_and_says_how_far_it_stands_off():
    (found,) = CHECK.run(base_and_cap(0.0))
    assert found.item == ITEM
    assert "0.00000" in found.detail and "1 m" in found.detail and "colour change" in found.detail


def test_a_part_sunk_a_little_below_the_surface_is_flush_and_a_deep_one_is_hidden():
    """Within the limit either side of the surface the part shows as a colour change. Buried deeper
    than that, nothing of it shows: it is not a marking, and the crossing checks own the depth."""
    under = 0.1 * MM  # the limit at 1 m is 0.29 mm
    assert items(base_and_cap(-under, depth=0.01)) == [ITEM]
    assert items(base_and_cap(-3 * under, depth=0.01)) == []
    assert items(base_and_cap(-0.05, depth=0.06)) == []  # wholly inside the slab


def test_the_stand_off_is_the_largest_over_the_part_not_the_mean():
    """A cap with one corner 2 mm proud and the rest flush is a part."""
    parts = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1)),
             "cap": box((-0.02, 0.09, -0.02), (0.02, 0.1, 0.02))}
    tris = parts["cap"].copy()
    tris[tris[:, :, 1] > 0.0995] += [0.0, 0.002, 0.0]  # the whole top, 2 mm up
    parts["cap"] = tris
    ctx = viewed(make_ctx(parts, {"cap": {"on": "base"}}), "kit", "Thing")
    assert items(ctx) == []


# --- an open host -----------------------------


def sheet(y=0.1, half=0.1):
    """An open host: a single upward-facing square, with no inside."""
    a, b, c, d = ((-half, y, -half), (half, y, -half), (half, y, half), (-half, y, half))
    return np.array([[a, c, b], [a, d, c]], dtype=float)


def test_a_part_far_behind_an_open_host_is_far_from_it_not_flush():
    """A signed distance would make a vertex behind a one-sided host negative however far away, so
    a part 0.3 m behind it would read as flush. An open host has no inside: the distance is unsigned."""
    host = sheet()
    assert not _host_of(PartMesh("h", host, "solid", "box", "g", None)).closed
    behind = box((-0.02, -0.5, -0.02), (0.02, -0.3, 0.02))
    ctx = viewed(make_ctx({"base": host, "cap": behind}, {"cap": {"on": "base"}}), "kit", "Thing")
    assert items(ctx) == []
    touching = box((-0.02, 0.0999, -0.02), (0.02, 0.1001, 0.02))
    ctx = viewed(make_ctx({"base": host, "cap": touching}, {"cap": {"on": "base"}}), "kit", "Thing")
    assert items(ctx) == [ITEM]
    thin_behind = box((-0.02, 0.0998, -0.02), (0.02, 0.0999, 0.02))
    ctx = viewed(make_ctx({"base": host, "cap": thin_behind}, {"cap": {"on": "base"}}), "kit",
                 "Thing")
    assert items(ctx) == [ITEM]  # a wafer 0.1 mm behind the sheet shows as a mark on it


def test_a_part_far_behind_a_closed_host_is_a_part_too():
    """Behind a closed host the same: the host's slab lies between, and a part below it is far."""
    parts = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1)),
             "cap": box((-0.02, -0.5, -0.02), (0.02, -0.3, 0.02))}
    ctx = viewed(make_ctx(parts, {"cap": {"on": "base"}}), "kit", "Thing")
    assert items(ctx) == []


# --- what is skipped, and what is unmeasured --------------------------------------------------


def test_a_flush_part_on_a_hidden_side_is_not_judged():
    """The visible rules skip what the piece never shows. The cap is flush with the slab's top,
    and its outermost face looks up: hide the top and the cap is quiet; hide only the bottom and
    it is still a marking."""
    shown = base_and_cap(0.0)
    assert items(shown) == [ITEM]
    piece = shown.packs["kit"].pieces["Thing"]

    def hiding(*sides):
        view = dataclasses.replace(piece.view, hidden=sides, hidden_why="a test: never seen")
        return pieces.with_piece(shown, "kit", "Thing", dataclasses.replace(piece, view=view))

    assert items(hiding("top")) == []
    assert items(hiding("bottom")) == [ITEM]
    assert items(hiding("top", "bottom")) == []


def test_a_decal_and_a_part_with_no_host_are_not_judged():
    parts = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1)),
             "cap": box((-0.02, 0.09, -0.02), (0.02, 0.1, 0.02)),
             "mark": box((-0.02, 0.0999, -0.02), (0.02, 0.1001, 0.02)),
             "lone": box((0.5, 0.0, 0.5), (0.52, 0.1, 0.52))}
    ctx = viewed(make_ctx(parts, {"cap": {"on": "base"}, "mark": {"on": "base", "shape": "decal",
                                                                  "role": "decal"}}),
                 "kit", "Thing")
    assert items(ctx) == [ITEM]
    lines = CHECK.coverage(ctx)
    assert any("not applicable" in line and "3 of 4 parts" in line for line in lines)
    assert not [x for x in lines if isinstance(x, Unmeasured)]


def test_a_part_of_a_marked_row_is_left_to_rules_1_and_2():
    """A row whose `finish` says print, paint, inlay or display is already a marking. Rows have no
    `finish` until the markings check reads it, so the test reads one off a stand-in."""
    row = SimpleNamespace(id="stock", finish="print")
    view = SimpleNamespace(piece=SimpleNamespace(listed=(row,)))
    assert _marked(view, SimpleNamespace(listed="stock"))
    view.piece.listed = (SimpleNamespace(id="stock", finish="solid"),)
    assert not _marked(view, SimpleNamespace(listed="stock"))
    view.piece.listed = (SimpleNamespace(id="stock"),)  # a row with no finish at all
    assert not _marked(view, SimpleNamespace(listed="stock"))


def test_no_view_is_partial():
    ctx = unviewed(make_ctx({"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1)),
                             "cap": box((-0.02, 0.09, -0.02), (0.02, 0.1, 0.02))},
                            {"cap": {"on": "base"}}))
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == [] and CHECK.measured(ctx) == set()
    (line,) = [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
    assert "no view" in line and "kit:Thing" in line
    assert run_gate([MarkingsFlush()], ctx, None).status == "partial"


def test_a_piece_with_no_hosted_part_needs_no_view():
    ctx = unviewed(make_ctx({"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1))}))
    lines = CHECK.coverage(ctx)
    assert not [x for x in lines if isinstance(x, Unmeasured)]
    assert any("not applicable" in line for line in lines)
    assert run_gate([MarkingsFlush()], ctx, None).status == "passed"


def test_a_host_with_no_geometry_is_unmeasured():
    ctx = base_and_cap(0.0)
    key = piece_key("kit", "Thing")
    parts = {k: v for k, v in ctx.pieces[key].parts.items() if k != "base"}
    ctx = dataclasses.replace(ctx, pieces={key: PieceGeom("Thing", parts)})
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == []  # not judged: no unit, a coverage line
    assert any(isinstance(x, Unmeasured) and "no geometry" in x for x in CHECK.coverage(ctx))


def test_the_same_part_in_other_units_gives_the_same_verdict():
    ctx = base_and_cap(0.1 * MM)
    scaled = dataclasses.replace(ctx, pieces={k: PieceGeom(g.name, {n: t * 2.0 for n, t in
                                                                    g.parts.items()})
                                              for k, g in ctx.pieces.items()})
    pack = scaled.packs["kit"]
    scaled = dataclasses.replace(scaled, packs={"kit": dataclasses.replace(pack,
                                                                           units_per_metre=2.0)})
    assert items(scaled) == items(ctx) == [ITEM]


# --- plants -----------------------------------------------------------------------------------


def caught(plant) -> bool:
    return any(f.item == plant.item for f in CHECK.run(plant.ctx, only=plant.scope))


def first_caught(unit):
    """The candidate the gate would accept: the first the check catches."""
    return next((p for p in unit.candidates if caught(p)), None)


def test_the_flush_plant_pushes_a_part_in_as_a_rigid_body():
    ctx = base_and_cap(5 * MM)
    (unit,) = CHECK.plants(ctx)
    plant = first_caught(unit)
    assert unit.item == ITEM and plant.label.startswith("pushed") and items(ctx) == []
    before = ctx.pieces[piece_key("kit", "Thing")].parts["cap"]
    after = plant.ctx.pieces[piece_key("kit", "Thing")].parts["cap"]
    move = after - before
    assert np.allclose(move, move[0, 0])  # every corner moved by the same vector


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_flush_plant_caught_on_two_pieces():
    """The test Panel (a plate bolted to a board) and the Strat regression fixture, each seen
    from 1 m: quiet before, one plant each, and caught."""
    panel = viewed(humanoid_scene(), "humanoid", "Panel")
    strat = viewed(load_scene(STRAT), "strat", "Strat")
    for label, ctx, key in (("panel", panel, "humanoid:Panel"), ("strat", strat, "strat:Strat")):
        units = [u for u in CHECK.plants(ctx) if u.item.startswith(f"{key} ")]
        assert len(units) == 1, label
        assert first_caught(units[0]) is not None, label
        assert units[0].item not in {f.item for f in CHECK.run(ctx)}, label
        run_gate([MarkingsFlush()], ctx, None)  # no BlindCheck


def test_a_part_no_rigid_push_brings_flush_is_pressed_onto_its_host():
    """Bar 03's limbs sit beside curved hosts: a rigid push leaves some vertex proud, so the plant
    presses the part onto the host instead. Seen from 1 m, quiet before and caught after."""
    ctx = viewed(bar03_scene(), "chibi", "chibi")
    (unit,) = CHECK.plants(ctx)
    assert [p.label.split(":")[0] for p in unit.candidates] == ["pushed flush", "pressed flush"]
    plant = first_caught(unit)
    assert plant.label.startswith("pressed flush")
    assert unit.item not in {f.item for f in CHECK.run(ctx)}


def test_a_piece_whose_candidate_is_flush_already_has_its_finding_and_the_gate_skips_the_unit():
    ctx = base_and_cap(0.0)
    (unit,) = CHECK.plants(ctx)
    assert unit.item == ITEM and items(ctx) == [ITEM]
    assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
    assert run_gate([MarkingsFlush()], ctx, None).status == "failed"


def test_a_part_buried_below_the_limit_is_pushed_out_and_caught():
    """A part buried deeper than the eye resolves is judged (and quiet). Its plant is pushed out
    to the limit; pushed the wrong way it left the unit with no caught candidate."""
    ctx = base_and_cap(-5 * MM)
    assert items(ctx) == []
    (unit,) = CHECK.plants(ctx)
    plant = first_caught(unit)
    assert unit.item == ITEM and plant is not None and plant.label.startswith("pushed")
    before = ctx.pieces[piece_key("kit", "Thing")].parts["cap"]
    after = plant.ctx.pieces[piece_key("kit", "Thing")].parts["cap"]
    assert after[..., 1].max() > before[..., 1].max()  # moved up, out of the slab
    assert run_gate([MarkingsFlush()], ctx, None).status == "passed"


def test_a_visible_part_no_plant_reaches_is_a_unit_with_no_candidates_and_no_reason(monkeypatch):
    """The check judges the cap, so the unit is never "not applicable": with no push or press the
    gate has nothing to try and stops the run (the cap is not flagged, so it is not skipped)."""
    from model_kit.core.checks import markings_flush
    from model_kit.core.errors import BlindCheck

    monkeypatch.setattr(markings_flush, "_pushed", lambda *a: None)
    monkeypatch.setattr(markings_flush, "_pressed", lambda *a: None)
    ctx = base_and_cap(5 * MM)
    (unit,) = CHECK.plants(ctx)
    assert unit.item == ITEM and unit.candidates == () and unit.not_applicable == ""
    with pytest.raises(BlindCheck, match="markings-flush"):
        run_gate([MarkingsFlush()], ctx, None)


def test_a_later_part_is_planted_when_the_smallest_cannot_be(monkeypatch):
    from model_kit.core.checks import markings_flush

    real = markings_flush._pushed
    monkeypatch.setattr(markings_flush, "_pushed", lambda part, host, limit: (
        None if np.ptp(part[:, :, 0]) < 0.05 else real(part, host, limit)))
    monkeypatch.setattr(markings_flush, "_pressed", lambda *a: None)
    parts = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.1, 0.1)),
             "dot": box((-0.01, 0.09, -0.01), (0.01, 0.105, 0.01)),  # the smallest part
             "cap": box((-0.04, 0.09, -0.04), (0.04, 0.105, 0.04))}
    ctx = viewed(make_ctx(parts, {"dot": {"on": "base"}, "cap": {"on": "base"}}), "kit", "Thing")
    (unit,) = CHECK.plants(ctx)
    assert unit.item == "kit:Thing cap flush with base: a marking?" and first_caught(unit)


def test_a_visible_part_whose_moved_copies_look_only_at_a_hidden_side_stops_the_run(monkeypatch):
    """The cap itself is seen, so the check judges it; only its pushed and pressed copies look
    toward a hidden side. That is a part the plants cannot reach, not a piece with nothing to
    judge: no "not applicable" reason, and the gate raises BlindCheck."""
    from model_kit.core.checks import markings_flush
    from model_kit.core.errors import BlindCheck

    ctx = base_and_cap(5 * MM)
    real = markings_flush._on_hidden_sides
    original = ctx.pieces[piece_key("kit", "Thing")].parts["cap"]
    monkeypatch.setattr(markings_flush, "_on_hidden_sides", lambda part, reach, limit, hidden: (
        not np.array_equal(part, original) or real(part, reach, limit, hidden)))
    (unit,) = CHECK.plants(ctx)
    assert unit.item == ITEM and unit.candidates == () and unit.not_applicable == ""
    with pytest.raises(BlindCheck, match="markings-flush"):
        run_gate([MarkingsFlush()], ctx, None)


def test_markings_flush_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import markings_flush

    ctx = base_and_cap(5 * MM)
    refuses_the_core(monkeypatch, markings_flush, "_findings", "_is_flush", "standoff")
    monkeypatch.setattr(markings_flush.MarkingsFlush, "run", lambda *a, **k: 1 / 0)
    assert [u.item for u in MarkingsFlush().plants(ctx)] == [ITEM]


def test_a_part_on_a_hidden_side_is_not_planted():
    # the cap's outermost faces look up; with `top` hidden the check never judges it, so no
    # fault of it can be caught and the unit says so
    ctx = base_and_cap(5 * MM)
    key = piece_key("kit", "Thing")
    piece = ctx.packs["kit"].pieces["Thing"]
    hidden = dataclasses.replace(piece.view, hidden=("top",), hidden_why="a test: never seen")
    ctx = pieces.with_piece(ctx, "kit", "Thing", dataclasses.replace(piece, view=hidden))
    (unit,) = CHECK.plants(ctx)
    assert unit.item == key and unit.candidates == ()
    assert unit.not_applicable == "every part with a host looks only toward a side the view hides"
    assert run_gate([MarkingsFlush()], ctx, None).status == "passed"


def test_the_gate_runs_the_plants_and_a_good_piece_passes():
    verdict = run_gate([MarkingsFlush()], base_and_cap(5 * MM), None)
    assert verdict.status == "passed" and verdict.failed == 0


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_the_strat_fixture_has_its_flush_parts_by_distance():
    """The Strat's solid fret dots and neck-plate logo stand about 0.56 mm off their hosts (the
    pack is half scale), so from 2 m they are colour changes and from 1 m they are not; its jack
    ring stands 0.2 mm off the socket and is flush from 1 m. From 0.5 m nothing is."""
    got = {}
    for distance in (0.5, 1.0, 2.0):
        ctx = viewed(load_scene(STRAT), "strat", "Strat", distance)
        got[distance] = [i.split()[1] for i in items(ctx)]
    assert got[0.5] == []
    assert got[1.0] == ["JackHole"]
    assert got[2.0] == ["Dot_03", "Dot_05", "Dot_07", "Dot_09", "Dot_15", "Dot_17", "Dot_19",
                        "Dot_21", "JackHole", "PlateLogo"]
