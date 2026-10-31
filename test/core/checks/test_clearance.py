"""Check clearance, layout clearance: unjoined placed pieces keep a finger's gap, and a piece that rests on
another touches it.

The scene is a pack `kit` of boxes in a layout `yard` (built here, with a `units_per_metre` of
choice), plus the shop fixture layout and the humanoid test body's layout with geometry. Items are
`<layout> <a> x <b>` and `<layout> <a> rests on <b>`.
"""

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.clearance import Clearance, _gap, _Scan
from model_kit.core.errors import BlindCheck
from model_kit.core.geom import surface_gap
from model_kit.core.manifest import (
    Layout,
    LayoutPlacement,
    Ref,
    load_layouts,
    load_pack,
)
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom, SceneCtx
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    MIN_GAP_M,
)
from model_kit.core.units import Style, load_style

from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene
from ..shapes import box
from .assembly_scenes import Q_Y90, plants_caught
from .piece_scenes import refuses_the_core, tried_and_caught
from .sight_scenes import unviewed, view_json, viewed

FIX = Path(__file__).parents[1] / "fixtures"
Q_ID = (0.0, 0.0, 0.0, 1.0)
BASE_M = 0.001  # the listed row's thickness, in metres
TOUCH = 0.0107  # the contact length of a test piece: its seen floor, in metres
SIDE = 0.2


def findings(ctx, only=None):
    return Clearance().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


# --- the box scene ------------------------------------------------------------------------------


def _piece(parts, kind="assembly"):
    return {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": kind, "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test",
                    "min_thickness": BASE_M}],
        "view": view_json(TOUCH),
        "parts": [{"name": n, "listed": "stock", "shape": shape, "group": "g", "role": role,
                   # a decal is printed on the first other part (the loader needs an `on`)
                   "on": parts[0][0] if shape == "decal" else None} for n, shape, role in parts],
        "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {},
    }


def kit_pack(units_per_metre: float):
    style = Style(units_per_metre=units_per_metre, page_camera={}, dials={})
    data = {"pack": "kit", "units_per_metre": units_per_metre, "pieces": {
        "Block": _piece([("body", "box", "solid"), ("cord", "cable", "solid")]),
        "Slab": _piece([("top", "box", "solid"), ("print", "decal", "decal")]),
        "Cord": _piece([("wire", "cable", "solid")]),
        "Tree": _piece([("trunk", "box", "solid")], kind="branching"),
    }}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return style, load_pack(path, style)


def kit_geometry() -> dict[str, PieceGeom]:
    return {
        # the cord reaches 1.0 either side: it would touch everything if cables counted
        "kit:Block": PieceGeom("Block", {"body": box((0, 0, 0), (SIDE, SIDE, SIDE)),
                                         "cord": box((-1.0, 0.05, 0.05), (1.0, 0.06, 0.06))}),
        "kit:Slab": PieceGeom("Slab", {"top": box((0, 0, 0), (SIDE, 0.05, SIDE)),
                                       "print": box((0.05, 0.05, 0.05), (0.1, 0.0501, 0.1))}),
        "kit:Cord": PieceGeom("Cord", {"wire": box((0, 0, 0), (0.5, 0.004, 0.004))}),
        "kit:Tree": PieceGeom("Tree", {"trunk": box((0, 0, 0), (SIDE, SIDE, SIDE))}),
    }


def at(id, piece, x=0.0, y=0.0, z=0.0, quaternion=Q_ID, **joins):
    return LayoutPlacement(id, "kit", piece, (x, y, z), quaternion, **joins)


def yard(*placements, units_per_metre=1.0, geometry=None) -> SceneCtx:
    style, pack = kit_pack(units_per_metre)
    layout = Layout("yard", "gltf", {p.id: p for p in placements}, {})
    return SceneCtx(style=style, packs={"kit": pack}, clips={}, samples={}, rigs={},
                    rig_of_clip={}, layouts={"yard": layout},
                    pieces=kit_geometry() if geometry is None else geometry)


def pair(gap, units_per_metre=1.0, **joins):
    """Two blocks, `gap` apart along x."""
    return yard(at("a", "Block"), at("b", "Block", SIDE + gap, **joins),
                units_per_metre=units_per_metre)


LIMIT = MIN_GAP_M  # at a units_per_metre of 1
GAP_ITEM = "yard a x b"


# --- registration and the fixed rules ---------------------------------------------------------


def test_the_check_is_registered_for_assembly_pieces_and_cites_its_source():
    assert gate.CHECKS["clearance"] is Clearance
    assert Clearance.name == "clearance"
    assert Clearance.kinds == frozenset({"assembly"})
    assert "ISO 13854" in Clearance.source and "crushing" in Clearance.source


def test_the_minimum_gap_is_a_finger_in_metres():
    assert MIN_GAP_M == 0.025


def test_single_placement_layout_is_clean():
    ctx = yard(at("a", "Block"))
    assert findings(ctx) == [] and Clearance().plants(ctx) == []
    lines = Clearance().coverage(ctx)
    assert any("not applicable" in line and "yard" in line for line in lines)
    assert not any(isinstance(line, Unmeasured) for line in lines)
    assert run_gate([Clearance()], ctx, None).status == "passed"


def test_measured_lists_the_solid_parts_of_pieces_placed_beside_another():
    ctx = yard(at("a", "Block"), at("b", "Slab", 1.0), at("c", "Cord", 2.0))
    assert Clearance().measured(ctx) == {"kit:Block body", "kit:Slab top"}  # no cable, no decal
    assert Clearance().measured(yard(at("a", "Block"))) == set()


def test_a_scene_with_no_layout_is_not_applicable():
    ctx = dataclasses.replace(yard(), layouts={})
    assert findings(ctx) == [] and Clearance().plants(ctx) == []
    assert "clearance: not applicable (no layout): nothing to measure" in Clearance().coverage(ctx)


def test_a_layout_with_no_placements_is_not_applicable():
    ctx = yard()
    assert findings(ctx) == []
    assert "clearance: not applicable (no placements): yard" in Clearance().coverage(ctx)


def test_a_gap_over_the_minimum_passes_and_under_it_fails():
    assert items(pair(CALIBRATION_FAIL * LIMIT)) == []
    (found,) = findings(pair(LIMIT / CALIBRATION_FAIL))
    assert (found.check, found.item) == ("clearance", GAP_ITEM)
    assert "kit:Block and kit:Block" in found.detail and "0.0125 apart" in found.detail
    assert items(pair(LIMIT * 1.01)) == [] and items(pair(LIMIT * 0.99)) == [GAP_ITEM]


def test_joined_pairs_skipped():
    touching = 0.0
    assert items(pair(touching)) == [GAP_ITEM]
    on = Ref("kit", "a", "socket")
    assert items(pair(touching, on=on)) == []  # b is on a
    reverse = yard(at("a", "Block", on=Ref("kit", "b", "socket")), at("b", "Block", SIDE))
    assert items(reverse) == []  # a is on b
    # resting is a join too: the pair is judged by touching, not by the gap
    assert items(pair(touching, rests_on="a")) == []


def test_a_join_does_not_excuse_a_third_piece():
    ctx = yard(at("a", "Block"), at("b", "Block", SIDE, on=Ref("kit", "a", "socket")),
               at("c", "Block", 2 * SIDE + LIMIT / 2))
    assert items(ctx) == ["yard b x c"]  # a and c are far apart; b is joined to a only


def test_overlap_fails():
    crossing = yard(at("a", "Block"), at("b", "Block", SIDE / 2))
    (found,) = findings(crossing)
    assert found.item == GAP_ITEM and "touch or overlap" in found.detail


def test_a_piece_wholly_inside_another_is_an_overlap():
    big = box((0, 0, 0), (1.0, 1.0, 1.0))
    small = box((0.4, 0.4, 0.4), (0.5, 0.5, 0.5))  # no surface of it comes near the big one's
    geometry = {**kit_geometry(), "kit:Block": PieceGeom("Block", {"body": small}),
                "kit:Slab": PieceGeom("Slab", {"top": big})}
    ctx = yard(at("a", "Block"), at("b", "Slab"), geometry=geometry)
    assert surface_gap(small, big, 10.0) > 3 * LIMIT  # nothing crosses
    assert items(ctx) == ["yard a x b"]
    out = yard(at("a", "Block", 5.0), at("b", "Slab"), geometry=geometry)
    assert items(out) == []


def test_the_nearest_part_decides_not_the_piece_box():
    # an L: a block whose far arm reaches round to a neighbour that sits inside its bounding box
    arm = box((0, 0, 0), (0.1, 0.1, 1.0))
    foot = box((0, 0, 0.9), (1.0, 0.1, 1.0))
    geometry = {**kit_geometry(), "kit:Block": PieceGeom("Block", {"body": np.concatenate(
        [arm, foot])})}
    ctx = yard(at("a", "Block"), at("b", "Slab", 0.5, 0.0, 0.0), geometry=geometry)
    assert items(ctx) == []  # the slab's box sits in a's box, but 0.4 clear of the L's solids


def test_cables_and_printed_parts_are_not_measured():
    # block a's cord runs through b; b's decal lies on its top; neither counts
    ctx = yard(at("a", "Block"), at("b", "Slab", SIDE + LIMIT * 2, 0.0, 0.0))
    assert items(ctx) == []
    onto = yard(at("a", "Block"), at("b", "Slab", 0.0, SIDE + LIMIT * 2, 0.0))
    assert items(onto) == []
    decal_only = yard(at("a", "Slab"), at("b", "Slab", 0.0, 0.05 + LIMIT / 2, 0.0))
    assert items(decal_only) == [GAP_ITEM]  # the tops are a half gap apart; the decals do not help


def test_a_piece_with_no_solid_part_is_not_applicable():
    ctx = yard(at("a", "Block"), at("b", "Cord", SIDE))
    assert items(ctx) == []
    lines = Clearance().coverage(ctx)
    assert "clearance: not applicable (no solid part): yard b" in lines
    assert not any(isinstance(line, Unmeasured) for line in lines)


def test_a_branching_piece_is_not_applicable():
    ctx = yard(at("a", "Block"), at("b", "Tree", SIDE))
    assert items(ctx) == []
    assert "clearance: not applicable (branching piece): kit:Tree" in Clearance().coverage(ctx)
    swapped = yard(at("a", "Tree"), at("b", "Tree", SIDE / 2))
    assert items(swapped) == []


def test_a_piece_with_no_geometry_is_unmeasured():
    geometry = {k: v for k, v in kit_geometry().items() if k != "kit:Slab"}
    ctx = yard(at("a", "Block"), at("b", "Slab", SIDE), geometry=geometry)
    assert items(ctx) == []
    lines = Clearance().coverage(ctx)
    (missing,) = [x for x in lines if "no geometry" in x]
    assert isinstance(missing, Unmeasured) and missing.endswith("yard b")
    assert run_gate([Clearance()], ctx, None).status == "partial"


def test_the_turn_of_a_placement_is_applied():
    bar = box((0, 0, 0), (0.6, 0.1, 0.1))  # 0.6 long along x, 0.1 across z
    geometry = {**kit_geometry(), "kit:Slab": PieceGeom("Slab", {"top": bar})}
    near = SIDE + LIMIT / 2  # a half gap from the block's +x face
    along = yard(at("a", "Block"), at("b", "Slab", near, 0.0, -0.05), geometry=geometry)
    assert items(along) == [GAP_ITEM]  # the bar's end faces the block across a half gap
    # a quarter turn about y sends the bar's length from +x to -z: it now ends 0.05 short of
    # the block's z range, and clear of it
    turned = yard(at("a", "Block"), at("b", "Slab", near, 0.0, -0.05, Q_Y90), geometry=geometry)
    assert items(turned) == []
    # moved 0.75 along +z, the turned bar runs along the block's +x face, a half gap off it
    beside = yard(at("a", "Block"), at("b", "Slab", near, 0.0, 0.7, Q_Y90), geometry=geometry)
    assert items(beside) == [GAP_ITEM]


def test_gap_scales_with_units_per_metre():
    """The same pair of boxes, 0.02 apart, is under a finger in model units of a metre, and over
    it where a model unit is half a metre (25 mm is 0.0125 units)."""
    assert items(pair(0.02, units_per_metre=1.0)) == [GAP_ITEM]
    assert items(pair(0.02, units_per_metre=0.5)) == []
    (found,) = findings(pair(0.01, units_per_metre=0.5))
    assert "0.0125" in found.detail
    assert items(pair(0.2, units_per_metre=10.0)) == [GAP_ITEM]  # 0.25 units: 0.2 is under it


def test_only_scopes_the_run_to_the_named_pairs():
    ctx = yard(at("a", "Block"), at("b", "Block", SIDE), at("c", "Block", 3.0),
               at("d", "Block", 3.0 + SIDE))
    assert items(ctx) == ["yard a x b", "yard c x d"]
    assert [f.item for f in findings(ctx, only={"yard c x d"})] == ["yard c x d"]
    assert findings(ctx, only={"yard a x c"}) == []


def test_the_item_names_its_ids_in_order():
    ctx = yard(at("zed", "Block", SIDE), at("alpha", "Block"))
    assert items(ctx) == ["yard alpha x zed"]


# --- rests on ---------------------------------------------------------------------------------


def stacked(lift, **joins):
    """A slab resting on a block: the slab `lift` above the block's top."""
    return yard(at("base", "Block"), at("top", "Slab", 0.0, SIDE + lift, 0.0, rests_on="base",
                                        **joins))


REST_ITEM = "yard top rests on base"


def test_rests_on_must_touch():
    assert items(stacked(0.0)) == []
    assert items(stacked(TOUCH * 0.9)) == []
    (found,) = findings(stacked(CALIBRATION_FAIL * TOUCH))
    assert (found.item, "kit:Slab rests on kit:Block" in found.detail) == (REST_ITEM, True)
    assert f"{CALIBRATION_FAIL * TOUCH:.4f} off it" in found.detail
    assert f"{TOUCH:.4f}" in found.detail


def test_a_resting_piece_listed_before_its_host_is_the_same_pair():
    def listed_first(lift):
        return yard(at("top", "Slab", 0.0, SIDE + lift, 0.0, rests_on="base"), at("base", "Block"))

    assert items(listed_first(0.0)) == []
    assert [f.item for f in findings(listed_first(CALIBRATION_FAIL * TOUCH))] == [REST_ITEM]
    (lift,) = [u for u in Clearance().plants(listed_first(0.0)) if "rests on" in u.item]
    assert lift.item == REST_ITEM and plants_caught(Clearance(), listed_first(0.0))


def test_a_resting_pair_is_not_a_gap_pair():
    ctx = stacked(0.0)  # they touch: a gap pair would fail them
    assert items(ctx) == []
    assert [i for i in items(stacked(LIMIT / 2)) if " x " in i] == []


def test_the_touch_limit_follows_the_resting_pieces_view():
    thin = stacked(TOUCH * 1.5)
    assert items(thin) == [REST_ITEM]
    assert items(viewed(thin, 4 * TOUCH)) == []  # the same lift is under four contact lengths
    thick = dataclasses.replace(thin, style=Style(1.0, {}, {"min_thickness": 4 * BASE_M}))
    assert items(thick) == [REST_ITEM]  # the thickness base no longer sets it


def test_a_join_on_a_third_piece_is_still_a_gap_pair():
    ctx = yard(at("base", "Block"), at("top", "Slab", 0.0, SIDE, 0.0, rests_on="base"),
               at("side", "Block", SIDE + LIMIT / 2, 0.0, 0.0))
    # a join excuses its own pair only: the block beside the base is a half gap off the slab too
    assert items(ctx) == ["yard base x side", "yard side x top"]


# --- plants ------------------------------------------------------------------------------------


def shop_scene() -> SceneCtx:
    """The shop fixture layout (a bench, a lamp on it, a second lamp apart) with geometry."""
    style = load_style(FIX / "shop.style.json")
    pack = load_pack(FIX / "shop.pack.json", style)
    lamp = {"base": box((-0.1, 0.0, -0.1), (0.1, 0.02, 0.1)),
            "stem": box((-0.01, 0.02, -0.01), (0.01, 0.8, 0.01)),
            "lead": box((-0.3, 0.0, -0.1), (0.0, 0.01, -0.09))}
    bench = {"leg_L": box((-0.3, 0.0, -0.15), (-0.2, 0.44, -0.05)),
             "leg_R": box((0.2, 0.0, -0.15), (0.3, 0.44, -0.05)),
             "seat_top": box((-0.3, 0.44, -0.15), (0.3, 0.46, 0.15))}
    layouts = load_layouts(FIX / "layouts", {"shop": pack})
    return viewed(SceneCtx(style=style, packs={"shop": pack}, clips={}, samples={}, rigs={},
                           rig_of_clip={}, layouts=layouts,
                           pieces={"shop:Bench": PieceGeom("Bench", bench),
                                   "shop:Lamp": PieceGeom("Lamp", lamp)}), TOUCH)


def planted_gap(plant) -> float:
    """The planted pair's real gap: the nearest distance between their solids."""
    layout_name, a, _, b = plant.item.split()
    layout = plant.ctx.layouts[layout_name]
    scan = _Scan(plant.ctx)
    return _gap(scan.body(layout, layout.placements[a]), scan.body(layout, layout.placements[b]),
                10.0)


@pytest.mark.parametrize("make, upm", [(shop_scene, 0.5), (humanoid_scene, 1.0),
                                       (quadruped_scene, 1.0), (worm_scene, 1.0)])
def test_gap_plant_caught_on_two_layouts(make, upm):
    ctx = make()
    assert findings(ctx) == []  # known good
    units = Clearance().plants(ctx)
    gap_units = [u for u in units if " x " in u.item]
    assert len(gap_units) == 1
    assert plants_caught(Clearance(), ctx)
    (plant, *_) = gap_units[0].candidates
    # the planted pair is half the minimum gap apart (exact: the nearest corner of each of these
    # layouts' boxes faces a flat side; a skewed pair would land a little under)
    assert planted_gap(plant) == pytest.approx(MIN_GAP_M * upm / CALIBRATION_FAIL)
    assert run_gate([Clearance()], ctx, None).status == "passed"
    assert not [x for x in Clearance().coverage(ctx) if isinstance(x, Unmeasured)]


def test_the_shop_plant_moves_the_unjoined_lamp():
    ctx = shop_scene()
    (unit,) = Clearance().plants(ctx)
    assert unit.item == "shop Lamp_1 x Lamp_2"  # the nearest unjoined pair
    plant = unit.candidates[0]
    moved = {i: p.position for i, p in plant.ctx.layouts["shop"].placements.items()}
    was = {i: p.position for i, p in ctx.layouts["shop"].placements.items()}
    assert [i for i in moved if moved[i] != was[i]] == ["Lamp_2"]  # the one joined to nothing


def shop_with_lamp_resting():
    """The shop scene with the second lamp set on the bench seat, clear of the first lamp."""
    ctx = shop_scene()
    layout = ctx.layouts["shop"]
    lamp = dataclasses.replace(layout.placements["Lamp_2"], position=(-0.15, 0.46, 0.05),
                               rests_on="Bench_1")
    layout = dataclasses.replace(layout, placements={**layout.placements, "Lamp_2": lamp})
    return dataclasses.replace(ctx, layouts={"shop": layout})


def test_the_lift_plant_is_caught_on_the_shop_layout():
    ctx = shop_with_lamp_resting()
    assert findings(ctx) == []  # as built: the lamp touches the seat, and clears the other lamp
    plants = Clearance().plants(ctx)
    assert sorted(u.item for u in plants) == ["shop Lamp_1 x Lamp_2", "shop Lamp_2 rests on Bench_1"]
    assert plants_caught(Clearance(), ctx)
    assert run_gate([Clearance()], ctx, None).status == "passed"
    assert not [x for x in Clearance().coverage(ctx) if isinstance(x, Unmeasured)]


def test_the_humanoid_layout_has_an_unjoined_piece_a_body_size_away():
    ctx = humanoid_scene()
    assert findings(ctx) == []
    assert any(" x " in u.item for u in Clearance().plants(ctx))


def test_the_rest_plant_lifts_a_resting_piece_twice_the_touch_off_its_host():
    ctx = stacked(0.0)
    plants = Clearance().plants(ctx)
    (unit,) = [u for u in plants if "rests on" in u.item]
    assert unit.item == REST_ITEM and len(unit.candidates) == 2  # then lifted by its own height too
    top = unit.candidates[0].ctx.layouts["yard"].placements["top"]
    assert top.position[1] - SIDE == pytest.approx(CALIBRATION_FAIL * TOUCH)
    assert plants_caught(Clearance(), ctx)
    assert run_gate([Clearance()], ctx, None).status == "passed"


def test_a_resting_piece_sunk_into_its_host_is_lifted_clear():
    ctx = stacked(-0.04)  # sunk 4 cm: lifting twice the touch limit still leaves it in
    (unit,) = [u for u in Clearance().plants(ctx) if "rests on" in u.item]
    first, second = unit.candidates
    assert not any(f.item == REST_ITEM for f in findings(first.ctx, only=first.scope))
    assert any(f.item == REST_ITEM for f in findings(second.ctx, only=second.scope))


def test_a_pair_that_already_fails_has_no_move_to_make_and_the_gate_skips_it():
    ctx = pair(0.0)
    (unit,) = Clearance().plants(ctx)
    assert unit.item == GAP_ITEM and unit.candidates == ()  # closer than half the gap already
    assert not [x for x in Clearance().coverage(ctx) if isinstance(x, Unmeasured)]
    assert items(ctx) == [GAP_ITEM]
    assert run_gate([Clearance()], ctx, None).status == "failed"  # no BlindCheck


def test_a_pair_closer_than_half_the_gap_that_the_check_misses_stops_the_run():
    class Blind(Clearance):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match=r"clearance: yard a x b"):
        run_gate([Blind()], pair(0.0), None)


def test_clearance_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import clearance

    ctx = shop_with_lamp_resting()
    refuses_the_core(monkeypatch, clearance, "_gap")
    monkeypatch.setattr(clearance._Scan, "findings", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(clearance._Scan, "_fault", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(clearance.Clearance, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in Clearance().plants(ctx)) == [
        "shop Lamp_1 x Lamp_2", "shop Lamp_2 rests on Bench_1"]


def test_clearance_plants_caught_on_two_pieces():
    for name, ctx in (("shop", shop_with_lamp_resting()), ("humanoid", humanoid_scene())):
        assert tried_and_caught(Clearance(), ctx), name  # tried, not skipped, and caught
        run_gate([Clearance()], ctx, None)


# --- a plug seated in its socket ---------------------------------------------------------------

SOCKET = (0.1, 0.03, 0.025)  # the strip's socket, on the housing's top face


def plug_pack(touch=TOUCH):
    """Pack `kit`: a `Strip` (housing and switch, a power mount `socket` that takes `plug`) and a
    `Cord` (a base, a plug on it, a blade on the plug, a wire) whose connector `feed` plugs into
    the strip's socket through the layout `yard`."""
    def piece(parts, **fields):
        return {
            "frame": {"axes": "gltf", "origin": "centre of the base"},
            "recipe": {"kind": "assembly", "source": "test"},
            "listed": [{"id": "stock", "text": "Test stock", "source": "test",
                        "min_thickness": BASE_M}],
            "view": view_json(touch),
            "parts": [{"name": n, "listed": "stock", "shape": shape, "group": "g",
                       "role": "solid", "on": on} for n, shape, on in parts],
            "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
            "size_class": "piece", "measured": {}, **fields}

    strip = piece([("housing", "box", None), ("switch", "box", "housing")],
                  mounts=[{"id": "socket", "kind": "power", "point": list(SOCKET),
                           "axis": [0.0, 1.0, 0.0], "takes": "plug", "part": "housing"}])
    cord = piece(
        [("base", "box", None), ("plug", "box", "base"), ("blade", "box", "plug"),
         ("wire", "cable", "base")],
        mounts=[{"id": "exit", "kind": "power", "point": [0.05, 0.025, 0.5],
                 "axis": [0.0, 0.0, -1.0], "takes": "wire"}],
        connectors=[{"id": "feed", "part": "wire", "cap": 1.0, "why": "a test cord",
                     "ends": [{"point": list(SOCKET), "layout": "yard", "host": "kit:strip.socket",
                               "plug": "plug"},
                              {"point": [0.05, 0.025, 0.5], "host": "self.exit"}]}])
    data = {"pack": "kit", "units_per_metre": 1.0, "pieces": {"Strip": strip, "Cord": cord}}
    style = Style(units_per_metre=1.0, page_camera={}, dials={})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return style, load_pack(path, style)


def plugged(*, connector=True, plug_x=0.0, base_z=0.5, plug_y=(0.03, 0.05),
            blade_y=(0.03, 0.04), touch=TOUCH) -> SceneCtx:
    """The cord's plug stands on the strip's housing at the socket (`plug_x` slides the cord along
    the strip), the cord's base `base_z` along z; the switch stands apart on the housing."""
    style, pack = plug_pack(touch)
    if not connector:
        cord = dataclasses.replace(pack.pieces["Cord"], connectors=())
        pack = dataclasses.replace(pack, pieces={**pack.pieces, "Cord": cord})
    geometry = {
        "kit:Strip": PieceGeom("Strip", {"housing": box((0, 0, 0), (0.4, 0.03, 0.05)),
                                         "switch": box((0.30, 0.03, 0.01), (0.34, 0.04, 0.04))}),
        "kit:Cord": PieceGeom("Cord", {
            "base": box((0, 0, base_z), (0.1, 0.05, base_z + 0.1)),
            "plug": box((0.08, plug_y[0], 0.005), (0.12, plug_y[1], 0.045)),
            "blade": box((0.09, blade_y[0], 0.015), (0.11, blade_y[1], 0.035)),  # on the housing's top
            "wire": box((0.05, 0.0, 0.05), (0.1, 0.01, base_z))}),
    }
    layout = Layout("yard", "gltf", {
        "strip": LayoutPlacement("strip", "kit", "Strip", (0.0, 0.0, 0.0), Q_ID),
        "cord": LayoutPlacement("cord", "kit", "Cord", (plug_x, 0.0, 0.0), Q_ID)}, {})
    return SceneCtx(style=style, packs={"kit": pack}, clips={}, samples={}, rigs={}, rig_of_clip={},
                    layouts={"yard": layout}, pieces=geometry)


PLUG_ITEM = "yard cord x strip"


def test_a_plug_seated_in_its_socket_is_not_a_clearance_fault():
    ctx = plugged()
    assert findings(ctx) == []
    line = [x for x in Clearance().coverage(ctx) if "seats a plug" in x]
    assert len(line) == 1 and "housing with blade" in line[0] and "housing with plug" in line[0]
    assert "housing with base" not in line[0] and "switch" not in line[0]


def test_the_same_scene_with_no_connector_is_a_fault():
    assert items(plugged(connector=False)) == [PLUG_ITEM]


def with_records(ctx, *, part="housing", plug="plug", mount="socket"):
    """The scene with the strip's mount naming `part` (or none) and the cord's end naming `plug`
    (or none)."""
    pack = ctx.packs["kit"]
    strip, cord = pack.pieces["Strip"], pack.pieces["Cord"]
    mounts = tuple(dataclasses.replace(m, part=part) if m.id == mount else m
                   for m in strip.mounts)
    (conn,) = cord.connectors
    layout_end, own_end = conn.ends  # the first end plugs into the strip through the layout
    cord = dataclasses.replace(cord, connectors=(dataclasses.replace(
        conn, ends=(dataclasses.replace(layout_end, plug=plug), own_end)),))
    pieces = {**pack.pieces, "Strip": dataclasses.replace(strip, mounts=mounts), "Cord": cord}
    return dataclasses.replace(ctx, packs={"kit": dataclasses.replace(pack, pieces=pieces)})


def test_a_mount_that_names_no_part_excuses_nothing_and_says_so():
    ctx = with_records(plugged(), part=None)
    assert items(ctx) == [PLUG_ITEM]  # the part is not guessed from where the point lies
    (line,) = [x for x in Clearance().coverage(ctx) if "no contact is excused" in x]
    assert not isinstance(line, Unmeasured)
    assert PLUG_ITEM in line and "the mount names no host part" in line
    assert not [x for x in Clearance().coverage(ctx) if "seats a plug" in x]


def test_an_end_that_names_no_plug_excuses_nothing_and_says_so():
    ctx = with_records(plugged(), plug=None)
    assert items(ctx) == [PLUG_ITEM]
    (line,) = [x for x in Clearance().coverage(ctx) if "no contact is excused" in x]
    assert "the end names no plug" in line


def test_the_host_part_is_the_one_the_mount_names_not_the_one_nearest_its_point():
    # the socket's point lies on the housing; the mount names the switch instead, so the plug's
    # contact with the housing is not excused
    ctx = with_records(plugged(), part="switch")
    assert items(ctx) == [PLUG_ITEM]
    (line,) = [x for x in Clearance().coverage(ctx) if "seats a plug" in x]
    assert "switch with plug" in line and "housing" not in line


def test_a_plug_touching_the_strip_outside_its_socket_is_still_found():
    # slid 0.2 along the strip, the plug stands on the switch, a part the socket does not sit on
    ctx = plugged(plug_x=0.2)
    assert items(ctx) == [PLUG_ITEM]
    (found,) = findings(ctx)
    assert "touch or overlap" in found.detail


def test_another_part_of_the_cord_near_the_strip_is_still_found():
    assert items(plugged(base_z=0.06)) == [PLUG_ITEM]  # the base is 1 cm off the housing
    assert items(plugged(base_z=0.5)) == []


def test_a_second_placement_of_the_cord_seats_nothing():
    ctx = plugged()
    layout = ctx.layouts["yard"]
    twin = LayoutPlacement("twin", "kit", "Cord", (0.0, 0.0, 0.0), Q_ID)
    ctx = dataclasses.replace(ctx, layouts={"yard": dataclasses.replace(
        layout, placements={**layout.placements, "twin": twin})})
    assert PLUG_ITEM in items(ctx)  # which placement is plugged is not said


def test_the_plant_is_caught_on_the_plug_scene_and_on_the_shop_scene():
    for name, ctx in (("plug", plugged()), ("shop", shop_with_lamp_resting())):
        assert tried_and_caught(Clearance(), ctx), name
        assert plants_caught(Clearance(), ctx), name
    assert run_gate([Clearance()], plugged(), None).status == "passed"


# A seated plug excuses contact, not penetration. These scenes are seen from a distance at which
# the contact length is 3.2 mm.
THIN = 0.0032


def test_a_plug_sunk_into_the_housing_is_still_found():
    ctx = plugged(plug_y=(0.02, 0.04), touch=THIN)  # sunk 10 mm
    assert findings(plugged(touch=THIN)) == []
    assert items(ctx) == [PLUG_ITEM]
    (found,) = findings(ctx)
    assert "plug" in found.detail and "housing" in found.detail and "sink" in found.detail


def test_a_plug_sunk_into_a_rotated_host_is_found():
    """The lamp's strip is turned a quarter turn in the layout `desk`: a closed part read in the
    layout's frame loses its exact corners, so closedness is read on the piece's own triangles.
    The lamp is seen from 0.5 m, so its contact length is 0.145 mm: a plug sunk 0.05 mm is in
    contact, and one sunk 0.3 mm or more is found."""
    from model_kit.core.scene import load_scene

    ctx = load_scene(Path(__file__).resolve().parents[3] / "bars" / "05-lamp" / "scene")
    assert findings(ctx) == []

    def sunk(mm):
        geom = ctx.pieces["lamp:Lamp"]
        down = np.array([0.0, -mm / 1000.0, 0.0])  # the lamp is not turned: its y is the world's
        parts = {**geom.parts, **{n: geom.parts[n] + down
                                  for n in ("plug", "blade_hot", "blade_neutral")}}
        return dataclasses.replace(ctx, pieces={**ctx.pieces, "lamp:Lamp": PieceGeom(
            geom.name, parts)})

    assert findings(sunk(0.05)) == []
    for mm in (0.3, 1.0, 8.0, 13.0):
        (found,) = findings(sunk(mm))
        assert found.item == "desk lamp x strip" and "sink" in found.detail, (mm, found)


def test_a_plug_driven_through_the_housing_is_still_found():
    assert items(plugged(plug_y=(-0.01, 0.05), touch=THIN)) == [PLUG_ITEM]


def test_a_plug_sunk_less_than_the_touch_distance_is_contact():
    sunk = 0.03 - THIN / 2
    assert items(plugged(plug_y=(sunk, sunk + 0.02), touch=THIN)) == []


def test_the_plant_does_not_call_the_depth_test(monkeypatch):
    from model_kit.core.checks import clearance

    ctx = plugged(touch=THIN)
    monkeypatch.setattr(clearance, "_shared_depth", lambda *a, **k: 1 / 0)
    assert [u.item for u in Clearance().plants(ctx)] == [PLUG_ITEM]


def test_a_blade_sunk_into_the_housing_is_still_found():
    assert items(plugged(blade_y=(0.02, 0.04), touch=THIN)) == [PLUG_ITEM]


# --- the contact length is the piece's own -----------------------------------------------------


def without_view(ctx, piece):
    """The scene with one piece of the pack `kit` declaring no view."""
    pack = ctx.packs["kit"]
    bare = dataclasses.replace(pack.pieces[piece], view=None)
    return dataclasses.replace(ctx, packs={"kit": dataclasses.replace(
        pack, pieces={**pack.pieces, piece: bare})})


def test_a_resting_piece_with_no_view_has_no_contact_length_and_is_not_judged():
    ctx = unviewed(stacked(10 * TOUCH))  # lifted far off its host, were a length to judge it
    assert items(stacked(10 * TOUCH)) == [REST_ITEM]
    assert findings(ctx) == []
    assert not [u for u in Clearance().plants(ctx) if "rests on" in u.item]
    (line,) = [x for x in Clearance().coverage(ctx) if "no contact length" in x]
    assert isinstance(line, Unmeasured) and line.endswith(f": {REST_ITEM}")
    assert run_gate([Clearance()], ctx, None).status == "partial"


def test_a_seated_plug_with_no_view_reads_unmeasured_on_its_sink_but_not_its_gap():
    ctx = plugged(plug_y=(0.02, 0.04), touch=THIN)  # sunk 10 mm
    assert items(ctx) == [PLUG_ITEM]
    bare = without_view(ctx, "Cord")
    assert items(bare) == []  # the sink needs both pieces' lengths: it is not read
    (line,) = [x for x in Clearance().coverage(bare) if "no contact length" in x]
    assert isinstance(line, Unmeasured) and line.endswith(f": {PLUG_ITEM}")
    assert run_gate([Clearance()], bare, None).status == "partial"


def test_the_sink_bound_is_the_shorter_of_the_two_pieces_lengths():
    sunk = 0.004  # 4 mm: under the cord's contact length (10.7 mm), over the strip's (3.2 mm)
    ctx = plugged(plug_y=(0.03 - sunk, 0.05), touch=TOUCH)
    assert items(ctx) == []
    pack = ctx.packs["kit"]
    strip = dataclasses.replace(pack.pieces["Strip"], view=viewed(ctx, THIN).packs["kit"]
                                .pieces["Strip"].view)
    near = dataclasses.replace(ctx, packs={"kit": dataclasses.replace(
        pack, pieces={**pack.pieces, "Strip": strip})})
    assert items(near) == [PLUG_ITEM]


# --- a mount that names a part the check cannot measure ----------------------------------------


def with_mount_part_printed(ctx, part="switch"):
    """The scene with the strip's part `part` a printed one (role `eyes`)."""
    pack = ctx.packs["kit"]
    strip = pack.pieces["Strip"]
    parts = tuple(dataclasses.replace(p, role="eyes") if p.name == part else p
                  for p in strip.parts)
    pieces = {**pack.pieces, "Strip": dataclasses.replace(strip, parts=parts)}
    return dataclasses.replace(ctx, packs={"kit": dataclasses.replace(pack, pieces=pieces)})


# The cord's plug and blade stand off a corner of the strip's one measured solid, within the gap
# in each axis (so the boxes meet) but over it by distance: the contact rule is the only one left.
OFF_HOUSING_END = {"plug_x": 0.339, "plug_y": (0.049, 0.059), "blade_y": (0.05, 0.055)}
OFF_SWITCH = {"plug_x": 0.279, "plug_y": (0.059, 0.069), "blade_y": (0.06, 0.065)}


def test_a_mount_that_names_a_printed_part_seats_nothing_and_says_so():
    # Looking the printed part up among the host's solids would raise KeyError.
    ctx = with_records(with_mount_part_printed(plugged(**OFF_HOUSING_END)), part="switch")
    assert findings(ctx) == []
    (line,) = [x for x in Clearance().coverage(ctx) if "no contact is excused" in x]
    assert not isinstance(line, Unmeasured)
    assert PLUG_ITEM in line and "the mount's part switch is no solid with geometry" in line
    assert not [x for x in Clearance().coverage(ctx) if "seats a plug" in x]


def test_a_mount_that_names_a_part_with_no_geometry_seats_nothing_and_says_so():
    ctx = plugged(**OFF_SWITCH)
    strip = ctx.pieces["kit:Strip"]
    bare = PieceGeom("Strip", {k: v for k, v in strip.parts.items() if k != "housing"})
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Strip": bare})
    assert findings(ctx) == []
    (line,) = [x for x in Clearance().coverage(ctx) if "no contact is excused" in x]
    assert PLUG_ITEM in line and "the mount's part housing is no solid with geometry" in line


def test_a_mount_that_names_a_printed_part_still_judges_the_contact():
    ctx = with_records(with_mount_part_printed(plugged()), part="switch")
    assert items(ctx) == [PLUG_ITEM]  # the plug touches the housing, which nothing excuses
