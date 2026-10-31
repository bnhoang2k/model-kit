"""Check detail: a curve has the facets the eye can see at the piece's nearest viewing distance, and
no more.

Rule 1 (blocky) and rule 2 (finer than the eye) are planted on two pieces each: the test humanoid's
capsule limb (a rigged piece, crease 25) and a fixture table leg (an assembly piece, crease 35).
A ring of N sides turns 360 / N degrees; the limb and the leg use N = 32 as the good build, 16 as
the blocky one and 64 as the wasteful one, at 1 m (the eye resolves 0.29 mm there).
"""

import dataclasses
import functools
import math

import numpy as np
import pytest

from model_kit.core import gate, sight
from model_kit.core.checks import detail, pieces
from model_kit.core.checks.detail import Detail, faces_on_sides
from model_kit.core.edges import interior_edges, sagittas, segments_for
from model_kit.core.geom import signed_volume
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.mesh import limb_tris, round_sides, tube_tris
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import CONE
from model_kit.core.manifest import View
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import EYE_RESOLUTION_ARCMIN
from model_kit.core.units import Style

from ..bodies.bar03 import bar03_scene
from ..bodies.build import sweep
from ..bodies.furniture import turned_leg_tris
from ..bodies.humanoid import ELBOW_Y, SHOULDER_X, humanoid_scene
from ..shapes import box
from ..test_edges import revolve, sphere
from .branching.branching_scene import (
    BUD_SPECIES,
    UPM,
    tube_skeleton,
    v2_rules_file,
    viewed_ctx,
)
from .piece_scenes import make_ctx, refuses_the_core, tried_and_caught
from .sight_scenes import unviewed

CHECK = Detail()
D = 1.0  # the nearest viewing distance of the planted pieces, in metres
EYE_PER_METRE = math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60.0))  # 0.29 mm a metre


def viewed(ctx, pack, name, distance=D, **more):
    """The scene with a `view` declared on one piece."""
    piece = ctx.packs[pack].pieces[name]
    view = View(distance, "a test piece, seen from this far", **more)
    return pieces.with_piece(ctx, pack, name, dataclasses.replace(piece, view=view))


def thing(parts, specs=None, distance=D, **more):
    """A scene of the assembly piece `kit:Thing` with a view (crease angle 35)."""
    return viewed(make_ctx(parts, specs), "kit", "Thing", distance, **more)


def limb(segments, radius=0.04, length=0.3):
    """A round capsule limb about y whose ring and end caps both turn 360 / segments degrees: the
    humanoid's upper arm, at its shoulder and elbow."""
    top, bottom = ELBOW_Y + length, ELBOW_Y
    verts, faces = sweep([(top, SHOULDER_X, 0.0, radius, radius),
                          (bottom, SHOULDER_X, 0.0, radius, radius)], "y",
                         segments=segments, cap_rings=segments // 4 - 1)
    return verts.astype(float)[faces]


@functools.cache
def humanoid():
    """The test humanoid (crease 25) with a view on its person: seen from 1 m."""
    return viewed(humanoid_scene(), "humanoid", "Person")


def humanoid_limb(segments):
    ctx = humanoid()
    view = next(iter(pieces.measured_pieces(ctx, CHECK.kinds, {"humanoid:Person "})))
    geometry = {p.name: p.tris for p in view.parts} | {"arm_L": limb(segments)}
    return pieces.with_geometry(ctx, view, geometry), "humanoid:Person arm_L"


def leg(segments):
    return thing({"leg": turned_leg_tris(segments)}), "kit:Thing leg"


BODIES = {"limb": humanoid_limb, "table leg": leg}


def findings(ctx, item):
    return [f for f in CHECK.run(ctx, only={f"{item} blocky at {D:g} m",
                                            f"{item} finer than the eye at {D:g} m"})]


def caught(plant) -> bool:
    return any(f.item == plant.item for f in CHECK.run(plant.ctx, only=plant.scope))


def test_the_check_is_registered_and_states_its_principle():
    assert gate.CHECKS["detail"] is Detail
    assert Detail.kinds == frozenset({"assembly", "branching"})
    assert CHECK.principle.endswith(".") and "http" in CHECK.source


# --- the limit: what the eye resolves at the declared distance --------------------------------


def test_the_limit_is_the_eye_at_the_distance_in_pack_units():
    ctx = thing({"leg": turned_leg_tris(32)}, distance=2.0)
    (view,) = pieces.measured_pieces(ctx, CHECK.kinds)
    assert sight.seen_floor(view.piece, 1.0) == pytest.approx(2.0 * EYE_PER_METRE, rel=1e-12)
    assert sight.seen_floor(view.piece, 2.0) == pytest.approx(4.0 * EYE_PER_METRE, rel=1e-12)
    assert 0.29e-3 < sight.seen_floor(viewed_one(1.0).piece, 1.0) < 0.2910e-3


def viewed_one(distance):
    (view,) = pieces.measured_pieces(thing({"leg": turned_leg_tris(32)}, distance=distance),
                                     CHECK.kinds)
    return view


def worst_sagitta(tris) -> float:
    size = float(np.linalg.norm(tris.reshape(-1, 3).max(0) - tris.reshape(-1, 3).min(0)))
    edges = interior_edges(tris, 8e-6 * size)
    return float(sagittas(tris, edges)[edges.turn_deg < 35.0].max())


def test_a_curve_is_blocky_just_inside_the_eyes_distance_and_fine_just_outside():
    tris = turned_leg_tris(24)
    near = worst_sagitta(tris) / EYE_PER_METRE  # the distance where the gap is exactly the limit
    item = "kit:Thing leg"
    blocky = CHECK.run(thing({"leg": tris}, distance=near * 0.99))
    assert [f.item for f in blocky] == [f"{item} blocky at {near * 0.99:g} m"]
    assert CHECK.run(thing({"leg": tris}, distance=near * 1.01)) == []


def test_the_same_curve_in_other_units_gives_the_same_verdict():
    tris = turned_leg_tris(16)
    metres = CHECK.run(thing({"leg": tris}))
    ctx = thing({"leg": tris * 2.0})
    pack = ctx.packs["kit"]
    ctx = dataclasses.replace(ctx, packs={"kit": dataclasses.replace(pack, units_per_metre=2.0)})
    assert [f.item for f in CHECK.run(ctx)] == [f.item for f in metres] != []


# --- rule 1: blocky ---------------------------------------------------------------------------


def test_blocky_plant_caught_on_two_pieces():
    """Halve the ring segments (N = 32 to 16): the sagitta multiplies by about 4, from under the
    eye's limit at 1 m to over it, on the limb and on the table leg."""
    for label, make in BODIES.items():
        good, item = make(32)
        halved, _ = make(16)
        assert findings(good, item) == [], label
        (found,) = findings(halved, item)
        assert found.item == f"{item} blocky at {D:g} m", label
        assert "add segments" in found.detail and "1 arcmin" in found.detail, label


def test_a_blocky_finding_names_the_worst_edge_its_gap_and_its_turn():
    ctx, item = leg(16)
    (found,) = CHECK.run(ctx)
    assert found.item == f"{item} blocky at 1 m"
    assert "edge " in found.detail and "22.50 deg" in found.detail and "mm" in found.detail


# --- rule 2: finer than the eye ---------------------------------------------------------------


def test_wasted_plant_caught_on_two_pieces():
    """Double the segments (N = 32 to 64) on a piece declared from 1 m: half of them would still
    pass rule 1 and their doubled turn stays under the crease angle."""
    for label, make in BODIES.items():
        good, item = make(32)
        fine, _ = make(64)
        assert findings(good, item) == [], label
        (found,) = findings(fine, item)
        assert found.item == f"{item} finer than the eye at {D:g} m", label
        assert "halve the segments" in found.detail, label


def test_halving_past_crease_is_not_waste():
    """A ring of 20 sides turns 18 degrees; doubled it is 36, past this scene's crease of 35.
    From far away the halved ring would pass rule 1, but halving would add a drawn line. A ring of
    40 sides, doubled 18, is waste."""
    far = 10.0
    coarse = thing({"ring": revolve([(0.0, 0.0), (0.0, 0.05), (0.2, 0.05), (0.2, 0.0)], 20)},
                   distance=far)
    assert CHECK.run(coarse) == []
    assert any("rule 2 not applicable" in line for line in CHECK.coverage(coarse))
    fine = thing({"ring": revolve([(0.0, 0.0), (0.0, 0.05), (0.2, 0.05), (0.2, 0.0)], 40)},
                 distance=far)
    assert [f.item for f in CHECK.run(fine)] == [f"kit:Thing ring finer than the eye at {far:g} m"]


def test_sphere_poles_are_not_waste():
    """A UV sphere whose rings pass rule 2 is not failed by its small pole triangles; a finer one
    is waste, judged by its sagitta and not by its triangles. Radius 0.1 m at 4 m."""
    scale = 0.1

    def ball(rings, segments):
        return {"ball": sphere(rings, segments) * scale}

    assert CHECK.run(thing(ball(12, 24), distance=4.0)) == []
    waste = CHECK.run(thing(ball(36, 72), distance=4.0))
    assert [f.item for f in waste] == ["kit:Thing ball finer than the eye at 4 m"]


# --- not applicable, unmeasured, skipped ------------------------------------------------------


def test_box_is_not_applicable():
    from ..shapes import box

    ctx = thing({"block": box((0, 0, 0), (0.2, 0.1, 0.05))})
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == []
    lines = CHECK.coverage(ctx)
    assert any("not applicable (no smooth edge" in line and "kit:Thing block" in line
               for line in lines)
    assert not any(isinstance(line, Unmeasured) for line in lines)
    assert run_gate([Detail()], ctx, None).status == "passed"


def test_no_view_is_partial():
    ctx = unviewed(make_ctx({"leg": turned_leg_tris(16)}))  # no view declared
    assert CHECK.run(ctx) == []
    (line,) = [line for line in CHECK.coverage(ctx) if isinstance(line, Unmeasured)]
    assert "no view" in line and "kit:Thing" in line
    assert CHECK.measured(ctx) == set()  # the check could not judge any part
    assert run_gate([Detail()], ctx, None).status == "partial"


def test_no_crease_angle_is_partial():
    ctx = thing({"leg": turned_leg_tris(16)})
    ctx = dataclasses.replace(ctx, style=Style(1.0, {}, {}))
    assert CHECK.run(ctx) == []
    (line,) = [line for line in CHECK.coverage(ctx) if isinstance(line, Unmeasured)]
    assert "crease_deg" in line
    assert run_gate([Detail()], ctx, None).status == "partial"


def test_the_style_without_an_outline_still_judges_smooth_shading():
    ctx = thing({"leg": turned_leg_tris(16)})
    edges = {"outline": "none", "crease_deg": 35, "crease_why": "a test value"}
    ctx = dataclasses.replace(ctx, style=Style(1.0, {}, {"edges": edges}))
    assert [f.item for f in CHECK.run(ctx)] == ["kit:Thing leg blocky at 1 m"]


def test_decal_and_eyes_skipped():
    """A decal follows its host and an eyes part draws no outline: both are coarse and silent."""
    coarse = turned_leg_tris(16)
    ctx = thing({"leg": turned_leg_tris(32), "mark": coarse, "eye": coarse},
                {"mark": {"shape": "decal", "role": "decal"}, "eye": {"role": "eyes"}})
    assert CHECK.run(ctx) == []
    lines = CHECK.coverage(ctx)
    assert "detail: not applicable (decal part): kit:Thing mark" in lines
    assert "detail: not applicable (eyes part): kit:Thing eye" in lines
    assert CHECK.measured(ctx) == {"kit:Thing leg"}


def test_a_branching_piece_is_judged_too():
    ctx = unviewed(make_ctx({"leg": turned_leg_tris(32)},
                            also={"Tree": {"stem": turned_leg_tris(16)}}))
    ctx = viewed(ctx, "kit", "Tree")
    ctx = viewed(ctx, "kit", "Thing")
    assert [f.item for f in CHECK.run(ctx)] == ["kit:Tree stem blocky at 1 m"]


# --- hidden sides -----------------------------------------------------------------------------


def half_tube(segments, radius=0.05, height=0.2):
    """A closed D-shaped bar about y: a flat front at z = 0 facing +z and a half-round back."""
    theta = np.pi * np.arange(segments + 1) / segments
    low = np.stack([radius * np.cos(theta), np.zeros_like(theta), -radius * np.sin(theta)], 1)
    high = low + [0.0, height, 0.0]
    tris = []
    for k in range(segments):
        tris += [[low[k], high[k + 1], low[k + 1]], [low[k], high[k], high[k + 1]]]
    for k in range(1, segments):
        tris += [[low[0], low[k], low[k + 1]], [high[0], high[k + 1], high[k]]]
    tris += [[low[0], low[segments], high[segments]], [low[0], high[segments], high[0]]]
    tris = np.array(tris)
    return tris if signed_volume(tris) > 0 else tris[:, ::-1]


def test_faces_belong_to_the_side_their_normal_points_most_toward():
    tris = half_tube(8)
    assert not interior_edges(tris, 1e-9).open_boundary.any()  # closed, wound one way
    every = ("front", "back", "left", "right", "top", "bottom")
    assert faces_on_sides(tris, every).all()
    assert not faces_on_sides(tris, ()).any()
    front = faces_on_sides(tris, ("front",))
    assert front.sum() == 2 and np.allclose(tris[front][:, :, 2], 0.0)
    assert faces_on_sides(tris, ("top", "bottom")).sum() == 2 * 7  # the two flat ends


def test_a_hidden_side_is_not_judged_but_a_visible_face_still_shows_its_facet():
    coarse = {"bar": half_tube(10)}
    assert [f.item for f in CHECK.run(thing(coarse))] == ["kit:Thing bar blocky at 1 m"]
    back_only = thing(coarse, hidden=("back",), hidden_why="the bar stands against a wall")
    assert [f.item for f in CHECK.run(back_only)] == ["kit:Thing bar blocky at 1 m"]
    all_but_front = thing(coarse, hidden=("back", "left", "right", "top", "bottom"),
                          hidden_why="only the flat front is seen")
    assert CHECK.run(all_but_front) == []
    assert any("no smooth edge in view" in line for line in CHECK.coverage(all_but_front))


# --- the check's own plants -------------------------------------------------------------------


def units_for(ctx, part, word):
    """The plant units of one part for one rule (`blocky` or `finer than the eye`)."""
    return [u for u in CHECK.plants(ctx) if u.item.startswith(f"{part} ") and word in u.item]


def test_the_plants_never_call_the_verdict(monkeypatch):
    """The units are built from the part's gaps and the piece's view. The functions that decide
    blocky and wasted are not called."""
    ctx, item = leg(32)
    refuses_the_core(monkeypatch, detail, "_blocky", "_wasted")
    assert units_for(ctx, item, "blocky") and units_for(ctx, item, "finer than the eye")


def test_the_checks_own_plants_are_caught_on_two_pieces():
    """Rule 1 scales one part about its centroid (every sagitta by one factor, every turn kept),
    so it needs no ring to halve. Rule 2 declares the piece from so far that the part's merged gap
    passes. Each unit is tried and caught on the limb and on the table leg, quiet before."""
    for label, make in BODIES.items():
        ctx, item = make(32)
        tried = tried_and_caught(CHECK, ctx)
        assert f"{item} blocky at {D:g} m" in tried, label
        (waste,) = units_for(ctx, item, "finer than the eye")
        assert waste.item in tried and waste.item not in {f.item for f in CHECK.run(ctx)}, label
        (blocky,) = units_for(ctx, item, "blocky")
        assert blocky.item.startswith(item.split()[0]) and len(blocky.candidates) == 1, label


def test_a_blocky_part_can_still_carry_the_rule_2_plant():
    """Bar 03 seen from 1 m has blocky parts and no waste. A part that fails rule 1 still gets a
    rule 2 unit: the piece declared from far enough away for half its segments to pass."""
    ctx = viewed(bar03_scene(), "chibi", "chibi")
    assert any(f.item.endswith("blocky at 1 m") for f in CHECK.run(ctx))
    tried = tried_and_caught(CHECK, ctx)
    assert [i for i in tried if "finer than the eye" in i]
    assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]


def test_the_rule_2_plant_declares_the_piece_from_so_far_that_half_the_segments_pass():
    """The item names the new distance, and the part is the one the check judges."""
    ctx, item = leg(24)
    assert [f.item for f in CHECK.run(ctx)] == [f"{item} blocky at 1 m"]
    (unit,) = units_for(ctx, item, "finer than the eye")
    (plant,) = unit.candidates
    assert plant.label.startswith("declared from") and caught(plant)
    assert unit.item != f"{item} finer than the eye at 1 m"


def test_the_rule_2_plant_keeps_a_farthest_view_legal():
    """A piece that declares a farthest view still gets the plant: declared from far away, its
    farthest view moves out with its nearest, so the `View` stays legal."""
    ctx = thing({"leg": turned_leg_tris(24)}, farthest_m=2 * D, farthest_why="a test range")
    (unit,) = [u for u in CHECK.plants(ctx) if "finer than the eye" in u.item]
    (plant,) = unit.candidates
    declared = plant.ctx.packs["kit"].pieces["Thing"].view
    assert declared.nearest_m > 2 * D and declared.farthest_m == declared.nearest_m
    assert caught(plant)


def test_a_piece_with_no_smooth_edge_has_no_plant_and_no_unmeasured_line():
    from ..shapes import box

    ctx = thing({"block": box((0, 0, 0), (0.2, 0.1, 0.05))})
    assert CHECK.plants(ctx) == []
    assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]


def test_a_piece_whose_every_smooth_part_is_coarse_has_no_rule_2_plant_and_says_so():
    ctx, item = leg(20)  # 18 degrees a ring: doubled it passes the crease angle of 35
    assert not units_for(ctx, item, "finer than the eye")
    assert units_for(ctx, item, "blocky")  # rule 1 is still planted
    assert any("rule 2 not applicable" in line for line in CHECK.coverage(ctx))
    assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]


def test_the_gate_runs_the_plants_and_a_good_piece_passes():
    ctx, _ = leg(32)
    verdict = run_gate([Detail()], ctx, None)
    assert verdict.status == "passed" and verdict.failed == 0


def test_a_blocky_piece_fails_the_gate_and_a_scoped_run_measures_only_its_items():
    ctx, _ = leg(16)
    verdict = run_gate([Detail()], ctx, None)
    assert verdict.status == "failed" and verdict.failed == 1
    assert CHECK.run(ctx, only={"kit:Thing other blocky at 1 m"}) == []


# --- rule 2 judges the gap left by taking an edge out --------------------------


def kink_part(a, b, turn_deg):
    from ..test_edges import kink

    return {"string": kink(a, b, turn_deg)}


def test_a_lone_kink_between_unequal_faces_is_visible_not_waste():
    """The string over the nut: faces 41.24 and 3.47 mm high turn 8.55 degrees. Equal faces would
    leave 0.258 mm if the edge went, under the eye's 0.291 mm at 1 m, so a rule that assumed equal
    faces would call the string waste. The real gap is 0.476 mm: visible at 1 m, waste only from 2 m (0.582 mm)."""
    parts = kink_part(0.04124, 0.00347, 8.55)
    assert CHECK.run(thing(parts, distance=1.0)) == []
    (found,) = CHECK.run(thing(parts, distance=2.0))
    assert found.item == "kit:Thing string finer than the eye at 2 m"
    assert "0.476 mm" in found.detail


def test_a_ring_of_equal_faces_is_still_waste_by_the_halving_formula():
    ctx, item = leg(64)
    (found,) = findings(ctx, item)
    assert found.item.endswith(f"finer than the eye at {D:g} m")


# --- a smooth edge does not depend on the piece's size -------------------------


@pytest.mark.parametrize("segments", [64, 256, 384, 512])
def test_a_finely_cut_tube_is_judged_at_any_segment_count(segments):
    from ..test_edges import revolve

    tube = revolve([(0.0, 0.0), (0.0, 0.05), (0.2, 0.05), (0.2, 0.0)], segments)
    ctx = thing({"tube": tube})
    assert [f.item for f in CHECK.run(ctx)] == [f"kit:Thing tube finer than the eye at {D:g} m"]
    assert not any("no smooth edge" in line for line in CHECK.coverage(ctx))


def test_a_curve_is_judged_the_same_beside_a_part_far_away():
    """Growing the piece must not turn a curve into a flat one: the floor does not scale with the piece."""
    near = thing({"leg": turned_leg_tris(64)})
    far = thing({"leg": turned_leg_tris(64), "far": box((50.0, 0.0, 0.0), (50.1, 0.1, 0.1))})
    assert [f.item for f in CHECK.run(far)] == [f.item for f in CHECK.run(near)] != []


@pytest.mark.held("bars/01-strat")
def test_the_frets_of_bar_01_are_all_judged():
    """Fret_01 has 36 edges turning 0.5 to 0.9 degrees. Like Fret_02 to Fret_22, cut the same way,
    it is judged and does not read 'no smooth edge'. At a scratch view of 1 m (not the bar's data)
    all 22 read the same."""
    from pathlib import Path

    from model_kit.core.scene import load_scene

    scene = Path(__file__).resolve().parents[3] / "bars" / "01-strat" / "scene"
    ctx = viewed(load_scene(scene), "strat", "Strat")
    frets = [f"strat:Strat Fret_{n:02d}" for n in range(1, 23)]
    got = {f.item for f in CHECK.run(ctx, only={f"{f} finer than the eye at 1 m" for f in frets})}
    assert got == {f"{f} finer than the eye at 1 m" for f in frets}
    assert not any("Fret_01" in line and "no smooth edge" in line for line in CHECK.coverage(ctx))


# --- a hidden side: an edge is skipped only when both its faces are hidden ---------------------


def test_an_edge_between_a_hidden_face_and_a_seen_one_is_still_judged():
    """One edge, one face looking front and the other left: with only the front hidden the edge
    still shows the left face's facet, so it is judged; with both hidden it is not. (A rule that
    skipped an edge when either face was hidden would stay silent in the first case.)"""
    from ..test_edges import roof

    turn = np.radians(45.0)
    c, s = np.cos(turn), np.sin(turn)
    rotate_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    turn_edge_upright = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # the roof's edge runs along y now, and its two faces' normals sit 40 and 50 degrees from +z
    # toward +x: one is front by a hair, the other left
    tris = roof(0.2, 10.0) @ turn_edge_upright.T @ rotate_y.T
    side = [next(x for x in ("front", "back", "left", "right", "top", "bottom")
                 if faces_on_sides(tris[k:k + 1], (x,))[0]) for k in (0, 1)]
    assert side[0] != side[1], side
    assert [f.item for f in CHECK.run(thing({"kink": tris}))] == ["kit:Thing kink blocky at 1 m"]
    one = thing({"kink": tris}, hidden=(side[0],), hidden_why="one side stands against a wall")
    assert [f.item for f in CHECK.run(one)] == ["kit:Thing kink blocky at 1 m"]
    both = thing({"kink": tris}, hidden=tuple(side), hidden_why="both sides stand against walls")
    assert CHECK.run(both) == []


# --- a grown tube is cut by the same inequality ----------------------------------------------


GAP = D * EYE_PER_METRE  # what the eye resolves at D, in these metre units
TUBE_SIGHT = sight.Sight(GAP, 35.0)  # the pieces' seen floor and crease angle


def tube_part(sides=None, radius=0.04):
    """The buds-v2 mesh of one straight tube (as `grown_parts` makes it from the piece's sight),
    or, with `sides`, the same tube with that many corners to a ring."""
    sk = tube_skeleton(radius)
    if sides is None:
        return limb_tris(sk, 0, TUBE_SIGHT)
    return tube_tris(sk.limb_points(0)[[0, -1]], np.full(2, radius), sides, CONE)


@pytest.mark.parametrize("radius", [0.04, 0.1])
def test_tube_meshed_with_segments_for_is_quiet_on_two_tubes(radius):
    """The count `segments_for` gives is the least that `detail` passes: the tube the mesher cuts
    has no finding, and its sagitta, not the round floor, decided the sides."""
    n = segments_for(radius, 360.0, GAP)
    assert n > round_sides(35.0)
    tris = tube_part(radius=radius)
    assert np.array_equal(tris, tube_part(n, radius))
    assert CHECK.run(thing({"tube": tris})) == []


@pytest.mark.parametrize("radius", [0.04, 0.1])
def test_one_side_fewer_is_blocky_on_two_tubes(radius):
    n = segments_for(radius, 360.0, GAP)
    found = CHECK.run(thing({"tube": tube_part(n - 1, radius)}))
    assert [f.item for f in found] == [f"kit:Thing tube blocky at {D:g} m"]


@pytest.mark.parametrize("distance", [1.0, 0.3])
def test_the_grown_tiny_trees_are_quiet(tmp_path_factory, distance):
    """The bud grower's two fixture species, grown without `tubes.sides` (buds-v2) and meshed from
    the sight of a piece seen from `distance` metres: the check finds nothing blocky and nothing
    wasteful in any tube part, the lobed trunk included, even close up. It is judged on a branching
    piece, as the bars' trees are. (The mesher raises rather than return a tube it knows is over
    the gap, so a tree that passes here was cut fine enough.)"""
    folder = tmp_path_factory.mktemp("v2-rules")
    for species in BUD_SPECIES:
        sk = grow_buds(load_rules(v2_rules_file(species, folder), kit=KIT_RULES), piece="Tree",
                       height_m=1.5, seed=1, lite=False, units_per_metre=UPM)
        assert sk.info["mesh"] == "buds-v2"
        ctx = viewed_ctx(sk, view=View(distance, "a test tree, seen from this far"))
        assert Detail().measured(ctx), species
        assert Detail().run(ctx) == [], species
