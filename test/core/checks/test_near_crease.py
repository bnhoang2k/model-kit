"""Check near-crease: a drawn line follows a real edge.

Rule A (an edge within reach of the crease angle, or bent across it) and rule B (a drawn line that
stops mid-surface) are planted on at least two pieces each. The torso and arm fixtures are chibi
torsos and arms that carry the faults these rules judge (`bodies/test_regressions.py`).
"""

import dataclasses
import functools
import math
import re

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import near_crease, pieces
from model_kit.core.checks.detail import faces_on_sides
from model_kit.core.checks.near_crease import NearCrease
from model_kit.core.edges import interior_edges, line_ends, position_step, rounding_band_deg
from model_kit.core.errors import BlindCheck
from model_kit.core.geom import signed_volume
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import WELD_OF_PIECE_SIZE
from model_kit.core.units import Style

from ..bodies.bar03 import bar03_scene
from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.test_regressions import _chibi, _mesh
from ..shapes import box
from ..test_edges import revolve, sphere
from .piece_scenes import make_ctx, refuses_the_core, tried_and_caught
from .sight_scenes import unviewed, viewed

CREASE = 25.0
UNKNOWN = object()  # the piece records no measured.export


def style(angle=CREASE, outline="runtime") -> Style:
    edges = {"outline": outline}
    if angle is not None:
        edges |= {"crease_deg": angle, "crease_why": "a test value, set by the test"}
    return Style(1.0, {}, {"edges": edges})


def with_export(ctx, bits):
    """The scene with `measured.export.position_bits` set on every piece (None is float32)."""
    for pack_name, pack in ctx.packs.items():
        for name, piece in pack.pieces.items():
            measured = {**piece.measured, "export": {"position_bits": bits}}
            ctx = pieces.with_piece(ctx, pack_name, name, dataclasses.replace(piece,
                                                                              measured=measured))
    return ctx


def scene(parts, specs=None, *, bits=None, angle=CREASE, outline="runtime", also=None):
    ctx = dataclasses.replace(unviewed(make_ctx(parts, specs, also=also)),
                              style=style(angle, outline))
    return ctx if bits is UNKNOWN else with_export(ctx, bits)


def crease_25(ctx, bits=None):
    """A scene with the test crease angle, an outline, and (with `bits`) a shipped precision."""
    ctx = dataclasses.replace(ctx, style=style())
    return ctx if bits is UNKNOWN else with_export(ctx, bits)


@functools.cache
def bar03():
    """The chibi with crease 25 and a 14-bit file, both set here in memory (its style says
    35 and its pack records no export)."""
    return crease_25(bar03_scene(), 14)


@functools.cache
def humanoid():
    return crease_25(unviewed(humanoid_scene()))


def lathe_parts():
    return {"ball": sphere(12, 24)}


CHECK = NearCrease()


def kind_of(item: str) -> str:
    return re.search(r"( near| bends across| sliver| line end(?: \d+)?)$", item).group(1).split()[0]


def plants_of(check, ctx, kind, key=""):
    """The first candidate of each plant unit of this kind (`near`, `line` or `bends`), of the
    piece `key` when one is given."""
    return [u.candidates[0] for u in check.plants(ctx)
            if kind_of(u.item) == kind and u.item.startswith(key)]


def caught(check, plant) -> bool:
    return any(f.item == plant.item
               for f in check.run(plant.ctx, only=plant.scope, samples=plant.samples))


def fixture_of(plant) -> str:
    """The name of the part a plant added to its piece (the plant's item names it)."""
    return plant.item.split()[1]


def test_the_check_is_registered_and_states_its_principle():
    assert gate.CHECKS["near-crease"] is NearCrease
    assert NearCrease.kinds == frozenset({"assembly", "branching"})
    assert CHECK.principle.endswith(".") and "http" in CHECK.source


def test_a_smooth_scene_passes_and_its_plants_are_caught():
    ctx = scene(lathe_parts())
    assert CHECK.run(ctx) == []
    kinds = {kind_of(u.item) for u in CHECK.plants(ctx)}
    assert kinds == {"near", "line"}  # a rigid ball has no bending to plant
    assert len(tried_and_caught(CHECK, ctx)) == 3  # a fold, a plate and a ring step
    assert run_gate([NearCrease()], ctx, None).status == "passed"


# --- rule A -----------------------------------------------------------------------------------


def test_rule_a_plant_caught_on_two_pieces():
    """An edge folded to half a band over the crease angle, on a bar 03 part and on a fixture
    assembly part."""
    fixture = scene(lathe_parts(), bits=14)
    for label, ctx in (("bar 03", bar03()), ("fixture", fixture)):
        (plant,) = plants_of(CHECK, ctx, "near")
        assert caught(CHECK, plant), label
        assert plant.item not in {f.item for f in CHECK.run(ctx, only={plant.item})}, label
    assert re.fullmatch(r"chibi:chibi \w+ edge \d+ near", plants_of(CHECK, bar03(), "near")[0].item)
    assert plants_of(CHECK, fixture, "near")[0].item.startswith("kit:Thing ball edge ")


def test_rule_a_plant_sits_half_a_band_over_the_angle():
    ctx = scene(lathe_parts(), bits=14)
    (plant,) = plants_of(CHECK, ctx, "near")
    j = int(re.search(r"edge (\d+) near", plant.item).group(1))
    (view,) = pieces.measured_pieces(plant.ctx, {"assembly"})
    (part,) = [p for p in view.parts if p.name == fixture_of(plant)]
    found = interior_edges(part.tris, WELD_OF_PIECE_SIZE * view.size)
    step = position_step(view.piece, found.points)
    band, capped = rounding_band_deg(found.points, found, step, CREASE)
    assert not capped[j]
    assert found.turn_deg[j] - CREASE == pytest.approx(0.5 * band[j], rel=0.2)


def test_rounding_decides_the_band():
    """The same fold is within reach of the angle at 10 bits and clear of it at float32."""
    ctx = scene(lathe_parts(), bits=14)
    (plant,) = plants_of(CHECK, ctx, "near")
    coarse = with_export(plant.ctx, 8)
    fine = with_export(plant.ctx, None)
    assert any(f.item.endswith(" near") for f in CHECK.run(coarse))
    assert not any(f.item.endswith(" near") for f in CHECK.run(fine))


def test_a_capped_band_is_one_sliver_finding_per_part_and_no_near_crease():
    ctx = scene({"ball": sphere(12, 24), "dome": sphere(8, 16)}, bits=4)
    found = CHECK.run(ctx)
    assert sorted(f.item for f in found if f.item.endswith(" sliver")) == [
        "kit:Thing ball sliver", "kit:Thing dome sliver"]
    assert not any(f.item.endswith(" near") for f in found)


def test_missing_export_is_partial():
    ctx = scene(lathe_parts(), bits=UNKNOWN)
    lines = [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
    assert any("no measured.export" in x and "kit:Thing" in x for x in lines)
    assert run_gate([NearCrease()], ctx, None).status == "partial"
    # rule A needs the precision; nothing on a smooth ball would be found without it
    assert CHECK.run(ctx) == []


def test_rule_b_runs_without_an_export():
    torso = _chibi("torso_dash.npz")
    v, f = _mesh(torso, "torso")
    ctx = scene({"torso": v[f]}, {"torso": {"shape": "ring"}}, bits=UNKNOWN)
    found = CHECK.run(ctx)
    assert found and all("line end" in x.item for x in found)
    assert run_gate([NearCrease()], ctx, None).status == "partial"


# --- rule B -----------------------------------------------------------------------------------


def test_rule_b_plant_caught_on_two_pieces():
    """A flat plate with one raised vertex of three edges: its three spokes turn over the crease
    angle and its ring under it, so three lines stop. It stands on the humanoid person and on a
    fixture lathe, whose own meshes are smooth."""
    for label, ctx, key in (("person", humanoid(), "humanoid:Person"),
                            ("lathe", scene(lathe_parts()), "kit:Thing")):
        for stem in ("dash_plant", "ring_plant"):
            (plant,) = plants_of(CHECK, ctx, "line", f"{key} {stem} ")
            assert caught(CHECK, plant), (label, stem)
            assert plant.item not in {f.item for f in CHECK.run(ctx, only={plant.item})}, label


def plate_turns(crease):
    """The turns of the plate's interior edges at `crease`: (spokes, the others)."""
    tris, top = near_crease._plate(1.0, crease)
    edges = interior_edges(tris, 1e-9)
    peak = near_crease._index_at(edges.points, top)
    spoke = (edges.v == peak).any(axis=1)
    return edges, edges.turn_deg[spoke], edges.turn_deg[~spoke]


@pytest.mark.parametrize("crease", [10.0, 25.0, 35.0, 60.0, 80.0])
def test_the_plate_turns_its_spokes_as_far_over_the_crease_as_its_ring_turns_under(crease):
    edges, spokes, rest = plate_turns(crease)
    assert len(spokes) == 3 and not edges.open_boundary[edges.v[:, 0]].all()
    assert (spokes > crease).all() and (rest < crease).all()
    ring = rest.max()
    assert spokes.min() - crease == pytest.approx(crease - ring, rel=1e-6)


def test_the_plate_is_wound_one_way_and_its_inner_corners_are_the_line_ends():
    tris, _ = near_crease._plate(1.0, 25.0)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert (normal[:, 2] > 0).all()
    edges, _, _ = plate_turns(25.0)
    drawn = edges.turn_deg > 25.0
    ends = line_ends(edges, drawn, edges.open_boundary)
    assert len(ends) == 3 and not edges.open_boundary[ends].any()


@pytest.mark.parametrize("side", ["front", "back", "left", "right", "top", "bottom"])
def test_the_dash_fixtures_turn_to_a_side_the_view_shows(side):
    """With one side hidden, each fixture (the plate and the bent tube) still stands on a side
    that is shown, and the plant is caught."""
    ctx = viewed(scene(lathe_parts()), 1e-9, (side,))
    for stem in ("dash_plant", "ring_plant"):
        (plant,) = plants_of(CHECK, ctx, "line", f"kit:Thing {stem} ")
        assert caught(CHECK, plant), stem
        assert plant.item not in hidden_ends(plant.ctx, stem, [side])[0], stem


def test_a_piece_with_every_side_hidden_has_no_dash_plant_and_says_why():
    every = ("front", "back", "left", "right", "top", "bottom")
    ctx = viewed(scene(lathe_parts()), 1e-9, every)
    (unit,) = [u for u in CHECK.plants(ctx) if u.not_applicable]
    assert unit.candidates == () and "every side as hidden" in unit.not_applicable
    verdict = run_gate([NearCrease()], ctx, None)
    assert "near-crease: not applicable (the view lists every side as hidden, so no line end "\
        "can be seen): kit:Thing" in verdict.coverage
    assert verdict.status == "passed"


def test_a_piece_of_boxes_gets_a_dash_plant():
    """A plant that raises a vertex of a smooth part cannot be planted on a piece made only of
    boxes, which would read partial. The plate stands on any piece."""
    ctx = scene({"block": box((0, 0, 0), (0.2, 0.1, 0.05))}, bits=14)
    assert CHECK.run(ctx) == []
    assert {kind_of(u.item) for u in CHECK.plants(ctx)} == {"near", "line"}
    assert not [x for x in CHECK.coverage(ctx) if "no plant can fail" in x]
    assert run_gate([NearCrease()], ctx, None).status == "passed"


def _ends_at(ctx, mesh) -> list[np.ndarray]:
    """The positions of the line ends the check finds on a one-part scene of `mesh`."""
    found = interior_edges(mesh, WELD_OF_PIECE_SIZE * float(np.linalg.norm(
        mesh.reshape(-1, 3).max(axis=0) - mesh.reshape(-1, 3).min(axis=0))))
    return [found.points[int(x.item.rsplit(" ", 1)[1])]
            for x in CHECK.run(ctx) if "line end" in x.item]


def test_d2_torso_before_fix_has_line_ends():
    """A chibi torso fixture: 8 of the 40 edges of the ring step at y 0.3867 run 0.09 to 0.87 deg
    over the crease angle, so two lines stop at each end of two arcs. The check finds those four
    ends and four more on the ring at y 0.1131, where the torso has a second pair of arcs."""
    z = _chibi("torso_dash.npz")
    v, f = _mesh(z, "torso")
    ctx = scene({"torso": v[f]}, {"torso": {"shape": "ring"}}, bits=None)
    at = _ends_at(ctx, v[f])
    assert len(at) == 8
    rings = sorted(round(float(p[1]), 4) for p in at)
    assert rings == [0.1131] * 4 + [0.3867] * 4


def test_d3_arm_before_fix_has_line_ends():
    """The chibi's old arms: the thumb bump draws a short line (11 and 10 edges over the crease)
    that stops on the thumb."""
    z = _chibi("arm_dash.npz")
    for part in ("arm_L", "arm_R"):
        v, f = _mesh(z, part)
        ctx = scene({part: v[f]}, {part: {"shape": "ring"}}, bits=None)
        thumb = v[z[f"thumb:{part}"]]
        on_thumb = [p for p in _ends_at(ctx, v[f]) if np.abs(thumb - p).max(axis=1).min() < 1e-4]
        assert len(on_thumb) >= 2, part


def test_the_current_bar_03_torso_and_legs_draw_no_dash_at_crease_25():
    ctx = bar03()
    found = CHECK.run(ctx)
    assert not [x for x in found if "line end" in x.item
                and x.item.split(" ")[1] in ("torso", "head", "thigh_L", "thigh_R", "shin_L",
                                             "shin_R")]


def test_a_closed_rim_is_not_a_dash():
    drum = revolve([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], 24)
    assert CHECK.run(scene({"drum": drum})) == []


# --- bending ----------------------------------------------------------------------------------


def test_bend_across_caught_at_a_clip_sample():
    """A skinned edge bent across the crease angle at one clip sample, on two bodies."""
    for label, ctx in (("humanoid", humanoid()), ("quadruped", crease_25(quadruped_scene()))):
        plants = plants_of(CHECK, ctx, "bends")
        assert plants, label
        plant = plants[0]
        assert len(plant.samples) == 1
        found = CHECK.run(plant.ctx, only=plant.scope, samples=plant.samples)
        (hit,) = [f for f in found if f.item == plant.item]
        assert "across the crease angle" in hit.detail or "band" in hit.detail
        assert plant.item not in {f.item for f in CHECK.run(ctx, only={plant.item})}


def test_a_run_scoped_to_samples_reads_the_clips_only():
    ctx = humanoid()
    (plant,) = plants_of(CHECK, ctx, "bends")[:1]
    found = CHECK.run(plant.ctx, samples=plant.samples)
    assert not any(f.item.endswith(" near") or f.item.endswith(" sliver") for f in found)


def test_a_rigid_part_is_measured_at_rest_only():
    ctx = bar03()
    found = CHECK.run(ctx)
    assert not [x for x in found if " head " in x.item and "bends across" in x.item]


def test_a_skinned_part_with_no_clip_samples_is_unmeasured():
    ctx = humanoid()
    bare = dataclasses.replace(ctx, samples={}, rig_of_clip={}, clips={})
    lines = [x for x in CHECK.coverage(bare) if isinstance(x, Unmeasured)
             and "no clip samples" in x]
    assert lines


# --- what it leaves out -----------------------------------------------------------------------


def test_outline_none_not_applicable():
    ctx = scene(lathe_parts(), outline="none")
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == []
    lines = CHECK.coverage(ctx)
    assert any("not applicable" in x for x in lines) and not any(isinstance(x, Unmeasured)
                                                                  for x in lines)
    assert run_gate([NearCrease()], ctx, None).status == "passed"


def test_no_crease_angle_is_unmeasured():
    ctx = scene(lathe_parts(), angle=None)
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == []
    assert run_gate([NearCrease()], ctx, None).status == "partial"
    assert any(isinstance(x, Unmeasured) and "crease_deg" in x for x in CHECK.coverage(ctx))


def borders_only_style() -> Style:
    return Style(1.0, {}, {"edges": {"outline": "runtime", "borders_only": True,
                                     "borders_why": "the page draws silhouette borders only"}})


def test_borders_only_style_prints_not_applicable_not_unmeasured():
    """A look that draws outlines and no fold lines has no angle to measure against, and says so:
    the check is not applicable, with the style's reason, and the run passes."""
    ctx = dataclasses.replace(scene(lathe_parts(), angle=None), style=borders_only_style())
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == [] and CHECK.measured(ctx) == set()
    line = ("near-crease: not applicable (the style draws borders only, so no fold line can dash "
            "or flip: the page draws silhouette borders only)")
    assert CHECK.coverage(ctx) == [line]
    assert run_gate([NearCrease()], ctx, None).status == "passed"


def test_borders_only_is_not_applicable_on_three_pieces():
    """The line is the style's, so it reads the same on a lathe, on bar 03 and on the humanoid."""
    for base in (scene(lathe_parts(), angle=None), bar03(), humanoid()):
        ctx = dataclasses.replace(base, style=borders_only_style())
        lines = CHECK.coverage(ctx)
        assert len(lines) == 1 and "not applicable" in lines[0]
        assert not isinstance(lines[0], Unmeasured) and CHECK.run(ctx) == []


def test_style_with_neither_crease_nor_borders_only_reads_unmeasured():
    """Saying nothing is not the same as saying borders only: `borders_only: false` is still
    silence, and the check reads Unmeasured."""
    for edges in ({"outline": "runtime"}, {"outline": "runtime", "borders_only": False}):
        ctx = dataclasses.replace(scene(lathe_parts(), angle=None),
                                  style=Style(1.0, {}, {"edges": edges}))
        lines = CHECK.coverage(ctx)
        assert len(lines) == 1 and isinstance(lines[0], Unmeasured) and "crease_deg" in lines[0]
        assert run_gate([NearCrease()], ctx, None).status == "partial"


def test_no_outline_roles_skipped():
    """Bar 03's eye lens (32 x 16, rim edges up to 44 deg) as it was: a part of role `eyes` draws
    no outline, so it gives no finding; the same mesh as a solid does."""
    z = _chibi("eye_rims.npz")
    v, f = _mesh(z, "eye_L")
    assert CHECK.run(scene({"lens": v[f]}, {"lens": {"role": "solid"}}, bits=None)) != []
    for spec in ({"role": "eyes"}, {"role": "skip_outline"}, {"shape": "cable"},
                 {"shape": "decal", "role": "decal"}):
        ctx = scene({"lens": v[f], "body": sphere()}, {"lens": spec}, bits=None)
        found = CHECK.run(ctx)
        assert not [x for x in found if x.item.startswith("kit:Thing lens ")], spec
        assert any("not applicable (" in line and " part): kit:Thing lens" in line
                   for line in CHECK.coverage(ctx)), spec
        assert not any("not for" in line for line in CHECK.coverage(ctx)), spec


# --- the seen floor ---------------------------------------------------------------------------
# Rule B is judged only as it will be seen: a line end whose stub is shorter than one arcminute at
# the piece's nearest viewing distance is dropped and counted, and so is one whose stub lies on a
# side the view never shows. A piece with no view is judged as before.


def line_ends_of(ctx, prefix):
    """The line-end items of the parts whose item starts with `prefix`."""
    return {f.item for f in CHECK.run(ctx) if " line end " in f.item and f.item.startswith(prefix)}


def person(ctx):
    return next(v for v in pieces.measured_pieces(ctx, CHECK.kinds) if v.name in ("Person", "Thing"))


def rest_stubs(ctx, part):
    """The length of the stub under each rest line end of `part`, by item, read from the mesh."""
    view = person(ctx)
    mesh_part = next(p for p in view.parts if p.name == part)
    mesh = near_crease._mesh(mesh_part.tris, view.size, view.piece, CREASE)
    stubs = near_crease._stubs(mesh.edges, mesh.drawn, mesh.ends)
    return {f"{pieces.item_of(view, part)} line end {v}": length for v, (length, _) in stubs.items()}


def bend_stubs(ctx, part):
    """The longest stub under each line end of a skinned part over its clip samples, by item."""
    view = person(ctx)
    bend = near_crease._bend_mesh(ctx.rests[view.key], part, view.size, view.piece, CREASE)
    longest = {}
    for clip in near_crease._clips_of(ctx, view.key, part):
        for sample in clip.shells[part]:
            moved = bend.mesh.edges.moved(sample.astype(float)[bend.rep])
            drawn = moved.turn_deg > CREASE
            stops = line_ends(moved, drawn, moved.open_boundary)
            for v, (length, _) in near_crease._stubs(moved, drawn, stops).items():
                longest[v] = max(longest.get(v, 0.0), length)
    return {f"{pieces.item_of(view, part)} line end {v}": x for v, x in longest.items()}


def drops(ctx):
    """The count in the check's `under 1 arcmin` coverage lines, by piece key."""
    out = {}
    for line in CHECK.coverage(ctx):
        found = re.fullmatch(r"near-crease: (\d+) under 1 arcmin at (\S+) m: (\S+)", str(line))
        if found:
            out[found.group(3)] = int(found.group(1))
    return out


@functools.cache
def planted_lathe():
    """The lathe scene with the dash plate added: three stubs of one length."""
    return plants_of(CHECK, scene(lathe_parts()), "line", "kit:Thing dash_plant ")[0].ctx


@functools.cache
def d3_arm():
    """The chibi's old left arm: 16 line ends round the thumb bump, with stubs from 1 mm to 65 mm
    and faces on several sides. A real mesh, so the floor and the hidden sides are read on more
    than the plant."""
    z = _chibi("arm_dash.npz")
    v, f = _mesh(z, "arm_L")
    return scene({"arm_L": v[f]}, {"arm_L": {"shape": "ring"}}, bits=None)


@pytest.mark.parametrize(("label", "ctx", "part", "key", "all_gone"), [
    ("lathe", planted_lathe, "dash_plant", "kit:Thing", True),
    ("arm", d3_arm, "arm_L", "kit:Thing", False),
])
def test_a_stub_just_over_the_floor_fires_and_one_just_under_is_dropped_and_counted(
        label, ctx, part, key, all_gone):
    base = ctx()
    stubs = rest_stubs(base, part)
    shortest = min(stubs.values())
    here = pieces.item_of(person(base), part)
    assert len(stubs) > 1 and line_ends_of(base, here) == set(stubs), label
    over = viewed(base, 0.999 * shortest)
    assert line_ends_of(over, here) == set(stubs), label
    assert drops(over)[key] == 0, label
    under = viewed(base, 1.001 * shortest)
    gone = {item for item, length in stubs.items() if length < 1.001 * shortest}
    assert (gone == set(stubs)) is all_gone and gone, label
    assert line_ends_of(under, here) == set(stubs) - gone, label
    assert drops(under)[key] == len(gone), label
    gaps = [x for x in CHECK.coverage(base) if isinstance(x, Unmeasured)]
    assert [x for x in CHECK.coverage(under) if isinstance(x, Unmeasured)] == gaps, label


def test_a_skinned_part_is_judged_through_its_clip_samples_by_its_longest_stub():
    base = humanoid()
    stubs = bend_stubs(base, "leg_R")
    assert len(stubs) > 1
    shortest = min(stubs.values())
    assert line_ends_of(viewed(base, 0.999 * shortest), "humanoid:Person leg_R ") == set(stubs)
    under = viewed(base, 1.001 * shortest)
    gone = {item for item, length in stubs.items() if length < 1.001 * shortest}
    assert gone and line_ends_of(under, "humanoid:Person leg_R ") == set(stubs) - gone
    assert drops(under)["humanoid:Person"] >= len(gone)


def test_a_piece_with_no_view_gets_no_floor_no_line_and_no_gap():
    for ctx in (planted_lathe(), humanoid()):
        assert all(p.view is None for pack in ctx.packs.values() for p in pack.pieces.values())
        tiny = viewed(ctx, 1e-9)  # a floor under every stub: the view changes nothing
        assert {f.item for f in CHECK.run(ctx)} == {f.item for f in CHECK.run(tiny)}
        assert not any("arcmin" in x for x in CHECK.coverage(ctx))
        assert [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)] == [
            x for x in CHECK.coverage(tiny) if isinstance(x, Unmeasured)]
        assert drops(tiny) and set(drops(tiny).values()) == {0}


def hidden_ends(ctx, part, sides):
    """(the ends whose stub lies wholly on `sides`, the ends with a face on them that are not),
    read from the faces of the mesh."""
    view = person(ctx)
    mesh_part = next(p for p in view.parts if p.name == part)
    mesh = near_crease._mesh(mesh_part.tris, view.size, view.piece, CREASE)
    on_side = faces_on_sides(mesh_part.tris, sides)
    whole, mixed = set(), set()
    for v, (_, run) in near_crease._stubs(mesh.edges, mesh.drawn, mesh.ends).items():
        faces = on_side[mesh.edges.faces[run].ravel()]
        (whole if faces.all() else mixed if faces.any() else set()).add(
            f"{pieces.item_of(view, part)} line end {v}")
    return whole, mixed


def test_a_stub_on_a_hidden_side_is_skipped_and_one_beside_a_seen_face_still_fires():
    ctx = d3_arm()
    here = "kit:Thing arm_L"
    whole, mixed = hidden_ends(ctx, "arm_L", ["front"])
    assert whole and mixed  # a stub wholly hidden, and one with a hidden face and a seen one
    every = line_ends_of(viewed(ctx, 1e-9), here)
    seen = line_ends_of(viewed(ctx, 1e-9, ("front",)), here)
    assert every - seen == whole
    assert mixed <= seen  # an edge between a hidden face and a seen face still counts
    assert drops(viewed(ctx, 1e-9, ("front",)))["kit:Thing"] == 0
    lines = CHECK.coverage(viewed(ctx, 1e-9, ("front",)))
    assert f"near-crease: {len(whole)} line ends on hidden sides (front): kit:Thing" in lines


def test_the_plate_is_skipped_from_the_hidden_side_it_faces_and_fires_from_the_other():
    ctx = planted_lathe()  # the plate faces front (+z): its three stubs lie on that side
    whole, _ = hidden_ends(ctx, "dash_plant", ["front"])
    assert len(whole) == 3
    assert line_ends_of(viewed(ctx, 1e-9, ("front",)), "kit:Thing dash_plant") == set()
    assert line_ends_of(viewed(ctx, 1e-9, ("back",)), "kit:Thing dash_plant") == whole


def test_rule_b_plants_stay_caught_under_a_floor_and_a_hidden_side_on_two_pieces():
    lathe = scene(lathe_parts())
    for label, scene_, key, floor, hidden in (
            ("lathe", lathe, "kit:Thing", 0.4, ()),
            ("lathe hidden", lathe, "kit:Thing", 1e-9, ("front",)),
            ("person", humanoid(), "humanoid:Person", 0.06, ()),
            ("person hidden", humanoid(), "humanoid:Person", 1e-9, ("front", "right"))):
        seen_ctx = viewed(scene_, floor, hidden)
        for stem in ("dash_plant", "ring_plant"):
            (plant,) = plants_of(CHECK, seen_ctx, "line", f"{key} {stem} ")
            assert caught(CHECK, plant), (label, stem)
            assert plant.item not in {f.item for f in CHECK.run(seen_ctx, only={plant.item})}
            assert rest_stubs(plant.ctx, stem)[plant.item] >= floor, (label, stem)
            if hidden:
                assert plant.item not in hidden_ends(plant.ctx, stem, list(hidden))[0]


# --- a piece that draws no outline ------------------------------------------------------------

WHY = "the page draws it with no lines"


def unlined(ctx, only=None):
    """The scene with these pieces (all, by default) declared as drawing no outline."""
    for pack_name, pack in ctx.packs.items():
        for name, piece in pack.pieces.items():
            if only is None or name in only:
                ctx = pieces.with_piece(ctx, pack_name, name, dataclasses.replace(
                    piece, outlined=False, outlined_why=WHY))
    return ctx


def na_line(key):
    return f"near-crease: not applicable (the piece's shipping look draws no outline: {WHY}): {key}"


def keys_of(ctx):
    return [pieces.piece_key(p, n) for p, pack in ctx.packs.items() for n in pack.pieces]


PLANTED = [  # (kind, label, a scene that has this plant); each kind is on two pieces
    ("near", "bar", bar03),
    ("near", "lathe", lambda: scene(lathe_parts(), bits=14)),
    ("line", "lathe", lambda: scene(lathe_parts())),
    ("line", "person", humanoid),
    ("bends", "humanoid", humanoid),
    ("bends", "quadruped", lambda: crease_25(quadruped_scene())),
]


@pytest.mark.parametrize(("kind", "label", "make"), PLANTED, ids=lambda x: x if isinstance(x, str) else "")
def test_a_piece_that_draws_no_outline_loses_what_an_outlined_piece_keeps(kind, label, make):
    """The same planted scene, twice. Left at the default, the piece is outlined and the plant is
    found. Declared `outlined: false`, the piece gets no finding, no plant and no part in
    `measured`, and a not-applicable line says why."""
    ctx = make()
    (plant, *_) = plants_of(CHECK, ctx, kind)
    run = {"only": plant.scope, "samples": plant.samples}
    assert caught(CHECK, plant), (kind, label)
    assert CHECK.measured(ctx)
    flat = unlined(plant.ctx)
    assert plant.item not in {f.item for f in CHECK.run(flat, **run)}, (kind, label)
    assert CHECK.run(unlined(ctx)) == [] and CHECK.plants(unlined(ctx)) == []
    assert CHECK.measured(unlined(ctx)) == set()
    lines = CHECK.coverage(unlined(ctx))
    assert all(na_line(key) in lines for key in keys_of(ctx)), (kind, label)
    assert not [x for x in lines if isinstance(x, Unmeasured)]


def test_an_unlined_lathe_passes_the_gate_and_an_outlined_one_fails():
    ctx = scene(lathe_parts(), bits=14)
    (plant,) = plants_of(CHECK, ctx, "near")
    assert run_gate([NearCrease()], plant.ctx, None).status == "failed"
    assert run_gate([NearCrease()], unlined(plant.ctx), None).status == "passed"


def test_a_piece_with_no_field_is_outlined():
    ctx = scene(lathe_parts(), bits=14)
    assert all(p.outlined and p.outlined_why == ""
               for pack in ctx.packs.values() for p in pack.pieces.values())
    assert not any("shipping look" in x for x in CHECK.coverage(ctx))


def twinned(ctx):
    """The scene with a second assembly piece, `Twin`, made of the same parts and record."""
    pack = ctx.packs["kit"]
    thing = pack.pieces["Thing"]
    twin = pieces.with_piece(ctx, "kit", "Twin", dataclasses.replace(thing, name="Twin"))
    geom = ctx.pieces["kit:Thing"]
    return dataclasses.replace(twin, pieces={
        **twin.pieces, "kit:Twin": dataclasses.replace(geom, name="Twin")})


def test_one_unlined_piece_leaves_its_neighbour_judged():
    both = twinned(planted_lathe())
    base = {f.item.split()[0] for f in CHECK.run(both)}
    assert base == {"kit:Thing", "kit:Twin"}
    gone = unlined(both, only={"Thing"})
    assert {f.item.split()[0] for f in CHECK.run(gone)} == {"kit:Twin"}
    assert {f.item.split()[0] for f in CHECK.run(unlined(both, only={"Twin"}))} == {"kit:Thing"}
    lines = CHECK.coverage(gone)
    assert na_line("kit:Thing") in lines and na_line("kit:Twin") not in lines
    assert {i.split()[0] for i in CHECK.measured(gone)} == {"kit:Twin"}
    assert {p.item.split()[0] for p in CHECK.plants(gone)} == {"kit:Twin"}


def test_the_style_that_draws_no_outline_still_says_so_once_and_wins():
    ctx = unlined(scene(lathe_parts(), outline="none"))
    assert CHECK.coverage(ctx) == ["near-crease: not applicable (the style draws no outline)"]
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == []


def test_the_declaration_quiets_only_near_crease():
    """Every other check reads the same scene the same way, declared or not."""
    ctx = scene(lathe_parts(), bits=14)
    flat = unlined(ctx)
    for name, check in gate.CHECKS.items():
        if name == "near-crease":
            continue
        assert check().run(ctx) == check().run(flat), name
        assert check().coverage(ctx) == check().coverage(flat), name


# --- the plants are built blind to the verdict ------------------------------------------------


def test_the_plants_never_call_the_verdict(monkeypatch):
    """Every unit is built from the piece's manifest and meshes. The functions that decide a
    near edge, a bending edge and a line end are not called, on a rigid piece and on a skinned
    one."""
    scenes = (scene(lathe_parts(), bits=14), humanoid())
    refuses_the_core(monkeypatch, near_crease, "_near", "_rest_findings", "_bend_findings",
                     "_seen_ends", "_stubs", "_dropped", "_unseen_edges", "line_ends", "_mesh",
                     "_bend_mesh")
    kinds = {kind_of(u.item) for ctx in scenes for u in CHECK.plants(ctx)}
    assert kinds == {"near", "line", "bends"}


@pytest.mark.parametrize(("kind", "make", "key"), [
    ("near", bar03, "chibi:chibi "),
    ("near", lambda: scene(lathe_parts(), bits=14), "kit:Thing "),
    ("line", lambda: scene(lathe_parts()), "kit:Thing "),
    ("line", humanoid, "humanoid:Person "),
    ("bends", humanoid, "humanoid:Person "),
    ("bends", lambda: crease_25(quadruped_scene()), "quadruped:Animal "),
], ids=["near-bar", "near-lathe", "line-lathe", "line-person", "bends-humanoid", "bends-quadruped"])
def test_every_unit_is_tried_and_caught_on_two_pieces(kind, make, key):
    """The gate would try each unit and accept a candidate. For each of two named pieces per kind,
    a unit of THAT piece was tried and caught (a unit of another piece in the scene does not
    count); a line end is planted twice, on the plate and on the bent tube."""
    tried = [i for i in tried_and_caught(CHECK, make()) if kind_of(i) == kind and i.startswith(key)]
    assert tried, (kind, key)
    if kind == "line":
        assert {i.split()[1] for i in tried} == {"dash_plant", "ring_plant"}
    if kind == "near":
        assert all("near_plant" not in i for i in tried)  # on the piece's own part


def test_a_unit_has_its_candidates_on_one_item():
    ctx = humanoid()
    (unit,) = [u for u in CHECK.plants(ctx) if kind_of(u.item) == "bends"
               and u.item.startswith("humanoid:Person ")]
    assert len(unit.candidates) == 3 and {p.item for p in unit.candidates} == {unit.item}
    assert min(next(iter(p.samples)) for p in unit.candidates) == 0


# --- the bending plant needs no precision ---------------------------------------------


def without_export(ctx):
    """The scene with no piece recording `measured.export`."""
    for pack_name, pack in ctx.packs.items():
        for name, piece in pack.pieces.items():
            measured = {k: v for k, v in (piece.measured or {}).items() if k != "export"}
            ctx = pieces.with_piece(ctx, pack_name, name,
                                    dataclasses.replace(piece, measured=measured))
    return ctx


def test_the_bending_plant_stands_without_a_shipped_precision():
    """Rule A at rest needs `measured.export`; a bend across the crease angle does not, so the
    bending unit is planted and caught on a piece that records none."""
    for label, ctx in (("humanoid", without_export(humanoid())),
                       ("quadruped", without_export(crease_25(quadruped_scene())))):
        assert not [u for u in CHECK.plants(ctx) if kind_of(u.item) == "near"], label
        (plant, *_) = plants_of(CHECK, ctx, "bends")
        assert caught(CHECK, plant), label
        assert plant.item not in {f.item for f in CHECK.run(ctx, only={plant.item})}, label
        assert "across the crease angle" in next(
            f.detail for f in CHECK.run(plant.ctx, only=plant.scope, samples=plant.samples)
            if f.item == plant.item), label


def test_the_bending_plant_folds_to_the_other_side_of_the_crease_angle_from_rest():
    ctx = humanoid()
    (plant, *_) = plants_of(CHECK, ctx, "bends")
    rest_turn, sample_turn = _turns_of(ctx, plant)
    crease = CREASE
    assert (rest_turn > crease) != (sample_turn > crease)


def _turns_of(ctx, plant):
    """The turn of the plant's edge at rest and at its sample, read from the shells."""
    (view,) = [v for v in pieces.measured_pieces(ctx, CHECK.kinds)
               if plant.item.startswith(f"{v.key} ")]
    part = plant.item.split()[1]
    e = int(re.search(r"edge (\d+) bends across", plant.item).group(1))
    edges, rep = near_crease._skin(ctx.rests[view.key], part, view.size)
    (clip,) = [c for c in near_crease._clips_of(plant.ctx, view.key, part)
               if c.shells[part] is not ctx.samples[c.clip].shells[part]]
    (s,) = plant.samples
    moved = edges.moved(clip.shells[part][s].astype(float)[rep])
    return float(edges.turn_deg[e]), float(moved.turn_deg[e])


# --- edge numbers agree between the rules ---------------------------------------------


@pytest.mark.parametrize("make", [humanoid, lambda: crease_25(quadruped_scene())],
                         ids=["humanoid", "quadruped"])
def test_edge_numbers_agree_between_the_rest_rule_and_the_bending_rule(make):
    """`edge 5 near` and `edge 5 bends across` name the same edge of a part: the rest rule welds
    the part's triangles, the bending rule the rest shell's, and both number the edges by their
    welded ends. If the two meshes were ordered differently the numbers would drift apart."""
    ctx = make()
    checked = 0
    for view in pieces.measured_pieces(ctx, CHECK.kinds):
        rest = ctx.rests.get(view.key)
        for part in view.parts:
            if rest is None or not near_crease._bends(rest, part.name):
                continue
            at_rest = near_crease._mesh(part.tris, view.size, view.piece, CREASE).edges
            bending = near_crease._bend_mesh(rest, part.name, view.size, view.piece,
                                             CREASE).mesh.edges
            assert len(at_rest) == len(bending), part.name
            assert np.allclose(at_rest.points[at_rest.v], bending.points[bending.v]), part.name
            checked += 1
    assert checked


# --- a piece declared as drawing no outline, with no geometry -------------------------


def test_an_unlined_piece_with_no_geometry_gets_the_declaration_not_a_gap():
    ctx = unlined(scene(lathe_parts(), bits=14))
    ctx = dataclasses.replace(ctx, pieces={})  # the scene holds no mesh for the piece
    lines = CHECK.coverage(ctx)
    assert na_line("kit:Thing") in lines
    assert not [x for x in lines if isinstance(x, Unmeasured)]
    assert run_gate([NearCrease()], ctx, None).status == "passed"


def test_a_lined_piece_with_no_geometry_is_still_a_gap():
    ctx = dataclasses.replace(scene(lathe_parts(), bits=14), pieces={})
    lines = CHECK.coverage(ctx)
    assert [x for x in lines if isinstance(x, Unmeasured)] == [
        "near-crease: skipped (no geometry): kit:Thing"]
    assert not any("shipping look" in x for x in lines)


def test_a_precision_too_coarse_to_call_any_edge_near_says_so_and_is_not_blind():
    """At 2 bits every edge of the ball has its band cut (a sliver's), so rule A judges nothing:
    the unit says why, where the gate prints it, and rule B still stands."""
    ctx = scene(lathe_parts(), bits=2)
    (unit,) = [u for u in CHECK.plants(ctx) if u.not_applicable]
    assert unit.item == "kit:Thing" and unit.candidates == ()
    assert "every edge of every part has its band cut" in unit.not_applicable
    verdict = run_gate([NearCrease()], ctx, None)
    assert any("near-crease: not applicable (no edge is judged: every edge of every part has "
               "its band cut" in x for x in verdict.coverage)


def test_a_piece_with_no_interior_edge_says_that_not_a_cut_band():
    """A single triangle has no edge between two faces: nothing is judged, and the reason is not
    the precision."""
    ctx = scene({"card": np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])},
                bits=14)
    (unit,) = [u for u in CHECK.plants(ctx) if u.not_applicable]
    assert unit.item == "kit:Thing" and unit.not_applicable.endswith("no part has an interior edge")
    assert "band" not in unit.not_applicable


# --- the ring step and the fallbacks --------------------------------------------------


def tube_edges(crease, share):
    sides = max(3, math.ceil(360.0 / (0.5 * crease)))
    tris = near_crease._bent_tube(1.0, sides, share * crease)
    return tris, interior_edges(tris, 1e-9), sides


@pytest.mark.parametrize("crease", [10.0, 25.0, 35.0, 60.0])
def test_the_bent_tube_draws_only_part_of_its_junction_ring_and_its_lines_stop(crease):
    """The tube is closed and wound outward. At the first bend tried, a little over the crease
    angle, some junction ring edges draw and others do not, and lines stop (a partial ring step):
    the ring's lines have line ends, none of them on an open boundary."""
    tris, edges, sides = tube_edges(crease, near_crease.BEND_OF_CREASE[0])
    assert signed_volume(tris) > 0 and not edges.open_boundary.any()
    drawn = edges.turn_deg > crease
    ring = np.isclose(edges.points[edges.v][:, :, 1], 0.0).all(axis=1)  # the junction ring, y = 0
    assert 0 < (drawn & ring).sum() < sides
    assert len(line_ends(edges, drawn, edges.open_boundary)) >= 2


def test_the_ring_step_is_a_unit_of_its_own_beside_the_plate():
    ctx = scene(lathe_parts())
    items = [u.item for u in CHECK.plants(ctx) if kind_of(u.item) == "line"]
    assert [i.split()[1] for i in items] == ["dash_plant", "ring_plant"]


def test_a_judged_part_that_cannot_be_folded_stops_the_gate(monkeypatch):
    """No substitute fixture: when the parts rule A judges cannot be folded by name, the unit
    names the nearest judged edge and has no candidate, so the gate raises BlindCheck."""
    monkeypatch.setattr(near_crease, "_find_edge", lambda *args: None)
    ctx = scene(lathe_parts(), bits=14)
    (unit,) = [u for u in CHECK.plants(ctx) if kind_of(u.item) == "near"]
    assert unit.candidates == () and not unit.not_applicable
    assert re.fullmatch(r"kit:Thing ball edge \d+ near", unit.item)
    with pytest.raises(BlindCheck, match="near-crease: kit:Thing ball edge"):
        run_gate([NearCrease()], ctx, None)


def test_the_bending_plant_passes_to_the_next_skinned_part_and_names_an_edge_when_none_folds(
        monkeypatch):
    ctx = humanoid()
    first = next(u for u in CHECK.plants(ctx) if kind_of(u.item) == "bends"
                 and u.item.startswith("humanoid:Person ")).item.split()[1]
    skin, at = near_crease._skin, {}

    def watching(rest, part, size):
        at["part"] = part
        return skin(rest, part, size)

    fold = near_crease._folded_point
    monkeypatch.setattr(near_crease, "_skin", watching)
    monkeypatch.setattr(near_crease, "_folded_point",
                        lambda *a, **k: None if at.get("part") == first else fold(*a, **k))
    (unit,) = [u for u in CHECK.plants(ctx) if kind_of(u.item) == "bends"
               and u.item.startswith("humanoid:Person ")]
    assert unit.candidates and unit.item.split()[1] != first
    monkeypatch.setattr(near_crease, "_folded_point", lambda *a, **k: None)
    (unit,) = [u for u in CHECK.plants(ctx) if kind_of(u.item) == "bends"
               and u.item.startswith("humanoid:Person ")]
    assert unit.candidates == () and kind_of(unit.item) == "bends"
    assert unit.item.startswith(f"humanoid:Person {first} edge ")
    with pytest.raises(BlindCheck, match="near-crease"):
        run_gate([NearCrease()], ctx, None)
