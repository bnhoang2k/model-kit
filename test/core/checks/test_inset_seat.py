"""Check inset-seat: a printed or inset part meets its host along one line.

The printed rule is planted on the chibi's eye and on the fixture panel's decal. The inset rules
(one closed loop; a rim that does not weave) are run on a lens in a block, and on a chibi eye
with a rim that weaves (the eye-rim fixture, `bodies/test_regressions.py`).

The piece's contact length is the distance one arcminute covers at its `view.nearest_m`, in
project units. The scenes here are seen from 1 m (`NEAREST_M`), which makes it about 0.29 mm
(`EYE`). It is the printed rule's touch and the weave rule's reach alike.
"""

import dataclasses
import functools
import math

import numpy as np
import pytest

from model_kit.core import gate, geom, sight
from model_kit.core.checks import inset_seat, pieces
from model_kit.core.checks.inset_seat import InsetSeat
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.geom import SurfaceIndex
from model_kit.core.manifest import Pack, Part, View
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    COMPARE_TOL,
    EYE_RESOLUTION_ARCMIN,
    WELD_OF_PIECE_SIZE,
)
from model_kit.core.units import Style

from ..bodies.bar03 import bar03_scene
from ..bodies.build import SEEN_FLOOR
from ..bodies.humanoid import humanoid_scene
from ..bodies.test_regressions import _chibi, _mesh
from ..shapes import block, open_plate
from .piece_scenes import make_ctx, plants_caught, refuses_the_core
from .sight_scenes import viewed

CHECK = InsetSeat()
CREASE = 25.0
NEAREST_M = 1.0  # the distance these scenes are seen from
EYE = NEAREST_M * math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60))  # 0.29 mm, in metres
HOST_TOP = 0.08  # the block's top face
HOST_SPEC = {"base": {"on": None}}
LENS_SPEC = {"lens": {"on": "base", "role": "eyes", "shape": "blob"}}
MARK_SPEC = {"mark": {"on": "base", "role": "decal", "shape": "decal"}}


def style(angle=CREASE) -> Style:
    edges = {"outline": "none"}  # a printed part draws no outline: the rim must not need one
    if angle is not None:
        edges |= {"crease_deg": angle, "crease_why": "a test value, set by the test"}
    return Style(1.0, {}, {"edges": edges})


def with_view(ctx, nearest_m=NEAREST_M, piece="kit:Thing"):
    """The scene with this piece seen from `nearest_m` (None takes the view away)."""
    pack, _, name = piece.partition(":")
    old = ctx.packs[pack].pieces[name]
    view = None if nearest_m is None else View(nearest_m=nearest_m, why="a test distance")
    return pieces.with_piece(ctx, pack, name, dataclasses.replace(old, view=view))


# --- shapes --------------------------------------------------------------------------------


def patch(half=0.05, n=4, y=HOST_TOP + 0.0001) -> np.ndarray:
    """An open square grid of 2 n^2 triangles, flat at height y, facing up: a decal."""
    xs = np.linspace(-half, half, n + 1)
    tris = []
    for i in range(n):
        for j in range(n):
            p = lambda a, b: (xs[a], y, xs[b])
            tris += [(p(i, j), p(i + 1, j), p(i, j + 1)), (p(i + 1, j), p(i + 1, j + 1),
                                                         p(i, j + 1))]
    return np.array(tris)


def lens(r=0.05, h=0.03, centre=(0.0, HOST_TOP, 0.0), n=16, tilt_deg=0.0) -> np.ndarray:
    """A closed lens: a ring of n corners of radius r in a plane, an apex h above it and h
    below. Its rim (where the two cones meet) turns far more than the crease angle. `tilt_deg`
    turns it about x."""
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(t), np.zeros(n), r * np.sin(t)], axis=1)
    up, down = np.array([0.0, h, 0.0]), np.array([0.0, -h, 0.0])
    tris = []
    for i in range(n):
        a, b = ring[i], ring[(i + 1) % n]
        tris += [(up, b, a), (down, a, b)]
    out = np.array(tris)
    if geom.signed_volume(out) < 0:
        out = out[:, ::-1]
    c, s = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    turn = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return out @ turn.T + np.array(centre)


def post_through(top=0.1, bottom=-0.05) -> np.ndarray:
    from ..shapes import box
    return box((-0.02, bottom, -0.02), (0.02, top, 0.02))


def stud(top_above_host, half=0.02, depth=0.05) -> np.ndarray:
    """A box stood in the host with its top `top_above_host` over the host's top (negative: sunk)."""
    from ..shapes import box
    top = HOST_TOP + top_above_host
    return box((-half, top - depth, -half), (half, top, half))


def disc(top_above_host, r=0.03, depth=0.05, n=24) -> np.ndarray:
    """A closed cylinder stood in the host, its top `top_above_host` over the host's top."""
    top = HOST_TOP + top_above_host
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.stack([r * np.cos(t), np.zeros(n), r * np.sin(t)], axis=1)
    hi, lo = ring + [0, top, 0], ring + [0, top - depth, 0]
    up, down = np.array([0.0, top, 0.0]), np.array([0.0, top - depth, 0.0])
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris += [(up, hi[j], hi[i]), (down, lo[i], lo[j]),
                 (hi[i], hi[j], lo[j]), (hi[i], lo[j], lo[i])]
    out = np.array(tris)
    return out[:, ::-1] if geom.signed_volume(out) < 0 else out


def aligned(tris, normal, centre) -> np.ndarray:
    """`tris` (built about the origin, its axis +y) turned so that its axis is `normal`, then
    moved to `centre`."""
    up = np.array([0.0, 1.0, 0.0])
    axis = np.cross(up, normal)
    sin, cos = float(np.linalg.norm(axis)), float(up @ normal)
    if sin < 1e-12:
        turn = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        turn = np.eye(3) + k + k @ k * ((1 - cos) / sin**2)
    # rounded, so a corner shared by several triangles stays one point (the mesh stays closed)
    return np.round(tris @ turn.T + np.asarray(centre), 12)


def scene(extra, specs, nearest_m=NEAREST_M):
    """A block `base` and the parts in `extra`, with the test style, seen from `nearest_m`."""
    ctx = make_ctx({"base": block(0.2, HOST_TOP), **extra}, {**HOST_SPEC, **specs})
    return with_view(dataclasses.replace(ctx, style=style()), nearest_m)


def findings(ctx, only=None):
    return CHECK.run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


# --- registration -------------------------------------------------------------------------


def test_the_check_is_registered_for_assembly_pieces_and_states_its_principle():
    assert gate.CHECKS["inset-seat"] is InsetSeat
    assert InsetSeat.kinds == frozenset({"assembly"})
    assert CHECK.principle.endswith(".") and "http" in CHECK.source


# --- the printed rule ---------------------------------------------------------------------


def test_printed_decal_passes():
    ctx = scene({"mark": patch()}, MARK_SPEC)
    assert findings(ctx) == []
    assert plants_caught(CHECK, ctx)
    assert run_gate([InsetSeat()], ctx, None).status == "passed"


def test_a_decal_a_touch_off_its_host_is_printed_and_one_past_it_lifts_off():
    near = scene({"mark": patch(y=HOST_TOP + 0.5 * EYE)}, MARK_SPEC)
    assert findings(near) == []
    far = scene({"mark": patch(y=HOST_TOP + CALIBRATION_FAIL * EYE)}, MARK_SPEC)
    (found,) = findings(far)
    assert (found.check, found.item) == ("inset-seat", "kit:Thing mark lifts off base")
    assert f"{CALIBRATION_FAIL * EYE:.4f}" in found.detail or "limit" in found.detail


def test_the_touch_is_the_pieces_contact_length_the_floaters_touch_distance_too():
    ctx = scene({"mark": patch()}, MARK_SPEC)
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    assert inset_seat.touch_of(view) == view.contact == pytest.approx(EYE)


def test_half_the_rim_of_a_printed_decal_sunk_into_its_host_lifts_off():
    patch_ = patch()
    sunk = patch_.copy()
    sunk[np.abs(sunk[:, :, 0] - 0.05) < 1e-9] -= [0, 0.05, 0]  # the +x edge of the grid
    (found,) = findings(scene({"mark": sunk}, MARK_SPEC))
    assert found.item == "kit:Thing mark lifts off base"


def test_a_closed_part_that_stays_within_the_touch_of_its_host_is_printed():
    slab = np.array([t for t in lens(r=0.05, h=0.00005, centre=(0, HOST_TOP + 0.00015, 0))])
    ctx = scene({"lens": slab}, LENS_SPEC)
    assert findings(ctx) == []


# --- the inset rules ----------------------------------------------------------------------


def test_an_inset_lens_with_its_rim_inside_meets_the_host_in_one_loop_and_passes():
    ctx = scene({"lens": lens(centre=(0.0, HOST_TOP - 0.005, 0.0))}, LENS_SPEC)
    assert findings(ctx) == []
    assert len(geom.cut_loops(ctx.pieces["kit:Thing"].parts["lens"], block(0.2, HOST_TOP))) == 1


def test_a_post_through_its_host_meets_it_in_two_loops():
    ctx = scene({"lens": post_through()}, LENS_SPEC)
    (found,) = findings(ctx)
    assert found.item == "kit:Thing lens meets base in 2 loops"
    assert found.check == "inset-seat"


def test_a_part_wholly_inside_its_host_meets_it_in_no_loops():
    ctx = scene({"lens": lens(r=0.03, h=0.02, centre=(0.0, 0.04, 0.0))}, LENS_SPEC)
    assert items(ctx) == ["kit:Thing lens meets base in 0 loops"]


def test_a_cut_that_does_not_close_is_a_finding_whatever_its_count(monkeypatch):
    ctx = scene({"lens": lens(centre=(0.0, HOST_TOP - 0.005, 0.0))}, LENS_SPEC)
    open_line = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1, 0]])
    monkeypatch.setattr(inset_seat.geom, "cut_loops", lambda *a, **k: [open_line])
    (found,) = findings(ctx)
    assert found.item == "kit:Thing lens meets base in 1 loop"  # one loop, said in the singular
    assert "in 1 line," in found.detail and "open" in found.detail


def test_a_lens_whose_rim_weaves_across_the_host_surface_is_a_finding():
    ctx = scene({"lens": lens(tilt_deg=15.0)}, LENS_SPEC)
    (found,) = findings(ctx)
    assert (found.check, found.item) == ("inset-seat", "kit:Thing lens weaves across base")
    # one loop only: the loop rule alone does not see the weave
    loops = geom.cut_loops(ctx.pieces["kit:Thing"].parts["lens"], block(0.2, HOST_TOP))
    assert len(loops) == 1


def test_a_rim_that_sits_wholly_outside_or_on_the_host_does_not_weave():
    on = scene({"lens": lens(centre=(0.0, HOST_TOP + EYE / 2, 0.0))}, LENS_SPEC)
    assert findings(on) == []  # rim on the surface within the reach
    # a thick lens whose back is deep (an inset part) and whose rim stands 1 cm outside
    out = scene({"lens": lens(h=0.04, centre=(0.0, HOST_TOP + 0.01, 0.0))}, LENS_SPEC)
    reading = inset_seat._read(out, next(iter(pieces.measured_pieces(out, {"assembly"}))),
                               next(p for p in pieces.measured_pieces(out, {"assembly"})
                                    .__next__().parts if p.name == "lens"))
    assert reading.kind == "inset"  # so the weave rule really ran
    assert findings(out) == []


@pytest.mark.parametrize("proud", [0.0005, 0.002, 0.01])
@pytest.mark.parametrize("make", [stud, disc])
def test_a_seated_part_with_its_top_proud_does_not_weave(make, proud):
    """A box or a disc set 5 cm into its host with the top face standing proud meets it in one
    clean loop; the rim seen from outside is the top face's outline, wholly outside."""
    ctx = scene({"lens": make(proud)}, LENS_SPEC)
    assert findings(ctx) == []


@pytest.mark.parametrize("sunk", [0.0, -0.005])
@pytest.mark.parametrize("make", [stud, disc])
def test_a_seated_part_with_its_top_flush_or_sunk_does_not_weave(make, sunk):
    ctx = scene({"lens": make(sunk)}, LENS_SPEC)
    assert [f.item for f in findings(ctx) if "weaves" in f.item] == []
    if sunk == 0.0:
        assert findings(ctx) == []  # flush: one loop, rim on the surface within the reach
    else:
        assert [f.item for f in findings(ctx)] == ["kit:Thing lens meets base in 0 loops"]


def test_a_box_tipped_so_its_top_rim_crosses_the_surface_weaves():
    # (a tipped disc is not tried: seen from outside, its raised side band faces out too, so
    # the outline is the band's buried lower edge and the tilted half ring, not the top ring)
    part = aligned(stud(0.0) - [0.0, HOST_TOP, 0.0], np.array([math.sin(0.35), math.cos(0.35), 0.0]),
                   (0.0, HOST_TOP, 0.0))
    ctx = scene({"lens": part}, LENS_SPEC)
    assert [f.item for f in findings(ctx) if "weaves" in f.item] == [
        "kit:Thing lens weaves across base"]


def test_each_rim_loop_is_judged_on_its_own():
    # a lens with both rings of a thick pillbox... a disc has a top ring and a bottom ring; the
    # buried bottom ring is inside and the top ring outside, and the disc does not weave
    ctx = scene({"lens": disc(0.002)}, LENS_SPEC)
    assert findings(ctx) == []


def test_the_reach_is_what_one_arcminute_covers_at_the_nearest_distance():
    ctx = scene({"lens": lens(tilt_deg=15.0)}, LENS_SPEC, nearest_m=2.0)
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    assert inset_seat.touch_of(view) == pytest.approx(2.0 * math.tan(math.radians(1 / 60)))
    assert sight.seen_floor(view.piece, 0.5) == pytest.approx(
        2.0 * 0.5 * math.tan(math.radians(1 / 60)))


def test_a_piece_with_no_view_has_no_contact_length_so_nothing_is_judged():
    for extra, specs in (({"lens": lens(tilt_deg=15.0)}, LENS_SPEC),
                         ({"mark": patch(y=HOST_TOP + 0.05)}, MARK_SPEC)):  # a decal 5 cm off
        ctx = scene(extra, specs, nearest_m=None)
        assert findings(ctx) == []
        assert CHECK.plants(ctx) == []
        lines = [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
        assert any("kit:Thing" in x and "no view" in x for x in lines)
        assert run_gate([InsetSeat()], ctx, None).status == "partial"


def test_the_weave_needs_no_crease_angle():
    ctx = dataclasses.replace(scene({"lens": lens(tilt_deg=15.0)}, LENS_SPEC),
                              style=style(None))
    assert [f.item for f in findings(ctx)] == ["kit:Thing lens weaves across base"]
    assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)
                and "plant" not in x]


def test_a_part_with_no_rim_edge_is_unmeasured_never_silent(monkeypatch):
    ctx = scene({"lens": lens(centre=(0.0, HOST_TOP - 0.005, 0.0))}, LENS_SPEC)
    monkeypatch.setattr(inset_seat, "_rim_loops", lambda tris, normal, size: (None, []))
    assert findings(ctx) == []
    lines = [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
    assert any("no rim" in x and "kit:Thing lens" in x for x in lines)
    assert run_gate([InsetSeat()], ctx, None).status == "partial"


def test_a_part_through_two_host_faces_has_no_outward_direction_and_says_so():
    ctx = scene({"lens": post_through()}, LENS_SPEC)
    assert [f.item for f in findings(ctx)] == ["kit:Thing lens meets base in 2 loops"]
    assert any("no outward direction" in x for x in CHECK.coverage(ctx)
               if isinstance(x, Unmeasured))


def test_a_closed_part_on_an_open_host_is_unmeasured():
    ctx = make_ctx({"base": open_plate(), "lens": lens(centre=(0.0, 0.0, 0.0))},
                   {"lens": LENS_SPEC["lens"]})
    ctx = dataclasses.replace(ctx, style=style())
    assert findings(ctx) == []
    assert any("not closed" in x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured))


def test_a_printed_part_with_no_host_is_unmeasured():
    # a decal shape with no `on` cannot load; a printed role on another shape still can
    ctx = scene({"mark": patch()}, {"mark": {"on": None, "role": "decal", "shape": "box"}})
    assert findings(ctx) == []
    assert any("no host" in x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured))


def test_a_host_with_no_geometry_raises():
    ctx = scene({"mark": patch()}, MARK_SPEC)
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    ctx = pieces.with_geometry(ctx, view, {"mark": patch()})  # the host's mesh is gone
    with pytest.raises(ManifestError, match="base"):
        findings(ctx)


def test_solid_parts_and_branching_pieces_are_left_alone():
    ctx = make_ctx({"base": block(0.2, HOST_TOP), "mark": patch()}, MARK_SPEC,
                   also={"Tree": {"base": block(0.2, HOST_TOP), "mark": patch(y=1.0)}})
    ctx = dataclasses.replace(ctx, style=style())
    assert findings(ctx) == []
    lines = CHECK.coverage(ctx)
    assert "inset-seat: not applicable (branching piece): kit:Tree" in lines
    plain = make_ctx({"base": block(0.2, HOST_TOP)}, {})
    assert any("not applicable" in x and "no part printed" in x for x in CHECK.coverage(plain))


def test_a_scoped_run_measures_only_the_parts_its_items_name():
    ctx = scene({"mark": patch(y=HOST_TOP + 0.1), "other": patch(y=HOST_TOP + 0.1)},
                {**MARK_SPEC, "other": {"on": "base", "role": "decal", "shape": "decal"}})
    assert len(findings(ctx)) == 2
    assert [f.item for f in findings(ctx, only={"kit:Thing mark lifts off base"})] == [
        "kit:Thing mark lifts off base"]


def test_measured_lists_the_printed_parts_it_judges():
    ctx = scene({"mark": patch()}, MARK_SPEC)
    assert CHECK.measured(ctx) == {"kit:Thing mark"}


# --- plants -------------------------------------------------------------------------------


@functools.cache
def bar03():
    return dataclasses.replace(bar03_scene(), style=Style(
        0.747, {}, {"edges": {"outline": "runtime", "crease_deg": CREASE,
                              "crease_why": "a test value, set by the test"}}))


def test_bar03_eyes_are_lifted_by_the_contact_length_and_read_printed():
    """The builder lifts a decal off its host by the piece's contact length (`sight.decal_lift`):
    one arcminute at bar 03's 1.3 m, 0.38 mm. A gap equal to it is printed on its host, up to the
    float32 rounding of the saved coordinates."""
    ctx = bar03_scene()
    assert findings(ctx) == []
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    eye = next(p for p in view.parts if p.name == "eye_L")
    head = next(p for p in view.parts if p.name == "head")
    gap = SurfaceIndex(head.tris).closest(eye.tris.reshape(-1, 3)).dist
    assert gap.min() == pytest.approx(view.contact, abs=COMPARE_TOL)
    assert gap.max() == pytest.approx(view.contact, abs=COMPARE_TOL)


def test_bar03_eyes_lifted_past_the_contact_length_lift_off():
    """Seen from twice as far, the contact length doubles; seen from half as far, it halves and the
    same lift is now over it."""
    near = with_view(bar03_scene(), 0.5 * 1.3, piece="chibi:chibi")
    assert sorted(f.item for f in findings(near)) == ["chibi:chibi eye_L lifts off head",
                                                      "chibi:chibi eye_R lifts off head"]
    far = with_view(bar03_scene(), 2.0 * 1.3, piece="chibi:chibi")
    assert findings(far) == []
    lines = CHECK.coverage(far)
    assert not [x for x in lines if isinstance(x, Unmeasured)]
    assert CHECK.measured(far) == {"chibi:chibi eye_L", "chibi:chibi eye_R"}


def test_half_rim_plant_caught_on_two_pieces():
    """Half of one rim pushed into its host, on bar 03's eye and on the panel's decal."""
    for label, ctx, piece in (("bar 03", bar03_scene(), "chibi:chibi"),
                              ("panel", viewed(humanoid_scene(), SEEN_FLOOR), "humanoid:Panel")):
        units = [u for u in CHECK.plants(ctx) if u.item.startswith(f"{piece} ")]
        assert len(units) == 1, label
        (unit,) = units
        assert unit.item.endswith(" lifts off " + ("head" if piece == "chibi:chibi"
                                                   else "plate")), label
        assert first_caught(unit) is not None, label
        assert unit.item not in {f.item for f in CHECK.run(ctx, only={unit.item})}, label
        run_gate([CHECK], ctx, None)  # no BlindCheck
    assert unit.item == "humanoid:Panel mark lifts off plate"


def test_the_plant_sinks_half_the_rim_twice_the_touch_and_leaves_the_rest():
    ctx = scene({"mark": patch()}, MARK_SPEC)
    (unit,) = CHECK.plants(ctx)
    assert [p.label.split(" ")[:3] for p in unit.candidates] == [
        ["half", "the", "rim"], ["half", "the", "rim"]]  # pushed into, then lifted off
    plant = unit.candidates[0]
    before = ctx.pieces["kit:Thing"].parts["mark"].reshape(-1, 3)
    after = plant.ctx.pieces["kit:Thing"].parts["mark"].reshape(-1, 3)
    moved = np.abs(after - before).max(axis=1) > 0
    assert 0 < moved.sum() < len(moved)
    depth = (HOST_TOP - after[moved][:, 1])
    assert depth == pytest.approx(np.full(len(depth), CALIBRATION_FAIL * EYE))
    # no vertex of the interior moved: only rim corners
    rim = (np.abs(before[moved][:, [0, 2]]) == 0.05).any(axis=1)
    assert rim.all()


def renamed(ctx, new, old="Thing", pack="kit"):
    """The scene with its piece `old` named `new`: a second piece for the plant tests."""
    piece = dataclasses.replace(ctx.packs[pack].pieces[old], name=new)
    kept = ctx.packs[pack]
    named = Pack(name=kept.name, units_per_metre=kept.units_per_metre, spacing=kept.spacing,
                 pieces={(new if k == old else k): (piece if k == old else v)
                         for k, v in kept.pieces.items()})
    geoms = {(f"{pack}:{new}" if k == f"{pack}:{old}" else k):
             (PieceGeom(new, v.parts) if k == f"{pack}:{old}" else v)
             for k, v in ctx.pieces.items()}
    return dataclasses.replace(ctx, packs={**ctx.packs, pack: named}, pieces=geoms)


SEATED = lens(r=0.04, centre=(0.0, HOST_TOP - 0.005, 0.05))


def thing_scene():
    """A piece with a printed mark and a seated lens on one block."""
    return scene({"mark": patch(half=0.03) + [0.0, 0.0, -0.055], "lens": SEATED},
                 {**MARK_SPEC, **LENS_SPEC})


def plaque_scene():
    """A second piece, named Plaque, with a seated lens and nothing printed."""
    return renamed(scene({"lens": lens(r=0.04, centre=(0.0, HOST_TOP - 0.005, 0.0))}, LENS_SPEC),
                   "Plaque")


def chibi_scene():
    """Bar 03 with a seated lens added on the head next to its printed eyes."""
    ctx = bar03()
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    geometry = {p.name: p.tris for p in view.parts}
    head = SurfaceIndex(geometry["head"])
    eye = geometry["eye_L"].reshape(-1, 3).mean(axis=0, keepdims=True)
    at = head.closest(eye)
    shape = lens(r=0.03, h=0.03, centre=(0.0, 0.0, 0.0))
    geometry["lens"] = aligned(shape, at.normal[0], at.point[0] - at.normal[0] * 0.005)
    record = Part(name="lens", listed="eye", shape="blob", on="head", group="eyes", role="eyes",
                  bones=(), bind=None)
    ctx = pieces.with_part_record(pieces.with_geometry(ctx, view, geometry), view, record)
    return with_view(ctx, sight.seen_distance(0.002, 0.747), piece="chibi:chibi")


PIECES = {"kit:Thing": thing_scene, "kit:Plaque": plaque_scene, "chibi:chibi": chibi_scene}


def plants_on(ctx, piece):
    """The unit of each rule on the piece, by rule."""
    return {inset_seat._rule_of(u.item): u for u in CHECK.plants(ctx)
            if u.item.startswith(f"{piece} ")}


def first_caught(unit):
    """The candidate the gate would accept: the first the check catches."""
    return next((p for p in unit.candidates
                 if any(f.item == p.item for f in CHECK.run(p.ctx, only=p.scope))), None)


def test_the_seated_fixtures_are_quiet_and_their_parts_are_inset():
    for piece, make in PIECES.items():
        ctx = make()
        assert findings(ctx) == [], piece
        (view,) = [v for v in pieces.measured_pieces(ctx, {"assembly"}) if v.key == piece]
        lens_part = next(p for p in view.parts if p.name == "lens")
        assert inset_seat._read(ctx, view, lens_part).kind == "inset", piece
        assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)], piece


@pytest.mark.parametrize("rule, pieces_with", [
    ("printed", ["kit:Thing", "chibi:chibi"]),
    ("loops", ["kit:Thing", "kit:Plaque", "chibi:chibi"]),
    ("weave", ["kit:Thing", "kit:Plaque", "chibi:chibi"]),
])
def test_each_rule_has_a_plant_caught_on_two_pieces(rule, pieces_with):
    """Printed lift, loops and weave each get their own plant, and an inset part's plant is
    never taken by the printed part on the same piece."""
    for piece in pieces_with:
        ctx = PIECES[piece]()
        unit = plants_on(ctx, piece)[rule]
        part = {"printed": ("mark", "eye_L"), "loops": ("lens",), "weave": ("lens",)}[rule]
        assert unit.item.split(" ")[1] in part, (rule, piece, unit.item)
        assert first_caught(unit) is not None, piece
        assert unit.item not in {f.item for f in CHECK.run(ctx, only={unit.item})}, piece
        run_gate([CHECK], ctx, None)  # no BlindCheck
    assert {"printed", "loops", "weave"} == set(plants_on(thing_scene(), "kit:Thing"))


def test_the_loops_plant_sinks_the_part_to_no_line_and_the_weave_plant_leaves_one_loop():
    ctx = thing_scene()
    plants = plants_on(ctx, "kit:Thing")
    assert plants["loops"].item == "kit:Thing lens meets base in 0 loops"
    assert plants["weave"].item == "kit:Thing lens weaves across base"
    weave = first_caught(plants["weave"])
    assert [f.item for f in findings(weave.ctx, only={"kit:Thing lens "})] == [
        "kit:Thing lens weaves across base"]  # one loop still: only the weave rule sees it


def test_a_post_through_its_host_is_shrunk_before_it_is_sunk_and_caught():
    # the post is longer than the host is thick: sunk whole it would poke out the other side, so
    # the loops plant halves it first. The weave rule has no outward direction there, so it has
    # no unit (the check reports it `Unmeasured`).
    ctx = scene({"lens": post_through()}, LENS_SPEC)
    units = {u.item: u for u in CHECK.plants(ctx)}
    assert list(units) == ["kit:Thing lens meets base in 0 loops"]
    labels = [p.label for p in units["kit:Thing lens meets base in 0 loops"].candidates]
    assert labels and all(label.startswith(("halved", "quartered")) for label in labels)
    assert first_caught(units["kit:Thing lens meets base in 0 loops"]) is not None
    run_gate([CHECK], ctx, None)  # no BlindCheck


def test_a_rule_judged_on_a_part_with_no_fault_to_make_stops_the_run(monkeypatch):
    # the unit has no candidates and no reason: not applicable is only for a rule nothing judges
    ctx = thing_scene()
    monkeypatch.setattr(inset_seat, "_moves", lambda *a, **k: [])
    units = [u for u in CHECK.plants(ctx)]
    assert units and all(u.candidates == () and u.not_applicable == "" for u in units)
    with pytest.raises(BlindCheck, match="inset-seat: kit:Thing"):
        run_gate([CHECK], ctx, None)


def test_a_rule_no_part_is_judged_by_has_no_unit():
    # a rule with nothing to judge is not planted: a rim with no outward direction (a post
    # through its host) has no weave unit
    ctx = scene({"lens": post_through()}, LENS_SPEC)
    assert "kit:Thing lens weaves across base" not in {u.item for u in CHECK.plants(ctx)}


def test_an_inset_part_without_a_view_has_no_contact_length_so_no_rule_and_no_plant():
    ctx = scene({"lens": SEATED}, LENS_SPEC, nearest_m=None)
    assert CHECK.plants(ctx) == []  # every rule reads the contact length
    lines = [str(x) for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
    assert any("no view" in x for x in lines) and not any("no plant" in x for x in lines)


def test_inset_seat_plants_never_call_the_core(monkeypatch):
    ctx = thing_scene()
    refuses_the_core(monkeypatch, inset_seat, "_read", "_lifts_off", "_loops", "_weave")
    monkeypatch.setattr(inset_seat.InsetSeat, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in CHECK.plants(ctx)) == [
        "kit:Thing lens meets base in 0 loops", "kit:Thing lens weaves across base",
        "kit:Thing mark lifts off base"]


# --- the eye-rim fixture ---------------------------------------------------------------------------


@functools.cache
def d7_fixture():
    return _chibi("eye_rims.npz")


def d7_scene():
    """The chibi with its head and eyes taken from the eye-rim fixture: closed
    lenses of the 32 x 16 form, 66 rim edges inside the head and 38 outside."""
    z = d7_fixture()
    ctx = bar03()
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    geometry = {p.name: p.tris for p in view.parts}
    for part in ("head", "eye_L", "eye_R"):
        v, f = _mesh(z, part)
        geometry[part] = v[f]
    return with_view(pieces.with_geometry(ctx, view, geometry), piece="chibi:chibi")


def test_d7_eye_cuts_the_head_in_one_loop_so_the_loop_rule_alone_misses_it():
    z = d7_fixture()
    hv, hf = _mesh(z, "head")
    for part in ("eye_L", "eye_R"):
        v, f = _mesh(z, part)
        size = np.linalg.norm(np.concatenate([hv, v]).max(axis=0) - np.concatenate([hv, v]).min(axis=0))
        loops = geom.cut_loops(v[f], hv[hf], weld=WELD_OF_PIECE_SIZE * size)
        assert len(loops) == 1 and np.array_equal(loops[0][0], loops[0][-1]), part


def test_d7_eye_before_fix_fails():
    """The eye of the eye-rim fixture, seen from 1 m: its rim
    weaves across the head by 7 mm out and 11 mm in (real), against 0.29 mm for one arcmin."""
    got = {f.item: f for f in findings(d7_scene())}
    assert set(got) == {"chibi:chibi eye_L weaves across head",
                        "chibi:chibi eye_R weaves across head"}


def test_d7_eye_without_a_view_is_unmeasured_not_quiet():
    ctx = with_view(d7_scene(), None, piece="chibi:chibi")
    assert findings(ctx) == []
    assert any("view" in x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured))
