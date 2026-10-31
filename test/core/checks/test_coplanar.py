"""Check coplanar: two faces that face the same way, overlap and lie closer than one depth step
flicker.

The scenes are boxes seen from `NEAREST_M` out to `FARTHEST_M`, through a three.js-like camera of
the test's own (near 0.1 m, far 2000 m, 24 bits). The gap is `units.depth_step` at the farthest
view, so a test moves a face by a share of that gap it reads from the same function, never by a
number picked to pass. Items are `<pack>:<piece> <a> x <b> flicker` and
`<layout> <a> x <b> flicker`.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate, sight
from model_kit.core.checks import coplanar
from model_kit.core.checks.coplanar import Coplanar
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import Layout, View
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom, load_scene
from model_kit.core.units import Style, depth_step

from ..shapes import box
from .piece_scenes import STYLE, make_ctx, refuses_the_core, tried_and_caught
from .test_clearance import at, kit_geometry, yard

CHECK = Coplanar()
BARS = Path(__file__).resolve().parents[3] / "bars"
NEAREST_M = 1.0  # the nearest a test piece is seen from: its contact length is 0.29 mm
FARTHEST_M = 5.0  # and the farthest: one 24-bit depth step there is 0.015 mm
HOST_TOP = 0.08
CAMERA = {"depth_projection": "perspective", "near_m": 0.1, "far_m": 2000.0, "depth_bits": 24,
          "depth_why": "a test camera, three.js's default near and far planes"}


def seen(ctx, nearest_m=NEAREST_M, farthest_m=FARTHEST_M, bits=24, camera=True,
         only: dict[str, float] | None = None):
    """The scene seen through the test camera (`camera` off: a style with none), every piece from
    `nearest_m` out to `farthest_m` (`only` names a piece's own farthest view; a piece left out
    of `only`, when it is given, declares none)."""
    page = {**CAMERA, "depth_bits": bits} if camera else {}
    ctx = dataclasses.replace(ctx, style=Style(ctx.style.units_per_metre, page, ctx.style.dials))
    for name, pack in ctx.packs.items():
        for piece_name, piece in pack.pieces.items():
            far = farthest_m if only is None else only.get(piece_name)
            view = View(nearest_m, "a test distance",
                        **({} if far is None else {"farthest_m": far,
                                                   "farthest_why": "a test distance"}))
            pack = dataclasses.replace(pack, pieces={**pack.pieces, piece_name: dataclasses.replace(
                piece, view=view)})
        ctx = dataclasses.replace(ctx, packs={**ctx.packs, name: pack})
    return ctx


def gap_of(ctx, farthest_m=FARTHEST_M):
    return depth_step(ctx.style, farthest_m) * ctx.packs["kit"].units_per_metre


def contact_of(ctx):
    return sight.contact_length(ctx.packs["kit"].pieces["Thing"], 1.0)


def findings(ctx, only=None):
    return CHECK.run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def two_parts(extra, specs=None, **kw):
    """A block `base` (top at HOST_TOP) and the parts in `extra`, seen through the test camera."""
    block = box((-0.1, 0.0, -0.1), (0.1, HOST_TOP, 0.1))
    return seen(make_ctx({"base": block, **extra}, specs or {}), **kw)


def patch(half=0.05, y=HOST_TOP, tilt=0.0) -> np.ndarray:
    """An open square of two triangles at height y facing up, tilted about x by `tilt` radians."""
    x, z = half, half
    tris = np.array([[(-x, y, -z), (-x, y, z), (x, y, -z)], [(x, y, -z), (-x, y, z), (x, y, z)]])
    if tilt:
        c, s = np.cos(tilt), np.sin(tilt)
        turn = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        tris = (tris - [0, y, 0]) @ turn.T + [0, y, 0]
    return tris


MARK = {"mark": {"on": "base", "role": "decal", "shape": "decal"}}


def lifted_mark_scene():
    """A block with a printed mark 10 mm over it, far over one depth step: it passes, and its
    plant (a face of the block copied into the mark) must be caught."""
    return two_parts({"mark": patch(y=HOST_TOP + 0.01)}, MARK)


# --- registration ---------------------------------------------------------------------------


def test_the_check_is_registered_for_assembly_pieces_and_cites_khronos():
    assert gate.CHECKS["coplanar"] is Coplanar
    assert Coplanar.kinds == frozenset({"assembly"})
    assert CHECK.principle.endswith(".")
    assert "wikis.khronos.org/opengl/Depth_Buffer_Precision" in CHECK.source
    assert "web.archive.org" in CHECK.source


# --- the piece rule -------------------------------------------------------------------------


def test_a_decal_at_the_computed_lift_is_quiet():
    """The lift is the contact length (what a viewer cannot resolve); over one depth step there."""
    ctx = two_parts({"mark": patch(y=HOST_TOP + 0.0)}, MARK)
    lift = contact_of(ctx)
    assert lift > gap_of(ctx)
    ctx = two_parts({"mark": patch(y=HOST_TOP + lift)}, MARK)
    assert findings(ctx) == []
    assert run_gate([Coplanar()], ctx, None).status == "passed"


def test_a_decal_lifted_by_less_than_the_gap_flickers():
    ctx = two_parts({"mark": patch()}, MARK)
    ctx = two_parts({"mark": patch(y=HOST_TOP + 0.5 * gap_of(ctx))}, MARK)
    (found,) = findings(ctx)
    assert (found.check, found.item) == ("coplanar", "kit:Thing base x mark flicker")
    assert "same depth" in found.detail and "24-bit" in found.detail


def test_a_decal_lifted_by_more_than_the_gap_is_quiet():
    ctx = two_parts({"mark": patch()}, MARK)
    assert items(two_parts({"mark": patch(y=HOST_TOP + 2.0 * gap_of(ctx))}, MARK)) == []


def test_the_gap_is_one_depth_step_of_the_cameras_bits():
    """The same lift flickers in a 24-bit buffer and not in a 32-bit one: the gap is derived
    from the camera, not picked."""
    lift = 1e-5
    scene = {"mark": patch(y=HOST_TOP + lift)}
    assert gap_of(two_parts(scene, MARK)) > lift > gap_of(two_parts(scene, MARK, bits=32))
    assert items(two_parts(scene, MARK)) == ["kit:Thing base x mark flicker"]
    assert items(two_parts(scene, MARK, bits=32)) == []


def test_the_gap_grows_with_the_farthest_view():
    scene = {"mark": patch(y=HOST_TOP + 1e-5)}
    assert items(two_parts(scene, MARK, farthest_m=1.0)) == []
    assert items(two_parts(scene, MARK, farthest_m=5.0)) == ["kit:Thing base x mark flicker"]


def test_a_face_moved_into_its_neighbours_plane_flickers():
    """A stud whose top lies flush with the block's top: both face up and overlap."""
    flush = box((-0.02, HOST_TOP - 0.05, -0.02), (0.02, HOST_TOP, 0.02))
    assert items(two_parts({"stud": flush})) == ["kit:Thing base x stud flicker"]
    proud = box((-0.02, HOST_TOP - 0.05, -0.02), (0.02, HOST_TOP + 0.01, 0.02))
    assert items(two_parts({"stud": proud})) == []


def test_faces_that_face_the_other_way_do_not_flicker():
    """A box standing on the block: its bottom faces down against the block's top, facing up."""
    standing = box((-0.02, HOST_TOP, -0.02), (0.02, HOST_TOP + 0.05, 0.02))
    assert items(two_parts({"post": standing})) == []


def test_faces_in_one_plane_that_do_not_overlap_do_not_flicker():
    """Two boxes side by side with their tops level."""
    beside = box((0.1, 0.0, -0.1), (0.3, HOST_TOP, 0.1))
    assert items(two_parts({"beside": beside})) == []


def test_a_face_tilted_a_hair_is_judged_over_the_overlap_not_at_the_far_corner():
    """A small face tilted so that it departs from the block's plane by half a gap across itself
    flickers, and by two gaps does not. A binning of rounded normals would see neither."""
    ctx = two_parts({"mark": patch()}, MARK)
    gap, half = gap_of(ctx), 0.05
    near = two_parts({"mark": patch(half=half, tilt=0.25 * gap / half)}, MARK)
    assert items(near) == ["kit:Thing base x mark flicker"]
    far = two_parts({"mark": patch(half=half, tilt=2.0 * gap / half)}, MARK)
    assert items(far) == []


def tilted_plate(degrees) -> np.ndarray:
    """A thin plate through the block's top plane, turned about z, so its top face crosses that
    plane along a seam 0.18 long (x = 0, z from -0.09 to 0.09) at `degrees`."""
    t = np.radians(degrees)
    turn = np.array([[np.cos(t), -np.sin(t), 0.0], [np.sin(t), np.cos(t), 0.0], [0.0, 0.0, 1.0]])
    plate = box((-0.1, -0.002, -0.09), (0.1, 0.002, 0.09))
    return plate @ turn.T + [0.0, HOST_TOP, 0.0]


def spot_floor_zero(monkeypatch):
    """Take the resolvable-spot floor away: any overlap area above 0 counts."""
    real = coplanar._find
    monkeypatch.setattr(coplanar, "_find",
                        lambda a, b, gap, contact, *hiders: real(a, b, gap, 0.0, *hiders))


def test_a_face_crossing_another_steeply_along_a_long_seam_is_quiet_even_with_no_floor(
        monkeypatch):
    """Two faces 10 degrees apart meet along a seam 0.18 long. Their overlap is a sliver along the
    seam whose corners all lie within the gap of both planes, so a plane test alone (one way or
    both) counts them. The normals must agree (`COPLANAR_NORMAL_COS`), and that is what keeps the
    seam quiet. On a made piece and on a lamp (its two arm tubes cross at the elbow)."""
    spot_floor_zero(monkeypatch)
    ctx = two_parts({"plate": tilted_plate(10.0)})
    assert items(ctx) == []
    lamp = load_scene(BARS / "05-lamp" / "scene")
    assert [i for i in items(lamp) if i.startswith("lamp:Lamp ")] == []
    # without the normal test the sliver counts, on both pieces
    monkeypatch.setattr(coplanar, "COPLANAR_NORMAL_COS", 0.0)
    assert items(ctx) == ["kit:Thing base x plate flicker"]
    assert "lamp:Lamp arm_lower x arm_upper flicker" in items(lamp)


def test_a_crossing_inside_the_normal_test_leaves_a_band_that_does_flicker():
    """At 0.5 degrees (cos 0.99996, over the 0.9999 limit) the planes stay within the gap
    over a band 1.7 mm wide along the seam (gap / sin 0.5 deg), which is a real z-fight of 0.18
    by 1.7 mm, far over one resolvable spot."""
    ctx = two_parts({"plate": tilted_plate(0.5)})
    assert items(ctx) == ["kit:Thing base x plate flicker"]


def test_a_patch_smaller_than_one_resolvable_spot_is_not_reported():
    """The overlap must cover more than the contact length squared: the eye cannot see less."""
    ctx = two_parts({"mark": patch()}, MARK)
    spot = contact_of(ctx)
    lift = 0.5 * gap_of(ctx)
    tiny = two_parts({"mark": patch(half=0.4 * spot, y=HOST_TOP + lift)}, MARK)
    assert items(tiny) == []
    big = two_parts({"mark": patch(half=1.0 * spot, y=HOST_TOP + lift)}, MARK)
    assert items(big) == ["kit:Thing base x mark flicker"]


def test_the_lift_under_the_gap_raises_in_the_build_helper():
    """`sight.decal_lift` is the build's check: the contact length against one depth step at the
    farthest view. Seen from 40 m the step is 0.95 mm, over the 0.29 mm lift."""
    ctx = two_parts({})
    style = ctx.style
    ok = sight.decal_lift("piece 'Thing'", View(1.0, "x", farthest_m=5.0, farthest_why="y"),
                          style, 1.0)
    assert ok.lift == pytest.approx(sight.view_floor(View(1.0, "x"), 1.0)) and ok.lift > ok.gap
    wide = View(1.0, "x", farthest_m=40.0, farthest_why="y")
    with pytest.raises(ManifestError) as raised:
        sight.decal_lift("piece 'Thing'", wide, style, 1.0)
    said = str(raised.value)
    assert "'Thing'" in said and f"{sight.view_floor(wide, 1.0):.7f}" in said
    assert f"{sight.depth_gap(40.0, style, 1.0):.7f}" in said and "too wide" in said
    for view, style_, why in ((None, style, "no view"),
                              (View(1.0, "x"), style, "farthest_m"),
                              (View(1.0, "x", farthest_m=5.0, farthest_why="y"),
                               dataclasses.replace(style, page_camera={}), "depth camera")):
        with pytest.raises(ManifestError, match=why):
            sight.decal_lift("piece 'Thing'", view, style_, 1.0)


def test_a_lift_at_the_gap_raises_and_one_over_it_does_not():
    """One rule for the boundary: two faces a depth step or less apart flicker (`coplanar` flags a
    distance `<= gap`), so a decal lift at the gap raises and one over it builds."""
    gap = sight.depth_gap(5.0, two_parts({}).style, 1.0)
    with pytest.raises(ManifestError, match="not over the depth gap"):
        sight.DecalLift("piece 'Thing'", gap, gap)
    assert sight.DecalLift("piece 'Thing'", gap * (1 + 1e-9), gap).lift > gap
    below = np.array([[(0, 0, 0), (0, 0, 1), (1, 0, 0)]], float)  # the plane y = 0, facing up
    up = np.array([0.0, 1.0, 0.0])
    for apart, seen_as in ((gap, True), (gap * (1 + 1e-9), False)):
        raised = np.array([[(0, apart, 0), (0, apart, 1), (1, apart, 0)]], float)
        area, _ = coplanar._shared_plane(below[0], up, 0.0, raised[0], up, apart, gap)
        assert bool(area > 0.0) is seen_as


# --- measured and not -----------------------------------------------------------------------


def flush_scene(**kw):
    return two_parts({"stud": box((-0.02, HOST_TOP - 0.05, -0.02), (0.02, HOST_TOP, 0.02))}, **kw)


def test_a_piece_without_a_farthest_view_is_unmeasured_and_not_planted():
    bare = flush_scene(only={})
    assert findings(bare) == []
    lines = [x for x in CHECK.coverage(bare) if isinstance(x, Unmeasured)]
    assert any("farthest" in x and "kit:Thing" in x for x in lines), lines
    assert CHECK.plants(bare) == []
    assert run_gate([Coplanar()], bare, None).status == "partial"


def test_a_style_without_a_depth_camera_reads_unmeasured():
    bare = flush_scene(camera=False)
    assert findings(bare) == []
    lines = [x for x in CHECK.coverage(bare) if isinstance(x, Unmeasured)]
    assert any("depth camera" in x for x in lines), lines
    assert run_gate([Coplanar()], bare, None).status == "partial"


def test_a_piece_with_no_view_reads_unmeasured():
    ctx = flush_scene()
    old = ctx.packs["kit"].pieces["Thing"]
    from model_kit.core.checks import pieces
    bare = pieces.with_piece(ctx, "kit", "Thing", dataclasses.replace(old, view=None))
    assert findings(bare) == []
    assert any(isinstance(x, Unmeasured) for x in CHECK.coverage(bare))


def test_a_piece_with_one_part_has_no_pair_and_says_so():
    ctx = seen(make_ctx({"base": box((-0.1, 0.0, -0.1), (0.1, HOST_TOP, 0.1))}))
    assert findings(ctx) == []
    assert any("not applicable" in x and "fewer than two parts" in x
               for x in CHECK.coverage(ctx))
    assert not any(isinstance(x, Unmeasured) for x in CHECK.coverage(ctx))


def test_another_recipe_kind_is_not_judged():
    ctx = seen(make_ctx({"base": box((0, 0, 0), (1, 1, 1))},
                        also={"Tree": {"a": box((0, 0, 0), (1, 1, 1)),
                                       "b": box((0, 0, 0), (1, 1, 1))}}))
    assert findings(ctx) == []
    assert any("not applicable (branching piece)" in x for x in CHECK.coverage(ctx))


# --- a face a closed part hides ------------------------------------------------------------------
#
# A fragment behind a nearer surface fails the depth test and is never drawn, so the flicker of a
# face buried in a closed part cannot be seen. The patches here lie in the plane y = 0.04, inside
# the test block `base` (x and z from -0.1 to 0.1, y from 0 to 0.08).


def test_two_faces_buried_in_a_third_closed_part_do_not_flicker():
    buried = two_parts({"a": patch(y=0.04), "b": patch(y=0.04)})
    assert items(buried) == []
    # the same two faces outside every closed part do flicker
    out = two_parts({"a": patch(y=0.5), "b": patch(y=0.5)})
    assert items(out) == ["kit:Thing a x b flicker"]


def test_two_faces_that_stick_out_of_the_closed_part_still_flicker():
    # the patches are wider than the block, so part of their overlap is in the open
    exposed = two_parts({"a": patch(half=0.15, y=0.04), "b": patch(half=0.15, y=0.04)})
    assert items(exposed) == ["kit:Thing a x b flicker"]


def test_two_feet_pressed_onto_a_plate_face_into_it_and_do_not_flicker():
    """Two boxes stand on the block's top; their overlapping bottoms lie flush on the plate and
    face down into it (a guitar's ArmCollar and ClawScrew_6 on its BridgePlate)."""
    foot_a = box((-0.03, HOST_TOP, -0.03), (0.03, HOST_TOP + 0.05, 0.03))
    foot_b = box((-0.02, HOST_TOP, -0.02), (0.04, HOST_TOP + 0.07, 0.04))
    assert items(two_parts({"a": foot_a, "b": foot_b})) == []


def test_the_verdict_on_a_hidden_pair_does_not_depend_on_the_order_of_the_parts():
    """Two feet stand on the plate and overlap. One bottom lies flush on the plate (hidden by it);
    the other is lifted 0.9 of a depth step, so it is not on the plate's surface and the plate
    hides only the first. A flicker is seen where at least one face is seen: the pair stays a
    finding, whichever foot comes first."""
    ctx = two_parts({})
    lift = 0.9 * gap_of(ctx)
    flush = box((-0.03, HOST_TOP, -0.03), (0.03, HOST_TOP + 0.05, 0.03))
    lifted = box((-0.02, HOST_TOP + lift, -0.02), (0.04, HOST_TOP + lift + 0.07, 0.04))
    got = [items(two_parts({"a": flush, "b": lifted})), items(two_parts({"a": lifted, "b": flush})),
           items(two_parts({"b": lifted, "a": flush})), items(two_parts({"b": flush, "a": lifted}))]
    assert got == [["kit:Thing a x b flicker"]] * 4, got
    # both bottoms flush on the plate: both hidden by it, so quiet in either order
    other = box((-0.02, HOST_TOP, -0.02), (0.04, HOST_TOP + 0.07, 0.04))
    assert items(two_parts({"a": flush, "b": other})) == []
    assert items(two_parts({"a": other, "b": flush})) == []


def test_two_faces_flush_on_the_plate_and_facing_out_still_flicker():
    """The same two faces, turned up: they face away from the plate, so it does not hide them."""
    assert items(two_parts({"a": patch(y=HOST_TOP), "b": patch(y=HOST_TOP)})
                 ) == ["kit:Thing a x b flicker", "kit:Thing a x base flicker",
                       "kit:Thing b x base flicker"]


def test_a_part_that_is_not_closed_hides_nothing():
    open_box = box((-0.1, 0.0, -0.1), (0.1, HOST_TOP, 0.1))[:-1]  # one triangle short of closed
    ctx = seen(make_ctx({"base": open_box, "a": patch(y=0.04), "b": patch(y=0.04)}))
    assert items(ctx) == ["kit:Thing a x b flicker"]


def plate(centre, half, y=0.1) -> np.ndarray:
    """An open square of two triangles at height y facing up, `half` a side from `centre` (x, z)."""
    cx, cz = centre
    return patch(half=half, y=y) + [cx, 0.0, cz]


def buried_yard(half_slab, half_block):
    """A slab placement holding a closed box (0 to 0.2 on each axis) and a plate in it, and a block
    placement at the same spot with a plate of its own in the same plane."""
    geometry = {**kit_geometry(),
                "kit:Slab": PieceGeom("Slab", {"top": box((0, 0, 0), (0.2, 0.2, 0.2)),
                                               "print": plate((0.1, 0.1), half_slab)}),
                "kit:Block": PieceGeom("Block", {"body": plate((0.1, 0.1), half_block)})}
    ctx = yard(at("a", "Slab"), at("b", "Block"), geometry=geometry)
    return seen(ctx, only={"Slab": 40.0, "Block": 40.0, "Cord": 40.0, "Tree": 40.0})


def test_two_placements_whose_faces_are_buried_in_one_of_them_do_not_flicker():
    assert [f.item for f in findings(buried_yard(0.05, 0.05)) if f.item.startswith("yard")] == []
    # wider than the box they sit in: the overlap sticks out into the open
    assert [f.item for f in findings(buried_yard(0.15, 0.15))
            if f.item.startswith("yard")] == ["yard a x b flicker"]


# --- the layout rule ------------------------------------------------------------------------


def slabs(first=0.0, second=0.0, farthest=None, apart=0.1):
    """Two placements of the slab piece, the second `apart` along x and z (so no side faces share a
    plane) and `second - first` higher,
    with the farthest views of the Block and the Slab piece in `farthest` (a dict by piece)."""
    geometry = {**kit_geometry(),
                "kit:Slab": PieceGeom("Slab", {"top": box((0, 0, 0), (0.2, 0.05, 0.2))})}
    ctx = yard(at("a", "Slab", y=first), at("b", "Slab", apart, second, apart), geometry=geometry)
    return seen(ctx, nearest_m=NEAREST_M, only=farthest or {"Slab": 40.0, "Block": 40.0,
                                                            "Cord": 40.0, "Tree": 40.0})


def test_two_placements_in_one_plane_flicker():
    ctx = slabs()
    (found,) = [f for f in findings(ctx) if f.item.startswith("yard")]
    assert found.item == "yard a x b flicker"
    assert "a top with b top" in found.detail


def test_two_placements_a_gap_and_a_half_apart_do_not_flicker():
    ctx = slabs()
    gap = depth_step(ctx.style, 40.0)
    assert items(slabs(second=1.5 * gap)) == []
    assert items(slabs(second=0.5 * gap)) == ["yard a x b flicker"]


def slab_and_block(lift, slab_far, block_far):
    """A slab and a block placed so that the block's top is `lift` over the slab's top, a corner
    of each over the other, each piece seen out to its own farthest view."""
    geometry = {**kit_geometry(),
                "kit:Slab": PieceGeom("Slab", {"top": box((0, 0, 0), (0.2, 0.05, 0.2))}),
                "kit:Block": PieceGeom("Block", {"body": box((0, 0, 0), (0.2, 0.2, 0.2))})}
    ctx = yard(at("slab", "Slab"), at("block", "Block", 0.1, 0.05 - 0.2 + lift, 0.1),
               geometry=geometry)
    return seen(ctx, only={"Slab": slab_far, "Block": block_far, "Cord": 40.0, "Tree": 40.0})


def test_a_pair_of_placements_uses_the_larger_farthest_view():
    """The two tops are 3.5 mm apart. Each piece seen out to 40 m has a 0.95 mm step and the
    faces separate; a piece seen out to 100 m has a 6 mm step, and the pair uses the larger."""
    ctx = slab_and_block(0.0, 40.0, 40.0)
    near_step, far_step = depth_step(ctx.style, 40.0), depth_step(ctx.style, 100.0)
    lift = 0.5 * (near_step + far_step)
    assert near_step < lift < far_step

    def layout(c):
        return [f.item for f in findings(c) if f.item.startswith("yard")]

    assert layout(slab_and_block(lift, 40.0, 40.0)) == []
    assert layout(slab_and_block(lift, 40.0, 100.0)) == ["yard block x slab flicker"]
    assert layout(slab_and_block(lift, 100.0, 40.0)) == ["yard block x slab flicker"]
    assert layout(slab_and_block(0.5 * near_step, 40.0, 40.0)) == ["yard block x slab flicker"]


def test_a_joined_pair_is_judged_too():
    ctx = slabs()
    layout = ctx.layouts["yard"]
    b = dataclasses.replace(layout.placements["b"], rests_on="a")
    ctx = dataclasses.replace(ctx, layouts={"yard": Layout(
        "yard", layout.frame, {**layout.placements, "b": b}, {})})
    assert "yard a x b flicker" in items(ctx)


def test_a_layout_of_one_placement_has_no_pair():
    ctx = seen(yard(at("a", "Block")), only={"Block": 40.0})
    assert not [f for f in findings(ctx) if f.item.startswith("yard")]
    assert any("fewer than two pieces" in x for x in CHECK.coverage(ctx))


# --- measured -------------------------------------------------------------------------------


def test_measured_lists_the_parts_a_pair_could_be_compared_in():
    ctx = flush_scene()
    assert CHECK.measured(ctx) == {"kit:Thing base", "kit:Thing stud"}
    # one part has no pair; no farthest view, no depth camera, no view: nothing is judged
    one = seen(make_ctx({"base": box((-0.1, 0.0, -0.1), (0.1, HOST_TOP, 0.1))}))
    assert CHECK.measured(one) == set()
    assert CHECK.measured(flush_scene(only={})) == set()
    assert CHECK.measured(flush_scene(camera=False)) == set()
    # and a branching piece is not a part of it
    other = seen(make_ctx({"base": box((0, 0, 0), (1, 1, 1))},
                          also={"Tree": {"a": box((0, 0, 0), (1, 1, 1)),
                                         "b": box((0, 0, 0), (1, 1, 1))}}))
    assert not any(k.startswith("kit:Tree ") for k in CHECK.measured(other))


def test_measured_lists_the_parts_of_the_pieces_in_a_layout_that_can_be_judged():
    assert "kit:Slab top" in CHECK.measured(slabs())
    assert CHECK.measured(slabs(farthest={"Slab": None})) == set()  # no farthest view: none


# --- plants ---------------------------------------------------------------------------------


def test_a_face_copied_into_its_neighbour_is_caught_on_two_pieces():
    """The plant on a made piece and on bar 05's two assembly pieces (Lamp, Strip) and its
    layout."""
    made = tried_and_caught(CHECK, two_parts({"mark": patch(y=HOST_TOP + 2.0 * 0.0015)}, MARK))
    assert made == ["kit:Thing base x mark flicker"]
    lamp = tried_and_caught(CHECK, load_scene(BARS / "05-lamp" / "scene"))
    pieces_planted = {i.split(" ")[0] for i in lamp if ":" in i.split(" ")[0]}
    assert {"lamp:Lamp", "lamp:Strip"} <= pieces_planted, lamp
    assert any(i.startswith("desk ") and i.endswith(" flicker") for i in lamp), lamp


def test_the_plants_come_from_the_geometry_and_never_call_the_verdict(monkeypatch):
    ctx = two_parts({"mark": patch(y=HOST_TOP + 0.01)}, MARK)
    refuses_the_core(monkeypatch, coplanar, "_shared_plane", "_overlap", "_find", "_candidates")
    assert [u.item for u in CHECK.plants(ctx)] == ["kit:Thing base x mark flicker"]
    layout = slabs()
    assert [u.item for u in CHECK.plants(layout) if u.item.startswith("yard")] == [
        "yard a x a_plant flicker"]


def test_a_blind_core_stops_the_run(monkeypatch):
    ctx = two_parts({"mark": patch(y=HOST_TOP + 0.01)}, MARK)
    assert run_gate([Coplanar()], ctx, None).status == "passed"
    monkeypatch.setattr(coplanar, "_shared_plane", lambda *a, **k: (0.0, np.zeros((0, 3))))
    with pytest.raises(BlindCheck, match="coplanar"):
        run_gate([Coplanar()], ctx, None)


# --- the known-good bar ---------------------------------------------------------------------


def test_lamp_bar_is_quiet():
    """The lamp (on a strip on a desk) is a known-good piece: its decal is lifted by the
    contact length, and no two of its faces share a plane."""
    ctx = load_scene(BARS / "05-lamp" / "scene")
    assert findings(ctx) == []
    assert not [x for x in CHECK.coverage(ctx) if isinstance(x, Unmeasured)]
    assert run_gate([Coplanar()], ctx, None).status == "passed"


def test_the_stock_style_in_these_tests_declares_no_camera():
    assert STYLE.depth_camera is None
