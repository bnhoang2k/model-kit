import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import frames
from model_kit.core.checks.frames import Frames
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import Frame, Prop, Recipe
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    OFF_ANGLE_DEG,
)

from ..bodies.humanoid import humanoid_scene
from ..shapes import box, open_plate
from .manifest_scenes import (
    HOLD_Z,
    LIMIT,
    TOP,
    make_ctx,
    plants_caught,
    with_clip,
    with_hold,
    with_piece,
)
from .piece_scenes import accepted, refuses_the_core, tried_and_caught
from .sight_scenes import unviewed, viewed


def findings(ctx, only=None):
    return Frames().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["frames"] is Frames
    assert Frames.name == "frames"
    assert Frames.kinds == frozenset({"assembly"})


def test_a_clean_scene_passes_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(Frames(), ctx)
    assert run_gate([Frames()], ctx, None).status == "passed"


# --- frame match ------------------------------------------------------------------------------


def test_a_clip_that_reads_another_frame_fails():
    ctx = with_clip(make_ctx(), "play", frame="blender")
    assert items(ctx) == ["kit:Deck frame clip play"]
    assert "blender" in findings(ctx)[0].detail and "gltf" in findings(ctx)[0].detail


def test_a_layout_that_reads_another_frame_fails():
    ctx = make_ctx()
    layout = dataclasses.replace(ctx.layouts["floor"], frame="blender")
    assert items(dataclasses.replace(ctx, layouts={"floor": layout})) == [
        "kit:Deck frame layout floor"]


def test_a_piece_in_another_frame_fails_for_each_consumer_that_reads_it():
    ctx = with_piece(make_ctx(), "Deck", frame=Frame("blender", "centre of the base"))
    assert items(ctx) == ["kit:Deck frame clip play", "kit:Deck frame clip play_L",
                          "kit:Deck frame layout floor"]


def test_a_carried_prop_is_a_consumer_too():
    ctx = make_ctx()
    carried = Prop("stick", "Deck", "kit", "hand.R", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                   ((1.0, 10.0),))
    gear = dataclasses.replace(ctx.clips["play"].floor_gear, pieces={})
    ctx = with_clip(ctx, "play", props={"stick": carried}, floor_gear=gear, holds=(), targets=(),
                    authoring="keyed")  # a solved clip with no key targets cannot be built
    ctx = with_piece(ctx, "Deck", frame=Frame("blender", "centre of the base"))
    assert "kit:Deck frame clip play" in items(ctx)


def test_a_piece_nobody_places_has_no_frame_item():
    ctx = with_piece(make_ctx(), "Doll", frame=Frame("blender", "centre of the base"))
    assert findings(ctx) == []


def test_a_scene_that_places_nothing_says_so_in_a_plain_line():
    ctx = dataclasses.replace(make_ctx(), clips={}, layouts={}, samples={}, rig_of_clip={})
    (line,) = [c for c in Frames().coverage(ctx) if "places" in c]
    assert "frame match" in line and not isinstance(line, Unmeasured)


# --- on surface -------------------------------------------------------------------------------


def lifted(ctx, id, gap, kind="hold"):
    point = (0.1, TOP + gap, HOLD_Z)
    if kind == "hold":
        return with_hold(ctx, id, point=point)
    piece = ctx.packs["kit"].pieces["Deck"]
    field = "seats" if kind == "seat" else "mounts"
    rows = tuple(dataclasses.replace(r, point=point) for r in getattr(piece, field))
    return with_piece(ctx, "Deck", **{field: rows})


@pytest.mark.parametrize("kind, id", [("hold", "hold_R"), ("seat", "seat_1"), ("mount", "jack")])
def test_calibration_half_the_limit_passes_and_double_fails(kind, id):
    ctx = make_ctx()
    assert findings(lifted(ctx, id, CALIBRATION_PASS * LIMIT, kind)) == []
    (found,) = findings(lifted(ctx, id, CALIBRATION_FAIL * LIMIT, kind))
    assert (found.check, found.item) == ("frames", f"kit:Deck {kind} {id} on-surface")


def test_a_point_below_the_surface_is_measured_by_distance_too():
    ctx = lifted(make_ctx(), "hold_R", -TOP - CALIBRATION_FAIL * LIMIT)  # under the plate
    assert "kit:Deck hold hold_R on-surface" in items(ctx)


def test_a_mount_may_lie_on_any_part_of_the_piece_but_a_hold_only_on_its_surface_parts():
    parts = {"top": box((-0.2, 0.0, -0.1), (0.2, TOP, 0.1)),
             "post": box((0.5, 0.0, 0.5), (0.6, 0.3, 0.6))}
    ctx = make_ctx()
    ctx = dataclasses.replace(ctx, pieces={"kit:Deck": PieceGeom("Deck", parts)})
    jack = dataclasses.replace(ctx.packs["kit"].pieces["Deck"].mounts[0], point=(0.5, 0.2, 0.55))
    ctx = with_piece(ctx, "Deck", mounts=(jack,))
    assert findings(ctx) == []  # on the post, a part that is not a hold surface
    ctx = with_hold(ctx, "hold_R", point=(0.5, 0.2, 0.55))
    assert items(ctx) == ["kit:Deck hold hold_R on-surface"]


def test_a_piece_in_blender_axes_puts_every_point_off_its_surface():
    parts = {n: t[..., [0, 2, 1]] for n, t in make_ctx().pieces["kit:Deck"].parts.items()}
    ctx = dataclasses.replace(make_ctx(), pieces={"kit:Deck": PieceGeom("Deck", parts)})
    found = {f.item for f in findings(ctx)}
    assert {"kit:Deck hold hold_R on-surface", "kit:Deck seat seat_1 on-surface",
            "kit:Deck mount jack on-surface"} <= found


# --- off --------------------------------------------------------------------------------------


def tilted(ctx, degrees, id="hold_R"):
    a = np.radians(degrees)
    return with_hold(ctx, id, off=(float(np.sin(a)), float(np.cos(a)), 0.0))


def test_calibration_half_the_angle_passes_and_double_fails():
    ctx = make_ctx()
    assert findings(tilted(ctx, CALIBRATION_PASS * OFF_ANGLE_DEG)) == []
    (found,) = findings(tilted(ctx, CALIBRATION_FAIL * OFF_ANGLE_DEG))
    assert found.item == "kit:Deck hold hold_R off"
    assert f"{CALIBRATION_FAIL * OFF_ANGLE_DEG:.1f}" in found.detail


def test_a_hand_that_passes_under_a_thin_plate_fails():
    ctx = with_hold(make_ctx(), "hold_R", off=(0.0, -1.0, 0.0))
    (found,) = findings(ctx)
    assert found.item == "kit:Deck hold hold_R off"
    assert "180" in found.detail


def test_faces_tied_for_nearest_all_count():
    # inside the plate, halfway between its top and bottom: both faces are the nearest
    mid = (0.1, TOP / 2, HOLD_Z)
    ctx = with_hold(make_ctx(), "hold_R", point=mid, off=(0.0, -1.0, 0.0))
    assert "kit:Deck hold hold_R off" not in items(ctx)
    ctx = with_hold(make_ctx(), "hold_R", point=mid, off=(1.0, 0.0, 0.0))
    assert "kit:Deck hold hold_R off" in items(ctx)


def test_an_open_surface_is_judged_by_its_winding():
    parts = {"top": open_plate()}  # the +y face of a 0.2 plate, two triangles
    ctx = dataclasses.replace(make_ctx(), pieces={"kit:Deck": PieceGeom("Deck", parts)})
    ctx = with_hold(ctx, "hold_R", point=(0.05, 0.002, 0.04))
    assert "kit:Deck hold hold_R off" not in items(ctx)
    flipped = with_hold(ctx, "hold_R", point=(0.05, 0.002, 0.04), off=(0.0, -1.0, 0.0))
    assert "kit:Deck hold hold_R off" in items(flipped)


def test_a_closed_part_wound_inward_is_turned_outward_even_with_a_sliver():
    inward = box((-0.2, 0.0, -0.1), (0.2, TOP, 0.1))[:, [0, 2, 1]]  # every face reversed
    sliver = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    parts = {"top": np.concatenate([inward, sliver])}
    ctx = dataclasses.replace(make_ctx(), pieces={"kit:Deck": PieceGeom("Deck", parts)})
    assert "kit:Deck hold hold_R off" not in items(ctx)
    down = with_hold(ctx, "hold_R", off=(0.0, -1.0, 0.0))
    assert "kit:Deck hold hold_R off" in items(down)


def test_a_hold_surface_part_with_no_geometry_raises():
    ctx = dataclasses.replace(make_ctx(), pieces={"kit:Deck": PieceGeom("Deck", {"other": box(
        (0, 0, 0), (1, 1, 1))})})
    with pytest.raises(ManifestError, match="top"):
        findings(ctx)


# --- scope, coverage and plants ---------------------------------------------------------------


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = tilted(tilted(make_ctx(), 90.0), 90.0, "hold_L")
    assert items(ctx) == ["kit:Deck hold hold_L off", "kit:Deck hold hold_R off"]
    assert [f.item for f in findings(ctx, only={"kit:Deck hold hold_L off"})] == [
        "kit:Deck hold hold_L off"]


def test_a_scoped_run_of_one_item_of_a_hold_leaves_out_its_sibling_item():
    # hold_R fails both its on-surface item and its off item
    ctx = tilted(lifted(make_ctx(), "hold_R", CALIBRATION_FAIL * LIMIT), 90.0)
    on, off = "kit:Deck hold hold_R on-surface", "kit:Deck hold hold_R off"
    assert {f.item for f in findings(ctx)} >= {on, off}
    assert [f.item for f in findings(ctx, only={on})] == [on]
    assert [f.item for f in findings(ctx, only={off})] == [off]
    assert {f.item for f in findings(ctx, only={on, off})} == {on, off}


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = with_piece(make_ctx(), "Doll", recipe=Recipe("branching", "grow.py"))
    (line,) = [c for c in Frames().coverage(ctx) if "branching" in c]
    assert line == "frames: not applicable (branching piece): kit:Doll"
    assert not isinstance(line, Unmeasured)
    assert run_gate([Frames()], ctx, None).status == "passed"


def test_a_piece_with_points_but_no_geometry_is_unmeasured_for_them():
    ctx = dataclasses.replace(make_ctx(), pieces={})
    (line,) = [c for c in Frames().coverage(ctx) if "no geometry" in c]
    assert line == "frames: skipped (no geometry): kit:Deck"
    assert isinstance(line, Unmeasured)
    assert run_gate([Frames()], ctx, None).status == "partial"
    assert findings(ctx) == []  # the frame match needs no geometry, and it passes


def test_a_piece_without_points_needs_no_geometry():
    ctx = make_ctx()  # the doll has neither points nor geometry
    assert not [c for c in Frames().coverage(ctx) if "Doll" in c]


def test_every_item_has_a_plant_that_fails_it():
    plants = Frames().plants(make_ctx())
    assert {p.item for p in plants} == {
        "kit:Deck frame clip play", "kit:Deck frame clip play_L", "kit:Deck frame layout floor",
        "kit:Deck hold hold_R off", "kit:Deck hold hold_L off", "kit:Deck hold hold_B off",
        "kit:Deck hold hold_R on-surface", "kit:Deck hold hold_L on-surface",
        "kit:Deck hold hold_B on-surface", "kit:Deck seat seat_1 on-surface",
        "kit:Deck mount jack on-surface"}


def test_the_plants_are_frame_tilted_and_swapped():
    check, ctx = Frames(), make_ctx()
    labels = {accepted(check, u).label.split(":")[0] for u in check.plants(ctx)}
    assert labels == {"frame", "tilted", "swapped"}  # a frame, an off, a y/z swap


def test_frames_plants_never_call_the_core(monkeypatch):
    ctx = make_ctx()
    want = {u.item for u in Frames().plants(ctx)}
    refuses_the_core(monkeypatch, frames, "_findings", "_entries", "_point_entries", "_frame_entry")
    monkeypatch.setattr(Frames, "run", lambda *a, **k: 1 / 0)
    assert {u.item for u in Frames().plants(ctx)} == want and len(want) == 11


def test_frames_plants_caught_on_two_pieces():
    """The Deck (frame, on-surface and off items) and the humanoid scene's Chair and Bench (a seat
    each): every unit the gate tries has a candidate the check catches, and the gate does not
    stop."""
    deck = tried_and_caught(Frames(), make_ctx())
    assert "kit:Deck hold hold_R off" in deck and "kit:Deck mount jack on-surface" in deck
    people = viewed(humanoid_scene(), LIMIT)
    seats = tried_and_caught(Frames(), people)
    assert {"humanoid:Chair seat seat_1 on-surface", "humanoid:Bench seat seat_1 on-surface"
            } <= set(seats)
    for ctx in (make_ctx(), people):
        run_gate([Frames()], ctx, None)  # no BlindCheck


def test_an_off_plant_has_four_turns_and_a_reverse_for_each_normal_tied_for_nearest():
    (unit,) = [u for u in Frames().plants(make_ctx()) if u.item == "kit:Deck hold hold_R off"]
    assert len(unit.candidates) >= 5 and len(unit.candidates) % 5 == 0


def test_a_blind_check_stops_the_run():
    class Blind(Frames):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="frames"):
        run_gate([Blind()], make_ctx(), None)


# --- the contact length is the piece's own ----------------------------------------------------


def test_a_piece_with_no_view_has_no_on_surface_verdict_and_still_judges_off():
    far = unviewed(lifted(make_ctx(), "hold_R", 10 * LIMIT))  # well off the surface, were it read
    assert findings(far) == []
    assert not [u for u in Frames().plants(far) if u.item.endswith("on-surface")]
    (line,) = [c for c in Frames().coverage(far) if "no view" in c]
    assert isinstance(line, Unmeasured) and line.endswith("kit:Deck")
    assert run_gate([Frames()], far, None).status == "partial"
    # the `off` rule does not read a length: a tilted hold still fails, and still has its plant
    turned = unviewed(tilted(make_ctx(), 90.0))
    assert items(turned) == ["kit:Deck hold hold_R off"]
    assert any(u.item == "kit:Deck hold hold_R off" for u in Frames().plants(turned))


def test_the_limit_follows_the_pieces_view_not_a_share_of_its_wall():
    ctx = lifted(make_ctx(), "hold_R", 3 * LIMIT)
    assert items(ctx) == ["kit:Deck hold hold_R on-surface"]
    assert items(viewed(ctx, 5 * LIMIT)) == []  # seen from further off, the same gap touches
