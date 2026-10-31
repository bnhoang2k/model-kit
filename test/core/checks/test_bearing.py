import dataclasses
import functools
import re
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate, geom
from model_kit.core.allow import Allowlist
from model_kit.core.checks import bearing, pieces
from model_kit.core.checks.bearing import Bearing, depth_in, hull_2d
from model_kit.core.checks.crossings import Crossings
from model_kit.core.errors import ManifestError
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom, load_scene, piece_key
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    ENGAGE_OF_DIAMETER,
)

from ..bodies.humanoid import humanoid_scene
from ..shapes import box, cylinder
from .bearing_scenes import (
    PUSHED_PIECES,
    STRADDLE_PIECES,
    SUNK_PIECES,
    TALL_PIECES,
    against_a_faceted_round,
    bevelled_block,
    bolt,
    bolted_post,
    bond,
    creased,
    declare,
    hanging_clamp,
    held_bar,
    ledge_and_wall,
    noise_of,
    pushed_post,
    rigged,
    shifted,
    sideways,
    stack,
    straddling,
    sunk_post,
    tabletop_on_legs,
    tall_post,
    wedge_on_block,
)
from .piece_scenes import TOUCH, make_ctx, plants_caught, refuses_the_core
from .sight_scenes import unviewed, viewed

REGRESSIONS = Path(__file__).resolve().parents[1] / "fixtures" / "regressions"
BARS = Path(__file__).resolve().parents[3] / "bars"
THING = "kit:Thing"


def findings(ctx, only=None):
    return Bearing().run(ctx, only=only)


def items(ctx):
    return {f.item: f.detail for f in findings(ctx)}


def lines(ctx):
    return [str(x) for x in Bearing().coverage(ctx)]


def entry(ctx, part: str, rule: str):
    """The check's entry for one rule on one part of `kit:Thing`."""
    view = next(pieces.measured_pieces(ctx, {"assembly"}))
    return next(e for e in bearing.measure(ctx, view, {part}) if e.rule == rule)


def mm_in(detail: str, after: str) -> float:
    return float(re.search(rf"{after}\D*?(\d+\.\d+) mm", detail).group(1))


# --- the check, its source and its tolerances ---------------------------------------------------


def test_the_check_is_registered_with_coverage_last_and_measures_assembly_pieces():
    assert gate.CHECKS["bearing"] is Bearing
    assert Bearing.kinds == frozenset({"assembly"})
    assert list(gate.CHECKS)[-1] == "coverage"


def test_it_states_a_principle_and_cites_the_thread_engagement_sources():
    check = Bearing()
    assert "nord-lock.com" in check.source and "ntrs.nasa.gov" in check.source
    assert check.principle.endswith(".")


def test_engagement_is_one_diameter_in_steel():
    assert ENGAGE_OF_DIAMETER == 1.0


def test_the_touch_distance_is_the_pieces_contact_length_the_floaters_distance_too():
    ctx = make_ctx({"slab": box((-0.3, 0.0, -0.3), (0.3, 0.1, 0.3))})
    view = next(pieces.measured_pieces(ctx, {"assembly"}))
    assert bearing._Piece(ctx, view).touch == view.contact == pytest.approx(TOUCH)


# --- rule 1: rest or hang -----------------------------------------------------------------------


def test_block_on_block_is_stable():
    ctx = stack()
    assert findings(ctx) == []
    assert run_gate([Bearing()], ctx, None).status == "passed"


def test_tabletop_on_legs_is_stable():
    """Each leg is `on` the top and hangs from its underside; its top face sits wholly under it."""
    ctx = tabletop_on_legs()
    assert findings(ctx) == []
    assert plants_caught(Bearing(), ctx)
    assert not [x for x in lines(ctx) if "rest rule" in x and "not applicable" in x]


def test_overhanging_plate_tips():
    ctx = stack(x=0.35)  # its centre of mass is 0.05 past the slab's edge
    got = items(ctx)
    assert f"{THING} block tips off slab" in got
    assert "centre of mass is outside" in got[f"{THING} block tips off slab"]
    # the same part shifted back is stable
    assert findings(stack(x=0.1)) == []


def test_a_part_inside_its_patch_by_less_than_the_touch_is_stable():
    """The margin is float noise, not a touch (the touch is a contact band, not a statics fact).
    The centre of mass is 15 mm inside the slab's edge, under the 21 mm touch distance: the part
    stands, though it overhangs, which rule 3 reports."""
    got = items(stack(x=0.285))
    assert not [i for i in got if "tips off" in i]
    assert f"{THING} block overhangs slab" in got
    # on the edge itself the centre of mass is not inside
    assert "on the edge of" in items(stack(x=0.3))[f"{THING} block tips off slab"]


def test_a_cap_centred_on_its_own_post_is_stable():
    """A 300 mm cap on a 40 mm post: the post is narrower than two touch distances, and the cap
    still stands, since its centre of mass lies over the post. Moved 30 mm, it is past the edge."""
    parts = {"post": box((-0.02, 0.0, -0.02), (0.02, 0.5, 0.02)),
             "cap": box((-0.15, 0.5, -0.15), (0.15, 0.55, 0.15))}
    ctx = make_ctx(parts, {"cap": {"on": "post"}})
    assert not [i for i in items(ctx) if "tips off" in i]
    assert f"{THING} cap tips off post" in items(shifted(ctx, "cap", (0.03, 0.0, 0.0)))


def test_hanging_clamp_is_stable():
    ctx = hanging_clamp()
    assert findings(ctx) == []
    assert plants_caught(Bearing(), ctx)


def test_a_clamp_hung_off_the_end_of_its_rail_swings():
    got = items(hanging_clamp(x=0.45))
    assert f"{THING} clamp swings off rail" in got


def test_a_hanging_part_must_lie_under_its_patch():
    # the clamp's body hangs under the rail, but a heavier weight on it sits far above the rail
    weight = box((-0.05, 1.5, -0.04), (0.05, 2.5, 0.04))
    parts = {"rail": box((-0.4, 0.8, -0.05), (0.4, 0.9, 0.05)),
             "clamp": np.concatenate([box((-0.05, 0.5, -0.04), (0.05, 0.8, 0.04)), weight])}
    got = items(make_ctx(parts, {"clamp": {"on": "rail"}}, hosts=["rail"]))
    assert "above the patch's lowest point" in got[f"{THING} clamp swings off rail"]


def test_sideways_patch_not_applicable():
    ctx = sideways()
    assert findings(ctx) == []
    got = lines(ctx)
    assert any("not applicable" in x and "faces neither up nor down as a whole" in x
               and "rest rule" in x and "shelf" in x for x in got)
    assert not any(isinstance(x, Unmeasured) for x in Bearing().coverage(ctx))


def test_a_small_up_facing_ledge_does_not_make_a_sideways_patch_a_rest():
    """The patch is read as a whole. A shelf on a thin ledge and against a wall is held
    mostly by the wall: its patch's area-weighted normal is sideways."""
    ctx = ledge_and_wall()
    assert not [f for f in findings(ctx) if "tips off" in f.item or "swings off" in f.item]
    assert entry(ctx, "shelf", "rest").status == "na"
    assert any("faces neither up nor down as a whole" in x and "shelf" in x for x in lines(ctx))


def test_a_part_held_between_facing_surfaces_is_not_judged_by_gravity():
    # one frame part with a floor and a ceiling (two shells); the block sits between them
    frame = np.concatenate([box((-0.3, 0.0, -0.3), (0.3, 0.1, 0.3)),
                            box((-0.3, 0.4, -0.3), (0.3, 0.5, 0.3))])
    parts = {"frame": frame, "block": box((0.25, 0.1, -0.1), (0.45, 0.4, 0.1))}
    ctx = make_ctx(parts, {"block": {"on": "frame"}})
    assert not [f for f in findings(ctx) if "tips off" in f.item or "swings off" in f.item]
    assert any("faces neither up nor down as a whole" in x and "block" in x for x in lines(ctx))
    assert any("held between surfaces" in x and "block" in x and "seat rule" in x
               for x in lines(ctx))


def test_a_narrow_patch_holds_a_part_centred_on_it_and_loses_one_off_it():
    """There is no minimum patch size: a 10 mm strip under a centred cap holds it (no rule in
    the spec or its sources sets one); a cap whose centre is past the strip's edge tips."""
    parts = {"strip": box((-0.005, 0.0, -0.005), (0.005, 0.1, 0.005)),
             "cap": box((-0.1, 0.1, -0.1), (0.1, 0.2, 0.1))}
    ctx = make_ctx(parts, {"cap": {"on": "strip"}})
    assert not [f for f in findings(ctx) if "tips off" in f.item]
    assert f"{THING} cap tips off strip" in items(shifted(ctx, "cap", (0.02, 0.0, 0.0)))


def test_a_part_with_no_contact_with_its_host_is_for_floaters():
    parts = {"slab": box((-0.3, 0.0, -0.3), (0.3, 0.1, 0.3)),
             "block": box((-0.1, 0.5, -0.1), (0.1, 0.7, 0.1))}
    ctx = make_ctx(parts, {"block": {"on": "slab"}})
    assert findings(ctx) == []
    assert any("no face of it bears on its host" in x for x in lines(ctx))


def test_a_part_that_is_not_a_closed_solid_has_no_seat_thickness_and_is_unmeasured():
    parts = {"slab": box((-0.3, 0.0, -0.3), (0.3, 0.1, 0.3)),
             "block": box((-0.1, 0.1, -0.1), (0.1, 0.3, 0.1))[:-2]}  # one end face missing
    ctx = make_ctx(parts, {"block": {"on": "slab"}})
    assert findings(ctx) == []
    assert any(isinstance(x, Unmeasured) and "no seat thickness" in x
               for x in Bearing().coverage(ctx))


def _bears(ctx) -> bool:
    """Some face of the post bears on the base (the rest rule judges it, or finds it stable)."""
    return not any("no face of it bears on its host" in x for x in lines(ctx))


@pytest.mark.parametrize("piece", SUNK_PIECES)
def test_sunk_just_under_half_thinner_keeps_bearing_contact_on_two_pieces(piece):
    """The slack is the depth crossings fails at, so a part crossings accepts keeps its contact."""
    assert _bears(sunk_post(piece, 0.96))
    assert findings(sunk_post(piece, 0.96)) == []
    assert not _bears(sunk_post(piece, 1.04))  # past it the faces no longer bear: floaters' part


def test_the_sink_is_read_once_per_part_host_pair(monkeypatch):
    seen = []
    real = geom.thinner_thickness
    monkeypatch.setattr(geom, "thinner_thickness",
                        lambda a, b, point: seen.append(point) or real(a, b, point))
    ctx = tabletop_on_legs()  # four legs, each on the top
    lines(ctx)
    assert len(seen) == 4


def test_a_seat_with_no_readable_thickness_is_unmeasured_and_the_run_goes_on(monkeypatch):
    ctx = sunk_post("square post on a thin base", 0.5)

    def unreadable(tris, point):
        raise ValueError("local_thickness: no ray hit an opposite surface")

    assert findings(ctx) == [] and not any(isinstance(x, Unmeasured) for x in lines(ctx))
    monkeypatch.setattr(geom, "local_thickness", unreadable)
    assert findings(ctx) == []  # no number, and no exit 2
    got = [x for x in Bearing().coverage(ctx) if isinstance(x, Unmeasured)]
    assert {("; rest rule)" in x, "; seat rule)" in x, "; sink rule)" in x) for x in got} == {
        (True, False, False), (False, True, False), (False, False, True)}
    assert all("no seat thickness" in x and "kit:Thing post" in x for x in got)
    assert run_gate([Bearing()], ctx, None).status == "partial"


def _ramp(x0: float, top: float) -> np.ndarray:
    """A ramp face at the host's edge x0, leaning 60 degrees from the host's top face and facing
    up: two triangles of the slope, wound to face up."""
    run, drop = 0.03, 0.03 * np.tan(np.radians(60))
    far = x0 + run, top - drop
    tris = np.array([[(x0, top, -0.05), (*far, -0.05), (x0, top, 0.05)],
                     [(x0, top, 0.05), (*far, -0.05), (*far, 0.05)]])
    return tris if np.cross(tris[0, 1] - tris[0, 0], tris[0, 2] - tris[0, 0])[1] > 0 \
        else tris[:, ::-1]


@pytest.mark.parametrize("guest", ["square block", "round pin"])
def test_a_part_sitting_flat_takes_its_plane_from_its_seat_not_from_a_slanted_host_face(guest):
    """A guest seated flush on a host's top face, with a slanted host face (a ramp at the host's
    edge) forced into the patch, as the slack can let one in. The seat corners lie in one plane, so
    that plane is the contact plane and the ramp does not tilt it."""
    host = box((-0.1, 0.0, -0.1), (0.1, 0.02, 0.1))
    mesh = {"square block": box((-0.03, 0.02, -0.03), (0.03, 0.03, 0.03)),
            "round pin": cylinder((0.0, 0.0), 0.03, 0.02, 0.03, sides=16)}[guest]

    def piece_of(plate):
        ctx = make_ctx({"host": plate, "guest": mesh}, {"guest": {"on": "host"}})
        piece = bearing._Piece(ctx, next(pieces.measured_pieces(ctx, {"assembly"})))
        return piece, piece.by_name["guest"], piece.by_name["host"]

    closed, part, plate = piece_of(host)  # the contact is read on the closed host alone
    seat_faces, _ = bearing._contact(closed, part, plate)
    ramped = np.concatenate([host, _ramp(0.09, 0.02)])  # then the ramp joins the host
    piece, part, plate = piece_of(ramped)
    piece.sinks.update(closed.sinks)  # its thickness, read on the closed host
    got = bearing._seat_plan(piece, part, plate, seat_faces)
    assert got.status == "ready"
    assert np.allclose(got.plan["normal"], [0.0, -1.0, 0.0], atol=1e-6)  # into the host, untilted
    assert bearing._seat(piece, part, got).status == "pass"


def _inside(p, a, b, c) -> bool:
    cross = [(q[0] - r[0]) * (p[1] - r[1]) - (q[1] - r[1]) * (p[0] - r[0])
             for q, r in ((b, a), (c, b), (a, c))]
    return all(x > 0 for x in cross)


def stepped_slab(xs, heights, z0=-0.1, z1=0.1) -> np.ndarray:
    """A closed slab on y = 0, 200 mm across (z0 to z1), whose top is a step at each x in
    `xs[1:-1]`: column i runs from xs[i] to xs[i + 1] at height heights[i]. It is the prism of its
    side profile (counter-clockwise, ear-clipped for the two caps), so every edge is shared."""
    prof = [(xs[0], 0.0), (xs[-1], 0.0)]
    for i in reversed(range(len(heights))):
        prof += [(xs[i + 1], heights[i]), (xs[i], heights[i])]
    pts = np.array(prof, dtype=float)
    out, n = [], len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        out += [[(*a, z0), (*b, z0), (*b, z1)], [(*a, z0), (*b, z1), (*a, z1)]]
    idx = list(range(n))
    while len(idx) > 3:
        for k in range(len(idx)):
            i, j, m = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = pts[i], pts[j], pts[m]
            turn = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if turn <= 0 or any(_inside(pts[t], a, b, c) for t in idx if t not in (i, j, m)):
                continue
            out += [[(*a, z1), (*b, z1), (*c, z1)], [(*a, z0), (*c, z0), (*b, z0)]]
            idx.pop(k)
            break
    a, b, c = (pts[i] for i in idx)
    out += [[(*a, z1), (*b, z1), (*c, z1)], [(*a, z0), (*c, z0), (*b, z0)]]
    return np.array(out, dtype=float)


def _pocket(side: float) -> np.ndarray:
    """A slab 60 mm high with a floor at 20 mm, 220 mm long between two end walls: the wall at
    x = 0.1 (`side` 1) or x = -0.1 (`side` -1) and the other 20 mm further out."""
    if side > 0:
        return stepped_slab([-0.2, -0.12, 0.1, 0.2], [0.06, 0.02, 0.06])
    return stepped_slab([-0.2, -0.1, 0.12, 0.2], [0.06, 0.02, 0.06])


@pytest.mark.parametrize("side", [1.0, -1.0])
def test_a_block_in_a_pocket_is_seated_on_the_floor_and_not_the_end_wall(side):
    """A block lies 0.5 mm over a pocket's floor (its large bearing face) with its end against one
    end wall. Its corners nearest the host are those of the end face: a plane at 90 degrees to the
    patch's mean normal, a side contact, which does not stand in for the seat. The seat is the
    floor, and the block does not overhang it."""
    lo, hi = sorted((0.1 * side, 0.1 * side - 0.2 * side))
    ctx = make_ctx({"pocket": _pocket(side), "block": box((lo, 0.0205, -0.03), (hi, 0.0405, 0.03))},
                   {"block": {"on": "pocket"}})
    view = next(pieces.measured_pieces(ctx, {"assembly"}))
    piece = bearing._Piece(ctx, view)
    part, host = piece.by_name["block"], piece.by_name["pocket"]
    seat_faces, _ = bearing._contact(piece, part, host)
    corners = bearing._seat_corners_plane(piece, part.tris[seat_faces], host, np.array([-side, 0, 0]))
    assert corners is not None  # the end face alone is flat: the rule would take it,
    assert np.allclose(np.abs(corners), [1.0, 0.0, 0.0])
    got = entry(ctx, "block", "seat")
    assert got.status == "pass"  # but it disagrees with the patch, which is the floor
    assert np.allclose(np.abs(got.plan["normal"]), [0.0, 1.0, 0.0], atol=0.05)


def test_a_seat_that_is_not_flat_leaves_the_plane_to_the_host_patch():
    """A tube lying on a plane touches along a line: its nearest corners do not span a plane."""
    tube = cylinder((0.0, 0.0), 0.01, 0.0, 0.2, sides=16)[..., [0, 2, 1]][:, ::-1]
    plate = box((-0.1, -0.01, -0.1), (0.1, 0.0, 0.3))
    ctx = make_ctx({"plate": plate, "tube": tube + np.array([0.0, 0.01, 0.0])},
                   {"tube": {"on": "plate"}})
    view = next(pieces.measured_pieces(ctx, {"assembly"}))
    piece = bearing._Piece(ctx, view)
    part, host = piece.by_name["tube"], piece.by_name["plate"]
    seat_faces, _ = bearing._contact(piece, part, host)
    down = np.array([0.0, -1.0, 0.0])
    assert bearing._seat_corners_plane(piece, part.tris[seat_faces], host, down) is None


def test_the_corner_plane_is_a_sound_test_of_flatness_and_of_a_line():
    """Three corners always span a plane, so three are not enough; four on a line span none; four
    in a plane do, and a fifth off it spoils it."""
    plate = box((-1.0, -0.01, -1.0), (1.0, 0.0, 1.0))
    ctx = make_ctx({"plate": plate, "p": box((0.0, 0.0, 0.0), (0.01, 0.01, 0.01))},
                   {"p": {"on": "plate"}})
    piece = bearing._Piece(ctx, next(pieces.measured_pieces(ctx, {"assembly"})))
    host, up = piece.by_name["plate"], np.array([0.0, 1.0, 0.0])

    def plane(*corners):
        tri = np.array([[corners[i], corners[(i + 1) % len(corners)], corners[0]]
                        for i in range(len(corners))], dtype=float)
        return bearing._seat_corners_plane(piece, tri, host, -up)

    flat = [(0, 0, 0), (0.01, 0, 0), (0, 0, 0.01), (0.01, 0, 0.01)]
    assert np.allclose(np.abs(plane(*flat)), up)
    assert plane(*flat[:3]) is None  # three always span a plane
    assert plane((0, 0, 0), (0.01, 0, 0), (0.02, 0, 0), (0.03, 0, 0)) is None  # on a line
    assert plane(*flat, (1.0, -0.005, 0.005)) is None  # a fifth, as near the host, off the plane


def test_a_rest_with_no_centre_of_mass_is_unmeasured(monkeypatch):
    """The rest rule's own `no centre of mass` path (a part whose weight cannot be found), which a
    part with no thickness no longer reaches."""
    def refuse(tris):
        raise ValueError("the centre of mass needs a closed mesh")

    monkeypatch.setattr(geom, "centre_of_mass", refuse)
    got = [x for x in lines(stack()) if "rest rule" in x and "skipped" in x]
    assert got and all("no centre of mass" in x for x in got)
    assert any(isinstance(x, Unmeasured) for x in Bearing().coverage(stack()))
    assert run_gate([Bearing()], stack(), None).status == "partial"


def test_a_shank_reads_at_its_circumradius_not_its_apothem():
    """A faceted round shank of N sides stands for a round bolt through its corners: the wall
    radius is the corner distance, `radius`, and not the face distance, `radius cos(pi/N)`."""
    radius, sides = 0.005, 8
    shank = cylinder((0.0, 0.0), radius, 0.0, 0.1, sides=sides)
    got = bearing._wall_radius(shank, np.array([0.0, 0.05, 0.0]), np.array([0.0, 1.0, 0.0]))
    assert got == pytest.approx(radius, rel=1e-9)
    assert got > radius * np.cos(np.pi / sides) * 1.05


def test_hull_and_depth():
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.5, 0.0]], dtype=float)
    hull = hull_2d(corners)
    assert len(hull) == 4
    assert depth_in(hull, np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert depth_in(hull, np.array([1.2, 0.5])) == pytest.approx(-0.2)
    assert depth_in(hull_2d(np.array([[0, 0], [1, 1], [2, 2]], dtype=float)),
                    np.array([1.0, 1.0])) == -np.inf  # a line holds nothing


# --- rule 2: fastener engagement ----------------------------------------------------------------


def test_fastener_shank_not_head():
    """The head is 30 mm wide but the shank 10: engaged 12 mm, the bolt holds. Had the head set
    the diameter, 12 mm would be short of 30."""
    ctx = bolted_post(sunk=0.012)
    assert findings(ctx) == []
    plan = entry(ctx, "bolt", "fastener").plan
    assert plan["need"] == pytest.approx(ENGAGE_OF_DIAMETER * 0.01, rel=1e-6)
    assert plan["engaged"] == pytest.approx(0.012, abs=0.037 / bearing.ENGAGE_SAMPLES)
    assert np.allclose(np.abs(plan["axis"]), [0, 1, 0])


def test_fastener_in_air_fails():
    ctx = bolted_post(sunk=0.006)  # 6 mm in, and 10 mm wide
    got = items(ctx)
    (item,) = got
    assert re.fullmatch(rf"{THING} bolt engages post [56]\.\d mm", item)  # read to 0.5 mm
    assert "shank diameter is 10.00 mm, so it needs 10.00 mm" in got[item]
    # lifted clear of the post: no engagement at all
    away = items(shifted(ctx, "bolt", (0.0, 0.2, 0.0)))
    assert any(i.startswith(f"{THING} bolt engages post 0.0 mm") for i in away)


def test_a_bolt_sunk_one_diameter_holds_and_half_a_diameter_does_not():
    assert findings(bolted_post(sunk=0.0105)) == []
    assert findings(bolted_post(sunk=0.0095)) != []


def test_a_head_only_disc_is_read_along_its_short_axis():
    disc = box((-0.02, 0.4, -0.02), (0.02, 0.405, 0.02))  # 40 mm across, 5 mm high, on the post
    parts = {"post": box((-0.1, 0.0, -0.1), (0.1, 0.4, 0.1)), "screw": disc}
    ctx = make_ctx(parts, {"screw": {"on": "post", "fastener": True}}, hosts=["post"])
    (item,) = items(ctx)
    assert item.startswith(f"{THING} screw engages post 0.0 mm")


def holed_plank(xs, zs):
    """A plank 50 mm thick with a through hole in the middle cell of the grid `xs` x `zs`: a
    closed mesh, as the top and bottom faces of the eight cells round the hole and the walls
    between a cell and the outside or the hole. The bearing check reads a host's thickness, which
    an open mesh does not have."""
    cells = {(i, j) for i in range(3) for j in range(3)} - {(1, 1)}
    out = []

    def quad(a, b, c, d, away):
        for tri in ((a, b, c), (a, c, d)):
            t = np.array(tri, dtype=float)
            out.append(t[::-1] if np.cross(t[1] - t[0], t[2] - t[0]) @ away < 0 else t)

    for i, j in cells:
        x0, x1, z0, z1 = xs[i], xs[i + 1], zs[j], zs[j + 1]
        quad((x0, 0.05, z0), (x1, 0.05, z0), (x1, 0.05, z1), (x0, 0.05, z1), np.array([0, 1.0, 0]))
        quad((x0, 0.0, z0), (x1, 0.0, z0), (x1, 0.0, z1), (x0, 0.0, z1), np.array([0, -1.0, 0]))
        for di, dj, a, b in ((-1, 0, (x0, z0), (x0, z1)), (1, 0, (x1, z0), (x1, z1)),
                             (0, -1, (x0, z0), (x1, z0)), (0, 1, (x0, z1), (x1, z1))):
            if (i + di, j + dj) not in cells:
                quad((a[0], 0.0, a[1]), (b[0], 0.0, b[1]), (b[0], 0.05, b[1]),
                     (a[0], 0.05, a[1]), np.array([di, 0.0, dj]))
    return np.array(out)


def bored_plank():
    """A plank with a 20 mm square bore through it."""
    return holed_plank([-0.2, -0.01, 0.01, 0.2], [-0.2, -0.01, 0.01, 0.2])


def slotted_plank():
    """A plank with a 200 mm by 40 mm slot through it."""
    return holed_plank([-0.2, -0.1, 0.1, 0.2], [-0.2, -0.02, 0.02, 0.2])


def test_a_shank_in_a_bore_the_host_closes_round_it_is_engaged():
    """`geom.inside` reads the bore as empty; the rays that leave the axis meet the bore wall."""
    shaft = box((-0.009, 0.0, -0.009), (0.009, 0.1, 0.009))  # 18 mm square, 50 mm in the bore
    ctx = make_ctx({"plank": bored_plank(), "bolt": shaft},
                   {"bolt": {"on": "plank", "fastener": True}}, hosts=["plank"])
    assert not geom.inside(bored_plank(), np.array([[0.0, 0.025, 0.0]]))[0]
    got = entry(ctx, "bolt", "fastener")
    assert got.status == "pass" and got.plan["engaged"] == pytest.approx(0.05, abs=0.002)
    # pulled up until 10 mm are left in the bore, short of its own 18 mm
    out = shifted(ctx, "bolt", (0.0, 0.04, 0.0))
    assert [f.item for f in findings(out) if "engages" in f.item] == [
        f"{THING} bolt engages plank 9.4 mm"]  # 10 mm, read to 0.6 mm


def test_a_fastener_with_no_host_is_unmeasured():
    parts = {"post": box((-0.1, 0.0, -0.1), (0.1, 0.4, 0.1)), "bolt": bolt(sunk=0.02)}
    ctx = make_ctx(parts, {"bolt": {"fastener": True}}, hosts=["post", "bolt"])
    assert findings(ctx) == []
    assert any(isinstance(x, Unmeasured) and "fastener with no host" in x and "bolt" in x
               for x in Bearing().coverage(ctx))
    assert run_gate([Bearing()], ctx, None).status == "partial"


def test_a_bolt_loose_in_a_cavity_is_not_engaged_by_far_walls():
    """A 10 mm bolt floating 50 mm over the floor of a cup, 145 mm from its walls. Rays that
    ran out to the walls read it as engaged 100 mm; a ray counts only within the wall radius plus
    the touch distance of the axis."""
    cup = np.concatenate([
        box((-0.15, 0.0, -0.15), (0.15, 0.01, 0.15)),  # floor
        box((-0.15, 0.01, -0.15), (-0.14, 0.2, 0.15)), box((0.14, 0.01, -0.15), (0.15, 0.2, 0.15)),
        box((-0.14, 0.01, -0.15), (0.14, 0.2, -0.14)), box((-0.14, 0.01, 0.14), (0.14, 0.2, 0.15))])
    shaft = box((-0.005, 0.06, -0.005), (0.005, 0.16, 0.005))
    ctx = make_ctx({"cup": cup, "bolt": shaft}, {"bolt": {"on": "cup", "fastener": True}},
                   hosts=["cup"])
    (item,) = [i for i in items(ctx) if "engages" in i]
    assert item.startswith(f"{THING} bolt engages cup 0.0 mm")


# --- rule 3: full-face seating ------------------------------------------------------------------


def seated(overhang: float):
    """A bracket and a plate on it, the plate sticking out past the bracket's end by `overhang`."""
    parts = {"bracket": box((-0.2, 0.0, -0.2), (0.2, 0.1, 0.2)),
             "plate": box((0.2 - 0.3 + overhang, 0.1, -0.1), (0.2 + overhang, 0.12, 0.1))}
    return make_ctx(parts, {"plate": {"on": "bracket"}})


def test_a_plate_wholly_over_its_host_is_seated():
    assert findings(seated(0.0)) == []


def test_saddle_overhang_fails_rule_3():
    """2.8 mm is the overhang of a guitar's saddle, and it is under the touch distance: the rule measures
    area past the edge, so only float noise is let go."""
    got = items(seated(0.0028))
    assert list(got) == [f"{THING} plate overhangs bracket"]
    assert mm_in(got[f"{THING} plate overhangs bracket"], "reaches") == pytest.approx(2.8, abs=0.01)


def test_the_overhang_tolerance_is_float_noise_and_no_more():
    ctx = seated(0.0)
    assert findings(shifted(ctx, "plate", (noise_of(ctx) / 2, 0, 0))) == []
    assert findings(shifted(ctx, "plate", (noise_of(ctx) * 4, 0, 0))) != []


def test_only_the_seat_in_contact_is_counted():
    """A metre-long bar sheared up off its block touches it at the low end only. Counting
    the whole triangles of its under face would read the climb past the block's edge as an
    overhang; reading the seat only where it lies at the level of the block's top (to the contact
    length) finds none."""
    assert not [i for i in items(wedge_on_block()) if "overhangs" in i]


def test_a_cable_spanning_its_host_is_a_finding_of_the_size_in_contact():
    """A long thin part laid across a short host overhangs it at both ends. The size reported is
    what lies within touch, never the length of the triangles that happen to span it."""
    parts = {"nut": box((-0.01, 0.0, -0.05), (0.01, 0.03, 0.05)),
             "string": box((-0.5, 0.03, -0.001), (0.5, 0.032, 0.001))}
    got = items(make_ctx(parts, {"string": {"on": "nut", "shape": "cable"}}))
    assert mm_in(got[f"{THING} string overhangs nut"], "reaches") < 2 * TOUCH * 1000


def test_a_head_over_its_own_hole_is_seated():
    """The host's outline holds its holes. A washer over a bore, and a plate across a slot,
    are wholly over the host."""
    plank = bored_plank()
    washer = box((-0.03, 0.05, -0.03), (0.03, 0.06, 0.03))
    assert not [i for i in items(make_ctx({"plank": plank, "washer": washer},
                                          {"washer": {"on": "plank"}})) if "overhangs" in i]
    slotted = slotted_plank()  # a 200 mm slot
    bridge = box((-0.15, 0.05, -0.1), (0.15, 0.06, 0.1))
    slot_ctx = make_ctx({"plank": slotted, "plate": bridge}, {"plate": {"on": "plank"}})
    assert not [i for i in items(slot_ctx) if "overhangs" in i]
    assert entry(slot_ctx, "plate", "seat").status == "pass"  # judged, not skipped as unmeasured
    # a washer that does hang past the plank's edge still overhangs
    edge = box((0.15, 0.05, -0.03), (0.25, 0.06, 0.03))
    assert f"{THING} washer overhangs plank" in items(
        make_ctx({"plank": plank, "washer": edge}, {"washer": {"on": "plank"}}))


def test_a_tabletop_is_not_an_overhang_when_the_legs_are_on_it():
    """Reversed, with the top `on` one leg, it overhangs that leg's face and tips."""
    parts = {"top": box((-0.5, 0.7, -0.3), (0.5, 0.75, 0.3)),
             "leg": box((0.425, 0.0, 0.225), (0.475, 0.7, 0.275))}
    got = items(make_ctx(parts, {"top": {"on": "leg"}}))
    assert f"{THING} top overhangs leg" in got and f"{THING} top tips off leg" in got
    assert findings(make_ctx(parts, {"leg": {"on": "top"}})) == []


# --- rule 4: a retainer touches -----------------------------------------------------------------


def test_a_retainer_that_touches_what_it_retains_passes():
    assert findings(held_bar()) == []


def test_floating_retainer_fails_rule_4():
    ctx = held_bar(lift=CALIBRATION_FAIL * TOUCH)
    got = items(ctx)
    assert f"{THING} clip off post" in got
    assert "from post, which it retains" in got[f"{THING} clip off post"]


def test_retainer_contact_is_float_noise_and_no_more():
    """Real contact. A gap under float noise touches; half a touch
    distance, which a looser rule would let go, does not."""
    noise = noise_of(held_bar())
    assert findings(held_bar(lift=noise / 2)) == []
    assert findings(held_bar(lift=noise * 4)) != []
    assert f"{THING} clip off post" in items(held_bar(lift=0.5 * TOUCH))


def test_a_rigged_piece_is_judged_per_part():
    """Rule 1 does not apply on a rigged piece: a joint carries each part, not gravity. Rule 3
    does not apply to a part whose bones differ from its host's; with the same bones it runs."""
    ctx = rigged(stack(x=0.35), {"slab": ("a",), "block": ("b",)})
    got = findings(ctx)
    assert not [f for f in got if "tips off" in f.item or "overhangs" in f.item]
    text = lines(ctx)
    assert any("its piece is rigged" in x and "rest rule" in x and "block" in x for x in text)
    assert any("its bones differ from its host's" in x and "seat rule" in x and "block" in x
               for x in text)
    same = rigged(stack(x=0.35), {"slab": ("a",), "block": ("a",)})
    assert f"{THING} block overhangs slab" in items(same)
    assert not [i for i in items(same) if "tips off" in i]


def test_a_part_of_a_bonded_row_is_not_judged_by_gravity():
    """A glued joint holds by the bond. Rule 1 prints a line naming
    it; the other rules still run."""
    ctx = stack(x=0.35)
    assert f"{THING} block tips off slab" in items(ctx)
    glued = bond(ctx)
    got = items(glued)
    assert not [i for i in got if "tips off" in i]
    assert f"{THING} block overhangs slab" in got
    assert any("held bonded" in x and "rest rule" in x and "block" in x for x in lines(glued))


def test_a_retained_part_with_no_geometry_raises():
    ctx = held_bar()
    key = piece_key("kit", "Thing")
    gone = {k: v for k, v in ctx.pieces[key].parts.items() if k != "post"}
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom("Thing", gone)})
    with pytest.raises(ManifestError, match="no geometry"):
        findings(ctx)


# --- scope, coverage and plants -----------------------------------------------------------------


def test_a_scoped_run_measures_only_the_named_part():
    ctx = declare(stack(x=0.35), "kit", "Thing", fasteners=())
    got = findings(ctx, only={f"{THING} block tips off slab"})
    assert [f.item for f in got] == [f"{THING} block tips off slab"]
    assert findings(ctx, only={f"{THING} slab tips off nothing"}) == []
    assert findings(ctx, only={"kit:Other block tips off slab"}) == []


def test_every_rule_that_does_not_apply_prints_a_line():
    got = lines(stack())
    assert any("no part is a fastener" in x for x in got)
    assert any("no part retains another" in x for x in got)
    assert not any(isinstance(x, Unmeasured) for x in Bearing().coverage(stack()))


def test_printed_parts_are_left_out():
    ctx = viewed(humanoid_scene(), TOUCH)
    assert any(x == "bearing: not applicable (decal part, printed on a surface): humanoid:Panel mark" for x in lines(ctx))


def first_caught(unit):
    """The candidate the gate would accept: the first the check catches."""
    return next((p for p in unit.candidates
                 if any(f.item == p.item for f in Bearing().run(p.ctx, only=p.scope))), None)


def units_of(ctx, rule):
    return [u for u in Bearing().plants(ctx)
            if u.candidates and u.candidates[0].label.startswith(f"{rule}:")]


def test_plants_caught_on_two_pieces():
    """Each rule's plant is made and caught on at least two different pieces, and the gate does
    not stop on any of them."""
    scenes = {
        "rest": [stack(), hanging_clamp(), tabletop_on_legs()],
        "fastener": [bolted_post(0.03), viewed(humanoid_scene(), TOUCH / 4)],
        "seat": [seated(0.0), stack()],
        "retainer": [held_bar(), viewed(humanoid_scene(), TOUCH / 4)],
        "sink": [pushed_post(piece, 0.0) for piece in PUSHED_PIECES],
    }
    for rule, contexts in scenes.items():
        caught = 0
        for ctx in contexts:
            mine = units_of(ctx, rule)
            for unit in mine:
                assert first_caught(unit) is not None, unit.item
            caught += bool(mine)
            run_gate([Bearing()], ctx, None)  # no BlindCheck
        assert caught >= 2, rule


def test_a_plant_is_a_move_of_one_part_and_the_real_run_is_clean():
    ctx = stack()
    check = Bearing()
    units = {u.candidates[0].label.split(":")[0]: u for u in check.plants(ctx)}
    assert set(units) == {"rest", "seat", "sink"}
    assert units["sink"].item == f"{THING} block sinks into slab 42.6 mm"
    assert len(units["sink"].candidates) == 1
    assert units["rest"].item == f"{THING} block tips off slab"
    assert units["seat"].item == f"{THING} block overhangs slab"
    assert len(units["rest"].candidates) == 4
    assert len(units["seat"].candidates) == 12  # four ways, at three slide lengths
    moved = units["rest"].candidates[0].ctx.pieces[piece_key("kit", "Thing")].parts
    assert np.array_equal(moved["slab"], ctx.pieces[piece_key("kit", "Thing")].parts["slab"])
    assert not np.array_equal(moved["block"], ctx.pieces[piece_key("kit", "Thing")].parts["block"])
    assert check.run(ctx) == []


def test_a_fastener_plant_pulls_the_bolt_clear_so_it_holds_nothing():
    ctx = bolted_post(0.03)
    (unit,) = units_of(ctx, "fastener")
    assert unit.item == f"{THING} bolt engages post 0.0 mm"  # the item names the engagement
    assert len(unit.candidates) == 2  # clear of the host along the axis, one way and the other
    assert first_caught(unit) is not None
    assert items(ctx) == {}  # as built, the bolt holds


def test_a_plant_slides_a_narrow_part_only_half_its_own_extent_past_the_edge():
    # two touches past the edge would take a part narrower than four touches clear of its host,
    # and the rule would see no contact; the slide stops at half the part's extent instead
    assert bearing._margin(1.0, 10.0) == CALIBRATION_FAIL * 1.0
    assert bearing._margin(1.0, 1.0) == 0.5


def test_a_fasteners_seat_is_planted_with_a_short_slide_and_caught():
    # the bolt's shank crosses the post: the slides take its head over the edge and the overhang
    # is read
    ctx = bolted_post(0.03)
    (unit,) = units_of(ctx, "seat")
    assert unit.item == f"{THING} bolt overhangs post" and unit.not_applicable == ""
    caught = [i for i, p in enumerate(unit.candidates)
              if any(f.item == p.item for f in Bearing().run(p.ctx, only=p.scope))]
    assert caught  # with the seat's direction read from the bolt's own faces the longest slides too
    assert run_gate([Bearing()], ctx, None).status == "passed"
    assert not [x for x in Bearing().coverage(ctx) if "not applicable (only a fastener" in x]


def test_the_chairs_seat_rule_is_planted_and_caught_though_no_old_filter_chose_it():
    """The test bodies' Chair has a seat rule that applies. A plant that chooses a part by
    running the check finds no move that fails (`partial`). Planted by geometry alone, the
    back's slide off the seat is caught."""
    ctx = viewed(humanoid_scene(), TOUCH)
    chair = [u for u in Bearing().plants(ctx) if u.item.startswith("humanoid:Chair ")]
    assert sorted(u.item for u in chair) == ["humanoid:Chair back overhangs seat",
                                             "humanoid:Chair back tips off seat"]
    assert all(first_caught(u) is not None for u in chair)
    assert not any("no plant can fail" in x for x in lines(ctx))


def test_bearing_plants_never_call_the_core(monkeypatch):
    ctx = stack()
    refuses_the_core(monkeypatch, bearing, "_rest", "_engage", "_seat", "_retain", "_sinks",
                     "measure")
    monkeypatch.setattr(bearing, "_JUDGE", {})  # the verdicts are never looked up
    monkeypatch.setattr(bearing.Bearing, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in Bearing().plants(ctx)) == [
        f"{THING} block overhangs slab", f"{THING} block sinks into slab 42.6 mm",
        f"{THING} block tips off slab"]


def test_the_parts_it_measured_are_those_a_rule_judged():
    """A part whose every rule is not applicable is not measured; a failing one is."""
    got = Bearing().measured(stack(x=0.35))
    assert got == {f"{THING} block"}  # the slab is the host and has no rule of its own
    assert Bearing().measured(sideways()) == {f"{THING} shelf"}  # rule 3 judges it
    joined = rigged(stack(), {"slab": ("a",), "block": ("b",)})  # rules 1 and 3 do not apply
    assert Bearing().measured(joined) == set()


def test_bearing_is_not_a_body_check():
    from .test_body_plans import NOT_BODY
    assert "bearing" in NOT_BODY


# --- the Strat before its neck-plate fix --------------------------------------------------------

STRAT_FASTENERS = (r"NeckBolt", r"GuardScrew", r"SaddleScrew", r"ClawScrew", r"TremCoverScrew",
                   r"StrapButton(Lower|Upper)Shaft$")
STRAT_RETAINS = {"TreeWing": ("String_5", "String_6")}


STRAT_TOUCH = 0.0016  # the contact length of the scene's pieces


@functools.cache
def strat_before():
    """The scene of a guitar neck-plate regression case. It has no `fastener` or `retains`
    fields, so they are declared here in memory. It has no style crease angle either, so the
    build's (35 degrees) is set in memory (`creased`)."""
    ctx = creased(load_scene(REGRESSIONS / "strat_before_neck_plate"))
    ctx = declare(ctx, "strat", "Strat", fasteners=STRAT_FASTENERS, retains=STRAT_RETAINS)
    return viewed(ctx, STRAT_TOUCH)


@functools.cache
def strat_found():
    return {f.item: f.detail for f in Bearing().run(strat_before())}


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_strat_before_fix_fails_rules_2_and_3():
    """Rule 1 does not fire on the plate: in the Strat's frame (y up, the body's face toward +z)
    the plate sits on the body's back, a sideways patch. Rule 3 finds the plate hanging past the
    body's outline and rule 2 finds the bolts: heads that stand on the plate and sink nowhere."""
    got = strat_found()
    plate = got["strat:Strat NeckPlate overhangs Body"]
    # the seat is read at the level of the body's back it bears on, over that surface's extent
    # padded by the contact length (3.2 mm), so it reads up to the 29.12 mm the fix commit
    # measured to the plate's farthest vertex (28.87 mm), and never more.
    assert 2.0 < mm_in(plate, "reaches") < 29.2
    for name in ("NeckBolt", "NeckBolt_001", "NeckBolt_002", "NeckBolt_003"):
        (item,) = [i for i in got if i.startswith(f"strat:Strat {name} engages NeckPlate")]
        assert "inside NeckPlate" in got[item]
    assert not any(i.startswith("strat:Strat NeckPlate tips") or "NeckPlate swings" in i
                   for i in got)


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_strat_string_tree_is_off_its_strings():
    """Retainer contact is float noise. The tree wing stands 0.3 to
    0.5 mm off strings 5 and 6, which a half-touch rule would let go."""
    got = strat_found()
    for n, mm in ((5, 0.51), (6, 0.32)):
        detail = got[f"strat:Strat TreeWing off String_{n}"]
        assert float(detail.split()[0]) == pytest.approx(mm, abs=0.02)
        assert "which it retains" in detail


def strat_with_bolt(*, out: float = 0.0):
    """The Strat with NeckBolt given a shank and a host that does not move with it: the body.
    The head is the part's own; the shank is 3.3 mm across and goes 8 mm into the body. `out`
    pulls the whole bolt that many mm out of the body, along its axis (-z)."""
    ctx = strat_before()
    key = piece_key("strat", "Strat")
    mm = 1e-3 * ctx.packs["strat"].units_per_metre  # a mm in the pack's units
    head = ctx.pieces[key].parts["NeckBolt"]
    lo, hi = head.reshape(-1, 3).min(axis=0), head.reshape(-1, 3).max(axis=0)
    body_back = float(ctx.pieces[key].parts["Body"].reshape(-1, 3)[:, 2].min())
    half = 1.65 * mm
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    # from the head's underside (its high z) to 8 mm into the body
    shank = box((cx - half, cy - half, hi[2]), (cx + half, cy + half, body_back + 8 * mm))
    bolt_tris = np.concatenate([head, shank]) - np.array([0.0, 0.0, out * mm])
    parts = {**ctx.pieces[key].parts, "NeckBolt": bolt_tris}
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom("Strat", parts)})
    return declare(ctx, "strat", "Strat", fasteners=STRAT_FASTENERS, retains=STRAT_RETAINS,
                   hosts={"NeckBolt": "Body"})


def neck_bolt_entry(ctx):
    view = next(pieces.measured_pieces(ctx, {"assembly"}))
    return next(e for e in bearing.measure(ctx, view, {"NeckBolt"}) if e.rule == "fastener")


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_strat_bolt_into_the_body_holds_and_pulled_out_does_not():
    """The bolt's host is the body, which does not move with it. With a 3.3 mm shank 8 mm into
    the body it holds; pulled 17.89 mm out (the original defect) it is in the air."""
    held = neck_bolt_entry(strat_with_bolt())
    assert held.status == "pass" and held.plan["engaged"] * 2000 == pytest.approx(8.0, abs=0.7)
    air = neck_bolt_entry(strat_with_bolt(out=17.89))
    assert air.status == "fail" and air.finding.item == "strat:Strat NeckBolt engages Body 0.0 mm"


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_strat_head_only_bolts_have_no_shank_to_engage():
    """As the model stands the bolts are heads: on the body they read 0.0 mm. That is the model's
    data, not the check's."""
    ctx = declare(strat_before(), "strat", "Strat", fasteners=STRAT_FASTENERS,
                  hosts={"NeckBolt": "Body"})
    got = neck_bolt_entry(ctx)
    assert got.status == "fail" and got.finding.item == "strat:Strat NeckBolt engages Body 0.0 mm"


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_strat_saddles_overhang_the_bridge_plate_at_both_ends():
    got = strat_found()
    ends = {n: mm_in(got[f"strat:Strat Saddle_{n} overhangs BridgePlate"], "reaches")
            for n in (1, 6)}
    assert all(2.0 < mm < 3.0 for mm in ends.values()), ends  # the fix commit: 2.8 mm each
    assert not [i for i in got if re.match(r"strat:Strat Saddle_[2-5] overhangs", i)]


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_strat_plants_are_caught():
    ctx = strat_before()
    check = Bearing()
    units = check.plants(ctx)
    assert {u.candidates[0].label.split(":")[0] for u in units} == {
        "seat", "fastener", "retainer", "sink"}
    assert plants_caught(check, ctx)
    run_gate([check], ctx, None)  # no BlindCheck
    got = [str(x) for x in check.coverage(ctx)]
    assert not [x for x in got if "no plant can fail" in x]


# --- the contact length is the piece's own ----------------------------------------------------


def test_a_piece_with_no_view_has_no_contact_length_so_every_rule_is_unmeasured():
    ctx = unviewed(stack(x=0.35))  # its centre is past the slab edge: a length would find it
    assert findings(stack(x=0.35)) != []
    assert findings(ctx) == []
    assert Bearing().plants(ctx) == [] and Bearing().measured(ctx) == set()
    (line,) = [x for x in Bearing().coverage(ctx) if "no view" in x]
    assert isinstance(line, Unmeasured) and line.endswith(f": {THING}")
    assert run_gate([Bearing()], ctx, None).status == "partial"


def test_the_touch_follows_the_pieces_view():
    ctx = shifted(stack(), "block", (0.0, 2 * TOUCH, 0.0))  # floating 2 touches over its slab
    near = entry(viewed(ctx, TOUCH), "block", "rest")
    far = entry(viewed(ctx, 5 * TOUCH), "block", "rest")
    assert (near.status, far.status) != ("pass", "pass") and far.status == "pass", (near, far)


# --- rule 5: a resting part sits on its host's surface ------------------------------------------


@pytest.mark.parametrize("piece", PUSHED_PIECES)
def test_part_pushed_in_by_twice_contact_length_sinks_on_two_pieces(piece):
    """A guard: the post is pushed twice the contact length into its base, which is under half the
    thinner part (so crossings stays quiet). The push is plain geometry, not a verdict."""
    by = CALIBRATION_FAIL * TOUCH
    assert findings(pushed_post(piece, 0.0)) == []  # flush: it sits on the surface
    ctx = pushed_post(piece, by)
    got = items(ctx)
    item = f"{THING} post sinks into base {by * 1000:.1f} mm"
    assert item in got and "inside base" in got[item]
    assert Crossings().run(ctx) == []
    assert mm_in(got[item], "sits") == pytest.approx(by * 1000, abs=0.05)


@pytest.mark.parametrize("piece", PUSHED_PIECES)
def test_a_part_sunk_less_than_the_contact_length_rests_on_the_surface(piece):
    assert findings(pushed_post(piece, 0.5 * TOUCH)) == []
    assert entry(pushed_post(piece, 0.5 * TOUCH), "post", "sink").status == "pass"


@pytest.mark.parametrize("piece", PUSHED_PIECES)
def test_a_part_lifted_off_its_host_does_not_sink(piece):
    ctx = pushed_post(piece, -0.5 * TOUCH)  # lifted by half the contact length
    assert not [i for i in items(ctx) if "sinks into" in i]


@pytest.mark.parametrize("piece", PUSHED_PIECES)
def test_bonded_row_is_not_judged(piece):
    ctx = bond(pushed_post(piece, CALIBRATION_FAIL * TOUCH))
    assert not [i for i in items(ctx) if "sinks into" in i]
    got = entry(ctx, "post", "sink")
    assert got.status == "na" and "held bonded" in got.note


def test_a_fastener_is_not_judged_by_the_sink_rule():
    """A fastener sits in its host along its axis: rule 2 reads that, rule 5 does not."""
    ctx = bolted_post(0.03)
    assert not [e for e in bearing.measure(
        ctx, next(pieces.measured_pieces(ctx, {"assembly"}))) if e.rule == "sink"]


def test_allowlisted_embed_is_allowed(tmp_path):
    ctx = pushed_post("square post on a thick base", CALIBRATION_FAIL * TOUCH)
    item = next(iter(items(ctx)))
    path = tmp_path / "a.allow"
    path.write_text(f"bearing | {item} | the post is pressed into its base on purpose\n")
    verdict = run_gate([Bearing()], ctx, Allowlist.load(path))
    assert verdict.status == "passed"
    assert "pressed into its base on purpose" in verdict.allowed[0]


@pytest.mark.parametrize("piece", PUSHED_PIECES)
def test_the_sink_rule_has_a_plant_and_it_is_caught(piece):
    ctx = pushed_post(piece, 0.0)
    units = [u for u in Bearing().plants(ctx) if "sinks into" in u.item]
    assert len(units) == 1 and not units[0].not_applicable
    assert f"{THING} post sinks into base {CALIBRATION_FAIL * TOUCH * 1000:.1f} mm" == units[0].item
    assert plants_caught(Bearing(), ctx)


@pytest.mark.parametrize("piece", PUSHED_PIECES)
def test_a_part_that_rests_with_a_small_gap_is_still_planted_and_caught(piece):
    """The push is the seat's own gap plus the depth: a seat 0.3 or 0.6 contact lengths off the
    surface still ends at the depth the item names, so the gate does not stop on it."""
    for lift in (0.3, 0.6):
        ctx = pushed_post(piece, -lift * TOUCH)
        (unit,) = [u for u in Bearing().plants(ctx) if "sinks into" in u.item]
        assert unit.candidates and not unit.not_applicable
        assert first_caught(unit) is not None, (lift, unit.item)
        assert run_gate([Bearing()], ctx, None).status == "passed"


def test_a_pair_whose_depth_band_is_empty_is_not_judged_by_the_sink_rule():
    """Half the thinner part's thickness (here 5 mm) is under the contact length (21 mm): no depth
    lies between a seat and crossings' limit, so the rule has nothing to judge and no unit."""
    parts = {"slab": box((-0.3, 0.0, -0.3), (0.3, 0.01, 0.3)),
             "block": box((-0.1, 0.01, -0.1), (0.1, 0.4, 0.1))}
    ctx = make_ctx(parts, {"block": {"on": "slab"}})
    got = entry(ctx, "block", "sink")
    assert got.status == "na" and "not over the contact length" in got.note
    assert not [u for u in Bearing().plants(ctx) if "sinks into" in u.item]


@pytest.mark.parametrize("host", [box((-0.05, 0.0, -0.3), (0.05, 0.5, 0.3)),
                                  box((-0.04, 0.0, -0.2), (0.04, 0.4, 0.2))],
                         ids=["thick-edged host", "short-edged host"])
def test_the_seat_direction_is_the_parts_own_and_a_host_edge_does_not_tilt_it(host):
    """A part hangs over a host's long side face. That face is in reach of the part's side and
    outweighs the top in the host patch, which tilted the seat 68 degrees (a sink reading).
    Each contact point counts the host face it meets only where it meets it (Baraff's contact
    normal), so the top gives the direction the seat lies in."""
    top = float(host[:, :, 1].max())
    part = box((0.0, top, -0.1), (0.15, top + 0.1, 0.1))
    ctx = make_ctx({"host": host, "part": part}, {"part": {"on": "host"}})
    for rule in ("seat", "sink"):
        got = entry(ctx, "part", rule)
        assert got.status != "na"
        assert np.allclose(got.plan["normal"], [0.0, -1.0, 0.0], atol=0.02)
    assert f"{THING} part overhangs host" in items(ctx)


@pytest.mark.parametrize("width", [0.6, 0.9])
def test_a_seat_wider_than_its_host_on_every_side_is_found(width):
    """The seat is cut to the host's box padded by the contact length, so its boundary lies a
    contact length past the host on every side, at the host's top level: it is read, and it
    overhangs."""
    post = box((-0.05, 0.0, -0.05), (0.05, 0.4, 0.05))
    plate = box((-width / 2, 0.4, -width / 2), (width / 2, 0.5, width / 2))
    ctx = make_ctx({"post": post, "plate": plate}, {"plate": {"on": "post"}})
    assert f"{THING} plate overhangs post" in items(ctx)


def _sink_unit(ctx):
    (unit,) = [u for u in Bearing().plants(ctx) if "sinks into" in u.item]
    return unit


@pytest.mark.parametrize("piece", TALL_PIECES)
def test_a_part_taller_than_it_is_wide_is_as_deep_as_its_foot_and_planted(piece):
    """The part's side faces outweigh its foot. The depth is where it has gone into its host,
    not along the side faces' direction, and the contact direction is the foot's."""
    by = CALIBRATION_FAIL * TOUCH
    assert findings(tall_post(piece, 0.0)) == []
    ctx = tall_post(piece, by)
    got = entry(ctx, "post", "sink")
    assert got.status == "fail" and got.plan["depth"] == pytest.approx(by, abs=1e-6)
    assert np.allclose(got.plan["normal"], [0.0, -1.0, 0.0], atol=0.02)
    assert Crossings().run(ctx) == []
    unit = _sink_unit(tall_post(piece, 0.0))
    assert unit.item == f"{THING} post sinks into base {by * 1000:.1f} mm"
    assert first_caught(unit) is not None


@pytest.mark.parametrize("piece", STRADDLE_PIECES)
def test_a_part_straddling_its_hosts_edge_is_as_deep_as_it_lies_under_the_top(piece):
    """As a fret hangs over a fingerboard's end: pushed `by` into the top, it is `by` deep, the
    shortest way out, and not the length it runs over the top along the slab."""
    by = CALIBRATION_FAIL * TOUCH
    ctx = straddling(piece, by)
    got = entry(ctx, "part", "sink")
    assert got.status == "fail" and got.plan["depth"] == pytest.approx(by, abs=1e-6)
    assert f"{THING} part sinks into slab {by * 1000:.1f} mm" in items(ctx)
    unit = _sink_unit(straddling(piece, 0.0))
    assert unit.item == f"{THING} part sinks into slab {by * 1000:.1f} mm"
    assert first_caught(unit) is not None


@pytest.mark.parametrize("host", [box((-0.3, 0.0, -0.3), (0.3, 0.4, 0.3)),
                                  cylinder((0.0, 0.0), 0.3, 0.0, 0.4, sides=24)],
                         ids=["box host", "round host"])
def test_a_rod_set_into_its_host_past_crossings_limit_is_never_a_pass(host):
    """A rod's end set 100 mm into its host, past half its 50 mm thickness: rule 5 may leave the
    pair to crossings, but it never reads it as a pass (TremArm_001 in its elbow)."""
    rod = box((-0.025, 0.3, -0.025), (0.025, 0.9, 0.025))
    ctx = make_ctx({"host": host, "rod": rod}, {"rod": {"on": "host"}})
    got = entry(ctx, "rod", "sink")
    assert got.status != "pass"
    assert got.status == "fail" and got.plan["depth"] == pytest.approx(0.1, abs=0.005)


@pytest.mark.parametrize(("radius", "sides"), [(1.0, 24), (1.5, 36)])
def test_a_flat_end_against_a_faceted_round_host_bears_on_the_round(radius, sides):
    """A cable meeting a lamp base: a flat end centred on the crease of two facets of a lathed host. Each
    facet's far corner lies further than the contact length off the end's plane, but the facets
    turn under the crease angle: one round surface, and the end bears on it."""
    ctx = against_a_faceted_round(radius, sides)
    got = entry(ctx, "block", "seat")
    assert got.status == "pass", got.note
    assert not [i for i in items(ctx) if "overhangs" in i]


@pytest.mark.parametrize("angle", [40.0, 60.0])
def test_a_seat_past_a_bevel_turned_more_than_the_crease_angle_overhangs(angle):
    """The knob pointer class: a plate flush on a block's top runs out over a bevel turned more
    than the style's crease angle (35). The bevel is an edge of the host, not more of its top."""
    host = bevelled_block(angle)
    plate = box((-0.1, 0.2, -0.05), (0.27, 0.22, 0.05))
    ctx = make_ctx({"host": host, "plate": plate}, {"plate": {"on": "host"}})
    assert f"{THING} plate overhangs host" in items(ctx)
    clear = box((-0.1, 0.2, -0.05), (0.15, 0.22, 0.05))  # wholly on the flat top
    assert findings(make_ctx({"host": host, "plate": clear}, {"plate": {"on": "host"}})) == []


def test_a_style_with_no_crease_angle_cannot_read_a_seat():
    ctx = stack(x=0.25)
    dials = {**ctx.style.dials, "edges": {"outline": "runtime"}}
    ctx = dataclasses.replace(ctx, style=dataclasses.replace(ctx.style, dials=dials))
    got = entry(ctx, "block", "seat")
    assert got.status == "unmeasured" and "crease_deg" in got.note


@functools.cache
def strat_bar():
    return load_scene(BARS / "01-strat" / "scene")


def strat_entry(part: str, rule: str):
    ctx = strat_bar()
    view = next(v for v in pieces.contact_pieces(ctx, Bearing.kinds) if v.key == "strat:Strat")
    return next(e for e in bearing.measure(ctx, view, {part}) if e.rule == rule)


@pytest.mark.held("bars/01-strat")
def test_strat_bar_fret_22_is_as_deep_as_it_lies_under_the_board_not_its_run_off_the_end():
    """Fret_22 hangs over the board's end: 0.60 mm under the top, 0.82 mm along the board. The
    shortest way out is under 0.62 mm (along the top's normal, or slant through its corner)."""
    got = strat_entry("Fret_22", "sink")
    mm = got.plan["depth"] * 2000  # this scene is 0.5 units a metre
    assert got.status == "fail" and 0.5 < mm < 0.62


@pytest.mark.held("bars/01-strat")
def test_strat_bar_trem_arm_set_into_its_elbow_is_not_a_pass():
    got = strat_entry("TremArm_001", "sink")
    assert got.status == "fail" and got.plan["depth"] * 2000 > 2.5


@pytest.mark.held("bars/01-strat")
def test_strat_bar_switch_tip_wider_than_its_blade_is_not_a_pass():
    """The tip rests flush on the blade's 1.4 by 3.2 mm flat top and runs 1.4 mm past it each
    side: a seat read nowhere is never a pass, and here it is read and overhangs."""
    got = strat_entry("SwitchTip", "seat")
    assert got.status == "fail"


@pytest.mark.held("bars/01-strat")
def test_strat_bar_dot_03_inlay_is_not_an_overhang():
    """Every seat-side corner of the inlay lies 0.10 mm inside the radiused board."""
    assert strat_entry("Dot_03", "seat").status == "pass"


def test_lamp_bar_seats_are_judged_and_pass():
    """The lamp as built: the arms and the cable's end are judged by rule 3, not skipped, and bear.
    The cable's flat end meets the base's 72-gon side across a facet crease (4.19 degrees)."""
    ctx = load_scene(BARS / "05-lamp" / "scene")
    view = next(v for v in pieces.contact_pieces(ctx, Bearing.kinds) if v.key == "lamp:Lamp")
    seats = {e.part: e.status for e in bearing.measure(ctx, view) if e.rule == "seat"}
    assert {seats[p] for p in ("arm_lower", "arm_upper", "cable")} == {"pass"}, seats


def test_sink_source_cites_the_contact_rule_and_the_other_rules_a_source():
    check = Bearing()
    assert "Baraff" in check.source and "Nonpenetration" in check.source
    assert "Make It Stand" in check.source and "Bicchi" in check.source
    assert "no outside source" not in check.source


def test_lamp_bar_resting_parts_are_quiet():
    """Known good: the lamp. Its one live `rests` row is the switch on the strip's body, which no
    up/down patch test judges (its patch faces neither up nor down as a whole, so rule 1 prints
    not-applicable); rule 5 judges it by depth, so the unit is not blind."""
    ctx = load_scene(BARS / "05-lamp" / "scene")
    assert findings(ctx) == []
    judged = []
    for view in pieces.contact_pieces(ctx, Bearing.kinds):
        for e in bearing.measure(ctx, view):
            if e.rule == "sink" and e.status in ("pass", "fail"):
                judged.append((view.key, e.part, e.status))
    assert judged == [("lamp:Strip", "switch", "pass")]
    view = next(v for v in pieces.contact_pieces(ctx, Bearing.kinds) if v.key == "lamp:Strip")
    rest = next(e for e in bearing.measure(ctx, view, {"switch"}) if e.rule == "rest")
    assert rest.status == "na" and "neither up nor down" in rest.note
    units = [u for u in Bearing().plants(ctx) if "sinks into" in u.item]
    assert [u.item.split(" sinks")[0] for u in units] == ["lamp:Strip switch"]
    assert plants_caught(Bearing(), ctx)
    assert run_gate([Bearing()], ctx, None).status == "passed"


def test_a_sink_plan_with_no_finite_level_builds_no_plant(monkeypatch):
    """`level` is -inf when no contact normal lies within 45 degrees of the contact direction. A
    plant pushed by `_sink_push - level` would then move the part an infinite distance, so that
    part is skipped and the unit lands on the next, or on none."""
    ctx = tabletop_on_legs()
    real = bearing._sink_plan

    def no_level(piece, part, host):
        entry = real(piece, part, host)
        if entry.status == "ready":
            entry.plan["level"] = -np.inf
        return entry

    assert [u.item for u in Bearing().plants(ctx) if "sinks into" in u.item]  # there is one
    monkeypatch.setattr(bearing, "_sink_plan", no_level)
    sinks = [u for u in Bearing().plants(ctx) if "sinks into" in u.item]
    assert sinks == []
    for unit in Bearing().plants(ctx):  # nothing else is moved by an infinite distance either
        for plant in unit.candidates:
            geometry = plant.ctx.pieces[piece_key("kit", "Thing")].parts
            assert all(np.isfinite(tris).all() for tris in geometry.values())
