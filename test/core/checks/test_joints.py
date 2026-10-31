"""The joint helper (`joints.seat`) and the two checks that judge through it, `receiver` and
`connectors`: a guest rests on the host part its mount names, and a plug in a port stands in a bore
of that part.

The meshes are upright (y up). `HOST_HOLE` is a ring, a tube with a through-hole of radius `HOLE`;
`HOST_PLAIN` is a block of the same height. The first scenes call `joints.seat` on bare meshes; the
guards run the checks on the assembly scene (the rack's jack has a pit, the lead's plug pin hangs in
it) and on a lamp (a plug whose blades stand in the bores of a power strip).
"""

import dataclasses
import functools
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks import joints
from model_kit.core.checks.connectors import Connectors
from model_kit.core.checks.pieces import PartMesh
from model_kit.core.checks.receiver import Receiver
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom, load_scene

from ..shapes import box, cylinder
from .assembly_scenes import (
    make_ctx,
    two_guests,
    with_geometry,
    with_mount,
    with_piece,
    with_stool,
)
from .piece_scenes import chosen, tried_and_caught
from .sight_scenes import viewed

BARS = Path(__file__).resolve().parents[3] / "bars"
LOCATE = 0.025
UP = (0.0, 1.0, 0.0)
HOLE, RING, FACE = 0.02, 0.05, 0.1  # the hole's radius, the ring's, the height of the face


def mesh(name: str, tris: np.ndarray) -> PartMesh:
    return PartMesh(name, tris, role="solid", shape=None, group="g", on=None)


HOST_HOLE = mesh("ring", cylinder((0.0, 0.0), RING, 0.0, FACE, hole=HOLE))
HOST_PLAIN = mesh("block", box((-RING, 0.0, -RING), (RING, FACE, RING)))
POINT = (0.0, FACE, 0.0)


def peg(tip: float, top: float = FACE + 0.04, radius: float = 0.012) -> np.ndarray:
    return cylinder((0.0, 0.0), radius, tip, top)


def plug_with(pegs: np.ndarray) -> PartMesh:
    """A plug: a body lying on the face and the pegs given."""
    return mesh("plug", np.concatenate([box((-0.04, FACE, -0.04), (0.04, FACE + 0.03, 0.04)),
                                        pegs]))


def with_deck(ctx):
    """The scene with a second part of the rack, `deck`, a block apart from the frame: in the
    rack's manifest as well as its geometry, so a mount can name it."""
    rack = ctx.packs["kit"].pieces["Rack"]
    frame = next(p for p in rack.parts if p.name == "frame")
    ctx = with_piece(ctx, "Rack", parts=(*rack.parts, dataclasses.replace(frame, name="deck")))
    return with_geometry(ctx, "kit:Rack", deck=box((0.3, 0.0, -0.1), (0.4, 0.1, 0.1)))


def seat(guest, host, *, point=POINT, bore=UP, locate=LOCATE):
    return joints.seat(make_ctx(), guest=guest, host=host, point=point, locate=locate,
                       bore_axis=bore)


# --- the helper on bare meshes -----------------------------------------------------------------


def test_a_peg_standing_in_a_hole_is_seated():
    result = seat(plug_with(peg(0.04, top=FACE)), HOST_HOLE)
    assert (result.status, result.seated) == (joints.SEATED, True)


def test_a_peg_alone_in_a_hole_is_seated_the_body_is_not_needed():
    assert seat(mesh("peg", peg(0.04)), HOST_HOLE).status == joints.SEATED


def test_a_plug_that_stands_on_a_face_with_no_bore_is_no_bore():
    plug = mesh("plug", box((-0.01, FACE, -0.01), (0.01, FACE + 0.04, 0.01)))
    result = seat(plug, HOST_PLAIN)
    assert result.status == joints.NO_BORE and "no hole" in result.detail


def test_a_peg_sunk_into_material_is_no_bore():
    result = seat(plug_with(peg(0.04, top=FACE)), HOST_PLAIN)
    assert result.status == joints.NO_BORE and "material" in result.detail


def test_a_peg_half_in_the_hole_is_no_bore():
    # shifted so that part of it lies in the ring's material
    moved = peg(0.04, top=FACE) + np.array([HOLE, 0.0, 0.0])
    result = seat(plug_with(moved), HOST_HOLE, point=(HOLE / 2, FACE, 0.0))
    assert result.status == joints.NO_BORE and "material" in result.detail


def test_a_peg_in_a_hole_is_no_bore_without_a_wall_all_round():
    # a post beside a wall is not in a hole: the ring's wall does not close round it
    tray = mesh("tray", box((-RING, 0.0, -RING), (-0.02, FACE, RING)))  # one wall only
    beside = mesh("peg", peg(0.04, radius=0.008) + np.array([-0.012, 0.0, 0.0]))
    result = seat(beside, tray, point=(-0.012, FACE, 0.0))
    assert result.status == joints.NO_BORE and "all round" in result.detail


def test_a_rest_needs_no_bore():
    plug = mesh("plug", box((-0.01, FACE, -0.01), (0.01, FACE + 0.04, 0.01)))
    assert seat(plug, HOST_PLAIN, bore=None).status == joints.SEATED


def test_a_guest_far_from_the_point_is_off_the_host():
    result = seat(mesh("peg", peg(0.04) + np.array([0.3, 0.0, 0.0])), HOST_HOLE)
    assert result.status == joints.OFF_HOST and "from peg" in result.detail


def test_a_point_far_from_the_host_part_is_off_the_host():
    # the guest stands at the point; the host part is elsewhere (the mount names the wrong part)
    far = mesh("elsewhere", box((0.3, 0.0, 0.3), (0.4, FACE, 0.4)))
    result = seat(mesh("peg", peg(0.04)), far)
    assert result.status == joints.OFF_HOST and "host part elsewhere" in result.detail


def test_a_point_buried_deeper_than_locate_is_off_the_host_not_a_crash():
    # the point lies in the host's material, further than `locate` from every face of it, while
    # the guest touches the top: no mouth can be read, which is a named result
    block = mesh("block", box((-0.5, 0.0, -0.5), (0.5, 1.0, 0.5)))
    plug = mesh("plug", box((-0.01, 0.98, -0.01), (0.01, 1.01, 0.01)))
    result = seat(plug, block, point=(0.0, 0.96, 0.0), locate=0.025)
    assert result.status == joints.OFF_HOST and "inside the material" in result.detail


def test_seat_has_no_default_for_the_bore():
    # every caller says whether the joint needs a hole
    with pytest.raises(TypeError, match="bore_axis"):
        joints.seat(make_ctx(), guest=HOST_PLAIN, host=HOST_PLAIN, point=POINT, locate=LOCATE)


def test_a_guest_hovering_over_the_host_is_off_the_host():
    # the point lies between them, within the locate of both, but they do not touch
    hover = mesh("plug", box((-0.01, FACE + 0.03, -0.01), (0.01, FACE + 0.06, 0.01)))
    result = seat(hover, HOST_PLAIN, point=(0.0, FACE + 0.015, 0.0), bore=None, locate=0.02)
    assert result.status == joints.OFF_HOST and "rest on it" in result.detail


# --- the guards: two pieces each, through the checks -------------------------------------------


@functools.cache
def bar05():
    return load_scene(BARS / "05-lamp" / "scene")


LEAD_SEAT = "kit:Lead run end0 seat"
LAMP_SEAT = "lamp:Lamp mains end1 seat"


def test_plug_on_a_face_with_no_bore_fails_on_two_pieces():
    """A plug moved onto the face beside its bore fails the seat rule, on the assembly scene's
    lead and on bar 05's lamp; the gate keeps that move as the plant of both."""
    for ctx, item in ((make_ctx(), LEAD_SEAT), (bar05(), LAMP_SEAT)):
        assert Connectors().run(ctx) == []
        assert item in tried_and_caught(Connectors(), ctx)
        plant = next(p for p in chosen(Connectors(), ctx) if p.item == item)
        assert plant.label.startswith("moved a peg's width aside")
        (found,) = Connectors().run(plant.ctx, only={item})
        assert "no bore round it" in found.detail, found.detail


def withdrawn(ctx, key, parts, by):
    """The scene with the plug's `parts` of piece `key` pulled `by` straight up the mount's axis
    (the piece stands unturned, so its y is the world's)."""
    geom = ctx.pieces[key]
    up = np.array([0.0, by, 0.0])
    return dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom(geom.name, {
        **geom.parts, **{n: geom.parts[n] + up for n in parts}})})


def blades(radius: float = 0.004, apart: float = 0.012) -> np.ndarray:
    """Two pegs either side of the joint point, as a plug's blades stand either side of its
    socket's centre."""
    return np.concatenate([peg(0.04, radius=radius) + np.array([s * apart, 0.0, 0.0])
                           for s in (1.0, -1.0)])


# A socket with two bores, 12 mm either side of its centre, in two rings that nearly meet there.
TWO_BORES = mesh("sockets", np.concatenate([
    cylinder((s * 0.012, 0.0), 0.011, 0.0, FACE, hole=0.006) for s in (1.0, -1.0)]))


def test_a_plug_half_withdrawn_from_its_socket_is_not_seated_on_two_pieces():
    """A plug with its blades beside the joint point, pulled half way out, no longer rests on its
    host: the face under its body is further from the point than the contact length. A fixed 66 mm
    limit would read it as seated. On bare meshes (blades 100 mm long) and on a lamp (17.1 mm)."""
    plug, up = plug_with(blades()), np.array([0.0, 0.05, 0.0])
    assert seat(plug, TWO_BORES, locate=0.005).status == joints.SEATED
    pulled = dataclasses.replace(plug, tris=plug.tris + up)
    result = seat(pulled, TWO_BORES, locate=0.005)
    assert result.status == joints.OFF_HOST and "from plug" in result.detail, result
    assert seat(pulled, TWO_BORES, locate=0.2).status == joints.SEATED  # a long contact length
    ctx = bar05()
    assert Connectors().run(ctx, only={LAMP_SEAT}) == []
    assert LAMP_SEAT in tried_and_caught(Connectors(), ctx)
    pulled_lamp = withdrawn(ctx, "lamp:Lamp", ("plug", "blade_hot", "blade_neutral"), 0.5 * 0.0171)
    (found,) = Connectors().run(pulled_lamp, only={LAMP_SEAT})
    assert "the joint point is" in found.detail and "no bore" not in found.detail, found.detail
    # seen from so far off that the pull cannot be told from standing on the face, it is seated
    far = viewed(ctx, 4 * 0.5 * 0.0171)
    assert Connectors().run(withdrawn(far, "lamp:Lamp", ("plug", "blade_hot", "blade_neutral"),
                                      0.5 * 0.0171), only={LAMP_SEAT}) == []


def test_a_pin_through_the_joint_point_is_seated_while_it_stays_in_the_bore():
    """The limit of the rule: no insertion depth is judged. The lead's pin passes through the
    point, so a lead pulled half way out still stands in its pit and reads seated.

    This pins a limit of the rule ("a pin through the joint
    point reads seated while half withdrawn"). It is not wanted behaviour: a rule that judged
    insertion depth would read this case differently."""
    ctx = make_ctx()
    pulled = withdrawn(ctx, "kit:Lead", ("plug", "pin"), 0.5 * 0.04)
    assert Connectors().run(pulled, only={LEAD_SEAT}) == []


def test_the_lead_in_a_rack_without_a_pit_is_no_bore():
    plain = with_geometry(make_ctx(), "kit:Rack", frame=box((-0.2, 0.0, -0.1), (0.2, 0.1, 0.1)))
    (found,) = Connectors().run(plain, only={LEAD_SEAT})
    assert "material" in found.detail


def test_a_mount_that_declares_a_rest_needs_no_bore():
    # the pin hangs in the rack's material, the cap on its face: a rest declares no bore, so
    # the contact is all that is asked
    plain = with_geometry(make_ctx(), "kit:Rack", frame=box((-0.2, 0.0, -0.1), (0.2, 0.1, 0.1)))
    assert len(Connectors().run(plain, only={LEAD_SEAT})) == 1  # a port: no bore, no seat
    rest = with_mount(plain, "Rack", "jack", kind="rest")
    assert Connectors().run(rest, only={LEAD_SEAT}) == []


def test_guest_moved_off_its_host_part_fails_on_two_pieces():
    """A guest that does not stand on the host part its mount names fails the touch rule, on the
    rack's two lamps. The mount names a part of the rack far from the guest; and the gate's plant
    (the guest moved off) is caught on both."""
    ctx = with_deck(two_guests())
    assert Receiver().run(ctx) == []
    touches = {"room lamp touch", "room lamp2 touch"}
    assert touches <= set(tried_and_caught(Receiver(), ctx))
    named_elsewhere = with_mount(with_mount(ctx, "Rack", "cup", part="deck"), "Rack", "cup2",
                                 part="deck")
    got = {f.item: f.detail for f in Receiver().run(named_elsewhere)}
    assert set(got) == touches
    assert all("host part deck" in detail for detail in got.values()), got
    for plant in (p for p in chosen(Receiver(), ctx) if p.item in touches):
        (found,) = Receiver().run(plant.ctx, only={plant.item})
        assert "from" in found.detail


def test_stool_moved_off_its_host_part_fails_on_a_second_shape():
    """The same rule on a guest that is not the lamp or a copy of it: a stool with a round pin,
    on the rack's deck. The gate's plant (the stool moved off) is caught, and a mount that names
    the frame, a part the stool does not stand on, fails the touch rule."""
    ctx = with_stool()
    assert Receiver().run(ctx) == []
    assert {"room lamp touch", "room stool touch"} <= set(tried_and_caught(Receiver(), ctx))
    for plant in (p for p in chosen(Receiver(), ctx) if p.item == "room stool touch"):
        (found,) = Receiver().run(plant.ctx, only={plant.item})
        assert "from" in found.detail
    named_elsewhere = with_mount(ctx, "Rack", "perch", part="frame")
    (found,) = Receiver().run(named_elsewhere)
    assert found.item == "room stool touch" and "host part frame" in found.detail, found


def test_a_guest_resting_on_another_part_of_its_host_is_off_the_host_part():
    # the pin stands on the frame at the cup; the mount names the deck, so the point (on the
    # frame) is far from the host part
    ctx = with_deck(make_ctx())
    assert Receiver().run(ctx) == []
    (found,) = Receiver().run(with_mount(ctx, "Rack", "cup", part="deck"))
    assert found.item == "room lamp touch" and "deck" in found.detail


# --- the known-good joint and the missing records ---------------------------------------------


def test_lamp_bar_joint_is_quiet():
    ctx = bar05()
    assert [f for f in Connectors().run(ctx) if "seat" in f.item] == []
    lines = Connectors().coverage(ctx)
    assert not [x for x in lines if isinstance(x, Unmeasured)], lines
    assert {"lamp:Lamp plug", "lamp:Lamp blade_hot", "lamp:Lamp blade_neutral",
            "lamp:Strip body"} <= Connectors().measured(ctx)
    assert run_gate([Connectors()], ctx, None).status == "passed"


def bar_with_records(ctx, *, part="body", plug="plug"):
    pack = ctx.packs["lamp"]
    strip, lamp = pack.pieces["Strip"], pack.pieces["Lamp"]
    mounts = tuple(dataclasses.replace(m, part=part) if m.id == "socket_3" else m
                   for m in strip.mounts)
    (conn,) = lamp.connectors
    own, layout_end = conn.ends
    lamp = dataclasses.replace(lamp, connectors=(dataclasses.replace(
        conn, ends=(own, dataclasses.replace(layout_end, plug=plug))),))
    pieces = {**pack.pieces, "Strip": dataclasses.replace(strip, mounts=mounts), "Lamp": lamp}
    return dataclasses.replace(ctx, packs={"lamp": dataclasses.replace(pack, pieces=pieces)})


def test_mount_without_part_reads_unmeasured():
    ctx = bar_with_records(bar05(), part=None)
    assert Connectors().run(ctx) == []  # nothing is guessed from the geometry
    (line,) = [x for x in Connectors().coverage(ctx) if LAMP_SEAT in x]
    assert isinstance(line, Unmeasured) and "no host part named" in line
    assert not [u for u in Connectors().plants(ctx) if u.item == LAMP_SEAT]
    assert run_gate([Connectors()], ctx, None).status == "partial"
    # and the receiver reads a guest on a mount that names no part the same way
    scene = with_mount(make_ctx(), "Rack", "cup", part=None)
    (rline,) = [x for x in Receiver().coverage(scene) if isinstance(x, Unmeasured)]
    assert "no host part named" in rline and "room lamp" in rline
    assert Receiver().run(scene) == []
    assert not [u for u in Receiver().plants(scene) if u.item == "room lamp touch"]
    assert run_gate([Receiver()], scene, None).status == "partial"


def test_end_without_plug_reads_unmeasured():
    ctx = bar_with_records(bar05(), plug=None)
    (line,) = [x for x in Connectors().coverage(ctx) if LAMP_SEAT in x]
    assert isinstance(line, Unmeasured) and "no plug named" in line


def test_a_host_that_is_no_closed_solid_leaves_a_port_unmeasured():
    # the hole cannot be told from material in a mesh that is not a closed solid
    open_frame = box((-0.2, 0.0, -0.1), (0.2, 0.1, 0.1))[:-2]
    ctx = with_geometry(make_ctx(), "kit:Rack", frame=open_frame)
    (line,) = [x for x in Connectors().coverage(ctx) if LEAD_SEAT in x]
    assert isinstance(line, Unmeasured) and "closed solid" in line


def test_a_plug_that_names_no_geometry_is_unmeasured():
    ctx = make_ctx(pieces=False)
    (line,) = [x for x in Connectors().coverage(ctx)
               if LEAD_SEAT in x and isinstance(x, Unmeasured)]
    assert "no geometry" in line


def test_the_check_states_its_principle_and_source():
    for check in (Receiver(), Connectors()):
        assert check.principle.endswith(".") and ". " not in check.principle
        assert "Boothroyd, Dewhurst and Knight" in check.source
        assert "section 3.12, pp. 106-108" in check.source and "UNVERIFIED" not in check.source


def test_the_seat_plants_never_call_the_helper(monkeypatch):
    ctx = make_ctx()

    def refuse(*args, **kwargs):
        raise AssertionError("a plant called the joint helper")

    monkeypatch.setattr(joints, "seat", refuse)
    assert LEAD_SEAT in [u.item for u in Connectors().plants(ctx)]
    assert "room lamp touch" in [u.item for u in Receiver().plants(ctx)]


def test_a_blind_joint_verdict_stops_the_run(monkeypatch):
    """The verdict function, `joints.seat`, always says seated: the gate finds no plant it
    catches on the seat item or the touch item, and raises, instead of reading it as quiet."""
    from model_kit.core.errors import BlindCheck

    monkeypatch.setattr(joints, "seat", lambda *a, **k: joints.JointResult(joints.SEATED, ""))
    with pytest.raises(BlindCheck, match="end0 seat"):
        run_gate([Connectors()], make_ctx(), None)
    with pytest.raises(BlindCheck, match="room lamp touch"):
        run_gate([Receiver()], make_ctx(), None)


def test_the_off_host_candidate_is_caught_on_two_real_pieces():
    """The plug moved out of reach of the joint's point reads off_host on the assembly scene's
    lead and on bar 05's lamp, two different pieces with different sockets."""
    for ctx, item in ((make_ctx(), LEAD_SEAT), (bar05(), LAMP_SEAT)):
        (unit,) = [u for u in Connectors().plants(ctx) if u.item == item]
        away = [p for p in unit.candidates if p.label.startswith("moved out of reach")]
        assert away
        for plant in away:
            (found,) = Connectors().run(plant.ctx, only={item})
            assert "the joint point is" in found.detail and "no bore" not in found.detail
