"""Check receiver, receiver and ports: a guest placed on a host mount touches the mount, the mount takes
that piece, and a power or signal mount takes only a connector end.

The scene is `assembly_scenes`: the lamp stands on the rack's cup in the layout `room`. Items are
`<layout> <placement> touch|fits|port`.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.pieces import with_piece
from model_kit.core.checks.receiver import Receiver
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import Placement, Ref, View
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.sight import seen_distance
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS

from ..shapes import box
from .assembly_scenes import (
    CUP,
    END,
    LAMP_AT,
    PERCH,
    Q_ID,
    Q_Y90,
    make_ctx,
    plants_caught,
    two_guests,
    with_geometry,
    with_mount,
    with_placement,
    with_stool,
)
from .piece_scenes import accepted, refuses_the_core, tried_and_caught

TOUCH_ITEM = "room lamp touch"
FITS_ITEM = "room lamp fits"
PORT_ITEM = "room lamp port"


def findings(ctx, only=None):
    return Receiver().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def lamp_lifted(by):
    """The lamp `by` above the cup, along the mount's axis."""
    return with_placement(make_ctx(), "lamp", position=(LAMP_AT[0], LAMP_AT[1] + by, LAMP_AT[2]))


def test_the_check_is_registered_and_is_a_layout_check():
    assert gate.CHECKS["receiver"] is Receiver
    assert Receiver.name == "receiver"
    assert Receiver.kinds == frozenset()


def test_a_guest_on_its_cup_passes_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(Receiver(), ctx)
    assert run_gate([Receiver()], ctx, None).status == "passed"


def test_calibration_half_the_end_limit_passes_and_double_fails():
    assert findings(lamp_lifted(CALIBRATION_PASS * END)) == []
    (found,) = findings(lamp_lifted(CALIBRATION_FAIL * END))
    assert (found.check, found.item) == ("receiver", TOUCH_ITEM)
    assert "kit:Lamp pin" in found.detail


def _with_length(ctx, piece, length):
    """The scene with this piece's `view` set so that its contact length is `length`."""
    old = ctx.packs["kit"].pieces[piece]
    view = View(seen_distance(length, ctx.packs["kit"].units_per_metre), "a test distance")
    return with_piece(ctx, "kit", piece, dataclasses.replace(old, view=view))


def test_the_touch_gap_is_judged_by_the_guests_contact_length_not_the_hosts():
    """The lamp (guest) and the rack (host) declare different lengths. A lamp lifted 0.75 x its own
    length is seated even when the rack's length is a tenth of that; a lamp lifted 1.5 x its own
    length is found even when the rack's length is ten times that."""
    short, long = END / 10, END * 10
    near = _with_length(lamp_lifted(0.75 * END), "Rack", short)
    assert findings(near) == []
    far = _with_length(lamp_lifted(1.5 * END), "Rack", long)
    (found,) = findings(far)
    assert (found.item, "kit:Lamp pin" in found.detail) == (TOUCH_ITEM, True)
    # the same lifts read the other way once the guest's length is the one that changes
    assert [f.item for f in findings(_with_length(near, "Lamp", short))] == [TOUCH_ITEM]
    assert findings(_with_length(far, "Lamp", long)) == []


def test_the_mount_point_is_read_through_the_hosts_turn():
    turned = with_placement(make_ctx(), "rack", quaternion=Q_Y90)
    assert items(turned) == [TOUCH_ITEM]  # the lamp stayed where the cup used to be
    cup = Placement("rack", "Rack", "kit", (0.0, 0.0, 0.0), Q_Y90).to_world(CUP)
    assert not np.allclose(cup, CUP)
    assert findings(with_placement(turned, "lamp", position=tuple(cup))) == []


def test_the_takes_part_is_read_through_the_guests_placement():
    # the pin stands 0.21 out from the lamp's origin, so the guest's turn moves it
    side = box((0.2, 0.0, -0.01), (0.22, 0.05, 0.01))
    ctx = with_geometry(make_ctx(), "kit:Lamp", pin=side)
    pin_foot = (0.21, 0.0, 0.0)
    turned = Placement("lamp", "Lamp", "kit", (0.0, 0.0, 0.0), Q_Y90).to_world(pin_foot)
    seated = np.asarray(CUP) - turned
    good = with_placement(ctx, "lamp", position=tuple(seated), quaternion=Q_Y90)
    assert findings(good) == []
    unturned = with_placement(good, "lamp", quaternion=Q_ID)
    assert items(unturned) == [TOUCH_ITEM]


def test_a_mount_that_fits_other_pieces_refuses_the_guest():
    ctx = with_mount(make_ctx(), "Rack", "cup", fits=("Fan",))
    (found,) = findings(ctx)
    assert (found.item, "Fan" in found.detail) == (FITS_ITEM, True)
    assert findings(with_mount(make_ctx(), "Rack", "cup", fits=None)) == []
    assert findings(with_mount(make_ctx(), "Rack", "cup", fits=("Fan", "Lamp"))) == []


def test_a_power_or_signal_mount_takes_a_connector_end_not_a_piece():
    for kind in ("power", "signal"):
        (found,) = findings(with_mount(make_ctx(), "Rack", "cup", kind=kind))
        assert found.item == PORT_ITEM
        assert kind in found.detail
    assert findings(with_mount(make_ctx(), "Rack", "cup", kind="rest")) == []


def test_every_rule_has_a_plant_that_the_check_catches():
    units = Receiver().plants(make_ctx())
    assert sorted(u.item for u in units) == sorted([TOUCH_ITEM, FITS_ITEM, PORT_ITEM])
    assert sorted(tried_and_caught(Receiver(), make_ctx())) == sorted(
        [TOUCH_ITEM, FITS_ITEM, PORT_ITEM])


def test_the_touch_plant_moves_the_guest_twice_the_limit_along_the_axis():
    (unit,) = [u for u in Receiver().plants(make_ctx()) if u.item == TOUCH_ITEM]
    plant = accepted(Receiver(), unit)
    moved = plant.ctx.layouts["room"].placements["lamp"].position
    delta = np.asarray(moved) - np.asarray(LAMP_AT)
    assert np.linalg.norm(delta) == pytest.approx(CALIBRATION_FAIL * END)
    assert abs(delta[1]) == pytest.approx(CALIBRATION_FAIL * END)  # along the cup's axis, y


def test_a_scoped_run_yields_only_the_items_it_names():
    ctx = with_mount(lamp_lifted(CALIBRATION_FAIL * END), "Rack", "cup", fits=("Fan",))
    assert items(ctx) == [FITS_ITEM, TOUCH_ITEM]
    assert [f.item for f in findings(ctx, only={FITS_ITEM})] == [FITS_ITEM]
    assert findings(ctx, only={"room rack touch"}) == []


def test_a_blind_check_stops_the_run():
    class Blind(Receiver):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="receiver"):
        run_gate([Blind()], make_ctx(), None)


def test_a_guest_with_no_geometry_is_unmeasured_and_the_run_partial():
    ctx = make_ctx(pieces=False)
    assert TOUCH_ITEM not in items(ctx)  # nothing to measure, so no finding
    lines = Receiver().coverage(ctx)
    assert any(isinstance(line, Unmeasured) and "no geometry" in line and "room lamp" in line
               for line in lines), lines
    verdict = run_gate([Receiver()], ctx, None)
    assert verdict.status == "partial"


def test_a_scene_with_no_placement_on_a_mount_has_nothing_to_receive():
    ctx = with_placement(make_ctx(), "lamp", on=None)
    assert findings(ctx) == []
    lines = Receiver().coverage(ctx)
    assert lines and not any(isinstance(line, Unmeasured) for line in lines)
    assert run_gate([Receiver()], ctx, None).status == "passed"
    bare = dataclasses.replace(make_ctx(), layouts={})
    assert Receiver().coverage(bare) == lines


def test_a_placement_on_a_mount_its_host_does_not_publish_raises():

    ctx = with_placement(make_ctx(), "lamp", on=Ref("kit", "rack", "nowhere"))
    with pytest.raises(ManifestError, match="nowhere"):
        findings(ctx)


def test_a_mount_whose_every_guest_already_fails_needs_no_plant():
    ctx = with_mount(lamp_lifted(CALIBRATION_FAIL * END), "Rack", "cup", fits=("Fan",),
                     kind="power")
    assert items(ctx) == [FITS_ITEM, PORT_ITEM, TOUCH_ITEM]
    assert sorted(u.item for u in Receiver().plants(ctx)) == [FITS_ITEM, PORT_ITEM, TOUCH_ITEM]
    assert tried_and_caught(Receiver(), ctx) == []  # the real run reports each: the gate skips them
    assert run_gate([Receiver()], ctx, None).status == "failed"
    assert not any(isinstance(line, Unmeasured) for line in Receiver().coverage(ctx))


# --- the plants are built blind to the check -------------------------------------------------


def test_receiver_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import receiver

    ctx = make_ctx()
    want = [u.item for u in Receiver().plants(ctx)]
    refuses_the_core(monkeypatch, receiver, "_findings", "_entries", "_joint")
    refuses_the_core(monkeypatch, receiver.joints, "seat")
    monkeypatch.setattr(Receiver, "run", lambda *a, **k: 1 / 0)
    assert [u.item for u in Receiver().plants(ctx)] == want and len(want) == 3


def test_receiver_plants_caught_on_two_guests():
    """The lamp on the cup and the second lamp on the second cup: each rule of each is tried
    and caught, and the gate does not stop."""
    ctx = two_guests()
    assert findings(ctx) == []
    got = tried_and_caught(Receiver(), ctx)
    assert sorted(got) == sorted(f"room {g} {rule}" for g in ("lamp", "lamp2")
                                 for rule in ("touch", "fits", "port"))
    assert run_gate([Receiver()], ctx, None).status == "passed"


def test_receiver_plants_caught_on_a_guest_of_another_shape_on_another_host_part():
    """The lamp (a square pin, on the rack's frame) and a stool (a round pin of another radius, on
    the rack's deck): each rule of each is tried and caught. The stool shares no mesh, mount or
    host part with the lamp, so the touch rule is proven on more than one shape."""
    ctx = with_stool()
    assert findings(ctx) == []
    got = tried_and_caught(Receiver(), ctx)
    assert sorted(got) == sorted(f"room {g} {rule}" for g in ("lamp", "stool")
                                 for rule in ("touch", "fits", "port"))
    assert run_gate([Receiver()], ctx, None).status == "passed"


def test_the_stool_touch_rule_reads_its_own_pin_and_its_own_host_part():
    ctx = with_stool()
    lifted = with_placement(ctx, "stool", position=(PERCH[0], PERCH[1] + CALIBRATION_FAIL * END,
                                                    PERCH[2]))
    (found,) = findings(lifted)
    assert (found.item, "kit:Stool pin" in found.detail) == ("room stool touch", True)
    nearer = with_placement(ctx, "stool", position=(PERCH[0], PERCH[1] + CALIBRATION_PASS * END,
                                                    PERCH[2]))
    assert findings(nearer) == []
