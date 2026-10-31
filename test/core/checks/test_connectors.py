"""Check connectors, connector ends and spans: each end lies within the end limit of the mount it plugs
into, and the cable part is no longer than its cap.

The lead of `assembly_scenes` has the connector `run`: end 0 plugs into the rack's jack through
the layout `room`, end 1 into the lead's own mount `tip`; the cable is 0.3 long, capped at 0.35.
Items are `<pack>:<piece> <connector> end0|end1|end0 cable|end1 cable|end0 seat|cap`. The cable's
far end lies on the jack in the layout `room` and on the lead's own `tip` in its own frame, so the
cable rules pass on a clean scene; the lead's `plug` pin hangs in the jack's pit, so the seat rule
does too (the self end is attached, not seated: it has no seat item).
"""

import copy
import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.connectors import Connectors
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import LayoutPlacement, Placement
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS

from ..shapes import box
from .assembly_scenes import (
    CABLE_LENGTH,
    CAP,
    END,
    JACK,
    LEAD_AT,
    Q_ID,
    Q_Y90,
    load,
    make_ctx,
    pack_json,
    plants_caught,
    with_geometry,
    with_piece,
    with_placement,
)
from .piece_scenes import accepted, refuses_the_core, tried_and_caught

END0, END1, CAP_ITEM = "kit:Lead run end0", "kit:Lead run end1", "kit:Lead run cap"
CABLE0, CABLE1 = "kit:Lead run end0 cable", "kit:Lead run end1 cable"
SEAT0 = "kit:Lead run end0 seat"


def findings(ctx, only=None):
    return Connectors().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def with_end(ctx, index, **fields):
    """The scene with these fields of the connector's end `index` replaced."""
    lead = ctx.packs["kit"].pieces["Lead"]
    (conn,) = lead.connectors
    ends = tuple(dataclasses.replace(e, **fields) if i == index else e
                 for i, e in enumerate(conn.ends))
    return with_piece(ctx, "Lead", connectors=(dataclasses.replace(conn, ends=ends),))


def end_moved(ctx, index, dx):
    (conn,) = ctx.packs["kit"].pieces["Lead"].connectors
    x, y, z = conn.ends[index].point
    return with_end(ctx, index, point=(x + dx, y, z))


def cable(length):
    half = length / 2
    return {"cable": box((-half, -0.002, -0.002), (half, 0.002, 0.002))}


def test_the_check_is_registered_and_is_a_pack_check():
    assert gate.CHECKS["connectors"] is Connectors
    assert Connectors.name == "connectors"
    assert Connectors.kinds == frozenset()


def test_a_lead_on_its_jack_within_its_cap_passes_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(Connectors(), ctx)
    assert run_gate([Connectors()], ctx, None).status == "passed"


@pytest.mark.parametrize("index, item", [(0, END0), (1, END1)])
def test_calibration_half_the_end_limit_passes_and_double_fails(index, item):
    assert findings(end_moved(make_ctx(), index, CALIBRATION_PASS * END)) == []
    (found,) = findings(end_moved(make_ctx(), index, CALIBRATION_FAIL * END), only={item})
    assert (found.check, found.item) == ("connectors", item)


def test_a_layout_end_is_read_through_the_hosts_turn():
    turned = with_placement(make_ctx(), "rack", quaternion=Q_Y90)
    # the jack moved; the end did not, and the plug no longer stands in the pit
    assert items(turned) == sorted([END0, SEAT0])
    jack = Placement("rack", "Rack", "kit", (0.0, 0.0, 0.0), Q_Y90).to_world(JACK)
    assert not np.allclose(jack, JACK)
    # the end follows the jack; the cable and plug stay where they were, so the cable and seat
    # rules are left
    assert items(with_end(turned, 0, point=tuple(jack))) == sorted([CABLE0, SEAT0])


def test_a_self_end_is_read_in_the_pieces_own_frame():
    # moving the lead in the layout moves neither its own end nor its own mount nor the cable
    # against its own end; it takes the cable and plug away from the layout end
    moved = with_placement(make_ctx(), "lead", position=(5.0, 5.0, 5.0))
    assert items(moved) == sorted([CABLE0, SEAT0])


def test_calibration_a_cable_under_its_cap_passes_and_one_stretched_past_it_fails():
    under = with_geometry(make_ctx(), "kit:Lead", **cable(CAP - CALIBRATION_PASS * END))
    assert findings(under) == []
    over = with_geometry(make_ctx(), "kit:Lead", **cable(CAP + CALIBRATION_FAIL * END))
    (found,) = findings(over)
    assert (found.item, "0.35" in found.detail) == (CAP_ITEM, True)


def test_the_length_runs_along_the_cable_not_along_the_axes():
    # a 0.3 cable turned 45 degrees about z spans only 0.21 along x or y, but is 0.3 long
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    tilted = {"cable": cable(CABLE_LENGTH)["cable"] @ turn.T}
    assert np.ptp(tilted["cable"].reshape(-1, 3), axis=0).max() < 0.25
    short_cap = dataclasses.replace(make_ctx().packs["kit"].pieces["Lead"].connectors[0], cap=0.25)
    ctx = with_piece(with_geometry(make_ctx(), "kit:Lead", **tilted), "Lead",
                     connectors=(short_cap,))
    assert [f.item for f in findings(ctx, only={CAP_ITEM})] == [CAP_ITEM]


def test_the_cap_plant_stretches_the_cable_past_the_cap_by_twice_the_cap():
    (unit,) = [u for u in Connectors().plants(make_ctx()) if u.item == CAP_ITEM]
    plant = accepted(Connectors(), unit)
    assert plant.label.startswith("stretched")
    stretched = plant.ctx.pieces["kit:Lead"].parts["cable"]
    assert np.ptp(stretched.reshape(-1, 3), axis=0).max() == pytest.approx(
        CAP * (1 + CALIBRATION_FAIL))


def test_each_item_has_a_plant_that_the_check_catches():
    units = Connectors().plants(make_ctx())
    assert sorted(u.item for u in units) == sorted([END0, END1, CABLE0, CABLE1, SEAT0, CAP_ITEM])
    assert sorted(tried_and_caught(Connectors(), make_ctx())) == sorted(
        [END0, END1, CABLE0, CABLE1, SEAT0, CAP_ITEM])


def test_an_end_plant_moves_the_end_twice_the_limit():
    (unit,) = [u for u in Connectors().plants(make_ctx()) if u.item == END1]
    plant = accepted(Connectors(), unit)
    (before,) = make_ctx().packs["kit"].pieces["Lead"].connectors
    (after,) = plant.ctx.packs["kit"].pieces["Lead"].connectors
    moved = np.asarray(after.ends[1].point) - np.asarray(before.ends[1].point)
    assert np.linalg.norm(moved) == pytest.approx(CALIBRATION_FAIL * END)


def test_a_scoped_run_yields_only_the_items_it_names():
    ctx = with_geometry(end_moved(make_ctx(), 0, CALIBRATION_FAIL * END), "kit:Lead",
                        **cable(CAP + CALIBRATION_FAIL * END))
    assert items(ctx) == [CAP_ITEM, END0]
    assert [f.item for f in findings(ctx, only={CAP_ITEM})] == [CAP_ITEM]


def test_a_blind_check_stops_the_run():
    class Blind(Connectors):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="connectors"):
        run_gate([Blind()], make_ctx(), None)


def test_a_lead_with_no_geometry_still_has_its_ends_measured_and_its_cap_unmeasured():
    ctx = end_moved(make_ctx(pieces=False), 0, CALIBRATION_FAIL * END)
    assert items(ctx) == [END0]
    lines = Connectors().coverage(ctx)
    assert any(isinstance(line, Unmeasured) and "no geometry" in line and "kit:Lead run" in line
               for line in lines), lines
    assert run_gate([Connectors()], make_ctx(pieces=False), None).status == "partial"


def test_a_cable_part_with_no_triangles_is_unmeasured():
    ctx = with_geometry(make_ctx(), "kit:Lead", cable=np.zeros((0, 3, 3)))
    assert CAP_ITEM not in items(ctx)
    assert any(isinstance(line, Unmeasured) for line in Connectors().coverage(ctx))


def test_a_scene_with_no_connector_has_nothing_to_measure():
    ctx = with_piece(make_ctx(), "Lead", connectors=())
    assert findings(ctx) == []
    lines = Connectors().coverage(ctx)
    assert lines and not any(isinstance(line, Unmeasured) for line in lines)
    assert run_gate([Connectors()], ctx, None).status == "passed"


def test_an_end_on_a_layout_the_scene_lacks_raises():
    ctx = dataclasses.replace(make_ctx(), layouts={})
    with pytest.raises(ManifestError, match="room"):
        findings(ctx)


def test_an_end_on_a_placement_or_mount_that_is_missing_raises():
    from model_kit.core.manifest import Ref

    with pytest.raises(ManifestError, match="nowhere"):
        findings(with_end(make_ctx(), 0, host=Ref("kit", "rack", "nowhere")))
    with pytest.raises(ManifestError, match="ghost"):
        findings(with_end(make_ctx(), 0, host=Ref("kit", "ghost", "jack")))


def test_a_cable_that_is_over_its_cap_already_needs_no_cap_plant():
    ctx = with_geometry(make_ctx(), "kit:Lead", **cable(CAP + CALIBRATION_FAIL * END))
    assert CAP_ITEM in [u.item for u in Connectors().plants(ctx)]
    assert CAP_ITEM not in tried_and_caught(Connectors(), ctx)  # the gate skips it: it fails
    assert not any(isinstance(line, Unmeasured) for line in Connectors().coverage(ctx))


def test_a_cable_of_no_length_cannot_be_stretched_so_its_cap_is_planted_with_a_bar():
    point = np.zeros((1, 3, 3))
    ctx = with_geometry(make_ctx(), "kit:Lead", cable=point)
    (unit,) = [u for u in Connectors().plants(ctx) if u.item == CAP_ITEM]
    assert [p.label.split(":")[0] for p in unit.candidates] == ["a bar"]  # nothing to stretch
    plant = accepted(Connectors(), unit)
    assert plant is not None
    assert np.ptp(plant.ctx.pieces["kit:Lead"].parts["cable"].reshape(-1, 3),
                  axis=0).max() == pytest.approx(CAP * (1 + CALIBRATION_FAIL))
    verdict = run_gate([Connectors()], ctx, None)  # no BlindCheck: the cap's unit is tried
    assert all(CAP_ITEM not in line for line in verdict.lines)  # a point is within its cap
    assert not any(isinstance(line, Unmeasured) and "cap" in line for line in verdict.coverage)


# --- the cable's own surface reaches each end -------------------------------------------------


def moved_cable(ctx, dx):
    """The scene with the cable's geometry shifted `dx` along x; the manifest is untouched."""
    tris = ctx.pieces["kit:Lead"].parts["cable"]
    return with_geometry(ctx, "kit:Lead", cable=tris + np.array([dx, 0.0, 0.0]))


def test_a_clean_cable_reaches_both_ends():
    assert findings(make_ctx(), only={CABLE0, CABLE1}) == []


@pytest.mark.parametrize("dx, item", [(-CALIBRATION_FAIL * END, CABLE1),
                                      (CALIBRATION_FAIL * END, CABLE0)])
def test_a_cable_that_no_longer_reaches_an_end_fails_though_the_manifest_numbers_agree(dx, item):
    ctx = moved_cable(make_ctx(), dx)
    # the manifest end points still sit on their mounts; only the geometry moved
    assert [f.item for f in findings(ctx) if f.item in (END0, END1)] == []
    assert item in items(ctx)


def test_a_cord_reaches_its_end_by_way_of_the_plug_it_enters_on_two_pieces():
    """A cord enters the back of its plug, not the mount's point: the cable reaches the end when it
    touches the plug and the plug touches the point. On the lead (the cable pulled 4 mm back along
    its plug) and on a lamp (the cable enters the plug's side, 12.7 mm from the point)."""
    from pathlib import Path

    from model_kit.core.scene import load_scene

    from .sight_scenes import viewed

    bar = load_scene(Path(__file__).resolve().parents[3] / "bars" / "05-lamp" / "scene")
    lamp = "lamp:Lamp mains end1 cable"
    assert findings(bar, only={lamp}) == []
    assert lamp in tried_and_caught(Connectors(), bar)
    lead = viewed(make_ctx(), 0.002)
    pulled = moved_cable(lead, 0.004)  # the cable's end lies 4 mm off the point, within the plug
    assert findings(pulled, only={CABLE0}) == []
    assert CABLE0 in tried_and_caught(Connectors(), lead)
    # with no plug named, only the point counts
    assert [f.item for f in findings(with_end(pulled, 0, plug=None), only={CABLE0})] == [CABLE0]
    # a cable pulled clear of the plug as well still fails, and says it went by way of the plug
    (found,) = findings(moved_cable(lead, 0.04), only={CABLE0})
    assert "by way of its plug" in found.detail


def test_calibration_half_the_end_limit_passes_for_the_cable_and_double_fails():
    # shifting the cable along +x moves its far end (local +0.15) off end 1 by the shift
    assert findings(moved_cable(make_ctx(), -CALIBRATION_PASS * END), only={CABLE0, CABLE1}) == []
    assert items(moved_cable(make_ctx(), CALIBRATION_FAIL * END)) == [CABLE0]


def test_a_turned_lead_is_read_through_its_placement():
    jack_end = np.array([-CABLE_LENGTH / 2, 0.0, 0.0])
    turn = Placement("lead", "Lead", "kit", (0.0, 0.0, 0.0), Q_Y90)
    position = np.asarray(JACK) - (turn.to_world(jack_end) - turn.position)
    ctx = with_placement(make_ctx(), "lead", position=tuple(position), quaternion=Q_Y90)
    assert findings(ctx, only={CABLE0}) == []
    nudged = with_placement(ctx, "lead", position=tuple(position + [0.1, 0.0, 0.0]))
    assert [f.item for f in findings(nudged, only={CABLE0})] == [CABLE0]


def test_the_cable_plants_move_the_geometry_not_the_manifest():
    units = {u.item: u for u in Connectors().plants(make_ctx())}
    for item in (CABLE0, CABLE1):
        plant = accepted(Connectors(), units[item])
        assert plant.ctx.packs["kit"].pieces["Lead"].connectors == (
            make_ctx().packs["kit"].pieces["Lead"].connectors)
        assert not np.allclose(plant.ctx.pieces["kit:Lead"].parts["cable"],
                               make_ctx().pieces["kit:Lead"].parts["cable"])


def test_a_layout_with_no_placement_of_the_lead_leaves_the_layout_end_unmeasured():
    ctx = make_ctx()
    layout = ctx.layouts["room"]
    bare = dataclasses.replace(layout, placements={k: v for k, v in layout.placements.items()
                                                   if k != "lead"})
    ctx = dataclasses.replace(ctx, layouts={"room": bare})
    assert findings(ctx, only={CABLE0}) == []
    lines = Connectors().coverage(ctx)
    assert [x for x in lines if CABLE0 in x] == [
        f"connectors: skipped (no placement): {CABLE0}"]
    assert isinstance(next(x for x in lines if CABLE0 in x), Unmeasured)
    assert not any(CABLE1 in x for x in lines)  # the self end needs no placement
    assert run_gate([Connectors()], ctx, None).status == "partial"


def test_a_layout_with_two_placements_of_the_lead_leaves_the_layout_end_unmeasured():
    ctx = make_ctx()
    layout = ctx.layouts["room"]
    twin = dataclasses.replace(layout.placements["lead"], id="lead2")
    both = dataclasses.replace(layout, placements={**layout.placements, "lead2": twin})
    lines = Connectors().coverage(dataclasses.replace(ctx, layouts={"room": both}))
    assert [x for x in lines if CABLE0 in x] == [
        f"connectors: skipped (several placements): {CABLE0}"]


def test_a_lead_with_no_geometry_leaves_both_cable_ends_unmeasured():
    lines = Connectors().coverage(make_ctx(pieces=False))
    for item in (CABLE0, CABLE1):
        assert any(isinstance(x, Unmeasured) and "no geometry" in x and item in x for x in lines)


# --- the plants are built blind to the check -------------------------------------------------


def test_connectors_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import connectors

    ctx = make_ctx()
    want = [u.item for u in Connectors().plants(ctx)]
    refuses_the_core(monkeypatch, connectors, "_findings", "_entries", "_end_entry",
                     "_cable_end_entry", "_seat_entry", "_cap_entry", "length")
    refuses_the_core(monkeypatch, connectors.joints, "seat")
    monkeypatch.setattr(Connectors, "run", lambda *a, **k: 1 / 0)
    assert [u.item for u in Connectors().plants(ctx)] == want and len(want) == 6


def two_leads():
    """The scene with a second lead, `Lead2`, with its own connector `run2` and placement."""
    data = pack_json()
    second = copy.deepcopy(data["pieces"]["Lead"])
    second["connectors"][0]["id"] = "run2"
    data["pieces"]["Lead2"] = second
    ctx = make_ctx()
    room = ctx.layouts["room"]
    placed = LayoutPlacement("lead2", "kit", "Lead2", LEAD_AT, Q_ID)
    return dataclasses.replace(
        ctx, packs={"kit": load(data)}, pieces={**ctx.pieces, "kit:Lead2": ctx.pieces["kit:Lead"]},
        layouts={"room": dataclasses.replace(room, placements={**room.placements,
                                                               "lead2": placed})})


def test_connectors_plants_caught_on_two_leads():
    """Each lead has two ends, two cable rules, a seat and a cap; each is tried and caught."""
    ctx = two_leads()
    assert findings(ctx) == []
    got = tried_and_caught(Connectors(), ctx)
    assert sorted(got) == sorted(f"kit:{lead} {run} {rule}"
                                 for lead, run in (("Lead", "run"), ("Lead2", "run2"))
                                 for rule in ("end0", "end1", "end0 cable", "end1 cable",
                                              "end0 seat", "cap"))
    assert run_gate([Connectors()], ctx, None).status == "passed"
