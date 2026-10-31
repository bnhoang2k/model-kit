"""Check keep-off: each mount clears every keep-off point of its piece by that point's radius.

The rack of `assembly_scenes` has two mounts (`cup` and `jack`) and one keep-off point (`knot`,
radius `KNOT_RADIUS`). The item is `<pack>:<piece> <mount> <keep-off>`.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import keep_off
from model_kit.core.checks.keep_off import KeepOffCheck
from model_kit.core.errors import BlindCheck
from model_kit.core.manifest import Recipe
from model_kit.core.metagate import Unmeasured, run_gate

from .assembly_scenes import KNOT, KNOT_RADIUS, make_ctx, plants_caught, with_mount, with_piece
from .piece_scenes import refuses_the_core, tried_and_caught

CUP_ITEM = "kit:Rack cup knot"
JACK_ITEM = "kit:Rack jack knot"


def findings(ctx, only=None):
    return KeepOffCheck().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def cup_at(distance):
    """The cup `distance` from the knot, straight out along x."""
    return with_mount(make_ctx(), "Rack", "cup",
                      point=(KNOT[0] + distance, KNOT[1], KNOT[2]))


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["keep-off"] is KeepOffCheck
    assert KeepOffCheck.name == "keep-off"
    assert KeepOffCheck.kinds == frozenset({"assembly"})


def test_mounts_clear_of_the_knot_pass_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(KeepOffCheck(), ctx)
    assert run_gate([KeepOffCheck()], ctx, None).status == "passed"


def test_a_mount_inside_the_radius_fails_and_one_just_outside_passes():
    assert items(cup_at(0.9 * KNOT_RADIUS)) == [CUP_ITEM]
    assert items(cup_at(1.1 * KNOT_RADIUS)) == []
    (found,) = findings(cup_at(0.5 * KNOT_RADIUS))
    assert (found.check, found.item) == ("keep-off", CUP_ITEM)
    assert "knot" in found.detail


def test_a_mount_on_the_point_fails():
    assert items(with_mount(make_ctx(), "Rack", "cup", point=KNOT)) == [CUP_ITEM]


def test_every_mount_is_measured_against_every_keep_off_point():
    ctx = make_ctx()
    rack = ctx.packs["kit"].pieces["Rack"]
    extra = dataclasses.replace(rack.keep_off[0], id="coupler", point=rack.mounts[1].point)
    ctx = with_piece(ctx, "Rack", keep_off=(*rack.keep_off, extra))
    assert items(ctx) == ["kit:Rack jack coupler"]


def test_the_plant_moves_a_mount_onto_a_keep_off_point():
    units = KeepOffCheck().plants(make_ctx())
    assert sorted(u.item for u in units) == [CUP_ITEM, JACK_ITEM]  # one unit per pair
    for unit in units:
        (plant,) = unit.candidates
        moved = {m.id: m.point for m in plant.ctx.packs["kit"].pieces["Rack"].mounts}
        assert np.allclose(moved[plant.item.split()[1]], KNOT)


def test_a_pair_that_fails_already_still_gets_a_unit_and_the_gate_skips_it():
    ctx = with_mount(make_ctx(), "Rack", "cup", point=KNOT)
    assert CUP_ITEM in {u.item for u in KeepOffCheck().plants(ctx)}
    assert run_gate([KeepOffCheck()], ctx, None).status == "failed"


def test_a_scoped_run_yields_only_the_items_it_names():
    ctx = with_mount(cup_at(0.5 * KNOT_RADIUS), "Rack", "jack", point=KNOT)
    assert items(ctx) == [CUP_ITEM, JACK_ITEM]
    assert [f.item for f in findings(ctx, only={JACK_ITEM})] == [JACK_ITEM]


def test_a_blind_check_stops_the_run():
    class Blind(KeepOffCheck):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="keep-off"):
        run_gate([Blind()], make_ctx(), None)


def test_a_piece_with_a_mount_but_no_keep_off_point_has_nothing_to_measure():
    ctx = with_piece(make_ctx(), "Rack", keep_off=())
    assert findings(ctx) == []
    lines = KeepOffCheck().coverage(ctx)
    assert any("kit:Rack" in line and "no keep-off" in line for line in lines), lines
    assert not any(isinstance(line, Unmeasured) for line in lines)
    assert KeepOffCheck().plants(ctx) == []
    assert run_gate([KeepOffCheck()], ctx, None).status == "passed"


def test_a_piece_with_a_keep_off_point_but_no_mount_has_nothing_to_measure():
    ctx = with_piece(make_ctx(), "Rack", mounts=())
    lines = KeepOffCheck().coverage(ctx)
    assert any("kit:Rack" in line and "no mounts" in line for line in lines), lines
    assert run_gate([KeepOffCheck()], ctx, None).status == "passed"


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = with_piece(make_ctx(), "Rack", recipe=Recipe("branching", "grow.py"))
    assert findings(with_mount(ctx, "Rack", "cup", point=KNOT)) == []
    assert "keep-off: not applicable (branching piece): kit:Rack" in KeepOffCheck().coverage(ctx)


def test_the_check_reads_the_manifest_so_a_piece_with_no_geometry_is_measured():
    ctx = with_mount(make_ctx(pieces=False), "Rack", "cup", point=KNOT)
    assert items(ctx) == [CUP_ITEM]


def test_the_check_measures_points_not_parts():
    # a check that measures no parts has no `measured` hook (metagate: Measures)
    assert not hasattr(KeepOffCheck(), "measured")


def test_the_plants_never_call_the_clearance(monkeypatch):
    refuses_the_core(monkeypatch, keep_off, "_clearance")
    assert len(KeepOffCheck().plants(make_ctx())) == 2


def test_the_plants_are_caught_on_two_pieces():
    ctx = make_ctx()
    rack = ctx.packs["kit"].pieces["Rack"]
    two = with_piece(ctx, "Lamp", mounts=rack.mounts, keep_off=rack.keep_off)
    assert tried_and_caught(KeepOffCheck(), two) == [
        "kit:Rack cup knot", "kit:Rack jack knot", "kit:Lamp cup knot", "kit:Lamp jack knot"]
