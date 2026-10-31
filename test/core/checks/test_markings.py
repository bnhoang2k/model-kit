from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.markings import Markings
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import load_scene

from ..shapes import box
from .furniture_scenes import furniture_ctx, with_finish, with_part_role, with_part_shape
from .piece_scenes import make_ctx, plants_caught, refuses_the_core

STRAT = Path(__file__).resolve().parents[1] / "fixtures" / "regressions" / "strat_before_neck_plate"

LEGEND = "kit:Label legend"
BADGE = "kit:Label badge"


def findings(ctx, only=None):
    return Markings().run(ctx, only=only)


def items(ctx):
    return [f.item for f in findings(ctx)]


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["markings"] is Markings
    assert Markings.kinds == frozenset({"assembly"})


def test_a_printed_legend_and_a_raised_badge_pass_and_the_plants_are_caught():
    ctx = furniture_ctx("Label")
    assert findings(ctx) == []
    assert plants_caught(Markings(), ctx)
    assert run_gate([Markings()], ctx, None).status == "passed"


def test_solid_print_row_fails():
    """Rule 1: the legend's row says print, but its part is built as a solid box."""
    ctx = with_part_shape(furniture_ctx("Label"), "Label", "legend", "box")
    (found,) = findings(ctx)
    assert found.item == f"{LEGEND} is a solid print"
    assert "decal" in found.detail


@pytest.mark.parametrize("finish", ["print", "paint", "inlay", "display"])
def test_every_marked_finish_wants_a_decal(finish):
    ctx = with_finish(furniture_ctx("Label"), "Label", "ink", finish)
    assert findings(ctx) == []
    ctx = with_part_shape(ctx, "Label", "legend", "box")
    assert items(ctx) == [f"{LEGEND} is a solid {finish}"]


def test_decal_of_a_solid_row_fails():
    """Rule 2: the legend is a decal, but its row says solid."""
    ctx = with_finish(furniture_ctx("Label"), "Label", "ink", "solid")
    (found,) = findings(ctx)
    assert found.item == f"{LEGEND} decal of a solid row"
    assert "print" in found.detail


@pytest.mark.parametrize("role", ["decal", "eyes"])
def test_a_printed_role_on_a_box_in_a_solid_row_fails_rule_2(role):
    """A printed role is a marking whatever shape the part is built with."""
    ctx = with_part_shape(furniture_ctx("Label"), "Label", "badge", "box", role=role)
    assert items(ctx) == [f"{BADGE} decal of a solid row"]


@pytest.mark.parametrize(("piece", "part"), [("Label", "badge"), ("Inlay", "plate")])
@pytest.mark.parametrize("role", ["decal", "eyes"])
def test_the_role_alone_plants_rule_2_on_two_pieces(piece, part, role):
    """Plant: only the role of a closed solid part of a solid row becomes printed (its shape stays).
    Rule 2 must name it, and the same piece with its own role is quiet."""
    ctx = furniture_ctx(piece)
    assert items(ctx) == []
    planted = with_part_role(ctx, piece, part, role)
    assert items(planted) == [f"kit:{piece} {part} decal of a solid row"]


def test_a_solid_part_of_a_solid_row_is_not_judged_by_rules_1_and_2():
    ctx = furniture_ctx("Label")
    assert not [i for i in items(ctx) if "is a solid" in i or "decal of" in i]


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = with_finish(furniture_ctx("Label"), "Label", "ink", "solid")
    ctx = with_part_shape(ctx, "Label", "badge", "decal", role="decal")
    assert len(findings(ctx)) == 2
    only = {f"{LEGEND} decal of a solid row"}
    assert [f.item for f in findings(ctx, only=only)] == [f"{LEGEND} decal of a solid row"]


def test_the_plants_are_one_per_rule_and_each_names_its_own_item():
    plants = Markings().plants(furniture_ctx("Label"))
    assert all(len(u.candidates) == 1 for u in plants)
    assert sorted(p.item for p in plants) == sorted([
        f"{LEGEND} is a solid print", f"{LEGEND} decal of a solid row"])


def test_a_piece_with_no_decal_plants_the_manifest_rules_on_a_solid_part():
    plants = Markings().plants(furniture_ctx("Inlay"))
    assert sorted(p.item for p in plants) == ["kit:Inlay plate decal of a solid row",
                                              "kit:Inlay plate is a solid print"]


def test_a_flush_plate_is_left_to_the_eye_test():
    """Rule 3 (flush by the eye's resolution at the nearest view) lands with `view`."""
    assert items(furniture_ctx("Inlay")) == []


def test_a_piece_with_no_solid_part_to_plant_on_says_not_applicable():
    mark = box((0, 0, 0), (0.1, 0.001, 0.1))
    empty = np.zeros((0, 3, 3))  # a solid part with no triangles still sets the thickness base
    ctx = make_ctx({"stock": empty, "mark": mark},
                   {"mark": {"shape": "decal", "role": "decal", "on": "stock"}})
    (unit,) = Markings().plants(ctx)
    assert unit.item == "kit:Thing" and unit.candidates == ()
    verdict = run_gate([Markings()], ctx, None)
    assert ("markings: not applicable (no solid part to plant a marking fault on): "
            "kit:Thing") in verdict.coverage
    assert not any(isinstance(line, Unmeasured) for line in verdict.coverage)
    assert verdict.status == "failed"  # the decal in a solid row is a finding


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = make_ctx({"base": box((-0.1, 0.0, -0.1), (0.1, 0.02, 0.1))},
                   also={"Tree": {"trunk": box((0, 0, 0), (0.1, 0.1, 0.1))}})
    lines = Markings().coverage(ctx)
    assert any("kit:Tree" in line and "not applicable" in line for line in lines)


@pytest.mark.held("test/core/fixtures/regressions/strat_before_neck_plate")
def test_plants_caught_on_two_pieces():
    """The Strat regression scene's dots and the fixture label."""
    strat = load_scene(STRAT)
    check = Markings()
    strat_plants = check.plants(strat)
    label_plants = check.plants(furniture_ctx("Label"))
    assert {p.item.split()[0] for p in strat_plants} == {"strat:Strat"}
    assert {p.item.split()[0] for p in label_plants} == {"kit:Label"}
    for plants in (strat_plants, label_plants):  # each rule is planted on each piece
        assert {p.item.rsplit(" ", 3)[-1] for p in plants} == {"print", "row"}
    for ctx in (strat, furniture_ctx("Label")):
        assert plants_caught(check, ctx)
        run_gate([check], ctx, None)  # no BlindCheck


def test_markings_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import markings

    ctx = furniture_ctx("Label")
    refuses_the_core(monkeypatch, markings, "_findings", "_rule1", "_rule2")
    monkeypatch.setattr(markings.Markings, "run", lambda *a, **k: 1 / 0)
    assert sorted(u.item for u in Markings().plants(ctx)) == sorted([
        f"{LEGEND} is a solid print", f"{LEGEND} decal of a solid row"])


