import dataclasses

import pytest

from model_kit.core import gate
from model_kit.core.checks import done
from model_kit.core.checks.done import Done
from model_kit.core.errors import BlindCheck
from model_kit.core.manifest import Recipe
from model_kit.core.metagate import Unmeasured, run_gate

from .manifest_scenes import make_ctx, plants_caught, with_piece
from .piece_scenes import refuses_the_core, tried_and_caught


def findings(ctx, only=None):
    return Done().run(ctx, only=only)


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["done"] is Done
    assert Done.name == "done"
    assert Done.kinds == frozenset({"assembly"})


def test_a_clean_scene_passes_and_every_plant_is_caught():
    ctx = make_ctx()
    assert findings(ctx) == []
    assert plants_caught(Done(), ctx)
    assert run_gate([Done()], ctx, None).status == "passed"


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_a_piece_with_an_empty_done_fails(text):
    (found,) = findings(with_piece(make_ctx(), "Deck", done=text))
    assert (found.check, found.item) == ("done", "kit:Deck done")


@pytest.mark.parametrize("size", ["", "huge", None])
def test_a_piece_with_a_missing_or_unknown_size_class_fails(size):
    (found,) = findings(with_piece(make_ctx(), "Deck", size_class=size))
    assert (found.check, found.item) == ("done", "kit:Deck size_class")


@pytest.mark.parametrize("size", ["prop", "piece", "hero"])
def test_each_size_class_passes(size):
    assert findings(with_piece(make_ctx(), "Deck", size_class=size)) == []


def test_both_faults_are_two_items():
    ctx = with_piece(make_ctx(), "Deck", done="", size_class="")
    assert sorted(f.item for f in findings(ctx)) == ["kit:Deck done", "kit:Deck size_class"]


def test_the_check_needs_no_geometry():
    ctx = dataclasses.replace(make_ctx(), pieces={})
    assert findings(ctx) == []
    assert Done().coverage(ctx) == []
    assert run_gate([Done()], ctx, None).status == "passed"


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = with_piece(make_ctx(), "Deck", recipe=Recipe("branching", "grow.py"), done="")
    assert findings(ctx) == []
    (line,) = Done().coverage(ctx)
    assert line == "done: not applicable (branching piece): kit:Deck"
    assert not isinstance(line, Unmeasured)


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = with_piece(make_ctx(), "Deck", done="", size_class="")
    assert [f.item for f in findings(ctx, only={"kit:Deck size_class"})] == ["kit:Deck size_class"]


def test_the_plants_empty_the_done_text_and_drop_the_size_class():
    plants = Done().plants(make_ctx())
    assert {p.item for p in plants} == {"kit:Deck done", "kit:Deck size_class",
                                        "kit:Doll done", "kit:Doll size_class"}


def test_the_plants_never_call_the_core(monkeypatch):
    refuses_the_core(monkeypatch, done, "_done_empty", "_size_unknown")
    assert len(Done().plants(make_ctx())) == 4


def test_the_plants_are_caught_on_two_pieces():
    """Both faults are planted on the deck and on the doll, one unit each."""
    assert sorted(tried_and_caught(Done(), make_ctx())) == [
        "kit:Deck done", "kit:Deck size_class", "kit:Doll done", "kit:Doll size_class"]


def test_a_blind_check_stops_the_run():
    class Blind(Done):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="done"):
        run_gate([Blind()], make_ctx(), None)
