import dataclasses
import pathlib
import textwrap

import pytest

from model_kit.core import gate
from model_kit.core.checks import no_fallback
from model_kit.core.checks.no_fallback import (
    PLANT_FAULTS,
    PLANT_SOURCE,
    NoFallback,
    _scan_source,
    with_anchor,
)
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.manifest import anchor
from model_kit.core.metagate import Unmeasured, run_gate

from .manifest_scenes import make_ctx
from .piece_scenes import refuses_the_core, tried_and_caught

CLEAN = "x = 1\nmounts = {}\nclamp = mounts['clamp']\n"


def scene(source=CLEAN, folder="stage"):
    return make_ctx(sources={"build.py": textwrap.dedent(source)}, folder=folder)


def findings(ctx, only=None):
    return NoFallback().run(ctx, only=only)


def items(source):
    return sorted(f.item for f in findings(scene(source)))


def test_the_check_is_registered_and_is_not_a_piece_check():
    assert gate.CHECKS["no-fallback"] is NoFallback
    assert NoFallback.name == "no-fallback"
    assert NoFallback.kinds == frozenset()


def test_clean_sources_pass_and_every_plant_is_caught():
    ctx = scene()
    assert findings(ctx) == []
    assert tried_and_caught(NoFallback(), ctx)
    assert run_gate([NoFallback()], ctx, None).status == "passed"


def test_the_listed_plant_faults_are_what_the_scan_finds_in_the_plant_script():
    """`PLANT_FAULTS` is written out beside the script so a plant never asks the scan for its
    items; this pins the two together."""
    found = [item.partition(" ")[2] for item in _scan_source("f.py", PLANT_SOURCE)]
    assert sorted(found) == sorted(PLANT_FAULTS)
    units = NoFallback().plants(scene())
    assert [u.item for u in units] == [
        *(f"plant_fallback.py {fault}" for fault in PLANT_FAULTS), "anchor missing-mount"]


def test_the_plants_never_call_the_core(monkeypatch):
    refuses_the_core(monkeypatch, no_fallback, "_scan_source", "_source_findings",
                     "_anchor_findings")
    assert len(NoFallback().plants(scene())) == len(PLANT_FAULTS) + 1


# --- the .get rule ----------------------------------------------------------------------------


def test_a_default_on_a_dict_of_mounts_fails():
    (found,) = findings(scene("x = mounts.get('clamp', (0, 0, 0))\n"))
    assert (found.check, found.item) == ("no-fallback", "build.py get-default mounts")
    assert "line 1" in found.detail


@pytest.mark.parametrize("name", ["anchor", "hold", "mount", "clamp", "seat", "use"])
def test_each_name_is_watched(name):
    assert items(f"x = {name}s.get(key, None)\n") == [f"build.py get-default {name}s"]


@pytest.mark.parametrize("target", ["self.mounts", "scene.holds", "rig.seat_table",
                                    "AnchorMap", "per_use"])
def test_a_name_or_an_attribute_that_holds_the_word_is_watched(target):
    assert len(items(f"x = {target}.get(k, [])\n")) == 1


@pytest.mark.parametrize("target", ["threshold", "amount", "house", "reuse", "thresholds",
                                    "households", "self.amount", "per_house"])
def test_a_word_that_only_contains_a_watched_word_is_not_watched(target):
    assert items(f"x = {target}.get(k, 0)\n") == []


@pytest.mark.parametrize("target", ["hold", "mount_point", "seat_map", "mountPoint", "per_use"])
def test_a_watched_word_as_a_whole_token_is_watched(target):
    assert items(f"x = {target}.get(k, 0)\n") == [f"build.py get-default {target}"]


def test_get_on_an_unwatched_dict_and_indexing_pass():
    assert items("a = settings.get('colour')\nb = settings.get('colour', 'red')\n") == []
    assert items("c = mounts['clamp']\nd = mounts.setdefault('clamp', 1)\n") == []


# --- the two forms the .get rule also sees ----------------------------------------------------


def test_one_argument_get_on_a_watched_table_is_found():
    (found,) = findings(scene("a = HELD_ANCHOR.get(k)\n"))
    assert found.item == "build.py get-no-default HELD_ANCHOR"
    assert "line 1" in found.detail


def test_conditional_index_is_found():
    (found,) = findings(scene("a = T_mounts[k] if k in T_mounts else d\n"))
    assert found.item == "build.py conditional-index T_mounts"
    assert "line 1" in found.detail


@pytest.mark.parametrize("source", [
    "a = mounts[j] if k in mounts else d\n",         # another key than the one tested
    "a = mounts[k] if k in other else d\n",           # another table than the one indexed
    "a = mounts[k] if k not in mounts else d\n",      # the test is the other way round
    "a = settings[k] if k in settings else d\n",      # a table that is not watched
    "a = mounts[k] if ready else d\n",                # no membership test
])
def test_a_conditional_that_is_not_the_lookup_form_passes(source):
    assert items(source) == []


def test_a_keyword_default_is_a_default_and_a_bare_get_is_not():
    assert items("a = mounts.get(k, default=0)\nb = holds.get(k)\n") == [
        "build.py get-default mounts", "build.py get-no-default holds"]


def test_a_get_with_two_arguments_is_still_a_default_not_a_bare_get():
    assert items("a = mounts.get(k, 0)\n") == ["build.py get-default mounts"]


# the sources the two new plants must be caught on: a scene with a clean script, one with a
# script that already holds another fault, and one with no sources at all
PLANT_SCENES = {
    "clean script": lambda: scene(),
    "script with another fault": lambda: scene("holds = {}\nx = holds.get(1, 0)\n"),
    "no sources": lambda: make_ctx(folder="cafe"),
}


@pytest.mark.parametrize("scene_name", list(PLANT_SCENES))
@pytest.mark.parametrize("rule", ["get-no-default", "conditional-index"])
def test_both_new_plants_are_caught_on_two_sources(scene_name, rule):
    ctx = PLANT_SCENES[scene_name]()
    (unit,) = [u for u in NoFallback().plants(ctx) if f" {rule} " in u.item]
    (plant,) = unit.candidates
    assert [f.item for f in NoFallback().run(plant.ctx, only=plant.scope)] == [plant.item]


def bar_sources():
    """Every bar build source, keyed by its path from the repo root, so no two share a key."""
    repo = pathlib.Path(__file__).resolve().parents[3]
    paths = sorted([*(repo / "bars").glob("*/build.py"),
                    *(repo / "bars").glob("*/scene/sources/*.py")])
    sources = {str(p.relative_to(repo)): p.read_text() for p in paths}
    assert len(sources) == len(paths) >= 4
    return sources


def test_bars_stay_quiet():
    assert findings(make_ctx(sources=bar_sources())) == []


def test_the_bar_scan_reports_a_fault_in_any_bar_source():
    sources = bar_sources()
    for name in sources:
        planted = {**sources, name: sources[name] + "\nx = HELD_ANCHOR.get(k)\n"}
        got = findings(make_ctx(sources=planted))
        assert [f.item for f in got] == [f"{name} get-no-default HELD_ANCHOR"]


def test_two_defaults_on_one_name_are_one_item_with_both_lines():
    (found,) = findings(scene("a = holds.get(1, 0)\nb = holds.get(2, 0)\n"))
    assert "lines 1, 2" in found.detail


# --- prints and swallowed errors --------------------------------------------------------------


@pytest.mark.parametrize("source, item", [
    ("print('using fallback point')\n", "build.py print fallback"),
    ("print('Fallback!')\n", "build.py print fallback"),
    ("print(f'MISMATCH on {name}')\n", "build.py print MISMATCH"),
    ("print('chain', 'MISMATCH')\n", "build.py print MISMATCH"),
])
def test_a_print_that_announces_a_fallback_fails(source, item):
    assert items(source) == [item]


@pytest.mark.parametrize("text", ["mismatch", "Mismatch", "a size mismatch"])
def test_mismatch_is_matched_as_written(text):
    assert items(f"print('{text}')\n") == []


def test_an_ordinary_print_passes():
    assert items("print('built the kit')\nlog('fallback')\n") == []


def test_except_exception_pass_fails():
    source = "try:\n    work()\nexcept Exception:\n    pass\n"
    assert items(source) == ["build.py except-pass"]


@pytest.mark.parametrize("names", ["(Exception, KeyError)", "(KeyError, BaseException)",
                                   "(builtins.Exception,)"])
def test_a_tuple_that_names_a_broad_error_fails(names):
    source = f"try:\n    work()\nexcept {names}:\n    pass\n"
    assert items(source) == ["build.py except-pass"]


def test_a_tuple_of_narrow_errors_passes():
    assert items("try:\n    work()\nexcept (KeyError, ValueError):\n    pass\n") == []


def test_a_bare_except_pass_fails_too():
    assert items("try:\n    work()\nexcept:\n    pass\n") == ["build.py except-pass"]


def test_a_handler_that_does_something_or_names_a_narrow_error_passes():
    assert items("try:\n    work()\nexcept Exception:\n    raise\n") == []
    assert items("try:\n    work()\nexcept KeyError:\n    pass\n") == []
    assert items("try:\n    work()\nexcept Exception as exc:\n    log(exc)\n") == []


def test_a_source_that_does_not_parse_raises():
    with pytest.raises(ManifestError, match="build.py"):
        findings(scene("def broken(:\n"))


# --- the anchor rule --------------------------------------------------------------------------


def test_the_real_anchor_raises_on_a_missing_mount():
    assert findings(make_ctx(sources={"b.py": CLEAN})) == []


def test_an_anchor_that_returns_a_default_fails():
    def lenient(packs, ref, *, clip=None):
        return None

    ctx = with_anchor(scene(), lenient)
    (found,) = findings(ctx)
    assert found.item == "anchor missing-mount"


def test_an_anchor_that_fails_with_another_error_fails():
    def broken(packs, ref, *, clip=None):
        raise TypeError("oops")

    (found,) = findings(with_anchor(scene(), broken))
    assert found.item == "anchor missing-mount"
    assert "TypeError" in found.detail


def test_the_anchor_rule_asks_for_a_ref_the_piece_does_not_have():
    asked = []

    def spy(packs, ref, *, clip=None):
        asked.append(ref)
        return anchor(packs, ref, clip=clip)

    findings(with_anchor(scene(), spy))
    (ref,) = asked
    piece = make_ctx().packs[ref.pack].pieces[ref.piece]
    taken = {x.id for group in (piece.holds, piece.seats, piece.mounts, piece.uses)
             for x in group}
    assert ref.id not in taken


def test_a_scene_with_no_pack_still_gets_the_anchor_rule():
    ctx = dataclasses.replace(scene(), packs={}, clips={}, layouts={}, samples={}, pieces={})
    assert findings(ctx) == []


# --- no sources -------------------------------------------------------------------------------


def test_a_scene_with_no_sources_is_partial_and_still_runs_the_anchor_rule():
    ctx = make_ctx(folder="cafe")
    (line,) = NoFallback().coverage(ctx)
    assert line == "no-fallback: skipped (no sources): cafe"
    assert isinstance(line, Unmeasured)
    verdict = run_gate([NoFallback()], ctx, None)
    assert verdict.status == "partial"
    lenient = with_anchor(ctx, lambda packs, ref, *, clip=None: None)
    assert [f.item for f in findings(lenient)] == ["anchor missing-mount"]


def test_the_plants_run_on_a_scene_with_no_sources():
    ctx = make_ctx(folder="cafe")
    assert "anchor missing-mount" in tried_and_caught(NoFallback(), ctx)


def test_an_unnamed_scene_folder_reads_scene():
    (line,) = NoFallback().coverage(make_ctx())
    assert line == "no-fallback: skipped (no sources): scene"


def test_a_scene_with_sources_has_no_coverage_line():
    assert NoFallback().coverage(scene()) == []


# --- scope ------------------------------------------------------------------------------------


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = scene("a = holds.get(1, 0)\nprint('fallback')\n")
    assert len(findings(ctx)) == 2
    assert [f.item for f in findings(ctx, only={"build.py print fallback"})] == [
        "build.py print fallback"]


def test_a_blind_check_stops_the_run():
    class Blind(NoFallback):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="no-fallback"):
        run_gate([Blind()], scene(), None)
