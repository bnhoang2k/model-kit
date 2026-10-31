"""The guard for the reach, contact and source checks' plants: a check whose core
never reports a finding cannot see its plant, and the gate stops the run with `BlindCheck`.

Each check below is blinded on a clean scene of its own by patching the function its verdict rests
on (not by subclassing `run`: a plant may call those functions to choose what to plant, so only a
patched core blinds the check and its plants alike). A plant that the check's own verdict drops
(topology's "clean part" test, reach's "passes now" test, export hygiene's and coverage's trial
runs, no-fallback's trial scan) would leave no plant, so a blind core would read as a pass.

Some checks catch a blind core by construction: topology, keep-off, reach and no-fallback's lookup
keep a plant whenever their verdict says "clean", so a core that never finds anything keeps every
plant and the plant goes uncaught; must-touch builds its plants from the hold's geometry and
samples. They are units for the gate's one contract, not because a blind core fools them.
Three checks are the ones a blind core fools if the plant is never built (the run would read as a
pass): export hygiene, coverage's contact rule and no-fallback's script scan.
"""

import pytest

from model_kit.core.checks import (
    coverage,
    export_hygiene,
    keep_off,
    must_touch,
    no_fallback,
    reach,
    topology,
)
from model_kit.core.checks.coverage import Contacts, Coverage
from model_kit.core.checks.export_hygiene import ExportHygiene
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.keep_off import KeepOffCheck
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.checks.no_fallback import NoFallback
from model_kit.core.checks.reach import Reach
from model_kit.core.checks.topology import Topology
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from ..scenes import make_ctx as reach_ctx
from . import assembly_scenes, manifest_scenes
from . import test_must_touch as must_touch_tests
from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base
from .test_overlap import make_ctx as clip_ctx


def blind_topology(monkeypatch):
    monkeypatch.setattr(topology, "problems", lambda tris, size: [])


def blind_keep_off(monkeypatch):
    monkeypatch.setattr(keep_off, "_clearance", lambda mount, keep: 1e9)


def blind_must_touch(monkeypatch):
    monkeypatch.setattr(must_touch, "hold_gap", lambda *a, **k: (0.0, 0.0))


def blind_reach(monkeypatch):
    monkeypatch.setattr(reach._Job, "distance", property(lambda self: 0.0))


def blind_coverage_contacts(monkeypatch):
    # the scan lives in coverage alone: the run and every trial run call this name
    monkeypatch.setattr(coverage, "unheld_contacts", lambda *a, **k: Contacts({}, []))


def blind_coverage_parts(monkeypatch):
    monkeypatch.setattr(Coverage, "_unmeasured_parts", lambda self, ctx: [])


def blind_export_hygiene(monkeypatch):
    monkeypatch.setattr(export_hygiene, "_pack_findings", lambda *a, **k: [])


def blind_no_fallback_scan(monkeypatch):
    monkeypatch.setattr(no_fallback, "_scan_source", lambda name, text: {})


def blind_no_fallback_lookup(monkeypatch):
    monkeypatch.setattr(no_fallback, "_anchor_findings", lambda ctx, only: [])


def manifest_with_sources():
    return manifest_scenes.make_ctx(sources={"build.py": "x = 1\n"}, folder="stage")


def coverage_check():
    return Coverage([Floaters()])


coverage_check.name = "coverage"

SCENES = {
    "topology": (Topology, blind_topology, lambda: piece_ctx(post_on_base())),
    "keep-off": (KeepOffCheck, blind_keep_off, assembly_scenes.make_ctx),
    "must-touch": (MustTouch, blind_must_touch, must_touch_tests.make_ctx),
    "reach": (Reach, blind_reach, lambda: reach_ctx(at=(0.5, 0.0, 0.0))),
    "coverage-contacts": (coverage_check, blind_coverage_contacts, clip_ctx),
    "coverage-parts": (coverage_check, blind_coverage_parts, lambda: piece_ctx(post_on_base())),
    "export-hygiene": (ExportHygiene, blind_export_hygiene,
                       lambda: assembly_scenes.with_nodes(assembly_scenes.make_ctx(spacing=1.0))),
    "no-fallback-scan": (NoFallback, blind_no_fallback_scan, manifest_with_sources),
    "no-fallback-lookup": (NoFallback, blind_no_fallback_lookup, manifest_with_sources),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_blind_core_stops_the_run(name, monkeypatch):
    make_check, blind, make = SCENES[name]
    ctx = make()
    run_gate([make_check()], ctx, None)  # the clean scene, with its plants, does not stop
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=make_check().name):
        run_gate([make_check()], ctx, None)
