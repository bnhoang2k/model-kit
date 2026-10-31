"""The guard for the clip and rig checks' plants: a check whose verdict never
reports a finding cannot see its plant, and the gate stops the run with `BlindCheck`.

Each case blinds one check on a clean scene of its own by patching the function its verdict rests
on (not `run`: a plant may call those functions to choose what to plant, so only a patched
verdict blinds the check and its plants alike). A plant that the check's own verdict drops would
leave no plant, and the item would read `Unmeasured` (`skipped (no plant can fail)`), so a blind
verdict would read as `partial`, never as a stop. Some cases are caught only through this guard; others keep a
plant whatever the verdict says.
"""

import numpy as np
import pytest

from model_kit.core.checks import bend_quality, clip_hygiene, overlap, seat_fit, stress_poses
from model_kit.core.checks.bend_quality import BendQuality
from model_kit.core.checks.clip_hygiene import ClipHygiene
from model_kit.core.checks.overlap import Overlap
from model_kit.core.checks.seat_fit import SeatFit
from model_kit.core.checks.solver_error import SolverError, _Job
from model_kit.core.checks.stress_poses import StressPoses
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene
from .test_clip_hygiene import make_ctx as hygiene_ctx
from .test_overlap import make_ctx as overlap_ctx
from .test_seat_fit import scene as seat_scene
from .test_solver_error import scene as solver_scene


def blind_hygiene(rule):
    def blind(monkeypatch):
        monkeypatch.setattr(clip_hygiene, rule, lambda *a, **k: None)
    return blind


def hygiene_scene():
    return hygiene_ctx(loop=True)


def blind_seat_fit(monkeypatch):
    monkeypatch.setattr(seat_fit, "_finding", lambda *a, **k: None)


def blind_solver_error(monkeypatch):
    monkeypatch.setattr(_Job, "miss", property(lambda self: 0.0))


def blind_bend_quality(monkeypatch):
    monkeypatch.setattr(bend_quality, "_measure_joint", lambda *a, **k: None)


def blind_overlap(monkeypatch):
    monkeypatch.setattr(overlap, "_measure", lambda *a, **k: overlap._Worst())


def blind_overlap_budgets(monkeypatch):
    """The budget half of the verdict: every point may sink any depth, so nothing is a finding."""
    monkeypatch.setattr(overlap, "_budgets",
                        lambda zones, s, skin, pts, **k: np.full(len(pts), np.inf))


def blind_stress(rule, result):
    def blind(monkeypatch):
        monkeypatch.setattr(stress_poses, rule, lambda *a, **k: result)
    return blind


SCENES = {
    f"clip-hygiene {rule}": (ClipHygiene, blind_hygiene(rule), hygiene_scene)
    for rule in ("_floor", "_slide", "_stand", "_pop", "_prop")
}
SCENES["seat-fit"] = (SeatFit, blind_seat_fit, seat_scene)
SCENES["solver-error"] = (SolverError, blind_solver_error, solver_scene)
SCENES["bend-quality"] = (BendQuality, blind_bend_quality, worm_scene)
SCENES["overlap"] = (Overlap, blind_overlap, overlap_ctx)
SCENES["overlap budgets"] = (Overlap, blind_overlap_budgets, overlap_ctx)
SCENES["stress-poses self"] = (StressPoses, blind_stress("_self", None), worm_scene)
SCENES["stress-poses inverted"] = (StressPoses, blind_stress("_inverted", None), worm_scene)
SCENES["stress-poses C"] = (StressPoses, blind_stress("body_contacts", []), quadruped_scene)


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_blind_verdict_stops_the_run(name, monkeypatch):
    make_check, blind, make = SCENES[name]
    ctx = make()
    run_gate([make_check()], ctx, None)  # the clean scene, with its plants, does not stop
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=make_check().name):
        run_gate([make_check()], ctx, None)
