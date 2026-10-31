"""The guard for the mesh-look checks' plants: a check whose core never reports a
finding cannot see its plant, and the gate stops the run with `BlindCheck`.

Each case blinds one rule of a check on a clean scene of its own by patching the function its
verdict rests on (not by subclassing `run`: a plant may call `run` and the module functions to
choose what to plant, so only a patched core blinds the check and its plants alike). A plant kept
only when the core catches it would vanish under a blind core, so near crease and detail would
read `partial` with a "no plant can fail" line and stop nowhere.

`done` builds plain plants from the manifest, so a blind `done` fails the gate by construction.
Near crease and detail need the guard (without it there is no `BlindCheck`).
"""

import numpy as np
import pytest

from model_kit.core.checks import detail, done, near_crease
from model_kit.core.checks.detail import Detail
from model_kit.core.checks.done import Done
from model_kit.core.checks.near_crease import NearCrease
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from .manifest_scenes import make_ctx as manifest_ctx
from .sight_scenes import viewed
from .test_detail import leg
from .test_near_crease import humanoid, lathe_parts, scene


def blind_near(monkeypatch):
    monkeypatch.setattr(near_crease, "_near", lambda turn, *a, **k: np.zeros(len(turn), bool))


def blind_dash(monkeypatch):
    monkeypatch.setattr(near_crease, "line_ends", lambda *a, **k: np.zeros(0, dtype=int))


def blind_drop(monkeypatch):
    """The check's rule for what a viewer would not see drops every line end."""
    monkeypatch.setattr(near_crease, "_dropped", lambda *a, **k: "unseen")


def blind_bend(monkeypatch):
    monkeypatch.setattr(near_crease, "_bend_findings", lambda *a, **k: [])


def blind_blocky(monkeypatch):
    monkeypatch.setattr(detail, "_blocky", lambda *a, **k: False)


def blind_wasted(monkeypatch):
    monkeypatch.setattr(detail, "_wasted", lambda *a, **k: False)


def blind_done_text(monkeypatch):
    monkeypatch.setattr(done, "_done_empty", lambda piece: False)


def blind_size_class(monkeypatch):
    monkeypatch.setattr(done, "_size_unknown", lambda piece: False)


SCENES = {
    "near-crease rule A at rest": (NearCrease, blind_near, lambda: scene(lathe_parts(), bits=14)),
    "near-crease rule B": (NearCrease, blind_dash, lambda: scene(lathe_parts())),
    "near-crease drop rule": (NearCrease, blind_drop,
                              lambda: viewed(scene(lathe_parts()), 1e-9)),
    "near-crease rule A bending": (NearCrease, blind_bend, humanoid),
    "detail rule 1": (Detail, blind_blocky, lambda: leg(32)[0]),
    "detail rule 2": (Detail, blind_wasted, lambda: leg(32)[0]),
    "done text": (Done, blind_done_text, manifest_ctx),
    "done size class": (Done, blind_size_class, manifest_ctx),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_blind_core_stops_the_run(name, monkeypatch):
    make_check, blind, make = SCENES[name]
    ctx = make()
    run_gate([make_check()], ctx, None)  # the clean scene, with its plants, does not stop
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=make_check().name):
        run_gate([make_check()], ctx, None)
