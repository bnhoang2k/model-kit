"""The guard for the frame and joint checks' plants: a check whose core never
reports a finding cannot see its plant, and the gate stops the run with `BlindCheck`.

Each check below is blinded on a clean scene of its own by patching the function its verdict rests
on (not by subclassing `run`: a plant may call the module's `_findings` and its measures, so only a
patched core blinds the check and its plants alike). A plant kept only when the core catches it
would vanish for an item-by-item check under a blind core, so the run would read `partial` and
stop nowhere. The scenes are the ones the check's own tests plant on.

Two checks build their plants without the core: trace (its plants are taken from the manifest
and the samples) and mirror-exact on a target with a quad to flip. Mirror-exact also needs a plant
for a target with no quad, which would otherwise leave an `Unmeasured` line: the second
mirror-exact scene is a target of one triangle, where a blind core would read `partial`.
"""

import numpy as np
import pytest

from model_kit.core.checks import (
    connectors,
    frames,
    mirror_exact,
    mirror_pair,
    receiver,
    sides,
    trace,
)
from model_kit.core.checks.connectors import Connectors
from model_kit.core.checks.frames import Frames
from model_kit.core.checks.mirror_exact import MirrorExact
from model_kit.core.checks.mirror_pair import MirrorPair
from model_kit.core.checks.receiver import Receiver
from model_kit.core.checks.sides import Sides
from model_kit.core.checks.trace import Trace
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from ..bodies.bar03 import bar03_scene
from . import assembly_scenes, manifest_scenes
from .test_mirror_exact import one_triangle_pair
from .test_trace import make_ctx as trace_ctx


def blind_findings(module):
    def blind(monkeypatch):
        monkeypatch.setattr(module, "_findings", lambda *a, **k: [])
    return blind


def blind_mirror_exact(monkeypatch):
    monkeypatch.setattr(mirror_exact, "_detail", lambda *a, **k: None)


def blind_trace(monkeypatch):
    monkeypatch.setattr(trace, "_stale_gap", lambda job: 0.0)
    monkeypatch.setattr(trace, "_off_surface", lambda ctx, job: None)
    monkeypatch.setattr(trace, "_seat_root", lambda ctx, clip, samples: [
        (0, np.zeros(3), np.zeros(3))])


SCENES = {
    "frames": (Frames, blind_findings(frames), manifest_scenes.make_ctx),
    "mirror-pair": (MirrorPair, blind_findings(mirror_pair), manifest_scenes.make_ctx),
    "sides": (Sides, blind_findings(sides), manifest_scenes.make_ctx),
    "receiver": (Receiver, blind_findings(receiver), assembly_scenes.make_ctx),
    "connectors": (Connectors, blind_findings(connectors), assembly_scenes.make_ctx),
    "mirror-exact": (MirrorExact, blind_mirror_exact, bar03_scene),
    "mirror-exact-one-triangle": (MirrorExact, blind_mirror_exact, one_triangle_pair),
    "trace": (Trace, blind_trace, lambda: trace_ctx(seat="kit:Throne.top")),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_blind_core_stops_the_run(name, monkeypatch):
    check, blind, make = SCENES[name]
    ctx = make()
    assert run_gate([check()], ctx, None).status == "passed"  # the clean scene, with its plants
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=check.name):
        run_gate([check()], ctx, None)
