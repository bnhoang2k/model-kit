"""The guard for the assembly checks' plants: a check whose core never reports a
finding cannot see its plant, and the gate stops the run with `BlindCheck`.

Each check below is blinded on a clean scene of its own by patching the function its verdict rests
on (not by subclassing `run`: a plant may call fresh instances and module functions, so only a
patched core blinds the check and its plants alike). A plant kept only when the core catches it
would vanish under a blind core, so the run would read `partial` and stop nowhere. The scenes are the ones the check's own tests plant on.
"""

import numpy as np
import pytest

from model_kit.core.checks import (
    bearing,
    clearance,
    coplanar,
    crossings,
    floaters,
    inset_seat,
    markings,
    markings_flush,
    take_apart,
)
from model_kit.core.checks.bearing import Bearing
from model_kit.core.checks.clearance import Clearance
from model_kit.core.checks.coplanar import Coplanar
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.inset_seat import InsetSeat
from model_kit.core.checks.markings import Markings
from model_kit.core.checks.markings_flush import MarkingsFlush
from model_kit.core.checks.take_apart import TakeApart
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from .bearing_scenes import PUSHED_PIECES, pushed_post, stack
from .furniture_scenes import furniture_ctx
from .piece_scenes import make_ctx, post_on_base
from .support_scenes import bench_ctx, pallet_ctx
from .test_clearance import shop_with_lamp_resting
from .test_coplanar import lifted_mark_scene
from .test_inset_seat import thing_scene
from .test_markings_flush import MM, base_and_cap
from .test_take_apart import lane


def blind_floaters(monkeypatch):
    monkeypatch.setattr(floaters, "_problem", lambda *a, **k: None)


def blind_crossings(monkeypatch):
    monkeypatch.setattr(crossings, "measure", lambda a, b: crossings.Cross(0.0, 0.0, 0.0))


def blind_take_apart(monkeypatch):
    monkeypatch.setattr(take_apart, "measure", lambda a, b: crossings.Cross(0.0, 0.0, 0.0))


def blind_markings(monkeypatch):
    monkeypatch.setattr(markings, "_rule1", lambda *a, **k: iter(()))
    monkeypatch.setattr(markings, "_rule2", lambda *a, **k: iter(()))


def blind_markings_flush(monkeypatch):
    monkeypatch.setattr(markings_flush, "_is_flush", lambda *a, **k: False)


def blind_clearance(monkeypatch):
    monkeypatch.setattr(clearance._Scan, "_fault", lambda *a, **k: None)


def blind_inset_seat(monkeypatch):
    monkeypatch.setattr(inset_seat, "_read", lambda *a, **k: inset_seat._Reading())


def blind_coplanar(monkeypatch):
    monkeypatch.setattr(coplanar, "_shared_plane", lambda *a, **k: (0.0, np.zeros((0, 3))))


def blind_bearing(monkeypatch):
    monkeypatch.setattr(bearing, "measure", lambda *a, **k: [])


SCENES = {
    "floaters": (Floaters, blind_floaters,
                 lambda: make_ctx(post_on_base(), {"post": {"on": "base"}})),
    "floaters (pallet)": (Floaters, blind_floaters, pallet_ctx),
    "floaters (bench)": (Floaters, blind_floaters, bench_ctx),
    "crossings": (Crossings, blind_crossings, lambda: make_ctx(post_on_base())),
    "take-apart": (TakeApart, blind_take_apart, lambda: lane(("anchor", "b", "a"))),
    "markings": (Markings, blind_markings, lambda: furniture_ctx("Label")),
    "markings-flush": (MarkingsFlush, blind_markings_flush, lambda: base_and_cap(5 * MM)),
    "clearance": (Clearance, blind_clearance, shop_with_lamp_resting),
    "inset-seat": (InsetSeat, blind_inset_seat, thing_scene),
    "bearing": (Bearing, blind_bearing, stack),
    "bearing (pallet)": (Bearing, blind_bearing, pallet_ctx),
    "bearing (bench)": (Bearing, blind_bearing, bench_ctx),
    "coplanar": (Coplanar, blind_coplanar, lifted_mark_scene),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_blind_core_stops_the_run(name, monkeypatch):
    check, blind, make = SCENES[name]
    ctx = make()
    assert run_gate([check()], ctx, None).status == "passed"  # the clean scene, with its plants
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=name.split(" (")[0]):
        run_gate([check()], ctx, None)


@pytest.mark.parametrize("piece", sorted(PUSHED_PIECES))
def test_a_blind_sink_rule_stops_the_run(piece, monkeypatch):
    """The guard above blinds bearing as a whole; this one blinds only rule 5, so its plant is the
    one that must be caught: a sink rule that judges every seat on the surface stops the gate."""
    ctx = pushed_post(piece, 0.0)
    assert run_gate([Bearing()], ctx, None).status == "passed"
    monkeypatch.setitem(bearing._JUDGE, bearing.SINK,
                        lambda p, part, entry: bearing.Entry(bearing.SINK, part.name, "pass",
                                                             plan=entry.plan))
    with pytest.raises(BlindCheck, match="sinks into"):
        run_gate([Bearing()], ctx, None)
