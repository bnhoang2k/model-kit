"""The whole-check guard: every registered check, blinded, stops the run.

The per-verdict guards (`test_*_plants_blind.py`) patch one rule of one check. This one patches the
whole check: `run` finds nothing, on the real build and on every planted copy. Then each plant unit
that holds a candidate must raise `BlindCheck`, because no candidate is caught. A check with no
such unit on any test scene is named and fails: a check the meta-gate cannot test is the blind
spot this guard exists to find. The test never skips a check. It takes its scenes from the
per-verdict guards, the scenes their checks' own tests plant on.
"""

import pytest

from model_kit.core import gate
from model_kit.core.checks.coverage import Coverage
from model_kit.core.checks.floaters import Floaters
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from .checks import (
    test_assembly_plants_blind,
    test_clip_rig_plants_blind,
    test_frame_joint_plants_blind,
    test_mesh_look_plants_blind,
    test_reach_contact_plants_blind,
)
from .checks.branching import test_branching_plants_blind


def make_check(name):
    """The check as the gate builds it. Coverage reads the checks of its run, so it gets one."""
    return Coverage([Floaters()]) if name == Coverage.name else gate.CHECKS[name]()


def _scenes() -> dict[str, list]:
    """The scene factories of the per-verdict guards, by the name of the check they plant."""
    tables = (test_assembly_plants_blind.SCENES, test_clip_rig_plants_blind.SCENES,
              test_frame_joint_plants_blind.SCENES, test_mesh_look_plants_blind.SCENES,
              test_reach_contact_plants_blind.SCENES, test_branching_plants_blind.CASES)
    by_check: dict[str, list] = {}
    for table in tables:
        for check, _blind, make in table.values():
            by_check.setdefault(check.name, []).append(make)
    for check, pieces in test_branching_plants_blind.TWO_PIECES.values():
        by_check.setdefault(check.name, []).extend(pieces.values())
    return by_check


SCENES = _scenes()


def _units(check, ctx):
    """The units of `check` on `ctx` that hold a candidate (the ones the gate tries)."""
    return [unit for unit in check.plants(ctx) if unit.candidates]


def _scene_with_a_unit(name):
    """The first test scene on which the check named `name` has a unit to plant, or None."""
    check = make_check(name)
    for make in SCENES.get(name, []):
        ctx = make()
        if _units(check, ctx):
            return ctx
    return None


def test_every_check_has_a_scene_where_it_plants():
    unplanted = [name for name in gate.CHECKS if _scene_with_a_unit(name) is None]
    assert not unplanted, f"no test scene gives these checks a unit to plant: {unplanted}"


@pytest.mark.parametrize("name", list(gate.CHECKS))
def test_every_check_blinded_raises(name, monkeypatch):
    check = make_check(name)
    scene = _scene_with_a_unit(name)
    assert scene is not None, f"{name}: no test scene gives it a unit to plant"
    monkeypatch.setattr(type(check), "run", lambda self, ctx, only=None, samples=frozenset(): [])
    with pytest.raises(BlindCheck, match=name):
        run_gate([check], scene, None)
