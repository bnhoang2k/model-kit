from model_kit.core import gate
from model_kit.core.checks.branching import BRANCHING_CHECKS


def test_branching_checks_registered():
    names = [c.name for c in BRANCHING_CHECKS]
    assert names == ["limb-seat", "floor-organic", "crossing-organic", "near-miss-organic",
                     "skeleton-match", "reference-range", "taper", "leaf-angle"]
    for c in BRANCHING_CHECKS:
        assert gate.CHECKS[c.name] is c
        assert c.kinds == frozenset({"branching"})
