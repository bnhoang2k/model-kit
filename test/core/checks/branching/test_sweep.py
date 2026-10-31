"""The class sweep: honest trees across the bud grower's range pass the branching checks, and each
check's plants are caught on every one of them.

The bud grower (`buds.py`): the two tiny species (two-ranked with no terminal bud, spiral with
one), seeds 1 to 8, at 0.5, 1.0 and 1.5 m full and 1.5 m lite (`bud_sweep_cases`), against the
five kept checks. A downstream check only: a finding here is a grower defect, fixed by a general
rule for every species in `core/grow/buds.py`, never by moving a tolerance. The tolerances come
from the four measured trees and their sources (`core/tolerances.py`, `core/grow/shape.py`).
"""

import pytest

from model_kit.core.checks.branching import (
    CrossingOrganic,
    FloorOrganic,
    LimbSeat,
    NearMissOrganic,
    SkeletonMatch,
)
from model_kit.core.checks.branching._pairs import limb_pairs
from model_kit.core.tolerances import CALIBRATION_FAIL, NEAR_OF_RADIUS, SEAT_ZONE_OPENS_OF_RADIUS

from .branching_scene import (
    bud_case_label,
    bud_ctx,
    bud_sweep_cases,
    bud_tree,
    plants_caught,
)

# The five checks that judge any skeleton. The plant checks (also in BRANCHING_CHECKS) have
# their own tests.
KEPT = (LimbSeat, FloorOrganic, CrossingOrganic, NearMissOrganic, SkeletonMatch)

# --- the bud grower ----------------------------------------------------------------------------


def _id(case) -> str:
    return bud_case_label(*case).replace(" ", "_")


@pytest.mark.parametrize("case", bud_sweep_cases(), ids=_id)
def test_new_grower_sweep_is_clean(case):
    ctx = bud_ctx(*case)
    for make in KEPT:
        check = make()
        assert check.run(ctx) == [], check.name


@pytest.mark.parametrize("case", bud_sweep_cases(), ids=_id)
def test_new_grower_plants_are_caught(case):
    ctx = bud_ctx(*case)
    for make in KEPT:
        check = make()
        caught = plants_caught(check, ctx)
        # Every check places a plant on every tree, even a seedling with no twig on an unrelated
        # limb (crossing and near-miss then move another limb against another).
        assert caught, check.name
        assert all(ok for _, ok in caught), (check.name, caught)


@pytest.mark.parametrize("case", bud_sweep_cases(), ids=_id)
def test_new_grower_gaps_clear_the_calibration(case):
    """Past their seat zones no two limbs come within CALIBRATION_FAIL x NEAR_OF_RADIUS."""
    sk = bud_tree(*case)
    reach = CALIBRATION_FAIL * NEAR_OF_RADIUS
    shares = [gap / r for _, _, gap, r in limb_pairs(sk, reach=reach,
                                                      opens=SEAT_ZONE_OPENS_OF_RADIUS)]
    assert all(share > reach for share in shares), min(shares)
