import dataclasses

import numpy as np

from model_kit.core.checks.branching._pairs import place_pair
from model_kit.core.checks.branching.near_miss import NearMissOrganic
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    CROSS_OF_RADIUS,
    NEAR_OF_RADIUS,
)

from .branching_scene import grown_ctx, item, limb_index, plants_caught, small_tree, tiny_ctx

CHECK = NearMissOrganic()
PAIR = item("leader.1 x twig.0")


def _placed(share):
    sk = small_tree()
    moved = place_pair(sk, limb_index(sk, "twig.0"), limb_index(sk, "leader.1"), share)
    assert moved is not None
    return grown_ctx(moved)


def test_clean_passes():
    assert CHECK.run(grown_ctx(small_tree())) == []
    assert CHECK.run(tiny_ctx()) == []


def test_plants_caught():
    for ctx in (grown_ctx(small_tree()), tiny_ctx()):
        caught = plants_caught(CHECK, ctx)
        assert [label.split(":")[0] for label, _ in caught] == ["graze"]
        assert all(ok for _, ok in caught), caught


def test_calibration():
    # far enough passes; close enough fails
    assert CHECK.run(_placed(CALIBRATION_FAIL * NEAR_OF_RADIUS), only={PAIR}) == []
    found = CHECK.run(_placed(CALIBRATION_PASS * NEAR_OF_RADIUS), only={PAIR})
    assert [f.item for f in found] == [PAIR]


def test_graze_fails():
    found = CHECK.run(_placed(-CALIBRATION_PASS * CROSS_OF_RADIUS), only={PAIR})
    assert [f.item for f in found] == [PAIR]


def test_deep_crossing_is_not_a_near_miss():
    assert CHECK.run(_placed(-CALIBRATION_FAIL * CROSS_OF_RADIUS), only={PAIR}) == []


def test_a_gap_of_exactly_the_crossing_limit_is_a_near_miss_and_not_a_crossing(monkeypatch):
    from model_kit.core.checks.branching import near_miss

    r = 0.0607
    row = [(0, 1, -CROSS_OF_RADIUS * r, r)]
    monkeypatch.setattr(near_miss, "limb_pairs", lambda *a, **k: row)
    assert len(CHECK.run(grown_ctx(small_tree()))) == 1


def test_leaf_near_branch_passes():
    sk = small_tree()
    # leaf 0 (on the twig) laid with its blade base on the second leader's axis
    q, _, _ = sk.at(limb_index(sk, "leader.1"), 0.5)
    base = np.array(sk.base)
    shift = q - base[0]
    node = np.array(sk.node)
    node[0] += shift
    base[0] = q
    ctx = grown_ctx(dataclasses.replace(sk, base=base, node=node))
    assert CHECK.run(ctx) == []


def test_leaves_listed_once_per_piece():
    assert CHECK.coverage(grown_ctx(small_tree())) == [
        "near-miss-organic: not applicable (assembly piece): grove:Crate",
        "near-miss-organic: not applicable (leaf cards touch in a real crown): grove:Tree"]


def test_no_skeleton_is_unmeasured():
    ctx = grown_ctx(None, listed_from=small_tree())
    assert "near-miss-organic: skipped (no skeleton): grove:Tree" in CHECK.coverage(ctx)
    assert any(isinstance(line, Unmeasured) for line in CHECK.coverage(ctx))
    assert run_gate([CHECK], ctx, None).status == "partial"
