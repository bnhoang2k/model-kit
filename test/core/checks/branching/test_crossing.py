import numpy as np

from model_kit.core.checks.branching._pairs import place_pair
from model_kit.core.checks.branching.crossing import CrossingOrganic
from model_kit.core.checks.branching.near_miss import NearMissOrganic
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    CROSS_OF_RADIUS,
    NEAR_OF_RADIUS,
)

from .branching_scene import (
    Limb,
    grown_ctx,
    item,
    limb_index,
    line,
    make_skeleton,
    plants_caught,
    seat,
    small_leaves,
    small_limbs,
    small_tree,
    tiny_ctx,
)

CHECK = CrossingOrganic()


def test_clean_passes():
    assert CHECK.run(grown_ctx(small_tree())) == []
    assert CHECK.run(tiny_ctx()) == []


def test_plants_caught():
    for ctx in (grown_ctx(small_tree()), tiny_ctx()):
        caught = plants_caught(CHECK, ctx)
        assert [label.split(":")[0] for label, _ in caught] == ["through"]
        assert all(ok for _, ok in caught), caught


def test_calibration():
    sk = small_tree()
    x, y = limb_index(sk, "twig.0"), limb_index(sk, "leader.1")
    pair = item("leader.1 x twig.0")
    for share, bad in ((CALIBRATION_PASS, False), (CALIBRATION_FAIL, True)):
        moved = place_pair(sk, x, y, -share * CROSS_OF_RADIUS)
        assert moved is not None
        assert [f.item for f in CHECK.run(grown_ctx(moved), only={pair})] == (
            [pair] if bad else [])


def test_twig_back_through_parent_fails():
    limbs = small_limbs()
    start = seat(limbs, 4, 0.6)
    # out from the branch, over 5 cm along it, then back down through it and out the far side
    # (the way down is beside the way up: a twig that retraced its own line would cross itself)
    pts = start + np.array([[0.0, 0.0, 0.0], [0.0, 0.1, 0.0], [-0.05, 0.1, 0.01],
                            [-0.05, -0.05, 0.01], [-0.05, -0.1, 0.01]])
    limbs[5] = Limb("twig.0", 3, 4, 0.6, pts, np.linspace(0.0055, 0.0045, 5))
    sk = make_skeleton(limbs, small_leaves(limbs))
    assert [f.item for f in CHECK.run(grown_ctx(sk))] == [item("branch.0 x twig.0")]


def test_siblings_at_fork_pass():
    limbs = small_limbs()[:2]
    top = seat(limbs, 0, 0.97)
    tilt = np.radians(11.6)  # three directions this far off vertical lie 20 degrees apart
    for k in range(3):
        az = np.radians(120 * k)
        d = np.array([np.sin(tilt) * np.cos(az), np.cos(tilt), np.sin(tilt) * np.sin(az)])
        limbs.append(Limb(f"leader.{k}", 1, 0, 0.97, line(top, d, 0.8, 6),
                          np.linspace(0.022, 0.008, 7)))
    sk = make_skeleton(limbs, [])
    dirs = [sk.limb_points(i)[-1] - top for i in (2, 3, 4)]
    cos = np.dot(dirs[0], dirs[1]) / np.linalg.norm(dirs[0]) / np.linalg.norm(dirs[1])
    assert abs(np.degrees(np.arccos(cos)) - 20) < 0.5
    ctx = grown_ctx(sk)
    assert CHECK.run(ctx) == []
    assert NearMissOrganic().run(ctx) == []


def test_leaves_listed_once_per_piece():
    lines = CHECK.coverage(grown_ctx(small_tree()))
    assert lines == ["crossing-organic: not applicable (assembly piece): grove:Crate",
                     "crossing-organic: not applicable (leaf cards have no inside): grove:Tree"]


def test_no_skeleton_is_unmeasured():
    ctx = grown_ctx(None, listed_from=small_tree())
    assert "crossing-organic: skipped (no skeleton): grove:Tree" in CHECK.coverage(ctx)
    assert any(isinstance(line, Unmeasured) for line in CHECK.coverage(ctx))
    assert run_gate([CHECK], ctx, None).status == "partial"


def _read(monkeypatch, gap, r):
    """The findings of both checks on a tree whose pair search reports one pair at this gap."""
    from model_kit.core.checks.branching import crossing, near_miss

    ctx = grown_ctx(small_tree())
    row = [(0, 1, gap, r)]
    monkeypatch.setattr(crossing, "limb_pairs", lambda *a, **k: row)
    monkeypatch.setattr(near_miss, "limb_pairs", lambda *a, **k: row)
    return CHECK.run(ctx), NearMissOrganic().run(ctx)


def test_crossing_and_near_miss_meet_at_one_boundary(monkeypatch):
    """The two checks split every gap up to the near-miss reach between them: one reports it,
    never both and never neither. At exactly -CROSS x r the pair is at the limit and not over
    it, so it is a near miss."""
    for r in (0.0607, 0.0055, 0.3, 1.0 / 3.0):
        edge = -CROSS_OF_RADIUS * r
        for gap, who in ((edge - 1e-9, "crossing"), (edge, "near"), (edge + 1e-9, "near"),
                         (NEAR_OF_RADIUS * r, "near"), (NEAR_OF_RADIUS * r + 1e-9, None)):
            crossing, near = _read(monkeypatch, gap, r)
            assert (len(crossing), len(near)) == {"crossing": (1, 0), "near": (0, 1),
                                                  None: (0, 0)}[who], (r, gap)
