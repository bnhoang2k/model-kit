import dataclasses

import numpy as np

from model_kit.core.checks.branching.floor import FloorOrganic
from model_kit.core.grow.mesh import leaf_tris
from model_kit.core.grow.skeleton import ring_lows
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    FLOOR_OF_THICKNESS,
    ROOTED_DEPTH_OF_RADIUS,
)

from ..piece_scenes import caught
from .branching_scene import (
    MIN_THICKNESS_M,
    UPM,
    grown_ctx,
    item,
    limb_index,
    make_skeleton,
    moved_limb,
    plants_caught,
    small_limbs,
    small_tree,
    tiny_ctx,
)

CHECK = FloorOrganic()
# The small tree's tube thickness base: its twig row (twice a 0.0045 u radius: 0.018 m) meets the
# dial.
BASE = MIN_THICKNESS_M * UPM
TOL = FLOOR_OF_THICKNESS * BASE


def _sunk_branch(sk, depth):
    """branch.0 moved down so its lowest point minus its radius sits at -depth."""
    i = limb_index(sk, "branch.0")
    pts, radii = sk.limb_points(i), sk.limb_radii(i)
    low = float((pts[:, 1] - radii).min())
    return moved_limb(sk, "branch.0", pts - np.array([0.0, low + depth, 0.0]))


def test_clean_passes():
    assert CHECK.run(grown_ctx(small_tree())) == []
    assert CHECK.run(tiny_ctx()) == []


def test_plants_caught():
    for ctx in (grown_ctx(small_tree()), tiny_ctx()):
        caught = plants_caught(CHECK, ctx)
        assert [label.split(":")[0] for label, _ in caught] == ["lift", "sunk", "sink leaf"]
        assert all(ok for _, ok in caught), caught


def test_calibration_free_limb():
    sk = small_tree()
    assert CHECK.run(grown_ctx(_sunk_branch(sk, CALIBRATION_PASS * TOL))) == []
    found = CHECK.run(grown_ctx(_sunk_branch(sk, CALIBRATION_FAIL * TOL)))
    assert item("branch.0") in [f.item for f in found]


def test_calibration_leaf():
    sk = small_tree()
    low = float(leaf_tris(sk, 2)[..., 1].min())
    for depth, bad in ((CALIBRATION_PASS * TOL, False), (CALIBRATION_FAIL * TOL, True)):
        drop = np.array([0.0, low + depth, 0.0])
        node, base = np.array(sk.node), np.array(sk.base)
        node[2] -= drop
        base[2] -= drop
        found = CHECK.run(grown_ctx(dataclasses.replace(sk, node=node, base=base)))
        assert [f.item for f in found] == ([item("leaf.2")] if bad else [])


def test_rooted_base_above_soil_fails():
    sk = small_tree()
    lifted = dataclasses.replace(sk, pts=sk.pts + np.array([0.0, 0.06, 0.0]))
    found = [f.item for f in CHECK.run(grown_ctx(lifted))]
    assert item("trunk") in found


def test_calibration_rooted_dip_past_base():
    """The trunk comes back down below the soil past the stub that may stand in it,
    while its base stays honest. The trunk is upright there, so its ring lies at the point's own
    height: the point goes to -depth."""
    sk, only = small_tree(), {item("trunk")}
    i = limb_index(sk, "trunk")
    for depth, bad in ((CALIBRATION_PASS * TOL, False), (CALIBRATION_FAIL * TOL, True)):
        pts = sk.limb_points(i).copy()
        pts[2, 1] = -depth
        assert ring_lows(pts, sk.limb_radii(i))[2] == -depth
        found = CHECK.run(grown_ctx(moved_limb(sk, "trunk", pts)), only=only)
        assert [f.item for f in found] == ([item("trunk")] if bad else [])
        assert all("past its base it dips" in f.detail for f in found)


def test_a_short_first_segment_stands_in_the_soil():
    """The rooted dip is read by arc length. The trunk's second point 5 mm along its arc
    is still in the stub that may stand in the soil (the base may sink 1.2 x its radius), so its
    ring below the soil is not a dip. Read by index (`pts[1:]`) it would be one."""
    sk, only = small_tree(), {item("trunk")}
    i = limb_index(sk, "trunk")
    pts = sk.limb_points(i).copy()
    pts[1] = pts[0] + np.array([0.0, 0.005, 0.0])
    assert pts[1, 1] < -TOL  # below the soil by more than the tolerance
    assert ring_lows(pts, sk.limb_radii(i))[1] < -TOL
    assert ROOTED_DEPTH_OF_RADIUS * sk.limb_radii(i)[0] > 0.005  # inside the stub
    assert CHECK.run(grown_ctx(moved_limb(sk, "trunk", pts)), only=only) == []


def test_ring_lows_read_the_ring_not_the_radius():
    """A ring of radius r on a unit direction d reaches r sqrt(1 - d_y^2) below its point: an
    upright limb's rings lie at their points, a level one's a whole r below, a 45 degree one's
    r / sqrt(2) below; at a peak the mesh's ring is level and reaches the whole r."""
    r = np.array([0.1, 0.1, 0.1])
    upright = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
    assert np.array_equal(ring_lows(upright, r), upright[:, 1])
    level = np.array([[0.0, 0.5, 0.0], [1.0, 0.5, 0.0], [2.0, 0.5, 0.0]])
    assert np.allclose(ring_lows(level, r), 0.4)
    slope = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    assert np.allclose(ring_lows(slope, r), slope[:, 1] - 0.1 / np.sqrt(2.0))
    peak = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    assert np.allclose(ring_lows(peak, r), [-0.1 / np.sqrt(2.0), 0.9, -0.1 / np.sqrt(2.0)])


def test_upright_trunk_standing_at_the_soil_passes():
    """The trunk's second point at y = 0, upright: its ring lies at 0, though the point less its
    radius would sit far below -tol."""
    sk, only = small_tree(), {item("trunk")}
    i = limb_index(sk, "trunk")
    pts = sk.limb_points(i).copy()
    pts[1, 1] = 0.0
    assert pts[1, 1] - sk.limb_radii(i)[1] < -CALIBRATION_FAIL * TOL
    assert CHECK.run(grown_ctx(moved_limb(sk, "trunk", pts)), only=only) == []


def test_tilted_rooted_ring_below_the_soil_fails():
    """The trunk's second point above the soil, but on a tilted limb: its ring dips below -tol."""
    sk, only = small_tree(), {item("trunk")}
    i = limb_index(sk, "trunk")
    pts = sk.limb_points(i).copy()
    pts[1] = [0.3, 0.01, 0.0]
    assert ring_lows(pts, sk.limb_radii(i))[1] < -CALIBRATION_FAIL * TOL
    found = CHECK.run(grown_ctx(moved_limb(sk, "trunk", pts)), only=only)
    assert [f.item for f in found] == [item("trunk")]
    assert "past its base it dips" in found[0].detail


def test_buried_tip_above_soil_fails():
    sk = small_tree()
    i = limb_index(sk, "root.0")
    pts = sk.limb_points(i).copy()
    pts[-1, 1] = 0.01
    assert [f.item for f in CHECK.run(grown_ctx(moved_limb(sk, "root.0", pts)))] == [
        item("root.0")]


def test_no_skeleton_is_unmeasured():
    ctx = grown_ctx(None, listed_from=small_tree())
    assert "floor-organic: skipped (no skeleton): grove:Tree" in CHECK.coverage(ctx)
    assert any(isinstance(line, Unmeasured) for line in CHECK.coverage(ctx))
    assert run_gate([CHECK], ctx, None).status == "partial"


def test_assembly_piece_listed_in_coverage():
    assert CHECK.coverage(grown_ctx(small_tree())) == [
        "floor-organic: not applicable (assembly piece): grove:Crate"]


def test_a_free_limb_hidden_in_its_seat_is_sunk_clear_of_it():
    """A tree with no leaves sinks its lowest free limb. Its lowest point may lie in the seat part
    (buried in the parent's wood), which the check leaves out: sunk to that point alone the limb
    still stands clear of the soil past the seat (a plant that sinks only to that point is caught on no leafless tree). The
    second candidate sinks the limb until its highest point is below the soil."""
    ctx = grown_ctx(make_skeleton(small_limbs(), []))
    unit = next(u for u in CHECK.plants(ctx) if "sink limb" in u.candidates[0].label)
    assert [caught(CHECK, p) for p in unit.candidates] == [False, True]
    assert run_gate([CHECK], ctx, None).status == "passed"
