import dataclasses

import numpy as np
import pytest

from model_kit.core.checks.branching.limb_seat import LimbSeat
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    SEAT_CHILD_OF_PARENT_RADIUS,
    SEAT_OF_PARENT_RADIUS,
)

from .branching_scene import (
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

CHECK = LimbSeat()


def _slid(sk, name, share):
    """The limb moved off its parent's axis by share x the parent's radius at its seat."""
    i = limb_index(sk, name)
    p = int(sk.parent[i])
    _, r, tangent = sk.at(p, float(sk.t[i]))
    side = np.cross(tangent, [0.0, 0.0, 1.0])
    side /= np.linalg.norm(side)
    return moved_limb(sk, name, sk.limb_points(i) + side * share * r)


def test_clean_passes():
    assert CHECK.run(grown_ctx(small_tree())) == []
    assert CHECK.run(tiny_ctx()) == []


def test_plants_caught():
    for ctx in (grown_ctx(small_tree()), tiny_ctx()):
        caught = plants_caught(CHECK, ctx)
        assert [label.split(":")[0] for label, _ in caught] == ["off axis", "blade far"]
        assert all(ok for _, ok in caught), caught


def test_a_tree_without_leaves_gets_only_the_limb_plant():
    ctx = grown_ctx(make_skeleton(small_limbs(), []))
    caught = plants_caught(CHECK, ctx)
    assert [label.split(":")[0] for label, _ in caught] == ["off axis"]
    assert all(ok for _, ok in caught)


def test_calibration_base_off_axis():
    sk, only = small_tree(), {item("branch.0")}  # the twig on it moves off its seat too
    i = limb_index(sk, "branch.0")
    radii = np.array(sk.radii)
    radii[sk.start[i]:sk.start[i + 1]] = 0.001  # thin, so its ring stays inside the bark
    sk = dataclasses.replace(sk, radii=radii)
    passing = _slid(sk, "branch.0", CALIBRATION_PASS * SEAT_OF_PARENT_RADIUS)
    assert CHECK.run(grown_ctx(passing), only=only) == []
    failing = _slid(sk, "branch.0", CALIBRATION_FAIL * SEAT_OF_PARENT_RADIUS)
    assert {f.item for f in CHECK.run(grown_ctx(failing), only=only)} == {item("branch.0")}


def test_calibration_base_ring_pokes_out():
    """The room is SEAT_CHILD_OF_PARENT_RADIUS x the parent's radius less the base radius."""
    sk, only = small_tree(), {item("branch.0")}
    i = limb_index(sk, "branch.0")
    _, r = sk.nearest(int(sk.parent[i]), sk.limb_points(i)[:1])
    room = SEAT_CHILD_OF_PARENT_RADIUS * float(r[0]) - float(sk.limb_radii(i)[0])
    for share, bad in ((CALIBRATION_PASS, False), (CALIBRATION_FAIL, True)):
        moved = _slid(sk, "branch.0", share * room / float(r[0]))
        found = CHECK.run(grown_ctx(moved), only=only)
        assert bool(found) == bad
        assert all("pokes out" in f.detail for f in found)


def test_calibration_base_radius():
    sk = small_tree()
    i = limb_index(sk, "twig.0")
    _, r = sk.nearest(int(sk.parent[i]), sk.limb_points(i)[:1])
    for share, bad in ((CALIBRATION_PASS, False), (CALIBRATION_FAIL, True)):
        radii = np.array(sk.radii)
        radii[sk.start[i]] = share * SEAT_CHILD_OF_PARENT_RADIUS * r[0]
        found = CHECK.run(grown_ctx(dataclasses.replace(sk, radii=radii)))
        assert [f.item for f in found] == ([item("twig.0")] if bad else [])


def _leaf_moved(sk, f, base=None, node=None):
    b, n = np.array(sk.base), np.array(sk.node)
    if base is not None:
        b[f] = base
    if node is not None:
        n[f] = node
    return dataclasses.replace(sk, base=b, node=n)


def test_blade_far_from_node_fails():
    sk = small_tree()
    out = (sk.base[0] - sk.node[0]) / np.linalg.norm(sk.base[0] - sk.node[0])
    far = sk.node[0] + out * 5 * sk.petiole[0]
    assert [f.item for f in CHECK.run(grown_ctx(_leaf_moved(sk, 0, base=far)))] == [item("leaf.0")]


@pytest.mark.parametrize("lite", [False, True])
def test_calibration_blade_base(lite):
    """The blade base may sit its petiole (none on a lite tree) plus the host's radius from its
    node."""
    sk = dataclasses.replace(small_tree(), lite=lite)
    _, r = sk.nearest(int(sk.host[0]), sk.node[:1])
    out = (sk.base[0] - sk.node[0]) / np.linalg.norm(sk.base[0] - sk.node[0])
    allowed = 0.0 if lite else float(sk.petiole[0])
    for share, bad in ((CALIBRATION_PASS, False), (CALIBRATION_FAIL, True)):
        base = sk.node[0] + out * (allowed + share * SEAT_OF_PARENT_RADIUS * float(r[0]))
        found = CHECK.run(grown_ctx(_leaf_moved(sk, 0, base=base)), only={item("leaf.0")})
        assert [f.item for f in found] == ([item("leaf.0")] if bad else [])


def test_leaf_node_off_host_axis():
    sk = small_tree()
    host = int(sk.host[1])
    _, r, tangent = sk.at(host, float(sk.leaf_t[1]))
    side = np.cross(tangent, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    for share, bad in ((CALIBRATION_PASS, False), (CALIBRATION_FAIL, True)):
        moved = _leaf_moved(sk, 1, node=sk.node[1] + side * share * SEAT_OF_PARENT_RADIUS * r,
                            base=sk.base[1] + side * share * SEAT_OF_PARENT_RADIUS * r)
        assert [f.item for f in CHECK.run(grown_ctx(moved))] == ([item("leaf.1")] if bad else [])


def test_only_measures_named_items():
    sk = _slid(small_tree(), "branch.0", CALIBRATION_FAIL)
    assert CHECK.run(grown_ctx(sk), only={item("root.0")}) == []
    assert {f.item for f in CHECK.run(grown_ctx(sk), only={item("branch.0")})} == {
        item("branch.0")}


def test_no_skeleton_is_unmeasured():
    ctx = grown_ctx(None, listed_from=small_tree())
    lines = CHECK.coverage(ctx)
    assert "limb-seat: skipped (no skeleton): grove:Tree" in lines
    assert any(isinstance(line, Unmeasured) for line in lines)
    assert CHECK.run(ctx) == []
    assert run_gate([CHECK], ctx, None).status == "partial"


def test_assembly_piece_listed_in_coverage():
    lines = CHECK.coverage(grown_ctx(small_tree()))
    assert lines == ["limb-seat: not applicable (assembly piece): grove:Crate"]
    assert not any(isinstance(line, Unmeasured) for line in lines)


def test_kinds():
    assert CHECK.kinds == frozenset({"branching"})
    assert run_gate([CHECK], grown_ctx(small_tree()), None).status == "passed"
