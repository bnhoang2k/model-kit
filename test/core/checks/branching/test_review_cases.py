"""Hand-built trees, each with the verdict the
seat zones, the limb-seat, floor and skeleton-match items give.

Each case runs all five branching checks and names exactly the checks that fail and the items
they name; every other check passes."""

import dataclasses
import json

import numpy as np
import pytest

from model_kit.core.checks.branching import BRANCHING_CHECKS
from model_kit.core.checks.branching.skeleton_match import SkeletonMatch
from model_kit.core.grow.mesh import grown_parts
from model_kit.core.grow.rules import KIT_RULES
from model_kit.core.grow.skeleton import DOME, limb_at, unit
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import piece_key
from model_kit.core.tolerances import SEAT_ZONE_OPENS_OF_RADIUS

from .branching_scene import (
    PACK,
    PIECE,
    UP,
    Limb,
    bud_tree,
    grown_ctx,
    limb_index,
    line,
    make_skeleton,
    seat,
    small_leaves,
    small_limbs,
    small_tree,
)

BRANCH, TWIG = 4, 5
TWIG_R = np.linspace(0.0055, 0.0045, 5)


def verdict(sk, parts=None) -> dict[str, set[str]]:
    """{check name: the items it names} for every check that fails, item without the piece."""
    ctx = grown_ctx(sk, parts=parts)
    out = {}
    for make in BRANCHING_CHECKS:
        check = make()
        found = {f.item.split(" ", 1)[1] for f in check.run(ctx)}
        if found:
            out[check.name] = found
    return out


def tree(limbs) -> object:
    return make_skeleton(limbs, small_leaves(limbs))


def _on_branch(t):
    limbs = small_limbs()
    q, r, d = limb_at(limbs[BRANCH].pts, limbs[BRANCH].radii, t)
    return limbs, q, r, d, unit(np.cross(d, UP))


# --- the seat zone ------------------------------------------------------------------------


def test_fused_siblings_fail_crossing():
    """Two twigs from one seat 1 degree apart for their whole length."""
    limbs = small_limbs()
    q = seat(limbs, BRANCH, 0.6)
    d0 = unit(np.array([-0.3, 0.6, 0.8]))
    d1 = unit(d0 + np.tan(np.radians(1.0)) * unit(np.cross(d0, UP)))
    limbs[TWIG] = Limb("twig.0", 3, BRANCH, 0.6, line(q, d0, 0.15), TWIG_R)
    limbs.append(Limb("twig.1", 3, BRANCH, 0.6, line(q, d1, 0.15), TWIG_R))
    assert verdict(tree(limbs)) == {"crossing-organic": {"twig.0 x twig.1"}}


def test_twig_through_its_own_branch_fails_crossing():
    """Out on one side, then through the branch to the far side."""
    limbs, q, big_r, d, n = _on_branch(0.6)
    out = big_r + 1.6 * 0.0055
    limbs[TWIG].pts = np.array([q, q + n * out + d * 0.010, q - n * out + d * 0.020,
                                q - n * (big_r + 1.8 * 0.0055) + d * 0.040,
                                q - n * (big_r + 1.9 * 0.0055) + d * 0.060])
    assert verdict(tree(limbs)) == {"crossing-organic": {"branch.0 x twig.0"}}


def test_twig_weaving_through_its_branch_fails_crossing():
    """A twig poking out of each side of its branch in turn."""
    limbs = small_limbs()
    b = limbs[BRANCH]
    side = unit(np.cross(b.pts[-1] - b.pts[0], UP))
    pts = []
    for k, t in enumerate(np.linspace(0.1, 0.9, 9)):
        q, rb, _ = limb_at(b.pts, b.radii, t)
        pts.append(q if k == 0 else q + side * (rb + 0.0025) * (1 if k % 2 else -1))
    limbs.append(Limb("twig.x", 3, BRANCH, 0.1, np.array(pts), np.full(9, 0.005)))
    assert verdict(tree(limbs)) == {"crossing-organic": {"branch.0 x twig.x"}}


def test_first_segment_through_the_parent_fails():
    """Seated on the near skin, its first segment runs out through the far side. Its
    base ring pokes out of the bark (limb-seat) and the segment crosses the branch."""
    limbs, q, big_r, d, n = _on_branch(0.6)
    b0 = q + n * 0.95 * big_r
    p1 = q - n * (big_r + 0.03) + d * 0.005
    limbs[TWIG].pts = np.array([b0, p1] + [p1 - n * 0.03 * k + UP * 0.01 * k for k in (1, 2, 3)])
    assert verdict(tree(limbs)) == {"limb-seat": {"twig.0"},
                                    "crossing-organic": {"branch.0 x twig.0"}}


def test_twig_hugging_its_branch_fails_near_miss():
    """A twig lying along its branch at a 0.3 r gap for its whole length."""
    limbs, q, _, _, n = _on_branch(0.5)
    pts = [q]
    for k in range(1, 5):
        qk, rk, _ = limb_at(limbs[BRANCH].pts, limbs[BRANCH].radii, 0.5 + 0.09 * k)
        pts.append(qk + n * (rk + 1.3 * 0.0055))
    limbs[TWIG].pts = np.array(pts)
    assert verdict(tree(limbs)) == {"near-miss-organic": {"branch.0 x twig.0"}}


def _along_leader(off_share=None):
    """Branch.0 laid along leader.0, grazing it (None) or buried in it (axis
    off_share x the leader's radius off)."""
    limbs = small_limbs()
    lead = limbs[2]
    on = [limb_at(lead.pts, lead.radii, t) for t in np.linspace(0.3, 0.9, 6)]
    side = unit(np.cross(on[0][2], np.array([0.0, 0.0, 1.0])))
    rb = np.linspace(0.010, 0.006, 6)
    pts = [on[0][0]]
    for k, (q, r, _) in enumerate(on[1:], start=1):
        pts.append(q + side * ((r + 1.1 * rb[k]) if off_share is None else off_share * r))
    limbs[BRANCH] = Limb("branch.0", 2, 2, 0.3, np.array(pts), rb)
    limbs = limbs[:5]
    leaves = [leaf for leaf in small_leaves(small_limbs()) if leaf["host"] == 3]
    return make_skeleton(limbs, leaves)


def test_branch_grazing_along_its_leader_fails_near_miss():
    assert verdict(_along_leader()) == {"near-miss-organic": {"branch.0 x leader.0"}}


def test_branch_glued_along_its_leader_fails_crossing():
    assert verdict(_along_leader(0.3)) == {"crossing-organic": {"branch.0 x leader.0"}}


def test_siblings_crossing_in_an_x_fail_crossing():
    """Two twigs seated 0.16 apart on one branch cross just above the bark."""
    limbs = small_limbs()
    bp, br = limbs[BRANCH].pts, limbs[BRANCH].radii

    def twig(s0, step, th0, th1):
        _, _, d = limb_at(bp, br, s0 / 0.4)
        n = unit(np.cross(d, UP))
        b = unit(np.cross(d, n))
        pts = []
        for k in range(5):
            q, rk, _ = limb_at(bp, br, (s0 + step * 0.04 * k) / 0.4)
            rho = 0.0 if k == 0 else rk + 1.6 * 0.0055
            th = np.radians(th0 + (th1 - th0) * k / 4)
            pts.append(q + rho * (np.cos(th) * n + np.sin(th) * b))
        return np.array(pts)

    limbs[TWIG] = Limb("twig.0", 3, BRANCH, 0.35, twig(0.14, 1, -20, 20), TWIG_R)
    limbs.append(Limb("twig.1", 3, BRANCH, 0.75, twig(0.30, -1, 20, -20), TWIG_R))
    assert verdict(tree(limbs)) == {"crossing-organic": {"twig.0 x twig.1"}}


def test_self_loop_fails_crossing():
    """A twig whose last segment passes back through its second."""
    limbs, q, _, d, _ = _on_branch(0.6)
    up = unit(np.array([-0.3, 0.6, 0.8]))
    side = unit(np.cross(up, d))
    limbs[TWIG].pts = np.array([q, q + up * 0.05, q + up * 0.10 + side * 0.03,
                                q + up * 0.07 + side * 0.03, q + up * 0.04 - side * 0.02])
    assert verdict(tree(limbs)) == {"crossing-organic": {"twig.0 x twig.0"}}


@pytest.mark.parametrize("t", [0.06, 0.05])
def test_branch_seated_just_above_the_fork_passes(t):
    """A branch low on leader.0 meets the trunk and leader.1 only inside its seat."""
    limbs = small_limbs()
    limbs[BRANCH] = Limb("branch.0", 2, 2, t, line(seat(limbs, 2, t), (-1, 0.3, 0.2), 0.4, 5),
                         np.linspace(0.012, 0.006, 6))
    q, _, _ = limb_at(limbs[BRANCH].pts, limbs[BRANCH].radii, 0.6)
    limbs[TWIG].pts = limbs[TWIG].pts - limbs[TWIG].pts[0] + q
    assert verdict(tree(limbs)) == {}


def test_branch_at_the_collar_passes():
    """A branch seated on leader.0 at t = 0.03, its base inside the trunk."""
    limbs = small_limbs()
    q = seat(limbs, 2, 0.03)
    limbs.append(Limb("branch.collar", 2, 2, 0.03, line(q, (0.0, 0.5, -1.0), 0.35, 5),
                      np.linspace(0.01, 0.005, 6)))
    assert verdict(tree(limbs)) == {}


@pytest.mark.parametrize("t", [0.6, 0.1, 0.03])
def test_twig_in_the_crotch_passes(t):
    """A twig seated on branch.0 close to branch.0's own seat on leader.0."""
    limbs = small_limbs()
    limbs[TWIG] = Limb("twig.0", 3, BRANCH, t,
                       line(seat(limbs, BRANCH, t), (-0.3, 0.6, 0.8), 0.15), TWIG_R)
    assert verdict(tree(limbs)) == {}


def test_low_shoot_seat_inside_the_trunk_passes():
    """A leader seated at the soil line; only its seat, inside the rooted trunk, sits
    below the floor."""
    limbs = small_limbs()
    q = seat(limbs, 0, 0.05)
    limbs.append(Limb("leader.low", 1, 0, 0.05, line(q, (0.0, 1.0, -1.0), 0.35, 7),
                      np.linspace(0.02, 0.008, 8)))
    assert verdict(tree(limbs)) == {}


# --- limb-seat ---------------------------------------------------------------------------------


def test_floating_limb_fails_limb_seat():
    """Parent -1, neither rooted nor buried, hanging in the air."""
    limbs = small_limbs()
    limbs.append(Limb("branch.float", 2, -1, 0.0, line((0.3, 0.6, 0.4), (1, 0.2, 0), 0.4, 5),
                      np.linspace(0.012, 0.006, 6)))
    assert verdict(tree(limbs)) == {"limb-seat": {"branch.float"}}


def test_base_ring_on_the_skin_fails_limb_seat():
    """A thick branch seated 0.99 of the trunk's radius off its axis."""
    limbs = small_limbs()
    p, r, _ = limb_at(limbs[0].pts, limbs[0].radii, 0.5)
    out = np.array([0.0, 0.0, 1.0])
    limbs.append(Limb("branch.skin", 2, 0, 0.5, line(p + out * 0.99 * r, out, 0.4, 5),
                      np.linspace(0.022, 0.008, 6)))
    assert verdict(tree(limbs)) == {"limb-seat": {"branch.skin"}}


def _long_petiole(lite):
    """Leaf 0 claims a 0.15 u petiole and its blade sits that far out. A lite
    tree draws no petiole, so its other blades sit on their nodes."""
    limbs = small_limbs()
    leaves = small_leaves(limbs)
    out = unit(leaves[0]["base"] - leaves[0]["node"])
    if lite:
        for leaf in leaves:
            leaf["base"], leaf["petiole"] = leaf["node"], 0.0
    leaves[0]["petiole"] = 0.15
    leaves[0]["base"] = leaves[0]["node"] + out * 0.15
    return make_skeleton(limbs, leaves, lite=lite)


def test_lite_leaf_far_from_its_node_fails_limb_seat():
    assert verdict(_long_petiole(lite=True)) == {"limb-seat": {"leaf.0"}}


def test_full_leaf_with_a_drawn_petiole_passes():
    """The full skeleton draws the petiole, and skeleton-match ties the mesh to it."""
    assert verdict(_long_petiole(lite=False)) == {}


def test_lite_bud_tree_leaf_far_from_its_node_fails_limb_seat():
    """A lite bud-grown leaf moved 0.5 u out along its petiole field."""
    sk = bud_tree("tiny-spiral", 1, 1.5, True)
    pet, base = np.array(sk.petiole), np.array(sk.base)
    base[0] = sk.node[0] + unit(base[0] - sk.node[0]) * 0.5
    pet[0] = 0.5
    assert verdict(dataclasses.replace(sk, petiole=pet, base=base)) == {"limb-seat": {"leaf.0"}}


# --- floor-organic -----------------------------------------------------------------------------


def test_buried_limb_not_on_a_rooted_limb_fails_floor():
    """Branch.0, seated high on leader.0, labelled buried and bent into the soil."""
    limbs = small_limbs()
    p0 = limbs[BRANCH].pts[0]
    limbs[BRANCH].pts = np.linspace(p0, [p0[0] - 0.5, -0.3, p0[2] + 0.1], 6)
    limbs[BRANCH].buried = True
    limbs[TWIG].pts = line(seat(limbs, BRANCH, 0.6), (-0.3, 0.6, 0.8), 0.15)
    assert verdict(tree(limbs)) == {"floor-organic": {"branch.0"}}


def test_buried_limb_seated_high_fails_floor():
    """A root of the trunk seated far above the soil, labelled buried."""
    limbs = small_limbs()
    q = seat(limbs, 0, 0.5)
    limbs[1] = Limb("root.0", 0, 0, 0.5, np.linspace(q, [0.4, -0.2, 0.0], 5),
                    np.linspace(0.024, 0.009, 5), buried=True)
    assert verdict(tree(limbs)) == {"floor-organic": {"root.0"}}


def test_rooted_limb_with_a_parent_fails_floor():
    limbs = small_limbs()
    limbs[1].rooted, limbs[1].buried = True, False
    assert "root.0" in verdict(tree(limbs)).get("floor-organic", set())


def test_sunk_tree_fails_floor():
    """The whole tree sunk a third of its height."""
    sk = bud_tree("tiny-spiral", 1, 1.5, False)
    drop = np.array([0.0, 0.25, 0.0])
    sunk = dataclasses.replace(sk, pts=sk.pts - drop, node=sk.node - drop, base=sk.base - drop)
    found = verdict(sunk)  # a grown tree has low limbs too: they sink with the trunk
    assert set(found) == {"floor-organic"}
    assert "trunk" in found["floor-organic"]


# --- skeleton-match ----------------------------------------------------------------------------


def _shipped(edit, sk=None):
    sk = sk or small_tree()
    parts = {name: tris.copy() for name, tris in grown_parts(sk, None).items()}
    edit(parts)
    return verdict(sk, parts=parts)


def test_rewired_triangle_fails_skeleton_match():
    """One leader triangle's corner re-wired to a far vertex of the same part."""
    def rewire(parts):
        lead = parts["Leaders"]
        flat = lead.reshape(-1, 3)
        lead[0, 2] = flat[np.argmax(flat[:, 0])]

    assert _shipped(rewire) == {"skeleton-match": {"Leaders"}}


def test_zero_area_leaf_fails_skeleton_match():
    """Leaf 0's fan collapsed to zero area over its own corners."""
    def collapse(parts):
        k = len(small_tree().outline)
        parts["Leaves_Leaf"][:k, 2] = parts["Leaves_Leaf"][:k, 1]

    assert _shipped(collapse) == {"skeleton-match": {"Leaves_Leaf"}}


def test_flipped_winding_fails_skeleton_match():
    """Every triangle's winding reversed (an inside-out tree)."""
    found = _shipped(lambda parts: parts.update(
        {name: np.ascontiguousarray(tris[:, ::-1]) for name, tris in parts.items()}))
    assert set(found) == {"skeleton-match"}
    assert found["skeleton-match"] == set(grown_parts(small_tree(), None))


def test_twigs_at_half_radius_fail_skeleton_match():
    """Every twig shipped at half its radius."""
    sk = bud_tree("tiny-spiral", 1, 1.5, False)
    twigs = sk.level_names.index("Twigs")
    radii = np.array(sk.radii)
    for i in np.flatnonzero(sk.level == twigs):
        radii[sk.start[i]:sk.start[i + 1]] *= 0.5
    thin = grown_parts(dataclasses.replace(sk, radii=radii), None)
    assert _shipped(lambda parts: parts.update(Twigs=thin["Twigs"]), sk) == {
        "skeleton-match": {"Twigs"}}


def test_no_geometry_is_unmeasured():
    """A skeleton whose piece has no geometry in the scene."""
    ctx = grown_ctx(small_tree())
    ctx = dataclasses.replace(ctx, pieces={k: v for k, v in ctx.pieces.items()
                                           if k != piece_key(PACK, PIECE)})
    check = SkeletonMatch()
    assert check.run(ctx) == []
    lines = check.coverage(ctx)
    assert "skeleton-match: skipped (no geometry): grove:Tree" in lines
    assert any(isinstance(line, Unmeasured) for line in lines)
    assert run_gate([check], ctx, None).status == "partial"


# --- small items -------------------------------------------------------------------------------


def test_seat_zone_matches_the_kit_clearance():
    """The bud grower's clearance is the kit file's, so the bud grower's stop-short rule and the
    checks draw the crotch at one line (`SEAT_ZONE_OPENS_OF_RADIUS` cites that row)."""
    params = json.loads(KIT_RULES.read_text())["params"]
    assert params["limbs.clearance_of_radius"]["value"] == SEAT_ZONE_OPENS_OF_RADIUS


def test_small_tree_names_its_limbs():
    sk = small_tree()
    assert limb_index(sk, "twig.0") == TWIG and limb_index(sk, "branch.0") == BRANCH


# --- the self-window is an arc length ----------------------------------------------------------

SELF_R = 0.06
SELF_CHECKS = {"crossing-organic", "near-miss-organic"}


def _lone_limb(pts) -> object:
    """A skeleton of one thick, rooted limb with no leaves."""
    pts = np.asarray(pts, float)
    return make_skeleton([Limb("trunk", 0, -1, 0.0, pts, np.full(len(pts), SELF_R), tip=DOME,
                               rooted=True)], [])


def test_straight_thick_limb_short_internodes_passes():
    """The self-window: 40 points 0.01 apart on a limb of radius 0.06, so the
    internode is under 0.75 of the radius. Pieces three segments apart touch by construction."""
    pts = line((0, 0, 0), UP, 0.39, 39)
    assert SELF_CHECKS.isdisjoint(verdict(_lone_limb(pts)))


def test_long_internodes_pass():
    """An internode of ten radii: no two pieces of the straight limb are near each other."""
    pts = line((0, 0, 0), UP, 4 * 10 * SELF_R, 4)
    assert SELF_CHECKS.isdisjoint(verdict(_lone_limb(pts)))


def test_folded_limb_still_crosses_itself():
    """The straight limb of the first case, bent back through itself at mid-length."""
    out = line((0, 0, 0), UP, 0.19, 19)
    back = line((0.03, 0.19, 0), -UP, 0.19, 19)
    got = verdict(_lone_limb(np.vstack([out, back])))
    assert got.get("crossing-organic") == {"trunk x trunk"}


def test_hairpin_of_long_segments_crosses_itself():
    """Two single long legs joined by a short turn: the turn's arc is inside the self-window,
    but the legs overlap along their whole length."""
    pts = np.array([(0, 0, 0), (0, 0.19, 0), (0.015, 0.2, 0), (0.03, 0.19, 0), (0.03, 0, 0)])
    got = verdict(_lone_limb(pts))
    assert got.get("crossing-organic") == {"trunk x trunk"}


def test_wide_hairpin_of_long_segments_crosses_itself():
    """Legs 0.5 long and 0.07 apart (overlap 0.05, over the crossing share of the radius),
    joined by a 4-segment turn of about 0.11 arc, inside the self-window."""
    turn = [(0.035 - 0.035 * np.cos(a), 0.5 + 0.035 * np.sin(a), 0)
            for a in np.linspace(0, np.pi, 5)[1:-1]]
    pts = np.array([(0, 0, 0), (0, 0.5, 0), *turn, (0.07, 0.5, 0), (0.07, 0, 0)])
    got = verdict(_lone_limb(pts))
    assert got.get("crossing-organic") == {"trunk x trunk"}
