import dataclasses
import itertools
import json
import math
import random
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks.branching._pairs import limb_pairs
from model_kit.core.errors import ManifestError
from model_kit.core.grow.buds import (
    MESH_V1,
    MESH_VERSION,
    _bend_to_gsa,
    _blade_angle,
    _flat_weight,
    _grow_tree,
    _Metamers,
    _radii,
    _rank_sides,
    _read,
    _shed,
    _stop_short,
    grow_buds,
    years_for_height,
)
from model_kit.core.grow.mesh import all_leaf_tris, grown_parts, leaf_tris
from model_kit.core.grow.rules import KIT_RULES, RuleReader, load_rules
from model_kit.core.grow.segments import Limb
from model_kit.core.grow.shape import FLAT_EPS, HALF_TURN_DEG, SCALE_MAX, SCALE_MIN
from model_kit.core.grow.skeleton import arc_lengths, check_skeleton, ring_lows, same_skeleton
from model_kit.core.sight import Sight

FIXTURES = Path(__file__).parent / "fixtures"
TINY_BUDS = FIXTURES / "tiny-buds.rules.json"  # two-ranked, no terminal bud
TINY_SPIRAL = FIXTURES / "tiny-spiral.rules.json"  # spiral, terminal bud present
UPM = 0.5
TURN_TOLERANCE_DEG = 0.5  # a design call of this kit: the band on a measured turn
UP = np.array([0.0, 1.0, 0.0])


def load(path=TINY_BUDS, kit=KIT_RULES):
    return load_rules(path, kit=kit)


def tree_of(rules, height_m=1.5, seed=1, lite=False):
    return _grow_tree(_read(RuleReader(rules), lite), height_m=height_m, seed=seed)


def skeleton(rules, height_m=1.5, seed=1, lite=False):
    return grow_buds(rules, piece="Tiny", height_m=height_m, seed=seed, lite=lite,
                     units_per_metre=UPM)


def with_params(tmp_path, source, **changes):
    """A copy of a species file with some param values replaced."""
    data = json.loads(Path(source).read_text())
    for key, value in changes.items():
        data["params"][key.replace("__", ".")]["value"] = value
    path = tmp_path / "temp.rules.json"
    path.write_text(json.dumps(data))
    return path


def with_params_kit(tmp_path, **changes):
    """A copy of the kit file with some param values replaced (keys written in full)."""
    data = json.loads(KIT_RULES.read_text())
    for key, value in changes.items():
        data["params"][key]["value"] = value
    path = tmp_path / "kit.rules.json"
    path.write_text(json.dumps(data))
    return path


def from_vertical_deg(d) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(d[1])))))


def twist_deg(tree, p, i, slot=0) -> float:
    """The turn of node i's bud `slot` from the same bud of its parent node, round the axis, in the
    parent's carried frame: the parent's bud direction is moved to node i's axis by the smallest
    turn, then the signed angle to node i's bud is read. Written out here, not taken from the
    grower."""
    a, d0, d1 = tree.side[p][slot], tree.dir[p], tree.dir[i]
    cross = np.cross(d0, d1)
    s = float(np.linalg.norm(cross))
    if s > 1e-12:
        k, ang = cross / s, math.atan2(s, float(d0 @ d1))
        a = a * math.cos(ang) + np.cross(k, a) * math.sin(ang) + k * (k @ a) * (1 - math.cos(ang))
    b = tree.side[i][slot]
    return math.degrees(math.atan2(float(np.cross(a, b) @ d1), float(a @ b)))


def same_axis_pairs(tree):
    return [(tree.parent[i], i) for i in range(len(tree))
            if tree.parent[i] >= 0 and tree.axis[tree.parent[i]] == tree.axis[i]]


@pytest.fixture(scope="module")
def buds_rules():
    return load(TINY_BUDS)


@pytest.fixture(scope="module")
def spiral_rules():
    return load(TINY_SPIRAL)


# ------------------------------------------------------------------------------ determinism

@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_same_seed_same_tree(path):
    rules = load(path)
    assert same_skeleton(skeleton(rules), skeleton(rules))


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_seed_changes_tree(path):
    rules = load(path)
    assert not same_skeleton(skeleton(rules, seed=1), skeleton(rules, seed=2))


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_parents_before_children(path):
    rules = load(path)
    tree, _ = tree_of(rules)
    assert tree.parent[0] == -1
    assert all(tree.parent[i] < i for i in range(len(tree)))
    sk = skeleton(rules)
    assert sk.parent[0] == -1
    assert np.all(sk.parent[1:] < np.arange(1, len(sk.parent)))


# --------------------------------------------------------------------- leaf arrangement

def test_spiral_turns_by_the_divergence(spiral_rules):
    pairs = 0
    for seed in (1, 2, 3):
        tree, _ = tree_of(spiral_rules, seed=seed)
        for p, i in same_axis_pairs(tree):
            assert abs(twist_deg(tree, p, i) - 137.5) < TURN_TOLERANCE_DEG, (seed, p, i)
            pairs += 1
        trunk = [(p, i) for p, i in same_axis_pairs(tree) if tree.axis[i] == 0]
        assert len(trunk) >= 16  # the seedling's year-1 shoot alone has 16 metamers
    assert pairs > 100


def bearing_deg(d) -> float:
    """The compass bearing of a direction: the azimuth of its horizontal projection."""
    return math.degrees(math.atan2(float(d[2]), float(d[0])))


def bearing_change_deg(d0, d1) -> float:
    turn = abs(bearing_deg(d1) - bearing_deg(d0)) % 360.0
    return min(turn, 360.0 - turn)


def twist_error_deg(tree, p, i) -> float:
    """How far a two-ranked node's side is from the opposite of its parent's, in degrees."""
    return abs(abs(twist_deg(tree, p, i)) - 180.0)


def two_ranked_margins(rules, seeds):
    """(upright errors, [(error, bound) for every pair]) over trees of `seeds`: the data both
    halves of the two-ranked test read."""
    par = _read(RuleReader(rules), False).par
    upright, every = [], []
    for seed in seeds:
        tree, _ = tree_of(rules, seed=seed)
        for p, i in same_axis_pairs(tree):
            err = twist_error_deg(tree, p, i)
            if max(from_vertical_deg(tree.dir[p]), from_vertical_deg(tree.dir[i])) \
                    <= par.round_within:
                upright.append(err)
            band = abs(_flat_weight(par, tree.dir[i]) - _flat_weight(par, tree.dir[p]))
            every.append((err, bearing_change_deg(tree.dir[p], tree.dir[i])
                          + band * HALF_TURN_DEG + TURN_TOLERANCE_DEG))
    return upright, every


def test_two_ranked_turns_by_half(buds_rules):
    """Each two-ranked node's side is the opposite of the last (a half turn), but on a slanted
    shoot the ranks follow gravity: they lie in the horizontal plane, so they turn with the
    shoot's compass bearing (Barthelemy & Caraglio 2007: plagiotropic axes carry their leaves
    'generally arranged in one plane', orthotropic ones radially). A half turn measured in a
    parallel-transported frame is the wrong yardstick there. So:
    (1) where both shoots are within the round band's lower end of vertical, the turn is
    180 +/- 0.5 deg;
    (2) on every same-axis pair in every regime, the error from 180 is at most the change in the
    shoot's bearing between the two nodes, plus the change the smooth band itself makes (at most
    a half turn for a whole band weight), plus 0.5 deg. This catches a rank that jumps."""
    upright, every = two_ranked_margins(buds_rules, (1, 2, 3))
    assert len(upright) > 50
    assert max(upright) < TURN_TOLERANCE_DEG
    assert len(every) > 500
    assert all(err <= bound for err, bound in every), max(err - bound for err, bound in every)


def test_decussate_pairs(tmp_path):
    """A maple. Two buds at a node, 180 degrees apart; each pair is 90 degrees
    from the last."""
    # A real maple keeps its terminal bud: the two laterals and the terminal share one cell.
    path = with_params(tmp_path, TINY_SPIRAL, leaves__per_node=2, leaves__divergence_deg=90)
    rules = load(path)
    tree, _ = tree_of(rules)
    assert all(side.shape == (2, 3) for side in tree.side)
    for side, d in zip(tree.side, tree.dir, strict=True):
        assert float(side[0] @ d) == pytest.approx(0.0, abs=1e-9)
        assert float(side[0] @ side[1]) == pytest.approx(-1.0, abs=1e-9)  # 180 degrees apart
    pairs = same_axis_pairs(tree)
    assert len(pairs) > 50
    for p, i in pairs:
        assert abs(twist_deg(tree, p, i, slot=0) - 90.0) < TURN_TOLERANCE_DEG
        assert abs(twist_deg(tree, p, i, slot=1) - 90.0) < TURN_TOLERANCE_DEG
    slots = {slot for _, slot in tree.buds}
    assert slots <= {-1, 0, 1} and {0, 1} <= slots


def test_decussate_with_terminal_grows(tmp_path):
    """The buds of one node sit at one point and do not shade each other, so a maple that keeps
    its terminal bud grows: more than 3 metamers a year, and its cited height with the scale
    inside the band. (It makes fewer metamers than a one-leaf tree: two buds per node cast twice
    the shadow below, as Palubicki's per-bud shadow has it.)"""
    rules = load(with_params(tmp_path, TINY_SPIRAL, leaves__per_node=2, leaves__divergence_deg=90))
    tree, years = tree_of(rules, height_m=1.5, seed=1)
    assert len(tree) > 3 * years
    sk = skeleton(rules, height_m=1.5, seed=1)
    assert SCALE_MIN <= sk.info["scale"] <= SCALE_MAX
    assert sk.pts[:, 1].max() == pytest.approx(1.5 * UPM)


def test_plagiotropic_ranks_lie_flat(buds_rules):
    """Two-ranked buds on a shoot more than 60 degrees from vertical point out sideways, with a
    vertical part under 0.2; on a shoot within 20 degrees of vertical they go round the axis."""
    slanted = upright = 0
    for seed in (1, 2, 3):
        tree, _ = tree_of(buds_rules, seed=seed)
        for d, sides in zip(tree.dir, tree.side, strict=True):
            if from_vertical_deg(d) > 60.0:
                assert np.all(np.abs(sides[:, 1]) < 0.2)
                slanted += 1
        for p, i in same_axis_pairs(tree):
            if from_vertical_deg(tree.dir[p]) < 20.0 and from_vertical_deg(tree.dir[i]) < 20.0:
                assert abs(abs(twist_deg(tree, p, i)) - 180.0) < TURN_TOLERANCE_DEG
                upright += 1
    assert slanted > 20 and upright > 20  # both cases happen on these trees


def test_flat_ranks_alternate_sides(buds_rules):
    par = _read(RuleReader(buds_rules), False).par
    slanted = np.array([0.95, 0.31, 0.0])  # 18 degrees up from horizontal, 72 from vertical
    slanted = slanted / np.linalg.norm(slanted)
    ref = np.array([0.0, 0.0, 1.0])
    first = _rank_sides(par, slanted, ref, 0.0)[0]
    second = _rank_sides(par, slanted, ref, 180.0)[0]
    assert first[1] == 0.0 and second[1] == 0.0
    assert np.allclose(first, -second)
    # An upright axis keeps the ranks round it, turned by the carried reference.
    up = _rank_sides(par, UP, np.array([1.0, 0.0, 0.0]), 90.0)[0]
    assert np.allclose(up, [0.0, 0.0, -1.0], atol=1e-12)


def test_the_band_between_round_and_flat_is_smooth(buds_rules):
    par = _read(RuleReader(buds_rules), False).par
    assert (par.round_within, par.flat_from) == (20.0, 60.0)
    ref = np.array([0.0, 0.0, 1.0])

    def lean(deg):  # a unit direction `deg` from vertical, toward +x
        r = math.radians(deg)
        return np.array([math.sin(r), math.cos(r), 0.0])

    weights = [_flat_weight(par, lean(deg)) for deg in range(91)]
    assert weights[:21] == [0.0] * 21 and weights[60:] == [1.0] * 31
    assert all(b >= a for a, b in itertools.pairwise(weights))
    assert max(b - a for a, b in itertools.pairwise(weights)) < 0.1
    # The side moves from round (here a +z/-y tilt) to flat without a jump: at 40 degrees it is
    # neither, and both its ends are unit and at right angles to the shoot.
    for deg in (10, 40, 70):
        side = _rank_sides(par, lean(deg), ref, 0.0)[0]
        assert float(np.linalg.norm(side)) == pytest.approx(1.0)
        assert float(side @ lean(deg)) == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < abs(_rank_sides(par, lean(40), np.array([0.0, 0.5, 0.866]), 0.0)[0][1]) < \
        abs(_rank_sides(par, lean(10), np.array([0.0, 0.5, 0.866]), 0.0)[0][1])


@pytest.mark.parametrize("round_within,flat_from", [(60, 20), (20, 20), (-1, 60), (20, 91)])
def test_the_rank_band_must_be_ordered(tmp_path, round_within, flat_from):
    path = with_params_kit(tmp_path, **{"ranks.round_within_vertical_deg": round_within,
                                        "ranks.flat_from_vertical_deg": flat_from})
    with pytest.raises(ManifestError, match="ranks.round_within_vertical_deg"):
        _read(RuleReader(load(TINY_BUDS, kit=path)), False)


def test_flat_ranks_are_only_for_two_ranked_leaves(spiral_rules):
    par = _read(RuleReader(spiral_rules), False).par
    slanted = np.array([1.0, 0.1, 0.0])
    slanted = slanted / np.linalg.norm(slanted)
    side = _rank_sides(par, slanted, np.array([0.0, 1.0, 0.0]), 0.0)[0]
    assert side[1] > 0.9  # carried round the axis, not laid flat


# ------------------------------------------------------------------------------ light and age

def test_stops_at_the_cited_age(spiral_rules):
    for height_m, years in ((0.5, 1), (1.0, 2), (1.5, 3)):
        sk = skeleton(spiral_rules, height_m=height_m)
        assert sk.info["years"] == years
        points = RuleReader(spiral_rules).points("age.height_points")
        assert sk.info["years"] == years_for_height(points, height_m)
        tree, grown = tree_of(spiral_rules, height_m=height_m)
        assert grown == years and max(tree.year) == years


def test_years_for_height():
    points = np.array([[2.0, 6.0], [1.0, 2.0]])  # out of order on purpose
    assert years_for_height(points, 1.0) == 2
    assert years_for_height(points, 1.5) == 4
    assert years_for_height(points, 1.4) == 4  # 3.6 rounds to 4
    assert years_for_height(points, 0.1) == 2  # the end values hold past the ends
    assert years_for_height(points, 9.0) == 6
    assert years_for_height(np.array([[5.0, 10.0]]), 3.0) == 10  # one point
    assert years_for_height(np.array([[1.0, 0.2]]), 1.0) == 1  # at least a year
    with pytest.raises(ManifestError, match="heights"):
        years_for_height(np.array([[1.0, 2.0], [1.0, 3.0]]), 1.0)


def test_more_shade_means_less_growth(tmp_path):
    """Light steers growth: a heavier shadow leaves the crown's inner buds asleep."""
    kit = json.loads(KIT_RULES.read_text())
    kit["params"]["light.shadow_a"]["value"] = 1.0
    shaded_kit = tmp_path / "kit.rules.json"
    shaded_kit.write_text(json.dumps(kit))
    base = len(tree_of(load(TINY_SPIRAL))[0])
    shaded = len(tree_of(load(TINY_SPIRAL, kit=shaded_kit))[0])
    assert shaded < base


def test_buds_are_left_asleep(spiral_rules):
    tree, _ = tree_of(spiral_rules)
    asleep = {m for m, slot in tree.buds if slot >= 0}
    assert asleep and len(asleep) < len(tree)  # some axils sprouted, some did not


# ------------------------------------------------------------------------------- the skeleton

@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_small_tree_is_valid(path, seed):
    """0.5 m is one year. Every limb has two points or more, every number is
    finite, and the tree is as tall as it was asked to be."""
    rules = load(path)
    sk = skeleton(rules, height_m=0.5, seed=seed)
    assert sk.info["years"] == 1
    check_skeleton(sk)
    assert np.all(np.diff(sk.start) >= 2)
    for name in ("pts", "radii"):
        assert np.all(np.isfinite(getattr(sk, name)))
    assert sk.pts[:, 1].max() == pytest.approx(0.5 * UPM)


def test_info_keys(buds_rules):
    sk = skeleton(buds_rules, height_m=1.0, seed=7)
    info = sk.info
    for key in ("piece", "rules", "seed", "height_m", "H", "lite", "years", "rules_sha256",
                "kit_sha256", "fitted", "mesh"):
        assert key in info, key
    assert info["rules"] == "tiny-buds.rules.json"
    assert info["piece"] == "Tiny" and info["seed"] == 7 and info["lite"] is False
    assert info["height_m"] == 1.0 and info["H"] == 1.0 * UPM
    assert info["rules_sha256"] == buds_rules.sha256
    assert info["kit_sha256"] == buds_rules.kit_sha256
    assert info["fitted"] == {"apical.control": 0.5}
    assert info["mesh"] == MESH_V1 == "buds-v1"
    assert info["counts"]["leaves"] == len(sk.host) > 0
    assert sum(v for k, v in info["counts"].items() if k != "leaves") == len(sk.limb_id)


def test_the_tree_is_scaled_to_the_asked_height(spiral_rules):
    sk = skeleton(spiral_rules, height_m=1.5)
    assert sk.pts[:, 1].max() == pytest.approx(1.5 * UPM)
    assert sk.info["raw_height_m"] * sk.info["scale"] == pytest.approx(1.5)


@pytest.mark.parametrize("height_m", [10.0, 0.05])
def test_height_outside_the_cited_ages_raises(spiral_rules, height_m):
    """No cited age grows a tree past the ends of age.height_points (0.5 to 1.5 m here): the error
    names the piece and the cited span."""
    with pytest.raises(ManifestError, match=r"Tiny.*outside the heights age.height_points cites"):
        skeleton(spiral_rules, height_m=height_m)


def test_stalled_grow_raises(tmp_path):
    """A grow that misses its cited height by more than the band is a model failure, not a fit
    miss: with vigour under one metamer per unit of light, the tree never grows past its seedling,
    and the error names the piece, the raw height and the scale."""
    rules = load_rules(TINY_SPIRAL, kit=with_params_kit(tmp_path, **{"resource.alpha": 0.5}))
    with pytest.raises(ManifestError, match=r"Tiny.*raw height.*scale"):
        skeleton(rules, height_m=1.5)


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_no_raise_across_the_cited_ages(path):
    """Every height between the species' cited age points grows without the guard raising, though
    the age is rounded to whole years (the heights 0.75, 0.8 and 1.25 m must not raise)."""
    rules = load(path)
    for height_m in np.round(np.arange(0.5, 1.5001, 0.05), 2):
        for seed in (1, 2, 3):
            sk = skeleton(rules, height_m=float(height_m), seed=seed)
            assert sk.info["height_m"] == pytest.approx(float(height_m))


def test_info_round_trips_through_json(tmp_path, buds_rules):
    from model_kit.core.grow.skeleton import load_skeleton, save_skeleton
    sk = skeleton(buds_rules)
    save_skeleton(sk, tmp_path / "t.skeleton.npz")
    back = load_skeleton(tmp_path / "t.skeleton.npz")
    assert same_skeleton(sk, back) and back.info == sk.info


def test_the_trunk_stands_in_the_soil(spiral_rules):
    sk = skeleton(spiral_rules)
    assert sk.limb_id[0] == "trunk" and sk.level[0] == 0
    assert np.flatnonzero(sk.rooted).tolist() == [0]
    assert sk.buried.any() and not sk.buried[0]  # roots grow into the soil
    assert sk.limb_points(0)[0, 1] < 0.0 <= sk.limb_points(0)[1, 1]
    assert sk.limb_points(0)[0, 1] >= -1.2 * sk.limb_radii(0)[0]  # the floor check's depth limit
    assert len(set(sk.limb_id)) == len(sk.limb_id)


# Both tiny species, fit seeds 1 to 4, at each cited end and the middle, and the tallest lite.
SOIL_CASES = [(path, seed, height_m, lite) for path in (TINY_BUDS, TINY_SPIRAL)
              for seed in (1, 2, 3, 4)
              for height_m, lite in ((0.5, False), (1.0, False), (1.5, False), (1.5, True))]


def _soil_id(case) -> str:
    path, seed, height_m, lite = case
    return f"{path.name.split('.')[0]}-s{seed}-{height_m}m" + ("-lite" if lite else "")


@cache
def soil_tree(path, seed, height_m, lite):
    return skeleton(load(path), height_m=height_m, seed=seed, lite=lite)


@pytest.mark.parametrize("case", SOIL_CASES, ids=_soil_id)
def test_no_limb_grows_into_the_soil(case):
    """The soil (y = 0) is an obstacle: the grower keeps every limb axis above it, so no ring of
    a limb lies below it by more than the tree's thinnest radius (far inside floor-organic's
    tolerance of a tube thickness). The trunk's foot (its first point, sunk into the soil on
    purpose) and the buried roots are left out."""
    sk = soil_tree(*case)
    touch = float(sk.radii.min())
    for i in np.flatnonzero(~sk.buried):
        axis = sk.limb_points(i)[1 if sk.rooted[i] else 0:, 1]
        assert axis.min() >= 0.0, (sk.limb_id[i], float(axis.min()))
        low = ring_lows(sk.limb_points(i), sk.limb_radii(i))[1 if sk.rooted[i] else 0:]
        assert low.min() >= -touch, (sk.limb_id[i], float(low.min()))


@pytest.mark.parametrize("case", SOIL_CASES, ids=_soil_id)
def test_no_leaf_grows_into_the_soil(case):
    """No vertex of a leaf's triangles (blade and petiole, as the mesh draws them) lies below the
    soil."""
    sk = soil_tree(*case)
    assert len(sk.host)
    low = all_leaf_tris(sk)[..., 1].reshape(len(sk.host), -1).min(axis=1)
    assert low.min() >= 0.0, [(int(f), float(low[f])) for f in np.flatnonzero(low < 0.0)]


def test_the_tree_carries_leaves(spiral_rules):
    sk = skeleton(spiral_rules)
    assert len(sk.host) > 20 and len(sk.node) == len(sk.host)


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_limbs_are_continuation_chains(path):
    """A limb seats on its parent's axis, and at every node the parent runs on through the
    thicker child: the limb seated there is never thicker than the parent's next piece."""
    sk = skeleton(load(path))
    assert len(sk.limb_id) > 20
    for c in range(1, len(sk.limb_id)):
        p = int(sk.parent[c])
        point, _, _ = sk.at(p, float(sk.t[c]))
        assert np.allclose(point, sk.limb_points(c)[0], atol=1e-9), sk.limb_id[c]
        at = np.flatnonzero(np.all(np.isclose(sk.limb_points(p), sk.limb_points(c)[0]), axis=1))
        assert len(at), sk.limb_id[c]  # seated at a node of the parent
        k = int(at[0])
        if k < len(sk.limb_points(p)) - 1:  # the parent runs on past this node
            assert sk.limb_radii(p)[k] >= sk.limb_radii(c)[0] - 1e-12, sk.limb_id[c]


def crown_points(sk):
    """(the points of the limbs that are not roots, how many such limbs there are)."""
    crown = np.flatnonzero(~sk.buried)
    return int(sum(len(sk.limb_points(int(c))) for c in crown)), len(crown)


def assert_points_are_metamers(sk, tree):
    """A limb of k metamers has k + 1 points, and the trunk has one more, sunk into the soil.
    Stop-short may cut a limb inside a segment, which leaves the metamers past the cut out of
    `info["metamers"]` less `cut_metamers` and puts the cut point in as one more point of that
    limb (at most one per cut metamer). Roots are separate limbs and are not counted."""
    points, limbs = crown_points(sk)
    kept = sum(tree.alive) - sk.info["cut_metamers"]
    assert kept + limbs + 1 <= points <= kept + limbs + 1 + sk.info["cut_metamers"]


def test_every_metamer_is_in_one_limb(spiral_rules):
    """Every live metamer is drawn in exactly one limb. Which axis the trunk limb follows is the
    thickest-child rule's call (test_chains.py), not the leader's."""
    tree, _ = tree_of(spiral_rules)
    sk = skeleton(spiral_rules)
    assert_points_are_metamers(sk, tree)


def test_level_is_the_chain_order(spiral_rules):
    sk = skeleton(spiral_rules)
    top = len(sk.level_names) - 1
    for c in np.flatnonzero(~sk.buried & (sk.parent >= 0)):  # roots are level 0, whatever the chain
        assert sk.level[c] == min(sk.level[sk.parent[c]] + 1, top) or sk.level[c] == top
    assert sk.level.max() >= 2


def test_lite_changes_only_the_tube_sides(buds_rules):
    full, lite = skeleton(buds_rules), skeleton(buds_rules, lite=True)
    assert np.array_equal(full.pts, lite.pts) and np.array_equal(full.parent, lite.parent)
    assert full.sides.tolist() == [12, 9, 7, 5, 4] and lite.sides.tolist() == [8, 6, 5, 3, 3]
    assert lite.lite and lite.info["lite"] is True


# ------------------------------------------------------------------------- buds-v2: no tube sides

def without_tube_sides(tmp_path, source):
    """A copy of a species file without `tubes.sides` and `lite.tube_sides`, or the rows only they
    cite: the rules of a tree whose tube sides come from the view when it is meshed."""
    data = json.loads(Path(source).read_text())
    for key in ("tubes.sides", "lite.tube_sides"):
        del data["params"][key]
    data["rows"] = [r for r in data["rows"] if r["id"] not in ("tubes.sides", "lite.tree-line")]
    path = tmp_path / "v2.rules.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("lite", [False, True])
def test_rules_without_tube_sides_grow_a_buds_v2_skeleton(tmp_path, lite):
    """The mesh version follows the rules: a file with `tubes.sides` grows a buds-v1 skeleton and
    keeps its sides (rules files written for that mesh), one without grows buds-v2,
    which holds no sides because the mesh reads the view."""
    old = skeleton(load(), lite=lite)
    assert old.info["mesh"] == MESH_V1 and old.sides.tolist() == ([8, 6, 5, 3, 3] if lite else
                                                                   [12, 9, 7, 5, 4])
    new = skeleton(load(without_tube_sides(tmp_path, TINY_BUDS)), lite=lite)
    assert new.info["mesh"] == MESH_VERSION == "buds-v2"
    assert new.sides.tolist() == [0, 0, 0, 0, 0]
    assert np.array_equal(new.pts, old.pts) and np.array_equal(new.parent, old.parent)


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_every_param_of_a_buds_v2_rules_file_is_read(tmp_path, path):
    """Without the tube sides there is no exemption: after a full and a lite grow, every param of
    the species and the kit has been read."""
    rules = load(without_tube_sides(tmp_path, path))
    seen = set()
    for lite in (False, True):
        reader = RuleReader(rules)
        _read(reader, lite)
        seen |= reader.read
    assert set(rules.params) - seen == set()


def test_a_buds_v2_skeleton_meshes_with_a_sight_and_not_without(tmp_path):
    sk = skeleton(load(without_tube_sides(tmp_path, TINY_SPIRAL)), height_m=0.5)
    with pytest.raises(ManifestError, match="sight"):
        grown_parts(sk, None)
    parts = grown_parts(sk, Sight(gap=1e-3, crease_deg=35.0))
    assert parts and all(np.isfinite(t).all() for t in parts.values())


# ------------------------------------------------------------------------------------ errors

def test_a_missing_param_raises_naming_it(tmp_path):
    data = json.loads(TINY_BUDS.read_text())
    del data["params"]["branch.angle_deg"]
    data["rows"] = [r for r in data["rows"] if r["id"] != "branch.angle"]
    path = tmp_path / "short.rules.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match=r"branch\.angle_deg"):
        skeleton(load(path))


def test_the_terminal_bud_must_be_present_or_absent(tmp_path):
    path = with_params(tmp_path, TINY_BUDS, buds__terminal=["sometimes"])
    with pytest.raises(ManifestError, match="buds.terminal"):
        skeleton(load(path))


@pytest.mark.parametrize("height_m, upm", [(0.0, UPM), (-1.0, UPM), (1.0, 0.0)])
def test_height_and_units_must_be_positive(buds_rules, height_m, upm):
    with pytest.raises(ManifestError, match="over 0"):
        grow_buds(buds_rules, piece="T", height_m=height_m, seed=1, lite=False,
                  units_per_metre=upm)


def test_params_are_frozen():
    par = _read(RuleReader(load(TINY_BUDS)), False).par
    with pytest.raises(dataclasses.FrozenInstanceError):
        par.alpha = 1.0


# ------------------------------------------------- resource, relay, gravity and shedding

def order_path(tree, order=0):
    """The metamers of one axis order, in index order, for an order that forms a single path
    (order 0: the trunk). Dead metamers are left out."""
    return [i for i in range(len(tree)) if tree.order[i] == order and tree.alive[i]]


def test_no_terminal_bud_relays(buds_rules):
    """No terminal bud: each year's extension of the trunk chain is seated on the previous year's
    top node, and the chain stays one limb from base to tip."""
    relays = 0
    for seed in (1, 2, 3):
        tree, years = tree_of(buds_rules, seed=seed)
        path = order_path(tree)
        assert len(path) > years  # the trunk grows more than one metamer a year
        for p, i in itertools.pairwise(path):
            assert tree.parent[i] == p
            if tree.year[i] == tree.year[p]:
                assert tree.main[p] == i and not tree.relay[i]  # inside one year's shoot
            else:
                # A new year's extension: a relay shoot, from the last year's top node.
                assert tree.relay[i] and tree.lat[p][0] == i and tree.main[p] == -1
                assert tree.year[i] == tree.year[p] + 1
                relays += 1
        assert [tree.year[i] for i in path] == sorted(tree.year[i] for i in path)
        assert tree.year[path[-1]] == years  # the trunk is still growing in the last year
        sk = skeleton(buds_rules, seed=seed)
        assert len(sk.limb_points(0)) == len(path) + 2  # the soil point, the base, then each end
    assert relays >= 3


def test_every_relay_goes_on_from_a_tip_of_the_year_before(buds_rules):
    """Every relay shoot on the tree, not only the trunk's: it is seated on the tip node of an
    axis (no main child) that stopped growing the year before, takes the axis' order, and no
    skeleton limb starts at that node, so the two share one limb."""
    relays = 0
    for seed in (1, 2, 3):
        tree, _ = tree_of(buds_rules, seed=seed)
        sk = skeleton(buds_rules, seed=seed)
        f = sk.info["scale"] * UPM
        starts = sk.pts[sk.start[:-1]]
        for i in range(len(tree)):
            if not tree.relay[i]:
                continue
            p = tree.parent[i]
            assert tree.main[p] == -1 and tree.order[i] == tree.order[p]
            assert tree.year[i] == tree.year[p] + 1  # measured: all 23 relays of seeds 1 to 3
            assert not np.any(np.all(np.isclose(starts, tree.end[p] * f, atol=1e-9), axis=1))
            relays += 1
    assert relays > 15  # measured 23 (seeds 1 to 3, shedding on)


def test_a_terminal_bud_is_not_a_relay(spiral_rules):
    tree, _ = tree_of(spiral_rules)
    assert not any(tree.relay)


def trunk_top_of_tree(tree):
    """(the top of the trunk chain, the top of the tree, the trunk's last internode length), for
    the live metamers."""
    alive = [i for i in range(len(tree)) if tree.alive[i]]
    trunk = [i for i in alive if tree.order[i] == 0]
    last = trunk[-1]
    return (max(tree.end[i][1] for i in trunk), max(tree.end[i][1] for i in alive),
            float(np.linalg.norm(tree.end[last] - tree.start[last])))


def test_terminal_bud_keeps_one_leader(tmp_path):
    """A terminal bud and a high apical control: the trunk chain reaches within one internode of
    the top of the tree. (A control of 0.7 grows trees far past their cited height, so this reads
    the grown metamers, not a skeleton that the scale guard would refuse.)"""
    for lam in (0.5, 0.7):
        rules = load(with_params(tmp_path, TINY_SPIRAL, apical__control=lam))
        for seed in (1, 2, 3):
            trunk, top, internode = trunk_top_of_tree(tree_of(rules, seed=seed)[0])
            assert top - trunk <= internode, (lam, seed, top - trunk, internode)
    # The skeleton's trunk limb is the thickest-child chain (this kit's limb rule, shared with the
    # measured trees), not the leader: with pipe-model radii a basal lateral can be thicker
    # (tiny-spiral seed 1, 0.0373 against 0.0358), so the leader is judged on the metamers above.


def trunk_share(rules, seeds=(1, 2, 3, 4)):
    """The trunk chain's height as a share of the tree's, averaged over seeds."""
    shares = []
    for seed in seeds:
        trunk, top, _ = trunk_top_of_tree(tree_of(rules, seed=seed)[0])
        shares.append(trunk / top)
    return float(np.mean(shares))


def test_apical_control_orders_the_crown(tmp_path):
    """The stronger the apical control, the more of the tree's height the trunk chain carries: a
    weak control lets side branches overtop it, a strong one keeps one leader."""
    shares = [trunk_share(load(with_params(tmp_path, TINY_SPIRAL, apical__control=lam)))
              for lam in (0.1, 0.3, 0.5)]
    assert shares[0] < shares[1] < shares[2], shares
    assert shares[2] - shares[0] > 0.5


def from_gravity_deg(d) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(d[1])))))


def order_one_shoots(tree):
    """Each order-1 side shoot as one list of metamers, relay shoots followed whole: the shoot a
    lateral bud grew, then the relay shoot from its tip, then that one's relay, and so on. A relay
    shoot is the same shoot going on, so it is not a shoot of its own."""
    by_axis = {}
    for i in range(len(tree)):
        if tree.alive[i]:
            by_axis.setdefault(tree.axis[i], []).append(i)
    shoots = []
    for ms in by_axis.values():
        if tree.order[ms[0]] != 1 or tree.relay[ms[0]]:
            continue
        whole = list(ms)
        while True:
            relay = tree.lat[whole[-1]][0]
            if relay < 0 or not tree.alive[relay] or not tree.relay[relay]:
                break
            whole.extend(by_axis[tree.axis[relay]])
        shoots.append(whole)
    return shoots


def gsa_distances(rules, seeds=(1, 2, 3)):
    """(start, end) lists: how far, in degrees, the direction each order-1 shoot's bud emerged in
    and the direction of the shoot's last internode are from the order-1 set-point angle."""
    gsa = RuleReader(rules).choices("gsa.by_order_deg")[1]
    start, end = [], []
    for seed in seeds:
        tree, _ = tree_of(rules, seed=seed)
        for ms in order_one_shoots(tree):
            start.append(abs(from_gravity_deg(tree.emerge[ms[0]]) - gsa))
            end.append(abs(from_gravity_deg(tree.dir[ms[-1]]) - gsa))
    return start, end


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_shoots_bend_toward_their_gsa(tmp_path, path):
    """With the light pull off (`growth.light_weight` 0 in a copy of the kit),
    every order-1 shoot's final direction is nearer its GSA (an angle from gravity) than the
    direction its bud emerged in. With light on, a shoot settles where gravitropism and
    phototropism balance, so this is not a fact there."""
    rules = load_rules(path, kit=with_params_kit(tmp_path, **{"growth.light_weight": 0.0}))
    start, end = gsa_distances(rules)
    assert len(start) > 15  # measured with the light off: 23 on tiny-buds, 66 on tiny-spiral
    nearer = sum(e < s for s, e in zip(start, end, strict=True))
    assert nearer == len(start), (nearer, len(start), np.mean(start), np.mean(end))


def test_the_bend_follows_the_sine_law():
    """The turn toward the set-point is `rate x sin(angle between the two)`: a shoot at right
    angles to its set-point turns by the rate; one near it turns less; it never overshoots."""
    par = _read(RuleReader(load(TINY_SPIRAL)), False).par
    par = dataclasses.replace(par, gsa=(180.0,))  # straight up
    rate = par.gravity
    east = np.array([1.0, 0.0, 0.0])
    turned = _bend_to_gsa(par, east, 0)
    assert math.acos(float(east @ turned)) == pytest.approx(rate)
    assert turned[1] > 0.0 and turned[2] == pytest.approx(0.0)  # same compass heading, nearer up
    near = np.array([math.sin(0.5), math.cos(0.5), 0.0])  # 0.5 rad from the set-point
    assert math.acos(float(near @ _bend_to_gsa(par, near, 0))) == pytest.approx(rate * math.sin(0.5))
    swift = dataclasses.replace(par, gravity=5.0)
    assert np.allclose(_bend_to_gsa(swift, east, 0), [0.0, 1.0, 0.0])  # it stops at the set-point
    # An order past the list's end takes the last entry.
    low = dataclasses.replace(par, gsa=(180.0, 90.0))
    assert np.allclose(_bend_to_gsa(low, east, 5), east)  # 90 deg from gravity is horizontal


def lowest_third_side_limbs(sk):
    """The limbs seated on the trunk in its lowest third."""
    pts = sk.limb_points(0)
    low = pts[1, 1] + (pts[:, 1].max() - pts[1, 1]) / 3.0
    return [c for c in range(1, len(sk.limb_id))
            if sk.parent[c] == 0 and not sk.buried[c] and sk.limb_points(c)[0, 1] < low]


@pytest.mark.parametrize("path", [TINY_BUDS, TINY_SPIRAL])
def test_shaded_limbs_are_shed(tmp_path, path):
    """With a high shedding threshold, set in a copy of the kit file, the lowest
    third of the trunk carries no live side limbs; with the kit's own threshold it does. Each
    year's shed pass judges that year's new shoots too: a shaded flush dies at the end of its
    season."""
    strict = load_rules(path, kit=with_params_kit(tmp_path, **{"shed.light_per_internode": 1e9}))
    for seed in (1, 2, 3):
        assert lowest_third_side_limbs(skeleton(load(path), seed=seed)), (path.name, seed)
        sk = skeleton(strict, seed=seed)
        assert not lowest_third_side_limbs(sk), (path.name, seed)


def test_a_relay_is_never_shed(tmp_path):
    """Shed everything that may be shed: the relay chain of the trunk still stands whole."""
    strict = load_rules(TINY_BUDS, kit=with_params_kit(
        tmp_path, **{"shed.light_per_internode": 1e9}))
    tree, years = tree_of(strict)
    path = order_path(tree)
    assert tree.year[path[-1]] == years
    assert sum(tree.relay[i] for i in path) >= 2
    assert all(tree.alive[i] for i in path)


def build_shed_tree():
    """A hand-built tree: a trunk of 3 metamers (0-2); a branch B of 2 metamers (3, 4) seated at
    node 1; and a sub-branch S of 3 metamers (5, 6, 7) seated at B's first node (3)."""
    par = _read(RuleReader(load(TINY_SPIRAL)), False).par
    tree = _Metamers(par)
    up, side = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])

    def add(parent, link, start, d, order, axis):
        return tree.add(start, d, 0.05, parent, order, axis, 1, side, 0.0, link=link)

    a = tree.new_axis()
    add(-1, -1, np.zeros(3), up, 0, a)
    add(0, -1, tree.end[0], up, 0, a)
    add(1, -1, tree.end[1], up, 0, a)
    b = tree.new_axis()
    add(1, 0, tree.end[1], side, 1, b)
    add(3, -1, tree.end[3], side, 1, b)
    s = tree.new_axis()
    add(3, 0, tree.end[3], up, 2, s)
    add(5, -1, tree.end[5], up, 2, s)
    add(6, -1, tree.end[6], up, 2, s)
    return tree


def test_shedding_takes_the_smallest_branches_first():
    """B carries 1.0 of light over 5 internodes (B's 2 and S's 3) = 0.2 each, and S carries none.
    Judged outside in, B would go with S for 0.2 < 0.3 only if S were counted; judged smallest
    first, S goes (0 < 0.3) and B, with 1.0 over its own 2 internodes = 0.5, stays."""
    tree = build_shed_tree()
    q_at = [0.0] * len(tree)
    q_at[4] = 1.0  # a bud at B's top node
    assert _shed(tree, q_at, 0.3, len(tree)) == 1
    assert [i for i in range(len(tree)) if not tree.alive[i]] == [5, 6, 7]
    assert tree.lat[3][0] == -1 and tree.lat[1][0] == 3


def test_shedding_leaves_a_lit_branch_and_a_relay():
    tree = build_shed_tree()
    q_at = [0.0] * len(tree)
    q_at[4], q_at[7] = 1.0, 1.0
    assert _shed(tree, q_at, 0.3, len(tree)) == 0 and all(tree.alive)
    dark = build_shed_tree()
    dark.relay[3] = True  # B is a relay shoot
    assert _shed(dark, [0.0] * len(dark), 0.3, len(dark)) == 1  # only S goes
    assert [i for i in range(len(dark)) if not dark.alive[i]] == [5, 6, 7]


def test_shed_wood_is_not_in_the_skeleton_or_the_buds(tmp_path):
    strict = load_rules(TINY_SPIRAL, kit=with_params_kit(
        tmp_path, **{"shed.light_per_internode": 1e9}))
    tree, _ = tree_of(strict)
    dead = {i for i in range(len(tree)) if not tree.alive[i]}
    assert dead
    assert all(m not in dead for m, _ in tree.buds)
    sk = skeleton(strict)
    assert_points_are_metamers(sk, tree)
    assert sk.info["metamers"] == sum(tree.alive)


# ----------------------------------------- radii, shedding memory, stop-short, leaves and roots

BOTH = [TINY_BUDS, TINY_SPIRAL]


def kit_param(rules, key):
    return rules.params[key].value


def children_of(tree):
    kids = [[] for _ in range(len(tree))]
    for i, p in enumerate(tree.parent):
        if p >= 0:
            kids[p].append(i)
    return kids


def live_only_radii(tree, tip, exponent):
    """The pipe model with no memory: shed branches left out. Written out here, not taken from the
    grower."""
    kids = children_of(tree)
    r = np.zeros(len(tree))
    for i in range(len(tree) - 1, -1, -1):
        alive = [c for c in kids[i] if tree.alive[c]]
        r[i] = tip if not alive else sum(r[c] ** exponent for c in alive) ** (1.0 / exponent)
    return r


@pytest.mark.parametrize("path", BOTH)
def test_pipe_model_radii(path):
    """Leonardo's rule at every fork: the parent's radius to the power D is at least the sum of
    its children's, and equal where no child was shed (the wood of a shed child stays in the
    parent). A tip has the twig's tip radius."""
    rules = load(path)
    exponent, tip = kit_param(rules, "leonardo.exponent"), kit_param(rules, "twig.tip_radius_m")
    tree, _ = tree_of(rules)
    r = _radii(tree)
    kids = children_of(tree)
    forks = lost_forks = 0
    for i in range(len(tree)):
        if not tree.alive[i]:
            continue
        if not kids[i]:
            assert r[i] == pytest.approx(tip)
            continue
        alive = [c for c in kids[i] if tree.alive[c]]
        total = sum(r[c] ** exponent for c in alive)
        assert r[i] ** exponent >= total - 1e-12
        if len(alive) == len(kids[i]):
            assert r[i] ** exponent == pytest.approx(total, abs=1e-9)
        else:
            assert r[i] ** exponent > total
            lost_forks += 1
        forks += len(alive) > 1
    assert forks > 5 and lost_forks > 0  # both kinds of fork occur on these trees


@pytest.mark.parametrize("path", BOTH)
def test_pipe_model_radii_in_the_skeleton(path):
    """The same rule read off the skeleton, at the nodes where limbs fork. The radius at a node
    is the pipe below it; the limb runs on to a thinner point above it, and each limb seated there
    starts at its own radius. (Roots are not part of the crown's pipe.) Stop-short can only take
    children away, so the sum above never passes the radius below."""
    rules = load(path)
    exponent = kit_param(rules, "leonardo.exponent")
    sk = skeleton(rules)
    forks = equal = 0
    for c in range(len(sk.limb_id)):
        radii, cum = sk.limb_radii(c), arc_lengths(sk.limb_points(c))
        seated = {}
        for k in np.flatnonzero((sk.parent == c) & ~sk.buried):
            node = int(np.argmin(np.abs(cum / cum[-1] - sk.t[k])))
            assert abs(cum[node] / cum[-1] - sk.t[k]) < 1e-9  # seated on a node
            seated.setdefault(node, []).append(int(k))
        for node, ks in seated.items():
            above = sum(sk.limb_radii(k)[0] ** exponent for k in ks)
            if node + 1 < len(radii):
                above += radii[node + 1] ** exponent
            assert radii[node] ** exponent >= above * (1.0 - 1e-9)
            forks += 1
            equal += abs(radii[node] ** exponent - above) <= 1e-9 * radii[node] ** exponent
    assert forks > 10 and equal > 0


def test_shed_wood_is_kept(tmp_path):
    """Shedding never thins the trunk below the radius it had with the wood on it: the radii with
    memory are the radii of the same tree with every shed limb standing, and at least those of the
    live wood alone."""
    strict = load_rules(TINY_SPIRAL, kit=with_params_kit(
        tmp_path, **{"shed.light_per_internode": 1e9}))
    tree, _ = tree_of(strict)
    dead = [i for i in range(len(tree)) if not tree.alive[i]]
    assert dead
    kept = _radii(tree)
    exponent = kit_param(strict, "leonardo.exponent")
    bare = live_only_radii(tree, kit_param(strict, "twig.tip_radius_m"), exponent)
    live = [i for i in range(len(tree)) if tree.alive[i]]
    assert all(kept[i] >= bare[i] - 1e-12 for i in live)
    assert kept[0] > bare[0]  # the trunk keeps the wood of its shed limbs
    # And the skeleton's trunk is as thick as that wood says.
    sk = skeleton(strict)
    assert sk.limb_radii(0)[0] == pytest.approx(kept[0] * sk.info["scale"] * UPM)


@pytest.mark.parametrize("path", BOTH)
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_stop_short_runs_on_final_radii(path, seed):
    """No two limbs come within the clearance times the thinner radius outside their zones, read
    by the checks' own pair search; the radii are the pipe model's, not thinned by the cuts; and
    stop-short run again on its own result cuts nothing more."""
    rules = load(path)
    share = kit_param(rules, "limbs.clearance_of_radius")
    tree, _ = tree_of(rules, seed=seed)
    sk = skeleton(rules, seed=seed)
    f = sk.info["scale"] * UPM
    assert sk.limb_radii(0)[0] == pytest.approx(_radii(tree)[0] * f)
    close = [(sk.limb_id[i], sk.limb_id[j], gap / r) for i, j, gap, r in
             limb_pairs(sk, reach=share, opens=share) if gap < share * r * (1.0 - 1e-9)]
    assert close == []
    again = {}
    for i in range(len(sk.limb_id)):
        parent = int(sk.parent[i])
        again[i] = Limb(str(sk.limb_id[i]), int(sk.level[i]), sk.limb_points(i),
                        sk.limb_radii(i), again.get(parent), float(sk.t[i]),
                        tip=int(sk.tip[i]), rooted=bool(sk.rooted[i]), buried=bool(sk.buried[i]))
    limbs = list(again.values())
    made = _stop_short(limbs, share)
    assert len(made) == len(limbs)
    assert all(limb.keep == pytest.approx(limb.length) for limb in made)


@pytest.mark.parametrize("path", BOTH)
def test_leaves_on_current_year_only(path):
    """Every leaf sits on a node grown in the last year, and there are as many leaves at a node as
    the arrangement says."""
    rules = load(path)
    per_node = kit_param(rules, "leaves.per_node")
    tree, years = tree_of(rules)
    sk = skeleton(rules)
    f = sk.info["scale"] * UPM
    year_of = {tuple(np.round(tree.end[i], 9)): tree.year[i]
               for i in range(len(tree)) if tree.alive[i]}
    assert len(sk.host) > 20
    nodes = Counter(tuple(np.round(n / f, 9)) for n in sk.node)
    assert all(year_of[n] == years for n in nodes)
    assert set(nodes.values()) == {per_node}
    # Every leaf's node is on its host's axis, at its arc fraction.
    for leaf in range(len(sk.host)):
        point, _, _ = sk.at(int(sk.host[leaf]), float(sk.leaf_t[leaf]))
        assert np.allclose(point, sk.node[leaf], atol=1e-9)


def test_decussate_leaves(tmp_path):
    """A maple (2 leaves a node, pairs 90 degrees apart): 2 leaves per node, petioles opposite."""
    rules = load(with_params(tmp_path, TINY_SPIRAL, leaves__per_node=2, leaves__divergence_deg=90))
    sk = skeleton(rules)
    nodes = {}
    for leaf in range(len(sk.host)):
        nodes.setdefault(tuple(np.round(sk.node[leaf], 9)), []).append(leaf)
    assert len(nodes) > 10
    for pair in nodes.values():
        assert len(pair) == 2
        a, b = (sk.base[k] - sk.node[k] for k in pair)
        assert float(a @ b) / float(np.linalg.norm(a) * np.linalg.norm(b)) == pytest.approx(-1.0)


def test_blade_angle_draws_follow_the_beta():
    """The draws are theta = pi/2 x Beta(a, b); the mean of 2 theta / pi is a / (a + b)."""
    for a, b in ((2.1984, 2.2862), (6.1932, 1.8352)):
        rng = random.Random(7)
        t = np.array([2.0 * _blade_angle(rng, a, b) / math.pi for _ in range(2000)])
        assert abs(float(t.mean()) - a / (a + b)) < 0.02
        assert t.min() >= 0.0 and t.max() <= 1.0
    # The same stream gives the same draws.
    assert _blade_angle(random.Random(3), 2.0, 3.0) == _blade_angle(random.Random(3), 2.0, 3.0)


@pytest.mark.parametrize("beta", [[0.0, 2.0], [2.0, -1.0]])
def test_the_beta_must_be_positive(tmp_path, beta):
    path = with_params(tmp_path, TINY_BUDS, leaves__angle_beta=beta)
    with pytest.raises(ManifestError, match=r"leaves\.angle_beta"):
        skeleton(load(path))


def petiole_out(sk, leaf):
    """The petiole's horizontal outward unit vector, as the grower takes it: the horizontal part
    of node-to-base, or the heading of the host's shoot where that is vertical. None where the
    petiole is nearly vertical (the grower's fallback)."""
    out = sk.base[leaf] - sk.node[leaf]
    flat = np.array([out[0], 0.0, out[2]])
    n = float(np.linalg.norm(flat))
    return flat / n if n > FLAT_EPS * float(np.linalg.norm(out)) else None


@pytest.mark.parametrize("path", BOTH)
def test_blades_tilt_toward_the_petiole(path):
    """Each blade's normal leans out the way its petiole points: the horizontal part of the normal
    is under 90 degrees from the petiole's. The blade's angle from horizontal is the drawn one,
    the face looks up, and the frame is square."""
    sk = skeleton(load(path))
    seen = 0
    for leaf in range(len(sk.host)):
        mid, across, normal = sk.mid[leaf], sk.across[leaf], sk.normal[leaf]
        assert normal[1] >= 0.0 and mid[1] <= 1e-12  # the blade droops out from its petiole
        assert float(mid @ normal) == pytest.approx(0.0, abs=1e-9)
        assert float(across @ normal) == pytest.approx(0.0, abs=1e-9)
        assert float(across @ mid) == pytest.approx(0.0, abs=1e-9)
        out = petiole_out(sk, leaf)
        flat = np.array([normal[0], 0.0, normal[2]])
        if out is None or float(np.linalg.norm(flat)) < 1e-6:
            continue
        assert float(flat @ out) > 0.0
        seen += 1
    assert seen > 0.8 * len(sk.host)


@pytest.mark.parametrize("lite", [False, True])
def test_blade_faces_along_its_normal(lite):
    """The mesh winds each blade so its face looks along the leaf normal (a blade wound the other
    way shows its back to the light, a dark patch on the leaf)."""
    sk = skeleton(load(TINY_BUDS), lite=lite)
    for leaf in range(0, len(sk.host), 7):
        tris = leaf_tris(sk, leaf)
        faces = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        assert float((faces @ sk.normal[leaf]).sum()) > 0.0
        assert (faces @ sk.normal[leaf] > 0.0).all()


@pytest.mark.parametrize("path", BOTH)
def test_blade_angles_follow_the_species_beta(path):
    rules = load(path)
    a, b = kit_param(rules, "leaves.angle_beta")
    angles = []
    for seed in (1, 2, 3):
        sk = skeleton(rules, seed=seed)
        angles += [2.0 * math.acos(min(1.0, float(n[1]))) / math.pi for n in sk.normal]
    assert len(angles) > 200
    assert abs(float(np.mean(angles)) - a / (a + b)) < 0.06


@pytest.mark.parametrize("path", BOTH)
def test_roots_leave_near_the_soil(path):
    """Every root starts at most one trunk base radius above the soil and ends in it."""
    rules = load(path)
    sk = skeleton(rules)
    base_radius = float(sk.limb_radii(0)[1])  # the trunk's radius where it leaves the soil
    roots = np.flatnonzero(sk.buried)
    assert len(roots) == kit_param(rules, "roots.count")
    for c in roots:
        pts = sk.limb_points(int(c))
        assert -1e-12 <= pts[0, 1] <= base_radius
        assert pts[-1, 1] < 0.0
        out = pts[1] - pts[0]
        assert np.hypot(out[0], out[2]) > FLAT_EPS * np.linalg.norm(out)  # leaves sideways


@pytest.mark.parametrize("path", BOTH)
def test_roots_share_the_trunk_part(path):
    """Roots are level 0 (Trunk_Roots) on the trunk limb, and the chain orders of everything else
    count from the trunk chain: a root is nobody's parent."""
    sk = skeleton(load(path))
    roots = np.flatnonzero(sk.buried)
    assert len(roots)
    assert (sk.level[roots] == 0).all() and (sk.parent[roots] == 0).all()
    assert all(str(sk.limb_id[c]).startswith("root.") for c in roots)
    assert sk.level_names[0] == "Trunk_Roots"
    assert not np.isin(sk.parent, roots).any()
    first = np.flatnonzero((sk.parent == 0) & ~sk.buried)
    assert len(first) and (sk.level[first] == 1).all()


def blade_area(sk, leaves):
    """The summed area of the leaves `leaves`: the outline's area times the blade length squared."""
    x, y = sk.outline[:, 0], sk.outline[:, 1]
    polygon = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    return polygon * float(np.sum(sk.length[leaves] ** 2))


def shoot_areas(sk, tree):
    """Leaf area per shoot, a shoot being one axis' growth in one year: each leaf is put on the
    metamer whose end is its node."""
    f = sk.info["scale"] * UPM
    shoot_of = {tuple(np.round(tree.end[i], 9)): (tree.axis[i], tree.year[i])
                for i in range(len(tree)) if tree.alive[i]}
    by_shoot = {}
    for leaf in range(len(sk.host)):
        by_shoot.setdefault(shoot_of[tuple(np.round(sk.node[leaf] / f, 9))], []).append(leaf)
    return {key: blade_area(sk, leaves) for key, leaves in by_shoot.items()}


@pytest.mark.parametrize("path", BOTH)
def test_lite_keeps_leaf_area(path):
    """Lite thins the leaves and enlarges the rest: the same wood, fewer leaves, and each shoot
    keeps its leaf area to within 10% of the full tree's (so does the whole tree)."""
    rules = load(path)
    full, lite = skeleton(rules), skeleton(rules, lite=True)
    tree, _ = tree_of(rules)
    assert np.array_equal(full.pts, lite.pts)
    assert 0.3 * len(full.host) < len(lite.host) < 0.7 * len(full.host)
    assert lite.outline.shape != full.outline.shape  # lite has its own, simpler outline
    full_areas, lite_areas = shoot_areas(full, tree), shoot_areas(lite, tree)
    assert set(full_areas) == set(lite_areas) and len(full_areas) > 10
    for key, area in full_areas.items():
        assert lite_areas[key] == pytest.approx(area, rel=0.10), key
    assert blade_area(lite, np.arange(len(lite.host))) == pytest.approx(
        blade_area(full, np.arange(len(full.host))), rel=0.10)


@pytest.mark.parametrize("path", BOTH)
def test_every_param_is_read(path):
    """After a full and a lite grow, every species and kit param has been read; the tube sides are
    the budget params the mesh reads, and they are exempt."""
    rules = load(path)
    seen = set()
    for lite in (False, True):
        reader = RuleReader(rules)
        _read(reader, lite)
        seen |= reader.read
    assert set(rules.params) - seen - {"tubes.sides", "lite.tube_sides"} == set()


@pytest.mark.parametrize("lite", [False, True])
def test_an_unread_param_raises(tmp_path, lite):
    data = json.loads(TINY_BUDS.read_text())
    data["params"]["trunk.spare_m"] = {"value": 1.0, "rows": ["twig.tip"], "tag": "measured"}
    path = tmp_path / "spare.rules.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match=r"trunk\.spare_m"):
        skeleton(load(path), lite=lite)


def test_a_small_tree_has_leaves_and_roots():
    """The one-year case with the leaves and roots on: 0.5 m and one year."""
    for path in BOTH:
        sk = skeleton(load(path), height_m=0.5)
        check_skeleton(sk)
        assert len(sk.host) > 0 and sk.buried.any()
        parts = grown_parts(sk, None)
        assert all(np.isfinite(t).all() for t in parts.values())
