import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks.branching.leaf_angle import (
    LeafAngle,
    blade_angles,
    spreads,
    third_of,
    turned,
)
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    LEAF_ANGLE_MEAN_MAX_DEG,
    LEAF_ANGLE_MEAN_MIN_DEG,
    LEAF_ANGLE_MIN_LEAVES,
    LEAF_ANGLE_PLANT_STEEP_DEG,
    LEAF_ANGLE_SD_MIN_DEG,
)

from ...grow.qsm_fixtures import QSM_FILES
from .branching_scene import UPM, grown_ctx, item, plants_caught, small_tree
from .tiny_trees import HEIGHT_M, PIECES, REDBUD_SEED1, SPECIES, tiny_tree

CHECK = LeafAngle()
REFS = json.loads((Path(__file__).parent / "leaf_angle_refs.json").read_text())
THIRD_ITEMS = [item(f"third.{k}") for k in range(3)]


def redbud_seed1():
    sk = load_skeleton(REDBUD_SEED1)
    assert (sk.info["seed"], sk.info["height_m"]) == (1, 5.0)  # the fixture is a redbud grown from seed 1, 5.0 m
    return sk


def facing(sk):
    """The horizontal part of each blade's normal, as a unit vector (the way it faces)."""
    n = sk.normal * np.array([1.0, 0.0, 1.0]) * np.where(sk.normal[:, 1:2] >= 0, 1.0, -1.0)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


# --- the measure -------------------------------------------------------------------------------


def test_a_blade_angle_is_from_horizontal():
    sk = small_tree()
    flat = turned(sk, 0.0)
    assert blade_angles(flat) == pytest.approx(np.zeros(len(sk.host)), abs=1e-6)
    assert np.allclose(np.abs(flat.normal[:, 1]), 1.0)  # a flat blade's face looks straight up
    upright = turned(sk, 90.0)
    assert blade_angles(upright) == pytest.approx(np.full(len(sk.host), 90.0))
    assert np.allclose(upright.normal[:, 1], 0.0, atol=1e-12)  # a vertical blade's face is level
    assert blade_angles(sk) == pytest.approx(np.degrees(np.arccos(np.abs(sk.normal[:, 1]))))


def test_turning_keeps_the_frame_and_the_facing():
    for sk in (tiny_tree(SPECIES[0], 1), redbud_seed1()):
        before = facing(sk)
        for degrees in (0.0, 30.0, 85.0):
            moved = turned(sk, degrees)
            assert blade_angles(moved) == pytest.approx(np.full(len(sk.host), degrees), abs=1e-6)
            if degrees:  # a flat blade has no horizontal part to face with
                assert facing(moved) == pytest.approx(before, abs=1e-6)
            assert np.abs((moved.mid * moved.normal).sum(axis=1)).max() < 1e-9  # still a frame
            assert np.abs((moved.across * moved.normal).sum(axis=1)).max() < 1e-9
            assert np.linalg.norm(moved.mid, axis=1) == pytest.approx(np.linalg.norm(sk.mid,
                                                                                    axis=1))
            assert np.array_equal(moved.base, sk.base)  # turned about its base, not moved


def test_the_thirds_are_equal_height_slices_of_the_leaf_nodes():
    sk = small_tree()
    ys = sk.node[:, 1]
    which = third_of(sk)
    assert which[np.argmin(ys)] == 0
    assert which[np.argmax(ys)] == 2  # the top leaf is in the top third, not past it
    sk = dataclasses.replace(sk, node=np.array([[0, 0.0, 0], [0, 0.4, 0], [0, 0.9, 0]]))
    assert third_of(sk).tolist() == [0, 1, 2]
    level = dataclasses.replace(sk, node=np.ones((3, 3)))
    assert third_of(level).tolist() == [0, 0, 0]  # one height: no spread to divide


# --- the measured data -------------------------------------------------------------------------


def test_the_band_and_the_floor_come_from_the_dataset():
    """The constants in tolerances.py follow from the file's per-species mean, SD and n (the file
    holds them rounded to two places, so the widened ends agree to a hundredth)."""
    low = min(REFS.items(), key=lambda kv: kv[1]["mean_deg"])
    high = max(REFS.items(), key=lambda kv: kv[1]["mean_deg"])
    assert low[0] == "Carpinus betulus"
    assert high[0] == "Liquidambar styraciflua"
    for (_, ref), sign, want in ((low, -1, LEAF_ANGLE_MEAN_MIN_DEG),
                                 (high, 1, LEAF_ANGLE_MEAN_MAX_DEG)):
        widened = ref["mean_deg"] + sign * 2 * ref["sd_deg"] / math.sqrt(ref["n"])
        assert widened == pytest.approx(want, abs=0.011)
    assert min(r["sd_deg"] for r in REFS.values()) == LEAF_ANGLE_SD_MIN_DEG
    assert min(r["n"] for r in REFS.values()) >= LEAF_ANGLE_MIN_LEAVES


def test_measured_species_pass():
    """Every species of the dataset lies inside the mean band and spreads its angles by at least
    the floor. This is what the file can prove: it holds one mean, SD and n per species, not the
    per-sector sets. So it does not prove that every third of a species' crown reaches the floor;
    the floor is the smallest SD over all of the dataset's sets, whole canopies included, which is
    the highest value no measured set contradicts."""
    assert len(REFS) == 138
    for name, ref in REFS.items():
        assert LEAF_ANGLE_MEAN_MIN_DEG <= ref["mean_deg"] <= LEAF_ANGLE_MEAN_MAX_DEG, name
        assert ref["sd_deg"] >= LEAF_ANGLE_SD_MIN_DEG, name


# --- the judgement -----------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_in_band_passes(path, seed):
    ctx = grown_ctx(tiny_tree(path, seed))
    assert CHECK.run(ctx) == []
    assert not [c for c in CHECK.coverage(ctx) if isinstance(c, Unmeasured)]
    assert run_gate([CHECK], ctx, None).status == "passed"


def test_leaf_angle_catches_plates():
    """Blades all at one angle: every third's SD falls to 0, the mean (its own) stays in band."""
    for sk in (tiny_tree(SPECIES[0], 1), tiny_tree(SPECIES[1], 1), redbud_seed1()):
        whole, _ = spreads(sk)
        assert CHECK.run(grown_ctx(sk)) == []
        plate = turned(sk, whole.mean)
        found = CHECK.run(grown_ctx(plate))
        assert [f.item for f in found] == THIRD_ITEMS
        assert all("under the smallest measured" in f.detail for f in found)
        assert [s.sd for s in spreads(plate)[1]] == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)


def test_the_seed1_redbud_spreads_its_angles_so_this_check_cannot_see_its_plates():
    """The seed-1 redbud passes the floor in every third:
    its blade angles spread 12.6 to 12.9 deg in every third (a draw from a distribution), above
    9.69. Its plates are in the way the blades face, which a blade angle does not measure. Nothing
    is tuned to change this."""
    sk = redbud_seed1()
    whole, thirds = spreads(sk)
    assert LEAF_ANGLE_MEAN_MIN_DEG <= whole.mean <= LEAF_ANGLE_MEAN_MAX_DEG
    assert [round(t.sd, 1) for t in thirds] == [12.6, 12.8, 12.9]
    assert all(t.sd >= LEAF_ANGLE_SD_MIN_DEG and t.n >= LEAF_ANGLE_MIN_LEAVES for t in thirds)
    assert CHECK.run(grown_ctx(sk)) == []


def test_leaf_angle_catches_an_impossible_species(tmp_path):
    """A species file whose blade-angle beta has a mean of 85 deg."""
    data = json.loads(SPECIES[0].read_text())
    data["params"]["leaves.angle_beta"]["value"] = [17.0, 1.0]  # 90 x 17 / 18 = 85 deg
    path = tmp_path / "steep.rules.json"
    path.write_text(json.dumps(data))
    sk = grow_buds(load_rules(path, kit=KIT_RULES), piece="Tree", height_m=HEIGHT_M, seed=1,
                   lite=False, units_per_metre=UPM)
    ctx = grown_ctx(sk)
    found = CHECK.run(ctx)
    assert item("mean") in [f.item for f in found]
    assert "outside the measured range" in next(f for f in found if f.item == item("mean")).detail
    assert run_gate([CHECK], ctx, None).status == "failed"
    # The same on the tree of a normal species, its blades set to 85 deg.
    steep = CHECK.run(grown_ctx(turned(tiny_tree(SPECIES[0], 1), LEAF_ANGLE_PLANT_STEEP_DEG)))
    assert item("mean") in [f.item for f in steep]


def test_flat_blades_fail_the_low_end():
    flat = turned(tiny_tree(SPECIES[0], 1), 0.0)
    assert item("mean") in [f.item for f in CHECK.run(grown_ctx(flat))]


@pytest.mark.parametrize("path", QSM_FILES)
def test_leaf_angle_leafless_not_applicable(path):
    """A scan has no leaves: no finding, no raise, and not `Unmeasured`."""
    ctx = grown_ctx(load_skeleton(path))
    assert CHECK.run(ctx) == []
    assert CHECK.plants(ctx) == []
    assert CHECK.measured(ctx) == set()
    lines = CHECK.coverage(ctx)
    assert not [c for c in lines if isinstance(c, Unmeasured)]
    assert any("leaf-angle: not applicable (no leaves)" in c and "grove:Tree" in c for c in lines)
    assert run_gate([CHECK], ctx, None).status == "passed"


def test_too_few_leaves_are_not_judged():
    sk = small_tree()  # three leaves
    assert len(sk.host) < LEAF_ANGLE_MIN_LEAVES
    ctx = grown_ctx(turned(sk, LEAF_ANGLE_PLANT_STEEP_DEG))
    assert CHECK.run(ctx) == []  # 85 deg, but too few leaves to judge
    assert CHECK.plants(ctx) == []
    lines = CHECK.coverage(ctx)
    assert not [c for c in lines if isinstance(c, Unmeasured)]
    assert any("not judged (3 leaves, fewer than 26)" in c and c.endswith(item("mean"))
               for c in lines)
    assert run_gate([CHECK], ctx, None).status == "passed"


def test_a_sparse_third_is_not_judged_but_the_others_are():
    sk = tiny_tree(SPECIES[0], 1)
    order = np.argsort(sk.node[:, 1])
    keep = np.sort(np.concatenate([order[:40], order[-3:]]))  # 40 low leaves, 3 at the very top
    leaf = {k: getattr(sk, k)[keep] for k in (
        "host", "leaf_t", "node", "base", "mid", "across", "normal", "length", "petiole", "shade")}
    sparse = dataclasses.replace(sk, **leaf)
    whole, thirds = spreads(sparse)
    assert [t.n >= LEAF_ANGLE_MIN_LEAVES for t in thirds] == [True, False, False]
    assert whole.n >= LEAF_ANGLE_MIN_LEAVES
    ctx = grown_ctx(turned(sparse, whole.mean), listed_from=sk)
    assert [f.item for f in CHECK.run(ctx)] == THIRD_ITEMS[:1]  # the judged third: a plate
    lines = CHECK.coverage(ctx)
    for rest in (THIRD_ITEMS[1], THIRD_ITEMS[2]):
        assert any("not judged" in c and c.endswith(rest) for c in lines)


# --- the plants --------------------------------------------------------------------------------


def test_a_plate_is_planted_for_each_judged_third_and_for_no_other():
    """Units follow the same count rule as the findings: a third with fewer leaves than
    `LEAF_ANGLE_MIN_LEAVES` is not judged, so it gets no plant (coverage says so)."""
    sk = tiny_tree(SPECIES[0], 1)
    order = np.argsort(sk.node[:, 1])
    keep = np.sort(np.concatenate([order[:40], order[-3:]]))  # 40 low leaves, 3 at the very top
    leaf = {k: getattr(sk, k)[keep] for k in (
        "host", "leaf_t", "node", "base", "mid", "across", "normal", "length", "petiole", "shade")}
    ctx = grown_ctx(dataclasses.replace(sk, **leaf), listed_from=sk)
    assert [(u.item, u.candidates[0].label.split(":")[0]) for u in CHECK.plants(ctx)] == [
        (THIRD_ITEMS[0], "plates"), (item("mean"), "steep"), (item("mean"), "flat")]


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_plants_on_two_species(path, seed):
    ctx = grown_ctx(tiny_tree(path, seed))
    assert CHECK.run(ctx) == []  # the piece passes unplanted first
    caught = plants_caught(CHECK, ctx)
    assert [label.split(":")[0] for label, _ in caught] == ["plates"] * 3 + ["steep", "flat"]
    assert all(ok for _, ok in caught), caught
    assert run_gate([CHECK], ctx, None).status == "passed"  # run_gate runs every plant too


def test_the_plants_reach_the_seed1_redbud():
    ctx = grown_ctx(redbud_seed1())
    caught = plants_caught(CHECK, ctx)
    assert len(caught) == 5  # a plate for each judged third, then steep and flat
    assert all(ok for _, ok in caught), caught
