import dataclasses
import json
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks.branching.reference_range import (
    ReferenceRange,
    crown_base,
    height,
    spread,
)
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import limb_at, load_skeleton
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import RANGE_PLANT_OVER

from ...grow.qsm_fixtures import QSM_FILES
from ..piece_scenes import caught
from .branching_scene import (
    UPM,
    Limb,
    grown_ctx,
    item,
    line,
    make_skeleton,
    plants_caught,
    small_tree,
)
from .tiny_trees import REDBUD_SEED1

CHECK = ReferenceRange()
FIXTURES = Path(__file__).resolve().parents[2] / "grow" / "fixtures"
SPECIES = [FIXTURES / "tiny-buds.rules.json", FIXTURES / "tiny-spiral.rules.json"]
# The pieces every test grows, by a rule fixed before any range was read: test_buds' default
# height (1.5 m), seeds 1 and 2, on both tiny species.
HEIGHT_M = 1.5
SEEDS = (1, 2)
PIECES = [pytest.param(path, seed, id=f"{path.name.split('.')[0]}-seed{seed}")
          for path in SPECIES for seed in SEEDS]


@cache
def rules_of(path: Path):
    return load_rules(path, kit=KIT_RULES)


@cache
def tree(path: Path, seed: int):
    return grow_buds(rules_of(path), piece="Tree", height_m=HEIGHT_M, seed=seed, lite=False,
                     units_per_metre=UPM)


def scene(sk, rules=None, name=None):
    """A scene holding sk, with the rules file it names (or `rules` under `name`)."""
    ctx = grown_ctx(sk)
    if rules is None:
        return ctx
    return dataclasses.replace(ctx, rules={name or rules.path.name: rules})


def piece_scene(path: Path, seed: int):
    return scene(tree(path, seed), rules_of(path))


def widened(sk, factor):
    """The crown scaled across x and z about the origin (the trees stand there)."""
    def across(a):
        out = np.array(a)
        out[:, [0, 2]] *= factor
        return out
    return dataclasses.replace(sk, pts=across(sk.pts), node=across(sk.node), base=across(sk.base))


# --- the measures ------------------------------------------------------------------------------


def _rhombus(angle_deg: float = 30.0, a: float = 2.0, b: float = 1.0):
    """A trunk with four bare limbs reaching the ends of an ellipse's axes (semi-axes a, b),
    turned by angle_deg in plan view: the widest width is 2a, the width across it 2b."""
    turn = np.radians(angle_deg)
    ends = [(a, 0.0), (-a, 0.0), (0.0, b), (0.0, -b)]
    limbs = [Limb("trunk", 0, -1, 0.0, line((0, 0, 0), (0, 1, 0), 1.0), np.full(5, 0.05),
                  rooted=True)]
    for n, (u, v) in enumerate(ends):
        x, z = u * np.cos(turn) - v * np.sin(turn), u * np.sin(turn) + v * np.cos(turn)
        limbs.append(Limb(f"branch.{n}", 1, 0, 1.0, line((0, 1, 0), (x, 0.0, z), np.hypot(x, z)),
                          np.full(5, 0.02)))
    return make_skeleton(limbs, [])


def test_spread_is_the_mean_of_the_widest_width_and_the_width_across():
    sk = _rhombus()
    assert spread(sk) == pytest.approx(0.5 * (4.0 + 2.0))
    assert spread(_rhombus(angle_deg=75.0)) == pytest.approx(3.0)  # turning the crown changes nothing
    assert height(sk) == pytest.approx(1.0)
    assert crown_base(sk) is None  # no leaves, no crown base


def test_spread_counts_leaf_vertices_and_leaves_out_roots():
    sk = small_tree()
    root = int(np.flatnonzero(sk.buried)[0])
    pts = np.array(sk.pts)
    pts[sk.start[root]:sk.start[root + 1], 0] += 10.0  # the root reaches far out: not the crown
    assert spread(dataclasses.replace(sk, pts=pts)) == pytest.approx(spread(sk))
    base = np.array(sk.base)
    base[0, 0] += 5.0  # one blade far out: the crown reaches it
    assert spread(dataclasses.replace(sk, base=base)) > spread(sk) + 2.0  # the widest width grows by about 5, the mean by half


def test_crown_base_is_the_lowest_leaf():
    sk = small_tree()
    assert crown_base(sk) == pytest.approx(float(sk.node[:, 1].min()))
    assert crown_base(sk) > 0.0


def test_a_stem_seated_at_the_soil_does_not_drop_the_crown_base():
    sk = small_tree()  # a second stem seated at the soil, with its leaves high up
    child = next(int(h) for h in np.unique(sk.host) if sk.parent[h] >= 0)
    pts = np.array(sk.pts)
    pts[sk.start[child], 1] = 0.0
    assert crown_base(dataclasses.replace(sk, pts=pts)) == pytest.approx(crown_base(sk))
    assert crown_base(sk) > 0.0


def test_a_leaf_on_the_trunk_chain_counts_at_its_own_height():
    sk = small_tree()
    ts = (0.5, 0.6, 0.7)
    nodes = np.array([limb_at(sk.limb_points(0), sk.limb_radii(0), t)[0] for t in ts])
    low = dataclasses.replace(sk, host=np.zeros(3, dtype=int), leaf_t=np.array(ts), node=nodes)
    assert crown_base(low) == pytest.approx(float(nodes[:, 1].min()))  # not the soil, not 0


# --- the judgement -----------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_in_range_passes(path, seed):
    ctx = piece_scene(path, seed)
    assert CHECK.run(ctx) == []
    assert not [c for c in CHECK.coverage(ctx) if isinstance(c, Unmeasured)]
    assert run_gate([CHECK], ctx, None).status == "passed"


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_out_of_range_fails_naming_the_key(path, seed):
    sk, rules = tree(path, seed), rules_of(path)
    wide, narrow = widened(sk, 3.0), widened(sk, 0.1)
    for moved, word in ((wide, "outside"), (narrow, "outside")):
        found = CHECK.run(scene(moved, rules))
        assert [f.item for f in found] == [item("spread_of_height")]
        assert word in found[0].detail
        assert "size.spread" in found[0].detail  # the cited row
        assert f"{rules.ranges['spread_of_height'].lo}" in found[0].detail
    assert spread(wide) > spread(sk) > spread(narrow)


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_plants_on_two_species(path, seed):
    ctx = piece_scene(path, seed)
    assert CHECK.run(ctx) == []  # the piece passes unplanted first
    caught = plants_caught(CHECK, ctx)
    assert [label.split(":")[0] for label, _ in caught] == ["too wide", "bare below"]
    assert all(ok for _, ok in caught), caught
    assert run_gate([CHECK], ctx, None).status == "passed"  # run_gate runs every plant too


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_the_wide_plant_goes_past_its_target(path, seed):
    """The plant is blind to `spread`: it scales the wood by a lower bound, so the spread lands
    at the target or past it."""
    sk, rules = tree(path, seed), rules_of(path)
    unit = next(u for u in CHECK.plants(piece_scene(path, seed))
                if u.item.endswith("spread_of_height"))
    plant = unit.candidates[0]
    moved = plant.ctx.skeletons[next(iter(plant.ctx.skeletons))]
    goal = RANGE_PLANT_OVER * rules.ranges["spread_of_height"].hi
    assert spread(moved) / height(moved) >= goal * (1 - 1e-9)
    assert height(moved) == pytest.approx(height(sk))  # only the crown's width changes


def test_the_bare_plant_lifts_the_crown_base():
    path, seed = SPECIES[0], 1
    rules = rules_of(path)
    unit = next(u for u in CHECK.plants(piece_scene(path, seed))
                if u.item.endswith("crown_base_of_height"))
    plant = unit.candidates[0]
    moved = plant.ctx.skeletons[next(iter(plant.ctx.skeletons))]
    assert crown_base(moved) / height(moved) >= RANGE_PLANT_OVER * rules.ranges[
        "crown_base_of_height"].hi
    assert len(moved.host) < len(tree(path, seed).host)


def test_the_bare_plant_lifts_the_leaves_when_every_leaf_is_low():
    """No leaf sits above the line to strip down to: the plant lifts all leaves instead."""
    path, seed = SPECIES[0], 1
    sk, rules = tree(path, seed), rules_of(path)
    line = RANGE_PLANT_OVER * rules.ranges["crown_base_of_height"].hi * height(sk)
    low = sk.node[:, 1] < line
    kept = {k: getattr(sk, k)[low] for k in (
        "host", "leaf_t", "node", "base", "mid", "across", "normal", "length", "petiole", "shade")}
    ctx = scene(dataclasses.replace(sk, **kept), rules)
    unit = next(u for u in CHECK.plants(ctx) if u.item.endswith("crown_base_of_height"))
    assert [p.label.split(":")[0] for p in unit.candidates] == ["lifted"]
    plant = unit.candidates[0]
    moved = plant.ctx.skeletons[next(iter(plant.ctx.skeletons))]
    assert len(moved.host) == len(kept["host"])
    assert crown_base(moved) / height(moved) >= RANGE_PLANT_OVER * rules.ranges[
        "crown_base_of_height"].hi * (1 - 1e-9)
    assert (unit.item, True) in [(u.item, any(caught(CHECK, p) for p in u.candidates))
                                 for u in CHECK.plants(ctx)]


# --- the files behind the ranges ---------------------------------------------------------------


def changed_copy(source: Path, directory: Path, name: str | None = None) -> Path:
    """The same file with one more newline: valid, and a different hash."""
    target = directory / (name or source.name)
    target.write_bytes(source.read_bytes() + b"\n")
    return target


def test_stale_rules_file_fails_naming_both_hashes(tmp_path):
    path, seed = SPECIES[0], 1
    sk = tree(path, seed)
    now = load_rules(changed_copy(path, tmp_path), kit=KIT_RULES)
    assert now.sha256 != sk.info["rules_sha256"]
    found = CHECK.run(scene(sk, now))
    # Only the stale file is reported: the ranges belong to another file, so they are not judged.
    assert [f.item for f in found] == [item("rules")]
    assert sk.info["rules_sha256"] in found[0].detail
    assert now.sha256 in found[0].detail
    assert run_gate([CHECK], scene(sk, now), None).status == "failed"


def test_stale_kit_file_fails_naming_both_hashes(tmp_path):
    path, seed = SPECIES[0], 1
    kit = changed_copy(KIT_RULES, tmp_path, "kit.rules.json")
    grown = grow_buds(load_rules(path, kit=kit), piece="Tree", height_m=HEIGHT_M, seed=seed,
                      lite=False, units_per_metre=UPM)
    now = rules_of(path)  # the shipped kit file, as the scene has it
    assert grown.info["kit_sha256"] != now.kit_sha256
    assert grown.info["rules_sha256"] == now.sha256
    found = CHECK.run(scene(grown, now))
    assert [f.item for f in found] == [item("kit")]
    assert grown.info["kit_sha256"] in found[0].detail
    assert now.kit_sha256 in found[0].detail


def test_missing_rules_file_is_unmeasured():
    sk = tree(SPECIES[0], 1)
    ctx = scene(sk)  # the scene has no rules/ file
    assert CHECK.run(ctx) == []
    assert CHECK.plants(ctx) == []
    skipped = [c for c in CHECK.coverage(ctx) if isinstance(c, Unmeasured)]
    assert len(skipped) == 1
    assert "tiny-buds.rules.json" in skipped[0]
    assert run_gate([CHECK], ctx, None).status == "partial"


def test_wrongly_named_rules_file_is_missing():
    rules = rules_of(SPECIES[1])  # the scene holds the other species' file, under its own name
    ctx = scene(tree(SPECIES[0], 1), rules)
    assert any(isinstance(c, Unmeasured) for c in CHECK.coverage(ctx))


@pytest.mark.parametrize("make", [
    pytest.param(lambda: grown_ctx(load_skeleton(REDBUD_SEED1)), id="redbud-seed1"),
    pytest.param(lambda: grown_ctx(small_tree()), id="hand-made"),
    pytest.param(lambda: grown_ctx(load_skeleton(QSM_FILES[2])), id="measured-ash"),
])
def test_a_skeleton_without_hashes_is_not_applicable(make):
    ctx = make()
    assert CHECK.run(ctx) == []
    assert CHECK.plants(ctx) == []
    assert CHECK.measured(ctx) == set()
    lines = CHECK.coverage(ctx)
    assert not [c for c in lines if isinstance(c, Unmeasured)]
    assert any("reference-range: not applicable" in c and "grove:Tree" in c for c in lines)
    assert run_gate([CHECK], ctx, None).status == "passed"


def test_fork_range_is_not_judged(tmp_path):
    data = json.loads(SPECIES[0].read_text())
    data["ranges"]["fork_of_height"] = {"lo": 0.05, "hi": 0.06, "rows": ["size.spread"]}
    path = tmp_path / "tiny-fork.rules.json"
    path.write_text(json.dumps(data))
    rules = load_rules(path, kit=KIT_RULES)
    assert "fork_of_height" in rules.ranges
    sk = grow_buds(rules, piece="Tree", height_m=HEIGHT_M, seed=1, lite=False,
                   units_per_metre=UPM)
    ctx = scene(sk, rules)
    assert CHECK.run(ctx) == []  # no finding, whatever the tree's forks are
    fork = [c for c in CHECK.coverage(ctx) if "fork_of_height" in c]
    assert len(fork) == 1
    assert not isinstance(fork[0], Unmeasured)
    assert "taste-only" in fork[0]
    assert run_gate([CHECK], ctx, None).status == "passed"
    assert not [c for c in CHECK.coverage(piece_scene(SPECIES[0], 1)) if "fork_of_height" in c]


def test_leafless_tree_has_no_crown_base_to_judge():
    sk = dataclasses.replace(tree(SPECIES[0], 1), **{
        k: getattr(tree(SPECIES[0], 1), k)[:0] for k in (
            "host", "leaf_t", "node", "base", "mid", "across", "normal", "length", "petiole",
            "shade")})
    ctx = scene(sk, rules_of(SPECIES[0]))
    assert crown_base(sk) is None
    assert [f.item for f in CHECK.run(ctx) if f.item.endswith("crown_base_of_height")] == []
    lines = CHECK.coverage(ctx)
    assert any("no crown base" in c for c in lines)
    assert not [c for c in lines if isinstance(c, Unmeasured)]
