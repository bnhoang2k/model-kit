"""The pieces the tree-file check tests grow: the two tiny fixture species, grown by `grow_buds`
at test_buds' default height (1.5 m), seeds 1 and 2 (a rule fixed before any check was run)."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pytest

from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import Skeleton

from .branching_scene import UPM

FIXTURES = Path(__file__).resolve().parents[2] / "grow" / "fixtures"
SPECIES = [FIXTURES / "tiny-buds.rules.json", FIXTURES / "tiny-spiral.rules.json"]
REDBUD_SEED1 = FIXTURES / "redbud-seed1.skeleton.npz"  # a redbud grown from seed 1, 5.0 m
HEIGHT_M = 1.5
SEEDS = (1, 2)
PIECES = [pytest.param(path, seed, id=f"{path.name.split('.')[0]}-seed{seed}")
          for path in SPECIES for seed in SEEDS]


@cache
def tiny_tree(path: Path, seed: int) -> Skeleton:
    rules = load_rules(path, kit=KIT_RULES)
    return grow_buds(rules, piece="Tree", height_m=HEIGHT_M, seed=seed, lite=False,
                     units_per_metre=UPM)
