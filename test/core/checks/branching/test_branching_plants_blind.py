"""The guard for the branching checks' plants : a check whose verdict never reports a
finding cannot see its plant, and the gate stops the run with `BlindCheck`.

Each case blinds one rule of one check on a clean scene of its own by patching the function its
verdict rests on (not by subclassing `run`). A case that blinds only one rule leaves the check's
other plants caught, so it shows the unit of that rule is tried and stops the run.

A plant built without calling the verdict (a plain `Plant`, or one that reads a measure but never
the verdict function) is wrapped by the gate, so a blind check stops at once. A plant must also
cover every judged item: where `pair_plant`, `_bare_plant`, `_plant` of skeleton-match or an
`if p is not None` filter drops the plant because none can be built, a blind check would pass.
The cases named "seedling", "every leaf low" and "no vertex stands clear" guard those.

Below the guard: the plants never call the verdict, two pieces per check are tried and caught, and
the pieces a plant must also reach.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core.checks.branching import (
    CrossingOrganic,
    FloorOrganic,
    LeafAngle,
    LimbSeat,
    NearMissOrganic,
    ReferenceRange,
    SkeletonMatch,
    Taper,
    crossing,
    floor,
    leaf_angle,
    limb_seat,
    near_miss,
    reference_range,
    skeleton_match,
    taper,
)
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from ..piece_scenes import refuses_the_core, tried_and_caught
from .branching_scene import (
    PACK,
    PIECE,
    bud_ctx,
    bud_tree,
    card_ctx,
    grown_ctx,
    make_skeleton,
    small_leaves,
    small_limbs,
)
from .test_reference_range import SPECIES as RANGE_SPECIES
from .test_reference_range import piece_scene
from .tiny_trees import SPECIES, tiny_tree

# --- the scenes: two named pieces per check ----------------------------------------------------

BUDS, SPIRAL = "tiny-buds seed 1 1.5 m", "tiny-spiral seed 1 1.5 m"


def bud(species):
    return lambda: bud_ctx(species, 1, 1.5)


def tree(index):
    return lambda: grown_ctx(tiny_tree(SPECIES[index], 1))


def ranged(index):
    return lambda: piece_scene(RANGE_SPECIES[index], 1)


def card(species):
    """A card copy of the tiny tree, beside its source: the check judges both."""
    return lambda: card_ctx(bud_tree(species, 1, 1.5))


def seedling(species):
    return lambda: bud_ctx(species, 1, 0.5)


def leafless():
    """The small tree with no leaves: its free limbs are what the floor rule reads."""
    return grown_ctx(make_skeleton(small_limbs(), []))


TWO_PIECES = {
    "crossing-organic": (CrossingOrganic, {BUDS: bud("tiny-buds"), SPIRAL: bud("tiny-spiral")}),
    "near-miss-organic": (NearMissOrganic, {BUDS: bud("tiny-buds"), SPIRAL: bud("tiny-spiral")}),
    "limb-seat": (LimbSeat, {BUDS: bud("tiny-buds"), SPIRAL: bud("tiny-spiral")}),
    "floor-organic": (FloorOrganic, {BUDS: bud("tiny-buds"), SPIRAL: bud("tiny-spiral")}),
    "skeleton-match": (SkeletonMatch, {BUDS: bud("tiny-buds"), SPIRAL: bud("tiny-spiral")}),
    "floor-organic cards": (FloorOrganic, {BUDS: card("tiny-buds"), SPIRAL: card("tiny-spiral")}),
    "skeleton-match cards": (SkeletonMatch, {BUDS: card("tiny-buds"), SPIRAL: card("tiny-spiral")}),
    "leaf-angle": (LeafAngle, {BUDS: tree(0), SPIRAL: tree(1)}),
    "taper": (Taper, {BUDS: tree(0), SPIRAL: tree(1)}),
    "reference-range": (ReferenceRange, {BUDS: ranged(0), SPIRAL: ranged(1)}),
}
PIECE_CASES = [pytest.param(name, piece, id=f"{name}-{piece.split()[0]}")
               for name, (_, pieces) in TWO_PIECES.items() for piece in pieces]


@pytest.mark.parametrize(("name", "piece"), PIECE_CASES)
def test_plants_are_tried_and_caught_on_two_pieces(name, piece):
    make_check, pieces = TWO_PIECES[name]
    check, ctx = make_check(), pieces[piece]()
    assert check.run(ctx) == []  # a clean piece: nothing hides a plant
    assert tried_and_caught(check, ctx)  # at least one unit tried, and every one tried is caught


# --- the guard ---------------------------------------------------------------------------------


def nothing(*args, **kwargs):
    return []


def without(module, name, drops):
    """Patch module.name so the findings whose item `drops` accepts are lost: one rule blinded."""
    real = getattr(module, name)

    def blinded(*args, **kwargs):
        return [f for f in real(*args, **kwargs) if not drops(f.item)]

    return lambda monkeypatch: monkeypatch.setattr(module, name, blinded)


def patched(module, name, value):
    return lambda monkeypatch: monkeypatch.setattr(module, name, value)


CASES = {
    "crossing": (CrossingOrganic, patched(crossing, "_findings", nothing), bud("tiny-spiral")),
    "crossing seedling": (CrossingOrganic, patched(crossing, "_findings", nothing),
                          seedling("tiny-buds")),
    "near-miss": (NearMissOrganic, patched(near_miss, "_findings", nothing), bud("tiny-buds")),
    "near-miss seedling": (NearMissOrganic, patched(near_miss, "_findings", nothing),
                           seedling("tiny-spiral")),
    "limb-seat limb": (LimbSeat, patched(limb_seat, "_limb_findings", nothing), bud("tiny-buds")),
    "limb-seat leaf": (LimbSeat, patched(limb_seat, "_leaf_findings", nothing), bud("tiny-buds")),
    "floor rooted limb": (FloorOrganic, patched(floor, "_limb_findings", nothing),
                          bud("tiny-spiral")),
    "floor leaf": (FloorOrganic, patched(floor, "_leaf_low", lambda sk, f: 0.0),
                   bud("tiny-spiral")),
    "floor free limb": (FloorOrganic, patched(floor, "_low_past", lambda *a: np.inf), leafless),
    "floor card": (FloorOrganic, patched(floor, "_card_lows", lambda tris: np.zeros(len(tris) // 2)),
                   card("tiny-spiral")),
    "skeleton-match": (SkeletonMatch, patched(skeleton_match, "_part_findings", nothing),
                       bud("tiny-buds")),
    "skeleton-match card": (SkeletonMatch, patched(skeleton_match, "_part_findings", nothing),
                            card("tiny-buds")),
    "taper": (Taper, patched(taper, "_findings", nothing), tree(0)),
    "leaf-angle third": (LeafAngle, without(leaf_angle, "_findings", lambda i: " third." in i),
                         tree(0)),
    "leaf-angle mean": (LeafAngle, without(leaf_angle, "_findings", lambda i: i.endswith(" mean")),
                        tree(1)),
    "reference-range spread": (ReferenceRange, without(
        reference_range, "_share_findings", lambda i: i.endswith(reference_range.SPREAD)),
        ranged(0)),
    "reference-range crown base": (ReferenceRange, without(
        reference_range, "_share_findings", lambda i: i.endswith(reference_range.CROWN_BASE)),
        ranged(0)),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_a_blind_check_stops_the_run(name, monkeypatch):
    make_check, blind, make = CASES[name]
    ctx = make()
    run_gate([make_check()], ctx, None)  # the clean scene, with its plants, does not stop
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=make_check().name):
        run_gate([make_check()], ctx, None)


def test_a_crown_base_with_every_leaf_low_is_still_planted_and_stops_a_blind_check(monkeypatch):
    """The range's upper end is raised so no leaf is above the plant's line: a plant
    that is None there would let a blind check pass, so the plant lifts the leaves instead."""
    ctx = piece_scene(RANGE_SPECIES[0], 1)
    rules = next(iter(ctx.rules.values()))
    high = dataclasses.replace(rules.ranges["crown_base_of_height"], hi=2.0, lo=0.0)
    wide = dataclasses.replace(rules, ranges={**rules.ranges, "crown_base_of_height": high})
    ctx = dataclasses.replace(ctx, rules={rules.path.name: wide})
    check = ReferenceRange()
    run_gate([check], ctx, None)  # clean, and its plants are caught
    units = [u for u in check.plants(ctx) if u.item.endswith(reference_range.CROWN_BASE)]
    assert [p.label.split(":")[0] for p in units[0].candidates] == ["lifted"]
    without(reference_range, "_share_findings",
            lambda i: i.endswith(reference_range.CROWN_BASE))(monkeypatch)
    with pytest.raises(BlindCheck, match="reference-range"):
        run_gate([check], ctx, None)


def test_a_piece_whose_limbs_take_no_placement_stops_the_gate():
    """A lone trunk has no second limb to cross or graze. The check judges its limb against
    itself, so the unit holds no candidate and the gate stops (a plant that dropped it would not)."""
    sk = make_skeleton(small_limbs()[:1], [])
    ctx = grown_ctx(sk)
    for make_check in (CrossingOrganic, NearMissOrganic):
        unit, = make_check().plants(ctx)
        assert unit.candidates == () and not unit.not_applicable
        with pytest.raises(BlindCheck, match=make_check().name):
            run_gate([make_check()], ctx, None)


def test_a_skeleton_match_with_no_vertex_standing_clear_still_gets_a_plant(monkeypatch):
    """The tolerance is so wide that every vertex has neighbours within the clearance: a plant
    that is None there would let a blind check pass. The unit moves a vertex far past the part."""
    ctx = grown_ctx(make_skeleton(small_limbs(), small_leaves(small_limbs())))
    pack = ctx.packs[PACK]
    piece = pack.pieces[PIECE]
    wide = dataclasses.replace(piece, listed=tuple(
        dataclasses.replace(row, min_thickness=row.min_thickness * 1000.0) for row in piece.listed))
    pack = dataclasses.replace(pack, pieces={**pack.pieces, PIECE: wide})
    ctx = dataclasses.replace(ctx, packs={PACK: pack})
    check = SkeletonMatch()
    unit, = check.plants(ctx)
    assert [p.label.split(":")[0] for p in unit.candidates] == ["vertex far"]
    run_gate([check], ctx, None)
    patched(skeleton_match, "_part_findings", nothing)(monkeypatch)
    with pytest.raises(BlindCheck, match="skeleton-match"):
        run_gate([check], ctx, None)


@pytest.mark.parametrize("make_check", [CrossingOrganic, NearMissOrganic])
@pytest.mark.parametrize("species", ["tiny-buds", "tiny-spiral"])
def test_a_seedling_without_a_twig_pair_is_planted(make_check, species):
    """A 0.5 m tree has no twig on an unrelated limb (every limb grows from the trunk): a pair-only
    plant would give it none. Another limb is moved against another instead, and caught."""
    ctx = bud_ctx(species, 1, 0.5)
    check = make_check()
    assert check.run(ctx) == []
    units = check.plants(ctx)
    assert len(units) == 1 and units[0].candidates
    assert len(tried_and_caught(check, ctx)) == 1


# --- the plants never call the verdict ---------------------------------------------------------

NEVER = {
    "crossing-organic": (CrossingOrganic, crossing, ("limb_pairs", "_findings")),
    "near-miss-organic": (NearMissOrganic, near_miss, ("limb_pairs", "_findings")),
    "limb-seat": (LimbSeat, limb_seat, ("_limb_findings", "_leaf_findings", "_allowance")),
    "floor-organic": (FloorOrganic, floor, ("_limb_findings", "_findings", "_low_past", "_leaf_low",
                                            "_leaf_bounds")),
    "floor-organic cards": (FloorOrganic, floor, ("_limb_findings", "_findings", "_card_findings",
                                                  "_card_lows", "_low_past", "_leaf_low")),
    "skeleton-match": (SkeletonMatch, skeleton_match, ("_part_findings", "matched", "_far")),
    "skeleton-match cards": (SkeletonMatch, skeleton_match, ("_part_findings", "matched", "_far")),
    "taper": (Taper, taper, ("_findings", "fit", "fits")),
    "leaf-angle": (LeafAngle, leaf_angle, ("_findings", "blade_angles", "spreads", "spread_of")),
    "reference-range": (ReferenceRange, reference_range, ("_share_findings", "_stale_findings",
                                                          "spread", "height")),
}


@pytest.mark.parametrize("name", sorted(NEVER))
def test_plants_never_call_the_verdict(name, monkeypatch):
    make_check, module, names = NEVER[name]
    _, pieces = TWO_PIECES[name]
    ctx = next(iter(pieces.values()))()
    check = make_check()
    refuses_the_core(monkeypatch, module, *names)
    refuses_the_core(monkeypatch, make_check, "run")
    assert check.plants(ctx)


def test_a_many_limb_piece_with_no_placement_names_the_pair_it_tried_first(monkeypatch):
    """The empty unit's item is the first pair offered, not limb 0 against itself, so the
    `BlindCheck` points at a real pair."""
    from model_kit.core.checks.branching import _pairs

    ctx = bud_ctx("tiny-buds", 1, 1.5)
    monkeypatch.setattr(_pairs, "placements", lambda *a, **k: iter(()))
    unit, = CrossingOrganic().plants(ctx)
    sk = ctx.skeletons[next(iter(ctx.skeletons))]
    x, y = next(iter(_pairs.pair_order(sk)))
    assert unit.candidates == ()
    assert unit.item.endswith(" x ".join(sorted((str(sk.limb_id[x]), str(sk.limb_id[y])))))
    assert (x, y) != (0, 0)
