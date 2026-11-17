import dataclasses

from model_kit.core import gate
from model_kit.core.checks.branching import BRANCHING_CHECKS
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.scene import PieceGeom, piece_key

from .branching_scene import PACK, PIECE, UPM, grown_ctx, tiny_ctx
from .tiny_trees import SPECIES

SKELETON_ONLY = ["limb-seat", "crossing-organic", "near-miss-organic", "taper", "leaf-angle",
                 "reference-range"]
CARD = "Tree_Cards"


def test_branching_checks_registered():
    names = [c.name for c in BRANCHING_CHECKS]
    assert names == ["limb-seat", "floor-organic", "crossing-organic", "near-miss-organic",
                     "skeleton-match", "reference-range", "taper", "leaf-angle"]
    for c in BRANCHING_CHECKS:
        assert gate.CHECKS[c.name] is c
        assert c.kinds == frozenset({"branching"})


def _with_card(ctx):
    """The tiny scene plus a card piece that copies its Tree and shares the Tree's skeleton."""
    pack = ctx.packs[PACK]
    card = dataclasses.replace(pack.pieces[PIECE], name=CARD, cards_of=PIECE, outlined=False,
                               outlined_why="cards draw no lines")
    key, ckey = piece_key(PACK, PIECE), piece_key(PACK, CARD)
    return dataclasses.replace(
        ctx, packs={PACK: dataclasses.replace(pack, pieces={**pack.pieces, CARD: card})},
        pieces={**ctx.pieces, ckey: PieceGeom(CARD, dict(ctx.pieces[key].parts))},
        skeletons={**ctx.skeletons, ckey: ctx.skeletons[key]})


def test_skeleton_only_checks_skip_cards():
    """Each skeleton-only check prints its coverage line for the card piece. On a faulted tree
    the card piece shares the fault, and still no item names the card piece."""
    clean = tiny_ctx()
    rules = load_rules(SPECIES[0], kit=KIT_RULES)
    ranged = dataclasses.replace(grown_ctx(grow_buds(
        rules, piece=PIECE, height_m=1.5, seed=1, lite=False, units_per_metre=UPM)),
        rules={SPECIES[0].name: rules})  # reference-range reads the species' rules from the scene
    key, ckey = piece_key(PACK, PIECE), piece_key(PACK, CARD)
    for check in (c() for c in BRANCHING_CHECKS if c.name in SKELETON_ONLY):
        want = f"{check.name}: {ckey} is judged on its source {key}"
        base = ranged if check.name == "reference-range" else clean
        lines = check.coverage(_with_card(base))
        assert want in lines, (check.name, lines)
        assert type(next(x for x in lines if x == want)) is str  # plain, not Unmeasured
        on_tree = 0
        for unit in check.plants(base):
            for plant in unit.candidates:
                items = [f.item for f in check.run(_with_card(plant.ctx), only=plant.scope | {ckey + i[len(key):] for i in plant.scope})]
                assert not any(i.startswith(ckey) for i in items), (check.name, items)
                on_tree += any(i.startswith(key) for i in items)
        assert on_tree, f"{check.name}: no plant raised a finding on the source"
