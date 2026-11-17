import re
from pathlib import Path

import pytest

from model_kit.core import gate
from model_kit.core.errors import ManifestError
from model_kit.core.kinds import RECIPES, piece_class, pieces_of_kind, skipped_kinds

from .test_scene import branching_ctx, shop_ctx

KINDS_DIR = Path(__file__).resolve().parents[2] / "src" / "model_kit" / "kinds"


def test_piece_class():
    pieces = shop_ctx().packs["shop"].pieces
    assert piece_class(pieces["Bench"]) == "assembly"
    assert piece_class(pieces["Doll"]) == "rigged"
    assert piece_class(branching_ctx().packs["shop"].pieces["Lamp"]) == "branching"


def test_pieces_of_kind_filters():
    ctx = branching_ctx()
    assert [(p, n) for p, n, _ in pieces_of_kind(ctx, ["branching"])] == [("shop", "Lamp")]
    assembly = pieces_of_kind(ctx, {"assembly"})
    assert [(p, n) for p, n, _ in assembly] == [("shop", "Bench"), ("shop", "Doll")]
    assert all(piece is ctx.packs[p].pieces[n] for p, n, piece in assembly)
    both = pieces_of_kind(ctx, RECIPES)
    assert [(p, n) for p, n, _ in both] == [("shop", "Bench"), ("shop", "Lamp"), ("shop", "Doll")]
    assert pieces_of_kind(ctx, []) == []


def test_pieces_of_kind_unknown_kind_raises():
    with pytest.raises(ManifestError, match="sculpture"):
        pieces_of_kind(shop_ctx(), ["sculpture"])


def test_skipped_kinds_lines():
    ctx = branching_ctx()
    assert skipped_kinds(ctx, "limb-seat", ["branching"]) == [
        "limb-seat: not applicable (assembly piece): shop:Bench",
        "limb-seat: not applicable (assembly piece): shop:Doll",
    ]
    assert skipped_kinds(ctx, "wall-gap", ["assembly"]) == [
        "wall-gap: not applicable (branching piece): shop:Lamp"]
    assert skipped_kinds(ctx, "any", RECIPES) == []


def test_every_check_declares_kinds():
    for name, check in gate.CHECKS.items():
        kinds = getattr(check, "kinds", None)
        assert isinstance(kinds, frozenset), name
        assert kinds <= set(RECIPES), name


def _headings(kind: str) -> list[str]:
    text = (KINDS_DIR / kind / "rules.md").read_text()
    found = re.findall(r"^## (\S+)\s*$", text, flags=re.MULTILINE)
    return found


@pytest.mark.parametrize("kind", ["assembly", "branching"])
def test_rules_md_names_every_kind_check(kind):
    registered = sorted(n for n, c in gate.CHECKS.items() if kind in c.kinds)
    assert sorted(_headings(kind)) == registered
    assert len(set(_headings(kind))) == len(_headings(kind))


def test_assembly_rules_md_intro_names_every_kindless_check():
    """The intro lists the checks that declare no kind and so have no heading in any rules.md."""
    text = (KINDS_DIR / "assembly" / "rules.md").read_text()
    intro = text.split("\n## ", 1)[0]
    named = {n for n in re.findall(r"`([^`]+)`", intro) if n in gate.CHECKS}
    kindless = {n for n, c in gate.CHECKS.items() if not c.kinds}
    assert sorted(named) == sorted(kindless)


def test_branching_rules_md_names_its_checks():
    """even-cover and leonardo are taste-only: their headings carry the word, so not a check."""
    assert _headings("branching") == [
        "limb-seat", "floor-organic", "crossing-organic", "near-miss-organic", "skeleton-match",
        "near-crease", "detail", "reference-range", "taper", "leaf-angle"]
