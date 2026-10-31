"""Recipe kinds: which pieces a check measures.

A piece is built by one of two recipes: `assembly` (parts put together by a build script) or
`branching` (a tree grown from cited rules). Each registered check declares `kinds`, a frozenset
of the kinds it measures. A check skips every piece of another kind and names each one in its
coverage, using `skipped_kinds`.

`piece_class` adds a third class, `rigged`, for a piece that has a rig; a sheet uses it to
pick a bar.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Literal

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import RECIPES, Piece
from model_kit.core.manifest.records import RecipeKind
from model_kit.core.scene import SceneCtx, piece_key

PieceClass = Literal[RecipeKind, "rigged"]


def piece_class(piece: Piece) -> PieceClass:
    """`rigged` when the piece has a rig, else its recipe kind."""
    return "rigged" if piece.rig is not None else piece.recipe.kind


def _check_kinds(kinds: Collection[str]) -> None:
    unknown = sorted(set(kinds) - set(RECIPES))
    if unknown:
        raise ManifestError(f"unknown recipe kind {unknown}; the kinds are {list(RECIPES)}")


def _all_pieces(ctx: SceneCtx) -> list[tuple[str, str, Piece]]:
    return [(pack, name, piece) for pack, p in sorted(ctx.packs.items())
            for name, piece in p.pieces.items()]


def pieces_of_kind(ctx: SceneCtx, kinds: Collection[str]) -> list[tuple[str, str, Piece]]:
    """`(pack, piece name, piece)` for each piece whose recipe kind is in `kinds`, in pack then
    piece order. A rigged piece counts by its recipe kind. An unknown kind raises."""
    _check_kinds(kinds)
    return [(pack, name, piece) for pack, name, piece in _all_pieces(ctx)
            if piece.recipe.kind in kinds]


def skipped_kinds(ctx: SceneCtx, check_name: str, kinds: Collection[str]) -> list[str]:
    """One coverage line per piece of a kind outside `kinds`:
    `<check>: not applicable (<kind> piece): <pack>:<piece>`."""
    _check_kinds(kinds)
    return [f"{check_name}: not applicable ({piece.recipe.kind} piece): {piece_key(pack, name)}"
            for pack, name, piece in _all_pieces(ctx) if piece.recipe.kind not in kinds]
