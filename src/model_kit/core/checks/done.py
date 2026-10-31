"""Check done and size: every piece says what done looks like and how big it is.

For each piece of a kind in `kinds` (assembly), `done` must be text with something in it, and
`size_class` must be one of `SIZE_CLASSES`. The pack loader already refuses both faults, so the
check guards a pack built in code. It reads the manifest only: a piece with no geometry in the
scene is measured too.

The items are `<pack>:<piece> done` and `<pack>:<piece> size_class`. The plants empty one
copy's `done` and blank another's `size_class`. Each is one unit of one candidate, built from the
manifest alone; a check blind to either fault stops the gate with `BlindCheck`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

from model_kit.core import kinds as recipe_kinds
from model_kit.core.checks.pieces import with_piece
from model_kit.core.manifest import SIZE_CLASSES, Piece
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx, piece_key

NAME = "done"


def _items(key: str) -> tuple[str, str]:
    return f"{key} done", f"{key} size_class"


def _done_empty(piece: Piece) -> bool:
    return not (piece.done or "").strip()


def _size_unknown(piece: Piece) -> bool:
    return piece.size_class not in SIZE_CLASSES


class Done:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        out = []
        for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, self.kinds):
            done_item, size_item = _items(piece_key(pack, name))
            emptied = with_piece(ctx, pack, name, dataclasses.replace(piece, done=""))
            blanked = with_piece(ctx, pack, name, dataclasses.replace(piece, size_class=""))
            out.append(PlantUnit(done_item, (Plant(f"emptied: {done_item}", done_item, emptied),)))
            out.append(PlantUnit(size_item, (Plant(f"blanked: {size_item}", size_item, blanked),)))
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, self.kinds):
            done_item, size_item = _items(piece_key(pack, name))
            if (only is None or done_item in only) and _done_empty(piece):
                found.append(Finding(NAME, done_item, "`done` is empty: say what finished "
                                                      "looks like"))
            if (only is None or size_item in only) and _size_unknown(piece):
                found.append(Finding(NAME, size_item, f"size_class {piece.size_class!r} is not "
                                                      f"one of {list(SIZE_CLASSES)}"))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return recipe_kinds.skipped_kinds(ctx, NAME, self.kinds)
