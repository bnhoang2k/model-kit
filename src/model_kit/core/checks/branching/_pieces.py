"""What the branching checks share: the branching pieces of a scene with their skeletons,
item names, the coverage lines, and the piece's tube thickness base.

Items follow one form: `<pack>:<piece> <limb id>`, `<pack>:<piece> leaf.<n>`, a pair
`<pack>:<piece> <a> x <b>` (ids sorted), or `<pack>:<piece> <part>`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

from model_kit.core.grow.mesh import leaf_part_name
from model_kit.core.grow.skeleton import Skeleton
from model_kit.core.kinds import pieces_of_kind, skipped_kinds
from model_kit.core.manifest import Piece
from model_kit.core.metagate import Unmeasured
from model_kit.core.scene import SceneCtx, part_key, piece_key
from model_kit.core.units import thickness_base

KINDS = frozenset({"branching"})


@dataclass(frozen=True)
class Grown:
    """A branching piece and its skeleton."""

    pack: str
    name: str
    piece: Piece
    skeleton: Skeleton

    @property
    def key(self) -> str:
        return piece_key(self.pack, self.name)

    @property
    def card(self) -> bool:
        """Whether this is a card copy of another piece (it shares that piece's skeleton)."""
        return bool(self.piece.cards_of)

    def item(self, rest: str) -> str:
        return part_key(self.pack, self.name, rest)

    def limb(self, i: int) -> str:
        return self.item(str(self.skeleton.limb_id[i]))

    def leaf(self, f: int) -> str:
        return self.item(f"leaf.{f}")

    def pair(self, i: int, j: int) -> str:
        a, b = sorted((str(self.skeleton.limb_id[i]), str(self.skeleton.limb_id[j])))
        return self.item(f"{a} x {b}")

    def limb_parts(self) -> set[str]:
        """The part keys of the levels that hold a limb: the tube parts the skeleton grows."""
        sk = self.skeleton
        return {self.item(name) for i, name in enumerate(sk.level_names) if (sk.level == i).any()}

    def leaf_parts(self) -> set[str]:
        """The part keys of the leaf shades that hold a leaf."""
        sk = self.skeleton
        return {self.item(leaf_part_name(shade)) for i, shade in enumerate(sk.shades)
                if (sk.shade == i).any()}

    def wants(self, only: Collection[str] | None, item: str) -> bool:
        return only is None or item in only

    def named(self, only: Collection[str] | None) -> set[str] | None:
        """The rest of each `only` item that names this piece (None: every item)."""
        if only is None:
            return None
        prefix = f"{self.key} "
        return {i[len(prefix):] for i in only if i.startswith(prefix)}

    def with_skeleton(self, ctx: SceneCtx, skeleton: Skeleton) -> SceneCtx:
        """A copy of the scene with this piece's skeleton replaced."""
        return dataclasses.replace(ctx, skeletons={**ctx.skeletons, self.key: skeleton})


def grown(ctx: SceneCtx, only: Collection[str] | None = None,
          cards: bool = False) -> Iterator[Grown]:
    """Each branching piece that has a skeleton, in pack then piece order. With `only`, just
    the pieces some item names. A card piece is skipped unless `cards` is true: the checks that
    read only the skeleton judge its source."""
    for pack, name, piece in pieces_of_kind(ctx, KINDS):
        if piece.cards_of and not cards:
            continue
        sk = ctx.skeletons.get(piece_key(pack, name))
        if sk is None:
            continue
        g = Grown(pack, name, piece, sk)
        if only is None or any(i.startswith(f"{g.key} ") for i in only):
            yield g


def coverage(ctx: SceneCtx, check: str, judges_cards: bool = False) -> list[str]:
    """Pieces of other kinds, then each branching piece with no skeleton (`Unmeasured`). A card
    piece gets a plain line naming the source that judged it, unless the check judges card pieces
    itself (`judges_cards`): it then reads like any other piece."""
    lines: list[str] = skipped_kinds(ctx, check, KINDS)
    for pack, name, piece in pieces_of_kind(ctx, KINDS):
        key = piece_key(pack, name)
        if piece.cards_of and not judges_cards:
            lines.append(f"{check}: {key} is judged on its source "
                         f"{piece_key(pack, piece.cards_of)}")
        elif key not in ctx.skeletons:
            lines.append(Unmeasured(f"{check}: skipped (no skeleton): {key}"))
    return lines


def tube_base(ctx: SceneCtx, piece: Piece) -> float:
    """The piece's tube thickness base: its listed rows that have a tube part (`Part.tube`). A
    leaf card has no thickness, so leaf rows do not count."""
    tube_rows = {part.listed for part in piece.parts if part.tube}
    return thickness_base(ctx.style, [row.min_thickness for row in piece.listed
                                      if row.id in tube_rows])


def pairs_named(g: Grown, only: Collection[str] | None) -> set[tuple[int, int]] | None:
    """The limb index pairs that `only`'s pair items name on this piece (None: every pair). An
    item that names no pair of this piece's limbs is skipped."""
    named = g.named(only)
    if named is None:
        return None
    index = {str(name): i for i, name in enumerate(g.skeleton.limb_id)}
    out = set()
    for rest in named:
        a, sep, b = rest.partition(" x ")
        if sep and a in index and b in index:
            out.add((index[a], index[b]))
    return out
