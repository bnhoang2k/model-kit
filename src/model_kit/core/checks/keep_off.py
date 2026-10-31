"""Check keep-off: a mount stays clear of the features its piece says to keep off.

A piece lists `keep_off` points, each with a radius: a truss coupler, a brace node, any spot
where a mount would land on something it must not sit on. For each piece of a kind in `kinds`
(assembly), every mount is measured against every keep-off point: the distance from the mount's
point to the keep-off point must be at least the keep-off's radius. The item is
`<pack>:<piece> <mount id> <keep-off id>`.

Both points are in the piece's own frame, so the check reads the manifest only: a piece with no
geometry in the scene is measured too. The radius is data, not a tolerance, so there is no
factor here. A piece with mounts and no keep-off points (or the reverse) has nothing to measure
and says so in a plain line.

The plant moves a mount onto a keep-off point: one unit per pair, built from the manifest alone.
The gate skips a pair that already fails in the real run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator

import numpy as np

from model_kit.core import kinds as recipe_kinds
from model_kit.core.checks.pieces import with_piece
from model_kit.core.manifest import KeepOff, Mount, Piece
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx, piece_key

NAME = "keep-off"


def _item(key: str, mount: Mount, keep: KeepOff) -> str:
    return f"{key} {mount.id} {keep.id}"


def _clearance(mount: Mount, keep: KeepOff) -> float:
    return float(np.linalg.norm(np.asarray(mount.point, dtype=float)
                                - np.asarray(keep.point, dtype=float)))


def _pairs(ctx: SceneCtx, kinds: Collection[str]
           ) -> Iterator[tuple[str, str, str, Piece, Mount, KeepOff]]:
    for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, kinds):
        for mount in piece.mounts:
            for keep in piece.keep_off:
                yield pack, name, piece_key(pack, name), piece, mount, keep


class KeepOffCheck:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        out = []
        for pack, name, key, piece, mount, keep in _pairs(ctx, self.kinds):
            mounts = tuple(dataclasses.replace(m, point=keep.point) if m.id == mount.id else m
                           for m in piece.mounts)
            item = _item(key, mount, keep)
            planted = with_piece(ctx, pack, name, dataclasses.replace(piece, mounts=mounts))
            out.append(PlantUnit(item, (Plant(f"on the point: {item}", item, planted),)))
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for _, _, key, _, mount, keep in _pairs(ctx, self.kinds):
            item = _item(key, mount, keep)
            gap = _clearance(mount, keep)
            if (only is None or item in only) and gap < keep.radius:
                found.append(Finding(NAME, item, f"mount {mount.id} is {gap:.4f} from keep-off "
                                                 f"point {keep.id} (it must clear {keep.radius:g})"))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = list(recipe_kinds.skipped_kinds(ctx, NAME, self.kinds))
        for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, self.kinds):
            key = piece_key(pack, name)
            if not piece.mounts:
                lines.append(f"{NAME}: no mounts to keep clear: {key}" if piece.keep_off
                             else f"{NAME}: no mounts and no keep-off points: {key}")
            elif not piece.keep_off:
                lines.append(f"{NAME}: no keep-off points to keep clear of: {key}")
        return lines
