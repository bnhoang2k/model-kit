"""A scoped run: the piece-by-piece checks measure only the pieces a run names.

A `Scope` holds piece keys (`<pack>:<piece>`). `Scope.scene` is the scene without the pieces the
scope does not name; a check marked `scope = "piece"` runs on it. Layout, clip and export checks
read the whole scene, so they run on the original. A scoped run is never a pass: `Scope.line` is
an `Unmeasured` coverage line, so the run is `partial`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from model_kit.core.errors import ManifestError
from model_kit.core.metagate import Unmeasured
from model_kit.core.scene import SceneCtx, piece_key


def _listed(ctx: SceneCtx) -> list[str]:
    """Every piece key some pack lists, in pack then piece order."""
    return [piece_key(pack.name, name) for pack in ctx.packs.values() for name in pack.pieces]


@dataclass(frozen=True)
class Scope:
    pieces: frozenset[str]
    _built: tuple[SceneCtx, SceneCtx] | None = field(
        default=None, init=False, compare=False, hash=False, repr=False)

    def dropped(self, ctx: SceneCtx) -> frozenset[str]:
        """The keys a pack lists that this scope does not name. A piece no pack lists is never
        dropped."""
        return frozenset(_listed(ctx)) - self.pieces

    def scene(self, ctx: SceneCtx) -> SceneCtx:
        """The scene without the dropped pieces: their geometry, rigs, rest bodies, skeletons and
        pack records. Clips, samples, layouts, nodes and sources stay as they are. The last
        scene is kept: the same `ctx` gives the same scene, so a check that caches on the scene
        keeps its cache across callers."""
        if self._built is not None and self._built[0] is ctx:
            return self._built[1]
        gone = self.dropped(ctx)

        def kept(table: dict) -> dict:
            return {key: value for key, value in table.items() if key not in gone}

        packs = {}
        for name, pack in ctx.packs.items():
            packs[name] = dataclasses.replace(
                pack, pieces={p: rec for p, rec in pack.pieces.items()
                              if piece_key(pack.name, p) not in gone})
        scene = dataclasses.replace(ctx, packs=packs, pieces=kept(ctx.pieces), rigs=kept(ctx.rigs),
                                    rests=kept(ctx.rests), skeletons=kept(ctx.skeletons))
        object.__setattr__(self, "_built", (ctx, scene))
        return scene

    def line(self) -> Unmeasured:
        names = ", ".join(sorted(self.pieces))
        return Unmeasured(f"gate: scoped run (pieces: {names}): the part checks measured only "
                          "these pieces; a gate on the whole scene is the pass")


def parse_pieces(text: str, ctx: SceneCtx) -> Scope:
    """The scope of a comma-separated list of piece keys. An empty list, a key named twice or a
    key no pack lists raises `ManifestError`."""
    keys = [key.strip() for key in text.split(",") if key.strip()]
    if not keys:
        raise ManifestError("--pieces names no piece")
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise ManifestError(f"a piece is named twice: {', '.join(repeated)}")
    known = _listed(ctx)
    unknown = [key for key in keys if key not in known]
    if unknown:
        raise ManifestError(f"no pack lists {', '.join(map(repr, unknown))}; the pieces are "
                            f"{', '.join(known)}")
    return Scope(frozenset(keys))
