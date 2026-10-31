"""Scenes of the test furniture alone, for the tests of checks that read a piece's manifest and
decals. A helper module.

`furniture_ctx("Label", "Inlay")` is a scene with a pack `kit` of those furniture pieces from
`bodies/furniture.py`, built as `bodies.build.assemble` builds them: every manifest goes through
the loader and a decal stands off its host by the pieces' contact length.
"""

import dataclasses
import tempfile
from pathlib import Path

from model_kit.core.manifest import load_pack
from model_kit.core.scene import PieceGeom, SceneCtx, piece_key

from ..bodies.build import SEEN_FLOOR, body_style, write_json
from ..bodies.furniture import furniture

PACK = "kit"


def furniture_ctx(*names: str) -> SceneCtx:
    style = body_style()
    specs = furniture(*names)
    raw = {"pack": PACK, "units_per_metre": 1.0, "pieces": {n: s.raw for n, s in specs.items()}}
    with tempfile.TemporaryDirectory(prefix="model-kit-furniture-") as tmp:
        pack = load_pack(write_json(Path(tmp) / "kit.pack.json", raw), style)
    lift = SEEN_FLOOR  # as `bodies.build.assemble`: the contact length its pieces would declare
    pieces = {piece_key(PACK, n): PieceGeom(n, s.geometry(lift)) for n, s in specs.items()}
    return SceneCtx(style=style, packs={PACK: pack}, clips={}, pieces=pieces, samples={},
                    rigs={}, rig_of_clip={})


def with_finish(ctx: SceneCtx, piece: str, row: str, finish: str) -> SceneCtx:
    """The scene with one listed row of a piece given this `finish`."""
    old = ctx.packs[PACK].pieces[piece]
    listed = tuple(dataclasses.replace(r, finish=finish) if r.id == row else r for r in old.listed)
    return with_piece_record(ctx, piece, dataclasses.replace(old, listed=listed))


def with_part_shape(ctx: SceneCtx, piece: str, part: str, shape: str,
                    role: str | None = None) -> SceneCtx:
    """The scene with one part record given this shape (and no form), and this role if given."""
    old = ctx.packs[PACK].pieces[piece]
    parts = tuple(dataclasses.replace(p, shape=shape, form=None, role=role or p.role)
                  if p.name == part else p for p in old.parts)
    return with_piece_record(ctx, piece, dataclasses.replace(old, parts=parts))


def with_part_role(ctx: SceneCtx, piece: str, part: str, role: str) -> SceneCtx:
    """The scene with one part record given this role and nothing else changed."""
    old = ctx.packs[PACK].pieces[piece]
    parts = tuple(dataclasses.replace(p, role=role) if p.name == part else p for p in old.parts)
    return with_piece_record(ctx, piece, dataclasses.replace(old, parts=parts))


def with_piece_record(ctx: SceneCtx, piece: str, record) -> SceneCtx:
    """The scene with this piece record in place of the piece."""
    pack = ctx.packs[PACK]
    pack = dataclasses.replace(pack, pieces={**pack.pieces, piece: record})
    return dataclasses.replace(ctx, packs={**ctx.packs, PACK: pack})
