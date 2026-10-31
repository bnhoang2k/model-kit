"""Check markings: a print, paint film, inlay or screen image is a decal on its host.

A marking is a change of colour on a surface, not an object. A listed row says so in its
`finish`: `solid` (the default), `print`, `paint`, `inlay` or `display`. Two rules, over each
piece of a kind in `kinds` (assembly).

1. Every part of a row whose `finish` is not `solid` has shape `decal`. The item is
   `<pack>:<piece> <part> is a solid <finish>`.
2. Every printed part (`Part.printed`: role `eyes` or `decal`) belongs to a row whose `finish` is
   not `solid`. A part of shape `decal` is always printed, because the loader refuses a decal
   with a solid role; a printed role on another shape is found here. The item is
   `<pack>:<piece> <part> decal of a solid row`.

Rule 3, the safety net for rows that say `solid` by default, is the markings-flush check, which
reads the piece's `view`: a solid part counts as flush with its `on` host, and may be a marking,
only when its height is under what the eye resolves (1 arcmin) at the piece's nearest viewing distance
(`view.nearest_m`). It is judged by the eye's resolution, not by a share of the thickness base.

Both rules read the manifest only. The plants are one unit per rule per piece, made from the
manifest and the part boxes alone (never from a rule's verdict). Rule 1: a decal part of a printed
row is rebuilt as a solid box shape (a piece with no decal makes its smallest solid part's row a
print instead). Rule 2: the same decal's row is made `solid` (or the smallest solid part is given
the printed role `decal` in a solid row, and keeps its shape, because a record with shape `decal`
and no `on` cannot be built). The gate skips a unit whose part already fails. A piece with no solid
part to plant on says so in a `not applicable` line.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PieceView, item_of
from model_kit.core.manifest import FINISHES, Part
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx

NAME = "markings"
SOLID = FINISHES[0]
DECAL = "decal"

PRINCIPLE = ("A print, a paint film, an inlay or a screen image changes the colour of a surface "
             "and is not an object, because ink and paint films are microns thick and an inlay "
             "is cut flush with its host, so each is built as a decal on the host's surface and "
             "never as a solid part that stands off it.")
SOURCE = (
    "Film thickness: a dried pad-print ink deposit is 4 microns (Inkcups ICN-B150 pad printer "
    "manual, ch. 1, https://tagless.inkcups.com/wp-content/uploads/2017/08/"
    "ICN-B150-Manual-Compressed.pdf); printing inks deposit 1 to 20 microns by process "
    "(Podhajny, PFFC, 2002, https://pffc-online.com/controls/308-paper-measure-ink-film); a "
    "coat of paint on wood is 25 to 30 microns (Nippon Paint 1-Pack PU Timbercoat data sheet, "
    "https://www.nipponpaint.com.my/wp-content/uploads/2024/11/1-pack-pu-timbercoat.pdf) and a "
    "full pigmented system 50 to 70 microns (Adler Aquawood Fineline D data sheet, "
    "https://www.adler-italia.it/Canto/tmb/aquawood-fineline-d_tmb_5019_en.pdf). Every value is "
    "under 0.1 mm. Screen printing on plastic has no verified number, only printing.org's "
    "'more than a few microns' (https://www.printing.org/content/2014/07/11/"
    "screen-printing-fine-features-and-thin-ink-film-for-printed-electronics-applications).")


def _row_finish(view: PieceView) -> dict[str, str]:
    return {row.id: row.finish for row in view.piece.listed}


class _Judge:
    """One piece's records, so each rule reads them once."""

    def __init__(self, view: PieceView) -> None:
        self.view = view
        self.finish = _row_finish(view)
        self.mesh = {p.name: p for p in view.parts}

    def of(self, record: Part) -> str:
        return self.finish.get(record.listed, SOLID)


def _rule1(view: PieceView, judge: _Judge, only: Collection[str] | None):
    for record in view.piece.parts:
        finish = judge.of(record)
        item = f"{item_of(view, record.name)} is a solid {finish}"
        if finish != SOLID and record.shape != DECAL and (only is None or item in only):
            yield Finding(NAME, item, f"row {record.listed!r} is a {finish}, so the part is "
                                      f"shape decal, not {record.shape!r}: build it as a decal "
                                      f"on its host")


def _rule2(view: PieceView, judge: _Judge, only: Collection[str] | None):
    for record in view.piece.parts:
        item = f"{item_of(view, record.name)} decal of a solid row"
        if record.printed and judge.of(record) == SOLID and (only is None or item in only):
            yield Finding(NAME, item, f"row {record.listed!r} says finish solid, but the part "
                                      f"is printed (role {record.role!r}, shape "
                                      f"{record.shape!r}): give the row the finish it is made with "
                                      f"({', '.join(FINISHES[1:])})")


def _findings(view: PieceView, only: Collection[str] | None):
    judge = _Judge(view)
    yield from _rule1(view, judge, only)
    yield from _rule2(view, judge, only)


# --- plants ---------------------------------------------------------------------------------


def _with_record(ctx: SceneCtx, view: PieceView, old: Part, new: Part,
                 listed=None) -> SceneCtx:
    parts = tuple(new if p.name == old.name else p for p in view.piece.parts)
    piece = dataclasses.replace(view.piece, parts=parts,
                                **({} if listed is None else {"listed": listed}))
    return pieces.with_piece(ctx, view.pack, view.name, piece)


def _with_finish(view: PieceView, row_id: str, finish: str):
    return tuple(dataclasses.replace(row, finish=finish) if row.id == row_id else row
                 for row in view.piece.listed)


def _plant_units(ctx: SceneCtx, view: PieceView) -> list[PlantUnit]:
    """The units for rules 1 and 2, made in the manifest. On a decal part of a marked row, the part
    is rebuilt as a solid box (rule 1) and its row is made solid (rule 2). A piece with no such
    decal uses its smallest solid part: its row is made a print (rule 1), and the part is given the
    printed role `decal` in a solid row (rule 2)."""
    judge = _Judge(view)
    has_tris = {p.name for p in view.parts if len(p.tris)}
    parts = [r for r in view.piece.parts if r.name in has_tris]
    for record in parts:
        finish = judge.of(record)
        if record.shape == DECAL and finish != SOLID:
            return [
                _unit(f"built solid: {record.name}", view, f"is a solid {finish}", record.name,
                      _with_record(ctx, view, record, dataclasses.replace(
                          record, shape="box", form=None))),
                _unit(f"decal of a solid row: {record.name}", view, "decal of a solid row",
                      record.name, _with_record(ctx, view, record, record,
                                                _with_finish(view, record.listed, SOLID)))]
    solids = [r for r in parts if r.shape != DECAL and judge.mesh[r.name].solid]
    if not solids:
        return [PlantUnit(view.key, (), not_applicable="no solid part to plant a marking fault on")]
    record = min(solids, key=lambda r: float(np.linalg.norm(judge.mesh[r.name].hi
                                                            - judge.mesh[r.name].lo)))
    return [
        _unit(f"made a print: {record.name}", view, "is a solid print", record.name,
              _with_record(ctx, view, record, record, _with_finish(view, record.listed, "print"))),
        _unit(f"made a decal: {record.name}", view, "decal of a solid row", record.name,
              _with_record(ctx, view, record, dataclasses.replace(record, role="decal"),
                           _with_finish(view, record.listed, SOLID)))]


def _unit(label: str, view: PieceView, tail: str, part: str, ctx: SceneCtx) -> PlantUnit:
    item = f"{item_of(view, part)} {tail}"
    return PlantUnit(item, (Plant(f"{label}: {item}", item, ctx),))


class Markings:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [unit for view in pieces.measured_pieces(ctx, self.kinds) if view.parts
                for unit in _plant_units(ctx, view)]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part with triangles of a measured piece: rules 1 and 2 read each one's record."""
        return {item_of(view, p.name) for view in pieces.measured_pieces(ctx, self.kinds)
                for p in view.parts if len(p.tris)}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return sorted((f for view in pieces.measured_pieces(ctx, self.kinds, only)
                       for f in _findings(view, only)), key=lambda f: f.item)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return pieces.coverage(ctx, NAME, self.kinds)
