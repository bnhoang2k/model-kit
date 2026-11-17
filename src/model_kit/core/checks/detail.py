"""Check detail: a curve has the facets the eye can see and no more, at the distance it is seen from.

An edge under the style's crease angle shades smooth, so it stands for a curve. The two flat
faces that meet there differ from that curve by a gap, the sagitta (`core.edges.sagittas`). A
viewer sees the curve as blocky when that gap subtends more than the eye resolves
(`EYE_RESOLUTION_ARCMIN`) at the nearest distance the piece is seen from (`Piece.view.nearest_m`,
which the author declares from how the piece is used). Any edge can lie on the outline from some
direction, so every smooth edge is judged and no camera is assumed. The check measures each part
at rest, in real metres (`Pack.units_per_metre`); a count of triangles is never compared with a
limit.

An edge is smooth when its turn is under the crease angle and over the rounding a float32 mesh
carries (`edges.rounding_band_deg` at `edges.float32_step`, the kit's own model of how far a turn
can move): under that, the two faces are flat, as the diagonal of a flat quad is. The floor is on
the turn, so it does not grow with the piece or hide a finely cut curve. An edge beside a sliver
face, whose band the model caps, is not judged. Parts that skip their outline
(`Part.skips_outline`: a decal follows its host, and the roles in `NO_OUTLINE_ROLES` opt out) are
left out. An edge whose two faces
both look toward a side the piece's `view` lists as `hidden` is left out too; a visible face still
shows its facet.

- **Rule 1, blocky.** The worst smooth edge of a part has a sagitta `s` over `D * tan(arcmin)`
  (`D` the nearest distance). The item is `<pack>:<piece> <part> blocky at <D> m`.
- **Rule 2, finer than the eye.** Remove an edge and its two faces become one chord; the old
  corner then stands off that chord by the edge's merged gap (`edges.merged_gaps`). For a ring
  of equal faces that is `4 * cos^2(turn / 4)` times the sagitta, the gap of a ring with half its
  segments; for unequal faces (a string kinked over a nut) it is more. A part
  fails when the largest merged gap over its smooth edges still passes rule 1 and its largest
  smooth turn doubled stays under the crease angle (past it, halving would add a drawn line and
  change the look). The item is `<pack>:<piece> <part> finer than the eye at <D> m`. It is
  measured by gap, never by triangle size, so the small triangles at a sphere's pole or a fan's
  centre are not waste.

A part with no smooth edge (a box) is not applicable to both rules. A piece with no `view`, or a
style with no `crease_deg`, is `Unmeasured`.

Plants, one unit per part with a smooth edge and rule, each a single candidate built from the
part's own gaps and the piece's `view`, never from the check's verdict. Rule 1: the part scaled
about its centroid until its worst sagitta is `CALIBRATION_FAIL` times the limit. Scaling
multiplies every gap by the same factor and keeps every turn, so the plant never crosses the
crease angle. Rule 2 (only for a part whose largest smooth turn doubled stays under the crease
angle, the same condition coverage reports): the piece declared from so far that the part's
largest merged gap is `CALIBRATION_PASS` times the limit. The part is the same, the distance is
the fault, so the plant cannot fail to build; the item names the new distance. A part with no
smooth edge has no unit (coverage says so). A unit none of whose candidates the check catches
stops the gate with `BlindCheck`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.edges import (
    float32_step,
    interior_edges,
    merged_gaps,
    rounding_band_deg,
    sagittas,
)
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.sight import seen_distance, seen_floor
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    EYE_RESOLUTION_ARCMIN,
    WELD_OF_PIECE_SIZE,
)

NAME = "detail"
PRINCIPLE = ("A curve is built from flat facets, and the gap between a facet and the curve it "
             "stands for shows only when it subtends more than the eye resolves at the nearest "
             "distance the piece is seen from, so a curve is blocky when that gap is over what "
             "the eye resolves and finer than it needs to be when half the segments would look "
             "the same.")
SOURCE = (
    "Normal sight resolves about 1 minute of arc (minimum angle of "
    "resolution of acuity 1.0, EyeWiki, 'Visual Acuity Testing in Adults', "
    "https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, secondary; the International "
    "Council of Ophthalmology's 1984 Visual Acuity Measurement Standard, cited for the decimal "
    "scale, was not found in its own text). The gap is the sagitta of the circle through two "
    "chords, (w / 2) tan(turn / 4); halving a circle's segments multiplies it by "
    "4 cos^2(turn / 4), from 1 - cos x = sin x tan(x / 2).")

# Which side a face belongs to: the axis of its normal's largest component, and the sign.
# glTF axes in the piece frame (`VIEW_SIDES`): front +z, back -z, left +x, right -x, top +y,
# bottom -y.
_SIDE_AXIS = {"front": (2, 1.0), "back": (2, -1.0), "left": (0, 1.0), "right": (0, -1.0),
              "top": (1, 1.0), "bottom": (1, -1.0)}


def faces_on_sides(tris: np.ndarray, sides: Collection[str]) -> np.ndarray:
    """Which rows of `tris` (N, 3, 3) belong to one of `sides`: a face belongs to the side its
    outward normal points most toward (the meshes are wound outward). A face with no area
    belongs to none."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    axis = np.abs(normal).argmax(axis=1)
    sign = np.sign(normal[np.arange(len(normal)), axis])
    out = np.zeros(len(tris), dtype=bool)
    for side in sides:
        a, s = _SIDE_AXIS[side]
        out |= (axis == a) & (sign == s)
    return out


def _exempt(part: PartMesh) -> str | None:
    """What kind of part this is, when the check leaves it out."""
    return part.skipped_as if part.skips_outline else None


# --- one part ---------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Read:
    """The smooth edges of a part that can be seen: the number each has in
    `interior_edges`' order, its turn in degrees, its sagitta and its merged gap (the gap left
    if the edge were removed), both in the pack's units."""

    index: np.ndarray
    turn: np.ndarray
    sagitta: np.ndarray
    merged: np.ndarray

    @property
    def worst(self) -> int:
        """The row of the edge with the largest sagitta."""
        return int(self.sagitta.argmax())

    @property
    def worst_merged(self) -> int:
        """The row of the edge with the largest merged gap."""
        return int(self.merged.argmax())


def _read(tris: np.ndarray, size: float, crease: float, hidden: tuple[str, ...]) -> _Read:
    edges = interior_edges(tris, WELD_OF_PIECE_SIZE * size)
    if not len(edges):
        empty = np.zeros(0)
        return _Read(np.zeros(0, dtype=np.intp), empty, empty, empty)
    # smooth: over the rounding a float32 turn carries, under the crease angle, and not beside a
    # sliver face (its band is capped: the turn cannot be trusted)
    band, capped = rounding_band_deg(edges.points, edges, float32_step(edges.points), crease)
    keep = (edges.turn_deg < crease) & (edges.turn_deg > band) & ~capped
    if hidden:
        keep &= ~faces_on_sides(tris, hidden)[edges.faces].all(axis=1)
    index = np.flatnonzero(keep)
    return _Read(index, edges.turn_deg[index], sagittas(tris, edges)[index],
                 merged_gaps(tris, edges)[index])


def _doubled_stays_smooth(read: _Read, crease: float) -> bool:
    return 2.0 * float(read.turn.max()) < crease


def _blocky(read: _Read, limit: float) -> bool:
    return len(read.index) > 0 and float(read.sagitta.max()) > limit


def _wasted(read: _Read, limit: float, crease: float) -> bool:
    if not len(read.index) or _blocky(read, limit):
        return False
    return float(read.merged.max()) <= limit and _doubled_stays_smooth(read, crease)


def _metres(value: float, units_per_metre: float) -> str:
    return f"{value / units_per_metre * 1000.0:.3f} mm"


def _far_item(item: str, distance: float) -> str:
    return f"{item} finer than the eye at {distance:g} m"


class Detail:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly", "branching"})
    scope = "piece"
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._reads: dict[tuple, tuple[np.ndarray, _Read]] = {}

    # -- what to measure

    @staticmethod
    def _crease(ctx: SceneCtx) -> float | None:
        """The crease angle, or None when the style names none. The angle sets what shades
        smooth whether or not the style draws an outline, so it is not read from the outline."""
        return ctx.style.crease_deg

    def _read(self, view: PieceView, part: PartMesh, crease: float) -> _Read:
        hidden = view.piece.view.hidden if view.piece.view is not None else ()
        key = (id(part.tris), view.size, crease, hidden)
        held = self._reads.get(key)
        if held is None or held[0] is not part.tris:
            held = (part.tris, _read(part.tris, view.size, crease, hidden))
            self._reads[key] = held
        return held[1]

    def _parts(self, ctx: SceneCtx, only: Collection[str] | None
               ) -> Iterator[tuple[PieceView, PartMesh, str]]:
        for view in pieces.measured_pieces(ctx, self.kinds, only):
            if view.piece.view is None:
                continue
            for part in view.parts:
                item = item_of(view, part.name)
                if _exempt(part) is not None or not len(part.tris):
                    continue
                if only is not None and not any(i.startswith(f"{item} ") for i in only):
                    continue
                yield view, part, item

    # -- the check

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        crease = self._crease(ctx)
        if crease is None:
            return []
        out: list[Finding] = []
        for view, part, item in self._parts(ctx, only):
            for finding in self._part_findings(ctx, view, part, item, crease):
                if only is None or finding.item in only:
                    out.append(finding)
        return out

    def _part_findings(self, ctx: SceneCtx, view: PieceView, part: PartMesh, item: str,
                       crease: float) -> list[Finding]:
        read = self._read(view, part, crease)
        if not len(read.index):
            return []
        upm = ctx.packs[view.pack].units_per_metre
        limit = seen_floor(view.piece, upm)
        distance = view.piece.view.nearest_m
        i = read.worst
        s, turn = float(read.sagitta[i]), float(read.turn[i])
        seen = f"{_metres(limit, upm)} at {distance:g} m ({EYE_RESOLUTION_ARCMIN:g} arcmin)"
        if _blocky(read, limit):
            return [Finding(NAME, f"{item} blocky at {distance:g} m", (
                f"edge {int(read.index[i])}: a gap of {_metres(s, upm)} between its facets and "
                f"the curve they stand for, over a turn of {turn:.2f} deg, is over what the eye "
                f"resolves, {seen}: add segments"))]
        if _wasted(read, limit, crease):
            j = read.worst_merged
            return [Finding(NAME, f"{item} finer than the eye at {distance:g} m", (
                f"edge {int(read.index[j])}: taking the edge out would leave a gap of "
                f"{_metres(float(read.merged[j]), upm)} over a turn of {float(read.turn[j]):.2f} "
                f"deg, still under what the eye resolves, {seen}, and the largest turn doubled, "
                f"{2.0 * float(read.turn.max()):.2f} deg, stays under the crease angle "
                f"{crease:g}: halve the segments"))]
        return []

    # -- coverage

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part with triangles that the check could judge: of a piece with a `view`, not
        a decal or a part that draws no outline, in a style with a `crease_deg`."""
        if self._crease(ctx) is None:
            return set()
        return {item for _view, _part, item in self._parts(ctx, None)}

    def coverage(self, ctx: SceneCtx) -> list[str]:
        crease = self._crease(ctx)
        if crease is None:
            return [Unmeasured(f"{NAME}: skipped (the style has no crease_deg in dials.edges, "
                               "so what shades smooth is unknown)")]
        lines: list[str] = list(pieces.coverage(ctx, NAME, self.kinds))
        for view in pieces.measured_pieces(ctx, self.kinds):
            if view.piece.view is None:
                lines.append(Unmeasured(f"{NAME}: skipped (the piece declares no view, so there "
                                        f"is no distance to judge from): {view.key}"))
                continue
            kept = [p for p in view.parts if _exempt(p) is None and len(p.tris)]
            lines += [f"{NAME}: not applicable ({kind} part): {view.key} {p.name}"
                      for p in view.parts if (kind := _exempt(p)) is not None]
            reads = {p.name: self._read(view, p, crease) for p in kept}
            lines += [f"{NAME}: not applicable (no smooth edge in view): {view.key} {p.name}"
                      for p in kept if not len(reads[p.name].index)]
            smooth = [r for r in reads.values() if len(r.index)]
            fixed = [r for r in smooth if not _doubled_stays_smooth(r, crease)]
            if fixed:
                lines.append(f"{NAME}: rule 2 not applicable to {len(fixed)} of {len(smooth)} "
                             f"parts with a smooth edge (halving their segments would turn an "
                             f"edge past the crease angle {crease:g}): {view.key}")
        return list(dict.fromkeys(lines))

    # -- plants

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        crease = self._crease(ctx)
        if crease is None:
            return []
        units: list[PlantUnit] = []
        for view in pieces.measured_pieces(ctx, self.kinds):
            if view.piece.view is not None:
                units += self._plant_piece(ctx, view, crease)
        return units

    def _plant_piece(self, ctx: SceneCtx, view: PieceView, crease: float) -> list[PlantUnit]:
        upm = ctx.packs[view.pack].units_per_metre
        limit = seen_floor(view.piece, upm)
        distance = view.piece.view.nearest_m
        units: list[PlantUnit] = []
        for part in view.parts:
            if _exempt(part) is not None or not len(part.tris):
                continue
            read = self._read(view, part, crease)
            if not len(read.index):
                continue  # no smooth edge in view: coverage says so, and no rule judges it
            here = item_of(view, part.name)
            blocky = f"{here} blocky at {distance:g} m"
            scale = CALIBRATION_FAIL * limit / float(read.sagitta.max())
            units.append(PlantUnit(blocky, (self._scaled(ctx, view, part, blocky, scale),)))
            if _doubled_stays_smooth(read, crease):  # else halving adds a line: never waste
                far = self._seen_from_far(ctx, view, part, upm, read)
                units.append(PlantUnit(far.item, (far,)))
        return units

    def _seen_from_far(self, ctx: SceneCtx, view: PieceView, part: PartMesh, upm: float,
                       read: _Read) -> Plant:
        """The scene with the piece declared from so far that half the segments of `part` would
        still pass at `CALIBRATION_PASS` of the limit (its largest merged gap). The part is the
        same, the distance is the fault. The item names the new distance."""
        need = float(read.merged.max()) / CALIBRATION_PASS
        distance = seen_distance(need, upm)
        # A farthest view cannot lie nearer than the nearest, so it rides out with the plant.
        farthest = view.piece.view.farthest_m
        declared = dataclasses.replace(
            view.piece.view, nearest_m=distance,
            farthest_m=None if farthest is None else max(farthest, distance))
        piece = dataclasses.replace(view.piece, view=declared)
        item = _far_item(item_of(view, part.name), distance)
        return Plant(f"declared from {distance:g} m: {item}", item,
                     pieces.with_piece(ctx, view.pack, view.name, piece))

    def _scaled(self, ctx: SceneCtx, view: PieceView, part: PartMesh, item: str, factor: float
                ) -> Plant:
        """The scene with `part` scaled about its centroid by `factor`."""
        centre = pieces.centroid(part.tris)
        moved = centre + (part.tris - centre) * factor
        geometry = {p.name: p.tris for p in view.parts} | {part.name: moved}
        return Plant(f"part scaled by {factor:.3g}: {item}", item,
                     pieces.with_geometry(ctx, view, geometry))
