"""Branching check: skeleton-match. The mesh that ships is the mesh the skeleton makes.

The other branching checks read the skeleton, so this check proves they judge what ships. For
each part of a branching piece (item `<pack>:<piece> <part>`), against `grown_parts(skeleton,
sight)`, with `tol` = `SKELETON_MATCH_OF_THICKNESS` x the piece's tube thickness base:
- a shipped part the skeleton does not grow;
- a triangle count that differs (a part the skeleton grows and the scene lacks has 0);
- a shipped triangle that matches no triangle of the skeleton's part, or the reverse. Two
  triangles match when each corner of one lies within `tol` of a corner of the other, both
  ways, and their normals point the same way (a positive dot product), so re-wired, flipped or
  zero-area triangles over the right vertices fail;
- a shipped vertex more than `tol` from every vertex of the skeleton's part (in the mesh, not
  the skeleton's), or the reverse (missing from the shipped mesh).
A branching piece with a skeleton but no geometry in the scene is `Unmeasured`.

A card piece (`Piece.cards_of`) is compared with `card_parts(skeleton, sight, dials.cards, ...)`
in place of `grown_parts`: its wood levels and its `Cards_<shade>` part, whose triangles are the
leaf cards. It needs a view, a crease angle and the style's `dials.cards` and `dials.leaves`;
without one it is `Unmeasured`.

A buds-v2 skeleton takes its tube sides from the piece's view, so it is re-meshed with the same
sight the build meshed it with (`sight.sight_for`: the seen floor of the piece's `view` and the
style's crease angle). Without a view, or a crease angle, there is nothing to re-mesh with, and
the piece is `Unmeasured`; the older meshes do not read the sight and are compared as before.

Nearest points are found on a grid of cells one tolerance wide, so each point is compared only
with the points of its own and the 26 cells around it.

Plant, one unit per grown piece with geometry, built from the two meshes' vertices alone (never
`_part_findings`, `matched` or `_far`): on the first part both hold, one shipped vertex whose
nearest other vertex of the skeleton's part lies over `PLANT_CLEAR_OF_TOL` x the tolerance away,
moved `CALIBRATION_FAIL` x the tolerance along +Y (every copy of it in that part). If no vertex
of that part is so far from the skeleton's, a second candidate moves the part's first vertex
along +Y by the part's bounding box diagonal, which no skeleton vertex lies within. A card piece
is planted the same way on the first part both its card copy and the scene hold, and again on its
first `Cards_<shade>` part (a second unit), so the leaf cards are planted too.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Collection

import numpy as np

from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown, tube_base
from model_kit.core.errors import ManifestError
from model_kit.core.grow.buds import MESH_VERSION
from model_kit.core.grow.cards import CARDS_PREFIX, card_dials, card_parts
from model_kit.core.grow.mesh import grown_parts, leaf_lean, needs_sight
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import PieceGeom, SceneCtx
from model_kit.core.sight import Sight, sight_for
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    PLANT_CLEAR_OF_TOL,
    SKELETON_MATCH_OF_THICKNESS,
)

NAME = "skeleton-match"
_OFFSETS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def close_pairs(a: np.ndarray, b: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """(ia, ib): every pair of a point of a and a point of b within `radius`."""
    if not len(a) or not len(b):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    lo = np.minimum(a.min(axis=0), b.min(axis=0))
    cell_a = np.floor((a - lo) / radius).astype(np.int64) + 1
    cell_b = np.floor((b - lo) / radius).astype(np.int64) + 1
    dims = np.maximum(cell_a.max(axis=0), cell_b.max(axis=0)) + 2

    def key(cells):
        return (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]

    order = np.argsort(key(cell_b), kind="stable")
    keys = key(cell_b)[order]
    found_a, found_b = [], []
    for off in _OFFSETS:
        k = key(cell_a + off)
        start = np.searchsorted(keys, k, side="left")
        n = np.searchsorted(keys, k, side="right") - start
        rows = np.repeat(np.arange(len(a)), n)
        idx = order[np.repeat(start, n) + np.arange(len(rows)) - np.repeat(np.cumsum(n) - n, n)]
        close = ((a[rows] - b[idx]) ** 2).sum(axis=1) <= radius * radius
        found_a.append(rows[close])
        found_b.append(idx[close])
    return np.concatenate(found_a), np.concatenate(found_b)


def neighbours(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """(len(a),) how many points of b lie within `radius` of each point of a."""
    ia, _ = close_pairs(a, b, radius)
    return np.bincount(ia, minlength=len(a))


def _normals(tris: np.ndarray) -> np.ndarray:
    return np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])


def matched(a: np.ndarray, b: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Which triangles of a match some triangle of b, and the reverse: each corner of one within
    `tol` of a corner of the other, both ways, and normals with a positive dot product."""
    ok_a, ok_b = np.zeros(len(a), dtype=bool), np.zeros(len(b), dtype=bool)
    ia, ib = close_pairs(a.mean(axis=1), b.mean(axis=1), tol)
    if len(ia):
        d = np.sqrt(((a[ia][:, :, None, :] - b[ib][:, None, :, :]) ** 2).sum(axis=-1))
        corners = (d.min(axis=2) <= tol).all(axis=1) & (d.min(axis=1) <= tol).all(axis=1)
        facing = (_normals(a[ia]) * _normals(b[ib])).sum(axis=1) > 0.0
        good = corners & facing
        ok_a[ia[good]] = True
        ok_b[ib[good]] = True
    return ok_a, ok_b


def _rows(points: np.ndarray) -> np.ndarray:
    """Each (x, y, z) row as one opaque value, for exact set tests."""
    points = np.ascontiguousarray(points, dtype=float)
    return points.view(np.dtype((np.void, points.dtype.itemsize * 3))).ravel()


def _vertices(tris: np.ndarray) -> np.ndarray:
    """The distinct corners of a part's triangles."""
    corners = np.ascontiguousarray(np.asarray(tris, dtype=float).reshape(-1, 3))
    _, first = np.unique(_rows(corners), return_index=True)
    return corners[np.sort(first)]


def _far(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """The points of a with no point of b within tol. A point b holds exactly is near."""
    rest = a[~np.isin(_rows(a), _rows(b))]
    return rest[neighbours(rest, b, tol) == 0]


def _part_findings(g: Grown, part: str, shipped, grown_tris, tol: float) -> list[Finding]:
    item = g.item(part)
    if grown_tris is None:
        return [Finding(NAME, item, f"a part the skeleton does not grow ({len(shipped)} "
                                    "triangles)")]
    if shipped is None or len(shipped) != len(grown_tris):
        got = 0 if shipped is None else len(shipped)
        found = [Finding(NAME, item, f"{got} triangles; the skeleton grows {len(grown_tris)}")]
        if shipped is None:
            return found
    else:
        found = []
    shipped = np.asarray(shipped, dtype=float)
    grown_tris = np.asarray(grown_tris, dtype=float)
    if shipped.shape == grown_tris.shape and np.array_equal(shipped, grown_tris):
        return found  # the same triangles, bit for bit
    ok_mine, ok_theirs = matched(shipped, grown_tris, tol)
    if not ok_mine.all():
        k = int(np.argmin(ok_mine))
        found.append(Finding(NAME, item, f"{int((~ok_mine).sum())} shipped triangles match no "
                                         f"triangle of the skeleton's mesh within {tol:.5f} "
                                         "(moved, re-wired, flipped or zero-area), e.g. at "
                                         f"{np.round(shipped[k].mean(axis=0), 4).tolist()}"))
    if not ok_theirs.all():
        k = int(np.argmin(ok_theirs))
        found.append(Finding(NAME, item, f"{int((~ok_theirs).sum())} triangles of the skeleton's "
                                         "mesh have no match in the shipped mesh, e.g. at "
                                         f"{np.round(grown_tris[k].mean(axis=0), 4).tolist()}"))
    mine, theirs = _vertices(shipped), _vertices(grown_tris)
    extra = _far(mine, theirs, tol)
    if len(extra):
        found.append(Finding(NAME, item, f"{len(extra)} vertices not in the skeleton's mesh "
                                         f"(none of its vertices within {tol:.5f}), e.g. at "
                                         f"{np.round(extra[0], 4).tolist()}"))
    lost = _far(theirs, mine, tol)
    if len(lost):
        found.append(Finding(NAME, item, f"{len(lost)} vertices of the skeleton's mesh missing "
                                         f"from the shipped mesh (none within {tol:.5f}), e.g. "
                                         f"at {np.round(lost[0], 4).tolist()}"))
    return found


def _sight(ctx: SceneCtx, g: Grown) -> Sight | None:
    return sight_for(g.piece, ctx.style, ctx.packs[g.pack].units_per_metre)


def _card_inputs(ctx: SceneCtx):
    """The style's card dials and leaf lean, or None when it lacks either."""
    try:
        return card_dials(ctx.style), leaf_lean(ctx.style)
    except ManifestError:
        return None


def _meshable(ctx: SceneCtx, g: Grown) -> bool:
    """Whether the skeleton can be re-meshed: it needs no sight, or the piece has one. A card
    copy always needs a sight and the style's card dials and leaf lean."""
    if g.card:
        return _sight(ctx, g) is not None and _card_inputs(ctx) is not None
    return not needs_sight(g.skeleton) or _sight(ctx, g) is not None


def _expected(ctx: SceneCtx, g: Grown) -> dict[str, np.ndarray]:
    """The parts the skeleton makes for this piece: the card copy's for a card piece."""
    if not g.card:
        return grown_parts(g.skeleton, _sight(ctx, g))
    dials, lean = _card_inputs(ctx)
    return card_parts(g.skeleton, _sight(ctx, g), dials, ctx.packs[g.pack].units_per_metre,
                      lean)[0]


def _findings(ctx: SceneCtx, g: Grown, geom: PieceGeom, only) -> list[Finding]:
    tol = SKELETON_MATCH_OF_THICKNESS * tube_base(ctx, g.piece)
    if not _meshable(ctx, g):
        return []
    expected = _expected(ctx, g)
    found = []
    for part in sorted(set(expected) | set(geom.parts)):
        if g.wants(only, g.item(part)):
            found += _part_findings(g, part, geom.parts.get(part), expected.get(part), tol)
    return found


def _moved(ctx: SceneCtx, g: Grown, geom: PieceGeom, part: str, v: np.ndarray, by: float
           ) -> SceneCtx:
    """The scene with every copy of vertex v in the shipped `part` moved `by` along +Y."""
    tris = np.array(geom.parts[part], dtype=float)
    tris[np.all(tris == v, axis=-1)] += np.array([0.0, by, 0.0])
    moved = PieceGeom(geom.name, {**geom.parts, part: tris})
    return dataclasses.replace(ctx, pieces={**ctx.pieces, g.key: moved})


def _part_unit(ctx: SceneCtx, g: Grown, geom: PieceGeom, expected, tol: float, part: str
               ) -> PlantUnit | None:
    """The unit that moves one vertex of `part`, or None when the shipped part has no vertex."""
    mine, theirs = _vertices(geom.parts[part]), _vertices(expected[part])
    if not len(mine):
        return None
    item, plants = g.item(part), []
    lonely = np.flatnonzero(neighbours(mine, theirs, PLANT_CLEAR_OF_TOL * tol) <= 1)
    if len(lonely):
        plants.append(Plant(f"vertex: {item}", item,
                            _moved(ctx, g, geom, part, mine[lonely[0]], CALIBRATION_FAIL * tol)))
    reach = float(np.linalg.norm(mine.max(axis=0) - mine.min(axis=0)))
    plants.append(Plant(f"vertex far: {item}", item,
                        _moved(ctx, g, geom, part, mine[0], reach + CALIBRATION_FAIL * tol)))
    return PlantUnit(item, tuple(plants))


def _units(ctx: SceneCtx, g: Grown, geom: PieceGeom) -> list[PlantUnit]:
    """The first part both the skeleton and the scene hold, in name order, with a vertex moved. A
    card piece gets a second unit on its first `Cards_<shade>` part, so the leaf cards are
    planted too."""
    tol = SKELETON_MATCH_OF_THICKNESS * tube_base(ctx, g.piece)
    expected = _expected(ctx, g)
    held = sorted(set(expected) & set(geom.parts))
    cards = [n for n in held if n.startswith(CARDS_PREFIX)] if g.card else []
    units = []
    for group in (cards, [n for n in held if n not in cards]):
        unit = next((u for n in group if (u := _part_unit(ctx, g, geom, expected, tol, n))), None)
        if unit is not None:
            units.append(unit)
    if units:
        return units
    # Parts are judged but none is held by both: nothing to move a vertex of. No candidate, so the
    # gate stops unless the real run already reports the part.
    judged = sorted(set(expected) | set(geom.parts))
    return [PlantUnit(g.item(judged[0]), ())] if judged else []


def _comparable(ctx: SceneCtx) -> list[Grown]:
    """The grown pieces with geometry that can be re-meshed: each has a sight when it needs one."""
    return [g for g in grown(ctx, cards=True) if g.key in ctx.pieces and _meshable(ctx, g)]


class SkeletonMatch:
    name = NAME
    kinds: frozenset[str] = KINDS
    scope = "piece"

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [u for g in _comparable(ctx) for u in _units(ctx, g, ctx.pieces[g.key])]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part the skeleton grows (a card piece: its card copy makes) and every part
        shipped, for a piece that has both a skeleton and geometry: each is compared with the
        skeleton's mesh."""
        out: set[str] = set()
        for g in _comparable(ctx):
            shipped = {g.item(name) for name in ctx.pieces[g.key].parts}
            if g.card:
                out |= shipped | {g.item(name) for name in _expected(ctx, g)}
            else:
                out |= g.limb_parts() | g.leaf_parts() | shipped
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for g in grown(ctx, only, cards=True) if g.key in ctx.pieces
                for f in _findings(ctx, g, ctx.pieces[g.key], only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        every = list(grown(ctx, cards=True))
        return (coverage(ctx, NAME, judges_cards=True)
                + [Unmeasured(f"{NAME}: skipped (no geometry): {g.key}")
                   for g in every if g.key not in ctx.pieces]
                + [Unmeasured(f"{NAME}: skipped (the {MESH_VERSION} mesh takes its tube sides from "
                              f"the piece's view and the style's crease angle, and one is "
                              f"missing): {g.key}")
                   for g in every if g.key in ctx.pieces and not g.card
                   and not _meshable(ctx, g)]
                + [Unmeasured(f"{NAME}: skipped (a card copy needs the piece's view, the style's "
                              f"crease angle and its dials.cards and dials.leaves, and one is "
                              f"missing): {g.key}")
                   for g in every if g.key in ctx.pieces and g.card and not _meshable(ctx, g)])
