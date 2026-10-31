"""Check frames, frames and off: each point and direction in the manifest is read in the frame it is
written in, and lies where it says.

Three rules, over each piece of a kind in `kinds` (assembly):

- **Frame match.** Every consumer names the frame it reads: each clip's `frame` and each layout's
  `frame` must equal the `frame.axes` of every piece they place (a clip also counts the pieces
  it carries as props). It reads the manifest only. Item `<pack>:<piece> frame <clip|layout>
  <name>`.
- **On surface.** Every hold, seat and mount point lies within the piece's contact length
  (`sight.contact_length`: one arcminute at `view.nearest_m`) of its surface parts (a mount: of
  any part of the piece). Items `<pack>:<piece> <hold|seat|mount> <id> on-surface`. A piece with
  no `view` has no contact length, so its on-surface items are `Unmeasured`; its `off` items do
  not read it.
- **Off.** A hold's `off` lies within `OFF_ANGLE_DEG` of the outward normal at the nearest
  surface point of its surface parts. A closed part's normals face out by its winding's volume;
  an open part's are taken as wound. Faces tied for nearest all count (a point inside a round part
  is as near to a whole ring of faces), so the `off` need match only one. Item
  `<pack>:<piece> hold <id> off`. This is the check for a tilted `off`, and for a hand that
  passes under a thin plate.

The plants are one unit per item, built from the manifest and the geometry alone (never from a
distance or an angle the check measures): a copy of the piece in another frame than its consumer
reads; a copy whose geometry swaps y and z (a Blender-axes piece), then one moved clear of the
points by `CALIBRATION_FAIL` times the contact length; a copy whose `off` is turned away from an
outward normal at the nearest surface, one way after another and then reversed, by
`CALIBRATION_FAIL` times the angle. The gate tries a unit's candidates in order and stops the run when the check
catches none. A piece with points but no geometry in the scene is `Unmeasured` for them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core import kinds as recipe_kinds
from model_kit.core.checks.entries import Entry, unit
from model_kit.core.checks.pieces import with_piece
from model_kit.core.errors import ManifestError
from model_kit.core.geom import (
    NO_DIRECTION,
    TIE_DISTANCE,
    drop_slivers,
    face_distances,
    is_closed,
    signed_volume,
)
from model_kit.core.manifest import Frame, Hold, Mount, Piece, Seat
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import PieceGeom, SceneCtx, part_key, piece_key
from model_kit.core.sight import contact_length
from model_kit.core.tolerances import CALIBRATION_FAIL, OFF_ANGLE_DEG

NAME = "frames"
KINDS = frozenset({"assembly"})
BLENDER = "blender"  # the axes a plant gives a piece; the loaders accept only `gltf`


@dataclass(frozen=True)
class _Consumer:
    kind: str  # "clip" or "layout"
    name: str
    frame: str
    pack: str
    piece: str


def _consumers(ctx: SceneCtx) -> list[_Consumer]:
    """Who reads which piece's frame: each placement and carried prop of a clip, and each
    placement of a layout. A consumer is listed once per piece."""
    found: dict[tuple[str, str, str, str], _Consumer] = {}
    for clip in ctx.clips.values():
        for ref in (*clip.floor_gear.pieces.values(), *clip.props.values()):
            found.setdefault(("clip", clip.name, ref.pack, ref.piece),
                             _Consumer("clip", clip.name, clip.frame, ref.pack, ref.piece))
    for layout in ctx.layouts.values():
        for ref in layout.placements.values():
            found.setdefault(("layout", layout.name, ref.pack, ref.piece),
                             _Consumer("layout", layout.name, layout.frame, ref.pack, ref.piece))
    return list(found.values())


def _piece_of(ctx: SceneCtx, c: _Consumer) -> Piece:
    pack = ctx.packs.get(c.pack)
    piece = None if pack is None else pack.pieces.get(c.piece)
    if piece is None:
        raise ManifestError(f"{c.kind} {c.name!r} places {piece_key(c.pack, c.piece)}, which "
                            "the scene lacks")
    return piece


# --- frame match ------------------------------------------------------------------------------


def _frame_item(key: str, c: _Consumer) -> str:
    return f"{key} frame {c.kind} {c.name}"


def _frame_entry(pack: str, name: str, piece: Piece, c: _Consumer) -> Entry:
    detail = (f"{c.kind} {c.name} reads frame {c.frame!r}, but the piece's frame.axes is "
              f"{piece.frame.axes!r}")
    return Entry(_frame_item(piece_key(pack, name), c), piece.frame.axes == c.frame, detail)


# --- on surface and off -----------------------------------------------------------------------


def _surface(geom: PieceGeom, names: Collection[str], what: str
             ) -> list[tuple[np.ndarray, float]]:
    """The surface parts' triangles (zero-area ones dropped) with each part's outward sign:
    -1 when a closed part is wound inward, else 1."""
    out = []
    for part in names:
        if part not in geom.parts:
            raise ManifestError(f"{what}: surface part {part!r} has no geometry in the scene; "
                                f"it has {sorted(geom.parts)}")
        whole = np.asarray(geom.parts[part], dtype=float)
        tris = drop_slivers(whole)
        if len(tris):  # a sliver may close a gap or open one, so judge both ways
            closed = is_closed(whole) or is_closed(tris)
            out.append((tris, -1.0 if closed and signed_volume(tris) < 0 else 1.0))
    if not out:
        raise ManifestError(f"{what}: the surface has no triangle with area")
    return out


def _nearest(surface: list[tuple[np.ndarray, float]], point) -> tuple[float, np.ndarray]:
    """The distance from `point` to the surface, and the outward unit normals of the faces tied
    for nearest (within `TIE_DISTANCE` of the least distance)."""
    rows = [face_distances(tris, point) for tris, _ in surface]
    least = min(float(d.min()) for d, _ in rows)
    normals = np.concatenate([n[d <= least + TIE_DISTANCE] * sign
                              for (d, n), (_, sign) in zip(rows, surface, strict=True)])
    return least, normals[np.linalg.norm(normals, axis=1) > 0.5]


def _unit(v, what: str) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    length = float(np.linalg.norm(v))
    if length < NO_DIRECTION:
        raise ManifestError(f"{what}: the direction has no length")
    return v / length


def _swapped(ctx: SceneCtx, key: str) -> SceneCtx:
    geom = ctx.pieces[key]
    parts = {n: np.asarray(t, dtype=float)[..., [0, 2, 1]] for n, t in geom.parts.items()}
    return dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom(geom.name, parts)})


def _moved(ctx: SceneCtx, key: str, clear: float) -> SceneCtx:
    """The geometry moved along +x past its own extent and `clear` more."""
    geom = ctx.pieces[key]
    every = np.concatenate([np.asarray(t, dtype=float).reshape(-1, 3)
                            for t in geom.parts.values()])
    delta = np.array([float(np.ptp(every[:, 0])) + clear, 0.0, 0.0])
    parts = {n: np.asarray(t, dtype=float) + delta for n, t in geom.parts.items()}
    return dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom(geom.name, parts)})


def _tilts(ctx: SceneCtx, pack: str, piece: Piece, hold: Hold, normals: np.ndarray
           ) -> Iterator[tuple[str, SceneCtx]]:
    """Copies whose hold `off` is turned `CALIBRATION_FAIL` times the limit from an outward
    normal, one way after another around it, and last the reverse of it; for each normal in turn
    (those tied for nearest to the hold's point, each once)."""
    a = np.radians(min(CALIBRATION_FAIL * OFF_ANGLE_DEG, 180.0))
    seen: set[tuple[float, ...]] = set()
    for normal in normals:
        if (key := tuple(np.round(normal, 4).tolist())) in seen:
            continue
        seen.add(key)
        axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
        first = _unit(np.cross(normal, axis), "off plant")
        second = np.cross(normal, first)
        for turn in (first, -first, second, -second):
            yield "tilted", _with_off(ctx, pack, piece, hold, np.cos(a) * normal + np.sin(a) * turn)
        yield "tilted", _with_off(ctx, pack, piece, hold, -normal)


def _with_off(ctx: SceneCtx, pack: str, piece: Piece, hold: Hold, off: np.ndarray) -> SceneCtx:
    off = tuple(float(v) for v in off)
    holds = tuple(dataclasses.replace(h, off=off) if h.id == hold.id else h for h in piece.holds)
    return with_piece(ctx, pack, piece.name, dataclasses.replace(piece, holds=holds))


def _rows(piece: Piece, geom: PieceGeom) -> list[tuple[str, Hold | Seat | Mount, tuple[str, ...]]]:
    """Each hold, seat and mount of the piece with the surface parts it is read against."""
    return [("hold", h, h.surface) for h in piece.holds] \
        + [("seat", s, s.surface) for s in piece.seats] \
        + [("mount", m, tuple(geom.parts)) for m in piece.mounts]


def _limit(ctx: SceneCtx, pack: str, piece: Piece) -> float | None:
    """How far a point may sit from its surface: the piece's contact length, None without a
    `view`."""
    return contact_length(piece, ctx.packs[pack].units_per_metre)


def _point_entries(ctx: SceneCtx, pack: str, piece: Piece,
                   only: Collection[str] | None) -> Iterator[Entry]:
    key = piece_key(pack, piece.name)
    geom = ctx.pieces[key]
    limit = _limit(ctx, pack, piece)
    for kind, spot, names in _rows(piece, geom):
        on_item = f"{key} {kind} {spot.id} on-surface"
        off_item = f"{key} hold {spot.id} off" if kind == "hold" else None
        want_on = limit is not None and (only is None or on_item in only)
        want_off = off_item is not None and (only is None or off_item in only)
        if not (want_on or want_off):
            continue
        what = f"{key} {kind} {spot.id}"
        distance, normals = _nearest(_surface(geom, names, what), spot.point)
        if want_on:
            yield Entry(on_item, distance <= limit,
                        f"{distance:.4f} from its surface (limit {limit:.4f})")
        if not want_off:
            continue
        off = _unit(spot.off, what)
        cosines = np.clip(normals @ off, -1.0, 1.0)
        best = int(np.argmax(cosines))
        angle = float(np.degrees(np.arccos(cosines[best])))
        yield Entry(off_item, angle <= OFF_ANGLE_DEG,
                    f"off turns {angle:.1f} degrees from the outward normal (limit "
                    f"{OFF_ANGLE_DEG:g})" + (", so it points into the surface" if angle > 90.0
                                             else ""))


def _point_units(ctx: SceneCtx, pack: str, piece: Piece) -> Iterator[PlantUnit]:
    """A unit for each on-surface item and each hold's `off`, from the points and the geometry
    alone: no distance or angle is compared with its limit. A piece with no contact length has
    no on-surface items."""
    key = piece_key(pack, piece.name)
    geom = ctx.pieces[key]
    limit = _limit(ctx, pack, piece)
    away = None if limit is None else [
        ("swapped", _swapped(ctx, key)),
        ("moved", _moved(ctx, key, CALIBRATION_FAIL * limit))]  # once a piece
    for kind, spot, names in _rows(piece, geom):
        if away is not None:
            yield unit(f"{key} {kind} {spot.id} on-surface", away)
        if kind == "hold":
            what = f"{key} {kind} {spot.id}"
            normals = _nearest(_surface(geom, names, what), spot.point)[1]
            yield unit(f"{key} hold {spot.id} off", _tilts(ctx, pack, piece, spot, normals))


# --- the check --------------------------------------------------------------------------------


def _entries(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[Entry]:
    consumers = _consumers(ctx)
    for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, KINDS):
        key = piece_key(pack, name)
        for c in consumers:
            if (c.pack, c.piece) == (pack, name):
                entry = _frame_entry(pack, name, piece, c)
                if only is None or entry.item in only:
                    yield entry
        if key not in ctx.pieces:
            continue
        yield from _point_entries(ctx, pack, piece, only)


def _findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    for c in _consumers(ctx):  # a consumer of a piece the scene lacks stops the run
        _piece_of(ctx, c)
    return [Finding(NAME, e.item, e.detail) for e in _entries(ctx, only) if not e.ok]


def _units(ctx: SceneCtx) -> Iterator[PlantUnit]:
    consumers = _consumers(ctx)
    for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, KINDS):
        key = piece_key(pack, name)
        for c in consumers:
            if (c.pack, c.piece) == (pack, name):
                other = BLENDER if c.frame != BLENDER else "gltf"  # a frame the consumer is not in
                yield unit(_frame_item(key, c), [("frame", with_piece(
                    ctx, pack, name, dataclasses.replace(piece, frame=Frame(other,
                                                                            piece.frame.origin))))])
        if key not in ctx.pieces:
            continue
        yield from _point_units(ctx, pack, piece)


class Frames:
    name = NAME
    kinds: frozenset[str] = KINDS

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return list(_units(ctx))

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The parts the piece's points are measured against: the surface parts of its holds
        and seats, and, for each mount, the one part nearest to it (a mount is measured against
        any part, but it only vouches for the part it sits on)."""
        out: set[str] = set()
        for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, KINDS):
            geom = ctx.pieces.get(piece_key(pack, name))
            if geom is None:
                continue
            named = {part for spot in (*piece.holds, *piece.seats) for part in spot.surface}
            parts = {part for part in named if part in geom.parts}
            for mount in piece.mounts:
                point = np.asarray(mount.point, dtype=float)
                near = {part: float(face_distances(tris, point)[0].min())
                        for part, tris in geom.parts.items() if len(tris)}
                if near:
                    parts.add(min(near, key=near.__getitem__))
            out |= {part_key(pack, name, part) for part in parts}
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _findings(ctx, only)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = list(recipe_kinds.skipped_kinds(ctx, NAME, KINDS))
        if not _consumers(ctx):
            lines.append(f"{NAME}: no clip or layout places a piece: the frame match is not "
                         "measured")
        for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, KINDS):
            if (piece_key(pack, name) not in ctx.pieces
                    and (piece.holds or piece.seats or piece.mounts)):
                lines.append(Unmeasured(f"{NAME}: skipped (no geometry): "
                                        f"{piece_key(pack, name)}"))
            elif (piece.holds or piece.seats or piece.mounts) and _limit(ctx, pack, piece) is None:
                lines.append(Unmeasured(f"{NAME}: skipped (the piece declares no view, so it has "
                                        f"no contact length; on-surface rule): "
                                        f"{piece_key(pack, name)}"))
        return lines
