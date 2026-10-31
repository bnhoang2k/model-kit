"""What the piece checks share: the geometry checks (crossings, take-apart, floaters, topology)
read a piece's parts through it, and the manifest and assembly checks (frames, sides, receiver,
connectors, done, keep-off) and coverage use its helpers to build a plant (`with_piece`,
`free_name`).

A piece check measures only the pieces whose recipe kind is in its `kinds` (`core.kinds`), reads
each piece's parts from `ctx.pieces`, and builds its plants by changing a copy of one piece.

A piece's parts are its geometry keys, in the order the geometry file holds them. The pack's
part record, when there is one, gives a part its `role`, `shape`, `group` and `on`. A geometry
part with no record is measured as a solid of the group "unlisted" that stands on nothing, so a
scene with a thin manifest is still measured, and the gaps show in the findings. A part record
with no geometry is not measured.

A piece of a measured kind with no geometry in the scene is missing data: it is an `Unmeasured`
coverage line, so the run is `partial`. A check that cannot show it can fail on a piece does not
say so here: its plants are units (`metagate.PlantUnit`), and the gate stops the run with
`BlindCheck` when no candidate of a unit is caught.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from model_kit.core import kinds as recipe_kinds
from model_kit.core.manifest import Layout, LayoutPlacement, Pack, Part, Piece
from model_kit.core.manifest.records import PartKind
from model_kit.core.metagate import Unmeasured
from model_kit.core.scene import PieceGeom, SceneCtx, part_key, piece_key
from model_kit.core.sight import contact_length, seen_floor
from model_kit.core.tolerances import EYE_RESOLUTION_ARCMIN

# The group of a geometry part that has no record in the pack's manifest.
NO_GROUP = "unlisted"


@dataclass(frozen=True)
class PartMesh(PartKind):
    """A part's geometry. The facts derived from its role and shape (`printed`, `solid`,
    `skips_outline`, `open_surface`, `bends`) are `PartKind`'s, the same as its `Part`'s."""

    name: str
    tris: np.ndarray  # (N, 3, 3), in the piece's frame
    role: str
    shape: str | None
    group: str
    on: str | None

    @property
    def lo(self) -> np.ndarray:
        return self.tris.reshape(-1, 3).min(axis=0) if len(self.tris) else np.zeros(3)

    @property
    def hi(self) -> np.ndarray:
        return self.tris.reshape(-1, 3).max(axis=0) if len(self.tris) else np.zeros(3)


@dataclass(frozen=True)
class PieceView:
    pack: str
    name: str
    piece: Piece
    contact: float | None  # the piece's contact length (`sight.contact_length`); None: no view
    parts: tuple[PartMesh, ...]

    @property
    def key(self) -> str:
        return piece_key(self.pack, self.name)

    @property
    def size(self) -> float:
        """The piece size: the diagonal of the box of all its parts (0.0 with no triangles)."""
        points = [p.tris.reshape(-1, 3) for p in self.parts if len(p.tris)]
        if not points:
            return 0.0
        every = np.concatenate(points)
        return float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))

    def solids(self) -> list[PartMesh]:
        return [p for p in self.parts if p.solid]


def item_of(view: PieceView, *names: str) -> str:
    """`<pack>:<piece> <name>`, or `<pack>:<piece> <a> x <b>` for two names, sorted."""
    if len(names) == 1:
        return part_key(view.pack, view.name, names[0])
    return f"{view.key} {' x '.join(sorted(names))}"


def part_mesh(piece: Piece, name: str, tris: np.ndarray) -> PartMesh:
    """A geometry part read with its record in `piece` (see the module note for a part with no
    record)."""
    record = next((p for p in piece.parts if p.name == name), None)
    if record is None:
        return PartMesh(name, tris, role="solid", shape=None, group=NO_GROUP, on=None)
    return PartMesh(name, tris, role=record.role, shape=record.shape, group=record.group,
                    on=record.on)


def family(parts: Iterable[PartMesh], name: str) -> list[PartMesh]:
    """The solid part `name` and every solid part that rests on it, down the `on` chain (a plug
    body and the blades on it), in the order of `parts`. A part with no triangles or that bends
    (a cable) is left out. Empty when `name` is no such solid."""
    solid = {p.name: p for p in parts if p.solid and not p.bends and len(p.tris)}
    if name not in solid:
        return []
    found = {name}
    grew = True
    while grew:
        grew = False
        for part in solid.values():
            if part.name not in found and part.on in found:
                found.add(part.name)
                grew = True
    return [p for p in solid.values() if p.name in found]


def merged(name: str, parts: Sequence[PartMesh]) -> PartMesh:
    """One mesh of these parts, named `name`: their triangles together, the role, shape, group
    and `on` of the first (the part the others rest on)."""
    first = parts[0]
    return PartMesh(name, np.concatenate([p.tris for p in parts]), first.role, first.shape,
                    first.group, first.on)


def measured_pieces(ctx: SceneCtx, kinds: Collection[str],
                    only: Collection[str] | None = None) -> Iterator[PieceView]:
    """The pieces of these recipe kinds that the scene holds geometry for, in pack then piece
    order. With `only` (a scoped run's items), just the pieces the items name."""
    for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, kinds):
        key = piece_key(pack, name)
        geom = ctx.pieces.get(key)
        if geom is None or (only is not None and not any(i.startswith(f"{key} ") for i in only)):
            continue
        parts = tuple(part_mesh(piece, n, t) for n, t in geom.parts.items())
        contact = contact_length(piece, ctx.packs[pack].units_per_metre)
        yield PieceView(pack, name, piece, contact, parts)


def contact_pieces(ctx: SceneCtx, kinds: Collection[str],
                   only: Collection[str] | None = None) -> Iterator[PieceView]:
    """`measured_pieces` that can be judged for contact: the pieces with a `view`, so a contact
    length (`sight.contact_length`). A piece with none is left to `unviewed`."""
    return (v for v in measured_pieces(ctx, kinds, only) if v.contact is not None)


def unviewed(ctx: SceneCtx, check: str, kinds: Collection[str]) -> list[str]:
    """An `Unmeasured` line for each piece of these kinds that has geometry and no `view`: with no
    nearest viewing distance it has no contact length, so a contact rule cannot judge it."""
    return [Unmeasured(f"{check}: skipped (the piece declares no view, so it has no contact "
                       f"length): {v.key}")
            for v in measured_pieces(ctx, kinds) if v.contact is None]


def coverage(ctx: SceneCtx, check: str, kinds: Collection[str], *,
             printed: bool = False) -> list[str]:
    """What the check leaves to others or could not measure. A piece of another recipe kind is
    a plain line: that kind's own checks measure it (the branching checks for a
    `branching` piece). A piece of these kinds with no geometry in the scene is `Unmeasured`.
    With `printed`, plain `not applicable (...)` lines for the parts printed on a surface (role
    `eyes` or `decal`), which the check leaves out by design; the gate prints its own plant
    lines in the same form."""
    lines: list[str] = list(recipe_kinds.skipped_kinds(ctx, check, kinds))
    for pack, name, _ in recipe_kinds.pieces_of_kind(ctx, kinds):
        if piece_key(pack, name) not in ctx.pieces:
            lines.append(Unmeasured(f"{check}: skipped (no geometry): {piece_key(pack, name)}"))
    if printed:
        lines += [f"{check}: not applicable ({p.role} part, printed on a surface): "
                  f"{v.key} {p.name}"
                  for v in measured_pieces(ctx, kinds) for p in v.parts if not p.solid]
    return lines


@dataclass
class Sight:
    """How a piece will be seen, for a check that drops what a viewer would not see.
    `floor` is the length the eye resolves at its nearest viewing distance and
    `hidden` the sides it never shows. The check puts each finding it drops into `under` (smaller
    than `floor`) or `unseen` (on a hidden side) by item, and each it keeps into `shown`: an item
    kept at one clip sample is not counted as dropped."""

    floor: float
    hidden: tuple[str, ...] = ()
    under: set[str] = field(default_factory=set)
    unseen: set[str] = field(default_factory=set)
    shown: set[str] = field(default_factory=set)

    @property
    def dropped(self) -> int:
        """How many items the floor dropped."""
        return len(self.under - self.shown)

    @property
    def hid(self) -> int:
        """How many items lay on hidden sides."""
        return len(self.unseen - self.shown)


def sight_of(piece: Piece, units_per_metre: float) -> Sight | None:
    """The sight of a piece, or None when it declares no `view`: no floor and no hidden side, so
    the check judges it as it always did."""
    floor = seen_floor(piece, units_per_metre)
    return None if floor is None else Sight(floor, piece.view.hidden)


def view_sight(ctx: SceneCtx, view: PieceView) -> Sight | None:
    """`sight_of` for a piece the scene holds (its pack gives the units)."""
    return sight_of(view.piece, ctx.packs[view.pack].units_per_metre)


def under_floor_line(check: str, sight: Sight, piece: Piece, key: str) -> str:
    """The coverage line for the seen floor: how many findings of `check` on the piece `key` were
    dropped because they are smaller than the eye resolves at its nearest viewing distance."""
    return (f"{check}: {sight.dropped} under {EYE_RESOLUTION_ARCMIN:g} arcmin at "
            f"{piece.view.nearest_m:g} m: {key}")


def hidden_line(check: str, sight: Sight, piece: Piece, key: str, what: str) -> list[str]:
    """The coverage line for findings dropped for lying on a side the view never shows (none when
    there are none)."""
    if not sight.hid:
        return []
    return [f"{check}: {sight.hid} {what} on hidden sides ({', '.join(piece.view.hidden)}): {key}"]


def placements_of(layout: Layout, pack: str, piece: str) -> list[LayoutPlacement]:
    """The placements of a piece in a layout. A check that reads one piece through "its"
    placement (a cable's end, a plug's seat) needs exactly one: none or several leave it
    unsaid which one is meant."""
    return [p for p in layout.placements.values() if (p.pack, p.piece) == (pack, piece)]


def centroid(tris: np.ndarray) -> np.ndarray:
    """The surface's centroid: triangle centres weighted by area."""
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    if area.sum() <= 0.0:
        return tris.mean(axis=(0, 1))
    return (tris.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()


# --- plants -----------------------------------------------------------------------------------


def free_name(taken: Collection[str], stem: str) -> str:
    name = f"{stem}_plant"
    while name in taken:
        name += "_"
    return name


def box_mesh(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """A closed box between two corners: 12 triangles, all facing out."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    quads = [
        [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)],  # +y
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # -y
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # +x
        [(x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0)],  # -x
        [(x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)],  # +z
        [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],  # -z
    ]
    return np.array([t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))],
                    dtype=float)


def with_geometry(ctx: SceneCtx, view: PieceView, parts: dict[str, np.ndarray]) -> SceneCtx:
    """The scene with this piece's geometry replaced by `parts`."""
    geom = PieceGeom(view.name, parts)
    return dataclasses.replace(ctx, pieces={**ctx.pieces, view.key: geom})


def with_piece(ctx: SceneCtx, pack_name: str, name: str, piece: Piece) -> SceneCtx:
    """The scene with this piece's manifest record replaced by `piece`."""
    pack = ctx.packs[pack_name]
    pack = Pack(name=pack.name, units_per_metre=pack.units_per_metre, spacing=pack.spacing,
                pieces={**pack.pieces, name: piece})
    return dataclasses.replace(ctx, packs={**ctx.packs, pack_name: pack})


def with_part_record(ctx: SceneCtx, view: PieceView, record: Part) -> SceneCtx:
    """The scene with a part record added to this piece's manifest."""
    piece = dataclasses.replace(view.piece, parts=(*view.piece.parts, record))
    return with_piece(ctx, view.pack, view.name, piece)


def box_overlaps(parts: list[PartMesh], pad: float = 0.0) -> Iterator[tuple[int, int]]:
    """Index pairs `(i, j)`, `i < j`, of parts whose boxes, grown by `pad`, overlap."""
    if len(parts) < 2:
        return
    lo = np.array([p.lo for p in parts])
    hi = np.array([p.hi for p in parts])
    near = ((lo[:, None, :] - pad <= hi[None, :, :]) & (lo[None, :, :] - pad <= hi[:, None, :])
            ).all(axis=2)
    for i, j in np.argwhere(np.triu(near, k=1)):
        yield int(i), int(j)
