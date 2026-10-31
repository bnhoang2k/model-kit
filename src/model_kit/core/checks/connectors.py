"""Check connectors, connector ends and spans: a cable lands where it says, and is no longer than its cap.

A piece lists `connectors`: a cable part with two ends and a `cap`, the longest it may be in model
units. For each connector, up to seven items named `<pack>:<piece> <connector id> <rule>`, the rule
one of `end0`, `end1`, `end0 cable`, `end1 cable`, `end0 seat`, `end1 seat` and `cap`:

- **end0, end1.** The end's point lies within the connector piece's contact length
  (`sight.contact_length`: one arcminute at its `view.nearest_m`) of the mount it plugs into. An end
  that names a placement (`host`, with the `layout` the end names) is compared with the mount's
  point through that placement, in the layout's frame. An end on `self.<mount>` is compared with
  the mount of the connector's own
  piece, in the piece's frame.
- **end0 cable, end1 cable.** The cable part's surface comes within the same limit of the end's
  point, or, when the end names a `plug`, the cable's surface comes within it of the plug's and the
  plug's surface within it of the point (a cord enters the back of its plug, which stands at the
  mount's point). `end0` and `end1` compare two manifest numbers, so they pass by construction; this
  rule reads the cable's geometry. For an end on `self.<mount>` the cable is read in the piece's own
  frame, where the end's point is written. For an end on a layout, the cable is moved through the
  placement of the cable's piece in that layout, into the layout's frame. A layout with no
  placement of the cable's piece, or with several (the end then does not say which one it
  means), leaves the item `Unmeasured` ("no placement" or "several placements"), and so does a
  cable with no geometry.
- **end0 seat, end1 seat.** An end on a layout seats: its `plug` (the part of the connector's
  piece that seats, with the parts resting on it, the blades of a plug body) sits in the host part
  the mount names (`Mount.part`), judged by the joint helper (`joints.seat`) in the layout's frame,
  within the same limit. The plug must rest on the host part, and, for a `power` or `signal`
  mount, stand in a bore of it: a plug on a face with no bore round it fails. A `rest` mount
  declares that it has no bore. An end on `self.<mount>` is attached to its own piece, not seated,
  and has no such item. An end that names no `plug`, a mount that names no `part`, a part the
  scene has no geometry for, a layout with no placement or several placements of the cable's
  piece, or a host that is no closed solid leaves the item `Unmeasured`.
- **cap.** The cable part's length is at most `cap`. The length is the part's longest side in a
  box turned to follow it: the extent of its corners along their principal axis, so a cable
  lying slantwise reads as long as it is. It is the straight run from end to end, so a cable
  that winds on the way is longer than it reads.

A connector on a piece that declares no `view` has no contact length: its end, end cable and end
seat items are `Unmeasured`, and its cap is still judged.

A layout, placement or mount an end names that the scene lacks raises, naming both (the loaders
refuse this too; a scene built in code is guarded like a loaded one). A cable with no geometry,
or no triangles, in the scene gets no cap item and is `Unmeasured` ("no geometry"); its ends are
still measured, because they read the manifest.

The plants are one unit per item, built from the manifest and the geometry alone (never from the
gap or the span the check measures). The gate tries a unit's candidates in order and stops the run
when the check catches none:

- an end: its point moved `CALIBRATION_FAIL` times the limit along each axis, both ways (one of
  each pair leaves it more than the limit from the mount, whichever way it sat);
- an end's cable: the cable's geometry moved (the manifest is not touched) by the farthest any of
  its corners lies from the end's point, plus the farthest any corner of the end's plug lies from
  it, and `CALIBRATION_FAIL` times the limit more, so every point of it ends that far from the
  end's point and from the plug;
- an end's seat: the plug and the parts resting on it moved (the manifest is not touched) in the
  layout's frame, along each side at right angles to the mount's axis, both ways, by
  `CALIBRATION_FAIL` times the width of the thinnest of those parts along that side (a blade
  leaves its bore for the face beside it), then by `CALIBRATION_FAIL` times the plug's extent
  along that side plus the limit (clear of the mount's point), then along the axis, both ways
  (off the host, into it);
- the cap: the cable stretched along its axis to `cap` plus `CALIBRATION_FAIL` times the cap
  and, for a cable of no length that cannot be stretched, replaced by a bar of that length.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks import joints
from model_kit.core.checks.entries import Entry, unit
from model_kit.core.checks.pieces import (
    PartMesh,
    family,
    merged,
    part_mesh,
    placements_of,
    with_piece,
)
from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION, closest_points, is_closed, surface_gap
from model_kit.core.manifest import Connector, End, LayoutPlacement, Mount, Piece, Placement
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import PieceGeom, SceneCtx, part_key, piece_key
from model_kit.core.sight import contact_length
from model_kit.core.tolerances import CALIBRATION_FAIL

NAME = "connectors"
PRINCIPLE = joints.PRINCIPLE
SOURCE = joints.SOURCE


@dataclass(frozen=True)
class _Conn:
    pack: str
    piece_name: str
    piece: Piece
    conn: Connector
    limit: float | None  # the piece's contact length; None: it declares no view

    @property
    def key(self) -> str:
        return piece_key(self.pack, self.piece_name)

    def item(self, rule: str) -> str:
        return f"{self.key} {self.conn.id} {rule}"


def _connectors(ctx: SceneCtx, only: Collection[str] | None = None) -> Iterator[_Conn]:
    for pack_name in sorted(ctx.packs):
        pack = ctx.packs[pack_name]
        for name, piece in pack.pieces.items():
            limit = contact_length(piece, pack.units_per_metre)
            for conn in piece.connectors:
                if only is not None and not any(
                        i.startswith(f"{piece_key(pack_name, name)} {conn.id} ") for i in only):
                    continue
                yield _Conn(pack_name, name, piece, conn, limit)


def _host(ctx: SceneCtx, c: _Conn, end: End, index: int
          ) -> tuple[LayoutPlacement, Piece, Mount]:
    """The placement, piece and mount a layout end plugs into. What the scene lacks raises."""
    where = f"{c.key} connector {c.conn.id!r} ends[{index}]"
    layout = ctx.layouts.get(end.layout)
    if layout is None:
        raise ManifestError(f"{where}: names layout {end.layout!r}, which the scene lacks "
                            f"(it has {sorted(ctx.layouts)})")
    host = layout.placements.get(end.host.piece)
    if host is None:
        raise ManifestError(f"{where}: {end.host} names no placement {end.host.piece!r} of "
                            f"layout {layout.name!r}")
    host_pack = ctx.packs.get(host.pack)
    host_piece = None if host_pack is None else host_pack.pieces.get(host.piece)
    if host_piece is None:
        raise ManifestError(f"{where}: placement {host.id!r} is of {host.pack}:{host.piece}, "
                            "which the scene lacks")
    mount = next((m for m in host_piece.mounts if m.id == end.host.id), None)
    if mount is None:
        raise ManifestError(f"{where}: {end.host}: piece {host.pack}:{host.piece} publishes no "
                            f"mount {end.host.id!r}")
    return host, host_piece, mount


def _placed(p: LayoutPlacement) -> Placement:
    return Placement(p.id, p.piece, p.pack, p.position, p.quaternion)


def _target(ctx: SceneCtx, c: _Conn, end: End, index: int) -> np.ndarray:
    """Where the end's mount is, in the frame the end's point is written in."""
    where = f"{c.key} connector {c.conn.id!r} ends[{index}]"
    if end.host.is_self:
        mount = next((m for m in c.piece.mounts if m.id == end.host.id), None)
        if mount is None:
            raise ManifestError(f"{where}: {end.host}: the piece has no mount {end.host.id!r}")
        return np.asarray(mount.point, dtype=float)
    host, _, mount = _host(ctx, c, end, index)
    return _placed(host).to_world(mount.point)


def _with_connector(ctx: SceneCtx, c: _Conn, conn: Connector) -> SceneCtx:
    conns = tuple(conn if x.id == c.conn.id else x for x in c.piece.connectors)
    return with_piece(ctx, c.pack, c.piece_name, dataclasses.replace(c.piece, connectors=conns))


def _cable(ctx: SceneCtx, c: _Conn) -> np.ndarray | None:
    geom = ctx.pieces.get(c.key)
    tris = None if geom is None else geom.parts.get(c.conn.part)
    return None if tris is None or len(tris) == 0 else np.asarray(tris, dtype=float)


def _axis(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, principal axis, position of each corner along it) of a point cloud."""
    centre = corners.mean(axis=0)
    if len(corners) < 2:
        return centre, np.array([1.0, 0.0, 0.0]), np.zeros(len(corners))
    _, _, vt = np.linalg.svd(corners - centre, full_matrices=False)
    axis = vt[0]
    return centre, axis, (corners - centre) @ axis


def length(tris: np.ndarray) -> float:
    """A cable part's length: the extent of its corners along their principal axis."""
    _, _, along = _axis(np.unique(np.asarray(tris, dtype=float).reshape(-1, 3), axis=0))
    return float(along.max() - along.min()) if len(along) else 0.0


def _stretched(tris: np.ndarray, to: float) -> np.ndarray | None:
    """The part stretched along its principal axis, about its middle, to length `to`; None when
    it has no length to stretch."""
    corners = np.unique(tris.reshape(-1, 3), axis=0)
    centre, axis, along = _axis(corners)
    now = float(along.max() - along.min()) if len(along) else 0.0
    if now <= NO_DIRECTION:
        return None
    grown = tris.reshape(-1, 3) - centre
    along_each = grown @ axis
    stretched = grown + np.outer(along_each * (to / now - 1.0), axis) + centre
    return stretched.reshape(tris.shape)


NO_LENGTH = "no contact length: the piece declares no view"


def _end_entry(ctx: SceneCtx, c: _Conn, index: int) -> Entry | str:
    """The rule that end `index`'s point is on its mount, or why it cannot be read."""
    end = c.conn.ends[index]
    target = _target(ctx, c, end, index)
    if c.limit is None:
        return NO_LENGTH
    gap = float(np.linalg.norm(np.asarray(end.point, dtype=float) - target))
    return Entry(c.item(f"end{index}"), gap <= c.limit,
                 f"end {index} is {gap:.4f} from {end.host} (limit {c.limit:.4f})")


def _in_end_frame(ctx: SceneCtx, c: _Conn, end: End, tris: np.ndarray
                  ) -> tuple[np.ndarray, Placement | None] | str:
    """The cable's triangles in the frame the end's point is written in (and the placement that
    carried them there, None for the piece's own frame), or the reason there is no such frame."""
    if end.host.is_self:
        return tris, None
    mine = placements_of(ctx.layouts[end.layout], c.pack, c.piece_name)
    if not mine:
        return "no placement"
    if len(mine) > 1:
        return "several placements"
    placement = _placed(mine[0])
    return placement.to_world(tris), placement


def _cable_in_frame(ctx: SceneCtx, c: _Conn, end: End, index: int
                    ) -> tuple[np.ndarray, Placement | None] | str:
    """The cable's triangles in the frame the end's point is written in (with the placement that
    carried them there), or why the end's cable rule cannot be read."""
    tris = _cable(ctx, c)
    if tris is None:
        return "no geometry"
    _target(ctx, c, end, index)  # a layout, placement or mount the scene lacks raises
    return _in_end_frame(ctx, c, end, tris)


def _plug_in_frame(ctx: SceneCtx, c: _Conn, end: End) -> np.ndarray | None:
    """The triangles of the end's plug (its part and the parts resting on it) in the frame the
    end's point is written in; None when the end names no plug or the scene has none of it."""
    geom = ctx.pieces.get(c.key)
    if end.plug is None or geom is None:
        return None
    plug = family([part_mesh(c.piece, n, t) for n, t in geom.parts.items()], end.plug)
    if not plug:
        return None
    framed = _in_end_frame(ctx, c, end, np.concatenate([p.tris for p in plug]))
    return None if isinstance(framed, str) else framed[0]


def _cable_end_entry(ctx: SceneCtx, c: _Conn, index: int) -> Entry | str:
    """The rule that the cable's surface reaches end `index`'s point, or that of the plug the end
    names (a cord enters the back of its plug, not the mount's point), or why it cannot be read."""
    end = c.conn.ends[index]
    framed = _cable_in_frame(ctx, c, end, index)
    if isinstance(framed, str):
        return framed
    if c.limit is None:
        return NO_LENGTH
    gap = float(closest_points(framed[0], np.asarray(end.point, dtype=float)[None, :]).dist[0])
    plug = _plug_in_frame(ctx, c, end)
    if plug is not None and gap > c.limit:
        # a cord enters its plug, and the plug stands at the point: the cable reaches the point
        # through the plug when both joins hold
        at_plug = surface_gap(framed[0], plug, gap)
        at_point = float(closest_points(plug, np.asarray(end.point, dtype=float)[None, :]).dist[0])
        gap = min(gap, max(at_plug, at_point))
    where = (f"to end {index}'s point" if plug is None
             else f"to end {index}'s point, by way of its plug")
    return Entry(c.item(f"end{index} cable"), gap <= c.limit,
                 f"the cable {c.conn.part} comes no nearer than {gap:.4f} {where} "
                 f"(limit {c.limit:.4f})")


@dataclass(frozen=True)
class _Seat:
    """What the seat rule reads for one layout end, in the layout's frame: the plug (its part
    and the parts resting on it, joined into one mesh) and the host part, the mount's point, and
    the way the mount points out of the host when it needs a hole. `members` names the plug's
    parts, `placement` carries the cable's piece into the layout."""

    guest: PartMesh
    parts: tuple[PartMesh, ...]
    host: PartMesh
    point: np.ndarray
    out: np.ndarray
    bore_axis: np.ndarray | None
    members: tuple[str, ...]
    placement: Placement
    host_key: str


def _seat_read(ctx: SceneCtx, c: _Conn, index: int) -> _Seat | str:
    """The end's seat as the joint helper reads it, or why it cannot be read."""
    end = c.conn.ends[index]
    host, host_piece, mount = _host(ctx, c, end, index)  # what the scene lacks raises
    if mount.part is None:
        return "no host part named"
    if end.plug is None:
        return "no plug named"
    geom = ctx.pieces.get(c.key)
    plug = family([part_mesh(c.piece, n, t) for n, t in (geom.parts if geom else {}).items()],
                  end.plug)
    if not plug:
        return "no geometry"
    host_geom = ctx.pieces.get(piece_key(host.pack, host.piece))
    host_tris = None if host_geom is None else host_geom.parts.get(mount.part)
    if host_tris is None or len(host_tris) == 0:
        return "no geometry for the host part"
    if mount.kind in joints.PORT_KINDS and not is_closed(host_tris):
        # checked in the host's own frame: a turn blurs the exact corners `is_closed` matches
        return (f"the host part {mount.part} is not a closed solid, so a hole in it cannot be "
                "told from its material")
    mine = placements_of(ctx.layouts[end.layout], c.pack, c.piece_name)
    if not mine:
        return "no placement"
    if len(mine) > 1:
        return "several placements"
    placement, hosted = _placed(mine[0]), _placed(host)
    placed = [dataclasses.replace(p, tris=placement.to_world(p.tris)) for p in plug]
    out = hosted.to_world(mount.axis) - hosted.to_world(np.zeros(3))
    host_mesh = part_mesh(host_piece, mount.part, hosted.to_world(host_tris))
    return _Seat(merged(end.plug, placed), tuple(placed), host_mesh,
                 hosted.to_world(mount.point), out,
                 out if mount.kind in joints.PORT_KINDS else None,
                 tuple(p.name for p in plug), placement, piece_key(host.pack, host.piece))


def _seat_entry(ctx: SceneCtx, c: _Conn, index: int) -> Entry | str:
    """The rule that end `index`'s plug seats in the host part its mount names, or why it cannot
    be read."""
    read = _seat_read(ctx, c, index)
    if isinstance(read, str):
        return read
    if c.limit is None:
        return NO_LENGTH
    joint = joints.seat(ctx, guest=read.guest, host=read.host, point=read.point, locate=c.limit,
                        bore_axis=read.bore_axis)
    end = c.conn.ends[index]
    return Entry(c.item(f"end{index} seat"), joint.seated,
                 f"end {index}'s plug {end.plug} in {read.host_key} {read.host.name} "
                 f"({end.host}): {joint.detail}")


def _layout_ends(c: _Conn) -> list[int]:
    """The ends that plug into a mount on a layout: a self end is attached, not seated."""
    return [i for i, e in enumerate(c.conn.ends) if not e.host.is_self]


def _cap_entry(c: _Conn, tris: np.ndarray) -> Entry:
    span = length(tris)
    return Entry(c.item("cap"), span <= c.conn.cap,
                 f"the cable {c.conn.part} is {span:.4f} long, over its cap {c.conn.cap:g}")


def _entries(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[Entry]:
    for c in _connectors(ctx, only):
        for index in (0, 1):
            if only is None or c.item(f"end{index}") in only:
                entry = _end_entry(ctx, c, index)
                if isinstance(entry, Entry):
                    yield entry
            if only is None or c.item(f"end{index} cable") in only:
                entry = _cable_end_entry(ctx, c, index)
                if isinstance(entry, Entry):
                    yield entry
        for index in _layout_ends(c):
            if only is None or c.item(f"end{index} seat") in only:
                seated = _seat_entry(ctx, c, index)
                if isinstance(seated, Entry):
                    yield seated
        tris = _cable(ctx, c)
        if tris is not None and (only is None or c.item("cap") in only):
            yield _cap_entry(c, tris)


# --- plants -----------------------------------------------------------------------------------


def _axes() -> list[np.ndarray]:
    """Each axis, both ways: of a pair, one leaves a displaced point the full move away from
    where it sat, whichever way it sat."""
    return [s * np.eye(3)[i] for i in range(3) for s in (1.0, -1.0)]


def _bar(tris: np.ndarray, span: float) -> np.ndarray:
    """One thin triangle `span` long through the middle of the part: a cable of no length, given
    a length."""
    middle = tris.reshape(-1, 3).mean(axis=0)
    long, thin = np.array([span / 2.0, 0.0, 0.0]), np.array([0.0, span * 1e-3, 0.0])
    return (middle + np.array([-long, long, thin]))[None]


def _with_cable(ctx: SceneCtx, c: _Conn, tris: np.ndarray) -> SceneCtx:
    geom = ctx.pieces[c.key]
    return dataclasses.replace(ctx, pieces={
        **ctx.pieces, c.key: PieceGeom(geom.name, {**geom.parts, c.conn.part: tris})})


def _end_units(ctx: SceneCtx, c: _Conn, index: int) -> Iterator[PlantUnit]:
    end = c.conn.ends[index]
    point = np.asarray(end.point, dtype=float)

    def moved(direction: np.ndarray) -> SceneCtx:
        there = tuple(float(v) for v in point + CALIBRATION_FAIL * c.limit * direction)
        ends = tuple(dataclasses.replace(e, point=there) if i == index else e
                     for i, e in enumerate(c.conn.ends))
        return _with_connector(ctx, c, dataclasses.replace(c.conn, ends=ends))

    if c.limit is None:
        return  # no contact length: the coverage line says so
    yield unit(c.item(f"end{index}"), [("moved", moved(d)) for d in _axes()])
    framed = _cable_in_frame(ctx, c, end, index)
    if isinstance(framed, str):
        return  # no geometry, or no one placement to read it through: the coverage line says so
    in_frame, placement = framed
    far = float(np.linalg.norm(in_frame.reshape(-1, 3) - point, axis=1).max())
    plug = _plug_in_frame(ctx, c, end)
    if plug is not None:  # the cable must also clear the plug: past the plug's reach from the point
        far += float(np.linalg.norm(plug.reshape(-1, 3) - point, axis=1).max())
    clear = far + CALIBRATION_FAIL * c.limit

    def cleared(direction: np.ndarray) -> SceneCtx:
        away = in_frame + direction * clear
        return _with_cable(ctx, c, away if placement is None else placement.to_piece(away))

    yield unit(c.item(f"end{index} cable"), [("moved clear", cleared(d)) for d in _axes()])


def _moved_plug(ctx: SceneCtx, c: _Conn, read: _Seat, by: np.ndarray) -> SceneCtx:
    """The scene with the plug's parts (not the cable) moved `by` in the layout's frame."""
    delta = read.placement.to_piece(np.asarray(read.placement.position) + by)
    geom = ctx.pieces[c.key]
    parts = {**geom.parts, **{n: geom.parts[n] + delta for n in read.members}}
    return dataclasses.replace(ctx, pieces={**ctx.pieces, c.key: PieceGeom(geom.name, parts)})


def _sides(read: _Seat) -> list[np.ndarray]:
    """The layout's axes at right angles to the mount's axis, as unit vectors, each both ways
    (the receiver's plants move a guest along the same sides)."""
    axis = read.out / np.linalg.norm(read.out)
    sides = []
    for e in np.eye(3):
        flat = e - (e @ axis) * axis
        if np.linalg.norm(flat) > 0.5:
            sides.append(flat / np.linalg.norm(flat))
    return [s * sign for s in sides for sign in (1.0, -1.0)]


def _seat_unit(ctx: SceneCtx, c: _Conn, index: int) -> PlantUnit | None:
    """The unit for end `index`'s seat, from the manifest and the geometry alone, the moves most
    likely to fault the seat first: the plug moved sideways, along each side of the mount's axis
    both ways, by twice the width of its thinnest part there (a blade leaves its bore for the face
    beside it) and then by twice the plug's extent along that side plus the limit (clear of the
    joint's point, which lies within the limit of the plug); then along the
    axis, off the host and into it. None when the seat cannot be read."""
    read = _seat_read(ctx, c, index)
    if isinstance(read, str) or c.limit is None:
        return None
    little = CALIBRATION_FAIL * c.limit
    candidates = []
    for d in _sides(read):
        width = min(float(np.ptp(p.tris.reshape(-1, 3) @ d)) for p in read.parts)
        by = d * CALIBRATION_FAIL * width
        candidates.append(("moved a peg's width aside", _moved_plug(ctx, c, read, by)))
    corners = np.concatenate([p.tris.reshape(-1, 3) for p in read.parts])
    candidates += [("moved out of reach", _moved_plug(ctx, c, read, d * CALIBRATION_FAIL * (
        float(np.ptp(corners @ d)) + c.limit))) for d in _sides(read)]
    axis = read.out / np.linalg.norm(read.out)
    candidates += [("lifted off the host", _moved_plug(ctx, c, read, axis * little)),
                   ("pushed into the host", _moved_plug(ctx, c, read, -axis * little))]
    return unit(c.item(f"end{index} seat"), candidates)


def _cap_unit(ctx: SceneCtx, c: _Conn, tris: np.ndarray) -> PlantUnit:
    to = c.conn.cap * (1.0 + CALIBRATION_FAIL)
    candidates = []
    if (longer := _stretched(tris, to)) is not None:
        candidates.append(("stretched", _with_cable(ctx, c, longer)))
    return unit(c.item("cap"), [*candidates, ("a bar", _with_cable(ctx, c, _bar(tris, to)))])


def _units(ctx: SceneCtx) -> Iterator[PlantUnit]:
    for c in _connectors(ctx):
        for index in (0, 1):
            yield from _end_units(ctx, c, index)
        for index in _layout_ends(c):
            if (seat := _seat_unit(ctx, c, index)) is not None:
                yield seat
        tris = _cable(ctx, c)
        if tris is not None:
            yield _cap_unit(ctx, c, tris)


def _findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    return [Finding(NAME, e.item, e.detail) for e in _entries(ctx, only) if not e.ok]


class Connectors:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a pack check: connectors of any piece
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return list(_units(ctx))

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _findings(ctx, only)

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The cable part of each connector that has triangles in the scene, and the plug parts
        and host part of each end whose seat is judged."""
        out = {part_key(c.pack, c.piece_name, c.conn.part)
               for c in _connectors(ctx) if _cable(ctx, c) is not None}
        for c in _connectors(ctx):
            for index in _layout_ends(c):
                read = _seat_read(ctx, c, index)
                if not isinstance(read, str) and c.limit is not None:
                    out |= {part_key(c.pack, c.piece_name, n) for n in read.members}
                    out.add(f"{read.host_key} {read.host.name}")
        return out

    def coverage(self, ctx: SceneCtx) -> list[str]:
        conns = list(_connectors(ctx))
        if not conns:
            return [f"{NAME}: no piece has a connector: nothing to measure"]
        lines: list[str] = [
            Unmeasured(f"{NAME}: skipped (no geometry): {c.key} {c.conn.id} cap")
            for c in conns if _cable(ctx, c) is None]
        for c in conns:
            for index in (0, 1):
                end_entry = _end_entry(ctx, c, index)
                if isinstance(end_entry, str):
                    lines.append(Unmeasured(f"{NAME}: skipped ({end_entry}): "
                                            f"{c.item(f'end{index}')}"))
                entry = _cable_end_entry(ctx, c, index)
                if isinstance(entry, str):
                    lines.append(Unmeasured(
                        f"{NAME}: skipped ({entry}): {c.item(f'end{index} cable')}"))
            for index in _layout_ends(c):
                seated = _seat_entry(ctx, c, index)
                if isinstance(seated, str):
                    lines.append(Unmeasured(
                        f"{NAME}: skipped ({seated}): {c.item(f'end{index} seat')}"))
        return lines
