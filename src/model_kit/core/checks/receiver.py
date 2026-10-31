"""Check receiver, receiver and ports: a guest placed on a host mount touches the mount, the mount
takes that piece, and a power or signal mount takes only a connector end.

For each placement of a layout that has `on` (a mount of another placement of that layout), three
items, each named `<layout> <placement id> <rule>`:

- **touch.** The guest's `takes` part (the part the mount names) seats on the host part the
  mount names (`Mount.part`), judged by the joint helper (`joints.seat`): the mount's world point
  lies within the guest piece's contact length (`sight.contact_length`) of both parts, and the two
  are in contact. The point is the mount's `point` through the host placement; the parts are read
  through their placements. A guest that sits where the mount is not, on a host that has moved,
  or on another part of the host, fails. A mount is a rest here (a port takes only a connector
  end), so the joint needs no hole round the guest.
- **fits.** A host mount with `fits` takes only the pieces it names.
- **port.** A mount of kind `power` or `signal` takes a connector end, not a placed piece
  (`connectors` measures the end). A placement on one fails.

A placement on a mount its host does not publish raises, naming both: a lookup never falls back
to a default point. A placement that cannot be read as a joint gets no touch item and is
`Unmeasured`, so the run is `partial`: its mount names no host part ("no host part named"), the
guest has no geometry for its `takes` part, or the host has none for the part the mount names
("no geometry"), or the guest piece declares no `view` ("no contact length").

The plants are one unit per item, built from the layout and the manifest alone (never from the
gap the check measures): the guest moved `CALIBRATION_FAIL` times the contact length along the
mount's
axis (turned by the host placement), forward and back, then along each side of the host's frame,
the gate trying each in turn; the host mount's `fits` set to a piece the guest is not; the host
mount made a `power` port. The gate stops the run when the check catches none of a unit's
candidates. A scene where no placement sits on a mount has nothing to receive, and says so in a
plain line.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks import joints
from model_kit.core.checks.entries import Entry, unit
from model_kit.core.checks.pieces import free_name, part_mesh, with_piece
from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION
from model_kit.core.manifest import Layout, LayoutPlacement, Mount, Piece, Placement
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx, part_key, piece_key
from model_kit.core.sight import contact_length
from model_kit.core.tolerances import CALIBRATION_FAIL

NAME = "receiver"
PRINCIPLE = joints.PRINCIPLE
SOURCE = joints.SOURCE
PORT_KINDS = joints.PORT_KINDS  # the mount kinds that take a connector end


@dataclass(frozen=True)
class _Seat:
    """A placement on a mount, with everything its three rules read."""

    layout: Layout
    guest: LayoutPlacement
    host: LayoutPlacement
    mount: Mount
    host_piece: Piece
    limit: float | None  # the guest piece's contact length; None: it declares no view

    def item(self, rule: str) -> str:
        return f"{self.layout.name} {self.guest.id} {rule}"

    @property
    def guest_key(self) -> str:
        return piece_key(self.guest.pack, self.guest.piece)


def _placement(p: LayoutPlacement) -> Placement:
    return Placement(p.id, p.piece, p.pack, p.position, p.quaternion)


def _seats(ctx: SceneCtx, only: Collection[str] | None = None) -> Iterator[_Seat]:
    """Each placement of each layout that sits on a mount. A dangling `on` raises."""
    for layout in ctx.layouts.values():
        for pid, guest in layout.placements.items():
            if guest.on is None:
                continue
            if only is not None and not any(i.startswith(f"{layout.name} {pid} ") for i in only):
                continue
            where = f"layout {layout.name!r} placement {pid!r}"
            host = layout.placements.get(guest.on.piece)
            if host is None:
                raise ManifestError(f"{where}: on {guest.on} names no placement "
                                    f"{guest.on.piece!r} of the layout")
            host_pack = ctx.packs.get(host.pack)
            host_piece = None if host_pack is None else host_pack.pieces.get(host.piece)
            if host_piece is None:
                raise ManifestError(f"{where}: its host placement {host.id!r} is of "
                                    f"{host.pack}:{host.piece}, which the scene lacks")
            mount = next((m for m in host_piece.mounts if m.id == guest.on.id), None)
            if mount is None:
                raise ManifestError(
                    f"{where}: on {guest.on}: piece {host.pack}:{host.piece} publishes no mount "
                    f"{guest.on.id!r} (it has {[m.id for m in host_piece.mounts]})")
            guest_pack = ctx.packs.get(guest.pack)
            if guest_pack is None or guest.piece not in guest_pack.pieces:
                raise ManifestError(f"{where}: it is {guest.pack}:{guest.piece}, which the "
                                    "scene lacks")
            yield _Seat(layout, guest, host, mount, host_piece,
                        contact_length(guest_pack.pieces[guest.piece], guest_pack.units_per_metre))


def _takes(ctx: SceneCtx, seat: _Seat) -> np.ndarray | None:
    """The guest's `takes` part, or None when the scene has no triangles for it."""
    geom = ctx.pieces.get(seat.guest_key)
    tris = None if geom is None else geom.parts.get(seat.mount.takes)
    return None if tris is None or len(tris) == 0 else np.asarray(tris, dtype=float)


def _host_part(ctx: SceneCtx, seat: _Seat) -> np.ndarray | str:
    """The host part the mount names, or why the joint cannot be read: no part named, or the
    scene has no triangles for it."""
    if seat.mount.part is None:
        return "no host part named"
    geom = ctx.pieces.get(piece_key(seat.host.pack, seat.host.piece))
    tris = None if geom is None else geom.parts.get(seat.mount.part)
    return "no geometry" if tris is None or len(tris) == 0 else np.asarray(tris, dtype=float)


def _unread(ctx: SceneCtx, seat: _Seat) -> str | None:
    """Why the touch rule cannot be read for this seat, or None when it can."""
    if seat.limit is None:
        return "no contact length: the guest piece declares no view"
    if _takes(ctx, seat) is None:
        return "no geometry"
    host = _host_part(ctx, seat)
    return host if isinstance(host, str) else None


def _point(seat: _Seat) -> np.ndarray:
    """The host mount's point in the layout's frame."""
    return _placement(seat.host).to_world(seat.mount.point)


def _joint(ctx: SceneCtx, seat: _Seat) -> joints.JointResult | None:
    """The joint helper's answer for the guest's `takes` part on the host's part, in the layout's
    frame; None when the seat cannot be read (`_unread`). A mount here is a rest: no hole."""
    if _unread(ctx, seat) is not None:
        return None
    guest = part_mesh(ctx.packs[seat.guest.pack].pieces[seat.guest.piece], seat.mount.takes,
                      _placement(seat.guest).to_world(_takes(ctx, seat)))
    host = part_mesh(seat.host_piece, seat.mount.part,
                     _placement(seat.host).to_world(_host_part(ctx, seat)))
    return joints.seat(ctx, guest=guest, host=host, point=_point(seat), locate=seat.limit,
                       bore_axis=None)


def _with_position(ctx: SceneCtx, seat: _Seat, position) -> SceneCtx:
    layout = ctx.layouts[seat.layout.name]
    moved = dataclasses.replace(seat.guest, position=tuple(float(v) for v in position))
    layout = dataclasses.replace(layout, placements={**layout.placements, seat.guest.id: moved})
    return dataclasses.replace(ctx, layouts={**ctx.layouts, layout.name: layout})


def _with_mount(ctx: SceneCtx, seat: _Seat, **fields) -> SceneCtx:
    mounts = tuple(dataclasses.replace(m, **fields) if m.id == seat.mount.id else m
                   for m in seat.host_piece.mounts)
    piece = dataclasses.replace(seat.host_piece, mounts=mounts)
    return with_piece(ctx, seat.host.pack, seat.host.piece, piece)


def _shifts(seat: _Seat) -> Iterator[np.ndarray]:
    """The directions to move a guest in: along the mount's axis (forward, then back), then the
    other two sides of the host's frame."""
    host = _placement(seat.host)
    axis = np.asarray(seat.mount.axis, dtype=float)
    axes = [axis / max(float(np.linalg.norm(axis)), NO_DIRECTION)]
    axes += [np.eye(3)[i] for i in range(3)]
    for a in axes:
        world = host.to_world(a) - host.to_world(np.zeros(3))
        for sign in (1.0, -1.0):
            yield sign * world


def _entries(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[Entry]:
    for seat in _seats(ctx, only):
        touch, fits, port = (seat.item(r) for r in ("touch", "fits", "port"))
        if only is None or touch in only:
            joint = _joint(ctx, seat)
            if joint is not None:
                yield Entry(touch, joint.seated,
                            f"{seat.guest_key} {seat.mount.takes} on {seat.host.pack}:"
                            f"{seat.host.piece}.{seat.mount.id} ({seat.mount.part}): "
                            f"{joint.detail}")
        if only is None or fits in only:
            allowed = seat.mount.fits
            names = "" if allowed is None else f" (it takes {list(allowed)})"
            yield Entry(fits, allowed is None or seat.guest.piece in allowed,
                        f"mount {seat.mount.id} of {seat.host.pack}:{seat.host.piece} does not "
                        f"take {seat.guest.piece!r}{names}")
        if only is None or port in only:
            yield Entry(port, seat.mount.kind not in PORT_KINDS,
                        f"mount {seat.mount.id} is a {seat.mount.kind} port: it takes a "
                        f"connector end, not the placed piece {seat.guest_key}")


def _units(ctx: SceneCtx) -> Iterator[PlantUnit]:
    """A unit for each rule of each placement on a mount, from the layout and the manifest alone.
    The touch rule exists only where the joint can be read: the guest has triangles for the part
    the mount takes, and the host for the part the mount names."""
    for seat in _seats(ctx):
        if _unread(ctx, seat) is None:
            here = np.asarray(seat.guest.position)
            yield unit(seat.item("touch"), [
                ("moved", _with_position(ctx, seat, here + CALIBRATION_FAIL * seat.limit * d))
                for d in _shifts(seat)])
        other = free_name(seat.mount.fits or (), seat.guest.piece)
        yield unit(seat.item("fits"), [("refused", _with_mount(ctx, seat, fits=(other,)))])
        yield unit(seat.item("port"), [("port", _with_mount(ctx, seat, kind=PORT_KINDS[0]))])


def _findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    return [Finding(NAME, e.item, e.detail) for e in _entries(ctx, only) if not e.ok]


class Receiver:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a layout check: it measures no piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return list(_units(ctx))

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _findings(ctx, only)

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The `takes` part of each guest on a mount, and the host part the mount names, that
        the touch rule reads."""
        out: set[str] = set()
        for s in _seats(ctx):
            if _unread(ctx, s) is None:
                out |= {part_key(s.guest.pack, s.guest.piece, s.mount.takes),
                        part_key(s.host.pack, s.host.piece, s.mount.part)}
        return out

    def coverage(self, ctx: SceneCtx) -> list[str]:
        seats = list(_seats(ctx))
        if not seats:
            return [f"{NAME}: no placement sits on a mount: nothing to receive"]
        return [Unmeasured(f"{NAME}: skipped ({why}): {s.layout.name} {s.guest.id}")
                for s in seats if (why := _unread(ctx, s)) is not None]
