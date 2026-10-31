"""Check markings-flush: a solid part that is flush with its host, as far as the eye can tell, is a
marking built as an object.

This is rule 3 of the markings check, the safety net for a listed row that says `solid` by
default. Rules 1 and 2 read the manifest. This one reads the mesh and the piece's `view`, and
judges by eye (a design call of this kit): a solid part counts as flush with its `on` host, and may
be a marking, only when its largest stand-off from the host is under what the eye resolves
(`EYE_RESOLUTION_ARCMIN`) at the piece's nearest viewing distance (`Piece.view.nearest_m`, in
real metres through `Pack.units_per_metre`). So a gap or a seam too small to see at that
distance is not judged, and a 1.5 mm float on an instrument held at about 1 m still is.

The stand-off of a part is the largest outward distance of its vertices from the host's surface.
A closed host tells inside from outside by ray parity; an open host (a leaf card, a sheet) has no
inside, so the distance is unsigned and a part far behind it is far from it, never flush. A part
whose every vertex is buried deeper than the limit inside a closed host is hidden, not a
marking: it shows no colour change, and the crossing checks own how deep a join may be. So a part
is flagged only when its outermost vertex lies within the limit of the host's surface, either side.

A part whose outermost faces all look toward a side the piece's `view` lists as `hidden` is not
judged, as the other visible rules skip hidden faces. Parts of a row that is not `solid` (a print,
a paint film, an inlay, a screen) are already marked and left out, as are decals, printed roles and parts with no `on` host. A piece with such a part
and no `view` is `Unmeasured`; so is a part whose host has no triangles.

The item is `<pack>:<piece> <part> flush with <host>: a marking?`. A real flush part (a
flush-mounted plate) is answered by its row's `finish` or an allow line with a reason. The plant
is one unit per piece with a view, built from the geometry alone (never from the stand-off
verdict): the smallest solid part with a host, pushed straight toward its host, as a rigid body,
until its outermost vertex stands `CALIBRATION_PASS` times the limit off it, and then pressed onto
the host instead, each vertex pulled toward it by one share (a part that no rigid push brings
flush). A candidate whose outermost faces look only toward a hidden side is dropped, as the check
would not judge it. The gate skips the unit when the part is flush already (the check flags it,
and a part that sits on its host has no push or press to make). A part buried deeper than the limit
is pushed out to the limit, the way a proud part is pushed in. The unit says `not applicable` only
when every part with a host looks only toward a hidden side, so the check judges none of them; a
visible part the plants cannot reach has a unit with no candidates, and the gate stops the run.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import NamedTuple

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.detail import faces_on_sides
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.geom import NO_DIRECTION, SurfaceIndex, is_closed
from model_kit.core.manifest import Part
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.sight import seen_floor
from model_kit.core.tolerances import CALIBRATION_PASS

NAME = "markings-flush"
SOLID = "solid"
PRINCIPLE = ("A solid part that stands off its host by less than the eye resolves at the nearest "
             "distance it is seen from shows only as a change of colour on the surface, so it is "
             "a print, a paint film or an inlay and belongs on the host as a decal.")
SOURCE = (
    "Judged by eye, a design call of this kit: normal sight resolves about "
    "1 minute of arc (EyeWiki, 'Visual Acuity Testing in Adults', "
    "https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, secondary), about 0.29 mm a metre. "
    "A printed or painted film is a few microns to under 0.1 mm (4 microns of "
    "dried pad-print ink, Inkcups ICN-B150 manual, https://tagless.inkcups.com/wp-content/"
    "uploads/2017/08/ICN-B150-Manual-Compressed.pdf; 25 to 30 microns of paint on wood, Nippon "
    "Paint PU Timbercoat data sheet, https://www.nipponpaint.com.my/wp-content/uploads/2024/11/"
    "1-pack-pu-timbercoat.pdf).")


class _Host(NamedTuple):
    index: SurfaceIndex
    closed: bool


class _Reach(NamedTuple):
    """Each distinct vertex of a part, the nearest point on the host, and the signed distance
    (outward positive; unsigned for an open host)."""

    points: np.ndarray  # (K, 3)
    inverse: np.ndarray  # (3N,) the vertex of each triangle corner
    nearest: np.ndarray  # (K, 3)
    signed: np.ndarray  # (K,)


def _host_of(host: PartMesh) -> _Host:
    return _Host(SurfaceIndex(host.tris), is_closed(host.tris))


def _reach(part: np.ndarray, host: _Host) -> _Reach:
    points, inverse = np.unique(part.reshape(-1, 3), axis=0, return_inverse=True)
    near = host.index.closest(points)
    signed = near.dist
    if host.closed:
        signed = np.where(host.index.inside(points), -near.dist, near.dist)
    return _Reach(points, inverse.reshape(-1), near.point, signed)


def standoff(part: np.ndarray, host: _Host) -> float:
    """The largest outward distance of the part's vertices from the host's surface."""
    return float(_reach(part, host).signed.max())


def _is_flush(reach: float, limit: float) -> bool:
    """Whether the outermost vertex is within the limit of the surface. Deeper inside than that
    is a part the eye never sees, and too far out is a part."""
    return -limit <= reach <= limit


def _on_hidden_sides(part: np.ndarray, reach: _Reach, limit: float, hidden: tuple[str, ...]
                     ) -> bool:
    """Whether the part's outermost faces all look toward a side the view lists as never seen.
    The outermost faces are those every vertex of which is within the limit of the outermost
    vertex: the surface a viewer would see, not the walls that run down into the host."""
    if not hidden:
        return False
    corner = reach.signed[reach.inverse].reshape(-1, 3)
    outer = corner.min(axis=1) >= float(reach.signed.max()) - limit
    return bool(outer.any() and faces_on_sides(part, hidden)[outer].all())


def _marked(view: PieceView, record: Part) -> bool:
    """Whether the part's listed row already says it is a marking (its `finish` is not solid).
    A pack whose rows carry no `finish` has no marked row."""
    row = next((r for r in view.piece.listed if r.id == record.listed), None)
    return getattr(row, "finish", SOLID) != SOLID


class _Judge:
    """One piece's candidates and hosts, read once."""

    def __init__(self, view: PieceView) -> None:
        self.view = view
        self.mesh = {p.name: p for p in view.parts}
        self._hosts: dict[str, _Host] = {}

    def host(self, name: str) -> _Host | None:
        """The host's index, or None when the host has no triangles in the scene."""
        mesh = self.mesh.get(name)
        if mesh is None or not len(mesh.tris):
            return None
        if name not in self._hosts:
            self._hosts[name] = _host_of(mesh)
        return self._hosts[name]

    def candidates(self) -> list[tuple[Part, PartMesh]]:
        """Solid parts of solid rows, not decals, with triangles and an `on` part."""
        out = []
        for record in self.view.piece.parts:
            mesh = self.mesh.get(record.name)
            if (mesh is None or not mesh.solid or not len(mesh.tris) or record.on is None
                    or record.on == record.name or record.printed
                    or _marked(self.view, record)):
                continue
            out.append((record, mesh))
        return out


def _flush_item(view: PieceView, record: Part) -> str:
    return f"{item_of(view, record.name)} flush with {record.on}: a marking?"


def _findings(ctx: SceneCtx, view: PieceView, only: Collection[str] | None):
    if view.piece.view is None:
        return
    limit = seen_floor(view.piece, ctx.packs[view.pack].units_per_metre)
    judge = _Judge(view)
    for record, mesh in judge.candidates():
        item = _flush_item(view, record)
        host = judge.host(record.on)
        if host is None or (only is not None and item not in only):
            continue
        reached = _reach(mesh.tris, host)
        reach = float(reached.signed.max())
        if _is_flush(reach, limit) and not _on_hidden_sides(
                mesh.tris, reached, limit, view.piece.view.hidden):
            distance = view.piece.view.nearest_m
            yield Finding(NAME, item, (
                f"stands off {record.on} by {max(reach, 0.0):.5f} at most (limit {limit:.5f}, "
                f"what the eye resolves at {distance:g} m), so it shows only as a colour change: "
                "a print, paint or inlay? Give its row that finish, or allow it with a reason"))


# --- plants -----------------------------------------------------------------------------------


def _pushed(part: np.ndarray, host: _Host, limit: float) -> np.ndarray | None:
    """A copy of the part moved as one body, straight along the line to its host, until its
    outermost vertex stands `CALIBRATION_PASS` times the limit off the surface: in when it stands
    proud, out when it is buried. None when it has no direction to move in."""
    reach = _reach(part, host)
    top = int(reach.signed.argmax())
    gap = float(reach.signed[top])
    toward = reach.nearest[top] - reach.points[top]
    size = float(np.linalg.norm(toward))
    if size < NO_DIRECTION:
        return None
    outward = -toward / size if gap > 0.0 else toward / size  # away from the surface
    return part + outward * (CALIBRATION_PASS * limit - gap)


def _pressed(part: np.ndarray, host: _Host, limit: float) -> np.ndarray | None:
    """A copy of the part with every vertex that stands outside the host pulled toward its
    nearest point on the host, by one share, until the outermost stands `CALIBRATION_PASS` times
    the limit off. It flattens the part onto the host, for a part that no rigid push can bring
    flush (a long limb beside a curved trunk). None when no vertex stands outside the host."""
    reach = _reach(part, host)
    top = float(reach.signed.max())
    if not top > 0.0:
        return None
    share = CALIBRATION_PASS * limit / top
    out = np.where((reach.signed > 0.0)[:, None],
                   reach.nearest + (reach.points - reach.nearest) * share, reach.points)
    return out[reach.inverse].reshape(part.shape)


def _unit(ctx: SceneCtx, view: PieceView, judge: _Judge, limit: float) -> PlantUnit | None:
    """The piece's plant unit: its smallest candidate whose host has triangles, pushed into the
    host as a rigid body and pressed flat onto it. A candidate whose outermost faces look only
    toward a hidden side is not judged by the check, so it is not planted. Read from the
    geometry alone; no stand-off is compared with the limit.

    A candidate that sits on its host already has no push or press. If the eye sees it, the real
    check flags it and the gate skips its unit, so it is kept as a unit with no candidates (a
    later candidate that can be planted takes its place). The same holds for a seen part whose
    pushed and pressed copies all look toward a hidden side: the check judges it, so the unit has
    no candidate and the gate stops the run. Only a part that is itself unseen is excused. None
    when no candidate has a host: the coverage line says so."""
    by_size = sorted(judge.candidates(),
                     key=lambda row: float(np.linalg.norm(row[1].hi - row[1].lo)))
    geometry = {p.name: p.tris for p in view.parts}
    hidden = view.piece.view.hidden

    def unseen(part: np.ndarray, host: _Host) -> bool:
        return _on_hidden_sides(part, _reach(part, host), limit, hidden)

    on_host: PlantUnit | None = None  # a visible part that no plant reaches
    excused = False  # a part with a host that the check never judges
    for record, mesh in by_size:
        host = judge.host(record.on)
        if host is None:
            continue
        item = _flush_item(view, record)
        moves = [(how, moved) for how, make in (("pushed", _pushed), ("pressed", _pressed))
                 if (moved := make(mesh.tris, host, limit)) is not None]
        planted = [Plant(f"{how} flush: {item}", item,
                         pieces.with_geometry(ctx, view, geometry | {record.name: moved}))
                   for how, moved in moves if not unseen(moved, host)]
        if planted:
            return PlantUnit(item, tuple(planted))
        if unseen(mesh.tris, host):
            excused = True  # the check never judges this part
        elif on_host is None:
            # judged, but no plant reaches it (none to make, or every copy looks at a hidden side)
            on_host = PlantUnit(item, ())
    if on_host is not None:
        return on_host
    if excused:
        return PlantUnit(view.key, (), not_applicable="every part with a host looks only toward "
                                                      "a side the view hides")
    return None


class MarkingsFlush:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    principle = PRINCIPLE
    source = SOURCE

    def _views(self, ctx: SceneCtx, only: Collection[str] | None = None):
        return pieces.measured_pieces(ctx, self.kinds, only)

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every candidate part whose host has triangles, in a piece with a `view`."""
        out = set()
        for view in self._views(ctx):
            if view.piece.view is None:
                continue
            judge = _Judge(view)
            out |= {item_of(view, r.name) for r, _ in judge.candidates()
                    if judge.host(r.on) is not None}
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return sorted((f for view in self._views(ctx, only) for f in _findings(ctx, view, only)),
                      key=lambda f: f.item)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = list(pieces.coverage(ctx, NAME, self.kinds))
        for view in self._views(ctx):
            judge = _Judge(view)
            judged = judge.candidates()
            if not judged:
                if view.parts:
                    lines.append(f"{NAME}: not applicable (no solid part with an on host that "
                                 f"is not already a marking): {view.key}")
                continue
            if view.piece.view is None:
                lines.append(Unmeasured(f"{NAME}: skipped (the piece declares no view, so there "
                                        f"is no distance to judge flush by): {view.key}"))
                continue
            lines += [Unmeasured(f"{NAME}: skipped (host {r.on!r} has no geometry): "
                                 f"{item_of(view, r.name)}")
                      for r, _ in judged if judge.host(r.on) is None]
            if skipped := len(view.parts) - len(judged):
                lines.append(f"{NAME}: not applicable (a decal, a marked row, a part with no "
                             f"on host or an empty part) to {skipped} of {len(view.parts)} "
                             f"parts: {view.key}")
        return lines

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units = []
        for view in self._views(ctx):
            judge = _Judge(view)
            if view.piece.view is None or not judge.candidates():
                continue
            limit = seen_floor(view.piece, ctx.packs[view.pack].units_per_metre)
            if (unit := _unit(ctx, view, judge, limit)) is not None:
                units.append(unit)
        return units
