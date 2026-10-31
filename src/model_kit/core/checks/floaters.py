"""Check floaters: a part must reach something that stands.

Two rules, over each piece of a kind in `kinds` (assembly). A part with `on` must touch its `on`
part: the surfaces come within the piece's contact length, or cross. A part with `on: null` must
rest on the piece's floor, the lowest y of any part, within that same length, or be an anchor: named
in the piece's `measured.hosts`, a part that stands on
something outside the piece. A `hosts` name that is no part of the piece raises, and so does an
`on` part with no geometry, and so does an `on` cycle (a part on itself, or parts on each
other), since no chain of touches would then reach the floor.

The item is `<pack>:<piece> <part>`.

The contact length is the piece's seen floor (`sight.contact_length`): one arcminute at
`view.nearest_m`, so a gap a viewer can tell apart is a float and a smaller one touches. A gap
equal to it touches too, the float32 rounding of a built decal's coordinates (`COMPARE_TOL`)
aside. A side the view lists as `hidden` changes nothing here: a float is a physical fault, and
an unseen part still falls. A piece with no `view` has no contact length: it is `Unmeasured` and
judged on nothing.

The plant is one unit per piece, built from the geometry and the manifest only (never from
the check's own verdict): it lifts the piece's smallest part that a lift can free, in each of
six directions in turn, by `CALIBRATION_FAIL` times the contact length, away from the part it is
on (or up, off the floor). A part that a lift can free is one
with an `on` part, or one with no `on` that is not the only part at the lowest y. An anchor is
never lifted. The gate skips the unit when the real run already reports that part. A piece
of two or more parts with no such part (every part an anchor, say) says so in a plain
`not applicable` line; a piece of one part needs no plant and says so.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import NamedTuple

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.errors import ManifestError
from model_kit.core.geom import SurfaceIndex, surface_gap
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import CALIBRATION_FAIL, COMPARE_TOL

NAME = "floaters"
Y = 1  # the up axis in the piece frame (glTF)


def touch_of(view: PieceView) -> float:
    """The piece's contact length. Only a piece with a `view` reaches here (`contact_pieces`)."""
    if view.contact is None:
        raise ValueError(f"{view.key}: no contact length without a view")
    return view.contact


def _check_chains(view: PieceView) -> None:
    """Every `on` chain must end at a part with no `on`. A part on itself, or parts on each
    other, would pass as touching and never reach the floor, so a cycle raises."""
    on = {p.name: p.on for p in view.parts}
    for start in on:
        seen = [start]
        while (nxt := on.get(seen[-1])) is not None:
            if nxt in seen:
                loop = " -> ".join([*seen[seen.index(nxt):], nxt])
                raise ManifestError(f"{view.key}: `on` makes a cycle: {loop}")
            seen.append(nxt)


def _hosts(view: PieceView) -> frozenset[str]:
    _check_chains(view)
    hosts = view.piece.measured.get("hosts") or ()
    stray = sorted(set(hosts) - {p.name for p in view.parts})
    if stray:
        raise ManifestError(f"{view.key}: measured.hosts names {stray}, which are no parts of "
                            f"the piece; it has {sorted(p.name for p in view.parts)}")
    return frozenset(hosts)


def _floor(view: PieceView) -> float:
    return min(float(p.lo[Y]) for p in view.parts if len(p.tris))


def _gap(view: PieceView, part: PartMesh, host: PartMesh, touch: float) -> float:
    """The distance between the part and its host. The search stops at `touch` first, which is
    cheap and exact whenever the part touches. Only when no surface lies within `touch` does a
    wider search (the whole joint box) give the number for the finding."""
    a, b = SurfaceIndex(part.tris), SurfaceIndex(host.tris)
    gap = surface_gap(a, b, touch)
    if gap <= touch:
        return gap
    size = float(np.linalg.norm(np.maximum(part.hi, host.hi) - np.minimum(part.lo, host.lo)))
    return surface_gap(a, b, max(touch, size))


class _Float(NamedTuple):
    """Why a part floats and by how much: `size` is the gap to its `on` part or its height over the
    floor, None for a part with no triangles."""

    why: str
    size: float | None


def _problem(view: PieceView, part: PartMesh, floor: float, hosts: frozenset[str]
             ) -> _Float | None:
    """How this part floats, or None when it stands."""
    touch = touch_of(view)
    if len(part.tris) == 0:
        return _Float("has no triangles", None)
    if part.on is not None:
        host = next((p for p in view.parts if p.name == part.on), None)
        if host is None or len(host.tris) == 0:
            raise ManifestError(f"{view.key} {part.name}: on {part.on!r}, which has no geometry "
                                f"in the scene; the scene has {sorted(p.name for p in view.parts)}")
        # A decal stands off its host by exactly the contact length (`sight.decal_lift`), and the
        # build's float32 coordinates round that gap by up to `COMPARE_TOL`: a gap equal to the
        # touch touches. The allowance applies to every part with an `on` host, not only decals.
        gap = _gap(view, part, host, touch)
        if gap > touch + COMPARE_TOL:
            return _Float(f"{gap:.4f} from its `on` part {part.on} (limit {touch:.4f})", gap)
        return None
    if part.name in hosts:
        return None
    above = float(part.lo[Y]) - floor
    if above > touch:
        return _Float(f"rests on nothing: its lowest point is {above:.4f} above the piece's "
                      f"floor (limit {touch:.4f})", above)
    return None


def _findings(view: PieceView, only: Collection[str] | None):
    hosts, floor = _hosts(view), _floor(view)
    for part in view.parts:
        item = item_of(view, part.name)
        found = None if only is not None and item not in only else _problem(
            view, part, floor, hosts)
        if found is None:
            continue
        yield Finding(NAME, item, found.why)


# The directions a plant tries to lift a part: up first, then the other axes, both ways.
LIFTS = (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]),
         np.array([-1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0]))


def _free_to_lift(view: PieceView, part: PartMesh, hosts: frozenset[str]) -> bool:
    """Whether a lift can float the part: it is no anchor, and it either rests on an `on` part or
    shares the lowest y with another part. The only part at the floor stays the floor when
    lifted, so no lift floats it."""
    if len(part.tris) == 0 or part.name in hosts:
        return False
    if part.on is not None:
        return True
    return any(q is not part and len(q.tris) and float(q.lo[Y]) <= float(part.lo[Y])
               for q in view.parts)


def _unit(ctx: SceneCtx, view: PieceView) -> PlantUnit:
    """The piece's plant unit: its smallest part that a lift can float, lifted six ways. Built
    from the geometry and the manifest alone."""
    hosts = frozenset(view.piece.measured.get("hosts") or ())
    reach = touch_of(view)
    by_size = sorted((p for p in view.parts if _free_to_lift(view, p, hosts)),
                     key=lambda p: float(np.linalg.norm(p.hi - p.lo)))
    if not by_size:
        return PlantUnit(view.key, (), not_applicable="every part is an anchor or the only "
                                                      "part on the floor: no lift floats one")
    part = by_size[0]
    item = item_of(view, part.name)
    geometry = {p.name: p.tris for p in view.parts}
    return PlantUnit(item, tuple(
        Plant(f"lifted: {item}", item, pieces.with_geometry(
            ctx, view, geometry | {part.name: part.tris + direction * CALIBRATION_FAIL * reach}))
        for direction in LIFTS))


class Floaters:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [_unit(ctx, view) for view in pieces.contact_pieces(ctx, self.kinds)
                if len(view.parts) >= 2]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part with triangles of a measured piece. A part with none is reported as a
        finding but has no surface to judge."""
        return {item_of(view, p.name) for view in pieces.contact_pieces(ctx, self.kinds)
                for p in view.parts if len(p.tris)}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for view in pieces.contact_pieces(ctx, self.kinds, only)
                for f in _findings(view, only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        views = list(pieces.contact_pieces(ctx, self.kinds))
        lone = [f"{NAME}: one part, no plant needed: {v.key}" for v in views if len(v.parts) == 1]
        return (pieces.coverage(ctx, NAME, self.kinds) + pieces.unviewed(ctx, NAME, self.kinds)
                + lone)
