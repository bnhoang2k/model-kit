"""Check floaters: a part must reach something that stands.

Two rules, over each piece of a kind in `kinds` (assembly). A part with `on` must touch its `on`
part: the surfaces come within the piece's contact length, or cross. A part with `on: null` must
rest on the piece's floor, the lowest y of any part, within that same length, or be an anchor: named
in the piece's `measured.hosts`, a part that stands on
something outside the piece. A `hosts` name that is no part of the piece raises, and so does an
`on` part with no geometry, and so does an `on` cycle (a part on itself, or parts on each
other), since no chain of touches would then reach the floor.

A part with `rests_on` (a part that rests on two or more supports, a tabletop on its legs) must
touch each support it names, judged the same way as an `on` part. Its `on` is null, and the floor
rule does not apply to it: its supports carry it. A support with no geometry raises, and so does a
cycle through a `rests_on` name. Each support it misses is its own finding.

The item is `<pack>:<piece> <part>`, and for a part with `rests_on` one per missed support,
`<pack>:<piece> <part> off <support>`.

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
`not applicable` line; a piece of one part needs no plant and says so. A part with `rests_on` can
be lifted too: its item is then `<part> off <first support>`.

A piece with a part that has `rests_on` gets two more units, on its smallest such part, also built
from the geometry and the manifest alone. The tilt turns the part about the outer edge of the
support farthest from the one it shares least contact with, until it clears that support by
`CALIBRATION_FAIL` times the contact length: `<part> off <that support>`. The phantom names a part
it does not touch and does not rest on, even through other parts, as one more support:
`<part> off <that part>`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from typing import NamedTuple

import numpy as np

from model_kit.core import geom
from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.errors import ManifestError
from model_kit.core.geom import SurfaceIndex, surface_gap
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import CALIBRATION_FAIL, COMPARE_TOL, WELD_OF_PIECE_SIZE

NAME = "floaters"
Y = 1  # the up axis in the piece frame (glTF)


def touch_of(view: PieceView) -> float:
    """The piece's contact length. Only a piece with a `view` reaches here (`contact_pieces`)."""
    if view.contact is None:
        raise ValueError(f"{view.key}: no contact length without a view")
    return view.contact


def _check_chains(view: PieceView) -> None:
    """Every chain of `on` and `rests_on` names must end at a part with neither. A part on itself,
    or parts resting on each other, would pass as touching and never reach the floor, so a cycle
    raises. A loop of `on` edges alone says `on`; one through a `rests_on` edge says `rests_on`."""
    edges = {p.name: [(p.on, "on")] if p.on is not None else [(s, "rests_on") for s in p.rests_on]
             for p in view.parts}
    done: set[str] = set()

    for start, first in edges.items():
        if start in done:
            continue
        path, kinds = [start], []
        stack = [iter(first)]
        while stack:
            step = next(stack[-1], None)
            if step is None:
                done.add(path.pop())
                stack.pop()
                if kinds:
                    kinds.pop()
                continue
            nxt, kind = step
            if nxt in path:
                at = path.index(nxt)
                word = "on" if {*kinds[at:], kind} == {"on"} else "rests_on"
                loop = " -> ".join([*path[at:], nxt])
                raise ManifestError(f"{view.key}: `{word}` makes a cycle: {loop}")
            if nxt in edges and nxt not in done:
                path.append(nxt)
                kinds.append(kind)
                stack.append(iter(edges[nxt]))


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


def _problem(view: PieceView, part: PartMesh, floor: float, hosts: frozenset[str],
             support: str | None = None) -> _Float | None:
    """How this part floats, or None when it stands. With `support`, the gap to that one support
    of a part that rests on several."""
    touch = touch_of(view)
    if len(part.tris) == 0:
        return _Float("has no triangles", None)
    named = support or part.on
    if named is not None:
        host = next((p for p in view.parts if p.name == named), None)
        if host is None or len(host.tris) == 0:
            word = "rests on" if support else "on"
            raise ManifestError(f"{view.key} {part.name}: {word} {named!r}, which has no "
                                f"geometry in the scene; the scene has "
                                f"{sorted(p.name for p in view.parts)}")
        # A decal stands off its host by exactly the contact length (`sight.decal_lift`), and the
        # build's float32 coordinates round that gap by up to `COMPARE_TOL`: a gap equal to the
        # touch touches. The allowance applies to every part with an `on` host, not only decals.
        gap = _gap(view, part, host, touch)
        if gap > touch + COMPARE_TOL:
            if support:
                return _Float(f"{gap:.4f} from its support {support} (limit {touch:.4f})", gap)
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
        if only is not None and not any(i == item or i.startswith(f"{item} ") for i in only):
            continue
        if part.rests_on and len(part.tris):
            for support in part.rests_on:
                found = _problem(view, part, floor, hosts, support)
                found_item = f"{item} off {support}"
                if found is not None and (only is None or found_item in only):
                    yield Finding(NAME, found_item, found.why)
            continue
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
    if part.on is not None or part.rests_on:
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
    if part.rests_on:
        item = f"{item} off {part.rests_on[0]}"
    geometry = {p.name: p.tris for p in view.parts}
    return PlantUnit(item, tuple(
        Plant(f"lifted: {item}", item, pieces.with_geometry(
            ctx, view, geometry | {part.name: part.tris + direction * CALIBRATION_FAIL * reach}))
        for direction in LIFTS))


def _area(tris: np.ndarray) -> float:
    return float(0.5 * np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1).sum())


def _contact(part: PartMesh, support: PartMesh, touch: float) -> np.ndarray:
    """The support's faces that bear on the part, clipped to the part's box: `contact_faces` keeps
    whole triangles, and a support face that runs past the part would move the centroid and the
    pivot edge."""
    tris = support.tris[geom.contact_faces(part.tris, support.tris, touch)[1]]
    if len(tris) == 0:
        return tris
    return geom.clip_to_box(tris, (part.lo, part.hi))


def _tilted(part: PartMesh, target: np.ndarray, pivot: np.ndarray, touch: float,
            noise: float = 0.0) -> np.ndarray | None:
    """The part turned about the pivot contact's outer edge, so its side over the target contact
    rises by `CALIBRATION_FAIL` times the touch at the target's nearest point. None when the two
    contacts share a centre in the floor plane or the target does not lie on the near side of that
    edge (no lever)."""
    up = np.array([0.0, 1.0, 0.0])
    d = pieces.centroid(pivot) - pieces.centroid(target)
    d[Y] = 0.0
    if np.linalg.norm(d) <= noise:
        return None
    d /= np.linalg.norm(d)
    edge = float((pivot.reshape(-1, 3) @ d).max())
    lever = edge - float((target.reshape(-1, 3) @ d).max())
    if lever <= 0.0:
        return None
    theta = np.arcsin(min(1.0, CALIBRATION_FAIL * touch / lever))
    axis = np.cross(up, d)
    axis /= np.linalg.norm(axis)
    origin = d * edge
    origin[Y] = float(pivot.reshape(-1, 3)[:, Y].min())
    rel = part.tris - origin
    turned = (rel * np.cos(theta) + np.cross(axis, rel) * np.sin(theta)
              + axis * (rel @ axis)[..., None] * (1.0 - np.cos(theta)))
    return turned + origin


def _reaches(view: PieceView, start: str, goal: str) -> bool:
    """Whether the `on`/`rests_on` chain of `start` reaches `goal`. A visited set keeps a cycle
    from looping: `plants` can run on a manifest with one before `run` raises."""
    edges = {p.name: ([p.on] if p.on is not None else list(p.rests_on)) for p in view.parts}
    seen: set[str] = set()
    stack = [start]
    while stack:
        name = stack.pop()
        if name == goal:
            return True
        if name not in seen:
            seen.add(name)
            stack.extend(edges.get(name, ()))
    return False


def _phantom(view: PieceView, part: PartMesh, reach: float) -> str | None:
    """A part to name as a support the given part does not have: the first solid part with
    triangles, in the piece's order, that is not a support of it, is clear of it by `reach`, and
    whose own chain does not reach it (naming it would make a cycle)."""
    for other in view.parts:
        if (other is part or other.name in part.rests_on or not other.solid
                or len(other.tris) == 0 or _reaches(view, other.name, part.name)):
            continue
        if not list(pieces.box_overlaps([part, other], reach)):
            return other.name
    return None


def _bridge_units(ctx: SceneCtx, view: PieceView) -> list[PlantUnit]:
    """The tilt and the phantom units of the piece's smallest part that rests on several
    supports, built from the geometry and the manifest alone. None when no part has `rests_on`."""
    bridging = sorted((p for p in view.parts if p.rests_on and len(p.tris)),
                      key=lambda p: float(np.linalg.norm(p.hi - p.lo)))
    if not bridging:
        return []
    part, touch = bridging[0], touch_of(view)
    base = item_of(view, part.name)
    geometry = {p.name: p.tris for p in view.parts}
    by_name = {p.name: p for p in view.parts}
    noise = WELD_OF_PIECE_SIZE * view.size

    # The tilt: lift the part off the support it shares least contact with, about the edge of the
    # support farthest from that one. Areas tie within the float noise along a contact's edge.
    contacts = {s: _contact(part, by_name[s], touch) for s in part.rests_on if s in by_name}
    areas = {s: _area(t) if len(t) else 0.0 for s, t in contacts.items()}
    target = next(iter(contacts), None)
    for s in contacts:
        a, b = areas[s], areas[target]
        if a < b and abs(a - b) > noise * (np.sqrt(a) + np.sqrt(b)):
            target = s
    tilt = PlantUnit(base if target is None else f"{base} off {target}", (), not_applicable=(
        "its contact with a support cannot be read: no tilt lifts it off one"))
    others = [s for s in contacts if s != target and len(contacts[s])]
    if target is not None and len(contacts[target]) and others:
        c_target = pieces.centroid(contacts[target])
        pivot = max(others, key=lambda s: float(np.linalg.norm(
            (pieces.centroid(contacts[s]) - c_target)[[0, 2]])))
        turned = _tilted(part, contacts[target], contacts[pivot], touch, noise)
        if turned is not None:
            item = f"{base} off {target}"
            tilt = PlantUnit(item, (Plant(f"tilted: {item}", item, pieces.with_geometry(
                ctx, view, geometry | {part.name: turned})),))

    # The phantom: name a part it does not touch as one more support.
    named = _phantom(view, part, CALIBRATION_FAIL * touch)
    if named is None:
        phantom = PlantUnit(base, (), not_applicable="every other part is a support of it, within "
                                                    "reach of it, or rests on it: no support it "
                                                    "does not touch can be named")
    else:
        item = f"{base} off {named}"
        records = tuple(dataclasses.replace(r, rests_on=(*r.rests_on, named))
                        if r.name == part.name else r for r in view.piece.parts)
        piece = dataclasses.replace(view.piece, parts=records)
        phantom = PlantUnit(item, (Plant(f"phantom support: {item}", item, pieces.with_piece(
            ctx, view.pack, view.name, piece)),))
    return [tilt, phantom]


class Floaters:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    scope = "piece"

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [unit for view in pieces.contact_pieces(ctx, self.kinds) if len(view.parts) >= 2
                for unit in [_unit(ctx, view), *_bridge_units(ctx, view)]]

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
