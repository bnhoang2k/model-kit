"""Check clearance, layout clearance: two separate pieces placed side by side keep a gap a finger fits
through, or they are joined.

A gap narrower than a finger can be neither reached into nor seen as a gap; it reads as jammed.
The number is `MIN_GAP_M` (ISO 13854's minimum gap for a finger), in real metres, times the
style's `units_per_metre`. The standard's scope is crushing between moving machine parts; this kit
borrows only its body-part size.

In every layout, for every pair of placements, two items:

- **gap** `<layout> <a> x <b>` (ids sorted). A pair with no `on` or `rests_on` between them, in
  either direction, fails when the nearest distance between their solid parts, at their placed
  transforms, is under the minimum gap. Touching or crossing is distance 0. A part wholly inside
  a closed part of the other piece is an overlap too, though no surfaces cross.
- **rests on** `<layout> <a> rests on <b>`. A placement with `rests_on` must touch its host: the
  nearest distance between their solid parts is at most the resting piece's contact length
  (`sight.contact_length`). How deep the piece sinks into its host is not measured here.

A placement joined by `on` is measured by the receiver check, and its `rests_on` pair by the
second item above; neither is a gap pair.

A plug seated in a socket is the one contact between two unjoined pieces that is meant. A
connector end that names a layout and a host placement seats the cable piece's plug part (the
end's `plug` record) and every part resting `on` it (the blades), in the part of the host the
mount names (the mount's `part` record). Only those pairs of parts are not judged; every other
pair of the two placements is, so a plug against the host's other parts, or the lamp's base against
the strip, still fails. An end with no one placement of the cable's piece in the layout seats
nothing. An end with no `plug`, a mount with no `part`, a mount whose `part` is no solid with
geometry (a printed or cable part, or one the scene has no triangles for), or a plug with no solid
part with geometry, excuses nothing and says so in a plain line: the parts are never guessed from
the geometry.

A seated plug excuses contact, not penetration. For each excused pair the surface samples of each
part that lie inside the other (both closed) give a shared depth, the thinnest extent of those
samples (`geom.overlap_depth`; `crossings` measured it so before it read the shortest move), and a
depth over the contact length is a finding of the pair (the shorter of the two pieces' lengths): a
plug sunk into, or driven through, the part its socket sits on. A blade in a cut bore shares none.
The plants never call this depth test. Solid parts are the parts of role `solid` or
`skip_outline` (printed parts are not solids); parts of shape `cable` are left out, because a
lead runs where it must. A pair is measured with a box prefilter and `geom.surface_gap`, which is
exact up to the limit.

Not applicable, said in a plain line: a placement of a `branching` piece (tree crowns interleave
by nature), a placement of a piece with no solid part, and a layout with fewer than two pieces to
compare. A placement of an assembly piece the scene has no geometry for is `Unmeasured`. So is a
resting pair, or a plug seated in a socket, with a piece that declares no `view`: it has no contact
length.

Plants, per layout and kind of pair, one unit on the nearest such pair by the gap between the
placements' boxes (read from the geometry alone; the pair's own verdict is never asked for). A
gap unit tries to move each of the pair's placements in turn toward the other until the gap is half
the minimum; a rests-on unit lifts the resting placement `CALIBRATION_FAIL` times its touch limit
off its host, then by its own height as well. A gap unit whose pair already lies closer than half
the minimum has no move to make; the gate skips it when the real run reports the pair, and stops
the run when it does not. The check measures layouts and no body, so it reads no rig and no clip.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from model_kit.core import kinds as recipe_kinds
from model_kit.core.checks.pieces import PieceView, family, measured_pieces, placements_of
from model_kit.core.geom import (
    NO_DIRECTION,
    SurfaceIndex,
    is_closed,
    overlap_depth,
    shells,
    surface_gap,
    surface_samples,
)
from model_kit.core.manifest import Layout, LayoutPlacement, Placement
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx, part_key, piece_key
from model_kit.core.tolerances import CALIBRATION_FAIL, MIN_GAP_M

NAME = "clearance"
KINDS = frozenset({"assembly"})
PRINCIPLE = ("Two separate pieces placed side by side keep a gap a finger fits through, or they "
             "are joined.")
SOURCE = (
    "ISO 13854:2017, the minimum gap for a finger, 25 mm: the scope is from the standard's sample "
    "pages (https://cdn.standards.iteh.ai/samples/66459/46ce3b47790e4526802912dbbd921354/"
    "ISO-13854-2017.pdf), the finger row via BG ETEM's EN ISO 13854 sheet "
    "(https://medien.bgetem.de/medienportal/artikel/UzA0NC1F/@@download/download). The standard "
    "covers crushing between moving machine parts; this kit borrows only its body-part size.")


# --- a placement's solids, in the layout's frame ----------------------------------------------


@dataclass(eq=False)
class _Solid:
    tris: np.ndarray  # (N, 3, 3) in the layout's frame
    name: str = ""  # the part's name in its piece
    # Whether the part is a closed mesh, read on its piece-frame triangles: `is_closed` matches
    # corners by exact coordinates, and a turn into the layout's frame rounds them apart.
    closed: bool = False
    # One triangle of each island of the part (its connected surfaces), read on the piece-frame
    # triangles for the same reason. None: take the first triangle (a hand-made solid).
    islands: np.ndarray | None = None

    def __post_init__(self) -> None:
        points = self.tris.reshape(-1, 3)
        self.lo, self.hi = points.min(axis=0), points.max(axis=0)

    @cached_property
    def index(self) -> SurfaceIndex:
        return SurfaceIndex(self.tris)

    @cached_property
    def probes(self) -> np.ndarray:
        """One corner of every island of the part. With no surface of the other part within reach,
        an island lies wholly inside the other's mesh or wholly outside it, so one corner of it
        says which; a part of a hundred boxes is probed on all hundred."""
        rows = np.zeros(1, dtype=int) if self.islands is None else self.islands
        return self.tris[rows, 0]


@dataclass(eq=False)
class _Body:
    """One placement's solid parts. At least one part."""

    solids: list[_Solid]

    def __post_init__(self) -> None:
        self.lo = np.min([s.lo for s in self.solids], axis=0)
        self.hi = np.max([s.hi for s in self.solids], axis=0)


def _placement(p: LayoutPlacement) -> Placement:
    return Placement(p.id, p.piece, p.pack, p.position, p.quaternion)


def _island_rows(tris: np.ndarray) -> np.ndarray:
    """One triangle row of each island of a mesh: the triangles that share corners (matched by
    exact coordinates, as `is_closed` does) are one island."""
    _, ids = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    faces = ids.reshape(-1, 3)
    return np.array([rows[0] for _, rows in shells(faces, np.zeros(faces.max() + 1, dtype=int))])


def _body(view: PieceView, p: LayoutPlacement) -> _Body | None:
    """The placement's solid parts that have triangles and are not cables, or None with none."""
    placed = _placement(p)
    solids = [_Solid(placed.to_world(part.tris), part.name, is_closed(part.tris),
                     _island_rows(part.tris))
              for part in view.solids()
              if not part.bends and len(part.tris)]
    return _Body(solids) if solids else None


def _nested(x: _Solid, y: _Solid) -> bool:
    """Whether one part lies inside the other's closed mesh, with no surface crossing."""
    return any(b.closed and b.index.inside(a.probes).any() for a, b in ((x, y), (y, x)))


Seated = frozenset[tuple[str, str]]
"""(part of the first body, part of the second) pairs a seated plug excuses."""


def _gap(a: _Body, b: _Body, within: float, seated: Seated = frozenset()) -> float:
    """The nearest distance between the two bodies' solids: 0 when they touch, cross or nest.
    Exact when at most `within`; a larger answer is only some value over `within`. A pair of parts
    in `seated` (a plug in its socket) is not measured."""
    best = float("inf")
    for x in a.solids:
        for y in b.solids:
            if (x.name, y.name) in seated:
                continue
            if (x.lo - within > y.hi).any() or (y.lo - within > x.hi).any():
                continue
            gap = surface_gap(x.index, y.index, within)
            boxes_meet = bool((x.lo <= y.hi).all() and (y.lo <= x.hi).all())
            if gap == 0.0 or (gap > within and boxes_meet and _nested(x, y)):
                return 0.0
            best = min(best, gap)
    return best


def _shared_depth(x: _Solid, y: _Solid, spacing: float) -> float:
    """How far two parts share space (0 for none): the thinnest extent of the surface samples of
    each that lie inside the other (`geom.overlap_depth`; `crossings` measured it so before it read
    the shortest move). A part that is not closed holds no sample of the other. `spacing` is how
    far apart the samples lie."""
    if (x.lo > y.hi).any() or (y.lo > x.hi).any():
        return 0.0
    inside = []
    for held, holder in ((x, y), (y, x)):
        if not holder.closed:
            continue
        points = surface_samples(held.tris, spacing)
        points = points[((points >= holder.lo) & (points <= holder.hi)).all(axis=1)]
        if len(points):
            inside.append(points[holder.index.inside(points)])
    return overlap_depth(np.concatenate(inside)) if inside else 0.0


def _near(mover: _Body, other: _Body, seated: Seated = frozenset()
          ) -> tuple[float, np.ndarray, np.ndarray]:
    """The nearest vertex of either body to a solid of the other, as `(distance, point on the
    mover, point on the other)`. It is the gap itself for facing flat sides and an upper bound on
    the gap otherwise."""
    best = (float("inf"), np.zeros(3), np.zeros(3))
    for x in mover.solids:
        for y in other.solids:
            if (x.name, y.name) in seated:
                continue
            ours, theirs = x.tris.reshape(-1, 3), y.tris.reshape(-1, 3)
            to_y, to_x = y.index.closest(ours), x.index.closest(theirs)
            i, j = int(to_y.dist.argmin()), int(to_x.dist.argmin())
            if to_y.dist[i] < best[0]:
                best = (float(to_y.dist[i]), ours[i], to_y.point[i])
            if to_x.dist[j] < best[0]:
                best = (float(to_x.dist[j]), to_x.point[j], theirs[j])
    return best


# --- the items of a layout --------------------------------------------------------------------


@dataclass(frozen=True)
class _Pair:
    layout: Layout
    a: LayoutPlacement  # for a resting pair, the resting placement
    b: LayoutPlacement
    rests: bool

    @property
    def item(self) -> str:
        if self.rests:
            return f"{self.layout.name} {self.a.id} rests on {self.b.id}"
        first, second = sorted((self.a.id, self.b.id))
        return f"{self.layout.name} {first} x {second}"


def _on(a: LayoutPlacement, b: LayoutPlacement) -> bool:
    return a.on is not None and a.on.piece == b.id


def _pairs(layout: Layout, ids: Collection[str]) -> Iterator[_Pair]:
    """Every pair of these placements: a resting pair, an unjoined pair, and not a pair joined
    by `on` (the receiver's)."""
    placed = [layout.placements[i] for i in ids]
    for n, a in enumerate(placed):
        for b in placed[n + 1:]:
            if a.rests_on == b.id:
                yield _Pair(layout, a, b, True)
            elif b.rests_on == a.id:
                yield _Pair(layout, b, a, True)
            elif not (_on(a, b) or _on(b, a)):
                yield _Pair(layout, a, b, False)


def _joins(layout: Layout, p: LayoutPlacement) -> int:
    """How many placements of the layout `p` is joined to, by `on` or `rests_on`."""
    return sum(1 for q in layout.placements.values() if q is not p
               and (_on(p, q) or _on(q, p) or p.rests_on == q.id or q.rests_on == p.id))


def _moved(ctx: SceneCtx, layout: Layout, p: LayoutPlacement, delta: np.ndarray) -> SceneCtx:
    position = tuple(float(v) for v in np.asarray(p.position, dtype=float) + delta)
    moved = dataclasses.replace(p, position=position)
    layout = dataclasses.replace(layout, placements={**layout.placements, p.id: moved})
    return dataclasses.replace(ctx, layouts={**ctx.layouts, layout.name: layout})


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < NO_DIRECTION else v / n


class _Scan:
    """One scene's layouts: which placements it can measure, and their bodies on demand."""

    def __init__(self, ctx: SceneCtx) -> None:
        self.ctx = ctx
        self.views: dict[str, PieceView] = {v.key: v for v in measured_pieces(ctx, KINDS)}
        self._bodies: dict[tuple[str, str], _Body | None] = {}

    @property
    def limit(self) -> float:
        return MIN_GAP_M * self.ctx.style.units_per_metre

    def view(self, p: LayoutPlacement) -> PieceView | None:
        return self.views.get(piece_key(p.pack, p.piece))

    def body(self, layout: Layout, p: LayoutPlacement) -> _Body | None:
        key = (layout.name, p.id)
        if key not in self._bodies:
            view = self.view(p)
            self._bodies[key] = None if view is None else _body(view, p)
        return self._bodies[key]

    def seated(self, layout: Layout, p: LayoutPlacement, q: LayoutPlacement) -> Seated:
        """The (part of p, part of q) pairs a connector seats between the two placements, in
        either direction: a connector of p's piece whose end names this layout and q, or the
        reverse. Empty when no end does."""
        pairs = set(self._seats(layout, p, q)[0])
        pairs |= {(mine, theirs) for theirs, mine in self._seats(layout, q, p)[0]}
        return frozenset(pairs)

    def unseated(self, layout: Layout, p: LayoutPlacement, q: LayoutPlacement) -> list[str]:
        """Why a connector end that plugs one of the two into the other seats nothing: the
        records it needs (`End.plug`, `Mount.part`) are missing. Empty when none is."""
        return self._seats(layout, p, q)[1] + self._seats(layout, q, p)[1]

    def _seats(self, layout: Layout, cable: LayoutPlacement, host: LayoutPlacement
               ) -> tuple[set[tuple[str, str]], list[str]]:
        """((plug part of `cable`'s piece, part of `host`'s piece its mount names) for each end
        of a connector of `cable`'s piece that plugs into a mount of `host`, why an end seats
        nothing). The plug is the end's `plug` and every part resting on it; the host part is
        the mount's `part`. An end or mount that names none seats nothing and is said so; it is
        never guessed from the geometry. The cable's piece must have one placement in the layout,
        as `connectors` reads it; else nothing is seated."""
        cview, hview = self.view(cable), self.view(host)
        if cview is None or hview is None:
            return set(), []
        if len(placements_of(layout, cable.pack, cable.piece)) != 1:
            return set(), []
        seats: set[tuple[str, str]] = set()
        said: list[str] = []
        for conn in cview.piece.connectors:
            for end in conn.ends:
                if end.host.is_self or end.layout != layout.name or end.host.piece != host.id:
                    continue
                mount = next((m for m in hview.piece.mounts if m.id == end.host.id), None)
                if mount is None:
                    continue
                where = f"{cview.key} {conn.id} into {hview.key} {mount.id}"
                if end.plug is None:
                    said.append(f"{where}: the end names no plug")
                elif mount.part is None:
                    said.append(f"{where}: the mount names no host part")
                else:
                    mine, theirs = (self._solid_names(layout, p) for p in (cable, host))
                    plug = {p.name for p in family(cview.parts, end.plug)} & mine
                    if mount.part not in theirs:
                        said.append(f"{where}: the mount's part {mount.part} is no solid with "
                                    f"geometry")
                    elif not plug:
                        said.append(f"{where}: the end's plug {end.plug} is no solid with "
                                    f"geometry")
                    else:
                        seats |= {(name, mount.part) for name in plug}
        return seats, said

    def _solid_names(self, layout: Layout, p: LayoutPlacement) -> set[str]:
        """The names of the placement's solid parts that have triangles (see `_body`)."""
        body = self.body(layout, p)
        return set() if body is None else {s.name for s in body.solids}

    def touch(self, p: LayoutPlacement) -> float | None:
        """The placement's piece's contact length, None when the piece declares no view."""
        return self.view(p).contact

    def unjudged(self, pair: _Pair) -> str | None:
        """Why the contact rule of this pair cannot be read, or None when it can. A resting pair
        needs the resting piece's contact length; a pair that a seated plug excuses needs the
        shorter of its two pieces' lengths to bound how far the plug sinks (its gap rule does
        not)."""
        if pair.rests:
            needed = [pair.a]
        elif self.seated(pair.layout, pair.a, pair.b):
            needed = [pair.a, pair.b]
        else:
            return None
        if any(self.touch(p) is None for p in needed):
            return "a piece declares no view, so it has no contact length"
        return None

    def measurable(self, layout: Layout) -> list[str]:
        """The ids of the placements with a solid to measure, in the layout's order."""
        return [i for i, p in layout.placements.items() if self.body(layout, p) is not None]

    def findings(self, only: Collection[str] | None) -> list[Finding]:
        found: list[Finding] = []
        for layout in self.ctx.layouts.values():
            ids = [i for i, p in layout.placements.items() if self.view(p) is not None]
            for pair in _pairs(layout, ids):
                if only is not None and pair.item not in only:
                    continue
                a, b = self.body(layout, pair.a), self.body(layout, pair.b)
                if a is None or b is None or (pair.rests and self.unjudged(pair)):
                    continue
                detail = self._fault(pair, a, b)
                if detail is not None:
                    found.append(Finding(NAME, pair.item, detail))
        return found

    def _fault(self, pair: _Pair, a: _Body, b: _Body) -> str | None:
        ka, kb = piece_key(pair.a.pack, pair.a.piece), piece_key(pair.b.pack, pair.b.piece)
        if pair.rests:
            limit = self.touch(pair.a)
            if _gap(a, b, limit) <= limit:
                return None
            far = _gap(a, b, _near(a, b)[0])  # the vertex distance is an upper bound
            return (f"{ka} rests on {kb} but is {far:.4f} off it (it must touch within "
                    f"{limit:.4f})")
        limit = self.limit
        if (a.lo - limit > b.hi).any() or (b.lo - limit > a.hi).any():
            return None
        seated = self.seated(pair.layout, pair.a, pair.b)
        gap = _gap(a, b, limit, seated)
        if gap < limit:
            said = "touch or overlap" if gap == 0.0 else f"are {gap:.4f} apart"
            return (f"{ka} and {kb} {said}: unjoined pieces keep at least {limit:.4f} "
                    f"({MIN_GAP_M} m)")
        return None if self.unjudged(pair) else self._sunk(pair, a, b, seated)

    def _sunk(self, pair: _Pair, a: _Body, b: _Body, seated: Seated) -> str | None:
        """A seated plug excuses contact, not penetration: a pair it excuses that shares more
        space than the contact length is a fault. A blade in its cut bore shares none."""
        touch = min(self.touch(pair.a), self.touch(pair.b))
        by_a = {s.name: s for s in a.solids}
        by_b = {s.name: s for s in b.solids}
        worst = max(((_shared_depth(by_a[x], by_b[y], touch), x, y) for x, y in sorted(seated)),
                    default=(0.0, "", ""))
        if worst[0] <= touch:
            return None
        ka, kb = piece_key(pair.a.pack, pair.a.piece), piece_key(pair.b.pack, pair.b.piece)
        return (f"{ka} {worst[1]} and {kb} {worst[2]} share {worst[0]:.4f} of depth: a seated "
                f"plug may touch its socket's part, not sink into it (limit {touch:.4f})")

    # --- plants ---

    def plants(self) -> list[PlantUnit]:
        units: list[PlantUnit] = []
        for layout in self.ctx.layouts.values():
            pairs = [p for p in _pairs(layout, self.measurable(layout))
                     if not (p.rests and self.unjudged(p))]
            for rests, make in ((False, self._gap_unit), (True, self._lift_unit)):
                kind = [p for p in pairs if p.rests is rests]
                if kind:
                    units.append(make(self._nearest_first(kind)[0]))
        return units

    def _nearest_first(self, pairs: list[_Pair]) -> list[_Pair]:
        def apart(p: _Pair) -> float:
            a, b = self.body(p.layout, p.a), self.body(p.layout, p.b)
            return float(np.linalg.norm(np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))))
        return sorted(pairs, key=apart)

    def _gap_unit(self, pair: _Pair) -> PlantUnit:
        """Move a placement of the pair toward the other, to half the minimum gap: the less joined
        one first, then the other."""
        a, b = pair.a, pair.b
        first, second = (a, b) if _joins(pair.layout, a) < _joins(pair.layout, b) else (b, a)
        target = self.limit / CALIBRATION_FAIL
        candidates = []
        for mover, other in ((first, second), (second, first)):
            dist, mine, theirs = _near(self.body(pair.layout, mover),
                                       self.body(pair.layout, other),
                                       self.seated(pair.layout, mover, other))
            toward = _unit(theirs - mine)
            if toward is not None and dist > target:
                planted = _moved(self.ctx, pair.layout, mover, toward * (dist - target))
                candidates.append(Plant(f"moved to half the gap: {pair.item}", pair.item,
                                        planted))
        return PlantUnit(pair.item, tuple(candidates))

    def _lift_unit(self, pair: _Pair) -> PlantUnit:
        """Lift the resting placement off its host: twice the touch limit, then, if it still
        reaches the host, by its own height as well."""
        body = self.body(pair.layout, pair.a)
        base = CALIBRATION_FAIL * self.touch(pair.a)
        return PlantUnit(pair.item, tuple(
            Plant(f"lifted off its host: {pair.item}", pair.item,
                  _moved(self.ctx, pair.layout, pair.a, np.array([0.0, lift, 0.0])))
            for lift in (base, base + float(body.hi[1] - body.lo[1]))))

    # --- coverage ---

    def coverage(self) -> list[str]:
        if not self.ctx.layouts:
            return [f"{NAME}: not applicable (no layout): nothing to measure"]
        lines: list[str] = []
        for layout in self.ctx.layouts.values():
            lines += self._layout_lines(layout)
        return lines

    def _layout_lines(self, layout: Layout) -> list[str]:
        lines: list[str] = []
        if not layout.placements:
            return [f"{NAME}: not applicable (no placements): {layout.name}"]
        for pid, p in layout.placements.items():
            piece = self.ctx.packs[p.pack].pieces[p.piece]
            if piece.recipe.kind not in KINDS:
                continue  # `skipped_kinds` names the piece
            if self.view(p) is None:
                lines.append(Unmeasured(f"{NAME}: skipped (no geometry): {layout.name} {pid}"))
            elif self.body(layout, p) is None:
                lines.append(f"{NAME}: not applicable (no solid part): {layout.name} {pid}")
        ids = self.measurable(layout)
        for pair in _pairs(layout, ids):
            seated = self.seated(layout, pair.a, pair.b)
            if seated:
                first, second = sorted((pair.a.id, pair.b.id))
                lines.append(f"{NAME}: {layout.name} {first} x {second}: a connector seats a plug "
                             f"in its socket, so these part pairs are not judged: "
                             f"{', '.join(f'{x} with {y}' for x, y in sorted(seated))}")
            for why in self.unseated(layout, pair.a, pair.b):
                lines.append(f"{NAME}: {pair.item}: no contact is "
                             f"excused, a connector end cannot say which parts seat: {why}")
            if (why := self.unjudged(pair)) is not None:
                lines.append(Unmeasured(f"{NAME}: skipped ({why}): {pair.item}"))
        if len(ids) < 2:
            lines.append(f"{NAME}: not applicable (fewer than two pieces to compare): "
                         f"{layout.name}")
        elif not list(_pairs(layout, ids)):
            lines.append(f"{NAME}: not applicable (every pair is joined): {layout.name}")
        return lines


class Clearance:
    name = NAME
    kinds: frozenset[str] = KINDS
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return _Scan(ctx).plants()

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _Scan(ctx).findings(only)

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The solid parts (not cables) with triangles of each piece that a layout places beside
        another measurable piece: the parts a pair can be judged on."""
        scan = _Scan(ctx)
        keys: set[str] = set()
        for layout in ctx.layouts.values():
            ids = scan.measurable(layout)
            if len(ids) < 2:
                continue
            for pid in ids:
                view = scan.view(layout.placements[pid])
                keys |= {part_key(view.pack, view.name, p.name) for p in view.solids()
                         if not p.bends and len(p.tris)}
        return keys

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = list(recipe_kinds.skipped_kinds(ctx, NAME, self.kinds))
        return lines + _Scan(ctx).coverage()
