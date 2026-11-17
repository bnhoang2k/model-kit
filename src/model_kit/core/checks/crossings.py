"""Check crossings, part crossings: two parts of one piece must not sink into each other.

For each piece of a kind in `kinds` (assembly), every pair of its parts whose boxes overlap is
measured by `measure`: how far the two parts share space, and how thick the thinner one is.

Shared space is the shortest move that separates the two parts (`geom.shortest_exit`, the
penetration depth of collision detection). The points are the surface samples of each part that lie
inside the other (spaced a quarter of the sampled part's thickness at its centre). A way is one of
26 fixed directions, crossings' 13 (the 3 axes, 6 face diagonals and 4 body diagonals) both ways.
Along a way, one part moves by t and the other by t the opposite way; t is the furthest any
sample must go to leave the part it is in. The depth is the least t over the ways. So a fret sunk
into a tilted board reads its depth under the board's top, and not the run of the region it shares
along the board. (The extent of the shared points over 13 directions, `geom.overlap_depth`, would read
the run, which is longer than the depth.) A rod through a plate reads at least the
plate's thickness. The thickness of a part is `geom.local_thickness` (the cited shape diameter, a
cone of rays) at the sample nearest the middle of the shared region, and
`geom.thinner_thickness` reads the thinner of the two there.
The pair fails when the shared depth is over `CROSS_OF_PART_THICKNESS` times that thinner
thickness: a part sunk more than halfway into another is embedded and not resting (a design call of this kit).
The limit follows the parts and not the pack. Flush contact (a foot on a
deck) shares no space; a pair whose points all leave along one common way is read as flush before
any ray is cast. Parts printed on a surface (role `eyes` or `decal`) are left out.

A pair whose surfaces never meet is skipped before any sampling: no corner of either part inside
the other and no triangle of one cutting a triangle of the other.

Depth needs closed meshes: parity says what is inside, and a ray from the surface says how thick.
A solid part that is open, or has no thickness at its centre, cannot be measured: it is an
`Unmeasured` line (its shape is topology's finding) and its pairs are left out. A pair whose
thickness cannot be read at the middle of its shared region (`geom.thinner_thickness` raises
`Unmeasurable`) is an `Unmeasured` line for that pair, and the run goes on. There is no fall-back
to a thickness read elsewhere on the part.

The item is `<pack>:<piece> <a> x <b>`, the two part names sorted, so one allowlist line names
one pair. A designed join (a pin through its socket, a tenon in a mortise, a bolt through a plate)
is an allowlist line with a reason; the allowlist fails a line with no reason and a line that
matched nothing.

The plant is one unit per piece with two or more measurable solid parts, built from the
geometry alone (never from `measure`). Each part has a host: the part whose box lies nearest it
(or overlaps it; the first in part order on a tie). A part costs the estimated sample count of
its surface: its area over the square of the finer of its own and its host's sample spacing,
because the check samples that many points. The plant
copies the part of least cost (the first in part order on a tie) and sinks the copy into its host
by `CALIBRATION_FAIL` limits, going toward the host's centre, then along each axis, then right
onto its centre (which reaches a pair however far apart). The copy is a new part, so the
unit's item is new and the pair itself may fail or pass.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.errors import TooManySamples, Unmeasurable
from model_kit.core.geom import (
    DEPTH_WAYS,
    NO_DIRECTION,
    TIE_DISTANCE,
    Exit,
    Nearest,
    SurfaceIndex,
    check_sample_budget,
    is_closed,
    local_thickness,
    shortest_exit,
    surface_area,
    surface_samples,
    thinner_thickness,
    tri_tri_cut,
)
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CROSS_OF_PART_THICKNESS,
    CROSS_SAMPLE_OF_THICKNESS,
)

NAME = "crossings"
PRINCIPLE = ("Two solid parts cannot share space: a part sunk more than halfway into its "
             "neighbour is embedded, and what looks wrong is the size of the dent against the "
             "size of the thing dented.")
SOURCE = ("A part sunk more than halfway into another is embedded (a design call of this kit); "
          "local thickness is the "
          "shape diameter of Shapira et al. 2008, \"Consistent mesh partitioning and "
          "skeletonisation using the shape diameter function\", The Visual Computer 24.")


class Solid:
    """A closed part ready to be measured: its surface, how thick it is at its centre (which sets
    how finely its surface is sampled: a resolution, never a limit) and, made once and kept, an
    index, samples and corners.

    `moved` gives the same part shifted. It shares the unshifted part's index and samples (a
    query asks the index about the point shifted back), so a sweep of many shares builds nothing
    new per share.
    """

    def __init__(self, tris: np.ndarray, thickness: float, root: Solid | None = None,
                 shift: np.ndarray | None = None) -> None:
        self.tris = tris
        self.thickness = thickness
        self._root = self if root is None else root
        self.shift = np.zeros(3) if shift is None else shift
        self.lo, self.hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
        self._made: dict[str, np.ndarray | SurfaceIndex] = {}

    @property
    def spacing(self) -> float:
        return CROSS_SAMPLE_OF_THICKNESS * self.thickness

    @property
    def sample_estimate(self) -> float:
        """About how many surface samples the part has: its area over the spacing squared. It
        ranks cost between parts and is never compared with a limit."""
        return surface_area(self._root.tris) / self.spacing ** 2

    def moved(self, by: np.ndarray) -> Solid:
        return Solid(self.tris + by, self.thickness, self._root, self.shift + by)

    def _root_made(self, key: str):
        made = self._root._made
        if key not in made:
            root = self._root
            made[key] = {
                "index": lambda: SurfaceIndex(root.tris),
                "samples": lambda: surface_samples(root.tris, root.spacing),
                "corners": lambda: np.unique(root.tris.reshape(-1, 3), axis=0),
            }[key]()
        return made[key]

    @property
    def samples(self) -> np.ndarray:
        return self._root_made("samples") + self.shift

    @property
    def corners(self) -> np.ndarray:
        return self._root_made("corners") + self.shift

    def contains(self, points: np.ndarray) -> np.ndarray:
        """True for each point that lies inside the part."""
        return self._root_made("index").inside(points - self.shift)

    def nearest(self, points: np.ndarray) -> Nearest:
        """The nearest point of the part's surface to each point, with its face's normal."""
        near = self._root_made("index").closest(points - self.shift)
        return near._replace(point=near.point + self.shift)

    @property
    def index(self) -> SurfaceIndex:
        if "own" not in self._made:
            self._made["own"] = SurfaceIndex(self.tris)
        return self._made["own"]


def solid_of(tris: np.ndarray) -> Solid:
    """The part ready to measure. Its thickness is `geom.local_thickness` at its own centre; the
    samples of its surface are spaced from it. That centre thickness is not the thinner thickness
    at the overlap of a pair (`measure` reads that), so a thin place away from the centre is
    sampled no finer. Raises `ValueError` for an open mesh, or one with no
    thickness at its centre, and `TooManySamples` when that thickness is so thin that its samples
    would pass the kit's cap (`geom.check_sample_budget`: a shell sunk a hair inside another shell
    of the same part reads as the hair); each message reads as the reason in a coverage line."""
    if not is_closed(tris):
        raise ValueError("open part")
    centre = pieces.centroid(tris)
    try:
        thickness = local_thickness(tris, centre)
    except ValueError as e:
        raise ValueError("no thickness at its centre") from e
    if not thickness > 0.0:
        raise ValueError("no thickness at its centre")
    solid = Solid(tris, thickness)
    try:
        check_sample_budget(tris, solid.spacing)
    except TooManySamples as e:
        at = ", ".join(f"{c:.3g}" for c in centre)
        raise TooManySamples(e.points, e.spacing, e.cap,
                             lead=f"its thickness at its centre ({at}) reads {thickness:.3g}, and ",
                             note="a thin spot there, or a shell sunk inside another shell of "
                                  "the part") from e
    return solid


class Cross(NamedTuple):
    depth: float  # how far the two parts share space (0 for none)
    limit: float  # what that may be: CROSS_OF_PART_THICKNESS x the thinner part's thickness
    thickness: float  # that thinner thickness
    resolution: float = 0.0  # the finer sample spacing of the two parts: what a depth can miss by

    def beyond(self, rest: float = 0.0) -> bool:
        """True when the depth is over the limit by more than float noise and, for a pair that
        shared `rest` already, deeper than that by more than the samples can tell apart. A part
        that reaches exactly half the thickness is at the limit and not over it, whichever way
        the last bit of the sum rounds."""
        deeper = rest <= 0.0 or self.depth > rest + self.resolution
        return self.depth > self.limit + TIE_DISTANCE and deeper

    @property
    def over(self) -> bool:
        return self.beyond()


def _in_box(points: np.ndarray, other: Solid) -> np.ndarray:
    return points[((points >= other.lo) & (points <= other.hi)).all(axis=1)]


def _touch(a: Solid, b: Solid) -> bool:
    """False when the surfaces of the two parts never meet: no corner of either inside the other
    and no triangle of one cutting a triangle of the other. Then they share no space."""
    for x, y in ((a, b), (b, a)):
        corners = _in_box(x.corners, y)
        if len(corners) and y.contains(corners).any():
            return True
    return tri_tri_cut(a.index, b.index) > 0.0


def _all_leave(sides: Sequence[Exit]) -> bool:
    """True when one way takes every point of every side out at once: the pair is flush, and
    `shortest_exit` would read 0 for it."""
    return bool(np.logical_and.reduce([side.leaves.all(axis=0) for side in sides]).any())


def measure(a: Solid, b: Solid) -> Cross:
    """How far the two parts share space, and the limit for that. A depth within float noise is
    flush contact and shares nothing. Raises `Unmeasurable` when no direction takes the parts out
    of each other, or when the thinner part's thickness cannot be read at the middle of the shared
    region."""
    if (a.lo > b.hi).any() or (b.lo > a.hi).any() or not _touch(a, b):
        return Cross(0.0, 0.0, 0.0)
    ways = DEPTH_WAYS
    sides, inside = [], []
    for x, y in ((a, b), (b, a)):
        near = _in_box(x.samples, y)
        held = near[y.contains(near)] if len(near) else near
        inside.append(held)
        sides.append((x, y, held))
    points = np.concatenate(inside)
    if not len(points):
        return Cross(0.0, 0.0, 0.0)
    # x moves by t along a way and y by t along its reverse, one way for both: x's points leave y
    # along `ways`, and y's points leave x along `-ways`.
    exits = []
    for (x, y, held), sign in zip(sides, (1.0, -1.0), strict=True):
        if not len(held):
            continue
        near = y.nearest(held)
        dirs = sign * ways
        # A point inside by no more than float noise lies on the other part's surface, and it
        # leaves at once along a way that looks out of it (one the surface's normal faces).
        leaves = (near.dist <= TIE_DISTANCE)[:, None] & (near.normal @ dirs.T > 0.0)
        exits.append((Exit(y.tris, held, dirs, leaves), float(near.dist.max())))
    if _all_leave([side for side, _ in exits]):
        return Cross(0.0, 0.0, 0.0)
    reach = max(2.0 * max(deepest for _, deepest in exits), min(a.spacing, b.spacing))
    depth = shortest_exit([side for side, _ in exits], reach)
    if not np.isfinite(depth):
        raise Unmeasurable("no direction takes the parts out of each other")
    if not depth > TIE_DISTANCE:
        return Cross(0.0, 0.0, 0.0)
    middle = points[np.linalg.norm(points - points.mean(axis=0), axis=1).argmin()]
    thickness = thinner_thickness(a.tris, b.tris, middle)
    return Cross(depth, CROSS_OF_PART_THICKNESS * thickness, thickness,
                 min(a.spacing, b.spacing))


@dataclass(frozen=True)
class _Made:
    """A piece's measurable solid parts, and why the others are not."""
    parts: list[PartMesh]  # solid parts with triangles that `solid_of` took
    solids: dict[str, Solid]
    skipped: dict[str, str]  # part name -> why it cannot be measured


def made_of(view: PieceView) -> _Made:
    parts, solids, skipped = [], {}, {}
    for part in view.solids():
        if len(part.tris) == 0:
            continue
        try:
            solids[part.name] = solid_of(part.tris)
        except (ValueError, TooManySamples) as e:
            skipped[part.name] = str(e)
        else:
            parts.append(part)
    return _Made(parts, solids, skipped)


def _cuts(view: PieceView, made: _Made, only: Collection[str] | None
          ) -> tuple[list[tuple[Cross, str]], list[tuple[str, str]]]:
    """`([(cross, item)], [(item, why)])` for the piece's measurable solid parts: each pair that
    sinks too deep, and each pair whose thickness cannot be read where it is measured."""
    cuts, lost = [], []
    for i, j in pieces.box_overlaps(made.parts):
        a, b = made.parts[i], made.parts[j]
        item = item_of(view, a.name, b.name)
        if only is not None and item not in only:
            continue
        try:
            cross = measure(made.solids[a.name], made.solids[b.name])
        except Unmeasurable as err:
            lost.append((item, str(err)))
            continue
        if cross.over:
            cuts.append((cross, item))
    return cuts, lost


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return None if n < NO_DIRECTION else v / n


def _apart(a: PartMesh, b: PartMesh) -> float:
    """How far the two parts' boxes lie apart (0 for boxes that overlap)."""
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(a.lo - b.hi, b.lo - a.hi))))


def _sinks(ctx: SceneCtx, view: PieceView, made: _Made) -> PlantUnit:
    """The piece's plant unit: a copy of the part of least cost, sunk into its host (the part
    whose box lies nearest it). Read from the parts' boxes, centres, areas and thickness;
    `measure` is never called."""
    parts, solids = made.parts, made.solids

    def host(i: int) -> int:
        return min((j for j in range(len(parts)) if j != i),
                   key=lambda j: (_apart(parts[i], parts[j]), j))

    def cost(i: int) -> float:
        spacing = min(solids[parts[i].name].spacing, solids[parts[host(i)].name].spacing)
        return surface_area(parts[i].tris) / spacing ** 2

    copy = min(range(len(parts)), key=lambda i: (cost(i), i))
    a, b = parts[copy], parts[host(copy)]
    depth = CALIBRATION_FAIL * CROSS_OF_PART_THICKNESS * min(
        made.solids[a.name].thickness, made.solids[b.name].thickness)
    name = pieces.free_name([p.name for p in view.parts], a.name)
    item = item_of(view, name, b.name)
    to_b = pieces.centroid(b.tris) - pieces.centroid(a.tris)
    toward = _unit(to_b)
    moves = [toward * depth] if toward is not None else []
    moves += [axis * depth for axis in np.vstack([np.eye(3), -np.eye(3)])] + [to_b]
    geometry = {p.name: p.tris for p in view.parts}
    return PlantUnit(item, tuple(
        Plant(f"sunk: {item}", item, pieces.with_geometry(ctx, view, geometry | {name: a.tris + m}))
        for m in moves))


class Crossings:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    scope = "piece"
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._seen: tuple[SceneCtx, dict[str, tuple[list, list]]] | None = None

    def _all(self, ctx: SceneCtx) -> dict[str, tuple[list, list]]:
        """Every piece's cuts and lost pairs, kept for the scene they were made for, so the run
        and the coverage lines measure each pair once."""
        if self._seen is None or self._seen[0] is not ctx:
            self._seen = (ctx, {v.key: _cuts(v, made_of(v), None)
                                for v in pieces.measured_pieces(ctx, self.kinds)})
        return self._seen[1]

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units = []
        for view in pieces.measured_pieces(ctx, self.kinds):
            made = made_of(view)
            if len(made.parts) >= 2:
                units.append(_sinks(ctx, view, made))
        return units

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every solid part that can be measured, of a piece that has two or more: each is tested
        against all the others. A lone part, a printed part, an empty part and an open part have
        no pair."""
        out: set[str] = set()
        for view in pieces.measured_pieces(ctx, self.kinds):
            made = made_of(view)
            if len(made.parts) >= 2:
                out |= {item_of(view, p.name) for p in made.parts}
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for view in pieces.measured_pieces(ctx, self.kinds, only):
            cuts = self._all(ctx)[view.key][0] if only is None else _cuts(
                view, made_of(view), only)[0]
            found += [(cross.depth, Finding(
                NAME, item, f"surfaces share {cross.depth:.3g} of depth "
                            f"(limit {cross.limit:.3g}, {CROSS_OF_PART_THICKNESS} x the "
                            f"thinner part's thickness {cross.thickness:.3g})"))
                      for cross, item in cuts]
        return [f for _, f in sorted(found, key=lambda row: (-row[0], row[1].item))]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        views = list(pieces.measured_pieces(ctx, self.kinds))
        mades = {v.key: made_of(v) for v in views}
        lone = [f"{NAME}: not applicable (fewer than two solid parts): {v.key}"
                for v in views if len([p for p in v.solids() if len(p.tris)]) < 2]
        skipped = [Unmeasured(f"{NAME}: skipped ({why}): {v.key} {name}")
                   for v in views for name, why in mades[v.key].skipped.items()]
        lost = [Unmeasured(f"{NAME}: skipped ({why}): {item}")
                for v in views for item, why in sorted(self._all(ctx)[v.key][1])]
        return pieces.coverage(ctx, NAME, self.kinds, printed=True) + lone + skipped + lost
