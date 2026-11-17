"""Take-apart: each group of a piece must come away from the others without cutting them.

A piece's parts belong to groups, and the groups are listed in build order: the order their first
part appears in (`group_order`). Taking apart reverses putting together, so the check takes the
piece apart in the reverse of that order: the last group made comes off first. A part taken off
earlier no longer exists for the parts taken off after it, as in a real teardown: the moving
group is measured only against the groups still on the piece, those that come off after it, as
built.

A group is tried first along its pull, the line from the piece's centre to its own centre. If
that cuts, it is tried along each of the 26 ways of `geom.DEPTH_WAYS` (the 13 directions of
`geom.DEPTH_DIRECTIONS`, both signs; crossings and bearing read the same set), the ways
nearest its pull first. A group at the piece's centre has no pull and goes straight to the 26
ways. Along each way it moves out through `TAKE_APART_STEPS` shares of the piece size from 0 to
1. At each share its parts and theirs are measured by `crossings.measure`, which gives the depth
of the space the two parts share and the limit there: `CROSS_OF_PART_THICKNESS` times the thinner
part's thickness, the same rule `crossings` applies. A way is clear when no pair cuts along it.
The group comes away when any way is clear, and then it has no finding. So the check is never
stricter than the pull alone, for a group that has one. When every way cuts, the finding names
the way that cuts least deep, as a unit vector, and reports the group pairs that cut along it.

A part that bends (`Part.bends`, a cable) neither moves nor blocks: a cable can be pushed aside,
so only its rigid plugs are tried and the cable never blocks them or any other group. It is left
out of the piece's centre and size too, and says so in a `not applicable` line. Parts printed on
a surface (role `eyes` or `decal`) are left out, and so is a part that cannot be measured (open,
or with no thickness at its centre): an `Unmeasured` line, as in `crossings`, except on a piece
with fewer than two groups, which has nothing to take apart and says so in one `not applicable`
line. A group pair whose parts' thickness cannot be read where they overlap
(`geom.thinner_thickness` raises `Unmeasurable`) along a way is not judged along it, and a group
whose every way is cut or unreadable and has an unreadable one gets no finding, only an
`Unmeasured` line, and the run goes on.

A pair of parts is judged against its depth at rest (0 for a clean pair): it cuts when the
sweep's depth is more than that depth plus the limit. A seated pair (a pin in its socket) that
the sweep drives deeper cuts along that way, while one that comes straight out does not; the seat
at rest is `crossings`' to judge and to allow, and take-apart has no allowlist line for it. The
item is `<pack>:<piece> <a> x <b>`, the two group names sorted. A part with no group in the
manifest is `Unmeasured` ("no group"), and a piece in one group has nothing to take apart.

The plant is one unit per piece with two or more swept groups, built from the groups' geometry
alone (never from the sweep). It takes the group with the least estimated sample count (the sum of
`Solid.sample_estimate` over its parts), because its sweeps hold the fewest points, and boxes it
in: six slabs around the box of its parts, a gap clear of it, each as thick as the piece is
large, in a group of its own that is built first and so comes off last. No way out is free, so the
check must flag the pair.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.crossings import Solid, made_of, measure
from model_kit.core.checks.pieces import NO_GROUP, PartMesh, PieceView, item_of
from model_kit.core.errors import Unmeasurable
from model_kit.core.geom import DEPTH_WAYS, NO_DIRECTION
from model_kit.core.manifest import Piece
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import CAGE_GAP_OF_GROUP, CAGE_WALL_OF_PIECE, TAKE_APART_STEPS

NAME = "take-apart"
PRINCIPLE = ("A part comes out of an assembly along a direction in which no other part blocks "
             "it, and a fixed set of directions is a stated sample of all of them.")
SOURCE = ("Wilson and Latombe 1994, \"Geometric reasoning about mechanical assembly\", "
          "Artificial Intelligence 71, 371-396: the non-directional blocking graph (p. 372; in "
          "3D p. 382) partitions all translation directions into finitely many regions over "
          "which the blocking between parts is constant; section 5.3 (p. 384) calls a part "
          "blocked along a direction when the volume it sweeps along it meets another part. The "
          "kit does not build that arrangement or sweep to infinity: it moves a group out by "
          "up to one piece size, in 21 samples, along a fixed set of directions (its pull and "
          "26 ways). A free region narrower than that set's spacing can be missed, and the "
          "group is then reported as stuck (a false finding); and a blocker thinner than a "
          "sample step can be passed between two samples, so a finding can also be missed.")


def shares() -> np.ndarray:
    """The shares of the piece size a group moves out by."""
    return np.linspace(0.0, 1.0, TAKE_APART_STEPS)


def group_order(piece: Piece) -> list[str]:
    """The piece's groups in build order: first seen among its parts. The sweep walks it reversed."""
    return list(dict.fromkeys(part.group for part in piece.parts))


def _rigid(view: PieceView) -> PieceView:
    """The piece without its cables: a part that bends neither moves nor blocks."""
    return dataclasses.replace(view, parts=tuple(p for p in view.parts if not p.bends))


def _one_group(view: PieceView) -> bool:
    """The piece's rigid solid parts fall in fewer than two groups: nothing to take apart."""
    return len({p.group for p in _rigid(view).solids() if p.group != NO_GROUP}) < 2


@dataclass(frozen=True)
class _Layout:
    parts: list[PartMesh]  # the parts that are swept: rigid, solid, measurable, with a group
    solids: list[Solid]  # the same parts, ready to be measured
    order: list[str]  # their groups, in take-apart order (the reverse of the build order)
    toward: dict[str, np.ndarray]  # each group's unit pull (zero at the centre)
    size: float  # the piece size: the diagonal of the swept parts' box
    stage: np.ndarray  # each part's place in `order`
    lo: np.ndarray  # (N, 3) each part's box
    hi: np.ndarray


def _layout(view: PieceView) -> _Layout | None:
    """None when fewer than two groups are swept."""
    made = made_of(_rigid(view))
    parts = [p for p in made.parts if p.group != NO_GROUP]
    order = [g for g in reversed(group_order(view.piece)) if any(p.group == g for p in parts)]
    if len(order) < 2:
        return None
    every = np.concatenate([p.tris for p in parts]).reshape(-1, 3)
    centre = pieces.centroid(every.reshape(-1, 3, 3))
    toward = {}
    for group in order:
        tris = np.concatenate([p.tris for p in parts if p.group == group])
        away = pieces.centroid(tris) - centre
        length = float(np.linalg.norm(away))
        toward[group] = away / length if length >= NO_DIRECTION else np.zeros(3)
    size = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    return _Layout(parts, [made.solids[p.name] for p in parts], order, toward, size,
                   np.array([order.index(p.group) for p in parts]),
                   np.array([p.lo for p in parts]), np.array([p.hi for p in parts]))


def _vector(way: np.ndarray) -> str:
    return "(" + ", ".join(f"{c:+.3f}" for c in way) + ")"


def _ways(pull: np.ndarray) -> list[np.ndarray]:
    """The ways a group is tried along, in order: its pull (none at the centre), then the 26
    ways of `DEPTH_WAYS`, nearest the pull first."""
    ranked = [w for w in DEPTH_WAYS[np.argsort(-(DEPTH_WAYS @ pull), kind="stable")]
              if not pull.any() or np.linalg.norm(w - pull) >= NO_DIRECTION]
    return [pull, *ranked] if pull.any() else ranked


@dataclass(frozen=True)
class _Way:
    """What a group cuts while it moves out along one way."""

    direction: np.ndarray
    cuts: dict[str, tuple[float, float, float]]  # item -> (deepest, limit, first share that cuts)
    lost: dict[str, str]  # item -> why its thickness could not be read

    @property
    def deepest(self) -> float:
        return max(depth for depth, _, _ in self.cuts.values())


def _along(view: PieceView, lay: _Layout, stage: int, way: np.ndarray, at_rest,
           beat: float = np.inf) -> _Way | None:
    """The group at `stage` moved out along `way`, through the shares of the piece size, against
    the groups that come off after it. None when it cuts as deep as `beat` already: that way
    cannot be the least deep, so the rest of its sweep is not run."""
    mover = np.flatnonzero(lay.stage == stage)
    others = np.flatnonzero(lay.stage > stage)
    cuts: dict[str, tuple[float, float, float]] = {}
    lost: dict[str, str] = {}
    moved: dict[tuple[int, int], Solid] = {}
    for n, share in enumerate(shares()):
        off = way * share * lay.size
        near = ((lay.lo[mover] + off)[:, None, :] <= lay.hi[others][None, :, :]) & (
            lay.lo[others][None, :, :] <= (lay.hi[mover] + off)[:, None, :])
        for a, b in np.argwhere(near.all(axis=2)):
            i, j = int(mover[a]), int(others[b])
            item = item_of(view, lay.parts[i].group, lay.parts[j].group)
            if item in lost:
                continue
            if (i, n) not in moved:
                moved[i, n] = lay.solids[i].moved(off)
            try:
                rest = at_rest(i, j)  # 0 for a pair that is clean at rest
                cross = measure(moved[i, n], lay.solids[j])
            except Unmeasurable as err:
                lost[item] = str(err)
                continue
            if cross.beyond(rest):
                deepest, limit, first = cuts.get(item, (0.0, 0.0, float(share)))
                if cross.depth > deepest:
                    deepest, limit = cross.depth, cross.limit
                cuts[item] = (deepest, limit, first)
                if deepest >= beat and item not in lost:
                    return None
    return _Way(way, {item: cut for item, cut in cuts.items() if item not in lost}, lost)


def _leaves(view: PieceView, lay: _Layout, stage: int, at_rest
            ) -> tuple[list[tuple[str, float, float, str, float, np.ndarray]], dict[str, str]]:
    """`(hits, lost)` for the group at `stage`: nothing when some way is clear. When none is, and
    each has a cut, the pairs that cut along the least deep way. When one way is only unreadable,
    no hit, only what could not be read."""
    mover = lay.order[stage]
    best = unread = None
    for way in _ways(lay.toward[mover]):
        got = _along(view, lay, stage, way, at_rest, best.deepest if best else np.inf)
        if got is None:
            continue
        if not got.cuts and not got.lost:
            return [], {}
        if not got.cuts:
            unread = unread or got
        elif best is None or got.deepest < best.deepest:
            best = got
    if unread is not None:
        return [], unread.lost
    if best is None:  # no way to try
        return [], {}
    return ([(item, depth, limit, mover, first, best.direction)
             for item, (depth, limit, first) in best.cuts.items()], best.lost)


def _sweep(view: PieceView, only: Collection[str] | None
           ) -> tuple[list[tuple[str, float, float, str, float, np.ndarray]], list[tuple[str, str]]]:
    """`([(item, deepest, limit, mover, first share, way)], [(item, why)])`: each group pair that
    cuts along the least deep way of a group that has no clear way, and each group pair whose
    thickness cannot be read where a part pair overlaps (`crossings.measure` raises
    `Unmeasurable`). Such a pair gets no hit, only the `Unmeasured` item: a verdict on part of
    its pairs would hide the rest. With `only`, just the groups whose items are named are swept,
    whole (a group's way out depends on every group left on the piece)."""
    lay = _layout(view)
    if lay is None:
        return [], []
    rest_depth: dict[tuple[int, int], float | Unmeasurable] = {}

    def at_rest(i: int, j: int) -> float:
        if (i, j) not in rest_depth:
            try:
                rest_depth[i, j] = measure(lay.solids[i], lay.solids[j]).depth
            except Unmeasurable as err:
                rest_depth[i, j] = err
        got = rest_depth[i, j]
        if isinstance(got, Unmeasurable):
            raise got
        return got

    hits, lost = [], {}
    for stage, mover in enumerate(lay.order[:-1]):
        if only is not None and not any(
                item_of(view, mover, other) in only for other in lay.order[stage + 1:]):
            continue
        got, why = _leaves(view, lay, stage, at_rest)
        hits += got
        lost |= why
    return ([h for h in hits if only is None or h[0] in only],
            [(item, why) for item, why in lost.items() if only is None or item in only])


def _cage(lay: _Layout, mover: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Six slabs, as (low corner, high corner), that tile a shell around the box of the mover's
    parts: a gap clear of the box (`CAGE_GAP_OF_GROUP` of its least extent) and `CAGE_WALL_OF_PIECE`
    of the piece size thick."""
    mine = lay.stage == lay.order.index(mover)
    lo, hi = lay.lo[mine].min(axis=0), lay.hi[mine].max(axis=0)
    centre, inner = (lo + hi) / 2, (hi - lo) / 2 + CAGE_GAP_OF_GROUP * float((hi - lo).min())
    outer = inner + CAGE_WALL_OF_PIECE * lay.size
    slabs = []
    for axis in range(3):  # x slabs span the whole shell; y and z slabs only what the x slabs leave
        for sign in (1.0, -1.0):
            low, high = -outer.copy(), outer.copy()
            for other in range(axis):
                low[other], high[other] = -inner[other], inner[other]
            low[axis], high[axis] = sorted((sign * inner[axis], sign * outer[axis]))
            slabs.append((centre + low, centre + high))
    return slabs


def _group_centre(lay: _Layout, group: str) -> np.ndarray:
    return pieces.centroid(np.concatenate([p.tris for p in lay.parts if p.group == group]))


def _unit(ctx: SceneCtx, view: PieceView, lay: _Layout) -> PlantUnit:
    """The piece's plant unit: the group with the least estimated sample count (its sweeps hold
    the fewest points, and a boxed group is swept in all 27 ways), boxed in by a cage. The cage
    is a group of its own, built first, so it is the last to come off and stays on the piece for
    every group. Its item is new, so no real pair of the piece (one that fails already, or cannot
    be read) stands in for it. Read from the layout alone; `_sweep` is never called."""
    def samples(group: str) -> float:
        return sum(lay.solids[k].sample_estimate
                   for k, p in enumerate(lay.parts) if p.group == group)

    mover = min(lay.order, key=lambda g: (samples(g), lay.order.index(g)))
    donor = next(p for p in lay.parts if p.group == mover)
    record = next(r for r in view.piece.parts if r.name == donor.name)
    taken = [p.name for p in view.parts]
    group = pieces.free_name(group_order(view.piece), f"{mover}_cage")
    geometry = {p.name: p.tris for p in view.parts}
    records = []
    for k, (lo, hi) in enumerate(_cage(lay, mover)):
        name = pieces.free_name(taken, f"{donor.name}_cage{k}")
        taken.append(name)
        geometry[name] = pieces.box_mesh(lo, hi)
        records.append(dataclasses.replace(record, name=name, group=group, on=None, retains=(),
                                           rests_on=(), fastener=False))
    piece = dataclasses.replace(view.piece, parts=(*records, *view.piece.parts))
    planted = pieces.with_geometry(pieces.with_piece(ctx, view.pack, view.name, piece), view,
                                   geometry)
    item = item_of(view, mover, group)
    return PlantUnit(item, (Plant(f"boxed in: {item}", item, planted),))


class TakeApart:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    scope = "piece"
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._seen: tuple[SceneCtx, dict[str, tuple[list, list]]] | None = None

    def _all(self, ctx: SceneCtx) -> dict[str, tuple[list, list]]:
        """Every piece's full sweep, kept for the scene it was made for, so the run and the
        coverage lines sweep each piece once."""
        if self._seen is None or self._seen[0] is not ctx:
            self._seen = (ctx, {v.key: _sweep(v, None)
                                for v in pieces.measured_pieces(ctx, self.kinds)})
        return self._seen[1]

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units = []
        for view in pieces.measured_pieces(ctx, self.kinds):
            lay = _layout(view)
            if lay is not None:
                units.append(_unit(ctx, view, lay))
        return units

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The swept parts (solid, with triangles and a group) of each piece in two or more
        groups. A piece in one group has nothing to take apart."""
        out: set[str] = set()
        for view in pieces.measured_pieces(ctx, self.kinds):
            lay = _layout(view)
            if lay is not None:
                out |= {item_of(view, p.name) for p in lay.parts}
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for view in pieces.measured_pieces(ctx, self.kinds, only):
            hits = self._all(ctx)[view.key][0] if only is None else _sweep(view, only)[0]
            found += [(depth, Finding(
                NAME, item, f"groups cut through each other while {mover} moves out along "
                            f"{_vector(way)}, the least deep way of those tried, from share "
                            f"{share:.2f} of the piece size, deepest {depth:.4f} "
                            f"(limit {limit:.4f})"))
                      for item, depth, limit, mover, share, way in hits]
        return [f for _, f in sorted(found, key=lambda row: (-row[0], row[1].item))]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        views = list(pieces.measured_pieces(ctx, self.kinds))
        ungrouped = [Unmeasured(f"{NAME}: skipped (no group): {v.key} {p.name}")
                     for v in views for p in v.parts if p.solid and p.group == NO_GROUP]
        cables = [f"{NAME}: not applicable (cable part, it bends: it neither moves nor blocks): "
                  f"{v.key} {p.name}" for v in views for p in v.parts if p.solid and p.bends]
        unmeasurable = [Unmeasured(f"{NAME}: skipped ({why}): {v.key} {name}")
                        for v in views if not _one_group(v)
                        for name, why in made_of(_rigid(v)).skipped.items()]
        lost = [Unmeasured(f"{NAME}: skipped ({why}): {item}")
                for v in views for item, why in sorted(self._all(ctx)[v.key][1])]
        needs = {v.key for v in views if _layout(v) is not None}
        lone = [f"{NAME}: not applicable (fewer than two swept groups): {v.key}"
                for v in views if v.key not in needs]
        return (pieces.coverage(ctx, NAME, self.kinds, printed=True) + ungrouped + cables
                + unmeasurable + lost + lone)
