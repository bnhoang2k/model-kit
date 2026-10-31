"""Check inset-seat: a printed or inset part meets its host along one line.

A part of role `eyes` or `decal` is a marking: a change of colour on its host's surface (a print,
an inlay, a painted eye), not an object that stands off it. The check reads each such part and its
`on` host, in every piece of a kind in `kinds` (assembly). It reads the part's fill shape, not an
outline: eyes draw none.

- **Printed.** An open patch, or a closed part with no corner deeper inside the host than the
  piece's contact length (`sight.contact_length`: one arcminute at `view.nearest_m`), is printed.
  Every corner must lie within that distance of the host's surface, or the part `lifts off` it (it
  floats above it or is sunk below it). A decal is built to stand off its host by exactly the
  contact length (`sight.decal_lift`), so a gap equal to it is printed; the build's float32
  coordinates round that gap by up to `COMPARE_TOL`, which the test allows. Item
  `<pack>:<piece> <part> lifts off <host>`.
- **Inset.** A closed part with a corner or a face centre deeper inside a closed host than that
  distance is set into it. Rule 1, loops: it cuts the host's surface in exactly one closed line
  (`geom.cut_loops`), else item `<pack>:<piece> <part> meets <host> in <n> loops`. Rule 2, weave:
  the part's rim is its outline as seen from outside the host. Take the host's outward normal at
  the part (the area-weighted normal of the host faces near it); the rim is the edges between a
  face of the part that looks out (a positive dot with that normal) and one that does not. Each
  connected rim loop is judged alone and weaves when it has corners inside the host by more than
  the contact length and others outside by more than that. The cut is still one loop then, so the
  loop rule cannot see it: item `<pack>:<piece> <part> weaves across <host>`. A box or disc set
  deep with its top proud shows the top's outline, wholly outside, and is quiet. No crease angle
  is read.

A part with no `on` host, a closed part on an open host (inside is undefined), one whose host
faces cancel (no outward normal) and one with no rim edge are `Unmeasured`. So is every part of a
piece with no `view`: it has no contact length, and no rule runs. A piece with no printed part
says so in a plain line.

Plants, one unit per rule per piece that has a part of the right kind, built from the geometry
alone (the part's depth in its host, its rim, the host's normal; never from a finding). The gate
skips a unit whose part already fails that rule. A printed part: half of its rim (the open
boundary; for a closed one, its rim loops) is pushed into the host, or lifted off it where a thin
host lets the corners through, by `CALIBRATION_FAIL` times the contact length (item `lifts off`).
Loops: an inset part is sunk until every corner lies inside the host, so it meets the surface in
no line (item `meets <host> in 0 loops`). Weave: half of an inset part's rim is moved out of the
host and half into it, each `CALIBRATION_FAIL` times the contact length past the surface. A rule
that has parts of its kind and no fault to make for any of them stops the run (`BlindCheck`),
unless the real run reports the rule's item.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core import geom
from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.edges import EdgeSet, interior_edges
from model_kit.core.errors import ManifestError
from model_kit.core.geom import NO_DIRECTION, SurfaceIndex
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    COMPARE_TOL,
    WELD_OF_PIECE_SIZE,
)

NAME = "inset-seat"
PRINCIPLE = ("A part printed on a surface or set into it meets that surface along one closed "
             "line, with its rim wholly on or inside the surface or wholly on or outside it and "
             "never weaving across it.")
SOURCE = (
    "Inset parts seat cleanly (a design call of this kit: a rim half "
    "in and half out of a surface reads as a fault). A print is a film "
    "of microns: pad-printing ink dries to about 4 microns (Inkcups ICN-B150 pad printer manual, "
    "https://tagless.inkcups.com/wp-content/uploads/2017/08/ICN-B150-Manual-Compressed.pdf), so "
    "a printed part lies on its surface. A weave is judged at what the eye resolves, one "
    "arcminute at the piece's nearest viewing distance (a design call of this kit; EyeWiki, Visual "
    "Acuity Testing in Adults, https://eyewiki.aao.org/Visual_Acuity_Testing_in_Adults, "
    "secondary). No source was read for the thickness of an inlay cut flush; the rule is this "
    "kit's own, from its call on eye rims.")


def touch_of(view: PieceView) -> float:
    """The piece's contact length. Only a piece with a `view` reaches here (`contact_pieces`)."""
    if view.contact is None:
        raise ValueError(f"{view.key}: no contact length without a view")
    return view.contact


@dataclass
class _Reading:
    """What the check found for one part, and what it could not measure. `kind` is "printed" or
    "inset" once the part is classed (None when it could not be), and `normal` is the host's
    outward unit normal at an inset part (None when it has none)."""

    findings: list[Finding] = dataclasses.field(default_factory=list)
    gaps: list[str] = dataclasses.field(default_factory=list)
    kind: str | None = None
    normal: np.ndarray | None = None


def _host_of(view: PieceView, part: PartMesh) -> PartMesh | None:
    if part.on is None:
        return None
    host = next((p for p in view.parts if p.name == part.on), None)
    if host is None or len(host.tris) == 0:
        raise ManifestError(f"{view.key} {part.name}: on {part.on!r}, which has no geometry in "
                            f"the scene; the scene has {sorted(p.name for p in view.parts)}")
    return host


def _printed_parts(view: PieceView) -> list[PartMesh]:
    return [p for p in view.parts if not p.solid and len(p.tris)]


# --- one part ---------------------------------------------------------------------------------


def _lifts_off(item: str, host: str, near_dist: np.ndarray, touch: float) -> Finding | None:
    # a gap equal to the touch is printed; float32 rounding of the coordinates is allowed
    over = near_dist > touch + COMPARE_TOL
    if not over.any():
        return None
    return Finding(NAME, f"{item} lifts off {host}", (
        f"{int(over.sum())} of {len(near_dist)} corners are over {touch:.4f} from {host}'s "
        f"surface, the furthest {near_dist.max():.4f} (limit {touch:.4f}): the part floats above "
        "its host or is sunk below it"))


def _loops(item: str, host: str, loops: list[np.ndarray]) -> Finding | None:
    shut = [len(line) > 3 and bool(np.array_equal(line[0], line[-1])) for line in loops]
    if len(loops) == 1 and shut[0]:
        return None
    opened = len(loops) - sum(shut)
    how = f", {opened} of them open" if opened else ""
    n = len(loops)
    return Finding(NAME, f"{item} meets {host} in {n} {'loop' if n == 1 else 'loops'}", (
        f"it cuts {host}'s surface in {n} {'line' if n == 1 else 'lines'}{how}; a part set into "
        "a surface cuts it in exactly one closed loop"))


def _outward(tris: np.ndarray) -> np.ndarray:
    """The triangles wound so that a closed mesh's normals point out of it."""
    return tris[:, ::-1] if geom.signed_volume(tris) < 0 else tris


def _face_normals(tris: np.ndarray) -> np.ndarray:
    return np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])


def _host_normal(part: np.ndarray, host: np.ndarray, hosts: SurfaceIndex) -> np.ndarray | None:
    """The host's outward unit normal where the part sits: the area-weighted normal of the host
    triangles whose boxes meet the part's box, or, when none does (the part lies wholly inside),
    that of the host face nearest the part's centre. None when the faces cancel (a part through
    a plate meets two faces that look opposite ways)."""
    outward = _face_normals(_outward(host)) if geom.is_closed(host) else _face_normals(host)
    lo, hi = part.reshape(-1, 3).min(axis=0), part.reshape(-1, 3).max(axis=0)
    near = ((host.min(axis=1) <= hi) & (host.max(axis=1) >= lo)).all(axis=1)
    if near.any():
        total = outward[near].sum(axis=0)
        area = np.linalg.norm(outward[near], axis=1).sum()
        length = float(np.linalg.norm(total))
        return total / length if length > NO_DIRECTION * area else None
    face = hosts.closest(part.reshape(-1, 3).mean(axis=0, keepdims=True)).face[0]
    length = float(np.linalg.norm(outward[face]))
    return outward[face] / length if length > 0.0 else None


def _rim_loops(tris: np.ndarray, normal: np.ndarray, size: float
               ) -> tuple[EdgeSet, list[np.ndarray]]:
    """The mesh's welded edges and its rim as seen from outside the host: the edges between a
    face that looks out (its normal has a positive dot with the host's outward `normal`) and one
    that does not, grouped into the connected loops they make, each a list of welded vertex ids.
    A box standing proud shows its top face's outline; a lens, the ring where its front meets its
    back."""
    tris = _outward(tris)
    edges = interior_edges(tris, WELD_OF_PIECE_SIZE * size)
    facing = _face_normals(tris) @ normal > 0.0
    rim = edges.v[facing[edges.faces[:, 0]] != facing[edges.faces[:, 1]]]
    parent: dict[int, int] = {}

    def find(v: int) -> int:
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for a, b in rim.tolist():
        parent[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for v in sorted(parent):
        groups.setdefault(find(v), []).append(v)
    return edges, [np.array(g) for g in groups.values()]


def _weave(item: str, host: str, loops: list[np.ndarray], edges: EdgeSet, hosts: SurfaceIndex,
           reach: float) -> Finding | None:
    """A rim loop weaves when some of its corners lie inside the host by more than `reach` and
    others outside it by more than that. Each loop is judged alone."""
    for loop in loops:
        points = edges.points[loop]
        dist = hosts.closest(points).dist
        depth = np.where(hosts.inside(points), dist, -dist)  # positive inside the host
        inward, outward = depth > reach, depth < -reach
        if inward.any() and outward.any():
            return Finding(NAME, f"{item} weaves across {host}", (
                f"{int(inward.sum())} of {len(loop)} corners of a rim loop lie inside {host} by "
                f"more than {reach:.4g} (the deepest {depth.max():.4g}) and {int(outward.sum())} "
                f"outside it by more than that (the furthest {-depth.min():.4g}): the rim, "
                "seen from outside the host, weaves across its surface; one arcminute at the "
                "piece's nearest viewing distance is the reach"))
    return None


def _is_inset(view: PieceView, part: PartMesh, host: PartMesh, hosts: SurfaceIndex,
              corners: np.ndarray) -> bool | None:
    """Whether the part is set into its host (a closed part with a corner or a face centre deeper
    inside the host than the touch distance) or printed on it. None for a closed part on an open
    host, where inside is undefined. Geometry only: it judges nothing."""
    tris = part.tris
    if not geom.is_closed(tris):
        return False
    if not geom.is_closed(host.tris):
        return None
    # a part can pass through its host with no corner inside it, so the face centres count too
    probe = np.concatenate([corners, tris.mean(axis=1)])
    inside = hosts.inside(probe)
    return bool(inside.any() and hosts.closest(probe[inside]).dist.max() > touch_of(view))


def _read(ctx: SceneCtx, view: PieceView, part: PartMesh) -> _Reading:
    out = _Reading()
    item, host = item_of(view, part.name), _host_of(view, part)
    if host is None:
        out.gaps.append(f"no host: {part.name} names no `on` part to be printed on")
        return out
    touch, tris = touch_of(view), part.tris
    hosts = SurfaceIndex(host.tris)
    corners = np.unique(tris.reshape(-1, 3), axis=0)
    near = hosts.closest(corners)
    inset = _is_inset(view, part, host, hosts, corners)
    if inset is None:
        out.gaps.append(f"host {host.name} is not closed, so a closed part cannot be placed "
                        "in or on it")
        return out
    if not inset:
        out.kind = "printed"
        if (found := _lifts_off(item, host.name, near.dist, touch)) is not None:
            out.findings.append(found)
        return out
    out.kind = "inset"
    if (found := _loops(item, host.name, geom.cut_loops(
            tris, hosts, weld=WELD_OF_PIECE_SIZE * view.size))) is not None:
        out.findings.append(found)
    out.normal = _host_normal(tris, host.tris, hosts)
    reach = touch
    if out.normal is None:
        out.gaps.append(f"no outward direction: the faces of {host.name} that meet {part.name} "
                        "cancel, so its rim cannot be found")
        return out
    edges, loops = _rim_loops(tris, out.normal, view.size)
    if not loops:
        out.gaps.append(f"no rim: no edge of {part.name} separates its faces that look out of "
                        f"{host.name} from the rest")
    elif (found := _weave(item, host.name, loops, edges, hosts, reach)) is not None:
        out.findings.append(found)
    return out


# --- planting ---------------------------------------------------------------------------------

# The most steps the loops plant takes to sink a part under a curved host's surface.
SINK_STEPS = 8
RULES = ("printed", "loops", "weave")
"""The three rules a plant can fail: a printed part's lift, an inset part's loops, its rim's weave."""


def _rule_of(item: str) -> str:
    return ("printed" if " lifts off " in item else "weave" if " weaves across " in item
            else "loops")


def _inward(points: np.ndarray, hosts: SurfaceIndex, host_closed: bool) -> np.ndarray:
    """A unit vector at each point pointing into the host: away from the nearest surface point
    for a point inside, toward it for a point outside (the nearest face's inward normal for a
    point on the surface)."""
    near = hosts.closest(points)
    offset = points - near.point
    length = np.linalg.norm(offset, axis=1)
    on = length < NO_DIRECTION
    away = offset / np.where(on, 1.0, length)[:, None]
    inside = hosts.inside(points) if host_closed else np.zeros(len(points), dtype=bool)
    return np.where(on[:, None], -near.normal, np.where(inside[:, None], away, -away))


def _split(edges: EdgeSet, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Two halves of these welded vertices, one each side of their centre along their widest
    spread, or None when there are fewer than two."""
    if len(ids) < 2:
        return None
    points = edges.points[ids]
    centre = points.mean(axis=0)
    axis = np.linalg.svd(points - centre, full_matrices=False)[2][0]
    along = (points - centre) @ axis
    high = along > np.median(along)
    return (ids[high], ids[~high]) if high.any() and (~high).any() else None


def _rim_ids(tris: np.ndarray, normal: np.ndarray | None, size: float
             ) -> tuple[EdgeSet, np.ndarray]:
    """The mesh's welded edges and the welded vertices of its rim: the open boundary of an open
    part, else the corners of its rim loops seen from outside the host (none without a normal)."""
    edges = interior_edges(tris, WELD_OF_PIECE_SIZE * size)
    if edges.open_boundary.any():
        return edges, np.flatnonzero(edges.open_boundary)
    if normal is None:
        return edges, np.zeros(0, dtype=np.intp)
    edges, loops = _rim_loops(tris, normal, size)
    return edges, (np.unique(np.concatenate(loops)) if loops else np.zeros(0, dtype=np.intp))


def _with_corners(tris: np.ndarray, edges: EdgeSet, ids: np.ndarray, points: np.ndarray
                  ) -> np.ndarray:
    out = tris.copy()
    for vertex, point in zip(ids, points, strict=True):
        out[edges.corners == vertex] = point
    return out


def _planted(ctx: SceneCtx, view: PieceView, part: PartMesh, tris: np.ndarray) -> SceneCtx:
    geometry = {p.name: p.tris for p in view.parts} | {part.name: tris}
    return pieces.with_geometry(ctx, view, geometry)


def _moves(ctx: SceneCtx, view: PieceView, part: PartMesh, host: PartMesh,
           normal: np.ndarray | None, rule: str) -> list[tuple[str, np.ndarray]]:
    """The faulted copies of this part's mesh to try for the rule, as (how, triangles). `normal`
    is the host's outward unit normal at the part, if it has one."""
    hosts, closed = SurfaceIndex(host.tris), geom.is_closed(host.tris)
    if rule == "loops":
        # sink the part along the host's normal until every corner lies inside the host: it then
        # meets the surface along no line. Each step moves it by its furthest outside corner's
        # distance from the surface, and a little over (twice the weld of the cut); a curved
        # host takes a few steps. A part longer than the host is thick (a post through it) is
        # shrunk about its centre first, to a half and then a quarter.
        centre = pieces.centroid(part.tris)
        options = []
        for scale, how in ((1.0, "sunk wholly into"), (0.5, "halved and sunk wholly into"),
                           (0.25, "quartered and sunk wholly into")):
            moved = centre + scale * (part.tris - centre)
            for _ in range(SINK_STEPS if normal is not None else 0):  # no direction, no sinking
                points = moved.reshape(-1, 3)
                outside = ~hosts.inside(points)
                if not outside.any():
                    break
                far = float(hosts.closest(points[outside]).dist.max())
                moved = moved - normal * (far + CALIBRATION_FAIL * WELD_OF_PIECE_SIZE * view.size)
            if hosts.inside(moved.reshape(-1, 3)).all():
                options.append((how, moved))
        return options
    shut = geom.is_closed(part.tris)
    flip = shut and geom.signed_volume(part.tris) < 0
    tris = _outward(part.tris) if shut else part.tris  # the rim reads an outward-wound mesh
    if shut and normal is None:
        normal = _host_normal(part.tris, host.tris, hosts)
    edges, ids = _rim_ids(tris, normal, view.size)
    halves = _split(edges, ids)
    if halves is None:
        return []

    def made(chosen: np.ndarray, sign: float, depth: float) -> np.ndarray:
        near = hosts.closest(edges.points[chosen])
        return near.point + sign * _inward(edges.points[chosen], hosts, closed) * depth

    options: list[tuple[str, np.ndarray]] = []
    if rule == "weave":
        reach = touch_of(view)
        out, into = halves
        both = np.concatenate([out, into])
        there = np.concatenate([made(out, -1.0, CALIBRATION_FAIL * reach),
                                made(into, 1.0, CALIBRATION_FAIL * reach)])
        options.append(("half the rim out of and half into",
                        _with_corners(tris, edges, both, there)))
    else:
        depth = CALIBRATION_FAIL * touch_of(view)
        for sign, how in ((1.0, "half the rim pushed into"), (-1.0, "half the rim lifted off")):
            options.append((how, _with_corners(tris, edges, halves[0],
                                               made(halves[0], sign, depth))))
    return [(how, moved[:, ::-1] if flip else moved) for how, moved in options]


# The item each rule's plant makes the check report, from the part's name and its host's.
_FAULT = {"printed": "{item} lifts off {host}", "loops": "{item} meets {host} in 0 loops",
          "weave": "{item} weaves across {host}"}


def _weave_judged(ctx: SceneCtx, view: PieceView, part: PartMesh, normal: np.ndarray | None
                  ) -> bool:
    """Whether the weave rule can judge this inset part: the host has an outward direction there,
    and the part has a rim. Otherwise `_read` leaves a gap (`Unmeasured`) in place of a verdict.
    Read from the geometry."""
    return normal is not None and bool(_rim_loops(part.tris, normal, view.size)[1])


def _unit(ctx: SceneCtx, view: PieceView, rule: str) -> PlantUnit | None:
    """The unit for one rule: on the first part of the right kind that has a fault to make, every
    fault that could make the check report that rule's item on it. A part of the right kind with
    no fault to make still gives its unit, with no candidates: the gate skips it when the real
    run reports the item and stops the run when not. None when the piece has no part of the
    right kind. Read from the geometry alone."""
    want = "printed" if rule == "printed" else "inset"
    first = None  # the unit of the first part of the right kind, if no part gives a candidate
    for part in _printed_parts(view):
        host = _host_of(view, part)
        if host is None:
            continue
        hosts = SurfaceIndex(host.tris)
        inset = _is_inset(view, part, host, hosts, np.unique(part.tris.reshape(-1, 3), axis=0))
        if inset is None or ("inset" if inset else "printed") != want:
            continue
        normal = _host_normal(part.tris, host.tris, hosts) if inset else None
        if rule == "weave" and not _weave_judged(ctx, view, part, normal):
            continue  # the check reports this rule `Unmeasured` for the part: nothing to prove
        item = item_of(view, part.name)
        fault = _FAULT[rule].format(item=item, host=host.name)
        planted = [Plant(f"{how} {host.name}: {item}", fault, _planted(ctx, view, part, moved))
                   for how, moved in _moves(ctx, view, part, host, normal, rule)]
        if planted:
            return PlantUnit(fault, tuple(planted))
        first = first or PlantUnit(fault, ())
    return first


# --- the check --------------------------------------------------------------------------------


class InsetSeat:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    principle = PRINCIPLE
    source = SOURCE

    def _readings(self, ctx: SceneCtx, only: Collection[str] | None):
        """`(view, part, reading)` for each printed part the scope names. A scope item names a
        part when it starts with the part's item and a space."""
        for view in pieces.contact_pieces(ctx, self.kinds, only):
            for part in _printed_parts(view):
                prefix = f"{item_of(view, part.name)} "
                if only is None or any(i.startswith(prefix) for i in only):
                    yield view, part, _read(ctx, view, part)

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        """One unit per rule per piece that has a part of the right kind to plant."""
        return [unit for view in pieces.contact_pieces(ctx, self.kinds)
                for rule in RULES if (unit := _unit(ctx, view, rule)) is not None]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every printed part with triangles of a measured piece."""
        return {item_of(view, p.name) for view in pieces.contact_pieces(ctx, self.kinds)
                for p in _printed_parts(view)}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for _, _, reading in self._readings(ctx, only) for f in reading.findings]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = pieces.coverage(ctx, NAME, self.kinds)
        lines += pieces.unviewed(ctx, NAME, self.kinds)
        for view in pieces.contact_pieces(ctx, self.kinds):
            if not _printed_parts(view):
                lines.append(f"{NAME}: not applicable (no part printed on a surface): "
                             f"{view.key}")
                continue
            for part in _printed_parts(view):
                lines += [Unmeasured(f"{NAME}: skipped ({why}): {item_of(view, part.name)}")
                          for why in _read(ctx, view, part).gaps]
            lines += [f"{NAME}: skipped (no triangles): {item_of(view, p.name)}"
                      for p in view.parts if not p.solid and not len(p.tris)]
        return lines
