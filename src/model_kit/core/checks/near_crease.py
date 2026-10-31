"""Check near-crease: a drawn line follows a real edge.

The runtime outline draws an edge when the angle between its two faces (its turn) is over the
style's one crease angle (`dials.edges.crease_deg`, `Style.crease_deg`). Two faults follow from
that single angle, and both break the rule that a line follows a whole edge:

- **A, an ambiguous edge.** An edge whose turn is within reach of the crease angle can be drawn
  in one build and not in the next. The reach is the edge's band (`core.edges.rounding_band_deg`):
  how far the position rounding of the shipped file can move the turn, from the dihedral angle's
  gradient. The step comes from `measured.export` (`core.edges.position_step`); a piece with no
  `measured.export` cannot be measured and is `Unmeasured`. On a part bound to more than one
  bone the turn is also measured at every clip sample of its rig, with the rest pose as the
  base: an edge fails when its drawn state differs between two of them, or when a sample brings
  it within its band. A rigid part does not bend and is measured at rest only.
- **B, a dangling dash.** Join the drawn edges of a part into lines. A line end is a vertex
  with exactly one drawn edge that is not on an open boundary (a leaf card's rim has one). It
  fails at any angle: a line that stops in the middle of a smooth surface is not an edge. It is
  measured at rest and, on a part bound to more than one bone, at every clip sample. It is judged
  only as it will be seen (a design call of this kit), by the piece's `view`. A line end is
  dropped, and counted in a coverage line, when its stub is shorter than the eye resolves at
  `view.nearest_m` (`sight.seen_floor`, one arcminute). The stub is the run of drawn edges from
  the end to the next junction or the other end of the line. A line end is dropped too when its
  stub is hidden: every one of its edges has both faces on a side the view lists as `hidden`
  (an edge between a hidden face and a seen face draws the silhouette and still counts). A piece
  with no `view` is judged as before, with no floor and no line. The floor only drops findings:
  rule A, the band and the crease angle are not touched.

An edge is a pair of faces after topology's weld (`WELD_OF_PIECE_SIZE` times the piece size). A
band that rounding alone would widen past `CREASE_BAND_CAP_OF_ANGLE` times the crease angle
belongs to a sliver face, which topology's zero-area rule owns: the edge is not
called a near crease, and the part gets one `sliver` finding.

Parts that skip their outline (`Part.skips_outline`) and cables (`Part.bends`) draw no outline,
so they cannot dash and are listed in coverage. The style draws no outline: not
applicable. A style that declares `borders_only` (outlines and no fold lines, with a
`borders_why`) gets a not-applicable line that gives the reason. A style that declares neither
`crease_deg` nor `borders_only` says nothing about its lines: `Unmeasured`.

A piece that declares `outlined: false` (with an `outlined_why`) has a shipping look that draws no
outline on it, so no line can dash or flip: the check skips the piece with a not-applicable line
that gives the reason. A piece that says nothing is outlined and judged in full.

Items: `<pack>:<piece> <part> edge <i> near` (rest), `... edge <i> bends across` (clip samples),
`... sliver` (once per part), `... line end <v>`. `i` numbers the edges of the part's mesh in the
order `core.edges.interior_edges` gives them, and `v` its welded vertices.

Plants, one unit per rule per piece, built from the piece's manifest and meshes and never from this
check's verdict or its drop rule. Rule A at rest folds the piece's own part: in the first part with
an edge whose band is not cut, the edge nearest the crease angle goes to `FOLD_IN_BAND` of its band
over the angle, and the unit names it again by its two ends. Rule B stands on two small fixture
parts added to the piece, each a unit of its own: a flat plate with one raised vertex of three
edges, whose spokes turn as far over the crease angle as the ring around them turns under it, and
a closed tube bent a little over the angle at its middle ring, whose junction ring draws over part
of its length (a partial ring step). Each fixture is turned toward a side the view shows and sized
so that its stub is over what the eye resolves. Rule A bending is a real skinned part: its edge
nearest the crease angle at rest is folded across the angle at one clip sample (the first, middle
or last), whatever the shipped precision. A unit none of whose candidates the check catches stops
the gate with `BlindCheck`, and so does a rule that judges an item no plant can be built for: its
unit has no candidate. A unit is not applicable only when nothing is judged: no part has an
interior edge or an edge whose band is not cut (rule A at rest), or the view lists all six sides
as hidden (rule B).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Iterator
from dataclasses import dataclass

import numpy as np

from model_kit.core import kinds as recipe_kinds
from model_kit.core.checks import pieces
from model_kit.core.checks.detail import faces_on_sides
from model_kit.core.checks.pieces import (
    PartMesh,
    PieceView,
    Sight,
    hidden_line,
    item_of,
    under_floor_line,
    view_sight,
)
from model_kit.core.edges import (
    EdgeSet,
    interior_edges,
    line_ends,
    position_step,
    rounding_band_deg,
    turns_deg,
)
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import ClipSamples, RigRest, SceneCtx, piece_key
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS, WELD_OF_PIECE_SIZE

NAME = "near-crease"
PRINCIPLE = ("A drawn line follows a real edge: it runs from corner to corner where two faces "
             "meet, never stops in the middle of a smooth surface, and never comes and goes "
             "because rounding or bending moves an angle.")
SOURCE = (
    "A drawn line follows a real edge (a design call of this kit). Supporting evidence: drafting leaves "
    "tangent (smooth-join) edges out of orthographic views (Ford, Engineering Graphics and "
    "Design, https://uw.pressbooks.pub/enggraphics/chapter/orthographic-projection/) though "
    "pictorial views often show them (SolidWorks help, https://help.solidworks.com/2017/english/"
    "Solidworks/sldworks/c_tangent_edge_display_overview.htm). The band is the gradient of the "
    "dihedral angle (Bridson, Marino and Fedkiw 2003, Section 4, "
    "https://www.cs.ubc.ca/~rbridson/docs/cloth2003.pdf).")

# The bending plant starts from the edges of a skinned part whose turn at rest is nearest the crease
# angle: the move is small, and the rest of the part stays as it was. It tries this many.
PLANT_EDGES = 12
# The ring-step fixture bends a tube by one of these shares of the crease angle, in this order, and
# keeps the first at which a drawn arc of its junction ring stops short of the ring's far side.
BEND_OF_CREASE = (1.25, 1.5, 2.0, 3.0)
# The plant puts an edge this share of its band over the crease angle: inside the band, so the
# edge fails, and clear of the crease so the plant is not on the line itself.
FOLD_IN_BAND = CALIBRATION_PASS


def _exempt(part: PartMesh) -> str | None:
    """What kind of part this is, when it draws no outline."""
    return part.skipped_as if part.skips_outline or part.bends else None


# --- one mesh --------------------------------------------------------------------------------


@dataclass(eq=False)
class _Mesh:
    """A part's mesh read once: its edges, which of them draw, where lines stop and, when the
    shipped precision is known, each edge's band and the edges whose band had to be cut."""

    edges: EdgeSet
    drawn: np.ndarray
    ends: np.ndarray
    step: np.ndarray | None
    band: np.ndarray
    capped: np.ndarray
    hidden: np.ndarray  # (E,) both faces of the edge look toward a side the view never shows
    # `hidden` is read once, from the rest faces, and applied unchanged to every clip sample of a
    # skinned part. A face that turns toward a hidden side in a clip is still judged by its rest
    # side. It can only drop a line end, never add a finding, so it only quiets. The cost is a
    # missed finding: a face hidden at rest that turns into view during a clip has its line end
    # dropped.


def _unseen_edges(tris: np.ndarray, edges: EdgeSet, hidden: tuple[str, ...]) -> np.ndarray:
    """(E,) which edges have both faces looking toward a side the view never shows."""
    if not hidden or not len(edges):
        return np.zeros(len(edges), dtype=bool)
    return faces_on_sides(tris, hidden)[edges.faces].all(axis=1)


def _dropped(length: float, run: list[int], unseen: np.ndarray, sight: Sight) -> str | None:
    """Why a viewer would not see a line end whose stub is `run` and `length` long: "unseen" when
    every edge of the stub lies on hidden sides, "under" when it is shorter than the eye resolves,
    None when it is seen. The one rule, for the check and for the plants that must pick a seen end."""
    if unseen[run].all():
        return "unseen"
    return "under" if length < sight.floor else None


def _mesh(tris: np.ndarray, size: float, piece, crease: float,
          hidden: tuple[str, ...] = ()) -> _Mesh:
    edges = interior_edges(tris, WELD_OF_PIECE_SIZE * size)
    drawn = edges.turn_deg > crease
    step = position_step(piece, edges.points) if len(edges.points) else None
    band, capped = (rounding_band_deg(edges.points, edges, step, crease) if step is not None
                    else (np.zeros(len(edges)), np.zeros(len(edges), dtype=bool)))
    unseen = _unseen_edges(tris, edges, hidden)
    return _Mesh(edges, drawn, line_ends(edges, drawn, edges.open_boundary), step, band, capped,
                 unseen)


def _near(turn: np.ndarray, band: np.ndarray, capped: np.ndarray, crease: float) -> np.ndarray:
    """The edges whose turn is within their own band of the crease angle."""
    return (np.abs(turn - crease) <= band) & ~capped


def _hidden_of(view: PieceView) -> tuple[str, ...]:
    """The sides the piece's view lists as never seen (none without a view)."""
    return () if view.piece.view is None else view.piece.view.hidden


def _stubs(edges: EdgeSet, drawn: np.ndarray, ends: np.ndarray
           ) -> dict[int, tuple[float, list[int]]]:
    """For each line end, the stub it stands on: its length and its edges. The stub runs from the
    end along the drawn edges while each vertex it reaches has exactly two, and stops at a
    junction or at the other end of the line."""
    incident: dict[int, list[int]] = {}
    for e in np.flatnonzero(drawn):
        for v in edges.v[e]:
            incident.setdefault(int(v), []).append(int(e))
    out: dict[int, tuple[float, list[int]]] = {}
    for end in ends:
        at, length, run = int(end), 0.0, []
        while True:
            e = next(x for x in incident[at] if not run or x != run[-1])
            a, b = int(edges.v[e, 0]), int(edges.v[e, 1])
            length += float(np.linalg.norm(edges.points[a] - edges.points[b]))
            run.append(e)
            at = b if a == at else a
            if len(incident[at]) != 2:
                break
        out[int(end)] = (length, run)
    return out


def _seen_ends(item: str, edges: EdgeSet, drawn: np.ndarray, unseen: np.ndarray,
               ends: np.ndarray, sight: Sight | None) -> list[int]:
    """The line ends a viewer would see: all of them without a `sight`. With one, an end is
    dropped when its stub lies wholly on hidden sides, or is shorter than the eye resolves."""
    if sight is None or not len(ends):
        return [int(v) for v in ends]
    kept: list[int] = []
    for v, (length, run) in _stubs(edges, drawn, ends).items():
        name = f"{item} line end {v}"
        why = _dropped(length, run, unseen, sight)
        if why == "unseen":
            sight.unseen.add(name)
        elif why == "under":
            sight.under.add(name)
        else:
            sight.shown.add(name)
            kept.append(v)
    return kept


def _rest_findings(item: str, mesh: _Mesh, crease: float, sight: Sight | None = None
                   ) -> list[Finding]:
    out: list[Finding] = []
    edges = mesh.edges
    if mesh.step is not None:
        if mesh.capped.any():
            out.append(Finding(NAME, f"{item} sliver", (
                f"{int(mesh.capped.sum())} edges have a band over "
                f"{mesh.band[mesh.capped][0]:g} deg, wider than rounding explains for a face "
                "of ordinary height: a sliver face, for topology and the zero-area rule")))
        for i in np.flatnonzero(_near(edges.turn_deg, mesh.band, mesh.capped, crease)):
            out.append(Finding(NAME, f"{item} edge {i} near", (
                f"turn {edges.turn_deg[i]:.2f} deg is within {mesh.band[i]:.2f} deg of the "
                f"crease angle {crease:g}: rounding of the shipped file can flip it")))
    for v in _seen_ends(item, edges, mesh.drawn, mesh.hidden, mesh.ends, sight):
        out.append(Finding(NAME, f"{item} line end {v}", (
            f"a drawn line stops at {np.round(edges.points[v], 4).tolist()}, where one edge "
            f"over {crease:g} deg meets none; it dashes across a smooth surface")))
    return out


# --- a skinned part through the clips --------------------------------------------------------


def _bone_count(rest: RigRest, part: str) -> int:
    """How many bones a part is bound to: those with weight over some vertex, or the number of
    deform groups when the rest file holds no weights."""
    weights = rest.weights.get(part)
    groups = rest.weight_groups.get(part)
    if weights is None or groups is None:
        return len(rest.vertex_groups[part])
    return len(np.unique(np.asarray(groups)[np.asarray(weights) > 0]))


def _bends(rest: RigRest, part: str) -> bool:
    return part in rest.vertex_groups and _bone_count(rest, part) > 1


def _clips_of(ctx: SceneCtx, key: str, part: str) -> list[ClipSamples]:
    return [ctx.samples[clip] for clip, rig in ctx.rig_of_clip.items()
            if rig == key and clip in ctx.samples and part in ctx.samples[clip].shells]


@dataclass(eq=False)
class _Bend:
    """A skinned part's rest mesh and the welded vertices' place in the sample meshes."""

    mesh: _Mesh
    rep: np.ndarray  # (V,) a vertex of the rest shell for each welded vertex


def _bend_mesh(rest: RigRest, part: str, size: float, piece, crease: float,
               hidden: tuple[str, ...] = ()) -> _Bend:
    """The topology of the rest shell, with `rep` mapping each welded vertex to a shell vertex,
    so that a sample's points can be read at the same edges with no new weld."""
    faces = rest.faces[part].astype(int)
    tris = rest.shells[part][0].astype(float)[faces]
    mesh = _mesh(tris, size, piece, crease, hidden)
    rep = np.zeros(len(mesh.edges.points), dtype=int)
    rep[mesh.edges.corners.ravel()] = faces.ravel()
    return _Bend(mesh, rep)


def _skin(rest: RigRest, part: str, size: float) -> tuple[EdgeSet, np.ndarray]:
    """The edges of a skinned part's rest shell and, for each welded vertex, a vertex of the
    shell: the numbering `_bend_mesh` gives, read with no verdict."""
    faces = rest.faces[part].astype(int)
    edges = interior_edges(rest.shells[part][0].astype(float)[faces], WELD_OF_PIECE_SIZE * size)
    rep = np.zeros(len(edges.points), dtype=int)
    rep[edges.corners.ravel()] = faces.ravel()
    return edges, rep


def _bend_findings(item: str, bend: _Bend, clips: list[ClipSamples], part: str, crease: float,
                   samples: frozenset[int], sight: Sight | None = None) -> list[Finding]:
    mesh, edges = bend.mesh, bend.mesh.edges
    turns: list[np.ndarray] = [edges.turn_deg]
    within = np.zeros(len(edges), dtype=bool)
    ends: dict[int, str] = {}
    for clip in clips:
        data = clip.shells[part]
        for s in range(len(data)):
            if samples and s not in samples:
                continue
            moved = edges.moved(data[s].astype(float)[bend.rep])
            turns.append(moved.turn_deg)
            drawn = moved.turn_deg > crease
            stops = line_ends(moved, drawn, moved.open_boundary)
            for v in _seen_ends(item, moved, drawn, mesh.hidden, stops, sight):
                ends.setdefault(int(v), f"{clip.clip} sample {s}")
            if mesh.step is not None:
                band, capped = rounding_band_deg(moved.points, moved, mesh.step, crease)
                within |= _near(moved.turn_deg, band, capped, crease)
    out: list[Finding] = []
    stack = np.array(turns)
    states = stack > crease
    crosses = states.any(axis=0) & ~states.all(axis=0)
    at_rest = _near(edges.turn_deg, mesh.band, mesh.capped, crease)
    flagged = (crosses | (within & ~at_rest)) if mesh.step is not None else crosses
    # a rest edge already caught as capped is a sliver's, and a crossing needs no band
    flagged &= crosses | ~mesh.capped
    for i in np.flatnonzero(flagged):
        how = ("turns from " f"{stack[:, i].min():.2f} to {stack[:, i].max():.2f} deg over the "
               f"clip samples, across the crease angle {crease:g}" if crosses[i] else
               f"comes within its {mesh.band[i]:.2f} deg band of the crease angle {crease:g} "
               "at a clip sample")
        out.append(Finding(NAME, f"{item} edge {i} bends across", how))
    for v, where in ends.items():
        out.append(Finding(NAME, f"{item} line end {v}", (
            f"a drawn line stops at {np.round(edges.points[v], 4).tolist()} at {where}: the "
            f"bend turns an edge over {crease:g} deg and leaves its neighbour under it")))
    return out


# --- planting ---------------------------------------------------------------------------------


def _folded_point(edges: EdgeSet, e: int, target_deg: float, side: int = 1
                  ) -> np.ndarray | None:
    """Where the wing of face `side` (0 or 1) of edge `e` must stand, turned about the edge, for
    the edge's turn to be `target_deg`, or None when no turn gives it. Turning a face about the
    edge it shares changes the edge's turn by exactly the angle turned."""
    points = edges.points
    a, b = points[edges.v[e, 0]], points[edges.v[e, 1]]
    axis = (b - a) / np.linalg.norm(b - a)
    rel = points[edges.wings[e, side]] - a
    delta = math.radians(target_deg - float(edges.turn_deg[e]))
    for sign in (1.0, -1.0):
        turn = sign * delta
        new = a + (rel * math.cos(turn) + np.cross(axis, rel) * math.sin(turn)
                   + axis * (axis @ rel) * (1.0 - math.cos(turn)))
        trial = points.copy()
        trial[edges.wings[e, side]] = new
        got = float(turns_deg(trial, edges.v[e:e + 1], edges.wings[e:e + 1])[0])
        if abs(got - target_deg) < 1e-6:
            return new
    return None


# Rule B's fixtures are small parts added to a piece, so a plant never depends on a real part's
# mesh. `_plate` is a flat triangle with an inner triangle and one raised vertex at its centre;
# `_bent_tube` is a closed tube bent at its middle ring.

_SIDES = (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]),
          np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]))


def _spoke_turn_deg(tilt_deg: float) -> float:
    """The turn between neighbouring faces of a vertex of three faces, each tilted `tilt_deg` from
    the flat, their normals 120 degrees apart round it: cos(turn) = cos^2(tilt) - sin^2(tilt) / 2."""
    t = math.radians(tilt_deg)
    return math.degrees(math.acos(math.cos(t) ** 2 - 0.5 * math.sin(t) ** 2))


def _balanced_tilt(crease: float) -> float:
    """The tilt of the three faces round a raised vertex at which the spoke edges turn as far over
    the crease angle as the ring edges (turn = the tilt) turn under it."""
    lo, hi = 0.0, crease
    for _ in range(60):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if _spoke_turn_deg(mid) + mid < 2.0 * crease else (lo, mid)
    return (lo + hi) / 2.0


def _plate(outer: float, crease: float) -> tuple[np.ndarray, np.ndarray]:
    """A flat triangle of circumradius `outer` in the xy plane (normal +z), an inner triangle of
    half that radius and the vertex at the centre raised so that its three faces tilt
    `_balanced_tilt(crease)`. Returns the triangles (9, 3, 3) and the raised vertex."""
    angles = np.radians(90.0 + 120.0 * np.arange(3))
    ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(3)], axis=1)
    c, n = outer * ring, 0.5 * outer * ring
    # a face runs from the centre to an inner side, whose foot is the inner triangle's inradius:
    # half its circumradius
    height = 0.5 * (0.5 * outer) * math.tan(math.radians(_balanced_tilt(crease)))
    top = np.array([0.0, 0.0, height])
    tris = []
    for i in range(3):
        j = (i + 1) % 3
        tris += [[top, n[i], n[j]], [c[i], c[j], n[j]], [c[i], n[j], n[i]]]
    return np.array(tris), top


def _turned_to(tris: np.ndarray, toward: np.ndarray) -> np.ndarray:
    """`tris` turned so that +z points along the unit vector `toward` (a proper rotation)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, toward)
    if not axis.any():
        flip = np.diag([1.0, -1.0, -1.0]) if toward[2] < 0 else np.eye(3)
        return tris @ flip.T
    axis /= np.linalg.norm(axis)
    cos, sin = float(z @ toward), math.sqrt(1.0 - float(z @ toward) ** 2)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return tris @ (np.eye(3) + sin * cross + (1.0 - cos) * cross @ cross).T


def _centre_of(view: PieceView) -> np.ndarray:
    """The middle of the box of all the piece's parts."""
    every = np.concatenate([p.tris.reshape(-1, 3) for p in view.parts if len(p.tris)])
    return (every.max(axis=0) + every.min(axis=0)) / 2.0


def _index_at(points: np.ndarray, at: np.ndarray) -> int:
    """The welded vertex nearest the point `at`."""
    return int(np.linalg.norm(points - at, axis=1).argmin())


def _find_edge(edges: EdgeSet, a: np.ndarray, b: np.ndarray) -> int | None:
    """The edge whose ends are the points `a` and `b`, in either order, or None."""
    p, v = edges.points, edges.v

    def ends_at(i: int, q: np.ndarray) -> np.ndarray:
        return np.abs(p[v[:, i]] - q).max(axis=1) <= 0.0

    hit = np.flatnonzero((ends_at(0, a) & ends_at(1, b)) | (ends_at(0, b) & ends_at(1, a)))
    return int(hit[0]) if len(hit) else None


def _bent_tube(radius: float, sides: int, bend_deg: float) -> np.ndarray:
    """A closed tube of `sides` sides along y: a straight section from y = -2 radius to 0, then a
    section of the same length whose axis is turned `bend_deg` about z, and a cap on each end. The
    ring at y = 0 is shared by both sections, where they meet: its edges turn most where they
    face the bend and least at right angles to it, so with a bend a little over the crease angle
    only part of the ring is drawn and its lines stop where the turn falls under the angle (a
    ring step that is only partly drawn). Wound outward."""
    angle = 2.0 * math.pi * np.arange(sides) / sides
    ring = np.stack([radius * np.cos(angle), np.zeros(sides), radius * np.sin(angle)], axis=1)
    up = np.array([0.0, 2.0 * radius, 0.0])
    b = math.radians(bend_deg)
    turn = np.array([[math.cos(b), -math.sin(b), 0.0], [math.sin(b), math.cos(b), 0.0],
                     [0.0, 0.0, 1.0]])
    rings = [ring - up, ring, (ring + up) @ turn.T]
    centres = [-up, up @ turn.T]
    tris = []
    for lo, hi in ((rings[0], rings[1]), (rings[1], rings[2])):
        for i in range(sides):
            j = (i + 1) % sides
            tris += [[lo[i], hi[j], lo[j]], [lo[i], hi[i], hi[j]]]
    for i in range(sides):
        j = (i + 1) % sides
        tris += [[centres[0], rings[0][i], rings[0][j]], [centres[1], rings[2][j], rings[2][i]]]
    return np.array(tris)


# --- the check --------------------------------------------------------------------------------


class NearCrease:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly", "branching"})
    principle = PRINCIPLE
    source = SOURCE

    def __init__(self) -> None:
        self._told: tuple[SceneCtx, dict[str, Sight]] | None = None

    # -- what to measure

    @staticmethod
    def _crease(ctx: SceneCtx) -> float | None:
        """The crease angle, or None when the style draws no outline or names no angle."""
        angle = ctx.style.crease_deg
        return angle if angle is not None and ctx.style.draws_outline else None

    def _outlined(self, ctx: SceneCtx, only: Collection[str] | None = None
                  ) -> Iterator[PieceView]:
        """The pieces to judge: those whose shipping look draws an outline (the default)."""
        return (v for v in pieces.measured_pieces(ctx, self.kinds, only) if v.piece.outlined)

    def _parts(self, ctx: SceneCtx, only: Collection[str] | None
               ) -> Iterator[tuple[PieceView, PartMesh, str]]:
        for view in self._outlined(ctx, only):
            for part in view.parts:
                item = item_of(view, part.name)
                if _exempt(part) is not None or not len(part.tris):
                    continue
                if only is not None and not any(i.startswith(f"{item} ") for i in only):
                    continue
                yield view, part, item

    # -- the check

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        crease = self._crease(ctx)
        if crease is None:
            return []
        found: list[Finding] = []
        sights: dict[str, Sight | None] = {}
        for view, part, item in self._parts(ctx, only):
            if view.key not in sights:
                sights[view.key] = view_sight(ctx, view)
            found += self._part_findings(ctx, view, part, item, crease, samples,
                                         sights[view.key])
        seen: set[str] = set()
        out = []
        for f in found:
            if f.item not in seen and (only is None or f.item in only):
                seen.add(f.item)
                out.append(f)
        return out

    def _part_findings(self, ctx: SceneCtx, view: PieceView, part: PartMesh, item: str,
                       crease: float, samples: frozenset[int], sight: Sight | None = None
                       ) -> list[Finding]:
        """One part's findings: at rest, then through the clips. A run scoped to samples reads
        the clips only. `sight` is how the piece will be seen (None for a piece with no view)."""
        found: list[Finding] = []
        if not samples:
            mesh = _mesh(part.tris, view.size, view.piece, crease, _hidden_of(view))
            found += _rest_findings(item, mesh, crease, sight)
        return found + self._bending(ctx, view, part, item, crease, samples, sight)

    def _bending(self, ctx: SceneCtx, view: PieceView, part: PartMesh, item: str, crease: float,
                 samples: frozenset[int], sight: Sight | None = None) -> list[Finding]:
        key = piece_key(view.pack, view.name)
        rest = ctx.rests.get(key)
        if rest is None or not _bends(rest, part.name):
            return []
        clips = _clips_of(ctx, key, part.name)
        if not clips:
            return []
        bend = _bend_mesh(rest, part.name, view.size, view.piece, crease, _hidden_of(view))
        return _bend_findings(item, bend, clips, part.name, crease, samples, sight)

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part with triangles whose edges it judges: none when the style names no crease
        angle or draws no outline, and not the cable, decal and no-outline parts."""
        if self._crease(ctx) is None:
            return set()
        return {item for _, _, item in self._parts(ctx, None)}

    # -- coverage

    def coverage(self, ctx: SceneCtx) -> list[str]:
        if ctx.style.borders_only:
            line = (f"{NAME}: not applicable (the style draws borders only, so no fold line can "
                    f"dash or flip: {ctx.style.borders_why})")
            return [line]
        if ctx.style.crease_deg is None:
            return [Unmeasured(f"{NAME}: skipped (the style has no crease_deg in dials.edges, "
                               "so there is no angle to measure edges against)")]
        if not ctx.style.draws_outline:
            return [f"{NAME}: not applicable (the style draws no outline)"]
        lines = self._unlined(ctx)
        for view in pieces.measured_pieces(ctx, self.kinds):
            if not view.piece.outlined:
                continue
            lines += [f"{NAME}: not applicable ({kind} part): {view.key} {p.name}"
                      for p in view.parts if (kind := _exempt(p)) is not None]
            if "export" not in (view.piece.measured or {}) and any(
                    _exempt(p) is None and len(p.tris) for p in view.parts):
                lines.append(Unmeasured(
                    f"{NAME}: skipped (no measured.export, so the rounding of the shipped "
                    f"file is unknown and rule A cannot run): {view.key}"))
            lines += self._bending_gaps(ctx, view)
            lines += self._floor_lines(ctx, view)
        return list(dict.fromkeys(lines))

    def _unlined(self, ctx: SceneCtx) -> list[str]:
        """The pieces' lines that do not depend on a piece's parts: those of the other recipe
        kinds and of a piece with no geometry, and for a piece declared `outlined: false` the
        not-applicable line that gives the reason, whether or not it has geometry (it draws no
        outline, so a missing mesh costs nothing)."""
        unlined = {piece_key(pack, name): piece
                   for pack, name, piece in recipe_kinds.pieces_of_kind(ctx, self.kinds)
                   if not piece.outlined}
        gaps = {Unmeasured(f"{NAME}: skipped (no geometry): {key}") for key in unlined}
        lines = [x for x in pieces.coverage(ctx, NAME, self.kinds) if x not in gaps]
        return lines + [f"{NAME}: not applicable (the piece's shipping look draws no outline: "
                        f"{piece.outlined_why}): {key}" for key, piece in unlined.items()]

    def _sights(self, ctx: SceneCtx) -> dict[str, Sight]:
        """For each piece with a `view`, what rule B dropped on it: one unscoped pass over every
        part, kept for the scene it was made for (a scoped `run` would miss the rest)."""
        if self._told is None or self._told[0] is not ctx:
            out: dict[str, Sight] = {}
            crease = self._crease(ctx)
            if crease is not None:
                for view in self._outlined(ctx):
                    if (sight := view_sight(ctx, view)) is not None:
                        out[view.key] = sight
                for view, part, item in self._parts(ctx, None):
                    if view.key in out:
                        self._part_findings(ctx, view, part, item, crease, frozenset(),
                                            out[view.key])
            self._told = (ctx, out)
        return self._told[1]

    def _floor_lines(self, ctx: SceneCtx, view: PieceView) -> list[str]:
        """The coverage lines of the seen floor: how many line ends it dropped for being under
        what the eye resolves, and how many for lying on hidden sides. None for a piece with
        no view."""
        sight = self._sights(ctx).get(view.key)
        if sight is None:
            return []
        return [under_floor_line(NAME, sight, view.piece, view.key),
                *hidden_line(NAME, sight, view.piece, view.key, "line ends")]

    def _bending_gaps(self, ctx: SceneCtx, view: PieceView) -> list[str]:
        if view.piece.rig is None:
            return []
        rest = ctx.rests.get(view.key)
        if rest is None:
            return [Unmeasured(f"{NAME}: skipped (no rest file, so the bones that bind each "
                               f"part are unknown): {view.key}")]
        return [Unmeasured(f"{NAME}: skipped (no clip samples of the rig, so bending is not "
                           f"measured): {view.key} {p.name}")
                for p in view.parts if _exempt(p) is None and len(p.tris)
                and _bends(rest, p.name) and not _clips_of(ctx, view.key, p.name)]

    # -- plants

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        crease = self._crease(ctx)
        if crease is None:
            return []
        units: list[PlantUnit] = []
        for view in self._outlined(ctx):
            units += self._plant_piece(ctx, view, crease)
        return units

    def _plant_piece(self, ctx: SceneCtx, view: PieceView, crease: float) -> list[PlantUnit]:
        measured = [p for p in view.parts if _exempt(p) is None and len(p.tris)]
        if not measured:
            return []
        units = []
        if "export" in (view.piece.measured or {}):
            units.append(self._near_unit(ctx, view, measured, crease))
        units += self._dash_units(ctx, view, crease)
        rest = ctx.rests.get(view.key)
        if rest is not None:
            bend = self._bend_unit(ctx, view, measured, rest, crease)
            if bend is not None:
                units.append(bend)
        return units

    @staticmethod
    def _with_fixture(ctx: SceneCtx, view: PieceView, stem: str, tris: np.ndarray
                      ) -> tuple[str, SceneCtx]:
        """The scene with a part of `tris` added to the piece, and the part's name."""
        name = pieces.free_name({p.name for p in view.parts}, stem)
        geometry = {p.name: p.tris for p in view.parts} | {name: tris}
        return name, pieces.with_geometry(ctx, view, geometry)

    def _near_unit(self, ctx: SceneCtx, view: PieceView, measured: list[PartMesh], crease: float
                   ) -> PlantUnit:
        """Rule A at rest, on the piece's own parts: in the first part with an edge whose band is
        not cut, the edge whose turn is nearest the crease angle (and outside its band), folded
        to `FOLD_IN_BAND` of its band over it and found again in the folded mesh by its two ends.
        The candidates fold the second face's wing and then the first's. A fold that cannot be
        named passes to the next of the part's nearest edges, then to the next part. When a part
        is judged and none can be folded, the unit names the nearest judged edge and has no
        candidate, so the gate stops. It is not applicable only when nothing is judged: no part
        has an interior edge, or every edge of every part has its band cut."""
        weld = WELD_OF_PIECE_SIZE * view.size
        has_edges, unplanted = False, ""
        for part in measured:
            edges = interior_edges(part.tris, weld)
            if not len(edges):
                continue
            has_edges = True
            band, capped = rounding_band_deg(edges.points, edges,
                                             position_step(view.piece, edges.points), crease)
            ok = np.flatnonzero(~capped)
            gap = np.abs(edges.turn_deg[ok] - crease)
            # nearest the angle first; an edge already in its band would be a real finding of the
            # same item, which the gate skips, so the edges outside their bands go first
            order = ok[np.lexsort((gap, gap <= band[ok]))]
            if len(order) and not unplanted:
                unplanted = f"{item_of(view, part.name)} edge {int(order[0])} near"
            for e in order[:PLANT_EDGES]:
                unit = self._fold_unit(ctx, view, part, edges, int(e), float(band[e]), crease)
                if unit is not None:
                    return unit
        if unplanted:
            return PlantUnit(unplanted, ())
        return PlantUnit(view.key, (), not_applicable=(
            "no edge is judged: " + ("every edge of every part has its band cut, the shipped "
                                     "precision being too coarse to call an edge near the "
                                     "crease angle" if has_edges else
                                     "no part has an interior edge")))

    def _fold_unit(self, ctx: SceneCtx, view: PieceView, part: PartMesh, edges: EdgeSet, e: int,
                   band_e: float, crease: float) -> PlantUnit | None:
        """The unit for folding edge `e` of `part`, or None when no fold of it can be named."""
        weld = WELD_OF_PIECE_SIZE * view.size
        a, b = edges.points[edges.v[e, 0]], edges.points[edges.v[e, 1]]
        item, candidates = "", []
        for side in (1, 0):
            wing = edges.wings[e, side]
            target, new = crease + FOLD_IN_BAND * band_e, None
            for _ in range(2):  # the band changes as the face turns: settle it once
                new = _folded_point(edges, e, target, side)
                if new is None:
                    break
                points = edges.points.copy()
                points[wing] = new
                moved = edges.moved(points)
                band, _ = rounding_band_deg(points, moved, position_step(view.piece, points),
                                            crease)
                target = crease + FOLD_IN_BAND * float(band[e])
            if new is None:
                continue
            folded = part.tris.copy()
            folded[edges.corners == wing] = new
            found = interior_edges(folded, weld)
            j = _find_edge(found, a, b)
            if j is None:
                continue
            band, capped = rounding_band_deg(found.points, found,
                                             position_step(view.piece, found.points), crease)
            name = f"{item_of(view, part.name)} edge {j} near"
            if capped[j] or (item and name != item):
                continue
            item = name
            geometry = {p.name: p.tris for p in view.parts} | {part.name: folded}
            candidates.append(Plant(f"edge folded into its band: {name}", name,
                                    pieces.with_geometry(ctx, view, geometry)))
        return PlantUnit(item, tuple(candidates)) if candidates else None

    def _dash_units(self, ctx: SceneCtx, view: PieceView, crease: float) -> list[PlantUnit]:
        """Rule B on two fixtures, each a unit of its own (they name different parts and
        vertices, so they cannot be candidates of one item): a plate with one raised vertex of
        three edges (its three spokes turn over the crease angle, its ring under it, so each
        spoke ends at an inner corner), and a bent tube whose junction ring is drawn over part
        of its length (a partial ring step). Each is turned toward a side the view shows and
        sized so that a stub is over what the eye resolves. The line end is picked from the
        fixture's own construction and the view's `hidden` list, never from the check's stub or
        drop rule. Only a view that lists all six sides as hidden makes the unit not applicable;
        any other fixture with no seen end has no candidate, so the gate stops."""
        if len(set(_hidden_of(view))) == len(_SIDES):
            return [PlantUnit(view.key, (), not_applicable=(
                "the view lists every side as hidden, so no line end can be seen"))]
        sight = view_sight(ctx, view)
        floor = 0.0 if sight is None else sight.floor
        outer = max(CALIBRATION_PASS * view.size, CALIBRATION_FAIL * floor / CALIBRATION_PASS)
        sides = max(3, math.ceil(360.0 / (CALIBRATION_PASS * crease)))
        # every edge of the tube, ring edges included, is over what the eye resolves
        radius = max(CALIBRATION_PASS * outer,
                     CALIBRATION_FAIL * floor * sides / (2.0 * math.pi))
        return [self._plate_unit(ctx, view, _plate(outer, crease)[0]),
                self._tube_unit(ctx, view, [_bent_tube(radius, sides, share * crease)
                                            for share in BEND_OF_CREASE], crease)]

    def _plate_unit(self, ctx: SceneCtx, view: PieceView, plate: np.ndarray) -> PlantUnit:
        """The plate's unit: its inner corners are the line ends, each on one spoke between two
        faces of the fan (faces `3k` and `3(k-1)` of the plate); the first corner, on the first
        side to turn the plate toward, with a face that is not on a hidden side."""
        hidden, centre = _hidden_of(view), _centre_of(view)
        for toward in _SIDES:
            tris = _turned_to(plate, toward) + centre
            on = faces_on_sides(tris, hidden) if hidden else np.zeros(len(tris), dtype=bool)
            for k in range(3):
                if on[3 * k] and on[3 * ((k - 1) % 3)]:
                    continue
                points = interior_edges(tris, WELD_OF_PIECE_SIZE * view.size).points
                return self._fixture_unit(ctx, view, "dash", tris,
                                          _index_at(points, tris[3 * k, 1]))
        return self._fixture_unit(ctx, view, "dash", plate, None)

    def _tube_unit(self, ctx: SceneCtx, view: PieceView, tubes: list[np.ndarray], crease: float
                   ) -> PlantUnit:
        """The bent tube's unit: of the first tube (the bends in order) and the first side to
        turn it toward at which a vertex has exactly one drawn edge, off an open boundary, and
        that edge has a face that is not on a hidden side, the lowest such vertex.

        Known limit: the plant tests only the stub's one drawn edge against the hidden sides,
        while the check drops a stub only when every edge of it is hidden. So a crease near 80
        deg with five sides hidden can give an empty unit (a BlindCheck) where the check would
        still see an end. More candidate tubes (another `sides` or bend) would close it."""
        hidden, centre = _hidden_of(view), _centre_of(view)
        for tube in tubes:
            for toward in _SIDES:
                tris = _turned_to(tube, toward) + centre
                edges = interior_edges(tris, WELD_OF_PIECE_SIZE * view.size)
                drawn = np.flatnonzero(edges.turn_deg > crease)
                count = np.bincount(edges.v[drawn].ravel(), minlength=len(edges.points))
                on = faces_on_sides(tris, hidden) if hidden else np.zeros(len(tris), dtype=bool)
                for e in drawn:
                    for v in edges.v[e]:
                        if count[v] == 1 and not edges.open_boundary[v] and not on[
                                edges.faces[e]].all():
                            return self._fixture_unit(ctx, view, "ring", tris, int(v))
        return self._fixture_unit(ctx, view, "ring", tubes[0], None)

    def _fixture_unit(self, ctx: SceneCtx, view: PieceView, stem: str, tris: np.ndarray,
                      end: int | None) -> PlantUnit:
        """A unit planting `tris` as a part of the piece with the line end `end`; with no end the
        unit has no candidate, so the gate stops."""
        name, planted = self._with_fixture(ctx, view, stem, tris)
        if end is None:
            return PlantUnit(f"{item_of(view, name)} line end", ())
        item = f"{item_of(view, name)} line end {end}"
        return PlantUnit(item, (Plant(f"a drawn line cut short: {item}", item, planted),))

    def _bend_unit(self, ctx: SceneCtx, view: PieceView, measured: list[PartMesh],
                   rest: RigRest, crease: float) -> PlantUnit | None:
        """Rule A bending: on the first skinned part that has clip samples and an edge that can be
        folded, the edge whose rest turn is nearest the crease angle, folded at one clip sample to
        `CALIBRATION_PASS` of the crease angle on the other side of it. Whatever the shipped
        precision, the drawn state then differs between rest and that sample. The three
        candidates are the first, middle and last sample of the first clip. When no skinned part
        can be folded the unit has no candidate, so the gate stops, and it names the nearest edge
        of the first such part."""
        unplanted = ""
        for part in measured:
            if not _bends(rest, part.name):
                continue
            clips = _clips_of(ctx, view.key, part.name)
            if not clips:
                continue
            edges, rep = _skin(rest, part.name, view.size)
            if not len(edges):
                continue
            item = item_of(view, part.name)
            faces = rest.faces[part.name].astype(int)
            clip = clips[0]
            data = clip.shells[part.name]
            gap = np.abs(edges.turn_deg - crease)
            nearest = np.argsort(gap, kind="stable")[:PLANT_EDGES]
            unplanted = unplanted or f"{item} edge {int(nearest[0])} bends across"
            for e in nearest:
                e = int(e)
                name = f"{item} edge {e} bends across"
                over = edges.turn_deg[e] <= crease
                target = crease * ((1.0 + CALIBRATION_PASS) if over else (1.0 - CALIBRATION_PASS))
                candidates = []
                for s in dict.fromkeys([0, len(data) // 2, len(data) - 1]):
                    at = edges.moved(data[s].astype(float)[rep])
                    new = _folded_point(at, e, target)
                    if new is None:
                        continue
                    shells = np.array(data)
                    shells[s][faces[edges.corners == edges.wings[e, 1]]] = new
                    moved = dataclasses.replace(clip, shells={**clip.shells, part.name: shells})
                    candidates.append(Plant(
                        f"edge bent across the crease angle at sample {s}: {name}", name,
                        dataclasses.replace(ctx, samples={**ctx.samples, clip.clip: moved}),
                        samples=frozenset({s})))
                if candidates:
                    return PlantUnit(name, tuple(candidates))
        return PlantUnit(unplanted, ()) if unplanted else None
