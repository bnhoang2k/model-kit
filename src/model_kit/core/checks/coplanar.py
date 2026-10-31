"""Check coplanar: two parts, or two placements, never draw a face in the same depth.

A perspective depth buffer stores depth in steps that grow with the square of the distance from
the eye. Two faces that face the same way, overlap and lie closer than one step round to the
same stored depth, so the renderer picks between them by float noise and they flicker. The step
at a distance is `units.depth_step`: it comes from the style's reference camera (`near_m`,
`far_m`, `depth_bits`), and the coarsest step is at the farthest distance a piece is seen from
(`View.farthest_m`). That step, in the pack's units, is the **gap**. No number is chosen.

Two items, in every piece of a kind in `kinds` (assembly):

- **piece** `<pack>:<piece> <a> x <b> flicker` (parts sorted). A face of part `a` and a face of
  part `b` that face the same way, overlap, and lie within the gap of one plane over the whole
  overlap. The gap is the piece's own.
- **layout** `<layout> <a> x <b> flicker` (placement ids sorted). The same, between a part of
  one placement and a part of the other, at their placed transforms. A pair of placements takes
  the larger of the two pieces' `farthest_m`, because the coarser depth step decides whether the
  two faces separate. A joined pair (`on`, `rests_on`) is judged too: a rug resting on a floor
  flickers when their faces share a plane.

The test is a pairwise z-fighting test, without bins. Each pair of triangles of two
different parts whose unit normals agree to `COPLANAR_NORMAL_COS` (0.9999, so a
face that crosses another steeply is not "the same plane": its overlap is a sliver along the
seam) is clipped against the other, in the projection along the
first one's widest normal axis. The planes must lie within the gap of each other at every corner
of the overlap (the distance between planes is linear, so its largest value is at a corner).
Binning planes by a rounded normal would miss a pair whose normals straddle a
rounding edge, so a bounding box and a plane-side test find the candidates and the clip decides.

A patch the eye could not resolve is not a flicker anyone sees. The overlap of a pair of parts,
summed over all its face pairs, must cover more than one resolvable spot: the square of the
piece's contact length (`sight.contact_length`, one arcminute at `view.nearest_m`). A pair of
placements uses the smaller of the two pieces' lengths, because the nearer view sees finer. This
replaces a fixed area of 4 mm^2.

A patch no one can see is not a flicker anyone sees. A face pair is left out when one closed solid
part other than the two (of the same piece, or of either placement) hides the whole overlap on both
faces, each read on its own plane (they can lie a depth step apart): the overlap's corners and its
centre each lie inside that part, or on its surface with the face turned into it (the part's outward
normal there points against the face's, as a foot pressed onto its plate). A patch partly in the
open stays a finding. The principle is the definition of visibility under a depth test: a fragment
behind a nearer surface is discarded, and a face inside a closed part has that part's surface in
front of it from every viewpoint outside the part. No source is quoted for it; it is not the check's
measure, only the reason a hidden face is not judged. A part that is not closed hides nothing.

A piece without a `farthest_m`, or a style without a depth camera, reads `Unmeasured`: there is
no depth step to judge by. A piece of another recipe kind (a tree) is not judged. A piece with
fewer than two parts that have triangles has no pair; a layout with fewer than two measurable
placements has none. Each says so in a plain line.

Plants, one unit per piece with two parts that have triangles, and one per layout with two
measurable placements, built from the geometry alone (never from the gap or a verdict). Piece:
the largest triangle of one part is copied exactly, into the plane it already lies in, into the
other part; the other direction is the second candidate. A copy in the same plane is closer than
any gap above 0. Layout: one placement is copied under a new id at the same position and turn,
so every face of it lies in the plane of its original.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from model_kit.core import kinds as recipe_kinds
from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, PieceView, item_of
from model_kit.core.geom import NO_DIRECTION, PAIR_BLOCK, TIE_DISTANCE, SurfaceIndex, is_closed
from model_kit.core.manifest import Layout, LayoutPlacement, Placement
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx, piece_key
from model_kit.core.sight import depth_gap
from model_kit.core.tolerances import COPLANAR_NORMAL_COS

NAME = "coplanar"
KINDS = frozenset({"assembly"})
PRINCIPLE = ("A perspective depth buffer resolves depth in steps that grow with the square of the "
             "distance, so two faces that face the same way, overlap and lie closer than one step "
             "draw in the same depth and flicker.")
SOURCE = (
    "Khronos, 'Depth Buffer Precision', OpenGL Wiki: eye depth z_e relates to the stored depth "
    "z_w by z_e / w_e = f n / ((z_w / s) (f - n) - f), with near n, far f and s the buffer's "
    "scale, so one stored step spans z^2 (f - n) / (f n s) of eye distance "
    "(https://wikis.khronos.org/opengl/Depth_Buffer_Precision; an archived copy is at "
    "https://web.archive.org/web/2024/https://www.khronos.org/opengl/wiki/"
    "Depth_Buffer_Precision). `units.depth_step` derives the step. The face test is a "
    "pairwise z-fighting test; an earlier form of it used a fixed 0.1 mm gap. "
    "A face hidden inside a closed part "
    "is not judged because it is never visible (the definition of visibility under a depth test: "
    "a fragment behind a nearer surface is discarded); no source was opened and quoted for that "
    "step, and none is claimed.")


# --- one part's faces -------------------------------------------------------------------------


@dataclass(eq=False)
class _Faces:
    """The triangles of one part that have a face, in one frame: unit normals, plane offsets
    (normal . first corner) and boxes."""

    name: str
    tris: np.ndarray  # (N, 3, 3)
    owner: str = ""  # the placement the part belongs to; "" for a part of a piece
    normal: np.ndarray = field(init=False)
    offset: np.ndarray = field(init=False)
    lo: np.ndarray = field(init=False)
    hi: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        tris = np.asarray(self.tris, dtype=float).reshape(-1, 3, 3)
        cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        length = np.linalg.norm(cross, axis=1)
        keep = length > NO_DIRECTION
        self.tris = tris[keep]
        self.normal = cross[keep] / length[keep, None]
        self.offset = (self.normal * self.tris[:, 0]).sum(axis=1)
        self.lo, self.hi = self.tris.min(axis=1), self.tris.max(axis=1)

    def __len__(self) -> int:
        return len(self.tris)

    @cached_property
    def box(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lo.min(axis=0), self.hi.max(axis=0)

    @property
    def largest(self) -> np.ndarray:
        """The triangle with the biggest area."""
        cross = np.cross(self.tris[:, 1] - self.tris[:, 0], self.tris[:, 2] - self.tris[:, 0])
        return self.tris[int(np.linalg.norm(cross, axis=1).argmax())]


def _faces(part: PartMesh) -> _Faces | None:
    faces = _Faces(part.name, part.tris)
    return faces if len(faces) else None


@dataclass(eq=False)
class _Occluder:
    """A closed solid part in one frame, which hides a face that lies inside it."""

    name: str
    owner: str  # the placement it belongs to; "" for a part of a piece
    tris: np.ndarray  # (N, 3, 3), wound outward

    def __post_init__(self) -> None:
        points = self.tris.reshape(-1, 3)
        self.lo, self.hi = points.min(axis=0), points.max(axis=0)

    @cached_property
    def index(self) -> SurfaceIndex:
        return SurfaceIndex(self.tris)

    def hides(self, points: np.ndarray, normal: np.ndarray) -> bool:
        """Whether every point lies where a viewer cannot see a face that faces `normal`: inside
        the part, or on its surface with the face turned into it (the part's outward normal there
        points against the face's), as a foot is pressed onto its plate."""
        if ((points < self.lo - TIE_DISTANCE) | (points > self.hi + TIE_DISTANCE)).any():
            return False
        near = self.index.closest(points)
        on = near.dist <= TIE_DISTANCE
        flush = on & (near.normal @ normal < 0.0)
        return bool((flush | (~on & self.index.inside(points))).all())


def _buried(a_points: np.ndarray, a_normal: np.ndarray, b_points: np.ndarray,
            b_normal: np.ndarray, occluders: Sequence[_Occluder]) -> bool:
    """Whether one closed part hides the whole patch on both faces: a face behind a nearer surface
    fails the depth test and is never drawn. A flicker is seen only where at least one of the two
    faces is seen, so the pair is left out only when a single part hides both. Each face is read
    on its own plane with its own normal, so the verdict does not depend on which comes first."""
    return any(o.hides(a_points, a_normal) and o.hides(b_points, b_normal) for o in occluders)


def _clip(subject: list, clipper: list) -> list:
    """Sutherland-Hodgman: the 2D polygon `subject` cut by the convex, counter-clockwise polygon
    `clipper`."""
    out = subject
    for i, a in enumerate(clipper):
        b = clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break

        def side(p, a=a, b=b):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        for j, p in enumerate(inp):
            q = inp[(j + 1) % len(inp)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                out.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out


def _area(poly) -> float:
    """Signed area of a 2D polygon (positive when counter-clockwise)."""
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                     - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))


def _shared_plane(ta, na, da, tb, nb, db, gap: float) -> tuple[float, np.ndarray]:
    """(the area, in the plane of triangle `a`, where triangles `a` and `b` overlap and the two
    planes lie within `gap` of each other all over it, the sample points of that patch) or
    `(0.0, empty)` when they do not. The sample points are the corners of the overlap, lifted onto
    the plane of `a`, and their mean (inside the overlap, which is convex). `n` is a unit normal
    and `d` a plane offset (normal . corner). The caller has already required the normals to
    agree (`COPLANAR_NORMAL_COS`)."""
    none = (0.0, np.zeros((0, 3)))
    ax = int(np.abs(na).argmax())
    u, v = (i for i in range(3) if i != ax)
    pa = [(p[u], p[v]) for p in ta]
    pb = [(p[u], p[v]) for p in tb]
    if _area(pa) < 0:
        pa.reverse()
    if _area(pb) < 0:
        pb.reverse()
    overlap = _clip(pb, pa)
    if len(overlap) < 3:
        return none
    shadow = abs(_area(overlap))
    if shadow <= 0.0:
        return none
    # Lift every corner of the overlap onto the plane of `a` and measure its distance to the
    # plane of `b`. The corners of the whole triangles are the wrong place: a small face tilted
    # a hair on a big one is far off at the big one's far corner and near over the small one.
    far = 0.0
    corners = []
    for pu, pv in overlap:
        point = np.zeros(3)
        point[u], point[v] = pu, pv
        point[ax] = (da - na[u] * pu - na[v] * pv) / na[ax]
        corners.append(point)
        far = max(far, abs(float(nb @ point) - db))
    if far > gap:
        return none
    return shadow / abs(float(na[ax])), np.array(corners + [np.mean(corners, axis=0)])


def _candidates(a: _Faces, b: _Faces, gap: float):
    """Index arrays `(i, j)` of the triangle pairs of two parts that could share a plane within
    `gap`: boxes grown by the gap meet, the normals agree (`COPLANAR_NORMAL_COS`), and neither
    triangle lies wholly beyond the gap on one side of the other's plane. A necessary test only;
    `_shared_plane` decides."""
    (alo, ahi), (blo, bhi) = a.box, b.box
    ai = np.flatnonzero(((a.lo <= bhi + gap) & (a.hi >= blo - gap)).all(axis=1))
    bj = np.flatnonzero(((b.lo <= ahi + gap) & (b.hi >= alo - gap)).all(axis=1))
    if not len(ai) or not len(bj):
        return
    step = max(1, PAIR_BLOCK // len(bj))
    for start in range(0, len(ai), step):
        rows = ai[start:start + step]
        meet = ((a.lo[rows, None] <= b.hi[bj][None] + gap)
                & (a.hi[rows, None] >= b.lo[bj][None] - gap)).all(axis=2)
        meet &= (a.normal[rows, None] * b.normal[bj][None]).sum(axis=2) >= COPLANAR_NORMAL_COS
        i, j = np.nonzero(meet)
        i, j = rows[i], bj[j]
        if not len(i):
            continue
        # the corners of b against the plane of a, and of a against the plane of b
        to_a = ((b.tris[j] - a.tris[i][:, :1]) * a.normal[i][:, None]).sum(axis=2)
        to_b = ((a.tris[i] - b.tris[j][:, :1]) * b.normal[j][:, None]).sum(axis=2)
        near = ~((to_a > gap).all(axis=1) | (to_a < -gap).all(axis=1)
                 | (to_b > gap).all(axis=1) | (to_b < -gap).all(axis=1))
        yield i[near], j[near]


def _overlap(a: _Faces, b: _Faces, gap: float, occluders: Sequence[_Occluder] = ()
             ) -> tuple[float, int]:
    """(the area where a face of `a` and a face of `b` face the same way, overlap and lie within
    `gap` of one plane, the number of such face pairs). A face pair whose whole overlap a closed
    part other than the two hides (`_buried`) is left out."""
    total, pairs = 0.0, 0
    for rows, cols in _candidates(a, b, gap):
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            area, points = _shared_plane(a.tris[i], a.normal[i], a.offset[i],
                                         b.tris[j], b.normal[j], b.offset[j], gap)
            if area <= 0.0:
                continue
            # the same patch on the plane of b: its face can lie up to one depth step off a's
            onto_b = points - np.outer(points @ b.normal[j] - b.offset[j], b.normal[j])
            if not _buried(points, a.normal[i], onto_b, b.normal[j], occluders):
                total, pairs = total + area, pairs + 1
    return total, pairs


# --- what a scene can be judged by -------------------------------------------------------------


@dataclass(frozen=True)
class _Reading:
    """What one piece is judged by: the gap (one depth step at its farthest view, in the pack's
    units), its contact length and the farthest view the gap was read at."""

    gap: float
    contact: float
    farthest_m: float


def _unmeasured_why(ctx: SceneCtx, view: PieceView) -> str | None:
    """Why the piece has no depth step, or None when it has one."""
    if ctx.style.depth_camera is None:
        return "the style declares no depth camera, so there is no depth step"
    if view.piece.view is None:
        return "the piece declares no view, so it has no farthest view"
    if view.piece.view.farthest_m is None:
        return "the piece declares no farthest view, so there is no depth step to judge by"
    return None


def _reading(ctx: SceneCtx, view: PieceView) -> _Reading | None:
    if _unmeasured_why(ctx, view) is not None:
        return None
    far = view.piece.view.farthest_m
    upm = ctx.packs[view.pack].units_per_metre
    return _Reading(depth_gap(far, ctx.style, upm), view.contact, far)


def _detail(a: str, b: str, area: float, pairs: int, gap: float, far: float, bits: int) -> str:
    return (f"{pairs} face pair(s) of {a} and {b} face the same way and overlap over "
            f"{area:.3g} square units, within {gap:.3g} of one plane (one depth step of a "
            f"{bits}-bit buffer at {far:g} m): they draw in the same depth and flicker")


def _find(a: _Faces, b: _Faces, gap: float, contact: float, hiders: Sequence[_Occluder] = ()):
    """(overlap area, face pairs) when the two parts share a plane over a patch the eye resolves
    (more than one spot of `contact` a side), else None. `hiders` are the closed parts that may
    hide a patch; the two parts themselves never do."""
    if not len(a) or not len(b):
        return None
    (alo, ahi), (blo, bhi) = a.box, b.box
    if (alo > bhi + gap).any() or (blo > ahi + gap).any():
        return None
    others = [o for o in hiders if (o.owner, o.name) not in ((a.owner, a.name), (b.owner, b.name))]
    area, pairs = _overlap(a, b, gap, others)
    return (area, pairs) if pairs and area > contact * contact else None


def _world(view: PieceView, placement: LayoutPlacement) -> list[_Faces]:
    placed = Placement(placement.id, placement.piece, placement.pack, placement.position,
                       placement.quaternion)
    return [f for p in view.parts if len(p.tris)
            if len(f := _Faces(p.name, placed.to_world(p.tris), placement.id))]


def _occluders(view: PieceView, placement: LayoutPlacement | None = None) -> list[_Occluder]:
    """The closed solid parts of a piece, in its frame or, for a placement, in the layout's.
    Closedness is read on the piece-frame triangles: `is_closed` matches corners by exact
    coordinates, and a turn into the layout's frame rounds them apart."""
    placed = None if placement is None else Placement(
        placement.id, placement.piece, placement.pack, placement.position, placement.quaternion)
    return [_Occluder(p.name, "" if placed is None else placed.id,
                      p.tris if placed is None else placed.to_world(p.tris))
            for p in view.parts if p.solid and len(p.tris) and is_closed(p.tris)]


class _Scan:
    """One scene: the pieces it can judge, and the placements of its layouts."""

    def __init__(self, ctx: SceneCtx) -> None:
        self.ctx = ctx
        self.views = {v.key: v for v in pieces.measured_pieces(ctx, KINDS)}
        self._faces: dict[str, list[_Faces]] = {}
        self._placed: dict[tuple[str, str], list[_Faces]] = {}
        self._hiders: dict[tuple[str, str], list[_Occluder]] = {}

    def faces(self, view: PieceView) -> list[_Faces]:
        if view.key not in self._faces:
            self._faces[view.key] = [f for p in view.parts if (f := _faces(p)) is not None]
        return self._faces[view.key]

    def view_of(self, p: LayoutPlacement) -> PieceView | None:
        return self.views.get(piece_key(p.pack, p.piece))

    def placed(self, layout: Layout, p: LayoutPlacement, view: PieceView) -> list[_Faces]:
        key = (layout.name, p.id)
        if key not in self._placed:
            self._placed[key] = _world(view, p)
        return self._placed[key]

    def hiders(self, view: PieceView, layout: Layout | None = None,
               p: LayoutPlacement | None = None) -> list[_Occluder]:
        """The closed solid parts of a piece, or of a placement of it, that could hide a patch."""
        key = (view.key, "" if layout is None else f"{layout.name} {p.id}")
        if key not in self._hiders:
            self._hiders[key] = _occluders(view, p)
        return self._hiders[key]

    def pair_reading(self, a: PieceView, b: PieceView) -> _Reading | None:
        """The gap of two placed pieces: the larger farthest view of the two (the coarser depth
        step), in the style's units; the contact length is the smaller (the finer sight)."""
        ra, rb = _reading(self.ctx, a), _reading(self.ctx, b)
        if ra is None or rb is None:
            return None
        far = max(ra.farthest_m, rb.farthest_m)
        return _Reading(depth_gap(far, self.ctx.style, self.ctx.style.units_per_metre),
                        min(ra.contact, rb.contact), far)

    def measurable(self, layout: Layout) -> list[str]:
        """The ids of the placements of a judged piece that has a face, in the layout's order."""
        return [i for i, p in layout.placements.items()
                if (v := self.view_of(p)) is not None and self.placed(layout, p, v)]

    def pairs(self, layout: Layout, ids: Collection[str]):
        """Every pair of these placements, ids sorted."""
        ordered = [i for i in layout.placements if i in set(ids)]
        for n, first in enumerate(ordered):
            for second in ordered[n + 1:]:
                yield tuple(sorted((first, second)))

    def item(self, layout: Layout, pair: tuple[str, ...]) -> str:
        return f"{layout.name} {pair[0]} x {pair[1]} flicker"

    # --- findings ---

    def findings(self, only: Collection[str] | None) -> list[Finding]:
        found: list[Finding] = []
        bits = self.ctx.style.depth_camera.depth_bits if self.ctx.style.depth_camera else 0
        for view in pieces.measured_pieces(self.ctx, KINDS, only):
            reading = _reading(self.ctx, view)
            if reading is None:
                continue
            faces = self.faces(view)
            hiders = self.hiders(view)
            for n, a in enumerate(faces):
                for b in faces[n + 1:]:
                    item = f"{item_of(view, a.name, b.name)} flicker"
                    if only is not None and item not in only:
                        continue
                    if (hit := _find(a, b, reading.gap, reading.contact,
                                     hiders)) is not None:
                        found.append(Finding(NAME, item, _detail(
                            a.name, b.name, *hit, reading.gap, reading.farthest_m, bits)))
        for layout in self.ctx.layouts.values():
            for pair in self.pairs(layout, self.measurable(layout)):
                item = self.item(layout, pair)
                if only is not None and item not in only:
                    continue
                pa, pb = (layout.placements[i] for i in pair)
                va, vb = self.view_of(pa), self.view_of(pb)
                reading = self.pair_reading(va, vb)
                if reading is None:
                    continue
                hiders = self.hiders(va, layout, pa) + self.hiders(vb, layout, pb)
                hits = [(a.name, b.name, *hit) for a in self.placed(layout, pa, va)
                        for b in self.placed(layout, pb, vb)
                        if (hit := _find(a, b, reading.gap, reading.contact,
                                         hiders)) is not None]
                if hits:
                    area = sum(h[2] for h in hits)
                    count = sum(h[3] for h in hits)
                    where = ", ".join(f"{pair[0]} {h[0]} with {pair[1]} {h[1]}" for h in hits)
                    found.append(Finding(NAME, item, _detail(
                        f"{pa.piece} ({pair[0]})", f"{pb.piece} ({pair[1]})", area, count,
                        reading.gap, reading.farthest_m, bits) + f": {where}"))
        return found

    # --- plants ---

    def plants(self) -> list[PlantUnit]:
        units: list[PlantUnit] = []
        for view in self.views.values():
            reading = _reading(self.ctx, view)
            faces = self.faces(view)
            if reading is not None and len(faces) >= 2:
                units.append(self._piece_unit(view, faces))
        for layout in self.ctx.layouts.values():
            ids = [i for i in self.measurable(layout)
                   if _reading(self.ctx, self.view_of(layout.placements[i])) is not None]
            if len(ids) >= 2:
                units.append(self._layout_unit(layout, layout.placements[ids[0]]))
        return units

    def _piece_unit(self, view: PieceView, faces: list[_Faces]) -> PlantUnit:
        """The first two parts with faces; each copies its largest triangle, where it lies, into
        the other."""
        first, second = faces[0], faces[1]
        item = f"{item_of(view, first.name, second.name)} flicker"
        geometry = {p.name: p.tris for p in view.parts}
        candidates = []
        for source, target in ((first, second), (second, first)):
            copied = np.concatenate([geometry[target.name], source.largest[None]])
            candidates.append(Plant(
                f"a face of {source.name} copied into {target.name}: {item}", item,
                pieces.with_geometry(self.ctx, view, geometry | {target.name: copied})))
        return PlantUnit(item, tuple(candidates))

    def _layout_unit(self, layout: Layout, placement: LayoutPlacement) -> PlantUnit:
        """A copy of one placement, at the same position and turn under a new id."""
        copy_id = pieces.free_name(set(layout.placements), placement.id)
        copy = dataclasses.replace(placement, id=copy_id, on=None, rests_on=None)
        planted = dataclasses.replace(layout, placements={**layout.placements, copy_id: copy})
        item = self.item(layout, tuple(sorted((placement.id, copy_id))))
        ctx = dataclasses.replace(self.ctx, layouts={**self.ctx.layouts, layout.name: planted})
        return PlantUnit(item, (Plant(f"{placement.id} copied in place as {copy_id}: {item}",
                                      item, ctx),))

    # --- measured ---

    def measured(self) -> set[str]:
        """The parts the check could compare with another: those with a face, of a piece that
        has a depth step and a second part with a face, or of a placement in a layout with two
        placements that have one."""
        out: set[str] = set()
        for view in self.views.values():
            faces = self.faces(view)
            if _reading(self.ctx, view) is not None and len(faces) >= 2:
                out |= {item_of(view, f.name) for f in faces}
        for layout in self.ctx.layouts.values():
            ids = [i for i in self.measurable(layout)
                   if _reading(self.ctx, self.view_of(layout.placements[i])) is not None]
            if len(ids) >= 2:
                for i in ids:
                    view = self.view_of(layout.placements[i])
                    out |= {item_of(view, f.name) for f in self.faces(view)}
        return out

    # --- coverage ---

    def coverage(self) -> list[str]:
        lines: list[str] = []
        for pack, name, _ in recipe_kinds.pieces_of_kind(self.ctx, KINDS):
            key = piece_key(pack, name)
            view = self.views.get(key)
            if view is None:
                continue  # `pieces.coverage` says there is no geometry
            if (why := _unmeasured_why(self.ctx, view)) is not None:
                lines.append(Unmeasured(f"{NAME}: skipped ({why}): {key}"))
            elif len(self.faces(view)) < 2:
                lines.append(f"{NAME}: not applicable (fewer than two parts with a face to "
                             f"compare): {key}")
        if not self.ctx.layouts:
            return lines + [f"{NAME}: not applicable (no layout): nothing to compare placements"]
        for layout in self.ctx.layouts.values():
            lines += self._layout_lines(layout)
        return lines

    def _layout_lines(self, layout: Layout) -> list[str]:
        lines: list[str] = []
        ids = self.measurable(layout)
        for pid in ids:
            view = self.view_of(layout.placements[pid])
            if (why := _unmeasured_why(self.ctx, view)) is not None:
                lines.append(Unmeasured(f"{NAME}: skipped ({why}): {layout.name} {pid}"))
        if len(ids) < 2:
            lines.append(f"{NAME}: not applicable (fewer than two pieces to compare): "
                         f"{layout.name}")
        return lines


class Coplanar:
    name = NAME
    kinds: frozenset[str] = KINDS
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return _Scan(ctx).plants()

    def measured(self, ctx: SceneCtx) -> set[str]:
        return _Scan(ctx).measured()

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return sorted(_Scan(ctx).findings(only), key=lambda f: f.item)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return pieces.coverage(ctx, NAME, self.kinds) + _Scan(ctx).coverage()
