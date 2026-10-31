"""Edges of a triangle mesh: where two faces meet, how far they turn, and how far that turn can
move.

The runtime outline draws an edge when the angle between its two face normals (its turn, in
degrees) is over the style's crease angle. This module is what the near-crease check reads, and
what later checks read to ask how sure an edge's angle is.

- `interior_edges(tris, weld)` welds a triangle soup (`geom.weld`) and returns its interior
  edges: an edge on exactly two faces, both with an area, wound the same way round the mesh.
  Everything else (an open edge, an edge on three faces, a flipped neighbour, an edge beside a
  face with no area) is not an interior edge. Topology reports those faults, and the vertices on
  them are flagged in `EdgeSet.open_boundary` so that a drawn line may end there.
- `rounding_band_deg` is how far the turn of each edge can move when its four corners each move
  by the worst rounding of the shipped file, from the gradient of the dihedral angle.
- `position_step` is that rounding for a piece: a Draco step or a float32 spacing.
- `line_ends` finds the vertices where a drawn line stops.
- `sagittas` is the gap between two flat faces and the curve they stand for, and `merged_gaps`
  the gap left if the edge were removed; the detail check judges both by what the eye resolves.
  `segments_for` is the same inequality turned round: the fewest segments a curve needs for that
  gap to stay under what the eye resolves, which the grower meshes its tubes by.
- `float32_step` is the least rounding a turn carries, for the edges that are flat to float32.

Row ids: `EdgeSet.faces` hold row indexes of the input triangles, and `EdgeSet.corners` the
welded vertex ids of every input row. A mesh that keeps its rows (a skinned part at a clip
sample) can therefore be read again at other positions with `EdgeSet.moved`, with no new weld.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from model_kit.core import geom
from model_kit.core.errors import ManifestError
from model_kit.core.tolerances import CREASE_BAND_CAP_OF_ANGLE


@dataclass(frozen=True, eq=False)
class EdgeSet:
    """The interior edges of one mesh, in the order of their welded vertex ids (low id, then
    high id), so a given mesh always numbers them the same way.

    `v` (E, 2) are the welded ids of the edge's ends and `wings` (E, 2) the welded ids of the
    third corner of each of its two faces: the first face runs `v[0]` to `v[1]`, the second runs
    back. `faces` (E, 2) are row indexes into the input triangles. `turn_deg` (E,) is the angle
    between the two faces' normals, 0 for a flat join and 180 for a fold back on itself.

    The rest describe the mesh the edges came from: `points` (V, 3) are the welded vertices,
    `corners` (T, 3) the welded ids of every input row and `open_boundary` (V,) marks a vertex on
    an edge that is not an interior edge.
    """

    v: np.ndarray
    wings: np.ndarray
    faces: np.ndarray
    turn_deg: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.intp))
    open_boundary: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    def __len__(self) -> int:
        return len(self.v)

    def moved(self, points: np.ndarray) -> EdgeSet:
        """The same edges with the welded vertices at `points` (V, 3): the turns are measured
        again, the topology is kept."""
        points = np.asarray(points, dtype=float)
        return EdgeSet(self.v, self.wings, self.faces, turns_deg(points, self.v, self.wings),
                       points, self.corners, self.open_boundary)


def _normals(points: np.ndarray, v: np.ndarray, wings: np.ndarray
             ) -> tuple[np.ndarray, np.ndarray]:
    """The two face normals of each edge, not unit: `N1 = (x1 - x3) x (x1 - x4)` and
    `N2 = (x2 - x4) x (x2 - x3)`, with x3, x4 the edge's ends and x1, x2 the faces' wings (the
    order of Bridson, Marino and Fedkiw 2003, Section 4). For a mesh wound one way round, both
    point out of the same side."""
    x3, x4 = points[v[:, 0]], points[v[:, 1]]
    x1, x2 = points[wings[:, 0]], points[wings[:, 1]]
    return np.cross(x1 - x3, x1 - x4), np.cross(x2 - x4, x2 - x3)


def turns_deg(points: np.ndarray, v: np.ndarray, wings: np.ndarray) -> np.ndarray:
    """The turn of each edge in degrees, from the angle between its two normals."""
    n1, n2 = _normals(points, v, wings)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(n1, n2), axis=1),
                                 np.einsum("ij,ij->i", n1, n2)))


def interior_edges(tris: np.ndarray, weld: float) -> EdgeSet:
    """The interior edges of the triangle soup `tris` (N, 3, 3) after welding corners within
    `weld` (`geom.weld`; topology's tolerance is `WELD_OF_PIECE_SIZE` times the piece size).

    A row with a repeated corner after the weld is left out of the count, as topology leaves it
    out. A row whose normal has no length has no direction, so an edge beside one is not an
    interior edge. An edge on two faces that run it the same way round has a flipped neighbour:
    the turn would be 180 minus the true one, so it is not an interior edge either."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    if len(tris) == 0:
        return EdgeSet(np.zeros((0, 2), np.intp), np.zeros((0, 2), np.intp),
                       np.zeros((0, 2), np.intp), np.zeros(0))
    points, corners = geom.weld(tris, weld)
    n_points = len(points)
    boundary = np.zeros(n_points, dtype=bool)
    rows = np.flatnonzero(~((corners[:, 0] == corners[:, 1]) | (corners[:, 1] == corners[:, 2])
                            | (corners[:, 2] == corners[:, 0])))
    a, b, c = (corners[rows, k] for k in range(3))
    # every directed edge of the kept rows, the corner across from it, and the row it is of
    start = np.concatenate([a, b, c])
    end = np.concatenate([b, c, a])
    across = np.concatenate([c, a, b])
    row = np.tile(rows, 3)
    key = np.minimum(start, end).astype(np.int64) * n_points + np.maximum(start, end)
    order = np.argsort(key, kind="stable")
    key, start, end, across, row = key[order], start[order], end[order], across[order], row[order]
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    size = np.diff(np.r_[first, len(key)])
    lone = first[size != 2]  # an open edge or an edge on three or more faces
    boundary[start[lone]] = boundary[end[lone]] = True
    keep = first[size == 2]
    i, j = keep, keep + 1
    flipped = (start[i] == start[j])
    face_normals = np.cross(points[corners[:, 1]] - points[corners[:, 0]],
                            points[corners[:, 2]] - points[corners[:, 0]])
    flat = ~(np.linalg.norm(face_normals, axis=1) > 0.0)
    bad = flipped | flat[row[i]] | flat[row[j]]
    boundary[start[i[bad]]] = boundary[end[i[bad]]] = True
    i, j = i[~bad], j[~bad]
    v = np.stack([start[i], end[i]], axis=1)
    wings = np.stack([across[i], across[j]], axis=1)
    faces = np.stack([row[i], row[j]], axis=1)
    return EdgeSet(v, wings, faces, turns_deg(points, v, wings), points, corners, boundary)


def position_step(piece, points: np.ndarray, style=None) -> np.ndarray | None:
    """How far a corner of one part of `piece` can sit from where the build put it, per corner
    (V,): the size of one step of the shipped file's position storage. None when the piece has no
    `measured.export`, because then the file's precision is unknown.

    With `position_bits` set, the file is Draco-compressed, which quantises each primitive over
    its own bounding box: a step is the largest side of the part's box over `2 ** position_bits`.
    With `position_bits` null the file keeps float32, whose step at a coordinate is the spacing
    between neighbouring float32 values there (`numpy.spacing`); a corner takes the largest of
    its three. `style` is accepted for the callers that hold it and is not read: the step is in
    the pack's own units already."""
    export = (piece.measured or {}).get("export")
    if export is None:
        return None
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    bits = export.get("position_bits")
    if bits is None:
        return float32_step(points)
    extent = float((points.max(axis=0) - points.min(axis=0)).max()) if len(points) else 0.0
    return np.full(len(points), extent / 2.0**bits)


def float32_step(points: np.ndarray) -> np.ndarray:
    """How far a corner of a float32 mesh can sit from where it was built, per corner (V,): the
    spacing between neighbouring float32 values at its coordinates (`numpy.spacing`), the largest
    of its three. Any mesh the build exports holds float32 positions, so this is the least
    rounding a turn can carry."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return np.spacing(np.abs(points).astype(np.float32)).astype(float).max(axis=1)


def _gradient(points: np.ndarray, v: np.ndarray, wings: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The gradient of each edge's turn in radians with respect to its four corners
    `(x1, x2, x3, x4)`, each (E, 3), by Bridson, Marino and Fedkiw 2003, Section 4:

        u1 = |E| N1 / |N1|^2                      u2 = |E| N2 / |N2|^2
        u3 = ((x1 - x4).e) N1 / |N1|^2 + ((x2 - x4).e) N2 / |N2|^2
        u4 = -((x1 - x3).e) N1 / |N1|^2 - ((x2 - x3).e) N2 / |N2|^2

    with E = x4 - x3, e = E / |E|, and N1, N2 as in `_normals`. Their sign is the paper's; the
    band uses lengths."""
    x3, x4 = points[v[:, 0]], points[v[:, 1]]
    x1, x2 = points[wings[:, 0]], points[wings[:, 1]]
    n1, n2 = _normals(points, v, wings)
    edge = x4 - x3
    length = np.linalg.norm(edge, axis=1, keepdims=True)
    unit = edge / length
    m1 = n1 / np.einsum("ij,ij->i", n1, n1)[:, None]
    m2 = n2 / np.einsum("ij,ij->i", n2, n2)[:, None]

    def dot(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", p, q)[:, None]

    u1 = length * m1
    u2 = length * m2
    u3 = dot(x1 - x4, unit) * m1 + dot(x2 - x4, unit) * m2
    u4 = -dot(x1 - x3, unit) * m1 - dot(x2 - x3, unit) * m2
    return u1, u2, u3, u4


def rounding_band_deg(points: np.ndarray, edges: EdgeSet, step: np.ndarray, crease_deg: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Per edge, the first-order bound on how far its turn can change in degrees when the file
    rounds each coordinate of its four corners to the nearest step (`step` (V,), one per welded
    vertex), and a mask of the edges whose band had to be cut.

    A coordinate rounds by at most half a step. A quantiser rounds a value once, so corners that
    share an x, y or z value move by the same amount along that axis: the bound is, per axis, the
    sum over each group of corners with an equal coordinate of the length of the group's summed
    gradient (see `_gradient`) times half the group's step. It includes the two end corners'
    terms, not only the wings'. Corners that share nothing are bounded by the sum of their
    gradients' components times half a step, never more than the ball of radius `sqrt(3) / 2`
    steps would give. A sliver face makes the gradient huge (its length is one over the face's
    height), so the band is capped at `CREASE_BAND_CAP_OF_ANGLE` times `crease_deg`: past that the
    edge is a sliver's, which topology's zero-area rule owns, and the mask says
    which edges. `points` (V, 3) are the welded vertices the edges index."""
    points = np.asarray(points, dtype=float)
    step = np.asarray(step, dtype=float)
    cap = CREASE_BAND_CAP_OF_ANGLE * crease_deg
    if len(edges) == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        grads = np.stack(_gradient(points, edges.v, edges.wings), axis=1)  # (E, 4, 3)
        ids = np.stack([edges.wings[:, 0], edges.wings[:, 1], edges.v[:, 0], edges.v[:, 1]],
                       axis=1)
        where, reach = points[ids], step[ids]  # (E, 4, 3), (E, 4)
        radians = np.zeros(len(edges))
        for axis in range(3):
            same = where[:, :, None, axis] == where[:, None, :, axis]  # (E, 4, 4)
            total = np.einsum("ekj,ej->ek", same, grads[:, :, axis])
            widest = np.where(same, reach[:, None, :], 0.0).max(axis=2)
            # each group is summed once: its members share its weight
            radians += (np.abs(total) * widest / (2.0 * same.sum(axis=2))).sum(axis=1)
    band = np.degrees(radians)
    capped = ~(band <= cap)  # a NaN or an infinity (a face with no height) is capped too
    return np.where(capped, cap, band), capped


def line_ends(edges: EdgeSet, drawn: np.ndarray, open_boundary: np.ndarray) -> np.ndarray:
    """The ids of the vertices where a drawn line stops: those with exactly one drawn edge and
    not on an open boundary. `drawn` (E,) says which edges draw; `open_boundary` (V,) marks the
    vertices on an edge that is not an interior edge (a leaf card's rim, a hole)."""
    open_boundary = np.asarray(open_boundary, dtype=bool)
    count = np.bincount(edges.v[np.asarray(drawn, dtype=bool)].ravel(),
                        minlength=len(open_boundary))
    return np.flatnonzero((count == 1) & ~open_boundary)


def face_widths(tris: np.ndarray, edges: EdgeSet) -> np.ndarray:
    """Per edge (E, 2), the height of each of its two faces over the edge (`2 * area / edge
    length`), in the units of `tris`. `EdgeSet.faces` indexes the rows of `tris`."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    length = np.linalg.norm(edges.points[edges.v[:, 0]] - edges.points[edges.v[:, 1]], axis=1)
    return 2.0 * area[edges.faces] / length[:, None]


def chord_sagitta(width: np.ndarray | float, turn_deg: np.ndarray | float) -> np.ndarray | float:
    """The gap between two chords of length `width` that meet at a turn of `turn_deg` and the
    circle through them: `(width / 2) * tan(turn / 4)`. The one place the gap is written; `sagittas`
    reads it from a mesh and `segments_for` from a curve."""
    return 0.5 * width * np.tan(np.radians(turn_deg) / 4.0)


def sagittas(tris: np.ndarray, edges: EdgeSet) -> np.ndarray:
    """Per edge (E,), the gap between the two flat faces that meet there and the curve they
    stand for: `(w / 2) * tan(turn / 4)` (`chord_sagitta`), in the units of `tris`.

    `w` is the narrower of the edge's two faces, measured across the edge as the face's height
    over it (`face_widths`). Two chords of length `w` that meet at the edge's turn lie on a
    circle, and the gap between that circle and the chords' join is this. For a regular polygon
    of radius r and n sides it is exactly `r * (1 - cos(pi / n))`, from
    `1 - cos x = sin x * tan(x / 2)`. A flat join (turn 0) has none, and a fold of 180 degrees
    has no circle, so its tangent is large; callers read only the edges under the crease angle.
    `tris` (N, 3, 3) are the rows `edges` was made from: `EdgeSet.faces` indexes them."""
    if len(edges) == 0:
        return np.zeros(0)
    return chord_sagitta(face_widths(tris, edges).min(axis=1), edges.turn_deg)


def segments_for(radius: float, turn_deg: float, gap: float) -> int:
    """The fewest segments that cut a circular arc of `radius` through `turn_deg` degrees so that
    no joint's gap to the curve is over `gap`: `detail`'s inequality (`sagittas`) turned round.

    With `count` equal chords the arc's joints each turn `turn_deg / count`, and the chords are
    `2 * radius * sin(turn / count / 2)` long, so the gap at a joint is
    `radius * (1 - cos(radians(turn_deg / count) / 2))`. That falls as the count grows, and this
    is the least count at which it is at or under `gap`. A whole circle is `turn_deg` 360; a count
    never has a joint turn of 180 degrees or more, so a circle takes at least 3. A `radius` of 0
    has no gap at any count. `gap` (the piece's seen floor, `Sight.gap`) is in the units of
    `radius`; one at or under 0, or a turn at or under 0, raises."""
    if not (gap > 0.0 and turn_deg > 0.0 and radius >= 0.0):
        raise ManifestError(f"segments_for: radius {radius!r}, turn {turn_deg!r} and gap {gap!r} "
                            "must be 0 or more, over 0 and over 0")

    def gap_at(count: int) -> float:
        each = turn_deg / count
        return float(chord_sagitta(2.0 * radius * np.sin(np.radians(each) / 2.0), each))

    count = lowest = int(turn_deg // 180.0) + 1
    if radius == 0.0 or gap_at(count) <= gap:
        return count
    # the count at which the closed form meets `gap`, then walked to the exact least count
    count = max(lowest, int(np.ceil(np.radians(turn_deg) / (2.0 * np.arccos(1.0 - gap / radius)))))
    while gap_at(count) > gap:
        count += 1
    while count > lowest and gap_at(count - 1) <= gap:
        count -= 1
    return count


def merged_gaps(tris: np.ndarray, edges: EdgeSet) -> np.ndarray:
    """Per edge (E,), how far the edge's shared corner would stand off the chord that joins the
    far corners of its two faces: the gap the eye would see if the edge were removed, in the
    units of `tris`.

    With the two faces' heights `a` and `b` over the edge and its turn `t`, the faces are chords
    of those lengths that meet at turn `t`, and the corner lies at `a b sin t / sqrt(a^2 + b^2 +
    2 a b cos t)` from the chord that joins their far ends (the height of a triangle with sides
    `a`, `b` and the angle `pi - t` between them). For equal faces this is `a sin(t / 2)`, which
    is `4 cos^2(t / 4)` times the sagitta: the ring that has half its segments. For unequal faces
    it is more, because the narrow face's corner is pulled across the long one."""
    if len(edges) == 0:
        return np.zeros(0)
    a, b = face_widths(tris, edges).T
    t = np.radians(edges.turn_deg)
    return a * b * np.sin(t) / np.sqrt(a * a + b * b + 2.0 * a * b * np.cos(t))
