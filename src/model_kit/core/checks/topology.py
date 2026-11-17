"""Check topology, topology for outline: each part is one closed, outward, well-formed surface.

The runtime outline is drawn from the mesh, so a hole, a doubled edge or a flipped face shows
as a break in the line. Each part's triangles are welded (`geom.weld`: corners within
`WELD_OF_PIECE_SIZE` times the piece size merge; the size is the diagonal of the box of all the
piece's parts), then:

- a face with a repeated corner, or with an area under `ZERO_AREA_OF_LONGEST_EDGE_SQUARED` times
  its own longest edge squared, is a zero-area face: a face that has collapsed to a line. Only
  a face with a repeated corner is dropped before the edges are counted, so a thin sliver
  never opens an edge;
- an edge on one face is open, and an edge on 3 or more faces is non-manifold;
- an edge whose two faces both run along it the same way is a flipped neighbour (winding is
  not consistent);
- a closed, consistent surface must also enclose a positive volume, or it is inside out.

Open surfaces (`Part.open_surface`: shape `leaf`, `cable` or `decal`) and solid parts that skip
their outline (role `skip_outline`) are not measured: a leaf is a sheet, a cable an open tube and
a decal an open patch printed on its host, and a `skip_outline` part draws no outline. They
are listed in coverage. The item is `<pack>:<piece> <part>`.

The plant is one unit per measured part with triangles: it flips one face (the first, then the
last). It is built from the part's triangles alone, never from whether the part is clean; the
gate skips a part that already fails. A part with no triangles is reported by the check itself, so
it is not planted on.

Coverage lists each part left out as `not applicable (<shape or role> part)`.
"""

from __future__ import annotations

from collections.abc import Collection

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PartMesh, item_of
from model_kit.core.geom import signed_volume, weld
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import WELD_OF_PIECE_SIZE, ZERO_AREA_OF_LONGEST_EDGE_SQUARED

NAME = "topology"


def _exempt(part: PartMesh) -> str | None:
    """What kind of part this is, when topology does not apply to it: an open surface by shape,
    or a solid part that opts out of the outline (role `skip_outline`). A printed part with a
    closed shape (role `eyes`) is measured."""
    if not (part.open_surface or (part.solid and part.skips_outline)):
        return None
    return part.shape if part.open_surface else part.role


def problems(tris: np.ndarray, size: float) -> list[str]:
    """What is wrong with one part's surface, each as a phrase; empty when it is clean. `size`
    is the part's piece size."""
    if len(tris) == 0:
        return ["has no triangles"]
    if not size > 0.0:
        return ["has no extent"]
    verts, faces = weld(tris, WELD_OF_PIECE_SIZE * size)
    corners = verts[faces]
    sides = np.roll(corners, -1, axis=1) - corners
    area = 0.5 * np.linalg.norm(np.cross(sides[:, 0], -sides[:, 2]), axis=1)
    longest = np.max(np.sum(sides**2, axis=2), axis=1)
    repeated = ((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])
                | (faces[:, 2] == faces[:, 0]))
    flat = repeated | (area < ZERO_AREA_OF_LONGEST_EDGE_SQUARED * longest)
    out = [f"{int(flat.sum())} zero-area faces"] if flat.any() else []
    faces, corners = faces[~repeated], corners[~repeated]
    if len(faces) == 0:
        return [*out, "has no faces left"]
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    low, high = directed.min(axis=1), directed.max(axis=1)
    edges, inverse, count = np.unique(low * len(verts) + high, return_inverse=True,
                                      return_counts=True)
    along = np.where(directed[:, 0] < directed[:, 1], 1.0, -1.0)
    run = np.bincount(inverse, weights=along, minlength=len(edges))
    open_edges = int((count == 1).sum())
    many = int((count >= 3).sum())
    flipped = int(((count == 2) & (np.abs(run) == 2)).sum())
    if open_edges:
        out.append(f"{open_edges} open edges")
    if many:
        out.append(f"{many} edges on 3 or more faces")
    if flipped:
        out.append(f"{flipped} flipped edges (winding is not consistent)")
    if not out:
        volume = signed_volume(corners)
        if not volume > 0.0:
            out.append(f"volume {volume:.6f}, not positive (inside out)")
    return out


class Topology:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    scope = "piece"

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        out = []
        for view in pieces.measured_pieces(ctx, self.kinds):
            for part in view.parts:
                if _exempt(part) is not None or not len(part.tris):
                    continue
                item = item_of(view, part.name)
                candidates = []
                for face in dict.fromkeys((0, len(part.tris) - 1)):
                    flipped = part.tris.copy()
                    flipped[face] = flipped[face][[0, 2, 1]]
                    geometry = {p.name: p.tris for p in view.parts} | {part.name: flipped}
                    candidates.append(Plant(f"flipped face {face}: {item}", item,
                                            pieces.with_geometry(ctx, view, geometry)))
                out.append(PlantUnit(item, tuple(candidates)))
        return out

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Every part with triangles except the open surfaces and `skip_outline` parts it leaves out
        by design."""
        return {item_of(view, p.name) for view in pieces.measured_pieces(ctx, self.kinds)
                for p in view.parts if _exempt(p) is None and len(p.tris)}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for view in pieces.measured_pieces(ctx, self.kinds, only):
            size = view.size
            for part in view.parts:
                item = item_of(view, part.name)
                skipped = _exempt(part) is not None or (only is not None and item not in only)
                if not skipped and (why := problems(part.tris, size)):
                    found.append(Finding(NAME, item, "; ".join(why)))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        skipped = [f"{NAME}: not applicable ({kind} part): {view.key} {part.name}"
                   for view in pieces.measured_pieces(ctx, self.kinds) for part in view.parts
                   if (kind := _exempt(part)) is not None]
        return pieces.coverage(ctx, NAME, self.kinds) + skipped
