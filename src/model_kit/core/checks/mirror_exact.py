"""Check mirror-exact: a mirrored part is its source's exact image, triangle for triangle.

A part declared the mirror of another is that part reflected across a plane through the piece's
origin. A reflection keeps every vertex, so the two parts must hold the same triangles. A quad
that the twin splits along its other diagonal draws a different crease on each side.

Two kinds of pair are measured, each part once (a part in a pair by its form is never paired
again by its bones):

- **form.** A part whose `form` is `mirror_of` another part. The plane is normal to x, as
  `blender/soft.py`'s `mirror_part` negates x.
- **bones.** Two parts that no form pairs, where the bones of one are the bones of the other
  swapped by the rig's mirror map, and at least one bone is in a pair. The plane is normal to the
  map's axis. The earlier part in the piece is the source. A part with no such twin is left
  alone: a body may carry a part on one side only. When each side holds the same number of parts
  on those bones (an arm and its sleeve), each source is paired with the target its reflection
  matches best, one to one. When the counts differ, no pairing is honest and the group is
  `Unmeasured`.

The source's triangles, reflected, must be the target's triangles as sets of corner triples. All
corners of both are welded together (`WELD_OF_PIECE_SIZE` times the piece size, as topology
welds), then each triangle is its three welded corners without order, so a winding that was
reversed to face outward again still matches. Winding itself is topology's.

Item: `<pack>:<piece> <target> mirror of <source>`. A piece with no pair prints a not-applicable
line. A pair where a part has no geometry in the scene is `Unmeasured`, and so is a piece whose
extent is zero: it has no weld distance, so no triangle can differ.

A part that is asymmetric on purpose but is bound to mirrored bones is a finding, not an
exception: the check cannot tell a deliberate difference from a slip. Clear it with an allow line
that gives the reason.

The plants are one unit per measured pair, built from the target's triangles alone (never from the
count of unmatched triangles): the two triangles that share the flattest convex edge replaced by
the two on the quad's other diagonal; then, for a target with no such quad (one triangle, or no
convex fold), one corner moved `CALIBRATION_FAIL` times the weld distance; then its last triangle
dropped. The gate tries them in order and skips a pair that fails already.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks import pieces
from model_kit.core.checks.pieces import PieceView, item_of
from model_kit.core.geom import weld
from model_kit.core.manifest import AXES
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import CALIBRATION_FAIL, WELD_OF_PIECE_SIZE

NAME = "mirror-exact"
PRINCIPLE = ("A part declared the mirror of another is the reflection of it, so it holds the "
             "same triangles with the same corners, not only the same outline.")
SOURCE = (
    "A mirror reflects a mesh across a plane through its object origin, normal to one axis "
    "(Blender Manual, Mirror modifier: "
    "https://docs.blender.org/manual/en/latest/modeling/modifiers/generate/mirror.html). The "
    "kit's own mirror negates x (`blender/soft.py`, `mirror_part`); the rig's mirror map names "
    "the axis for parts paired by their bones. A mirrored part can differ from "
    "its source by the diagonals along which its quads are split.")
FORM_AXIS = "x"  # `blender/soft.py` `mirror_part` negates x


@dataclass(frozen=True)
class Pair:
    """Two parts of a piece, the source and its mirror, and the axis they reflect across."""

    source: str
    target: str
    axis: str

    def item(self, view: PieceView) -> str:
        return f"{view.key} {self.target} mirror of {self.source}"


def _swap(piece_pairs: tuple[tuple[str, str], ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for a, b in piece_pairs:
        out[a], out[b] = b, a
    return out


def _by_shape(view: PieceView, sources: list[str], targets: list[str], axis: str) -> list[tuple[str, str]] | None:
    """Each source with the target its reflection matches best, one to one (the pair with the
    fewest unmatched triangles first). None when a part has no geometry to match."""
    mesh = {p.name: p.tris for p in view.parts}
    if any(n not in mesh for n in sources + targets):
        return None
    tol = WELD_OF_PIECE_SIZE * view.size
    cost = {(s, t): (sum(compare(mesh[s], mesh[t], axis, tol)) if tol > 0.0 else 0)
            for s in sources for t in targets}
    out: list[tuple[str, str]] = []
    for (s, t), _ in sorted(cost.items(), key=lambda kv: kv[1]):  # stable: ties keep part order
        if all(s != a and t != b for a, b in out):
            out.append((s, t))
    return sorted(out, key=lambda st: sources.index(st[0]))


def pairs_of(view: PieceView) -> tuple[list[Pair], list[str]]:
    """The mirrored pairs of a piece, and the parts it could not pair (each as a phrase).

    Parts bound to the same bones form a group. A group and the group on the mirrored bones are
    paired part for part by shape when both hold the same number of parts (an arm and its sleeve
    on each side). When the counts differ no pairing is honest, and the group is unpaired."""
    found: list[Pair] = []
    taken: set[str] = set()
    for part in view.piece.parts:
        if part.form is not None and part.form.mirror_of is not None:
            found.append(Pair(part.form.mirror_of, part.name, FORM_AXIS))
            taken |= {part.form.mirror_of, part.name}
    mirror = view.piece.rig.mirror if view.piece.rig is not None else None
    unpaired: list[str] = []
    if mirror is None:
        return found, unpaired
    swap = _swap(mirror.pairs)
    rest = [p for p in view.piece.parts if p.name not in taken]
    order = {p.name: i for i, p in enumerate(view.piece.parts)}
    done: set[str] = set()

    def on(bones: frozenset[str]) -> list[str]:
        return [q.name for q in rest if frozenset(q.bones) == bones and q.name not in done]

    for part in rest:
        if part.name in done or not any(b in swap for b in part.bones):
            continue
        here = frozenset(part.bones)
        there = frozenset(swap.get(b, b) for b in here)
        mine, theirs = on(here), on(there) if there != here else []
        done |= set(mine) | set(theirs)
        if there == here:  # the bones are their own mirror: the group pairs within itself
            if len(mine) == 2:
                found.append(Pair(mine[0], mine[1], mirror.axis))
            elif len(mine) > 2:
                unpaired.append(f"{len(mine)} parts hold the mirrored bones of {part.name} "
                                f"({', '.join(mine)}), so it has no one twin")
            continue
        if not theirs:
            continue  # a part on one side only
        if len(mine) != len(theirs):
            unpaired.append(f"{len(mine)} parts hold the bones of {part.name} and {len(theirs)} "
                            f"hold the mirrored bones, so it has no one twin ({', '.join(mine)} "
                            f"against {', '.join(theirs)})")
            continue
        if order[theirs[0]] < order[mine[0]]:
            mine, theirs = theirs, mine  # the earlier part in the piece is the source
        if len(mine) == 1:
            found.append(Pair(mine[0], theirs[0], mirror.axis))
            continue
        matched = _by_shape(view, mine, theirs, mirror.axis)
        if matched is None:
            unpaired.append(f"{len(mine)} parts hold the bones of {part.name}, and a part has no "
                            f"geometry to pair them by shape, so it has no one twin")
            continue
        found += [Pair(s, t, mirror.axis) for s, t in matched]
    return found, unpaired


def _reflect(tris: np.ndarray, axis: str) -> np.ndarray:
    out = np.array(tris, dtype=float)
    out[..., AXES.index(axis)] *= -1.0
    return out


def _keys(faces: np.ndarray) -> Counter:
    """Each triangle as its welded corners without order, counted."""
    return Counter(map(tuple, np.sort(faces, axis=1).tolist()))


def compare(source: np.ndarray, target: np.ndarray, axis: str, tol: float) -> tuple[int, int]:
    """How many triangles of the source, reflected, have no match in the target, and how many of
    the target's have none in the source. Corners within `tol` are one corner."""
    if len(source) == 0 and len(target) == 0:
        return 0, 0
    reflected = _reflect(source, axis)
    _, faces = weld(np.concatenate([reflected.reshape(-1, 3, 3), np.asarray(target, float)
                                    .reshape(-1, 3, 3)]), tol)
    a, b = _keys(faces[:len(source)]), _keys(faces[len(source):])
    return sum((a - b).values()), sum((b - a).values())


def _meshes(view: PieceView, pair: Pair) -> tuple[np.ndarray, np.ndarray] | None:
    by_name = {p.name: p.tris for p in view.parts}
    if pair.source in by_name and pair.target in by_name:
        return by_name[pair.source], by_name[pair.target]
    return None


def _detail(view: PieceView, pair: Pair, source: np.ndarray, target: np.ndarray) -> str | None:
    """Why the target is not the source's image, or None when it is."""
    tol = WELD_OF_PIECE_SIZE * view.size
    if not tol > 0.0:
        return None  # a piece with no extent holds no triangle to differ
    lost, extra = compare(source, target, pair.axis, tol)
    if not lost and not extra:
        return None
    return (f"{lost} of {len(source)} triangles of {pair.source}, mirrored across {pair.axis}, "
            f"are not in {pair.target}, and {extra} of its {len(target)} are not in the source")


# --- plants ----------------------------------------------------------------------------------


def flip_diagonal(tris: np.ndarray, tol: float) -> np.ndarray | None:
    """A copy of the triangles with one quad split along its other diagonal, or None if no two
    triangles make a convex quad.

    Two triangles that share an edge (a, b), run a to b by one and b to a by the other, with the
    far corners c and d, are the quad c a d b. Its other diagonal c d gives the triangles (c, a,
    d) and (c, d, b), wound as the pair was. The edge with the flattest fold is taken (a quad of
    a grid has a fold near zero), and only if both new triangles face the way the pair did."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    if len(tris) < 2:
        return None
    _, faces = weld(tris, tol)
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    unit = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-300)
    along: dict[tuple[int, int], tuple[int, int]] = {}
    for i, f in enumerate(faces.tolist()):
        for k in range(3):
            if f[k] != f[(k + 1) % 3]:
                along[(f[k], f[(k + 1) % 3])] = (i, k)
    best: tuple[float, int, int, int, int] | None = None
    for (a, b), (i, k) in along.items():
        other = along.get((b, a))
        if other is None or a > b:
            continue
        j, m = other
        if i == j:
            continue
        fold = float(1.0 - unit[i] @ unit[j])
        if best is None or fold < best[0]:
            pa, pb, pc = tris[i][k], tris[i][(k + 1) % 3], tris[i][(k + 2) % 3]
            pd = tris[j][(m + 2) % 3]
            was = normal[i] + normal[j]
            one, two = np.cross(pa - pc, pd - pc), np.cross(pd - pc, pb - pc)
            if one @ was > 0.0 and two @ was > 0.0:
                best = (fold, i, j, k, m)
    if best is None:
        return None
    _, i, j, k, m = best
    pa, pb, pc = tris[i][k], tris[i][(k + 1) % 3], tris[i][(k + 2) % 3]
    pd = tris[j][(m + 2) % 3]
    out = tris.copy()
    out[i] = np.array([pc, pa, pd])
    out[j] = np.array([pc, pd, pb])
    return out


def _nudged(tris: np.ndarray, tol: float) -> np.ndarray | None:
    """A copy of the triangles with the first corner of the first triangle moved `CALIBRATION_FAIL`
    times the weld distance along x, so it no longer welds to the corner it was; None for no
    triangle."""
    tris = np.array(tris, dtype=float).reshape(-1, 3, 3)
    if len(tris) == 0:
        return None
    tris[0, 0, 0] += CALIBRATION_FAIL * tol
    return tris


def _dropped(tris: np.ndarray) -> np.ndarray | None:
    """A copy of the triangles without the last one; None for no triangle."""
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    return tris[:-1] if len(tris) else None


def _measured_pairs(view: PieceView) -> list[tuple[Pair, np.ndarray, np.ndarray, float]]:
    """Each pair the check can judge: both parts have geometry and the piece has an extent."""
    tol = WELD_OF_PIECE_SIZE * view.size
    if not tol > 0.0:
        return []
    return [(pair, *meshes, tol) for pair in pairs_of(view)[0]
            if (meshes := _meshes(view, pair)) is not None]


# --- the check -------------------------------------------------------------------------------


class MirrorExact:
    name = NAME
    kinds: frozenset[str] = frozenset({"assembly"})
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units = []
        for view in pieces.measured_pieces(ctx, self.kinds):
            geometry = {p.name: p.tris for p in view.parts}
            for pair, _, target, tol in _measured_pairs(view):
                item = pair.item(view)
                candidates = [(how, bad) for how, bad in (
                    ("flipped diagonal", flip_diagonal(target, tol)),
                    ("moved corner", _nudged(target, tol)),
                    ("dropped triangle", _dropped(target))) if bad is not None]
                units.append(PlantUnit(item, tuple(
                    Plant(f"{how}: {item}", item,
                          pieces.with_geometry(ctx, view, geometry | {pair.target: bad}))
                    for how, bad in candidates)))
        return units

    def measured(self, ctx: SceneCtx) -> set[str]:
        """Both parts of every pair it compared."""
        out: set[str] = set()
        for view in pieces.measured_pieces(ctx, self.kinds):
            for pair in pairs_of(view)[0]:
                if _meshes(view, pair) is not None:
                    out |= {item_of(view, pair.source), item_of(view, pair.target)}
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for view in pieces.measured_pieces(ctx, self.kinds, only):
            for pair in pairs_of(view)[0]:
                item = pair.item(view)
                meshes = _meshes(view, pair)
                if meshes is None or (only is not None and item not in only):
                    continue
                if (why := _detail(view, pair, *meshes)) is not None:
                    found.append(Finding(NAME, item, why))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = pieces.coverage(ctx, NAME, self.kinds)
        for view in pieces.measured_pieces(ctx, self.kinds):
            pairs, unpaired = pairs_of(view)
            lines += [Unmeasured(f"{NAME}: skipped ({why}): {view.key}") for why in unpaired]
            if not pairs and not unpaired:
                lines.append(f"{NAME}: not applicable (no part is the mirror of another): "
                             f"{view.key}")
            if pairs and not WELD_OF_PIECE_SIZE * view.size > 0.0:
                lines.append(Unmeasured(f"{NAME}: skipped (the piece has no extent, so no weld "
                                        f"distance): {view.key}"))
            for pair in pairs:
                if _meshes(view, pair) is None:
                    missing = [n for n in (pair.source, pair.target)
                               if n not in {p.name for p in view.parts}]
                    lines.append(Unmeasured(
                        f"{NAME}: skipped (no geometry for {', '.join(missing)}): "
                        f"{pair.item(view)}"))
        return list(dict.fromkeys(lines))
