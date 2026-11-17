"""Check seam-jump: the baked light does not step across a lightmap seam by more than the bend explains.

Principle: the irradiance on a surface follows its normal through Lambert's cosine law. Light of
radiance `L(w)` arriving from directions `w` gives a surface of normal `n` the irradiance
`E(n) = integral of L(w) max(0, n . w) dw`. So where the normal has no step the light has none,
and where the normal bends the light can step only as far as the bend allows. A larger step at a
lightmap seam, where the unwrap cuts the surface into islands, is the unwrap's fault (an island
the bake lit differently, a margin read the wrong way), not the light's.

The allowance, from Lambert's law. `max(0, x)` changes by no more than `x` does, and
`|n_a . w - n_b . w| <= |n_a - n_b|` for a unit direction `w` (Cauchy-Schwarz), so
`|E_a - E_b| <= |n_a - n_b| x integral of L(w) dw` over the directions either side faces. For
light that arrives near the normal that integral is about the irradiance itself, so the step is at
most about `|n_a - n_b| x E`: the **allowance** is `bend x max(|v_a|, |v_b|)`, per channel, where
`bend` is `|n_a - n_b|` and `v` is each side's light at the edge (below). A seam on a flat surface
gets almost none; a seam at a real corner gets about the full light. Two consequences:

- A seam whose `bend` is 1 or more (its sides 60 degrees or more apart) cannot step past its
  allowance whatever the light, because light is never negative, so `|v_a - v_b| <= max(|v_a|,
  |v_b|)`. It needs no read, and an unreadable one is not counted as unmeasured.
- The bound is exact for light along the normal. For one light `theta` off the normal the step can
  reach `tan(theta)` times the allowance, so a bent seam under grazing light (more than 45 degrees
  off) can step past the line by a real light change. That finding is reported: the person judges
  whether the light is grazing there.

A **seam edge** is a mesh edge that lies inside one part, whose two triangles map to different
lightmap UV. The check welds a part's corners by position (to one millionth of the part's extent),
so two triangles share an edge when two corners match. The edge is a seam when the UV of either
shared corner differs between the two triangles. Every seam edge is judged. Its `bend` is the
larger, over its two shared corners, of the distance between the two sides' unit vertex normals
(world frame), and 0 when at both corners the two normals are one surface within the file's
own precision (`same_normal`). Two sources of error count, and nothing else:

- The exporter rounds each normal component to `NORMAL_DIGITS` decimals and normalises, so an
  exported normal lies within `sqrt(3) / 2 x 10^-NORMAL_DIGITS` of the normal it was given
  (normalising moves it no farther), and two exported normals of one normal differ by up to
  `SAME_NORMAL = sqrt(3) x 10^-NORMAL_DIGITS`.
- Blender computes a face's normal from its corners, which it holds in float32, as the .glb
  does. A corner is off by up to `sqrt(3) / 2 x eps x m` (`eps` = 2^-23, `m` the face's largest
  coordinate magnitude in its mesh's frame), and that tilts the normal by up to `k x` that over
  the smallest distance between two of the face's `k` corners (`normal_noise`). A tiny face far from its mesh's
  origin, like a decal 0.2 mm across 0.8 m out, tilts by about 1e-3: a flat surface whose
  facets' normals differ by that much is still flat.

Rotation keeps these distances; a non-uniform scale on a node does not, and the rule ignores it.
The bake's unwrap (`blender/bake_render.py`, `smooth_corners`) joins faces into one region by this
same function. It reads polygons where the check reads their triangles: a triangle has no more
corners, no larger coordinate and no shorter corner distance than its polygon, so the check's
bound is never looser than the unwrap's, and the unwrap cuts no seam that the check calls flat
except where it must (a tube's one cut, a fallback unwrap's own islands). Edges between different parts are never compared, and a triangle with zero lightmap
area (a face no camera sees) takes no part: a seam touching one is not sampled.

An **island** is a connected lightmap region of one part: the triangles of area joined through
shared edges whose UV agree on both sides. A texel is **covered** by an island when its centre lies
inside, or on the edge of, one of the island's triangles (the bake's own coverage rule, rasterised
by `bake_data.texels`). A texel two islands of the part both cover belongs to neither.

For each seam edge and each layer whose picture is loaded, each side's light AT the edge is
estimated from the picture at 5 points along the edge (at 1/10, 3/10, 5/10, 7/10 and 9/10 of its
length). Each point is read inside its own side, at depth `d` and `2 d` texels along the inward
perpendicular (bilinear), and extrapolated to the edge: `v_edge = 2 v(d) - v(2 d)`. A light that
changes linearly across the seam gives the same `v_edge` as a read on the edge, so a smooth gradient
is not mistaken for a step. A read counts only when all 4 of its bilinear taps are covered by its
side's own island: the margin outside an island is the bake's fill, not the surface's light, and a
read that crosses into the next triangle of the same island is still the same surface. For each
side and point, `d` is the first of `INSETS` (0.5 to 1.5 texels, nearest first) whose two reads
both count. A point counts when both sides have one; the two sides' `v_edge` are subtracted and
averaged over the points that count; the edge's step is the RMS over the three channels of that
mean difference, the same RMS the bake records its noise in. Each side's `v` is the mean, over
the same points, of its own `v_edge`, and the allowance is the channel RMS of `bend x max(|v_a|,
|v_b|)`. An edge with no point that counts cannot be read: its island is too thin there for two
reads inside it. It is not judged, and `coverage` names the part and how many of its seam edges
with a bend under 1 could not be read, as an `Unmeasured` line, so the run is partial: an
unreadable seam is an unwrap problem the gate must not hide.

Threshold. The bake records `noise` per doubled texel; a shipped texel inside an island is the mean
of the 4 doubled texels of its 2 x 2 block, so its noise is `ns = noise / 2`. A texel at an island's
edge averages only its block's 1 to 3 covered texels, so its noise is `noise / sqrt(k)`, up to
`noise`; the check takes `noise / 2` for every texel, which makes the line stricter at an edge.
One side's `v_edge` is `2 a - b` for two reads of noise `ns`; treating the two reads as independent
(the conservative choice: neighbours of one texel share nothing, but a bilinear read mixes four),
its noise is `sqrt(2^2 + 1^2) x ns = sqrt(5) x ns`, at any depth `d`. The picture also cannot
resolve a step finer than its own resolution: the bake records `error`, the RMS over the 4
block-grid phases of the picture shrunk and grown back to the doubled size against the doubled
bake (phase 0 is the shipped picture), and a light that curves inside a
texel shows at the edge as a step of about that size. Light the picture under-resolves is
`lightmap-density`'s finding, not this check's. Each side carries its placement's `error` `e`,
independent of the other side. So an edge is reported when its step exceeds

    SEAM_JUMP_SIGMAS x sqrt(5 (ns_a^2 + ns_b^2) + e_a^2 + e_b^2) + allowance

with `ns` and `e` taken from each side's placement for the layer (one placement on both sides, as
a seam lies inside one part). A placement with no recorded `error` takes 0, the stricter line. The
average over the points along the edge is not credited (a lenient choice).

A layer whose picture is not loaded, or is not `size` x `size` (bake-files reports both), or whose
`noise` has no entry for the part's placement (bake-files reports that too), is skipped for that
layer, with no finding and no coverage line. One item per part: `seam <part> jump`, and the detail
names the layer, the count of read edges over the line and the worst one.

A scene with no bake has no finding, no plant and no coverage line. With a bake the plant is one
unit: of the seam edges with a bend under 1 that can be read, with a texel to change, the one with
the least bend (parts sorted, then edge order), the picture brightened at the texels the check
reads on side two (the bilinear taps of its counted reads, less side one's own taps). Candidate
one multiplies them by 1.5; candidate two adds `4 x line / (1 - bend)` to them, `line` being the
edge's own threshold: the allowance grows with the brightened side by `bend` of what is added, so
what is added beyond it is `(1 - bend)` of it, and no light level can hide that. A bake with no
such edge to plant on, or no layer judged, has one not-applicable unit that says so (an
informational coverage line, not an unmeasured one).
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Collection

import numpy as np

from model_kit.core.bake_data import BakeData, area2, texels
from model_kit.core.checks.entries import unit
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import SEAM_JUMP_SIGMAS

NAME = "seam-jump"
PRINCIPLE = ("Light changes across a surface only as its normal does, by Lambert's law, so a "
             "step in the baked light at a lightmap seam beyond what the bend explains is the "
             "unwrap's, not the light's.")
SOURCE = ("Lambert's cosine law, E(n) = integral of L(w) max(0, n . w) dw, so |E_a - E_b| <= "
          "|n_a - n_b| x the light arriving, about |n_a - n_b| x E for light near the normal "
          "(https://en.wikipedia.org/wiki/Lambert%27s_cosine_law); the glTF exporter's normal "
          "rounding, io_scene_gltf2 constants.ROUNDING_DIGIT = 4, applied in "
          "blender/exp/primitive_extract.py; light-bake spec, 'Seam jump' line and 'Checks' "
          "section; model-kit AGENTS.md, 'Adding a check'.")
_ITEM = re.compile(r"seam (.+) jump")
ALONG = (0.1, 0.3, 0.5, 0.7, 0.9)  # where on the edge each side is sampled: 5 points
# Texels: the nearer read's depths, tried nearest first; the other read is at twice the depth. A
# bilinear read's taps lie within one texel of its point on each axis, so a read sqrt(2) = 1.41
# texels inside a straight edge at any angle has all 4 taps on the edge's inner side: 1.5 is the
# first step past it. Quarter-texel steps keep the extrapolation short where the island allows.
INSETS = (0.5, 0.75, 1.0, 1.25, 1.5)
WELD = 1e-6  # corners within this fraction of the part's extent are one corner
SAME_UV = 1e-3  # texels: a shared corner's UV that differs by less is the same place
# The decimals the glTF exporter rounds each normal component to (io_scene_gltf2's
# ROUNDING_DIGIT); test/blender/test_bake_unwrap.py checks the installed exporter agrees.
NORMAL_DIGITS = 4
# Two exported normals of one true normal differ by up to this (unit vectors, Euclidean): each is
# within sqrt(3) / 2 x 10^-NORMAL_DIGITS of the true normal (see the module docstring).
SAME_NORMAL = math.sqrt(3.0) * 10.0 ** -NORMAL_DIGITS
# float32's machine epsilon, 2^-23: a stored coordinate x is within F32_EPS / 2 x |x| of the value
# it stands for (IEEE 754 round to nearest). The .glb and Blender both hold positions in float32.
F32_EPS = float(np.finfo(np.float32).eps)
BRIGHTEN = 1.5  # the plant's candidate one: brighten by 50%
STEP_SIGMAS = 4.0  # the plant's candidate two: add this many thresholds, over (1 - bend)


def normal_noise(corners: np.ndarray) -> np.ndarray:
    """(n,) for faces (n, k, 3), given by their k corners as stored (float32, the mesh's own
    frame): the most each face's unit normal can tilt because its corners are float32.

    Each coordinate is within `F32_EPS / 2 x |x|` of its true value, so a corner is within
    `e = sqrt(3) / 2 x F32_EPS x m` of its true place, `m` the face's largest coordinate magnitude.
    Moving one corner by `e` across the face tilts its normal by about `e / L` (radians, and the
    unit normal moves by as much), `L` the smallest distance between two of the face's corners
    (a triangle's shortest edge; a polygon's diagonals count too); the k corners at worst add, so
    the bound is `k x e / L`. A face far from its mesh's origin, or a tiny one, gets a large bound:
    the same float32 step is a larger share of a short edge. A triangle cut from a polygon has no
    more corners, no larger coordinate and no shorter `L` than the polygon (its edges are distances
    between the polygon's corners), so its bound is never larger than its polygon's."""
    corners = np.asarray(corners, dtype=np.float64)
    k = corners.shape[1]
    gaps = [np.linalg.norm(corners[:, i] - corners[:, j], axis=1)
            for i in range(k) for j in range(i + 1, k)]
    shortest = np.min(gaps, axis=0)
    m = np.abs(corners).max(axis=(1, 2))
    e = math.sqrt(3.0) / 2.0 * F32_EPS * m
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(shortest > 0.0, k * e / shortest, np.inf)


def same_normal(a: np.ndarray, b: np.ndarray, noise_a: float = 0.0, noise_b: float = 0.0) -> bool:
    """Whether unit normals `a` and `b`, as the glTF exporter writes them, are one surface: they
    differ by no more than the exporter's own rounding (`SAME_NORMAL`) plus how far float32
    corners can tilt each face's normal (`noise_a`, `noise_b`, from `normal_noise`). The bake's
    unwrap joins its smooth regions by this same rule."""
    gap = float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))
    return gap <= SAME_NORMAL + noise_a + noise_b


@dataclasses.dataclass(frozen=True)
class Seam:
    """One seam edge of a part. `tri` are the two triangles' indices; `uv` is each side's three
    corners as (a, b, third) in texels, a and b being the edge's ends; `at` the edge's two ends in
    the world; `island` each side's island; `bend` the larger distance between the two sides'
    unit normals at the edge's two ends, 0 where they are one normal (`same_normal`)."""

    tri: tuple[int, int]
    uv: np.ndarray  # (2, 3, 2) texels
    at: np.ndarray  # (2, 3) world
    island: tuple[int, int] = (0, 0)
    bend: float = 0.0

    @property
    def flat(self) -> bool:
        """Whether the two sides carry one normal at both ends (`same_normal`)."""
        return self.bend == 0.0


@dataclasses.dataclass(frozen=True)
class Reads:
    """Where the check reads one part's seams. `points` (E, 2, 2, 5, 2) texels, indexed (seam,
    side, near/far, along, x/y), at each side's chosen depth; `counts` (E, 5) which points along
    each seam count on both sides. A seam with no point that counts cannot be read."""

    seams: list[Seam]
    points: np.ndarray
    counts: np.ndarray

    @property
    def read(self) -> np.ndarray:
        """(E,) bool: the seams with at least one point that counts."""
        return self.counts.any(axis=1)

    @property
    def bends(self) -> np.ndarray:
        """(E,) each seam's bend."""
        return np.array([s.bend for s in self.seams], dtype=float)

    @property
    def need_read(self) -> np.ndarray:
        """(E,) bool: the seams whose bend is under 1, so their light could step past the
        allowance and only a read can tell. At a bend of 1 or more the allowance is at least the
        brighter side's light, which no step of light (never negative) can pass."""
        return self.bends < 1.0


def _placement(node: str) -> str:
    return node.rsplit("/", 1)[0]


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _seams(bake: BakeData, part: str) -> tuple[list[Seam], np.ndarray]:
    """The seam edges of `part`, triangle by triangle, and each triangle's island (-1 for a
    triangle with no lightmap area)."""
    pos, nrm, uv = bake.positions[part], bake.normals[part], bake.uv[part] * bake.size
    stored = bake.stored.get(part)  # the corners as the .glb holds them (float32, own frame)
    noise = normal_noise(stored) if stored is not None else np.zeros(len(uv))
    live = area2(uv) != 0.0
    extent = float(np.ptp(pos.reshape(-1, 3), axis=0).max()) if len(pos) else 0.0
    keys = np.round(pos / (WELD * extent if extent > 0.0 else 1.0)).astype(np.int64)
    edges: dict[tuple, list[tuple[int, int, int]]] = {}
    for t in np.nonzero(live)[0]:
        for a in range(3):
            b = (a + 1) % 3
            ka, kb = tuple(keys[t, a]), tuple(keys[t, b])
            edges.setdefault((min(ka, kb), max(ka, kb)), []).append((int(t), a, b))
    parent = list(range(len(uv)))
    found = []
    for (low, _high), sides in edges.items():
        if len(sides) != 2:
            continue
        (t1, a1, b1), (t2, a2, b2) = sides
        # side 2's corners, in side 1's order (a is the corner at `low` on both sides)
        if tuple(keys[t1, a1]) != low:
            a1, b1 = b1, a1
        if tuple(keys[t2, a2]) != low:
            a2, b2 = b2, a2
        gap = max(np.abs(uv[t1, a1] - uv[t2, a2]).max(), np.abs(uv[t1, b1] - uv[t2, b2]).max())
        if gap <= SAME_UV:  # one island: the UV runs on across the edge
            parent[_find(parent, t1)] = _find(parent, t2)
            continue
        ends = ((nrm[t1, a1], nrm[t2, a2]), (nrm[t1, b1], nrm[t2, b2]))
        bend = 0.0 if all(same_normal(n1, n2, noise[t1], noise[t2]) for n1, n2 in ends) else max(
            float(np.linalg.norm(n1 - n2)) for n1, n2 in ends)
        sides_uv = np.array([[uv[t1, a1], uv[t1, b1], uv[t1, 3 - a1 - b1]],
                             [uv[t2, a2], uv[t2, b2], uv[t2, 3 - a2 - b2]]])
        found.append(((t1, t2), sides_uv, np.array([pos[t1, a1], pos[t1, b1]]), bend))
    roots = np.array([_find(parent, t) for t in range(len(uv))], dtype=np.int64)
    _, island = np.unique(roots, return_inverse=True)
    island = np.where(live, island, -1)
    out = [Seam(tri, sides_uv, at, (int(island[tri[0]]), int(island[tri[1]])), bend)
           for tri, sides_uv, at, bend in found]
    return out, island


def _owners(bake: BakeData, part: str, island: np.ndarray) -> np.ndarray:
    """(size * size,) the island of `part` covering each texel (flat id), -1 for none, -2 for a
    texel two of its islands cover."""
    owner = np.full(bake.size * bake.size, -1, dtype=np.int64)
    uv = bake.uv[part]
    for one in np.unique(island[island >= 0]):
        ids = texels(uv[island == one], bake.size, edges=True)
        owner[ids] = np.where(owner[ids] == -1, one, -2)
    return owner


def _bilinear(pixels: np.ndarray, at: np.ndarray) -> np.ndarray:
    """The picture (H, W, 3) at points `at` (n, 2) given as (x, y) in texels (a texel's centre is
    at +0.5; y counts from the top row), bilinear, edges clamped: (n, 3)."""
    h, w = pixels.shape[:2]
    x, y = at[:, 0] - 0.5, at[:, 1] - 0.5
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    xi = np.clip(np.stack([x0, x0 + 1]).astype(int), 0, w - 1)
    yi = np.clip(np.stack([y0, y0 + 1]).astype(int), 0, h - 1)
    px = pixels.astype(np.float64)
    return ((1 - fx) * (1 - fy) * px[yi[0], xi[0]] + fx * (1 - fy) * px[yi[0], xi[1]]
            + (1 - fx) * fy * px[yi[1], xi[0]] + fx * fy * px[yi[1], xi[1]])


def _tap_ids(points: np.ndarray, size: int) -> np.ndarray:
    """(..., 4) the flat ids of the 4 texels a bilinear read of each point (..., 2) takes; -1 for a
    tap outside the picture."""
    x0 = np.floor(points[..., 0] - 0.5).astype(np.int64)
    y0 = np.floor(points[..., 1] - 0.5).astype(np.int64)
    out = []
    for dy in (0, 1):
        for dx in (0, 1):
            x, y = x0 + dx, y0 + dy
            inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
            out.append(np.where(inside, y * size + x, -1))
    return np.stack(out, axis=-1)


def _candidates(seam: Seam) -> np.ndarray:
    """Every read point the check may take on both sides: (2, D, 2, 5, 2) texels, indexed (side,
    depth in INSETS, near/far, along, x/y); near is at the depth inside the triangle, far at twice
    it."""
    t = np.array(ALONG)[:, None]
    out = []
    for a, b, third in seam.uv:
        edge = b - a
        perp = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
        if perp @ (third - a) < 0.0:
            perp = -perp
        out.append([[a + edge * t + k * d * perp for k in (1.0, 2.0)] for d in INSETS])
    return np.array(out)


def _reads(bake: BakeData, part: str) -> Reads:
    """Where the check reads each seam of `part` (see the module's docstring)."""
    seams, island = _seams(bake, part)
    if not seams:
        return Reads([], np.zeros((0, 2, 2, len(ALONG), 2)), np.zeros((0, len(ALONG)), bool))
    owner = _owners(bake, part, island)
    every = np.array([_candidates(s) for s in seams])  # (E, 2, D, 2, 5, 2)
    taps = _tap_ids(every, bake.size)  # (E, 2, D, 2, 5, 4)
    own = np.array([s.island for s in seams])[:, :, None, None, None, None]  # (E, 2, 1, 1, 1, 1)
    good = ((taps >= 0) & (owner[np.maximum(taps, 0)] == own)).all(axis=-1)  # (E, 2, D, 2, 5)
    pair = good.all(axis=3)  # (E, 2, D, 5): both reads of one depth count
    depth = pair.argmax(axis=2)  # the first depth that counts
    has = pair.any(axis=2)  # (E, 2, 5)
    pick = depth[:, :, None, None, :, None]  # (E, 2, 1, 1, 5, 1)
    points = np.take_along_axis(every, np.broadcast_to(
        pick, (*every.shape[:2], 1, *every.shape[3:])), axis=2)[:, :, 0]  # (E, 2, 2, 5, 2)
    return Reads(seams, points, has[:, 0] & has[:, 1])


def _judged(bake: BakeData, placement: str) -> list[str]:
    """The layers a part of `placement` is judged in: the picture loaded at `size` x `size`, and a
    noise recorded for the placement."""
    return [name for name, one in bake.layers.items()
            if one.pixels is not None and one.pixels.shape[:2] == (bake.size, bake.size)
            and placement in one.noise]


def _threshold(bake: BakeData, layer: str, placement: str) -> float:
    """The step an edge may show: SEAM_JUMP_SIGMAS x sqrt(5 (ns_a^2 + ns_b^2) + e_a^2 + e_b^2),
    `ns` the shipped noise of a texel inside an island (noise / 2; an edge texel's is up to
    noise, see the module docstring) and `e` the recorded error of each side's placement; a seam
    lies inside one part, so both sides are `placement`."""
    one = bake.layers[layer]
    ns = one.noise[placement] / 2.0
    e = one.error.get(placement, 0.0)
    return SEAM_JUMP_SIGMAS * math.sqrt(5.0 * (ns * ns + ns * ns) + e * e + e * e)


def _wanted(only: Collection[str] | None) -> set[str] | None:
    if only is None:
        return None
    return {m.group(1) for item in only if (m := _ITEM.fullmatch(item))}


def _sides(pixels: np.ndarray, reads: Reads) -> np.ndarray:
    """(E, 2, 3) each seam's two sides' light at the edge: the mean, over the points that count,
    of each side's extrapolated edge value `2 v(d) - v(2 d)`. A seam with no point that counts
    gets nan."""
    shape = reads.points.shape  # (E, 2, 2, 5, 2)
    got = _bilinear(pixels, reads.points.reshape(-1, 2)).reshape(*shape[:-1], 3)
    edge = 2.0 * got[:, :, 0] - got[:, :, 1]  # (E, 2, 5, 3): each side's light at the edge
    many = reads.counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (edge * reads.counts[:, None, :, None]).sum(axis=2) / many[:, None, None]


def _steps(sides: np.ndarray) -> np.ndarray:
    """(E,) each seam's step: the channel RMS of the two sides' difference (`_sides`). The mean of
    the differences is the difference of the means, as both sides count at the same points."""
    return np.sqrt(np.mean((sides[:, 0] - sides[:, 1]) ** 2, axis=-1))


def _allowances(sides: np.ndarray, bends: np.ndarray) -> np.ndarray:
    """(E,) the step each seam's bend explains, by Lambert's law: the channel RMS of
    `bend x max(|v_a|, |v_b|)` (see the module docstring)."""
    most = np.abs(sides).max(axis=1)  # (E, 3)
    return bends * np.sqrt(np.mean(most ** 2, axis=-1))


def _lines(bake: BakeData, layer: str, placement: str, reads: Reads,
           sides: np.ndarray) -> np.ndarray:
    """(E,) each seam's line: the noise and error threshold plus its Lambert allowance."""
    return _threshold(bake, layer, placement) + _allowances(sides, reads.bends)


def _jumps(bake: BakeData, only: Collection[str] | None = None) -> list[tuple]:
    """(part, layer, over, of, worst step, its line, its bend, where) for each part and layer
    with a read seam edge over its line: `over` edges of the `of` read, the one furthest over its
    line, with its step, line, bend and the world position of its first end."""
    wanted = _wanted(only)
    out = []
    for part in sorted(bake.uv):
        if wanted is not None and part not in wanted:
            continue
        placement = _placement(part)
        layers = _judged(bake, placement)
        if not layers:
            continue
        reads = _reads(bake, part)
        read = reads.read
        if not read.any():
            continue
        for layer in layers:
            sides = _sides(bake.layers[layer].pixels, reads)
            step = np.where(read, _steps(sides), -np.inf)
            line = np.where(read, _lines(bake, layer, placement, reads, sides), np.inf)
            over = step > line
            if over.any():
                worst = int(np.where(over, step - line, -np.inf).argmax())
                out.append((part, layer, int(over.sum()), int(read.sum()), float(step[worst]),
                            float(line[worst]), reads.seams[worst].bend,
                            reads.seams[worst].at[0]))
    return out


def _unread(bake: BakeData) -> list[tuple[str, int, int]]:
    """(part, unread, of) for each part judged in some layer with a seam edge that needs a read
    (a bend under 1, `Reads.need_read`) and cannot be read: `unread` of its `of` such edges."""
    out = []
    for part in sorted(bake.uv):
        if not _judged(bake, _placement(part)):
            continue
        reads = _reads(bake, part)
        need = reads.need_read
        unread = need & ~reads.read
        if unread.any():
            out.append((part, int(unread.sum()), int(need.sum())))
    return out


def _side_two_taps(bake: BakeData, reads: Reads, index: int) -> np.ndarray:
    """The texels the check reads on side two of seam `index`, less the ones it reads on side
    one: the taps of its counted points only."""
    counts = reads.counts[index]
    taps = _tap_ids(reads.points[index], bake.size)  # (2, 2, 5, 4)
    one = taps[0][:, counts].ravel()
    two = taps[1][:, counts].ravel()
    return np.setdiff1d(two[two >= 0], one)


def _plant(ctx: SceneCtx, layers: list[str], ids: np.ndarray, placement: str, bright: bool,
           reads: Reads, index: int) -> SceneCtx:
    new = dict(ctx.bake.layers)
    bend = reads.seams[index].bend
    for name in layers:
        one = ctx.bake.layers[name]
        flat = one.pixels.reshape(-1, 3).copy()
        if bright:
            flat[ids] *= BRIGHTEN
        else:
            sides = _sides(one.pixels, reads)
            line = float(_lines(ctx.bake, name, placement, reads, sides)[index])
            flat[ids] += STEP_SIGMAS * line / (1.0 - bend)
        new[name] = dataclasses.replace(one, pixels=flat.reshape(one.pixels.shape))
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers=new))


class SeamJump:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a scene check: it measures no piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        bake = ctx.bake
        if bake is None:
            return []
        best = None  # (bend, part, placement, layers, reads, index, ids)
        for part in sorted(bake.uv):
            placement = _placement(part)
            layers = _judged(bake, placement)
            if not layers:
                continue
            reads = _reads(bake, part)
            for index in np.nonzero(reads.read & reads.need_read)[0]:
                bend = reads.seams[int(index)].bend
                if best is not None and bend >= best[0]:
                    continue
                ids = _side_two_taps(bake, reads, int(index))
                if len(ids):
                    best = (bend, part, placement, layers, reads, int(index), ids)
        if best is not None:
            _, part, placement, layers, reads, index, ids = best
            return [unit(f"seam {part} jump", [
                ("one island's light is 50% brighter",
                 _plant(ctx, layers, ids, placement, True, reads, index)),
                ("one island's light is a step above",
                 _plant(ctx, layers, ids, placement, False, reads, index))])]
        return [PlantUnit("seam jump", (), not_applicable=(
            "no seam edge with a bend under 1 that can be read, with a loaded picture and a "
            "recorded noise, to plant on"))]

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        if ctx.bake is None:
            return []
        rows: dict[str, list[str]] = {}
        for part, layer, over, of, step, line, bend, at in _jumps(ctx.bake, only):
            where = ", ".join(f"{v:.4g}" for v in at)
            rows.setdefault(part, []).append(
                f"layer {layer!r}: {over} of {of} read seam edges step past their line; the "
                f"worst steps by {step:.4g} over a line of {line:.4g} (bend {bend:.3g}) at "
                f"({where})")
        return [Finding(NAME, f"seam {part} jump", f"light steps across a lightmap seam of "
                                                   f"{part}: " + "; ".join(lines))
                for part, lines in sorted(rows.items())
                if only is None or f"seam {part} jump" in only]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        if ctx.bake is None:
            return []
        return [Unmeasured(
            f"{NAME}: {unread} of {of} seam edges of {part} with a bend under 1 cannot be read: a "
            "side's island holds no bilinear read inside it at both depths (the unwrap is too "
            "thin there), so the light across them is not judged")
            for part, unread, of in _unread(ctx.bake)]
