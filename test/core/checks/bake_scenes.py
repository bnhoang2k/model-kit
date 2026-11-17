"""Two scenes in code with a bake: a slab with two boxes, and a shelf with three planks.

The bake is built in numpy, as `load_bake` would return it: each part's lightmap UV is a quad of
two triangles (an island) laid on a `SIZE` px picture with `MARGIN` px between neighbours, plus one
collapsed triangle (an unseen face, all three corners at one point). `positions` are flat stand-ins:
the checks under test read the UV only.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from model_kit.core.bake_data import BakeData, BakeLayer, texels
from model_kit.core.scene import SceneCtx

from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base

SIZE = 64  # the picture's side in pixels
LIGHT_NOISE = 0.02  # the bake's recorded noise for each placement, per doubled texel
MARGIN = 2  # pixels between two islands, as the bake packs them


def island(x: float, y: float, w: float, h: float, size: int = SIZE) -> np.ndarray:
    """A quad of two triangles at pixel (x, y), w by h px, then one collapsed triangle at its
    corner: (3, 3, 2) in 0..1."""
    quad = np.array([[(x, y), (x + w, y), (x + w, y + h)],
                     [(x, y), (x + w, y + h), (x, y + h)],
                     [(x, y), (x, y), (x, y)]], dtype=float)
    return quad / size


def _bake(uv: dict[str, np.ndarray], files=("lamp.exr",), layers=("lamp",)) -> BakeData:
    placements = {n.rsplit("/", 1)[0] for n in uv}
    return BakeData(
        mode="full",
        layers={name: BakeLayer(picture=f"{name}.exr", pixels=np.zeros((SIZE, SIZE, 3), np.float32),
                                noise={p: LIGHT_NOISE for p in placements},
                                error={p: LIGHT_NOISE / 2 for p in placements})
                for name in layers},
        uv=uv, positions={n: np.zeros((len(u), 3, 3)) for n, u in uv.items()},
        normals={n: np.tile([0.0, 0.0, 1.0], (len(u), 3, 1)) for n, u in uv.items()},
        size=SIZE, unseen=frozenset(), files=frozenset(files))


def _with(bake: BakeData) -> SceneCtx:
    return dataclasses.replace(piece_ctx(post_on_base()), bake=bake)


def slab_and_boxes() -> SceneCtx:
    """A big slab island and two boxes' islands to its right, each `MARGIN` px apart."""
    return _with(_bake({
        "slab/body": island(MARGIN, MARGIN, 30, 30),
        "Box/body": island(32 + MARGIN, MARGIN, 12, 12),
        "box_b/body": island(32 + MARGIN, 12 + 2 * MARGIN, 12, 12),
    }))


def shelf_of_planks() -> SceneCtx:
    """Three planks side by side, each `MARGIN` px from the next."""
    return _with(_bake({
        f"plank_{n}/body": island(MARGIN + i * (16 + MARGIN), MARGIN, 16, 40)
        for i, n in enumerate("abc")}))


SCENES = {"slab_and_boxes": slab_and_boxes, "shelf_of_planks": shelf_of_planks}

# the part to move, and the part it is moved onto, in each scene
MOVES = {"slab_and_boxes": ("Box/body", "slab/body"),
         "shelf_of_planks": ("plank_b/body", "plank_a/body")}


def moved_onto(ctx: SceneCtx, part: str, onto: str) -> SceneCtx:
    """`part`'s island shifted so its corner sits 4 px inside `onto`'s corner."""
    here, there = ctx.bake.uv[part], ctx.bake.uv[onto]
    shift = there[0, 0] + 4 / SIZE - here[0, 0]
    uv = {**ctx.bake.uv, part: here + shift}
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, uv=uv))


# --- scenes with light: a surface cut into islands, a picture from a light field ---------------

CELL = 30  # the side of the pixel cell each triangle's island sits in

def light(pos: np.ndarray) -> np.ndarray:
    """A light field continuous in the world: a gentle ramp in x and y, per channel."""
    return np.stack([0.1 * pos[..., 0] + 0.2, 0.05 * pos[..., 1] + 0.1,
                     np.full(pos[..., 0].shape, 0.3)], axis=-1)


def surface(quads: int) -> tuple[np.ndarray, np.ndarray]:
    """`quads` unit squares side by side in z = 0, two triangles each: positions (n, 3, 3) and the
    +z normals."""
    tris = []
    for q in range(quads):
        p00, p10, p11, p01 = ((q, 0, 0), (q + 1, 0, 0), (q + 1, 1, 0), (q, 1, 0))
        tris += [(p00, p10, p11), (p00, p11, p01)]
    pos = np.array(tris, dtype=float)
    return pos, np.tile([0.0, 0.0, 1.0], (len(pos), 3, 1))


def island_of(index: int) -> np.ndarray:
    """Triangle `index`'s corners in pixels: a right triangle in its own cell of a grid of cells
    two across, MARGIN px inside the cell."""
    x0 = (index % 2) * CELL + MARGIN
    y0 = (index // 2) * CELL + MARGIN
    side = CELL - 2 * MARGIN
    return np.array([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side)], dtype=float)


def half_black_rim(pixels: np.ndarray, tri: np.ndarray, x0: int, y0: int) -> None:
    """In the cell at (x0, y0), the texels the triangle `tri` (3, 2) px does not cover (centre
    inside or on an edge): halved when one touches a covered texel (8 neighbours), black beyond.
    The margin a plain 2 x 2 shrink of a 1-pixel fill leaves."""
    ids = texels(tri[None] / SIZE, SIZE, edges=True)
    covered = np.zeros((SIZE, SIZE), dtype=bool)
    covered.flat[ids] = True
    near = np.zeros_like(covered)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near |= np.roll(np.roll(covered, dy, axis=0), dx, axis=1)
    cell = np.zeros_like(covered)
    cell[y0:y0 + CELL, x0:x0 + CELL] = True
    pixels[cell & near & ~covered] *= 0.5
    pixels[cell & ~near] = 0.0


def seamed(quads: int, name: str = "slab", *, hard: bool = False, step: float = 0.0,
           collapse_second: bool = False, noise: float = LIGHT_NOISE, grain: bool = True,
           field=None, error: float | None = None, offset=(0.0, 0.0),
           rim: bool = False) -> SceneCtx:
    """One part `<name>/body`: `quads` unit squares, each triangle in its own island, so each
    shared edge is a seam. The picture holds `light` at each texel's point of the surface, and
    every texel of the triangle's cell (outside the triangle too: the margin fill) takes the
    triangle's own linear extension, so a bilinear read near an edge stays on it. `step` is added
    to every texel of each odd triangle; `hard` tilts each odd triangle's normals; `collapse_second`
    gives the second triangle one point for its three corners. `grain` adds Gaussian noise of
    the shipped texel's size, `noise / 2` (fixed seed), to every texel; `field` replaces `light`.
    `error` is the recorded resolution error (default: `noise`); `offset` (px, under 1) shifts
    every triangle off the texel grid; `rim` makes each cell's margin half black next to its
    triangle and black beyond (`half_black_rim`)."""
    pos, normals = surface(quads)
    if hard:
        normals = normals.copy()
        normals[1::2] = [0.0, 0.6, 0.8]
    return _islands_part(name, pos, normals,
                         lambda i, at: (field or light)(at) + (step if i % 2 else 0.0),
                         noise=noise, grain=grain, error=error, offset=offset, rim=rim,
                         collapse_second=collapse_second)


def _islands_part(name: str, pos: np.ndarray, normals: np.ndarray, value_of, *,
                  noise: float = LIGHT_NOISE, grain: bool = True, error: float | None = None,
                  offset=(0.0, 0.0), rim: bool = False,
                  collapse_second: bool = False) -> SceneCtx:
    """One part `<name>/body` of the triangles `pos` (n, 3, 3) with vertex `normals`, each
    triangle in its own island (`island_of`). Each texel of triangle i's cell takes
    `value_of(i, world points)` at its point of the triangle's plane; the rest as `seamed` says."""
    uv = (np.array([island_of(i) for i in range(len(pos))]) + np.asarray(offset)) / SIZE
    pixels = np.zeros((SIZE, SIZE, 3), np.float32)
    for i, (p, px) in enumerate(zip(pos, uv * SIZE, strict=True)):
        # the affine map from pixel (x, y, 1) to the world, from the triangle's three corners
        to_world = np.linalg.solve(np.c_[px, np.ones(3)], p)
        x0, y0 = int(px[:, 0].min()) - MARGIN, int(px[:, 1].min()) - MARGIN
        ys, xs = np.mgrid[y0:y0 + CELL, x0:x0 + CELL]
        at = np.stack([xs + 0.5, ys + 0.5, np.ones(xs.shape)], axis=-1) @ to_world
        pixels[y0:y0 + CELL, x0:x0 + CELL] = value_of(i, at)
        if rim:
            half_black_rim(pixels, px, x0, y0)
    if grain:
        pixels += np.random.default_rng(7).normal(0.0, noise / 2, pixels.shape).astype(np.float32)
    if collapse_second:
        uv[1] = uv[1][0]
    part = f"{name}/body"
    error = noise if error is None else error
    layer = BakeLayer(picture="lamp.exr", noise={name: noise}, error={name: error}, pixels=pixels)
    bake = BakeData(mode="full", layers={"lamp": layer}, uv={part: uv}, positions={part: pos},
                    normals={part: normals}, size=SIZE, unseen=frozenset(),
                    files=frozenset({"lamp.exr"}))
    return _with(bake)


# The bent strip: 4 flat facets round an arc of this radius, bottom corners STRIP_STEP apart in
# angle and top corners offset by half a step, so no two neighbouring facets are coplanar.
STRIP_RADIUS = 1.0
STRIP_STEP = np.radians(20.0)
STRIP_SUN = np.array([0.0, 0.0, 1.0])  # toward the light: along the arc's middle normal
STRIP_SUN_LIGHT = np.array([0.5, 0.4, 0.3])  # the sun's light per channel at normal incidence
STRIP_SKY = np.array([0.08, 0.1, 0.12])  # light from all round: the same on every facet


def strip_facets() -> tuple[np.ndarray, np.ndarray]:
    """The bent strip's 4 triangles (4, 3, 3) and their flat normals (4, 3, 3), facing +z on
    average: a fan of facets round an arc in the x-z plane, one unit high in y."""
    def at(angle: float, y: float) -> np.ndarray:
        return np.array([STRIP_RADIUS * np.sin(angle), y, STRIP_RADIUS * np.cos(angle)])
    bottom = [at((k - 1.0) * STRIP_STEP, 0.0) for k in range(3)]
    top = [at((k - 0.5) * STRIP_STEP, 1.0) for k in range(3)]
    tris = np.array([(bottom[0], bottom[1], top[0]), (bottom[1], top[1], top[0]),
                     (bottom[1], bottom[2], top[1]), (bottom[2], top[2], top[1])])
    face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    face *= np.sign(face[:, 2:3])  # outward, toward +z
    return tris, np.repeat(face[:, None, :], 3, axis=1)


def bent_strip(name: str = "strip", *, step: float = 0.0, noise: float = LIGHT_NOISE,
               grain: bool = True, error: float | None = None) -> SceneCtx:
    """One part `<name>/body`: the 4 flat facets of `strip_facets`, each its own island, so every
    seam is at a bend. Each facet's light is Lambert's law under a sun along the arc's middle
    normal, `sun x max(0, n . STRIP_SUN)`, plus the same sky light on every facet: constant
    across a facet, stepping at each bend only as the normal does. `step` is added to every texel
    of each odd facet."""
    pos, normals = strip_facets()
    lambert = np.maximum(0.0, normals[:, 0] @ STRIP_SUN)[:, None] * STRIP_SUN_LIGHT + STRIP_SKY

    def value_of(i: int, at: np.ndarray) -> np.ndarray:
        return np.broadcast_to(lambert[i] + (step if i % 2 else 0.0), (*at.shape[:-1], 3))
    return _islands_part(name, pos, normals, value_of, noise=noise, grain=grain, error=error)


def smooth_slab() -> SceneCtx:
    return seamed(1, "slab")


def smooth_shelf() -> SceneCtx:
    return seamed(2, "shelf")


SEAM_SCENES = {"slab": smooth_slab, "shelf": smooth_shelf, "strip": bent_strip}
