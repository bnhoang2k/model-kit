"""Scenes of one assembly piece for the bearing tests. A helper module.

Every piece is `kit:Thing`, from `piece_scenes.make_ctx`: it declares the view whose seen floor is
`piece_scenes.TOUCH` (about 21 mm), its contact length and so the touch distance. Sizes are in
metres
(the pack is one unit a metre), y up.
"""

import dataclasses
import re

import numpy as np

from model_kit.core import geom
from model_kit.core.checks import pieces
from model_kit.core.scene import SceneCtx, piece_key
from model_kit.core.tolerances import CROSS_OF_PART_THICKNESS, WELD_OF_PIECE_SIZE

from ..shapes import box, cylinder
from .piece_scenes import make_ctx, with_parts


def noise_of(ctx: SceneCtx) -> float:
    """The overhang tolerance for `kit:Thing`: float noise at the piece's size."""
    size = float(np.linalg.norm(np.ptp(np.concatenate(
        [t.reshape(-1, 3) for t in ctx.pieces[piece_key("kit", "Thing")].parts.values()]),
        axis=0)))
    return WELD_OF_PIECE_SIZE * size


def shifted(ctx: SceneCtx, part: str, by) -> SceneCtx:
    """The scene with one part of `kit:Thing` moved by a vector."""
    tris = ctx.pieces[piece_key("kit", "Thing")].parts[part]
    return with_parts(ctx, **{part: tris + np.asarray(by, dtype=float)})


def stack(*, x: float = 0.0):
    """A slab and a block standing on it, `x` to the side. At 0 its centre is over the slab."""
    parts = {"slab": box((-0.3, 0.0, -0.3), (0.3, 0.1, 0.3)),
             "block": box((x - 0.1, 0.1, -0.1), (x + 0.1, 0.3, 0.1))}
    return make_ctx(parts, {"block": {"on": "slab"}})


def _box_post(top: float, sink: float):
    return box((-0.02, top - sink, -0.02), (0.02, top + 0.1 - sink, 0.02))


def _round_post(top: float, sink: float):
    return cylinder((0.0, 0.0), 0.025, top - sink, top + 0.1 - sink, sides=16)


# Two pieces of different shape: a square post on a 20 mm base, and a round (16-sided) post on a
# 30 mm base. The thinner part is the base in both. The read of the depth slack, one for crossings
# and bearing, is on these two.
SUNK_PIECES = {"square post on a thin base": (0.02, _box_post),
               "round post on a thicker base": (0.03, _round_post)}


def _sunk_parts(piece: str, sink: float):
    top, post = SUNK_PIECES[piece]
    return {"base": box((-0.15, 0.0, -0.15), (0.15, top, 0.15)), "post": post(top, sink)}


def half_of_thinner(piece: str) -> float:
    """`CROSS_OF_PART_THICKNESS` times the thinner part's thickness where the post stands, read
    the way both checks read it."""
    parts = _sunk_parts(piece, 0.0)
    return CROSS_OF_PART_THICKNESS * geom.thinner_thickness(
        parts["base"], parts["post"], (0.0, SUNK_PIECES[piece][0], 0.0))


def sunk_post(piece: str, share: float):
    """The piece with its post sunk into the base by `share` of the depth crossings fails at."""
    return make_ctx(_sunk_parts(piece, share * half_of_thinner(piece)), {"post": {"on": "base"}})


# Two pieces of different shape whose parts are thick against the contact length (TOUCH, 21 mm):
# a square post on a 400 mm base, and a wide block on a 500 mm base. The thinner part is over
# eight contact lengths thick in both, so a push of twice the contact length stays under half of
# it: contact is still found and crossings stays quiet.
PUSHED_PIECES = {"square post on a thick base": (0.4, lambda top: box(
                     (-0.1, top, -0.1), (0.1, top + 0.4, 0.1))),
                 "wide block on a thicker base": (0.5, lambda top: box(
                     (-0.2, top, -0.1), (0.2, top + 0.4, 0.1)))}


def pushed_post(piece: str, by: float):
    """The piece with its post pushed `by` into the base: the post's foot is `by` under the base's
    top face. The numbers come from the shapes above and `by`, never from a check."""
    top, post = PUSHED_PIECES[piece]
    parts = {"base": box((-0.3, 0.0, -0.3), (0.3, top, 0.3)), "post": post(top - by)}
    return make_ctx(parts, {"post": {"on": "base"}})


# Two pieces whose resting part is taller than it is wide (its side faces outweigh its foot): a
# square post 0.8 high and a flat post 0.6 high, each on a 400 mm base. Each is over eight contact
# lengths thick across, so a push of twice the contact length keeps crossings quiet.
TALL_PIECES = {
    "square post 0.8 high": lambda top: box((-0.06, top, -0.06), (0.06, top + 0.8, 0.06)),
    "flat post 0.6 high": lambda top: box((-0.075, top, -0.05), (0.075, top + 0.6, 0.05))}


def tall_post(piece: str, by: float):
    """The piece with its tall post pushed `by` into a 400 mm base (its foot `by` under the top)."""
    parts = {"base": box((-0.3, 0.0, -0.3), (0.3, 0.4, 0.3)), "post": TALL_PIECES[piece](0.4 - by)}
    return make_ctx(parts, {"post": {"on": "base"}})


# Two pieces whose resting part straddles its host's edge: a block half over the end of a 400 mm
# slab (as a fret hangs over a fingerboard's end), and a post at its corner, over two edges.
STRADDLE_PIECES = {
    "block over the slab's end": lambda top: box((0.2, top, -0.1), (0.6, top + 0.2, 0.1)),
    "post over the slab's corner": lambda top: box((0.34, top, 0.24), (0.46, top + 0.4, 0.36))}


def straddling(piece: str, by: float):
    """The piece with its straddling part pushed `by` into the slab's top face."""
    parts = {"slab": box((-0.4, 0.0, -0.3), (0.4, 0.4, 0.3)),
             "part": STRADDLE_PIECES[piece](0.4 - by)}
    return make_ctx(parts, {"part": {"on": "slab"}})


def prism(profile, z0: float, z1: float) -> np.ndarray:
    """A closed prism with outward triangles: the convex (x, y) `profile`, counter-clockwise,
    from z0 to z1."""
    pts = np.asarray(profile, dtype=float)
    lo = np.column_stack([pts, np.full(len(pts), z0)])
    hi = np.column_stack([pts, np.full(len(pts), z1)])
    out = []
    for i in range(len(pts)):
        j = (i + 1) % len(pts)
        out += [[lo[i], lo[j], hi[j]], [lo[i], hi[j], hi[i]]]
    for k in range(1, len(pts) - 1):
        out += [[hi[0], hi[k], hi[k + 1]], [lo[0], lo[k + 1], lo[k]]]
    tris = np.array(out)
    centre = tris.reshape(-1, 3).mean(axis=0)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    flip = np.einsum("ij,ij->i", n, tris.mean(axis=1) - centre) < 0
    tris[flip] = tris[flip][:, ::-1]
    return tris


def bevelled_block(angle_deg: float, run: float = 0.1) -> np.ndarray:
    """A block 0.6 long and 0.2 high whose top's +x end falls away in a bevel `run` long, turned
    `angle_deg` from the top: a knob's chamfered rim, in section."""
    drop = run * np.tan(np.radians(angle_deg))
    return prism([(-0.3, 0.0), (0.3, 0.0), (0.3, 0.2 - drop), (0.3 - run, 0.2), (-0.3, 0.2)],
                 -0.2, 0.2)


def against_a_faceted_round(radius: float, sides: int):
    """An upright `sides`-gon of `radius` (a lathed base) with a block pressed end on against its
    side, centred on the crease between two facets (as a cable meets its base). Each facet
    turns 360 / `sides` degrees from the next, under the crease angle (35): one round surface."""
    host = cylinder((0.0, 0.0), radius, 0.0, 0.5, sides=sides)
    part = box((radius, 0.1, -0.12), (radius + 0.3, 0.3, 0.12))
    return make_ctx({"base": host, "block": part}, {"block": {"on": "base"}})


def tabletop_on_legs():
    """A top and four legs, each leg `on` the top, its top face wholly under the top."""
    parts = {"top": box((-0.5, 0.7, -0.3), (0.5, 0.8, 0.3))}
    specs = {}
    for i, (x, z) in enumerate([(0.45, 0.25), (-0.45, 0.25), (0.45, -0.25), (-0.45, -0.25)]):
        parts[f"leg_{i}"] = box((x - 0.025, 0.0, z - 0.025), (x + 0.025, 0.7, z + 0.025))
        specs[f"leg_{i}"] = {"on": "top"}
    return make_ctx(parts, specs, hosts=["top"])


def hanging_clamp(*, x: float = 0.0):
    """A rail and a clamp hanging under it, `x` to the side."""
    parts = {"rail": box((-0.4, 0.8, -0.05), (0.4, 0.9, 0.05)),
             "clamp": box((x - 0.05, 0.5, -0.04), (x + 0.05, 0.8, 0.04))}
    return make_ctx(parts, {"clamp": {"on": "rail"}}, hosts=["rail"])


def sideways():
    """A wall and a block fixed to its side face: gravity decides nothing for it."""
    parts = {"wall": box((-0.05, 0.0, -0.3), (0.0, 0.5, 0.3)),
             "shelf": box((0.0, 0.2, -0.1), (0.1, 0.3, 0.1))}
    return make_ctx(parts, {"shelf": {"on": "wall"}})


def bolt(*, sunk: float, shank: float = 0.01, head: float = 0.03, top: float = 0.4):
    """A bolt standing in a post whose top face is at `top`: a round shank (24 sides, `shank`
    across its corners, so the bolt's diameter is exactly `shank`) sunk `sunk` into it and
    standing 20 mm out, with a head `head` wide and 5 mm high. One mesh of a closed prism and a
    closed box."""
    half, big = shank / 2, head / 2
    stem = cylinder((0.0, 0.0), half, top - sunk, top + 0.02)
    cap = box((-big, top + 0.02, -big), (big, top + 0.025, big))
    return np.concatenate([stem, cap])


def bolted_post(sunk: float, **kwargs):
    """A post (0.2 wide, 0.4 high) with `bolt(sunk=...)` in its top face."""
    parts = {"post": box((-0.1, 0.0, -0.1), (0.1, 0.4, 0.1)), "bolt": bolt(sunk=sunk, **kwargs)}
    return make_ctx(parts, {"bolt": {"on": "post", "fastener": True}}, hosts=["post"])


def wedge_on_block(*, slope: float = 0.3):
    """A block and a long bar sheared up off it: the bar's under face touches the block only at
    its low end and climbs away at `slope`, to a metre past the block's edge."""
    parts = {"block": box((-0.2, -0.1, -0.2), (0.2, 0.0, 0.2))}
    bar = box((0.0, 0.0, -0.1), (1.0, 0.05, 0.1))
    bar[..., 1] += slope * bar[..., 0]
    parts["bar"] = bar
    return make_ctx(parts, {"bar": {"on": "block"}})


def ledge_and_wall():
    """A wall with a thin ledge on its side, and a shelf standing on the ledge against the wall.
    The shelf's patch is the ledge's top (small) and the wall's side (large): sideways as a whole."""
    wall = np.concatenate([box((-0.05, 0.0, -0.3), (0.0, 0.5, 0.3)),
                           box((0.0, 0.19, -0.1), (0.02, 0.2, 0.1))])
    parts = {"wall": wall, "shelf": box((0.0, 0.2, -0.1), (0.1, 0.3, 0.1))}
    return make_ctx(parts, {"shelf": {"on": "wall"}})


def held_bar(*, lift: float = 0.0):
    """A post and a clip that holds it (`retains`), `lift` off the post's side face."""
    parts = {"post": box((-0.05, 0.0, -0.05), (0.05, 0.5, 0.05)),
             "clip": box((0.05 + lift, 0.2, -0.04), (0.12 + lift, 0.3, 0.04))}
    return make_ctx(parts, {"post": {}, "clip": {"on": "post", "retains": ["post"]}},
                    hosts=["post"])


def declare(ctx: SceneCtx, pack: str, piece: str, *, fasteners=(), retains=None,
            hosts=None) -> SceneCtx:
    """The scene with the newer fields declared in memory: parts whose names match any of the
    `fasteners` patterns become fasteners, `retains` maps a part name to the names it holds, and
    `hosts` maps a part name to the part it is `on` instead of the one the pack gives it.
    The regression fixtures do not carry these fields."""
    record = ctx.packs[pack].pieces[piece]
    retains, hosts = retains or {}, hosts or {}
    parts = tuple(dataclasses.replace(
        p, fastener=any(re.match(pattern, p.name) for pattern in fasteners),
        retains=tuple(retains.get(p.name, ())), on=hosts.get(p.name, p.on))
        for p in record.parts)
    return pieces.with_piece(ctx, pack, piece, dataclasses.replace(record, parts=parts))


def creased(ctx: SceneCtx) -> SceneCtx:
    """The scene with the crease angle (35 degrees) set in memory. The
    regression fixtures do not declare `dials.edges.crease_deg`, which the seat rule reads."""
    dials = {**ctx.style.dials, "edges": {**ctx.style.dials.get("edges", {}), "crease_deg": 35,
                                          "crease_why": "the shading angle of the build"}}
    return dataclasses.replace(ctx, style=dataclasses.replace(ctx.style, dials=dials))


def bond(ctx: SceneCtx) -> SceneCtx:
    """The scene with every parts-list row of `kit:Thing` held `bonded` ."""
    record = ctx.packs["kit"].pieces["Thing"]
    rows = tuple(dataclasses.replace(r, held="bonded") for r in record.listed)
    return pieces.with_piece(ctx, "kit", "Thing", dataclasses.replace(record, listed=rows))


def rigged(ctx: SceneCtx, bones: dict[str, tuple[str, ...]]) -> SceneCtx:
    """The scene with `kit:Thing` rigged (the humanoid body's rig stands in) and the named parts
    bound to the given bones."""
    from ..bodies.humanoid import humanoid_scene
    rig = humanoid_scene().packs["humanoid"].pieces["Person"].rig
    record = ctx.packs["kit"].pieces["Thing"]
    parts = tuple(dataclasses.replace(p, bones=bones.get(p.name, p.bones)) for p in record.parts)
    return pieces.with_piece(ctx, "kit", "Thing",
                             dataclasses.replace(record, parts=parts, rig=rig))
