"""Leaf cards for a game-weight copy of a grown tree.

`balls` gathers a skeleton's leaves into round balls (twig sprays, biggest first, each joining the
nearest ball within a radius). `card_leaves` covers each ball with leaf cards: they sit on real leaf
centres, face the sky or the ground, shade by their ball and the crown, and are dropped when buried
in a neighbouring ball and dark. Core stays numpy-only.

Ported from Razel's "Just a simple tree", github.com/MikeFernandez-Pro/just-a-simple-tree,
MIT License, Copyright (c) 2026 Mike Fernandez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.grow.card_image import PAINTED_SHARE, coverage, paint_cells
from model_kit.core.grow.mesh import LeafLean, limb_tris, tube_tris
from model_kit.core.grow.skeleton import Skeleton
from model_kit.core.sight import Sight
from model_kit.core.units import Style

CARDS_PREFIX = "Cards_"  # a card piece's leaf part is `Cards_<shade>`


def card_part_name(shade: str) -> str:
    """The name of a card piece's leaf part, beside `mesh.leaf_part_name`."""
    return CARDS_PREFIX + shade


# Razel's constants (foliage.js names in the comments).
CARD_TILT = 0.6  # `tilt`: how far a card's face is turned off its ball's direction
ROUNDNESS = 0.58  # `roundness`: how far a card's corners are pulled onto the ball's sphere
BURIED = 0.8  # `buriedRadius`: a card inside this share of a neighbour's radius is buried
HIDDEN = 0.1  # `hiddenThreshold`: a card under this share of open light is dark
FADE_IN = 0.15  # `fadeIn`: the smoothstep reaches 1 at this share above its start
AO_RAYS = 12  # `aoRays`: cosine-weighted rays per card
AO_BOOST = 1.45  # `aoBoost`: colour at full light
AO_MIX = 0.54  # `aoMix`: how far the colour follows the light
AO_REACH_M = 2.5  # `aoDistance`: how far a ray looks for neighbours, in metres
LEAF_DENSITY = -math.log(0.1) / 2  # 10% of light gets through one ball diameter
GOLDEN = math.pi * (3 - math.sqrt(5))  # the golden angle, for the spiral spread
SPREAD_JITTER = 0.15  # a card sits 0.9 to 1.05 of the radius out
SIZE_DRAW = (0.9, 1.1)  # a card's side, times a draw in this range
SHOWN_CUT = 0.5  # a card shown under this share is dropped
_UP = np.array([0.0, 1.0, 0.0])
_QUAD = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
_QUV = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_TRIANGLES = ((0, 1, 2), (0, 2, 3))


@dataclass(frozen=True)
class CardDials:
    """How a card copy is built, read from style.json `dials.cards`. `ball_m`: the target ball
    radius, metres. `normal_mix`: the share of a card's normal that is its ball's direction (the
    rest points from the crown centre). `sky`: how hard a card's face is pushed to face up or
    down."""

    ball_m: float
    normal_mix: float
    sky: float


def card_dials(style: Style) -> CardDials:
    """The card dials of a style. A style with no `dials.cards` raises, naming it. A dial that is
    there was checked when the `Style` was built (`Style.validate`)."""
    dial = (style.dials or {}).get("cards")
    if dial is None:
        raise ManifestError("style: dials.cards is missing; it must hold ball_m, normal_mix "
                            "and sky (the card copy is a render dial)")
    return CardDials(ball_m=float(dial["ball_m"]), normal_mix=float(dial["normal_mix"]),
                     sky=float(dial["sky"]))


@dataclass(frozen=True)
class Ball:
    centre: np.ndarray  # (3,)
    radius: float
    leaves: np.ndarray  # (k,) leaf indices


@dataclass(frozen=True)
class CardLeaves:
    tris: np.ndarray  # (N, 3, 3)
    normals: np.ndarray  # (N, 3, 3)
    uvs: np.ndarray  # (N, 3, 2)
    colours: np.ndarray  # (N, 3, 3)
    centres: np.ndarray  # (cards, 3), before the wrap
    faces: np.ndarray  # (cards, 3), after the sky tilt
    cards: int
    dropped: int


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 0.0)


def smoothstep(x, a: float, b: float):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def leaf_centres(sk: Skeleton) -> np.ndarray:
    """(F, 3) the middle of each blade."""
    return sk.base + sk.mid * sk.length[:, None] / 2


def balls(sk: Skeleton, ball_r: float) -> list[Ball]:
    """Twig sprays gathered into balls near `ball_r` (model units). A leaf on the top level counts
    to its host's parent, any other leaf to its host. Each spray, biggest first, joins the nearest
    ball whose centre lies within `ball_r`, else starts one. A ball's radius is the 90th percentile
    of its leaf-centre distances plus half the mean blade length."""
    centres = leaf_centres(sk)
    top = len(sk.level_names) - 1
    group = np.where(sk.level[sk.host] == top, sk.parent[sk.host], sk.host)
    sprays = [np.flatnonzero(group == g) for g in np.unique(group)]
    sprays.sort(key=len, reverse=True)
    seeds: list[np.ndarray] = []
    members: list[list[np.ndarray]] = []
    for idx in sprays:
        c = centres[idx].mean(axis=0)
        if seeds:
            d = np.linalg.norm(np.array(seeds) - c, axis=1)
            k = int(d.argmin())
            if d[k] < ball_r:
                members[k].append(idx)
                continue
        seeds.append(c)
        members.append([idx])
    out = []
    for m in members:
        idx = np.concatenate(m)
        p = centres[idx]
        c = p.mean(axis=0)
        spread = float(np.percentile(np.linalg.norm(p - c, axis=1), 90))
        out.append(Ball(centre=c, radius=spread + 0.5 * float(sk.length[idx].mean()), leaves=idx))
    return out


def _blade_area_per_length2(outline: np.ndarray) -> float:
    x, y = outline[:, 0], outline[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def ball_cards(ball: Ball, sk: Skeleton, coverage: float) -> tuple[int, float]:
    """(cards before the cull, mean card side) of a ball: the side is the mean blade length over
    the painted share, and the count makes the cards' painted area (side^2 x coverage each) match
    the ball's real blade area, at least 1."""
    lengths = sk.length[ball.leaves]
    side = float(lengths.mean()) / PAINTED_SHARE
    area = _blade_area_per_length2(sk.outline) * float((lengths**2).sum())
    return max(1, round(area / (side**2 * coverage))), side


def _frame_to(n: np.ndarray) -> np.ndarray:
    """The rotation taking +Y to the unit vector n."""
    v = np.cross(_UP, n)
    s, c = float(np.linalg.norm(v)), float(_UP @ n)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - c) / s**2)


def _ao_dirs() -> np.ndarray:
    i = np.arange(AO_RAYS)
    rr, a = np.sqrt((i + 0.5) / AO_RAYS), i * GOLDEN
    return np.stack([rr * np.cos(a), np.sqrt(1 - rr * rr), rr * np.sin(a)], axis=1)


_AO_DIRS = _ao_dirs()


def _open_light(p: np.ndarray, normal: np.ndarray, cs: np.ndarray, rs: np.ndarray,
                reach: float) -> float:
    """The mean light left after the neighbour balls (`cs`, `rs`) along the cosine-weighted rays
    from p about `normal`: exp(-LEAF_DENSITY x leaf thickness, in radii) per ray."""
    if len(cs) == 0:
        return 1.0
    rays = _AO_DIRS @ _frame_to(normal).T  # (rays, 3)
    x = cs - p  # (m, 3)
    b = rays @ x.T  # (rays, m)
    disc = b * b - ((x * x).sum(axis=1) - rs * rs)
    h = np.sqrt(np.maximum(disc, 0.0))
    through_ball = np.maximum(np.minimum(b + h, reach) - np.maximum(b - h, 0.0), 0.0)
    through = (np.where(disc > 0, through_ball, 0.0) / rs).sum(axis=1)
    return float(np.exp(-LEAF_DENSITY * through).mean())


def card_shown(p: np.ndarray, normal: np.ndarray, cs: np.ndarray, rs: np.ndarray,
               reach: float) -> tuple[float, float]:
    """(shown, open light) of a card at p facing `normal`, among the neighbour balls (`cs`, `rs`).
    `burial` is how deep p lies in the most-covering neighbour, `dark` how little light reaches
    it; `shown = 1 - burial x dark`. A card under `SHOWN_CUT` is dropped, else scaled by shown."""
    if len(cs):
        depth = np.linalg.norm(cs - p, axis=1) / (BURIED * rs)
        burial = float((1 - smoothstep(depth, 1 - FADE_IN, 1)).max())
    else:
        burial = 0.0
    open_ = _open_light(p, normal, cs, rs, reach)
    dark = float(1 - smoothstep(open_, HIDDEN, HIDDEN * (1 + FADE_IN)))
    return 1 - burial * dark, open_


def crown_centre(centres: np.ndarray, lean: LeafLean) -> np.ndarray:
    """The middle of the box of `centres`, lowered `lean.centre_drop` of the box height."""
    lo, hi = centres.min(axis=0), centres.max(axis=0)
    c = (lo + hi) / 2.0
    c[1] -= lean.centre_drop * (hi[1] - lo[1])
    return c


def _spiral(i: int, n: int, turn: np.ndarray) -> np.ndarray:
    y = 1 - 2 * (i + 0.5) / n
    ring = math.sqrt(1 - y * y)
    a = i * GOLDEN
    return turn @ np.array([math.cos(a) * ring, y, math.sin(a) * ring])


def _stack(rows: list, shape: tuple[int, ...]) -> np.ndarray:
    return np.array(rows, dtype=float).reshape(shape) if rows else np.zeros(shape)


def card_leaves(sk: Skeleton, dials: CardDials, units_per_metre: float,
                lean: LeafLean) -> CardLeaves:
    """Two triangles per card, over every ball of `sk` (balls of radius `dials.ball_m`)."""
    seed = int(sk.info["seed"])
    cover = coverage(paint_cells(sk.outline, seed=seed))
    all_balls = balls(sk, dials.ball_m * units_per_metre)
    reach = AO_REACH_M * units_per_metre
    leaf_c = leaf_centres(sk)
    crown = crown_centre(leaf_c, lean)
    cs = np.array([b.centre for b in all_balls])
    rs = np.array([b.radius for b in all_balls])
    tris, nrms, uvs, cols, centres, faces = [], [], [], [], [], []
    dropped = 0
    for bi, ball in enumerate(all_balls):
        rng = np.random.default_rng([seed, bi])
        n_cards, side = ball_cards(ball, sk, cover)
        gap = np.linalg.norm(cs - ball.centre, axis=1) - rs - ball.radius
        near = np.flatnonzero((gap < reach) & (np.arange(len(all_balls)) != bi))
        lc = leaf_c[ball.leaves]
        turn = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        for i in range(n_cards):
            d = _spiral(i, n_cards, turn)
            p = ball.centre + d * ball.radius * (0.9 + rng.random() * SPREAD_JITTER)
            p = lc[int(np.argmin(np.linalg.norm(lc - p, axis=1)))]  # sit on a real leaf
            normal = _unit(dials.normal_mix * d + (1 - dials.normal_mix) * _unit(p - crown))
            shown, open_ = card_shown(p, normal, cs[near], rs[near], reach)
            tilt = _unit(rng.normal(size=3)) * CARD_TILT * rng.random()
            spin = rng.random() * 2 * math.pi
            size = side * (SIZE_DRAW[0] + rng.random() * (SIZE_DRAW[1] - SIZE_DRAW[0]))
            cell = int(rng.integers(4))
            if shown < SHOWN_CUT:
                dropped += 1
                continue
            face = _unit(d + tilt)
            face = _unit(face + dials.sky * _UP * np.sign(face[1] + 1e-9))
            across = _unit(np.cross([0, 1, 0] if abs(face[1]) < 0.99 else [1, 0, 0], face))
            along = np.cross(face, across)
            tan = across * math.cos(spin) + along * math.sin(spin)
            bit = np.cross(face, tan)
            dist = float(np.linalg.norm(p - ball.centre))
            corner = np.empty((4, 3))
            for k, (qx, qy) in enumerate(_QUAD):
                flat = p + (tan * qx + bit * qy) * size * shown
                wrapped = ball.centre + _unit(flat - ball.centre) * dist
                corner[k] = flat + (wrapped - flat) * ROUNDNESS
            cell_uv = (np.array([cell % 2, cell // 2]) + _QUV) * 0.5
            ao = 1 + (min(open_**2.7 * 7.6, 1.0) * AO_BOOST - 1) * AO_MIX
            for tri in _TRIANGLES:
                tris.append(corner[list(tri)])
                uvs.append(cell_uv[list(tri)])
                nrms.append(np.tile(normal, (3, 1)))
                cols.append(np.full((3, 3), ao))
            centres.append(p)
            faces.append(face)
    n = len(tris)
    return CardLeaves(tris=_stack(tris, (n, 3, 3)), normals=_stack(nrms, (n, 3, 3)),
                      uvs=_stack(uvs, (n, 3, 2)), colours=_stack(cols, (n, 3, 3)),
                      centres=_stack(centres, (n // 2, 3)), faces=_stack(faces, (n // 2, 3)),
                      cards=n // 2, dropped=dropped)


def card_wood(sk: Skeleton, sight: Sight) -> dict[str, np.ndarray]:
    """One part per level name, as `grown_parts`, for the card copy. A limb thinner than the eye
    floor (`sight.gap`) along its whole length is one straight 3-sided stick from its base to its
    tip when it holds a leaf, and is left out when it holds none. Every other limb is meshed by
    `limb_tris` at `sight`."""
    holders = {int(h) for h in sk.host}
    parts: dict[str, np.ndarray] = {}
    for li, name in enumerate(sk.level_names):
        tris = []
        for i in np.flatnonzero(sk.level == li):
            i = int(i)
            radii = sk.limb_radii(i)
            if 2 * float(radii.max()) < sight.gap:
                if i in holders:
                    pts = sk.limb_points(i)
                    tris.append(tube_tris(np.stack([pts[0], pts[-1]]),
                                          np.array([radii[0], radii[-1]]), 3, int(sk.tip[i])))
            else:
                tris.append(limb_tris(sk, i, sight))
        if tris:
            parts[name] = np.concatenate(tris)
    return parts


def card_parts(sk: Skeleton, sight: Sight, dials: CardDials, units_per_metre: float,
               lean: LeafLean) -> tuple[dict[str, np.ndarray], CardLeaves | None]:
    """The card copy's parts: `card_wood` and a `Cards_<shade>` part of leaf cards. A skeleton
    with no leaves gives wood only, `None` for the leaves and a printed line saying so. One card
    mesh serves one shade, so a skeleton with several raises."""
    parts = card_wood(sk, sight)
    piece = sk.info["piece"]
    if not len(sk.host):
        print(f"cards: {piece} has no leaves; wood only")
        return parts, None
    if len(sk.shades) != 1:
        raise ManifestError(f"cards: {piece} has shades {list(sk.shades)}; "
                            "one card mesh serves one shade")
    leaves = card_leaves(sk, dials, units_per_metre, lean)
    parts[card_part_name(sk.shades[0])] = leaves.tris
    return parts, leaves
