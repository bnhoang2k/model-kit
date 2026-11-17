"""The leaf image for a game-weight card copy of a grown tree.

Four cells in a 2 x 2 image, each a cluster of the grower's own blade outline drawn white to
light grey on clear, with a midrib. Later steps paint this image on leaf cards. Core stays
numpy-only: the polygons are filled here by a point-in-polygon test, with no anti-aliasing.

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

import numpy as np

_LEN_MIN, _LEN_SPAN = 0.20, 0.12  # leaf length, in cell sides: 0.20 to 0.32
PAINTED_SHARE = _LEN_MIN + _LEN_SPAN / 2  # mean painted leaf length over the cell side: 0.26
_RIB_HALF_WIDTH = 1.0  # pixels


def _inside(poly: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test for points (px, py)."""
    inside = np.zeros(px.shape, dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2):
        if ay == by:
            continue
        crosses = (ay > py) != (by > py)
        xcut = ax + (py - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (px < xcut)
    return inside


def _bleed(layer: np.ndarray) -> None:
    """Give every clear texel the colour of its nearest opaque texel in the same cell.

    A viewer box-filters the image into mips. Under clear texels the colour would be black and
    would darken the leaves of a far mip; a leaf colour keeps them the colour of the leaves.
    Alpha is not touched. A cell with no opaque texel stays as it is.
    """
    rgb, opaque = layer[..., :3], layer[..., 3] > 0
    if not opaque.any():
        return
    cell = opaque.shape[0]
    known = opaque.copy()
    while not known.all():
        grown_rgb, grown = rgb.copy(), known.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                src = (slice(max(-dy, 0), cell - max(dy, 0)), slice(max(-dx, 0), cell - max(dx, 0)))
                dst = (slice(max(dy, 0), cell - max(-dy, 0)), slice(max(dx, 0), cell - max(-dx, 0)))
                take = known[src] & ~grown[dst]
                grown_rgb[dst][take] = rgb[src][take]
                grown[dst] |= take
        rgb[...] = grown_rgb
        known = grown


def _paint_cell(outline: np.ndarray, cell: int, leaves: int, rng: np.random.Generator):
    layer = np.zeros((cell, cell, 4), dtype=np.uint8)
    for _ in range(leaves):
        length = cell * (_LEN_MIN + rng.random() * _LEN_SPAN)
        d = np.sqrt(rng.random()) * (cell * 0.46 - length * 0.45)  # out near the cell edge
        t = rng.random() * 2 * np.pi
        cx, cy = cell / 2 + np.cos(t) * d, cell / 2 + np.sin(t) * d
        ang = rng.random() * 2 * np.pi
        ca, sa = np.cos(ang), np.sin(ang)
        grey = int(200 + rng.random() * 55)
        o = (outline - [0, 0.5]) * length
        poly = np.stack([cx + o[:, 0] * ca - o[:, 1] * sa, cy + o[:, 0] * sa + o[:, 1] * ca], 1)
        x0, y0 = np.floor(poly.min(0)).astype(int) - 2
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, cell), min(y1, cell)
        if x0 >= x1 or y0 >= y1:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        px, py = gx + 0.5, gy + 0.5
        window = layer[y0:y1, x0:x1]
        window[_inside(poly, px, py)] = (grey, grey, grey, 255)
        # midrib, a little darker, from the leaf's root to its tip
        tip = np.array([cx - 0.5 * length * sa, cy + 0.5 * length * ca])
        root = np.array([cx + 0.5 * length * sa, cy - 0.5 * length * ca])
        seg = tip - root
        s = np.clip(((px - root[0]) * seg[0] + (py - root[1]) * seg[1]) / (seg @ seg), 0, 1)
        dist = np.hypot(px - (root[0] + s * seg[0]), py - (root[1] + s * seg[1]))
        rib = int(grey * 0.8)
        window[dist <= _RIB_HALF_WIDTH] = (rib, rib, rib, 255)
    # the border stays clear so a card never bleeds into its neighbour cell
    layer[0, :], layer[-1, :], layer[:, 0], layer[:, -1] = 0, 0, 0, 0
    _bleed(layer)
    return layer


def paint_cells(outline: np.ndarray, *, cell: int = 256, leaves: int = 16, seed: int = 0):
    """A (2*cell, 2*cell, 4) uint8 RGBA image: four cells of `leaves` copies of `outline`.

    Alpha is 0 or 255. Under each clear texel the RGB is the colour of the nearest opaque texel
    in the same cell, so a box-filtered mip keeps the leaf colour.
    """
    outline = np.asarray(outline, dtype=float)
    outline = outline - outline.mean(0) * [1, 0]
    img = np.zeros((2 * cell, 2 * cell, 4), dtype=np.uint8)
    for k in range(4):
        rng = np.random.default_rng([seed, k])
        r, c = (k // 2) * cell, (k % 2) * cell
        img[r : r + cell, c : c + cell] = _paint_cell(outline, cell, leaves, rng)
    return img


def coverage(image: np.ndarray) -> float:
    """The share of pixels with alpha above 127."""
    return float((np.asarray(image)[..., 3] > 127).mean())
