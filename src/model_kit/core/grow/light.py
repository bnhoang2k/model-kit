"""The shadow grid: how much light each bud gets (Palubicki et al. 2009, sec 4.1).

Think of a tree as a stack of grids seen from above: every bud drops a cone of shade on the
cells below it, and a bud reads how much shade lies on its own cell. This module holds that grid.

Every bud adds a shadow `a * b**-q` to the cells `q` layers below it, over a block of
`(2q + 1) x (2q + 1)` cells centred under the bud, for `q` from 0 to a depth (the shadow
pyramid). A bud's light is `Q = max(C - s + n a, 0)`, where `s` is the shadow on its cell, `C` the
full exposure, and the `n a` takes back the bud's own shadow, and with it the shadow of the other
buds at its node (`n` buds sit at one point, so they do not shade each other; n is 1 for a lone
bud). Y is up (glTF), so shade falls
toward -Y. The grid never imports a species: depth, `a`, `b` and `C` are the kit's constants.
"""

from __future__ import annotations

import numpy as np

from model_kit.core.grow.shape import SEGMENT_EPS


class ShadowGrid:
    """A sparse-to-dense shadow field over the buds added to it.

    `add_shadow` records a batch of buds; the field is built the first time it is read after a
    change, over the box that holds every batch and its pyramid. A point outside the box is in
    full light. Every batch on one grid shares one `a`, because `light` gives each bud its own
    `a` back."""

    def __init__(self, cell: float) -> None:
        if not cell > 0.0:
            raise ValueError(f"a shadow cell must be over 0, got {cell!r}")
        self.cell = float(cell)
        self._batches: list[tuple[np.ndarray, int, float, float]] = []
        self._a = 0.0
        self._lo = np.zeros(3)
        self._field: np.ndarray | None = None

    def add_shadow(self, points: np.ndarray, depth: int, a: float, b: float) -> None:
        """Buds at `points` (N, 3) cast shade `a * b**-q` on the cells q layers below, q = 0 to
        `depth`. Needs `a` over 0, `b` over 1, `depth` 0 or more, and the `a` of earlier batches."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if depth < 0 or depth != int(depth):
            raise ValueError(f"shadow depth must be a whole number of layers, got {depth!r}")
        if not a > 0.0:
            raise ValueError(f"shadow a must be over 0, got {a!r}")
        if not b > 1.0:
            raise ValueError(f"shadow b must be over 1, got {b!r}")
        if self._batches and a != self._a:
            raise ValueError(f"shadow a is {self._a!r} on this grid, got {a!r}")
        self._a = float(a)
        if len(points):
            self._batches.append((points, int(depth), float(a), float(b)))
            self._field = None

    # --------------------------------------------------------------------------- the field

    def _build(self) -> np.ndarray:
        if self._field is not None:
            return self._field
        if not self._batches:
            self._lo = np.zeros(3)
            self._field = np.zeros((1, 1, 1))
            return self._field
        everything = np.concatenate([p for p, *_ in self._batches])
        reach = max(depth for _, depth, _, _ in self._batches) + 1
        self._lo = everything.min(axis=0) - self.cell * reach
        hi = everything.max(axis=0) + self.cell * reach
        shape = tuple(int(n) for n in np.ceil((hi - self._lo) / self.cell).astype(int) + 1)
        field = np.zeros(shape)
        nx, ny, nz = shape
        xs, zs = np.arange(nx), np.arange(nz)
        for points, depth, a, b in self._batches:
            idx = np.floor((points - self._lo) / self.cell).astype(int)
            count = np.zeros(shape)
            np.add.at(count, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
            # Summed-area table over x and z in every layer, so a (2q+1)^2 block is four reads.
            sat = np.zeros((nx + 1, ny, nz + 1))
            sat[1:, :, 1:] = count.cumsum(axis=0).cumsum(axis=2)
            for q in range(depth + 1):
                x0, x1 = np.clip(xs - q, 0, nx), np.clip(xs + q + 1, 0, nx)
                z0, z1 = np.clip(zs - q, 0, nz), np.clip(zs + q + 1, 0, nz)
                box = (sat[x1][:, :, z1] - sat[x0][:, :, z1]
                       - sat[x1][:, :, z0] + sat[x0][:, :, z0])
                if q < ny:  # shade falls q layers: a bud in layer y shades layer y - q
                    field[:, : ny - q, :] += a * b ** (-q) * box[:, q:, :]
        self._field = field
        return field

    def _cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N, 3) cell indexes of the points, and (N,) True where the point is inside the box."""
        field = self._build()
        idx = np.floor((np.asarray(points, dtype=float).reshape(-1, 3) - self._lo) / self.cell)
        idx = idx.astype(int)
        inside = np.all((idx >= 0) & (idx < np.array(field.shape)), axis=1)
        return idx, inside

    # ------------------------------------------------------------------------------ reads

    def shadow(self, points: np.ndarray) -> np.ndarray:
        """(N,) the shade on each point's cell; 0 outside the box."""
        field = self._build()
        idx, inside = self._cells(points)
        out = np.zeros(len(idx))
        if inside.any():
            ii = idx[inside]
            out[inside] = field[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out

    def light(self, points: np.ndarray, full: float, own: np.ndarray | int = 1) -> np.ndarray:
        """(N,) the light `max(full - s + own a, 0)` of buds at `points`. The `own a` gives a bud
        back its own shadow and that of the other buds at its node: `own` is how many buds of the
        grid sit at that point (one number, or one per point). It is for points that hold buds of
        the grid; a lone bud with no shade on it reads `full + a`."""
        return np.maximum(full - self.shadow(points) + self._a * np.asarray(own), 0.0)

    def toward_light(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) the unit direction of falling shade, reversed: where the shade is least. The
        shade's gradient by central differences at each point's cell; a zero vector where there
        is no gradient (or the point is outside the box)."""
        field = self._build()
        idx, inside = self._cells(points)
        last = np.array(field.shape) - 1
        out = np.zeros((len(idx), 3))
        if not inside.any():
            return out
        ii = idx[inside]
        grad = np.zeros((len(ii), 3))
        for axis in range(3):
            step = np.zeros(3, dtype=int)
            step[axis] = 1
            hi = np.minimum(ii + step, last)
            lo = np.maximum(ii - step, 0)
            grad[:, axis] = (field[hi[:, 0], hi[:, 1], hi[:, 2]]
                             - field[lo[:, 0], lo[:, 1], lo[:, 2]])
        length = np.sqrt((grad * grad).sum(axis=1, keepdims=True))
        out[inside] = np.where(length > SEGMENT_EPS, -grad / np.maximum(length, SEGMENT_EPS), 0.0)
        return out
