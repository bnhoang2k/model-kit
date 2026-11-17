"""The light bake's grid arithmetic, in plain numpy: the shipped picture's 2 x 2 shrink and the
resolution error the density check reads. `blender/bake_render.py` calls these; tests call them
with no Blender.

The bake renders each layer at twice the shipped size (the doubled grid). A shipped texel is the
mean of the covered doubled texels of its 2 x 2 block (`shrink_covered`). Where the block borders
fall on a surface depends on where the packer put its island: moving the island by one doubled
texel moves every border across it. A hard light edge that lies on a border loses nothing to the
shrink; the same edge one doubled texel over is averaged across a block. On bar 05's desk this
alone moved the window layer's loss from 0.81 to 2.2 times its noise
(measured by moving the 2 x 2 block grid by one doubled pixel). So `phase_error` measures the
error with the block grid at each of its 4 phases and takes their RMS: a verdict must not depend
on where the packer lands an island.
"""

from __future__ import annotations

import numpy as np


def shrink_covered(pic: np.ndarray, covered: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A picture (n2, n2, 3) at half the side, and which of its texels are filled (n, n). Each
    texel is the mean of the texels of its 2x2 block that `covered` (n2, n2) marks; a block with
    none stays empty (0). A plain 2x2 mean would average an island's edge with the black outside
    it, and a bilinear read there would show a dark rim at every seam."""
    n = pic.shape[0] // 2
    weight = covered.reshape(n, 2, n, 2).sum(axis=(1, 3)).astype(np.float64)
    total = (pic * covered[:, :, None]).reshape(n, 2, n, 2, pic.shape[2]).sum(axis=(1, 3))
    filled = weight > 0
    out = np.zeros((n, n, pic.shape[2]), dtype=np.float64)
    out[filled] = total[filled] / weight[filled, None]
    return out, filled


def phase_error(mean: np.ndarray, covered: np.ndarray,
                masks: dict[str, np.ndarray]) -> dict[str, float]:
    """For each mask (n2, n2) of `masks`, the RMS over the 4 grid phases and the mask's texels of
    (the shrunk picture grown back by 2 - `mean`), `mean` (n2, n2, 3) being the doubled bake.

    Phase (dy, dx) starts the 2 x 2 blocks at doubled row dy and column dx. Each phase shrinks
    the picture as the shipped one is shrunk (`shrink_covered` over `covered`, then a float16
    cast) and grows it back by 2. Phase (0, 0) is the shipped picture itself.

    The shifted grid wraps: its last block row pairs the picture's last doubled row with its
    first, and likewise for columns. No covered texel is ever in that block, because the packer
    keeps every island its margin away from the atlas edge, so the wrap adds nothing to any mask.
    A `covered` that reaches the picture's edge raises `ValueError` rather than wrap an island
    onto itself."""
    if covered[0].any() or covered[-1].any() or covered[:, 0].any() or covered[:, -1].any():
        raise ValueError("a covered texel lies on the picture's edge, so a shifted block grid "
                         "would wrap it onto the other side")
    squares = {pid: 0.0 for pid in masks}
    for dy in (0, 1):
        for dx in (0, 1):
            pic = np.roll(mean, (-dy, -dx), axis=(0, 1))
            small, _ = shrink_covered(pic, np.roll(covered, (-dy, -dx), axis=(0, 1)))
            shipped = small.astype(np.float16).astype(np.float32)
            back = shipped.repeat(2, axis=0).repeat(2, axis=1)
            diff = np.roll(back.astype(np.float64) - pic, (dy, dx), axis=(0, 1))
            for pid, mask in masks.items():
                if mask.any():
                    squares[pid] += float(np.mean(diff[mask] ** 2))
    return {pid: float(np.sqrt(total / 4.0)) for pid, total in squares.items()
            if masks[pid].any()}
