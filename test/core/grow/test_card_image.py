"""The leaf image: painted in numpy from the grower's outline, deterministic."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model_kit.core.grow.card_image import coverage, paint_cells
from model_kit.core.grow.skeleton import load_skeleton

CELL = 128
SKELETON = Path("bars/02-redbud/scene/pieces/redbud/Redbud.skeleton.npz")


def _outline() -> np.ndarray:
    return load_skeleton(SKELETON).outline


def _cells(img: np.ndarray):
    return [img[r : r + CELL, c : c + CELL] for r in (0, CELL) for c in (0, CELL)]


def test_same_seed_same_bytes():
    a = paint_cells(_outline(), cell=CELL, seed=3)
    b = paint_cells(_outline(), cell=CELL, seed=3)
    assert a.shape == (2 * CELL, 2 * CELL, 4) and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_cells_differ():
    cells = _cells(paint_cells(_outline(), cell=CELL))
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(cells[i], cells[j])


def test_coverage_in_range():
    # a sanity band for the painter, not a gate limit
    assert 0.30 <= coverage(paint_cells(_outline(), cell=256)) <= 0.50


def test_clear_border():
    for cell in _cells(paint_cells(_outline(), cell=CELL)):
        alpha = cell[..., 3]
        assert not alpha[0].any() and not alpha[-1].any()
        assert not alpha[:, 0].any() and not alpha[:, -1].any()


# sha256 of the redbud bar's alpha plane (default cell, the bar's seed), taken before the colour
# bleed was added. The bleed must not touch alpha, so coverage, cards and gates stay as they were.
REDBUD_ALPHA_SHA256 = "7614c67083949b8a0aa26b5d4d7aa3ec1cb2eb4e7be85ae5486878a435a4e0e6"


def _redbud_image() -> np.ndarray:
    sk = load_skeleton(SKELETON)
    return paint_cells(sk.outline, seed=int(sk.info["seed"]))


def test_alpha_unchanged_by_the_bleed():
    import hashlib

    alpha = np.ascontiguousarray(_redbud_image()[..., 3])
    assert set(np.unique(alpha)) == {0, 255}
    assert hashlib.sha256(alpha.tobytes()).hexdigest() == REDBUD_ALPHA_SHA256


def test_clear_texels_carry_the_colour_of_an_opaque_texel_in_their_cell():
    img = _redbud_image()
    cell = img.shape[0] // 2
    for r in (0, cell):
        for c in (0, cell):
            block = img[r : r + cell, c : c + cell]
            leaf_colours = {tuple(p) for p in block[block[..., 3] > 0][:, :3]}
            clear = block[block[..., 3] == 0][:, :3]
            assert len(clear) and not (clear.sum(1) == 0).any()  # none left black
            assert {tuple(p) for p in clear} <= leaf_colours


def test_far_mips_keep_the_leaf_colour():
    """The mean colour of texels that pass alpha 0.5 holds down a 2 x 2 box-filtered mip chain.

    The bound is one 8-bit step, 1/255, fixed before measuring: the file stores each colour as an
    8-bit integer, so a shift smaller than that cannot show. The drift it bounds is not rounding:
    the bleed copies nearby leaf texels, whose colours differ a little from the leaf mean, and the
    set of passing texels changes down the chain. Measured on the redbud: about 0.0013.
    """
    level = _redbud_image().astype(float) / 255
    step = 1 / 255
    mean0 = level[level[..., 3] >= 0.5][:, :3].mean()
    depth = 0
    while True:
        passing = level[..., 3] >= 0.5
        if not passing.any():
            break
        assert abs(level[passing][:, :3].mean() - mean0) <= step, f"mip {depth}"
        depth += 1
        h = level.shape[0] // 2
        level = level.reshape(h, 2, h, 2, 4).mean((1, 3))
    assert depth >= 6  # the chain really went down to where the old image went dark
