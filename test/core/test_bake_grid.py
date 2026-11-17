"""`core.bake_grid.phase_error`: the density error does not follow where the packer lands an island.

Pure numpy, so a plain `uv run pytest` runs it without Blender."""

import numpy as np
import pytest

from model_kit.core.bake_grid import phase_error, shrink_covered


def _edge_on_a_border(shift: int) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """A doubled picture (64 x 64) of one island 4 doubled texels in from the edge (the packer's
    margin), lit 0.2 left of a hard edge and 0 right of it, two bakes with independent noise, all
    moved `shift` doubled texels to the right. At shift 0 the edge falls between doubled columns 29
    and 30, a border of the shipped 2 x 2 blocks; at shift 1 it cuts across blocks. Returns the
    mean of the two bakes, the covered mask, the masks by placement and the bake noise."""
    n2, sigma = 64, 0.004
    cols = np.arange(n2)[None, :, None]
    light = np.where(cols < 30, 0.2, 0.0) + 0.001 * np.arange(n2)[:, None, None]
    light = np.broadcast_to(light, (n2, n2, 3))
    rng = np.random.default_rng(3)
    a = light + rng.normal(0.0, sigma, light.shape)
    b = light + rng.normal(0.0, sigma, light.shape)
    covered = np.zeros((n2, n2), dtype=bool)
    covered[4:n2 - 4, 4:n2 - 5] = True
    a, b, covered = (np.roll(x, shift, axis=1) for x in (a, b, covered))
    noise = float(np.sqrt(np.mean(((a - b) / 2)[covered] ** 2)))
    return ((a + b) / 2).astype(np.float32), covered, {"p": covered}, noise


def _shipped_phase_error(mean, covered, mask) -> float:
    """The error as recorded before the phase mean: phase (0, 0), the shipped picture only."""
    small, _ = shrink_covered(mean, covered)
    back = small.astype(np.float16).astype(np.float32).repeat(2, axis=0).repeat(2, axis=1)
    return float(np.sqrt(np.mean((back.astype(np.float64) - mean)[mask] ** 2)))


def test_error_does_not_depend_on_where_the_island_lands():
    """The desk-flip measurement: a hard light edge on a block border loses nothing to the shrink,
    and one doubled texel over it loses a lot. The recorded `error` must not follow the packer.

    Tolerance, derived: moving the island by one doubled texel maps the 4 grid phases onto the
    same 4 phases (phase dx becomes 1 - dx), so every block averages the same texels and the
    squared differences are the same numbers, summed in another order. Float64 summation of N
    terms is exact to N x 2^-53 of the sum; the RMS moves by at most half that share. N is
    4 phases x the mask's texels x 3 channels. The bake noise is the scale the verdict reads, and
    the tolerance is far under it."""
    still = _edge_on_a_border(0)
    moved = _edge_on_a_border(1)
    mask = still[2]["p"]
    shipped = [_shipped_phase_error(m, c, k["p"]) for m, c, k, _ in (still, moved)]
    noise = still[3]
    assert shipped[1] - shipped[0] > noise  # the fixture's edge does sit on a border
    errors = [phase_error(m, c, k)["p"] for m, c, k, _ in (still, moved)]
    terms = 4 * int(mask.sum()) * 3
    tolerance = errors[0] * terms * 2.0 ** -53
    assert abs(errors[1] - errors[0]) <= tolerance, (errors, shipped, noise)
    assert tolerance < noise


def test_phase_error_refuses_a_covered_texel_on_the_edge():
    mean, covered, masks, _ = _edge_on_a_border(0)
    covered = covered.copy()
    covered[0, 10] = True
    with pytest.raises(ValueError, match="edge"):
        phase_error(mean, covered, masks)
