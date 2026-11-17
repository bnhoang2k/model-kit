"""Check lightmap-density: the baked light does not change when the lightmap density doubles.

The bake runs each layer twice on a grid of 2 x `size` pixels with two seeds (`blender.bake_render`).
Per placement it records two numbers in `bake.json`:

- `noise`: the RMS of (A - B) / 2 over the placement's seen texels, the noise of the mean of the
  two bakes, per doubled texel.
- `error`: the RMS of (the picture shrunk and grown back by 2 - the mean) over the same seen
  doubled texels AND over the 4 phases of the 2 x 2 block grid (`core.bake_grid.phase_error`).
  Each shrunk texel is the mean of the seen (covered) doubled texels of its 2 x 2 block: 4 of
  them inside an island, 1 to 3 at an island's edge. Phase (0, 0) is the shipped picture; the
  others start the blocks one doubled texel lower, to the right, or both.

Why the phase mean: a verdict must not depend on where the packer lands an island. A hard light
edge on a block border loses nothing to the shrink, and the same edge one doubled texel over is
averaged across a block. On bar 05's desk, moving the block grid by one doubled texel moved the
window layer's loss from 0.81 to 2.2 times its noise, and a repack for another placement flipped
the desk's verdict. The mean over the 4 phases is the loss at an average landing, and it does
not change when the island moves by a doubled texel. The noise part of the derivation below holds
at every phase, so the 3/4 share is unchanged.

Derivation. Let each doubled texel be `signal + noise`, the noise independent with RMS `n`. A
shipped texel holds the mean of its block's `k` covered texels, so each of them differs from it by
its own noise minus the mean of `k`: variance `n^2 (1 - 1/k)`. Inside an island `k = 4`, giving
`3/4 n^2`; at an island's edge `k` is 1 to 3, giving 0, 1/2 or 2/3 of `n^2`. The check takes the
inside value for every texel: for pure noise, `error = sqrt(3/4) x noise` at most. Any more error is
signal the block's average smoothed away, light the shipped density cannot hold:

    loss = sqrt(max(0, error^2 - 3/4 noise^2))

Taking 3/4 for the edge texels too subtracts more noise than they carry, so the check is a little
lenient on a placement with many edge texels (a thin island).

A placement is reported when `loss > noise` in ANY layer: the light lost at the shipped density is
larger than the noise the bake cannot resolve either. One item per placement,
`lightmap <placement> density`, with a detail line per failing layer naming the layer, `error`,
`noise` and `loss`.

A placement a layer records an `error` for but no `noise` is not judged in that layer. A scene with
no bake has no finding, no plant and no coverage line.

The plant is one unit, on a placement whose real data passes: in layer order, the first placement
(sorted) with a positive `noise` in that layer and no real finding in any layer. Its `error` is set
to 4 x `noise` in that layer, from the record alone; the real verdict only picks the placement. A
plant on a placement that already fails would be caught trivially and prove nothing, and the gate
skips such a unit, so an allowed finding on the first placement would leave the check unproven.
When every placement with a noise above 0 already fails, there is no plant and the coverage holds
one unmeasured line that says no clean placement is left, so the run reads `partial`. A bake that
records no noise above 0 has one not-applicable unit that says so (an informational coverage line,
not an unmeasured one).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection

from model_kit.core.bake_data import BakeData
from model_kit.core.checks.entries import unit
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import DENSITY_PURE_NOISE_SHARE

NAME = "lightmap-density"
PRINCIPLE = ("A bake whose light still changes when its lightmap density doubles has not "
             "converged, so the picture holds less light than the surface has.")
SOURCE = ("Light-bake spec, 'Pixels per piece' line (the bake converges when a doubled density "
          "changes the light by no more than the bake noise) and 'Checks' section; model-kit "
          "AGENTS.md, 'Adding a check'. The derivation of sqrt(3/4) x noise is in this module's "
          "docstring (the variance of a sum of independent terms is the sum of their variances: "
          "https://en.wikipedia.org/wiki/Variance).")
PLANT_FACTOR = 4.0  # the plant's error as a multiple of its noise: loss = sqrt(15.25) x noise


def _loss(error: float, noise: float) -> float:
    return math.sqrt(max(0.0, error * error - DENSITY_PURE_NOISE_SHARE * noise * noise))


def _unconverged(bake: BakeData) -> list[tuple[str, str, float, float, float]]:
    """(placement, layer, error, noise, loss) for each placement whose loss exceeds its noise in a
    layer, in placement then layer order."""
    out = []
    for layer, one in bake.layers.items():
        for placement, error in one.error.items():
            if placement not in one.noise:
                continue
            noise = one.noise[placement]
            loss = _loss(error, noise)
            if loss > noise:
                out.append((placement, layer, error, noise, loss))
    return sorted(out, key=lambda row: (row[0], list(bake.layers).index(row[1])))


def _item(placement: str) -> str:
    return f"lightmap {placement} density"


def _noisy(bake: BakeData) -> set[str]:
    """The placements with a noise above 0 in some layer: the ones a plant could go on."""
    return {p for one in bake.layers.values() for p, noise in one.noise.items() if noise > 0.0}


def _plant_site(bake: BakeData) -> tuple[str, str] | None:
    """(layer, placement) of the plant: in layer order, the first placement (sorted) with a noise
    above 0 in that layer and no real finding in any layer. None when no such placement is left."""
    failing = {row[0] for row in _unconverged(bake)}
    for layer, one in bake.layers.items():
        clean = sorted(p for p, noise in one.noise.items() if noise > 0.0 and p not in failing)
        if clean:
            return layer, clean[0]
    return None


def _planted(ctx: SceneCtx, layer: str, placement: str) -> SceneCtx:
    one = ctx.bake.layers[layer]
    error = {**one.error, placement: PLANT_FACTOR * one.noise[placement]}
    layers = {**ctx.bake.layers, layer: dataclasses.replace(one, error=error)}
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers=layers))


class LightmapDensity:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a scene check: it measures no piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        if ctx.bake is None:
            return []
        site = _plant_site(ctx.bake)
        if site is not None:
            layer, placement = site
            return [unit(_item(placement),
                         [(f"its {layer} error is {PLANT_FACTOR:g} x its noise",
                           _planted(ctx, layer, placement))])]
        if _noisy(ctx.bake):
            return []  # coverage() says no clean placement is left
        return [PlantUnit("lightmap density", (), not_applicable=(
            "no layer records a noise above 0 for any placement"))]

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        if ctx.bake is None:
            return []
        lines: dict[str, list[str]] = {}
        for placement, layer, error, noise, loss in _unconverged(ctx.bake):
            lines.setdefault(placement, []).append(
                f"layer {layer!r}: error {error:.4g}, noise {noise:.4g}, loss {loss:.4g}")
        return [Finding(NAME, _item(p), f"placement {p!r} still changes with density: "
                                        + "; ".join(rows))
                for p, rows in sorted(lines.items()) if only is None or _item(p) in only]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        if ctx.bake is None or not _noisy(ctx.bake) or _plant_site(ctx.bake) is not None:
            return []
        return [Unmeasured(f"{NAME}: no clean placement left to plant on (every placement with "
                           "a noise above 0 already fails, so the check's catch is not shown): "
                           + ", ".join(sorted(_noisy(ctx.bake))))]
