"""Check bake-files: `bake.json` and the pictures on disk agree.

A bake (`core.bake_data`) is a record, `bake.json`, and the files it describes. Each layer in the
record names its picture. Four rules, one item each:

- **missing** `bake <layer> missing`. The record lists the layer's picture, and the folder does
  not hold it. A reader that opens the picture finds nothing.
- **extra** `bake <picture> extra`. The folder holds a picture that no layer lists: a left-over
  of an earlier bake, or a layer dropped from the record. A reader that trusts the folder reads
  light that no layer made.
- **size** `bake <layer> size`. The picture on disk is not `size` x `size` pixels, as the record
  says. Every lightmap UV is a fraction of the picture, so a picture of another size is read at
  the wrong texels. A layer whose picture is not on disk is `missing`, not `size`.

- **noise** `bake <layer> noise <placement> missing`. A placement a camera sees, and the bake does
  not leave live, has no `noise` entry in the layer. The seam and density checks read that entry;
  with none they cannot judge the placement, so the record is incomplete. A placement is seen when
  a part of it (not listed `unseen`) has a texel centre of the doubled grid (2 x `size`, the grid
  the bake measures noise on) inside a lightmap triangle; a smaller one gets no entry.

A scene with no bake has nothing to compare: no finding, no plant and no coverage line. The
plants are one unit per rule, built from the record alone: the first listed picture taken out of
`files`, a picture no layer lists put into `files`, and (when a layer's picture is loaded) the
first loaded picture cut one pixel short on each side, and the first layer's noise entry for the
first seen placement taken out.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection

from model_kit.core.bake_data import BakeData, texels
from model_kit.core.checks.entries import unit
from model_kit.core.metagate import Finding, PlantUnit
from model_kit.core.scene import SceneCtx

NAME = "bake-files"
PRINCIPLE = ("A record and the files it describes must agree, or a reader trusts a picture "
             "that is not the one the record names.")
SOURCE = ("Light-bake spec, 'Bake files' line and 'Checks' section (the behaviour 'the gate "
          "fails when bake.json and the pictures on disk disagree'); model-kit AGENTS.md, "
          "'Adding a check'. Pictures: "
          "https://openexr.com/en/latest/OpenEXRFileLayout.html")
STRAY = "unlisted.exr"  # the name the extra-picture plant uses, when no real file has it


def _missing(bake: BakeData) -> list[tuple[str, str]]:
    return [(layer, one.picture) for layer, one in bake.layers.items()
            if one.picture not in bake.files]


def _extra(bake: BakeData) -> list[str]:
    listed = {one.picture for one in bake.layers.values()}
    return sorted(bake.files - listed)


def _resized(bake: BakeData) -> list[tuple[str, tuple[int, int]]]:
    """The layers whose loaded picture is not `size` x `size`, with the picture's (H, W)."""
    return [(layer, (int(one.pixels.shape[0]), int(one.pixels.shape[1])))
            for layer, one in bake.layers.items()
            if one.pixels is not None and one.pixels.shape[:2] != (bake.size, bake.size)]


def _seen(bake: BakeData) -> list[str]:
    """The placements (sorted) the bake records a noise for: a part not `unseen` with a texel centre
    of the doubled grid (2 x `size`, where the bake measures noise) strictly inside a triangle of
    area. A seen triangle smaller than half a shipped texel can hold none, and then the bake writes
    no noise entry for it."""
    return sorted({part.rsplit("/", 1)[0] for part, uv in bake.uv.items()
                   if part not in bake.unseen and len(texels(uv, 2 * bake.size))})


def _unnoised(bake: BakeData) -> list[tuple[str, str]]:
    """(layer, placement) for each seen placement with no noise entry in a layer."""
    seen = _seen(bake)
    return [(layer, p) for layer, one in bake.layers.items() for p in seen if p not in one.noise]


def _cut(ctx: SceneCtx, layer: str) -> SceneCtx:
    one = ctx.bake.layers[layer]
    layers = {**ctx.bake.layers,
              layer: dataclasses.replace(one, pixels=one.pixels[:-1, :-1])}
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers=layers))


def _without(ctx: SceneCtx, picture: str) -> SceneCtx:
    bake = dataclasses.replace(ctx.bake, files=ctx.bake.files - {picture})
    return dataclasses.replace(ctx, bake=bake)


def _with(ctx: SceneCtx, picture: str) -> SceneCtx:
    bake = dataclasses.replace(ctx.bake, files=ctx.bake.files | {picture})
    return dataclasses.replace(ctx, bake=bake)


class BakeFiles:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a scene check: it measures no piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        bake = ctx.bake
        if bake is None:
            return []
        units = []
        listed = [(layer, one.picture) for layer, one in bake.layers.items()
                  if one.picture in bake.files]
        if listed:
            layer, picture = listed[0]
            units.append(unit(f"bake {layer} missing",
                              [("its picture is not on disk", _without(ctx, picture))]))
        stray = STRAY if STRAY not in bake.files else f"x-{STRAY}"
        units.append(unit(f"bake {stray} extra",
                          [("a picture no layer lists is on disk", _with(ctx, stray))]))
        loaded = [layer for layer, one in bake.layers.items()
                  if one.pixels is not None and one.pixels.shape[:2] == (bake.size, bake.size)]
        if loaded:
            units.append(unit(f"bake {loaded[0]} size",
                              [("its picture is a pixel short", _cut(ctx, loaded[0]))]))
        for layer, one in bake.layers.items():
            known = [p for p in _seen(bake) if p in one.noise]
            if known:
                noise = {k: v for k, v in one.noise.items() if k != known[0]}
                gone = dataclasses.replace(one, noise=noise)
                layers = {**bake.layers, layer: gone}
                units.append(unit(f"bake {layer} noise {known[0]} missing", [
                    ("its noise entry is not recorded",
                     dataclasses.replace(ctx, bake=dataclasses.replace(bake, layers=layers)))]))
                break
        if not any(u.item.endswith(" missing") and " noise " in u.item for u in units):
            units.append(PlantUnit("bake noise", (), not_applicable=(
                "no placement is seen on the doubled grid with a noise entry to take out")))
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        bake = ctx.bake
        if bake is None:
            return []
        out = [Finding(NAME, f"bake {layer} missing",
                       f"bake.json lists {picture!r} for layer {layer!r}; the folder holds "
                       f"no such file")
               for layer, picture in _missing(bake)]
        out += [Finding(NAME, f"bake {picture} extra",
                        f"the folder holds {picture!r}, which no layer in bake.json lists")
                for picture in _extra(bake)]
        out += [Finding(NAME, f"bake {layer} size",
                        f"the picture of layer {layer!r} is {h} x {w} px; bake.json says "
                        f"{bake.size} x {bake.size}")
                for layer, (h, w) in _resized(bake)]
        out += [Finding(NAME, f"bake {layer} noise {p} missing",
                        f"layer {layer!r} records no noise for placement {p!r}, which a camera "
                        f"sees; bake.json is incomplete")
                for layer, p in _unnoised(bake)]
        return [f for f in out if only is None or f.item in only]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return []
