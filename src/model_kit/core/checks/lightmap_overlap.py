"""Check lightmap-overlap: no two parts of a bake claim one texel of the picture.

A baked texel holds the light that reaches one surface point. A texel that two parts' lightmap
islands both cover holds one value for two points, so one of them reads the other's light.

Each seen part of the bake (`BakeData.uv`) has a lightmap UV, `(u, v)` in 0..1 with v from the top,
as the picture's rows run. Every triangle is rasterised into the picture at the bake's own size
(`BakeData.size`), into one owner image that holds the first part to claim each texel. A texel
belongs to a triangle when its **centre** lies strictly inside it, so two islands a margin apart
never collide, and a triangle that covers no texel centre claims none. A triangle with zero area
claims nothing: a face no camera sees has its UV collapsed to one point, and the point owns no
texel. Two triangles of one part never conflict; only two different parts do.

One item per pair of parts, parts sorted: `lightmap <a> x <b> overlap`, naming the parts as the
.glb does (`<placement>/<part>`), with the count of shared texels.

A scene with no bake has no UV to judge: no finding, no plant, no coverage line. The plants move
one part's island onto another's, from the UV alone: one unit for the first two placements that
each have a seen part, and one for the first placement that has two seen parts. Candidate one
shifts the first part so the centroid of its largest triangle lands on the centroid of the other's
largest. Candidate two puts the other's largest triangle in place of the first's largest.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Collection

import numpy as np

from model_kit.core.bake_data import BakeData, area2, texels
from model_kit.core.checks.entries import unit
from model_kit.core.metagate import Finding, PlantUnit
from model_kit.core.scene import SceneCtx

_texels = texels  # the rasteriser lives in `core.bake_data`; the name stays for the tests
NAME = "lightmap-overlap"
PRINCIPLE = ("A texel holds the light of one surface point, so two islands that cover one "
             "texel make one point read the other's light.")
SOURCE = ("Light-bake spec, 'Checks' section (the behaviour 'the gate fails a bake whose "
          "lightmap UV islands overlap'); model-kit AGENTS.md, 'Adding a check'. Texel centres: "
          "https://registry.khronos.org/vulkan/specs/1.3/html/chap27.html#primsrast-polygons. "
          "Engine rule: Unity manual, 'Generating lightmap UVs' (lightmap UV islands must not "
          "overlap), and Unreal Engine documentation, 'Lightmap UVs' (no overlapping UV charts "
          "in a lightmap UV channel).")
_ITEM = re.compile(r"lightmap (.+) x (.+) overlap")


def _wanted(only: Collection[str] | None) -> set[str] | None:
    """The parts the items of `only` name; None for every part."""
    if only is None:
        return None
    names: set[str] = set()
    for item in only:
        found = _ITEM.fullmatch(item)
        if found:
            names.update(found.groups())
    return names


def _overlaps(bake: BakeData, only: Collection[str] | None = None) -> dict[tuple[str, str], int]:
    """The texels shared by each pair of parts (parts sorted), over the parts `only` names."""
    wanted = _wanted(only)
    names = sorted(n for n in bake.uv if wanted is None or n in wanted)
    owner = np.full(bake.size * bake.size, -1, dtype=np.int32)
    shared: dict[tuple[str, str], int] = {}
    for index, name in enumerate(names):
        ids = _texels(bake.uv[name], bake.size)
        prior = owner[ids]
        for other in np.unique(prior[(prior >= 0) & (prior != index)]):
            pair = (names[int(other)], name)
            shared[pair] = shared.get(pair, 0) + int((prior == other).sum())
        owner[ids[prior < 0]] = index
    return shared


def _largest(uv: np.ndarray) -> int | None:
    """The index of the triangle of greatest area, or None when every triangle is collapsed."""
    area = np.abs(area2(uv))
    return int(area.argmax()) if len(area) and area.max() > 0.0 else None


def _placement(node: str) -> str:
    return node.rsplit("/", 1)[0]


def _item(a: str, b: str) -> str:
    a, b = sorted((a, b))
    return f"lightmap {a} x {b} overlap"


def _moved(ctx: SceneCtx, name: str, uv: np.ndarray) -> SceneCtx:
    bake = dataclasses.replace(ctx.bake, uv={**ctx.bake.uv, name: uv})
    return dataclasses.replace(ctx, bake=bake)


def _unit(ctx: SceneCtx, a: str, b: str) -> PlantUnit:
    """Move part `a`'s island onto part `b`'s."""
    ua, ub = ctx.bake.uv[a], ctx.bake.uv[b]
    ia, ib = _largest(ua), _largest(ub)
    shifted = ua + (ub[ib].mean(axis=0) - ua[ia].mean(axis=0))
    pasted = ua.copy()
    pasted[ia] = ub[ib]
    return unit(_item(a, b), [("its island is moved onto the other's", _moved(ctx, a, shifted)),
                              ("its largest triangle is the other's", _moved(ctx, a, pasted))])


class LightmapOverlap:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a scene check: it measures no piece's recipe
    principle = PRINCIPLE
    source = SOURCE

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        if ctx.bake is None:
            return []
        drawn = [n for n in sorted(ctx.bake.uv) if _largest(ctx.bake.uv[n]) is not None]
        by_placement: dict[str, list[str]] = {}
        for node in drawn:
            by_placement.setdefault(_placement(node), []).append(node)
        units = []
        firsts = [nodes[0] for nodes in by_placement.values()]
        if len(firsts) >= 2:
            units.append(_unit(ctx, firsts[0], firsts[1]))
        for nodes in by_placement.values():
            if len(nodes) >= 2:
                units.append(_unit(ctx, nodes[0], nodes[1]))
                break
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        if ctx.bake is None:
            return []
        out = []
        for (a, b), count in sorted(_overlaps(ctx.bake, only).items()):
            item = _item(a, b)
            if only is None or item in only:
                out.append(Finding(NAME, item, f"{count} texels of the {ctx.bake.size} px "
                                               f"picture are covered by both parts"))
        return out

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return []
