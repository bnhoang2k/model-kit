"""Check lightmap-overlap: no two parts of a bake claim one texel of the picture.

Item: `lightmap <a> x <b> overlap` (parts sorted, named as the .glb names them). The scenes are
`bake_scenes`: islands `MARGIN` px apart on a `SIZE` px picture.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import lightmap_overlap
from model_kit.core.checks.lightmap_overlap import LightmapOverlap
from model_kit.core.metagate import run_gate

from ..scenes import make_ctx as reach_ctx
from . import assembly_scenes, manifest_scenes
from .bake_scenes import MOVES, SCENES, SIZE, island, moved_onto, slab_and_boxes
from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base, tried_and_caught

CHECK = LightmapOverlap()
PLANTED = {"slab_and_boxes": "lightmap Box/body x box_b/body overlap",
           "shelf_of_planks": "lightmap plank_a/body x plank_b/body overlap"}


def with_uv(ctx, uv):
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, uv=uv))


def test_registered_before_coverage_and_cites_its_principle():
    assert gate.CHECKS["lightmap-overlap"] is LightmapOverlap
    assert list(gate.CHECKS)[-1] == "coverage"
    assert CHECK.principle.endswith(".") and ". " not in CHECK.principle
    assert "http" in CHECK.source


@pytest.mark.parametrize("name", SCENES)
def test_touching_islands_quiet(name):
    """Islands one margin apart claim no texel twice."""
    ctx = SCENES[name]()
    assert CHECK.run(ctx) == []
    assert run_gate([LightmapOverlap()], ctx, None).status == "passed"


@pytest.mark.parametrize("name", SCENES)
def test_overlap_found(name):
    part, onto = MOVES[name]
    ctx = moved_onto(SCENES[name](), part, onto)
    found = CHECK.run(ctx)
    assert [f.item for f in found] == [f"lightmap {' x '.join(sorted((part, onto)))} overlap"]
    assert found[0].check == "lightmap-overlap" and "texels" in found[0].detail


def test_only_limits_the_pairs():
    part, onto = MOVES["slab_and_boxes"]
    ctx = moved_onto(slab_and_boxes(), part, onto)
    item = CHECK.run(ctx)[0].item
    assert [f.item for f in CHECK.run(ctx, only=[item])] == [item]
    assert CHECK.run(ctx, only=["lightmap Box/body x box_b/body overlap"]) == []


def test_two_parts_of_one_placement_are_judged_too():
    uv = {"Box/lid": island(2, 2, 10, 10), "Box/body": island(8, 8, 10, 10)}
    ctx = with_uv(slab_and_boxes(), uv)
    assert [f.item for f in CHECK.run(ctx)] == ["lightmap Box/body x Box/lid overlap"]


def test_a_collapsed_triangle_claims_no_texel():
    """The unseen face's point sits inside the slab's island and owns nothing."""
    ctx = slab_and_boxes()
    inside = np.full((1, 3, 2), 10.5 / SIZE)
    uv = {**ctx.bake.uv, "Box/body": np.concatenate([ctx.bake.uv["Box/body"], inside])}
    assert CHECK.run(with_uv(ctx, uv)) == []


def test_a_shared_edge_through_texel_centres_claims_none():
    """Two parts' triangles meet along y = 10.5 px, an edge that runs through a row of texel
    centres: a centre on the edge is strictly inside neither, so the parts do not collide."""
    y = 10.5 / SIZE
    above = np.array([[[0.1, y - 0.2], [0.4, y - 0.2], [0.4, y]], [[0.1, y - 0.2], [0.4, y],
                                                                    [0.1, y]]])
    below = np.array([[[0.1, y], [0.4, y], [0.4, y + 0.2]], [[0.1, y], [0.4, y + 0.2],
                                                              [0.1, y + 0.2]]])
    assert CHECK.run(with_uv(slab_and_boxes(), {"a/body": above, "b/body": below})) == []


def test_texels_are_claimed_by_their_centre():
    """A triangle too small to hold a texel centre claims nothing, so it does not collide; two
    triangles that both hold the centre do."""
    c = 10.5 / SIZE
    small = np.array([[[c - 0.002, c - 0.002], [c - 0.001, c - 0.002], [c - 0.002, c - 0.001]]])
    around = np.array([[[c - 0.01, c - 0.01], [c + 0.03, c - 0.01], [c - 0.01, c + 0.03]]])
    base = slab_and_boxes()
    assert CHECK.run(with_uv(base, {"a/body": small, "b/body": around})) == []
    assert [f.item for f in CHECK.run(with_uv(base, {"a/body": around, "b/body": around}))] == [
        "lightmap a/body x b/body overlap"]


@pytest.mark.parametrize("name", SCENES)
def test_the_plants_are_caught(name):
    # the first two placements, in name order
    assert tried_and_caught(LightmapOverlap(), SCENES[name]()) == [PLANTED[name]]
    assert run_gate([LightmapOverlap()], SCENES[name](), None).status == "passed"


def test_a_part_pair_in_one_placement_has_its_own_plant():
    ctx = slab_and_boxes()
    ctx = with_uv(ctx, {**ctx.bake.uv, "slab/rim": island(34, 40, 10, 10)})
    assert len(tried_and_caught(LightmapOverlap(), ctx)) == 2


def test_plants_are_built_from_the_uv_alone(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a plant called the check's own core")
    monkeypatch.setattr(lightmap_overlap, "_overlaps", refuse)
    monkeypatch.setattr(lightmap_overlap, "_texels", refuse)
    assert len(LightmapOverlap().plants(slab_and_boxes())) == 1


@pytest.mark.parametrize("make", [reach_ctx, assembly_scenes.make_ctx, manifest_scenes.make_ctx,
                                  lambda: piece_ctx(post_on_base())])
def test_no_stage_quiet(make):
    ctx = make()
    assert ctx.bake is None
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == [] and CHECK.coverage(ctx) == []
    verdict = run_gate([LightmapOverlap()], ctx, None)
    assert verdict.status == "passed" and verdict.coverage == []
