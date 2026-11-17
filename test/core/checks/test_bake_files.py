"""Check bake-files: `bake.json` and the pictures on disk agree.

Items: `bake <layer> missing` (a listed picture is not on disk), `bake <picture> extra` (a
picture on disk no layer lists) and `bake <layer> size` (a loaded picture is not size x size) and
`bake <layer> noise <placement> missing` (a seen placement has no noise entry). The scenes are `bake_scenes`; a scene with no bake is quiet.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.bake_data import BakeLayer
from model_kit.core.checks import bake_files
from model_kit.core.checks.bake_files import BakeFiles
from model_kit.core.metagate import run_gate

from ..scenes import make_ctx as reach_ctx
from . import assembly_scenes, manifest_scenes
from .bake_scenes import SCENES, shelf_of_planks, slab_and_boxes
from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base, tried_and_caught

CHECK = BakeFiles()


def with_files(ctx, files):
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, files=frozenset(files)))


def test_registered_and_cites_its_principle():
    assert gate.CHECKS["bake-files"] is BakeFiles
    assert list(gate.CHECKS)[-1] == "coverage"
    assert CHECK.principle.endswith(".") and ". " not in CHECK.principle
    assert "http" in CHECK.source


@pytest.mark.parametrize("name", SCENES)
def test_a_bake_that_agrees_is_quiet(name):
    ctx = SCENES[name]()
    assert CHECK.run(ctx) == [] and CHECK.coverage(ctx) == []
    assert run_gate([BakeFiles()], ctx, None).status == "passed"


@pytest.mark.parametrize("name", SCENES)
def test_missing_picture_found(name):
    ctx = with_files(SCENES[name](), [])
    found = CHECK.run(ctx)
    assert [f.item for f in found] == ["bake lamp missing"]
    assert "lamp.exr" in found[0].detail


@pytest.mark.parametrize("name", SCENES)
def test_extra_picture_found(name):
    ctx = with_files(SCENES[name](), ["lamp.exr", "old.exr"])
    assert [f.item for f in CHECK.run(ctx)] == ["bake old.exr extra"]


def test_only_limits_the_items():
    ctx = with_files(slab_and_boxes(), ["old.exr"])
    assert [f.item for f in CHECK.run(ctx)] == ["bake lamp missing", "bake old.exr extra"]
    assert [f.item for f in CHECK.run(ctx, only=["bake old.exr extra"])] == ["bake old.exr extra"]


def test_two_layers_each_need_a_picture():
    ctx = slab_and_boxes()
    layers = {**ctx.bake.layers, "sun": BakeLayer(picture="sun.exr", noise=ctx.bake.layers["lamp"].noise)}
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers=layers))
    assert [f.item for f in CHECK.run(ctx)] == ["bake sun missing"]


def with_pixels(ctx, pixels):
    one = dataclasses.replace(ctx.bake.layers["lamp"], pixels=pixels)
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers={"lamp": one}))


@pytest.mark.parametrize("name", SCENES)
def test_a_picture_of_another_size_found(name):
    ctx = SCENES[name]()
    size = ctx.bake.size
    found = CHECK.run(with_pixels(ctx, np.zeros((size, size - 1, 3), np.float32)))
    assert [f.item for f in found] == ["bake lamp size"]
    assert f"{size} x {size - 1}" in found[0].detail and f"{size} x {size}" in found[0].detail
    assert CHECK.run(with_pixels(ctx, np.zeros((size + 8, size, 3), np.float32)))[0].item == (
        "bake lamp size")


def test_a_picture_not_loaded_is_missing_not_resized():
    assert CHECK.run(with_pixels(slab_and_boxes(), None)) == []


def without_noise(ctx, placement):
    one = ctx.bake.layers["lamp"]
    one = dataclasses.replace(one, noise={k: v for k, v in one.noise.items() if k != placement})
    return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers={"lamp": one}))


@pytest.mark.parametrize("name", SCENES)
def test_a_seen_placement_with_no_noise_found(name):
    ctx = SCENES[name]()
    placement = min({n.rsplit("/", 1)[0] for n in ctx.bake.uv})
    found = CHECK.run(without_noise(ctx, placement))
    assert [f.item for f in found] == [f"bake lamp noise {placement} missing"]


def test_an_unseen_part_needs_no_noise():
    ctx = slab_and_boxes()
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(
        ctx.bake, unseen=frozenset({"slab/body"})))
    assert CHECK.run(without_noise(ctx, "slab")) == []


def test_a_placement_of_collapsed_triangles_needs_no_noise():
    ctx = slab_and_boxes()
    uv = {**ctx.bake.uv, "slab/body": ctx.bake.uv["slab/body"][:, :1].repeat(3, axis=1)}
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, uv=uv))
    assert CHECK.run(without_noise(ctx, "slab")) == []


def test_a_seen_triangle_with_no_doubled_texel_centre_needs_no_noise():
    """The bake writes a noise entry only for a placement whose mask holds a texel centre of the
    doubled grid. A triangle 0.2 px across holds none, so no entry is not a fault."""
    ctx = slab_and_boxes()
    size = ctx.bake.size
    c = np.array([10.1, 10.1])  # centres of the doubled grid sit at 0.25 + 0.5 k px: none inside
    tiny = np.array([[c, c + [0.2, 0.0], c + [0.0, 0.2]]]) / size
    uv = {**ctx.bake.uv, "slab/body": tiny}
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, uv=uv))
    assert CHECK.run(without_noise(ctx, "slab")) == []


def test_no_seen_placement_has_a_not_applicable_noise_unit():
    ctx = slab_and_boxes()
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, uv={}))
    units = [u for u in CHECK.plants(ctx) if u.not_applicable]
    assert [u.item for u in units] == ["bake noise"]
    verdict = run_gate([BakeFiles()], ctx, None)
    assert any("not applicable" in c for c in verdict.coverage)


@pytest.mark.parametrize("name", SCENES)
def test_all_four_plants_are_caught(name):
    ctx = SCENES[name]()
    first = min({n.rsplit("/", 1)[0] for n in ctx.bake.uv})
    assert tried_and_caught(BakeFiles(), ctx) == [
        "bake lamp missing", "bake unlisted.exr extra", "bake lamp size",
        f"bake lamp noise {first} missing"]


def test_plants_are_built_from_the_record_alone(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a plant called the check's own core")
    monkeypatch.setattr(bake_files, "_missing", refuse)
    monkeypatch.setattr(bake_files, "_extra", refuse)
    monkeypatch.setattr(bake_files, "_resized", refuse)
    monkeypatch.setattr(bake_files, "_unnoised", refuse)
    assert len(BakeFiles().plants(slab_and_boxes())) == 4


@pytest.mark.parametrize("make", [reach_ctx, assembly_scenes.make_ctx, manifest_scenes.make_ctx,
                                  lambda: piece_ctx(post_on_base())])
def test_no_stage_quiet(make):
    ctx = make()
    assert ctx.bake is None
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == [] and CHECK.coverage(ctx) == []
    verdict = run_gate([BakeFiles()], ctx, None)
    assert verdict.status == "passed" and verdict.coverage == []


def test_a_pass_needs_both_plants_caught():
    assert run_gate([BakeFiles()], shelf_of_planks(), None).status == "passed"
