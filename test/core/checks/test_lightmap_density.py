"""Check lightmap-density: the baked light does not change when the lightmap density doubles.

Item: `lightmap <placement> density`. For pure noise the grown-back error is sqrt(3/4) x noise; the
light lost at half density is sqrt(error^2 - 3/4 noise^2), reported when it exceeds the noise.
"""

import dataclasses
import math

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.allow import Allowlist
from model_kit.core.bake_data import BakeLayer
from model_kit.core.checks import lightmap_density
from model_kit.core.checks.lightmap_density import LightmapDensity
from model_kit.core.metagate import Unmeasured, run_gate

from ..scenes import make_ctx as reach_ctx
from . import assembly_scenes, manifest_scenes
from .bake_scenes import SCENES, SIZE
from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base, tried_and_caught

CHECK = LightmapDensity()
NOISE = 0.02
FIRST = {"slab_and_boxes": "Box", "shelf_of_planks": "plank_a"}  # first placement, in name order
PLACEMENTS = {"slab_and_boxes": ["slab", "Box", "box_b"],
              "shelf_of_planks": ["plank_a", "plank_b", "plank_c"]}


def with_errors(ctx, lamp, sun):
    """Two layers, `lamp` and `sun`, each with noise NOISE per placement and the given errors
    (a factor of the noise, per placement)."""
    placements = sorted({n.rsplit("/", 1)[0] for n in ctx.bake.uv})
    pixels = np.zeros((SIZE, SIZE, 3), np.float32)

    def layer(name, factor):
        return BakeLayer(picture=f"{name}.exr", pixels=pixels,
                         noise={p: NOISE for p in placements},
                         error={p: NOISE * factor.get(p, 0.5) for p in placements})
    layers = {"lamp": layer("lamp", lamp), "sun": layer("sun", sun)}
    bake = dataclasses.replace(ctx.bake, layers=layers,
                               files=frozenset({"lamp.exr", "sun.exr"}))
    return dataclasses.replace(ctx, bake=bake)


def test_registered_before_coverage_and_cites_its_principle():
    assert gate.CHECKS["lightmap-density"] is LightmapDensity
    assert list(gate.CHECKS)[-1] == "coverage"
    assert CHECK.principle.endswith(".") and ". " not in CHECK.principle
    assert "Pixels per piece" in CHECK.source


@pytest.mark.parametrize("name", SCENES)
def test_converged_quiet(name):
    ctx = with_errors(SCENES[name](), {}, {})  # every error is 0.5 x noise
    assert CHECK.run(ctx) == []
    assert run_gate([LightmapDensity()], ctx, None).status == "passed"


@pytest.mark.parametrize("name", SCENES)
def test_unconverged_found(name):
    """Only the second layer, `sun`, has an unconverged placement."""
    placement = PLACEMENTS[name][1]
    ctx = with_errors(SCENES[name](), {}, {placement: 4.0})
    found = CHECK.run(ctx)
    assert [f.item for f in found] == [f"lightmap {placement} density"]
    detail = found[0].detail
    assert "sun" in detail and "0.08" in detail and "0.02" in detail and placement in detail
    assert "loss" in detail
    assert found[0].check == "lightmap-density"


def test_pure_noise_is_quiet():
    """For pure noise the grown-back error is sqrt(3/4) x noise, and no light is lost."""
    ctx = with_errors(SCENES["slab_and_boxes"](), {}, {"slab": math.sqrt(0.75)})
    assert CHECK.run(ctx) == []


def test_the_line_is_loss_equal_noise():
    """loss = sqrt(error^2 - 3/4 noise^2) = noise at error = sqrt(7/4) x noise: just under is
    quiet, just over is reported."""
    edge = math.sqrt(1.75)
    ctx = SCENES["slab_and_boxes"]()
    assert CHECK.run(with_errors(ctx, {}, {"slab": edge * 0.99})) == []
    assert len(CHECK.run(with_errors(ctx, {}, {"slab": edge * 1.01}))) == 1


def test_a_placement_is_reported_once_for_all_layers():
    ctx = with_errors(SCENES["slab_and_boxes"](), {"slab": 4.0}, {"slab": 4.0})
    found = CHECK.run(ctx)
    assert [f.item for f in found] == ["lightmap slab density"]
    assert "lamp" in found[0].detail and "sun" in found[0].detail


def test_only_limits_the_items():
    ctx = with_errors(SCENES["slab_and_boxes"](), {}, {"slab": 4.0})
    assert len(CHECK.run(ctx, only=["lightmap slab density"])) == 1
    assert CHECK.run(ctx, only=["lightmap Box density"]) == []


@pytest.mark.parametrize("name", SCENES)
def test_the_plant_is_caught(name):
    ctx = with_errors(SCENES[name](), {}, {})
    assert tried_and_caught(LightmapDensity(), ctx) == [f"lightmap {FIRST[name]} density"]
    assert run_gate([LightmapDensity()], ctx, None).status == "passed"


def test_all_zero_noise_has_one_not_applicable_unit():
    ctx = with_errors(SCENES["slab_and_boxes"](), {}, {})
    layers = {n: dataclasses.replace(one, noise={p: 0.0 for p in one.noise}, error={})
              for n, one in ctx.bake.layers.items()}
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers=layers))
    units = CHECK.plants(ctx)
    assert [u.item for u in units] == ["lightmap density"] and units[0].not_applicable
    verdict = run_gate([LightmapDensity()], ctx, None)
    assert verdict.status == "passed" and any("not applicable" in c for c in verdict.coverage)


def test_the_planted_fault_is_built_from_the_record_alone(monkeypatch):
    """The real verdict only picks where to plant; the planted error is 4 x the recorded noise,
    whatever the check's own loss would say."""
    def refuse(*a, **k):
        raise AssertionError("a plant called the check's own loss")
    ctx = with_errors(SCENES["slab_and_boxes"](), {}, {})
    monkeypatch.setattr(lightmap_density, "_unconverged", lambda bake: [])
    monkeypatch.setattr(lightmap_density, "_loss", refuse)
    (one,) = LightmapDensity().plants(ctx)
    (plant,) = one.candidates
    assert plant.ctx.bake.layers["lamp"].error["Box"] == lightmap_density.PLANT_FACTOR * NOISE


def test_the_plant_skips_a_failing_placement(tmp_path):
    """A plant on an item that already fails is caught trivially and proves nothing, so the gate
    skips it. The plant goes on the first placement whose real data passes: `Box` fails in the
    lamp layer, so `box_b` is planted. With `Box` allowed, the run passes on the strength of a
    plant that ran."""
    ctx = with_errors(SCENES["slab_and_boxes"](), {"Box": 4.0}, {})
    assert [u.item for u in CHECK.plants(ctx)] == ["lightmap box_b density"]
    assert tried_and_caught(LightmapDensity(), ctx) == ["lightmap box_b density"]
    allow = tmp_path / "box.allow"
    allow.write_text("lightmap-density | lightmap Box density | a reason\n")
    verdict = run_gate([LightmapDensity()], ctx, Allowlist.load(allow))
    assert verdict.status == "passed" and len(verdict.allowed) == 1
    assert not any(isinstance(line, Unmeasured) for line in verdict.coverage)


def test_every_placement_failing_leaves_the_plant_unmeasured():
    """No placement whose real data passes is left to plant on: the check's catch is not shown on
    this scene, so the run reads partial, not passed or failed alone."""
    ctx = with_errors(SCENES["slab_and_boxes"](),
                      {p: 4.0 for p in PLACEMENTS["slab_and_boxes"]}, {})
    assert CHECK.plants(ctx) == []
    (line,) = CHECK.coverage(ctx)
    assert isinstance(line, Unmeasured) and "no clean placement" in line
    assert run_gate([LightmapDensity()], ctx, None).status == "partial"


@pytest.mark.parametrize("make", [reach_ctx, assembly_scenes.make_ctx, manifest_scenes.make_ctx,
                                  lambda: piece_ctx(post_on_base())])
def test_no_stage_quiet(make):
    ctx = make()
    assert ctx.bake is None
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == [] and CHECK.coverage(ctx) == []
    verdict = run_gate([LightmapDensity()], ctx, None)
    assert verdict.status == "passed" and verdict.coverage == []
