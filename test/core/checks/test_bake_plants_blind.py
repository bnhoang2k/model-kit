"""The guard for the bake checks' plants: a check whose core never reports a finding cannot see
its plant, and the gate stops the run with `BlindCheck`.

Each case blinds the function a check's verdict rests on, on a bake scene of its own. A plant is
built from the record or the UV alone, so a blind core leaves the plant standing and uncaught.
`SCENES` is also read by `test_every_check_is_seen_blind`.
"""

import dataclasses

import pytest

from model_kit.core.checks import bake_files, lightmap_density, lightmap_overlap, seam_jump
from model_kit.core.checks.bake_files import BakeFiles
from model_kit.core.checks.lightmap_density import LightmapDensity
from model_kit.core.checks.lightmap_overlap import LightmapOverlap
from model_kit.core.checks.seam_jump import SeamJump
from model_kit.core.errors import BlindCheck
from model_kit.core.metagate import run_gate

from .bake_scenes import shelf_of_planks, slab_and_boxes, smooth_shelf, smooth_slab


def with_noise(make):
    """The scene `make` builds, its layer given a noise per placement (so a plant can be made)."""
    def build():
        ctx = make()
        layers = {name: dataclasses.replace(
            one, noise={n.rsplit("/", 1)[0]: 0.02 for n in ctx.bake.uv},
            error={n.rsplit("/", 1)[0]: 0.01 for n in ctx.bake.uv})
            for name, one in ctx.bake.layers.items()}
        return dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers=layers))
    return build


def blind_missing(monkeypatch):
    monkeypatch.setattr(bake_files, "_missing", lambda bake: [])


def blind_extra(monkeypatch):
    monkeypatch.setattr(bake_files, "_extra", lambda bake: [])


def blind_overlap(monkeypatch):
    monkeypatch.setattr(lightmap_overlap, "_overlaps", lambda bake, only=None: {})


def blind_size(monkeypatch):
    monkeypatch.setattr(bake_files, "_resized", lambda bake: [])


def blind_noise(monkeypatch):
    monkeypatch.setattr(bake_files, "_unnoised", lambda bake: [])


def blind_jumps(monkeypatch):
    monkeypatch.setattr(seam_jump, "_jumps", lambda bake, only=None: [])


def blind_density(monkeypatch):
    monkeypatch.setattr(lightmap_density, "_unconverged", lambda bake: [])


# One entry per rule, with a scene it plants on.
SCENES = {
    "bake-files-missing": (BakeFiles, blind_missing, slab_and_boxes),
    "bake-files-extra": (BakeFiles, blind_extra, shelf_of_planks),
    "bake-files-size": (BakeFiles, blind_size, slab_and_boxes),
    "bake-files-size-planks": (BakeFiles, blind_size, shelf_of_planks),
    "bake-files-noise": (BakeFiles, blind_noise, slab_and_boxes),
    "bake-files-noise-planks": (BakeFiles, blind_noise, shelf_of_planks),
    "seam-jump": (SeamJump, blind_jumps, smooth_slab),
    "seam-jump-shelf": (SeamJump, blind_jumps, smooth_shelf),
    "lightmap-density": (LightmapDensity, blind_density, with_noise(slab_and_boxes)),
    "lightmap-density-planks": (LightmapDensity, blind_density, with_noise(shelf_of_planks)),
    "lightmap-overlap": (LightmapOverlap, blind_overlap, slab_and_boxes),
    "lightmap-overlap-planks": (LightmapOverlap, blind_overlap, shelf_of_planks),
}


@pytest.mark.parametrize("name", sorted(SCENES))
def test_a_blind_core_stops_the_run(name, monkeypatch):
    make_check, blind, make = SCENES[name]
    ctx = make()
    run_gate([make_check()], ctx, None)  # the clean scene, with its plants, does not stop
    blind(monkeypatch)
    with pytest.raises(BlindCheck, match=make_check.name):
        run_gate([make_check()], ctx, None)
