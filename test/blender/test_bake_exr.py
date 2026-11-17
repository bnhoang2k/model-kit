"""The gate's reader of a bake (`core.exr`, `core.bake_data`) on files `model-kit bake` really wrote.

One tiny bake of the `box_on_slab` scene. The pictures are compared bit for bit with the reader
the layer tests use (`bake_fixtures.read_exr`), the lightmap UV with the .glb as the unwrap tests
read it, and the positions with their world triangles.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from bake_fixtures import make_scene, read_exr, write_stage
from test_bake_unwrap import corner_uv, world_tris

from model_kit import bake
from model_kit.core import exr
from model_kit.core.bake_data import load_bake
from model_kit.core.scene import load_scene

pytestmark = pytest.mark.blender


@pytest.fixture(scope="module")
def baked(tmp_path_factory) -> tuple[Path, dict]:
    bar = make_scene(tmp_path_factory.mktemp("slab"), "box_on_slab")
    stage = write_stage(bar / "stage.json", "box_on_slab", live=["doll"])
    out = bake.run_bake(bake.plan_bake(bar / "scene", stage), replace=True, timeout=300)
    return bar / "scene", json.loads((out / "bake.json").read_text())


def test_the_picture_reads_bit_for_bit(baked):
    scene, info = baked
    path = scene / "bake" / info["layers"]["lamp"]["picture"]
    got = exr.read_exr(path)
    assert got.shape == (info["size"], info["size"], 3)
    assert np.array_equal(got, read_exr(path))


def test_load_bake_reads_what_the_bake_wrote(baked):
    scene, info = baked
    data = load_bake(scene)
    assert data.size == info["size"] and data.mode == "full"
    assert data.files == {"lamp.exr"} and data.unseen == frozenset(info["unseen"])
    assert np.array_equal(data.layers["lamp"].pixels, read_exr(scene / "bake" / "lamp.exr"))
    glb = scene / "bake" / f"{info['layout']}.glb"
    assert set(data.uv) == set(info["uv_channel"]) == set(data.positions)
    for node, channel in info["uv_channel"].items():
        assert np.array_equal(data.uv[node], corner_uv(glb, node, channel))
        assert np.allclose(data.positions[node], world_tris(glb, node), atol=1e-6)


def test_load_scene_carries_the_bake(baked):
    scene, info = baked
    assert load_scene(scene).bake.size == info["size"]
