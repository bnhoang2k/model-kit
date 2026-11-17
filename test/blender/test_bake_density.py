"""A stage's `placement_texels_per_m` gives one still placement its own lightmap density.

The slab scene's `box_b` bakes at twice the stage density, so its lightmap holds 4 times the
pixels per square metre; `Box` and the slab keep the stage's.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from bake_fixtures import make_scene, write_stage
from test_bake_unwrap import channel_of, corner_uv, uv_areas, world_tris

from model_kit import bake

pytestmark = pytest.mark.blender


@pytest.fixture(scope="module")
def box_on_slab(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("slab"), "box_on_slab")


def px_per_m2(out: Path, js: dict, node: str) -> float:
    """Pixels at the doubled size per square metre of the node's seen faces."""
    glb = out / "box_on_slab.glb"
    uv = corner_uv(glb, node, channel_of(js, node))
    tris = world_tris(glb, node)
    lit = uv_areas(uv) > 0
    world = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
                           axis=1) / 2
    return uv_areas(uv)[lit].sum() * (2 * js["size"]) ** 2 / world[lit].sum()


def test_one_placement_bakes_at_its_own_density(box_on_slab):
    # texels_per_m 16 seen from 1 m: 32 px a metre at the doubled size; box_b's 32: 64.
    stage_file = write_stage(box_on_slab / "stage.json", "box_on_slab", live=["doll"],
                             texels_per_m=16)
    data = json.loads(stage_file.read_text())
    data["placement_texels_per_m"] = {"box_b": 32}
    stage_file.write_text(json.dumps(data))
    out = bake.run_bake(bake.plan_bake(box_on_slab / "scene", stage_file), replace=True,
                        timeout=300)
    js = json.loads((out / "bake.json").read_text())
    assert px_per_m2(out, js, "box_b/body") == pytest.approx(64 ** 2, rel=0.02)
    for node in ("slab/body", "Box/body"):
        assert px_per_m2(out, js, node) == pytest.approx(32 ** 2, rel=0.02)
