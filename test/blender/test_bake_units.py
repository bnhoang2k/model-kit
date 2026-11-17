"""`UNITS`, measured: a white plane under a Blender sun of strength 1, baked, then lit in three.js
two ways, and the two renders must agree.

The bake gives the plane's texel T (its slab top, full mode). In Chromium the same white plane is
rendered (a) lit by a `DirectionalLight` of intensity 1 at normal incidence, and (b) with no live
light and a `lightMap` that the shipped mixer draws from the real bake at the sun's strength 1.
Mixing the texel with `UNITS` must light the plane as the live light does, within 1%.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from bake_fixtures import make_scene, read_exr, write_stage
from test_bake_layers import SLAB_ONLY, SUN, face_texels

from model_kit import bake, compare

pytestmark = [pytest.mark.blender, pytest.mark.browser]

MEASURE = """async ({ row, col, channel }) => {
  const THREE = await import('three');
  const m = await import('/js/bake-mixer.js');
  const renderer = new THREE.WebGLRenderer({ canvas: document.createElement('canvas') });
  const bake = await m.loadBake('/bake/bake.json', renderer);
  const plane = (name) => {
    const geometry = new THREE.PlaneGeometry(2, 2);
    const uv = new Float32Array(8);
    for (let i = 0; i < 4; i++) {
      uv[2 * i] = (col + 0.5) / bake.size;  // a texel's centre: the sample is that texel
      uv[2 * i + 1] = (row + 0.5) / bake.size;
    }
    geometry.setAttribute(channel ? `uv${channel}` : 'uv', new THREE.BufferAttribute(uv, 2));
    const made = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: 0xffffff }));
    made.name = name;
    return made;
  };
  const camera = new THREE.PerspectiveCamera(20, 1, 0.1, 100);
  camera.position.set(0, 0, 4);
  const target = new THREE.WebGLRenderTarget(4, 4, { type: THREE.FloatType });
  const centre = (scene) => {
    renderer.setRenderTarget(target);
    renderer.render(scene, camera);
    const out = new Float32Array(4 * 4 * 4);
    renderer.readRenderTargetPixels(target, 0, 0, 4, 4, out);
    renderer.setRenderTarget(null);
    return out[(2 * 4 + 2) * 4];
  };
  const live = new THREE.Scene();
  const sun = new THREE.DirectionalLight(0xffffff, 1);
  sun.position.set(0, 0, 5);
  live.add(sun, sun.target, plane('live/plane'));
  const mapped = new THREE.Scene();
  const still = plane('slab/body');
  mapped.add(still);
  m.applyBake(mapped, bake);
  m.setLights(bake, { sun: { color: [1, 1, 1], strength: 1 } });
  return { live: centre(live), mapped: centre(mapped), units: m.UNITS };
}"""


@pytest.fixture(scope="module")
def baked(tmp_path_factory) -> tuple[Path, dict]:
    bar = make_scene(tmp_path_factory.mktemp("units"), "box_on_slab")
    stage = write_stage(bar / "stage.json", "box_on_slab", live=[*SLAB_ONLY, "doll"],
                        layers=["sun"], lights=SUN, mode="full", texels_per_m=32)
    out = bake.run_bake(bake.plan_bake(bar / "scene", stage), replace=True, timeout=300)
    return out, json.loads((out / "bake.json").read_text())


def test_units_white_plane(baked, capsys):
    out, info = baked
    rows, cols = face_texels(out, info, "slab/body", (0, 1, 0))
    texels = read_exr(out / info["layers"]["sun"]["picture"])[rows, cols]
    # the texel of the face nearest its middle: an edge texel is filled by the bake's margin
    pick = int(np.argmin((rows - np.median(rows)) ** 2 + (cols - np.median(cols)) ** 2))
    row, col = int(rows[pick]), int(cols[pick])
    with compare.serve({"/js/": compare.JS_DIR, "/bake/": out}) as base, \
            compare.browser_page() as page:
        page.goto(f"{base}/js/compare.html")
        page.wait_for_function("window.compareReady === true")
        got = page.evaluate(MEASURE, {"row": row, "col": col,
                                      "channel": info["uv_channel"]["slab/body"]})
    texel = float(texels[pick, 0])
    measured = got["units"] * got["live"] / got["mapped"]  # the UNITS that would match exactly
    with capsys.disabled():
        print(f"\nUNITS measured: texel {texel:.6f}, live pixel {got['live']:.6f}, "
              f"mapped pixel {got['mapped']:.6f} at UNITS {got['units']:.6f} "
              f"-> UNITS that matches {measured:.6f} (pi {np.pi:.6f})")
    assert got["live"] > 0.1 and texel > 0.1
    assert abs(got["mapped"] - got["live"]) <= 0.01 * got["live"]
