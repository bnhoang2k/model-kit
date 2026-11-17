"""The kit's light mixer (`js/bake-mixer.js`) in headless Chromium.

Each test loads the shipped page `js/compare.html` (it holds the import map for three.js), builds
a tiny bake in the page (two 4x4 layers of known values) and reads the light map back.
"""

import json
import struct

import numpy as np
import pytest

from model_kit import compare

pytestmark = pytest.mark.browser

# Shared by every test: three.js, the mixer, a renderer, a 4x4 two-layer bake and a mesh to put a
# light map on. Pictures are RGBA float32, top row first.
PRELUDE = """
const THREE = await import('three');
const m = await import('/js/bake-mixer.js');
const renderer = new THREE.WebGLRenderer({ canvas: document.createElement('canvas') });
const picture = (f) => {
  const data = new Float32Array(4 * 4 * 4);
  for (let i = 0; i < 16; i++) data.set([...f(i), 1], i * 4);
  return { width: 4, height: 4, data };
};
const pictures = {
  x: picture((i) => [i * 0.02, 0.1, 0.3]),
  y: picture((i) => [0.05, i * 0.01, 0.2]),
};
const makeInfo = (extra) => ({ layout: 't', mode: 'full', size: 4, uv_channel: { 'a/body': 1 },
  unseen: [], layers: { x: {}, y: {} }, ...extra });
const mesh = (name) => {
  const geometry = new THREE.PlaneGeometry(2, 2);
  geometry.setAttribute('uv1', geometry.attributes.uv);
  const made = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial({ color: 0xffffff }));
  made.name = name;
  return made;
};
const expected = (values) => {
  const out = new Float32Array(64);
  for (const [name, pic] of Object.entries(pictures)) {
    const { color = [1, 1, 1], strength = 1 } = values[name] || {};
    for (let i = 0; i < 16; i++) for (let c = 0; c < 3; c++) {
      out[i * 4 + c] += pic.data[i * 4 + c] * color[c] * strength * m.UNITS;
    }
  }
  return out;
};
const worst = (a, b) => {
  let w = 0;
  for (let i = 0; i < 16; i++) for (let c = 0; c < 3; c++) w = Math.max(w, Math.abs(a[i * 4 + c] - b[i * 4 + c]));
  return w;
};
const biggest = (a) => Math.max(...a.filter((_, i) => i % 4 < 3));
const setup = (info, nodes) => {
  const root = new THREE.Group();
  nodes.forEach((n) => root.add(n));
  const bake = m.makeBake(info, pictures, renderer);
  m.applyBake(root, bake);
  return { root, bake };
};
"""

VALUES = ("{ x: { color: [1, 0.5, 0.25], strength: 1.5 }, "
          "y: { color: [0.5, 1, 0.75], strength: 2 } }")


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return tmp_path_factory.mktemp("data")


@pytest.fixture(scope="module")
def page(data):
    routes = {"/js/": compare.JS_DIR, "/data/": data}
    with compare.serve(routes) as base, compare.browser_page() as page:
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(f"{base}/js/compare.html")
        page.wait_for_function("window.compareReady === true")
        page.run_js = lambda body: page.evaluate(f"async () => {{ {PRELUDE} {body} }}")
        yield page
        assert errors == []


def test_mixer_sums_layers(page):
    got = page.run_js(f"""
      const values = {VALUES};
      const {{ bake }} = setup(makeInfo(), [mesh('a/body')]);
      m.setLights(bake, values);
      const want = expected(values);
      return {{ error: worst(m.readTarget(bake, 1), want), size: biggest(want) }};""")
    assert got["size"] > 0.5  # a sum worth reading, not a row of zeros
    assert got["error"] < 1e-3


def test_mixer_scales_by_strength(page):
    got = page.run_js("""
      const { bake } = setup(makeInfo(), [mesh('a/body')]);
      const once = (strength) => {
        const values = { x: { color: [1, 1, 1], strength }, y: { color: [1, 1, 1], strength: 0 } };
        m.setLights(bake, values);
        return { error: worst(m.readTarget(bake, 1), expected(values)), top: biggest(m.readTarget(bake, 1)) };
      };
      return [once(0.5), once(1), once(3)];""")
    assert [round(g["top"] / got[1]["top"], 3) for g in got] == [0.5, 1.0, 3.0]
    assert max(g["error"] for g in got) < 1e-3


def test_mixer_zero_strength_gives_zero(page):
    got = page.run_js("""
      const { bake } = setup(makeInfo(), [mesh('a/body')]);
      const off = { x: { color: [1, 1, 1], strength: 0 }, y: { color: [1, 1, 1], strength: 0 } };
      m.setLights(bake, off);
      const dark = biggest(m.readTarget(bake, 1));
      const onlyY = { x: { color: [1, 1, 1], strength: 0 }, y: { color: [1, 1, 1], strength: 1 } };
      m.setLights(bake, onlyY);
      return { dark, error: worst(m.readTarget(bake, 1), expected(onlyY)) };""")
    assert got["dark"] == 0
    assert got["error"] < 1e-3


def test_mixer_uses_the_recorded_channel(page):
    got = page.run_js("""
      const info = makeInfo({ uv_channel: { 'a/body': 1, 'a/plain': 0 } });
      const one = mesh('a/body'), zero = mesh('a/plain');
      setup(info, [one, zero]);
      return [one.material.lightMap.channel, zero.material.lightMap.channel,
              one.material.lightMap === zero.material.lightMap];""")
    assert got == [1, 0, False]


def test_mixer_counts_the_baked_parts_it_found(page):
    """`applyBake` returns how many baked parts got a light map: a part the bake names and the
    scene lacks is not counted; a part with two meshes (a group) counts once."""
    got = page.run_js("""
      const info = makeInfo({ uv_channel: { 'a/body': 1, 'a/lid': 1, 'a/missing': 1 } });
      const group = new THREE.Group();
      group.name = 'a/lid';
      group.add(mesh('lid-top'), mesh('lid-side'));
      const root = new THREE.Group();
      root.add(mesh('a/body'), group, mesh('b/moves'));
      return m.applyBake(root, m.makeBake(info, pictures, renderer));""")
    assert got == 2


def test_mixer_skips_unseen(page):
    got = page.run_js("""
      const info = makeInfo({ unseen: ['a/hidden'] });
      const hidden = mesh('a/hidden'), live = mesh('b/moves');
      const held = live.material;
      const { bake } = setup(info, [mesh('a/body'), hidden, live]);
      m.setLights(bake, {});
      return [hidden.material.lightMap, live.material.lightMap, live.material === held];""")
    assert got == [None, None, True]


def test_mixer_full_mode_turns_live_light_off_on_still_meshes(page):
    got = page.run_js("""
      const shade = (mode, name) => {
        const info = makeInfo({ mode });
        const scene = new THREE.Scene();
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(0, 0, 5);
        scene.add(light, light.target);
        const { root, bake } = setup(info, [mesh(name)]);
        scene.add(root);
        m.setLights(bake, { x: { color: [1, 1, 1], strength: 1 }, y: { color: [1, 1, 1], strength: 0 } });
        const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);
        camera.position.set(0, 0, 4);
        const target = new THREE.WebGLRenderTarget(8, 8);
        const pixel = () => {
          renderer.setRenderTarget(target);
          renderer.render(scene, camera);
          const out = new Uint8Array(4 * 8 * 8);
          renderer.readRenderTargetPixels(target, 0, 0, 8, 8, out);
          renderer.setRenderTarget(null);
          return Array.from(out.slice((4 * 8 + 4) * 4, (4 * 8 + 4) * 4 + 3));
        };
        light.intensity = 1;
        const lit = pixel();
        light.intensity = 0;
        return { lit, unlit: pixel() };
      };
      return { still: shade('full', 'a/body'), live: shade('full', 'b/moves'),
               indirect: shade('indirect', 'a/body') };""")
    still, live, indirect = got["still"], got["live"], got["indirect"]
    assert still["lit"] == still["unlit"] and max(still["lit"]) > 0  # light map only
    assert live["lit"] != live["unlit"]
    assert indirect["lit"] != indirect["unlit"]


def write_exr(path, rows):
    """An uncompressed half-float EXR of `rows` ((h, w, 3) floats, top row first)."""
    height, width, _ = rows.shape

    def attr(name, kind, data):
        return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<i", len(data)) + data

    chans = b"".join(c.encode() + b"\0" + struct.pack("<i", 1) + b"\0\0\0\0"
                     + struct.pack("<ii", 1, 1) for c in "BGR") + b"\0"
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    head = (b"\x76\x2f\x31\x01" + struct.pack("<i", 2) + attr("channels", "chlist", chans)
            + attr("compression", "compression", b"\0") + attr("dataWindow", "box2i", box)
            + attr("displayWindow", "box2i", box) + attr("lineOrder", "lineOrder", b"\0")
            + attr("pixelAspectRatio", "float", struct.pack("<f", 1))
            + attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
            + attr("screenWindowWidth", "float", struct.pack("<f", 1)) + b"\0")
    size = 3 * width * 2
    at = len(head) + 8 * height
    body, table = b"", b""
    for y in range(height):
        table += struct.pack("<Q", at + len(body))
        body += struct.pack("<ii", y, size)
        for c in (2, 1, 0):  # B, G, R
            body += rows[y, :, c].astype("<f2").tobytes()
    path.write_bytes(head + table + body)


def test_mixer_reads_a_picture_top_row_first(page, data):
    """A real EXR file through `loadBake`: the picture's top row is row 0 of the light map."""
    rows = np.array([[[0.25, 0.5, 1], [0.5, 0.5, 1]], [[0.75, 0.5, 1], [1.0, 0.5, 1]]])
    write_exr(data / "x.exr", rows)
    (data / "bake.json").write_text(json.dumps({
        "layout": "t", "mode": "full", "size": 2, "uv_channel": {"a/body": 0}, "unseen": [],
        "layers": {"x": {"picture": "x.exr"}}}))
    got = page.evaluate("""async () => {
      const THREE = await import('three');
      const m = await import('/js/bake-mixer.js');
      const renderer = new THREE.WebGLRenderer({ canvas: document.createElement('canvas') });
      const bake = await m.loadBake('/data/bake.json', renderer);
      const geometry = new THREE.PlaneGeometry(1, 1);
      const root = new THREE.Group();
      const mesh = new THREE.Mesh(geometry, new THREE.MeshLambertMaterial());
      mesh.name = 'a/body';
      root.add(mesh);
      m.applyBake(root, bake);
      m.setLights(bake, { x: { color: [1, 1, 1], strength: 1 / m.UNITS } });
      return Array.from(m.readTarget(bake, 0));
    }""")
    mixed = np.array(got).reshape(2, 2, 4)[:, :, :3]
    assert np.allclose(mixed, rows, atol=1e-3)


def test_strength_from_a_three_js_intensity(page):
    """A layer's `strength` is Blender's: a three.js sun of 1 is strength 1, a three.js point or
    spot of 1 candela is 4 pi watts, a three.js hemisphere of 1 is a world of 1 / pi. An area
    light has no factor and is refused."""
    got = page.run_js("""
      const kinds = ['sun', 'point', 'spot', 'hemisphere'];
      const out = Object.fromEntries(kinds.map((k) => [k, m.strengthFromThree(k, 2)]));
      try { m.strengthFromThree('area', 1); out.area = null; } catch (e) { out.area = e.message; }
      return out;""")
    assert got["sun"] == pytest.approx(2.0)
    assert got["point"] == pytest.approx(8 * np.pi) and got["spot"] == pytest.approx(8 * np.pi)
    assert got["hemisphere"] == pytest.approx(2 / np.pi)
    assert '"area"' in got["area"]


def test_mixer_refuses_a_key_that_is_no_layer(page):
    got = page.run_js("""
      const { bake } = setup(makeInfo(), [mesh('a/body')]);
      try { m.setLights(bake, { z: { color: [1, 1, 1], strength: 1 } }); return null; }
      catch (e) { return e.message; }""")
    assert '"z"' in got and "x, y" in got
