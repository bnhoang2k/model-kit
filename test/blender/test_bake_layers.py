"""The bake's layer step, run for real in Blender on the `box_on_slab` scene.

Pictures are tiny (16 samples, 16 texels per metre at a 1 m viewing distance). Where a test
needs a slab with nothing else on it, the boxes and the rigged doll are made live, so the bake
leaves them out and the slab is the only still piece.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from bake_fixtures import make_scene, read_exr, write_stage
from test_bake_unwrap import corner_uv, normals, pixels, uv_areas, world_tris

from model_kit import bake
from model_kit.blender import runner
from model_kit.core.bake_data import texels as texels_of

pytestmark = pytest.mark.blender

SLAB_ONLY = ["Box", "box_b"]
SUN = {"sun": {"kind": "sun", "position": [0, 5, 0], "direction": [0, -1, 0], "layer": "sun"}}
# two layers, one point light each, over the slab and the boxes
A = {"kind": "point", "position": [-0.6, 2.0, 0.3], "layer": "a"}
B = {"kind": "point", "position": [0.6, 2.0, -0.3], "layer": "b"}
# (colour, strength) each light is scaled by when layers a and b are summed
SLAB_PAINT = (0.8, 0.45, 0.2)  # the fixture's one material, `Fill`
VALUES = {"a": ((1.0, 0.5, 0.25), 2.0), "b": ((0.2, 0.4, 1.0), 3.0)}


@pytest.fixture(scope="module")
def box_on_slab(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("slab"), "box_on_slab")


def run(bar: Path, **stage) -> tuple[Path, dict]:
    """Bake the slab scene (over any earlier bake) and return the bake folder and bake.json.
    The rigged `doll` is always live."""
    stage["live"] = [*stage.get("live", ()), "doll"]
    stage_file = write_stage(bar / "stage.json", "box_on_slab", **stage)
    plan = bake.plan_bake(bar / "scene", stage_file)
    out = bake.run_bake(plan, replace=True, timeout=300)
    return out, json.loads((out / "bake.json").read_text())


def picture(out: Path, info: dict, layer: str) -> np.ndarray:
    return read_exr(out / info["layers"][layer]["picture"])


def texels(out: Path, info: dict, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """The (rows, columns) of a placement's seen texels in a picture read by `read_exr` (top row
    first): the pixels its seen lightmap triangles cover, from the exported .glb."""
    size, glb = info["size"], out / f"{info['layout']}.glb"
    rows, cols = [], []
    for node in info["uv_channel"]:
        if node.split("/")[0] != pid:
            continue
        uv = corner_uv(glb, node, info["uv_channel"][node])
        for x, y in pixels(uv[uv_areas(uv) > 0], size):
            rows.append(y)  # the .glb stores v from the top, as the picture's rows run
            cols.append(x)
    return np.array(rows), np.array(cols)


def worst_noise(info: dict, layer: str) -> float:
    return max(info["layers"][layer]["noise"].values())


def test_files_written(box_on_slab, capsys):
    out, info = run(box_on_slab)
    printed = capsys.readouterr().out
    assert (out / "lamp.exr").is_file()
    assert sorted(info["layers"]) == ["lamp"]
    layer = info["layers"]["lamp"]
    assert layer["picture"] == "lamp.exr"
    assert layer["lights"] == ["bulb"]
    assert (info["format"], info["units"], info["samples"], info["seed"]) == (
        "exr-half-zip", "irradiance/pi", 16, 1)
    assert layer["seed"] == 1 and info["blender"]
    pic = picture(out, info, "lamp")
    assert pic.shape == (info["size"], info["size"], 3)
    assert np.isfinite(pic).all() and pic.min() >= 0 and pic.max() > 0
    # one noise and one error value per still placement, in the picture's units
    assert sorted(layer["noise"]) == sorted(layer["error"]) == ["Box", "box_b", "slab"]
    assert all(math.isfinite(v) and v >= 0 for v in layer["noise"].values())
    assert all(math.isfinite(v) and v >= 0 for v in layer["error"].values())
    assert f"PICTURE lamp.exr {(out / 'lamp.exr').stat().st_size} bytes" in printed


def test_indirect_has_no_direct(box_on_slab):
    """A slab under one sun, with nothing else to bounce off: its indirect layer holds no light
    (within noise), its full layer does."""
    kw = {"live": SLAB_ONLY, "layers": ["sun"], "lights": SUN}
    out, info = run(box_on_slab, mode="indirect", **kw)
    indirect = picture(out, info, "sun")
    assert indirect.max() <= 3 * worst_noise(info, "sun")
    out, info = run(box_on_slab, mode="full", **kw)
    full = picture(out, info, "sun")
    assert full.max() > 10 * max(worst_noise(info, "sun"), 1e-9)


def test_layers_add_up(box_on_slab, tmp_path):
    """Layer a and layer b, each baked at white and strength 1, scaled by their colour and
    strength and summed, match one bake of both lights at those values, within 3 x noise."""
    out, info = run(box_on_slab, layers=["a", "b"], lights={"a": A, "b": B})
    # the reference: both lights on one layer, at the values the layers are scaled by
    stage_file = write_stage(box_on_slab / "ref.json", "box_on_slab", live=["doll"],
                             layers=["ab"], lights={"a": {**A, "layer": "ab"},
                                                    "b": {**B, "layer": "ab"}})
    spec = bake.plan_bake(box_on_slab / "scene", stage_file).spec()
    spec["out"] = str(tmp_path / "ref")
    spec["light_values"] = {k: {"color": list(c), "strength": s} for k, (c, s) in VALUES.items()}
    result = runner.run_blender(bake.SCRIPT, [json.dumps(spec)], timeout=300)
    assert result.ok, result.log[-2000:]
    ref_info = json.loads((tmp_path / "ref" / "bake.json").read_text())
    ref = read_exr(tmp_path / "ref" / "ab.exr")
    assert ref_info["size"] == info["size"]

    # Per placement, on the shipped grid. `noise` is on the doubled grid; a shipped texel is the
    # mean of 2 x 2 of them, so its noise is half. The layers' and the reference's noise are
    # independent, so they add in quadrature; a layer's noise scales with its colour x strength.
    scales = {n: np.array(c) * s for n, (c, s) in VALUES.items()}
    total = sum(picture(out, info, n) * k for n, k in scales.items())
    checked = 0
    for pid in ref_info["layers"]["ab"]["noise"]:
        rows, cols = texels(out, info, pid)
        bound = 3 * np.sqrt(
            sum((k * info["layers"][n]["noise"][pid] / 2) ** 2 for n, k in scales.items())
            + (ref_info["layers"]["ab"]["noise"][pid] / 2) ** 2)
        diff = np.sqrt(np.mean((total - ref)[rows, cols] ** 2, axis=0))
        assert ref[rows, cols].mean() > 0
        assert (diff <= bound).all(), (pid, diff, bound)
        checked += 1
    assert checked == 3


def test_same_seed_same_pictures(box_on_slab):
    out, info = run(box_on_slab)
    first = picture(out, info, "lamp")
    out, info = run(box_on_slab)
    assert first.max() > 0
    assert np.array_equal(first, picture(out, info, "lamp"))


def test_white_unit_lights(box_on_slab):
    """A stage light carries no colour or strength, so every Blender light the script made is
    white and strength 1: one entry per light in `baked_lights`, read back from Blender when it
    baked, with the light's kind (a page needs it to convert a three.js intensity)."""
    lights = {"a": A, "b": B, "c": {"kind": "area", "position": [0, 2, 0], "layer": "b",
                                    "direction": [0, -1, 0], "size": [0.5, 0.5]}}
    _, info = run(box_on_slab, layers=["a", "b"], lights=lights)
    used = {k: v for layer in info["layers"].values() for k, v in layer["baked_lights"].items()}
    assert sorted(used) == ["a", "b", "c"]
    assert {k: v.pop("kind") for k, v in used.items()} == {
        "a": A["kind"], "b": B["kind"], "c": "area"}
    assert all(v == {"color": [1.0, 1.0, 1.0], "strength": 1.0} for v in used.values())


def test_hemisphere_lights_their_own_half(box_on_slab):
    """The slab's top faces up: the sky layer lights it, the ground layer does not. The ground
    layer may hold a stray path at the horizon: one texel, far under the sky's light. Its noise is
    no yardstick for it (one spike over a thousand texels has an RMS of the spike / 32), so the
    bound is also one half-float step at the sky's level (2^-10 of it): a light under that cannot
    show in the picture beside the sky."""
    lights = {"up": {"kind": "hemisphere", "position": [0, 0, 0], "sky_ground": "sky",
                     "layer": "sky"},
              "down": {"kind": "hemisphere", "position": [0, 0, 0], "sky_ground": "ground",
                       "layer": "ground"}}
    out, info = run(box_on_slab, live=SLAB_ONLY, layers=["sky", "ground"], lights=lights)
    sky, ground = picture(out, info, "sky"), picture(out, info, "ground")
    assert sky.max() > 10 * max(worst_noise(info, "sky"), 1e-9)
    assert ground.max() <= max(3 * worst_noise(info, "ground"), 2.0 ** -10 * sky.max())


def test_hemisphere_recorded(box_on_slab):
    """A hemisphere light's entry is its world, at full value in its own half."""
    lights = {"up": {"kind": "hemisphere", "position": [0, 0, 0], "sky_ground": "sky",
                     "layer": "sky"}}
    _, info = run(box_on_slab, live=SLAB_ONLY, layers=["sky"], lights=lights)
    assert info["layers"]["sky"]["baked_lights"] == {
        "up": {"kind": "hemisphere", "color": [1.0, 1.0, 1.0], "strength": 1.0, "half": "sky"}}


def test_spot_cone_is_the_half_angle(box_on_slab):
    """A spot straight down at 1.8 m over the slab's top (cone_deg 20, no soft edge) lights a
    disc of radius 1.8 x tan(20 deg). The radius is read from the disc's texel count: 16 texels
    a metre on the shipped picture (16 texels_per_m at a 1 m viewing distance). The rim is
    ragged by about a texel each side, so the radius may be off by two texels."""
    spot = {"s": {"kind": "spot", "position": [0, 2.0, 0], "direction": [0, -1, 0],
                  "cone_deg": 20, "blend": 0.0, "layer": "s"}}
    out, info = run(box_on_slab, live=SLAB_ONLY, mode="full", layers=["s"], lights=spot)
    pic = picture(out, info, "s")[:, :, 0]
    lit = pic > 0.5 * pic.max()
    texels_per_m = 16
    radius = math.sqrt(lit.sum() / math.pi) / texels_per_m
    want = 1.8 * math.tan(math.radians(20))
    assert abs(radius - want) <= 2 / texels_per_m, (radius, want)


def face_texels(out: Path, info: dict, node: str, facing: tuple[float, float, float]):
    """(rows, columns) of the seen texels of the faces of `node` whose normal is `facing`."""
    glb, size = out / f"{info['layout']}.glb", info["size"]
    uv = corner_uv(glb, node, info["uv_channel"][node])
    pick = (normals(world_tris(glb, node)) @ np.array(facing) > 0.9) & (uv_areas(uv) > 0)
    cells = pixels(uv[pick], size)
    assert cells, f"{node} has no seen face towards {facing}"
    return (np.array([y for _, y in cells]), np.array([x for x, _ in cells]))


def test_indirect_layer_holds_bounced_light(box_on_slab):
    """Under a sun straight down, the +x side of `Box` (it stands left of the camera, so the
    camera sees that side) gets no direct light, only bounce from the slab. The indirect layer
    holds that bounce well above its noise, and the full layer holds the same (no direct light
    falls on that face). The face's value is the mean of its texels; its noise is one shipped
    texel's (half the recorded, doubled-grid noise) over the root of the texel count.

    Above-noise alone is empty when a layer holds no light and so has no noise, so the bounce is
    also pinned by its colour: light bounced once off the orange slab (0.8, 0.45, 0.2) carries
    that tint, so the face's colour is a multiple of it, within noise. Sun light is white."""
    kw = {"live": ["box_b"], "layers": ["sun"], "lights": SUN}
    out, info = run(box_on_slab, mode="indirect", **kw)
    rows, cols = face_texels(out, info, "Box/body", (1, 0, 0))
    root = math.sqrt(len(rows))
    colour = picture(out, info, "sun")[rows, cols].mean(axis=0)
    noise_i = info["layers"]["sun"]["noise"]["Box"] / 2 / root
    assert colour.mean() > 10 * noise_i, (colour, noise_i)
    paint = np.array(SLAB_PAINT)
    fit = paint * (colour @ paint) / (paint @ paint)  # the multiple of the paint nearest it
    assert np.abs(colour - fit).max() <= 3 * noise_i, (colour, fit, noise_i)
    out, info = run(box_on_slab, mode="full", **kw)
    full = picture(out, info, "sun")[rows, cols].mean(axis=0)
    noise_f = info["layers"]["sun"]["noise"]["Box"] / 2 / root
    assert np.abs(full - colour).max() <= 3 * math.hypot(noise_i, noise_f), (full, colour)


def test_colour_pass_is_off(box_on_slab):
    """The fixture's paint is orange (0.8, 0.45, 0.2). A sun straight down on the slab's top,
    in full mode, bakes the sun's own light, 1 / pi at strength 1: the paint does not scale it.
    Tolerance: one half-float step at that value, plus 3 x noise."""
    out, info = run(box_on_slab, live=SLAB_ONLY, mode="full", layers=["sun"], lights=SUN)
    pic = picture(out, info, "sun")
    top = pic[pic.max(axis=2) > 0.5 * pic.max()]
    want = 1 / math.pi
    slack = float(np.spacing(np.float16(want))) + 3 * info["layers"]["sun"]["noise"]["slab"]
    assert np.abs(top - want).max() <= slack, (top.min(axis=0), top.max(axis=0), want)


def covered_shipped(out: Path, info: dict) -> np.ndarray:
    """(size, size) int: for each shipped texel (top row first, as `read_exr`), the index of the
    placement whose seen lightmap covers one of its 2 x 2 doubled texels, or -1. Coverage is the
    bake's own: a doubled texel's centre inside (or on the edge of) a seen triangle."""
    size, glb = info["size"], out / f"{info['layout']}.glb"
    owner = np.full((2 * size) ** 2, -1)
    pids = sorted({node.split("/")[0] for node in info["uv_channel"]})
    for node, channel in info["uv_channel"].items():
        uv = corner_uv(glb, node, channel).astype(np.float64)
        ids = texels_of(uv[uv_areas(uv) > 0], 2 * size, edges=True)
        owner[ids] = pids.index(node.split("/")[0])
    blocks = owner.reshape(size, 2, size, 2).transpose(0, 2, 1, 3).reshape(size, size, 4)
    return blocks.max(axis=2)


def window(values: np.ndarray, fill: float, pick) -> np.ndarray:
    """`pick` (np.minimum or np.maximum) of `values` (n, n, ...) over each texel's 5 x 5 window."""
    n = values.shape[0]
    pad = np.pad(values, [(2, 2), (2, 2)] + [(0, 0)] * (values.ndim - 2), constant_values=fill)
    out = np.full(values.shape, fill, dtype=values.dtype)
    for dy in range(5):
        for dx in range(5):
            out = pick(out, pad[dy:dy + n, dx:dx + n])
    return out


def between_covered(cov: np.ndarray) -> np.ndarray:
    """The empty texels with a covered texel within 2 on both sides of them along one line (a row,
    a column or a diagonal): the gap between two islands, or between two arms of one."""
    n = cov.shape[0]
    pad = np.pad(cov, 2)

    def hit(dy, dx):
        return np.any([pad[2 + k * dy:2 + k * dy + n, 2 + k * dx:2 + k * dx + n]
                       for k in (1, 2)], axis=0)
    return ~cov & np.any([hit(dy, dx) & hit(-dy, -dx)
                          for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1))], axis=0)


def test_margin_holds_the_island_edge(box_on_slab):
    """Every empty texel within 2 texels of an island (no doubled texel of its block covered)
    holds a value inside the range of the covered texels round it (its 5 x 5 window): the margin
    is filled from the island's edge, so a bilinear read at the edge never averages in black. And
    an empty texel between two lit islands (a covered texel within 2 on both sides) is not black.
    Tolerance: one half-float step."""
    out, info = run(box_on_slab)
    pic = picture(out, info, "lamp").astype(np.float64)
    cov = covered_shipped(out, info) >= 0
    lo = window(np.where(cov[:, :, None], pic, np.inf), np.inf, np.minimum)
    hi = window(np.where(cov[:, :, None], pic, -np.inf), -np.inf, np.maximum)
    margin = ~cov & np.isfinite(lo).all(axis=2)
    lit = margin & (lo.min(axis=2) > 1e-4)
    assert lit.sum() > 20, lit.sum()  # the fixture's lit islands have a rim to test
    slack = 2.0 ** -10 * np.maximum(np.abs(np.where(np.isfinite(hi), hi, 0.0)), 1e-3)
    m = np.repeat(margin[:, :, None], 3, axis=2)
    assert (pic[m] >= (lo - slack)[m]).all(), "a margin texel is darker than its island's edge"
    assert (pic[m] <= (hi + slack)[m]).all(), "a margin texel is brighter than its island's edge"
    between = lit & between_covered(cov)
    assert between.any(), "no empty texel between two lit islands"
    assert (pic[between].max(axis=1) > 0).all(), "a black texel between two islands"
