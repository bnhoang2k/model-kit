"""Tiny scene folders for the bake tests, written through the kit's own exporter.

`box_on_slab`: one box piece placed twice (`Box`, `box_b`) on a slab whose faces are cut into
0.25 m cells, so some slab triangles lie wholly under a box. One camera hangs above, looking
straight down.

`shelf`: a plank with a UV set of its own (`plank`), a board with none (`board`), and a small
block (`hidden`) behind a wall (`wall`) that the one camera cannot see past.

`tube`: one thin smooth tube with flat ends (`tube`), 5 mm across and 30 cm long, lying along x,
with three cameras round it 120 degrees apart, so every face of its side is seen.

`mixed`: a flat sheet (`sheet`, one disk region) beside a smooth closed ball (`ball`, which falls
back to Smart UV Project), both seen by one camera above.

`chip`: a flat 6 mm sheet of 0.2 mm flat-shaded cells (`chip`), tilted, far from its piece's
origin, so float32 positions put noise in each cell's normal; one camera faces it.

Each folder has the bar's layout: `<name>/scene/` (the scene folder), `<name>/<pack>.glb`,
`<name>/<pack>.pack.json`. Nothing is committed: `make_scene` builds into `tmp_path`.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np

from model_kit.bars_build import install_bar

FIXTURES = Path(__file__).resolve().parent.parent / "core" / "fixtures"

BUILD = '''
import dataclasses
import math
import bmesh
import bpy
from mathutils import Vector
from pathlib import Path
from model_kit.blender import build, runner, soft
from model_kit.blender.export import export_pieces
from model_kit.core.manifest import (
    Layout, LayoutPlacement, Listed, Recipe, View, load_pack)
from model_kit.core.writer import write_layout
from model_kit.core.units import load_style

out, scene, style_path, which, toy_path, toy_style = runner.script_args()
style = load_style(Path(style_path))
build.reset()
mats = build.slots({"Fill": (0.8, 0.45, 0.2)})
pack_name = {"box_on_slab": "yard", "shelf": "stock", "tube": "pipe", "chip": "decal",
             "mixed": "lot"}[which]
pack = build.PackBuilder(pack_name, style.units_per_metre, Path(out), 10.0,
                         crease_deg=style.crease_deg)
view = View(1.0, "a test piece, seen from arm's length")
PROPS = {"listed": "body", "group": "body", "shape": "box"}


def add(name, bm, uv=False):
    root = build.piece(name)
    build.mesh_obj("body", bm, mats["Fill"], root, smooth=False, props=PROPS)
    pack.register(root, view=view, origin="floor, under the middle",
                  recipe=Recipe("assembly", "build.py"),
                  listed=[Listed("body", "A box", "a drawing", 0.01, None)],
                  done="A box.", size_class="piece", kind="box")
    return root


def box(x0, x1, y0, y1, z0, z1, cuts=0, uv_cells=0):
    bm = build.box_bm(x0, x1, y0, y1, z0, z1)
    if cuts:
        bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=cuts, use_grid_fill=True)
    if uv_cells:
        layer = bm.loops.layers.uv.new("UVMap")
        for face in bm.faces:
            for loop in face.loops:
                # exact in float32 both ways: the plank's corners land on 0 and 1
                loop[layer].uv = (loop.vert.co.x * 4 + 0.5, loop.vert.co.z * 4)
    return bm


if which == "box_on_slab":
    add("Slab", box(-1, 1, 0, 0.2, -1, 1, cuts=7))
    add("Box", box(-0.25, 0.25, 0, 0.5, -0.25, 0.25))
    # a rigged piece (the kit's toy figure): its armature sits under the root and is not
    # named `toy/<part>`, as a rigged character's is not
    toy = load_pack(Path(toy_path), load_style(Path(toy_style)))
    toy = dataclasses.replace(toy, units_per_metre=style.units_per_metre, pieces={
        n: dataclasses.replace(p, view=view) for n, p in toy.pieces.items()})
    toy_piece = toy.pieces["toy"]
    soft.build_piece(pack, toy_piece, {p.name: mats["Fill"] for p in toy_piece.parts})
    names = ["Slab", "Box", "toy"]
    placements = {
        "slab": LayoutPlacement("slab", pack_name, "Slab", (0.0, 0.0, 0.0), (0, 0, 0, 1)),
        "Box": LayoutPlacement("Box", pack_name, "Box", (-0.5, 0.2, 0.0), (0, 0, 0, 1),
                                 rests_on="slab"),
        "box_b": LayoutPlacement("box_b", pack_name, "Box", (0.5, 0.2, 0.0), (0, 0, 0, 1),
                                 rests_on="slab"),
        "doll": LayoutPlacement("doll", pack_name, "toy", (3.0, 0.0, 0.0), (0, 0, 0, 1)),
    }
elif which == "tube":
    # a smooth round tube along x with flat ends: TUBE_SIDES faces round, one ring a centimetre
    bm = bmesh.new()
    rings = []
    for i in range(TUBE_RINGS + 1):
        x = -TUBE_LENGTH / 2 + TUBE_LENGTH * i / TUBE_RINGS
        rings.append([bm.verts.new(build.B(
            x, TUBE_RADIUS * (1 + math.cos(2 * math.pi * k / TUBE_SIDES)),
            TUBE_RADIUS * math.sin(2 * math.pi * k / TUBE_SIDES))) for k in range(TUBE_SIDES)])
    for lo, hi in zip(rings, rings[1:]):
        for k in range(TUBE_SIDES):
            bm.faces.new((lo[k], lo[(k + 1) % TUBE_SIDES], hi[(k + 1) % TUBE_SIDES], hi[k]))
    bm.faces.new(rings[0])
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    root = build.piece("Tube")
    build.mesh_obj("body", bm, mats["Fill"], root, smooth=True, props=PROPS)
    pack.register(root, view=view, origin="floor, under the middle",
                  recipe=Recipe("assembly", "build.py"),
                  listed=[Listed("body", "A tube", "a drawing", 0.001, None)],
                  done="A tube.", size_class="piece", kind="box")
    names = ["Tube"]
    placements = {
        "tube": LayoutPlacement("tube", pack_name, "Tube", (0.0, 0.5, 0.0), (0, 0, 0, 1)),
    }
elif which == "mixed":
    # a flat sheet (one disk region) beside a smooth ball (closed: Smart UV Project's fallback)
    sheet = bmesh.new()
    bmesh.ops.create_grid(sheet, x_segments=4, y_segments=4, size=0.25)
    for v in sheet.verts:
        v.co = Vector(build.B(v.co.x, 0.0, v.co.y))
    add("Sheet", sheet)
    ball = bmesh.new()
    bmesh.ops.create_uvsphere(ball, u_segments=16, v_segments=8, radius=0.15)
    root = build.piece("Ball")
    build.mesh_obj("body", ball, mats["Fill"], root, smooth=True, props=PROPS)
    pack.register(root, view=view, origin="its centre",
                  recipe=Recipe("assembly", "build.py"),
                  listed=[Listed("body", "A ball", "a drawing", 0.01, None)],
                  done="A ball.", size_class="piece", kind="box")
    names = ["Sheet", "Ball"]
    placements = {
        "sheet": LayoutPlacement("sheet", pack_name, "Sheet", (0.0, 0.0, 0.0), (0, 0, 0, 1)),
        "ball": LayoutPlacement("ball", pack_name, "Ball", (0.6, 0.15, 0.0), (0, 0, 0, 1)),
    }
elif which == "chip":
    # a flat, tilted sheet of tiny flat-shaded cells far from its piece's origin: float32
    # positions put noise in each cell's own normal, as on a small decal
    bm = bmesh.new()
    c, u, w = (CHIP[k] for k in ("CHIP_CENTRE", "CHIP_U", "CHIP_V"))
    n, step = CHIP["CHIP_CELLS"], CHIP["CHIP_STEP"]
    grid = [[bm.verts.new(build.B(*(c[a] + (i - n / 2) * step * u[a] + (j - n / 2) * step * w[a]
                                    for a in range(3))))
             for j in range(n + 1)] for i in range(n + 1)]
    for i in range(n):
        for j in range(n):
            bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
    root = build.piece("Chip")
    build.mesh_obj("body", bm, mats["Fill"], root, smooth=False, props=PROPS)
    pack.register(root, view=view, origin="floor, under the middle",
                  recipe=Recipe("assembly", "build.py"),
                  listed=[Listed("body", "A chip", "a drawing", 0.001, None)],
                  done="A chip.", size_class="piece", kind="box")
    names = ["Chip"]
    placements = {
        "chip": LayoutPlacement("chip", pack_name, "Chip", (0.0, 0.0, 0.0), (0, 0, 0, 1)),
    }
else:
    add("Plank", box(-0.125, 0.125, 0, 0.25, -0.125, 0.125, uv_cells=1))
    add("Board", box(-0.125, 0.125, 0, 0.25, -0.125, 0.125))
    add("Hidden", box(-0.125, 0.125, 0, 0.25, -0.125, 0.125))
    add("Wall", box(-3, 3, 0, 4, -0.05, 0.05))
    # two parts that share one mesh (an instance), and a face narrower than the ray spacing
    twin = build.piece("Twin")
    left = build.mesh_obj("left", box(-0.05, 0.05, 0, 0.2, -0.05, 0.05), mats["Fill"], twin,
                          smooth=False, props=PROPS)
    right = bpy.data.objects.new("Twin/right", left.data)
    bpy.context.scene.collection.objects.link(right)
    right.parent = twin
    right.location = (0.3, 0, 0)
    build.tag(right, "Twin", "right", listed="body", group="body", shape="box")
    pack.register(twin, view=view, origin="floor, under the left part",
                  recipe=Recipe("assembly", "build.py"),
                  listed=[Listed("body", "A box", "a drawing", 0.01, None)],
                  done="Two boxes.", size_class="piece", kind="box")
    add("Wire", box(-0.002, 0.002, 0, 0.004, -0.002, 0.002))
    names = ["Plank", "Board", "Hidden", "Wall", "Twin", "Wire"]
    placements = {
        "wall": LayoutPlacement("wall", pack_name, "Wall", (0.0, 0.0, 0.0), (0, 0, 0, 1)),
        "plank": LayoutPlacement("plank", pack_name, "Plank", (-0.6, 1.0, 1.0), (0, 0, 0, 1)),
        "board": LayoutPlacement("board", pack_name, "Board", (0.6, 1.0, 1.0), (0, 0, 0, 1)),
        "hidden": LayoutPlacement("hidden", pack_name, "Hidden", (0.0, 1.0, -1.0), (0, 0, 0, 1)),
        "twin": LayoutPlacement("twin", pack_name, "Twin", (-0.15, 2.0, 1.0), (0, 0, 0, 1)),
        # 4 mm across, centred on the camera's axis: between two grid rays
        "wire": LayoutPlacement("wire", pack_name, "Wire", (0.0, 1.498, 1.0), (0, 0, 0, 1)),
    }
built = pack.finish(names)
export_pieces(built, Path(scene), pieces=names)
(Path(scene) / "layouts").mkdir(parents=True, exist_ok=True)
write_layout(Layout(name=which, frame="gltf", placements=placements, stands={}),
             Path(scene) / "layouts" / f"{which}.json")
runner.done()
'''

PACKS = {"box_on_slab": "yard", "shelf": "stock", "tube": "pipe", "chip": "decal",
         "mixed": "lot"}
# The chip: 30 x 30 cells of 0.2 mm (a 6 mm sheet), about 0.8 m from its piece's origin, on a plane
# tilted off every axis. Its normal is CHIP_N; CHIP_U and CHIP_V span it (U x V = N).
_N = np.array([0.2, 0.15, 1.0]) / np.linalg.norm([0.2, 0.15, 1.0])
_U = np.cross([0.3, 1.0, 0.0], _N)
_U /= np.linalg.norm(_U)
CHIP = {"CHIP_CENTRE": [0.6, 0.4, 0.3], "CHIP_U": _U.tolist(), "CHIP_V": np.cross(_N, _U).tolist(),
        "CHIP_N": _N.tolist(), "CHIP_CELLS": 30, "CHIP_STEP": 0.0002}
# The tube: 5 mm across, 30 cm long, 16 faces round, 30 rings along.
TUBE = {"TUBE_RADIUS": 0.0025, "TUBE_LENGTH": 0.3, "TUBE_SIDES": 16, "TUBE_RINGS": 30}
# Its axis runs along x at this height; three cameras 120 degrees apart round it, 0.5 m off.
TUBE_AXIS_Y = 0.5 + TUBE["TUBE_RADIUS"]


def _round_tube(angle_deg: float) -> dict:
    a = math.radians(angle_deg)
    return {"position": [0, TUBE_AXIS_Y + 0.5 * math.cos(a), 0.5 * math.sin(a)],
            "target": [0, TUBE_AXIS_Y, 0], "fov_deg": 40, "aspect": 1}


CAMERAS = {
    "box_on_slab": {"top": {"position": [0, 3, 0], "target": [0, 0, 0], "fov_deg": 60,
                            "aspect": 1}},
    "shelf": {"front": {"position": [0, 1.5, 5], "target": [0, 1.5, 0], "fov_deg": 40,
                        "aspect": 1}},
    "tube": {f"round{i}": _round_tube(angle) for i, angle in enumerate((0, 120, 240))},
    "mixed": {"above": {"position": [0.3, 1.5, 0.4], "target": [0.3, 0.0, 0.0], "fov_deg": 50,
                        "aspect": 1}},
    "chip": {"face": {"position": (np.array(CHIP["CHIP_CENTRE"]) + 0.1 * _N).tolist(),
                      "target": CHIP["CHIP_CENTRE"], "fov_deg": 20, "aspect": 1}},
}


def make_scene(tmp_path: Path, name: str) -> Path:
    """Build fixture `name` into `tmp_path/<name>/`; returns that folder (its `scene/` holds the
    scene)."""
    bar = tmp_path / name
    bar.mkdir()
    work = tmp_path / f"{name}-work"
    work.mkdir()
    style = work / "style.json"
    data = json.loads((FIXTURES / "shop.style.json").read_text())
    data["units_per_metre"] = 1.0
    style.write_text(json.dumps(data))
    script = work / "build_fixture.py"
    script.write_text("".join(f"{k} = {v!r}\n" for k, v in TUBE.items())
                      + f"CHIP = {CHIP!r}\n" + BUILD)
    install_bar(bar, PACKS[name], tmp=work / "tmp", script=script, style=style,
                args=lambda out, scene: [str(out), str(scene), str(style), name,
                              str(FIXTURES / "toy.pack.json"),
                              str(FIXTURES / "shop.style.json")],
                sources=[script], timeout=240)
    return bar


def write_stage(path: Path, name: str, *, live=(), texels_per_m: float = 16,
                mode: str = "full", cameras: dict | None = None, layers=("lamp",),
                lights: dict | None = None, seed: int = 1) -> Path:
    """A stage file for fixture `name`: by default one layer `lamp` with one point light."""
    path.write_text(json.dumps({
        "stage": 1, "layout": name, "mode": mode, "texels_per_m": texels_per_m,
        "samples": 16, "seed": seed, "layers": list(layers), "live": list(live),
        "lights": lights if lights is not None else {
            "bulb": {"kind": "point", "position": [0, 2.5, 0], "layer": "lamp"}},
        "cameras": cameras if cameras is not None else CAMERAS[name],
    }))
    return path


def read_exr(path: Path) -> np.ndarray:
    """An EXR written as half-float, ZIP (16 scanlines a block), channels B G R or R G B, as a
    (height, width, 3) float32 array in R, G, B order, top row first. Reads nothing else."""
    data = Path(path).read_bytes()
    assert data[:4] == b"\x76\x2f\x31\x01", "not an OpenEXR file"
    at, header = 8, {}
    while data[at] != 0:
        end = data.index(b"\0", at)
        name = data[at:end].decode()
        at = data.index(b"\0", end + 1) + 1
        size = int.from_bytes(data[at:at + 4], "little")
        header[name] = data[at + 4:at + 4 + size]
        at += 4 + size
    at += 1
    box = np.frombuffer(header["dataWindow"], dtype="<i4")
    width, height = int(box[2] - box[0] + 1), int(box[3] - box[1] + 1)
    assert header["compression"][0] == 3, "not ZIP"
    names, raw = [], header["channels"]
    i = 0
    while raw[i] != 0:
        end = raw.index(b"\0", i)
        names.append(raw[i:end].decode())
        assert int.from_bytes(raw[end + 1:end + 5], "little") == 1, "not half-float"
        i = end + 1 + 16
    blocks = -(-height // 16)
    table = np.frombuffer(data[at:at + 8 * blocks], dtype="<u8")
    out = np.zeros((height, width, len(names)), dtype=np.float32)
    for off in table:
        y = int.from_bytes(data[off:off + 4], "little", signed=True) - int(box[1])
        length = int.from_bytes(data[off + 4:off + 8], "little")
        packed = np.frombuffer(zlib.decompress(data[off + 8:off + 8 + length]), dtype=np.uint8)
        deltas = packed.astype(np.int64) - 128
        deltas[0] = packed[0]
        flat = (np.cumsum(deltas) % 256).astype(np.uint8)
        half = (len(flat) + 1) // 2
        joined = np.empty(len(flat), dtype=np.uint8)
        joined[0::2], joined[1::2] = flat[:half], flat[half:]
        rows = min(16, height - y)
        pixels = joined.view("<f2").astype(np.float32).reshape(rows, len(names), width)
        out[y:y + rows] = pixels.transpose(0, 2, 1)
    order = [names.index(c) for c in "RGB"]
    return out[:, :, order]
