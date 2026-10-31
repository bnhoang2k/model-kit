"""Rigged pieces in PackBuilder, and `soft.build_piece`: the toy figure built from its pack.json."""

import dataclasses
import json
from pathlib import Path

import pytest

from model_kit.blender.runner import run_blender
from model_kit.core.glb import read_glb_json
from model_kit.core.manifest import View, load_pack
from model_kit.core.units import load_style

TIMEOUT = 300
FIX = Path(__file__).resolve().parents[1] / "core" / "fixtures"
TOY = FIX / "toy.pack.json"
STYLE = FIX / "shop.style.json"

pytestmark = pytest.mark.blender

PRELUDE = """
import dataclasses
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from model_kit.blender import bind, build, export, runner, soft
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import View, load_pack
from model_kit.core.units import load_style

toy_path, style_path, out_arg = runner.script_args()[:3]
STYLE = load_style(Path(style_path))
VIEW = View(1.0, "a test figure, seen from arm's length")
AUTHORED = load_pack(Path(toy_path), STYLE)
AUTHORED = dataclasses.replace(AUTHORED, pieces={
    name: dataclasses.replace(piece, view=VIEW) for name, piece in AUTHORED.pieces.items()})
OUT = Path(out_arg)


def build_toy(pack, out):
    \"\"\"Reset, then build the toy's piece from `pack` and return (builder, root, armature).\"\"\"
    build.reset()
    mats = build.slots({"Fill": (1, 1, 1)})
    piece = pack.pieces["toy"]
    builder = build.PackBuilder(pack.name, pack.units_per_metre, out, pack.spacing, crease_deg=35)
    root, armature = soft.build_piece(builder, piece, {p.name: mats["Fill"] for p in piece.parts})
    return builder, root, armature


def weights(ob):
    return [{ob.vertex_groups[g.group].name: round(g.weight, 6) for g in v.groups}
            for v in ob.data.vertices]


def parts_of(root):
    return {o.name.split("/", 1)[1]: o for o in root.children_recursive if o.type == "MESH"}
"""


def _run(tmp_path: Path, body: str, *, blend: Path | None = None, out: Path | None = None):
    script = tmp_path / "script.py"
    script.write_text(PRELUDE + body)
    out = out if out is not None else tmp_path / "out"
    result = run_blender(script, [str(TOY), str(STYLE), str(out)], blend=blend, timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    return result


def _lines(result, tag: str) -> list:
    return [json.loads(line[len(tag) + 1:]) for line in result.log.splitlines()
            if line.startswith(tag + " ")]


def _toy():
    return load_pack(TOY, load_style(STYLE))


ROUND_TRIP = """
builder, root, armature = build_toy(AUTHORED, OUT)
builder.finish(["toy"])
runner.done()
"""


def test_rigged_toy_round_trip(tmp_path):
    _run(tmp_path, ROUND_TRIP)
    authored = _toy().pieces["toy"]
    written = load_pack(tmp_path / "out" / "toy.pack.json", load_style(STYLE)).pieces["toy"]
    assert written.rig == authored.rig
    assert [p.form for p in written.parts] == [p.form for p in authored.parts]
    assert all(p.form is not None for p in written.parts)
    assert written.measured["kind"] == "character"

    gltf = read_glb_json(tmp_path / "out" / "toy.glb")
    assert len(gltf["skins"]) == 1
    joints = [gltf["nodes"][j]["name"] for j in gltf["skins"][0]["joints"]]
    assert sorted(joints) == sorted(b.name for b in authored.rig.bones)
    assert len(joints) == 6
    by_part = {n["name"]: n for n in gltf["nodes"] if "mesh" in n}
    assert sorted(by_part) == sorted(f"toy/{p.name}" for p in authored.parts)
    assert all(n.get("skin") == 0 for n in by_part.values())


KEEPS = """
builder, root, armature = build_toy(AUTHORED, OUT)
before = {name: weights(ob) for name, ob in parts_of(root).items()}
builder.finish(["toy"])
(OUT / "weights.json").write_text(json.dumps(before))
runner.done()
"""

READ_BACK = """
before = json.loads((OUT / "weights.json").read_text())
armature = bpy.data.objects[AUTHORED.pieces["toy"].rig.armature]
rows = []
for ob in bpy.data.objects:
    if ob.type != "MESH":
        continue
    name = ob.name.split("/", 1)[1]
    mods = [(m.type, m.object.name if m.object else None) for m in ob.modifiers]
    rows.append({"part": name, "mods": mods, "same": weights(ob) == before[name],
                 "groups": sorted(g.name for g in ob.vertex_groups)})
print("ROWS", json.dumps(rows))
runner.done()
"""


def test_finish_keeps_groups_and_armature(tmp_path):
    _run(tmp_path, KEEPS)
    authored = _toy().pieces["toy"]
    again = _run(tmp_path, READ_BACK, blend=tmp_path / "out" / "toy.blend")
    (rows,) = _lines(again, "ROWS")
    assert sorted(r["part"] for r in rows) == sorted(p.name for p in authored.parts)
    for row in rows:
        assert row["mods"] == [["ARMATURE", authored.rig.armature]], row
        assert row["groups"] == sorted(authored.rig.bind[row["part"]].bones), row
        assert row["same"], row


POSE = """
builder, root, armature = build_toy(AUTHORED, OUT)
arm = parts_of(root)["arm_L"]
rest = [tuple(v.co) for v in arm.data.vertices]

action = bpy.data.actions.new("bend")
armature.animation_data_create().action = action
pb = armature.pose.bones["arm.L"]
pb.rotation_euler = (0.0, 0.0, 1.2)
pb.keyframe_insert("rotation_euler", frame=1)
bpy.context.scene.frame_set(1)

# The pose really bends the mesh, so a bake would show.
dg = bpy.context.evaluated_depsgraph_get()
posed = [tuple(v.co) for v in arm.evaluated_get(dg).data.vertices]
print("BENT", max(np.linalg.norm(np.subtract(a, b)) for a, b in zip(rest, posed)))

builder.finish(["toy"])
after = [tuple(v.co) for v in parts_of(root)["arm_L"].data.vertices]
print("MOVED", len(rest) != len(after) or max(
    np.linalg.norm(np.subtract(a, b)) for a, b in zip(rest, after)))
runner.done()
"""


def test_finish_does_not_bake_pose(tmp_path):
    result = _run(tmp_path, POSE)
    (bent,) = _lines(result, "BENT")
    (moved,) = _lines(result, "MOVED")
    assert bent > 0.01
    assert moved == pytest.approx(0.0, abs=1e-6)


REBUILD = """
builder, root, armature = build_toy(AUTHORED, OUT / "one")
builder.finish(["toy"])
second = load_pack(OUT / "one" / "toy.pack.json", STYLE)
builder, root, armature = build_toy(second, OUT / "two")
builder.finish(["toy"])
runner.done()
"""


def test_rebuild_is_a_fixed_point(tmp_path):
    _run(tmp_path, REBUILD)
    one = (tmp_path / "out" / "one" / "toy.pack.json").read_bytes()
    two = (tmp_path / "out" / "two" / "toy.pack.json").read_bytes()
    assert one == two
    authored = _toy().pieces["toy"]
    written = load_pack(tmp_path / "out" / "one" / "toy.pack.json", load_style(STYLE)).pieces["toy"]
    # the authored toy has no view; the script gives it one, and it comes back as declared
    assert written.view == View(1.0, "a test figure, seen from arm's length")
    assert dataclasses.replace(written, measured={}, view=None) == authored


REFUSALS = """
def raises(label, fn):
    try:
        fn()
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)


def box_piece(name="Crate"):
    mats = build.slots({"Fill": (1, 1, 1)})
    root = build.piece(name)
    bm = build.box_bm(-0.1, 0.1, 0.0, 0.2, -0.1, 0.1)
    build.mesh_obj("lid", bm, mats["Fill"], root, props={"listed": "soft", "group": "lid", "shape": "box"})
    return root, mats


def register(builder, root, **extra):
    authored = AUTHORED.pieces["toy"]
    builder.register(root, origin="floor", recipe=authored.recipe, listed=authored.listed,
                     done="d", size_class="piece", kind="character", view=VIEW, **extra)


piece = AUTHORED.pieces["toy"]
rig = piece.rig


def armature_under_unrigged():
    build.reset()
    root, _ = box_piece()
    bind.make_armature(root, "stray", rig.bones)
    builder = build.PackBuilder("p", 1.0, OUT / "a", None, crease_deg=35)
    register(builder, root)
    builder.finish(["Crate"])


def second_armature():
    builder, root, armature = build_toy(AUTHORED, OUT / "b")
    bind.make_armature(root, "second", rig.bones)
    builder.finish(["toy"])


def form_for_missing_part():
    build.reset()
    root, _ = box_piece()
    builder = build.PackBuilder("p", 1.0, OUT / "c", None, crease_deg=35)
    register(builder, root, forms={"ghost": piece.parts[0].form})
    builder.finish(["Crate"])


def rig_without_armature():
    build.reset()
    root, _ = box_piece("toy")
    builder = build.PackBuilder("p", 1.0, OUT / "d", None, crease_deg=35)
    register(builder, root, rig=rig)
    builder.finish(["toy"])


def piece_without_view():
    pack = dataclasses.replace(AUTHORED, pieces={
        name: dataclasses.replace(piece, view=None) for name, piece in AUTHORED.pieces.items()})
    build_toy(pack, OUT / "e")


raises("armature_under_unrigged", armature_under_unrigged)
raises("second_armature", second_armature)
raises("form_for_missing_part", form_for_missing_part)
raises("rig_without_armature", rig_without_armature)
raises("piece_without_view", piece_without_view)
runner.done()
"""


def test_armature_under_unrigged_piece_raises(tmp_path):
    result = _run(tmp_path, REFUSALS)
    lines = {line.split()[1]: line for line in result.log.splitlines()
             if line.startswith(("RAISED ", "NO RAISE "))}
    assert not [k for k, v in lines.items() if v.startswith("NO RAISE")], lines
    assert "not rigged" in lines["armature_under_unrigged"]
    assert "stray" in lines["armature_under_unrigged"]
    assert "second" in lines["second_armature"]
    assert "ghost" in lines["form_for_missing_part"]
    assert "toy_rig" in lines["rig_without_armature"]
    assert "no view" in lines["piece_without_view"] and "toy" in lines["piece_without_view"]
