"""Armatures from rest bones, and rigid and chain binds, in Blender."""

import json
from pathlib import Path

import pytest

from model_kit.blender.runner import run_blender

TIMEOUT = 240
FIX = Path(__file__).resolve().parents[1] / "core" / "fixtures"
TOY = FIX / "toy.pack.json"
STYLE = FIX / "shop.style.json"

pytestmark = pytest.mark.blender

# Shared set-up: the toy pack, an armature from its bones, and a gridded box that spans the
# arm (glTF x 0.15 to 0.25, y 0.3 to 0.6) so a chain bind has vertices in every band.
PRELUDE = """
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

from model_kit.blender import bind, build, export, runner
from model_kit.core.errors import ManifestError
from model_kit.core.forms import chain_weights
from model_kit.core.manifest import load_pack
from model_kit.core.tolerances import BIND_WEIGHT_FLOOR
from model_kit.core.units import load_style

toy_path, style_path = runner.script_args()
pack = load_pack(Path(toy_path), load_style(Path(style_path)))
rig = pack.pieces["toy"].rig

build.reset()
mats = build.slots({"Fill": (1, 1, 1)})
root = build.piece("toy")


def grid(name, x0, x1, y0, y1, z0, z1, cuts=12):
    bm = build.box_bm(x0, x1, y0, y1, z0, z1)
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    return build.mesh_obj(name, bm, mats["Fill"], root,
                          props={"listed": name, "group": name})


def to_rig(armature, ob, co):
    return Vector(build.G3(armature.matrix_world.inverted() @ ob.matrix_world @ co))
"""


def _run(tmp_path: Path, body: str):
    script = tmp_path / "script.py"
    script.write_text(PRELUDE + body)
    result = run_blender(script, [str(TOY), str(STYLE)], timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    return result


def _lines(log: str, tag: str) -> list:
    return [json.loads(line[len(tag) + 1:]) for line in log.splitlines()
            if line.startswith(tag + " ")]


ARMATURE = """
armature = bind.make_armature(root, rig.armature, rig.bones)
export._check_bones(armature)
rows = []
for b in rig.bones:
    bone = armature.data.bones[b.name]
    rows.append({
        "name": b.name,
        "parent": bone.parent.name if bone.parent else None,
        "head": [abs(a - c) for a, c in zip(build.G3(bone.head_local), b.rest.head)],
        "tail": [abs(a - c) for a, c in zip(build.G3(bone.tail_local), b.rest.tail)],
        "roll_want": b.rest.roll,
        "deform": [bone.use_deform, b.rest.deform],
        "connect": bone.use_connect,
        "mode": armature.pose.bones[b.name].rotation_mode,
        "want_parent": b.parent,
    })
print("ROWS", json.dumps(rows))
print("NAME", armature.name, armature.parent.name, armature.data.display_type)
for label, bones in (
    ("norest", [type(rig.bones[0])("x", None, 1.0, None)]),
    ("taken", rig.bones),
):
    try:
        bind.make_armature(root, rig.armature if label == "taken" else "other", bones)
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)
runner.done()
"""


def test_armature_matches_rest(tmp_path):
    result = _run(tmp_path, ARMATURE)
    rows = _lines(result.log, "ROWS")[0]
    assert len(rows) == 6
    for row in rows:
        assert row["parent"] == row["want_parent"], row
        assert max(row["head"]) < 1e-6 and max(row["tail"]) < 1e-6, row
        assert row["deform"][0] == row["deform"][1], row
        assert row["connect"] is False and row["mode"] == "XYZ", row
    assert "NAME toy_rig toy STICK" in result.log
    assert not [ln for ln in result.log.splitlines() if ln.startswith("NO RAISE")], result.log
    raised = {ln.split()[1]: ln for ln in result.log.splitlines() if ln.startswith("RAISED ")}
    assert set(raised) == {"norest", "taken"}
    assert "'x'" in raised["norest"] and "toy_rig" in raised["taken"]


ROLL = """
bones = [type(rig.bones[0])(b.name, b.parent, b.length,
                            type(b.rest)(b.rest.head, b.rest.tail, 0.3, b.rest.deform))
         for b in rig.bones]
armature = bind.make_armature(root, "rolled", bones)
export._check_bones(armature)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode="EDIT")
print("ROLLS", json.dumps([round(armature.data.edit_bones[b.name].roll, 6) for b in bones]))
bpy.ops.object.mode_set(mode="OBJECT")
runner.done()
"""


def test_roll_is_set(tmp_path):
    result = _run(tmp_path, ROLL)
    assert _lines(result.log, "ROLLS")[0] == [0.3] * 6


RIGID = """
armature = bind.make_armature(root, rig.armature, rig.bones)
body = grid("body", -0.1, 0.1, 0.2, 0.7, -0.08, 0.08, cuts=3)
bind.bind_part(body, armature, rig.bind["body"])
mods = [(m.type, m.object.name, m.use_vertex_groups, m.use_bone_envelopes)
        for m in body.modifiers]
groups = {}
for v in body.data.vertices:
    groups.setdefault(tuple((body.vertex_groups[g.group].name, g.weight) for g in v.groups), 0)
    groups[tuple((body.vertex_groups[g.group].name, g.weight) for g in v.groups)] += 1
print("RIGID", json.dumps({"mods": mods, "n": len(body.data.vertices),
                           "groups": [[list(map(list, k)), c] for k, c in groups.items()]}))
runner.done()
"""


def test_rigid_bind(tmp_path):
    result = _run(tmp_path, RIGID)
    data = _lines(result.log, "RIGID")[0]
    assert data["mods"] == [["ARMATURE", "toy_rig", True, False]]
    assert data["groups"] == [[[["body", 1.0]], data["n"]]]


CHAIN = """
armature = bind.make_armature(root, rig.armature, rig.bones)
arm = grid("arm_L", 0.15, 0.25, 0.28, 0.62, -0.05, 0.05)
b = rig.bind["arm_L"]
bind.bind_part(arm, armature, b)
pts = np.array([to_rig(armature, arm, v.co) for v in arm.data.vertices])
want = chain_weights(pts, b.falloff, len(b.bones))
worst, sums, missing, present = 0.0, [], 0, 0
for v, w in zip(arm.data.vertices, want):
    got = {arm.vertex_groups[g.group].name: g.weight for g in v.groups}
    sums.append(sum(got.values()))
    for i, name in enumerate(b.bones):
        if w[i] <= BIND_WEIGHT_FLOOR:
            missing += name in got
        else:
            present += 1
            worst = max(worst, abs(got.get(name, 9.0) - w[i]))
print("CHAIN", json.dumps({"worst": worst, "sum_lo": min(sums), "sum_hi": max(sums),
                           "dropped_kept": missing, "present": present,
                           "names": sorted(g.name for g in arm.vertex_groups),
                           "both": int(sum(1 for w in want if (w > 0.01).all()))}))
nofall = type(b)(b.bones, "chain", None)
other = grid("arm_R", -0.25, -0.15, 0.28, 0.62, -0.05, 0.05, cuts=2)
try:
    bind.bind_part(other, armature, nofall)
    print("NO RAISE nofalloff")
except ManifestError as exc:
    print("RAISED nofalloff", exc)
runner.done()
"""


def test_chain_bind_matches_formula(tmp_path):
    result = _run(tmp_path, CHAIN)
    data = _lines(result.log, "CHAIN")[0]
    assert data["worst"] < 1e-6, data
    assert 0.997 <= data["sum_lo"] and data["sum_hi"] <= 1.003, data
    assert data["dropped_kept"] == 0 and data["both"] > 0, data
    assert data["names"] == ["arm.L", "hand.L"]
    assert any(ln.startswith("RAISED nofalloff") and "arm_R" in ln
               for ln in result.log.splitlines()), result.log


FOLLOW = """
armature = bind.make_armature(root, rig.armature, rig.bones)
arm = grid("arm_L", 0.15, 0.25, 0.28, 0.62, -0.05, 0.05, cuts=3)
bind.bind_part(arm, armature, type(rig.bind["body"])(("arm.L",), "rigid"))
rest = [Vector(v.co) for v in arm.data.vertices]
pb = armature.pose.bones["arm.L"]
bone = armature.data.bones["arm.L"]
pb.rotation_euler = (np.radians(30), 0, 0)
bpy.context.view_layer.update()
ev = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
me = ev.to_mesh()
delta = armature.matrix_world @ pb.matrix @ bone.matrix_local.inverted() \
    @ armature.matrix_world.inverted()
worst = 0.0
moved = 0.0
for v0, v1 in zip(rest, me.vertices):
    want = delta @ arm.matrix_world @ v0
    got = arm.matrix_world @ v1.co
    worst = max(worst, (want - got).length)
    moved = max(moved, (got - arm.matrix_world @ v0).length)
print("FOLLOW", json.dumps({"worst": worst, "moved": moved}))
runner.done()
"""


def test_bound_part_follows_bone(tmp_path):
    result = _run(tmp_path, FOLLOW)
    data = _lines(result.log, "FOLLOW")[0]
    assert data["moved"] > 0.05, data
    assert data["worst"] < 1e-6, data


RADIUS = """
bones = list(rig.bones)
armature = bind.make_armature(root, rig.armature, bones)
bm = bmesh.new()
bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=0.05)
for v in bm.verts:
    v.co = Vector(build.B(0.2, 0.375, 0)) + Vector((v.co.x, v.co.y, v.co.z))
ball = build.mesh_obj("ball", bm, mats["Fill"], root, props={"listed": "ball", "group": "ball"})
bind.bind_part(ball, armature, type(rig.bind["body"])(("hand.L",), "rigid"))
chain = rig.chains["arm.L"]
r = bind.effector_radius(armature, chain, rig.contact_groups)
print("RADIUS", json.dumps(r))
try:
    bind.effector_radius(armature, rig.chains["arm.R"], rig.contact_groups)
    print("NO RAISE empty")
except ManifestError as exc:
    print("RAISED empty", exc)
runner.done()
"""


def test_effector_radius(tmp_path):
    result = _run(tmp_path, RADIUS)
    assert abs(_lines(result.log, "RADIUS")[0] - 0.05) < 1e-3, result.log
    assert any(ln.startswith("RAISED empty") for ln in result.log.splitlines()), result.log


NODEFORM = """
armature = bind.make_armature(root, rig.armature, rig.bones)
ob = grid("body", -0.1, 0.1, 0.2, 0.7, -0.08, 0.08, cuts=2)
try:
    bind.bind_part(ob, armature, type(rig.bind["body"])(("root",), "rigid"))
    print("NO RAISE nodeform")
except ManifestError as exc:
    print("RAISED nodeform", exc)
print("STATE", len(ob.vertex_groups), len(ob.modifiers))
runner.done()
"""


def test_bind_to_non_deform_bone_raises(tmp_path):
    result = _run(tmp_path, NODEFORM)
    lines = result.log.splitlines()
    assert not [ln for ln in lines if ln.startswith("NO RAISE")], result.log
    assert any(ln.startswith("RAISED nodeform") and "'root'" in ln and "toy/body" in ln
               for ln in lines), result.log
    assert "STATE 0 0" in lines
