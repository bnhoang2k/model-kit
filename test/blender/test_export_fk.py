"""The exporters and the numpy forward kinematics, checked against Blender itself."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import model_kit.blender
from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.blender.runner import run_blender
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.metagate import run_gate
from model_kit.core.rig import pose_world
from model_kit.core.scene import _read_rig, load_scene

FIX = Path(__file__).parents[1] / "core" / "fixtures"
SELFTEST = Path(model_kit.blender.__file__).parent / "_selftest_fk.py"
TIMEOUT = 240

pytestmark = pytest.mark.blender


def _pose(raw: dict) -> dict:
    return {bone: (np.array(euler), np.array(loc)) for bone, (euler, loc) in raw.items()}


def _run(tmp_path: Path, body: str, *args: str):
    script = tmp_path / "script.py"
    script.write_text(body)
    return run_blender(script, list(args), timeout=TIMEOUT)


def test_axis_matrix_is_z_up_to_y_up():
    # Blender (x, y, z) lands at glTF (x, z, -y): up stays up, Blender's front (-Y) is +Z.
    assert BLENDER_TO_GLTF @ np.array([1, 2, 3, 1]) == pytest.approx([1, 3, -2, 1])


def test_fk_matches_blender(tmp_path):
    result = run_blender(SELFTEST, [str(tmp_path)], timeout=TIMEOUT)
    assert result.ok, result.log
    rig, clips = _read_rig(tmp_path / "rigs" / "shop" / "fk.json")
    assert clips == []
    assert rig.order == ("root", "body", "hand.R")
    poses = json.loads((tmp_path / "fk_poses.json").read_text())["poses"]
    assert len(poses) == 5
    for case in poses:
        got = pose_world(rig, _pose(case["pose"]))
        for bone, want in case["world"].items():
            np.testing.assert_allclose(got[bone], np.array(want), rtol=0, atol=1e-6,
                                       err_msg=f"bone {bone} in pose {case['pose']}")
    # The poses are not all rest, and the armature object is not at the origin.
    assert not np.allclose(rig.world, BLENDER_TO_GLTF)
    moved = [case for case in poses if any(np.any(e) or np.any(v)
                                           for e, v in case["pose"].values())]
    assert len(moved) == 4


def test_export_rejects_quaternion_bones(tmp_path):
    result = _run(tmp_path, (
        "from model_kit.blender import export, runner\n"
        "from model_kit.blender._selftest_fk import build_rig, rig_pack\n"
        "arm = build_rig()\n"
        "arm.pose.bones['body'].rotation_mode = 'QUATERNION'\n"
        "export.export_rig(arm, rig_pack(), 'Doll', runner.script_args()[0], clips=[])\n"
        "runner.done()\n"
    ), str(tmp_path / "out"))
    assert result.ok is False
    assert "body" in result.log and "rotation_mode" in result.log and "'XYZ'" in result.log
    assert not (tmp_path / "out" / "rigs").exists()


def test_export_rejects_constrained_bones(tmp_path):
    result = _run(tmp_path, (
        "from model_kit.blender import export, runner\n"
        "from model_kit.blender._selftest_fk import build_rig, rig_pack\n"
        "arm = build_rig()\n"
        "arm.pose.bones['hand.R'].constraints.new('COPY_LOCATION')\n"
        "export.export_rig(arm, rig_pack(), 'Doll', runner.script_args()[0], clips=[])\n"
        "runner.done()\n"
    ), str(tmp_path / "out"))
    assert result.ok is False
    assert "'hand.R'" in result.log and "constraint" in result.log
    assert not (tmp_path / "out" / "rigs").exists()


@pytest.mark.parametrize("breaks, words", [
    pytest.param("arm.data.bones['body'].use_inherit_rotation = False\n",
                 ["'body'", "inherit"], id="no-rotation-inheritance"),
    pytest.param("arm.data.bones['hand.R'].inherit_scale = 'NONE'\n",
                 ["'hand.R'", "inherit"], id="no-scale-inheritance"),
    pytest.param("arm.data.bones['root'].use_local_location = False\n",
                 ["'root'", "use_local_location"], id="global-location"),
])
def test_export_rig_refuses_a_structural_break_of_the_bones(tmp_path, breaks, words):
    """The rig part of the bone contract reads no pose value, so export_rig still enforces it."""
    result = _run(tmp_path, (
        "from model_kit.blender import export, runner\n"
        "from model_kit.blender._selftest_fk import build_rig, rig_pack\n"
        "arm = build_rig()\n"
        + breaks +
        "export.export_rig(arm, rig_pack(), 'Doll', runner.script_args()[0], clips=[])\n"
        "runner.done()\n"
    ), str(tmp_path / "out"))
    assert result.ok is False
    for word in words:
        assert word in result.log, (word, result.log[-1500:])
    assert not (tmp_path / "out" / "rigs").exists()


POSED_RIG = """
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, hand_chain, rig_pack

at_rest, posed, keyed = (Path(p) for p in runner.script_args())
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.matrix_world = arm.matrix_world.copy()
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")  # a skinned weight on it
pack = rig_pack(contact_groups={"hand.R": ["hand.R"]}, chains={"arm.R": hand_chain("hand.R")},
                parts={"Hand": "hand.R"})
export.export_rig(arm, pack, "Doll", at_rest, clips=[])

# The current pose scales the skinned bone and moves the connected one. No action yet.
hand_bone, body = arm.pose.bones["hand.R"], arm.pose.bones["body"]
assert arm.animation_data is None or arm.animation_data.action is None
hand_bone.scale = (1e-4, 1e-4, 1e-4)
body.location = (0.0, 0.1, 0.0)
print("POSE", tuple(hand_bone.scale), tuple(body.location))
export.export_rig(arm, pack, "Doll", posed, clips=[])
print("POSED_OK")

# An active action keys the same scale and move, and the scene sits on a keyed frame.
hand_bone.keyframe_insert("scale", frame=1)
body.keyframe_insert("location", frame=1)
bpy.context.scene.frame_set(1)
assert arm.animation_data.action is not None
print("KEYED", tuple(hand_bone.scale), tuple(body.location))
export.export_rig(arm, pack, "Doll", keyed, clips=[])
print("KEYED_OK")
runner.done()
"""


def test_export_rig_does_not_read_the_current_pose(tmp_path):
    """A scaled skinned bone and a moved connected bone, bare or keyed by the active action,
    leave the rig file exactly as the rest-pose export made it."""
    outs = [tmp_path / name for name in ("at_rest", "posed", "keyed")]
    result = _run(tmp_path, POSED_RIG, *map(str, outs))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert "POSE (9.999999747378752e-05, 9.999999747378752e-05, 9.999999747378752e-05) " \
        "(0.0, 0.10000000149011612, 0.0)" in lines
    assert "POSED_OK" in lines and "KEYED_OK" in lines
    assert any(line.startswith("KEYED (9.99") for line in lines)  # the action holds the scale
    want = {name: (outs[0] / "rigs" / "shop" / name).read_bytes()
            for name in ("Doll.json", "Doll.rest.npz")}
    for out in outs[1:]:
        for name, data in want.items():
            assert (out / "rigs" / "shop" / name).read_bytes() == data, (out.name, name)


ROUND_TRIP = """
import json, math
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, keyed_clip
from model_kit.core.manifest import Prop, load_pack
from model_kit.core.units import load_style

out, side, pack_file, style_file = (Path(p) for p in runner.script_args())
pack = load_pack(pack_file, load_style(style_file))
scene = bpy.context.scene
arm = build_rig("DollRig")

# Skinned meshes, in the armature's space, named for the Doll's parts. "arm_R" is all hand.R.
# In "torso", body is the strongest group of the first three vertices only (the other five lean
# to root). "foot_R" is all root.
hand = cube("arm_R", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
arm_mesh = cube("torso", (0.0, 0.05, 0.7), (0.05, 0.05, 0.05))
foot = cube("foot_R", (0.1, 0.0, 0.1), (0.05, 0.05, 0.05))
for obj in (hand, arm_mesh, foot):
    obj.matrix_world = arm.matrix_world.copy()
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
g = hand.vertex_groups.new(name="hand.R")
g.add(range(8), 1.0, "REPLACE")
body, root = arm_mesh.vertex_groups.new(name="body"), arm_mesh.vertex_groups.new(name="root")
body.add(range(3), 0.9, "REPLACE")
root.add(range(3), 0.1, "REPLACE")
body.add(range(3, 8), 0.4, "REPLACE")
root.add(range(3, 8), 0.6, "REPLACE")
foot.vertex_groups.new(name="root").add(range(8), 1.0, "REPLACE")
# A mask group names no bone, so it never takes a vertex, even at full weight.
arm_mesh.vertex_groups.new(name="mask").add(range(8), 1.0, "REPLACE")

# An outline shell that the export switches off; it would change the vertex count.
shell = hand.modifiers.new("Outline", "SOLIDIFY")
shell.thickness = 0.01

# A two-part piece whose root is moved and turned.
bench = bpy.data.objects.new("Bench", None)
scene.collection.objects.link(bench)
bench.location = (2.0, 1.0, 0.0)
bench.rotation_euler = (0.0, 0.0, math.pi / 2)
seat = cube("seat_top", (0.0, 0.0, 0.0), (0.3, 0.1, 0.02))
seat.parent = bench
seat.location = (0.0, 0.2, 0.45)
leg = cube("leg_L", (0.0, 0.0, 0.0), (0.02, 0.05, 0.2))
leg.parent = bench
leg.location = (-0.25, 0.0, 0.2)
leg.rotation_euler = (0.0, 0.0, math.pi / 2)
# The seat has an outward outline shell; the export must leave it out of the bounds.
seat_shell = seat.modifiers.new("Outline", "SOLIDIFY")
seat_shell.thickness, seat_shell.offset = 0.05, 1.0
# A one-part piece with no outline: hide_modifiers is checked over the whole call.
lamp_obj = bpy.data.objects.new("Lamp", None)
scene.collection.objects.link(lamp_obj)
stem = cube("stem", (0.0, 0.0, 0.6), (0.01, 0.01, 0.6))
stem.parent = lamp_obj

# A keyed clip.
keys = {
    1: {"root": ((0, 0, 0), (0, 0, 0)), "body": ((0, 0, 0), (0, 0, 0)),
        "hand.R": ((0, 0, 0), (0, 0, 0))},
    3: {"root": ((0.2, -0.1, 0.5), (0.1, 0.0, -0.05)), "body": ((0.4, 0.3, -0.2), (0, 0, 0)),
        "hand.R": ((-0.6, 0.2, 0.9), (0.02, 0.03, -0.01))},
}
for frame, pose in keys.items():
    for name, (euler, loc) in pose.items():
        pb = arm.pose.bones[name]
        pb.rotation_euler = euler
        pb.location = loc
        pb.keyframe_insert("rotation_euler", frame=frame)
        pb.keyframe_insert("location", frame=frame)
action = arm.animation_data.action
action.name = "sit_reach"
scene.frame_set(1)

lamp = Prop(id="lamp", piece="Lamp", pack="shop", bone="hand.R",
            offset_position=(0.0, 0.02, 0.0),
            offset_quaternion=(0.0, 0.0, 0.3826834323650898, 0.9238795325112867),
            show_frames=((1.0, 48.0),))

export.export_rig(arm, pack, "Doll", out, clips=["sit_reach"])
export.export_pieces(pack, out, pieces=["Bench", "Lamp"], hide_modifiers=["Outline"])
export.export_samples(arm, action, pack, "Doll",
                      keyed_clip("sit_reach", frames=(1, 3), props={"lamp": lamp}), out)
assert hand.modifiers["Outline"].show_viewport, "the export restores the outline"
assert seat.modifiers["Outline"].show_viewport, "the export restores the outline"

poses = []
for i in range(5):
    t = 1 + 0.5 * i
    scene.frame_set(int(math.floor(t)), subframe=t - math.floor(t))
    poses.append({b.name: [list(b.rotation_euler), list(b.location)] for b in arm.pose.bones})
side.write_text(json.dumps({"poses": poses}))
runner.done()
"""


def _shop_with_test_rig() -> dict:
    """The shop pack, its Doll rigged like `_selftest_fk.build_rig` and its Bench and Lamp
    cut down to the parts the round trip builds (the Lamp loses its connector with its `lead`):
    load checks that a rig file has the bones and parents of its piece's rig, and the export
    that the meshes are the parts."""
    pack = json.loads((FIX / "shop.pack.json").read_text())
    bench = pack["pieces"]["Bench"]
    bench["parts"] = [p for p in bench["parts"] if p["name"] != "leg_R"]
    lamp = pack["pieces"]["Lamp"]
    lamp["parts"] = [dict(p, on=None) for p in lamp["parts"] if p["name"] == "stem"]
    del lamp["connectors"]  # the connector's part, `lead`, is cut with the other parts
    rig = pack["pieces"]["Doll"]["rig"]
    rig["bones"] = [{"name": "root", "parent": None, "length": 0.4},
                    {"name": "body", "parent": "root", "length": 0.6},
                    {"name": "hand.R", "parent": "body", "length": 0.3}]
    rig["bind"] = {"torso": {"bones": ["body"], "mode": "rigid"},
                   "arm_R": {"bones": ["hand.R"], "mode": "rigid"},
                   "foot_R": {"bones": ["root"], "mode": "rigid"}}
    rig["contact_groups"] = {"hand.R": ["hand.R"], "torso": ["body"]}
    rig["chains"]["arm.R"].update(bones=["hand.R"], dofs={"hand.R": "xyz"}, limits={})
    rig["mirror"] = {"pairs": [], "axis": "x"}
    rig["outline_modifiers"] = ["Outline"]
    # The fixture's supports and joints name the shop Doll's own bones and groups; keep them
    # true to the rig cut down above (the supports are groups, the joints bend this rig).
    rig["supports"] = ["torso"]
    rig["joints"] = {"hand.R": {
        "range_deg": {"x": [-60, 60], "y": [-60, 60], "z": [-60, 60]},
        "why": "test fixture: schema contract, not anatomy"}}
    return pack


@pytest.fixture(scope="module")
def round_trip(tmp_path_factory) -> SimpleNamespace:
    """One Blender run exports the round-trip scene; the loaded scene and Blender's own poses."""
    tmp_path = tmp_path_factory.mktemp("round_trip")
    out, side = tmp_path / "scene", tmp_path / "side.json"
    pack_file = tmp_path / "shop.pack.json"
    pack_file.write_text(json.dumps(_shop_with_test_rig()))
    result = _run(tmp_path, ROUND_TRIP, str(out), str(side), str(pack_file),
                  str(FIX / "shop.style.json"))
    assert result.ok, result.log
    # The caller owns the manifests; the exporters own rigs/, pieces/ and samples/.
    assert sorted(p.name for p in out.iterdir()) == ["pieces", "rigs", "samples"]
    assert (out / "rigs" / "shop" / "Doll.json").is_file()
    shutil.copy(FIX / "shop.style.json", out / "style.json")
    shutil.copy(FIX / "shop.clips.json", out / "clips.json")
    (out / "packs").mkdir()
    shutil.copy(pack_file, out / "packs" / "shop.pack.json")
    poses = json.loads(side.read_text())["poses"]
    return SimpleNamespace(out=out, ctx=load_scene(out), poses=poses)


def test_export_round_trips_through_load_scene(round_trip):
    ctx = round_trip.ctx
    assert ctx.rig_of_clip == {"sit_reach": "shop:Doll"}
    rig = ctx.rigs["shop:Doll"]
    s = ctx.samples["sit_reach"]
    np.testing.assert_array_equal(s.times, [1.0, 1.5, 2.0, 2.5, 3.0])
    assert set(s.bones) == {"root", "body", "hand.R"}

    # Bone matrices: Blender's posed bones equal numpy FK on the exported rig.
    for i, raw in enumerate(round_trip.poses):
        got = pose_world(rig, _pose(raw))
        for bone in s.bones:
            np.testing.assert_allclose(s.bones[bone][i], got[bone], rtol=0, atol=1e-6)
    assert not np.allclose(s.bones["hand.R"][0], s.bones["hand.R"][-1])

    # Group points: a vertex belongs to its strongest deform group (the mask never wins);
    # the outline shell is off.
    assert s.groups["hand.R"].shape == (5, 8, 3)
    assert s.groups["torso"].shape == (5, 3, 3)
    # The rigid hand's centre rides its bone: effector_mid carried by the posed matrix.
    mid = rig.effector_mid["hand.R"]
    for i in range(5):
        m = s.bones["hand.R"][i]
        np.testing.assert_allclose(s.groups["hand.R"][i].mean(axis=0),
                                   m[:3, :3] @ mid + m[:3, 3], rtol=0, atol=1e-5)

    # The prop: the bone's matrix, then the offset move, then the offset turn.
    c = np.cos(np.pi / 4)
    offset = np.array([[c, -c, 0, 0], [c, c, 0, 0.02], [0, 0, 1, 0], [0, 0, 0, 1]])
    for i in range(5):
        np.testing.assert_allclose(s.props["lamp"][i], s.bones["hand.R"][i] @ offset,
                                   rtol=0, atol=1e-6)

    # Pieces: triangles in the piece's own frame, glTF axes (x, z, -y).
    parts = ctx.pieces["shop:Bench"].parts
    assert set(parts) == {"seat_top", "leg_L"}
    for name, (lo, hi) in {
        "seat_top": ([-0.3, 0.43, -0.3], [0.3, 0.47, -0.1]),
        "leg_L": ([-0.3, 0.0, -0.02], [-0.2, 0.4, 0.02]),
    }.items():
        tris = parts[name]
        assert tris.shape == (12, 3, 3)
        flat = tris.reshape(-1, 3)
        np.testing.assert_allclose(flat.min(axis=0), lo, rtol=0, atol=1e-6)
        np.testing.assert_allclose(flat.max(axis=0), hi, rtol=0, atol=1e-6)
    lamp = ctx.pieces["shop:Lamp"].parts
    assert set(lamp) == {"stem"} and lamp["stem"].shape == (12, 3, 3)


def test_samples_hold_every_part_shell(round_trip):
    s = round_trip.ctx.samples["sit_reach"]
    parts = ["arm_R", "foot_R", "torso"]
    assert sorted(s.shells) == sorted(s.faces) == sorted(s.labels) == parts
    for part in parts:
        # 5 samples of the base mesh's 8 vertices (the outline shell is off); a cube's 6 quads
        # are 12 triangles, once.
        assert s.shells[part].shape == (5, 8, 3) and s.shells[part].dtype == np.float32
        assert s.faces[part].shape == (12, 3) and s.faces[part].dtype == np.int32
        assert s.labels[part].shape == (8,) and s.labels[part].dtype == np.int32
        assert 0 <= s.faces[part].min() and s.faces[part].max() < 8
    # The hand mesh is all of contact group hand.R, so its posed shell is that group's points.
    np.testing.assert_allclose(s.shells["arm_R"], s.groups["hand.R"], rtol=0, atol=1e-5)
    assert not np.allclose(s.shells["arm_R"][0], s.shells["arm_R"][-1])  # it moves with the clip


def test_labels_name_the_strongest_group(round_trip):
    ctx = round_trip.ctx
    rest = ctx.rests["shop:Doll"]
    # Per part, the mesh's deform groups in group order (the mask names no bone), and per
    # vertex the index of its strongest one: torso's first three lean to body, the rest to root.
    assert rest.vertex_groups == {"torso": ("body", "root"), "arm_R": ("hand.R",),
                                  "foot_R": ("root",)}
    np.testing.assert_array_equal(rest.labels["torso"], [0, 0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(rest.labels["arm_R"], np.zeros(8))
    np.testing.assert_array_equal(rest.labels["foot_R"], np.zeros(8))
    s = ctx.samples["sit_reach"]
    for part in rest.labels:
        np.testing.assert_array_equal(s.labels[part], rest.labels[part])
        np.testing.assert_array_equal(s.faces[part], rest.faces[part])
    raw = json.loads((round_trip.out / "rigs" / "shop" / "Doll.json").read_text())
    assert raw["vertex_groups"] == {"torso": ["body", "root"], "arm_R": ["hand.R"],
                                    "foot_R": ["root"]}


def test_rest_file_written_and_loaded(round_trip):
    path = round_trip.out / "rigs" / "shop" / "Doll.rest.npz"
    assert path.is_file()
    with np.load(path) as npz:
        assert sorted(npz.files) == sorted(f"{k}:{p}" for k in "vflwj"
                                           for p in ("arm_R", "foot_R", "torso"))
    ctx = round_trip.ctx
    rest = ctx.rests["shop:Doll"]  # keyed like rigs; the rest file is no stray under rigs/
    assert sorted(ctx.rigs) == sorted(ctx.rests) == ["shop:Doll"]
    s = ctx.samples["sit_reach"]
    for part, shell in rest.shells.items():
        assert shell.shape == (1, 8, 3) and shell.dtype == np.float32
        # Frame 1 is keyed at rest, and the clip moves the body away from it.
        np.testing.assert_allclose(shell[0], s.shells[part][0], rtol=0, atol=1e-5)
    assert not np.allclose(rest.shells["arm_R"][0], s.shells["arm_R"][-1])


def test_rest_weights_are_the_vertex_group_weights(round_trip):
    """`torso` leans body 0.9 / root 0.1 on its first three vertices and 0.4 / 0.6 on the rest;
    the mask group never enters; a one-group part pads nothing."""
    rest = round_trip.ctx.rests["shop:Doll"]
    assert rest.vertex_groups["torso"] == ("body", "root")
    want_w = np.array([[0.9, 0.1]] * 3 + [[0.4, 0.6]] * 5, dtype=np.float32)
    np.testing.assert_array_equal(rest.weights["torso"], want_w)
    np.testing.assert_array_equal(rest.weight_groups["torso"], [[0, 1]] * 8)
    for part in ("arm_R", "foot_R"):
        np.testing.assert_array_equal(rest.weights[part], np.ones((8, 1), np.float32))
        np.testing.assert_array_equal(rest.weight_groups[part], np.zeros((8, 1), np.int32))


_MID_CLIP = (
    "from pathlib import Path\n"
    "from model_kit.blender import export, runner\n"
    "from model_kit.blender._selftest_fk import build_rig, cube, keyed_clip, rig_pack\n"
    "out = Path(runner.script_args()[0])\n"
    "arm = build_rig()\n"
    "hand = cube('Hand', (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))\n"
    "hand.modifiers.new('Armature', 'ARMATURE').object = arm\n"
    "hand.vertex_groups.new(name='hand.R').add(range(8), 1.0, 'REPLACE')\n"
    "pack = rig_pack(contact_groups={{'hand.R': ['hand.R']}}, parts={{'Hand': 'hand.R'}})\n"
    "clip = keyed_clip('c')\n"
    "{keys}"
    "import bpy\n"
    "bpy.context.scene.frame_set(1)\n"
    "export.export_rig(arm, pack, 'Doll', out, clips=['c'])\n"
    "print('RIG_OK')\n"
    "{after_rig}"
    "export.export_samples(arm, arm.animation_data.action, pack, 'Doll', clip, out)\n"
    "runner.done()\n"
)


@pytest.mark.parametrize("keys, words", [
    pytest.param(
        "pb = arm.pose.bones['root']\n"
        "pb.keyframe_insert('scale', frame=1)\n"
        "pb.scale = (1.0, 1.3, 1.0)\n"
        "pb.keyframe_insert('scale', frame=2)\n",
        ["'root'", "pose scale", "at frame 1.5"], id="keyed-scale"),
    pytest.param(
        "c = arm.pose.bones['hand.R'].constraints.new('COPY_LOCATION')\n"
        "c.influence = 0.0\n"
        "c.keyframe_insert('influence', frame=1)\n"
        "c.influence = 1.0\n"
        "c.keyframe_insert('influence', frame=2)\n",
        ["'hand.R'", "active constraints", "at frame 1.5"], id="constraint-turns-on"),
    pytest.param(
        "pb = arm.pose.bones['body']\n"
        "pb.keyframe_insert('location', frame=1)\n"
        "pb.location = (0.0, 0.1, 0.0)\n"
        "pb.keyframe_insert('location', frame=2)\n",
        ["'body'", "connected", "at frame 1.5"], id="connected-bone-moves"),
    pytest.param(
        "arm.keyframe_insert('location', frame=1)\n"
        "arm.location.x += 0.5\n"
        "arm.keyframe_insert('location', frame=2)\n"
        "arm.pose.bones['root'].keyframe_insert('location', frame=1)\n",
        ["object moved", "at frame 1.5"], id="armature-object-moves"),
])
def test_export_samples_rechecks_the_fk_contract_at_every_sample(tmp_path, keys, words):
    """The rig exports cleanly at frame 1; the clip breaks the contract later on."""
    result = _run(tmp_path, _MID_CLIP.format(keys=keys, after_rig=""), str(tmp_path / "out"))
    assert "RIG_OK" in result.log, result.log
    assert result.ok is False
    for word in words:
        assert word in result.log, (word, result.log[-2000:])
    assert not (tmp_path / "out" / "samples").exists()


@pytest.mark.parametrize("after_rig, words", [
    pytest.param("arm.location.x += 0.5\n",
                 ["object moved", "rigs/shop/Doll.json", "at frame 1"],
                 id="placed-after-the-rig"),
    pytest.param("import shutil\nshutil.rmtree(out / 'rigs')\n",
                 ["no rig file", "'c'", "export_rig"], id="no-rig-file"),
    pytest.param("import json\nf = out / 'rigs' / 'shop' / 'Doll.json'\n"
                 "d = json.loads(f.read_text())\nd['clips'] = ['other']\n"
                 "f.write_text(json.dumps(d))\n",
                 ["'c'", "poses ['other']", "export_rig"], id="rig-file-poses-another-clip"),
])
def test_export_samples_checks_the_armature_against_the_rig_file(tmp_path, after_rig, words):
    """The rig file holds one world matrix; samples taken from another placement would be
    solved from the wrong one."""
    keys = "arm.pose.bones['root'].keyframe_insert('location', frame=1)\n"
    result = _run(tmp_path, _MID_CLIP.format(keys=keys, after_rig=after_rig),
                  str(tmp_path / "out"))
    assert "RIG_OK" in result.log, result.log
    assert result.ok is False
    for word in words:
        assert word in result.log, (word, result.log[-2000:])
    assert not (tmp_path / "out" / "samples").exists()


def test_export_rig_allows_a_muted_constraint(tmp_path):
    result = _run(tmp_path, (
        "from model_kit.blender import export, runner\n"
        "from model_kit.blender._selftest_fk import build_rig, rig_pack\n"
        "arm = build_rig()\n"
        "arm.pose.bones['hand.R'].constraints.new('COPY_LOCATION').mute = True\n"
        "export.export_rig(arm, rig_pack(), 'Doll', runner.script_args()[0], clips=[])\n"
        "runner.done()\n"
    ), str(tmp_path / "out"))
    assert result.ok, result.log


_ISOLATION_HEAD = """
from pathlib import Path
import bpy
import numpy as np
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, keyed_clip, rig_pack
from model_kit.blender.axes import BLENDER_TO_GLTF

out = Path(runner.script_args()[0])
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
pack = rig_pack(contact_groups={"hand.R": ["hand.R"]}, parts={"Hand": "hand.R"})
scene = bpy.context.scene


def rest_world(name):
    \"\"\"The bone's rest matrix in glTF world, from the armature data alone.\"\"\"
    local = np.array(arm.data.bones[name].matrix_local)
    return BLENDER_TO_GLTF @ np.array(arm.matrix_world) @ local


def evaluated():
    \"\"\"Every bone's evaluated pose matrix, after a fresh depsgraph update.\"\"\"
    bpy.context.view_layer.update()
    posed = arm.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return {pb.name: np.array(pb.matrix) for pb in posed.pose.bones}


def rest():
    for pb in arm.pose.bones:
        pb.location = (0.0, 0.0, 0.0)
        pb.rotation_euler = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)


def key_root():
    \"\"\"A new action that keys only `root` (a small slide), left active on `arm`.\"\"\"
    arm.animation_data_create().action = None
    rest()
    pb = arm.pose.bones["root"]
    pb.keyframe_insert("location", frame=1)
    pb.location = (0.0, 0.1, 0.0)
    pb.keyframe_insert("location", frame=2)
    pb.location = (0.0, 0.0, 0.0)
    return arm.animation_data.action


def samples(clip_name):
    with np.load(out / "samples" / f"{clip_name}.npz") as npz:
        return {k: npz[k] for k in npz.files}


def same(a, b):
    assert sorted(a) == sorted(b)
    return all(np.allclose(a[k], b[k], rtol=0, atol=1e-6) for k in a)

"""

_LEAK = _ISOLATION_HEAD + """
# Action A moves hand.R (location and a turn) away from rest.
arm.animation_data_create().action = None
rest()
hand_pb = arm.pose.bones["hand.R"]
for frame, x in ((1, 0.0), (2, 0.2)):
    hand_pb.location = (x, 0.0, 0.0)
    hand_pb.rotation_euler = (0.0, x, 0.0)
    hand_pb.keyframe_insert("location", frame=frame)
    hand_pb.keyframe_insert("rotation_euler", frame=frame)
action_a = arm.animation_data.action
action_b = key_root()  # keys only root
scene.frame_set(1)
names = ["a", "b_first", "b_after_a"]
export.export_rig(arm, pack, "Doll", out, clips=names)
export.export_samples(arm, action_b, pack, "Doll", keyed_clip("b_first"), out, step=1.0)
export.export_samples(arm, action_a, pack, "Doll", keyed_clip("a"), out, step=1.0)
export.export_samples(arm, action_b, pack, "Doll", keyed_clip("b_after_a"), out, step=1.0)
a = samples("a")
assert not np.allclose(a["b:hand.R"][0], a["b:hand.R"][1])  # A moves hand.R
b = samples("b_after_a")
# hand.R sits where the rig puts it, relative to root, at every sample.
want = np.linalg.inv(rest_world("root")) @ rest_world("hand.R")
at_rest = all(np.allclose(np.linalg.inv(r) @ h, want, rtol=0, atol=1e-5)
              for r, h in zip(b["b:root"], b["b:hand.R"], strict=True))
print("HAND AT REST", at_rest)
print("LEAK", "none" if same(samples("b_first"), b) else "FOUND")
runner.done()
"""


def test_samples_do_not_inherit_the_previous_action(tmp_path):
    """B keys only `root`; its hand.R must read rest, whatever clip A left behind."""
    result = _run(tmp_path, _LEAK, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "HAND AT REST True" in lines, result.log[-2000:]
    assert "LEAK none" in lines, result.log[-2000:]


_NLA = _ISOLATION_HEAD + """
# An unmuted NLA track slides hand.R; the clip's own action keys only root.
arm.animation_data_create().action = None
rest()
hand_pb = arm.pose.bones["hand.R"]
for frame, x in ((1, 0.3), (2, 0.3)):
    hand_pb.location = (x, 0.0, 0.0)
    hand_pb.keyframe_insert("location", frame=frame)
offset = arm.animation_data.action
arm.animation_data.action = None
action = key_root()
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, action, pack, "Doll", keyed_clip("plain"), out, step=1.0)
data = arm.animation_data
data.action = None
track = data.nla_tracks.new()
track.strips.new("offset", 1, offset)
data.action = action
assert track.mute is False
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
print("NLA MUTE AFTER", track.mute)
print("NLA", "ignored" if same(samples("plain"), samples("c")) else "PLAYED")
runner.done()
"""


def test_nla_tracks_do_not_play_while_sampling(tmp_path):
    result = _run(tmp_path, _NLA, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "NLA ignored" in lines, result.log[-2000:]
    assert "NLA MUTE AFTER False" in lines


_RESTORE = _ISOLATION_HEAD + """
# The active action keys only root; the clip to sample is another action that moves hand.R.
arm.animation_data_create().action = None
rest()
hand_pb = arm.pose.bones["hand.R"]
for frame, x in ((1, 0.0), (2, 0.2)):
    hand_pb.location = (x, 0.0, 0.0)
    hand_pb.rotation_euler = (0.0, x, 0.0)
    hand_pb.keyframe_insert("location", frame=frame)
    hand_pb.keyframe_insert("rotation_euler", frame=frame)
clip_action = arm.animation_data.action
active = key_root()
data = arm.animation_data
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["c"])
data.nla_tracks.new().mute = True
data.nla_tracks.new().is_solo = True
scene.frame_set(1, subframe=0.25)
hand_pb.location = (0.3, 0.0, 0.0)  # a hand-set pose, not animated by the active action
hand_pb.rotation_euler = (0.1, 0.2, 0.3)


def state():
    return {
        "action": data.action.name, "slot": data.action_slot.identifier,
        "frame": (scene.frame_current, scene.frame_subframe),
        "tracks": [(t.mute, t.is_solo) for t in data.nla_tracks],
        "pose": {pb.name: (tuple(pb.location), tuple(pb.rotation_euler), tuple(pb.scale))
                 for pb in arm.pose.bones},
    }


before, before_matrix = state(), evaluated()
export.export_samples(arm, clip_action, pack, "Doll", keyed_clip("c"), out, step=1.0)
after, after_matrix = state(), evaluated()
same_matrix = all(np.allclose(before_matrix[k], after_matrix[k], rtol=0, atol=1e-6)
                  for k in before_matrix)
print("EVALUATED SAME", same_matrix)
print("RESTORED", before == after)
if before != after:
    print("BEFORE", before)
    print("AFTER", after)
runner.done()
"""


def test_pose_and_action_restored_after_export_samples(tmp_path):
    result = _run(tmp_path, _RESTORE, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "RESTORED True" in lines, result.log[-2000:]
    assert "EVALUATED SAME True" in lines, result.log[-2000:]


_SOLO = _ISOLATION_HEAD + """
# Two NLA tracks offset hand.R (x 0.5, then 0.9); the first one is solo, the last is not.
def offset_action(x):
    arm.animation_data_create().action = None
    rest()
    pb = arm.pose.bones["hand.R"]
    pb.location = (x, 0.0, 0.0)
    pb.keyframe_insert("location", frame=1)
    made = arm.animation_data.action
    arm.animation_data.action = None
    pb.location = (0.0, 0.0, 0.0)
    return made


low, high = offset_action(0.5), offset_action(0.9)
action = key_root()
data = arm.animation_data
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["c"])
first = data.nla_tracks.new()
first.strips.new("low", 1, low)
last = data.nla_tracks.new()
last.strips.new("high", 1, high)
first.is_solo = True
last.mute = False
scene.frame_set(1)


def flags():
    return [(t.mute, t.is_solo) for t in data.nla_tracks]


data.use_nla = False
rest()
silent = evaluated()  # the same scene with the NLA off
data.use_nla = True
scene.frame_set(1)
before, before_matrix = flags(), evaluated()
print("NLA PLAYS", not np.allclose(before_matrix["hand.R"], silent["hand.R"], atol=1e-4))
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
scene.frame_set(2)
scene.frame_set(1)
after, after_matrix = flags(), evaluated()
print("FLAGS", before, after)
print("FLAGS SAME", before == after)
print("MATRIX SAME", all(np.allclose(before_matrix[k], after_matrix[k], rtol=0, atol=1e-6)
                          for k in before_matrix))
runner.done()
"""


def test_a_solo_nla_track_that_is_not_the_last_comes_back_as_it_was(tmp_path):
    result = _run(tmp_path, _SOLO, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "FLAGS SAME True" in lines, result.log[-2000:]
    assert "MATRIX SAME True" in lines, result.log[-2000:]
    assert "NLA PLAYS True" in lines, result.log[-2000:]  # the test would be vacuous otherwise


_SCALED = """
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, keyed_clip, rig_pack
from model_kit.core.manifest import Prop

out = Path(runner.script_args()[0])
TIP = ("prop.R", (0.5, 0.1, 1.1), (0.6, 0.1, 1.1), 0.0, "hand.R", False)  # no child bone
arm = build_rig(extra=(TIP,))
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
props = {}
# SETUP
pack = rig_pack(extra=(TIP,), parts={"Hand": "hand.R"})
arm.pose.bones["# BONE"].scale = (1e-4, 1e-4, 1e-4)
arm.pose.bones["# BONE"].keyframe_insert("scale", frame=1)  # samples start from scale 1
arm.pose.bones["root"].keyframe_insert("location", frame=1)
bpy.context.scene.frame_set(1)
clip = keyed_clip("c", props=props)
export.export_rig(arm, pack, "Doll", out, clips=["c"])
print("RIG_OK")
export.export_samples(arm, arm.animation_data.action, pack, "Doll", clip, out)
print("SAMPLES_OK")
runner.done()
"""

_LAMP_ON_TIP = (
    "props = {'lamp': Prop(id='lamp', piece='Lamp', pack='shop', bone='prop.R',\n"
    "                      offset_position=(0.0, 0.0, 0.0),\n"
    "                      offset_quaternion=(0.0, 0.0, 0.0, 1.0),\n"
    "                      show_frames=((1.0, 2.0),))}\n")


def _scaled(tmp_path, bone: str, setup: str = ""):
    script = _SCALED.replace("# SETUP", setup).replace("# BONE", bone)
    return _run(tmp_path, script, str(tmp_path / "out"))


def test_inert_scaled_bone_is_allowed_and_logged(tmp_path):
    """prop.R has no child bone, no skinned weight and no clip prop: its 1e-4 scale passes."""
    result = _scaled(tmp_path, "prop.R")
    assert result.ok, result.log
    assert "SAMPLES_OK" in result.log
    lines = [line for line in result.log.splitlines() if line.startswith("inert scaled bone")]
    assert lines and all("'prop.R'" in line for line in lines), result.log
    assert len(lines) == 1  # export_samples only: export_rig reads no pose value
    with np.load(tmp_path / "out" / "samples" / "c.npz") as npz:
        assert "b:prop.R" in npz.files


def test_scaled_bone_with_a_child_raises_in_samples(tmp_path):
    result = _scaled(tmp_path, "body")  # hand.R is its child
    assert "RIG_OK" in result.log  # export_rig does not read the pose scale
    assert result.ok is False and "SAMPLES_OK" not in result.log
    for word in ("'body'", "pose scale", "child bone", "'hand.R'", "at frame 1"):
        assert word in result.log, (word, result.log[-1500:])
    assert not (tmp_path / "out" / "samples").exists()


def test_scaled_bone_with_a_skinned_weight_raises_in_samples(tmp_path):
    # hand.R wins the vertex (1.0 against 0.1), but prop.R still weighs it. A
    # file may open on a pose that scales prop.R; export_rig no longer reads that pose.
    result = _scaled(tmp_path, "prop.R",
                     "hand.vertex_groups.new(name='prop.R').add([0], 0.1, 'REPLACE')\n")
    assert "RIG_OK" in result.log
    assert result.ok is False and "SAMPLES_OK" not in result.log
    for word in ("'prop.R'", "pose scale", "skinned weight", "at frame 1"):
        assert word in result.log, (word, result.log[-1500:])
    assert not (tmp_path / "out" / "samples").exists()


def test_scaled_bone_that_a_clip_prop_rides_raises_in_samples(tmp_path):
    result = _scaled(tmp_path, "prop.R", _LAMP_ON_TIP)
    assert "RIG_OK" in result.log  # the rig file cannot know the clip's props
    assert result.ok is False and "SAMPLES_OK" not in result.log
    for word in ("'prop.R'", "pose scale", "prop 'lamp'"):
        assert word in result.log, (word, result.log[-1500:])
    assert not (tmp_path / "out" / "samples").exists()


UNWEIGHTED = """
from pathlib import Path
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, rig_pack

out = Path(runner.script_args()[0])
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(7), 1.0, "REPLACE")  # vertex 7 has no weight
export.export_rig(arm, rig_pack(parts={"Hand": "hand.R"}), "Doll", out, clips=[])
runner.done()
"""


def test_shell_vertex_without_a_deform_weight_raises(tmp_path):
    result = _run(tmp_path, UNWEIGHTED, str(tmp_path / "out"))
    assert result.ok is False
    for word in ("part 'Hand'", "1 of 8", "no deform weight"):
        assert word in result.log, (word, result.log[-1500:])
    assert not (tmp_path / "out" / "rigs").exists()


REFUSALS = """
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import (
    build_rig, cube, hand_chain, keyed_clip, plain_pack, rig_pack)
from model_kit.core.errors import ManifestError

out = Path(runner.script_args()[0])
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
arm.pose.bones["root"].keyframe_insert("location", frame=1)
action = arm.animation_data.action
clip = keyed_clip("c")

bench = bpy.data.objects.new("Bench", None)
bpy.context.scene.collection.objects.link(bench)
for name in ("seat_top", "ghost_leg"):
    cube(name, (0, 0, 0), (0.1, 0.1, 0.1)).parent = bench


FINGERS = ("fingers.R", (0.5, 0.1, 1.1), (0.6, 0.1, 1.1), 0.0, "hand.R", False)
MOVED = (FINGERS[0], *FINGERS[1:4], "body", FINGERS[5])  # the same bone under another parent
arm2 = build_rig("Rig2", extra=(FINGERS,))


def not_deform():
    arm.data.bones["hand.R"].use_deform = False
    try:
        export.export_rig(arm, pack_of(contact_groups=groups), "Doll", out, clips=[])
    finally:
        arm.data.bones["hand.R"].use_deform = True


def pack_of(parts=None, **kwargs):
    return plain_pack("shop", {"Bench": ["seat_top"], "Lamp": ["stem"], "Box": ["part"]},
                      base=rig_pack(parts=parts or {"Hand": "hand.R"}, **kwargs))


plain = pack_of()
groups = {"hand.R": ["hand.R"]}
cases = {
    "rig_piece_missing": lambda: export.export_rig(arm, plain, "Ghost", out, clips=[]),
    "rig_piece_unrigged": lambda: export.export_rig(arm, plain, "Lamp", out, clips=[]),
    "rig_armature_name": lambda: export.export_rig(
        arm, pack_of(armature="Other"), "Doll", out, clips=[]),
    "rig_clips_repeat": lambda: export.export_rig(arm, plain, "Doll", out, clips=["c", "c"]),
    "rig_group_no_vertices": lambda: export.export_rig(
        arm, pack_of(contact_groups={"torso": ["body"]}), "Doll", out, clips=[]),
    "rig_group_not_deform": not_deform,
    "rig_bones_only_in_armature": lambda: export.export_rig(
        arm2, pack_of(armature="Rig2"), "Doll", out, clips=[]),
    "rig_bones_only_in_pack": lambda: export.export_rig(
        arm, pack_of(extra=(FINGERS,)), "Doll", out, clips=[]),
    "rig_hide_typo": lambda: export.export_rig(
        arm, pack_of(outline_modifiers=["Outlnie"]), "Doll", out, clips=[]),
    "rig_bone_moved": lambda: export.export_rig(
        arm2, pack_of(armature="Rig2", extra=(MOVED,)), "Doll", out, clips=[]),
    "rig_part_without_mesh": lambda: export.export_rig(
        arm, pack_of(parts={"Hand": "hand.R", "Ghost": "body"}), "Doll", out, clips=[]),
    "rig_mesh_not_a_part": lambda: export.export_rig(
        arm, pack_of(parts={"Palm": "hand.R"}), "Doll", out, clips=[]),
    "rig_chains_share_an_effector": lambda: export.export_rig(
        arm, pack_of(contact_groups={"a": ["hand.R"], "b": ["body"]},
                     chains={"one": hand_chain("a"), "two": hand_chain("b")}),
        "Doll", out, clips=[]),
    "rig_chain_group_missing": lambda: export.export_rig(
        arm, pack_of(chains={"arm.R": hand_chain("hand.R")}), "Doll", out, clips=[]),
    "samples_piece_missing": lambda: export.export_samples(
        arm, action, plain, "Ghost", clip, out),
    "samples_piece_unrigged": lambda: export.export_samples(
        arm, action, plain, "Lamp", clip, out),
    "samples_armature_name": lambda: export.export_samples(
        arm, action, pack_of(armature="Other"), "Doll", clip, out),
    "samples_group_no_vertices": lambda: export.export_samples(
        arm, action, pack_of(contact_groups={"torso": ["body"]}), "Doll", clip, out),
    "samples_part_without_mesh": lambda: export.export_samples(
        arm, action, pack_of(parts={"Hand": "hand.R", "Ghost": "body"}), "Doll", clip, out),
    "samples_mesh_not_a_part": lambda: export.export_samples(
        arm, action, pack_of(parts={"Palm": "hand.R"}), "Doll", clip, out),
    "pieces_piece_missing": lambda: export.export_pieces(plain, out, pieces=["Ghost"]),
    "pieces_repeat": lambda: export.export_pieces(plain, out, pieces=["Box", "Box"]),
    "pieces_second_bad": lambda: export.export_pieces(plain, out, pieces=["Box", "Bench"]),
    "pieces_both_problems": lambda: export.export_pieces(
        plain_pack("shop", {"Box": ["part"], "Bench": ["seat_top", "leg_L"]}), out,
        pieces=["Box", "Bench"]),
    "pieces_no_object": lambda: export.export_pieces(plain, out, pieces=["Lamp"]),
    "pieces_no_armature": lambda: export.export_pieces(
        pack_of(armature="Nope"), out, pieces=["Doll"]),
    "pieces_hide_typo": lambda: export.export_pieces(
        plain, out, pieces=["Box"], hide_modifiers=["Outlnie"]),
    "pieces_unlisted_mesh": lambda: export.export_pieces(plain, out, pieces=["Bench"]),
    "pieces_part_with_no_mesh": lambda: export.export_pieces(
        plain_pack("shop", {"Bench": ["seat_top", "ghost_leg", "leg_L"]}), out,
        pieces=["Bench"]),
}
box = bpy.data.objects.new("Box", None)
bpy.context.scene.collection.objects.link(box)
cube("part", (0, 0, 0), (0.1, 0.1, 0.1)).parent = box
for label, call in cases.items():
    try:
        call()
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, "|", exc)
print("WRITTEN", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
      if out.exists() else [])
runner.done()
"""


@pytest.fixture(scope="module")
def refusals(tmp_path_factory) -> dict[str, str]:
    """One Blender run tries every refusal; each case's message by label."""
    tmp_path = tmp_path_factory.mktemp("refusals")
    result = _run(tmp_path, REFUSALS, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "WRITTEN []" in lines, "an exporter wrote a file before it raised"
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    return {line.split(" ", 2)[1]: line.partition("|")[2]
            for line in lines if line.startswith("RAISED")}


def test_every_refusal_raises_and_writes_nothing(refusals):
    assert len(refusals) == 29
    assert "'Ghost'" in refusals["rig_piece_missing"]
    assert "has no rig" in refusals["rig_piece_unrigged"]
    assert "Lamp" in refusals["rig_piece_unrigged"]
    assert "repeat" in refusals["rig_clips_repeat"]
    assert "'torso'" in refusals["rig_group_no_vertices"]
    assert "'body'" in refusals["rig_group_no_vertices"]
    assert "no vertices" in refusals["rig_group_no_vertices"]
    assert "'hand.R'" in refusals["rig_group_not_deform"]
    assert "not a deform bone" in refusals["rig_group_not_deform"]
    assert "Outlnie" in refusals["rig_hide_typo"]
    assert "names no modifier" in refusals["rig_hide_typo"]
    assert "'Ghost'" in refusals["samples_piece_missing"]
    assert "has no rig" in refusals["samples_piece_unrigged"]
    assert "'body'" in refusals["samples_group_no_vertices"]
    assert "no vertices" in refusals["samples_group_no_vertices"]
    assert "'Ghost'" in refusals["pieces_piece_missing"]
    assert "pieces repeat a name" in refusals["pieces_repeat"]
    assert "'Box'" in refusals["pieces_repeat"]
    assert "'Lamp'" in refusals["pieces_no_object"]
    assert "'Nope'" in refusals["pieces_no_armature"]
    assert "Outlnie" in refusals["pieces_hide_typo"]


def test_shell_part_without_mesh_raises(refusals):
    for label in ("rig_part_without_mesh", "samples_part_without_mesh"):
        assert "part 'Ghost' has no mesh" in refusals[label], label
    for label in ("rig_mesh_not_a_part", "samples_mesh_not_a_part"):
        message = refusals[label]
        assert "mesh 'Hand' is part 'Hand', which the pack does not list" in message, label
        assert "part 'Palm' has no mesh" in message, label


def test_export_rig_refuses_wrong_armature_name(refusals):
    for label in ("rig_armature_name", "samples_armature_name"):
        assert "'Rig'" in refusals[label] and "'Other'" in refusals[label], label


def test_export_rig_refuses_bones_unlike_the_pack(refusals):
    assert "only in the armature ['fingers.R']" in refusals["rig_bones_only_in_armature"]
    assert "only in the pack ['fingers.R']" in refusals["rig_bones_only_in_pack"]


def test_export_rig_refuses_a_bone_under_another_parent(refusals):
    assert "with another parent ['fingers.R']" in refusals["rig_bone_moved"]


def test_export_rig_refuses_chains_that_disagree_or_miss_their_group(refusals):
    message = refusals["rig_chains_share_an_effector"]
    assert "'two'" in message and "both end at 'hand.R'" in message
    assert "different contact groups" in message
    message = refusals["rig_chain_group_missing"]
    assert "chain 'arm.R'" in message and "'hand.R'" in message
    assert "no contact group" in message


def test_export_pieces_names_every_problem_and_writes_no_piece(refusals):
    # The first piece is fine and the second is not: neither is written (the run checks that).
    assert "'ghost_leg'" in refusals["pieces_second_bad"]
    assert "'Box'" not in refusals["pieces_second_bad"]
    both = refusals["pieces_both_problems"]
    assert "mesh 'ghost_leg' is part 'ghost_leg', which the pack does not list" in both
    assert "part 'leg_L' has no mesh" in both


ROOT_MESH = """
from pathlib import Path
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import cube, plain_pack

out = Path(runner.script_args()[0])
box = cube("Box", (0.0, 0.0, 0.0), (0.1, 0.1, 0.1))
cube("lid", (0.0, 0.0, 0.2), (0.1, 0.1, 0.05)).parent = box
paths = export.export_pieces(plain_pack("shop", {"Box": ["Box", "lid"]}), out, pieces=["Box"])
print("PATHS", [p.relative_to(out).as_posix() for p in paths])
runner.done()
"""


def test_export_pieces_takes_a_mesh_root_as_a_part(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, ROOT_MESH, str(out))
    assert result.ok, result.log
    assert "PATHS ['pieces/shop/Box.npz']" in result.log.splitlines()
    with np.load(out / "pieces" / "shop" / "Box.npz") as npz:
        assert sorted(npz.files) == ["Box", "lid"]  # the root itself and its child


def test_export_pieces_refuses_an_unlisted_mesh(refusals):
    message = refusals["pieces_unlisted_mesh"]
    assert "'ghost_leg'" in message and "which the pack does not list" in message


def test_export_pieces_refuses_a_part_with_no_mesh(refusals):
    message = refusals["pieces_part_with_no_mesh"]
    assert "part 'leg_L' has no mesh" in message


HAND_MID = """
import json
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, hand_chain, rig_pack

out, side = (Path(p) for p in runner.script_args())
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.matrix_world = arm.matrix_world.copy()
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
# A second mesh in another group: its vertices must not move the hand's centre.
other = cube("Other", (-0.5, 0.0, 0.3), (0.05, 0.05, 0.05))
other.matrix_world = arm.matrix_world.copy()
other.modifiers.new("Armature", "ARMATURE").object = arm
other.vertex_groups.new(name="body").add(range(8), 1.0, "REPLACE")
pack = rig_pack(contact_groups={"hand.R": ["hand.R"]}, chains={"arm.R": hand_chain("hand.R")},
                parts={"Hand": "hand.R", "Other": "body"})
path = export.export_rig(arm, pack, "Doll", out, clips=[])
print("PATH", path.relative_to(out))
mean = sum((hand.matrix_world @ v.co for v in hand.data.vertices), start=hand.location * 0) / 8
side.write_text(json.dumps({"centre": list(mean)}))
runner.done()
"""


def test_export_rig_reads_the_pack(tmp_path):
    out, side = tmp_path / "out", tmp_path / "side.json"
    result = _run(tmp_path, HAND_MID, str(out), str(side))
    assert result.ok, result.log
    assert "PATH rigs/shop/Doll.json" in result.log.splitlines()
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "rigs/shop/Doll.json", "rigs/shop/Doll.rest.npz"]
    rig, clips = _read_rig(out / "rigs" / "shop" / "Doll.json")
    assert clips == [] and list(rig.effector_mid) == ["hand.R"]
    # The effector point, carried by the rest bone, lands on the centre of the contact group.
    bone = pose_world(rig, {})["hand.R"]
    got = (bone @ np.append(rig.effector_mid["hand.R"], 1.0))[:3]
    want = (BLENDER_TO_GLTF @ np.append(json.loads(side.read_text())["centre"], 1.0))[:3]
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-5)


TWO_GROUPS = """
import json
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, hand_chain, keyed_clip, rig_pack

out, side = (Path(p) for p in runner.script_args())
fingers_bone = ("fingers.R", (0.5, 0.1, 1.1), (0.6, 0.1, 1.1), 0.0, "hand.R", False)
arm = build_rig(extra=(fingers_bone,))
expected = {}
for name, centre, group in (("Hand", (0.35, 0.1, 1.05), "hand.R"),
                            ("Fingers", (0.55, 0.1, 1.1), "fingers.R")):
    obj = cube(name, centre, (0.04, 0.04, 0.04))
    obj.matrix_world = arm.matrix_world.copy()
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    obj.vertex_groups.new(name=group).add(range(8), 1.0, "REPLACE")
    expected[group] = [list(obj.matrix_world @ v.co) for v in obj.data.vertices]
pack = rig_pack(extra=(fingers_bone,), chains={"arm.R": hand_chain("hand.R")},
                contact_groups={"hand.R": ["hand.R", "fingers.R"]},
                parts={"Hand": "hand.R", "Fingers": "fingers.R"})
arm.pose.bones["root"].keyframe_insert("location", frame=1)
bpy.context.scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["c"])
export.export_samples(arm, arm.animation_data.action, pack, "Doll", keyed_clip("c"), out)
side.write_text(json.dumps(expected))
runner.done()
"""


def test_contact_group_of_two_vertex_groups(tmp_path):
    out, side = tmp_path / "out", tmp_path / "side.json"
    result = _run(tmp_path, TWO_GROUPS, str(out), str(side))
    assert result.ok, result.log
    expected = json.loads(side.read_text())
    both = np.array(expected["hand.R"] + expected["fingers.R"])
    want = (BLENDER_TO_GLTF @ np.c_[both, np.ones(len(both))].T).T[:, :3]
    with np.load(out / "samples" / "c.npz") as npz:
        got = npz["g:hand.R"]
    assert got.shape == (3, 16, 3)  # frames 1, 1.5, 2: the points of both vertex groups
    for sample in got:
        np.testing.assert_allclose(sample[np.lexsort(sample.T)], want[np.lexsort(want.T)],
                                   rtol=0, atol=1e-5)
    # The effector point is the centre of both groups, carried by the rest bone.
    rig, _ = _read_rig(out / "rigs" / "shop" / "Doll.json")
    bone = pose_world(rig, {})["hand.R"]
    mid = (bone @ np.append(rig.effector_mid["hand.R"], 1.0))[:3]
    np.testing.assert_allclose(mid, want.mean(axis=0), rtol=0, atol=1e-5)
    only_hand = np.array(expected["hand.R"]).mean(axis=0)
    assert not np.allclose((BLENDER_TO_GLTF @ np.append(only_hand, 1.0))[:3], mid, atol=1e-3)


RIGGED_ROOT = """
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, rig_pack

out = Path(runner.script_args()[0])
arm = build_rig()
for name, centre, bone in (("torso", (0.0, 0.05, 0.7), "body"), ("hand", (0.35, 0.1, 1.05),
                                                                  "hand.R")):
    obj = cube(name, centre, (0.05, 0.05, 0.05))
    obj.matrix_world = arm.matrix_world.copy()
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    obj.vertex_groups.new(name=bone).add(range(8), 1.0, "REPLACE")
# A mesh parented to the armature but not skinned: it is still a part of the piece.
hat = cube("hat", (0.0, 0.0, 0.0), (0.05, 0.05, 0.05))
hat.parent = arm
hat.location = (0.0, 0.0, 1.5)
# The outline shell is the rig's, so the export leaves it out of the bounds.
shell = bpy.data.objects["torso"].modifiers.new("Outline", "SOLIDIFY")
shell.thickness = 0.5
pack = rig_pack(parts={"torso": "body", "hand": "hand.R", "hat": "body"},
                outline_modifiers=["Outline"])
print("NO DOLL OBJECT", "Doll" not in bpy.data.objects)
paths = export.export_pieces(pack, out, pieces=["Doll"])
print("PATHS", [p.relative_to(out).as_posix() for p in paths])
print("OUTLINE BACK ON", shell.show_viewport)
runner.done()
"""


def test_export_pieces_rigged_piece_root_is_the_armature(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, RIGGED_ROOT, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "NO DOLL OBJECT True" in lines and "OUTLINE BACK ON True" in lines
    assert "PATHS ['pieces/shop/Doll.npz']" in lines
    with np.load(out / "pieces" / "shop" / "Doll.npz") as npz:
        parts = {name: npz[name] for name in npz.files}
    assert sorted(parts) == ["hand", "hat", "torso"]  # skinned meshes and the armature's child
    # In the armature's own frame (scale and turn undone), in glTF axes: (x, z, -y).
    for name, centre in {"hand": (0.35, 1.05, -0.1), "hat": (0.0, 1.5, 0.0),
                         "torso": (0.0, 0.7, -0.05)}.items():
        tris = parts[name]
        assert tris.shape == (12, 3, 3)
        flat = tris.reshape(-1, 3)
        np.testing.assert_allclose((flat.min(axis=0) + flat.max(axis=0)) / 2, centre,
                                   rtol=0, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(flat.max(axis=0) - flat.min(axis=0), 0.1, rtol=0, atol=1e-5)


CLIP_FRAMES = """
from pathlib import Path
import bpy
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, keyed_clip, rig_pack
from model_kit.core.manifest import Prop

out = Path(runner.script_args()[0])
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.modifiers.new("Armature", "ARMATURE").object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
torso = cube("Torso", (0.0, 0.05, 0.7), (0.05, 0.05, 0.05))
torso.modifiers.new("Armature", "ARMATURE").object = arm
torso.vertex_groups.new(name="body").add(range(8), 1.0, "REPLACE")
# The action is keyed over frames 1 to 6; the clip covers only 2 to 4.
for frame, x in ((1, 0.0), (6, 0.5)):
    pb = arm.pose.bones["hand.R"]
    pb.location = (x, 0.0, 0.0)
    pb.keyframe_insert("location", frame=frame)
action = arm.animation_data.action
action.name = "walk_action"
bpy.context.scene.frame_set(1)
lamp = Prop(id="lamp", piece="Lamp", pack="shop", bone="hand.R", offset_position=(0.0, 0.02, 0.0),
            offset_quaternion=(0.0, 0.0, 0.0, 1.0), show_frames=((2.0, 4.0),))
cup = Prop(id="cup", piece="Cup", pack="shop", bone="body", offset_position=(0.0, 0.0, 0.1),
           offset_quaternion=(0.0, 0.0, 0.0, 1.0), show_frames=((2.0, 4.0),))
pack = rig_pack(contact_groups={"hand.R": ["hand.R"], "torso": ["body"]},
                parts={"Hand": "hand.R", "Torso": "body"})
clip = keyed_clip("walk", frames=(2, 4), props={"lamp": lamp, "cup": cup})
export.export_rig(arm, pack, "Doll", out, clips=["walk"])
path = export.export_samples(arm, action, pack, "Doll", clip, out, step=1.0)
print("PATH", path.relative_to(out))
print("FRAME BACK", bpy.context.scene.frame_current)
runner.done()
"""


def test_export_samples_uses_clip_frames_and_props(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, CLIP_FRAMES, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "PATH samples/walk.npz" in lines and "FRAME BACK 1" in lines  # the clip's name
    with np.load(out / "samples" / "walk.npz") as npz:
        arrays = {k: npz[k] for k in npz.files}
    np.testing.assert_array_equal(arrays["times"], [2.0, 3.0, 4.0])  # the clip's frames
    # Every contact group of the rig, and every prop of the clip.
    assert {k for k in arrays if k[:2] in ("g:", "p:")} == {
        "g:hand.R", "g:torso", "p:lamp", "p:cup"}
    assert arrays["g:hand.R"].shape == (3, 8, 3) and arrays["g:torso"].shape == (3, 8, 3)
    for i in range(3):
        offset = np.eye(4)
        offset[:3, 3] = (0.0, 0.02, 0.0)
        np.testing.assert_allclose(arrays["p:lamp"][i], arrays["b:hand.R"][i] @ offset,
                                   rtol=0, atol=1e-6)
    assert not np.allclose(arrays["b:hand.R"][0], arrays["b:hand.R"][2])  # the pose moved


SEAM = """
import dataclasses
from pathlib import Path
import bpy
from model_kit.blender import build, export, runner
from model_kit.blender._selftest_fk import cube
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Listed, Recipe, View, load_pack
from model_kit.core.units import Style

build_dir, scene_dir = (Path(p) for p in runner.script_args())
build.reset()
fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
pack = build.PackBuilder("shop", 1.0, build_dir, 2.0, crease_deg=35)
bench = build.piece("Bench")
build.slab("leg_L", -0.28, -0.22, 0, 0.42, -0.12, 0.12, fill, bench,
           props={"listed": "pine", "group": "legs"})
build.slab("seat_top", -0.3, 0.3, 0.42, 0.46, -0.15, 0.15, fill, bench,
           props={"listed": "pine", "group": "top", "on": "leg_L"})
pack.register(bench, view=View(1.0, "a test piece, seen from arm's length"), origin="floor, under the seat centre", recipe=Recipe("assembly", "x.py"),
              listed=[Listed("pine", "Pine board, 40 mm", "a catalogue", 0.04, None)],
              done="A bench.", size_class="piece", kind="furniture")
pack.use(bench, "front", point=(0, 0, 0.5), facing=(0, 0, -1), posture="sit")
pack.hold(bench, "edge_R", use="front", side="R", surface=["seat_top"], point=(0.2, 0.46, 0.0),
          grip=False)
pack.finish(["Bench"])
print("OBJECTS", sorted(ob.name for ob in bench.children_recursive))
style = Style(units_per_metre=1.0, page_camera={}, dials={})
shop = load_pack(build_dir / "shop.pack.json", style)
export.export_pieces(shop, scene_dir, pieces=["Bench"])

# Two objects that map to one part name are refused, and nothing is written for them.
box = bpy.data.objects.new("Box", None)
bpy.context.scene.collection.objects.link(box)
for name in ("Box/body", "body"):
    cube(name, (0, 0, 0), (0.1, 0.1, 0.1)).parent = box
clash = dataclasses.replace(shop, name="clash", pieces={"Box": dataclasses.replace(
    shop.pieces["Bench"], name="Box", holds=(), seats=(),
    parts=(dataclasses.replace(shop.pieces["Bench"].parts[0], name="body"),))})
try:
    export.export_pieces(clash, scene_dir, pieces=["Box"])
    print("NO RAISE clash")
except ManifestError as exc:
    print("RAISED clash", exc)
runner.done()
"""

DOLL_PACK = {
    "pack": "kit", "units_per_metre": 1.0, "pieces": {"Doll": {
        "frame": {"axes": "gltf", "origin": "between the feet"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "skin", "text": "Skin", "source": "test", "min_thickness": 0.002}],
        "parts": [{"name": "body", "listed": "skin", "shape": "box", "on": None, "group": "g",
                   "role": "solid"}],
        "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A doll.",
        "size_class": "piece", "measured": {},
        "rig": {
            "armature": "DollRig",
            "bones": [{"name": "body", "parent": None, "length": 0.3},
                      {"name": "hand.R", "parent": "body", "length": 0.05}],
            "bind": {"body": {"bones": ["body"], "mode": "rigid"}},
            "contact_groups": {"hand.R": ["hand.R"]},
            "chains": {"arm.R": {"root": "body", "bones": ["hand.R"], "effector": "hand.R",
                                 "effector_radius": 0.0586, "dofs": {"hand.R": "xyz"},
                                 "limits": {}, "effector_group": "hand.R"}},
            "sit": {"below_seat": 0.08, "forward": 0.03},
            "mirror": {"pairs": [], "axis": "x"},
        },
    }},
}


def _seam_scene(scene: Path, build_dir: Path, lift: float) -> None:
    """Around the exported pieces: style, both packs, one keyed clip holding the bench's
    hold, a rig and samples with the hand `lift` over the seat top."""
    (scene / "style.json").write_text(json.dumps(
        {"units_per_metre": 1.0, "page_camera": {}, "dials": {}}))
    (scene / "packs").mkdir()
    shutil.copy(build_dir / "shop.pack.json", scene / "packs" / "shop.pack.json")
    (scene / "packs" / "kit.pack.json").write_text(json.dumps(DOLL_PACK))
    (scene / "clips.json").write_text(json.dumps({"clips": {"touch": {
        "frames": [1, 10], "fps": 24, "authoring": "keyed",
        "floor_gear": {"Bench": {"piece": "Bench", "pack": "shop", "position": [0, 0, 0],
                                 "quaternion": [0, 0, 0, 1]}},
        "holds": [{"ref": "shop:Bench.edge_R", "who": "hand.R", "when": [[1, 10]]}],
    }}}))
    (scene / "rigs" / "kit").mkdir(parents=True)
    eye = np.eye(4).tolist()
    (scene / "rigs" / "kit" / "Doll.json").write_text(json.dumps({
        "order": ["body", "hand.R"], "parent": {"body": None, "hand.R": "body"},
        "rest": {"body": eye, "hand.R": eye}, "world": eye,
        "effector_mid": {"hand.R": [0.0, 0.0, 0.0]}, "clips": ["touch"]}))
    (scene / "samples").mkdir()
    times = np.arange(1.0, 11.0)
    hand = np.tile([[0.2, 0.46 + lift, 0.0]], (len(times), 1, 1))
    np.savez(scene / "samples" / "touch.npz", times=times, **{"g:hand.R": hand})


def test_a_packbuilder_piece_exports_loads_and_is_touched(tmp_path):
    build_dir, scene = tmp_path / "build", tmp_path / "scene"
    result = _run(tmp_path, SEAM, str(build_dir), str(scene))
    assert result.ok, result.log
    assert "OBJECTS ['Bench/leg_L', 'Bench/seat_top']" in result.log.splitlines()
    clash = next(line for line in result.log.splitlines() if line.startswith(("RAISED clash",
                                                                              "NO RAISE clash")))
    assert clash.startswith("RAISED clash") and "'body'" in clash, clash
    assert not (scene / "pieces" / "clash").exists()
    # The pack names the bare part; the export must too.
    with np.load(scene / "pieces" / "shop" / "Bench.npz") as npz:
        assert sorted(npz.files) == ["leg_L", "seat_top"]

    resting = tmp_path / "resting"
    shutil.copytree(scene, resting)
    _seam_scene(resting, build_dir, 0.002)
    ctx = load_scene(resting)
    assert MustTouch().run(ctx) == []
    assert run_gate([MustTouch()], ctx, None).status == "passed"  # its plants are caught too

    lifted = tmp_path / "lifted"
    shutil.copytree(scene, lifted)
    _seam_scene(lifted, build_dir, 0.02)
    (found,) = MustTouch().run(load_scene(lifted))
    assert "shop:Bench.edge_R" in found.item and "gap 0.0200" in found.detail


STORED_WEIGHTS = """
from pathlib import Path
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, rig_pack

out = Path(runner.script_args()[0])
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
hand.modifiers.new("Armature", "ARMATURE").object = arm
body = hand.vertex_groups.new(name="body")
fist = hand.vertex_groups.new(name="hand.R")
root = hand.vertex_groups.new(name="root")
fist.add(range(8), 1.0, "REPLACE")
# Hand-weighted after the bind step: one vertex under the bind floor, one split three ways.
body.add([0], 0.0005, "REPLACE")
fist.add([0], 0.9995, "REPLACE")
body.add([5], 0.25, "REPLACE")
fist.add([5], 0.5, "REPLACE")
root.add([5], 0.25, "REPLACE")
pack = rig_pack(contact_groups={"hand.R": ["hand.R"]}, parts={"Hand": "hand.R"})
export.export_rig(arm, pack, "Doll", out, clips=[])
runner.done()
"""


def test_export_rig_writes_stored_weights(tmp_path):
    """The weights are Blender's own, before the bind step's `BIND_WEIGHT_FLOOR` could drop any."""
    result = _run(tmp_path, STORED_WEIGHTS, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    rig = json.loads((tmp_path / "out" / "rigs" / "shop" / "Doll.json").read_text())
    assert rig["vertex_groups"]["Hand"] == ["body", "hand.R", "root"]
    with np.load(tmp_path / "out" / "rigs" / "shop" / "Doll.rest.npz") as npz:
        w, j = npz["w:Hand"], npz["j:Hand"]
    assert w.dtype == np.float32 and j.dtype == np.int32 and w.shape == j.shape == (8, 3)
    # K is the most groups on any vertex (vertex 5 has three); the others pad with 0 / 0.
    np.testing.assert_array_equal(w[0], np.float32([0.0005, 0.9995, 0.0]))
    np.testing.assert_array_equal(j[0], [0, 1, 0])
    np.testing.assert_array_equal(w[5], np.float32([0.25, 0.5, 0.25]))
    np.testing.assert_array_equal(j[5], [0, 1, 2])
    for i in (1, 2, 3, 4, 6, 7):
        np.testing.assert_array_equal(w[i], np.float32([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(j[i], [1, 0, 0])


BLEND_REFUSAL = """
from pathlib import Path
from model_kit.blender import export, runner
from model_kit.blender._selftest_fk import build_rig, cube, rig_pack
from model_kit.core.errors import ManifestError

out = Path(runner.script_args()[0])
arm = build_rig()
hand = cube("Hand", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
mod = hand.modifiers.new("Armature", "ARMATURE")
mod.object = arm
hand.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
hand.modifiers.new("Outline", "SOLIDIFY")  # the rig's outline modifier, after the Armature
parts = {{"Hand": "hand.R"}}
{setup}
pack = rig_pack(contact_groups={{"hand.R": ["hand.R"]}}, parts=parts,
                outline_modifiers=["Outline"])
try:
    export.export_rig(arm, pack, "Doll", out, clips=[])
    print("NO RAISE")
except ManifestError as exc:
    print("RAISED |", exc)
print("WRITTEN", out.exists())
runner.done()
"""


def _refused(tmp_path: Path, setup: str, *words: str) -> None:
    result = _run(tmp_path, BLEND_REFUSAL.format(setup=setup), str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert "NO RAISE" not in lines, result.log[-1500:]
    message = next(line for line in lines if line.startswith("RAISED |"))
    for word in words:
        assert word in message, (word, message)
    assert "WRITTEN False" in lines  # it raised before writing anything


def test_preserve_volume_refused(tmp_path):
    _refused(tmp_path, "mod.use_deform_preserve_volume = True", "'Hand'",
             "use_deform_preserve_volume")


@pytest.mark.parametrize("setup, word", [
    pytest.param("mod.use_multi_modifier = True", "use_multi_modifier", id="multi-modifier"),
    pytest.param("mod.use_vertex_groups = False", "use_vertex_groups", id="no-vertex-groups"),
    pytest.param("mod.use_bone_envelopes = True", "use_bone_envelopes", id="envelopes"),
    pytest.param("mod.vertex_group = 'hand.R'", "vertex group 'hand.R'", id="masked"),
])
def test_other_blend_settings_refused(tmp_path, setup, word):
    _refused(tmp_path, setup, "'Hand'", word)


def test_a_body_mesh_without_an_armature_modifier_is_refused(tmp_path):
    # The Hand is a child of the armature, so it is still a body mesh. A second, skinned part
    # keeps the contact group fed, so the check under test is reached.
    setup = ("palm = cube('Palm', (0.4, 0.1, 1.05), (0.05, 0.05, 0.05))\n"
             "palm.modifiers.new('Armature', 'ARMATURE').object = arm\n"
             "palm.vertex_groups.new(name='hand.R').add(range(8), 1.0, 'REPLACE')\n"
             "parts['Palm'] = 'hand.R'\n"
             "hand.parent = arm\n"
             "hand.modifiers.remove(mod)")
    _refused(tmp_path, setup, "'Hand'", "no Armature modifier")


def test_bbone_segments_refused(tmp_path):
    _refused(tmp_path, "arm.data.bones['hand.R'].bbone_segments = 4", "bbone_segments",
             "'hand.R'")


def test_a_bbone_on_a_bone_that_does_not_deform_is_allowed(tmp_path):
    setup = ("arm.data.bones['root'].use_deform = False\n"
             "arm.data.bones['root'].bbone_segments = 4")
    result = _run(tmp_path, BLEND_REFUSAL.format(setup=setup), str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert "NO RAISE" in result.log.splitlines(), result.log[-1500:]
    assert "WRITTEN True" in result.log.splitlines()


def test_shape_key_with_rest_value_0_is_allowed(tmp_path):
    """A key at rest value 0 leaves the rest shell where the base mesh is, so the weights hold."""
    setup = "hand.shape_key_add(name='Basis')\nhand.shape_key_add(name='Smile').value = 0.0"
    result = _run(tmp_path, BLEND_REFUSAL.format(setup=setup), str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert "NO RAISE" in result.log.splitlines(), result.log[-1500:]
    assert "WRITTEN True" in result.log.splitlines()


@pytest.mark.parametrize("setup, words", [
    pytest.param("hand.shape_key_add(name='Basis')\n"
                 "hand.shape_key_add(name='Smile').value = 0.5",
                 ["'Hand'", "'Smile'", "0.5"], id="rest-value-0.5"),
    pytest.param("hand.shape_key_add(name='Basis')\n"
                 "hand.shape_key_add(name='Smile').value = 0.0\n"
                 "hand.data.shape_keys.use_relative = False",
                 ["'Hand'", "use_relative"], id="absolute-keys"),
])
def test_shape_key_off_rest_is_refused(tmp_path, setup, words):
    _refused(tmp_path, setup, *words, "shape key")


@pytest.mark.parametrize("setup, words", [
    # Neither changes the vertex count, so only the order rule can refuse them.
    pytest.param("hand.modifiers.new('Taper', 'SIMPLE_DEFORM')", ["'Taper'", "after"],
                 id="deform-after"),
    pytest.param("hand.modifiers.new('Wiggle', 'DISPLACE')", ["'Wiggle'", "after"],
                 id="after-the-outline"),
])
def test_a_modifier_after_the_armature_is_refused(tmp_path, setup, words):
    _refused(tmp_path, setup, "'Hand'", *words)


def test_the_outline_modifier_may_follow_the_armature(tmp_path):
    result = _run(tmp_path, BLEND_REFUSAL.format(setup="pass"), str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert "NO RAISE" in result.log.splitlines(), result.log[-1500:]


BITS = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.manifest import Listed, Recipe, View

out = Path(runner.script_args()[0])
mode = runner.script_args()[1]
options = {
    "draco": build.DRACO,
    "tuned": {**build.DRACO, "export_draco_position_quantization": 10},
    "draco_off": {"export_draco_mesh_compression_enable": False,
                  "export_draco_position_quantization": 10},
    "zero": {**build.DRACO, "export_draco_position_quantization": 0},
    "plain": None,
}[mode]
build.reset()
fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
pack = build.PackBuilder("shop", 1.0, out, 2.0, crease_deg=35)
root = build.piece("Box")
build.slab("body", -0.1, 0.1, 0, 0.2, -0.1, 0.1, fill, root,
           props={"listed": "body", "group": "body"})
pack.register(root, view=View(1.0, "a test piece, seen from arm's length"), origin="floor", recipe=Recipe("assembly", "x.py"),
              listed=[Listed("body", "A box", "a drawing", 0.01, None)], done="A box.",
              size_class="prop", kind="box")
pack.use(root, "front", point=(0, 0, 0.3), facing=(0, 0, -1), posture="stand")
written = pack.finish(["Box"], glb_options=options)
print("EXPORT", written.pieces["Box"].measured["export"])
runner.done()
"""


def _bits(tmp_path: Path, mode: str) -> dict:
    out = tmp_path / "build"
    tmp_path.mkdir()
    result = _run(tmp_path, BITS, str(out), mode)
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    raw = json.loads((out / "shop.pack.json").read_text())
    return raw["pieces"]["Box"]["measured"]["export"]


def test_draco_bits_recorded(tmp_path):
    # Draco on with no bits set: the exporter's own default, 14.
    assert _bits(tmp_path / "default", "draco") == {"position_bits": 14}
    # Bits the build sets are the bits recorded.
    assert _bits(tmp_path / "tuned", "tuned") == {"position_bits": 10}


def test_float32_records_null(tmp_path):
    assert _bits(tmp_path / "plain", "plain") == {"position_bits": None}
    # Bits set while Draco is off change nothing: the file holds float32.
    assert _bits(tmp_path / "off", "draco_off") == {"position_bits": None}
    # 0 bits is the exporter's "no quantization": floats, like no Draco.
    assert _bits(tmp_path / "zero", "zero") == {"position_bits": None}


_KEY_SLOT = _ISOLATION_HEAD + """
# The Hand has a shape key `Smile` that moves every vertex +0.1 in x. The clip keys `root` on
# the armature's OBJECT slot and `Smile` on a KEY slot of the same action.
hand.shape_key_add(name="Basis")
smile = hand.shape_key_add(name="Smile")
smile.value = 0.0  # the rest value; the clip keys it
for point in smile.data:
    point.co.x += 0.1
key = hand.data.shape_keys
plain = key_root()
action = key_root()  # the same root slide, plus Smile on a KEY slot
kslot = action.slots.new("KEY", key.name)
kbag = action.layers[0].strips[0].channelbag(kslot, ensure=True)
fc = kbag.fcurves.new('key_blocks["Smile"].value')
for frame, value in ((1, 0.0), (2, 1.0)):
    fc.keyframe_points.insert(frame, value)
arm.animation_data.action = None
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, plain, pack, "Doll", keyed_clip("plain"), out, step=1.0)
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
gap = samples("c")["v:Hand"][:, :, 0] - samples("plain")["v:Hand"][:, :, 0]
print("DX", gap.astype(float).round(4).tolist())
print("KEY ANIMATION", key.animation_data)
print("SMILE VALUE", smile.value)
runner.done()
"""


def test_samples_follow_the_key_slot(tmp_path):
    """A shape key keyed on the clip's KEY slot moves the sampled vertices, and nothing of it
    stays on the key afterwards."""
    result = _run(tmp_path, _KEY_SLOT, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    # Frame 1 keys Smile at 0 and frame 2 at 1: the clip moves every vertex by 0.1 in x.
    assert f"DX {[[0.0] * 8, [0.1] * 8]}" in lines, result.log[-2000:]
    assert "KEY ANIMATION None" in lines and "SMILE VALUE 0.0" in lines


_KEY_NAME_MISMATCH = _ISOLATION_HEAD + """
from model_kit.core.errors import ManifestError
# Two meshes with a shape key each; the action has two KEY slots, named for neither Key, and
# both animate the Hand key's block, so nothing says which one is the Hand key's.
hand.shape_key_add(name="Basis")
hand.shape_key_add(name="Smile").value = 0.0
palm = cube("Palm", (0.4, 0.1, 1.05), (0.05, 0.05, 0.05))
palm.modifiers.new("Armature", "ARMATURE").object = arm
palm.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
palm.shape_key_add(name="Basis")
palm.shape_key_add(name="Frown").value = 0.0
pack = rig_pack(contact_groups={"hand.R": ["hand.R"]}, parts={"Hand": "hand.R", "Palm": "hand.R"})
action = key_root()
for name in ("Elsewhere", "Nowhere"):
    kslot = action.slots.new("KEY", name)
    kbag = action.layers[0].strips[0].channelbag(kslot, ensure=True)
    kbag.fcurves.new('key_blocks["Smile"].value').keyframe_points.insert(1, 1.0)
arm.animation_data.action = None
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["c"])
try:
    export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
    print("NO RAISE")
except ManifestError as exc:
    print("RAISED |", exc)
runner.done()
"""


def test_two_key_slots_that_fit_one_key_are_refused(tmp_path):
    result = _run(tmp_path, _KEY_NAME_MISMATCH, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert "NO RAISE" not in lines, result.log[-1500:]
    message = next(line for line in lines if line.startswith("RAISED |"))
    assert "KEElsewhere" in message and "Key" in message, message


_KEY_OWN_ACTION = _ISOLATION_HEAD + """
# `Smile` has an action of its own (value 0 at frame 1, 1 at frame 2). The clip has no KEY slot,
# so Smile must read rest while sampling, as if it had no animation at all.
hand.shape_key_add(name="Basis")
smile = hand.shape_key_add(name="Smile")
smile.value = 0.0
for point in smile.data:
    point.co.x += 0.1
key = hand.data.shape_keys
action = key_root()
arm.animation_data.action = None
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, action, pack, "Doll", keyed_clip("plain"), out, step=1.0)
smile.keyframe_insert("value", frame=1)
smile.value = 1.0
smile.keyframe_insert("value", frame=2)
scene.frame_set(1)
own = key.animation_data.action
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
print("OWN ACTION", "ignored" if same(samples("plain"), samples("c")) else "PLAYED")
print("OWN ACTION BACK", key.animation_data.action == own)
runner.done()
"""


def test_a_shape_keys_own_action_does_not_play_while_sampling(tmp_path):
    result = _run(tmp_path, _KEY_OWN_ACTION, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert "OWN ACTION ignored" in lines, result.log[-2000:]
    assert "OWN ACTION BACK True" in lines


_TWO_KEYS_HEAD = _ISOLATION_HEAD + """
# Two meshes with a shape key each, and both keys have a block called Smile, so the curves cannot
# say which slot is whose. The Hand's Smile moves it 0.1 and the Palm's moves it 0.3. The Hand's
# slot keys 0 to 1 and the Palm's 0 to 0.5. The slots are made in the opposite order to the
# meshes.
palm = cube("Palm", (0.4, 0.1, 1.05), (0.05, 0.05, 0.05))
palm.modifiers.new("Armature", "ARMATURE").object = arm
palm.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
for obj, dx in ((hand, 0.1), (palm, 0.3)):
    obj.shape_key_add(name="Basis")
    block = obj.shape_key_add(name="Smile")
    block.value = 0.0
    for point in block.data:
        point.co.x += dx
pack = rig_pack(contact_groups={"hand.R": ["hand.R"]}, parts={"Hand": "hand.R", "Palm": "hand.R"})
plain = key_root()
action = key_root()
slots = {}
for obj, ident, peak in SLOTS:
    slots[obj.name] = kslot = action.slots.new("KEY", ident)
    kbag = action.layers[0].strips[0].channelbag(kslot, ensure=True)
    fc = kbag.fcurves.new('key_blocks["Smile"].value')
    for frame, value in ((1, 0.0), (2, peak)):
        fc.keyframe_points.insert(frame, value)
RECORD
arm.animation_data.action = None
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, plain, pack, "Doll", keyed_clip("plain"), out, step=1.0)
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
base, moved = samples("plain"), samples("c")
for part in ("Hand", "Palm"):
    gap = moved[f"v:{part}"][:, :, 0] - base[f"v:{part}"][:, :, 0]
    print("GAP", part, gap.astype(float).round(4).tolist())
runner.done()
"""
# By name: each slot is named for its key's current name ("KE" and the name), made Palm first.
_TWO_KEYS = _TWO_KEYS_HEAD.replace("SLOTS", (
    '((palm, palm.data.shape_keys.name, 0.5), (hand, hand.data.shape_keys.name, 1.0))'
)).replace("RECORD", "")
# By record: the slots' names point the other way round, but Blender holds each key's assignment.
_RECORDED = _TWO_KEYS_HEAD.replace("SLOTS", (
    '((hand, palm.data.shape_keys.name, 1.0), (palm, hand.data.shape_keys.name, 0.5))'
)).replace("RECORD", """for obj in (hand, palm):
    data = obj.data.shape_keys.animation_data_create()
    data.action = action
    data.action_slot = slots[obj.name]
""")


def _check_two_keys(tmp_path, body):
    result = _run(tmp_path, body, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert f"GAP Hand {[[0.0] * 8, [0.1] * 8]}" in lines, result.log[-2000:]
    assert f"GAP Palm {[[0.0] * 8, [0.15] * 8]}" in lines, result.log[-2000:]


def test_each_key_slot_plays_on_the_key_it_is_named_for(tmp_path):
    _check_two_keys(tmp_path, _TWO_KEYS)


def test_the_recorded_pairing_beats_the_slot_name(tmp_path):
    _check_two_keys(tmp_path, _RECORDED)


_RENAMED = _ISOLATION_HEAD + """
# The KEY slot was made for a key called DinerKey; the key was renamed later.
# No assignment is recorded, so only the slot's curves (block wave_fat) say whose it is.
hand.shape_key_add(name="Basis")
smile = hand.shape_key_add(name="wave_fat")
smile.value = 0.0
for point in smile.data:
    point.co.x += 0.1
key = hand.data.shape_keys
plain = key_root()
action = key_root()
kslot = action.slots.new("KEY", "DinerKey")
kbag = action.layers[0].strips[0].channelbag(kslot, ensure=True)
fc = kbag.fcurves.new('key_blocks["wave_fat"].value')
for frame, value in ((1, 0.0), (2, 1.0)):
    fc.keyframe_points.insert(frame, value)
arm.animation_data.action = None
scene.frame_set(1)
print("SLOT", kslot.identifier, "KEY", key.name, "USERS", len(kslot.users()))
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, plain, pack, "Doll", keyed_clip("plain"), out, step=1.0)
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
gap = samples("c")["v:Hand"][:, :, 0] - samples("plain")["v:Hand"][:, :, 0]
print("GAP", gap.astype(float).round(4).tolist())
runner.done()
"""


def test_a_key_renamed_after_its_slot_was_made_is_paired_by_its_blocks(tmp_path):
    result = _run(tmp_path, _RENAMED, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert "SLOT KEDinerKey KEY Key USERS 0" in lines, result.log[-2000:]
    assert f"GAP {[[0.0] * 8, [0.1] * 8]}" in lines, result.log[-2000:]


_OTHER_HEAD = _ISOLATION_HEAD + """
from model_kit.core.errors import ManifestError
# One key (Smile) on the body. The action's only KEY slot is named for another mesh's key.
hand.shape_key_add(name="Basis")
smile = hand.shape_key_add(name="Smile")
smile.value = 0.0
for point in smile.data:
    point.co.x += 0.1
plain = key_root()
action = key_root()
kslot = action.slots.new("KEY", "ElsewhereKey")
kbag = action.layers[0].strips[0].channelbag(kslot, ensure=True)
for block in BLOCKS:
    fc = kbag.fcurves.new(f'key_blocks["{block}"].value')
    for frame, value in ((1, 0.0), (2, 1.0)):
        fc.keyframe_points.insert(frame, value)
arm.animation_data.action = None
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, plain, pack, "Doll", keyed_clip("plain"), out, step=1.0)
try:
    export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
    print("OTHER SLOT", "ignored" if same(samples("plain"), samples("c")) else "BOUND")
except ManifestError as exc:
    print("RAISED |", exc)
runner.done()
"""


def test_a_lone_key_is_not_bound_to_a_slot_for_another_mesh(tmp_path):
    """The slot animates a block (Frown) that the body key lacks, so it is not the key's."""
    result = _run(tmp_path, _OTHER_HEAD.replace("BLOCKS", '["Frown"]'), str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    assert "OTHER SLOT ignored" in result.log.splitlines(), result.log[-2000:]


def test_a_slot_that_fits_a_key_in_part_is_refused(tmp_path):
    """It animates Smile (the key has it) and Frown (the key lacks it): not told from a mismatch."""
    result = _run(tmp_path, _OTHER_HEAD.replace("BLOCKS", '["Smile", "Frown"]'),
                  str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    message = next((line for line in lines if line.startswith("RAISED |")), None)
    assert message is not None, result.log[-1500:]
    assert "KEElsewhereKey" in message and "'Key'" in message.replace('"', "'"), message


_USERS = _TWO_KEYS_HEAD.replace("SLOTS", (
    '((hand, palm.data.shape_keys.name, 1.0), (palm, hand.data.shape_keys.name, 0.5))'
)).replace("RECORD", """# An NLA strip that plays the slot is Blender's record too (the slot's users() list the key), but
# the key's own animation data holds no action.
for obj in (hand, palm):
    data = obj.data.shape_keys.animation_data_create()
    strip = data.nla_tracks.new().strips.new("s", 1, action)
    strip.action_slot = slots[obj.name]
    assert data.action is None and [u.name for u in strip.action_slot.users()]
""")


def test_the_slots_users_decide_when_the_key_holds_no_action(tmp_path):
    _check_two_keys(tmp_path, _USERS)


_FOREIGN = _ISOLATION_HEAD + """
# The body key has blocks Basis, Smile and Blink and no slot. The file also holds a Face mesh, not
# in the piece, whose key has a block called Smile; the action's one KEY slot is that key's and
# keys Smile. Its curves fit the body key too, but the slot belongs to the Face's key.
hand.shape_key_add(name="Basis")
for name in ("Smile", "Blink"):
    block = hand.shape_key_add(name=name)
    block.value = 0.0
    for point in block.data:
        point.co.x += 0.1
face = cube("Face", (2.0, 0.0, 1.0), (0.05, 0.05, 0.05))
face.shape_key_add(name="Basis")
face.shape_key_add(name="Smile").value = 0.0
plain = key_root()
action = key_root()
kslot = action.slots.new("KEY", face.data.shape_keys.name)
kbag = action.layers[0].strips[0].channelbag(kslot, ensure=True)
fc = kbag.fcurves.new('key_blocks["Smile"].value')
for frame, value in ((1, 0.0), (2, 1.0)):
    fc.keyframe_points.insert(frame, value)
arm.animation_data.action = None
scene.frame_set(1)
export.export_rig(arm, pack, "Doll", out, clips=["plain", "c"])
export.export_samples(arm, plain, pack, "Doll", keyed_clip("plain"), out, step=1.0)
export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)
print("FOREIGN SLOT", "ignored" if same(samples("plain"), samples("c")) else "BOUND")
runner.done()
"""


def test_a_slot_for_a_key_outside_the_body_is_not_bound_to_a_body_key(tmp_path):
    result = _run(tmp_path, _FOREIGN, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    assert "FOREIGN SLOT ignored" in result.log.splitlines(), result.log[-2000:]


_FITS_TWO = _TWO_KEYS_HEAD.replace("SLOTS", (
    '((hand, "Mystery", 1.0),)'
)).replace("RECORD", "").replace(
    'export.export_samples(arm, plain, pack, "Doll", keyed_clip("plain"), out, step=1.0)\n'
    'export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)\n',
    'from model_kit.core.errors import ManifestError\n'
    'try:\n'
    '    export.export_samples(arm, action, pack, "Doll", keyed_clip("c"), out, step=1.0)\n'
    '    print("NO RAISE")\n'
    'except ManifestError as exc:\n'
    '    print("RAISED |", exc)\n'
    'runner.done()\n').split("base, moved")[0]


def test_a_slot_that_fits_two_keys_is_refused(tmp_path):
    """Both keys have a block called Smile, nothing records or names the slot's key."""
    result = _run(tmp_path, _FITS_TWO, str(tmp_path / "out"))
    assert "Traceback" not in result.log, result.log[-2000:]
    assert result.ok, result.log[-2000:]
    lines = result.log.splitlines()
    assert "NO RAISE" not in lines, result.log[-1500:]
    message = next(line for line in lines if line.startswith("RAISED |"))
    assert "KEMystery" in message and "fits shape keys" in message, message
