"""The numpy skinner against Blender's Armature modifier, on a chain-bound part posed to the
ends of its joints' ranges."""

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from model_kit.blender.runner import run_blender
from model_kit.core.manifest import load_pack
from model_kit.core.pose import skin
from model_kit.core.scene import _read_rest, _read_rig, _read_vertex_groups, _rig_json
from model_kit.core.tolerances import FLOAT32_EPS, NODE_FLOAT32_ULPS, NODE_OF_PIECE_SIZE
from model_kit.core.units import load_style

FIX = Path(__file__).parents[1] / "core" / "fixtures"
TIMEOUT = 480

pytestmark = pytest.mark.blender

# The shop Doll cut down to three bones: root, body (connected to root) and hand.R. `torso`
# rides body, `foot_R` rides root, and `arm_R` is bound to body and hand.R together, so its
# vertices between the two bones are mixed. Both body and hand.R bend the skin, so both carry a
# joint range; the ends are not symmetric, so a sign slip would show.
JOINTS = {
    "body": {"range_deg": {"x": [-50, 70], "y": [-40, 40], "z": [-60, 30]},
             "why": "test fixture: schema contract, not anatomy"},
    "hand.R": {"range_deg": {"x": [-60, 60], "y": [-45, 65], "z": [-60, 50]},
               "why": "test fixture: schema contract, not anatomy"},
}
# arm_R spans glTF y 0.45 to 1.05 in the armature's frame; body weighs on its low end and
# hand.R on its high end, with a smooth change between 0.2 and 0.8 of the way from 0.5 to 1.0.
FALLOFF = {"along": "y", "start": 0.5, "end": 1.0, "bands": [[0.2, 0.8]]}


def _pack_json() -> dict:
    pack = json.loads((FIX / "shop.pack.json").read_text())
    rig = pack["pieces"]["Doll"]["rig"]
    rig["bones"] = [{"name": "root", "parent": None, "length": 0.4},
                    {"name": "body", "parent": "root", "length": 0.6},
                    {"name": "hand.R", "parent": "body", "length": 0.3}]
    rig["bind"] = {"torso": {"bones": ["body"], "mode": "rigid"},
                   "arm_R": {"bones": ["body", "hand.R"], "mode": "chain", "falloff": FALLOFF},
                   "foot_R": {"bones": ["root"], "mode": "rigid"}}
    rig["contact_groups"] = {"hand.R": ["hand.R"], "torso": ["body"]}
    rig["chains"]["arm.R"].update(bones=["hand.R"], dofs={"hand.R": "xyz"}, limits={})
    rig["mirror"] = {"pairs": [], "axis": "x"}
    rig["outline_modifiers"] = []
    rig["supports"] = ["torso"]
    rig["joints"] = JOINTS
    return pack


def _poses(joints: dict) -> list[dict[str, list[float]]]:
    """Rest; each joint's axis alone at each end; then every mix of the two joints' axis ends.
    A pose is `{bone: [x, y, z]}` in degrees. All the angles come from the joints' ranges."""
    ends = {bone: [(axis, end) for axis in "xyz" for end in joints[bone].range_deg[axis]]
            for bone in joints}
    corners = {bone: [dict(zip("xyz", c, strict=True)) for c in itertools.product(
        *(joints[bone].range_deg[a] for a in "xyz"))] for bone in joints}
    poses: list[dict[str, list[float]]] = [{}]
    for bone, axes in ends.items():
        for axis, end in axes:
            poses.append({bone: [end if a == axis else 0.0 for a in "xyz"]})
    for body, hand in itertools.product(corners["body"], corners["hand.R"]):
        poses.append({"body": [body[a] for a in "xyz"], "hand.R": [hand[a] for a in "xyz"]})
    return poses


SCRIPT = """
import json, math
from pathlib import Path
import bmesh
import bpy
from model_kit.blender import bind, export, runner
from model_kit.blender._selftest_fk import build_rig, keyed_clip
from model_kit.core.manifest import load_pack
from model_kit.core.units import load_style

out, pack_file, style_file, poses_file = (Path(p) for p in runner.script_args())
pack = load_pack(pack_file, load_style(style_file))
rig = pack.pieces["Doll"].rig
poses = json.loads(poses_file.read_text())
scene = bpy.context.scene
arm = build_rig("DollRig")


def box(name, centre, half, cuts):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=2.0)
    for v in bm.verts:
        v.co.x = centre[0] + v.co.x * half[0]
        v.co.y = centre[1] + v.co.y * half[1]
        v.co.z = centre[2] + v.co.z * half[2]
    bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=cuts, use_grid_fill=True)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.matrix_world = arm.matrix_world.copy()  # the mesh is drawn in the armature's frame
    return obj


box("torso", (0.0, 0.05, 0.7), (0.1, 0.1, 0.2), 3)
box("arm_R", (0.35, 0.1, 0.75), (0.05, 0.05, 0.3), 10)
box("foot_R", (0.1, 0.0, 0.1), (0.05, 0.05, 0.05), 2)
for part, b in rig.bind.items():
    bind.bind_part(bpy.data.objects[part], arm, b)

# One keyed frame per pose; frames are whole numbers, so every sample lands on a key.
for i, pose in enumerate(poses):
    for name in ("body", "hand.R"):
        pb = arm.pose.bones[name]
        pb.rotation_euler = [math.radians(a) for a in pose.get(name, [0.0, 0.0, 0.0])]
        pb.keyframe_insert("rotation_euler", frame=1 + i)
action = arm.animation_data.action
action.name = "ends"
scene.frame_set(1)

export.export_rig(arm, pack, "Doll", out, clips=["ends"])
export.export_samples(arm, action, pack, "Doll", keyed_clip("ends", frames=(1, len(poses))),
                      out, step=1.0)
runner.done()
"""


def _limit(size: float, points: np.ndarray) -> np.ndarray:
    """Export hygiene's node limit for each point: `NODE_OF_PIECE_SIZE` times the body's size,
    plus `NODE_FLOAT32_ULPS` float32 epsilons of the point's distance from the origin."""
    return NODE_OF_PIECE_SIZE * size + NODE_FLOAT32_ULPS * FLOAT32_EPS * np.linalg.norm(
        points, axis=1)


def test_chain_part_at_range_ends_matches(tmp_path):
    pack_file = tmp_path / "shop.pack.json"
    pack_file.write_text(json.dumps(_pack_json()))
    poses_file = tmp_path / "poses.json"
    poses = _poses(load_pack(pack_file, load_style(FIX / "shop.style.json"))
                   .pieces["Doll"].rig.joints)
    poses_file.write_text(json.dumps(poses))
    # 1 rest, 12 single ends, 8 x 8 mixes.
    assert len(poses) == 1 + 12 + 64
    out = tmp_path / "scene"
    script = tmp_path / "script.py"
    script.write_text(SCRIPT)
    shutil.copy(FIX / "shop.style.json", tmp_path / "shop.style.json")
    result = run_blender(script, [str(out), str(pack_file), str(tmp_path / "shop.style.json"),
                                  str(poses_file)], timeout=TIMEOUT)
    assert "Traceback" not in result.log and "Error" not in result.log, result.log
    assert result.ok, result.log

    rig_path = out / "rigs" / "shop" / "Doll.json"
    rig, _ = _read_rig(rig_path)
    raw = _rig_json(rig_path)
    rest = _read_rest(rig_path.with_name("Doll.rest.npz"), _read_vertex_groups(rig_path, raw, rig))
    assert set(rest.shells) == {"torso", "arm_R", "foot_R"}
    assert rest.vertex_groups["arm_R"] == ("body", "hand.R")
    # The proof needs mixed vertices: arm_R has many that weigh on both bones.
    w = rest.weights["arm_R"]
    assert int(((w > 0.05).sum(axis=1) >= 2).sum()) >= 20

    with np.load(out / "samples" / "ends.npz") as npz:
        saved = {k: npz[k] for k in npz.files}
    assert len(saved["times"]) == len(poses)
    every = np.concatenate([s[0] for s in rest.shells.values()]).astype(float)
    size = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))

    worst, worst_limit, moved = 0.0, 0.0, 0.0
    ratio = 0.0
    for i in range(len(poses)):
        posed = skin(rest, rig, {k[2:]: v[i] for k, v in saved.items() if k[:2] == "b:"})
        for part, points in posed.items():
            blender = saved[f"v:{part}"][i].astype(float)
            err = np.linalg.norm(points - blender, axis=1)
            limit = _limit(size, blender)
            worst, worst_limit = max(worst, err.max()), max(worst_limit, limit[err.argmax()])
            ratio = max(ratio, float((err / limit).max()))
            moved = max(moved, float(np.linalg.norm(blender - rest.shells[part][0], axis=1).max()))
    print(f"max skin vs Blender error {worst:.3g} (limit there {worst_limit:.3g}); "
          f"worst share of the limit {ratio:.3f}; poses {len(poses)}; "
          f"largest move from rest {moved:.3g}; body size {size:.3g}")
    assert moved > 0.5  # the ends really bend the body
    assert ratio <= 1.0, f"skin differs from Blender by {ratio:.3g} times the node limit"
