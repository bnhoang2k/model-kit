"""Keyed clips as data: an armature action dumped to JSON keys and keyed back, in Blender."""

import json
from pathlib import Path

import pytest

from model_kit.blender.runner import run_blender

TIMEOUT = 240

pytestmark = pytest.mark.blender

# Shared by the scripts below: build an action curve by curve, and read curves back by key.
HELPERS = '''
import json, sys
import bpy
from model_kit.blender import runner
from model_kit.blender import keys as K
from model_kit.blender._selftest_fk import build_rig


def make_action(arm, name, curves, key_curves=()):
    """curves: (data_path, index, [key dict]). key_curves land on a KEY slot."""
    act = bpy.data.actions.new(name)
    slot = act.slots.new("OBJECT", arm.name)
    layer = act.layers.new("Layer")
    strip = layer.strips.new(type="KEYFRAME")
    bag = strip.channelbag(slot, ensure=True)
    for path, index, rows in curves:
        fc = bag.fcurves.new(path, index=index)
        fc.keyframe_points.add(len(rows))
        for kp, row in zip(fc.keyframe_points, rows):
            kp.co = (row["f"], row["v"])
            kp.interpolation = row.get("interp", "BEZIER")
            kp.easing = row.get("easing", "AUTO")
            kp.handle_left_type = row.get("lt", "AUTO_CLAMPED")
            kp.handle_right_type = row.get("rt", "AUTO_CLAMPED")
            if "lh" in row:
                kp.handle_left = row["lh"]
            if "rh" in row:
                kp.handle_right = row["rh"]
    if key_curves:
        kslot = act.slots.new("KEY", "Shapes")
        kbag = strip.channelbag(kslot, ensure=True)
        for path, index, rows in key_curves:
            fc = kbag.fcurves.new(path, index=index)
            fc.keyframe_points.add(len(rows))
            for kp, row in zip(fc.keyframe_points, rows):
                kp.co = (row["f"], row["v"])
    return act


def object_curves(arm):
    act = arm.animation_data.action
    slot = arm.animation_data.action_slot
    bag = act.layers[0].strips[0].channelbag(slot)
    return {(fc.data_path, fc.array_index): fc for fc in bag.fcurves}
'''

ROUND_TRIP = HELPERS + '''
out = sys.argv[sys.argv.index("--") + 1]
src = build_rig("Src")
rows_a = [
    {"f": 1, "v": 0.0, "lt": "FREE", "rt": "FREE", "lh": (-1.0, 0.4), "rh": (3.5, -0.8)},
    {"f": 5, "v": 1.0, "lt": "ALIGNED", "rt": "ALIGNED", "lh": (3.0, 0.2), "rh": (7.0, 1.6),
     "easing": "EASE_IN_OUT"},
    {"f": 9, "v": 0.2, "lt": "VECTOR", "rt": "VECTOR"},
    {"f": 12, "v": -0.6, "lt": "AUTO", "rt": "AUTO"},
    {"f": 16, "v": 0.5},
]
rows_b = [{"f": 1, "v": 0.5, "interp": "CONSTANT"}, {"f": 8, "v": 2.0, "interp": "CONSTANT"},
          {"f": 16, "v": -1.0, "interp": "CONSTANT"}]
rows_c = [{"f": 2, "v": 0.1, "interp": "LINEAR", "easing": "EASE_OUT"},
          {"f": 14, "v": 1.1, "interp": "LINEAR"}]
act = make_action(src, "ClipSrc", [
    ('pose.bones["body"].rotation_euler', 0, rows_a),
    ('pose.bones["body"].rotation_euler', 2, [dict(r, v=-r["v"]) for r in rows_a]),
    ('pose.bones["root"].location', 1, rows_b),
    ('pose.bones["hand.R"].scale', 0, rows_c),
])
src.animation_data_create()
src.animation_data.action = act
src.animation_data.action_slot = act.slots[0]

keys, left_out = K.action_keys(act, ["root", "body", "hand.R"])
K.write_keys(keys, out + "/clip.json")
again = K.read_keys(out + "/clip.json")
assert again == keys, "JSON round trip changed the keys"

dst = build_rig("Dst")
bpy.context.scene.render.fps = again["fps"]
bpy.context.scene.render.fps_base = 1.0
new = K.key_action(dst, "ClipDst", again)
assert dst.animation_data.action is new
assert dst.animation_data.action_slot is not None

want, got = object_curves(src), object_curves(dst)
lo, hi = keys["frames"]
frames = [lo + 0.5 * i for i in range(int((hi - lo) * 2) + 1)]
result = {
    "left_out": left_out, "frames": keys["frames"], "fps": keys["fps"],
    "same_curves": sorted(map(list, want)) == sorted(map(list, got)),
    "slots": [[s.target_id_type, s.identifier] for s in new.slots],
    "curve_count": len(keys["curves"]),
    "key_rows": {f"{c['bone']}.{c['path']}[{c['index']}]": c["keys"] for c in keys["curves"]},
    "evals": {f"{p}[{i}]": [[fc.evaluate(f) for f in frames], [got[(p, i)].evaluate(f) for f in frames]]
              for (p, i), fc in want.items()},
}
json.dump(result, open(out + "/result.json", "w"))
runner.done()
'''


def _run(tmp_path: Path, body: str):
    script = tmp_path / "script.py"
    script.write_text(body)
    return run_blender(script, [str(tmp_path)], timeout=TIMEOUT)


def test_keys_round_trip(tmp_path):
    result = _run(tmp_path, ROUND_TRIP)
    assert result.ok, result.log
    out = json.loads((tmp_path / "result.json").read_text())
    assert out["left_out"] == []
    assert out["frames"] == [1, 16] and out["fps"] == 24
    assert out["same_curves"] and out["curve_count"] == 4
    assert len(out["slots"]) == 1 and out["slots"][0][0] == "OBJECT"
    # The dump holds the handle types and easing, not only the values.
    rows = out["key_rows"]["body.rotation_euler[0]"]
    assert [r[4] for r in rows] == ["FREE", "ALIGNED", "VECTOR", "AUTO", "AUTO_CLAMPED"]
    assert rows[1][3] == "EASE_IN_OUT" and rows[0][5:7] == pytest.approx([-1.0, 0.4])
    assert {r[2] for r in out["key_rows"]["root.location[1]"]} == {"CONSTANT"}
    assert len(rows[0]) == 10
    assert len(out["evals"]) == 4
    for name, (want, got) in out["evals"].items():
        assert len(want) == 31
        assert got == pytest.approx(want, abs=1e-9, rel=0), name


LEFT_OUT = HELPERS + '''
out = sys.argv[sys.argv.index("--") + 1]
arm = build_rig("Src")
two = [{"f": 1, "v": 0.0}, {"f": 9, "v": 1.0}]
act = make_action(arm, "Clip", [
    ('pose.bones["body"].rotation_euler', 0, two),
    ('pose.bones["prop.R"].scale', 0, two),
    ('pose.bones["prop.R"].location', 2, two),
    ('pose.bones["hand.R"].location', 1, two),
    ('pose.bones["body"].rotation_quaternion', 1, two),
    ('["wave"]', -1, two),
], key_curves=[('key_blocks["wave"].value', 0, two)])
keys, left_out = K.action_keys(act, ["body", "prop.R"], leave_out=[("prop.R", "scale")])
result = {"left_out": left_out,
          "kept": [[c["bone"], c["path"], c["index"]] for c in keys["curves"]]}
json.dump(result, open(out + "/result.json", "w"))
runner.done()
'''


def test_left_out_curves_are_listed(tmp_path):
    result = _run(tmp_path, LEFT_OUT)
    assert result.ok, result.log
    out = json.loads((tmp_path / "result.json").read_text())
    assert sorted(out["left_out"]) == sorted([
        'key_blocks["wave"].value',           # the KEY slot
        'pose.bones["hand.R"].location',      # a bone outside `bones`
        'pose.bones["prop.R"].scale',         # named in leave_out
        'pose.bones["body"].rotation_quaternion',  # a path the dump does not hold
        '["wave"]',                           # a custom property
    ])
    assert out["kept"] == [["body", "rotation_euler", 0], ["prop.R", "location", 2]]


FMODIFIER = HELPERS + '''
arm = build_rig("Src")
two = [{"f": 1, "v": 0.0}, {"f": 9, "v": 1.0}]
act = make_action(arm, "Clip", [('pose.bones["body"].rotation_euler', 1, two)])
bag = act.layers[0].strips[0].channelbag(act.slots[0])
bag.fcurves[0].modifiers.new("NOISE")
K.action_keys(act, ["body"])
runner.done()
'''


def test_fmodifier_raises(tmp_path):
    result = _run(tmp_path, FMODIFIER)
    assert result.ok is False
    assert "ManifestError" in result.log
    assert 'pose.bones["body"].rotation_euler' in result.log and "modifier" in result.log


MISSING_BONE = HELPERS + '''
arm = build_rig("Dst")
keys = {"frames": [1, 9], "fps": 24, "curves": [
    {"bone": "ghost", "path": "location", "index": 0,
     "keys": [[1, 0.0, "BEZIER", "AUTO", "AUTO_CLAMPED", 0.0, 0.0, "AUTO_CLAMPED", 2.0, 0.0]]}]}
bpy.context.scene.render.fps = keys["fps"]
bpy.context.scene.render.fps_base = 1.0
K.key_action(arm, "Clip", keys)
runner.done()
'''


def test_missing_bone_raises(tmp_path):
    result = _run(tmp_path, MISSING_BONE)
    assert result.ok is False
    assert "ManifestError" in result.log
    assert 'pose.bones["ghost"].location' in result.log


FRAMES = HELPERS + '''
out = sys.argv[sys.argv.index("--") + 1]
arm = build_rig("Src")
act = make_action(arm, "Clip", [
    ('pose.bones["body"].rotation_euler', 0, [{"f": 1, "v": 0.0}, {"f": 40, "v": 1.0}]),
    ('pose.bones["prop.R"].scale', 0, [{"f": 1, "v": 1.0}, {"f": 73, "v": 2.0}]),
])
keys, left_out = K.action_keys(act, ["body", "prop.R"], leave_out=[("prop.R", "scale")])
dst = build_rig("Dst")
bpy.context.scene.render.fps = keys["fps"]
bpy.context.scene.render.fps_base = 1.0
new = K.key_action(dst, "ClipDst", keys)
json.dump({"frames": keys["frames"], "left_out": left_out,
           "new_range": list(new.frame_range)}, open(out + "/result.json", "w"))
runner.done()
'''


def test_frames_are_the_action_range_not_the_kept_keys(tmp_path):
    result = _run(tmp_path, FRAMES)
    assert result.ok, result.log
    out = json.loads((tmp_path / "result.json").read_text())
    assert out["frames"] == [1, 73]
    assert out["left_out"] == ['pose.bones["prop.R"].scale']
    assert out["new_range"] == [1, 73]  # keying back keeps the clip's length


KEY_OUTSIDE_RANGE = HELPERS + '''
arm = build_rig("Src")
act = make_action(arm, "Clip", [
    ('pose.bones["body"].rotation_euler', 0, [{"f": 1, "v": 0.0}, {"f": 40, "v": 1.0}])])
act.use_frame_range = True
act.frame_start, act.frame_end = 1, 20
K.action_keys(act, ["body"])
runner.done()
'''


def test_key_outside_frame_range_raises(tmp_path):
    result = _run(tmp_path, KEY_OUTSIDE_RANGE)
    assert result.ok is False
    assert "ManifestError" in result.log
    assert 'pose.bones["body"].rotation_euler' in result.log and "frame range" in result.log


EXTRAPOLATION = HELPERS + '''
out = sys.argv[sys.argv.index("--") + 1]
arm = build_rig("Src")
two = [{"f": 1, "v": 0.0}, {"f": 9, "v": 1.0}]
act = make_action(arm, "Clip", [
    ('pose.bones["body"].rotation_euler', 1, two),
    ('pose.bones["hand.R"].location', 0, two),
])
bag = act.layers[0].strips[0].channelbag(act.slots[0])
fcs = {fc.data_path: fc for fc in bag.fcurves}
fcs['pose.bones["hand.R"].location'].extrapolation = "LINEAR"
# A left-out curve may extrapolate: it is not dumped.
K.action_keys(act, ["body"])
json.dump({"ok": True}, open(out + "/result.json", "w"))
fcs['pose.bones["body"].rotation_euler'].extrapolation = "LINEAR"
K.action_keys(act, ["body"])
runner.done()
'''


def test_non_constant_extrapolation_raises(tmp_path):
    result = _run(tmp_path, EXTRAPOLATION)
    assert (tmp_path / "result.json").exists(), result.log  # the left-out curve did not raise
    assert result.ok is False
    assert "ManifestError" in result.log
    assert 'pose.bones["body"].rotation_euler' in result.log and "extrapolation" in result.log
    assert "hand.R" not in result.log.split("ManifestError")[-1]


WRONG_RATE = HELPERS + '''
arm = build_rig("Dst")
keys = {"frames": [1, 9], "fps": 30, "curves": []}
bpy.context.scene.render.fps = 24
bpy.context.scene.render.fps_base = 1.0
K.key_action(arm, "Clip", keys)
runner.done()
'''


def test_scene_rate_other_than_the_keys_raises(tmp_path):
    result = _run(tmp_path, WRONG_RATE)
    assert result.ok is False
    assert "ManifestError" in result.log
    assert "24 fps" in result.log and "30 fps" in result.log
