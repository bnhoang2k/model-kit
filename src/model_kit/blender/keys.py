"""A hand-keyed clip stored as data: dump an armature action's keys to JSON and key them back.

A clip keyed by hand in Blender is the one thing a script cannot rebuild from the pack. This
module reads such an action into a plain dict (`action_keys`), keeps it in a file
(`write_keys`, `read_keys`), and keys it onto an armature again (`key_action`).

Only pose-bone `location`, `rotation_euler` and `scale` curves of the named bones are kept, and
only on the action's OBJECT slot (a KEY slot, for shape keys, is left out). Every key holds its
interpolation, easing and both handles, so a Bezier curve comes back with the same shape. The
dict:

    {"frames": [start, end], "fps": 24,
     "curves": [{"bone": "arm.L", "path": "rotation_euler", "index": 0,
                 "keys": [[frame, value, interpolation, easing,
                           left_type, left_x, left_y, right_type, right_x, right_y], ...]}]}

`frames` is the action's frame range, so a curve left out of the dump still sets the clip's
length; `key_action` sets the new action's range to it. `fps` is the scene rate when the keys
were read; `key_action` does not change the scene rate and raises when the scene runs at another
rate, so the caller sets it first. A kept curve with an F-modifier or an
extrapolation other than CONSTANT raises, because its keys alone would not reproduce it, and
so does a kept key outside the frame range. Curve groups and mute flags are not stored. Needs
Blender 4.4 or later (action slots).
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterator
from pathlib import Path

import bpy

from model_kit.core.errors import ManifestError

PATHS = ("location", "rotation_euler", "scale")
_POSE_PATH = re.compile(r'pose\.bones\["([^"]+)"\]\.(\w+)')


def _bone_path(bone: str, path: str) -> str:
    return f'pose.bones["{bone}"].{path}'


def _channelbags(action, slot) -> Iterator:
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type == "KEYFRAME":
                bag = strip.channelbag(slot)
                if bag is not None:
                    yield bag


def _number(value: float) -> int | float:
    """A whole number as an int, so `1.0` is stored as `1`."""
    return int(value) if float(value).is_integer() else float(value)


def _key_row(kp) -> list:
    return [_number(kp.co[0]), float(kp.co[1]), kp.interpolation, kp.easing,
            kp.handle_left_type, float(kp.handle_left[0]), float(kp.handle_left[1]),
            kp.handle_right_type, float(kp.handle_right[0]), float(kp.handle_right[1])]


def action_keys(action, bones: Collection[str], *,
                leave_out: Collection[tuple[str, str]] = ()) -> tuple[dict, list[str]]:
    """The keys of `action`'s OBJECT-slot curves for `bones`, and the curves left out.

    Kept: `location`, `rotation_euler` and `scale` of each bone in `bones`, except the
    (bone, path) pairs in `leave_out`. The list holds the data path of every other curve:
    other slots, other bones, other paths, and `leave_out`, each path once. A kept curve with
    an F-modifier or an extrapolation other than CONSTANT raises `ManifestError` naming its data
    path. `frames` is the action's frame range; a kept key outside it raises too.
    """
    wanted = set(bones)
    skipped = set(leave_out)
    curves, left_out = [], []
    for slot in action.slots:
        for bag in _channelbags(action, slot):
            for fc in bag.fcurves:
                match = _POSE_PATH.fullmatch(fc.data_path)
                kept = (slot.target_id_type == "OBJECT" and match is not None
                        and match[1] in wanted and match[2] in PATHS
                        and (match[1], match[2]) not in skipped)
                if not kept:
                    if fc.data_path not in left_out:
                        left_out.append(fc.data_path)
                    continue
                if len(fc.modifiers):
                    raise ManifestError(
                        f"action {action.name!r}: curve {fc.data_path!r} [{fc.array_index}] has "
                        f"an F-modifier ({fc.modifiers[0].type}); its keys alone would not "
                        f"reproduce it, so bake or remove it first")
                if fc.extrapolation != "CONSTANT":
                    raise ManifestError(
                        f"action {action.name!r}: curve {fc.data_path!r} [{fc.array_index}] "
                        f"extrapolates {fc.extrapolation}; keyed back it would hold constant, "
                        f"so set its extrapolation to CONSTANT first")
                curves.append({"bone": match[1], "path": match[2], "index": fc.array_index,
                               "keys": [_key_row(kp) for kp in fc.keyframe_points]})
    curves.sort(key=lambda c: (c["bone"], c["path"], c["index"]))
    span = [_number(v) for v in action.frame_range]
    for c in curves:
        for row in c["keys"]:
            if not span[0] <= row[0] <= span[1]:
                raise ManifestError(
                    f"action {action.name!r}: curve {_bone_path(c['bone'], c['path'])!r} "
                    f"[{c['index']}] has a key at frame {row[0]}, outside the frame range "
                    f"{span}")
    return ({"frames": span, "fps": int(bpy.context.scene.render.fps), "curves": curves},
            left_out)


def key_action(armature, name: str, keys: dict):
    """Key `keys` (the dict `action_keys` returns) into a new action called `name`.

    The action gets a new OBJECT slot and is assigned to `armature`. It also gets a manual frame
    range, `keys["frames"]`. The scene rate (with `fps_base` 1.0) must equal `keys["fps"]`: the
    keys are in frames, so a scene at another rate would play the clip at the wrong speed. A
    scene at another rate raises `ManifestError` naming both rates, and so does a curve for a
    bone the armature lacks (naming its data path), each before anything is made.
    """
    if armature.pose is None:
        raise ManifestError(f"{armature.name!r} has no pose; it is not an armature")
    render = bpy.context.scene.render
    if render.fps_base != 1.0 or render.fps != keys["fps"]:
        raise ManifestError(
            f"the scene runs at {render.fps} fps (fps_base {render.fps_base}) but the keys "
            f"are for {keys['fps']} fps; set scene.render.fps to {keys['fps']} and fps_base "
            f"to 1.0 before keying")
    for curve in keys["curves"]:
        if curve["bone"] not in armature.pose.bones:
            raise ManifestError(
                f"curve {_bone_path(curve['bone'], curve['path'])!r} [{curve['index']}]: "
                f"armature {armature.name!r} has no bone {curve['bone']!r}")
    action = bpy.data.actions.new(name)
    action.use_frame_range = True
    action.frame_start, action.frame_end = keys["frames"]
    slot = action.slots.new("OBJECT", armature.name)
    strip = action.layers.new(name).strips.new(type="KEYFRAME")
    bag = strip.channelbag(slot, ensure=True)
    for curve in keys["curves"]:
        fc = bag.fcurves.new(_bone_path(curve["bone"], curve["path"]), index=curve["index"])
        points = fc.keyframe_points
        points.add(len(curve["keys"]))
        for kp, row in zip(points, curve["keys"], strict=True):
            frame, value, interp, easing, lt, lx, ly, rt, rx, ry = row
            kp.co = (frame, value)
            kp.interpolation = interp
            kp.easing = easing
            kp.handle_left_type, kp.handle_right_type = lt, rt
            kp.handle_left, kp.handle_right = (lx, ly), (rx, ry)
    if armature.animation_data is None:
        armature.animation_data_create()
    armature.animation_data.action = action
    armature.animation_data.action_slot = slot
    return action


def read_keys(path: str | Path) -> dict:
    """The keys dict that `write_keys` wrote to `path`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _curve_text(curve: dict) -> str:
    items = []
    for name in sorted(curve):
        if name == "keys":
            rows = ",\n".join("    " + json.dumps(row) for row in curve["keys"])
            items.append(f'"keys": [\n{rows}\n  ]')
        else:
            items.append(f"{json.dumps(name)}: {json.dumps(curve[name])}")
    return "  {" + ", ".join(items) + "}"


def write_keys(keys: dict, path: str | Path) -> None:
    """Write `keys` as JSON with the dict keys sorted and floats as Python writes them.

    Each key row sits on its own line, so a change to one key is a one-line diff.
    """
    items = []
    for name in sorted(keys):
        if name == "curves":
            body = ",\n".join(_curve_text(c) for c in keys["curves"])
            items.append(f'"curves": [\n{body}\n]' if body else '"curves": []')
        else:
            items.append(f"{json.dumps(name)}: {json.dumps(keys[name])}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("{" + ",\n".join(items) + "}\n", encoding="utf-8")
