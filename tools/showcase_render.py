"""Render one showcase loop in Blender: the Blender half of tools/showcase.py, which runs it:

    blender --background --python showcase_render.py -- <shot.json> <frames folder>

The shot file names the .glb files to place, the mode and the frame size. `turntable` turns the
models once round their vertical axis over the frames. `clip` plays each model's first animation,
each copy shifted by its own number of frames, over one length of the clip. The background is
transparent and a shadow catcher under the models keeps their contact shadow, so a loop sits on
any page colour. Frames are written as numbered PNG files.
"""

import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

from model_kit.blender import runner

# Suns, so the light does not depend on the model's size: a key from the front left, high enough
# that its shadow stays under the feet, a cool fill from the right and a rim from behind. Only the
# key casts a shadow. Name, rotation, strength, colour, angular size in degrees, casts a shadow.
LIGHTS = (
    ("key", (math.radians(28), 0.0, math.radians(-34)), 3.4, (1.0, 0.95, 0.88), 12.0, True),
    ("fill", (math.radians(64), 0.0, math.radians(70)), 0.9, (0.82, 0.88, 1.0), 20.0, False),
    ("rim", (math.radians(58), 0.0, math.radians(170)), 2.0, (1.0, 0.97, 0.92), 6.0, False),
)
# Angles the turntable framing samples, so the model fits the frame at every turn.
FIT_TURNS = 16


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def place(entry: dict, index: int, pivot: bpy.types.Object) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=entry["path"])
    new = [o for o in bpy.data.objects if o not in before]
    piece = entry.get("piece")
    if piece is not None:  # a pack holds several pieces side by side; keep the one named
        drop = [o for o in new if o.parent is None and o.type == "EMPTY" and o.name != piece]
        doomed = {d for root in drop for d in (root, *root.children_recursive)}
        new = [o for o in new if o not in doomed]
        for obj in doomed:
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj in new:
        if obj.parent is None:
            obj.location += Vector(entry.get("offset", (0.0, 0.0, 0.0)))
            obj.parent = pivot
    tint = entry.get("tint")
    if tint:
        rgb = [srgb_to_linear(int(tint[i:i + 2], 16) / 255) for i in (1, 3, 5)]
        for obj in new:
            for slot in getattr(obj, "material_slots", []):
                if slot.material is not None and slot.material.name.startswith(entry["tint_material"]):
                    mat = slot.material.copy()
                    mat.name = f"{entry['tint_material']}_{index}"
                    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*rgb, 1.0)
                    slot.material = mat
    return new


def shift_clip(objects: list[bpy.types.Object], shift: int, length: int) -> None:
    """Play the first action on an NLA strip that starts `shift` frames early and repeats, so the
    loop of `length` frames has no seam."""
    for obj in objects:
        anim = obj.animation_data
        if anim is None or anim.action is None:
            continue
        action, slot = anim.action, anim.action_slot
        anim.action = None
        track = anim.nla_tracks.new()
        strip = track.strips.new(action.name, 1 - shift, action)
        if slot is not None:
            strip.action_slot = slot
        strip.repeat = 3.0


def points(objects: list[bpy.types.Object]) -> np.ndarray:
    """Every evaluated vertex of the placed meshes, in world space, as an (n, 3) array."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    out = []
    for obj in objects:
        if obj.type != "MESH" or not obj.visible_get():  # a hidden bone shape is not drawn
            continue
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        co = np.empty(len(mesh.vertices) * 3)
        mesh.vertices.foreach_get("co", co)
        ev.to_mesh_clear()
        m = np.array(ev.matrix_world)
        co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        out.append(co)
    return np.concatenate(out)


def camera(pts: np.ndarray, spec: dict) -> None:
    """A camera that frames every point, at every turn of a turntable, with `fill` of the frame."""
    if spec["mode"] == "turntable":
        turns = []
        for k in range(FIT_TURNS):
            a = 2 * math.pi * k / FIT_TURNS
            rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            turns.append(pts @ rot.T)
        pts = np.concatenate(turns)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    look = (lo + hi) / 2
    width, tall = spec["size"]
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = spec.get("lens_mm", 70.0)
    cam_data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    fill = spec.get("fill", 0.82)
    tan_x = math.tan(math.atan(cam_data.sensor_width / (2 * cam_data.lens))) * fill
    tan_y = tan_x * tall / width
    tilt = math.radians(spec.get("tilt_deg", 14.0))
    turn = math.radians(spec.get("turn_deg", 32.0))
    back = np.array([math.sin(turn) * math.cos(tilt), -math.cos(turn) * math.cos(tilt), math.sin(tilt)])
    right = np.cross(back, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, back)
    rel = pts - look
    a, b, c = rel @ right, rel @ up, rel @ back
    dist = float(max((c + np.abs(a) / tan_x).max(), (c + np.abs(b) / tan_y).max()))
    cam.location = Vector(look + dist * back)
    cam.rotation_euler = (Vector(look) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam


def lights_and_floor(pts: np.ndarray) -> None:
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    for name, rot, energy, colour, angle, shadow in LIGHTS:
        data = bpy.data.lights.new(name, "SUN")
        data.energy, data.color, data.angle = energy, colour, math.radians(angle)
        data.use_shadow = shadow
        sun = bpy.data.objects.new(name, data)
        sun.rotation_euler = rot
        bpy.context.scene.collection.objects.link(sun)
    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.6, 0.68, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55
    bpy.context.scene.world = world
    size = 20 * max(float(np.linalg.norm(hi - lo)), 1e-3)
    # The ground is height 0 in every piece's frame; a tree's buried roots stay under it.
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    bpy.context.active_object.is_shadow_catcher = True


def render_settings(spec: dict, frames: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"  # CPU rendering needs no GPU, so it also runs headless
    scene.cycles.samples = spec.get("samples", 32)
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.use_persistent_data = True  # keep the scene between frames
    scene.render.resolution_x, scene.render.resolution_y = spec["size"]
    scene.render.resolution_percentage = 100
    scene.render.fps = spec.get("fps", 24)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "AgX"
    scene.frame_start, scene.frame_end = 1, spec["frames"]
    scene.render.filepath = str(frames / "f_")


def main() -> None:
    shot_file, frames = runner.script_args()
    spec = json.loads(Path(shot_file).read_text())
    frames = Path(frames)
    clear_scene()
    pivot = bpy.data.objects.new("pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    placed = []
    for i, entry in enumerate(spec["models"]):
        objects = place(entry, i, pivot)
        placed += objects
        if spec["mode"] == "clip":
            shift_clip(objects, entry.get("shift", 0), spec["frames"])
    bpy.context.scene.frame_set(1)
    pts = points(placed)
    camera(pts, spec)
    lights_and_floor(pts)
    if spec["mode"] == "turntable":
        pivot.rotation_euler = (0, 0, 0)
        pivot.keyframe_insert("rotation_euler", index=2, frame=1)
        pivot.rotation_euler = (0, 0, 2 * math.pi)
        pivot.keyframe_insert("rotation_euler", index=2, frame=spec["frames"] + 1)
        _linear(pivot)
    render_settings(spec, frames)
    bpy.ops.render.render(animation=True)
    runner.done()


def _linear(obj: bpy.types.Object) -> None:
    """Make every key of the object's action linear, on legacy and slotted actions alike."""
    action = obj.animation_data.action
    curves = list(getattr(action, "fcurves", []) or [])
    for layer in getattr(action, "layers", []):
        for strip in layer.strips:
            for bag in getattr(strip, "channelbags", []):
                curves += list(bag.fcurves)
    for fcurve in curves:
        for key in fcurve.keyframe_points:
            key.interpolation = "LINEAR"


if __name__ == "__main__":
    main()
