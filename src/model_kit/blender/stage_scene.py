"""What the bake and the comparison render share in Blender: the glTF to Blender axes, a stage
camera's frame and a stage light as a Blender light. Run inside Blender; `bake_render.py` and
`compare_render.py` import it."""

import math

import bpy
from mathutils import Vector

WHITE = (1.0, 1.0, 1.0)  # the colour and strength every layer light is baked at
UNIT = 1.0


def to_blender(p) -> Vector:
    """A glTF point (x, y up, z front) as a Blender point."""
    return Vector((p[0], -p[2], p[1]))


def camera_frame(cam: dict):
    """A camera's origin, forward, right and up unit vectors, in Blender axes, and the tangent of
    half its vertical field. `fov_deg` is vertical, as in three.js."""
    origin = to_blender(cam["position"])
    forward = (to_blender(cam["target"]) - origin).normalized()
    up_ref = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(up_ref)) > 0.999:
        up_ref = Vector((0.0, 1.0, 0.0))
    right = forward.cross(up_ref).normalized()
    up = right.cross(forward)
    return origin, forward, right, up, math.tan(math.radians(cam["fov_deg"]) / 2)


def make_light(light: dict, values: dict):
    """A Blender light for one stage light: white, strength 1, unless `values` (light id ->
    {color, strength}; the comparison render and a test hook pass it) names it. `direction` is the way the light travels, in the glTF frame."""
    kind = light["kind"]
    given = values.get(light["id"], {})
    data = bpy.data.lights.new(light["id"], {"sun": "SUN", "point": "POINT", "spot": "SPOT",
                                             "area": "AREA"}[kind])
    data.color = tuple(given.get("color", WHITE))
    data.energy = given.get("strength", UNIT)
    if kind in ("point", "spot"):
        data.shadow_soft_size = light["radius_m"] or 0.0
    if kind == "spot":
        data.spot_size = math.radians(2 * light["cone_deg"])  # cone_deg is the half angle
        data.spot_blend = light["blend"]
    if kind == "area":
        data.shape = "RECTANGLE"
        data.size, data.size_y = light["size"]
    ob = bpy.data.objects.new(light["id"], data)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = to_blender(light["position"])
    if kind != "point":
        ob.rotation_euler = to_blender(light["direction"]).to_track_quat("-Z", "Y").to_euler()
    return ob
