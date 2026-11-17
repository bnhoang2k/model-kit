"""The comparison's Blender half. `model_kit.compare` runs this script.

It reads one JSON argument:

    glb, cameras, lights, values, out, samples

imports the baked layout (`<scene>/bake/<layout>.glb`, the placed layout in the glTF frame),
lights it with the stage's lights at each layer's colour and strength (`values`: layer name ->
{color, strength}; a layer it lacks is white at 1), and renders every camera with Cycles to
`<out>/blender-<camera>.png`: black world but for the hemisphere lights, the Standard view
transform (no tone map), as the browser draws. `cameras` carry their pixel `width` and `height`.
The film is transparent and the PNG is RGBA: the world still lights the scene, but the camera
does not see it, so a pixel the scene does not cover has alpha 0, as in the browser's picture.
"""

import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix

from model_kit.blender import runner
from model_kit.blender.stage_scene import UNIT, WHITE, camera_frame, make_light


def import_layout(path: str) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=path)


def value_of(values: dict, layer: str) -> dict:
    given = values.get(layer, {})
    return {"color": tuple(given.get("color", WHITE)), "strength": given.get("strength", UNIT)}


def set_world(lights: list, values: dict) -> None:
    """The world: the sky hemisphere lights' sum above the horizon, the ground ones' below."""
    total = {"sky": [0.0, 0.0, 0.0], "ground": [0.0, 0.0, 0.0]}
    for light in lights:
        if light["kind"] == "hemisphere":
            v = value_of(values, light["layer"])
            for i in range(3):
                total[light["sky_ground"]][i] += v["color"][i] * v["strength"]
    world = bpy.data.worlds.new("compare")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    coord = nodes.new("ShaderNodeTexCoord")
    split = nodes.new("ShaderNodeSeparateXYZ")
    step = nodes.new("ShaderNodeMath")
    step.operation = "GREATER_THAN"
    step.inputs[1].default_value = 0.0
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs[6].default_value = (*total["ground"], 1.0)
    mix.inputs[7].default_value = (*total["sky"], 1.0)
    links.new(coord.outputs["Generated"], split.inputs["Vector"])
    links.new(split.outputs["Z"], step.inputs[0])
    links.new(step.outputs["Value"], mix.inputs[0])
    links.new(mix.outputs[2], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0


def make_camera(cam: dict):
    origin, forward, right, up, _half = camera_frame(cam)
    data = bpy.data.cameras.new(cam["id"])
    data.sensor_fit = "VERTICAL"
    data.sensor_height = 24.0
    data.lens = 12.0 / math.tan(math.radians(cam["fov_deg"]) / 2)
    data.clip_start, data.clip_end = 0.01, 1000.0
    ob = bpy.data.objects.new(cam["id"], data)
    bpy.context.scene.collection.objects.link(ob)
    basis = Matrix(((right.x, up.x, -forward.x), (right.y, up.y, -forward.y),
                    (right.z, up.z, -forward.z)))
    ob.matrix_world = Matrix.Translation(origin) @ basis.to_4x4()
    return ob


def main() -> None:
    (arg,) = runner.script_args()
    spec = json.loads(arg)
    import_layout(spec["glb"])
    scene = bpy.context.scene
    values = spec["values"]
    given = {li["id"]: value_of(values, li["layer"]) for li in spec["lights"]}
    for light in spec["lights"]:
        if light["kind"] != "hemisphere":
            make_light(light, given)
    set_world(spec["lights"], values)
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = spec["samples"]
    scene.cycles.use_denoising = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.display_settings.display_device = "sRGB"
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    for cam in spec["cameras"]:
        scene.camera = make_camera(cam)
        scene.render.resolution_x, scene.render.resolution_y = cam["width"], cam["height"]
        scene.render.resolution_percentage = 100
        scene.render.filepath = str(out / f"blender-{cam['id']}.png")
        bpy.ops.render.render(write_still=True)
        print(f"RENDER blender-{cam['id']}.png", flush=True)
    runner.done()


main()
