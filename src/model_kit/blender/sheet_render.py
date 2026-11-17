"""Render the cells of a sheet in Blender (Workbench). `model_kit.sheet` runs this script.

It reads one JSON argument:

    rows         [{glb, root, metres_per_unit, height_m, bounds_m, posts, targets}], one per
                 piece: bounds_m is X0, X1, Z0, Z1 in metres (glTF); posts and targets hold
                 one glTF (x, z) in metres per view, planned by `model_kit.sheet`. The post
                 stands beside the piece as each view's camera sees it, so it moves per view.
    views        [[name, turn_deg, tilt_deg]]    turn is a compass bearing (0 looks from the
                                                 glTF front, +Z); tilt is the elevation
    width        the span every cell shares, in metres, across the cell's width
    centre_height  the height in metres every cell centres on, so the ground lines agree
    post_m       the height of the scale post in metres
    cell         [width_px, height_px]
    (a row's `cards` true draws that row with EEVEE; see below)
    out_dir      the folder for the cells, written as r<row>_v<view>.png

Each row gets a fresh empty scene: the glb is imported, everything outside the named root is
deleted, the root is scaled to metres, and a post of `post_m` stands at each view's planned spot. The mesh
is checked against the pack's measured bounds. The look is STUDIO light, material
colour and an object outline.

A card piece (`cards: true` in its row) has leaves that are an image with a clear background, and
Workbench cannot cut an image by its alpha. Its row renders with EEVEE instead: the paper-coloured
world is the only light (even, no shadows) and the .glb's alpha mask clips each card. It has no
outline: Freestyle would draw each card's whole rectangle, not its leaf. Every other row stays
Workbench.
"""

import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

from model_kit.blender import runner
from model_kit.core.errors import ManifestError
from model_kit.sheet import POST_FOOT_RADIUS, eevee_engine

POST_COLOUR = (0.35, 0.37, 0.42, 1.0)
POST_RADIUS = 0.04
CAMERA_DISTANCE = 40.0


def _linear(c: float) -> float:
    """sRGB 0..1 to linear: world colour is linear, and the view transform is Standard."""
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


PAPER = tuple(_linear(c / 255) for c in (246, 245, 241))  # the montage's paper colour


def clear() -> None:
    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)


def meshes_under(root):
    return [ob for ob in [root, *root.children_recursive] if ob.type == "MESH"]


def corners(obs):
    for ob in obs:
        for c in ob.bound_box:
            yield ob.matrix_world @ Vector(c)


def paint_viewport_colours() -> None:
    """Workbench's material colour is the material's viewport colour. The glTF importer leaves
    it grey, so copy each material's base colour there."""
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            continue
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is not None:
            mat.diffuse_color = tuple(bsdf.inputs["Base Color"].default_value)


def add_post(name: str, x: float, y: float, height: float):
    mat = bpy.data.materials.get("sheet_post") or bpy.data.materials.new("sheet_post")
    mat.diffuse_color = POST_COLOUR
    parts = []
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=POST_RADIUS, depth=height,
                                        location=(x, y, height / 2))
    parts.append(bpy.context.object)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=POST_FOOT_RADIUS, depth=0.02,
                                        location=(x, y, 0.01))
    parts.append(bpy.context.object)
    for ob in parts:
        ob.data.materials.append(mat)
        ob.name = f"{name}_{ob.name}"
    return parts


def matte_for_eevee() -> None:
    """EEVEE shades a material from its node tree: make every material matte, and give the post
    (which has only a viewport colour) a node tree of its own."""
    for mat in bpy.data.materials:
        if mat.node_tree is None:
            mat.use_nodes = True
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        if mat.name == "sheet_post":
            bsdf.inputs["Base Color"].default_value = POST_COLOUR
        bsdf.inputs["Metallic"].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0
        bsdf.inputs["Specular IOR Level"].default_value = 0.0


def setup_eevee(scene) -> None:
    """Flat even light for a card row: the paper-coloured world lights every face alike."""
    scene.render.engine = eevee_engine(scene.render.bl_rna.properties["engine"].enum_items.keys())
    world = scene.world
    world.use_nodes = True
    background = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    background.inputs["Color"].default_value = (*PAPER, 1.0)  # the sky, so it is also the light
    background.inputs["Strength"].default_value = 1.0
    for name, value in (("taa_render_samples", 16), ("use_raytracing", False)):
        if hasattr(scene.eevee, name):  # not every Blender version has every EEVEE setting
            setattr(scene.eevee, name, value)
    scene.render.use_freestyle = False  # it would outline each card's whole rectangle


def setup_render(size, eevee: bool = False) -> tuple:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.use_freestyle = False
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "MATERIAL"
    shading.show_object_outline = True
    shading.object_outline_color = (0.02, 0.02, 0.03)
    scene.render.resolution_x, scene.render.resolution_y = size
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.dither_intensity = 0.0
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Standard"
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    world.color = PAPER
    scene.world = world
    if eevee:
        setup_eevee(scene)
    cam_data = bpy.data.cameras.new("SheetCam")
    cam_data.type = "ORTHO"
    cam_data.clip_end = 200
    cam = bpy.data.objects.new("SheetCam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    return scene, cam


def look(cam, target: Vector, turn: float, tilt: float, width: float) -> None:
    a, e = math.radians(turn), math.radians(tilt)
    # glTF (x, y up, z front) is Blender (x, -z, y).
    direction = Vector((math.cos(e) * math.sin(a), -math.cos(e) * math.cos(a), math.sin(e)))
    cam.data.ortho_scale = width  # Blender's ortho scale spans the cell's longer side
    cam.location = target + direction * CAMERA_DISTANCE
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()


def check_against_pack(row: dict, lo: Vector, hi: Vector) -> None:
    """The planned camera and post come from the pack's measured bounds. If the imported mesh
    disagrees, the units are mixed up or the glb is not the pack's, so stop."""
    x0, x1, z0, z1 = row["bounds_m"]
    want = (x0, x1, -z1, -z0, row["height_m"])
    got = (lo.x, hi.x, lo.y, hi.y, hi.z)
    tol = max(0.01, 0.01 * row["height_m"])
    if any(abs(g - w) > tol for g, w in zip(got, want, strict=True)):
        raise ManifestError(
            f"{row['glb']}: the {row['root']!r} mesh spans x {got[0]:.3f}..{got[1]:.3f}, y "
            f"{got[2]:.3f}..{got[3]:.3f}, top {got[4]:.3f} (Blender axes, metres), but its pack "
            f"measured x {want[0]:.3f}..{want[1]:.3f}, y {want[2]:.3f}..{want[3]:.3f}, top "
            f"{want[4]:.3f}")


def render_row(index: int, row: dict, spec: dict) -> None:
    clear()
    bpy.ops.import_scene.gltf(filepath=row["glb"])
    root = bpy.data.objects.get(row["root"])
    if root is None:
        raise ManifestError(f"{row['glb']}: no node named {row['root']!r} to render")
    if root.parent is not None:
        raise ManifestError(
            f"{row['glb']}: {row['root']!r} sits under {root.parent.name!r}; a piece must be a "
            "top-level node, or its parent's scale would be dropped")
    if abs(root.location.z) > 1e-6:
        raise ManifestError(
            f"{row['glb']}: {row['root']!r} sits {root.location.z} off the floor; the sheet "
            "scales pieces about the floor point")
    keep = {root, *root.children_recursive}
    for ob in list(bpy.data.objects):
        if ob not in keep:
            bpy.data.objects.remove(ob, do_unlink=True)
    root.location.x = root.location.y = 0.0
    root.scale = tuple(s * row["metres_per_unit"] for s in root.scale)
    bpy.context.view_layer.update()
    if not meshes_under(root):
        raise ManifestError(f"{row['glb']}: {row['root']!r} holds no mesh")
    paint_viewport_colours()

    pts = list(corners(meshes_under(root)))
    check_against_pack(row, Vector((min(p.x for p in pts), min(p.y for p in pts), 0.0)),
                       Vector((max(p.x for p in pts), max(p.y for p in pts),
                               max(p.z for p in pts))))
    if not len(row["posts"]) == len(row["targets"]) == len(spec["views"]):
        raise ManifestError(
            f"{row['glb']}: {len(row['posts'])} posts and {len(row['targets'])} targets for "
            f"{len(spec['views'])} views")

    scene, cam = setup_render(tuple(spec["cell"]), eevee=bool(row.get("cards")))
    for j, (name, turn, tilt) in enumerate(spec["views"]):
        post_x, post_z = row["posts"][j]
        post = add_post(f"Post{index}_{j}", post_x, -post_z, spec["post_m"])
        if row.get("cards"):
            matte_for_eevee()
        tx, tz = row["targets"][j]
        look(cam, Vector((tx, -tz, spec["centre_height"])), turn, tilt, spec["width"])
        scene.render.filepath = str(Path(spec["out_dir"]) / f"r{index}_v{j}.png")
        bpy.ops.render.render(write_still=True)
        for ob in post:
            bpy.data.objects.remove(ob, do_unlink=True)
        print(f"CELL row {index} view {name} {scene.render.filepath}", flush=True)


def main() -> None:
    (arg,) = runner.script_args()
    spec = json.loads(arg)
    Path(spec["out_dir"]).mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(spec["rows"]):
        render_row(i, row, spec)
    runner.done()


main()
