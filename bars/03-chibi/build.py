"""Build the chibi from its authored chibi.pack.json, in Blender.

    blender --background --python build.py -- <out folder> [<pack.json to read>]

Reads the pack (the authored fields only; `measured` is ignored) and the idle keys, builds the
rigged piece, and writes chibi.pack.json, chibi.glb and chibi.blend into the out folder through
`PackBuilder.finish`. Run it through `model_kit.blender.runner.run_blender`.
"""

from pathlib import Path

import bpy

from model_kit.blender import bind, build, keys, runner, soft
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import load_clips, load_pack
from model_kit.core.units import load_style

HERE = Path(__file__).resolve().parent

# The two flat colours of the unlit look: the part's role picks one (eyes are ink).
ROLES = {
    "Fill": (0.9936, 0.9845, 0.9659),
    "Ink": (0.0132, 0.0101, 0.0073),
}
ROLE_MATERIAL = {"eyes": "Ink"}
CLIP = "idle"


def main() -> None:
    args = runner.script_args()
    out = Path(args[0])
    pack_path = Path(args[1]) if len(args) > 1 else HERE / "chibi.pack.json"

    build.reset()
    mats = build.slots(ROLES)
    build.selftest_axes()

    style = load_style(HERE / "style.json")
    pack = load_pack(pack_path, style)
    piece = pack.pieces["chibi"]
    builder = build.PackBuilder("chibi", style.units_per_metre, out, spacing=2.0,
                                crease_deg=style.crease_deg)
    part_mats = {p.name: mats[ROLE_MATERIAL.get(p.role, "Fill")] for p in piece.parts}
    # The eyes are decals; their lift off the head is the piece's contact length, which the build
    # checks against one depth step at the piece's farthest view (the style's camera).
    _, rig = soft.build_piece(builder, piece, part_mats, style=style)
    # The scene starts at the factory rate (24 fps); the clip is keyed in frames at its own rate.
    # `key_action` raises when idle.keys.json disagrees with clips.json.
    clip = {c.name: c for c in load_clips(HERE / "clips.json")}[CLIP]
    if clip.fps != int(clip.fps):
        raise ManifestError(f"{HERE / 'clips.json'}: clip {CLIP!r} fps {clip.fps} is not whole")
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = int(clip.fps), 1.0
    keys.key_action(rig, CLIP, keys.read_keys(HERE / "idle.keys.json"))

    for name, chain in piece.rig.chains.items():
        radius = bind.effector_radius(rig, chain, piece.rig.contact_groups)
        print(f"EFFECTOR {name} {radius!r}")
    builder.finish(["chibi"])
    runner.done()


main()
