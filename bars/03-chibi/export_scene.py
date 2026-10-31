"""Export the chibi's scene folder, the input of `model-kit gate`, from its built .blend.

    blender --background chibi.blend --python export_scene.py -- <scene folder>

Run `build.py` first: it writes `chibi.blend` and `chibi.pack.json` into its out folder. This
script reads the pack from the folder of the opened .blend (the one `build.py` just wrote, with
its measured fields), and `style.json` and `clips.json` from beside this script. It stages the
manifests, then writes the piece's triangles, the rig and the samples of `idle` (every half
frame, from rest, NLA off) into a fresh folder next to <scene folder>, writes the node table of
`chibi.glb` (beside the .blend), copies `build.py` into `sources/`, and only then replaces
<scene folder> whole. The scene folder must be named
`scene`: the script deletes what is there, so it refuses any other name. Run it through
`model_kit.blender.runner.run_blender`.
"""

import shutil
import tempfile
from pathlib import Path

import bpy

from model_kit.blender import export, runner
from model_kit.core.manifest import load_clips, load_pack
from model_kit.core.scene import stage_manifests, write_nodes
from model_kit.core.units import load_style

HERE = Path(__file__).resolve().parent
PIECE = "chibi"
CLIP = "idle"
SCENE_NAME = "scene"


def main() -> None:
    out = Path(runner.script_args()[0]).resolve()
    if out.name != SCENE_NAME:
        raise ValueError(f"the scene folder must be named {SCENE_NAME!r}, not {out.name!r}: "
                         f"this script deletes the folder it is given ({out})")
    out.parent.mkdir(parents=True, exist_ok=True)
    pack_path = Path(bpy.data.filepath).parent / "chibi.pack.json"
    style_path, clips_path = HERE / "style.json", HERE / "clips.json"
    pack = load_pack(pack_path, load_style(style_path))
    clip = {c.name: c for c in load_clips(clips_path)}[CLIP]
    armature = bpy.data.objects[pack.pieces[PIECE].rig.armature]

    temp = Path(tempfile.mkdtemp(prefix=".scene-", dir=out.parent))
    try:
        stage_manifests(temp, style=style_path, clips=clips_path, packs=[pack_path])
        export.export_pieces(pack, temp, pieces=[PIECE])
        export.export_rig(armature, pack, PIECE, temp, clips=[CLIP])
        export.export_samples(armature, bpy.data.actions[CLIP], pack, PIECE, clip, temp,
                              step=0.5)
        write_nodes(temp, pack.name, pack_path.with_name(f"{pack.name}.glb"))
        (temp / "sources").mkdir()
        shutil.copyfile(HERE / "build.py", temp / "sources" / "build.py")
        if out.exists():
            shutil.rmtree(out)
        temp.rename(out)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    runner.done()


main()
