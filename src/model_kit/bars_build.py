"""The shared half of a bar's `build.py`: run the Blender build, stage the scene, install it.

System Python only. A bar's `build.py` makes what is its own (the Blender arguments, any files
its scene folder also needs) and calls `install_bar` for the rest:

1. runs the bar's Blender script and checks its log (Blender exits 0 after a Python traceback);
2. stages the manifests (`style.json`, an empty `clips.json`, the pack) into the scene folder the
   script exported its pieces into;
3. lets the caller add its own files to that folder (`add_to_scene`);
4. writes the node table of the .glb, copies the bar's build scripts into `sources/`, and checks
   that the folder loads as a whole;
5. puts the folder in place of the bar's `scene/`;
6. only then copies the pack and the .glb into the bar, so a bar never holds a new pack beside
   an old scene.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from model_kit.blender.runner import run_blender
from model_kit.core.scene import load_scene, stage_manifests, write_nodes


class BuildFailed(RuntimeError):
    """The bar's Blender build failed: it exited badly, printed a traceback, or never reached its
    last line. The message holds what to look at."""


def install_bar(bar: Path, pack: str, *, tmp: Path, script: Path, style: Path,
                args: Callable[[Path, Path], list[str]], sources: Sequence[Path],
                timeout: float,
                add_to_scene: Callable[[Path], None] | None = None) -> None:
    """Build the bar in `bar` and install its pack, .glb and scene.

    `tmp` is a scratch folder (the caller owns it). `args(out, scene)` gives the arguments of the
    Blender script `script`: it writes `<pack>.pack.json` and `<pack>.glb` into `out` and exports
    the pieces into `scene`. `sources` are the scripts copied into `scene/sources/`. `add_to_scene`
    is called with the scene folder after the manifests are staged. Raises `BuildFailed` when the
    Blender build fails; nothing in the bar changes then.
    """
    out, scene = tmp / "out", tmp / "scene"
    result = run_blender(script, args(out, scene), timeout=timeout)
    print(result.log)
    if not result.ok or "Traceback" in result.log or "MODEL_KIT_DONE" not in result.log:
        raise BuildFailed("the Blender build failed; its log is above")

    (tmp / "clips.json").write_text(json.dumps({"clips": {}}))
    stage_manifests(scene, style=style, clips=tmp / "clips.json",
                    packs=[out / f"{pack}.pack.json"])
    if add_to_scene is not None:
        add_to_scene(scene)
    write_nodes(scene, pack, out / f"{pack}.glb")
    (scene / "sources").mkdir()
    for source in sources:
        shutil.copyfile(source, scene / "sources" / source.name)
    load_scene(scene)  # the folder must load as a whole before it replaces the old one

    old = bar / "scene"
    if old.exists():  # keep the old folder, outside the bar, until this run ends
        shutil.move(old, tmp / "scene.before")
    shutil.move(scene, old)
    # the pack and glb change only after the new scene is in place, so the bar never holds a
    # new pack beside an old scene
    shutil.copyfile(out / f"{pack}.pack.json", bar / f"{pack}.pack.json")
    shutil.copyfile(out / f"{pack}.glb", bar / f"{pack}.glb")
