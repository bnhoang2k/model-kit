"""Build bar 05, the desk lamp and its power strip. System Python; it starts Blender itself.

    MODEL_KIT_BLENDER=/path/to/blender uv run python bars/05-lamp/build.py

Runs `lamp_build.py` in Blender: it builds the two pieces, makes the pack, and writes
`lamp.pack.json` and `lamp.glb` into a temporary folder, then exports the pieces' triangles and
the layout `desk` into a scene folder. `model_kit.bars_build.install_bar` stages the manifests
(an empty `clips.json`) around them, copies this script and `lamp_build.py` into the scene's
`sources/`, and puts the scene in place of `scene/`, and only then copies the pack and the .glb
into the bar. Then the light bake of `stage.json` runs on that scene and writes `scene/bake/`.
"""

import sys
import tempfile
import time
from pathlib import Path

from model_kit import bake
from model_kit.bars_build import BuildFailed, install_bar
from model_kit.core.errors import ManifestError

HERE = Path(__file__).resolve().parent
PACK = "lamp"
STYLE = HERE / "style.json"
STAGE = HERE / "stage.json"
LAMP_BUILD = HERE / "lamp_build.py"
TIMEOUT = 900


def build() -> None:
    with tempfile.TemporaryDirectory(prefix="lamp-bar-") as tmp:
        install_bar(HERE, PACK, tmp=Path(tmp), script=LAMP_BUILD, style=STYLE,
                    args=lambda out, scene: [str(out), str(scene), str(STYLE)],
                    sources=[Path(__file__).resolve(), LAMP_BUILD], timeout=TIMEOUT)
    # `install_bar` replaced `scene/`, so the bake goes in after it: it writes `scene/bake/`.
    started = time.monotonic()
    bake.run_bake(bake.plan_bake(HERE / "scene", STAGE), replace=True, timeout=bake.BAKE_TIMEOUT)
    print(f"bake wall time {time.monotonic() - started:.0f} s")


if __name__ == "__main__":
    try:
        build()
    except (BuildFailed, ManifestError) as exc:
        sys.exit(str(exc))
