"""Build bar 06, the toy letter sorter. System Python; it starts Blender itself.

    MODEL_KIT_BLENDER=/path/to/blender uv run python bars/06-sorter/build.py

Runs `machine_build.py` in Blender in the toy style, with this folder as the machine folder:
`resolve.py` places every part of `parts.py` on its host and derives the toy with the numbers in
`toy.py`, and `machine_build.py` turns each placed part into a closed solid, makes the pack and
writes `sorter_toy.pack.json` and `sorter_toy.glb` into a temporary folder, then exports the piece
and the layout `floor` into a scene folder. `model_kit.bars_build.install_bar` stages the
manifests (an empty `clips.json`) around them, copies the five scripts into the scene's
`sources/`, and puts the scene in place of `scene/`, and only then copies the pack and the .glb
into the bar.
"""

import sys
import tempfile
from pathlib import Path

from model_kit.bars_build import BuildFailed, install_bar

HERE = Path(__file__).resolve().parent
PACK = "sorter_toy"
LOOK = "toy"
STYLE = HERE / "style.json"
SCRIPT = HERE / "machine_build.py"
SOURCES = [Path(__file__).resolve(), SCRIPT, HERE / "parts.py", HERE / "resolve.py", HERE / "toy.py"]
TIMEOUT = 900


def build() -> None:
    with tempfile.TemporaryDirectory(prefix="sorter-bar-") as tmp:
        install_bar(HERE, PACK, tmp=Path(tmp), script=SCRIPT, style=STYLE,
                    args=lambda out, scene: [str(out), str(scene), str(STYLE), LOOK, str(HERE)],
                    sources=SOURCES, timeout=TIMEOUT)


if __name__ == "__main__":
    try:
        build()
    except BuildFailed as exc:
        sys.exit(str(exc))
