"""Build bar 04, the sweetgum, from its growth rules. System Python; it starts Blender itself.

    MODEL_KIT_BLENDER=/path/to/blender uv run python bars/04-sweetgum/build.py

1. Grows `Sweetgum` (3.87 m, the 6-year size, seed 7, full) and `Sweetgum_Lite` (3.17 m, the
   5-year size, seed 6, lite) with the bud grower from `sweetgum.rules.json` merged with the kit
   file, and saves both skeletons. Seeds 1 to 4 are the fit seeds of `apical.control`; the bar
   seeds are held out (5 and up).
2. Runs `model_kit/blender/grow_build.py` in Blender: it makes the pack with Draco and writes
   `sweetgum.pack.json` and `sweetgum.glb` here, then exports the pieces into a scene folder.
3. `model_kit.bars_build.install_bar` stages the manifests (`style.json`, an empty `clips.json`,
   the pack), copies the skeletons, the rules file (into `rules/`, which `reference-range`
   reads) and this script into the scene folder, writes the node table of the .glb beside them,
   and puts the folder in place of `scene/`.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import model_kit
from model_kit.bars_build import BuildFailed, install_bar
from model_kit.core.grow import KIT_RULES, grow_buds, load_rules
from model_kit.core.grow.skeleton import save_skeleton
from model_kit.core.scene import RULES_DIR, SKELETON_SUFFIX

HERE = Path(__file__).resolve().parent
PACK = "sweetgum"
RULES = HERE / "sweetgum.rules.json"
STYLE = HERE / "style.json"
GROW_BUILD = Path(model_kit.__file__).resolve().parent / "blender" / "grow_build.py"
TIMEOUT = 900
SPACING = 5.0  # model units between the pieces in the pack: each is under 2 units wide

# Colours of the roles (linear, 0 to 1): grey-brown bark, a lighter twig, three mid greens.
# Colours chosen for the look; no colour source was opened (a taste choice, not a species fact).
COLOURS = {
    "Bark": [0.16, 0.12, 0.09],
    "Twig": [0.20, 0.16, 0.08],
    "Leaf": [0.10, 0.30, 0.08],
    "LeafB": [0.14, 0.36, 0.10],
    "LeafC": [0.07, 0.24, 0.06],
}

DONE_FULL = ("At the page camera the tree reads as one straight central leader with an open, "
             "narrow crown of star leaves, taller than wide, over a short flare into the soil. "
             "Every limb sits on its parent and every leaf on its shoot.")
DONE_LITE = ("At the page camera the tree reads as one straight central leader with an open, "
             "narrow crown, taller than wide. Its leaves are coarser and fewer, so many copies "
             "stay cheap. Every limb sits on its parent and every leaf on its shoot.")

# name, height in metres, seed, lite, size class, what done looks like
PIECES = [
    ("Sweetgum", 3.87, 7, False, "hero", DONE_FULL),
    ("Sweetgum_Lite", 3.17, 6, True, "piece", DONE_LITE),
]
ORIGIN = "centre of the base"

# The nearest and farthest distances each piece is seen from, in metres, and why (the pack's `view`;
# the README's "Viewing distance" section).
VIEWS = {
    "Sweetgum": {"nearest_m": 4.0, "why": "in the bar scene a viewer stands about one tree height "
                                           "away, the nearest distance at which the whole 3.9 m "
                                           "tree fits in view",
                 "farthest_m": 15.0, "farthest_why": "the full tree hands over to Sweetgum_Lite "
                                                     "where a backdrop row begins, about four "
                                                     "tree heights, 15 m (the README's Viewing "
                                                     "distance), so it is not seen from "
                                                     "farther"},
    "Sweetgum_Lite": {"nearest_m": 15.0, "why": "the copy for a backdrop row of many trees, about "
                                                 "four tree heights away, where a coarser leaf no "
                                                 "longer shows",
                      "farthest_m": 40.0, "farthest_why": "an estimate from how it is used, to be "
                                                          "confirmed: the far end of the backdrop "
                                                          "row of many trees seen from the "
                                                          "scene's camera; the README says only "
                                                          "where the row begins"},
}


def build() -> None:
    style = json.loads(STYLE.read_text())
    upm = style["units_per_metre"]
    rules = load_rules(RULES, kit=KIT_RULES)
    with tempfile.TemporaryDirectory(prefix="sweetgum-bar-") as tmp:
        tmp = Path(tmp)
        skeletons = {}
        for name, height_m, seed, lite, _, _ in PIECES:
            skeletons[name] = tmp / f"{name}{SKELETON_SUFFIX}"
            start = time.perf_counter()
            sk = grow_buds(rules, piece=name, height_m=height_m, seed=seed, lite=lite,
                           units_per_metre=upm)
            print(f"{name}: grown in {time.perf_counter() - start:.1f} s")
            save_skeleton(sk, skeletons[name])

        def args(out: Path, scene: Path) -> list[str]:
            spec = {
                "pack": PACK, "units_per_metre": upm, "spacing": SPACING, "rules": str(RULES),
                "style": str(STYLE),
                "out": str(out), "scene": str(scene), "colours": COLOURS,
                "pieces": [{"skeleton": str(skeletons[name]), "size_class": size_class,
                            "done": done, "origin": ORIGIN, "view": VIEWS[name]}
                           for name, _, _, _, size_class, done in PIECES],
            }
            (tmp / "spec.json").write_text(json.dumps(spec))
            return [str(tmp / "spec.json")]

        def add_to_scene(scene: Path) -> None:
            for name, path in skeletons.items():
                shutil.copyfile(path, scene / "pieces" / PACK / f"{name}{SKELETON_SUFFIX}")
            (scene / RULES_DIR).mkdir()
            shutil.copyfile(RULES, scene / RULES_DIR / RULES.name)

        install_bar(HERE, PACK, tmp=tmp, script=GROW_BUILD, style=STYLE, args=args,
                    sources=[Path(__file__).resolve()], timeout=TIMEOUT,
                    add_to_scene=add_to_scene)


if __name__ == "__main__":
    try:
        build()
    except BuildFailed as exc:
        sys.exit(str(exc))
