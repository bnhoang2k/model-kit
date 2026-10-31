"""Build bar 02, the redbud, from its growth rules. System Python; it starts Blender itself.

    MODEL_KIT_BLENDER=/path/to/blender uv run python bars/02-redbud/build.py

1. Grows `Redbud` (5.0 m, seed 6, full) and `Redbud_Lite` (5.0 m, seed 7, lite) bud by bud
   (`model_kit.core.grow.buds.grow_buds`) from `redbud.rules.json` merged with the kit file
   (`KIT_RULES`), prints each grow time and saves both skeletons. Seeds 1 to 4 fitted
   `apical.control`; the bar's seeds are held out (5 and up).
2. Runs `model_kit/blender/grow_build.py` in Blender: it makes the pack with Draco and writes
   `redbud.pack.json` and `redbud.glb` here, then exports the pieces into a scene folder.
3. `model_kit.bars_build.install_bar` stages the manifests (`style.json`, an empty `clips.json`,
   the pack), copies the skeletons, the rules file (into `rules/`, where `reference-range` reads
   its ranges) and this script into the scene folder, writes the node table of the .glb beside
   them, and puts the folder in place of `scene/`.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import model_kit
from model_kit.bars_build import BuildFailed, install_bar
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import save_skeleton
from model_kit.core.scene import RULES_DIR, SKELETON_SUFFIX

HERE = Path(__file__).resolve().parent
PACK = "redbud"
RULES = HERE / "redbud.rules.json"
STYLE = HERE / "style.json"
GROW_BUILD = Path(model_kit.__file__).resolve().parent / "blender" / "grow_build.py"
TIMEOUT = 900
SPACING = 5.0  # model units between the pieces in the pack: each is under 4 units wide

# Colours of the roles (linear, 0 to 1). Bark and Twig are the dark brown and nearly black of
# rule row bark.colour; Leaf is the green of row leaves.colour (both notes). The bud grower draws
# one leaf shade, Leaf; LeafB and LeafC are kept for a grower that draws more.
COLOURS = {
    "Bark": [0.11, 0.075, 0.06],
    "Twig": [0.06, 0.045, 0.04],
    "Leaf": [0.17, 0.35, 0.10],
    "LeafB": [0.23, 0.43, 0.13],
    "LeafC": [0.12, 0.28, 0.09],
}

DONE_FULL = ("At the page camera the crown reads as a loose, irregular dome of leaf clusters, "
             "a little taller than wide, on a short trunk that forks low and flares into the soil. "
             "Gaps between the clusters show dark limbs. Every limb sits on its parent and every "
             "leaf on its shoot.")
DONE_LITE = ("At the page camera the same loose, irregular dome of leaf clusters on a short, low-"
             "forking trunk. Its leaves are fewer and plainer, so many copies stay cheap. Every "
             "limb sits on its parent and every leaf on its shoot.")

# name, height in metres, seed, lite, size class, what done looks like. Seeds 5 to 8 all pass
# reference-range (README); 6 and 5 are picked so the two pieces are different trees.
PIECES = [
    ("Redbud", 5.0, 6, False, "hero", DONE_FULL),
    ("Redbud_Lite", 5.0, 7, True, "piece", DONE_LITE),
]
ORIGIN = "centre of the base"

# The nearest and farthest distances each piece is seen from, in metres, and why (the pack's `view`;
# the README's "Nearest view" section).
VIEWS = {
    "Redbud": {"nearest_m": 5.0, "why": "the bar scene frames the whole 5 m tree, and a person "
                                         "stands back about one tree height to see a small tree "
                                         "whole",
                "farthest_m": 20.0, "farthest_why": "the full tree hands over to Redbud_Lite at "
                                                    "about four tree heights, 20 m (the "
                                                    "README's Nearest view), so it is not "
                                                    "seen from farther"},
    "Redbud_Lite": {"nearest_m": 20.0, "why": "the copy for scenes that repeat many trees, seen "
                                               "as a backdrop from about four tree heights, where "
                                               "single leaves no longer read",
                    "farthest_m": 50.0, "farthest_why": "an estimate from how it is used, to be "
                                                        "confirmed: the far end of the backdrop "
                                                        "of many trees seen from the scene's "
                                                        "camera; the README says only where the "
                                                        "backdrop begins"},
}


def build() -> None:
    style = json.loads(STYLE.read_text())
    upm = style["units_per_metre"]
    rules = load_rules(RULES, kit=KIT_RULES)
    with tempfile.TemporaryDirectory(prefix="redbud-bar-") as tmp:
        tmp = Path(tmp)
        skeletons = {}
        for name, height_m, seed, lite, _, _ in PIECES:
            skeletons[name] = tmp / f"{name}{SKELETON_SUFFIX}"
            start = time.perf_counter()
            sk = grow_buds(rules, piece=name, height_m=height_m, seed=seed, lite=lite,
                           units_per_metre=upm)
            print(f"grew {name} in {time.perf_counter() - start:.1f} s", flush=True)
            save_skeleton(sk, skeletons[name])

        def args(out: Path, scene: Path) -> list[str]:
            spec = {
                "pack": PACK, "units_per_metre": upm, "spacing": SPACING, "rules": str(RULES),
                "kit": str(KIT_RULES), "style": str(STYLE),
                "out": str(out), "scene": str(scene), "colours": COLOURS,
                "pieces": [{"skeleton": str(skeletons[name]), "size_class": size_class,
                            "done": done, "origin": ORIGIN, "view": VIEWS[name]}
                           for name, _, _, _, size_class, done in PIECES],
            }
            (tmp / "spec.json").write_text(json.dumps(spec))
            return [str(tmp / "spec.json")]

        def add_skeletons(scene: Path) -> None:
            for name, path in skeletons.items():
                shutil.copyfile(path, scene / "pieces" / PACK / f"{name}{SKELETON_SUFFIX}")
            (scene / RULES_DIR).mkdir()
            shutil.copyfile(RULES, scene / RULES_DIR / RULES.name)

        install_bar(HERE, PACK, tmp=tmp, script=GROW_BUILD, style=STYLE, args=args,
                    sources=[Path(__file__).resolve()], timeout=TIMEOUT,
                    add_to_scene=add_skeletons)


if __name__ == "__main__":
    try:
        build()
    except BuildFailed as exc:
        sys.exit(str(exc))
