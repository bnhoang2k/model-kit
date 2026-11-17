"""Bar 05's real stage, baked in `full` mode into a temporary copy, then gated.

The committed bake of the bar is `indirect` mode (`bars/05-lamp/build.py` writes it). This test
bakes the other mode, `full`, of the same stage into a copy of the bar's scene, so the shipped
folder is never touched, with few samples and few texels per metre so that it finishes fast.
It checks that `bake.json` is written and that `bake-files` and `lightmap-overlap` are quiet.
`lightmap-density` and `seam-jump` are not asserted: at this size they measure the tiny bake's
own noise and resolution, and a hard direct light needs far more pixels to meet the density rule.
"""

import json
import shutil
from pathlib import Path

import pytest

from model_kit import bake
from model_kit.core import gate

pytestmark = pytest.mark.blender

BAR = Path(__file__).resolve().parents[2] / "bars" / "05-lamp"
QUIET_CHECKS = ("bake-files", "lightmap-overlap")


@pytest.fixture(scope="module")
def baked(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("lamp")
    shutil.copytree(BAR / "scene", root / "scene", ignore=shutil.ignore_patterns("bake"))
    shutil.copyfile(BAR / "lamp.glb", root / "lamp.glb")
    stage = json.loads((BAR / "stage.json").read_text())
    stage.update(mode="full", samples=8, texels_per_m=32)
    stage.pop("placement_texels_per_m", None)  # the bar's finer lamp would make the test slow
    (root / "stage.json").write_text(json.dumps(stage))
    bake.run_bake(bake.plan_bake(root / "scene", root / "stage.json"), replace=True, timeout=900)
    return root / "scene"


def test_the_bake_writes_bake_json(baked):
    info = json.loads((baked / "bake" / "bake.json").read_text())
    assert info["mode"] == "full" and info["layout"] == "desk"
    assert set(info["layers"]) == {"window", "sky", "ground", "bulb"}
    assert all((baked / "bake" / entry["picture"]).is_file() for entry in info["layers"].values())


def test_file_and_overlap_checks_are_quiet(baked, capsys):
    gate.main([str(baked)])
    err = capsys.readouterr().err
    said = [line for line in err.splitlines()
            if any(line.startswith(f"{name}") or f" {name}" in line.split(":")[0]
                   for name in QUIET_CHECKS)]
    assert not [line for line in said if not line.startswith(("coverage: ", "time: "))], err
