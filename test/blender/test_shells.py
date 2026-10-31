"""pose_arrays: the body as labelled points and triangles, on a toy rig."""

import json

import numpy as np
import pytest

from model_kit.blender.runner import run_blender

TIMEOUT = 240

pytestmark = pytest.mark.blender

# "torso": the four low vertices lean to body, the four high ones to root, so its two end quads
# are single-group and its four side quads span both. "pad": all hand.R. "knob": a mesh whose
# vertices have no deform weight at all. The rig is moved, so a point in the wrong axes shows.
SCRIPT = """
import json
from pathlib import Path

import bpy
from model_kit.blender import runner, shells
from model_kit.blender._selftest_fk import build_rig, cube

out = Path(runner.script_args()[0])
arm = build_rig("Rig")
torso = cube("torso", (0.0, 0.05, 0.7), (0.05, 0.05, 0.05))
pad = cube("pad", (0.35, 0.1, 1.05), (0.05, 0.05, 0.05))
knob = cube("knob", (0.1, 0.0, 0.1), (0.05, 0.05, 0.05))
for obj in (torso, pad, knob):
    obj.matrix_world = arm.matrix_world.copy()
    obj.modifiers.new("Armature", "ARMATURE").object = arm
body, root = torso.vertex_groups.new(name="body"), torso.vertex_groups.new(name="root")
body.add(range(4), 1.0, "REPLACE")
root.add(range(4, 8), 1.0, "REPLACE")
pad.vertex_groups.new(name="hand.R").add(range(8), 1.0, "REPLACE")
shell = pad.modifiers.new("Outline", "SOLIDIFY")
shell.thickness = 0.01


def dump(**kwargs):
    got = shells.pose_arrays(arm, hide_modifiers=["Outline"], **kwargs)
    return {k: v.tolist() for k, v in got.items()}


result = {
    "plain": dump(),
    "relabel": dump(relabel={"pad": "eyes"}),
    "leave": dump(leave_groups=["root", "eyes"], relabel={"pad": "eyes"}),
    "shell_on": None,
}
try:
    shells.pose_arrays(arm)
except Exception as exc:
    result["shell_on"] = f"{type(exc).__name__}: {exc}"
try:
    shells.pose_arrays(arm, hide_modifiers=["Outlne"])
except Exception as exc:
    result["typo"] = f"{type(exc).__name__}: {exc}"
try:
    shells.pose_arrays(arm, relabel={"nothing": "x"})
except Exception as exc:
    result["unknown"] = f"{type(exc).__name__}: {exc}"
out.write_text(json.dumps(result))
runner.done()
"""


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("shells")
    script, out = tmp / "script.py", tmp / "out.json"
    script.write_text(SCRIPT)
    result = run_blender(script, [str(out)], timeout=TIMEOUT)
    assert result.ok, result.log
    return json.loads(out.read_text())


def _arrays(raw: dict) -> dict:
    return {"points": np.array(raw["points"]).reshape(-1, 3),
            "labels": np.array(raw["labels"], dtype=str),
            "tris": np.array(raw["tris"]).reshape(-1, 3, 3),
            "tri_labels": np.array(raw["tri_labels"], dtype=str)}


def test_pose_arrays_labels(toy):
    got = _arrays(toy["plain"])
    # Three cubes of 8 vertices and 12 triangles. A vertex with no deform weight is "".
    assert got["points"].shape == (24, 3) and got["tris"].shape == (36, 3, 3)
    # Meshes come sorted by name: knob, pad, torso.
    assert list(got["labels"]) == [""] * 8 + ["hand.R"] * 8 + ["body"] * 4 + ["root"] * 4
    tri = got["tri_labels"]
    assert list(tri[:12]) == [""] * 12                          # knob: all "" corners
    assert list(tri[12:24]) == ["hand.R"] * 12                  # pad
    assert sorted(tri[24:]) == ["", "", "", "", "", "", "", "", "body", "body", "root", "root"]

    # A triangle that spans two groups gets "": the torso's side quads.
    torso = tri[24:]
    assert (torso == "").sum() == 8

    # relabel overrides the strongest group for all of one mesh's vertices and triangles.
    relabelled = _arrays(toy["relabel"])
    assert list(relabelled["labels"][8:16]) == ["eyes"] * 8
    assert list(relabelled["tri_labels"][12:24]) == ["eyes"] * 12
    assert list(relabelled["labels"][16:]) == list(got["labels"][16:])

    # leave_groups drops those vertices and every triangle that touches one.
    left = _arrays(toy["leave"])
    assert left["points"].shape == (12, 3)                      # knob "" (8) + torso body (4)
    assert set(left["labels"]) == {"", "body"}
    assert left["tris"].shape == (14, 3, 3)                     # knob 12 + the torso's 2 end tris
    assert sorted(left["tri_labels"]) == [""] * 12 + ["body"] * 2
    np.testing.assert_array_equal(left["points"][8:], got["points"][16:20])


def test_pose_arrays_in_gltf_axes(toy):
    got = _arrays(toy["plain"])
    # The rig stands at Blender (0.3, -0.2, 0.5), turned and scaled by 1.5; whatever it does,
    # glTF up is Y, so the torso (a cube about 0.1 wide) sits above the knob in Y.
    knob, torso = got["points"][:8], got["points"][16:]
    assert torso[:, 1].mean() > knob[:, 1].mean()


def test_pose_arrays_raises_on_a_typo_or_a_modifier_left_on(toy):
    assert "ManifestError" in toy["shell_on"] and "hide_modifiers" in toy["shell_on"]
    assert "ManifestError" in toy["typo"] and "Outlne" in toy["typo"]
    assert "ManifestError" in toy["unknown"] and "nothing" in toy["unknown"]
