"""Bar 03: the chibi built from bars/03-chibi/chibi.pack.json by bars/03-chibi/build.py."""

import dataclasses
import json
from pathlib import Path

import pytest

from model_kit.blender.runner import frozen_copy, run_blender
from model_kit.core.glb import read_glb_json
from model_kit.core.manifest import load_clips, load_pack
from model_kit.core.units import load_style

BAR = Path(__file__).resolve().parents[2] / "bars" / "03-chibi"
TIMEOUT = 600

pytestmark = pytest.mark.blender

# RECIPE.md section 1: the 16 bones and their parents.
BONES = {
    "root": None, "body": "root", "head": "body",
    "arm.L": "body", "forearm.L": "arm.L", "hand.L": "forearm.L",
    "arm.R": "body", "forearm.R": "arm.R", "hand.R": "forearm.R", "prop.R": "hand.R",
    "leg.L": "root", "knee.L": "leg.L", "ankle.L": "knee.L",
    "leg.R": "root", "knee.R": "leg.R", "ankle.R": "knee.R",
}
GROUPS = {"head", "body", "arm.L", "arm.R", "forearm.L", "forearm.R", "hand.L", "hand.R",
          "knee.L", "knee.R", "ankle.L", "ankle.R"}

CHECK = """
import json
from pathlib import Path

import bpy
import numpy as np

from model_kit.blender import bind, export, runner
from model_kit.core import geom
from model_kit.core.geom import SurfaceIndex
from model_kit.core.manifest import load_pack
from model_kit.core.sight import view_floor
from model_kit.core.units import load_style

pack_path, style_path, scene_dir = runner.script_args()
pack = load_pack(Path(pack_path), load_style(Path(style_path)))
piece = pack.pieces["chibi"]
files = export.export_pieces(pack, Path(scene_dir), pieces=["chibi"])
npz = np.load(files[0])
parts = {}
for name in npz.files:
    tris = npz[name]
    parts[name] = {"closed": bool(geom.is_closed(tris)), "volume": float(geom.signed_volume(tris)),
                   "y_min": float(tris[..., 1].min()), "y_max": float(tris[..., 1].max())}

away = SurfaceIndex(npz["head"]).closest(npz["eye_L"].reshape(-1, 3)).dist
eye = {"lift": view_floor(piece.view, pack.units_per_metre), "gap_lo": float(away.min()),
       "gap_hi": float(away.max())}

armatures = [o for o in bpy.data.objects if o.type == "ARMATURE"]
assert len(armatures) == 1
armature = armatures[0]
deform = {b.name for b in armature.data.bones if b.use_deform}
used, unweighted, skinned = set(), [], []
for ob in bpy.data.objects:
    if ob.type != "MESH":
        continue
    skinned.append([ob.name, [m.object.name for m in ob.modifiers if m.type == "ARMATURE"
                              and m.object]])
    names = {g.index: g.name for g in ob.vertex_groups}
    for v in ob.data.vertices:
        got = {names[g.group] for g in v.groups if g.weight > 0 and names[g.group] in deform}
        if not got:
            unweighted.append([ob.name, v.index])
        used |= got

radii = {name: bind.effector_radius(armature, chain, piece.rig.contact_groups)
         for name, chain in piece.rig.chains.items()}
bones = {b.name: (b.parent.name if b.parent else None) for b in armature.data.bones}
print("RESULT " + json.dumps({"parts": parts, "eye": eye, "used": sorted(used), "unweighted": unweighted[:5],
                              "n_unweighted": len(unweighted), "skinned": skinned,
                              "radii": radii, "bones": bones, "armature": armature.name}))
runner.done()
"""


def _run(script: Path, args: list, *, blend: Path | None = None):
    result = run_blender(script, [str(a) for a in args], blend=blend, timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    return result


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build once from the authored pack, again from that output, and inspect the first build."""
    root = tmp_path_factory.mktemp("chibi")
    one, two = root / "one", root / "two"
    _run(BAR / "build.py", [one])
    _run(BAR / "build.py", [two, one / "chibi.pack.json"])
    script = root / "check.py"
    script.write_text(CHECK)
    copy = frozen_copy(one / "chibi.blend", root / "frozen")
    result = _run(script, [one / "chibi.pack.json", BAR / "style.json", root / "scene"],
                  blend=copy)
    (line,) = [ln for ln in result.log.splitlines() if ln.startswith("RESULT ")]
    return {"one": one, "two": two, "check": json.loads(line[len("RESULT "):])}


def _pack(path: Path):
    return load_pack(path, load_style(BAR / "style.json"))


def test_output_equals_authored_but_measured(built):
    authored = _pack(BAR / "chibi.pack.json")
    written = _pack(built["one"] / "chibi.pack.json")
    assert (written.name, written.units_per_metre, written.spacing) == (
        authored.name, authored.units_per_metre, authored.spacing)
    a, w = authored.pieces["chibi"], written.pieces["chibi"]
    assert w.measured["kind"] == "character"
    # The folder's own pack.json may already hold an earlier build's `measured`; it is not read.
    assert dataclasses.replace(w, measured={}) == dataclasses.replace(a, measured={})


def test_second_build_is_byte_identical(built):
    first = (built["one"] / "chibi.pack.json").read_bytes()
    assert (built["two"] / "chibi.pack.json").read_bytes() == first


def test_eight_closed_outward_parts_and_two_printed_eyes(built):
    parts = built["check"]["parts"]
    authored = _pack(BAR / "chibi.pack.json").pieces["chibi"]
    assert len(parts) == 10
    assert sorted(parts) == sorted(p.name for p in authored.parts)
    shapes = {p.name: p.shape for p in authored.parts}
    assert sorted(n for n, s in shapes.items() if s == "decal") == ["eye_L", "eye_R"]
    for name, p in parts.items():
        if shapes[name] == "decal":
            assert not p["closed"], name  # printed on the head: an open patch
            continue
        assert p["closed"], name
        assert p["volume"] > 0, name


def test_the_eyes_stand_off_the_head_by_the_contact_length(built):
    """A decal is lifted by the piece's contact length (one arcminute at its nearest view), not a
    share of a thickness: every vertex of the left eye lies that far off the head, up to the
    rounding of the saved coordinates (COMPARE_TOL)."""
    eye = built["check"]["eye"]
    assert eye["lift"] == pytest.approx(0.378e-3 * 0.747, rel=1e-3)
    assert eye["gap_lo"] == pytest.approx(eye["lift"], abs=1e-6)
    assert eye["gap_hi"] == pytest.approx(eye["lift"], abs=1e-6)


def test_height_and_head_span(built):
    parts = built["check"]["parts"]
    top = max(p["y_max"] for p in parts.values())
    bottom = min(p["y_min"] for p in parts.values())
    assert top - bottom == pytest.approx(0.998, abs=0.005)
    assert parts["head"]["y_min"] == pytest.approx(0.42, abs=0.005)
    assert parts["head"]["y_max"] == pytest.approx(1.0, abs=0.005)


def test_rig_bones_and_parents(built):
    assert built["check"]["bones"] == BONES
    authored = _pack(BAR / "chibi.pack.json").pieces["chibi"]
    assert {b.name: b.parent for b in authored.rig.bones} == BONES
    assert authored.rig.armature == built["check"]["armature"] == "chibi_rig"


def test_vertex_groups_and_weights(built):
    check = built["check"]
    assert set(check["used"]) == GROUPS
    assert check["n_unweighted"] == 0, check["unweighted"]
    assert len(check["skinned"]) == 10
    assert all(mods == [check["armature"]] for _, mods in check["skinned"])


def test_effector_radii_are_the_measured_ones(built):
    chains = _pack(built["one"] / "chibi.pack.json").pieces["chibi"].rig.chains
    authored = _pack(BAR / "chibi.pack.json").pieces["chibi"].rig.chains
    assert sorted(built["check"]["radii"]) == ["arm.L", "arm.R", "leg.L", "leg.R"]
    for name, measured in built["check"]["radii"].items():
        assert chains[name].effector_radius == pytest.approx(measured, abs=1e-6)
        assert authored[name].effector_radius == pytest.approx(measured, abs=1e-6)


def test_glb_has_one_skin_and_the_idle_animation(built):
    gltf = read_glb_json(built["one"] / "chibi.glb")
    assert len(gltf["skins"]) == 1
    assert [a["name"] for a in gltf["animations"]] == ["idle"]
    # The scene runs at the clip's 30 fps, so frames 1..73 land at 1/30 s .. 73/30 s.
    (anim,) = gltf["animations"]
    times = [gltf["accessors"][s["input"]] for s in anim["samplers"]]
    assert min(t["min"][0] for t in times) == pytest.approx(1 / 30, abs=1e-6)
    assert max(t["max"][0] for t in times) == pytest.approx(73 / 30, abs=1e-6)


def test_clips_and_keys_agree():
    (clip,) = load_clips(BAR / "clips.json")
    keys = json.loads((BAR / "idle.keys.json").read_text())
    assert (clip.name, clip.frames, clip.authoring, clip.holds) == ("idle", (1, 73), "keyed", ())
    assert clip.fps == keys["fps"] == 30
    assert list(clip.frames) == keys["frames"]


def test_idle_keeps_the_feet_planted():
    """The breath lifts the body, not the root: every leg hangs from the root, so a keyed root
    location would lift the soles off the floor."""
    keys = json.loads((BAR / "idle.keys.json").read_text())
    root = [c for c in keys["curves"] if c["bone"] == "root" and c["path"] == "location"]
    assert root and all(k[1] == 0.0 for c in root for k in c["keys"])
    lift = next(c for c in keys["curves"]
                if c["bone"] == "body" and c["path"] == "location" and c["index"] == 1)
    assert max(k[1] for k in lift["keys"]) > 0.0
