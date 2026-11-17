"""The grower's Blender half: a saved skeleton becomes one object per level and per leaf shade,
registered as a branching piece, with leaf normals that lean to the crown, and a Draco .glb.

Growing never runs in Blender. Each test takes a saved skeleton of bar 02 and saves it in this Python first, then
passes the .npz to the Blender script.
"""

import dataclasses
import io
import json
import struct
from functools import cache
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from model_kit.blender.runner import run_blender
from model_kit.core.checks.branching.skeleton_match import SkeletonMatch
from model_kit.core.grow.buds import MESH_V1, MESH_VERSION, _read
from model_kit.core.grow.card_image import paint_cells
from model_kit.core.grow.cards import CardDials, card_parts
from model_kit.core.grow.mesh import crown_normals, grown_parts, leaf_lean
from model_kit.core.grow.rules import RuleReader, load_rules
from model_kit.core.grow.skeleton import load_skeleton, save_skeleton
from model_kit.core.manifest import View, load_pack
from model_kit.core.scene import SKELETON_SUFFIX, load_scene, stage_manifests
from model_kit.core.sight import Sight, view_floor
from model_kit.core.units import Style, load_style

TIMEOUT = 600
UPM = 0.5
MIN_THICKNESS_M = 0.018
PACK = "grove"
PIECE = "Redbud"
REPO = Path(__file__).resolve().parents[2]
BAR = REPO / "bars" / "02-redbud"
RULES = BAR / "redbud.rules.json"
SKELETONS = BAR / "scene" / "pieces" / "redbud"  # bar 02's saved bud-grown trees
STYLE = Style(UPM, {}, {"min_thickness": MIN_THICKNESS_M})
BAR_STYLE = BAR / "style.json"  # its dials.leaves is the leaf-normal lean
LEAN = leaf_lean(load_style(BAR_STYLE))
# Blender packs a custom normal into two 16-bit numbers measured from a triangle edge. A normal
# that leans sideways within a degree of that edge comes back off by up to 0.011 (measured: every
# stray corner leans within 0.8 degrees of an edge, and all others are within 2.3e-4). So the
# normals must be within NORMALS_TOL at all but NORMALS_STRAY_SHARE of the corners, and none may
# be more than NORMALS_MAX off (about 1 degree).
NORMALS_TOL = 1e-3
NORMALS_STRAY_SHARE = 0.01
NORMALS_MAX = 0.02
DRACO_MAX_BYTES = 4_000_000  # bar 02's redbud is about 2.5 MB with Draco
CREASE_DEG = 35
FULL_NEAREST_M = 5.0  # the nearest view of the full tree, and of the pack these tests build
LITE_NEAREST_M = 20.0  # bar 02's lite tree is seen from farther; its fine roots need that view

pytestmark = pytest.mark.blender


def sight_at(nearest_m: float) -> Sight:
    """What `add_grown` meshes by for a piece seen from `nearest_m` in a 35 degree crease style."""
    return Sight(view_floor(View(nearest_m, "a test tree"), UPM), CREASE_DEG)


SIGHT = sight_at(FULL_NEAREST_M)

# The 6 parts of bar 02's redbud: (name, shape, on, role). Its leaves grow on limbs of every
# level, so the leaf part is on the lowest, the trunk (#3).
PARTS = [
    ("Trunk_Roots", "tube", None, "solid"),
    ("Leaders", "tube", "Trunk_Roots", "solid"),
    ("Branches", "tube", "Leaders", "solid"),
    ("Twigs", "tube", "Branches", "skip_outline"),
    ("Shoots", "tube", "Twigs", "skip_outline"),
    ("Leaves_Leaf", "leaf", "Trunk_Roots", "skip_outline"),
]

BUILD = """
import sys
from pathlib import Path

import bpy
import numpy as np

from model_kit.blender import build, grow, runner
from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.blender.export import export_pieces
from model_kit.core.grow.mesh import crown_normals, grown_parts, leaf_lean
from model_kit.core.grow.rules import RuleReader, load_rules
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.manifest import View
from model_kit.core.sight import Sight, view_floor
from model_kit.core.units import load_style

skeleton, rules, out_dir, scene_dir, draco, style, nearest = runner.script_args()
lean = leaf_lean(load_style(Path(style)))
sk = load_skeleton(Path(skeleton))
build.reset()
mats = build.slots({"Bark": (0.25, 0.17, 0.12), "Twig": (0.1, 0.08, 0.07),
                    "Leaf": (0.4, 0.62, 0.25), "LeafB": (0.5, 0.7, 0.3),
                    "LeafC": (0.3, 0.5, 0.2)})
by_part = {"Trunk_Roots": mats["Bark"], "Leaders": mats["Bark"], "Branches": mats["Bark"],
           "Twigs": mats["Twig"], "Shoots": mats["Twig"], "Leaves_Leaf": mats["Leaf"],
           "Leaves_LeafB": mats["LeafB"], "Leaves_LeafC": mats["LeafC"]}
pack = build.PackBuilder("grove", 0.5, Path(out_dir), None, crease_deg=35)
view = View(float(nearest), "a test tree, seen from the path beside it")
sight = Sight(view_floor(view, pack.units_per_metre), pack.crease_deg)  # what add_grown meshes by
root, apply_normals = grow.add_grown(
    pack, sk, rules=load_rules(Path(rules)), materials=by_part,
    done="A redbud grown from its rules.", size_class="hero", origin="centre of the base", lean=lean,
    view=view)
options = build.DRACO if draco == "draco" else None
written = pack.finish([root.name], glb_options=options, before_export=apply_normals)
print("PARTS", [ob.name for ob in root.children_recursive])

# What the Blender meshes hold: the crown normals, in Blender axes.
leaves = {k: v for k, v in grown_parts(sk, sight).items() if k.startswith("Leaves_")}
turn = BLENDER_TO_GLTF[:3, :3]
for part, want in crown_normals(leaves, lean).items():
    me = bpy.data.objects[build.object_name(root.name, part)].data
    got = np.array([c.vector for c in me.corner_normals])
    dev = np.abs(got - want.reshape(-1, 3) @ turn).max(axis=1)
    print("NORMALS", part, len(got), f"{dev.max():.2e}", int((dev > NORMALS_TOL).sum()))

if scene_dir != "-":
    export_pieces(written, Path(scene_dir), pieces=[root.name])
runner.done()
"""


BUILD = BUILD.replace("NORMALS_TOL", repr(NORMALS_TOL))


@cache
def _redbud():
    return load_skeleton(SKELETONS / "Redbud.skeleton.npz")


@cache
def _lite():
    return load_skeleton(SKELETONS / "Redbud_Lite.skeleton.npz")


def _build(tmp_path: Path, *, draco: bool, scene: bool = False, sk=None):
    """Take bar 02's redbud (or `sk`), save it, then build the pack in Blender, with the view the
    tree is seen from (the lite tree from `LITE_NEAREST_M`). Returns the folders and the log."""
    tree = sk or _redbud()
    skeleton = tmp_path / "redbud.npz"
    save_skeleton(tree, skeleton)
    script = tmp_path / "script.py"
    script.write_text(BUILD)
    out, scene_dir = tmp_path / "out", tmp_path / "scene"
    args = [str(skeleton), str(RULES), str(out), str(scene_dir) if scene else "-",
            "draco" if draco else "plain", str(BAR_STYLE),
            str(LITE_NEAREST_M if tree.lite else FULL_NEAREST_M)]
    result = run_blender(script, args, timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    return out, scene_dir, result.log


@pytest.fixture(scope="module")
def plain(tmp_path_factory):
    """One build without Draco that also exports the piece array."""
    return _build(tmp_path_factory.mktemp("plain"), draco=False, scene=True)


def _glb(path: Path):
    """The JSON and binary chunks of a .glb."""
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    (json_len,) = struct.unpack_from("<I", data, 12)
    js = json.loads(data[20:20 + json_len])
    start = 20 + json_len
    (bin_len,) = struct.unpack_from("<I", data, start)
    return js, data[start + 8:start + 8 + bin_len]


def _accessor(js, binary, index):
    acc = js["accessors"][index]
    view = js["bufferViews"][acc["bufferView"]]
    assert "byteStride" not in view
    dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    return np.frombuffer(binary, dtype, acc["count"] * width, start).reshape(-1, width)


def test_grown_pack_loads(tmp_path):
    out, _, _ = _build(tmp_path, draco=True)
    assert sorted(p.name for p in out.iterdir()) == ["grove.blend", "grove.glb", "grove.pack.json"]
    pack = load_pack(out / "grove.pack.json", STYLE)
    piece = pack.pieces[PIECE]
    assert piece.recipe.kind == "branching"
    assert piece.recipe.source == "redbud.rules.json, seed 6, 5.0 m"
    assert piece.size_class == "hero"
    assert [(p.name, p.shape, p.on, p.role) for p in piece.parts] == PARTS
    assert [p.group for p in piece.parts] == [name for name, *_ in PARTS]
    # One listed row per part: its text from the rules rows, its source the file and the row ids.
    assert [p.listed for p in piece.parts] == [r.id for r in piece.listed]
    rows = {r.id: r for r in piece.listed}
    assert set(rows) == {name.lower() for name, *_ in PARTS}
    for row in rows.values():
        assert row.source.startswith("redbud.rules.json: ")
    sk = _redbud()
    for li, name in enumerate(sk.level_names):
        smallest = min(float(sk.limb_radii(int(i)).min()) for i in np.flatnonzero(sk.level == li))
        assert rows[name.lower()].min_thickness == pytest.approx(2 * smallest / UPM)
    assert all(rows[f"leaves_{s.lower()}"].min_thickness == 0 for s in sk.shades)
    assert piece.measured["tris"] == sum(len(t) for t in grown_parts(sk, SIGHT).values())
    # Draco: the extension is required, and the file is far below the plain one (3.5 MB).
    js, _ = _glb(out / "grove.glb")
    assert "KHR_draco_mesh_compression" in js["extensionsRequired"]
    assert len(js["meshes"]) == len(PARTS)
    assert (out / "grove.glb").stat().st_size < DRACO_MAX_BYTES


def test_a_buds_v2_tree_is_meshed_from_the_view_the_pack_declares(tmp_path):
    """`add_grown` meshes a buds-v2 skeleton (bar 02's, grown from a rules file without
    `tubes.sides`) with the sight of the piece's view (5 m, here) and the pack's crease angle (35):
    the triangles in the pack are those of `grown_parts` for that sight, and not those of the same
    tree meshed the buds-v1 way with fixed sides."""
    v2 = _redbud()
    assert v2.info["mesh"] == MESH_VERSION and not v2.sides.any()
    v1 = dataclasses.replace(v2, sides=np.array([12, 9, 7, 5, 4]), info={**v2.info, "mesh": MESH_V1})
    out, _, _ = _build(tmp_path, draco=False, sk=v2)
    piece = load_pack(out / "grove.pack.json", STYLE).pieces[PIECE]
    assert piece.measured["tris"] == sum(len(t) for t in grown_parts(v2, SIGHT).values())
    assert piece.measured["tris"] != sum(len(t) for t in grown_parts(v1, None).values())


def test_exported_parts_match_skeleton(plain, tmp_path):
    out, scene_dir, _ = plain
    style = tmp_path / "style.json"
    style.write_text(json.dumps({"units_per_metre": UPM, "page_camera": {},
                                 "dials": {"min_thickness": MIN_THICKNESS_M,
                                           "edges": {"bevel": 0, "outline": None,
                                                     "crease_deg": CREASE_DEG,
                                                     "crease_why": "the test style's angle"}}}))
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps({"clips": {}}))
    stage_manifests(scene_dir, style=style, clips=clips, packs=[out / "grove.pack.json"])
    save_skeleton(_redbud(), scene_dir / "pieces" / PACK / f"{PIECE}{SKELETON_SUFFIX}")
    ctx = load_scene(scene_dir)
    assert sorted(ctx.pieces[f"{PACK}:{PIECE}"].parts) == sorted(name for name, *_ in PARTS)
    check = SkeletonMatch()
    assert check.run(ctx) == []
    plants = check.plants(ctx)
    assert plants
    for unit in plants:  # the check can still fail here: some candidate of each unit is caught
        assert unit.candidates, unit.item
        assert any(f.item == unit.item for plant in unit.candidates
                   for f in check.run(plant.ctx, only=plant.scope))


def test_leaf_normals_written(plain):
    out, _, log = plain
    sk = _redbud()
    leaves = {k: v for k, v in grown_parts(sk, SIGHT).items() if k.startswith("Leaves_")}
    want = crown_normals(leaves, LEAN)
    # In Blender: every leaf mesh carries the crown normals.
    lines = [line.split() for line in log.splitlines() if line.startswith("NORMALS ")]
    assert [line[1] for line in lines] == list(want)
    for _, part, count, worst, stray in lines:
        assert int(count) == 3 * len(want[part])
        assert float(worst) < NORMALS_MAX, log
        assert int(stray) < NORMALS_STRAY_SHARE * int(count), log
    # In the shipped .glb: the same, corner by corner.
    js, binary = _glb(out / "grove.glb")
    for part, normals in want.items():
        mesh = next(m for m in js["meshes"] if m["name"] == f"{PIECE}/{part}")
        prim = mesh["primitives"][0]["attributes"]
        idx = _accessor(js, binary, mesh["primitives"][0]["indices"]).ravel()
        pos = _accessor(js, binary, prim["POSITION"])[idx].reshape(-1, 3, 3)
        nor = _accessor(js, binary, prim["NORMAL"])[idx].reshape(-1, 3, 3)
        assert np.abs(pos - grown_parts(sk, SIGHT)[part]).max() < 1e-5
        dev = np.abs(nor - normals).max(axis=-1).ravel()
        assert dev.max() < NORMALS_MAX
        assert (dev > NORMALS_TOL).mean() < NORMALS_STRAY_SHARE


REFUSE = """
from pathlib import Path

import bpy

from model_kit.blender import build, grow, runner
from model_kit.core.errors import ManifestError
from model_kit.core.grow.mesh import LeafLean
from model_kit.core.grow.buds import MESH_VERSION, _read
from model_kit.core.grow.rules import RuleReader, load_rules
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.manifest import View

skeleton, rules, other_rules, out_dir = runner.script_args()
lean = LeafLean(toward_out=0.8, centre_drop=0.35)
sk = load_skeleton(Path(skeleton))
build.reset()
mat = build.slots({"Bark": (0.25, 0.17, 0.12)})["Bark"]
every = {name: mat for name in ("Trunk_Roots", "Leaders", "Branches", "Twigs", "Shoots",
                                "Leaves_Leaf", "Leaves_LeafB", "Leaves_LeafC")}
pack = build.PackBuilder("grove", 0.5, Path(out_dir), None, crease_deg=35)
metres = build.PackBuilder("grove", 1.0, Path(out_dir), None, crease_deg=35)
broken = build.PackBuilder("grove", 0.5, Path(out_dir), None, crease_deg=35)


def refuse_register(*args, **kwargs):
    raise ManifestError("register refused")


broken.register = refuse_register
good = dict(rules=load_rules(Path(rules)), materials=every, done="x", size_class="hero",
            origin="centre of the base", lean=lean, view=View(5.0, "a test tree, seen from the path"))
cases = {
    "material": (pack, dict(good, materials={k: v for k, v in every.items() if k != "Twigs"})),
    "rules": (pack, dict(good, rules=load_rules(Path(other_rules)))),
    "units": (metres, good),
    "size_class": (pack, dict(good, size_class="huge")),
    "view": (pack, {k: v for k, v in good.items() if k != "view"}),
    "register": (broken, good),
}
for label, (builder, kwargs) in cases.items():
    try:
        grow.add_grown(builder, sk, **kwargs)
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)
    print("OBJECTS", label, sorted(ob.name for ob in bpy.data.objects),
          sorted(me.name for me in bpy.data.meshes))
runner.done()
"""


def test_refusals_raise_and_leave_nothing(tmp_path):
    skeleton = tmp_path / "redbud.npz"
    save_skeleton(_redbud(), skeleton)
    script = tmp_path / "script.py"
    script.write_text(REFUSE)
    other = REPO / "test" / "core" / "grow" / "fixtures" / "tiny.rules.json"
    result = run_blender(script, [str(skeleton), str(RULES), str(other), str(tmp_path / "out")],
                         timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert not [line for line in lines if line.startswith("NO RAISE")]
    said = {line.split()[1]: line for line in lines if line.startswith("RAISED ")}
    assert set(said) == {"material", "rules", "units", "size_class", "view", "register"}, result.log
    assert "no view" in said["view"] and "redbud" in said["view"].lower(), result.log
    assert "'Twigs'" in said["material"]
    assert "redbud.rules.json" in said["rules"] and "tiny.rules.json" in said["rules"]
    assert "0.5" in said["units"] and "1.0" in said["units"]
    assert "huge" in said["size_class"]
    assert "register refused" in said["register"]
    # Every refusal, the register one too, leaves no object and no mesh.
    for label in said:
        assert f"OBJECTS {label} [] []" in lines, result.log


def _rows_cited(piece, part):
    return piece.listed[[r.id for r in piece.listed].index(part.listed)].source.split(": ")[1]


@pytest.mark.parametrize("which", ["full", "lite"])
def test_listed_rows_follow_the_params_read(tmp_path, which):
    """Each part cites the rows of its level that a param the grower really read cites; a lite
    tree cites the lite row, not the full-mode rows its swapped params replace."""
    sk = _redbud() if which == "full" else _lite()
    out, _, _ = _build(tmp_path, draco=False, sk=sk)
    piece = load_pack(out / "grove.pack.json", STYLE).pieces[sk.info["piece"]]
    assert piece.recipe.source.endswith(", lite") == (which == "lite")
    # The keys the grower reads in this mode (growing never runs in Blender).
    reader = RuleReader(load_rules(RULES))
    _read(reader, sk.lite)
    rules = reader.rules
    cited = {row for key in reader.read for row in rules.params[key].rows}
    for part in piece.parts:
        words = {"leaves"} if part.shape == "leaf" else set(part.name.lower().split("_"))
        if sk.lite:
            words.add("lite")
        want = [row.id for row in rules.rows.values() if row.level in words and row.id in cited]
        assert _rows_cited(piece, part).split(", ") == want, part.name
    leaf = next(p for p in piece.parts if p.shape == "leaf")
    assert ("lite.leaves" in _rows_cited(piece, leaf)) == (which == "lite")


FIRST = """
from pathlib import Path

import bpy

from model_kit.blender import build, grow, runner
from model_kit.core.errors import ManifestError
from model_kit.core.grow.mesh import LeafLean
from model_kit.core.grow.rules import load_rules
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.manifest import View

skeleton, rules, out_dir = runner.script_args()
sk = load_skeleton(Path(skeleton))
build.reset()
pack = build.PackBuilder("grove", 0.5, Path(out_dir), None, crease_deg=35)


def refused(**kwargs):
    raise ManifestError("check_registration said no")


pack.check_registration = refused
# The materials are missing too: if anything but `check_registration` ran first, it would be the
# one to complain.
try:
    grow.add_grown(pack, sk, rules=load_rules(Path(rules)), materials={}, done="x",
                   size_class="hero", origin="centre of the base",
                   lean=LeafLean(toward_out=0.8, centre_drop=0.35),
                   view=View(5.0, "a test tree, seen from the path"))
    print("NO RAISE")
except ManifestError as exc:
    print("RAISED", exc)
print("OBJECTS", sorted(ob.name for ob in bpy.data.objects))
runner.done()
"""


def test_add_grown_with_bad_view_leaves_no_objects(tmp_path):
    """`add_grown` asks the builder to check the registration before anything else, so a piece
    the builder would refuse (a bad view, say) makes no object. What the builder refuses in a
    view is `test_build_without_view_stops`; the missing view of `add_grown` is the `view` case
    of `test_refusals_raise_and_leave_nothing`."""
    skeleton = tmp_path / "redbud.npz"
    save_skeleton(_redbud(), skeleton)
    script = tmp_path / "script.py"
    script.write_text(FIRST)
    result = run_blender(script, [str(skeleton), str(RULES), str(tmp_path / "out")],
                         timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "RAISED check_registration said no" in lines, result.log
    assert "OBJECTS []" in lines, result.log


LEVELS = """
import dataclasses
import json
from pathlib import Path

import numpy as np

from model_kit.blender import grow, runner
from model_kit.core.errors import ManifestError
from model_kit.core.grow.mesh import grown_parts
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.manifest import View
from model_kit.core.sight import Sight, view_floor

sk = load_skeleton(Path(runner.script_args()[0]))
parts = grown_parts(sk, Sight(view_floor(View(5.0, "a test tree"), 0.5), 35.0))


def leaf_on(tree):
    return {w[0]: w[2] for w in grow._parts_wanted(tree, parts) if w[1] == "leaf"}


def only_hosts_on(levels):
    keep = np.isin(sk.level[sk.host], levels)
    return dataclasses.replace(sk, host=sk.host[keep], shade=sk.shade[keep])


def refusal(tree, grown):
    try:
        grow._parts_wanted(tree, grown)
        return "no raise"
    except ManifestError as exc:
        return str(exc)


print("LEVELS", json.dumps({
    "no_leaves": refusal(only_hosts_on([]), parts),
    "no_part": refusal(sk, {k: v for k, v in parts.items() if k != sk.level_names[0]}),
    "all": leaf_on(sk), "shoots": leaf_on(only_hosts_on([4])),
    "twigs_and_shoots": leaf_on(only_hosts_on([3, 4])), "names": list(sk.level_names),
    "hosts": [int(n) for n in np.bincount(sk.level[sk.host], minlength=len(sk.level_names))]}))
runner.done()
"""


def test_leaf_parts_sit_on_their_own_level(tmp_path):
    """A leaf part is `on` the level its leaves' host limbs are of, not always the twigs; when
    the hosts span levels it is the lowest of them (#3). Bar 02's redbud has leaves on every
    level, from the trunk to the shoots."""
    skeleton = tmp_path / "redbud.npz"
    save_skeleton(_redbud(), skeleton)
    script = tmp_path / "script.py"
    script.write_text(LEVELS)
    result = run_blender(script, [str(skeleton)], timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    line = next(ln for ln in result.log.splitlines() if ln.startswith("LEVELS "))
    got = json.loads(line[len("LEVELS "):])
    names = got["names"]
    assert all(n > 0 for n in got["hosts"]), got  # the tree really has leaves on every level
    assert got["all"] == {"Leaves_Leaf": names[0]}, got
    assert got["shoots"] == {"Leaves_Leaf": names[4]}, got
    assert got["twigs_and_shoots"] == {"Leaves_Leaf": names[3]}, got
    # A shade with no leaf, or a host level with no part, is a clear refusal, not a bare error.
    assert "'Leaf' has no leaves" in got["no_leaves"], got
    assert f"level {names[0]!r}" in got["no_part"] and "grew no part" in got["no_part"], got


# --- the card piece: the game-weight copy of a grown tree --------------------------------------

CARD = f"{PIECE}_Cards"
CARD_NEAREST_M = 10.0
CARD_FARTHEST_M = 70.0
CARD_DIALS = CardDials(ball_m=0.5, normal_mix=0.5, sky=3.0)
CARD_WHY = "cards draw no lines: the leaves are an image with an alpha cut"
CARD_SIGHT = Sight(view_floor(View(CARD_NEAREST_M, "a test tree"), UPM), CREASE_DEG, round=False)

CARD_BUILD = """
from pathlib import Path

import bpy
import numpy as np

from model_kit.blender import build, grow, runner
from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.blender.export import export_pieces
from model_kit.core.grow.card_image import paint_cells
from model_kit.core.grow.cards import card_dials, card_parts
from model_kit.core.grow.mesh import leaf_lean
from model_kit.core.grow.rules import load_rules
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.manifest import View
from model_kit.core.sight import Sight, view_floor
from model_kit.core.units import load_style

skeleton, rules, out_dir, scene_dir, style_path = runner.script_args()
style = load_style(Path(style_path))
lean, dials = leaf_lean(style), card_dials(style)
sk = load_skeleton(Path(skeleton))
rules = load_rules(Path(rules))
build.reset()
mats = build.slots({"Bark": (0.25, 0.17, 0.12), "Twig": (0.1, 0.08, 0.07),
                    "Leaf": (0.4, 0.62, 0.25)})
wood = {"Trunk_Roots": mats["Bark"], "Leaders": mats["Bark"], "Branches": mats["Bark"],
        "Twigs": mats["Twig"], "Shoots": mats["Twig"]}
pack = build.PackBuilder("grove", 0.5, Path(out_dir), None, crease_deg=style.crease_deg)
root, apply_normals = grow.add_grown(
    pack, sk, rules=rules, materials={**wood, "Leaves_Leaf": mats["Leaf"]},
    done="A tree grown from its rules.", size_class="hero", origin="centre of the base",
    lean=lean, view=View(5.0, "a test tree, seen from the path beside it"))
view = View(NEAREST, "a test tree, seen from the road", farthest_m=FARTHEST,
            farthest_why="the farthest a test viewer stands")
cards, apply_corners = grow.add_cards(
    pack, sk, source=sk.info["piece"], rules=rules, materials={**wood, "Cards_Leaf": mats["Leaf"]},
    image=paint_cells(sk.outline, seed=int(sk.info["seed"])), done="The card copy.",
    size_class="hero", origin="centre of the base", dials=dials, lean=lean, view=view,
    outlined_why=WHY)


def before_export():
    apply_normals()
    apply_corners()


written = pack.finish([root.name, cards.name], before_export=before_export)

# What the Blender card mesh holds, against what `card_parts` made.
sight = Sight(view_floor(view, pack.units_per_metre), pack.crease_deg, round=False)
leaves = card_parts(sk, sight, dials, pack.units_per_metre, lean)[1]
me = bpy.data.objects[build.object_name(cards.name, "Cards_Leaf")].data
turn = BLENDER_TO_GLTF[:3, :3]
got = np.array([c.vector for c in me.corner_normals])
dev = np.abs(got - leaves.normals.reshape(-1, 3) @ turn).max(axis=1)
print("NORMALS", len(got), f"{dev.max():.2e}", int((dev > NORMALS_TOL).sum()))
uv = np.array([d.uv for d in me.uv_layers["UVMap"].data])
print("UVS", len(uv), f"{np.abs(uv - leaves.uvs.reshape(-1, 2)).max():.2e}")
col = np.array([d.color for d in me.color_attributes["Color"].data])
want = np.concatenate([leaves.colours.reshape(-1, 3), np.ones((len(col), 1))], axis=1)
print("COLOURS", len(col), f"{np.abs(col - want).max():.2e}")

export_pieces(written, Path(scene_dir), pieces=[root.name, cards.name])
runner.done()
"""
CARD_BUILD = (CARD_BUILD.replace("NORMALS_TOL", repr(NORMALS_TOL))
              .replace("NEAREST", repr(CARD_NEAREST_M)).replace("FARTHEST", repr(CARD_FARTHEST_M))
              .replace("WHY", repr(CARD_WHY)))


CARD_TWO_BUILD = """
from pathlib import Path

import bpy

from model_kit.blender import build, grow, runner
from model_kit.core.grow.card_image import paint_cells
from model_kit.core.grow.cards import card_dials
from model_kit.core.grow.mesh import leaf_lean
from model_kit.core.grow.rules import load_rules
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.manifest import View
from model_kit.core.units import load_style

skeleton, rules, out_dir, style_path = runner.script_args()
style = load_style(Path(style_path))
lean, dials = leaf_lean(style), card_dials(style)
sk = load_skeleton(Path(skeleton))
rules = load_rules(Path(rules))
build.reset()
mats = build.slots({"Bark": (0.25, 0.17, 0.12), "Twig": (0.1, 0.08, 0.07),
                    "Leaf": (0.4, 0.62, 0.25)})
wood = {"Trunk_Roots": mats["Bark"], "Leaders": mats["Bark"], "Branches": mats["Bark"],
        "Twigs": mats["Twig"], "Shoots": mats["Twig"]}
pack = build.PackBuilder("grove", 0.5, Path(out_dir), None, crease_deg=style.crease_deg)
root, apply_normals = grow.add_grown(
    pack, sk, rules=rules, materials={**wood, "Leaves_Leaf": mats["Leaf"]},
    done="A tree grown from its rules.", size_class="hero", origin="centre of the base",
    lean=lean, view=View(5.0, "a test tree, seen from the path beside it"))
view = View(NEAREST, "a test tree, seen from the road", farthest_m=FARTHEST,
            farthest_why="the farthest a test viewer stands")
image = paint_cells(sk.outline, seed=int(sk.info["seed"]))
made, fixes = [root], [apply_normals]
for name in ("First_Cards", "Second_Cards"):
    card, apply_corners = grow.add_cards(
        pack, sk, source=sk.info["piece"], rules=rules, name=name,
        materials={**wood, "Cards_Leaf": mats["Leaf"]}, image=image, done="A card copy.",
        size_class="hero", origin="centre of the base", dials=dials, lean=lean, view=view,
        outlined_why=WHY)
    made.append(card)
    fixes.append(apply_corners)
mat_names = sorted(m.name for m in bpy.data.materials if m.name.endswith("_Cards"))
print("CARD_MATS", mat_names)
pack.finish([ob.name for ob in made], before_export=lambda: [f() for f in fixes])
print("TWO_DONE")
runner.done()
"""
CARD_TWO_BUILD = (CARD_TWO_BUILD.replace("NEAREST", repr(CARD_NEAREST_M))
                  .replace("FARTHEST", repr(CARD_FARTHEST_M)).replace("WHY", repr(CARD_WHY)))


def _card_tree():
    """Bar 02's redbud (buds-v2: its tube sides come from the view, so its wood depends on
    `Sight.round`, which a card piece sets False)."""
    return _redbud()


def _card_style(folder: Path) -> Path:
    """Bar 02's style with a `dials.cards` (a test budget, not measured)."""
    raw = json.loads(BAR_STYLE.read_text())
    raw["dials"]["cards"] = {"ball_m": CARD_DIALS.ball_m, "normal_mix": CARD_DIALS.normal_mix,
                             "sky": CARD_DIALS.sky, "why": "a test budget, not measured"}
    path = folder / "card-style.json"
    path.write_text(json.dumps(raw))
    return path


@pytest.fixture(scope="module")
def cards_built(tmp_path_factory):
    """Bar 02's redbud and its card copy, built plain in one pack, with the scene exported."""
    folder = tmp_path_factory.mktemp("cards")
    skeleton = folder / "tree.npz"
    save_skeleton(_card_tree(), skeleton)
    script = folder / "script.py"
    script.write_text(CARD_BUILD)
    out, scene = folder / "out", folder / "scene"
    style = _card_style(folder)
    result = run_blender(script, [str(skeleton), str(RULES), str(out), str(scene),
                                  str(style)], timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    return out, scene, style, result.log


def test_two_card_pieces_in_one_pack(tmp_path):
    """Two card pieces of one tree build in one pack: each gets its own leaf material, named for
    its piece, since each image is painted for it and a shared colour role name would collide."""
    skeleton = tmp_path / "tree.npz"
    save_skeleton(_card_tree(), skeleton)
    script = tmp_path / "script.py"
    script.write_text(CARD_TWO_BUILD)
    out = tmp_path / "out"
    result = run_blender(script, [str(skeleton), str(RULES), str(out),
                                  str(_card_style(tmp_path))], timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok and "TWO_DONE" in result.log, result.log
    assert "CARD_MATS ['First_Cards_Leaf_Cards', 'Second_Cards_Leaf_Cards']" in result.log
    pack = load_pack(out / "grove.pack.json", load_style(_card_style(tmp_path)))
    assert {p.name for p in pack.pieces.values() if p.cards_of} == {"First_Cards", "Second_Cards"}


def test_card_piece_builds(cards_built, tmp_path):
    """The pack loads with the card piece (`cards_of` the tree, not outlined), and the scene's
    card piece is clean under skeleton-match, which rebuilds its wood with a sight that is not
    round (`sight_for` reads `Piece.outlined`)."""
    out, scene_dir, style, _ = cards_built
    pack = load_pack(out / "grove.pack.json", load_style(style))
    card = pack.pieces[CARD]
    assert (card.cards_of, card.outlined, card.recipe.kind) == (PIECE, False, "branching")
    assert card.outlined_why == CARD_WHY
    assert (card.view.nearest_m, card.view.farthest_m) == (CARD_NEAREST_M, CARD_FARTHEST_M)
    names = [p.name for p in card.parts]
    assert names == [*card_parts(_card_tree(), CARD_SIGHT, CARD_DIALS, UPM, LEAN)[0]]
    assert [(p.shape, p.on, p.role) for p in card.parts][-1] == ("leaf", "Trunk_Roots",
                                                                 "skip_outline")
    assert pack.pieces[PIECE].cards_of == "" and pack.pieces[PIECE].outlined
    clips = tmp_path / "clips.json"
    clips.write_text(json.dumps({"clips": {}}))
    stage_manifests(scene_dir, style=style, clips=clips, packs=[out / "grove.pack.json"])
    save_skeleton(_card_tree(), scene_dir / "pieces" / PACK / f"{PIECE}{SKELETON_SUFFIX}")
    ctx = load_scene(scene_dir)
    assert sorted(ctx.pieces[f"{PACK}:{CARD}"].parts) == sorted(names)
    check = SkeletonMatch()
    assert check.run(ctx) == []
    units = [u for u in check.plants(ctx) if u.item.startswith(f"{PACK}:{CARD} ")]
    assert any("Cards_Leaf" in u.item for u in units), [u.item for u in check.plants(ctx)]
    for unit in units:  # the check still fails on the card piece when a vertex is moved
        assert any(f.item == unit.item for plant in unit.candidates
                   for f in check.run(plant.ctx, only=plant.scope))


def test_card_corner_attributes_written(cards_built):
    """The card mesh's normals, UVs and corner colours in Blender are `card_parts`'; and the .glb's
    TEXCOORD_0 of each corner is (u, 1 - v) of its UV (the exporter flips v; the card image is
    painted with v up from the bottom, as Blender stores it)."""
    out, _, _, log = cards_built
    lines = {ln.split()[0]: ln.split() for ln in log.splitlines()
             if ln.split()[:1] in (["NORMALS"], ["UVS"], ["COLOURS"])}
    _, count, worst, stray = lines["NORMALS"]
    assert int(count) > 0 and float(worst) < NORMALS_MAX
    assert int(stray) <= NORMALS_STRAY_SHARE * int(count)
    for key in ("UVS", "COLOURS"):
        assert int(lines[key][1]) == int(count) and float(lines[key][2]) < 1e-6, lines[key]
    leaves = card_parts(_card_tree(), CARD_SIGHT, CARD_DIALS, UPM, LEAN)[1]
    js, binary = _glb(out / "grove.glb")
    node = next(n for n in js["nodes"] if n.get("name", "").endswith(f"{CARD}/Cards_Leaf"))
    prim = js["meshes"][node["mesh"]]["primitives"][0]
    pos = _accessor(js, binary, prim["attributes"]["POSITION"])
    uv = _accessor(js, binary, prim["attributes"]["TEXCOORD_0"])
    corners = leaves.tris.reshape(-1, 3)
    corner_uv = leaves.uvs.reshape(-1, 2)
    far = np.flatnonzero(np.abs(corner_uv[:, 1] - 0.5) > 0.1)[::37][:60]  # v is not 1 - v
    assert len(far) > 20
    for i in far:
        at = np.flatnonzero(np.abs(pos - corners[i]).max(axis=1) < 1e-5)
        want = np.array([corner_uv[i, 0], 1 - corner_uv[i, 1]])
        assert len(at) and np.abs(uv[at] - want).max(axis=1).min() < 1e-5, (i, corner_uv[i])


@pytest.fixture(scope="module")
def cards_spec_built(tmp_path_factory):
    """The same pair through `grow_build.py`, from a spec: Draco on, spacing 5."""
    folder = tmp_path_factory.mktemp("cards-spec")
    skeleton = folder / "tree.npz"
    save_skeleton(_card_tree(), skeleton)
    view = {"nearest_m": CARD_NEAREST_M, "why": "a test tree, seen from the road",
            "farthest_m": CARD_FARTHEST_M, "farthest_why": "the farthest a test viewer stands"}
    spec = {"pack": "grove", "units_per_metre": UPM, "spacing": 5.0, "rules": str(RULES),
            "style": str(_card_style(folder)), "out": str(folder / "out"),
            "scene": str(folder / "scene"),
            "colours": {"Bark": [0.25, 0.17, 0.12], "Twig": [0.1, 0.08, 0.07],
                        "Leaf": [0.4, 0.62, 0.25]},
            "pieces": [
                {"skeleton": str(skeleton), "size_class": "hero", "done": "The tree.",
                 "origin": "centre of the base", "view": {**view, "nearest_m": 5.0}},
                {"cards_of": PIECE, "size_class": "hero", "done": "The card copy.",
                 "origin": "centre of the base", "view": view, "outlined_why": CARD_WHY}]}
    path = folder / "spec.json"
    path.write_text(json.dumps(spec))
    result = run_blender(REPO / "src" / "model_kit" / "blender" / "grow_build.py", [str(path)],
                         timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    return folder / "out", result.log


def test_card_material_is_masked(cards_spec_built):
    """The card piece's leaf part ships the leaf image: its material is alpha MASK (cutoff 0.5),
    double-sided, with a base colour texture that is a PNG; the tree's own leaves and wood are
    not masked. The image's alpha is `paint_cells`' (v up from the bottom, so the PNG's rows run
    top-down the other way)."""
    out, _ = cards_spec_built
    js, binary = _glb(out / "grove.glb")
    by_name = {n["name"]: n for n in js["nodes"] if "name" in n and "mesh" in n}
    card = by_name[f"{CARD}/Cards_Leaf"]
    material = js["materials"][js["meshes"][card["mesh"]]["primitives"][0]["material"]]
    assert material["name"] == f"{CARD}_Leaf_Cards"
    assert material.get("alphaMode", "OPAQUE") == "MASK"
    assert material.get("alphaCutoff", 0.5) == 0.5  # glTF's default, which the exporter omits
    assert material["doubleSided"] is True
    texture = js["textures"][material["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
    image = js["images"][texture["source"]]
    assert image["mimeType"] == "image/png"
    start = js["bufferViews"][image["bufferView"]]
    png = binary[start.get("byteOffset", 0):start.get("byteOffset", 0) + start["byteLength"]]
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    sk = _card_tree()
    painted = paint_cells(sk.outline, seed=int(sk.info["seed"]))
    got = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert got.shape == painted.shape
    assert np.array_equal(got[::-1, :, 3], painted[:, :, 3])
    for name in (f"{PIECE}/Leaves_Leaf", f"{PIECE}/Branches", f"{CARD}/Branches"):
        other = js["materials"][js["meshes"][by_name[name]["mesh"]]["primitives"][0]["material"]]
        assert other.get("alphaMode", "OPAQUE") == "OPAQUE", name
        assert "baseColorTexture" not in other["pbrMetallicRoughness"], name


def _glb_counts(out: Path, piece: str) -> dict[str, int]:
    """The POSITION accessor count of each part mesh of `piece` in `out/grove.glb`."""
    js, _ = _glb(out / "grove.glb")
    counts = {}
    for node in js["nodes"]:
        if node.get("name", "").startswith(f"{piece}/") and "mesh" in node:
            prims = js["meshes"][node["mesh"]]["primitives"]
            counts[node["name"].split("/", 1)[1]] = sum(
                js["accessors"][p["attributes"]["POSITION"]]["count"] for p in prims)
    return counts


def test_card_build_prints_vertex_counts(cards_spec_built):
    """The build prints `<piece>: <n> vertices (wood <w>, leaves <l>)`: the vertices the .glb
    holds for the card piece (what a game loads), which are the sums of its meshes' POSITION
    accessor counts, read from the Draco file as the page reads them."""
    out, log = cards_spec_built
    counts = _glb_counts(out, CARD)
    leaves = counts["Cards_Leaf"]
    wood = sum(n for part, n in counts.items() if part != "Cards_Leaf")
    line = f"{CARD}: {wood + leaves} vertices (wood {wood}, leaves {leaves})"
    assert line in log.splitlines(), (line, [ln for ln in log.splitlines() if "vertices" in ln])


def test_card_wood_is_shaded_smooth(cards_spec_built):
    """No wood part of the card piece has a sharp edge: each exports one vertex per distinct
    position (the wood has no UVs, so no seam splits it). The full piece is shaded as before."""
    out, _ = cards_spec_built
    parts, _ = card_parts(_card_tree(), CARD_SIGHT, CARD_DIALS, UPM, LEAN)
    counts = _glb_counts(out, CARD)
    wood = [n for n in parts if not n.startswith("Cards_")]
    assert wood
    for name in wood:
        distinct = len(np.unique(np.round(parts[name].reshape(-1, 3), 6), axis=0))
        assert counts[name] == distinct, (name, counts[name], distinct)


def test_card_corners_export_once(cards_spec_built):
    """A card is one flat quad: each of its four corners carries one normal, UV and colour, so
    `Cards_Leaf` exports one vertex per distinct position. (Flat faces gave the two triangles'
    shared corners normals Blender packed 0.011 apart, and the exporter split them.)"""
    out, _ = cards_spec_built
    parts, leaves = card_parts(_card_tree(), CARD_SIGHT, CARD_DIALS, UPM, LEAN)
    distinct = len(np.unique(np.round(parts["Cards_Leaf"].reshape(-1, 3), 6), axis=0))
    assert distinct == 4 * leaves.cards
    assert _glb_counts(out, CARD)["Cards_Leaf"] == distinct
