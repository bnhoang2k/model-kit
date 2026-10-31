"""The build helpers in Blender: material slots that raise, and a two-piece pack end to end."""

import json
import struct
from pathlib import Path

import pytest

from model_kit.blender.runner import run_blender
from model_kit.core.manifest import End, Ref, load_pack
from model_kit.core.units import Style

TIMEOUT = 240
UNITS = 1.0
SPACING = 2.5

pytestmark = pytest.mark.blender


def _run(tmp_path: Path, body: str, *args: str):
    script = tmp_path / "script.py"
    script.write_text(body)
    result = run_blender(script, list(args), timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    return result


SLOTS = """
import bpy
from model_kit.blender import build, runner
from model_kit.core.errors import ManifestError

build.reset()
mats = build.slots({"Fill": (1, 1, 1), "Ink": (0.1, 0.1, 0.1)})
bsdf = mats["Fill"].node_tree.nodes["Principled BSDF"]
print("MADE", sorted(mats), tuple(round(c, 3) for c in bsdf.inputs["Base Color"].default_value))

cases = {
    "empty": {},
    "short_rgb": {"Fill": (1, 1)},
    "out_of_range": {"Fill": (1, 2, 1)},
    "not_a_number": {"Fill": (1, "x", 1)},
    "blank_name": {"": (1, 1, 1)},
}
for label, roles in cases.items():
    try:
        build.slots(roles)
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)

# A material of that name already there, but with no Principled BSDF: the old helper skipped it.
broken = bpy.data.materials.new("Shade")
if broken.node_tree is None:
    broken.use_nodes = True
for node in list(broken.node_tree.nodes):
    broken.node_tree.nodes.remove(node)
try:
    build.slots({"Shade": (0.85, 0.85, 0.85)})
    print("NO RAISE", "no_bsdf")
except ManifestError as exc:
    print("RAISED", "no_bsdf", exc)

# The public `slot` makes one material and raises on every failure `slots` does.
one = build.slot("Paint", (0.2, 0.4, 0.6))
print("SLOT", one.name, tuple(round(c, 3) for c in one.node_tree.nodes["Principled BSDF"]
                              .inputs["Base Color"].default_value))
slot_cases = {
    "slot_short_rgb": ("Paint", (1, 1)),
    "slot_out_of_range": ("Paint", (1, 2, 1)),
    "slot_blank_name": ("", (1, 1, 1)),
    "slot_number_name": (7, (1, 1, 1)),
    "slot_no_bsdf": ("Shade", (0.85, 0.85, 0.85)),
}
for label, (name, rgb) in slot_cases.items():
    try:
        build.slot(name, rgb)
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)
runner.done()
"""


def test_slot_failure_raises(tmp_path):
    result = _run(tmp_path, SLOTS)
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "MADE ['Fill', 'Ink'] (1.0, 1.0, 1.0, 1.0)" in lines
    raised = {line.split()[1] for line in lines if line.startswith("RAISED ")}
    assert raised == {"empty", "short_rgb", "out_of_range", "not_a_number", "blank_name",
                      "no_bsdf", "slot_short_rgb", "slot_out_of_range", "slot_blank_name",
                      "slot_number_name", "slot_no_bsdf"}, result.log
    assert "SLOT Paint (0.2, 0.4, 0.6, 1.0)" in lines
    assert not [line for line in lines if line.startswith("NO RAISE")]
    for label in ("no_bsdf", "slot_no_bsdf"):
        assert any("Shade" in line and "Principled BSDF" in line for line in lines
                   if line.startswith(f"RAISED {label} "))


PACK = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.manifest import Listed, Recipe, SitSpec, View

VIEW = View(1.0, "a test piece, seen from arm's length")

out_dir = Path(runner.script_args()[0])
build.reset()
build.selftest_axes()
mats = build.slots({"Fill": (1, 1, 1), "Ink": (0.1, 0.1, 0.1)})
FILL, INK = mats["Fill"], mats["Ink"]
pack = build.PackBuilder("shop", UNITS, out_dir, SPACING, crease_deg=35)
recipe = Recipe("assembly", "build_shop.py")

# A slab bench: two legs and a seat board on them, made leg, seat, leg.
bench = build.piece("Bench")
build.slab("leg_L", -0.28, -0.22, 0, 0.42, -0.12, 0.12, FILL, bench,
           props={"listed": "leg", "group": "legs"})
build.slab("seat_top", -0.3, 0.3, 0.42, 0.46, -0.15, 0.15, FILL, bench, r=0.01,
           props={"listed": "seat_board", "group": "top", "on": "leg_L"})
build.slab("leg_R", 0.22, 0.28, 0, 0.42, -0.12, 0.12, FILL, bench,
           props={"listed": "leg", "group": "legs"})
pack.register(bench, view=VIEW, origin="floor, under the seat centre", recipe=recipe,
              listed=[Listed("seat_board", "Pine seat board, 40 mm", "a catalogue", 0.04, None),
                      Listed("leg", "Pine leg, 60 mm square", "a catalogue", 0.06, None)],
              done="A plain bench one sitter wide.", size_class="piece", kind="furniture",
              fill=True)
pack.use(bench, "front", point=(0, 0, 0.5), facing=(0, 0, -1), posture="sit")
pack.hold(bench, "edge_R", use="front", side="R", surface=["seat_top"], point=(0.2, 0.46, 0.0),
          grip=False)
pack.seat(bench, "seat_1", use="front", point=(0, 0.46, 0), facing=(0, 0, 1),
          surface=["seat_top"])

# A lathe lamp: a puck base, a flared stem, and a knob parented to the stem (a nested part).
lamp = build.piece("Lamp")
build.lathe("base", build.puck_pts(0.15, 0.04, 0.01), FILL, lamp,
            props={"listed": "base", "group": "base"})
stem = build.lathe("stem", build.flare_pts(0.05, 0.02, 0.012, 1.2, R=0.02), INK, lamp, y0=0.04,
                   props={"listed": "stem", "group": "stem", "on": "base",
                          "role": "skip_outline"})
build.slab("knob", 0.012, 0.2, 0.88, 0.92, -0.01, 0.01, INK, stem,
           props={"listed": "stem", "group": "stem", "on": "stem"})
pack.register(lamp, view=VIEW, origin="floor, under the stem", recipe=recipe,
              listed=[Listed("base", "Cast base", "a catalogue", 0.01, None),
                      Listed("stem", "Steel tube, 24 mm", "a catalogue", 0.0012, "Lamp")],
              done="A floor lamp: a round base and one stem.", size_class="prop",
              kind="light", skin="white")
pack.use(lamp, "side", point=(0.4, 0, 0), facing=(-1, 0, 0), posture="stand")
pack.hold(lamp, "switch", use="side", side="both", surface=["stem"], point=(0.012, 0.9, 0),
          off=(1, 0, 0), grip=True)
pack.mount(lamp, "plug", kind="power", point=(0, 0.01, -0.15), axis=(0, 0, -1), takes="lead",
           fits=["socket"])

# A stool whose parts share the bench's part names.
stool = build.piece("Stool")
build.slab("leg_L", -0.02, 0.02, 0, 0.4, -0.02, 0.02, FILL, stool,
           props={"listed": "leg", "group": "legs"})
build.slab("seat_top", -0.12, 0.12, 0.4, 0.43, -0.12, 0.12, FILL, stool,
           props={"listed": "leg", "group": "top", "on": "leg_L"})
pack.register(stool, view=VIEW, origin="floor, under the seat", recipe=recipe,
              listed=[Listed("leg", "Pine, 40 mm", "a catalogue", 0.04, None)],
              done="A one-leg stool.", size_class="piece", kind="furniture")

seen = []
written = pack.finish(["Bench", "Lamp", "Stool"], gate=lambda p: seen.append(sorted(p.pieces)))
print("GATE", seen)
print("ROOT", build.seated_root((0, 0.46, 0), (0, 0, 1), SitSpec(0.083, 0.03)))
print("ROOT90", build.seated_root((1, 0.5, 2), (-1, 0, 0), SitSpec(0.1, 0.05)))
print("WRITTEN", sorted(written.pieces))
runner.done()
""".replace("UNITS", repr(UNITS)).replace("SPACING", repr(SPACING))


def _glb_json(path: Path) -> dict:
    data = path.read_bytes()
    assert data[:4] == b"glTF"
    (length,) = struct.unpack_from("<I", data, 12)
    return json.loads(data[20:20 + length])


def test_build_two_piece_pack(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, PACK, str(out))
    assert result.ok, result.log
    assert "AXIS TEST OK" in result.log
    files = ["shop.blend", "shop.glb", "shop.pack.json"]
    assert sorted(p.name for p in out.iterdir()) == files
    lines = result.log.splitlines()
    assert "GATE [['Bench', 'Lamp', 'Stool']]" in lines
    assert "WRITTEN ['Bench', 'Lamp', 'Stool']" in lines

    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    assert list(pack.pieces) == ["Bench", "Lamp", "Stool"]
    bench, lamp, stool = (pack.pieces[n] for n in ("Bench", "Lamp", "Stool"))
    assert bench.frame.axes == "gltf" and bench.frame.origin == "floor, under the seat centre"
    # Parts in the order they were made; groups first seen give the take-apart order.
    assert [(p.name, p.listed, p.shape, p.on, p.group, p.role) for p in bench.parts] == [
        ("leg_L", "leg", "box", None, "legs", "solid"),
        ("seat_top", "seat_board", "box", "leg_L", "top", "solid"),
        ("leg_R", "leg", "box", None, "legs", "solid"),
    ]
    assert list(dict.fromkeys(p.group for p in bench.parts)) == ["legs", "top"]
    # The knob is parented to the stem, not the root, and is still a part of the lamp.
    assert [(p.name, p.shape, p.on, p.role) for p in lamp.parts] == [
        ("base", "lathe", None, "solid"), ("stem", "lathe", "base", "skip_outline"),
        ("knob", "box", "stem", "solid")]
    assert lamp.measured["bounds"]["X1"] == 0.2  # the knob reaches past the base's 0.15
    assert lamp.listed[1].owner == "Lamp"
    # Two pieces may share part names.
    assert [p.name for p in stool.parts] == ["leg_L", "seat_top"]

    # The hold names a part, and its way off the surface came from the seat top's normal.
    (edge,) = bench.holds
    assert edge.surface == ("seat_top",)
    assert {p.name for p in bench.parts} >= set(edge.surface)
    assert edge.off == pytest.approx((0, 1, 0), abs=1e-6)
    assert lamp.holds[0].off == (1.0, 0.0, 0.0) and lamp.holds[0].grip is True
    assert bench.seats[0].point == (0.0, 0.46, 0.0)
    assert lamp.mounts[0].fits == ("socket",)
    assert lamp.uses[0].posture == "stand"

    # The measured fields, in the piece's own frame (layout does not move them).
    m = bench.measured
    assert m["kind"] == "furniture" and m["snap"] == "free" and m["zones"] == ["d"]
    assert m["foot"] == {"w": 0.6, "d": 0.3}
    assert m["bounds"] == {"X0": -0.3, "X1": 0.3, "Z0": -0.15, "Z1": 0.15}
    assert m["heights"] == {"top": 0.46}
    assert m["slots"] == ["Fill"] and m["tris"] > 12
    assert 0 < m["fill"] <= 1
    assert sorted(lamp.measured["slots"]) == ["Fill", "Ink"] and lamp.measured["skin"] == "white"
    assert "fill" not in lamp.measured

    # Pieces are laid out SPACING apart along X in the .glb. Objects are `<piece>/<part>`.
    glb = _glb_json(out / "shop.glb")
    nodes = {n["name"]: n for n in glb["nodes"]}
    assert sorted(m["name"] for m in glb["meshes"]) == [
        "Bench/leg_L", "Bench/leg_R", "Bench/seat_top", "Lamp/base", "Lamp/knob", "Lamp/stem",
        "Stool/leg_L", "Stool/seat_top"]
    assert nodes["Bench"].get("translation", [0, 0, 0]) == pytest.approx([0, 0, 0])
    assert nodes["Lamp"]["translation"] == pytest.approx([SPACING, 0, 0])
    assert nodes["Stool"]["translation"] == pytest.approx([2 * SPACING, 0, 0])
    assert not any("build_order" in n.get("extras", {}) for n in glb["nodes"])

    # seated_root reads the rig's sit: forward along the seat's facing, below the seat top.
    root = next(line for line in lines if line.startswith("ROOT "))
    assert eval(root[5:]) == pytest.approx((0, 0.46 - 0.083, 0.03))
    root90 = next(line for line in lines if line.startswith("ROOT90 "))
    assert eval(root90[7:]) == pytest.approx((1 - 0.05, 0.4, 2))

    # A second build into the same folder replaces the three files and leaves no staging files.
    before = (out / "shop.glb").stat().st_mtime_ns
    again = _run(tmp_path, PACK, str(out))
    assert again.ok, again.log
    assert sorted(p.name for p in out.iterdir()) == files
    assert (out / "shop.glb").stat().st_mtime_ns > before
    assert load_pack(out / "shop.pack.json", Style(UNITS, {}, {})) == pack


TOOLS = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Listed, Recipe, View

VIEW = View(1.0, "a test piece, seen from arm's length")

out_dir = Path(runner.script_args()[0])
build.reset()
fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
pack = build.PackBuilder("tools", 1.0, out_dir, 2.5, crease_deg=35)
recipe = Recipe("assembly", "x.py")


def stick(name, z0, z1, tool):
    root = build.piece(name)
    # A thin shaft along Z. Its butt end reaches 0.5 behind the origin; its tip 0.3 ahead of it.
    build.slab("shaft", -0.01, 0.03, 0.0, 0.02, z0, z1, fill, root,
               props={"listed": "shaft", "group": "shaft"})
    pack.register(root, view=VIEW, origin="the stick's grip point", recipe=recipe,
                  listed=[Listed("shaft", "Hickory shaft", "a catalogue", 0.01, None)],
                  done="A stick.", size_class="prop", kind="tool", tool=tool)
    return root


stick("Stick", -0.5, 0.3, True)
stick("Rod", -0.5, 0.3, False)
stick("Butt", -0.5, 0.0, True)  # nothing past the origin along +Z: no tip
try:
    pack.finish(["Stick", "Rod", "Butt"])
    print("NO RAISE")
except ManifestError as exc:
    print("RAISED", exc)

build.reset()
fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
pack = build.PackBuilder("tools", 1.0, out_dir, 2.5, crease_deg=35)
stick("Stick", -0.5, 0.3, True)
stick("Rod", -0.5, 0.3, False)
pack.finish(["Stick", "Rod"])
runner.done()
"""


def test_held_tool_publishes_measured_tip_length(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, TOOLS, str(out))
    assert result.ok, result.log
    # A tool with no vertex past its origin along the tool axis has no tip: the build refuses.
    refused = [line for line in result.log.splitlines() if line.startswith("RAISED ")]
    assert len(refused) == 1 and "'Butt'" in refused[0] and "needs a tip" in refused[0], result.log

    pack = load_pack(out / "tools.pack.json", Style(UNITS, {}, {}))
    stick, rod = pack.pieces["Stick"], pack.pieces["Rod"]
    # An independent measure: the largest Z of the exported mesh's positions in the .glb.
    glb = _glb_json(out / "tools.glb")
    node = next(n for n in glb["nodes"] if n["name"] == "Stick/shaft")
    assert node.get("translation", [0, 0, 0])[2] == pytest.approx(0)
    accessor = glb["accessors"][glb["meshes"][node["mesh"]]["primitives"][0]["attributes"]
                                ["POSITION"]]
    # It is the reach ahead of the origin (0.3), not the span (0.8) and not the butt end (0.5).
    assert accessor["max"][2] == pytest.approx(0.3)
    assert stick.measured["tip_length"] == pytest.approx(accessor["max"][2], abs=1e-4)
    # A piece that is not marked as a tool publishes none.
    assert "tip_length" not in rod.measured


REFUSED = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.manifest import Listed, Recipe, SitSpec, View

VIEW = View(1.0, "a test piece, seen from arm's length")

out_dir = Path(runner.script_args()[0])


def box_pack(out, spacing=2.0, names=("Box",)):
    build.reset()
    fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
    pack = build.PackBuilder("shop", 1.0, out, spacing, crease_deg=35)
    roots = []
    for name in names:
        root = build.piece(name)
        build.slab("body", -0.1, 0.1, 0, 0.2, -0.1, 0.1, fill, root,
                   props={"listed": "body", "group": "body"})
        pack.register(root, view=VIEW, origin="floor", recipe=Recipe("assembly", "x.py"),
                      listed=[Listed("body", "A box", "a drawing", 0.01, None)], done="A box.",
                      size_class="prop", kind="box")
        pack.use(root, "front", point=(0, 0, 0.3), facing=(0, 0, -1), posture="stand")
        roots.append(root)
    return pack, roots


def missing_order():
    pack, _ = box_pack(out_dir)
    pack.finish(["Box", "Other"])


def bad_surface():
    pack, (box,) = box_pack(out_dir)
    pack.hold(box, "lid", use="front", side="both", surface=["lid"], point=(0, 0.2, 0),
              grip=False)
    pack.finish(["Box"])


def gate_sees_checked_records():
    pack, (box,) = box_pack(out_dir)
    pack.hold(box, "lid", use="front", side="both", surface=["lid"], point=(0, 0.2, 0),
              off=(0, 1, 0), grip=False)
    pack.finish(["Box"], gate=lambda p: print("GATE CALLED"))


def unknown_empty():
    pack, (box,) = box_pack(out_dir)
    build.empty("Marker", box, (0, 0, 0), {"kind": "marker"})
    pack.finish(["Box"])


def seat_empty():
    pack, (box,) = box_pack(out_dir)
    build.empty("OldSeat", box, (0, 0.2, 0), {"kind": "seat", "clip": "sit"})
    pack.finish(["Box"])


def tight_spacing():
    pack, _ = box_pack(out_dir, spacing=0.2, names=("A", "B"))
    pack.finish(["A", "B"])


def str_surface():
    pack, (box,) = box_pack(out_dir)
    pack.hold(box, "lid", use="front", side="both", surface="body", point=(0, 0.2, 0),
              grip=False)


def vertical_seat():
    build.seated_root((0, 0, 0), (1e-13, 1, 0), SitSpec(0.1, 0.05))


for case in (missing_order, bad_surface, gate_sees_checked_records, unknown_empty, seat_empty,
             tight_spacing, str_surface, vertical_seat):
    try:
        case()
        print("NO RAISE", case.__name__)
    except Exception as exc:
        print("RAISED", case.__name__, type(exc).__name__, exc)
print("FILES", sorted(p.name for p in out_dir.iterdir()) if out_dir.exists() else [])

# An export that fails leaves the old files as they were, and no staging files.
fail_dir = out_dir.parent / "out_fail"
fail_dir.mkdir()
for ext in ("pack.json", "glb", "blend"):
    (fail_dir / f"shop.{ext}").write_text("old")
pack, _ = box_pack(fail_dir)


def boom(path, **kw):
    raise RuntimeError("export failed")


build.export_glb = boom
try:
    pack.finish(["Box"])
    print("NO RAISE", "export_fail")
except RuntimeError as exc:
    print("RAISED", "export_fail", exc)
print("FAIL FILES", sorted((p.name, p.read_text()) for p in fail_dir.iterdir()))
runner.done()
"""


def test_finish_refuses_before_writing(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, REFUSED, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    raised = {line.split()[1]: line for line in lines if line.startswith("RAISED ")}
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    assert "Other" in raised["missing_order"]
    assert "'lid'" in raised["bad_surface"]
    assert "ManifestError" in raised["gate_sees_checked_records"]
    assert "lid" in raised["gate_sees_checked_records"]
    assert "GATE CALLED" not in lines
    assert "'Marker'" in raised["unknown_empty"] and "'marker'" in raised["unknown_empty"]
    assert "PackBuilder.seat()" in raised["seat_empty"]
    assert "spacing 0.2" in raised["tight_spacing"]
    assert "list of names" in raised["str_surface"]
    assert "straight up or down" in raised["vertical_seat"]
    assert "FILES []" in lines
    assert "export failed" in raised["export_fail"]
    assert ("FAIL FILES [('shop.blend', 'old'), ('shop.glb', 'old'), ('shop.pack.json', 'old')]"
            in lines)


# --- meshes made by other code: tag(), pivots, placed packs --------------------------------

ADOPT_HEAD = """
import math
import bpy
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Listed, Recipe, View

out_dir = Path(runner.script_args()[0])
build.reset()
mat = build.slots({"Fill": (1, 1, 1)})["Fill"]
RECIPE = Recipe("assembly", "build_shop.py")
LISTED = [Listed("board", "Pine board", "a catalogue", 0.02, None)]
VIEW = View(1.0, "a test piece, seen from arm's length")


def made(name, parent, smooth_of=lambda i: True, x=0.0):
    # A box that other code made: plain bpy, no mesh_obj, no custom properties.
    bm = build.box_bm(x - 0.1, x + 0.1, 0, 0.2, -0.1, 0.1)
    bm.faces.ensure_lookup_table()
    for i, face in enumerate(bm.faces):
        face.smooth = smooth_of(i)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = parent
    return ob


def register(pack, root, view=VIEW):
    pack.register(root, view=view, origin="floor", recipe=RECIPE, listed=LISTED, done="A box.",
                  size_class="prop", kind="box")
"""

TAGGED = ADOPT_HEAD + """
pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
bench = build.piece("Bench")
late, early = made("foreign_late", bench, x=0.4), made("foreign_early", bench)
print("TAGGED", build.tag(early, "Bench", "early", listed="board", group="top",
                          shape="box") is None)
build.tag(late, "Bench", "late", listed="board", group="legs", shape="box", on="early",
          role="skip_outline")
made_by_kit = build.slab("kit", -0.5, -0.3, 0, 0.2, -0.1, 0.1, mat, bench,
                         props={"listed": "board", "group": "legs"})
print("NAMES", early.name, late.name, made_by_kit.name, early.data.name, late.data.name)
print("PROPS early", early["listed"], early["group"], early["shape"], early.get("on"),
      early.get("role"))
print("PROPS late", late["listed"], late["group"], late["shape"], late["on"], late["role"])
print("ORDER", early["build_order"] < late["build_order"] < made_by_kit["build_order"])
register(pack, bench)
written = pack.finish(["Bench"])
print("PARTS", [(p.name, p.group, p.on, p.role) for p in written.pieces["Bench"].parts])
runner.done()
"""

TAG_REFUSED = ADOPT_HEAD + """
bench = build.piece("Bench")
build.slab("leg", -0.5, -0.3, 0, 0.2, -0.1, 0.1, mat, bench,
           props={"listed": "board", "group": "legs"})
taker = made("foreign", bench)
mixed = bpy.data.objects.new("marker", None)
bpy.context.scene.collection.objects.link(mixed)
ok = dict(listed="board", group="top", shape="box")


def taken():
    build.tag(taker, "Bench", "leg", **ok)


def bad_part():
    build.tag(taker, "Bench", "a b", **ok)


def bad_piece():
    build.tag(taker, "", "x", **ok)


def not_a_mesh():
    build.tag(mixed, "Bench", "x", **ok)


def listed_not_text():
    build.tag(taker, "Bench", "x", listed=3, group="top", shape="box")


def role_not_text():
    build.tag(taker, "Bench", "x", role=1.5, **ok)


def mesh_obj_bad_value():
    build.slab("badlisted", 0, 1, 0, 1, 0, 1, mat, bench, props={"listed": 3, "group": "top"})


for case in (taken, bad_part, bad_piece, not_a_mesh, listed_not_text, role_not_text,
             mesh_obj_bad_value):
    try:
        case()
        print("NO RAISE", case.__name__)
    except ManifestError as exc:
        print("RAISED", case.__name__, exc)
print("ORPHAN", bpy.data.objects.get("Bench/badlisted"), bpy.data.meshes.get("Bench/badlisted"))
print("UNCHANGED", taker.name, "build_order" in taker, "listed" in taker)
runner.done()
"""

SHADED = ADOPT_HEAD + """
pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
root = build.piece("Box")
smooth = made("s", root, lambda i: True)
flat = made("f", root, lambda i: False, x=0.4)
mixed = made("m", root, lambda i: i % 2 == 0, x=0.8)
for ob, part in ((smooth, "smooth"), (flat, "flat"), (mixed, "mixed")):
    build.tag(ob, "Box", part, listed="board", group=part, shape="box")
print("FLAGS", [(ob.get("smooth"), ob.get("shade")) for ob in (smooth, flat, mixed)])
before = [p.use_smooth for p in mixed.data.polygons]
register(pack, root)
pack.finish(["Box"])
after = [p.use_smooth for p in mixed.data.polygons]
print("MIXED KEPT", before == after, sum(before), len(before))
print("FLAT", any(p.use_smooth for p in flat.data.polygons))
print("LEFT", [sorted(k for k in ob.keys() if k in ("smooth", "shade", "build_order"))
               for ob in (smooth, flat, mixed)])
runner.done()
"""


def _lines(result, tag: str) -> list[str]:
    return [line.split(" ", 1)[1] for line in result.log.splitlines()
            if line.startswith(tag + " ")]


def test_tag_names_and_orders_parts(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, TAGGED, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "TAGGED True" in lines
    # Objects and their mesh data take the `<piece>/<part>` name; kit-made parts stay as they were.
    assert ("NAMES Bench/early Bench/late Bench/kit Bench/early Bench/late") in lines
    assert "PROPS early board top box None None" in lines
    assert "PROPS late board legs box early skip_outline" in lines
    assert "ORDER True" in lines
    # Parts come in the order they were tagged or made, not the order Blender made the meshes.
    assert ("PARTS [('early', 'top', None, 'solid'), ('late', 'legs', 'early', 'skip_outline'),"
            " ('kit', 'legs', None, 'solid')]") in lines
    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    assert [p.name for p in pack.pieces["Bench"].parts] == ["early", "late", "kit"]
    glb = _glb_json(out / "shop.glb")
    assert sorted(m["name"] for m in glb["meshes"]) == ["Bench/early", "Bench/kit", "Bench/late"]
    extras = [n.get("extras", {}) for n in glb["nodes"] if n["name"] == "Bench/early"]
    assert extras == [{"listed": "board", "group": "top", "shape": "box"}]


ROUND_TRIP = ADOPT_HEAD + """
import dataclasses
from model_kit.core.manifest.records import PART_TAGGED

# Every tagged field, set to a value that is not its default. A tagged field with no entry here
# fails the script, so a new field forces the test to be extended.
NON_DEFAULT = {"listed": "board", "group": "legs", "shape": "box", "on": "host",
               "role": "skip_outline", "fastener": True, "retains": ("host", "other")}
assert set(NON_DEFAULT) == set(PART_TAGGED), (sorted(NON_DEFAULT), PART_TAGGED)
assert sorted(build.PART_PROPS) == sorted(PART_TAGGED), build.PART_PROPS

pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
bench = build.piece("Bench")
for name, x in (("host", 0.0), ("other", 0.4)):
    build.tag(made("foreign_" + name, bench, x=x), "Bench", name, listed="board", group="legs",
              shape="box")
bolt = made("foreign_bolt", bench, x=0.8)
build.tag(bolt, "Bench", "bolt", **NON_DEFAULT)
viamesh = build.slab("viamesh", 1.2, 1.4, 0, 0.2, -0.1, 0.1, mat, bench, props=dict(NON_DEFAULT))
plain = made("foreign_plain", bench, x=1.6)
build.tag(plain, "Bench", "plain", listed="board", group="legs", shape="box")

for ob in (bolt, viamesh):
    got = build._part("Bench", ob)
    for key in PART_TAGGED:
        print("ROUND", ob.name, key, getattr(got, key) == NON_DEFAULT[key])
print("DEFAULTS", [k for k in ("fastener", "retains") if k in plain])
print("PLAIN", build._part("Bench", plain).fastener, build._part("Bench", plain).retains)
register(pack, bench)
written = pack.finish(["Bench"])
for p in written.pieces["Bench"].parts:
    print("WRITTEN", p.name, p.fastener, list(p.retains))
runner.done()
"""


def test_every_part_field_declares_whether_it_is_tagged():
    from dataclasses import fields

    from model_kit.core.manifest.records import PART_TAGGED, Part

    undeclared = [f.name for f in fields(Part) if "tag" not in f.metadata]
    assert not undeclared, f"declare metadata={{'tag': True or False}} on Part.{undeclared}"
    assert PART_TAGGED == tuple(f.name for f in fields(Part) if f.metadata["tag"])
    # The fields something else sets stay out, so `tag` is never asked to write them.
    assert {f.name for f in fields(Part)} - set(PART_TAGGED) == {"name", "bones", "bind", "form"}


def test_every_tagged_part_field_round_trips(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, ROUND_TRIP, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    rounds = [x for x in lines if x.startswith("ROUND ")]
    assert len(rounds) == 2 * 7, result.log
    assert all(x.endswith(" True") for x in rounds), rounds
    assert "DEFAULTS []" in lines  # a default is not written
    assert "PLAIN False ()" in lines
    assert "WRITTEN bolt True ['host', 'other']" in lines
    assert "WRITTEN viamesh True ['host', 'other']" in lines
    assert "WRITTEN plain False []" in lines
    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    bolt = {p.name: p for p in pack.pieces["Bench"].parts}["bolt"]
    assert bolt.fastener is True and bolt.retains == ("host", "other")


def test_tag_refuses_a_taken_name(tmp_path):
    result = _run(tmp_path, TAG_REFUSED, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    raised = {line.split()[1]: line for line in lines if line.startswith("RAISED ")}
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    assert "'Bench/leg'" in raised["taken"] and "taken" in raised["taken"]
    assert "'a b'" in raised["bad_part"]
    assert "piece name" in raised["bad_piece"]
    assert "MESH" in raised["not_a_mesh"] and "'marker'" in raised["not_a_mesh"]
    assert "'listed'" in raised["listed_not_text"]
    assert "'role'" in raised["role_not_text"]
    # A part refused for a bad value leaves no object and no mesh behind, so a retry can name it.
    assert "'listed'" in raised["mesh_obj_bad_value"]
    assert "ORPHAN None None" in lines
    # A refused tag changes nothing on the object.
    assert "UNCHANGED foreign False False" in lines


def test_tag_keeps_smooth_shading(tmp_path):
    result = _run(tmp_path, SHADED, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    # All-smooth sets smooth, all-flat clears it, mixed flags set neither and ask to be kept.
    assert "FLAGS [(True, None), (False, None), (None, 'keep')]" in lines
    assert "MIXED KEPT True 3 6" in lines
    assert "FLAT False" in lines
    assert "LEFT [[], [], []]" in lines


CREASE = ADOPT_HEAD + """
import bmesh

def fold():
    # Two quads sharing one edge, the second turned 30 degrees: a rim whose faces turn 30.
    me = bpy.data.meshes.new("fold")
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (1, 1 + c, s), (0, 1 + c, s)],
                   [], [(0, 1, 2, 3), (3, 2, 4, 5)])
    return me

def edge_smooth(me):
    bm = bmesh.new()
    bm.from_mesh(me)
    both = [e.smooth for e in bm.edges if len(e.link_faces) == 2]
    bm.free()
    return both

for crease in (35, 25):
    me = fold()
    build._shade(me, True, crease)
    print("RIM", crease, edge_smooth(me))

for crease in (100, 80):
    # A box turns 90 degrees at every edge, so the builder's angle sets all of them.
    out = out_dir / f"c{crease}"
    pack = build.PackBuilder("shop", 1.0, out, 2.5, crease_deg=crease)
    root = build.piece(f"Box{crease}")
    build.tag(made(f"m{crease}", root), root.name, "board", listed="board", group="top",
              shape="box")
    register(pack, root)
    pack.finish([root.name])
    print("BOX", crease, set(edge_smooth(root.children[0].data)))
runner.done()
"""

NO_CREASE = ADOPT_HEAD + """
for bad in (None, 0, 180, "35", True):
    try:
        build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=bad)
        print("NO RAISE", repr(bad))
    except ManifestError as exc:
        print("RAISED", repr(bad), exc)
try:
    build.PackBuilder("shop", 1.0, out_dir, 2.5)
    print("NO RAISE", "left out")
except ManifestError as exc:
    print("RAISED", "left_out", exc)
runner.done()
"""


NO_VIEW = ADOPT_HEAD + """
pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
for label, view in (("none", None), ("a number", 1.0), ("a dict", {"nearest_m": 1.0})):
    root = build.piece(f"Box_{label.replace(' ', '_')}")
    build.tag(made(f"m_{label.replace(' ', '_')}", root), root.name, "board", listed="board",
              group="top", shape="box")
    try:
        register(pack, root, view=view)
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)
try:
    root = build.piece("Box_left_out")
    pack.register(root, origin="floor", recipe=RECIPE, listed=LISTED, done="A box.",
                  size_class="prop", kind="box")
    print("NO RAISE", "left out")
except ManifestError as exc:
    print("RAISED", "left_out", exc)
runner.done()
"""


def test_build_without_view_stops(tmp_path):
    """A piece is registered with the distance it is seen from, as the pack is built with a
    crease angle: left out, wrong or a plain number, the build raises and names `view`."""
    result = _run(tmp_path, NO_VIEW, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    raised = [line for line in lines if line.startswith("RAISED")]
    assert len(raised) == 4, result.log
    assert all("view" in line and "Box_" in line for line in raised), result.log


CHECK_REGISTRATION = ADOPT_HEAD + """
pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
root = build.piece("Box")
build.tag(made("m", root), root.name, "board", listed="board", group="top", shape="box")
before = len(bpy.data.objects)
for label, kwargs in (("none", dict(view=None)), ("a number", dict(view=1.0)),
                      ("outlined", dict(view=VIEW, outlined="yes")),
                      ("zones", dict(view=VIEW, zones="d"))):
    try:
        pack.check_registration(name="Box", **kwargs)
        print("NO RAISE", label)
    except ManifestError as exc:
        print("RAISED", label, exc)
pack.check_registration(name="Box", view=VIEW)
print("FREE", len(bpy.data.objects) == before)
register(pack, root)
try:
    pack.check_registration(name="Box", view=VIEW)
    print("NO RAISE", "twice")
except ManifestError as exc:
    print("RAISED", "twice", exc)
runner.done()
"""


def test_check_registration_judges_without_an_object(tmp_path):
    """`check_registration` takes a name, not a root: it refuses what `register` refuses in its
    own arguments (a view that is no `View`, `outlined` that is no bool, zones that are no list of
    names, a name already registered) and makes nothing."""
    result = _run(tmp_path, CHECK_REGISTRATION, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    raised = {line.split()[1] if line.split()[1] != "a" else "a number": line
              for line in lines if line.startswith("RAISED")}
    assert set(raised) == {"none", "a number", "outlined", "zones", "twice"}, result.log
    assert "view" in raised["none"] and "'Box'" in raised["none"], result.log
    assert "outlined" in raised["outlined"] and "zones" in raised["zones"], result.log
    assert "registered twice" in raised["twice"], result.log
    assert "FREE True" in lines, result.log


def test_shade_uses_style_crease(tmp_path):
    result = _run(tmp_path, CREASE, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    # A 30 degree rim shades smooth under a 35 degree crease and hard under a 25 degree one.
    assert "RIM 35 [True]" in lines
    assert "RIM 25 [False]" in lines
    # The builder hands its own angle to the shading, not a module constant.
    assert "BOX 100 {True}" in lines
    assert "BOX 80 {False}" in lines


def test_build_without_crease_stops(tmp_path):
    result = _run(tmp_path, NO_CREASE, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    raised = [line for line in lines if line.startswith("RAISED")]
    assert len(raised) == 6, result.log  # None, 0, 180, "35", True, and no argument at all
    assert all("crease_deg" in line for line in raised)


PIVOTS = ADOPT_HEAD + """
pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
first = build.piece("First")
made("foreign_first", first)
build.tag(first.children[0], "First", "board", listed="board", group="top", shape="box")
register(pack, first)
head = build.piece("Head")
arm = made("foreign_arm", head)
build.tag(arm, "Head", "arm", listed="board", group="top", shape="box")
build.empty("pan.001", head, (0.1, 0.5, 0.2), {"kind": "pivot", "name": "pan"})
build.empty("Head/tilt", arm, (-0.3, 0.7, 0.0), {"kind": "pivot", "name": "tilt"})
register(pack, head)
pack.finish(["First", "Head"])
print("DONE")

def twice():
    pack = build.PackBuilder("dup", 1.0, out_dir / "dup", 2.5, crease_deg=35)
    root = build.piece("Dup")
    build.tag(made("m", root), "Dup", "board", listed="board", group="top", shape="box")
    build.empty("a", root, (0, 0, 0), {"kind": "pivot", "name": "pan"})
    build.empty("b", root, (0, 1, 0), {"kind": "pivot", "name": "pan"})
    register(pack, root)
    pack.finish(["Dup"])

def nameless():
    pack = build.PackBuilder("nameless", 1.0, out_dir / "nameless", 2.5, crease_deg=35)
    root = build.piece("Nameless")
    build.tag(made("m", root), "Nameless", "board", listed="board", group="top", shape="box")
    build.empty("pan.001", root, (0, 0, 0), {"kind": "pivot"})
    register(pack, root)
    pack.finish(["Nameless"])

for case in (twice, nameless):
    build.reset()
    mat = build.slots({"Fill": (1, 1, 1)})["Fill"]
    try:
        case()
        print("NO RAISE", case.__name__)
    except ManifestError as exc:
        print("RAISED", case.__name__, exc)
runner.done()
"""


def test_pivot_empty_measured(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, PIVOTS, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "DONE" in lines and not [x for x in lines if x.startswith("NO RAISE")], result.log
    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    assert "pivots" not in pack.pieces["First"].measured
    # In the piece's own frame, though Head sits 2.5 along X in the .glb; sorted by name.
    assert pack.pieces["Head"].measured["pivots"] == [
        {"name": "pan", "x": 0.1, "y": 0.5, "z": 0.2},
        {"name": "tilt", "x": -0.3, "y": 0.7, "z": 0.0}]
    nodes = {n["name"]: n for n in _glb_json(out / "shop.glb")["nodes"]}
    assert nodes["pan.001"]["extras"]["kind"] == "pivot" and "Head/tilt" in nodes
    assert nodes["pan.001"]["translation"] == pytest.approx([0.1, 0.5, 0.2])
    dup = next(x for x in lines if x.startswith("RAISED twice"))
    assert "'pan'" in dup and "pivot" in dup
    # A pivot without a `name` property is refused by its object name, never named after it.
    bare = next(x for x in lines if x.startswith("RAISED nameless"))
    assert "'pan.001'" in bare and "'name'" in bare
    assert "pan.001" not in {p["name"] for p in pack.pieces["Head"].measured["pivots"]}


PLACED = ADOPT_HEAD + """
for bad in (0, -1.0, "2", True):
    try:
        build.PackBuilder("shop", 1.0, out_dir, bad)
        print("NO RAISE", repr(bad))
    except ManifestError as exc:
        print("RAISED", repr(bad), exc)
pack = build.PackBuilder("stage", 1.0, out_dir, None, crease_deg=35)
roots = []
for name, loc, turn in (("Deck", (3.0, 0.0, 1.0), 0.0), ("Rig", (3.1, 0.0, 1.0), math.pi / 2)):
    root = build.piece(name)
    build.tag(made("m_" + name, root), name, "board", listed="board", group="top", shape="box")
    root.location = build.B(*loc)
    root.rotation_euler = (0, 0, turn)
    register(pack, root)
    roots.append(root)
pack.finish(["Deck", "Rig"])  # the two overlap: with no spacing there is no spacing check
print("ROOTS", [tuple(round(c, 3) for c in r.location) for r in roots],
      [round(r.rotation_euler.z, 3) for r in roots])
runner.done()
"""


def test_placed_pack_keeps_positions(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, PLACED, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert not [x for x in lines if x.startswith("NO RAISE")], result.log
    assert len([x for x in lines if x.startswith("RAISED ")]) == 4
    assert "ROOTS [(3.0, -1.0, 0.0), (3.1, -1.0, 0.0)] [0.0, 1.571]" in lines
    raw = json.loads((out / "stage.pack.json").read_text())
    # The writer leaves a None spacing out; a missing key and null read alike.
    assert raw.get("spacing") is None
    pack = load_pack(out / "stage.pack.json", Style(UNITS, {}, {}))
    assert pack.spacing is None
    # Measures are in each piece's own frame, whatever its place on the stage.
    for name in ("Deck", "Rig"):
        assert pack.pieces[name].measured["bounds"] == {"X0": -0.1, "X1": 0.1, "Z0": -0.1,
                                                        "Z1": 0.1}
    nodes = {n["name"]: n for n in _glb_json(out / "stage.glb")["nodes"]}
    assert nodes["Deck"]["translation"] == pytest.approx([3.0, 0.0, 1.0])
    assert nodes["Rig"]["translation"] == pytest.approx([3.1, 0.0, 1.0])
    assert nodes["Rig"]["rotation"][1] == pytest.approx(0.7071, abs=1e-3)


def test_spacing_written(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, PACK, str(out))
    assert result.ok, result.log
    assert json.loads((out / "shop.pack.json").read_text())["spacing"] == SPACING
    assert load_pack(out / "shop.pack.json", Style(UNITS, {}, {})).spacing == SPACING


INSTANCED = ADOPT_HEAD + """
# the mirrored twin spreads wider, so its pack needs more room between pieces
pack = build.PackBuilder("shop", 1.0, out_dir, 8.0 if "mirror" in runner.script_args() else 2.5,
                         crease_deg=35)
first, second = build.piece("First"), build.piece("Second")
one = made("shared", first)
two = bpy.data.objects.new("twin", one.data)  # one mesh, two objects: an instance
bpy.context.scene.collection.objects.link(two)
two.parent = second
two.location = (0.5, 0.0, 0.0)
EXTRA = "extra" in runner.script_args()
MIRROR = "mirror" in runner.script_args()
UV = "uv" in runner.script_args()
for ob in (one, two):
    mod = ob.modifiers.new("Thick", "SOLIDIFY")
    mod.thickness = 0.02
if EXTRA:  # the twin's stack differs, so its geometry differs: it must not share the mesh
    two.modifiers.new("Sub", "SUBSURF").levels = 1
if MIRROR:  # equal stacks, but each result depends on where the object stands
    two.location = (3.0, 0.0, 0.0)
    axis = bpy.data.objects.new("mirror_at", None)
    bpy.context.scene.collection.objects.link(axis)
    for ob in (one, two):
        mod = ob.modifiers.new("Flip", "MIRROR")
        mod.mirror_object = axis


if UV:  # equal stacks that move no vertex: a projector writes each object's UVs from its place
    two.location = (3.0, 0.0, 0.0)
    one.data.uv_layers.new(name="UVMap")
    lens = bpy.data.objects.new("projector", None)
    bpy.context.scene.collection.objects.link(lens)
    for ob in (one, two):
        mod = ob.modifiers.new("Project", "UV_PROJECT")
        mod.uv_layer = "UVMap"
        mod.projector_count = 1
        mod.projectors[0].object = lens


def shape(me):  # counts, every vertex place and every UV, so a same-count result still differs
    uvs = [sorted(tuple(round(c, 5) for c in d.uv) for d in layer.data) for layer in me.uv_layers]
    return (len(me.vertices), len(me.polygons),
            sorted(tuple(round(c, 5) for c in v.co) for v in me.vertices), uvs)


def evaluated(ob):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), depsgraph=dg)
    got = shape(me)
    bpy.data.meshes.remove(me)
    return got


expected = [evaluated(one), evaluated(two)]
build.tag(one, "First", "board", listed="board", group="top", shape="box")
build.tag(two, "Second", "board", listed="board", group="top", shape="box")
print("TAGGED SHARED", one.data == two.data, one.data.users)
register(pack, first)
register(pack, second)
pack.finish(["First", "Second"])
got = [shape(ob.data) for ob in (one, two)]
print("GEOMETRY", got == expected, got[0][:2], got[1][:2], expected[0][:2], expected[1][:2])
print("MOVED", expected[0][2:] != expected[1][2:])
print("SAME DATA", one.data == two.data, one.data.users, len(one.modifiers), len(two.modifiers))
runner.done()
"""


def _instanced_nodes(out: Path) -> tuple[dict, list[dict]]:
    glb = _glb_json(out / "shop.glb")
    nodes = [n for n in glb["nodes"] if n.get("name") in ("First/board", "Second/board")]
    return glb, sorted(nodes, key=lambda n: n["name"])


def test_instanced_meshes_stay_shared(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, INSTANCED, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "TAGGED SHARED True 2" in lines  # tag leaves an instance sharing its mesh
    assert [x for x in lines if x.startswith("GEOMETRY True")], result.log
    glb, nodes = _instanced_nodes(out)
    assert len(nodes) == 2
    assert nodes[0]["mesh"] == nodes[1]["mesh"]  # one glTF mesh, used twice
    assert len(glb["meshes"]) == 1
    assert "SAME DATA True 2 0 0" in lines  # applied once, relinked, modifiers gone
    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    assert [p.name for p in pack.pieces["First"].parts] == ["board"]
    assert [p.name for p in pack.pieces["Second"].parts] == ["board"]


def test_instances_with_different_modifiers_do_not_share(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, INSTANCED, str(out), "extra")
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert [x for x in lines if x.startswith("GEOMETRY True")], result.log
    assert "SAME DATA False 1 0 0" in lines
    glb, nodes = _instanced_nodes(out)
    assert nodes[0]["mesh"] != nodes[1]["mesh"]
    assert len(glb["meshes"]) == 2


def test_instances_whose_modifiers_read_their_place_do_not_share(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, INSTANCED, str(out), "mirror")
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "MOVED True" in lines  # the two evaluate to different places, so the plant bites
    assert [x for x in lines if x.startswith("GEOMETRY True")], result.log  # each keeps its own
    assert "SAME DATA False 1 0 0" in lines
    glb, nodes = _instanced_nodes(out)
    assert nodes[0]["mesh"] != nodes[1]["mesh"]
    assert len(glb["meshes"]) == 2


def test_instances_whose_modifiers_write_their_uvs_from_their_place_do_not_share(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, INSTANCED, str(out), "uv")
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "MOVED True" in lines  # the two evaluate to different UVs, so the plant bites
    assert [x for x in lines if x.startswith("GEOMETRY True")], result.log  # each keeps its own
    assert "SAME DATA False 1 0 0" in lines
    glb, nodes = _instanced_nodes(out)
    assert nodes[0]["mesh"] != nodes[1]["mesh"]
    assert len(glb["meshes"]) == 2


PARTIAL = ADOPT_HEAD + """
pack = build.PackBuilder("shop", 1.0, out_dir, 2.5, crease_deg=35)
bench = build.piece("Bench")
build.slab("whole", -0.5, -0.3, 0, 0.2, -0.1, 0.1, mat, bench,
           props={"listed": "board", "group": "legs"})
part = build.slab("nolisted", 0, 1, 0, 1, 0, 1, mat, bench, props={"group": "top"})
print("MADE", part.name, part["group"], part["smooth"], "build_order" in part, "listed" in part)
register(pack, bench)
try:
    pack.finish(["Bench"])
    print("NO RAISE", "finish")
except ManifestError as exc:
    print("RAISED finish", exc)
runner.done()
"""


def test_mesh_obj_with_partial_props_waits_for_finish(tmp_path):
    result = _run(tmp_path, PARTIAL, str(tmp_path / "out"))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert "MADE Bench/nolisted top True True False" in lines
    (refused,) = [x for x in lines if x.startswith("RAISED finish")]
    assert "'Bench/nolisted'" in refused and "'listed'" in refused
    assert not [x for x in lines if x.startswith("NO RAISE")]
    assert not (tmp_path / "out").exists()  # nothing written


# --- keep-offs and connectors --------------------------------------------------------------

RECORDS = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.manifest import End, Listed, Recipe, Ref, View

VIEW = View(1.0, "a test piece, seen from arm's length")
out_dir = Path(runner.script_args()[0])
OWN = Ref("self", None, "plug")
FAR = Ref("shop", "Bench_1", "socket")


target = out_dir


def lamp_pack(out, cable_shape="cable"):
    build.reset()
    fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
    pack = build.PackBuilder("shop", 1.0, out, 2.0, crease_deg=35)
    root = build.piece("Lamp")
    build.slab("body", -0.1, 0.1, 0, 0.2, -0.1, 0.1, fill, root,
               props={"listed": "body", "group": "body"})
    build.slab("lead", 0.1, 0.4, 0.0, 0.01, -0.01, 0.01, fill, root,
               props={"listed": "lead", "group": "lead", "shape": cable_shape})
    pack.register(root, view=VIEW, origin="floor", recipe=Recipe("assembly", "x.py"),
                  listed=[Listed("body", "A body", "a drawing", 0.01, None),
                          Listed("lead", "A lead", "a drawing", 0.01, None)],
                  done="A lamp.", size_class="prop", kind="light")
    pack.use(root, "front", point=(0, 0, 0.3), facing=(0, 0, -1), posture="stand")
    pack.mount(root, "plug", kind="power", point=(0.1, 0.1, 0), axis=(1, 0, 0), takes="lead",
               part="body")
    return pack, root


def ends(plug="body"):
    return [End((0.1, 0.1, 0), None, OWN), End((0.5, 0.2, 1), "shop", FAR, plug)]


def fine():
    pack, root = lamp_pack(target)
    pack.keep_off(root, "coupler", point=(0.0, 0.1, 0.0), radius=0.05, why="truss coupler")
    rec = pack.connector(root, "feed", part="lead", ends=ends(), cap=1.5, why="mains lead")
    print("RECORD", rec.id, rec.cap, rec.ends[1].layout)
    pack.finish(["Lamp"])


def radius_zero():
    pack, root = lamp_pack(target)
    pack.keep_off(root, "coupler", point=(0, 0, 0), radius=0, why="x")


def radius_nan():
    pack, root = lamp_pack(target)
    pack.keep_off(root, "coupler", point=(0, 0, 0), radius=float("nan"), why="x")


def point_short():
    pack, root = lamp_pack(target)
    pack.keep_off(root, "coupler", point=(0, 0), radius=0.1, why="x")


def cap_zero():
    pack, root = lamp_pack(target)
    pack.connector(root, "feed", part="lead", ends=ends(), cap=0, why="x")


def one_end():
    pack, root = lamp_pack(target)
    pack.connector(root, "feed", part="lead", ends=ends()[:1], cap=1, why="x")


def end_not_an_end():
    pack, root = lamp_pack(target)
    pack.connector(root, "feed", part="lead", ends=[ends()[0], ((0, 0, 0), None, FAR)], cap=1,
                   why="x")


def missing_part():
    pack, root = lamp_pack(target)
    pack.connector(root, "feed", part="flex", ends=ends(), cap=1, why="x")
    pack.finish(["Lamp"])


def not_a_cable():
    pack, root = lamp_pack(target, cable_shape="box")
    pack.connector(root, "feed", part="lead", ends=ends(), cap=1, why="x")
    pack.finish(["Lamp"])


def plug_names_no_part():
    pack, root = lamp_pack(target)
    pack.connector(root, "feed", part="lead", ends=ends("ghost"), cap=1, why="x")
    pack.finish(["Lamp"])


def mount_part_not_a_name():
    pack, root = lamp_pack(target)
    pack.mount(root, "again", kind="rest", point=(0, 0, 0), axis=(0, 1, 0), takes="lead", part=3)


def repeated_keep_off():
    pack, root = lamp_pack(target)
    for _ in range(2):
        pack.keep_off(root, "coupler", point=(0, 0, 0), radius=0.1, why="x")
    pack.finish(["Lamp"])


def repeated_connector():
    pack, root = lamp_pack(target)
    for _ in range(2):
        pack.connector(root, "feed", part="lead", ends=ends(), cap=1, why="x")
    pack.finish(["Lamp"])


for case in (fine, radius_zero, radius_nan, point_short, cap_zero, one_end, end_not_an_end,
             missing_part, not_a_cable, plug_names_no_part, mount_part_not_a_name,
             repeated_keep_off, repeated_connector):
    target = out_dir if case is fine else out_dir.parent / "refused" / case.__name__
    try:
        case()
        print("DONE", case.__name__)
    except Exception as exc:
        print("RAISED", case.__name__, type(exc).__name__, exc)
    print("FILES", case.__name__, sorted(p.name for p in target.iterdir())
          if target.exists() else [])
runner.done()
"""


def test_keep_offs_and_connectors(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, RECORDS, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    raised = {line.split()[1]: line for line in lines if line.startswith("RAISED ")}
    # The good build finished; no refusal escaped.
    assert "DONE fine" in lines and "RECORD feed 1.5 shop" in lines
    assert sorted(raised) == ["cap_zero", "end_not_an_end", "missing_part",
                              "mount_part_not_a_name", "not_a_cable", "one_end",
                              "plug_names_no_part", "point_short", "radius_nan", "radius_zero",
                              "repeated_connector", "repeated_keep_off"], result.log
    assert "coupler.radius" in raised["radius_zero"] and "above 0" in raised["radius_zero"]
    assert "coupler.radius" in raised["radius_nan"]
    assert "coupler.point" in raised["point_short"]
    assert "feed.cap" in raised["cap_zero"] and "above 0" in raised["cap_zero"]
    assert "feed.ends" in raised["one_end"]
    assert "feed.ends[1]" in raised["end_not_an_end"]
    # finish names the piece and the part, and writes nothing.
    assert "'Lamp'" in raised["missing_part"] and "'flex'" in raised["missing_part"]
    assert "FILES missing_part []" in lines
    assert "cable" in raised["not_a_cable"] and "FILES not_a_cable []" in lines
    # a plug that names no part of the piece is the loader's refusal; a part that is no name is
    # refused as the mount is made
    assert "'ghost'" in raised["plug_names_no_part"]
    assert "FILES plug_names_no_part []" in lines
    assert "again.part" in raised["mount_part_not_a_name"]
    for dup in ("repeated_keep_off", "repeated_connector"):
        assert "repeat" in raised[dup]
        assert f"FILES {dup} []" in lines

    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    lamp = pack.pieces["Lamp"]
    (keep,) = lamp.keep_off
    assert (keep.id, keep.point, keep.radius, keep.why) == (
        "coupler", (0.0, 0.1, 0.0), 0.05, "truss coupler")
    (conn,) = lamp.connectors
    assert (conn.id, conn.part, conn.cap, conn.why) == ("feed", "lead", 1.5, "mains lead")
    assert conn.ends == (End((0.1, 0.1, 0.0), None, Ref("self", None, "plug")),
                         End((0.5, 0.2, 1.0), "shop", Ref("shop", "Bench_1", "socket"), "body"))
    assert lamp.mounts[0].part == "body"


OUTLINED = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Listed, Recipe, View

out_dir = Path(runner.script_args()[0])
build.reset()
fill = build.slots({"Fill": (1, 1, 1)})["Fill"]
pack = build.PackBuilder("shop", 1.0, out_dir, 2.0, crease_deg=35)


def add(name, **extra):
    root = build.piece(name)
    build.slab("body", -0.1, 0.1, 0, 0.2, -0.1, 0.1, fill, root,
               props={"listed": "body", "group": "body"})
    pack.register(root, view=View(1.0, "a test piece, seen from arm's length"), origin="floor",
                  recipe=Recipe("assembly", "x.py"),
                  listed=[Listed("body", "A box", "a drawing", 0.01, None)], done="A box.",
                  size_class="prop", kind="box", **extra)
    return root


add("Flat", outlined=False, outlined_why="the page draws it with no lines")
add("Lined")
try:
    add("Bad", outlined="no")
    print("NO RAISE", "non-bool")
except ManifestError as exc:
    print("RAISED", "non-bool", exc)
pack.finish(["Flat", "Lined"])
runner.done()
"""


def test_register_carries_outlined_into_the_pack(tmp_path):
    out = tmp_path / "out"
    result = _run(tmp_path, OUTLINED, str(out))
    assert result.ok, result.log
    lines = result.log.splitlines()
    assert not [line for line in lines if line.startswith("NO RAISE")], result.log
    assert any(line.startswith("RAISED non-bool") and "Bad" in line for line in lines), result.log
    pack = load_pack(out / "shop.pack.json", Style(UNITS, {}, {}))
    assert (pack.pieces["Flat"].outlined, pack.pieces["Flat"].outlined_why) == (
        False, "the page draws it with no lines")
    assert (pack.pieces["Lined"].outlined, pack.pieces["Lined"].outlined_why) == (True, "")
