"""Soft parts in Blender: blob, ring and meta forms, seats and mirrors."""

import json
from pathlib import Path

import pytest

from model_kit.blender.runner import run_blender

TIMEOUT = 240

pytestmark = pytest.mark.blender

# A metaball shin: two balls and an ellipsoid, as a build passes them.
SHIN = """
SHIN = (Element("ball", (0.105, 0.107, 0.006), (0.058, 0.058, 0.058)),
        Element("ball", (0.105, 0.085, 0.008), (0.056, 0.056, 0.056)),
        Element("ellipsoid", (0.105, 0.046, 0.038), (0.072, 0.044, 0.084)))
"""

PRELUDE = '''
import json, math
import bpy, numpy as np
from mathutils import Vector
from model_kit.blender import build, runner, soft
from model_kit.core import forms, geom
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Part
from model_kit.core.sight import DecalLift
from model_kit.core.manifest.forms import (
    BlobForm, DecalForm, Element, Flatten, MetaForm, RingForm, SeatOn, Smooth, SoftForm,
    Station,
)
from mathutils.bvhtree import BVHTree

build.reset()
MATS = build.slots({"Fill": (1, 1, 1)})
ROOT = build.piece("toy")
''' + SHIN + '''

def part(name, shape, body=None, seat=None, deforms=(), mirror_of=None, on=None):
    form = SoftForm(body, mirror_of, seat, tuple(deforms), None)
    role = "decal" if shape == "decal" else "solid"  # a decal needs a printed role
    return Part(name=name, listed="soft", shape=shape, on=on, group=name, role=role,
                bones=(), bind=None, form=form)

def without_on(p):
    """The part with its `on` taken away past `Part.validate`: `soft_part` keeps its own guard for
    a decal with no host, and this proves it."""
    object.__setattr__(p, "on", None)
    return p

def blob(half=(0.15, 0.2, 0.1), centre=(0.0, 0.5, 0.0), powers=(0.8, 0.7, 1.2)):
    return BlobForm(half, centre, *powers, 16, 8)

def piece_verts(ob):
    """The object's vertices in the piece's glTF frame."""
    mw = ROOT.matrix_world.inverted() @ ob.matrix_world
    return np.array([build.G3(mw @ v.co) for v in ob.data.vertices])

def faces_of(ob):
    return [tuple(p.vertices) for p in ob.data.polygons]

def tris_of(ob):
    me = ob.data
    me.calc_loop_triangles()
    v = piece_verts(ob)
    return np.array([[v[i] for i in t.vertices] for t in me.loop_triangles])

def show(**kw):
    print("RESULT " + json.dumps(kw))
'''


def _run(tmp_path: Path, body: str) -> dict:
    script = tmp_path / "script.py"
    script.write_text(PRELUDE + body + "\nrunner.done()\n")
    result = run_blender(script, [], timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    lines = [ln for ln in result.log.splitlines() if ln.startswith("RESULT ")]
    assert len(lines) == 1, result.log
    return json.loads(lines[0][len("RESULT "):])


BLOB = """
b = blob()
ob = soft.soft_part(part("body", "blob", b, on=None), MATS["Fill"], ROOT)
want, want_faces = forms.blob_mesh(b)
diff = float(np.abs(piece_verts(ob) - want).max())

flat = Flatten("z", 0.03, 0.05)
ob2 = soft.soft_part(part("shell", "blob", b, deforms=[flat], on="body"), MATS["Fill"], ROOT)
want2 = forms.apply_deforms(want, [flat])
diff2 = float(np.abs(piece_verts(ob2) - want2).max())

ring = RingForm(top=(0.2, 0.6, 0.0), bottom=(0.2, 0.3, 0.02), rings=8, segments=12,
                stations=(Station(0.0, 0.05, 0.05), Station(1.0, 0.04, 0.04)),
                smooth=Smooth(2, 0.5, None))
ob3 = soft.soft_part(part("arm", "ring", ring), MATS["Fill"], ROOT)
want3, want3_faces = forms.ring_mesh(ring)
diff3 = float(np.abs(piece_verts(ob3) - want3).max())

show(diff=diff, diff2=diff2, diff3=diff3, name=ob.name, parent=ob.parent.name,
     faces=faces_of(ob) == [tuple(f) for f in want_faces],
     faces3=faces_of(ob3) == [tuple(f) for f in want3_faces],
     props={k: ob[k] for k in ("listed", "group", "shape", "role")},
     on=ob.get("on"), on2=ob2.get("on"), n=len(ob.data.vertices),
     volume=geom.signed_volume(tris_of(ob)), closed=geom.is_closed(tris_of(ob)))
"""


def test_blob_part_matches_core(tmp_path):
    r = _run(tmp_path, BLOB)
    assert r["diff"] < 1e-6 and r["diff2"] < 1e-6 and r["diff3"] < 1e-6, r
    assert r["faces"] and r["faces3"]
    assert r["name"] == "toy/body" and r["parent"] == "toy"
    assert r["props"] == {"listed": "soft", "group": "body", "shape": "blob", "role": "solid"}
    assert r["on"] is None and r["on2"] == "body"
    assert r["n"] == 114 and r["closed"] and r["volume"] > 0


BALL = """
m = MetaForm((Element("ball", (0.0, 0.5, 0.0), (0.1, 0.1, 0.1)),), 0.006, 0.6,
             Smooth(0, 0.5, None))
ob = soft.soft_part(part("ball", "meta", m), MATS["Fill"], ROOT)
d = np.linalg.norm(piece_verts(ob) - np.array([0.0, 0.5, 0.0]), axis=1)
show(mean=float(d.mean()), n=len(d), metaballs=len(bpy.data.metaballs),
     objects=sorted(o.name for o in bpy.data.objects))
"""


def test_meta_ball_radius(tmp_path):
    r = _run(tmp_path, BALL)
    assert abs(r["mean"] - 0.1) < 0.002, r
    assert r["metaballs"] == 0
    assert r["objects"] == ["toy", "toy/ball"], r


SHIN_PART = """
m = MetaForm(SHIN, 0.006, 0.6, Smooth(5, 0.5, None), decimate=0.4)
ob = soft.soft_part(part("shin", "meta", m), MATS["Fill"], ROOT)
tris = tris_of(ob)
# One island: a vertex walk over the polygons reaches every vertex.
seen, todo = set(), [0]
adj = {}
for f in faces_of(ob):
    for a in f:
        adj.setdefault(a, set()).update(f)
while todo:
    k = todo.pop()
    if k not in seen:
        seen.add(k)
        todo.extend(adj[k] - seen)
show(closed=geom.is_closed(tris), volume=geom.signed_volume(tris), islands=len(seen) == len(ob.data.vertices),
     n=len(ob.data.vertices), faces=len(ob.data.polygons), metaballs=len(bpy.data.metaballs),
     meshes=sorted(bpy.data.meshes.keys()))
"""


def test_meta_part_closed(tmp_path):
    r = _run(tmp_path, SHIN_PART)
    assert r["closed"] and r["volume"] > 0 and r["islands"], r
    assert r["metaballs"] == 0
    assert r["meshes"] == ["toy/shin"], r


BELOW = """
elements = (Element("ball", (0.0, 0.2, 0.0), (0.1, 0.1, 0.1)),
            Element("ball", (0.0, 0.4, 0.0), (0.1, 0.1, 0.1)))
raw = soft.soft_part(part("raw", "meta", MetaForm(elements, 0.01, 0.6, Smooth(0, 0.5, None))),
                     MATS["Fill"], ROOT)
sm = soft.soft_part(part("sm", "meta", MetaForm(elements, 0.01, 0.6, Smooth(4, 0.5, 0.27))),
                    MATS["Fill"], ROOT)
a, b = piece_verts(raw), piece_verts(sm)
high, low = a[:, 1] >= 0.27, a[:, 1] < 0.27
move = np.linalg.norm(b - a, axis=1)
show(same_count=len(a) == len(b), high_max=float(move[high].max()),
     low_max=float(move[low].max()), n_high=int(high.sum()), n_low=int(low.sum()))
"""


def test_smooth_below_moves_only_low_verts(tmp_path):
    r = _run(tmp_path, BELOW)
    assert r["same_count"] and r["n_high"] > 10 and r["n_low"] > 10, r
    assert r["high_max"] < 1e-6, r
    assert r["low_max"] > 1e-4, r


MIRROR = """
ring = RingForm(top=(0.2, 0.6, 0.0), bottom=(0.25, 0.3, 0.05), rings=8, segments=12,
                stations=(Station(0.0, 0.05, 0.04), Station(1.0, 0.03, 0.05, 0.01, 0.0)))
host = soft.soft_part(part("body", "blob", blob()), MATS["Fill"], ROOT)
src = soft.soft_part(part("arm_L", "ring", ring, seat=SeatOn("body", (0.0, 0.5), 0.01)),
                     MATS["Fill"], ROOT, host=host)
plain = soft.soft_part(part("arm_P", "ring", ring), MATS["Fill"], ROOT)
moved = float(np.abs(piece_verts(src) - piece_verts(plain)).max())
out = {}

def tri_set(tris):
    # Each triangle as a sorted tuple of rounded corners, so two meshes compare by shape.
    return {tuple(sorted(tuple(round(float(c), 6) + 0.0 for c in p) for p in t)) for t in tris}

for key, source, name in (("plain", plain, "arm_Q"), ("seated", src, "arm_R")):
    twin = soft.mirror_part(part(name, "ring", mirror_of=source.name.split("/")[1]),
                            source, MATS["Fill"], ROOT)
    sv, tv = piece_verts(source), piece_verts(twin)
    flipped = sv * np.array([-1.0, 1.0, 1.0])
    source.data.calc_loop_triangles()
    out[key] = dict(
        diff=float(np.abs(tv - flipped).max()),
        faces=faces_of(twin) == [tuple(reversed(t.vertices)) for t in source.data.loop_triangles],
        vol_src=geom.signed_volume(tris_of(source)), vol_twin=geom.signed_volume(tris_of(twin)),
        closed=geom.is_closed(tris_of(twin)), props=twin["group"], name=twin.name,
        tri_faces=all(len(f) == 3 for f in faces_of(twin)),
        same_tris=tri_set(tris_of(twin)) == tri_set(tris_of(source) * np.array([-1.0, 1.0, 1.0])))
try:
    soft.mirror_part(part("arm_Z", "ring", mirror_of="arm_L"), plain, MATS["Fill"], ROOT)
    wrong = "no raise"
except ManifestError as exc:
    wrong = str(exc)
show(out=out, wrong=wrong, moved=moved)
"""


def test_mirror_is_outward(tmp_path):
    r = _run(tmp_path, MIRROR)
    for key in ("plain", "seated"):
        o = r["out"][key]
        assert o["diff"] < 1e-6, (key, r)
        assert o["faces"] and o["closed"], (key, r)
        assert o["vol_twin"] > 0 and abs(o["vol_twin"] - o["vol_src"]) < 1e-7, (key, r)
        # The exporter splits a quad by its vertex order, which a mirror reverses: the twin is
        # made of the source's own triangles, so the two ship triangle for triangle.
        assert o["tri_faces"] and o["same_tris"], (key, r)
    assert r["moved"] > 0.05, r  # the seated source really sits elsewhere
    assert r["out"]["plain"]["name"] == "toy/arm_Q"
    assert "arm_L" in r["wrong"] and "arm_P" in r["wrong"], r["wrong"]


MIRROR_SOURCE = """
ring = RingForm(top=(0.2, 0.6, 0.0), bottom=(0.25, 0.3, 0.05), rings=8, segments=12,
                stations=(Station(0.0, 0.05, 0.04), Station(1.0, 0.03, 0.05, 0.01, 0.0)))
src = soft.soft_part(part("arm_L", "ring", ring), MATS["Fill"], ROOT)
sizes = lambda ob: [len(p.vertices) for p in ob.data.polygons]
before, verts_before = sizes(src), piece_verts(src).tolist()
twin = soft.mirror_part(part("arm_Q", "ring", mirror_of="arm_L"), src, MATS["Fill"], ROOT)
show(before=before, after=sizes(src), twin=sorted(set(sizes(twin))),
     verts_kept=piece_verts(src).tolist() == verts_before)
"""


def test_mirror_part_leaves_its_source_untriangulated(tmp_path):
    """The twin is made from a triangulated copy: the source keeps the quads it was built with,
    so a part seated on it later casts rays against those faces (#24)."""
    r = _run(tmp_path, MIRROR_SOURCE)
    assert 4 in r["before"], r  # the ring source really has quads
    assert r["after"] == r["before"], r
    assert r["verts_kept"], r
    assert r["twin"] == [3], r


SURFACE = """
sphere = BlobForm((0.2, 0.2, 0.2), (0.0, 0.5, 0.0), 1.0, 1.0, 1.0, 32, 16)
host = soft.soft_part(part("head", "blob", sphere), MATS["Fill"], ROOT)
verts, faces = soft.decal_mesh("p", DecalForm((0.05, 0.04), 16, 4, (0.08, 0.55, 0.0)), host, 0.002)

def verdict(v, f, **kw):
    try:
        soft._check_surface("p", v, f, closed=False, **kw)
        return "no raise"
    except ManifestError as exc:
        return str(exc)

# Two triangles that meet at vertex 0 only: each edge is clean and the mesh is one piece by
# vertex, but vertex 0 has two fans.
bowtie = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], float)
show(good=verdict(verts, faces, host=host),
     bowtie=verdict(bowtie, [(0, 1, 2), (0, 3, 4)]),
     inward=verdict(verts, [tuple(reversed(f)) for f in faces], host=host),
     inward_no_host=verdict(verts, [tuple(reversed(f)) for f in faces]))
"""


def test_open_patch_with_a_bowtie_is_refused(tmp_path):
    r = _run(tmp_path, SURFACE)
    assert r["good"] == "no raise", r
    assert "more than one fan" in r["bowtie"] and "'p'" in r["bowtie"], r


def test_open_patch_wound_into_its_host_is_refused(tmp_path):
    r = _run(tmp_path, SURFACE)
    assert "wound against its host" in r["inward"], r
    # Without a host (a mirrored patch) the facing is not judged: its source was.
    assert r["inward_no_host"] == "no raise", r


BAD_VIEW = """
import dataclasses
import json
from pathlib import Path

import bpy

from model_kit.blender import build, runner, soft
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import load_pack
from model_kit.core.units import load_style

toy, style, out = runner.script_args()[:3]
pack = load_pack(Path(toy), load_style(Path(style)))
piece = pack.pieces["toy"]
build.reset()
mats = build.slots({"Fill": (1, 1, 1)})
builder = build.PackBuilder(pack.name, pack.units_per_metre, Path(out), pack.spacing, crease_deg=35)

def count():
    return [len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.armatures)]

before, said = count(), {}
for label, view in (("none", None), ("a number", 1.0)):
    try:
        soft.build_piece(builder, dataclasses.replace(piece, view=view),
                         {p.name: mats["Fill"] for p in piece.parts})
        said[label] = "no raise"
    except ManifestError as exc:
        said[label] = str(exc)
print("RESULT " + json.dumps({"before": before, "after": count(), "said": said}))
runner.done()
"""


def test_build_piece_with_bad_view_leaves_no_objects(tmp_path):
    """`build_piece` asks the builder to check the registration first (#65): a piece with no
    view, or a plain number for one, raises before any object, mesh or armature exists."""
    script = tmp_path / "script.py"
    script.write_text(BAD_VIEW)
    fixtures = Path(__file__).resolve().parents[1] / "core" / "fixtures"
    result = run_blender(script, [str(fixtures / "toy.pack.json"),
                                  str(fixtures / "shop.style.json"), str(tmp_path / "out")],
                         timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    line = next(ln for ln in result.log.splitlines() if ln.startswith("RESULT "))
    r = json.loads(line[len("RESULT "):])
    assert r["before"] == r["after"] == [0, 0, 0], r
    assert all("no view" in said and "'toy'" in said for said in r["said"].values()), r


SHIPPED_SOURCE = """
import dataclasses
import json
from pathlib import Path

import bpy
import numpy as np

from model_kit.blender import build, runner, soft
from model_kit.core.manifest import View, load_pack
from model_kit.core.units import load_style

toy, style, out = runner.script_args()[:3]
pack = load_pack(Path(toy), load_style(Path(style)))
piece = dataclasses.replace(pack.pieces["toy"], view=View(1.0, "a test figure, seen from arm's length"))
build.reset()
mats = build.slots({"Fill": (1, 1, 1)})
builder = build.PackBuilder(pack.name, pack.units_per_metre, Path(out), pack.spacing, crease_deg=35)
soft.build_piece(builder, piece, {p.name: mats["Fill"] for p in piece.parts})
src, twin = bpy.data.objects["toy/arm_L"], bpy.data.objects["toy/arm_R"]

def sizes(ob):
    return sorted({len(p.vertices) for p in ob.data.polygons})

def tris(ob, flip):
    ob.data.calc_loop_triangles()
    return {tuple(sorted(tuple(round(float(c), 6) + 0.0 for c in flip * np.array(ob.data.vertices[i].co))
                         for i in t.vertices)) for t in ob.data.loop_triangles}

print("RESULT " + json.dumps({
    "src": sizes(src), "twin": sizes(twin),
    "mirror": tris(src, np.array([-1.0, 1.0, 1.0])) == tris(twin, np.array([1.0, 1.0, 1.0]))}))
runner.done()
"""


def test_build_piece_ships_a_mirror_source_split_by_beauty(tmp_path):
    """A mirror source is split into triangles by `beauty_triangulate` (a named step of
    `build_piece`), so it ships those diagonals, not the exporter's, which fold over the crease
    angle on bar 03's shins (a dash at a crease). Its twin is the exact mirror of it."""
    script = tmp_path / "script.py"
    script.write_text(SHIPPED_SOURCE)
    fixtures = Path(__file__).resolve().parents[1] / "core" / "fixtures"
    result = run_blender(script, [str(fixtures / "toy.pack.json"),
                                  str(fixtures / "shop.style.json"), str(tmp_path / "out")],
                         timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    line = next(ln for ln in result.log.splitlines() if ln.startswith("RESULT "))
    r = json.loads(line[len("RESULT "):])
    assert r == {"src": [3], "twin": [3], "mirror": True}, r


SEAT = """
ROOT.location = (0.3, -0.2, 0.1)
ROOT.rotation_euler = (0.2, 0.3, 0.9)
bpy.context.view_layer.update()
sphere = BlobForm((0.2, 0.2, 0.2), (0.0, 0.5, 0.0), 1.0, 1.0, 1.0, 32, 16)
host = soft.soft_part(part("head", "blob", sphere), MATS["Fill"], ROOT)
lens_form = BlobForm((0.05, 0.05, 0.02), (0.0, 0.0, 0.0), 1.0, 1.0, 1.0, 16, 8)
AT, SINK = (0.08, 0.55), 0.015
lens = soft.soft_part(part("lens", "blob", lens_form, seat=SeatOn("head", AT, SINK)),
                      MATS["Fill"], ROOT, host=host)

# The expected hit, by a plain ray against the host's triangles (Moller-Trumbore), in the piece frame.
tris = tris_of(host)
origin, direction = np.array([AT[0], AT[1], 5.0]), np.array([0.0, 0.0, -1.0])
best = None
for t in tris:
    e1, e2 = t[1] - t[0], t[2] - t[0]
    p = np.cross(direction, e2)
    det = e1 @ p
    if abs(det) < 1e-12:
        continue
    s = origin - t[0]
    u = (s @ p) / det
    q = np.cross(s, e1)
    v = (direction @ q) / det
    dist = (e2 @ q) / det
    if u >= 0 and v >= 0 and u + v <= 1 and dist > 0 and (best is None or dist < best[0]):
        n = np.cross(e1, e2)
        best = (dist, origin + direction * dist, n / np.linalg.norm(n))
hit, normal = best[1], best[2]

# The lens is a symmetric blob built about the origin, so the object's origin is its centre.
centre = np.array(build.G3(ROOT.matrix_world.inverted() @ lens.matrix_world.translation))
turn = ROOT.matrix_world.inverted().to_3x3() @ lens.matrix_world.to_3x3()
front = turn @ Vector(build.B(0.0, 0.0, 1.0))      # the lens's local +Z (glTF front), in the piece frame
front_g = np.array(build.G3(front))

try:
    soft.seat_on(lens, host, SeatOn("head", (0.9, 0.5), 0.01))
    miss = "no raise"
except ManifestError as exc:
    miss = str(exc)
try:
    soft.soft_part(part("lens2", "blob", lens_form, seat=SeatOn("elsewhere", AT, SINK)),
                   MATS["Fill"], ROOT, host=host)
    other = "no raise"
except ManifestError as exc:
    other = str(exc)

show(hit=hit.tolist(), normal=normal.tolist(), front=front_g.tolist(),
     behind=(centre - hit).tolist(), miss=miss, other=other)
"""


def test_seat_sinks_and_faces_normal(tmp_path):
    r = _run(tmp_path, SEAT)
    # The centre sits `sink` (0.015) behind the hit along -Z; the lens's local +Z meets the normal.
    assert max(abs(a - b) for a, b in zip(r["front"], r["normal"], strict=True)) < 1e-6, r
    assert max(abs(a - b) for a, b in zip(r["behind"], [0.0, 0.0, -0.015], strict=True)) < 1e-5, r
    assert r["hit"][2] > 0.1, r
    assert "lens" in r["miss"] and "0.9" in r["miss"] and "0.5" in r["miss"], r["miss"]
    assert "elsewhere" in r["other"] and "head" in r["other"], r["other"]


REFUSE = """
errors = {}
for label, p in (
        ("no_form", Part("a", "soft", "blob", None, "a", "solid", (), None, None)),
        ("mirror", part("b", "blob", mirror_of="a")),
        ("no_host", part("c", "blob", blob(), seat=SeatOn("body", (0.0, 0.5), 0.01)))):
    try:
        soft.soft_part(p, MATS["Fill"], ROOT)
        errors[label] = "no raise"
    except ManifestError as exc:
        errors[label] = str(exc)
show(errors=errors)
"""


def test_soft_part_refuses_no_form_and_mirror(tmp_path):
    r = _run(tmp_path, REFUSE)
    assert "'a'" in r["errors"]["no_form"], r
    # A seated form with no host is refused, not left unseated.
    assert "'c'" in r["errors"]["no_host"] and "'body'" in r["errors"]["no_host"], r
    assert "mirror" in r["errors"]["mirror"] and "'b'" in r["errors"]["mirror"], r


DECAL = """
ROOT.location = (0.3, -0.2, 0.1)
ROOT.rotation_euler = (0.2, 0.3, 0.9)
bpy.context.view_layer.update()
sphere = BlobForm((0.2, 0.2, 0.2), (0.0, 0.5, 0.0), 1.0, 1.0, 1.0, 32, 16)
host = soft.soft_part(part("head", "blob", sphere), MATS["Fill"], ROOT)
LIFT = 0.001  # the piece's contact length
LF = DecalLift("piece 'toy'", LIFT, 0.0001)  # the depth gap at its farthest view, well under it
front = DecalForm((0.05, 0.04), 16, 4, (0.08, 0.55, 0.0))
side = DecalForm((0.05, 0.04), 16, 4, (0.0, 0.5, 0.0), (-1.0, 0.0, 0.0))

def made(name, form):
    return soft.soft_part(part(name, "decal", form, on="head"), MATS["Fill"], ROOT, host=host,
                          lift=LF)

bpy.context.view_layer.update()
host_tris = tris_of(host)
tree = BVHTree.FromPolygons([tuple(map(float, c)) for t in host_tris for c in t],
                            [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(host_tris))])

def measure(ob):
    bpy.context.view_layer.update()
    v = piece_verts(ob)
    gaps = [tree.find_nearest(Vector(tuple(map(float, p))))[3] for p in v]
    tris = tris_of(ob)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    out = ((tris.mean(axis=1) - np.array([0.0, 0.5, 0.0])) * normals).sum(axis=1)
    edges = [(f[k], f[(k + 1) % len(f)]) for f in faces_of(ob) for k in range(len(f))]
    open_edges = sum((b, a) not in set(edges) for a, b in edges)
    return {"gap_lo": min(gaps), "gap_hi": max(gaps), "outward": bool((out > 0).all()),
            "open": open_edges, "n": len(v), "mean": v.mean(axis=0).tolist(),
            "span": (v.max(axis=0) - v.min(axis=0)).tolist(), "shape": ob["shape"]}

dot = made("dot", front)
out = {"front": measure(dot), "side": measure(made("dot_side", side))}

# A host turned under the piece root: its local frame is not the piece frame, so the casts must
# read its normals in the piece frame. Turned about glTF Y through the sphere's centre.
turned = {}
for key, turn in (("a", 0.5), ("b", 1.2)):
    head = soft.soft_part(part(f"head_{key}", "blob", sphere), MATS["Fill"], ROOT)
    head.rotation_euler = (0.0, 0.0, turn)
    bpy.context.view_layer.update()
    ob = soft.soft_part(part(f"dot_{key}", "decal", front, on=f"head_{key}"), MATS["Fill"],
                        ROOT, host=head, lift=LF)
    bpy.context.view_layer.update()
    ht = tris_of(head)
    t2 = BVHTree.FromPolygons([tuple(map(float, c)) for t in ht for c in t],
                              [(3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(ht))])
    gaps = [t2.find_nearest(Vector(tuple(map(float, p))))[3] for p in piece_verts(ob)]
    turned[str(turn)] = [min(gaps), max(gaps)]
out["turned"] = turned
twin = soft.mirror_part(part("dot_twin", "decal", mirror_of="dot", on="head"), dot, MATS["Fill"],
                        ROOT)
bpy.context.view_layer.update()
a, b = piece_verts(dot), piece_verts(twin)
b[:, 0] = -b[:, 0]
def canon(tris):
    # Each triangle's corners in a fixed order, then the triangles in a fixed order (by a coarse
    # key, so float noise cannot reorder them), so two meshes compare corner for corner.
    tris = np.array([sorted(t.tolist(), key=lambda c: tuple(np.round(c, 4))) for t in tris])
    order = sorted(range(len(tris)), key=lambda i: tuple(np.round(tris[i].ravel(), 4)))
    return tris[order]

ca, cb = canon(tris_of(dot)), canon(tris_of(twin) * np.array([-1.0, 1.0, 1.0]))
same_tris = ca.shape == cb.shape and float(np.abs(ca - cb).max()) < 1e-6
out["mirror"] = {"diff": float(np.abs(a - b).max()), "same_tris": same_tris,
                 "open": measure(twin)["open"], "outward": measure(twin)["outward"]}

errors = {}
for label, p, kw in (
        ("no_host", part("e1", "decal", front, on="head"), {"lift": LF}),
        ("no_on", without_on(part("e2", "decal", front, on="head")), {"host": host, "lift": LF}),
        ("wrong_host", part("e3", "decal", front, on="body"), {"host": host, "lift": LF}),
        ("no_lift", part("e4", "decal", front, on="head"), {"host": host}),
        ("miss", part("e5", "decal", DecalForm((0.05, 0.04), 16, 4, (0.9, 0.5, 0.0)), on="head"),
         {"host": host, "lift": LF}),
        ("off_host", part("e6", "decal", DecalForm((0.5, 0.5), 16, 4, (0.0, 0.5, 0.0)),
                          on="head"), {"host": host, "lift": LF}),
        ("a_number", part("e7", "decal", front, on="head"), {"host": host, "lift": 0.001}),
        ("coarse", part("e8", "decal", DecalForm((0.05, 0.04), 3, 1, (0.08, 0.55, 0.0)),
                        on="head"), {"host": host, "lift": LF})):
    try:
        soft.soft_part(p, MATS["Fill"], ROOT, **kw)
        errors[label] = "no raise"
    except ManifestError as exc:
        errors[label] = str(exc)
try:
    DecalLift("piece 'toy'", 0.0001, 0.001)  # a lift under the depth gap
    errors["under_gap"] = "no raise"
except ManifestError as exc:
    errors["under_gap"] = str(exc)
show(lift=LIFT, out=out, errors=errors)
"""


def test_decal_follows_a_curved_host(tmp_path):
    r = _run(tmp_path, DECAL)
    lift = r["lift"]
    for key in ("front", "side"):
        o = r["out"][key]
        # Every vertex sits the lift off the host's surface, its faces turn outward, and only
        # the rim's 16 edges are open.
        assert abs(o["gap_lo"] - lift) < 1e-6 and abs(o["gap_hi"] - lift) < 1e-6, (key, r)
        assert o["outward"] and o["open"] == 16 and o["n"] == 65, (key, r)
        assert o["shape"] == "decal"
    for turn, (lo, hi) in r["out"]["turned"].items():
        assert abs(lo - lift) < 1e-5 and abs(hi - lift) < 1e-5, (turn, r["out"]["turned"])
    span = r["out"]["front"]["span"]
    assert span[0] > span[1], span  # the oval is 0.05 across and 0.04 high
    assert r["out"]["front"]["mean"][2] > 0.15, r  # cast from the front lands on the front
    assert r["out"]["side"]["mean"][0] > 0.15, r  # cast along -X lands on the +X side
    m = r["out"]["mirror"]
    assert m["diff"] < 1e-6 and m["same_tris"] and m["open"] == 16 and m["outward"], r
    e = r["errors"]
    assert "'e1'" in e["no_host"] and "host given is None" in e["no_host"], e
    assert "'e2'" in e["no_on"] and "its on is None" in e["no_on"], e
    assert "'body'" in e["wrong_host"] and "contact length" in e["no_lift"], e
    assert "contact length" in e["a_number"], e
    assert "'toy'" in e["under_gap"] and "0.0001000" in e["under_gap"], e
    assert "0.0010000" in e["under_gap"] and "too wide" in e["under_gap"], e
    assert "misses" in e["miss"] and "runs off" in e["off_host"], e
    assert "dips under its host" in e["coarse"], e


TOO_WIDE = """
import dataclasses
import json
from pathlib import Path

import bpy

from model_kit.blender import build, runner, soft
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import load_pack
from model_kit.core.sight import view_floor
from model_kit.core.units import load_style

bar, out = (Path(a) for a in runner.script_args()[:2])
style = load_style(bar / "style.json")
piece = load_pack(bar / "chibi.pack.json", style).pieces["chibi"]
# The same piece seen from as far as the camera's far plane: one depth step there is metres wide.
wide = dataclasses.replace(
    piece, view=dataclasses.replace(piece.view, farthest_m=style.depth_camera.far_m))
build.reset()
mats = build.slots({"Fill": (1, 1, 1)})
builder = build.PackBuilder("chibi", style.units_per_metre, out, 2.0, crease_deg=style.crease_deg)

def count():
    return [len(bpy.data.objects), len(bpy.data.meshes), len(bpy.data.armatures)]

before, said = count(), {}
for label, p, kw in (("wide", wide, {"style": style}), ("no_style", piece, {})):
    try:
        soft.build_piece(builder, p, {q.name: mats["Fill"] for q in p.parts}, **kw)
        said[label] = "no raise"
    except ManifestError as exc:
        said[label] = str(exc)
print("RESULT " + json.dumps({"before": before, "after": count(), "said": said,
                              "lift": view_floor(piece.view, style.units_per_metre)}))
runner.done()
"""


def test_a_view_range_too_wide_for_a_decal_makes_the_build_raise(tmp_path):
    """The eyes are lifted by the contact length. Seen out to the far plane the depth step is
    wider than that lift, so the build raises, naming the piece, the lift and the gap, before it
    makes an object. A piece with a decal and no style to check it against raises too."""
    script = tmp_path / "script.py"
    script.write_text(TOO_WIDE)
    bar = Path(__file__).resolve().parents[2] / "bars" / "03-chibi"
    result = run_blender(script, [str(bar), str(tmp_path / "out")], timeout=TIMEOUT)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    line = next(ln for ln in result.log.splitlines() if ln.startswith("RESULT "))
    r = json.loads(line[len("RESULT "):])
    assert r["before"] == r["after"] == [0, 0, 0], r
    wide = r["said"]["wide"]
    assert "'chibi'" in wide and f"{r['lift']:.7f}" in wide and "too wide" in wide, r
    assert "depth gap" in wide, r
    assert "'chibi'" in r["said"]["no_style"] and "style" in r["said"]["no_style"], r
