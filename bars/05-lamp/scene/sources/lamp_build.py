"""Build bar 05, a desk lamp and its power strip, in Blender. Run it through
`model_kit.blender.runner.run_blender` (bars/05-lamp/build.py does):

    blender --background --python lamp_build.py -- <out folder> <scene folder> <style.json>

Three pieces in one pack, `lamp` (the third is `Desk`, below):

- `Lamp`: the Anglepoise Type 75 Mini desk lamp (the maker's US technical sheet, "US - V2"): a
  cone-shaped base, two round arms, a bell shade, the cable and its 2-blade NEMA 1-15 plug.
- `Strip`: the Tripp Lite PS615 six-outlet power strip (the maker's datasheet): one housing with
  six NEMA 5-15R outlets cut in as real bores, a switch and the printed "TRIPP-LITE" logo.

The lamp's cable (the connector `mains`) runs from the cord exit on the base to the strip's
outlet `socket_3`, and the plug's two blades stand in that outlet's two blade bores. The layout
`desk` (written into the scene folder) puts the two pieces on a third, `Desk`, a plain slab: y 0 is
the desk top, and the lamp's base and the strip's back rest on it (`rests_on`).

Units. glTF metres (`units_per_metre` is 1). Axes: X right, Y up, Z front. Every size below is
either published (the sheets and the NEMA numbers named next to it) or a modelling choice (marked
`CHOICE`, with its reason). The sources do not publish an arm segment length, the base height,
the plug body, the cable diameter or any bore depth or clearance.
"""

import math
from itertools import pairwise
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from model_kit.blender import build, runner, soft
from model_kit.blender.export import export_pieces
from model_kit.core.manifest import (
    DecalForm,
    End,
    Layout,
    LayoutPlacement,
    Listed,
    Part,
    Recipe,
    Ref,
    SoftForm,
    View,
)
from model_kit.core.sight import DecalLift, decal_lift
from model_kit.core.units import load_style
from model_kit.core.writer import write_layout

PACK = "lamp"
LAYOUT = "desk"
INCH = 0.0254
MM = 0.001

LAMP_URL = ("https://cdn11.bigcommerce.com/s-kvxianwgwu/content/product-files/"
            "type-75-mini-desk-lamp.pdf")
STRIP_URL = ("https://assets.testequity.com/te1/Documents/pdf/tripplite/"
             "Tripp-Lite_PS615_Datasheet.pdf")
NEMA_URL = "https://en.wikipedia.org/wiki/NEMA_connector"
SPACING = 1.0  # metres between the two pieces in the pack: the lamp piece spans about 0.7 m of X

# Colours of the roles (linear, 0 to 1). The look is flat colour; a part picks one by name.
ROLES = {
    "Paint": (0.045, 0.045, 0.05),  # the lamp's matt paint finish ("Matt paint finish")
    "Iron": (0.16, 0.16, 0.17),  # the cast iron base
    "Cord": (0.02, 0.02, 0.02),
    "Brass": (0.62, 0.40, 0.11),
    "Plastic": (0.56, 0.56, 0.54),  # the strip's housing ("Color: Light Gray")
    "Red": (0.55, 0.03, 0.02),  # the strip's illuminated switch
    "Ink": (0.02, 0.02, 0.02),
    "Wood": (0.38, 0.27, 0.17),  # the desk top
}

# --- the lamp: published sizes (Anglepoise Type 75 Mini, sheet "US - V2") ------------------------
SHADE_DIAMETER = 5.1 * INCH  # "Shade diameter: 5.1in"
SHADE_HEIGHT = 5.9 * INCH  # "Shade height: 5.9in"
BASE_DIAMETER = 6.7 * INCH  # "Base diameter: 6.7in"
CABLE_LENGTH = 70.9 * INCH  # "Cable length: 70.9in"
MAX_REACH_H = 21.6 * INCH  # "Max Horizontal Reach 21.6 in", base to shade
MAX_REACH_V = 28.4 * INCH  # "Max Vertical Reach 28.4 in"

# --- the lamp: modelling choices (the sheet gives none of these) ---------------------------------
BASE_HEIGHT = 30 * MM  # CHOICE: a low cast iron cone, a weight and not a stand
BASE_RIM = 6 * MM  # CHOICE: a straight rim before the cone rises, taller than the 5 mm cable
BASE_TOP_RADIUS = 32 * MM  # CHOICE: where the cone's slope would meet the flat top: about 24
#                            degrees of slope, as a shallow cone sits
BASE_EDGE_RADIUS = 40 * MM  # CHOICE: a cast edge is rounded, so the cone meets the top in an arc
EDGE_STEPS = 6  # the arc's facets: about 4 degrees each
ARM_RADIUS = 8 * MM  # CHOICE: a 16 mm aluminium tube
ARM_LOWER = 240 * MM  # CHOICE: the arm segment lengths are not published; two equal segments
ARM_UPPER = 240 * MM  # whose sum (480 mm) plus the shade radius (65 mm) stays under the 548.6 mm
#                       horizontal reach and the pose under the 721.4 mm vertical reach
LOWER_LEAN_DEG = 20.0  # CHOICE: the lower arm leans 20 degrees from vertical toward +X
UPPER_RISE_DEG = 10.0  # CHOICE: the upper arm rises 10 degrees from horizontal toward +X
SHADE_TILT_DEG = 35.0  # CHOICE: the shade axis points 35 degrees below horizontal toward +X
SHADE_WALL = 1 * MM  # CHOICE: an aluminium spinning is about a millimetre thick
NECK_RADIUS = 7 * MM  # CHOICE, driven by `bearing`'s seat rule: the shade's seat must lie wholly
#                       on the arm's end, so the collar is narrower than the arm's 8 mm radius
NECK_LENGTH = 20 * MM  # CHOICE
CEILING = 35 * MM  # CHOICE: the flat inside the shade, measured from the arm end along the axis
CABLE_RADIUS = 2.5 * MM  # CHOICE: a 5 mm round lamp cord
PLUG_SIZE = (38 * MM, 22 * MM, 19 * MM)  # CHOICE: x along the blade line, y up, z along the blades
# Plug blades: NEMA 1-15P, Wikipedia "NEMA connector" (secondary; the primary ANSI/NEMA WD 6 is
# paywalled): 1/4 in wide, 0.06 in thick, 5/8 to 23/32 in long, 1/2 in apart; the polarized
# neutral blade is 5/16 in wide.
BLADE_THICK = 0.06 * INCH
BLADE_HOT = 0.25 * INCH
BLADE_NEUTRAL = 5 / 16 * INCH
BLADE_LENGTH = (15.9 + 18.3) / 2 * MM  # CHOICE: the middle of the published 15.9 to 18.3 mm
BLADE_PITCH = 0.5 * INCH

# --- the strip: published sizes (Tripp Lite PS615 datasheet) ------------------------------------
STRIP_L = 10.8 * INCH  # "Unit Dimensions (hwd / in.) 10.8 x 1.85 x 1.23"; read as length x width
STRIP_W = 1.85 * INCH  # x height: the datasheet's "hwd" label cannot be right for a strip that
STRIP_H = 1.23 * INCH  # lies on a desk, and the shape settles it
OUTLET_PITCH = 28.5 * MM  # "Outlets Measurement (Center to Center) 28.5mm"
OUTLETS = 6  # "(6) 5-15R", "Single row of 6"
GROUND_DIAMETER = 3 / 16 * INCH  # NEMA 5-15: 3/16 in round ground pin
GROUND_OFFSET = 15 / 32 * INCH  # centre to centre from the blades' line

# --- the strip: modelling choices -------------------------------------------------------------
BORE_CLEAR = 0.12 * MM  # CHOICE: each bore is this much wider than what goes into it, per side;
# CHECK-DRIVEN, not true to the object: a real receptacle slot is looser than this. The joints
# check's ring reach is the peg radius plus the contact length (0.145 mm at the strip's 0.5 m), and
# it reads any bore clearance over that length as "no bore round it". So the value stays under
# 0.145 mm so the wall rings the blade all round. (See the README.)
BORE_DEPTH = 20 * MM  # CHOICE: a blind bore 2.9 mm deeper than the blade is long
END_MARGIN = 22 * MM  # CHOICE: the last outlet's centre sits this far from the plain end
SWITCH_SIZE = (22 * MM, 12 * MM, 4 * MM)  # CHOICE: the sheet gives no switch size
SWITCH_X = -STRIP_L / 2 + 30 * MM  # CHOICE: near the cord end, as in the datasheet photo
LOGO_HALF = (16 * MM, 4.5 * MM)  # CHOICE: the sheet gives no logo size; an oval stands for it
LOGO_X = -65 * MM  # CHOICE: between the switch and the first outlet
SOCKET_X = [STRIP_L / 2 - END_MARGIN - (OUTLETS - 1 - i) * OUTLET_PITCH for i in range(OUTLETS)]
PLUGGED = 3  # the outlet the lamp's plug stands in: `socket_3`

# --- the layout: modelling choices -------------------------------------------------------------
# The plug stands on the strip's outlet face; the strip lies on the desk, outlets up. Turning it
# a quarter about X (its +z, the outlet face, to world +y) puts the back flat on the desk.
STRIP_QUAT = (-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
DESK_SIZE = (1.2, 0.03, 0.8)  # CHOICE: a plain slab; no source, it only gives the pieces a top
DESK_AT = (0.1, 0.0, -0.25)  # CHOICE: its top face is y 0; it spans x -0.5 to 0.7, z -0.65 to 0.15
PLUG_AT = (0.30, -0.60)  # CHOICE: where the plug stands on the desk, (x, z)
STRIP_POS = (PLUG_AT[0] - SOCKET_X[PLUGGED], 0.0, PLUG_AT[1])

# The farthest a piece on the desk is seen from; the three pieces share it (see the README table).
FARTHEST_WHY = (
    "Everything on the desk lies within its far edge: the chair puts the eyes about 0.5 m from the "
    "desk's near edge and the slab is 0.8 m deep, so 1.3 m. This is an estimate from that use, "
    "to be confirmed.")

VIEW_LAMP = View(
    0.5,
    "A desk lamp, seen from the chair. The person's eyes are about an arm's length, 0.5 m, from "
    "the shade they reach for and move, and nobody looks at it from nearer in use. This is an "
    "estimate from that use, not a measured study.",
    hidden=("bottom",),
    hidden_why="the base stands on the desk, so its underside is never seen",
    farthest_m=1.3,
    farthest_why=FARTHEST_WHY)
VIEW_STRIP = View(
    0.5,
    "A power strip lying on the desk. The nearest anyone looks at it is when "
    "they plug something in, leaning over the outlets with the eyes about 0.5 m away. This is "
    "an estimate from that use, not a measured study.",
    hidden=("back",),
    hidden_why="the back lies flat on the desk (the layout turns it down), so it is never seen",
    farthest_m=1.3,
    farthest_why=FARTHEST_WHY)

VIEW_DESK = View(
    0.5,
    "A desk top seen from the chair. The person sits at it, and the nearest part of it, the "
    "surface under their forearms, is about 0.5 m from their eyes. This is an estimate from that "
    "use, not a measured study.",
    hidden=("bottom",),
    hidden_why="the underside is hidden by the top itself from the chair",
    farthest_m=1.3,
    farthest_why=FARTHEST_WHY)

DONE_LAMP = ("An Anglepoise Type 75 Mini desk lamp in one reading pose: a cone-shaped base, two "
             "round arms, a bell shade, and a cable from the base to a 2-blade plug.")
DONE_DESK = "A plain desk top, a slab 1.2 m by 0.8 m and 30 mm thick, for the lamp and the strip."
DONE_STRIP = ("A Tripp Lite PS615 six-outlet power strip lying flat: a housing with six outlets "
              "cut in as bores, a switch and the printed logo on the outlet face.")


# --- geometry helpers (glTF vectors; B() turns a point into Blender's axes) -----------------------


def V(*p) -> Vector:
    return Vector(p)


def basis(axis: Vector) -> tuple[Vector, Vector]:
    a = axis.normalized()
    h = Vector((0, 1, 0)) if abs(a.y) < 0.95 else Vector((1, 0, 0))
    u = a.cross(h).normalized()
    return u, a.cross(u).normalized()


def vert(bm, p: Vector):
    return bm.verts.new(build.B(p.x, p.y, p.z))


def add_box(bm, x0, x1, y0, y1, z0, z1) -> None:
    c = [[vert(bm, V(x, y, z)) for z in (z0, z1)] for x in (x0, x1) for y in (y0, y1)]
    # c index: x (0,1) * 2 + y (0,1), then z (0,1)
    v = {(i, j, k): c[i * 2 + j][k] for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for quad in (
        ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)),
        ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)),
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)),
    ):
        bm.faces.new([v[q] for q in quad])


def add_prism(bm, cx, cy, radius, z0, z1, segs=12) -> None:
    """A round prism along Z (glTF) centred at (cx, cy)."""
    lo = [vert(bm, V(cx + radius * math.cos(2 * math.pi * k / segs),
                     cy + radius * math.sin(2 * math.pi * k / segs), z0)) for k in range(segs)]
    hi = [vert(bm, V(cx + radius * math.cos(2 * math.pi * k / segs),
                     cy + radius * math.sin(2 * math.pi * k / segs), z1)) for k in range(segs)]
    bm.faces.new(lo)
    bm.faces.new(hi)
    for k in range(segs):
        n = (k + 1) % segs
        bm.faces.new((lo[k], lo[n], hi[n], hi[k]))


def finished(bm):
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def lathe_axis(profile, origin: Vector, axis: Vector, segs: int):
    """[(radius, t)] spun about `axis` through `origin`, t along the axis. A profile that starts
    and ends on radius 0 closes on the axis; the rest closes the profile on itself."""
    u, v = basis(axis)
    a = axis.normalized()
    bm = bmesh.new()
    rings = [[vert(bm, origin + a * t + (u * math.cos(2 * math.pi * k / segs)
                                         + v * math.sin(2 * math.pi * k / segs)) * r)
              for k in range(segs)] for r, t in profile]
    for lo, hi in pairwise(rings):
        for k in range(segs):  # a ring of radius 0 gives zero-area quads; the weld in finished()
            bm.faces.new((lo[k], lo[(k + 1) % segs], hi[(k + 1) % segs], hi[k]))  # makes fans
    return finished(bm)


def mitred_tube(p0: Vector, p1: Vector, radius: float, plane0, plane1, segs: int):
    """A round tube from p0 to p1 whose two ends are cut flat in the planes `(point, normal)`.
    Two tubes cut in one plane meet face to face with no overlap."""
    a = (p1 - p0).normalized()
    u, v = basis(a)
    bm = bmesh.new()
    rings = []
    for p, (pp, n) in ((p0, plane0), (p1, plane1)):
        ring = []
        for k in range(segs):
            turn = 2 * math.pi * k / segs
            q = p + (u * math.cos(turn) + v * math.sin(turn)) * radius
            ring.append(vert(bm, q - a * (n.dot(q - pp) / n.dot(a))))
        rings.append(ring)
    lo, hi = rings
    for k in range(segs):
        bm.faces.new((lo[k], lo[(k + 1) % segs], hi[(k + 1) % segs], hi[k]))
    bm.faces.new(lo)
    bm.faces.new(hi)
    return finished(bm)


def cable_bm(points: list[Vector], radius: float, segs: int):
    """A closed round tube along `points`, its ring turned by parallel transport."""
    bm = bmesh.new()
    tans = [(points[min(i + 1, len(points) - 1)] - points[max(i - 1, 0)]).normalized()
            for i in range(len(points))]
    normal = Vector((0, 1, 0)) - tans[0] * Vector((0, 1, 0)).dot(tans[0])
    normal.normalize()
    rings = []
    for p, t in zip(points, tans):
        normal = (normal - t * normal.dot(t)).normalized()
        bino = t.cross(normal)
        rings.append([vert(bm, p + (normal * math.cos(2 * math.pi * k / segs)
                                    + bino * math.sin(2 * math.pi * k / segs)) * radius)
                      for k in range(segs)])
    for lo, hi in pairwise(rings):
        for k in range(segs):
            bm.faces.new((lo[k], lo[(k + 1) % segs], hi[(k + 1) % segs], hi[k]))
    bm.faces.new(rings[0])  # the ends are closed: the cord ends in the base and in the plug
    bm.faces.new(rings[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def cut(box_bm, cutters_bm):
    """`box_bm` minus `cutters_bm` (exact boolean), as a new bmesh. Both are freed."""
    objs = []
    for name, bm in (("cut_body", box_bm), ("cut_tool", cutters_bm)):
        me = bpy.data.meshes.new(name)
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(name, me)
        bpy.context.scene.collection.objects.link(ob)
        objs.append(ob)
    body, tool = objs
    mod = body.modifiers.new("cut", "BOOLEAN")
    mod.operation, mod.solver, mod.object = "DIFFERENCE", "EXACT", tool
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(body.evaluated_get(dg), depsgraph=dg)
    out = bmesh.new()
    out.from_mesh(me)
    bpy.data.meshes.remove(me)
    for ob in objs:
        data = ob.data
        bpy.data.objects.remove(ob)
        bpy.data.meshes.remove(data)
    bmesh.ops.recalc_face_normals(out, faces=out.faces)
    return out


# --- the lamp -----------------------------------------------------------------------------------


def pose() -> dict:
    """The lamp's pose, in the lamp's frame: joint points and axes."""
    p0 = V(0, BASE_HEIGHT, 0)
    lean = math.radians(LOWER_LEAN_DEG)
    a1 = V(math.sin(lean), math.cos(lean), 0)
    p1 = p0 + a1 * ARM_LOWER
    rise = math.radians(UPPER_RISE_DEG)
    a2 = V(math.cos(rise), math.sin(rise), 0)
    p2 = p1 + a2 * ARM_UPPER
    tilt = math.radians(SHADE_TILT_DEG)
    a3 = V(math.cos(tilt), -math.sin(tilt), 0)
    return {"p0": p0, "p1": p1, "p2": p2, "a1": a1, "a2": a2, "a3": a3}


def check_reach(p: dict) -> None:
    """The pose must lie inside the maker's published envelope, or the build stops."""
    rim = p["p2"] + p["a3"] * SHADE_HEIGHT
    shade_out = max(p["p2"].x, rim.x + SHADE_DIAMETER / 2 * math.sin(math.radians(SHADE_TILT_DEG)))
    top = max(p["p1"].y, p["p2"].y + NECK_RADIUS)
    if shade_out > MAX_REACH_H:
        raise ValueError(f"the pose reaches {shade_out:.3f} m out; the sheet allows "
                         f"{MAX_REACH_H:.3f}")
    if top > MAX_REACH_V:
        raise ValueError(f"the pose is {top:.3f} m high; the sheet allows {MAX_REACH_V:.3f}")


def base_profile():
    """(radius, y) round the base: the flat underside, a straight rim, the cone, an arc where the
    cone meets the flat top (a cast edge is rounded), and the top."""
    base_r = BASE_DIAMETER / 2
    run, rise = base_r - BASE_TOP_RADIUS, BASE_HEIGHT - BASE_RIM
    slope = math.atan2(rise, run)  # the cone's angle over the desk
    tangent = BASE_EDGE_RADIUS * math.tan(slope / 2)
    if tangent >= math.hypot(run, rise) or tangent >= BASE_TOP_RADIUS:
        raise ValueError("the base's edge radius does not fit its cone and its flat")
    centre = (BASE_TOP_RADIUS - tangent, BASE_HEIGHT - BASE_EDGE_RADIUS)
    arc = [(centre[0] + BASE_EDGE_RADIUS * math.sin(slope * (1 - i / EDGE_STEPS)),
            centre[1] + BASE_EDGE_RADIUS * math.cos(slope * (1 - i / EDGE_STEPS)))
           for i in range(EDGE_STEPS + 1)]
    return [(0.0, 0.0), (base_r, 0.0), (base_r, BASE_RIM), *arc, (0.0, BASE_HEIGHT)]


def shade_profile():
    """(radius, t) round the shade shell, from the arm end: the neck, a bell whose wall goes
    straight at the rim, the rim, the inside wall and a flat ceiling."""
    r_rim = SHADE_DIAMETER / 2

    def outer(s: float) -> tuple[float, float]:
        t = NECK_LENGTH + (SHADE_HEIGHT - NECK_LENGTH) * s
        return NECK_RADIUS + (r_rim - NECK_RADIUS) * (1 - (1 - s) ** 2), t

    steps = 16
    out = [(0.0, 0.0), (NECK_RADIUS, 0.0), (NECK_RADIUS, NECK_LENGTH)]
    out += [outer(i / steps) for i in range(1, steps + 1)]
    inner = []
    for i in range(steps, -1, -1):
        r, t = outer(i / steps)
        s = i / steps
        slope = (r_rim - NECK_RADIUS) * 2 * (1 - s) / (SHADE_HEIGHT - NECK_LENGTH)  # dr/dt
        inner.append((r - SHADE_WALL * math.sqrt(1 + slope * slope), t))
    inner = [(r, t) for r, t in inner if t >= CEILING]
    out += inner + [(0.0, CEILING)]
    return out


def route_cable(start: Vector, end: Vector, sway: float):
    """The cable's centre line: down the desk toward the plug with a gentle S, then up to the
    plug's side. Both ends run along -Z. Returns points about 8 mm apart."""
    def at(u: float) -> Vector:
        smooth = 3 * u * u - 2 * u ** 3
        t = min(max((u - 0.6) / 0.4, 0.0), 1.0)
        rise = 3 * t * t - 2 * t ** 3
        return V(start.x + (end.x - start.x) * smooth + sway * math.sin(math.pi * u) ** 2
                 * math.sin(2 * math.pi * u),
                 start.y + (end.y - start.y) * rise, start.z + (end.z - start.z) * u)

    fine = [at(i / 4000) for i in range(4001)]
    run = np.cumsum([0.0] + [(b - a).length for a, b in pairwise(fine)])
    count = math.ceil(run[-1] / (8 * MM))
    picks = np.interp(np.linspace(0, run[-1], count + 1), run, np.arange(len(fine)))
    return [fine[round(i)] for i in picks]


# --- the pieces ---------------------------------------------------------------------------------


def part_props(row: str, shape: str, on: str | None, **more) -> dict:
    return {"listed": row, "group": row, "shape": shape, "on": on, **more}


def build_lamp(mats, plug_seat: Vector):
    """The Lamp piece. `plug_seat` is where the plug's blades enter the strip's outlet face, in
    the lamp's frame (the desk layout puts the lamp at the origin, so it is the world point)."""
    root = build.piece("Lamp")
    p = pose()
    check_reach(p)

    base_r = BASE_DIAMETER / 2
    base_bm = lathe_axis(base_profile(), V(0, 0, 0), V(0, 1, 0), 72)
    build.mesh_obj("base", base_bm, mats["Iron"], root, props=part_props("l01", "lathe", None))

    up = V(0, 1, 0)
    bis = (p["a1"] + p["a2"]).normalized()
    lower = mitred_tube(p["p0"], p["p1"], ARM_RADIUS, (p["p0"], up), (p["p1"], bis), 18)
    build.mesh_obj("arm_lower", lower, mats["Paint"], root,
                   props=part_props("l02", "extrude", "base"))
    upper = mitred_tube(p["p1"], p["p2"], ARM_RADIUS, (p["p1"], bis), (p["p2"], p["a3"]), 18)
    build.mesh_obj("arm_upper", upper, mats["Paint"], root,
                   props=part_props("l03", "extrude", "arm_lower"))
    shade_bm = lathe_axis(shade_profile(), p["p2"], p["a3"], 72)
    build.mesh_obj("shade", shade_bm, mats["Paint"], root,
                   props=part_props("l04", "lathe", "arm_upper"))

    # the cable, from the cord exit on the base to the plug's side
    exit_point = V(0, CABLE_RADIUS, -base_r)
    plug_centre = V(plug_seat.x, plug_seat.y + PLUG_SIZE[1] / 2, plug_seat.z)
    entry = V(plug_centre.x, plug_centre.y, plug_centre.z + PLUG_SIZE[2] / 2)
    path = route_cable(exit_point, entry, sway=0.06)
    build.mesh_obj("cable", cable_bm(path, CABLE_RADIUS, 12), mats["Cord"], root,
                   props=part_props("l05", "cable", "base"))

    # the plug: a body standing on the outlet face, two blades hanging from it into the bores
    px, py, pz = PLUG_SIZE
    body = bmesh.new()
    add_box(body, plug_seat.x - px / 2, plug_seat.x + px / 2, plug_seat.y, plug_seat.y + py,
            plug_seat.z - pz / 2, plug_seat.z + pz / 2)
    build.mesh_obj("plug", finished(body), mats["Cord"], root, props=part_props("l06", "box", None))
    for name, dx, width in (("blade_hot", -BLADE_PITCH / 2, BLADE_HOT),
                            ("blade_neutral", BLADE_PITCH / 2, BLADE_NEUTRAL)):
        blade = bmesh.new()
        add_box(blade, plug_seat.x + dx - BLADE_THICK / 2, plug_seat.x + dx + BLADE_THICK / 2,
                plug_seat.y - BLADE_LENGTH, plug_seat.y, plug_seat.z - width / 2,
                plug_seat.z + width / 2)
        build.mesh_obj(name, finished(blade), mats["Brass"], root,
                       props=part_props("l07", "box", "plug"))

    return root, exit_point


def build_strip(mats, lift: DecalLift):
    """The Strip piece, outlet face toward +Z with its back on z 0: how the maker's photographs
    show it, and the face a printed label can be cast onto."""
    root = build.piece("Strip")
    hw, hl = STRIP_W / 2, STRIP_L / 2
    body = bmesh.new()
    add_box(body, -hl, hl, -hw, hw, 0.0, STRIP_H)
    tool = bmesh.new()
    floor_z, top_z = STRIP_H - BORE_DEPTH, STRIP_H + 1 * MM
    for x in SOCKET_X:
        for dx, width in ((-BLADE_PITCH / 2, BLADE_HOT), (BLADE_PITCH / 2, BLADE_NEUTRAL)):
            add_box(tool, x + dx - BLADE_THICK / 2 - BORE_CLEAR,
                    x + dx + BLADE_THICK / 2 + BORE_CLEAR,
                    -width / 2 - BORE_CLEAR, width / 2 + BORE_CLEAR, floor_z, top_z)
        add_prism(tool, x, GROUND_OFFSET, GROUND_DIAMETER / 2 + BORE_CLEAR, floor_z, top_z)
    body_ob = build.mesh_obj("body", finished(cut(body, finished(tool))), mats["Plastic"], root,
                             props=part_props("s01", "box", None))

    sx, sy, sz = SWITCH_SIZE
    sw = bmesh.new()
    add_box(sw, SWITCH_X - sx / 2, SWITCH_X + sx / 2, -sy / 2, sy / 2, STRIP_H, STRIP_H + sz)
    build.mesh_obj("switch", finished(sw), mats["Red"], root,
                   props=part_props("s02", "box", "body"))

    logo = Part(name="logo", listed="s03", shape="decal", on="body", group="s03", role="decal",
                bones=(), bind=None,
                form=SoftForm(DecalForm(LOGO_HALF, 32, 3, (LOGO_X, 0.0, STRIP_H)), None, None, (),
                              "an oval the size of the logo's box stands for the wordmark"))
    soft.soft_part(logo, mats["Ink"], root, host=body_ob, lift=lift)

    return root, logo.form


def build_desk(mats):
    """The Desk piece: a plain slab, its top face at y 0 and its centre at x 0, z 0. It is not a
    product; it gives the lamp and the strip a top to rest on."""
    root = build.piece("Desk")
    w, t, d = DESK_SIZE
    slab = bmesh.new()
    add_box(slab, -w / 2, w / 2, -t, 0.0, -d / 2, d / 2)
    build.mesh_obj("top", finished(slab), mats["Wood"], root, props=part_props("d01", "box", None))
    return root


# --- rows ---------------------------------------------------------------------------------------

LAMP_SHEET = f"Anglepoise Type 75 Mini technical sheet (US - V2), {LAMP_URL}"
STRIP_SHEET = f"Tripp Lite PS615 datasheet, {STRIP_URL}"


def lamp_rows() -> list[Listed]:
    return [
        Listed("l01", "Cone-shaped cast iron base, 170.2 mm across", f"{LAMP_SHEET}: base diameter "
               "6.7 in, cast iron base. The height, rim and top flat are modelling choices. "
               "min_thickness is the height.", BASE_HEIGHT, None),
        Listed("l02", "Lower arm, a round aluminium tube standing on the base", f"{LAMP_SHEET}: "
               "aluminum arms. The 16 mm diameter, the 240 mm length and the lean are modelling "
               "choices. The lamp is spring-balanced and the arm is held by its pivot, not by its "
               "weight on the base, so the row is bonded. min_thickness is the diameter.",
               2 * ARM_RADIUS, None, held="bonded"),
        Listed("l03", "Upper arm, a round aluminium tube meeting the lower arm", f"{LAMP_SHEET}: "
               "aluminum arms. The sizes and the rise are modelling choices. A spring and "
               "friction joint holds it, not its weight, so the row is bonded. min_thickness is "
               "the diameter.", 2 * ARM_RADIUS, None, held="bonded"),
        Listed("l04", "Bell shade with a neck, on the end of the upper arm", f"{LAMP_SHEET}: "
               "shade diameter 5.1 in, shade height 5.9 in, aluminum shade. The profile, the "
               "neck and the 1 mm wall are modelling choices. The arm's joint holds it, so the "
               "row is bonded. min_thickness is the wall.", SHADE_WALL, None, held="bonded"),
        Listed("l05", "Cable, from the base to the plug", f"{LAMP_SHEET}: cable length 70.9 in. "
               "The 5 mm diameter and the route are modelling choices; only part of the 70.9 in "
               "is drawn, the rest is slack off the desk. It is moulded into the base and the "
               "plug, so the row is bonded. min_thickness is the diameter.", 2 * CABLE_RADIUS,
               None, held="bonded"),
        Listed("l06", "Plug body, standing on the strip's outlet face", f"{LAMP_SHEET}: plug "
               "2-blade US plug to NEMA 1-15. The body size is a modelling choice (no source "
               "publishes a moulded plug body). min_thickness is its smallest side.",
               min(PLUG_SIZE), None),
        Listed("l07", "Two plug blades, hanging from the plug body", f"{NEMA_URL} (secondary; "
               "the primary ANSI/NEMA WD 6 is paywalled): NEMA 1-15P blades 1/4 in wide (5/16 in "
               "for the neutral), 0.06 in thick, 15.9 to 18.3 mm long, 1/2 in apart. The length "
               "is the middle of that range. They are moulded into the plug body, so the row is "
               "bonded. min_thickness is the blade thickness.", BLADE_THICK, None, held="bonded"),
    ]


def strip_rows() -> list[Listed]:
    return [
        Listed("s01", "Housing with six NEMA 5-15R outlets cut in as bores", f"{STRIP_SHEET}: "
               "10.8 x 1.85 x 1.23 in read as length x width x height, six outlets 28.5 mm apart "
               "in a single row, plastic, light gray. The bores are modelling choices: the "
               "slots are the NEMA 5-15 blade and ground-pin sizes from "
               f"{NEMA_URL} (secondary) with 0.3 mm clearance a side, 20 mm deep. "
               "min_thickness is the height.", STRIP_H, None),
        Listed("s02", "Illuminated switch, on the outlet face near the cord end", f"{STRIP_SHEET}: "
               "illuminated on/off switch. Its size and place are modelling choices (not "
               "published). min_thickness is its smallest side.", min(SWITCH_SIZE), None),
        Listed("s03", "Printed TRIPP-LITE logo, on the outlet face", f"{STRIP_SHEET}, product "
               "photograph: the logo is printed on the top face at the cord end. Its size is a "
               "modelling choice; an oval stands for the lettering. min_thickness is a 4 micron "
               "pad-print ink film (Inkcups ICN-B150 manual).", 4e-6, None, finish="print"),
    ]


def desk_rows() -> list[Listed]:
    return [
        Listed("d01", "Desk top, a plain slab, with the lamp and the strip resting on it",
               "No source: a modelling choice. A slab 1.2 m by 0.8 m and 30 mm thick, large enough "
               "for the lamp's pose, the cable and the strip, so the two pieces have a top to "
               "rest on. min_thickness is the slab's thickness.", DESK_SIZE[1], None),
    ]


# --- main ---------------------------------------------------------------------------------------


def main() -> None:
    out, scene, style_path = (Path(a) for a in runner.script_args())
    style = load_style(style_path)
    if style.units_per_metre != 1.0:
        raise ValueError("this build is written in metres: the style must say units_per_metre 1")

    build.reset()
    mats = build.slots(ROLES)
    build.selftest_axes()

    rows = {"Lamp": lamp_rows(), "Strip": strip_rows(), "Desk": desk_rows()}
    # The printed logo is lifted off the housing by the strip's contact length (what a viewer
    # resolves at its nearest distance), so it stays within touching distance of its host. The
    # lift is checked against one depth step at the strip's farthest view, or the build raises.
    lift = decal_lift("piece 'Strip'", VIEW_STRIP, style, style.units_per_metre)

    builder = build.PackBuilder(PACK, style.units_per_metre, out, SPACING,
                                crease_deg=style.crease_deg)
    plug_seat = V(PLUG_AT[0], STRIP_H, PLUG_AT[1])
    lamp, exit_point = build_lamp(mats, plug_seat)
    strip, logo_form = build_strip(mats, lift)
    desk = build_desk(mats)

    builder.register(lamp, origin="centre of the base underside, on the desk (y 0)",
                     recipe=Recipe("assembly", f"bars/05-lamp/lamp_build.py, from {LAMP_URL}, "
                                   f"with the plug blades from {NEMA_URL}"),
                     listed=rows["Lamp"], done=DONE_LAMP, size_class="prop", kind="lamp",
                     snap="free", hosts=["base", "plug"], view=VIEW_LAMP)
    builder.register(strip, origin="centre of the back face (z 0), outlet face toward +z",
                     recipe=Recipe("assembly", f"bars/05-lamp/lamp_build.py, from {STRIP_URL}, "
                                   f"with the outlet bores from {NEMA_URL}"),
                     listed=rows["Strip"], done=DONE_STRIP, size_class="prop",
                     kind="power strip", snap="free", hosts=["body"], view=VIEW_STRIP,
                     forms={"logo": logo_form})
    builder.register(desk, origin="centre of the top face, on the desk top (y 0)",
                     recipe=Recipe("assembly", "bars/05-lamp/lamp_build.py: a plain slab, no "
                                   "product"),
                     listed=rows["Desk"], done=DONE_DESK, size_class="piece", kind="desk",
                     snap="free", hosts=["top"], view=VIEW_DESK)
    builder.mount(lamp, "cord_exit", kind="power", point=tuple(exit_point), axis=(0, 0, -1),
                  takes="cable", part="base")
    for i, x in enumerate(SOCKET_X):
        builder.mount(strip, f"socket_{i}", kind="power", point=(x, 0.0, STRIP_H), axis=(0, 0, 1),
                      takes="plug", part="body")
    socket = Ref.parse(f"{PACK}:strip.socket_{PLUGGED}")
    builder.connector(
        lamp, "mains", part="cable",
        ends=[End(tuple(exit_point), None, Ref.parse("self.cord_exit", self_ok=True)),
              End(tuple(plug_seat), LAYOUT, socket, plug="plug")],
        cap=CABLE_LENGTH,
        why="the lamp's own cable is 70.9 in (1.80 m) long (maker's sheet); it runs from the "
            "cord exit on the base to the plug standing in an outlet of the strip")
    pack = builder.finish(["Lamp", "Strip", "Desk"])
    export_pieces(pack, scene, pieces=["Lamp", "Strip", "Desk"])
    layout = Layout(
        name=LAYOUT, frame="gltf",
        placements={
            "desk": LayoutPlacement("desk", PACK, "Desk", DESK_AT, (0.0, 0.0, 0.0, 1.0)),
            "lamp": LayoutPlacement("lamp", PACK, "Lamp", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                                    rests_on="desk"),
            "strip": LayoutPlacement("strip", PACK, "Strip", STRIP_POS, STRIP_QUAT,
                                     rests_on="desk"),
        },
        stands={})
    (scene / "layouts").mkdir(parents=True, exist_ok=True)
    write_layout(layout, scene / "layouts" / f"{LAYOUT}.json")
    runner.done()


main()
