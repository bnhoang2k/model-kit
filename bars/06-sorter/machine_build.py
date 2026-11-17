"""Build one machine in Blender, in one style. Run it through build.py, which starts Blender:

    blender --background --python machine_build.py -- <out> <scene> <style.json> <real|toy> <machine folder>

resolve.py places every part of the machine folder's parts.py; this file only turns each placed
part into a closed mesh, registers the one piece named by parts.META["piece"] and writes the pack
`<name>_<style>`, its .glb and the gate's scene folder (with a one-placement layout, so export
hygiene can measure it).

The solids are built by a few small helpers (rrect, lathe, tube, fill_cap, steps):
a soft solid is a rounded-rectangle footprint lofted up, its top rim rounded through a sweep. In
the real style the rim sweeps 40 degrees and then turns 50 onto the top, so the runtime ink draws
exactly one crease there; in the toy style it sweeps the full 90 and draws none (only the
silhouette, as on the chibi). Every curve takes the fewest facets that stay under what the eye
resolves at the piece's nearest view (resolve.steps), so the gate's detail check finds it neither
blocky nor finer than the eye.

A tilted part needs nothing here: resolve.py hands over its turned frame (u, v, n) and the loft
follows it. An arc is swept round its axis (`arc`).
"""

import math
import sys
from pathlib import Path

import os  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["MACHINE_DIR"] = str(Path(sys.argv[sys.argv.index("--") + 5]).resolve())  # before resolve reads it

import bmesh  # noqa: E402
from mathutils import Vector, geometry  # noqa: E402

from model_kit.blender import build, runner  # noqa: E402
from model_kit.blender.export import export_pieces  # noqa: E402
from model_kit.core.manifest import Layout, LayoutPlacement, Listed, Recipe, View  # noqa: E402
from model_kit.core.units import load_style  # noqa: E402
from model_kit.core.writer import write_layout  # noqa: E402

import resolve as R  # noqa: E402  (it loads MACHINE_DIR/parts.py)

P = R.P

PIECE = P.META["piece"]
LAYOUT = "floor"

NEAR_WHY = ("A viewer frames one step at a time from the front or side of a machine; "
            "the nearest a camera gets to any machine on this floor is taken as 4 m, the same "
            "modelling choice for every machine drawn in this style, to be measured in use.")
FAR_WHY = "The wide shot frames the whole floor; its far corner is taken as 30 m (the same choice for every machine in this style)."
VIEW = View(R.NEAREST_M, NEAR_WHY, hidden=("bottom",),
            hidden_why="it stands on the factory floor, so its underside is never seen",
            farthest_m=30.0, farthest_why=FAR_WHY)


STEP = 0.10  # metres: no flat face or wall runs longer than this between corners (see fill_cap)


def rrect(w, d, r, inset, n):
    """rrect: a rounded rectangle w x d centred on 0, inset by `inset`, CCW. The corner
    steps follow the uninset radius, so every ring of one solid has the same count. Each straight
    side is cut into pieces no longer than STEP (the count from the uninset side, so every ring
    matches): a side tessellated from its corners alone leaves slivers metres long, and a join check
    then misreads a 0.018 mm seat as over a millimetre deep."""
    cuts_x = max(1, math.ceil((w - 2 * min(r, w / 2)) / STEP))
    cuts_y = max(1, math.ceil((d - 2 * min(r, d / 2)) / STEP))
    w, d = w - 2 * inset, d - 2 * inset
    rr = max(0.0, min(r - inset, w / 2, d / 2))
    flat_x, flat_y = r >= (w + 2 * inset) / 2 - 1e-12, r >= (d + 2 * inset) / 2 - 1e-12
    pts = []
    for i, (cx, cy, a0) in enumerate(((w / 2 - rr, d / 2 - rr, 0), (-w / 2 + rr, d / 2 - rr, 90),
                                      (-w / 2 + rr, -d / 2 + rr, 180), (w / 2 - rr, -d / 2 + rr, 270))):
        for k in range(n + 1):
            if k == 0 and ((i in (1, 3) and flat_x) or (i == 2 and flat_y)):
                continue
            if i == 3 and k == n and flat_y:
                continue
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            if k == n:  # the straight side to the next corner, cut to STEP
                nx, ny = ((-w / 2 + rr, d / 2), (-w / 2, -d / 2 + rr), (w / 2 - rr, -d / 2), (w / 2, d / 2 - rr))[i]
                if i == 3:
                    nx, ny = w / 2, d / 2 - rr
                cuts = cuts_x if i in (0, 2) else cuts_y
                if (i in (0, 2) and not flat_x) or (i in (1, 3) and not flat_y):
                    px, py = pts[-1]
                    for j in range(1, cuts):
                        pts.append((px + (nx - px) * j / cuts, py + (ny - py) * j / cuts))
    return pts


def rings_of(p):
    """[(inset, y)] from the bottom up: a bottom rim when the part has one, the walls, the top
    rim, and for a tray the lip and its inner floor."""
    sw = math.radians(p["sweep"])
    H, e, eb = p["H"], p["e"], p["eb"]
    rings = []
    if eb:
        m = R.steps(eb, p["sweep"])
        for k in range(m + 1):
            t = sw * (m - k) / m
            rings.append((eb * (1 - math.cos(t)), eb * math.sin(sw) - eb * math.sin(t)))
    else:
        rings.append((0.0, 0.0))
    top0 = H - e * math.sin(sw) if e else H
    y0 = rings[-1][1]
    walls = max(1, math.ceil((top0 - y0) / STEP))
    rings += [(0.0, y0 + (top0 - y0) * j / walls) for j in range(1, walls)]
    if e:
        base = H - e * math.sin(sw)
        m = R.steps(e, p["sweep"])
        for k in range(m + 1):
            t = sw * k / m
            ring = (e * (1 - math.cos(t)), base + e * math.sin(t))
            if abs(ring[1] - rings[-1][1]) > 1e-9 or abs(ring[0] - rings[-1][0]) > 1e-9:
                rings.append(ring)
    else:
        rings.append((0.0, H))
    if p["kind"] == "tray":
        inner = max(1, math.ceil((H - p["t"]) / STEP))
        rings += [(p["t"], H - (H - p["t"]) * j / inner) for j in range(inner + 1)]
    return rings


def to_world(p, a, b, h):
    o, u, v, n = p["o"], p["u"], p["v"], p["n"]
    return Vector(build.B(*(o[i] + a * u[i] + b * v[i] + h * n[i] for i in range(3))))


def loft(p):
    n = R.steps(p["r"], 90)
    bm = bmesh.new()
    loops = []
    for inset, y in rings_of(p):
        loops.append([bm.verts.new(to_world(p, a, b, y)) for a, b in rrect(p["w"], p["d"], p["r"], inset, n)])
    for lo, hi in zip(loops, loops[1:]):
        k = len(lo)
        for i in range(k):
            bm.faces.new((lo[i], lo[(i + 1) % k], hi[(i + 1) % k], hi[i]))
    rs = rings_of(p)
    for (inset, y), loop in ((rs[0], loops[0]), (rs[-1], loops[-1])):
        fill_cap(bm, loop, rrect(p["w"], p["d"], p["r"], inset, n), lambda a, b, y=y: to_world(p, a, b, y))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def fill_cap(bm, loop, pts2d, to3d):
    """Close a ring with a flat cap: a constrained Delaunay fill of the ring and a grid of points
    STEP apart inside it (half a step from the rim, or a single middle row on a narrow cap), as
    a flat fill usually does.
    The ring's own corners map to its existing vertices; only the grid points are new."""
    xs, ys = [q[0] for q in pts2d], [q[1] for q in pts2d]
    grid = []
    W, D = max(xs) - min(xs), max(ys) - min(ys)
    if max(W, D) > 1.5 * STEP:  # a long, narrow cap gets one row down its middle
        poly = [Vector((x, y)) for x, y in pts2d]
        clear = min(STEP / 2, 0.35 * min(W, D))
        gx, gy = max(1, math.floor(W / STEP)), max(1, math.floor(D / STEP))
        x0 = (min(xs) + max(xs)) / 2 - (gx - 1) * STEP / 2
        y0 = (min(ys) + max(ys)) / 2 - (gy - 1) * STEP / 2
        for i in range(gx):
            for j in range(gy):
                q = Vector((x0 + i * STEP, y0 + j * STEP))
                if _inside(q, poly):
                    if min(_seg_dist(q, poly[k], poly[(k + 1) % len(poly)]) for k in range(len(poly))) >= clear:
                        grid.append(q)
    if not grid:
        bm.faces.new(loop)
        return
    nring = len(pts2d)
    coords = [Vector((x, y)) for x, y in pts2d] + grid
    edges = [(k, (k + 1) % nring) for k in range(nring)]
    out = geometry.delaunay_2d_cdt(coords, edges, [list(range(nring))], 1, 1e-9)
    vs, faces, orig = out[0], out[2], out[3]
    made = []  # ring vertices are the loop's own; grid ones are placed in the cap's plane
    for i, v in enumerate(vs):
        src = [o for o in orig[i] if o < nring]
        if src:
            made.append(loop[src[0]])
        else:
            made.append(bm.verts.new(to3d(v[0], v[1])))
    for f in faces:
        bm.faces.new([made[k] for k in f])


def _inside(q, poly):
    c = False
    for k in range(len(poly)):
        a, b = poly[k], poly[(k + 1) % len(poly)]
        if (a.y > q.y) != (b.y > q.y) and q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x:
            c = not c
    return c


def _seg_dist(q, a, b):
    ab = b - a
    t = max(0.0, min(1.0, (q - a).dot(ab) / ab.dot(ab))) if ab.length_squared else 0.0
    return (a + ab * t - q).length


def lathe(p):
    """lathe: a closed solid of revolution about the part's n axis, the profile
    [(radius, h)] from the seat round to the far end; a radius of 0 closes to a point."""
    prof = p["profile"]
    n = 4 * R.steps(max(r for r, _ in prof), 90)
    bm = bmesh.new()
    loops = []
    for r, h in prof:  # h runs from the part's origin, already sunk SINK into its host
        if r <= 0:
            loops.append([bm.verts.new(to_world(p, 0.0, 0.0, h))])
        else:
            loops.append([bm.verts.new(to_world(p, r * math.cos(2 * math.pi * k / n),
                                                r * math.sin(2 * math.pi * k / n), h)) for k in range(n)])
    for lo, hi in zip(loops, loops[1:]):
        if len(lo) == 1:
            for k in range(n):
                bm.faces.new((lo[0], hi[k], hi[(k + 1) % n]))
        elif len(hi) == 1:
            for k in range(n):
                bm.faces.new((lo[k], lo[(k + 1) % n], hi[0]))
        else:
            for k in range(n):
                bm.faces.new((lo[k], lo[(k + 1) % n], hi[(k + 1) % n], hi[k]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def tube(p):
    """tube: a round tube along the points, rings turned with the path (parallel
    transport), flat caps at both ends (each sunk into the face it meets)."""
    segs = max(8, 4 * R.steps(p["R"], 90))
    pts = [Vector(x) for x in p["pts"]]
    bm = bmesh.new()
    t0 = (pts[1] - pts[0]).normalized()
    side = t0.cross(Vector((0, 1, 0)) if abs(t0.y) < 0.9 else Vector((1, 0, 0))).normalized()
    loops = []
    for i, c in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)]).normalized()
        side = (side - t * side.dot(t)).normalized()
        up = t.cross(side)
        loops.append([bm.verts.new(build.B(*(c + p["R"] * (math.cos(a) * side + math.sin(a) * up))))
                      for a in (2 * math.pi * k / segs for k in range(segs))])
    for lo, hi in zip(loops, loops[1:]):
        for k in range(segs):
            bm.faces.new((lo[k], lo[(k + 1) % segs], hi[(k + 1) % segs], hi[k]))
    bm.faces.new(list(reversed(loops[0])))
    bm.faces.new(loops[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def arc(p):
    """An annular sector swept round the part's vertical axis: the profile [(r, y)] (resolve.py) cut
    to STEP, one ring per facet of the shared grid (resolve._share_grids), each end ring on its
    trimmed end plane, both ends capped flat."""
    prof = []
    loop = p["prof"] + [p["prof"][0]]
    for (r0, y0), (r1, y1) in zip(loop, loop[1:]):
        cuts = max(1, math.ceil(math.hypot(r1 - r0, y1 - y0) / STEP))
        prof += [(r0 + (r1 - r0) * j / cuts, y0 + (y1 - y0) * j / cuts) for j in range(cuts)]
    a0, a1 = (math.radians(x) for x in p["arc"])
    m = p["steps"]

    def angle(r, k):
        if k in (0, m):
            return R.arc_angles(p, r)[0 if k == 0 else 1]
        return a0 + (a1 - a0) * k / m

    def at(r, y, k):
        t = angle(r, k)
        return to_world(p, r * math.cos(t), r * math.sin(t), y)

    bm = bmesh.new()
    rings = [[bm.verts.new(at(r, y, k)) for r, y in prof] for k in range(m + 1)]
    n = len(prof)
    for lo, hi in zip(rings, rings[1:]):
        for i in range(n):
            bm.faces.new((lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]))
    for k in (0, m):
        fill_cap(bm, rings[k], prof, lambda r, y, k=k: at(r, y, k))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def thinnest(p):
    if p["kind"] == "tube":
        return 2 * p["R"]
    if p["kind"] == "lathe":
        return min(2 * r for r, _ in p["profile"] if r > 0)
    if p["kind"] == "arc":
        return min(p["R1"] - p["R0"], p["h"])
    return min(p["w"], p["d"], p["h"])


def main() -> None:
    args = runner.script_args()
    out, scene, style_path = (Path(a) for a in args[:3])
    look = args[3]
    if look not in ("real", "toy"):
        raise ValueError(f"style must be real or toy, got {look!r}")
    style = load_style(style_path)
    if style.units_per_metre != 1.0:
        raise ValueError("this build is written in metres: the style must say units_per_metre 1")
    placed = R.resolve(look)
    bad = R.check(placed)
    if bad:
        raise ValueError("resolve.check found problems:\n" + "\n".join(bad))

    build.reset()
    mats = build.slots(P.ROLES)
    build.selftest_axes()
    pack = f"{P.META['name']}_{look}"
    builder = build.PackBuilder(pack, style.units_per_metre, out, None, crease_deg=style.crease_deg)
    root = build.piece(PIECE)
    used = {}
    # build in row order, so take-apart (last built comes off first) removes parcels and fittings first
    order = list(P.ROWS)
    for p in sorted(placed, key=lambda q: order.index(q["row"])):
        bm = (tube(p) if p["kind"] == "tube" else lathe(p) if p["kind"] in ("lathe",)
              else arc(p) if p["kind"] == "arc" else loft(p))
        shape = "lathe" if p["kind"] in ("round", "lathe", "arc") else "extrude"
        build.mesh_obj(p["id"], bm, mats[p["role"]], root,
                       props={"listed": p["row"], "group": p["row"], "shape": shape, "on": p["host"]})
        used.setdefault(p["row"], []).append(thinnest(p))
    rows = []
    for rid in order:
        if rid not in used:
            continue
        text, source, held = P.ROWS[rid]
        rows.append(Listed(rid, text, source + " min_thickness is the row's thinnest part side.",
                           min(used[rid]), None, held=held))
    done = P.META["done"][look] + P.META["done"].get("common", "")
    builder.register(root, origin="centre of the footprint, on the floor (y 0)",
                     recipe=Recipe("assembly", f"bars/{Path(os.environ['MACHINE_DIR']).name}: "
                                   "parts.py resolved by resolve.py, built by machine_build.py (the machine kit); no product"),
                     listed=rows, done=done, size_class=P.META["size_class"], kind=P.META["kind"], snap="free",
                     hosts=[P.META.get("root", "base")], view=VIEW)
    result = builder.finish([PIECE])
    export_pieces(result, scene, pieces=[PIECE])
    layout = Layout(name=LAYOUT, frame="gltf", placements={
        P.META["name"]: LayoutPlacement(P.META["name"], pack, PIECE, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))},
        stands={})
    (scene / "layouts").mkdir(parents=True, exist_ok=True)
    write_layout(layout, scene / "layouts" / f"{LAYOUT}.json")
    runner.done()


main()
