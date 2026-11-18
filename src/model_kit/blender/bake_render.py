"""The light bake's Blender half. `model_kit.bake` runs this script.

It reads one JSON argument (`BakePlan.spec()`):

    layout, mode, placements, live, cameras, lights, layers, texels_per_m, samples, seed, out

and works in five steps, each a function below:

    place   import each placement's piece from its pack .glb, name its mesh objects
            `<placement>/<part>` (other objects, a rig's armature, `<placement>/<name>`), give
            each instanced part its own mesh, and stand it where the layout says. Live
            placements are hidden at once: a piece that moves must not occlude a still piece's
            face.
    see     ray-cast every stage camera over a pixel grid (256 rays on the long side) and mark
            the polygons a ray hits first; then one ray at the centroid of each polygon the grid
            missed, from each camera that has the centroid in view.
    unwrap  give every still object with a seen polygon a lightmap UV, one smooth region at a
            time (faces joined where they share a position and a normal, so a hard edge bounds
            a region). A disk-like region is unwrapped whole; a tube (an annulus) gets one
            lengthwise cut first, through faces no camera sees where it can, and unrolls into one
            strip as wide as its circumference; any other region, or an unwrap that folds,
            crosses itself or stretches more than Smart UV Project may, takes Smart UV Project of
            its seen polygons. Each region's islands are scaled so its object gets its density,
            each unseen polygon's UV collapsed to one point (so it takes no area, and no room:
            the packer keeps clear of UVs it is not packing), then one Pack Islands over all
            objects at twice the shipped size with a 4 pixel margin (2 shipped texels). The atlas
            grows until the islands fit; the density never shrinks to fit.
    bake    per layer, switch on only that layer's lights (white, strength 1; a hemisphere light
            is the world, 1 in its half and 0 in the other; the world is black otherwise), bake
            DIFFUSE twice at the doubled size with two seeds and half the samples each, shrink the
            mean by 2 counting only covered texels (`shrink_covered`), fill 2 rings of margin
            round every island at the shipped size (`dilate`), and write it as
            `<out>/<layer>.exr`, half-float, ZIP.
            `indirect` mode drops the direct pass. Colour is off: it is light, not paint. The
            spec may carry `light_values` ({light id: {color, strength}}), a hook for tests that
            bake a layer at other values; a stage never produces it, and `bake.json` records the
            values each light was baked at, with its kind (`baked_lights`).
    export  write `<out>/<layout>.glb` and `<out>/bake.json`.

The lightmap goes after any UV set a part already has, so a part with one keeps it as set 0 and
its lightmap is set 1; a part with none has its lightmap as set 0. `bake.json` records which.
Polygons, not triangles, are marked: a quad is seen when a ray hits it, and its triangles are all
seen.
"""

import heapq
import json
import math
from pathlib import Path

import bmesh
import bpy
import numpy as np
from io_scene_gltf2.io.com.constants import ROUNDING_DIGIT as GLTF_NORMAL_DIGITS
from mathutils import Quaternion, Vector

from model_kit.blender import build, runner
from model_kit.blender.stage_scene import camera_frame, make_light, to_blender
from model_kit.core.bake_grid import phase_error, shrink_covered
from model_kit.core.checks.seam_jump import NORMAL_DIGITS, WELD, normal_noise, same_normal
from model_kit.core.errors import ManifestError

if GLTF_NORMAL_DIGITS != NORMAL_DIGITS:  # the check's rule rests on the exporter's rounding
    raise RuntimeError(f"the glTF exporter rounds normals to {GLTF_NORMAL_DIGITS} decimals, but "
                       f"seam_jump.NORMAL_DIGITS is {NORMAL_DIGITS}; set it to match")

LIGHTMAP = "lightmap"
STAGE = "@@"  # on a name while placements are imported
RAYS_LONG_SIDE = 256
# Pack Islands' margin at the doubled size. With `margin_method="ADD"` Blender pads EACH island by
# the margin, so two islands sit at least 2 x 4 = 8 px apart (a bar-05 bake measured no gap under
# 8 doubled px, none under 3 shipped texels). 4 px alone would not be enough: texel-centre coverage
# and the 2 x 2 block grid can each take a texel from the gap. A bilinear read at an island's edge
# touches only the first texel outside it; with 3 or more shipped texels between two islands, each
# island's first ring is its own, so a read never takes the other island's light.
MARGIN_PX = 4
SMART_ANGLE = math.radians(66.0)
# The most Smart UV Project lets one object's faces differ in UV area per world area: a face at
# its angle limit to the projection keeps cos(angle) of its area. A region's own unwrap is kept
# only when it is no less even (`unfold_fault`).
SMART_STRETCH = 1.0 / math.cos(SMART_ANGLE)
DISK, TUBE, FALLBACK, UNSEEN = "disk", "tube", "fallback", "unseen"  # region shapes
FILL = 0.5  # the share of the first atlas guess the islands are expected to fill
MAX_ATLAS = 16384
FIT_TOL = 1e-6
DENSITY_TOL = 1e-3  # packing may not change an island's area by more than this share
COLLAPSE_AT = (0.0, 0.0)
# Blender's own edge fill, at the doubled size. The shipped picture never reads it
# (`shrink_covered` takes only the texels of the seen mask); it only keeps a texel the mask counts
# and Blender's rasteriser does not (a centre on an edge) from reading black.
BAKE_MARGIN_PX = 1
# Rings of margin filled at the shipped size, diagonals included. A bilinear read takes one texel
# either side of its sample, so a sample anywhere on an island reads at most the first ring; the
# second fills the rest of the 2-texel gap between islands, so no read meets black.
DILATE_PX = 2
THREADS = 4  # fixed, so two runs give the same pictures
INSIDE_EPS = 1e-9  # a texel centre on a triangle's edge counts as inside


def clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def meshes_of(root) -> list:
    return [ob for ob in root.children_recursive if ob.type == "MESH"]


# --- place ----------------------------------------------------------------------------------


def place(spec: dict) -> dict[str, list]:
    """Import and stand every placement. Returns each placement's mesh objects, by id.

    A mesh object under the piece is named `<piece>/<part>` in the .glb and becomes
    `<placement>/<part>`; any other object under it (a rig's armature, an empty) becomes
    `<placement>/<its name>`. A mesh two objects share (an instanced part) is copied, so each
    object has its own data and its own lightmap. While placing, names carry a `@@` stage prefix,
    so a placement named like its piece cannot clash with the next import; the prefix comes off
    at the end."""
    objects: dict[str, list] = {}
    live = set(spec["live"])
    for p in spec["placements"]:
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=p["glb"])
        new = [ob for ob in bpy.data.objects if ob not in before]
        root = next((ob for ob in new if ob.name == p["piece"] and ob.parent is None), None)
        if root is None:
            raise ManifestError(f"{p['glb']}: no top-level node {p['piece']!r} for placement "
                                f"{p['id']!r}")
        keep = {root, *root.children_recursive}
        for ob in new:
            if ob not in keep:
                data = ob.data if ob.type == "MESH" else None
                bpy.data.objects.remove(ob, do_unlink=True)
                if data is not None and data.users == 0:
                    bpy.data.meshes.remove(data)
        prefix = p["piece"] + "/"
        for ob in root.children_recursive:
            part = ob.name[len(prefix):] if ob.name.startswith(prefix) else None
            if ob.type == "MESH":
                if part is None:
                    raise ManifestError(f"{p['glb']}: mesh object {ob.name!r} under "
                                        f"{p['piece']!r} is not named {prefix}<part>")
                if ob.data.users > 1:
                    ob.data = ob.data.copy()
            ob.name = f"{STAGE}{p['id']}/{part if part is not None else ob.name}"
            if ob.type == "MESH":
                ob.data.name = ob.name
        root.name = STAGE + p["id"]
        x, y, z = p["position"]
        qx, qy, qz, qw = p["quaternion"]
        root.location = to_blender((x, y, z))
        root.rotation_mode = "QUATERNION"
        root.rotation_quaternion = Quaternion((qw, qx, -qz, qy)).normalized()
        objects[p["id"]] = meshes_of(root)
        if p["id"] in live:
            for ob in [root, *root.children_recursive]:
                ob.hide_render = True
                ob.hide_set(True)
    for ob in bpy.data.objects:
        if ob.name.startswith(STAGE):
            ob.name = ob.name[len(STAGE):]
            if ob.type == "MESH":
                ob.data.name = ob.name
    bpy.context.view_layer.update()
    return objects


# --- see ------------------------------------------------------------------------------------


def camera_rays(cam: dict):
    """The origin and the unit directions (N, 3) of a camera's pixel grid, in Blender axes.
    `aspect` is width over height."""
    origin, forward, right, up, half = camera_frame(cam)
    aspect = cam["aspect"]
    nx = RAYS_LONG_SIDE if aspect >= 1 else max(1, round(RAYS_LONG_SIDE * aspect))
    ny = RAYS_LONG_SIDE if aspect < 1 else max(1, round(RAYS_LONG_SIDE / aspect))
    xs = ((np.arange(nx) + 0.5) / nx * 2 - 1) * half * aspect
    ys = (1 - (np.arange(ny) + 0.5) / ny * 2) * half
    gx, gy = np.meshgrid(xs, ys)
    f, r, u = (np.array(v) for v in (forward, right, up))
    dirs = f + gx.reshape(-1, 1) * r + gy.reshape(-1, 1) * u
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return origin, dirs


def in_frustum(cam: dict, point: Vector) -> bool:
    origin, forward, right, up, half = camera_frame(cam)
    v = point - origin
    depth = v.dot(forward)
    if depth <= 0:
        return False
    return (abs(v.dot(right)) <= depth * half * cam["aspect"]
            and abs(v.dot(up)) <= depth * half)


def see(spec: dict, still: dict[str, list]) -> dict[str, np.ndarray]:
    """For each still object, a bool array over its polygons: true where some stage camera sees
    that polygon first. Two passes. The grid: each camera casts 256 rays on its long side. The
    centroids: a polygon the grid missed (one smaller than the ray spacing) gets one ray from
    each camera at its centroid, when the centroid is inside that camera's view. Live objects
    are hidden, so they never block a ray."""
    objs = {ob.name: ob for obs in still.values() for ob in obs}
    seen = {name: np.zeros(len(ob.data.polygons), dtype=bool) for name, ob in objs.items()}
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for cam in spec["cameras"]:
        origin, dirs = camera_rays(cam)
        for d in dirs:
            hit, _loc, _nrm, index, ob, _mat = scene.ray_cast(depsgraph, origin, Vector(d))
            if hit and ob.name in seen:
                seen[ob.name][index] = True
    grid = sum(int(flags.sum()) for flags in seen.values())
    for name, ob in objs.items():
        for index in np.nonzero(~seen[name])[0]:
            centre = ob.matrix_world @ ob.data.polygons[int(index)].center
            for cam in spec["cameras"]:
                if not in_frustum(cam, centre):
                    continue
                origin = to_blender(cam["position"])
                hit, _loc, _nrm, hit_index, hit_ob, _mat = scene.ray_cast(
                    depsgraph, origin, (centre - origin).normalized())
                if hit and hit_ob.name == name and hit_index == index:
                    seen[name][index] = True
                    break
    total = sum(len(flags) for flags in seen.values())
    both = sum(int(flags.sum()) for flags in seen.values())
    print(f"SEE grid-marked {grid}, centroid-marked {both - grid}, unseen {total - both} "
          f"of {total} polygons", flush=True)
    return seen


# --- unwrap ---------------------------------------------------------------------------------


def select_polygons(ob, flags: np.ndarray) -> None:
    """In edit mode, select exactly the polygons flagged."""
    bm = bmesh.from_edit_mesh(ob.data)
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        f.select_set(False)
    for f in bm.faces:
        if flags[f.index]:
            f.select_set(True)
    bmesh.update_edit_mesh(ob.data)


def enter_edit(objs: list) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.mode_set(mode="EDIT")


def read_uv(ob) -> np.ndarray:
    layer = ob.data.uv_layers[LIGHTMAP]
    out = np.empty(len(ob.data.loops) * 2, dtype=np.float32)
    layer.uv.foreach_get("vector", out)
    return out.reshape(-1, 2).astype(np.float64)


def write_uv(ob, uv: np.ndarray) -> None:
    ob.data.uv_layers[LIGHTMAP].uv.foreach_set("vector", uv.astype(np.float32).reshape(-1))
    ob.data.update()


def poly_loops(me) -> np.ndarray:
    """For each loop, the index of its polygon."""
    starts = np.empty(len(me.polygons), dtype=np.int32)
    totals = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    return np.repeat(np.arange(len(me.polygons)), totals)


def uv_area(uv: np.ndarray, loop_poly: np.ndarray, keep: np.ndarray) -> float:
    """The summed area of the polygons flagged in `keep`, by the shoelace formula."""
    idx = np.nonzero(keep[loop_poly])[0]
    if len(idx) == 0:
        return 0.0
    polys = loop_poly[idx]
    cut = np.nonzero(np.diff(polys, prepend=-1))[0]
    total = 0.0
    for a, b in zip(cut, [*cut[1:], len(idx)], strict=True):
        pts = uv[idx[a:b]]
        x, y = pts[:, 0], pts[:, 1]
        total += abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2
    return total


def world_areas(ob) -> np.ndarray:
    """Each polygon's area, in project units squared (uniform scale)."""
    areas = np.empty(len(ob.data.polygons), dtype=np.float64)
    ob.data.polygons.foreach_get("area", areas)
    scale = ob.matrix_world.to_scale()
    if max(scale) - min(scale) > 1e-6 * max(scale):
        raise ManifestError(f"{ob.name}: world scale {tuple(scale)} is not uniform, so its "
                            "texel density has no one area scale")
    return areas * scale[0] ** 2


def collapse_unseen(ob, seen: np.ndarray, loop_poly: np.ndarray) -> None:
    """Put every loop of an unseen polygon at one point, so the polygon has no area in the
    atlas. It must come before the packing: Blender's packer keeps clear of UVs it is not
    packing, so an unseen polygon left as it was (a square, or the first set's copy) takes room."""
    uv = read_uv(ob)
    uv[~seen[loop_poly]] = COLLAPSE_AT
    write_uv(ob, uv)


def pack_at(objs: list, size2: int) -> None:
    """Pack every polygon of `objs` (their UVs written as pixels / `size2`) in one go. Pack
    Islands moves only selected polygons and does not keep clear of unselected UVs, so every
    polygon is selected here, whatever an earlier step left selected."""
    enter_edit(objs)
    for ob in objs:
        select_polygons(ob, np.ones(len(ob.data.polygons), dtype=bool))
    bpy.context.tool_settings.use_uv_select_sync = True
    bpy.ops.uv.pack_islands(rotate=True, scale=False, margin_method="ADD",
                            margin=MARGIN_PX / size2)
    bpy.ops.object.mode_set(mode="OBJECT")


# --- smooth regions -------------------------------------------------------------------------


def next_loops(me) -> np.ndarray:
    """For each loop, the next loop round its polygon."""
    starts = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_start", starts)
    nxt = np.arange(len(me.loops)) + 1
    nxt[np.append(starts[1:], len(me.loops)) - 1] = starts
    return nxt


def polygon_noise(me, co: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """(n_polygons,) how far float32 corners can tilt each polygon's normal
    (`seam_jump.normal_noise`), from its corners `co[verts]` as Blender holds them (float32, the
    mesh's own frame, as the .glb stores them)."""
    starts = np.empty(len(me.polygons), dtype=np.int64)
    totals = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    out = np.zeros(len(me.polygons))
    for k in np.unique(totals):  # polygons of k corners at a time
        which = np.nonzero(totals == k)[0]
        corners = co[verts[starts[which][:, None] + np.arange(k)]]  # (n_k, k, 3)
        out[which] = normal_noise(corners)
    return out


def smooth_corners(ob) -> np.ndarray:
    """For each loop, the id of its smooth corner: the seam-jump check's own rule for a smooth
    edge, applied to the mesh the export will write. Corners at one place (world positions in
    float32, rounded to `WELD` of the extent) are one place. An edge two polygons share is smooth
    when, at both its ends, the two polygons' corner normals (as the glTF exporter writes them:
    float32, rounded to its `ROUNDING_DIGIT` decimals, normalised) are one surface within the
    file's own precision (`seam_jump.same_normal`, the check's rule: the exporter's rounding plus
    how far each polygon's float32 corners can tilt its normal, `seam_jump.normal_noise`); then the
    two corners at each end are one smooth corner. Joined corner by corner through smooth edges,
    faces form smooth regions. A bend the .glb's normals show, beyond that precision, parts them;
    a split that only the first UV set made does not, nor a near-flat edge whose normals differ
    only by rounding or float32 noise."""
    me = ob.data
    n = len(me.loops)
    verts = np.empty(n, dtype=np.int64)
    me.loops.foreach_get("vertex_index", verts)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    noise = polygon_noise(me, co.reshape(-1, 3), verts)[poly_loops(me)]  # per loop
    m = np.array(ob.matrix_world, dtype=np.float64)
    world = co.reshape(-1, 3).astype(np.float64) @ m[:3, :3].T + m[:3, 3]
    # the normals as the glTF exporter writes them: float32, rounded, then normalised
    nrm = np.empty(n * 3, dtype=np.float32)
    me.corner_normals.foreach_get("vector", nrm)
    nrm = np.round(nrm.reshape(-1, 3), GLTF_NORMAL_DIGITS)
    length = np.linalg.norm(nrm, axis=1, keepdims=True)
    np.divide(nrm, length, out=nrm, where=length != 0)
    nrm = nrm.astype(np.float64)
    extent = float(np.ptp(world, axis=0).max()) if len(world) else 0.0
    keys = np.round(world[verts] / (WELD * extent if extent > 0.0 else 1.0)).astype(np.int64)
    _, place = np.unique(keys, axis=0, return_inverse=True)
    place = place.reshape(-1)
    nxt = next_loops(me)
    lo, hi = np.minimum(place, place[nxt]), np.maximum(place, place[nxt])
    order = np.lexsort((hi, lo))
    same = (lo[order][1:] == lo[order][:-1]) & (hi[order][1:] == hi[order][:-1])
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    starts = np.nonzero(np.append(True, ~same))[0]
    sizes = np.diff(np.append(starts, n))
    for s in starts[sizes == 2]:  # an edge exactly two polygons share
        i, j = int(order[s]), int(order[s + 1])
        # (corner at one end, corner at the other) on each side, in the same end order
        ends_i = (i, int(nxt[i]))
        ends_j = (j, int(nxt[j])) if place[j] == place[i] else (int(nxt[j]), j)
        if all(same_normal(nrm[a], nrm[b], noise[a], noise[b])
               for a, b in zip(ends_i, ends_j, strict=True)):
            for a, b in zip(ends_i, ends_j, strict=True):
                parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    return np.unique(roots, return_inverse=True)[1].reshape(-1)


def smooth_proxy(ob):
    """A stand-in object for `ob` whose mesh joins faces only at smooth corners: the same
    polygons, in the same order, with the same loops, each loop on its smooth corner's vertex.
    It carries an empty `lightmap` set. A polygon two of whose corners weld into one keeps
    corners of its own, so no polygon repeats a vertex."""
    me = ob.data
    corner = smooth_corners(ob)
    verts = np.empty(len(me.loops), dtype=np.int64)
    me.loops.foreach_get("vertex_index", verts)
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    starts = np.empty(len(me.polygons), dtype=np.int64)
    totals = np.empty(len(me.polygons), dtype=np.int64)
    me.polygons.foreach_get("loop_start", starts)
    me.polygons.foreach_get("loop_total", totals)
    made = int(corner.max()) + 1 if len(corner) else 0
    for start, total in zip(starts, totals, strict=True):
        ids = corner[start:start + total]
        if len(np.unique(ids)) < total:
            corner[start:start + total] = np.arange(made, made + total)
            made += total
    points = np.zeros((made, 3))
    points[corner] = co[verts]
    faces = [corner[a:a + t].tolist() for a, t in zip(starts, totals, strict=True)]
    pm = bpy.data.meshes.new(f"{ob.name}.smooth")
    pm.from_pydata(points.tolist(), [], faces)
    check = np.empty(len(pm.loops), dtype=np.int64)
    pm.loops.foreach_get("vertex_index", check)
    if len(check) != len(corner) or not np.array_equal(check, corner):
        raise RuntimeError(f"{ob.name}: the smooth stand-in's loops do not follow the mesh's")
    pm.uv_layers.new(name=LIGHTMAP, do_init=False)
    proxy = bpy.data.objects.new(pm.name, pm)
    bpy.context.scene.collection.objects.link(proxy)
    proxy.matrix_world = ob.matrix_world.copy()
    return proxy


def border_loops(border: list) -> list[set[int]]:
    """The vertex sets of the loops the border edges form (connected through shared vertices)."""
    parent: dict[int, int] = {}

    def find(v: int) -> int:
        while parent.setdefault(v, v) != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for e in border:
        a, b = (find(v.index) for v in e.verts)
        parent[a] = b
    loops: dict[int, set[int]] = {}
    for v in list(parent):
        loops.setdefault(find(v), set()).add(v)
    return list(loops.values())


def cut_tube(edges: set, border: set, ends: list[set[int]], seen: np.ndarray) -> list[int] | None:
    """The edges of one cut along a tube, from border loop `ends[0]` to `ends[1]` through the
    tube's inner edges: of the paths that border the fewest seen faces, the shortest. A cut
    through faces no camera sees makes a seam no one sees. None when no path joins the loops."""
    start, goal = ends
    by_vert: dict[int, list] = {}
    for e in edges:
        if e not in border:
            for v in e.verts:
                by_vert.setdefault(v.index, []).append(e)
    dist = {v: (0, 0.0) for v in start}
    back: dict[int, tuple[int, int]] = {}
    heap = [(0, 0.0, v) for v in start]
    heapq.heapify(heap)
    end = None
    while heap:
        n_seen, length, v = heapq.heappop(heap)
        if (n_seen, length) > dist[v]:
            continue
        if v in goal:
            end = v
            break
        for e in by_vert.get(v, ()):
            w = e.verts[0].index if e.verts[1].index == v else e.verts[1].index
            step = (n_seen + sum(bool(seen[f.index]) for f in e.link_faces),
                    length + e.calc_length())
            if w not in dist or step < dist[w]:
                dist[w] = step
                back[w] = (v, e.index)
                heapq.heappush(heap, (*step, w))
    if end is None:
        return None
    path = []
    while end in back:
        end, edge = back[end]
        path.append(edge)
    return path


def smooth_regions(bm, seen: np.ndarray) -> tuple[np.ndarray, list[str], list[list[int]]]:
    """Split a smooth stand-in's faces into regions (faces joined through shared edges) and name
    each region's shape from its topology: a disk (one border loop, Euler characteristic 1), a
    tube (two border loops, Euler characteristic 0: an annulus), `UNSEEN` (no seen face), or a
    fallback whose reason says why (a closed surface, more holes or handles). A region with an
    edge of three faces or more, or a vertex where two sheets touch, falls back. Returns each
    face's region, each region's shape, and each tube's cut (edge indices)."""
    bm.faces.ensure_lookup_table()
    region = np.full(len(bm.faces), -1, dtype=np.int64)
    shapes: list[str] = []
    cuts: list[list[int]] = []
    for first in bm.faces:
        if region[first.index] >= 0:
            continue
        r = len(shapes)
        region[first.index] = r
        faces, stack = [], [first]
        while stack:
            f = stack.pop()
            faces.append(f)
            for e in f.edges:
                for g in e.link_faces:
                    if region[g.index] < 0:
                        region[g.index] = r
                        stack.append(g)
        cuts.append([])
        if not seen[[f.index for f in faces]].any():
            shapes.append(UNSEEN)
            continue
        verts = {v for f in faces for v in f.verts}
        edges = {e for f in faces for e in f.edges}
        border = {e for e in edges if len(e.link_faces) == 1}
        on_border: dict[int, int] = {}
        for e in border:
            for v in e.verts:
                on_border[v.index] = on_border.get(v.index, 0) + 1
        if any(len(e.link_faces) > 2 for e in edges):
            shapes.append(FALLBACK + ": an edge of three faces or more")
            continue
        if any(n != 2 for n in on_border.values()) or not all(v.is_manifold for v in verts):
            shapes.append(FALLBACK + ": two sheets touch at a vertex")
            continue
        loops = border_loops(list(border))
        euler = len(verts) - len(edges) + len(faces)
        if euler == 1 and len(loops) == 1:
            shapes.append(DISK)
        elif euler == 0 and len(loops) == 2:
            cut = cut_tube(edges, border, loops, seen)
            shapes.append(TUBE if cut else FALLBACK + ": no path joins the tube's ends")
            cuts[-1] = cut or []
        else:
            shapes.append(f"{FALLBACK}: {len(loops)} border loops, Euler characteristic {euler}")
    return region, shapes, cuts


def segments_cross(p: np.ndarray, q: np.ndarray, block: int = 512) -> bool:
    """Whether two of the segments `p[i]`-`q[i]` (n, 2) cross properly: each one's ends lie
    strictly on the two sides of the other. Segments that share an end never do."""

    def orient(a, b, c):
        return ((b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1])
                - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0]))

    lo, hi = np.minimum(p, q), np.maximum(p, q)
    for i in range(0, len(p), block):
        a, b = p[i:i + block, None], q[i:i + block, None]
        near = ((lo[i:i + block, None] <= hi[None]) & (lo[None] <= hi[i:i + block, None])).all(-1)
        if not near.any():
            continue
        c, d = p[None], q[None]
        hit = ((orient(a, b, c) * orient(a, b, d) < 0) & (orient(c, d, a) * orient(c, d, b) < 0)
               & near)
        if hit.any():
            return True
    return False


def members(groups: np.ndarray) -> list[np.ndarray]:
    """For each group id in `groups` (0 to its largest), the ascending indices that hold it. One
    sort serves every group, so a mesh of many small regions costs no more than one of few."""
    return np.split(np.argsort(groups, kind="stable"), np.cumsum(np.bincount(groups))[:-1])


def signed_uv_areas(uv: np.ndarray, nxt: np.ndarray, loop_poly: np.ndarray) -> np.ndarray:
    """Each polygon's signed area in `uv`, by the shoelace formula."""
    cross = uv[:, 0] * uv[nxt, 1] - uv[:, 1] * uv[nxt, 0]
    return np.bincount(loop_poly, weights=cross, minlength=int(loop_poly.max()) + 1) / 2


def unfold_fault(uv: np.ndarray, verts: np.ndarray, nxt: np.ndarray, signed: np.ndarray,
                 faces: np.ndarray, loops: np.ndarray, seen: np.ndarray, areas: np.ndarray,
                 cut: set[tuple[int, int]]) -> str | None:
    """Why the unwrap of one region (its polygons `faces` and their `loops`, as indices) is not
    fit for a lightmap, or None. `verts`, `nxt` and `signed` describe the whole mesh and are read
    once per mesh by the caller, so this costs only the region's own size. Three tests, none with
    a chosen limit:

    - a fold: a polygon whose UV winds against the region's (it lies over its neighbours), or a
      seen polygon with no UV area (its light has nowhere to go);
    - a crossing: two edges of the region's UV outline cross. With no fold and an outline that
      does not cross itself, the map is one to one, so no two points share a texel;
    - stretch: among its seen polygons, the largest UV area per world area over the smallest
      exceeds `SMART_STRETCH`, the most Smart UV Project allows. The region's own unwrap is kept
      only when it is no less even than the fallback.
    """
    mine = signed[faces]
    wind = np.sign(mine.sum())
    lit = faces[seen[faces] & (areas[faces] > 0)]
    if wind == 0 or (mine * wind < 0).any() or (signed[lit] == 0).any():
        return "a fold"
    ratio = np.abs(signed[lit]) / areas[lit]
    if ratio.max() > SMART_STRETCH * ratio.min():
        return "stretch"
    a, b = verts[loops], verts[nxt[loops]]
    keys = np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    outline = counts[inverse.reshape(-1)] == 1
    if cut:
        outline |= np.array([(int(x), int(y)) in cut for x, y in keys])
    if segments_cross(uv[loops[outline]], uv[nxt[loops[outline]]]):
        return "a crossing"
    return None


def unwrap_regions(targets: list) -> dict[str, tuple[bpy.types.Object, np.ndarray]]:
    """Unwrap each target's smooth stand-in (`smooth_proxy`) region by region. Returns, per
    object name, its stand-in (lightmap UV written) and its polygons' regions. The stand-ins stay
    for the packing: Pack Islands finds islands through shared vertices, and only the stand-in
    shares a vertex across a split the first UV set or the import made inside a smooth region.
    `smooth_regions` names its regions' shapes. Disks are unwrapped whole, angle based. A tube
    first gets its one lengthwise cut marked as a seam, so it unrolls into one strip as wide as
    its circumference, as a label peels off a can. A region of another shape, and an unwrap
    `unfold_fault` refuses, falls back to Smart UV Project of its seen polygons, as every object
    had before. Regions no camera sees are left for `collapse_unseen`."""
    work = []  # (object, seen, proxy, region, shapes, cut edges as vertex pairs)
    for ob, flags, _ in targets:
        proxy = smooth_proxy(ob)
        bm = bmesh.new()
        bm.from_mesh(proxy.data)
        region, shapes, cuts = smooth_regions(bm, flags)
        bm.edges.ensure_lookup_table()
        seam = np.zeros(len(proxy.data.edges), dtype=bool)
        pairs = set()
        for cut in cuts:
            for ei in cut:
                seam[ei] = True
                a, b = (v.index for v in bm.edges[ei].verts)
                pairs.add((min(a, b), max(a, b)))
        bm.free()
        proxy.data.edges.foreach_set("use_seam", seam)
        work.append((ob, flags, proxy, region, shapes, pairs))
    proxies = [w[2] for w in work]

    def unwrap_selected(select: list[np.ndarray], op) -> None:
        if not any(s.any() for s in select):
            return
        enter_edit(proxies)
        for proxy, flags in zip(proxies, select, strict=True):
            select_polygons(proxy, flags)
        bpy.context.tool_settings.use_uv_select_sync = True
        op()
        bpy.ops.object.mode_set(mode="OBJECT")

    def having(region: np.ndarray, shapes: list[str], keep) -> np.ndarray:
        return np.isin(region, [r for r, s in enumerate(shapes) if keep(s)])

    unwrap_selected([having(w[3], w[4], lambda s: s in (DISK, TUBE)) for w in work],
                    lambda: bpy.ops.uv.unwrap(method="ANGLE_BASED", fill_holes=True,
                                              correct_aspect=True, margin=0.0))
    for _ob, flags, proxy, region, shapes, pairs in work:
        uv = read_uv(proxy)
        areas = world_areas(proxy)
        me = proxy.data
        loop_poly = poly_loops(me)
        verts = np.empty(len(me.loops), dtype=np.int64)
        me.loops.foreach_get("vertex_index", verts)
        nxt = next_loops(me)
        signed = signed_uv_areas(uv, nxt, loop_poly)
        faces_of, loops_of = members(region), members(region[loop_poly])
        for r, shape in enumerate(shapes):
            if shape in (DISK, TUBE):
                fault = unfold_fault(uv, verts, nxt, signed, faces_of[r], loops_of[r], flags,
                                     areas, pairs)
                if fault:
                    shapes[r] = f"{FALLBACK}: the {shape} unwrap has {fault}"
    unwrap_selected([w[1] & having(w[3], w[4], lambda s: s.startswith(FALLBACK)) for w in work],
                    lambda: bpy.ops.uv.smart_project(
                        angle_limit=SMART_ANGLE, island_margin=0.0, area_weight=0.0,
                        correct_aspect=True, scale_to_bounds=False))
    out = {}
    for ob, _flags, proxy, region, shapes, _ in work:
        out[ob.name] = (proxy, region)
        counts: dict[str, int] = {}
        for shape in shapes:
            counts[shape] = counts.get(shape, 0) + 1
        print(f"REGIONS {ob.name}: " + "; ".join(f"{s} {n}" for s, n in sorted(counts.items())),
              flush=True)
    return out


def unwrap(spec: dict, still: dict[str, list], seen: dict[str, np.ndarray]) -> tuple[int, dict]:
    """Give each still object with a seen polygon a lightmap UV. Returns the shipped atlas side
    in pixels and the `uv_channel` record."""
    density = {p["id"]: p["px_per_unit"] for p in spec["placements"]}
    targets = []  # (object, seen flags, pixels per unit)
    for pid, obs in still.items():
        for ob in obs:
            if seen[ob.name].any():
                targets.append((ob, seen[ob.name], density[pid]))
    if not targets:
        raise ManifestError("no stage camera sees any still piece, so there is nothing to bake")
    channel = {}
    for ob, _, _ in targets:
        channel[ob.name] = len(ob.data.uv_layers)  # the lightmap goes after any set it has
        # Blender fills the new set (from the first set, else a square a face). Unseen
        # polygons keep that until the last step collapses them.
        layer = ob.data.uv_layers.new(name=LIGHTMAP, do_init=True)
        ob.data.uv_layers.active = layer
    work = unwrap_regions(targets)
    stand = {name: proxy for name, (proxy, _) in work.items()}
    regions = {name: region for name, (_, region) in work.items()}
    proxies = list(stand.values())

    # Scale each smooth region's islands to its object's density, in pixels at the doubled size.
    # The stand-ins carry the UV until the packing is done.
    pixels, loop_polys = {}, {}
    for ob, flags, ppu in targets:
        loop_poly = poly_loops(ob.data)
        uv = read_uv(stand[ob.name])
        region = regions[ob.name][loop_poly]
        areas = world_areas(ob)
        seen_in = regions[ob.name][flags]  # each seen polygon's region
        count = int(regions[ob.name].max()) + 1
        raw = np.bincount(seen_in, np.abs(signed_uv_areas(uv, next_loops(ob.data), loop_poly))[flags],
                          minlength=count)
        world = np.bincount(seen_in, areas[flags], minlength=count)
        lit = np.unique(seen_in)
        if (raw[lit] <= 0).any():
            raise ManifestError(f"{ob.name}: the unwrap left its seen polygons no area")
        factor = np.zeros(count)
        factor[lit] = ppu * np.sqrt(world[lit] / raw[lit])
        pixels[ob.name] = uv * factor[region][:, None]
        loop_polys[ob.name] = loop_poly
    wanted = sum(uv_area(pixels[ob.name], loop_polys[ob.name], flags) for ob, flags, _ in targets)
    for ob, flags, _ in targets:
        collapse_unseen(stand[ob.name], flags, loop_polys[ob.name])

    size2 = max(4, 4 * math.ceil(math.sqrt(wanted / FILL) / 4))
    while True:
        if size2 > MAX_ATLAS:
            raise ManifestError(f"the islands need an atlas over {MAX_ATLAS} px at the doubled "
                                "size; lower texels_per_m or the stage's reach")
        for ob, flags, _ in targets:
            uv = read_uv(stand[ob.name])
            sel = flags[loop_polys[ob.name]]
            uv[sel] = pixels[ob.name][sel] / size2
            write_uv(stand[ob.name], uv)
        pack_at(proxies, size2)
        fits = True
        packed_area = 0.0
        for ob, flags, _ in targets:
            uv = read_uv(stand[ob.name])
            sel = flags[loop_polys[ob.name]]
            packed_area += uv_area(uv, loop_polys[ob.name], flags) * size2 ** 2
            if sel.any() and (uv[sel].min() < -FIT_TOL or uv[sel].max() > 1 + FIT_TOL):
                fits = False
        if abs(packed_area - wanted) > DENSITY_TOL * wanted:
            raise ManifestError(f"packing changed the islands' area from {wanted:.1f} to "
                                f"{packed_area:.1f} px squared; the density rule would break")
        if fits:
            break
        size2 += 4 * max(1, size2 // 40)
    for ob, _, _ in targets:  # the stand-in's loops are the object's loops, in order
        write_uv(ob, read_uv(stand[ob.name]))
        pm = stand[ob.name].data
        bpy.data.objects.remove(stand[ob.name], do_unlink=True)
        bpy.data.meshes.remove(pm)
    size = size2 // 2
    print(f"UNWRAP atlas {size} px square, packed at {size2} px; {len(targets)} objects, "
          f"{wanted:.0f} px squared of islands", flush=True)
    return size, channel


# --- bake -----------------------------------------------------------------------------------


def set_world(halves: set) -> None:
    """The world for one layer. `halves` holds "sky" and/or "ground" for the layer's hemisphere
    lights: white strength 1 in the named half, 0 elsewhere. No hemisphere light: black."""
    world = bpy.data.worlds.new("bake")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    bg = nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0 if halves else 0.0
    if len(halves) == 1:
        coord = nodes.new("ShaderNodeTexCoord")
        split = nodes.new("ShaderNodeSeparateXYZ")
        step = nodes.new("ShaderNodeMath")
        step.operation = "GREATER_THAN" if "sky" in halves else "LESS_THAN"
        step.inputs[1].default_value = 0.0
        links.new(coord.outputs["Generated"], split.inputs["Vector"])
        links.new(split.outputs["Z"], step.inputs[0])
        links.new(step.outputs["Value"], bg.inputs["Strength"])


def texel_masks(still: dict[str, list], seen: dict[str, np.ndarray], size2: int) -> dict:
    """For each placement, a (size2, size2) bool array (row 0 at the bottom, as Blender's
    pixels): the texels whose centre lies in a seen lightmap triangle of that placement. The
    collapsed unseen triangles and the empty atlas are outside it."""
    masks = {}
    for pid, obs in still.items():
        mask = np.zeros((size2, size2), dtype=bool)
        for ob in obs:
            if LIGHTMAP not in ob.data.uv_layers or not seen[ob.name].any():
                continue
            me = ob.data
            me.calc_loop_triangles()
            uv = read_uv(ob) * size2
            for tri in me.loop_triangles:
                if not seen[ob.name][tri.polygon_index]:
                    continue
                a, b, c = (uv[i] for i in tri.loops)
                x0, x1 = int(np.floor(min(a[0], b[0], c[0]))), int(np.ceil(max(a[0], b[0], c[0])))
                y0, y1 = int(np.floor(min(a[1], b[1], c[1]))), int(np.ceil(max(a[1], b[1], c[1])))
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, size2 - 1), min(y1, size2 - 1)
                det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
                if x1 < x0 or y1 < y0 or abs(det) < 1e-12:
                    continue
                gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
                w0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / det
                w1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / det
                inside = (w0 >= -INSIDE_EPS) & (w1 >= -INSIDE_EPS) & (1 - w0 - w1 >= -INSIDE_EPS)
                mask[y0:y1 + 1, x0:x1 + 1] |= inside
        masks[pid] = mask
    return masks


def prepare_materials(objs: list, image) -> tuple[list, list]:
    """Give every material of the bake targets an active image node holding `image`, the bake's
    target. An object with no material gets a temporary one. Returns the nodes added and the
    (object, material) pairs added, for `restore_materials`."""
    nodes, temps, done = [], [], set()
    for ob in objs:
        if not any(slot.material for slot in ob.material_slots):
            mat = bpy.data.materials.new(f"{ob.name}.bake")
            ob.data.materials.append(mat)
            temps.append((ob, mat))
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None or mat.name in done:
                continue
            done.add(mat.name)
            node = mat.node_tree.nodes.new("ShaderNodeTexImage")
            node.image = image
            mat.node_tree.nodes.active = node
            nodes.append((mat, node))
    return nodes, temps


def restore_materials(nodes: list, temps: list) -> None:
    for mat, node in nodes:
        mat.node_tree.nodes.remove(node)
    for ob, mat in temps:
        ob.data.materials.clear()
        bpy.data.materials.remove(mat)


def select_only(objs: list) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for ob in objs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def bake_once(image, seed: int, samples: int) -> np.ndarray:
    """One DIFFUSE bake of the selected objects into `image`; returns its RGB, (n, n, 3)."""
    cycles = bpy.context.scene.cycles
    cycles.seed = seed
    cycles.samples = samples
    bpy.ops.object.bake(type="DIFFUSE")
    n = image.size[0]
    buf = np.empty(n * n * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    return buf.reshape(n, n, 4)[:, :, :3].copy()


def dilate(pic: np.ndarray, filled: np.ndarray, rings: int) -> np.ndarray:
    """`pic` (n, n, 3) with `rings` rings of margin filled round its `filled` texels: each pass,
    every empty texel with a filled one among its 8 neighbours takes the mean of those."""
    pic, filled = pic.copy(), filled.copy()
    n = pic.shape[0]
    for _ in range(rings):
        value = np.pad(pic * filled[:, :, None], ((1, 1), (1, 1), (0, 0)))
        count = np.pad(filled.astype(np.float64), 1)
        total = np.zeros_like(pic)
        many = np.zeros((n, n))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    total += value[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]
                    many += count[1 + dy:1 + dy + n, 1 + dx:1 + dx + n]
        grow = ~filled & (many > 0)
        pic[grow] = total[grow] / many[grow, None]
        filled |= grow
    return pic


def write_exr(path: Path, pic: np.ndarray) -> int:
    """Write an (n, n, 3) picture as half-float ZIP EXR, no tone mapping. Returns the bytes."""
    n = pic.shape[0]
    img = bpy.data.images.new("out", n, n, float_buffer=True, alpha=False)
    img.colorspace_settings.name = "Non-Color"
    rgba = np.ones((n, n, 4), dtype=np.float32)
    rgba[:, :, :3] = pic
    img.pixels.foreach_set(rgba.reshape(-1))
    settings = bpy.context.scene.render.image_settings
    settings.file_format = "OPEN_EXR"
    settings.color_depth = "16"
    settings.exr_codec = "ZIP"
    settings.color_mode = "RGB"
    img.save_render(str(path), scene=bpy.context.scene)
    bpy.data.images.remove(img)
    return path.stat().st_size


def rms(values: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values[mask] ** 2))) if mask.any() else 0.0


def bake_layers(spec: dict, still: dict[str, list], seen: dict[str, np.ndarray],
                size: int) -> dict:
    """Bake every layer and write its picture. Returns the `layers` record for bake.json.
    `noise` is the RMS of (A - B) / 2 over a placement's seen (covered) texels, `error` the RMS
    over the same texels and over the 4 phases of the 2 x 2 block grid of the picture shrunk at
    that phase and grown back by 2, minus the mean of A and B (`core.bake_grid.phase_error`);
    both at the doubled size. The shrink is covered-only, so `error` measures the picture's
    resolution and not the black outside an island; it takes every phase, so it does not depend
    on where the packer put the island. The shipped picture is phase (0, 0)."""
    scene = bpy.context.scene
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    size2 = 2 * size
    objs = [ob for obs in still.values() for ob in obs if seen[ob.name].any()]
    masks = texel_masks(still, seen, size2)
    covered = np.logical_or.reduce(list(masks.values()))  # every seen texel of the atlas
    values = spec.get("light_values", {})
    half = max(1, spec["samples"] // 2)
    if 2 * half != spec["samples"]:
        print(f"BAKE samples: {2 * half} baked (two bakes of {half}); the stage asked for "
              f"{spec['samples']}", flush=True)

    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.use_denoising = False
    scene.cycles.sample_clamp_direct = 0.0  # no clamp: a clamp would make light non-linear
    scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_animated_seed = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = THREADS
    bake = scene.render.bake
    bake.use_selected_to_active = False
    bake.use_clear = True
    bake.margin_type = "EXTEND"
    bake.margin = BAKE_MARGIN_PX
    bake.use_pass_color = False  # light, not paint
    bake.use_pass_direct = spec["mode"] == "full"
    bake.use_pass_indirect = True

    image = bpy.data.images.new("bake", size2, size2, float_buffer=True, alpha=False)
    image.colorspace_settings.name = "Non-Color"
    nodes, temps = prepare_materials(objs, image)
    select_only(objs)
    record = {}
    for i, layer in enumerate(spec["layers"]):
        lights = [li for li in spec["lights"] if li["layer"] == layer]
        made = {li["id"]: make_light(li, values) for li in lights if li["kind"] != "hemisphere"}
        set_world({li["sky_ground"] for li in lights if li["kind"] == "hemisphere"})
        kind = {li["id"]: li["kind"] for li in lights}
        used = {lid: {"kind": kind[lid], "color": [round(float(c), 6) for c in ob.data.color],
                      "strength": float(ob.data.energy)} for lid, ob in made.items()}
        bg = scene.world.node_tree.nodes["Background"]
        for li in lights:
            if li["kind"] == "hemisphere":  # the world, at full value in its own half
                used[li["id"]] = {
                    "kind": li["kind"],
                    "color": [round(float(c), 6) for c in bg.inputs["Color"].default_value[:3]],
                    "strength": float(bg.inputs["Strength"].default_value),
                    "half": li["sky_ground"]}
        seeds = [spec["seed"] + 2 * i, spec["seed"] + 2 * i + 1]
        a = bake_once(image, seeds[0], half)
        b = bake_once(image, seeds[1], half)
        mean = (a + b) / 2
        small, filled = shrink_covered(mean, covered)
        shipped = dilate(small, filled, DILATE_PX).astype(np.float16).astype(np.float32)
        noise = {pid: rms((a - b) / 2, mask) for pid, mask in masks.items() if mask.any()}
        error = phase_error(mean, covered, masks)  # the 4 grid phases' RMS (core/bake_grid.py)
        nbytes = write_exr(out / f"{layer}.exr", shipped)
        print(f"PICTURE {layer}.exr {nbytes} bytes ({size} px square)", flush=True)
        record[layer] = {"lights": [li["id"] for li in lights], "picture": f"{layer}.exr",
                         "seed": seeds[0], "noise": noise, "error": error,
                         "baked_lights": used}
        for ob in made.values():
            data = ob.data
            bpy.data.objects.remove(ob)
            bpy.data.lights.remove(data)
        bpy.data.worlds.remove(scene.world)
        scene.world = None
    restore_materials(nodes, temps)
    bpy.data.images.remove(image)
    return record


# --- export ---------------------------------------------------------------------------------


def export(spec: dict, size: int, channel: dict, unseen: list, layers: dict) -> None:
    out = Path(spec["out"])
    out.mkdir(parents=True, exist_ok=True)
    for ob in bpy.data.objects:  # hidden objects still ship; hide_render stays set
        ob.hide_set(False)
    build.export_glb(out / f"{spec['layout']}.glb")
    (out / "bake.json").write_text(json.dumps({
        "bake": 1, "layout": spec["layout"], "mode": spec["mode"], "size": size,
        "uv_channel": channel, "unseen": unseen, "live": spec["live"],
        # a texel is its layer's irradiance at strength 1, over pi (the mixer's UNITS undoes it)
        "format": "exr-half-zip", "units": "irradiance/pi",
        "samples": 2 * max(1, spec["samples"] // 2),
        "seed": spec["seed"], "blender": bpy.app.version_string, "layers": layers,
    }, indent=1))


def main() -> None:
    (arg,) = runner.script_args()
    spec = json.loads(arg)
    clear()
    objects = place(spec)
    live = set(spec["live"])
    still = {pid: obs for pid, obs in objects.items() if pid not in live}
    seen = see(spec, still)
    size, channel = unwrap(spec, still, seen)
    unseen = sorted(name for name, flags in seen.items() if not flags.any())
    layers = bake_layers(spec, still, seen, size)
    export(spec, size, channel, unseen, layers)
    runner.done()


main()
