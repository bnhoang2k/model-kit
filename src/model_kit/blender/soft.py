"""Soft parts in Blender: a part's `form` (blob, ring, meta or decal) becomes a mesh object.

The geometry is built in the piece's frame. Blob and ring meshes come from `core.forms` in glTF
axes and go through `build.B`. A meta form is a metaball, converted to a mesh in Blender, then
smoothed, decimated and deformed. A ring smooths in `core.forms` already, so it is not smoothed
again here. Every part ends as one closed mesh wound outward, or the build raises; a decal is the
one open patch, cast onto its `on` host (`decal_mesh`) and lifted off it by the piece's contact
length (`sight.decal_lift`), which must clear one depth step at the piece's farthest view.

`seat_on` stands a part on a host by a ray from the front (glTF +Z) toward -Z. `mirror_part`
copies a built part's triangles (as Blender splits them) with X negated and every face reversed,
so the two ship as mirror images triangle for triangle; the source is left as it was.
`beauty_triangulate` is the separate step that fixes a mirror source's diagonals, which
`build_piece` runs before it mirrors.
`build_piece` builds a whole rigged piece from its authored pack.json: the armature, every part
and its bind, and the registration.
"""

from __future__ import annotations

from collections.abc import Mapping

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

from model_kit.blender.bind import bind_part, make_armature
from model_kit.blender.build import G3, B, PackBuilder, mesh_obj, object_name
from model_kit.blender.build import piece as piece_root
from model_kit.core import forms, geom
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Part, Piece
from model_kit.core.manifest.forms import BlobForm, DecalForm, MetaForm, RingForm, SeatOn
from model_kit.core.sight import DecalLift, decal_lift
from model_kit.core.units import Style

# A metaball ball of element radius r has a surface radius of r * this.
META_SURFACE_PER_RADIUS = 0.575  # measured 0.5748 in Blender
# The element radius of every ellipsoid; its size triple carries the half-extents.
META_ELLIPSOID_RADIUS = 0.1  # the chibi build passes r = 0.1 * META_SURFACE_PER_RADIUS
# The ray starts this far in front of the host's frontmost vertex.
SEAT_RAY_MARGIN = 1.0
# The temporary metaball's name. Metaballs that share a name before the first '.' fuse into one
# surface, so the name stays one fixed word and no two metaballs exist at once.
_META_TMP = "model_kit_meta_tmp"
# The glTF front (+Z) in Blender axes, the direction a seated part turns onto the hit normal.
_FRONT = Vector(B(0.0, 0.0, 1.0))
# The glTF -Z (the way the seat ray travels) in Blender axes.
_AWAY = Vector(B(0.0, 0.0, -1.0))
# The glTF up (+Y) in Blender axes: a decal's oval keeps its height along it.
_UP = Vector(B(0.0, 1.0, 0.0))
# Shortest up direction, once laid onto a decal's tangent plane, that still gives the oval an up.
# A float-noise floor only: a cast near glTF +-Y still builds, and the oval's turn then follows
# the facet the cast hits.
_FRAME_MIN = 1e-9


# --- the piece frame -----------------------------------------------------------------------


def _piece_root(ob):
    """The top of `ob`'s parent chain, or None when it has no parent."""
    while ob.parent is not None:
        ob = ob.parent
    return ob if ob is not None and ob.type == "EMPTY" else None


def _to_piece(ob) -> Matrix:
    """The matrix from `ob`'s local space to its piece's frame."""
    root = _piece_root(ob)
    return (root.matrix_world.inverted() if root else Matrix.Identity(4)) @ ob.matrix_world


def _piece_name(parent) -> str | None:
    while parent is not None and parent.parent is not None:
        parent = parent.parent
    return parent.name if parent is not None else None


def _object_name(parent, part: str) -> str:
    piece = _piece_name(parent)
    return part if piece is None else object_name(piece, part)


# --- checks --------------------------------------------------------------------------------


def _check_surface(name: str, verts: np.ndarray, faces, closed: bool = True, host=None) -> None:
    """Raise unless the mesh is one connected surface whose faces agree on winding (glTF verts).
    With `closed` it must also be closed and wound outward; without, it is an open patch (a
    decal) whose edges may end at its rim. A patch must also be one fan at every vertex (two
    patches that touch at a single vertex, a bowtie, are two parts), and given its `host` it
    must face the way the host does: every face's normal agrees in sign with the normal of the
    host face nearest it. A mirrored patch has no host given; its source was judged."""
    n = len(verts)
    edges = np.array([(f[k], f[(k + 1) % len(f)]) for f in faces for k in range(len(f))])
    key = edges[:, 0] * n + edges[:, 1]
    back = edges[:, 1] * n + edges[:, 0]
    if len(np.unique(key)) != len(key) or (closed and not np.isin(key, back).all()):
        raise ManifestError(f"part {name!r}: the mesh is not "
                            f"{'closed' if closed else 'a clean patch'}, or its faces disagree "
                            "on winding")
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for f in faces:
        for a in f[1:]:
            parent[find(a)] = find(f[0])
    used = {find(a) for f in faces for a in f}
    if len(used) != 1:
        raise ManifestError(f"part {name!r}: the mesh has {len(used)} separate pieces; "
                            "a part is one mesh")
    if not closed:
        _check_fans(name, faces)
        if host is not None:
            _check_facing(name, verts, faces, host)
        return
    tris = np.array([(f[0], f[k], f[k + 1]) for f in faces for k in range(1, len(f) - 1)])
    if geom.signed_volume(verts[tris]) <= 0:
        raise ManifestError(f"part {name!r}: the mesh faces inward (its signed volume is not "
                            "above 0)")


def _check_fans(name: str, faces) -> None:
    """Raise when the faces round a vertex form more than one fan: two faces round a vertex are
    in one fan when a chain of faces, each sharing an edge at that vertex with the next, joins
    them. Two patches that meet at a single vertex give that vertex two fans."""
    round_vertex: dict[int, list[int]] = {}
    for fi, f in enumerate(faces):
        for v in f:
            round_vertex.setdefault(int(v), []).append(fi)
    for v, round_faces in round_vertex.items():
        # The two neighbours of v in a face are the far ends of its two edges at v.
        edges = {}
        for fi in round_faces:
            f = list(faces[fi])
            k = f.index(v)
            edges[fi] = {int(f[(k + 1) % len(f)]), int(f[k - 1])}
        seen, todo = {round_faces[0]}, [round_faces[0]]
        while todo:
            here = todo.pop()
            for other in round_faces:
                if other not in seen and edges[here] & edges[other]:
                    seen.add(other)
                    todo.append(other)
        if len(seen) != len(round_faces):
            raise ManifestError(f"part {name!r}: its faces meet at vertex {v} in more than one "
                                "fan (two patches touch at a single point)")


def _check_facing(name: str, verts: np.ndarray, faces, host) -> None:
    """Raise when a face of the open patch is wound against its host: its normal (the winding's)
    and the normal of the host face nearest its middle point opposite ways."""
    tree, _, _ = _host_tree(host)
    for fi, f in enumerate(faces):
        corners = [Vector(B(*map(float, verts[i]))) for i in f]
        normal = Vector()
        for k, a in enumerate(corners):  # Newell's normal: right for any simple polygon
            b = corners[(k + 1) % len(corners)]
            normal += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x),
                              (a.x - b.x) * (a.y + b.y)))
        middle = sum(corners, Vector()) / len(corners)
        nearest, host_normal, _, _ = tree.find_nearest(middle)
        if nearest is None:
            raise ManifestError(f"part {name!r}: no host face near face {fi}")
        if normal.dot(host_normal) <= 0:
            raise ManifestError(f"part {name!r}: face {fi} is wound against its host (its "
                                "normal points into the host)")


# --- the metaball --------------------------------------------------------------------------


def _meta_mesh(name: str, meta: MetaForm) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Polygonise `meta`'s elements, then smooth, then decimate. Returns glTF verts and faces."""
    stale = [o.name for o in bpy.data.objects if o.name.split(".")[0] == _META_TMP]
    if stale:
        raise ManifestError(f"part {name!r}: temporary object {stale[0]!r} is left from an "
                            "earlier build")
    mb = bpy.data.metaballs.new(_META_TMP)
    mb.resolution, mb.threshold = meta.resolution, meta.threshold
    for el in meta.elements:
        e = mb.elements.new(type=el.kind.upper())
        e.co = B(*el.centre)
        if el.kind == "ball":
            e.radius = el.half[0] / META_SURFACE_PER_RADIUS
        else:
            e.radius = META_ELLIPSOID_RADIUS
            hx, hy, hz = el.half
            scale = META_SURFACE_PER_RADIUS * META_ELLIPSOID_RADIUS
            e.size_x, e.size_y, e.size_z = hx / scale, hz / scale, hy / scale  # glTF -> Blender
    ob = bpy.data.objects.new(_META_TMP, mb)
    bpy.context.scene.collection.objects.link(ob)
    made = []
    try:
        bpy.context.view_layer.update()  # drop the entries of objects removed since the last part
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.convert(target="MESH")
        conv = bpy.context.view_layer.objects.active
        if conv is None or conv.type != "MESH":
            raise ManifestError(f"part {name!r}: Blender did not convert the metaball to a mesh")
        mesh = _smooth(conv.data, meta)
        if meta.decimate is not None:
            conv.modifiers.new("decimate", "DECIMATE").ratio = meta.decimate
            bpy.context.view_layer.update()
            mesh = bpy.data.meshes.new_from_object(
                conv.evaluated_get(bpy.context.evaluated_depsgraph_get()))
            made.append(mesh)
        verts = np.array([G3(v.co) for v in mesh.vertices], float)
        faces = [tuple(p.vertices) for p in mesh.polygons]
    finally:
        for o in [o for o in bpy.data.objects if o.name.split(".")[0] == _META_TMP]:
            data = o.data
            bpy.data.objects.remove(o)
            if data is not None and data.users == 0 and isinstance(data, bpy.types.Mesh):
                bpy.data.meshes.remove(data)
        for me in made:
            bpy.data.meshes.remove(me)
        for m in [m for m in bpy.data.metaballs if m.name.split(".")[0] == _META_TMP]:
            bpy.data.metaballs.remove(m)
    return verts, faces


def _smooth(mesh, meta: MetaForm):
    """Laplacian passes on X, Y and Z. With `below` set, only vertices under that glTF Y (Blender
    z), taken before the first pass, move."""
    smooth = meta.smooth
    bm = bmesh.new()
    bm.from_mesh(mesh)
    free = [v for v in bm.verts if smooth.below is None or v.co.z < smooth.below]
    for _ in range(smooth.passes):
        bmesh.ops.smooth_vert(bm, verts=free, factor=smooth.factor, use_axis_x=True,
                              use_axis_y=True, use_axis_z=True)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


# --- parts ---------------------------------------------------------------------------------


def _object(part: Part, verts: np.ndarray, faces, mat, parent):
    """A mesh object for `part` from glTF `verts` and outward-wound `faces`."""
    bm = bmesh.new()
    made = [bm.verts.new(B(*map(float, v))) for v in verts]
    for f in faces:
        bm.faces.new([made[i] for i in f])
    props = {"listed": part.listed, "group": part.group, "shape": part.shape, "role": part.role}
    if part.on is not None:
        props["on"] = part.on
    return mesh_obj(part.name, bm, mat, parent, props=props)


def soft_part(part: Part, mat, parent, host=None, lift: DecalLift | None = None):
    """The mesh object of `part`'s soft form, named `<piece>/<part>` under `parent`.

    Order: form geometry, smoothing (meta here, ring in core), decimate (meta), deforms, then
    the seat. `part.form` must hold a body: a mirror goes through `mirror_part`. A form with a
    seat needs `host`, the object it sits on; without one it raises. A host given to a form
    without a seat is ignored. The result must be one closed mesh wound outward, or it raises.

    A decal is cast onto `host`, which must be its `on` part, and lifted off it by `lift.lift`,
    the piece's contact length (`sight.decal_lift` makes it and refuses one under the depth gap);
    without either it raises. It is the one open mesh.
    """
    form = part.form
    if form is None:
        raise ManifestError(f"part {part.name!r}: has no soft form")
    if form.body is None:
        raise ManifestError(f"part {part.name!r}: is a mirror of {form.mirror_of!r}; "
                            "build it with mirror_part")
    body = form.body
    if isinstance(body, DecalForm):
        if part.on is None or host is None:
            raise ManifestError(f"part {part.name!r}: a decal is cast onto its `on` part, but "
                                f"its on is {part.on!r} and the host given is "
                                f"{None if host is None else host.name!r}")
        want = _object_name(parent, part.on)
        if host.name != want:
            raise ManifestError(f"part {part.name!r}: its decal host is {part.on!r}, but the "
                                f"host given is {host.name!r}, not {want!r}")
        if not isinstance(lift, DecalLift):
            raise ManifestError(f"part {part.name!r}: a decal's lift is the piece's contact "
                                f"length (a `sight.DecalLift`); got {lift!r}")
        verts, faces = decal_mesh(part.name, body, host, lift.lift)
        _check_surface(part.name, verts, faces, closed=False, host=host)
        return _object(part, verts, faces, mat, parent)
    if isinstance(body, BlobForm):
        verts, faces = forms.blob_mesh(body)
    elif isinstance(body, RingForm):
        verts, faces = forms.ring_mesh(body)
    elif isinstance(body, MetaForm):
        verts, faces = _meta_mesh(part.name, body)
    else:
        raise ManifestError(f"part {part.name!r}: unknown form body {type(body).__name__}")
    verts = forms.apply_deforms(verts, form.deforms)
    _check_surface(part.name, verts, faces)
    ob = _object(part, verts, faces, mat, parent)
    if form.seat is not None:
        if host is None:
            raise ManifestError(f"part {part.name!r}: its seat is on {form.seat.on!r}, but no "
                                "host was given")
        want = _object_name(parent, form.seat.on)
        if host.name != want:
            raise ManifestError(f"part {part.name!r}: its seat is on {form.seat.on!r}, but the "
                                f"host given is {host.name!r}, not {want!r}")
        seat_on(ob, host, form.seat)
    return ob


def beauty_triangulate(ob) -> None:
    """Split every face of `ob` with more than 3 corners into triangles, in place, by Blender's
    BEAUTY rule (the diagonal that makes the better-shaped pair), which reads the shape only.

    This is a build step with a visible effect: the part ships these diagonals. The exporter's
    own split of a quad can fold over the crease angle where BEAUTY does not (bar 03's shins
    drew a dash at a crease), so a mirror source is split here before `mirror_part` reads it.
    A mesh that is all triangles is left as it is."""
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    big = [f for f in bm.faces if len(f.verts) > 3]
    if big:
        bmesh.ops.triangulate(bm, faces=big, quad_method="BEAUTY", ngon_method="BEAUTY")
        bm.to_mesh(ob.data)
        ob.data.update()
    bm.free()


def _loop_triangles(ob) -> list[tuple[int, int, int]]:
    """`ob`'s faces as triangles, by Blender's own split (the one the exporter ships), winding
    kept. `ob` is read, never changed."""
    me = ob.data
    me.calc_loop_triangles()
    return [tuple(t.vertices) for t in me.loop_triangles]


def mirror_part(part: Part, src, mat, parent):
    """`part`, the X-mirror of the built part `src`: X negated, every face reversed, so the
    signed volume stays positive. `src` is read as it sits in the piece (seat and all), and
    is left as it is.

    The twin is made from `src`'s triangles, split as Blender splits them (`_loop_triangles`),
    not from its faces. The glTF exporter splits a quad along a diagonal picked by its vertex
    order, and a reversed face starts at another corner, so a quad twin and its quad source
    could ship different diagonals and fold (and draw outline creases) differently. A twin of
    the source's own triangles is the exact mirror of the shipped source, and `src` is left
    as it is.

    The diagonals a quad source ships are the exporter's, which can fold over the crease angle
    where BEAUTY does not. `build_piece` therefore runs `beauty_triangulate` on a source first, as
    a named step of its own; a source split that way gives the same twin as before."""
    form = part.form
    if form is None or form.mirror_of is None:
        raise ManifestError(f"part {part.name!r}: has no mirror_of")
    want = _object_name(parent, form.mirror_of)
    if src.name != want:
        raise ManifestError(f"part {part.name!r}: mirror_of names {form.mirror_of!r}, but the "
                            f"source given is {src.name!r}, not {want!r}")
    to_piece = _to_piece(src)
    verts = np.array([G3(to_piece @ v.co) for v in src.data.vertices], float)
    verts[:, 0] = -verts[:, 0]
    faces = [tuple(reversed(t)) for t in _loop_triangles(src)]
    _check_surface(part.name, verts, faces, closed=part.shape != "decal")
    return _object(part, verts, faces, mat, parent)


# --- seats ---------------------------------------------------------------------------------


def _host_tree(host) -> tuple[BVHTree, np.ndarray, np.ndarray]:
    """A ray tree of `host`'s evaluated mesh in the piece frame (Blender axes), and the low and
    high corners of its box."""
    bpy.context.view_layer.update()
    evaluated = host.evaluated_get(bpy.context.evaluated_depsgraph_get())
    bm = bmesh.new()
    bm.from_mesh(evaluated.to_mesh())
    evaluated.to_mesh_clear()
    bm.transform(_to_piece(host))
    bm.normal_update()  # the tree reads the stored face normals, which transform leaves behind
    co = np.array([tuple(v.co) for v in bm.verts], float)
    tree = BVHTree.FromBMesh(bm)
    bm.free()
    return tree, co.min(axis=0), co.max(axis=0)


def seat_on(ob, host, seat: SeatOn) -> None:
    """Stand `ob` on `host`: cast from the front (+Z) toward -Z at glTF (x, y) = `seat.at`, turn
    `ob` so its local +Z (glTF front, Blender -Y) meets the hit's face normal, and put its centre
    (the middle of its local bounding box) `seat.sink` behind the hit along the ray. A miss
    raises. Casting is in the piece frame, against the host's evaluated mesh."""
    tree, lo, _ = _host_tree(host)
    front = float(lo[1])  # Blender -Y is the front
    origin = Vector((seat.at[0], front - SEAT_RAY_MARGIN, seat.at[1]))
    hit, normal, _, _ = tree.ray_cast(origin, _AWAY)
    if hit is None:
        raise ManifestError(f"part {ob.name!r}: seat on {seat.on!r} at {seat.at}: a ray from "
                            "the front misses the host there")
    lo = Vector(map(min, zip(*(v.co for v in ob.data.vertices), strict=True)))
    hi = Vector(map(max, zip(*(v.co for v in ob.data.vertices), strict=True)))
    centre = (lo + hi) / 2
    turn: Quaternion = _FRONT.rotation_difference(normal)
    place = (Matrix.Translation(hit + _AWAY * seat.sink) @ turn.to_matrix().to_4x4()
             @ Matrix.Translation(-centre))
    root = _piece_root(ob)
    ob.matrix_world = (root.matrix_world if root else Matrix.Identity(4)) @ place


# --- decals --------------------------------------------------------------------------------


def decal_mesh(name: str, decal: DecalForm, host, lift: float
               ) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """`decal` cast onto `host`, as glTF verts in the piece frame and faces wound outward.

    A cast along `decal.along` through `decal.at`, from outside the host's box, finds the first
    hit and its face normal. The oval (`core.forms.decal_grid`) lies in the plane through the
    hit square to that normal, its height along glTF +Y laid onto the plane. Each grid point is
    then cast back onto the host along the normal, from outside the box, and set `lift` off the
    host along the normal of the face it hits. A miss (the outline runs off the host), a hit on
    a face turned away from the cast, or a normal straight up or down raises. So does a patch
    face whose middle or edge middle dips under the host: a flat face spans the host's curve as
    a chord, and a grid too coarse for the host's curve cuts into it."""
    tree, lo, hi = _host_tree(host)
    reach = float(np.linalg.norm(hi - lo))
    centre = Vector(((lo + hi) / 2).tolist())
    along = Vector(B(*decal.along))
    at = Vector(B(*decal.at))
    hit, normal, _, _ = tree.ray_cast(at - along * ((at - centre).length + reach), along)
    if hit is None:
        raise ManifestError(f"part {name!r}: a decal cast along {list(decal.along)} through "
                            f"{list(decal.at)} misses its host")
    if normal.dot(along) >= 0:
        raise ManifestError(f"part {name!r}: the decal cast meets its host on a face turned "
                            "away from it")
    up = _UP - normal * _UP.dot(normal)
    if up.length < _FRAME_MIN:
        raise ManifestError(f"part {name!r}: the host faces straight up or down at the decal, "
                            "so its oval has no up")
    up.normalize()
    side = up.cross(normal)
    points, faces = forms.decal_grid(decal)
    verts = []
    for s, t in points:
        p = hit + side * float(s) + up * float(t)
        on, face_normal, _, _ = tree.ray_cast(p + normal * reach, -normal)
        if on is None:
            raise ManifestError(f"part {name!r}: the decal's outline runs off its host at "
                                f"{list(G3(p))}")
        if face_normal.dot(normal) <= 0:
            raise ManifestError(f"part {name!r}: the decal lands on a host face turned away "
                                f"from it at {list(G3(on))}")
        verts.append(on + face_normal * lift)
    for f in faces:
        corners = [verts[i] for i in f]
        middles = [sum(corners, Vector()) / len(corners)]
        middles += [(corners[k] + corners[(k + 1) % len(corners)]) / 2 for k in range(len(f))]
        for p in middles:
            on, _, _, _ = tree.ray_cast(p + normal * reach, -normal)
            if on is not None and (p - on).dot(normal) <= 0:
                raise ManifestError(f"part {name!r}: a decal face dips under its host at "
                                    f"{list(G3(p))}; its grid is too coarse for the host's "
                                    "curve (more segments or rings)")
    return np.array([G3(v) for v in verts], float), faces


# --- a whole piece -------------------------------------------------------------------------


def build_piece(builder: PackBuilder, piece: Piece, mats: Mapping[str, bpy.types.Material],
                style: Style | None = None) -> tuple[bpy.types.Object, bpy.types.Object]:
    """Build the rigged piece `piece` (from the authored pack.json, via `load_pack`) and
    register it with `builder`. Returns the piece root and its armature.

    `mats` maps each part name to its material. The armature comes first, under the root. Then
    each part in order: `soft_part` (seated on the part its form's seat names; a decal cast onto
    its `on` part, lifted by the piece's contact length) or `mirror_part`,
    parented to the armature, and bound by `rig.bind[part]`. Last, the piece is registered as a
    `character` with its rig and every part's form, so the written pack.json holds the authored
    fields unchanged, the piece's `view` too.

    A piece with a decal needs `style` (its depth camera): the decal's lift is the piece's
    contact length, and the build raises, naming the piece, the lift and the gap, when that is
    under one depth step at the piece's farthest view (`sight.decal_lift`).

    Everything `builder.check_registration` can judge is judged first, so a piece with no view
    raises before it makes an object. A piece with no rig, a part with no form or no material,
    or a seat or mirror source that is not built yet raises too.
    """
    builder.check_registration(name=piece.name, view=piece.view)
    lift = None
    if any(isinstance(p.form.body, DecalForm) for p in piece.parts if p.form is not None):
        if style is None:
            raise ManifestError(f"piece {piece.name!r}: has a decal, so the build needs the "
                                "style (its depth camera) to check the decal's lift")
        lift = decal_lift(f"piece {piece.name!r}", piece.view, style, builder.units_per_metre)
    rig = piece.rig
    if rig is None:
        raise ManifestError(f"piece {piece.name!r}: has no rig; build_piece builds rigged pieces")
    missing = [p.name for p in piece.parts if p.name not in mats]
    if missing:
        raise ManifestError(f"piece {piece.name!r}: no material for parts {missing}")
    bare = [p.name for p in piece.parts if p.form is None]
    if bare:
        raise ManifestError(f"piece {piece.name!r}: parts {bare} have no soft form")
    root = piece_root(piece.name)
    armature = make_armature(root, rig.armature, rig.bones)
    built: dict[str, bpy.types.Object] = {}

    def earlier(part: Part, name: str, what: str) -> bpy.types.Object:
        if name not in built:
            raise ManifestError(f"part {part.name!r}: its {what} is {name!r}, which is not built "
                                "before it")
        return built[name]

    for part in piece.parts:
        form = part.form
        if form.mirror_of is not None:
            source = earlier(part, form.mirror_of, "mirror source")
            beauty_triangulate(source)  # the shipped source carries the diagonals it is mirrored by
            ob = mirror_part(part, source, mats[part.name], armature)
        else:
            if isinstance(form.body, DecalForm):
                host = earlier(part, part.on, "decal host") if part.on is not None else None
            elif form.seat is not None:
                host = earlier(part, form.seat.on, "seat host")
            else:
                host = None
            ob = soft_part(part, mats[part.name], armature, host=host, lift=lift)
        bind_part(ob, armature, rig.bind[part.name])
        built[part.name] = ob
    builder.register(root, origin=piece.frame.origin, recipe=piece.recipe, listed=piece.listed,
                     done=piece.done, size_class=piece.size_class, kind="character", rig=rig,
                     forms={p.name: p.form for p in piece.parts}, view=piece.view)
    return root, armature
