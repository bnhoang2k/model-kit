"""Build helpers that run inside Blender: axes, shapes, material slots and `PackBuilder`.

A build script: `reset()`; `mats = slots(style_roles)`; `selftest_axes()`; make a
`PackBuilder`; build each piece under a `piece()` root and `register` it; add its uses, holds,
seats, mounts, keep-offs and connectors; then `finish(order)` writes `<pack>.glb`,
`<pack>.pack.json` and `<pack>.blend` into the builder's `out_dir` and nowhere else.

Axes. Blender is Z up. The glTF exporter (`export_yup=True`) maps Blender (x, y, z) to glTF
(x, z, -y), the change `axes.BLENDER_TO_GLTF` holds. So a glTF point (X, Y up, Z front) is
built at Blender (X, -Z, Y): that is `B()`. Every position a helper takes is glTF and goes
through `B()`; every record the builder writes is glTF in the piece's own frame.

Parts. Every mesh below a piece root, at any depth, is one part. Blender names are global,
so the object is named `<piece>/<part>` and pack.json keeps the bare part name. Parts are
written in the order they were made, so the first-seen order of their groups is the take-apart
order. Custom properties give the rest: `listed` and `group` (required), `on` (optional),
`role` (optional, default `solid`) and `shape` (set by `slab`, `lathe` and `extrude_profile`;
give it in `props` for any other mesh). They also reach the .glb as node extras. An empty under
a piece must be a `surface`, `stand` or `pivot` empty; any other object there raises.

A rigged piece (`register(rig=...)`) also holds exactly one armature object, named
`rig.armature`, under its root. Its parts may sit under the armature, the parenting a glTF skin
expects. Every part carries an Armature modifier to that armature, and `finish` keeps the
modifiers and the vertex groups. Any other armature raises.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import count, pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any

import bmesh
import bpy
import numpy as np
from mathutils import Vector

from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.core.errors import ManifestError, RecordError
from model_kit.core.glb import node_name, read_glb_json
from model_kit.core.manifest import (
    Connector,
    End,
    Frame,
    Hold,
    KeepOff,
    Listed,
    Mount,
    Pack,
    Part,
    Piece,
    Recipe,
    Ref,
    RigSpec,
    Seat,
    SitSpec,
    Use,
    View,
)
from model_kit.core.manifest.forms import SoftForm
from model_kit.core.manifest.records import PART_TAGGED
from model_kit.core.targets import TIP_AXIS, seat_root
from model_kit.core.units import check_crease_deg
from model_kit.core.writer import pack_text, write_pack

# glTF exporter options for a pack of many small parts: Draco mesh compression. `finish` takes
# them as `glb_options`; the bars share them.
DRACO = {"export_draco_mesh_compression_enable": True,
         "export_draco_mesh_compression_level": 6}
BEVEL = 0.035  # side-profile bevel of extrude_profile, in build units
FILL_GRID = (24, 12)  # rays across X and up Y that fill_fraction casts front to back
DEFAULT_ROLE = "solid"
_NAME = re.compile(r"[A-Za-z0-9_-]+")
_ORDER = count()  # build order of parts: finish() writes parts in the order they were made
MEASURED_EMPTY_KINDS = ("surface", "stand", "pivot")
_EXTRAS_ORDER = ("listed", "group", "shape", "on", "role")  # the order the bars' .glb extras use
# The properties tag() sets and _part() reads: every tagged Part field. The five older ones keep
# their order, so a bar that sets no new field writes the same .glb bytes as before.
PART_PROPS = (*_EXTRAS_ORDER, *(k for k in PART_TAGGED if k not in _EXTRAS_ORDER))
_FLAG_PROPS = ("fastener",)  # true or false; unset means false
_LIST_PROPS = ("retains",)  # a list of part names; unset means none
REQUIRED_PART_PROPS = ("listed", "group", "shape")  # the ones tag() cannot do without


# --- axes ----------------------------------------------------------------------------------


def B(X, Y, Z_):
    """glTF (X, Y up, Z front) -> Blender (x, y, z)."""
    return (X, -Z_, Y)


def G3(v):
    """Blender (x, y, z) -> glTF (X, Y, Z)."""
    return (v[0], v[2], -v[1])


def _check_axes() -> None:
    """B and G3 are the plain-tuple form of `axes.BLENDER_TO_GLTF`; they must agree with it."""
    turn = BLENDER_TO_GLTF[:3, :3]
    for p in np.eye(3) + np.arange(3.0):
        if not (np.allclose(G3(p), turn @ p) and np.allclose(B(*(turn @ p)), p)):
            raise ManifestError("build.B / build.G3 disagree with axes.BLENDER_TO_GLTF")


_check_axes()


# --- the file, materials and objects -------------------------------------------------------


def reset():
    """Empty the file. Headless: factory settings. Live (an add-on would be unloaded by factory
    settings): remove every object, mesh, material and action, plus cameras, lights and child
    collections."""
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.materials, bpy.data.actions,
                 bpy.data.cameras, bpy.data.lights):
        for block in list(coll):
            coll.remove(block)
    for scn in bpy.data.scenes:
        for c in list(scn.collection.children):
            scn.collection.children.unlink(c)
    for c in list(bpy.data.collections):
        if c.users == 0:
            bpy.data.collections.remove(c)


def _rgb(name: str, rgb) -> tuple[float, float, float]:
    if (not isinstance(rgb, Sequence) or isinstance(rgb, str) or len(rgb) != 3
            or not all(isinstance(c, int | float) and not isinstance(c, bool)
                       and math.isfinite(c) and 0 <= c <= 1 for c in rgb)):
        raise ManifestError(f"style role {name!r}: colour must be 3 numbers from 0 to 1, "
                            f"got {rgb!r}")
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def _slot(name: str, rgb: tuple[float, float, float]):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    if m.name != name:
        raise ManifestError(f"style role {name!r}: Blender named the material {m.name!r}")
    m.diffuse_color = (*rgb, 1.0)
    if m.node_tree is None:  # Blender before 5.0; 5.x materials always have a node tree
        m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF") if m.node_tree else None
    if bsdf is None:
        raise ManifestError(f"style role {name!r}: material {name!r} has no 'Principled BSDF' "
                            "node to colour")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    return m


def _checked_role(name, rgb) -> tuple[str, tuple[float, float, float]]:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ManifestError(f"style role name must be letters, digits, '_' or '-', "
                            f"got {name!r}")
    return name, _rgb(name, rgb)


def slot(name: str, rgb: Sequence[float]):
    """One material for a style role (colour from 0 to 1). Call after `reset()`. Any failure
    raises: a bad name or colour, or a material with no Principled BSDF to colour."""
    return _slot(*_checked_role(name, rgb))


def slots(style_roles: Mapping[str, Sequence[float]]) -> dict:
    """One material per style role (`{"Fill": (1, 1, 1), ...}`, colours from 0 to 1).

    Returns `{role: material}`. Call after `reset()`. Any failure raises: an empty mapping, a
    bad name or colour, or a material with no Principled BSDF to colour. Every role is checked
    before the first material is made.
    """
    if not isinstance(style_roles, Mapping) or not style_roles:
        raise ManifestError(f"style roles must be a non-empty mapping, got {style_roles!r}")
    roles = [_checked_role(name, rgb) for name, rgb in style_roles.items()]
    return {name: _slot(name, rgb) for name, rgb in roles}


def _named(ob, name: str):
    if ob.name != name:
        raise ManifestError(f"object name {name!r} is taken; Blender named it {ob.name!r}")
    return ob


def _top(ob):
    while ob.parent is not None:
        ob = ob.parent
    return ob


def object_name(piece_name: str, part: str) -> str:
    """The Blender object name of a piece's part. Blender names are global, so a part is named
    `<piece>/<part>`; pack.json keeps the bare part name, so two pieces may share one."""
    return node_name(piece_name, part)


def _check_part_props(where: str, fields: Mapping) -> None:
    """`listed`, `group` and `shape` must be strings; `on` and `role` may also be None;
    `fastener` must be true or false; `retains` must be a list or tuple of strings."""
    for key, value in fields.items():
        if key in _FLAG_PROPS:
            if not isinstance(value, bool):
                raise ManifestError(f"{where}: {key!r} must be true or false, got {value!r}")
        elif key in _LIST_PROPS:
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise ManifestError(f"{where}: {key!r} must be a list of part names, got "
                                    f"{value!r}")
        elif not isinstance(value, str) and not (value is None and key in ("on", "role")):
            raise ManifestError(f"{where}: {key!r} must be a string, got {value!r}")


def tag(ob, piece: str, part: str, *, listed: str, group: str, shape: str,
        on: str | None = None, role: str | None = None, fastener: bool = False,
        retains: tuple[str, ...] = ()) -> None:
    """Make a mesh that other code built into a part of `piece`.

    Names the object `object_name(piece, part)` (a name another object holds raises, and `ob`
    keeps its name), names its mesh data the same when that name is free, sets the part
    properties and gives the part the next build order. A mesh shared with another object (an
    instance) stays shared and keeps its name: `_apply_and_shade` applies the modifiers once per
    mesh and the instances export as one glTF mesh. The mesh itself is not otherwise touched.

    Shading: `ob["smooth"]` comes from the mesh's own face flags, true when every face is smooth
    and false when none is. A mesh with mixed flags gets `ob["shade"] = "keep"`, and
    `_apply_and_shade` leaves its shading alone. The caller parents `ob` under the piece root.
    """
    for what, v in (("piece name", piece), ("part name", part)):
        if not isinstance(v, str) or not _NAME.fullmatch(v):
            raise ManifestError(f"{what} must be letters, digits, '_' or '-', got {v!r}")
    full = object_name(piece, part)
    if ob.type != "MESH":
        raise ManifestError(f"tag: {ob.name!r} is a {ob.type}; only a MESH becomes the part "
                            f"{full!r}")
    given = {"listed": listed, "group": group, "shape": shape, "on": on, "role": role,
             "fastener": fastener, "retains": retains}
    fields = {key: given[key] for key in PART_PROPS}  # a tagged field tag() lacks is a KeyError
    _check_part_props(f"piece {piece!r} part {part!r}", fields)
    other = bpy.data.objects.get(full)
    if other is not None and other != ob:
        raise ManifestError(f"tag: object name {full!r} is taken by another object; "
                            f"{ob.name!r} keeps its name")
    ob.name = full
    _named(ob, full)
    me = ob.data
    if me.users == 1 and bpy.data.meshes.get(full) is None:  # an instance's mesh keeps its name
        me.name = full
    for key, value in fields.items():
        if value is None or value is False or (key in _LIST_PROPS and not value):
            ob.pop(key, None)  # a default is not written, so a part without it is unchanged
        else:
            ob[key] = list(value) if key in _LIST_PROPS else value
    flags = [p.use_smooth for p in me.polygons]
    ob.pop("smooth", None)
    ob.pop("shade", None)
    if not flags or all(flags) or not any(flags):
        ob["smooth"] = bool(flags) and all(flags)  # read by finish(), removed there
    else:
        ob["shade"] = "keep"  # mixed flags: _apply_and_shade leaves the shading as it is
    ob["build_order"] = next(_ORDER)  # read by finish(), removed before the export


def mesh_obj(name, bm, mat, parent, smooth=True, props=None):
    """A mesh object. Under a piece, `name` is the part name and the object is named
    `<piece>/<name>` (see `object_name`); parts are written in the order they were made.

    `props` is set on the object as it is. When it holds every property `tag` requires
    (`listed`, `group` and `shape`), the object goes through `tag`, which checks them now. When
    one is missing, the object still gets `smooth` and a build order, and `finish` refuses the
    incomplete part, naming it and the missing property.
    """
    full = name
    extra = dict(props or {})
    if parent is not None:
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            bm.free()
            raise ManifestError(f"part name must be letters, digits, '_' or '-', got {name!r}")
        full = object_name(_top(parent).name, name)
    whole = parent is not None and all(k in extra for k in REQUIRED_PART_PROPS)
    if whole:  # check before anything is made, so a refused part leaves no object behind
        fields = {k: extra.pop(k) for k in PART_PROPS if k in extra}
        try:
            _check_part_props(f"part {full!r}", fields)
        except ManifestError:
            bm.free()
            raise
    me = bpy.data.meshes.new(full)
    for f in bm.faces:
        f.smooth = smooth
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(full, me)
    bpy.context.scene.collection.objects.link(ob)
    _named(ob, full)
    ob.parent = parent
    if whole:
        for k, v in extra.items():
            ob[k] = v
        tag(ob, _top(parent).name, name, **fields)
    else:  # not a whole part yet: finish() shades it and refuses it if it is a part
        ob["smooth"] = bool(smooth)  # read by finish(), removed there
        ob["build_order"] = next(_ORDER)  # read by finish(), removed before the export
        for k, v in extra.items():
            ob[k] = v
    return ob


def piece(name):
    """A piece root: a plain-axes empty. Every mesh below it is one of the piece's parts."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ManifestError(f"piece name must be letters, digits, '_' or '-', got {name!r}")
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    _named(e, name)
    e.empty_display_type = "PLAIN_AXES"
    e.empty_display_size = 0.15
    return e


def empty(name, parent, loc, props, facing_deg=0.0, kind_display="ARROWS"):
    """loc is glTF (X, Y, Z). facing is a compass bearing; it turns the empty about glTF Y by
    -facing."""
    e = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(e)
    e.parent = parent
    e.empty_display_type = kind_display
    e.empty_display_size = 0.06
    e.location = B(*loc)
    e.rotation_euler = (0, 0, math.radians(-facing_deg))
    for k, v in props.items():
        e[k] = v
    return e


# --- geometry (every position is glTF, mapped through B) -----------------------------------


def box_bm(x0, x1, y0, y1, z0, z1):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        X = x0 if v.co.x < 0 else x1
        Zv = z1 if v.co.y < 0 else z0  # Blender y = -Z
        Yv = y0 if v.co.z < 0 else y1
        v.co = Vector(B(X, Yv, Zv))
    return bm


def slab(name, x0, x1, y0, y1, z0, z1, mat, parent, r=0.0, crisp=False, props=None, seg=3):
    """A box. r > 0 adds a Bevel modifier (soft); crisp boxes stay flat and sharp."""
    bm = box_bm(x0, x1, y0, y1, z0, z1)
    ob = mesh_obj(name, bm, mat, parent, smooth=not crisp,
                  props=dict({"shape": "box"}, **(props or {})))
    if r > 0 and not crisp:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width = min(r, (x1 - x0) / 2 - 1e-3, (y1 - y0) / 2 - 1e-3, (z1 - z0) / 2 - 1e-3)
        mod.segments = seg
        mod.limit_method = "ANGLE"
    return ob


def rounded_profile(pts, segs=4):
    """pts: [(u, y, radius)] around a closed side profile. Returns 2D points with quadratic
    rounded corners."""
    out = []
    n = len(pts)
    for i, (u, y, r) in enumerate(pts):
        P = Vector((u, y))
        if r <= 0:
            out.append(P)
            continue
        a, b = Vector(pts[i - 1][:2]), Vector(pts[(i + 1) % n][:2])
        p0 = P + (a - P).normalized() * min(r, (a - P).length / 2)
        p1 = P + (b - P).normalized() * min(r, (b - P).length / 2)
        for k in range(segs + 1):
            t = k / segs
            out.append(p0 * (1 - t) ** 2 + P * 2 * (1 - t) * t + p1 * t * t)
    return out


def extrude_profile(name, pts, plane, lo, hi, mat, parent, props=None):
    """Closed profile extruded between lo and hi, then beveled: one shell.
    plane 'ZY': pts are (Z, Y), extruded across X. plane 'XY': pts are (X, Y), extruded
    across Z."""
    bm = bmesh.new()
    ring2 = rounded_profile(pts)
    if plane == "ZY":
        ring = [bm.verts.new(B(lo, p.y, p.x)) for p in ring2]
        vec = B(hi - lo, 0, 0)
    else:
        ring = [bm.verts.new(B(p.x, p.y, lo)) for p in ring2]
        vec = B(0, 0, hi - lo)
    face = bm.faces.new(ring)
    r = bmesh.ops.extrude_face_region(bm, geom=[face])
    bmesh.ops.translate(bm, verts=[v for v in r["geom"] if isinstance(v, bmesh.types.BMVert)],
                        vec=vec)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    ob = mesh_obj(name, bm, mat, parent, props=dict({"shape": "extrude"}, **(props or {})))
    mod = ob.modifiers.new("bevel", "BEVEL")
    mod.width = BEVEL
    mod.segments = 3
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(40)
    return ob


def lathe_into(bm, pts, x=0.0, z=0.0, segs=32, y0=0.0):
    """[(radius, y)] bottom to top, spun about glTF Y at (x, z), added to bm."""
    rings = []
    for (r, y) in pts:
        rings.append([bm.verts.new(B(x + r * math.cos(2 * math.pi * k / segs), y0 + y,
                                     z + r * math.sin(2 * math.pi * k / segs)))
                      for k in range(segs)])
    for a, b in pairwise(rings):
        for k in range(segs):
            try:
                bm.faces.new((a[k], a[(k + 1) % segs], b[(k + 1) % segs], b[k]))
            except ValueError:  # a ring of radius 0 makes repeat faces; remove_doubles welds it
                pass
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)


def lathe(name, pts, mat, parent, x=0.0, z=0.0, segs=32, y0=0.0, props=None):
    """A turned part. Every lathe is round: the `round` extra tells a renderer to colour its
    sides as one. Keep segs >= 15, or the facets turn more than a 25 degree edge threshold and
    draw outline lines."""
    bm = bmesh.new()
    lathe_into(bm, pts, x, z, segs, y0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_obj(name, bm, mat, parent,
                    props=dict({"round": True, "shape": "lathe"}, **(props or {})))


def puck_pts(r, h, b, n=4):
    """A round puck with a rounded rim, bottom at 0."""
    pts = [(0, 0)]
    for i in range(n + 1):
        a = -math.pi / 2 + i * math.pi / (2 * n)
        pts.append((r - b + math.cos(a) * b, b + math.sin(a) * b))
    for i in range(n + 1):
        a = i * math.pi / (2 * n)
        pts.append((r - b + math.cos(a) * b, h - b + math.sin(a) * b))
    pts.append((0, h))
    return pts


def flare_pts(r0, h0, rp, y1, R=0.1):
    """A foot disc (radius r0, height h0, round rim) flowing into a post of radius rp through a
    concave fillet of radius R, up to y1."""
    b = h0 / 2
    out = [(0, 0)]
    for i in range(9):
        a = -math.pi / 2 + i * math.pi / 8
        out.append((r0 - b + math.cos(a) * b, b + math.sin(a) * b))
    for i in range(7):
        a = -math.pi / 2 - i * math.pi / 12
        out.append((rp + R + math.cos(a) * R, h0 + R + math.sin(a) * R))
    out += [(rp, y1), (0, y1)]
    return out


def add_face(bm, pts, hint):
    """A face from glTF points, wound so its normal points along hint (open shells need
    explicit winding)."""
    n = Vector((0, 0, 0))
    for i in range(len(pts)):
        a, b = Vector(pts[i]), Vector(pts[(i + 1) % len(pts)])
        n += Vector(((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x),
                     (a.x - b.x) * (a.y + b.y)))
    order = list(pts) if n.dot(Vector(hint)) >= 0 else list(reversed(pts))
    return bm.faces.new([bm.verts.new(B(*p)) for p in order])


# --- export and the axis self-test ---------------------------------------------------------


def export_glb(path, **kw):
    """Export the scene as one .glb. Live, add-ons register plain properties on the scene,
    and export_extras would write every set one into the GLB's scene extras. They are unset for
    the export and put back, so a live GLB matches a headless one byte for byte."""
    scene = bpy.context.scene
    held = {}
    for p in scene.bl_rna.properties:
        if (p.is_runtime and p.type in ("BOOLEAN", "INT", "FLOAT", "STRING")
                and getattr(p, "array_length", 0) == 0 and scene.is_property_set(p.identifier)):
            held[p.identifier] = getattr(scene, p.identifier)
            scene.property_unset(p.identifier)
    try:
        bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", export_yup=True,
                                  export_extras=True, export_apply=True, **kw)
    finally:
        for k, v in held.items():
            setattr(scene, k, v)


def _exporter_option(name: str):
    """The glTF exporter's own property `name`, which carries its default and range."""
    props = bpy.ops.export_scene.gltf.get_rna_type().properties
    if name not in props:
        raise ManifestError(f"this Blender's glTF exporter has no option {name!r}, so the "
                            "build cannot tell what precision it ships")
    return props[name]


def position_bits(glb_options: Mapping[str, Any] | None) -> int | None:
    """The Draco position quantization, in bits, that `export_glb(**glb_options)` ships, or None
    when it ships float32 (Draco off).

    With Draco on and no bits set, the exporter's default is used. It is read from the exporter's
    own property `export_draco_position_quantization` (14 in Blender 5.2.2, checked there with
    `bpy.ops.export_scene.gltf.get_rna_type()`), so a Blender that changes it changes the record.
    Draco quantises each primitive over its own bounding box, so the size of a step also needs
    that box; the checks that use this number take it from the mesh. 0 bits is the exporter's
    "no quantization" (the property's minimum), so it records None, like float32.
    """
    options = glb_options or {}
    enable = "export_draco_mesh_compression_enable"
    if not options.get(enable, _exporter_option(enable).default):
        return None
    key = "export_draco_position_quantization"
    prop = _exporter_option(key)
    bits = options.get(key, prop.default)
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 <= bits <= prop.hard_max:
        raise ManifestError(f"glb_options {key} is {bits!r}; it must be a whole number of bits "
                            f"from 0 to {prop.hard_max}")
    return bits or None


def selftest_axes():
    """A 1 x 0.5 x 0.2 slab must export as X +-0.5, Y 0..0.5, Z +-0.1, and an empty must keep
    its place and facing. Removes what it made, and only that."""
    mat = bpy.data.materials.new("AxisTestMaterial")
    root = piece("AxisTest")
    slab_ob = slab("AxisSlab", -0.5, 0.5, 0, 0.5, -0.1, 0.1, mat, root, r=0.05,
                   props={"listed": "axis", "group": "axis"})
    seat = empty("AxisSeat", root, (0.1, 0.2, 0.3), {"kind": "seat"}, facing_deg=90)
    try:
        with tempfile.TemporaryDirectory(prefix="model-kit-axis-") as tmp:
            glb = os.path.join(tmp, "axis_test.glb")
            export_glb(glb)
            js = read_glb_json(glb)
    finally:
        me = slab_ob.data
        for ob in (seat, slab_ob, root):
            bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(me)
        bpy.data.materials.remove(mat)
    want = object_name("AxisTest", "AxisSlab")
    mesh = next(i for i, m in enumerate(js["meshes"]) if m["name"] == want)
    acc = js["accessors"][js["meshes"][mesh]["primitives"][0]["attributes"]["POSITION"]]
    lo, hi = acc["min"], acc["max"]
    want_lo, want_hi = (-0.5, 0.0, -0.1), (0.5, 0.5, 0.1)
    if not (all(abs(a - b) < 1e-4 for a, b in zip(lo, want_lo))
            and all(abs(a - b) < 1e-4 for a, b in zip(hi, want_hi))):
        raise ManifestError(f"axis test: glTF bounds {lo} {hi}")
    node = next(n for n in js["nodes"] if n.get("name") == "AxisSeat")
    t, q = node["translation"], node["rotation"]
    if not all(abs(a - b) < 1e-4 for a, b in zip(t, (0.1, 0.2, 0.3))):
        raise ManifestError(f"axis test: empty at {t}")
    # facing 90 -> glTF direction (-1, 0, 0): a turn of -90 degrees about Y
    if not (abs(q[1] + math.sin(math.pi / 4)) < 1e-4 and abs(q[3] - math.cos(math.pi / 4)) < 1e-4):
        raise ManifestError(f"axis test: empty turn {q}")
    print("AXIS TEST OK: bounds", lo, hi, "empty", t, q)


# --- seats and measures --------------------------------------------------------------------


def seated_root(seat_point, facing, sit: SitSpec):
    """A seated rig's root: `sit.forward` along the seat's facing from the seat top point, and
    `sit.below_seat` below it (`core.targets.seat_root`, the rule the trace check reads).
    `facing` is the seat's glTF facing vector; only its X and Z count."""
    return tuple(float(v) for v in seat_root(seat_point, facing, sit, "seated_root"))


def tris_of(ob):
    return sum(len(p.vertices) - 2 for p in ob.data.polygons)


def r4(v):
    return round(v + 0.0, 4)


def fill_fraction(root, lo, hi):
    """The fraction of a 24 x 12 grid of rays, front (+Z) to back over the piece's X-Y bounds,
    that hit the piece."""
    scene = bpy.context.scene
    dg = bpy.context.evaluated_depsgraph_get()
    mine = {ch.name for ch in root.children_recursive}
    nx, ny = FILL_GRID
    depth = hi[2] - lo[2] + 2.0
    direction = (root.matrix_world.to_3x3() @ Vector(B(0, 0, -1))).normalized()
    hits = 0
    for i in range(nx):
        X = lo[0] + (i + 0.5) / nx * (hi[0] - lo[0])
        for j in range(ny):
            Y = lo[1] + (j + 0.5) / ny * (hi[1] - lo[1])
            origin = root.matrix_world @ Vector(B(X, Y, hi[2] + 1.0))
            ok, _loc, _n, _i, ob, _m = scene.ray_cast(dg, origin, direction, distance=depth)
            if ok and ob is not None and ob.name in mine:
                hits += 1
    return hits / float(nx * ny)


# --- the pack: register pieces, then finish ------------------------------------------------


def _v3(what: str, v) -> tuple[float, float, float]:
    if len(v) != 3:
        raise ManifestError(f"{what}: must be 3 numbers, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _names(what: str, v) -> tuple[str, ...]:
    """A list of names. A bare string is refused: `"seat_top"` would read as its letters."""
    if isinstance(v, str) or not isinstance(v, Sequence):
        raise ManifestError(f"{what}: must be a list of names, got {v!r}")
    return tuple(v)


def _above_zero(what: str, v) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
        raise ManifestError(f"{what}: must be a finite number above 0, got {v!r}")
    return float(v)


def _end(what: str, e) -> End:
    """One connector end, checked: the point is 3 numbers, the host a `Ref`, the layout a name
    or None, the plug a part name or None. Whether the host and layout agree, and whether the
    plug names a part of the piece, is the loader's check, at `finish`."""
    if not isinstance(e, End):
        raise ManifestError(f"{what}: must be an End(point, layout, host), got {e!r}")
    if not isinstance(e.host, Ref):
        raise ManifestError(f"{what}.host: must be a Ref, got {e.host!r}")
    if e.layout is not None and not isinstance(e.layout, str):
        raise ManifestError(f"{what}.layout: must be a name or None, got {e.layout!r}")
    if e.plug is not None and not isinstance(e.plug, str):
        raise ManifestError(f"{what}.plug: must be a part name or None, got {e.plug!r}")
    return End(_v3(f"{what}.point", e.point), e.layout, e.host, e.plug)


@dataclass
class _Entry:
    root: object
    frame: Frame
    recipe: Recipe
    listed: tuple[Listed, ...]
    done: str
    size_class: str
    meta: dict
    view: View | None = None
    outlined: bool = True
    outlined_why: str = ""
    uses: list[Use] = field(default_factory=list)
    holds: list[dict] = field(default_factory=list)
    seats: list[Seat] = field(default_factory=list)
    mounts: list[Mount] = field(default_factory=list)
    keep_offs: list[KeepOff] = field(default_factory=list)
    connectors: list[Connector] = field(default_factory=list)
    rig: RigSpec | None = None
    forms: dict[str, SoftForm] = field(default_factory=dict)


class PackBuilder:
    """Collects a pack's pieces and writes them with `finish()`.

    Every point and direction given to `use`, `hold`, `seat`, `mount`, `keep_off` and `connector`
    is glTF, in the piece's own frame (its root at the origin), except a connector's layout
    end, whose point is in that layout's frame. `finish` lays the pieces out `spacing` apart
    along X; the records do not move with them. With `spacing=None` the pieces stay where they
    stand (a pack placed in its stage frame): `finish` moves no root and runs no spacing check.
    Either way `finish` writes `spacing` into pack.json.

    `crease_deg` is the style's `dials.edges.crease_deg`: faces that turn more than this get a
    hard edge in `finish`. It is required; a builder without it raises, so no build assumes an
    angle.
    """

    def __init__(self, name: str, units_per_metre: float, out_dir: Path,
                 spacing: float | None, *, crease_deg: float | None = None):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ManifestError(f"pack name must be letters, digits, '_' or '-', got {name!r}")
        for what, v in (("units_per_metre", units_per_metre), ("spacing", spacing)):
            if what == "spacing" and v is None:
                continue
            if not (isinstance(v, int | float) and not isinstance(v, bool)
                    and math.isfinite(v) and v > 0):
                raise ManifestError(f"pack {name!r}: {what} must be above 0, got {v!r}")
        if crease_deg is None:
            raise ManifestError(f"pack {name!r}: no crease_deg; the build needs the style's "
                                "dials.edges.crease_deg and never assumes an angle")
        crease_deg = check_crease_deg(crease_deg, f"pack {name!r}")
        self.name = name
        self.units_per_metre = float(units_per_metre)
        self.out_dir = Path(out_dir)
        self.spacing = None if spacing is None else float(spacing)
        self.crease_deg = float(crease_deg)
        self._entries: dict[str, _Entry] = {}

    def _entry(self, root) -> _Entry:
        entry = self._entries.get(root.name)
        if entry is None or entry.root != root:
            raise ManifestError(f"pack {self.name!r}: piece {root.name!r} is not registered")
        return entry

    def check_registration(self, *, name: str, view: View | None, outlined: bool = True,
                           zones: Sequence[str] = ("d",),
                           hosts: Sequence[str] | None = None) -> None:
        """Raise unless a piece called `name` could be registered with these arguments: the
        checks `register` makes on its own arguments, which need no object. A builder calls it
        before it makes any object, so a refusal leaves the scene as it was (`soft.build_piece`
        and `grow.add_grown` do); `register` calls it again, so a direct caller is covered.

        `view` must be a `View`: a build without one raises, as one without `crease_deg` does.
        `name` must be free in this pack. `outlined` must be true or false; `zones` and `hosts`
        must be lists of names."""
        if not isinstance(view, View):
            raise ManifestError(
                f"pack {self.name!r}: piece {name!r} has no view (got {view!r}); the build "
                "needs the nearest distance the piece is seen from and why, and an agent that "
                "does not know it asks the person, never picks one so a check passes")
        if not isinstance(outlined, bool):
            raise ManifestError(f"pack {self.name!r}: piece {name!r}: outlined must be true "
                                f"or false, got {outlined!r}")
        if name in self._entries:
            raise ManifestError(f"pack {self.name!r}: piece {name!r} is registered twice")
        _names(f"{name}.zones", zones)
        if hosts is not None:
            _names(f"{name}.hosts", hosts)

    def register(self, root, *, origin: str, recipe: Recipe, listed: Sequence[Listed],
                 done: str, size_class: str, kind: str, snap: str = "free",
                 zones: Sequence[str] = ("d",), skin=None, hosts=None, fill: bool = False,
                 tool: bool = False, rig: RigSpec | None = None,
                 forms: Mapping[str, SoftForm] = MappingProxyType({}),
                 view: View | None = None, outlined: bool = True, outlined_why: str = ""):
        """Add a piece. `origin` says in words where the piece frame's origin sits (the
        manifest's `frame.origin`). `kind`, `snap`, `zones`, `skin`, `hosts` and `fill` go to
        `measured` as before; `fill` asks for the measured fill fraction.

        `tool` marks a held tool (a drumstick, a glowstick). `finish` then writes
        `measured.tip_length`: how far the piece's vertices reach from its origin along the
        tool axis, `targets.TIP_AXIS` (the piece's +Z), in the piece's own units. A tool with no
        vertex past its origin along that axis has no tip, and `finish` raises.

        `view` is the piece's `View` (the nearest distance it is seen from, in real metres, and
        why). It is required: a build without it raises, as one without `crease_deg` does.

        `outlined` says whether the piece's shipping look draws outlines (the default). A piece
        that does not needs an `outlined_why`; the loader checks the pair at `finish`.

        `rig` makes the piece rigged: `finish` writes it into the `Piece`, takes each part's
        `bones` and `bind` from `rig.bind`, and needs the armature object `rig.armature` under
        the root (see the module doc). `forms` maps part names to their soft `form`, which
        `finish` writes onto the parts; a name the piece has no part for raises there."""
        self.check_registration(name=root.name, zones=zones, hosts=hosts, view=view,
                                outlined=outlined)
        if root.parent is not None:
            raise ManifestError(f"pack {self.name!r}: piece {root.name!r} has a parent; a "
                                "piece root stands alone")
        self._entries[root.name] = _Entry(
            root=root, frame=Frame("gltf", origin), recipe=recipe, listed=tuple(listed),
            done=done, size_class=size_class,
            meta={"kind": kind, "snap": snap, "zones": list(_names(f"{root.name}.zones", zones)),
                  "skin": skin,
                  "hosts": None if hosts is None else list(_names(f"{root.name}.hosts", hosts)),
                  "fill": fill, "tool": tool},
            rig=rig, forms={k: v for k, v in forms.items() if v is not None}, view=view,
            outlined=outlined, outlined_why=outlined_why,
        )
        return root

    def stand(self, root, loc, facing):
        """A stand point (glTF loc, compass facing), measured into `measured.stands`."""
        return empty("Stand", root, loc, {"kind": "stand"}, facing, "SINGLE_ARROW")

    def use(self, root, id: str, *, point, facing, posture: str) -> Use:
        """A use station: where a user stands or sits to use the piece, and their facing."""
        rec = Use(id, _v3(f"{id}.point", point), _v3(f"{id}.facing", facing), posture)
        self._entry(root).uses.append(rec)
        return rec

    def hold(self, root, id: str, *, use: str, side: str, surface: Sequence[str], point,
             grip: bool, off=None, sub_box=None) -> None:
        """A hold on the parts named in `surface`. With no `off`, finish() takes it from the
        normal of the surface point nearest `point`."""
        self._entry(root).holds.append({
            "id": id, "use": use, "side": side, "surface": _names(f"{id}.surface", surface),
            "sub_box": None if sub_box is None else (_v3(f"{id}.sub_box", sub_box[0]),
                                                     _v3(f"{id}.sub_box", sub_box[1])),
            "point": _v3(f"{id}.point", point),
            "off": None if off is None else _v3(f"{id}.off", off), "grip": grip,
        })

    def seat(self, root, id: str, *, use: str, point, facing, surface: Sequence[str]) -> Seat:
        """A seat top: the point is on the seat, not a rig root (see `seated_root`)."""
        rec = Seat(id, use, _v3(f"{id}.point", point), _v3(f"{id}.facing", facing),
                   _names(f"{id}.surface", surface))
        self._entry(root).seats.append(rec)
        return rec

    def mount(self, root, id: str, *, kind: str, point, axis, takes: str,
              fits: Sequence[str] | None = None, part: str | None = None) -> Mount:
        """A mount on this piece. `takes` is the guest's part it takes; `part` is this piece's
        own part the mount is cut in, which the joint checks judge a guest against (the loader
        refuses a name this piece lacks, at `finish`)."""
        if part is not None and not isinstance(part, str):
            raise ManifestError(f"{id}.part: must be a part name or None, got {part!r}")
        rec = Mount(id, kind, _v3(f"{id}.point", point), _v3(f"{id}.axis", axis), takes,
                    None if fits is None else _names(f"{id}.fits", fits), part)
        self._entry(root).mounts.append(rec)
        return rec

    def keep_off(self, root, id: str, *, point, radius: float, why: str) -> KeepOff:
        """A point a mount must stay `radius` clear of (a coupler, a brace node). The keep-off
        check judges it. `radius` is in model units and above 0."""
        rec = KeepOff(id, _v3(f"{id}.point", point), _above_zero(f"{id}.radius", radius), why)
        self._entry(root).keep_offs.append(rec)
        return rec

    def connector(self, root, id: str, *, part: str, ends: Sequence[End], cap: float,
                  why: str) -> Connector:
        """A cable between two mounts. `part` is a part of this piece (shape `cable`; `finish`
        raises if the piece has no such part), `ends` are two `End(point, layout, host, plug)`
        records, and `cap` is the longest the cable may be, in model units, above 0. An end's
        `host` is a `Ref`. A self end has `layout=None`, its point in this piece's frame, and
        its host `Ref.parse("self.<mount>", self_ok=True)`, a mount of this piece; it has no
        `plug`. A layout end names its layout, its point is in that layout's frame, and its
        host's piece slot is a placement id in that layout. Its `plug` is the part of this piece
        that seats in that mount (the joint checks read an end with no `plug` `Unmeasured`)."""
        if not isinstance(part, str):
            raise ManifestError(f"{id}.part: must be a part name, got {part!r}")
        if isinstance(ends, str) or not isinstance(ends, Sequence) or len(ends) != 2:
            raise ManifestError(f"{id}.ends: must be two End records, got {ends!r}")
        rec = Connector(id, part, (_end(f"{id}.ends[0]", ends[0]), _end(f"{id}.ends[1]", ends[1])),
                        _above_zero(f"{id}.cap", cap), why)
        self._entry(root).connectors.append(rec)
        return rec

    def finish(self, order: Sequence[str], gate: Callable[[Pack], None] | None = None, *,
               glb_options: Mapping[str, Any] | None = None,
               before_export: Callable[[], None] | None = None) -> Pack:
        """Apply modifiers, lay the pieces out (unless `spacing` is None), measure, and write
        the pack into `out_dir`.

        `glb_options` are extra keywords for the glTF exporter (Draco, for one): `export_glb`
        gets them as they are, and every piece records the position precision they ship
        (`measured.export.position_bits`, see `position_bits`). `before_export` runs once,
        right after the modifiers are applied and before anything is measured or written: it is
        the place for work that `_apply_and_shade` would undo, such as custom normals.

        Before `gate(pack)` runs, the records pass the loader's own check (`writer.pack_text`)
        and the layout is checked (unless `spacing` is None): `spacing` must keep neighbouring
        pieces apart. The gate runs before any file is written; a gate that raises stops the
        export. The .glb and .blend are written to staging names first, then the pack.json, then
        both are moved into place, so a failed export never leaves a new pack.json beside an
        old .glb.
        """
        if len(set(order)) != len(order) or sorted(order) != sorted(self._entries):
            raise ManifestError(
                f"pack {self.name!r}: order {list(order)} must name each registered piece once; "
                f"registered: {sorted(self._entries)}")
        bits = position_bits(glb_options)  # a bad option raises before the scene is changed
        scene = bpy.context.scene
        _apply_and_shade(scene, self.crease_deg)
        if before_export is not None:
            before_export()
        if self.spacing is not None:
            for i, name in enumerate(order):
                self._entries[name].root.location = (i * self.spacing, 0, 0)
        bpy.context.view_layer.update()

        pieces = {name: self._measure(name, self._entries[name], bits) for name in order}
        if self.spacing is not None:
            self._check_spacing(order, pieces)
        pack = Pack(name=self.name, units_per_metre=self.units_per_metre, pieces=pieces,
                    spacing=self.spacing)
        out = self.out_dir
        final = {ext: out / f"{self.name}.{ext}" for ext in ("pack.json", "glb", "blend")}
        pack_text(pack, final["pack.json"])
        if gate is not None:
            gate(pack)

        for name in order:
            for ob in self._entries[name].root.children_recursive:
                if "build_order" in ob:
                    del ob["build_order"]
        out.mkdir(parents=True, exist_ok=True)
        staged = {ext: out / f".staging-{self.name}.{ext}" for ext in ("glb", "blend")}
        leftovers = [*staged.values(), out / f".staging-{self.name}.blend1"]
        for path in leftovers:
            path.unlink(missing_ok=True)
        try:
            export_glb(staged["glb"], **(glb_options or {}))
            bpy.ops.wm.save_as_mainfile(filepath=str(staged["blend"]), copy=True)
            write_pack(pack, final["pack.json"])
            os.replace(staged["glb"], final["glb"])
            os.replace(staged["blend"], final["blend"])
        finally:
            for path in leftovers:
                path.unlink(missing_ok=True)
        (out / f"{self.name}.blend1").unlink(missing_ok=True)  # no stale .blend1 backup
        print("PACK OK", {n: p.measured["tris"] for n, p in pieces.items()})
        return pack

    def _check_spacing(self, order: Sequence[str], pieces: dict[str, Piece]) -> None:
        """Piece i spans X from i*spacing + X0 to i*spacing + X1; neighbours may not meet."""
        for a, b in pairwise(order):
            need = pieces[a].measured["bounds"]["X1"] - pieces[b].measured["bounds"]["X0"]
            if self.spacing <= need:
                raise ManifestError(
                    f"pack {self.name!r}: spacing {self.spacing} lays {a!r} and {b!r} into each "
                    f"other; it must be above {need} (the first's X1 minus the next one's X0)")

    def _measure(self, name: str, entry: _Entry, bits: int | None) -> Piece:
        root, meta = entry.root, entry.meta
        inv = root.matrix_world.inverted()
        lo, hi = [math.inf] * 3, [-math.inf] * 3
        tip = -math.inf  # the furthest any vertex lies along TIP_AXIS
        slot_names, tris = [], 0
        surfaces, stands, pivots, meshes, armatures = [], [], [], [], []
        for ob in root.children_recursive:
            if ob.type == "MESH":
                meshes.append(ob)
            elif ob.type == "ARMATURE":
                armatures.append(ob)
            elif ob.type == "EMPTY":
                kind = ob.get("kind")
                if kind == "seat":
                    raise ManifestError(
                        f"piece {name!r}: empty {ob.name!r} is an old seat empty; a seat is a "
                        "record now: call PackBuilder.seat()")
                if kind not in MEASURED_EMPTY_KINDS:
                    raise ManifestError(
                        f"piece {name!r}: empty {ob.name!r} has kind {kind!r}; an empty under a "
                        f"piece must be one of {list(MEASURED_EMPTY_KINDS)}")
                loc = G3(inv @ ob.matrix_world.translation)
                if kind == "surface":
                    surfaces.append({"name": str(ob["name"]), "w": r4(ob["w"]), "d": r4(ob["d"]),
                                     "face": str(ob["face"]),
                                     "x": r4(loc[0]), "y": r4(loc[1]), "z": r4(loc[2])})
                elif kind == "pivot":
                    pivots.append({"name": _pivot_name(name, ob), "x": r4(loc[0]),
                                   "y": r4(loc[1]), "z": r4(loc[2])})
                else:
                    f = round(-math.degrees((inv @ ob.matrix_world).to_euler().z)) % 360
                    stands.append({"x": r4(loc[0]), "y": r4(loc[1]), "z": r4(loc[2]),
                                   "facing": f})
            else:
                raise ManifestError(f"piece {name!r}: {ob.name!r} is a {ob.type}; a piece holds "
                                    "only mesh parts, an armature (a rigged piece) and "
                                    "surface, stand or pivot empties")
        armature = _check_armature(name, entry.rig, armatures)
        if not meshes:
            raise ManifestError(f"piece {name!r}: no mesh parts under its root")
        for ob in meshes:
            if "build_order" not in ob:
                raise ManifestError(f"piece {name!r}: part {ob.name!r} has no build order; make "
                                    "parts with mesh_obj (or slab, lathe, extrude_profile) "
                                    "or tag")
        meshes.sort(key=lambda ob: ob["build_order"])
        parts = _rig_parts(name, entry, armature, [_part(name, ob) for ob in meshes], meshes)
        by_part = dict(zip((p.name for p in parts), meshes, strict=True))
        for ob in meshes:
            tris += tris_of(ob)
            for m in ob.data.materials:
                if m and m.name not in slot_names:
                    slot_names.append(m.name)
            mw = inv @ ob.matrix_world
            for v in ob.data.vertices:
                g = G3(mw @ v.co)
                for a in range(3):
                    lo[a], hi[a] = min(lo[a], g[a]), max(hi[a], g[a])
                tip = max(tip, float(np.dot(g, TIP_AXIS)))
        measured = {
            "kind": meta["kind"], "zones": meta["zones"], "snap": meta["snap"],
            "foot": {"w": r4(hi[0] - lo[0]), "d": r4(hi[2] - lo[2])},
            "bounds": {"X0": r4(lo[0]), "X1": r4(hi[0]), "Z0": r4(lo[2]), "Z1": r4(hi[2])},
            "heights": {"top": r4(hi[1])},
            "surfaces": surfaces, "slots": sorted(slot_names), "tris": tris,
            "export": {"position_bits": bits},
        }
        # optional keys, written only when set
        if meta["skin"] is not None:
            measured["skin"] = meta["skin"]
        if meta["hosts"] is not None:
            measured["hosts"] = meta["hosts"]
        if stands:
            measured["stands"] = stands
        if pivots:
            names = [p["name"] for p in pivots]
            if len(set(names)) != len(names):
                raise ManifestError(f"piece {name!r}: two pivots are named alike: "
                                    f"{sorted(n for n in set(names) if names.count(n) > 1)}")
            measured["pivots"] = sorted(pivots, key=lambda p: p["name"])
        if meta["fill"]:
            measured["fill"] = r4(fill_fraction(root, lo, hi))
        if meta["tool"]:
            if tip <= 0:
                raise ManifestError(
                    f"piece {name!r}: a tool needs a tip, but no vertex lies past its origin "
                    f"along the tool axis {TIP_AXIS.tolist()} (furthest is {tip:.4f})")
            measured["tip_length"] = r4(tip)
        holds = tuple(
            Hold(id=h["id"], use=h["use"], side=h["side"], surface=h["surface"],
                 sub_box=h["sub_box"], point=h["point"],
                 off=h["off"] if h["off"] is not None else _surface_off(name, root, by_part, h),
                 grip=h["grip"])
            for h in entry.holds
        )
        for conn in entry.connectors:
            if conn.part not in by_part:
                raise ManifestError(
                    f"piece {name!r}: connector {conn.id!r} names part {conn.part!r}, which the "
                    f"piece does not have; its parts are {sorted(by_part)}")
        return Piece(
            name=name, frame=entry.frame, recipe=entry.recipe, listed=entry.listed,
            parts=tuple(parts), uses=tuple(entry.uses), holds=holds, seats=tuple(entry.seats),
            mounts=tuple(entry.mounts), done=entry.done, size_class=entry.size_class,
            measured=measured, rig=entry.rig, connectors=tuple(entry.connectors),
            keep_off=tuple(entry.keep_offs), view=entry.view,
            outlined=entry.outlined, outlined_why=entry.outlined_why,
        )


def _pivot_name(piece_name: str, ob) -> str:
    """A pivot's name: its `name` property. The object name is no stand-in, because Blender
    may add a `.001` to it."""
    name = ob.get("name")
    if not isinstance(name, str) or not name:
        raise ManifestError(f"piece {piece_name!r}: pivot {ob.name!r} needs a 'name' property "
                            f"(a non-empty string), got {name!r}")
    return name


def _check_armature(piece_name: str, rig: RigSpec | None, armatures: list):
    """The one armature of a rigged piece, or None for an unrigged piece. An armature under an
    unrigged piece, a second armature, a missing one or one not named `rig.armature` raises."""
    names = sorted(a.name for a in armatures)
    if rig is None:
        if armatures:
            raise ManifestError(f"piece {piece_name!r}: armature {names[0]!r} is under a piece "
                                "that is not rigged; register it with rig=")
        return None
    if len(armatures) > 1:
        raise ManifestError(f"piece {piece_name!r}: a rigged piece holds one armature, found "
                            f"{names}")
    if not armatures:
        raise ManifestError(f"piece {piece_name!r}: its rig names the armature {rig.armature!r}, "
                            "but no armature is under the root")
    if armatures[0].name != rig.armature:
        raise ManifestError(f"piece {piece_name!r}: the armature is {armatures[0].name!r}, but "
                            f"its rig names {rig.armature!r}")
    return armatures[0]


def _rig_parts(piece_name: str, entry: _Entry, armature, parts: list[Part], meshes: list
               ) -> list[Part]:
    """`parts` with each part's `bones` and `bind` from `rig.bind`, as `load_pack` fills them,
    and its `form` from the forms given to `register`. A form for a part the piece lacks, a
    part with no bind (or a bind for no part) and a part with no Armature modifier to the rig's
    armature raise."""
    names = [p.name for p in parts]
    ghosts = sorted(set(entry.forms) - set(names))
    if ghosts:
        raise ManifestError(f"piece {piece_name!r}: forms are given for {ghosts}, which are no "
                            f"parts; its parts are {names}")
    rig = entry.rig
    if rig is None:
        return [replace(p, form=entry.forms.get(p.name)) for p in parts]
    loose = sorted(set(rig.bind) ^ set(names))
    if loose:
        raise ManifestError(f"piece {piece_name!r}: rig.bind and the parts disagree on {loose}")
    unskinned = [ob.name for ob in meshes
                 if not any(m.type == "ARMATURE" and m.object == armature for m in ob.modifiers)]
    if unskinned:
        raise ManifestError(f"piece {piece_name!r}: parts {unskinned} have no Armature modifier "
                            f"to {armature.name!r}")
    return [replace(p, form=entry.forms.get(p.name), bones=rig.bind[p.name].bones,
                    bind=rig.bind[p.name].mode) for p in parts]


# Modifier settings that only describe the editor, never the result.
_MODIFIER_UI_ONLY = frozenset({"rna_type", "is_active", "show_expanded", "show_in_editmode",
                               "show_on_cage", "execution_time", "persistent_uid"})


def _stack(ob) -> list:
    """What the non-Armature modifiers of `ob` do: each one's type, name and settings (an object
    or texture it points to by name). Equal stacks on one mesh only make a pair of objects
    candidates to share a result: a modifier may read where its object stands, so
    `_apply_and_shade` still compares the evaluated geometry. A list, because settings such as
    flag sets cannot be hashed."""
    stack = []
    for mod in ob.modifiers:
        if mod.type == "ARMATURE":
            continue
        settings = []
        for prop in mod.bl_rna.properties:
            if prop.identifier in _MODIFIER_UI_ONLY or prop.type == "COLLECTION":
                continue
            value = getattr(mod, prop.identifier)
            if prop.type == "POINTER":
                value = getattr(value, "name", None)
            elif getattr(prop, "array_length", 0):
                value = [list(v) if hasattr(v, "__len__") else v for v in value]
            settings.append((prop.identifier, value))
        stack.append((mod.type, mod.name, settings))
    return stack


def _instance_groups(meshes: list) -> list[list]:
    """The objects of `meshes` that may share one applied mesh: the same mesh, an equal modifier
    stack and the same shading marks."""
    keyed: list[tuple[tuple, list]] = []
    for ob in meshes:
        key = (ob.data, _stack(ob), ob.get("smooth"), ob.get("shade"))
        for held, users in keyed:
            if held == key:
                users.append(ob)
                break
        else:
            keyed.append((key, [ob]))
    return [users for _, users in keyed]


# How an attribute's values are read in bulk: the item property, the numbers per item, the dtype.
_ATTR_READ = {
    "FLOAT": ("value", 1, np.float32), "INT": ("value", 1, np.int32),
    "INT8": ("value", 1, np.int32), "BOOLEAN": ("value", 1, np.int8),
    "FLOAT_VECTOR": ("vector", 3, np.float32), "FLOAT2": ("vector", 2, np.float32),
    "FLOAT_COLOR": ("color", 4, np.float32), "BYTE_COLOR": ("color", 4, np.float32),
    "QUATERNION": ("value", 4, np.float32), "FLOAT4X4": ("value", 16, np.float32),
    "INT32_2D": ("value", 2, np.int32),
}


def _layer(attr):
    """One attribute's values as an array, or None for a kind not read here."""
    kind = _ATTR_READ.get(attr.data_type)
    if kind is None:
        return None
    prop, width, dtype = kind
    arr = np.empty(len(attr.data) * width, dtype)
    attr.data.foreach_get(prop, arr)
    return arr


def _same_geometry(a, b) -> bool:
    """Whether two evaluated meshes hold everything the export reads: the same vertex places and
    faces, every attribute layer (UV maps, colours, material indices, sharp flags, custom normals;
    a modifier such as UV Project writes these from where its object stands) and the same vertex
    group weights. A layer of a kind not read here counts as different, so the pair stays apart."""
    if (len(a.vertices), len(a.loops), len(a.polygons)) != (
            len(b.vertices), len(b.loops), len(b.polygons)):
        return False
    starts = []
    for me in (a, b):
        arr = np.empty(len(me.polygons), np.int32)
        me.polygons.foreach_get("loop_start", arr)
        starts.append(arr)
    if not np.array_equal(*starts):
        return False
    # Editor selection flags are not part of the result.
    layers = [{at.name: at for at in me.attributes if not at.name.startswith(".select")}
              for me in (a, b)]
    if layers[0].keys() != layers[1].keys():
        return False
    for name, left in layers[0].items():
        right = layers[1][name]
        if (left.domain, left.data_type) != (right.domain, right.data_type):
            return False
        got = (_layer(left), _layer(right))
        if got[0] is None or got[1] is None or not np.array_equal(*got):
            return False
    weights = [[[(g.group, g.weight) for g in v.groups] for v in me.vertices] for me in (a, b)]
    return weights[0] == weights[1]


def _apply_and_shade(scene, crease_deg: float) -> None:
    """Apply every mesh's modifiers except Armature, then shade smooth with a crease wherever
    faces turn more than `crease_deg` degrees (meshes made `smooth`), or flat. A mesh marked
    `shade = "keep"` (mixed flags from `tag`) keeps the shading it has.

    Objects that share one mesh and carry equal modifier stacks (instances) are evaluated, and
    those whose result is the same geometry are applied as one mesh, so they stay one mesh in the
    export. An instance whose stack differs, or whose modifiers read its place (a Mirror about an
    object, a Boolean, a Curve, a UV Project) and so give another result in any layer the export
    reads, gets a mesh of its own. No object's geometry changes.

    The evaluation runs with the Armature modifiers switched off, so the pose the frame holds is
    never baked into the mesh. The vertex groups come through, and the Armature modifiers stay
    on their objects, switched on again."""
    meshes = [o for o in scene.objects if o.type == "MESH"]
    armature_mods = [m for ob in meshes for m in ob.modifiers
                     if m.type == "ARMATURE" and m.show_viewport]
    for mod in armature_mods:
        mod.show_viewport = False
    candidates = _instance_groups(meshes)
    merged: list[tuple[list, Any]] = []  # (objects, their one evaluated mesh)
    try:
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        for cand in candidates:
            kept: list[tuple[list, Any]] = []
            for ob in cand:
                me = bpy.data.meshes.new_from_object(
                    ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
                for users, held in kept:
                    if _same_geometry(held, me):
                        users.append(ob)
                        bpy.data.meshes.remove(me)
                        break
                else:
                    kept.append(([ob], me))
            merged.extend(kept)
    finally:
        for mod in armature_mods:
            mod.show_viewport = True
    names = []
    for users, me in merged:
        olds = {ob.data for ob in users}
        names.append(users[0].data.name)
        for ob in users:
            for mod in [m for m in ob.modifiers if m.type != "ARMATURE"]:
                ob.modifiers.remove(mod)
            ob.data = me
        for old in olds:
            if old.users == 0:  # a mesh another object still uses stays
                bpy.data.meshes.remove(old)
        if users[0].get("shade") != "keep":
            _shade(me, bool(users[0].get("smooth")), crease_deg)
        for ob in users:
            for key in ("smooth", "shade"):
                if key in ob:
                    del ob[key]
    for (_, me), name in zip(merged, names, strict=True):
        me.name = name  # the new mesh took a `.001` name while the old one was there
    bpy.context.view_layer.update()


def _shade(me, smooth: bool, crease_deg: float) -> None:
    """Smooth with an edge crease wherever faces turn more than `crease_deg` degrees, or flat."""
    crease = math.radians(crease_deg)
    bm = bmesh.new()
    bm.from_mesh(me)
    if smooth:
        for e in bm.edges:
            if len(e.link_faces) == 2:
                e.smooth = e.calc_face_angle(0.0) < crease
    else:
        for f in bm.faces:
            f.smooth = False
    bm.to_mesh(me)
    bm.free()


def _part(piece_name: str, ob) -> Part:
    def prop(key: str):
        value = ob.get(key)
        if key in _FLAG_PROPS:
            if value is not None and not isinstance(value, bool):
                raise ManifestError(f"piece {piece_name!r}: part {ob.name!r}: {key!r} must be "
                                    f"true or false, got {value!r}")
            return bool(value)
        if key in _LIST_PROPS:
            if value is None:
                return ()
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise ManifestError(f"piece {piece_name!r}: part {ob.name!r}: {key!r} must be a "
                                    f"list of part names, got {value!r}")
            return tuple(value)
        if value is None:
            if key in REQUIRED_PART_PROPS:
                raise ManifestError(f"piece {piece_name!r}: part {ob.name!r} has no {key!r} "
                                    "property")
            return None
        if not isinstance(value, str):
            raise ManifestError(f"piece {piece_name!r}: part {ob.name!r}: {key!r} must be a "
                                f"string, got {value!r}")
        return value

    prefix = object_name(piece_name, "")
    if not ob.name.startswith(prefix):
        raise ManifestError(f"piece {piece_name!r}: part object {ob.name!r} must be named "
                            f"{prefix}<part>; make parts with mesh_obj or tag")
    try:
        read = {key: prop(key) for key in PART_PROPS}
        read["role"] = read["role"] or DEFAULT_ROLE
        return Part(name=ob.name[len(prefix):], bones=(), bind=None, **read)
    except RecordError as exc:  # the part refused itself: say which piece it is in
        raise ManifestError(f"piece {piece_name!r}: {exc}") from exc


def _surface_off(piece_name: str, root, meshes: dict, hold: dict) -> tuple[float, float, float]:
    """The way off the surface at a hold: the normal at the surface point nearest the hold's
    point, over the hold's surface parts, in the piece's glTF frame."""
    inv_root = root.matrix_world.inverted()
    world = root.matrix_world @ Vector(B(*hold["point"]))
    best = None
    for part in hold["surface"]:
        ob = meshes.get(part)
        if ob is None:
            raise ManifestError(f"piece {piece_name!r}: hold {hold['id']!r}: surface names no "
                                f"part: {part!r}")
        ok, loc, normal, _ = ob.closest_point_on_mesh(ob.matrix_world.inverted() @ world)
        if not ok:
            raise ManifestError(f"piece {piece_name!r}: hold {hold['id']!r}: no surface point "
                                f"found on {part!r}")
        to_root = inv_root @ ob.matrix_world
        dist = ((to_root @ loc) - (inv_root @ world)).length
        n = (to_root.to_3x3().inverted().transposed() @ normal).normalized()
        if best is None or dist < best[0]:
            best = (dist, n)
    g = G3(best[1])
    return (round(g[0], 6) + 0.0, round(g[1], 6) + 0.0, round(g[2], 6) + 0.0)
