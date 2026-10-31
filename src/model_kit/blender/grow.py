"""The grower's Blender half: a saved skeleton becomes the objects of one branching piece.

Growing never runs here. A tree is grown in the system Python and saved
(`core.grow.skeleton.save_skeleton`); this module reads the triangles `core.grow.mesh` makes from
it. `add_grown` makes one object per level (a tube part) and one per leaf shade (a leaf part),
registers them as a `branching` piece, and hands back a callback that writes the leaf normals,
which lean to the crown. Pass it to `PackBuilder.finish(before_export=...)`: `finish` rebuilds
every mesh while it applies modifiers, so normals set earlier would be lost.

The parts are named like the core's: `grown_parts` gives the level names, then
`Leaves_<shade>`. Axes: the core's arrays are glTF, and this module turns them into Blender's
with `axes.BLENDER_TO_GLTF`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import bmesh
import bpy
import numpy as np

from model_kit.blender import build
from model_kit.blender.axes import BLENDER_TO_GLTF
from model_kit.core.errors import ManifestError
from model_kit.core.grow.buds import LITE_ONLY, V1_LITE_SWAPS
from model_kit.core.grow.mesh import LeafLean, crown_normals, grown_parts, leaf_part_name
from model_kit.core.grow.rules import GrowthRules
from model_kit.core.grow.skeleton import SHOOT, TWIG, Skeleton
from model_kit.core.manifest import SIZE_CLASSES, Listed, Recipe, View
from model_kit.core.sight import Sight, view_floor

WELD_DECIMALS = 6  # corners equal to this many decimals (model units) are one vertex
LEAF_LEVEL = "leaves"  # the rules rows that say how leaves look
TUBE, LEAF = "tube", "leaf"
SOLID, SKIP_OUTLINE = "solid", "skip_outline"


def _to_blender(points: np.ndarray) -> np.ndarray:
    """glTF points (..., 3) as Blender points: a glTF (X, Y, Z) is Blender (X, -Z, Y)."""
    return np.asarray(points, dtype=float) @ BLENDER_TO_GLTF[:3, :3]


def mesh_from_tris(name: str, tris: np.ndarray, mat, parent, *, smooth: bool,
                   props: Mapping[str, Any] | None = None):
    """A mesh object from (N, 3, 3) glTF triangles, one face per triangle in the given order.

    Equal corners are welded (to `WELD_DECIMALS` decimals; the first copy's position is kept), so
    the faces share vertices and `smooth` can round them. The object is made with
    `build.mesh_obj`, so under a piece it is the part `<piece>/<name>` with the next build order;
    `props` go to `mesh_obj` as they are (give `listed`, `group` and `shape` to make it a part).
    """
    tris = np.asarray(tris, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3) or not len(tris):
        raise ManifestError(f"mesh_from_tris {name!r}: need (N, 3, 3) triangles with N > 0, "
                            f"got shape {tris.shape}")
    corners = tris.reshape(-1, 3)
    _, first, inverse = np.unique(np.round(corners, WELD_DECIMALS), axis=0, return_index=True,
                                  return_inverse=True)
    points = _to_blender(corners[first])
    faces = np.asarray(inverse).reshape(-1, 3)
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in points.tolist()]
    for i, face in enumerate(faces.tolist()):
        try:
            bm.faces.new([verts[v] for v in face])
        except ValueError as exc:  # a repeated corner or a face that exists: no silent drop
            bm.free()
            raise ManifestError(f"mesh_from_tris {name!r}: triangle {i} cannot be a face "
                                f"({exc})") from exc
    return build.mesh_obj(name, bm, mat, parent, smooth=smooth, props=props)


# The params a lite grow reads in place of a full one are `buds.V1_LITE_SWAPS` (in a buds-v1 rules
# file only) and `LITE_ONLY`.
LITE_LEVEL = "lite"


def used_params(rules: GrowthRules, lite: bool) -> set[str]:
    """The keys of the params a grow in this mode reads: all but the other mode's swaps."""
    out = set(rules.params)
    if lite:
        return out - set(V1_LITE_SWAPS)
    return out - set(V1_LITE_SWAPS.values()) - set(LITE_ONLY)


def _level_rows(rules: GrowthRules, part: str, leaf_shade: str | None, lite: bool):
    """The rules rows behind a part, in file order: those whose level is a word of its name
    (`Trunk_Roots` is `trunk` and `roots`), or `leaves` for a leaf part, that a param used in
    this mode cites. A lite tree also cites the `lite` rows: its swapped params cite them. None
    found raises."""
    words = {LEAF_LEVEL} if leaf_shade is not None else set(part.lower().split("_"))
    if lite:
        words.add(LITE_LEVEL)
    cited = {row for key in used_params(rules, lite) for row in rules.params[key].rows}
    rows = [row for row in rules.rows.values() if row.level in words and row.id in cited]
    if not rows:
        raise ManifestError(f"{rules.path}: no rows of level {sorted(words)} for part {part!r}")
    return rows


def _listed(rules: GrowthRules, part: str, leaf_shade: str | None, lite: bool,
            min_thickness: float) -> Listed:
    rows = _level_rows(rules, part, leaf_shade, lite)
    what = f"leaves, shade {leaf_shade}" if leaf_shade is not None else part.replace("_", " and ")
    text = f"{what}: " + "; ".join(f"{row.rule}: {row.value}" for row in rows)
    source = f"{rules.path.name}: " + ", ".join(row.id for row in rows)
    return Listed(part.lower(), text, source, min_thickness, None)


def _leaf_host_level(sk: Skeleton, shade: int, grown: Mapping[str, np.ndarray]) -> str:
    """The name of the level the leaves of shade index `shade` sit on: the level of their host
    limbs. When the hosts span levels, it is the nearest level they all descend from (the
    lowest-numbered one, since a limb's parent is of a lower level). `grown` holds the grown
    parts; a shade with no leaf, or a level that grew no part, raises."""
    hosts = sk.host[sk.shade == shade]
    if not len(hosts):
        raise ManifestError(f"leaf shade {sk.shades[shade]!r} has no leaves, so no host level")
    level = sk.level_names[int(sk.level[hosts].min())]
    if level not in grown:
        raise ManifestError(f"leaf shade {sk.shades[shade]!r} sits on level {level!r}, which "
                            "grew no part")
    return level


def _parts_wanted(sk: Skeleton, parts: Mapping[str, np.ndarray]) -> list[tuple]:
    """For each grown part, in order: (name, shape, on, role, leaf shade or None, level or None)."""
    names = sk.level_names
    skip = {names[i] for i in (TWIG, SHOOT) if i < len(names)}
    grown = [name for name in names if name in parts]
    if not grown:
        raise ManifestError("the skeleton grows no limb level")
    wanted, previous = [], None
    for li, name in enumerate(names):
        if name in parts:
            role = SKIP_OUTLINE if name in skip else SOLID
            wanted.append((name, TUBE, previous, role, None, li))
            previous = name
    shades = {leaf_part_name(shade): si for si, shade in enumerate(sk.shades)}
    for name in parts:
        if name in shades:
            wanted.append((name, LEAF, _leaf_host_level(sk, shades[name], parts), SKIP_OUTLINE,
                           sk.shades[shades[name]], None))
    unknown = sorted(set(parts) - {w[0] for w in wanted})
    if unknown:
        raise ManifestError(f"grown parts {unknown} are neither levels nor leaf shades")
    return wanted


def _check_units(builder: build.PackBuilder, sk: Skeleton) -> None:
    """The builder's units per metre must be the skeleton's. The grower records the model-unit
    height `H` = `height_m` x units per metre, so the skeleton's own units follow from it."""
    info = sk.info
    for key in ("H", "height_m"):
        if not isinstance(info.get(key), int | float) or not info[key] > 0:
            raise ManifestError(f"skeleton info {key!r} must be a number above 0, got "
                                f"{info.get(key)!r}")
    grown_at = info["H"] / info["height_m"]
    if not math.isclose(grown_at, builder.units_per_metre, rel_tol=1e-9):
        raise ManifestError(f"piece {info['piece']!r} was grown at {grown_at} units per metre "
                            f"but pack {builder.name!r} is {builder.units_per_metre}")


def _check_names(info: Mapping[str, Any], parts) -> None:
    """The piece root and its part objects must be free names: Blender would rename a clash."""
    taken = [n for n in (info["piece"], *(build.object_name(info["piece"], p) for p in parts))
             if n in bpy.data.objects]
    if taken:
        raise ManifestError(f"piece {info['piece']!r}: object names already taken: {taken}")


def add_grown(builder: build.PackBuilder, sk: Skeleton, *, rules: GrowthRules,
              materials: Mapping[str, Any], done: str, size_class: str, origin: str,
              lean: LeafLean, view: View | None = None) -> tuple[Any, Callable[[], None]]:
    """Make a saved skeleton's piece: one object per entry of `grown_parts(sk, sight)`, registered
    with `builder` as a `branching` piece named `sk.info["piece"]`.

    `rules` is the rules file the tree was grown from (`sk.info["rules"]` is its file name; a
    different file raises). Each part's listed text and source come from the rows of the params
    that grow used: a lite tree cites the `lite` rows in place of the full-mode rows its swapped
    params replace. `view` is the piece's `View`; it is required and a grown piece without one
    raises; a buds-v2 tree's tube sides come from it (`grown_parts` takes the view's seen floor and
    the pack's crease angle as the piece's sight). `materials` maps each part name (`Twigs`,
    `Leaves_Leaf`) to its Blender material; a part with none raises. The builder's units per
    metre must be the ones the tree was grown at. Returns the piece root and a callback that sets
    each leaf part's custom normals from `crown_normals`, leaning them by `lean` (a render dial
    from the style; `PackBuilder` holds no style, so the caller passes it). Pass the callback to
    `PackBuilder.finish(before_export=...)`.

    Parts: shape `tube` for levels and `leaf` for leaf parts; each level is `on` the level before
    it and every leaf part `on` the level its leaves' host limbs are of (the lowest, when they span
    levels); `group` is the part name; `role` is
    `skip_outline` for the twig and shoot levels and every leaf part, else `solid`. A level's
    listed `min_thickness` is twice its smallest radius in metres; a leaf's is 0.

    A refusal leaves nothing behind. Everything that can be checked from the arguments (info,
    rules, units, materials, names, size class, view, the rows and the crown normals) is read
    before the first object is made; if making the objects or registering them still fails, the
    objects made so far are removed and the error is raised.
    """
    info = sk.info
    for key in ("piece", "rules", "seed", "height_m", "H"):
        if key not in info:
            raise ManifestError(f"skeleton info has no {key!r}; got {sorted(info)}")
    builder.check_registration(name=info["piece"], view=view)
    if rules.path.name != info["rules"]:
        raise ManifestError(f"skeleton was grown from {info['rules']!r}, not {rules.path.name!r}")
    _check_units(builder, sk)
    sight = Sight(view_floor(view, builder.units_per_metre), builder.crease_deg)
    parts = grown_parts(sk, sight)
    if not parts:
        raise ManifestError(f"piece {info['piece']!r}: the skeleton grows no parts")
    wanted = _parts_wanted(sk, parts)
    missing = [name for name, *_ in wanted if materials.get(name) is None]
    if missing:
        raise ManifestError(f"piece {info['piece']!r}: no material for parts {missing}; "
                            f"materials given for {sorted(materials)}")
    if size_class not in SIZE_CLASSES:
        raise ManifestError(f"piece {info['piece']!r}: size_class must be one of "
                            f"{list(SIZE_CLASSES)}, got {size_class!r}")
    for what, text in (("done", done), ("origin", origin)):
        if not isinstance(text, str) or not text.strip():
            raise ManifestError(f"piece {info['piece']!r}: {what} must be a non-empty string, "
                                f"got {text!r}")
    _check_names(info, parts)

    listed = []
    for name, _shape, _on, _role, shade, level in wanted:
        thickness = 0.0
        if level is not None:
            smallest = min(float(sk.limb_radii(int(i)).min())
                           for i in np.flatnonzero(sk.level == level))
            thickness = 2.0 * smallest / builder.units_per_metre
        listed.append(_listed(rules, name, shade, sk.lite, thickness))
    leaf_names = [name for name, shape, *_ in wanted if shape == LEAF]
    normals = crown_normals({name: parts[name] for name in leaf_names}, lean)
    recipe = Recipe("branching", f"{info['rules']}, seed {info['seed']}, {info['height_m']} m"
                                 f"{', lite' if sk.lite else ''}")

    root = build.piece(info["piece"])
    made = [root]
    try:
        for (name, shape, on, role, _shade, _level), row in zip(wanted, listed, strict=True):
            made.append(mesh_from_tris(
                name, parts[name], materials[name], root, smooth=shape == TUBE,
                props={"listed": row.id, "group": name, "shape": shape, "on": on, "role": role}))
        builder.register(root, origin=origin, recipe=recipe, listed=listed, done=done,
                         size_class=size_class, kind="tree", view=view)
    except Exception:
        _remove(made)
        raise
    leaf_objects = {ob.name.split("/", 1)[1]: ob for ob in made[1:] if ob["shape"] == LEAF}

    def set_crown_normals() -> None:
        """Write each leaf mesh's custom normals: the crown's, one per corner."""
        for name, ob in leaf_objects.items():
            me = ob.data
            want = _to_blender(normals[name].reshape(-1, 3))
            if len(me.loops) != len(want):
                raise ManifestError(f"leaf part {name!r}: {len(me.loops)} corners but "
                                    f"{len(want)} crown normals")
            me.normals_split_custom_set(want.tolist())

    return root, set_crown_normals


def _remove(objects) -> None:
    """Delete objects (and the meshes they alone use) that a refused build made."""
    for ob in reversed(objects):
        me = ob.data if ob.type == "MESH" else None
        bpy.data.objects.remove(ob, do_unlink=True)
        if me is not None and me.users == 0:
            bpy.data.meshes.remove(me)
