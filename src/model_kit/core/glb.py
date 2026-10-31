"""A .glb's JSON chunk and its node table, read with `struct` and `json` only.

`read_glb_json` returns the glTF JSON. `glb_nodes` turns its `nodes` into `GlbNode` records in
file order: a node's name, its parent's name, its local transform (glTF order, `[x, y, z, w]`
rotation) and the name of its mesh. A node without a name, a node with two parents, a mesh
without a name, or a node that carries a `matrix` instead of a transform raises, because the
checks that read the table compare names. Two nodes may share a name (a real glTF can give a
bone and an object one name): the table keeps both, parents are resolved by index, and the
export-hygiene check reports the duplicate.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_kit.core.errors import ManifestError

_MAGIC = b"glTF"
_JSON_CHUNK = b"JSON"
_HEADER = 12
_CHUNK_HEAD = 8


@dataclass(frozen=True)
class GlbNode:
    """One node of a .glb. `parent` and `mesh` are names, or None."""

    name: str
    parent: str | None = None
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh: str | None = None


def node_name(piece: str, part: str) -> str:
    """The name of the node that carries a piece's part: `<piece>/<part>`. A Blender object
    name is global, so the build names each part object this way (`blender.build.object_name`),
    and the exporter keeps the name as the node's."""
    return f"{piece}/{part}"


def read_glb_json(path) -> dict:
    """The JSON chunk of a .glb. A file that is not a .glb, or lacks the chunk, raises."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{path}: cannot read the .glb: {exc}") from exc
    if len(data) < _HEADER + _CHUNK_HEAD or data[:4] != _MAGIC:
        raise ManifestError(f"{path}: not a .glb file (no 'glTF' header)")
    length, kind = struct.unpack_from("<I4s", data, _HEADER)
    start = _HEADER + _CHUNK_HEAD
    if kind != _JSON_CHUNK or start + length > len(data):
        raise ManifestError(f"{path}: the .glb's first chunk is not a whole JSON chunk")
    try:
        return json.loads(data[start:start + length])
    except ValueError as exc:
        raise ManifestError(f"{path}: the .glb's JSON chunk is not valid JSON: {exc}") from exc


def glb_nodes(path) -> list[GlbNode]:
    """The node table of a .glb, in file order."""
    path = Path(path)
    js = read_glb_json(path)
    raw_nodes = js.get("nodes", [])
    meshes = js.get("meshes", [])
    names: list[str] = []
    for i, node in enumerate(raw_nodes):
        name = node.get("name")
        if not isinstance(name, str) or not name:
            raise ManifestError(f"{path}: nodes[{i}] has no name")
        names.append(name)
    parent: dict[int, str] = {}
    for i, node in enumerate(raw_nodes):
        for child in node.get("children", []):
            if not isinstance(child, int) or not 0 <= child < len(names):
                raise ManifestError(f"{path}: node {names[i]!r} lists child {child!r}, "
                                    "which is no node")
            if child in parent:
                raise ManifestError(f"{path}: node {names[child]!r} has two parents: "
                                    f"{parent[child]!r} and {names[i]!r}")
            parent[child] = names[i]
    out = []
    for i, node in enumerate(raw_nodes):
        if "matrix" in node:
            raise ManifestError(f"{path}: node {names[i]!r} carries a matrix; the exporter "
                                "writes translation, rotation and scale")
        mesh = node.get("mesh")
        mesh_name = None
        if mesh is not None:
            if not isinstance(mesh, int) or not 0 <= mesh < len(meshes):
                raise ManifestError(f"{path}: node {names[i]!r} names mesh {mesh!r}, "
                                    "which is not in the file")
            mesh_name = meshes[mesh].get("name")
            if not isinstance(mesh_name, str) or not mesh_name:
                raise ManifestError(f"{path}: the mesh of node {names[i]!r} has no name")
        out.append(GlbNode(
            name=names[i], parent=parent.get(i),
            translation=_numbers(path, names[i], node, "translation", 3, (0.0, 0.0, 0.0)),
            rotation=_numbers(path, names[i], node, "rotation", 4, (0.0, 0.0, 0.0, 1.0)),
            scale=_numbers(path, names[i], node, "scale", 3, (1.0, 1.0, 1.0)),
            mesh=mesh_name,
        ))
    return out


def _numbers(path: Path, node: str, raw: dict, key: str, n: int, default: tuple) -> tuple:
    """A node's transform part; glTF's own default when the file leaves it out."""
    if key not in raw:
        return default
    value = raw[key]
    if (not isinstance(value, list) or len(value) != n
            or not all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)):
        raise ManifestError(f"{path}: node {node!r}: {key} must be {n} numbers, got {value!r}")
    return tuple(float(v) for v in value)


def nodes_to_json(nodes: list[GlbNode]) -> list[dict[str, Any]]:
    """A node table as the JSON list `nodes_from_json` reads."""
    return [{"name": n.name, "parent": n.parent, "translation": list(n.translation),
             "rotation": list(n.rotation), "scale": list(n.scale), "mesh": n.mesh}
            for n in nodes]


def nodes_from_json(path: Path, raw: Any) -> list[GlbNode]:
    """Read a node table written by `nodes_to_json`. Each failure names `path` and the node."""
    if not isinstance(raw, list):
        raise ManifestError(f"{path}: the node table must be a list, got {type(raw).__name__}")
    out = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"name", "parent", "translation",
                                                       "rotation", "scale", "mesh"}:
            raise ManifestError(f"{path}: nodes[{i}] must hold exactly name, parent, translation, "
                                f"rotation, scale and mesh, got {item!r}")
        name = item["name"]
        for key in ("name", "parent", "mesh"):
            if item[key] is not None and not isinstance(item[key], str):
                raise ManifestError(f"{path}: nodes[{i}].{key} must be a string or null")
        if not name:
            raise ManifestError(f"{path}: nodes[{i}] has no name")
        out.append(GlbNode(
            name=name, parent=item["parent"],
            translation=_numbers(path, name, item, "translation", 3, ()),
            rotation=_numbers(path, name, item, "rotation", 4, ()),
            scale=_numbers(path, name, item, "scale", 3, ()), mesh=item["mesh"],
        ))
    names = [n.name for n in out]
    for n in out:
        if n.parent is not None and n.parent not in names:
            raise ManifestError(f"{path}: node {n.name!r} has parent {n.parent!r}, "
                                "which is no node of the table")
    return out
