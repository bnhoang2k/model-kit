"""A .glb's JSON chunk and its node table, read with `struct` and `json` only.

`read_glb_json` returns the glTF JSON. `glb_nodes` turns its `nodes` into `GlbNode` records in
file order: a node's name, its parent's name, its local transform (glTF order, `[x, y, z, w]`
rotation) and the name of its mesh. A node without a name, a node with two parents, a mesh
without a name, or a node that carries a `matrix` instead of a transform raises, because the
checks that read the table compare names. Two nodes may share a name (a real glTF can give a
bone and an object one name): the table keeps both, parents are resolved by index, and the
export-hygiene check reports the duplicate.

`glb_mesh_attributes` reads one node's mesh into numpy arrays: each vertex attribute under its
glTF name (`POSITION`, `TEXCOORD_0`, ...) and the triangle `indices`.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from model_kit.core.errors import ManifestError

_MAGIC = b"glTF"
_JSON_CHUNK = b"JSON"
_BIN_CHUNK = b"BIN\x00"
_HEADER = 12
_CHUNK_HEAD = 8
# glTF accessor component types, and the number of components of each accessor type.
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


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


def _bin_chunk(path: Path) -> bytes:
    data = path.read_bytes()
    start = _HEADER
    while start + _CHUNK_HEAD <= len(data):
        length, kind = struct.unpack_from("<I4s", data, start)
        body = start + _CHUNK_HEAD
        if kind == _BIN_CHUNK:
            return data[body:body + length]
        start = body + length
    raise ManifestError(f"{path}: the .glb has no binary chunk")


def _accessor(path: Path, js: dict, blob: bytes, index: int) -> np.ndarray:
    """One accessor as an array of shape (count,) for a scalar, else (count, width). A sparse
    accessor, or a component type this reader does not know, raises."""
    accessors = js.get("accessors", [])
    if not isinstance(index, int) or not 0 <= index < len(accessors):
        raise ManifestError(f"{path}: accessor {index!r} is not in the file")
    acc = accessors[index]
    if "sparse" in acc or "bufferView" not in acc:
        raise ManifestError(f"{path}: accessor {index} is sparse or has no buffer view; this "
                            "reader does not take it")
    dtype = _COMPONENT.get(acc.get("componentType"))
    width = _WIDTH.get(acc.get("type"))
    if dtype is None or width is None:
        raise ManifestError(f"{path}: accessor {index} has component type "
                            f"{acc.get('componentType')!r} and type {acc.get('type')!r}, which "
                            "this reader does not know")
    views = js.get("bufferViews", [])
    if not isinstance(acc["bufferView"], int) or not 0 <= acc["bufferView"] < len(views):
        raise ManifestError(f"{path}: accessor {index} names buffer view {acc['bufferView']!r}, "
                            "which is not in the file")
    view = views[acc["bufferView"]]
    count = acc["count"]
    size = np.dtype(dtype).itemsize
    stride = view.get("byteStride") or size * width
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    # `as_strided` reads wherever it is told: a short or malformed file would read memory past
    # the buffer, so the accessor must end inside its view, and the view inside the BIN chunk.
    counted = isinstance(count, int) and count >= 0
    end = (acc.get("byteOffset", 0) + (count - 1) * stride + size * width
           if counted and count > 0 else 0)
    view_end = view.get("byteOffset", 0) + view.get("byteLength", 0)
    if (not counted or min(start, stride) < 0 or end > view.get("byteLength", 0)
            or view_end > len(blob)):
        raise ManifestError(f"{path}: accessor {index} ({count!r} x {stride} bytes from byte "
                            f"{start}) runs past its buffer view or the file's {len(blob)} "
                            "binary bytes")
    flat = np.frombuffer(blob, dtype=np.uint8)
    rows = np.lib.stride_tricks.as_strided(
        flat[start:], shape=(count, size * width), strides=(stride, 1), writeable=False)
    out = np.ascontiguousarray(rows).view(dtype).reshape(count, width)
    return out[:, 0].copy() if width == 1 else out.copy()


def glb_mesh_attributes(path, node: str) -> dict[str, np.ndarray]:
    """The mesh of the node named `node`: each vertex attribute under its glTF name, and the
    triangle `indices` (a flat array into the vertices; the vertices of every primitive are
    joined, and each primitive's indices shifted to match). The name must belong to exactly one
    node, which holds a mesh of triangle primitives that carry the same attributes. A miss
    raises."""
    path = Path(path)
    js = read_glb_json(path)
    named = [n for n in js.get("nodes", []) if n.get("name") == node]
    if len(named) != 1 or "mesh" not in named[0]:
        raise ManifestError(f"{path}: {len(named)} nodes named {node!r} (need exactly one), or "
                            "it holds no mesh")
    blob = _bin_chunk(path)
    joined: dict[str, list[np.ndarray]] = {}
    indices: list[np.ndarray] = []
    base = 0
    for prim in js["meshes"][named[0]["mesh"]]["primitives"]:
        if prim.get("mode", 4) != 4:
            raise ManifestError(f"{path}: node {node!r} has a primitive that is not triangles")
        attrs = {k: _accessor(path, js, blob, v) for k, v in prim["attributes"].items()}
        if joined and set(attrs) != set(joined):
            raise ManifestError(f"{path}: node {node!r}: its primitives carry different "
                                f"attributes ({sorted(joined)} and {sorted(attrs)})")
        if "indices" in prim:
            local = _accessor(path, js, blob, prim["indices"]).astype(np.int64)
        else:
            local = np.arange(len(attrs["POSITION"]), dtype=np.int64)
        indices.append(local + base)
        for key, value in attrs.items():
            joined.setdefault(key, []).append(value)
        base += len(attrs["POSITION"])
    out = {key: np.concatenate(value) for key, value in joined.items()}
    out["indices"] = np.concatenate(indices)
    return out
