"""A light bake read for the gate: `<scene>/bake/` as plain data.

`model-kit bake` writes `<scene>/bake/` holding `<layout>.glb` (the placed layout; each part is a
node `<placement>/<part>`, and a part a camera sees carries a lightmap UV on the TEXCOORD index
`bake.json` records), one half-float EXR per layer and `bake.json`. `load_bake` reads that folder
into a `BakeData`; the checks do no file I/O.

- `uv[node]` is the lightmap UV of each seen part, (n_tris, 3, 2), per triangle corner. The .glb
  stores v from the top, as the picture's rows run, so a texel is `(u * size, v * size)` with no
  flip. A triangle of an unseen face has its three corners at one point (zero area).
- `positions[node]` is the same triangles' corners in the world, (n_tris, 3, 3), in glTF axes:
  each node's own transform and every parent's applied.
- `normals[node]` is the same triangles' vertex normals in the world, (n_tris, 3, 3): the .glb's
  NORMAL with each node's world matrix applied (unit length). A .glb with no NORMAL gets each
  triangle's flat face normal at all three corners.
- `stored[node]` is the same corners as the .glb stores them, (n_tris, 3, 3) float32 in the
  mesh's own frame, before any node transform: what float32 rounding acts on.
- `files` is the picture files the folder holds (every `*.exr`), whether or not `bake.json`
  lists them; `layers[name].pixels` is the picture as (H, W, 3) float32, or None when the file is
  not on disk.

A scene with no `bake/` folder has no bake: `load_bake` returns None.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.exr import read_exr
from model_kit.core.glb import GlbNode, glb_mesh_attributes, glb_nodes

BAKE_DIR = "bake"
MANIFEST = "bake.json"


@dataclass(frozen=True, eq=False)
class BakeLayer:
    """One layer of the bake: its picture file, the noise and error `bake.json` records per
    placement, and the picture's pixels (None when the file is not on disk)."""

    picture: str
    noise: dict[str, float] = field(default_factory=dict)
    error: dict[str, float] = field(default_factory=dict)
    pixels: np.ndarray | None = None  # (H, W, 3) float32, top row first


@dataclass(frozen=True, eq=False)
class BakeData:
    mode: str
    layers: dict[str, BakeLayer]
    uv: dict[str, np.ndarray]  # "<placement>/<part>" -> (n_tris, 3, 2) lightmap UV per corner
    positions: dict[str, np.ndarray]  # "<placement>/<part>" -> (n_tris, 3, 3) world corners
    normals: dict[str, np.ndarray]  # (n_tris, 3, 3) world-frame vertex normals, unit length
    size: int  # the shipped picture's side, in pixels
    unseen: frozenset[str]  # placements' parts no camera sees (no lightmap)
    files: frozenset[str]  # the pictures on disk
    # (n_tris, 3, 3) the same corners as the .glb stores them: float32, the mesh's own frame. The
    # seam check reads how far float32 can tilt each face's normal from them. Empty: no record.
    stored: dict[str, np.ndarray] = field(default_factory=dict)


def _rotation(q: tuple[float, float, float, float]) -> np.ndarray:
    """The 3x3 matrix of a glTF `[x, y, z, w]` rotation."""
    x, y, z, w = (float(c) for c in q)
    n = x * x + y * y + z * z + w * w
    s = 0.0 if n == 0 else 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def _world(nodes: dict[str, GlbNode], name: str) -> np.ndarray:
    """The 4x4 world matrix of node `name`: its own scale, rotation and translation, then each
    parent's."""
    matrix = np.eye(4)
    seen: set[str] = set()
    at: GlbNode | None = nodes[name]
    while at is not None:
        if at.name in seen:
            raise ManifestError(f"node {at.name!r} is its own ancestor")
        seen.add(at.name)
        local = np.eye(4)
        local[:3, :3] = _rotation(at.rotation) * np.asarray(at.scale, dtype=float)
        local[:3, 3] = at.translation
        matrix = local @ matrix
        at = nodes.get(at.parent) if at.parent else None
    return matrix


def area2(tri: np.ndarray) -> np.ndarray:
    """Twice the signed area of each triangle in (n, 3, 2)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])


def texels(uv: np.ndarray, size: int, *, edges: bool = False) -> np.ndarray:
    """The flat ids (`row * size + column`, unique) of the texels whose centres lie strictly
    inside a triangle of `uv` (n, 3, 2), on a `size` px picture. A triangle with zero area owns
    none. With `edges`, a centre on a triangle's edge counts as inside too, as the bake's own mask
    counts it: then the triangles of one island cover their shared edges with no gap."""
    out = []
    px = uv * size
    for tri, area in zip(px, area2(px), strict=True):
        if area == 0.0:
            continue
        lo = np.maximum(np.ceil(tri.min(axis=0) - 0.5), 0).astype(int)
        hi = np.minimum(np.floor(tri.max(axis=0) - 0.5), size - 1).astype(int)
        if (hi < lo).any():
            continue
        xs = np.arange(lo[0], hi[0] + 1) + 0.5
        ys = np.arange(lo[1], hi[1] + 1) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        inside = np.ones(gx.shape, dtype=bool)
        for i in range(3):
            p, q = tri[i], tri[(i + 1) % 3]
            edge = (q[0] - p[0]) * (gy - p[1]) - (q[1] - p[1]) * (gx - p[0])
            inside &= edge * area >= 0.0 if edges else edge * area > 0.0
        rows, cols = np.nonzero(inside)
        out.append((rows + lo[1]) * size + cols + lo[0])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)


def _read_layers(folder: Path, record: dict, files: frozenset[str]) -> dict[str, BakeLayer]:
    layers = {}
    for name, entry in record.get("layers", {}).items():
        picture = entry["picture"]
        pixels = read_exr(folder / picture) if picture in files else None
        layers[name] = BakeLayer(
            picture=picture,
            noise={k: float(v) for k, v in entry.get("noise", {}).items()},
            error={k: float(v) for k, v in entry.get("error", {}).items()},
            pixels=pixels)
    return layers


def _world_normals(attrs: dict, tri: np.ndarray, matrix: np.ndarray,
                   corners: np.ndarray) -> np.ndarray:
    """Unit vertex normals (n_tris, 3, 3) in the world: the NORMAL attribute through the inverse
    transpose of the node's world matrix (right under a non-uniform scale), or, with no NORMAL, each
    triangle's face normal from its world `corners`."""
    if "NORMAL" in attrs:
        inv_t = np.linalg.inv(matrix[:3, :3]).T
        out = attrs["NORMAL"][tri].astype(np.float64) @ inv_t.T
    else:
        face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        out = np.repeat(face[:, None, :], 3, axis=1)
    length = np.linalg.norm(out, axis=2, keepdims=True)
    return out / np.where(length == 0.0, 1.0, length)


def load_bake(scene: Path) -> BakeData | None:
    """The bake of scene folder `scene`, or None when it has no `bake/` folder. A `bake/` with
    no `bake.json` or no .glb, a part the record names that the .glb lacks, or a picture the
    reader cannot read raises `ManifestError`."""
    folder = Path(scene) / BAKE_DIR
    if not folder.is_dir():
        return None
    manifest = folder / MANIFEST
    if not manifest.is_file():
        raise ManifestError(f"{folder}: no {MANIFEST}")
    try:
        record = json.loads(manifest.read_text())
        layout, size = record["layout"], int(record["size"])
        channel, mode = record["uv_channel"], record["mode"]
        unseen = frozenset(record.get("unseen", ()))
    except (ValueError, KeyError, TypeError) as exc:
        raise ManifestError(f"{manifest}: not a bake record: {exc!r}") from exc
    glb = folder / f"{layout}.glb"
    if not glb.is_file():
        raise ManifestError(f"{folder}: {MANIFEST} names layout {layout!r}, but {glb.name} "
                            "is not there")
    files = frozenset(p.name for p in folder.glob("*.exr") if not p.name.startswith("."))
    nodes = {n.name: n for n in glb_nodes(glb)}
    uv: dict[str, np.ndarray] = {}
    positions: dict[str, np.ndarray] = {}
    normals: dict[str, np.ndarray] = {}
    stored: dict[str, np.ndarray] = {}
    for node, index in sorted(channel.items()):
        if node not in nodes:
            raise ManifestError(f"{manifest}: uv_channel names {node!r}, which {glb.name} lacks")
        attrs = glb_mesh_attributes(glb, node)
        key = f"TEXCOORD_{index}"
        if key not in attrs:
            raise ManifestError(f"{glb}: node {node!r} has no {key}, as {MANIFEST} records")
        tri = attrs["indices"].reshape(-1, 3)
        uv[node] = attrs[key][tri].astype(np.float64)
        matrix = _world(nodes, node)
        stored[node] = attrs["POSITION"][tri]
        corners = stored[node].astype(np.float64)
        positions[node] = corners @ matrix[:3, :3].T + matrix[:3, 3]
        normals[node] = _world_normals(attrs, tri, matrix, positions[node])
    return BakeData(mode=mode, layers=_read_layers(folder, record, files), uv=uv,
                    positions=positions, normals=normals, size=size, unseen=unseen, files=files,
                    stored=stored)
