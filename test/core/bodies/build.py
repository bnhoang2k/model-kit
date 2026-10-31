"""Building blocks for the test bodies: meshing, skin weights, a rig and a scene in numpy.

A body is a few skinned parts (`BodyPart`), a rig (`BoneDef` list), clips posed from bone
matrices and a pack that holds the body and its furniture. Everything is built in glTF axes with
no Blender, and every manifest goes through its loader, so a body that loads is a body the kit
accepts.

Bone frames. A bone's local y runs from its head to its tail. Its local x is world +x (the
figure's left side, as the figure faces +z) and its local z is x cross y. `pose_world` turns a
bone about these local axes, and `RigData.world` is the identity. A right-hand turn about local
x tips a bone that points down toward the back (-z) and a bone that points up toward the front.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_kit.core.forms import chain_weights
from model_kit.core.manifest import (
    Falloff,
    check_stands,
    load_clips,
    load_layouts,
    load_pack,
)
from model_kit.core.pose import skin
from model_kit.core.rig import RigData, pose_world
from model_kit.core.scene import (
    ClipSamples,
    PieceGeom,
    RigRest,
    SceneCtx,
    check_clip_rigs,
    check_seats,
    piece_key,
)
from model_kit.core.tolerances import BIND_WEIGHT_FLOOR
from model_kit.core.units import Style

from ..checks.sight_scenes import view_json

# Every piece of a body scene is seen from the distance at which one arcminute covers this much
# (in metres): the contact length of its pieces.
SEEN_FLOOR = 0.0213

STYLE_DIALS = {
    "proportion": 0, "detail_floor_px": 0, "min_thickness": 0, "shading": "hairline",
    "edges": {"bevel": 0, "outline": "runtime", "crease_deg": 25,
              "crease_why": "a test value: the bodies need one angle for the checks to read, and "
                            "no source sets it"},
}


# The reference camera every real piece style declares (the bars': near 0.1 m, far 2000 m, a
# 24-bit integer depth buffer, which is unverified), so the `coplanar` check has a depth step.
DEPTH_CAMERA = {"depth_projection": "perspective", "near_m": 0.1, "far_m": 2000.0,
                "depth_bits": 24,
                "depth_why": "a test camera, three.js's default near and far planes"}


def body_style() -> Style:
    return Style(units_per_metre=1.0, page_camera={"projection": "ortho", "turn_deg": 45,
                                                    "tilt_deg": 30, "px_per_unit": None,
                                                    **DEPTH_CAMERA},
                 dials=json.loads(json.dumps(STYLE_DIALS)))


def body_view() -> dict:
    """The `view` entry of every piece of a body scene. Its `farthest_m` equals its `nearest_m`: a
    zero view range, a design call of this kit.

    Why not the widest shipping-bar range (03-chibi, 4.615x)? The bodies' nearest is 73 m (their
    contact length, `SEEN_FLOOR`, is 2.1 cm), so 4.615x puts them at 338 m, where a 24-bit depth
    step is 0.068 m. That is harsher on them than on any bar, and their 2.1 cm decal lift would
    flicker by `sight.decal_lift`'s own rule (the step passes the lift at 2.58x). With the
    closed-part exemption of `coplanar`, the bodies stay quiet only up to about 1.4x (a scan of the
    three bodies: 0 findings at 1.37x, 1 each at 1.5x, 13/6/6 at 2.58x, 16/9/9 at 4.615x).
    `coplanar` at range is covered by its own unit tests and the bars."""
    view = view_json(SEEN_FLOOR)
    return {**view, "farthest_m": view["nearest_m"],
            "farthest_why": "zero view range, a design choice of the test bodies; see body_view"}


# --- meshes ----------------------------------------------------------------------------------


def sweep(stations, axis: str, caps=("round", "round"), segments: int = 16,
          cap_rings: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A closed tube along `axis` ("y" or "z"): (vertices (V, 3), faces (T, 3)), wound outward.

    Each station is (t, centre_a, centre_b, half_a, half_b): a position along the axis, the ring's
    centre and half sizes on the other two axes (x and z for a y tube, x and y for a z tube).
    Stations may run up or down the axis. Each end is "round" (a hemisphere as long as the end
    ring's smaller half) or "flat" (a fan across the end ring, so a flat end sits exactly on the
    station). The ring has `segments` points; a multiple of 4 puts a vertex straight down."""
    st = np.asarray(stations, dtype=float)
    ax = "xyz".index(axis)
    cols = [i for i in range(3) if i != ax]
    sign = 1.0 if st[-1, 0] >= st[0, 0] else -1.0

    def cap(row, outward):
        t, ca, cb, ha, hb = row
        reach = min(ha, hb)
        if caps[0 if outward == -sign else 1] == "flat":
            return [], t
        phi = (np.pi / 2) * np.arange(1, cap_rings + 1) / (cap_rings + 1)
        return ([(t + outward * reach * math.sin(p), ca, cb, ha * math.cos(p), hb * math.cos(p))
                 for p in phi], t + outward * reach)

    head, apex_a = cap(st[0], -sign)
    tail, apex_b = cap(st[-1], sign)
    rings = np.array([*head[::-1], *st, *tail])
    theta = 2 * np.pi * np.arange(segments) / segments
    n = len(rings)
    pts = np.zeros((n, segments, 3))
    pts[:, :, ax] = rings[:, 0, None]
    pts[:, :, cols[0]] = rings[:, 1, None] + rings[:, 3, None] * np.cos(theta)
    pts[:, :, cols[1]] = rings[:, 2, None] + rings[:, 4, None] * np.sin(theta)
    apexes = np.zeros((2, 3))
    for k, (row, apex) in enumerate(((rings[0], apex_a), (rings[-1], apex_b))):
        apexes[k, ax] = apex
        apexes[k, cols[0]], apexes[k, cols[1]] = row[1], row[2]
    verts = np.vstack([pts.reshape(-1, 3), apexes])
    j = np.arange(segments)
    j1 = (j + 1) % segments
    faces = []
    for r in range(n - 1):
        a, b = r * segments, (r + 1) * segments
        faces += [np.stack([a + j, a + j1, b + j1], 1), np.stack([a + j, b + j1, b + j], 1)]
    first, last = n * segments, n * segments + 1
    faces.append(np.stack([np.full(segments, first), j1, j], 1))
    faces.append(np.stack([np.full(segments, last), (n - 1) * segments + j,
                           (n - 1) * segments + j1], 1))
    faces = np.vstack(faces).astype(np.int32)
    tris = verts[faces]
    if np.einsum("nk,nk->", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) < 0:
        faces = faces[:, ::-1].copy()
    return verts, faces


# --- skin ------------------------------------------------------------------------------------


@dataclass
class BodyPart:
    """One skinned shell. A part bound to several bones carries `falloff` (the manifest's
    `{along, start, end, bands}`), which is the only source of its weights."""

    name: str
    verts: np.ndarray  # (V, 3) float32
    faces: np.ndarray  # (T, 3) int32
    bones: tuple[str, ...]
    falloff: dict | None
    listed: str
    group: str
    shape: str = "ring"
    on: str | None = None

    def __post_init__(self) -> None:
        self.verts = np.asarray(self.verts, dtype=np.float32)  # one rounding, used everywhere

    def dense_weights(self) -> np.ndarray:
        """(V, n bones) weights, rows summing to 1, none at or under `BIND_WEIGHT_FLOOR`."""
        n = len(self.bones)
        if self.falloff is None:
            return np.ones((len(self.verts), 1))
        f = self.falloff
        bands = tuple((lo, hi) for lo, hi in f["bands"])
        falloff = Falloff(f["along"], f["start"], f["end"], bands)
        w = chain_weights(self.verts.astype(float), falloff, n)
        w = np.where(w <= BIND_WEIGHT_FLOOR, 0.0, w)
        return w / w.sum(axis=1, keepdims=True)


def fall(along: str, start: float, end: float, *bands) -> dict:
    """A chain falloff. Each band is the pair of coordinates where a weight passes from one bone
    to the next; the pair may run either way."""
    def u(c):
        return (c - start) / (end - start)
    return {"along": along, "start": start, "end": end,
            "bands": [sorted((u(a), u(b))) for a, b in bands]}


def mirror_part(part: BodyPart, name: str, bones: tuple[str, ...]) -> BodyPart:
    """The part as its mirror image across x = 0, triangle for triangle: x negated, each face's
    winding reversed so it still faces outward, vertex order kept (so vertex i mirrors vertex i)."""
    return BodyPart(name, part.verts * np.array([-1.0, 1.0, 1.0], dtype=np.float32),
                    part.faces[:, ::-1].copy(), bones, part.falloff, part.listed, part.group,
                    part.shape, part.on)


def rest_body(parts: Sequence[BodyPart]) -> RigRest:
    """The rig's rest file: shells, faces, strongest-group labels and the compact skin."""
    out = RigRest({}, {}, {}, {})
    for part in sorted(parts, key=lambda p: p.name):
        w = part.dense_weights()
        k = int((w > 0).sum(axis=1).max())
        order = np.argsort(w <= 0, axis=1, kind="stable")[:, :k]  # nonzero slots first, by group
        out.shells[part.name] = part.verts[None].astype(np.float32)
        out.faces[part.name] = part.faces.astype(np.int32)
        out.labels[part.name] = np.argmax(w, axis=1).astype(np.int32)  # a tie goes to the first
        out.vertex_groups[part.name] = part.bones
        out.weights[part.name] = np.take_along_axis(w, order, 1).astype(np.float32)
        out.weight_groups[part.name] = np.where(out.weights[part.name] > 0, order, 0
                                                ).astype(np.int32)
    return out


# --- rig -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BoneDef:
    name: str
    parent: str | None
    head: tuple[float, float, float]
    tail: tuple[float, float, float]


def bone_frame(head, tail) -> np.ndarray:
    """The bone's rest matrix: local y head to tail, local x world +x, local z = x cross y."""
    y = np.asarray(tail, float) - np.asarray(head, float)
    y = y / np.linalg.norm(y)
    x = np.array([1.0, 0.0, 0.0]) - y[0] * y
    x = x / np.linalg.norm(x)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2] = x, y, np.cross(x, y)
    m[:3, 3] = head
    return m


def rig_data(bones: Sequence[BoneDef]) -> RigData:
    """The rig as the checks read it. Bones are listed parents first, as given."""
    return RigData(order=tuple(b.name for b in bones), parent={b.name: b.parent for b in bones},
                   rest={b.name: bone_frame(b.head, b.tail) for b in bones}, world=np.eye(4),
                   effector_mid={})


def bones_json(bones: Sequence[BoneDef]) -> list[dict]:
    return [{"name": b.name, "parent": b.parent, "length": math.dist(b.head, b.tail)}
            for b in bones]


def bind_json(parts: Sequence[BodyPart]) -> dict:
    return {p.name: ({"bones": list(p.bones), "mode": "rigid"} if p.falloff is None else
                     {"bones": list(p.bones), "mode": "chain", "falloff": p.falloff})
            for p in parts}


def set_effector(rig: RigData, parts: Sequence[BodyPart], bone: str, group: Sequence[str]
                 ) -> float:
    """Store the effector's middle in `rig.effector_mid` and return its radius, both from the rest
    mesh as `export_rig` and `bind.effector_radius` take them: the patch is every vertex whose
    strongest bone is in `group`; the middle is its centroid in the effector bone's rest frame; the
    radius is the patch's reach along the effector bone from that centroid."""
    patch = []
    for p in parts:
        w = p.dense_weights()
        strongest = np.array(p.bones)[np.argmax(w, axis=1)]
        patch.append(p.verts[np.isin(strongest, list(group))].astype(float))
    patch = np.concatenate(patch)
    centre = patch.mean(axis=0)
    rest = rig.rest[bone]
    rig.effector_mid[bone] = (np.linalg.inv(rest) @ np.append(centre, 1.0))[:3]
    return float(((patch - centre) @ rest[:3, 1]).max())


def flip(rng: tuple[float, float]) -> tuple[float, float]:
    """A left-side range for the right side: the turns about local y and z change sign."""
    return (-rng[1], -rng[0])


# --- poses and clips -------------------------------------------------------------------------


def posed(rig: RigData, turns: dict[str, Sequence[float]], moves: dict[str, Sequence[float]]
          | None = None) -> dict[str, np.ndarray]:
    """Posed world matrices. `turns` are euler degrees per bone (x, y, z); `moves` are world
    moves of a parentless bone, turned into the bone's local frame."""
    pose = {b: (np.radians(e), np.zeros(3)) for b, e in turns.items()}
    for b, d in (moves or {}).items():
        assert rig.parent[b] is None, f"{b} has a parent: only a parentless bone moves"
        euler = pose.get(b, (np.zeros(3), None))[0]
        pose[b] = (euler, rig.rest[b][:3, :3].T @ np.asarray(d, float))
    return pose_world(rig, pose)


@dataclass
class ClipSpec:
    raw: dict  # the clip's entry in clips.json
    times: np.ndarray  # frame numbers, one per pose
    poses: list[dict[str, np.ndarray]]  # posed world matrices, from `posed`


def sample_clip(name: str, spec: ClipSpec, rig: RigData, rest: RigRest,
                groups: dict[str, Sequence[str]]) -> ClipSamples:
    """The saved samples of a clip: `b:` bone matrices, `v:` skinned shells and `g:` the points
    of each contact group (the vertices whose strongest bone is in it, parts in name order)."""
    s = len(spec.poses)
    bones = {b: np.stack([pose[b] for pose in spec.poses]) for b in rig.order}
    shells = {p: np.empty((s, rest.shells[p].shape[1], 3), np.float32) for p in rest.shells}
    for i, pose in enumerate(spec.poses):
        for part, points in skin(rest, rig, pose).items():
            shells[part][i] = points
    points = {}
    for group, members in groups.items():
        per = []
        for part in sorted(rest.shells):
            names = np.array(rest.vertex_groups[part])
            keep = np.isin(names[rest.labels[part]], list(members))
            per.append(shells[part][:, keep])
        points[group] = np.concatenate(per, axis=1)
    return ClipSamples(clip=name, times=np.asarray(spec.times, float), groups=points, bones=bones,
                       props={}, shells=shells, faces=dict(rest.faces), labels=dict(rest.labels))


# --- the scene -------------------------------------------------------------------------------


@dataclass
class PieceSpec:
    """A furniture piece: its manifest entry and its triangles. `geometry` is given the decal
    lift (the pieces' contact length, `SEEN_FLOOR`) and returns one (N, 3, 3) array per part."""

    raw: dict
    geometry: Callable[[float], dict[str, np.ndarray]]


def write_json(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def assemble(*, pack: str, body: str, body_raw: dict, parts: Sequence[BodyPart], rig: RigData,
             furniture: dict[str, PieceSpec], clips: dict[str, ClipSpec], layout: dict,
             groups: dict[str, Sequence[str]]) -> SceneCtx:
    """A full scene. Pack, clips and layout are written to a scratch folder and read back by
    their loaders; the rest file and the samples are made here."""
    style = body_style()
    raw = {"pack": pack, "units_per_metre": 1.0,
           "pieces": {n: {"view": body_view(), **piece} for n, piece in {
               body: body_raw, **{n: f.raw for n, f in furniture.items()}}.items()}}
    with tempfile.TemporaryDirectory(prefix="model-kit-body-") as tmp:
        folder = Path(tmp)
        loaded = load_pack(write_json(folder / "p.pack.json", raw), style)
        clip_list = load_clips(write_json(folder / "clips.json", {"clips": {
            n: c.raw for n, c in clips.items()}}))
        layouts = load_layouts(write_json(folder / "layouts" / "l.json", layout).parent,
                               {pack: loaded})
    lift = SEEN_FLOOR  # a decal stands off its host by its piece's contact length
    rest = rest_body(parts)
    pieces = {piece_key(pack, body): PieceGeom(body, {
        p.name: p.verts.astype(float)[p.faces] for p in sorted(parts, key=lambda p: p.name)})}
    for name, spec in furniture.items():
        pieces[piece_key(pack, name)] = PieceGeom(name, spec.geometry(lift))
    key = piece_key(pack, body)
    clip_map = {c.name: c for c in clip_list}
    ctx = SceneCtx(
        style=style, packs={pack: loaded}, clips=clip_map, pieces=pieces,
        samples={n: sample_clip(n, c, rig, rest, groups) for n, c in clips.items()},
        rigs={key: rig}, rig_of_clip={n: key for n in clips}, layouts=layouts, rests={key: rest})
    check_seats(ctx.packs, ctx.clips)
    check_clip_rigs(ctx.packs, ctx.clips, ctx.rig_of_clip)
    check_stands(ctx.layouts, ctx.clips)
    return ctx
