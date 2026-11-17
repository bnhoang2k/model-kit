"""The scene format: a built scene saved to a folder and loaded back for the checks.

Pieces are triangles in each piece's own frame. Samples are in the world frame, in glTF axes,
and their `times` are clip frame numbers, the same unit as a hold's `when` windows. The layout:

    style.json
    packs/<pack>.pack.json         the pack manifest (binds under each rig's `bind`)
    clips.json                     the clip manifest
    pieces/<pack>/<piece>.npz      one (N, 3, 3) array per part, keyed by part name
    samples/<clip>.npz             `times`, `g:<group>`, `b:<bone>`, `p:<prop>`, and per body
                                   part `v:<part>` (S, V, 3) float32 posed points, `f:<part>`
                                   (T, 3) int32 faces, `l:<part>` (V,) int32 vertex group labels
    rigs/<pack>/<piece>.json       the rig data, plus `clips`: the clips that pose this rig, and
                                   `vertex_groups`: per part, the group names its labels index
    rigs/<pack>/<piece>.rest.npz   the rig's body shells at rest (`v:`, `f:`, `l:` per part) and,
                                   optionally, the skin: `w:<part>` (V, K) float32 weights and
                                   `j:<part>` (V, K) int32 indices into `vertex_groups[part]`,
                                   padded with weight 0, index 0. Present exactly when the rig
                                   file holds `vertex_groups`
    layouts/<layout>.json          where pieces stand (see `manifest/layouts.py`); optional
    sources/<name>.py              the build scripts, as text; optional
    nodes/<pack>.json              the packed .glb's node table, `{"nodes": [...]}`; optional
                                   (an exporter writes it with `write_nodes`)
    pieces/<pack>/<piece>.skeleton.npz   a branching piece's grown skeleton; optional
    rules/<file>.rules.json        a species' growth rules, loaded with the kit file; optional
                                   (`sources/` takes only .py, so rules have their own folder)
    bake/                          a light bake (`model-kit bake` writes it; see `core.bake_data`);
                                   optional, read into `SceneCtx.bake`

Load checks what the checks would otherwise meet late: each geometry key is a part name (not a
Blender object name) and the piece's hold and seat surface parts all have geometry; sample
times are finite and strictly increasing; shell faces and labels are in range; and each rig file
is the rig of the rigged piece its path names (the folder is the loaded pack, the stem the
piece), with the same bones and parents as that piece's rig. Two packs may each have a rigged
piece of one name. A skeleton file must belong to a piece the pack has, and that piece must be
a branching one.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from model_kit.core.bake_data import BakeData, load_bake
from model_kit.core.errors import ManifestError
from model_kit.core.glb import GlbNode, glb_nodes, nodes_from_json, nodes_to_json
from model_kit.core.grow.rules import KIT_RULES, GrowthRules, load_rules
from model_kit.core.grow.skeleton import Skeleton, load_skeleton, save_skeleton
from model_kit.core.manifest import (
    Chain,
    Clip,
    Layout,
    Pack,
    Piece,
    RigSpec,
    anchor,
    check_connectors,
    check_stands,
    load_clips,
    load_layouts,
    load_pack,
)
from model_kit.core.rig import RigData
from model_kit.core.units import Style, load_style
from model_kit.core.writer import clips_json, write_layout, write_pack


@dataclass(eq=False)
class PieceGeom:
    """A piece's triangles in its own frame: one (N, 3, 3) array per part."""

    name: str
    parts: dict[str, np.ndarray]


@dataclass(eq=False)
class ClipSamples:
    """A clip sampled in the world frame. `times` (S,) are clip frame numbers."""

    clip: str
    times: np.ndarray  # (S,)
    groups: dict[str, np.ndarray]  # contact group -> (S, V, 3) points
    bones: dict[str, np.ndarray]  # bone -> (S, 4, 4) posed world matrix
    props: dict[str, np.ndarray]  # prop id -> (S, 4, 4) world matrix of the prop's frame
    # The body: per part, the posed points of its mesh, the faces (the same at every sample)
    # and each vertex's strongest deform group as an index into the rig's `vertex_groups[part]`.
    shells: dict[str, np.ndarray] = field(default_factory=dict)  # part -> (S, V, 3) float32
    faces: dict[str, np.ndarray] = field(default_factory=dict)  # part -> (T, 3) int32
    labels: dict[str, np.ndarray] = field(default_factory=dict)  # part -> (V,) int32


@dataclass(eq=False)
class RigRest:
    """A rig's body at rest: the same three arrays as `ClipSamples`, with S = 1, the group names
    that the labels index into and, when the file has them, the raw skin weights."""

    shells: dict[str, np.ndarray]  # part -> (1, V, 3) float32
    faces: dict[str, np.ndarray]  # part -> (T, 3) int32
    labels: dict[str, np.ndarray]  # part -> (V,) int32
    vertex_groups: dict[str, tuple[str, ...]]  # part -> deform group names, label order
    # The raw skin weights, empty for a rest file written before skin weights were stored: per part, (V, K) float32
    # weights and (V, K) int32 indices into `vertex_groups[part]`, short vertices padded with
    # weight 0 and index 0. They are as stored in the vertex groups: the bind step already floored
    # what it wrote, and the skinner renormalises each vertex.
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    weight_groups: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class SceneCtx:
    style: Style
    packs: dict[str, Pack]
    clips: dict[str, Clip]
    pieces: dict[str, PieceGeom]  # keyed by piece_key(pack, piece)
    samples: dict[str, ClipSamples]  # keyed by clip name
    rigs: dict[str, RigData]  # keyed by piece_key(pack, piece) of the rigged piece
    rig_of_clip: dict[str, str]  # clip name -> the rigged piece's key in `rigs`
    layouts: dict[str, Layout] = field(default_factory=dict)  # keyed by layout name
    sources: dict[str, str] = field(default_factory=dict)  # build script file name -> text
    nodes: dict[str, list[GlbNode]] = field(default_factory=dict)  # pack -> .glb node table
    rests: dict[str, RigRest] = field(default_factory=dict)  # keyed like `rigs`
    skeletons: dict[str, Skeleton] = field(default_factory=dict)  # keyed by piece_key
    folder: str = ""  # the scene folder's name, set by `load_scene`; empty for a scene in code
    rules: dict[str, GrowthRules] = field(default_factory=dict)  # `rules/` file name -> rules
    bake: BakeData | None = None  # `bake/`, read by `load_scene`; None for a scene with none


def piece_key(pack: str, piece: str) -> str:
    """The key of a piece's geometry in `SceneCtx.pieces`."""
    return f"{pack}:{piece}"


def part_key(pack: str, piece: str, part: str) -> str:
    """The key of one exported part: `<pack>:<piece> <part>`. It is the item of floaters and
    topology, and the form every check's `measured` hook returns (see `metagate.Measures`)."""
    return f"{piece_key(pack, piece)} {part}"


def piece_geom(ctx: SceneCtx, pack: str, piece: str) -> PieceGeom:
    """A piece's geometry. A miss raises."""
    geom = ctx.pieces.get(piece_key(pack, piece))
    if geom is None:
        raise ManifestError(f"{piece_key(pack, piece)}: the scene holds no geometry for it")
    return geom


# --- effector radius ------------------------------------------------------------------------


def rig_spec(ctx: SceneCtx, clip: str) -> RigSpec:
    """The rig spec of the piece that a clip poses. A miss raises."""
    if clip not in ctx.rig_of_clip:
        raise ManifestError(f"clip {clip!r}: the scene names no rig for it (rig_of_clip)")
    key = ctx.rig_of_clip[clip]
    pack_name, _, name = key.partition(":")
    pack = ctx.packs.get(pack_name)
    piece = None if pack is None else pack.pieces.get(name)
    if piece is None or piece.rig is None:
        raise ManifestError(f"clip {clip!r}: rig {key!r} is no rigged piece of the loaded "
                            "packs")
    return piece.rig


def check_times(times, where: str) -> None:
    """Sample times must be a non-empty run of finite, strictly increasing frame numbers."""
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        raise ManifestError(f"{where}: the samples hold no times")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
        raise ManifestError(f"{where}: sample times must be finite and strictly increasing")


# A sample is due when its time lies in a window, give or take this many frames (float rounding
# of sample times, not a measuring tolerance).
DUE_SLACK_FRAMES = 1e-6


def check_windows(clip: Clip, windows, item: str, field: str = "when") -> None:
    """Raise a `ManifestError` naming `item` and `field` unless `windows` is a non-empty run of
    forward windows inside the clip's frames. A clip built in code is guarded like a loaded
    one."""
    if not windows:
        raise ManifestError(f"{item}: its {field} windows are empty")
    for t0, t1 in windows:
        if t0 > t1:
            raise ManifestError(f"{item}: {field} window {[t0, t1]} runs backwards")
        if t0 < clip.frames[0] or t1 > clip.frames[1]:
            raise ManifestError(f"{item}: {field} window {[t0, t1]} is outside the clip's "
                                f"frames {list(clip.frames)}")


def due(times: np.ndarray, windows) -> np.ndarray:
    """The indices of the samples whose time lies in any of `windows`."""
    mask = np.zeros(len(times), dtype=bool)
    for t0, t1 in windows:
        mask |= (times >= t0 - DUE_SLACK_FRAMES) & (times <= t1 + DUE_SLACK_FRAMES)
    return np.nonzero(mask)[0]


def _one_chain(chains: list[tuple[str, Chain]], clip: str, what: str) -> Chain:
    if len(chains) != 1:
        names = [name for name, _ in chains]
        raise ManifestError(f"clip {clip!r}: {what} must match one chain, matched {names}")
    return chains[0][1]


def effector_radius(ctx: SceneCtx, clip: str, who: str) -> float:
    """The effector radius for a hold's `who`.

    A contact group uses the chain whose `effector_group` is that group. A held prop uses the
    chain of the bone it rides: the chain whose effector is that bone, else the one chain
    that holds the bone.
    """
    rig = rig_spec(ctx, clip)
    if clip not in ctx.clips:
        raise ManifestError(f"clip {clip!r}: not in the scene's clips")
    prop = ctx.clips[clip].props.get(who)
    if prop is None:
        chains = [(n, c) for n, c in rig.chains.items() if c.effector_group == who]
        return _one_chain(chains, clip, f"contact group {who!r}").effector_radius
    chains = [(n, c) for n, c in rig.chains.items() if c.effector == prop.bone]
    if not chains:
        chains = [(n, c) for n, c in rig.chains.items() if prop.bone in c.bones]
    return _one_chain(chains, clip, f"prop {who!r} on bone {prop.bone!r}").effector_radius


# --- checks on the whole scene --------------------------------------------------------------


def check_seats(packs: dict[str, Pack], clips: dict[str, Clip]) -> None:
    """Each clip's `seat` ref must name a seat. The clip loader cannot see packs."""
    for clip in clips.values():
        if clip.seat is None:
            continue
        found = anchor(packs, clip.seat, clip=clip)
        if found.kind != "seat":
            raise ManifestError(f"clip {clip.name!r}: seat {clip.seat} names a {found.kind}, "
                                "not a seat")


def check_clip_rigs(packs: dict[str, Pack], clips: dict[str, Clip],
                    rig_of_clip: dict[str, str]) -> None:
    """What a clip asks of its rig, for each clip a rig poses (the clip loader cannot see rigs):
    a lift names a support the rig lists, `mirror_of` needs the rig's mirror map, and a seat
    needs the rig's sit. A clip that no rig poses is left alone."""
    for name, key in rig_of_clip.items():
        clip = clips[name]
        pack_name, _, piece = key.partition(":")
        spec = packs[pack_name].pieces[piece].rig
        for i, lift in enumerate(clip.lifts):
            if lift.support not in spec.supports:
                raise ManifestError(f"clip {name!r}: lifts[{i}] names {lift.support!r}, which "
                                    f"is not a support of rig {key!r} "
                                    f"(its supports: {list(spec.supports)})")
        if clip.mirror_of is not None and spec.mirror is None:
            raise ManifestError(f"clip {name!r}: mirror_of {clip.mirror_of!r}, but rig {key!r} "
                                "has no mirror map")
        if clip.seat is not None and spec.sit is None:
            raise ManifestError(f"clip {name!r}: names a seat, but rig {key!r} has no sit")


# --- save and load --------------------------------------------------------------------------

_SAFE = re.compile(r"[A-Za-z0-9_.-]+")
SKELETON_SUFFIX = ".skeleton.npz"  # beside a piece's geometry: `pieces/<pack>/<piece>.skeleton.npz`
_PART = re.compile(r"[A-Za-z0-9_-]+")  # the manifest's part names
REST_SUFFIX = ".rest.npz"  # `rigs/<pack>/<piece>.rest.npz` sits beside the rig file
RULES_DIR = "rules"  # `rules/<file>.rules.json`: a species' growth rules
RULES_SUFFIX = ".rules.json"


def _file_name(name: str, what: str) -> str:
    if not _SAFE.fullmatch(name) or name in {".", ".."}:
        raise ManifestError(f"{what} {name!r} cannot be a file name")
    return name


def _rules_name(name: str, what: str) -> str:
    _file_name(name, what)
    if not name.endswith(RULES_SUFFIX) or name.startswith("."):
        raise ManifestError(f"{what} {name!r}: a rules file name ends in {RULES_SUFFIX} and "
                            "does not start with '.' (load skips hidden files)")
    return name


def _vec(v) -> list[float]:
    return [float(x) for x in v]


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_scene(ctx: SceneCtx, dir: Path) -> None:
    """Write a scene to an empty or new folder. A folder with files in it is refused.

    The scene is written to a private temporary folder first and read back there with
    `load_scene`, so save refuses exactly what load refuses. Each pack goes through
    `writer.write_pack`, so it must also read back as the same records. Only a scene that loads
    is copied into `dir`; a refused scene leaves `dir` as it was.
    """
    dir = Path(dir)
    if dir.exists() and (not dir.is_dir() or any(dir.iterdir())):
        raise ManifestError(f"{dir}: the scene folder is not empty")
    _check_keys(ctx)
    with tempfile.TemporaryDirectory(prefix="model-kit-scene-") as tmp:
        staged = Path(tmp) / "scene"
        try:
            _write_scene(ctx, staged)
            load_scene(staged)
        except ManifestError as exc:  # name the user's folder, not the deleted staging one
            raise type(exc)(str(exc).replace(str(staged), str(dir))) from None
        shutil.copytree(staged, dir, dirs_exist_ok=True)


def write_nodes(dir: Path, pack: str, glb: Path) -> None:
    """Write `nodes/<pack>.json` into a scene folder from the packed .glb of that pack: the node
    table export hygiene reads. An exporter calls this once the pack's .glb is written, so the
    table is the file's own."""
    _file_name(pack, "nodes pack")
    _write_json(Path(dir) / "nodes" / f"{pack}.json", {"nodes": nodes_to_json(glb_nodes(glb))})


def stage_manifests(dir: Path, *, style: Path, clips: Path, packs: Sequence[Path],
                    rules: Sequence[Path] = ()) -> None:
    """Lay out a scene folder's manifests: `style.json`, `clips.json`, `packs/<name>.pack.json`
    and `rules/<file>.rules.json` for each of `rules`.

    Each file is loaded first, which validates it: the style, then each pack against that style,
    then the clips, then each rules file with the kit file (`KIT_RULES`). Only then are the files
    copied, byte for byte. A file that does not load, two packs or two rules files of one name,
    or a target file already in `dir` raises `ManifestError`, and nothing is written. Exporters fill the rest of the folder; `load_scene` reads the whole.
    """
    dir = Path(dir)
    loaded = load_style(style)
    named: dict[str, Path] = {}
    for path in packs:
        name = load_pack(path, loaded).name
        _file_name(name, "pack")
        if name in named:
            raise ManifestError(f"two packs are named {name!r}: {named[name]} and {path}")
        named[name] = Path(path)
    load_clips(clips)
    rules_named: dict[str, Path] = {}
    for path in rules:
        name = _rules_name(Path(path).name, "rules file")
        if name in rules_named:
            raise ManifestError(f"two rules files are named {name!r}: {rules_named[name]} "
                                f"and {path}")
        load_rules(Path(path), kit=KIT_RULES)
        rules_named[name] = Path(path)
    targets = [(Path(style), dir / "style.json"), (Path(clips), dir / "clips.json"),
               *((src, dir / "packs" / f"{name}.pack.json") for name, src in named.items()),
               *((src, dir / RULES_DIR / name) for name, src in rules_named.items())]
    for _, target in targets:
        if target.exists():
            raise ManifestError(f"{target}: the scene folder already holds this file")
    for src, target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)


def _check_keys(ctx: SceneCtx) -> None:
    """What the files cannot show: each dict key agrees with the record it holds."""
    for name, pack in ctx.packs.items():
        if name != pack.name:
            raise ManifestError(f"packs[{name!r}] holds pack {pack.name!r}")
    for name, clip in ctx.clips.items():
        if name != clip.name:
            raise ManifestError(f"clips[{name!r}] holds clip {clip.name!r}")
    for key, geom in ctx.pieces.items():
        pack, _, piece = key.partition(":")
        if key != piece_key(pack, geom.name):
            raise ManifestError(f"pieces[{key!r}] holds piece {geom.name!r}; the key must be "
                                f"{piece_key(pack or '<pack>', geom.name)!r}")
        _file_name(pack, "pack")
        _file_name(piece, "piece")
    for name, s in ctx.samples.items():
        if name != s.clip:
            raise ManifestError(f"samples[{name!r}] holds clip {s.clip!r}")
        _file_name(name, "clip")
    for key in ctx.rigs:
        pack, _, piece = key.partition(":")
        if key != piece_key(pack, piece) or not pack:
            raise ManifestError(f"rigs[{key!r}]: a rig is keyed by piece_key(pack, piece)")
        if pack not in ctx.packs:
            raise ManifestError(f"rigs[{key!r}]: pack {pack!r} is not one of the scene's packs "
                                f"{sorted(ctx.packs)}")
        _file_name(pack, "rig pack")
        _file_name(piece, "rig piece")
    for key in ctx.rests:
        if key not in ctx.rigs:
            raise ManifestError(f"rests[{key!r}]: the scene has no rig of that key; a rest "
                                "belongs to a rig in `rigs`")
    for clip, rig in ctx.rig_of_clip.items():
        if rig not in ctx.rigs:
            raise ManifestError(f"clip {clip!r}: rig {rig!r} is not in the scene's rigs")
    for name, layout in ctx.layouts.items():
        if name != layout.name:
            raise ManifestError(f"layouts[{name!r}] holds layout {layout.name!r}")
        _file_name(name, "layout")
    for name in ctx.sources:
        if not name.endswith(".py") or name.startswith("."):
            raise ManifestError(f"sources[{name!r}]: a build script file name ends in .py and "
                                "does not start with '.' (load skips hidden files)")
        _file_name(name, "source")
    for pack in ctx.nodes:
        if pack not in ctx.packs:
            raise ManifestError(f"nodes[{pack!r}]: pack {pack!r} is not one of the scene's "
                                f"packs {sorted(ctx.packs)}")
        _file_name(pack, "nodes pack")
    for key in ctx.skeletons:
        pack, _, piece = key.partition(":")
        if key != piece_key(pack, piece) or not pack or not piece:
            raise ManifestError(f"skeletons[{key!r}]: a skeleton is keyed by "
                                "piece_key(pack, piece)")
        _file_name(pack, "skeleton pack")
        _file_name(piece, "skeleton piece")
    for name in ctx.rules:
        _rules_name(name, f"rules[{name!r}]: the file name")


def _write_scene(ctx: SceneCtx, dir: Path) -> None:
    _write_json(dir / "style.json", {"units_per_metre": ctx.style.units_per_metre,
                                     "page_camera": ctx.style.page_camera,
                                     "dials": ctx.style.dials})
    for name, pack in ctx.packs.items():
        write_pack(pack, dir / "packs" / f"{name}.pack.json")
    _write_json(dir / "clips.json", clips_json(ctx.clips))
    _write_data(ctx, dir)


def _write_data(ctx: SceneCtx, dir: Path) -> None:
    """The scene's pieces, samples and rigs: everything the manifests do not hold."""
    for key, geom in ctx.pieces.items():
        pack, _, piece = key.partition(":")
        folder = dir / "pieces" / pack
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{piece}.npz",
                 **{part: _tris(key, part, tris) for part, tris in geom.parts.items()})
    for name, s in ctx.samples.items():
        arrays = {"times": np.asarray(s.times, dtype=float)}
        for prefix, group in (("g", s.groups), ("b", s.bones), ("p", s.props)):
            arrays.update({f"{prefix}:{k}": np.asarray(v, dtype=float) for k, v in group.items()})
        arrays.update(shell_arrays(s.shells, s.faces, s.labels))
        folder = dir / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{name}.npz", **arrays)
    for key, rig in ctx.rigs.items():
        pack, _, piece = key.partition(":")
        raw = {
            "order": list(rig.order),
            "parent": dict(rig.parent),
            "rest": {b: np.asarray(m, dtype=float).tolist() for b, m in rig.rest.items()},
            "world": np.asarray(rig.world, dtype=float).tolist(),
            "effector_mid": {b: _vec(v) for b, v in rig.effector_mid.items()},
            "clips": sorted(c for c, r in ctx.rig_of_clip.items() if r == key),
        }
        rest = ctx.rests.get(key)
        if rest is not None:
            raw["vertex_groups"] = {part: list(g) for part, g in rest.vertex_groups.items()}
        _write_json(dir / "rigs" / pack / f"{piece}.json", raw)
        if rest is not None:
            np.savez_compressed(
                dir / "rigs" / pack / f"{piece}{REST_SUFFIX}",
                **shell_arrays(rest.shells, rest.faces, rest.labels, rest.weights,
                               rest.weight_groups))
    for name, layout in ctx.layouts.items():
        write_layout(layout, dir / "layouts" / f"{name}.json")
    for name, text in ctx.sources.items():
        path = dir / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")  # byte for byte
    for pack, nodes in ctx.nodes.items():
        _write_json(dir / "nodes" / f"{pack}.json", {"nodes": nodes_to_json(nodes)})
    for key, skeleton in ctx.skeletons.items():
        pack, _, piece = key.partition(":")
        if (p := ctx.packs.get(pack)) and piece in p.pieces and p.pieces[piece].cards_of:
            continue  # a card piece borrows its source's skeleton on load; it has no file
        folder = dir / "pieces" / pack
        folder.mkdir(parents=True, exist_ok=True)
        save_skeleton(skeleton, folder / f"{piece}{SKELETON_SUFFIX}")
    for name, rules in ctx.rules.items():  # the file's own bytes, so the hash is unchanged
        target = dir / RULES_DIR / name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(rules.path, target)
        except OSError as exc:
            raise ManifestError(f"rules[{name!r}]: cannot copy {rules.path}: {exc}") from exc


def _tris(where: str, part: str, tris) -> np.ndarray:
    tris = np.asarray(tris, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ManifestError(f"{where}: part {part!r} must be (N, 3, 3) triangles, "
                            f"got {tris.shape}")
    return tris


def shell_arrays(shells, faces, labels, weights=None, weight_groups=None
                 ) -> dict[str, np.ndarray]:
    """The `v:`, `f:` and `l:` arrays of a body, and its `w:` and `j:` skin arrays when it has
    them (a rest file does, a sample file does not), in the dtypes the files hold."""
    arrays: dict[str, np.ndarray] = {}
    for prefix, group, dtype in (("v", shells, np.float32), ("f", faces, np.int32),
                                 ("l", labels, np.int32), ("w", weights or {}, np.float32),
                                 ("j", weight_groups or {}, np.int32)):
        arrays.update({f"{prefix}:{k}": np.asarray(v, dtype=dtype) for k, v in group.items()})
    return arrays


def _check_shells(where: str, arrays: dict[str, np.ndarray], n: int) -> None:
    """Each part with a shell key has all three, `v` over `n` samples, faces inside `v`'s vertex
    count and one label per vertex. (Whether a label names a group needs the rig.)"""
    parts = sorted({k[2:] for k in arrays if k[:2] in ("v:", "f:", "l:")})
    for part in parts:
        missing = [f"{p}:{part}" for p in "vfl" if f"{p}:{part}" not in arrays]
        if missing:
            raise ManifestError(f"{where}: part {part!r} lacks {missing}; a body part holds "
                                "v:, f: and l: together")
        v, f, lab = arrays[f"v:{part}"], arrays[f"f:{part}"], arrays[f"l:{part}"]
        if v.ndim != 3 or v.shape[0] != n or v.shape[2] != 3:
            raise ManifestError(f"{where}: v:{part} must have shape ({n}, V, 3), got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
            raise ManifestError(f"{where}: f:{part} must be (T, 3) integers, got {f.shape} "
                                f"of {f.dtype}")
        if lab.ndim != 1 or lab.shape[0] != v.shape[1] or not np.issubdtype(lab.dtype, np.integer):
            raise ManifestError(f"{where}: l:{part} must be ({v.shape[1]},) integers, one per "
                                f"vertex of v:{part}, got {lab.shape} of {lab.dtype}")
        if f.size and (f.min() < 0 or f.max() >= v.shape[1]):
            raise ManifestError(f"{where}: f:{part} names vertices outside 0..{v.shape[1] - 1} "
                                f"(its range is {f.min()}..{f.max()})")


def _check_labels(where: str, part: str, labels: np.ndarray, groups) -> None:
    """Each label indexes a group of the rig's `vertex_groups[part]`."""
    if labels.size and (labels.min() < 0 or labels.max() >= len(groups)):
        raise ManifestError(f"{where}: l:{part} names a vertex group outside the rig's "
                            f"{len(groups)} for that part {list(groups)} (its range is "
                            f"{labels.min()}..{labels.max()})")


def _check_samples(where: str, arrays: dict[str, np.ndarray]) -> None:
    times = arrays.get("times")
    if times is None or times.ndim != 1:
        raise ManifestError(f"{where}: samples need a 1-D 'times' array")
    check_times(times, where)
    n = len(times)
    for key, arr in arrays.items():
        if key == "times":
            continue
        prefix = key[:2]
        if prefix == "g:":
            ok = arr.ndim == 3 and arr.shape[0] == n and arr.shape[2] == 3
            want = f"({n}, V, 3)"
        elif prefix in ("b:", "p:"):
            ok = arr.shape == (n, 4, 4)
            want = f"({n}, 4, 4)"
        elif prefix in ("v:", "f:", "l:"):
            continue  # checked together below
        else:
            raise ManifestError(f"{where}: unknown sample key {key!r}")
        if not ok:
            raise ManifestError(f"{where}: {key} must have shape {want}, got {arr.shape}")
    _check_shells(where, arrays, n)


def _check_body(samples: dict[str, ClipSamples], rig_of_clip: dict[str, str],
                rests: dict[str, RigRest], dir: Path) -> None:
    """A clip posed by a rig with rest shells carries the body of exactly the rest's parts, with
    the rest's faces and labels (the rig fixes both), the rest's vertex counts and labels that
    name a vertex group. Without rest shells a clip carries none."""
    for name, s in samples.items():
        where = str(dir / "samples" / f"{name}.npz")
        key = rig_of_clip.get(name)
        rest = rests.get(key) if key is not None else None
        if rest is None:
            if s.shells:
                raise ManifestError(f"{where}: the clip carries body shells, but its rig "
                                    f"{key!r} holds no vertex_groups and rest shells")
            continue
        if set(s.shells) != set(rest.vertex_groups):
            raise ManifestError(
                f"{where}: the clip's body parts {sorted(s.shells)} differ from the parts "
                f"{sorted(rest.vertex_groups)} of rig {key!r}: missing "
                f"{sorted(set(rest.vertex_groups) - set(s.shells))}, extra "
                f"{sorted(set(s.shells) - set(rest.vertex_groups))}")
        for part, shell in s.shells.items():
            have, want = shell.shape[1], rest.shells[part].shape[1]
            if have != want:
                raise ManifestError(f"{where}: v:{part} has {have} vertices, the rig's rest "
                                    f"shell has {want}")
            _check_labels(where, part, s.labels[part], rest.vertex_groups[part])
            for prefix, got, fixed in (("f", s.faces[part], rest.faces[part]),
                                       ("l", s.labels[part], rest.labels[part])):
                if got.shape != fixed.shape or not np.array_equal(got, fixed):
                    raise ManifestError(f"{where}: {prefix}:{part} differs from the rig's rest "
                                        f"{prefix}:{part}; faces and labels are the rig's, the "
                                        "same at every sample")


def _check_rest_parts(path: Path, packs: dict[str, Pack], key: str, rest: RigRest) -> None:
    """The rest shells (and `vertex_groups`) are for exactly the parts of the rig's piece."""
    pack_name, _, piece = key.partition(":")
    want = {p.name for p in packs[pack_name].pieces[piece].parts}
    if set(rest.vertex_groups) != want:
        raise ManifestError(
            f"{path}: the rest shells are for parts {sorted(rest.vertex_groups)}, but piece "
            f"{key} has parts {sorted(want)}: missing {sorted(want - set(rest.vertex_groups))}, "
            f"extra {sorted(set(rest.vertex_groups) - want)}")


def _rig_json(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path}: cannot read the rig file: {exc!r}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: cannot read the rig file: not a JSON object")
    return raw


def _read_vertex_groups(path: Path, raw: dict, rig: RigData
                        ) -> dict[str, tuple[str, ...]] | None:
    """The `vertex_groups` of a rig file's JSON `raw`: per part, distinct bone names of the rig.
    None when the file has no such key."""
    if "vertex_groups" not in raw:
        return None
    found = raw["vertex_groups"]
    if not isinstance(found, dict):
        raise ManifestError(f"{path}: vertex_groups must map a part name to a list of names")
    out: dict[str, tuple[str, ...]] = {}
    for part, names in found.items():
        if (not isinstance(names, list) or not all(isinstance(n, str) for n in names)
                or len(set(names)) != len(names)):
            raise ManifestError(f"{path}: vertex_groups[{part!r}] must be a list of distinct "
                                "group names")
        unknown = [n for n in names if n not in rig.parent]
        if unknown:
            raise ManifestError(f"{path}: vertex_groups[{part!r}] names {unknown}, which are "
                                "no bones of the rig")
        out[str(part)] = tuple(names)
    return out


def _read_rest(path: Path, groups: dict[str, tuple[str, ...]]) -> RigRest:
    try:
        with np.load(path) as npz:
            arrays = {k: npz[k] for k in npz.files}
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path}: cannot read the rest shells: {exc!r}") from exc
    for key in arrays:
        if key[:2] not in ("v:", "f:", "l:", "w:", "j:"):
            raise ManifestError(f"{path}: unknown rest key {key!r}; a rest file holds v:, f:, "
                                "l:, w: and j: keys")
    _check_shells(str(path), arrays, 1)
    parts = {k[2:] for k in arrays if k[:2] in ("v:", "f:", "l:")}
    if parts != set(groups):
        raise ManifestError(f"{path}: the rest shells are for parts {sorted(parts)}, but the "
                            f"rig file's vertex_groups are for {sorted(groups)}")
    for part, g in groups.items():
        _check_labels(str(path), part, arrays[f"l:{part}"], g)
    _check_skin(str(path), arrays, groups)
    return RigRest(
        shells={p: arrays[f"v:{p}"].astype(np.float32) for p in sorted(parts)},
        faces={p: arrays[f"f:{p}"].astype(np.int32) for p in sorted(parts)},
        labels={p: arrays[f"l:{p}"].astype(np.int32) for p in sorted(parts)},
        vertex_groups=dict(groups),
        weights={p: arrays[f"w:{p}"].astype(np.float32) for p in sorted(parts)
                 if f"w:{p}" in arrays},
        weight_groups={p: arrays[f"j:{p}"].astype(np.int32) for p in sorted(parts)
                       if f"j:{p}" in arrays})


def _check_skin(where: str, arrays: dict[str, np.ndarray],
                groups: dict[str, tuple[str, ...]]) -> None:
    """The optional skin arrays of a rest file: every part has `w:` and `j:` together, or no
    part has either (a file from before skin weights were stored). Both are (V, K) over the part's V vertices,
    `w:` finite and not negative, `j:` an index into the part's `vertex_groups`."""
    skinned = {k[2:] for k in arrays if k[:2] in ("w:", "j:")}
    if not skinned:
        return
    for part in sorted(skinned - set(groups)):
        raise ManifestError(f"{where}: w:/j: for part {part!r}, which has no shell")
    for part, names in groups.items():
        missing = [f"{p}:{part}" for p in "wj" if f"{p}:{part}" not in arrays]
        if len(missing) == 2:
            raise ManifestError(f"{where}: part {part!r} has no w: and j:, but other parts do; "
                                "skin weights cover every part or none")
        if missing:
            raise ManifestError(f"{where}: {missing[0]} is missing; a part holds w: and j: "
                                "together")
        w, j = arrays[f"w:{part}"], arrays[f"j:{part}"]
        n = arrays[f"v:{part}"].shape[1]
        if w.ndim != 2 or w.shape[0] != n or not np.issubdtype(w.dtype, np.floating):
            raise ManifestError(f"{where}: w:{part} must have shape ({n}, K) of floats, one row "
                                f"per vertex of v:{part}, got {w.shape} of {w.dtype}")
        if j.shape != w.shape or not np.issubdtype(j.dtype, np.integer):
            raise ManifestError(f"{where}: j:{part} must have the shape of w:{part} {w.shape} "
                                f"and hold integers, got {j.shape} of {j.dtype}")
        if w.size and not (np.isfinite(w).all() and w.min() >= 0):
            raise ManifestError(f"{where}: w:{part} must be finite and not negative (its range "
                                f"is {w.min()}..{w.max()})")
        if j.size and (j.min() < 0 or j.max() >= len(names)):
            raise ManifestError(f"{where}: j:{part} names a vertex group outside the rig's "
                                f"{len(names)} for that part {list(names)} (its range is "
                                f"{j.min()}..{j.max()})")


def _read_rig(path: Path) -> tuple[RigData, list[str]]:
    return _rig_from(path, _rig_json(path))


def _rig_from(path: Path, raw: dict) -> tuple[RigData, list[str]]:
    """The rig and the clips it poses, from the JSON `raw` of the rig file at `path`."""
    try:
        rig = RigData(
            order=tuple(raw["order"]),
            parent=dict(raw["parent"]),
            rest={b: np.asarray(m, dtype=float) for b, m in raw["rest"].items()},
            world=np.asarray(raw["world"], dtype=float),
            effector_mid={b: np.asarray(v, dtype=float) for b, v in raw["effector_mid"].items()},
        )
        clips = list(raw["clips"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ManifestError(f"{path}: cannot read the rig file: {exc!r}") from exc
    return rig, clips


def _check_parts(path: Path, piece: Piece, geom: PieceGeom) -> None:
    """Each geometry key is a part name, and every hold and seat surface part has geometry."""
    for part in geom.parts:
        if not _PART.fullmatch(part):
            raise ManifestError(f"{path}: {part!r} is not a part name (letters, digits, '_' or "
                                "'-'); an exporter must key parts by the pack's part names, "
                                "not by Blender object names")
    for kind, items in (("hold", piece.holds), ("seat", piece.seats)):
        for item in items:
            missing = [p for p in item.surface if p not in geom.parts]
            if missing:
                raise ManifestError(f"{path}: {kind} {item.id!r} names surface parts "
                                    f"{missing} that have no geometry; the file holds "
                                    f"{sorted(geom.parts)}")


def _rig_key(path: Path, packs: dict[str, Pack], rig: RigData) -> str:
    """The key of the rigged piece a rig file belongs to: `rigs/<pack>/<piece>.json` names the
    pack (a loaded one) and the piece (a rigged one of it). The bones must match the pack's."""
    pack_name, piece = path.parent.name, path.stem
    pack = packs.get(pack_name)
    if pack is None:
        raise ManifestError(f"{path}: the folder names pack {pack_name!r}, which is not loaded "
                            f"(loaded: {sorted(packs)}); rig files live at "
                            "rigs/<pack>/<piece>.json")
    if piece not in pack.pieces or pack.pieces[piece].rig is None:
        raise ManifestError(f"{path}: {piece!r} is no rigged piece of pack {pack_name!r}; "
                            "rig files live at rigs/<pack>/<piece>.json")
    spec = pack.pieces[piece].rig
    want = {b.name: b.parent for b in spec.bones}
    got = {b: rig.parent[b] for b in rig.order}
    if want != got:
        extra, lacking = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        moved = sorted(b for b in set(got) & set(want) if got[b] != want[b])
        raise ManifestError(f"{path}: the rig file disagrees with {pack_name}:{piece}'s rig: "
                            f"bones only in the file {extra}, only in the pack {lacking}, "
                            f"with another parent {moved}")
    return piece_key(pack_name, piece)


def _rig_files(folder: Path) -> list[Path]:
    """The rig files under `rigs/`: exactly `rigs/<pack>/<piece>.json`, plus the rest shells
    `rigs/<pack>/<piece>.rest.npz` beside them (`_rest_files` lists those). A file or folder
    whose name starts with "." is ignored; anything else raises."""
    found = []
    for path in sorted(folder.rglob("*")):
        parts = path.relative_to(folder).parts
        if any(part.startswith(".") for part in parts):
            continue  # a hidden file or folder, such as macOS's .DS_Store, is no rig file
        depth = len(parts)
        if path.is_dir() and depth == 1:
            continue
        if depth == 2 and path.is_file() and path.suffix == ".json":
            found.append(path)
            continue
        if depth == 2 and path.is_file() and path.name.endswith(REST_SUFFIX):
            continue
        why = ("a rig file at the top of rigs/" if depth == 1 and path.suffix == ".json"
               else "not a rig file")
        raise ManifestError(f"{path}: {why}; rig files live at rigs/<pack>/<piece>.json")
    return found


def _rest_files(folder: Path) -> list[Path]:
    """The rest shell files under `rigs/`; `_rig_files` has already refused anything else."""
    return [path for path in sorted(folder.glob(f"*/*{REST_SUFFIX}"))
            if not path.parent.name.startswith(".") and not path.name.startswith(".")]


def load_scene(dir: Path, *, bake: bool = True) -> SceneCtx:
    """Read a scene folder written by `save_scene`. Every file is checked as it loads. With
    `bake=False` the `bake/` folder is not read (`SceneCtx.bake` is None): the bake's own planner
    reads the scene that way, so a half-written or stale bake never stops a new one."""
    dir = Path(dir)
    style = load_style(dir / "style.json")
    packs: dict[str, Pack] = {}
    for path in sorted((dir / "packs").glob("*.pack.json")):
        pack = load_pack(path, style)
        if path.name != f"{pack.name}.pack.json":
            raise ManifestError(f"{path}: holds pack {pack.name!r}; the file name must match")
        packs[pack.name] = pack
    clips = {clip.name: clip for clip in load_clips(dir / "clips.json")}
    check_seats(packs, clips)
    pieces: dict[str, PieceGeom] = {}
    for path in sorted((dir / "pieces").glob("*/*.npz")):
        if path.name.endswith(SKELETON_SUFFIX):
            continue  # a grown skeleton, read below
        pack = packs.get(path.parent.name)
        if pack is None:
            raise ManifestError(f"{path}: geometry for pack {path.parent.name!r}, which is "
                                "not loaded")
        if path.stem not in pack.pieces:
            raise ManifestError(f"{path}: geometry for piece {path.stem!r}, which pack "
                                f"{pack.name!r} lacks")
        key = piece_key(path.parent.name, path.stem)
        with np.load(path) as npz:
            pieces[key] = PieceGeom(path.stem, {p: _tris(str(path), p, npz[p]) for p in npz.files})
        _check_parts(path, pack.pieces[path.stem], pieces[key])
    skeletons = dict(_read_skeletons(dir / "pieces", packs))
    _alias_card_skeletons(skeletons, packs)
    samples: dict[str, ClipSamples] = {}
    for path in sorted((dir / "samples").glob("*.npz")):
        with np.load(path) as npz:
            arrays = {k: npz[k] for k in npz.files}
        _check_samples(str(path), arrays)
        if path.stem not in clips:
            raise ManifestError(f"{path}: samples for clip {path.stem!r}, which clips.json lacks")
        samples[path.stem] = ClipSamples(
            clip=path.stem, times=arrays["times"],
            groups={k[2:]: v for k, v in arrays.items() if k.startswith("g:")},
            bones={k[2:]: v for k, v in arrays.items() if k.startswith("b:")},
            props={k[2:]: v for k, v in arrays.items() if k.startswith("p:")},
            shells={k[2:]: v.astype(np.float32) for k, v in arrays.items()
                    if k.startswith("v:")},
            faces={k[2:]: v.astype(np.int32) for k, v in arrays.items() if k.startswith("f:")},
            labels={k[2:]: v.astype(np.int32) for k, v in arrays.items() if k.startswith("l:")},
        )
    rigs: dict[str, RigData] = {}
    rests: dict[str, RigRest] = {}
    rig_of_clip: dict[str, str] = {}
    for path in _rig_files(dir / "rigs"):
        raw = _rig_json(path)
        rig, posed = _rig_from(path, raw)
        key = _rig_key(path, packs, rig)
        rigs[key] = rig
        groups = _read_vertex_groups(path, raw, rig)
        rest_path = path.with_name(f"{path.stem}{REST_SUFFIX}")
        if groups is not None:
            if not rest_path.is_file():
                raise ManifestError(f"{path}: holds vertex_groups, but its rest shells "
                                    f"{rest_path.name} are missing")
            rests[key] = _read_rest(rest_path, groups)
            _check_rest_parts(rest_path, packs, key, rests[key])
        elif rest_path.is_file():
            raise ManifestError(f"{rest_path}: rest shells whose rig file {path.name} holds no "
                                "vertex_groups")
        for clip in posed:
            if clip not in clips:
                raise ManifestError(f"{path}: poses clip {clip!r}, which clips.json lacks")
            if clip in rig_of_clip:
                raise ManifestError(f"{path}: clip {clip!r} is posed by two rigs")
            rig_of_clip[clip] = key
    for path in _rest_files(dir / "rigs"):
        if piece_key(path.parent.name, path.name.removesuffix(REST_SUFFIX)) not in rigs:
            raise ManifestError(f"{path}: rest shells with no rig file "
                                f"{path.name.removesuffix(REST_SUFFIX)}.json beside them")
    _check_body(samples, rig_of_clip, rests, dir)
    check_clip_rigs(packs, clips, rig_of_clip)
    if (dir / "layouts").is_dir():
        layouts = load_layouts(dir / "layouts", packs)  # also checks the connector ends
    else:
        layouts = {}
        check_connectors(packs, layouts)  # a connector then has no layout to name
    check_stands(layouts, clips)
    sources = {path.name: _read_source(path) for path in sorted((dir / "sources").glob("*.py"))
               if not path.name.startswith(".")}
    nodes = {path.stem: _read_nodes(path, packs) for path in sorted((dir / "nodes").glob("*.json"))
             if not path.name.startswith(".")}
    rules = {path.name: load_rules(path, kit=KIT_RULES)
             for path in sorted((dir / RULES_DIR).glob(f"*{RULES_SUFFIX}"))
             if not path.name.startswith(".")}
    return SceneCtx(style=style, packs=packs, clips=clips, pieces=pieces, samples=samples,
                    rigs=rigs, rig_of_clip=rig_of_clip, layouts=layouts, sources=sources,
                    nodes=nodes, rests=rests, skeletons=skeletons, folder=dir.name, rules=rules,
                    bake=load_bake(dir) if bake else None)


def _alias_card_skeletons(skeletons: dict[str, Skeleton], packs: dict[str, Pack]) -> None:
    """Each card piece shares its source piece's skeleton: its key maps to the source's. A card
    piece with a skeleton file of its own is refused when the files are read."""
    for pack in packs.values():
        for name, piece in pack.pieces.items():
            if not piece.cards_of:
                continue
            key, src = piece_key(pack.name, name), piece_key(pack.name, piece.cards_of)
            if src in skeletons:
                skeletons[key] = skeletons[src]


def _read_skeletons(folder: Path, packs: dict[str, Pack]) -> list[tuple[str, Skeleton]]:
    """The skeleton files under `pieces/`. Each belongs to a branching piece of a loaded pack, and
    its `info["piece"]` names that piece."""
    found = []
    for path in sorted(folder.glob(f"*/*{SKELETON_SUFFIX}")):
        if path.name.startswith("."):
            continue
        piece = path.name[: -len(SKELETON_SUFFIX)]
        pack = packs.get(path.parent.name)
        if pack is None:
            raise ManifestError(f"{path}: a skeleton for pack {path.parent.name!r}, which is "
                                "not loaded")
        if piece not in pack.pieces:
            raise ManifestError(f"{path}: a skeleton for {piece_key(pack.name, piece)}, a piece "
                                f"that pack {pack.name!r} lacks")
        kind = pack.pieces[piece].recipe.kind
        if kind != "branching":
            raise ManifestError(f"{path}: a skeleton for {piece_key(pack.name, piece)}, an "
                                f"{kind} piece; only a branching piece has one")
        if source := pack.pieces[piece].cards_of:
            raise ManifestError(f"{path}: {piece_key(pack.name, piece)} is a card copy of "
                                f"{piece_key(pack.name, source)} and shares its skeleton; "
                                "delete this file")
        skeleton = load_skeleton(path)
        named = skeleton.info.get("piece")
        if named != piece:
            raise ManifestError(f"{path}: the file is for piece {piece!r}, but its skeleton "
                                f"says it is for piece {named!r}")
        found.append((piece_key(pack.name, piece), skeleton))
    return found


def _read_source(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", newline="") as f:  # no newline translation
            return f.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{path}: cannot read the build script: {exc}") from exc


def _read_nodes(path: Path, packs: dict[str, Pack]) -> list[GlbNode]:
    if path.stem not in packs:
        raise ManifestError(f"{path}: a node table for pack {path.stem!r}, which is not loaded "
                            f"(loaded: {sorted(packs)})")
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path}: cannot read the node table: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"nodes"}:
        raise ManifestError(f"{path}: the node table file must be an object holding 'nodes'")
    return nodes_from_json(path, raw["nodes"])
