"""Check export-hygiene: the packed .glb's node table agrees with the pack that made it.

A scene holds the node table of each pack's .glb (`ctx.nodes`, from `nodes/<pack>.json`). For
each pack that has one:

- **duplicate-name.** No two nodes share a name (`<pack> <node> duplicate-name`). The page finds
  props by node name, so a second node of a name hides the first.
- **root-offset.** Each piece's root node (the parentless node named for the piece) stands where
  the pack says: on the pack's `spacing` grid (`x` a whole number of spacings, `y` and `z` zero),
  or, when `spacing` is null, at the position of a layout placement of that piece. A piece that
  any clip carries as a prop has its root at the origin. A piece with no root node fails too.
  The tolerance is `NODE_OF_PIECE_SIZE` times the piece's size (the diagonal of the box of its
  parts): a position is exact up to float32 noise, so the pack's wall thickness is the wrong
  scale. Each root adds `NODE_FLOAT32_ULPS` float32 epsilons of its distance from the origin,
  the rounding a far-out root carries. A piece with no geometry has no size, and its root is
  `Unmeasured`. Item
  `<pack>:<piece> root-offset`.
- **no-node, not-a-part.** Every part of the pack has a mesh node named `<piece>/<part>`
  (`glb.node_name`), and every mesh node is a part of the pack. A part with no node is
  `<pack>:<piece> <part> no-node`; a mesh node no part owns is `<pack> <node> not-a-part`. A
  rename after the pack was written shows as both. A node with no mesh (a pivot, a bone) need not
  be a part.

Nodes are read by their place in the table, never by name, so two nodes of one name are both
looked at: a duplicate root is checked at each of its places. A pack in `ctx.packs` with no node
table is `Unmeasured` ("no node table"), and so is a piece with no geometry ("no geometry"),
and a piece of a pack with no spacing that no layout places and no clip carries ("no layout
places it and no clip carries it").

The plants are one unit per rule and pack, built from the node table and the pack alone, never
from the findings: a second node copied from the first node; a root of the first piece that has
one, a size, and a grid, a placement or a clip carrying it (the rule judges those), moved
`CALIBRATION_FAIL` times the tolerance, along x, then y, then z; the first part's mesh
node taken out; the first part's mesh node renamed (or, when no part owns one, the first mesh
node). The gate skips a unit whose item already fails, and stops the run when a unit's plant is
not caught. A rule with nothing in the table to plant on (a pack with no node, no piece with a
root and a size, no mesh node) has no unit.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Collection

import numpy as np

from model_kit.core.glb import GlbNode, node_name
from model_kit.core.manifest import Pack
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx, part_key, piece_key
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    FLOAT32_EPS,
    NODE_FLOAT32_ULPS,
    NODE_OF_PIECE_SIZE,
)

NAME = "export-hygiene"


def _limit(ctx: SceneCtx, pack: str, piece: str) -> float | None:
    """How far a root of this piece may sit from its place, or None when the scene holds no
    triangles for the piece (it has no size)."""
    geom = ctx.pieces.get(piece_key(pack, piece))
    drawn = [np.asarray(t, dtype=float).reshape(-1, 3) for t in (geom.parts.values() if geom
                                                                  else ()) if len(t)]
    if not drawn:
        return None
    every = np.concatenate(drawn)
    return NODE_OF_PIECE_SIZE * float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))


def _at(limit: float, translation) -> float:
    """The limit for one root: the piece's limit plus the float32 rounding its distance from the
    origin carries."""
    return limit + NODE_FLOAT32_ULPS * FLOAT32_EPS * float(np.linalg.norm(translation))


def _carried(ctx: SceneCtx, pack: str, piece: str) -> bool:
    return any((p.pack, p.piece) == (pack, piece) for clip in ctx.clips.values()
               for p in clip.props.values())


def _placed(ctx: SceneCtx, pack: str, piece: str) -> list[tuple[float, float, float]]:
    return [p.position for layout in ctx.layouts.values() for p in layout.placements.values()
            if (p.pack, p.piece) == (pack, piece)]


def _off_grid(at: np.ndarray, spacing: float) -> float:
    """How far a root is from the nearest point of the grid: x on a multiple of the spacing,
    y and z zero."""
    slot = spacing * round(float(at[0]) / spacing)
    return float(np.linalg.norm(at - np.array([slot, 0.0, 0.0])))


def _root_problems(ctx: SceneCtx, pack: Pack, name: str, at: np.ndarray, limit: float
                   ) -> list[str]:
    out = []
    if pack.spacing is not None:
        gap = _off_grid(at, pack.spacing)
        if gap > limit:
            out.append(f"root at {np.round(at, 4).tolist()} is {gap:.4f} off the "
                       f"{pack.spacing:g} grid (limit {limit:.4f})")
    else:
        places = _placed(ctx, pack.name, name)
        if places:
            gap = min(float(np.linalg.norm(at - np.asarray(p))) for p in places)
            if gap > limit:
                out.append(f"root at {np.round(at, 4).tolist()} is {gap:.4f} from where the "
                           f"layout places the piece (limit {limit:.4f})")
    if _carried(ctx, pack.name, name):
        gap = float(np.linalg.norm(at))
        if gap > limit:
            out.append(f"a clip carries the piece, so its root belongs at the origin, but it "
                       f"stands {gap:.4f} from it (limit {limit:.4f})")
    return out


def _pack_findings(ctx: SceneCtx, pack: Pack, nodes: list[GlbNode]) -> list[Finding]:
    out: list[Finding] = []
    counts = Counter(n.name for n in nodes)
    for name in sorted(n for n, c in counts.items() if c > 1):
        out.append(Finding(NAME, f"{pack.name} {name} duplicate-name",
                           f"{counts[name]} nodes are named {name!r}"))
    for name in pack.pieces:
        item = f"{piece_key(pack.name, name)} root-offset"
        limit = _limit(ctx, pack.name, name)
        roots = [i for i, n in enumerate(nodes) if n.name == name and n.parent is None]
        if not roots:
            out.append(Finding(NAME, item, f"the table has no root node (a parentless node "
                                           f"named {name!r})"))
            continue
        if limit is None:
            continue  # no size to judge a position by: unmeasured, and coverage says so
        problems = [why for i in roots for why in _root_problems(
            ctx, pack, name, np.asarray(nodes[i].translation, dtype=float),
            _at(limit, nodes[i].translation))]
        if problems:
            out.append(Finding(NAME, item, "; ".join(dict.fromkeys(problems))))
    meshes = {n.name for n in nodes if n.mesh is not None}
    owned = {node_name(name, part.name): part_key(pack.name, name, part.name)
             for name, piece in pack.pieces.items() for part in piece.parts}
    for node, key in owned.items():
        if node not in meshes:
            out.append(Finding(NAME, f"{key} no-node", f"no mesh node named {node!r}"))
    for name in sorted(meshes - set(owned)):
        out.append(Finding(NAME, f"{pack.name} {name} not-a-part",
                           f"the mesh node {name!r} is no part of pack {pack.name!r}"))
    return out


def _all_findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    out = []
    for pack_name, nodes in ctx.nodes.items():
        pack = ctx.packs.get(pack_name)
        if pack is None:
            continue
        out += [f for f in _pack_findings(ctx, pack, nodes)
                if only is None or f.item in only]
    return out


# --- plants -------------------------------------------------------------------------------------


def _with_nodes(ctx: SceneCtx, pack: str, nodes: list[GlbNode]) -> SceneCtx:
    return dataclasses.replace(ctx, nodes={**ctx.nodes, pack: nodes})


def _plant_units(ctx: SceneCtx, pack: Pack, nodes: list[GlbNode]) -> list[PlantUnit]:
    """The units of one pack, one per rule that has a node to plant on."""
    out: list[PlantUnit] = []

    def unit(rule: str, item: str, *planted: list[GlbNode]) -> None:
        out.append(PlantUnit(item, tuple(Plant(f"{rule}: {item}", item,
                                               _with_nodes(ctx, pack.name, table))
                                         for table in planted)))

    if nodes:
        unit("duplicate-name", f"{pack.name} {nodes[0].name} duplicate-name", [*nodes, nodes[0]])
    for name in pack.pieces:
        limit = _limit(ctx, pack.name, name)
        root = next((i for i, n in enumerate(nodes) if n.name == name and n.parent is None), None)
        # a piece with no grid, no placement and no clip is not judged: coverage says so
        judged = (pack.spacing is not None or _placed(ctx, pack.name, name)
                  or _carried(ctx, pack.name, name))
        if limit is None or root is None or not judged:
            continue
        moved = []
        for axis in range(3):
            at = list(nodes[root].translation)
            at[axis] += CALIBRATION_FAIL * _at(limit, nodes[root].translation)
            moved.append([*nodes[:root], dataclasses.replace(nodes[root], translation=tuple(at)),
                          *nodes[root + 1:]])
        unit("root-offset", f"{piece_key(pack.name, name)} root-offset", *moved)
        break
    parts = [(name, part.name) for name, piece in pack.pieces.items() for part in piece.parts]
    if parts:
        name, part = parts[0]
        full = node_name(name, part)
        unit("no-node", f"{part_key(pack.name, name, part)} no-node",
             [n for n in nodes if not (n.name == full and n.mesh is not None)])
    owned = {node_name(name, part) for name, part in parts}
    first = next((i for i, n in enumerate(nodes) if n.mesh is not None and n.name in owned),
                 next((i for i, n in enumerate(nodes) if n.mesh is not None), None))
    if first is not None:
        renamed = f"{nodes[first].name}_renamed"
        unit("not-a-part", f"{pack.name} {renamed} not-a-part",
             [*nodes[:first], dataclasses.replace(nodes[first], name=renamed), *nodes[first + 1:]])
    return out


class ExportHygiene:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a pack check: the packed file, not a recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [u for pack_name, nodes in ctx.nodes.items() if (pack := ctx.packs.get(pack_name))
                for u in _plant_units(ctx, pack, nodes)]

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _all_findings(ctx, only)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        for pack_name, pack in ctx.packs.items():
            if pack_name not in ctx.nodes:
                lines.append(Unmeasured(f"{NAME}: skipped (no node table): {pack_name}"))
                continue
            lines += [Unmeasured(f"{NAME}: skipped (no geometry): {piece_key(pack_name, name)}")
                      for name in pack.pieces if _limit(ctx, pack_name, name) is None]
            if pack.spacing is None:
                lines += [Unmeasured(f"{NAME}: skipped (no layout places it and no clip carries "
                                     f"it): {piece_key(pack_name, name)}")
                          for name in pack.pieces
                          if not _placed(ctx, pack_name, name) and not _carried(ctx, pack_name,
                                                                                name)]
        return lines
