"""Check overlap, pose overlap: a posed body, the pieces a clip places and the props it carries never
sink into each other, shell by shell.

The body is the rig's shells: each part of the rigged piece is a mesh whose connected closed
shells (`geom.shells`) each get their own `SurfaceIndex`, because parity `inside()` is wrong over
overlapping closed shells. A shell is named by its part and its strongest vertex group,
`rig:<part>/<group>`. A placed or carried piece is named `<pack>:<piece>`, a part of it
`<pack>:<piece> <part>`. Parts of role `eyes` or `decal` are printed on a surface: they are
neither a vertex source nor a target, in the body or in a piece.

Four tests on every sample of every clip. Each is a vertex inside a closed shell or part, and
its depth is the nearest distance to that surface:

- **A** a body vertex inside a placed or carried piece's part.
- **B** a piece vertex (a carried spoon) inside a body shell.
- **C** a body vertex inside a body shell of another vertex group, unless the pair is excused.
- **D** a carried prop's vertex inside a placed piece's part, and a placed piece's vertex inside
  a carried prop's part.

A depth over the budget is a finding. The budget is `SKIN_OF_EFFECTOR` times the rig's radius
(`supports.body_radius`: the smallest over its chains' effector radii and its supports', the
strictest, so a body with no chain still has one). Zones raise it, from holds and the seat:

- A `grip` hold excuses the holding contact group (the bone groups `contact_groups` names) within
  `(1 + SOLVE_ERR_OF_EFFECTOR)` times the effector radius of the hold point, against the hold's
  piece, in A and B. In A the group is the offending vertex's own; in B it is the groups at the
  three corners of the body face nearest the piece's vertex (a limb is one mesh holding the arm,
  forearm and hand groups, and a wrist face has corners in two). Any other hold allows
  `PRESS_OF_EFFECTOR` times the radius in the same zone, and a hold whose `who` is a carried prop
  allows it for D (a stick tip on a head). A hold counts at the samples of its `when` windows.
- The clip's seat allows `SEAT_OF_EFFECTOR` times the radius for the whole body against the
  seat's surface parts, at the samples of `seat_windows(clip)` (the whole clip by default).

Excused body pairs (C): the pairs of vertex groups that already cross in the rig's rest file (a
vertex of the first group inside a shell of the second), and each deforming bone with its nearest
deforming ancestor (the shoulder, the elbow). A pair is judged as (the vertex's group, the
shell's group), and an excused pair is excused at any depth. A vertex
of a shell's own group is never a C finding.

A carried prop is measured only at the samples inside its `show_frames`. Depth inside a carried
piece is in its own units, so it is multiplied by the pose's scale (the cube root of the
determinant) to give world units.

Items: `<clip> <A|B|C|D> <side> x <side>`; no sample index appears, so one allow line covers a
whole clip. A finding carries the worst excess over the clip, its depth, budget and frame.

The check measures clips, not a recipe, so `kinds` is empty. Plants: one unit per clip and kind,
built from the geometry and the manifest alone. They never read the test that judges them, nor
`_budgets`, which is half of its finding condition. A vertex (A, C) or a piece vertex (B, D) is
moved to a point inside a shell or part, at `ROOM_FRACTIONS` of the room that target has (the
radius of the largest ball inside it), at the first, the middle and the last sample, and the check
must report it. An item that a hold or the seat excuses is planted last, because the manifest
says its contact may be deeper. A kind with no item to judge in the clip is a plain
not-applicable line. A judged item with no fault the check reports stops the gate (`BlindCheck`).
Missing data is `Unmeasured`
too: a clip with no samples, a rig with no rest file, a clip whose samples carry no body shells,
a piece with no geometry, a closed-shell test on an open part. A kind with nothing to measure in a
clip (no carried prop for D) is a plain line, because it does not apply by design.
"""

from __future__ import annotations

import dataclasses
import weakref
from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass, field

import numpy as np

from model_kit.core.checks.entries import first_middle_last, unit
from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.geom import SurfaceIndex, is_closed, shells, signed_volume, weld
from model_kit.core.manifest import Clip, anchor, seat_windows
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import (
    ClipSamples,
    PieceGeom,
    RigRest,
    SceneCtx,
    due,
    effector_radius,
    piece_key,
    rig_spec,
)
from model_kit.core.supports import body_radius
from model_kit.core.targets import piece_frames
from model_kit.core.tolerances import (
    PRESS_OF_EFFECTOR,
    SEAT_OF_EFFECTOR,
    SKIN_OF_EFFECTOR,
    SOLVE_ERR_OF_EFFECTOR,
    WELD_OF_PIECE_SIZE,
)

NAME = "overlap"
TESTS = ("A", "B", "C", "D")
NO_BUDGET_LIMIT = float("inf")  # a grip zone: any depth is excused

# --- work kept between runs ------------------------------------------------------------------

# A plant search runs the clip again for each candidate, and the body's shells, the joints and
# the indexes of a piece's parts do not change between those runs. A value is kept against the
# identity of the object it was built from (a triangle array, a rest file), which is held weakly,
# so a reused id never answers for another object.
_KEPT: dict[int, tuple[weakref.ref, object]] = {}


def _kept(source, build):
    hit = _KEPT.get(id(source))
    if hit is not None and hit[0]() is source:
        return hit[1]
    for stale in [k for k, (ref, _) in _KEPT.items() if ref() is None]:
        del _KEPT[stale]
    value = build()
    _KEPT[id(source)] = (weakref.ref(source), value)
    return value


# --- the body ---------------------------------------------------------------------------------


@dataclass(eq=False)
class Shell:
    part: str
    group: str  # the strongest vertex group, by name
    gid: int  # the group's id in `Body.groups`
    tris: np.ndarray  # (T, 3) global vertex ids
    verts: np.ndarray  # global vertex ids the shell uses
    closed: bool

    @property
    def name(self) -> str:
        return f"rig:{self.part}/{self.group}"


@dataclass(eq=False)
class Body:
    parts: tuple[str, ...]  # the measured body parts (printed ones left out)
    groups: list[str]  # every vertex group name
    vpart: np.ndarray  # (V,) index into `parts`
    vgroup: np.ndarray  # (V,) index into `groups`
    vshell: np.ndarray  # (V,) index into `shells`, -1 for a vertex no face uses
    vlocal: np.ndarray  # (V,) the vertex's index inside its part
    shells: list[Shell]
    excused: np.ndarray  # (G, G) True where (vertex group, shell group) is a joint

    def item(self, v: int) -> str:
        return f"rig:{self.parts[self.vpart[v]]}/{self.groups[self.vgroup[v]]}"

    def positions(self, samples: ClipSamples, s: int) -> np.ndarray:
        return np.concatenate([samples.shells[p][s] for p in self.parts]).astype(float)


class _Indexes:
    """The shell indexes of one pose, built when a test first asks."""

    def __init__(self, body: Body, pos: np.ndarray) -> None:
        self.body, self.pos = body, pos
        self._made: dict[int, SurfaceIndex] = {}

    def __call__(self, k: int) -> SurfaceIndex:
        if k not in self._made:
            self._made[k] = SurfaceIndex(self.pos[self.body.shells[k].tris])
        return self._made[k]


def hits(index: SurfaceIndex, pts: np.ndarray
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rows of the points inside the closed index, their depth: the nearest distance to it, and
    the index's nearest face for each)."""
    if len(pts) == 0:
        return np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=int)
    rows = np.nonzero(index.inside(pts))[0]
    if len(rows) == 0:
        return rows, np.zeros(0), np.zeros(0, dtype=int)
    near = index.closest(pts[rows])
    return rows, near.dist, near.face


def part_contacts(local: np.ndarray, part: PiecePart
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The points of `local` (N, 3, in the piece's frame) that lie inside the closed `part`: (their
    rows, their depth, the part's nearest face for each). The one scan of a body vertex in a piece
    part, shared by test A and by coverage's contact rule, which asks the same question without a
    budget. A part with no index (an open one) has no inside: callers leave it out."""
    cand = np.nonzero((local >= part.lo).all(axis=1) & (local <= part.hi).all(axis=1))[0]
    rows, depth, face = hits(part.index, local[cand])
    return cand[rows], depth, face


def _closed(tris: np.ndarray, size: float) -> bool:
    """Whether a triangle soup is a closed surface once its corners are welded, as topology
    welds them (`WELD_OF_PIECE_SIZE` times the piece size). glTF splits a vertex at a UV or
    normal seam, and the copies differ by float rounding."""
    if len(tris) == 0 or not size > 0.0:
        return False
    verts, faces = weld(tris, WELD_OF_PIECE_SIZE * size)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 2] != faces[:, 0])]
    return is_closed(verts[faces])


def _extent(points: np.ndarray) -> float:
    """The diagonal of the box round the points: the piece size."""
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _shell_candidates(body: Body, pos: np.ndarray, *, judged: bool = False,
                      together: np.ndarray | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """For each closed shell: (its index, the body vertices of other shells that may be inside it).

    The vertices are those in the shell's box. With `judged`, a vertex of the shell's own group or
    of an excused pair is left out before the inside test: those are never findings, and at the
    joints there are thousands of them.

    With `together` (V,), the vertices with the same id of 0 or more have moved as one rigid body
    since the rest pose, and an id under 0 moved alone. A vertex that has stayed in step with
    every vertex of a shell stands where it did, so it is left out for that shell: a pair that
    was not crossing at rest is not crossing now, and one that was is excused."""
    for k, shell in enumerate(body.shells):
        if not shell.closed:
            continue
        corner = pos[shell.verts]
        near = (pos >= corner.min(axis=0)).all(axis=1) & (pos <= corner.max(axis=0)).all(axis=1)
        if judged:
            near &= (body.vgroup != shell.gid) & ~body.excused[body.vgroup, shell.gid]
        if together is not None:
            ids = together[shell.verts]
            if ids[0] >= 0 and (ids == ids[0]).all():
                near &= together != ids[0]
        yield k, np.nonzero(near & (body.vshell >= 0) & (body.vshell != k))[0]


def _inside_shells(body: Body, pos: np.ndarray, indexes: _Indexes, *, judged: bool = False,
                   together: np.ndarray | None = None
                   ) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """For each closed shell: (its index, the body vertices of other shells inside it, depths).
    The candidates are `_shell_candidates`'."""
    for k, cand in _shell_candidates(body, pos, judged=judged, together=together):
        rows, depth, _ = hits(indexes(k), pos[cand])
        if len(rows):
            yield k, cand[rows], depth


def _is_printed(piece, part: str) -> bool:
    record = next((p for p in piece.parts if p.name == part), None)
    return record is not None and record.printed


def body_parts(ctx: SceneCtx, key: str) -> tuple[str, ...]:
    """The parts of the rig `key` that the body tests read: all of its parts but the printed ones
    (roles `eyes` and `decal`)."""
    pack, _, name = key.partition(":")
    piece = ctx.packs[pack].pieces[name]
    return tuple(p for p in ctx.rests[key].vertex_groups if not _is_printed(piece, p))


def _body(ctx: SceneCtx, clip: Clip, key: str, rest: RigRest) -> Body:
    parts = body_parts(ctx, key)
    groups = sorted({g for p in parts for g in rest.vertex_groups[p]})
    gid = {g: i for i, g in enumerate(groups)}
    offset, total = {}, 0
    for p in parts:
        offset[p] = total
        total += rest.shells[p].shape[1]
    vpart = np.concatenate([np.full(rest.shells[p].shape[1], i) for i, p in enumerate(parts)])
    vlocal = np.concatenate([np.arange(rest.shells[p].shape[1]) for p in parts])
    vgroup = np.concatenate([[gid[rest.vertex_groups[p][lab]] for lab in rest.labels[p]]
                             for p in parts]).astype(int)
    vshell = np.full(total, -1)
    found: list[Shell] = []
    size = _extent(np.concatenate([rest.shells[p][0] for p in rest.shells]).astype(float))
    for p in parts:
        tris = rest.faces[p].astype(int) + offset[p]
        local = rest.shells[p][0].astype(float)
        for label, rows in shells(rest.faces[p], rest.labels[p]):
            group = rest.vertex_groups[p][label]
            closed = _closed(local[rest.faces[p][rows]], size)
            used = np.unique(tris[rows])
            vshell[used] = len(found)
            found.append(Shell(p, group, gid[group], tris[rows], used, closed))
    body = Body(parts, groups, vpart, vgroup, vshell, vlocal, found,
                 np.zeros((len(groups), len(groups)), dtype=bool))
    body.excused = _joints(ctx, clip, body, rest)
    return body


def _joints(ctx: SceneCtx, clip: Clip, body: Body, rest: RigRest) -> np.ndarray:
    """The excused (vertex group, shell group) pairs: those that cross in the rest file, and each
    deforming bone with its nearest deforming ancestor."""
    excused = np.zeros((len(body.groups), len(body.groups)), dtype=bool)
    pos = np.concatenate([rest.shells[p][0] for p in body.parts]).astype(float)
    for k, vertices, _ in _inside_shells(body, pos, _Indexes(body, pos)):
        excused[body.vgroup[vertices], body.shells[k].gid] = True
    parent = {b.name: b.parent for b in rig_spec(ctx, clip.name).bones}
    known = set(body.groups)
    for bone in body.groups:
        up = parent.get(bone)
        while up is not None and up not in known:
            up = parent.get(up)
        if up is not None:
            i, j = body.groups.index(bone), body.groups.index(up)
            excused[i, j] = excused[j, i] = True
    return excused


# --- the placed and carried pieces -----------------------------------------------------------


@dataclass(eq=False)
class PiecePart:
    name: str
    tris: np.ndarray  # (N, 3, 3), piece frame
    verts: np.ndarray  # (U, 3) distinct corners, piece frame
    index: SurfaceIndex | None  # None when the part is not closed
    lo: np.ndarray
    hi: np.ndarray
    _room: list[np.ndarray] | None = field(default=None, repr=False)

    def room(self) -> list[np.ndarray]:
        """`room_ladder` of the part, in the piece frame; none for an open part."""
        if self._room is None:
            self._room = room_ladder(self.index) if self.index is not None else []
        return self._room


@dataclass(eq=False)
class Placed:
    id: str  # the placement key, or `prop:<prop id>` (as `Anchor.placed_by`)
    pack: str
    piece: str
    carried: bool
    parts: dict[str, PiecePart]
    frames: np.ndarray  # (S, 4, 4) world from piece
    shown: np.ndarray  # (S,) bool

    @property
    def label(self) -> str:
        return piece_key(self.pack, self.piece)

    def scale(self, s: int) -> float:
        return float(abs(np.linalg.det(self.frames[s, :3, :3])) ** (1.0 / 3.0))

    def world(self, pts: np.ndarray, s: int) -> np.ndarray:
        m = self.frames[s]
        return pts @ m[:3, :3].T + m[:3, 3]

    def local(self, pts: np.ndarray, s: int) -> np.ndarray:
        m = np.linalg.inv(self.frames[s])
        return pts @ m[:3, :3].T + m[:3, 3]

    def closed_parts(self) -> list[PiecePart]:
        """The closed parts, biggest box first (a plant has room in a big part)."""
        closed = [p for p in self.parts.values() if p.index is not None]
        return sorted(closed, key=lambda p: -float(np.prod(p.hi - p.lo)))


def _make_part(name: str, tris: np.ndarray, size: float) -> PiecePart:
    corners = tris.reshape(-1, 3)
    return PiecePart(name, tris, np.unique(corners, axis=0),
                 SurfaceIndex(tris) if _closed(tris, size) else None,
                 corners.min(axis=0), corners.max(axis=0))


def _inst_parts(piece, geom: PieceGeom) -> dict[str, PiecePart]:
    out = {}
    drawn = [np.asarray(x, dtype=float).reshape(-1, 3) for x in geom.parts.values() if len(x)]
    size = _extent(np.concatenate(drawn)) if drawn else 0.0  # the piece size, printed parts too
    for name, tris in geom.parts.items():
        if _is_printed(piece, name) or len(tris) == 0:
            continue
        tris = np.asarray(tris, dtype=float)
        out[name] = _kept(tris, lambda tris=tris, name=name: _make_part(name, tris, size))
    return out


def _shown(times: np.ndarray, windows) -> np.ndarray:
    """Per sample: True when the time lies in a window, by the scene's `due` rule."""
    mask = np.zeros(len(times), dtype=bool)
    mask[due(times, windows)] = True
    return mask


def _insts(ctx: SceneCtx, clip: Clip, samples: ClipSamples) -> tuple[list[Placed], list[str]]:
    """The clip's placed pieces, then its carried props, and a coverage line for each piece the
    scene holds no geometry for."""
    out, lines = [], []
    n = len(samples.times)
    rows = [(pid, p.pack, p.piece, False) for pid, p in clip.floor_gear.pieces.items()]
    rows += [(f"prop:{pid}", p.pack, p.piece, True) for pid, p in clip.props.items()]
    for iid, pack, name, carried in rows:
        geom = ctx.pieces.get(piece_key(pack, name))
        if geom is None:
            lines.append(Unmeasured(f"{NAME}: skipped (no geometry): {piece_key(pack, name)}"))
            continue
        piece = ctx.packs[pack].pieces[name]
        shown = (_shown(samples.times, clip.props[iid.removeprefix("prop:")].show_frames)
                 if carried else np.ones(n, dtype=bool))
        out.append(Placed(iid, pack, name, carried, _inst_parts(piece, geom),
                         piece_frames(clip, samples, iid, f"{clip.name} {iid}"), shown))
    return out, lines


# --- zones ------------------------------------------------------------------------------------


@dataclass(eq=False)
class Zone:
    """Where a pair may sink deeper than the skin: `budget` (a world depth) inside `radius` of
    `centre` at the samples `active` says.

    `target` is the id of the piece the zone is on. A hold zone for a contact group names its
    bone `groups` (tests A and B); a seat zone names the seat's surface `parts` (A and B); a
    prop zone names the carried `prop`'s id (test D)."""

    target: str
    budget: float
    radius: float
    centre: np.ndarray  # (S, 3)
    active: np.ndarray  # (S,) bool
    groups: frozenset[str] | None = None
    parts: frozenset[str] | None = None
    prop: str | None = None


def _zones(ctx: SceneCtx, clip: Clip, samples: ClipSamples, insts: list[Placed],
           base: float) -> list[Zone]:
    by_id = {i.id: i for i in insts}
    spec = rig_spec(ctx, clip.name)
    zones = []
    n = len(samples.times)
    for i, hold in enumerate(clip.holds):
        found = anchor(ctx.packs, hold.ref, clip=clip)
        target = by_id.get(found.placed_by)
        if found.kind != "hold" or target is None:
            continue
        if hold.who not in clip.props and hold.who not in spec.contact_groups:
            raise ManifestError(f"{clip.name} holds[{i}]: the rig has no contact group "
                                f"{hold.who!r}")
        radius = effector_radius(ctx, clip.name, hold.who)
        active = np.zeros(n, dtype=bool)
        active[due(samples.times, hold.when)] = True
        point = np.asarray(found.item.point, dtype=float)
        centre = target.frames[:, :3, :3] @ point + target.frames[:, :3, 3]
        budget = NO_BUDGET_LIMIT if found.item.grip else PRESS_OF_EFFECTOR * radius
        if hold.who in clip.props:
            zones.append(Zone(target.id, budget, (1 + SOLVE_ERR_OF_EFFECTOR) * radius, centre,
                               active, prop=f"prop:{hold.who}"))
            continue
        zones.append(Zone(target.id, budget, (1 + SOLVE_ERR_OF_EFFECTOR) * radius, centre,
                           active, groups=frozenset(spec.contact_groups[hold.who])))
    if clip.seat is not None:
        found = anchor(ctx.packs, clip.seat, clip=clip)
        if found.kind == "seat" and found.placed_by in by_id:
            active = np.zeros(n, dtype=bool)
            active[due(samples.times, seat_windows(clip))] = True
            zones.append(Zone(found.placed_by, SEAT_OF_EFFECTOR * base, NO_BUDGET_LIMIT,
                               np.zeros((n, 3)), active, parts=frozenset(found.item.surface)))
    return zones


def _budgets(zones: list[Zone], s: int, skin: float, pts: np.ndarray, *, target: str,
             groups: list[Collection[str]] | None = None, part: str | None = None,
             prop: str | None = None) -> np.ndarray:
    """The depth each world point may reach: the skin, or a zone's budget when a zone on `target`
    holds the point. For a body test (`groups`, `part`), a hold zone of a contact group applies
    to a point when one of its bone groups is among that point's `groups` (one collection per
    point: the vertex's own group in A, the groups at the corners of the nearest body face in B),
    and a seat zone when `part` is a seat surface part; for test D (`prop`), the prop's zones
    do."""
    allowed = np.full(len(pts), skin)
    for z in zones:
        if z.target != target or not z.active[s]:
            continue
        holds = np.ones(len(pts), dtype=bool)
        if prop is None:
            if z.prop is not None:
                continue
            if z.groups is not None:
                holds = np.array([not z.groups.isdisjoint(g) for g in groups], dtype=bool)
            if z.parts is not None and part not in z.parts:
                continue
        elif z.prop != prop:
            continue
        within = holds & (np.linalg.norm(pts - z.centre[s], axis=1) <= z.radius)
        allowed[within] = np.maximum(allowed[within], z.budget)
    return allowed


# --- one prepared clip --------------------------------------------------------------------------


@dataclass(eq=False)
class PreparedClip:
    ctx: SceneCtx
    clip: Clip
    samples: ClipSamples
    body: Body
    insts: list[Placed]
    zones: list[Zone]
    skin: float
    lines: list[str]  # coverage: what could not be measured, and what does not apply

    def tests(self) -> list[str]:
        """The tests that have something to measure in this clip."""
        have = [bool(self.insts), bool(self.insts),
                sum(s.closed for s in self.body.shells) >= 2,
                any(i.carried for i in self.insts) and any(not i.carried for i in self.insts)]
        return [t for t, ok in zip(TESTS, have, strict=True) if ok]


@dataclass
class Skipped:
    lines: list[str]


def prepare_clip(ctx: SceneCtx, clip: Clip) -> PreparedClip | Skipped:
    samples = ctx.samples.get(clip.name)
    if samples is None:
        return Skipped([Unmeasured(f"{NAME}: skipped (no samples): {clip.name}")])
    key = ctx.rig_of_clip.get(clip.name)
    if key is None:
        return Skipped([Unmeasured(f"{NAME}: skipped (no rig): {clip.name}")])
    rest = ctx.rests.get(key)
    if rest is None:
        return Skipped([Unmeasured(f"{NAME}: skipped (no body shells): {key}")])
    if not samples.shells:
        return Skipped([Unmeasured(f"{NAME}: skipped (no body shells): {clip.name}")])
    if set(samples.shells) != set(rest.vertex_groups):
        raise ManifestError(f"clip {clip.name!r}: its body parts {sorted(samples.shells)} differ "
                            f"from the rest file's {sorted(rest.vertex_groups)}")
    spec = rig_spec(ctx, clip.name)
    base = body_radius(spec, rest)
    if base is None:
        return Skipped([Unmeasured(f"{NAME}: skipped (no radius to measure by): {clip.name}")])
    pack, _, name = key.partition(":")
    piece = ctx.packs[pack].pieces[name]
    # per rest file: the body for each rig spec and piece record it was built from (the body
    # reads the spec's bones and the piece's part roles)
    kept = _kept(rest, dict)
    ident = (id(spec), id(piece))
    if ident not in kept or kept[ident][0] is not spec or kept[ident][1] is not piece:
        kept[ident] = (spec, piece, _body(ctx, clip, key, rest))
    body = kept[ident][2]
    insts, lines = _insts(ctx, clip, samples)
    lines += [Unmeasured(f"{NAME}: skipped (open shell, no inside): {s.name}")
              for s in body.shells if not s.closed]
    seen = set()
    for inst in insts:
        for part in inst.parts.values():
            if part.index is None and (inst.label, part.name) not in seen:
                seen.add((inst.label, part.name))
                lines.append(Unmeasured(
                    f"{NAME}: skipped (open part, no inside): {inst.label} {part.name}"))
    return PreparedClip(ctx, clip, samples, body, insts, _zones(ctx, clip, samples, insts, base),
                 SKIN_OF_EFFECTOR * base, lines)


# --- body in body (test C), for other checks ----------------------------------------------------


@dataclass(frozen=True)
class BodyContact:
    """One body shell through another at one sample: the vertex side `a` (`rig:<part>/<group>`)
    inside the shell `b`, `depth` deep, past the skin `budget`. Per sample, the deepest vertex of
    the pair. A pair of an excused joint, or inside the skin, is no contact."""

    sample: int
    a: str
    b: str
    depth: float
    budget: float = 0.0


def _first_inside(index: SurfaceIndex, pts: np.ndarray) -> int:
    """The row of the first point inside the closed index, taking points in growing batches."""
    start, size = 0, 16
    while start < len(pts):
        held = np.flatnonzero(index.inside(pts[start : start + size]))
        if len(held):
            return start + int(held[0])
        start, size = start + size, size * 4
    raise ValueError("no point of the group is inside")


def _contacts_at(c: PreparedClip, s: int, pos: np.ndarray, indexes: _Indexes,
                 together: np.ndarray | None = None) -> list[BodyContact]:
    """The contacts of one pose: per shell, the deepest vertex of each (part, group) that sinks
    into it past the skin. A group's deepest comes from `SurfaceIndex.deepest`, which leaves out a
    vertex that cannot beat the deepest found. The groups come in the order of their first vertex
    inside the shell, as they did when every vertex was tried."""
    body = c.body
    out: list[BodyContact] = []
    for k, cand in _shell_candidates(body, pos, judged=True, together=together):
        group = body.vpart[cand] * len(body.groups) + body.vgroup[cand]
        won = indexes(k).deepest(pos[cand], group, c.skin)
        first = [_first_inside(indexes(k), pos[cand[group == g]]) for g in won.group]
        starts = [int(cand[group == g][i]) for g, i in zip(won.group, first, strict=True)]
        out += [BodyContact(s, body.item(int(cand[won.row[i]])), body.shells[k].name,
                            float(won.depth[i]), c.skin) for i in np.argsort(starts)]
    return out


def _skip_reason(lines: list[str]) -> str:
    """The reason in a `skipped (...)` coverage line."""
    return next(line.split("skipped (", 1)[1].split("):", 1)[0] for line in lines
                if "skipped (" in line)


def body_contacts(ctx: SceneCtx, clip: Clip, samples: frozenset[int] = frozenset(),
                  together: np.ndarray | None = None) -> list[BodyContact]:
    """Test C alone, with its budget (`SKIN_OF_EFFECTOR` times the rig's radius) and its excused
    pairs: the body shells that sink into each other deeper than the skin, at every sample of the
    clip, or at just the sample indexes in `samples`. A clip the scene holds no data for raises
    `Unmeasurable`, naming what is missing.

    `together` (S, V) is a caller's knowledge of a posed body: at each sample, the vertices with
    the same id of 0 or more have moved as one rigid body since rest, and an id under 0 moved
    alone (`_inside_shells` says what that saves). Left out, every pair is tested."""
    c = prepare_clip(ctx, clip)
    if isinstance(c, Skipped):
        raise Unmeasurable(_skip_reason(c.lines))
    out: list[BodyContact] = []
    for s in range(len(c.samples.times)):
        if samples and s not in samples:
            continue
        pos = c.body.positions(c.samples, s)
        out += _contacts_at(c, s, pos, _Indexes(c.body, pos),
                            None if together is None else together[s])
    return out


def has_body_test(ctx: SceneCtx, clip: Clip) -> bool:
    """Whether the clip's body has two closed shells to sink into each other (test C applies)."""
    c = prepare_clip(ctx, clip)
    return not isinstance(c, Skipped) and "C" in c.tests()


def body_unreadable(ctx: SceneCtx, clip: Clip) -> str | None:
    """Why test C cannot read the clip (the scene holds no data for it), or None when it can."""
    c = prepare_clip(ctx, clip)
    return _skip_reason(c.lines) if isinstance(c, Skipped) else None


def body_contact_plants(ctx: SceneCtx, clip: Clip, s: int
                        ) -> Iterator[tuple[str, str, list[SceneCtx]]]:
    """Copies of the scene with one body vertex sunk into a shell it is not excused from, at
    sample `s`, one list per pair of a vertex group and a shell, in the order of `_candidates`:
    (the vertex side `rig:<part>/<group>`, the shell, the copies, shallowest first). Each sinks to
    a fraction of the room the shell has. Built from the geometry alone: the check's own test is
    never run on them, so a blind test cannot thin them out. A pair whose shell has no room has
    no copies. A clip the scene holds no data for gives no pair."""
    c = prepare_clip(ctx, clip)
    if isinstance(c, Skipped):
        return
    for item, copies in _candidates(c, "C", s):
        a, _, b = item.removeprefix(f"{clip.name} C ").partition(" x ")
        yield a, b, copies


# --- the measure ---------------------------------------------------------------------------------


class _Worst:
    """The deepest excess over its budget per item."""

    def __init__(self) -> None:
        self.by_item: dict[str, tuple[float, float, float, float]] = {}

    def add(self, item: str, depth: float, budget: float, frame: float) -> None:
        excess = depth - budget
        if excess > 0 and excess > self.by_item.get(item, (0.0,))[0]:
            self.by_item[item] = (excess, depth, budget, frame)

    def add_all(self, labels: list[str], depth: np.ndarray, budget: np.ndarray,
                frame: float) -> None:
        """Offending points with one item label each: the worst per label."""
        rows: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            rows.setdefault(label, []).append(i)
        for label, ix in rows.items():
            j = ix[int(np.argmax(depth[ix] - budget[ix]))]
            self.add(label, float(depth[j]), float(budget[j]), frame)

    def findings(self) -> list[Finding]:
        return [Finding(NAME, item, f"depth {d:.4f} over budget {b:.4f} at frame {t:g}")
                for item, (_, d, b, t) in self.by_item.items()]


def _measure(c: PreparedClip, sample_ids: Collection[int], tests: Collection[str]) -> _Worst:
    worst = _Worst()
    name = c.clip.name
    for s in sample_ids:
        t = float(c.samples.times[s])
        pos = c.body.positions(c.samples, s)
        indexes = _Indexes(c.body, pos)
        shown = [i for i in c.insts if i.shown[s]]
        if "C" in tests:
            _test_c(c, worst, name, s, t, pos, indexes)
        if "A" in tests:
            _test_a(c, worst, name, s, t, pos, shown)
        if "B" in tests:
            _test_b(c, worst, name, s, t, indexes, shown)
        if "D" in tests:
            _test_d(c, worst, name, s, t, shown)
    return worst


def _test_c(c, worst, name, s, t, pos, indexes) -> None:
    for contact in _contacts_at(c, s, pos, indexes):
        worst.add(f"{name} C {contact.a} x {contact.b}", contact.depth, contact.budget, t)


def _test_a(c, worst, name, s, t, pos, shown) -> None:
    body = c.body
    for inst in shown:
        local = inst.local(pos, s)
        scale = inst.scale(s)
        for part in inst.parts.values():
            if part.index is None:
                continue
            vertices, depth, _ = part_contacts(local, part)
            if not len(vertices):
                continue
            depth = depth * scale
            budget = np.empty(len(vertices))
            for g in np.unique(body.vgroup[vertices]):
                m = body.vgroup[vertices] == g
                budget[m] = _budgets(c.zones, s, c.skin, pos[vertices[m]], target=inst.id,
                                     groups=[(body.groups[g],)] * int(m.sum()), part=part.name)
            labels = [f"{name} A {body.item(v)} x {inst.label} {part.name}" for v in vertices]
            worst.add_all(labels, depth, budget, t)


def _test_b(c, worst, name, s, t, indexes, shown) -> None:
    body = c.body
    for inst in shown:
        for part in inst.parts.values():
            world = inst.world(part.verts, s)
            for k, shell in enumerate(body.shells):
                if not shell.closed:
                    continue
                corner = indexes.pos[shell.verts]
                cand = np.nonzero((world >= corner.min(axis=0)).all(axis=1)
                                  & (world <= corner.max(axis=0)).all(axis=1))[0]
                rows, depth, face = hits(indexes(k), world[cand])
                if not len(rows):
                    continue
                near = [frozenset(body.groups[g] for g in body.vgroup[shell.tris[f]])
                        for f in face]  # the groups at the corners of each nearest face
                budget = _budgets(c.zones, s, c.skin, world[cand[rows]], target=inst.id,
                                  groups=near, part=part.name)
                j = int(np.argmax(depth - budget))
                worst.add(f"{name} B {inst.label} {part.name} x {shell.name}",
                          float(depth[j]), float(budget[j]), t)


def _test_d(c, worst, name, s, t, shown) -> None:
    """A carried prop's vertices in a placed piece's parts, and the placed piece's in the prop's."""
    for prop in (i for i in shown if i.carried):
        for placed in (i for i in shown if not i.carried):
            for mover, holder in ((prop, placed), (placed, prop)):
                scale = holder.scale(s)
                for mpart in mover.parts.values():
                    world = mover.world(mpart.verts, s)
                    local = holder.local(world, s)
                    for hpart in holder.closed_parts():
                        cand = np.nonzero((local >= hpart.lo).all(axis=1)
                                          & (local <= hpart.hi).all(axis=1))[0]
                        rows, depth, _ = hits(hpart.index, local[cand])
                        if not len(rows):
                            continue
                        depth = depth * scale
                        budget = _budgets(c.zones, s, c.skin, world[cand[rows]],
                                          target=placed.id, prop=prop.id)
                        j = int(np.argmax(depth - budget))
                        worst.add(f"{name} D {mover.label} {mpart.name} x {holder.label} "
                                  f"{hpart.name}", float(depth[j]), float(budget[j]), t)


# --- plants -------------------------------------------------------------------------------------


def deep_point(index: SurfaceIndex, depth: float) -> np.ndarray | None:
    """A point `depth` inside a closed mesh, straight in from some face, or None when no face has
    that much room. Of the points inside, the one furthest from any surface."""
    tris = index.tris
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normal, axis=1)
    live = length > 0
    if not live.any():
        return None
    outward = 1.0 if signed_volume(tris) > 0 else -1.0
    pts = (tris[live].mean(axis=1)
           - outward * depth * normal[live] / length[live][:, None])
    rows = np.nonzero(index.inside(pts))[0]
    if not len(rows):
        return None
    return pts[rows[int(np.argmax(index.closest(pts[rows]).dist))]]


def with_body_vertex(ctx: SceneCtx, clip: str, part: str, local: int, s: int,
                      point: np.ndarray) -> SceneCtx:
    samples = ctx.samples[clip]
    moved = np.array(samples.shells[part])
    moved[s, local] = point
    return dataclasses.replace(ctx, samples={**ctx.samples, clip: dataclasses.replace(
        samples, shells={**samples.shells, part: moved})})


def _with_piece_vertex(ctx: SceneCtx, inst: Placed, part: PiecePart, vertex: np.ndarray,
                       point: np.ndarray) -> SceneCtx:
    """The scene with every corner of `part` at `vertex` moved to `point` (piece frame)."""
    tris = np.array(part.tris)
    tris[np.all(tris == vertex, axis=2)] = point
    geom = ctx.pieces[piece_key(inst.pack, inst.piece)]
    moved = PieceGeom(geom.name, {**geom.parts, part.name: tris})
    return dataclasses.replace(ctx, pieces={**ctx.pieces, piece_key(inst.pack, inst.piece): moved})


def _first_of_each_group(body: Body) -> list[int]:
    """One vertex per (part, vertex group), as a global vertex id."""
    seen: dict[tuple[int, int], int] = {}
    for v in range(len(body.vgroup)):
        if body.vshell[v] >= 0:
            seen.setdefault((int(body.vpart[v]), int(body.vgroup[v])), v)
    return list(seen.values())


# The depths a plant goes at, as fractions of the room the part or shell has (the radius of the
# largest ball inside it), shallowest first. The deepest uses all the room there is. The depths
# come from the shape alone: no tolerance, budget or zone decides them.
ROOM_FRACTIONS = (0.5, 0.75, 1.0)
# The probe depths for the room shrink by this ratio, each a point straight in from some face.
_PROBE_RATIO = 0.8
_MOST_PROBES = 80


def room_ladder(index: SurfaceIndex) -> list[np.ndarray]:
    """Points inside a closed mesh at `ROOM_FRACTIONS` of the room it has, shallowest first, or
    none when it has no inside. Each lies on the line from the nearest wall to the centre of the
    largest ball found inside the mesh, so a point at a fraction `f` of the radius is `f` of the
    radius from that wall, and no nearer another."""
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    depth = 0.5 * _extent(index.tris.reshape(-1, 3))
    for _ in range(_MOST_PROBES):
        if not depth > 0 or (best is not None and depth <= best[0]):
            break
        deep = deep_point(index, depth)
        if deep is not None:
            near = index.closest(deep[None])
            if best is None or float(near.dist[0]) > best[0]:
                best = (float(near.dist[0]), deep, near.point[0])
        depth *= _PROBE_RATIO
    if best is None:
        return []
    _, centre, wall = best
    return [wall + f * (centre - wall) for f in ROOM_FRACTIONS]


def _radius(points: list[np.ndarray]) -> float:
    """The room a `room_ladder` was made at (the radius of its ball); 0 for no points."""
    return 2.0 * float(np.linalg.norm(points[-1] - points[0])) if points else 0.0


def _roominess(tris: np.ndarray) -> float:
    """Volume over surface area of a closed triangle soup: a cheap measure of how thick a shape is
    (a ball of radius r gives r/3, a plate of thickness t about t/2). It only orders shells, so
    that the one `room_ladder` a unit needs is built for the thickest, not for all of them."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())
    return abs(signed_volume(tris)) / area if area > 0 else 0.0


@dataclass
class _Plan:
    """One item a test judges, and the faults that can be planted on it: `ladder()` gives the
    points inside the target (shallowest first; built on the first call, and only then), and
    `make` turns one into the scene with that fault.

    `zoned` says a hold or the seat of the clip is on the item's piece. That is a fact of the
    manifest, and it only puts the item later in the order: no size or point depends on it.
    `room` is how much room the target has, for the order alone."""

    item: str
    zoned: bool
    room: float
    ladder: Callable[[], list[np.ndarray]]
    make: Callable[[np.ndarray], SceneCtx]


def _plans(c: PreparedClip, test: str, s: int, want: str | None) -> Iterator[_Plan]:
    """Every item the test judges at sample `s` (just `want`, when given), each with the points
    inside its target that a fault can go at. Built from the geometry and the manifest alone."""
    ctx, name, body = c.ctx, c.clip.name, c.body
    pos = body.positions(c.samples, s)
    shown = [i for i in c.insts if i.shown[s]]
    movers = _first_of_each_group(body)
    shells = [k for k, sh in enumerate(body.shells) if sh.closed]
    laid: dict[int, list[np.ndarray]] = {}

    def zoned(target: str, part: str | None = None, group: str | None = None,
              prop: str | None = None) -> bool:
        """Whether a hold or the seat names this piece (this part, this vertex group, this carried
        prop) in the manifest. It never reads a zone's budget, radius or window."""
        return any(z.target == target and z.prop == prop
                   and (z.parts is None or part in z.parts)
                   and (z.groups is None or group is None or group in z.groups)
                   for z in c.zones)

    def shell_ladder(k: int) -> Callable[[], list[np.ndarray]]:
        def build() -> list[np.ndarray]:
            if k not in laid:
                laid[k] = room_ladder(SurfaceIndex(pos[body.shells[k].tris]))
            return laid[k]
        return build

    if test == "A":
        for inst in shown:
            for part in inst.closed_parts():
                for v in movers:
                    item = f"{name} A {body.item(v)} x {inst.label} {part.name}"
                    if want is not None and item != want:
                        continue
                    yield _Plan(item, zoned(inst.id, part.name, body.groups[body.vgroup[v]]),
                                _radius(part.room()), part.room, lambda p, v=v, inst=inst: (
                        with_body_vertex(ctx, name, body.parts[body.vpart[v]],
                                         int(body.vlocal[v]), s, inst.world(p, s))))
    elif test == "B":
        for k in shells:
            shell = body.shells[k]
            thick = _roominess(pos[shell.tris])
            for inst in shown:
                for part in inst.parts.values():
                    item = f"{name} B {inst.label} {part.name} x {shell.name}"
                    if want is not None and item != want:
                        continue
                    yield _Plan(item, zoned(inst.id, part.name), thick, shell_ladder(k), (
                        lambda p, inst=inst, part=part: _with_piece_vertex(
                            ctx, inst, part, part.verts[0], inst.local(p[None], s)[0])))
    elif test == "C":
        for k in shells:
            shell = body.shells[k]
            thick = _roominess(pos[shell.tris])
            for v in movers:
                g = body.vgroup[v]
                if body.vshell[v] == k or g == shell.gid or body.excused[g, shell.gid]:
                    continue
                item = f"{name} C {body.item(v)} x {shell.name}"
                if want is not None and item != want:
                    continue
                yield _Plan(item, False, thick, shell_ladder(k), lambda p, v=v: with_body_vertex(
                    ctx, name, body.parts[body.vpart[v]], int(body.vlocal[v]), s, p))
    else:
        for prop in (i for i in shown if i.carried):
            for placed in (i for i in shown if not i.carried):
                for hpart in placed.closed_parts():
                    for mpart in prop.parts.values():
                        item = (f"{name} D {prop.label} {mpart.name} x {placed.label} "
                                f"{hpart.name}")
                        if want is not None and item != want:
                            continue
                        yield _Plan(item, zoned(placed.id, prop=prop.id), _radius(hpart.room()),
                                    hpart.room, lambda p, prop=prop, mpart=mpart, placed=placed: (
                            _with_piece_vertex(ctx, prop, mpart, mpart.verts[0],
                                               prop.local(placed.world(p[None], s), s)[0])))


def _candidates(c: PreparedClip, test: str, s: int, want: str | None = None
                ) -> Iterator[tuple[str, list[SceneCtx]]]:
    """Faulted copies of the scene at sample `s`: (an item the test judges, the copies, shallowest
    fault first), best item first. With `want`, just that item.

    A fault sinks one vertex into a part or shell, to a fraction of the room that target has
    (`ROOM_FRACTIONS`). The order is: items with no hold or seat on their piece before those with
    one, and in each the target with the most room first, so a big thin part never stands in for a
    thicker one. An item with no room at all comes last, with no copies, and only when no item
    has room. The room ladders are built one item at a time, as the consumer asks."""
    plans = sorted(_plans(c, test, s, want), key=lambda p: (p.zoned, -p.room))
    given = False
    for plan in plans:
        points = plan.ladder()
        if points:
            given = True
            yield plan.item, [plan.make(p) for p in points]
    if plans and not given:
        yield plans[0].item, []


def _plant_samples(c: PreparedClip, test: str) -> list[int]:
    """The samples a plant may go at: the first, the middle and the last that the test reads."""
    n = len(c.samples.times)
    if test in ("A", "B"):
        ids = [s for s in range(n) if any(i.shown[s] for i in c.insts)]
    elif test == "D":
        ids = [s for s in range(n) if any(i.shown[s] and i.carried for i in c.insts)]
    else:
        ids = list(range(n))
    return first_middle_last(ids)


# What a test says when it judges no item at all in the clip.
_NO_ITEM = {
    "A": "no shown piece has a closed part to sink into",
    "B": "no closed body shell and shown piece part to sink into each other",
    "C": "no pair of closed body shells is judged",
    "D": "no placed piece has a closed part for a carried part to sink into",
}


def _unit(c: PreparedClip, test: str) -> PlantUnit:
    """The unit of one test in one clip, from the geometry and the manifest alone: the first item
    its candidates name, with that item's faults at the first, the middle and the last sample that
    the test reads. A test that judges no item says so. An item that is judged but has no room
    for a fault is a unit with no candidates and no reason, which the gate stops on."""
    item: str | None = None
    tries: list[tuple] = []
    for s in _plant_samples(c, test):
        first = next(_candidates(c, test, s, want=item), None)
        if first is None:
            continue
        named, copies = first
        item = item or named
        tries += [(test, moved, {s}) for moved in copies]
    if item is None:
        return PlantUnit(f"{c.clip.name} {test}", (), not_applicable=_NO_ITEM[test])
    return unit(item, tries)


# --- the check ----------------------------------------------------------------------------------

_NOT_APPLICABLE = {
    "A": "no placed or carried piece",
    "B": "no placed or carried piece",
    "C": "fewer than two closed body shells",
    "D": "needs a carried prop and a placed piece",
}


def _clips(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[Clip]:
    for clip in ctx.clips.values():
        if only is None or any(item.startswith(f"{clip.name} ") for item in only):
            yield clip


def _findings(ctx: SceneCtx, only: Collection[str] | None, samples: frozenset[int]
              ) -> list[Finding]:
    found: list[Finding] = []
    for clip in _clips(ctx, only):
        c = prepare_clip(ctx, clip)
        if isinstance(c, Skipped):
            continue
        tests = [t for t in c.tests() if only is None
                 or any(i.startswith(f"{clip.name} {t} ") for i in only)]
        n = len(c.samples.times)
        ids = [s for s in range(n) if not samples or s in samples]
        found += [f for f in _measure(c, ids, tests).findings()
                  if only is None or f.item in only]
    return found


class Overlap:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units: list[PlantUnit] = []
        for clip in ctx.clips.values():
            c = prepare_clip(ctx, clip)
            if isinstance(c, Skipped):
                continue
            units += [_unit(c, test) for test in c.tests()]
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        return _findings(ctx, only, samples)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        for clip in ctx.clips.values():
            c = prepare_clip(ctx, clip)
            if isinstance(c, Skipped):
                lines += c.lines
                continue
            lines += c.lines
            have = c.tests()
            lines += [f"{NAME}: not applicable ({_NOT_APPLICABLE[t]}): {clip.name} {t}"
                      for t in TESTS if t not in have]
        return list(dict.fromkeys(lines))
