"""Check coverage: nothing exported goes unmeasured.

Two rules.

- **Exported parts.** Every part with geometry in the scene (an exported mesh) must be measured
  by at least one of the other checks named in this run. A check says which parts it measured
  through its `measured(ctx)` hook (`metagate.Measures`); a check with no hook, such as a clip
  check, measures none. The item is `<pack>:<piece> <part>`, the item of floaters and topology.
  A part no check reads is the hole the other checks cannot see: a branching piece with no
  skeleton, a printed part in a run with neither floaters nor topology (both measure printed
  parts), a mesh no manifest lists.
- **Contacts with no hold.** Overlap's test A finds a body vertex inside a part of a placed or
  carried piece. Where overlap judges how deep, coverage asks why: at a sample where the body is
  inside a piece, a hold of the clip (in the window of its `when`) or the clip's seat (in its
  `seat_windows`) must be on that piece. A contact at any depth counts. The item is
  `<clip> <pack>:<piece> no-hold`, one per clip and piece, naming the first frame.

`Coverage(others)` takes the other checks of the run, in the order they ran; `gate.build_checks`
builds it last. The scene's plants, as units built from the scene alone (never from the scan or
the part check that judges them): a copy with one more exported mesh in a piece no manifest
lists (no check can reach it), and, per clip and piece with a closed part, a body vertex sunk into
that piece with its holds and seat taken away (`sunk_pieces`; a piece never shown in
the clip is not applicable). A clip the scan has no data for (no samples, no body shells, a part
with no inside) is `Unmeasured`; a clip that places no piece has no contact to find, and says so in
a plain line.

A rigged piece's body is `rests`, which overlap measures, but its piece geometry is exported like
any other: its `pieces/<name>/<name>.npz` is measured, part by part, by crossings, take-apart,
floaters and topology.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import numpy as np

from model_kit.core.checks import overlap
from model_kit.core.checks.overlap import (
    Placed,
    PreparedClip,
    Skipped,
    part_contacts,
    prepare_clip,
    with_body_vertex,
)
from model_kit.core.checks.pieces import free_name
from model_kit.core.manifest import Clip, anchor
from model_kit.core.metagate import Check, Finding, Measures, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import PieceGeom, SceneCtx, part_key, piece_key

NAME = "coverage"
NO_HOLD = "no-hold"
_TRIANGLE = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


def _is_contact(item: str) -> bool:
    return item.endswith(f" {NO_HOLD}")


# --- contacts with no hold: the scan, and the plants for it -----------------------------------------
# Overlap's test A and this rule ask the same question of a body vertex and a piece part, and both
# call `overlap.part_contacts` for it. The prepared clip is overlap's (`prepare_clip`).


@dataclass
class Contacts:
    """The body's contacts with the pieces of one clip that no hold or seat explains: the frame
    of the first contact per piece (`<pack>:<piece>`), and the `Unmeasured` lines for what the
    scan could not read (a clip or part it had no data for)."""

    first: dict[str, float]
    lines: list[str]


def _as_coverage(lines: Collection[str]) -> list[str]:
    """Overlap's skip lines renamed as coverage's own."""
    return [Unmeasured(f"{NAME}{line[len(overlap.NAME):]}") for line in lines]


def _held(c: PreparedClip, inst: Placed, s: int) -> bool:
    """Whether a hold or the seat of the clip is on this piece at sample `s`."""
    return any(z.target == inst.id and z.active[s] for z in c.zones)


def unheld_contacts(ctx: SceneCtx, clip: Clip, samples: frozenset[int] = frozenset()) -> Contacts:
    """Test A without a budget: every body vertex inside a part of a shown placed or carried
    piece, at any depth, at a sample where no hold or seat of the clip is on that piece. With
    `samples`, just those sample indexes. A clip the scene holds no data for, or an open part
    (which has no inside), is a coverage line, not a contact."""
    c = prepare_clip(ctx, clip)
    if isinstance(c, Skipped):
        return Contacts({}, _as_coverage(c.lines))
    lines = _as_coverage([line for line in c.lines
                          if "(open part" in line or "(no geometry" in line])
    first: dict[str, float] = {}
    for s in range(len(c.samples.times)):
        if samples and s not in samples:
            continue
        todo = [i for i in c.insts if i.shown[s] and i.label not in first and not _held(c, i, s)]
        if not todo:
            continue
        pos = c.body.positions(c.samples, s)
        for inst in todo:
            local = inst.local(pos, s)
            for part in inst.parts.values():
                if part.index is not None and len(part_contacts(local, part)[0]):
                    first[inst.label] = float(c.samples.times[s])
                    break
    return Contacts(first, lines)


def has_contact_test(ctx: SceneCtx, clip: Clip) -> bool:
    """Whether the clip has a placed or carried piece for the body to touch (test A applies)."""
    c = prepare_clip(ctx, clip)
    return not isinstance(c, Skipped) and "A" in c.tests()


def _without_holds(ctx: SceneCtx, clip: Clip, inst: Placed) -> SceneCtx:
    """The scene with the clip's holds and seat on `inst` taken away."""
    holds = tuple(h for h in clip.holds
                  if anchor(ctx.packs, h.ref, clip=clip).placed_by != inst.id)
    changes: dict = {"holds": holds}
    if clip.seat is not None and anchor(ctx.packs, clip.seat, clip=clip).placed_by == inst.id:
        # a clip with no seat cannot be a `sit` clip with a seat window: the plant's clip says
        # nothing about posture, as a clip written before posture does
        changes |= {"seat": None, "seat_when": None, "posture": None}
    moved = dataclasses.replace(clip, **changes)
    return dataclasses.replace(ctx, clips={**ctx.clips, clip.name: moved})


@dataclass
class SunkPiece:
    """One placed or carried piece (by label) with a closed part, and the plants for it.

    `candidates` are (sample, the scene with a body vertex sunk into the piece and its holds
    taken away), in the order to try. `why_none` says why there are none when that is by design:
    no copy of the piece is ever shown, or the body has no vertex. A piece that is shown and has
    no candidates has no reason, and the gate stops on it."""

    label: str
    candidates: list[tuple[int, SceneCtx]] = field(default_factory=list)
    why_none: str = ""
    shown: bool = False


def _samples_shown(shown: np.ndarray) -> list[int]:
    """The first, middle and last sample at which the piece is shown."""
    ids = np.flatnonzero(shown)
    if not len(ids):
        return []
    return [int(i) for i in dict.fromkeys((ids[0], ids[len(ids) // 2], ids[-1]))]


def sunk_pieces(ctx: SceneCtx, clip: Clip) -> list[SunkPiece]:
    """The pieces of the clip with a closed part, one entry per `<pack>:<piece>` label, each with
    its plants. Built from the geometry alone, never from `unheld_contacts`, which judges them:
    a plant must not ask the scan which piece it can fail, or a scan that never finds a contact
    would leave no plant and read as a pass.

    Per piece, a plant sinks the body's first vertex into a closed part, at the first, middle and
    last sample where the piece is shown, and takes away the clip's holds and seat on that piece.
    The scan judges every body vertex, so any vertex will do. The vertex goes to the fractions
    of the room the part has (`overlap.ROOM_FRACTIONS`), shallowest first; the first part, biggest
    box first, with an inside takes it. A clip that places one piece twice has one entry: its
    first copy that can be planted. A piece with no geometry or only open parts has no inside, and
    overlap's coverage lines already say so. Empty when the scene holds no data for the clip."""
    c = prepare_clip(ctx, clip)
    if isinstance(c, Skipped):
        return []
    body = c.body
    by_label: dict[str, SunkPiece] = {}
    for inst in c.insts:
        parts = inst.closed_parts()
        if not parts:
            continue
        sunk = by_label.setdefault(inst.label, SunkPiece(inst.label))
        if sunk.candidates:
            continue  # an earlier copy of the piece has the plants
        if not len(body.vgroup):
            sunk.why_none = "the body has no vertex"
            continue
        samples = _samples_shown(inst.shown)
        if not samples:
            if not sunk.shown:
                sunk.why_none = "the piece is never shown in the clip"
            continue
        sunk.shown, sunk.why_none = True, ""
        for s in samples:
            for part in parts:
                if not part.room():
                    continue
                for point in part.room():
                    moved = with_body_vertex(ctx, clip.name, body.parts[body.vpart[0]],
                                             int(body.vlocal[0]), s, inst.world(point, s))
                    sunk.candidates.append((s, _without_holds(moved, clip, inst)))
                break
    return list(by_label.values())


class Coverage:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a scene check: it measures no piece's recipe

    def __init__(self, others: Iterable[Check] = ()) -> None:
        self.others = tuple(others)
        self._scanned: tuple[SceneCtx, dict[str, Contacts]] | None = None

    # --- the exported parts ---------------------------------------------------------------

    def _measured(self, ctx: SceneCtx) -> set[str]:
        out: set[str] = set()
        for other in self.others:
            if isinstance(other, Measures):
                out |= other.measured(ctx)
        return out

    def _unmeasured_parts(self, ctx: SceneCtx) -> list[str]:
        exported = {part_key(*key.partition(":")[::2], part)
                    for key, geom in ctx.pieces.items() for part in geom.parts}
        return sorted(exported - self._measured(ctx))

    # --- the contacts ---------------------------------------------------------------------

    def _scan(self, ctx: SceneCtx) -> dict[str, Contacts]:
        if self._scanned is None or self._scanned[0] is not ctx:
            self._scanned = (ctx, {c.name: unheld_contacts(ctx, c) for c in ctx.clips.values()})
        return self._scanned[1]

    def _contacts(self, ctx: SceneCtx, clip: Clip, whole: bool, samples: frozenset[int]
                  ) -> Contacts:
        return self._scan(ctx)[clip.name] if whole else unheld_contacts(ctx, clip, samples)

    # --- the check ------------------------------------------------------------------------

    def _ghost(self, ctx: SceneCtx) -> PlantUnit | None:
        """A copy with one more exported mesh, in a piece that no pack lists: no check reaches
        it, so only coverage can see that nothing measured it."""
        if not ctx.packs:
            return None
        pack = min(ctx.packs)
        taken = {k.partition(":")[2] for k in ctx.pieces} | set(ctx.packs[pack].pieces)
        name = free_name(taken, "Ghost")
        part = "mesh"
        geometry = {**ctx.pieces, piece_key(pack, name): PieceGeom(name, {part: _TRIANGLE.copy()})}
        item = part_key(pack, name, part)
        planted = dataclasses.replace(ctx, pieces=geometry)
        return PlantUnit(item, (Plant(f"an unlisted mesh: {item}", item, planted),))

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        units: list[PlantUnit] = []
        ghost = self._ghost(ctx)
        if ghost is not None:
            units.append(ghost)
        for clip in ctx.clips.values():
            if not has_contact_test(ctx, clip):
                continue
            for sunk in sunk_pieces(ctx, clip):
                item = f"{clip.name} {sunk.label} {NO_HOLD}"
                units.append(PlantUnit(
                    item,
                    tuple(Plant(f"sunk, no hold: {item}", item, planted, samples=frozenset({s}))
                          for s, planted in sunk.candidates),
                    not_applicable=sunk.why_none))
        return units

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        found: list[Finding] = []
        if only is None or any(not _is_contact(i) for i in only):
            names = ", ".join(o.name for o in self.others) or "none"
            found += [Finding(NAME, item, f"exported, and no check measured it (checks run: "
                                          f"{names})")
                      for item in self._unmeasured_parts(ctx) if only is None or item in only]
        whole = only is None and not samples
        for clip in ctx.clips.values():
            if only is not None and not any(i.startswith(f"{clip.name} ") for i in only):
                continue
            for label, frame in self._contacts(ctx, clip, whole, samples).first.items():
                item = f"{clip.name} {label} {NO_HOLD}"
                if only is None or item in only:
                    found.append(Finding(NAME, item, f"the body is inside {label} from frame "
                                                     f"{frame:g}, and no hold or seat of the "
                                                     "clip is on it then"))
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines: list[str] = []
        for clip in ctx.clips.values():
            lines += self._scan(ctx)[clip.name].lines
            if not has_contact_test(ctx, clip) and not self._scan(ctx)[clip.name].lines:
                lines.append(f"{NAME}: not applicable (no placed or carried piece): {clip.name}")
        return list(dict.fromkeys(lines))
