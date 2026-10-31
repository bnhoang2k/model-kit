"""Check mirror-pair: a mirrored clip uses the side-swapped holds of its source.

For a clip with `mirror_of`, every hold and every target of the source clip must have its mirror
on the mirrored clip. The rig's mirror pairs give the swap (`solve.other_side`: `hand.R` to
`hand.L`; a name in no pair, such as a prop, is its own).

- **Holds.** Source hold `i` maps to a hold of the mirrored clip with the swapped `who`, the same
  `when`, and on the same placement a hold of the piece that is the source hold's mirror. For a
  hold of side `R` or `L` that is a hold of the other side whose point lies within the piece's
  contact length (`sight.contact_length`) of the source point reflected through the use
  station's centre plane. A piece with no `view` has no contact length, so the hold of a side is
  `Unmeasured` (a `both` hold needs none). A `both` hold maps to itself: the same hold, with the
  swapped
  `who`. Item `<mirrored> mirror-of <source> holds[<i>]`.
- **Targets.** Source target `j` maps to a mirrored target at the same key on the mirrored
  chain. Item `<mirrored> mirror-of <source> targets[<j>]`.

A scene with a rigged clip, rigs that all have a mirror map and no clip that mirrors another prints
one not-applicable line, so a body with a mirror map and no
mirrored clip is not a silent pass.

A rig with no `mirror` has no left and right to swap: the check prints
`not applicable (the rig has no mirror map)` for it, and measures no clip on it.

Only the source's holds and targets are mapped; a mirrored clip may hold more. The plants are one
unit per item, built from the clips alone (never from which hold or target the check matches):
each drops one swap on every hold (or every target) of a copy of the mirrored clip. For a hold:
the `who` is left unswapped, the hold is left unswapped, the `when` windows move past every
window of the two clips, and last the mirrored clip loses its holds. For a target: the chain is
left unswapped, and the keys move past every key of the two clips. The gate tries them in order
and stops the run when the check catches none.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Iterator

import numpy as np

from model_kit.core.checks.entries import Entry, unit
from model_kit.core.checks.sides import OTHER, right_of, use_of
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Clip, ClipHold, MirrorSpec, anchor
from model_kit.core.metagate import Finding, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx, piece_key, rig_spec
from model_kit.core.sight import contact_length
from model_kit.core.solve import other_side

NAME = "mirror-pair"


def _with_clip(ctx: SceneCtx, clip: Clip) -> SceneCtx:
    return dataclasses.replace(ctx, clips={**ctx.clips, clip.name: clip})


def _hold_of(ctx: SceneCtx, source: Clip, i: int, item: str):
    """The source hold `i`'s anchor, which must name a hold."""
    found = anchor(ctx.packs, source.holds[i].ref, clip=source)
    if found.kind != "hold":
        raise ManifestError(f"{item}: {source.holds[i].ref} names a {found.kind}, not a hold")
    return found


def _reach_of(ctx: SceneCtx, source: Clip, i: int) -> float | None:
    """How far a mirrored hold's point may sit from the reflected source point: the contact length
    of the source hold's piece, None when the piece declares no view."""
    found = anchor(ctx.packs, source.holds[i].ref, clip=source)
    return contact_length(found.piece, ctx.packs[source.holds[i].ref.pack].units_per_metre)


def _unread(ctx: SceneCtx, source: Clip, i: int, item: str) -> str | None:
    """Why source hold `i` cannot be judged, or None when it can. A hold of a side needs the
    piece's contact length; a `both` hold needs none."""
    if _hold_of(ctx, source, i, item).item.side != "both" and _reach_of(ctx, source, i) is None:
        return "the piece declares no view, so it has no contact length"
    return None


def _hold_entry(ctx: SceneCtx, source: Clip, mirrored: Clip, i: int, mirror, item: str
                ) -> Entry | str:
    """The hold's entry, or why it cannot be read (`_unread`)."""
    if (why := _unread(ctx, source, i, item)) is not None:
        return why
    src = source.holds[i]
    found = _hold_of(ctx, source, i, item)
    spot = found.item
    use = use_of(found.piece, spot.use, item)
    if spot.side == "both":
        partners = [spot.id]
    else:
        right = right_of(use, item)
        right = right / np.linalg.norm(right)
        point = np.asarray(spot.point, dtype=float)
        station = np.asarray(use.point, dtype=float)
        reflected = point - 2.0 * float((point - station) @ right) * right
        tol = _reach_of(ctx, source, i)
        partners = [h.id for h in found.piece.holds if h.side == OTHER[spot.side]
                    and float(np.linalg.norm(np.asarray(h.point, dtype=float) - reflected)) <= tol]
    who = other_side(src.who, mirror)

    def matches(h: ClipHold) -> bool:
        return (h.who == who and h.when == src.when and h.ref.pack == src.ref.pack
                and h.ref.piece == src.ref.piece and h.ref.id in partners)

    hits = [k for k, h in enumerate(mirrored.holds) if matches(h)]
    if partners:
        detail = (f"no hold of {mirrored.name} has who {who!r}, when {[list(w) for w in src.when]}"
                  f" and a ref on {src.ref.pack}:{src.ref.piece} naming {sorted(partners)}")
    else:
        detail = (f"{src.ref.pack}:{found.piece.name} has no hold of side {OTHER[spot.side]} "
                  f"within reach of {src.ref}'s point reflected through station {use.id!r}")

    return Entry(item, bool(hits), detail)


def _target_entry(ctx: SceneCtx, source: Clip, mirrored: Clip, j: int, mirror, item: str
                  ) -> Entry:
    src = source.targets[j]
    chain = other_side(src.chain, mirror)
    hits = [k for k, t in enumerate(mirrored.targets) if t.key == src.key and t.chain == chain]
    detail = (f"{mirrored.name} has no target at key {src.key} on chain {chain!r} "
              f"(the mirror of {src.chain!r})")

    return Entry(item, bool(hits), detail)


def _mirrored(ctx: SceneCtx) -> Iterator[tuple[Clip, Clip, MirrorSpec, str]]:
    """Each clip with `mirror_of` whose rig has a mirror map: it, its source, the map and the
    head of its items."""
    for mirrored in ctx.clips.values():
        if mirrored.mirror_of is None:
            continue
        source = ctx.clips.get(mirrored.mirror_of)
        if source is None:
            raise ManifestError(f"clip {mirrored.name!r}: mirror_of names no clip "
                                f"{mirrored.mirror_of!r}")
        mirror = rig_spec(ctx, mirrored.name).mirror
        if mirror is None:
            continue
        yield mirrored, source, mirror, f"{mirrored.name} mirror-of {source.name}"


def _entries(ctx: SceneCtx, only: Collection[str] | None) -> Iterator[Entry]:
    for mirrored, source, mirror, head in _mirrored(ctx):
        for i in range(len(source.holds)):
            if only is None or f"{head} holds[{i}]" in only:
                entry = _hold_entry(ctx, source, mirrored, i, mirror, f"{head} holds[{i}]")
                if isinstance(entry, Entry):
                    yield entry
        for j in range(len(source.targets)):
            if only is None or f"{head} targets[{j}]" in only:
                yield _target_entry(ctx, source, mirrored, j, mirror, f"{head} targets[{j}]")


# --- plants -----------------------------------------------------------------------------------


def _past(*spans: tuple[float, float]) -> float:
    """A shift that takes every span clear of where the spans were: the length they cover, and
    one more."""
    return max(b for _, b in spans) - min(a for a, _ in spans) + 1.0


def _hold_units(ctx: SceneCtx, source: Clip, mirrored: Clip, head: str) -> Iterator[PlantUnit]:
    """Per source hold, copies of the mirrored clip with a swap dropped on every one of its holds,
    so whichever hold the check matches is broken."""
    def edit(change) -> SceneCtx:
        holds = tuple(dataclasses.replace(h, **change(h)) for h in mirrored.holds)
        return _with_clip(ctx, dataclasses.replace(mirrored, holds=holds))

    windows = [w for h in (*source.holds, *mirrored.holds) for w in h.when]
    by = _past(*windows) if windows else 0.0
    for i, src in enumerate(source.holds):
        if _unread(ctx, source, i, f"{head} holds[{i}]") is not None:
            continue  # no contact length: the coverage line says so
        yield unit(f"{head} holds[{i}]", [
            ("unswapped who", edit(lambda h, src=src: {"who": src.who})),
            ("unswapped hold", edit(lambda h, src=src: {"ref": src.ref})),
            ("shifted when", edit(lambda h: {"when": tuple((a + by, b + by) for a, b in h.when)})),
            ("no holds", _with_clip(ctx, dataclasses.replace(mirrored, holds=())))])


def _target_units(ctx: SceneCtx, source: Clip, mirrored: Clip, head: str
                  ) -> Iterator[PlantUnit]:
    def edit(change) -> SceneCtx:
        targets = tuple(dataclasses.replace(t, **change(t)) for t in mirrored.targets)
        return _with_clip(ctx, dataclasses.replace(mirrored, targets=targets))

    keys = [t.key for t in (*source.targets, *mirrored.targets)]
    by = int(_past(*[(k, k) for k in keys])) if keys else 0
    for j, src in enumerate(source.targets):
        yield unit(f"{head} targets[{j}]", [
            ("unswapped chain", edit(lambda t, src=src: {"chain": src.chain})),
            ("moved key", edit(lambda t: {"key": t.key + by}))])


def _units(ctx: SceneCtx) -> Iterator[PlantUnit]:
    for mirrored, source, _, head in _mirrored(ctx):
        yield from _hold_units(ctx, source, mirrored, head)
        yield from _target_units(ctx, source, mirrored, head)


def _findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    return [Finding(NAME, e.item, e.detail) for e in _entries(ctx, only) if not e.ok]


def _not_applicable(ctx: SceneCtx) -> Iterator[str]:
    """One line for each rigged piece in the scene that has no mirror map. A body with no sides
    is fine by design."""
    for pack_name, pack in sorted(ctx.packs.items()):
        for piece in pack.pieces.values():
            if piece.rig is not None and piece.rig.mirror is None:
                yield (f"{NAME}: not applicable (the rig has no mirror map): "
                       f"{piece_key(pack_name, piece.name)}")


class MirrorPair:
    name = NAME
    kinds: frozenset[str] = frozenset()  # a clip check: it measures no piece's recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return list(_units(ctx))

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _findings(ctx, only)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        no_map = list(_not_applicable(ctx))
        rigged = any(name in ctx.rig_of_clip for name in ctx.clips)
        # Say it once: a rig with no mirror map already explains why nothing is mirrored.
        none = [f"{NAME}: not applicable (no clip has mirror_of): {ctx.folder or 'the scene'}"
                ] if rigged and not no_map and not any(
                    c.mirror_of for c in ctx.clips.values()) else []
        unread = [Unmeasured(f"{NAME}: skipped ({why}): {head} holds[{i}]")
                  for mirrored, source, mirror, head in _mirrored(ctx)
                  for i in range(len(source.holds))
                  if isinstance(why := _hold_entry(ctx, source, mirrored, i, mirror,
                                                   f"{head} holds[{i}]"), str)]
        return [*no_map, *none, *unread]
