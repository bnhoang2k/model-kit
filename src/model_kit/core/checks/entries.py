"""What the checks that measure items one by one (frames, sides, mirror-pair, receiver,
connectors) share.

An `Entry` is one measured item: whether it passes now and why not. It holds no plant. Each of
those checks builds its plant units in `plants`, from the manifest and the geometry alone, and
never from an `Entry`: an entry carries the verdict, and a plant built from it would be blind
only as far as the check is. `unit` turns an item and its candidate copies of the scene, in the
order to try, into a `PlantUnit`. The clip and rig checks (clip hygiene, seat fit, solver error,
overlap, stress poses, bend quality, coverage's no-hold rule) build theirs the same way, adding the
sample indexes each fault touches. No candidate is a unit that the gate cannot satisfy, so it
stops the run unless the real check already fails the item.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from model_kit.core.metagate import Plant, PlantUnit


@dataclass(frozen=True)
class Entry:
    item: str
    ok: bool
    detail: str = ""


# One candidate: (label, the planted scene), and for a clip check the sample indexes the fault
# touches (`Plant.samples`) as a third member.
Candidate = tuple[str, Any] | tuple[str, Any, Collection[int]]


def unit(item: str, candidates: Iterable[Candidate]) -> PlantUnit:
    """The plant unit of `item`: each candidate is one plant, in the order to try."""
    return PlantUnit(item, tuple(
        Plant(f"{label}: {item}", item, planted, samples=frozenset(rest[0]) if rest else frozenset())
        for label, planted, *rest in candidates))


def first_middle_last(ids: Sequence[int]) -> list[int]:
    """The first, the middle and the last of `ids`, without repeats, in that order: the samples a
    clip check's plant goes at. Empty for no ids."""
    return list(dict.fromkeys([ids[0], ids[len(ids) // 2], ids[-1]])) if len(ids) else []
