"""Branching check: crossing-organic. Two limbs do not grow through each other.

Every limb pair, a parent and its child included, and each limb against itself, is read at its
segment pair with the smallest gap share, leaving out the zones `core/grow/segments` defines
(each limb's seat part, the shared zone of siblings from one place, a limb's own neighbouring
segments). It fails when the limbs overlap by more than
`CROSS_OF_RADIUS` x the thinner radius there (item `<pack>:<piece> <a> x <b>`; the line between
this and a near miss is `_pairs.classify_gap`). Leaf cards have no inside, so they are left out,
listed in coverage once per piece.

Plant, one unit per grown piece (`_pairs.pair_units`): one twig moved rigidly so it overlaps an
unrelated limb by `CALIBRATION_FAIL` x the tolerance; a tree with no twig on an unrelated limb
moves another limb against another. Built from the limbs' geometry alone, never from the pair
search above. A piece whose limbs take no placement stops the gate (`BlindCheck`).
"""

from __future__ import annotations

from collections.abc import Collection

from model_kit.core.checks.branching._pairs import CROSSING, classify_gap, limb_pairs, pair_units
from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown, pairs_named
from model_kit.core.metagate import Finding, PlantUnit
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CROSS_OF_RADIUS,
    SEAT_ZONE_OPENS_OF_RADIUS,
)

NAME = "crossing-organic"


def _findings(g: Grown, only) -> list[Finding]:
    found = []
    for i, j, gap, r in limb_pairs(g.skeleton, pairs_named(g, only), reach=0.0,
                                   opens=SEAT_ZONE_OPENS_OF_RADIUS):
        if classify_gap(gap, r) == CROSSING:
            found.append(Finding(NAME, g.pair(i, j), f"overlap {-gap:.4f}, over "
                                                     f"{CROSS_OF_RADIUS * r:.4f} (the thinner "
                                                     f"radius there {r:.4f})"))
    return found


class CrossingOrganic:
    name = NAME
    kinds: frozenset[str] = KINDS

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        share = -CALIBRATION_FAIL * CROSS_OF_RADIUS
        return [u for g in grown(ctx)
                for u in pair_units(ctx, g, share, SEAT_ZONE_OPENS_OF_RADIUS, "through")]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The tube parts of each grown piece. Leaf cards have no inside and are left out."""
        return {part for g in grown(ctx) for part in g.limb_parts()}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return [f for g in grown(ctx, only) for f in _findings(g, only)]

    def coverage(self, ctx: SceneCtx) -> list[str]:
        return coverage(ctx, NAME) + [f"{NAME}: not applicable (leaf cards have no inside): {g.key}"
                                      for g in grown(ctx) if len(g.skeleton.host)]
