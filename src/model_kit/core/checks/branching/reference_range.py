"""Branching check: reference-range. A grown tree's size and form lie inside its species' cited
ranges.

A result check, and the fit target: it is the one plant check that may be fixed by refitting a
fitted input of the species file. It judges two quantities, each as a share of the tree's height
(the highest limb point, as the grower scales the tree to `height_m`), against the range of the
same key in the species' rules file (`GrowthRules.ranges`, copied into the scene's `rules/`
folder):
- `spread_of_height`: the spread is the forest-inventory crown diameter, the mean of the widest
  width of the crown in plan view (glTF axes: x-z) and its width at right angles to it
  (Schomaker et al. 2007, USDA FS GTR SRS-102, sec 2.3.8.1). The crown is every limb point that
  is not buried (roots), at or above the crown base, and every vertex of every leaf. The
  inventory's rule that leaves out abnormally long one-sided branches is not applied.
- `crown_base_of_height`: the crown base is the seat height of the lowest leaf-bearing limb (this
  kit's definition; FIA sec 2.3.2.2: "the lowest live foliage of the obvious live crown"). A
  limb is a continuation chain, so the trunk chain bears leaves too and its seat is the soil.
  A limb with no parent has no seat on another limb, so it counts at the height of the lowest
  leaf it bears. A tree with no leaves has no crown base: that range is not applicable, and the
  spread is then measured on its wood.
- `fork_of_height` is taste-only: no cited numeric definition of a codominant fork exists (Gilman
  2003 gives none). A species file with that range gets a coverage line, no finding.

The rules file is found by the name in the skeleton's `info["rules"]`. The skeleton records the
hashes of the species file and of the kit file it was grown from (`rules_sha256`, `kit_sha256`);
a copy in the scene that differs fails, naming both hashes (items `<pack>:<piece> rules` and
`<pack>:<piece> kit`), and the ranges are then not judged, because they belong to another file.
A skeleton with no hashes (a tree from the first grower or a measured tree) is not applicable; a rules file missing
from the scene is `Unmeasured`. Items are `<pack>:<piece> <range key>`.

Plants, one unit for each range key a judged piece has (never `_share_findings`, `spread`,
`crown_base` or `height`): the crown scaled across x and z about the trunk by a factor that puts its
spread over `RANGE_PLANT_OVER` x the range's upper end of the height, read from the crown wood's
widest width alone (a lower bound on the spread; height unchanged); and every leaf whose node sits
below `RANGE_PLANT_OVER` x the crown-base range's upper end of the height removed, or, when no leaf
sits above that line, every leaf lifted until the lowest sits on it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np

from model_kit.core.checks.branching._pieces import KINDS, Grown, coverage, grown
from model_kit.core.grow.mesh import all_leaf_tris
from model_kit.core.grow.rules import GrowthRules, Range
from model_kit.core.grow.skeleton import Skeleton
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import RANGE_PLANT_OVER

NAME = "reference-range"
SPREAD, CROWN_BASE, FORK = "spread_of_height", "crown_base_of_height", "fork_of_height"
# The plant's spread solve (blades keep their size, so one factor does not reach the target): up
# to SCALE_STEPS fixed-point steps, stopped once the spread is within SCALE_MATCH of the target.
# Numerics of the solve, not a tolerance on a tree.
SCALE_STEPS = 24
SCALE_MATCH = 1e-9

# What a piece's rules and hashes say about whether it can be judged.
APPLICABLE, NOT_APPLICABLE, MISSING, STALE = "applicable", "not applicable", "missing", "stale"


# --- the measures ------------------------------------------------------------------------------


def height(sk: Skeleton) -> float:
    """The tree's height: the highest limb point, as the grower scales the tree to `height_m`."""
    return float(sk.pts[:, 1].max())


def crown_base(sk: Skeleton) -> float | None:
    """The height of the lowest live leaf (the inventory's "lowest live foliage of the obvious live
    crown"). None when the tree has no leaves."""
    if not len(sk.host):
        return None
    return float(sk.node[:, 1].min())


def _leaf_vertices(sk: Skeleton) -> np.ndarray:
    if not len(sk.host):
        return np.zeros((0, 3))
    return all_leaf_tris(sk).reshape(-1, 3)


def _hull(xy: np.ndarray) -> np.ndarray:
    """The convex hull of (N, 2) points (Andrew's monotone chain), counter-clockwise."""
    pts = np.unique(xy, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(order) -> list[int]:
        out: list[int] = []
        for i in order:
            while len(out) >= 2:
                a, b = pts[out[-2]], pts[out[-1]]
                if (b[0] - a[0]) * (pts[i, 1] - a[1]) - (b[1] - a[1]) * (pts[i, 0] - a[0]) > 0.0:
                    break
                out.pop()
            out.append(i)
        return out

    lower, upper = half(range(len(pts))), half(range(len(pts) - 1, -1, -1))
    return pts[lower[:-1] + upper[:-1]]


def crown_wood(sk: Skeleton) -> np.ndarray:
    """(N, 3) the limb points that count toward the crown: not buried (roots), at or above the
    crown base (the soil, for a tree with no leaves)."""
    base = crown_base(sk)
    free = ~np.repeat(sk.buried, np.diff(sk.start))
    return sk.pts[free & (sk.pts[:, 1] >= (0.0 if base is None else base))]


def widest(xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(width, a, b): the widest distance between two of the (N, 2) points and its two ends. Width
    0 for fewer than two distinct points."""
    hull = _hull(xy)
    if len(hull) < 2:
        return 0.0, hull[0] if len(hull) else np.zeros(2), np.zeros(2)
    gaps = hull[:, None, :] - hull[None, :, :]
    d2 = (gaps * gaps).sum(axis=2)
    a, b = np.unravel_index(int(np.argmax(d2)), d2.shape)
    return float(np.sqrt(d2[a, b])), hull[a], hull[b]


def spread(sk: Skeleton) -> float:
    """The crown diameter: the mean of the crown's widest plan-view width and its width at right
    angles to it. 0 when the crown is a point."""
    xy = np.concatenate([crown_wood(sk), _leaf_vertices(sk)])[:, [0, 2]]
    wide, a, b = widest(xy)
    if not wide > 0.0:
        return 0.0
    along = (b - a) / wide
    across = _hull(xy) @ np.array([-along[1], along[0]])
    return 0.5 * (wide + float(across.max() - across.min()))


# --- which rules judge a piece -----------------------------------------------------------------


@dataclass(frozen=True)
class Standing:
    """Whether a grown piece can be judged: `kind`, the rules when found, and for a stale piece
    the (what, hash in the skeleton, hash in the scene) of each file that differs."""

    kind: str
    rules: GrowthRules | None = None
    stale: tuple[tuple[str, str, str], ...] = ()


def standing(ctx: SceneCtx, g: Grown) -> Standing:
    info = g.skeleton.info
    if "rules_sha256" not in info or "kit_sha256" not in info:
        return Standing(NOT_APPLICABLE)
    rules = ctx.rules.get(str(info.get("rules")))
    if rules is None:
        return Standing(MISSING)
    stale = tuple((what, str(grown_from), str(now)) for what, grown_from, now in (
        ("rules", info["rules_sha256"], rules.sha256), ("kit", info["kit_sha256"], rules.kit_sha256)
    ) if grown_from != now)
    return Standing(STALE if stale else APPLICABLE, rules, stale)


def _judged(ctx: SceneCtx) -> list[tuple[Grown, GrowthRules]]:
    return [(g, s.rules) for g in grown(ctx) if (s := standing(ctx, g)).kind == APPLICABLE]


# --- the findings ------------------------------------------------------------------------------


def _share_findings(g: Grown, rules: GrowthRules, only) -> list[Finding]:
    sk, found = g.skeleton, []
    top = height(sk)
    for key, value in ((SPREAD, spread(sk)), (CROWN_BASE, crown_base(sk))):
        item, rng = g.item(key), rules.ranges.get(key)
        if rng is None or value is None or not g.wants(only, item):
            continue
        share = value / top
        if not rng.lo <= share <= rng.hi:
            found.append(Finding(NAME, item, f"{share:.4f} of the height ({value:.4f} over "
                                             f"{top:.4f}), outside the cited range {rng.lo} to "
                                             f"{rng.hi} (rows {', '.join(rng.rows)} of "
                                             f"{rules.path.name})"))
    return found


def _stale_findings(g: Grown, stand: Standing, only) -> list[Finding]:
    found = []
    for what, grown_from, now in stand.stale:
        item = g.item(what)
        if g.wants(only, item):
            file = "the species file" if what == "rules" else "the kit file"
            found.append(Finding(NAME, item, f"{file} changed since the tree was grown: the "
                                             f"skeleton holds sha256 {grown_from}, the scene's "
                                             f"copy is {now}"))
    return found


# --- the plants --------------------------------------------------------------------------------


def _trunk_xz(sk: Skeleton) -> np.ndarray:
    """The x-z of the trunk's base: the thickest rooted limb that does not grow into the soil."""
    trunks = np.flatnonzero(sk.rooted & ~sk.buried)
    if not len(trunks):
        return sk.pts[0, [0, 2]]
    thickest = max(trunks, key=lambda i: float(sk.radii[sk.start[i]]))
    return sk.pts[sk.start[thickest], [0, 2]]


def _scaled(sk: Skeleton, factor: float, centre: np.ndarray) -> Skeleton:
    """The crown scaled across x and z about `centre`; y, the radii and the blades' frames keep."""
    def across(a: np.ndarray) -> np.ndarray:
        out = np.array(a)
        out[:, [0, 2]] = centre + (a[:, [0, 2]] - centre) * factor
        return out
    return dataclasses.replace(sk, pts=across(sk.pts), node=across(sk.node), base=across(sk.base))


def _spread_units(g: Grown, rules: GrowthRules, ctx: SceneCtx) -> list[PlantUnit]:
    """The crown widened across x and z about the trunk. The spread is the mean of the widest width
    and the width across it, so at least half the widest width of the crown wood (the hull of
    the wood and the blades only widens it); scaling the wood by `f` scales that width by `f`. So
    `f` = 2 x `RANGE_PLANT_OVER` x the range's upper end of the height over the wood's width puts the
    spread over the range, whatever the blades add. A crown of one spot cannot be widened by
    scaling: the unit holds no candidate, and the gate stops unless the real run already
    reports it."""
    sk, rng = g.skeleton, rules.ranges.get(SPREAD)
    if rng is None:
        return []  # no range to judge: the coverage line says so
    # The tree's top below is a deliberate copy of `height` (here and in `_bare_units`), so a
    # blinded `height` cannot move the plant (the guard refuses the original).
    item = g.item(SPREAD)
    wide, _, _ = widest(crown_wood(sk)[:, [0, 2]])
    if not wide > 0.0:
        return [PlantUnit(item, ())]
    factor = 2.0 * RANGE_PLANT_OVER * rng.hi * float(sk.pts[:, 1].max()) / wide
    moved = _scaled(sk, factor, _trunk_xz(sk))
    return [PlantUnit(item, (Plant(f"too wide: {item}", item, g.with_skeleton(ctx, moved)),))]


def _bare_units(g: Grown, rules: GrowthRules, ctx: SceneCtx) -> list[PlantUnit]:
    """No leaf below the line `RANGE_PLANT_OVER` x the range's upper end of the height: the
    leaves below it removed; or, if every leaf is below it, all leaves lifted until the lowest
    sits on it."""
    sk, rng = g.skeleton, rules.ranges.get(CROWN_BASE)
    if rng is None or not len(sk.host):
        return []  # no range to judge, or no leaf: the coverage line says which
    item = g.item(CROWN_BASE)
    line = RANGE_PLANT_OVER * rng.hi * float(sk.pts[:, 1].max())
    plants = []
    keep = sk.node[:, 1] >= line  # the crown base is the lowest leaf: strip every leaf below
    if keep.any():
        sel = dataclasses.replace(
            sk, host=sk.host[keep], leaf_t=sk.leaf_t[keep], node=sk.node[keep],
            base=sk.base[keep], mid=sk.mid[keep], across=sk.across[keep],
            normal=sk.normal[keep], length=sk.length[keep], petiole=sk.petiole[keep],
            shade=sk.shade[keep])
        plants.append(Plant(f"bare below: {item}", item, g.with_skeleton(ctx, sel)))
    up = np.array([0.0, line - float(sk.node[:, 1].min()), 0.0])
    if up[1] > 0.0:
        lifted = dataclasses.replace(sk, node=sk.node + up, base=sk.base + up)
        plants.append(Plant(f"lifted: {item}", item, g.with_skeleton(ctx, lifted)))
    return [PlantUnit(item, tuple(plants))]


class ReferenceRange:
    name = NAME
    kinds: frozenset[str] = KINDS
    scope = "piece"

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        return [u for g, rules in _judged(ctx) for make in (_spread_units, _bare_units)
                for u in make(g, rules, ctx)]

    def measured(self, ctx: SceneCtx) -> set[str]:
        """The tube and leaf parts of each judged piece: its limbs and leaves are read."""
        return {part for g, _ in _judged(ctx) for part in g.limb_parts() | g.leaf_parts()}

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        found = []
        for g in grown(ctx, only):
            stand = standing(ctx, g)
            if stand.kind == APPLICABLE and stand.rules is not None:
                found += _share_findings(g, stand.rules, only)
            elif stand.kind == STALE:
                found += _stale_findings(g, stand, only)
        return found

    def coverage(self, ctx: SceneCtx) -> list[str]:
        lines = coverage(ctx, NAME)
        for g in grown(ctx):
            stand = standing(ctx, g)
            if stand.kind == NOT_APPLICABLE:
                lines.append(f"{NAME}: not applicable (no rules hashes in the skeleton, so it "
                             f"was not grown by the bud grower): {g.key}")
            elif stand.kind == MISSING:
                lines.append(Unmeasured(f"{NAME}: skipped (rules file "
                                        f"{g.skeleton.info.get('rules')!r} is not in the scene's "
                                        f"rules/ folder): {g.key}"))
            elif stand.rules is not None:
                lines += _range_lines(g, stand.rules)
        return lines


def _range_lines(g: Grown, rules: GrowthRules) -> list[str]:
    lines = []
    if FORK in rules.ranges:
        lines.append(f"{NAME}: not judged (taste-only, no cited codominant definition): "
                     f"{g.item(FORK)}")
    for key in (SPREAD, CROWN_BASE):
        rng: Range | None = rules.ranges.get(key)
        if rng is None:
            lines.append(f"{NAME}: not judged ({rules.path.name} has no {key} range): "
                         f"{g.item(key)}")
        elif key == CROWN_BASE and crown_base(g.skeleton) is None:
            lines.append(f"{NAME}: not applicable (no leaves, so no crown base): {g.item(key)}")
    return lines
