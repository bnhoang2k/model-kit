"""The bud grower: a tree grown bud by bud, year by year (Palubicki et al. 2009).

Read it as a year loop. A tree is a heap of metamers: an internode (a stem piece), the node at
its top, and the buds in the leaf axils at that node (Barthelemy & Caraglio 2007). Each year:

1. every dormant bud reads its light from a shadow grid (`light.ShadowGrid`);
2. light flows down to the base, then resource flows back up, split at each node between the
   main axis and the side buds by the species' apical control (extended Borchert-Honda,
   Palubicki sec 4.2);
3. a bud grows a shoot of as many metamers as its share of resource allows (it needs at least one);
4. relay: a species with no terminal bud ends each shoot; the next year the lateral bud at its top
   node carries the axis on (and keeps the axis' order);
5. each new metamer turns toward the light, then toward its gravity set-point angle (GSA) by the
   sine law, at the kit's gravity rate; the soil is an obstacle (a shoot cannot grow into the
   ground), so a shoot ends at its last node above it, and a bud whose first node would lie in it
   stays asleep;
6. each new node sets its buds by the species' leaf arrangement (spiral, two-ranked, decussate);
7. the light is read again after the growth, and a side branch whose light per internode falls
   under the kit's threshold is shed, new shoots included (a shaded flush dies at the end of its
   season), the smallest branches first, and a relay shoot is never shed (Takenaka 1994, after
   Palubicki sec 4.4);
8. growth stops at the year the species' cited age gives for the asked-for height.

Three general rules were found to be needed for working trees, for
every species: a relay is never shed, shedding judges the smallest branches first, and the
resource split is Borchert-Honda's with a per-species apical control. Shed metamers stay in the
lists (indexes do not move) with `alive` False, and the skeleton leaves them out.

After the years, `_emit` finishes the tree:

- radii come from the tips down by the pipe model (`_radii`, Leonardo's rule with the kit's
  exponent): a tip has the twig's tip radius and a parent's radius to the power D is the sum of its
  children's. A shed limb still counts, so the wood it grew stays in the trunk (Palubicki 2009);
- limbs are continuation chains through the thickest child (`chains.continuation_chains`), roots
  are added at the trunk base (`_roots`), and the stop-short rule (`segments.Clearance`) cuts every
  limb where it would come within the clearance of a thicker limb made before it. It runs on the
  final radii and again until a pass cuts nothing (`_stop_short`);
- leaves sit on the nodes grown in the last year that stop-short left standing (`_leaves`): the
  petiole points out along the node's rank, the blade's angle from horizontal is drawn from the
  species' beta, and the blade hangs from the petiole's end toward the petiole's outward direction.
  A leaf that would reach below the soil is not placed (`_leaves`).

The tree grows in metres. At the end it is scaled uniformly to `height_m` (wood and, later, leaves),
and the raw height and the scale go to `info`, because a piece must be as tall as it was asked to be
(`blender.grow` checks `H / height_m`, the sheet render checks the built height). A limb in the
skeleton is a continuation chain (`chains.continuation_chains`).

Every kit constant comes from the merged kit rules file, every species number from the species
file, and every numeric of the algorithm from `shape.py`. `random.Random(seed)` is the only random
source.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.grow.chains import continuation_chains
from model_kit.core.grow.light import ShadowGrid
from model_kit.core.grow.rules import GrowthRules, RuleReader
from model_kit.core.grow.segments import Clearance, Limb, make_limbs
from model_kit.core.grow.shape import (
    CLEAR_SUBSTEPS,
    FLAT_EPS,
    FULL_TURN_DEG,
    HALF_TURN_DEG,
    KEEP_EPS,
    LEAF_BLADE_CENTRE,
    LEAF_FOLD,
    LEAF_PETIOLE_HALF_WIDTH_OF_TIP_RADIUS,
    LEAF_PETIOLE_TIP_OF_BLADE,
    LEAF_SHADES,
    PERP_EPS,
    PETIOLE_START_OF_RADIUS,
    QUARTER_TURN_DEG,
    RELAY_SHARE_OF_ANGLE,
    ROOT_BELOW_HORIZONTAL_DEG,
    ROOT_DROOP,
    ROOT_JITTER_DEG,
    ROOT_LENGTH_OF_RADIUS,
    ROOT_RADIUS_OF_TRUNK,
    ROOT_SEGMENTS,
    ROOT_TIP_OF_BASE,
    SCALE_MAX,
    SCALE_MIN,
    SELF_PIECE_OF_WINDOW,
    SHADOW_CELL_OF_INTERNODE,
    SIDE,
    SOIL_Y,
    STOP_SHORT_PASSES,
    TRANSPORT_EPS,
    TRUNK_SINK_OF_RADIUS,
    UP,
)
from model_kit.core.grow.skeleton import CONE, Skeleton, arc_lengths, check_skeleton, unit

MESH_V1 = "buds-v1"  # `info["mesh"]` of a tree grown from a rules file with `tubes.sides`
MESH_VERSION = "buds-v2"  # `info["mesh"]` of a tree grown without it: the mesh reads the view
TERMINAL = -1  # a bud's slot when it is the axis' terminal bud; laterals are slots 0, 1, ...
TERMINAL_NAMES = ("present", "absent")
# The params a lite grow reads in place of a full one (`_read`'s `pick`), and those only a lite
# grow reads. `blender.grow.used_params` lists a part's rows from them. A buds-v2 rules file swaps
# nothing: its tube sides come from the view at mesh time. This pair is the buds-v1 swap, for the
# rules files that still carry `tubes.sides`.
V1_LITE_SWAPS = {"tubes.sides": "lite.tube_sides"}
LITE_ONLY = ("lite.leaf_outline", "lite.leaf_thin")


# ------------------------------------------------------------------------------------- helpers

def _rotate(v: np.ndarray, axis: np.ndarray, rad: float) -> np.ndarray:
    """v turned `rad` radians about `axis`, right-handed (Rodrigues)."""
    k = unit(axis)
    c, s = math.cos(rad), math.sin(rad)
    return v * c + np.cross(k, v) * s + k * (k @ v) * (1.0 - c)


def _horizontal_side(d: np.ndarray) -> np.ndarray:
    """The horizontal unit vector at right angles to d (SIDE when d is vertical)."""
    s = np.cross(d, UP)
    return unit(s) if float(np.sqrt(s @ s)) > PERP_EPS else SIDE.copy()


def _transport(ref: np.ndarray, d_old: np.ndarray, d_new: np.ndarray) -> np.ndarray:
    """ref carried along an axis that turns from d_old to d_new by the smallest turn, so it keeps
    its place round the axis."""
    turn = np.cross(d_old, d_new)
    s = float(np.sqrt(turn @ turn))
    if s < TRANSPORT_EPS:
        return ref
    moved = _rotate(ref, turn, math.atan2(s, float(d_old @ d_new)))
    return unit(moved - d_new * (moved @ d_new))


def _bend_to_gsa(par: _Params, d: np.ndarray, order: int) -> np.ndarray:
    """d turned toward the gravity set-point of an axis of `order`, by a sine law (a model choice,
    not a cited formula): the turn is `gravity x sin(angle between d and the set-point)`, so a
    shoot at right angles to its set-point turns the most and one on it not at all. The set-point keeps d's
    compass heading and lies `gsa` degrees from gravity (180 is straight up, 90 horizontal); an
    order past the list's end takes the last entry. The turn never passes the set-point."""
    gsa = par.gsa[min(order, len(par.gsa) - 1)]
    rise = math.radians(gsa - QUARTER_TURN_DEG)
    flat = np.array([d[0], 0.0, d[2]])
    heading = unit(flat) if float(np.sqrt(flat @ flat)) > PERP_EPS else _horizontal_side(d)
    target = unit(heading * math.cos(rise) + UP * math.sin(rise))
    turn = np.cross(d, target)
    s = float(np.sqrt(turn @ turn))
    if s < TRANSPORT_EPS:
        return d
    gap = math.atan2(s, float(d @ target))
    return unit(_rotate(d, turn, min(par.gravity * math.sin(gap), gap)))


def years_for_height(points: np.ndarray, height_m: float) -> int:
    """The age `age.height_points` ([height_m, years] pairs) gives for a tree of `height_m`:
    straight lines between the points, the end values past the ends, rounded to a whole year, at
    least one."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    order = np.argsort(points[:, 0], kind="stable")
    heights, years = points[order, 0], points[order, 1]
    if np.any(np.diff(heights) <= 0.0):
        raise ManifestError("age.height_points: the heights must all differ")
    return max(1, math.floor(float(np.interp(height_m, heights, years)) + 0.5))


def height_at_years(points: np.ndarray, years: int) -> float:
    """The height `age.height_points` cites for a tree of `years`: straight lines between the
    points, the end values past the ends. The inverse of `years_for_height` before its rounding."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    order = np.argsort(points[:, 1], kind="stable")
    return float(np.interp(years, points[order, 1], points[order, 0]))


@dataclass(frozen=True)
class _Params:
    """Every number the year loop reads: the kit's constants and the species' facts, in metres
    and degrees."""

    cell: float  # shadow cell edge
    depth: int  # shadow pyramid layers
    a: float
    b: float
    full: float  # full exposure C
    alpha: float  # vigour per unit of light
    apical: float  # apical control: the main axis' share of the Borchert-Honda split, in (0, 1)
    light_weight: float  # how hard a shoot turns toward the light
    gravity: float  # sine-law rate of a shoot's turn toward its set-point, radians per metamer
    shed: float  # a side branch under this much light per internode is shed
    gsa: tuple[float, ...]  # set-point angle from gravity per axis order, degrees (180 is up)
    flat: bool  # two-ranked leaves lie flat on a plagiotropic axis
    round_within: float  # degrees from vertical: within this, ranks go round the axis
    flat_from: float  # degrees from vertical: past this, ranks lie flat; between, a smooth change
    divergence: float  # degrees from one node to the next
    per_node: int  # leaves (and buds) at a node
    has_terminal: bool
    internode: float  # metres
    angle: tuple[float, float]  # a lateral shoot's angle from its parent axis, degrees
    tip_radius: float  # metres
    exponent: float  # Leonardo's pipe exponent D
    clearance: float  # stop-short: the gap a limb keeps from a thicker one, in thinner radii
    beta: tuple[float, float]  # the blade angle's Beta(a, b) on t = 2 theta / pi
    leaf_size: tuple[float, float]  # blade length range, metres
    petiole_of_blade: float  # petiole length over blade length (0 in lite: no petiole)
    current_year: bool  # leaves sit on the last year's nodes only
    roots: int  # buried roots at the trunk base
    thin: int  # lite keeps one leaf in `thin` of each shoot (1 in full)

    @property
    def distichous(self) -> bool:
        """Two-ranked: one leaf per node, each 180 degrees from the last."""
        return self.per_node == 1 and math.isclose(self.divergence, HALF_TURN_DEG)


def _start_phyl(par: _Params, rng: random.Random) -> float:
    """Where a new axis starts in its leaf arrangement: either side first when two-ranked (so the
    sign of a rank is steady), any angle otherwise."""
    if par.distichous:
        return rng.choice((0.0, HALF_TURN_DEG))
    return rng.uniform(0.0, FULL_TURN_DEG)


def _flat_weight(par: _Params, d: np.ndarray) -> float:
    """How flat the ranks of a two-ranked axis lie where it points along `d`: 0 within
    `round_within` degrees of vertical, 1 past `flat_from`, a smoothstep between (so no jump)."""
    if not (par.flat and par.distichous):
        return 0.0
    away = math.degrees(math.acos(min(1.0, abs(float(d[1])))))
    x = min(1.0, max(0.0, (away - par.round_within) / (par.flat_from - par.round_within)))
    return x * x * (3 - 2 * x)


def _rank_parity(phyl: float) -> float:
    """+1 or -1: which of the two ranks a two-ranked node is on (the rank turns by half each node)."""
    return 1.0 if math.cos(math.radians(phyl)) >= 0.0 else -1.0


def _flat_anchor(par: _Params, d: np.ndarray, ref: np.ndarray, phyl: float,
                 last: float | None) -> float:
    """The sign that ties a node's flat side to its rank: the flat side is `anchor x parity x h`
    (h the horizontal side). Where the ranks go round the axis, or on a new axis (`last` None), it
    is read from which horizontal side is nearest the round side; where the ranks are going
    flat it is the last node's, so the flat sign stays with the carried rank and does not flip
    when the round side drifts a quarter turn from the horizontal."""
    if last is not None and _flat_weight(par, d) > 0.0:
        return last
    near = float(_rotate(ref, d, math.radians(phyl)) @ _horizontal_side(d))
    return (1.0 if near >= 0.0 else -1.0) * _rank_parity(phyl)


def _rank_sides(par: _Params, d: np.ndarray, ref: np.ndarray, phyl: float,
                anchor: float | None = None) -> np.ndarray:
    """(per_node, 3) the directions out of the axis, at right angles to it, where the node's buds
    point. Each is the round side (`ref` turned `phyl` degrees, then evenly round the node). A
    two-ranked axis mixes in the flat side by `_flat_weight`: the horizontal side on this node's
    rank (`_flat_anchor`), so each node still lies opposite the last. Past `flat_from` degrees
    from vertical the buds lie in the horizontal plane."""
    w = _flat_weight(par, d)
    out = []
    for j in range(par.per_node):
        side = _rotate(ref, d, math.radians(phyl + j * FULL_TURN_DEG / par.per_node))
        if w > 0.0:
            h = _horizontal_side(d)
            sign = _flat_anchor(par, d, ref, phyl, None) if anchor is None else anchor
            flat = sign * _rank_parity(phyl) * h
            # Turn the round side toward the flat one by the weight's share of the angle between
            # them (the short way), so the side changes smoothly with the weight.
            gap = math.atan2(float(np.cross(side, flat) @ d), float(side @ flat))
            side = _rotate(side, d, w * gap)
        out.append(side)
    return np.array(out)


# ----------------------------------------------------------------------------------- the tree

class _Metamers:
    """Metamers in parallel lists. Metamer `i` is an internode from `start[i]` to `end[i]` along
    `dir[i]`; its node (at `end[i]`) holds `per_node` lateral buds, pointing out along
    `side[i][j]`. A parent comes before its children. `main[i]` is the metamer that carries the
    axis on from i (-1 at an axis' tip) and `lat[i][j]` the first metamer of the shoot that bud j
    grew (-1 while it sleeps or after the shoot was shed). `relay[i]` marks the first metamer of a
    relay shoot. `emerge[i]` is the direction the bud of i's axis came out in, before any bend.
    A shed metamer stays in the lists with `alive[i]` False. `ref` and `phyl` are the
    axis' carried reference and its leaf-arrangement angle in degrees at this node. `buds` are the
    dormant buds, as (metamer, slot) pairs; `fresh` are this year's new ones, which wait for
    next year."""

    def __init__(self, par: _Params) -> None:
        self.par = par
        self.start: list[np.ndarray] = []
        self.end: list[np.ndarray] = []
        self.dir: list[np.ndarray] = []
        self.emerge: list[np.ndarray] = []
        self.side: list[np.ndarray] = []
        self.parent: list[int] = []
        self.main: list[int] = []
        self.lat: list[list[int]] = []
        self.relay: list[bool] = []
        self.alive: list[bool] = []
        self.order: list[int] = []
        self.axis: list[int] = []
        self.year: list[int] = []
        self.ref: list[np.ndarray] = []
        self.phyl: list[float] = []
        self.anchor: list[float] = []
        self.buds: list[tuple[int, int]] = []
        self.fresh: list[tuple[int, int]] = []
        self.axes = 0
        self.rng: random.Random | None = None  # the grow's random stream, which `_emit` carries on

    def __len__(self) -> int:
        return len(self.start)

    def new_axis(self) -> int:
        self.axes += 1
        return self.axes - 1

    def add(self, start: np.ndarray, d: np.ndarray, length: float, parent: int, order: int,
            axis: int, year: int, ref: np.ndarray, phyl: float, link: int = TERMINAL,
            relay: bool = False, emerge: np.ndarray | None = None) -> int:
        """A new metamer on `parent`: its main child when `link` is TERMINAL, else the shoot of
        the parent's lateral bud `link`."""
        i = len(self.start)
        self.start.append(start)
        self.end.append(start + d * length)
        self.dir.append(d)
        self.emerge.append(d if emerge is None else emerge)
        last = self.anchor[parent] if parent >= 0 and self.axis[parent] == axis else None
        anchor = _flat_anchor(self.par, d, ref, phyl, last)
        self.anchor.append(anchor)
        self.side.append(_rank_sides(self.par, d, ref, phyl, anchor))
        self.parent.append(parent)
        self.main.append(-1)
        self.lat.append([-1] * self.par.per_node)
        self.relay.append(relay)
        self.alive.append(True)
        self.order.append(order)
        self.axis.append(axis)
        self.year.append(year)
        self.ref.append(ref)
        self.phyl.append(phyl)
        if parent >= 0:
            if link == TERMINAL:
                self.main[parent] = i
            else:
                self.lat[parent][link] = i
        self.fresh.extend((i, j) for j in range(self.par.per_node))
        return i

    def settle(self) -> None:
        """The year's new buds join the dormant ones."""
        self.buds.extend(self.fresh)
        self.fresh = []

    def is_relay_bud(self, m: int, slot: int) -> bool:
        """With no terminal bud, the first lateral bud at the tip node of an axis carries the axis
        on, at the axis' own order. Every other lateral bud starts a side shoot."""
        return not self.par.has_terminal and slot == 0 and self.main[m] < 0


def _shed(tree: _Metamers, q_at: list[float], threshold: float, n0: int) -> int:
    """Step 7 (Takenaka 1994, after Palubicki sec 4.4): a side branch whose light per internode
    falls under `threshold` dies, with all it carries. `q_at[m]` is the light of the buds that sleep
    at metamer m, read after the year's growth, so a shoot grown this year is judged with the
    older branches: a shaded flush dies at the end of its season. Only the first `n0` metamers are
    judged (the grower passes them all). Branches are judged smallest first (children before
    parents, and a shed sub-branch no longer counts in its parent branch), and a relay shoot is
    never shed: it is the axis going on. Returns how many branches were shed."""
    light = [0.0] * n0
    count = [0] * n0
    shed = 0
    for i in range(n0 - 1, -1, -1):
        if not tree.alive[i]:
            continue
        light[i], count[i] = q_at[i], 1
        children = [tree.main[i]] + tree.lat[i]
        for slot, c in enumerate(children):
            if not 0 <= c < n0 or not tree.alive[c]:
                continue
            if slot > 0 and not tree.relay[c] and light[c] / count[c] < threshold:
                shed += 1
                _kill(tree, c)
                tree.lat[i][slot - 1] = -1
            else:
                light[i] += light[c]
                count[i] += count[c]
    tree.buds = [b for b in tree.buds if tree.alive[b[0]]]
    return shed


def _kill(tree: _Metamers, root: int) -> None:
    """Metamer `root` and everything grown from it die."""
    stack = [root]
    while stack:
        k = stack.pop()
        if k < 0 or not tree.alive[k]:
            continue
        tree.alive[k] = False
        stack.append(tree.main[k])
        stack.extend(tree.lat[k])


class _Year:
    """One year: the dormant buds, their shade and light, their share of resource, and the
    shoots they grow."""

    def __init__(self, tree: _Metamers, par: _Params, year: int) -> None:
        self.tree, self.par, self.year = tree, par, year
        self.n0 = len(tree)  # the metamers that stand when the year begins
        self.buds = list(tree.buds)
        # Step 1: Q per bud, and the way to the light.
        self.light, self.toward = self._read_light(self.buds)

    def _read_light(self, buds: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
        """The light of each of `buds` (a bud's own node does not shade it: the buds of one node
        sit at one point and do not shade each other) and the direction where its shade is least."""
        tree, par = self.tree, self.par
        pos = np.array([tree.end[m] for m, _ in buds]).reshape(-1, 3)
        grid = ShadowGrid(par.cell)
        grid.add_shadow(pos, par.depth, par.a, par.b)
        at_node = Counter(m for m, _ in buds)
        own = np.array([at_node[m] for m, _ in buds], dtype=int)
        return grid.light(pos, par.full, own=own), grid.toward_light(pos)

    def _light_at_nodes(self, buds: list[tuple[int, int]]) -> list[float]:
        """The summed light of the buds at each metamer, read now."""
        light, _ = self._read_light(buds)
        q_at = [0.0] * len(self.tree)
        for (m, _), q in zip(buds, light, strict=True):
            q_at[m] += float(q)
        return q_at

    def vigour(self) -> dict[tuple[int, int], float]:
        """Step 2, the extended Borchert-Honda split (Palubicki sec 4.2). Light flows down to the
        base: a node's main light `Qm` is its terminal bud's plus the main child's whole, its
        lateral light `Ql` the lateral buds' and children's. Resource `alpha x Q` at the base
        flows back up; a node with resource v gives its main side `v lam Qm / (lam Qm + (1 - lam)
        Ql)` and its lateral side the rest, and the laterals share theirs by their light. Returns
        the resource that reaches each dormant bud."""
        tree, par, n0 = self.tree, self.par, self.n0
        q_bud = {b: float(q) for b, q in zip(self.buds, self.light, strict=True)}
        lam = par.apical
        q_main, q_lat, q_sub = [0.0] * n0, [0.0] * n0, [0.0] * n0
        slot_q: list[list[float]] = [[] for _ in range(n0)]
        for i in range(n0 - 1, -1, -1):
            if not tree.alive[i]:
                continue
            c = tree.main[i]
            q_main[i] = q_bud.get((i, TERMINAL), 0.0) + (q_sub[c] if 0 <= c < n0 else 0.0)
            slot_q[i] = [q_sub[c] if 0 <= c < n0 else q_bud.get((i, j), 0.0)
                         for j, c in enumerate(tree.lat[i])]
            q_lat[i] = sum(slot_q[i])
            q_sub[i] = q_main[i] + q_lat[i]
        resource = [0.0] * n0
        resource[0] = par.alpha * q_sub[0]
        out: dict[tuple[int, int], float] = {}
        for i in range(n0):
            v = resource[i]
            if not tree.alive[i] or v <= 0.0:
                continue
            den = lam * q_main[i] + (1.0 - lam) * q_lat[i]
            if den <= 0.0:
                continue
            c = tree.main[i]
            if 0 <= c < n0:
                resource[c] = v * lam * q_main[i] / den
            elif q_main[i] > 0.0:
                out[(i, TERMINAL)] = v * lam * q_main[i] / den
            if q_lat[i] > 0.0:
                lateral = v * (1.0 - lam) * q_lat[i] / den
                for j, c in enumerate(tree.lat[i]):
                    share = lateral * slot_q[i][j] / q_lat[i]
                    if 0 <= c < n0:
                        resource[c] = share
                    elif share > 0.0:
                        out[(i, j)] = share
        return out

    def grow(self, rng: random.Random) -> int:
        """Steps 2 to 7: every bud with resource for a metamer grows, then side branches in too
        little light are shed. Returns how many metamers the year made."""
        tree, par = self.tree, self.par
        resource = self.vigour()
        kept: list[tuple[int, int]] = []
        made = 0
        for k, (m, slot) in enumerate(self.buds):
            vigour = resource.get((m, slot), 0.0)
            n = int(vigour)
            grew = self._shoot(rng, m, slot, n, par.internode * vigour / n,
                               self.toward[k]) if n >= 1 else 0
            if grew < 1:  # no resource for a metamer, or its first node would be in the soil
                kept.append((m, slot))
            made += grew
        tree.buds = kept
        tree.settle()
        # Step 7 reads the light again, after the growth: this year's shoots shade the tree and
        # have buds of their own, so they are judged with the older branches.
        _shed(tree, self._light_at_nodes(tree.buds), par.shed, len(tree))
        return made

    def _shoot(self, rng: random.Random, m: int, slot: int, n: int, length: float,
               toward_light: np.ndarray) -> int:
        """Bud (m, slot) grows a shoot of up to n metamers of `length`. Returns how many it grew.

        The soil is an obstacle: a shoot cannot grow into the ground, so it ends at its last node
        above the soil. The test is on each new node after the gravity bend, the last turn a
        metamer takes: emergence alone cannot see a shoot that the bend then carries down over
        several metamers. A bud whose first node would lie in the soil grows nothing (the caller
        leaves it asleep, and no terminal bud is set on its node)."""
        tree, par = self.tree, self.par
        relay = False
        if slot == TERMINAL:  # the axis goes on
            d, order, axis = tree.dir[m], tree.order[m], tree.axis[m]
            ref, phyl = tree.ref[m], tree.phyl[m]
            emerge = tree.emerge[m]
        else:  # a lateral bud starts a new axis
            relay = tree.is_relay_bud(m, slot)
            beta = math.radians(rng.uniform(*par.angle))  # drawn for a relay too: same stream
            # A relay takes the lost terminal's place, so it goes on nearly along the axis (a
            # small zig-zag, as a sympodial stem has); a side shoot leaves at the branch angle.
            # At the full angle a relay axis kinks every year and curls into loops.
            if relay:
                beta *= RELAY_SHARE_OF_ANGLE
            d = unit(tree.dir[m] * math.cos(beta) + tree.side[m][slot] * math.sin(beta))
            order = tree.order[m] + (0 if relay else 1)  # a relay is the same axis going on
            axis = tree.new_axis()
            ref = _horizontal_side(d)
            phyl = _start_phyl(par, rng) - par.divergence
            emerge = d
        parent, point, link = m, tree.end[m], slot
        grew = 0
        for _ in range(n):
            bent = unit(d + par.light_weight * toward_light)
            d_new = _bend_to_gsa(par, bent if bent.any() else d, order)
            if point[1] + d_new[1] * length < SOIL_Y:
                break
            grew += 1
            ref = _transport(ref, d, d_new)
            phyl += par.divergence
            parent = tree.add(point, d_new, length, parent, order, axis, self.year, ref, phyl,
                              link=link, relay=relay and link == slot, emerge=emerge)
            point, d, link = tree.end[parent], d_new, TERMINAL
        if par.has_terminal and grew:
            tree.fresh.append((parent, TERMINAL))
        return grew


# --------------------------------------------------------------------------------- the grower

@dataclass(frozen=True)
class _Setup:
    """Everything `_read` takes from the rules: the year loop's params, the age points, the
    skeleton's tube sides per level and level names, and the blade outlines (the one the mesh
    draws, and the full one lite's leaf area is measured against)."""

    par: _Params
    ages: np.ndarray
    sides: np.ndarray  # the buds-v1 sides per level; all 0 in a buds-v2 tree
    levels: tuple[str, ...]
    outline: np.ndarray
    full_outline: np.ndarray
    full_petiole: float  # the full tree's petiole over blade length, which lite's soil test reads
    lite: bool
    mesh: str  # `info["mesh"]`: buds-v1 when the rules carry `tubes.sides`, else buds-v2


def _read(r: RuleReader, lite: bool) -> _Setup:
    """Every param the year loop and the skeleton read, once, up front. A param of the other mode
    (full or lite) is not read on this grow; any other param left unread raises, naming it."""
    passed: set[str] = set()

    def pick(full_key: str, lite_key: str) -> str:
        passed.add(full_key if lite else lite_key)
        return lite_key if lite else full_key

    def fail(why: str):
        raise ManifestError(f"{r.rules.path}: params: {why}")

    terminal = r.names("buds.terminal")
    if len(terminal) != 1 or terminal[0] not in TERMINAL_NAMES:
        fail(f"'buds.terminal' must be one of {list(TERMINAL_NAMES)} in a list of one, "
             f"got {list(terminal)!r}")
    round_within, flat_from = r.num("ranks.round_within_vertical_deg"), r.num(
        "ranks.flat_from_vertical_deg")
    if not 0.0 <= round_within < flat_from <= QUARTER_TURN_DEG:
        fail(f"'ranks.round_within_vertical_deg' and 'ranks.flat_from_vertical_deg' must satisfy "
             f"0 <= round < flat <= {QUARTER_TURN_DEG:g}, got {round_within!r} and {flat_from!r}")
    per_node = r.int("leaves.per_node")
    if per_node < 1:
        fail(f"'leaves.per_node' must be 1 or more, got {per_node!r}")
    levels = r.names("levels.names")
    if "tubes.sides" in r.rules.params:  # a buds-v1 rules file: its sides are the mesh's
        sides = np.array(r.int_choices(pick("tubes.sides", V1_LITE_SWAPS["tubes.sides"])),
                         dtype=int)
        if len(sides) != len(levels):
            fail("tube sides need one count per level")
        mesh = MESH_V1
    else:  # buds-v2: the sides come from the piece's view when the tree is meshed
        sides = np.zeros(len(levels), dtype=int)
        mesh = MESH_VERSION
    internode = r.num("internode.length_m")
    if not internode > 0.0:
        fail(f"'internode.length_m' must be over 0, got {internode!r}")
    apical = r.num("apical.control")
    if not 0.0 < apical < 1.0:
        fail(f"'apical.control' must lie between 0 and 1, got {apical!r}")
    gsa = r.choices("gsa.by_order_deg")
    if not gsa or not all(0.0 <= g <= HALF_TURN_DEG for g in gsa):
        fail(f"'gsa.by_order_deg' must be angles from gravity between 0 and {HALF_TURN_DEG:g}, "
             f"one per axis order, got {list(gsa)!r}")
    gravity = r.num("growth.gravity_rate")
    if not 0.0 <= gravity <= 1.0:
        fail(f"'growth.gravity_rate' must lie between 0 and 1, got {gravity!r}")
    shed = r.num("shed.light_per_internode")
    if shed < 0.0:
        fail(f"'shed.light_per_internode' must be 0 or more, got {shed!r}")
    tip_radius = r.num("twig.tip_radius_m")
    if not tip_radius > 0.0:
        fail(f"'twig.tip_radius_m' must be over 0, got {tip_radius!r}")
    exponent = r.num("leonardo.exponent")
    if not exponent > 0.0:
        fail(f"'leonardo.exponent' must be over 0, got {exponent!r}")
    clearance = r.num("limbs.clearance_of_radius")
    if clearance < 0.0:
        fail(f"'limbs.clearance_of_radius' must be 0 or more, got {clearance!r}")
    beta = r.choices("leaves.angle_beta")
    if len(beta) != 2 or not all(b > 0.0 for b in beta):
        fail(f"'leaves.angle_beta' must be [a, b] with both over 0, got {list(beta)!r}")
    leaf_size = r.range("leaves.length_m")
    if not leaf_size[0] > 0.0:
        fail(f"'leaves.length_m' must be a range over 0, got {list(leaf_size)!r}")
    full_outline = r.points("leaves.outline")
    # Lite reads the full petiole too: the soil test judges each leaf as the full tree draws it.
    full_petiole = r.num("leaves.petiole_of_blade")
    if full_petiole < 0.0:
        fail(f"'leaves.petiole_of_blade' must be 0 or more, got {full_petiole!r}")
    outline, thin, petiole = full_outline, 1, full_petiole
    if lite:  # a lite leaf has no petiole: the blade sits on the node
        outline, thin, petiole = r.points("lite.leaf_outline"), r.int("lite.leaf_thin"), 0.0
        if thin < 1:
            fail(f"'lite.leaf_thin' must be 1 or more, got {thin!r}")
    else:
        passed.update(LITE_ONLY)
    if len(full_outline) < 3 or len(outline) < 3:
        fail(f"a blade outline needs 3 points or more, got {len(full_outline)} and {len(outline)}")
    roots = r.int("roots.count")
    if roots < 0:
        fail(f"'roots.count' must be 0 or more, got {roots!r}")
    par = _Params(
        cell=internode * SHADOW_CELL_OF_INTERNODE,
        depth=r.int("light.pyramid_depth"),
        a=r.num("light.shadow_a"),
        b=r.num("light.shadow_b"),
        full=r.num("light.full_exposure"),
        alpha=r.num("resource.alpha"),
        apical=apical,
        light_weight=r.num("growth.light_weight"),
        gravity=gravity,
        shed=shed,
        gsa=gsa,
        flat=bool(r.int("ranks.plagiotropic_flat")),
        round_within=r.num("ranks.round_within_vertical_deg"),
        flat_from=r.num("ranks.flat_from_vertical_deg"),
        divergence=r.num("leaves.divergence_deg"),
        per_node=per_node,
        has_terminal=terminal[0] == TERMINAL_NAMES[0],
        internode=internode,
        angle=r.range("branch.angle_deg"),
        tip_radius=tip_radius,
        exponent=exponent,
        clearance=clearance,
        beta=(beta[0], beta[1]),
        leaf_size=leaf_size,
        petiole_of_blade=petiole,
        current_year=bool(r.int("leaves.on_current_year")),
        roots=roots,
        thin=thin,
    )
    setup = _Setup(par, r.points("age.height_points"), sides, levels, outline, full_outline,
                   full_petiole, lite, mesh)
    unread = sorted(set(r.rules.params) - r.read - passed)
    if unread:
        fail(f"the grower never reads {unread}")
    return setup


def _grow_tree(setup: _Setup, *, height_m: float, seed: int) -> tuple[_Metamers, int]:
    """The year loop. Returns the metamers and the years it grew."""
    par = setup.par
    years = years_for_height(setup.ages, height_m)
    rng = random.Random(seed)
    tree = _Metamers(par)
    tree.rng = rng
    # The seedling: one metamer, standing straight up, in year 0.
    ref = _rotate(SIDE, UP, math.radians(rng.uniform(0.0, FULL_TURN_DEG)))
    tree.add(np.zeros(3), UP.copy(), par.internode, -1, 0, tree.new_axis(), 0, ref,
             _start_phyl(par, rng))
    if par.has_terminal:
        tree.fresh.append((0, TERMINAL))
    tree.settle()
    for year in range(1, years + 1):
        if not tree.buds:
            break
        _Year(tree, par, year).grow(rng)
    return tree, years


def grow_buds(rules: GrowthRules, *, piece: str, height_m: float, seed: int, lite: bool,
              units_per_metre: float) -> Skeleton:
    """Grow one tree from a species file merged with the kit file (`load_rules(path)`). The tree
    is scaled to `height_m` metres of `units_per_metre` model units each."""
    if not height_m > 0.0:
        raise ManifestError(f"height_m must be over 0, got {height_m!r}")
    if not units_per_metre > 0.0:
        raise ManifestError(f"units_per_metre must be over 0, got {units_per_metre!r}")
    setup = _read(RuleReader(rules), lite)
    cited = np.asarray(setup.ages, dtype=float).reshape(-1, 2)[:, 0]
    if not cited.min() <= height_m <= cited.max():
        raise ManifestError(
            f"{piece}: {height_m:.4g} m is outside the heights age.height_points cites "
            f"({cited.min():.4g} to {cited.max():.4g} m), so no cited age grows it ({rules.path})")
    tree, years = _grow_tree(setup, height_m=height_m, seed=seed)
    return _emit(tree, setup, rules, piece=piece, height_m=height_m, seed=seed, lite=lite,
                 years=years, upm=units_per_metre)


# --------------------------------------------------------------------------- radii and leaves

def _radii(tree: _Metamers) -> np.ndarray:
    """The pipe model, tips down (Leonardo's rule, Eloy 2011; Palubicki et al. 2009 sec 4.5): a tip
    metamer has the twig's tip radius and any other has the radius whose power D is the sum of its
    children's powers. Shed metamers are counted: the wood a shed limb grew stays in its parent, so
    shedding never thins the trunk (Palubicki 2009). A metamer sits after its parent in the lists,
    so one pass from the end sees every child before its parent."""
    par = tree.par
    n = len(tree)
    power = np.zeros(n)
    radius = np.zeros(n)
    for i in range(n - 1, -1, -1):
        radius[i] = par.tip_radius if power[i] == 0.0 else power[i] ** (1.0 / par.exponent)
        if tree.parent[i] >= 0:
            power[tree.parent[i]] += radius[i] ** par.exponent
    return radius


def _blade_angle(rng: random.Random, a: float, b: float) -> float:
    """A blade's angle from horizontal, in radians: theta = pi/2 x Beta(a, b), so t = 2 theta / pi
    is Beta(a, b), the de Wit form the leaf-inclination data are fitted with (Chianucci et al.
    2018). a and b are the species' beta, as numpy's `rng.beta(a, b)` takes them."""
    return 0.5 * math.pi * rng.betavariate(a, b)


def _roots(par: _Params, rng: random.Random, trunk: Limb) -> list[Limb]:
    """`roots.count` buried limbs from the trunk's base point, which sits at the soil (the seedling
    starts at y = 0), so every root leaves within one trunk radius of it. They are spread evenly
    round the trunk with a jitter, start below horizontal and curve down, so the tip is in the
    soil. How a root is drawn is `shape.ROOT_*`. They are level 0, so they share the trunk's part
    (Trunk_Roots), and a root carries no limb: the chain orders of the crown count from the trunk."""
    if not par.roots:
        return []
    base, radius = trunk.pts[1], float(trunk.radii[1])  # point 0 is the trunk's sunk foot
    seat = float(trunk.cum[1] / trunk.length)
    start = rng.uniform(0.0, FULL_TURN_DEG)
    out = []
    for k in range(par.roots):
        az = math.radians(start + FULL_TURN_DEG / par.roots * k
                          + rng.uniform(-ROOT_JITTER_DEG, ROOT_JITTER_DEG))
        el = math.radians(-ROOT_BELOW_HORIZONTAL_DEG)
        d = np.array([math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)])
        step = radius * rng.uniform(*ROOT_LENGTH_OF_RADIUS) / ROOT_SEGMENTS
        pts = [base]
        for j in range(1, ROOT_SEGMENTS + 1):
            d = unit(d - UP * (ROOT_DROOP * 2.0 * j / ROOT_SEGMENTS / ROOT_SEGMENTS))
            pts.append(pts[-1] + d * step)
        r0 = ROOT_RADIUS_OF_TRUNK * radius
        rad = [r0 * (1.0 - (1.0 - ROOT_TIP_OF_BASE) * j / ROOT_SEGMENTS)
               for j in range(ROOT_SEGMENTS + 1)]
        out.append(Limb(f"root.{k}", 0, np.array(pts), rad, trunk, seat, buried=True))
    return out


def _stop_short(grown: list[Limb], share: float) -> list[Limb]:
    """The stop-short rule on the finished tree, on the final radii: every limb is made thick level
    first (`segments.make_limbs`), cut where it would come within `share` times the thinner radius
    of a limb already made, with pieces no longer than the checks cut them
    (`SELF_PIECE_OF_WINDOW`). Then each limb's kept part is taken as its plan and the pass is run
    again, until a pass cuts nothing more: the checks judge the finished tree, so what the grower
    leaves must hold when it is looked at again. Returns the limbs made, in growing order."""
    def clearance() -> Clearance:
        return Clearance(share, CLEAR_SUBSTEPS, SELF_PIECE_OF_WINDOW)

    made = make_limbs(grown, clearance())
    for _ in range(STOP_SHORT_PASSES):
        before = {id(limb): limb.length for limb in made}
        for limb in made:
            limb.replan()
        for limb in made:  # a seat is a fraction of the parent's plan, which may have got shorter
            if limb.on is not None:
                limb.t *= before[id(limb.on)] / limb.on.length
        again = make_limbs(made, clearance())
        settled = len(again) == len(made) and all(limb.keep >= limb.length for limb in again)
        made = again
        if settled:
            return made
    raise ManifestError(f"stop-short was still cutting after {STOP_SHORT_PASSES} passes; "
                        "this is a model failure, not a fit miss")


def _polygon_area(outline: np.ndarray) -> float:
    x, y = outline[:, 0], outline[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _blade_frame(petiole: np.ndarray, heading: np.ndarray, theta: float
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mid, across, normal) of a blade hung from a petiole pointing along `petiole`, at `theta`
    from horizontal. The blade tilts toward the petiole's outward direction: it droops out along
    the horizontal part of the petiole, so the face looks up and out (the normal's horizontal part
    is the petiole's). Where the petiole is vertical it has no outward side, and the blade takes
    the heading of the shoot it sits on."""
    out = np.array([petiole[0], 0.0, petiole[2]])
    if float(np.sqrt(out @ out)) <= FLAT_EPS:
        out = np.array([heading[0], 0.0, heading[2]])
    out = unit(out) if float(np.sqrt(out @ out)) > FLAT_EPS else SIDE.copy()
    mid = out * math.cos(theta) - UP * math.sin(theta)
    normal = out * math.sin(theta) + UP * math.cos(theta)
    return mid, unit(np.cross(mid, normal)), normal


def _leaf_lows(tree: _Metamers, setup: _Setup, radius: np.ndarray, cand: list[tuple[int, int]],
               frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]], size: np.ndarray,
               full: bool) -> np.ndarray:
    """(len(cand),) the lowest y, in metres, of each candidate leaf's triangles as the mesh draws
    them (`mesh.leaf_triangles`): as the full tree draws it (`full`: the full outline and petiole)
    or as this grow draws it (lite: its own outline, no petiole ribbon)."""
    # mesh imports this module (`MESH_VERSION`), so it is imported here, when leaves are placed.
    from model_kit.core.grow.mesh import leaf_triangles

    if not cand:
        return np.zeros(0)
    par = setup.par
    outline = setup.full_outline if full else setup.outline
    of_blade = setup.full_petiole if full else par.petiole_of_blade
    node = np.array([tree.end[i] for i, _ in cand])
    side = np.array([tree.side[i][j] for i, j in cand])
    petiole = radius[[i for i, _ in cand]] * PETIOLE_START_OF_RADIUS + of_blade * size
    mid, across, normal = (np.array([f[n] for f in frames]) for n in range(3))
    tris = leaf_triangles(node, node + side * petiole[:, None], mid, across, normal, size,
                          outline=outline, fold=LEAF_FOLD,
                          blade_centre=np.array(LEAF_BLADE_CENTRE, dtype=float),
                          petiole_tip=LEAF_PETIOLE_TIP_OF_BLADE,
                          petiole_width=LEAF_PETIOLE_HALF_WIDTH_OF_TIP_RADIUS * par.tip_radius,
                          lite=setup.lite and not full)
    return tris[..., 1].reshape(len(cand), -1).min(axis=1)


def _leaves(tree: _Metamers, setup: _Setup, radius: np.ndarray, years: int,
            nodes: list[int]) -> dict[str, list]:
    """The leaves of the nodes in `nodes` (kept metamers, in index order), in metres: for each
    node the year allows, one leaf per bud of the node, on the petiole direction of its rank (the
    direction out of the shoot where the bud points).

    Every leaf draws its size and its blade angle, in order, whether or not lite then thins it, so
    a lite tree's leaves are the full tree's. Lite keeps one leaf in `thin` of each shoot (a
    shoot is one axis' growth in one year), evenly spaced, and sizes the kept ones so the
    shoot's leaf area stays what it was (area is the outline's area times the blade length
    squared; lite has its own, simpler outline). A lite leaf has no petiole.

    The soil is an obstacle to a leaf as to a shoot: a leaf whose blade or petiole would reach
    below it is not placed. A candidate is judged first as the full tree draws it, in full and
    lite alike, so both trees keep leaves on the same shoots. Lite's larger leaves are judged
    again as lite draws them: one that would reach the soil is dropped and its shoot's area goes
    to the shoot's other kept leaves, until none reaches it, so a lite shoot keeps the full
    one's area while any leaf of it stands. Nothing is drawn after the sizes and angles, so the
    soil changes no other leaf's draw."""
    par, rng = setup.par, tree.rng
    lo, hi = par.leaf_size
    cand = [(i, j) for i in nodes if not par.current_year or tree.year[i] == years
            for j in range(par.per_node)]
    size = np.empty(len(cand))
    theta = np.empty(len(cand))
    for k in range(len(cand)):
        size[k] = rng.uniform(lo, hi)
        theta[k] = _blade_angle(rng, *par.beta)
    frames = [_blade_frame(tree.side[i][j], tree.dir[i], float(theta[k]))
              for k, (i, j) in enumerate(cand)]
    stands = _leaf_lows(tree, setup, radius, cand, frames, size, full=True) >= SOIL_Y
    shoots: dict[tuple[int, int], list[int]] = {}
    for k, (i, _) in enumerate(cand):
        if stands[k]:
            shoots.setdefault((tree.axis[i], tree.year[i]), []).append(k)
    keep: list[int] = []
    if par.thin == 1:
        keep = [k for k in range(len(cand)) if stands[k]]
    else:
        ratio = _polygon_area(setup.full_outline) / _polygon_area(setup.outline)
        drawn = size.copy()
        for group in shoots.values():
            n = len(group)
            count = max(1, math.floor(n / par.thin + 0.5))
            kept = [group[int((q + 0.5) * n / count)] for q in range(count)]
            area = float(np.sum(drawn[group] ** 2)) * ratio
            while kept:
                size[kept] = drawn[kept] * math.sqrt(area / float(np.sum(drawn[kept] ** 2)))
                low = _leaf_lows(tree, setup, radius, [cand[k] for k in kept],
                                 [frames[k] for k in kept], size[kept], full=False)
                if (low >= SOIL_Y).all():
                    break
                kept = [k for k, ok in zip(kept, low >= SOIL_Y, strict=True) if ok]
            keep += kept
        keep.sort()
    # Every kept leaf, judged as this grow draws it (a lite tree that thins nothing included).
    if keep and setup.lite:
        low = _leaf_lows(tree, setup, radius, [cand[k] for k in keep], [frames[k] for k in keep],
                         size[keep], full=False)
        keep = [k for k, ok in zip(keep, low >= SOIL_Y, strict=True) if ok]
    out: dict[str, list] = {key: [] for key in ("metamer", "node", "base", "mid", "across",
                                                "normal", "length", "petiole")}
    for k in keep:
        i, j = cand[k]
        mid, across, normal = frames[k]
        petiole = radius[i] * PETIOLE_START_OF_RADIUS + par.petiole_of_blade * size[k]
        out["metamer"].append(i)
        out["node"].append(tree.end[i])
        out["base"].append(tree.end[i] + tree.side[i][j] * petiole)
        out["mid"].append(mid)
        out["across"].append(across)
        out["normal"].append(normal)
        out["length"].append(size[k])
        out["petiole"].append(petiole)
    return out


# ------------------------------------------------------------------------------ the skeleton

def _plain(value):
    """A param value as JSON holds it: tuples become lists."""
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _emit(tree: _Metamers, setup: _Setup, rules: GrowthRules, *, piece: str, height_m: float,
          seed: int, lite: bool, years: int, upm: float) -> Skeleton:
    par, level_names, sides = setup.par, setup.levels, setup.sides
    # Shed metamers drop out; the rest keep their order, so a parent still comes before its
    # children. Radii first: the pipe model counts the shed wood.
    radius = _radii(tree)
    live = np.flatnonzero(np.array(tree.alive, dtype=bool))
    remap = np.full(len(tree), -1, dtype=int)
    remap[live] = np.arange(len(live))
    n = len(live)
    old_parent = np.array(tree.parent, dtype=int)[live]
    parent = np.where(old_parent >= 0, remap[np.maximum(old_parent, 0)], -1)
    starts = [tree.start[i] for i in live]
    ends = [tree.end[i] for i in live]
    direction = np.array([tree.dir[i] for i in live])
    top = len(level_names) - 1
    wood = radius[live]
    turn = np.zeros(n)
    kids = np.flatnonzero(parent >= 0)
    turn[kids] = np.arccos(np.clip((direction[kids] * direction[parent[kids]]).sum(axis=1),
                                   -1.0, 1.0))
    chains = continuation_chains(parent, wood, turn)

    # One limb per chain. A point is a node: the limb's radius there is the pipe below the node,
    # so a limb runs on, past a fork, to a thinner point, and a limb seated there starts at its own
    # radius; a limb's first point carries the radius of its first metamer.
    limb_of = np.zeros(n, dtype=int)
    point_of = np.zeros(n, dtype=int)  # the index of a metamer's end point in its limb
    grown: list[Limb] = []
    for c, chain in enumerate(chains):
        head = int(chain[0])
        pts = [starts[head]] + [ends[int(m)] for m in chain]
        rad = [wood[head]] + [wood[int(m)] for m in chain]
        lift = 0
        if parent[head] < 0:  # the trunk's base sinks into the soil
            pts.insert(0, -UP * TRUNK_SINK_OF_RADIUS * wood[head])
            rad.insert(0, wood[head])
            lift = 1
        for k, m in enumerate(chain):
            limb_of[int(m)], point_of[int(m)] = c, k + 1 + lift
        if parent[head] < 0:
            limb = Limb("trunk", 0, np.array(pts), rad, None, 0.0, tip=CONE, rooted=True)
        else:
            p = int(parent[head])
            host = grown[int(limb_of[p])]
            limb = Limb("chain", host.level + 1, np.array(pts), rad, host,
                        float(host.cum[point_of[p]] / host.length), tip=CONE)
        grown.append(limb)
    chain_limbs = list(grown)
    plan_cum = [limb.cum.copy() for limb in grown]
    made = _stop_short([grown[0], *_roots(par, tree.rng, grown[0]), *grown[1:]], par.clearance)
    if not chain_limbs[0].alive:
        raise ManifestError(f"{piece}: stop-short cut the trunk away ({rules.path})")

    # What stands: a metamer whose end node lies on the kept part of a limb that was made.
    standing = np.zeros(n, dtype=bool)
    for c, limb in enumerate(chain_limbs):
        if limb.alive:
            for m in chains[c]:
                standing[int(m)] = plan_cum[c][point_of[int(m)]] <= limb.keep * (1.0 + KEEP_EPS)
    leaves = _leaves(tree, setup, radius, years, [int(live[m]) for m in np.flatnonzero(standing)])

    raw_height = float(max(limb.out_pts[:, 1].max() for limb in made))
    if not raw_height > 0.0:
        raise ManifestError(f"{rules.path}: the grown tree has no height")
    scale = height_m / raw_height
    # A grow that misses the height its rules cite for the years it grew, by more than the band,
    # is a model failure, not a fit miss: stretching it would hide the failure behind a tree of
    # the right height and wrong shape. It is judged at the grown age, not the asked height,
    # because the age is rounded to whole years.
    cited = height_at_years(setup.ages, years)
    miss = cited / raw_height
    if not SCALE_MIN <= miss <= SCALE_MAX:
        raise ManifestError(
            f"{piece}: after {years} years the grown tree is {raw_height:.4g} m tall against the "
            f"cited {cited:.4g} m (raw height {raw_height:.4g} m, scale {miss:.4g}); the scale "
            f"must lie in {SCALE_MIN} to {SCALE_MAX}, so this is a model failure, not a fit miss "
            f"({rules.path})")
    f = scale * upm
    level = np.array([min(limb.level, top) for limb in made], dtype=int)
    counts = {name: int(np.count_nonzero(level == i)) for i, name in enumerate(level_names)}
    counts["leaves"] = len(leaves["metamer"])
    ids = ["trunk" if limb.rooted else limb.kind if limb.buried
           else f"{level_names[level[k]].lower()}.{k}" for k, limb in enumerate(made)]
    sizes = [len(limb.out_pts) for limb in made]

    # Each leaf sits on the kept limb that carries its metamer, at the node's arc fraction.
    host_of = {int(live[m]): m for m in range(n)}
    leaf_host, leaf_t = [], []
    for i in leaves["metamer"]:
        m = host_of[i]
        limb = chain_limbs[int(limb_of[m])]
        leaf_host.append(limb.index)
        leaf_t.append(float(plan_cum[int(limb_of[m])][point_of[m]]
                            / arc_lengths(limb.out_pts)[-1]))

    def vectors(key):
        return np.array(leaves[key], dtype=float).reshape(-1, 3)

    sk = Skeleton(
        level_names=tuple(level_names),
        limb_id=np.array(ids, dtype=str),
        level=level,
        parent=np.array([limb.parent for limb in made], dtype=int),
        t=np.array([limb.seat_t for limb in made], dtype=float),
        tip=np.array([limb.tip for limb in made], dtype=int),
        rooted=np.array([limb.rooted for limb in made], dtype=bool),
        buried=np.array([limb.buried for limb in made], dtype=bool),
        start=np.concatenate([[0], np.cumsum(sizes)]).astype(int),
        pts=np.concatenate([limb.out_pts for limb in made]) * f,
        radii=np.concatenate([limb.out_radii for limb in made]) * f,
        host=np.array(leaf_host, dtype=int),
        leaf_t=np.array(leaf_t, dtype=float),
        node=vectors("node") * f, base=vectors("base") * f, mid=vectors("mid"),
        across=vectors("across"), normal=vectors("normal"),
        length=np.array(leaves["length"], dtype=float) * f,
        petiole=np.array(leaves["petiole"], dtype=float) * f,
        shade=np.zeros(len(leaf_host), dtype=int),
        shades=LEAF_SHADES, sides=sides, outline=setup.outline,
        fold=LEAF_FOLD, blade_centre=np.array(LEAF_BLADE_CENTRE, dtype=float),
        petiole_width=LEAF_PETIOLE_HALF_WIDTH_OF_TIP_RADIUS * par.tip_radius * f,
        petiole_tip=LEAF_PETIOLE_TIP_OF_BLADE,
        lite=lite,
        info={
            "piece": piece, "rules": rules.path.name, "seed": seed, "height_m": float(height_m),
            "H": float(height_m * upm), "lite": lite, "years": years,
            "rules_sha256": rules.sha256, "kit_sha256": rules.kit_sha256,
            "fitted": {k: _plain(p.value) for k, p in rules.params.items() if p.tag == "fitted"},
            "mesh": setup.mesh, "counts": counts, "raw_height_m": raw_height,
            "scale": float(scale), "metamers": n, "cut_metamers": int(n - standing.sum()),
        },
    )
    check_skeleton(sk, f"{piece} (grown)")
    return sk

