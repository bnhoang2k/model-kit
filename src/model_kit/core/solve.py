"""The clip solver: turn a chain's joints so its effector reaches a target.

`solve_chain` runs a plain Nelder-Mead search over the chain's `dofs`. The cost has no units:
the miss is measured in chain reaches (`rig.reach`) and every other term is in radians, or in
a gap rescaled to the reference rig (see `Cost terms`). So a rig built ten times larger, or
with a bigger effector, solves to the same angles. The weights live in `tolerances.py`.

`solve_clip` solves every target of a clip, key by key, each chain starting from its own last
solve, and returns the results keyed by (frame, chain). Where a chain's straight blend from
one key to the next sweeps its skin through a solid, it then solves an extra mid-key there.

`mirror_pose` and `other_side` swap left and right by the manifest's mirror pairs and axis.

Cost terms. The main term is `(miss / L)^2`, with `L` the chain's own reach. The avoid, plane
and solid terms are gaps (lengths), so each gap enters as `(gap * SOLVER_REFERENCE_REACH / L)^2`,
a gap in reference-rig units; the weight then keeps the replaced solver's value. Every length
the solver takes from a factor in `tolerances.py` is that factor times `L` or times the chain's
`effector_radius`, so it scales with the rig the same way.

A bone matrix used for a shape (an avoid shape, a solid, a chunk) is taken as rigid: the
shape's frame has world lengths.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise

import numpy as np

from model_kit.core import tolerances as tol
from model_kit.core.errors import ManifestError
from model_kit.core.geom import Nearest, SurfaceIndex
from model_kit.core.manifest import Avoid, Chain, Clip, MirrorSpec, RigSpec, Target
from model_kit.core.rig import Pose, RigData, pose_world, reach

AXES = "xyz"
_REST = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

# The reference solver's search schedule: a coarse run, then a fine run from where it stopped.
STEPS = (0.25, 0.06)
# The second phase (with solids) searches from the first phase's best, at these steps.
# Source: the reference solver's Nelder-Mead runs, at step 0.12 then 0.03.
SOLID_STEPS = (0.12, 0.03)
# With `level`, extra starts turn the effector by each of these on each of its axes.
LEVEL_TURNS = (-np.pi / 2, np.pi / 2, np.pi)


@dataclass(frozen=True, eq=False)
class SolveResult:
    """One solve. `pose` is the input pose with the varied angles set; from `solve_clip` it
    may also carry other chains' bones solved at the same key, so read only `chain.bones`."""

    pose: Pose
    err: float  # distance left from the measured point to the target, in world units
    point: np.ndarray  # (3,) the measured point: the tool tip, else the effector centre
    # From `solve_clip`: how far the chain's skin goes past a solid's surface on the straight
    # blend from this key to the chain's next key, when that sweep is left unfixed. 0.0: none.
    sweep_to_next: float = 0.0


def nelder(f: Callable[[list[float]], float], x0: Sequence[float], step: float,
           iters: int = 900, tol: float = 1e-11) -> tuple[list[float], float]:
    """A plain Nelder-Mead minimiser, as in the replaced solver: an axis simplex of `step`,
    reflect by 1, expand by 2, contract by 1/2, and shrink toward the best point.

    It stops after `iters` rounds, or when the best and worst values differ by less than `tol`.
    Returns the best point and its value.
    """
    n = len(x0)
    pts = [list(map(float, x0))] + [[float(x0[j]) + (step if j == i else 0.0) for j in range(n)]
                                     for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        if vals[-1] - vals[0] < tol:
            break
        cen = [sum(p[j] for p in pts[:-1]) / n for j in range(n)]
        xr = [2 * cen[j] - pts[-1][j] for j in range(n)]
        fr = f(xr)
        if fr < vals[0]:
            xe = [3 * cen[j] - 2 * pts[-1][j] for j in range(n)]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [(cen[j] + pts[-1][j]) / 2 for j in range(n)]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = [(pts[0][j] + pts[i][j]) / 2 for j in range(n)]
                    vals[i] = f(pts[i])
    best = min(range(n + 1), key=lambda i: vals[i])
    return pts[best], vals[best]


# --- what a key asks of a chain --------------------------------------------------------------


def _vec3(value, what: str) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ManifestError(f"{what} is 3 finite numbers, got {v.tolist()}")
    return v


@dataclass(frozen=True, eq=False)
class Tool:
    """A tool in the fist. `tip` is its tip in the effector bone's rest frame (the slide is
    already in it); the solver measures the miss at the tip. `slide` is how far the tool is
    slid along itself through the fist, kept so a check and the solver read one record."""

    tip: np.ndarray
    slide: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tip", _vec3(self.tip, "a tool's tip"))
        if not np.isfinite(self.slide):
            raise ManifestError(f"a tool's slide must be finite, got {self.slide!r}")


@dataclass(frozen=True, eq=False)
class Plane:
    """A half-space: every skin point of the effector keeps `margin` in front of it, on the
    side `normal` points to (`normal` need not be a unit vector). `point` is on the plane."""

    point: np.ndarray
    normal: np.ndarray
    margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", _vec3(self.point, "a plane's point"))
        normal = _vec3(self.normal, "a plane's normal")
        size = float(np.linalg.norm(normal))
        if size == 0.0:
            raise ManifestError("a plane's normal is not all 0")
        object.__setattr__(self, "normal", normal / size)
        if not np.isfinite(self.margin):
            raise ManifestError(f"a plane's margin must be finite, got {self.margin!r}")


@dataclass(frozen=True, eq=False)
class ChainArgs:
    """What one key asks of one chain.

    - `level`: a 3x3 world rotation the effector turns toward, or None. `level_weight` weighs
      that turn against the reach.
    - `tool`: the tool in the fist, or None. With a tool the miss is measured at its tip.
    - `plane`: a plane the effector's skin keeps in front of, or None. Needs `skin`.
    - `avoid_clearance`: how far the elbow, wrist and effector centre keep out of the avoid
      shapes, as a length. 0 turns the term off; a negative value lets them overlap that much.
    - `keep_last`: `solve_clip` reuses the chain's last solve instead of solving.

    The caller builds this from its own key syntax; the kit knows no key syntax.
    """

    level: np.ndarray | None = None
    level_weight: float = tol.LEVEL_WEIGHT
    tool: Tool | None = None
    plane: Plane | None = None
    avoid_clearance: float = 0.0
    keep_last: bool = False

    def __post_init__(self) -> None:
        if self.level is not None:
            level = np.asarray(self.level, dtype=float)
            if level.shape != (3, 3) or not np.all(np.isfinite(level)):
                raise ManifestError("a level is a finite 3x3 rotation, got shape "
                                    f"{level.shape}")
            object.__setattr__(self, "level", level)
        if not np.isfinite(self.level_weight) or not np.isfinite(self.avoid_clearance):
            raise ManifestError("a key's level weight and avoid clearance must be finite")


def _same_args(a: ChainArgs, b: ChainArgs) -> bool:
    def arr(u, v) -> bool:
        return (u is None and v is None) or (u is not None and v is not None
                                             and np.array_equal(u, v))

    if (a.tool is None) != (b.tool is None) or (a.plane is None) != (b.plane is None):
        return False
    if a.tool is not None and not (arr(a.tool.tip, b.tool.tip) and a.tool.slide == b.tool.slide):
        return False
    if a.plane is not None and not (arr(a.plane.point, b.plane.point)
                                    and arr(a.plane.normal, b.plane.normal)
                                    and a.plane.margin == b.plane.margin):
        return False
    return (arr(a.level, b.level) and a.level_weight == b.level_weight
            and a.avoid_clearance == b.avoid_clearance and a.keep_last == b.keep_last)


# An avoid shape is the manifest's `Avoid`: a box or an ellipsoid in a bone's rest frame.
AvoidShape = Avoid


@dataclass(frozen=True, eq=False)
class SkinPoints:
    """Linear-blend skin points: P points, each following K bones.

    `bones` (P, K) names the bones, `weights` (P, K) are the blend weights (each row sums to 1;
    pad a row with weight 0), and `local` (P, K, 3) is the point in each bone's rest frame. A
    wrist cuff is half forearm and half hand. List the points furthest first, so a prefix of
    them still covers the part.
    """

    bones: np.ndarray
    weights: np.ndarray
    local: np.ndarray
    _names: tuple[str, ...] = field(init=False, repr=False)
    _index: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bones = np.asarray(self.bones, dtype=str)
        weights = np.asarray(self.weights, dtype=float)
        local = np.asarray(self.local, dtype=float)
        if bones.ndim != 2 or weights.shape != bones.shape or local.shape != (*bones.shape, 3):
            raise ManifestError(f"skin points: bones, weights and local must share shape (P, K) "
                                f"and (P, K, 3), got shape {bones.shape}, {weights.shape} and "
                                f"{local.shape}")
        if bones.size == 0 or not np.all(np.isfinite(weights)) or not np.all(np.isfinite(local)):
            raise ManifestError("skin points: need at least one point, with finite weights "
                                "and positions")
        if np.any(weights < 0.0) or not np.allclose(weights.sum(axis=1), 1.0,
                                                    atol=tol.SKIN_WEIGHT_SLACK):
            raise ManifestError("skin points: each point's weights are not negative and sum "
                                "to 1")
        names = tuple(sorted(set(bones.ravel().tolist())))
        object.__setattr__(self, "bones", bones)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "local", local)
        object.__setattr__(self, "_names", names)
        object.__setattr__(self, "_index", np.searchsorted(names, bones))

    def __len__(self) -> int:
        return len(self.bones)

    def head(self, n: int) -> SkinPoints:
        """The first `n` points."""
        return SkinPoints(self.bones[:n], self.weights[:n], self.local[:n])

    def posed(self, world: Mapping[str, np.ndarray]) -> np.ndarray:
        """The (P, 3) world positions under posed bone matrices `world`."""
        missing = [b for b in self._names if b not in world]
        if missing:
            raise ManifestError(f"skin points follow bones with no posed matrix: {missing}")
        mats = np.stack([np.asarray(world[b], dtype=float) for b in self._names])[self._index]
        pts = np.einsum("pkij,pkj->pki", mats[..., :3, :3], self.local) + mats[..., :3, 3]
        return np.einsum("pk,pki->pi", self.weights, pts)


@dataclass(frozen=True, eq=False)
class Chunk:
    """One arm part's surface: a closed mesh in `bone`'s rest frame. A solid's back points
    inside it cost `W_SOLID_BACK`."""

    bone: str
    index: SurfaceIndex


@dataclass(frozen=True, eq=False)
class Solid:
    """A mesh the arm keeps out of, as the reference solver's solid cost has it.

    - `index`: the mesh, closed and wound outward, in the solid's own frame. The frame rides
      `bone` (None: fixed in the world) through `offset` (a 4x4, identity by default).
    - `margins`: per skin chunk (a bone name, the same keys as `skin`), how far its skin keeps
      from the surface. A chunk with no key is not tested against this solid: a body solid
      has no `arm` key, so the shoulder may sit inside it. A press zone sets the hand's margin
      to `-PRESS_MARGIN_SHARE x budget`.
    - `held_by`: the contact group that holds this solid. A chain whose `effector_group` it
      names skips the solid at its effector.
    - `skip`: per side, keyed by contact group (a chain's `effector_group`): balls
      `(centre, r)` in the solid's frame. An effector skin point (or back point) inside one
      of its own group's balls skips the solid. A grip zone is one, so it frees only the
      fist that grips.
    - `back_points`: (Q, 3) points of the solid in its frame. Any inside an arm chunk costs
      `W_SOLID_BACK`, so a thin screw or a cymbal edge is kept out of the arm.
    """

    name: str
    index: SurfaceIndex
    margins: Mapping[str, float]
    held_by: str | None = None
    skip: Mapping[str, tuple[tuple[np.ndarray, float], ...]] = field(default_factory=dict)
    back_points: np.ndarray | None = None
    bone: str | None = None
    offset: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.back_points is not None:
            pts = np.asarray(self.back_points, dtype=float)
            if pts.ndim != 2 or pts.shape[1] != 3 or not np.all(np.isfinite(pts)):
                raise ManifestError(f"solid {self.name!r}: back points are (Q, 3) finite "
                                    f"numbers, got shape {pts.shape}")
            object.__setattr__(self, "back_points", pts)
        if self.offset is not None:
            m = np.asarray(self.offset, dtype=float)
            if m.shape != (4, 4) or not np.all(np.isfinite(m)):
                raise ManifestError(f"solid {self.name!r}: offset is a finite 4x4")
            object.__setattr__(self, "offset", m)
        if not all(np.isfinite(v) for v in self.margins.values()):
            raise ManifestError(f"solid {self.name!r}: margins must be finite")
        if not isinstance(self.skip, Mapping):
            raise ManifestError(f"solid {self.name!r}: skip is a mapping from contact group to "
                                "balls, not a flat list")
        object.__setattr__(self, "skip", {
            group: tuple((_vec3(c, f"solid {self.name!r}: a skip centre"), float(r))
                         for c, r in balls)
            for group, balls in self.skip.items()})


# --- the chain problem ----------------------------------------------------------------------


def _cut_rig(rig: RigData, chain: Chain, extra: Sequence[str] = ()) -> RigData:
    """The rig cut down to the effector bone, the `extra` bones and their ancestors: all the
    matrices the cost reads."""
    if chain.effector not in rig.parent:
        raise ManifestError(f"chain effector {chain.effector!r} is not in the rig")
    keep: set[str] = set()
    for start in (chain.effector, *extra):
        if start not in rig.parent:
            raise ManifestError(f"bone {start!r} is not in the rig")
        bone: str | None = start
        while bone is not None:
            keep.add(bone)
            bone = rig.parent.get(bone)
    if chain.effector not in rig.effector_mid:
        raise ManifestError(f"rig has no effector_mid for effector bone {chain.effector!r}")
    return RigData(order=tuple(b for b in rig.order if b in keep),
                   parent={b: rig.parent[b] for b in keep},
                   rest={b: rig.rest[b] for b in keep if b in rig.rest},
                   world=rig.world, effector_mid={chain.effector: rig.effector_mid[chain.effector]})


def _check_rigid(path: RigData, carrying: set[str]) -> None:
    """Shapes, solids and chunks are measured in a bone's frame and read as world lengths, so
    `rig.world` and the rest matrix of each carrying bone and of every ancestor that composes
    its posed matrix must be rotations (no scale, no shear): a turned, stretched ancestor
    shears everything under it. `path` is the cut rig, which holds those ancestors."""
    def rigid(m) -> bool:
        r = np.asarray(m, dtype=float)[:3, :3]
        return bool(np.abs(r @ r.T - np.eye(3)).max() <= tol.BONE_RIGID_SLACK)

    if not rigid(path.world):
        raise ManifestError("avoid shapes, solids and chunks need a rigid rig.world (a "
                            "rotation, no scale or shear)")
    for bone in sorted(carrying):
        up: str | None = bone
        while up is not None:
            if not rigid(path.rest[up]):
                raise ManifestError(
                    f"bone {up!r} composes the matrix of {bone!r}, which carries an avoid "
                    "shape, solid or chunk, so its rest matrix must be rigid (no scale or shear)")
            up = path.parent[up]


def _keys(chain: Chain, turn_effector: bool) -> list[tuple[str, int]]:
    """The varied angles as (bone, axis index), bone by bone along the chain, X before Y before
    Z. The effector's own axes are in only when `turn_effector`."""
    keys = []
    for bone in chain.bones:
        if bone == chain.effector and not turn_effector:
            continue
        keys += [(bone, i) for i, axis in enumerate(AXES) if axis in chain.dofs[bone]]
    return keys


def _apply(pose: Pose, keys: list[tuple[str, int]], x: Sequence[float]) -> Pose:
    out = dict(pose)
    for (bone, i), value in zip(keys, x, strict=True):
        euler, loc = out.get(bone, _REST)
        euler = np.array(euler, dtype=float)
        euler[i] = value
        out[bone] = (euler, np.asarray(loc, dtype=float))
    return out


def _start(source: Pose, keys: list[tuple[str, int]]) -> list[float]:
    return [float(np.asarray(source.get(bone, _REST)[0], dtype=float)[i]) for bone, i in keys]


# A cost term: (angle vector, posed bone matrices, effector centre) -> a scale-free cost.
Term = Callable[[np.ndarray, dict[str, np.ndarray], np.ndarray], float]


@dataclass(frozen=True)
class Phase:
    """One stage of the search: the Nelder-Mead steps, and whether the solid terms count.

    The first phase starts from x0 (and the level starts); a later phase starts from the best
    so far and from x0. A phase with solids runs only when a solid term exists."""

    steps: tuple[float, ...]
    solids: bool


PHASES = (Phase(STEPS, False), Phase(SOLID_STEPS, True))


def _clearance_term(chain: Chain, shapes: Sequence[Avoid], clearance: float, length: float
                    ) -> Term:
    """Keep the elbow, wrist and effector centre `clearance` out of each shape.

    Source: the reference solver's clearance and avoid term. The points are the
    heads of the chain's bones between the first and the effector (radius
    `AVOID_ELBOW_OF_EFFECTOR`), the effector bone's head (`AVOID_WRIST_OF_EFFECTOR`) and the
    effector centre (`AVOID_CENTRE_OF_EFFECTOR`), each times the effector radius.
    """
    radius = chain.effector_radius
    elbows = [b for b in chain.bones[1:] if b != chain.effector]
    head_scale = tol.AVOID_HEAD_SCALE_OF_REACH * length
    unit = tol.SOLVER_REFERENCE_REACH / length
    if not shapes:
        raise ManifestError("a key asks for avoid clearance but the rig has no avoid shapes")
    parts = [(shape.bone, shape.kind, np.asarray(shape.centre, dtype=float),
              np.asarray(shape.half, dtype=float)) for shape in shapes]

    def term(x, world, mid):
        points = [(world[b][:3, 3], tol.AVOID_ELBOW_OF_EFFECTOR * radius) for b in elbows]
        points.append((world[chain.effector][:3, 3], tol.AVOID_WRIST_OF_EFFECTOR * radius))
        points.append((mid, tol.AVOID_CENTRE_OF_EFFECTOR * radius))
        worst = np.inf
        for bone, kind, centre, half in parts:
            inv = np.linalg.inv(world[bone])
            for p, r in points:
                q = inv[:3, :3] @ p + inv[:3, 3]
                if kind == "box":
                    d = np.abs(q - centre) - half
                    outside = float(np.linalg.norm(np.maximum(d, 0.0)))
                    sd = outside if np.any(d > 0.0) else float(d.max())
                else:  # ellipsoid: its distance in radii, times the head scale, is a length
                    k = float(np.linalg.norm((q - centre) / half))
                    sd = (k - 1.0) * head_scale
                worst = min(worst, sd - r)
        short = max(0.0, clearance - worst) * unit
        return tol.W_AVOID * short * short

    return term


def _terms(chain: Chain, keys: list[tuple[str, int]], x0: np.ndarray, target: np.ndarray,
           point: np.ndarray, length: float, args: ChainArgs, shapes: Sequence[Avoid],
           skin: Mapping[str, SkinPoints]) -> list[Term]:
    """The cost terms that always count. Each is `(x, world, mid)`, with `mid` the world
    position of the effector centre."""

    def miss(x, world, mid):
        m = world[chain.effector]
        d = m[:3, :3] @ point + m[:3, 3] - target
        return float(d @ d) / length**2

    def stay(x, world, mid):
        d = x - x0
        return tol.W_STAY * float(d @ d)

    terms: list[Term] = [miss, stay]
    if args.level is not None:
        w_level = args.level_weight * tol.W_LEVEL_PER_WEIGHT

        def level_term(x, world, mid):
            d = world[chain.effector][:3, :3] - args.level
            return w_level * float(np.sum(d * d))

        terms.append(level_term)
    index = {key: k for k, key in enumerate(keys)}
    limited = [(index[(bone, AXES.index(axis))], lo, hi)
               for name, (lo, hi) in chain.limits.items()
               for bone, _, axis in [name.rpartition(":")]
               if (bone, AXES.index(axis)) in index]
    if limited:
        at = np.array([k for k, _, _ in limited])
        lo = np.array([v for _, v, _ in limited])
        hi = np.array([v for _, _, v in limited])

        def limit(x, world, mid):
            a = x[at]
            over = np.maximum(0.0, a - hi) ** 2 + np.maximum(0.0, lo - a) ** 2
            return tol.W_LIMIT * float(over.sum())

        terms.append(limit)
    if args.avoid_clearance != 0.0:
        terms.append(_clearance_term(chain, shapes, args.avoid_clearance, length))
    if args.plane is not None:
        terms.append(_plane_term(chain, args.plane, skin, length))
    return terms


def _plane_term(chain: Chain, plane: Plane, skin: Mapping[str, SkinPoints], length: float
                ) -> Term:
    """Keep every skin point of the effector in front of the plane.

    Source: the reference solver's plane term, `4 * max(0, m - n . (w - p0))^2` over the hand's 90 points
    (`SKIN_SAMPLES_PER_CHUNK`)."""
    if chain.effector not in skin:
        raise ManifestError(f"a plane needs skin points for the effector {chain.effector!r}; "
                            f"the skin holds {sorted(skin)}")
    pts = skin[chain.effector].head(tol.SKIN_SAMPLES_PER_CHUNK)
    unit = tol.SOLVER_REFERENCE_REACH / length

    def term(x, world, mid):
        w = pts.posed(world)
        short = np.maximum(0.0, plane.margin - (w - plane.point) @ plane.normal) * unit
        return tol.W_PLANE * float(short @ short)

    return term


def _solid_points(chain: Chain, skin: Mapping[str, SkinPoints]) -> dict[str, SkinPoints]:
    """The skin points each chain bone offers the solid phase: the first N of its chunk. N is
    `SOLID_SKIN_POINTS_HAND` at the effector, `_FOREARM` one bone back and `_ARM` further."""
    counts = (tol.SOLID_SKIN_POINTS_HAND, tol.SOLID_SKIN_POINTS_FOREARM,
              tol.SOLID_SKIN_POINTS_ARM)
    out = {}
    for bone, pts in skin.items():
        if bone not in chain.bones:
            raise ManifestError(f"skin names bone {bone!r}, which is not on the chain "
                                f"{list(chain.bones)}")
        back = len(chain.bones) - 1 - chain.bones.index(bone)
        out[bone] = pts.head(counts[min(back, len(counts) - 1)])
    return out


@dataclass
class _Prepared:
    """A solid with what the cost reads about it, built once per solve."""

    solid: Solid
    lo: np.ndarray
    hi: np.ndarray
    skin_parts: list[str]  # skin chunks tested against it
    back_parts: list[Chunk]  # arm chunks its back points are tested against

    def matrix(self, world: dict[str, np.ndarray]) -> np.ndarray:
        base = np.eye(4) if self.solid.bone is None else world[self.solid.bone]
        return base if self.solid.offset is None else base @ self.solid.offset


@dataclass(frozen=True, eq=False)
class _SolidGauge:
    """What the solid phase reads, shared by the cost and the in-between check.

    - `term`: the solid phase's cost (`W_SOLID`, `W_SOLID_BACK`).
    - `depth(world)`: the deepest any tested point sits inside a solid, in world units: past
      the surface itself, so a positive margin is a buffer and not part of the solid. 0.0 when
      every point is outside the solids or in a margin.
    - `exits(world, travel)`: the world vectors that carry the deepest skin point out to its
      keep-out surface, one per way out: along a solid's own axis either way, except the axis
      `travel` (a world direction, or None) most follows. Out the way the point came in would
      put a key behind the solid and leave the sweep to the next one. Empty when no skin point
      is past a solid, and when no way out clears the solid's keep-out surface (a way that never
      clears is left out, so the caller aims at the pose it has).
    - `moving(world)`: the skin points tested against a solid, and the corners of each solid's
      box that rides a bone. Their motion between two poses bounds how far a point can travel
      past a solid without being tested.
    - `thinnest`: the smallest side of the box round any tested solid, in world units.
    """

    term: Term
    depth: Callable[[dict[str, np.ndarray]], float]
    exits: Callable[[dict[str, np.ndarray], np.ndarray | None], list[np.ndarray]]
    moving: Callable[[dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]]
    thinnest: float


def _solid_gauge(chain: Chain, solids: Sequence[Solid], skin: Mapping[str, SkinPoints],
                 chunks: Sequence[Chunk], length: float) -> _SolidGauge | None:
    """The solid phase's cost and what the in-between check reads (see `_SolidGauge`): keep the
    arm's skin out of the solids and the solids' back points out of the arm.

    Source: the reference solver's solid cost. Returns None when no (solid, chunk) pair is tested,
    so a chain no solid reaches runs no second phase.
    """
    pts = _solid_points(chain, skin)
    held = chain.effector
    prepared: list[_Prepared] = []
    for solid in solids:
        skips_effector = solid.held_by is not None and solid.held_by == chain.effector_group
        skin_parts = [b for b in pts if b in solid.margins
                      and not (skips_effector and b == held)]
        back_parts = [c for c in chunks if solid.back_points is not None
                      and c.bone in solid.margins and not (skips_effector and c.bone == held)]
        if not skin_parts and not back_parts:
            continue
        corners = solid.index.tris.reshape(-1, 3)
        prepared.append(_Prepared(solid, corners.min(axis=0), corners.max(axis=0), skin_parts,
                                  back_parts))
    if not prepared:
        return None
    unit = tol.SOLVER_REFERENCE_REACH / length
    prefilter = tol.SOLID_PREFILTER_OF_REACH * length
    back_prefilter = tol.BACK_PREFILTER_OF_EFFECTOR * chain.effector_radius
    boxes = {c.bone: (c.index.tris.reshape(-1, 3).min(axis=0),
                      c.index.tris.reshape(-1, 3).max(axis=0)) for c in chunks}

    def past(near, q: np.ndarray, margin: float) -> np.ndarray:
        """How far each point (in the solid's frame) is past the keep-out surface, given its
        nearest surface point `near`; 0 if clear."""
        inside = np.einsum("ij,ij->i", q - near.point, near.normal) < 0.0
        return np.maximum(np.where(inside, near.dist + margin, margin - near.dist), 0.0)

    def penetration(solid: Solid, q: np.ndarray, margin: float) -> np.ndarray:
        """How far each point (in the solid's frame) is past the keep-out surface; 0 if clear."""
        return past(solid.index.closest(q), q, margin)

    def gaps(world) -> list[tuple[float, np.ndarray, float, tuple | None]]:
        """Each tested pair's penetration, in reference-rig units, with the weight it costs, the
        buffer of its margin (0 for a negative one) and, for a skin pair, what `exits` needs:
        the solid, its matrix, the margin, and the kept points in the solid's frame (None for a
        back pair)."""
        out: list[tuple[float, np.ndarray, float, tuple | None]] = []
        posed: dict[str, np.ndarray] = {}
        for prep in prepared:
            solid = prep.solid
            m = prep.matrix(world)
            inv = np.linalg.inv(m)
            tested: list[tuple[float, np.ndarray]] = []  # (margin, kept points) per skin part
            for part in prep.skin_parts:
                if part not in posed:
                    posed[part] = pts[part].posed(world)
                q = posed[part] @ inv[:3, :3].T + inv[:3, 3]
                keep = np.all((q > prep.lo - prefilter) & (q < prep.hi + prefilter), axis=1)
                if part == held:
                    for centre, r in solid.skip.get(chain.effector_group, ()):
                        keep &= np.linalg.norm(q - centre, axis=1) >= r
                if not keep.any():
                    continue
                tested.append((solid.margins[part], q[keep]))
            if tested:
                # One nearest-surface query for all of this solid's skin parts: a point's answer
                # does not depend on the others, and a query's cost is mostly per call.
                every = np.concatenate([kept for _, kept in tested])
                near = solid.index.closest(every)
                at = 0
                for margin, kept in tested:
                    rows = slice(at, at + len(kept))
                    at += len(kept)
                    pen = past(Nearest(*(a[rows] for a in near)), kept, margin)
                    out.append((tol.W_SOLID, pen * unit, max(margin, 0.0), (solid, m, margin, kept)))
            for chunk in prep.back_parts:
                c = np.linalg.inv(world[chunk.bone]) @ m
                gp = solid.back_points
                q = gp @ c[:3, :3].T + c[:3, 3]
                lo, hi = boxes[chunk.bone]
                keep = np.all((q > lo - back_prefilter) & (q < hi + back_prefilter), axis=1)
                if chunk.bone == held:
                    for centre, r in solid.skip.get(chain.effector_group, ()):
                        keep &= np.linalg.norm(gp - centre, axis=1) >= r
                if not keep.any():
                    continue
                near = chunk.index.closest(q[keep])
                inside = np.einsum("ij,ij->i", q[keep] - near.point, near.normal) < 0.0
                # A press zone's margin is below zero: the point may sit that deep in the arm.
                short = np.maximum(near.dist + solid.margins[chunk.bone], 0.0)
                out.append((tol.W_SOLID_BACK, short[inside] * unit,
                            max(solid.margins[chunk.bone], 0.0), None))
        return out

    def term(x, world, mid):
        total = 0.0
        for weight, gap, _, _ in gaps(world):
            total += weight * float(gap @ gap)
        return total

    def depth(world) -> float:
        return max([0.0] + [float(g.max()) / unit - buffer
                            for _, g, buffer, _ in gaps(world) if g.size])

    def exit_distance(solid: Solid, margin: float, q: np.ndarray, d: np.ndarray) -> float | None:
        """How far from `q` along `d` (both in the solid's frame) the keep-out surface lies:
        the first clear point on a march of half the solid's thinnest side, found to
        `COMPARE_TOL`. A start in the margin on one side must cross the box and the margin on
        the other, so the march is the box diagonal plus two margins. None when no point on
        the march is clear: that way is no way out."""
        corners = solid.index.tris.reshape(-1, 3)
        step = float((corners.max(axis=0) - corners.min(axis=0)).min()) / 2
        far = float(np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))) + 2 * abs(margin)
        marks = np.arange(0.0, far + step, step)
        clear = np.flatnonzero(penetration(solid, q + marks[:, None] * d, margin) <= 0.0)
        if clear.size == 0:
            return None
        if clear[0] == 0:
            return 0.0
        lo, hi = marks[clear[0] - 1], marks[clear[0]]
        while hi - lo > tol.COMPARE_TOL:
            mid = (lo + hi) / 2
            if penetration(solid, (q + mid * d)[None], margin)[0] <= 0.0:
                hi = mid
            else:
                lo = mid
        return hi

    def exits(world, travel) -> list[np.ndarray]:
        best, ctx, k = 0.0, None, 0
        for _, gap, _, c in gaps(world):
            if c is not None and gap.size and float(gap.max()) > best:
                best, ctx, k = float(gap.max()), c, int(gap.argmax())
        if ctx is None:
            return []
        solid, m, margin, q = ctx
        skip = -1  # the solid axis travel follows most
        if travel is not None and float(np.linalg.norm(travel)) > 0.0:
            skip = int(np.argmax(np.abs(m[:3, :3].T @ travel)))
        axes = [d * e for i, e in enumerate(np.eye(3)) if i != skip for d in (1.0, -1.0)]
        found = [(exit_distance(solid, margin, q[k], d), d) for d in axes]
        return [m[:3, :3] @ (dist * d) for dist, d in found if dist is not None]

    parts = sorted({p for prep in prepared for p in prep.skin_parts})
    riding = [prep for prep in prepared if prep.solid.bone is not None]

    def moving(world) -> tuple[np.ndarray, np.ndarray]:
        skin_pts = (np.concatenate([pts[p].posed(world) for p in parts]) if parts
                    else np.zeros((0, 3)))
        corners = []
        for prep in riding:
            m = prep.matrix(world)
            box = np.array([[x, y, z] for x in (prep.lo[0], prep.hi[0])
                            for y in (prep.lo[1], prep.hi[1]) for z in (prep.lo[2], prep.hi[2])])
            corners.append(box @ m[:3, :3].T + m[:3, 3])
        return skin_pts, (np.concatenate(corners) if corners else np.zeros((0, 3)))

    thinnest = min(float((prep.hi - prep.lo).min()) for prep in prepared)
    return _SolidGauge(term, depth, exits, moving, thinnest)


def _solid_term(chain: Chain, solids: Sequence[Solid], skin: Mapping[str, SkinPoints],
                chunks: Sequence[Chunk], length: float) -> Term | None:
    """The solid phase's cost term (see `_solid_gauge`), or None when no pair is tested."""
    gauge = _solid_gauge(chain, solids, skin, chunks, length)
    return None if gauge is None else gauge.term


def _cut_for_shapes(rig: RigData, chain: Chain, avoid: Sequence[Avoid],
                    solids: Sequence[Solid], skin: Mapping[str, SkinPoints],
                    chunks: Sequence[Chunk]) -> RigData:
    """The cut rig that also holds every bone an avoid shape, solid, skin point or chunk
    follows. Those bones must be rigid."""
    needed = {s.bone for s in avoid}
    needed |= {s.bone for s in solids if s.bone is not None}
    needed |= {c.bone for c in chunks}
    for pts in skin.values():
        needed |= set(pts._names)
    path = _cut_rig(rig, chain, sorted(needed))
    if avoid or solids or chunks:
        _check_rigid(path, {s.bone for s in avoid} | {s.bone for s in solids if s.bone}
                     | {c.bone for c in chunks})
    return path


def solve_chain(rig: RigData, pose: Pose, chain: Chain, target, *, args: ChainArgs,
                x0: Pose | None = None, quick: bool = False,
                avoid: Sequence[Avoid] = (), solids: Sequence[Solid] = (),
                skin: Mapping[str, SkinPoints] | None = None,
                chunks: Sequence[Chunk] = ()) -> SolveResult:
    """Turn the chain's joints so its measured point reaches `target` (world, glTF axes).

    - The varied angles are the chain's `dofs`. The effector bone's own dofs vary only when
      `args.level` (a 3x3 world rotation for the effector) or `args.plane` is given;
      `args.level_weight` weighs the level turn against the reach.
    - The measured point is the effector bone's `effector_mid`, or with `args.tool` the tool's
      tip. The reach check allows the tool's `tip_length` past the hand for such a target.
    - `x0` is the last key: the search starts from its angles and the stay term pulls toward
      them. It defaults to `pose`. Every bone and value the search does not vary comes from
      `pose`.
    - The cost: `(miss / L)^2 + W_STAY |x - x0|^2 + W_LEVEL |R_eff - level|^2 + W_LIMIT x
      (angle past chain.limits)^2`, with `L` the chain's reach (`rig.reach`, world units), plus
      the terms below, each a gap `g` entering as `W (g R_ref / L)^2`. The effector radius sets
      no weight: the angles depend on the chain's shape only.
    - `avoid` shapes (with `args.avoid_clearance`): the elbow, wrist and effector centre keep
      that far out of each (`W_AVOID`).
    - `args.plane` with `skin`: every skin point of the effector stays in front (`W_PLANE`).
    - `solids` with `skin` (a `SkinPoints` per chain bone) and `chunks`: the arm's skin keeps
      each solid's margin out, and a solid's back points stay out of the arm's chunks
      (`W_SOLID`, `W_SOLID_BACK`). They count in the second phase only.
    - The schedule (`PHASES`): phase 1, without solids, runs from each start at `STEPS`. With
      a level and not `quick`, nine more starts turn the effector by -pi/2, pi/2 and pi on each
      of its axes. Phase 2, with solids, runs at `SOLID_STEPS` from phase 1's best and from x0.

    Returns the solved pose (the input pose with the varied angles set), the distance left and
    the measured point.
    """
    target = _vec3(target, "a solve target")
    skin = {} if skin is None else dict(skin)
    turn_effector = args.level is not None or args.plane is not None
    keys = _keys(chain, turn_effector=turn_effector)
    if not keys:
        raise ManifestError(f"chain ending {chain.effector!r} has no angle to vary")
    path = _cut_for_shapes(rig, chain, avoid, solids, skin, chunks)
    start = _start(pose if x0 is None else x0, keys)
    mid = np.asarray(path.effector_mid[chain.effector], dtype=float)
    point = mid if args.tool is None else args.tool.tip
    length = reach(rig, chain)
    always = _terms(chain, keys, np.array(start), target, point, length, args, avoid, skin)
    solid_term = (_solid_term(chain, solids, skin, chunks, length) if solids else None)
    fixed = {b: v for b, v in pose.items() if b in path.parent}

    def run_cost(extra: list[Term]) -> Callable[[Sequence[float]], float]:
        terms = always + extra

        def cost(x: Sequence[float]) -> float:
            xv = np.asarray(x, dtype=float)
            world = pose_world(path, _apply(fixed, keys, xv))
            m = world[chain.effector]
            centre = m[:3, :3] @ mid + m[:3, 3]
            return sum(term(xv, world, centre) for term in terms)

        return cost

    starts = [start]
    if args.level is not None and not quick:
        turned = [k for k, (bone, _) in enumerate(keys) if bone == chain.effector]
        for d in LEVEL_TURNS:
            for k in turned:
                starts.append([v + (d if i == k else 0.0) for i, v in enumerate(start)])
    best: list[float] = start
    for n, phase in enumerate(PHASES):
        if phase.solids and solid_term is None:
            continue
        cost = run_cost([solid_term] if phase.solids and solid_term else [])
        if n == 0:
            origins = starts
        else:
            origins = [best] + ([start] if start != best else [])
        found = []
        for x in origins:
            fx = 0.0
            for step in phase.steps:
                x, fx = nelder(cost, x, step)
            found.append((fx, x))
        best = min(found, key=lambda t: t[0])[1]
    solved = _apply(pose, keys, best)
    return SolveResult(pose=solved, **_measure(path, fixed, keys, best, chain, point, target))


def _measure(path: RigData, fixed: Pose, keys, x, chain: Chain, point: np.ndarray,
             target: np.ndarray) -> dict:
    m = pose_world(path, _apply(fixed, keys, x))[chain.effector]
    here = m[:3, :3] @ point + m[:3, 3]
    return {"err": float(np.linalg.norm(here - target)), "point": here}


# --- whole clips ----------------------------------------------------------------------------


def _same_pose(a: Pose, b: Pose) -> bool:
    if set(a) != set(b):
        return False
    return all(np.array_equal(a[k][0], b[k][0]) and np.array_equal(a[k][1], b[k][1]) for k in a)


def _measured(rig: RigData, pose: Pose, chain: Chain, args: ChainArgs, target: np.ndarray
              ) -> SolveResult:
    """A pose's miss at the measured point, with nothing solved."""
    path = _cut_rig(rig, chain)
    point = np.asarray(path.effector_mid[chain.effector], dtype=float)
    if args.tool is not None:
        point = args.tool.tip
    m = pose_world(path, {b: v for b, v in pose.items() if b in path.parent})[chain.effector]
    here = m[:3, :3] @ point + m[:3, 3]
    return SolveResult(pose=pose, err=float(np.linalg.norm(here - target)), point=here)


def _progress(text: str) -> None:
    """One progress line on stderr, flushed, so a caller polling a long background solve sees it
    at once. stderr keeps stdout for a caller's own output."""
    print(text, file=sys.stderr, flush=True)


def solve_clip(rig: RigData, spec: RigSpec, clip: Clip,
               resolve: Callable[[Target, Pose], np.ndarray], *, base: Callable[[int], Pose],
               args: Callable[[Target, Pose], ChainArgs], solids: Sequence[Solid] = (),
               skin: Mapping[str, SkinPoints] | None = None, chunks: Sequence[Chunk] = ()
               ) -> dict[tuple[int, str], SolveResult]:
    """Solve every target of a clip. Returns the results keyed by (frame, chain), so the right
    and left arms solve at one key without a clash.

    - Targets run in key order. `base(frame)` is the pose of everything the solve does not
      turn at that frame. A chain solved later at the same key sees the chains already solved
      there at their solved angles, so a solid that rides the other fist sits where that fist
      went. `base` itself is called once per target; `resolve(target, pose)` is the target's world point under it and
      `args(target, pose)` is what the key asks of the chain.
    - Each key starts from that chain's last solve (the stay term pulls toward it). With
      `keep_last` the key reuses the chain's last solve instead.
    - A tool-tip miss over `TIP_RETRY_MISS_OF_EFFECTOR` of the effector radius moves the aim
      by the miss and solves again, up to `TIP_RETRIES` times. The reported miss is always
      against the real target.
    - A key with a tool that has an earlier solve starts quick (one start); if that leaves the
      miss over `QUICK_RETRY_MISS_OF_REACH` of the chain's reach, it solves again from every
      start.
    - In a `loop` clip, a chain's last key that asks the same as its first reuses the first
      solve, so the loop does not pop.
    - A caller may take one clip's level from another clip's solve (a `level_ref`); the caller
      builds that in its own `args`.
    - Between two neighbouring keys of a chain, the player blends each bone's angles straight
      from one key to the other. When `solids` are given and that blend puts the chain's skin
      deeper past a solid than both keys do, the clip gets a mid-key: an extra entry in the
      result, keyed (frame, chain) at a whole frame between the two. It is solved from the
      blended pose, aimed at the point the blend reaches there, with the same terms as a key. The
      authored keys are solved first and never move. Mid-keys sit on whole frames, so they do
      not clear every sweep: a sweep left between keys one frame apart, or one that the one
      frame tried does not clear, is reported in `sweep_to_next` on the earlier key's result
      (depth past the solid, 0.0 when none), never silent. See `_fill_between`.
    - A solve can run for hours, so one line goes to stderr, flushed, as each key is done:
      `solve <clip>: key <frame> <chain> (<n>/<of>) took <s> s, <s> s in all`. With solids, one
      more per chain pair when its in-between check is done.
    """
    started = time.perf_counter()
    on_a_chain = {b for c in spec.chains.values() for b in c.bones}
    for bone in sorted(set(skin or ()) - on_a_chain):
        raise ManifestError(f"skin names bone {bone!r}, which is on no chain of the rig")
    for chunk in chunks:
        if chunk.bone not in on_a_chain:
            raise ManifestError(f"chunk names bone {chunk.bone!r}, which is on no chain of "
                                "the rig")
    ordered = sorted(clip.targets, key=lambda t: t.key)
    last_frame = ordered[-1].key if ordered else None
    results: dict[tuple[int, str], SolveResult] = {}
    last: dict[str, SolveResult] = {}
    first: dict[str, tuple[int, np.ndarray, Pose, ChainArgs]] = {}
    solved_at: dict[int, Pose] = {}  # per key: the bones of the chains already solved there
    seen: dict[tuple[int, str], tuple[Pose, ChainArgs]] = {}  # per slot: the pose and the ask
    for done, target in enumerate(ordered, 1):
        key_started = time.perf_counter()
        if target.chain not in spec.chains:
            raise ManifestError(f"clip {clip.name!r}: target at {target.key} names chain "
                                f"{target.chain!r}, which the rig lacks")
        slot = (target.key, target.chain)
        if slot in results:
            raise ManifestError(f"clip {clip.name!r}: two targets for chain {target.chain!r} "
                                f"at key {target.key}")
        chain = spec.chains[target.chain]
        pose = {**base(target.key), **solved_at.get(target.key, {})}
        ask = args(target, pose)
        aim = _vec3(resolve(target, pose), f"clip {clip.name!r}: the target point")
        prev = last.get(target.chain)
        if ask.keep_last and prev is not None:
            kept = dict(pose)
            for bone in chain.bones:
                if bone in prev.pose:
                    kept[bone] = prev.pose[bone]
                else:
                    kept.pop(bone, None)
            result = _measured(rig, kept, chain, ask, aim)
        elif (clip.loop and target.chain in first and target.key == last_frame
              and target.key != first[target.chain][0]
              and np.array_equal(aim, first[target.chain][1])
              and _same_pose(pose, first[target.chain][2])
              and _same_args(ask, first[target.chain][3])):
            result = results[(first[target.chain][0], target.chain)]
        else:
            result = _solve_key(rig, spec, chain, pose, aim, ask, prev, solids, skin, chunks)
        results[slot] = result
        last[target.chain] = result
        solved_at.setdefault(target.key, {}).update(
            {b: result.pose[b] for b in chain.bones if b in result.pose})
        first.setdefault(target.chain, (target.key, aim, pose, ask))
        seen[slot] = (pose, ask)
        now = time.perf_counter()
        _progress(f"solve {clip.name}: key {target.key} {target.chain} ({done}/{len(ordered)}) "
                  f"took {now - key_started:.1f} s, {now - started:.1f} s in all")
    if solids:
        for name, chain in spec.chains.items():
            keys = sorted(frame for frame, chain_name in results if chain_name == name)
            ends = {f: ({**seen[(f, name)][0], **solved_at[f]}, seen[(f, name)][1])
                    for f in keys}
            for fa, fb in pairwise(keys):
                pair_started = time.perf_counter()
                mids, left = _fill_between(rig, spec, chain, (fa, ends[fa][0]),
                                           (fb, ends[fb][0]), ends[fa][1], solids, skin, chunks)
                results.update({(f, name): got for f, got in mids.items()})
                for f, depth in left:
                    results[(f, name)] = replace(results[(f, name)], sweep_to_next=depth)
                now = time.perf_counter()
                _progress(f"solve {clip.name}: between keys {fa} and {fb} {name}, {len(mids)} "
                          f"mid-keys, took {now - pair_started:.1f} s, {now - started:.1f} s in all")
    return results


def _blend(a: Pose, b: Pose, t: float) -> Pose:
    """The pose a fraction `t` of the way from `a` to `b`, each bone's angles and shift blended
    straight, as the player blends two keys. A bone one pose lacks rests there."""
    out: Pose = {}
    for bone in sorted(set(a) | set(b)):
        (ea, la), (eb, lb) = a.get(bone, _REST), b.get(bone, _REST)
        out[bone] = ((1 - t) * np.asarray(ea, dtype=float) + t * np.asarray(eb, dtype=float),
                     (1 - t) * np.asarray(la, dtype=float) + t * np.asarray(lb, dtype=float))
    return out


def _worst_between(path: RigData, gauge: _SolidGauge, a: Pose, b: Pose
                   ) -> tuple[float, float, float, float]:
    """How deep the chain's skin goes past the solids along the straight blend from `a` to `b`:
    the depth at `a`, at `b`, and the worst depth strictly between with the blend fraction
    where it falls.

    The blend is sampled until no tested point travels further along a stretch of it than the
    thinnest solid is thick (`_SolidGauge.thinnest`). A point that travels less than that cannot
    get from one side of a solid to the other without a sample finding it inside, so a sweep is
    not stepped over. How far a point travels along a stretch is read from three samples, the
    two ends and the middle: the sum of the two chords, not the one chord from end to end, for a
    turning bone carries its point round an arc that bows far from the chord. The step comes from
    the solids, not from the clip.
    """
    def sample(t: float):
        pose = _blend(a, b, t)
        world = pose_world(path, {bone: v for bone, v in pose.items() if bone in path.parent})
        return gauge.depth(world), gauge.moving(world)

    def travel(first, mid, last) -> float:
        # Per tested point, the two chords; the longest of the skin points plus the longest of
        # the riding solids' corners, as the two move apart in the worst case.
        return sum(float((np.linalg.norm(m - f, axis=1) + np.linalg.norm(e - m, axis=1))
                         .max(initial=0.0)) for f, m, e in zip(first, mid, last))

    samples = {0.0: sample(0.0), 1.0: sample(1.0)}
    todo = [(0.0, 1.0)]
    while todo:
        t0, t1 = todo.pop()
        mid = (t0 + t1) / 2
        samples[mid] = sample(mid)
        if travel(samples[t0][1], samples[mid][1], samples[t1][1]) > gauge.thinnest:
            todo += [(t0, mid), (mid, t1)]
    inner = max((depth, t) for t, (depth, _) in samples.items() if 0.0 < t < 1.0)
    return samples[0.0][0], samples[1.0][0], inner[0], inner[1]


def _fill_between(rig: RigData, spec: RigSpec, chain: Chain,
                  first: tuple[int, Pose], last: tuple[int, Pose], ask: ChainArgs,
                  solids: Sequence[Solid], skin: Mapping[str, SkinPoints] | None,
                  chunks: Sequence[Chunk]
                  ) -> tuple[dict[int, SolveResult], list[tuple[int, float]]]:
    """The solved mid-keys (by whole frame) that clear the sweeps of the blend between two keys
    of a chain, and the sweeps left: (frame of the pair's earlier key, depth past the solid).

    The contract: every sweep of the blend is either cleared by a mid-key on a whole frame, or
    returned in the sweeps left (and so reported in `sweep_to_next`). A sweep left between keys
    one frame apart, or with no whole frame that helps, is reported, never silent.

    - A sweep is a blended pose that is inside a solid (`_SolidGauge.depth`, past its surface)
      deeper than both keys are, by more than `COMPARE_TOL`, the float allowance on a distance.
      So a key that rests against a solid, or inside one, does not make the blend towards it
      a sweep, and a margin is a buffer that a blend may cross.
    - A mid-key sits at a whole frame strictly between the two keys, never at a key's own frame:
      the player shows only whole frames. Two keys one frame apart have no whole frame between
      them, so a sweep there is left.
    - It starts from the blended pose at that frame and aims at the point the chain's measured
      point reaches there, moved by one of `_SolidGauge.exits` so the deepest skin point is
      clear of its solid: the one that makes the shortest way from key to key through the
      mid-key. Aimed at the blended point itself, the reach and the solid terms would settle
      with the skin still part way in. It is kept only when it is shallower than the worst
      blended pose it replaces.
    - One whole frame is tried: the one nearest the worst pose. There is no search over other
      frames (it gave no better result and was slow). Each half is then checked as a pair of keys, the same way. The mid-keys stay unless they leave
      more sweeps than doing nothing does; on a tie they stay whatever the depth of what is
      left. The sweeps left are returned, never dropped. A pair of whole frames shrinks at
      every split, so this ends.
    - Only skin points are sampled and pushed. A solid's back points against arm chunks are
      not, so a sweep made only of them is not seen.
    """
    own_skin = {b: p for b, p in (skin or {}).items() if b in chain.bones}
    own_chunks = [c for c in chunks if c.bone in chain.bones]
    gauge = _solid_gauge(chain, solids, own_skin, own_chunks, reach(rig, chain))
    if gauge is None:
        return {}, []
    if not gauge.thinnest > 0.0:
        raise ManifestError("a solid with a side of no thickness cannot be swept against")
    path = _cut_for_shapes(rig, chain, (), solids, own_skin, own_chunks)
    slack = tol.COMPARE_TOL
    Found = tuple[dict[int, SolveResult], list[tuple[int, float]]]

    def depth_of(pose: Pose) -> float:
        return gauge.depth(pose_world(path, {b: v for b, v in pose.items() if b in path.parent}))

    def mid_key(fa: int, pa: Pose, fm: int, fb: int, pb: Pose, worst: float
                ) -> SolveResult | None:
        blended = _blend(pa, pb, (fm - fa) / (fb - fa))
        here, there, mid = (_measured(rig, p, chain, ask, np.zeros(3)).point
                            for p in (pa, pb, blended))
        ways = gauge.exits(pose_world(path, {b: v for b, v in blended.items()
                                             if b in path.parent}), there - here)
        aim = min((mid + way for way in ways), default=mid,
                  key=lambda p: float(np.linalg.norm(p - here) + np.linalg.norm(there - p)))
        got = _solve_key(rig, spec, chain, blended, aim, ask,
                         SolveResult(pose=blended, err=0.0, point=aim), solids, skin, chunks)
        return got if depth_of(got.pose) <= worst - slack else None

    def fill(fa: int, pa: Pose, fb: int, pb: Pose) -> Found:
        depth_a, depth_b, worst, at = _worst_between(path, gauge, pa, pb)
        if worst <= max(depth_a, depth_b) + slack:
            return {}, []
        stays: Found = ({}, [(fa, worst)])
        if fb - fa < 2:
            return stays
        fm = min(max(round(fa + at * (fb - fa)), fa + 1), fb - 1)
        got = mid_key(fa, pa, fm, fb, pb, worst)
        if got is None:
            return stays
        (mids_a, left_a), (mids_b, left_b) = fill(fa, pa, fm, got.pose), fill(fm, got.pose, fb, pb)
        tried: Found = ({fm: got, **mids_a, **mids_b}, left_a + left_b)
        # Mid-keys that clear part of the sweep stay, unless they leave more sweeps than doing
        # nothing.
        return tried if len(tried[1]) <= len(stays[1]) else stays

    return fill(first[0], first[1], last[0], last[1])


def _solve_key(rig: RigData, spec: RigSpec, chain: Chain, pose: Pose, aim: np.ndarray,
               ask: ChainArgs, prev: SolveResult | None, solids, skin, chunks) -> SolveResult:
    """One key: solve from the chain's last solve, and fix a tool tip that missed."""
    x0 = None if prev is None else prev.pose
    quick = prev is not None and ask.tool is not None
    # Each arm solves against its own skin and chunks, not the other arm's.
    own_skin = None if skin is None else {b: p for b, p in skin.items() if b in chain.bones}
    own_chunks = [c for c in chunks if c.bone in chain.bones]
    extra = {"avoid": spec.avoid, "solids": solids, "skin": own_skin, "chunks": own_chunks}
    result = solve_chain(rig, pose, chain, aim, args=ask, x0=x0, quick=quick, **extra)
    if quick and result.err > tol.QUICK_RETRY_MISS_OF_REACH * reach(rig, chain):
        result = solve_chain(rig, pose, chain, aim, args=ask, x0=x0, quick=False, **extra)
    if ask.tool is None:
        return result
    # The wrist cannot always turn the tool exactly: move the aim by what the tip missed.
    shifted = aim
    for _ in range(tol.TIP_RETRIES):
        miss = aim - result.point
        if float(np.linalg.norm(miss)) < tol.TIP_RETRY_MISS_OF_EFFECTOR * chain.effector_radius:
            break
        shifted = shifted + miss
        result = solve_chain(rig, result.pose, chain, shifted, args=ask, x0=result.pose,
                             quick=True, **extra)
    return SolveResult(pose=result.pose, err=float(np.linalg.norm(aim - result.point)),
                       point=result.point)


# --- left and right -------------------------------------------------------------------------


def other_side(name: str, mirror: MirrorSpec) -> str:
    """The bone's mirror partner by the manifest's pairs. A bone in no pair is its own."""
    for a, b in mirror.pairs:
        if name == a:
            return b
        if name == b:
            return a
    return name


def mirror_pose(pose: Pose, mirror: MirrorSpec) -> Pose:
    """The left/right mirror of a pose, across the plane normal to `mirror.axis`.

    Paired bones swap names. A turn is a pseudovector, so its angle about the mirror axis stays
    and the other two flip; a shift flips only along the mirror axis. This holds when paired
    bones rest as mirror images with the same axes, as the manifest's mirror spec states.
    Mirroring twice gives the pose back.
    """
    i = AXES.index(mirror.axis)
    turn = -np.ones(3)
    turn[i] = 1.0
    shift = np.ones(3)
    shift[i] = -1.0
    out: Pose = {}
    for bone, (euler, loc) in pose.items():
        out[other_side(bone, mirror)] = (np.asarray(euler, dtype=float) * turn,
                                         np.asarray(loc, dtype=float) * shift)
    return out
