"""A grown piece before it has triangles: every limb's axis and radii, every leaf's frame, and
the mesh settings that turn them into parts.

Arrays use glTF axes (X, Y up, Z toward the camera), in model units, in the piece frame (the
trunk centre on the ground). Limb `i` holds points `pts[start[i]:start[i + 1]]` and is emitted
as part `level_names[level[i]]`. A parent always comes before its children.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.grow.shape import LEAF_FRAME_EPS, SEGMENT_EPS

CONE, DOME = 0, 1  # limb tip kinds: a point, or a round end

# The level index of each limb order, in the order the rules' `levels.names` lists them.
TRUNK, LEADER, BRANCH, TWIG, SHOOT = range(5)


@dataclass(frozen=True, eq=False)
class Skeleton:
    level_names: tuple[str, ...]
    # limbs
    limb_id: np.ndarray  # (L,) str: trunk, root.0, leader.2, branch.17, twig.40, shoot.311
    level: np.ndarray  # (L,) int, index into level_names
    parent: np.ndarray  # (L,) int, -1 when rooted on nothing
    t: np.ndarray  # (L,) the seat's arc fraction on the parent
    tip: np.ndarray  # (L,) CONE or DOME
    rooted: np.ndarray  # (L,) bool: stands on the soil, so its base must be in it
    buried: np.ndarray  # (L,) bool: grows into the soil, so its tip must be in it
    start: np.ndarray  # (L + 1,) offsets into pts and radii
    pts: np.ndarray  # (P, 3)
    radii: np.ndarray  # (P,)
    # leaves
    host: np.ndarray  # (F,) the limb a leaf grows on
    leaf_t: np.ndarray  # (F,) the node's arc fraction on the host
    node: np.ndarray  # (F, 3) on the host's axis
    base: np.ndarray  # (F, 3) where the blade starts
    mid: np.ndarray  # (F, 3) the midrib direction
    across: np.ndarray  # (F, 3) across the blade
    normal: np.ndarray  # (F, 3) the blade's face
    length: np.ndarray  # (F,) blade length
    petiole: np.ndarray  # (F,) the distance from node to base the grower used
    shade: np.ndarray  # (F,) int, index into shades
    # mesh settings
    shades: tuple[str, ...]
    sides: np.ndarray  # (len(level_names),) tube sides per level
    outline: np.ndarray  # (K, 2) the blade outline, in blade lengths
    fold: float  # the blade halves rise fold * |x| along the normal
    blade_centre: np.ndarray  # (3,) the fan centre (across, along, up), in blade lengths
    petiole_width: float  # the petiole ribbon's half-width (vertices at +-w), model units
    petiole_tip: float  # the petiole ends this share of the blade length up the midrib
    lite: bool
    info: dict[str, Any]

    def limb_points(self, i: int) -> np.ndarray:
        return self.pts[self.start[i]:self.start[i + 1]]

    def limb_radii(self, i: int) -> np.ndarray:
        return self.radii[self.start[i]:self.start[i + 1]]

    def at(self, i: int, t: float) -> tuple[np.ndarray, float, np.ndarray]:
        """(point, radius, unit tangent) at arc fraction t of limb i."""
        return limb_at(self.limb_points(i), self.limb_radii(i), t)

    def nearest(self, i: int, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For each point of q (M, 3): its distance to limb i's axis and the radius there."""
        return limb_nearest(self.limb_points(i), self.limb_radii(i), q)


def unit(v: np.ndarray) -> np.ndarray:
    """v scaled to length 1; a zero vector stays zero (as mathutils does)."""
    n = float(np.sqrt(v @ v))
    return v / n if n > 0.0 else np.zeros_like(v)


def arc_lengths(pts: np.ndarray) -> np.ndarray:
    """(n,) the arc length from the first point to each point."""
    seg = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def _level_share(v: np.ndarray) -> np.ndarray:
    """(m,) how far a ring perpendicular to each row of v reaches below its centre, per unit
    radius: the level part of the unit direction, hypot(d_x, d_z). That is sqrt(1 - d_y^2), read
    from the level part so an upright ring reaches exactly 0. A zero row has no direction: 1."""
    n = np.sqrt((v * v).sum(axis=1))
    level = np.hypot(v[:, 0], v[:, 2])
    return np.where(n > 0.0, level / np.where(n > 0.0, n, 1.0), 1.0)


def ring_lows(pts: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """(n,) the lowest y of the ring at each point of a limb drawn as a tube of rings.

    A ring of radius r perpendicular to a unit direction d reaches r x hypot(d_x, d_z) below its
    centre, so an upright ring lies at its centre's height and a level one reaches the whole r. A
    point's ring is read on every direction it may face, and the lowest is taken: each adjacent
    segment's, and at a joint the central difference of its neighbours (the tangent the mesh draws
    it on, `mesh.tube_tris`), which at a peak or a dip of the axis lies level. A band between two
    rings joins their corners by straight lines, so no part of the tube between them lies lower
    than the lower ring. The tip cap is not a ring: a cone tip that points down reaches past it."""
    pts, radii = np.asarray(pts, dtype=float), np.asarray(radii, dtype=float)
    share = np.zeros(len(pts))
    if len(pts) > 1:
        seg = _level_share(pts[1:] - pts[:-1])
        share[:-1] = np.maximum(share[:-1], seg)
        share[1:] = np.maximum(share[1:], seg)
    if len(pts) > 2:
        share[1:-1] = np.maximum(share[1:-1], _level_share(pts[2:] - pts[:-2]))
    return pts[:, 1] - radii * share


def limb_at(
    pts: np.ndarray, radii: np.ndarray, t: float, cum: np.ndarray | None = None
) -> tuple[np.ndarray, float, np.ndarray]:
    """Walk the arc length to fraction t, clamped to [0, 1]. `cum` is
    `arc_lengths(pts)`, passed in when the caller keeps it."""
    if cum is None:
        cum = arc_lengths(pts)
    s = max(0.0, min(1.0, t)) * cum[-1]
    last = len(pts) - 2
    for i in range(last + 1):
        if s <= cum[i + 1] or i == last:
            d = cum[i + 1] - cum[i]
            f = 0.0 if d <= 0 else (s - cum[i]) / d
            p = pts[i] + (pts[i + 1] - pts[i]) * f
            r = radii[i] + (radii[i + 1] - radii[i]) * f
            return p, float(r), unit(pts[i + 1] - pts[i])
    raise ValueError("a limb needs at least two points")


def limb_nearest(pts: np.ndarray, radii: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The nearest point on a limb, vectorised over q: the first closest segment wins a tie."""
    q = np.asarray(q, dtype=float).reshape(-1, 3)
    a, ab = pts[:-1], pts[1:] - pts[:-1]  # (S, 3)
    den = np.maximum((ab * ab).sum(axis=1), SEGMENT_EPS)  # (S,)
    f = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(axis=2) / den, 0.0, 1.0)  # (M, S)
    closest = a[None] + ab[None] * f[..., None]
    dist = np.sqrt(((closest - q[:, None, :]) ** 2).sum(axis=2))
    k = np.argmin(dist, axis=1)
    rows = np.arange(len(q))
    r = radii[k] + (radii[k + 1] - radii[k]) * f[rows, k]
    return dist[rows, k], r


# ------------------------------------------------------------------------------------- the file

_ARRAYS = ("limb_id", "level", "parent", "t", "tip", "rooted", "buried", "start", "pts", "radii",
           "host", "leaf_t", "node", "base", "mid", "across", "normal", "length", "petiole",
           "shade", "sides", "outline", "blade_centre")
_SCALARS = ("fold", "petiole_width", "petiole_tip")
# Skeleton files saved by an earlier version of the grower hold
# these two numbers, from before the leaf-normal lean moved to style.json. `load_skeleton` reads
# and drops them, so those files keep loading.
_LEGACY_SCALARS = ("normal_to_crown", "crown_centre_drop")
_NAMES = ("level_names", "shades")


# The dtype each array must have.
_KINDS = {
    **dict.fromkeys(("level", "parent", "start", "host", "shade", "tip", "sides"), np.integer),
    **dict.fromkeys(("rooted", "buried"), np.bool_),
    **dict.fromkeys(("t", "pts", "radii", "leaf_t", "node", "base", "mid", "across", "normal",
                     "length", "petiole", "outline", "blade_centre"), np.floating),
    "limb_id": np.str_,
}


def _plain(value: Any) -> Any:
    """info with numpy scalars and arrays turned into Python numbers and lists, for JSON."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    return value


def check_skeleton(sk: Skeleton, where: str = "skeleton") -> None:
    """Every array has the shape the others imply, every number is finite, radii and leaf lengths
    are positive, and every parent comes before its child."""

    def fail(field: str, why: str):
        raise ManifestError(f"{where}: {field}: {why}")

    for name, kind in _KINDS.items():
        value = np.asarray(getattr(sk, name))
        if not np.issubdtype(value.dtype, kind):
            fail(name, f"dtype {value.dtype}, want {kind.__name__}")
    n_limbs = len(sk.limb_id)
    n_leaves = len(sk.host)
    n_pts = int(sk.start[-1]) if len(sk.start) else -1
    want = {
        "limb_id": (n_limbs,), "level": (n_limbs,), "parent": (n_limbs,), "t": (n_limbs,),
        "tip": (n_limbs,), "rooted": (n_limbs,), "buried": (n_limbs,), "start": (n_limbs + 1,),
        "pts": (n_pts, 3), "radii": (n_pts,),
        "host": (n_leaves,), "leaf_t": (n_leaves,), "node": (n_leaves, 3),
        "base": (n_leaves, 3), "mid": (n_leaves, 3), "across": (n_leaves, 3),
        "normal": (n_leaves, 3), "length": (n_leaves,), "petiole": (n_leaves,),
        "shade": (n_leaves,), "sides": (len(sk.level_names),), "blade_centre": (3,),
    }
    for name, shape in want.items():
        got = np.shape(getattr(sk, name))
        if got != shape:
            fail(name, f"shape {got}, want {shape}")
    if np.ndim(sk.outline) != 2 or np.shape(sk.outline)[1] != 2 or len(sk.outline) < 3:
        fail("outline", f"shape {np.shape(sk.outline)}, want (K >= 3, 2)")
    for name, kind in _KINDS.items():
        if kind is np.floating:
            value = np.asarray(getattr(sk, name), dtype=float)
            bad = np.argwhere(~np.isfinite(value))
            if len(bad):
                at = tuple(int(i) for i in bad[0])
                fail(name, f"index {at if len(at) > 1 else at[0]} is {float(value[tuple(bad[0])])!r}; "
                           "every number must be finite")
    for name in _SCALARS:
        if not math.isfinite(float(getattr(sk, name))):
            fail(name, f"is {getattr(sk, name)!r}; it must be finite")
    for name in ("radii", "length"):
        value = np.asarray(getattr(sk, name), dtype=float)
        bad = np.flatnonzero(value <= 0.0)
        if len(bad):
            fail(name, f"index {int(bad[0])} is {float(value[bad[0]])!r}; it must be over 0")
    if sk.start[0] != 0 or np.any(np.diff(sk.start) < 2):
        fail("start", "must start at 0 and give every limb at least two points")
    bad = np.flatnonzero(sk.parent >= np.arange(n_limbs))
    if len(bad):
        i = int(bad[0])
        fail("parent", f"limb {i} ({sk.limb_id[i]}) has parent {int(sk.parent[i])}, "
                       "which does not come before it")
    if np.any(sk.parent < -1):
        fail("parent", "must be -1 or a limb index")
    if np.any((sk.level < 0) | (sk.level >= len(sk.level_names))):
        fail("level", "must index level_names")
    if n_leaves and (np.any(sk.host < 0) or np.any(sk.host >= n_limbs)):
        fail("host", "must index the limbs")
    if n_leaves and (np.any(sk.shade < 0) or np.any(sk.shade >= len(sk.shades))):
        fail("shade", "must index shades")
    for name in ("mid", "across", "normal"):
        length = np.sqrt((np.asarray(getattr(sk, name), dtype=float) ** 2).sum(axis=1))
        bad = np.flatnonzero(np.abs(length - 1.0) > LEAF_FRAME_EPS)
        if len(bad):
            fail(name, f"leaf {int(bad[0])} has length {length[bad[0]]:.6g}; a leaf frame is "
                       "unit vectors")


def save_skeleton(sk: Skeleton, path: Path) -> None:
    path = Path(path)
    check_skeleton(sk, str(path))
    data: dict[str, np.ndarray] = {name: np.asarray(getattr(sk, name)) for name in _ARRAYS}
    data |= {name: np.asarray(float(getattr(sk, name))) for name in _SCALARS}
    data |= {name: np.array(getattr(sk, name), dtype=str) for name in _NAMES}
    data["lite"] = np.asarray(bool(sk.lite))
    data["info"] = np.asarray(json.dumps(_plain(sk.info), sort_keys=True))
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **data)


def load_skeleton(path: Path) -> Skeleton:
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        want = {*_ARRAYS, *_SCALARS, *_NAMES, "lite", "info"}
        keys -= set(_LEGACY_SCALARS)
        if keys != want:
            raise ManifestError(f"{path}: arrays: missing {sorted(want - keys)}, "
                                f"unknown {sorted(keys - want)}")
        kw: dict[str, Any] = {name: z[name] for name in _ARRAYS}
        kw["limb_id"] = kw["limb_id"].astype(str)
        kw |= {name: float(z[name]) for name in _SCALARS}
        kw |= {name: tuple(str(v) for v in z[name]) for name in _NAMES}
        kw["lite"] = bool(z["lite"])
        kw["info"] = json.loads(str(z["info"]))
    sk = Skeleton(**kw)
    check_skeleton(sk, str(path))
    return sk


def same_skeleton(a: Skeleton, b: Skeleton) -> bool:
    """Every array and field equal."""
    for f in fields(Skeleton):
        x, y = getattr(a, f.name), getattr(b, f.name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape or not np.array_equal(x, y):
                return False
        elif x != y:
            return False
    return True
