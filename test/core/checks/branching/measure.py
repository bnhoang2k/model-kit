"""Measure the honest limb-pair gaps and fork angles of a set of trees (a script, not a test).

    cd test && uv run python -m core.checks.branching.measure [--trees qsm|buds]
        [--opens X] [--reach X]

The trees:
- `qsm` (the default): the four measured trees under test/fixtures/qsm (Demol, Gielen &
  Verbeeck 2021). A laser-scan radius under `R_REL_M` (0.05 m) is not reliable (Demol et al.
  2022: diameters under 10 cm are overestimated, by +73% under 2.5 cm), so only pairs and forks
  whose radii are all at least `R_REL_M` count as honest. Every other row is printed for
  information.
- `buds`: the bud grower's sweep (`bud_sweep_cases`: the two tiny species, seeds 1 to 8). A
  grown radius is exact, so every row counts.

Two tables, and the constants they read on (`core/tolerances.py`, `core/grow/shape.py`):
- Gaps: for every limb pair past its zones, the gap as a share of the thinner radius. The 20
  deepest overlaps and the 20 smallest positive gaps, each with its tree and limb ids and
  whether the two limbs are related (grow from one place). `CROSS_OF_RADIUS` and
  `NEAR_OF_RADIUS` read this table.
- Forks, per order pair (the two limbs' level names): the narrowest angle between a child's
  first segment and its parent's axis there, and between the first segments of two siblings
  that grow from one place (their seat spheres overlap, as the zones read them). This table
  bounds `SIBLING_MIN_ANGLE_DEG` from above (a line over the narrowest honest fork would fail a
  real tree); it does not set it. The measured target comes from all 31 Demol trees
  (`measure_qsm_angles.py`). The zones end where a gap grows slower than limbs parting at that
  angle, for a child against its parent and for siblings alike.
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterable

import numpy as np

from model_kit.core.checks.branching._pairs import limb_pairs, related
from model_kit.core.grow.segments import from_one_place, parent_radius, seat_sphere
from model_kit.core.grow.shape import SEGMENT_EPS
from model_kit.core.grow.skeleton import Skeleton, unit
from model_kit.core.tolerances import SEAT_ZONE_OPENS_OF_RADIUS

from .branching_scene import bud_sweep_trees
from .measure_refs import R_REL_M, fixtures

Tree = tuple[str, Skeleton, float]  # (label, skeleton, the radius from which a row is honest)


def qsm_trees() -> list[Tree]:
    return [(name, sk, R_REL_M * upm) for name, sk, upm in fixtures()]


def grown(trees: Iterable[tuple[str, Skeleton]]) -> list[Tree]:
    return [(label, sk, 0.0) for label, sk in trees]


def gap_rows(trees: list[Tree], opens: float, reach: float) -> list[tuple]:
    """(gap share, tree, a, b, related or not, gap, thinner radius, honest) per limb pair."""
    out = []
    for label, sk, r_honest in trees:
        rel = related(sk)
        for i, j, gap, r in limb_pairs(sk, reach=reach, opens=opens):
            a, b = sorted((str(sk.limb_id[i]), str(sk.limb_id[j])))
            out.append((gap / r, label, a, b, "related" if (i, j) in rel else "unrelated",
                        gap, r, r >= r_honest))
    return out


def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(unit(u) @ unit(v))))))


def _first_dir(sk: Skeleton, i: int) -> np.ndarray:
    pts = sk.limb_points(i)
    return pts[1] - pts[0]


def _parent_tangent(sk: Skeleton, i: int) -> np.ndarray:
    """The parent's axis direction on its segment nearest limb i's base."""
    pp = sk.limb_points(int(sk.parent[i]))
    q = sk.limb_points(i)[0]
    ab = pp[1:] - pp[:-1]
    f = np.clip(((q - pp[:-1]) * ab).sum(axis=1)
                / np.maximum((ab * ab).sum(axis=1), SEGMENT_EPS), 0.0, 1.0)
    d = np.sqrt(((pp[:-1] + ab * f[:, None] - q) ** 2).sum(axis=1))
    return ab[int(np.argmin(d))]


def fork_rows(trees: list[Tree], clearance: float) -> list[tuple]:
    """(angle, kind, order pair, tree, a, b, honest) for every child against its parent and for
    every sibling pair that grows from one place."""
    out = []
    for label, sk, r_honest in trees:
        names = sk.level_names
        kids: dict[int, list[int]] = {}
        for i, p in enumerate(sk.parent):
            if p < 0:
                continue
            kids.setdefault(int(p), []).append(i)
            r0 = float(sk.limb_radii(i)[0])
            out.append((_angle_deg(_first_dir(sk, i), _parent_tangent(sk, i)), "child",
                        f"{names[sk.level[p]]}/{names[sk.level[i]]}", label,
                        str(sk.limb_id[p]), str(sk.limb_id[i]), r0 >= r_honest))
        for p, group in kids.items():
            pts_p, rad_p = sk.limb_points(p), sk.limb_radii(p)
            sphere = {i: seat_sphere(parent_radius(sk.limb_points(i), pts_p, rad_p),
                                     float(sk.limb_radii(i)[0]), clearance) for i in group}
            for n, a in enumerate(group):
                for b in group[n + 1:]:
                    if not from_one_place(sk.limb_points(a)[0], sphere[a],
                                          sk.limb_points(b)[0], sphere[b]):
                        continue
                    la, lb = sorted((names[sk.level[a]], names[sk.level[b]]))
                    honest = min(float(sk.limb_radii(a)[0]),
                                 float(sk.limb_radii(b)[0])) >= r_honest
                    out.append((_angle_deg(_first_dir(sk, a), _first_dir(sk, b)), "siblings",
                                f"{la}/{lb}", label, str(sk.limb_id[a]), str(sk.limb_id[b]),
                                honest))
    return out


def print_gaps(found: list[tuple], reach: float, opens: float, n_trees: int,
               took: float) -> None:
    found = sorted(found)
    honest = [row for row in found if row[7]]
    print(f"{len(found)} limb pairs within reach {reach} (zone opens at {opens:.3f}), "
          f"{n_trees} trees, {took:.1f} s; {len(honest)} honest")
    for title, rows in (("honest", honest), ("all (information)", found)):
        print(f"\n20 deepest overlaps, {title} (gap / thinner radius):")
        for row in [r for r in rows if r[0] <= 0.0][:20]:
            print(f"  {row[0]:+.4f}  {row[1]:<20} {row[2]} x {row[3]}  {row[4]}  "
                  f"gap {row[5]:+.5f} r {row[6]:.5f}")
        print(f"\n20 smallest positive gaps, {title} (gap / thinner radius):")
        for row in [r for r in rows if r[0] > 0.0][:20]:
            print(f"  {row[0]:+.4f}  {row[1]:<20} {row[2]} x {row[3]}  {row[4]}  "
                  f"gap {row[5]:+.5f} r {row[6]:.5f}")


def print_forks(rows: list[tuple]) -> None:
    print("\nNarrowest forks per order pair (degrees; n honest of n all):")
    print(f"  {'kind':<9} {'order pair':<20} {'honest':>7} {'all':>7}  narrowest honest")
    groups: dict[tuple[str, str], list[tuple]] = {}
    for row in rows:
        groups.setdefault((row[1], row[2]), []).append(row)
    for (kind, pair), group in sorted(groups.items()):
        honest = sorted(r for r in group if r[6])
        every = min(r[0] for r in group)
        best = (f"{honest[0][0]:7.2f}  {honest[0][3]} {honest[0][4]} x {honest[0][5]}"
                if honest else "      -")
        print(f"  {kind:<9} {pair:<20} {len(honest):>7} {len(group):>7}  {best}"
              f"   (all: {every:.2f})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trees", choices=("qsm", "buds"), default="qsm")
    parser.add_argument("--opens", type=float, default=SEAT_ZONE_OPENS_OF_RADIUS,
                        help="the seat zone ends where the gap share first exceeds this")
    parser.add_argument("--reach", type=float, default=3.0,
                        help="find every pair with a gap share up to this")
    args = parser.parse_args()
    t0 = time.perf_counter()
    if args.trees == "qsm":
        trees = qsm_trees()
    else:
        trees = grown(bud_sweep_trees())
    found = gap_rows(trees, args.opens, args.reach)
    print_gaps(found, args.reach, args.opens, len(trees), time.perf_counter() - t0)
    print_forks(fork_rows(trees, args.opens))


if __name__ == "__main__":
    main()
