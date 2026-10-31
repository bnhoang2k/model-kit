"""Measure the plant-check candidates on the four measured trees and the leaf-angle data (a
script, not a test). Its output is the reference sweep for the plant checks.

    uv run python -I test/core/checks/branching/measure_refs.py [--csv dataset_full.csv]
        [--refs-json out.json] [--skip-kept]

`-I` keeps the working folder off the import path, so nothing in a download folder runs. The
script finds the repo's `test/` folder from its own path. `--csv` is the Chianucci et al. 2018
leaf-angle file (Mendeley doi:10.17632/4rmc7r8zvy.2, `dataset_full.csv`), read as data with the
csv module only. Without it, the leaf-angle tables are skipped.

Measures, one function each:

- `taper_rows`: per limb, the least-squares slope of radius over arc length (both in metres),
  fitted over the limb's points whose radius is at least `r_rel`, on limbs with 4 such points.
- `leonardo_fit`: per tree, the exponent D that minimises sum (D log r_p - log sum r_c^D)^2 over
  forks whose parent radius is at least `r_rel`. A fork is one parent segment with side
  children: the parent is that segment's radius, the children are the next segment (the chain
  runs on) and every side child's base radius.
- `kept_findings`: the five branching checks that judge any skeleton, on each tree, for information.
- `strahler_counts`: Strahler stream counts and bifurcation ratios, all limbs and limbs from
  `r_rel` up.
- `balance_proxy`: the leafless trees' stand-in for the leaf-area centre: terminal limbs weighted
  by their base cross-section (the pipe model: a stem's area follows the leaf area it carries).
- `leaf_angle_tables`: species means, sector SDs as `Script.r` groups them, the mean band and the
  beta (a, b) of two species by the method of moments on t = 2 theta / pi.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from model_kit.core.grow.skeleton import Skeleton, arc_lengths, load_skeleton

TEST = Path(__file__).resolve().parents[3]  # the repo's test/ folder

# The reliability radius, in metres. Demol et al. 2022 (Silva Fennica 56(1) 10550), on leaf-off
# ash scanned and filtered as the Demol et al. 2021 set was: "TLS-derived diameters of small
# branches (<10 cm) of leaf-off trees were generally larger than manually measured diameters";
# parts over 10 cm were overestimated by +4% on average. So a diameter of 10 cm, radius 0.05 m.
R_REL_M = 0.05
INFO_RADII_M = (0.0125, 0.025, 0.05)  # 2.5, 5 and 10 cm diameters, the classes Demol 2022 uses
MIN_POINTS = 4  # a limb is fitted only with 4 or more points (a design call of this kit)
LEONARDO_BAND = (1.8, 2.3)  # Eloy 2011
D_GRID = np.linspace(0.5, 6.0, 1101)  # the D search for the fork fit


# --- the trees ---------------------------------------------------------------------------------


def fixtures() -> list[tuple[str, Skeleton, float]]:
    """(name, skeleton, units per metre) for the four measured trees."""
    if str(TEST) not in sys.path:
        sys.path.insert(0, str(TEST))
    from core.grow.qsm_fixtures import QSM_FILES

    out = []
    for path in QSM_FILES:
        sk = load_skeleton(path)
        out.append((path.name.split(".")[0], sk, float(sk.info["units_per_metre"])))
    return out


def limb_metres(sk: Skeleton, upm: float, i: int) -> tuple[np.ndarray, np.ndarray]:
    """(arc length, radius) along limb i, in metres."""
    return arc_lengths(sk.limb_points(i)) / upm, sk.limb_radii(i) / upm


def children(sk: Skeleton) -> list[list[int]]:
    kids: list[list[int]] = [[] for _ in range(len(sk.limb_id))]
    for k, p in enumerate(sk.parent):
        if p >= 0:
            kids[int(p)].append(k)
    return kids


# --- taper -------------------------------------------------------------------------------------


def taper_rows(sk: Skeleton, upm: float, r_rel: float) -> list[tuple[int, int, float, float, int]]:
    """(limb, level, base radius m, slope dr/ds, points used) for each judged limb."""
    out = []
    for i in range(len(sk.limb_id)):
        s, r = limb_metres(sk, upm, i)
        keep = r >= r_rel
        if keep.sum() < MIN_POINTS or r[0] < r_rel:
            continue
        slope = float(np.polyfit(s[keep], r[keep], 1)[0])
        out.append((i, int(sk.level[i]), float(r[0]), slope, int(keep.sum())))
    return out


def print_taper(trees, r_rel: float) -> None:
    print(f"\n== taper: radius slope per limb, points with r >= {r_rel} m, limbs with >= "
          f"{MIN_POINTS} such points and base r >= {r_rel} m")
    print(f"{'tree':<9}{'level':>6}{'base r band (m)':>18}{'limbs':>7}{'median':>10}"
          f"{'max':>10}{'> 0':>6}")
    bands = ((r_rel, 2 * r_rel), (2 * r_rel, 4 * r_rel), (4 * r_rel, math.inf))
    for name, sk, upm in trees:
        rows = taper_rows(sk, upm, r_rel)
        for level in sorted({row[1] for row in rows}):
            for lo, hi in bands:
                sel = [row for row in rows if row[1] == level and lo <= row[2] < hi]
                if not sel:
                    continue
                sl = np.array([row[3] for row in sel])
                print(f"{name:<9}{level:>6}{f'{lo:.3f}-{hi:.3f}':>18}{len(sel):>7}"
                      f"{np.median(sl):>10.4f}{sl.max():>10.4f}{int((sl > 0).sum()):>6}")
        up = [row for row in rows if row[3] > 0]
        for row in up:
            print(f"  rising: {name} {sk.limb_id[row[0]]} level {row[1]} base r {row[2]:.4f} m "
                  f"slope {row[3]:+.5f} over {row[4]} points")
        print(f"  {name}: {len(rows)} limbs judged, {len(up)} with slope > 0")


def print_taper_summary(trees) -> None:
    print("\n== taper: rising limbs by reliability radius (information)")
    print(f"{'r_rel (m)':>10}" + "".join(f"{name:>16}" for name, _, _ in trees))
    for r_rel in INFO_RADII_M:
        cells = []
        for _, sk, upm in trees:
            rows = taper_rows(sk, upm, r_rel)
            cells.append(f"{sum(row[3] > 0 for row in rows)}/{len(rows)}")
        print(f"{r_rel:>10.4f}" + "".join(f"{c:>16}" for c in cells))


# --- Leonardo ----------------------------------------------------------------------------------


def forks(sk: Skeleton, upm: float) -> tuple[list[tuple[float, list[float]]], dict[str, int]]:
    """[(parent r m, [child r m])] per fork, and counts of the cases the rule settles."""
    kids = children(sk)
    out = []
    counts = {"side_children": 0, "at_segment_end": 0, "on_last_segment": 0}
    for p in range(len(sk.limb_id)):
        if not kids[p]:
            continue
        cum = arc_lengths(sk.limb_points(p))
        rad = sk.limb_radii(p) / upm
        last = len(cum) - 2  # the last segment: its end point repeats its radius
        groups: dict[int, list[float]] = defaultdict(list)
        for k in kids[p]:
            s = float(sk.t[k]) * cum[-1]
            j = int(np.searchsorted(cum, s, side="left")) - 1
            j = min(max(j, 0), last)
            if np.isclose(s, cum[j + 1], rtol=0.0, atol=1e-9 * max(cum[-1], 1.0)):
                counts["at_segment_end"] += 1
            counts["side_children"] += 1
            groups[j].append(float(sk.limb_radii(k)[0] / upm))
        for j, side in groups.items():
            if j == last:  # no segment runs on past it, so its children are not a full fork
                counts["on_last_segment"] += len(side)
                continue
            out.append((float(rad[j]), [float(rad[j + 1]), *side]))
    return out, counts


def leonardo_fit(fk: list[tuple[float, list[float]]], log_radius: bool = False) -> float:
    """The D in D_GRID that minimises sum (D log r_p - log sum r_c^D)^2 (the original form), or
    with `log_radius` sum (log r_p - log sum r_c^D / D)^2 (the same equation, residual in log
    radius)."""
    lp = np.log(np.array([f[0] for f in fk]))
    best, best_d = math.inf, float("nan")
    for d in D_GRID:
        rhs = np.log(np.array([np.sum(np.power(f[1], d)) for f in fk]))
        res = lp - rhs / d if log_radius else d * lp - rhs
        cost = float(np.sum(res ** 2))
        if cost < best:
            best, best_d = cost, float(d)
    return best_d


def fork_d(rp: float, rc: list[float]) -> float:
    """The D with r_p^D = sum r_c^D for one fork, or nan when no D > 0 gives it."""
    x = np.array(rc) / rp
    if np.any(x >= 1.0) or len(x) < 2:
        return float("nan")
    lo, hi = 0.05, 50.0  # sum x^D falls from len(x) to 0 as D grows
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.sum(x ** mid) > 1.0 else (lo, mid)
    return 0.5 * (lo + hi)


def print_leonardo(trees, r_rel: float) -> None:
    print(f"\n== leonardo: forks with parent r >= {r_rel} m; band {LEONARDO_BAND}")
    print(f"{'tree':<9}{'forks':>7}{'fit D':>8}{'in band':>9}{'log-r fit D':>13}"
          f"{'median fork D':>15}{'no fork D':>10}{'median area ratio':>19}")
    for name, sk, upm in trees:
        all_fk, counts = forks(sk, upm)
        fk = [f for f in all_fk if f[0] >= r_rel]
        if not fk:
            print(f"{name:<9}{0:>7}  (no fork)")
            continue
        d = leonardo_fit(fk)
        per = np.array([fork_d(rp, rc) for rp, rc in fk])
        ratio = np.array([sum(c * c for c in rc) / (rp * rp) for rp, rc in fk])
        inside = LEONARDO_BAND[0] <= d <= LEONARDO_BAND[1]
        print(f"{name:<9}{len(fk):>7}{d:>8.3f}{'yes' if inside else 'NO':>9}"
              f"{leonardo_fit(fk, log_radius=True):>13.3f}{np.nanmedian(per):>15.3f}"
              f"{int(np.isnan(per).sum()):>10}{np.median(ratio):>19.3f}")
        print(f"  {name}: {counts['side_children']} side children, {counts['at_segment_end']} "
              f"seated on a segment end (put on the lower segment), "
              f"{counts['on_last_segment']} on a limb's last segment (left out)")
    print("\n== leonardo: fitted D by reliability radius (information)")
    print(f"{'r_rel (m)':>10}" + "".join(f"{name:>16}" for name, _, _ in trees))
    for rr in INFO_RADII_M:
        cells = []
        for _, sk, upm in trees:
            fk = [f for f in forks(sk, upm)[0] if f[0] >= rr]
            cells.append(f"{leonardo_fit(fk):.3f} ({len(fk)})" if fk else "none")
        print(f"{rr:>10.4f}" + "".join(f"{c:>16}" for c in cells))


# --- the five kept checks ----------------------------------------------------------------------


def kept_findings(trees) -> None:
    if str(TEST) not in sys.path:
        sys.path.insert(0, str(TEST))
    from core.checks.branching.branching_scene import grown_ctx
    from model_kit.core.checks.branching import BRANCHING_CHECKS

    print("\n== the five kept checks on each tree (information)")
    for name, sk, _ in trees:
        t0 = time.perf_counter()
        ctx = grown_ctx(sk)
        print(f"{name}: scene built in {time.perf_counter() - t0:.1f} s")
        for make in BRANCHING_CHECKS:
            check = make()
            t0 = time.perf_counter()
            try:
                found = check.run(ctx)
                cover = check.coverage(ctx)
            except Exception as exc:  # noqa: BLE001 - a script: report and go on
                print(f"  {check.name:<20} raised {type(exc).__name__}: {exc}")
                continue
            took = time.perf_counter() - t0
            print(f"  {check.name:<20} {len(found):>5} findings  {took:6.1f} s"
                  f"  coverage: {'; '.join(cover) if cover else '-'}")
            for f in found[:5]:
                print(f"      {f.item} | {f.detail}")


# --- Strahler ----------------------------------------------------------------------------------


def strahler_counts(sk: Skeleton, upm: float, r_min: float) -> dict[int, int]:
    """Strahler stream counts {order: n} over limbs whose base radius is at least r_min (and
    whose parents all are)."""
    kids = children(sk)
    n_limbs = len(sk.limb_id)
    keep = np.zeros(n_limbs, dtype=bool)
    for i in range(n_limbs):  # parents come first
        ok = sk.limb_radii(i)[0] / upm >= r_min
        keep[i] = ok and (sk.parent[i] < 0 or keep[int(sk.parent[i])])
    order = np.zeros(n_limbs, dtype=int)
    counts: dict[int, int] = defaultdict(int)
    for i in reversed(range(n_limbs)):
        if not keep[i]:
            continue
        cur = 1
        counts[1] += 1  # the limb's tip starts a first-order stream
        for k in sorted((k for k in kids[i] if keep[k]), key=lambda k: -float(sk.t[k])):
            o = int(order[k])
            if o == cur:
                cur += 1
                counts[cur] += 1
            elif o > cur:
                cur = o
        order[i] = cur
    return dict(counts)


def print_strahler(trees, r_rel: float) -> None:
    print("\n== branching-stats: Strahler streams and bifurcation ratios N_k / N_k+1 "
          "(information)")
    for r_min in (0.0, r_rel):
        print(f"limbs with base r >= {r_min} m:")
        for name, sk, upm in trees:
            c = strahler_counts(sk, upm, r_min)
            ks = sorted(c)
            ratios = [c[k] / c[k + 1] for k in ks[:-1]]
            slope = np.polyfit(ks, np.log10([c[k] for k in ks]), 1)[0] if len(ks) > 1 else 0.0
            print(f"  {name:<9} N = {[c[k] for k in ks]}  R_k = "
                  f"{[round(r, 2) for r in ratios]}  R_b (log fit) = {10 ** -slope:.2f}")


# --- crown balance -----------------------------------------------------------------------------


def balance_proxy(sk: Skeleton, upm: float) -> tuple[float, float]:
    """(horizontal offset m, offset / height) of the terminal limbs' area-weighted centre from
    the trunk base."""
    kids = children(sk)
    w, c = [], []
    for i in range(len(sk.limb_id)):
        if kids[i]:
            continue
        r0 = sk.limb_radii(i)[0] / upm
        w.append(math.pi * r0 * r0)
        c.append(sk.limb_points(i).mean(axis=0) / upm)
    w_arr, c_arr = np.array(w), np.array(c)
    centre = (w_arr[:, None] * c_arr).sum(axis=0) / w_arr.sum()
    base = sk.limb_points(0)[0] / upm
    off = float(math.hypot(centre[0] - base[0], centre[2] - base[2]))
    height = float((sk.pts[:, 1].max() - sk.pts[:, 1].min()) / upm)
    return off, off / height


def print_balance(trees) -> None:
    print("\n== crown-balance: stand-in leaf-area centre (terminal limbs weighted by base area); "
          "the trees have no leaves (information)")
    for name, sk, upm in trees:
        off, share = balance_proxy(sk, upm)
        print(f"  {name:<9} offset {off:.2f} m, {share:.3f} of the height")


# --- leaf angles -------------------------------------------------------------------------------


def read_angles(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def stats(theta: np.ndarray) -> tuple[float, float, int]:
    """Mean and SD (n - 1, as R's sd) in degrees, and n."""
    return float(theta.mean()), float(theta.std(ddof=1)), len(theta)


def moments_beta(theta: np.ndarray) -> tuple[float, float]:
    """(a, b) of Beta(a, b) on t = 2 theta / pi, density t^(a-1) (1-t)^(b-1), by the method of
    moments as `Script.r` does: k = tm (1 - tm) / var - 1, a = tm k, b = (1 - tm) k. Script.r
    names them the other way round: its nu is a, its mu is b."""
    t = np.radians(theta) * 2.0 / math.pi
    tm, var = float(t.mean()), float(t.var(ddof=1))
    k = tm * (1.0 - tm) / var - 1.0
    return tm * k, (1.0 - tm) * k


def leaf_angle_tables(path: Path, refs_json: Path | None) -> None:
    rows = read_angles(path)
    print(f"\n== leaf-angle: {len(rows)} leaves from {path.name}")
    by_species: dict[str, list[float]] = defaultdict(list)
    by_combo: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_sector: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        sp = f"{r['Genus']} {r['Species']}"
        a = float(r["Angle_degree"])
        by_species[sp].append(a)
        by_combo[(sp, r["Date"])].append(a)
        if r["Canopy_sector"] != "whole canopy":
            by_sector[(sp, r["Canopy_sector"], r["Date"])].append(a)

    sp_stats = {sp: stats(np.array(v)) for sp, v in by_species.items()}
    means = sorted((m, sp) for sp, (m, _, _) in sp_stats.items())
    print(f"species: {len(sp_stats)}; species-by-date sets (Script.r's Combo): {len(by_combo)}; "
          f"sector sets (Script.r's commented line): {len(by_sector)}")
    print(f"species means: lowest {means[0][0]:.2f} ({means[0][1]}), highest {means[-1][0]:.2f} "
          f"({means[-1][1]}), median {float(np.median([m for m, _ in means])):.2f}")
    combo_means = sorted((float(np.mean(v)), sp, d) for (sp, d), v in by_combo.items())
    print(f"species-by-date means: lowest {combo_means[0][0]:.2f} {combo_means[0][1:]}, highest "
          f"{combo_means[-1][0]:.2f} {combo_means[-1][1:]}")

    print("\nthe mean band, each end widened by 2 SD / sqrt(n) of its own species:")
    lo_m, lo_sp = means[0]
    hi_m, hi_sp = means[-1]
    _, lo_sd, lo_n = sp_stats[lo_sp]
    _, hi_sd, hi_n = sp_stats[hi_sp]
    print(f"  low  {lo_m:.2f} - 2 x {lo_sd:.2f} / sqrt({lo_n}) = "
          f"{lo_m - 2 * lo_sd / math.sqrt(lo_n):.2f}  ({lo_sp})")
    print(f"  high {hi_m:.2f} + 2 x {hi_sd:.2f} / sqrt({hi_n}) = "
          f"{hi_m + 2 * hi_sd / math.sqrt(hi_n):.2f}  ({hi_sp})")

    sec = sorted((stats(np.array(v))[1], key, len(v)) for key, v in by_sector.items())
    print("\nsmallest sector SDs (species, sector, date, n):")
    for sd, key, n in sec[:8]:
        print(f"  SD {sd:6.2f}  {key[0]}, {key[1]}, {key[2]}, n {n}")
    big = [row for row in sec if row[2] >= 75]
    if big:
        sd, key, n = big[0]
        print(f"  smallest with n >= 75: SD {sd:.2f}  {key[0]}, {key[1]}, {key[2]}, n {n}")
    print(f"  sector set sizes: min {min(r[2] for r in sec)}, median "
          f"{int(np.median([r[2] for r in sec]))}, max {max(r[2] for r in sec)}")
    # A whole crown's variance is at least the mean of its thirds' variances, so a whole-canopy
    # set with SD s has a third with SD <= s: a floor above s fails that measured species.
    by_set: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_set[(f"{r['Genus']} {r['Species']}", r["Canopy_sector"], r["Date"])].append(
            float(r["Angle_degree"]))
    every = sorted((stats(np.array(v))[1], key, len(v)) for key, v in by_set.items())
    print(f"smallest SD over all {len(every)} species-sector-date sets, whole canopy included: "
          f"SD {every[0][0]:.2f}  {every[0][1][0]}, {every[0][1][1]}, {every[0][1][2]}, "
          f"n {every[0][2]}")
    print(f"sets with SD under the smallest sector SD ({sec[0][0]:.2f}):")
    for sd, key, n in [row for row in every if row[0] < sec[0][0]]:
        print(f"  SD {sd:6.2f}  {key[0]}, {key[1]}, {key[2]}, n {n}")

    print("\nbeta on t = 2 theta / pi by the method of moments (numpy: rng.beta(a, b)):")
    for sp in ("Cercis siliquastrum", "Liquidambar styraciflua"):
        theta = np.array(by_species[sp])
        m, sd, n = stats(theta)
        a, b = moments_beta(theta)
        print(f"  {sp:<24} mean {m:.2f} SD {sd:.2f} n {n}  a {a:.4f} b {b:.4f}  "
              f"check 90 a/(a+b) = {90 * a / (a + b):.2f}")

    if refs_json is not None:
        refs = {sp: {"mean_deg": round(m, 2), "sd_deg": round(sd, 2), "n": n}
                for sp, (m, sd, n) in sorted(sp_stats.items())}
        refs_json.write_text(json.dumps(refs, indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {len(refs)} species to {refs_json.name}")


# --- main --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", type=Path, help="the Chianucci et al. 2018 dataset_full.csv")
    ap.add_argument("--refs-json", type=Path, help="write per-species mean, SD and n here")
    ap.add_argument("--skip-kept", action="store_true", help="skip the five kept checks")
    args = ap.parse_args()
    t0 = time.perf_counter()
    trees = fixtures()
    for name, sk, upm in trees:
        height = float((sk.pts[:, 1].max() - sk.pts[:, 1].min()) / upm)
        print(f"{name}: {len(sk.limb_id)} limbs, {len(sk.pts)} points, height {height:.1f} m, "
              f"trunk base r {sk.limb_radii(0)[0] / upm:.3f} m")
    print(f"r_rel = {R_REL_M} m (Demol et al. 2022: diameters under 10 cm overestimated)")
    print_taper(trees, R_REL_M)
    print_taper_summary(trees)
    print_leonardo(trees, R_REL_M)
    print_strahler(trees, R_REL_M)
    print_balance(trees)
    if args.csv is not None:
        leaf_angle_tables(args.csv, args.refs_json)
    if not args.skip_kept:
        kept_findings(trees)
    print(f"\ntotal {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
