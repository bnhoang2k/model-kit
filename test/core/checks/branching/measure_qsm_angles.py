"""The narrowest real forks in the Demol trees: TreeQSM `branch.angle` over all 31 trees (a
maintainer script, not a test; it sets the measured line in `shape.SIBLING_MIN_ANGLE_DEG`'s
comment).

    uv run --group qsm python -I test/core/checks/branching/measure_qsm_angles.py \\
        <optimal_QSMs.zip> [--min-diameter-m 0.10]

The zip is Demol, Gielen & Verbeeck 2021 (https://doi.org/10.5281/zenodo.4557401, CC-BY 4.0):
TreeQSM models of beech (FSYL, 15 trees) and ash (FEXC, 16 trees), with other species beside
them that this script leaves out. Download it into its own new, empty folder and pass its path;
`-I` keeps the working folder off the import path, and the .mat files are read as data from the
zip, never unpacked or run. Reading .mat files needs scipy, which only the `qsm` dev group holds
(the kit never imports it).

Per branch the zip holds `qsm.branch.angle` (degrees between a branch and its parent at the
branching point), `.diameter` (its base diameter, metres) and `.order` (0 is the trunk). A
branch counts when its order is 1 or more and its base diameter is at least `--min-diameter-m`:
a laser-scan diameter under 10 cm is not reliable (Demol et al. 2022). The script prints how many
branches count and the narrowest angles over both species, each with its tree, order and
diameter, then the narrowest per species.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import PurePosixPath

import numpy as np
import scipy.io

SPECIES = ("FEXC", "FSYL")  # ash and beech; the zip's other species are left out
NARROWEST = 5  # how many of the narrowest angles to print


def branches(zip_path: str) -> list[tuple[str, float, float, int]]:
    """(tree, angle in degrees, base diameter in metres, order) for every branch of order 1 or
    more of every FEXC and FSYL tree in the zip."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(z.namelist()):
            tree = PurePosixPath(name).stem
            if not name.endswith(".mat") or not tree.startswith(SPECIES):
                continue
            qsm = scipy.io.loadmat(io.BytesIO(z.read(name)), squeeze_me=True,
                                   struct_as_record=False)["qsm"]
            b = qsm.branch
            angle = np.atleast_1d(np.asarray(b.angle, dtype=float))
            diameter = np.atleast_1d(np.asarray(b.diameter, dtype=float))
            order = np.atleast_1d(np.asarray(b.order, dtype=int))
            rows += [(tree, float(a), float(d), int(o))
                     for a, d, o in zip(angle, diameter, order, strict=True) if o >= 1]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip", help="the Demol et al. 2021 optimal_QSMs.zip")
    ap.add_argument("--min-diameter-m", type=float, default=0.10)
    args = ap.parse_args()
    rows = branches(args.zip)
    trees = sorted({r[0] for r in rows})
    kept = sorted((r for r in rows if r[2] >= args.min_diameter_m), key=lambda r: r[1])
    per = ", ".join(f"{sum(t.startswith(s) for t in trees)} {s}" for s in SPECIES)
    print(f"{len(trees)} trees ({per}); {len(kept)} branches of order 1 or more with a base diameter of "
          f"{args.min_diameter_m:g} m or more")
    print(f"the {NARROWEST} narrowest, both species:")
    for tree, angle, diameter, order in kept[:NARROWEST]:
        print(f"  {angle:6.2f} deg  {tree:<8} order {order}  diameter {diameter:.3f} m")
    for sp in SPECIES:
        mine = [r for r in kept if r[0].startswith(sp)]
        if mine:
            print(f"{sp}: n {len(mine)}, narrowest "
                  + ", ".join(f"{a:.2f} ({t})" for t, a, _, _ in mine[:2]))


if __name__ == "__main__":
    main()
