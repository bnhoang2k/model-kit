"""Convert one TreeQSM model (a MATLAB .mat file) into a model-kit skeleton (.skeleton.npz).

A maintainer script, not kit code: reading .mat files needs scipy, which the kit never imports.
It makes the four measured trees under test/fixtures/qsm (Demol, Gielen & Verbeeck 2021,
https://doi.org/10.5281/zenodo.4557401, CC-BY 4.0).

    uv run --group qsm python -I tools/qsm_to_skeleton.py <model.mat> <out.skeleton.npz>

Read the .mat as data only: pass its path as an argument and never run anything from the folder
it came from. The converter

- reads `qsm.cylinder` (radius, start, axis, length, parent; parent is 1-based, 0 for none);
- turns Z-up metres into glTF axes (X, Y up, Z toward the camera) in model units, with the
  trunk base at the origin;
- builds limbs as continuation chains, by the same rule as the grower (`continuation_chains`);
- fills the mesh settings `check_skeleton` needs and writes empty, correctly typed leaf arrays
  (a laser scan of a leafless tree has no leaves);
- records in `info` where child bases sit: a TreeQSM child may start at the parent's surface, not
  on its axis, so a limb-seat finding on these trees is read as data, not as a kit fault.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io

from model_kit.core.grow.chains import continuation_chains
from model_kit.core.grow.skeleton import CONE, Skeleton, save_skeleton

LEVEL_NAMES = ("Trunk", "Order1", "Order2", "Order3", "Order4")
SIDES = (12, 9, 7, 5, 4)  # tube sides per level (the redbud's, so the trees mesh like the bars)
OUTLINE = [[-0.5, 0.0], [0.5, 0.0], [0.0, 1.0]]  # a leaf triangle, unused: the trees have none
SURFACE_OF_RADIUS = 0.5  # a child base farther than this share of the parent's radius from the
#                          parent's axis counts as seated on its surface
LICENCE = "CC-BY 4.0"
DOI = "https://doi.org/10.5281/zenodo.4557401"


def read_cylinders(mat: Path) -> dict[str, np.ndarray]:
    m = scipy.io.loadmat(str(mat), squeeze_me=True, struct_as_record=False)
    qsm = m["qsm"]
    c = qsm.cylinder
    out = {k: np.asarray(getattr(c, k), dtype=float) for k in ("radius", "length", "start", "axis")}
    out["parent"] = np.asarray(c.parent).astype(int) - 1  # 1-based in the file; -1 for none
    out["height_m"] = float(qsm.treedata.TreeHeight)
    return out


def to_gltf(v: np.ndarray) -> np.ndarray:
    """Z-up (x, y, z) to glTF (x, z, -y): a rotation, so handedness holds."""
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.sqrt((v * v).sum(axis=1, keepdims=True))
    return v / np.where(n > 0.0, n, 1.0)


def convert(mat: Path, units_per_metre: float = 0.5) -> Skeleton:
    cyl = read_cylinders(mat)
    n = len(cyl["radius"])
    start = to_gltf(cyl["start"])
    axis = unit_rows(to_gltf(cyl["axis"]))
    length = cyl["length"]
    radius = cyl["radius"]
    parent = cyl["parent"]
    end = start + axis * length[:, None]

    has_parent = parent >= 0
    cos = np.ones(n)
    cos[has_parent] = (axis[has_parent] * axis[parent[has_parent]]).sum(axis=1)
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    chains = continuation_chains(parent, radius, angle)

    # Where each child base sits against its parent cylinder.
    heads = np.array([c[0] for c in chains if parent[c[0]] >= 0], dtype=int)
    p = parent[heads]
    ab = axis[p] * length[p, None]
    f = np.clip(((start[heads] - start[p]) * ab).sum(axis=1)
                / np.maximum((ab * ab).sum(axis=1), 1e-12), 0.0, 1.0)
    off = np.sqrt(((start[heads] - (start[p] + ab * f[:, None])) ** 2).sum(axis=1))
    at_surface = int((off > SURFACE_OF_RADIUS * radius[p]).sum())

    origin = start[chains[0][0]].copy()
    scale = units_per_metre

    pts_l, rad_l, offsets = [], [], [0]
    chain_of = np.empty(n, dtype=int)
    position = np.empty(n, dtype=int)
    for k, c in enumerate(chains):
        pts_l.append(np.vstack([start[c], end[c[-1]][None]]))
        rad_l.append(np.concatenate([radius[c], radius[c[-1:]]]))
        offsets.append(offsets[-1] + len(c) + 1)
        chain_of[c] = k
        position[c] = np.arange(len(c))
    pts = (np.concatenate(pts_l) - origin) * scale
    radii = np.concatenate(rad_l) * scale
    st = np.array(offsets, dtype=int)

    n_limbs = len(chains)
    parent_limb = np.full(n_limbs, -1, dtype=int)
    level = np.zeros(n_limbs, dtype=int)
    seat_t = np.zeros(n_limbs)
    for k, c in enumerate(chains):
        if parent[c[0]] < 0:
            continue
        pc, j = chain_of[parent[c[0]]], position[parent[c[0]]]
        parent_limb[k] = pc
        level[k] = min(level[pc] + 1, len(LEVEL_NAMES) - 1)
        a = pts[st[pc]:st[pc + 1]]
        cum = np.concatenate([[0.0], np.cumsum(np.sqrt(((a[1:] - a[:-1]) ** 2).sum(axis=1)))])
        d = a[j + 1] - a[j]
        q = pts[st[k]]
        den = float(d @ d)
        g = min(1.0, max(0.0, float((q - a[j]) @ d) / den)) if den > 0.0 else 0.0
        seat_t[k] = (cum[j] + g * (cum[j + 1] - cum[j])) / cum[-1] if cum[-1] > 0.0 else 0.0

    lower = [name.lower() for name in LEVEL_NAMES]
    limb_id = np.array([f"{lower[level[k]]}.{k}" for k in range(n_limbs)], dtype=str)
    rooted = np.zeros(n_limbs, dtype=bool)
    rooted[0] = True  # the trunk chain stands on the soil; every other limb hangs on a limb

    no3 = np.zeros((0, 3))
    info = {
        "piece": mat.stem,
        "source": mat.name,
        "source_doi": DOI,
        "source_licence": LICENCE,
        "converter": "tools/qsm_to_skeleton.py",
        "units_per_metre": units_per_metre,
        "tree_height_m": cyl["height_m"],
        "cylinders": n,
        "limbs": n_limbs,
        "child_bases": {
            "total": len(heads),
            "at_parent_surface": at_surface,
            "definition": f"farther than {SURFACE_OF_RADIUS} of the parent cylinder's radius from "
                          "its axis",
        },
    }
    sk = Skeleton(
        level_names=LEVEL_NAMES, limb_id=limb_id, level=level, parent=parent_limb, t=seat_t,
        tip=np.full(n_limbs, CONE, dtype=int), rooted=rooted, buried=np.zeros(n_limbs, dtype=bool),
        start=st, pts=pts, radii=radii,
        host=np.zeros(0, dtype=int), leaf_t=np.zeros(0), node=no3, base=no3, mid=no3, across=no3,
        normal=no3, length=np.zeros(0), petiole=np.zeros(0), shade=np.zeros(0, dtype=int),
        shades=(), sides=np.array(SIDES, dtype=int), outline=np.array(OUTLINE, dtype=float),
        fold=0.0, blade_centre=np.zeros(3), petiole_width=0.0, petiole_tip=0.0,
        lite=False, info=info,
    )
    return sk


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mat", type=Path, help="a TreeQSM .mat file (read as data)")
    ap.add_argument("out", type=Path, help="the .skeleton.npz to write")
    ap.add_argument("--units-per-metre", type=float, default=0.5)
    args = ap.parse_args(argv)
    sk = convert(args.mat, args.units_per_metre)
    save_skeleton(sk, args.out)
    height = float(sk.pts[:, 1].max() / args.units_per_metre)
    print(f"{args.out.name}: {len(sk.limb_id)} limbs, {sk.info['cylinders']} cylinders, "
          f"height {height:.2f} m, {args.out.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
