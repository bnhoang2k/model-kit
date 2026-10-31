"""The export compare: did a rebuild change a piece, ignoring how Blender ordered it?

Two scene folders hold `pieces/<pack>/<piece>.npz`, one (N, 3, 3) triangle array per part. A
rebuild may number its vertices and triangles differently and still make the same shape, so
each array is put in one canonical form before the compare. Numpy only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.tolerances import COMPARE_TOL

Parts = dict[str, np.ndarray]


def _lower(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row, whether row `a` sorts before row `b`, comparing the first differing value."""
    differs = a != b
    first = differs.argmax(axis=1)
    rows = np.arange(len(a))
    return differs.any(axis=1) & (a[rows, first] < b[rows, first])


def canonical(tris: np.ndarray, tol: float = COMPARE_TOL) -> np.ndarray:
    """The triangles with each one turned to its smallest form, then sorted.

    A triangle has three turns (the same corners in cyclic order, so the winding is kept: a
    flipped triangle stays flipped). On a grid of `tol`, each turn is a row of nine values, and
    the turn with the lexicographically smallest row wins. Choosing by the whole row, not by the
    lowest corner alone, settles a tie when two corners are one grid cell apart (a zero-area
    triangle or a sliver), where input order would otherwise decide. Triangles then sort by the
    same rows (the stable sort keeps input order on a full tie). The result holds the original
    coordinates, not the rounded ones.
    """
    n = len(tris)
    if n == 0:
        return tris
    grid = np.round(tris / tol)
    rows = np.arange(n)[:, None]
    turns = [(np.arange(3) + k) % 3 for k in range(3)]
    best = np.zeros(n, dtype=int)
    best_row = grid[rows, turns[0]].reshape(n, 9)
    for k in (1, 2):
        row = grid[rows, turns[k]].reshape(n, 9)
        better = _lower(row, best_row)
        best, best_row = np.where(better, k, best), np.where(better[:, None], row, best_row)
    order = np.stack(turns)[best]
    tris = tris[rows, order]
    rank = np.lexsort(best_row.T[::-1])
    return tris[rank]


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(canonical(a), canonical(b),
                                                   rtol=0.0, atol=COMPARE_TOL))


def _read(scene: Path) -> dict[str, Parts]:
    """`{"<pack>:<piece>": {part: triangles}}` for every piece file of a scene folder."""
    if not scene.is_dir():
        raise ManifestError(f"{scene}: not a scene folder")
    found: dict[str, Parts] = {}
    for path in sorted((scene / "pieces").glob("*/*.npz")):
        with np.load(path) as data:
            parts = {part: np.asarray(data[part], dtype=float) for part in data.files}
        for part, tris in parts.items():
            if tris.ndim != 3 or tris.shape[1:] != (3, 3):
                raise ManifestError(f"{path}: part {part!r} must be (N, 3, 3) triangles, "
                                    f"got {tris.shape}")
        found[f"{path.parent.name}:{path.stem}"] = parts
    return found


def export_changes(old: Path, new: Path) -> list[str]:
    """What a rebuild changed, as sorted lines, or `[]` when it only reordered.

    Compares `pieces/<pack>/<piece>.npz` of two scene folders part by part. One line for each
    piece or part added, removed or changed: `<pack>:<piece> added`, `<pack>:<piece> removed`,
    `<pack>:<piece> <part> added`, `<pack>:<piece> <part> removed`, `<pack>:<piece> <part>
    changed`. A part is changed when its triangle count differs or any canonical triangle
    differs by more than `COMPARE_TOL`.
    """
    before, after = _read(Path(old)), _read(Path(new))
    lines = []
    for key in sorted(before.keys() | after.keys()):
        if key not in after:
            lines.append(f"{key} removed")
        elif key not in before:
            lines.append(f"{key} added")
        else:
            a, b = before[key], after[key]
            for part in sorted(a.keys() | b.keys()):
                if part not in b:
                    lines.append(f"{key} {part} removed")
                elif part not in a:
                    lines.append(f"{key} {part} added")
                elif not _same(a[part], b[part]):
                    lines.append(f"{key} {part} changed")
    return sorted(lines)
