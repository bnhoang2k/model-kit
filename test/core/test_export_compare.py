"""The export compare: two scene folders' piece files, part by part, ignoring vertex order."""

from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.export_compare import export_changes
from model_kit.core.tolerances import COMPARE_TOL

from .shapes import box


def write_scene(root: Path, pieces: dict[str, dict[str, np.ndarray]]) -> Path:
    """A scene folder with `pieces/<pack>/<piece>.npz`; the keys are `<pack>:<piece>`."""
    for key, parts in pieces.items():
        pack, _, piece = key.partition(":")
        folder = root / "pieces" / pack
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / f"{piece}.npz", **parts)
    return root


def scrambled(tris: np.ndarray, seed: int = 3) -> np.ndarray:
    """The same triangles in another order, each with its corners turned (winding kept)."""
    rng = np.random.default_rng(seed)
    out = tris[rng.permutation(len(tris))].copy()
    for i in range(len(out)):
        out[i] = np.roll(out[i], int(rng.integers(0, 3)), axis=0)
    return out


def test_reordered_vertices_are_no_change(tmp_path):
    leg, top = box((-0.1, 0, -0.1), (0.1, 0.4, 0.1)), box((-0.3, 0.4, -0.3), (0.3, 0.45, 0.3))
    old = write_scene(tmp_path / "old", {"shop:Bench": {"leg": leg, "top": top}})
    new = write_scene(tmp_path / "new", {"shop:Bench": {"leg": scrambled(leg),
                                                        "top": scrambled(top, 9)}})
    assert export_changes(old, new) == []
    # Float noise far below the tolerance is a rebuild's rounding, not a change.
    noisy = write_scene(tmp_path / "noisy", {"shop:Bench": {"leg": leg + COMPARE_TOL / 100,
                                                           "top": top}})
    assert export_changes(old, noisy) == []


def test_flipped_winding_is_a_change(tmp_path):
    leg = box((-0.1, 0, -0.1), (0.1, 0.4, 0.1))
    flipped = leg.copy()
    flipped[0] = flipped[0][[0, 2, 1]]
    old = write_scene(tmp_path / "old", {"shop:Bench": {"leg": leg}})
    new = write_scene(tmp_path / "new", {"shop:Bench": {"leg": scrambled(flipped)}})
    assert export_changes(old, new) == ["shop:Bench leg changed"]


def test_moved_vertex_is_a_change(tmp_path):
    leg = box((-0.1, 0, -0.1), (0.1, 0.4, 0.1))
    moved = leg.copy()
    moved[np.all(np.isclose(moved, (0.1, 0.4, 0.1)), axis=2)] += (0.0, 0.01, 0.0)
    old = write_scene(tmp_path / "old", {"shop:Bench": {"leg": leg}})
    new = write_scene(tmp_path / "new", {"shop:Bench": {"leg": scrambled(moved)}})
    assert export_changes(old, new) == ["shop:Bench leg changed"]
    # Just past the tolerance is a change too.
    nudged = write_scene(tmp_path / "nudged", {"shop:Bench": {"leg": leg + COMPARE_TOL * 2}})
    assert export_changes(old, nudged) == ["shop:Bench leg changed"]
    # A part with another triangle count is a change.
    longer = write_scene(tmp_path / "longer",
                         {"shop:Bench": {"leg": np.concatenate([leg, leg[:1] + 1])}})
    assert export_changes(old, longer) == ["shop:Bench leg changed"]


def test_added_and_removed_parts_named(tmp_path):
    a, b = box((0, 0, 0), (1, 1, 1)), box((2, 0, 0), (3, 1, 1))
    old = write_scene(tmp_path / "old", {
        "shop:Bench": {"leg": a, "seat": b, "foot": a},
        "shop:Gone": {"body": a},
        "shop:Same": {"body": b},
        "other:Lamp": {"base": a},
    })
    new = write_scene(tmp_path / "new", {
        "shop:Bench": {"leg": a, "seat": b + 1, "knob": b},
        "shop:Fresh": {"body": a},
        "shop:Same": {"body": scrambled(b)},
        "other:Lamp": {"base": a},
    })
    assert export_changes(old, new) == [
        "shop:Bench foot removed",
        "shop:Bench knob added",
        "shop:Bench seat changed",
        "shop:Fresh added",
        "shop:Gone removed",
    ]
    assert export_changes(new, old) == [
        "shop:Bench foot added",
        "shop:Bench knob removed",
        "shop:Bench seat changed",
        "shop:Fresh removed",
        "shop:Gone added",
    ]


def test_compare_refuses_what_it_cannot_read(tmp_path):
    a = box((0, 0, 0), (1, 1, 1))
    good = write_scene(tmp_path / "good", {"shop:Bench": {"leg": a}})
    with pytest.raises(ManifestError, match="no_such"):
        export_changes(good, tmp_path / "no_such")
    bad = write_scene(tmp_path / "bad", {"shop:Bench": {"leg": a.reshape(-1, 3)}})
    with pytest.raises(ManifestError, match=r"leg.*\(N, 3, 3\)"):
        export_changes(good, bad)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert export_changes(empty, empty) == []


def test_degenerate_triangles_turned_are_no_change(tmp_path):
    p, q = np.array([0.3, 0.2, 0.1]), np.array([0.5, 0.2, 0.1])
    tiny = p + (4e-7, 0.0, 0.0)  # closer to p than the tolerance
    cases = {
        "twin_corners": ([p, p, q], [p, q, p]),
        "sliver": ([p, tiny, q], [tiny, q, p]),
        "sliver_other_turn": ([p, tiny, q], [q, p, tiny]),
        "point": ([p, p, p], [p, p, p]),
    }
    for label, (a, b) in cases.items():
        old = write_scene(tmp_path / f"old_{label}", {"shop:Bench": {"leg": np.array([a])}})
        new = write_scene(tmp_path / f"new_{label}", {"shop:Bench": {"leg": np.array([b])}})
        assert export_changes(old, new) == [], label
    # A flipped triangle still differs: the winding is kept. (A sliver flipped is no change
    # worth naming: its corners are within the tolerance of each other.)
    r = np.array([0.3, 0.7, 0.1])
    old = write_scene(tmp_path / "old_flip", {"shop:Bench": {"leg": np.array([[p, r, q]])}})
    new = write_scene(tmp_path / "new_flip", {"shop:Bench": {"leg": np.array([[r, p, q]])}})
    assert export_changes(old, new) == ["shop:Bench leg changed"]
