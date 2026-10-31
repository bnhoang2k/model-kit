import numpy as np
import pytest

from model_kit.core.compare import box_drift, group_boxes, iou, silhouette, surface_gap
from model_kit.core.errors import ManifestError

from .shapes import box

CENTRE = (0.0, 0.0, 0.0)


def _cube():
    return box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))


def _mask(tris, turn=0.0, tilt=0.0, px=200):
    return silhouette(tris, turn_deg=turn, tilt_deg=tilt, px=px, half_extent=1.0, centre=CENTRE)


def test_iou_same_and_apart():
    a = np.zeros((10, 10), dtype=bool)
    a[:5] = True
    b = np.zeros((10, 10), dtype=bool)
    b[5:] = True
    assert iou(a, a) == 1.0
    assert iou(a, b) == 0.0
    with pytest.raises(ManifestError):
        iou(np.zeros((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool))


def test_silhouette_square():
    mask = _mask(_cube())
    assert mask.shape == (200, 200) and mask.dtype == bool
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert abs(mask.sum() - 200 * 200 / 4) <= 2 * 200
    assert abs(len(rows) - 100) <= 1 and abs(len(cols) - 100) <= 1
    # centred, row 0 at the top
    assert abs(rows.mean() - 99.5) <= 1 and abs(cols.mean() - 99.5) <= 1


def test_silhouette_row_zero_is_up():
    tris = box((-0.1, 0.5, -0.1), (0.1, 0.9, 0.1))
    rows = np.flatnonzero(_mask(tris).any(axis=1))
    assert rows.max() < 100


def test_silhouette_turn_45_widens():
    w0 = int(_mask(_cube(), turn=0).any(axis=0).sum())
    w45 = int(_mask(_cube(), turn=45).any(axis=0).sum())
    assert abs(w45 - w0 * np.sqrt(2)) <= 2


def test_silhouette_turn_moves_the_right_side():
    # A box at +X: from the front it is on the image right; turned 90 degrees (the camera
    # moves to +X) it sits in front of the camera and moves to the middle.
    tris = box((0.6, -0.1, -0.1), (0.8, 0.1, 0.1))
    cols0 = np.flatnonzero(_mask(tris, turn=0).any(axis=0))
    cols90 = np.flatnonzero(_mask(tris, turn=90).any(axis=0))
    assert cols0.min() > 100
    assert abs(cols90.mean() - 99.5) <= 2


def test_silhouette_tilt_shrinks_height_of_cube_side():
    flat = _mask(_cube(), tilt=0)
    tilted = _mask(_cube(), tilt=30)
    h_flat = int(flat.any(axis=1).sum())
    h_tilt = int(tilted.any(axis=1).sum())
    assert abs(h_tilt - h_flat * (np.cos(np.radians(30)) + np.sin(np.radians(30)))) <= 2


def test_group_boxes_skip_unlabelled():
    pts = np.array([[0, 0, 0], [1, 2, 3], [9, 9, 9], [-1, 0, 0]], dtype=float)
    boxes = group_boxes(pts, ["a", "a", None, "b"])
    assert set(boxes) == {"a", "b"}
    np.testing.assert_allclose(boxes["a"][0], [0, 0, 0])
    np.testing.assert_allclose(boxes["a"][1], [1, 2, 3])


def test_group_boxes_treat_empty_label_as_no_group():
    pts = np.array([[0, 0, 0], [1, 2, 3], [9, 9, 9], [-1, 0, 0]], dtype=float)
    boxes = group_boxes(pts, ["a", "", None, "b"])
    assert set(boxes) == {"a", "b"}
    np.testing.assert_allclose(boxes["a"][1], [0, 0, 0])
    np.testing.assert_allclose(boxes["b"][0], [-1, 0, 0])


def test_box_drift_and_missing_group():
    pts = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [6, 6, 6]], dtype=float)
    labels = ["a", "a", "b", "b"]
    moved = pts.copy()
    moved[2:] += [0.01, 0, 0]
    drift = box_drift(group_boxes(pts, labels), group_boxes(moved, labels))
    assert drift["a"] == pytest.approx(0.0, abs=1e-12)
    assert drift["b"] == pytest.approx(0.01, abs=1e-12)
    with pytest.raises(ManifestError, match="b"):
        box_drift(group_boxes(pts, labels), group_boxes(pts[:2], labels[:2]))
    with pytest.raises(ManifestError, match="b"):
        box_drift(group_boxes(pts[:2], labels[:2]), group_boxes(pts, labels))


def _mesh_points(tris):
    return np.unique(tris.reshape(-1, 3), axis=0)


def test_surface_gap():
    tris = box((-0.5, -0.5, -0.5), (0.5, 0.5, 0.5))
    pts = _mesh_points(tris)
    p99, mx = surface_gap(pts, tris, pts, tris)
    assert (p99, mx) == (0.0, 0.0)

    up = tris + [0.0, 0.02, 0.0]
    p99, mx = surface_gap(pts, tris, _mesh_points(up), up)
    assert p99 == pytest.approx(0.02, abs=1e-9)
    assert mx == pytest.approx(0.02, abs=1e-9)

    # A finely cut cube, so one stray vertex is a rare outlier among many points.
    fine = _grid_cube(20)
    pulled = fine.copy()
    apex = np.all(np.isclose(pulled, [0.5, 0.0, 0.0]), axis=2)
    assert apex.sum() == 6  # the face centre is a corner of six triangles
    pulled[apex] += [0.1, 0.0, 0.0]
    p99, mx = surface_gap(_mesh_points(fine), fine, _mesh_points(pulled), pulled)
    assert mx == pytest.approx(0.1, abs=1e-9)
    assert p99 < 0.001


def _grid_cube(n):
    """A cube of side 1 about the origin, each face cut into n x n quads (2 triangles each)."""
    g = np.linspace(-0.5, 0.5, n + 1)
    tris = []
    for axis in range(3):
        for side in (-0.5, 0.5):
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = [0.0, 0.0, 0.0]
                        p[axis] = side
                        p[(axis + 1) % 3] = g[i + di]
                        p[(axis + 2) % 3] = g[j + dj]
                        quad.append(p)
                    tris += [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
    return np.array(tris, dtype=float)
