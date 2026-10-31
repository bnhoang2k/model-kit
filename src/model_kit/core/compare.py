"""Compare a rebuilt model with its source: group boxes, silhouettes and the surface gap.

Everything is numpy. Points and triangles are in glTF axes (X right, Y up, Z front).
"""

from collections.abc import Sequence

import numpy as np

from . import geom
from .errors import ManifestError

Vec3 = tuple[float, float, float]
Box = tuple[np.ndarray, np.ndarray]

# A pixel centre this close to a triangle edge counts as inside (float noise only), so two
# triangles that share an edge leave no hairline crack between them.
_EDGE_SLACK = 1e-9
# The share of the pooled gap distances the first number of `surface_gap` covers.
_GAP_PERCENTILE = 99.0


def group_boxes(points: np.ndarray, labels: Sequence[str | None]) -> dict[str, Box]:
    """Axis-aligned (min, max) corners of the points in each labelled group.

    A point whose label is `None` or `""` belongs to no group and is left out.
    """
    points = np.asarray(points, dtype=float)
    if len(points) != len(labels):
        raise ValueError(f"{len(points)} points but {len(labels)} labels")
    names = np.array([label or "" for label in labels], dtype=object)
    keep = np.array([bool(label) for label in labels], dtype=bool)
    out: dict[str, Box] = {}
    for name in dict.fromkeys(label for label in labels if label):
        group = points[keep & (names == name)]
        out[name] = (group.min(axis=0), group.max(axis=0))
    return out


def box_drift(a: dict[str, Box], b: dict[str, Box]) -> dict[str, float]:
    """Per group, the largest difference between the two boxes' corner coordinates."""
    for name in (*a, *b):
        if name not in a or name not in b:
            side = "the second" if name in a else "the first"
            raise ManifestError(f"group {name!r} is missing from {side} model")
    return {name: float(max(np.abs(a[name][i] - b[name][i]).max() for i in (0, 1))) for name in a}


def _camera(turn_deg: float, tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Right and up axes of an orthographic camera in world coordinates.

    At turn 0 and tilt 0 the camera sits on +Z looking toward -Z. The camera turns about +Y
    (positive turn moves it from +Z toward +X), then looks down by the tilt.
    """
    turn, tilt = np.radians(turn_deg), np.radians(tilt_deg)
    toward = np.array(
        [np.sin(turn) * np.cos(tilt), np.sin(tilt), np.cos(turn) * np.cos(tilt)]
    )  # from the centre to the camera
    right = np.array([np.cos(turn), 0.0, -np.sin(turn)])
    return right, np.cross(toward, right)


def silhouette(
    tris: np.ndarray,
    *,
    turn_deg: float,
    tilt_deg: float,
    px: int,
    half_extent: float,
    centre: Vec3,
) -> np.ndarray:
    """A bool (px, px) mask of the triangles seen by an orthographic camera.

    The view covers `half_extent` either side of `centre`, mapped onto the pixels with row 0
    at the top. A pixel is set when its centre lies inside a triangle's projection.
    """
    tris = np.asarray(tris, dtype=float)
    if tris.ndim != 3 or tris.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape (N, 3, 3), got {tris.shape}")
    right, up = _camera(turn_deg, tilt_deg)
    rel = tris - np.asarray(centre, dtype=float)
    scale = px / (2.0 * half_extent)
    # Pixel coordinates of every corner: (N, 3, 2), x to the right and y down.
    xy = np.stack([(rel @ right + half_extent) * scale, (half_extent - rel @ up) * scale], axis=-1)

    mask = np.zeros((px, px), dtype=bool)
    lo = np.floor(xy.min(axis=1) - 0.5).astype(int).clip(0, px)  # first pixel centre inside
    hi = np.ceil(xy.max(axis=1) + 0.5).astype(int).clip(0, px)
    for tri, (x0, y0), (x1, y1) in zip(xy, lo, hi, strict=True):
        if x1 <= x0 or y1 <= y0:
            continue
        (ax, ay), (bx, by), (cx, cy) = tri
        area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
        if area == 0.0:
            continue  # seen edge-on: no area
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((bx - gx) * (cy - gy) - (cx - gx) * (by - gy)) / area
        w1 = ((cx - gx) * (ay - gy) - (ax - gx) * (cy - gy)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -_EDGE_SLACK) & (w1 >= -_EDGE_SLACK) & (w2 >= -_EDGE_SLACK)
        mask[y0:y1, x0:x1] |= inside
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two bool masks."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"masks differ in shape: {a.shape} and {b.shape}")
    union = int((a | b).sum())
    if union == 0:
        raise ManifestError("both silhouettes are empty: there is nothing to compare")
    return float((a & b).sum() / union)


def surface_gap(
    points_a: np.ndarray, tris_a: np.ndarray, points_b: np.ndarray, tris_b: np.ndarray
) -> tuple[float, float]:
    """The 99th percentile and the maximum of the two-sided distance between two surfaces.

    Each side's points are measured to the other side's triangles, and the distances are
    pooled. One stray vertex shows in the maximum but not in the percentile.
    """
    to_b = geom.closest_points(tris_b, points_a).dist
    to_a = geom.closest_points(tris_a, points_b).dist
    pooled = np.concatenate([to_b, to_a])
    if pooled.size == 0:
        raise ManifestError("neither model has points to measure a gap from")
    return float(np.percentile(pooled, _GAP_PERCENTILE)), float(pooled.max())
