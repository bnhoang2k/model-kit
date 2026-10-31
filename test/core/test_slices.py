"""Slices through a joint: the planes, the outlines they cut and the one they keep."""

import math

import numpy as np
import pytest

from model_kit.core.errors import Unmeasurable
from model_kit.core.slices import (
    Slice,
    band_offsets,
    crossing_triangles,
    cut_slices,
    joint_slices,
    ring_loop,
    ring_slices,
)

from .bodies.build import sweep

SEGMENTS = 16
BAND = (0.0, 1.0)
UP = np.array([0.0, 1.0, 0.0])


def polygon_area(radius: float) -> float:
    """The area of the regular polygon `sweep` makes for a round ring."""
    return 0.5 * SEGMENTS * radius**2 * math.sin(2 * math.pi / SEGMENTS)


def polygon_perimeter(radius: float) -> float:
    return SEGMENTS * 2 * radius * math.sin(math.pi / SEGMENTS)


def tube(radius: float, x: float = 0.0, y0: float = 0.0, y1: float = 1.0, rings: int = 11):
    """A closed tube along y: (triangles (T, 3, 3), the vertex y of every corner (T, 3))."""
    stations = [(y, x, 0.0, radius, radius) for y in np.linspace(y0, y1, rings)]
    verts, faces = sweep(stations, "y", caps=("flat", "flat"))
    tris = verts[faces].astype(float)
    return tris, tris[:, :, 1]


def ramp(y, lo=0.25, hi=0.75):
    """A weight that is 0 under `lo`, 1 over `hi` and rises evenly between."""
    return np.clip((y - lo) / (hi - lo), 0.0, 1.0)


def test_cylinder_slice_area():
    tris, y = tube(0.05)
    found = joint_slices(tris, ramp(y), (0, 0.5, 0), UP, BAND, 5)
    assert len(found) == 5 and all(isinstance(s, Slice) for s in found)
    for s in found:
        assert s.area == pytest.approx(polygon_area(0.05), rel=1e-9)
        assert s.perimeter == pytest.approx(polygon_perimeter(0.05), rel=1e-9)
    # the planes lie between the last wholly parent-side ring and the first wholly child-side one
    offsets = [s.offset for s in found]
    assert offsets == pytest.approx(list(np.linspace(-0.3, 0.3, 7)[1:-1]))


def test_band_offsets_need_both_sides():
    tris, y = tube(0.05)
    with pytest.raises(Unmeasurable, match="wholly on one side"):
        band_offsets(tris, np.ones_like(y), (0, 0.5, 0), UP, BAND, 3)
    with pytest.raises(Unmeasurable, match="ahead"):
        band_offsets(tris, ramp(y), (0, 0.5, 0), -UP, BAND, 3)


def test_a_nan_corner_takes_no_part_in_the_band():
    tris, y = tube(0.05)
    weights = ramp(y)
    weights[y < 0.1] = np.nan  # a corner of another bone tree
    a = band_offsets(tris, ramp(y), (0, 0.5, 0), UP, BAND, 3)
    b = band_offsets(tris, weights, (0, 0.5, 0), UP, BAND, 3)
    assert a == pytest.approx(b)


def test_slice_keeps_outline_nearest_the_joint():
    """A fold cuts two segments: the plane meets the limb at the joint and a bent-back stub far
    from it. The slice is the limb's outline, not the stub's."""
    limb, y = tube(0.05)
    stub, _ = tube(0.08, x=0.4, y0=0.3, y1=0.7)
    tris = np.concatenate([limb, stub])
    weights = np.concatenate([ramp(y), np.full(stub.shape[:2], 0.5)])
    found = joint_slices(tris, weights, (0, 0.5, 0), UP, BAND, 3)
    assert [s.area for s in found] == pytest.approx([polygon_area(0.05)] * 3)
    far = joint_slices(tris, weights, (0.4, 0.5, 0), UP, BAND, 3)
    assert [s.area for s in far] == pytest.approx([polygon_area(0.08)] * 3)


def test_a_cut_with_no_closed_outline_raises():
    tris, _ = tube(0.05)
    with pytest.raises(Unmeasurable, match="no closed outline"):
        cut_slices(tris, (0, 0.5, 0), UP, [5.0])  # the plane misses the part
    one = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)  # an open sheet
    with pytest.raises(Unmeasurable, match="no closed outline"):
        cut_slices(one, (0, 0.5, 0), UP, [0.0])


def test_an_oblique_cut_is_longer_and_a_fan_squares_it():
    """A plane tilted 30 degrees off the tube's axis cuts an ellipse 1 / cos(30) larger. With the
    limb running the same way before and after the joint, the fan turns the plane back square."""
    tris, _ = tube(0.05)
    tilt = np.array([math.sin(math.radians(30)), math.cos(math.radians(30)), 0.0])
    oblique = cut_slices(tris, (0, 0.5, 0), tilt, [0.0])[0]
    assert oblique.area == pytest.approx(polygon_area(0.05) / math.cos(math.radians(30)),
                                         rel=0.02)
    square = cut_slices(tris, (0, 0.5, 0), tilt, [0.0], fan=(UP, UP))[0]
    assert square.area == pytest.approx(polygon_area(0.05), rel=1e-9)


def test_a_fan_turns_the_planes_from_one_side_to_the_other():
    """The middle plane of an odd count is the bisecting plane itself."""
    tris, _ = tube(0.05)
    a, b = UP, np.array([math.sin(math.radians(40)), math.cos(math.radians(40)), 0.0])
    bisector = (a + b) / np.linalg.norm(a + b)
    mid = cut_slices(tris, (0, 0.5, 0), bisector, [0.0, 0.0, 0.0], fan=(a, b))[1]
    plain = cut_slices(tris, (0, 0.5, 0), bisector, [0.0])[0]
    assert mid.area == pytest.approx(plain.area, rel=1e-9)
    with pytest.raises(Unmeasurable, match="opposite"):
        cut_slices(tris, (0, 0.5, 0), bisector, [0.0], fan=(UP, -UP))


def test_the_same_edge_gives_the_same_point_whichever_triangle_cuts_it():
    """A plane through a vertex of the mesh still gives one closed outline (no gap or double)."""
    tris, _ = tube(0.05, rings=11)
    found = cut_slices(tris, (0, 0.0, 0), UP, [0.5])[0]  # y = 0.5 is a ring of the tube
    assert found.area == pytest.approx(polygon_area(0.05), rel=1e-9)


# --- rings: the line of equal child-side weight --------------------------------------------------


def test_ring_slices_of_a_cylinder():
    tris, y = tube(0.05)
    found = ring_slices(tris, ramp(y), (0, 0.5, 0), 7)
    assert [s.offset for s in found] == pytest.approx([k / 8 for k in range(1, 8)])
    for s in found:
        assert s.area == pytest.approx(polygon_area(0.05), rel=1e-9)
        assert s.perimeter == pytest.approx(polygon_perimeter(0.05), rel=1e-9)


def test_a_ring_keeps_the_outline_nearest_the_joint():
    limb, y = tube(0.05)
    other, y2 = tube(0.08, x=0.4)
    tris = np.concatenate([limb, other])
    weights = np.concatenate([ramp(y), ramp(y2)])
    near = ring_slices(tris, weights, (0, 0.5, 0), 3)
    assert [s.area for s in near] == pytest.approx([polygon_area(0.05)] * 3)
    far = ring_slices(tris, weights, (0.4, 0.5, 0), 3)
    assert [s.area for s in far] == pytest.approx([polygon_area(0.08)] * 3)


def test_a_ring_is_the_same_ring_wherever_the_part_goes():
    """Turn the whole part: each ring keeps its area and perimeter, however sharply the limb
    bends in the world (a plane cut across it would not)."""
    tris, y = tube(0.05)
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    before = ring_slices(tris, ramp(y), (0, 0.5, 0), 5)
    head = np.array([0.0, 0.5, 0.0]) @ turn.T + 3.0
    after = ring_slices(tris @ turn.T + 3.0, ramp(y), head, 5)
    assert [s.area for s in after] == pytest.approx([s.area for s in before])
    assert [s.perimeter for s in after] == pytest.approx([s.perimeter for s in before])


def test_a_pinched_ring_is_smaller():
    """Pull the middle rings of a tube toward its axis: the ring at the middle weight shrinks by
    the same share, the rings at the ends of the band do not."""
    tris, y = tube(0.05)
    pinched = tris.copy()
    middle = np.isclose(y, 0.5)
    pinched[..., [0, 2]] = np.where(middle[..., None], tris[..., [0, 2]] * 0.5,
                                    tris[..., [0, 2]])
    now = ring_slices(pinched, ramp(y), (0, 0.5, 0), 7)
    then = ring_slices(tris, ramp(y), (0, 0.5, 0), 7)
    assert now[3].perimeter == pytest.approx(0.5 * then[3].perimeter)
    assert now[3].area == pytest.approx(0.25 * then[3].area)
    assert now[0].perimeter > 0.7 * then[0].perimeter


def test_a_ring_that_closes_nowhere_raises():
    tris, y = tube(0.05)
    with pytest.raises(Unmeasurable, match="no closed outline"):
        ring_slices(tris, np.zeros_like(y), (0, 0.5, 0), 3)  # the weight never rises
    one = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=float)  # an open sheet
    with pytest.raises(Unmeasurable, match="no closed outline"):
        ring_loop(one, np.array([[0.0, 1.0, 0.0]]), (0, 0, 0), 0.5)


def test_a_nan_triangle_has_no_ring():
    tris, y = tube(0.05)
    weights = ramp(y)
    weights[y > 0.55] = np.nan
    assert not crossing_triangles(weights, 0.9).any()
    assert len(ring_loop(tris, weights, (0, 0.5, 0), 0.3)) >= SEGMENTS  # the lower half has rings
    with pytest.raises(Unmeasurable, match="no closed outline"):
        ring_loop(tris, weights, (0, 0.5, 0), 0.9)
