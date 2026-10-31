"""core/edges.py: interior edges, their turn, the rounding band and the line ends."""

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from model_kit.core import edges as E
from model_kit.core.errors import ManifestError
from model_kit.core.tolerances import CREASE_BAND_CAP_OF_ANGLE

from .shapes import blob, box

WELD = 1e-9
CREASE = 25.0


def revolve(profile, segments=24):
    """A closed surface of revolution about y: `profile` runs bottom to top as (y, radius), the
    first and last radius 0 (a pole). Triangles wound outward."""
    theta = 2 * np.pi * np.arange(segments) / segments
    rings = [np.stack([r * np.cos(theta), np.full(segments, y), r * np.sin(theta)], axis=1)
             for y, r in profile[1:-1]]
    south, north = np.array([0.0, profile[0][0], 0.0]), np.array([0.0, profile[-1][0], 0.0])
    tris = []
    for j in range(segments):
        k = (j + 1) % segments
        tris.append([south, rings[0][k], rings[0][j]])
        tris.append([north, rings[-1][j], rings[-1][k]])
        for a, b in itertools.pairwise(rings):
            tris.append([a[j], a[k], b[k]])
            tris.append([a[j], b[k], b[j]])
    return np.array(tris)


def sphere(rings=12, segments=24):
    phi = np.linspace(0.0, np.pi, rings + 1)
    return revolve([(-np.cos(p), np.sin(p) if 0 < i < rings else 0.0)
                    for i, p in enumerate(phi)], segments)


def roof(height, turn_deg=30.0):
    """Two triangles sharing the edge from (0,0,0) to (1,0,0); each wing is `height` from it. The
    faces turn by `turn_deg` (0 is a flat join). Winding is consistent."""
    half = np.radians(turn_deg / 2.0)
    c, s = height * np.cos(half), height * np.sin(half)
    x3, x4 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    x1, x2 = np.array([0.5, c, s]), np.array([0.5, -c, s])
    return np.array([[x3, x4, x1], [x4, x3, x2]])


def test_turn_of_a_box_edge_is_90():
    found = E.interior_edges(box((0, 0, 0), (1, 2, 3)), WELD)
    assert len(found) == 18  # 12 box edges and the 6 diagonals of the faces
    assert np.sort(np.unique(found.turn_deg.round(6))).tolist() == [0.0, 90.0]
    assert int((found.turn_deg.round(6) == 90.0).sum()) == 12
    assert not found.open_boundary.any()


def test_a_flat_join_turns_nothing_and_a_fold_turns_its_angle():
    flat = E.interior_edges(roof(0.5, 0.0), WELD)
    assert flat.turn_deg == pytest.approx([0.0], abs=1e-9)
    folded = E.interior_edges(roof(0.5, 30.0), WELD)
    assert folded.turn_deg == pytest.approx([30.0], abs=1e-9)


def test_open_edges_and_flipped_neighbours_are_not_interior():
    tris = box((0, 0, 0), (1, 1, 1))
    opened = E.interior_edges(tris[:-1], WELD)
    assert len(opened) < 18 and opened.open_boundary.any()
    flipped = tris.copy()
    flipped[0] = flipped[0][[0, 2, 1]]
    bad = E.interior_edges(flipped, WELD)
    assert len(bad) == 18 - 3 and bad.open_boundary.sum() >= 3  # the flipped row's three edges


def test_edges_number_the_same_way_every_time_and_name_input_rows():
    tris = blob(6, 10)
    one, two = E.interior_edges(tris, WELD), E.interior_edges(tris, WELD)
    assert np.array_equal(one.v, two.v) and np.array_equal(one.faces, two.faces)
    assert one.faces.max() < len(tris) and np.array_equal(one.corners.shape, (len(tris), 3))
    low = np.minimum(one.v[:, 0], one.v[:, 1]) * len(one.points) + np.maximum(one.v[:, 0],
                                                                              one.v[:, 1])
    assert (np.diff(low) > 0).all()


def test_moved_measures_the_same_edges_at_other_points():
    found = E.interior_edges(roof(0.5, 30.0), WELD)
    points = found.points.copy()
    points[found.wings[0, 1]] += [0.0, 0.0, 0.0]
    assert found.moved(points).turn_deg == pytest.approx(found.turn_deg)
    flat = found.points.copy()
    flat[:, 2] = 0.0
    assert found.moved(flat).turn_deg == pytest.approx([0.0], abs=1e-9)


def test_the_gradient_matches_finite_differences():
    """Bridson, Marino and Fedkiw's u1..u4 are the gradient of the turn: each corner's vector
    agrees with a central difference, up to the paper's sign."""
    found = E.interior_edges(blob(8, 12), WELD)
    grads = E._gradient(found.points, found.v, found.wings)
    corners = (found.wings[:, 0], found.wings[:, 1], found.v[:, 0], found.v[:, 1])
    h = 1e-6
    for e in range(0, len(found), 11):
        for u, ids in zip(grads, corners, strict=True):
            for axis in range(3):
                plus, minus = found.points.copy(), found.points.copy()
                plus[ids[e], axis] += h
                minus[ids[e], axis] -= h
                diff = np.radians(E.turns_deg(plus, found.v[e:e + 1], found.wings[e:e + 1])
                                  - E.turns_deg(minus, found.v[e:e + 1], found.wings[e:e + 1]))
                assert abs(u[e, axis]) == pytest.approx(abs(diff[0]) / (2 * h), rel=1e-4,
                                                        abs=1e-6)


def test_band_matches_the_worst_rounding_of_a_mesh_with_no_shared_coordinate():
    """Every coordinate moves by at most half a step. With no two corners sharing a coordinate,
    the band is the largest change the 4096 end-of-range moves can make."""
    x3, x4 = [0.0, 0.0, 0.0], [1.0, 0.013, 0.021]
    x1, x2 = [0.43, 0.31, 0.17], [0.57, -0.29, 0.23]
    tris = np.array([[x3, x4, x1], [x4, x3, x2]])
    found = E.interior_edges(tris, WELD)
    step = np.full(len(found.points), 1e-4)
    band, capped = E.rounding_band_deg(found.points, found, step, CREASE)
    assert not capped.any()
    used = np.unique(np.r_[found.v[0], found.wings[0]])
    assert len(np.unique(found.points[used].round(12), axis=0)) == 4
    for axis in range(3):
        assert len(np.unique(found.points[used, axis].round(12))) == 4
    worst = 0.0
    for signs in itertools.product((-1.0, 1.0), repeat=12):
        moved = found.points.copy()
        moved[used] += np.array(signs).reshape(4, 3) * step[0] / 2.0
        worst = max(worst, abs(float(found.moved(moved).turn_deg[0] - found.turn_deg[0])))
    assert worst <= band[0] * 1.01
    assert worst >= band[0] * 0.9  # the bound is close, not a loose guess


def prism(sides, height=0.4):
    """A closed prism on a regular polygon in x and y, extruded along z, wound outward."""
    a = 2 * np.pi * (np.arange(sides) + 0.5) / sides
    ring = np.stack([np.cos(a), np.sin(a)], axis=1)
    lo, hi = np.c_[ring, np.zeros(sides)], np.c_[ring, np.full(sides, height)]
    tris = []
    for j in range(sides):
        k = (j + 1) % sides
        tris += [[lo[j], lo[k], hi[k]], [lo[j], hi[k], hi[j]],
                 [[0, 0, 0], lo[k], lo[j]], [[0, 0, height], hi[j], hi[k]]]
    return np.array(tris)


def test_a_shared_coordinate_rounds_once_so_the_band_is_no_wider_and_never_beaten():
    """Simulate many quantised builds (a fixed seed, a random grid offset each) of a prism whose
    walls turn 45 deg: corners with one z, or the same x or y, round together. No edge flips
    outside its band, no turn moves further than the band, and the band is never wider than the
    one that rounds every corner alone."""
    crease = 44.0
    found = E.interior_edges(prism(8), WELD)
    step_size = 1.0 / 2**5
    step = np.full(len(found.points), step_size)
    band, capped = E.rounding_band_deg(found.points, found, step, crease)
    assert not capped.any()
    grads = E._gradient(found.points, found.v, found.wings)
    corners = (found.wings[:, 0], found.wings[:, 1], found.v[:, 0], found.v[:, 1])
    alone = np.degrees(sum(np.linalg.norm(u, axis=1) * np.sqrt(3.0) / 2.0 * step[ids]
                           for u, ids in zip(grads, corners, strict=True)))
    assert (band <= alone * (1 + 1e-9)).all()
    assert (band < alone * 0.9).any()  # the prism's shared z really narrows some edges
    inside = np.abs(found.turn_deg - crease) <= band
    assert inside.any() and not inside.all()
    rng = np.random.default_rng(7)
    moved_most = np.zeros(len(found))
    flips = np.zeros(len(found), dtype=bool)
    for _ in range(400):
        offset = rng.random(3) * step_size
        rounded = np.round((found.points - offset) / step_size) * step_size + offset
        turn = found.moved(rounded).turn_deg
        moved_most = np.maximum(moved_most, np.abs(turn - found.turn_deg))
        flips |= (turn > crease) != (found.turn_deg > crease)
    assert flips.any()  # the simulation reaches the edges the band names
    assert not flips[~inside].any()
    assert (moved_most <= band * 1.05).all()


def test_band_grows_on_a_thin_face():
    thick = E.interior_edges(roof(0.5), WELD)
    thin = E.interior_edges(roof(0.005), WELD)
    b_thick, _ = E.rounding_band_deg(thick.points, thick, np.full(4, 1e-6), CREASE)
    b_thin, _ = E.rounding_band_deg(thin.points, thin, np.full(4, 1e-6), CREASE)
    # the gradient is one over a face's height: a face 100 times lower has a band 100 times wider
    assert b_thin[0] / b_thick[0] == pytest.approx(100.0, rel=0.05)


def test_band_capped_and_reported_as_sliver():
    thin = E.interior_edges(roof(1e-4), WELD)
    band, capped = E.rounding_band_deg(thin.points, thin, np.full(len(thin.points), 1e-4), CREASE)
    assert capped.tolist() == [True]
    assert band[0] == pytest.approx(CREASE_BAND_CAP_OF_ANGLE * CREASE)
    ok = E.interior_edges(roof(0.5), WELD)
    _, capped = E.rounding_band_deg(ok.points, ok, np.full(len(ok.points), 1e-6), CREASE)
    assert not capped.any()


def test_draco_step_from_part_box():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.5], [1.0, 0.5, 0.25]])
    piece = SimpleNamespace(measured={"export": {"position_bits": 10}})
    step = E.position_step(piece, points, None)
    assert step.shape == (3,) and step == pytest.approx(2.0 / 2**10)  # the longest side of the box
    wider = E.position_step(SimpleNamespace(measured={"export": {"position_bits": 14}}),
                            points * 2, None)
    assert wider[0] == pytest.approx(4.0 / 2**14)


def test_float32_step_is_the_spacing_at_the_coordinate():
    piece = SimpleNamespace(measured={"export": {"position_bits": None}})
    points = np.array([[1.0, 0.0, 0.0], [1000.0, 1.0, 0.5]])
    step = E.position_step(piece, points, None)
    assert step[0] == pytest.approx(np.spacing(np.float32(1.0)))
    assert step[1] == pytest.approx(np.spacing(np.float32(1000.0)))  # the largest of the three


def test_an_unknown_precision_gives_no_step():
    assert E.position_step(SimpleNamespace(measured={}), np.zeros((3, 3)), None) is None


def test_closed_rim_has_no_dangling_end():
    # a short drum: its rims are closed loops of 90 degree edges, its side steps are 15 degrees
    drum = revolve([(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)], segments=24)
    found = E.interior_edges(drum, WELD)
    drawn = found.turn_deg > CREASE
    assert drawn.sum() == 48
    assert len(E.line_ends(found, drawn, found.open_boundary)) == 0


def test_smooth_sphere_draws_nothing():
    found = E.interior_edges(sphere(), WELD)
    assert found.turn_deg.max() < CREASE
    assert len(E.line_ends(found, found.turn_deg > CREASE, found.open_boundary)) == 0


def test_a_partial_ring_leaves_line_ends():
    found = E.interior_edges(sphere(), WELD)
    drawn = np.zeros(len(found), dtype=bool)
    drawn[[10, 11, 12]] = True  # three edges, some of them joined or not
    ends = E.line_ends(found, drawn, found.open_boundary)
    degree = np.bincount(found.v[drawn].ravel(), minlength=len(found.points))
    assert set(ends.tolist()) == set(np.flatnonzero(degree == 1).tolist()) and len(ends) >= 2


def test_leaf_boundary_is_not_a_line_end():
    # a card folded along its middle: the fold runs rim to rim, so its ends are on the open rim
    h = 0.5
    x = np.array([[0, 0, 0], [0, 1, 0], [h, 0, 0], [h, 1, 0], [2 * h, 0, h], [2 * h, 1, h]],
                 dtype=float)
    tris = x[np.array([[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5]])]
    found = E.interior_edges(tris, WELD)
    drawn = found.turn_deg > 20.0
    assert drawn.sum() >= 1
    assert found.open_boundary.any()
    assert len(E.line_ends(found, drawn, found.open_boundary)) == 0
    # the same drawn edges with the rim not marked open do end there
    assert len(E.line_ends(found, drawn, np.zeros(len(found.points), dtype=bool))) >= 2


# --- sagittas ---------------------------------------------------------------------------------


def tube(n, radius=1.0, length=2.0):
    """A closed n-sided tube about y: n vertical side edges that each turn 360 / n degrees."""
    return revolve([(0.0, 0.0), (0.0, radius), (length, radius), (length, 0.0)], n)


def smooth_sagittas(tris, below=60.0):
    edges = E.interior_edges(tris, WELD)
    s = E.sagittas(tris, edges)
    keep = (edges.turn_deg > 1e-6) & (edges.turn_deg < below)
    return s[keep], edges.turn_deg[keep], s[~keep]


@pytest.mark.parametrize("n", [8, 12, 24, 40])
def test_sagitta_of_a_regular_polygon(n):
    """Each side edge of an n-gon tube has the sagitta of its chord: r (1 - cos(pi / n)). The
    flat joins of the quads' diagonals have none."""
    for r in (1.0, 0.037):
        s, turn, flat = smooth_sagittas(tube(n, radius=r))
        assert len(s) == n and np.allclose(turn, 360.0 / n)
        np.testing.assert_allclose(s, r * (1.0 - np.cos(np.pi / n)), rtol=1e-9)
        assert np.all(flat[flat < 1e-12] == 0.0)


@pytest.mark.parametrize("n", [16, 32, 48])
def test_halving_segments_sagitta_ratio(n):
    """Half the segments double the turn and multiply the sagitta by 4 cos^2(pi / 2n)."""
    fine, _, _ = smooth_sagittas(tube(n))
    coarse, _, _ = smooth_sagittas(tube(n // 2))
    np.testing.assert_allclose(coarse.max() / fine.max(), 4.0 * np.cos(np.pi / (2 * n)) ** 2,
                               rtol=1e-9)


def test_a_sagitta_uses_the_narrower_face_and_a_flat_join_has_none():
    edges = E.interior_edges(roof(1.0, 30.0), WELD)
    s = E.sagittas(roof(1.0, 30.0), edges)
    assert s[0] == pytest.approx(0.5 * np.tan(np.radians(30.0 / 4.0)), rel=1e-9)
    assert E.sagittas(roof(1.0, 0.0), E.interior_edges(roof(1.0, 0.0), WELD))[0] == 0.0
    # the narrower wing sets the width: the second face is half as high
    tris = roof(1.0, 30.0).copy()
    tris[1][:, 1:] *= 0.5
    edges = E.interior_edges(tris, WELD)
    narrow = E.sagittas(tris, edges)[0]
    assert narrow < s[0]
    assert narrow == pytest.approx(0.5 * 0.5 * np.tan(np.radians(edges.turn_deg[0] / 4.0)),
                                   rel=1e-9)


# --- the gap left when an edge is removed, and the float32 step --------------------------------


def kink(a, b, turn_deg):
    """Two triangles on the edge from (0,0,0) to (1,0,0), `a` and `b` high, turning `turn_deg`."""
    half = np.radians(turn_deg / 2.0)
    x3, x4 = np.array([0.0, 0, 0]), np.array([1.0, 0, 0])
    x1 = np.array([0.5, a * np.cos(half), a * np.sin(half)])
    x2 = np.array([0.5, -b * np.cos(half), b * np.sin(half)])
    return np.array([[x3, x4, x1], [x4, x3, x2]])


def test_face_widths_are_the_heights_over_the_edge():
    tris = kink(0.4, 0.1, 20.0)
    edges = E.interior_edges(tris, WELD)
    width = E.face_widths(tris, edges)
    assert sorted(width[0]) == pytest.approx([0.1, 0.4], rel=1e-12)


@pytest.mark.parametrize("n", [16, 32, 48])
def test_the_merged_gap_of_equal_faces_is_the_halving_formula(n):
    """With equal faces the gap left by removing an edge is 4 cos^2(turn / 4) times the sagitta
    (a ring with half its segments), to 1e-9."""
    tris = tube(n)
    edges = E.interior_edges(tris, WELD)
    smooth = (edges.turn_deg > 1e-6) & (edges.turn_deg < 60.0)
    s, gap = E.sagittas(tris, edges)[smooth], E.merged_gaps(tris, edges)[smooth]
    want = 4.0 * np.cos(np.radians(edges.turn_deg[smooth]) / 4.0) ** 2 * s
    np.testing.assert_allclose(gap, want, rtol=1e-9)


def test_the_merged_gap_of_unequal_faces_is_the_height_over_the_joined_chord():
    """Two faces 41.24 and 3.47 mm high, turning 8.55 degrees. Removing the
    edge leaves the old corner 0.476 mm off the long chord, not the 0.258 mm of equal faces."""
    tris = kink(0.04124, 0.00347, 8.55)
    edges = E.interior_edges(tris, WELD)
    (gap,) = E.merged_gaps(tris, edges)
    a, b, t = 41.24, 3.47, np.radians(8.55)
    assert gap * 1000 == pytest.approx(a * b * np.sin(t) / np.sqrt(a * a + b * b + 2 * a * b * np.cos(t)),
                                       rel=1e-9)
    assert gap * 1000 == pytest.approx(0.476, abs=1e-3)
    (s,) = E.sagittas(tris, edges)
    assert 4.0 * np.cos(t / 4.0) ** 2 * s * 1000 == pytest.approx(0.258, abs=1e-3)  # the old model


def test_float32_step_is_the_spacing_at_the_coordinate_and_position_step_agrees():
    points = np.array([[0.0, 1.0, 100.0], [0.5, 0.25, 2.0]])
    step = E.float32_step(points)
    assert step == pytest.approx([np.spacing(np.float32(100.0)), np.spacing(np.float32(2.0))])
    piece = SimpleNamespace(measured={"export": {"position_bits": None}})
    np.testing.assert_array_equal(E.position_step(piece, points), step)


# --- segments_for: the sagitta inequality turned round ------------------------------------------


@pytest.mark.parametrize(("radius", "gap"), [(0.037, 0.0003), (1.0, 0.0029), (0.2, 0.0005)])
def test_segments_for_inverts_the_detail_inequality(radius, gap):
    """The count it gives has no side edge over `gap` in `sagittas` (the numbers detail judges),
    and one side fewer has: it is the least such count, read off real meshes."""
    n = E.segments_for(radius, 360.0, gap)
    fine, _, _ = smooth_sagittas(tube(n, radius))
    coarse, _, _ = smooth_sagittas(tube(n - 1, radius))
    assert 7 <= n and fine.max() <= gap < coarse.max()


@pytest.mark.parametrize("turn", [10.0, 45.0, 90.0, 180.0, 360.0])
def test_segments_for_is_the_least_count_whose_joint_gap_is_under_the_gap(turn):
    """A joint of an arc cut in `count` chords stands `radius (1 - cos(turn / count / 2))` off the
    curve; the count is the least that brings it to the gap, whatever the radius and gap."""
    def joint_gap(radius, count):
        return radius * (1.0 - np.cos(np.radians(turn / count) / 2.0))

    for radius, gap in itertools.product((0.004, 0.3, 12.0), (1e-5, 3e-4, 0.02)):
        n = E.segments_for(radius, turn, gap)
        assert n >= 1 and (n > 1 or turn < 180.0) and (n >= 3 or turn < 360.0)
        assert joint_gap(radius, n) <= gap or radius == 0.0
        assert n == 1 or n == int(turn // 180.0) + 1 or joint_gap(radius, n - 1) > gap


def test_segments_for_gives_the_fewest_count_that_cannot_be_fewer_and_refuses_no_gap():
    assert E.segments_for(0.0, 360.0, 0.001) == 3  # a point has no gap at any count
    assert E.segments_for(1.0, 360.0, 5.0) == 3  # a gap wider than the circle: a triangle
    assert E.segments_for(1.0, 90.0, 5.0) == 1  # and one chord for a quarter turn
    for bad in ((1.0, 360.0, 0.0), (1.0, 360.0, -1.0), (1.0, 0.0, 0.1), (-1.0, 360.0, 0.1)):
        with pytest.raises(ManifestError):
            E.segments_for(*bad)
