import math

import numpy as np
import pytest

from model_kit.core import geom
from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.geom import (
    NO_DIRECTION,
    RAYS,
    SurfaceIndex,
    clip_to_box,
    closest_points,
    footprint_distance,
    inside,
    is_closed,
    ray_counts,
    self_crossings,
    self_cuts,
    signed_volume,
    surface_area,
    surface_gap,
    transform,
    tri_tri_cut,
    weld,
)

from .shapes import blob, block, box, cylinder, open_plate, plate, soup, wavy_grid


def test_nearest_above():
    near = closest_points(plate(), np.array([[0.0, 0.004, 0.0]]))
    assert near.dist[0] == pytest.approx(0.002, abs=1e-9)
    assert near.normal[0] == pytest.approx([0, 1, 0])
    assert near.point[0] == pytest.approx([0, 0.002, 0])


def test_nearest_to_edge():
    near = closest_points(plate(), np.array([[0.13, 0.004, 0.0]]))
    assert near.dist[0] == pytest.approx(math.hypot(0.03, 0.002), abs=1e-6)
    assert near.point[0] == pytest.approx([0.1, 0.002, 0.0], abs=1e-9)


def test_inside_press():
    assert inside(plate(), np.array([[0.0, 0.001, 0.0]]))[0]


def test_inside_wrong_side_of_thin_plate():
    assert not inside(plate(), np.array([[0.0, -0.02, 0.0]]))[0]


def test_degenerate_triangle_finite():
    zero_area = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.6, 0.5, 0.5]]])
    point_tri = np.array([[[0.4, 0.4, 0.4]] * 3])
    collinear = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=float)
    pts = np.array([[0.0, 0.004, 0.0], [0.55, 0.6, 0.5], [3.0, 3.0, 3.0]])
    for extra in (zero_area, point_tri, collinear):
        tris = np.concatenate([plate(), extra])
        near = closest_points(tris, pts)
        assert np.isfinite(near.dist).all()
        assert np.isfinite(near.point).all()
        assert np.isfinite(near.normal).all()
    # A lone degenerate triangle is a segment: nearest point is on it.
    near = closest_points(zero_area, np.array([[0.55, 0.6, 0.5]]))
    assert near.dist[0] == pytest.approx(0.1)
    assert near.point[0] == pytest.approx([0.55, 0.5, 0.5])


def test_closed_and_volume():
    assert is_closed(plate())
    assert signed_volume(plate()) == pytest.approx(8e-5)
    assert not is_closed(open_plate())


def test_block_volume_and_outward_normals():
    tris = block()
    assert signed_volume(tris) == pytest.approx(0.2 * 0.2 * 0.08)
    centre = np.array([0.0, 0.04, 0.0])
    near = closest_points(tris, tris.mean(axis=1))
    assert near.dist == pytest.approx(0, abs=1e-12)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    assert ((normals * (tris.mean(axis=1) - centre)).sum(axis=1) > 0).all()


def test_is_closed_rejects_a_missing_or_doubled_face():
    assert not is_closed(plate()[:-1])
    assert not is_closed(np.concatenate([plate(), plate()[:1]]))
    assert not is_closed(np.zeros((0, 3, 3)))


def test_rays_are_unit_and_as_specified():
    assert RAYS.shape == (3, 3)
    assert np.linalg.norm(RAYS, axis=1) == pytest.approx(1.0)
    want = np.array([[1, 0.31, 0.17], [-0.23, 1, 0.41], [0.37, -0.29, 1]])
    want /= np.linalg.norm(want, axis=1, keepdims=True)
    assert RAYS == pytest.approx(want)


def test_ray_counts_count_every_hit():
    # Nested boxes: a point inside both sees 2 hits on a ray out of the shell pair.
    outer = block(0.4, 0.4)
    inner = block(0.1, 0.1) + np.array([0.0, 0.1, 0.0])
    tris = np.concatenate([outer, inner])
    counts = ray_counts(tris, np.array([[0.0, 0.15, 0.0], [0.0, 0.3, 0.0]]), RAYS)
    assert counts[0].tolist() == [2, 2, 2]
    assert counts[1].tolist() == [1, 1, 1]


def test_inside_block_and_outside():
    pts = np.array([[0, 0.04, 0], [0.09, 0.07, -0.09], [0.11, 0.04, 0], [0, 0.1, 0], [0, -1, 0]])
    assert inside(block(), pts).tolist() == [True, True, False, False, False]


def test_inside_survives_a_ray_through_an_edge():
    # From this origin the first ray runs exactly through the diagonal of the +x face, an edge
    # shared by two triangles, so it counts that face twice. The 2-of-3 vote still gets it right.
    box = block(0.2, 0.2)
    origin = np.array([[0.0, 0.1, 0.014]])
    assert ray_counts(box, origin, RAYS)[0].tolist() == [2, 1, 1]
    assert inside(box, origin)[0]


def test_closest_points_vertex_regions_by_hand():
    tri = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    pts = np.array(
        [
            [-1.0, -1, 0],  # vertex A region
            [2, -1, 0],  # vertex B region
            [-1, 2, 0],  # vertex C region
            [0.5, -1, 0],  # edge AB
            [1, 1, 0],  # edge BC, foot at (0.5, 0.5, 0)
            [0.25, 0.25, 2],  # face interior, above
        ]
    )
    near = closest_points(tri, pts)
    assert near.point == pytest.approx(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.25, 0.25, 0]])
    )
    assert near.dist == pytest.approx(
        [math.sqrt(2), math.sqrt(2), math.sqrt(2), 1.0, math.sqrt(0.5), 2.0]
    )
    assert near.normal[-1] == pytest.approx([0, 0, 1])


def test_degenerate_nearest_face_has_zero_normal():
    near = closest_points(np.array([[[0.0, 0, 0], [1, 0, 0], [2, 0, 0]]]), np.array([[0.5, 1, 0]]))
    assert near.normal[0] == pytest.approx([0, 0, 0])


def test_closest_points_matches_brute_force_on_random_input():
    rng = np.random.default_rng(7)
    tris = rng.normal(size=(60, 3, 3))
    pts = rng.normal(size=(80, 3)) * 2
    near = closest_points(tris, pts)
    # Reference: the unpruned pair routine over every point and triangle.
    pi, fi = np.meshgrid(np.arange(80), np.arange(60), indexing="ij")
    q = geom._closest_on_triangles(
        pts[pi.ravel()], tris[fi.ravel(), 0], tris[fi.ravel(), 1], tris[fi.ravel(), 2]
    )
    d = np.linalg.norm(pts[pi.ravel()] - q, axis=1).reshape(80, 60)
    assert near.dist == pytest.approx(d.min(axis=1), abs=1e-12)
    assert near.face.tolist() == d.argmin(axis=1).tolist()
    assert np.linalg.norm(near.normal, axis=1) == pytest.approx(1.0)


def test_chunking_does_not_change_results(monkeypatch):
    rng = np.random.default_rng(3)
    tris = rng.normal(size=(40, 3, 3))
    pts = rng.normal(size=(50, 3))
    whole_n = closest_points(tris, pts)
    whole_c = ray_counts(tris, pts, RAYS)
    monkeypatch.setattr(geom, "PAIR_BUDGET", 97)
    monkeypatch.setattr(geom, "PAIR_BLOCK", 31)
    small_n = closest_points(tris, pts)
    assert small_n.dist == pytest.approx(whole_n.dist, abs=1e-12)
    assert (small_n.face == whole_n.face).all()
    assert (ray_counts(tris, pts, RAYS) == whole_c).all()


def test_transform_applies_a_4x4():
    m = np.eye(4)
    m[:3, 3] = [1, 2, 3]
    m[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    out = transform(np.array([[1.0, 0, 0], [0, 1, 0]]), m)
    assert out == pytest.approx(np.array([[1, 3, 3], [0, 2, 3]]))
    tri_out = transform(plate(), np.eye(4))
    assert tri_out.shape == (12, 3, 3)


def test_empty_points():
    near = closest_points(plate(), np.zeros((0, 3)))
    assert near.dist.shape == (0,) and near.normal.shape == (0, 3)
    assert inside(plate(), np.zeros((0, 3))).shape == (0,)


UP = np.array([0.0, 1.0, 0.0])


def test_footprint_inside_is_zero():
    # Plate top is x, z in +-0.1; any height, any face of the closed box.
    pts = [[0.0, 5.0, 0.0], [0.05, -3.0, -0.07], [0.1, 0.001, 0.0], [0.1, 0.0, 0.1]]
    assert footprint_distance(plate(), UP, pts) == pytest.approx([0, 0, 0, 0], abs=1e-12)


def test_footprint_outside_an_edge_and_a_corner():
    pts = [[0.13, 0.4, 0.0], [0.13, -0.2, 0.14], [0.1 + 1e-6, 0.0, 0.0]]
    got = footprint_distance(plate(), UP, pts)
    assert got == pytest.approx([0.03, math.hypot(0.03, 0.04), 1e-6], rel=1e-9, abs=1e-15)


def test_footprint_of_a_concave_union_and_a_tilted_n():
    # An L of two squares: (0..1, 0..1) and (1..2, 0..1) and (0..1, 1..2), in the x-z plane.
    def square(x0, z0):
        a, b, c, d = ((x0, 0, z0), (x0 + 1, 0, z0), (x0 + 1, 0, z0 + 1), (x0, 0, z0 + 1))
        return [(a, b, c), (a, c, d)]
    ell = np.array(square(0, 0) + square(1, 0) + square(0, 1), dtype=float)
    pts = [[1.5, 0.0, 1.5], [1.5, 0.0, 1.2], [0.5, 9.0, 1.5], [1.0, 0.0, 1.0], [1.3, 0.0, 1.4]]
    got = footprint_distance(ell, UP, pts)
    assert got == pytest.approx([0.5, 0.2, 0.0, 0.0, 0.3], abs=1e-12)
    # Seen along a tilted n, the unit square's footprint shrinks along z by cos(tilt).
    tilt = np.radians(30.0)
    n = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    unit = np.array(square(0, 0), dtype=float)
    far = np.array([[0.5, 0.0, 2.0]])  # 1 past the square's edge along z
    assert footprint_distance(unit, n, far)[0] == pytest.approx(np.cos(tilt))


def test_footprint_edge_on_triangle_is_a_segment():
    wall = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])  # in the x-y plane
    got = footprint_distance(wall, UP, [[0.5, 3.0, 0.0], [0.5, 0.0, 0.2], [1.5, 0.0, 0.0]])
    assert got == pytest.approx([0.0, 0.2, 0.5], abs=1e-12)


def test_footprint_rejects_a_zero_n():
    with pytest.raises(ValueError, match="non-zero"):
        footprint_distance(plate(), [0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]])


def test_footprint_runs_in_chunks():
    pts = np.random.default_rng(3).uniform(-0.3, 0.3, size=(geom.PAIR_BLOCK // 12 + 7, 3))
    got = footprint_distance(plate(), UP, pts)
    off = np.maximum(np.abs(pts[:, [0, 2]]) - 0.1, 0.0)
    assert got == pytest.approx(np.hypot(off[:, 0], off[:, 1]), abs=1e-12)


def test_footprint_edge_on_triangle_under_random_rotations():
    # Rotated off the axes, an edge-on triangle's projected area is float noise, not 0.
    # A point on the segment's line 1.0 past its end must still read 1.0.
    wall = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    point = np.array([[2.0, 0.3, 0.0]])
    rng = np.random.default_rng(20261019)
    got = []
    for _ in range(2000):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        rot = q * np.sign(np.diag(r))
        got.append(footprint_distance(wall @ rot.T, rot @ UP, point @ rot.T)[0])
    assert np.asarray(got) == pytest.approx(np.ones(2000), abs=1e-9)


def _areas(tris):
    return 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)


def test_clip_to_box_leaves_no_zero_area_triangles():
    # One corner sits on the box's x = 0 bound and its neighbour lies outside.
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]])
    cut = clip_to_box(tri, ((0.0, -1.0, -1.0), (2.0, 1.0, 2.0)))
    assert len(cut) >= 1 and (_areas(cut) > 0).all()
    assert _areas(cut).sum() == pytest.approx(0.5)
    n_in = np.cross(tri[0, 1] - tri[0, 0], tri[0, 2] - tri[0, 0])
    n_out = np.cross(cut[:, 1] - cut[:, 0], cut[:, 2] - cut[:, 0])
    assert (n_out @ n_in > 0).all()  # winding kept
    # A box whose floor is the top face's height keeps the top and no side slivers.
    top = clip_to_box(block(), ((-0.2, 0.08, -0.2), (0.2, 0.2, 0.2)))
    assert (_areas(top) > 0).all() and np.allclose(top[:, :, 1], 0.08)


def test_the_float_guards_live_here():
    # One no-direction guard for every check: a vector this short has no direction.
    assert NO_DIRECTION == 1e-9
    assert geom.DEGENERATE_SHARE == 1e-12
    sliver = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    assert len(geom.drop_slivers(np.concatenate([plate(), sliver]))) == len(plate())


def test_face_distances_give_every_face_its_distance_and_normal():
    slab = plate(0.4, 0.02)  # x and z in +-0.2, y in 0..0.02
    inside_middle = np.array([0.05, 0.01, 0.03])
    dist, normals = geom.face_distances(slab, inside_middle)
    assert dist.shape == (len(slab),) and normals.shape == (len(slab), 3)
    assert dist.min() == pytest.approx(0.01)
    tied = normals[dist <= dist.min() + geom.TIE_DISTANCE]
    assert sorted(tied[:, 1].tolist()) == [-1.0, 1.0]  # the nearest top and bottom triangles
    assert geom.TIE_DISTANCE == 1e-9  # float noise, like NO_DIRECTION


def test_face_distances_agree_with_the_index_on_the_nearest_distance():
    tris = soup(60, 0.3, seed=5)
    p = np.array([0.1, -0.2, 0.3])
    dist, _ = geom.face_distances(tris, p)
    assert dist.min() == pytest.approx(float(closest_points(tris, p[None]).dist[0]), abs=1e-12)


# --- the leaf index ---------------------------------------------------------------------------


def _lowest_tie_reference(tris, pts):
    """Per point: the lowest face index among the triangles at exactly the least distance,
    and the nearest point on that face.

    The dense path takes the highest index at an exact tie, so faces and points compare
    against this. At a tie the nearest points differ between faces, so the dense point is no
    reference there.
    """
    tris = np.asarray(tris)
    faces, points = [], []
    for p in pts:
        q = geom._closest_on_triangles(np.tile(p, (len(tris), 1)), tris[:, 0], tris[:, 1],
                                       tris[:, 2])
        d2 = geom._dot(p - q, p - q)  # the arithmetic the index itself uses
        faces.append(int(np.flatnonzero(d2 == d2.min())[0]))
        points.append(q[faces[-1]])
    return np.array(faces), np.array(points)


@pytest.mark.parametrize(("n", "size"), [(10, 1.0), (300, 1.0), (300, 0.1), (6000, 0.08)])
def test_index_matches_dense(n, size):
    tris = soup(n, size, seed=n)
    pts = np.random.default_rng(5).uniform(-1.5, 1.5, size=(2000, 3))
    fast = SurfaceIndex(tris).closest(pts)
    dense = geom._closest_dense(tris, pts)
    assert fast.dist == pytest.approx(dense.dist, abs=1e-12)
    assert fast.point == pytest.approx(dense.point, abs=1e-12)
    assert fast.normal == pytest.approx(dense.normal, abs=1e-12)
    # Random soups have no exact ties, so the dense faces agree too.
    assert (fast.face == dense.face).all()


@pytest.mark.parametrize("shape", ["plate", "block"])
def test_index_matches_dense_where_exact_ties_are_common(shape):
    tris = {"plate": plate(), "block": block()}[shape]
    # A lattice that hits quad diagonals, edges and corners, so equal distances happen.
    axis = np.linspace(-0.15, 0.15, 13)
    pts = np.stack(np.meshgrid(axis, np.linspace(-0.05, 0.12, 8), axis, indexing="ij"),
                   axis=-1).reshape(-1, 3)
    fast = SurfaceIndex(tris).closest(pts)
    dense = geom._closest_dense(tris, pts)
    faces, points = _lowest_tie_reference(tris, pts)
    assert fast.dist == pytest.approx(dense.dist, abs=1e-12)
    assert fast.point == pytest.approx(points, abs=1e-12)
    assert (fast.face == faces).all()
    assert (fast.face != dense.face).any()  # the lattice does hit exact ties


def test_closest_points_builds_a_one_off_index_and_gives_the_dense_distances():
    tris = blob()
    pts = np.random.default_rng(2).uniform(-1.5, 1.5, size=(400, 3))
    near = closest_points(tris, pts)
    dense = geom._closest_dense(tris, pts)
    faces, points = _lowest_tie_reference(tris, pts)
    assert near.dist == pytest.approx(dense.dist, abs=1e-12)
    assert near.point == pytest.approx(points, abs=1e-12)
    assert (near.face == faces).all()


def test_non_finite_input_raises_a_named_error():
    index = SurfaceIndex(plate())
    bad = np.array([[0.0, 0.1, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="points hold a non-finite value"):
        index.closest(bad)
    with pytest.raises(ValueError, match="points hold a non-finite value"):
        index.inside(bad)
    with pytest.raises(ValueError, match="points hold a non-finite value"):
        closest_points(plate(), bad)
    with pytest.raises(ValueError, match="points hold a non-finite value"):
        inside(plate(), [[np.inf, 0.0, 0.0]])
    broken = plate().copy()
    broken[3, 1, 2] = np.nan
    with pytest.raises(ValueError, match="triangles hold a non-finite value"):
        SurfaceIndex(broken)


@pytest.mark.parametrize("shape", ["plate", "block", "blob"])
def test_index_inside_matches_dense(shape):
    tris = {"plate": plate(), "block": block(), "blob": blob()}[shape]
    rng = np.random.default_rng(11)
    reach = {"plate": 0.15, "block": 0.15, "blob": 1.6}[shape]
    pts = rng.uniform(-reach, reach, size=(3000, 3))
    if shape != "blob":
        pts[:, 1] = rng.uniform(-0.02, 0.1, size=3000)
    dense = (ray_counts(tris, pts, RAYS) % 2 == 1).sum(axis=1) >= 2
    got = SurfaceIndex(tris).inside(pts)
    assert dense.any() and not dense.all()
    assert (got == dense).all()
    assert (inside(tris, pts) == dense).all()


def test_index_prunes():
    tris = wavy_grid(50, 60)
    assert len(tris) == 6000
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(0, 1, 2000), rng.uniform(-0.03, 0.08, 2000),
                           rng.uniform(0, 1, 2000)])
    index = SurfaceIndex(tris)
    near = index.closest(pts)
    ratio = index.last_pairs / (len(pts) * len(tris))
    print(f"prune ratio {ratio:.5f} ({index.last_pairs} of {len(pts) * len(tris)} pairs)")
    assert ratio < 0.10
    assert near.dist == pytest.approx(geom._closest_dense(tris, pts).dist, abs=1e-12)
    # The inside query prunes too: a slab test keeps the rays off most leaves.
    ball = blob(40, 80)
    ib = SurfaceIndex(ball)
    ib.inside(pts * 2 - 1)
    assert ib.last_pairs < 0.10 * len(pts) * 3 * len(ball)


def test_index_breaks_ties_by_the_lowest_original_face():
    tris = soup(100, 0.1, seed=9)
    tris[90] = tris[5]  # the same triangle twice: equal distance from every point
    tris[40] = tris[5] + [0.0, 0.0, 0.0]
    pts = tris[5].mean(axis=0) + np.random.default_rng(1).normal(size=(50, 3)) * 0.01
    got = SurfaceIndex(tris).closest(pts)
    twins = np.isin(got.face, [5, 40, 90])
    assert twins.any()
    assert (got.face[twins] == 5).all()


def test_index_leaf_boundaries_do_not_change_answers(monkeypatch):
    tris = soup(500, 0.15, seed=3)
    pts = np.random.default_rng(6).uniform(-1.5, 1.5, size=(300, 3))
    whole = SurfaceIndex(tris).closest(pts)
    monkeypatch.setattr(geom, "LEAF_SIZE", 7)
    monkeypatch.setattr(geom, "PAIR_BUDGET", 97)
    monkeypatch.setattr(geom, "PAIR_BLOCK", 31)
    small = SurfaceIndex(tris).closest(pts)
    assert (small.face == whole.face).all()
    assert small.dist == pytest.approx(whole.dist, abs=1e-12)
    ball = blob()
    dense = (ray_counts(ball, pts, RAYS) % 2 == 1).sum(axis=1) >= 2
    assert (SurfaceIndex(ball).inside(pts) == dense).all()


def test_index_needs_a_triangle_and_takes_no_points():
    with pytest.raises(ValueError, match="at least one triangle"):
        SurfaceIndex(np.zeros((0, 3, 3)))
    index = SurfaceIndex(plate())
    assert index.closest(np.zeros((0, 3))).dist.shape == (0,)
    assert index.inside(np.zeros((0, 3))).shape == (0,)


# --- tri_tri_cut ------------------------------------------------------------------------------


def _edge_dist(p, tri):
    best = math.inf
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        ab = b - a
        t = 0.0 if ab @ ab < 1e-18 else min(1.0, max(0.0, (p - a) @ ab / (ab @ ab)))
        best = min(best, float(np.linalg.norm(a + ab * t - p)))
    return best


def _interval(tri, n, p0, d):
    s = [(v - p0) @ n for v in tri]
    if all(x > 1e-9 for x in s) or all(x < -1e-9 for x in s):
        return None
    pts = [tri[i] for i in range(3) if abs(s[i]) <= 1e-9]
    for i in range(3):
        j = (i + 1) % 3
        if (s[i] > 1e-9 and s[j] < -1e-9) or (s[i] < -1e-9 and s[j] > 1e-9):
            pts.append(tri[i] + (tri[j] - tri[i]) * (s[i] / (s[i] - s[j])))
    if not pts:
        return None
    ts = [q @ d for q in pts]
    return min(ts), max(ts), pts[ts.index(min(ts))], pts[ts.index(max(ts))]


def reference_cut(va, vb, graze):
    """One triangle pair, one point at a time: a plain per-point reference for the cut."""
    na = np.cross(va[1] - va[0], va[2] - va[0])
    nb = np.cross(vb[1] - vb[0], vb[2] - vb[0])
    if np.linalg.norm(na) < 1e-14 or np.linalg.norm(nb) < 1e-14:
        return 0.0
    na, nb = na / np.linalg.norm(na), nb / np.linalg.norm(nb)
    d = np.cross(na, nb)
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    d = d / np.linalg.norm(d)
    ia, ib = _interval(va, nb, vb[0], d), _interval(vb, na, va[0], d)
    if ia is None or ib is None:
        return 0.0
    reach_a = max(0.0, -min((v - vb[0]) @ nb for v in va))
    reach_b = max(0.0, -min((v - va[0]) @ na for v in vb))
    if min(reach_a, reach_b) <= graze:
        return 0.0
    t0, t1 = max(ia[0], ib[0]), min(ia[1], ib[1])
    if t1 < t0:
        return 0.0
    span = ia[1] - ia[0]
    best = 0.0
    for k in range(5):
        t = t0 + (t1 - t0) * k / 4
        p = ia[2] + (ia[3] - ia[2]) * ((t - ia[0]) / span) if span > 1e-12 else ia[2]
        best = max(best, min(_edge_dist(p, va), _edge_dist(p, vb)))
    return best


def reference_soup_cut(a, b, graze):
    return max((reference_cut(x, y, graze) for x in a for y in b), default=0.0)


def sunk_pair(depth):
    """A 1 m cube and a post whose foot is `depth` below the cube's top face."""
    return box((0, 0, 0), (1, 1, 1)), box((0.2, 1 - depth, 0.2), (0.8, 2 - depth, 0.8))


def test_a_sunk_post_cuts_about_its_depth():
    cube, post = sunk_pair(0.1)
    cut = tri_tri_cut(cube, post)
    assert 0.05 < cut <= 0.1 + 1e-9


def test_cut_deepens_with_the_sink():
    cuts = [tri_tri_cut(*sunk_pair(depth)) for depth in (0.01, 0.02, 0.05, 0.1)]
    assert cuts == sorted(cuts) and cuts[0] > 0.0


def test_flush_and_separate_soups_cut_nothing():
    cube = box((0, 0, 0), (1, 1, 1))
    assert tri_tri_cut(cube, box((0.2, 1, 0.2), (0.8, 2, 0.8))) == 0.0  # standing on the top
    assert tri_tri_cut(cube, box((2, 0, 0), (3, 1, 1))) == 0.0  # apart
    assert tri_tri_cut(cube, box((1, 0, 0), (2, 1, 1))) == 0.0  # face to face


def test_cut_is_the_same_both_ways_round():
    cube, post = sunk_pair(0.07)
    assert tri_tri_cut(cube, post) == tri_tri_cut(post, cube)


def test_a_graze_inside_the_limit_is_not_a_cut():
    # Each triangle must reach past the other's face by more than `graze`: a post sunk 0.01
    # grazes at 0.02 and cuts at 0.005.
    cube, post = sunk_pair(0.01)
    assert tri_tri_cut(cube, post, graze=0.02) == 0.0
    assert tri_tri_cut(cube, post, graze=0.005) > 0.0


def test_tri_tri_cut_takes_surface_indexes_as_well_as_arrays():
    cube, post = sunk_pair(0.1)
    assert tri_tri_cut(SurfaceIndex(cube), SurfaceIndex(post)) == tri_tri_cut(cube, post)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_cut_matches_the_one_pair_at_a_time_reference(seed):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.5, 0.5, size=(14, 3, 3))
    b = rng.uniform(-0.5, 0.5, size=(11, 3, 3))
    want = reference_soup_cut(a, b, 0.0)
    assert want > 0.0
    assert tri_tri_cut(a, b) == pytest.approx(want, abs=1e-12)
    assert tri_tri_cut(a, b, graze=0.05) == pytest.approx(reference_soup_cut(a, b, 0.05),
                                                           abs=1e-12)


def test_cut_matches_the_reference_on_sunk_boxes():
    for depth in (0.02, 0.1, 0.3):
        cube, post = sunk_pair(depth)
        assert tri_tri_cut(cube, post) == pytest.approx(reference_soup_cut(cube, post, 0.0),
                                                         abs=1e-12)


def test_cut_does_not_depend_on_how_the_leaves_fall(monkeypatch):
    cube, post = sunk_pair(0.1)
    whole = tri_tri_cut(cube, post)
    monkeypatch.setattr(geom, "LEAF_SIZE", 3)
    monkeypatch.setattr(geom, "PAIR_BLOCK", 17)
    assert tri_tri_cut(cube, post) == whole


def test_cut_of_nothing_raises():
    with pytest.raises(ValueError, match="at least one triangle"):
        tri_tri_cut(np.zeros((0, 3, 3)), plate())


# --- self_crossings: faces of one soup that cut each other -----------------------------------


def post_through_cube(depth: float = 0.5) -> np.ndarray:
    """A unit cube and a post driven `depth` into its top, as one soup: cube faces 0..11, post 12..23."""
    return np.concatenate([box((0, 0, 0), (1, 1, 1)),
                           box((0.4, 1 - depth, 0.4), (0.6, 1.5, 0.6))])


def test_a_closed_shell_that_does_not_fold_has_no_self_crossing():
    assert len(self_crossings(box((0, 0, 0), (1, 1, 1)))) == 0
    assert len(self_crossings(blob())) == 0


def test_faces_that_cut_each_other_and_share_no_vertex_are_pairs():
    pairs = self_crossings(post_through_cube())
    assert len(pairs) > 0
    assert (pairs[:, 0] < 12).all() and (pairs[:, 1] >= 12).all()  # cube faces x post faces
    assert (pairs[:, 0] < pairs[:, 1]).all()
    assert len({tuple(p) for p in pairs}) == len(pairs)  # each pair once


def test_a_pair_that_shares_a_vertex_is_not_a_crossing():
    # Two triangles sharing a corner that pierce each other: not a pair, by definition.
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    b = np.array([[0.2, 0.2, -0.5], [0.2, 0.2, 0.5], [0.0, 0.0, 0.0]], float)
    assert len(self_crossings(np.stack([a, b]))) == 0
    c = b + np.array([0.0, 0.0, 0.0])
    c[2] = [0.9, 0.9, 0.0]  # the same, but no shared corner now
    assert len(self_crossings(np.stack([a, c]))) == 1


def test_ids_say_which_corners_are_one_vertex():
    cube = post_through_cube()
    ids = np.arange(len(cube) * 3).reshape(-1, 3)
    every = len(self_crossings(cube, ids=ids))
    assert every == len(self_crossings(cube))  # no shared corners either way
    assert len(self_crossings(cube, ids=np.zeros_like(ids))) == 0  # one vertex: all share it
    with pytest.raises(ValueError, match="ids must be"):
        self_crossings(cube, ids=ids[:5])


def test_faces_that_moved_together_are_not_paired():
    """`together` names faces that moved as one body: a pair of them stands as it did."""
    cube = post_through_cube()
    want = {tuple(p) for p in self_crossings(cube)}
    apart = np.array([0] * 12 + [1] * 12)
    assert {tuple(p) for p in self_crossings(cube, together=apart)} == want
    assert len(self_crossings(cube, together=np.zeros(24, dtype=int))) == 0  # one body
    assert {tuple(p) for p in self_crossings(cube, together=-np.ones(24, dtype=int))} == want
    with pytest.raises(ValueError, match="together must be"):
        self_crossings(cube, together=apart[:5])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_together_only_drops_the_pairs_inside_one_body(seed):
    rng = np.random.default_rng(seed)
    tris = rng.uniform(-0.5, 0.5, size=(60, 3, 3))
    every = self_cuts(tris)
    assert len(every[0]) > 10
    label = rng.integers(-1, 3, size=60)
    pairs, depths = self_cuts(tris, together=label)
    same = (label[every[0][:, 0]] == label[every[0][:, 1]]) & (label[every[0][:, 0]] >= 0)
    assert (pairs == every[0][~same]).all() and np.allclose(depths, every[1][~same])


def test_graze_keeps_a_shallow_crossing_out():
    deep = post_through_cube(0.5)
    assert len(self_crossings(deep, graze=0.01)) > 0
    pairs, depths = self_cuts(deep)
    assert len(pairs) == len(depths) and (depths > 0).all()
    assert len(self_crossings(deep, graze=float(depths.max()) + 1e-9)) == 0
    assert len(self_crossings(post_through_cube(0.01), graze=0.02)) == 0


def test_a_vertex_pushed_through_the_far_wall_of_a_blob_makes_crossings():
    tris = blob()
    assert len(self_crossings(tris)) == 0
    top = tris.reshape(-1, 3)[:, 1].argmax()
    moved = tris.copy().reshape(-1, 3)
    moved[np.all(tris.reshape(-1, 3) == tris.reshape(-1, 3)[top], axis=1)] = [0.0, -1.6, 0.0]
    assert len(self_crossings(moved.reshape(-1, 3, 3))) > 0


def test_self_crossings_of_nothing_or_one_face_is_empty():
    assert self_crossings(np.zeros((0, 3, 3))).shape == (0, 2)
    assert self_crossings(plate()[:1]).shape == (0, 2)


# --- surface_gap ------------------------------------------------------------------------------


def test_surface_gap_between_two_faces():
    a = box((0, 0, 0), (1, 1, 1))
    assert surface_gap(a, box((0, 1.25, 0), (1, 2, 1)), within=1.0) == pytest.approx(0.25)
    assert surface_gap(a, box((0, 1, 0), (1, 2, 1)), within=1.0) == pytest.approx(0.0, abs=1e-12)


def test_surface_gap_is_zero_for_a_cut_and_for_a_part_inside_the_other():
    cube, post = sunk_pair(0.1)
    assert surface_gap(cube, post, within=1.0) == 0.0
    assert surface_gap(cube, box((0.4, 0.4, 0.4), (0.6, 0.6, 0.6)), within=1.0) > 0.0  # inside


def test_surface_gap_sees_two_skew_edges_that_no_corner_is_near():
    # Two sticks cross at right angles, 0.05 apart: the nearest points are mid-edge.
    along_x = box((-1, 0, -0.01), (1, 0.01, 0.01))
    along_z = box((-0.01, 0.06, -1), (0.01, 0.07, 1))
    assert surface_gap(along_x, along_z, within=1.0) == pytest.approx(0.05)


def test_surface_gap_past_within_is_not_exact_but_is_over_within():
    a = box((0, 0, 0), (1, 1, 1))
    far = box((0, 5, 0), (1, 6, 1))
    assert surface_gap(a, far, within=0.5) > 0.5


def test_surface_gap_agrees_with_the_nearest_point_query():
    rng = np.random.default_rng(4)
    a = box((0, 0, 0), (1, 1, 1))
    for _ in range(5):
        lo = rng.uniform(1.05, 1.4, size=3) * [1, 0.2, 0.2]
        b = box(lo, lo + [0.3, 0.3, 0.3])
        nearest = closest_points(a, b.reshape(-1, 3)).dist.min()
        assert surface_gap(a, b, within=1.0) <= nearest + 1e-12


# --- weld -------------------------------------------------------------------------------------


def test_weld_merges_the_repeated_corners_of_a_soup():
    verts, faces = weld(box((0, 0, 0), (1, 1, 1)), 1e-3)
    assert verts.shape == (8, 3) and faces.shape == (12, 3)
    assert (verts[faces] == box((0, 0, 0), (1, 1, 1))).all()


def test_weld_snaps_corners_a_little_apart_and_keeps_those_far_apart():
    tris = box((0, 0, 0), (1, 1, 1))
    near = tris + np.random.default_rng(2).uniform(-1e-4, 1e-4, size=tris.shape)
    assert len(weld(near, 1e-3)[0]) == 8
    assert len(weld(near, 1e-6)[0]) > 8


def test_weld_keeps_every_triangle_even_one_that_collapses():
    tris = np.array([[[0, 0, 0], [1e-5, 0, 0], [0, 1, 0]], [[0, 0, 0], [1, 0, 0], [0, 1, 0]]])
    verts, faces = weld(tris, 1e-3)
    assert faces.shape == (2, 3)
    assert faces[0, 0] == faces[0, 1] and len(set(faces[1])) == 3
    assert len(verts) == 3


def test_weld_needs_a_positive_tolerance():
    with pytest.raises(ValueError, match="above 0"):
        weld(plate(), 0.0)


def test_weld_merges_corners_that_straddle_a_cell_edge():
    tol = 1e-3
    tris = np.array([[[0.45 * tol, 0, 0], [1, 0, 0], [0, 1, 0]],
                     [[0.55 * tol, 0, 0], [0, 1, 0], [1, 0, 0]]])  # 0.1 tol apart, a cell edge between
    verts, faces = weld(tris, tol)
    assert len(verts) == 3 and faces[0, 0] == faces[1, 0]


def test_weld_keeps_corners_over_the_tolerance_apart():
    tol = 1e-3
    tris = np.array([[[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1.5 * tol, 0, 0], [1, 0, 0], [0, 1, 0]]])
    assert len(weld(tris, tol)[0]) == 4


def test_weld_merges_to_a_seed_corner_and_never_transitively():
    tol = 1.0  # ten corners 0.6 tol apart in a row, 5.4 tol from end to end
    row = np.array([[i * 0.6, 0, 0] for i in range(10)], dtype=float)
    tris = np.stack([row[:-2], row[1:-1], row[2:]], axis=1)
    verts, _ = weld(tris, tol)
    # seeds at 0, 1.2, 2.4, 3.6, 4.8: each takes the corner after it, none takes a corner > tol away
    assert len(verts) == 5
    assert verts[:, 0] == pytest.approx([0.0, 1.2, 2.4, 3.6, 4.8])


def test_weld_gives_a_corner_between_two_seeds_to_the_first():
    tol = 1.0
    tris = np.array([[[0.0, 0, 0], [0.9, 0, 0], [1.8, 0, 0]], [[0.0, 5, 0], [0.9, 5, 0], [5, 5, 5]]])
    _, faces = weld(tris, tol)
    assert faces[0, 0] == faces[0, 1]  # 0.9 joins the seed at 0 and not the one at 1.8
    assert faces[0, 2] != faces[0, 1]


# --- shells: a mesh split into its connected closed pieces ---------------------------------------


def _indexed(tris):
    """A triangle soup as shared vertices and faces."""
    verts, inverse = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    return verts, inverse.reshape(-1, 3)


def _two_boxes():
    """One vertex list holding two closed boxes that overlap in space: a body of two groups."""
    va, fa = _indexed(box((0, 0, 0), (1, 1, 1)))
    vb, fb = _indexed(box((0.5, 0.5, 0.5), (1.5, 1.5, 1.5)))
    verts = np.concatenate([va, vb])
    faces = np.concatenate([fa, fb + len(va)])
    labels = np.concatenate([np.zeros(len(va), int), np.ones(len(vb), int)])
    return verts, faces, labels


def test_shells_split_one_mesh_by_group():
    _, faces, labels = _two_boxes()
    found = geom.shells(faces, labels)
    assert [label for label, _ in found] == [0, 1]
    assert [len(rows) for _, rows in found] == [12, 12]
    assert np.array_equal(found[0][1], np.arange(12))
    assert np.array_equal(found[1][1], np.arange(12, 24))


def test_each_shell_has_its_own_index_where_one_index_would_miscount():
    # inside() counts ray hits by parity, so a point inside both overlapping boxes reads as
    # outside in one shared index (two crossings of each) and inside in each shell's own index
    verts, faces, labels = _two_boxes()
    point = np.array([[0.75, 0.75, 0.75]])
    assert not SurfaceIndex(verts[faces]).inside(point)[0]
    for _, rows in geom.shells(faces, labels):
        assert SurfaceIndex(verts[faces[rows]]).inside(point)[0]


def test_a_shell_takes_the_label_most_of_its_vertices_hold():
    _, faces = _indexed(box((0, 0, 0), (1, 1, 1)))
    labels = np.array([2, 2, 2, 2, 2, 7, 7, 7])
    assert geom.shells(faces, labels)[0][0] == 2
    assert geom.shells(faces, np.array([4, 4, 4, 4, 9, 9, 9, 9]))[0][0] == 4  # a tie: the lowest


def test_shells_ignore_unused_vertices_and_an_empty_mesh():
    verts, faces = _indexed(box((0, 0, 0), (1, 1, 1)))
    labels = np.concatenate([np.zeros(len(verts), int), [5, 5, 5]])  # three loose vertices
    assert [label for label, _ in geom.shells(faces, labels)] == [0]
    assert geom.shells(np.zeros((0, 3), int), labels) == []


def test_shells_refuse_a_face_outside_the_labels():
    with pytest.raises(ValueError, match="outside"):
        geom.shells(np.array([[0, 1, 5]]), np.zeros(3, int))


# --- cut_loops: where two meshes cut, chained into lines -----------------------------------


def _is_closed_line(line: np.ndarray) -> bool:
    return len(line) > 3 and bool(np.array_equal(line[0], line[-1]))


def test_a_box_sunk_into_a_plate_cuts_it_in_one_closed_loop():
    post = box((-0.02, 0.001, -0.02), (0.02, 0.1, 0.02))  # foot 1 mm into the plate's top, y 0.002
    loops = geom.cut_loops(post, plate())
    assert len(loops) == 1 and _is_closed_line(loops[0])
    # the loop runs round the post, on its four sides at the plate's top height
    assert np.allclose(loops[0][:, 1], 0.002)
    assert np.abs(loops[0][:, [0, 2]]).max() == pytest.approx(0.02)


def test_a_post_through_a_plate_cuts_it_in_two_closed_loops():
    post = box((-0.02, -0.05, -0.02), (0.02, 0.05, 0.02))
    loops = geom.cut_loops(post, plate())
    assert len(loops) == 2 and all(_is_closed_line(loop) for loop in loops)
    assert sorted(round(float(loop[0, 1]), 4) for loop in loops) == [0.0, 0.002]


def test_two_meshes_that_do_not_touch_have_no_loops():
    assert geom.cut_loops(box((0, 0, 0), (1, 1, 1)), box((2, 0, 0), (3, 1, 1))) == []


def test_a_cut_that_runs_off_the_edge_of_an_open_sheet_is_an_open_line():
    # the block straddles the sheet's +x edge: the sheet's top face cuts its sides in three
    # lines that start and end on the sheet's border, never closing
    block_over_the_edge = box((0.05, -0.01, -0.05), (0.15, 0.01, 0.05))
    loops = geom.cut_loops(block_over_the_edge, open_plate())
    assert len(loops) == 1 and not _is_closed_line(loops[0])


def test_the_loops_are_the_same_whichever_mesh_is_named_first():
    post = box((-0.02, 0.001, -0.02), (0.02, 0.1, 0.02))
    one, other = geom.cut_loops(post, plate()), geom.cut_loops(plate(), post)
    assert len(one) == len(other) == 1
    key = lambda loop: {tuple(np.round(p, 9)) for p in loop}
    assert key(one[0]) == key(other[0])


def test_a_cut_along_a_shared_edge_of_the_other_mesh_is_not_counted_twice():
    # the post straddles the plate top's diagonal, so the loop crosses the edge two triangles
    # share, and the corner points there are shared by segments of both triangles
    post = box((-0.02, 0.001, -0.02), (0.02, 0.1, 0.02))
    loops = geom.cut_loops(post, plate())
    sides = np.diff(loops[0], axis=0)
    assert np.all(np.linalg.norm(sides, axis=1) > 0)  # no zero-length step, no repeated point


def test_a_cut_that_touches_itself_is_one_loop_whichever_order_its_segments_come_in():
    # a figure of eight: two triangles of cut joined at vertex 0 (degree 4)
    import itertools

    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]
    for order in itertools.permutations(range(len(edges))):
        for flip_bits in range(4):
            rows = [edges[i][::-1] if (flip_bits >> (i % 2)) & 1 else edges[i] for i in order]
            (line,) = geom._chain(np.array(rows))
            assert line[0] == line[-1] and len(line) == 7 and sorted(set(line)) == [0, 1, 2, 3, 4]


def test_edges_that_form_an_open_path_chain_from_its_end_and_loops_come_after_open_lines():
    path = geom._chain(np.array([(5, 4), (3, 4), (3, 2)]))
    assert path == [[2, 3, 4, 5]] or path == [[5, 4, 3, 2]]
    both = geom._chain(np.array([(0, 1), (1, 2), (2, 0), (7, 8), (8, 9)]))
    assert [line[0] == line[-1] for line in both] == [False, True]

# --- centre_of_mass and contact_faces (the bearing check) ------------------------------------


def test_centre_of_mass_of_a_box_is_its_middle_and_of_a_moved_blob_follows_it():
    assert geom.centre_of_mass(box((0, 0, 0), (2, 4, 6))) == pytest.approx([1, 2, 3])
    far = box((10, -3, 7), (12, 3, 9))  # far from the origin, where signed tetrahedra cancel
    assert geom.centre_of_mass(far) == pytest.approx([11, 0, 8])
    moved = blob() + np.array([5.0, -2.0, 1.0])
    assert geom.centre_of_mass(moved) - geom.centre_of_mass(blob()) == pytest.approx([5, -2, 1])


def test_centre_of_mass_of_two_boxes_weighs_by_volume():
    both = np.concatenate([box((0, 0, 0), (1, 1, 1)), box((5, 0, 0), (7, 1, 1))])
    # volumes 1 and 2, centres x 0.5 and 6: (0.5 + 2 * 6) / 3
    assert geom.centre_of_mass(both)[0] == pytest.approx(12.5 / 3)


def test_centre_of_mass_needs_a_closed_outward_mesh():
    with pytest.raises(ValueError, match="closed"):
        geom.centre_of_mass(open_plate())
    with pytest.raises(ValueError, match="outward"):
        geom.centre_of_mass(box((0, 0, 0), (1, 1, 1))[:, ::-1])  # wound inward


def test_contact_faces_are_the_facing_faces_within_reach():
    low, high = box((0, 0, 0), (2, 1, 2)), box((0.5, 1.0, 0.5), (1.5, 2, 1.5))
    on_low, on_high = geom.contact_faces(low, high, 0.01)
    # the low box's top face (+y, 2 triangles) meets the high box's bottom face (-y)
    assert on_low.sum() == 2 and np.all(low[on_low].reshape(-1, 3)[:, 1] == 1.0)
    assert on_high.sum() == 2 and np.all(high[on_high].reshape(-1, 3)[:, 1] == 1.0)
    assert len(on_low) == len(low) and len(on_high) == len(high)


def test_contact_faces_leave_out_edges_sides_and_the_far_side_of_a_thin_part():
    low = box((0, 0, 0), (2, 1, 2))
    flush = box((0, 1.0, 0), (1, 2, 1))  # its side faces meet the low box only along an edge
    on_low, on_flush = geom.contact_faces(low, flush, 0.01)
    assert on_low.sum() == 2 and on_flush.sum() == 2
    # a slab thinner than the reach: only its face that looks at the part counts
    slab = box((0, 1.0, 0), (2, 1.005, 2))
    _, on_slab = geom.contact_faces(low, slab, 0.01)
    assert on_slab.sum() == 2 and np.all(slab[on_slab].reshape(-1, 3)[:, 1] == 1.0)


def test_contact_faces_are_empty_for_parts_out_of_reach():
    on_a, on_b = geom.contact_faces(box((0, 0, 0), (1, 1, 1)), box((0, 3, 0), (1, 4, 1)), 0.5)
    assert not on_a.any() and not on_b.any()


def test_contact_faces_leave_out_the_back_to_back_sides_of_flush_parts():
    """A 20 mm cap flush on a 20 mm post, with a reach (21.3 mm) larger than the parts. The
    post's side faces oppose the cap's opposite sides and are near them, but look away from
    them: only the cap's underside and the post's top meet."""
    post = box((0, 0, 0), (0.02, 0.2, 0.02))
    cap = box((0, 0.2, 0), (0.02, 0.22, 0.02))
    on_post, on_cap = geom.contact_faces(post, cap, 0.0213)
    assert on_post.sum() == 2 and np.all(post[on_post].reshape(-1, 3)[:, 1] == 0.2)
    assert on_cap.sum() == 2 and np.all(cap[on_cap].reshape(-1, 3)[:, 1] == 0.2)


def test_contact_faces_let_a_part_sit_a_little_in_its_host():
    low = box((0, 0, 0), (2, 1, 2))
    sunk = box((0.5, 0.97, 0.5), (1.5, 2, 1.5))  # 0.03 into the low box
    _, bare = geom.contact_faces(low, sunk, 0.1)
    assert not bare.any()  # its underside is behind the low box's top face
    on_low, on_sunk = geom.contact_faces(low, sunk, 0.1, sunk=0.05)
    assert on_low.sum() == 2 and on_sunk.sum() == 2


def test_ray_counts_with_a_reach_ignore_hits_beyond_it():
    wall = box((-1, 5, -1), (1, 6, 1))
    origin, up = np.array([[0.1, 0.0, 0.3]]), np.array([[0.0, 1.0, 0.0]])
    assert geom.ray_counts(wall, origin, up)[0, 0] == 2
    assert geom.ray_counts(wall, origin, up, reach=4.0)[0, 0] == 0
    assert geom.ray_counts(wall, origin, 2 * up, reach=5.5)[0, 0] == 1  # the near face at 5.0
    assert geom.ray_counts(wall, origin, up, reach=7.0)[0, 0] == 2


# --- surface samples, overlap depth and local thickness (crossing depth by part thickness) -----


def test_surface_samples_keep_neighbours_within_the_spacing_and_include_the_corners():
    cube = box((0, 0, 0), (1, 1, 1))
    pts = geom.surface_samples(cube, 0.25)
    for corner in ((0, 0, 0), (1, 1, 1), (1, 0, 1)):
        assert (np.abs(pts - corner).max(axis=1) < 1e-12).any()
    assert len(np.unique(pts, axis=0)) == len(pts)  # every point once
    on_top = pts[pts[:, 1] == 1.0][:, [0, 2]]
    assert len(on_top) >= 25  # the top face is at least a 5 x 5 lattice of 0.25 steps
    for point in ((0.25, 0.5), (0.75, 0.0), (0.5, 0.5)):
        assert (np.abs(on_top - point).max(axis=1) < 1e-12).any()


def test_surface_samples_count_a_shared_vertex_once():
    cube = box((0, 0, 0), (1, 1, 1))
    pts = geom.surface_samples(cube, 10.0)  # spacing far above every edge: the corners alone
    assert len(pts) == 8  # 12 triangles share 8 corners, which a per-triangle lattice repeats 36 times


def test_surface_samples_of_a_sliver_cost_a_line_and_nothing_is_lost_for_no_triangles():
    sliver = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.001, 0.0]]])
    assert len(geom.surface_samples(sliver, 0.5)) < 100
    assert geom.surface_samples(np.zeros((0, 3, 3)), 0.5).shape == (0, 3)
    with pytest.raises(ValueError, match="spacing"):
        geom.surface_samples(box((0, 0, 0), (1, 1, 1)), 0.0)


def test_there_are_13_depth_directions_all_unit_and_none_repeated():
    d = geom.DEPTH_DIRECTIONS
    assert d.shape == (13, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert len({tuple(np.round(abs(x), 9)) for x in d}) == 13 or len(np.unique(np.round(d, 9), axis=0)) == 13


def test_the_26_depth_ways_are_the_13_directions_with_both_signs():
    w = geom.DEPTH_WAYS
    assert w.shape == (26, 3)
    assert np.array_equal(w[:13], geom.DEPTH_DIRECTIONS) and np.array_equal(w[13:], -w[:13])
    assert len(np.unique(np.round(w, 9), axis=0)) == 26


def test_overlap_depth_is_the_thinnest_extent_of_the_points():
    cuboid = box((0, 0, 0), (4, 0.5, 3))  # corners and nothing else
    assert geom.overlap_depth(cuboid.reshape(-1, 3)) == pytest.approx(0.5)
    assert geom.overlap_depth(np.zeros((0, 3))) == 0.0
    assert geom.overlap_depth(np.array([[1.0, 2.0, 3.0]])) == 0.0


def test_overlap_depth_of_a_tilted_slab_is_found_along_a_diagonal():
    # a slab 0.1 thick whose faces lie in planes x + y = 0 and x + y = 0.1414: thinnest along (1,1,0)
    s = 0.1 / math.sqrt(2)
    pts = np.array([[x, y, z] for x in (0.0, 5.0) for z in (0.0, 5.0) for y in (-x, s * 2 - x)])
    assert geom.overlap_depth(pts) == pytest.approx(0.1, abs=1e-9)


def test_local_thickness_of_a_slab_and_a_rod():
    slab = box((0.0, 0.0, 0.0), (1.0, 0.02, 1.0))
    # slanted rays of the cone run a little longer than the slab is thick
    assert geom.local_thickness(slab, (0.5, 0.02, 0.5)) == pytest.approx(0.02, rel=0.15)  # on it
    assert geom.local_thickness(slab, (0.5, 0.01, 0.5)) == pytest.approx(0.02, rel=0.15)  # inside
    rod = cylinder((0.0, 0.0), 0.01, 0.0, 0.4, sides=96)
    assert geom.local_thickness(rod, (0.01, 0.2, 0.0)) == pytest.approx(0.02, rel=0.15)  # its side
    assert geom.local_thickness(rod, (0.0, 0.2, 0.0)) == pytest.approx(0.02, rel=0.15)  # its axis


def test_local_thickness_of_a_rods_flat_end_is_its_width_and_not_its_length():
    post = box((0.0, 0.0, 0.0), (0.04, 0.4, 0.04))
    end = geom.local_thickness(post, (0.02, 0.4, 0.02))  # the middle of the end face
    assert end < 0.1  # a straight ray would read 0.4
    assert end == pytest.approx(0.04, rel=0.15)  # about its width


def test_local_thickness_does_not_care_how_the_faces_wind():
    slab = box((0.0, 0.0, 0.0), (1.0, 0.02, 1.0))
    assert geom.local_thickness(slab[:, ::-1], (0.5, 0.02, 0.5)) == pytest.approx(
        geom.local_thickness(slab, (0.5, 0.02, 0.5)))


def test_local_thickness_raises_on_an_open_mesh():
    with pytest.raises(ValueError, match="closed"):
        geom.local_thickness(open_plate(), (0.0, 0.0, 0.0))


def test_thinner_thickness_is_the_smaller_of_the_two_local_thicknesses():
    slab = box((0.0, 0.0, 0.0), (1.0, 0.02, 1.0))
    block = box((0.0, 0.0, 0.0), (1.0, 0.2, 1.0))
    point = (0.5, 0.02, 0.5)
    assert geom.thinner_thickness(slab, block, point) == geom.local_thickness(slab, point)
    assert geom.thinner_thickness(block, slab, point) == geom.local_thickness(slab, point)
    assert geom.thinner_thickness(slab, block, point) < geom.local_thickness(block, point)


def test_thinner_thickness_raises_unmeasurable_when_either_part_has_none():
    slab = box((0.0, 0.0, 0.0), (1.0, 0.02, 1.0))
    for a, b in ((open_plate(), slab), (slab, open_plate())):
        with pytest.raises(Unmeasurable, match="no thickness"):
            geom.thinner_thickness(a, b, (0.0, 0.0, 0.0))
    assert issubclass(Unmeasurable, ManifestError)  # a check must catch it, or the run exits 2


# --- exits ---------------------------------------------------------------------------------------


def test_exits_is_the_way_out_of_a_closed_mesh_along_each_direction():
    cube = box((0, 0, 0), (1, 1, 1))
    origins = np.array([[0.5, 0.25, 0.5], [0.2, 0.5, 0.5], [2.0, 0.5, 0.5]])
    dirs = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    out = geom.exits(cube, origins, dirs)
    assert out.shape == (3, 2)
    assert out[0] == pytest.approx([0.75, 0.5]) and out[1] == pytest.approx([0.5, 0.2])
    assert np.isinf(out[2, 0]) and out[2, 1] == pytest.approx(1.0)  # outside: it meets the far face


def test_exits_of_nothing_is_inf_and_a_point_on_a_face_does_not_meet_it():
    cube = box((0, 0, 0), (1, 1, 1))
    assert np.isinf(geom.exits(np.zeros((0, 3, 3)), np.zeros((1, 3)), np.eye(3)[:1])).all()
    assert geom.exits(cube, np.zeros((0, 3)), np.eye(3)).shape == (0, 3)
    on_top = geom.exits(cube, np.array([[0.5, 1.0, 0.5]]), np.array([[0.0, -1.0, 0.0]]))
    assert on_top[0, 0] == pytest.approx(1.0)  # it leaves its own face and meets the bottom


# --- shortest_exit -------------------------------------------------------------------------------


def test_shortest_exit_is_the_least_over_the_ways_of_the_furthest_point():
    cube = box((0, 0, 0), (1, 1, 1))
    pts = np.array([[0.5, 0.25, 0.5], [0.5, 0.5, 0.5]])
    ways = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
    # out through the top: 0.75 and 0.5, the furthest is 0.75; down 0.25 and 0.5; sideways 0.5 both
    assert geom.shortest_exit([geom.Exit(cube, pts, ways)], 0.1) == pytest.approx(0.5)
    assert geom.shortest_exit([geom.Exit(cube, pts[:1], ways)], 0.1) == pytest.approx(0.25)


def test_shortest_exit_widens_its_reach_until_the_answer_is_inside_it():
    cube = box((0, 0, 0), (10, 1, 1))
    pts = np.array([[5.0, 0.5, 0.5]])
    ways = np.array([[1.0, 0.0, 0.0]])
    assert geom.shortest_exit([geom.Exit(cube, pts, ways)], 1e-3) == pytest.approx(5.0)


def test_shortest_exit_of_no_points_is_zero_and_of_no_way_out_is_inf():
    cube = box((0, 0, 0), (1, 1, 1))
    ways = np.array([[0.0, 1.0, 0.0]])
    assert geom.shortest_exit([geom.Exit(cube, np.zeros((0, 3)), ways)], 0.1) == 0.0
    assert geom.shortest_exit([], 0.1) == 0.0
    assert np.isinf(geom.shortest_exit([geom.Exit(cube[:0], np.array([[0.5, 0.5, 0.5]]), ways)], 0.1))


def test_a_point_that_leaves_at_once_goes_nowhere_along_that_way():
    cube = box((0, 0, 0), (1, 1, 1))
    on_top = np.array([[0.5, 1.0, 0.5]])  # on the top face: its own face is not met going up
    ways = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    plain = geom.shortest_exit([geom.Exit(cube, on_top, ways)], 0.1)
    assert plain == pytest.approx(0.5)  # going up meets nothing: only the sideways way remains
    leaves = np.array([[True, False]])
    assert geom.shortest_exit([geom.Exit(cube, on_top, ways, leaves)], 0.1) == 0.0


def test_two_sides_move_one_way_and_its_reverse():
    # a small cube sunk 0.3 into the top of a big one. Moving the small one up by t takes its
    # points out of the big one at t = 0.3, and moves the big one's points down, relative to the
    # small one, out of the small one at t = 0.3 (the small cube's bottom is 0.3 below the top).
    big, small = box((-1, -1, -1), (1, 0, 1)), box((-0.1, -0.3, -0.1), (0.1, 0.4, 0.1))
    held_small = np.array([[0.0, -0.3, 0.0], [0.05, -0.15, 0.05]])  # the small cube's, in the big
    held_big = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, -0.05]])  # the big cube's top, in the small
    ways = np.array([[0.0, 1.0, 0.0]])
    sides = [geom.Exit(big, held_small, ways), geom.Exit(small, held_big, -ways)]
    assert geom.shortest_exit(sides, 0.01) == pytest.approx(0.3)
    assert geom.shortest_exit(sides[:1], 0.01) == pytest.approx(0.3)


def test_surface_area_of_a_unit_cube_is_6():
    assert surface_area(box((0, 0, 0), (1, 1, 1))) == pytest.approx(6.0)
