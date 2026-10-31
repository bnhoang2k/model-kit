"""The bounding-volume hierarchy answers what the brute-force functions answer, with far fewer
exact tests: nearest point on a mesh (`geom._closest_dense`) and inside by ray parity
(`geom.ray_counts`, two odd rays out of three)."""

import numpy as np
import pytest

from model_kit.core import bvh as bvh_module
from model_kit.core import geom
from model_kit.core.bvh import Bvh, Deepest
from model_kit.core.geom import RAYS, SurfaceIndex, ray_counts

from .shapes import blob, block, open_plate, plate, soup, wavy_grid

REL = 1e-12  # the exactness line: distances within 1e-12 relative


def _brute_inside(tris, pts):
    return (ray_counts(tris, pts, RAYS) % 2 == 1).sum(axis=1) >= 2


def _lowest_tie(tris, pts):
    """Per point: the lowest face index among the triangles at exactly the least distance (the
    arithmetic the index itself uses), and that face's nearest point."""
    faces, points = [], []
    for p in pts:
        q = geom._closest_on_triangles(np.tile(p, (len(tris), 1)), tris[:, 0], tris[:, 1],
                                       tris[:, 2])
        d2 = geom._dot(p - q, p - q)
        faces.append(int(np.flatnonzero(d2 == d2.min())[0]))
        points.append(q[faces[-1]])
    return np.array(faces), np.array(points)


def _in_box(tris, pts):
    """The points inside the mesh's own box. `inside` answers False outside it, by design: a closed
    mesh holds nothing there. An open mesh can have a ray hit from outside its box, so the brute
    count agrees with `inside` only for the points in the box (grown by a float margin, since the
    tree grows its own box by a float pad)."""
    lo, hi = tris.min(axis=(0, 1)), tris.max(axis=(0, 1))
    margin = 1e-6 * float((hi - lo).max())
    return (pts >= lo - margin).all(axis=1) & (pts <= hi + margin).all(axis=1)


def _around(tris, n, seed):
    """`n` random points in the mesh's box grown by a fifth on every side."""
    lo, hi = tris.min(axis=(0, 1)), tris.max(axis=(0, 1))
    pad = 0.2 * (hi - lo)
    return np.random.default_rng(seed).uniform(lo - pad, hi + pad, size=(n, 3))


def _same_distance(got, want):
    scale = max(float(np.max(want)), 1e-3) if len(want) else 1.0
    assert got == pytest.approx(want, rel=REL, abs=REL * scale * 1e-3)


def _on_the_surface(tris, seed=0):
    """Points exactly on faces (centres), edges (midpoints), corners, and a few off them."""
    rng = np.random.default_rng(seed)
    centres = tris.mean(axis=1)
    mids = (tris + np.roll(tris, -1, axis=1)).reshape(-1, 3) / 2
    corners = tris.reshape(-1, 3)
    weights = rng.dirichlet(np.ones(3), size=len(tris))
    inside_face = np.einsum("nk,nkj->nj", weights, tris)
    return np.concatenate([centres, mids, corners, inside_face])


SHAPES = {
    "plate": plate,
    "block": block,
    "blob": blob,
    "wavy": lambda: wavy_grid(20, 20),
    "soup": lambda: soup(700, 0.15, seed=3),
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("leaf", [1, 3, 4, 16, 10_000])
def test_closest_equals_the_dense_function(shape, leaf, monkeypatch):
    monkeypatch.setattr(bvh_module, "BVH_LEAF", leaf)
    tris = SHAPES[shape]()
    pts = np.concatenate([_around(tris, 600, 7), _on_the_surface(tris)[:900]])
    got = Bvh(tris).closest(pts)
    dense = geom._closest_dense(tris, pts)
    _same_distance(got.dist, dense.dist)
    faces, points = _lowest_tie(tris, pts)
    assert got.point == pytest.approx(points, abs=1e-12)
    assert (got.face == faces).all()
    assert got.normal == pytest.approx(geom._unit_normals(tris)[faces], abs=1e-12)


@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("leaf", [1, 4, 16, 10_000])
def test_inside_equals_the_ray_parity_count(shape, leaf, monkeypatch):
    monkeypatch.setattr(bvh_module, "BVH_LEAF", leaf)
    tris = SHAPES[shape]()
    pts = np.concatenate([_around(tris, 1500, 11), _on_the_surface(tris)[:900]])
    got = Bvh(tris).inside(pts)
    want = _brute_inside(tris, pts)
    box = _in_box(tris, pts)
    assert box.sum() > 500
    assert (got[box] == want[box]).all()
    if shape in ("plate", "block", "blob"):  # closed: nothing outside the box can be inside
        assert (got == want).all()
    assert not got[~box].any()


def test_inside_has_both_answers_on_a_closed_mesh():
    tris = blob()
    pts = np.random.default_rng(1).uniform(-1.3, 1.3, size=(2000, 3))
    got = Bvh(tris).inside(pts)
    assert got.any() and not got.all()
    assert (got == _brute_inside(tris, pts)).all()


def test_an_open_mesh_gets_the_same_answers_as_brute_force():
    """Parity means little on an open mesh, but the answer is still the brute-force one, and the
    three rays disagree there, so the third is needed."""
    tris = np.concatenate([open_plate(), soup(60, 0.2, seed=5)])
    pts = np.random.default_rng(2).uniform(-1.2, 1.2, size=(2500, 3))
    got = Bvh(tris).inside(pts)
    parity = ray_counts(tris, pts, RAYS) % 2 == 1
    box = _in_box(tris, pts)
    assert (parity[box, 0] != parity[box, 1]).any()
    assert (got[box] == _brute_inside(tris, pts)[box]).all()
    _same_distance(Bvh(tris).closest(pts).dist, geom._closest_dense(tris, pts).dist)


def test_degenerate_triangles_do_not_break_either_query():
    zero_area = np.array([[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.6, 0.5, 0.5]]])
    point_tri = np.array([[[0.4, 0.4, 0.4]] * 3])
    collinear = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=float)
    pts = np.concatenate([np.random.default_rng(4).uniform(-0.5, 2.5, size=(300, 3)),
                          [[0.55, 0.6, 0.5], [0.4, 0.4, 0.4], [1.0, 0.0, 0.0]]])
    for extra in (zero_area, point_tri, collinear):
        tris = np.concatenate([blob(8, 12), extra])
        got = Bvh(tris).closest(pts)
        assert np.isfinite(got.dist).all() and np.isfinite(got.normal).all()
        _same_distance(got.dist, geom._closest_dense(tris, pts).dist)
        box = _in_box(tris, pts)
        assert (Bvh(tris).inside(pts)[box] == _brute_inside(tris, pts)[box]).all()
    lone = Bvh(zero_area).closest(np.array([[0.55, 0.6, 0.5]]))
    assert lone.dist[0] == pytest.approx(0.1)


def test_a_tie_goes_to_the_lowest_original_face():
    tris = soup(100, 0.1, seed=9)
    tris[90] = tris[5]
    tris[40] = tris[5]
    pts = tris[5].mean(axis=0) + np.random.default_rng(1).normal(size=(50, 3)) * 0.01
    got = Bvh(tris).closest(pts)
    twins = np.isin(got.face, [5, 40, 90])
    assert twins.any() and (got.face[twins] == 5).all()


def test_a_point_on_a_shared_edge_ties_to_the_lowest_face_not_a_pruned_one(monkeypatch):
    """Distance 0 gives a zero relative bound, so no leaf may be dropped by it alone: the faces
    that share the edge all tie at 0 and the lowest wins, however the tree splits them."""
    tris = wavy_grid(12, 12)
    pts = _on_the_surface(tris, seed=3)
    faces, _ = _lowest_tie(tris, pts)
    for leaf in (1, 2, 4, 8):
        monkeypatch.setattr(bvh_module, "BVH_LEAF", leaf)
        got = Bvh(tris).closest(pts)
        assert (got.dist == 0.0).sum() > len(tris)  # the points that lie on the surface
        assert (got.face == faces).all(), leaf


def test_the_hierarchy_prunes_far_below_the_leaf_grid():
    tris = wavy_grid(50, 60)
    rng = np.random.default_rng(4)
    pts = np.column_stack([rng.uniform(0, 1, 2000), rng.uniform(-0.03, 0.08, 2000),
                           rng.uniform(0, 1, 2000)])
    tree = Bvh(tris)
    near = tree.closest(pts)
    per_point = tree.last_pairs / len(pts)
    assert per_point < 0.01 * len(tris)
    _same_distance(near.dist, geom._closest_dense(tris, pts).dist)
    ball = blob(40, 80)
    ib = Bvh(ball)
    ib.inside(pts * 2 - 1)
    per_ray = ib.last_pairs / (len(pts) * 3)
    assert per_ray < 0.05 * len(ball)


def test_chunking_does_not_change_answers(monkeypatch):
    tris = soup(500, 0.15, seed=3)
    pts = np.random.default_rng(6).uniform(-1.5, 1.5, size=(300, 3))
    ball = blob()
    whole = Bvh(tris).closest(pts)
    whole_in = Bvh(ball).inside(pts)
    monkeypatch.setattr(geom, "PAIR_BUDGET", 97)
    monkeypatch.setattr(geom, "PAIR_BLOCK", 31)
    small = Bvh(tris).closest(pts)
    assert (small.face == whole.face).all()
    assert small.dist == pytest.approx(whole.dist, abs=1e-12)
    assert (Bvh(ball).inside(pts) == whole_in).all()


def test_the_surface_index_answers_through_the_hierarchy():
    """`SurfaceIndex.closest` and `.inside` are the hierarchy's answers, so every check that
    holds a SurfaceIndex gains the speed."""
    tris = blob()
    pts = np.random.default_rng(8).uniform(-1.3, 1.3, size=(800, 3))
    index = SurfaceIndex(tris)
    tree = Bvh(index)
    a, b = index.closest(pts), tree.closest(pts)
    assert (a.face == b.face).all() and (a.dist == b.dist).all()
    assert (index.inside(pts) == tree.inside(pts)).all()
    assert index.last_pairs == tree.last_pairs > 0
    assert geom.closest_points(tris, pts).dist == pytest.approx(b.dist, rel=REL)


def test_empty_and_bad_input():
    tree = Bvh(plate())
    assert tree.closest(np.zeros((0, 3))).dist.shape == (0,)
    assert tree.inside(np.zeros((0, 3))).shape == (0,)
    bad = np.array([[0.0, 0.1, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="points hold a non-finite value"):
        tree.closest(bad)
    with pytest.raises(ValueError, match="points hold a non-finite value"):
        tree.inside(bad)
    with pytest.raises(ValueError, match="at least one triangle"):
        Bvh(np.zeros((0, 3, 3)))
    one = Bvh(plate()[:1])
    assert one.closest(np.array([[0.0, 1.0, 0.0]])).dist.shape == (1,)


# --- the deepest inside point per group ----------------------------------------------------------


def _brute_deepest(tris, pts, groups, floor):
    """Inside by the ray count, depth by the tree's own `closest` (itself held to the dense
    function above), a maximum per group, the lowest row on a tie, kept when over the floor."""
    inside = _brute_inside(tris, pts)
    depth = Bvh(tris).closest(pts).dist
    rows, gs, ds = [], [], []
    for g in np.unique(groups):
        mine = np.flatnonzero(inside & (groups == g))
        if not len(mine):
            continue
        top = depth[mine].max()
        if top > floor:
            rows.append(int(mine[depth[mine] == top][0]))
            gs.append(g)
            ds.append(top)
    return np.array(gs), np.array(ds), np.array(rows)


def _lattice(tris, n):
    """Points on a regular lattice through the mesh's box: symmetric meshes give equal depths."""
    lo, hi = tris.min(axis=(0, 1)), tris.max(axis=(0, 1))
    axes = [np.linspace(a, b, n) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


@pytest.mark.parametrize("shape", ["plate", "block", "blob"])
@pytest.mark.parametrize("leaf", [1, 4])
@pytest.mark.parametrize("floor_share", [0.0, 0.3, 5.0])
def test_deepest_equals_a_brute_force_maximum_per_group(shape, leaf, floor_share, monkeypatch):
    monkeypatch.setattr(bvh_module, "BVH_LEAF", leaf)
    tris = SHAPES[shape]()
    rng = np.random.default_rng(21)
    pts = np.concatenate([_around(tris, 900, 3), _lattice(tris, 9), _on_the_surface(tris)[:200]])
    groups = rng.integers(0, 6, size=len(pts))
    top = Bvh(tris).closest(pts).dist[_brute_inside(tris, pts)].max()
    floor = floor_share * top
    got = Bvh(tris).deepest(pts, groups, floor)
    want_g, want_d, want_r = _brute_deepest(tris, pts, groups, floor)
    assert isinstance(got, Deepest)
    assert (got.group == want_g).all() and (got.row == want_r).all()
    assert got.depth == pytest.approx(want_d, rel=REL)
    if floor_share == 0.0:
        assert len(got.group) > 3
    if floor_share == 5.0:
        assert len(got.group) == 0


def test_deepest_ties_go_to_the_lowest_row():
    """A symmetric block and a lattice give many points at the very same depth."""
    tris = block()
    pts = _lattice(tris, 7)
    groups = np.zeros(len(pts), dtype=int)
    depth = Bvh(tris).closest(pts).dist
    inside = _brute_inside(tris, pts)
    assert (depth[inside] == depth[inside].max()).sum() > 1  # the tie is real
    got = Bvh(tris).deepest(pts, groups)
    assert got.row[0] == np.flatnonzero(inside & (depth == depth[inside].max()))[0]
    # The same point again, later in the query, never takes the row.
    twice = np.concatenate([pts, pts])
    assert Bvh(tris).deepest(twice, np.zeros(len(twice), dtype=int)).row[0] == got.row[0]


def test_deepest_with_many_points_in_one_group_takes_several_rounds_and_stays_exact():
    tris = blob(30, 60)
    pts = _around(tris, 6000, 8)
    groups = np.zeros(len(pts), dtype=int)
    tree = Bvh(tris)
    got = tree.deepest(pts, groups)
    _, want_d, want_r = _brute_deepest(tris, pts, groups, 0.0)
    assert got.row.tolist() == want_r.tolist() and got.depth == pytest.approx(want_d, rel=REL)


def test_deepest_skips_the_points_that_cannot_win(monkeypatch):
    tris = blob(30, 60)
    pts = _around(tris, 4000, 9)
    groups = np.random.default_rng(2).integers(0, 5, size=len(pts))
    seen = []
    real = Bvh.inside

    def spy(self, p):
        seen.append(len(p))
        return real(self, p)

    monkeypatch.setattr(Bvh, "inside", spy)
    tree = Bvh(tris)
    tree.deepest(pts, groups)
    assert sum(seen) < 0.5 * len(pts)  # `inside` ran on under half the points
    full = Bvh(tris)
    held = full.inside(pts)
    tests = full.last_pairs
    full.closest(pts[held])
    tests += full.last_pairs
    assert tree.last_pairs < 0.8 * tests  # the exact tests fell, bounds included


def test_deepest_chunked_and_degenerate(monkeypatch):
    tris = blob(12, 24)
    pts = _around(tris, 800, 5)
    groups = np.arange(len(pts)) % 3
    whole = Bvh(tris).deepest(pts, groups)
    monkeypatch.setattr(geom, "PAIR_BUDGET", 97)
    monkeypatch.setattr(geom, "PAIR_BLOCK", 31)
    small = Bvh(tris).deepest(pts, groups)
    assert (small.group == whole.group).all() and (small.row == whole.row).all()
    assert small.depth == pytest.approx(whole.depth, rel=REL)
    none = Bvh(tris).deepest(np.zeros((0, 3)), np.zeros(0, dtype=int))
    assert len(none.group) == 0
    far = Bvh(tris).deepest(pts + 100.0, groups)  # outside the box: nothing inside
    assert len(far.group) == 0
    with pytest.raises(ValueError, match="groups must have shape"):
        Bvh(tris).deepest(pts, groups[:5])


def test_the_surface_index_still_offers_its_face_normals_in_original_order():
    """Face-contact code reads `index._normals[original face]`; they are formed when first asked."""
    tris = blob(10, 20)
    index = SurfaceIndex(tris)
    index.closest(np.zeros((3, 3)))
    assert (index._normals == geom._unit_normals(tris)).all()


# --- the fast path equals the old per-point path, bit for bit -------------------------------
#
# A solver sends `closest` thousands of small batches, so the walk was sped up without changing a
# result: the three axes go through one array (`_far2`, `_bounds2`), the walk starts below the top
# levels for a few points (`START_PAIRS`), and `np.cross` and `np.select` became `geom._cross` and a
# chain of `np.where`. The copies below are the code as it was. Each new piece does the same
# arithmetic in the same order, so every answer is equal to the last bit, not just close.


def _old_far2(lo, hi, p):
    total, extra = 0.0, None
    for a in range(3):
        mid = (lo[a] + hi[a]) / 2
        near = np.where(p[a] <= mid, lo[a], hi[a])
        far = np.where(p[a] >= mid, lo[a], hi[a])
        d_far = (p[a] - far) ** 2
        total = total + d_far
        swap = (p[a] - near) ** 2 - d_far
        extra = swap if extra is None else np.minimum(extra, swap)
    return total + extra


def _old_bounds2(lo, hi, p):
    gap2 = 0.0
    for a in range(3):
        gap = np.maximum(np.maximum(lo[a] - p[a], p[a] - hi[a]), 0.0)
        gap2 = gap2 + gap * gap
    return gap2, _old_far2(lo, hi, p)


def _old_closest_on_triangles(p, a, b, c):
    """`geom._closest_on_triangles` before `_cross` and the `np.where` chain."""
    dot = geom._dot
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    with np.errstate(divide="ignore", invalid="ignore"):
        face = a + (ab * (vb / (va + vb + vc))[:, None]) + (ac * (vc / (va + vb + vc))[:, None])
        on_ab = a + ab * (d1 / (d1 - d3))[:, None]
        on_ac = a + ac * (d2 / (d2 - d6))[:, None]
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        on_bc = b + (c - b) * w[:, None]
    conds = [
        (d1 <= 0) & (d2 <= 0),
        (d3 >= 0) & (d4 <= d3),
        (vc <= 0) & (d1 >= 0) & (d3 <= 0),
        (d6 >= 0) & (d5 <= d6),
        (vb <= 0) & (d2 >= 0) & (d6 <= 0),
        (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0),
    ]
    q = np.select([k[:, None] for k in conds], [a, b, on_ab, c, on_ac, on_bc], default=face)
    longest2 = np.maximum(np.maximum(dot(ab, ab), dot(ac, ac)), dot(c - b, c - b))
    cross = np.cross(ab, ac)
    flat = (np.sqrt(dot(cross, cross)) <= geom.DEGENERATE_SHARE * longest2) | ~np.isfinite(q).all(
        axis=1)
    if flat.any():
        pf, af, bf, cf = p[flat], a[flat], b[flat], c[flat]
        cands = [geom._closest_on_segments(pf, af, bf), geom._closest_on_segments(pf, bf, cf),
                 geom._closest_on_segments(pf, cf, af)]
        d2s = np.stack([dot(pf - k, pf - k) for k in cands])
        best = np.argmin(d2s, axis=0)
        q[flat] = np.stack(cands)[best, np.arange(len(pf))]
    return q


def _old_path(tris, pts):
    """The old answer, one point at a time against every triangle, with the old arithmetic: the
    least squared distance, the lowest original face on a tie."""
    dist, points, faces = [], [], []
    for p in pts:
        q = _old_closest_on_triangles(np.tile(p, (len(tris), 1)), tris[:, 0], tris[:, 1],
                                      tris[:, 2])
        d2 = geom._dot(p - q, p - q)
        face = int(np.flatnonzero(d2 == d2.min())[0])
        dist.append(np.sqrt(d2[face]))
        points.append(q[face])
        faces.append(face)
    faces = np.array(faces)
    n = np.cross(tris[faces, 1] - tris[faces, 0], tris[faces, 2] - tris[faces, 0])
    length = np.linalg.norm(n, axis=1, keepdims=True)
    normal = np.where(length > 0, n / np.where(length > 0, length, 1.0), 0.0)
    return np.array(dist), np.array(points), faces, normal


def _queries(tris, seed):
    """Off the mesh, on its faces, edges and corners, and a spread of batch sizes' worth."""
    return np.concatenate([_around(tris, 300, seed), _on_the_surface(tris, seed)[:200]])


def _assert_old_path(tree, tris, pts, sizes=(1, 3, 12, 40, 500)):
    dist, point, face, normal = _old_path(tris, pts)
    for size in sizes:  # the batch size decides where the walk starts
        got = [tree.closest(pts[i : i + size]) for i in range(0, len(pts), size)]
        assert np.array_equal(np.concatenate([g.dist for g in got]), dist), size
        assert np.array_equal(np.concatenate([g.point for g in got]), point), size
        assert np.array_equal(np.concatenate([g.face for g in got]), face), size
        assert np.array_equal(np.concatenate([g.normal for g in got]), normal), size


MESHES = {"blob": lambda: blob(14, 24), "wavy": lambda: wavy_grid(16, 16),
          "soup": lambda: soup(400, 0.15, seed=3)}


@pytest.mark.parametrize("mesh", sorted(MESHES))
@pytest.mark.parametrize("start_pairs", [1, 64, 2048, 10**9])
def test_closest_equals_the_old_per_point_path_bit_for_bit(mesh, start_pairs, monkeypatch):
    monkeypatch.setattr(bvh_module, "START_PAIRS", start_pairs)
    tris = MESHES[mesh]()
    _assert_old_path(Bvh(tris), tris, _queries(tris, 5))


def test_the_faster_kernels_equal_the_old_ones_bit_for_bit():
    rng = np.random.default_rng(12)
    for _ in range(20):
        lo = rng.uniform(-1, 1, size=(3, 500))
        hi = lo + rng.uniform(0, 1, size=(3, 500))
        p = rng.uniform(-1.5, 2.5, size=(3, 500))
        assert np.array_equal(bvh_module._far2(lo, hi, p), _old_far2(lo, hi, p))
        for new, old in zip(bvh_module._bounds2(lo, hi, p), _old_bounds2(lo, hi, p), strict=True):
            assert np.array_equal(new, old)
        u, v = rng.normal(size=(2, 500, 3))
        assert np.array_equal(geom._cross(u, v), np.cross(u, v))
    tris = np.concatenate([blob(8, 12), np.array([[[0.5] * 3, [0.5] * 3, [0.6, 0.5, 0.5]]])])
    pts = _around(tris, 300, 2)
    a = np.tile(tris, (len(pts), 1, 1))[: len(pts)]
    new = geom._closest_on_triangles(pts, a[:, 0], a[:, 1], a[:, 2])
    assert np.array_equal(new, _old_closest_on_triangles(pts, a[:, 0], a[:, 1], a[:, 2]))


def test_a_wrong_pruning_bound_breaks_the_old_path_pin(monkeypatch):
    """The planted fault: a bound that is too small drops the leaf that holds the nearest triangle.
    The pin above must see it: as a wrong answer, or as a crash when no leaf is left at all."""
    tris = blob(14, 24)
    pts = _queries(tris, 5)
    _assert_old_path(Bvh(tris), tris, pts, sizes=(12,))  # sound as it is
    real = bvh_module._far2
    monkeypatch.setattr(bvh_module, "_far2", lambda lo, hi, p: real(lo, hi, p) * 0.5)
    with pytest.raises((AssertionError, IndexError)):
        _assert_old_path(Bvh(tris), tris, pts, sizes=(12,))
