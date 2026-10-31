import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import edges, geom
from model_kit.core.errors import ManifestError
from model_kit.core.grow import mesh as mesh_module
from model_kit.core.grow.buds import MESH_V1, MESH_VERSION, grow_buds
from model_kit.core.grow.mesh import (
    LeafLean,
    _limb_tris,
    _lobed_rings,
    _lobes,
    _merge_collinear,
    _round_bends,
    _seated_points,
    _share_room,
    crown_normals,
    grown_parts,
    leaf_lean,
    leaf_part_name,
    leaf_tris,
    limb_tris,
    round_sides,
    tube_tris,
)
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.shape import (
    LOBE_OF_ROOT_RADIUS,
    LOBE_SPREAD,
    MERGE_OFFSET_OF_RADIUS,
    MERGE_RADIUS_STEP,
)
from model_kit.core.grow.skeleton import CONE, DOME, Skeleton, limb_at, load_skeleton
from model_kit.core.sight import Sight
from model_kit.core.tolerances import WELD_OF_PIECE_SIZE
from model_kit.core.units import load_style

from ..checks.branching.branching_scene import (
    UP,
    Limb,
    line,
    make_skeleton,
    seat,
    small_limbs,
    tube_skeleton,
    v2_skeleton,
)

REPO = Path(__file__).resolve().parents[3]
UPM = 0.5
STYLE = REPO / "bars" / "02-redbud" / "style.json"
LEAN = LeafLean(toward_out=0.8, centre_drop=0.35)  # the leaf lean values of the Redbud bar's style


@pytest.fixture(scope="module")
def rules():
    return load_rules(TINY_SPECIES["tiny-spiral"], kit=KIT_RULES)


def tree(rules, seed=1, height_m=1.5, lite=False) -> Skeleton:
    return grow_buds(rules, piece="Tiny", height_m=height_m, seed=seed, lite=lite,
                     units_per_metre=UPM)


@pytest.fixture(scope="module")
def seed1(rules) -> Skeleton:
    return tree(rules)


BENT = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.3, 0.9, 0.0], [0.8, 1.1, 0.2]])
RADII = np.array([0.12, 0.1, 0.07, 0.04])


@pytest.mark.parametrize("tip", [CONE, DOME])
def test_tube_closed_and_outward(tip):
    tris = tube_tris(BENT, RADII, 7, tip)
    assert geom.is_closed(tris)
    assert geom.signed_volume(tris) > 0.0


@pytest.mark.parametrize(("tip", "tip_tris"), [(CONE, 1), (DOME, 5)])
def test_tube_triangle_count(tip, tip_tris):
    n, s = len(BENT), 7
    assert len(tube_tris(BENT, RADII, s, tip)) == 2 * s * (n - 1) + (s - 2) + tip_tris * s


def test_tube_repeated_point_stays_finite_and_closed():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]])
    tris = tube_tris(pts, np.array([0.1, 0.1, 0.08, 0.05]), 6, CONE)
    assert np.isfinite(tris).all()


def test_leaf_counts(rules, seed1):
    assert len(leaf_tris(seed1, 0)) == len(seed1.outline) + 2
    lite = tree(rules, lite=True)
    assert len(leaf_tris(lite, 0)) == len(lite.outline)


def test_leaf_faces_along_normal(seed1):
    for i in (0, 1, len(seed1.host) - 1):
        tris = leaf_tris(seed1, i)
        face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        assert (face @ seed1.normal[i] > 0.0).all()


def test_grown_parts_names(seed1):
    parts = grown_parts(seed1, None)
    made = [name for i, name in enumerate(seed1.level_names) if np.any(seed1.level == i)]
    assert list(parts) == [*made, *(leaf_part_name(s) for s in seed1.shades)]
    assert len(seed1.level_names) == 5
    assert leaf_part_name("a") == "Leaves_a"


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_small_tree_finite_and_closed(rules, seed):
    sk = tree(rules, seed=seed, height_m=0.5)
    for name, tris in grown_parts(sk, None).items():
        assert np.isfinite(tris).all(), name
    for i in range(len(sk.limb_id)):
        tris = tube_tris(sk.limb_points(i), sk.limb_radii(i), int(sk.sides[sk.level[i]]),
                         int(sk.tip[i]))
        assert np.isfinite(tris).all()
        assert geom.is_closed(tris), sk.limb_id[i]


def test_crown_normals_lean_out(seed1):
    # a ring of horizontal leaves round the vertical axis, each face pointing up
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    tris = []
    for a in angles:
        c = 10.0 * np.array([np.cos(a), 0.0, np.sin(a)])
        up = np.array([0.0, 1.0, 0.0])
        out = c / np.linalg.norm(c)
        p0, p1, p2 = c, c + 0.5 * out, c + 0.5 * np.cross(up, out)
        tris.append([p0, p1, p2])
    tris = np.array(tris)
    n = crown_normals({"Leaves_a": tris}, LEAN)["Leaves_a"]
    allv = tris.reshape(-1, 3)
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    centre = (lo + hi) / 2.0
    centre[1] -= LEAN.centre_drop * (hi[1] - lo[1])
    radial = tris - centre
    radial /= np.linalg.norm(radial, axis=2, keepdims=True)
    assert np.isfinite(n).all()
    assert ((n * radial).sum(axis=2) > 0.75).all()
    assert np.allclose(np.linalg.norm(n, axis=2), 1.0)


def _crown_normals(tris_by_part, toward_out, centre_drop):
    """The crown-normal formula, written out again as the reference."""
    allv = np.concatenate([t.reshape(-1, 3) for t in tris_by_part.values()])
    lo, hi = allv.min(axis=0), allv.max(axis=0)
    centre = (lo + hi) / 2.0
    centre[1] -= centre_drop * (hi[1] - lo[1])
    out = {}
    for name, tris in tris_by_part.items():
        face = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        face /= np.linalg.norm(face, axis=1, keepdims=True)
        radial = tris - centre
        radial /= np.linalg.norm(radial, axis=2, keepdims=True)
        mixed = toward_out * radial + (1.0 - toward_out) * face[:, None, :]
        out[name] = mixed / np.linalg.norm(mixed, axis=2, keepdims=True)
    return out


def test_lean_comes_from_style(seed1):
    lean = leaf_lean(load_style(STYLE))
    assert lean == LeafLean(toward_out=0.8, centre_drop=0.35)
    leaves = {k: v for k, v in grown_parts(seed1, None).items() if k.startswith("Leaves_")}
    assert leaves
    got = crown_normals(leaves, lean)
    want = _crown_normals(leaves, 0.8, 0.35)
    assert list(got) == list(want)
    for part in want:
        assert np.allclose(got[part], want[part], atol=1e-12), part


def test_a_different_lean_changes_the_normals(seed1):
    leaves = {k: v for k, v in grown_parts(seed1, None).items() if k.startswith("Leaves_")}
    flat = crown_normals(leaves, LeafLean(toward_out=0.0, centre_drop=0.35))
    out = crown_normals(leaves, LEAN)
    assert any(not np.allclose(flat[k], out[k]) for k in out)


@pytest.mark.parametrize("dials", [{}, {"leaves": None}])
def test_missing_lean_stops_the_build(dials):
    style = dataclasses.replace(load_style(STYLE), dials=dials)
    with pytest.raises(ManifestError, match=r"dials\.leaves"):
        leaf_lean(style)


@pytest.mark.parametrize("leaves", [{"centre_drop": 0.35, "why": "x"},
                                    {"normal_lean": 0.8, "why": "x"},
                                    {"normal_lean": "0.8", "centre_drop": 0.35, "why": "x"},
                                    {"normal_lean": 1.5, "centre_drop": 0.35, "why": "x"}])
def test_a_bad_lean_cannot_make_a_style(leaves):
    """The lean is checked when the `Style` is built, so `leaf_lean` never meets a bad one."""
    with pytest.raises(ManifestError, match=r"dials\.leaves"):
        dataclasses.replace(load_style(STYLE), dials={"leaves": leaves})


# --------------------------------------------------------------------- the original mesh

FIXTURES = Path(__file__).parent / "fixtures"
TINY_SPECIES = {"tiny-buds": FIXTURES / "tiny-buds.rules.json",
                "tiny-spiral": FIXTURES / "tiny-spiral.rules.json"}
REDBUD_SEED1 = FIXTURES / "redbud-seed1.skeleton.npz"  # a Redbud skeleton for the original mesh

# The triangles of the Redbud skeleton as `grown_parts` makes them: per part, (triangle count,
# sha256 of the corners rounded to 9 places). Only the full tree is held, not a lite one.
BAR02_MESH = {
    "Redbud": {
        "Trunk_Roots": (780, "1e152c1fb546bf3726b867cb40aeb9899705136acb64752380ef9e769b5b5260"),
        "Leaders": (372, "8f539c39b4b09233f31a4f7e36b15516b9be7e1352be4a81de50013affbb4a7d"),
        "Branches": (1886, "ccedef15cb8ab983e5d421174de65f02311f119e1bafb8800a8b152e57bc4986"),
        "Twigs": (5970, "8a32c6104b6b0d86bab8bd535eb73c0495901957f2f261b3cf33bded8c7c78c2"),
        "Shoots": (7594, "6203948814cc17665e7e0510b0183b43b861f25ec457576a05e74f0edfd6e310"),
        "Leaves_Leaf": (9612, "89d4bbcf2aecd87455fbe018ea75f6777c12bcc1acbdf86daa5d3d0b202ce759"),
        "Leaves_LeafB": (16404, "8fe47fa7b3c26e9043122f9b8bf25a6893a0e6ac4169c12dda3f932e63bf6bc6"),
        "Leaves_LeafC": (9108, "3949cf0b4d6041f0b51a8d689488066de9d24d192f9a65e4427f52f3b8237df7"),
    },
}


def digest(tris: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.round(tris, 9) + 0.0).tobytes()).hexdigest()


@pytest.fixture(scope="module", params=sorted(TINY_SPECIES))
def tiny_tree(request) -> Skeleton:
    rules = load_rules(TINY_SPECIES[request.param], kit=KIT_RULES)
    return grow_buds(rules, piece="Tiny", height_m=1.5, seed=1, lite=False, units_per_metre=UPM)


ROOT_SEAT_T = 0.03  # between the trunk's first two points, so no point sits at the seat
ROOT_RADIUS_SHARE = 0.5  # of the trunk radius at the seat (prototype trees: 0.51)
ROOT_DROP = 0.4  # a root leaves 22 deg below horizontal, 112 deg from the trunk axis (B3: 110-117)


def rooted_tree(azimuths_deg=(17.0, 140.0, 250.0), *, mesh: str | None = MESH_V1) -> Skeleton:
    """A buds-v1 skeleton made by hand: a straight trunk on its own, one buried root per azimuth
    (about the trunk axis), none of the root azimuths on a ring corner."""
    limbs = small_limbs()[:1]
    for k, az in enumerate(azimuths_deg):
        base = seat(limbs, 0, ROOT_SEAT_T)
        radius = limb_at(limbs[0].pts, limbs[0].radii, ROOT_SEAT_T)[1]
        d = (np.cos(np.radians(az)), -ROOT_DROP, np.sin(np.radians(az)))
        limbs.append(Limb(f"root.{k}", 0, 0, ROOT_SEAT_T, line(base, d, 0.3),
                          np.linspace(ROOT_RADIUS_SHARE * radius, 0.009, 5), buried=True))
    sk = make_skeleton(limbs, [])
    if mesh is not None:
        sk = dataclasses.replace(sk, info={**sk.info, "mesh": mesh})
    return sk


@pytest.mark.parametrize("name", sorted(BAR02_MESH))
def test_old_skeletons_keep_their_mesh(name):
    sk = load_skeleton(REDBUD_SEED1)
    assert sk.info.get("mesh") is None
    parts = grown_parts(sk, None)
    assert list(parts) == list(BAR02_MESH[name])
    got = {part: (len(tris), digest(tris)) for part, tris in parts.items()}
    assert got == BAR02_MESH[name]


def ring_gap_deg(corner: np.ndarray, centre: np.ndarray, azimuth_deg: float) -> float:
    """The angle about the vertical between a ring corner and a root azimuth, 0 to 180."""
    off = corner - centre
    got = np.degrees(np.arctan2(off[2], off[0]))
    return abs((got - azimuth_deg + 180.0) % 360.0 - 180.0)


def base_ring(sk: Skeleton, i: int = 0):
    """(corners, centre, radius) of limb i's first ring in the mesh `limb_tris` makes."""
    tris = limb_tris(sk, i, None)
    sides = int(sk.sides[sk.level[i]])
    return tris[:sides, 0], sk.limb_points(i)[0], float(sk.limb_radii(i)[0])


def test_lobes_follow_the_roots():
    sk = rooted_tree((17.0, 140.0, 250.0))
    corners, centre, radius = base_ring(sk)
    reach = np.linalg.norm(corners - centre, axis=1)
    for az in (17.0, 140.0, 250.0):
        gaps = np.array([ring_gap_deg(c, centre, az) for c in corners])
        assert reach[np.argmin(gaps)] > 1.05 * radius, az
    # Without the buds-v1 mesh the same ring is round.
    old_corners, _, _ = base_ring(rooted_tree(mesh=None))
    assert np.allclose(np.linalg.norm(old_corners - centre, axis=1), radius)


def test_lobe_is_one_bump_and_round_away_from_it():
    az = 17.0
    sk = rooted_tree((az,))
    corners, centre, radius = base_ring(sk)
    reach = np.linalg.norm(corners - centre, axis=1)
    gaps = np.array([ring_gap_deg(c, centre, az) for c in corners])
    order = np.argsort(gaps)
    assert (np.diff(reach[order]) <= 1e-12).all()  # reach falls as the corner turns from the root
    spread_deg = np.degrees(LOBE_SPREAD * np.arctan2(ROOT_RADIUS_SHARE * radius, radius))
    away = gaps > spread_deg + 1.0
    assert away.any()
    assert np.allclose(reach[away], radius, atol=1e-12)
    assert reach.max() <= radius * (1.0 + LOBE_OF_ROOT_RADIUS * ROOT_RADIUS_SHARE) + 1e-12


def test_lobed_trunk_is_closed_and_outward():
    sk = rooted_tree()
    tris = limb_tris(sk, 0, None)
    assert np.isfinite(tris).all()
    assert geom.is_closed(tris)
    assert geom.signed_volume(tris) > 0.0
    assert geom.signed_volume(tris) > geom.signed_volume(limb_tris(rooted_tree(mesh=None), 0, None))


def test_a_limb_without_roots_has_no_lobes(tiny_tree):
    # The trunk of a buds-v1 tree with no roots: every ring is as round as the skeleton's radius.
    # (The grown tiny trees have roots, so the trunk is made as if it had none.)
    tris = _limb_tris(tiny_tree, 0, [], None)
    sides = int(tiny_tree.sides[tiny_tree.level[0]])
    corners = tris[:sides, 0]
    reach = np.linalg.norm(corners - tiny_tree.limb_points(0)[0], axis=1)
    assert np.allclose(reach, tiny_tree.limb_radii(0)[0])


def test_grown_roots_pull_lobes_out_of_the_trunk(tiny_tree):
    """Real roots: the grown trunk bulges toward them, where the same trunk made as if
    it had none is round, and it stays closed with outward faces."""
    kids = [int(c) for c in np.flatnonzero(tiny_tree.parent == 0)]
    assert tiny_tree.buried.sum() >= 4
    lobed = _limb_tris(tiny_tree, 0, kids, None)
    round_trunk = _limb_tris(tiny_tree, 0, [], None)
    assert np.isfinite(lobed).all() and geom.is_closed(lobed)
    assert geom.signed_volume(lobed) > geom.signed_volume(round_trunk) > 0.0
    ring = tiny_tree.limb_points(0)[1]  # the soil point: the lobes are at their fullest here
    reach = np.linalg.norm(lobed.reshape(-1, 3) - ring, axis=1)
    reach_round = np.linalg.norm(round_trunk.reshape(-1, 3) - ring, axis=1)
    near = reach < 1.5 * tiny_tree.limb_radii(0)[1] * (1.0 + LOBE_OF_ROOT_RADIUS)
    assert reach[near].max() > reach_round[reach_round < 1.5 * tiny_tree.limb_radii(0)[1]].max()


def test_merge_collinear_is_repeatable(tiny_tree):
    for sk in (tiny_tree, rooted_tree()):
        first, second = grown_parts(sk, None), grown_parts(sk, None)
        assert list(first) == list(second)
        for name in first:
            assert np.array_equal(first[name], second[name]), name
    again = grow_buds(load_rules(TINY_SPECIES["tiny-buds"], kit=KIT_RULES), piece="Tiny",
                      height_m=1.5, seed=1, lite=False, units_per_metre=UPM)
    base = grow_buds(load_rules(TINY_SPECIES["tiny-buds"], kit=KIT_RULES), piece="Tiny",
                     height_m=1.5, seed=1, lite=False, units_per_metre=UPM)
    for name, tris in grown_parts(base, None).items():
        assert np.array_equal(tris, grown_parts(again, None)[name]), name


def test_merge_keeps_seats(tiny_tree):
    for sk in (tiny_tree, rooted_tree()):
        kids = [np.flatnonzero(sk.parent == i) for i in range(len(sk.limb_id))]
        for i, kid in enumerate(kids):
            if not len(kid):
                continue
            pts, radii = sk.limb_points(i), sk.limb_radii(i)
            seated = _seated_points(pts, sk.t[kid])
            assert seated.any()
            kept = _merge_collinear(pts, radii, seated)
            assert seated[kept].sum() == seated.sum(), (sk.limb_id[i], "a seated point was merged")
            axis = pts[kept]
            for c in kid:  # the child's base still lies on the merged axis
                base = sk.limb_points(int(c))[0]
                d = _distance_to_polyline(base, axis)
                assert d <= 1e-9 * max(1.0, float(np.abs(pts).max())), sk.limb_id[c]


def _distance_to_polyline(q: np.ndarray, pts: np.ndarray) -> float:
    a, ab = pts[:-1], pts[1:] - pts[:-1]
    den = np.maximum((ab * ab).sum(axis=1), 1e-300)
    f = np.clip(((q - a) * ab).sum(axis=1) / den, 0.0, 1.0)
    return float(np.sqrt(((a + ab * f[:, None] - q) ** 2).sum(axis=1)).min())


def test_merge_drops_collinear_points_and_keeps_corners():
    none = np.zeros(6, dtype=bool)
    straight = line((0, 0, 0), UP, 1.0, 5)
    assert list(_merge_collinear(straight, np.linspace(0.05, 0.03, 6), none)) == [0, 5]
    bent = np.concatenate([line((0, 0, 0), UP, 0.5, 2), line((0, 0.5, 0), (1, 0, 0), 0.5, 2)[1:]])
    assert list(_merge_collinear(bent, np.full(5, 0.02), np.zeros(5, dtype=bool))) == [0, 2, 4]
    stepped = np.array([0.05, 0.05, 0.05, 0.02, 0.02, 0.02])  # a sudden thinning is kept
    assert list(_merge_collinear(straight, stepped, none)) != [0, 5]
    assert {2, 3} & set(_merge_collinear(straight, stepped, none))


def test_merge_follows_a_curve_instead_of_erasing_it():
    # A quarter circle of 48 points: each turn is 1.9 deg, the whole is 90 deg. Two steps of it
    # sag 0.0005 of the arc radius, inside the band (0.1 of a 0.02 tube radius = 0.002).
    ang = np.linspace(0.0, np.pi / 2, 48)
    arc = np.stack([np.sin(ang), 1.0 - np.cos(ang), np.zeros_like(ang)], axis=1)
    radii = np.full(48, 0.02)
    kept = _merge_collinear(arc, radii, np.zeros(48, dtype=bool))
    assert 8 <= len(kept) < 48  # some merged, but the 90 deg of turn is not flattened away
    for q in arc:  # nothing the merge dropped lies farther from the axis than its band
        assert _distance_to_polyline(q, arc[kept]) <= MERGE_OFFSET_OF_RADIUS * 0.02 + 1e-12
    assert MERGE_RADIUS_STEP > 0.0


def test_tri_count_reported(tiny_tree, record_property):
    """Report-only (no pass or fail line): the grown tiny tree's triangle count, with the original
    mesh beside it. Run with `-s` to see it."""
    old = dataclasses.replace(tiny_tree, info={k: v for k, v in tiny_tree.info.items()
                                               if k != "mesh"})
    count = sum(len(t) for t in grown_parts(tiny_tree, None).values())
    before = sum(len(t) for t in grown_parts(old, None).values())
    name = tiny_tree.info["rules"]
    print(f"\ntri count {name} full: {count} (original mesh: {before}); "
          f"limbs {len(tiny_tree.limb_id)}, points {len(tiny_tree.pts)}")
    record_property("tri_count", count)
    record_property("tri_count_redbud_seed1_mesh", before)


def test_tri_count_reported_lite(record_property):
    for name, path in TINY_SPECIES.items():
        sk = grow_buds(load_rules(path, kit=KIT_RULES), piece="Tiny", height_m=1.5, seed=1,
                       lite=True, units_per_metre=UPM)
        old = dataclasses.replace(sk, info={k: v for k, v in sk.info.items() if k != "mesh"})
        count = sum(len(t) for t in grown_parts(sk, None).values())
        before = sum(len(t) for t in grown_parts(old, None).values())
        print(f"\ntri count {name} lite: {count} (original mesh: {before})")
        record_property(f"tri_count_{name}_lite", count)


def test_facet_turns_clear_the_crease():
    """A tube's facet turn (360 / sides) must lie outside the crease angle plus or minus the
    rounding band at that edge: `edges.rounding_band_deg` at `edges.float32_step`, welded at
    `WELD_OF_PIECE_SIZE` times the tree's size, as the detail check reads a part."""
    style = load_style(REPO / "bars" / "02-redbud" / "style.json")
    crease = float(style.dials["edges"]["crease_deg"])
    for name, path in TINY_SPECIES.items():
        sk = grow_buds(load_rules(path, kit=KIT_RULES), piece="Tiny", height_m=1.5, seed=1,
                       lite=False, units_per_metre=UPM)
        parts = grown_parts(sk, None)
        size = float(np.ptp(np.concatenate([p.reshape(-1, 3) for p in parts.values()]),
                            axis=0).max())
        for level, part in parts.items():
            if level not in sk.level_names:
                continue
            sides = int(sk.sides[sk.level_names.index(level)])
            found = edges.interior_edges(part, WELD_OF_PIECE_SIZE * size)
            band, _capped = edges.rounding_band_deg(found.points, found,
                                                    edges.float32_step(found.points), crease)
            turn = 360.0 / sides
            assert (np.abs(turn - crease) > band).all(), (name, level, turn, crease)


# --- the tube resolution from the view ----------------------------------

BUDS_V1_TINY = FIXTURES / "buds-v1-tiny.skeleton.npz"  # tiny-spiral, 1 m, seed 1, grown as buds-v1
# Its parts as `grown_parts` made them before buds-v2: (triangle count, sha256 of the corners
# rounded to 9 places), taken from the unchanged mesh.
BUDS_V1_MESH = {
    "Trunk_Roots": (904, "dac9f82951a80bf1021155980d4988de7075e9fda7b7c4e38e90ad7b2c078223"),
    "Leaders": (1188, "4cee9a1f4987bdba6a452613edadbce015d790df6f3a85839de304fa095a871e"),
    "Branches": (934, "fdafa0c2eb8173f0fcb9f5f789d31758e8e0adadfe82295139842190888cdc51"),
    "Leaves_Leaf": (2100, "3ea3c1c0170e8ee8f96301cb2510be2c4a86dec13eebccfcc3f02e3be5e31089"),
}
CREASE = 35.0
SIGHT = Sight(gap=1e-3, crease_deg=CREASE)  # a tube of 0.04 needs 15 sides here, one of 0.004 three


def smooth_gap(tris: np.ndarray, crease: float = CREASE) -> float:
    """The worst gap `detail` would judge in `tris`: the largest sagitta over the edges that turn
    by less than the crease angle and more than float noise."""
    size = float(np.ptp(tris.reshape(-1, 3), axis=0).max())
    found = edges.interior_edges(tris, WELD_OF_PIECE_SIZE * size)
    keep = (found.turn_deg < crease) & (found.turn_deg > 1e-3)
    return float(edges.sagittas(tris, found)[keep].max())


@pytest.mark.parametrize("sight", [None, SIGHT, Sight(gap=0.5, crease_deg=10.0)])
def test_old_mesh_versions_mesh_byte_equal(sight):
    """The original and buds-v1 branches read no sight and keep `Skeleton.sides`: a skeleton of
    either meshes to the same bytes whatever sight it is given."""
    grown = grown_parts(load_skeleton(REDBUD_SEED1), sight)
    assert {k: (len(v), digest(v)) for k, v in grown.items()} == BAR02_MESH["Redbud"]
    v1 = load_skeleton(BUDS_V1_TINY)
    assert v1.info["mesh"] == MESH_V1 == "buds-v1"
    assert {k: (len(v), digest(v)) for k, v in grown_parts(v1, sight).items()} == BUDS_V1_MESH


def test_a_buds_v2_skeleton_needs_a_sight_and_an_unknown_mesh_is_refused():
    sk = tube_skeleton(0.04)
    assert sk.info["mesh"] == MESH_VERSION == "buds-v2"
    with pytest.raises(ManifestError, match="sight"):
        grown_parts(sk, None)
    odd = dataclasses.replace(sk, info={**sk.info, "mesh": "buds-v9"})
    with pytest.raises(ManifestError, match="buds-v9"):
        grown_parts(odd, SIGHT)


@pytest.mark.parametrize("radius", [0.04, 0.1])
def test_a_buds_v2_tube_has_the_sides_the_sagitta_decides(radius):
    sk = tube_skeleton(radius)
    n = edges.segments_for(radius, 360.0, SIGHT.gap)
    assert n > round_sides(CREASE)  # the sagitta, not the round floor, sets this one
    tris = limb_tris(sk, 0, SIGHT)
    points = sk.limb_points(0)[[0, -1]]
    assert np.array_equal(tris, tube_tris(points, np.full(2, radius), n, CONE))
    assert smooth_gap(tris) <= SIGHT.gap < smooth_gap(tube_tris(points, np.full(2, radius), n - 1,
                                                                CONE))


def test_thin_limb_round_floor_has_no_edge_over_crease():
    """A tube so thin that the sagitta would let it have three sides still has more than
    360 / crease_deg: no edge along it turns by the crease angle (which would draw a line)."""
    radius = 0.004
    assert edges.segments_for(radius, 360.0, SIGHT.gap) < 360.0 / CREASE
    tris = limb_tris(tube_skeleton(radius), 0, SIGHT)
    found = edges.interior_edges(tris, 1e-9)
    along = np.abs((found.points[found.v[:, 1]] - found.points[found.v[:, 0]]) @ UP)
    side = along > 0.99 * np.linalg.norm(found.points[found.v[:, 1]] - found.points[found.v[:, 0]],
                                         axis=1)
    side &= found.turn_deg > 1.0
    assert side.sum() == round_sides(CREASE)  # one edge for each of the tube's sides
    assert found.turn_deg[side].max() < CREASE
    assert 360.0 / (round_sides(CREASE) - 1) >= CREASE  # one side fewer would draw a line
    assert smooth_gap(tris) <= SIGHT.gap  # and it is more than fine enough


def test_round_sides_is_over_360_over_crease():
    assert round_sides(35.0) == 11 and 360.0 / 11 < 35.0 <= 360.0 / 10
    assert round_sides(30.0) == 13  # 360 / 30 is whole: more than that, so one more
    assert round_sides(179.0) == 3


def bent_path(kink_deg: float, radius: float = 0.004, step: float = 0.2):
    """Points and radii of a straight run, a kink of `kink_deg`, and another run."""
    turn = np.radians(kink_deg)
    d2 = np.array([np.sin(turn), np.cos(turn), 0.0])
    pts = np.array([[0, 0, 0], [0, step, 0], [0, 2 * step, 0]] + [
        d2 * step * k + [0, 2 * step, 0] for k in (1, 2)], dtype=float)
    return pts, np.full(len(pts), radius)


def joint_gaps(pts: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(the gap `detail` would find at each interior joint of the path, its turn in degrees), with
    the radius as a fourth coordinate so a step of the wall counts as a turn."""
    path = np.column_stack([pts, radii])
    seg = np.diff(path, axis=0)
    length = np.linalg.norm(seg, axis=1)
    unit_ = seg / length[:, None]
    turn = np.degrees(np.arccos(np.clip((unit_[:-1] * unit_[1:]).sum(axis=1), -1.0, 1.0)))
    return edges.chord_sagitta(np.minimum(length[:-1], length[1:]), turn), turn


@pytest.mark.parametrize("kink", [12.0, 50.0, 100.0])
def test_a_kink_that_shows_is_rounded_with_no_joint_over_the_gap_or_the_crease(kink):
    pts, radii = bent_path(kink)
    assert joint_gaps(pts, radii)[0].max() > SIGHT.gap  # as it stands, the kink shows
    rounded, rr = _round_bends(pts, radii, SIGHT)
    gaps, turns = joint_gaps(rounded, rr)
    assert len(rounded) > len(pts)
    assert gaps.max() <= SIGHT.gap and turns.max() < CREASE
    assert np.array_equal(rounded[0], pts[0]) and np.array_equal(rounded[-1], pts[-1])
    assert rr.min() > 0.0
    # the arc cuts the corner by less than two tube radii (its own radius is two, FILLET_...)
    assert np.linalg.norm(rounded - pts[2], axis=1).min() < 2.0 * radii[0]


@pytest.mark.parametrize(("radius", "kink"), [(0.01, 50.0), (0.02, 90.0)])
def test_a_filleted_tube_has_no_wall_edge_over_the_gap(radius, kink):
    """The wall on the outer side of a bend runs at the arc's radius plus the tube's, so the arc's
    chords are counted for that. Read with detail's own sagitta on the meshed tube, not on the
    axis: no smooth edge of the tube, kink or not, shows a gap."""
    fine = Sight(gap=1e-4, crease_deg=CREASE)
    pts, radii = bent_path(kink, radius)
    sk = v2_skeleton([Limb("trunk", 0, -1, 0.0, pts, radii, rooted=True)])
    tris = limb_tris(sk, 0, fine)
    assert len(tris) > len(limb_tris(tube_skeleton(radius), 0, fine))  # the kink was rounded
    assert smooth_gap(tris) <= fine.gap * (1.0 + 1e-6)


def test_a_known_over_gap_flare_is_never_returned(monkeypatch):
    """When more flare steps can no longer add a ring and the profile still shows a gap, the
    search raises naming the piece and the gap, instead of returning the mesh."""
    sk = rooted_tree((17.0,), mesh=MESH_VERSION)
    pts, radii = sk.limb_points(0), sk.limb_radii(0)
    lobes = _lobes(sk, pts, radii, [1])
    assert lobes
    wide = Sight(gap=lobes[0].height, crease_deg=CREASE)  # no step is longer than this
    monkeypatch.setattr(mesh_module, "_lobed_gaps", lambda *a, **k: (0.0, 10.0 * wide.gap))
    with pytest.raises(ManifestError, match=r"Tiny|trunk|flare"):
        _lobed_rings(pts, radii, lobes, wide, "skeleton 'Tiny', limb trunk")


def test_a_step_of_the_radius_is_rounded_like_a_bend():
    pts = line((0, 0, 0), UP, 0.6, 4)
    radii = np.array([0.01, 0.01, 0.006, 0.006, 0.006])
    fine = Sight(gap=1e-4, crease_deg=CREASE)
    assert joint_gaps(pts, radii)[0].max() > fine.gap
    rounded, rr = _round_bends(pts, radii, fine)
    assert joint_gaps(rounded, rr)[0].max() <= fine.gap
    assert len(rounded) > len(pts)


def test_a_straight_run_and_a_kink_that_cannot_show_are_left_alone():
    pts = line((0, 0, 0), UP, 0.6, 4)
    radii = np.full(5, 0.01)
    same, same_r = _round_bends(pts, radii, SIGHT)
    assert np.array_equal(same, pts) and np.array_equal(same_r, radii)
    gentle, gentle_r = bent_path(1.0)  # a 1 degree kink: its gap is under the 1 mm gap
    assert joint_gaps(gentle, gentle_r)[0].max() < SIGHT.gap
    kept, kept_r = _round_bends(gentle, gentle_r, SIGHT)
    assert np.array_equal(kept, gentle) and np.array_equal(kept_r, gentle_r)


# --- a kink too tight for half of its bands -------------------------------------------

FINE = Sight(gap=1e-4, crease_deg=CREASE)


def short_band_limb(radius: float, kink_deg: float, band: float) -> Skeleton:
    """A buds-v2 trunk of `radius`: a long run, a kink of `kink_deg` (under the crease angle, so a
    kept kink reads as a gap in `detail`, not as a drawn line), a band only `band` long, a 6 degree
    joint that does not show but that the merge keeps, and a long run."""
    def turned(deg: float) -> np.ndarray:
        return np.array([np.sin(np.radians(deg)), np.cos(np.radians(deg)), 0.0])

    p1 = np.array([0.0, 0.3, 0.0])
    p2 = p1 + band * turned(kink_deg)
    p3 = p2 + 0.3 * turned(kink_deg + 6.0)
    pts = np.array([[0.0, 0.0, 0.0], p1, p2, p3])
    return v2_skeleton([Limb("trunk", 0, -1, 0.0, pts, np.full(4, radius), tip=CONE,
                             rooted=True)])


# (radius, kink, band): an arc on half the short band would be no wider than the tube, so a half-band
# rule would keep the kink; the whole band leaves room for an arc wider than the tube.
ROOMY = [(0.01, 30.0, 0.004), (0.02, 25.0, 0.007)]
# the same tubes with a band too short for any arc wider than the tube: r tan(kink / 2) is over it
CRAMPED = [(0.01, 30.0, 0.002), (0.02, 25.0, 0.003)]


@pytest.mark.parametrize(("radius", "kink", "band"), ROOMY)
def test_a_kink_with_too_little_room_for_half_its_bands_is_rounded_quiet(radius, kink, band):
    """Half the short band is not enough (`0.5 band / tan(kink / 2)` is not over the radius), but
    the joint beyond it needs only the little it keeps clear, so the arc takes the rest. Read with
    detail's own sagittas on the meshed tube: no smooth edge shows a gap. (A half-band rule would keep
    this kink, which shows.)"""
    half = np.tan(np.radians(kink) / 2.0)
    assert 0.5 * band / half <= radius < band / half
    sk = short_band_limb(radius, kink, band)
    pts, radii = sk.limb_points(0), sk.limb_radii(0)
    gaps, _ = joint_gaps(pts, radii)
    assert gaps[0] > FINE.gap >= gaps[1]  # the kink shows as it stands, the joint beyond does not
    tris = limb_tris(sk, 0, FINE)
    assert len(tris) > len(limb_tris(tube_skeleton(radius), 0, FINE))  # the kink was rounded
    assert smooth_gap(tris) <= FINE.gap * (1.0 + 1e-6)


@pytest.mark.parametrize(("radius", "kink", "band"), CRAMPED)
def test_a_kink_with_no_room_for_an_arc_wider_than_the_tube_stays_a_sharp_crease(radius, kink,
                                                                                 band):
    """No arc on these bands is wider than the tube, so rounding would fold its inner wall: the bend
    is kept as the skeleton has it, a sharp crease (a limb can turn sharply). It builds, and detail
    judges it like any other edge: this one turns under the crease angle, so detail reads its gap,
    over what the eye resolves at this sight."""
    assert band < radius * np.tan(np.radians(kink) / 2.0)
    sk = short_band_limb(radius, kink, band)
    pts, radii = sk.limb_points(0), sk.limb_radii(0)
    axis, _ = _round_bends(pts, radii, FINE)
    assert any(np.array_equal(q, pts[1]) for q in axis)  # the node itself is a ring: no arc
    assert smooth_gap(limb_tris(sk, 0, FINE)) > FINE.gap


def two_bends_on_a_short_band() -> Skeleton:
    """A lite Redbud's trunk shape, made by hand: a 40 degree bend, a band of
    0.016, and a 6 degree bend back, with the radius stepping down 25% along the short band (a
    seat). Both bends show, and the band cannot hold both reaches that keep them from folding
    (`0.04 tan 20` and `0.04 tan 3`)."""
    def turned(deg: float) -> np.ndarray:
        return np.array([np.sin(np.radians(deg)), np.cos(np.radians(deg)), 0.0])

    p1 = np.array([0.0, 0.3, 0.0])
    p2 = p1 + 0.016 * turned(40.0)
    pts = np.array([[0.0, 0.0, 0.0], p1, p2, p2 + 0.3 * turned(34.0)])
    return v2_skeleton([Limb("trunk", 0, -1, 0.0, pts, np.array([0.04, 0.04, 0.03, 0.03]),
                             tip=CONE, rooted=True)])


def test_two_bends_that_crowd_one_band_build_with_the_sharper_one_a_crease():
    """The bend that cannot be rounded stays a sharp crease, and its neighbour, given the room
    back, is rounded. Detail reads the tube as it is: the crease turns the wall by more than the
    crease angle on its sharp side (a drawn line), its flanks turn less and show a gap at this near
    sight, and every edge over the gap touches the crease's own ring."""
    sk = two_bends_on_a_short_band()
    pts, radii = sk.limb_points(0), sk.limb_radii(0)
    least = 0.04 * (np.tan(np.radians(20.0)) + np.tan(np.radians(3.0)))
    assert np.linalg.norm(pts[2] - pts[1]) < least
    axis, _ = _round_bends(pts, radii, FINE)
    assert any(np.array_equal(q, pts[1]) for q in axis)  # the 40 degree bend is the skeleton's
    assert not any(np.array_equal(q, pts[2]) for q in axis)  # the 6 degree one was rounded
    tris = limb_tris(sk, 0, FINE)
    found = edges.interior_edges(tris, 1e-9)
    ends = found.points[found.v]
    on_crease = np.all(np.abs(np.linalg.norm(ends - pts[1], axis=2) - radii[1]) < 1e-9, axis=1)
    assert on_crease.sum() > 0 and found.turn_deg[on_crease].max() >= CREASE
    touches = np.any(np.abs(np.linalg.norm(ends - pts[1], axis=2) - radii[1]) < 1e-9, axis=1)
    over = (found.turn_deg < CREASE) & (edges.sagittas(tris, found) > FINE.gap)
    assert over.any() and touches[over].all()


def crowded_pair(radius: float, bend_deg: float) -> tuple[Skeleton, float]:
    """(a buds-v2 trunk with two equal bends of `bend_deg` in an S on one short band, the band):
    the band is too short for both arcs not to fold (`2 r tan(bend / 2)`), but long enough for
    one arc beside a kept kink (`r tan(bend / 2) + r sin(bend / 2)`). It is the middle of that
    window."""
    half = np.radians(bend_deg) / 2.0
    band = radius * 0.5 * ((np.tan(half) + np.sin(half)) + 2.0 * np.tan(half))

    def turned(deg: float) -> np.ndarray:
        return np.array([np.sin(np.radians(deg)), np.cos(np.radians(deg)), 0.0])

    p1 = np.array([0.0, 0.3, 0.0])
    p2 = p1 + band * turned(bend_deg)
    pts = np.array([[0.0, 0.0, 0.0], p1, p2, p2 + 0.3 * turned(0.0)])
    return v2_skeleton([Limb("trunk", 0, -1, 0.0, pts, np.full(4, radius), tip=CONE,
                             rooted=True)]), band


@pytest.mark.parametrize(("radius", "bend"), [(0.02, 60.0), (0.01, 90.0)])
def test_of_two_bends_that_crowd_one_band_only_one_is_a_crease(radius, bend):
    """Both arcs fold when they share the band, but with one bend kept the other's arc fits. So
    exactly one is the skeleton's crease, and the other is rounded quiet: every smooth edge over
    the gap, under detail's own sagittas, touches the crease's ring."""
    sk, band = crowded_pair(radius, bend)
    half = np.radians(bend) / 2.0
    assert radius * (np.tan(half) + np.sin(half)) < band < 2.0 * radius * np.tan(half)
    pts, radii = sk.limb_points(0), sk.limb_radii(0)
    axis, _ = _round_bends(pts, radii, FINE)
    kept = [k for k in (1, 2) if any(np.array_equal(q, pts[k]) for q in axis)]
    assert len(kept) == 1
    crease = kept[0]
    tris = limb_tris(sk, 0, FINE)
    found = edges.interior_edges(tris, 1e-9)
    ends = found.points[found.v]
    touches = np.any(np.abs(np.linalg.norm(ends - pts[crease], axis=2) - radius) < 1e-9, axis=1)
    over = (found.turn_deg < CREASE) & (edges.sagittas(tris, found) > FINE.gap)
    assert touches[over].all()


def test_the_bands_are_shared_first_by_what_keeps_each_arc_from_folding():
    """`_share_room`: two arcs that both want more than half a band get half each; an arc beside a
    joint that wants none of the band gets all it wants; and the reach an arc needs not to fold is
    served before any arc grows past its own."""
    length = np.array([1.0, 1.0, 1.0])
    want = np.array([0.0, 0.8, 0.8, 0.0])
    assert np.allclose(_share_room(length, want, np.zeros(4)), [0.0, 0.5, 0.5, 0.0])
    assert np.allclose(_share_room(length, np.array([0.0, 0.8, 0.0, 0.0]), np.zeros(4)),
                       [0.0, 0.8, 0.0, 0.0])
    least = np.array([0.0, 0.7, 0.1, 0.0])
    base = _share_room(length, least, np.zeros(4))
    assert np.allclose(base, least)
    # the 0.2 left on the middle band grows both by the same share (a quarter) of what each still
    # wants (0.1 and 0.7)
    assert np.allclose(_share_room(length, want, base), [0.0, 0.725, 0.275, 0.0])


def test_buds_v2_merge_drops_a_point_that_moves_the_wall_by_less_than_the_gap():
    ang = np.linspace(0.0, np.pi / 2, 48)
    arc = np.stack([np.sin(ang), 1.0 - np.cos(ang), np.zeros_like(ang)], axis=1)
    radii = np.full(48, 0.02)
    none = np.zeros(48, dtype=bool)
    coarse = _merge_collinear(arc, radii, none, 0.01)
    fine = _merge_collinear(arc, radii, none, 0.0007)
    assert len(coarse) < len(fine) < 48
    for within, kept in ((0.01, coarse), (0.0007, fine)):
        assert max(_distance_to_polyline(q, arc[kept]) for q in arc) <= within + 1e-12
    # a step in the radius over the gap is kept, one under it merges away
    stepped = line((0, 0, 0), UP, 1.0, 5)
    assert list(_merge_collinear(stepped, np.array([0.02] * 3 + [0.0199] * 3), none[:6],
                                 0.001)) == [0, 5]
    assert {2, 3} & set(_merge_collinear(stepped, np.array([0.02] * 3 + [0.01] * 3), none[:6],
                                         0.001))


def test_a_buds_v2_trunk_with_roots_is_closed_outward_and_under_the_gap():
    sk = v2_skeleton(small_limbs()[:1] + [
        Limb(f"root.{k}", 0, 0, ROOT_SEAT_T, line(seat(small_limbs()[:1], 0, ROOT_SEAT_T),
                                                 (np.cos(az), -ROOT_DROP, np.sin(az)), 0.3),
             np.linspace(0.5 * 0.0399, 0.009, 5), buried=True)
        for k, az in enumerate(np.radians((17.0, 140.0, 250.0)))])
    tris = limb_tris(sk, 0, SIGHT)
    assert np.isfinite(tris).all() and geom.is_closed(tris)
    assert geom.signed_volume(tris) > 0.0
    assert smooth_gap(tris) <= SIGHT.gap * (1.0 + 1e-6)
    plain = limb_tris(tube_skeleton(0.04), 0, SIGHT)
    assert len(tris) > len(plain)  # the lobes and the rings that carry them cost triangles
