import numpy as np

from model_kit.core.checks.branching._pieces import grown
from model_kit.core.checks.branching.skeleton_match import SkeletonMatch, _sight
from model_kit.core.grow.mesh import grown_parts
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.sight import Sight
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    SKELETON_MATCH_OF_THICKNESS,
)

from .branching_scene import (
    MIN_THICKNESS_M,
    UPM,
    grown_ctx,
    item,
    plants_caught,
    small_tree,
    tiny_ctx,
    tube_skeleton,
    viewed_ctx,
)

CHECK = SkeletonMatch()
TOL = SKELETON_MATCH_OF_THICKNESS * MIN_THICKNESS_M * UPM
LEAVES = "Leaves_Leaf"


def _shipped(edit):
    sk = small_tree()
    parts = {name: tris.copy() for name, tris in grown_parts(sk, None).items()}
    edit(parts)
    return grown_ctx(sk, parts=parts)


def _blade_centre_moved(parts, dy):
    """Every copy of the first leaf's blade centre moved dy along +Y."""
    tris = parts[LEAVES]
    centre = tris[0, 0].copy()
    hit = np.all(tris == centre, axis=-1)
    tris[hit] += np.array([0.0, dy, 0.0])


def test_clean_passes():
    assert CHECK.run(grown_ctx(small_tree())) == []
    assert CHECK.run(tiny_ctx()) == []


def test_plants_caught():
    for ctx in (grown_ctx(small_tree()), tiny_ctx()):
        caught = plants_caught(CHECK, ctx)
        assert [label.split(":")[0] for label, _ in caught] == ["vertex"]
        assert all(ok for _, ok in caught), caught


def test_calibration():
    assert CHECK.run(_shipped(lambda p: _blade_centre_moved(p, CALIBRATION_PASS * TOL))) == []
    found = CHECK.run(_shipped(lambda p: _blade_centre_moved(p, CALIBRATION_FAIL * TOL)))
    assert {f.item for f in found} == {item(LEAVES)}
    details = " ".join(f.detail for f in found)
    assert "not in the skeleton's mesh" in details  # moved away
    assert "missing from the shipped mesh" in details  # and gone from where it was


def test_skeleton_match_both_ways():
    def extra(parts):  # one corner of one blade triangle sent far off: only the mesh has it
        parts[LEAVES][1, 0] += np.array([0.0, 0.1, 0.0])

    def missing(parts):  # the first blade's fan replaced by copies of another triangle
        tris = parts[LEAVES]
        k = len(small_tree().outline)
        tris[:k] = tris[-1]

    found = CHECK.run(_shipped(extra))
    assert {f.item for f in found} == {item(LEAVES)}
    assert any("vertices not in the skeleton's mesh" in f.detail for f in found)
    assert not any("vertices of the skeleton's mesh missing" in f.detail for f in found)
    found = CHECK.run(_shipped(missing))
    assert {f.item for f in found} == {item(LEAVES)}
    assert any("vertices of the skeleton's mesh missing" in f.detail for f in found)
    assert not any("vertices not in the skeleton's mesh" in f.detail for f in found)


def test_triangle_count_differs():
    def drop(parts):
        parts["Twigs"] = parts["Twigs"][:-1]

    found = CHECK.run(_shipped(drop))
    assert item("Twigs") in [f.item for f in found]
    assert any("triangles" in f.detail for f in found)


def test_part_the_skeleton_does_not_grow():
    def add(parts):
        parts["Fruit"] = parts["Twigs"][:4].copy()

    found = CHECK.run(_shipped(add))
    assert [f.item for f in found] == [item("Fruit")]


def test_part_missing_from_the_mesh():
    found = CHECK.run(_shipped(lambda parts: parts.pop("Twigs")))
    assert [f.item for f in found] == [item("Twigs")]


def test_only_measures_named_parts():
    ctx = _shipped(lambda p: _blade_centre_moved(p, CALIBRATION_FAIL * TOL))
    assert CHECK.run(ctx, only={item("Twigs")}) == []


def test_no_skeleton_is_unmeasured():
    ctx = grown_ctx(None, listed_from=small_tree())
    assert "skeleton-match: skipped (no skeleton): grove:Tree" in CHECK.coverage(ctx)
    assert any(isinstance(line, Unmeasured) for line in CHECK.coverage(ctx))
    assert run_gate([CHECK], ctx, None).status == "partial"


def test_assembly_piece_listed_in_coverage():
    assert CHECK.coverage(grown_ctx(small_tree())) == [
        "skeleton-match: not applicable (assembly piece): grove:Crate"]


# --- a buds-v2 skeleton is re-meshed with the piece's own sight ---------------------------------

TUBE = "Trunk"


def test_skeleton_match_uses_the_same_sight():
    """The tube sides of a buds-v2 skeleton come from the sight, so the check re-meshes with the
    sight the piece's view and the style give. A mesh built with that sight matches; the same
    skeleton meshed with a sight four times coarser has fewer sides and does not."""
    sk = tube_skeleton(0.04)
    clean = viewed_ctx(sk)
    assert CHECK.run(clean) == []
    assert all(ok for _, ok in plants_caught(CHECK, clean))
    (g,) = grown(clean)
    seen = _sight(clean, g)
    assert seen == Sight(seen.gap, 35.0) and seen.gap > 0.0
    coarse = viewed_ctx(sk, meshed_with=Sight(4.0 * seen.gap, seen.crease_deg))
    found = CHECK.run(coarse)
    assert {f.item for f in found} == {item(TUBE)}
    assert "triangles" in found[0].detail  # fewer sides: the counts differ


def test_a_buds_v2_piece_with_no_view_is_unmeasured_and_the_old_meshes_need_none():
    sk = tube_skeleton(0.04)
    unseen = viewed_ctx(sk, view=None, meshed_with=Sight(1e-3, 35.0))
    assert CHECK.run(unseen) == []
    assert CHECK.measured(unseen) == set()
    assert CHECK.plants(unseen) == []
    lines = CHECK.coverage(unseen)
    assert any(isinstance(line, Unmeasured) and "grove:Tree" in line for line in lines), lines
    # a skeleton of the original mesh reads no sight: the piece is compared without a view
    assert CHECK.run(grown_ctx(small_tree())) == []
    assert CHECK.measured(grown_ctx(small_tree()))
