"""Check mirror-exact: a mirrored part is its source's exact image, triangle for triangle.

The passing cases are bar 03 as it stands (its right eye, arm, thigh and shin are mirror forms) and
the quadruped (its right legs are mirrors by their bones only). The failing case is the shins fixture:
a shin whose twin split its quads along other diagonals.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import pieces
from model_kit.core.checks.mirror_exact import MirrorExact, compare, flip_diagonal, pairs_of
from model_kit.core.errors import BlindCheck
from model_kit.core.manifest import SoftForm
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom, piece_key
from model_kit.core.tolerances import WELD_OF_PIECE_SIZE

from ..bodies.bar03 import bar03_scene
from ..bodies.quadruped import quadruped_scene
from ..shapes import box
from .piece_scenes import accepted, make_ctx, refuses_the_core, tried_and_caught

SHIN_SPECK = Path(__file__).resolve().parents[1] / "fixtures" / "regressions" / "shin_speck.npz"
BAR03 = piece_key("chibi", "chibi")
ANIMAL = piece_key("quadruped", "Animal")
SHINS = f"{BAR03} shin_R mirror of shin_L"
LEGS = (f"{ANIMAL} leg_fore_R mirror of leg_fore_L", f"{ANIMAL} leg_hind_R mirror of leg_hind_L")
FLIPPED = 2  # a diagonal flip changes two triangles of the source's image


def view_of(ctx, key):
    pack, _, name = key.partition(":")
    return next(v for v in pieces.measured_pieces(ctx, MirrorExact.kinds) if v.name == name
                and v.pack == pack)


def with_part(ctx, key, **parts):
    """The scene with these parts of the piece replaced."""
    geom = PieceGeom(ctx.pieces[key].name, {**ctx.pieces[key].parts, **parts})
    return dataclasses.replace(ctx, pieces={**ctx.pieces, key: geom})


def flipped(ctx, key, part):
    """The scene with one quad of `part` split along its other diagonal."""
    view = view_of(ctx, key)
    tris = next(p.tris for p in view.parts if p.name == part)
    out = flip_diagonal(tris, WELD_OF_PIECE_SIZE * view.size)
    assert out is not None
    return with_part(ctx, key, **{part: out})


# --- the passing cases --------------------------------------------------------------------------


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["mirror-exact"] is MirrorExact
    assert MirrorExact.kinds == frozenset({"assembly"})


def test_exact_mirror_passes():
    """Bar 03's four mirror forms are exact: the right eye, arm, thigh and shin."""
    ctx = bar03_scene()
    view = view_of(ctx, BAR03)
    pairs, unpaired = pairs_of(view)
    assert [(p.source, p.target, p.axis) for p in pairs] == [
        ("eye_L", "eye_R", "x"), ("arm_L", "arm_R", "x"), ("thigh_L", "thigh_R", "x"),
        ("shin_L", "shin_R", "x")]
    assert unpaired == []
    assert MirrorExact().run(ctx) == []
    assert SHINS in {p.item for p in MirrorExact().plants(ctx)}


def test_parts_paired_by_their_bones_pass_and_carry_the_rigs_axis():
    """The quadruped's right legs name no form; the rig's mirror map pairs them by their bones."""
    ctx = quadruped_scene()
    pairs, unpaired = pairs_of(view_of(ctx, ANIMAL))
    assert [(p.source, p.target, p.axis) for p in pairs] == [
        ("leg_fore_L", "leg_fore_R", "x"), ("leg_hind_L", "leg_hind_R", "x")]
    assert unpaired == []
    assert MirrorExact().run(ctx) == []


def test_a_reflection_with_its_winding_kept_still_matches():
    """Corners are compared without order: facing is topology's check."""
    ctx = quadruped_scene()
    source = ctx.pieces[ANIMAL].parts["leg_hind_L"]
    ctx = with_part(ctx, ANIMAL, leg_hind_R=source * np.array([-1.0, 1.0, 1.0]))
    assert MirrorExact().run(ctx) == []


def test_a_corner_off_by_less_than_the_weld_passes_and_by_more_fails():
    ctx = quadruped_scene()
    view = view_of(ctx, ANIMAL)
    tris = ctx.pieces[ANIMAL].parts["leg_hind_R"].copy()
    tol = WELD_OF_PIECE_SIZE * view.size
    near, far = tris.copy(), tris.copy()
    near[0, 0, 1] += 0.2 * tol
    far[0, 0, 1] += 5.0 * tol
    assert MirrorExact().run(with_part(ctx, ANIMAL, leg_hind_R=near)) == []
    (found,) = MirrorExact().run(with_part(ctx, ANIMAL, leg_hind_R=far))
    assert found.item == LEGS[1]


# --- the failing cases --------------------------------------------------------------------------


def d5_ctx():
    """The shins fixture's two shins as one piece whose right shin is the mirror of its left."""
    with np.load(SHIN_SPECK, allow_pickle=False) as z:
        parts = {name: z[f"v:{name}"].astype(float)[z[f"f:{name}"]] for name in ("shin_L", "shin_R")}
    ctx = make_ctx(parts)
    piece = ctx.packs["kit"].pieces["Thing"]
    records = tuple(dataclasses.replace(p, form=SoftForm(None, "shin_L", None, (), None))
                    if p.name == "shin_R" else p for p in piece.parts)
    return pieces.with_piece(ctx, "kit", "Thing", dataclasses.replace(piece, parts=records))


def test_d5_shins_before_fix_fail():
    """3250 triangles on each side and only 1966 of them are the other side's image (the
    regression test's count): 1284 of each are unmatched."""
    (found,) = MirrorExact().run(d5_ctx())
    assert (found.check, found.item) == ("mirror-exact", "kit:Thing shin_R mirror of shin_L")
    assert found.detail == ("1284 of 3250 triangles of shin_L, mirrored across x, are not in "
                            "shin_R, and 1284 of its 3250 are not in the source")


def test_diagonal_flip_caught_on_two_pieces():
    """One quad split the other way on the target's side: the shin of bar 03 and a leg of the
    quadruped."""
    for ctx, key, part, item in ((bar03_scene(), BAR03, "shin_R", SHINS),
                                 (quadruped_scene(), ANIMAL, "leg_hind_R", LEGS[1])):
        before = view_of(ctx, key)
        bad = flipped(ctx, key, part)
        (found,) = MirrorExact().run(bad)
        assert found.item == item
        assert f"{FLIPPED} of" in found.detail and f"{FLIPPED} of its" in found.detail
        # the plant is only a diagonal: the same corners and surface, so only this check can see it
        was = next(p.tris for p in before.parts if p.name == part)
        now = bad.pieces[key].parts[part]
        assert now.shape == was.shape
        assert {tuple(c) for c in now.reshape(-1, 3).tolist()} == {
            tuple(c) for c in was.reshape(-1, 3).tolist()}
        assert (np.any(now != was, axis=(1, 2))).sum() == FLIPPED
        area = lambda t: np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        assert area(now).sum() == pytest.approx(area(was).sum(), rel=1e-3)


def test_the_plants_flip_a_diagonal_on_each_target_and_the_gate_runs_them():
    for ctx, items in ((bar03_scene(), {SHINS}), (quadruped_scene(), set(LEGS))):
        units = MirrorExact().plants(ctx)
        assert items <= {u.item for u in units}
        for unit in units:
            plant = accepted(MirrorExact(), unit)
            assert plant.label.startswith("flipped diagonal")  # these meshes have quads to flip
            found = MirrorExact().run(plant.ctx, only=plant.scope)
            assert [f.item for f in found] == [plant.item]
        assert run_gate([MirrorExact()], ctx, None).status == "passed"


def test_a_check_that_misses_its_plant_stops_the_run():
    class Blind(MirrorExact):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="mirror-exact: quadruped:Animal leg_fore_R mirror of"):
        run_gate([Blind()], quadruped_scene(), None)


def test_a_run_scoped_to_an_item_measures_only_that_item():
    ctx = flipped(quadruped_scene(), ANIMAL, "leg_hind_R")
    assert MirrorExact().run(ctx, only={LEGS[0]}) == []
    assert [f.item for f in MirrorExact().run(ctx, only={LEGS[1]})] == [LEGS[1]]


def test_a_missing_triangle_fails():
    ctx = quadruped_scene()
    tris = ctx.pieces[ANIMAL].parts["leg_fore_R"]
    (found,) = MirrorExact().run(with_part(ctx, ANIMAL, leg_fore_R=tris[1:]))
    assert found.item == LEGS[0] and found.detail.startswith("1 of 288")


# --- no pair, no geometry, no twin --------------------------------------------------------------


def test_no_pairs_not_applicable():
    ctx = make_ctx({"slab": box((0, 0, 0), (0.1, 0.02, 0.1))})
    check = MirrorExact()
    assert check.run(ctx) == [] and check.plants(ctx) == []
    assert check.coverage(ctx) == [
        "mirror-exact: not applicable (no part is the mirror of another): kit:Thing"]
    assert run_gate([check], ctx, None).status == "passed"


def test_a_body_with_no_mirror_map_has_no_pairs():
    """Parts bound to bones pair only through the rig's map: without one, nothing pairs."""
    ctx = quadruped_scene()
    piece = ctx.packs["quadruped"].pieces["Animal"]
    bare = dataclasses.replace(piece, rig=dataclasses.replace(piece.rig, mirror=None))
    ctx = pieces.with_piece(ctx, "quadruped", "Animal", bare)
    assert pairs_of(view_of(ctx, ANIMAL)) == ([], [])
    assert any("not applicable" in line for line in MirrorExact().coverage(ctx))


def test_a_pair_with_a_part_that_has_no_geometry_is_unmeasured():
    ctx = quadruped_scene()
    parts = {k: v for k, v in ctx.pieces[ANIMAL].parts.items() if k != "leg_hind_R"}
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, ANIMAL: PieceGeom("Animal", parts)})
    (line,) = [x for x in MirrorExact().coverage(ctx) if isinstance(x, Unmeasured)]
    assert "no geometry for leg_hind_R" in line and LEGS[1] in line
    assert run_gate([MirrorExact()], ctx, None).status == "partial"


def test_a_part_with_two_possible_twins_is_unmeasured_not_guessed():
    ctx = quadruped_scene()
    view = view_of(ctx, ANIMAL)
    left = next(p for p in view.piece.parts if p.name == "leg_hind_L")
    twin = dataclasses.replace(left, name="leg_hind_L2")
    ctx = pieces.with_part_record(ctx, view, twin)
    pairs, unpaired = pairs_of(view_of(ctx, ANIMAL))
    assert [p.target for p in pairs] == ["leg_fore_R"]
    assert unpaired and all("no one twin" in why for why in unpaired)
    lines = [x for x in MirrorExact().coverage(ctx) if isinstance(x, Unmeasured)]
    assert lines and all("no one twin" in x for x in lines)


def with_sleeves(ctx, sides="LR", swap_right=False):
    """The quadruped with a sleeve on the hind leg of each named side, bound to the leg's bones.

    A sleeve is its leg grown a tenth about the origin, so the two sides are exact mirrors and
    a sleeve differs from its leg. With `swap_right`, the right leg holds the sleeve's shape and
    the right sleeve the leg's: the part order alone would pair them wrongly."""
    view = view_of(ctx, ANIMAL)
    for side in sides:
        leg = next(p for p in view.piece.parts if p.name == f"leg_hind_{side}")
        ctx = pieces.with_part_record(ctx, view_of(ctx, ANIMAL),
                                      dataclasses.replace(leg, name=f"sleeve_{side}"))
    parts = ctx.pieces[ANIMAL].parts
    grown = {side: parts[f"leg_hind_{side}"] * 1.1 for side in sides}
    new = {f"sleeve_{side}": grown[side] for side in sides}
    if swap_right:
        new["leg_hind_R"], new["sleeve_R"] = grown["R"], parts["leg_hind_R"]
    return with_part(ctx, ANIMAL, **new)


def test_two_parts_per_side_on_the_same_bones_pair_by_shape_and_read_quiet():
    """An exactly symmetric body with a sleeve over each hind leg is not `partial` for ever: each
    part is paired with the one its mirror matches, and the run passes with no Unmeasured line."""
    ctx = with_sleeves(quadruped_scene())
    pairs, unpaired = pairs_of(view_of(ctx, ANIMAL))
    assert unpaired == []
    assert {(p.source, p.target) for p in pairs} == {
        ("leg_fore_L", "leg_fore_R"), ("leg_hind_L", "leg_hind_R"), ("sleeve_L", "sleeve_R")}
    assert MirrorExact().run(ctx) == []
    assert not [x for x in MirrorExact().coverage(ctx) if isinstance(x, Unmeasured)]
    assert run_gate([MirrorExact()], ctx, None).status == "passed"


def test_the_pairing_by_shape_does_not_follow_the_part_order():
    ctx = with_sleeves(quadruped_scene(), swap_right=True)
    pairs, unpaired = pairs_of(view_of(ctx, ANIMAL))
    assert unpaired == []
    assert {(p.source, p.target) for p in pairs} == {
        ("leg_fore_L", "leg_fore_R"), ("leg_hind_L", "sleeve_R"), ("sleeve_L", "leg_hind_R")}
    assert MirrorExact().run(ctx) == []


def test_a_flipped_diagonal_on_a_sleeve_is_caught_among_parts_that_share_bones():
    ctx = with_sleeves(quadruped_scene())
    view = view_of(ctx, ANIMAL)
    tris = next(p.tris for p in view.parts if p.name == "sleeve_R")
    bad = with_part(ctx, ANIMAL, sleeve_R=flip_diagonal(tris, WELD_OF_PIECE_SIZE * view.size))
    (found,) = MirrorExact().run(bad)
    assert found.item == f"{ANIMAL} sleeve_R mirror of sleeve_L"


def test_sides_with_different_counts_on_the_same_bones_stay_unmeasured():
    """A sleeve on the left leg only: one side holds two parts, the other one, so no pairing is
    honest. The group is one Unmeasured line and the run reads `partial`."""
    ctx = with_sleeves(quadruped_scene(), sides="L")
    pairs, unpaired = pairs_of(view_of(ctx, ANIMAL))
    assert [p.target for p in pairs] == ["leg_fore_R"]
    (why,) = unpaired
    assert "2 parts" in why and "1 hold" in why and "no one twin" in why
    assert run_gate([MirrorExact()], ctx, None).status == "partial"


@pytest.mark.parametrize("sides", ["L", "R"])
def test_an_ambiguity_is_one_line_naming_the_group_whichever_side_holds_the_extra_part(sides):
    """The group is named once, with every part in it, however the parts are ordered."""
    ctx = with_sleeves(quadruped_scene(), sides=sides)
    _, unpaired = pairs_of(view_of(ctx, ANIMAL))
    (why,) = unpaired
    assert f"sleeve_{sides}" in why and f"leg_hind_{sides}" in why
    lines = [x for x in MirrorExact().coverage(ctx) if "no one twin" in x]
    assert len(lines) == 1 and isinstance(lines[0], Unmeasured)


def test_a_form_pair_is_not_paired_again_by_its_bones():
    ctx = bar03_scene()
    pairs, _ = pairs_of(view_of(ctx, BAR03))
    assert len(pairs) == len({p.target for p in pairs}) == 4


def one_triangle_pair():
    """Two parts of one triangle each, the second the mirror form of the first: no quad to flip."""
    tri = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]])
    ctx = make_ctx({"a": tri, "b": tri * np.array([-1.0, 1.0, 1.0])})
    piece = ctx.packs["kit"].pieces["Thing"]
    records = tuple(dataclasses.replace(p, form=SoftForm(None, "a", None, (), None))
                    if p.name == "b" else p for p in piece.parts)
    return pieces.with_piece(ctx, "kit", "Thing", dataclasses.replace(piece, parts=records))


def test_a_pair_with_nothing_to_flip_is_planted_with_a_moved_corner_instead():
    """A target of one triangle has no quad. It is still judged, so the unit holds other plants
    (a moved corner, a dropped triangle), and the run is not Unmeasured."""
    ctx = one_triangle_pair()
    check = MirrorExact()
    (unit,) = check.plants(ctx)
    assert unit.item == "kit:Thing b mirror of a" and unit.not_applicable == ""
    assert [p.label.split(":")[0] for p in unit.candidates] == ["moved corner", "dropped triangle"]
    assert accepted(check, unit).label.startswith("moved corner")
    assert check.run(ctx) == []
    assert not [x for x in check.coverage(ctx) if isinstance(x, Unmeasured)]
    assert run_gate([check], ctx, None).status == "passed"


def test_a_dropped_triangle_is_caught_when_a_corner_cannot_be_moved(monkeypatch):
    from model_kit.core.checks import mirror_exact

    monkeypatch.setattr(mirror_exact, "_nudged", lambda tris, tol: None)
    (unit,) = MirrorExact().plants(one_triangle_pair())
    assert [p.label.split(":")[0] for p in unit.candidates] == ["dropped triangle"]
    assert accepted(MirrorExact(), unit) is not None


def test_mirror_exact_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import mirror_exact

    ctx = bar03_scene()
    want = [u.item for u in MirrorExact().plants(ctx)]
    refuses_the_core(monkeypatch, mirror_exact, "_detail")
    monkeypatch.setattr(MirrorExact, "run", lambda *a, **k: 1 / 0)
    assert [u.item for u in MirrorExact().plants(ctx)] == want and SHINS in want


def test_mirror_exact_plants_caught_on_two_pieces():
    """Bar 03 (four mirror forms) and the quadruped (legs paired by their bones)."""
    chibi = tried_and_caught(MirrorExact(), bar03_scene())
    assert SHINS in chibi and len(chibi) == 4
    animal = tried_and_caught(MirrorExact(), quadruped_scene())
    assert set(LEGS) <= set(animal)


def test_coverage_does_not_build_the_plants(monkeypatch):
    """Coverage reads the plants only for their skip lines, so building them would flip every quad
    of every target for nothing."""
    from model_kit.core.checks import mirror_exact

    monkeypatch.setattr(mirror_exact, "flip_diagonal", lambda *a: 1 / 0)
    assert MirrorExact().coverage(bar03_scene()) == []


def test_a_piece_with_no_extent_is_unmeasured_not_silent():
    """A piece of size zero has no weld distance, so no pair differs; it says so."""
    ctx = one_triangle_pair()
    zero = {name: np.zeros((1, 3, 3)) for name in ("a", "b")}
    key = piece_key("kit", "Thing")
    ctx = dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom("Thing", zero)})
    assert MirrorExact().run(ctx) == [] and MirrorExact().plants(ctx) == []
    (line,) = [x for x in MirrorExact().coverage(ctx) if isinstance(x, Unmeasured)]
    assert "no extent" in line and "kit:Thing" in line
    assert run_gate([MirrorExact()], ctx, None).status == "partial"


def test_the_measured_parts_are_both_sides_of_each_pair():
    assert {k.rsplit(" ", 1)[1] for k in MirrorExact().measured(bar03_scene())} == {
        "eye_L", "eye_R", "arm_L", "arm_R", "thigh_L", "thigh_R", "shin_L", "shin_R"}


def test_compare_counts_the_triangles_with_no_match_on_each_side():
    a = box((0, 0, 0), (1, 1, 1))
    image = a * np.array([-1.0, 1.0, 1.0])
    assert compare(a, image, "x", 1e-6) == (0, 0)
    assert compare(a, image[:-2], "x", 1e-6) == (2, 0)
    assert compare(a[:-2], image, "x", 1e-6) == (0, 2)
    other = a.copy()
    other[0] = other[0][[1, 2, 0]] + [0.0, 0.5, 0.0]
    assert compare(a, other * np.array([-1.0, 1.0, 1.0]), "x", 1e-6) == (1, 1)
    assert compare(a[:0], a[:0], "x", 1e-6) == (0, 0)
