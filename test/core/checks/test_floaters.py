import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from model_kit.core import gate, sight
from model_kit.core.checks import pieces
from model_kit.core.checks.floaters import Floaters, _check_chains
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.geom import surface_gap
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import CALIBRATION_FAIL, CALIBRATION_PASS

from ..shapes import box
from .piece_scenes import (
    TOUCH,
    make_ctx,
    plants_caught,
    post_on_base,
    refuses_the_core,
    tried_and_caught,
)
from .sight_scenes import unviewed, viewed
from .support_scenes import (
    BRIDGES,
    SUPPORT_TOUCH,
    bench_ctx,
    pallet_ctx,
    shortened,
)

ON_BASE = {"post": {"on": "base"}}
ITEM = "kit:Thing post"


def findings(ctx, only=None):
    return Floaters().run(ctx, only=only)


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["floaters"] is Floaters
    assert Floaters.kinds == frozenset({"assembly"})


def test_a_part_on_its_host_and_a_base_on_the_floor_pass():
    ctx = make_ctx(post_on_base(), ON_BASE)
    assert findings(ctx) == []
    assert plants_caught(Floaters(), ctx)
    assert run_gate([Floaters()], ctx, None).status == "passed"


def test_calibration_half_the_touch_passes_and_double_fails():
    half = make_ctx(post_on_base(lift=CALIBRATION_PASS * TOUCH), ON_BASE)
    assert findings(half) == []
    double = make_ctx(post_on_base(lift=CALIBRATION_FAIL * TOUCH), ON_BASE)
    (found,) = findings(double)
    assert (found.check, found.item) == ("floaters", ITEM)
    assert "base" in found.detail


def test_a_touching_part_searches_only_within_the_touch_distance(monkeypatch):
    from model_kit.core.checks import floaters

    seen = []
    real = floaters.surface_gap
    monkeypatch.setattr(floaters, "surface_gap",
                        lambda a, b, within: seen.append(within) or real(a, b, within))
    assert findings(make_ctx(post_on_base(), ON_BASE)) == []
    assert seen == [pytest.approx(TOUCH)] * len(seen) and seen


def test_a_far_part_still_reports_its_exact_gap(monkeypatch):
    from model_kit.core.checks import floaters

    seen = []
    real = floaters.surface_gap
    monkeypatch.setattr(floaters, "surface_gap",
                        lambda a, b, within: seen.append(within) or real(a, b, within))
    (found,) = findings(make_ctx(post_on_base(lift=0.1), ON_BASE))
    # the post's lowest face is 0.1 above the base top; the wider search measures the true gap
    assert f"{0.1:.4f} from its `on` part base" in found.detail
    assert seen[0] == pytest.approx(TOUCH) and seen[-1] > seen[0]


def test_a_part_buried_in_its_host_still_touches_it():
    assert findings(make_ctx(post_on_base(sink=0.01), ON_BASE)) == []


def test_a_part_with_no_host_rests_on_the_floor_or_fails():
    parts = {"base": box((-0.1, 0, -0.1), (0.1, 0.02, 0.1)),
             "pad": box((0.3, 0, -0.1), (0.4, 0.02, 0.1))}
    assert findings(make_ctx(parts)) == []  # both stand on the lowest y
    lifted = {**parts, "pad": box((0.3, 0.05, -0.1), (0.4, 0.07, 0.1))}
    (found,) = findings(make_ctx(lifted))
    assert found.item == "kit:Thing pad"
    assert "floor" in found.detail


def test_floor_calibration_half_the_touch_passes_and_double_fails():
    def pad_at(y):
        return make_ctx({"base": box((-0.1, 0, -0.1), (0.1, 0.02, 0.1)),
                         "pad": box((0.3, y, -0.1), (0.4, y + 0.02, 0.1))})

    assert findings(pad_at(CALIBRATION_PASS * TOUCH)) == []
    assert [f.item for f in findings(pad_at(CALIBRATION_FAIL * TOUCH))] == ["kit:Thing pad"]


def test_an_anchor_part_is_excused_from_the_floor():
    parts = {"base": box((-0.1, 0, -0.1), (0.1, 0.02, 0.1)),
             "lid": box((-0.1, 0.5, -0.1), (0.1, 0.52, 0.1))}
    assert len(findings(make_ctx(parts))) == 1
    ctx = make_ctx(parts, hosts=["lid"])
    assert findings(ctx) == []


def test_hosts_that_name_no_part_raise():
    ctx = make_ctx(post_on_base(), hosts=["nothing"])
    with pytest.raises(ManifestError, match="nothing"):
        findings(ctx)


def test_a_host_part_with_no_geometry_raises():
    parts = {"post": post_on_base()["post"]}
    with pytest.raises(ManifestError, match="base"):
        findings(make_ctx(parts, {"post": {"on": "base"}}))


def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = make_ctx({**post_on_base(lift=0.1), "pad": box((0.3, 0.1, -0.1), (0.4, 0.12, 0.1))},
                   ON_BASE)
    assert {f.item for f in findings(ctx)} == {ITEM, "kit:Thing pad"}
    assert [f.item for f in findings(ctx, only={ITEM})] == [ITEM]


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = make_ctx(post_on_base(), ON_BASE, also={"Tree": post_on_base(lift=0.1)})
    assert findings(ctx) == []
    (line,) = Floaters().coverage(ctx)
    assert line == "floaters: not applicable (branching piece): kit:Tree"
    assert not isinstance(line, Unmeasured)  # the branching checks measure it
    assert run_gate([Floaters()], ctx, None).status == "passed"


def test_a_piece_with_no_geometry_in_the_scene_is_unmeasured():
    ctx = dataclasses.replace(make_ctx(post_on_base(), ON_BASE), pieces={})
    (line,) = Floaters().coverage(ctx)
    assert line == "floaters: skipped (no geometry): kit:Thing"
    assert isinstance(line, Unmeasured)
    assert run_gate([Floaters()], ctx, None).status == "partial"


def test_a_blind_check_stops_the_run():
    class Blind(Floaters):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="floaters"):
        run_gate([Blind()], make_ctx(post_on_base(), ON_BASE), None)


def test_a_plant_lifts_a_small_part_twice_the_touch_distance():
    ctx = make_ctx(post_on_base(), ON_BASE)
    (unit,) = Floaters().plants(ctx)
    assert unit.item == ITEM  # the post is the smaller part
    assert len(unit.candidates) == 6  # up first, then the other axes, both ways
    base = ctx.pieces["kit:Thing"].parts["post"]
    lifted = unit.candidates[0].ctx.pieces["kit:Thing"].parts["post"]
    assert abs(lifted - base).max() == pytest.approx(CALIBRATION_FAIL * TOUCH)


def test_a_part_on_itself_or_a_cycle_of_parts_raises():
    parts = post_on_base()
    with pytest.raises(ManifestError, match="cycle"):
        findings(make_ctx(parts, {"post": {"on": "post"}}))
    with pytest.raises(ManifestError, match="base.*post.*base|post.*base.*post"):
        findings(make_ctx(parts, {"base": {"on": "post"}, "post": {"on": "base"}}))


def test_a_chain_of_hosts_ends_at_a_part_that_rests_on_the_floor():
    parts = {**post_on_base(), "cap": box((-0.02, 0.12, -0.02), (0.02, 0.14, 0.02))}
    ctx = make_ctx(parts, {"post": {"on": "base"}, "cap": {"on": "post"}})
    assert findings(ctx) == []


def test_a_piece_of_anchors_only_says_no_lift_floats_one():
    ctx = make_ctx(post_on_base(), hosts=["base", "post"])  # every part is an anchor
    (unit,) = Floaters().plants(ctx)
    assert unit.item == "kit:Thing" and unit.candidates == ()
    verdict = run_gate([Floaters()], ctx, None)
    (line,) = [x for x in verdict.coverage if "not applicable" in x]
    assert line == ("floaters: not applicable (every part is an anchor or the only part on the "
                    "floor: no lift floats one): kit:Thing")
    assert not isinstance(line, Unmeasured) and verdict.status == "passed"


def test_the_only_part_on_the_floor_is_not_the_plant_item():
    # the base is the only part at the lowest y, so a lift leaves it the floor: the unit takes
    # the post, which rests `on` it
    (unit,) = Floaters().plants(make_ctx(post_on_base(), ON_BASE))
    assert unit.item == ITEM
    # with no `on` links the post is not at the lowest y either, and the base is alone there
    parts = post_on_base()
    (unit,) = Floaters().plants(make_ctx(parts))
    assert unit.item == ITEM


def test_a_piece_of_one_part_needs_no_plant_and_says_so():
    ctx = make_ctx({"base": post_on_base()["base"]})
    (line,) = Floaters().coverage(ctx)
    assert line == "floaters: one part, no plant needed: kit:Thing"
    assert not isinstance(line, Unmeasured)
    assert run_gate([Floaters()], ctx, None).status == "passed"


# --- the contact length -----------------------------------------------------------------------
# Touching is a gap at or under the piece's contact length: one arcminute at its nearest viewing
# distance (`sight.contact_length`). FLOOR is a contact length wider than TOUCH, the fixture's own.

FLOOR = 0.05  # in the test pack's units, more than twice TOUCH (0.0213)


def pad_at(y):
    return make_ctx({"base": box((-0.1, 0, -0.1), (0.1, 0.02, 0.1)),
                     "pad": box((0.3, y, -0.1), (0.4, y + 0.02, 0.1))})


@pytest.mark.parametrize(("label", "make", "item"), [
    ("a gap to its host", lambda g: make_ctx(post_on_base(lift=g), ON_BASE), ITEM),
    ("a height over the floor", pad_at, "kit:Thing pad"),
])
def test_a_float_just_over_the_contact_length_fires_and_one_just_under_touches(label, make, item):
    over = viewed(make(1.01 * FLOOR), FLOOR)
    assert [f.item for f in findings(over)] == [item], label
    assert findings(viewed(make(0.99 * FLOOR), FLOOR)) == [], label
    # a nearer view has a shorter contact length: the same gap is a float there
    assert [f.item for f in findings(viewed(make(0.99 * FLOOR), 0.5 * FLOOR))] == [item], label
    assert not any("arcmin" in x for x in Floaters().coverage(over))  # nothing is dropped


def test_the_contact_length_is_the_pieces_seen_floor_not_a_share_of_its_wall():
    ctx = viewed(make_ctx(post_on_base(), ON_BASE), FLOOR)
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    assert view.contact == pytest.approx(FLOOR, rel=1e-12)
    assert view.contact == sight.contact_length(view.piece, ctx.packs["kit"].units_per_metre)


def with_air_under(ctx, part, mm):
    """The scene with `part` of its one piece lifted straight up until `mm` millimetres of air lie
    between it and the `on` part it rests on (found by bisection on the surface gap, so a part
    that starts sunk into its host is lifted clear first)."""
    (view,) = pieces.measured_pieces(ctx, {"assembly"})
    mesh = next(p for p in view.parts if p.name == part)
    host = next(p for p in view.parts if p.name == mesh.on)
    want = mm / 1000.0 * ctx.packs[view.pack].units_per_metre
    lo, hi = 0.0, float(np.ptp(np.concatenate([mesh.tris, host.tris]).reshape(-1, 3)[:, 1])) + want
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lifted = mesh.tris + np.array([0.0, mid, 0.0])
        if surface_gap(lifted, host.tris, want) > want:
            hi = mid
        else:
            lo = mid
    geometry = {p.name: p.tris for p in view.parts} | {part: mesh.tris + np.array([0.0, hi, 0.0])}
    ctx = pieces.with_geometry(ctx, view, geometry)
    assert surface_gap(geometry[part], host.tris, 2 * want) == pytest.approx(want, rel=1e-6)
    return ctx


def bar03_head_lift(mm):
    """Bar 03's scene with its head lifted until `mm` millimetres of air lie under it (the pack's
    units are 0.747 a metre)."""
    from ..bodies.bar03 import bar03_scene

    return with_air_under(bar03_scene(), "head", mm)


def test_air_gap_of_1p5_mm_is_a_floater_on_two_pieces():
    """A resting part with 1.5 mm of air under it is a float where a viewer can see 1.5 mm: on
    pieces whose seen floor is under that. A touch distance of 10.7 bases would let it pass."""
    fixture = viewed(with_air_under(make_ctx(post_on_base(), ON_BASE), "post", 1.5), 0.0003)
    bar03 = bar03_head_lift(1.5)
    for label, ctx, item in (("fixture", fixture, ITEM), ("bar 03", bar03, "chibi:chibi head")):
        (view,) = pieces.measured_pieces(ctx, {"assembly"})
        upm = ctx.packs[view.pack].units_per_metre
        floor_mm = sight.contact_length(view.piece, upm) * 1000.0 / upm
        assert floor_mm < 1.5, (label, floor_mm)  # the guard's precondition
        assert [f.item for f in findings(ctx)] == [item], label
    # and the same 1.5 mm is no float on a piece seen from so far that a viewer cannot see it
    assert findings(viewed(make_ctx(post_on_base(lift=0.0015), ON_BASE), 0.003)) == []


def test_a_piece_with_no_view_reads_unmeasured_and_is_judged_on_nothing():
    ctx = unviewed(make_ctx(post_on_base(lift=0.1), ON_BASE))
    assert findings(ctx) == []  # a 0.1 gap is no finding when there is no contact length
    assert Floaters().plants(ctx) == []
    (line,) = [x for x in Floaters().coverage(ctx) if "no view" in x]
    assert isinstance(line, Unmeasured) and line.endswith(": kit:Thing")
    assert run_gate([Floaters()], ctx, None).status == "partial"


def test_a_float_on_a_hidden_side_still_fires():
    side = box((0.15, 0.0, -0.02), (0.19, 0.04, 0.02))  # beside the base, 0.05 off its +x face
    ctx = make_ctx({**post_on_base(lift=0.05), "side": side}, {**ON_BASE, "side": {"on": "base"}})
    every = viewed(ctx, 1e-9)
    hidden = viewed(ctx, 1e-9, ("top", "bottom", "left", "right", "front", "back"))
    assert {f.item for f in findings(hidden)} == {f.item for f in findings(every)} != set()
    assert not any("hidden" in x for x in Floaters().coverage(hidden))
    lifted = viewed(pad_at(0.1), 1e-9, ("bottom", "front", "back", "left", "right"))
    assert [f.item for f in findings(lifted)] == ["kit:Thing pad"]


def test_plants_stay_caught_under_a_far_view_on_two_pieces():
    from ..bodies.bar03 import bar03_scene

    bar03 = bar03_scene()
    for label, ctx in (("fixture", make_ctx(post_on_base(), ON_BASE)), ("bar 03", bar03)):
        base = viewed(ctx, FLOOR if label == "fixture" else 0.03)
        assert Floaters().plants(base), label
        assert plants_caught(Floaters(), base), label


@pytest.mark.parametrize("make,units", [
    (lambda: make_ctx(post_on_base(), ON_BASE), 1), (pallet_ctx, 3), (bench_ctx, 3)],
    ids=["post on a base", "pallet", "bench"])
def test_floaters_plants_never_call_the_core(monkeypatch, make, units):
    from model_kit.core.checks import floaters

    ctx = make()
    refuses_the_core(monkeypatch, floaters, "_problem", "_findings", "_gap")
    monkeypatch.setattr(floaters.Floaters, "run", lambda *a, **k: 1 / 0)
    assert len(Floaters().plants(ctx)) == units
    if units == 1:
        assert [u.item for u in Floaters().plants(ctx)] == [ITEM]


def test_floaters_plants_caught_on_two_pieces():
    from ..bodies.bar03 import bar03_scene

    for name, ctx in (("post on a base", make_ctx(post_on_base(), ON_BASE)),
                      ("bar 03", bar03_scene())):
        # a unit is tried (not skipped as failing already) and a candidate of it is caught
        assert tried_and_caught(Floaters(), ctx), name
        run_gate([Floaters()], ctx, None)


def test_a_check_that_misses_every_float_stops_the_run():
    class Blind(Floaters):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match=r"floaters: kit:Thing post"):
        run_gate([Blind()], make_ctx(post_on_base(), ON_BASE), None)


# --- a part that rests on several supports ----------------------------------------------------


@pytest.mark.parametrize("piece", sorted(BRIDGES))
def test_an_honest_bridging_piece_passes(piece):
    ctx = BRIDGES[piece][0]()
    assert Floaters().run(ctx) == []
    assert run_gate([Floaters()], ctx, None).status == "passed"


@pytest.mark.parametrize("piece,part,support", [
    ("pallet", "stringer_3", "block_31"), ("bench", "seat", "trestle_r")])
def test_a_part_off_one_support_is_one_finding_naming_it(piece, part, support):
    make, meshes = BRIDGES[piece]
    # shorten that support by 2 contact lengths, so only it lets go of the part
    tris = shortened(meshes()[support], CALIBRATION_FAIL * SUPPORT_TOUCH)
    found = Floaters().run(make(parts={support: tris}))
    assert [f.item for f in found] == [f"kit:Thing {part} off {support}"]
    assert f"from its support {support}" in found[0].detail


def test_a_scoped_run_returns_only_the_support_it_names():
    pallet = pallet_ctx(parts={"block_31": shortened(
        BRIDGES["pallet"][1]()["block_31"], CALIBRATION_FAIL * SUPPORT_TOUCH)})
    item = "kit:Thing stringer_3 off block_31"
    assert [f.item for f in Floaters().run(pallet, only={item})] == [item]


def test_a_cycle_through_rests_on_raises():
    with pytest.raises(ManifestError, match="`rests_on` makes a cycle"):
        Floaters().run(bench_ctx(specs={"trestle_l": {"on": "seat"}}))


def test_a_support_with_no_geometry_raises():
    ctx = bench_ctx(parts={"trestle_r": np.zeros((0, 3, 3))})
    with pytest.raises(ManifestError, match=r"seat: rests on 'trestle_r', which has no geometry"):
        Floaters().run(ctx)


def test_a_chain_of_1500_parts_is_walked_without_recursion():
    names = [f"p{i}" for i in range(1500)]
    parts = [SimpleNamespace(name=n, on=m, rests_on=()) for n, m in zip(names, [*names[1:], None])]
    _check_chains(SimpleNamespace(key="kit:Thing", parts=parts))


EXPECTED_BRIDGE_ITEMS = {
    "pallet": ("kit:Thing stringer_1 off block_11", "kit:Thing stringer_1 off stringer_2"),
    "bench": ("kit:Thing seat off trestle_l", "kit:Thing seat off stretcher"),
}


@pytest.mark.parametrize("piece", sorted(BRIDGES))
def test_bridging_plants_are_caught_on_two_pieces(piece):
    ctx = BRIDGES[piece][0]()
    caught_items = tried_and_caught(Floaters(), ctx)
    tilt, phantom = EXPECTED_BRIDGE_ITEMS[piece]
    assert tilt in caught_items and phantom in caught_items
    assert run_gate([Floaters()], ctx, None).status == "passed"


@pytest.mark.parametrize("piece", sorted(BRIDGES))
def test_the_tilt_lifts_the_target_and_keeps_the_pivot(piece):
    ctx = BRIDGES[piece][0]()
    tilt, _ = EXPECTED_BRIDGE_ITEMS[piece]
    unit = next(u for u in Floaters().plants(ctx) if u.item == tilt)
    (plant,) = unit.candidates
    part, target = tilt.removeprefix("kit:Thing ").split(" off ")
    view = next(pieces.contact_pieces(plant.ctx, Floaters.kinds))
    tilted = next(p for p in view.parts if p.name == part)
    supports = {p.name: p for p in view.parts if p.name in tilted.rests_on}
    limit = CALIBRATION_FAIL * SUPPORT_TOUCH * 2
    assert surface_gap(tilted.tris, supports[target].tris, limit) >= (
        CALIBRATION_FAIL * SUPPORT_TOUCH * (1 - 1e-6))
    pivot = [n for n in supports if n != target]
    assert any(surface_gap(tilted.tris, supports[n].tris, limit) <= SUPPORT_TOUCH
               for n in pivot)


def test_the_phantom_never_names_a_part_whose_chain_reaches_the_part():
    ctx = pallet_ctx()
    unit = next(u for u in Floaters().plants(ctx) if u.item.endswith("stringer_1 off stringer_2"))
    (plant,) = unit.candidates
    view = next(pieces.contact_pieces(plant.ctx, Floaters.kinds))
    named = next(p for p in view.parts if p.name == "stringer_1").rests_on
    assert not any(n.startswith("top_") for n in named)
    assert Floaters().run(plant.ctx)  # runs with no ManifestError and reports the phantom


def test_a_bridging_part_lifted_reports_its_first_support():
    ctx = bench_ctx(hosts=["trestle_l", "trestle_r", "stretcher"])
    unit = Floaters().plants(ctx)[0]
    assert unit.item == "kit:Thing seat off trestle_l"
    assert any(f.item == unit.item for plant in unit.candidates
               for f in Floaters().run(plant.ctx, only=plant.scope))


def test_a_piece_with_no_bridging_part_gets_the_same_units():
    ctx = make_ctx(post_on_base(), ON_BASE)
    assert [u.item for u in Floaters().plants(ctx)] == [ITEM]
