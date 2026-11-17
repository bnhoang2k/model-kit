import dataclasses

import numpy as np
import pytest

from model_kit.core import gate, geom
from model_kit.core.checks import crossings, pieces, take_apart
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.take_apart import TakeApart
from model_kit.core.errors import BlindCheck, Unmeasurable
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import piece_key
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    CROSS_OF_PART_THICKNESS,
    TAKE_APART_STEPS,
)

from ..shapes import box
from .piece_scenes import (
    make_ctx,
    plants_caught,
    post_on_base,
    refuses_the_core,
    sheet_and_knob,
    strip_and_post,
    tried_and_caught,
)

GROUPS = {"base": {"group": "base"}, "post": {"group": "post"}}
ITEM = "kit:Thing base x post"
LANE = "kit:Thing a x b"
# The lane's runner `a` and the cubes of `b` are all 0.4 wide: the limit is half of that.
CROSS = CROSS_OF_PART_THICKNESS * 0.4


@pytest.fixture
def centre_line_only(monkeypatch):
    """A group is tried along its pull only: the sweep's own mechanics (the order, the rest depth,
    the limit) read on one line, free of the 26 other ways."""
    monkeypatch.setattr(take_apart, "DEPTH_WAYS", np.empty((0, 3)))


def centred(centre, size):
    centre, half = np.array(centre, dtype=float), np.array(size, dtype=float) / 2
    return box(centre - half, centre + half)


def _lane_groups(overlap):
    return {
        "anchor": {"anchor": centred((0, 0, 0), (1, 1, 1))},
        "a": {"a": centred((2, 0, 0), (0.4, 0.4, 0.4))},
        "b": {"b_up": centred((4, 0.4 - overlap, 0), (0.4, 0.4, 0.4)),
              "b_dn": centred((4, overlap - 0.4, 0), (0.4, 0.4, 0.4))},
    }


def lane(order=("anchor", "a", "b"), overlap=2 * CROSS):
    """The lane scene with its groups taken apart in `order` (built in the reverse).

    Group `a` (a cube) and group `b` (a pair of cubes, one above the lane and one below) sit
    in a line along +x, far from the anchor cube at the middle. Every group moves out along +x,
    so `a` runs into `b` when `a` goes first, and not when `b` has gone already. `overlap` is how
    far `b`'s cubes reach into the lane: the default, twice the limit, blocks it, and half the
    limit only grazes `a`'s top and bottom. At rest nothing touches."""
    return lane_built(tuple(reversed(order)), overlap)


def lane_built(build, overlap=0.2):
    """The lane scene whose groups are built (first appear among the parts) in `build` order."""
    made = _lane_groups(overlap)
    parts = {name: tris for group in build for name, tris in made[group].items()}
    specs = {name: {"group": group} for group in build for name in made[group]}
    return make_ctx(parts, specs)


def findings(ctx, only=None):
    return TakeApart().run(ctx, only=only)


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["take-apart"] is TakeApart
    assert TakeApart.kinds == frozenset({"assembly"})


def test_the_sweep_is_21_shares_from_0_to_1():
    assert TAKE_APART_STEPS == 21
    assert take_apart.shares() == pytest.approx(np.linspace(0.0, 1.0, 21))


def test_groups_are_built_in_the_order_they_are_first_seen_and_come_off_in_reverse():
    ctx = make_ctx(post_on_base(), {"base": {"group": "z"}, "post": {"group": "a"}})
    piece = ctx.packs["kit"].pieces["Thing"]
    assert take_apart.group_order(piece) == ["z", "a"]
    (view,) = pieces.measured_pieces(ctx, TakeApart.kinds)
    assert take_apart._layout(view).order == ["a", "z"]


@pytest.mark.usefixtures("centre_line_only")
def test_the_last_group_made_comes_off_first():
    # b is built after a, so b comes off first and a's run is clear: no finding
    assert findings(lane_built(("anchor", "a", "b"))) == []
    # built the other way round, a is made last and comes off first into b's way
    (found,) = findings(lane_built(("anchor", "b", "a")))
    assert found.item == LANE


def test_a_piece_whose_groups_move_clear_of_each_other_passes():
    ctx = make_ctx(post_on_base(), GROUPS)
    assert findings(ctx) == []
    assert plants_caught(TakeApart(), ctx)
    assert run_gate([TakeApart()], ctx, None).status == "passed"


@pytest.mark.usefixtures("centre_line_only")
def test_the_order_of_the_groups_decides_whether_a_group_runs_into_another():
    assert Crossings().run(lane()) == []  # clean at rest: only the sweep shows it
    (found,) = findings(lane())
    assert (found.check, found.item) == ("take-apart", LANE)
    assert "while a moves out along (+1.000, +0.000, +0.000)" in found.detail
    assert "from share 0.40 of the piece size" in found.detail
    assert findings(lane(("anchor", "b", "a"))) == []  # b has gone before a sets out


@pytest.mark.usefixtures("centre_line_only")
def test_the_groups_taken_off_before_a_stage_are_gone_and_the_ones_after_sit_as_built():
    # With b first, a's whole run is clear; with a first, b is still at rest in a's way. The
    # verdict follows the order, so the later group must have stayed as built for a's stage.
    first = findings(lane(("a", "b", "anchor")))
    assert [f.item for f in first] == [LANE]
    assert findings(lane(("b", "a", "anchor"))) == []


def test_a_cut_at_rest_is_crossings_and_not_take_apart():
    ctx = make_ctx(post_on_base(sink=0.05), GROUPS)
    assert len(Crossings().run(ctx)) == 1
    assert findings(ctx) == []


def seat_over_limit(width):
    """A sink a fifth over the limit for a pin of this width in the 0.2 plate: the limit is half
    the pin's width, and a pin sunk further than its own width shares only that width."""
    return 1.2 * CROSS_OF_PART_THICKNESS * width


def pin_in_plate(sink, width=0.1, anchor_y=6.0):
    """A pin stands in a plate, seated `sink` deep, with a heavy anchor far above (the pin's
    group is driven down through the plate when the groups are taken apart) or far below (the
    pin comes straight out)."""
    parts = {"anchor": centred((0, anchor_y, 0), (3, 3, 3)),
             "pin": centred((0, 0.1 + 0.25 - sink, 0), (width, 0.5, width)),
             "plate": centred((0, 0, 0), (1, 0.2, 1))}
    built = dict(reversed(parts.items()))  # taken apart anchor, pin, plate: built the reverse
    return make_ctx(built, {name: {"group": name} for name in built})


@pytest.mark.parametrize("width", [0.05, 0.1])
@pytest.mark.usefixtures("centre_line_only")
def test_a_seated_pair_is_judged_against_its_rest_depth_not_skipped(width):
    seated = pin_in_plate(seat_over_limit(width), width)
    assert len(Crossings().run(seated)) == 1  # crossings owns the seat at rest
    (found,) = findings(seated)  # the sweep still drives the pin deeper through the plate
    assert found.item == "kit:Thing pin x plate"
    assert [f.item for f in findings(pin_in_plate(0.0, width))] == ["kit:Thing pin x plate"]


@pytest.mark.parametrize("width", [0.05, 0.1])
def test_a_seated_pin_that_comes_straight_out_is_not_flagged(width):
    out = pin_in_plate(seat_over_limit(width), width, anchor_y=-6.0)
    assert len(Crossings().run(out)) == 1
    assert findings(out) == []


def test_a_seated_pair_that_comes_straight_out_is_not_flagged():
    ctx = make_ctx(post_on_base(sink=0.05), GROUPS)
    assert len(Crossings().run(ctx)) == 1
    assert findings(ctx) == []


def test_parts_of_one_group_move_together_and_are_not_measured_against_each_other():
    ctx = make_ctx(post_on_base(sink=0.05), {"post": {"group": "p"}, "base": {"group": "p"}})
    assert findings(ctx) == []


@pytest.mark.usefixtures("centre_line_only")
def test_calibration_half_the_limit_passes_and_double_fails():
    assert findings(lane(overlap=CALIBRATION_PASS * CROSS)) == []
    assert [f.item for f in findings(lane(overlap=CALIBRATION_FAIL * CROSS))] == [LANE]


def test_a_plant_boxes_one_group_in_with_six_slabs_in_a_group_of_their_own():
    ctx = lane(("anchor", "b", "a"))
    assert findings(ctx) == []
    assert plants_caught(TakeApart(), ctx)
    (unit,) = TakeApart().plants(ctx)
    assert len(unit.candidates) == 1
    plant = unit.candidates[0]
    piece = plant.ctx.packs["kit"].pieces["Thing"]
    assert len(piece.parts) == 4 + 6
    slabs = [p for p in piece.parts if "_cage" in p.name]
    assert len(slabs) == 6
    (cage_group,) = {p.group for p in slabs}
    assert cage_group not in {p.group for p in ctx.packs["kit"].pieces["Thing"].parts}
    assert take_apart.group_order(piece)[0] == cage_group  # built first, so it comes off last
    (mover,) = set(unit.item.removeprefix("kit:Thing ").split(" x ")) - {cage_group}
    assert unit.item == "kit:Thing " + " x ".join(sorted((mover, cage_group)))  # a new item
    geometry = plant.ctx.pieces[piece_key("kit", "Thing")].parts
    assert all(len(geometry[p.name]) == 12 for p in slabs)  # six closed boxes
    # clean at rest: the cage is a gap clear of the group it boxes in
    names = {p.name for p in piece.parts if p.group == mover}
    assert not [f for f in Crossings().run(plant.ctx) if any(n in f.item.split(" ") for n in names)]


def test_the_plant_is_made_whichever_group_goes_first():
    for order in (("anchor", "a", "b"), ("b", "a", "anchor")):
        ctx = lane(order, overlap=0.0)  # b only just clear of the lane, so nothing is found
        assert findings(ctx) == []
        assert plants_caught(TakeApart(), ctx)


def test_a_group_at_the_centre_has_no_pull_and_is_tried_in_all_26_ways():
    """A group whose pull is zero is still tested. It is tried along the 26 ways, so a core
    walled in on every side fails, and it fails whichever of its ways is looked at."""
    ctx = caged()
    (view,) = pieces.measured_pieces(ctx, TakeApart.kinds)
    assert not take_apart._layout(view).toward["core"].any()
    assert [f.item for f in findings(ctx)] == [CASE]
    assert len(TakeApart().plants(ctx)) == 1  # a piece of two groups always has a plant


def test_take_apart_plants_never_call_the_core(monkeypatch):
    from model_kit.core.checks import take_apart

    ctx = lane(("anchor", "b", "a"))
    refuses_the_core(monkeypatch, take_apart, "_sweep", "measure")
    monkeypatch.setattr(take_apart.TakeApart, "run", lambda *a, **k: 1 / 0)
    (unit,) = TakeApart().plants(ctx)
    assert unit.candidates


def test_take_apart_plants_caught_on_two_pieces():
    from ..bodies.bar03 import bar03_scene

    for name, ctx in (("lane", lane(("anchor", "b", "a"))), ("bar 03", bar03_scene())):
        assert tried_and_caught(TakeApart(), ctx), name  # tried, not skipped, and caught
        run_gate([TakeApart()], ctx, None)


def test_a_piece_in_one_group_has_nothing_to_take_apart_and_needs_no_plant():
    ctx = make_ctx(post_on_base())
    assert list(TakeApart().plants(ctx)) == []
    (line,) = TakeApart().coverage(ctx)
    assert line == "take-apart: not applicable (fewer than two swept groups): kit:Thing"
    assert not isinstance(line, Unmeasured)


def test_a_part_with_no_group_is_unmeasured_and_the_run_is_partial():
    ctx = make_ctx(post_on_base(sink=0.05), GROUPS)
    piece = ctx.packs["kit"].pieces["Thing"]
    bare = dataclasses.replace(piece, parts=piece.parts[:1])  # the post has no row
    pack = dataclasses.replace(ctx.packs["kit"], pieces={"Thing": bare})
    ctx = dataclasses.replace(ctx, packs={"kit": pack})
    line = "take-apart: skipped (no group): kit:Thing post"
    assert line in TakeApart().coverage(ctx)
    assert isinstance(TakeApart().coverage(ctx)[0], Unmeasured)
    assert findings(ctx) == []  # a lone group is nothing to take apart
    assert run_gate([TakeApart()], ctx, None).status == "partial"


@pytest.mark.usefixtures("centre_line_only")
def test_a_scoped_run_measures_only_the_items_it_is_given():
    ctx = lane()
    assert findings(ctx, only={"kit:Thing anchor x b"}) == []
    assert len(findings(ctx, only={LANE})) == 1


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = make_ctx(post_on_base(), GROUPS, also={"Tree": post_on_base()})
    (line,) = TakeApart().coverage(ctx)
    assert line == "take-apart: not applicable (branching piece): kit:Tree"
    assert not isinstance(line, Unmeasured)  # the branching checks measure it
    assert run_gate([TakeApart()], ctx, None).status == "passed"


def test_a_piece_with_no_geometry_in_the_scene_is_unmeasured():
    ctx = dataclasses.replace(make_ctx(post_on_base(), GROUPS), pieces={})
    (line,) = TakeApart().coverage(ctx)
    assert line == "take-apart: skipped (no geometry): kit:Thing"
    assert isinstance(line, Unmeasured)
    assert run_gate([TakeApart()], ctx, None).status == "partial"


def test_a_blind_check_stops_the_run():
    class Blind(TakeApart):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="take-apart"):
        run_gate([Blind()], make_ctx(post_on_base(), GROUPS), None)


def taken_apart(groups):
    """A piece from `(group, {part: mesh})` in take-apart order (built in the reverse)."""
    built = list(reversed(groups))
    parts = {name: tris for _, made in built for name, tris in made.items()}
    specs = {name: {"group": group} for group, made in built for name in made}
    return make_ctx(parts, specs)


def two_arm_group():
    # Group A is a cube on +x and one on +y, so it leaves on the diagonal. B, off to the side,
    # leaves on a line through where A ends up, not through where A was built.
    return taken_apart([
        ("A", {"a1": centred((2, 0, 0), (0.4, 0.4, 0.4)),
               "a2": centred((0, 2, 0), (0.4, 0.4, 0.4))}),
        ("B", {"b": centred((3.25, 2.5, 0), (0.4, 0.4, 0.4))}),
        ("anchor", {"core": centred((0, 0, 0), (1, 1, 1))})])


def guard_past_saddles():
    # Two saddles come off first, straight out along +x from either side of a body. A thin guard
    # then slides out past the place the upper saddle went to: a guard's honest exit.
    return taken_apart([
        ("saddles", {"s1": centred((1.5, 1.5, 0), (0.3, 0.2, 0.3)),
                     "s2": centred((1.5, -1.5, 0), (0.3, 0.2, 0.3))}),
        ("guard", {"g": centred((3.25, 0.5, 0), (1.0, 0.1, 0.6))}),
        ("body", {"body": centred((0, 0, 0), (2, 0.5, 1))})])


@pytest.mark.parametrize("piece", [two_arm_group, guard_past_saddles])
def test_a_group_taken_off_earlier_is_gone_for_the_groups_after_it(piece):
    # A part removed earlier no longer exists for the parts removed after it, as in a real
    # teardown. Measured against the earlier group parked at the end of its exit, both pieces
    # cut it; the earlier group is gone, so neither is a finding.
    ctx = piece()
    assert Crossings().run(ctx) == []
    assert findings(ctx) == []
    assert plants_caught(TakeApart(), ctx)


def limit_in(detail: str) -> float:
    return float(detail.split("limit ")[1].split(")")[0].split(",")[0])


@pytest.mark.usefixtures("centre_line_only")
def test_take_apart_reads_the_same_limit():
    # A 0.05 pin seated past its limit in a 0.2 plate: crossings judges the seat at rest, the
    # sweep drives the pin deeper, and both name the limit of half the pin's width, found from
    # the parts and not from the pack.
    seated = pin_in_plate(seat_over_limit(0.05), 0.05)
    (at_rest,) = Crossings().run(seated)
    (swept,) = findings(seated)
    assert limit_in(at_rest.detail) == pytest.approx(CROSS_OF_PART_THICKNESS * 0.05, rel=0.1)
    # the two read the thickness at different points of the pair, so they agree to a few percent
    assert limit_in(swept.detail) == pytest.approx(limit_in(at_rest.detail), rel=0.05)


@pytest.mark.usefixtures("centre_line_only")
def test_a_group_that_grazes_by_less_than_half_the_thinner_part_is_clear():
    # the runner is a 0.4 cube, which a cone of rays reads a little thinner than 0.4, so its limit
    # is a little under 0.2: a reach of 0.12 into the way is under it, 0.26 is over
    assert findings(lane(overlap=0.12)) == []
    assert [f.item for f in findings(lane(overlap=0.26))] == [LANE]


@pytest.mark.usefixtures("centre_line_only")
def test_a_runner_driven_through_blockers_as_wide_as_itself_is_stopped():
    # equal widths: a 0.4 runner through 0.4 blockers that fill its whole lane
    assert [f.item for f in findings(lane(overlap=0.4))] == [LANE]


def test_a_pair_with_no_thickness_where_it_overlaps_is_unmeasured_and_the_run_goes_on(monkeypatch):
    parts = post_on_base(sink=CALIBRATION_FAIL * CROSS_OF_PART_THICKNESS * 0.02)  # seated deep
    real = geom.local_thickness
    seat = np.array([0.0, 0.01, 0.0])  # the middle of the space the seated post shares

    def patched(tris, point):  # an open patch of the post, where the seated parts overlap
        if np.array_equal(tris, parts["post"]) and np.linalg.norm(point - seat) < 0.015:
            raise ValueError("local_thickness: no ray hit an opposite surface")
        return real(tris, point)

    ctx = make_ctx(parts, GROUPS)
    assert not [x for x in TakeApart().coverage(ctx) if isinstance(x, Unmeasured)]  # sound
    monkeypatch.setattr(geom, "local_thickness", patched)
    assert findings(ctx) == []  # no number, and no exit 2
    (line,) = [x for x in TakeApart().coverage(ctx) if ITEM in x]
    assert isinstance(line, Unmeasured) and "no thickness there" in line  # so a run is partial
    # the deep-seated scene's plant is caught, so the gate does not stop on BlindCheck
    assert run_gate([TakeApart()], ctx, None).status == "partial"


@pytest.mark.usefixtures("centre_line_only")
def test_a_group_pair_with_a_part_pair_that_cannot_be_read_gets_no_finding(monkeypatch):
    """The whole group pair is left out: a finding from the pairs that read, beside an
    `Unmeasured` line for the one that does not, would say the item was judged."""
    ctx = lane()
    assert [f.item for f in findings(ctx)] == [LANE]
    real, calls = take_apart.measure, []

    def flaky(a, b):
        calls.append(1)
        if len(calls) % 7 == 0:
            raise Unmeasurable("the first part has no thickness there: x")
        return real(a, b)

    monkeypatch.setattr(take_apart, "measure", flaky)
    assert findings(ctx) == []
    assert any(isinstance(x, Unmeasured) and x.endswith(LANE) for x in TakeApart().coverage(ctx))


def test_an_open_part_is_unmeasured_and_not_swept():
    parts = post_on_base()
    ctx = make_ctx({"base": parts["base"], "post": parts["post"][:-2]}, GROUPS)  # a box with a hole
    assert findings(ctx) == []
    line = "take-apart: skipped (open part): kit:Thing post"
    assert line in TakeApart().coverage(ctx)
    assert isinstance(next(c for c in TakeApart().coverage(ctx) if c == line), Unmeasured)
    assert run_gate([TakeApart()], ctx, None).status == "partial"


# --- the 26 ways ------------------------------------------------------------------------------

CASE = "kit:Thing case x core"


def slabs_around(inner, walls):
    """Six slabs around the cavity of half-extents `inner`: `walls` is the thickness on each of
    +x, -x, +y, -y, +z, -z. The slabs reach out past the thickest wall on the other axes, so no
    way out of the cavity misses them."""
    inner, reach = np.array(inner, dtype=float), max(walls)
    out = {}
    for k, (axis, sign) in enumerate([(a, s) for a in range(3) for s in (1.0, -1.0)]):
        lo, hi = -(inner + reach), inner + reach
        lo[axis], hi[axis] = sorted((sign * inner[axis], sign * (inner[axis] + walls[k])))
        out[f"wall{k}"] = box(lo, hi)
    return out


def caged(core=((0, 0, 0), (0.4, 0.4, 0.4)), inner=(0.4, 0.4, 0.4), walls=(1.0,) * 6, extra=None):
    """A core cube walled in on every side, taken apart before its case. A core at the centre has
    no pull. `extra` adds groups (name -> {part: mesh}) built before the core, taken apart after
    the case."""
    case = slabs_around(inner, walls)
    groups = [("core", {"core": centred(*core)}), ("case", case)]
    return taken_apart(groups + list((extra or {}).items()))


def long_cage():
    """A long pin walled in along its length; an anchor far off makes its pull oblique to every
    wall."""
    return caged(core=((0, 0, 0), (0.1, 0.6, 0.1)), inner=(0.2, 0.5, 0.2),
                 extra={"anchor": {"anchor": centred((6, -4, 3), (3, 3, 3))}})


def direction_in(detail: str) -> np.ndarray:
    return np.array([float(x) for x in detail.split(" along (")[1].split(")")[0].split(",")])


@pytest.mark.parametrize("make", [caged, long_cage])
def test_group_boxed_in_on_all_sides_fails_on_two_pieces(make):
    (found,) = [f for f in findings(make()) if f.item.endswith("core")]
    assert found.item == CASE
    assert np.linalg.norm(direction_in(found.detail)) == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize("make, item", [(lambda: lane(), LANE),
                                        (lambda: pin_in_plate(0.0), "kit:Thing pin x plate")])
def test_group_freed_by_a_side_direction_passes_on_two_pieces(make, item, monkeypatch):
    """The centre line is blocked, and a way to the side is clear."""
    ctx = make()
    with monkeypatch.context() as only_the_pull:
        only_the_pull.setattr(take_apart, "DEPTH_WAYS", np.empty((0, 3)))
        assert [f.item for f in findings(ctx)] == [item]
    assert findings(ctx) == []
    assert plants_caught(TakeApart(), ctx)


def test_the_finding_names_the_least_deep_way():
    """Every wall is thick, the +x wall least: the core cuts least going out that way."""
    ctx = caged(walls=(0.5, 3.0, 3.0, 3.0, 3.0, 3.0))
    (found,) = findings(ctx)
    assert direction_in(found.detail) == pytest.approx([1.0, 0.0, 0.0], abs=2e-3)


@pytest.mark.parametrize("make", [lane, lambda: pin_in_plate(0.0), two_arm_group,
                                  lambda: lane(("a", "b", "anchor")), long_cage,
                                  lambda: make_ctx(post_on_base(sink=0.05), GROUPS)])
def test_never_stricter_than_the_centre_pull(make, monkeypatch):
    """Every group that comes away along its pull still comes away, so every pair the 26 ways
    report is one the pull alone reports. (A group with no pull is tried only by the 26 ways.)"""
    ctx = make()
    full = {f.item for f in findings(ctx)}
    with monkeypatch.context() as only_the_pull:
        only_the_pull.setattr(take_apart, "DEPTH_WAYS", np.empty((0, 3)))
        assert full <= {f.item for f in findings(ctx)}


def test_cable_does_not_block_its_plugs():
    """A core walled in by six rigid slabs fails. With one slab a cable, the cable can be pushed
    aside: the core comes away, and the cable is left out of the sweep with a plain line."""
    assert [f.item for f in findings(caged())] == [CASE]
    ctx = caged()
    piece = ctx.packs["kit"].pieces["Thing"]
    parts = tuple(dataclasses.replace(p, shape="cable") if p.name == "wall0" else p
                  for p in piece.parts)
    pack = dataclasses.replace(ctx.packs["kit"], pieces={
        "Thing": dataclasses.replace(piece, parts=parts)})
    ctx = dataclasses.replace(ctx, packs={"kit": pack})
    assert findings(ctx) == []
    line = ("take-apart: not applicable (cable part, it bends: it neither moves nor blocks): "
            "kit:Thing wall0")
    assert line in TakeApart().coverage(ctx)
    assert "kit:Thing wall0" not in TakeApart().measured(ctx)
    assert run_gate([TakeApart()], ctx, None).status == "passed"


def test_an_open_cable_is_not_an_unmeasured_part():
    parts = {**post_on_base(), "lead": box((0.3, 0, 0), (0.5, 0.01, 0.01))[:-2]}
    ctx = make_ctx(parts, {**GROUPS, "lead": {"group": "post", "shape": "cable"}})
    assert not [x for x in TakeApart().coverage(ctx) if isinstance(x, Unmeasured)]


def test_single_group_piece_has_no_unmeasured_line():
    """A piece in one group has nothing to take apart, so an open part in it is not
    something take-apart failed to measure."""
    parts = {**post_on_base(), "lid": box((0.3, 0, 0), (0.5, 0.01, 0.01))[:-2]}
    ctx = make_ctx(parts)
    lines = TakeApart().coverage(ctx)
    assert lines == ["take-apart: not applicable (fewer than two swept groups): kit:Thing"]
    assert not any(isinstance(x, Unmeasured) for x in lines)
    assert run_gate([TakeApart()], ctx, None).status == "passed"


def test_the_check_cites_its_principle_and_its_source():
    check = TakeApart()
    assert check.principle.endswith(".") and ". " not in check.principle
    assert "Wilson and Latombe 1994" in check.source and "section 5.3 (p. 384)" in check.source
    assert "UNVERIFIED" not in check.source


def test_the_shortcut_never_changes_a_reading(monkeypatch):
    scenes = [lane(), lane(overlap=CROSS / 2), lane(("anchor", "b", "a")), two_arm_group(),
              guard_past_saddles(), pin_in_plate(seat_over_limit(0.1)),
              pin_in_plate(seat_over_limit(0.1), anchor_y=-6.0), pin_in_plate(0.0),
              make_ctx(post_on_base(sink=0.05), GROUPS), caged(), long_cage()]
    fast = [TakeApart().run(ctx) for ctx in scenes]
    monkeypatch.setattr(crossings, "_all_leave", lambda sides: False)
    assert fast == [TakeApart().run(ctx) for ctx in scenes]


@pytest.mark.parametrize("scene,mover", [(sheet_and_knob, "knob"), (strip_and_post, "post")])
def test_the_plant_boxes_in_the_cheapest_group_not_the_fewest_triangles(scene, mover):
    parts = scene()
    ctx = make_ctx(parts, {name: {"group": name} for name in parts})
    (unit,) = TakeApart().plants(ctx)
    assert unit.item.split(" ")[1] == mover
    assert plants_caught(TakeApart(), ctx)
