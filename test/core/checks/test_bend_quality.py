"""Check bend-quality: the plants it must see, the lines it must print, and the items it names.

The test bodies are `test/core/bodies`: a humanoid, a quadruped with a tail and a worm, each with
real skin weights and joint ranges. They are skinned by linear blend skinning, and a single bone
hinge under that skinning pinches as a fixed function of its angle, whatever the weights or the
number of vertices in the band: a ring at share w of the blend loses girth by the same amount on
every body. The check is not tuned to them: the findings it makes there are real, so
the tests that need a quiet slice pick one the body is quiet on.
"""

import dataclasses
import math
import re
from functools import cache

import numpy as np
import pytest

from model_kit.core.checks import bend_quality as bq
from model_kit.core.checks.bend_quality import NAME, PRINCIPLE, SOURCE, BendQuality
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.slices import ring_slices
from model_kit.core.tolerances import BEND_AREA_FLOOR, BEND_GIRTH_CHANGE

from ..bodies.build import sweep
from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene
from .piece_scenes import chosen, tried_and_caught

BODIES = {"humanoid": humanoid_scene, "quadruped": quadruped_scene, "worm": worm_scene}
ITEM = re.compile(r"^\S+:\S+ \S+ \S+ slice [1-7] (girth|collapse) at .+$")


@cache
def scene(body: str):
    return BODIES[body]()


@cache
def checker(body: str) -> BendQuality:
    """One check per body for the tests that only read: it keeps what it measured."""
    return BendQuality()


def rig_of(ctx):
    key = next(iter(ctx.rigs))
    pack, _, name = key.partition(":")
    return key, ctx.packs[pack].pieces[name]


def with_rig(ctx, **fields):
    """The scene with the rigged piece's spec changed."""
    key, piece = rig_of(ctx)
    pack, _, name = key.partition(":")
    bare = dataclasses.replace(piece, rig=dataclasses.replace(piece.rig, **fields))
    pieces = {**ctx.packs[pack].pieces, name: bare}
    return dataclasses.replace(ctx, packs={**ctx.packs, pack: dataclasses.replace(
        ctx.packs[pack], pieces=pieces)})


def caught(check, plant) -> bool:
    found = check.run(plant.ctx, only=plant.scope, samples=plant.samples)
    return any(f.item == plant.item for f in found)


def kind_of(item: str) -> str:
    return re.search(r" slice \d (girth|collapse) at ", item).group(1)


def plants_of(body: str, kind: str):
    return [p for p in chosen(checker(body), scene(body)) if kind_of(p.item) == kind]


# --- plants ----------------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_pinch_plant_caught_on_two_bodies(body):
    """A band of the skin shrunk toward the bone axis by half at one clip sample: its girth."""
    plants = plants_of(body, "girth")
    assert plants and all(caught(checker(body), p) for p in plants)
    assert all(p.item.startswith(f"{rig_of(scene(body))[0]} ") and " frame " in p.item
               for p in plants)
    assert all(len(p.samples) == 1 for p in plants)


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_twist_plant_caught_on_two_bodies(body):
    """A band turned end for end about the axis loses its area (the candy wrapper): collapse."""
    plants = plants_of(body, "collapse")
    assert plants and all(caught(checker(body), p) for p in plants)
    for plant in plants:
        found = checker(body).run(plant.ctx, only=plant.scope, samples=plant.samples)
        area = next(f for f in found if f.item == plant.item)
        assert "under the floor" in area.detail


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_a_plant_is_a_finding_only_because_it_was_planted(body):
    """The slice a plant goes at is quiet without the fault."""
    check = checker(body)
    for plant in chosen(check, scene(body)):
        clean = check.run(scene(body), only={plant.item}, samples=plant.samples)
        assert not [f for f in clean if f.item == plant.item], plant.label


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_the_meta_gate_accepts_every_plant(body):
    """`run_gate` raises `BlindCheck` on a missed plant; any verdict means every plant was seen."""
    run_gate([checker(body)], scene(body), None)


def test_the_plants_do_not_reach_the_scene_they_came_from():
    ctx = scene("worm")
    check = BendQuality()
    before = check.run(ctx)
    [c.ctx for u in check.plants(ctx) for c in u.candidates]
    assert [f.line() for f in check.run(ctx)] == [f.line() for f in before]


def test_a_90_degree_twist_cannot_take_a_ring_under_the_area_floor():
    """Why the twist plant turns a band end for end. Two end rings, the far one turned `angle`: the
    rings between are blends of a ring and itself turned, and the ring at a share f of the way
    holds (1 - f)^2 + f^2 of the area at 90 degrees, never under 0.5; at 180 degrees (1 - 2f)^2."""
    verts, faces = sweep([(0.0, 0, 0, 0.05, 0.05), (1.0, 0, 0, 0.05, 0.05)], "y",
                         caps=("flat", "flat"))
    rest = 0.5 * 16 * 0.05**2 * math.sin(2 * math.pi / 16)  # the round ring's polygon
    for angle, floor_reached in ((math.pi / 2, False), (math.pi, True)):
        turned = verts.astype(float)
        far = turned[:, 1] > 0.5
        turned[far] = bq._turn(turned[far], np.zeros(3), np.array([0.0, 1.0, 0.0]), angle)
        weight = turned[:, 1].copy()
        weight[far] = 1.0
        weight[~far] = 0.0
        area = np.array([r.area for r in ring_slices(turned[faces], weight[faces], (0, 0.5, 0), 19)])
        assert bool((area / rest).min() < BEND_AREA_FLOOR) is floor_reached, (angle, area.min())


# --- what each test finds --------------------------------------------------------------------


def test_the_standing_body_is_quiet_where_nothing_bends():
    """A clip that only breathes (a degree or so at the spine and neck), and one that lifts a foot
    and keeps it level, bend no joint past the limits: no finding at their samples."""
    for body in BODIES:
        found = checker(body).run(scene(body))
        assert not [f for f in found if re.search(r" at (stand|step) frame ", f.item)], body


def test_findings_name_the_rig_the_part_the_joint_and_where():
    for body in BODIES:
        ctx = scene(body)
        found = checker(body).run(ctx)
        assert all(f.check == NAME for f in found)
        assert all(ITEM.match(f.item) for f in found), [f.item for f in found
                                                        if not ITEM.match(f.item)][:3]
        assert all(f.item.startswith(rig_of(ctx)[0] + " ") for f in found)
        assert len({f.item for f in found}) == len(found)


def test_a_finding_is_at_a_clip_sample_and_never_at_a_stress_pose():
    """Judge a piece only as it will be seen: every finding names a frame of a clip the rig uses,
    and no stress pose (the end of a joint's range) is judged or even built."""
    found = checker("humanoid").run(scene("humanoid"))
    assert any(re.search(r" at sit_chair frame \d+$", f.item) for f in found)
    for body in BODIES:
        ctx = scene(body)
        key, _ = rig_of(ctx)
        clips = {c for c, k in ctx.rig_of_clip.items() if k == key}
        for f in checker(body).run(ctx):
            where = f.item.partition(" at ")[2]
            assert where.rpartition(" frame ")[0] in clips, f.item
        assert not any(re.search(r" x (lo|hi)\b", f.item) for f in checker(body).run(ctx))
    ctx = scene("humanoid")
    assert checker("humanoid").run(ctx, only={
        f"{rig_of(ctx)[0]} arm_L elbow.L slice 4 girth at elbow.L x lo"}) == []


def test_girth_and_collapse_report_their_numbers():
    found = checker("humanoid").run(scene("humanoid"))
    girth = next(f for f in found if " girth at " in f.item)
    assert re.search(r"perimeter [\d.]+ against [\d.]+ at rest \([+-][\d.]+%\), over the allowed "
                     rf"{BEND_GIRTH_CHANGE:.1%}", girth.detail)
    plant = plants_of("humanoid", "collapse")[0]  # no clip sample of the body collapses a ring
    collapse = next(f for f in checker("humanoid").run(plant.ctx, only=plant.scope,
                                                       samples=plant.samples)
                    if f.item == plant.item)
    assert re.search(rf"area [\d.]+ is [\d.]+% of [\d.]+ at rest, under the floor "
                     rf"{BEND_AREA_FLOOR:.0%}", collapse.detail)


def test_only_measures_just_the_items_it_names():
    ctx = scene("humanoid")
    check = checker("humanoid")
    every = check.run(ctx)
    pick = next(f for f in every if " at sit_chair frame " in f.item)
    assert [f.item for f in check.run(ctx, only={pick.item})] == [pick.item]
    assert check.run(ctx, only={"other:rig a b slice 1 girth at x frame 1"}) == []
    assert check.run(ctx, only=set()) == []


def test_samples_limit_a_run_to_those_clip_samples():
    ctx = scene("humanoid")
    check = checker("humanoid")
    some = check.run(ctx, samples=frozenset({0}))
    assert some and all(re.search(r" frame 1$", f.item) for f in some)  # the first sample only


def test_a_bend_that_the_cited_studies_saw_is_not_a_finding():
    """A 6.4% girth change (the knee at 90 degrees) is inside the limit: the limit is the change
    itself, and the check fails only what is over it."""
    assert BEND_GIRTH_CHANGE == pytest.approx(0.064)
    assert BEND_AREA_FLOOR == 0.5


def tube_girth(scale: float):
    """The findings of `_measure_joint` on a round tube whose seven rings all have their radius
    taken to `scale` times the rest radius, so each ring's perimeter is `scale` times its rest
    perimeter."""
    verts, faces = sweep([(0.0, 0, 0, 0.05, 0.05), (1.0, 0, 0, 0.05, 0.05)], "y",
                         caps=("flat", "flat"))
    verts = verts.astype(float)
    weight = verts[:, 1].copy()  # the child-side weight rises from 0 to 1 along the tube
    head = np.array([0.0, 0.5, 0.0])
    rest = ring_slices(verts[faces], weight[faces], head, bq.SLICES)
    joint = bq._Joint("tube", "hinge", faces, weight, rest)
    posed = verts.copy()
    posed[:, [0, 2]] *= scale
    matrix = np.eye(4)
    matrix[:3, 3] = head
    where = bq._Where("bend frame 1", {"tube": posed}, {"hinge": matrix})
    scan = bq._Scan()
    bq._measure_joint(bq._Prepared("rig:tube", [joint]), joint, where, scan, None)
    return scan


def test_a_girth_change_at_the_limit_is_quiet_and_a_tenth_of_a_percent_over_is_found():
    """The limit is inclusive. A float product of 1.064 lands 6e-17 over 6.4%, so the
    quiet case is the limit less one part in 10^12, which is 6.4% to the digits the check
    prints; 6.5% is found at every one of the seven rings."""
    quiet = tube_girth(1 + BEND_GIRTH_CHANGE - 1e-12)
    assert quiet.slices == bq.SLICES and quiet.findings == []
    found = tube_girth(1 + BEND_GIRTH_CHANGE + 0.001)
    assert sorted(f.item.split(" slice ")[1].split(" ")[0] for f in found.findings) == list("1234567")
    assert all(" girth at bend frame 1" in f.item and "(+6.5%)" in f.detail for f in found.findings)
    shrunk = tube_girth(1 - BEND_GIRTH_CHANGE - 0.001)  # the limit is on the size, not the sign
    assert len(shrunk.findings) == bq.SLICES and "(-6.5%)" in shrunk.findings[0].detail


# --- coverage --------------------------------------------------------------------------------


def test_the_shape_ratio_is_reported_and_not_judged():
    lines = checker("humanoid").coverage(scene("humanoid"))
    shape = [x for x in lines if "shape ratio" in x]
    assert len(shape) == 1 and "reported and not judged" in shape[0]
    assert not isinstance(shape[0], Unmeasured)
    assert all("shape" not in f.item and "shape" not in f.detail
               for f in checker("humanoid").run(scene("humanoid")))


def test_rigid_part_not_applicable():
    """A part bound to one bone does not bend: a not-applicable line, no item, no plant on it."""
    ctx = scene("quadruped")
    key, _ = rig_of(ctx)
    rest = ctx.rests[key]
    part = "leg_hind_L"
    n = len(rest.labels[part])
    one = dataclasses.replace(
        rest, vertex_groups={**rest.vertex_groups, part: rest.vertex_groups[part][:1]},
        labels={**rest.labels, part: np.zeros(n, np.int32)},
        weights={**rest.weights, part: np.ones((n, 1), np.float32)},
        weight_groups={**rest.weight_groups, part: np.zeros((n, 1), np.int32)})
    rigid = dataclasses.replace(ctx, rests={key: one})
    check = BendQuality()
    lines = check.coverage(rigid)
    assert [x for x in lines if "not applicable" in x and part in x] == [
        f"{NAME}: not applicable (the part is bound to one bone and does not bend): {key} {part}"]
    assert not any(isinstance(x, Unmeasured) for x in lines)
    assert all(f" {part} " not in f.item for f in check.run(rigid))
    assert all(f" {part} " not in p.item for p in check.plants(rigid))


def test_a_body_of_rigid_parts_is_passed_with_a_line_each():
    ctx = with_rig(scene("humanoid"), bind={})  # no part binds anything: no joint bends
    check = BendQuality()
    assert check.run(ctx) == [] and check.plants(ctx) == []


def test_missing_weights_is_partial():
    ctx = scene("worm")
    key, _ = rig_of(ctx)
    rest = dataclasses.replace(ctx.rests[key], weights={}, weight_groups={})
    bare = dataclasses.replace(ctx, rests={key: rest})
    check = BendQuality()
    lines = check.coverage(bare)
    skipped = [x for x in lines if isinstance(x, Unmeasured)]
    assert len(skipped) == 1 and "no skin weights" in skipped[0] and key in skipped[0]
    assert check.run(bare) == [] and check.plants(bare) == []
    assert run_gate([check], bare, None).status == "partial"


def test_a_rig_with_no_rest_file_is_unmeasured():
    ctx = dataclasses.replace(scene("worm"), rests={})
    assert any(isinstance(x, Unmeasured) and "no rest file" in x
               for x in BendQuality().coverage(ctx))


def test_a_missing_joint_range_is_no_concern_of_this_check():
    """The check reads the clips, not the ends of the ranges, so a rig with no range for a joint
    (or for any) is measured at its clip samples in full and is not made partial by it."""
    for ctx in (with_rig(scene("humanoid"), joints={}),
                with_rig(scene("quadruped"), joints={k: v for k, v in rig_of(
                    scene("quadruped"))[1].rig.joints.items() if k != "fore_elbow.L"})):
        check = BendQuality()
        assert not [x for x in check.coverage(ctx) if isinstance(x, Unmeasured)]
        assert run_gate([check], ctx, None).status != "partial"
    whole = BendQuality().run(scene("humanoid"))
    bare = BendQuality().run(with_rig(scene("humanoid"), joints={}))
    assert bare and [f.line() for f in bare] == [f.line() for f in whole]


def test_unmeasurable_never_escapes_a_run():
    """A ring that closes nowhere is a coverage line."""
    ctx = scene("worm")
    key, _ = rig_of(ctx)
    shells = {p: np.full_like(v, 1e6) for p, v in ctx.samples["stand"].shells.items()}
    far = dataclasses.replace(ctx, samples={"stand": dataclasses.replace(
        ctx.samples["stand"], shells=shells)})
    check = BendQuality()
    check.run(far)
    lines = check.coverage(far)
    assert any(isinstance(x, Unmeasured) and "no closed outline" in x for x in lines), lines
    assert run_gate([check], far, None).status == "partial"
    assert key in " ".join(lines)


def test_a_scene_with_no_rig_has_nothing_to_say():
    ctx = dataclasses.replace(scene("humanoid"), rigs={})
    check = BendQuality()
    assert check.coverage(ctx) == [] and check.plants(ctx) == [] and check.run(ctx) == []


# --- the check's face ------------------------------------------------------------------------


def test_the_check_states_its_principle_and_its_source():
    check = BendQuality()
    assert check.name == "bend-quality" and check.kinds == frozenset()
    assert check.principle == PRINCIPLE and check.source == SOURCE
    assert PRINCIPLE.count(". ") == 0 and PRINCIPLE.endswith(".")  # one sentence
    assert "2312.04108" in SOURCE and "Ringenbach" in SOURCE and "Klepser" in SOURCE
    assert "design call" in SOURCE


# --- the plants at named joints --------------------------------------------------------------


@pytest.mark.parametrize("body, joint", [("humanoid", "elbow.L"), ("quadruped", "hind_knee.L")])
@pytest.mark.parametrize("kind", ["girth", "collapse"])
def test_a_plant_on_the_elbow_and_on_the_knee_goes_through_the_gate(body, joint, kind):
    """The gate's plants are one unit per bending joint, so the elbow and the knee are
    planted as much as the chest, at a clip sample where they are quiet without it."""
    ctx = scene(body)
    check = BendQuality()
    (unit,) = [u for u in check.plants(ctx)
               if f" {joint} slice " in u.item and kind_of(u.item) == kind]
    plant = next(p for p in unit.candidates if caught(check, p))
    assert plant.item == unit.item and len(plant.samples) == 1
    clean = check.run(ctx, only={plant.item}, samples=plant.samples)
    assert not [f for f in clean if f.item == plant.item]


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_every_bending_joint_inside_a_part_gets_a_unit_per_kind(body):
    """The units are the joints the check measures, each twice, none left to the largest part."""
    ctx = scene(body)
    key, _ = rig_of(ctx)
    check = BendQuality()
    measured = {(j.part, j.joint) for j in check._prep(ctx, key).joints}
    units = [u for u in check.plants(ctx) if not u.not_applicable]
    got = {(u.item.split(" ")[1], u.item.split(" ")[2], kind_of(u.item)) for u in units}
    assert got == {(part, joint, kind) for part, joint in measured
                   for kind in ("girth", "collapse")}
    assert tried_and_caught(check, ctx)  # each unit the gate tries has a candidate it catches


# --- a part blended across a joint, and a row with no weight --------------------------------------


def blended_arm(ctx, low=0.1, high=0.9):
    """The humanoid with its left arm's elbow share compressed into [low, high]: every vertex is
    blended across the elbow and none is wholly on a side. Each row carries two slots, shoulder
    and elbow, weighted to give that share."""
    key, piece = rig_of(ctx)
    rest = ctx.rests[key]
    part = "arm_L"
    parent = {b.name: b.parent for b in piece.rig.bones}
    share = bq._child_weight(rest, part, bq._below(parent, "elbow.L"), bq._tree(parent, "elbow.L"))
    share = low + (high - low) * share
    names = list(rest.vertex_groups[part])
    slots = np.tile([names.index("shoulder.L"), names.index("elbow.L")], (len(share), 1))
    weights = np.stack([1 - share, share], axis=1).astype(np.float32)
    return dataclasses.replace(ctx, rests={key: dataclasses.replace(
        rest, weights={**rest.weights, part: weights},
        weight_groups={**rest.weight_groups, part: slots.astype(np.int32)})}), key


def test_a_part_blended_across_a_joint_with_no_vertex_wholly_on_a_side_is_measured():
    """Such a part is not skipped as `no bending joint lies inside the part`: its rings are read."""
    ctx, key = blended_arm(scene("humanoid"))
    prep = BendQuality()._prep(ctx, key)
    assert ("arm_L", "elbow.L") in {(j.part, j.joint) for j in prep.joints}
    assert not [x for x in prep.lines if "arm_L" in x and "elbow.L" in x]


def test_a_part_blended_too_narrowly_for_its_rings_is_unmeasured_not_skipped():
    ctx, _ = blended_arm(scene("humanoid"), low=0.45, high=0.55)
    lines = [str(x) for x in BendQuality().coverage(ctx) if "arm_L elbow.L" in x]
    assert lines and all("skipped" in x and "no closed outline" in x for x in lines), lines
    assert not any("not applicable" in x and "arm_L" in x for x in BendQuality().coverage(ctx))


def test_a_part_wholly_on_one_side_of_a_joint_is_still_not_applicable():
    weight = np.full(10, 0.0)
    assert not bq._reaches_a_ring(weight) and not bq._reaches_a_ring(weight + 1.0)
    assert bq._reaches_a_ring(np.linspace(0.1, 0.9, 10)) and bq._reaches_a_ring(np.linspace(0, 1, 5))
    assert not bq._reaches_a_ring(np.full(3, np.nan))
    assert not bq._reaches_a_ring(np.array([0.95, 0.99, np.nan]))


def test_a_vertex_with_no_skin_weight_is_unmeasured_not_a_nan_shared_across_the_joint():
    """A zero-sum row divided by its own sum gives NaN, which reads as another tree's vertex
    and drops out of the rings in silence."""
    ctx = scene("humanoid")
    key, _ = rig_of(ctx)
    rest = ctx.rests[key]
    zero = np.array(rest.weights["arm_L"])
    zero[0, :] = 0
    bare = dataclasses.replace(ctx, rests={key: dataclasses.replace(
        rest, weights={**rest.weights, "arm_L": zero})})
    check = BendQuality()
    lines = [x for x in check.coverage(bare) if "arm_L" in x]
    assert len(lines) == 1 and isinstance(lines[0], Unmeasured) and "no skin weight" in lines[0]
    assert not any(" arm_L " in f.item for f in check.run(bare))
    assert run_gate([BendQuality()], bare, None).status == "partial"
