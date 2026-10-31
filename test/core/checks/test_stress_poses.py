"""Check stress-poses: the plants it must see, the lines it must print, and the items it names.

The test bodies are `test/core/bodies`: a humanoid, a quadruped with a tail and a worm, each with
real skin weights and joint ranges. Their joint ranges are cited or marked as design choices, and
the check is not tuned to them: the findings it makes at range ends are real folds and contacts,
so the tests that need a quiet pose pick one by name.
"""

import dataclasses
import re
from functools import cache

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.allow import Allowlist
from model_kit.core.checks import stress_poses as sp
from model_kit.core.checks.overlap import body_contact_plants, body_contacts, deep_point
from model_kit.core.checks.stress_poses import NAME, PRINCIPLE, SOURCE, StressPoses
from model_kit.core.errors import Unmeasurable
from model_kit.core.geom import SurfaceIndex, self_cuts
from model_kit.core.manifest import Bind
from model_kit.core.metagate import Finding, Unmeasured, run_gate
from model_kit.core.pose.stress import STRESS_CLIP, stress_poses, stress_scene
from model_kit.core.tolerances import CALIBRATION_FAIL

from ..bodies.build import sweep
from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene
from .piece_scenes import chosen, tried_and_caught

BODIES = {"humanoid": humanoid_scene, "quadruped": quadruped_scene, "worm": worm_scene}
ITEM = re.compile(r"^\S+:\S+ .+ (C rig:\S+ x rig:\S+|self \S+|inverted \S+)$")


@cache
def scene(body: str):
    return BODIES[body]()


@cache
def checker(body: str) -> StressPoses:
    """One check per body for the tests that only read: it keeps the stressed scene it built."""
    return StressPoses()


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


def pose_of(item: str) -> str:
    """The pose name in an item: what lies between the rig key and the kind."""
    return re.split(r" (?:C|self|inverted) ", item.partition(" ")[2], maxsplit=1)[0]


def kind_of(item: str) -> str:
    return re.search(r" (C|self|inverted) ", item).group(1)


def caught(check, plant) -> bool:
    found = check.run(plant.ctx, only=plant.scope, samples=plant.samples)
    return any(f.item == plant.item for f in found)


# --- plants ----------------------------------------------------------------------------------


@pytest.mark.parametrize("body", ["humanoid", "quadruped"])
def test_plant_caught_on_humanoid_and_quadruped(body):
    """A vertex pushed 2 skins into a shell it is not excused from, at one stress pose."""
    check = checker(body)
    plants = [p for p in chosen(check, scene(body)) if kind_of(p.item) == "C"]
    assert plants and all(caught(check, p) for p in plants)
    assert all(p.item.startswith(f"{rig_of(scene(body))[0]} ") for p in plants)
    assert all(len(p.samples) == 1 for p in plants)


@pytest.mark.parametrize("body", ["humanoid", "worm"])
def test_one_mesh_self_crossing_caught(body):
    """A vertex of a one-mesh part pushed through the part's far wall: no other shell is touched.
    The plant goes on the part with the widest faces: a cut is no deeper than its faces are wide."""
    check = checker(body)
    plants = [p for p in chosen(check, scene(body)) if kind_of(p.item) == "self"]
    assert plants and all(caught(check, p) for p in plants)
    ctx = scene(body)
    parts = {p.item.split(" self ")[1] for p in plants}
    assert len(parts) == 1 and parts <= set(next(iter(ctx.rests.values())).shells)


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_inverted_face_caught(body):
    check = checker(body)
    plants = [p for p in chosen(check, scene(body)) if kind_of(p.item) == "inverted"]
    assert plants and all(caught(check, p) for p in plants)


@pytest.mark.parametrize("body", ["humanoid", "quadruped", "worm"])
def test_a_plant_is_a_finding_only_because_it_was_planted(body):
    """The pose a plant goes at is quiet for that item without the fault."""
    check = checker(body)
    for plant in chosen(check, scene(body)):
        clean = check.run(scene(body), only={plant.item}, samples=plant.samples)
        assert not [f for f in clean if f.item == plant.item], plant.label


@pytest.mark.parametrize("body", ["quadruped", "worm"])
def test_the_meta_gate_accepts_every_plant(body):
    """`run_gate` raises `BlindCheck` on a missed plant; any verdict means every plant was seen."""
    run_gate([checker(body)], scene(body), None)


def test_the_worm_has_one_shell_so_body_in_body_does_not_apply():
    check = checker("worm")
    assert {kind_of(p.item) for p in chosen(check, scene("worm"))} == {"self", "inverted"}
    key = rig_of(scene("worm"))[0]
    (unit_c,) = [u for u in check.plants(scene("worm")) if u.item == f"{key} C"]
    assert unit_c.not_applicable == "fewer than two closed body shells" and not unit_c.candidates
    lines = [x for x in run_gate([check], scene("worm"), None).coverage
             if "closed body shells" in x]
    assert lines == [f"{NAME}: not applicable (fewer than two closed body shells): {key} C"]
    assert not [x for x in check.coverage(scene("worm")) if isinstance(x, Unmeasured)]


def test_the_plants_do_not_reach_the_scene_they_came_from():
    ctx = scene("worm")
    check = StressPoses()
    before = check.run(ctx)
    [c.ctx for u in check.plants(ctx) for c in u.candidates]
    assert [f.line() for f in check.run(ctx)] == [f.line() for f in before]


# --- what each test finds --------------------------------------------------------------------


def test_a_pose_equal_to_rest_has_no_finding():
    """An elbow at its high end (0 degrees) is the rest body: nothing to find in any test."""
    ctx = scene("humanoid")
    names = [p.name for p in stress_poses(rig_of(ctx)[1].rig)]
    at = names.index("elbow.L x hi")
    assert StressPoses().run(ctx, samples=frozenset({at})) == []


def test_findings_name_the_rig_the_pose_and_what_failed():
    ctx = scene("quadruped")
    found = StressPoses().run(ctx)
    assert found and all(f.check == NAME for f in found)
    assert all(ITEM.match(f.item) for f in found), [f.item for f in found if not ITEM.match(f.item)]
    assert all(f.item.startswith(rig_of(ctx)[0] + " ") for f in found)
    assert len({f.item for f in found}) == len(found)


def test_only_measures_just_the_items_it_names():
    ctx = scene("quadruped")
    check = checker("quadruped")
    every = check.run(ctx)
    pick = next(f for f in every if " C " in f.item)
    assert [f.item for f in check.run(ctx, only={pick.item})] == [pick.item]
    assert check.run(ctx, only={"other:rig x C rig:a/b x rig:c/d"}) == []
    assert check.run(ctx, only=set()) == []
    other = next(f for f in every if f.item != pick.item)
    assert {f.item for f in check.run(ctx, only={pick.item, other.item})} == {
        pick.item, other.item}


def test_a_skinned_face_turned_past_a_right_angle_is_inverted():
    """The humanoid's arm at a range end folds faces past a right angle (the joint's own crease is
    excused for crossings, not for a face turned inside out); an item names the part and how far
    the face turned."""
    ctx = scene("humanoid")
    found = [f for f in checker("humanoid").run(ctx) if " inverted " in f.item]
    assert found and all(re.search(r"turned \d+\.\d deg", f.detail) for f in found)
    assert all(float(re.search(r"turned ([\d.]+) deg", f.detail).group(1)) > 90 for f in found)


def test_one_allow_line_covers_one_pose(tmp_path):
    ctx = scene("quadruped")
    check = StressPoses()
    plant = next(p for p in chosen(check, ctx) if kind_of(p.item) == "C")
    here = check.run(plant.ctx, only={plant.item})
    assert [f.item for f in here] == [plant.item]
    pose = plant.item.split(" C ")[0]  # "<pack>:<piece> <pose name>"
    path = tmp_path / "allow.txt"
    path.write_text(f"{NAME} | {pose} * | the coupling's citation\n")
    kept, allowed = Allowlist.load(path).apply(here)
    assert kept == [] and len(allowed) == 1
    # the same line leaves a finding at any other pose, and any other kind at this pose is covered
    other = Finding(NAME, f"{rig_of(ctx)[0]} tail_1 x lo C rig:a/b x rig:c/d", "x")
    same = Finding(NAME, f"{pose} self body", "x")
    kept, allowed = Allowlist.load(path).apply([other, same])
    assert kept == [other] and [f for f, _ in allowed] == [same]


# --- the shortcut ----------------------------------------------------------------------------


def test_leaving_out_what_moved_together_changes_no_finding():
    """The check skips pairs of vertices and faces that a pose carries as one rigid body. Those
    stand as they did at rest, so the findings are the ones a test of every pair makes."""
    ctx = scene("quadruped")
    key, _ = rig_of(ctx)
    stressed, poses = stress_scene(ctx, key)
    clip = stressed.clips[STRESS_CLIP]
    brute = {f"{key} {poses[c.sample].name} C {c.a} x {c.b}": c.depth
             for c in body_contacts(stressed, clip)}
    found = {f.item: f.detail for f in checker("quadruped").run(ctx) if " C " in f.item}
    assert found.keys() == brute.keys() and brute
    for item, depth in brute.items():
        assert f"depth {depth:.4f} " in found[item], item


def test_self_crossings_are_the_same_with_and_without_the_shortcut():
    """At the poses that still hold a self-crossing (two joints at their range ends, off the
    crease) and one that holds only a fold, `_self` gives the same answer for every part whether
    or not it leaves out what moved together."""
    ctx = scene("humanoid")
    check = checker("humanoid")
    rig, _ = check._prepare(ctx, rig_of(ctx)[0])
    fixed, data = rig.fixed, rig.scene.samples[STRESS_CLIP]
    named = {f.item.partition(" ")[2].rsplit(" self ", 1)[0]
             for f in check.run(ctx) if " self " in f.item}
    pick = [*sorted(named)[::13][:3], "elbow.L x lo"]
    poses = [i for i, p in enumerate(fixed.poses) if p.name in pick]
    assert len(poses) == 4
    every = {}
    for s in poses:
        for part in fixed.parts:
            alone = np.full(data.shells[part].shape[1], -1)
            every[s, part] = sp._self(fixed, data, s, part, alone)
    assert any(v is not None for v in every.values())
    for s in poses:
        bodies = sp._rigid(fixed, rig.scene, rig.rig, s, True)
        for part in fixed.parts:
            got, want = sp._self(fixed, data, s, part, bodies[part]), every[s, part]
            assert (got is None) == (want is None), (s, part)
            if want is not None:
                assert got[1:] == want[1:] and got[0] == pytest.approx(want[0], rel=1e-9)


# --- coverage --------------------------------------------------------------------------------


def test_missing_joint_range_is_partial():
    ctx = scene("quadruped")
    _, piece = rig_of(ctx)
    joints = {k: v for k, v in piece.rig.joints.items() if k != "fore_elbow.L"}
    short = with_rig(ctx, joints=joints)
    check = StressPoses()
    lines = check.coverage(short)
    line = next(x for x in lines if isinstance(x, Unmeasured))
    assert "fore_elbow.L" in line and "no range" in line
    measured = next(x for x in lines if "measured" in x)
    full = next(x for x in check.coverage(ctx) if "measured" in x)
    assert int(re.search(r"measured (\d+)", measured).group(1)) < int(
        re.search(r"measured (\d+)", full).group(1))  # the other joints are still taken
    found = check.run(short)
    assert found and not any("fore_elbow.L" in pose_of(f.item) for f in found)
    assert run_gate([check], short, None).status == "partial"


def test_every_joint_unranged_is_one_unmeasured_line_and_no_plants():
    ctx = with_rig(scene("worm"), joints={})
    check = StressPoses()
    assert check.plants(ctx) == [] and check.run(ctx) == []
    lines = check.coverage(ctx)
    assert any(isinstance(x, Unmeasured) and "no range" in x for x in lines)
    assert any(isinstance(x, Unmeasured) and "no joint has a range" in x for x in lines)


def test_a_part_without_skin_weights_is_unmeasured_not_a_crash():
    ctx = scene("worm")
    key, _ = rig_of(ctx)
    rest = dataclasses.replace(ctx.rests[key], weights={}, weight_groups={})
    bare = dataclasses.replace(ctx, rests={key: rest})
    lines = StressPoses().coverage(bare)
    assert len(lines) == 1 and isinstance(lines[0], Unmeasured)
    assert "no skin weights" in lines[0] and key in lines[0]
    assert run_gate([StressPoses()], bare, None).status == "partial"


def test_a_rig_with_no_rest_file_is_unmeasured():
    ctx = dataclasses.replace(scene("worm"), rests={})
    assert any(isinstance(x, Unmeasured) and "no rest file" in x
               for x in StressPoses().coverage(ctx))


def test_rig_with_no_bending_joints_is_not_applicable():
    ctx = scene("humanoid")
    _, piece = rig_of(ctx)
    one = {part: Bind((piece.rig.root,), "rigid") for part in piece.rig.bind}
    rigid = with_rig(ctx, bind=one)
    check = StressPoses()
    lines = check.coverage(rigid)
    key = rig_of(ctx)[0]
    assert lines == [f"{NAME}: not applicable (the rig has no bending joint): {key}"]
    assert not any(isinstance(x, Unmeasured) for x in lines)
    assert check.plants(rigid) == [] and check.run(rigid) == []
    assert run_gate([check], rigid, None).status == "passed"


def test_a_scene_with_no_rig_has_nothing_to_say():
    ctx = dataclasses.replace(scene("humanoid"), rigs={})
    check = StressPoses()
    assert check.coverage(ctx) == [] and check.plants(ctx) == [] and check.run(ctx) == []


def test_coverage_says_how_many_poses_were_taken():
    ctx = scene("quadruped")
    key, piece = rig_of(ctx)
    line = [x for x in StressPoses().coverage(ctx) if "measured" in x]
    assert line == [f"{NAME}: measured {len(stress_poses(piece.rig))} poses: {key}"]


# --- the check's face ------------------------------------------------------------------------


def test_the_check_states_its_principle_and_its_source():
    check = StressPoses()
    assert check.name == "stress-poses" and check.kinds == frozenset()
    assert check.principle == PRINCIPLE and check.source == SOURCE
    assert PRINCIPLE.count(". ") == 0 and PRINCIPLE.endswith(".")  # one sentence
    assert "hendriks_gox-d-20-00155r2_sdc1.pdf" in SOURCE and "Hendriks et al. 2020" in SOURCE


# --- the seam with overlap -------------------------------------------------------------------


def test_body_contacts_reads_a_planted_scene_as_overlap_does():
    ctx = scene("humanoid")
    key, _ = rig_of(ctx)
    stressed, _ = stress_scene(ctx, key)
    clip = stressed.clips[STRESS_CLIP]
    s = 0
    a, b, copies = next(body_contact_plants(stressed, clip, s))
    moved = copies[-1]  # the deepest: all the room the shell has
    seen = [c for c in body_contacts(moved, clip, frozenset({s})) if (c.a, c.b) == (a, b)]
    assert len(seen) == 1 and seen[0].depth > seen[0].budget > 0
    assert all(c.sample == s for c in body_contacts(moved, clip, frozenset({s})))
    with pytest.raises(Exception, match="no rest file|no body shells"):
        body_contacts(dataclasses.replace(moved, rests={}), clip)


def test_finding_lines_read_as_the_gate_prints_them():
    f = Finding(NAME, "p:q pose C rig:a/b x rig:c/d", "depth 0.1000 over budget 0.0010")
    assert f.line() == "stress-poses | p:q pose C rig:a/b x rig:c/d | depth 0.1000 over budget 0.0010"


# --- the joint fold, excused --------------------------------------


def pose_index(ctx, name: str) -> int:
    return [p.name for p in stress_poses(rig_of(ctx)[1].rig)].index(name)


def test_a_fold_across_a_posed_joint_is_excused():
    """A bent joint's skin folds into itself at the crease. The elbow at the end of its range
    folds the arm's skin: that is the joint, not a fault, so no self-crossing is a finding."""
    ctx = scene("humanoid")
    found = StressPoses().run(ctx, samples=frozenset({pose_index(ctx, "elbow.L x lo")}))
    assert [f.item for f in found if " self " in f.item] == []


def test_the_two_sides_of_a_posed_joint_are_its_child_side_and_the_rest_of_its_chain():
    ctx = scene("humanoid")
    key, _ = rig_of(ctx)
    rig, _ = checker("humanoid")._prepare(ctx, key)
    s = pose_index(ctx, "elbow.L x lo")
    [(child, parent)] = sp._sides(rig.fixed, s, "arm_L")
    assert child.any() and parent.any() and not (child & parent).any()
    # the elbow bone carries the forearm and the hand; the shoulder bone carries the upper arm
    bones = rig.fixed.parts["arm_L"].bones
    main = rig.fixed.parts["arm_L"].bone
    assert {bones[b] for b in main[child]} <= {"elbow.L", "wrist.L"}
    assert {bones[b] for b in main[parent]} == {"shoulder.L"}
    one, other = np.nonzero(child)[0][:2], np.nonzero(parent)[0][:2]
    assert sp._folded([(child, parent)], one, other).all()
    assert sp._folded([(child, parent)], other, one).all()
    assert not sp._folded([(child, parent)], one, one[::-1]).any()
    assert not sp._folded([(child, parent)], other, other[::-1]).any()


def test_a_wrist_fold_reaches_the_rest_of_the_chain_above_it():
    ctx = scene("humanoid")
    key, _ = rig_of(ctx)
    rig, _ = checker("humanoid")._prepare(ctx, key)
    [(child, parent)] = sp._sides(rig.fixed, pose_index(ctx, "wrist.L x lo"), "arm_L")
    bones, main = rig.fixed.parts["arm_L"].bones, rig.fixed.parts["arm_L"].bone
    assert {bones[b] for b in main[child]} == {"wrist.L"}
    assert {bones[b] for b in main[parent]} == {"shoulder.L", "elbow.L"}


def test_in_a_pair_pose_a_crossing_across_either_joint_is_excused():
    ctx = scene("humanoid")
    key, _ = rig_of(ctx)
    rig, _ = checker("humanoid")._prepare(ctx, key)
    name = next(p.name for p in rig.fixed.poses if p.name.startswith("elbow.L+shoulder.L "))
    sides = sp._sides(rig.fixed, pose_index(ctx, name), "arm_L")
    assert len(sides) == 2  # one (child side, parent side) per posed joint, child joint first
    # faces 0, 1 | 2, 3 | 4, 5 are the two sides of the first joint and the two of the second
    on = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=bool)
    both = [(on[0], on[1]), (on[1], on[2])]
    i, j = np.array([0, 2, 0, 1, 4]), np.array([2, 4, 4, 0, 0])
    assert sp._folded(both, i, j).tolist() == [True, True, False, False, False]
    assert sp._folded(both, j, i).tolist() == [True, True, False, False, False]


def test_a_crossing_between_faces_of_one_side_stays_a_finding():
    """The self plant pushes a vertex through its limb's far wall: no crease lies between the
    faces that meet, so the excuse leaves it a finding."""
    check = checker("humanoid")
    plant = next(p for p in chosen(check, scene("humanoid")) if kind_of(p.item) == "self")
    assert caught(check, plant)


# --- a rig with several roots ----------------------------------------------------------------


def tree_of(spec, bone: str) -> str:
    parent = {b.name: b.parent for b in spec.bones}
    while parent[bone] is not None:
        bone = parent[bone]
    return bone


def seam_faces(ctx, part: str) -> set[int]:
    """The faces of a part whose corners are weighted by bones of two different trees, worked out
    here from the rest file and the spec."""
    key, piece = rig_of(ctx)
    rest = ctx.rests[key]
    names = rest.vertex_groups[part]
    trees = [{tree_of(piece.rig, names[g]) for g, w in zip(gs, ws, strict=True) if w > 0}
             for gs, ws in zip(rest.weight_groups[part], rest.weights[part], strict=True)]
    return {f for f, face in enumerate(rest.faces[part].astype(int))
            if len(set().union(*(trees[v] for v in face))) > 1}


def faces_named(finding) -> set[int]:
    return {int(n) for n in re.findall(r"\bfaces? (\d+)(?: and (\d+))?", finding.detail)[0] if n}


def test_a_rig_with_two_roots_skips_faces_weighted_across_both_trees():
    """The second tree stays at rest in a stress pose, so a face weighted across both trees has
    no pose: it tears. The check leaves such faces out and says how many."""
    ctx = scene("worm")
    key, _ = rig_of(ctx)
    seam = seam_faces(ctx, "body")
    assert seam
    lines = [x for x in StressPoses().coverage(ctx) if "bone trees" in x]
    want = (f"{NAME}: not applicable ({len(seam)} faces are weighted across two bone trees, "
            f"which no stress pose defines; they are skipped): {key}")
    assert lines == [want]
    assert not isinstance(lines[0], Unmeasured)
    found = [f for f in checker("worm").run(ctx)
             if " self " in f.item or " inverted " in f.item]
    assert not [f for f in found if faces_named(f) & seam], [f.line() for f in found]


def test_a_rig_with_one_root_prints_no_bone_trees_line():
    for body in ("humanoid", "quadruped"):
        assert not [x for x in checker(body).coverage(scene(body)) if "bone trees" in x]


def test_a_real_overlap_at_the_seam_of_a_two_root_rig_is_still_reported():
    """A vertex of a shell bound to one tree, near the seam, pushed through the far wall: the
    skip of the seam's own faces must not hide it."""
    ctx = scene("worm")
    key, _ = rig_of(ctx)
    check = StressPoses()
    rig, _ = check._prepare(ctx, key)
    fixed, stressed = rig.fixed, rig.scene
    part = fixed.parts["body"]
    seam = seam_faces(ctx, "body")
    used_by_seam = {int(v) for f in seam for v in part.faces[f]}
    data = stressed.samples[STRESS_CLIP]
    near = [v for v in np.unique(part.faces)
            if int(v) not in used_by_seam and abs(data.shells["body"][0][v][2]) < 0.15]
    assert near
    hit = None
    for s in range(0, len(fixed.poses), 7):
        item = f"{key} {fixed.poses[s].name} self body"
        if check.run(stressed, only={item}, samples=frozenset({s})):
            continue  # not quiet without the fault
        points = data.shells["body"][s].astype(float)
        deep = deep_point(SurfaceIndex(points[part.faces]),
                          CALIBRATION_FAIL * fixed.skin)
        for v in near[::5]:
            moved = sp._with_vertex(stressed, "body", s, int(v),
                                    points[v] + 3.0 * (deep - points[v]))  # a tube's radius past
            if check.run(moved, only={item}, samples=frozenset({s})):
                hit = (item, int(v))
                break
        if hit:
            break
    assert hit, "no vertex near the seam made a self-crossing the check reports"


# --- an unmeasurable body test ---------------------------------------------------------------


def test_an_unmeasurable_body_test_is_unmeasured_not_a_crash(monkeypatch):
    """`overlap.body_contacts` raises `Unmeasurable` for a clip it cannot read. That is a
    `ManifestError`, and one escaping `run` makes the gate exit "cannot start"."""
    def broken(*args, **kwargs):
        raise Unmeasurable("no body shells: test")

    monkeypatch.setattr(sp, "body_contacts", broken)
    monkeypatch.setattr(sp, "body_unreadable", lambda *a: "no body shells: test")
    ctx = scene("humanoid")
    check = StressPoses()
    found = check.run(ctx)
    assert found and not [f for f in found if " C " in f.item]  # self and inverted still run
    lines = [x for x in check.coverage(ctx) if isinstance(x, Unmeasured)]
    assert len(lines) == 1 and "no body shells: test" in lines[0] and lines[0].endswith(" C")
    assert not [u for u in check.plants(ctx) if u.item.endswith(" C")]
    quad = scene("quadruped")  # its findings are all C, so the run is clean and only partial
    assert run_gate([StressPoses()], quad, None).status == "partial"


# --- registration ----------------------------------------------------------------------------


def test_stress_poses_is_registered_and_coverage_stays_last():
    names = list(gate.CHECKS)
    assert "stress-poses" in names
    assert names[-1] == "coverage"
    assert gate.CHECKS["stress-poses"] is StressPoses


# --- plants as units ---------------------------------------------------------


@pytest.mark.parametrize("body", ["humanoid", "quadruped"])
def test_the_units_are_tried_and_caught_on_two_bodies(body):
    """Every kind has a unit the gate tries at some pose, and its first catching candidate."""
    tried = tried_and_caught(checker(body), scene(body))
    assert {kind_of(i) for i in tried} == {"C", "self", "inverted"}


def test_a_unit_is_one_item_at_one_pose_and_a_pose_the_real_run_fails_is_skipped():
    ctx = scene("humanoid")
    check = checker("humanoid")
    failing = {f.item for f in check.run(ctx)}
    units = [u for u in check.plants(ctx) if not u.not_applicable]
    assert all(len({c.item for c in u.candidates}) == 1 and u.candidates for u in units)
    assert all(len(c.samples) == 1 for u in units for c in u.candidates)
    # the humanoid's range ends hold real findings; a unit at such a pose is built all the same
    assert any(u.item in failing for u in units)


# --- the self rule's blind spot on narrow faces (an expected failure) -----------------------------


def narrow_faced_fold():
    """A fine tube (faces about a third of the skin wide) with one ring vertex pushed 3 skins
    past the far wall: a poke 3 skins deep through a limb, on a mesh whose faces are narrow."""
    skin = 0.01
    verts, faces = sweep([(y, 0.0, 0.0, 0.05, 0.05) for y in (0.0, 0.1, 0.2)], "y", segments=32)
    middle = np.abs(verts[:, 1] - 0.1) < 0.02  # a ring about half way along
    ring = int(np.flatnonzero(middle)[np.argmax(verts[middle, 0])])  # its +x vertex
    folded = verts.copy()
    folded[ring, 0] = -(0.05 + 3 * skin)
    return skin, verts[faces], folded[faces]


def test_the_narrow_faced_fold_is_a_real_crossing_of_faces_narrower_than_the_skin():
    skin, whole, folded = narrow_faced_fold()
    assert sp._face_reach(whole).max() < skin  # every face is narrower than the skin
    assert len(self_cuts(whole)[0]) == 0  # the tube alone does not cross itself
    assert len(self_cuts(folded)[0]) > 0  # the poke does, judged with no skin at all


@pytest.mark.xfail(strict=True, reason=(
    "geom.tri_tri_cut measures a cut's depth to the edges of both faces, so a cut is no deeper "
    "than its faces are wide; the self rule keeps only cuts deeper than the skin, so a fold "
    "through faces narrower than the skin is never found (as on a chibi's arm_L/R, shin_L/R, thigh_L/R). "
    "A fix to the rule's depth measure turns this to a pass; strict, so the mark must then go."))
def test_a_deep_fold_through_faces_narrower_than_the_skin_is_found():
    skin, _, folded = narrow_faced_fold()
    assert len(self_cuts(folded, skin)[0]) > 0
