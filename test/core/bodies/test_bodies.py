"""The three test bodies: a tall humanoid, a quadruped with a tail and a worm with no sides.

Every body check after this one plants its faults on these bodies, so the tests here hold what
those checks lean on: each body is closed, skinned and saved in the scene format; its supports
touch the floor in its standing clips; its joints each carry a range."""

import dataclasses
import json

import numpy as np
import pytest

from model_kit.core.checks.clip_hygiene import ClipHygiene
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.topology import Topology, problems
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import load_pack
from model_kit.core.pose import skin
from model_kit.core.rig import bending_joints, pose_world, rig_root
from model_kit.core.scene import load_scene, rig_spec, save_scene
from model_kit.core.targets import seat_root
from model_kit.core.writer import pack_json

from .humanoid import humanoid_scene
from .quadruped import quadruped_scene
from .worm import worm_scene

SCENES = {"humanoid": humanoid_scene, "quadruped": quadruped_scene, "worm": worm_scene}
BODY = {"humanoid": "humanoid:Person", "quadruped": "quadruped:Animal", "worm": "worm:Worm"}
# Float noise on a point that sits exactly on the floor (float32 samples, a few matrix products).
FLOOR_NOISE = 1e-6


@pytest.fixture(params=sorted(SCENES))
def scene(request):
    return request.param, SCENES[request.param]()


def _spec(name, ctx):
    return ctx.packs[name].pieces[BODY[name].split(":")[1]].rig


def test_each_body_round_trips(scene, tmp_path):
    name, ctx = scene
    save_scene(ctx, tmp_path / name)
    back = load_scene(tmp_path / name)
    key = BODY[name]
    assert back.packs == ctx.packs
    assert back.clips == ctx.clips
    assert back.layouts == ctx.layouts
    assert set(back.pieces) == set(ctx.pieces)
    for k, geom in ctx.pieces.items():
        for part, tris in geom.parts.items():
            np.testing.assert_array_equal(back.pieces[k].parts[part], tris)
    assert back.rigs[key].order == ctx.rigs[key].order
    assert back.rigs[key].parent == ctx.rigs[key].parent
    for bone, matrix in ctx.rigs[key].rest.items():
        np.testing.assert_allclose(back.rigs[key].rest[bone], matrix, atol=1e-12)
    rest, rest_back = ctx.rests[key], back.rests[key]
    assert rest_back.vertex_groups == rest.vertex_groups
    for part in rest.shells:
        for field in ("shells", "faces", "labels", "weights", "weight_groups"):
            np.testing.assert_array_equal(getattr(rest_back, field)[part],
                                          getattr(rest, field)[part])
    for clip, samples in ctx.samples.items():
        for part, shell in samples.shells.items():
            np.testing.assert_array_equal(back.samples[clip].shells[part], shell)
        for group, points in samples.groups.items():
            np.testing.assert_array_equal(back.samples[clip].groups[group], points)


def test_humanoid_bending_joints_include_hips_and_knees():
    spec = _spec("humanoid", humanoid_scene())
    joints = set(bending_joints(spec))
    assert {"hip.L", "hip.R", "knee.L", "knee.R"} <= joints
    # A bone with no range table row is no joint: the hip turns on x and z only, the knee on x.
    assert set(spec.joints["hip.L"].range_deg) == {"x", "z"}
    assert set(spec.joints["knee.L"].range_deg) == {"x"}
    assert set(spec.sit.past_edge) == {"knee.L", "knee.R"}


def test_every_bending_joint_has_a_range_and_a_why(scene):
    name, ctx = scene
    spec = _spec(name, ctx)
    assert set(spec.joints) == set(bending_joints(spec))
    for bone, joint in spec.joints.items():
        assert joint.range_deg and joint.why.strip(), bone


def test_worm_needs_its_root(tmp_path):
    ctx = worm_scene()
    spec = _spec("worm", ctx)
    assert [b.name for b in spec.bones if b.parent is None] == ["head_end", "tail_end"]
    assert rig_root(spec) == "head_end"
    with pytest.raises(ManifestError, match="head_end.*tail_end"):
        rig_root(dataclasses.replace(spec, root=None))
    raw = pack_json(ctx.packs["worm"])
    del raw["pieces"]["Worm"]["rig"]["root"]
    path = tmp_path / "worm.pack.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ManifestError, match="root"):
        load_pack(path, ctx.style)
    assert spec.mirror is None and spec.sit is None and spec.supports == ("belly",)


def test_quadruped_tail_is_not_a_support():
    spec = _spec("quadruped", quadruped_scene())
    tail = [f"tail_{k}" for k in range(1, 7)]
    assert {b.name for b in spec.bones if b.name.startswith("tail_")} == set(tail)
    assert len(spec.supports) == 4
    held = {bone for group in spec.supports for bone in spec.contact_groups[group]}
    assert held and not held & set(tail)
    # The tail is a six-bone chain of its own, and its tip group is no support.
    chain = spec.chains["tail"]
    assert chain.bones == tuple(tail) and chain.root == "pelvis" and chain.effector == "tail_6"
    assert chain.effector_group == "tail_tip" and chain.effector_group not in spec.supports
    assert spec.contact_groups["tail_tip"] == ("tail_6",)
    assert {b: set(d) for b, d in chain.dofs.items()} == {b: set(spec.joints[b].range_deg)
                                                           for b in tail}
    legs = [c.effector_group for n, c in spec.chains.items() if n != "tail"]
    assert sorted(legs) == sorted(spec.supports) and len(spec.chains) == 5
    assert spec.sit is None and spec.mirror is not None


def test_every_part_is_closed(scene):
    _name, ctx = scene
    assert Topology().run(ctx) == []
    # Only the panel's decal is open by design, and it is the only part topology leaves out.
    skipped = [line for line in Topology().coverage(ctx) if "not applicable" in line]
    open_parts = [(k, p) for k, geom in ctx.pieces.items() for p in geom.parts
                  if problems(geom.parts[p], 1.0)]
    assert [p for _, p in open_parts] == ["mark"]
    assert len(skipped) == len(open_parts)


def test_standing_clips_keep_every_support_on_the_floor(scene):
    name, ctx = scene
    spec = _spec(name, ctx)
    for clip in ctx.clips.values():
        if clip.posture != "stand":
            continue
        samples = ctx.samples[clip.name]
        for group in spec.supports:
            lowest = samples.groups[group][:, :, 1].min(axis=1)
            lifted = np.zeros(len(lowest), bool)
            for lift in clip.lifts:
                if lift.support == group:
                    for t0, t1 in lift.when:
                        lifted |= (samples.times >= t0) & (samples.times <= t1)
            assert np.all(np.abs(lowest[~lifted]) < FLOOR_NOISE), (clip.name, group)
            if lifted.any():
                assert lowest[lifted].max() > 0.01, (clip.name, group)
        for part, shell in samples.shells.items():
            assert shell[:, :, 1].min() > -FLOOR_NOISE, (clip.name, part)


def test_the_step_lifts_one_foot_inside_its_window():
    ctx = humanoid_scene()
    clip = ctx.clips["step"]
    assert [(lift.support, lift.why.strip() != "") for lift in clip.lifts] == [("foot.L", True)]
    low = {g: ctx.samples["step"].groups[g][:, :, 1].min(axis=1) for g in ("foot.L", "foot.R")}
    assert low["foot.L"].max() > 0.02
    np.testing.assert_allclose(low["foot.R"], 0.0, atol=FLOOR_NOISE)


@pytest.mark.parametrize("clip", ["sit_chair", "sit_bench"])
def test_a_sitter_is_at_the_seat_with_the_knees_past_the_front_edge(clip):
    ctx = humanoid_scene()
    spec = rig_spec(ctx, clip)
    samples = ctx.samples[clip]
    seat_ref = ctx.clips[clip].seat
    placed = ctx.clips[clip].floor_gear.pieces[seat_ref.piece]
    piece = ctx.packs[placed.pack].pieces[placed.piece]
    seat = next(s for s in piece.seats if s.id == seat_ref.id)
    root = seat_root(placed.to_world(seat.point), seat.facing, spec.sit)
    for s in range(len(samples.times)):
        np.testing.assert_allclose(samples.bones["pelvis"][s][:3, 3], root, atol=1e-9)
    tris = np.concatenate([ctx.pieces[f"{placed.pack}:{placed.piece}"].parts[p]
                           for p in seat.surface]).reshape(-1, 3)
    front = tris[:, 2].max()  # the seat faces +z
    for bone in spec.sit.past_edge:
        assert samples.bones[bone][:, 2, 3].min() > front


def test_the_humanoid_is_tall():
    shells = humanoid_scene().rests["humanoid:Person"].shells
    assert max(s[0, :, 1].max() for s in shells.values()) == pytest.approx(1.75, abs=0.01)


def test_the_unjoined_piece_stands_one_body_size_away(scene):
    """One body size is the longest side of the body's bounding box (a person's height)."""
    name, ctx = scene
    shells = np.concatenate([s[0] for s in ctx.rests[BODY[name]].shells.values()])
    size = float((shells.max(axis=0) - shells.min(axis=0)).max())
    layout = ctx.layouts[name]
    piece = BODY[name].split(":")[1]
    body = np.array(next(p.position for p in layout.placements.values() if p.piece == piece))
    far = [p for p in layout.placements.values() if p.piece == "Panel"]
    assert len(far) == 1 and far[0].on is None and far[0].rests_on is None
    assert np.linalg.norm(np.array(far[0].position) - body) == pytest.approx(size, abs=1e-6)
    if name == "humanoid":
        assert size == pytest.approx(1.75, abs=0.01)


def test_a_scene_is_fresh_each_call(scene):
    name, ctx = scene
    other = SCENES[name]()
    part = next(iter(ctx.rests[BODY[name]].shells))
    ctx.rests[BODY[name]].shells[part][:] = 99.0
    ctx.packs.clear()
    assert other.packs and other.rests[BODY[name]].shells[part].max() < 5.0


def test_the_rest_pose_gives_the_rest_shell_back(scene):
    name, ctx = scene
    rig, rest = ctx.rigs[BODY[name]], ctx.rests[BODY[name]]
    posed = skin(rest, rig, pose_world(rig, {}))
    for part, shell in rest.shells.items():
        np.testing.assert_allclose(posed[part], shell[0], atol=1e-6)


def _tail(ctx, pose, bone):
    """Where a bone's tail lands under `pose` (bone -> (euler degrees, no move))."""
    rig = ctx.rigs[BODY["humanoid"]]
    length = next(b.length for b in ctx.packs["humanoid"].pieces["Person"].rig.bones
                  if b.name == bone)
    world = pose_world(rig, {b: (np.radians(e), np.zeros(3)) for b, e in pose.items()})
    return world[bone][:3, 3] + world[bone][:3, 1] * length


def test_the_signs_in_the_humanoid_docstring_hold():
    ctx = humanoid_scene()
    rest = {b: _tail(ctx, {}, b) for b in ("knee.L", "hip.L", "elbow.L", "wrist.L", "spine",
                                            "ankle.L", "shoulder.L", "shoulder.R")}
    # A positive x flexes the knee and trunk (the tail goes back, the top goes forward), extends
    # the hip, and flexes the ankle down; the elbow and wrist flex on a negative x.
    assert _tail(ctx, {"knee.L": (30, 0, 0)}, "knee.L")[2] < rest["knee.L"][2]
    assert _tail(ctx, {"hip.L": (30, 0, 0)}, "hip.L")[2] < rest["hip.L"][2]
    assert _tail(ctx, {"spine": (30, 0, 0)}, "spine")[2] > rest["spine"][2]
    assert _tail(ctx, {"elbow.L": (-30, 0, 0)}, "elbow.L")[2] > rest["elbow.L"][2]
    assert _tail(ctx, {"wrist.L": (-30, 0, 0)}, "wrist.L")[2] > rest["wrist.L"][2]
    assert _tail(ctx, {"ankle.L": (30, 0, 0)}, "ankle.L")[1] < rest["ankle.L"][1]
    # A positive z moves a left limb's tip toward the midline and a right limb's away from it.
    left = _tail(ctx, {"shoulder.L": (0, 0, 30)}, "shoulder.L")[0] < rest["shoulder.L"][0]
    right = _tail(ctx, {"shoulder.R": (0, 0, 30)}, "shoulder.R")[0] < rest["shoulder.R"][0]
    assert left and right  # both tips move toward -x: toward the midline on the left only
    assert rest["shoulder.R"][0] < 0 < rest["shoulder.L"][0]
    joints = _spec("humanoid", ctx).joints
    assert joints["shoulder.L"].range_deg["z"] == (-180, 40)
    assert tuple(joints["shoulder.R"].range_deg["z"]) == (-40, 180)


def _turns(ctx, name, clip):
    """Per bone, the (S, 3) euler degrees (x, y, z; R = Rz Ry Rx) that each saved sample turns it
    by, read back from the bone matrices as `pose_world` composes them (world is the identity)."""
    rig = ctx.rigs[BODY[name]]
    world = ctx.samples[clip].bones
    out = {}
    for bone in rig.order:
        parent = rig.parent[bone]
        inv = np.linalg.inv(rig.rest[bone])
        basis = (np.einsum("ij,sjk->sik", inv, world[bone]) if parent is None else
                 np.einsum("ij,jk,skl,slm->sim", inv, rig.rest[parent],
                           np.linalg.inv(world[parent]), world[bone]))
        r = basis[:, :3, :3]
        ry = -np.arcsin(np.clip(r[:, 2, 0], -1.0, 1.0))
        rx = np.arctan2(r[:, 2, 1], r[:, 2, 2])
        rz = np.arctan2(r[:, 1, 0], r[:, 0, 0])
        out[bone] = np.degrees(np.stack([rx, ry, rz], axis=1))
    return out


def test_every_clip_stays_inside_its_cited_joint_ranges(scene):
    name, ctx = scene
    spec = _spec(name, ctx)
    slack = 1e-6  # degrees: the decomposition of float64 matrices
    for clip in ctx.clips:
        for bone, angles in _turns(ctx, name, clip).items():
            if bone not in spec.joints:
                continue
            for i, axis in enumerate("xyz"):
                lo, hi = spec.joints[bone].range_deg.get(axis, (0.0, 0.0))
                assert angles[:, i].min() >= lo - slack, (clip, bone, axis, angles[:, i].min())
                assert angles[:, i].max() <= hi + slack, (clip, bone, axis, angles[:, i].max())


# The joins a body's pieces are built with, which `crossings` reads as embedded parts (a part sunk
# more than half the thinner part's thickness into another). Each is a designed join, so each
# would be an allowlist line with a reason in a pack.
DESIGNED_JOINS = {
    "Person": {"leg_L x torso": "the leg's upper end is seated inside the torso",
               "leg_R x torso": "the leg's upper end is seated inside the torso"},
    "Animal": {"body x tail": "the tail's root end is seated inside the body",
               "body x leg_hind_L": "the leg's upper end is seated inside the body",
               "body x leg_hind_R": "the leg's upper end is seated inside the body",
               "body x leg_fore_L": "the leg's upper end is seated inside the body",
               "body x leg_fore_R": "the leg's upper end is seated inside the body"},
    "Panel": {"board x bolt_1": "the bolt's end is driven into the board it fastens",
              "board x bolt_2": "the bolt's end is driven into the board it fastens",
              "bolt_1 x plate": "the bolt is driven through the plate it fastens",
              "bolt_2 x plate": "the bolt is driven through the plate it fastens"},
}


def test_the_piece_and_clip_checks_find_nothing_on_a_body(scene):
    """A known-good body: the checks that already run on it say nothing (the gate's `partial`
    comes only from coverage lines). `crossings` finds exactly the joins it was built with."""
    name, ctx = scene
    for check in (Floaters(), ClipHygiene()):
        assert check.run(ctx) == [], check.name
    joins = {f"{name}:{piece} {pair}" for piece, pairs in DESIGNED_JOINS.items()
             for pair in pairs if f"{name}:{piece}" in ctx.pieces}
    assert {f.item for f in Crossings().run(ctx)} == joins


def _canonical(tris) -> set:
    """Triangles as a set, each with its corners turned to start at the smallest (the winding
    stays)."""
    out = set()
    for tri in np.round(np.asarray(tris, dtype=np.float64), 9):
        corners = [tuple(c) for c in tri]
        k = corners.index(min(corners))
        out.add(tuple(corners[k:] + corners[:k]))
    return out


def test_mirrored_parts_are_exact_mirror_images(scene):
    """Each part bound to a mirrored bone set matches its pair triangle for triangle: x negated,
    winding reversed so it still faces outward."""
    name, ctx = scene
    piece = ctx.packs[name].pieces[BODY[name].split(":")[1]]
    mirror = piece.rig.mirror
    if mirror is None:
        assert name == "worm"
        return
    swap = {a: b for a, b in mirror.pairs} | {b: a for a, b in mirror.pairs}
    geom = ctx.pieces[BODY[name]].parts
    by_bones = {frozenset(p.bones): p.name for p in piece.parts}
    paired = 0
    for part in piece.parts:
        twin = by_bones.get(frozenset(swap.get(b, b) for b in part.bones))
        if twin is None or twin == part.name or not any(b in swap for b in part.bones):
            continue
        paired += 1
        flipped = geom[part.name][:, ::-1] * np.array([-1.0, 1.0, 1.0])
        assert _canonical(flipped) == _canonical(geom[twin]), (part.name, twin)
        assert len(geom[twin]) == len(geom[part.name])
    assert paired == 4  # the humanoid's legs and arms, the quadruped's four legs


def test_the_worm_keeps_its_seam_closed():
    """The two segments that meet in the middle keep coincident ends in every sample."""
    ctx = worm_scene()
    spec = _spec("worm", ctx)
    length = {b.name: b.length for b in spec.bones}
    bones = ctx.samples["stand"].bones
    ends = {b: bones[b][:, :3, 3] + bones[b][:, :3, 1] * length[b] for b in ("seg_4", "seg_5")}
    np.testing.assert_allclose(ends["seg_4"], ends["seg_5"], atol=1e-9)
    # The clip does move the worm: the head tree turns, so the seam leaves the line.
    assert np.abs(ends["seg_4"][:, 0]).max() > 0.005


def test_the_panel_bolts_run_through_the_board():
    """A bolt is 12 mm square across its shank and longer along z, through the board."""
    tris = humanoid_scene().pieces["humanoid:Panel"].parts
    for name in ("bolt_1", "bolt_2"):
        pts = tris[name].reshape(-1, 3)
        size = pts.max(axis=0) - pts.min(axis=0)
        np.testing.assert_allclose(size[:2], 0.012, atol=1e-9)
        assert size[2] > 0.025
