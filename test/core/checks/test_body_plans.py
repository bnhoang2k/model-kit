"""The body-plan audit: every body check runs on every body, and none assumes a humanoid.

Bilateral symmetry, two legs and one root are common but not universal. So each body check runs
on a tall humanoid, a quadruped with a tail, a worm with no sides and a chibi, and
for each pair one of three things must hold:

- **plants caught.** The check plants a fault on that body and, run the way `metagate.run_gate`
  runs a plant, finds it.
- **by design.** The check has nothing to plant and says why in a `not applicable` coverage line
  (a plain status line such as `measured 4 items` does not count). The reason is never a humanoid
  assumption: no "top bone", "L and R", "left and right", "two feet", "two hands", and none of the
  whole words "feet", "legs", "left" or "right", in any case.
- **unmeasured, the chibi only.** The check names missing data in an `Unmeasured` line. The
  chibi gets its rig data from `bodies.bar03`, which injects `supports` and `sit.past_edge` in
  memory because its pack does not hold them. The test bodies carry full data, so an
  `Unmeasured` line from a body check on one of them fails the audit.

Anything else fails: a crash, a silent pass (no plant and no line), a check that measures an item
and cannot plant it (`skipped (no plant can fail)`), a plant the check misses, a finding on a
body that is known good. The one exception is stress-poses on a test body (see `audit`): a
finding there is a real coupled corner, and the test bodies only prove that its plants are caught.
So is bend-quality on a test body (see `audit`): linear blend skinning loses girth at a bent
joint, so a finding there is real.

**The filter rule.** The audit takes its checks from `gate.CHECKS`, so a check that registers
later enters it with no edit here. A body check is every registered check not named in `NOT_BODY`
below. The body checks are must-touch, reach, solver
error, trace, sides, mirror-pair, overlap, clip hygiene and the newer body checks, and
`NOT_BODY` names the other checks that register (piece checks of one recipe, and pack, scene and
file checks) with the reason each is left out. The default is in: a check is audited until someone excludes it on purpose.
That includes a check that declares `kinds = {"assembly"}`, because a rigged piece is an assembly
piece too.

**What the audit cannot see.** A check that yields plant units (`PlantUnit`) is judged the way
`metagate.run_gate` judges it: a unit whose item the real run already reports is skipped, a unit
that says it does not apply is a `not applicable` line, and any other unit must hold a candidate
the check catches (a unit with none is a blind check, which stops the run). A check still on plain
plants learns that it measures an item and cannot plant it only from its own `skipped (no plant
can fail)` line, and must print that line for every item it measures and does not plant.
"""

import dataclasses
import functools
import math
import re
from dataclasses import dataclass

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks.mirror_pair import MirrorPair
from model_kit.core.checks.overlap import Overlap
from model_kit.core.checks.sides import Sides
from model_kit.core.manifest.rig import LIMIT_SLACK_RAD, bending_joints
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.pose.stress import missing_ranges
from model_kit.core.rig import pose_world
from model_kit.core.scene import load_scene

from ..bodies.bar03 import SCENE as BAR03_SCENE
from ..bodies.bar03 import bar03_scene, with_proxy_weights
from ..bodies.humanoid import humanoid_scene
from ..bodies.quadruped import quadruped_scene
from ..bodies.worm import worm_scene

BODIES = {"humanoid": humanoid_scene, "quadruped": quadruped_scene, "worm": worm_scene,
          "bar03": bar03_scene}
TEST_BODIES = ("humanoid", "quadruped", "worm")

# The checks that read no body, each with its reason: a piece check of one recipe, or a pack, a
# scene or a file. Everything else that registers is audited.
NOT_LISTED = "it is not a body check (a design call of this kit)"
ASSEMBLY = f"{NOT_LISTED}: it measures the parts of assembly pieces"
BRANCHING = f"{NOT_LISTED}: it measures a grown tree"
NOT_BODY = {
    "crossings": ASSEMBLY,
    "take-apart": ASSEMBLY,
    "floaters": ASSEMBLY,
    "topology": ASSEMBLY,
    "near-crease": f"{NOT_LISTED}: it measures the edges of a piece's own parts, rigged or not",
    "inset-seat": f"{NOT_LISTED}: it measures printed parts against the host they sit on",
    "detail": f"{NOT_LISTED}: it judges the curves of a piece's own parts at its viewing distance",
    "markings-flush": ASSEMBLY,
    "frames": ASSEMBLY,
    "done": ASSEMBLY,
    "keep-off": ASSEMBLY,
    "markings": ASSEMBLY,
    "bearing": ASSEMBLY,
    "limb-seat": BRANCHING,
    "floor-organic": BRANCHING,
    "crossing-organic": BRANCHING,
    "near-miss-organic": BRANCHING,
    "skeleton-match": BRANCHING,
    "no-fallback": f"{NOT_LISTED}: it reads the build scripts and the lookup",
    "receiver": f"{NOT_LISTED}: it reads mount ports on pieces",
    "connectors": f"{NOT_LISTED}: it reads the pack's connector table",
    "export-hygiene": f"{NOT_LISTED}: it reads the packed file",
    "clearance": f"{NOT_LISTED}: it measures the gaps between the pieces of a layout",
    "coverage": f"{NOT_LISTED}: it is built from the other checks of a run and judges parts",
}

# A reason that names a humanoid is a fault: it says the check guessed a human body. Phrases and
# whole words, in any case.
HUMANOID_WORDS = ("top bone", "L and R", "left and right", "two feet", "two hands", "feet", "legs",
                  "left", "right")
HUMANOID = re.compile("|".join(rf"\b{re.escape(w)}\b" for w in HUMANOID_WORDS), re.IGNORECASE)

# What an Unmeasured line on bar 03 must name: data that is missing, not a plant that cannot fail.
MISSING_DATA = re.compile(r"\b(no|missing|lacks|without)\b", re.IGNORECASE)
NO_PLANT = "no plant can fail"
STRESS_POSES = "stress-poses"  # may find real coupled corners on a test body: see `audit`

BEND_QUALITY = "bend-quality"  # may find real pinching on a test body: see `audit`


def is_body_check(name: str) -> bool:
    """The filter rule above: every registered check the table does not exclude."""
    return name not in NOT_BODY


def body_checks() -> list[str]:
    return [name for name in gate.CHECKS if is_body_check(name)]


@functools.cache
def scene(body: str):
    """A body's scene, built once for the module. The checks never write to it."""
    return BODIES[body]()


@dataclass
class Outcome:
    kind: str  # "plants caught", "by design" or "unmeasured"
    detail: str


def audit(check, ctx, *, body: str) -> Outcome:
    """Judge one check on one body; raise AssertionError for a fault."""
    name = check.name
    real: list[Finding] = check.run(ctx)
    failing = {f.item for f in real}
    lines = list(check.coverage(ctx))
    plants: list[PlantUnit] = []
    for unit in check.plants(ctx):  # the way metagate.run_gate runs a unit
        if unit.not_applicable:
            lines.append(f"{name}: not applicable ({unit.not_applicable}): {unit.item}")
            continue
        if unit.item in failing:
            continue  # the gate skips a unit whose item already has a finding
        plants.append(unit)  # "plants caught" counts the units the gate tries
        caught = [c for c in unit.candidates if any(
            f.item == c.item for f in check.run(c.ctx, only=c.scope, samples=c.samples))]
        assert caught, f"{name} on {body}: no candidate of {unit.item} is caught"
    # Stress-poses: its poses go to the ends of the cited ranges,
    # where two neighbouring joints at once leave real coupled corners, so a test body may hold
    # findings of it. The test bodies only prove that each of its plants is caught. The known-good
    # set takes allow lines with a coupling citation instead.
    # The same holds for bend-quality: linear blend skinning loses girth
    # at a single-joint hinge bent past about 60 degrees, so the test bodies only prove that its
    # plants are caught. The known-good set (the bars) must be quiet.
    if body in TEST_BODIES and name not in {STRESS_POSES, BEND_QUALITY}:
        assert real == [], f"{name} finds a fault in {body}, a known-good body: {real}"
    for line in lines:
        said = HUMANOID.search(line)
        assert not said, f"{name} on {body} assumes a humanoid ({said.group()!r}): {line}"
    unmeasured = [x for x in lines if isinstance(x, Unmeasured)]
    blind = [x for x in unmeasured if NO_PLANT in x]
    assert not blind, f"{name} on {body} measures and has no plant: {blind[0]}"
    if body in TEST_BODIES:
        assert not unmeasured, f"{name} on {body}, a body with full data, is unmeasured: " \
                               f"{unmeasured[0]}"
    else:
        lost = [x for x in unmeasured if not MISSING_DATA.search(x)]
        assert not lost, f"{name} on {body}: an Unmeasured line that names no missing data: " \
                         f"{lost[0]}"
    if plants:
        return Outcome("plants caught", f"{len(plants)} plant{'s' if len(plants) > 1 else ''}")
    by_design = [x for x in lines if not isinstance(x, Unmeasured) and "not applicable" in x]
    if by_design:
        return Outcome("by design", by_design[0])
    assert body not in TEST_BODIES and unmeasured, \
        f"{name} on {body}: no plant and no line: a silent pass"
    return Outcome("unmeasured", unmeasured[0])


PAIRS = [pytest.param(name, body, id=f"{name}-{body}")
         for name in body_checks() for body in BODIES]


@pytest.mark.parametrize("name, body", PAIRS)
def test_every_body_check_runs_on_every_body(name, body):
    outcome = audit(gate.CHECKS[name](), scene(body), body=body)
    assert outcome.kind in ("plants caught", "by design", "unmeasured")


# --- the audit audits itself ---------------------------------------------------------------------


def test_the_audit_has_checks_and_bodies():
    names = body_checks()
    # the body checks, and the clip checks before them
    assert {"must-touch", "reach", "solver-error", "trace", "sides", "mirror-pair", "overlap",
            "clip-hygiene"} <= set(names)
    assert not set(names) & set(NOT_BODY)
    assert len(PAIRS) == len(names) * len(BODIES) and PAIRS
    assert set(BODIES) == {"humanoid", "quadruped", "worm", "bar03"}


def test_every_name_in_not_body_is_a_registered_check():
    assert set(NOT_BODY) <= set(gate.CHECKS)  # a renamed check must not slip out of the audit


def test_a_check_is_a_body_check_unless_it_is_excluded_by_name(monkeypatch):
    assert not is_body_check("floaters") and not is_body_check("limb-seat")
    assert is_body_check("clip-hygiene") and is_body_check("overlap")

    class LaterCheck:  # a later check that measures assembly pieces, rigged ones among them
        name = "later"
        kinds = frozenset({"assembly"})

    monkeypatch.setitem(gate.CHECKS, "later", LaterCheck)
    assert "later" in body_checks()  # in by default: a new check must be excluded on purpose
    assert {"must-touch", "reach", "trace", "sides", "mirror-pair", "overlap", "solver-error",
            "clip-hygiene"} <= set(body_checks())


class _Stub:
    """A check with chosen plants and lines, to feed the audit."""

    name = "stub"
    kinds: frozenset[str] = frozenset()

    def __init__(self, *, plants=(), lines=(), findings=()):
        self._plants, self._lines, self._findings = list(plants), list(lines), list(findings)

    def plants(self, ctx):
        return self._plants

    def run(self, ctx, only=None, samples=frozenset()):
        if only is not None:  # a planted copy: report what it plants
            return [f for f in self._findings if f.item in only]
        return []

    def coverage(self, ctx):
        return self._lines


def test_the_audit_rejects_a_silent_pass():
    with pytest.raises(AssertionError, match="silent pass"):
        audit(_Stub(), None, body="worm")


def test_the_audit_rejects_a_humanoid_reason():
    for reason in HUMANOID_WORDS:
        stub = _Stub(lines=[f"stub: not applicable ({reason}): clip"])
        with pytest.raises(AssertionError, match="assumes a humanoid"):
            audit(stub, None, body="quadruped")


def test_the_audit_rejects_a_plain_status_line_with_no_plants():
    """`measured 4 items` says what a check did, not why it has nothing to plant."""
    with pytest.raises(AssertionError, match="silent pass"):
        audit(_Stub(lines=["stub: measured 4 items"]), None, body="quadruped")
    with pytest.raises(AssertionError, match="silent pass"):
        audit(_Stub(lines=["stub: skipped (no holds): clip"]), None, body="humanoid")


def test_the_humanoid_words_match_whole_words_in_any_case():
    for text in ("on LEFT and Right", "the Legs", "two Feet", "Top Bone", "the right hand"):
        assert HUMANOID.search(text), text
    for text in ("foot.L", "copyright", "leftover", "playlegsX", "no loop"):
        assert not HUMANOID.search(text), text


def test_the_audit_accepts_a_by_design_line():
    out = audit(_Stub(lines=["stub: not applicable (no loop): clip"]), None, body="worm")
    assert out.kind == "by design"


def test_the_audit_rejects_a_plant_the_check_misses():
    unit = PlantUnit("clip item", (Plant("a fault", "clip item", None),))
    with pytest.raises(AssertionError, match="no candidate of clip item is caught"):
        audit(_Stub(plants=[unit]), None, body="humanoid")
    caught = _Stub(plants=[unit], findings=[Finding("stub", "clip item", "x")])
    assert audit(caught, None, body="humanoid").kind == "plants caught"


def test_the_audit_rejects_a_check_that_measures_and_cannot_plant():
    line = Unmeasured("stub: skipped (no plant can fail): clip item")
    with pytest.raises(AssertionError, match="no plant"):
        audit(_Stub(lines=[line]), None, body="bar03")


def test_unmeasured_is_allowed_on_bar_03_alone_and_only_for_missing_data():
    line = Unmeasured("stub: skipped (no joint ranges in the rig): clip")
    assert audit(_Stub(lines=[line]), None, body="bar03").kind == "unmeasured"
    with pytest.raises(AssertionError, match="full data"):
        audit(_Stub(lines=[line]), None, body="humanoid")
    with pytest.raises(AssertionError, match="names no missing data"):
        audit(_Stub(lines=[Unmeasured("stub: skipped (it was hard): clip")]), None,
              body="bar03")


def test_a_finding_on_a_test_body_fails_the_audit():
    class Finds(_Stub):
        def run(self, ctx, only=None, samples=frozenset()):
            return [Finding("stub", "clip item", "x")]

    stub = Finds(lines=["stub: not applicable (nothing to do)"])
    with pytest.raises(AssertionError, match="known-good"):
        audit(stub, None, body="worm")
    assert audit(stub, None, body="bar03").kind == "by design"  # bar 03's findings are its own


def test_only_stress_poses_and_bend_quality_may_find_on_a_test_body():
    """Stress-poses has real coupled corners at its range ends, and
    linear blend skinning pinches a bent joint, so both may find on a test body. Any other check
    that finds there fails the audit."""
    class Finds(_Stub):
        def run(self, ctx, only=None, samples=frozenset()):
            return [Finding(self.name, "clip item", "x")]

    for name in (STRESS_POSES, BEND_QUALITY):
        stub = Finds(lines=["stub: not applicable (nothing to do)"])
        stub.name = name
        assert audit(stub, None, body="worm").kind == "by design"
    stub = Finds(lines=["stub: not applicable (nothing to do)"])
    stub.name = "overlap"
    with pytest.raises(AssertionError, match="known-good"):
        audit(stub, None, body="worm")


def test_clip_hygiene_and_the_rig_checks_plant_something_on_every_body():
    """A loop over a check's units passes when the set is empty, so say it is not.
    Clip hygiene plants on all four bodies. Stress-poses and bend-quality plant on the test
    bodies, which carry the joint ranges and weights they need. Overlap needs two body shells,
    which the worm lacks."""
    wanted = {"clip-hygiene": tuple(BODIES), "stress-poses": TEST_BODIES,
              "bend-quality": TEST_BODIES, "overlap": ("humanoid", "quadruped")}
    for name, bodies in wanted.items():
        for body in bodies:
            out = audit(gate.CHECKS[name](), scene(body), body=body)
            assert out.kind == "plants caught" and out.detail != "0 plants", (name, body)


# --- the bodies the audit stands on -------------------------------------------------------------------


def test_bar_03_gets_its_supports_in_memory_only():
    bare = load_scene(BAR03_SCENE).packs["chibi"].pieces["chibi"].rig
    given = scene("bar03").packs["chibi"].pieces["chibi"].rig
    assert bare.supports == () and bare.sit is not None and bare.sit.past_edge == ()
    assert given.supports == ("ankle.L", "ankle.R")
    assert given.sit.past_edge == ("ankle.L", "ankle.R")
    assert given.sit.below_seat == bare.sit.below_seat
    assert bare.joints == {} and given.joints  # the helper adds the joints, never the bar's file
    assert set(given.chains) == set(bare.chains)


def test_bar_03_proxy_weights_are_in_memory_and_sum_to_one():
    """The timing proxy gives every vertex of every part a full weight, on a copy of the scene,
    in place of the weights the bar's rest file holds."""
    bare = bar03_scene()
    key = next(iter(bare.rigs))
    given = with_proxy_weights(bare)
    rest = given.rests[key]
    assert set(rest.weights) == set(rest.shells) and len(rest.shells) == 10
    for part, w in rest.weights.items():
        assert len(w) == len(rest.shells[part][0])
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-5), part
        assert rest.weight_groups[part].max() < len(rest.vertex_groups[part])


def test_bar_03_joints_meet_the_loader_s_rules():
    """The injected spec skips the loader, so check by hand what `_check_joints` would: each
    entry is a bending joint, each chain axis on an entry has a range, each chain limit lies
    inside it. And each why names its source column."""
    spec = scene("bar03").packs["chibi"].pieces["chibi"].rig
    assert set(spec.joints) <= set(bending_joints(spec))
    assert set(spec.joints) == {"body", "head", "arm.L", "arm.R", "forearm.L", "forearm.R",
                                "knee.L", "knee.R"}
    for chain in spec.chains.values():
        for bone in chain.bones:
            if bone in spec.joints:
                assert set(chain.dofs[bone]) <= set(spec.joints[bone].range_deg), bone
        for key, (lo, hi) in chain.limits.items():
            bone, _, axis = key.rpartition(":")
            if bone in spec.joints:
                j_lo, j_hi = (math.radians(v) for v in spec.joints[bone].range_deg[axis])
                assert j_lo - LIMIT_SLACK_RAD <= lo <= hi <= j_hi + LIMIT_SLACK_RAD, key
    assert all(re.search(r"column [ABC]\)", j.why) for j in spec.joints.values())
    # the joints the table has no cell for stay without an entry, so stress poses names them
    assert set(missing_ranges(spec)) == {"hand.L", "hand.R", "leg.L", "leg.R", "ankle.L", "ankle.R"}


def test_bar_03_chain_limits_are_cut_back_to_the_cited_range_in_memory_only():
    bare = load_scene(BAR03_SCENE).packs["chibi"].pieces["chibi"].rig
    given = scene("bar03").packs["chibi"].pieces["chibi"].rig
    assert bare.chains["arm.L"].limits["forearm.L:x"] == (-3.1416, 0.05)
    assert given.chains["arm.L"].limits["forearm.L:x"] == (-math.radians(150), 0.0)
    assert given.chains["arm.L"].limits["arm.L:y"] == (-1.3, math.radians(46))
    assert given.chains["arm.R"].limits["arm.R:y"] == (-math.radians(46), 1.3)
    assert given.chains["leg.L"].limits == bare.chains["leg.L"].limits == {}


def test_bar_03_joint_signs_follow_the_rest_bones():
    """The ranges' signs come from the bone frames, not from a name: pose each bone a little about
    one axis and see which way a point of the body moves. glTF axes: +x is the character's left,
    +y up and +z the front (the eyes sit at +z)."""
    ctx = scene("bar03")
    key = next(iter(ctx.rigs))
    rig = ctx.rigs[key]
    spec = ctx.packs["chibi"].pieces["chibi"].rig
    rest = pose_world(rig, {})

    def moves(bone: str, axis: int, point: tuple[float, float, float]) -> np.ndarray:
        """How a point, given from the bone's head in world axes, moves for +10 degrees about
        the bone's local axis."""
        euler = np.zeros(3)
        euler[axis] = math.radians(10)
        posed = pose_world(rig, {bone: (euler, np.zeros(3))})
        head = rest[bone][:3, 3]
        local = np.linalg.inv(rest[bone]) @ np.append(head + np.asarray(point), 1.0)
        return (posed[bone] @ local)[:3] - (head + np.asarray(point))

    below, above, front = (0, -0.1, 0), (0, 0.1, 0), (0, 0, 0.1)
    eyes = ctx.rests[key].shells["eye_L"][0].mean(axis=0)[2]
    assert eyes > ctx.rests[key].shells["head"][0].mean(axis=0)[2]  # the front is +z
    # flexion of a hanging limb is the negative x (the hand goes forward); the range agrees
    assert moves("forearm.L", 0, below)[2] < 0 and spec.joints["forearm.L"].range_deg["x"][0] < 0
    assert moves("arm.L", 0, below)[2] < 0 and spec.joints["arm.L"].range_deg["x"][0] == -180
    assert moves("knee.L", 0, below)[2] < 0 and spec.joints["knee.L"].range_deg["x"] == (-120, 20)
    # a trunk and a head flex on the positive x
    assert moves("body", 0, above)[2] > 0 and spec.joints["body"].range_deg["x"] == (-30, 90)
    assert moves("head", 0, above)[2] > 0 and spec.joints["head"].range_deg["x"] == (-45, 45)
    # +z abducts the left arm and adducts the right: the cited 180 is on the abduction side
    assert moves("arm.L", 2, below)[0] > 0 and spec.joints["arm.L"].range_deg["z"] == (-40, 180)
    assert moves("arm.R", 2, below)[0] > 0 and spec.joints["arm.R"].range_deg["z"] == (-180, 40)
    # +y turns the front of the arm toward the character's left: outward on the left arm (lateral
    # rotation, 46), inward on the right (medial, 91)
    assert moves("arm.L", 1, front)[0] > 0 and spec.joints["arm.L"].range_deg["y"] == (-91, 46)
    assert moves("arm.R", 1, front)[0] > 0 and spec.joints["arm.R"].range_deg["y"] == (-46, 91)


def test_the_worm_gives_its_mirror_reason_once():
    lines = [x for x in MirrorPair().coverage(scene("worm")) if "not applicable" in x]
    assert len(lines) == 1 and "no mirror map" in lines[0]
    assert [x for x in MirrorPair().coverage(scene("humanoid"))] == [
        "mirror-pair: not applicable (no clip has mirror_of): the scene"]


def test_a_scene_with_no_rigged_clip_gets_no_body_lines():
    ctx = dataclasses.replace(scene("humanoid"), rig_of_clip={})
    assert not [x for x in Sides().coverage(ctx) if "not applicable" in x]
    assert MirrorPair().coverage(ctx) == []


def test_a_body_with_no_chain_is_measured_by_its_supports():
    """The worm has no chain, so no effector radius; its belly's footprint stands in. Pose overlap
    used to skip it as `no effector radius`."""
    ctx = scene("worm")
    assert not ctx.packs["worm"].pieces["Worm"].rig.chains
    assert not [x for x in Overlap().coverage(ctx) if isinstance(x, Unmeasured)]
