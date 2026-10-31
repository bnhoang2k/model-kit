"""A tall humanoid test body: 1.75 tall, two legs, two arms, a trunk and a head.

Proportions are approximate adult proportions, a design choice: the joint heights follow the usual
anthropometric fractions of standing height (ankle 0.039, knee 0.285, hip 0.530, wrist 0.485,
elbow 0.630, shoulder 0.818), approximate and not checked against an anthropometric table. Radii are design
choices too.

Parts (each a closed skinned shell, so a fold at an elbow or a knee is a self-crossing inside one
shell): `torso` (trunk, neck and head, bound to pelvis, spine and neck), `leg_L`/`leg_R` (hip to
sole, bound to hip, knee and ankle) and `arm_L`/`arm_R` (shoulder to fingertip, bound to
shoulder, elbow and wrist). Each bone is named for the joint at its head, so the bones that bend
the skin are exactly the joints the range table lists: spine, neck, hip, knee, ankle, shoulder,
elbow, wrist. The root is `pelvis`. The figure faces +z; its left side is +x. The rest pose is
the anatomical position: arms at the sides, palms forward (so a hand is broad in x and thin
in z).

Joint ranges are degrees about each bone's local axes (see `build.py`): x flexion and extension,
z abduction or lateral bend, y twist about the bone. A positive x flexes the trunk, the neck and
the knee, and extends the hip and the shoulder; the elbow, the wrist and the ankle flex on a
negative x (the ankle's plantar flexion is positive). A positive z adducts the left hip and
shoulder (toward the midline) and deviates the left wrist toward the ulna. A positive y twists the
left shoulder medially and the left forearm toward pronation. The right side's z and y ranges are
the left side's with the sign changed. Every range comes from a cell of the joint-range sources table
(A is NASA HIDH, B is AAOS via Hendriks 2020, C is 38 CFR 4.71a). An axis with no cell is
not a dof: the hip has x and z only (no source for hip rotation), the knee x only, the ankle x
only.
"""

from __future__ import annotations

import math

import numpy as np

from model_kit.core.scene import SceneCtx

from .build import (
    BodyPart,
    BoneDef,
    ClipSpec,
    assemble,
    bind_json,
    bones_json,
    fall,
    flip,
    mirror_part,
    posed,
    rig_data,
    set_effector,
    sweep,
)
from .furniture import BENCH_SEAT_TOP, CHAIR_SEAT_TOP, furniture

PACK = "humanoid"
BODY = "Person"

HEIGHT = 1.75
PELVIS_Y, WAIST_Y, NECK_Y = 0.8325, 1.0, 1.45
HIP_X, HIP_Y, KNEE_Y, ANKLE_Y = 0.09, 0.9275, 0.49875, 0.06825
SHOULDER_X, SHOULDER_Y, ELBOW_Y, WRIST_Y, FINGER_Y = 0.22, 1.4315, 1.1025, 0.84875, 0.65975

SRC_A = "NASA HIDH Rev 1 Appendix B Table 2 (joint-range table, column A)"
SRC_B = "AAOS normal ROM via Hendriks 2020 SDC1 (joint-range table, column B)"
SRC_C = "38 CFR 4.71a Note (2) (joint-range table, column C)"

# Left-side ranges in degrees. The right side flips y and z (see the module docstring).
LEFT_RANGES = {
    "spine": ({"x": (-30, 90), "y": (-30, 30), "z": (-30, 30)},
              (f"{SRC_C}: trunk flexion 90, extension 30, rotation 30 each way, lateral "
               "flexion 30 each way")),
    "neck": ({"x": (-45, 45), "y": (-60, 60), "z": (-45, 45)},
             (f"{SRC_B}: neck flexion 45, extension 45, rotation 60, lateral bend 45")),
    "shoulder": ({"x": (-180, 60), "y": (-46, 91), "z": (-180, 40)},
                 (f"x and z from {SRC_B}: flexion 180, extension 60, abduction 180, adduction "
                  f"40 (adduction added by Hendriks, not AAOS); y from {SRC_A}: rotation "
                  "lateral 46, medial 91")),
    "elbow": ({"x": (-150, 0), "y": (-83, 78)},
              (f"x from {SRC_B}: elbow flexion 150, extension 0; y, the forearm twist carried "
               f"on the elbow bone, from {SRC_A}: supination 83, pronation 78")),
    "wrist": ({"x": (-80, 70), "z": (-16, 19)},
              (f"x from {SRC_B}: wrist flexion 80, extension 70; z from {SRC_A}: ulnar "
               "deviation 19, radial 16")),
    "hip": ({"x": (-120, 20), "z": (-40, 20)},
            (f"{SRC_B}: hip flexion 120, extension 20, abduction 40, adduction 20 (hip "
             "rotation has no source, so it is not a dof)")),
    "knee": ({"x": (0, 135)}, f"{SRC_B}: knee flexion 135, extension 0"),
    "ankle": ({"x": (-20, 50)}, f"{SRC_B}: ankle plantar flexion 50, dorsiflexion 20"),
}
CHAIN_DOFS = {"hip": "xz", "knee": "x", "ankle": "x",
              "shoulder": "xyz", "elbow": "x", "wrist": "xz"}
SIDES = {"L": 1.0, "R": -1.0}
# What each part rests on: the legs on the floor, the trunk on a leg, the arms on the trunk.
ON = {"leg_L": None, "leg_R": None, "torso": "leg_L", "arm_L": "torso", "arm_R": "torso"}


def _bones() -> list[BoneDef]:
    bones = [BoneDef("pelvis", None, (0, PELVIS_Y, 0), (0, WAIST_Y, 0)),
             BoneDef("spine", "pelvis", (0, WAIST_Y, 0), (0, NECK_Y, 0)),
             BoneDef("neck", "spine", (0, NECK_Y, 0), (0, 1.70, 0))]
    for side, s in SIDES.items():
        legs = [("hip", "pelvis", HIP_Y, KNEE_Y), ("knee", f"hip.{side}", KNEE_Y, ANKLE_Y)]
        for name, parent, y0, y1 in legs:
            bones.append(BoneDef(f"{name}.{side}", parent, (s * HIP_X, y0, 0), (s * HIP_X, y1, 0)))
        bones.append(BoneDef(f"ankle.{side}", f"knee.{side}", (s * HIP_X, ANKLE_Y, 0),
                             (s * HIP_X, 0.03, 0.2)))
        arms = [("shoulder", "spine", SHOULDER_Y, ELBOW_Y), ("elbow", f"shoulder.{side}",
                                                              ELBOW_Y, WRIST_Y),
                ("wrist", f"elbow.{side}", WRIST_Y, FINGER_Y)]
        for name, parent, y0, y1 in arms:
            bones.append(BoneDef(f"{name}.{side}", parent, (s * SHOULDER_X, y0, 0),
                                 (s * SHOULDER_X, y1, 0)))
    return bones


def _torso() -> BodyPart:
    st = [(0.8355, 0, -0.005, 0.12, 0.09), (0.88, 0, -0.005, 0.15, 0.105),
          (0.93, 0, -0.005, 0.165, 0.11), (1.00, 0, 0, 0.14, 0.10), (1.15, 0, 0.005, 0.15, 0.105),
          (1.30, 0, 0.01, 0.17, 0.115), (1.39, 0, 0, 0.19, 0.105), (1.43, 0, 0, 0.13, 0.085),
          (1.46, 0, 0, 0.055, 0.058), (1.52, 0, 0.01, 0.050, 0.055),
          (1.56, 0, 0.01, 0.052, 0.058), (1.60, 0, 0.015, 0.07, 0.085),
          (1.66, 0, 0.01, 0.075, 0.095), (1.69, 0, 0.0, 0.06, 0.07)]
    verts, faces = sweep(st, "y", caps=("flat", "round"))
    return BodyPart("torso", verts, faces, ("pelvis", "spine", "neck"),
                    fall("y", PELVIS_Y, HEIGHT, (0.95, 1.05), (1.43, 1.47)),
                    "torso", "torso", shape="blob")


def _leg() -> BodyPart:
    """The left leg; the right is its mirror image."""
    side, x = "L", HIP_X
    rows = [(0.95, 0.0, 0.082, 0.085), (0.80, 0.0, 0.078, 0.080), (0.65, 0.0, 0.068, 0.070),
            (0.55, 0.0, 0.060, 0.062), (KNEE_Y, 0.005, 0.056, 0.058), (0.40, 0.0, 0.050, 0.050),
            (0.25, 0.0, 0.042, 0.043), (0.12, 0.0, 0.036, 0.037), (ANKLE_Y, 0.0, 0.034, 0.035),
            (0.055, 0.02, 0.036, 0.050), (0.035, 0.05, 0.040, 0.085),
            (0.015, 0.07, 0.043, 0.115), (0.0, 0.07, 0.043, 0.125)]
    verts, faces = sweep([(y, x, cz, hx, hz) for y, cz, hx, hz in rows], "y",
                         caps=("round", "flat"))
    return BodyPart(f"leg_{side}", verts, faces,
                    (f"hip.{side}", f"knee.{side}", f"ankle.{side}"),
                    fall("y", HIP_Y, 0.0, (0.55, 0.45), (0.10, 0.04)), "leg", "legs")


def _arm() -> BodyPart:
    """The left arm; the right is its mirror image."""
    side, x = "L", SHOULDER_X
    rows = [(1.40, 0.050, 0.052), (1.30, 0.048, 0.050), (1.20, 0.043, 0.045),
            (ELBOW_Y, 0.038, 0.040), (1.00, 0.036, 0.038), (0.90, 0.031, 0.034),
            (0.85, 0.027, 0.030), (0.80, 0.040, 0.020), (0.72, 0.042, 0.018),
            (0.67, 0.032, 0.014)]
    verts, faces = sweep([(y, x, 0.0, hx, hz) for y, hx, hz in rows], "y")
    return BodyPart(f"arm_{side}", verts, faces,
                    (f"shoulder.{side}", f"elbow.{side}", f"wrist.{side}"),
                    fall("y", SHOULDER_Y, 0.66, (1.15, 1.05), (0.89, 0.81)), "arm", "arms")


def _joints() -> dict:
    out = {}
    for base, (ranges, why) in LEFT_RANGES.items():
        names = [base] if base in ("spine", "neck") else [f"{base}.L", f"{base}.R"]
        for name in names:
            right = name.endswith(".R")
            out[name] = {"range_deg": {a: list(flip(r) if right and a in "yz" else r)
                                       for a, r in ranges.items()}, "why": why}
    return out


def _rig_raw(bones, parts, rig) -> tuple[dict, dict]:
    groups = {f"{n}.{s}": [f"{b}.{s}"] for n, b in (("foot", "ankle"), ("hand", "wrist"))
              for s in SIDES}
    chains = {}
    for side in SIDES:
        for chain, root, names, effector, group in (
                ("leg", "pelvis", ("hip", "knee", "ankle"), "ankle", "foot"),
                ("arm", "spine", ("shoulder", "elbow", "wrist"), "wrist", "hand")):
            bones_ = [f"{n}.{side}" for n in names]
            radius = set_effector(rig, parts, f"{effector}.{side}", groups[f"{group}.{side}"])
            chains[f"{chain}.{side}"] = {
                "root": root, "bones": bones_, "effector": bones_[-1], "effector_radius": radius,
                "dofs": {b: CHAIN_DOFS[b.split(".")[0]] for b in bones_}, "limits": {},
                "effector_group": f"{group}.{side}"}
    pairs = [[f"{n}.L", f"{n}.R"] for n in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist")]
    return {"armature": "HumanoidRig", "bones": bones_json(bones), "bind": bind_json(parts),
            "contact_groups": groups, "chains": chains, "root": "pelvis",
            "sit": {"below_seat": 0.0, "forward": 0.03, "past_edge": ["knee.L", "knee.R"]},
            "mirror": {"pairs": pairs, "axis": "x"}, "supports": ["foot.L", "foot.R"],
            "joints": _joints()}, groups


def _breath(rig, k: int, n: int, extra: dict | None = None, moves=None):
    """One sample of quiet breathing: the spine and neck nod a degree or so. The root is still."""
    ph = 2 * math.pi * k / n
    turns = {"spine": (1.5 * math.sin(ph), 0, 0), "neck": (-0.8 * math.sin(ph), 0, 0)}
    turns.update(extra or {})
    return posed(rig, turns, moves)


def _lift(t: float) -> float:
    """The step's lift share, 0 to 1: still until frame 5, up by 8, held to 16, down by 19. The
    ramp is three frames long, so the foot has risen out of its planted band before it has moved
    a slide tolerance across the floor."""
    def smooth(u):
        u = min(max(u, 0.0), 1.0)
        return u * u * (3 - 2 * u)
    return smooth((t - 5) / 3) * (1 - smooth((t - 16) / 3))


def _sit_turns(seat_top: float, land_feet: bool) -> tuple[dict, float]:
    """Euler degrees for a sitter: thighs level, or sloping so the soles land on the floor; shins
    straight down; forearms held out level in front of the thighs. `seat_top` comes back too, the
    height of the pelvis bone's head."""
    hip_above = HIP_Y - PELVIS_Y
    thigh = HIP_Y - KNEE_Y
    drop = math.degrees(math.asin((seat_top + hip_above - KNEE_Y) / thigh)) if land_feet else 0.0
    bend = 90.0 - drop
    turns = {}
    for side in SIDES:
        turns[f"hip.{side}"] = (-bend, 0, 0)
        turns[f"knee.{side}"] = (bend, 0, 0)
        turns[f"shoulder.{side}"] = (-10.0, 0, 0)
        turns[f"elbow.{side}"] = (-80.0, 0, 0)
    return turns, seat_top


def _sit_clip(seat_piece: str, seat_top: float, land_feet: bool, rig) -> ClipSpec:
    turns, top = _sit_turns(seat_top, land_feet)
    move = (0.0, top - PELVIS_Y, 0.0)  # the seat point is (0, top, -0.03) and forward is 0.03
    n = 12
    poses = [_breath(rig, k, n, turns, {"pelvis": move}) for k in range(n + 1)]
    raw = {"frames": [1, n + 1], "fps": 24, "authoring": "keyed", "posture": "sit",
           "seat": f"{PACK}:{seat_piece}.seat_1", "loop": True, "holds": [],
           "floor_gear": {seat_piece: {"piece": seat_piece, "pack": PACK, "position": [0, 0, 0],
                                       "quaternion": [0, 0, 0, 1]}}}
    return ClipSpec(raw, np.arange(1.0, n + 2), poses)


def _clips(rig) -> dict[str, ClipSpec]:
    n = 24
    base = {"fps": 24, "authoring": "keyed", "posture": "stand", "holds": []}
    stand = ClipSpec({"frames": [1, n + 1], "loop": True, **base}, np.arange(1.0, n + 2),
                     [_breath(rig, k, n) for k in range(n + 1)])
    steps = []
    for t in range(1, 26):
        s = _lift(t)
        steps.append(posed(rig, {"hip.L": (-30 * s, 0, 0), "knee.L": (45 * s, 0, 0),
                                 "ankle.L": (-15 * s, 0, 0)}))  # the net turn is 0: foot level
    step = ClipSpec({"frames": [1, 25], **base, "lifts": [{
        "support": "foot.L", "when": [[6, 18]],
        "why": "a step: the left foot lifts and sets down again"}]},
        np.arange(1.0, 26), steps)
    return {"stand": stand, "step": step,
            "sit_chair": _sit_clip("Chair", CHAIR_SEAT_TOP, True, rig),
            "sit_bench": _sit_clip("Bench", BENCH_SEAT_TOP, False, rig)}


def _layout(height: float) -> dict:
    def at(piece, x, z=0.0):
        return {"pack": PACK, "piece": piece, "position": [x, 0.0, z],
                "quaternion": [0.0, 0.0, 0.0, 1.0]}

    def stand(x, posture, clip):
        return {"position": [x, 0.0, 0.0], "facing": [0.0, 0.0, 1.0], "posture": posture,
                "clip": clip}
    return {"layout": PACK, "frame": "gltf",
            "placements": {"Person_1": at(BODY, 0.0), "Chair_1": at("Chair", 2.0),
                           "Bench_1": at("Bench", 3.6), "Table_1": at("Table", 2.0, 1.4),
                           "Panel_1": at("Panel", -height)},
            "stands": {"stand": stand(0.0, "stand", "stand"), "step": stand(0.0, "stand", "step"),
                       "chair": stand(2.0, "sit", "sit_chair"),
                       "bench": stand(3.6, "sit", "sit_bench")}}


def humanoid_scene() -> SceneCtx:
    """A fresh scene each call (about 50 ms), so a plant on one never reaches another."""
    bones = _bones()
    rig = rig_data(bones)
    leg, arm = _leg(), _arm()
    parts = [_torso(), leg, mirror_part(leg, "leg_R", ("hip.R", "knee.R", "ankle.R")), arm,
             mirror_part(arm, "arm_R", ("shoulder.R", "elbow.R", "wrist.R"))]
    rig_raw, groups = _rig_raw(bones, parts, rig)
    mins = {"torso": 0.1, "leg": 0.06, "arm": 0.025}
    listed = [{"id": k, "text": f"The {k}, one closed skinned shell", "min_thickness": m,
               "source": "test body, a design choice"} for k, m in mins.items()]
    body_raw = {
        "frame": {"axes": "gltf", "origin": "floor, between the feet; +x the figure's left, "
                                            "+z its front"},
        "recipe": {"kind": "assembly", "source": "test/core/bodies/humanoid.py"},
        "listed": listed,
        "parts": [{"name": p.name, "listed": p.listed, "shape": p.shape, "on": ON[p.name],
                   "group": p.group, "role": "solid"} for p in parts],
        "uses": [], "holds": [], "seats": [], "mounts": [],
        "done": "A 1.75-tall person: five skinned shells, 14 bending joints, two feet that bear "
                "the weight.",
        "size_class": "hero", "measured": {"export": {"position_bits": None}}, "rig": rig_raw}
    height = max(float(p.verts[:, 1].max()) for p in parts)
    return assemble(pack=PACK, body=BODY, body_raw=body_raw, parts=parts, rig=rig,
                    furniture=furniture("Chair", "Bench", "Table", "Panel"), clips=_clips(rig),
                    layout=_layout(height), groups=groups)

