"""A quadruped test body: four legs, a barrel body with a neck and head, and a six-bone tail.

Sizes are design choices for a test: about 0.58 high at the head, 0.46 at the back and 1.5 long
with the tail. The animal faces +z; its left side is +x. The root is `pelvis`; all four legs hang
from it, so a nod of the chest never lifts a paw. Each leg is one skinned part from hip (or shoulder) to the paw, bound to three bones named for the joints at their heads.
The `body` part (barrel, neck and head) is bound to pelvis, chest and neck; the `tail` part is
bound to the six tail bones, which run back from the pelvis.

Supports are the four paws (contact groups `paw.FL`, `paw.FR`, `paw.HL`, `paw.HR`, each the paw
bone's strongest vertices). The tail is not a support: it is a six-bone chain (`tail`, rooted at
the pelvis, its tip the contact group `tail_tip`) kept out of `supports`. Each leg is also a
chain rooted at the pelvis, so the paws are foot groups for today's checks and have chain radii.
The right legs are the left legs mirrored, triangle for triangle.

The layout's unjoined piece (the panel) stands one body size from the body: the longest side of
the body's bounding box (a body height is not meant for a body this low).

Joint ranges. The joint-range sources table holds only human joints, so a quadruped's joints
have no cited cell. Each range here is a plain design choice for the test body, a symmetric or
simple span chosen without reading any clip; the `why` says so. Axes follow `build.py`: x pitch,
z side bend. A positive x on a downward leg bone swings its tip back.
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
    mirror_part,
    posed,
    rig_data,
    set_effector,
    sweep,
)
from .furniture import furniture

PACK = "quadruped"
BODY = "Animal"

LEG_X = 0.09
FORE_Z, HIND_Z = 0.22, -0.28
TOP_Y, MID_Y, LOW_Y = 0.42, 0.25, 0.08
TAIL_Y, TAIL_Z0, TAIL_STEP, TAIL_BONES = 0.50, -0.38, 0.07, 6
DESIGN = "design choice for a test body: the joint-range table has no row for this joint"
RANGES = {"chest": {"x": (-15, 15), "z": (-15, 15)}, "neck": {"x": (-30, 30), "z": (-30, 30)},
          "upper": {"x": (-60, 60)}, "middle": {"x": (0, 120)}, "paw": {"x": (-45, 45)},
          "tail": {"x": (-30, 30), "z": (-30, 30)}}
LEGS = (("fore", FORE_Z, ("shoulder", "elbow", "wrist"), "paw.F"),
        ("hind", HIND_Z, ("hip", "knee", "ankle"), "paw.H"))
SIDES = {"L": 1.0, "R": -1.0}


def _bones() -> list[BoneDef]:
    bones = [BoneDef("pelvis", None, (0, 0.46, -0.30), (0, 0.46, 0.0)),
             BoneDef("chest", "pelvis", (0, 0.46, 0.0), (0, 0.50, 0.38)),
             BoneDef("neck", "chest", (0, 0.50, 0.38), (0, 0.58, 0.72))]
    for kind, z, names, _ in LEGS:
        for side, s in SIDES.items():
            x = s * LEG_X
            parent = "pelvis"
            ys = (TOP_Y, MID_Y, LOW_Y)
            for i, name in enumerate(names):
                tail = (x, ys[i + 1], z) if i < 2 else (x, 0.03, z + 0.1)
                bones.append(BoneDef(f"{kind}_{name}.{side}", parent, (x, ys[i], z), tail))
                parent = f"{kind}_{name}.{side}"
    prev = "pelvis"
    for k in range(TAIL_BONES):
        z = TAIL_Z0 - TAIL_STEP * k
        bones.append(BoneDef(f"tail_{k + 1}", prev, (0, TAIL_Y, z), (0, TAIL_Y, z - TAIL_STEP)))
        prev = f"tail_{k + 1}"
    return bones


def _body() -> BodyPart:
    st = [(-0.40, 0, 0.46, 0.07, 0.09), (-0.30, 0, 0.47, 0.10, 0.13),
          (-0.10, 0, 0.46, 0.11, 0.14), (0.10, 0, 0.47, 0.11, 0.15), (0.25, 0, 0.48, 0.11, 0.15),
          (0.38, 0, 0.52, 0.06, 0.08), (0.46, 0, 0.57, 0.05, 0.06), (0.52, 0, 0.58, 0.07, 0.07),
          (0.62, 0, 0.58, 0.065, 0.06), (0.70, 0, 0.56, 0.035, 0.035)]
    verts, faces = sweep(st, "z")
    return BodyPart("body", verts, faces, ("pelvis", "chest", "neck"),
                    fall("z", -0.30, 0.72, (-0.08, 0.08), (0.33, 0.43)), "body", "body",
                    shape="blob")


def _tail() -> BodyPart:
    st = [(-0.36, 0, TAIL_Y, 0.035, 0.035), (-0.50, 0, TAIL_Y, 0.032, 0.032),
          (-0.64, 0, TAIL_Y, 0.025, 0.025), (-0.76, 0, TAIL_Y, 0.018, 0.018),
          (-0.80, 0, TAIL_Y, 0.012, 0.012)]
    verts, faces = sweep(st, "z", segments=12)
    names = tuple(f"tail_{k + 1}" for k in range(TAIL_BONES))
    joints = [TAIL_Z0 - TAIL_STEP * k for k in range(1, TAIL_BONES)]
    return BodyPart("tail", verts, faces, names,
                    fall("z", TAIL_Z0, TAIL_Z0 - TAIL_STEP * TAIL_BONES + 0.04,
                         *[(z + 0.02, z - 0.02) for z in joints]), "tail", "tail")


def _leg(kind: str, z: float, names) -> BodyPart:
    """The left leg; the right is its mirror image."""
    side, x = "L", LEG_X
    thick = 1.2 if kind == "hind" else 1.0
    rows = [(0.46, 0.0, 0.055 * thick, 0.065 * thick), (0.35, 0.0, 0.048 * thick, 0.055 * thick),
            (MID_Y, 0.0, 0.036, 0.04), (0.16, 0.0, 0.030, 0.034), (LOW_Y + 0.01, 0.0, 0.028, 0.032),
            (0.06, 0.02, 0.030, 0.045), (0.03, 0.04, 0.034, 0.065), (0.0, 0.05, 0.034, 0.07)]
    verts, faces = sweep([(y, x, z + cz, hx, hz) for y, cz, hx, hz in rows], "y",
                         caps=("round", "flat"), segments=12)
    return BodyPart(f"leg_{kind}_{side}", verts, faces,
                    tuple(f"{kind}_{n}.{side}" for n in names),
                    fall("y", TOP_Y, 0.0, (0.30, 0.20), (0.12, 0.05)), "leg", "legs")


def _joint(base: str, axes: dict) -> dict:
    return {"range_deg": {a: list(r) for a, r in axes.items()}, "why": f"{DESIGN} ({base})"}


def _joints() -> dict:
    out = {"chest": _joint("chest", RANGES["chest"]), "neck": _joint("neck", RANGES["neck"])}
    for kind, _, names, _ in LEGS:
        for side in SIDES:
            for name, role in zip(names, ("upper", "middle", "paw"), strict=True):
                out[f"{kind}_{name}.{side}"] = _joint(name, RANGES[role])
    for k in range(TAIL_BONES):
        out[f"tail_{k + 1}"] = _joint("tail", RANGES["tail"])
    return out


def _rig_raw(bones, parts, rig) -> tuple[dict, dict]:
    groups, chains = {}, {}
    for kind, _, names, prefix in LEGS:
        for side in SIDES:
            paw = f"{kind}_{names[2]}.{side}"
            group = f"{prefix}{side}"
            groups[group] = [paw]
            leg = [f"{kind}_{n}.{side}" for n in names]
            radius = set_effector(rig, parts, paw, [paw])
            chains[f"leg.{prefix[-1]}{side}"] = {
                "root": "pelvis", "bones": leg, "effector": paw, "effector_radius": radius,
                "dofs": {b: "x" for b in leg}, "limits": {}, "effector_group": group}
    tail = [f"tail_{k + 1}" for k in range(TAIL_BONES)]
    groups["tail_tip"] = [tail[-1]]
    chains["tail"] = {
        "root": "pelvis", "bones": tail, "effector": tail[-1],
        "effector_radius": set_effector(rig, parts, tail[-1], [tail[-1]]),
        "dofs": {b: "xz" for b in tail}, "limits": {}, "effector_group": "tail_tip"}
    pairs = [[f"{kind}_{n}.L", f"{kind}_{n}.R"] for kind, _, names, _ in LEGS for n in names]
    return {"armature": "QuadrupedRig", "bones": bones_json(bones), "bind": bind_json(parts),
            "contact_groups": groups, "chains": chains, "mirror": {"pairs": pairs, "axis": "x"},
            "supports": [g for g in groups if g.startswith("paw.")], "joints": _joints()}, groups


def _clips(rig) -> dict[str, ClipSpec]:
    n = 24
    poses = []
    for k in range(n + 1):
        ph = 2 * math.pi * k / n
        turns = {"chest": (1.0 * math.sin(ph), 0, 0), "neck": (-1.0 * math.sin(ph), 0, 0)}
        for j in range(TAIL_BONES):
            turns[f"tail_{j + 1}"] = (0, 0, 8.0 * math.sin(ph - 0.6 * j))
        poses.append(posed(rig, turns))
    raw = {"frames": [1, n + 1], "fps": 24, "authoring": "keyed", "posture": "stand",
           "holds": [], "loop": True}
    return {"stand": ClipSpec(raw, np.arange(1.0, n + 2), poses)}


def _on(part: str) -> str | None:
    """What a part rests on: the legs on the floor, the body on a front leg, the tail on the body."""
    return {"body": "leg_fore_L", "tail": "body"}.get(part)


def _layout(extent: float) -> dict:
    def at(piece, x):
        return {"pack": PACK, "piece": piece, "position": [x, 0.0, 0.0],
                "quaternion": [0.0, 0.0, 0.0, 1.0]}
    return {"layout": PACK, "frame": "gltf",
            "placements": {"Animal_1": at(BODY, 0.0), "Table_1": at("Table", 1.2),
                           "Panel_1": at("Panel", -extent)},
            "stands": {"stand": {"position": [0.0, 0.0, 0.0], "facing": [0.0, 0.0, 1.0],
                                 "posture": "stand", "clip": "stand"}}}


def quadruped_scene() -> SceneCtx:
    """A fresh scene each call (about 50 ms), so a plant on one never reaches another."""
    bones = _bones()
    rig = rig_data(bones)
    parts = [_body(), _tail()]
    for kind, z, names, _ in LEGS:
        left = _leg(kind, z, names)
        parts += [left, mirror_part(left, f"leg_{kind}_R", tuple(f"{kind}_{n}.R" for n in names))]
    rig_raw, groups = _rig_raw(bones, parts, rig)
    listed = [{"id": k, "text": f"The {k}, one closed skinned shell", "min_thickness": m,
               "source": "test body, a design choice"}
              for k, m in (("body", 0.1), ("tail", 0.02), ("leg", 0.05))]
    body_raw = {
        "frame": {"axes": "gltf", "origin": "floor, under the pelvis bone's head; +x the "
                                            "animal's left, +z its front"},
        "recipe": {"kind": "assembly", "source": "test/core/bodies/quadruped.py"},
        "listed": listed,
        "parts": [{"name": p.name, "listed": p.listed, "shape": p.shape,
                   "on": _on(p.name), "group": p.group, "role": "solid"}
                  for p in parts],
        "uses": [], "holds": [], "seats": [], "mounts": [],
        "done": "A four-legged animal with a six-bone tail: six skinned shells, four paws that "
                "bear the weight.",
        "size_class": "hero", "measured": {"export": {"position_bits": None}}, "rig": rig_raw}
    every = np.concatenate([p.verts for p in parts])
    extent = float((every.max(axis=0) - every.min(axis=0)).max())
    return assemble(pack=PACK, body=BODY, body_raw=body_raw, parts=parts, rig=rig,
                    furniture=furniture("Table", "Panel"), clips=_clips(rig),
                    layout=_layout(extent), groups=groups)

