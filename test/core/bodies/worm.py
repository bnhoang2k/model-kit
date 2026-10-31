"""A worm test body: eight segments in a line, with no left or right, no mirror and no seat.

Sizes are design choices for a test: 1 long and 0.08 across, lying on its belly along z (the
head end at +z, the underside at y 0). The rig has two parentless bones, `head_end` and
`tail_end`, each the root of a short chain of segment bones that meet at the middle, so the rig
names its root (`head_end`). One part, `body`, is bound across all eight bones, the head end
first. The one support is `belly`, every vertex of the body, so its lowest point is the underside.

Joint ranges. The joint-range sources table holds only human joints, so a worm segment has no
cited cell. Each range is a plain design choice for the test body, chosen without reading any
clip; the `why` says so. A segment bone turns about its local x (pitch) and local z (yaw about the
vertical, for a body lying flat); see `build.py`.

The layout's unjoined piece (the panel) stands one body size from the body: the longest side of
its bounding box, because a worm lying flat has almost no height.

The two bone trees meet at the middle. In a clip the tail tree's root is moved so the two
segments that meet there keep coincident ends, so the body never shears at the seam.
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
    posed,
    rig_data,
    sweep,
)
from .furniture import furniture

PACK = "worm"
BODY = "Worm"

RADIUS = 0.04
STEP = 0.125
DESIGN = "design choice for a test body: the joint-range table has no row for this joint"
# The segments from the head to the tail, as the part binds them.
ORDER = ("head_end", "seg_2", "seg_3", "seg_4", "seg_5", "seg_6", "seg_7", "tail_end")


def _bones() -> list[BoneDef]:
    def at(z0, z1):
        return (0, RADIUS, z0), (0, RADIUS, z1)
    bones = [BoneDef("head_end", None, *at(4 * STEP, 3 * STEP))]
    for k, name in enumerate(("seg_2", "seg_3", "seg_4"), start=1):
        bones.append(BoneDef(name, ORDER[k - 1], *at((3 - k + 1) * STEP, (3 - k) * STEP)))
    bones.append(BoneDef("tail_end", None, *at(-4 * STEP, -3 * STEP)))
    for k, name in enumerate(("seg_7", "seg_6", "seg_5"), start=1):
        parent = "tail_end" if k == 1 else ("seg_7", "seg_6")[k - 2]
        bones.append(BoneDef(name, parent, *at((-4 + k) * STEP, (-3 + k) * STEP)))
    return bones


def _body() -> BodyPart:
    zs = np.round(np.arange(-0.40, 0.4001, 0.05), 6)
    st = [(-0.47, 0, RADIUS, 0.025, 0.025), *[(z, 0, RADIUS, RADIUS, RADIUS) for z in zs],
          (0.47, 0, RADIUS, 0.025, 0.025)]
    verts, faces = sweep(st, "z", segments=16)
    edges = [(k * STEP + 0.03, k * STEP - 0.03) for k in range(3, -4, -1)]
    return BodyPart("body", verts, faces, ORDER, fall("z", 4 * STEP, -4 * STEP, *edges),
                    "body", "body")


def _joints() -> dict:
    return {b: {"range_deg": {"x": [-30, 30], "z": [-45, 45]}, "why": f"{DESIGN} ({b})"}
            for b in ("seg_2", "seg_3", "seg_4", "seg_5", "seg_6", "seg_7")}


def _seam(rig, pose) -> np.ndarray:
    """Where the end of `seg_4` and the end of `seg_5` lie in a pose, as two points."""
    return np.array([pose[b][:3, 3] + pose[b][:3, 1] * STEP for b in ("seg_4", "seg_5")])


def _clips(rig) -> dict[str, ClipSpec]:
    n = 24
    poses = []
    for k in range(n + 1):
        s = math.sin(2 * math.pi * k / n)
        turns = {"seg_2": (0, 0, 6 * s), "seg_3": (0, 0, -6 * s),
                 "seg_7": (0, 0, -6 * s), "seg_6": (0, 0, 6 * s)}
        head, tail = _seam(rig, posed(rig, turns))
        # Bending a tree shortens it a little (each end falls short of the middle), so the tail
        # tree's root moves by the gap and the seam ends meet exactly.
        poses.append(posed(rig, turns, {"tail_end": head - tail}))
    raw = {"frames": [1, n + 1], "fps": 24, "authoring": "keyed", "posture": "stand",
           "holds": [], "loop": True}
    return {"stand": ClipSpec(raw, np.arange(1.0, n + 2), poses)}


def _layout(extent: float) -> dict:
    def at(piece, x):
        return {"pack": PACK, "piece": piece, "position": [x, 0.0, 0.0],
                "quaternion": [0.0, 0.0, 0.0, 1.0]}
    return {"layout": PACK, "frame": "gltf",
            "placements": {"Worm_1": at(BODY, 0.0), "Table_1": at("Table", 1.2),
                           "Panel_1": at("Panel", -extent)},
            "stands": {"stand": {"position": [0.0, 0.0, 0.0], "facing": [0.0, 0.0, 1.0],
                                 "posture": "stand", "clip": "stand"}}}


def worm_scene() -> SceneCtx:
    """A fresh scene each call (about 50 ms), so a plant on one never reaches another."""
    bones = _bones()
    rig = rig_data(bones)
    part = _body()
    rig_raw = {"armature": "WormRig", "bones": bones_json(bones), "bind": bind_json([part]),
               "contact_groups": {"belly": list(ORDER)}, "chains": {}, "root": "head_end",
               "supports": ["belly"], "joints": _joints()}
    body_raw = {
        "frame": {"axes": "gltf", "origin": "floor, under the middle; the head end is +z"},
        "recipe": {"kind": "assembly", "source": "test/core/bodies/worm.py"},
        "listed": [{"id": "body", "text": "The body, one closed skinned shell",
                    "min_thickness": 0.05, "source": "test body, a design choice"}],
        "parts": [{"name": "body", "listed": "body", "shape": "ring", "on": None,
                   "group": "body", "role": "solid"}],
        "uses": [], "holds": [], "seats": [], "mounts": [],
        "done": "A worm of eight segments lying on its belly, with two parentless bones.",
        "size_class": "hero", "measured": {"export": {"position_bits": None}}, "rig": rig_raw}
    extent = float(np.ptp(part.verts, axis=0).max())
    return assemble(pack=PACK, body=BODY, body_raw=body_raw, parts=[part], rig=rig,
                    furniture=furniture("Table", "Panel"), clips=_clips(rig),
                    layout=_layout(extent), groups={"belly": list(ORDER)})

