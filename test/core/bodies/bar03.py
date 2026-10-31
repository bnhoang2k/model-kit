"""Bar 03's saved scene, with the rig data its pack.json will gain, added in memory.

The body-plan audit and the clip-hygiene tests run on bar 03 as a fourth body. Its pack.json has
no `supports`, no `sit.past_edge` and no `joints` yet: a later change writes them. Until then this helper
adds them to the loaded records (never to the bar's files): the ankle bones are the bar's shin
bones, their groups its two feet, and their heads are its knees.

Joint ranges (`JOINTS`). Each is a cell of the joint-range sources table (A is NASA HIDH, B is
AAOS via Hendriks 2020, C is 38 CFR 4.71a), written only for an axis that has a cell. The bar's
bone frames are not the humanoid's: every bone's local x is the character's left-right axis, its
local y is up and its local z is front. Each sign below was read off the rest bones by posing
(`test_bar_03_joint_signs_follow_the_rest_bones` repeats it), with the character's left side +x:

- A limb that hangs below its pivot swings back on +x, so flexion is a negative x (the elbow's
  150 degrees is x -150 to 0). A trunk or a head, above its pivot, tips forward on +x, so flexion
  is positive x.
- On a hanging arm, +z swings it toward the character's left: out for `arm.L` (abduction), in for
  `arm.R` (adduction). +y turns the front of the arm toward the left: outward (lateral rotation)
  for `arm.L`, inward (medial) for `arm.R`. The right arm's y and z ranges are the left's with
  the sign changed.
- The bar's bone names are off from the anatomy. `knee.*` pivots the thigh mesh at hip height
  (its pivot is 0.15 up, the thigh mesh runs 0.05 to 0.27 and the unbound `leg.*` head is 0.175),
  so it takes the hip's flexion and extension cell. `ankle.*` pivots the shin where thigh and shin
  overlap, which is the anatomical knee.

Left without an entry, because the loader refuses a joint with no range on a chain axis and the
table has no cell for the axis: `leg.*` (hip rotation, y), `ankle.*` (the knee holds only flexion,
and the chain turns it on xyz), `hand.*` (the wrist holds no rotation about its own bone, y). The
stress-poses check is then `Unmeasured` for those joints, which is accepted for bar 03 until then.

The chains' limits: where one lies outside its joint's cited range, the in-memory copy clamps it
(`forearm.*:x` is [-3.1416, 0.05] rad against [-150, 0] degrees; `arm.*:y` is plus or minus 1.3
rad against 46 and 91 degrees). Temporary: the narrowing is still to come. A Chain has no why, so the
reason is this paragraph.

A later change removes the injection: once the pack.json names the supports and joints, `bar03_scene` is
a plain `load_scene` and the audit's `Unmeasured` allowance for bar 03 goes with it.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np

from model_kit.core.forms import chain_weights
from model_kit.core.manifest import Chain, Joint
from model_kit.core.scene import SceneCtx, load_scene
from model_kit.core.tolerances import BIND_WEIGHT_FLOOR

SCENE = Path(__file__).resolve().parents[3] / "bars" / "03-chibi" / "scene"
SUPPORTS = ("ankle.L", "ankle.R")  # the contact groups of the two feet
PAST_EDGE = ("ankle.L", "ankle.R")  # the shin bones; their heads are the knees

SRC_A = "NASA HIDH Rev 1 Appendix B Table 2 (joint-range table, column A)"
SRC_B = "AAOS normal ROM via Hendriks 2020 SDC1 (joint-range table, column B)"
SRC_C = "38 CFR 4.71a Note (2) (joint-range table, column C)"


def flip(rng: tuple[float, float]) -> tuple[float, float]:
    """A left-side range for the right side: the turns about local y and z change sign."""
    return (-rng[1], -rng[0])


# Left-side (or midline) ranges in degrees, and their whys. See the module docstring for the signs.
LEFT = {
    "body": ({"x": (-30, 90), "y": (-30, 30), "z": (-30, 30)},
             (f"{SRC_C}: trunk flexion 90, extension 30, rotation 30 each way, lateral flexion "
              "30 each way")),
    "head": ({"x": (-45, 45), "y": (-60, 60), "z": (-45, 45)},
             (f"{SRC_B}: neck flexion 45, extension 45, rotation 60, lateral bend 45 (the `head` "
              "bone pivots at the neck)")),
    "arm": ({"x": (-180, 60), "y": (-91, 46), "z": (-40, 180)},
            (f"x and z from {SRC_B}: shoulder flexion 180, extension 60, abduction 180, "
             f"adduction 40 (adduction added by Hendriks, not AAOS); y from {SRC_A}: rotation "
             "lateral 46, medial 91")),
    "forearm": ({"x": (-150, 0)}, f"{SRC_B}: elbow flexion 150, extension 0"),
    "knee": ({"x": (-120, 20)},
             (f"{SRC_B}: hip flexion 120, extension 20; the bar's `knee` bone pivots the thigh "
              "at hip height, so it takes the hip's cell")),
}
BONES = {"body": "body", "head": "head", "arm": "arm.{}", "forearm": "forearm.{}",
         "knee": "knee.{}"}
SIDED = {"arm"}  # the joints whose y and z ranges change sign on the right


def joints() -> dict[str, Joint]:
    """The joints bar 03's pack.json will hold, by bone."""
    out = {}
    for base, (ranges, why) in LEFT.items():
        if base in ("body", "head"):
            out[BONES[base]] = Joint(dict(ranges), why)
            continue
        for side in "LR":
            right = side == "R" and base in SIDED
            out[BONES[base].format(side)] = Joint(
                {a: flip(r) if right and a in "yz" else r for a, r in ranges.items()}, why)
    return out


def _clamped(chain: Chain, ranges: dict[str, Joint]) -> Chain:
    """The chain with each limit that lies outside its joint's range cut back to the range.
    Temporary: the narrowing is still to come."""
    limits = {}
    for key, (lo, hi) in chain.limits.items():
        bone, _, axis = key.rpartition(":")
        if bone in ranges:
            j_lo, j_hi = (math.radians(v) for v in ranges[bone].range_deg[axis])
            lo, hi = max(lo, j_lo), min(hi, j_hi)
        limits[key] = (lo, hi)
    return dataclasses.replace(chain, limits=limits)


def bar03_scene() -> SceneCtx:
    """The bar's scene as saved, its rigs given `supports`, `joints` (with the chain limits cut
    back to them) and, where they have a `sit`, `sit.past_edge`. The scene is built fresh on each
    call (loading is the slow part)."""
    ctx = load_scene(SCENE)
    packs = {}
    for name, pack in ctx.packs.items():
        pieces = {}
        for piece_name, piece in pack.pieces.items():
            rig = piece.rig
            if rig is not None:
                sit = rig.sit
                if sit is not None:
                    sit = dataclasses.replace(sit, past_edge=PAST_EDGE)
                ranges = joints()
                chains = {key: _clamped(chain, ranges) for key, chain in rig.chains.items()}
                rig = dataclasses.replace(rig, supports=SUPPORTS, sit=sit, joints=ranges,
                                          chains=chains)
                piece = dataclasses.replace(piece, rig=rig)
            pieces[piece_name] = piece
        packs[name] = dataclasses.replace(pack, pieces=pieces)
    return dataclasses.replace(ctx, packs=packs)


def with_proxy_weights(ctx: SceneCtx) -> SceneCtx:
    """The scene with skin weights for its rig, in memory only. Temporary: bar 03's
    real weights are still to come (its rest file is `AWAITING_WEIGHTS` today, so stress-poses reads `Unmeasured`).

    The weights come from the pack's own bind, the way the bind step makes them: a rigid part
    weighs 1 on its one bone, a chain part gets `chain_weights` along the bind's falloff, floored
    at `BIND_WEIGHT_FLOOR` and renormalised. They exist so that a timing run measures bar 03's
    real meshes posed by every stress pose. They are a proxy and no finding on them is a verdict."""
    key = next(iter(ctx.rigs))
    rest = ctx.rests[key]
    spec = next(piece.rig for pack in ctx.packs.values() for piece in pack.pieces.values()
                if piece.rig is not None)
    weights, groups = {}, {}
    for part, shell in rest.shells.items():
        bind = spec.bind[part]
        assert tuple(rest.vertex_groups[part]) == tuple(bind.bones), part
        verts = shell[0].astype(float)
        if bind.mode == "rigid":
            w = np.ones((len(verts), 1))
        else:
            w = chain_weights(verts, bind.falloff, len(bind.bones))
            w = np.where(w <= BIND_WEIGHT_FLOOR, 0.0, w)
            w = w / w.sum(axis=1, keepdims=True)
        k = int((w > 0).sum(axis=1).max())
        order = np.argsort(w <= 0, axis=1, kind="stable")[:, :k]
        weights[part] = np.take_along_axis(w, order, 1).astype(np.float32)
        groups[part] = np.where(weights[part] > 0, order, 0).astype(np.int32)
    return dataclasses.replace(
        ctx, rests={key: dataclasses.replace(rest, weights=weights, weight_groups=groups)})
