"""Shared scenes and turns for the core tests. A helper module, not a test file.

`make_ctx` builds the reach scene (one solved clip, one target) that the reach, target and gate
tests share. `rest`, `turn_y`, `turn_z`, `quat_y` and `matrix` build the matrices every check
test needs.
"""

import json
import tempfile
from functools import cache
from pathlib import Path

import numpy as np

from model_kit.core.manifest import (
    Clip,
    ClipHold,
    Dangle,
    FloorGear,
    Pack,
    Placement,
    Prop,
    Ref,
    Target,
    load_pack,
)
from model_kit.core.scene import ClipSamples, PieceGeom, RigData, SceneCtx
from model_kit.core.tolerances import REACH_MARGIN_OF_EFFECTOR
from model_kit.core.units import Style

from .shapes import plate

RADIUS = 0.0586
MARGIN = REACH_MARGIN_OF_EFFECTOR * RADIUS
# Reach is measured from the chain's first bone (the shoulder, the hip), not from `body`.
SHOULDER_X = 0.3  # arm.R's head, from body
ARM_REACH = 0.5  # 0.3 between heads, plus 0.2 of effector_mid
HIP_Y = -0.4  # leg.R's head, from body
LEG_REACH = 0.5  # 0.4 between heads, plus 0.1 of effector_mid
TIP = 0.14
STYLE = Style(
    units_per_metre=1.0,
    page_camera={},
    dials={"edges": {"crease_deg": 35, "crease_why": "the build's shading angle"}},
)
TIMES = np.arange(1.0, 11.0)


def _plain_piece(holds, measured=None):
    return {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": 0.002}],
        "parts": [{"name": "top", "listed": "stock", "shape": "box", "on": None, "group": "g",
                   "role": "solid"}],
        "uses": [], "holds": holds, "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": measured or {},
        "view": {"nearest_m": 1.0, "why": "a test distance"},
    }


def _chain(root, bones, effector, group):
    return {"root": root, "bones": bones, "effector": effector, "effector_radius": RADIUS,
            "dofs": {b: "xyz" for b in bones}, "limits": {}, "effector_group": group}


@cache
def kit_pack() -> Pack:
    hold = {"id": "top_R", "use": "front", "side": "R", "surface": ["top"],
            "point": [0.0, 0.002, 0.0], "off": [0.0, 1.0, 0.0], "grip": False}
    plate = _plain_piece([hold])
    plate["uses"] = [{"id": "front", "point": [0.0, 0.0, 0.3], "facing": [0.0, 0.0, -1.0],
                      "posture": "stand"}]
    doll = _plain_piece([])
    doll["rig"] = {
        "armature": "TestRig",
        "bones": [{"name": "body", "parent": None, "length": 0.3},
                  {"name": "arm.R", "parent": "body", "length": 0.3},
                  {"name": "hand.R", "parent": "arm.R", "length": 0.3},
                  {"name": "leg.R", "parent": "body", "length": 0.4},
                  {"name": "foot.R", "parent": "leg.R", "length": 0.4}],
        "bind": {"top": {"bones": ["body"], "mode": "rigid"}},
        "contact_groups": {"hand.R": ["hand.R"], "foot.R": ["foot.R"]},
        "chains": {"arm.R": _chain("body", ["arm.R", "hand.R"], "hand.R", "hand.R"),
                   "leg.R": _chain("body", ["leg.R", "foot.R"], "foot.R", "foot.R")},
        "sit": {"below_seat": 0.08, "forward": 0.03},
        "mirror": {"pairs": [], "axis": "x"},
    }
    data = {"pack": "kit", "units_per_metre": 1.0, "pieces": {
        "Plate": plate, "Stick": _plain_piece([], {"tip_length": TIP}), "Doll": doll}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


def rest(x, y, z, spin=None):
    m = np.eye(4)
    if spin is not None:
        m[:3, :3] = spin
    m[:3, 3] = (x, y, z)
    return m


def turn_y(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # x -> (c, 0, -s), z -> (s, 0, c)


def turn_z(degrees):
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # x -> (c, s, 0)


def quat_y(degrees):
    half = np.radians(degrees) / 2
    return (0.0, float(np.sin(half)), 0.0, float(np.cos(half)))


RIG = RigData(
    order=("body", "arm.R", "hand.R", "leg.R", "foot.R"),
    parent={"body": None, "arm.R": "body", "hand.R": "arm.R", "leg.R": "body",
            "foot.R": "leg.R"},
    rest={"body": rest(0, 0, 0), "arm.R": rest(0.3, 0, 0), "hand.R": rest(0.6, 0, 0),
          "leg.R": rest(0, -0.4, 0), "foot.R": rest(0, -0.8, 0)},
    world=np.eye(4),
    effector_mid={"hand.R": np.array([0.2, 0.0, 0.0]), "foot.R": np.array([0.0, 0.0, 0.1])},
)


def make_ctx(*, at=(0.5, 0.0, 0.0), chain="arm.R", ref=None, tool=None, dangle=None,
             authoring="solved", offset=(0.0, 0.0, 0.0), key=5, root=(0.0, 0.0, 0.0),
             bones=None, carried=None, carried_spin=None, samples=True,
             dangle_who="hand.R", dangle_when=((1.0, 10.0),),
             quat=(0.0, 0.0, 0.0, 1.0), tool_scale=1.0) -> SceneCtx:
    """One clip, `reach`, frames 1..10, one target. The plate sits `at`; the body at `root`.

    With `tool`, the hand carries the stick, posed at the origin with `tool_scale` (the
    armature's scale, which a carried prop's pose takes)."""
    ref = Ref.parse(ref or "kit:Plate.top_R")
    holds = ()
    if dangle is not None:
        holds = (ClipHold(ref=ref, who=dangle_who, who_box=None, when=dangle_when, budget=None,
                          dangle=Dangle(dangle, "the feet hang"), cross=None),)
    props = {}
    if tool:
        props["stick"] = Prop("stick", "Stick", "kit", "hand.R", (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0, 1.0), ((1.0, 10.0),))
    if carried is not None:
        props["plate"] = Prop("plate", "Plate", "kit", "hand.R", (0.0, 0.0, 0.0),
                              (0.0, 0.0, 0.0, 1.0), ((1.0, 10.0),))
    gear = FloorGear({} if carried is not None else
                     {"Plate": Placement("Plate", "Plate", "kit", at, quat)})
    target = Target(key=key, chain=chain, ref=ref, offset=offset, why=None, baked=None,
                    tool="stick" if tool else None)
    clip = Clip(name="reach", frames=(1, 10), fps=24.0, authoring=authoring, seat=None,
                holds=holds, targets=() if authoring == "keyed" else (target,), floor_gear=gear,
                props=props, mirror_of=None)  # a keyed clip has no key targets
    n = len(TIMES)
    posed = {"body": np.tile(rest(*root), (n, 1, 1)),
             "arm.R": np.tile(rest(root[0] + SHOULDER_X, root[1], root[2]), (n, 1, 1)),
             "leg.R": np.tile(rest(root[0], root[1] + HIP_Y, root[2]), (n, 1, 1))}
    posed.update(bones or {})
    prop_poses = {}
    if carried is not None:
        prop_poses["plate"] = np.tile(rest(*carried, carried_spin), (n, 1, 1))
    if tool:
        prop_poses["stick"] = np.tile(rest(0.0, 0.0, 0.0, tool_scale * np.eye(3)), (n, 1, 1))
    clip_samples = ClipSamples(clip="reach", times=TIMES.copy(), groups={}, bones=posed,
                               props=prop_poses)
    return SceneCtx(style=STYLE, packs={"kit": kit_pack()}, clips={"reach": clip},
                    pieces={"kit:Plate": PieceGeom("Plate", {"top": plate()})},
                    samples={"reach": clip_samples} if samples else {},
                    rigs={"kit:Doll": RIG}, rig_of_clip={"reach": "kit:Doll"})


def matrix(position=(0.0, 0.0, 0.0), rot=None) -> np.ndarray:
    """A 4x4 matrix: the 3x3 `rot` (identity by default), then a move to `position`."""
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    m[:3, 3] = position
    return m
