"""A scene for the manifest checks (frames, sides, mirror-pair, done, no-fallback). A helper
module.

`make_ctx` builds the pack `kit`: a `Deck` (a plate with a use station in front, a right hold, a
left hold, a both-sides hold, a seat and a mount, all on the plate's top) and a rigged `Doll`
that stands at the station. Two clips use them: `play` (the doll's hands on the deck's two holds
and on the both-sides hold) and `play_L`, its mirror. Every point sits on the deck's top. The
pack has one listed row of 0.3 mm (`BASE`), and every piece declares the view at which one
arcminute covers `LIMIT`, so `LIMIT` is how far a point may sit from its surface.

The user stands at z 0.3 and looks toward -z, so the user's right is +x (`facing x up`).
"""

import copy
import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np

from model_kit.core.manifest import (
    Clip,
    ClipHold,
    FloorGear,
    Layout,
    LayoutPlacement,
    Pack,
    Placement,
    Ref,
    Target,
    load_pack,
)
from model_kit.core.scene import ClipSamples, PieceGeom, SceneCtx
from model_kit.core.units import Style

from ..shapes import box
from .piece_scenes import plants_caught as piece_plants_caught
from .sight_scenes import view_json

STYLE = Style(
    units_per_metre=1.0,
    page_camera={},
    dials={"edges": {"crease_deg": 35, "crease_why": "the build's shading angle"}},
)
BASE = 0.0003
LIMIT = 0.01  # every piece's contact length (its view's seen floor)
TIMES = np.arange(1.0, 11.0)
ROOT = (0.0, 0.0, 0.3)  # the doll's root, where the user stands
TOP = 0.02  # the deck's top
HOLD_Z = 0.04  # the holds sit in front of the deck's middle, so a y/z swap moves each of them

_RIG = {
    "armature": "DollRig",
    "bones": [{"name": "root", "parent": None, "length": 0.1},
              {"name": "body", "parent": "root", "length": 0.3},
              {"name": "arm.R", "parent": "body", "length": 0.1},
              {"name": "hand.R", "parent": "arm.R", "length": 0.05},
              {"name": "arm.L", "parent": "body", "length": 0.1},
              {"name": "hand.L", "parent": "arm.L", "length": 0.05}],
    "bind": {"torso": {"bones": ["body"], "mode": "rigid"}},
    "contact_groups": {"hand.R": ["hand.R"], "hand.L": ["hand.L"]},
    "chains": {
        name: {"root": "body", "bones": [arm, hand], "effector": hand, "effector_radius": 0.0586,
               "dofs": {arm: "xyz", hand: "xyz"}, "limits": {}, "effector_group": hand}
        for name, arm, hand in (("arm.R", "arm.R", "hand.R"), ("arm.L", "arm.L", "hand.L"))},
    "sit": {"below_seat": 0.08, "forward": 0.03},
    "mirror": {"pairs": [["arm.R", "arm.L"], ["hand.R", "hand.L"]], "axis": "x"},
}


def _piece(parts, **fields):
    piece = {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": BASE}],
        "parts": [{"name": n, "listed": "stock", "shape": "box", "on": None, "group": "g",
                   "role": "solid"} for n in parts],
        "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {}, "view": view_json(LIMIT),
    }
    return {**piece, **fields}


def _hold(id, side, x, **fields):
    return {"id": id, "use": "front", "side": side, "surface": ["top"], "point": [x, TOP, HOLD_Z],
            "off": [0.0, 1.0, 0.0], "grip": False, **fields}


def pack_json() -> dict:
    deck = _piece(
        ["top"],
        uses=[{"id": "front", "point": [0.0, 0.0, 0.3], "facing": [0.0, 0.0, -1.0],
               "posture": "stand"}],
        holds=[_hold("hold_R", "R", 0.1), _hold("hold_L", "L", -0.1), _hold("hold_B", "both", 0.0)],
        seats=[{"id": "seat_1", "use": "front", "point": [0.0, TOP, 0.05],
                "facing": [0.0, 0.0, -1.0], "surface": ["top"]}],
        mounts=[{"id": "jack", "kind": "power", "point": [0.15, TOP, 0.08], "axis": [0.0, 1.0, 0.0],
                 "takes": "lead"}],
    )
    doll = _piece(["torso"], rig=_RIG, size_class="hero")
    return {"pack": "kit", "units_per_metre": 1.0, "pieces": {"Deck": deck, "Doll": doll}}


def load(data: dict) -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


def _hold_of(ref, who) -> ClipHold:
    return ClipHold(ref=Ref.parse(ref), who=who, who_box=None, when=((1.0, 10.0),), budget=None,
                    dangle=None, cross=None)


def _target(chain, ref) -> Target:
    return Target(key=5, chain=chain, ref=Ref.parse(ref), offset=(0.0, 0.0, 0.0), why=None,
                  baked=None, tool=None)


def _clip(name, holds, targets, mirror_of=None, piece="Deck") -> Clip:
    gear = FloorGear({piece: Placement(piece, piece, "kit", (0.0, 0.0, 0.0),
                                       (0.0, 0.0, 0.0, 1.0))})
    return Clip(name=name, frames=(1, 10), fps=24.0, authoring="solved", seat=None,
                holds=tuple(holds), targets=tuple(targets), floor_gear=gear, props={},
                mirror_of=mirror_of)


def clips() -> dict[str, Clip]:
    play = _clip(
        "play",
        [_hold_of("kit:Deck.hold_R", "hand.R"), _hold_of("kit:Deck.hold_L", "hand.L"),
         _hold_of("kit:Deck.hold_B", "hand.R")],
        [_target("arm.R", "kit:Deck.hold_R"), _target("arm.L", "kit:Deck.hold_L")])
    mirror = _clip(
        "play_L",
        [_hold_of("kit:Deck.hold_L", "hand.L"), _hold_of("kit:Deck.hold_R", "hand.R"),
         _hold_of("kit:Deck.hold_B", "hand.L")],
        [_target("arm.L", "kit:Deck.hold_R"), _target("arm.R", "kit:Deck.hold_L")],
        mirror_of="play")
    return {"play": play, "play_L": mirror}


def bench_clips() -> dict[str, Clip]:
    """The deck's clips again, on a second station: `rest` and its mirror `rest_L` on `Bench`."""
    rest = _clip(
        "rest",
        [_hold_of("kit:Bench.hold_R", "hand.R"), _hold_of("kit:Bench.hold_L", "hand.L"),
         _hold_of("kit:Bench.hold_B", "hand.R")],
        [_target("arm.R", "kit:Bench.hold_R"), _target("arm.L", "kit:Bench.hold_L")],
        piece="Bench")
    mirror = _clip(
        "rest_L",
        [_hold_of("kit:Bench.hold_L", "hand.L"), _hold_of("kit:Bench.hold_R", "hand.R"),
         _hold_of("kit:Bench.hold_B", "hand.L")],
        [_target("arm.L", "kit:Bench.hold_R"), _target("arm.R", "kit:Bench.hold_L")],
        mirror_of="rest", piece="Bench")
    return {"rest": rest, "rest_L": mirror}


def with_bench() -> SceneCtx:
    """`make_ctx` and a second station, `Bench` (the deck's twin), with its clips `rest` and
    `rest_L`, so a check meets two pieces."""
    data = pack_json()
    data["pieces"]["Bench"] = copy.deepcopy(data["pieces"]["Deck"])
    ctx = make_ctx()
    more = bench_clips()
    return dataclasses.replace(
        ctx, packs={"kit": load(data)}, clips={**ctx.clips, **more},
        pieces={**ctx.pieces, "kit:Bench": PieceGeom("Bench", deck_geometry())},
        samples={**ctx.samples, **{n: samples(n, 0.1) for n in more}},
        rig_of_clip={**ctx.rig_of_clip, "rest": "kit:Doll", "rest_L": "kit:Doll"})


def _mat(x, y, z) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def samples(name: str, right_x: float = 0.1) -> ClipSamples:
    """The doll at its station: the right hand at x `right_x`, the left hand opposite, 4 cm
    above the deck. The root stays where the user stands."""
    n = len(TIMES)

    def group(x):
        return np.tile(np.array([[x - 0.01, 0.04, 0.0], [x + 0.01, 0.04, 0.0]]), (n, 1, 1))

    bones = {b: np.tile(_mat(*ROOT), (n, 1, 1)) for b in ("root", "body")}
    return ClipSamples(clip=name, times=TIMES.copy(),
                       groups={"hand.R": group(right_x), "hand.L": group(-right_x)},
                       bones=bones, props={})


def deck_geometry() -> dict[str, np.ndarray]:
    return {"top": box((-0.2, 0.0, -0.1), (0.2, TOP, 0.1))}


def make_ctx(*, sources=None, folder="") -> SceneCtx:
    pack = load(pack_json())
    cs = clips()
    layout = Layout(
        name="floor", frame="gltf",
        placements={"Deck_1": LayoutPlacement("Deck_1", "kit", "Deck", (0.0, 0.0, 0.0),
                                              (0.0, 0.0, 0.0, 1.0))},
        stands={})
    return SceneCtx(
        style=STYLE, packs={"kit": pack}, clips=cs,
        pieces={"kit:Deck": PieceGeom("Deck", deck_geometry())},
        samples={n: samples(n, 0.1) for n in cs}, rigs={},
        rig_of_clip={"play": "kit:Doll", "play_L": "kit:Doll"},
        layouts={"floor": layout}, sources=dict(sources or {}), folder=folder)


# --- changing a copy -------------------------------------------------------------------------


def with_piece(ctx: SceneCtx, name: str, **fields) -> SceneCtx:
    """The scene with these fields of `kit:<name>` replaced."""
    pack = ctx.packs["kit"]
    piece = dataclasses.replace(pack.pieces[name], **fields)
    pack = Pack(name=pack.name, units_per_metre=pack.units_per_metre, spacing=pack.spacing,
                pieces={**pack.pieces, name: piece})
    return dataclasses.replace(ctx, packs={"kit": pack})


def with_hold(ctx: SceneCtx, id: str, **fields) -> SceneCtx:
    """The scene with these fields of the deck's hold `id` replaced."""
    holds = tuple(dataclasses.replace(h, **fields) if h.id == id else h
                  for h in ctx.packs["kit"].pieces["Deck"].holds)
    return with_piece(ctx, "Deck", holds=holds)


def with_clip(ctx: SceneCtx, name: str, **fields) -> SceneCtx:
    clip = dataclasses.replace(ctx.clips[name], **fields)
    return dataclasses.replace(ctx, clips={**ctx.clips, name: clip})


def with_clip_hold(ctx: SceneCtx, clip: str, index: int, **fields) -> SceneCtx:
    holds = list(ctx.clips[clip].holds)
    holds[index] = dataclasses.replace(holds[index], **fields)
    return with_clip(ctx, clip, holds=tuple(holds))


def plants_caught(check, ctx) -> bool:
    """Every plant unit of the check that applies is caught (see `piece_scenes.plants_caught`)."""
    return piece_plants_caught(check, ctx)
