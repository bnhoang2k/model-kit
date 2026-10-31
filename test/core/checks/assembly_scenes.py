"""A scene of pieces that meet, for the assembly check tests (receiver, keep-off, connectors,
export hygiene). A helper module.

The pack `kit` has three assembly pieces, each declaring the view at which one arcminute covers
`END`, so `END` is the contact length of every piece (how far an end may sit from its point). Its
one listed row is 0.3 mm (`BASE`):

- `Rack` (the host): a `frame` with a blind pit (`PIT`) cut in its top at the jack, a rest mount
  `cup` that takes a `pin` of the `Lamp` only, a power mount `jack`, both cut in the `frame`, and
  one keep-off point, `knot`, well clear of both mounts.
- `Lamp` (the guest): a `pin` that stands on the cup's point, and a shade above it.
- `Lead`: a cable part `cable` (0.3 long) and a plug at the cable's end, a `plug` cap and a `pin`
  hanging from it, with the connector `run`, capped at 0.35. One end plugs into the rack's jack
  through the layout `room`, the cap on the rack's top and the pin in the pit; the other into the
  lead's own mount `tip`.

The layout `room` places the rack at the origin, the lamp on the rack's cup, and the lead so its
cable's end lies on the rack's jack.
The packed file's node table (`with_nodes`) holds each piece root, then its parts as mesh nodes
named `<piece>/<part>`, with the roots on a grid of `SPACING` (or, with no spacing, at the
layout's positions).
"""

import copy
import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np

from model_kit.core.glb import GlbNode, node_name
from model_kit.core.manifest import (
    Clip,
    FloorGear,
    Layout,
    LayoutPlacement,
    Pack,
    Prop,
    Ref,
    load_pack,
)
from model_kit.core.scene import PieceGeom, SceneCtx
from model_kit.core.tolerances import (
    FLOAT32_EPS,
    NODE_FLOAT32_ULPS,
    NODE_OF_PIECE_SIZE,
)
from model_kit.core.units import Style

from ..shapes import box, cylinder, pitted_box
from .piece_scenes import plants_caught as piece_plants_caught
from .sight_scenes import view_json

STYLE = Style(
    units_per_metre=1.0,
    page_camera={},
    dials={"edges": {"crease_deg": 35, "crease_why": "the build's shading angle"}},
)
BASE = 0.0003
END = 0.02  # how far an end may sit from its point: the pieces' contact length (seen floor)
SPACING = 1.0
CUP = (0.1, 0.1, 0.0)  # the rack's cup mount, in the rack's frame
JACK = (-0.1, 0.1, 0.0)
KNOT = (0.0, 0.1, 0.08)  # the rack's keep-off point
KNOT_RADIUS = 0.03
CAP = 0.35
CABLE_LENGTH = 0.3
LEAD_AT = (JACK[0] + CABLE_LENGTH / 2, JACK[1], JACK[2])  # the cable's far end lands on the jack
LAMP_AT = CUP
PIT_HALF = 0.007  # the jack's pit, centre to wall: the pin (0.005 half-width) has 2 mm to spare
PIT_FLOOR = 0.055
Q_ID = (0.0, 0.0, 0.0, 1.0)
Q_Y90 = (0.0, float(np.sin(np.pi / 4)), 0.0, float(np.cos(np.pi / 4)))  # a quarter turn about y


def _piece(parts, **fields):
    piece = {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": BASE}],
        "parts": [{"name": n, "listed": "stock", "shape": shape, "on": on, "group": "g",
                   "role": "solid"} for n, shape, on in ((*x, None) if len(x) == 2 else x
                                                         for x in parts)],
        "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {}, "view": view_json(END),
    }
    return {**piece, **fields}


def pack_json(spacing=None, base=BASE) -> dict:
    rack = _piece(
        [("frame", "box")],
        mounts=[{"id": "cup", "kind": "rest", "point": list(CUP), "axis": [0.0, 1.0, 0.0],
                 "takes": "pin", "fits": ["Lamp"], "part": "frame"},
                {"id": "jack", "kind": "power", "point": list(JACK), "axis": [0.0, 1.0, 0.0],
                 "takes": "plug", "part": "frame"}],
        keep_off=[{"id": "knot", "point": list(KNOT), "radius": KNOT_RADIUS,
                   "why": "a brace node the mounts keep off"}])
    lamp = _piece([("pin", "box"), ("shade", "box")])
    lead = _piece(
        [("cable", "cable"), ("plug", "box"), ("pin", "box", "plug")],
        mounts=[{"id": "tip", "kind": "signal", "point": [CABLE_LENGTH / 2, 0.0, 0.0],
                 "axis": [1.0, 0.0, 0.0], "takes": "cable"}],
        connectors=[{"id": "run", "part": "cable", "cap": CAP, "why": "a test lead",
                     "ends": [{"point": list(JACK), "layout": "room", "host": "kit:rack.jack",
                               "plug": "plug"},
                              {"point": [CABLE_LENGTH / 2, 0.0, 0.0], "host": "self.tip"}]}])
    data = {"pack": "kit", "units_per_metre": 1.0,
            "pieces": {"Rack": rack, "Lamp": lamp, "Lead": lead}}
    if spacing is not None:
        data["spacing"] = spacing
    for piece in data["pieces"].values():
        for row in piece["listed"]:
            row["min_thickness"] = base
    return data


def load(data: dict) -> Pack:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps(data))
        return load_pack(path, STYLE)


def geometry() -> dict[str, PieceGeom]:
    half = CABLE_LENGTH / 2
    return {
        "kit:Rack": PieceGeom("Rack", {"frame": pitted_box(
            (-0.2, 0.0, -0.1), (0.2, 0.1, 0.1),
            [((JACK[0] - PIT_HALF, JACK[2] - PIT_HALF), (JACK[0] + PIT_HALF, JACK[2] + PIT_HALF))],
            PIT_FLOOR)}),
        "kit:Lamp": PieceGeom("Lamp", {"pin": box((-0.01, 0.0, -0.01), (0.01, 0.05, 0.01)),
                                       "shade": box((-0.05, 0.05, -0.05), (0.05, 0.1, 0.05))}),
        "kit:Lead": PieceGeom("Lead", {
            "cable": box((-half, -0.002, -0.002), (half, 0.002, 0.002)),
            # a cap on the rack's top at the cable's end, and a pin hanging from it into the jack's
            # pit, 5 mm clear of its floor
            "plug": box((-half - 0.015, 0.0, -0.015), (-half + 0.015, 0.02, 0.015)),
            "pin": box((-half - 0.005, -0.04, -0.005), (-half + 0.005, 0.0, 0.005))}),
    }


def layout(*, rack_quaternion=Q_ID, lamp_at=LAMP_AT) -> Layout:
    return Layout(name="room", frame="gltf", stands={}, placements={
        "rack": LayoutPlacement("rack", "kit", "Rack", (0.0, 0.0, 0.0), rack_quaternion),
        "lamp": LayoutPlacement("lamp", "kit", "Lamp", lamp_at, Q_ID,
                                on=Ref("kit", "rack", "cup")),
        "lead": LayoutPlacement("lead", "kit", "Lead", LEAD_AT, Q_ID),
    })


def make_ctx(*, spacing=None, pieces=True, base=BASE) -> SceneCtx:
    return SceneCtx(style=STYLE, packs={"kit": load(pack_json(spacing, base))}, clips={},
                    pieces=geometry() if pieces else {}, samples={}, rigs={}, rig_of_clip={},
                    layouts={"room": layout()})


def node_tol(ctx: SceneCtx, piece: str) -> float:
    """How far a root may sit from its place: `NODE_OF_PIECE_SIZE` times the piece's size, plus
    the float32 rounding of the root's distance from the origin when the scene has its node."""
    every = np.concatenate([t.reshape(-1, 3) for t in ctx.pieces[f"kit:{piece}"].parts.values()])
    limit = NODE_OF_PIECE_SIZE * float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    roots = [n for n in ctx.nodes.get("kit", ()) if n.name == piece and n.parent is None]
    far = float(np.linalg.norm(roots[0].translation)) if roots else 0.0
    return limit + NODE_FLOAT32_ULPS * FLOAT32_EPS * far


def node(name, parent=None, at=(0.0, 0.0, 0.0), mesh=None) -> GlbNode:
    return GlbNode(name, parent, at, (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0), mesh)


def table(ctx: SceneCtx, pack="kit") -> list[GlbNode]:
    """The node table a clean export of the pack would hold: each piece root at its slot, then
    its parts as mesh nodes named `<piece>/<part>`."""
    p = ctx.packs[pack]
    placed = {pl.piece: pl.position for lay in ctx.layouts.values()
              for pl in lay.placements.values() if pl.pack == pack}
    out = []
    for i, (name, piece) in enumerate(p.pieces.items()):
        at = (i * p.spacing, 0.0, 0.0) if p.spacing is not None else placed.get(name, (0, 0, 0))
        out.append(node(name, None, at))
        out += [node(node_name(name, part.name), name, mesh=node_name(name, part.name))
                for part in piece.parts]
    return out


def with_nodes(ctx: SceneCtx, nodes=None, pack="kit") -> SceneCtx:
    return dataclasses.replace(ctx, nodes={**ctx.nodes, pack: table(ctx, pack) if nodes is None
                                           else nodes})


def carrying(ctx: SceneCtx, piece="Lamp") -> SceneCtx:
    """The scene with a clip `carry` whose doll's hand carries `piece` as a prop."""
    clip = Clip(name="carry", frames=(1, 4), fps=24.0, authoring="keyed", seat=None, holds=(),
                targets=(), floor_gear=FloorGear({}), mirror_of=None,
                props={"held": Prop("held", piece, "kit", "hand.R", (0.0, 0.0, 0.0), Q_ID,
                                    ((1.0, 4.0),))})
    return dataclasses.replace(ctx, clips={"carry": clip})


def with_piece(ctx: SceneCtx, name: str, **fields) -> SceneCtx:
    """The scene with these fields of `kit:<name>` replaced."""
    pack = ctx.packs["kit"]
    piece = dataclasses.replace(pack.pieces[name], **fields)
    pack = Pack(name=pack.name, units_per_metre=pack.units_per_metre, spacing=pack.spacing,
                pieces={**pack.pieces, name: piece})
    return dataclasses.replace(ctx, packs={"kit": pack})


def with_mount(ctx: SceneCtx, piece: str, mount: str, **fields) -> SceneCtx:
    mounts = tuple(dataclasses.replace(m, **fields) if m.id == mount else m
                   for m in ctx.packs["kit"].pieces[piece].mounts)
    return with_piece(ctx, piece, mounts=mounts)


def with_placement(ctx: SceneCtx, id: str, **fields) -> SceneCtx:
    lay = ctx.layouts["room"]
    placed = dataclasses.replace(lay.placements[id], **fields)
    return dataclasses.replace(ctx, layouts={"room": dataclasses.replace(
        lay, placements={**lay.placements, id: placed})})


def with_geometry(ctx: SceneCtx, key: str, **parts) -> SceneCtx:
    geom = ctx.pieces[key]
    return dataclasses.replace(ctx, pieces={**ctx.pieces, key: PieceGeom(
        geom.name, {**geom.parts, **parts})})


def plants_caught(check, ctx) -> bool:
    """Every plant of the check that applies is caught (see `piece_scenes.plants_caught`)."""
    return piece_plants_caught(check, ctx)


def two_guests():
    """The Rack with a second cup and a second lamp on it: a second guest and mount to plant."""
    data = pack_json()
    data["pieces"]["Rack"]["mounts"].append(
        {"id": "cup2", "kind": "rest", "point": [-0.1, 0.1, 0.05], "axis": [0.0, 1.0, 0.0],
         "takes": "pin", "part": "frame"})
    data["pieces"]["Lamp2"] = copy.deepcopy(data["pieces"]["Lamp"])
    ctx = make_ctx()
    room = ctx.layouts["room"]
    lamp2 = LayoutPlacement("lamp2", "kit", "Lamp2", (-0.1, 0.1, 0.05), Q_ID,
                            on=Ref("kit", "rack", "cup2"))
    geoms = {**ctx.pieces, "kit:Lamp2": ctx.pieces["kit:Lamp"]}
    return dataclasses.replace(
        ctx, packs={"kit": load(data)}, pieces=geoms,
        layouts={"room": dataclasses.replace(room, placements={**room.placements,
                                                               "lamp2": lamp2})})


PERCH = (0.35, 0.1, 0.0)  # the rack's perch mount, on the top of its `deck`, in the rack's frame


def with_stool():
    """The scene with a second kind of guest: a `Stool` whose pin is round (radius 0.03, a lathe
    shape, not the lamp's 0.02 square box) and 0.08 tall, on a rest mount `perch` that names the
    rack's `deck`, a second part 0.1 away from the `frame` the lamp's cup names. The deck is in the
    rack's manifest as well as its geometry."""
    data = pack_json()
    rack = data["pieces"]["Rack"]
    rack["parts"].append({"name": "deck", "listed": "stock", "shape": "box", "on": None,
                          "group": "g", "role": "solid"})
    rack["mounts"].append({"id": "perch", "kind": "rest", "point": list(PERCH),
                           "axis": [0.0, 1.0, 0.0], "takes": "pin", "fits": ["Stool"],
                           "part": "deck"})
    data["pieces"]["Stool"] = _piece([("pin", "lathe"), ("seat", "box")])
    ctx = make_ctx()
    room = ctx.layouts["room"]
    stool = LayoutPlacement("stool", "kit", "Stool", PERCH, Q_ID, on=Ref("kit", "rack", "perch"))
    rack_geom = ctx.pieces["kit:Rack"]
    geoms = {**ctx.pieces,
             "kit:Rack": PieceGeom("Rack", {**rack_geom.parts,
                                            "deck": box((0.3, 0.0, -0.1), (0.4, 0.1, 0.1))}),
             "kit:Stool": PieceGeom("Stool", {
                 "pin": cylinder((0.0, 0.0), 0.03, 0.0, 0.08),
                 "seat": box((-0.06, 0.08, -0.06), (0.06, 0.1, 0.06))})}
    return dataclasses.replace(
        ctx, packs={"kit": load(data)}, pieces=geoms,
        layouts={"room": dataclasses.replace(room, placements={**room.placements,
                                                               "stool": stool})})
