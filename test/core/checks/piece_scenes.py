"""Scenes of one assembly piece, for the piece-geometry check tests. A helper module.

`make_ctx` builds a pack `kit` with one piece, `Thing`, from named part meshes. Every piece made
here declares the view at which one arcminute covers `TOUCH`, so its contact length is `TOUCH`
(about 21 mm). Its one listed row is 2 mm (`BASE`).
"""

import dataclasses
import json
import tempfile
from pathlib import Path

import numpy as np

from model_kit.core.manifest import Pack, Recipe, load_pack
from model_kit.core.scene import PieceGeom, SceneCtx, piece_key
from model_kit.core.tolerances import CROSS_OF_PART_THICKNESS
from model_kit.core.units import Style

from ..shapes import box, cylinder
from .sight_scenes import view_json

STYLE = Style(
    units_per_metre=1.0,
    page_camera={},
    dials={"edges": {"crease_deg": 35, "crease_why": "the build's shading angle"}},
)
BASE = 0.002
SLAB = 0.02  # the base slab of `post_on_base`: the thinner part of its pair
CROSS = CROSS_OF_PART_THICKNESS * SLAB  # the deepest the post may sink into that slab
TOUCH = 0.0213  # the widest gap that still touches: the pieces' contact length (seen floor)


def piece_json(parts, specs, hosts):
    """The pack.json entry of a piece. `specs` gives a part's `group`, `on`, `role` and `shape`.
    A decal part is printed on a host, so unless a spec says otherwise it sits on the first other
    part (or on itself, alone): the loader refuses a decal with no `on`."""
    rows = []
    for name in parts:
        spec = {"group": "g", "on": None, "role": "solid", "shape": "box", **specs.get(name, {})}
        if spec["shape"] == "decal" and "on" not in specs.get(name, {}):
            spec["on"] = next((other for other in parts if other != name), name)
        rows.append({"name": name, "listed": "stock", **spec})
    return {
        "frame": {"axes": "gltf", "origin": "centre of the base"},
        "recipe": {"kind": "assembly", "source": "test"},
        "listed": [{"id": "stock", "text": "Test stock", "source": "test", "min_thickness": BASE}],
        "parts": rows, "uses": [], "holds": [], "seats": [], "mounts": [], "done": "A test piece.",
        "size_class": "piece", "measured": {} if hosts is None else {"hosts": list(hosts)},
        "view": view_json(TOUCH),
    }


def make_ctx(parts, specs=None, *, hosts=None, also=None) -> SceneCtx:
    """A scene with the piece `kit:Thing` of the given part meshes (name -> (N, 3, 3)).

    `specs` maps a part name to the manifest fields it changes (`group`, `on`, `role`, `shape`);
    `hosts` is the piece's `measured.hosts`. `also` maps another piece's name to its part meshes:
    those pieces are branching ones, so a check meets a recipe kind it does not measure.
    """
    specs = specs or {}
    also = also or {}
    pieces = {"Thing": piece_json(parts, specs, hosts)}
    pieces.update({name: piece_json(p, {}, None) for name, p in also.items()})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps({"pack": "kit", "units_per_metre": 1.0, "pieces": pieces}))
        pack = load_pack(path, STYLE)
    branching = Recipe("branching", "grow.py")
    pack = Pack(name=pack.name, units_per_metre=pack.units_per_metre, spacing=pack.spacing,
                pieces={name: dataclasses.replace(piece, recipe=branching) if name in also
                        else piece for name, piece in pack.pieces.items()})
    geoms = {piece_key("kit", "Thing"): PieceGeom("Thing", dict(parts))}
    geoms.update({piece_key("kit", name): PieceGeom(name, dict(p)) for name, p in also.items()})
    return SceneCtx(style=STYLE, packs={"kit": pack}, clips={}, pieces=geoms, samples={},
                    rigs={}, rig_of_clip={})


def with_parts(ctx, **parts) -> SceneCtx:
    """The scene with these parts of `kit:Thing` replaced (or added), the rest unchanged."""
    key = piece_key("kit", "Thing")
    geom = PieceGeom("Thing", {**ctx.pieces[key].parts, **parts})
    return dataclasses.replace(ctx, pieces={**ctx.pieces, key: geom})


def post_on_base(sink=0.0, lift=0.0):
    """A base slab and a post standing on it. `sink` buries the post's foot in the base; `lift`
    raises the post off it."""
    return {
        "base": box((-0.1, 0.0, -0.1), (0.1, 0.02, 0.1)),
        "post": box((-0.02, 0.02 - sink + lift, -0.02), (0.02, 0.12 - sink + lift, 0.02)),
    }


def sheet_and_knob():
    """A base slab with a thin sheet (12 triangles, very many samples) and a knob (192
    triangles, few samples) on it. Fewest triangles is the most samples here."""
    return {"sheet": box((0.2, 0.0, -0.15), (0.5, 0.004, 0.15)),
            "knob": cylinder((-0.3, 0.0), 0.03, 0.0, 0.06, sides=48),
            "base": box((-0.6, -0.05, -0.2), (0.6, 0.0, 0.2))}


def strip_and_post():
    """A long thin strip and a round post on a base: the second piece of the cost guard."""
    return {"strip": box((-0.3, 0.0, 0.1), (0.3, 0.05, 0.102)),
            "post": cylinder((0.0, -0.1), 0.02, 0.0, 0.08, sides=32),
            "base": box((-0.4, -0.04, -0.2), (0.4, 0.0, 0.2))}


def chunk_and_block():
    """A small chunky part on a very thin film, and a mid-sized part on a thick slab. By its own
    spacing the chunk is the cheapest part (about 94 samples, the block about 257). Against its
    thin host the chunk needs the film's fine spacing (about 192,000), so the block is cheapest."""
    return {"film": box((-0.6, -0.002, -0.2), (-0.2, 0.0, 0.2)),
            "chunk": box((-0.5, 0.0, -0.05), (-0.4, 0.1, 0.05)),
            "slab": box((0.2, -0.1, -0.2), (0.6, 0.0, 0.2)),
            "block": box((0.3, 0.0, -0.05), (0.5, 0.06, 0.05))}


def two_pieces(sink: float = 0.0, decal: bool = False) -> SceneCtx:
    """A scene with two assembly pieces, `kit:Thing` and `kit:Other`, each `post_on_base(sink)`.
    With `decal`, each also has a part `label` of role `decal` (no crossings run measures it)."""
    parts = post_on_base(sink=sink)
    if decal:
        parts = {**parts, "label": box((-0.05, 0.02, -0.05), (0.05, 0.021, 0.05))}
    specs = {"label": {"role": "decal", "shape": "decal"}} if decal else {}
    names = ("Thing", "Other")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kit.pack.json"
        path.write_text(json.dumps({"pack": "kit", "units_per_metre": 1.0, "pieces": {
            name: piece_json(parts, specs, None) for name in names}}))
        pack = load_pack(path, STYLE)
    geoms = {piece_key("kit", name): PieceGeom(name, dict(parts)) for name in names}
    return SceneCtx(style=STYLE, packs={"kit": pack}, clips={}, pieces=geoms, samples={},
                    rigs={}, rig_of_clip={})


def caught(check, plant) -> bool:
    """The check, run on just the plant's scope, reports on the plant's item."""
    return any(f.item == plant.item for f in check.run(plant.ctx, only=plant.scope))


def accepted(check, unit):
    """The candidate the gate accepts from a unit: the first the check catches (None if none)."""
    return next((p for p in unit.candidates if caught(check, p)), None)


def tried_and_caught(check, ctx) -> list[str]:
    """The items of the plant units the gate would try on this scene (they apply, and the real run
    does not report the item already) whose first catching candidate exists. A unit tried and not
    caught raises, as the gate would."""
    failing = {f.item for f in check.run(ctx)}
    out = []
    for unit in check.plants(ctx):
        if unit.not_applicable or unit.item in failing:
            continue
        assert any(caught(check, p) for p in unit.candidates), f"not caught: {unit.item}"
        out.append(unit.item)
    return out


def chosen(check, ctx) -> list:
    """The plant the gate keeps for each unit it tries (it applies, and the real run does not
    report the item): the first candidate the check catches. A unit tried and not caught fails
    here, as it stops the gate."""
    failing = {f.item for f in check.run(ctx)}
    out = []
    for unit in check.plants(ctx):
        if unit.not_applicable or unit.item in failing:
            continue
        hit = accepted(check, unit)
        assert hit is not None, f"not caught: {unit.item}"
        out.append(hit)
    return out


def plants_caught(check, ctx) -> bool:
    """Every plant of the check that applies is caught as `run_gate` would catch it. A plant unit
    passes when its item already fails in the real run or one of its candidates is caught. At
    least one plant applies."""
    plants = [u for u in check.plants(ctx) if not u.not_applicable]
    assert plants
    failing = {f.item for f in check.run(ctx)}
    return all(u.item in failing or any(caught(check, p) for p in u.candidates) for u in plants)


def refuses_the_core(monkeypatch, module, *names):
    """Make each named function of the check module raise: a plant built from the geometry alone
    never calls the check's own verdict."""
    def refuse(*args, **kwargs):
        raise AssertionError("a plant called the check's own core")
    for name in names:
        monkeypatch.setattr(module, name, refuse)


SLIVER = 2e-5  # the planted sliver: a shell sunk 0.02 mm inside another shell of the same part


def sunk_panel_monitor(gap=SLIVER):
    """A monitor whose `display` is one mesh of two shells: a slab and a screen panel whose back
    sinks `gap` into the slab's front (a converted monitor model can carry this fault). Its
    thickness at its centre reads that gap. The `stand` holds it from behind."""
    slab = box((-0.27, 0.10, -0.075), (0.27, 0.36, -0.033))
    panel = box((-0.25, 0.12, -0.033 - gap), (0.25, 0.34, -0.033 + 0.0007))
    return ({"display": np.concatenate([slab, panel]),
             "stand": box((-0.05, 0.0, -0.10), (0.05, 0.30, -0.075))},
            {"display": {"on": "stand", "group": "a"}, "stand": {"group": "b"}})


def sunk_sheet_base(gap=SLIVER):
    """`post_on_base` whose `base` is one mesh of two shells: the slab and a sheet on its top
    whose bottom sinks `gap` into it, so the base's thickness at its centre reads that gap."""
    parts = post_on_base()
    sheet = box((-0.09, 0.02 - gap, -0.09), (0.09, 0.0205, 0.09))
    return ({**parts, "base": np.concatenate([parts["base"], sheet])},
            {"post": {"on": "base", "group": "b"}, "base": {"group": "a"}})

