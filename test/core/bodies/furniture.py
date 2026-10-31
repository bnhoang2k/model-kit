"""Test furniture for the bodies: a chair, a bench, a table and a panel.

All sizes are design choices for tests, in metres (units_per_metre is 1.0), in each piece's own
frame: the floor is y 0, a sitter faces +z. The chair and bench each have a seat whose surface
parts are boxes. The panel holds the parts that the bearing, fastener, retainer and marking checks
plant on: a board, a plate bolted to it, two bolts (`fastener`), a bar that holds the plate
(`retains`) and a decal printed on the plate. Every piece is an assembly piece with
`measured.export.position_bits` null (a float32 file).
"""

from __future__ import annotations

import itertools

import numpy as np

from model_kit.core.forms import decal_grid
from model_kit.core.manifest.forms import DecalForm

from ..shapes import box
from .build import PieceSpec

CHAIR_SEAT_TOP = 0.43
BENCH_SEAT_TOP = 0.45
# A seat's front edge, along +z. The sitter's pelvis is 0.03 ahead of the seat point, which is
# the sit spec's `forward`; the seat runs 0.14 behind that and 0.36 ahead of it.
SEAT_Z = (-0.14, 0.36)
BENCH_SEAT_Z = (-0.14, 0.30)

_PLATE_FRONT = 0.016  # the panel plate's front face, where the decal lies
_DECAL_AT = (0.0, 0.35, _PLATE_FRONT)
_DECAL_HALF = (0.05, 0.03)
# A printed film is microns thick (about 4 microns of dried pad-print ink).
_INK_FILM_M = 4e-6


def _piece(frame_origin, listed, parts, done, *, uses=(), seats=()):
    return {
        "frame": {"axes": "gltf", "origin": frame_origin},
        "recipe": {"kind": "assembly", "source": "test/core/bodies/furniture.py"},
        "listed": [{"id": i, "text": t, "source": "test furniture, a design choice",
                    "min_thickness": m} for i, t, m in listed],
        "parts": parts, "uses": list(uses), "holds": [], "seats": list(seats), "mounts": [],
        "done": done, "size_class": "piece",
        "measured": {"export": {"position_bits": None}},
    }


def _part(name, listed, on, group, shape="box", role="solid", **extra):
    return {"name": name, "listed": listed, "shape": shape, "on": on, "group": group,
            "role": role, **extra}


def _legs(names, spots, top, half):
    return {n: box((x - half, 0.0, z - half), (x + half, top, z + half))
            for n, (x, z) in zip(names, spots, strict=True)}


def _sit(seat_top):
    use = {"id": "sit", "point": [0.0, 0.0, 0.0], "facing": [0.0, 0.0, 1.0], "posture": "sit"}
    seat = {"id": "seat_1", "use": "sit", "point": [0.0, seat_top, -0.03],
            "facing": [0.0, 0.0, 1.0], "surface": ["seat"]}
    return [use], [seat]


def chair() -> PieceSpec:
    z0, z1 = SEAT_Z
    top = CHAIR_SEAT_TOP
    names = ["leg_FL", "leg_FR", "leg_BL", "leg_BR"]
    spots = [(0.2, z1 - 0.04), (-0.2, z1 - 0.04), (0.2, z0 + 0.04), (-0.2, z0 + 0.04)]
    parts = [_part("seat", "board", "leg_FL", "seat"), _part("back", "back_board", "seat", "back"),
             *[_part(n, "leg", None, "legs") for n in names]]
    uses, seats = _sit(top)
    raw = _piece("floor, under the seat centre; the sitter faces +z",
                 [("board", "Seat board, 20 mm", 0.02), ("back_board", "Back board, 30 mm", 0.03),
                  ("leg", "Leg, 40 mm square", 0.04)],
                 parts, "A plain chair: a seat 0.43 high, a back and four square legs.",
                 uses=uses, seats=seats)

    def geometry(lift):
        return {"seat": box((-0.22, top - 0.02, z0), (0.22, top, z1)),
                "back": box((-0.22, top - 0.01, z0 - 0.02), (0.22, 0.9, z0 + 0.01)),
                **_legs(names, spots, top - 0.01, 0.02)}
    return PieceSpec(raw, geometry)


def bench() -> PieceSpec:
    z0, z1 = BENCH_SEAT_Z
    top = BENCH_SEAT_TOP
    parts = [_part("seat", "board", "leg_L", "seat"), _part("leg_L", "slab", None, "legs"),
             _part("leg_R", "slab", None, "legs")]
    uses, seats = _sit(top)
    raw = _piece("floor, under the seat centre; the sitter faces +z",
                 [("board", "Seat board, 20 mm", 0.02), ("slab", "Slab leg, 40 mm", 0.04)],
                 parts, "A plain bench: a long seat 0.45 high on two slab legs.",
                 uses=uses, seats=seats)

    def geometry(lift):
        return {"seat": box((-0.5, top - 0.02, z0), (0.5, top, z1)),
                "leg_L": box((0.42, 0.0, z0 + 0.02), (0.46, top - 0.01, z1 - 0.02)),
                "leg_R": box((-0.46, 0.0, z0 + 0.02), (-0.42, top - 0.01, z1 - 0.02))}
    return PieceSpec(raw, geometry)


def table() -> PieceSpec:
    names = ["leg_FL", "leg_FR", "leg_BL", "leg_BR"]
    spots = [(0.45, 0.25), (-0.45, 0.25), (0.45, -0.25), (-0.45, -0.25)]
    parts = [_part("top", "board", "leg_FL", "top"),
             *[_part(n, "leg", None, "legs") for n in names]]
    raw = _piece("floor, under the top centre",
                 [("board", "Top board, 30 mm", 0.03), ("leg", "Leg, 50 mm square", 0.05)],
                 parts, "A plain table: a top 0.75 high on four square legs.")

    def geometry(lift):
        return {"top": box((-0.5, 0.72, -0.3), (0.5, 0.75, 0.3)),
                **_legs(names, spots, 0.73, 0.025)}
    return PieceSpec(raw, geometry)


def _decal_tris(lift: float) -> np.ndarray:
    """The decal's oval, laid on the plate's front face and lifted off it by `lift`."""
    form = DecalForm(half=_DECAL_HALF, segments=24, rings=4, at=_DECAL_AT)
    points, faces = decal_grid(form)
    verts = np.column_stack([_DECAL_AT[0] + points[:, 0], _DECAL_AT[1] + points[:, 1],
                             np.full(len(points), _PLATE_FRONT + lift)])
    tris = []
    for f in faces:
        tris += [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
    return verts[np.array(tris)]


def panel() -> PieceSpec:
    """A board on the floor, a plate on its front, two bolts through the plate into the board, a
    bar that holds the plate's lower edge and a decal on the plate."""
    decal = {"half": list(_DECAL_HALF), "segments": 24, "rings": 4, "at": list(_DECAL_AT),
             "along": [0.0, 0.0, -1.0],
             "why": "a mark printed on the plate, cast on from the front"}
    parts = [_part("board", "board", None, "board"), _part("plate", "plate", "board", "plate"),
             _part("bolt_1", "bolt", "board", "bolts", fastener=True),
             _part("bolt_2", "bolt", "board", "bolts", fastener=True),
             _part("bar", "bar", "board", "bar", retains=["plate"]),
             _part("mark", "ink", "plate", "mark", shape="decal", role="decal", form=decal)]
    raw = _piece("floor, under the board centre; the front is +z",
                 [("board", "Board, 20 mm", 0.02), ("plate", "Plate, 6 mm", 0.006),
                  ("bolt", "Bolt, 12 mm square", 0.012), ("bar", "Retaining bar, 20 mm", 0.02),
                  ("ink", "Dried pad-print ink film, about 4 microns",
                   _INK_FILM_M)],
                 parts, "A board with a bolted plate, a bar that holds it and a printed mark.")
    raw["listed"][4]["finish"] = "print"  # the ink row: its part is a decal

    def geometry(lift):
        bolts = {f"bolt_{i}": box((x - 0.006, 0.444, -0.005), (x + 0.006, 0.456, _PLATE_FRONT + 0.006))
                 for i, x in ((1, 0.12), (2, -0.12))}
        return {"board": box((-0.25, 0.0, -0.01), (0.25, 0.6, 0.01)),
                "plate": box((-0.15, 0.2, 0.005), (0.15, 0.5, _PLATE_FRONT)),
                **bolts,
                # The bar ends at the plate's lower edge (y 0.20) and shares no face with it: two
                # parts sharing a face is a defect at any viewing distance (`coplanar`).
                "bar": box((-0.17, 0.17, 0.005), (0.17, 0.20, 0.03)),
                "mark": _decal_tris(lift)}
    return PieceSpec(raw, geometry)


BUILDERS = {"Chair": chair, "Bench": bench, "Table": table, "Panel": panel}


def furniture(*names: str) -> dict[str, PieceSpec]:
    """Fresh furniture specs by piece name."""
    return {n: BUILDERS[n]() for n in names}


# --- the marking fixtures (check markings) ----------------------------------------------------
# Rows of these pieces list a 0.5 mm minimum, so the thickness base is 0.0005 and the check's
# flush limit is 5.3 mm: the badge stands 8 mm off its board and the plate 0.2 mm.

_THIN_M = 0.0005
_LABEL_FRONT = 0.01  # the label board's front face, where the legend lies
_PLATE_PROUD = 0.0002  # how far the inlay plate's face stands off the board it is set in


def _legend_tris(lift: float) -> np.ndarray:
    """The label's printed oval, laid on the board's front face and lifted off it by `lift`."""
    at, half = (0.0, 0.2, _LABEL_FRONT), (0.12, 0.02)
    points, faces = decal_grid(DecalForm(half=half, segments=24, rings=4, at=at))
    verts = np.column_stack([at[0] + points[:, 0], at[1] + points[:, 1],
                             np.full(len(points), _LABEL_FRONT + lift)])
    tris = []
    for f in faces:
        tris += [(f[0], f[k], f[k + 1]) for k in range(1, len(f) - 1)]
    return verts[np.array(tris)]


def label() -> PieceSpec:
    """A board with a legend printed on its front (a decal of a `print` row) and a raised
    badge, a solid part that stands 8 mm off the board."""
    legend = {"half": [0.12, 0.02], "segments": 24, "rings": 4, "at": [0.0, 0.2, _LABEL_FRONT],
              "along": [0.0, 0.0, -1.0], "why": "a legend printed on the board, cast on from "
                                                "the front"}
    parts = [_part("board", "board", None, "board"), _part("badge", "badge", "board", "badge"),
             _part("legend", "ink", "board", "legend", shape="decal", role="decal", form=legend)]
    raw = _piece("floor, under the board centre; the front is +z",
                 [("board", "Board, 0.5 mm minimum", _THIN_M),
                  ("badge", "Raised badge, 0.5 mm minimum", _THIN_M),
                  ("ink", "Dried pad-print ink film, about 4 microns",
                   _INK_FILM_M)],
                 parts, "A board with a printed legend and a raised badge.")
    raw["listed"][2]["finish"] = "print"

    def geometry(lift):
        return {"board": box((-0.2, 0.0, -0.01), (0.2, 0.3, _LABEL_FRONT)),
                "badge": box((-0.04, 0.1, _LABEL_FRONT), (0.04, 0.16, _LABEL_FRONT + 0.008)),
                "legend": _legend_tris(lift)}
    return PieceSpec(raw, geometry)


def inlay() -> PieceSpec:
    """A board with a plate set into its front face so that the plate's face is 0.2 mm proud: a
    solid row that shows only as a colour change."""
    parts = [_part("board", "board", None, "board"), _part("plate", "plate", "board", "plate")]
    raw = _piece("floor, under the board centre; the front is +z",
                 [("board", "Board, 0.5 mm minimum", _THIN_M),
                  ("plate", "Flush plate, 0.5 mm minimum", _THIN_M)],
                 parts, "A board with a plate set flush into its front.")

    def geometry(lift):
        return {"board": box((-0.2, 0.0, -0.02), (0.2, 0.3, 0.02)),
                "plate": box((-0.05, 0.1, 0.01), (0.05, 0.2, 0.02 + _PLATE_PROUD))}
    return PieceSpec(raw, geometry)


BUILDERS.update({"Label": label, "Inlay": inlay})


# --- a turned leg ------------------------------------------------------------------------------
# A lathe-turned table leg: a round post whose radius swells and narrows in beads. The detail
# check plants on it: its ring has `segments` sides, so halving them is a blocky leg and doubling
# them a wasteful one. It is a fixture of the check's own, not part of the body scenes.

TURNED_LEG_HEIGHT = 0.72
TURNED_LEG_RADIUS = 0.030  # the post
TURNED_LEG_BEAD = 0.006  # how far a bead swells the radius
TURNED_LEG_PITCH = 0.18  # the distance between beads
TURNED_LEG_STEP = 0.005  # the profile's station spacing: its facets (about 0.02 mm) stay
# under those of the rings, so the ring's segments are what the detail check sees


def _lathe(profile, segments: int) -> np.ndarray:
    """A closed surface of revolution about y, wound outward: `profile` runs bottom to top as
    (y, radius), the first and last radius 0 (a flat end seen as a fan)."""
    theta = 2 * np.pi * np.arange(segments) / segments
    rings = [np.stack([r * np.cos(theta), np.full(segments, y), r * np.sin(theta)], axis=1)
             for y, r in profile[1:-1]]
    south = np.array([0.0, profile[0][0], 0.0])
    north = np.array([0.0, profile[-1][0], 0.0])
    tris = []
    for j in range(segments):
        k = (j + 1) % segments
        tris.append([south, rings[0][k], rings[0][j]])
        tris.append([north, rings[-1][j], rings[-1][k]])
        for a, b in itertools.pairwise(rings):
            tris.append([a[j], a[k], b[k]])
            tris.append([a[j], b[k], b[j]])
    return np.array(tris)


def turned_leg_tris(segments: int = 32) -> np.ndarray:
    """The turned leg's triangles (N, 3, 3), in metres, standing on the floor at the origin."""
    ys = np.arange(0.0, TURNED_LEG_HEIGHT + 1e-9, TURNED_LEG_STEP)
    radius = TURNED_LEG_RADIUS + TURNED_LEG_BEAD * np.sin(2 * np.pi * ys / TURNED_LEG_PITCH)
    profile = [(0.0, 0.0), *zip(ys, radius, strict=True), (float(ys[-1]), 0.0)]
    return _lathe(profile, segments)


def turned_leg(segments: int = 32) -> PieceSpec:
    parts = [_part("leg", "post", None, "leg")]
    raw = _piece("floor, under the leg's axis",
                 [("post", "Turned post, 60 mm", 0.06)], parts,
                 "A turned table leg: a round post with beads.")
    return PieceSpec(raw, lambda lift: {"leg": turned_leg_tris(segments)})


BUILDERS["Leg"] = turned_leg


# A rail with a post whose foot is a tenon driven into it, and a thin cap flush on the post's top
# (a clean pair, so the check has something to plant on). The model has no mortise cut in the
# rail, so the post shares space with it: a tenon seated in a mortise, which `crossings` reads as
# an embedded part. A test declares the join with an allowlist line for `post x rail`; undeclared,
# the same pair fails.
TENON_SEAT = 0.03  # how far the post's foot sits into the rail, along y


def tenon() -> PieceSpec:
    parts = [_part("rail", "rail", None, "rail"), _part("post", "post", "rail", "post"),
             _part("cap", "cap", "post", "cap")]
    raw = _piece("floor, under the rail centre",
                 [("rail", "Rail, 40 mm", 0.04), ("post", "Post, 60 mm square", 0.06),
                  ("cap", "Cap, 20 mm", 0.02)],
                 parts, "A rail with a post tenoned into it and a cap on the post.")

    def geometry(lift):
        return {"rail": box((-0.3, 0.0, -0.03), (0.3, 0.04, 0.03)),
                "post": box((-0.03, 0.04 - TENON_SEAT, -0.03), (0.03, 0.34, 0.03)),
                "cap": box((-0.06, 0.34, -0.06), (0.06, 0.36, 0.06))}
    return PieceSpec(raw, geometry)


BUILDERS["Tenon"] = tenon
