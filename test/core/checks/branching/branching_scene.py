"""Grown scenes for the branching check tests: a small hand-made tree, the sweep of redbuds, and
a scene that holds any skeleton as a branching piece beside one assembly piece."""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from pathlib import Path

import numpy as np

from model_kit.core.grow.buds import MESH_VERSION, grow_buds
from model_kit.core.grow.mesh import grown_parts, leaf_part_name
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import CONE, DOME, Skeleton, check_skeleton, limb_at, unit
from model_kit.core.manifest import Frame, Listed, Pack, Part, Piece, Recipe, View
from model_kit.core.scene import PieceGeom, SceneCtx, piece_key
from model_kit.core.sight import Sight, sight_for
from model_kit.core.units import Style

from ...shapes import block
from ..piece_scenes import caught

REPO = Path(__file__).resolve().parents[4]
UPM = 0.5
# The style's dial: the thinnest tube measured on a built tree, 9 mm radius, in metres.
MIN_THICKNESS_M = 0.018
STYLE = Style(units_per_metre=UPM, page_camera={}, dials={"min_thickness": MIN_THICKNESS_M})
PACK, PIECE, BOX = "grove", "Tree", "Crate"
UP = np.array([0.0, 1.0, 0.0])

SWEEP_SEEDS = range(1, 9)


# --- the bud grower's sweep --------------------------------------------------------------------

GROW_FIXTURES = REPO / "test" / "core" / "grow" / "fixtures"
BUD_SPECIES = ("tiny-buds", "tiny-spiral")  # two-ranked with no terminal bud; spiral with one
# The tiny species cite heights 0.5 to 1.5 m (`age.height_points`): both ends and the middle.
BUD_SWEEP_M = (0.5, 1.0, 1.5)
BUD_SWEEP_LITE_M = 1.5


@cache
def bud_rules(species: str):
    return load_rules(GROW_FIXTURES / f"{species}.rules.json", kit=KIT_RULES)


def v2_rules_file(species: str, folder: Path) -> Path:
    """A copy of the fixture species' rules in `folder` without `tubes.sides` and `lite.tube_sides`
    (and the rows only they cite): the rules of a buds-v2 tree, whose sides come from the view."""
    data = json.loads((GROW_FIXTURES / f"{species}.rules.json").read_text())
    for key in ("tubes.sides", "lite.tube_sides"):
        del data["params"][key]
    drop = ("tubes.sides", "lite.tree-line")
    data["rows"] = [row for row in data["rows"] if row["id"] not in drop]
    path = folder / f"{species}.rules.json"
    path.write_text(json.dumps(data))
    return path


@cache
def bud_tree(species: str, seed: int, height_m: float, lite: bool = False) -> Skeleton:
    return grow_buds(bud_rules(species), piece=PIECE, height_m=height_m, seed=seed, lite=lite,
                     units_per_metre=UPM)


def bud_sweep_cases() -> list[tuple[str, int, float, bool]]:
    """(species, seed, height in metres, lite) for the bud grower's sweep: each tiny species,
    seeds 1 to 8, at each sweep height full and at the tallest lite."""
    full = [(sp, s, h, False) for sp in BUD_SPECIES for h in BUD_SWEEP_M for s in SWEEP_SEEDS]
    return full + [(sp, s, BUD_SWEEP_LITE_M, True) for sp in BUD_SPECIES for s in SWEEP_SEEDS]


def bud_case_label(species: str, seed: int, height_m: float, lite: bool) -> str:
    return f"{species} seed {seed} {height_m} m{' lite' if lite else ''}"


def bud_sweep_trees() -> list[tuple[str, Skeleton]]:
    return [(bud_case_label(*case), bud_tree(*case)) for case in bud_sweep_cases()]


@cache
def bud_ctx(species: str, seed: int, height_m: float, lite: bool = False) -> SceneCtx:
    return grown_ctx(bud_tree(species, seed, height_m, lite))


# --- a small hand-made tree --------------------------------------------------------------------

LEVELS = ("Trunk", "Leaders", "Branches", "Twigs")
SHADES = ("Leaf",)
HEART = np.array([[0.0, 0.07], [0.44, 0.08], [0.42, 0.52], [0.0, 1.0], [-0.42, 0.52],
                  [-0.44, 0.08]])


@dataclasses.dataclass
class Limb:
    id: str
    level: int
    parent: int
    t: float
    pts: np.ndarray
    radii: np.ndarray
    tip: int = CONE
    rooted: bool = False
    buried: bool = False


def line(p0, d, length: float, n: int = 4) -> np.ndarray:
    """n + 1 points from p0 along d."""
    return np.asarray(p0, float) + unit(np.asarray(d, float))[None] * np.linspace(0, length,
                                                                                  n + 1)[:, None]


def seat(limbs: list[Limb], parent: int, t: float) -> np.ndarray:
    return limb_at(limbs[parent].pts, limbs[parent].radii, t)[0]


def small_limbs() -> list[Limb]:
    """A trunk with one root, two leaders from one seat, a branch and a twig."""
    s30, c30 = np.sin(np.radians(30)), np.cos(np.radians(30))
    limbs = [Limb("trunk", 0, -1, 0.0, line((0, -0.03, 0), UP, 1.03), np.linspace(0.04, 0.03, 5),
                  tip=DOME, rooted=True)]
    limbs.append(Limb("root.0", 0, 0, 0.12, line(seat(limbs, 0, 0.12), (1, -0.4, 0), 0.3),
                      np.linspace(0.024, 0.009, 5), buried=True))
    top = seat(limbs, 0, 0.97)
    limbs.append(Limb("leader.0", 1, 0, 0.97, line(top, (-s30, c30, 0), 0.8, 6),
                      np.linspace(0.022, 0.008, 7)))
    limbs.append(Limb("leader.1", 1, 0, 0.97, line(top, (s30, c30, 0), 0.8, 6),
                      np.linspace(0.022, 0.008, 7)))
    limbs.append(Limb("branch.0", 2, 2, 0.5, line(seat(limbs, 2, 0.5), (-1, 0.3, 0.2), 0.4, 5),
                      np.linspace(0.012, 0.006, 6)))
    limbs.append(Limb("twig.0", 3, 4, 0.6, line(seat(limbs, 4, 0.6), (-0.3, 0.6, 0.8), 0.15),
                      np.linspace(0.0055, 0.0045, 5)))
    return limbs


def _across(d: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """A unit vector across d: across ref, or across X when d runs along ref."""
    side = np.cross(ref, d)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross([1.0, 0.0, 0.0], d)
    return unit(side)


def small_leaves(limbs: list[Limb]) -> list[dict]:
    """Three leaves: two on the twig, one on the second leader."""
    out = []
    for n, (host, t) in enumerate(((5, 0.5), (5, 0.9), (3, 0.8))):
        q, r, d = limb_at(limbs[host].pts, limbs[host].radii, t)
        side = _across(d, UP) * (1 if n % 2 == 0 else -1)
        length = 0.06
        petiole = 0.5 * r + 0.2 * length
        pet = unit(side + 0.35 * d - 0.2 * UP)
        mid = unit(side + 0.45 * d - 0.3 * UP)
        across = _across(mid, UP)
        out.append({"host": host, "leaf_t": t, "node": q, "base": q + pet * petiole, "mid": mid,
                    "across": across, "normal": unit(np.cross(across, mid)), "length": length,
                    "petiole": petiole, "shade": 0})
    return out


def make_skeleton(limbs: list[Limb], leaves: list[dict], *, lite: bool = False,
                  levels=LEVELS) -> Skeleton:
    sizes = [len(limb.pts) for limb in limbs]

    def col(key, shape=(-1,), dtype=float):
        return np.array([leaf[key] for leaf in leaves], dtype=dtype).reshape(shape)

    sk = Skeleton(
        level_names=tuple(levels),
        limb_id=np.array([limb.id for limb in limbs], dtype=str),
        level=np.array([limb.level for limb in limbs], dtype=int),
        parent=np.array([limb.parent for limb in limbs], dtype=int),
        t=np.array([limb.t for limb in limbs], dtype=float),
        tip=np.array([limb.tip for limb in limbs], dtype=int),
        rooted=np.array([limb.rooted for limb in limbs], dtype=bool),
        buried=np.array([limb.buried for limb in limbs], dtype=bool),
        start=np.concatenate([[0], np.cumsum(sizes)]).astype(int),
        pts=np.concatenate([limb.pts for limb in limbs]).astype(float),
        radii=np.concatenate([limb.radii for limb in limbs]).astype(float),
        host=col("host", dtype=int), leaf_t=col("leaf_t"), node=col("node", (-1, 3)),
        base=col("base", (-1, 3)), mid=col("mid", (-1, 3)), across=col("across", (-1, 3)),
        normal=col("normal", (-1, 3)), length=col("length"), petiole=col("petiole"),
        shade=col("shade", dtype=int),
        shades=SHADES, sides=np.array([8, 6, 5, 4, 3][:len(levels)], dtype=int), outline=HEART,
        fold=0.1, blade_centre=np.array([0.0, 0.45, 0.02]), petiole_width=0.002,
        petiole_tip=0.07, lite=lite,
        info={"piece": PIECE},
    )
    check_skeleton(sk, "test skeleton")
    return sk


def small_tree() -> Skeleton:
    limbs = small_limbs()
    return make_skeleton(limbs, small_leaves(limbs))


def v2_skeleton(limbs: list[Limb], leaves: list[dict] | None = None) -> Skeleton:
    """A buds-v2 skeleton made by hand: it takes its tube sides from the piece's view when meshed,
    and `Skeleton.sides` is 0 as the grower leaves it."""
    sk = make_skeleton(limbs, leaves or [])
    return dataclasses.replace(sk, sides=np.zeros(len(sk.level_names), dtype=int),
                               info={**sk.info, "mesh": MESH_VERSION})


def tube_skeleton(radius: float, length: float = 0.5) -> Skeleton:
    """A buds-v2 skeleton of one straight, untapered tube of `radius` standing on the ground."""
    return v2_skeleton([Limb("trunk", 0, -1, 0.0, line((0, 0, 0), UP, length, 4),
                             np.full(5, radius), tip=CONE, rooted=True)])


def with_pts(sk: Skeleton, pts: np.ndarray, **changes) -> Skeleton:
    """A copy of sk with new limb points (and any other field changes)."""
    return dataclasses.replace(sk, pts=np.asarray(pts, float), **changes)


# --- the scene ---------------------------------------------------------------------------------


def _listed_and_parts(sk: Skeleton, parts: dict[str, np.ndarray]):
    listed, part_rows = [], []
    for li, name in enumerate(sk.level_names):
        if name not in parts:
            continue
        idx = np.flatnonzero(sk.level == li)
        smallest = min(float(sk.limb_radii(int(i)).min()) for i in idx)
        listed.append(Listed(name.lower(), f"{name} of the test tree", "test", 2 * smallest / UPM,
                             None))
        part_rows.append(Part(name, name.lower(), "tube", None, name, "solid", (), None))
    for shade in sk.shades:
        name = leaf_part_name(shade)
        if name not in parts:
            continue
        listed.append(Listed(name.lower(), f"{shade} leaves", "test", 0.0, None))
        part_rows.append(Part(name, name.lower(), "leaf", None, name, "skip_outline", (), None))
    return tuple(listed), tuple(part_rows)


def _piece(name: str, kind: str, listed, parts) -> Piece:
    return Piece(name=name, frame=Frame("gltf", "centre of the base"), recipe=Recipe(kind, "test"),
                 listed=listed, parts=parts, uses=(), holds=(), seats=(), mounts=(),
                 done="A test piece.", size_class="piece", measured={}, rig=None)


def grown_ctx(sk: Skeleton | None, *, parts: dict[str, np.ndarray] | None = None,
              listed_from: Skeleton | None = None, sight: Sight | None = None) -> SceneCtx:
    """A scene: pack `grove` with the branching piece `Tree` (the skeleton's parts, or `parts`)
    and the assembly piece `Crate`. `sk` None leaves the skeleton out. A buds-v2 skeleton takes
    its tube sides from `sight`; the older meshes ignore it."""
    source = listed_from or sk
    grown = grown_parts(source, sight)
    shipped = grown if parts is None else parts
    listed, part_rows = _listed_and_parts(source, grown)
    crate_rows = (Listed("board", "Board", "test", 0.01, None),)
    crate_parts = (Part("box", "board", "box", None, "box", "solid", (), None),)
    pack = Pack(PACK, UPM, {PIECE: _piece(PIECE, "branching", listed, part_rows),
                            BOX: _piece(BOX, "assembly", crate_rows, crate_parts)})
    return SceneCtx(
        style=STYLE, packs={PACK: pack}, clips={},
        pieces={piece_key(PACK, PIECE): PieceGeom(PIECE, dict(shipped)),
                piece_key(PACK, BOX): PieceGeom(BOX, {"box": block(0.1, 0.1)})},
        samples={}, rigs={}, rig_of_clip={},
        skeletons={} if sk is None else {piece_key(PACK, PIECE): sk},
    )


VIEWED_STYLE = dataclasses.replace(STYLE, dials={**STYLE.dials, "edges": {
    "crease_deg": 35.0, "crease_why": "the test style's shading angle"}})
VIEW = View(1.0, "a test tree, seen from a metre")


def viewed_ctx(sk: Skeleton, *, meshed_with: Sight | None = None, view: View | None = VIEW
               ) -> SceneCtx:
    """The scene of `grown_ctx` for a skeleton that needs a sight: the piece declares `view` (none
    leaves it unseen), the style a crease angle, and the shipped parts are meshed with the sight
    that view gives, or with `meshed_with`."""
    piece = dataclasses.replace(_piece(PIECE, "branching", (), ()), view=view)
    sight = meshed_with or sight_for(piece, VIEWED_STYLE, UPM)
    ctx = grown_ctx(sk, sight=sight)
    pack = ctx.packs[PACK]
    seen = dataclasses.replace(pack.pieces[PIECE], view=view)
    return dataclasses.replace(ctx, style=VIEWED_STYLE, packs={
        PACK: dataclasses.replace(pack, pieces={**pack.pieces, PIECE: seen})})


def item(rest: str) -> str:
    return f"{piece_key(PACK, PIECE)} {rest}"


# --- check helpers -----------------------------------------------------------------------------


def findings(check, ctx, only=None):
    return check.run(ctx) if only is None else check.run(ctx, only=only)


def plants_caught(check, ctx) -> list[tuple[str, bool]]:
    """(label, caught) for each plant unit: the label of its first candidate, caught when some
    candidate is (the run on its scope names its item). A unit with no candidate reads
    ("no plant: <item>", False), as it stops the gate."""
    return [(unit.candidates[0].label if unit.candidates else f"no plant: {unit.item}",
             any(caught(check, p) for p in unit.candidates))
            for unit in check.plants(ctx)]


def limb_index(sk: Skeleton, name: str) -> int:
    return int(np.flatnonzero(sk.limb_id == name)[0])


def moved_limb(sk: Skeleton, name: str, pts: np.ndarray) -> Skeleton:
    """A copy of sk with one limb's points replaced (same count)."""
    i = limb_index(sk, name)
    new = np.array(sk.pts, dtype=float)
    new[sk.start[i]:sk.start[i + 1]] = pts
    return with_pts(sk, new)


def tiny_ctx() -> SceneCtx:
    """A bud-grown scene for a check's clean-pass and plants tests: tiny spiral, seed 1, 1.5 m."""
    return bud_ctx("tiny-spiral", 1, 1.5)
