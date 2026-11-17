"""One home for verdict constants and vocabularies.

A module-level number outside `core/tolerances.py` is either a verdict limit (a check compares
a measured value against it to decide pass or fail), which belongs in `tolerances.py`, or it
is listed in REGISTRY with the category that says why it may stay. Each fact has one
authoritative home, so a limit is never copied (Hunt and Thomas, The Pragmatic Programmer, 1999,
"DRY").
"""

import ast
import typing
from pathlib import Path
from typing import Literal

from model_kit.core import kinds
from model_kit.core.manifest import forms as manifest_forms
from model_kit.core.manifest import records

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "model_kit"
HOME = "core/tolerances.py"

CATEGORIES = frozenset({
    "float guard", "sampling", "plant", "budget", "axis", "units", "version", "layout",
    "grower model", "build geometry", "card copy", "read point",
})

# Roots the scan walks, repo-relative: the whole package.
SCANNED: tuple[str, ...] = ("src/model_kit",)

# "<module>.<NAME>" (dotted from src/model_kit) -> category. A name whose category is unclear
# is settled by one question: does a check compare a measured value against it to decide
# pass or fail? Yes: it lives in tolerances.py, not here.
_BY_CATEGORY = {
    # Rounding and zero tests. None is compared with a measured value to decide pass or fail.
    "float guard": """
        core.checks.branching.reference_range.SCALE_MATCH
        core.compare._EDGE_SLACK core.forms._FRAME_MIN core.forms._NORMAL_STEP
        core.geom.DEGENERATE_SHARE core.geom.NO_DIRECTION core.geom.TIE_DISTANCE
        core.geom._BOUND_SLACK core.geom._CUT_WELD_OF_SIZE core.geom._NO_AREA core.geom._NO_EDGE
        core.geom._NO_SPAN core.geom._PARALLEL core.geom._PLANE_SLACK core.geom._RAY_EDGE_SLACK
        core.geom._RAY_T_MIN core.geom._SLAB_PAD
        core.grow.shape.FLAT_EPS core.grow.shape.KEEP_EPS core.grow.shape.LEAF_FRAME_EPS
        core.grow.shape.PERP_EPS core.grow.shape.PETIOLE_FLAT_EPS core.grow.shape.SEAT_T_EPS
        core.grow.shape.SEGMENT_EPS core.grow.shape.TRANSPORT_EPS
        core.manifest.forms.AXIS_PARALLEL_SINE core.manifest.rig.LIMIT_SLACK_RAD
        core.scene.DUE_SLACK_FRAMES core.slices.NO_DIRECTION
        blender.export.STEP_TOL blender.export.UNIT_SCALE_TOL blender.export.WORLD_DRIFT_TOL
        blender.export.ZERO_LOCATION_TOL blender.grow.WELD_DECIMALS blender.soft._FRAME_MIN
        blender.bake_render.DENSITY_TOL blender.bake_render.FIT_TOL blender.bake_render.INSIDE_EPS
        core.checks.seam_jump.WELD core.checks.seam_jump.SAME_UV core.checks.seam_jump.SAME_NORMAL
    """,
    # How finely a check or a measure looks: counts of probes, rings, points and samples.
    # `compare._GAP_PERCENTILE` is here because no check compares a measured value against it:
    # it picks which statistic of the gap is judged, and the limit that judges it is in tolerances.
    "sampling": """
        core.checks.bend_quality.SLICES core.checks.branching._pairs.PLACE_SAMPLES
        core.checks.seam_jump.ALONG
        core.checks.branching.skeleton_match._OFFSETS
        core.geom.CUT_SAMPLES core.manifest.forms.MIN_RINGS core.manifest.forms.MIN_SEGMENTS
        core.pose.stress.FPS core.compare._GAP_PERCENTILE
        blender.build.FILL_GRID blender.soft.SEAT_RAY_MARGIN blender.bake_render.RAYS_LONG_SIDE
    """,
    # How hard a planted fault pushes, and how many it tries.
    "plant": """
        core.checks.bend_quality.PINCH_PLANT
        core.checks.lightmap_density.PLANT_FACTOR
        core.checks.seam_jump.BRIGHTEN core.checks.seam_jump.STEP_SIGMAS
        core.checks.inset_seat.SINK_STEPS
        core.checks.near_crease.BEND_OF_CREASE core.checks.near_crease.PLANT_EDGES
        core.checks.overlap.ROOM_FRACTIONS core.checks.overlap._PROBE_RATIO
        core.checks.stress_poses.PLANT_BEYOND
        core.checks.stress_poses.PLANT_VERTICES
    """,
    # Memory, time and iteration caps: they change how long a run takes, never an answer.
    "budget": """
        core.bvh.BVH_LEAF core.bvh.SEED_STRIDE core.bvh.START_PAIRS
        core.checks.branching._pairs.CHUNK
        core.checks.branching.reference_range.SCALE_STEPS core.geom.LEAF_SIZE
        core.geom.PAIR_BLOCK core.geom.PAIR_BUDGET core.geom._MORTON_BITS
        core.checks.overlap._MOST_PROBES
        core.grow.shape.PAIR_CHUNK core.solve.SOLID_STEPS core.solve.STEPS
        cli.PROFILE_TIMEOUT sheet.RENDER_TIMEOUT bake.BAKE_TIMEOUT compare.COMPARE_TIMEOUT
        blender.bake_render.FILL blender.bake_render.MAX_ATLAS blender.bake_render.THREADS
    """,
    # Where a check reads the picture. Unlike a sampling count, `seam_jump.INSETS` CHANGES the
    # answer a little (the extrapolation is exact for a linear light only), and the threshold's
    # `sqrt(5)` assumes the far read is at twice the near one's depth; changing that means
    # re-deriving the threshold.
    "read point": """
        core.checks.seam_jump.INSETS
    """,
    # Directions and frame conventions.
    "axis": """
        core.checks.bearing.Y core.checks.floaters.Y core.grow.mesh.EMPTY
        core.grow.shape.SOIL_Y core.manifest.forms.DECAL_ALONG core.solve._REST
        blender.soft._AWAY blender.soft._FRONT blender.soft._UP
    """,
    # Unit conversions, and how a value is printed.
    "units": """
        core.checks.trace._PRINT_DIGITS core.grow.shape.FULL_TURN_DEG
        core.grow.shape.HALF_TURN_DEG core.grow.shape.QUARTER_TURN_DEG
        blender.soft.META_ELLIPSOID_RADIUS blender.soft.META_SURFACE_PER_RADIUS
    """,
    # A fixed external contract (a file format, an exit code), not a verdict.
    # `units.MAX_DEPTH_BITS` is the widest integer depth format Khronos's "Depth Buffer Precision"
    # page lists (32 bits; the source is cited beside it in units.py). It bounds what a style may
    # declare, so a declaration is refused or accepted; no measured value is compared with it.
    "version": """
        core.gate.CANNOT_START core.glb._CHUNK_HEAD core.glb._HEADER blender.export.MIN_BLENDER
        core.units.MAX_DEPTH_BITS blender.grow.ALPHA_CUT
        blender.stage_scene.WHITE blender.stage_scene.UNIT
        core.exr._BLOCK_ROWS core.exr._HALF core.exr._NOT_PLAIN core.exr._ZIP
        core.checks.seam_jump.NORMAL_DIGITS
    """,
    # The contact sheet's drawing numbers: paper, inks, cell sizes, the scale post. `sheet.py`
    # and `blender/sheet_render.py` only.
    "layout": """
        sheet.CELL sheet.HEAD_INK sheet.LABEL_COLUMN sheet.LABEL_INK sheet.MARGIN sheet.PAPER
        sheet.POST_FOOT_RADIUS sheet.POST_GAP sheet.POST_M sheet.TITLE_BAND sheet.TITLE_INK
        blender.sheet_render.CAMERA_DISTANCE blender.sheet_render.POST_COLOUR
        blender.sheet_render.POST_RADIUS
    """,
    # The builders' own modelling numbers (a bevel width). They shape a mesh; no check compares
    # a measured value against one.
    "build geometry": """
        blender.build.BEVEL core.grow.card_image._RIB_HALF_WIDTH
        blender.bake_render.COLLAPSE_AT blender.bake_render.MARGIN_PX
        blender.bake_render.SMART_ANGLE blender.bake_render.SMART_STRETCH
        blender.bake_render.BAKE_MARGIN_PX
        blender.bake_render.DILATE_PX
    """,
    # The card copy: Razel's foliage.js numbers, named in `cards.py`, and the budget draws
    # around them. Not a check's limit.
    "card copy": """
        core.grow.cards.AO_BOOST core.grow.cards.AO_MIX core.grow.cards.AO_RAYS
        core.grow.cards.AO_REACH_M core.grow.cards.BURIED core.grow.cards.CARD_TILT
        core.grow.cards.FADE_IN core.grow.cards.HIDDEN
        core.grow.cards.LEAF_DENSITY core.grow.cards.ROUNDNESS core.grow.cards.SHOWN_CUT
        core.grow.cards.SIZE_DRAW core.grow.cards.SPREAD_JITTER core.grow.cards._TRIANGLES
    """,
    # The grower's own model numbers: rules-file territory.
    "grower model": """
        core.grow.buds.TERMINAL core.grow.card_image.PAINTED_SHARE core.grow.rules.FIT_SEEDS
        core.grow.shape.CLEAR_SUBSTEPS core.grow.shape.CONE_POINT_OF_RADIUS
        core.grow.shape.DOME_POINT_OF_RADIUS core.grow.shape.DOME_RINGS
        core.grow.shape.FLARE_HEIGHT_OF_RADIUS core.grow.shape.FLARE_MIN_GAP_OF_RADIUS
        core.grow.shape.FILLET_RADIUS_OF_TUBE_RADIUS core.grow.shape.FLARE_RINGS
        core.grow.shape.LEAF_BLADE_CENTRE core.grow.shape.LEAF_FOLD
        core.grow.shape.LEAF_PETIOLE_HALF_WIDTH_OF_TIP_RADIUS
        core.grow.shape.LEAF_PETIOLE_TIP_OF_BLADE core.grow.shape.LOBE_MAX_HALF_WIDTH_DEG
        core.grow.shape.LOBE_OF_ROOT_RADIUS core.grow.shape.LOBE_SPREAD
        core.grow.shape.MERGE_OFFSET_OF_RADIUS core.grow.shape.MERGE_RADIUS_STEP
        core.grow.shape.PETIOLE_START_OF_RADIUS core.grow.shape.REACH_CAP_M
        core.grow.shape.REACH_FAR_F core.grow.shape.REACH_MIN_M core.grow.shape.REACH_STEP_M
        core.grow.shape.RELAY_SHARE_OF_ANGLE core.grow.shape.ROOT_BELOW_HORIZONTAL_DEG
        core.grow.shape.ROOT_DROOP core.grow.shape.ROOT_JITTER_DEG
        core.grow.shape.ROOT_LENGTH_OF_RADIUS core.grow.shape.ROOT_RADIUS_OF_TRUNK
        core.grow.shape.ROOT_SEGMENTS core.grow.shape.ROOT_TIP_OF_BASE core.grow.shape.SCALE_MAX
        core.grow.shape.SCALE_MIN core.grow.shape.SELF_PIECE_OF_WINDOW
        core.grow.shape.SELF_WINDOW_OF_RADII core.grow.shape.SHADOW_CELL_OF_INTERNODE
        core.grow.shape.SIBLING_MIN_ANGLE_DEG core.grow.shape.STOP_SHORT_PASSES
        core.grow.shape.TRUNK_SINK_OF_RADIUS core.grow.shape.WOBBLE_BOX
        core.grow.shape.ZONE_STEP_OF_RADIUS
    """,
}
REGISTRY: dict[str, str] = {
    name: category for category, names in _BY_CATEGORY.items() for name in names.split()
}


def _is_number(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and _is_number(node.operand))


def _numeric(node: ast.expr) -> bool:
    """A number, a unary minus of one, a tuple of such, or a call or binary operation whose
    operands are all such values or upper-case names."""
    if _is_number(node):
        return True
    if isinstance(node, ast.Name):
        return node.id.isupper()
    if isinstance(node, ast.Tuple):
        return bool(node.elts) and all(_numeric(e) for e in node.elts)
    if isinstance(node, ast.BinOp):
        return _numeric(node.left) and _numeric(node.right)
    if isinstance(node, ast.UnaryOp):
        return _numeric(node.operand)
    if isinstance(node, ast.Call):
        args = [*node.args, *(k.value for k in node.keywords)]
        return bool(args) and all(_numeric(a) for a in args)
    return False


def _has_number(node: ast.expr) -> bool:
    """True when the value holds a literal number (a bare alias of another name is not a
    new number)."""
    return any(isinstance(n, ast.Constant) and _is_number(n) for n in ast.walk(node))


def _dotted(rel: str) -> str:
    """The dotted module path of `rel` (below `src/model_kit`); the package root is ""."""
    path = Path(rel)
    parts = path.with_suffix("").parts if path.suffix else path.parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def module_level_numbers_in_source(rel: str, source: str) -> list[str]:
    """Dotted names of the module-level upper-case numbers in `source`. `rel` is the path
    below `src/model_kit`. `core/tolerances.py` is skipped: it is the home."""
    if Path(rel).as_posix() == HOME:
        return []
    module = _dotted(rel)
    names = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not (_numeric(value) and _has_number(value)):
            continue
        names += [f"{module}.{t.id}" if module else t.id for t in targets
                  if isinstance(t, ast.Name) and t.id.isupper()]
    return names


def _prefix(root: str) -> str:
    """The dotted-name prefix of every name under the repo-relative `root`; "" for the whole
    package."""
    dotted = _dotted(Path(root).relative_to("src/model_kit").as_posix())
    return f"{dotted}." if dotted else ""


def module_level_numbers(roots: tuple[str, ...]) -> list[str]:
    """Every module-level upper-case number under the repo-relative `roots`, as dotted names
    below `src/model_kit`, sorted."""
    found = []
    for root in roots:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = path.relative_to(PACKAGE).as_posix()
            found += module_level_numbers_in_source(rel, path.read_text())
    return sorted(found)


def _literal(cls: type, name: str):
    """The `Literal` type of field `name` on `cls`, with an optional `| None` taken off. The
    record module names the form types only under TYPE_CHECKING, so `forms` supplies them."""
    hint = typing.get_type_hints(cls, localns=vars(manifest_forms))[name]
    if typing.get_origin(hint) is Literal:
        return hint
    members = [a for a in typing.get_args(hint) if typing.get_origin(a) is Literal]
    assert len(members) == 1, f"{cls.__name__}.{name} is {hint}, not a Literal"
    return members[0]


# Each pair is a vocabulary tuple and the `Literal` it repeats: the type of the record field
# that the loader fills from the tuple.
VOCABULARIES: list[tuple[tuple, object]] = [
    (records.RECIPES, _literal(records.Recipe, "kind")),
    (kinds.RECIPES, _literal(records.Recipe, "kind")),
    (kinds.RECIPES + ("rigged",), kinds.PieceClass),
    (records.FINISHES, _literal(records.Listed, "finish")),
    (records.HELD, _literal(records.Listed, "held")),
    (records.BIND_MODES, _literal(records.Part, "bind")),
    (records.BIND_MODES, _literal(records.Bind, "mode")),
    (records.POSTURES, _literal(records.Use, "posture")),
    (records.POSTURES, _literal(records.Stand, "posture")),
    (records.SIDES, _literal(records.Hold, "side")),
    (records.AXES, _literal(records.MirrorSpec, "axis")),
    (records.SIZE_CLASSES, _literal(records.Piece, "size_class")),
    (records.AUTHORING, _literal(records.Clip, "authoring")),
    (records.CLIP_POSTURES, _literal(records.Clip, "posture")),
    (manifest_forms.CAP_KINDS, _literal(manifest_forms.Cap, "kind")),
    (manifest_forms.BLENDS, _literal(manifest_forms.RingForm, "blend")),
    (manifest_forms.ELEMENT_KINDS, _literal(manifest_forms.Element, "kind")),
    (manifest_forms.FALLOFF_AXES, _literal(manifest_forms.Falloff, "along")),
]


def vocabulary_mismatches(pairs) -> list[str]:
    """One line per `(tuple, Literal)` pair whose values differ."""
    return [f"{tuple(tup)} != {typing.get_args(lit)}" for tup, lit in pairs
            if tuple(tup) != typing.get_args(lit)]


def test_every_module_number_is_registered_or_in_tolerances():
    unregistered = sorted(set(module_level_numbers(SCANNED)) - set(REGISTRY))
    assert unregistered == []


def test_registry_names_still_exist():
    """A stale registry entry fails, so the registry cannot outlive its constants."""
    scanned = set(module_level_numbers(SCANNED))
    prefixes = tuple(_prefix(root) for root in SCANNED)
    in_scope = {n for n in REGISTRY if n.startswith(prefixes)}
    assert sorted(in_scope - scanned) == []


def test_a_registry_name_has_one_category():
    names = [n for names in _BY_CATEGORY.values() for n in names.split()]
    assert sorted(n for n in set(names) if names.count(n) > 1) == []


def test_registry_categories_are_known():
    assert sorted({c for c in REGISTRY.values() if c not in CATEGORIES}) == []


def test_a_planted_constant_fails():
    src = "LIMIT = 0.3\n"
    assert module_level_numbers_in_source("core/checks/x.py", src) == ["core.checks.x.LIMIT"]


def test_recopied_weight_floor_in_bind_fails():
    """A second copy of the bind weight floor in `blender/bind.py` is found."""
    assert module_level_numbers_in_source("blender/bind.py", "WEIGHT_FLOOR = 0.001\n") == [
        "blender.bind.WEIGHT_FLOOR"]


def test_the_package_root_is_scannable():
    """The widened root is the package itself, whose dotted prefix is empty."""
    assert _prefix("src/model_kit") == ""
    assert _prefix("src/model_kit/core") == "core."
    assert module_level_numbers_in_source("cli.py", "LIMIT = 3\n") == ["cli.LIMIT"]
    assert module_level_numbers_in_source("__init__.py", "LIMIT = 3\n") == ["LIMIT"]


def test_the_scan_sees_tuples_calls_and_sums_but_not_tolerances_or_lowercase():
    src = ("A = (1, 2)\nB = -1.0\nC = math.radians(5.0)\nD = A * 2\nE = A\nF: float = 2.0\n"
           "g = 1.0\nH = 'x'\nI = (1, 'a')\n")
    assert module_level_numbers_in_source("m.py", src) == ["m.A", "m.B", "m.C", "m.D", "m.F"]
    assert module_level_numbers_in_source("core/tolerances.py", "LIMIT = 0.3\n") == []


def test_vocabulary_tuples_match_their_literals():
    assert vocabulary_mismatches(VOCABULARIES) == []


def test_a_planted_vocabulary_mismatch_fails():
    assert vocabulary_mismatches([(("a", "b"), Literal["a", "c"])]) != []
