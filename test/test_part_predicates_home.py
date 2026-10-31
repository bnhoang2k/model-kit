"""One home for what a part is.

Whether a part is printed, solid, outlined or an open surface is derived from its role and shape
in one place, `PartKind` in `core/manifest/records.py` (`Part` and `PartMesh` inherit it). A check
reads the properties and never compares a `.role` or `.shape` with a vocabulary word. `markings`
alone reads them raw, because it judges a role and a shape that disagree. A derived fact has
one owner, so no check re-derives it (Hunt and Thomas, The Pragmatic Programmer, 1999, "DRY").

The scan is an AST walk over `core/checks/`: a comparison (`==`, `!=`, `in`, `not in`, ...) of a
`.role` or `.shape` attribute with a string literal, a name or a collection of them fails. Two
array `.shape`s compared with each other, or an array `.shape` compared with a tuple of numbers,
are not matched.
"""

import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks.pieces import PartMesh
from model_kit.core.manifest import PART_ROLES, SHAPES, Part
from model_kit.core.manifest.records import decal_problem

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "src" / "model_kit" / "core" / "checks"
FIELDS = frozenset({"role", "shape"})

# Modules that may read role and shape raw: `markings` judges a role and a shape that disagree.
EXEMPT = frozenset({"markings.py"})


def _vocabulary(node: ast.expr, inside: bool = False) -> bool:
    """Whether a compared operand names a word: a string, a name, a module-qualified name
    (`records.DECAL_SHAPE`), or a collection (a literal, or a `frozenset`, `set`, `tuple` or `list`
    call) holding one. Inside a collection a bare name counts only when it is upper-case, a
    constant by convention, so an array shape such as `(n, 3, 3)` is not matched."""
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, ast.Name):
        return node.id.isupper() if inside else True
    if isinstance(node, ast.Attribute):
        return node.attr not in FIELDS
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_vocabulary(item, True) for item in node.elts)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in ("frozenset", "set", "tuple", "list")):
        return any(_vocabulary(arg, inside) for arg in node.args)
    return False


def role_or_shape_comparisons_in(source: str) -> list[int]:
    """The line of each comparison of a `.role` or `.shape` attribute with a vocabulary word."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        reads = [o for o in operands if isinstance(o, ast.Attribute) and o.attr in FIELDS]
        others = [o for o in operands if o not in reads]
        if reads and any(_vocabulary(o) for o in others):
            found.append(node.lineno)
    return found


def role_or_shape_comparisons(exempt: frozenset[str] = EXEMPT) -> list[str]:
    out = []
    for path in sorted(CHECKS.rglob("*.py")):
        if path.name in exempt:
            continue
        rel = path.relative_to(REPO)
        out += [f"{rel}:{line}" for line in role_or_shape_comparisons_in(path.read_text())]
    return out


def test_role_and_shape_are_read_only_through_predicates():
    assert role_or_shape_comparisons() == []


@pytest.mark.parametrize("source", [
    'if p.role != "decal": pass',
    'x = part.shape == "cable"',
    "ok = part.role in THICKNESS_ROLES",
    'ok = record.shape in ("cable", "decal")',
    'ok = "eyes" == part.role',
    "ok = part.shape not in EXEMPT_SHAPES",
    "ok = p.role in manifest.PRINTED_ROLES",
    "ok = p.shape == records.DECAL_SHAPE",
    'ok = p.role in frozenset({"eyes", "decal"})',
    'ok = p.shape in set(("cable", "decal"))',
    "ok = p.shape in (TUBE, LEAF)",
])
def test_planted_role_comparison_fails(source):
    assert role_or_shape_comparisons_in(source) != []


@pytest.mark.parametrize("source", [
    "a.shape == b.shape",
    "tris.shape != (3, 3)",
    "ok = tris.shape == (n, 3, 3)",
    "ok = tris.shape == (len(tris), 3, 3)",
    "if n.shape != (3,) or not ok: pass",
    "same = weights.shape != bones.shape",
    "x = part.name == 'eye'",
    'item = f"{p.role} {p.shape}"',
])
def test_array_shape_comparison_is_not_matched(source):
    assert role_or_shape_comparisons_in(source) == []


def test_markings_is_the_only_exempt_module_and_still_reads_raw():
    assert EXEMPT == {"markings.py"}
    assert role_or_shape_comparisons_in((CHECKS / "markings.py").read_text()) != []


# --- the predicates --------------------------------------------------------------------------


def _part(role: str, shape: str, on: str | None = "host") -> Part:
    """A part with this role and shape, set past `Part.validate`. These tests ask what a role and
    a shape mean together, pairs the validator refuses included (a decal with no `on`, a decal
    with a solid role); `test_manifest` proves the constructor refuses them."""
    part = Part(name="p", listed="row", shape="box", on=on, group="g", role="solid", bones=(),
                bind=None)
    object.__setattr__(part, "shape", shape)
    object.__setattr__(part, "role", role)
    return part


def _mesh(role: str, shape: str | None) -> PartMesh:
    return PartMesh("p", np.zeros((0, 3, 3)), role, shape, "g", None)


def test_cable_bends_and_box_does_not():
    assert _part("solid", "cable").bends
    assert not _part("solid", "box").bends
    assert not _part("solid", "decal").bends


def test_tube_is_only_the_tube_shape():
    for shape in sorted(SHAPES):
        assert _part("solid", shape).tube == (shape == "tube")


def test_printed_and_solid_split_the_roles():
    for role in PART_ROLES:
        part = _part(role, "box")
        assert part.printed == (role in ("eyes", "decal"))
        assert part.solid == (role in ("solid", "skip_outline"))
        assert part.printed != part.solid


def test_skips_outline_is_a_decal_or_a_no_outline_role():
    assert _part("eyes", "box").skips_outline
    assert _part("skip_outline", "box").skips_outline
    assert _part("eyes", "decal").skips_outline
    assert _part("solid", "decal").skips_outline
    assert not _part("solid", "box").skips_outline
    assert not _part("decal", "box").skips_outline
    # a cable draws no outline lines, yet detail still judges its facets: `bends` is its own fact
    assert not _part("solid", "cable").skips_outline


def test_open_surface_is_a_leaf_a_cable_or_a_decal():
    for shape in sorted(SHAPES):
        assert _part("solid", shape).open_surface == (shape in ("leaf", "cable", "decal"))


def test_a_part_mesh_gives_the_same_answers_as_its_part():
    for role, shape in itertools.product(PART_ROLES, sorted(SHAPES)):
        part, mesh = _part(role, shape), _mesh(role, shape)
        for name in ("printed", "solid", "skips_outline", "open_surface", "bends", "tube",
                     "skipped_as"):
            assert getattr(mesh, name) == getattr(part, name), (role, shape, name)


def test_a_mesh_with_no_record_is_a_solid_with_no_shape():
    mesh = _mesh("solid", None)
    assert mesh.solid and not (mesh.printed or mesh.skips_outline or mesh.open_surface
                               or mesh.bends)


@pytest.mark.parametrize(("role", "shape", "word"), [
    ("eyes", "decal", "decal"),       # shape first: bar 03's eyes
    ("eyes", "box", "eyes"),
    ("skip_outline", "leaf", "skip_outline"),  # bars 02 and 04's leaves, as detail pins them
    ("solid", "cable", "cable"),
    ("skip_outline", "box", "skip_outline"),
    ("decal", "box", "decal"),
])
def test_skipped_as_keeps_the_pinned_lines(role, shape, word):
    assert _part(role, shape).skipped_as == word


# --- the two refusals of the loader ---------------------------------------------------------


def test_decal_problem_names_the_part():
    assert decal_problem(_part("eyes", "decal", on=None)) is not None
    assert "'p'" in decal_problem(_part("eyes", "decal", on=None))
    assert decal_problem(_part("solid", "decal")) is not None
    assert decal_problem(_part("skip_outline", "decal")) is not None
    assert decal_problem(_part("eyes", "decal")) is None
    assert decal_problem(_part("decal", "decal")) is None
    # a role and a shape that disagree in another way stay a markings finding
    assert decal_problem(_part("decal", "box", on=None)) is None
    assert decal_problem(_part("eyes", "box")) is None
