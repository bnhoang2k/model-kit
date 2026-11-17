import ast
import importlib
import inspect
import re

import pytest

from model_kit.core import gate
from model_kit.core.errors import ManifestError
from model_kit.core.metagate import scope_of
from model_kit.core.scope import Scope, parse_pieces

from .checks.piece_scenes import two_pieces

# The scene fields a piece-scoped check may read: each is keyed by piece, or is not cut by a scope.
PER_PIECE = {"pieces", "packs", "rigs", "rests", "skeletons", "style", "rules", "folder"}


def fields_read(source: str) -> set[str]:
    """The scene fields a source reads: each `ctx.<name>`."""
    return set(re.findall(r"\bctx\.([a-z_]+)", source))


def _modules_read(module, seen: set[str]) -> None:
    """The check modules `module` imports, walked down, added to `seen`."""
    if module.__name__ in seen:
        return
    seen.add(module.__name__)
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom) and node.module:
            for target in (node.module, *(f"{node.module}.{a.name}" for a in node.names)):
                if target.startswith("model_kit.core.checks"):
                    try:
                        _modules_read(importlib.import_module(target), seen)
                    except ImportError:
                        pass


def test_fields_read_finds_every_scene_field_a_source_reads():
    source = "def run(self, ctx):\n    return ctx.samples, self.ctx.layouts"
    assert fields_read(source) == {"samples", "layouts"}


def test_piece_scoped_checks_read_only_per_piece_fields():
    for cls in gate.CHECKS.values():
        if scope_of(cls) != "piece":
            continue
        modules: set[str] = set()
        _modules_read(importlib.import_module(cls.__module__), modules)
        for name in sorted(modules):
            extra = fields_read(inspect.getsource(importlib.import_module(name))) - PER_PIECE
            assert not extra, (f"{cls.name} is piece-scoped but {name} reads "
                               f"ctx.{', ctx.'.join(sorted(extra))}")


def test_the_known_piece_scoped_checks_are_piece_scoped():
    names = ("crossings", "take-apart", "bearing")
    assert [scope_of(gate.CHECKS[n]) for n in names] == ["piece"] * 3
    assert scope_of(gate.CHECKS["coverage"]) == "split"
    assert scope_of(gate.CHECKS["clearance"]) == "scene"


def test_the_scoped_scene_drops_only_the_pieces_not_named():
    ctx = two_pieces()
    scoped = Scope(frozenset({"kit:Thing"})).scene(ctx)
    assert set(scoped.pieces) == {"kit:Thing"}
    assert set(scoped.packs["kit"].pieces) == {"Thing"}
    assert scoped.layouts is ctx.layouts and scoped.clips is ctx.clips
    assert set(ctx.pieces) == {"kit:Thing", "kit:Other"}


def test_the_scoped_scene_is_built_once_per_scene():
    ctx = two_pieces()
    scope = Scope(frozenset({"kit:Thing"}))
    assert scope.scene(ctx) is scope.scene(ctx)
    assert scope.scene(two_pieces()) is not scope.scene(ctx)


@pytest.mark.parametrize("text", ["", "kit:Thing,kit:Thing", "kit:Nope"])
def test_parse_pieces_refuses_empty_repeated_and_unknown_keys(text):
    with pytest.raises(ManifestError):
        parse_pieces(text, two_pieces())


def test_parse_pieces_reads_a_list():
    scope = parse_pieces(" kit:Thing , kit:Other", two_pieces())
    assert scope.pieces == {"kit:Thing", "kit:Other"}
