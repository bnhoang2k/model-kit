"""Check no-fallback: a missed anchor stops the build; it never falls back to a point.

Two parts.

**The build scripts.** `ctx.sources` holds each build script as text (`sources/<name>.py`). Each
is read as a syntax tree, and these fail:

- `<x>.get(<key>, <default>)` where `<x>` is a name or an attribute whose lowercase name has a
  token `anchor`, `hold`, `mount`, `clamp`, `seat` or `use` (or its plural). The tokens are the
  words of a snake_case or CamelCase name, so `mount_point`, `holds` and `AnchorMap` match and
  `amount`, `house` and `threshold` do not. A default there turns a missed lookup into
  a made-up point. Item `<file> get-default <name>`.
- the same `<x>.get(<key>)` with one argument: a miss returns `None`, which the build then
  carries as if it were a point. Item `<file> get-no-default <name>`.
- `<x>[<key>] if <key> in <x> else <default>` on the same watched names: the lookup written out
  by hand, with the default in the `else`. Both sides must name the same table and the same key
  (compared as syntax trees). Item `<file> conditional-index <name>`.
- a `print` whose text holds `fallback` (any case) or `MISMATCH` (as written, upper case): a
  build that announces a mismatch and goes on. Item `<file> print <word>`.
- `except Exception: pass` (and a bare `except: pass`, or `BaseException`, alone or among the
  names of a tuple such as `except (Exception, KeyError): pass`): an error swallowed. Item
  `<file> except-pass`.

A finding names every line it covers. A scene with no `sources/` has nothing to read, so it is
`Unmeasured` ("skipped (no sources)": the run is `partial`, since the lint could not read the build
code). A source that does not parse raises.

**The lookup.** `anchor()` is run on a mount ref that no piece has, and it must raise
`AnchorMissing`. Item `anchor missing-mount`. This rule needs no sources, so it runs on every
scene. The plant for it is a stand-in lookup that returns a default (`with_anchor`); a scene
carries it to the check as `anchor_fn`, which only a plant sets.

The other plants add a build script that holds each fault. They are listed in `PLANT_FAULTS`,
beside the script, and not read back from the scan: a plant that asked the scan for its items
could not tell a blind scan from a clean one. A test pins the list to the script. The lookup's
plant is always built; the gate skips it when the real lookup already fails.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any

from model_kit.core.errors import AnchorMissing, ManifestError
from model_kit.core.manifest import Ref, anchor
from model_kit.core.metagate import Finding, Plant, PlantUnit, Unmeasured
from model_kit.core.scene import SceneCtx

NAME = "no-fallback"

# A `.get(key, default)` on a name with one of these as a token is a fallback to a point.
WATCHED = ("anchor", "hold", "mount", "clamp", "seat", "use")
# A print that holds one of these announces a fallback or a mismatch and carries on. `fallback`
# matches in any case; `MISMATCH` matches as written, the marker the build scripts print.
ANNOUNCED = ("fallback", "MISMATCH")
ANNOUNCED_ANY_CASE = ("fallback",)
# The errors an `except` ... `pass` may not swallow (a bare `except` is the same fault).
SWALLOWED = ("Exception", "BaseException")

ANCHOR_ITEM = "anchor missing-mount"
MISSING_ID = "missing_mount"

PLANT_SOURCE = """\
mounts = {}
clamp = mounts.get('clamp', (0, 0, 0))
print('using the fallback point')
print('MISMATCH on clamp')
spot = holds.get(key)
seat = seats[key] if key in seats else (0, 0, 0)
try:
    work()
except Exception:
    pass
"""

# What PLANT_SOURCE holds, as `<rule> <name>` (the item without its file name). A test pins this
# to what the scan finds in the script.
PLANT_FAULTS = (
    "get-default mounts",
    "print fallback",
    "print MISMATCH",
    "get-no-default holds",
    "conditional-index seats",
    "except-pass",
)


@dataclass(frozen=True, eq=False)
class _StandInCtx(SceneCtx):
    """A scene that carries a stand-in for `anchor()`; only a plant builds one."""

    anchor_fn: Callable[..., Any] | None = None


def with_anchor(ctx: SceneCtx, fn: Callable[..., Any]) -> SceneCtx:
    """The scene with `fn` standing in for `anchor()` in the lookup rule."""
    base = {f.name: getattr(ctx, f.name) for f in dataclasses.fields(SceneCtx)}
    return _StandInCtx(**base, anchor_fn=fn)


# --- the build scripts ------------------------------------------------------------------------


def _name_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _watched(name: str) -> bool:
    """Whether a name has a watched token, in the singular or the plural. Tokens are the words
    between underscores and at the humps of a CamelCase name, never pieces inside a word."""
    tokens = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower().split("_")
    return any(t in WATCHED or t.removesuffix("s") in WATCHED for t in tokens)


def _caught(node: ast.expr) -> list[ast.expr]:
    """The error names an `except` line catches: the elements of a tuple, or the one name."""
    return list(node.elts) if isinstance(node, ast.Tuple) else [node]


def _indexed_when_present(node: ast.IfExp) -> str | None:
    """The table's name in `T[k] if k in T else d`, or None when `node` is not that form."""
    body, test = node.body, node.test
    if not (isinstance(body, ast.Subscript) and isinstance(test, ast.Compare)
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.In)):
        return None
    same_key = ast.dump(test.left) == ast.dump(body.slice)
    same_table = ast.dump(test.comparators[0]) == ast.dump(body.value)
    return _name_of(body.value) if same_key and same_table else None


def _text_of(call: ast.Call) -> str:
    """Every string in a call's arguments, f-string parts included."""
    return " ".join(n.value for arg in (*call.args, *(k.value for k in call.keywords))
                    for n in ast.walk(arg) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str))


def _scan_source(name: str, text: str) -> dict[str, list[int]]:
    """Item -> the lines of each fault in one build script."""
    try:
        tree = ast.parse(text, filename=name)
    except SyntaxError as exc:
        raise ManifestError(f"{name}: cannot read the build script: {exc}") from exc
    hits: dict[str, list[int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get":
                receiver = _name_of(func.value)
                if receiver and _watched(receiver):
                    if len(node.args) >= 2 or any(k.arg == "default" for k in node.keywords):
                        rule = "get-default"
                    elif len(node.args) == 1 and not node.keywords:
                        rule = "get-no-default"
                    else:
                        continue
                    hits.setdefault(f"{name} {rule} {receiver}", []).append(node.lineno)
            elif isinstance(func, ast.Name) and func.id == "print":
                text_of = _text_of(node)
                for word in ANNOUNCED:
                    found = word.casefold() in text_of.casefold() if word in ANNOUNCED_ANY_CASE \
                        else word in text_of
                    if found:
                        hits.setdefault(f"{name} print {word}", []).append(node.lineno)
        elif isinstance(node, ast.IfExp):
            table = _indexed_when_present(node)
            if table and _watched(table):
                hits.setdefault(f"{name} conditional-index {table}", []).append(node.lineno)
        elif isinstance(node, ast.ExceptHandler):
            if ((node.type is None or any(_name_of(t) in SWALLOWED for t in _caught(node.type)))
                    and all(isinstance(b, ast.Pass) for b in node.body)):
                hits.setdefault(f"{name} except-pass", []).append(node.lineno)
    return hits


_WHY = {
    "get-default": "a default on a lookup of anchors, holds, mounts, clamps, seats or use stations "
                   "turns a miss into a made-up point; index it so a miss raises",
    "get-no-default": "a lookup of anchors, holds, mounts, clamps, seats or use stations that "
                      "returns None on a miss carries the miss on as a point; index it so a "
                      "miss raises",
    "conditional-index": "`T[k] if k in T else d` is a default on a lookup of anchors, holds, "
                         "mounts, clamps, seats or use stations; index it so a miss raises",
    "print": "the build announces a fallback or mismatch and carries on; stop it instead",
    "except-pass": "`except Exception: pass` swallows an error; let it stop the build",
}


def _source_findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    found = []
    for name, text in sorted(ctx.sources.items()):
        for item, lines in _scan_source(name, text).items():
            if only is not None and item not in only:
                continue
            rule = item.split(" ")[1]
            where = f"line {lines[0]}" if len(lines) == 1 else \
                f"lines {', '.join(map(str, sorted(lines)))}"
            found.append(Finding(NAME, item, f"{where}: {_WHY[rule]}"))
    return found


# --- the lookup -------------------------------------------------------------------------------


def _missing_ref(ctx: SceneCtx) -> Ref:
    """A mount ref that no piece has: on the first piece of the first pack, or on a pack the
    scene lacks when it holds none."""
    for pack_name, pack in sorted(ctx.packs.items()):
        for piece in pack.pieces.values():
            taken = {x.id for group in (piece.holds, piece.seats, piece.mounts, piece.uses)
                     for x in group}
            id = MISSING_ID
            while id in taken:
                id += "_"
            return Ref(pack_name, piece.name, id)
    return Ref("nopack", "Nothing", MISSING_ID)


def _anchor_findings(ctx: SceneCtx, only: Collection[str] | None) -> list[Finding]:
    if only is not None and ANCHOR_ITEM not in only:
        return []
    lookup = getattr(ctx, "anchor_fn", None) or anchor
    ref = _missing_ref(ctx)
    try:
        got = lookup(ctx.packs, ref)
    except AnchorMissing:
        return []
    except Exception as exc:  # noqa: BLE001 - any other error is the fault being reported
        return [Finding(NAME, ANCHOR_ITEM, f"anchor() on the missing mount {ref} raised "
                                           f"{type(exc).__name__}, not AnchorMissing")]
    return [Finding(NAME, ANCHOR_ITEM, f"anchor() on the missing mount {ref} returned {got!r} "
                                       "instead of raising: a missed lookup must stop the build")]


# --- the check --------------------------------------------------------------------------------


def _plant_file(ctx: SceneCtx) -> str:
    name = "plant_fallback.py"
    while name in ctx.sources:
        name = "plant_" + name
    return name


class NoFallback:
    name = NAME
    kinds: frozenset[str] = frozenset()  # build scripts and the lookup, not a recipe

    def plants(self, ctx: SceneCtx) -> list[PlantUnit]:
        name = _plant_file(ctx)
        planted = dataclasses.replace(ctx, sources={**ctx.sources, name: PLANT_SOURCE})
        out = [PlantUnit(item, (Plant(f"script: {item}", item, planted),))
               for item in (f"{name} {fault}" for fault in PLANT_FAULTS)]
        stand_in = with_anchor(ctx, lambda packs, ref, *, clip=None: None)
        out.append(PlantUnit(ANCHOR_ITEM, (Plant(f"stand-in: {ANCHOR_ITEM}", ANCHOR_ITEM,
                                                 stand_in),)))
        return out

    def run(self, ctx: SceneCtx, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        # No per-sample work here: `samples` is accepted and ignored.
        return _source_findings(ctx, only) + _anchor_findings(ctx, only)

    def coverage(self, ctx: SceneCtx) -> list[str]:
        if ctx.sources:
            return []
        return [Unmeasured(f"{NAME}: skipped (no sources): {ctx.folder or 'scene'}")]
