"""Field readers shared by the pack, rig and clip readers. Each failure names its place."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from model_kit.core.errors import ManifestError, RecordError

from .records import _NAME, Box, Quat, Ref, Window, window_problem


class _At:
    """A place in a manifest: the file and the dotted field path, for error messages."""

    def __init__(self, path: Path, where: str = "") -> None:
        self.path, self.where = path, where

    def key(self, name: str) -> _At:
        return _At(self.path, f"{self.where}.{name}" if self.where else name)

    def index(self, i: int) -> _At:
        return _At(self.path, f"{self.where}[{i}]")

    def fail(self, why: str):
        raise ManifestError(f"{self.path}: {self.where or '(top)'}: {why}")


def _fields(at: _At, raw: Any, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    if not isinstance(raw, dict):
        at.fail(f"must be an object, got {type(raw).__name__}")
    for name in required:
        if name not in raw:
            at.fail(f"missing field {name!r}")
    for name in raw:
        if name not in required and name not in optional:
            at.fail(f"unknown field {name!r}")
    return raw


def _obj(at: _At, raw: Any) -> dict:
    if not isinstance(raw, dict):
        at.fail(f"must be an object, got {type(raw).__name__}")
    return raw


def _list(at: _At, raw: Any, *, nonempty: bool = False) -> list:
    if not isinstance(raw, list):
        at.fail(f"must be a list, got {type(raw).__name__}")
    if nonempty and not raw:
        at.fail("must not be empty")
    return raw


def _text(at: _At, raw: Any) -> str:
    if not isinstance(raw, str) or not raw.strip():
        at.fail(f"must be a non-empty string, got {raw!r}")
    return raw


def _name(at: _At, raw: Any, pattern: str = _NAME) -> str:
    if not isinstance(raw, str) or not re.fullmatch(pattern, raw):
        at.fail(f"must be a name made of letters, digits, '_' or '-', got {raw!r}")
    return raw


def _names(at: _At, raw: Any, pattern: str = _NAME) -> tuple[str, ...]:
    items = _list(at, raw, nonempty=True)
    names = tuple(_name(at.index(i), v, pattern) for i, v in enumerate(items))
    if len(set(names)) != len(names):
        at.fail(f"names repeat: {list(names)}")
    return names


def _choice(at: _At, raw: Any, options: tuple[str, ...]) -> str:
    if raw not in options:
        at.fail(f"must be one of {list(options)}, got {raw!r}")
    return raw


def _num(at: _At, raw: Any) -> float:
    if isinstance(raw, bool) or not isinstance(raw, int | float) or not math.isfinite(raw):
        at.fail(f"must be a finite number, got {raw!r}")
    return float(raw)


def _positive(at: _At, raw: Any) -> float:
    value = _num(at, raw)
    if value <= 0:
        at.fail(f"must be above 0, got {raw!r}")
    return value


def _vec(at: _At, raw: Any, n: int, *, nonzero: bool = False) -> tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) != n:
        at.fail(f"must be a list of {n} numbers, got {raw!r}")
    vec = tuple(_num(at.index(i), v) for i, v in enumerate(raw))
    if nonzero and not any(vec):
        at.fail("must not be the zero vector")
    return vec


def _box(at: _At, raw: Any) -> Box:
    if not isinstance(raw, list) or len(raw) != 2:
        at.fail(f"must be [[min x, y, z], [max x, y, z]], got {raw!r}")
    lo, hi = _vec(at.index(0), raw[0], 3), _vec(at.index(1), raw[1], 3)
    if any(a > b for a, b in zip(lo, hi)):
        at.fail(f"min {list(lo)} is above max {list(hi)}")
    return lo, hi


def _windows(at: _At, raw: Any, frames: tuple[int, int]) -> tuple[Window, ...]:
    windows = []
    for i, win in enumerate(_list(at, raw, nonempty=True)):
        t0, t1 = _vec(at.index(i), win, 2)
        if (why := window_problem((t0, t1), frames)) is not None:
            at.index(i).fail(why)
        windows.append((t0, t1))
    return tuple(windows)


def _why(at: _At, raw: dict) -> str:
    if "why" not in raw:
        at.fail("missing field 'why': say why")
    return _text(at.key("why"), raw["why"])


class _RepeatedKey(Exception):
    pass


def _no_repeats(pairs: list[tuple[str, Any]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise _RepeatedKey(key)
        obj[key] = value
    return obj


def _read_json(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_no_repeats)
    except _RepeatedKey as exc:
        raise ManifestError(f"{path}: (json): key {exc.args[0]!r} repeats in one object") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestError(f"{path}: cannot read the {what} file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: the {what} file is not valid JSON: {exc}") from exc


def _unique(at: _At, ids: list[str], what: str = "ids") -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            at.fail(f"{what} repeat: {item!r}")
        seen.add(item)


def _ref(at: _At, raw: Any, *, bone_ok: bool, self_ok: bool = False) -> Ref:
    try:
        ref = Ref.parse(raw, self_ok=self_ok)
    except ManifestError as exc:
        at.fail(str(exc))
    if ref.is_bone and not bone_ok:
        at.fail(f"must be a <pack>:<piece>.<id> ref, not a bone ref: {raw!r}")
    return ref


def _quat(at: _At, raw: Any) -> Quat:
    return _vec(at, raw, 4, nonzero=True)


def _built[T](at: _At, make: Callable[[], T]) -> T:
    """Build a record with `make`. A record validates itself when built, so a refusal
    (`RecordError`) is re-raised at `at`, naming the file and the field's place."""
    try:
        return make()
    except RecordError as exc:
        if exc.field is None:
            at.fail(f"{exc.where}: {exc.why}")
        at.key(exc.field).fail(exc.why)
