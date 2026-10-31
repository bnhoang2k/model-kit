"""Growth rules: every number the grower reads names the rule rows that justify it.

A rules file holds a species, its sources, its rule rows and its params. A row says a rule in
words and cites a source by `[key]`. A param holds one number (or a range, a list of choices, a
list of names, or a list of `[x, y]` points) and names the rows behind it. A row that no param
or range cites must be listed in `notes`. Errors read `<file>: <field>: <why>`, like the
manifest loader.

A kit file (`KIT_RULES`) sits beside the species file. The kit file holds what is true of
every tree (tag `fact`) and the constants of the growth model (tag `model`). Every species
param needs a tag (`measured`, `fitted`, `budget` or `model`), a `fitted` param names the ranges
it was fitted to and its seeds, and a species file may not set a kit key. `budget` is a cost
limit, not a fact (how many root flares the page draws). `model` is a choice of the growth
model with no source behind it.

A tag that claims a source (`measured` in a species file, `fact` in the kit file) needs one: every
row such a param cites holds at least one `[key]` cite. A number with no source is `model`, in
either file, and its row says so.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from model_kit.core.errors import ManifestError
from model_kit.core.manifest.fields import (
    _At,
    _fields,
    _list,
    _name,
    _names,
    _num,
    _obj,
    _read_json,
    _text,
    _unique,
)
from model_kit.core.manifest.records import _NAME

# The shipped kit-wide file: facts true of every tree and the constants of the growth model.
KIT_RULES = Path(__file__).resolve().parents[2] / "kinds" / "branching" / "botany.rules.json"

SPECIES_TAGS = ("measured", "fitted", "budget", "model")
KIT_TAGS = ("model", "fact")
# The reference quantities a species file may give a cited range for (`reference-range` reads them).
RANGE_KEYS = ("spread_of_height", "crown_base_of_height", "fork_of_height")
# A fitted param is fitted on these seeds; held-out seeds check it.
FIT_SEEDS = (1, 2, 3, 4)
DIVERGENCE_KEY = "leaves.divergence_deg"
# The tags whose rows must cite a source.
CITED_TAGS = ("measured", "fact")
# The buds-v1 tube sides. A rules file without `tubes.sides` grows buds-v2, whose sides come from
# the piece's view when the tree is meshed, so it may hold none of these.
TUBE_SIDES_KEY = "tubes.sides"
LITE_TUBE_SIDES_KEY = "lite.tube_sides"
TUBE_SIDES_ROWS = ("tubes.sides", "lite.tree-line")
KIT_SPECIES = "kit"

_CITE = re.compile(r"\[([^\]]*)\]")
_PARAM_KEY = re.compile(r"[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)+")

# A param value: a number, a tuple of numbers, a tuple of names or a tuple of (x, y) tuples.
Value = Any


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    gives: str


@dataclass(frozen=True)
class RuleRow:
    """One cited rule. `id` is `<level>.<slug>`; `rule` and `source` are free text.

    `source` holds `[key]` cites of the sources."""

    id: str
    level: str
    rule: str
    value: str
    source: str


@dataclass(frozen=True)
class Fit:
    """What a `fitted` param was fitted to: range keys, and the seeds the fit used."""

    ranges: tuple[str, ...]
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class Param:
    key: str
    value: Value
    rows: tuple[str, ...]
    tag: str | None = None
    fit: Fit | None = None


@dataclass(frozen=True)
class Range:
    """A cited reference range for one quantity of a grown tree: `lo <= value <= hi`."""

    key: str
    lo: float
    hi: float
    rows: tuple[str, ...]


@dataclass(frozen=True)
class GrowthRules:
    species: str
    sources: dict[str, Source]
    rows: dict[str, RuleRow]
    notes: tuple[str, ...]
    params: dict[str, Param]
    path: Path
    ranges: dict[str, Range] = field(default_factory=dict)
    kit_keys: frozenset[str] = frozenset()
    sha256: str = ""
    kit_sha256: str = ""


def _number(at: _At, raw: Any) -> int | float:
    _num(at, raw)  # rejects bools, non-numbers and non-finite numbers
    return raw


def _is_number(raw: Any) -> bool:
    return isinstance(raw, int | float) and not isinstance(raw, bool)


def _value(at: _At, raw: Any) -> Value:
    if _is_number(raw):
        return _number(at, raw)
    if isinstance(raw, list) and raw:
        if all(_is_number(v) for v in raw):
            return tuple(_number(at.index(i), v) for i, v in enumerate(raw))
        if all(isinstance(v, str) for v in raw):
            return _names(at, raw)
        if all(isinstance(v, list) for v in raw):
            points = []
            for i, pair in enumerate(raw):
                if len(pair) != 2:
                    at.index(i).fail(f"a point must be [x, y], got {pair!r}")
                points.append(tuple(_number(at.index(i).index(k), v) for k, v in enumerate(pair)))
            return tuple(points)
    at.fail(
        "must be a number, a list of numbers, a list of names or a list of [x, y] points, "
        f"got {raw!r}"
    )


def _sources(at: _At, raw: Any) -> dict[str, Source]:
    items = _list(at, raw, nonempty=True)
    sources = []
    for i, item in enumerate(items):
        sat = at.index(i)
        fields = _fields(sat, item, ("key", "url", "gives"))
        sources.append(
            Source(
                key=_name(sat.key("key"), fields["key"]),
                url=_text(sat.key("url"), fields["url"]),
                gives=_text(sat.key("gives"), fields["gives"]),
            )
        )
    _unique(at, [s.key for s in sources], "source keys")
    return {s.key: s for s in sources}


def _rows(at: _At, raw: Any, sources: dict[str, Source]) -> dict[str, RuleRow]:
    items = _list(at, raw, nonempty=True)
    rows = []
    for i, item in enumerate(items):
        rat = at.index(i)
        fields = _fields(rat, item, ("id", "level", "rule", "value", "source"))
        level = _name(rat.key("level"), fields["level"])
        rule = _text(rat.key("rule"), fields["rule"])
        row_id = _text(rat.key("id"), fields["id"])
        slug = row_id.removeprefix(f"{level}.")
        if slug == row_id or not re.fullmatch(_NAME, slug):
            rat.key("id").fail(
                f"must be '<level>.<slug>' with level {level!r} and a slug of letters, "
                f"digits, '_' or '-', got {row_id!r}"
            )
        text = _text(rat.key("source"), fields["source"])
        for cite in _CITE.findall(text):
            if cite not in sources:
                rat.key("source").fail(f"cites {cite!r}, which is not a key in sources")
        rows.append(RuleRow(row_id, level, rule, _text(rat.key("value"), fields["value"]), text))
    _unique(at, [r.id for r in rows], "row ids")
    return {r.id: r for r in rows}


def _tag(at: _At, raw: Any, tags: tuple[str, ...], *, is_kit: bool) -> str:
    if raw not in tags:
        where = "the kit file" if is_kit else "a species file"
        at.fail(f"must be one of {list(tags)} in {where}, got {raw!r}")
    return raw


def _fit(at: _At, raw: Any) -> Fit:
    fields = _fields(at, raw, ("ranges", "seeds"))
    rat = at.key("ranges")
    names = tuple(_name(rat.index(i), v) for i, v in enumerate(_list(rat, fields["ranges"], nonempty=True)))
    _unique(rat, list(names), "fit ranges")
    sat = at.key("seeds")
    seeds = tuple(int(_num(sat.index(i), v)) for i, v in enumerate(_list(sat, fields["seeds"])))
    if seeds != FIT_SEEDS:
        sat.fail(f"must be {', '.join(map(str, FIT_SEEDS))}, got {list(seeds)}")
    return Fit(names, seeds)


def _check_row_ids(at: _At, raw: Any, rows: dict[str, RuleRow], notes: tuple[str, ...]) -> tuple[str, ...]:
    ids = []
    for i, row_id in enumerate(_list(at, raw, nonempty=True)):
        row_id = _text(at.index(i), row_id)
        if row_id not in rows:
            at.index(i).fail(f"cites {row_id!r}, which is not a row id")
        if row_id in notes:
            at.index(i).fail(f"cites {row_id!r}, which is a note")
        ids.append(row_id)
    _unique(at, ids, "cited rows")
    return tuple(ids)


def _ranges(
    at: _At, raw: Any, rows: dict[str, RuleRow], notes: tuple[str, ...]
) -> dict[str, Range]:
    ranges = {}
    for key, spec in _obj(at, raw).items():
        kat = at.key(key)
        if key not in RANGE_KEYS:
            kat.fail(f"is not a range key; the keys are {list(RANGE_KEYS)}")
        fields = _fields(kat, spec, ("lo", "hi", "rows"))
        lo = _num(kat.key("lo"), fields["lo"])
        hi = _num(kat.key("hi"), fields["hi"])
        if lo > hi:
            kat.fail(f"lo {lo!r} is above hi {hi!r}")
        ranges[key] = Range(key, lo, hi, _check_row_ids(kat.key("rows"), fields["rows"], rows, notes))
    return ranges


def _parse(path: Path, *, is_kit: bool, kit_keys: frozenset[str]) -> GrowthRules:
    """Read one rules file. `kit_keys` names the kit's params (empty for the kit file itself)."""
    path = Path(path)
    at = _At(path)
    raw = _fields(
        at,
        _read_json(path, "growth rules"),
        ("species", "sources", "rows", "notes", "params"),
        ("ranges",),
    )
    species = _name(at.key("species"), raw["species"])
    if is_kit and species != KIT_SPECIES:
        at.key("species").fail(f"the kit file's species must be {KIT_SPECIES!r}, got {species!r}")
    if not is_kit and species == KIT_SPECIES:
        at.key("species").fail(
            f"{KIT_SPECIES!r} is the kit-wide rules file, not a species file: "
            "load it as `kit=`, with a species file"
        )
    sources = _sources(at.key("sources"), raw["sources"])
    rows = _rows(at.key("rows"), raw["rows"], sources)

    nat = at.key("notes")
    notes = []
    for i, note in enumerate(_list(nat, raw["notes"])):
        note = _text(nat.index(i), note)
        if note not in rows:
            nat.index(i).fail(f"names no row: {note!r}")
        notes.append(note)
    _unique(nat, notes, "notes")

    ranges: dict[str, Range] = {}
    if "ranges" in raw:
        if is_kit:
            at.key("ranges").fail("the kit file has no ranges; a range belongs to one species")
        ranges = _ranges(at.key("ranges"), raw["ranges"], rows, tuple(notes))

    tags = KIT_TAGS if is_kit else SPECIES_TAGS
    pat = at.key("params")
    params: dict[str, Param] = {}
    cited: set[str] = {row_id for r in ranges.values() for row_id in r.rows}
    for key, spec in _obj(pat, raw["params"]).items():
        kat = pat.key(key)
        if not _PARAM_KEY.fullmatch(key):
            kat.fail("param keys look like '<level>.<name>'")
        if key in kit_keys and not is_kit:
            kat.fail(f"{key!r} is a kit key: kit-wide facts and model constants cannot be set per species")
        fields = _fields(kat, spec, ("value", "rows"), ("tag", "fit"))
        if "tag" not in fields:
            kat.fail(f"missing field 'tag' (one of {list(tags)})")
        tag = _tag(kat.key("tag"), fields["tag"], tags, is_kit=is_kit)
        fit = None
        if tag == "fitted":
            if "fit" not in fields:
                kat.fail("a fitted param needs a 'fit' field: the ranges it was fitted to and its seeds")
            fit = _fit(kat.key("fit"), fields["fit"])
            for i, name in enumerate(fit.ranges):
                if name not in ranges:
                    kat.key("fit").key("ranges").index(i).fail(
                        f"names {name!r}, which is not one of this file's ranges {list(ranges)}"
                    )
        elif "fit" in fields:
            kat.fail("has a 'fit' field but its tag is not 'fitted'")
        for cited_id in fields["rows"] if isinstance(fields["rows"], list) else []:
            if cited_id in notes:
                nat.index(notes.index(cited_id)).fail(f"{cited_id!r} is a note, but param {key!r} cites it")
        row_ids = _check_row_ids(kat.key("rows"), fields["rows"], rows, tuple(notes))
        if tag in CITED_TAGS:
            for i, row_id in enumerate(row_ids):
                if not _CITE.search(rows[row_id].source):
                    kat.key("rows").index(i).fail(
                        f"row {row_id!r} has no [key] cite in its source, but the param is "
                        f"tagged {tag!r}: cite a source, or tag it 'model' (no source) and say "
                        "so in the row"
                    )
        cited.update(row_ids)
        value = _value(kat.key("value"), fields["value"])
        if key == DIVERGENCE_KEY and not (_is_number(value) and 0 < value <= 180):
            kat.key("value").fail(f"a divergence angle must be a turn in (0, 180] degrees, got {value!r}")
        params[key] = Param(key, value, row_ids, tag, fit)
    if not params:
        pat.fail("must not be empty")

    if not is_kit and TUBE_SIDES_KEY not in params:
        stranded = [k for k in (LITE_TUBE_SIDES_KEY,) if k in params]
        stranded += [r for r in TUBE_SIDES_ROWS if r in rows or r in notes]
        if stranded:
            at.fail(
                f"{', '.join(map(repr, stranded))}: buds-v1 tube-sides keys, but this file has no "
                f"{TUBE_SIDES_KEY!r}, so it is a buds-v2 file: its tube sides come from the "
                "piece's view when it is meshed. Drop it, or add 'tubes.sides' for buds-v1"
            )

    fitted = [k for k, p in params.items() if p.tag == "fitted"]
    if len(fitted) > len(ranges):
        pat.fail(
            f"{len(fitted)} fitted params {fitted} but {len(ranges)} ranges: "
            "fitted params must not outnumber the ranges they are fitted to"
        )

    for row_id in rows:
        if row_id not in cited and row_id not in notes:
            at.key("rows").fail(f"row {row_id!r}: no param cites it and it is not in notes")
    return GrowthRules(
        species, sources, rows, tuple(notes), params, path, ranges,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )  # fmt: skip


def _merge_kit(species: GrowthRules, kit: GrowthRules) -> GrowthRules:
    """The kit's sources, rows and params first, then the species'. A shared name raises."""
    for what, mine, theirs in (
        ("source key", species.sources, kit.sources),
        ("row id", species.rows, kit.rows),
        ("param key", species.params, kit.params),
    ):
        for name in mine:
            if name in theirs:
                raise ManifestError(
                    f"{species.path}: {what} {name!r} is also in the kit file {kit.path.name}"
                )
    return GrowthRules(
        species.species,
        {**kit.sources, **species.sources},
        {**kit.rows, **species.rows},
        species.notes + kit.notes,
        {**kit.params, **species.params},
        species.path,
        species.ranges,
        frozenset(kit.params),
        species.sha256,
        kit.sha256,
    )


def load_rules(path: Path, *, kit: Path = KIT_RULES) -> GrowthRules:
    """Read a species rules file and merge the kit file into it. A bad cite, an unused row or a
    repeated key raises. The result holds the kit's sources, rows and params with the species',
    every param is tagged and `kit_keys` names the kit's params."""
    kit_rules = _parse(kit, is_kit=True, kit_keys=frozenset())
    species = _parse(path, is_kit=False, kit_keys=frozenset(kit_rules.params))
    return _merge_kit(species, kit_rules)


class RuleReader:
    """Reads params from one `GrowthRules` and remembers every key read.

    There are no defaults: a missing key raises, and so does a value of the wrong kind.
    """

    def __init__(self, rules: GrowthRules) -> None:
        self.rules = rules
        self._read: set[str] = set()

    @property
    def read(self) -> frozenset[str]:
        return frozenset(self._read)

    def _fail(self, key: str, why: str):
        raise ManifestError(f"{self.rules.path}: params: {key!r}: {why}")

    def _get(self, key: str) -> Value:
        param = self.rules.params.get(key)
        if param is None:
            raise ManifestError(f"{self.rules.path}: params: no {key!r}")
        self._read.add(key)
        return param.value

    def _numbers(self, key: str, want: str) -> tuple[float, ...]:
        value = self._get(key)
        if not (isinstance(value, tuple) and all(_is_number(v) for v in value)):
            self._fail(key, f"is not {want}, got {value!r}")
        return value

    def _whole(self, key: str, value: float) -> int:
        if value != int(value):
            self._fail(key, f"must be a whole number, got {value!r}")
        return int(value)

    def num(self, key: str) -> float:
        value = self._get(key)
        if not _is_number(value):
            self._fail(key, f"is not a number, got {value!r}")
        return float(value)

    def int(self, key: str) -> int:
        value = self._get(key)
        if not _is_number(value):
            self._fail(key, f"is not a number, got {value!r}")
        return self._whole(key, value)

    def range(self, key: str) -> tuple[float, float]:
        value = self._numbers(key, "a [lo, hi] range")
        if len(value) != 2 or value[0] > value[1]:
            self._fail(key, f"is not a [lo, hi] range, got {list(value)!r}")
        return float(value[0]), float(value[1])

    def int_range(self, key: str) -> tuple[int, int]:
        lo, hi = self.range(key)
        return self._whole(key, lo), self._whole(key, hi)

    def choices(self, key: str) -> tuple[float, ...]:
        return tuple(float(v) for v in self._numbers(key, "a list of numbers"))

    def int_choices(self, key: str) -> tuple[int, ...]:
        return tuple(self._whole(key, v) for v in self._numbers(key, "a list of numbers"))

    def names(self, key: str) -> tuple[str, ...]:
        value = self._get(key)
        if not (isinstance(value, tuple) and all(isinstance(v, str) for v in value)):
            self._fail(key, f"is not a list of names, got {value!r}")
        return value

    def points(self, key: str) -> np.ndarray:
        value = self._get(key)
        if not (
            isinstance(value, tuple) and all(isinstance(v, tuple) and len(v) == 2 for v in value)
        ):
            self._fail(key, f"is not a list of [x, y] points, got {value!r}")
        return np.array(value, dtype=float)

    def metres(self, key: str, upm: float) -> float:
        """A value in metres, as project units: `upm` is the style's units per metre."""
        return self.num(key) * upm

    def metres_range(self, key: str, upm: float) -> tuple[float, float]:
        lo, hi = self.range(key)
        return lo * upm, hi * upm
