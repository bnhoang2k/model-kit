"""The meta-gate: the real checks run first, then every check's planted fault, and a check that
cannot see its plant stops the run. A check that passes on a known fault would pass on anything,
so its silence on the real build means nothing. A blind run still tries every plant of every
check and keeps its findings: `BlindCheck` lists every blind item and carries the verdict, and
the gate command prints those lines before it stops. Then the verdict prints.

A plant is scoped. It names the items it planted (`Plant.scope`, at least its own item), and
the meta-gate measures only those items on the planted copy. The finding must still be on the
plant's item. The real run measures every item. So the plants cost one item each, not the
whole check, and a plant whose fault shows only on another item is still blind.

A check builds plants as units (`PlantUnit`), from geometry or its record, never from its own
verdict on the planted copy. A check may use its real verdict to choose where to plant: a plant
on an item that already fails is caught trivially and proves nothing (`lightmap-density` plants on
a placement whose real data passes).
A unit holds one item and the candidate plants to try, in order. `run_gate` runs the real checks
first and skips a unit whose item already has a real finding. Otherwise it accepts the first
candidate the check catches. A unit none of whose candidates is caught is blind. `run_gate` still
tries every unit, then raises `BlindCheck` once, with the verdict and every blind item. A unit
that does not apply says why (`PlantUnit.not_applicable`), and the gate prints that reason as a
coverage line.

A scoped run (`run_gate(..., scope=...)`, see `core.scope`) measures only the named pieces in
the checks marked `scope = "piece"` (`scope_of`): they judge each piece alone. Every other check
reads the whole scene and runs on it. The first coverage line is an `Unmeasured`, so a scoped run
is `partial`, never `passed`: a gate on the whole scene is the pass.

A plant can also name the samples it touched (`Plant.samples`, indexes into the clip's
samples). A check that works per sample measures just those; empty means every sample. Items
and samples are apart: items are what findings name, samples are how much of a clip to read.

Each check is timed in three phases: its real run, its coverage and its plants. `run_gate` reads
an injected clock around each and keeps the result in `Verdict.timings`; `emit_lines` prints one
`time:` line per check to stderr, after the coverage lines. A caller can pass `on_timing` to
hear each timing the moment its phase ends. Timing is recorded, never judged: no time passes or
fails a run. The gate command writes the log, not this module.

Output: coverage, time, allowed and failure lines to stderr, in that order so the failures are
the last lines above the status (`emit_lines`), then exactly one
stdout line (`passed`, `partial` or `failed: n`). Exit 0 passed, 1 failed or partial,
2 cannot start.

`partial` means a check could not measure something, so the run cannot vouch for the build:
a check marks such a coverage line `Unmeasured` (an item skipped for missing data, such as a
clip with no samples). Any one makes the run partial, whatever the findings.
A plain coverage line (a skip the spec allows, such as a keyed clip's targets) does not.

The coverage check needs to know which exported parts the other checks measured. A piece check
says so through the optional hook `measured(ctx) -> set[str]` (`Measures`). Its keys are
`scene.part_key(pack, piece, part)`: `<pack>:<piece> <part>`, the item of floaters and topology.
A check lists a part when it could judge it, not when it passed it: a part it leaves out by
design (a printed part for crossings, a leaf card for the limb checks), a piece of another recipe
kind and a piece with no geometry are not listed. A check that does not measure parts (a clip
check, a layout check) has no hook.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .errors import BlindCheck, ManifestError

if TYPE_CHECKING:
    from .allow import Allowlist
    from .scope import Scope


class Unmeasured(str):
    """A coverage line for an item a check could not measure for want of data. It reads and
    compares as the plain line; `run_gate` makes the run `partial` when it sees one."""

    __slots__ = ()


@dataclass(frozen=True)
class Finding:
    check: str
    item: str
    detail: str

    def line(self) -> str:
        return f"{self.check} | {self.item} | {self.detail}"


@dataclass
class Plant:
    """A faulted copy of the context. The check must report a finding on `item`.

    A finding on any other item does not count, so a real fault elsewhere cannot hide a blind plant.
    `scope` names the items the meta-gate measures on the copy; it always holds `item`.
    `samples` names the sample indexes the fault touches; empty means every sample.
    """
    label: str
    item: str
    ctx: Any
    scope: frozenset[str] = frozenset()
    samples: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        self.scope = frozenset(self.scope) | {self.item}
        self.samples = frozenset(self.samples)


@dataclass(frozen=True)
class PlantUnit:
    """One item to plant, with the candidate plants to try in order.

    The candidates are built from geometry only, never from the check's own verdict. The gate
    accepts the first candidate the check catches and stops; none caught is a `BlindCheck`.
    `not_applicable` set means the item has no candidates: the gate prints the reason.
    """
    item: str
    candidates: tuple[Plant, ...]
    not_applicable: str = ""

    def __post_init__(self) -> None:
        if self.not_applicable and self.candidates:
            raise ValueError(f"{self.item}: a not-applicable unit holds no candidates")
        for plant in self.candidates:
            if plant.item != self.item:
                raise ValueError(f"{self.item}: candidate plants {plant.item}; a fault that "
                                 "shows on another item is the blindness the gate rules out")


class Check(Protocol):
    """A check. `kinds` names the recipe kinds whose pieces it measures (`core.kinds`); a check
    that measures no piece's recipe, such as a clip check, holds the empty set. A check that
    measures parts may also define `measured` (see `Measures`)."""

    name: str
    kinds: frozenset[str]
    principle: str = ""
    """The physical or geometric fact the check stands on, in one sentence. Optional: the checks
    written earlier have none; a newer check must (a test fails one without)."""
    source: str = ""
    """Where the fact and its numbers come from: a link or a citation. Optional like `principle`."""
    scope: Literal["piece", "scene", "split"] = "scene"
    """How a scoped run treats the check (`scope_of`). Optional: the default is `scene`."""

    def plants(self, ctx: Any) -> Iterable[PlantUnit]:
        """The plant units, from geometry only."""
        ...
    def run(self, ctx: Any, only: Collection[str] | None = None,
            samples: frozenset[int] = frozenset()) -> list[Finding]:
        """The findings. With `only`, measure just the items it names; None measures every item.

        With `samples`, a check that works per sample measures just those sample indexes; empty
        means every sample. A check with no per-sample work accepts the keyword and ignores it.
        """
        ...
    def coverage(self, ctx: Any) -> list[str]:
        """Lines naming what was skipped; a line for missing data is an `Unmeasured`."""
        ...


def scope_of(check: Any) -> Literal["piece", "scene", "split"]:
    """How a scoped run treats a check. `piece`: it judges each piece alone, so a scoped run
    measures only the named pieces. `scene` (the default): it reads the whole scene. `split`:
    coverage, which reads the whole scene but must not report a dropped piece as unmeasured."""
    return getattr(check, "scope", "scene")


@runtime_checkable
class Measures(Protocol):
    """The optional hook of a check that measures parts: the part keys it measured in `ctx`."""

    def measured(self, ctx: Any) -> set[str]: ...


@dataclass(frozen=True)
class Timing:
    """Seconds one phase of one check took: its real run, its coverage or its plants."""
    check: str
    phase: Literal["real", "coverage", "plants"]
    seconds: float


@dataclass
class Verdict:
    status: Literal["passed", "partial", "failed"]
    failed: int
    lines: list[str]
    coverage: list[str]
    allowed: list[str] = field(default_factory=list)
    timings: list[Timing] = field(default_factory=list)


EXIT = {"passed": 0, "failed": 1, "partial": 1}


def run_gate(checks: list[Check], ctx: Any, allow: Allowlist | None,
             clock: Callable[[], float] = time.perf_counter, *,
             scope: Scope | None = None,
             on_timing: Callable[[Timing], None] | None = None) -> Verdict:
    """Run the real checks, then every plant unit. A unit whose item already has a real finding
    is skipped. Every unit is tried; when any is blind (none of its candidates is caught), the run
    raises one `BlindCheck` that lists them all and carries the verdict the run would have had.
    With `scope`, each `piece` check runs its real run, coverage and plants on `scope.scene(ctx)`,
    built once; every other check runs on `ctx`. `scope.line()` leads the coverage, so the run is
    `partial`. The allowlist hears every check as having run, but a line whose pattern matches a
    `piece` or `split` check is never judged stale, because the full run may use it.
    `Verdict.timings` holds the real, coverage and plants seconds of each check, read from
    `clock`. `on_timing` hears each `Timing` the moment its phase ends, in the order of
    `Verdict.timings`, so a run that stops keeps the phases that ended."""
    if not checks:
        return Verdict("partial", 0, ["no checks ran"], [])
    timings: list[Timing] = []

    def timed(check: Check, phase: str, start: float) -> None:
        timing = Timing(check.name, phase, clock() - start)
        timings.append(timing)
        if on_timing is not None:
            on_timing(timing)

    scoped = scope.scene(ctx) if scope is not None else ctx

    def seen(check: Check) -> Any:
        return scoped if scope_of(check) == "piece" else ctx

    real = {}
    for check in checks:
        start = clock()
        real[id(check)] = check.run(seen(check))
        timed(check, "real", start)
    coverage: list[str] = [scope.line()] if scope is not None else []
    blind: list[str] = []
    for check in checks:
        start = clock()
        coverage.extend(check.coverage(seen(check)))
        timed(check, "coverage", start)
        failing = {f.item for f in real[id(check)]}
        start = clock()
        for unit in check.plants(seen(check)):
            if unit.not_applicable:
                coverage.append(f"{check.name}: not applicable ({unit.not_applicable}): "
                                f"{unit.item}")
            elif unit.item not in failing and not any(
                    _caught(check, plant) for plant in unit.candidates):
                blind.append(f"{check.name}: {unit.item}")
        timed(check, "plants", start)
    findings = [f for check in checks for f in real[id(check)]]
    allowed: list[str] = []
    if allow is not None:
        findings, allowed_pairs = allow.apply(findings)
        allowed = [f"allowed: {f.line()} (reason: {reason})" for f, reason in allowed_pairs]
        spared = [check.name for check in checks
                  if scope is not None and scope_of(check) != "scene"]
        findings = findings + allow.problems(ran=[check.name for check in checks], spared=spared)
    unmeasured = any(isinstance(line, Unmeasured) for line in coverage)
    status = "partial" if unmeasured else "failed" if findings else "passed"
    verdict = Verdict(status, len(findings), [f.line() for f in findings], coverage, allowed,
                      timings)
    if blind:
        raise BlindCheck("; ".join(blind), verdict=verdict, blind=tuple(blind))
    return verdict


def _caught(check: Check, plant: Plant) -> bool:
    """The check, run on the planted copy over the plant's scope, reports on the plant's item."""
    found = check.run(plant.ctx, only=plant.scope, samples=plant.samples)
    return any(f.item == plant.item for f in found)


def emit_lines(verdict: Verdict) -> None:
    """Print the stderr lines: coverage, time, allowed, then failures, so failures are last."""
    for line in verdict.coverage:
        print(f"coverage: {line}", file=sys.stderr)
    seconds = {(t.check, t.phase): t.seconds for t in verdict.timings}
    for name in dict.fromkeys(t.check for t in verdict.timings):
        phases = " | ".join(f"{phase} {seconds[name, phase]:.1f} s"
                            for phase in ("real", "coverage", "plants"))
        print(f"time: {name} | {phases}", file=sys.stderr)
    for line in verdict.allowed:
        print(line, file=sys.stderr)
    for line in verdict.lines:
        print(line, file=sys.stderr)


def emit(verdict: Verdict) -> int:
    """Print the stderr lines (`emit_lines`) and one status line to stdout."""
    emit_lines(verdict)
    status = "failed" if verdict.status == "passed" and verdict.failed > 0 else verdict.status
    print(f"failed: {verdict.failed}" if status == "failed" else status)
    return EXIT[status]


def exit_code_for(exc: BaseException) -> int:
    """Exit code for an error that stops the run before a verdict: 2, cannot start."""
    if isinstance(exc, (ManifestError, BlindCheck)):
        return 2
    raise exc
