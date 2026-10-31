"""The meta-gate: the real checks run first, then every check's planted fault, and a check that
cannot see its plant stops the run. A check that passes on a known fault would pass on anything,
so its silence on the real build means nothing. Then the verdict prints.

A plant is scoped. It names the items it planted (`Plant.scope`, at least its own item), and
the meta-gate measures only those items on the planted copy. The finding must still be on the
plant's item. The real run measures every item. So the plants cost one item each, not the
whole check, and a plant whose fault shows only on another item is still blind.

A check builds plants as units (`PlantUnit`), from geometry only and blind to its own verdict.
A unit holds one item and the candidate plants to try, in order. `run_gate` runs the real checks
first and skips a unit whose item already has a real finding. Otherwise it accepts the first
candidate the check catches, and raises `BlindCheck` when it catches none: a check that cannot
see its plant stops the run. A unit that does not apply says why (`PlantUnit.not_applicable`),
and the gate prints that reason as a coverage line.

A plant can also name the samples it touched (`Plant.samples`, indexes into the clip's
samples). A check that works per sample measures just those; empty means every sample. Items
and samples are apart: items are what findings name, samples are how much of a clip to read.

Output: failure and coverage lines to stderr, then exactly one
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
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from .errors import BlindCheck, ManifestError

if TYPE_CHECKING:
    from .allow import Allowlist


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


@runtime_checkable
class Measures(Protocol):
    """The optional hook of a check that measures parts: the part keys it measured in `ctx`."""

    def measured(self, ctx: Any) -> set[str]: ...


@dataclass
class Verdict:
    status: Literal["passed", "partial", "failed"]
    failed: int
    lines: list[str]
    coverage: list[str]
    allowed: list[str] = field(default_factory=list)


EXIT = {"passed": 0, "failed": 1, "partial": 1}


def run_gate(checks: list[Check], ctx: Any, allow: Allowlist | None) -> Verdict:
    """Run the real checks, then every plant unit (raises `BlindCheck` on a unit none of whose
    candidates is caught). A unit whose item already has a real finding is skipped."""
    if not checks:
        return Verdict("partial", 0, ["no checks ran"], [])
    real = {id(check): check.run(ctx) for check in checks}
    coverage: list[str] = []
    for check in checks:
        coverage.extend(check.coverage(ctx))
        failing = {f.item for f in real[id(check)]}
        for unit in check.plants(ctx):
            if unit.not_applicable:
                coverage.append(f"{check.name}: not applicable ({unit.not_applicable}): "
                                f"{unit.item}")
            elif unit.item not in failing and not any(
                    _caught(check, plant) for plant in unit.candidates):
                raise BlindCheck(f"{check.name}: {unit.item}")
    findings = [f for check in checks for f in real[id(check)]]
    allowed: list[str] = []
    if allow is not None:
        findings, allowed_pairs = allow.apply(findings)
        allowed = [f"allowed: {f.line()} (reason: {reason})" for f, reason in allowed_pairs]
        findings = findings + allow.problems(ran=[check.name for check in checks])
    unmeasured = any(isinstance(line, Unmeasured) for line in coverage)
    status = "partial" if unmeasured else "failed" if findings else "passed"
    return Verdict(status, len(findings), [f.line() for f in findings], coverage, allowed)


def _caught(check: Check, plant: Plant) -> bool:
    """The check, run on the planted copy over the plant's scope, reports on the plant's item."""
    found = check.run(plant.ctx, only=plant.scope, samples=plant.samples)
    return any(f.item == plant.item for f in found)


def emit(verdict: Verdict) -> int:
    """Print failure, allowed and coverage lines to stderr and one status line to stdout."""
    for line in verdict.lines:
        print(line, file=sys.stderr)
    for line in verdict.allowed:
        print(line, file=sys.stderr)
    for line in verdict.coverage:
        print(f"coverage: {line}", file=sys.stderr)
    status = "failed" if verdict.status == "passed" and verdict.failed > 0 else verdict.status
    print(f"failed: {verdict.failed}" if status == "failed" else status)
    return EXIT[status]


def exit_code_for(exc: BaseException) -> int:
    """Exit code for an error that stops the run before a verdict: 2, cannot start."""
    if isinstance(exc, (ManifestError, BlindCheck)):
        return 2
    raise exc
