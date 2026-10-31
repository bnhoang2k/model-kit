"""Allowlists: each line names a finding to let through, with a reason.

Format: one `check | item | reason` per line. Blank lines and lines that start with `#` are
skipped. `check` and `item` are `fnmatch` patterns, matched case-sensitively. The first line
that matches a finding wins. A line with no reason allows nothing and is itself a failure.
A line that matched nothing in this run is a failure too, so dead lines cannot pile up. Only
lines whose `check` pattern matches a check that ran are judged stale: a run of some checks
cannot judge the lines for the others.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from .metagate import Finding


@dataclass
class _Line:
    check: str
    item: str
    reason: str
    used: bool = False

    @property
    def text(self) -> str:
        return f"{self.check} | {self.item}"


class Allowlist:
    def __init__(self, lines: list[_Line], malformed: list[str]):
        self._lines = lines
        self._malformed = malformed

    @classmethod
    def load(cls, path: str | Path) -> Allowlist:
        lines: list[_Line] = []
        malformed: list[str] = []
        for raw in Path(path).read_text().splitlines():
            text = raw.strip()
            if not text or text.startswith("#"):
                continue
            parts = [p.strip() for p in text.split("|", 2)]
            if len(parts) != 3 or not parts[0] or not parts[1]:
                malformed.append(text)
                continue
            lines.append(_Line(*parts))
        return cls(lines, malformed)

    def apply(self, findings: list[Finding]) -> tuple[list[Finding], list[tuple[Finding, str]]]:
        """Split findings into those that stay failures and those allowed, each with its reason."""
        kept: list[Finding] = []
        allowed: list[tuple[Finding, str]] = []
        for f in findings:
            line = next((ln for ln in self._lines if ln.reason
                         and fnmatch.fnmatchcase(f.check, ln.check)
                         and fnmatch.fnmatchcase(f.item, ln.item)), None)
            if line is None:
                kept.append(f)
            else:
                line.used = True
                allowed.append((f, line.reason))
        return kept, allowed

    def problems(self, ran: Collection[str] | None = None) -> list[Finding]:
        """Call after `apply`: lines with no reason, malformed lines and lines that matched nothing.

        With `ran`, the names of the checks that ran, a line that matched nothing is stale only
        when its `check` pattern matches one of them. None judges every line.
        """
        out = [Finding("allowlist", text, "allowlist line is not `check | item | reason`")
               for text in self._malformed]
        for ln in self._lines:
            judged = ran is None or any(fnmatch.fnmatchcase(name, ln.check) for name in ran)
            if not ln.reason:
                out.append(Finding("allowlist", ln.text, "allowlist line has no reason"))
            elif not ln.used and judged:
                out.append(Finding("allowlist", ln.text,
                                   f'stale: "{ln.text}" matches nothing in this build; delete it'))
        return out
