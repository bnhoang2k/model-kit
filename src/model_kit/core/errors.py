from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metagate import Verdict


class ManifestError(Exception):
    """A manifest is malformed or breaks a contract."""


class RecordError(ManifestError):
    """A record refused by its own validator (`View`, `Clip`, `Part`, `Style`). It keeps `where`
    (a short name for the record), `field` (the field at fault, or None for the record as a
    whole) and `why`, so a loader can re-raise it at the file and the field's place
    (`manifest.fields._built`) while a record built in code reads `<where>: <field>: <why>`."""

    def __init__(self, where: str, field: str | None, why: str) -> None:
        self.where, self.field, self.why = where, field, why
        super().__init__(f"{where}: {field}: {why}" if field else f"{where}: {why}")

    def __reduce__(self):  # a process pool or a test runner copies the error across a pipe
        return type(self), (self.where, self.field, self.why)


class AnchorMissing(ManifestError):
    """A named anchor the manifest requires is not present."""


class BlindCheck(Exception):
    """A check could not see what it is meant to check, so its result means nothing.

    `blind` holds one `"<check>: <item>"` per blind unit; the message joins them. `verdict` is
    what the run found before it stopped, so the findings are not lost with the blind items."""

    def __init__(self, message: str, verdict: Verdict | None = None,
                 blind: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.verdict = verdict
        self.blind = blind


class Unmeasurable(ManifestError):
    """A measurement needs data the scene does not hold (a part without skin weights). A check
    turns this into `Unmeasured`, so the run is `partial`."""


class TooManySamples(Unmeasurable):
    """Sampling a surface at the asked spacing would make more points than the kit's cap
    (`tolerances.MAX_SURFACE_SAMPLES`), so the points are never made. `points` is the most the
    lattice could hold, `spacing` the asked spacing. It is `Unmeasurable`: a check that catches it
    writes an `Unmeasured` line (the run is `partial`), and one that does not stops the gate with
    the message (exit 2), never with the memory the points would take."""

    def __init__(self, points: float, spacing: float, cap: float, lead: str = "",
                 note: str = "") -> None:
        self.points, self.spacing, self.cap = points, spacing, cap
        self.lead, self.note = lead, note
        super().__init__(f"{lead}sampling its surface {spacing:.3g} apart takes up to "
                         f"{points:.3g} points, over the kit's cap of {cap:.3g}"
                         + (f" ({note})" if note else ""))

    def __reduce__(self):  # a process pool or a test runner copies the error across a pipe
        return type(self), (self.points, self.spacing, self.cap, self.lead, self.note)
