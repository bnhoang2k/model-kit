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
    """A check could not see what it is meant to check, so its result means nothing."""


class Unmeasurable(ManifestError):
    """A measurement needs data the scene does not hold (a part without skin weights). A check
    turns this into `Unmeasured`, so the run is `partial`."""
