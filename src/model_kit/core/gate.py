"""The gate command: run named checks on a saved scene folder.

    python -m model_kit.core.gate <scene> --checks must-touch,reach,trace [--allow FILE]
                                  [--timing FILE] [--pieces KEYS]

`<scene>` is a folder written in `core.scene`'s layout. `--checks` names the checks to run,
comma-separated; left out, every registered check runs. An empty list runs none, and a run with
no checks is `partial`. `coverage` is built last from the other checks named (`build_checks`),
because it fails the parts that none of them measured. `model-kit gate` (`model_kit.cli`) runs
this same code. `--allow` names an allowlist (`core.allow`). `--timing` appends one JSON line per
check and phase (real, coverage, plants) to FILE: the scene, the kit version, the check, the
phase, the seconds, the one-minute load average and the pieces of a scoped run (null for a full
run). Its folder must exist, or the run cannot start. Timing is recorded, never judged.
`--pieces` takes piece keys `<pack>:<piece>`, comma-separated, and scopes the run (`core.scope`):
the checks that judge each piece alone measure only those pieces, and the layout, clip and export
checks still read the whole scene. A scoped run is never a pass: it ends `partial` (exit 1). An
empty, repeated or unknown key cannot start, before any check runs. The output and exit codes
are `metagate.emit`'s: coverage, time, allowed and failure lines to stderr, in that order (failures
last), then one stdout line. A run that cannot start (a usage error, an unknown check, a bad or
missing file, or any other error before a verdict) prints one `cannot start: <type>: <message>` line
to stderr, nothing to
stdout, and exits 2. A blind check also exits 2, but the run first tries every plant: it prints
its coverage, time, allowed and failure lines (`metagate.emit_lines`), then prints the
`cannot start` line last. Each `--timing` row is written the moment its phase ends, so a run
that stops keeps the rows of the phases that ended.

This module is the only registry of check names.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

from model_kit.core.allow import Allowlist
from model_kit.core.checks.bake_files import BakeFiles
from model_kit.core.checks.bearing import Bearing
from model_kit.core.checks.bend_quality import BendQuality
from model_kit.core.checks.branching import BRANCHING_CHECKS
from model_kit.core.checks.clearance import Clearance
from model_kit.core.checks.clip_hygiene import ClipHygiene
from model_kit.core.checks.connectors import Connectors
from model_kit.core.checks.coplanar import Coplanar
from model_kit.core.checks.coverage import Coverage
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.detail import Detail
from model_kit.core.checks.done import Done
from model_kit.core.checks.export_hygiene import ExportHygiene
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.frames import Frames
from model_kit.core.checks.inset_seat import InsetSeat
from model_kit.core.checks.keep_off import KeepOffCheck
from model_kit.core.checks.lightmap_density import LightmapDensity
from model_kit.core.checks.lightmap_overlap import LightmapOverlap
from model_kit.core.checks.markings import Markings
from model_kit.core.checks.markings_flush import MarkingsFlush
from model_kit.core.checks.mirror_exact import MirrorExact
from model_kit.core.checks.mirror_pair import MirrorPair
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.checks.near_crease import NearCrease
from model_kit.core.checks.no_fallback import NoFallback
from model_kit.core.checks.overlap import Overlap
from model_kit.core.checks.reach import Reach
from model_kit.core.checks.receiver import Receiver
from model_kit.core.checks.seam_jump import SeamJump
from model_kit.core.checks.seat_fit import SeatFit
from model_kit.core.checks.sides import Sides
from model_kit.core.checks.solver_error import SolverError
from model_kit.core.checks.stress_poses import StressPoses
from model_kit.core.checks.take_apart import TakeApart
from model_kit.core.checks.topology import Topology
from model_kit.core.checks.trace import Trace
from model_kit.core.errors import BlindCheck, ManifestError
from model_kit.core.metagate import Check, Timing, emit, emit_lines, exit_code_for, run_gate
from model_kit.core.scene import load_scene
from model_kit.core.scope import Scope, parse_pieces

# Coverage is registered last: it is built from the other checks of a run (`build_checks`).
CHECKS: dict[str, Callable[[], Check]] = {
    c.name: c
    for c in (MustTouch, Reach, Trace, Crossings, TakeApart, Floaters, Topology, Frames, Sides,
              MirrorPair, Done, NoFallback, Overlap, SolverError, ClipHygiene, *BRANCHING_CHECKS,
              Receiver, KeepOffCheck, Connectors, ExportHygiene,
              MirrorExact,
              SeatFit,
              Clearance,
              StressPoses,
              Markings,
              NearCrease,
              BendQuality,
              InsetSeat,
              Bearing,
              Detail,
              MarkingsFlush,
              Coplanar,
              BakeFiles,
              LightmapOverlap,
              LightmapDensity,
              SeamJump,
              Coverage)
}

CANNOT_START = 2


def check_names(text: str) -> list[str]:
    """The comma-separated check names, in order. An unknown or repeated name raises."""
    names = [name.strip() for name in text.split(",") if name.strip()]
    unknown = [name for name in names if name not in CHECKS]
    if unknown:
        raise ManifestError(f"unknown check {', '.join(map(repr, unknown))}; the checks are "
                            f"{', '.join(CHECKS)}")
    if len(set(names)) != len(names):
        raise ManifestError(f"a check is named twice: {names}")
    return names


def build_checks(names: list[str], scope: Scope | None = None) -> list[Check]:
    """The checks of a run, in order, with coverage built last. Coverage reads what the other
    checks of the run measured, so it takes them as its `others` (and the run's `scope`); named
    alone it has none."""
    checks = [CHECKS[name]() for name in names if name != Coverage.name]
    if Coverage.name in names:
        checks.append(Coverage(checks, scope))
    return checks


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The gate's arguments, shared by this module's parser and `model-kit gate`."""
    parser.add_argument("scene", type=Path, help="the scene folder")
    parser.add_argument("--checks",
                        help=f"comma-separated check names (default: all): {', '.join(CHECKS)}")
    parser.add_argument("--allow", type=Path, help="an allowlist: `check | item | reason` lines")
    parser.add_argument("--timing", type=Path,
                        help="append one JSON line per check and phase (seconds) to this file")
    parser.add_argument("--pieces",
                        help="comma-separated piece keys <pack>:<piece>: the part checks measure "
                             "only these; the run is partial")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m model_kit.core.gate",
                                     description="Run named checks on a saved scene folder.")
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate and return its exit code: 0 passed, 1 failed or partial, 2 cannot start."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exc:  # argparse has printed the usage error
        return CANNOT_START if exc.code else 0
    return run(args)


def run(args: argparse.Namespace) -> int:
    """Run the gate for parsed arguments and return its exit code."""
    try:
        names = list(CHECKS) if args.checks is None else check_names(args.checks)
        if not args.scene.is_dir():
            raise ManifestError(f"{args.scene}: no scene folder there")
        allow = None
        if args.allow is not None:
            if not args.allow.is_file():
                raise ManifestError(f"{args.allow}: no allowlist file there")
            allow = Allowlist.load(args.allow)
        if args.timing is not None and not args.timing.parent.is_dir():
            raise ManifestError(f"{args.timing}: the timing file's folder is not there")
        ctx = load_scene(args.scene)
        scope = None if args.pieces is None else parse_pieces(args.pieces, ctx)
        checks = build_checks(names, scope)
        pieces = None if scope is None else sorted(scope.pieces)
        on_timing = (None if args.timing is None
                     else _timing_writer(args.timing, ctx.folder, pieces))
        verdict = run_gate(checks, ctx, allow, scope=scope, on_timing=on_timing)
    except BlindCheck as exc:  # the findings of the run come first
        if exc.verdict is not None:
            emit_lines(exc.verdict)
        return _cannot_start(exc, exit_code_for(exc))
    except ManifestError as exc:  # AnchorMissing is a ManifestError
        return _cannot_start(exc, exit_code_for(exc))
    except Exception as exc:  # noqa: BLE001 - any other error, e.g. a cut-short samples file
        return _cannot_start(exc, CANNOT_START)
    return emit(verdict)


def _timing_writer(path: Path, scene: str, pieces: list[str] | None) -> Callable[[Timing], None]:
    """Return a callback that appends one flushed JSON line to `path` for each timing, so a run
    that stops keeps the rows of the phases that ended."""
    kit = importlib.metadata.version("model-kit")

    def write(t: Timing) -> None:
        row = {"scene": scene, "kit": kit, "check": t.check, "phase": t.phase,
               "seconds": t.seconds, "loadavg_1m": os.getloadavg()[0], "pieces": pieces}
        with path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
            log.flush()

    return write


def _cannot_start(exc: Exception, code: int) -> int:
    message = " ".join(str(exc).split())  # one line, whatever the error's text holds
    print(f"cannot start: {type(exc).__name__}: {message}", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
