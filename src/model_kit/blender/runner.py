"""Run a script in headless Blender and read its log honestly.

Blender `--background` exits 0 even when a Python script prints a traceback and carries on, so
the exit code alone cannot be trusted. A run is `ok` only when the code is 0, the log holds no
`Traceback`, and the script printed the line `MODEL_KIT_DONE` (call `done()` as its last step).

This module uses the standard library only, so a script inside Blender can import it too (for
`script_args()` and `done()`). The exporters need Blender 4.4 or later (action slots); the kit
is tested with 5.2. A log byte that is not UTF-8 is replaced, never fatal.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from model_kit.core.errors import ManifestError

DONE_LINE = "MODEL_KIT_DONE"
BLENDER_ENV = "MODEL_KIT_BLENDER"

# The folder that holds the `model_kit` package. Blender ignores PYTHONPATH, so the runner
# puts this folder on Blender's `sys.path` before the script runs.
KIT_SRC = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunResult:
    ok: bool
    code: int
    log: str  # stdout and stderr, interleaved as Blender wrote them


def blender_path() -> Path:
    """The Blender executable, from `MODEL_KIT_BLENDER`. Unset or missing raises."""
    value = os.environ.get(BLENDER_ENV)
    if not value:
        raise ManifestError(f"set {BLENDER_ENV} to the Blender executable")
    path = Path(value)
    if not path.is_file():
        raise ManifestError(f"{BLENDER_ENV}={value}: no file there")
    return path


def _ok(code: int, log: str) -> bool:
    return (code == 0 and "Traceback" not in log
            and any(line.strip() == DONE_LINE for line in log.splitlines()))


def run_blender(script: Path, args: list[str], *, blend: Path | None = None,
                timeout: float) -> RunResult:
    """Run `script` in headless Blender with `args` after `--`.

    Blender starts from factory settings, opens `blend` when given (pass a `frozen_copy`, never
    a source file), and exits 1 on an uncaught Python error. A timeout is a failed run whose
    log ends with a note saying so.
    """
    cmd = [str(blender_path()), "--background", "--factory-startup", "--python-exit-code", "1"]
    if blend is not None:
        cmd.append(str(blend))
    cmd += ["--python-expr", f"import sys; sys.path.insert(0, {str(KIT_SRC)!r})",
            "--python", str(script), "--", *args]
    try:
        done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return RunResult(ok=False, code=-1,
                         log=f"{out}\nmodel_kit: Blender timed out after {timeout} s\n")
    return RunResult(ok=_ok(done.returncode, done.stdout), code=done.returncode, log=done.stdout)


def frozen_copy(src: Path, dest_dir: Path) -> Path:
    """Copy a `.blend` into `dest_dir` and make the copy read-only. Returns the copy's path.

    Blender runs read the copy, so a run never writes into a source file and two runs never
    race on one. A missing source, a copy that already exists, or a copy onto the source
    itself raises.
    """
    src, dest_dir = Path(src), Path(dest_dir)
    if not src.is_file():
        raise ManifestError(f"{src}: no file to copy")
    dest = dest_dir / src.name
    if dest.resolve() == src.resolve():
        raise ManifestError(f"{src}: a frozen copy cannot be the source itself")
    if dest.exists():
        raise ManifestError(f"{dest}: a frozen copy is already there")
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    mode = os.stat(dest).st_mode
    os.chmod(dest, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    return dest


# --- for scripts running inside Blender -----------------------------------------------------


def script_args() -> list[str]:
    """The arguments after `--` on Blender's command line."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def done() -> None:
    """Print the line that tells `run_blender` the script reached its end."""
    print(DONE_LINE, flush=True)
