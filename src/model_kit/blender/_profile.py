"""Run a build script under cProfile inside Blender. Started by `model-kit profile`.

    blender ... --python _profile.py -- <build.py> <out.prof> [args for the build ...]

The build runs as `__main__` and sees only its own args after `--` in `script_args()`. The
build calls `runner.done()` itself, as it does under any other run. The profile is written
even when the build raises, so a slow failure can still be read.
"""

import cProfile
import runpy
import sys

from model_kit.blender import runner


def main() -> None:
    args = runner.script_args()
    if len(args) < 2:
        raise SystemExit("_profile.py needs <build.py> <out.prof>")
    build, out, rest = args[0], args[1], args[2:]
    sys.argv = [build, "--", *rest]
    profile = cProfile.Profile()
    try:
        profile.runcall(runpy.run_path, build, run_name="__main__")
    finally:
        profile.dump_stats(out)


main()
