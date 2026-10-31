"""`model-kit init` and `model-kit guide`.

init installs this release of the kit for a project and adds one line to the project's CLAUDE.md
(made if missing) and to its AGENTS.md when it has one. The line sends a coding agent to
`model-kit guide`, which prints the kit's AGENTS.md. A step already done is skipped, so a second
run changes nothing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from importlib import metadata, resources
from pathlib import Path

from model_kit import bars as bars_module
from model_kit.core.errors import ManifestError

REPO = "https://github.com/bnhoang2k/model-kit"
INIT_LINE = ("For any 3D model you build or check with Blender, run `model-kit guide` and follow "
             "the steps it prints.")
MARK = "model-kit guide"
BLENDER_ENV = "MODEL_KIT_BLENDER"
# A dependency line in pyproject.toml that already names the kit.
DEPENDS = re.compile(r"""["']model-kit\b""")


def kit_version() -> str:
    return metadata.version("model-kit")


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def guide_text() -> str:
    """The kit's AGENTS.md: shipped in the wheel, or read from the source tree."""
    shipped = resources.files("model_kit") / "AGENTS.md"
    if shipped.is_file():
        return shipped.read_text()
    tree = Path(__file__).resolve().parents[2] / "AGENTS.md"
    if tree.is_file():
        return tree.read_text()
    raise ManifestError("AGENTS.md is missing from this install of model-kit")


def _install(root: Path) -> str:
    spec = f"git+{REPO}@v{kit_version()}"
    project = root / "pyproject.toml"
    if project.is_file():
        if DEPENDS.search(project.read_text()):
            return "model-kit is already a dependency of this project"
        _run(["uv", "add", f"model-kit @ {spec}"])
        return f"added model-kit {kit_version()} to pyproject.toml"
    if any(line.split()[:1] == ["model-kit"] for line in _run(["uv", "tool", "list"]).splitlines()):
        return "model-kit is already installed as a uv tool"
    _run(["uv", "tool", "install", spec])
    return f"installed model-kit {kit_version()} as a uv tool"


def _point(path: Path) -> str | None:
    now = path.read_text() if path.is_file() else ""
    if MARK in now:
        return None
    body = now.rstrip()
    path.write_text(f"{body}\n\n{INIT_LINE}\n" if body else f"{INIT_LINE}\n")
    return f"added the guide line to {path.name}"


def init(root: Path) -> int:
    if shutil.which("uv") is None:
        raise ManifestError("init needs uv: https://docs.astral.sh/uv/getting-started/installation/")
    print(_install(root))
    # Claude Code reads CLAUDE.md, so it always gets the line; other tools read AGENTS.md.
    names = ["CLAUDE.md", "AGENTS.md"] if (root / "AGENTS.md").is_file() else ["CLAUDE.md"]
    for name in names:
        print(_point(root / name) or f"{name} already has the guide line")
    if not os.environ.get(BLENDER_ENV):
        print(f"set {BLENDER_ENV} to your Blender executable to build, render and profile")
    return 0


def run_init(args) -> int:
    return init(args.dir)


def run_guide(args) -> int:
    if args.where:
        print(f"bars: {bars_module.bars_root()}")
        print(f"kinds: {Path(__file__).resolve().parent / 'kinds'}")
        return 0
    sys.stdout.write(guide_text())
    return 0
