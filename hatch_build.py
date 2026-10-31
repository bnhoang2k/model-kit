"""Build hook: ship bars/ inside the wheel as model_kit/_bars, by the one rule in _shipped.py, and
AGENTS.md as model_kit/AGENTS.md, which `model-kit guide` prints."""

import importlib.util
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _shipped():
    path = Path(__file__).resolve().parent / "src" / "model_kit" / "_shipped.py"
    spec = importlib.util.spec_from_file_location("_model_kit_shipped", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BarsBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # An editable install reads bars/ from the source tree; a copy would go stale.
        if self.target_name != "wheel" or version == "editable":
            return
        root = Path(self.root) / "bars"
        if not (root / "INDEX.md").is_file():
            # An sdist cut without bars/ would otherwise build a wheel with no bars, silently.
            raise FileNotFoundError(
                f"{root / 'INDEX.md'} is missing: refusing to build a wheel with no bars. "
                "Build from a tree that holds bars/, or run `model-kit bars` to write INDEX.md.")
        for rel in _shipped().shipped_files(root):
            build_data["force_include"][str(root / rel)] = f"model_kit/_bars/{rel.as_posix()}"
        guide = Path(self.root) / "AGENTS.md"
        if not guide.is_file():
            raise FileNotFoundError(f"{guide} is missing: `model-kit guide` would have nothing to print.")
        build_data["force_include"][str(guide)] = "model_kit/AGENTS.md"
