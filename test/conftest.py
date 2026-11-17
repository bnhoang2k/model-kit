import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def pytest_collection_modifyitems(config, items):
    # Some inputs are not in every copy of this repository. A test marked `held(path)` skips
    # where one is missing, saying which input, and runs everywhere else.
    for item in items:
        for mark in item.iter_markers("held"):
            missing = [p for p in mark.args if not (REPO / p).exists()]
            if missing:
                item.add_marker(pytest.mark.skip(
                    reason=f"input not in this copy of the repository: {', '.join(missing)}"))
    # A test marked `browser` needs Playwright and a Chromium that starts here.
    browser = [item for item in items if item.get_closest_marker("browser") is not None]
    if browser:
        from model_kit.compare import chromium_available

        if not chromium_available():
            skip = pytest.mark.skip(reason="needs Playwright's Chromium: uv sync --extra browser, "
                                           "then playwright install chromium")
            for item in browser:
                item.add_marker(skip)
    if os.environ.get("MODEL_KIT_BLENDER"):
        return
    skip = pytest.mark.skip(reason="set MODEL_KIT_BLENDER to run Blender tests")
    for item in items:
        # The marker, not the keywords: a folder named `blender` is a keyword of every test
        # in it, and the tests there that need no Blender must still run.
        if item.get_closest_marker("blender") is not None:
            item.add_marker(skip)
