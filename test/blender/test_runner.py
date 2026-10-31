import os
import stat
from pathlib import Path

import pytest

from model_kit.blender.runner import DONE_LINE, _ok, blender_path, frozen_copy, run_blender
from model_kit.core.errors import ManifestError


def _script(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "script.py"
    path.write_text(body)
    return path


@pytest.mark.blender
def test_traceback_fails_even_on_exit_0(tmp_path):
    script = _script(tmp_path, (
        "import traceback\n"
        "try:\n"
        "    1 / 0\n"
        "except ZeroDivisionError:\n"
        "    traceback.print_exc()\n"
        f"print({DONE_LINE!r})\n"
    ))
    result = run_blender(script, [], timeout=120)
    assert result.code == 0
    assert "Traceback" in result.log
    assert DONE_LINE in result.log
    assert result.ok is False


@pytest.mark.blender
def test_missing_done_line_fails(tmp_path):
    script = _script(tmp_path, "print('the script ran but never said it finished')\n")
    result = run_blender(script, [], timeout=120)
    assert result.code == 0
    assert "Traceback" not in result.log
    assert result.ok is False


@pytest.mark.blender
def test_kit_importable_inside_blender(tmp_path):
    script = _script(tmp_path, (
        "import sys\n"
        "import model_kit.core.rig\n"
        "from model_kit.blender import export, runner\n"
        "print('ARGS', runner.script_args())\n"
        "runner.done()\n"
    ))
    result = run_blender(script, ["a b", "c"], timeout=120)
    assert result.ok, result.log
    assert "ARGS ['a b', 'c']" in result.log


@pytest.mark.blender
def test_python_error_fails(tmp_path):
    script = _script(tmp_path, "raise RuntimeError('boom')\n")
    result = run_blender(script, [], timeout=120)
    assert result.ok is False
    assert "boom" in result.log


@pytest.mark.blender
def test_nonzero_exit_fails_even_with_the_done_line(tmp_path):
    script = _script(tmp_path, f"import sys\nprint({DONE_LINE!r}, flush=True)\nsys.exit(3)\n")
    result = run_blender(script, [], timeout=120)
    assert result.code != 0
    assert "Traceback" not in result.log
    assert DONE_LINE in result.log.splitlines()
    assert result.ok is False


@pytest.mark.blender
def test_timeout_fails(tmp_path):
    script = _script(tmp_path, f"import time\ntime.sleep(60)\nprint({DONE_LINE!r})\n")
    result = run_blender(script, [], timeout=5)
    assert result.ok is False
    assert result.code == -1
    assert "timed out after 5 s" in result.log


@pytest.mark.parametrize("code, log, ok", [
    (0, f"start\n{DONE_LINE}\n", True),
    (0, f"start\n  {DONE_LINE}  \n", True),
    (3, f"start\n{DONE_LINE}\n", False),
    (0, f"Traceback (most recent call last):\n{DONE_LINE}\n", False),
    (0, "start\n", False),
    (0, f"not {DONE_LINE} alone\n", False),
])
def test_ok_needs_code_0_no_traceback_and_the_done_line(code, log, ok):
    assert _ok(code, log) is ok


def test_blender_path_unset_raises(monkeypatch):
    monkeypatch.delenv("MODEL_KIT_BLENDER", raising=False)
    with pytest.raises(ManifestError, match="MODEL_KIT_BLENDER"):
        blender_path()


def test_blender_path_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_KIT_BLENDER", str(tmp_path / "no-blender"))
    with pytest.raises(ManifestError, match="no-blender"):
        blender_path()


def test_frozen_copy_is_a_read_only_copy(tmp_path):
    src = tmp_path / "src" / "scene.blend"
    src.parent.mkdir()
    src.write_bytes(b"BLENDER-fake")
    copy = frozen_copy(src, tmp_path / "frozen")
    assert copy == tmp_path / "frozen" / "scene.blend"
    assert copy.read_bytes() == b"BLENDER-fake"
    assert not os.stat(copy).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    assert os.stat(src).st_mode & stat.S_IWUSR  # the source is untouched


def test_frozen_copy_refuses_a_missing_source_or_an_existing_copy(tmp_path):
    with pytest.raises(ManifestError, match="no file"):
        frozen_copy(tmp_path / "missing.blend", tmp_path / "frozen")
    src = tmp_path / "scene.blend"
    src.write_bytes(b"x")
    frozen_copy(src, tmp_path / "frozen")
    with pytest.raises(ManifestError, match="already"):
        frozen_copy(src, tmp_path / "frozen")
    with pytest.raises(ManifestError, match="itself"):
        frozen_copy(src, tmp_path)
