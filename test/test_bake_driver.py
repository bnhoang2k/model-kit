"""The bake driver's refusal and pass-through, with no Blender: the Blender run is replaced."""

import json
import shutil
from pathlib import Path

import pytest

from model_kit import bake
from model_kit.blender import runner
from model_kit.compare import Difference
from model_kit.core.errors import ManifestError
from model_kit.core.scene import load_scene

BAR05 = Path(__file__).resolve().parents[1] / "bars" / "05-lamp"


def plan_for(scene: Path) -> bake.BakePlan:
    return bake.BakePlan(
        scene=scene, layout="room", mode="full", placements=(), live=(), cameras=(), lights=(),
        layers=("lamp",), texels_per_m=16, samples=16, seed=1, out=scene / bake.BAKE_DIR)


@pytest.fixture
def fake_blender(monkeypatch):
    """Replace the Blender run. It records what the folder held when Blender would have
    started, and writes a bake.json and one picture line, as the real script does."""
    seen = {}

    def run_blender(script, args, *, timeout, blend=None):
        out = Path(json.loads(args[0])["out"])
        seen["folder_existed"] = out.exists()
        seen["old_kept_while_running"] = (out.parent / bake.BAKE_DIR).exists()
        out.mkdir(parents=True)
        (out / "bake.json").write_text(json.dumps({"size": 8}))
        return runner.RunResult(True, 0, "noise\nPICTURE lamp.exr 1234 bytes (8 px square)\n")

    monkeypatch.setattr(bake.runner, "run_blender", run_blender)
    return seen


def test_existing_bake_refused(tmp_path, fake_blender):
    """A scene with a `bake/` folder is refused, naming the folder, before Blender starts; the
    folder and what it holds are left alone. With `replace=True` Blender writes into an empty
    `bake.partial/`, and the old bake is still there while it runs; the new bake then takes its
    place."""
    plan = plan_for(tmp_path)
    (tmp_path / "bake").mkdir()
    (tmp_path / "bake" / "old.exr").write_bytes(b"old")
    with pytest.raises(ManifestError, match=str(tmp_path / "bake")):
        bake.run_bake(plan, replace=False, timeout=1)
    assert "folder_existed" not in fake_blender
    assert (tmp_path / "bake" / "old.exr").read_bytes() == b"old"
    bake.run_bake(plan, replace=True, timeout=1)
    assert fake_blender["folder_existed"] is False
    assert fake_blender["old_kept_while_running"] is True
    assert not (tmp_path / "bake" / "old.exr").exists()
    assert (tmp_path / "bake" / "bake.json").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bake"]  # no partial or old left


def stopped_blender(monkeypatch, *, write_manifest: bool = False):
    """A Blender run that stops part way: it writes one picture into its folder (and, with
    `write_manifest`, a bake.json) and is not ok, as a killed or crashed run is."""
    def run_blender(script, args, *, timeout, blend=None):
        out = Path(json.loads(args[0])["out"])
        out.mkdir(parents=True)
        (out / "window.exr").write_bytes(b"half")
        if write_manifest:
            (out / "bake.json").write_text("{}")
        return runner.RunResult(False, -1, "model_kit: Blender timed out after 1 s\n")
    monkeypatch.setattr(bake.runner, "run_blender", run_blender)


@pytest.mark.parametrize("write_manifest", [False, True])
def test_a_stopped_bake_leaves_the_old_bake(tmp_path, monkeypatch, write_manifest):
    """Blender stops after one picture: the old bake is left as it was, the half-written folder
    is cleared, and the error says the old bake was kept."""
    (tmp_path / "bake").mkdir()
    (tmp_path / "bake" / "bake.json").write_text('{"old": true}')
    stopped_blender(monkeypatch, write_manifest=write_manifest)
    with pytest.raises(ManifestError, match="left as it was"):
        bake.run_bake(plan_for(tmp_path), replace=True, timeout=1)
    assert (tmp_path / "bake" / "bake.json").read_text() == '{"old": true}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bake"]


def test_a_stale_partial_folder_is_cleared_before_blender(tmp_path, fake_blender):
    """A run stopped by Ctrl-C or a restart left `bake.partial/` with a picture and no
    bake.json: the next run clears it, and Blender starts on an empty folder."""
    (tmp_path / "bake.partial").mkdir()
    (tmp_path / "bake.partial" / "window.exr").write_bytes(b"half")
    bake.run_bake(plan_for(tmp_path), replace=False, timeout=1)
    assert fake_blender["folder_existed"] is False
    assert sorted(p.name for p in (tmp_path / "bake").iterdir()) == ["bake.json"]


def test_a_run_stopped_mid_swap_puts_the_old_bake_back(tmp_path, fake_blender):
    """Stopped after the old bake stepped aside and before the new one took its place: the old
    bake goes back first, so without --replace it is refused as any kept bake is."""
    (tmp_path / "bake.old").mkdir()
    (tmp_path / "bake.old" / "bake.json").write_text('{"old": true}')
    with pytest.raises(ManifestError, match="already exists"):
        bake.run_bake(plan_for(tmp_path), replace=False, timeout=1)
    assert (tmp_path / "bake" / "bake.json").read_text() == '{"old": true}'


def test_plan_ignores_a_half_written_bake(tmp_path):
    """A scene whose `bake/` holds one picture and no bake.json (a run stopped before this fix
    could leave one): the gate cannot read it, but `plan_bake` reads the scene without its bake,
    so `--replace` can bake over it."""
    bar = tmp_path / "05-lamp"
    shutil.copytree(BAR05, bar, ignore=shutil.ignore_patterns("bake"))
    (bar / "scene" / "bake").mkdir()
    (bar / "scene" / "bake" / "window.exr").write_bytes(b"half")
    with pytest.raises(ManifestError, match="no bake.json"):
        load_scene(bar / "scene")
    plan = bake.plan_bake(bar / "scene", bar / "stage.json")
    assert plan.layout == "desk" and plan.out == bar / "scene" / "bake"


def test_picture_bytes_pass_through(tmp_path, fake_blender, capsys):
    bake.run_bake(plan_for(tmp_path), replace=False, timeout=1)
    out = capsys.readouterr().out
    assert "PICTURE lamp.exr 1234 bytes (8 px square)" in out
    assert "noise" not in out


def test_compare_paths(tmp_path, monkeypatch, capsys):
    """`bake --compare`: no bake yet, bake then compare; a bake and no --replace, compare the
    kept bake without touching it; --replace, bake again then compare."""
    from model_kit import cli

    calls = []
    plan = plan_for(tmp_path)
    monkeypatch.setattr(bake, "plan_bake", lambda scene, stage: plan)
    monkeypatch.setattr(bake, "run_bake", lambda p, *, replace, timeout: (
        calls.append(("bake", replace)) or p.out))
    monkeypatch.setattr(cli.compare_module, "compare", lambda *a: (
        calls.append(("compare",)) or [("top", Difference(0.5, 0.9, 0.01)),
                                       ("side", Difference(None, 0.0, 0.2))]))
    argv = ["bake", str(tmp_path), "--stage", "s.json", "--compare"]
    assert cli.main(argv) == 0
    assert calls == [("bake", False), ("compare",)]
    calls.clear()
    (tmp_path / "bake").mkdir()
    assert cli.main(argv) == 0
    assert calls == [("compare",)]
    printed = capsys.readouterr().out
    assert "top 0.5000 (covered in both 90.0%, in only one 1.0%)" in printed
    assert "side no pixel covered in both (covered in both 0.0%, in only one 20.0%)" in printed
    calls.clear()
    assert cli.main([*argv, "--replace"]) == 0
    assert calls == [("bake", True), ("compare",)]
    calls.clear()
    assert cli.main(["bake", str(tmp_path), "--stage", "s.json"]) == 0  # no --compare: as before
    assert calls == [("bake", False)]
