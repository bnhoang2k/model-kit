import subprocess
import sys

import pytest

from model_kit.core import gate
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.checks.reach import Reach
from model_kit.core.checks.trace import Trace
from model_kit.core.metagate import Plant, PlantUnit
from model_kit.core.scene import save_scene

from .scenes import make_ctx


def test_unknown_check_exits_2(tmp_path, capsys):
    save_scene(make_ctx(), tmp_path / "scene")
    assert gate.main([str(tmp_path / "scene"), "--checks", "reach,no-such-check"]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert "no-such-check" in err
    assert "must-touch, reach, trace" in err


def test_gate_runs_named_checks_on_saved_scene(tmp_path, capsys):
    near, far = tmp_path / "near", tmp_path / "far"
    save_scene(make_ctx(at=(0.5, 0.0, 0.0)), near)
    save_scene(make_ctx(at=(1.2, 0.0, 0.0)), far)

    assert gate.main([str(near), "--checks", "must-touch,reach"]) == 0
    out, _ = capsys.readouterr()
    assert out == "passed\n"

    assert gate.main([str(far), "--checks", "reach"]) == 1
    out, err = capsys.readouterr()
    assert out == "failed: 1\n"
    assert err.startswith("reach | reach targets[0]: arm.R -> kit:Plate.top_R | ")


def test_checks_is_optional_and_means_every_check(tmp_path, capsys):
    save_scene(make_ctx(at=(1.2, 0.0, 0.0)), tmp_path / "far")
    assert gate.main([str(tmp_path / "far")]) == 1
    default = capsys.readouterr()
    assert gate.main([str(tmp_path / "far"), "--checks", ",".join(gate.CHECKS)]) == 1
    assert capsys.readouterr() == default


def test_allowlist_lets_a_finding_through_with_its_reason(tmp_path, capsys):
    save_scene(make_ctx(at=(1.2, 0.0, 0.0)), tmp_path / "far")
    allow = tmp_path / "proof.allow"
    allow.write_text("reach | * -> kit:Plate.top_R | the plate is out of reach on purpose\n")
    assert gate.main([str(tmp_path / "far"), "--checks", "reach", "--allow", str(allow)]) == 0
    out, err = capsys.readouterr()
    assert out == "passed\n"
    assert "(reason: the plate is out of reach on purpose)" in err


def test_no_checks_is_partial(tmp_path, capsys):
    save_scene(make_ctx(), tmp_path / "scene")
    assert gate.main([str(tmp_path / "scene"), "--checks", ""]) == 1
    assert capsys.readouterr().out == "partial\n"


@pytest.mark.parametrize("args", [
    ["{missing}", "--checks", "reach"],
    ["{scene}", "--checks", "reach", "--allow", "{missing}"],
])
def test_a_missing_file_cannot_start(tmp_path, capsys, args):
    save_scene(make_ctx(), tmp_path / "scene")
    names = {"scene": str(tmp_path / "scene"), "missing": str(tmp_path / "nothing")}
    assert gate.main([a.format(**names) for a in args]) == 2
    assert capsys.readouterr().out == ""


def test_usage_error_exits_2(capsys):
    assert gate.main([]) == 2
    assert capsys.readouterr().out == ""


def test_module_entry_point(tmp_path):
    save_scene(make_ctx(), tmp_path / "scene")
    done = subprocess.run([sys.executable, "-m", "model_kit.core.gate", str(tmp_path / "scene"),
                           "--checks", "reach"], capture_output=True, text=True, check=False)
    assert (done.returncode, done.stdout) == (0, "passed\n")


def test_the_registry_names_each_check_by_its_own_name():
    assert gate.CHECKS
    assert all(name == check.name for name, check in gate.CHECKS.items())
    assert {MustTouch, Reach, Trace} <= set(gate.CHECKS.values())


def test_the_registry_holds_the_assembly_checks_and_ends_with_coverage():
    assert {"receiver", "keep-off", "connectors", "export-hygiene", "coverage"} <= set(gate.CHECKS)
    assert list(gate.CHECKS)[-1] == "coverage"


def test_coverage_in_a_run_sees_the_other_checks_of_that_run(tmp_path, capsys):
    save_scene(make_ctx(), tmp_path / "scene")
    scene = str(tmp_path / "scene")
    assert gate.main([scene, "--checks", "coverage"]) == 1  # nothing else ran: no part measured
    assert "coverage | kit:Plate top | " in capsys.readouterr().err
    assert gate.main([scene, "--checks", "coverage,floaters"]) == 1  # partial: no body shells
    assert "coverage | kit:Plate top | " not in capsys.readouterr().err
    assert gate.main([scene, "--checks", "floaters,coverage"]) == 1  # the order is the user's
    assert "coverage | kit:Plate top | " not in capsys.readouterr().err


def _cannot_start(capsys, kind):
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"cannot start: {kind}: ") and err.count("\n") == 1, err


def test_a_cut_short_samples_file_cannot_start(tmp_path, capsys):
    scene = tmp_path / "scene"
    save_scene(make_ctx(), scene)
    samples = scene / "samples" / "reach.npz"
    samples.write_bytes(samples.read_bytes()[: samples.stat().st_size // 2])
    assert gate.main([str(scene), "--checks", "reach"]) == 2
    _cannot_start(capsys, "BadZipFile")


def test_a_blind_check_cannot_start(tmp_path, capsys, monkeypatch):
    class Blind:
        name = "blind"

        def plants(self, ctx):
            return [PlantUnit("cup", (Plant("never seen", "cup", ctx),))]

        def run(self, ctx, only=None, samples=frozenset()):
            return []

        def coverage(self, ctx):
            return []

    monkeypatch.setitem(gate.CHECKS, "blind", Blind)
    save_scene(make_ctx(), tmp_path / "scene")
    assert gate.main([str(tmp_path / "scene"), "--checks", "blind"]) == 2
    _cannot_start(capsys, "BlindCheck")


def test_a_manifest_error_from_loading_the_scene_cannot_start(tmp_path, capsys):
    scene = tmp_path / "scene"
    save_scene(make_ctx(), scene)
    (scene / "samples" / "ghost.npz").write_bytes((scene / "samples" / "reach.npz").read_bytes())
    assert gate.main([str(scene), "--checks", "reach"]) == 2
    _cannot_start(capsys, "ManifestError")
