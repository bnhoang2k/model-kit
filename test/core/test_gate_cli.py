import json
import subprocess
import sys

import pytest

from model_kit.core import gate
from model_kit.core.checks.must_touch import MustTouch
from model_kit.core.checks.reach import Reach
from model_kit.core.checks.trace import Trace
from model_kit.core.metagate import Finding, Plant, PlantUnit
from model_kit.core.scene import save_scene

from .checks.piece_scenes import two_pieces
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
    assert err.splitlines()[-1].startswith("reach | reach targets[0]: arm.R -> kit:Plate.top_R | ")


def _without_times(err):
    return [line for line in err.splitlines() if not line.startswith("time: ")]  # seconds differ


def test_checks_is_optional_and_means_every_check(tmp_path, capsys):
    save_scene(make_ctx(at=(1.2, 0.0, 0.0)), tmp_path / "far")
    assert gate.main([str(tmp_path / "far")]) == 1
    default = capsys.readouterr()
    assert gate.main([str(tmp_path / "far"), "--checks", ",".join(gate.CHECKS)]) == 1
    again = capsys.readouterr()
    assert again.out == default.out
    assert _without_times(again.err) == _without_times(default.err)


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
    out = capsys.readouterr()
    assert out.out == ""
    lines = out.err.splitlines()
    assert lines[-1].startswith("cannot start: BlindCheck: ")
    assert all(line.startswith("time: blind | ") for line in lines[:-1])
    assert any(line.startswith("time: blind | ") for line in lines[:-1])


def test_a_blind_run_appends_its_timing_rows(tmp_path, capsys, monkeypatch):
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
    log = tmp_path / "times.jsonl"
    assert gate.main([str(tmp_path / "scene"), "--checks", "blind", "--timing", str(log)]) == 2
    assert capsys.readouterr().out == ""
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["check"], r["phase"]) for r in rows] == [
        ("blind", "real"), ("blind", "coverage"), ("blind", "plants")]


def test_a_blind_run_prints_the_findings_then_cannot_start(tmp_path, capsys, monkeypatch):
    class AlwaysFail:
        name = "always_fail"

        def plants(self, ctx):
            return [PlantUnit("cup", (Plant("seen", "cup", ctx),))]

        def run(self, ctx, only=None, samples=frozenset()):
            return [Finding("always_fail", "cup", "bad")]

        def coverage(self, ctx):
            return []

    class Blind:
        name = "blind"

        def plants(self, ctx):
            return [PlantUnit("cup", (Plant("never seen", "cup", ctx),))]

        def run(self, ctx, only=None, samples=frozenset()):
            return []

        def coverage(self, ctx):
            return []

    monkeypatch.setitem(gate.CHECKS, "always_fail", AlwaysFail)
    monkeypatch.setitem(gate.CHECKS, "blind", Blind)
    save_scene(make_ctx(), tmp_path / "scene")
    assert gate.main([str(tmp_path / "scene"), "--checks", "always_fail,blind"]) == 2
    out = capsys.readouterr()
    assert out.out == ""
    lines = out.err.splitlines()
    assert lines[-1].startswith("cannot start: BlindCheck: ")
    assert sum(line.startswith("cannot start") for line in lines) == 1
    assert any(line.startswith("always_fail | ") for line in lines[:-1])


def test_a_manifest_error_from_loading_the_scene_cannot_start(tmp_path, capsys):
    scene = tmp_path / "scene"
    save_scene(make_ctx(), scene)
    (scene / "samples" / "ghost.npz").write_bytes((scene / "samples" / "reach.npz").read_bytes())
    assert gate.main([str(scene), "--checks", "reach"]) == 2
    _cannot_start(capsys, "ManifestError")


def test_timing_appends_one_json_line_per_check_and_phase(tmp_path):
    save_scene(make_ctx(), tmp_path / "scene")
    log = tmp_path / "times.jsonl"
    assert gate.main([str(tmp_path / "scene"), "--checks", "must-touch,reach",
                      "--timing", str(log)]) == 0
    assert gate.main([str(tmp_path / "scene"), "--checks", "reach", "--timing", str(log)]) == 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 9  # (2 checks + 1 check) x 3 phases
    assert set(rows[0]) == {"scene", "kit", "check", "phase", "seconds", "loadavg_1m", "pieces"}


def test_timing_in_a_missing_folder_cannot_start(tmp_path, capsys):
    save_scene(make_ctx(), tmp_path / "scene")
    log = tmp_path / "none" / "t.jsonl"
    assert gate.main([str(tmp_path / "scene"), "--timing", str(log)]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("cannot start: ManifestError:")
    assert str(log) in err
    assert not log.exists()


def test_a_run_that_stops_keeps_the_rows_of_the_phases_that_ended(tmp_path, capsys, monkeypatch):
    class Boom:
        name = "boom"

        def plants(self, ctx):
            return []

        def run(self, ctx, only=None, samples=frozenset()):
            raise RuntimeError("stopped")

        def coverage(self, ctx):
            return []

    monkeypatch.setitem(gate.CHECKS, "boom", Boom)
    save_scene(make_ctx(), tmp_path / "scene")
    log = tmp_path / "times.jsonl"
    assert gate.main([str(tmp_path / "scene"), "--checks", "reach,boom",
                      "--timing", str(log)]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("cannot start: RuntimeError")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r["check"], r["phase"]) for r in rows] == [("reach", "real")]


def test_pieces_scopes_the_run_and_prints_partial(tmp_path, capsys):
    save_scene(two_pieces(), tmp_path / "scene")
    assert gate.main([str(tmp_path / "scene"), "--pieces", "kit:Thing"]) == 1
    out, err = capsys.readouterr()
    assert out == "partial\n"
    assert "gate: scoped run (pieces: kit:Thing)" in err


@pytest.mark.parametrize("pieces", ["", "kit:Thing,kit:Thing", "kit:Nope"])
def test_pieces_refuses_a_bad_list_before_any_check_runs(tmp_path, capsys, pieces):
    save_scene(two_pieces(), tmp_path / "scene")
    log = tmp_path / "times.jsonl"
    assert gate.main([str(tmp_path / "scene"), "--pieces", pieces, "--timing", str(log)]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("cannot start: ManifestError:")
    assert not log.exists() or log.read_text() == ""


def test_timing_rows_name_the_pieces_of_a_scoped_run(tmp_path):
    save_scene(two_pieces(), tmp_path / "scene")
    scoped, full = tmp_path / "scoped.jsonl", tmp_path / "full.jsonl"
    gate.main([str(tmp_path / "scene"), "--checks", "crossings", "--pieces", "kit:Thing",
               "--timing", str(scoped)])
    gate.main([str(tmp_path / "scene"), "--checks", "crossings", "--timing", str(full)])
    rows = [json.loads(line) for line in scoped.read_text().splitlines()]
    assert rows and all(r["pieces"] == ["kit:Thing"] for r in rows)
    rows = [json.loads(line) for line in full.read_text().splitlines()]
    assert rows and all(r["pieces"] is None for r in rows)


def test_an_allow_line_for_a_dropped_piece_is_not_stale_in_a_scoped_run(tmp_path, capsys):
    save_scene(two_pieces(), tmp_path / "scene")
    allow = tmp_path / "proof.allow"
    allow.write_text("crossings | kit:Other nope x nope | a reason\n")
    scene = str(tmp_path / "scene")
    assert gate.main([scene, "--pieces", "kit:Thing", "--allow", str(allow)]) == 1
    assert "stale:" not in capsys.readouterr().err
    assert gate.main([scene, "--allow", str(allow)]) == 1
    assert "stale:" in capsys.readouterr().err
