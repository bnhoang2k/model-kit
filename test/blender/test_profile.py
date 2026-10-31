"""`model-kit profile` runs a build script under cProfile inside Blender."""

import pstats

import pytest

from model_kit import cli

SCRIPT = '''\
from model_kit.blender import runner


def slow_bit():
    return sum(i * i for i in range(200_000))


def main():
    slow_bit()
    print("ARGS", runner.script_args())
    runner.done()


main()
'''


@pytest.mark.blender
def test_profile_names_function(tmp_path, monkeypatch, capsys):
    build = tmp_path / "mybuild.py"
    build.write_text(SCRIPT)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert cli.main(["profile", str(build), "--top", "40", "--", "--alpha", "1"]) == 0
    out = capsys.readouterr().out
    assert "slow_bit" in out
    prof = work / "mybuild.prof"
    assert prof.is_file()
    assert "slow_bit" in {f[2] for f in pstats.Stats(str(prof)).stats}


@pytest.mark.blender
def test_profile_passes_the_script_its_args(tmp_path, monkeypatch, capsys):
    build = tmp_path / "mybuild.py"
    build.write_text(SCRIPT.replace('print("ARGS", runner.script_args())',
                                    'assert runner.script_args() == ["--alpha", "1"]'))
    monkeypatch.chdir(tmp_path)
    assert cli.main(["profile", str(build), "--", "--alpha", "1"]) == 0


@pytest.mark.blender
def test_profile_failing_script_exits_2(tmp_path, monkeypatch, capsys):
    build = tmp_path / "bad.py"
    build.write_text("raise RuntimeError('the build broke')\n")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["profile", str(build)]) == 2
    err = capsys.readouterr().err
    assert "the build broke" in err
