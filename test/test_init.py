"""`model-kit init` installs the kit and points the project's agent at `model-kit guide`;
`model-kit guide` prints the agent steps."""

from pathlib import Path

import pytest

from model_kit import cli
from model_kit import init as init_module

SPEC = f"git+{init_module.REPO}@v{init_module.kit_version()}"


class FakeUv:
    """Records each uv command. `tools` is what `uv tool list` prints."""

    def __init__(self, tools: str = ""):
        self.calls: list[list[str]] = []
        self.tools = tools

    def __call__(self, cmd: list[str]) -> str:
        self.calls.append(cmd)
        return self.tools if cmd[:3] == ["uv", "tool", "list"] else ""


@pytest.fixture
def uv(monkeypatch):
    fake = FakeUv()
    monkeypatch.setattr(init_module, "_run", fake)
    monkeypatch.setattr(init_module.shutil, "which", lambda name: f"/bin/{name}")
    return fake


def test_init_writes_the_line_to_a_new_claude_md(tmp_path, uv, capsys):
    assert init_module.init(tmp_path) == 0
    assert (tmp_path / "CLAUDE.md").read_text() == f"{init_module.INIT_LINE}\n"
    assert not (tmp_path / "AGENTS.md").exists()


def test_init_appends_to_claude_md_and_an_existing_agents_md_once(tmp_path, uv):
    (tmp_path / "CLAUDE.md").write_text("# Project\n\nRules.\n")
    (tmp_path / "AGENTS.md").write_text("# Agents\n")
    assert init_module.init(tmp_path) == 0
    assert init_module.init(tmp_path) == 0  # a second run changes nothing
    assert (tmp_path / "CLAUDE.md").read_text() == f"# Project\n\nRules.\n\n{init_module.INIT_LINE}\n"
    assert (tmp_path / "AGENTS.md").read_text() == f"# Agents\n\n{init_module.INIT_LINE}\n"


def test_init_installs_a_tool_outside_a_python_project(tmp_path, uv):
    init_module.init(tmp_path)
    assert ["uv", "tool", "install", SPEC] in uv.calls


def test_init_adds_a_dependency_inside_a_python_project(tmp_path, uv):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "game"\ndependencies = []\n')
    init_module.init(tmp_path)
    assert ["uv", "add", f"model-kit @ {SPEC}"] in uv.calls
    assert not [c for c in uv.calls if c[:2] == ["uv", "tool"]]


def test_init_skips_an_install_already_done(tmp_path, uv, capsys):
    uv.tools = "model-kit v0.1.0\n- model-kit\n"
    init_module.init(tmp_path)
    assert uv.calls == [["uv", "tool", "list"]]
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "game"\ndependencies = ["model-kit @ git+https://x"]\n')
    uv.calls.clear()
    init_module.init(tmp_path)
    assert uv.calls == []
    assert "already" in capsys.readouterr().out


def test_init_says_when_blender_is_not_set(tmp_path, uv, monkeypatch, capsys):
    monkeypatch.delenv("MODEL_KIT_BLENDER", raising=False)
    init_module.init(tmp_path)
    assert "MODEL_KIT_BLENDER" in capsys.readouterr().out


def test_init_without_uv_cannot_start(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(init_module.shutil, "which", lambda name: None)
    assert cli.main(["init", "--dir", str(tmp_path)]) == 2
    assert "uv" in capsys.readouterr().err
    assert not (tmp_path / "CLAUDE.md").exists()


def test_guide_prints_the_agent_steps(capsys):
    assert cli.main(["guide"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# model-kit")
    assert "model-kit gate" in out


def test_guide_says_how_to_iterate_and_where_the_times_go(capsys):
    assert cli.main(["guide"]) == 0
    out = capsys.readouterr().out
    assert "Only a gate run on the whole scene is a pass." in out
    assert "--timing" in out


def test_guide_where_prints_the_bars_and_kinds_folders(capsys):
    assert cli.main(["guide", "--where"]) == 0
    lines = dict(line.split(": ", 1) for line in capsys.readouterr().out.splitlines())
    assert sorted(lines) == ["bars", "kinds"]
    assert (Path(lines["bars"]) / "INDEX.md").is_file()
    assert (Path(lines["kinds"]) / "assembly" / "rules.md").is_file()
