"""The `model-kit` command: gate, sheet, profile and bars. Profile's Blender run is tested in
test/blender/test_profile.py."""

import builtins

from core.scenes import make_ctx
from test_bars_format import make_bar as make_bar_folder
from test_sheet import make_pack

from model_kit import cli
from model_kit.bars import INDEX_HEADER
from model_kit.core import gate
from model_kit.core.scene import save_scene


def test_gate_runs_all_checks_by_default(tmp_path, capsys):
    # make_ctx holds pieces with no geometry (a stick, a doll): the piece checks list them as
    # unmeasured, so the run is partial, not passed.
    save_scene(make_ctx(), tmp_path / "scene")
    assert cli.main(["gate", str(tmp_path / "scene")]) == 1
    default = capsys.readouterr()
    assert default.out == "partial\n"
    assert cli.main(["gate", str(tmp_path / "scene"), "--checks",
                     ",".join(gate.CHECKS)]) == 1
    assert capsys.readouterr() == default


def test_gate_unknown_check_exits_2(tmp_path, capsys):
    save_scene(make_ctx(), tmp_path / "scene")
    assert cli.main(["gate", str(tmp_path / "scene"), "--checks", "no-such-check"]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert "no-such-check" in err


def test_gate_keeps_its_exit_codes(tmp_path, capsys):
    save_scene(make_ctx(at=(1.2, 0.0, 0.0)), tmp_path / "far")
    assert cli.main(["gate", str(tmp_path / "far"), "--checks", "reach"]) == 1
    assert capsys.readouterr().out == "failed: 1\n"
    assert cli.main(["gate", str(tmp_path / "far"), "--checks", ""]) == 1
    assert capsys.readouterr().out == "partial\n"


def test_bars_writes_the_index(tmp_path, capsys):
    root = tmp_path / "bars"
    make_bar_folder(root)
    assert cli.main(["bars", "--root", str(root)]) == 0
    text = (root / "INDEX.md").read_text()
    assert text.startswith(INDEX_HEADER)
    assert "02-redbud" in text
    assert str(root / "INDEX.md") in capsys.readouterr().out


def test_bars_check_fresh_exits_0(tmp_path):
    root = tmp_path / "bars"
    make_bar_folder(root)
    cli.main(["bars", "--root", str(root)])
    assert cli.main(["bars", "--root", str(root), "--check"]) == 0


def test_bars_check_stale_exits_1(tmp_path, capsys):
    root = tmp_path / "bars"
    make_bar_folder(root)
    cli.main(["bars", "--root", str(root)])
    (root / "INDEX.md").write_text("old\n")
    assert cli.main(["bars", "--root", str(root), "--check"]) == 1
    assert (root / "INDEX.md").read_text() == "old\n"  # --check writes nothing
    assert "stale" in capsys.readouterr().err


def test_bars_check_missing_index_exits_1(tmp_path):
    root = tmp_path / "bars"
    make_bar_folder(root)
    assert cli.main(["bars", "--root", str(root), "--check"]) == 1
    assert not (root / "INDEX.md").exists()


def test_bars_bad_root_exits_2(tmp_path, capsys):
    assert cli.main(["bars", "--root", str(tmp_path / "nothing")]) == 2
    assert "cannot start" in capsys.readouterr().err


def test_the_bars_error_names_the_flag_the_user_passed(tmp_path, capsys):
    missing = tmp_path / "nothing"
    assert cli.main(["bars", "--root", str(missing)]) == 2
    err = capsys.readouterr().err
    assert f"--root names {missing}, which is not a folder" in err
    assert "--bars" not in err
    pack = make_pack(tmp_path / "new", "shop", "Bench", "assembly", 0.92, 0.5)
    assert cli.main(["sheet", str(pack), "--piece", "Bench", "--style",
                     str(pack.parent / "style.json"), "--bars", str(missing)]) == 2
    err = capsys.readouterr().err
    assert f"--bars names {missing}, which is not a folder" in err
    assert "--root" not in err


def test_sheet_bars_help_names_the_whole_chain(capsys):
    assert cli.main(["sheet", "--help"]) == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "MODEL_KIT_BARS" in text and "shipped with the package" in text
    assert "source tree" in text


def test_usage_error_exits_2(capsys):
    assert cli.main([]) == 2
    assert cli.main(["no-such-command"]) == 2
    assert cli.main(["gate"]) == 2
    assert cli.main(["sheet", "x.pack.json"]) == 2
    assert "usage" in capsys.readouterr().err


def test_help_exits_0_and_lists_the_four_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("gate", "sheet", "profile", "bars"):
        assert name in out


def test_sheet_missing_extra_exits_2(tmp_path, monkeypatch, capsys):
    pack = make_pack(tmp_path / "new", "shop", "Bench", "assembly", 0.92, 0.5)
    real_import = builtins.__import__

    def no_pillow(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no Pillow here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pillow)
    code = cli.main(["sheet", str(pack), "--piece", "Bench", "--style",
                     str(pack.parent / "style.json"), "--bar", "none",
                     "--out", str(tmp_path / "s.png")])
    assert code == 2
    assert "model-kit[sheet]" in capsys.readouterr().err


def test_sheet_unknown_view_exits_2(tmp_path, capsys):
    pack = make_pack(tmp_path / "new", "shop", "Bench", "assembly", 0.92, 0.5)
    code = cli.main(["sheet", str(pack), "--piece", "Bench", "--style",
                     str(pack.parent / "style.json"), "--views", "page,sideways"])
    assert code == 2
    assert "sideways" in capsys.readouterr().err


def test_sheet_prints_the_rows_and_views(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_make_sheet(pack, piece, style, *, bar, bars, out, views):
        calls.update(bar=bar, views=[v[0] for v in views], out=out)
        return out

    monkeypatch.setattr(cli.sheet, "make_sheet", fake_make_sheet)
    out = tmp_path / "s.png"
    assert cli.main(["sheet", "p.pack.json", "--piece", "Bench", "--style", "s.json",
                     "--bar", "none", "--views", "page,top", "--out", str(out)]) == 0
    assert capsys.readouterr().out == f"wrote {out}: 1 rows x 2 views\n"
    assert calls == {"bar": "none", "views": ["page", "top"], "out": out}
    assert cli.main(["sheet", "p.pack.json", "--piece", "Bench", "--style", "s.json",
                     "--out", str(out)]) == 0
    assert capsys.readouterr().out == f"wrote {out}: 2 rows x 5 views\n"


def test_profile_without_blender_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MODEL_KIT_BLENDER", raising=False)
    build = tmp_path / "build.py"
    build.write_text("pass\n")
    assert cli.main(["profile", str(build)]) == 2
    assert "MODEL_KIT_BLENDER" in capsys.readouterr().err


def test_profile_missing_script_exits_2(tmp_path, capsys):
    assert cli.main(["profile", str(tmp_path / "nothing.py")]) == 2
    assert "cannot start" in capsys.readouterr().err


def test_args_after_dashes_are_for_profile_only(tmp_path, capsys):
    save_scene(make_ctx(), tmp_path / "scene")
    assert cli.main(["gate", str(tmp_path / "scene"), "--", "extra"]) == 2
    assert "after `--`" in capsys.readouterr().err


def test_sheet_for_a_rigged_piece_with_no_rigged_bar_exits_2(tmp_path, capsys):
    pack = make_pack(tmp_path / "doll", "shop", "Doll", "assembly", 1.0, 0.5, rigged=True)
    bars = tmp_path / "bars"
    make_bar_folder(bars)  # a branching bar only
    code = cli.main(["sheet", str(pack), "--piece", "Doll", "--style",
                     str(pack.parent / "style.json"), "--bars", str(bars),
                     "--out", str(tmp_path / "s.png")])
    assert code == 2
    assert "no bar for rigged pieces; pass --bar none" in capsys.readouterr().err
