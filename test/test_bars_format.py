"""The bar.json format, the generated index and the wheel that ships the bars."""

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from model_kit import bars as bars_module
from model_kit.bars import INDEX_HEADER, bar_for, bars_root, load_bar, read_bars, render_index
from model_kit.core.errors import ManifestError

REPO = Path(__file__).resolve().parent.parent


def bar_json(slug="02-redbud", cls="branching", default=True, **over):
    data = {
        "slug": slug, "title": "Redbud", "class": cls, "default": default,
        "piece": "redbud:Redbud", "reads": "At the page camera: a small tree.",
        "fits": "Branching pieces.", "wrong_for": "Hard-surface props.",
        "references": [{"key": "W", "url": "https://example.com/w", "gives": "Height."}],
        "files": {"style": "style.json", "pack": "redbud.pack.json", "glb": "redbud.glb",
                  "scene": "scene", "sheet": "sheet.png"},
    }
    data.update(over)
    return data


def make_bar(root, slug="02-redbud", scene=True, **over):
    folder = root / slug
    folder.mkdir(parents=True)
    for name in ("style.json", "redbud.pack.json", "redbud.glb", "sheet.png"):
        (folder / name).write_bytes(b"x")
    if scene:
        (folder / "scene").mkdir()
    (folder / "bar.json").write_text(json.dumps(bar_json(slug=slug, **over)))
    return folder


def test_loads_fixture_bar(tmp_path):
    bar = load_bar(make_bar(tmp_path))
    assert (bar.slug, bar.cls, bar.default, bar.piece) == ("02-redbud", "branching", True, "redbud:Redbud")
    assert bar.superseded is None and bar.dir == tmp_path / "02-redbud"
    assert bar.references[0].key == "W" and bar.files.glb == "redbud.glb"


def test_scene_is_checked_only_when_the_folder_has_it(tmp_path):
    load_bar(make_bar(tmp_path, scene=False))


def test_slug_must_match_folder(tmp_path):
    folder = make_bar(tmp_path)
    (tmp_path / "02-redbud").rename(tmp_path / "03-other")
    with pytest.raises(ManifestError, match="slug"):
        load_bar(tmp_path / "03-other")
    assert folder.name == "02-redbud"


def test_missing_file_raises(tmp_path):
    folder = make_bar(tmp_path)
    (folder / "sheet.png").unlink()
    with pytest.raises(ManifestError, match="sheet.png"):
        load_bar(folder)


def test_unknown_and_missing_fields_raise(tmp_path):
    folder = make_bar(tmp_path, extra=1)
    with pytest.raises(ManifestError, match="unknown field 'extra'"):
        load_bar(folder)
    data = bar_json()
    del data["fits"]
    (folder / "bar.json").write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="missing field 'fits'"):
        load_bar(folder)


def test_repeated_key_and_bad_values_raise(tmp_path):
    folder = make_bar(tmp_path)
    (folder / "bar.json").write_text('{"slug": "a", "slug": "b"}')
    with pytest.raises(ManifestError, match="repeats"):
        load_bar(folder)
    for over, match in [({"class": "wall"}, "class"), ({"piece": "redbud"}, "piece"),
                        ({"default": "yes"}, "default")]:
        (folder / "bar.json").write_text(json.dumps(bar_json(**over)))
        with pytest.raises(ManifestError, match=match):
            load_bar(folder)


def test_no_bar_json_raises(tmp_path):
    (tmp_path / "02-redbud").mkdir()
    with pytest.raises(ManifestError, match="bar.json"):
        load_bar(tmp_path / "02-redbud")


def test_read_bars_in_folder_order_and_skips_folders_without_bar_json(tmp_path):
    make_bar(tmp_path, "05-b", cls="assembly")
    make_bar(tmp_path, "02-redbud")
    (tmp_path / "03-rules-only").mkdir()
    assert [b.slug for b in read_bars(tmp_path)] == ["02-redbud", "05-b"]


def test_one_default_per_class(tmp_path):
    make_bar(tmp_path, "02-a", default=False)
    with pytest.raises(ManifestError, match="branching.*default"):
        read_bars(tmp_path)
    make_bar(tmp_path, "03-b", default=True)
    assert len(read_bars(tmp_path)) == 2
    make_bar(tmp_path, "04-c", default=True)
    with pytest.raises(ManifestError, match="branching.*default"):
        read_bars(tmp_path)


def test_superseded_bar_left_out_of_index(tmp_path):
    make_bar(tmp_path, "02-old", default=False, superseded="03-new")
    make_bar(tmp_path, "03-new")
    bars = read_bars(tmp_path)
    assert bars[0].superseded == "03-new"
    text = render_index(bars)
    assert "03-new" in text and "02-old" not in text
    assert bar_for("branching", bars).slug == "03-new"


def test_bar_for_missing_class_names_it(tmp_path):
    make_bar(tmp_path)
    with pytest.raises(ManifestError, match="no bar for rigged pieces; pass --bar none"):
        bar_for("rigged", read_bars(tmp_path))


def test_bars_root_override_and_env(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("MODEL_KIT_BARS", str(tmp_path))
    assert bars_root(other) == other
    assert bars_root() == tmp_path
    monkeypatch.setenv("MODEL_KIT_BARS", str(tmp_path / "missing"))
    with pytest.raises(ManifestError, match="MODEL_KIT_BARS"):
        bars_root()
    monkeypatch.delenv("MODEL_KIT_BARS")
    assert bars_root() == REPO / "bars"  # no shipped copy in a source checkout: the tree wins
    shipped = tmp_path / "pkg"
    (shipped / "_bars").mkdir(parents=True)
    monkeypatch.setattr(bars_module.resources, "files", lambda name: shipped)
    assert bars_root() == shipped / "_bars"  # a shipped copy beats the tree
    monkeypatch.setenv("MODEL_KIT_BARS", str(tmp_path))
    assert bars_root() == tmp_path  # the env var beats the shipped copy


def test_index_table_row_and_header(tmp_path):
    make_bar(tmp_path)
    text = render_index(read_bars(tmp_path))
    assert text.startswith(INDEX_HEADER)
    assert "| [02-redbud](02-redbud/) | Redbud | yes | `redbud:Redbud` |" in text
    assert "branching" in text and "assembly" not in text


def test_index_is_current():
    root = REPO / "bars"
    assert render_index(read_bars(root)) == (root / "INDEX.md").read_text()


def test_private_bar_file_raises(tmp_path):
    folder = make_bar(tmp_path)
    (folder / "bar.private.json").write_text("{}")
    with pytest.raises(ManifestError, match="live outside the bar folder"):
        load_bar(folder)


def test_scene_folder_name_is_pinned(tmp_path):
    folder = make_bar(tmp_path)
    data = bar_json()
    data["files"]["scene"] = "blends"
    (folder / "bar.json").write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="'scene'"):
        load_bar(folder)


def test_ships_rule():
    from model_kit._shipped import ships

    for ok in ("INDEX.md", "02-a/bar.json", "02-a/a.glb", "02-a/sheet.png", "02-a/tool.py"):
        assert ships(Path(ok)), ok
    for no in ("02-a/scene/x.json", "02-a/a.blend", "02-a/a.blend1", "02-a/n.txt", "02-a/n.yaml",
               ".gitignore", "02-a/.DS_Store", "02-a/__pycache__/x.py", "02-a/bar.private.json"):
        assert not ships(Path(no)), no


def test_wheel_ships_bars(tmp_path):
    # Build from a copy of the repo that holds files which must stay out of the wheel.
    src = tmp_path / "repo"
    src.mkdir()
    for name in ("pyproject.toml", "hatch_build.py", "README.md", "AGENTS.md"):
        shutil.copy(REPO / name, src / name)
    ignore = shutil.ignore_patterns("__pycache__", "*.blend", "*.blend1", "scene")
    shutil.copytree(REPO / "src", src / "src", ignore=ignore)
    shutil.copytree(REPO / "bars", src / "bars", ignore=ignore)
    fixture = src / "bars" / "99-x"
    for rel in ("keep.json", "scene/s.json", "a.blend", "a.blend1", "notes.txt", "n.yaml",
                ".DS_Store", "__pycache__/c.pyc", "bar.private.json"):
        (fixture / rel).parent.mkdir(parents=True, exist_ok=True)
        (fixture / rel).write_text("{}")
    out = tmp_path / "out"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(out)], cwd=src, check=True,
                   capture_output=True, timeout=300)
    (wheel,) = out.glob("*.whl")
    bars = sorted(n for n in zipfile.ZipFile(wheel).namelist() if "/_bars/" in n)
    assert "model_kit/_bars/INDEX.md" in bars
    assert "model_kit/_bars/99-x/keep.json" in bars
    assert [n for n in bars if n.startswith("model_kit/_bars/99-x/")] == ["model_kit/_bars/99-x/keep.json"]
    assert not [n for n in bars if n.endswith((".gitignore", ".txt", ".yaml", ".blend", ".blend1"))]
    assert zipfile.ZipFile(wheel).read("model_kit/AGENTS.md") == (REPO / "AGENTS.md").read_bytes()
    assert "model_kit/kinds/assembly/rules.md" in zipfile.ZipFile(wheel).namelist()  # guide --where


def test_wheel_build_refuses_a_tree_without_bars(tmp_path):
    src = tmp_path / "repo"
    src.mkdir()
    for name in ("pyproject.toml", "hatch_build.py", "README.md"):
        shutil.copy(REPO / name, src / name)
    shutil.copytree(REPO / "src", src / "src", ignore=shutil.ignore_patterns("__pycache__"))
    done = subprocess.run(["uv", "build", "--wheel", "--out-dir", str(tmp_path / "out")], cwd=src,
                          check=False, capture_output=True, text=True, timeout=300)
    assert done.returncode != 0
    assert "refusing to build a wheel with no bars" in done.stdout + done.stderr
    assert not list((tmp_path / "out").glob("*.whl"))


def _with_refs(tmp_path, refs, files=()):
    folder = make_bar(tmp_path, references=refs)
    for name in files:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"x")
    return folder


def test_a_reference_may_be_a_file_the_bar_ships(tmp_path):
    folder = _with_refs(tmp_path, [{"key": "D", "file": "drawing.png", "gives": "The design."}],
                        files=["drawing.png"])
    ref = load_bar(folder).references[0]
    assert (ref.key, ref.file, ref.url) == ("D", "drawing.png", None)


@pytest.mark.parametrize(("ref", "files", "match"), [
    ({"key": "D", "gives": "x"}, [], "exactly one"),
    ({"key": "D", "url": "https://example.com/d", "file": "d.png", "gives": "x"}, ["d.png"],
     "exactly one"),
    ({"key": "D", "file": "missing.png", "gives": "x"}, [], "is not in"),
    ({"key": "D", "file": "../d.png", "gives": "x"}, [], "inside the bar's folder"),
    ({"key": "D", "file": "/tmp/d.png", "gives": "x"}, [], "inside the bar's folder"),
    ({"key": "D", "file": "scene/d.png", "gives": "x"}, ["scene/d.png"], "the bar ships"),
    ({"key": "D", "file": "d.jpg", "gives": "x"}, ["d.jpg"], "the bar ships"),
    ({"key": "D", "file": ".d.png", "gives": "x"}, [".d.png"], "the bar ships"),
    ({"key": "D", "url": "ftp://example.com/d", "gives": "x"}, [], "http"),
])
def test_a_bad_reference_raises(tmp_path, ref, files, match):
    folder = _with_refs(tmp_path, [ref], files=files)
    with pytest.raises(ManifestError, match=match):
        load_bar(folder)
