"""The shipped bars, checked as a user would meet them: each bar folder that has a bar.json.

A bar folder with no bar.json yet is not one of these tests' cases (it is not a bar until it has
one), so it never shows up as a skip. Tests that need a scene run only for bars that ship one.
"""

import json
from pathlib import Path

import pytest

from model_kit.bars import Bar, load_bar
from model_kit.core import gate
from model_kit.core.glb import read_glb_json
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import load_skeleton, same_skeleton
from model_kit.core.scene import SKELETON_SUFFIX

BARS = Path(__file__).resolve().parents[2] / "bars"


def _folders(keep=lambda folder, bar: True) -> list[Path]:
    """Bar folders with a bar.json that pass `keep`. A bar that does not load is kept: its own
    test_bar_json_loads must fail loud."""
    out = []
    for folder in sorted(BARS.glob("*/bar.json")):
        try:
            bar = load_bar(folder.parent)
        except Exception:  # noqa: BLE001 - the loading test reports it
            out.append(folder.parent)
            continue
        if keep(folder.parent, bar):
            out.append(folder.parent)
    return out


def _params(folders: list[Path]):
    return pytest.mark.parametrize("folder", folders, ids=[f.name for f in folders])


def _has_scene(folder: Path, bar: Bar) -> bool:
    return (folder / bar.files.scene).is_dir()


def _gated(folder: Path, bar: Bar) -> bool:
    return (folder / bar.files.scene / "sources").is_dir()


def _branching(folder: Path, bar: Bar) -> bool:
    return bar.cls == "branching" and _has_scene(folder, bar)


def test_there_are_bars():
    assert _folders(), f"no bar folder with a bar.json under {BARS}"


@_params(_folders())
def test_bar_json_loads(folder):
    bar = load_bar(folder)
    assert bar.slug == folder.name
    assert (folder / "README.md").is_file()


# A bar whose gate does not pass yet keeps its exact output here, with why and an issue, until
# the bar is fixed. The pin is not a pass: the bar still reads failed or partial, and any change
# to its output fails the test, so a fix (or a new fault) cannot go unseen.
KNOWN = Path(__file__).resolve().parent / "known"


@_params(_folders(_gated))
def test_gate_verdict(folder, capsys):
    scene = folder / load_bar(folder).files.scene
    code = gate.main([str(scene)])
    out = capsys.readouterr()
    pinned = KNOWN / f"{folder.name}.txt"
    if pinned.exists():
        text = pinned.read_text().splitlines()
        notes = [line for line in text if line.startswith("# ")]
        assert any(line.startswith("# Why: ") for line in notes), pinned
        assert any(line.startswith("# Issue: https://") for line in notes), pinned
        expected = [line for line in text if not line.startswith("# ")]
        assert code == 1 and (out.err + out.out).splitlines() == expected, out.err + out.out
        return
    assert code == 0
    lines = out.out.strip().splitlines()
    assert lines and lines[-1] == "passed", out.err


@_params(_folders(_has_scene))
def test_scene_pack_is_the_bar_pack(folder):
    bar = load_bar(folder)
    pack_name = bar.piece.split(":", 1)[0]
    scene_pack = folder / bar.files.scene / "packs" / f"{pack_name}.pack.json"
    assert json.loads(scene_pack.read_text()) == json.loads((folder / bar.files.pack).read_text())
    assert scene_pack.read_bytes() == (folder / bar.files.pack).read_bytes()


@_params(_folders(_has_scene))
def test_scene_source_is_the_bar_build(folder):
    """A scene that keeps its build script keeps the bar's current one, so a lint of the
    scene's sources reads the script that made it."""
    source = folder / load_bar(folder).files.scene / "sources" / "build.py"
    if source.exists():
        assert source.read_bytes() == (folder / "build.py").read_bytes()


@_params(_folders(_branching))
def test_regrows_from_rules(folder):
    """Every skeleton in the scene is exactly what the bud grower grows again from its own info,
    with the bar's species file merged with the kit file."""
    bar = load_bar(folder)
    units = json.loads((folder / bar.files.style).read_text())["units_per_metre"]
    skeletons = sorted((folder / bar.files.scene / "pieces").glob(f"*/*{SKELETON_SUFFIX}"))
    assert skeletons, f"{folder.name}: the scene holds no skeletons"
    assert bar.piece.split(":", 1)[1] in {p.name[: -len(SKELETON_SUFFIX)] for p in skeletons}
    for path in skeletons:
        saved = load_skeleton(path)
        info = saved.info
        rules = load_rules(folder / info["rules"], kit=KIT_RULES)
        fresh = grow_buds(rules, piece=info["piece"], height_m=info["height_m"], seed=info["seed"],
                     lite=saved.lite, units_per_metre=units)
        assert same_skeleton(saved, fresh), f"{path.name}: the rules no longer grow this tree"


def _glb_tris_by_root(glb: Path) -> dict[str, int]:
    """Triangles per top-level node of a .glb, from the index accessors (Draco keeps each count
    in the JSON chunk): the sum over the node's whole subtree, so a piece's meshes count under
    the piece's root."""
    doc = read_glb_json(glb)
    nodes, accessors = doc["nodes"], doc["accessors"]

    def mesh_tris(index: int) -> int:
        total = 0
        for prim in doc["meshes"][index]["primitives"]:
            assert prim.get("mode", 4) == 4, f"{glb.name}: a primitive that is not triangles"
            counted = prim["indices"] if "indices" in prim else prim["attributes"]["POSITION"]
            total += accessors[counted]["count"] // 3
        return total

    def subtree(index: int) -> int:
        node = nodes[index]
        own = mesh_tris(node["mesh"]) if "mesh" in node else 0
        return own + sum(subtree(c) for c in node.get("children", []))

    children = {c for n in nodes for c in n.get("children", [])}
    return {nodes[i]["name"]: subtree(i) for i in range(len(nodes)) if i not in children}


@_params(_folders())
def test_shipped_glb_triangles_are_the_pack_triangles(folder):
    """The .glb a page loads and the sheet renders holds the triangles its pack measured, per
    piece. Both packs record `measured.tris`, so that is the field compared."""
    bar = load_bar(folder)
    pack = json.loads((folder / bar.files.pack).read_text())
    by_root = _glb_tris_by_root(folder / bar.files.glb)
    for name, piece in pack["pieces"].items():
        assert name in by_root, f"{bar.files.glb}: no top-level node {name!r}; has {sorted(by_root)}"
        assert by_root[name] == piece["measured"]["tris"], f"{name}: glb vs pack tris"
