import json
import os
import runpy
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from model_kit._shipped import PRIVATE_BAR_FILE, TEXT_SUFFIXES, shipped_files


def test_core_never_imports_bpy():
    code = ("import sys, pkgutil, importlib, model_kit.core\n"
            "for m in pkgutil.walk_packages(model_kit.core.__path__, 'model_kit.core.'): importlib.import_module(m.name)\n"
            "print(sorted(k for k in sys.modules if k.split('.')[0] in {'bpy','mathutils','bmesh','scipy','trimesh'}))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": "src"}, check=True)
    assert out.stdout.strip() == "[]"


# The names this repository must never ship sit in a private list, held back from the public
# release; the scanner's own tests use made-up names.
PRIVATE_NAMES = Path(__file__).resolve().parent / "private" / "banned_names.py"
BANNED = runpy.run_path(str(PRIVATE_NAMES))["BANNED"] if PRIVATE_NAMES.exists() else []
MADE_UP = ["MadeRig", "Made_", "madeup", "made-tool", "/made/", "Fake_Guitar", "Fake_Drum"]


def _npz_text(path: Path) -> str:
    """Every name and string a .npz holds: its array names, and every string array (part names,
    rig key names, and a skeleton's JSON `info`). Number arrays hold no names and are skipped by
    their dtype, and an object array (which would need pickle) raises."""
    out = []
    with np.load(path, allow_pickle=False) as z:
        for key in z.files:
            out.append(key)
            array = z[key]
            if array.dtype.kind in "US":
                out += [str(v) for v in array.ravel().tolist()]
    return "\n".join(out)


def scan_names(root: Path, files: list[Path], banned: list[str] = BANNED) -> list[tuple[str, str]]:
    """Banned tokens in the given files, in the JSON chunk of every .glb under root, and in the
    strings of every .npz under root."""
    from model_kit.core.glb import read_glb_json

    hits = []
    for path in files:
        text = path.read_text()
        hits += [(str(path), b) for b in banned if b in text]
    for glb in sorted(root.rglob("*.glb")):
        text = json.dumps(read_glb_json(glb))
        hits += [(str(glb), b) for b in banned if b in text]
    for npz in sorted(root.rglob("*.npz")):
        text = _npz_text(npz)
        hits += [(str(npz), b) for b in banned if b in text]
    return hits


@pytest.mark.held("test/private/banned_names.py")
def test_no_private_names():
    bars = Path("bars")
    # Every text suffix the wheel ships (see model_kit._shipped), and the .glb JSON chunks.
    files = [*Path("src").rglob("*.py"), *Path("src").rglob("*.md"), *Path("test").rglob("*.py"),
             *Path("test").rglob("*.md"), *Path("tools").rglob("*.py"),
             *(p for p in bars.rglob("*") if p.is_file() and p.suffix in TEXT_SUFFIXES),
             Path("AGENTS.md"), Path("README.md")]
    files = [p for p in files if p.name != "test_hygiene.py"]
    assert scan_names(bars, files) == []
    # the strings (names and JSON `info`) of every .npz fixture under test/
    assert scan_names(Path("test"), []) == []


def test_no_private_bar_file():
    assert sorted(Path("bars").rglob(PRIVATE_BAR_FILE)) == []


def test_every_shipped_file_is_scanned():
    shipped = {p.suffix for p in shipped_files(Path("bars"))}
    assert shipped <= {*TEXT_SUFFIXES, ".glb", ".png"}


def test_glb_names_scanned(tmp_path):
    body = json.dumps({"asset": {"version": "2.0"}, "meshes": [{"name": MADE_UP[5] + "_01"}]}).encode()
    body += b" " * (-len(body) % 4)
    chunk = struct.pack("<I4s", len(body), b"JSON") + body
    (tmp_path / "a.glb").write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunk)) + chunk)
    assert scan_names(tmp_path, [], MADE_UP) == [(str(tmp_path / "a.glb"), MADE_UP[5])]
    (tmp_path / "a.glb").unlink()
    assert scan_names(tmp_path, [], MADE_UP) == []


def test_npz_strings_scanned(tmp_path):
    info = json.dumps({"piece": MADE_UP[5] + "_01", "seed": 1})
    np.savez(tmp_path / "a.skeleton.npz", limb_id=np.array(["trunk", "branch.1"]),
             pts=np.zeros((2, 3)), info=np.asarray(info))
    assert scan_names(tmp_path, [], MADE_UP) == [(str(tmp_path / "a.skeleton.npz"), MADE_UP[5])]
    np.savez(tmp_path / "a.skeleton.npz", limb_id=np.array(["trunk", MADE_UP[1] + "x"]))
    assert scan_names(tmp_path, [], MADE_UP) == [(str(tmp_path / "a.skeleton.npz"), MADE_UP[1])]
    np.savez(tmp_path / "a.skeleton.npz", **{MADE_UP[3]: np.zeros(1)})  # an array name counts
    assert scan_names(tmp_path, [], MADE_UP) == [(str(tmp_path / "a.skeleton.npz"), MADE_UP[3])]
    np.savez(tmp_path / "a.skeleton.npz", pts=np.zeros((2, 3)), info=np.asarray('{"seed": 1}'))
    assert scan_names(tmp_path, [], MADE_UP) == []


def test_test_fixtures_and_tools_are_scanned(tmp_path):
    """The scan reads tools/*.py and test/**/*.md, and the npz strings under test/."""
    (tmp_path / "t.md").write_text("see " + MADE_UP[3])
    (tmp_path / "m.skeleton.npz").write_bytes(b"")
    np.savez(tmp_path / "m.skeleton.npz", info=np.asarray(json.dumps({"source": MADE_UP[6] + "x"})))
    assert scan_names(tmp_path, [tmp_path / "t.md"], MADE_UP) == [
        (str(tmp_path / "t.md"), MADE_UP[3]), (str(tmp_path / "m.skeleton.npz"), MADE_UP[6])]
