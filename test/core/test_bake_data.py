"""`load_bake` and `load_scene`'s read of `<scene>/bake/`, on a tiny bake written here.

The .glb is one triangle on node `p/body` under a parent `p` that moves and turns it, with a
lightmap UV on TEXCOORD_1 (and a UV set 0 the bake leaves alone), and a one-row EXR beside it. The
real files `model-kit bake` writes are read in `test/blender/test_bake_exr.py`.
"""

import json
import struct

import numpy as np
import pytest

from model_kit.core.bake_data import load_bake
from model_kit.core.errors import ManifestError
from model_kit.core.scene import load_scene, save_scene

from .scenes import make_ctx
from .test_exr import encode

CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
UV0 = np.array([[0, 0], [1, 0], [0, 1]], dtype="<f4")
UV1 = np.array([[0.25, 0.5], [0.5, 0.5], [0.25, 0.75]], dtype="<f4")
S = np.sqrt(0.5)


def glb_bytes() -> bytes:
    indices = np.array([0, 1, 2, 0], dtype="<u2")  # the last is padding
    blob = CORNERS.tobytes() + UV0.tobytes() + UV1.tobytes() + indices.tobytes()
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": 36},
             {"buffer": 0, "byteOffset": 36, "byteLength": 24},
             {"buffer": 0, "byteOffset": 60, "byteLength": 24},
             {"buffer": 0, "byteOffset": 84, "byteLength": 6}]
    js = {
        "asset": {"version": "2.0"},
        "nodes": [{"name": "p", "translation": [10, 0, 0], "rotation": [0, 0, S, S],
                   "children": [1]},
                  {"name": "p/body", "scale": [2, 2, 2], "mesh": 0}],
        "meshes": [{"name": "body", "primitives": [{
            "attributes": {"POSITION": 0, "TEXCOORD_0": 1, "TEXCOORD_1": 2}, "indices": 3}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR"}],
        "bufferViews": views, "buffers": [{"byteLength": len(blob)}]}
    text = json.dumps(js).encode()
    text += b" " * (-len(text) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = (struct.pack("<I4s", len(text), b"JSON") + text
            + struct.pack("<I4s", len(blob), b"BIN\0") + blob)
    return b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body


PICTURE = (np.arange(2 * 4 * 3).reshape(2, 4, 3) / 8).astype(np.float16)


def write_bake(scene, *, with_glb=True, with_picture=True, extra=()):
    folder = scene / "bake"
    folder.mkdir(parents=True)
    (folder / "bake.json").write_text(json.dumps({
        "bake": 1, "layout": "yard", "mode": "full", "size": 4, "uv_channel": {"p/body": 1},
        "unseen": ["q/body"], "format": "exr-half-zip",
        "layers": {"lamp": {"lights": ["bulb"], "picture": "lamp.exr", "seed": 1,
                            "noise": {"p": 0.5}, "error": {"p": 0.25}, "baked_lights": {}}}}))
    if with_glb:
        (folder / "yard.glb").write_bytes(glb_bytes())
    if with_picture:
        (folder / "lamp.exr").write_bytes(encode(PICTURE))
    for name in extra:
        (folder / name).write_bytes(encode(PICTURE))
    return folder


def test_no_bake_folder_is_none(tmp_path):
    assert load_bake(tmp_path) is None
    save_scene(make_ctx(), tmp_path / "scene")
    assert load_scene(tmp_path / "scene").bake is None


def test_reads_the_record_the_glb_and_the_pictures(tmp_path):
    write_bake(tmp_path, extra=["old.exr"])
    bake = load_bake(tmp_path)
    assert (bake.mode, bake.size, bake.unseen) == ("full", 4, frozenset({"q/body"}))
    assert bake.files == {"lamp.exr", "old.exr"}
    layer = bake.layers["lamp"]
    assert (layer.picture, layer.noise, layer.error) == ("lamp.exr", {"p": 0.5}, {"p": 0.25})
    assert np.array_equal(layer.pixels, PICTURE.astype(np.float32))
    assert np.allclose(bake.uv["p/body"], UV1.reshape(1, 3, 2))  # TEXCOORD_1, as recorded


def test_positions_are_in_the_world(tmp_path):
    """Scaled by the node (2), turned a quarter about z by the parent, then moved 10 along x."""
    write_bake(tmp_path)
    got = load_bake(tmp_path).positions["p/body"]
    assert got.shape == (1, 3, 3)
    assert np.allclose(got[0], [[10, 0, 0], [10, 2, 0], [8, 0, 0]], atol=1e-6)


def test_a_listed_picture_that_is_not_there_has_no_pixels(tmp_path):
    write_bake(tmp_path, with_picture=False)
    bake = load_bake(tmp_path)
    assert bake.layers["lamp"].pixels is None and bake.files == frozenset()


def test_load_scene_reads_the_bake(tmp_path):
    save_scene(make_ctx(), tmp_path / "scene")
    write_bake(tmp_path / "scene")
    bake = load_scene(tmp_path / "scene").bake
    assert bake is not None and set(bake.uv) == {"p/body"}


def test_a_bake_folder_with_no_record_or_no_glb_raises(tmp_path):
    (tmp_path / "bake").mkdir()
    with pytest.raises(ManifestError, match="bake.json"):
        load_bake(tmp_path)
    other = tmp_path / "other"
    write_bake(other, with_glb=False)
    with pytest.raises(ManifestError, match="yard.glb"):
        load_bake(other)


def test_a_record_naming_a_part_the_glb_lacks_raises(tmp_path):
    folder = write_bake(tmp_path)
    record = json.loads((folder / "bake.json").read_text())
    record["uv_channel"] = {"nope/body": 1}
    (folder / "bake.json").write_text(json.dumps(record))
    with pytest.raises(ManifestError, match="nope/body"):
        load_bake(tmp_path)
