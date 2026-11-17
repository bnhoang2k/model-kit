"""The .glb node table, read from a tiny .glb that the test writes with `struct`."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.glb import (
    GlbNode,
    glb_mesh_attributes,
    glb_nodes,
    node_name,
    nodes_from_json,
    nodes_to_json,
    read_glb_json,
)


def write_glb(path: Path, js: dict, *, binary: bytes = b"") -> Path:
    """A .glb: the 12-byte header, a JSON chunk padded with spaces, and an optional BIN chunk."""
    body = json.dumps(js).encode()
    body += b" " * (-len(body) % 4)
    chunks = struct.pack("<I4s", len(body), b"JSON") + body
    if binary:
        binary += b"\0" * (-len(binary) % 4)
        chunks += struct.pack("<I4s", len(binary), b"BIN\0") + binary
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(chunks)) + chunks)
    return path


TREE = {
    "asset": {"version": "2.0"},
    "meshes": [{"name": "Lamp.base"}, {"name": "Lamp.stem"}],
    "nodes": [
        {"name": "Lamp", "children": [1, 2], "translation": [1.0, 0.0, 2.0]},
        {"name": "Lamp.base", "mesh": 0, "rotation": [0.0, 0.7071, 0.0, 0.7071]},
        {"name": "Lamp.stem", "mesh": 1, "scale": [1.0, 2.0, 1.0], "children": [3]},
        {"name": "Lamp.plug"},
    ],
}


def test_glb_nodes_reads_names_and_parents(tmp_path):
    nodes = glb_nodes(write_glb(tmp_path / "t.glb", TREE, binary=b"\1\2\3"))
    assert [n.name for n in nodes] == ["Lamp", "Lamp.base", "Lamp.stem", "Lamp.plug"]
    assert [n.parent for n in nodes] == [None, "Lamp", "Lamp", "Lamp.stem"]
    assert [n.mesh for n in nodes] == [None, "Lamp.base", "Lamp.stem", None]
    assert nodes[0].translation == (1.0, 0.0, 2.0)
    assert nodes[1].rotation == (0.0, 0.7071, 0.0, 0.7071)
    assert nodes[2].scale == (1.0, 2.0, 1.0)
    # glTF's own defaults for what the file leaves out.
    assert nodes[3] == GlbNode("Lamp.plug", "Lamp.stem", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
                               (1.0, 1.0, 1.0), None)


def test_read_glb_json_returns_the_json_chunk(tmp_path):
    assert read_glb_json(write_glb(tmp_path / "t.glb", TREE)) == TREE


def test_a_glb_with_no_nodes_gives_an_empty_table(tmp_path):
    assert glb_nodes(write_glb(tmp_path / "t.glb", {"asset": {"version": "2.0"}})) == []


def test_not_a_glb_raises_and_names_the_file(tmp_path):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"not a glb at all, just some text")
    with pytest.raises(ManifestError, match="bad.glb"):
        read_glb_json(bad)
    with pytest.raises(ManifestError, match="missing.glb"):
        glb_nodes(tmp_path / "missing.glb")
    cut = tmp_path / "cut.glb"
    cut.write_bytes(write_glb(tmp_path / "ok.glb", TREE).read_bytes()[:30])
    with pytest.raises(ManifestError, match="cut.glb"):
        read_glb_json(cut)


@pytest.mark.parametrize("edit, match", [
    (lambda js: js["nodes"][1].pop("name"), r"nodes\[1\] has no name"),
    (lambda js: js["nodes"][3].update(children=[2]), "two parents"),
    (lambda js: js["nodes"][3].update(children=[9]), "no node"),
    (lambda js: js["meshes"][0].pop("name"), "no name"),
    (lambda js: js["nodes"][1].update(mesh=7), "not in the file"),
    (lambda js: js["nodes"][1].update(matrix=[1] * 16), "matrix"),
    (lambda js: js["nodes"][0].update(translation=[1, 2]), "translation must be 3 numbers"),
])
def test_a_table_the_checks_could_not_trust_raises(tmp_path, edit, match):
    js = json.loads(json.dumps(TREE))
    edit(js)
    with pytest.raises(ManifestError, match=match) as err:
        glb_nodes(write_glb(tmp_path / "t.glb", js))
    assert "t.glb" in str(err.value)


def test_two_nodes_may_share_a_name(tmp_path):
    # A real glTF can give a bone and an object one name; export hygiene reports it.
    js = json.loads(json.dumps(TREE))
    js["nodes"][3]["name"] = "Lamp"  # the plug now shares the root's name
    nodes = glb_nodes(write_glb(tmp_path / "t.glb", js))
    assert [n.name for n in nodes] == ["Lamp", "Lamp.base", "Lamp.stem", "Lamp"]
    assert [n.parent for n in nodes] == [None, "Lamp", "Lamp", "Lamp.stem"]  # by index


def test_a_node_table_json_may_hold_a_duplicate_name(tmp_path):
    table = [GlbNode("Lamp"), GlbNode("Lamp.stem", "Lamp"), GlbNode("Lamp.stem", "Lamp.stem")]
    back = nodes_from_json(tmp_path / "nodes.json", nodes_to_json(table))
    assert back == table and [n.name for n in back].count("Lamp.stem") == 2
    # A parent name must still match a node.
    with pytest.raises(ManifestError, match="Ghost"):
        nodes_from_json(tmp_path / "nodes.json", nodes_to_json([GlbNode("Lamp", "Ghost")]))


def test_a_parts_node_is_named_piece_slash_part():
    assert node_name("Lamp", "shade") == "Lamp/shade"


def test_glb_mesh_attributes_reads_strided_attributes_and_joins_primitives(tmp_path):
    # Two primitives sharing one index accessor; each has interleaved POSITION and TEXCOORD_0
    # (stride 20), and the second's indices must shift past the first's three vertices.
    verts = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], dtype="<f4")
    second = verts + np.array([5, 0, 0, 0, 0], dtype="<f4")
    blob = verts.tobytes() + second.tobytes() + np.array([0, 1, 2], dtype="<u2").tobytes()
    js = {
        "asset": {"version": "2.0"},
        "meshes": [{"name": "m", "primitives": [
            {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 4},
            {"attributes": {"POSITION": 2, "TEXCOORD_0": 3}, "indices": 4}]}],
        "nodes": [{"name": "a/b", "mesh": 0}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 60, "byteStride": 20},
                        {"buffer": 0, "byteOffset": 60, "byteLength": 60, "byteStride": 20},
                        {"buffer": 0, "byteOffset": 120, "byteLength": 6}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 0, "byteOffset": 12, "componentType": 5126, "count": 3,
             "type": "VEC2"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "byteOffset": 12, "componentType": 5126, "count": 3,
             "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"}],
    }
    path = write_glb(tmp_path / "m.glb", js, binary=blob)
    got = glb_mesh_attributes(path, "a/b")
    assert got["POSITION"].tolist() == verts[:, :3].tolist() + second[:, :3].tolist()
    assert got["TEXCOORD_0"].tolist() == verts[:, 3:].tolist() * 2
    assert got["indices"].tolist() == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ManifestError, match="nodes named 'nope'"):
        glb_mesh_attributes(path, "nope")


def _one_triangle(view_length: int, count: int, blob_cut: int = 0) -> tuple[dict, bytes]:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
    blob = verts.tobytes()[:len(verts.tobytes()) - blob_cut]
    js = {"asset": {"version": "2.0"},
          "meshes": [{"name": "m", "primitives": [{"attributes": {"POSITION": 0}}]}],
          "nodes": [{"name": "a/b", "mesh": 0}],
          "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": view_length}],
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": count,
                         "type": "VEC3"}]}
    return js, blob


@pytest.mark.parametrize("view_length, count, blob_cut", [
    (36, 4, 0),  # one vertex more than the view holds
    (24, 3, 0),  # the view is shorter than the accessor
    (36, 3, 12),  # the view runs past the BIN chunk
])
def test_an_accessor_past_its_buffer_raises(tmp_path, view_length, count, blob_cut):
    """A short or malformed .glb raises, naming the accessor; it never reads memory past the
    buffer (numpy's `as_strided` would)."""
    js, blob = _one_triangle(view_length, count, blob_cut)
    path = write_glb(tmp_path / "short.glb", js, binary=blob)
    with pytest.raises(ManifestError, match="accessor 0 .* runs past"):
        glb_mesh_attributes(path, "a/b")


def test_an_accessor_that_ends_on_its_buffer_reads(tmp_path):
    js, blob = _one_triangle(36, 3)
    got = glb_mesh_attributes(write_glb(tmp_path / "t.glb", js, binary=blob), "a/b")
    assert got["POSITION"].tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
