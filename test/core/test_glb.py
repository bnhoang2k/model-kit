"""The .glb node table, read from a tiny .glb that the test writes with `struct`."""

import json
import struct
from pathlib import Path

import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.glb import (
    GlbNode,
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
