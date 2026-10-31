"""Soft fields in pack.json: a part's form, a bone's rest pose and a chain bind's falloff."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.forms import chain_weights
from model_kit.core.manifest import load_pack
from model_kit.core.manifest.forms import (
    BlobForm,
    DecalForm,
    Falloff,
    Flatten,
    MetaForm,
    Push,
    RingForm,
    SoftForm,
)
from model_kit.core.units import load_style
from model_kit.core.writer import pack_json, pack_text

FIX = Path(__file__).parent / "fixtures"
STYLE = load_style(FIX / "shop.style.json")
TOY = FIX / "toy.pack.json"


def _raw() -> dict:
    return json.loads(TOY.read_text())


def _piece(raw: dict) -> dict:
    return raw["pieces"]["toy"]


def _load(raw: dict, tmp_path: Path):
    path = tmp_path / "toy.pack.json"
    path.write_text(json.dumps(raw))
    return load_pack(path, STYLE)


def test_toy_pack_loads():
    pack = load_pack(TOY, STYLE)
    toy = pack.pieces["toy"]
    assert toy.measured == {}
    assert toy.rig is not None
    for bone in toy.rig.bones:
        assert bone.rest is not None
        gap = np.linalg.norm(np.subtract(bone.rest.tail, bone.rest.head))
        assert abs(gap - bone.length) < 1e-9
    for part in ("arm_L", "arm_R"):
        bind = toy.rig.bind[part]
        assert bind.mode == "chain"
        assert isinstance(bind.falloff, Falloff)
    assert toy.rig.bind["body"].falloff is None
    forms = {p.name: p.form for p in toy.parts}
    assert all(isinstance(f, SoftForm) for f in forms.values())
    assert isinstance(forms["body"].body, BlobForm)
    assert isinstance(forms["arm_L"].body, RingForm)
    assert forms["arm_L"].seat is None  # seats are for lens-like parts, not an arm
    assert forms["arm_R"].mirror_of == "arm_L" and forms["arm_R"].body is None
    assert toy.rig.bones[3].rest.why == "the hand bone ends at the fingertips"
    assert toy.rig.bones[0].rest.deform is False


def test_authored_pack_loads_with_plain_load_pack():
    """An authored pack carries `measured: {}` and reads before its first build."""
    assert load_pack(TOY, STYLE).pieces["toy"].measured == {}


def test_soft_fields_round_trip(tmp_path):
    pack = load_pack(TOY, STYLE)
    text = pack_text(pack, tmp_path / "out.pack.json")
    out = tmp_path / "out.pack.json"
    out.write_text(text)
    assert load_pack(out, STYLE) == pack
    wrote = json.loads(text)["pieces"]["toy"]
    assert wrote["parts"][2]["form"] == {"mirror_of": "arm_L"}
    assert wrote["rig"]["bones"][2]["rest"]["head"] == [0.2, 0.6, 0.0]
    assert wrote["rig"]["bind"]["arm_L"]["falloff"]["bands"] == [[0.4, 0.6]]
    assert "falloff" not in wrote["rig"]["bind"]["body"]


def test_every_form_kind_round_trips(tmp_path):
    raw = _raw()
    piece = _piece(raw)
    extra = {
        "cap": ("ring", {
            "top": [0.0, 0.9, 0.0], "bottom": [0.0, 0.7, 0.1], "rings": 6, "segments": 10,
            "stations": [{"u": 0.0, "half_side": 0.0, "half_depth": 0.0},
                         {"u": 0.5, "half_side": 0.1, "half_depth": 0.08, "off_side": 0.01,
                          "off_depth": -0.01, "inner_power": 2.0},
                         {"u": 1.0, "half_side": 0.08, "half_depth": 0.07}],
            "blend": "smooth",
            "caps": {"top": {"kind": "round", "length": 0.05},
                     "bottom": {"kind": "super", "length": 0.04, "power": 3.0}},
            "poles": {"top": [0.0, 0.01]},
            "bumps": [{"u": 0.5, "u_width": 0.2, "angle": 1.0, "angle_width": 0.5,
                       "height": 0.01, "lift": 0.0}],
            "smooth": {"passes": 2, "factor": 0.5, "below": 0.85},
            "seat": {"on": "body", "at": [0.0, 0.1], "sink": 0.0},
            "deforms": [
                {"kind": "push", "axis": "-z", "amount": [[0.6, 0.0], [0.9, 0.02]],
                 "along": "x", "ramp_from": 0.0, "ramp_to": 0.1, "ramp_max": 1.0},
                {"kind": "flatten", "axis": "-y", "start": -0.8, "end": -0.85},
            ],
            "why": "a cap with every option",
        }),
        "knob": ("meta", {
            "elements": [{"kind": "ball", "centre": [0.0, 0.0, 0.0], "half": [0.05] * 3},
                         {"kind": "ellipsoid", "centre": [0.05, 0.0, 0.0],
                          "half": [0.06, 0.04, 0.05]}],
            "resolution": 0.01, "threshold": 0.6,
            "smooth": {"passes": 1, "factor": 0.4}, "decimate": 0.5,
        }),
        "dot": ("decal", {
            "half": [0.02, 0.03], "segments": 12, "rings": 3, "at": [0.0, 0.5, 0.0],
            "along": [0.0, 0.0, -1.0], "why": "a printed dot",
        }),
    }
    for name, (shape, form) in extra.items():
        piece["parts"].append({"name": name, "listed": "soft", "shape": shape, "on": "body",
                               "group": "extras", "form": form,
                               # a decal is printed: the loader refuses a decal with a solid role
                               "role": "decal" if shape == "decal" else "solid"})
        piece["rig"]["bind"][name] = {"bones": ["body"], "mode": "rigid"}
    pack = _load(raw, tmp_path)
    forms = {p.name: p.form for p in pack.pieces["toy"].parts}
    assert isinstance(forms["knob"].body, MetaForm)
    assert isinstance(forms["dot"].body, DecalForm)
    assert isinstance(forms["cap"].deforms[0], Push)
    assert isinstance(forms["cap"].deforms[1], Flatten)
    out = tmp_path / "again.pack.json"
    out.write_text(pack_text(pack, out))  # pack_text itself reads the text back and compares
    assert load_pack(out, STYLE) == pack


def test_old_packs_unchanged():
    pack = load_pack(FIX / "shop.pack.json", STYLE)
    for piece in pack.pieces.values():
        assert all(p.form is None for p in piece.parts)
        if piece.rig is not None:
            assert all(b.rest is None for b in piece.rig.bones)
            assert all(b.falloff is None for b in piece.rig.bind.values())
    text = pack_text(pack, Path("shop.pack.json"))
    assert not any(f'"{k}"' in text for k in ("form", "rest", "falloff"))
    assert pack_json(pack)["pieces"]["Doll"]["rig"]["bones"][0] == {
        "name": "root", "parent": None, "length": 0.1}


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _coords(u, falloff):
    """Points whose coordinate along `falloff.along` sits at u."""
    c = falloff.start + (falloff.end - falloff.start) * np.asarray(u, float)
    pts = np.full((len(c), 3), 0.7)
    pts[:, "xyz".index(falloff.along)] = c
    return pts


def test_chain_weights_clamp_and_sum():
    falloff = Falloff("y", 1.0, 0.0, ((0.30, 0.45), (0.55, 0.70)))
    us = np.array([-0.5, 0.0, 0.29, 0.375, 0.5, 0.625, 0.71, 1.0, 1.5])
    w = chain_weights(_coords(us, falloff), falloff, 3)
    assert w.shape == (9, 3)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
    assert np.allclose(w[us <= 0.30], [1, 0, 0], atol=1e-12)
    assert np.allclose(w[us >= 0.70], [0, 0, 1], atol=1e-12)
    s0, s1 = _smooth((us - 0.30) / 0.15), _smooth((us - 0.55) / 0.15)
    assert np.allclose(w[:, 0], 1 - s0, atol=1e-12)  # wu
    assert np.allclose(w[:, 1], s0 - s1, atol=1e-12)  # wf
    assert np.allclose(w[:, 2], s1, atol=1e-12)  # wh


def test_chain_weights_reads_the_falloff_axis_and_bad_input():
    falloff = Falloff("x", 0.0, 2.0, ((0.25, 0.75),))
    w = chain_weights(np.array([[0.0, 9.0, 9.0], [1.0, 9.0, 9.0], [2.0, 9.0, 9.0]]), falloff, 2)
    assert np.allclose(w, [[1, 0], [0.5, 0.5], [0, 1]])
    with pytest.raises(ValueError, match="bands"):
        chain_weights(np.zeros((1, 3)), falloff, 3)
    with pytest.raises(ValueError, match="shape"):
        chain_weights(np.zeros(4), falloff, 2)


def _drop_bind(raw):
    del _piece(raw)["rig"]["bind"]["arm_R"]


def _later_seat(raw):
    _piece(raw)["parts"][1]["form"]["seat"] = {"on": "arm_R", "at": [0.0, 0.0], "sink": 0.0}


def _mirror_of_mirror(raw):
    piece = _piece(raw)
    piece["parts"].append({"name": "arm_R2", "listed": "soft", "shape": "ring", "on": "body",
                           "group": "arms", "role": "solid", "form": {"mirror_of": "arm_R"}})
    piece["rig"]["bind"]["arm_R2"] = {"bones": ["body"], "mode": "rigid"}


def _form_on_box(raw):
    piece = _piece(raw)
    piece["parts"].append({"name": "crate", "listed": "soft", "shape": "box", "on": None,
                           "group": "x", "role": "solid", "form": piece["parts"][0]["form"]})
    piece["rig"]["bind"]["crate"] = {"bones": ["body"], "mode": "rigid"}


def _set(path, value):
    def edit(raw):
        node = _piece(raw)
        for step in path[:-1]:
            node = node[step]
        node[path[-1]] = value
    return edit


def _drop_form(raw):
    del _piece(raw)["parts"][1]["form"]


def _swap_bones(raw):
    bones = _piece(raw)["rig"]["bones"]
    bones[1], bones[2] = bones[2], bones[1]


def _lose_roll(raw):
    del _piece(raw)["rig"]["bones"][0]["rest"]["roll"]


REJECTS = [
    ("form on a box part", _form_on_box, "parts[3].form"),
    ("seat on a later part", _later_seat, "parts[1].form.seat.on"),
    ("mirror of a mirror", _mirror_of_mirror, "parts[3].form.mirror_of"),
    ("mirror of a missing part", _set(["parts", 2, "form", "mirror_of"], "arm_X"),
     "parts[2].form.mirror_of"),
    ("mirror of a later part", _set(["parts", 1, "form"], {"mirror_of": "arm_R"}),
     "parts[1].form.mirror_of"),
    ("mirror of another shape", _set(["parts", 2, "shape"], "meta"), "parts[2].form.mirror_of"),
    ("mirror of a form-less part", _drop_form, "parts[2].form.mirror_of"),
    ("rest length disagrees", _set(["rig", "bones", 2, "length"], 0.5),
     "rig.bones[2].rest"),
    ("child bone before its parent", _swap_bones, "rig.bones[1].parent"),
    ("rest without a roll", _lose_roll, "rig.bones[0].rest"),
    ("too many bands", _set(["rig", "bind", "arm_L", "falloff", "bands"],
                            [[0.2, 0.3], [0.4, 0.6]]), "rig.bind.arm_L.falloff.bands"),
    ("no bands", _set(["rig", "bind", "arm_L", "falloff", "bands"], []),
     "rig.bind.arm_L.falloff.bands"),
    ("band runs backwards", _set(["rig", "bind", "arm_L", "falloff", "bands"], [[0.6, 0.4]]),
     "rig.bind.arm_L.falloff.bands[0]"),
    ("band outside 0..1", _set(["rig", "bind", "arm_L", "falloff", "bands"], [[0.4, 1.2]]),
     "rig.bind.arm_L.falloff.bands[0]"),
    ("falloff span of zero", _set(["rig", "bind", "arm_L", "falloff", "end"], 0.6),
     "rig.bind.arm_L.falloff.end"),
    ("falloff on an unknown axis", _set(["rig", "bind", "arm_L", "falloff", "along"], "w"),
     "rig.bind.arm_L.falloff.along"),
    ("falloff on a rigid bind",
     _set(["rig", "bind", "body", "falloff"],
          {"along": "y", "start": 1.0, "end": 0.0, "bands": []}), "rig.bind.body.falloff"),
    ("a part with no bind", _drop_bind, "rig.bind"),
]


@pytest.mark.parametrize("edit,where", [r[1:] for r in REJECTS], ids=[r[0] for r in REJECTS])
def test_soft_fields_reject(edit, where, tmp_path):
    raw = copy.deepcopy(_raw())
    edit(raw)
    with pytest.raises(ManifestError) as caught:
        _load(raw, tmp_path)
    message = str(caught.value)
    assert "toy.pack.json" in message
    assert f"pieces.toy.{where}" in message
