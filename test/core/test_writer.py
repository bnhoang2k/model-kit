"""write_pack: the one way a pack reaches disk, and it reads back as the same records."""

import dataclasses
import json
from pathlib import Path

import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import (
    Avoid,
    Hold,
    Layout,
    Lift,
    Listed,
    Mount,
    Ref,
    Stand,
    View,
    load_clips,
    load_layout,
    load_pack,
)
from model_kit.core.units import load_style
from model_kit.core.writer import _body_json, clips_json, write_layout, write_pack

FIX = Path(__file__).parent / "fixtures"
STYLE = load_style(FIX / "shop.style.json")


def _shop():
    return load_pack(FIX / "shop.pack.json", STYLE)


def test_pack_round_trip(tmp_path):
    pack = _shop()
    out = write_pack(pack, tmp_path / "shop.pack.json")
    assert out == tmp_path / "shop.pack.json"
    assert load_pack(out, STYLE) == pack
    # A second trip changes nothing: the file is a fixed point.
    again = write_pack(load_pack(out, STYLE), tmp_path / "again.pack.json")
    assert again.read_text() == out.read_text()
    # The file gets the mode any new file gets here, not the staging file's 0600.
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    assert out.stat().st_mode == plain.stat().st_mode


def test_round_trip_keeps_optional_fields_and_binds_under_rig(tmp_path):
    pack = _shop()
    raw = json.loads(write_pack(pack, tmp_path / "shop.pack.json").read_text())
    bench, lamp, doll = (raw["pieces"][n] for n in ("Bench", "Lamp", "Doll"))
    assert bench["listed"][1]["owner"] == "Bench" and "owner" not in bench["listed"][0]
    assert bench["holds"][0]["sub_box"] == [[0.1, 0.42, 0.1], [0.3, 0.46, 0.3]]
    assert "sub_box" not in lamp["holds"][0]
    assert lamp["mounts"][0]["fits"] == ["socket"]
    # Binds live only under rig.bind; a part never carries bones or a bind mode.
    assert all(set(p) == {"name", "listed", "shape", "on", "group", "role"}
               for piece in raw["pieces"].values() for p in piece["parts"])
    assert doll["rig"]["bind"]["arm_R"] == {"bones": ["arm.R", "forearm.R", "hand.R"],
                                            "mode": "chain"}
    assert "rig" not in bench


@pytest.mark.parametrize("modifiers", [(), ("Outline",), ("Outline", "Shell")])
def test_outline_modifiers_round_trip(tmp_path, modifiers):
    pack = _shop()
    doll = pack.pieces["Doll"]
    pack = dataclasses.replace(pack, pieces={**pack.pieces, "Doll": dataclasses.replace(
        doll, rig=dataclasses.replace(doll.rig, outline_modifiers=modifiers))})
    path = write_pack(pack, tmp_path / "p.json")
    assert json.loads(path.read_text())["pieces"]["Doll"]["rig"]["outline_modifiers"] == \
        list(modifiers)
    assert load_pack(path, STYLE) == pack


def test_rig_round_trip_with_new_fields(tmp_path):
    from model_kit.core.manifest import Joint

    pack = _shop()
    doll = pack.pieces["Doll"]
    why = "a cited table"
    rig = dataclasses.replace(
        doll.rig, root="root", supports=("foot.R",),
        sit=dataclasses.replace(doll.rig.sit, past_edge=("forearm.R",)),
        joints={"foot.R": Joint({"x": (-30.0, 45.0)}, why)})
    pack = dataclasses.replace(pack, pieces={**pack.pieces,
                                             "Doll": dataclasses.replace(doll, rig=rig)})
    path = write_pack(pack, tmp_path / "p.json")
    out = json.loads(path.read_text())["pieces"]["Doll"]["rig"]
    assert out["root"] == "root" and out["supports"] == ["foot.R"]
    assert out["sit"]["past_edge"] == ["forearm.R"]
    assert out["joints"] == {"foot.R": {"range_deg": {"x": [-30.0, 45.0]}, "why": why}}
    assert load_pack(path, STYLE) == pack
    # A rig with none of them writes none of them, and no sit or mirror is not a null.
    bare = dataclasses.replace(doll.rig, sit=None, mirror=None, supports=(), joints={},
                               root=None)
    pack = dataclasses.replace(pack, pieces={**pack.pieces,
                                             "Doll": dataclasses.replace(doll, rig=bare)})
    path = write_pack(pack, tmp_path / "bare.json")
    out = json.loads(path.read_text())["pieces"]["Doll"]["rig"]
    assert not {"sit", "mirror", "supports", "joints", "root"} & set(out)
    assert load_pack(path, STYLE) == pack


def test_round_trip_of_none_optionals_built_in_code(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    lamp = dataclasses.replace(
        lamp,
        listed=(Listed("stem", "Steel tube", "a catalogue", 0.0012, None),),
        holds=(Hold("h", "side", "L", ("stem", "base"), ((0, 0, 0), (1, 1, 1)), (0, 1, 0),
                    (0, 0, 1), False),),
        mounts=(Mount("plug", "power", (0, 0, 0), (0, 0, -1), "lead", None),),
    )
    pack = dataclasses.replace(pack, pieces={"Lamp": lamp})
    assert load_pack(write_pack(pack, tmp_path / "p.json"), STYLE) == pack


def test_mount_part_and_end_plug_round_trip(tmp_path):
    """`Mount.part` and `End.plug` are written, and `pack_text` reads them back equal; a mount
    with no part and an end with no plug write neither key."""
    from model_kit.core.writer import pack_text

    pack = _shop()
    lamp = pack.pieces["Lamp"]
    (mount,) = lamp.mounts
    (conn,) = lamp.connectors
    near, far = conn.ends
    assert mount.part is None and far.plug is None  # the fixture names neither
    bare = json.loads(pack_text(pack, tmp_path / "shop.pack.json"))["pieces"]["Lamp"]
    assert "part" not in bare["mounts"][0] and "plug" not in bare["connectors"][0]["ends"][1]

    named = dataclasses.replace(
        lamp,
        mounts=(dataclasses.replace(mount, part="base"),),
        connectors=(dataclasses.replace(conn, ends=(near, dataclasses.replace(far, plug="base"))),))
    pack = dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": named})
    text = pack_text(pack, tmp_path / "shop.pack.json")  # checks the read-back equals the pack
    written = json.loads(text)["pieces"]["Lamp"]
    assert written["mounts"][0]["part"] == "base"
    assert written["connectors"][0]["ends"][1]["plug"] == "base"
    assert "plug" not in written["connectors"][0]["ends"][0]
    path = write_pack(pack, tmp_path / "named.pack.json")
    back = load_pack(path, STYLE).pieces["Lamp"]
    assert back.mounts[0].part == "base" and back.connectors[0].ends[1].plug == "base"
    assert load_pack(path, STYLE) == pack


def test_a_mount_part_or_end_plug_that_names_no_part_is_refused(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    (mount,) = lamp.mounts
    bad_mount = dataclasses.replace(lamp, mounts=(dataclasses.replace(mount, part="ghost"),))
    with pytest.raises(ManifestError, match="ghost"):
        write_pack(dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": bad_mount}),
                   tmp_path / "a.json")
    (conn,) = lamp.connectors
    near, far = conn.ends
    bad_end = dataclasses.replace(
        lamp, connectors=(dataclasses.replace(
            conn, ends=(near, dataclasses.replace(far, plug="ghost"))),))
    with pytest.raises(ManifestError, match="ghost"):
        write_pack(dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": bad_end}),
                   tmp_path / "b.json")
    attached = dataclasses.replace(
        lamp, connectors=(dataclasses.replace(
            conn, ends=(dataclasses.replace(near, plug="base"), far)),))
    with pytest.raises(ManifestError, match="self end"):
        write_pack(dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": attached}),
                   tmp_path / "c.json")
    assert list(tmp_path.iterdir()) == []


def test_measured_that_json_cannot_hold_is_refused(tmp_path):
    pack = _shop()
    bench = dataclasses.replace(pack.pieces["Bench"], measured={"foot": (0.6, 0.3)})
    bad = dataclasses.replace(pack, pieces={"Bench": bench})
    with pytest.raises(ManifestError, match=r"Bench.*measured.*foot"):
        write_pack(bad, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_a_pack_the_loader_refuses_is_not_written(tmp_path):
    pack = _shop()
    bench = pack.pieces["Bench"]
    hold = dataclasses.replace(bench.holds[0], surface=("no_such_part",))
    bad = dataclasses.replace(pack, pieces={"Bench": dataclasses.replace(bench, holds=(hold,))})
    target = tmp_path / "p.json"
    target.write_text("old")
    with pytest.raises(ManifestError, match="no_such_part"):
        write_pack(bad, target)
    assert target.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["p.json"]
    # The error names the user's file, not the staging copy.
    with pytest.raises(ManifestError, match=str(target)):
        write_pack(bad, target)


def test_dict_key_must_match_piece_name(tmp_path):
    pack = _shop()
    bad = dataclasses.replace(pack, pieces={"Seat": pack.pieces["Bench"]})
    with pytest.raises(ManifestError, match="Seat.*Bench"):
        write_pack(bad, tmp_path / "p.json")


def test_a_mismatch_names_the_first_field_that_differs(tmp_path):
    pack = _shop()
    bench = pack.pieces["Bench"]
    hold = dataclasses.replace(bench.holds[0], surface=["seat_top"])  # a list reads back a tuple
    bad = dataclasses.replace(pack, pieces={"Bench": dataclasses.replace(bench, holds=(hold,))})
    with pytest.raises(ManifestError, match=r"pieces\.Bench\.holds: wrote .*read back"):
        write_pack(bad, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_pack_text_checks_without_writing(tmp_path):
    from model_kit.core.writer import pack_text

    pack = _shop()
    text = pack_text(pack, tmp_path / "p.json")
    assert json.loads(text)["pack"] == "shop"
    assert list(tmp_path.iterdir()) == []


def test_new_pack_records_round_trip(tmp_path):
    pack = _shop()
    path = write_pack(pack, tmp_path / "shop.pack.json")
    raw = json.loads(path.read_text())
    assert raw["spacing"] == 2.0
    lamp = raw["pieces"]["Lamp"]
    assert lamp["connectors"][0]["ends"][0] == {"point": [0.0, 0.01, -0.1], "host": "self.plug"}
    assert lamp["connectors"][0]["ends"][1]["layout"] == "shop"
    assert lamp["keep_off"][0]["id"] == "bulb"
    assert [a["id"] for a in raw["pieces"]["Doll"]["rig"]["avoid"]] == ["chest", "lap"]
    back = load_pack(path, STYLE)
    assert back == pack
    assert back.pieces["Lamp"].connectors == pack.pieces["Lamp"].connectors
    assert back.pieces["Doll"].rig.avoid == pack.pieces["Doll"].rig.avoid


def test_defaults_stay_out_of_the_file(tmp_path):
    pack = _shop()
    pieces = {n: dataclasses.replace(p, connectors=(), keep_off=()) for n, p in pack.pieces.items()}
    doll = pieces["Doll"]
    pieces["Doll"] = dataclasses.replace(doll, rig=dataclasses.replace(doll.rig, avoid=()))
    bare = dataclasses.replace(pack, spacing=None, pieces=pieces)
    raw = json.loads(write_pack(bare, tmp_path / "p.json").read_text())
    assert "spacing" not in raw
    assert all("connectors" not in p and "keep_off" not in p for p in raw["pieces"].values())
    assert "avoid" not in raw["pieces"]["Doll"]["rig"]
    assert load_pack(tmp_path / "p.json", STYLE) == bare


def test_a_connector_the_loader_refuses_is_not_written(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    conn = dataclasses.replace(lamp.connectors[0], part="stem")  # a tube, not a cable
    bad = dataclasses.replace(pack, pieces={"Lamp": dataclasses.replace(lamp, connectors=(conn,))})
    with pytest.raises(ManifestError, match="cable"):
        write_pack(bad, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_avoid_built_in_code_round_trips(tmp_path):
    pack = _shop()
    doll = pack.pieces["Doll"]
    avoid = (Avoid("head", "body", "box", (0, 0.3, 0), (0.1, 0.1, 0.1), "keep clear of the head"),)
    rig = dataclasses.replace(doll.rig, avoid=avoid)
    pack = dataclasses.replace(pack, pieces={**pack.pieces,
                                             "Doll": dataclasses.replace(doll, rig=rig)})
    assert load_pack(write_pack(pack, tmp_path / "p.json"), STYLE) == pack


def test_clip_loop_and_frame_round_trip(tmp_path):
    clips = {c.name: c for c in load_clips(FIX / "shop.clips.json")}
    idle = dataclasses.replace(clips["sit_idle"], loop=True, loop_from=3)
    raw = clips_json({**clips, "sit_idle": idle})["clips"]
    assert raw["sit_idle"]["loop"] is True and raw["sit_idle"]["loop_from"] == 3
    assert all(k not in raw["sit_reach"] for k in ("loop", "loop_from", "frame"))
    path = tmp_path / "clips.json"
    path.write_text(json.dumps({"clips": raw}))
    assert {c.name: c for c in load_clips(path)}["sit_idle"] == idle


def _layout() -> Layout:
    return load_layout(FIX / "layouts" / "shop.json")


def test_layout_round_trip(tmp_path):
    layout = _layout()
    out = write_layout(layout, tmp_path / "layouts" / "shop.json")
    assert out == tmp_path / "layouts" / "shop.json"
    assert load_layout(out) == layout
    raw = json.loads(out.read_text())
    assert raw["placements"]["Lamp_1"]["on"] == "shop:Bench_1.socket"
    assert "on" not in raw["placements"]["Bench_1"]
    again = write_layout(load_layout(out), tmp_path / "again.json")
    assert again.read_text() == out.read_text()  # a fixed point


def test_layout_the_loader_refuses_is_not_written(tmp_path):
    layout = _layout()
    sideways = dataclasses.replace(layout.stands["bench"], facing=(0.0, 1.0, 0.0))
    bad = dataclasses.replace(layout, stands={"bench": sideways})
    target = tmp_path / "shop.json"
    target.write_text("old")
    with pytest.raises(ManifestError, match="facing") as err:
        write_layout(bad, target)
    assert str(target) in str(err.value) and "model-kit-layout-" not in str(err.value)
    assert target.read_text() == "old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["shop.json"]


def test_layout_dict_keys_must_match_their_records(tmp_path):
    layout = _layout()
    wrong = dataclasses.replace(layout, stands={"other": layout.stands["bench"]})
    with pytest.raises(ManifestError, match="other"):
        write_layout(wrong, tmp_path / "shop.json")
    assert Stand("a", (0, 0, 0), (1.0, 0.0, 0.0), "stand", "c").clip == "c"
    assert Ref.parse("shop:Bench_1.socket") == layout.placements["Lamp_1"].on


def test_seat_when_round_trips(tmp_path):
    clips = {c.name: c for c in load_clips(FIX / "shop.clips.json")}
    timed = dataclasses.replace(clips["sit_reach"], seat_when=((20, 25),))
    data = clips_json({**clips, "sit_reach": timed})
    assert data["clips"]["sit_reach"]["seat_when"] == [[20, 25]]
    assert "seat_when" not in data["clips"]["sit_idle"]
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(data))
    back = {c.name: c for c in load_clips(path)}
    assert back["sit_reach"].seat_when == ((20, 25),)
    assert back["sit_idle"].seat_when is None


def test_an_unknown_form_body_is_refused():
    class Stand:
        pass

    with pytest.raises(ManifestError, match="Stand"):
        _body_json(Stand())


def test_view_round_trip(tmp_path):
    pack = _shop()
    lamp, doll = pack.pieces["Lamp"], pack.pieces["Doll"]
    pack = dataclasses.replace(pack, pieces={
        **pack.pieces,
        "Lamp": dataclasses.replace(lamp, view=View(0.8, "seen from the desk")),
        "Doll": dataclasses.replace(doll, view=View(0.3, "held in the hand", ("back",),
                                                    "it is held facing the viewer"))})
    path = write_pack(pack, tmp_path / "p.json")
    raw = json.loads(path.read_text())["pieces"]
    assert raw["Lamp"]["view"] == {"nearest_m": 0.8, "why": "seen from the desk"}
    assert raw["Doll"]["view"] == {"nearest_m": 0.3, "why": "held in the hand",
                                   "hidden": ["back"], "hidden_why": "it is held facing the viewer"}
    assert load_pack(path, STYLE) == pack
    again = write_pack(load_pack(path, STYLE), tmp_path / "again.json")
    assert again.read_text() == path.read_text()


def test_outlined_round_trip(tmp_path):
    """`outlined` and its reason are written only when the piece is not outlined, so an outlined
    piece's file is as it was; what is written reads back equal."""
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    plain = json.loads(write_pack(pack, tmp_path / "plain.json").read_text())["pieces"]
    assert all("outlined" not in p and "outlined_why" not in p for p in plain.values())
    unlined = dataclasses.replace(pack, pieces={
        **pack.pieces,
        "Lamp": dataclasses.replace(lamp, outlined=False, outlined_why="the page draws it flat")})
    path = write_pack(unlined, tmp_path / "p.json")
    raw = json.loads(path.read_text())["pieces"]
    assert (raw["Lamp"]["outlined"], raw["Lamp"]["outlined_why"]) == (
        False, "the page draws it flat")
    assert "outlined" not in raw["Doll"]
    assert load_pack(path, STYLE) == unlined
    again = write_pack(load_pack(path, STYLE), tmp_path / "again.json")
    assert again.read_text() == path.read_text()


def test_an_outlined_piece_the_loader_refuses_is_not_written(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    for edit in ({"outlined": False}, {"outlined_why": "stale"}):
        bad = dataclasses.replace(pack, pieces={"Lamp": dataclasses.replace(lamp, **edit)})
        with pytest.raises(ManifestError, match="outlined_why|unlined"):
            write_pack(bad, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_view_farthest_round_trips(tmp_path):
    pack = _shop()
    lamp, doll = pack.pieces["Lamp"], pack.pieces["Doll"]
    pack = dataclasses.replace(pack, pieces={
        **pack.pieces,
        "Lamp": dataclasses.replace(lamp, view=View(0.8, "seen from the desk", farthest_m=3.0,
                                                    farthest_why="seen across the room")),
        "Doll": dataclasses.replace(doll, view=View(0.3, "held", ("back",), "held facing out",
                                                    1.5, "seen at arm's length and a bit more"))})
    path = write_pack(pack, tmp_path / "p.json")
    raw = json.loads(path.read_text())["pieces"]
    assert raw["Lamp"]["view"] == {"nearest_m": 0.8, "why": "seen from the desk",
                                   "farthest_m": 3.0, "farthest_why": "seen across the room"}
    assert raw["Doll"]["view"]["farthest_m"] == 1.5
    assert load_pack(path, STYLE) == pack
    again = write_pack(load_pack(path, STYLE), tmp_path / "again.json")
    assert again.read_text() == path.read_text()
    # a view with no farthest writes neither key
    plain = dataclasses.replace(pack.pieces["Lamp"], view=View(0.8, "seen from the desk"))
    plain_pack = dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": plain})
    raw = json.loads(write_pack(plain_pack, tmp_path / "plain.json").read_text())["pieces"]
    assert raw["Lamp"]["view"] == {"nearest_m": 0.8, "why": "seen from the desk"}


def test_a_view_the_loader_refuses_cannot_be_built_and_a_legal_one_round_trips(tmp_path):
    """The writer once had to refuse a bad view. The `View` refuses to be built, so the writer only
    ever meets a legal one, which reads back equal."""
    with pytest.raises(ManifestError, match="above 0"):
        View(0.0, "x")
    pack = _shop()
    lamp = dataclasses.replace(pack.pieces["Lamp"], view=View(1.0, "x", ("back",), "a wall"))
    good = dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": lamp})
    assert load_pack(write_pack(good, tmp_path / "p.json"), STYLE) == good


def test_fastener_round_trip(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    parts = tuple(dataclasses.replace(p, fastener=True) if p.name == "stem" else
                  dataclasses.replace(p, retains=("base", "stem")) if p.name == "lead" else p
                  for p in lamp.parts)
    pack = dataclasses.replace(pack, pieces={**pack.pieces,
                                             "Lamp": dataclasses.replace(lamp, parts=parts)})
    path = write_pack(pack, tmp_path / "p.json")
    raw = {p["name"]: p for p in json.loads(path.read_text())["pieces"]["Lamp"]["parts"]}
    assert raw["stem"]["fastener"] is True and "retains" not in raw["stem"]
    assert raw["lead"]["retains"] == ["base", "stem"] and "fastener" not in raw["lead"]
    assert "fastener" not in raw["base"] and "retains" not in raw["base"]
    assert load_pack(path, STYLE) == pack


def test_finish_round_trip(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    listed = tuple(dataclasses.replace(x, finish="print") if i == 0 else x
                   for i, x in enumerate(lamp.listed))
    pack = dataclasses.replace(pack, pieces={**pack.pieces,
                                             "Lamp": dataclasses.replace(lamp, listed=listed)})
    path = write_pack(pack, tmp_path / "p.json")
    rows = json.loads(path.read_text())["pieces"]["Lamp"]["listed"]
    assert rows[0]["finish"] == "print"
    assert all("finish" not in row for row in rows[1:])  # the default is not written
    assert load_pack(path, STYLE) == pack


def test_held_round_trip(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    listed = tuple(dataclasses.replace(x, held="bonded") if i == 0 else x
                   for i, x in enumerate(lamp.listed))
    pack = dataclasses.replace(pack, pieces={**pack.pieces,
                                             "Lamp": dataclasses.replace(lamp, listed=listed)})
    path = write_pack(pack, tmp_path / "p.json")
    rows = json.loads(path.read_text())["pieces"]["Lamp"]["listed"]
    assert rows[0]["held"] == "bonded"
    assert all("held" not in row for row in rows[1:])  # the default is not written
    assert load_pack(path, STYLE) == pack


def test_retains_the_loader_refuses_is_not_written(tmp_path):
    pack = _shop()
    lamp = pack.pieces["Lamp"]
    parts = tuple(dataclasses.replace(p, retains=("Ghost",)) if p.name == "lead" else p
                  for p in lamp.parts)
    bad = dataclasses.replace(pack, pieces={**pack.pieces,
                                            "Lamp": dataclasses.replace(lamp, parts=parts)})
    with pytest.raises(ManifestError, match="names no part of this piece"):
        write_pack(bad, tmp_path / "p.json")
    assert list(tmp_path.iterdir()) == []


def test_posture_and_lifts_round_trip(tmp_path):
    clips = {c.name: c for c in load_clips(FIX / "shop.clips.json")}
    walk = dataclasses.replace(
        clips["sit_idle"], name="walk", seat=None, seat_when=None, mirror_of=None,
        posture="stand",
        lifts=(Lift("foot.R", ((5, 9), (12, 14)), "a step"),))
    data = clips_json({**clips, "walk": walk})
    assert data["clips"]["walk"]["posture"] == "stand"
    assert data["clips"]["walk"]["lifts"] == [
        {"support": "foot.R", "when": [[5.0, 9.0], [12.0, 14.0]], "why": "a step"}]
    sat = data["clips"]["sit_idle"]
    assert sat["posture"] == "sit" and "lifts" not in sat
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(data))
    assert {c.name: c for c in load_clips(path)}["walk"] == walk


def test_a_clip_without_posture_leaves_it_out_of_the_file(tmp_path):
    clips = {c.name: dataclasses.replace(c, posture=None) for c in load_clips(
        FIX / "shop.clips.json")}
    data = clips_json(clips)
    assert all("posture" not in c and "lifts" not in c for c in data["clips"].values())
    path = tmp_path / "clips.json"
    path.write_text(json.dumps(data))
    assert {c.name: c for c in load_clips(path)} == clips


def test_rests_on_round_trips(tmp_path):
    layout = _layout()
    lamp = dataclasses.replace(layout.placements["Lamp_2"], rests_on="Bench_1")
    layout = dataclasses.replace(layout, placements={**layout.placements, "Lamp_2": lamp})
    out = write_layout(layout, tmp_path / "shop.json")
    raw = json.loads(out.read_text())["placements"]
    assert raw["Lamp_2"]["rests_on"] == "Bench_1"
    assert all("rests_on" not in raw[k] for k in ("Bench_1", "Lamp_1"))
    assert load_layout(out) == layout


def test_layout_that_rests_on_nothing_is_not_written(tmp_path):
    layout = _layout()
    lamp = dataclasses.replace(layout.placements["Lamp_2"], rests_on="Table")
    bad = dataclasses.replace(layout, placements={**layout.placements, "Lamp_2": lamp})
    with pytest.raises(ManifestError, match="names no placement"):
        write_layout(bad, tmp_path / "shop.json")
    assert list(tmp_path.iterdir()) == []
