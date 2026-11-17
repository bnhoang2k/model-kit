import copy
import dataclasses
import inspect
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import AnchorMissing, ManifestError
from model_kit.core.manifest import (
    AVOID_KINDS,
    FINISHES,
    FRAME_AXES,
    HELD,
    MOUNT_KINDS,
    NO_OUTLINE_ROLES,
    PART_ROLES,
    SHAPES,
    Avoid,
    Clip,
    Connector,
    End,
    FloorGear,
    KeepOff,
    Layout,
    LayoutPlacement,
    Lift,
    Mount,
    Pack,
    Part,
    Piece,
    Placement,
    Ref,
    RigSpec,
    Stand,
    View,
    anchor,
    check_connectors,
    load_clips,
    load_layout,
    load_layouts,
    load_pack,
    parse_view,
    seat_windows,
)
from model_kit.core.units import Style, load_style

FIX = Path(__file__).parent / "fixtures"


def shop_style() -> Style:
    return load_style(FIX / "shop.style.json")


def shop_pack() -> Pack:
    return load_pack(FIX / "shop.pack.json", shop_style())


def shop_clips() -> dict[str, Clip]:
    return {clip.name: clip for clip in load_clips(FIX / "shop.clips.json")}


def pack_data() -> dict:
    return json.loads((FIX / "shop.pack.json").read_text())


def clips_data() -> dict:
    return json.loads((FIX / "shop.clips.json").read_text())


def load_pack_from(tmp_path: Path, data: dict, style: Style | None = None) -> Pack:
    path = tmp_path / "test.pack.json"
    path.write_text(json.dumps(data))
    return load_pack(path, style or shop_style())


def load_clips_from(tmp_path: Path, data: dict):
    path = tmp_path / "test.clips.json"
    path.write_text(json.dumps(data))
    return load_clips(path)


@contextmanager
def fails(match: str):
    """Expect a ManifestError whose reason matches: the text after the file and the field.

    The field path is cut off, so a test cannot pass because the field's name matches.
    """
    with pytest.raises(ManifestError) as err:
        yield
    reason = str(err.value).split(".json: ", 1)[-1].split(": ", 1)[-1]
    assert re.search(match, reason), reason


# --- the fixtures load whole ---------------------------------------------------------------


def test_fixtures_load():
    pack = shop_pack()
    assert pack.name == "shop" and pack.units_per_metre == 0.5
    assert list(pack.pieces) == ["Bench", "Lamp", "Doll"]
    bench = pack.pieces["Bench"]
    assert bench.name == "Bench"
    assert bench.holds[0].surface == ("seat_top",)
    assert bench.holds[0].sub_box == ((0.1, 0.42, 0.1), (0.3, 0.46, 0.3))
    assert bench.seats[0].point == (0.1, 0.45, 0.2)
    assert bench.rig is None and bench.parts[0].bones == () and bench.parts[0].bind is None
    assert bench.measured["heights"] == {"seat": 0.45}

    doll = pack.pieces["Doll"]
    chain = doll.rig.chains["arm.R"]
    assert chain.dofs == {"arm.R": "xyz", "forearm.R": "x", "hand.R": "xyz"}
    assert chain.limits == {"forearm.R:x": (-3.0, 0.05)}
    assert chain.effector_group == "hand.R" and chain.effector_radius == 0.0586
    assert doll.rig.sit.below_seat == 0.083 and doll.rig.sit.forward == 0.03
    assert doll.rig.mirror.axis == "x"
    assert doll.rig.mirror.pairs == (
        ("arm.R", "arm.L"), ("forearm.R", "forearm.L"), ("hand.R", "hand.L"))
    arm = next(p for p in doll.parts if p.name == "arm_R")
    assert arm.bones == ("arm.R", "forearm.R", "hand.R") and arm.bind == "chain"

    clips = shop_clips()
    assert list(clips) == ["sit_reach", "sit_idle", "sit_idle_L"]
    reach = clips["sit_reach"]
    assert reach.frames == (1, 48) and reach.authoring == "solved"
    assert reach.seat == Ref("shop", "Bench_L", "seat_1")
    assert reach.holds[2].dangle.budget == 0.2
    assert reach.targets[1].ref.is_bone and reach.targets[2].tool == "lamp"
    assert reach.targets[2].offset == (0.0, 0.0, 0.0) and reach.targets[2].why is None
    assert reach.floor_gear.pieces["Bench_L"].piece == "Bench"
    assert reach.props["lamp"].show_frames == ((1.0, 48.0),)
    assert pack.pieces["Lamp"].measured["tip_length"] == 0.3
    assert clips["sit_idle_L"].mirror_of == "sit_idle"


# --- the ref grammar -----------------------------------------------------------------------


def test_ref_grammar():
    ref = Ref.parse("shop:Bench.seat_1")
    assert (ref.pack, ref.piece, ref.id) == ("shop", "Bench", "seat_1")
    assert not ref.is_bone and str(ref) == "shop:Bench.seat_1"

    bone = Ref.parse("rig:body")
    assert bone.pack == "rig" and bone.is_bone and bone.id == "body"
    assert Ref.parse("rig:forearm.R").id == "forearm.R"
    assert str(Ref.parse("rig:hand.R")) == "rig:hand.R"

    for bad in ["Bench.seat_1", "shop:Bench", "shop:Bench.", ":Bench.seat_1", "shop:.seat_1",
                "shop:Bench.seat_1.x", "shop:Be nch.seat_1", "rig:", ""]:
        with pytest.raises(ManifestError):
            Ref.parse(bad)


# --- anchor() ------------------------------------------------------------------------------


def test_anchor_finds_each_kind():
    packs = {"shop": shop_pack()}
    for text, kind in [("shop:Bench.edge_R", "hold"), ("shop:Bench.seat_1", "seat"),
                       ("shop:Lamp.plug", "mount"), ("shop:Bench.front", "use")]:
        found = anchor(packs, text)
        assert found.kind == kind and found.item.id == text.split(".")[1]
        assert found.world is None
    assert anchor(packs, Ref("shop", "Bench", "seat_1")).piece.name == "Bench"


def test_anchor_raises_on_miss():
    packs = {"shop": shop_pack()}
    with pytest.raises(AnchorMissing, match="'nope'"):
        anchor(packs, "nope:Bench.seat_1")
    with pytest.raises(AnchorMissing, match="'Chair'"):
        anchor(packs, "shop:Chair.seat_1")
    with pytest.raises(AnchorMissing, match="'seat_9'"):
        anchor(packs, "shop:Bench.seat_9")


def test_anchor_has_no_default():
    params = inspect.signature(anchor).parameters
    assert "default" not in params
    assert [name for name, p in params.items() if p.default is not inspect.Parameter.empty] == [
        "clip"
    ]


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def test_anchor_resolves_placement_id():
    packs = {"shop": shop_pack()}
    clip = shop_clips()["sit_reach"]
    found = anchor(packs, "shop:Bench_L.seat_1", clip=clip)
    assert found.kind == "seat" and found.piece.name == "Bench"
    assert found.placed_by == "Bench_L"
    placement = clip.floor_gear.pieces["Bench_L"]
    expected = quat_matrix(placement.quaternion) @ np.array(found.item.point) + placement.position
    # a quarter turn about +Y takes (x, y, z) to (z, y, -x): (0.2, 0.45, -0.1), then the move
    assert np.allclose(expected, [1.2, 0.45, 1.9])
    assert np.allclose(found.world, expected, atol=1e-12)
    assert anchor(packs, "shop:Bench.seat_1").placed_by is None


def test_anchor_unplaced_piece_in_a_clip_raises():
    packs = {"shop": shop_pack()}
    with pytest.raises(AnchorMissing, match=r"shop:Bench\.seat_1.*clip 'sit_reach'"):
        anchor(packs, "shop:Bench.seat_1", clip=shop_clips()["sit_reach"])


def test_anchor_prop_piece_has_no_world_point():
    packs = {"shop": shop_pack()}
    found = anchor(packs, "shop:Lamp.switch_R", clip=shop_clips()["sit_reach"])
    assert found.kind == "hold" and found.piece.name == "Lamp"
    assert found.world is None and found.placed_by == "prop:lamp"


def test_a_clip_with_a_claimed_twice_name_cannot_be_built_in_code():
    """The clips that `anchor()` used to guard against cannot exist: the clip refuses them when it
    is built, so `anchor()` has no such clip to meet."""
    clip = shop_clips()["sit_reach"]
    with pytest.raises(ManifestError,
                       match=r"clip 'sit_reach': name 'Lamp' is claimed twice.*prop 'lamp_2'"):
        dataclasses.replace(clip, props={**clip.props, "lamp_2": clip.props["lamp"]})
    lamp = Placement("Lamp", "Bench", "shop", (5.0, 0.0, 5.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(ManifestError,
                       match=r"clip 'sit_reach': name 'Lamp' is claimed twice.*floor_gear 'Lamp'"):
        dataclasses.replace(clip, floor_gear=FloorGear({**clip.floor_gear.pieces, "Lamp": lamp}))


def test_carried_piece_may_also_stand_under_another_id(tmp_path):
    """shop:Lamp is the carried prop; shop:Lamp_desk is a standing copy of the same piece."""
    data = clips_data()
    data["clips"]["sit_reach"]["floor_gear"]["Lamp_desk"] = {
        "piece": "Lamp", "pack": "shop", "position": [2.0, 0.0, 0.0],
        "quaternion": [0.0, 0.0, 0.0, 1.0]}
    clip = {c.name: c for c in load_clips_from(tmp_path, data)}["sit_reach"]
    packs = {"shop": shop_pack()}
    carried = anchor(packs, "shop:Lamp.switch_R", clip=clip)
    assert carried.placed_by == "prop:lamp" and carried.world is None
    standing = anchor(packs, "shop:Lamp_desk.switch_R", clip=clip)
    assert standing.placed_by == "Lamp_desk" and standing.piece.name == "Lamp"
    assert np.allclose(standing.world, [2.0, 0.8, 0.0])


def test_every_fixture_ref_resolves_in_its_clip():
    packs = {"shop": shop_pack()}
    for clip in shop_clips().values():
        refs = [h.ref for h in clip.holds] + [t.ref for t in clip.targets if not t.ref.is_bone]
        for ref in refs + ([clip.seat] if clip.seat else []):
            anchor(packs, ref, clip=clip)


def test_anchor_placed_id_needs_its_placement():
    packs = {"shop": shop_pack()}
    with pytest.raises(AnchorMissing, match="'Bench_L'"):
        anchor(packs, "shop:Bench_L.seat_1")


def test_anchor_placement_pack_must_match_ref():
    packs = {"shop": shop_pack(), "other": shop_pack()}
    with fails("placement 'Bench_L' is of pack 'shop'"):
        anchor(packs, "other:Bench_L.seat_1", clip=shop_clips()["sit_reach"])


def test_anchor_rejects_bone_ref():
    with fails("a bone ref is resolved from samples"):
        anchor({"shop": shop_pack()}, "rig:body")


def test_placement_maps_both_ways():
    placement = shop_clips()["sit_reach"].floor_gear.pieces["Bench_L"]
    points = np.array([[0.1, 0.45, 0.2], [0.0, 0.0, 0.0], [-0.3, 1.0, 0.5]])
    world = placement.to_world(points)
    assert np.allclose(world[1], placement.position)
    assert np.allclose(placement.to_piece(world), points, atol=1e-12)


# --- units ---------------------------------------------------------------------------------


def test_pack_without_units_raises(tmp_path):
    data = pack_data()
    del data["units_per_metre"]
    with fails("missing field 'units_per_metre'"):
        load_pack_from(tmp_path, data)
    data["unit"] = "m"
    with fails("missing field 'units_per_metre'"):
        load_pack_from(tmp_path, data)


def test_pack_style_units_mismatch_raises(tmp_path):
    style = Style(units_per_metre=1.0, page_camera={}, dials={})
    with fails("differs from the style's units_per_metre"):
        load_pack_from(tmp_path, pack_data(), style)


def test_error_names_file_and_field(tmp_path):
    data = pack_data()
    data["pieces"]["Bench"]["holds"][0]["point"] = [0.2, 0.45]
    with pytest.raises(ManifestError) as err:
        load_pack_from(tmp_path, data)
    assert str(err.value).startswith(f"{tmp_path / 'test.pack.json'}: pieces.Bench.holds[0].point:")


# --- clip targets and holds ----------------------------------------------------------------


def test_clip_target_must_be_ref(tmp_path):
    data = clips_data()
    target = data["clips"]["sit_reach"]["targets"][0]
    del target["ref"]
    target["point"] = [1.2, 0.46, 1.8]
    with fails("never a bare point"):
        load_clips_from(tmp_path, data)


def test_rig_target_needs_why(tmp_path):
    data = clips_data()
    del data["clips"]["sit_reach"]["targets"][1]["why"]
    with fails("missing field 'why'"):
        load_clips_from(tmp_path, data)


def test_offset_target_needs_why(tmp_path):
    data = clips_data()
    del data["clips"]["sit_reach"]["targets"][0]["why"]
    with fails("missing field 'why'"):
        load_clips_from(tmp_path, data)


def test_keyed_clip_needs_no_targets(tmp_path):
    assert shop_clips()["sit_idle"].targets == ()
    data = clips_data()
    data["clips"]["sit_idle"]["targets"] = copy.deepcopy(data["clips"]["sit_reach"]["targets"][:1])
    with fails("a keyed clip has no key targets"):
        load_clips_from(tmp_path, data)
    data = clips_data()
    data["clips"]["sit_reach"]["targets"] = []
    with fails("a solved clip needs key targets"):
        load_clips_from(tmp_path, data)


def test_dangle_needs_why(tmp_path):
    data = clips_data()
    del data["clips"]["sit_reach"]["holds"][2]["dangle"]["why"]
    with fails("missing field 'why'"):
        load_clips_from(tmp_path, data)
    data["clips"]["sit_reach"]["holds"][2]["dangle"]["why"] = " "
    with fails("must be a non-empty string"):
        load_clips_from(tmp_path, data)


def test_cross_needs_why(tmp_path):
    data = clips_data()
    data["clips"]["sit_reach"]["holds"][1]["cross"] = {}
    with fails("missing field 'why'"):
        load_clips_from(tmp_path, data)


# --- strictness ----------------------------------------------------------------------------


def _at(data, *path):
    for key in path:
        data = data[key]
    return data


PACK_UNKNOWN = [
    (),
    ("pieces", "Bench"),
    ("pieces", "Bench", "holds", 0),
    ("pieces", "Bench", "parts", 0),
    ("pieces", "Bench", "measured"),
    ("pieces", "Doll", "rig"),
    ("pieces", "Doll", "rig", "chains", "arm.R"),
]
CLIPS_UNKNOWN = [
    (),
    ("clips", "sit_reach"),
    ("clips", "sit_reach", "holds", 0),
    ("clips", "sit_reach", "targets", 0),
    ("clips", "sit_reach", "floor_gear", "Bench_L"),
    ("clips", "sit_reach", "props", "lamp"),
]


@pytest.mark.parametrize("where", PACK_UNKNOWN)
def test_unknown_field_raises_in_pack(tmp_path, where):
    data = pack_data()
    _at(data, *where)["colour"] = "red"
    with fails("unknown field 'colour'"):
        load_pack_from(tmp_path, data)


@pytest.mark.parametrize("where", CLIPS_UNKNOWN)
def test_unknown_field_raises_in_clips(tmp_path, where):
    data = clips_data()
    _at(data, *where)["colour"] = "red"
    with fails("unknown field 'colour'"):
        load_clips_from(tmp_path, data)


def test_old_seat_with_clip_raises(tmp_path):
    data = pack_data()
    data["pieces"]["Bench"]["seats"][0]["clip"] = "sit"
    with fails("no longer names a clip"):
        load_pack_from(tmp_path, data)
    data = pack_data()
    data["pieces"]["Bench"]["measured"]["seats"] = [{"clip": "sit", "x": 0, "y": 0.45, "z": 0}]
    with fails("unknown field 'seats'"):
        load_pack_from(tmp_path, data)


def test_measured_accepts_tip_length(tmp_path):
    data = pack_data()
    data["pieces"]["Lamp"]["measured"]["tip_length"] = 0.14
    assert load_pack_from(tmp_path, data).pieces["Lamp"].measured["tip_length"] == 0.14
    data["pieces"]["Lamp"]["measured"]["tipLength"] = 0.14
    with fails("unknown field 'tipLength'"):
        load_pack_from(tmp_path, data)


@pytest.mark.parametrize("export", [{"position_bits": None}, {"position_bits": 14}])
def test_measured_export_round_trips(tmp_path, export):
    """The shipped position precision survives load and write; null (float32) stays null."""
    from model_kit.core.writer import write_pack
    data = pack_data()
    data["pieces"]["Lamp"]["measured"]["export"] = export
    pack = load_pack_from(tmp_path, data)
    assert pack.pieces["Lamp"].measured["export"] == export
    again = load_pack(write_pack(pack, tmp_path / "again.pack.json"), shop_style())
    assert again.pieces["Lamp"].measured["export"] == export


def test_measured_export_absent_stays_absent(tmp_path):
    """A pack from before measured exports reads as unknown: the loader never invents an entry."""
    from model_kit.core.writer import write_pack
    pack = load_pack_from(tmp_path, pack_data())
    assert "export" not in pack.pieces["Lamp"].measured
    again = load_pack(write_pack(pack, tmp_path / "again.pack.json"), shop_style())
    assert all("export" not in p.measured for p in again.pieces.values())


def test_outline_modifiers_default_empty(tmp_path):
    assert shop_pack().pieces["Doll"].rig.outline_modifiers == ()
    data = pack_data()
    data["pieces"]["Doll"]["rig"]["outline_modifiers"] = ["Outline", "Shell"]
    rig = load_pack_from(tmp_path, data).pieces["Doll"].rig
    assert rig.outline_modifiers == ("Outline", "Shell")
    data["pieces"]["Doll"]["rig"]["outline_modifiers"] = []
    assert load_pack_from(tmp_path, data).pieces["Doll"].rig.outline_modifiers == ()


def test_contact_group_entry_not_a_bone_raises(tmp_path):
    data = pack_data()
    data["pieces"]["Doll"]["rig"]["contact_groups"] = {"hand.R": ["hand.R", "nope"]}
    with fails("names no bone: 'nope'"):
        load_pack_from(tmp_path, data)


def test_every_spec_shape_loads(tmp_path):
    assert SHAPES == {"box", "lathe", "extrude", "blob", "ring", "meta", "cable", "tube", "leaf",
                      "decal"}
    data = pack_data()
    for shape in SHAPES:
        part = data["pieces"]["Bench"]["parts"][0]
        part["shape"] = shape
        # a decal is printed on a host: it needs an `on` and a printed role (`decal_problem`)
        part.update({"on": "seat_top", "role": "eyes"} if shape == "decal"
                    else {"on": None, "role": "solid"})
        assert load_pack_from(tmp_path, data).pieces["Bench"].parts[0].shape == shape


def _pack_mutations():
    def bench(d):
        return d["pieces"]["Bench"]

    def rig(d):
        return d["pieces"]["Doll"]["rig"]

    def chain(d):
        return rig(d)["chains"]["arm.R"]

    def limit(key):
        return lambda d: chain(d)["limits"].update({key: [-1, 1]})

    return [
        ("unknown shape", lambda d: bench(d)["parts"][0].update(shape="sphere"), "got 'sphere'"),
        ("surface names no part", lambda d: bench(d)["holds"][0].update(surface=["lid"]),
         "names no part: 'lid'"),
        ("hold names no use", lambda d: bench(d)["holds"][0].update(use="back"),
         "names no use station: 'back'"),
        ("part names no listed row", lambda d: bench(d)["parts"][0].update(listed="x"),
         "names no listed row: 'x'"),
        ("part on no part", lambda d: bench(d)["parts"][2].update(on="table"),
         "names no part: 'table'"),
        ("duplicate id", lambda d: bench(d)["seats"][0].update(id="edge_R"), "repeat: 'edge_R'"),
        ("bad side", lambda d: bench(d)["holds"][0].update(side="left"), "got 'left'"),
        ("zero off", lambda d: bench(d)["holds"][0].update(off=[0, 0, 0]), "zero vector"),
        ("bool point", lambda d: bench(d)["seats"][0].update(point=[0, True, 0]), "got True"),
        ("inverted box", lambda d: bench(d)["holds"][0].update(sub_box=[[1, 1, 1], [0, 0, 0]]),
         "is above max"),
        ("bad size class", lambda d: bench(d).update(size_class="huge"), "got 'huge'"),
        ("bad recipe", lambda d: bench(d)["recipe"].update(kind="carved"), "got 'carved'"),
        ("frame not gltf", lambda d: bench(d)["frame"].update(axes="blender"), "got 'blender'"),
        ("empty done", lambda d: bench(d).update(done=""), "must be a non-empty string"),
        ("missing measured", lambda d: bench(d).pop("measured"), "missing field 'measured'"),
        ("pack named rig", lambda d: d.update(pack="rig"), "reserved for bone refs"),
        ("bone parent unknown", lambda d: rig(d)["bones"][2].update(parent="neck"),
         "names no bone listed before it: 'neck'"),
        ("part not bound", lambda d: rig(d)["bind"].pop("foot_R"),
         "'foot_R' of a rigged piece has no bind"),
        ("rigid with two bones",
         lambda d: rig(d)["bind"]["torso"].update(bones=["root", "body"]), "exactly one bone"),
        ("dofs miss a bone", lambda d: chain(d)["dofs"].pop("hand.R"),
         "must name each chain bone once"),
        ("bad dof", lambda d: chain(d)["dofs"].update({"forearm.R": "xw"}), "got 'xw'"),
        ("limit on a locked axis", limit("forearm.R:y"), "one axis the bone turns on"),
        ("limit with no axis", limit("forearm.R:"), "one axis the bone turns on"),
        ("limit on two axes", limit("arm.R:xy"), "one axis the bone turns on"),
        ("limit low over high",
         lambda d: chain(d)["limits"].update({"forearm.R:x": [0.05, -3.0]}), "must be under high"),
        ("effector not last", lambda d: chain(d).update(effector="forearm.R"),
         "must be the chain's last bone"),
        ("bones reversed", lambda d: chain(d).update(bones=["hand.R", "forearm.R", "arm.R"]),
         "its parent is 'arm.R', not 'hand.R'"),
        ("bones skip a link", lambda d: chain(d).update(bones=["arm.R", "hand.R"]),
         "parent is 'forearm.R', not 'arm.R'"),
        ("root not an ancestor", lambda d: chain(d).update(root="foot.R"),
         "'foot.R' is not an ancestor of 'arm.R'"),
        ("root inside the chain", lambda d: chain(d).update(root="arm.R"),
         "'arm.R' is not an ancestor of 'arm.R'"),
        ("effector group unknown", lambda d: chain(d).update(effector_group="hand.L"),
         "names no contact group: 'hand.L'"),
        ("zero effector radius", lambda d: chain(d).update(effector_radius=0),
         "must be above 0"),
        ("mirror pair names no bone",
         lambda d: rig(d)["mirror"].update(pairs=[["arm.R", "wing.L"]]), "names no bone: 'wing.L'"),
        ("bad mirror axis", lambda d: rig(d)["mirror"].update(axis="w"), "got 'w'"),
        ("outline modifiers repeat",
         lambda d: rig(d).update(outline_modifiers=["Outline", "Outline"]),
         "modifier names repeat: 'Outline'"),
        ("outline modifiers not a list", lambda d: rig(d).update(outline_modifiers="Outline"),
         "must be a list"),
        ("outline modifier not a name", lambda d: rig(d).update(outline_modifiers=[""]),
         "must be a name"),
    ]


@pytest.mark.parametrize("name,mutate,match", _pack_mutations(), ids=lambda v: v
                         if isinstance(v, str) else "")
def test_bad_pack_raises(tmp_path, name, mutate, match):
    data = pack_data()
    mutate(data)
    with fails(match):
        load_pack_from(tmp_path, data)


def _clip_mutations():
    def reach(d):
        return d["clips"]["sit_reach"]

    return [
        ("when outside frames", lambda d: reach(d)["holds"][0].update(when=[[40, 60]]),
         "outside the clip's frames"),
        ("empty when", lambda d: reach(d)["holds"][0].update(when=[]), "must not be empty"),
        ("window backwards", lambda d: reach(d)["holds"][0].update(when=[[20, 10]]),
         "runs backwards"),
        ("key outside frames", lambda d: reach(d)["targets"][0].update(key=60),
         r"must be a whole frame inside \[1, 48\], got 60"),
        ("tool names no prop", lambda d: reach(d)["targets"][2].update(tool="Hammer"),
         "names no prop of this clip: 'Hammer'"),
        ("seat is a bone", lambda d: reach(d).update(seat="rig:body"), "not a bone ref"),
        ("hold ref is a bone", lambda d: reach(d)["holds"][0].update(ref="rig:hand.R"),
         "not a bone ref"),
        ("bad authoring", lambda d: reach(d).update(authoring="mocap"), "got 'mocap'"),
        ("frames backwards", lambda d: reach(d).update(frames=[48, 1]), "must be before end"),
        ("zero budget", lambda d: reach(d)["holds"][0].update(budget=0), "must be above 0"),
        ("zero quaternion",
         lambda d: reach(d)["floor_gear"]["Bench_L"].update(quaternion=[0, 0, 0, 0]),
         "zero vector"),
        ("mirror of nothing", lambda d: d["clips"]["sit_idle_L"].update(mirror_of="wave"),
         "names no clip: 'wave'"),
        ("mirror of itself",
         lambda d: d["clips"]["sit_idle_L"].update(mirror_of="sit_idle_L"), "cannot mirror itself"),
        ("bad ref", lambda d: reach(d)["targets"][0].update(ref="Bench.edge_R"),
         "is not <pack>:<piece>.<id>"),
        ("placement id is a carried piece's name, same piece",
         lambda d: reach(d)["floor_gear"].update(Lamp={
             "piece": "Lamp", "pack": "shop", "position": [5, 0, 5], "quaternion": [0, 0, 0, 1]}),
         (r"clip 'sit_reach': name 'Lamp' is claimed twice in the clip's ref namespace: "
          r"floor_gear 'Lamp' \(shop:Lamp\), prop 'lamp' \(shop:Lamp\)")),
        ("placement id is a carried piece's name, other piece",
         lambda d: reach(d)["floor_gear"].update(Lamp={
             "piece": "Bench", "pack": "shop", "position": [5, 0, 5], "quaternion": [0, 0, 0, 1]}),
         (r"clip 'sit_reach': name 'Lamp' is claimed twice in the clip's ref namespace: "
          r"floor_gear 'Lamp' \(shop:Bench\), prop 'lamp' \(shop:Lamp\)")),
        ("two props carry one piece",
         lambda d: reach(d)["props"].update(lamp_2=copy.deepcopy(reach(d)["props"]["lamp"])),
         (r"clip 'sit_reach': name 'Lamp' is claimed twice in the clip's ref namespace: "
          r"prop 'lamp' \(shop:Lamp\), prop 'lamp_2' \(shop:Lamp\)")),
    ]


@pytest.mark.parametrize("name,mutate,match", _clip_mutations(), ids=lambda v: v
                         if isinstance(v, str) else "")
def test_bad_clips_raise(tmp_path, name, mutate, match):
    data = clips_data()
    mutate(data)
    with fails(match):
        load_clips_from(tmp_path, data)


# --- seat_when ------------------------------------------------------------------------------


def test_seat_when_defaults_to_whole_clip(tmp_path):
    clip = {c.name: c for c in load_clips_from(tmp_path, clips_data())}["sit_idle"]
    assert clip.seat is not None and clip.seat_when is None
    assert seat_windows(clip) == (clip.frames,)
    ten = dataclasses.replace(clip, frames=(1, 10))
    assert seat_windows(ten) == ((1, 10),)


def test_seat_when_loads_as_windows(tmp_path):
    data = clips_data()
    data["clips"]["sit_idle"]["seat_when"] = [[2, 5], [10.5, 12]]
    clip = {c.name: c for c in load_clips_from(tmp_path, data)}["sit_idle"]
    assert clip.seat_when == ((2, 5), (10.5, 12))
    assert seat_windows(clip) == ((2, 5), (10.5, 12))


@pytest.mark.parametrize("name,mutate,match", [
    ("no seat", lambda c: (c.pop("seat"), c.update(seat_when=[[2, 5]])),
     "needs a seat"),
    ("empty", lambda c: c.update(seat_when=[]), "must not be empty"),
    ("outside frames", lambda c: c.update(seat_when=[[20, 40]]), "outside the clip's frames"),
    ("backwards", lambda c: c.update(seat_when=[[10, 5]]), "runs backwards"),
], ids=lambda v: v if isinstance(v, str) else "")
def test_bad_seat_when_raises(tmp_path, name, mutate, match):
    data = clips_data()
    mutate(data["clips"]["sit_idle"])
    with pytest.raises(ManifestError, match="seat_when") as err:
        load_clips_from(tmp_path, data)
    assert re.search(match, str(err.value)), str(err.value)


def test_placement_type_is_frozen():
    placement = Placement("Bench_L", "Bench", "shop", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    with pytest.raises(AttributeError):
        placement.piece = "Lamp"
    assert math.isclose(placement.quaternion[3], 1.0)


def test_placement_made_in_code_rejects_zero_quaternion():
    for quaternion in [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, math.nan, 1.0)]:
        with fails("quaternion"):
            Placement("Bench_L", "Bench", "shop", (0.0, 0.0, 0.0), quaternion)


# --- duplicate keys and wrong JSON types ---------------------------------------------------


@pytest.mark.parametrize("name,text", [
    ("pack", '{"pack": "shop", "pack": "shop", "units_per_metre": 0.5, "pieces": {}}'),
    ("clips", '{"clips": {"a": {}, "a": {}}}'),
])
def test_repeated_json_key_raises(tmp_path, name, text):
    path = tmp_path / f"test.{name}.json"
    path.write_text(text)
    with fails("repeats"):
        load_pack(path, shop_style()) if name == "pack" else load_clips(path)


def _paths(node, at=()):
    """Every place in a JSON tree: object keys and list indexes, outermost first."""
    items = node.items() if isinstance(node, dict) else enumerate(node) \
        if isinstance(node, list) else ()
    for key, child in items:
        yield (*at, key)
        yield from _paths(child, (*at, key))


WRONG_TYPES = [None, True, 3, "x", [], {}, [["x"]], {"x": 1}]


@pytest.mark.parametrize("which", ["pack", "clips"])
def test_wrong_json_types_raise_manifest_error_only(tmp_path, which):
    """Put each wrong type at each place: a load passes or raises ManifestError, never more."""
    base = pack_data() if which == "pack" else clips_data()
    others = []
    for where in _paths(base):
        for wrong in WRONG_TYPES:
            data = copy.deepcopy(base)
            _at(data, *where[:-1])[where[-1]] = copy.deepcopy(wrong)
            try:
                if which == "pack":
                    load_pack_from(tmp_path, data)
                else:
                    load_clips_from(tmp_path, data)
            except ManifestError:
                pass
            except Exception as exc:  # noqa: BLE001 - the test lists every escape
                others.append(f"{'.'.join(map(str, where))} = {wrong!r}: {exc!r}")
    assert others == []


def test_placement_matrix_is_the_world_from_piece_map():
    turn = Placement("A", "Plate", "kit", (1.0, 0.5, -2.0), (0.0, 0.3826834, 0.0, 0.9238795))
    m = turn.matrix()
    assert m.shape == (4, 4) and np.array_equal(m[3], [0.0, 0.0, 0.0, 1.0])
    pts = np.array([[0.1, 0.2, 0.3], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    np.testing.assert_allclose(pts @ m[:3, :3].T + m[:3, 3], turn.to_world(pts), atol=1e-12)
    np.testing.assert_allclose(turn.to_piece(turn.to_world(pts)), pts, atol=1e-12)


def test_public_names_survive_the_split() -> None:
    from model_kit.core import manifest

    names = {
        "Vec3", "Quat", "Box", "Window",
        "SHAPES", "MEASURED_KEYS", "RECIPES", "SIZE_CLASSES", "POSTURES", "SIDES",
        "BIND_MODES", "AUTHORING", "AXES", "FRAME_AXES", "RIG_PACK",
        "Ref", "Frame", "Recipe", "Listed", "Part", "Use", "Hold", "Seat", "Mount", "Bone",
        "Bind", "SitSpec", "MirrorSpec", "Chain", "RigSpec", "Piece", "Pack", "Dangle", "Cross",
        "ClipHold", "Target", "Placement", "FloorGear", "Prop", "Clip", "Anchor",
        "load_pack", "load_clips", "anchor",
    }
    assert len(names) == 44
    assert set(manifest.__all__) >= names
    for name in names:
        assert hasattr(manifest, name), name


# --- mount kinds, part roles, spacing ------------------------------------------------------


def _edit_piece(data: dict, piece: str, **changes) -> dict:
    data["pieces"][piece].update(changes)
    return data


def test_mount_kinds_are_the_three(tmp_path):
    assert MOUNT_KINDS == ("rest", "power", "signal")
    for kind in MOUNT_KINDS:
        data = pack_data()
        data["pieces"]["Lamp"]["mounts"][0]["kind"] = kind
        assert load_pack_from(tmp_path, data).pieces["Lamp"].mounts[0].kind == kind


def test_mount_part_and_end_plug_are_optional_and_read(tmp_path):
    bare = shop_pack().pieces["Lamp"]
    assert bare.mounts[0].part is None
    assert [e.plug for e in bare.connectors[0].ends] == [None, None]
    data = pack_data()
    lamp = data["pieces"]["Lamp"]
    lamp["mounts"][0]["part"] = "base"
    lamp["connectors"][0]["ends"][1]["plug"] = "base"
    named = load_pack_from(tmp_path, data).pieces["Lamp"]
    assert named.mounts[0].part == "base"
    assert [e.plug for e in named.connectors[0].ends] == [None, "base"]


@pytest.mark.parametrize("edit, match", [
    (lambda lamp: lamp["mounts"][0].update(part="ghost"), "names no part of this piece"),
    (lambda lamp: lamp["mounts"][0].update(part=3), "name"),
    (lambda lamp: lamp["connectors"][0]["ends"][1].update(plug="ghost"),
     "names no part of this piece"),
    (lambda lamp: lamp["connectors"][0]["ends"][0].update(plug="lead"), "names no plug"),
    (lambda lamp: lamp["connectors"][0]["ends"][1].update(plug="lead"), "must name a solid part"),
])
def test_mount_part_and_end_plug_refusals(tmp_path, edit, match):
    data = pack_data()
    edit(data["pieces"]["Lamp"])
    with fails(match):
        load_pack_from(tmp_path, data)


@pytest.mark.parametrize("kind", ["furniture", "", "Power", 3, None])
def test_mount_kind_outside_the_list_raises(tmp_path, kind):
    data = pack_data()
    data["pieces"]["Lamp"]["mounts"][0]["kind"] = kind
    with fails("must be one of"):
        load_pack_from(tmp_path, data)


def test_part_roles_are_the_four(tmp_path):
    assert PART_ROLES == ("solid", "eyes", "decal", "skip_outline")
    for role in PART_ROLES:
        data = pack_data()
        data["pieces"]["Bench"]["parts"][0]["role"] = role
        assert load_pack_from(tmp_path, data).pieces["Bench"].parts[0].role == role


def test_eyes_draw_no_outline():
    assert set(NO_OUTLINE_ROLES) == {"eyes", "skip_outline"} <= set(PART_ROLES)


def _decal_data(**changes):
    data = pack_data()
    part = data["pieces"]["Bench"]["parts"][0]
    part.update({"shape": "decal", "role": "eyes", "on": "seat_top"} | changes)
    return data


def test_a_decal_with_a_host_and_a_printed_role_loads(tmp_path):
    for role in ("eyes", "decal"):
        part = load_pack_from(tmp_path, _decal_data(role=role)).pieces["Bench"].parts[0]
        assert part.printed and not part.solid and part.skips_outline and part.open_surface


def test_a_decal_with_no_on_is_refused_and_names_the_part(tmp_path):
    with fails(r"'leg_L' is a decal but has no `on`"):
        load_pack_from(tmp_path, _decal_data(on=None))


@pytest.mark.parametrize("role", ["solid", "skip_outline"])
def test_a_decal_with_a_thickness_role_is_refused_and_names_the_part(tmp_path, role):
    with fails(r"'leg_L' is a decal but has the role .* solid"):
        load_pack_from(tmp_path, _decal_data(role=role))


@pytest.mark.parametrize("changes", [
    {"shape": "box", "role": "decal"},   # a printed role on a box: markings' finding
    {"shape": "box", "role": "eyes"},
])
def test_other_role_and_shape_disagreements_still_load(tmp_path, changes):
    data = _decal_data(**changes)
    assert load_pack_from(tmp_path, data).pieces["Bench"].parts[0].printed


def test_cable_bends_and_box_does_not(tmp_path):
    data = pack_data()
    parts = data["pieces"]["Bench"]["parts"]
    parts[0]["shape"] = "cable"
    loaded = load_pack_from(tmp_path, data).pieces["Bench"].parts
    assert loaded[0].bends and not loaded[1].bends


@pytest.mark.parametrize("role", ["Fill", "Ink", "", None])
def test_part_role_outside_the_list_raises(tmp_path, role):
    data = pack_data()
    data["pieces"]["Bench"]["parts"][0]["role"] = role
    with fails("must be one of"):
        load_pack_from(tmp_path, data)


def test_pack_spacing_reads_and_refuses(tmp_path):
    assert shop_pack().spacing == 2.0
    data = pack_data()
    del data["spacing"]
    assert load_pack_from(tmp_path, data).spacing is None
    data["spacing"] = None
    assert load_pack_from(tmp_path, data).spacing is None
    for bad in (0, -1.0, "2", True):
        data["spacing"] = bad
        with fails("above 0|finite number"):
            load_pack_from(tmp_path, data)


def test_measured_keys_take_pivots(tmp_path):
    data = pack_data()
    data["pieces"]["Doll"]["measured"]["pivots"] = [{"name": "wrist", "x": 0.0, "y": 0.5, "z": 0.1}]
    pack = load_pack_from(tmp_path, data)
    assert pack.pieces["Doll"].measured["pivots"][0]["name"] == "wrist"


# --- connectors, keep-offs, avoid shapes ---------------------------------------------------


def test_connector_reads():
    (conn,) = shop_pack().pieces["Lamp"].connectors
    assert conn.id == "lead_run" and conn.part == "lead" and conn.cap == 1.5
    near, far = conn.ends
    assert near == End((0.0, 0.01, -0.1), None, Ref("self", None, "plug"))
    assert near.host.is_self and str(near.host) == "self.plug"
    assert far == End((0.25, 0.46, -0.1), "shop", Ref("shop", "Bench_1", "socket"))
    assert not far.host.is_self
    assert isinstance(conn, Connector)


def _lamp_connector(data: dict) -> dict:
    return data["pieces"]["Lamp"]["connectors"][0]


@pytest.mark.parametrize("edit, match", [
    (lambda c: c.update(part="stem"), "names no part|cable"),
    (lambda c: c.update(part="ghost"), "names no part"),
    (lambda c: c.update(ends=c["ends"][:1]), "two ends"),
    (lambda c: c.update(cap=0), "above 0"),
    (lambda c: c.pop("why"), "missing field 'why'"),
    (lambda c: c.update(extra=1), "unknown field"),
    (lambda c: c["ends"][0].update(host="self.ghost"), "names no mount of this piece"),
    (lambda c: c["ends"][0].update(layout="shop"), "names no layout"),
    (lambda c: c["ends"][1].pop("layout"), "missing field 'layout'"),
    (lambda c: c["ends"][1].update(host="rig:body"), "not a bone ref|ref"),
])
def test_connector_refusals(tmp_path, edit, match):
    data = pack_data()
    edit(_lamp_connector(data))
    with fails(match):
        load_pack_from(tmp_path, data)


def test_connector_part_must_be_a_cable(tmp_path):
    data = pack_data()
    lamp = data["pieces"]["Lamp"]
    lamp["connectors"][0]["part"] = "stem"  # a tube
    with fails("cable"):
        load_pack_from(tmp_path, data)


def test_connector_ids_repeat_raises(tmp_path):
    data = pack_data()
    lamp = data["pieces"]["Lamp"]
    lamp["connectors"].append(copy.deepcopy(lamp["connectors"][0]))
    with fails("repeat"):
        load_pack_from(tmp_path, data)


def test_keep_off_reads_and_refuses(tmp_path):
    (keep,) = shop_pack().pieces["Lamp"].keep_off
    assert keep == KeepOff("bulb", (0.0, 1.1, 0.0), 0.08, keep.why)
    for edit, match in [(lambda k: k.update(radius=0), "above 0"),
                        (lambda k: k.update(point=[0, 1]), "list of 3"),
                        (lambda k: k.pop("why"), "missing field 'why'")]:
        data = pack_data()
        edit(data["pieces"]["Lamp"]["keep_off"][0])
        with fails(match):
            load_pack_from(tmp_path, data)


def _view(**changes) -> dict:
    return {"nearest_m": 1.0, "why": "a test piece seen at arm's length", **changes}


def test_view_round_trip(tmp_path):
    """A piece without `view` loads with None; one with it reads every field, and the writer's
    file reads back equal."""
    assert shop_pack().pieces["Lamp"].view is None
    data = pack_data()
    data["pieces"]["Lamp"]["view"] = _view(hidden=["back", "bottom"],
                                           hidden_why="the lamp stands against a wall")
    pack = load_pack_from(tmp_path, data)
    assert pack.pieces["Lamp"].view == View(1.0, "a test piece seen at arm's length",
                                            ("back", "bottom"), "the lamp stands against a wall")
    plain = {**data["pieces"]["Lamp"], "view": _view()}
    data["pieces"]["Lamp"] = plain
    assert load_pack_from(tmp_path, data).pieces["Lamp"].view == View(
        1.0, "a test piece seen at arm's length")


def test_outlined_defaults_to_true_and_reads_back(tmp_path):
    """A piece that says nothing is outlined; `outlined: false` with a reason reads both fields,
    and the reason is kept."""
    assert all(p.outlined and p.outlined_why == "" for p in shop_pack().pieces.values())
    data = pack_data()
    data["pieces"]["Lamp"] |= {"outlined": False, "outlined_why": "the page draws it flat"}
    pack = load_pack_from(tmp_path, data)
    assert (pack.pieces["Lamp"].outlined, pack.pieces["Lamp"].outlined_why) == (
        False, "the page draws it flat")
    assert pack.pieces["Doll"].outlined is True  # the other pieces are untouched
    data["pieces"]["Lamp"] = {**data["pieces"]["Lamp"], "outlined": True}
    del data["pieces"]["Lamp"]["outlined_why"]
    assert load_pack_from(tmp_path, data).pieces["Lamp"].outlined is True


def test_outlined_refuses_a_missing_reason_a_stale_reason_and_a_non_bool(tmp_path):
    for edit, match in [({"outlined": False}, "needs an outlined_why"),
                        ({"outlined": False, "outlined_why": ""}, "needs an outlined_why"),
                        ({"outlined": False, "outlined_why": "  "}, "needs an outlined_why"),
                        ({"outlined": False, "outlined_why": 3}, "needs an outlined_why"),
                        ({"outlined": True, "outlined_why": "stale"}, "names no unlined look"),
                        ({"outlined_why": "stale"}, "names no unlined look"),
                        ({"outlined": "no", "outlined_why": "x"}, "true or false"),
                        ({"outlined": 0, "outlined_why": "x"}, "true or false"),
                        ({"outlined": None}, "true or false")]:
        data = pack_data()
        data["pieces"]["Lamp"] |= edit
        with fails(match):
            load_pack_from(tmp_path, data)


def _card_data() -> dict:
    """The shop pack with its Lamp branching and a card copy `Lamp_Cards` of it."""
    data = pack_data()
    data["pieces"]["Lamp"]["recipe"]["kind"] = "branching"
    data["pieces"]["Lamp_Cards"] = {
        **copy.deepcopy(data["pieces"]["Lamp"]), "cards_of": "Lamp", "outlined": False,
        "outlined_why": "the shipping look draws cards flat"}
    return data


def test_cards_of_round_trip(tmp_path):
    from model_kit.core.writer import write_pack

    pack = load_pack_from(tmp_path, _card_data())
    assert pack.pieces["Lamp_Cards"].cards_of == "Lamp"
    assert pack.pieces["Lamp"].cards_of == ""
    path = write_pack(pack, tmp_path / "out.pack.json")
    raw = json.loads(path.read_text())["pieces"]
    assert raw["Lamp_Cards"]["cards_of"] == "Lamp" and "cards_of" not in raw["Lamp"]
    assert load_pack(path, shop_style()) == pack


def test_cards_of_refusals(tmp_path):
    def card(edit_source=None, **edit):
        data = _card_data()
        data["pieces"]["Lamp_Cards"] |= edit
        if edit_source:
            data["pieces"]["Lamp"] |= edit_source
        return data

    cases = [
        (card(cards_of="Ghost"), r"Lamp_Cards.*Ghost.*lacks"),
        (card(cards_of=""), "non-empty"),
        (card(cards_of="Doll"), r"Lamp_Cards.*Doll.*(assembly|rigged|\w+) piece"),
        (card(outlined=True, outlined_why=None), r"Lamp_Cards.*Lamp.*outlined: false"),
        (card({"cards_of": "Lamp_Cards"}), r"Lamp.*Lamp_Cards.*itself a card copy"),
    ]
    for data, match in cases:
        if data["pieces"]["Lamp_Cards"].get("outlined_why", 1) is None:
            del data["pieces"]["Lamp_Cards"]["outlined_why"]
        if data["pieces"]["Lamp"].get("cards_of"):
            data["pieces"]["Lamp"] |= {"outlined": False, "outlined_why": "x"}
        with fails(match):
            load_pack_from(tmp_path, data)
    data = _card_data()
    data["pieces"]["Lamp_Cards"]["recipe"]["kind"] = "assembly"
    with fails(r"Lamp_Cards.*Lamp.*a card piece is branching"):
        load_pack_from(tmp_path, data)


def test_view_farthest_round_trips_and_defaults_to_none(tmp_path):
    data = pack_data()
    data["pieces"]["Lamp"]["view"] = _view()
    assert load_pack_from(tmp_path, data).pieces["Lamp"].view.farthest_m is None
    data["pieces"]["Lamp"]["view"] = _view(farthest_m=6.0, farthest_why="seen across the room")
    view = load_pack_from(tmp_path, data).pieces["Lamp"].view
    assert (view.farthest_m, view.farthest_why) == (6.0, "seen across the room")
    # nearest and farthest may be equal: a piece seen from one distance only
    data["pieces"]["Lamp"]["view"] = _view(farthest_m=1.0, farthest_why="only ever at arm's length")
    assert load_pack_from(tmp_path, data).pieces["Lamp"].view.farthest_m == 1.0


def test_farthest_under_nearest_raises(tmp_path):
    for edit, match in [({"farthest_m": 0.5, "farthest_why": "x"}, "at least nearest_m"),
                        ({"farthest_m": 0, "farthest_why": "x"}, "above 0"),
                        ({"farthest_m": True, "farthest_why": "x"}, "finite number"),
                        ({"farthest_m": 5.0}, "needs a farthest_why"),
                        ({"farthest_m": 5.0, "farthest_why": " "}, "needs a farthest_why"),
                        ({"farthest_why": "stale"}, "names no farthest_m")]:
        data = pack_data()
        data["pieces"]["Lamp"]["view"] = _view(**edit)
        with fails(match):
            load_pack_from(tmp_path, data)
    with pytest.raises(ManifestError, match=r"view: farthest_m: must be at least nearest_m"):
        View(nearest_m=1.0, why="x", farthest_m=0.5, farthest_why="y")


def test_a_view_outside_the_styles_camera_is_refused_at_load(tmp_path):
    camera = {"depth_projection": "perspective", "near_m": 0.5, "far_m": 50,
              "depth_bits": 24, "depth_why": "a test camera"}
    style = Style(units_per_metre=shop_style().units_per_metre, page_camera=camera,
                  dials=shop_style().dials)
    data = pack_data()
    data["pieces"]["Lamp"]["view"] = _view(nearest_m=0.5, farthest_m=50.0, farthest_why="x")
    assert load_pack_from(tmp_path, data, style).pieces["Lamp"].view.farthest_m == 50.0
    data["pieces"]["Lamp"]["view"] = _view(nearest_m=0.4)
    with fails("0.4 m is nearer than the style's camera near plane"):
        load_pack_from(tmp_path, data, style)
    data["pieces"]["Lamp"]["view"] = _view(farthest_m=50.5, farthest_why="x")
    with fails("50.5 m is past the style's camera far plane"):
        load_pack_from(tmp_path, data, style)
    # a nearest view past the far plane is out of the buffer, with or without a farthest view
    data["pieces"]["Lamp"]["view"] = _view(nearest_m=60.0)
    with fails("60.0 m is past the style's camera far plane"):
        load_pack_from(tmp_path, data, style)


def test_view_refuses_zero_distance(tmp_path):
    for edit, match in [({"nearest_m": 0}, "above 0"), ({"nearest_m": -1.5}, "above 0"),
                        ({"nearest_m": True}, "finite number"),
                        ({"nearest_m": "1 m"}, "finite number"),
                        ({"why": " "}, "non-empty string"),
                        ({"hidden": ["underside"], "hidden_why": "x"}, "must be one of"),
                        ({"hidden": ["back"]}, "needs a hidden_why"),
                        ({"hidden": ["back"], "hidden_why": " "}, "needs a hidden_why"),
                        ({"hidden_why": "stale"}, "names no hidden side"),
                        ({"hidden": ["back", "back"], "hidden_why": "x"}, "repeat")]:
        data = pack_data()
        data["pieces"]["Lamp"]["view"] = _view(**edit)
        with fails(match):
            load_pack_from(tmp_path, data)
    data = pack_data()
    data["pieces"]["Lamp"]["view"] = {"nearest_m": 1.0}
    with fails("missing field 'why'"):
        load_pack_from(tmp_path, data)
    data["pieces"]["Lamp"]["view"] = _view(extra=1)
    with fails("unknown field 'extra'"):
        load_pack_from(tmp_path, data)


def test_avoid_reads_and_refuses(tmp_path):
    chest, lap = shop_pack().pieces["Doll"].rig.avoid
    assert chest == Avoid("chest", "body", "ellipsoid", (0.0, 0.1, 0.0), (0.12, 0.15, 0.08),
                          chest.why)
    assert lap.kind == "box" and AVOID_KINDS == ("box", "ellipsoid")
    for edit, match in [(lambda a: a.update(kind="sphere"), "must be one of"),
                        (lambda a: a.update(bone="ghost"), "names no bone"),
                        (lambda a: a.update(half=[0.1, 0.0, 0.1]), "above 0"),
                        (lambda a: a.pop("why"), "missing field 'why'")]:
        data = pack_data()
        edit(data["pieces"]["Doll"]["rig"]["avoid"][0])
        with fails(match):
            load_pack_from(tmp_path, data)


# --- clip loops and frame ------------------------------------------------------------------


def _clip_data(**changes) -> dict:
    data = clips_data()
    data["clips"]["sit_idle"].update(changes)
    return data


def test_clip_loop_and_frame_read(tmp_path):
    clips = {c.name: c for c in load_clips_from(tmp_path, _clip_data(loop=True, loop_from=5))}
    idle = clips["sit_idle"]
    assert idle.loop is True and idle.loop_from == 5 and idle.frame == "gltf"
    plain = shop_clips()["sit_reach"]
    assert plain.loop is False and plain.loop_from is None and plain.frame == FRAME_AXES[0]


def test_loop_from_needs_loop(tmp_path):
    with fails("needs loop"):
        load_clips_from(tmp_path, _clip_data(loop_from=5))
    with fails("needs loop"):
        load_clips_from(tmp_path, _clip_data(loop=False, loop_from=5))


def test_loop_from_must_be_a_frame_inside_the_clip(tmp_path):
    start, end = shop_clips()["sit_idle"].frames
    for bad in (start - 1, end, end + 3, 2.5, True, "5"):
        with fails("whole frame"):
            load_clips_from(tmp_path, _clip_data(loop=True, loop_from=bad))
    load_clips_from(tmp_path, _clip_data(loop=True, loop_from=start))


def test_loop_must_be_a_bool_and_frame_a_known_axis_set(tmp_path):
    with fails("true or false"):
        load_clips_from(tmp_path, _clip_data(loop="yes"))
    with fails("must be one of"):
        load_clips_from(tmp_path, _clip_data(frame="blender"))


# --- the original arguments still build every record -------------------------------------------


def test_new_fields_default():
    shop = shop_pack()
    lamp, doll = shop.pieces["Lamp"], shop.pieces["Doll"]
    clip = shop_clips()["sit_idle"]
    old_clip = Clip(clip.name, clip.frames, clip.fps, clip.authoring, clip.seat, clip.holds,
                    clip.targets, clip.floor_gear, clip.props, clip.mirror_of)
    assert (old_clip.loop, old_clip.loop_from, old_clip.frame) == (False, None, "gltf")
    rig = doll.rig
    old_rig = RigSpec(rig.armature, rig.bones, rig.bind, rig.contact_groups, rig.chains,
                      rig.sit, rig.mirror)
    assert old_rig.avoid == () and old_rig.outline_modifiers == ()
    old_piece = Piece(lamp.name, lamp.frame, lamp.recipe, lamp.listed, lamp.parts, lamp.uses,
                      lamp.holds, lamp.seats, lamp.mounts, lamp.done, lamp.size_class,
                      lamp.measured, None)
    assert old_piece.connectors == () and old_piece.keep_off == () and old_piece.view is None
    old_pack = Pack("shop", 0.5, {"Lamp": old_piece})
    assert old_pack.spacing is None
    assert Mount("plug", "power", (0, 0, 0), (0, 0, 1), "lead", None).fits is None
    assert Part("a", "l", "box", None, "g", "solid", (), None).role == "solid"


# --- the rig contract for any body ---------------------------------------------------------


def _doll_rig(data):
    return data["pieces"]["Doll"]["rig"]


def _body(*rows, binds):
    """A rig of bones `(name, parent)` with the given part -> bones binds, for `bending_joints`."""
    from model_kit.core.manifest import Bind, Bone

    return RigSpec(
        armature="T", bones=tuple(Bone(n, p, 0.1) for n, p in rows),
        bind={part: Bind(tuple(bones), "rigid" if len(bones) == 1 else "chain")
              for part, bones in binds.items()},
        contact_groups={}, chains={})


def test_rig_without_mirror_or_sit_loads(tmp_path):
    data = pack_data()
    _doll_rig(data).pop("sit")
    _doll_rig(data).pop("mirror")
    rig = load_pack_from(tmp_path, data).pieces["Doll"].rig
    assert rig.sit is None and rig.mirror is None


def test_new_rig_fields_read(tmp_path):
    from model_kit.core.manifest import Joint

    rig = shop_pack().pieces["Doll"].rig
    assert rig.supports == ("foot.R",) and rig.root is None
    assert rig.joints["forearm.R"] == Joint({"x": (-175.0, 10.0)},
                                            "test fixture: schema contract, not anatomy")
    assert set(rig.joints) == {"arm.R", "forearm.R", "hand.R", "foot.R"}
    assert rig.sit.past_edge == ()
    data = pack_data()
    _doll_rig(data)["sit"]["past_edge"] = ["forearm.R"]
    _doll_rig(data)["root"] = "root"
    rig = load_pack_from(tmp_path, data).pieces["Doll"].rig
    assert rig.sit.past_edge == ("forearm.R",) and rig.root == "root"


def test_two_roots_need_root(tmp_path):
    from model_kit.core.rig import rig_root

    rig = shop_pack().pieces["Doll"].rig
    assert rig_root(rig) == "root"
    from model_kit.core.manifest import Bone

    two = dataclasses.replace(rig, bones=(*rig.bones, Bone("tail", None, 0.1)))
    with pytest.raises(ManifestError, match=r"parentless bones.*tail"):
        rig_root(two)
    assert rig_root(dataclasses.replace(two, root="tail")) == "tail"
    # The loader meets the same rule for every rig, with or without joints.
    data = pack_data()
    _doll_rig(data)["bones"].append({"name": "tail", "parent": None, "length": 0.1})
    with fails(r"parentless bones.*tail"):
        load_pack_from(tmp_path, data)
    _doll_rig(data).pop("joints")
    with fails(r"parentless bones.*tail"):
        load_pack_from(tmp_path, data)
    _doll_rig(data)["root"] = "root"
    load_pack_from(tmp_path, data)


def test_rig_root_reads_root_field():
    from model_kit.core.rig import rig_root

    rig = shop_pack().pieces["Doll"].rig
    assert rig_root(dataclasses.replace(rig, root="body")) == "body"


def test_root_and_past_edge_name_bones(tmp_path):
    data = pack_data()
    _doll_rig(data)["root"] = "wing"
    with fails("names no bone: 'wing'"):
        load_pack_from(tmp_path, data)
    data = pack_data()
    _doll_rig(data)["sit"]["past_edge"] = ["wing"]
    with fails("names no bone: 'wing'"):
        load_pack_from(tmp_path, data)


def test_supports_must_be_contact_groups(tmp_path):
    data = pack_data()
    _doll_rig(data)["supports"] = ["foot.L"]
    with fails("names no contact group: 'foot.L'"):
        load_pack_from(tmp_path, data)


def test_joint_on_non_bending_bone_raises(tmp_path):
    data = pack_data()
    why = "test fixture: schema contract, not anatomy"
    _doll_rig(data)["joints"]["arm.L"] = {"range_deg": {"x": [-1, 1]}, "why": why}
    with fails("'arm.L' is not a bending joint"):
        load_pack_from(tmp_path, data)
    data = pack_data()
    _doll_rig(data)["joints"]["root"] = {"range_deg": {"x": [-1, 1]}, "why": why}
    with fails("'root' is not a bending joint"):
        load_pack_from(tmp_path, data)


@pytest.mark.parametrize("axes,match", [
    ({"w": [-1, 1]}, "got 'w'"),
    ({"x": [1, -1]}, "must be under high"),
    ({"x": [0, 0]}, "must be under high"),
    ({}, "must not be empty"),
])
def test_bad_joint_range_raises(tmp_path, axes, match):
    data = pack_data()
    _doll_rig(data)["joints"]["foot.R"]["range_deg"] = axes
    with fails(match):
        load_pack_from(tmp_path, data)


def test_joint_needs_a_why(tmp_path):
    data = pack_data()
    _doll_rig(data)["joints"]["foot.R"].pop("why")
    with fails("missing field 'why'"):
        load_pack_from(tmp_path, data)


def test_dof_axis_without_range_raises(tmp_path):
    data = pack_data()
    del _doll_rig(data)["joints"]["arm.R"]["range_deg"]["z"]
    with fails(r"'arm\.R' turns on z in chain 'arm\.R'.*no range"):
        load_pack_from(tmp_path, data)


def test_a_bone_with_no_joint_entry_needs_no_range(tmp_path):
    data = pack_data()
    del _doll_rig(data)["joints"]["arm.R"]
    assert "arm.R" not in load_pack_from(tmp_path, data).pieces["Doll"].rig.joints


def test_chain_limit_outside_joint_range_raises(tmp_path):
    data = pack_data()
    _doll_rig(data)["joints"]["forearm.R"]["range_deg"]["x"] = [-90, 10]
    with fails(r"limit 'forearm\.R:x'.*joint 'forearm\.R' range"):
        load_pack_from(tmp_path, data)


def test_chain_limit_on_the_joint_edge_loads(tmp_path):
    data = pack_data()
    _doll_rig(data)["joints"]["forearm.R"]["range_deg"]["x"] = [math.degrees(-3.0),
                                                                math.degrees(0.05)]
    load_pack_from(tmp_path, data)


def test_bending_joints_count_hips_under_a_non_deforming_root():
    from model_kit.core.rig import bending_joints

    # The root binds nothing. The hips carry the legs, the waist carries the chest.
    rig = _body(("root", None), ("hips", "root"), ("leg", "hips"), ("waist", "root"),
                ("chest", "waist"), binds={"legs": ["leg"], "trunk": ["waist", "chest"]})
    assert bending_joints(rig) == ("hips", "leg", "waist", "chest")


def test_unbound_prop_bone_is_not_a_joint():
    from model_kit.core.rig import bending_joints

    rig = _body(("root", None), ("spine", "root"), ("arm", "spine"), ("prop", "arm"),
                binds={"torso": ["spine"], "limb": ["arm"]})
    assert bending_joints(rig) == ("arm",)
    assert "prop" not in bending_joints(rig)


def test_bending_joints_of_the_fixture_doll():
    from model_kit.core.rig import bending_joints

    assert bending_joints(shop_pack().pieces["Doll"].rig) == (
        "arm.R", "forearm.R", "hand.R", "foot.R")


# --- layouts -------------------------------------------------------------------------------


def layout_data() -> dict:
    return json.loads((FIX / "layouts" / "shop.json").read_text())


def shop_packs() -> dict[str, Pack]:
    return {"shop": shop_pack()}


def layouts_from(tmp_path: Path, *datas: dict, packs=None) -> dict[str, Layout]:
    folder = tmp_path / "layouts"
    folder.mkdir(exist_ok=True)
    for i, data in enumerate(datas):
        (folder / f"file{i}.json").write_text(json.dumps(data))
    return load_layouts(folder, packs or shop_packs())


def test_the_shop_layout_loads():
    layouts = load_layouts(FIX / "layouts", shop_packs())
    assert list(layouts) == ["shop"]
    layout = layouts["shop"]
    assert layout.frame == "gltf"
    assert list(layout.placements) == ["Bench_1", "Lamp_1", "Lamp_2"]
    lamp = layout.placements["Lamp_1"]
    assert lamp == LayoutPlacement("Lamp_1", "shop", "Lamp", (0.25, 0.46, -0.1),
                                   (0.0, 0.0, 0.0, 1.0), Ref("shop", "Bench_1", "socket"))
    assert layout.placements["Bench_1"].on is None
    free = layout.placements["Lamp_2"]  # unjoined: the second piece the clearance check measures
    assert free.on is None and free.rests_on is None
    assert layout.stands["bench"] == Stand("bench", (0.0, 0.0, 0.5), (0.0, 0.0, -1.0), "sit",
                                           "sit_reach")
    assert load_layout(FIX / "layouts" / "shop.json") == layout


def test_layout_on_missing_mount_raises(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_1"]["on"] = "shop:Bench_1.no_such_mount"
    with pytest.raises(ManifestError) as err:
        layouts_from(tmp_path, data)
    text = str(err.value)
    assert "file0.json" in text and "Lamp_1" in text
    assert "no_such_mount" in text and "shop:Bench" in text  # names the mount and its host


def test_layout_on_missing_host_placement_raises(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_1"]["on"] = "shop:Ghost_1.socket"
    with pytest.raises(ManifestError, match=r"Lamp_1.*Ghost_1"):
        layouts_from(tmp_path, data)


def test_layout_on_a_host_of_another_pack_than_the_ref_raises(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_1"]["on"] = "other:Bench_1.socket"
    with pytest.raises(ManifestError, match=r"Lamp_1.*other"):
        layouts_from(tmp_path, data)


def test_layout_guest_lacks_takes_part_raises(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_1"].update(piece="Bench")  # a bench has no `lead` part
    with pytest.raises(ManifestError, match=r"Lamp_1.*'lead'.*shop:Bench"):
        layouts_from(tmp_path, data)


@pytest.mark.parametrize("edit, match", [
    (lambda d: d["placements"]["Lamp_1"].update(pack="other"), "pack 'other'.*not loaded"),
    (lambda d: d["placements"]["Lamp_1"].update(piece="Ghost"), "no piece 'Ghost'"),
])
def test_layout_placement_must_name_a_loaded_piece(tmp_path, edit, match):
    data = layout_data()
    edit(data)
    with pytest.raises(ManifestError, match=match):
        layouts_from(tmp_path, data)


def test_layout_name_twice_raises(tmp_path):
    with pytest.raises(ManifestError) as err:
        layouts_from(tmp_path, layout_data(), layout_data())
    assert "file1.json" in str(err.value) and "file0.json" in str(err.value)
    assert "'shop'" in str(err.value)


def test_connector_end_names_a_missing_layout_raises(tmp_path):
    other = layout_data()
    other["layout"] = "stage"
    with pytest.raises(ManifestError, match=r"lead_run.*layout 'shop'.*not loaded"):
        layouts_from(tmp_path, other)


def test_connector_end_host_must_be_in_its_layout(tmp_path):
    data = layout_data()
    del data["placements"]["Lamp_1"]
    del data["placements"]["Bench_1"]
    data["placements"]["Elsewhere"] = {"pack": "shop", "piece": "Doll", "position": [0, 0, 0],
                                       "quaternion": [0, 0, 0, 1]}
    with pytest.raises(ManifestError, match=r"lead_run.*Bench_1"):
        layouts_from(tmp_path, data)


def test_connector_end_host_must_publish_the_mount(tmp_path):
    data = layout_data()
    data["placements"]["Bench_1"]["piece"] = "Doll"  # a doll has no `socket`
    del data["placements"]["Lamp_1"]["on"]
    with pytest.raises(ManifestError, match=r"lead_run.*socket"):
        layouts_from(tmp_path, data)


@pytest.mark.parametrize("facing", [[0.0, 1.0, 0.0], [0.6, 0.3, 0.7], [0.0, 0.0, 2.0],
                                    [0.0, 0.0, 0.0], [1.0, 0.0]])
def test_stand_facing_must_be_horizontal(tmp_path, facing):
    data = layout_data()
    data["stands"]["bench"]["facing"] = facing
    with pytest.raises(ManifestError, match="facing"):
        layouts_from(tmp_path, data)


def test_stand_facing_accepts_any_horizontal_unit_vector(tmp_path):
    data = layout_data()
    data["stands"]["bench"]["facing"] = [0.6, 0.0, 0.8]
    assert layouts_from(tmp_path, data)["shop"].stands["bench"].facing == (0.6, 0.0, 0.8)


@pytest.mark.parametrize("edit, match", [
    (lambda d: d.update(frame="blender"), "must be one of"),
    (lambda d: d["stands"]["bench"].update(posture="lie"), "must be one of"),
    (lambda d: d["placements"]["Bench_1"].update(quaternion=[0, 0, 0, 0]), "zero vector"),
    (lambda d: d["placements"]["Bench_1"].update(position=[0, 0]), "list of 3"),
    (lambda d: d["placements"]["Bench_1"].update(extra=1), "unknown field"),
    (lambda d: d["stands"]["bench"].pop("clip"), "missing field 'clip'"),
    (lambda d: d.pop("stands"), "missing field 'stands'"),
])
def test_layout_refusals(tmp_path, edit, match):
    data = layout_data()
    edit(data)
    with pytest.raises(ManifestError, match=match) as err:
        layouts_from(tmp_path, data)
    assert "file0.json" in str(err.value)


def test_load_layouts_refuses_a_stray_file_and_skips_hidden_ones(tmp_path):
    folder = tmp_path / "layouts"
    folder.mkdir()
    (folder / "shop.json").write_text(json.dumps(layout_data()))
    (folder / ".DS_Store").write_text("x")
    assert list(load_layouts(folder, shop_packs())) == ["shop"]
    (folder / "stray.txt").write_text("x")
    with pytest.raises(ManifestError, match="stray.txt"):
        load_layouts(folder, shop_packs())
    with pytest.raises(ManifestError, match="no_such_folder"):
        load_layouts(tmp_path / "no_such_folder", shop_packs())


def test_layout_placement_on_itself_raises(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_1"]["on"] = "shop:Lamp_1.plug"
    with pytest.raises(ManifestError, match=r"Lamp_1.*into itself"):
        layouts_from(tmp_path, data)


def test_self_refs_parse_only_where_asked_and_round_trip():
    ref = Ref.parse("self.plug", self_ok=True)
    assert ref == Ref("self", None, "plug") and ref.is_self
    assert Ref.parse(str(ref), self_ok=True) == ref
    for text in ("shop:Bench.seat_1", "rig:hand.R"):
        assert Ref.parse(str(Ref.parse(text)), self_ok=True) == Ref.parse(text)
    with pytest.raises(ManifestError):
        Ref.parse("self.plug")  # clips and layouts never take a self ref
    for bad in ("self.", "self.a b", "self.a.b"):
        with pytest.raises(ManifestError):
            Ref.parse(bad, self_ok=True)


def test_clip_hold_refs_refuse_a_self_host(tmp_path):
    data = clips_data()
    data["clips"]["sit_reach"]["holds"][0]["ref"] = "self.plug"
    with fails("ref"):
        load_clips_from(tmp_path, data)


def test_check_connectors_guards_a_placement_of_an_unloaded_piece():
    layout = load_layouts(FIX / "layouts", shop_packs())["shop"]
    ghost = dataclasses.replace(layout.placements["Bench_1"], piece="Ghost")
    layout = dataclasses.replace(layout, placements={**layout.placements, "Bench_1": ghost})
    with pytest.raises(ManifestError, match=r"lead_run.*Bench_1.*shop:Ghost"):
        check_connectors(shop_packs(), {"shop": layout})
    with pytest.raises(ManifestError, match=r"lead_run.*Bench_1.*shop:Bench"):
        check_connectors({"shop": dataclasses.replace(shop_pack(), pieces={
            n: p for n, p in shop_pack().pieces.items() if n != "Bench"})},
            {"shop": load_layouts(FIX / "layouts", shop_packs())["shop"]})


# --- posture and lifts ---------------------------------------------------------------------


def _stand_data(**changes) -> dict:
    """The shop's `sit_idle` made a standing clip: no seat, posture `stand`."""
    data = clips_data()
    clip = data["clips"]["sit_idle"]
    clip.pop("seat")
    clip.update(posture="stand", **changes)
    return data


def _lift(**changes) -> dict:
    return {"support": "foot.R", "when": [[5, 9]], "why": "a step", **changes}


def _stand_clip(tmp_path, **changes) -> Clip:
    return {c.name: c for c in load_clips_from(tmp_path, _stand_data(**changes))}["sit_idle"]


def test_posture_absent_loads_as_none(tmp_path):
    data = clips_data()
    for clip in data["clips"].values():
        clip.pop("posture")
    clips = load_clips_from(tmp_path, data)
    assert all(c.posture is None and c.lifts == () for c in clips)
    assert {c.name: c.posture for c in shop_clips().values()} == {
        "sit_reach": "sit", "sit_idle": "sit", "sit_idle_L": "sit"}


@pytest.mark.parametrize("posture", ["stand", "lie", "free"])
def test_seat_clip_must_sit(tmp_path, posture):
    with fails(r"a clip with a seat must be 'sit'"):
        load_clips_from(tmp_path, _clip_data(posture=posture))


def test_sit_clip_needs_a_seat(tmp_path):
    data = clips_data()
    data["clips"]["sit_idle"].pop("seat")
    with fails(r"a 'sit' clip must name a seat"):
        load_clips_from(tmp_path, data)
    for posture in ("stand", "lie", "free"):  # no seat is fine for any other posture
        data["clips"]["sit_idle"]["posture"] = posture
        assert {c.name: c for c in load_clips_from(tmp_path, data)}["sit_idle"].posture == posture


def test_posture_must_be_a_known_one(tmp_path):
    with fails("must be one of"):
        load_clips_from(tmp_path, _clip_data(posture="kneel"))


def test_lifts_read_as_windows_with_a_support_and_a_why(tmp_path):
    clip = _stand_clip(tmp_path, lifts=[_lift(), _lift(support="foot.L", when=[[1, 2], [20, 24]],
                                                       why="a kick")])
    assert clip.lifts[0] == Lift("foot.R", ((5, 9),), "a step")
    assert clip.lifts[1].when == ((1, 2), (20, 24)) and clip.lifts[1].why == "a kick"


@pytest.mark.parametrize("posture", ["sit", "lie", "free", None])
def test_lifts_only_on_stand_clips(tmp_path, posture):
    data = _stand_data(lifts=[_lift()])
    if posture is None:
        data["clips"]["sit_idle"].pop("posture")
    else:
        data["clips"]["sit_idle"]["posture"] = posture
        if posture == "sit":
            data["clips"]["sit_idle"]["seat"] = "shop:Bench.seat_1"
    with fails("only a 'stand' clip has lifts"):
        load_clips_from(tmp_path, data)


@pytest.mark.parametrize("when,match", [
    ([[20, 40]], "outside the clip's frames"),
    ([[0, 5]], "outside the clip's frames"),
    ([[9, 5]], "runs backwards"),
    ([], "must not be empty"),
])
def test_lift_window_outside_frames_raises(tmp_path, when, match):
    with fails(match):
        load_clips_from(tmp_path, _stand_data(lifts=[_lift(when=when)]))


def test_a_lift_needs_a_why_and_a_support(tmp_path):
    with fails("missing field 'why'"):
        load_clips_from(tmp_path, _stand_data(lifts=[{"support": "foot.R", "when": [[5, 9]]}]))
    with fails("non-empty string"):
        load_clips_from(tmp_path, _stand_data(lifts=[_lift(why="  ")]))
    with fails("missing field 'support'"):
        load_clips_from(tmp_path, _stand_data(lifts=[{"when": [[5, 9]], "why": "a step"}]))


# --- fasteners and retainers ---------------------------------------------------------------


def _lamp_parts(tmp_path, **by_name) -> dict:
    """The shop pack loaded with each named Lamp part's raw record updated."""
    data = pack_data()
    for part in data["pieces"]["Lamp"]["parts"]:
        part.update(by_name.get(part["name"], {}))
    return {p.name: p for p in load_pack_from(tmp_path, data).pieces["Lamp"].parts}


def test_fastener_and_retains_default_off(tmp_path):
    parts = _lamp_parts(tmp_path)
    assert all(p.fastener is False and p.retains == () for p in parts.values())


def test_retains_names_parts_of_the_piece(tmp_path):
    parts = _lamp_parts(tmp_path, lead={"retains": ["base", "stem"]},
                        stem={"fastener": True})
    assert parts["lead"].retains == ("base", "stem") and parts["stem"].fastener is True
    assert parts["base"].fastener is False


@pytest.mark.parametrize("retains,match", [
    (["Bench"], "names no part of this piece: 'Bench'"),
    (["lead"], "cannot retain itself"),
    ([], "must not be empty"),
    (["base", "base"], "names repeat"),
    ("base", "must be a list"),
])
def test_retains_that_names_no_other_part_raises(tmp_path, retains, match):
    with fails(match):
        _lamp_parts(tmp_path, lead={"retains": retains})


def test_fastener_must_be_a_bool(tmp_path):
    with fails("true or false"):
        _lamp_parts(tmp_path, stem={"fastener": "yes"})


# --- finish --------------------------------------------------------------------------------


def _lamp_rows(tmp_path, **finish) -> dict:
    """The shop pack loaded with each named Lamp row's `finish` set."""
    data = pack_data()
    for row in data["pieces"]["Lamp"]["listed"]:
        if row["id"] in finish:
            row["finish"] = finish[row["id"]]
    return {r.id: r for r in load_pack_from(tmp_path, data).pieces["Lamp"].listed}


def test_a_row_is_solid_unless_it_says_otherwise(tmp_path):
    rows = _lamp_rows(tmp_path)
    assert {r.finish for r in rows.values()} == {"solid"}


@pytest.mark.parametrize("finish", FINISHES)
def test_every_finish_loads(tmp_path, finish):
    row_id = pack_data()["pieces"]["Lamp"]["listed"][0]["id"]
    assert _lamp_rows(tmp_path, **{row_id: finish})[row_id].finish == finish


def test_an_unknown_finish_raises(tmp_path):
    row_id = pack_data()["pieces"]["Lamp"]["listed"][0]["id"]
    with fails("must be one of"):
        _lamp_rows(tmp_path, **{row_id: "gilt"})


# --- held ----------------------------------------------------------------------------------


def _lamp_held(tmp_path, **held) -> dict:
    """The shop pack loaded with each named Lamp row's `held` set."""
    data = pack_data()
    for row in data["pieces"]["Lamp"]["listed"]:
        if row["id"] in held:
            row["held"] = held[row["id"]]
    return {r.id: r for r in load_pack_from(tmp_path, data).pieces["Lamp"].listed}


def test_a_row_rests_unless_it_says_it_is_bonded(tmp_path):
    assert {r.held for r in _lamp_held(tmp_path).values()} == {"rests"}


@pytest.mark.parametrize("held", HELD)
def test_every_held_loads(tmp_path, held):
    row_id = pack_data()["pieces"]["Lamp"]["listed"][0]["id"]
    assert _lamp_held(tmp_path, **{row_id: held})[row_id].held == held


def test_an_unknown_held_raises(tmp_path):
    row_id = pack_data()["pieces"]["Lamp"]["listed"][0]["id"]
    with fails("must be one of"):
        _lamp_held(tmp_path, **{row_id: "welded"})


# --- rests_on ------------------------------------------------------------------------------


def test_rests_on_names_a_placement(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_2"]["rests_on"] = "Bench_1"
    layout = layouts_from(tmp_path, data)["shop"]
    assert layout.placements["Lamp_2"].rests_on == "Bench_1"
    assert layout.placements["Lamp_2"].on is None


@pytest.mark.parametrize("target,match", [
    ("Lamp_2", "cannot rest on itself"),
    ("Table", "names no placement of this layout: 'Table'"),
    ("shop:Bench_1.socket", "must be a name"),
])
def test_rests_on_that_names_no_other_placement_raises(tmp_path, target, match):
    data = layout_data()
    data["placements"]["Lamp_2"]["rests_on"] = target
    with fails(match):
        layouts_from(tmp_path, data)


def test_rests_on_and_on_together_raise(tmp_path):
    data = layout_data()
    data["placements"]["Lamp_1"]["rests_on"] = "Bench_1"  # it already has `on`
    with fails("not both"):
        layouts_from(tmp_path, data)


# --- each record owns one validator ------------------------------------------------------------


def test_view_with_stale_hidden_why_raises_when_built():
    with pytest.raises(ManifestError, match=r"view: hidden_why: names no hidden side"):
        View(nearest_m=1, why="x", hidden=(), hidden_why="y")


@pytest.mark.parametrize(("changes", "field", "match"), [
    ({"nearest_m": 0}, "nearest_m", "above 0"),
    ({"nearest_m": True}, "nearest_m", "finite number"),
    ({"why": " "}, "why", "non-empty string"),
    ({"hidden": ("underside",), "hidden_why": "x"}, r"hidden\[0\]", "must be one of"),
    ({"hidden": ("back", "back"), "hidden_why": "x"}, "hidden", "repeat"),
    ({"hidden": ("back",)}, "hidden_why", "needs a hidden_why"),
])
def test_a_view_built_in_code_checks_every_rule_the_loader_does(changes, field, match):
    with pytest.raises(ManifestError, match=rf"view: {field}: .*{match}"):
        dataclasses.replace(View(1.0, "seen at arm's length"), **changes)


def test_sit_clip_without_seat_raises_when_built():
    sit = shop_clips()["sit_idle"]
    with pytest.raises(ManifestError, match=r"clip 'sit_idle': posture: a 'sit' clip must name"):
        dataclasses.replace(sit, seat=None)


@pytest.mark.parametrize(("changes", "field", "match"), [
    ({"frames": (10, 3)}, "frames", "must be before end"),
    ({"fps": 0}, "fps", "above 0"),
    ({"authoring": "solved"}, "targets", "needs key targets"),
    ({"seat_when": ((2, 5),), "seat": None, "posture": None}, "seat_when", "needs a seat"),
    ({"posture": "stand"}, "posture", "must be 'sit'"),
    ({"lifts": (Lift("foot.R", ((2, 5),), "a hop"),)}, "lifts", "only a 'stand' clip"),
    ({"seat_when": ((10, 5),)}, r"seat_when\[0\]", "runs backwards"),
    ({"seat_when": ((20, 40),)}, r"seat_when\[0\]", "outside the clip's frames"),
    ({"seat_when": ()}, "seat_when", "must not be empty"),
    ({"loop_from": 5}, "loop_from", "needs loop"),
    ({"mirror_of": "sit_idle"}, "mirror_of", "cannot mirror itself"),
])
def test_a_clip_built_in_code_checks_every_rule_the_loader_does(changes, field, match):
    with pytest.raises(ManifestError, match=rf"clip 'sit_idle': {field}: .*{match}"):
        dataclasses.replace(shop_clips()["sit_idle"], **changes)


def test_decal_part_without_on_raises_when_built():
    with pytest.raises(ManifestError, match=r"part 'dot': on: .*'dot' is a decal but has no `on`"):
        Part("dot", "paint", "decal", None, "g", "decal", (), None)
    with pytest.raises(ManifestError, match=r"part 'dot': role: .*is a decal but has the role"):
        Part("dot", "paint", "decal", "body", "g", "solid", (), None)


def test_a_part_built_in_code_checks_its_vocabulary_and_what_it_retains():
    with pytest.raises(ManifestError, match=r"part 'a': shape: must be one of"):
        Part("a", "l", "sphere", None, "g", "solid", (), None)
    with pytest.raises(ManifestError, match=r"part 'a': role: must be one of"):
        Part("a", "l", "box", None, "g", "paint", (), None)
    with pytest.raises(ManifestError, match=r"part 'a': retains: .*cannot retain itself"):
        Part("a", "l", "box", None, "g", "solid", (), None, retains=("a",))


def test_rests_on_defaults_empty_and_reads_two_supports(tmp_path):
    parts = _lamp_parts(tmp_path, lead={"on": None, "rests_on": ["base", "stem"]})
    assert parts["lead"].rests_on == ("base", "stem")
    assert parts["base"].rests_on == () and parts["stem"].rests_on == ()


@pytest.mark.parametrize("lead,match", [
    ({"on": None, "rests_on": ["base"]}, "use `on` for one support"),
    ({"on": "base", "rests_on": ["base", "stem"]}, "both `on` and `rests_on`"),
    ({"on": None, "rests_on": ["lead", "base"]}, "cannot rest on itself"),
    ({"on": None, "rests_on": ["base", "ghost"]}, "names no part of this piece: 'ghost'"),
    ({"on": None, "rests_on": ["base", "base"]}, "names repeat"),
    ({"on": None, "rests_on": []}, "must not be empty"),
    ({"on": None, "fastener": True, "rests_on": ["base", "stem"]}, "is a fastener"),
    ({"on": None, "shape": "decal", "role": "decal", "rests_on": ["base", "stem"]}, "is a decal"),
])
def test_a_bad_rests_on_is_a_load_error(tmp_path, lead, match):
    with fails(match):
        _lamp_parts(tmp_path, lead=lead)


def test_rests_on_on_a_rigged_piece_is_a_load_error(tmp_path):
    data = pack_data()
    doll = data["pieces"]["Doll"]["parts"]
    doll[0].update(on=None, rests_on=[doll[1]["name"], doll[2]["name"]])
    with fails("rigged"):
        load_pack_from(tmp_path, data)


def test_a_part_built_in_code_checks_its_supports():
    with pytest.raises(ManifestError, match=r"part 'a': rests_on: .*cannot rest on itself"):
        Part("a", "l", "box", None, "g", "solid", (), None, rests_on=("a", "b"))
    with pytest.raises(ManifestError, match="use `on` for one support"):
        Part("a", "l", "box", None, "g", "solid", (), None, rests_on=("b",))
    with pytest.raises(ManifestError, match="names 'b' twice"):
        Part("a", "l", "box", None, "g", "solid", (), None, rests_on=("b", "b"))


def test_loader_error_names_file_and_field(tmp_path):
    """The loader builds the record, and the record's own refusal comes back at the file and
    the field's place."""
    data = pack_data()
    data["pieces"]["Lamp"]["view"] = _view(hidden_why="stale")
    with pytest.raises(ManifestError) as err:
        load_pack_from(tmp_path, data)
    assert str(err.value).startswith(f"{tmp_path / 'test.pack.json'}: pieces.Lamp.view.hidden_why: ")
    data = clips_data()
    data["clips"]["sit_idle"].pop("seat")
    with pytest.raises(ManifestError) as err:
        load_clips_from(tmp_path, data)
    assert str(err.value).startswith(
        f"{tmp_path / 'test.clips.json'}: clips.sit_idle.posture: ")


def test_a_view_in_a_build_spec_reads_by_the_pack_rules(tmp_path):
    """`parse_view` is the one reader of a `view` object, for a pack and for a build spec."""
    spec = tmp_path / "spec.json"
    assert parse_view(_view(), spec, "pieces[0].view") == View(1.0, "a test piece seen at arm's "
                                                                    "length")
    with pytest.raises(ManifestError, match=r"spec.json: pieces\[0\].view.hidden_why: "):
        parse_view(_view(hidden_why="stale"), spec, "pieces[0].view")
    with pytest.raises(ManifestError, match=r"spec.json: pieces\[0\].view: unknown field 'extra'"):
        parse_view(_view(extra=1), spec, "pieces[0].view")
