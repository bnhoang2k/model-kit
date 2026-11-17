import json
from pathlib import Path

import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import (
    Layout,
    LayoutPlacement,
    Stage,
    StageCamera,
    StageLight,
    check_stage,
    load_stage,
)

Q = (0.0, 0.0, 0.0, 1.0)


def layouts(*ids: str) -> dict[str, Layout]:
    placements = {i: LayoutPlacement(i, "shop", "Box", (0.0, 0.0, 0.0), Q) for i in ids}
    return {"room": Layout("room", "gltf", placements, {})}


def stage_data() -> dict:
    return {
        "stage": 1, "layout": "room", "mode": "indirect", "texels_per_m": 256,
        "samples": 256, "seed": 1, "layers": ["window", "lamp"], "live": ["Lamp_1"],
        "lights": {
            "sun": {"kind": "sun", "position": [0, 5, 0], "direction": [0, -1, 0],
                    "layer": "window"},
            "bulb": {"kind": "point", "position": [1, 1, 0], "radius_m": 0.05,
                     "layer": "lamp"},
        },
        "cameras": {"main": {"position": [0, 1, 3], "target": [0, 1, 0], "fov_deg": 50,
                             "aspect": 1.5}},
    }


def write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "room.stage.json"
    path.write_text(json.dumps(data))
    return path


def test_round_trip(tmp_path):
    stage = load_stage(write(tmp_path, stage_data()))
    assert stage == Stage(
        layout="room", mode="indirect",
        lights={
            "sun": StageLight("sun", "sun", (0.0, 5.0, 0.0), (0.0, -1.0, 0.0), None, None,
                              None, None, None, "window"),
            "bulb": StageLight("bulb", "point", (1.0, 1.0, 0.0), None, None, None, None,
                               0.05, None, "lamp"),
        },
        layers=("window", "lamp"), live=frozenset({"Lamp_1"}),
        cameras={"main": StageCamera("main", (0.0, 1.0, 3.0), (0.0, 1.0, 0.0), 50.0, 1.5)},
        texels_per_m=256.0, samples=256, seed=1)
    check_stage(stage, layouts("Box_1", "Lamp_1"))


def test_unknown_live_piece_named(tmp_path):
    data = stage_data() | {"live": ["nope"]}
    stage = load_stage(write(tmp_path, data))
    with pytest.raises(ManifestError, match=r"'nope'.*'room'"):
        check_stage(stage, layouts("Box_1", "Lamp_1"))


def test_unknown_layout_named(tmp_path):
    stage = load_stage(write(tmp_path, stage_data() | {"layout": "attic"}))
    with pytest.raises(ManifestError, match="'attic'"):
        check_stage(stage, layouts("Box_1", "Lamp_1"))


def test_light_on_unknown_layer_named(tmp_path):
    data = stage_data()
    data["lights"]["bulb"]["layer"] = "ghost"
    with pytest.raises(ManifestError, match="'ghost'"):
        load_stage(write(tmp_path, data))


def test_layer_with_no_lights_refused(tmp_path):
    data = stage_data() | {"layers": ["window", "lamp", "spare"]}
    with pytest.raises(ManifestError, match="'spare'"):
        load_stage(write(tmp_path, data))


def test_all_live_refused(tmp_path):
    stage = load_stage(write(tmp_path, stage_data()))
    with pytest.raises(ManifestError, match="no still piece"):
        check_stage(stage, layouts("Lamp_1"))


def test_bad_mode_refused(tmp_path):
    with pytest.raises(ManifestError, match="mode"):
        load_stage(write(tmp_path, stage_data() | {"mode": "baked"}))


def test_spot_needs_cone(tmp_path):
    data = stage_data()
    data["lights"]["bulb"] = {"kind": "spot", "position": [1, 1, 0], "direction": [0, -1, 0],
                              "blend": 0.2, "layer": "lamp"}
    with pytest.raises(ManifestError, match="cone_deg"):
        load_stage(write(tmp_path, data))


def test_kind_specific_fields_refused_elsewhere(tmp_path):
    data = stage_data()
    data["lights"]["bulb"]["size"] = [1, 1]
    with pytest.raises(ManifestError, match="size"):
        load_stage(write(tmp_path, data))


def test_repeated_live_id_named(tmp_path):
    with pytest.raises(ManifestError, match="'Lamp_1'"):
        load_stage(write(tmp_path, stage_data() | {"live": ["Lamp_1", "Lamp_1"]}))


@pytest.mark.parametrize("version", [1.0, True, 2, "1"])
def test_bad_file_version_refused(tmp_path, version):
    with pytest.raises(ManifestError, match="stage"):
        load_stage(write(tmp_path, stage_data() | {"stage": version}))


def test_no_camera_refused(tmp_path):
    with pytest.raises(ManifestError, match="at least one camera"):
        load_stage(write(tmp_path, stage_data() | {"cameras": {}}))


def test_hemisphere_needs_sky_ground(tmp_path):
    data = stage_data()
    data["lights"]["bulb"] = {"kind": "hemisphere", "position": [0, 3, 0], "layer": "lamp"}
    with pytest.raises(ManifestError, match="sky_ground"):
        load_stage(write(tmp_path, data))


def test_spot_cone_is_a_half_angle_below_90(tmp_path):
    for bad in (0, 90, 120):
        data = stage_data()
        data["lights"]["bulb"] = {"kind": "spot", "position": [1, 1, 0], "direction": [0, -1, 0],
                                  "cone_deg": bad, "blend": 0.2, "layer": "lamp"}
        with pytest.raises(ManifestError, match="below 90"):
            load_stage(write(tmp_path, data))


def test_placement_density_round_trip(tmp_path):
    data = stage_data() | {"placement_texels_per_m": {"Box_1": 512}}
    stage = load_stage(write(tmp_path, data))
    assert stage.placement_texels_per_m == {"Box_1": 512.0}
    assert stage.texels_per_m_of("Box_1") == 512.0
    assert stage.texels_per_m_of("Box_2") == 256.0
    check_stage(stage, layouts("Box_1", "Box_2", "Lamp_1"))


def test_placement_density_absent_is_empty(tmp_path):
    stage = load_stage(write(tmp_path, stage_data()))
    assert stage.placement_texels_per_m == {}
    assert stage.texels_per_m_of("Box_1") == 256.0


@pytest.mark.parametrize("value", [0, -1, "big", True])
def test_placement_density_must_be_positive(tmp_path, value):
    data = stage_data() | {"placement_texels_per_m": {"Box_1": value}}
    with pytest.raises(ManifestError, match="placement_texels_per_m"):
        load_stage(write(tmp_path, data))


def test_placement_density_unknown_placement_named(tmp_path):
    data = stage_data() | {"placement_texels_per_m": {"nope": 512}}
    stage = load_stage(write(tmp_path, data))
    with pytest.raises(ManifestError, match=r"'nope'.*'room'"):
        check_stage(stage, layouts("Box_1", "Lamp_1"))


def test_placement_density_on_a_live_placement_refused(tmp_path):
    data = stage_data() | {"placement_texels_per_m": {"Lamp_1": 512}}
    stage = load_stage(write(tmp_path, data))
    with pytest.raises(ManifestError, match="'Lamp_1'.*live"):
        check_stage(stage, layouts("Box_1", "Lamp_1"))
