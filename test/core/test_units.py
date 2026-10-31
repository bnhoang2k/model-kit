import json
from pathlib import Path

import pytest

from model_kit.core import tolerances
from model_kit.core.errors import ManifestError
from model_kit.core.units import (
    DepthCamera,
    Style,
    depth_step,
    load_style,
    thickness_base,
)


def write_style(tmp_path: Path, data: dict, name: str = "style.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def make_style(units_per_metre: float = 0.5, min_thickness=None) -> Style:
    return Style(
        units_per_metre=units_per_metre,
        page_camera={},
        dials={"min_thickness": min_thickness},
    )


def test_load_style_reads_the_three_fields(tmp_path):
    path = write_style(
        tmp_path,
        {"units_per_metre": 0.5, "page_camera": {"fov": 30}, "dials": {"proportion": 0}},
    )
    style = load_style(path)
    assert style == Style(0.5, {"fov": 30}, {"proportion": 0})


def test_thickness_base_takes_larger_of_listed_and_dial():
    # listed min 0.002 m at 0.5 u/m is 0.001 units
    assert thickness_base(make_style(min_thickness=None), [0.004, 0.002]) == pytest.approx(0.001)
    assert thickness_base(make_style(min_thickness=0), [0.002]) == pytest.approx(0.001)
    # a dial floor of 0.004 m is 0.002 units and beats the listed 0.001
    assert thickness_base(make_style(min_thickness=0.004), [0.002]) == pytest.approx(0.002)


def test_thickness_base_works_from_the_dial_alone():
    assert thickness_base(make_style(min_thickness=0.004), []) == pytest.approx(0.002)


def test_thickness_base_never_zero():
    with pytest.raises(ManifestError):
        thickness_base(make_style(min_thickness=0), [])
    with pytest.raises(ManifestError):
        thickness_base(make_style(min_thickness=None), [])
    with pytest.raises(ManifestError):
        thickness_base(make_style(min_thickness=0), [0.0])


def test_thickness_base_rejects_negative_listed_values():
    with pytest.raises(ManifestError):
        thickness_base(make_style(), [-0.002])


def test_style_without_units_raises(tmp_path):
    path = write_style(tmp_path, {"page_camera": {}, "dials": {}}, name="no_units_style.json")
    with pytest.raises(ManifestError, match="no_units_style.json"):
        load_style(path)


@pytest.mark.parametrize("bad", [0, -1, -0.5, "0.5", None, True, float("nan"), float("inf")])
def test_units_must_be_positive(tmp_path, bad):
    path = write_style(tmp_path, {"units_per_metre": bad, "page_camera": {}, "dials": {}})
    with pytest.raises(ManifestError, match="style.json"):
        load_style(path)


def test_style_file_that_is_not_json_raises_and_names_the_file(tmp_path):
    path = tmp_path / "broken_style.json"
    path.write_text("{not json")
    with pytest.raises(ManifestError, match="broken_style.json"):
        load_style(path)


def test_missing_style_file_raises_and_names_the_file(tmp_path):
    with pytest.raises(ManifestError, match="absent_style.json"):
        load_style(tmp_path / "absent_style.json")


def test_dial_min_thickness_must_be_a_number_or_null(tmp_path):
    path = write_style(
        tmp_path, {"units_per_metre": 0.5, "page_camera": {}, "dials": {"min_thickness": "x"}}
    )
    style = load_style(path)
    with pytest.raises(ManifestError):
        thickness_base(style, [0.002])


def edges_style(tmp_path: Path, edges: dict) -> Path:
    return write_style(
        tmp_path, {"units_per_metre": 0.5, "page_camera": {}, "dials": {"edges": edges}}
    )


def test_crease_read_from_edges_dial(tmp_path):
    path = edges_style(tmp_path, {"outline": "runtime", "crease_deg": 35, "crease_why": "a reason"})
    style = load_style(path)
    assert style.crease_deg == 35
    assert style.crease_why == "a reason"


def test_style_built_in_code_reads_the_angle_from_dials():
    dials = {"edges": {"crease_deg": 30, "crease_why": "a reason"}}
    style = Style(0.5, {}, dials)
    assert (style.crease_deg, style.crease_why) == (30, "a reason")
    assert Style(0.5, {}, {}).crease_deg is None


@pytest.mark.parametrize(
    "edges",
    [{"crease_deg": 200, "crease_why": "x"}, {"crease_deg": 30}, [], "none"],
)
def test_style_built_in_code_refuses_a_bad_edges_dial(edges):
    with pytest.raises(ManifestError, match="dials.edges"):
        Style(0.5, {}, {"edges": edges})


def test_crease_absent_loads_as_none(tmp_path):
    for dials in ({}, {"edges": {"outline": "runtime"}}, {"edges": None}):
        path = write_style(tmp_path, {"units_per_metre": 0.5, "page_camera": {}, "dials": dials})
        style = load_style(path)
        assert style.crease_deg is None
        assert style.crease_why is None


def test_crease_without_why_raises(tmp_path):
    for edges in ({"crease_deg": 35}, {"crease_deg": 35, "crease_why": ""},
                  {"crease_deg": 35, "crease_why": "  "}, {"crease_deg": 35, "crease_why": 3}):
        with pytest.raises(ManifestError, match="crease_why"):
            load_style(edges_style(tmp_path, edges))


@pytest.mark.parametrize(
    "bad", [0, 180, -5, 200, "35", None, True, float("nan"), float("inf")]
)
def test_crease_out_of_range_raises(tmp_path, bad):
    path = edges_style(tmp_path, {"crease_deg": bad, "crease_why": "a reason"})
    with pytest.raises(ManifestError, match="crease_deg"):
        load_style(path)


@pytest.mark.parametrize(
    ("dials", "draws"),
    [
        ({"edges": {"outline": "runtime"}}, True),
        ({"edges": {"outline": "none"}}, False),
        ({"edges": {"outline": None}}, False),
        ({"edges": {}}, False),
        ({}, False),
    ],
)
def test_outline_none_draws_no_outline(dials, draws):
    assert Style(0.5, {}, dials).draws_outline is draws


def test_tolerance_factors_hold_the_spec_values():
    assert tolerances.TOUCH_OF_EFFECTOR == 0.085
    assert tolerances.BEHIND_MAX_OF_EFFECTOR == 0.5
    assert tolerances.REACH_MARGIN_OF_EFFECTOR == 0.1
    assert tolerances.PLANT_EXTRA_OF_BUDGET == 2.0
    assert tolerances.CALIBRATION_PASS == 0.5
    assert tolerances.CALIBRATION_FAIL == 2.0


def test_unit_length_slack_lives_with_the_other_tolerances():
    assert tolerances.UNIT_LENGTH_SLACK == 1e-6


# --- a style checks its own dials: the leaves dial and the two ways to declare lines -------------

LEAVES = {"normal_lean": 0.8, "centre_drop": 0.35, "why": "a render choice, no botanical fact"}
BORDERS = {"outline": "runtime", "borders_only": True, "borders_why": "the page draws outlines"}


def test_leaves_dial_without_why_raises():
    for why in (None, "", "  ", 3):
        leaves = {**LEAVES, "why": why}
        with pytest.raises(ManifestError, match=r"style: dials\.leaves\.why: "):
            Style(0.5, {}, {"leaves": leaves})
    with pytest.raises(ManifestError, match=r"dials\.leaves\.why"):
        Style(0.5, {}, {"leaves": {k: v for k, v in LEAVES.items() if k != "why"}})
    assert Style(0.5, {}, {"leaves": LEAVES}).dials["leaves"] == LEAVES


def test_loader_names_the_file_and_the_leaves_dial(tmp_path):
    path = write_style(tmp_path, {"units_per_metre": 0.5, "page_camera": {},
                                  "dials": {"leaves": {**LEAVES, "why": ""}}})
    with pytest.raises(ManifestError) as err:
        load_style(path)
    assert str(err.value).startswith(f"{path}: dials.leaves.why: ")


def test_borders_only_reads_back_with_its_reason():
    style = Style(0.5, {}, {"edges": BORDERS})
    assert style.borders_only is True
    assert style.borders_why == "the page draws outlines"
    assert style.crease_deg is None and style.draws_outline is True
    plain = Style(0.5, {}, {"edges": {"outline": "runtime", "crease_deg": 35, "crease_why": "x"}})
    assert plain.borders_only is False and plain.borders_why is None
    assert Style(0.5, {}, {}).borders_only is False


@pytest.mark.parametrize(("edges", "field"), [
    ({**BORDERS, "borders_why": ""}, "borders_why"),
    ({**BORDERS, "borders_why": None}, "borders_why"),
    ({k: v for k, v in BORDERS.items() if k != "borders_why"}, "borders_why"),
    ({**BORDERS, "crease_deg": 35, "crease_why": "x"}, "borders_only"),
    ({**BORDERS, "borders_only": "yes"}, "borders_only"),
    ({"outline": "runtime", "borders_why": "stale"}, "borders_why"),
    ({"outline": "runtime", "borders_only": False, "borders_why": "stale"}, "borders_why"),
])
def test_a_bad_borders_only_declaration_raises(tmp_path, edges, field):
    with pytest.raises(ManifestError, match=rf"dials\.edges\.{field}"):
        Style(0.5, {}, {"edges": edges})
    with pytest.raises(ManifestError, match=rf"style.json: dials\.edges\.{field}"):
        load_style(edges_style(tmp_path, edges))


def test_a_style_may_declare_borders_only_false_with_a_crease():
    edges = {"borders_only": False, "crease_deg": 35, "crease_why": "x"}
    assert Style(0.5, {}, {"edges": edges}).crease_deg == 35


# --- the reference camera's depth buffer ---------------------------------------------------------

THREE_JS_CAMERA = {"projection": "ortho", "turn_deg": 14, "tilt_deg": 52,
                   "depth_projection": "perspective", "near_m": 0.1, "far_m": 2000,
                   "depth_bits": 24, "depth_why": "three.js defaults, 24 bits assumed"}


def camera_style(**changes) -> Style:
    return Style(units_per_metre=0.5, page_camera={**THREE_JS_CAMERA, **changes}, dials={})


def test_depth_step_matches_the_khronos_derivation():
    n, f, bits, z = 0.1, 2000, 24, 10
    assert depth_step(camera_style(), z) == pytest.approx(z * z * (f - n) / (f * n * 2**bits))
    # An independent reading: the buffer stores z_w = s * f * (z - n) / (z * (f - n)); the eye
    # distance one stored step covers at z is 1 / (d z_w / d z), taken here by finite difference.
    s = 2**bits

    def stored(depth: float) -> float:
        return s * f * (depth - n) / (depth * (f - n))

    h = 1e-6
    slope = (stored(z + h) - stored(z - h)) / (2 * h)
    assert depth_step(camera_style(), z) == pytest.approx(1 / slope, rel=1e-6)


def test_depth_step_grows_with_the_square_of_the_distance_and_falls_with_more_bits():
    style = camera_style()
    assert depth_step(style, 20) == pytest.approx(4 * depth_step(style, 10))
    assert depth_step(camera_style(depth_bits=16), 10) == pytest.approx(
        256 * depth_step(style, 10))


def test_load_style_reads_the_depth_camera(tmp_path):
    style = load_style(write_style(
        tmp_path, {"units_per_metre": 0.5, "page_camera": THREE_JS_CAMERA, "dials": {}}))
    assert style.depth_camera == DepthCamera(0.1, 2000.0, 24, "three.js defaults, 24 bits assumed")
    assert make_style().depth_camera is None


@pytest.mark.parametrize("change, match", [
    ({"depth_projection": "ortho"}, "only a 'perspective' depth buffer"),
    ({"depth_projection": "logarithmic"}, "only a 'perspective' depth buffer"),
    ({"depth_bits": 24.0}, "integer depth buffer"),
    ({"depth_bits": True}, "integer depth buffer"),
    ({"depth_bits": "float32"}, "integer depth buffer"),
    ({"depth_bits": 0}, "integer depth buffer"),
    ({"depth_bits": 64}, "integer depth buffer"),
    ({"near_m": 0}, "near_m: must be a number above 0"),
    ({"far_m": "2000"}, "far_m: must be a number above 0"),
    ({"near_m": 5, "far_m": 5}, "far_m: must be past near_m"),
    ({"depth_why": " "}, "depth_why: needs a depth_why"),
])
def test_unsupported_depth_declaration_raises(tmp_path, change, match):
    camera = {**THREE_JS_CAMERA, **change}
    with pytest.raises(ManifestError, match=match):
        Style(units_per_metre=0.5, page_camera=camera, dials={})
    path = write_style(tmp_path, {"units_per_metre": 0.5, "page_camera": camera, "dials": {}})
    with pytest.raises(ManifestError, match=match):
        load_style(path)


def test_a_half_made_depth_declaration_raises():
    for key in ("depth_projection", "near_m", "far_m", "depth_bits", "depth_why"):
        camera = {k: v for k, v in THREE_JS_CAMERA.items() if k != key}
        with pytest.raises(ManifestError, match=f"{key} is missing"):
            Style(units_per_metre=0.5, page_camera=camera, dials={})


def test_depth_step_refuses_what_the_buffer_does_not_hold():
    with pytest.raises(ManifestError, match="declares no depth camera"):
        depth_step(make_style(), 10)
    for bad in (0, -1, 0.05, 2001):
        with pytest.raises(ManifestError):
            depth_step(camera_style(), bad)
