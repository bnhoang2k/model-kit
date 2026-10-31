"""Each solver factor, times the length it is a factor of, gives back the absolute
number it came from. A typo in a factor fails here, since the product is pinned to that number."""

import pytest

from model_kit.core import tolerances as tol
from model_kit.core.solve import SOLID_STEPS

REACH = tol.SOLVER_REFERENCE_REACH
RADIUS = tol.SOLVER_REFERENCE_EFFECTOR_RADIUS


def test_the_reference_lengths_are_the_arm_and_mitt():
    assert REACH == 0.2127
    assert RADIUS == 0.0586


@pytest.mark.parametrize("factor, length, original", [
    (tol.SOLID_PREFILTER_OF_REACH, REACH, 0.07),
    (tol.AVOID_HEAD_SCALE_OF_REACH, REACH, 0.30),
    (tol.QUICK_RETRY_MISS_OF_REACH, REACH, 0.01),
    (tol.SOLID_MARGIN_OF_EFFECTOR, RADIUS, 0.003),
    (tol.GEAR_MARGIN_OF_EFFECTOR, RADIUS, 0.006),
    (tol.BACK_PREFILTER_OF_EFFECTOR, RADIUS, 0.003),
    (tol.AVOID_ELBOW_OF_EFFECTOR, RADIUS, 0.05),
    (tol.AVOID_WRIST_OF_EFFECTOR, RADIUS, 0.055),
    (tol.AVOID_CENTRE_OF_EFFECTOR, RADIUS, 0.059),
    (tol.TIP_RETRY_MISS_OF_EFFECTOR, RADIUS, 0.004),
])
def test_a_length_factor_gives_back_the_original_length(factor, length, original):
    assert factor * length == pytest.approx(original, rel=1e-12)


@pytest.mark.parametrize("weight, original", [
    (tol.W_AVOID, 25.0), (tol.W_PLANE, 4.0), (tol.W_SOLID, 100.0), (tol.W_SOLID_BACK, 25.0),
    (tol.W_STAY, 0.0004), (tol.W_LIMIT, 1.0), (tol.W_LEVEL_PER_WEIGHT, 1.0),
])
def test_a_weight_is_the_original_over_the_reference_reach_squared(weight, original):
    assert weight * REACH**2 == pytest.approx(original, rel=1e-12)


def test_counts_shares_and_steps_are_the_originals():
    assert tol.SOLID_SAMPLES_PER_PART == 40
    assert tol.SKIN_SAMPLES_PER_CHUNK == 90
    assert tol.SOLID_SKIN_POINTS_HAND == 36
    assert tol.SOLID_SKIN_POINTS_FOREARM == 48
    assert tol.SOLID_SKIN_POINTS_ARM == 16
    assert tol.TIP_RETRIES == 3
    assert tol.PRESS_MARGIN_SHARE == 0.5
    assert SOLID_STEPS == (0.12, 0.03)


# --- piece-geometry factors (checks 3, 16, 18 and the take-apart sweep) ------------------------
# Each factor is an absolute value over the size of a reference gear piece.

def test_the_gear_base_and_the_decal_lift_factor_are_gone():
    """The decal lift is the piece's contact length (`sight.decal_lift`), checked against one
    depth step, so neither a gear base nor a share of the thickness base is left."""
    assert not hasattr(tol, "_SOURCE_GEAR_BASE")
    assert not hasattr(tol, "DECAL_LIFT_OF_THICKNESS")


@pytest.mark.parametrize("name", [
    "TOUCH_OF_THICKNESS", "ON_SURFACE_OF_THICKNESS", "END_OF_THICKNESS", "TRACE_OF_THICKNESS"])
def test_no_contact_length_is_a_share_of_the_thickness_base(name):
    """Touch, on-surface, end and trace limits are each piece's contact length (`core.sight`), its
    seen floor, and not a factor of the pack's thickness base."""
    assert not hasattr(tol, name)


# --- topology: the weld follows the piece size, zero-area is a shape ratio -----------
# Measured on a reference scene: the five gear piece diagonals, and the smallest face area over its
# longest edge squared there and on the chibi bar.
GEAR_PIECE_SIZES = (0.2912, 0.3171, 0.4982, 0.8731, 1.2458)
WORST_CLEAN_ZERO_AREA_RATIO = 7.99e-5  # a scrim hem strip in the reference scene; the chibi's is 1.1e-3


def test_the_weld_is_over_the_largest_gear_piece():
    # the weld distance, 1e-5, over the largest gear piece's diagonal
    assert tol._SOURCE_GEAR_PIECE_SIZE == max(GEAR_PIECE_SIZES)
    assert tol.WELD_OF_PIECE_SIZE * tol._SOURCE_GEAR_PIECE_SIZE == pytest.approx(1e-5, rel=1e-12)
    # so no gear piece welds wider than that distance
    assert all(tol.WELD_OF_PIECE_SIZE * size <= 1e-5 * (1 + 1e-12) for size in GEAR_PIECE_SIZES)


def test_the_zero_area_factor_is_pinned_under_the_worst_clean_ratio():
    assert tol.ZERO_AREA_OF_LONGEST_EDGE_SQUARED == 1e-5
    assert tol.ZERO_AREA_OF_LONGEST_EDGE_SQUARED < WORST_CLEAN_ZERO_AREA_RATIO
    # the two-way calibration keeps the clean ratio's side: 2 x the factor is still a face
    assert tol.CALIBRATION_FAIL * tol.ZERO_AREA_OF_LONGEST_EDGE_SQUARED < WORST_CLEAN_ZERO_AREA_RATIO


def test_the_retired_thickness_factors_are_gone():
    assert not hasattr(tol, "WELD_OF_THICKNESS")
    assert not hasattr(tol, "ZERO_AREA_OF_THICKNESS_SQUARED")


def test_the_take_apart_sweep_has_21_shares():
    assert tol.TAKE_APART_STEPS == 21


# --- manifest-check factors (checks 9 to 13) ---------------------------------------------------
# Measured on a reference scene. The worst clean value
# is the largest one among the holds that the scene's own contact gates accept; each factor
# keeps headroom over it. The values are pinned here so a loosened factor shows in a diff.

WORST_CLEAN_OFF_DEG = 15.5514  # a cymbal's own tilt
WORST_CLEAN_USE_DEG = 30.78  # a player's hold, in a performance clip


def test_the_angle_factors_are_pinned_above_the_worst_clean_value():
    assert tol.OFF_ANGLE_DEG == 30.0
    assert tol.USE_ANGLE_DEG == 45.0
    assert tol.OFF_ANGLE_DEG > WORST_CLEAN_OFF_DEG
    assert tol.USE_ANGLE_DEG > WORST_CLEAN_USE_DEG
    # the calibration plants must stay real angles
    assert tol.CALIBRATION_FAIL * tol.OFF_ANGLE_DEG < 180.0
    assert tol.CALIBRATION_FAIL * tol.USE_ANGLE_DEG < 180.0


# --- pose-overlap factors ------------------------------------------------------------
# Each factor is a depth over the mitt radius.


@pytest.mark.parametrize("factor, original", [
    (tol.SKIN_OF_EFFECTOR, 0.002),  # skin depth
    (tol.PRESS_OF_EFFECTOR, 0.010),  # press depth
    (tol.SEAT_OF_EFFECTOR, 0.025),  # seat depth
    (tol.SOLVE_ERR_OF_EFFECTOR, 0.036),  # the worst error of an honest key, a fret
])
def test_an_overlap_factor_gives_back_its_depth(factor, original):
    assert factor * RADIUS == pytest.approx(original, rel=1e-12)


def test_the_solve_error_is_the_worse_of_the_grip_and_fret_errors():
    assert tol.SOLVE_ERR_OF_EFFECTOR * RADIUS == pytest.approx(max(0.024, 0.036))
    # a grip zone is the radius plus that error: between a grip zone of 0.09 and a fret zone of 0.10
    assert (1 + tol.SOLVE_ERR_OF_EFFECTOR) * RADIUS == pytest.approx(0.0946, abs=1e-4)


def test_the_overlap_budgets_grow_from_skin_to_press_to_seat():
    assert 0 < tol.SKIN_OF_EFFECTOR < tol.PRESS_OF_EFFECTOR < tol.SEAT_OF_EFFECTOR


@pytest.mark.parametrize("factor, original", [
    (tol.FLOOR_OF_EFFECTOR, 0.01),  # foot tolerance
    (tol.SLIDE_OF_EFFECTOR, 0.02),  # slide tolerance
    (tol.POP_OF_EFFECTOR, 0.01),  # pop tolerance
])
def test_a_clip_hygiene_factor_gives_back_its_length(factor, original):
    assert factor * RADIUS == pytest.approx(original, rel=1e-12)


def test_the_node_tolerance_is_the_weld_tolerance_over_the_piece_size():
    # a node's position is exact up to float32 noise, which grows with the piece's extent
    assert tol.NODE_OF_PIECE_SIZE == tol.WELD_OF_PIECE_SIZE
    assert tol.FLOAT32_EPS == 2.0**-23 and tol.NODE_FLOAT32_ULPS == 4
    assert tol.NODE_OF_PIECE_SIZE * tol._SOURCE_GEAR_PIECE_SIZE == pytest.approx(1e-5, rel=1e-12)


def test_the_eye_resolves_one_minute_of_arc_and_the_header_names_the_kind():
    """A design call of this kit: normal sight resolves 1 arcmin, about 0.29 mm a metre."""
    import math

    assert tol.EYE_RESOLUTION_ARCMIN == 1.0
    assert math.tan(math.radians(tol.EYE_RESOLUTION_ARCMIN / 60.0)) == pytest.approx(2.909e-4,
                                                                                     rel=1e-3)
    assert "`*_ARCMIN`" in tol.__doc__


# --- crossing depth by part thickness (a design call of this kit) ---------------------------------------


def test_a_part_may_sink_half_the_thinner_part_and_samples_sit_a_quarter_apart():
    assert tol.CROSS_OF_PART_THICKNESS == 0.5  # a design call of this kit: half the thinner part
    assert tol.CROSS_SAMPLE_OF_THICKNESS == 0.25  # a design call of this kit
    assert not hasattr(tol, "CROSS_OF_THICKNESS")  # the pack-wide limit it replaces is gone
