"""`left_out`: the zones, and the self-window as an arc length."""

from pathlib import Path

import numpy as np
import pytest

from model_kit.core.grow.segments import Clearance, Limb, left_out, make_limbs
from model_kit.core.grow.shape import CLEAR_SUBSTEPS, SELF_PIECE_OF_WINDOW

FIXTURES = Path(__file__).resolve().parent / "fixtures"

CLEARANCE = 1.2


def _self_pair(s0_a, s1_a, s0_b, s1_b, r):
    one = lambda x: np.array([x], dtype=float)
    return bool(left_out(np.array([True]), one(0), one(0), one(s0_a), one(s1_a), one(s0_b),
                         one(s1_b), one(r), one(r), one(0.0), one(0.0), one(0.0), one(0.0),
                         CLEARANCE)[0])


def test_adjacent_pieces_are_left_out():
    assert _self_pair(0.0, 0.01, 0.01, 0.02, 0.06)


def test_window_scales_with_radius():
    """Radius 0.06: the window is 2 x 0.06 x 2.2 = 0.264 of arc."""
    assert _self_pair(0.0, 0.01, 0.27, 0.28, 0.06)  # arc gap 0.26: left out
    assert not _self_pair(0.0, 0.01, 0.28, 0.29, 0.06)  # arc gap 0.27: measured
    assert not _self_pair(0.0, 0.01, 0.07, 0.08, 0.005)  # a thin limb: window 0.022


def test_other_limbs_are_never_in_the_window():
    one = lambda x: np.array([x], dtype=float)
    out = left_out(np.array([False]), one(0), one(0), one(0.0), one(0.01), one(0.0), one(0.01),
                   one(0.06), one(0.06), one(0.0), one(0.0), one(0.0), one(0.0), CLEARANCE)
    assert not out[0]


# ------------------------------------------------------------------------ the shared Clearance

def test_clearance_has_no_piece_cap_by_default():
    assert Clearance(1.2, CLEAR_SUBSTEPS).max_piece is None


def hairpin(radius=0.01):
    """A V of two long arms from an apex, narrow enough that the tubes overlap over the first half
    metre of each arm: far apart along the limb, a hair apart in space."""
    arm, spread = 10.0, 0.32
    return Limb("hairpin", 0, [(0.0, spread, 0.0), (arm, 0.0, 0.0), (0.0, -spread, 0.0)],
                [radius] * 3, None, 0.0)


def cut_keep(max_piece):
    limb = hairpin()
    clear = Clearance(1.2, CLEAR_SUBSTEPS, max_piece)
    clear.cut(limb)
    return limb.keep, limb.length, limb.alive


def test_a_hairpin_of_long_segments_is_missed_without_the_piece_cap():
    """Without the cap a segment is cut in 8 equal pieces; the two arms' pieces next to the apex are
    each 1.25 m long and adjacent along the limb, so the pair is left out and the overlap is not
    seen. With the cap (a quarter of the self-window, as the checks cut a limb) the arms are cut
    into pieces shorter than the window and the hairpin is cut where the arms first overlap."""
    keep, length, alive = cut_keep(None)
    assert alive and keep == pytest.approx(length)
    keep, length, alive = cut_keep(SELF_PIECE_OF_WINDOW)
    assert alive and 10.0 < keep < 10.1  # past the apex by a little over one self-window


def test_the_piece_cap_cuts_pieces_no_longer_than_the_window():
    clear = Clearance(1.2, 1, SELF_PIECE_OF_WINDOW)
    radii = np.array([0.01, 0.02])
    window = 2.0 * 0.02 * (1.0 + 1.2)  # the thicker end, grown by the clearance
    assert clear._max_len(radii)[0] == pytest.approx(window * SELF_PIECE_OF_WINDOW)
    assert Clearance(1.2, 1)._max_len(radii) is None


def test_make_limbs_cuts_a_thin_limb_by_a_thick_one():
    """The shared `make_limbs`: limbs are made level by level, so a thin twig that runs at the
    trunk stops short of it, and a limb seated past the cut is not made."""
    trunk = Limb("trunk", 0, [(0, 0, 0), (0, 1, 0)], [0.05, 0.05], None, 0.0)
    twig = Limb("twig", 1, [(0.3, 0.5, 0), (-0.3, 0.5, 0)], [0.005, 0.005], None, 0.0)
    child = Limb("child", 2, [(-0.3, 0.5, 0), (-0.5, 0.7, 0)], [0.004, 0.004], twig, 1.0)
    made = make_limbs([child, twig, trunk], Clearance(1.2, CLEAR_SUBSTEPS))
    assert [limb.kind for limb in made] == ["twig", "trunk"] and not child.alive
    safe = 0.3 - (0.05 + 0.005 + 1.2 * 0.005)  # arc length at which the twig meets the clearance
    assert safe - 0.6 / CLEAR_SUBSTEPS < twig.keep <= safe  # cut at the start of a piece
    assert twig.out_pts[-1, 0] == pytest.approx(0.3 - twig.keep)
