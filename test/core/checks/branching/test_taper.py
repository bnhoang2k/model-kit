import dataclasses

import numpy as np
import pytest

from model_kit.core.checks.branching.taper import Taper, fit, fits, is_measured
from model_kit.core.grow.skeleton import load_skeleton
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import RELIABLE_RADIUS_M, TAPER_PLANT_RISE

from ...grow.qsm_fixtures import QSM_FILES
from .branching_scene import (
    UPM,
    Limb,
    grown_ctx,
    item,
    limb_index,
    line,
    make_skeleton,
    plants_caught,
    small_tree,
)
from .tiny_trees import PIECES, tiny_tree

CHECK = Taper()
SOURCE = {"source_doi": "10.5281/zenodo.4557401"}  # what a converted scan's `info` carries


def with_radii(sk, name, radii):
    """A copy of sk with one limb's radii (in model units) replaced."""
    i = limb_index(sk, name)
    new = np.array(sk.radii)
    new[sk.start[i]:sk.start[i + 1]] = radii
    return dataclasses.replace(sk, radii=new)


def one_limb(radii_m, measured=False):
    """A trunk standing on the soil with the given radii (metres), evenly spaced over 1 m."""
    n = len(radii_m) - 1
    sk = make_skeleton([Limb("trunk", 0, -1, 0.0, line((0, 0, 0), (0, 1, 0), 1.0 * UPM, n),
                             np.array(radii_m) * UPM, rooted=True)], [])
    return dataclasses.replace(sk, info={**sk.info, **(SOURCE if measured else {})})


# --- the measured trees ------------------------------------------------------------------------


def test_taper_quiet_on_measured_trees():
    judged = []
    for path in QSM_FILES:
        sk = load_skeleton(path)
        assert is_measured(sk)
        ctx = grown_ctx(sk)
        assert CHECK.run(ctx) == []
        assert not [c for c in CHECK.coverage(ctx) if isinstance(c, Unmeasured)]
        assert run_gate([CHECK], ctx, None).status == "passed"
        slopes = [f.slope for f in fits(sk, UPM).values()]
        assert slopes and max(slopes) <= 0.0  # the quiet case is not empty
        judged.append(len(slopes))
    # The sweep: FSYL_15, FSYL_13, FEXC_5, FEXC_14 judge 1, 4, 1 and 4 limbs at 0.05 m.
    assert judged == [1, 4, 1, 4]


def test_measured_tree_is_judged_only_on_reliable_points():
    # Four reliable points that thicken, then a thin tail that falls away to nothing.
    thin_tail = [0.050, 0.052, 0.054, 0.056, 0.030, 0.020, 0.010, 0.005]
    # Read whole, the tail makes the limb fall: a grown skeleton counts every point.
    assert fit(one_limb(thin_tail), 0, UPM).slope < 0.0
    assert CHECK.run(grown_ctx(one_limb(thin_tail))) == []
    # Read as a scan, the four points under the reliability radius are left out and it rises.
    measured = one_limb(thin_tail, measured=True)
    f = fit(measured, 0, UPM)
    assert f is not None
    assert f.points == 4
    assert f.slope > 0.0
    assert [x.item for x in CHECK.run(grown_ctx(measured))] == [item("trunk")]


def test_measured_limb_with_a_thin_base_or_few_points_is_not_judged():
    thin_base = one_limb([RELIABLE_RADIUS_M * 0.9, 0.04, 0.045, 0.05, 0.06, 0.07], measured=True)
    assert fit(thin_base, 0, UPM) is None
    few = one_limb([0.08, 0.075, 0.07, 0.01, 0.02, 0.03], measured=True)  # only 3 reliable points
    assert fit(few, 0, UPM) is None
    for sk in (thin_base, few):
        assert CHECK.run(grown_ctx(sk)) == []
        assert any("not judged (1 of 1 limbs" in c for c in CHECK.coverage(grown_ctx(sk)))


def test_the_reliability_radius_is_in_metres_whatever_the_units():
    # The same 0.06 m limb in a scene of 1 unit per metre and of 0.5: radius 0.06 and 0.03 units.
    for upm in (1.0, UPM):
        sk = one_limb([0.06, 0.059, 0.058, 0.057], measured=True)
        sk = dataclasses.replace(sk, pts=sk.pts / UPM * upm, radii=sk.radii / UPM * upm)
        assert fit(sk, 0, upm).points == 4


# --- a grown skeleton --------------------------------------------------------------------------


def test_a_grown_skeleton_judges_every_limb_with_enough_points():
    sk = small_tree()
    assert CHECK.run(grown_ctx(sk)) == []
    assert set(fits(sk, UPM)) == set(range(len(sk.limb_id)))  # every limb, thin twigs included
    short = one_limb([0.05, 0.06, 0.07])  # three points, rising: not enough to fit
    assert fit(short, 0, UPM) is None
    assert CHECK.run(grown_ctx(short)) == []


def test_equal_radii_never_rise():
    # A radius floor makes whole limbs equal; a fit that rounds must not call them thickening.
    sk = one_limb([0.0024] * 17)
    assert fit(sk, 0, UPM).slope == 0.0
    assert CHECK.run(grown_ctx(sk)) == []


def test_taper_catches_a_thickening_limb():
    sk = small_tree()
    i = limb_index(sk, "leader.0")
    base = float(sk.limb_radii(i)[0])
    grow30 = base * (1.0 + 0.3 * np.linspace(0.0, 1.0, len(sk.limb_radii(i))))
    thick = with_radii(sk, "leader.0", grow30)
    found = CHECK.run(grown_ctx(thick))
    assert [f.item for f in found] == [item("leader.0")]
    assert "rises" in found[0].detail
    assert run_gate([CHECK], grown_ctx(thick), None).status == "failed"
    assert CHECK.run(grown_ctx(thick), only={item("trunk")}) == []  # `only` names the items run


def test_a_thickening_that_does_not_sum_to_a_trend_is_left_alone():
    # One thick step in a falling limb is noise (measured trees grow in 8 to 9% of steps).
    sk = one_limb([0.10, 0.09, 0.095, 0.07, 0.06, 0.05, 0.04])
    assert fit(sk, 0, UPM).slope < 0.0
    assert CHECK.run(grown_ctx(sk)) == []


# --- the plants --------------------------------------------------------------------------------


@pytest.mark.parametrize(("path", "seed"), PIECES)
def test_plants_on_two_species(path, seed):
    ctx = grown_ctx(tiny_tree(path, seed))
    assert CHECK.run(ctx) == []  # the piece passes unplanted first
    caught = plants_caught(CHECK, ctx)
    assert [label.split(":")[0] for label, _ in caught] == ["thickening"]
    assert all(ok for _, ok in caught), caught
    assert run_gate([CHECK], ctx, None).status == "passed"  # run_gate runs every plant too


@pytest.mark.parametrize("path", QSM_FILES)
def test_the_plant_also_reaches_a_measured_tree(path):
    ctx = grown_ctx(load_skeleton(path))
    caught = plants_caught(CHECK, ctx)
    assert len(caught) == 1
    assert all(ok for _, ok in caught)


def test_the_plant_raises_the_radius_by_its_rise():
    ctx = grown_ctx(tiny_tree(PIECES[0].values[0], 1))
    plant = CHECK.plants(ctx)[0].candidates[0]
    moved = next(iter(plant.ctx.skeletons.values()))
    i = limb_index(moved, plant.item.split(" ", 1)[1])
    radii = moved.limb_radii(i)
    assert radii[-1] / radii[0] == pytest.approx(1.0 + TAPER_PLANT_RISE)


def test_a_scene_with_no_skeleton_is_unmeasured():
    ctx = grown_ctx(None, listed_from=small_tree())
    assert CHECK.run(ctx) == []
    assert CHECK.plants(ctx) == []
    assert any(isinstance(c, Unmeasured) for c in CHECK.coverage(ctx))
