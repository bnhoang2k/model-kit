"""The sample budget of the surface-sampling checks (crossings, take-apart, bearing).

A part can hold two shells in one mesh, a panel sunk 0.018 mm inside a slab. Its thickness at
its centre then reads that gap, the checks space their samples a quarter of it apart, and an
uncapped gate would ask for about 2e10 points, over 100 GB. The planted fault here is that part,
on two pieces: the checks refuse it in seconds, in under 500 MB, and say which part is thin and
where, so the run is partial and never passes.
"""

import pickle
import time
import tracemalloc

import pytest

from model_kit.core import geom
from model_kit.core.checks import pieces
from model_kit.core.checks.bearing import Bearing
from model_kit.core.checks.crossings import Crossings, made_of
from model_kit.core.checks.take_apart import TakeApart
from model_kit.core.errors import TooManySamples
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import CROSS_SAMPLE_OF_THICKNESS, MAX_SURFACE_SAMPLES

from ..shapes import box
from .piece_scenes import SLIVER, make_ctx, post_on_base, sunk_panel_monitor, sunk_sheet_base

PLANTED = {"monitor": (sunk_panel_monitor, "kit:Thing display"),
           "post on base": (sunk_sheet_base, "kit:Thing base")}
MEMORY = 500e6  # bytes: what the planted fault may take at its peak
SECONDS = 20.0


def _gated(ctx, checks):
    """The run's verdict, its seconds and its peak traced memory in bytes."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        verdict = run_gate(checks, ctx, None)
        return verdict, time.perf_counter() - start, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def _sliver_part(view, name):
    return next(p for p in view.parts if p.name == name)


# --- the count and the error ---------------------------------------------------------------------


@pytest.mark.parametrize("spacing", [0.5, 0.1, 0.03])
def test_the_bound_is_never_under_the_points_made(spacing):
    for tris in (box((0, 0, 0), (1, 0.2, 0.4)), sunk_panel_monitor()[0]["display"]):
        assert geom.lattice_bound(tris, spacing) >= len(geom.surface_samples(tris, spacing))


def test_a_spacing_that_passes_the_cap_makes_no_point_and_says_how_many():
    tracemalloc.start()
    try:
        with pytest.raises(TooManySamples) as err:
            geom.surface_samples(box((0, 0, 0), (1, 1, 1)), 1e-5)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert err.value.points > MAX_SURFACE_SAMPLES
    assert "over the kit's cap" in str(err.value)
    assert peak < 1e6  # the lattice was never built


def test_the_error_crosses_a_pipe_whole():
    err = TooManySamples(3e10, 5e-6, 1e7, lead="its thickness reads 2e-05, and ", note="a sliver")
    again = pickle.loads(pickle.dumps(err))
    assert str(again) == str(err)
    assert isinstance(again, TooManySamples)


# --- the planted fault ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", PLANTED)
def test_the_planted_sliver_would_have_asked_for_billions(name):
    """The sliver's own spacing asks for over a hundred times the cap: the planted fault is the
    2026-11-16 blow-up, not a smaller one."""
    make, part = PLANTED[name]
    view = next(pieces.measured_pieces(make_ctx(*make()), {"assembly"}))
    tris = _sliver_part(view, part.split()[-1]).tris
    thickness = geom.local_thickness(tris, pieces.centroid(tris))
    assert thickness == pytest.approx(SLIVER, rel=0.25)
    assert geom.lattice_bound(tris, CROSS_SAMPLE_OF_THICKNESS * thickness) > 100 * MAX_SURFACE_SAMPLES


@pytest.mark.parametrize("name", PLANTED)
def test_the_sliver_is_refused_fast_naming_the_part_and_its_thin_spot(name):
    make, part = PLANTED[name]
    verdict, seconds, peak = _gated(make_ctx(*make()), [Crossings(), TakeApart(), Bearing()])
    assert verdict.status == "partial"
    refused = [line for line in verdict.coverage
               if isinstance(line, Unmeasured) and "over the kit's cap" in line]
    assert {"crossings", "take-apart"} <= {line.split(":")[0] for line in refused}
    for line in refused:  # bearing's item is the part that rests, and its line names the host
        assert part.split()[-1] in line and "thin spot" in line
    crossing = next(line for line in refused if line.startswith("crossings"))
    assert crossing.endswith(part) and "its thickness at its centre (" in crossing
    assert seconds < SECONDS and peak < MEMORY, (seconds, peak)


def test_a_part_under_the_cap_is_sampled_as_before():
    view = next(pieces.measured_pieces(make_ctx(post_on_base()), {"assembly"}))
    made = made_of(view)
    assert made.skipped == {}
    for solid in made.solids.values():
        assert solid.spacing == CROSS_SAMPLE_OF_THICKNESS * solid.thickness
        assert len(solid.samples) <= geom.lattice_bound(solid.tris, solid.spacing)


def test_bearing_reads_an_over_cap_seat_as_unmeasured_naming_the_pair(monkeypatch):
    """With the cap at 1000 points the monitor's seat on its stand passes it, and the rule reads
    Unmeasured. (Crossings keeps its own reference to the real budget.)"""
    real = geom.check_sample_budget
    monkeypatch.setattr(geom, "check_sample_budget", lambda tris, spacing: real(tris, spacing, 1000))
    parts, specs = sunk_panel_monitor(gap=0.002)  # a panel 2 mm into the slab: no sliver
    verdict, _, _ = _gated(make_ctx(parts, specs), [Bearing()])
    lines = [line for line in verdict.coverage
             if isinstance(line, Unmeasured) and "over the kit's cap" in line]
    assert lines and all("on stand" in line and line.endswith("kit:Thing display")
                         for line in lines)
