import pytest

from model_kit.core.allow import Allowlist
from model_kit.core.checks.bake_files import BakeFiles
from model_kit.core.checks.bearing import Bearing
from model_kit.core.checks.bend_quality import BendQuality
from model_kit.core.checks.clearance import Clearance
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.detail import Detail
from model_kit.core.checks.inset_seat import InsetSeat
from model_kit.core.checks.lightmap_density import LightmapDensity
from model_kit.core.checks.lightmap_overlap import LightmapOverlap
from model_kit.core.checks.markings import Markings
from model_kit.core.checks.markings_flush import MarkingsFlush
from model_kit.core.checks.mirror_exact import MirrorExact
from model_kit.core.checks.near_crease import NearCrease
from model_kit.core.checks.seam_jump import SeamJump
from model_kit.core.checks.seat_fit import SeatFit
from model_kit.core.checks.stress_poses import StressPoses
from model_kit.core.errors import AnchorMissing, BlindCheck, ManifestError
from model_kit.core.metagate import (
    Finding,
    Plant,
    PlantUnit,
    Timing,
    Unmeasured,
    Verdict,
    emit,
    exit_code_for,
    run_gate,
)
from model_kit.core.scope import Scope
from model_kit.core.tolerances import CALIBRATION_FAIL

from .checks.piece_scenes import CROSS, two_pieces

# The checks that must cite a principle, as classes. Each new check adds its class here.
CITED_CHECKS = (
    StressPoses,
    MirrorExact,
    SeatFit,
    Clearance,
    Markings,
    NearCrease,
    BendQuality,
    InsetSeat,
    Bearing,
    Detail,
    MarkingsFlush,
    Crossings,
    BakeFiles,
    LightmapOverlap,
    LightmapDensity,
    SeamJump,
)


def unit(label, item, ctx, **kw):
    """A unit of one candidate plant."""
    return PlantUnit(item, (Plant(label, item, ctx, **kw),))


class Sees:
    """A check that fails every item in ctx (a list of bad item names)."""
    name = "sees"

    def plants(self, ctx):
        return [unit("bad item", "planted", [*ctx, "planted"])]

    def run(self, ctx, only=None, samples=frozenset()):
        return [Finding(self.name, i, "bad") for i in ctx if only is None or i in only]

    def coverage(self, ctx):
        return []


class AlwaysPass(Sees):
    name = "always_pass"

    def run(self, ctx, only=None, samples=frozenset()):
        return []


class WrongItem(Sees):
    name = "wrong_item"

    def run(self, ctx, only=None, samples=frozenset()):
        return [Finding(self.name, "somewhere_else", "bad")]


class Skips(Sees):
    name = "skips"

    def coverage(self, ctx):
        return ["skips: left out cup (no surface)"]


def test_always_pass_check_stops_the_run():
    with pytest.raises(BlindCheck, match="always_pass: planted") as e:
        run_gate([AlwaysPass()], [], None)
    assert exit_code_for(e.value) == 2


def test_finding_on_other_item_does_not_catch_plant():
    with pytest.raises(BlindCheck, match="wrong_item: planted"):
        run_gate([WrongItem()], [], None)


def test_real_checks_run_before_any_plant(capsys):
    calls = []

    class Spy(Sees):
        name = "spy"

        def run(self, ctx, only=None, samples=frozenset()):
            calls.append(list(ctx))
            return super().run(ctx, only, samples)

    with pytest.raises(BlindCheck):
        run_gate([Spy(), AlwaysPass()], ["real"], None)
    # Spy's real run, then its plant run; AlwaysPass is judged after every real run is in.
    assert calls == [["real"], ["real", "planted"]]


def test_exit_code_for_start_errors():
    for exc in (ManifestError("x"), AnchorMissing("x"), BlindCheck("x")):
        assert exit_code_for(exc) == 2


def test_no_checks_is_partial(capsys):
    assert emit(run_gate([], None, None)) == 1
    out = capsys.readouterr()
    assert out.out == "partial\n"
    assert "no checks ran" in out.err


def test_one_finding_fails(capsys):
    assert emit(run_gate([Sees()], ["cup"], None)) == 1
    out = capsys.readouterr()
    assert out.out == "failed: 1\n"
    assert "sees | cup | bad" in out.err


def test_clean_run_passes(capsys):
    assert emit(run_gate([Sees()], [], None)) == 0
    out = capsys.readouterr()
    assert out.out == "passed\n"
    assert [line for line in out.err.splitlines() if not line.startswith("time: ")] == []


def test_coverage_goes_to_stderr_and_does_not_fail(capsys):
    assert emit(run_gate([Skips()], [], None)) == 0
    out = capsys.readouterr()
    assert out.out == "passed\n"
    assert "skips: left out cup (no surface)" in out.err


def test_allowed_finding_is_reported_not_counted(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_text("sees | cup | known outlier\n")
    assert emit(run_gate([Sees()], ["cup"], Allowlist.load(p))) == 0
    out = capsys.readouterr()
    assert out.out == "passed\n"
    assert "allowed" in out.err and "known outlier" in out.err


def test_allowlist_problems_fail_the_run(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_text("sees | nothing | stale reason\nsees | cup | \n")
    assert emit(run_gate([Sees()], ["cup"], Allowlist.load(p))) == 1
    out = capsys.readouterr()
    assert out.out == "failed: 3\n"  # the cup finding, the no-reason line, the stale line
    assert "allowlist line has no reason" in out.err
    assert "stale:" in out.err


class Counts:
    """A check over named items. ctx is {item: bad?}; each plant makes one item bad.

    `measured` records each item a run looks at, so a test can see what each run measured.
    """
    name = "counts"

    def __init__(self):
        self.measured: list[list[str]] = []

    def plants(self, ctx):
        return [unit(f"bad {item}", item, {**ctx, item: True}) for item in ctx]

    def run(self, ctx, only=None, samples=frozenset()):
        seen = [item for item in ctx if only is None or item in only]
        self.measured.append(seen)
        return [Finding(self.name, item, "bad") for item in seen if ctx[item]]

    def coverage(self, ctx):
        return []


def test_a_plant_scope_defaults_to_its_item():
    assert Plant("x", "cup", None).scope == frozenset({"cup"})
    assert Plant("x", "cup", None, scope=frozenset({"plate"})).scope == frozenset({"cup", "plate"})


def test_each_plant_measures_only_its_scope_and_the_real_run_measures_every_item():
    items = {f"hold {i}": False for i in range(5)}
    check = Counts()
    assert run_gate([check], items, None).status == "passed"
    real_run, *plant_runs = check.measured
    assert plant_runs == [[item] for item in items]  # one item each, in plant order
    assert real_run == list(items)
    assert sum(map(len, check.measured)) == 2 * len(items)  # not len(items) ** 2 + len(items)


def test_a_plant_whose_fault_shows_only_on_another_item_is_blind():
    class Misplaced(Counts):
        name = "misplaced"

        def plants(self, ctx):  # the fault lands on `b`, but the plant claims `a`
            return [unit("bad a", "a", {**ctx, "b": True})]

    with pytest.raises(BlindCheck, match="misplaced: a"):
        run_gate([Misplaced()], {"a": False, "b": False}, None)
    # Even a scope that takes in `b` stays blind: the finding must be on the plant's item.
    class Widened(Misplaced):
        name = "widened"

        def plants(self, ctx):
            return [unit("bad a", "a", {**ctx, "b": True}, scope=frozenset({"b"}))]

    with pytest.raises(BlindCheck, match="widened: a"):
        run_gate([Widened()], {"a": False, "b": False}, None)


class Unsampled(Sees):
    """A check that could not measure one item: its clip has no samples."""
    name = "unsampled"

    def coverage(self, ctx):
        return [Unmeasured("unsampled: skipped (no samples): clip"), "unsampled: keyed: other"]


def test_an_unmeasured_item_makes_the_run_partial(capsys):
    verdict = run_gate([Unsampled()], [], None)
    assert verdict.status == "partial"
    assert emit(verdict) == 1
    out = capsys.readouterr()
    assert out.out == "partial\n"
    assert "unsampled: skipped (no samples): clip" in out.err


def test_partial_wins_over_failed(capsys):
    verdict = run_gate([Unsampled()], ["cup"], None)
    assert verdict.status == "partial" and verdict.failed == 1
    assert emit(verdict) == 1
    out = capsys.readouterr()
    assert out.out == "partial\n" and "unsampled | cup | bad" in out.err


def test_a_plain_coverage_line_is_an_unmeasured_line_only_when_marked():
    assert Unmeasured("x") == "x" and isinstance(Unmeasured("x"), str)
    assert run_gate([Skips()], [], None).status == "passed"


def test_allowlist_lines_for_checks_that_did_not_run_are_not_stale(tmp_path, capsys):
    p = tmp_path / "a.txt"
    p.write_text("other | cup | judged by a check this run skips\nsees | cup | known outlier\n")
    assert emit(run_gate([Sees()], ["cup"], Allowlist.load(p))) == 0
    assert capsys.readouterr().out == "passed\n"
    p.write_text("s* | nothing | a pattern that matches the check that ran\n")
    assert emit(run_gate([Sees()], [], Allowlist.load(p))) == 1
    assert "stale:" in capsys.readouterr().err


def test_a_plant_names_its_samples_apart_from_its_items():
    assert Plant("x", "cup", None).samples == frozenset()
    assert Plant("x", "cup", None, samples=frozenset({3})).samples == frozenset({3})
    assert Plant("x", "cup", None, samples=[3, 3, 5]).samples == frozenset({3, 5})
    assert Plant("x", "cup", None, samples={7}).samples == frozenset({7})

    class Reads(Counts):
        name = "reads"

        def __init__(self):
            super().__init__()
            self.sampled: list[frozenset[int]] = []

        def plants(self, ctx):
            return [unit("bad a", "a", {**ctx, "a": True}, samples=frozenset({4}))]

        def run(self, ctx, only=None, samples=frozenset()):
            self.sampled.append(samples)
            return super().run(ctx, only, samples)

    check = Reads()
    assert run_gate([check], {"a": False}, None).status == "passed"
    assert check.sampled == [frozenset(), frozenset({4})]  # the real run, then the plant's run


def test_the_newer_checks_cite_a_principle():
    """A newer check states the fact it stands on in one sentence, and where the fact is from."""
    for cls in CITED_CHECKS:
        check, name = cls(), cls.name
        assert check.principle.strip(), f"{name} has no principle"
        assert check.principle.endswith(".") and ". " not in check.principle, \
            f"{name}: not one sentence"
        assert check.source.strip(), f"{name} has no source"
        assert "http" in check.source or "et al." in check.source, \
            f"{name}: the source cites nothing"


def test_principle_and_source_are_optional_on_a_check():
    """A check written earlier has neither, and still runs in the meta-gate."""
    assert run_gate([Sees()], ["real"], None).status == "failed"
    assert not hasattr(Sees(), "principle") and not hasattr(Sees(), "source")


class Units(Counts):
    """A check whose `plants` yields the units a test gives it. Bad items are those in ctx."""
    name = "units"

    def __init__(self, units):
        super().__init__()
        self._units = units

    def plants(self, ctx):
        return self._units


def _plant(label, item, bad):
    return Plant(label, item, {item: bad})


def test_unit_with_no_caught_candidate_raises():
    unit = PlantUnit("a", (_plant("first", "a", False), _plant("second", "a", False)))
    with pytest.raises(BlindCheck, match="units: a$") as e:
        run_gate([Units([unit])], {"a": False}, None)
    assert exit_code_for(e.value) == 2


def test_unit_with_no_candidates_and_no_reason_raises():
    with pytest.raises(BlindCheck, match="units: a$"):
        run_gate([Units([PlantUnit("a", ())])], {"a": False}, None)


def test_unit_with_a_reason_and_candidates_is_refused():
    with pytest.raises(ValueError, match="not-applicable"):
        PlantUnit("a", (_plant("p", "a", True),), not_applicable="no geometry")


def test_unit_refuses_a_candidate_for_another_item():
    with pytest.raises(ValueError, match="another item"):
        PlantUnit("a", (_plant("p", "b", True),))


def test_first_caught_candidate_wins_and_later_ones_do_not_run():
    check = Units([PlantUnit("a", (
        _plant("blind", "a", False), _plant("caught", "a", True), _plant("never", "a", True)))])
    assert run_gate([check], {"a": False}, None).status == "passed"
    # the real run, then candidate 1 (blind) and candidate 2 (caught); candidate 3 never ran
    assert len(check.measured) == 3


def test_unit_whose_item_already_fails_is_skipped():
    check = Units([PlantUnit("a", (_plant("blind", "a", False),))])
    verdict = run_gate([check], {"a": True}, None)  # `a` fails in the real run
    assert verdict.status == "failed" and verdict.failed == 1
    assert len(check.measured) == 1  # the real run only; the plant never ran


def test_not_applicable_unit_prints_one_coverage_line():
    check = Units([PlantUnit("a", (), not_applicable="no contact")])
    verdict = run_gate([check], {"a": False}, None)
    assert verdict.status == "passed"
    assert verdict.coverage == ["units: not applicable (no contact): a"]
    assert len(check.measured) == 1  # no plant ran


def test_each_check_is_timed_in_three_phases():
    ticks = iter(range(100))
    verdict = run_gate([Sees(), Skips()], [], None, clock=lambda: float(next(ticks)))
    assert [(t.check, t.phase) for t in verdict.timings] == [
        ("sees", "real"), ("skips", "real"),
        ("sees", "coverage"), ("sees", "plants"),
        ("skips", "coverage"), ("skips", "plants")]
    assert all(t.seconds == 1.0 for t in verdict.timings)


def test_a_run_with_no_checks_has_no_timings():
    assert run_gate([], [], None).timings == []


def test_emit_prints_one_time_line_per_check_to_stderr(capsys):
    emit(Verdict("passed", 0, [], [], timings=[Timing("reach", "real", 1.25),
                                               Timing("reach", "coverage", 0.0),
                                               Timing("reach", "plants", 2.5)]))
    out = capsys.readouterr()
    assert "time: reach | real 1.2 s | coverage 0.0 s | plants 2.5 s" in out.err
    assert out.out == "passed\n"


def test_every_blind_unit_is_collected_and_the_real_findings_ride_along():
    with pytest.raises(BlindCheck) as e:
        run_gate([Sees(), AlwaysPass(), WrongItem()], ["a"], None)
    assert e.value.blind == ("always_pass: planted", "wrong_item: planted")
    assert str(e.value) == "always_pass: planted; wrong_item: planted"
    assert e.value.verdict.lines == ["sees | a | bad", "wrong_item | somewhere_else | bad"]


def test_failures_are_the_last_stderr_lines(capsys):
    emit(Verdict("failed", 1, ["reach | a | too far"], ["reach: skipped x"], ["allowed: y"]))
    err = capsys.readouterr().err.splitlines()
    assert err == ["coverage: reach: skipped x", "allowed: y", "reach | a | too far"]


def test_on_timing_hears_each_phase_as_it_ends():
    ticks = iter(range(100))
    heard = []
    verdict = run_gate([Sees(), Skips()], [], None, clock=lambda: float(next(ticks)),
                       on_timing=heard.append)
    assert heard == verdict.timings


def test_a_scoped_run_is_partial_with_the_scoped_line_first():
    ctx = two_pieces()
    verdict = run_gate([Crossings()], ctx, None, scope=Scope(frozenset({"kit:Thing"})))
    assert verdict.status == "partial" and verdict.failed == 0
    assert verdict.coverage[0].startswith("gate: scoped run (pieces: kit:Thing)")


def test_a_scoped_run_with_faults_is_still_partial_and_shows_them(capsys):
    ctx = two_pieces(sink=CALIBRATION_FAIL * CROSS)
    verdict = run_gate([Crossings()], ctx, None, scope=Scope(frozenset({"kit:Thing"})))
    assert verdict.status == "partial"
    assert [line.split(" | ")[1] for line in verdict.lines] == ["kit:Thing base x post"]
    assert emit(verdict) == 1
    assert capsys.readouterr().out.strip() == "partial"


def test_a_scoped_run_measures_the_named_pieces_as_the_full_run_does():
    ctx = two_pieces(sink=CALIBRATION_FAIL * CROSS)
    full = run_gate([Crossings()], ctx, None)
    scoped = run_gate([Crossings()], ctx, None, scope=Scope(frozenset({"kit:Thing"})))
    assert full.failed == 2
    assert scoped.lines == [line for line in full.lines if " | kit:Thing " in line]


def test_a_scoped_run_hands_the_allowlist_only_the_scene_checks(tmp_path):
    path = tmp_path / "a.allow"
    path.write_text("crossings | kit:Other base x post | pinned in its socket\n")
    ctx = two_pieces()
    scope = Scope(frozenset({"kit:Thing"}))
    # A piece check's stale line is not judged in a scoped run; a full run judges it.
    assert run_gate([Crossings()], ctx, Allowlist.load(path), scope=scope).failed == 0
    assert run_gate([Crossings()], ctx, Allowlist.load(path)).failed == 1


def test_a_wildcard_line_that_matches_a_piece_check_is_not_stale_in_a_scoped_run(tmp_path):
    path = tmp_path / "a.allow"
    path.write_text("c* | kit:Other base x post | pinned in its socket\n")
    ctx = two_pieces(sink=CALIBRATION_FAIL * CROSS)
    checks = [Crossings(), Clearance()]
    scoped = run_gate(checks, ctx, Allowlist.load(path), scope=Scope(frozenset({"kit:Thing"})))
    assert not any("stale" in line for line in scoped.lines)
    full = run_gate(checks, ctx, Allowlist.load(path))
    assert not any("stale" in line for line in full.lines)
    assert any("kit:Other base x post" in line for line in full.allowed)
