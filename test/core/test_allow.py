from model_kit.core.allow import Allowlist
from model_kit.core.metagate import Finding


def _allow(tmp_path, text):
    p = tmp_path / "allow.txt"
    p.write_text(text)
    return Allowlist.load(p)


def test_allow_line_without_reason_fails(tmp_path):
    al = _allow(tmp_path, "reach | arm_* | \n")
    kept, allowed = al.apply([Finding("reach", "arm_l", "too far")])
    assert kept == [Finding("reach", "arm_l", "too far")]
    assert allowed == []
    assert [p.detail for p in al.problems()] == ["allowlist line has no reason"]


def test_stale_allow_line_fails(tmp_path):
    al = _allow(tmp_path, "reach | arm_* | the rig is a known outlier\n")
    kept, allowed = al.apply([Finding("held", "cup", "off")])
    assert (kept, allowed) == ([Finding("held", "cup", "off")], [])
    [problem] = al.problems()
    assert problem.detail == 'stale: "reach | arm_*" matches nothing in this build; delete it'


def test_first_match_wins(tmp_path):
    al = _allow(tmp_path, "reach | arm_l | first reason\nreach | arm_* | second reason\n")
    kept, allowed = al.apply([Finding("reach", "arm_l", "x")])
    assert kept == []
    assert [(f.item, reason) for f, reason in allowed] == [("arm_l", "first reason")]
    # The second line matched nothing, because the first one won.
    assert [p.detail for p in al.problems()] == [
        'stale: "reach | arm_*" matches nothing in this build; delete it']


def test_blank_and_comment_lines_are_skipped(tmp_path):
    al = _allow(tmp_path, "# a note\n\nreach | arm_l | known\n")
    kept, allowed = al.apply([Finding("reach", "arm_l", "x")])
    assert kept == [] and len(allowed) == 1
    assert al.problems() == []


def test_match_is_case_sensitive_glob(tmp_path):
    al = _allow(tmp_path, "reach | Arm_* | known\n")
    kept, _ = al.apply([Finding("reach", "arm_l", "x")])
    assert len(kept) == 1


def test_malformed_line_is_a_problem(tmp_path):
    al = _allow(tmp_path, "just one field\n")
    assert [p.detail for p in al.problems()] == ["allowlist line is not `check | item | reason`"]


def test_only_lines_for_checks_that_ran_are_judged_stale(tmp_path):
    al = _allow(tmp_path, "must-touch | * | judged elsewhere\nreach | arm_* | outlier\n"
                          "* | anything | a wildcard check\n")
    al.apply([])
    stale = [p.item for p in al.problems(ran=["reach"])]
    assert stale == ["reach | arm_*", "* | anything"]
    assert [p.item for p in al.problems()] == [
        "must-touch | *", "reach | arm_*", "* | anything"]  # no `ran`: every line is judged
