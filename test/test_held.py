"""The `held` marker: a test that needs a file that not every copy of the repository has skips
where the file is missing, and runs where it is present."""

import pytest


@pytest.mark.held("no/such/held/path")
def test_a_missing_held_path_skips():
    raise AssertionError("the held marker should have skipped this test")


@pytest.mark.held("pyproject.toml")
def test_a_present_held_path_runs():
    pass
