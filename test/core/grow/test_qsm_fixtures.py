import json

import numpy as np
import pytest

from model_kit.core.grow.skeleton import check_skeleton, load_skeleton

from .qsm_fixtures import QSM_FILES, QSM_FIXTURES, QSM_SOURCES

MAX_BYTES = 600_000


@pytest.mark.parametrize("path", QSM_FILES, ids=lambda p: p.name)
def test_fixtures_load(path):
    sk = load_skeleton(path)
    check_skeleton(sk, path.name)
    assert len(sk.host) == 0  # a leaf-off scan
    assert sk.level_names == ("Trunk", "Order1", "Order2", "Order3", "Order4")
    # only the trunk chain stands on the soil, and it is the first limb
    assert np.flatnonzero(sk.rooted).tolist() == [0]
    assert not sk.buried.any()
    assert sk.level[0] == 0 and (sk.level[1:] > 0).all()
    # the converter's own record
    assert sk.info["source"] == QSM_SOURCES[path.name]
    assert sk.info["source_licence"] == "CC-BY 4.0"
    assert sk.info["child_bases"]["total"] == len(sk.limb_id) - 1
    assert sk.info["child_bases"]["at_parent_surface"] <= sk.info["child_bases"]["total"]
    # the trunk stands at the origin and the tree matches the height the scan reports
    assert np.allclose(sk.pts[0], 0.0)
    height_m = float(sk.pts[:, 1].max()) / sk.info["units_per_metre"]
    assert height_m == pytest.approx(sk.info["tree_height_m"], rel=0.01)


@pytest.mark.parametrize("path", QSM_FILES, ids=lambda p: p.name)
def test_fixtures_hold_no_path(path):
    # the hygiene scan bans home paths; the source is a basename and nothing else
    info = json.dumps(load_skeleton(path).info)
    assert "/" not in info.replace("tools/qsm_to_skeleton.py", "").replace("https://", "").replace(
        "doi.org/10.5281/zenodo.4557401", "")


def test_fixtures_are_credited():
    credits = (QSM_FIXTURES / "CREDITS.md").read_text()
    assert "CC-BY 4.0" in credits
    for name, source in QSM_SOURCES.items():
        assert name in credits
        assert source in credits


def test_every_fixture_file_is_listed():
    on_disk = {p.name for p in QSM_FIXTURES.glob("*.npz")}
    assert on_disk == set(QSM_SOURCES)


def test_fixtures_span_two_species_and_a_range_of_heights():
    heights = {}
    for path in QSM_FILES:
        heights.setdefault(path.name[:4], []).append(load_skeleton(path).info["tree_height_m"])
    assert sorted(heights) == ["FEXC", "FSYL"]
    for species, hs in heights.items():
        assert len(hs) == 2 and max(hs) - min(hs) > 4.0, species


@pytest.mark.parametrize("path", QSM_FILES, ids=lambda p: p.name)
def test_fixtures_are_small(path):
    assert path.stat().st_size < MAX_BYTES
