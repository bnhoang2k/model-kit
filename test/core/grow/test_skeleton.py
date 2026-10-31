import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.grow import buds, chains, light
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import (
    BRANCH,
    Skeleton,
    load_skeleton,
    same_skeleton,
    save_skeleton,
)

TINY_SPIRAL = Path(__file__).parent / "fixtures" / "tiny-spiral.rules.json"
UPM = 0.5


@pytest.fixture(scope="module")
def seed1() -> Skeleton:
    return grow_buds(load_rules(TINY_SPIRAL, kit=KIT_RULES), piece="Tree", height_m=1.5, seed=1,
                     lite=False, units_per_metre=UPM)


ALLOWED_FLOATS = {0.0, 0.5, 1.0, 2.0}
# The bud grower's modules read their numbers from the kit file, the species file or shape.py.
SPECIES_FREE = [buds, light, chains]


@pytest.mark.parametrize("module", SPECIES_FREE, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_no_species_number_in_grower(module):
    tree_ = ast.parse(Path(module.__file__).read_text())
    bad = []
    for node in ast.walk(tree_):
        if isinstance(node, ast.Constant) and not isinstance(node.value, bool | str | bytes):
            v = node.value
            if v is None or v is Ellipsis:
                continue
            species_float = isinstance(v, float) and v not in ALLOWED_FLOATS
            if species_float or (isinstance(v, int) and v > 3):
                bad.append((node.lineno, v))
    assert bad == []


def test_skeleton_round_trip(tmp_path, seed1):
    path = tmp_path / "Redbud.skeleton.npz"
    save_skeleton(seed1, path)
    back = load_skeleton(path)
    assert same_skeleton(seed1, back)
    assert back.info == seed1.info
    i = int(np.flatnonzero(seed1.level == BRANCH)[0])
    p, r, tan = back.at(i, 0.5)
    assert p.shape == (3,) and tan.shape == (3,) and r > 0


def test_bad_parent_order_raises(tmp_path, seed1):
    parent = seed1.parent.copy()
    parent[1] = 5
    bad = replace(seed1, parent=parent)
    with pytest.raises(ManifestError, match="parent"):
        save_skeleton(bad, tmp_path / "bad.skeleton.npz")
    good = tmp_path / "good.skeleton.npz"
    save_skeleton(seed1, good)
    with np.load(good) as z:
        arrays = dict(z)
    arrays["parent"] = parent
    np.savez_compressed(tmp_path / "edited.skeleton.npz", **arrays)
    with pytest.raises(ManifestError, match="parent"):
        load_skeleton(tmp_path / "edited.skeleton.npz")


def test_bad_shape_raises(tmp_path, seed1):
    bad = replace(seed1, radii=seed1.radii[:-1])
    with pytest.raises(ManifestError, match="radii"):
        save_skeleton(bad, tmp_path / "bad.skeleton.npz")


def _with(sk, name, index, value):
    data = np.array(getattr(sk, name), dtype=float)
    data[index] = value
    return replace(sk, **{name: data})


@pytest.mark.parametrize("name, index, value, words", [
    ("pts", (7, 1), np.nan, r"pts: index \(7, 1\) is nan"),
    ("radii", 5, np.inf, r"radii: index 5 is inf"),
    ("radii", 3, 0.0, r"radii: index 3 is 0\.0; it must be over 0"),
    ("radii", 2, -0.01, r"radii: index 2 is -0\.01"),
    ("length", 4, 0.0, r"length: index 4 is 0\.0; it must be over 0"),
])
def test_bad_numbers_raise_naming_the_field(tmp_path, seed1, name, index, value, words):
    bad = _with(seed1, name, index, value)
    with pytest.raises(ManifestError, match=words):
        save_skeleton(bad, tmp_path / "bad.skeleton.npz")
    good = tmp_path / "good.skeleton.npz"
    save_skeleton(seed1, good)
    with np.load(good) as z:
        arrays = dict(z)
    arrays[name] = np.asarray(getattr(bad, name))
    np.savez_compressed(tmp_path / "edited.skeleton.npz", **arrays)
    with pytest.raises(ManifestError, match=words):
        load_skeleton(tmp_path / "edited.skeleton.npz")


def test_a_non_finite_scalar_raises_naming_the_field(tmp_path, seed1):
    with pytest.raises(ManifestError, match="fold: is nan"):
        save_skeleton(replace(seed1, fold=float("nan")), tmp_path / "bad.skeleton.npz")


@pytest.mark.parametrize("name", ["mid", "across", "normal"])
def test_non_unit_leaf_frame_raises(tmp_path, seed1, name):
    frame = np.array(getattr(seed1, name))
    frame[0] *= 3.0
    with pytest.raises(ManifestError, match=f"{name}: leaf 0 has length 3"):
        save_skeleton(replace(seed1, **{name: frame}), tmp_path / "bad.skeleton.npz")


def test_nearest_matches_at(seed1):
    i = int(np.flatnonzero(seed1.level == seed1.level.max())[-1])  # a limb of the top level
    pts = np.array([seed1.at(i, t)[0] for t in (0.0, 0.3, 0.7, 1.0)])
    dist, r = seed1.nearest(i, pts)
    assert np.allclose(dist, 0.0, atol=1e-12)
    assert np.allclose(r, [seed1.at(i, t)[1] for t in (0.0, 0.3, 0.7, 1.0)])


def test_numpy_info_saves_as_numbers(tmp_path, seed1):
    sk = replace(seed1, info=seed1.info | {"W": np.float64(1.25), "leaders": np.int64(3)})
    save_skeleton(sk, tmp_path / "np.skeleton.npz")
    back = load_skeleton(tmp_path / "np.skeleton.npz")
    assert back.info["W"] == 1.25 and type(back.info["leaders"]) is int


def test_float_parent_raises(tmp_path, seed1):
    good = tmp_path / "good.skeleton.npz"
    save_skeleton(seed1, good)
    with np.load(good) as z:
        arrays = dict(z)
    arrays["parent"] = arrays["parent"].astype(float)
    np.savez_compressed(tmp_path / "float.skeleton.npz", **arrays)
    with pytest.raises(ManifestError, match="parent: dtype"):
        load_skeleton(tmp_path / "float.skeleton.npz")
