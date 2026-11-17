"""Balls and cards: leaves gathered into balls, each covered by leaf cards on real leaf centres."""

from __future__ import annotations

import dataclasses
import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError
from model_kit.core.grow.card_image import coverage, paint_cells
from model_kit.core.grow.cards import (
    HIDDEN,
    CardDials,
    ball_cards,
    balls,
    card_leaves,
    card_parts,
    card_shown,
    card_wood,
    leaf_centres,
)
from model_kit.core.grow.mesh import (
    LeafLean,
    grown_parts,
    limb_tris,
    round_sides,
    tube_sides,
    tube_tris,
)
from model_kit.core.grow.skeleton import Skeleton, load_skeleton
from model_kit.core.sight import Sight
from model_kit.core.tolerances import EYE_RESOLUTION_ARCMIN

from ..checks.branching.tiny_trees import SPECIES, tiny_tree

REDBUD = (Path(__file__).resolve().parents[3]
          / "bars/02-redbud/scene/pieces/redbud/Redbud.skeleton.npz")
UPM = 0.5
DIALS = CardDials(ball_m=0.5, normal_mix=0.5, sky=3.0)
LEAN = LeafLean(toward_out=0.8, centre_drop=0.35)


@cache
def redbud() -> Skeleton:
    return load_skeleton(REDBUD)


@cache
def tiny() -> Skeleton:
    return tiny_tree(SPECIES[0], 1)


def _cards(sk: Skeleton, dials: CardDials = DIALS):
    return card_leaves(sk, dials, UPM, LEAN)


def test_sprays_join_within_ball_r():
    sk, r = redbud(), DIALS.ball_m * UPM
    made = balls(sk, r)
    assert sorted(np.concatenate([b.leaves for b in made])) == list(range(len(sk.length)))
    # a ball starts only where no earlier ball's seed lay within r: a looser radius gives fewer
    assert len(balls(sk, 2 * r)) < len(made) < len(sk.length)
    assert len(balls(sk, 1e9)) == 1  # every spray is within reach of the first
    assert len(balls(sk, 0.0)) >= len(made)  # no radius joins none


def test_one_leaf_ball():
    sk = tiny()
    first = balls(sk, 0.0)  # no radius joins anything: one ball per spray
    sizes = sorted(len(b.leaves) for b in first)
    assert sizes[0] >= 1 and sum(sizes) == len(sk.length)
    one = min(first, key=lambda b: len(b.leaves))
    n, side = ball_cards(one, sk, coverage(paint_cells(sk.outline, seed=int(sk.info["seed"]))))
    assert n >= 1 and side > 0


def test_cards_sit_on_leaves():
    sk = redbud()
    out = _cards(sk)
    assert out.cards > 0
    leaves = leaf_centres(sk)
    gap = np.linalg.norm(out.centres[:, None, :] - leaves[None], axis=2).min(axis=1)
    assert gap.max() <= 1e-9


def test_painted_area_matches():
    sk = redbud()
    cover = coverage(paint_cells(sk.outline, seed=int(sk.info["seed"])))
    x, y = sk.outline[:, 0], sk.outline[:, 1]
    per_len2 = 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
    made = balls(sk, DIALS.ball_m * UPM)
    for b in made:
        n, side = ball_cards(b, sk, cover)
        area = per_len2 * float((sk.length[b.leaves] ** 2).sum())
        card = side**2 * cover
        assert abs(n * card - area) <= card
    total = sum(ball_cards(b, sk, cover)[0] for b in made)
    out = _cards(sk)
    assert out.cards + out.dropped == total


def test_no_card_edge_on():
    out = _cards(redbud())
    least = DIALS.sky / math.sqrt(1 + DIALS.sky**2) - 1e-9
    assert out.cards > 0 and np.abs(out.faces[:, 1]).min() >= least


def test_one_normal_per_card():
    out = _cards(tiny())
    assert out.cards > 0 and out.tris.shape == (2 * out.cards, 3, 3)
    per_card = out.normals.reshape(out.cards, 6, 3)
    assert np.allclose(per_card, per_card[:, :1, :])
    assert np.allclose(np.linalg.norm(per_card, axis=2), 1.0)


def test_buried_dark_dropped():
    reach = 2.5
    here = np.zeros(3)
    up = np.array([0.0, 1.0, 0.0])
    # a card at the shared centre of two big neighbour balls: buried and dark
    two = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
    buried, _ = card_shown(here, up, two, np.array([1.0, 1.0]), reach)
    assert buried < 0.5
    # on the far side of the second ball, facing away: open and unburied
    far = np.array([5.0, 0.0, 0.0])
    kept, light = card_shown(far, np.array([1.0, 0.0, 0.0]), two, np.array([1.0, 1.0]), reach)
    assert kept >= 0.5 and light > 0.9
    # buried but lit: inside one ball, with the neighbour's light reaching it almost whole
    lit, light = card_shown(here, up, two[:1], np.array([1.0]), 0.01)
    assert lit >= 0.5 and light > 0.9
    # dark but not buried: just outside the burial depth of three overlapping balls that shade it
    shaded = np.array([[0.0, 0.81, 0.0], [0.0, 0.81, 0.01], [0.01, 0.81, 0.0]])
    unburied, light = card_shown(here, up, shaded, np.ones(3), 50.0)
    assert unburied >= 0.5 and light < HIDDEN


def test_deterministic():
    a, b = _cards(tiny()), _cards(tiny())
    for name in ("tris", "normals", "uvs", "colours", "centres", "faces"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes()
    assert (a.cards, a.dropped) == (b.cards, b.dropped)



# ------------------------------------------------------------------------------- the light wood

REDBUD_SIGHT = Sight(gap=5.0 * math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60.0)) * UPM,
                     crease_deg=35.0)

# Seen from farther off than the bar's 5 m, most redbud twigs (diameters 0.0009 and up) are thinner
# than the floor: 869 of the 906 leaf-holding limbs and 3 of the 11 bare ones at this gap.
FAR_SIGHT = Sight(gap=0.01, crease_deg=35.0)


def _thin(sk: Skeleton, sight: Sight) -> np.ndarray:
    return np.array([2 * sk.limb_radii(i).max() < sight.gap for i in range(len(sk.tip))])


def _rows(tris: np.ndarray) -> set:
    return {t.tobytes() for t in tris}


def test_thin_leaf_twig_is_a_stick():
    sk = redbud()
    thin, hosts = _thin(sk, FAR_SIGHT), {int(h) for h in sk.host}
    i = next(i for i in range(len(sk.tip)) if thin[i] and i in hosts)
    pts, radii = sk.limb_points(i), sk.limb_radii(i)
    stick = tube_tris(np.stack([pts[0], pts[-1]]), np.array([radii[0], radii[-1]]), 3,
                      int(sk.tip[i]))
    part = card_wood(sk, FAR_SIGHT)[sk.level_names[int(sk.level[i])]]
    assert _rows(stick) <= _rows(part)
    # closed: a base cap, 3 sides of 2 triangles, and a 3-triangle tip sharing one point
    assert len(stick) == 1 + 6 + 3
    _, counts = np.unique(stick.reshape(-1, 3), axis=0, return_counts=True)
    assert (counts == 3).sum() >= 1


def test_bare_thin_limb_left_out():
    sk = redbud()
    thin, hosts = _thin(sk, FAR_SIGHT), {int(h) for h in sk.host}
    bare = [i for i in range(len(sk.tip)) if thin[i] and i not in hosts]
    assert bare, "the redbud has thin limbs that hold no leaf"
    wood = card_wood(sk, FAR_SIGHT)
    shipped = set().union(*(_rows(t) for t in wood.values()))
    for i in bare[:20]:
        assert not (_rows(limb_tris(sk, i, FAR_SIGHT)) & shipped)
    total = sum(len(t) for t in wood.values())
    sticks = sum(10 for i in hosts if thin[i])
    thick = sum(len(limb_tris(sk, i, FAR_SIGHT)) for i in range(len(sk.tip)) if not thin[i])
    assert total == sticks + thick


def test_unrounded_sides_floor_3():
    loose = Sight(gap=1.0, crease_deg=35.0, round=False)  # a gap so wide 3 sides serve
    assert tube_sides(1e-3, loose) == 3
    assert tube_sides(1e-3, Sight(gap=1.0, crease_deg=35.0)) == round_sides(35.0)
    assert Sight(gap=1.0, crease_deg=35.0).round is True


def test_leafless_source_gives_wood_only(capsys):
    sk = redbud()
    per_leaf = ("host", "leaf_t", "node", "base", "mid", "across", "normal", "length", "petiole",
                "shade")
    bare = dataclasses.replace(sk, **{f: getattr(sk, f)[:0] for f in per_leaf})
    parts, leaves = card_parts(bare, REDBUD_SIGHT, DIALS, UPM, LEAN)
    assert leaves is None and not any(n.startswith("Cards_") for n in parts)
    assert capsys.readouterr().out == "cards: Redbud has no leaves; wood only\n"


def test_two_shades_raise():
    two = dataclasses.replace(redbud(), shades=("Leaf", "Dark"))
    with pytest.raises(ManifestError, match="Redbud.*Leaf.*Dark"):
        card_parts(two, REDBUD_SIGHT, DIALS, UPM, LEAN)


def test_card_parts_deterministic():
    sk = redbud()
    a, la = card_parts(sk, REDBUD_SIGHT, DIALS, UPM, LEAN)
    b, lb = card_parts(sk, REDBUD_SIGHT, DIALS, UPM, LEAN)
    assert list(a) == list(b) and "Cards_Leaf" in a
    for k in a:
        assert a[k].tobytes() == b[k].tobytes()
    assert la.normals.tobytes() == lb.normals.tobytes()
    assert la.uvs.tobytes() == lb.uvs.tobytes()


def test_full_mesh_unchanged():
    parts = grown_parts(redbud(), REDBUD_SIGHT)
    assert sum(len(t) for t in parts.values()) == 349392
