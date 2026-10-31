from itertools import pairwise

import numpy as np

from model_kit.core.grow.chains import continuation_chains

NONE = -1


def chains(parent, radius, angle=None):
    parent = np.asarray(parent)
    angle = np.zeros(len(parent)) if angle is None else np.asarray(angle, dtype=float)
    return [c.tolist() for c in continuation_chains(parent, np.asarray(radius, dtype=float), angle)]


def test_chain_follows_the_thickest_child():
    # 0 -> 1 is the trunk; at segment 1 the left limb (2 -> 3) is thicker than the right (4).
    got = chains([NONE, 0, 1, 2, 1], [1.0, 1.0, 0.8, 0.8, 0.5])
    assert got == [[0, 1, 2, 3], [4]]


def test_thickest_child_wins_whatever_the_index_order():
    # The thin child comes first in the arrays; the thick one still carries the chain on.
    got = chains([NONE, 0, 0, 1, 2], [1.0, 0.4, 0.9, 0.4, 0.9])
    assert got == [[0, 2, 4], [1, 3]]


def test_tie_goes_to_the_straighter_child():
    got = chains([NONE, 0, 0], [1.0, 0.6, 0.6], angle=[0.0, 40.0, 10.0])
    assert got == [[0, 2], [1]]
    got = chains([NONE, 0, 0], [1.0, 0.6, 0.6], angle=[0.0, 10.0, 40.0])
    assert got == [[0, 1], [2]]


def test_parent_chain_first():
    # A bushy tree whose segments are not in parent-first order of chains.
    parent = [NONE, 0, 1, 1, 3, 3, 2, 6, 6, 0]
    radius = [1.0, 0.9, 0.5, 0.7, 0.6, 0.3, 0.4, 0.3, 0.1, 0.2]
    got = chains(parent, radius)
    which = {s: k for k, chain in enumerate(got) for s in chain}
    assert sorted(which) == list(range(len(parent)))  # every segment is in exactly one chain
    assert sum(len(c) for c in got) == len(parent)
    for k, chain in enumerate(got[1:], start=1):
        assert which[parent[chain[0]]] < k
    for chain in got:  # inside a chain each segment is the next one's parent
        assert all(parent[b] == a for a, b in pairwise(chain))


def test_relay_shoots_make_one_chain():
    # A sympodial zigzag: each year's shoot (a, b) ends at the node where the next year's
    # relay shoot starts, with a thinner side bud as a sibling.
    parent, radius = [NONE], [1.0]
    main = 0
    for year in range(4):
        a = len(parent)
        parent += [main, a]  # this year's shoot, two segments
        radius += [0.9 - 0.1 * year] * 2
        side = len(parent)
        parent.append(a + 1)  # the thinner sibling bud's shoot, seated on the same node
        radius.append(0.2)
        main = a + 1
        assert side == a + 2
    got = chains(parent, radius)
    # The last year's bud (12) has no sibling and ends the chain; the earlier ones are stubs.
    assert got[0] == [0, 1, 2, 4, 5, 7, 8, 10, 11, 12]
    assert got[1:] == [[3], [6], [9]]


def test_several_roots_are_each_a_chain():
    got = chains([NONE, 0, NONE, 2], [1.0, 1.0, 0.5, 0.5])
    assert got == [[0, 1], [2, 3]]


def test_empty():
    assert continuation_chains(np.zeros(0, dtype=int), np.zeros(0), np.zeros(0)) == []


def test_a_cycle_is_refused():
    import pytest

    with pytest.raises(ValueError, match="cycle|reach"):
        chains([1, 0], [1.0, 1.0])
