import numpy as np
import pytest

from model_kit.core.grow.light import ShadowGrid

A, B, C = 0.5, 2.0, 1.0
SOURCE = np.array([[0.0, 5.5, 0.0]])


def grid(depth=4) -> ShadowGrid:
    g = ShadowGrid(1.0)
    g.add_shadow(SOURCE, depth, A, B)
    return g


def test_a_leaf_shades_the_cells_below():
    g = grid()
    below = np.array([[0.0, 3.5, 0.0]])
    beside = np.array([[3.0, 3.5, 0.0]])  # two layers down the pyramid is only 2 cells wide
    assert g.light(below, C)[0] < g.light(beside, C)[0]
    assert g.light(beside, C)[0] == pytest.approx(C + A)  # no shade there: C - 0 + a


def test_a_bud_in_its_own_cell_reads_full_light():
    assert grid().light(SOURCE, C)[0] == pytest.approx(C)  # shade a, and + a gives it back


def test_nothing_is_shaded_above():
    g = grid()
    assert g.shadow(np.array([[0.0, 6.5, 0.0]]))[0] == 0.0


def test_shadow_falls_with_depth():
    g = grid(depth=5)
    column = np.array([[0.0, 5.5 - q, 0.0] for q in range(6)])
    s = g.shadow(column)
    assert s[0] == pytest.approx(A)
    assert s[1] == pytest.approx(A / B)  # a * b**-1
    assert s[3] == pytest.approx(A / B**3)
    assert np.all(np.diff(s) < 0.0)


def test_shadow_stops_at_the_pyramid_depth():
    g = grid(depth=2)
    assert g.shadow(np.array([[0.0, 5.5 - 2, 0.0]]))[0] > 0.0
    assert g.shadow(np.array([[0.0, 5.5 - 3, 0.0]]))[0] == 0.0


def test_the_pyramid_widens_with_depth():
    g = grid(depth=4)
    at_two_down = np.array([[x, 3.5, 0.0] for x in (-3.0, -2.0, 0.0, 2.0, 3.0)])
    s = g.shadow(at_two_down)
    assert s[0] == 0.0 and s[4] == 0.0  # past the (2q+1) block
    assert s[1] > 0.0 and s[3] > 0.0 and s[2] > 0.0


def test_shades_add_up():
    g = ShadowGrid(1.0)
    g.add_shadow(np.array([[0.0, 5.5, 0.0], [0.0, 5.5, 0.0]]), 4, A, B)
    assert g.shadow(SOURCE)[0] == pytest.approx(2 * A)


def test_buds_of_one_node_do_not_shade_each_other():
    """Three buds at one point (a maple node: two laterals and the terminal) each take back all
    three of their own shadows, so `own` is the count of buds at the node."""
    g = ShadowGrid(1.0)
    node = np.array([[0.0, 5.5, 0.0]] * 3)
    g.add_shadow(node, 4, A, B)
    assert g.light(node, C)[0] == pytest.approx(C - 2 * A)  # one's own shadow back, two others'
    assert g.light(node, C, own=3)[0] == pytest.approx(C)  # the node's whole shadow back
    assert np.allclose(g.light(node, C, own=np.array([3, 3, 3])), C)  # one count per bud


def test_two_batches_share_one_field():
    g = ShadowGrid(1.0)
    g.add_shadow(SOURCE, 4, A, B)
    g.add_shadow(np.array([[0.0, 4.5, 0.0]]), 4, A, B)
    assert g.shadow(np.array([[0.0, 4.5, 0.0]]))[0] == pytest.approx(A + A / B)


def test_a_point_outside_the_box_is_in_full_light():
    g = grid()
    far = np.array([[100.0, -50.0, 100.0]])
    assert g.shadow(far)[0] == 0.0
    assert np.all(g.toward_light(far) == 0.0)


def test_toward_light_points_away_from_the_shade():
    g = grid()
    on_the_source = g.toward_light(SOURCE)[0]  # shade thins upward from the source's own cell
    assert np.linalg.norm(on_the_source) == pytest.approx(1.0)
    assert on_the_source[1] > 0.0
    at_the_edge = g.toward_light(np.array([[2.0, 3.5, 0.0]]))[0]  # the pyramid's edge, 2 down
    assert at_the_edge[0] > 0.0  # out from under it, where the shade ends


def test_toward_light_of_an_unshaded_cell_is_zero():
    g = ShadowGrid(1.0)
    g.add_shadow(SOURCE, 4, A, B)
    assert np.all(g.toward_light(np.array([[0.0, 9.5, 0.0]]))[0] == 0.0)


def test_empty_grid_is_full_light():
    g = ShadowGrid(1.0)
    assert g.light(SOURCE, C)[0] == pytest.approx(C)
    assert g.toward_light(SOURCE).shape == (1, 3)


@pytest.mark.parametrize("args, words", [
    ((-1, A, B), "depth"), ((2, 0.0, B), "a must"), ((2, A, 1.0), "b must"),
])
def test_bad_shadow_numbers_raise(args, words):
    with pytest.raises(ValueError, match=words):
        ShadowGrid(1.0).add_shadow(SOURCE, *args)


def test_one_a_per_grid():
    g = grid()
    with pytest.raises(ValueError, match="shadow a"):
        g.add_shadow(SOURCE, 4, 0.25, B)


def test_cell_must_be_positive():
    with pytest.raises(ValueError, match="cell"):
        ShadowGrid(0.0)
