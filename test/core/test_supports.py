"""Supports: the radius each one carries, from its chain or from its own footprint."""

import dataclasses

import numpy as np
import pytest

from model_kit.core.errors import Unmeasurable
from model_kit.core.scene import RigRest, rig_spec
from model_kit.core.supports import body_radius, group_rest_points, support_radius

from .bodies.humanoid import humanoid_scene
from .bodies.worm import worm_scene


def test_a_support_that_is_a_chains_effector_group_uses_that_chains_radius():
    ctx = humanoid_scene()
    spec = rig_spec(ctx, "stand")
    chain = spec.chains["leg.L"]
    assert chain.effector_group == "foot.L"
    assert support_radius(spec, None, "foot.L") == chain.effector_radius  # no rest file needed
    assert support_radius(spec, ctx.rests["humanoid:Person"], "foot.L") == chain.effector_radius


def test_the_smaller_of_two_chains_to_one_group_counts():
    spec = rig_spec(humanoid_scene(), "stand")
    twin = dataclasses.replace(spec.chains["leg.L"], effector_radius=0.01)
    spec = dataclasses.replace(spec, chains={**spec.chains, "leg.twin": twin})
    assert support_radius(spec, None, "foot.L") == 0.01


def test_any_other_support_uses_half_the_smaller_horizontal_extent_of_its_rest_points():
    ctx = worm_scene()
    spec = rig_spec(ctx, "stand")
    rest = ctx.rests["worm:Worm"]
    assert not spec.chains and spec.supports == ("belly",)
    points = group_rest_points(spec, rest, "belly")
    assert len(points) > 0
    # Worked by hand from the belly's vertices: glTF y is up, so the floor plane is x and z.
    x_span = points[:, 0].max() - points[:, 0].min()
    z_span = points[:, 2].max() - points[:, 2].min()
    assert x_span != z_span
    assert support_radius(spec, rest, "belly") == pytest.approx(min(x_span, z_span) / 2)
    assert support_radius(spec, rest, "belly") > 0
    assert max(x_span, z_span) > 5 * min(x_span, z_span)  # long and narrow: the width counts


def test_the_footprint_ignores_the_height():
    ctx = worm_scene()
    spec = rig_spec(ctx, "stand")
    rest = ctx.rests["worm:Worm"]
    tall = {p: s * np.array([1.0, 7.0, 1.0], dtype=np.float32) for p, s in rest.shells.items()}
    assert support_radius(spec, dataclasses.replace(rest, shells=tall), "belly") == pytest.approx(
        support_radius(spec, rest, "belly"))


def test_a_footprint_with_nothing_to_measure_is_unmeasurable():
    ctx = worm_scene()
    spec = rig_spec(ctx, "stand")
    rest = ctx.rests["worm:Worm"]
    with pytest.raises(Unmeasurable, match="no rest file"):
        support_radius(spec, None, "belly")
    with pytest.raises(Unmeasurable, match="no contact group"):
        support_radius(spec, rest, "wing")
    ghost = dataclasses.replace(spec, contact_groups={**spec.contact_groups,
                                                      "ghost": ("no_such_bone",)})
    with pytest.raises(Unmeasurable, match="no rest points"):
        support_radius(ghost, rest, "ghost")
    # points in one line across the floor have no width
    part = next(iter(rest.shells))
    line = np.zeros((1, 4, 3), dtype=np.float32)
    line[0, :, 2] = [0.0, 1.0, 2.0, 3.0]
    thin = RigRest({part: line}, {part: rest.faces[part]},
                   {part: np.zeros(4, dtype=np.int32)}, {part: ("seg_1",)})
    flat = dataclasses.replace(spec, contact_groups={**spec.contact_groups, "belly": ("seg_1",)})
    with pytest.raises(Unmeasurable, match="no width"):
        support_radius(flat, thin, "belly")


def test_body_radius_is_the_smallest_over_chains_and_supports():
    ctx = humanoid_scene()
    spec = rig_spec(ctx, "stand")
    rest = ctx.rests["humanoid:Person"]
    radii = {c.effector_group: c.effector_radius for c in spec.chains.values()}
    assert body_radius(spec, rest) == min(radii.values()) == radii["hand.L"]
    # With no chain, a support is measured by its footprint, and with no rest file it is left out.
    small = dataclasses.replace(spec, chains={}, supports=("foot.L",))
    assert body_radius(small, rest) == support_radius(small, rest, "foot.L")
    assert body_radius(small, None) is None
    ctx = worm_scene()
    worm = rig_spec(ctx, "stand")
    assert body_radius(worm, ctx.rests["worm:Worm"]) == support_radius(
        worm, ctx.rests["worm:Worm"], "belly")
    assert body_radius(dataclasses.replace(worm, supports=()), None) is None
    assert body_radius(worm, None) is None  # the belly's footprint needs the rest file
