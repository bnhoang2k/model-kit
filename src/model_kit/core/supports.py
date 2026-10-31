"""Supports: the contact groups that bear a standing body's weight, and the size of each.

`rig.supports` names them (two feet, four paws, a snake's belly). A body check reads this list
instead of guessing the feet from the rig's top bone. Each support carries a radius `r`, the length
the clip-hygiene tolerances are shares of (`FLOOR_OF_EFFECTOR`, `SLIDE_OF_EFFECTOR`,
`SKIN_OF_EFFECTOR`):

- a support that is a chain's effector group uses that chain's `effector_radius`;
- any other support uses its own footprint: half the smaller horizontal extent (x and z, glTF y
  up) of the group's points at rest, the points of the vertices whose strongest bone is in the
  group.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from model_kit.core.errors import Unmeasurable
from model_kit.core.manifest import RigSpec
from model_kit.core.scene import RigRest


def group_rest_points(spec: RigSpec, rest: RigRest, group: str) -> np.ndarray:
    """The rest points of a contact group: (N, 3), every vertex whose strongest bone is in the
    group, parts in name order (the same points a clip's `g:` samples follow)."""
    members = list(spec.contact_groups[group])
    found = []
    for part in sorted(rest.shells):
        names = np.array(rest.vertex_groups[part])
        keep = np.isin(names[rest.labels[part]], members)
        found.append(np.asarray(rest.shells[part][0], dtype=float)[keep])
    return np.concatenate(found) if found else np.zeros((0, 3))


def support_radius(spec: RigSpec, rest: RigRest | None, group: str) -> float:
    """The radius of one support. A group with no chain and no rest points to measure raises
    `Unmeasurable`, so a check turns it into an `Unmeasured` line."""
    if group not in spec.contact_groups:
        raise Unmeasurable(f"{group!r} is no contact group of the rig")
    radii = [c.effector_radius for c in spec.chains.values() if c.effector_group == group]
    if radii:
        return min(radii)
    if rest is None:
        raise Unmeasurable(f"support {group!r} is no chain's effector group and the scene holds "
                           "no rest file to measure its footprint from")
    points = group_rest_points(spec, rest, group)
    if len(points) == 0:
        raise Unmeasurable(f"support {group!r} has no rest points to measure its footprint from")
    extent = points[:, [0, 2]].max(axis=0) - points[:, [0, 2]].min(axis=0)
    if not extent.min() > 0:
        raise Unmeasurable(f"support {group!r} has a footprint with no width: its rest points "
                           f"span {extent[0]:.4g} by {extent[1]:.4g} across the floor")
    return float(extent.min()) / 2


def body_radius(spec: RigSpec, rest: RigRest | None,
                radii: Mapping[str, float] | None = None) -> float | None:
    """The smallest radius over the rig's chains and supports: the strictest length a body-wide
    tolerance (the floor, a loop's pop, a pose-overlap skin) is a share of. A support with no
    radius to measure is left out (the rule that reads it says so). None for a rig with neither
    a chain nor a measurable support. A caller that has already sized the supports passes them as
    `radii` (support -> radius), so no footprint is measured twice."""
    every = [c.effector_radius for c in spec.chains.values()]
    if radii is not None:
        every += list(radii.values())
    else:
        for group in spec.supports:
            try:
                every.append(support_radius(spec, rest, group))
            except Unmeasurable:
                continue
    return min(every) if every else None
