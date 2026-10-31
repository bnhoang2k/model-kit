"""The branching checks: they read a grown piece's skeleton (`core.grow.skeleton`), not its
merged meshes, because one mesh per level hides which limb touches which. skeleton-match then
proves the shipped mesh is the skeleton's. What each measures is in `kinds/branching/rules.md`.
"""

from model_kit.core.checks.branching.crossing import CrossingOrganic
from model_kit.core.checks.branching.floor import FloorOrganic
from model_kit.core.checks.branching.leaf_angle import LeafAngle
from model_kit.core.checks.branching.limb_seat import LimbSeat
from model_kit.core.checks.branching.near_miss import NearMissOrganic
from model_kit.core.checks.branching.reference_range import ReferenceRange
from model_kit.core.checks.branching.skeleton_match import SkeletonMatch
from model_kit.core.checks.branching.taper import Taper

# The plant checks (reference-range, taper, leaf-angle) are registered. even-cover and leonardo
# are taste-only (kinds/branching/rules.md), so they are not registered.
BRANCHING_CHECKS = (LimbSeat, FloorOrganic, CrossingOrganic, NearMissOrganic, SkeletonMatch,
                    ReferenceRange, Taper, LeafAngle)

__all__ = ["BRANCHING_CHECKS", "CrossingOrganic", "FloorOrganic", "LeafAngle", "LimbSeat",
           "NearMissOrganic", "ReferenceRange", "SkeletonMatch", "Taper"]
