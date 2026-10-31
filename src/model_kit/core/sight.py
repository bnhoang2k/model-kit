"""How a piece is seen: the one length the kit judges touching and line defects by.

A viewer cannot tell two surfaces apart when they come closer than `EYE_RESOLUTION_ARCMIN`
covers at the piece's nearest viewing distance (`Piece.view.nearest_m`, in real metres). That
length, in the pack's units, is the piece's **seen floor**, and it is also its **contact length**:
two surfaces touch when their gap is at most this. Every check that asks "do these touch?"
(floaters, bearing, clearance, inset-seat, frames, mirror-pair, receiver, connectors, trace) reads
this one length for the piece it judges, and the grower reads it to size its segments. A piece
with no `view` has no such length: `seen_floor` and `contact_length` return None, and every
contact rule reads `Unmeasured` for it.

A decal is lifted off its host by the contact length (`decal_lift`): the largest lift no viewer can
see, so the printed patch draws apart from its host from as far as possible. The lift must also
clear the depth buffer: it has to be at least one depth step at the piece's farthest view
(`depth_gap`), or the decal and its host share a stored depth and flicker. A piece whose view range
is too wide for that cannot carry a lifted decal, and the build raises.

This is a core module and not a check's, so the grower can call it too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import Piece, View
from model_kit.core.tolerances import EYE_RESOLUTION_ARCMIN
from model_kit.core.units import Style, depth_step


@dataclass(frozen=True)
class Sight:
    """What a viewer resolves on one piece: `gap` is the seen floor in the pack's units and
    `crease_deg` the style's crease angle in degrees. The grower sizes its segments from it."""

    gap: float
    crease_deg: float


def sight_for(piece: Piece, style: Style, units_per_metre: float) -> Sight | None:
    """What a viewer resolves on `piece` in `style`, or None when the piece declares no `view` or
    the style names no `crease_deg`: the grower never assumes either (the caller raises or reads
    `Unmeasured`)."""
    gap = seen_floor(piece, units_per_metre)
    if gap is None or style.crease_deg is None:
        return None
    return Sight(gap, style.crease_deg)


def view_floor(view: View, units_per_metre: float) -> float:
    """The length, in the pack's units, that `EYE_RESOLUTION_ARCMIN` covers at the view's nearest
    distance (`View.nearest_m`, in real metres)."""
    return view.nearest_m * math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60.0)) * units_per_metre


def seen_floor(piece: Piece, units_per_metre: float) -> float | None:
    """The length, in the pack's units, that `EYE_RESOLUTION_ARCMIN` covers at the piece's nearest
    viewing distance (`Piece.view.nearest_m`, in real metres): what the eye resolves there. None
    when the piece declares no `view`, so no distance to judge from."""
    return None if piece.view is None else view_floor(piece.view, units_per_metre)


def seen_distance(length: float, units_per_metre: float) -> float:
    """The viewing distance, in real metres, at which `EYE_RESOLUTION_ARCMIN` covers `length` (in
    the pack's units): the inverse of `seen_floor`."""
    return length / (math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60.0)) * units_per_metre)


def contact_length(piece: Piece, units_per_metre: float) -> float | None:
    """How far apart two surfaces of the piece may be and still touch: the seen floor. Touching is
    a gap at or under it. None when the piece declares no `view` (the contact rules read
    `Unmeasured`)."""
    return seen_floor(piece, units_per_metre)


def depth_gap(farthest_m: float, style: Style, units_per_metre: float) -> float:
    """One depth step of the style's reference camera at `farthest_m` (real metres), in the pack's
    units: the gap under which two faces that face the same way draw in one stored depth."""
    return depth_step(style, farthest_m) * units_per_metre


@dataclass(frozen=True)
class DecalLift:
    """How far a decal stands off its host: `lift`, the piece's contact length, and `gap`, the
    depth step at its farthest view, both in the pack's units. `who` names the piece. Two faces
    flicker when they are at most one depth step apart (`coplanar` flags a distance `<= gap`), so a
    lift at or under the gap raises, naming `who`."""

    who: str
    lift: float
    gap: float

    def __post_init__(self) -> None:
        if not (self.lift > 0 and self.gap > 0):
            raise ManifestError(f"{self.who}: a decal's lift and the depth gap must be above 0; "
                                f"got lift {self.lift!r} and gap {self.gap!r}")
        if self.lift <= self.gap:
            raise ManifestError(
                f"{self.who}: the decal lift {self.lift:.7f} (the contact length, one arcminute at "
                f"the nearest view) is not over the depth gap {self.gap:.7f} (one depth step at "
                "the farthest view): the view range is too wide for a lifted decal, since the "
                "decal and its host would draw in the same depth and flicker")


def decal_lift(who: str, view: View | None, style: Style, units_per_metre: float) -> DecalLift:
    """The lift of a decal on a piece seen as `view`: its contact length, checked against the depth
    gap at its farthest view. Raises `ManifestError` naming `who` when the piece declares no view
    or no `farthest_m`, the style no depth camera, or the lift is at or under the gap: nothing is
    assumed."""
    if view is None:
        raise ManifestError(f"{who}: has a decal but declares no view, so it has no lift")
    if view.farthest_m is None:
        raise ManifestError(f"{who}: has a decal but declares no farthest_m, so there is no depth "
                            "gap to check its lift against")
    if style.depth_camera is None:
        raise ManifestError(f"{who}: has a decal but the style declares no depth camera, so "
                            "there is no depth gap to check its lift against")
    return DecalLift(who, view_floor(view, units_per_metre),
                     depth_gap(view.farthest_m, style, units_per_metre))
