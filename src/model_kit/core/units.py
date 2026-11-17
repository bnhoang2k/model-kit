"""Project units and the style file that fixes them."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from model_kit.core.errors import ManifestError, RecordError


@dataclass(frozen=True)
class Style:
    """A style file: how many project units make one metre, plus the page camera and dials.

    The shipping look's line style lives in `dials.edges`, and a style declares one of two things:
    `crease_deg` (with a `crease_why`): the look draws fold lines past that angle; or
    `borders_only: true` (with a `borders_why`): the look draws outlines and no fold lines. A
    style that declares neither says nothing, and the checks that need a line style read
    `Unmeasured`. `dials.leaves`, when present, holds the leaf-normal lean (`normal_lean`,
    `centre_drop`) and a `why`.

    `page_camera` may also declare the reference camera's depth buffer, the one a flicker check
    reads (see `DepthCamera`): `depth_projection` ("perspective", the only kind supported),
    `near_m`, `far_m`, `depth_bits` (a whole number) and `depth_why`. The five come together or
    not at all. They sit beside the page's own `projection`, `turn_deg` and `tilt_deg`, which say
    how the sheet is drawn and are not the game's camera.

    A style checks its own dials and its depth declaration when it is built (`validate`), so one
    built in code is as legal as one read from a file.
    """

    units_per_metre: float
    page_camera: dict
    dials: dict

    def __post_init__(self) -> None:
        self.validate("style")

    def validate(self, where: str) -> None:
        """Raise a `RecordError` naming `where` and the dial when `dials.edges`, `dials.leaves`,
        `dials.cards` or the page camera's depth declaration is not legal."""
        for fault in (_edges_fault(self.dials), _leaves_fault(self.dials),
                      _cards_fault(self.dials), _depth_fault(self.page_camera)):
            if fault is not None:
                raise RecordError(where, *fault)

    @property
    def depth_camera(self) -> DepthCamera | None:
        """The reference camera's depth buffer, or None when the style declares none (a check that
        needs it reads `Unmeasured`). `validate` has already refused a half-made declaration."""
        camera = self.page_camera if isinstance(self.page_camera, dict) else {}
        if not any(key in camera for key in DEPTH_KEYS):
            return None
        return DepthCamera(near_m=float(camera["near_m"]), far_m=float(camera["far_m"]),
                           depth_bits=camera["depth_bits"], why=camera["depth_why"])

    def _edges(self) -> dict:
        return (self.dials or {}).get("edges") or {}

    @property
    def crease_deg(self) -> float | None:
        """The angle over which two faces shade as a hard edge, or None when not declared."""
        angle = self._edges().get("crease_deg")
        return None if angle is None else float(angle)

    @property
    def crease_why(self) -> str | None:
        return self._edges().get("crease_why") if "crease_deg" in self._edges() else None

    @property
    def borders_only(self) -> bool:
        """The look draws outlines and no fold lines, declared with a `borders_why`."""
        return self._edges().get("borders_only") is True

    @property
    def borders_why(self) -> str | None:
        return self._edges().get("borders_why") if self.borders_only else None

    @property
    def draws_outline(self) -> bool:
        """Whether the style draws an outline: false when `dials.edges.outline` is "none" or
        null."""
        edges = (self.dials or {}).get("edges")
        outline = edges.get("outline") if isinstance(edges, dict) else None
        return outline is not None and outline != "none"


# The page-camera keys that declare the reference camera's depth buffer. They come together.
DEPTH_KEYS = ("depth_projection", "near_m", "far_m", "depth_bits", "depth_why")
# The one projection a depth step is derived for: the perspective divide of a fixed-point buffer.
DEPTH_PROJECTION = "perspective"
# The widest integer depth format Khronos's page lists (32 bits); a float buffer is not integer.
MAX_DEPTH_BITS = 32


@dataclass(frozen=True)
class DepthCamera:
    """A style's reference camera, as far as depth goes: near and far planes in real metres, the
    integer depth buffer's width in bits, and `why` (where those numbers come from)."""

    near_m: float
    far_m: float
    depth_bits: int
    why: str


def _depth_fault(page_camera: object) -> tuple[str, str] | None:
    """(dial, why) of the first fault in the depth declaration of `page_camera`, or None when it
    declares none or a legal one. Only a perspective camera with an integer depth buffer is
    supported: any other declaration is refused, never read as a near miss."""
    if not isinstance(page_camera, dict) or not any(key in page_camera for key in DEPTH_KEYS):
        return None
    for key in DEPTH_KEYS:
        if key not in page_camera:
            return (f"page_camera.{key}",
                    "a depth declaration needs all of " + ", ".join(DEPTH_KEYS) + f"; {key} is "
                    "missing")
    projection = page_camera["depth_projection"]
    if projection != DEPTH_PROJECTION:
        return ("page_camera.depth_projection",
                f"only a {DEPTH_PROJECTION!r} depth buffer is supported, got {projection!r}")
    near, far = page_camera["near_m"], page_camera["far_m"]
    for key, value in (("near_m", near), ("far_m", far)):
        if not _positive_number(value):
            return f"page_camera.{key}", f"must be a number above 0, got {value!r}"
    if not near < far:
        return ("page_camera.far_m",
                f"must be past near_m, got near_m {near!r} and far_m {far!r}")
    bits = page_camera["depth_bits"]
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= MAX_DEPTH_BITS:
        why = f"only an integer depth buffer of 1 to {MAX_DEPTH_BITS} bits is supported"
        return "page_camera.depth_bits", f"{why}, got {bits!r}"
    if not _has_text(page_camera["depth_why"]):
        got = page_camera["depth_why"]
        return ("page_camera.depth_why",
                f"needs a depth_why saying where the camera comes from, got {got!r}")
    return None


def _edges_fault(dials: dict) -> tuple[str, str] | None:
    """(dial, why) of the first fault in `dials.edges`, or None. It holds `crease_deg` with a
    `crease_why`, or `borders_only: true` with a `borders_why`, never both; `borders_why` with no
    `borders_only` is refused, so a stale reason does not outlive its flag."""
    edges = (dials or {}).get("edges")
    if edges is None:
        return None
    if not isinstance(edges, dict):
        return "dials.edges", "must be an object"
    angle = "crease_deg" in edges
    if angle:
        if not _is_crease_deg(edges["crease_deg"]):
            return ("dials.edges.crease_deg",
                    f"must be a number above 0 and below 180, got {edges['crease_deg']!r}")
        if not _has_text(edges.get("crease_why")):
            got = edges.get("crease_why")
            return ("dials.edges.crease_why",
                    f"crease_deg needs a crease_why saying where the angle comes from, got {got!r}")
    borders = edges.get("borders_only", False)
    if not isinstance(borders, bool):
        return "dials.edges.borders_only", f"must be true or false, got {borders!r}"
    if borders and angle:
        return ("dials.edges.borders_only",
                "a style that draws borders only has no crease_deg: declare one or the other")
    if borders and not _has_text(edges.get("borders_why")):
        got = edges.get("borders_why")
        return ("dials.edges.borders_why",
                f"borders_only needs a borders_why saying why there are no fold lines, got {got!r}")
    if not borders and "borders_why" in edges:
        return ("dials.edges.borders_why",
                "names no borders-only look to explain; set borders_only to true or drop it")
    return None


def _leaves_fault(dials: dict) -> tuple[str, str] | None:
    """(dial, why) of the first fault in `dials.leaves`, or None when the dial is absent. A
    present dial holds `normal_lean` (0 to 1), `centre_drop` (a number) and a `why`."""
    leaves = (dials or {}).get("leaves")
    if leaves is None:
        return None
    if not isinstance(leaves, dict):
        return "dials.leaves", "must be an object"
    for key in ("normal_lean", "centre_drop"):
        value = leaves.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            return f"dials.leaves.{key}", f"must be a number, got {value!r}"
    if not 0.0 <= leaves["normal_lean"] <= 1.0:
        return "dials.leaves.normal_lean", f"must be 0 to 1, got {leaves['normal_lean']!r}"
    if not _has_text(leaves.get("why")):
        got = leaves.get("why")
        return "dials.leaves.why", f"needs a why saying where the lean comes from, got {got!r}"
    return None


def _cards_fault(dials: dict) -> tuple[str, str] | None:
    """(dial, why) of the first fault in `dials.cards`, or None when the dial is absent. A present
    dial holds `ball_m` (above 0), `normal_mix` (0 to 1), `sky` (0 or more) and a `why`."""
    cards = (dials or {}).get("cards")
    if cards is None:
        return None
    if not isinstance(cards, dict):
        return "dials.cards", "must be an object"
    for key in ("ball_m", "normal_mix", "sky"):
        value = cards.get(key)
        if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
            return f"dials.cards.{key}", f"must be a number, got {value!r}"
    if cards["ball_m"] <= 0:
        return "dials.cards.ball_m", f"must be above 0, got {cards['ball_m']!r}"
    if not 0.0 <= cards["normal_mix"] <= 1.0:
        return "dials.cards.normal_mix", f"must be 0 to 1, got {cards['normal_mix']!r}"
    if cards["sky"] < 0:
        return "dials.cards.sky", f"must be 0 or more, got {cards['sky']!r}"
    if not _has_text(cards.get("why")):
        got = cards.get("why")
        return "dials.cards.why", f"needs a why saying where the dials come from, got {got!r}"
    return None


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def load_style(path: Path) -> Style:
    """Read a style file. Units are required: the loader never assumes them."""
    path = Path(path)
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise ManifestError(f"{path}: cannot read the style file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: the style file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: the style file must hold a JSON object")
    if "units_per_metre" not in data:
        raise ManifestError(f"{path}: the style file has no units_per_metre")
    units = data["units_per_metre"]
    if not _positive_number(units):
        raise ManifestError(f"{path}: units_per_metre must be a positive number, got {units!r}")
    page_camera = data.get("page_camera", {})
    dials = data.get("dials", {})
    if not isinstance(page_camera, dict):
        raise ManifestError(f"{path}: page_camera must be an object")
    if not isinstance(dials, dict):
        raise ManifestError(f"{path}: dials must be an object")
    try:
        return Style(units_per_metre=float(units), page_camera=page_camera, dials=dials)
    except RecordError as exc:
        raise ManifestError(f"{path}: {exc.field}: {exc.why}") from exc


def _is_crease_deg(value: object) -> bool:
    """The one range rule for a crease angle: a finite number above 0 and below 180."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 < value < 180
    )


def check_crease_deg(value: object, where: str) -> float:
    """`value` as a crease angle in degrees, or a `ManifestError` naming `where`.

    `where` says whose value it is (a file, a pack, a style).
    """
    if not _is_crease_deg(value):
        raise ManifestError(
            f"{where}: crease_deg must be a number above 0 and below 180, got {value!r}"
        )
    return float(value)


def thickness_base(style: Style, listed_min_thickness_m: Iterable[float]) -> float:
    """The thinnest wall a piece check may trust, in project units.

    Listed values are real metres. The result is the larger of the smallest listed value and
    the style's `min_thickness` dial (metres; null or 0 means no floor), times units_per_metre.
    """
    listed = list(listed_min_thickness_m)
    for value in listed:
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise ManifestError(f"listed min thickness must be a number in metres, got {value!r}")
        if not math.isfinite(value) or value < 0:
            raise ManifestError(f"listed min thickness must be 0 or more, got {value!r}")
    dial = (style.dials or {}).get("min_thickness")
    if dial is None:
        dial = 0.0
    if not isinstance(dial, int | float) or isinstance(dial, bool):
        raise ManifestError(f"dial min_thickness must be a number or null, got {dial!r}")
    if not math.isfinite(dial) or dial < 0:
        raise ManifestError(f"dial min_thickness must be 0 or more, got {dial!r}")
    smallest = min(listed) if listed else 0.0
    base = max(smallest, dial) * style.units_per_metre
    if base <= 0:
        raise ManifestError(
            "thickness base is 0: no listed min thickness and no dial min_thickness"
        )
    return base


def depth_step(style: Style, distance_m: float) -> float:
    """The smallest gap, in real metres, that a perspective integer depth buffer resolves at
    `distance_m` from the eye: `z**2 * (f - n) / (f * n * 2**bits)`.

    Khronos's page relates eye depth to the stored value:
    `z_e / w_e = f * n / ((z_w / s) * (f - n) - f)`. Solved for `z_w` and differentiated in `z`,
    one stored step spans `z**2 * (f - n) / (f * n * s)` of eye distance (Khronos, "Depth Buffer
    Precision", OpenGL Wiki: https://wikis.khronos.org/opengl/Depth_Buffer_Precision). An archived
    copy of the older page is at:
    https://web.archive.org/web/2024/https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
    and the step is derived from it here, not quoted. The page writes the scale
    `s = 2**bits - 1`; this uses `2**bits`, the buffer's count of steps, which differs by one part
    in `2**bits`. Two faces at least one step apart round to different stored depths; closer than
    that they may draw in the same depth and flicker.

    A style with no depth declaration, or a distance that is not above 0 or lies outside the
    camera's near and far planes, raises `ManifestError`: a step at a distance the buffer never
    holds means nothing.
    """
    camera = style.depth_camera
    if camera is None:
        raise ManifestError("the style declares no depth camera (page_camera near_m, far_m and "
                            "depth_bits), so there is no depth step")
    if not _positive_number(distance_m):
        raise ManifestError(f"a depth step needs a distance above 0, got {distance_m!r}")
    if not camera.near_m <= distance_m <= camera.far_m:
        raise ManifestError(f"distance {distance_m!r} m lies outside the camera's near "
                            f"{camera.near_m!r} m and far {camera.far_m!r} m planes")
    n, f = camera.near_m, camera.far_m
    return distance_m * distance_m * (f - n) / (f * n * 2 ** camera.depth_bits)
