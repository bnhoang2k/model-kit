"""Soft part forms: the blob, ring, meta and decal records, their seats and deforms, and the
readers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from model_kit.core.tolerances import UNIT_LENGTH_SLACK

from .fields import (
    _At,
    _choice,
    _fields,
    _list,
    _name,
    _num,
    _obj,
    _positive,
    _vec,
    _why,
)
from .records import Axis, Vec3

END_NAMES = ("top", "bottom")
Blend = Literal["linear", "smooth"]
BLENDS = get_args(Blend)
CapKind = Literal["round", "super"]
CAP_KINDS = get_args(CapKind)
ElementKind = Literal["ball", "ellipsoid"]
ELEMENT_KINDS = get_args(ElementKind)
DEFORM_KINDS = ("push", "flatten")
DEFORM_AXES = ("x", "y", "z", "-x", "-y", "-z")
PUSH_ALONG = ("x", "y", "z")
FORM_SHAPES = ("blob", "ring", "meta", "decal")
FALLOFF_AXES = get_args(Axis)

MIN_RINGS = 3
MIN_SEGMENTS = 3
# A ring axis within this sine of glTF X or Z has no side or depth direction to build.
AXIS_PARALLEL_SINE = 1e-9
# The way a decal's cast travels when its form names none: from the front (glTF +Z) to the back.
DECAL_ALONG = (0.0, 0.0, -1.0)


@dataclass(frozen=True)
class BlobForm:
    half: Vec3
    centre: Vec3
    power_side: float
    power_up: float
    power_down: float
    segments: int
    rings: int


@dataclass(frozen=True)
class Station:
    u: float
    half_side: float
    half_depth: float
    off_side: float = 0.0
    off_depth: float = 0.0
    inner_power: float = 1.0


@dataclass(frozen=True)
class Cap:
    kind: CapKind
    length: float
    power: float | None


@dataclass(frozen=True)
class Bump:
    u: float
    u_width: float
    angle: float
    angle_width: float
    height: float
    lift: float


@dataclass(frozen=True)
class Smooth:
    passes: int
    factor: float
    below: float | None


@dataclass(frozen=True)
class RingForm:
    top: Vec3
    bottom: Vec3
    rings: int
    segments: int
    stations: tuple[Station, ...]
    blend: Blend = "linear"
    caps: dict[str, Cap] = field(default_factory=dict, hash=False)
    poles: dict[str, tuple[float, float]] = field(default_factory=dict, hash=False)
    bumps: tuple[Bump, ...] = ()
    smooth: Smooth | None = None


@dataclass(frozen=True)
class Element:
    kind: ElementKind
    centre: Vec3
    half: Vec3


@dataclass(frozen=True)
class MetaForm:
    elements: tuple[Element, ...]
    resolution: float
    threshold: float
    smooth: Smooth
    decimate: float | None = None


@dataclass(frozen=True)
class DecalForm:
    """An open patch printed on the part's `on` host: an oval outline cast onto the host's
    surface and lifted a hair off it (the builder owns the lift).

    A cast along `along` (a unit vector, glTF axes) through the point `at` finds where the patch
    sits on the host. The oval lies in the host's tangent plane there, `half` = (half width, half
    height), with its height along glTF +Y laid onto that plane. `half` names the oval only; any
    other outline kind gets a field of its own. The patch is a polar grid: a centre point and
    `rings` rings of `segments` points each, every grid point cast back onto the host along the
    hit's normal."""

    half: tuple[float, float]
    segments: int
    rings: int
    at: Vec3
    along: Vec3 = DECAL_ALONG


@dataclass(frozen=True)
class SeatOn:
    on: str
    at: tuple[float, float]
    sink: float


@dataclass(frozen=True)
class Push:
    axis: str
    amount: tuple[tuple[float, float], ...]
    along: str
    ramp_from: float
    ramp_to: float
    ramp_max: float


@dataclass(frozen=True)
class Flatten:
    axis: str
    start: float
    end: float


@dataclass(frozen=True)
class SoftForm:
    body: BlobForm | RingForm | MetaForm | DecalForm | None
    mirror_of: str | None
    seat: SeatOn | None
    deforms: tuple[Push | Flatten, ...]
    why: str | None


@dataclass(frozen=True)
class Falloff:
    """How a chain bind spreads its weights: u runs 0 to 1 from `start` to `end` along `along`,
    and band i is where the weight passes from bone i to bone i + 1."""

    along: Axis
    start: float
    end: float
    bands: tuple[tuple[float, float], ...]


def _count(at: _At, raw: Any, minimum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        at.fail(f"must be a whole number, got {raw!r}")
    if raw < minimum:
        at.fail(f"must be at least {minimum}, got {raw!r}")
    return raw


def _non_negative(at: _At, raw: Any) -> float:
    value = _num(at, raw)
    if value < 0:
        at.fail(f"must be 0 or more, got {raw!r}")
    return value


def read_smooth(at: _At, raw: Any) -> Smooth:
    f = _fields(at, raw, ("passes", "factor"), ("below",))
    factor = _num(at.key("factor"), f["factor"])
    if not 0 < factor <= 1:
        at.key("factor").fail(f"must be above 0 and at most 1, got {f['factor']!r}")
    below = _num(at.key("below"), f["below"]) if "below" in f else None
    return Smooth(_count(at.key("passes"), f["passes"], 1), factor, below)


def read_falloff(at: _At, raw: Any) -> Falloff:
    """Read one chain bind's `falloff`. The caller checks the band count against the bones."""
    f = _fields(at, raw, ("along", "start", "end", "bands"))
    start, end = _num(at.key("start"), f["start"]), _num(at.key("end"), f["end"])
    if start == end:
        at.key("end").fail(f"must differ from start, got {f['end']!r}")
    bands_at = at.key("bands")
    bands: list[tuple[float, float]] = []
    for i, row in enumerate(_list(bands_at, f["bands"], nonempty=True)):
        lo, hi = _vec(bands_at.index(i), row, 2)
        if not 0 <= lo < hi <= 1:
            bands_at.index(i).fail(f"must satisfy 0 <= lo < hi <= 1, got {[lo, hi]}")
        if bands and lo < bands[-1][1]:
            bands_at.index(i).fail(f"starts at {lo!r}, inside the band before it, which "
                                   f"ends at {bands[-1][1]!r}")
        bands.append((lo, hi))
    return Falloff(_choice(at.key("along"), f["along"], FALLOFF_AXES), start, end, tuple(bands))


def read_seat(at: _At, raw: Any) -> SeatOn:
    f = _fields(at, raw, ("on", "at", "sink"))
    a, b = _vec(at.key("at"), f["at"], 2)
    return SeatOn(_name(at.key("on"), f["on"]), (a, b), _num(at.key("sink"), f["sink"]))


def read_deform(at: _At, raw: Any) -> Push | Flatten:
    kind = _choice(at.key("kind"), _obj(at, raw).get("kind"), DEFORM_KINDS)
    if kind == "flatten":
        f = _fields(at, raw, ("kind", "axis", "start", "end"))
        return Flatten(_choice(at.key("axis"), f["axis"], DEFORM_AXES),
                       _num(at.key("start"), f["start"]), _num(at.key("end"), f["end"]))
    f = _fields(at, raw, ("kind", "axis", "amount", "along", "ramp_from", "ramp_to",
                          "ramp_max"))
    amount_at = at.key("amount")
    rows = tuple(_vec(amount_at.index(i), row, 2)
                 for i, row in enumerate(_list(amount_at, f["amount"], nonempty=True)))
    for i in range(1, len(rows)):
        if rows[i][0] <= rows[i - 1][0]:
            amount_at.index(i).fail(f"y {rows[i][0]!r} does not rise above {rows[i - 1][0]!r}")
    ramp_from = _num(at.key("ramp_from"), f["ramp_from"])
    ramp_to = _num(at.key("ramp_to"), f["ramp_to"])
    if ramp_to == ramp_from:
        at.key("ramp_to").fail("must differ from ramp_from")
    return Push(
        axis=_choice(at.key("axis"), f["axis"], DEFORM_AXES),
        amount=rows,
        along=_choice(at.key("along"), f["along"], PUSH_ALONG),
        ramp_from=ramp_from,
        ramp_to=ramp_to,
        ramp_max=_positive(at.key("ramp_max"), f["ramp_max"]),
    )


def _blob(at: _At, f: dict) -> BlobForm:
    half = _vec(at.key("half"), f["half"], 3)
    if not all(h > 0 for h in half):
        at.key("half").fail(f"every half-size must be above 0, got {list(half)}")
    return BlobForm(
        half=half,
        centre=_vec(at.key("centre"), f["centre"], 3),
        power_side=_positive(at.key("power_side"), f["power_side"]),
        power_up=_positive(at.key("power_up"), f["power_up"]),
        power_down=_positive(at.key("power_down"), f["power_down"]),
        segments=_count(at.key("segments"), f["segments"], MIN_SEGMENTS),
        rings=_count(at.key("rings"), f["rings"], MIN_RINGS),
    )


def _station(at: _At, raw: Any) -> Station:
    f = _fields(at, raw, ("u", "half_side", "half_depth"),
                ("off_side", "off_depth", "inner_power"))
    return Station(
        u=_num(at.key("u"), f["u"]),
        half_side=_non_negative(at.key("half_side"), f["half_side"]),
        half_depth=_non_negative(at.key("half_depth"), f["half_depth"]),
        off_side=_num(at.key("off_side"), f["off_side"]) if "off_side" in f else 0.0,
        off_depth=_num(at.key("off_depth"), f["off_depth"]) if "off_depth" in f else 0.0,
        inner_power=_positive(at.key("inner_power"), f["inner_power"])
        if "inner_power" in f else 1.0,
    )


def _stations(at: _At, raw: Any) -> tuple[Station, ...]:
    items = _list(at, raw, nonempty=True)
    stations = tuple(_station(at.index(i), v) for i, v in enumerate(items))
    if len(stations) < 2:
        at.fail("needs at least 2 stations, one at u 0 and one at u 1")
    if stations[0].u != 0:
        at.index(0).key("u").fail(f"the first station must sit at u 0, got {stations[0].u!r}")
    if stations[-1].u != 1:
        last = len(stations) - 1
        at.index(last).key("u").fail(f"the last station must sit at u 1, got {stations[-1].u!r}")
    for i in range(1, len(stations)):
        if stations[i].u <= stations[i - 1].u:
            at.index(i).key("u").fail(
                f"u {stations[i].u!r} does not rise above {stations[i - 1].u!r}")
    for i, st in enumerate(stations[1:-1], start=1):
        for name in ("half_side", "half_depth"):
            if getattr(st, name) == 0:
                at.index(i).key(name).fail("may be 0 only at u 0 or u 1")
    return stations


def _cap(at: _At, raw: Any) -> Cap:
    f = _fields(at, raw, ("kind", "length"), ("power",))
    kind = _choice(at.key("kind"), f["kind"], CAP_KINDS)
    length = _positive(at.key("length"), f["length"])
    if kind == "round":
        if "power" in f:
            at.key("power").fail("a round cap has no power")
        return Cap(kind, length, None)
    if "power" not in f:
        at.fail("missing field 'power': a super cap needs one")
    return Cap(kind, length, _positive(at.key("power"), f["power"]))


def _ends(at: _At, raw: Any) -> dict:
    obj = _obj(at, raw)
    for end in obj:
        if end not in END_NAMES:
            at.fail(f"unknown end {end!r}; ends are {list(END_NAMES)}")
    return obj


def _bump(at: _At, raw: Any) -> Bump:
    f = _fields(at, raw, ("u", "u_width", "angle", "angle_width", "height", "lift"))
    return Bump(
        u=_num(at.key("u"), f["u"]),
        u_width=_positive(at.key("u_width"), f["u_width"]),
        angle=_num(at.key("angle"), f["angle"]),
        angle_width=_positive(at.key("angle_width"), f["angle_width"]),
        height=_num(at.key("height"), f["height"]),
        lift=_num(at.key("lift"), f["lift"]),
    )


def _axis_length(at: _At, top: Vec3, bottom: Vec3) -> float:
    ax, ay, az = (b - t for t, b in zip(top, bottom))
    length = math.sqrt(ax * ax + ay * ay + az * az)
    if length == 0:
        at.fail("top and bottom are the same point")
    if math.hypot(ay, az) / length < AXIS_PARALLEL_SINE:
        at.fail("the axis runs along X, so the ring's side direction is undefined")
    if math.hypot(ax, ay) / length < AXIS_PARALLEL_SINE:
        at.fail("the axis runs along Z, so the ring's depth direction is undefined")
    return length


def _ring(at: _At, f: dict) -> RingForm:
    top, bottom = _vec(at.key("top"), f["top"], 3), _vec(at.key("bottom"), f["bottom"], 3)
    length = _axis_length(at, top, bottom)
    caps_at = at.key("caps")
    caps = {end: _cap(caps_at.key(end), v)
            for end, v in _ends(caps_at, f.get("caps", {})).items()}
    if sum(c.length for c in caps.values()) >= length:
        caps_at.fail(f"cap lengths together must stay under the axis length {length}")
    poles_at = at.key("poles")
    poles = {}
    for end, v in _ends(poles_at, f.get("poles", {})).items():
        side, depth = _vec(poles_at.key(end), v, 2)
        poles[end] = (side, depth)
    bumps_at = at.key("bumps")
    bumps = tuple(_bump(bumps_at.index(i), v)
                  for i, v in enumerate(_list(bumps_at, f.get("bumps", []))))
    return RingForm(
        top=top,
        bottom=bottom,
        rings=_count(at.key("rings"), f["rings"], MIN_RINGS),
        segments=_count(at.key("segments"), f["segments"], MIN_SEGMENTS),
        stations=_stations(at.key("stations"), f["stations"]),
        blend=_choice(at.key("blend"), f.get("blend", "linear"), BLENDS),
        caps=caps,
        poles=poles,
        bumps=bumps,
        smooth=read_smooth(at.key("smooth"), f["smooth"]) if "smooth" in f else None,
    )


def _element(at: _At, raw: Any) -> Element:
    f = _fields(at, raw, ("kind", "centre", "half"))
    kind = _choice(at.key("kind"), f["kind"], ELEMENT_KINDS)
    half = _vec(at.key("half"), f["half"], 3)
    if not all(h > 0 for h in half):
        at.key("half").fail(f"every half-size must be above 0, got {list(half)}")
    if kind == "ball" and len(set(half)) != 1:
        at.key("half").fail(f"a ball has equal halves, got {list(half)}")
    return Element(kind, _vec(at.key("centre"), f["centre"], 3), half)


def _meta(at: _At, f: dict) -> MetaForm:
    elements_at = at.key("elements")
    elements = tuple(_element(elements_at.index(i), v)
                     for i, v in enumerate(_list(elements_at, f["elements"], nonempty=True)))
    decimate = None
    if "decimate" in f:
        decimate = _num(at.key("decimate"), f["decimate"])
        if not 0 < decimate <= 1:
            at.key("decimate").fail(f"must be above 0 and at most 1, got {f['decimate']!r}")
    return MetaForm(
        elements=elements,
        resolution=_positive(at.key("resolution"), f["resolution"]),
        threshold=_positive(at.key("threshold"), f["threshold"]),
        smooth=read_smooth(at.key("smooth"), f["smooth"]),
        decimate=decimate,
    )


def _decal(at: _At, f: dict) -> DecalForm:
    half = _vec(at.key("half"), f["half"], 2)
    if not all(h > 0 for h in half):
        at.key("half").fail(f"every half-size must be above 0, got {list(half)}")
    along = DECAL_ALONG
    if "along" in f:
        along = _vec(at.key("along"), f["along"], 3)
        if abs(math.sqrt(sum(v * v for v in along)) - 1.0) > UNIT_LENGTH_SLACK:
            at.key("along").fail(f"must be a unit vector, got {list(along)}")
    return DecalForm(
        half=half,
        segments=_count(at.key("segments"), f["segments"], MIN_SEGMENTS),
        rings=_count(at.key("rings"), f["rings"], 1),
        at=_vec(at.key("at"), f["at"], 3),
        along=along,
    )


_BODIES = {
    "blob": (_blob, ("half", "centre", "power_side", "power_up", "power_down", "segments",
                     "rings"), ()),
    "ring": (_ring, ("top", "bottom", "rings", "segments", "stations"),
             ("blend", "caps", "poles", "bumps", "smooth")),
    "meta": (_meta, ("elements", "resolution", "threshold", "smooth"), ("decimate",)),
    "decal": (_decal, ("half", "segments", "rings", "at"), ("along",)),
}
# A decal follows its host's surface, so it takes no seat and no deforms.
_NO_PLACEMENT = ("decal",)


def read_form(at: _At, shape: str, raw: Any) -> SoftForm:
    """Read one part's `form`. `shape` is the part's own shape and picks the body record."""
    if shape not in _BODIES:
        at.fail(f"the shape {shape!r} has no soft form; soft shapes are {list(FORM_SHAPES)}")
    if isinstance(raw, dict) and "mirror_of" in raw:
        f = _fields(at, raw, ("mirror_of",), ("why",))
        return SoftForm(None, _name(at.key("mirror_of"), f["mirror_of"]), None, (),
                        _why(at, f) if "why" in f else None)
    read_body, required, optional = _BODIES[shape]
    placement = () if shape in _NO_PLACEMENT else ("seat", "deforms")
    f = _fields(at, raw, required, (*optional, *placement, "why"))
    deforms_at = at.key("deforms")
    return SoftForm(
        body=read_body(at, f),
        mirror_of=None,
        seat=read_seat(at.key("seat"), f["seat"]) if "seat" in f else None,
        deforms=tuple(read_deform(deforms_at.index(i), v)
                      for i, v in enumerate(_list(deforms_at, f.get("deforms", [])))),
        why=_why(at, f) if "why" in f else None,
    )
