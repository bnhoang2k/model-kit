"""The stage file: the light bake's stage record, and `load_stage` with `check_stage`.

One file per stage, of any name (`model-kit bake --stage FILE`; bar 05 keeps `stage.json`
beside its scene folder). Snake_case keys, ids as object keys, the glTF frame. Positions, `size`
and `radius_m` are in the layout's units, as the layout's own positions are; only the densities
(`texels_per_m`, `placement_texels_per_m`) are per real metre.

    {
      "stage": 1,
      "layout": "room",                  the layout the bake lights
      "mode": "indirect" | "full",
      "texels_per_m": 256, "samples": 256, "seed": 1,
      "placement_texels_per_m": {"<placement id>": 512},
                                         optional: a still placement's own density, in place
                                         of `texels_per_m`, for a piece whose light needs more
                                         pixels than the rest (as an engine's lightmap scale
                                         per object)
      "layers": ["window", "lamp"],      one bake layer per name; each holds at least one light
      "live": ["<placement id>", ...],   placements that move, so the bake leaves them out
      "lights": {
        "<light id>": {
          "kind": "sun" | "point" | "spot" | "area" | "hemisphere",
          "position": [x, y, z], "layer": "<layer name>",
          "direction": [x, y, z],        sun, spot, area: required; the other kinds refuse it
          "size": [w, h],                area: required; the other kinds refuse it
          "cone_deg": 40, "blend": 0.2,  spot: both required; the other kinds refuse them
                                         (cone_deg is the half angle, 0 to 90 exclusive, from
                                         the direction to the cone edge, as glTF outerConeAngle
                                         and three.js SpotLight.angle; blend is the 0..1 soft-edge
                                         fraction, Blender spot_blend, three.js penumbra)
          "radius_m": 0.05,              point, spot: optional; the other kinds refuse it
          "sky_ground": "sky" | "ground" hemisphere: required; the other kinds refuse it
        }
      },
      "cameras": {
        "<camera id>": {"position": [x, y, z], "target": [x, y, z],
                        "fov_deg": 50, "aspect": 1.5}
      }
    }

A hemisphere light is two layers in effect: `sky` lights the upper half and `ground` the lower,
as the two colours of a three.js `HemisphereLight`.

`load_stage(path)` checks what the file shows: fields, kinds, that each light's layer is named in
`layers` and that each layer holds a light. `check_stage(stage, layouts)` checks the stage
against the scene: the layout exists, every live id is a placement of it, at least one
placement stays still, and each `placement_texels_per_m` id is a still placement of it. A refusal raises `ManifestError` naming the file or layout and the item.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from model_kit.core.errors import ManifestError

from .fields import (
    _At,
    _choice,
    _fields,
    _list,
    _name,
    _names,
    _num,
    _obj,
    _positive,
    _read_json,
    _unique,
    _vec,
)
from .records import Layout

LIGHT_KINDS = ("sun", "point", "spot", "area", "hemisphere")
STAGE_MODES = ("indirect", "full")
SKY_GROUND = ("sky", "ground")

# What each kind takes beyond `kind`, `position` and `layer`: (required, optional).
_KIND_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "sun": (("direction",), ()),
    "point": ((), ("radius_m",)),
    "spot": (("direction", "cone_deg", "blend"), ("radius_m",)),
    "area": (("direction", "size"), ()),
    "hemisphere": (("sky_ground",), ()),
}


@dataclass(frozen=True)
class StageLight:
    id: str
    kind: Literal["sun", "point", "spot", "area", "hemisphere"]
    position: tuple[float, float, float]
    direction: tuple[float, float, float] | None
    size: tuple[float, float] | None
    cone_deg: float | None
    blend: float | None
    radius_m: float | None
    sky_ground: Literal["sky", "ground"] | None
    layer: str


@dataclass(frozen=True)
class StageCamera:
    id: str
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    fov_deg: float
    aspect: float


@dataclass(frozen=True)
class Stage:
    layout: str
    mode: Literal["indirect", "full"]
    lights: dict[str, StageLight]
    layers: tuple[str, ...]
    live: frozenset[str]
    cameras: dict[str, StageCamera]
    texels_per_m: float
    samples: int
    seed: int
    placement_texels_per_m: Mapping[str, float] = field(default_factory=dict)

    def texels_per_m_of(self, placement: str) -> float:
        """The density a still placement bakes at: its own, else the stage's."""
        return self.placement_texels_per_m.get(placement, self.texels_per_m)


def load_stage(path: Path) -> Stage:
    """Read one stage file. Cross-checks against layouts are `check_stage`'s job."""
    path = Path(path)
    at = _At(path)
    raw = _fields(at, _read_json(path, "stage"),
                  ("stage", "layout", "mode", "texels_per_m", "samples", "seed", "layers",
                   "live", "lights", "cameras"), ("placement_texels_per_m",))
    if type(raw["stage"]) is not int or raw["stage"] != 1:
        at.key("stage").fail(f"the file version must be 1, got {raw['stage']!r}")
    layers = _names(at.key("layers"), raw["layers"])
    live_at = at.key("live")
    live = tuple(_name(live_at.index(i), v) for i, v in enumerate(_list(live_at, raw["live"])))
    _unique(live_at, list(live), "live ids")
    lights_at, cams_at = at.key("lights"), at.key("cameras")
    lights = {
        _name(lights_at, key): _light(lights_at.key(key), key, value, layers)
        for key, value in _obj(lights_at, raw["lights"]).items()
    }
    for layer in layers:
        if not any(light.layer == layer for light in lights.values()):
            at.key("layers").fail(f"layer {layer!r} has no light on it")
    cameras = {
        _name(cams_at, key): _camera(cams_at.key(key), key, value)
        for key, value in _obj(cams_at, raw["cameras"]).items()
    }
    if not cameras:
        cams_at.fail("must name at least one camera: a stage with none sees no face to unwrap")
    own_at = at.key("placement_texels_per_m")
    own = {_name(own_at, key): _positive(own_at.key(key), value)
           for key, value in _obj(own_at, raw.get("placement_texels_per_m", {})).items()}
    return Stage(layout=_name(at.key("layout"), raw["layout"]),
                 mode=_choice(at.key("mode"), raw["mode"], STAGE_MODES),
                 lights=lights, layers=layers, live=frozenset(live), cameras=cameras,
                 texels_per_m=_positive(at.key("texels_per_m"), raw["texels_per_m"]),
                 samples=_count(at.key("samples"), raw["samples"], minimum=1),
                 seed=_count(at.key("seed"), raw["seed"], minimum=0),
                 placement_texels_per_m=own)


def _count(at: _At, raw: Any, *, minimum: int) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        at.fail(f"must be a whole number of at least {minimum}, got {raw!r}")
    return raw


def _light(at: _At, id: str, raw: Any, layers: tuple[str, ...]) -> StageLight:
    base = ("kind", "position", "layer")
    # Read `kind` first, so the field list can follow it.
    kind_raw = _obj(at, raw).get("kind")
    if "kind" not in raw:
        at.fail("missing field 'kind'")
    kind = _choice(at.key("kind"), kind_raw, LIGHT_KINDS)
    required, optional = _KIND_FIELDS[kind]
    _fields(at, raw, base + required, optional)
    layer = _name(at.key("layer"), raw["layer"])
    if layer not in layers:
        at.key("layer").fail(f"names layer {layer!r}, which `layers` lacks (it has "
                             f"{list(layers)})")
    direction = None
    if "direction" in raw:
        direction = _vec(at.key("direction"), raw["direction"], 3, nonzero=True)
    size = None
    if "size" in raw:
        size = _vec(at.key("size"), raw["size"], 2)
        if min(size) <= 0:
            at.key("size").fail(f"must be above 0 on both sides, got {list(size)}")
    cone_deg = None
    if "cone_deg" in raw:
        cone_deg = _num(at.key("cone_deg"), raw["cone_deg"])
        if not 0 < cone_deg < 90:
            at.key("cone_deg").fail(f"must be above 0 and below 90 (the half angle), got {raw['cone_deg']!r}")
    blend = None
    if "blend" in raw:
        blend = _num(at.key("blend"), raw["blend"])
        if not 0 <= blend <= 1:
            at.key("blend").fail(f"must be from 0 to 1, got {raw['blend']!r}")
    radius_m = None
    if "radius_m" in raw:
        radius_m = _positive(at.key("radius_m"), raw["radius_m"])
    sky_ground = None
    if "sky_ground" in raw:
        sky_ground = _choice(at.key("sky_ground"), raw["sky_ground"], SKY_GROUND)
    return StageLight(id=id, kind=kind, position=_vec(at.key("position"), raw["position"], 3),
                      direction=direction, size=size, cone_deg=cone_deg, blend=blend,
                      radius_m=radius_m, sky_ground=sky_ground, layer=layer)


def _camera(at: _At, id: str, raw: Any) -> StageCamera:
    _fields(at, raw, ("position", "target", "fov_deg", "aspect"))
    position = _vec(at.key("position"), raw["position"], 3)
    target = _vec(at.key("target"), raw["target"], 3)
    if position == target:
        at.key("target").fail("must differ from `position`")
    fov = _positive(at.key("fov_deg"), raw["fov_deg"])
    if fov >= 180:
        at.key("fov_deg").fail(f"must be below 180, got {raw['fov_deg']!r}")
    aspect = _positive(at.key("aspect"), raw["aspect"])
    return StageCamera(id=id, position=position, target=target, fov_deg=fov, aspect=aspect)


def check_stage(stage: Stage, layouts: Mapping[str, Layout]) -> None:
    """The stage's layout exists, every live id is a placement of it, one placement is still,
    and every `placement_texels_per_m` id is a still placement of it."""
    layout = layouts.get(stage.layout)
    if layout is None:
        raise ManifestError(f"stage: names layout {stage.layout!r}, which is not loaded "
                            f"(loaded: {sorted(layouts)})")
    for pid in sorted(stage.live):
        if pid not in layout.placements:
            raise ManifestError(f"stage: live names placement {pid!r}, which layout "
                                f"{layout.name!r} lacks (it has {sorted(layout.placements)})")
    if not set(layout.placements) - stage.live:
        raise ManifestError(f"stage: layout {layout.name!r} would have no still piece: every "
                            f"placement is live, so there is nothing to bake")
    for pid in sorted(stage.placement_texels_per_m):
        if pid not in layout.placements:
            raise ManifestError(f"stage: placement_texels_per_m names placement {pid!r}, which "
                                f"layout {layout.name!r} lacks (it has {sorted(layout.placements)})")
        if pid in stage.live:
            raise ManifestError(f"stage: placement_texels_per_m names placement {pid!r}, which is "
                                "live: the bake leaves it out, so it has no lightmap to size")
