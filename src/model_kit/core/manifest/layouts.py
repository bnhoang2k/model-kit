"""layouts/<pack>.json: where a pack's pieces stand, and `load_layouts` with its cross-checks.

One file per pack, written by that pack's builder. The shape follows pack.json and clips.json:
snake_case keys, ids as object keys, refs as the ref strings `Ref.parse` reads.

    {
      "layout": "shop",                  the layout's name; no two files may use one name
      "frame": "gltf",                   one of FRAME_AXES; a stage layout is in its stage
                                         frame with the stage's core at the origin
      "placements": {
        "<placement id>": {
          "pack": "shop", "piece": "Lamp",
          "position": [x, y, z], "quaternion": [x, y, z, w],
          "on": "shop:<host placement id>.<mount id>",       optional
          "rests_on": "<placement id>"                       optional, not with "on"
        }
      },
      "stands": {
        "<stand id>": {
          "position": [x, y, z], "facing": [x, 0, z],         unit length, y 0
          "posture": "stand" | "sit", "clip": "<clip name>"
        }
      }
    }

`load_layout(path)` reads one file and checks only what the file can show. `load_layouts(dir,
packs)` reads every `*.json` in a folder and then checks each layout against the packs: the
placement's pack and piece exist; a placement with `on` names a host placement of the same
layout, its piece and its mount, and the guest piece has a part named in the mount's `takes`;
and every connector end names a loaded layout and a host placement in it. A placement with `on`
keeps its own position on purpose, so a check can compare the two. A placement with `rests_on`
names another placement of the same layout, not itself, and has no `on`; `load_layout` checks
that, as the file shows it all. Every refusal raises a `ManifestError` that names the file or
piece and the item, never a default point.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from model_kit.core.errors import ManifestError
from model_kit.core.tolerances import UNIT_LENGTH_SLACK

from .fields import (
    _At,
    _choice,
    _fields,
    _name,
    _obj,
    _quat,
    _read_json,
    _ref,
    _text,
    _vec,
)
from .records import (
    FRAME_AXES,
    POSTURES,
    Clip,
    End,
    Layout,
    LayoutPlacement,
    Pack,
    Stand,
)


def load_layout(path: Path) -> Layout:
    """Read one layout file. Cross-checks against packs are `load_layouts`'s job."""
    path = Path(path)
    at = _At(path)
    raw = _fields(at, _read_json(path, "layout"), ("layout", "frame", "placements", "stands"))
    placements_at, stands_at = at.key("placements"), at.key("stands")
    placements = {
        _name(placements_at, key): _placement(placements_at.key(key), key, value)
        for key, value in _obj(placements_at, raw["placements"]).items()
    }
    for key, placed in placements.items():
        if placed.rests_on == key:
            placements_at.key(key).key("rests_on").fail("a placement cannot rest on itself")
        if placed.rests_on is not None and placed.rests_on not in placements:
            placements_at.key(key).key("rests_on").fail(
                f"names no placement of this layout: {placed.rests_on!r}")
    stands = {
        _name(stands_at, key): _stand(stands_at.key(key), key, value)
        for key, value in _obj(stands_at, raw["stands"]).items()
    }
    return Layout(name=_name(at.key("layout"), raw["layout"]),
                  frame=_choice(at.key("frame"), raw["frame"], FRAME_AXES),
                  placements=placements, stands=stands)


def _placement(at: _At, id: str, raw: Any) -> LayoutPlacement:
    _fields(at, raw, ("pack", "piece", "position", "quaternion"), ("on", "rests_on"))
    on, rests_on = raw.get("on"), raw.get("rests_on")
    if on is not None and rests_on is not None:
        at.key("rests_on").fail("a placement is plugged in (`on`) or rests on another "
                                "(`rests_on`), not both")
    return LayoutPlacement(
        id=id, pack=_name(at.key("pack"), raw["pack"]), piece=_name(at.key("piece"), raw["piece"]),
        position=_vec(at.key("position"), raw["position"], 3),
        quaternion=_quat(at.key("quaternion"), raw["quaternion"]),
        on=None if on is None else _ref(at.key("on"), on, bone_ok=False),
        rests_on=None if rests_on is None else _name(at.key("rests_on"), rests_on),
    )


def _stand(at: _At, id: str, raw: Any) -> Stand:
    _fields(at, raw, ("position", "facing", "posture", "clip"))
    facing = _vec(at.key("facing"), raw["facing"], 3)
    if facing[1] != 0 or not math.isclose(math.hypot(*facing), 1.0, abs_tol=UNIT_LENGTH_SLACK):
        at.key("facing").fail(f"must be a unit vector with y 0 (horizontal), got {list(facing)}")
    return Stand(id=id, position=_vec(at.key("position"), raw["position"], 3), facing=facing,
                 posture=_choice(at.key("posture"), raw["posture"], POSTURES),
                 clip=_text(at.key("clip"), raw["clip"]))


def load_layouts(dir: Path, packs: Mapping[str, Pack]) -> dict[str, Layout]:
    """Read every `*.json` file in `dir`, keyed by layout name, and check it against `packs`.

    A layout name used by two files raises, naming both. A file or folder whose name starts
    with "." is ignored.
    """
    dir = Path(dir)
    if not dir.is_dir():
        raise ManifestError(f"{dir}: the layouts folder does not exist")
    layouts: dict[str, Layout] = {}
    files: dict[str, Path] = {}
    for path in sorted(dir.iterdir()):
        if path.name.startswith("."):
            continue
        if not (path.is_file() and path.suffix == ".json"):
            raise ManifestError(f"{path}: not a layout file; layouts live at layouts/<pack>.json")
        layout = load_layout(path)
        if layout.name in layouts:
            raise ManifestError(f"{path}: layout name {layout.name!r} is already used by "
                                f"{files[layout.name]}")
        layouts[layout.name], files[layout.name] = layout, path
    for name, layout in layouts.items():
        _check_layout(files[name], layout, packs)
    check_connectors(packs, layouts)
    return layouts


def _check_layout(path: Path, layout: Layout, packs: Mapping[str, Pack]) -> None:
    for pid, placed in layout.placements.items():
        where = f"{path}: placements.{pid}"
        pack = packs.get(placed.pack)
        if pack is None:
            raise ManifestError(f"{where}: names pack {placed.pack!r}, which is not loaded "
                                f"(loaded: {sorted(packs)})")
        if placed.piece not in pack.pieces:
            raise ManifestError(f"{where}: pack {placed.pack!r} has no piece {placed.piece!r}")
    for pid, placed in layout.placements.items():
        if placed.on is not None:
            _check_on(f"{path}: placements.{pid}", layout, packs, placed)


def _check_on(where: str, layout: Layout, packs: Mapping[str, Pack],
              placed: LayoutPlacement) -> None:
    on = placed.on
    if on.piece == placed.id:
        raise ManifestError(f"{where}.on: {on} plugs placement {placed.id!r} into itself")
    host = layout.placements.get(on.piece)
    if host is None:
        raise ManifestError(f"{where}.on: {on} names no placement {on.piece!r} of layout "
                            f"{layout.name!r} (it has {sorted(layout.placements)})")
    if host.pack != on.pack:
        raise ManifestError(f"{where}.on: {on} names pack {on.pack!r}, but placement "
                            f"{host.id!r} is of pack {host.pack!r}")
    host_piece = packs[host.pack].pieces[host.piece]
    mount = next((m for m in host_piece.mounts if m.id == on.id), None)
    if mount is None:
        raise ManifestError(f"{where}.on: {on}: piece {host.pack}:{host.piece} publishes no "
                            f"mount {on.id!r} (it has {[m.id for m in host_piece.mounts]})")
    guest = packs[placed.pack].pieces[placed.piece]
    if mount.takes not in {p.name for p in guest.parts}:
        raise ManifestError(f"{where}.on: {on} takes a part named {mount.takes!r}, but "
                            f"{placed.pack}:{placed.piece} has none "
                            f"(its parts: {[p.name for p in guest.parts]})")


def check_connectors(packs: Mapping[str, Pack], layouts: Mapping[str, Layout]) -> None:
    """Every connector end that is not on `self` names a loaded layout, a placement of it, and
    a mount of that placement's piece. Needs packs and layouts, so `load_layouts` makes it
    (and `load_scene` when it has no layouts folder). Each lookup is guarded: a placement
    of a pack or piece that is not loaded raises, naming the connector."""
    for pack in packs.values():
        for piece in pack.pieces.values():
            for conn in piece.connectors:
                for i, end in enumerate(conn.ends):
                    if not end.host.is_self:
                        _check_end(f"{pack.name}:{piece.name} connector {conn.id!r} ends[{i}]",
                                   end, packs, layouts)


def _check_end(where: str, end: End, packs: Mapping[str, Pack],
               layouts: Mapping[str, Layout]) -> None:
    layout = layouts.get(end.layout)
    if layout is None:
        raise ManifestError(f"{where}: names layout {end.layout!r}, which is not "
                            f"loaded (loaded: {sorted(layouts)})")
    host = layout.placements.get(end.host.piece)
    if host is None:
        raise ManifestError(f"{where}: host {end.host} names no placement "
                            f"{end.host.piece!r} in layout {layout.name!r}")
    if host.pack != end.host.pack:
        raise ManifestError(f"{where}: host {end.host} names pack {end.host.pack!r}, but "
                            f"placement {host.id!r} is of pack {host.pack!r}")
    host_pack = packs.get(host.pack)
    host_piece = None if host_pack is None else host_pack.pieces.get(host.piece)
    if host_piece is None:
        raise ManifestError(f"{where}: host {end.host}: placement {host.id!r} is of "
                            f"{host.pack}:{host.piece}, which is not a loaded piece")
    if end.host.id not in {m.id for m in host_piece.mounts}:
        raise ManifestError(f"{where}: host {end.host}: piece {host.pack}:{host.piece} "
                            f"publishes no mount {end.host.id!r}")


def check_stands(layouts: Mapping[str, Layout], clips: Mapping[str, Clip]) -> None:
    """Every stand names a clip of the scene. Needs layouts and clips, so `load_scene` makes it."""
    for layout in layouts.values():
        for stand in layout.stands.values():
            if stand.clip not in clips:
                raise ManifestError(f"layout {layout.name!r}: stand {stand.id!r} names clip "
                                    f"{stand.clip!r}, which clips.json lacks "
                                    f"(it has {sorted(clips)})")
