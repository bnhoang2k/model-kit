"""clips.json: `load_clips` and the readers for a clip, its holds, targets and props."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .fields import (
    _At,
    _box,
    _built,
    _choice,
    _fields,
    _list,
    _name,
    _obj,
    _positive,
    _quat,
    _read_json,
    _ref,
    _text,
    _vec,
    _why,
    _windows,
)
from .records import (
    _BONE,
    FRAME_AXES,
    Clip,
    ClipHold,
    Cross,
    Dangle,
    FloorGear,
    Lift,
    Placement,
    Prop,
    Target,
    frames_problem,
)


def load_clips(path: Path) -> tuple[Clip, ...]:
    """Read a clip set. Every point in it is a ref plus an offset, never a bare coordinate."""
    path = Path(path)
    at = _At(path)
    raw = _fields(at, _read_json(path, "clips"), ("clips",))
    clips_at = at.key("clips")
    clips = tuple(_clip(clips_at.key(name), _name(clips_at, name), value)
                  for name, value in _obj(clips_at, raw["clips"]).items())
    names = {clip.name for clip in clips}
    for clip in clips:
        if clip.mirror_of is not None and clip.mirror_of not in names:
            clips_at.key(clip.name).key("mirror_of").fail(f"names no clip: {clip.mirror_of!r}")
    return clips


def _clip(at: _At, name: str, raw: Any) -> Clip:
    """Read a clip's parts. The clip checks its own rules when it is built (`Clip.validate`);
    only `frames` is checked first, because the windows inside it are read against it."""
    _fields(at, raw, ("frames", "fps", "authoring", "holds"),
            ("seat", "seat_when", "targets", "floor_gear", "props", "mirror_of", "loop",
             "loop_from", "frame", "posture", "lifts"))
    start, end = _vec(at.key("frames"), raw["frames"], 2)
    if (why := frames_problem(raw["frames"])) is not None:
        at.key("frames").fail(why)
    frames = (int(start), int(end))

    props_at = at.key("props")
    props = {_name(props_at, k): _prop(props_at.key(k), k, v, frames)
             for k, v in _obj(props_at, raw.get("props", {})).items()}
    gear_at = at.key("floor_gear")
    gear = FloorGear({_name(gear_at, k): _placement(gear_at.key(k), k, v)
                      for k, v in _obj(gear_at, raw.get("floor_gear", {})).items()})
    holds = tuple(_clip_hold(at.key("holds").index(i), v, frames)
                  for i, v in enumerate(_list(at.key("holds"), raw["holds"])))
    targets = tuple(_target(at.key("targets").index(i), v, frames, props)
                    for i, v in enumerate(_list(at.key("targets"), raw.get("targets", []))))
    lifts = tuple(_lift(at.key("lifts").index(i), v, frames)
                  for i, v in enumerate(_list(at.key("lifts"), raw.get("lifts", []))))
    seat, mirror_of = raw.get("seat"), raw.get("mirror_of")
    return _built(at, lambda: Clip(
        name=name, frames=frames, fps=raw["fps"], authoring=raw["authoring"],
        seat=None if seat is None else _ref(at.key("seat"), seat, bone_ok=False),
        holds=holds, targets=targets, floor_gear=gear, props=props,
        mirror_of=None if mirror_of is None else _name(at.key("mirror_of"), mirror_of),
        loop=raw.get("loop", False), loop_from=raw.get("loop_from"),
        frame=_choice(at.key("frame"), raw.get("frame", FRAME_AXES[0]), FRAME_AXES),
        seat_when=_windows(at.key("seat_when"), raw["seat_when"], frames)
        if "seat_when" in raw else None,
        posture=raw.get("posture"), lifts=lifts,
    ))


def _lift(at: _At, raw: Any, frames: tuple[int, int]) -> Lift:
    _fields(at, raw, ("support", "when", "why"))
    return Lift(support=_name(at.key("support"), raw["support"], _BONE),
                when=_windows(at.key("when"), raw["when"], frames), why=_why(at, raw))


def _clip_hold(at: _At, raw: Any, frames: tuple[int, int]) -> ClipHold:
    _fields(at, raw, ("ref", "who", "when"), ("who_box", "budget", "dangle", "cross"))
    dangle = cross = None
    if raw.get("dangle") is not None:
        d = _fields(at.key("dangle"), raw["dangle"], ("budget",), ("why",))
        dangle = Dangle(_positive(at.key("dangle.budget"), d["budget"]), _why(at.key("dangle"), d))
    if raw.get("cross") is not None:
        c = _fields(at.key("cross"), raw["cross"], (), ("why",))
        cross = Cross(_why(at.key("cross"), c))
    who_box, budget = raw.get("who_box"), raw.get("budget")
    return ClipHold(
        ref=_ref(at.key("ref"), raw["ref"], bone_ok=False),
        who=_name(at.key("who"), raw["who"], _BONE),
        who_box=None if who_box is None else _box(at.key("who_box"), who_box),
        when=_windows(at.key("when"), raw["when"], frames),
        budget=None if budget is None else _positive(at.key("budget"), budget),
        dangle=dangle, cross=cross,
    )


def _target(at: _At, raw: Any, frames: tuple[int, int], props: dict[str, Prop]) -> Target:
    if isinstance(raw, dict) and "point" in raw:
        at.key("point").fail("a key target is a ref plus an offset, never a bare point")
    _fields(at, raw, ("key", "chain", "ref"), ("offset", "why", "baked", "tool"))
    key = raw["key"]
    if isinstance(key, bool) or not isinstance(key, int) or not frames[0] <= key <= frames[1]:
        at.key("key").fail(f"must be a whole frame inside {list(frames)}, got {key!r}")
    ref = _ref(at.key("ref"), raw["ref"], bone_ok=True)
    offset = _vec(at.key("offset"), raw["offset"], 3) if "offset" in raw else (0.0, 0.0, 0.0)
    if ref.is_bone or any(offset):
        why = _why(at, raw)
    else:
        why = None if raw.get("why") is None else _text(at.key("why"), raw["why"])
    tool, baked = raw.get("tool"), raw.get("baked")
    if tool is not None and _name(at.key("tool"), tool) not in props:
        at.key("tool").fail(f"names no prop of this clip: {tool!r}")
    return Target(
        key=key, chain=_name(at.key("chain"), raw["chain"], _BONE), ref=ref, offset=offset,
        why=why, baked=None if baked is None else _vec(at.key("baked"), baked, 3), tool=tool,
    )


def _placement(at: _At, id: str, raw: Any) -> Placement:
    _fields(at, raw, ("piece", "pack", "position", "quaternion"))
    return Placement(
        id=id, piece=_name(at.key("piece"), raw["piece"]), pack=_name(at.key("pack"), raw["pack"]),
        position=_vec(at.key("position"), raw["position"], 3),
        quaternion=_quat(at.key("quaternion"), raw["quaternion"]),
    )


def _prop(at: _At, id: str, raw: Any, frames: tuple[int, int]) -> Prop:
    _fields(at, raw, ("piece", "pack", "bone", "offset_position", "offset_quaternion",
                      "show_frames"))
    return Prop(
        id=id, piece=_name(at.key("piece"), raw["piece"]), pack=_name(at.key("pack"), raw["pack"]),
        bone=_name(at.key("bone"), raw["bone"], _BONE),
        offset_position=_vec(at.key("offset_position"), raw["offset_position"], 3),
        offset_quaternion=_quat(at.key("offset_quaternion"), raw["offset_quaternion"]),
        show_frames=_windows(at.key("show_frames"), raw["show_frames"], frames),
    )
