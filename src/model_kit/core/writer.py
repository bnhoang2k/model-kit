"""Manifests back to JSON, and `write_pack`: the one way a pack reaches disk.

`save_scene` and a build's `PackBuilder.finish` both write packs through `write_pack`.

`pack_json` and `clips_json` give the JSON that `load_pack` and `load_clips` read. A part
never carries bones or a bind mode; a rigged piece's binds live only under `rig.bind`.
Optional fields that are None are left out, and so are clip fields at their default
(`loop`, `loop_from`, `frame`, `lifts`). `layout_json` and `write_layout` do the same for a layout.
numpy and the standard library only.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path

from model_kit.core.errors import ManifestError
from model_kit.core.manifest import (
    FINISHES,
    FRAME_AXES,
    HELD,
    Clip,
    ClipHold,
    Connector,
    End,
    Layout,
    Listed,
    Pack,
    Piece,
    RigSpec,
    Target,
    load_layout,
    load_pack,
)
from model_kit.core.manifest.forms import (
    BlobForm,
    DecalForm,
    Falloff,
    MetaForm,
    Push,
    RingForm,
    Smooth,
    SoftForm,
)
from model_kit.core.units import Style


def _vec(v) -> list[float]:
    return [float(x) for x in v]


def _box(box) -> list[list[float]]:
    return [_vec(box[0]), _vec(box[1])]


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _smooth_json(s: Smooth) -> dict:
    return _drop_none({"passes": s.passes, "factor": s.factor, "below": s.below})


def _deform_json(d) -> dict:
    if isinstance(d, Push):
        return {"kind": "push", "axis": d.axis, "amount": [_vec(row) for row in d.amount],
                "along": d.along, "ramp_from": d.ramp_from, "ramp_to": d.ramp_to,
                "ramp_max": d.ramp_max}
    return {"kind": "flatten", "axis": d.axis, "start": d.start, "end": d.end}


def _body_json(body) -> dict:
    if isinstance(body, DecalForm):
        return {"half": _vec(body.half), "segments": body.segments, "rings": body.rings,
                "at": _vec(body.at), "along": _vec(body.along)}
    if isinstance(body, BlobForm):
        return {"half": _vec(body.half), "centre": _vec(body.centre),
                "power_side": body.power_side, "power_up": body.power_up,
                "power_down": body.power_down, "segments": body.segments, "rings": body.rings}
    if isinstance(body, RingForm):
        out = {
            "top": _vec(body.top), "bottom": _vec(body.bottom), "rings": body.rings,
            "segments": body.segments, "blend": body.blend,
            "stations": [{"u": s.u, "half_side": s.half_side, "half_depth": s.half_depth,
                          "off_side": s.off_side, "off_depth": s.off_depth,
                          "inner_power": s.inner_power} for s in body.stations],
        }
        if body.caps:
            out["caps"] = {end: _drop_none({"kind": c.kind, "length": c.length, "power": c.power})
                           for end, c in body.caps.items()}
        if body.poles:
            out["poles"] = {end: _vec(v) for end, v in body.poles.items()}
        if body.bumps:
            out["bumps"] = [{"u": b.u, "u_width": b.u_width, "angle": b.angle,
                             "angle_width": b.angle_width, "height": b.height, "lift": b.lift}
                            for b in body.bumps]
        if body.smooth is not None:
            out["smooth"] = _smooth_json(body.smooth)
        return out
    if not isinstance(body, MetaForm):
        raise ManifestError(f"unknown form body type {type(body).__name__!r}; the writer "
                            "knows BlobForm, RingForm, MetaForm and DecalForm")
    return _drop_none({
        "elements": [{"kind": e.kind, "centre": _vec(e.centre), "half": _vec(e.half)}
                     for e in body.elements],
        "resolution": body.resolution, "threshold": body.threshold,
        "smooth": _smooth_json(body.smooth), "decimate": body.decimate,
    })


def _form_json(form: SoftForm) -> dict:
    if form.mirror_of is not None:
        return _drop_none({"mirror_of": form.mirror_of, "why": form.why})
    out = _body_json(form.body)
    if form.seat is not None:
        out["seat"] = {"on": form.seat.on, "at": _vec(form.seat.at), "sink": form.seat.sink}
    if form.deforms:
        out["deforms"] = [_deform_json(d) for d in form.deforms]
    return {**out, **_drop_none({"why": form.why})}


def _falloff_json(f: Falloff) -> dict:
    return {"along": f.along, "start": f.start, "end": f.end,
            "bands": [_vec(b) for b in f.bands]}


def _bone_json(b) -> dict:
    out = {"name": b.name, "parent": b.parent, "length": b.length}
    if b.rest is not None:
        out["rest"] = _drop_none({"head": _vec(b.rest.head), "tail": _vec(b.rest.tail),
                                  "roll": b.rest.roll, "deform": b.rest.deform,
                                  "why": b.rest.why})
    return out


def _bind_json(b) -> dict:
    out = {"bones": list(b.bones), "mode": b.mode}
    if b.falloff is not None:
        out["falloff"] = _falloff_json(b.falloff)
    return out


def _part_json(p) -> dict:
    out = {"name": p.name, "listed": p.listed, "shape": p.shape, "on": p.on, "group": p.group,
           "role": p.role}
    if p.form is not None:
        out["form"] = _form_json(p.form)
    if p.fastener:
        out["fastener"] = True
    if p.retains:
        out["retains"] = list(p.retains)
    if p.rests_on:
        out["rests_on"] = list(p.rests_on)
    return out


def _rig_json(rig: RigSpec) -> dict:
    out = {
        "armature": rig.armature,
        "bones": [_bone_json(b) for b in rig.bones],
        "bind": {part: _bind_json(b) for part, b in rig.bind.items()},
        "contact_groups": {k: list(v) for k, v in rig.contact_groups.items()},
        "chains": {name: {
            "root": c.root, "bones": list(c.bones), "effector": c.effector,
            "effector_radius": c.effector_radius, "dofs": dict(c.dofs),
            "limits": {k: _vec(v) for k, v in c.limits.items()},
            "effector_group": c.effector_group,
        } for name, c in rig.chains.items()},
    }
    if rig.sit is not None:
        out["sit"] = {"below_seat": rig.sit.below_seat, "forward": rig.sit.forward,
                      **({"past_edge": list(rig.sit.past_edge)} if rig.sit.past_edge else {})}
    if rig.mirror is not None:
        out["mirror"] = {"pairs": [list(p) for p in rig.mirror.pairs], "axis": rig.mirror.axis}
    out["outline_modifiers"] = list(rig.outline_modifiers)
    if rig.avoid:
        out["avoid"] = [{"id": a.id, "bone": a.bone, "kind": a.kind, "centre": _vec(a.centre),
                         "half": _vec(a.half), "why": a.why} for a in rig.avoid]
    if rig.supports:
        out["supports"] = list(rig.supports)
    if rig.joints:
        out["joints"] = {bone: {"range_deg": {a: _vec(r) for a, r in j.range_deg.items()},
                                "why": j.why} for bone, j in rig.joints.items()}
    if rig.root is not None:
        out["root"] = rig.root
    return out


def _end_json(e: End) -> dict:
    return _drop_none({"point": _vec(e.point), "layout": e.layout, "host": str(e.host),
                       "plug": e.plug})


def _connector_json(c: Connector) -> dict:
    return {"id": c.id, "part": c.part, "ends": [_end_json(e) for e in c.ends], "cap": c.cap,
            "why": c.why}


def _listed_json(x: Listed) -> dict:
    out = _drop_none({"id": x.id, "text": x.text, "source": x.source,
                      "min_thickness": x.min_thickness, "owner": x.owner})
    if x.finish != FINISHES[0]:
        out["finish"] = x.finish
    if x.held != HELD[0]:
        out["held"] = x.held
    return out


def _piece_json(p: Piece) -> dict:
    out = {
        "frame": {"axes": p.frame.axes, "origin": p.frame.origin},
        "recipe": {"kind": p.recipe.kind, "source": p.recipe.source},
        "listed": [_listed_json(x) for x in p.listed],
        "parts": [_part_json(x) for x in p.parts],
        "uses": [{"id": x.id, "point": _vec(x.point), "facing": _vec(x.facing),
                  "posture": x.posture} for x in p.uses],
        "holds": [_drop_none({"id": x.id, "use": x.use, "side": x.side,
                              "surface": list(x.surface),
                              "sub_box": None if x.sub_box is None else _box(x.sub_box),
                              "point": _vec(x.point), "off": _vec(x.off), "grip": x.grip})
                  for x in p.holds],
        "seats": [{"id": x.id, "use": x.use, "point": _vec(x.point), "facing": _vec(x.facing),
                   "surface": list(x.surface)} for x in p.seats],
        "mounts": [_drop_none({"id": x.id, "kind": x.kind, "point": _vec(x.point),
                               "axis": _vec(x.axis), "takes": x.takes,
                               "fits": None if x.fits is None else list(x.fits),
                               "part": x.part})
                   for x in p.mounts],
        "done": p.done,
        "size_class": p.size_class,
        "measured": dict(p.measured),
    }
    if p.rig is not None:
        out["rig"] = _rig_json(p.rig)
    if p.connectors:
        out["connectors"] = [_connector_json(c) for c in p.connectors]
    if p.view is not None:
        out["view"] = {"nearest_m": p.view.nearest_m, "why": p.view.why}
        if p.view.hidden:
            out["view"] |= {"hidden": list(p.view.hidden), "hidden_why": p.view.hidden_why}
        if p.view.farthest_m is not None:
            out["view"] |= {"farthest_m": p.view.farthest_m,
                            "farthest_why": p.view.farthest_why}
    if not p.outlined:
        out |= {"outlined": False, "outlined_why": p.outlined_why}
    if p.cards_of:
        out["cards_of"] = p.cards_of
    if p.keep_off:
        out["keep_off"] = [{"id": k.id, "point": _vec(k.point), "radius": k.radius,
                            "why": k.why} for k in p.keep_off]
    return out


def pack_json(pack: Pack) -> dict:
    """A pack as the JSON `load_pack` reads."""
    return _drop_none({"pack": pack.name, "units_per_metre": pack.units_per_metre,
                       "spacing": pack.spacing,
                       "pieces": {name: _piece_json(p) for name, p in pack.pieces.items()}})


def _windows(windows) -> list[list[float]]:
    return [_vec(w) for w in windows]


def _hold_json(h: ClipHold) -> dict:
    return _drop_none({
        "ref": str(h.ref), "who": h.who,
        "who_box": None if h.who_box is None else _box(h.who_box),
        "when": _windows(h.when), "budget": h.budget,
        "dangle": None if h.dangle is None else {"budget": h.dangle.budget, "why": h.dangle.why},
        "cross": None if h.cross is None else {"why": h.cross.why},
    })


def _target_json(t: Target) -> dict:
    return _drop_none({
        "key": t.key, "chain": t.chain, "ref": str(t.ref), "offset": _vec(t.offset),
        "why": t.why, "baked": None if t.baked is None else _vec(t.baked), "tool": t.tool,
    })


def _clip_json(c: Clip) -> dict:
    return _drop_none({
        "frames": [int(c.frames[0]), int(c.frames[1])],
        "fps": c.fps,
        "authoring": c.authoring,
        "seat": None if c.seat is None else str(c.seat),
        "holds": [_hold_json(h) for h in c.holds],
        "targets": [_target_json(t) for t in c.targets],
        "floor_gear": {k: {"piece": p.piece, "pack": p.pack, "position": _vec(p.position),
                           "quaternion": _vec(p.quaternion)}
                       for k, p in c.floor_gear.pieces.items()},
        "props": {k: {"piece": p.piece, "pack": p.pack, "bone": p.bone,
                      "offset_position": _vec(p.offset_position),
                      "offset_quaternion": _vec(p.offset_quaternion),
                      "show_frames": _windows(p.show_frames)}
                  for k, p in c.props.items()},
        "mirror_of": c.mirror_of,
        "loop": True if c.loop else None,
        "loop_from": c.loop_from,
        "frame": None if c.frame == FRAME_AXES[0] else c.frame,
        "seat_when": None if c.seat_when is None else _windows(c.seat_when),
        "posture": c.posture,
        "lifts": [{"support": x.support, "when": _windows(x.when), "why": x.why}
                  for x in c.lifts] or None,
    })


def clips_json(clips: dict[str, Clip]) -> dict:
    """A clip set as the JSON `load_clips` reads."""
    return {"clips": {name: _clip_json(c) for name, c in clips.items()}}



def _check_measured(pack: Pack, path: Path) -> None:
    """Each measured value must come back from JSON as itself (lists, not tuples; no NaN)."""
    for name, piece in pack.pieces.items():
        for key, value in piece.measured.items():
            try:
                same = json.loads(json.dumps(value, allow_nan=False)) == value
            except (TypeError, ValueError) as exc:
                raise ManifestError(f"{path}: pieces.{name}.measured.{key}: JSON cannot hold "
                                    f"{value!r}: {exc}") from None
            if not same:
                raise ManifestError(f"{path}: pieces.{name}.measured.{key}: {value!r} does not come "
                                    "back from JSON as itself; use lists, dicts, str, "
                                    "numbers and bools")


def _first_difference(wrote: Pack, back: Pack) -> str:
    """Where two packs first differ, as `<field path>: wrote ..., read back ...`."""
    for field in ("name", "units_per_metre"):
        if getattr(wrote, field) != getattr(back, field):
            return f"{field}: wrote {getattr(wrote, field)!r}, read back {getattr(back, field)!r}"
    if list(wrote.pieces) != list(back.pieces):
        return f"pieces: wrote {list(wrote.pieces)}, read back {list(back.pieces)}"
    for name, piece in wrote.pieces.items():
        other = back.pieces[name]
        for field in dataclasses.fields(piece):
            a, b = getattr(piece, field.name), getattr(other, field.name)
            if a != b:
                return f"pieces.{name}.{field.name}: wrote {a!r}, read back {b!r}"
    return "the records differ"


def pack_text(pack: Pack, path: Path) -> str:
    """The JSON text of `pack`, checked: it is read back with `load_pack` and must give the same
    records. Raises a `ManifestError` naming `path` and the first field that differs. Writes
    nothing outside a private staging folder; `write_pack` and a build's pre-gate check use it.
    """
    path = Path(path)
    for name, piece in pack.pieces.items():
        if name != piece.name:
            raise ManifestError(f"{path}: pieces[{name!r}] holds piece {piece.name!r}")
    _check_measured(pack, path)
    try:
        text = json.dumps(pack_json(pack), indent=2, allow_nan=False) + "\n"
    except ValueError as exc:
        raise ManifestError(f"{path}: the pack holds a number JSON cannot: {exc}") from None
    style = Style(units_per_metre=pack.units_per_metre, page_camera={}, dials={})
    with tempfile.TemporaryDirectory(prefix="model-kit-pack-") as tmp:
        staged = Path(tmp) / path.name
        staged.write_text(text)
        try:
            back = load_pack(staged, style)
        except ManifestError as exc:  # name the user's file, not the deleted staging copy
            raise type(exc)(str(exc).replace(str(staged), str(path))) from None
    if back != pack:
        raise ManifestError(f"{path}: the pack does not read back as the same records: "
                            f"{_first_difference(pack, back)}")
    return text


def _atomic_write(path: Path, text: str) -> None:
    """Replace `path` with `text` in one step, with the mode any new file gets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, part = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(part, 0o666 & ~mask)  # mkstemp makes 0600; give the file the usual mode
        os.replace(part, path)
    except BaseException:
        Path(part).unlink(missing_ok=True)
        raise


def write_pack(pack: Pack, path: Path) -> Path:
    """Write `pack` to `path` and return `path`. `load_pack(write_pack(p, f), style) == p`.

    The text comes from `pack_text`, so a pack the loader refuses, or one that reads back as
    different records, raises and leaves `path` as it was. Only then is `path` replaced, in
    one step.
    """
    path = Path(path)
    _atomic_write(path, pack_text(pack, path))
    return path


def layout_json(layout: Layout) -> dict:
    """A layout as the JSON `load_layout` reads (the shape is in `manifest/layouts.py`)."""
    return {
        "layout": layout.name,
        "frame": layout.frame,
        "placements": {pid: _drop_none({
            "pack": p.pack, "piece": p.piece, "position": _vec(p.position),
            "quaternion": _vec(p.quaternion), "on": None if p.on is None else str(p.on),
            "rests_on": p.rests_on,
        }) for pid, p in layout.placements.items()},
        "stands": {sid: {"position": _vec(s.position), "facing": _vec(s.facing),
                         "posture": s.posture, "clip": s.clip}
                   for sid, s in layout.stands.items()},
    }


def layout_text(layout: Layout, path: Path) -> str:
    """The JSON text of `layout`, checked: read back with `load_layout` it gives the same
    records. Raises a `ManifestError` naming `path` otherwise. Writes nothing outside a
    private staging folder."""
    path = Path(path)
    for key, placed in layout.placements.items():
        if key != placed.id:
            raise ManifestError(f"{path}: placements[{key!r}] holds placement {placed.id!r}")
    for key, stand in layout.stands.items():
        if key != stand.id:
            raise ManifestError(f"{path}: stands[{key!r}] holds stand {stand.id!r}")
    try:
        text = json.dumps(layout_json(layout), indent=2, allow_nan=False) + "\n"
    except ValueError as exc:
        raise ManifestError(f"{path}: the layout holds a number JSON cannot: {exc}") from None
    with tempfile.TemporaryDirectory(prefix="model-kit-layout-") as tmp:
        staged = Path(tmp) / path.name
        staged.write_text(text)
        try:
            back = load_layout(staged)
        except ManifestError as exc:  # name the user's file, not the deleted staging copy
            raise type(exc)(str(exc).replace(str(staged), str(path))) from None
    if back != layout:
        raise ManifestError(f"{path}: the layout does not read back as the same records")
    return text


def write_layout(layout: Layout, path: Path) -> Path:
    """Write `layout` to `path` and return `path`. `load_layout(write_layout(l, f)) == l`.

    A layout the loader refuses leaves `path` as it was. Whether its placements fit the packs
    is `load_layouts`'s check, made when the folder is read.
    """
    path = Path(path)
    _atomic_write(path, layout_text(layout, path))
    return path
