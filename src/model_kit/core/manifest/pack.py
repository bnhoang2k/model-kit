"""pack.json: `load_pack` and the readers for a piece and the items it lists."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_kit.core.units import Style

from .fields import (
    _At,
    _box,
    _built,
    _choice,
    _fields,
    _list,
    _name,
    _names,
    _num,
    _obj,
    _positive,
    _read_json,
    _ref,
    _text,
    _unique,
    _vec,
    _why,
)
from .forms import read_form
from .records import (
    CABLE_SHAPE,
    FINISHES,
    FRAME_AXES,
    HELD,
    MEASURED_KEYS,
    MOUNT_KINDS,
    POSTURES,
    PRINTED_ROLES,
    RECIPES,
    RIG_PACK,
    SHAPES,
    SIDES,
    SIZE_CLASSES,
    Connector,
    End,
    Frame,
    Hold,
    KeepOff,
    Listed,
    Mount,
    Pack,
    Part,
    Piece,
    Recipe,
    Seat,
    Use,
    View,
)
from .rig import _rig


def load_pack(path: Path, style: Style) -> Pack:
    """Read a pack. Its units must be given and must equal the style's."""
    path = Path(path)
    at = _At(path)
    raw = _fields(at, _read_json(path, "pack"), ("pack", "units_per_metre", "pieces"),
                  ("spacing",))
    name = _name(at.key("pack"), raw["pack"])
    if name == RIG_PACK:
        at.key("pack").fail(f"{RIG_PACK!r} is reserved for bone refs")
    units = _positive(at.key("units_per_metre"), raw["units_per_metre"])
    if units != style.units_per_metre:
        at.key("units_per_metre").fail(
            f"{units} differs from the style's units_per_metre {style.units_per_metre}"
        )
    pieces_at = at.key("pieces")
    pieces = {
        _name(pieces_at, key): _piece(pieces_at.key(key), key, value)
        for key, value in _obj(pieces_at, raw["pieces"]).items()
    }
    _check_camera(at, pieces, style)
    spacing = raw.get("spacing")
    return Pack(name=name, units_per_metre=units, pieces=pieces,
                spacing=None if spacing is None else _positive(at.key("spacing"), spacing))


def _check_camera(at: _At, pieces: dict[str, Piece], style: Style) -> None:
    """Where the style and the pack meet: every viewing distance lies inside the style's camera.
    A piece seen nearer than the near plane (`near_m`) or farther than the far plane (`far_m`), at
    its nearest or its farthest, is out of the depth buffer, so the load raises. A style with no depth camera checks nothing."""
    camera = style.depth_camera
    if camera is None:
        return
    for name, piece in pieces.items():
        view = piece.view
        if view is None:
            continue
        here = at.key("pieces").key(name).key("view")
        if view.nearest_m > camera.far_m:
            here.key("nearest_m").fail(
                f"{view.nearest_m} m is past the style's camera far plane "
                f"(far_m {camera.far_m} m)")
        if view.nearest_m < camera.near_m:
            here.key("nearest_m").fail(
                f"{view.nearest_m} m is nearer than the style's camera near plane "
                f"(near_m {camera.near_m} m)")
        if view.farthest_m is not None and view.farthest_m > camera.far_m:
            here.key("farthest_m").fail(
                f"{view.farthest_m} m is past the style's camera far plane "
                f"(far_m {camera.far_m} m)")


_PIECE_FIELDS = ("frame", "recipe", "listed", "parts", "uses", "holds", "seats", "mounts",
                 "done", "size_class", "measured")


def _piece(at: _At, name: str, raw: Any) -> Piece:
    _fields(at, raw, _PIECE_FIELDS,
            ("rig", "connectors", "keep_off", "view", "outlined", "outlined_why"))
    f = _fields(at.key("frame"), raw["frame"], ("axes", "origin"))
    frame = Frame(_choice(at.key("frame.axes"), f["axes"], FRAME_AXES),
                  _text(at.key("frame.origin"), f["origin"]))
    r = _fields(at.key("recipe"), raw["recipe"], ("kind", "source"))
    recipe = Recipe(_choice(at.key("recipe.kind"), r["kind"], RECIPES),
                    _text(at.key("recipe.source"), r["source"]))
    listed = tuple(_listed(at.key("listed").index(i), v)
                   for i, v in enumerate(_list(at.key("listed"), raw["listed"])))
    _unique(at.key("listed"), [row.id for row in listed])
    listed_ids = {row.id for row in listed}

    parts_raw = [_part_raw(at.key("parts").index(i), v, listed_ids)
                 for i, v in enumerate(_list(at.key("parts"), raw["parts"]))]
    part_names = [p["name"] for p in parts_raw]
    _unique(at.key("parts"), part_names)
    for i, p in enumerate(parts_raw):
        if p["on"] is not None and p["on"] not in part_names:
            at.key("parts").index(i).key("on").fail(f"names no part: {p['on']!r}")
        for j, held in enumerate(p["retains"]):
            if held not in part_names:
                at.key("parts").index(i).key("retains").index(j).fail(
                    f"names no part of this piece: {held!r}")
    _check_forms(at.key("parts"), parts_raw)
    rig = _rig(at.key("rig"), raw["rig"], part_names) if "rig" in raw else None
    parts = tuple(
        _built(at.key("parts").index(i), lambda p=p: Part(
            **p, bones=rig.bind[p["name"]].bones if rig else (),
            bind=rig.bind[p["name"]].mode if rig else None))
        for i, p in enumerate(parts_raw)
    )

    uses = tuple(_use(at.key("uses").index(i), v)
                 for i, v in enumerate(_list(at.key("uses"), raw["uses"])))
    use_ids = {u.id for u in uses}
    holds = tuple(_hold(at.key("holds").index(i), v, use_ids, part_names)
                  for i, v in enumerate(_list(at.key("holds"), raw["holds"])))
    seats = tuple(_seat(at.key("seats").index(i), v, use_ids, part_names)
                  for i, v in enumerate(_list(at.key("seats"), raw["seats"])))
    mounts = tuple(_mount(at.key("mounts").index(i), v, set(part_names))
                   for i, v in enumerate(_list(at.key("mounts"), raw["mounts"])))
    _unique(at, [x.id for group in (uses, holds, seats, mounts) for x in group],
            "ids of uses, holds, seats and mounts")

    connectors = tuple(
        _connector(at.key("connectors").index(i), v, {p["name"]: p["shape"] for p in parts_raw},
                   {m.id for m in mounts},
                   {p["name"] for p in parts_raw if p["role"] in PRINTED_ROLES})
        for i, v in enumerate(_list(at.key("connectors"), raw.get("connectors", []))))
    _unique(at.key("connectors"), [c.id for c in connectors], "connector ids")
    keep_off = tuple(_keep_off(at.key("keep_off").index(i), v)
                     for i, v in enumerate(_list(at.key("keep_off"), raw.get("keep_off", []))))
    _unique(at.key("keep_off"), [k.id for k in keep_off], "keep-off ids")

    view = _view(at.key("view"), raw["view"]) if "view" in raw else None
    outlined, outlined_why = _outlined(at, raw)

    measured = _obj(at.key("measured"), raw["measured"])
    for key in measured:
        if key not in MEASURED_KEYS:
            at.key("measured").fail(f"unknown field {key!r}; measured keys are "
                                    f"{sorted(MEASURED_KEYS)}")
    return Piece(
        name=name, frame=frame, recipe=recipe, listed=listed, parts=parts, uses=uses,
        holds=holds, seats=seats, mounts=mounts, done=_text(at.key("done"), raw["done"]),
        size_class=_choice(at.key("size_class"), raw["size_class"], SIZE_CLASSES),
        measured=dict(measured), rig=rig, connectors=connectors, keep_off=keep_off,
        view=view, outlined=outlined, outlined_why=outlined_why,
    )


def _check_forms(at: _At, parts: list[dict]) -> None:
    """A seat or a mirror names an earlier part, so a build meets its source first. A mirror
    copies one source form, so the source is not itself a mirror and has the same shape."""
    earlier: dict[str, dict] = {}
    for i, part in enumerate(parts):
        form = part["form"]
        if form is not None and form.seat is not None:
            seat_at = at.index(i).key("form").key("seat").key("on")
            if form.seat.on not in earlier:
                seat_at.fail(f"names no earlier part: {form.seat.on!r}")
        if form is not None and form.mirror_of is not None:
            of_at = at.index(i).key("form").key("mirror_of")
            source = earlier.get(form.mirror_of)
            if source is None:
                of_at.fail(f"names no earlier part: {form.mirror_of!r}")
            if source["form"] is None:
                of_at.fail(f"{form.mirror_of!r} has no form to mirror")
            if source["form"].mirror_of is not None:
                of_at.fail(f"{form.mirror_of!r} is itself a mirror")
            if source["shape"] != part["shape"]:
                of_at.fail(f"{form.mirror_of!r} is shape {source['shape']!r}, but this part "
                           f"is {part['shape']!r}")
        earlier[part["name"]] = part


def _listed(at: _At, raw: Any) -> Listed:
    _fields(at, raw, ("id", "text", "source", "min_thickness"), ("owner", "finish", "held"))
    thickness = _num(at.key("min_thickness"), raw["min_thickness"])
    if thickness < 0:
        at.key("min_thickness").fail(f"must be 0 or more metres, got {thickness}")
    owner = raw.get("owner")
    return Listed(
        id=_name(at.key("id"), raw["id"]), text=_text(at.key("text"), raw["text"]),
        source=_text(at.key("source"), raw["source"]), min_thickness=thickness,
        owner=None if owner is None else _name(at.key("owner"), owner),
        finish=_choice(at.key("finish"), raw.get("finish", FINISHES[0]), FINISHES),
        held=_choice(at.key("held"), raw.get("held", HELD[0]), HELD),
    )


def _part_raw(at: _At, raw: Any, listed_ids: set[str]) -> dict:
    if isinstance(raw, dict) and ("bones" in raw or "bind" in raw):
        at.fail("a part's bones and bind mode live in the piece's rig.bind")
    _fields(at, raw, ("name", "listed", "shape", "on", "group", "role"),
            ("form", "fastener", "retains"))
    listed = _name(at.key("listed"), raw["listed"])
    if listed not in listed_ids:
        at.key("listed").fail(f"names no listed row: {listed!r}")
    shape = _choice(at.key("shape"), raw["shape"], tuple(sorted(SHAPES)))
    fastener = raw.get("fastener", False)
    if not isinstance(fastener, bool):
        at.key("fastener").fail(f"must be true or false, got {fastener!r}")
    retains = _names(at.key("retains"), raw["retains"]) if "retains" in raw else ()
    part = {
        "name": _name(at.key("name"), raw["name"]),
        "listed": listed,
        "shape": shape,
        "on": None if raw["on"] is None else _name(at.key("on"), raw["on"]),
        "group": _text(at.key("group"), raw["group"]),
        "role": raw["role"],
        "form": read_form(at.key("form"), shape, raw["form"]) if "form" in raw else None,
        "fastener": fastener,
        "retains": retains,
    }
    return part


def _use(at: _At, raw: Any) -> Use:
    _fields(at, raw, ("id", "point", "facing", "posture"))
    return Use(
        id=_name(at.key("id"), raw["id"]), point=_vec(at.key("point"), raw["point"], 3),
        facing=_vec(at.key("facing"), raw["facing"], 3, nonzero=True),
        posture=_choice(at.key("posture"), raw["posture"], POSTURES),
    )


def _station(at: _At, raw: str, use_ids: set[str]) -> str:
    use = _name(at, raw)
    if use not in use_ids:
        at.fail(f"names no use station: {use!r}")
    return use


def _surface(at: _At, raw: Any, part_names: list[str]) -> tuple[str, ...]:
    surface = _names(at, raw)
    for i, part in enumerate(surface):
        if part not in part_names:
            at.index(i).fail(f"names no part: {part!r}")
    return surface


def _hold(at: _At, raw: Any, use_ids: set[str], part_names: list[str]) -> Hold:
    _fields(at, raw, ("id", "use", "side", "surface", "point", "off", "grip"), ("sub_box",))
    if not isinstance(raw["grip"], bool):
        at.key("grip").fail(f"must be true or false, got {raw['grip']!r}")
    sub_box = raw.get("sub_box")
    return Hold(
        id=_name(at.key("id"), raw["id"]), use=_station(at.key("use"), raw["use"], use_ids),
        side=_choice(at.key("side"), raw["side"], SIDES),
        surface=_surface(at.key("surface"), raw["surface"], part_names),
        sub_box=None if sub_box is None else _box(at.key("sub_box"), sub_box),
        point=_vec(at.key("point"), raw["point"], 3),
        off=_vec(at.key("off"), raw["off"], 3, nonzero=True), grip=raw["grip"],
    )


def _seat(at: _At, raw: Any, use_ids: set[str], part_names: list[str]) -> Seat:
    if isinstance(raw, dict) and "clip" in raw:
        at.key("clip").fail("a seat no longer names a clip; the clip names its seat by ref")
    _fields(at, raw, ("id", "use", "point", "facing", "surface"))
    return Seat(
        id=_name(at.key("id"), raw["id"]), use=_station(at.key("use"), raw["use"], use_ids),
        point=_vec(at.key("point"), raw["point"], 3),
        facing=_vec(at.key("facing"), raw["facing"], 3, nonzero=True),
        surface=_surface(at.key("surface"), raw["surface"], part_names),
    )


def _mount(at: _At, raw: Any, part_names: set[str]) -> Mount:
    _fields(at, raw, ("id", "kind", "point", "axis", "takes"), ("fits", "part"))
    fits = raw.get("fits")
    part = None
    if "part" in raw:
        part = _name(at.key("part"), raw["part"])
        if part not in part_names:
            at.key("part").fail(f"names no part of this piece: {part!r}")
    return Mount(
        id=_name(at.key("id"), raw["id"]), kind=_choice(at.key("kind"), raw["kind"], MOUNT_KINDS),
        point=_vec(at.key("point"), raw["point"], 3),
        axis=_vec(at.key("axis"), raw["axis"], 3, nonzero=True),
        takes=_text(at.key("takes"), raw["takes"]),
        fits=None if fits is None else _names(at.key("fits"), fits), part=part,
    )


def _connector(at: _At, raw: Any, shapes: dict[str, str], mount_ids: set[str],
               printed: set[str]) -> Connector:
    _fields(at, raw, ("id", "part", "ends", "cap"), ("why",))
    part = _name(at.key("part"), raw["part"])
    if part not in shapes:
        at.key("part").fail(f"names no part: {part!r}")
    if shapes[part] != CABLE_SHAPE:
        at.key("part").fail(f"must name a part of shape {CABLE_SHAPE!r}, but {part!r} is "
                            f"{shapes[part]!r}")
    ends_at = at.key("ends")
    ends = _list(ends_at, raw["ends"])
    if len(ends) != 2:
        ends_at.fail(f"a connector has two ends, got {len(ends)}")
    return Connector(
        id=_name(at.key("id"), raw["id"]), part=part,
        ends=(_end(ends_at.index(0), ends[0], mount_ids, shapes, printed),
              _end(ends_at.index(1), ends[1], mount_ids, shapes, printed)),
        cap=_positive(at.key("cap"), raw["cap"]), why=_why(at, raw),
    )


def _end(at: _At, raw: Any, mount_ids: set[str], shapes: dict[str, str],
         printed: set[str]) -> End:
    """An end plugs into a mount of a placement in a layout, or (`self.<mount>`) into a mount
    of the connector's own piece. The first needs `layout` and may name its `plug`, a part of
    this piece; the second must have neither."""
    _fields(at, raw, ("point", "host"), ("layout", "plug"))
    point = _vec(at.key("point"), raw["point"], 3)
    host = _ref(at.key("host"), raw["host"], bone_ok=False, self_ok=True)
    if host.is_self:
        if "layout" in raw:
            at.key("layout").fail("a self end is on the connector's own piece and names no layout")
        if "plug" in raw:
            at.key("plug").fail("a self end is attached to the connector's own piece, not "
                                "seated: it names no plug")
        if host.id not in mount_ids:
            at.key("host").fail(f"names no mount of this piece: {host.id!r}")
        return End(point, None, host)
    if "layout" not in raw:
        at.fail("missing field 'layout': an end on another placement names its layout")
    plug = None
    if "plug" in raw:
        plug = _name(at.key("plug"), raw["plug"])
        if plug not in shapes:
            at.key("plug").fail(f"names no part of this piece: {plug!r}")
        if shapes[plug] == CABLE_SHAPE or plug in printed:
            at.key("plug").fail(f"must name a solid part, but {plug!r} is a cable or printed on "
                                "a surface")
    return End(point, _name(at.key("layout"), raw["layout"]), host, plug)


def _view(at: _At, raw: Any) -> View:
    """A piece's `view` object. `View.validate` owns the rules (a distance above 0, a `why`, sides
    never seen, each once, a `hidden_why` that matches them, and a `farthest_m` that is not nearer
    than `nearest_m` and has its `farthest_why`); this reads the fields."""
    _fields(at, raw, ("nearest_m", "why"),
            ("hidden", "hidden_why", "farthest_m", "farthest_why"))
    hidden = tuple(_list(at.key("hidden"), raw.get("hidden", [])))
    return _built(at, lambda: View(nearest_m=raw["nearest_m"], why=raw["why"], hidden=hidden,
                                   hidden_why=raw.get("hidden_why", ""),
                                   farthest_m=raw.get("farthest_m"),
                                   farthest_why=raw.get("farthest_why", "")))


def parse_view(raw: Any, path: Path | str, field: str = "view") -> View:
    """A `view` object read from a file other than a pack (a build spec), with the same rules and
    the same errors as a pack's. `field` is where it sits in `path`, for the message."""
    return _view(_At(Path(path), field), raw)


def _outlined(at: _At, raw: dict) -> tuple[bool, str]:
    """`outlined` is true or false and defaults to true: the piece is judged as outlined unless
    it says otherwise. `outlined: false` needs an `outlined_why` saying how the shipping look
    draws the piece without lines; an `outlined_why` with `outlined` true or missing is refused,
    so a stale reason does not outlive its flag."""
    outlined = raw.get("outlined", True)
    if not isinstance(outlined, bool):
        at.key("outlined").fail(f"must be true or false, got {outlined!r}")
    why_at = at.key("outlined_why")
    why = raw.get("outlined_why", "")
    if not outlined and (not isinstance(why, str) or not why.strip()):
        why_at.fail("a piece that is not outlined needs an outlined_why saying how its shipping "
                    f"look draws it without lines, got {why!r}")
    if outlined and "outlined_why" in raw:
        why_at.fail("names no unlined look to explain; set outlined to false or drop it")
    return outlined, why


def _keep_off(at: _At, raw: Any) -> KeepOff:
    _fields(at, raw, ("id", "point", "radius"), ("why",))
    return KeepOff(id=_name(at.key("id"), raw["id"]), point=_vec(at.key("point"), raw["point"], 3),
                   radius=_positive(at.key("radius"), raw["radius"]), why=_why(at, raw))
