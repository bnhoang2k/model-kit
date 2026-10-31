"""The manifest records: aliases, constants, name patterns and the frozen dataclasses."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from model_kit.core.errors import ManifestError, RecordError

if TYPE_CHECKING:  # the form types live in `forms.py`, which imports this module
    from .forms import Falloff, SoftForm

Vec3 = tuple[float, float, float]


Quat = tuple[float, float, float, float]


Box = tuple[Vec3, Vec3]


Window = tuple[float, float]


SHAPES = frozenset({"box", "lathe", "extrude", "blob", "ring", "meta", "cable", "tube", "leaf",
                    "decal"})


MEASURED_KEYS = frozenset({
    "kind", "zones", "snap", "foot", "bounds", "heights", "surfaces", "slots", "tris", "size",
    "fill", "skin", "hosts", "stands", "tip_length", "pivots", "export",
})


RecipeKind = Literal["assembly", "branching"]
RECIPES = get_args(RecipeKind)


SizeClass = Literal["prop", "piece", "hero"]
SIZE_CLASSES = get_args(SizeClass)


Posture = Literal["stand", "sit"]
POSTURES = get_args(Posture)


ClipPosture = Literal["stand", "sit", "lie", "free"]
CLIP_POSTURES = get_args(ClipPosture)


Side = Literal["L", "R", "both"]
SIDES = get_args(Side)


BindMode = Literal["rigid", "chain"]
BIND_MODES = get_args(BindMode)


Authoring = Literal["solved", "keyed"]
AUTHORING = get_args(Authoring)


Axis = Literal["x", "y", "z"]
AXES = get_args(Axis)


FRAME_AXES = ("gltf",)


CABLE_SHAPE = "cable"
"""The shape of a connector's part. A cable bends (`Part.bends`)."""


DECAL_SHAPE = "decal"
"""The shape of a part printed on its host (`DecalForm`). It needs an `on` and a printed role."""


LEAF_SHAPE = "leaf"
"""The shape of a grown leaf card."""


TUBE_SHAPE = "tube"
"""The shape of a grown limb part (`Part.tube`)."""


OPEN_SHAPES = (LEAF_SHAPE, CABLE_SHAPE, DECAL_SHAPE)
"""The shapes whose mesh is an open surface: a leaf is a sheet, a cable an open tube and a decal an
open patch printed on its host."""


MOUNT_KINDS = ("rest", "power", "signal")


Finish = Literal["solid", "print", "paint", "inlay", "display"]
FINISHES = get_args(Finish)
"""How a listed row is made. `solid` is an object with a body; the others (a print, a paint
film, an inlay cut flush, a screen image) are a colour change on a surface, so their parts are
decals."""

Held = Literal["rests", "bonded"]
HELD = get_args(Held)
"""How a listed row's parts are held to their hosts. `rests` is the default: gravity and the
part's fasteners hold them (the bearing check judges the rest). `bonded` is glued, welded or
moulded in: the bond holds them, and the gravity rule does not apply."""


PART_ROLES = ("solid", "eyes", "decal", "skip_outline")
"""`eyes` and `decal` are printed on a surface and left out of overlap tests."""


PRINTED_ROLES = ("eyes", "decal")
"""The roles printed on a surface (`Part.printed`). Every other role is a solid."""


NO_OUTLINE_ROLES = ("eyes", "skip_outline")
"""The roles a renderer draws no outline for. A part with role `eyes` is printed or inset on a
surface, as a decal, and has no edge of its own to outline; its fill carries its shape. A
`skip_outline` part opts out by name. Checks that judge outline lines leave these parts out."""


THICKNESS_ROLES = ("solid", "skip_outline")
"""The roles whose listed rows set the pack's thickness base."""


AVOID_KINDS = ("box", "ellipsoid")


VIEW_SIDES = ("front", "back", "left", "right", "top", "bottom")
"""The sides of a piece a `View` can call never seen, in the piece frame (glTF axes): front is
+z, back -z, left +x, right -x, top +y, bottom -y. A face belongs to the side its outward normal
points most toward."""


SELF_HOST = "self"
"""A connector end's host written `self.<mount>`: a mount on the connector's own piece."""


RIG_PACK = "rig"


_NAME = r"[A-Za-z0-9_-]+"


_BONE = r"[A-Za-z0-9_.-]+"


_REF = re.compile(rf"(?P<pack>{_NAME}):(?P<piece>{_NAME})\.(?P<id>{_NAME})")


_SELF_MOUNT = re.compile(_NAME)


_BONE_REF = re.compile(rf"{RIG_PACK}:(?P<bone>{_BONE})")


def _finite(value: object) -> bool:
    """True for a real, finite number (not a bool)."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive_problem(value: object) -> str | None:
    """Why `value` is not a finite number above 0, or None."""
    if not _finite(value):
        return f"must be a finite number, got {value!r}"
    if value <= 0:
        return f"must be above 0, got {value!r}"
    return None


def _text_problem(value: object) -> str | None:
    """Why `value` is not a non-empty string, or None."""
    if not isinstance(value, str) or not value.strip():
        return f"must be a non-empty string, got {value!r}"
    return None


def _choice_problem(value: object, options: tuple[str, ...]) -> str | None:
    """Why `value` is not one of `options`, or None."""
    return None if value in options else f"must be one of {list(options)}, got {value!r}"


@dataclass(frozen=True)
class Ref:
    """`<pack>:<piece>.<id>`, or `rig:<bone>` (then `piece` is None and `id` is the bone)."""

    pack: str
    piece: str | None
    id: str

    @property
    def is_bone(self) -> bool:
        return self.pack == RIG_PACK

    @classmethod
    def parse(cls, text: str, *, self_ok: bool = False) -> Ref:
        """Read a ref string. `self.<mount>` (a connector end's host) reads only with
        `self_ok`, so `Ref.parse(str(ref), self_ok=True) == ref` holds for every ref."""
        if not isinstance(text, str):
            raise ManifestError(f"a ref must be a string, got {text!r}")
        if self_ok and text.startswith(f"{SELF_HOST}."):
            if _SELF_MOUNT.fullmatch(text[len(SELF_HOST) + 1:]) is None:
                raise ManifestError(f"ref {text!r} is not self.<mount>")
            return cls(SELF_HOST, None, text[len(SELF_HOST) + 1:])
        if m := _BONE_REF.fullmatch(text):
            return cls(RIG_PACK, None, m["bone"])
        m = _REF.fullmatch(text)
        if m is None or m["pack"] == RIG_PACK:
            raise ManifestError(f"ref {text!r} is not <pack>:<piece>.<id> or rig:<bone>")
        return cls(m["pack"], m["piece"], m["id"])

    @property
    def is_self(self) -> bool:
        """True for a connector end's `self.<mount>`: pack is `self` and there is no piece."""
        return self.pack == SELF_HOST and self.piece is None

    def __str__(self) -> str:
        if self.is_bone:
            return f"{RIG_PACK}:{self.id}"
        if self.is_self:
            return f"{SELF_HOST}.{self.id}"
        return f"{self.pack}:{self.piece}.{self.id}"


@dataclass(frozen=True)
class Frame:
    axes: str
    origin: str


@dataclass(frozen=True)
class Recipe:
    kind: RecipeKind
    source: str


@dataclass(frozen=True)
class Listed:
    id: str
    text: str
    source: str
    min_thickness: float
    owner: str | None
    finish: Finish = "solid"
    """`solid` unless the row is a marking: then its parts are decals (check `markings`)."""

    held: Held = "rests"
    """`rests` unless the row is bonded to its host: then bearing skips the gravity rule."""


class PartKind:
    """What a part is, derived from its `role` and `shape` alone. `Part` (the manifest record) and
    `PartMesh` (the record's geometry in a check) both inherit it, so a fact such as "printed" has
    one owner. A check reads these properties and never compares a `role` or `shape` with a
    vocabulary word (`test/test_part_predicates_home.py` fails a check that does); `markings`
    alone reads them raw, because it judges a role and a shape that disagree."""

    role: str
    shape: str | None

    @property
    def printed(self) -> bool:
        """Printed on a surface (role `eyes` or `decal`): left out of overlap and of every test
        that needs a body of its own."""
        return self.role in PRINTED_ROLES

    @property
    def solid(self) -> bool:
        """A body a crossing, a sweep or a thickness is measured on: not printed on a surface.
        Roles `solid` and `skip_outline`; `printed` is the exact opposite."""
        return self.role in THICKNESS_ROLES

    @property
    def skips_outline(self) -> bool:
        """The part draws no outline of its own: a decal follows its host, and the roles in
        `NO_OUTLINE_ROLES` opt out. A cable is not in it: near-crease leaves a cable out
        through `bends`, and detail still judges a cable's facets. (`Piece.outlined` is the
        piece-wide fact: the look draws no line on any part of the piece.)"""
        return self.shape == DECAL_SHAPE or self.role in NO_OUTLINE_ROLES

    @property
    def open_surface(self) -> bool:
        """The mesh is an open surface by shape (a leaf sheet, a cable, a decal patch), so a
        boundary is not a fault. Topology also leaves out a solid part that skips its outline
        (role `skip_outline`)."""
        return self.shape in OPEN_SHAPES

    @property
    def bends(self) -> bool:
        """The part is a cable (shape `cable`): it bends, so it is not a rigid body. Rigid parts
        are the ones that are not cables."""
        return self.shape == CABLE_SHAPE

    @property
    def tube(self) -> bool:
        """The part is a grown limb (shape `tube`); a leaf card is not."""
        return self.shape == TUBE_SHAPE

    @property
    def skipped_as(self) -> str:
        """What an outline check's coverage line calls a part it leaves out: its shape when that
        is `decal` or `cable`, else its role."""
        return self.shape if self.shape in (DECAL_SHAPE, CABLE_SHAPE) else self.role


@dataclass(frozen=True)
class Part(PartKind):
    """One part of a piece. Each stored field says in its metadata whether the Blender builder
    writes it as a custom property (`tag: True`, see `PART_TAGGED`) or sets it another way
    (`tag: False`): rigging sets `name`, `bones` and `bind`, and `register(forms=)` sets `form`,
    which cannot be a Blender custom property. A new field with no `tag` key fails the guard
    test in `test/blender/test_build.py`."""

    name: str = field(metadata={"tag": False})
    listed: str = field(metadata={"tag": True})
    shape: str = field(metadata={"tag": True})
    on: str | None = field(metadata={"tag": True})
    group: str = field(metadata={"tag": True})
    role: str = field(metadata={"tag": True})
    bones: tuple[str, ...] = field(metadata={"tag": False})
    bind: BindMode | None = field(metadata={"tag": False})
    form: SoftForm | None = field(default=None, metadata={"tag": False})
    fastener: bool = field(default=False, metadata={"tag": True})
    """The part is a bolt, screw or pin; its `on` names the part it fastens into."""
    retains: tuple[str, ...] = field(default=(), metadata={"tag": True})
    """Parts of the same piece this part holds down or in (a string tree, a strap, a jaw)."""

    def __post_init__(self) -> None:
        self.validate(f"part {self.name!r}")

    def validate(self, where: str) -> None:
        """Raise a `RecordError` naming `where` and the field when the part is not legal: a shape
        or a role outside the vocabulary, a part that retains itself, or a decal that
        `decal_problem` refuses."""
        if why := _choice_problem(self.shape, tuple(sorted(SHAPES))):
            raise RecordError(where, "shape", why)
        if why := _choice_problem(self.role, PART_ROLES):
            raise RecordError(where, "role", why)
        if self.name in self.retains:
            raise RecordError(where, "retains", f"part {self.name!r} cannot retain itself")
        if (why := decal_problem(self)) is not None:
            raise RecordError(where, "on" if self.on is None else "role", why)


PART_TAGGED: tuple[str, ...] = tuple(f.name for f in fields(Part) if f.metadata.get("tag") is True)
"""The `Part` fields the Blender builder writes as custom properties and reads back, in field
order. Predicates (`printed`, `solid`, ...) are properties, not fields, so they are not here."""


def decal_problem(part: Part) -> str | None:
    """Why a decal part is refused, or None. A decal is printed on a host, so it needs an `on`,
    and it is not a body, so its role cannot be one that sets the pack's thickness. Any other
    disagreement between role and shape is a finding of the `markings` check, not a load error."""
    if part.shape != DECAL_SHAPE:
        return None
    if part.on is None:
        return f"part {part.name!r} is a decal but has no `on`: name the part it is printed on"
    if part.solid:
        return (f"part {part.name!r} is a decal but has the role {part.role!r}, which is a solid: "
                f"give it a printed role ({', '.join(PRINTED_ROLES)})")
    return None


@dataclass(frozen=True)
class Use:
    id: str
    point: Vec3
    facing: Vec3
    posture: Posture


@dataclass(frozen=True)
class Hold:
    id: str
    use: str
    side: Side
    surface: tuple[str, ...]
    sub_box: Box | None
    point: Vec3
    off: Vec3
    grip: bool


@dataclass(frozen=True)
class Seat:
    id: str
    use: str
    point: Vec3
    facing: Vec3
    surface: tuple[str, ...]


@dataclass(frozen=True)
class Mount:
    """A place on a piece that takes a guest part. `point` and `axis` locate the joint (the axis
    points out of the host, toward the guest); `takes` is the guest's part the mount takes;
    `part` is the host's own part the mount is cut in (the part the joint is judged against).
    A mount with no `part` cannot be judged against geometry: the joint checks read it
    `Unmeasured`."""

    id: str
    kind: str
    point: Vec3
    axis: Vec3
    takes: str
    fits: tuple[str, ...] | None
    part: str | None = None


@dataclass(frozen=True)
class BoneRest:
    """A bone's rest pose in the piece's glTF frame. `deform` says whether it moves vertices."""

    head: Vec3
    tail: Vec3
    roll: float
    deform: bool
    why: str | None = None


@dataclass(frozen=True)
class Bone:
    name: str
    parent: str | None
    length: float
    rest: BoneRest | None = None


@dataclass(frozen=True)
class Bind:
    bones: tuple[str, ...]
    mode: BindMode
    falloff: Falloff | None = None


@dataclass(frozen=True)
class SitSpec:
    below_seat: float
    forward: float
    past_edge: tuple[str, ...] = ()
    """Bones whose heads must lie past a seat's front edge: the knees of a person."""


@dataclass(frozen=True)
class MirrorSpec:
    pairs: tuple[tuple[str, str], ...]
    axis: Axis


@dataclass(frozen=True)
class Joint:
    """A bending joint's range of motion: degrees per axis, as euler angles in the bone's rest
    frame (the order `rig.pose_world` uses). `why` cites a table or the design."""

    range_deg: Mapping[str, tuple[float, float]]
    why: str


@dataclass(frozen=True)
class Chain:
    root: str
    bones: tuple[str, ...]
    effector: str
    effector_radius: float
    dofs: dict[str, str]
    limits: dict[str, tuple[float, float]]
    effector_group: str


@dataclass(frozen=True)
class Avoid:
    """A shape in `bone`'s rest frame that the solver keeps arms out of.

    `centre` is its middle; `half` is the half-size on each axis (box half-extents, or the
    ellipsoid's semi-axes).
    """

    id: str
    bone: str
    kind: str
    centre: Vec3
    half: Vec3
    why: str


@dataclass(frozen=True)
class RigSpec:
    armature: str
    bones: tuple[Bone, ...]
    bind: dict[str, Bind]
    contact_groups: dict[str, tuple[str, ...]]
    chains: dict[str, Chain]
    sit: SitSpec | None = None
    """None: the body does not sit, and no clip may name a seat."""
    mirror: MirrorSpec | None = None
    """None: the body has no left and right, and no clip may be the mirror of another."""
    outline_modifiers: tuple[str, ...] = ()
    """The outline shell's modifier names: the exporters switch them off. May be empty."""
    avoid: tuple[Avoid, ...] = ()
    """Shapes in a bone's rest frame that the solver keeps the arms out of."""
    supports: tuple[str, ...] = ()
    """Contact groups that bear the body's weight when it stands."""
    joints: Mapping[str, Joint] = field(default_factory=dict)
    """Each bending joint's range of motion, by bone."""
    root: str | None = None
    """The bone a seat and the side rules measure from. None: the one parentless bone."""


@dataclass(frozen=True)
class End:
    """One end of a connector. `point` is in the layout's frame, or in the piece's own frame
    when `host` is `self.<mount>` (then `layout` is None). `host` names the mount the end
    plugs into; its piece slot is a placement id of `layout`. `plug` is the part of the
    connector's own piece that seats in that mount (an end on a layout names it; the joint checks
    read an end with no `plug` `Unmeasured`). A self end is attached, not seated, and has none."""

    point: Vec3
    layout: str | None
    host: Ref
    plug: str | None = None


@dataclass(frozen=True)
class Connector:
    """A cable (`part`, shape `cable`) between two mounts, no longer than `cap` model units."""

    id: str
    part: str
    ends: tuple[End, End]
    cap: float
    why: str


@dataclass(frozen=True)
class KeepOff:
    """A point that a mount must stay `radius` clear of."""

    id: str
    point: Vec3
    radius: float
    why: str


@dataclass(frozen=True)
class View:
    """How close a viewer's eye comes to a piece. `nearest_m` is the nearest distance, in real
    metres, from which the piece is seen, and `why` says how it is used (a held instrument seen
    by its player, a tree seen from the path beside it). The checks that judge what is visible
    read it; none assumes a camera.

    `hidden` names sides of the piece that are never seen (`VIEW_SIDES`), and `hidden_why` says
    why. Checks skip the faces on those sides. The default is none: every side is judged.

    `farthest_m` is the farthest distance, in real metres, from which the piece is seen, and
    `farthest_why` says how it is used. A check that judges depth flicker reads it, because the
    depth buffer is coarsest at the far end. It is optional: a piece without it reads `Unmeasured`
    there. It is never nearer than `nearest_m`."""

    nearest_m: float
    why: str
    hidden: tuple[str, ...] = ()
    hidden_why: str = ""
    farthest_m: float | None = None
    farthest_why: str = ""

    def __post_init__(self) -> None:
        self.validate("view")

    def validate(self, where: str) -> None:
        """Raise a `RecordError` naming `where` and the field when the view is not legal: a
        distance that is not above 0, an empty `why`, a hidden side that is not in `VIEW_SIDES` or
        repeats, hidden sides with no `hidden_why`, a `hidden_why` with no side to explain (a
        stale reason does not outlive its list), a `farthest_m` that is not above 0 or is nearer
        than `nearest_m`, a `farthest_m` with no `farthest_why`, or a `farthest_why` with no
        `farthest_m`."""
        if why := _positive_problem(self.nearest_m):
            raise RecordError(where, "nearest_m", why)
        if why := _text_problem(self.why):
            raise RecordError(where, "why", why)
        for i, side in enumerate(self.hidden):
            if why := _choice_problem(side, VIEW_SIDES):
                raise RecordError(where, f"hidden[{i}]", why)
        if len(set(self.hidden)) != len(self.hidden):
            raise RecordError(where, "hidden", f"hidden sides repeat: {list(self.hidden)}")
        if self.hidden and _text_problem(self.hidden_why):
            raise RecordError(where, "hidden_why",
                              "a piece with hidden sides needs a hidden_why saying why they are "
                              f"never seen, got {self.hidden_why!r}")
        if not self.hidden and self.hidden_why != "":
            raise RecordError(where, "hidden_why", "names no hidden side to explain; list the "
                              "sides in hidden or drop it")
        if self.farthest_m is None:
            if self.farthest_why != "":
                raise RecordError(where, "farthest_why", "names no farthest_m to explain; set "
                                  "farthest_m or drop it")
            return
        if why := _positive_problem(self.farthest_m):
            raise RecordError(where, "farthest_m", why)
        if self.farthest_m < self.nearest_m:
            raise RecordError(where, "farthest_m", f"must be at least nearest_m "
                              f"{self.nearest_m!r}, got {self.farthest_m!r}")
        if why := _text_problem(self.farthest_why):
            raise RecordError(where, "farthest_why", "a view with a farthest_m needs a "
                              f"farthest_why saying how the piece is used; {why}")


@dataclass(frozen=True)
class Piece:
    name: str
    frame: Frame
    recipe: Recipe
    listed: tuple[Listed, ...]
    parts: tuple[Part, ...]
    uses: tuple[Use, ...]
    holds: tuple[Hold, ...]
    seats: tuple[Seat, ...]
    mounts: tuple[Mount, ...]
    done: str
    size_class: SizeClass
    measured: Mapping[str, Any]
    rig: RigSpec | None
    connectors: tuple[Connector, ...] = ()
    keep_off: tuple[KeepOff, ...] = ()
    view: View | None = None
    """The piece's nearest viewing distance, or None when it does not say (an older pack)."""
    outlined: bool = True
    """Whether the piece's shipping look draws outlines at all. False skips the checks that judge
    drawn lines (near-crease) on every part; the default is strict. It is the piece-wide fact. A
    single part that draws no outline of its own is `Part.skips_outline`, and a part that is an
    open surface is `Part.open_surface`."""
    outlined_why: str = ""
    """How the shipping look draws the piece without lines. Needed when `outlined` is false."""


@dataclass(frozen=True)
class Pack:
    name: str
    units_per_metre: float
    pieces: dict[str, Piece]
    spacing: float | None = None
    """The grid `PackBuilder` laid the pieces on, or None for a pack in its stage frame."""


@dataclass(frozen=True)
class Dangle:
    budget: float
    why: str


@dataclass(frozen=True)
class Cross:
    why: str


@dataclass(frozen=True)
class ClipHold:
    ref: Ref
    who: str
    who_box: Box | None
    when: tuple[Window, ...]
    budget: float | None
    dangle: Dangle | None
    cross: Cross | None


@dataclass(frozen=True)
class Target:
    key: int
    chain: str
    ref: Ref
    offset: Vec3
    why: str | None
    baked: Vec3 | None
    tool: str | None


@dataclass(frozen=True)
class Placement:
    """Where a clip puts a piece: rotate by `quaternion` ([x, y, z, w]), then move."""

    id: str
    piece: str
    pack: str
    position: Vec3
    quaternion: Quat

    def __post_init__(self) -> None:
        q = np.asarray(self.quaternion, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)) or not np.any(q):
            raise ManifestError(f"placement {self.id!r}: quaternion must be 4 finite numbers, "
                                f"not all 0, got {self.quaternion!r}")

    def _turn(self) -> np.ndarray:
        x, y, z, w = np.asarray(self.quaternion, dtype=float) / np.linalg.norm(self.quaternion)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def matrix(self) -> np.ndarray:
        """The (4, 4) world-from-piece matrix: the turn, then the move. No scale."""
        m = np.eye(4)
        m[:3, :3] = self._turn()
        m[:3, 3] = np.asarray(self.position, dtype=float)
        return m

    def to_world(self, points) -> np.ndarray:
        """Piece-frame points (..., 3) to world points."""
        return np.asarray(points, dtype=float) @ self._turn().T + np.asarray(self.position)

    def to_piece(self, points) -> np.ndarray:
        """World points (..., 3) to piece-frame points."""
        return (np.asarray(points, dtype=float) - np.asarray(self.position)) @ self._turn()


@dataclass(frozen=True)
class FloorGear:
    pieces: dict[str, Placement]


@dataclass(frozen=True)
class Prop:
    id: str
    piece: str
    pack: str
    bone: str
    offset_position: Vec3
    offset_quaternion: Quat
    show_frames: tuple[Window, ...]


@dataclass(frozen=True)
class Lift:
    """A support that may leave the floor in a standing clip: a step, a hop, a kick."""

    support: str
    """A contact group named in the rig's `supports`."""
    when: tuple[Window, ...]
    """Frame windows, each inside the clip's frames."""
    why: str


@dataclass(frozen=True)
class Clip:
    name: str
    frames: tuple[int, int]
    fps: float
    authoring: Authoring
    seat: Ref | None
    holds: tuple[ClipHold, ...]
    targets: tuple[Target, ...]
    floor_gear: FloorGear
    props: dict[str, Prop]
    mirror_of: str | None
    loop: bool = False
    loop_from: int | None = None
    """A frame inside `frames` where the loop starts; only with `loop`."""
    frame: str = "gltf"
    """The frame the clip's world reads (one of `FRAME_AXES`)."""
    seat_when: tuple[Window, ...] | None = None
    """Frame windows in which the clip sits; None means the whole clip."""
    posture: ClipPosture | None = None
    """What the body does (one of `CLIP_POSTURES`); None for a clip written before posture."""
    lifts: tuple[Lift, ...] = ()
    """Supports that may leave the floor, and when; only on a `stand` clip."""

    def __post_init__(self) -> None:
        self.validate(f"clip {self.name!r}")

    def validate(self, where: str) -> None:
        """Raise a `RecordError` naming `where` and the field when the clip is not legal: frames,
        fps, authoring, targets, seat, posture, lifts, loop or mirror that disagree with each
        other, or a ref slot that two placements claim."""

        def fail(field: str | None, why: str):
            raise RecordError(where, field, why)

        if why := frames_problem(self.frames):
            fail("frames", why)
        if why := _positive_problem(self.fps):
            fail("fps", why)
        if why := _choice_problem(self.authoring, AUTHORING):
            fail("authoring", why)
        if conflict := name_clash(self.floor_gear.pieces, self.props):
            fail(None, conflict)
        if self.authoring == "keyed" and self.targets:
            fail("targets", "a keyed clip has no key targets")
        if self.authoring == "solved" and not self.targets:
            fail("targets", "a solved clip needs key targets")
        if self.seat_when is not None and self.seat is None:
            fail("seat_when", "needs a seat: seat_when says when the clip sits")
        if self.seat_when is not None:
            if not self.seat_when:
                fail("seat_when", "must not be empty")
            for i, window in enumerate(self.seat_when):
                if not (isinstance(window, tuple | list) and len(window) == 2
                        and all(_finite(v) for v in window)):
                    fail(f"seat_when[{i}]", f"must be [start, end] frame numbers, got {window!r}")
                if why := window_problem(window, self.frames):
                    fail(f"seat_when[{i}]", why)
        if self.posture is not None and (why := _choice_problem(self.posture, CLIP_POSTURES)):
            fail("posture", why)
        if self.seat is not None and self.posture not in (None, "sit"):
            fail("posture", f"a clip with a seat must be 'sit', got {self.posture!r}")
        if self.seat is None and self.posture == "sit":
            fail("posture", "a 'sit' clip must name a seat")
        if self.lifts and self.posture != "stand":
            fail("lifts", f"only a 'stand' clip has lifts, but this clip's posture is "
                          f"{self.posture!r}")
        if not isinstance(self.loop, bool):
            fail("loop", f"must be true or false, got {self.loop!r}")
        if self.loop_from is not None:
            if not self.loop:
                fail("loop_from", "needs loop: true; a clip that does not loop has no start")
            if (isinstance(self.loop_from, bool) or not isinstance(self.loop_from, int)
                    or not self.frames[0] <= self.loop_from < self.frames[1]):
                fail("loop_from", f"must be a whole frame from {self.frames[0]} up to (not "
                                  f"including) {self.frames[1]}, got {self.loop_from!r}")
        if self.mirror_of == self.name:
            fail("mirror_of", "a clip cannot mirror itself")


def frames_problem(frames: object) -> str | None:
    """Why `frames` is not a clip's (start, end) pair of whole frame numbers with the start before
    the end, or None. The loader checks it before it reads the windows that sit inside the
    frames."""
    if (not isinstance(frames, tuple | list) or len(frames) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in frames)):
        return f"must be two whole frame numbers, got {frames!r}"
    if frames[0] >= frames[1]:
        return f"start {frames[0]} must be before end {frames[1]}"
    return None


def window_problem(window: Window, frames: tuple[int, int]) -> str | None:
    """Why a (start, end) frame window is not forward and inside a clip's `frames`, or None."""
    t0, t1 = window
    if t0 > t1:
        return f"window {[t0, t1]} runs backwards"
    if t0 < frames[0] or t1 > frames[1]:
        return f"window {[t0, t1]} is outside the clip's frames {list(frames)}"
    return None


def name_clash(placements: Mapping[str, Placement], props: Mapping[str, Prop]) -> str | None:
    """Why a ref's piece slot could mean two things in a clip, or None.

    The slot is one namespace per clip: every floor_gear placement id plus every carried
    prop's piece name. A name may appear in it once. A piece may still be carried and also
    stand under another placement id (`shop:Lamp` is the prop, `shop:Lamp_desk` the copy).
    """
    claims: dict[str, list[str]] = {}
    for pid, placement in placements.items():
        claims.setdefault(pid, []).append(
            f"floor_gear {pid!r} ({placement.pack}:{placement.piece})")
    for pid, prop in props.items():
        claims.setdefault(prop.piece, []).append(f"prop {pid!r} ({prop.pack}:{prop.piece})")
    for name, who in claims.items():
        if len(who) > 1:
            return f"name {name!r} is claimed twice in the clip's ref namespace: {', '.join(who)}"
    return None


def seat_windows(clip: Clip) -> tuple[Window, ...]:
    """The windows in which the clip's seat rule applies: `seat_when`, or the whole clip."""
    if clip.seat_when is None:
        return ((clip.frames[0], clip.frames[1]),)
    return clip.seat_when


@dataclass(frozen=True, eq=False)
class Anchor:
    kind: Literal["hold", "seat", "mount", "use"]
    piece: Piece
    item: Hold | Seat | Mount | Use
    world: np.ndarray | None
    placed_by: str | None = None
    """The placement key used, `prop:<prop id>` for a carried prop's piece, or None."""


@dataclass(frozen=True)
class LayoutPlacement:
    """A piece standing in a layout. `on` is a mount ref (its piece slot is another placement
    id of the same layout) that this placement is plugged into; the placement keeps its own
    position anyway, so a check can compare the two. `rests_on` names a placement it sits on
    with no mount (gear on a deck, a box on a table); a placement has `on` or `rests_on`, not
    both."""

    id: str
    pack: str
    piece: str
    position: Vec3
    quaternion: Quat
    on: Ref | None = None
    rests_on: str | None = None
    """Another placement id of the same layout that this one rests on without a mount."""

    def __post_init__(self) -> None:
        q = np.asarray(self.quaternion, dtype=float)
        if q.shape != (4,) or not np.all(np.isfinite(q)) or not np.any(q):
            raise ManifestError(f"layout placement {self.id!r}: quaternion must be 4 finite "
                                f"numbers, not all 0, got {self.quaternion!r}")


@dataclass(frozen=True)
class Stand:
    """Where a performer stands to play `clip`. `facing` is a unit vector with y 0."""

    id: str
    position: Vec3
    facing: Vec3
    posture: Posture
    clip: str


@dataclass(frozen=True)
class Layout:
    """Where a pack's pieces stand, in `frame`, with a stage's core at the origin."""

    name: str
    frame: str
    placements: dict[str, LayoutPlacement]
    stands: dict[str, Stand]
