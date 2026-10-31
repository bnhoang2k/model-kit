"""The `rig` section reader of a piece."""

from __future__ import annotations

import math
from typing import Any

from model_kit.core.errors import ManifestError
from model_kit.core.tolerances import REST_LENGTH_TOLERANCE

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
    _text,
    _unique,
    _vec,
    _why,
)
from .forms import read_falloff
from .records import (
    _BONE,
    AVOID_KINDS,
    AXES,
    BIND_MODES,
    Avoid,
    Bind,
    Bone,
    BoneRest,
    Chain,
    Joint,
    MirrorSpec,
    RigSpec,
    SitSpec,
)


def _rig(at: _At, raw: Any, part_names: list[str]) -> RigSpec:
    _fields(at, raw, ("armature", "bones", "bind", "contact_groups", "chains"),
            ("sit", "mirror", "outline_modifiers", "avoid", "supports", "joints", "root"))
    bones: list[Bone] = []
    for i, b in enumerate(_list(at.key("bones"), raw["bones"], nonempty=True)):
        b_at = at.key("bones").index(i)
        _fields(b_at, b, ("name", "parent", "length"), ("rest",))
        parent = None if b["parent"] is None else _name(b_at.key("parent"), b["parent"], _BONE)
        if parent is not None and parent not in {x.name for x in bones}:
            b_at.key("parent").fail(f"names no bone listed before it: {parent!r}")
        length = _positive(b_at.key("length"), b["length"])
        rest = _rest(b_at.key("rest"), b["rest"], length) if "rest" in b else None
        bones.append(Bone(_name(b_at.key("name"), b["name"], _BONE), parent, length, rest))
    bone_names = [b.name for b in bones]
    _unique(at.key("bones"), bone_names, "bone names")
    parents = {b.name: b.parent for b in bones}

    def bone_list(b_at: _At, value: Any) -> tuple[str, ...]:
        names = _names(b_at, value, _BONE)
        for i, bone in enumerate(names):
            if bone not in bone_names:
                b_at.index(i).fail(f"names no bone: {bone!r}")
        return names

    bind = {}
    bind_at = at.key("bind")
    for part, entry in _obj(bind_at, raw["bind"]).items():
        p_at = bind_at.key(part)
        if part not in part_names:
            p_at.fail(f"names no part: {part!r}")
        _fields(p_at, entry, ("bones", "mode"), ("falloff",))
        mode = _choice(p_at.key("mode"), entry["mode"], BIND_MODES)
        bound = bone_list(p_at.key("bones"), entry["bones"])
        if mode == "rigid" and len(bound) != 1:
            p_at.fail(f"a rigid bind takes exactly one bone, got {list(bound)}")
        if mode == "chain" and len(bound) < 2:
            p_at.fail(f"a chain bind takes two or more bones, got {list(bound)}")
        falloff = None
        if "falloff" in entry:
            f_at = p_at.key("falloff")
            if mode != "chain":
                f_at.fail("only a chain bind has a falloff")
            falloff = read_falloff(f_at, entry["falloff"])
            if len(falloff.bands) != len(bound) - 1:
                f_at.key("bands").fail(f"needs one band per pair of bones, {len(bound) - 1} "
                                       f"for {len(bound)} bones, got {len(falloff.bands)}")
        bind[part] = Bind(bound, mode, falloff)
    for part in part_names:
        if part not in bind:
            bind_at.fail(f"part {part!r} of a rigged piece has no bind")

    groups_at = at.key("contact_groups")
    groups = {_name(groups_at, k, _BONE): bone_list(groups_at.key(k), v)
              for k, v in _obj(groups_at, raw["contact_groups"]).items()}

    chains_at = at.key("chains")
    chains = {_name(chains_at, k, _BONE): _chain(chains_at.key(k), v, parents, bone_list, groups)
              for k, v in _obj(chains_at, raw["chains"]).items()}

    sit = None
    if "sit" in raw:
        s_at = at.key("sit")
        s = _fields(s_at, raw["sit"], ("below_seat", "forward"), ("past_edge",))
        past_edge = bone_list(s_at.key("past_edge"), s["past_edge"]) if "past_edge" in s else ()
        sit = SitSpec(_num(s_at.key("below_seat"), s["below_seat"]),
                      _num(s_at.key("forward"), s["forward"]), past_edge)

    mirror = None
    if "mirror" in raw:
        m_at = at.key("mirror")
        m = _fields(m_at, raw["mirror"], ("pairs", "axis"))
        pairs = []
        for i, pair in enumerate(_list(m_at.key("pairs"), m["pairs"])):
            if not isinstance(pair, list) or len(pair) != 2:
                m_at.key("pairs").index(i).fail(f"a pair is two different names, got {pair!r}")
            a, b = bone_list(m_at.key("pairs").index(i), pair)
            pairs.append((a, b))
        _unique(m_at.key("pairs"), [n for pair in pairs for n in pair], "mirrored names")
        mirror = MirrorSpec(tuple(pairs), _choice(m_at.key("axis"), m["axis"], AXES))

    supports: tuple[str, ...] = ()
    if "supports" in raw:
        s_at = at.key("supports")
        supports = _names(s_at, raw["supports"], _BONE)
        for i, group in enumerate(supports):
            if group not in groups:
                s_at.index(i).fail(f"names no contact group: {group!r}")

    root = None
    if "root" in raw:
        root = _name(at.key("root"), raw["root"], _BONE)
        if root not in bone_names:
            at.key("root").fail(f"names no bone: {root!r}")

    mods_at = at.key("outline_modifiers")
    mods = tuple(_name(mods_at.index(i), v)
                 for i, v in enumerate(_list(mods_at, raw.get("outline_modifiers", []))))
    _unique(mods_at, list(mods), "modifier names")

    avoid_at = at.key("avoid")
    avoid = tuple(_avoid(avoid_at.index(i), v, bone_names)
                  for i, v in enumerate(_list(avoid_at, raw.get("avoid", []))))
    _unique(avoid_at, [a.id for a in avoid], "avoid ids")

    joints_at = at.key("joints")
    joints = {_name(joints_at, k, _BONE): _joint(joints_at.key(k), v)
              for k, v in _obj(joints_at, raw.get("joints", {})).items()}

    spec = RigSpec(armature=_text(at.key("armature"), raw["armature"]), bones=tuple(bones),
                   bind=bind, contact_groups=groups, chains=chains, sit=sit, mirror=mirror,
                   outline_modifiers=mods, avoid=avoid, supports=supports, joints=joints,
                   root=root)
    try:
        rig_root(spec)  # several parentless bones need `root`, joints or not
    except ManifestError as err:
        at.key("root").fail(str(err))
    if joints:  # a rig with no joints entry has no joint rules to meet
        _check_joints(at, spec)
    return spec


def _joint(at: _At, raw: Any) -> Joint:
    f = _fields(at, raw, ("range_deg", "why"))
    ranges_at = at.key("range_deg")
    ranges = {}
    for axis, pair in _obj(ranges_at, f["range_deg"]).items():
        a_at = ranges_at.key(axis)
        _choice(a_at, axis, AXES)
        lo, hi = _vec(a_at, pair, 2)
        if lo >= hi:
            a_at.fail(f"low {lo} must be under high {hi}")
        ranges[axis] = (lo, hi)
    if not ranges:
        ranges_at.fail("must not be empty")
    return Joint(ranges, _text(at.key("why"), f["why"]))


# A chain limit is in radians and a joint range in degrees: the same angle written twice may
# differ by this much after the conversion (float rounding), and still lie inside the range.
LIMIT_SLACK_RAD = 1e-9


def _check_joints(at: _At, spec: RigSpec) -> None:
    """Each `joints` key is a bending joint; each chain axis on a joint has a range; each chain
    limit lies inside its joint's range."""
    bending = bending_joints(spec)
    for bone in spec.joints:
        if bone not in bending:
            at.key("joints").key(bone).fail(
                f"{bone!r} is not a bending joint: it needs weighted vertices both inside and "
                f"outside its subtree and is not the root; the joints are {list(bending)}")
    for name, chain in spec.chains.items():
        c_at = at.key("chains").key(name)
        for bone in chain.bones:
            if bone not in spec.joints:
                continue
            ranges = spec.joints[bone].range_deg
            for axis in chain.dofs[bone]:
                if axis not in ranges:
                    c_at.key("dofs").key(bone).fail(
                        f"{bone!r} turns on {axis} in chain {name!r}, but joint {bone!r} has "
                        f"no range on {axis}")
        for key, (lo, hi) in chain.limits.items():
            bone, _, axis = key.rpartition(":")
            if bone not in spec.joints:
                continue
            j_lo, j_hi = (math.radians(v) for v in spec.joints[bone].range_deg[axis])
            if lo < j_lo - LIMIT_SLACK_RAD or hi > j_hi + LIMIT_SLACK_RAD:
                c_at.key("limits").key(key).fail(
                    f"limit {key!r} [{lo}, {hi}] rad lies outside joint {bone!r} range on "
                    f"{axis} {list(spec.joints[bone].range_deg[axis])} deg")


def rig_root(spec: RigSpec) -> str:
    """The bone a seat and the side rules measure from: `spec.root`, else the one bone with no
    parent. Several parentless bones need `root`; the error names them."""
    names = [b.name for b in spec.bones]
    if spec.root is not None:
        if spec.root not in names:
            raise ManifestError(f"the rig's root names no bone: {spec.root!r}")
        return spec.root
    roots = [b.name for b in spec.bones if b.parent is None]
    if len(roots) != 1:
        raise ManifestError(f"the rig has {len(roots)} parentless bones, {roots}: name the root "
                            "bone in rig.root")
    return roots[0]


def bending_joints(spec: RigSpec) -> tuple[str, ...]:
    """The bones that bend the skin, in bone order. A bone bends it when it is not the root and
    not parentless, some part binds a bone in its subtree (itself included), and some part binds
    a bone outside it. A bone no part binds near, such as a prop's, is not a joint."""
    root = rig_root(spec)
    parent = {b.name: b.parent for b in spec.bones}

    def lineage(bone: str) -> set[str]:
        out = set()
        while bone is not None:
            out.add(bone)
            bone = parent[bone]
        return out

    bound = {bone for bind in spec.bind.values() for bone in bind.bones}
    lineages = {bone: lineage(bone) for bone in bound}
    out = []
    for b in spec.bones:
        if b.name == root or b.parent is None:
            continue
        inside = [x for x in bound if b.name in lineages[x]]
        if inside and len(inside) < len(bound):
            out.append(b.name)
    return tuple(out)


def _rest(at: _At, raw: Any, length: float) -> BoneRest:
    f = _fields(at, raw, ("head", "tail", "roll", "deform"), ("why",))
    head, tail = _vec(at.key("head"), f["head"], 3), _vec(at.key("tail"), f["tail"], 3)
    if not isinstance(f["deform"], bool):
        at.key("deform").fail(f"must be true or false, got {f['deform']!r}")
    gap = math.dist(head, tail)
    if abs(gap - length) > REST_LENGTH_TOLERANCE:
        at.fail(f"head to tail is {gap!r}, but the bone's length is {length!r}")
    return BoneRest(head, tail, _num(at.key("roll"), f["roll"]), f["deform"],
                    _text(at.key("why"), f["why"]) if "why" in f else None)


def _chain(at: _At, raw: Any, parents: dict[str, str | None], bone_list,
           groups: dict[str, tuple[str, ...]]) -> Chain:
    """A chain is a real parent line: root is an ancestor of bones[0], each bone's parent is
    the bone before it, and the effector is the last bone."""
    _fields(at, raw, ("root", "bones", "effector", "effector_radius", "dofs", "limits",
                      "effector_group"))
    root = _name(at.key("root"), raw["root"], _BONE)
    if root not in parents:
        at.key("root").fail(f"names no bone: {root!r}")
    bones = bone_list(at.key("bones"), raw["bones"])
    for i in range(1, len(bones)):
        if parents[bones[i]] != bones[i - 1]:
            at.key("bones").index(i).fail(
                f"{bones[i]!r} follows {bones[i - 1]!r}, but its parent is "
                f"{parents[bones[i]]!r}, not {bones[i - 1]!r}")
    ancestor = parents[bones[0]]
    while ancestor is not None and ancestor != root:
        ancestor = parents[ancestor]
    if ancestor is None:
        at.key("root").fail(f"{root!r} is not an ancestor of {bones[0]!r}, the chain's first bone")
    effector = _name(at.key("effector"), raw["effector"], _BONE)
    if effector != bones[-1]:
        at.key("effector").fail(f"must be the chain's last bone {bones[-1]!r}, got {effector!r}")
    dofs_raw = _obj(at.key("dofs"), raw["dofs"])
    if set(dofs_raw) != set(bones):
        at.key("dofs").fail(f"must name each chain bone once: {list(bones)}, got {list(dofs_raw)}")
    dofs = {}
    for bone in bones:
        axes = dofs_raw[bone]
        if (not isinstance(axes, str) or not axes or len(set(axes)) != len(axes)
                or not set(axes) <= set("xyz")):
            at.key("dofs").key(bone).fail(f"must be 'xyz' or a subset such as 'x', got {axes!r}")
        dofs[bone] = axes
    limits = {}
    for key, value in _obj(at.key("limits"), raw["limits"]).items():
        l_at = at.key("limits").key(key)
        bone, _, axis = key.rpartition(":")
        if bone not in dofs or len(axis) != 1 or axis not in dofs[bone]:
            l_at.fail("must be '<bone>:<axis>' with one axis the bone turns on")
        lo, hi = _vec(l_at, value, 2)
        if lo >= hi:
            l_at.fail(f"low {lo} must be under high {hi}")
        limits[key] = (lo, hi)
    group = _name(at.key("effector_group"), raw["effector_group"], _BONE)
    if group not in groups:
        at.key("effector_group").fail(f"names no contact group: {group!r}")
    return Chain(root=root, bones=bones, effector=effector,
                 effector_radius=_positive(at.key("effector_radius"), raw["effector_radius"]),
                 dofs=dofs, limits=limits, effector_group=group)


def _avoid(at: _At, raw: Any, bone_names: list[str]) -> Avoid:
    _fields(at, raw, ("id", "bone", "kind", "centre", "half"), ("why",))
    bone = _name(at.key("bone"), raw["bone"], _BONE)
    if bone not in bone_names:
        at.key("bone").fail(f"names no bone: {bone!r}")
    half = _vec(at.key("half"), raw["half"], 3)
    if not all(h > 0 for h in half):
        at.key("half").fail(f"must be three numbers above 0, got {list(half)}")
    return Avoid(id=_name(at.key("id"), raw["id"]), bone=bone,
                 kind=_choice(at.key("kind"), raw["kind"], AVOID_KINDS),
                 centre=_vec(at.key("centre"), raw["centre"], 3), half=half, why=_why(at, raw))
