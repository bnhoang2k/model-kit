"""`anchor()`: the one lookup for holds, seats, mounts and uses."""

from __future__ import annotations

from collections.abc import Mapping

from model_kit.core.errors import AnchorMissing, ManifestError

from .records import Anchor, Clip, Pack, Ref


def anchor(packs: Mapping[str, Pack], ref: str | Ref, *, clip: Clip | None = None) -> Anchor:
    """The one lookup for holds, seats, mounts and uses. A miss raises; there is no default.

    With no `clip`, the ref's piece id is a piece name, `world` is None and `placed_by` is
    None. With a `clip`, the ref must name a piece id the clip places in `floor_gear`: the
    piece is the placement's piece, `world` is the item's point through the placement, and
    `placed_by` is the placement key. The one exception is a piece the clip carries as a
    prop: `world` is None and `placed_by` is `prop:<prop id>`, because a prop's pose comes
    per sample from the scene. Any other piece raises. Bone refs (`rig:<bone>`) are
    resolved by the caller from samples, never here.
    """
    ref = ref if isinstance(ref, Ref) else Ref.parse(ref)
    if ref.is_bone:
        raise ManifestError(f"{ref}: a bone ref is resolved from samples by the caller, "
                            "not by anchor()")
    pack = packs.get(ref.pack)
    if pack is None:
        raise AnchorMissing(f"{ref}: no pack {ref.pack!r}; loaded packs are {sorted(packs)}")

    piece_name, placement, placed_by = ref.piece, None, None
    if clip is not None:
        placement = clip.floor_gear.pieces.get(ref.piece)
        if placement is not None:
            if placement.pack != ref.pack:
                raise ManifestError(f"{ref}: placement {ref.piece!r} is of pack "
                                    f"{placement.pack!r}, not {ref.pack!r}")
            piece_name, placed_by = placement.piece, ref.piece
        else:
            carried = [k for k, p in clip.props.items()
                       if (p.pack, p.piece) == (ref.pack, ref.piece)]
            if not carried:
                raise AnchorMissing(
                    f"{ref}: {ref.piece!r} is not placed by clip {clip.name!r} "
                    f"(placed: {sorted(clip.floor_gear.pieces)}; carried: "
                    f"{sorted(f'{p.pack}:{p.piece}' for p in clip.props.values())})")
            placed_by = f"prop:{carried[0]}"

    piece = pack.pieces.get(piece_name)
    if piece is None:
        how = f" (placement {ref.piece!r})" if placement is not None else ""
        raise AnchorMissing(f"{ref}: no piece {piece_name!r}{how} in pack {ref.pack!r}")
    for kind, items in (("hold", piece.holds), ("seat", piece.seats), ("mount", piece.mounts),
                        ("use", piece.uses)):
        for item in items:
            if item.id == ref.id:
                world = None if placement is None else placement.to_world(item.point)
                return Anchor(kind=kind, piece=piece, item=item, world=world, placed_by=placed_by)
    raise AnchorMissing(f"{ref}: no hold, seat, mount or use {ref.id!r} on piece {piece_name!r}")
