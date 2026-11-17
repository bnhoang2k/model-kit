"""Make a pack of grown pieces in Blender from saved skeletons. Run it through `run_blender`:

    blender --background --python grow_build.py -- <spec.json>

The spec is a JSON object:

    {"pack": "redbud", "units_per_metre": 0.5, "spacing": 5.0,
     "rules": "<path to the .rules.json the trees were grown from>",
     "kit": "<optional: the kit file the rules merge, for a bud-grown tree>",
     "style": "<path to the bar's style.json; its dials.leaves is the leaf-normal lean and its
               dials.edges.crease_deg the pack's crease angle>",
     "out": "<folder>",
     "scene": "<folder for the exported pieces>",
     "colours": {"Bark": [r, g, b], "Twig": [...], "<leaf shade>": [...]},
     "pieces": [{"skeleton": "<path to a .npz>", "size_class": "hero",
                 "done": "<what done looks like>", "origin": "<where the frame's origin sits>",
                 "view": {"nearest_m": <metres>, "why": "<how the piece is seen>",
                          "farthest_m": <metres>, "farthest_why": "<how it is seen from there>"}
                },
                {"cards_of": "<the name of another piece of this spec>", "name": "<optional;
                 default <source>_Cards>", "size_class": ..., "done": ..., "origin": ...,
                 "view": {...}, "outlined_why": "<how the shipping look draws it without lines>"}]}

A piece with `cards_of` is the card copy of that piece (`blender.grow.add_cards`): it has no
`skeleton` key and uses its source's skeleton, the style's `dials.cards` and `paint_cells` with
the skeleton's seed for its leaf image. The build prints `<piece>: <n> vertices (wood <w>, leaves
<l>)` for each: the vertices the .glb holds for it, as a game loads them (a count, never compared
with a limit).

Growing never runs here: each skeleton was grown and saved in the system Python
(`model_kit.core.grow`). This script reads it, makes the pieces through `blender.grow.add_grown`,
and writes `<pack>.pack.json`, `<pack>.glb` (Draco) and `<pack>.blend` into `out` through
`PackBuilder.finish`, in the order the spec lists the pieces. Then it exports the pieces into
`scene` (`blender.export.export_pieces`), where the caller stages the manifests.

Material per part: the tube levels from the twig level down (the twigs and the shoots) take the
`Twig` colour, every other tube level takes `Bark`, and a leaf part takes the colour named for its
shade (a card piece's leaf part, `Cards_<shade>`, takes the same colour, in its own material). A
colour the spec lacks raises. So does a piece with no `view`: the build never assumes the
distance a piece is seen from (an optional `hidden` list and `hidden_why` name sides never seen; an
optional `farthest_m` and `farthest_why` give the farthest distance it is seen from).
"""

import json
from pathlib import Path

from model_kit.blender import build, grow, runner
from model_kit.blender.export import export_pieces
from model_kit.core.errors import ManifestError
from model_kit.core.grow.card_image import paint_cells
from model_kit.core.grow.cards import card_dials, card_part_name
from model_kit.core.grow.mesh import leaf_lean, leaf_part_name, part_names
from model_kit.core.grow.rules import load_rules
from model_kit.core.grow.skeleton import SHOOT, TWIG, load_skeleton
from model_kit.core.manifest import View, parse_view
from model_kit.core.units import load_style

BARK, TWIG_ROLE = "Bark", "Twig"


def material_roles(sk, *, cards: bool = False) -> dict[str, str]:
    """The colour role of each part of a grown piece, or of its card copy (`cards`: the leaf part
    is `Cards_<shade>` and every level is listed, since the copy grows the limbs that hold
    leaves)."""
    twiggy = {sk.level_names[TWIG], sk.level_names[SHOOT]}
    if cards:
        roles = {card_part_name(shade): shade for shade in sk.shades}
        names = list(sk.level_names)
    else:
        roles = {leaf_part_name(shade): shade for shade in sk.shades}
        names = part_names(sk)
    for name in names:
        if name not in roles:
            roles[name] = TWIG_ROLE if name in twiggy else BARK
    return roles


def _view(raw, spec_path: Path, i: int) -> View | None:
    """The spec's `view` object as a `View` (the manifest's own reader, so a pack and a spec take
    the same rules), or None when the spec has none (`add_grown` then raises, naming the piece)."""
    if raw is None:
        return None
    return parse_view(raw, spec_path, f"pieces[{i}].view")


def main() -> None:
    (spec_path,) = runner.script_args()
    spec = json.loads(Path(spec_path).read_text())
    kit = spec.get("kit")  # the kit file to merge; the default is `rules.KIT_RULES`
    rules = load_rules(Path(spec["rules"]), **({"kit": Path(kit)} if kit else {}))
    style = load_style(Path(spec["style"]))
    lean = leaf_lean(style)
    loaded = [None if "cards_of" in piece else load_skeleton(Path(piece["skeleton"]))
              for piece in spec["pieces"]]  # None for a card piece: it uses its source's
    skeletons = {sk.info["piece"]: sk for sk in loaded if sk is not None}

    build.reset()
    mats = build.slots({role: tuple(rgb) for role, rgb in spec["colours"].items()})
    pack = build.PackBuilder(spec["pack"], spec["units_per_metre"], Path(spec["out"]),
                             spec["spacing"], crease_deg=style.crease_deg)
    order, callbacks, cards = [], [], []
    for i, piece in enumerate(spec["pieces"]):
        if "cards_of" in piece:
            source = piece["cards_of"]
            if "skeleton" in piece or source not in skeletons:
                raise ManifestError(
                    f"pieces[{i}]: a card piece names a grown piece of this spec in cards_of "
                    f"(have {sorted(skeletons)}) and has no skeleton of its own; got "
                    f"cards_of={source!r}, skeleton={piece.get('skeleton')!r}")
            sk = skeletons[source]
            by_part = {part: mats[role]
                       for part, role in material_roles(sk, cards=True).items()}
            root, set_corners = grow.add_cards(
                pack, sk, source=source, rules=rules, materials=by_part,
                image=paint_cells(sk.outline, seed=int(sk.info["seed"])), done=piece["done"],
                size_class=piece["size_class"], origin=piece["origin"],
                dials=card_dials(style), lean=lean, name=piece.get("name"),
                outlined_why=piece.get("outlined_why", ""),
                view=_view(piece.get("view"), Path(spec_path), i))
            order.append(root.name)
            callbacks.append(set_corners)
            cards.append(root)
            continue
        sk = loaded[i]
        by_part = {part: mats[role] for part, role in material_roles(sk).items()}
        root, set_normals = grow.add_grown(
            pack, sk, rules=rules, materials=by_part, done=piece["done"],
            size_class=piece["size_class"], origin=piece["origin"], lean=lean,
            view=_view(piece.get("view"), Path(spec_path), i))
        order.append(root.name)
        callbacks.append(set_normals)

    def before_export() -> None:
        for callback in callbacks:
            callback()

    written = pack.finish(order, glb_options=build.DRACO, before_export=before_export)
    for root in cards:
        total, wood, leaves = grow.vertex_counts(root, pack.out_dir / f"{pack.name}.glb")
        print(f"{root.name}: {total} vertices (wood {wood}, leaves {leaves})")
    export_pieces(written, Path(spec["scene"]), pieces=order)
    runner.done()


main()
