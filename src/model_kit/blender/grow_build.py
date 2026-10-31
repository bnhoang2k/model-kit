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
                }]}

Growing never runs here: each skeleton was grown and saved in the system Python
(`model_kit.core.grow`). This script reads it, makes the pieces through `blender.grow.add_grown`,
and writes `<pack>.pack.json`, `<pack>.glb` (Draco) and `<pack>.blend` into `out` through
`PackBuilder.finish`, in the order the spec lists the pieces. Then it exports the pieces into
`scene` (`blender.export.export_pieces`), where the caller stages the manifests.

Material per part: the tube levels from the twig level down (the twigs and the shoots) take the
`Twig` colour, every other tube level takes `Bark`, and a leaf part takes the colour named for its
shade. A colour the spec lacks raises. So does a piece with no `view`: the build never assumes the
distance a piece is seen from (an optional `hidden` list and `hidden_why` name sides never seen; an
optional `farthest_m` and `farthest_why` give the farthest distance it is seen from).
"""

import json
from pathlib import Path

from model_kit.blender import build, grow, runner
from model_kit.blender.export import export_pieces
from model_kit.core.grow.mesh import leaf_lean, leaf_part_name, part_names
from model_kit.core.grow.rules import load_rules
from model_kit.core.grow.skeleton import SHOOT, TWIG, load_skeleton
from model_kit.core.manifest import View, parse_view
from model_kit.core.units import load_style

BARK, TWIG_ROLE = "Bark", "Twig"


def material_roles(sk) -> dict[str, str]:
    """The colour role of each part of a grown piece."""
    twiggy = {sk.level_names[TWIG], sk.level_names[SHOOT]}
    roles = {leaf_part_name(shade): shade for shade in sk.shades}
    for name in part_names(sk):
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

    build.reset()
    mats = build.slots({role: tuple(rgb) for role, rgb in spec["colours"].items()})
    pack = build.PackBuilder(spec["pack"], spec["units_per_metre"], Path(spec["out"]),
                             spec["spacing"], crease_deg=style.crease_deg)
    order, callbacks = [], []
    for i, piece in enumerate(spec["pieces"]):
        sk = load_skeleton(Path(piece["skeleton"]))
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
    export_pieces(written, Path(spec["scene"]), pieces=order)
    runner.done()


main()
