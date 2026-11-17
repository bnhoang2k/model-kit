"""The light bake's driver: plan a run in plain Python, then start one Blender script.

    model-kit bake <scene> --stage FILE [--replace] [--compare [--lights FILE] [--compare-out DIR]]

`plan_bake` reads the scene folder and the stage file, checks the stage against the scene's
layouts (so a bad stage stops here, before Blender starts) and collects what the Blender script
needs into a `BakePlan`. `run_bake` hands the plan to `blender/bake_render.py` as one JSON
argument. That script places the stage's layout, finds the triangles a stage camera sees,
unwraps a lightmap UV on every still piece, bakes each light layer, and writes `<scene>/bake/`:
the placed layout as `<layout>.glb`, one `<layer>.exr` per layer and `bake.json`. It writes them
into `<scene>/bake.partial/`, which takes the place of `bake/` only when the run succeeds.

Where the .glb files are: a scene folder holds no .glb, so a pack's .glb is read from beside the
scene folder, as `<scene>/../<pack>.glb`, the way a bar keeps its `scene/` next to its `.glb`.

Units: the layout, the pack's .glb and the stage's lights and cameras (positions, `size`,
`radius_m`) share the project's units and go to Blender unconverted; only the density is in real
metres, so a piece's pixels per project unit are its pixels per metre divided by the style's
`units_per_metre`. A style with `units_per_metre` other than 1 is not tested yet.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_kit.blender import runner
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import check_stage, load_stage
from model_kit.core.scene import load_scene

SCRIPT = Path(__file__).resolve().parent / "blender" / "bake_render.py"
BAKE_DIR = "bake"
PARTIAL = ".partial"  # `bake.partial/`: where Blender writes until the run succeeds
OLD = ".old"  # `bake.old/`: the old bake, for the moment the new one takes its place
BAKE_TIMEOUT = 3600


@dataclass(frozen=True)
class BakePlan:
    """What one bake run does: the JSON spec the Blender script reads (`spec()`)."""

    scene: Path
    layout: str
    mode: str
    placements: tuple[dict[str, Any], ...]  # id, glb, piece, position, quaternion, px_per_unit
    live: tuple[str, ...]  # placement ids the bake leaves out
    cameras: tuple[dict[str, Any], ...]
    lights: tuple[dict[str, Any], ...]
    layers: tuple[str, ...]
    texels_per_m: float
    samples: int
    seed: int
    out: Path

    def spec(self) -> dict[str, Any]:
        return {
            "layout": self.layout, "mode": self.mode, "placements": list(self.placements),
            "live": list(self.live), "cameras": list(self.cameras), "lights": list(self.lights),
            "layers": list(self.layers), "texels_per_m": self.texels_per_m,
            "samples": self.samples, "seed": self.seed, "out": str(self.out),
        }


def plan_bake(scene: Path, stage_path: Path) -> BakePlan:
    """Plan a bake of `scene` by the stage file. A stage that names a layout, light layer or
    live piece the scene lacks, a still piece with no `view`, or a pack with no .glb raises
    `ManifestError`, naming it."""
    scene, stage_path = Path(scene), Path(stage_path)
    ctx = load_scene(scene, bake=False)  # an old or half-written bake never stops a new one
    stage = load_stage(stage_path)
    check_stage(stage, ctx.layouts)
    layout = ctx.layouts[stage.layout]
    per_unit = 1.0 / ctx.style.units_per_metre  # metres in one project unit
    placements = []
    for pid, place in layout.placements.items():
        glb = scene.parent / f"{place.pack}.glb"
        if not glb.is_file():
            raise ManifestError(f"{glb}: no .glb for pack {place.pack!r} (placement {pid!r}); "
                                "the bake reads it from beside the scene folder")
        px_per_unit = 0.0
        if pid not in stage.live:
            view = ctx.packs[place.pack].pieces[place.piece].view
            if view is None:
                raise ManifestError(
                    f"pack {place.pack!r}: piece {place.piece!r} (placement {pid!r}) has no "
                    "view, so its lightmap density has no viewing distance to follow; rebuild "
                    "the pack with a view")
            # pixels per metre at the doubled size, then per project unit
            px_per_unit = 2 * stage.texels_per_m_of(pid) / view.nearest_m * per_unit
        placements.append({
            "id": pid, "pack": place.pack, "piece": place.piece, "glb": str(glb),
            "position": list(place.position), "quaternion": list(place.quaternion),
            "px_per_unit": px_per_unit,
        })
    cameras = [{"id": c.id, "position": list(c.position), "target": list(c.target),
                "fov_deg": c.fov_deg, "aspect": c.aspect} for c in stage.cameras.values()]
    lights = [{"id": light.id, "kind": light.kind, "position": list(light.position),
               "direction": None if light.direction is None else list(light.direction),
               "size": None if light.size is None else list(light.size),
               "cone_deg": light.cone_deg, "blend": light.blend, "radius_m": light.radius_m,
               "sky_ground": light.sky_ground, "layer": light.layer}
              for light in stage.lights.values()]
    return BakePlan(
        scene=scene, layout=stage.layout, mode=stage.mode, placements=tuple(placements),
        live=tuple(sorted(stage.live)), cameras=tuple(cameras), lights=tuple(lights),
        layers=stage.layers, texels_per_m=stage.texels_per_m, samples=stage.samples,
        seed=stage.seed, out=scene / BAKE_DIR)


def run_bake(plan: BakePlan, *, replace: bool, timeout: float) -> Path:
    """Run the plan in Blender and return `<scene>/bake`. A scene that already has the folder
    raises, naming it, unless `replace` is true.

    Blender writes into `<scene>/bake.partial/`, which a stopped earlier run may have left and
    is cleared first. Only when Blender finishes and wrote `bake.json` does the new folder take
    the place of `bake/`: a failed or stopped run leaves the old bake as it was, and never half a
    bake under `bake/`."""
    out = plan.out
    partial = out.with_name(out.name + PARTIAL)
    old = out.with_name(out.name + OLD)
    if old.exists() and not out.exists():  # a run stopped mid-swap: the old bake goes back
        old.rename(out)
    if out.exists() and not replace:
        raise ManifestError(f"{out}: already exists; run with --replace to bake over it")
    for stale in (partial, old):  # left by a run that stopped before it finished
        if stale.exists():
            shutil.rmtree(stale)
    spec = {**plan.spec(), "out": str(partial)}
    result = runner.run_blender(SCRIPT, [json.dumps(spec)], timeout=timeout)
    manifest = partial / "bake.json"
    if not result.ok or not manifest.is_file():
        shutil.rmtree(partial, ignore_errors=True)  # never leave half a bake for a later run
        kept = f"; {out} is left as it was" if out.exists() else ""
        if result.ok:
            raise ManifestError(f"the bake run wrote no {manifest.name}{kept}")
        raise ManifestError(f"the bake run failed{kept}:\n{result.log[-3000:]}")
    if out.exists():  # the old bake steps aside only once the new one is whole
        out.rename(old)
        partial.rename(out)
        shutil.rmtree(old)
    else:
        partial.rename(out)
    manifest = out / "bake.json"
    for line in result.log.splitlines():
        # one per layer: its name and bytes; one per object: its smooth regions' shapes
        if line.startswith(("PICTURE ", "REGIONS ")):
            print(line)
    size = json.loads(manifest.read_text())["size"]
    print(f"atlas {size} px square (packed at {2 * size} px, each island padded 4 px)")
    return out
