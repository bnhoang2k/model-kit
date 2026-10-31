"""Render the README's showcase loops from the bars, as animated WebP files in media/.

    MODEL_KIT_BLENDER=/path/to/blender uv run --extra sheet python tools/showcase.py [shot ...]
        [--out media] [--samples N] [--frames N]

With no shot named, every shot renders. Each shot runs tools/showcase_render.py in Blender, which
writes PNG frames with a transparent background. This script lays them on GitHub's light and dark
page colours and joins each set into one looping WebP, `<shot>-light.webp` and `<shot>-dark.webp`.

The first loop in the README is a capture of a scene built with the kit. That scene is not in
this repository, so it has no shot here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from PIL import Image

from model_kit.blender import runner

ROOT = Path(__file__).resolve().parents[1]
RENDER = Path(__file__).resolve().parent / "showcase_render.py"
TIMEOUT = 3600
CHIBI = str(ROOT / "bars" / "03-chibi" / "chibi.glb")

# One accent: the guy in front is yellow; the two behind wear muted crowd tints.
SHOTS = {
    "redbud": {"mode": "turntable", "frames": 96, "fps": 16, "size": (640, 480), "width": 640, "tilt_deg": 12,
               "fill": 0.7, "models": [{"path": str(ROOT / "bars" / "02-redbud" / "redbud.glb"), "piece": "Redbud"}]},
    "sweetgum": {"mode": "turntable", "frames": 96, "fps": 16, "size": (640, 480), "width": 640, "tilt_deg": 12,
                 "fill": 0.7, "models": [{"path": str(ROOT / "bars" / "04-sweetgum" / "sweetgum.glb"),
                             "piece": "Sweetgum"}]},
    "guys": {"mode": "clip", "frames": 73, "fps": 30, "size": (880, 495), "width": 880,
             "tilt_deg": 16,
             "turn_deg": 20, "fill": 0.72,
             "models": [
                 {"path": CHIBI, "offset": (-0.62, 0.55, 0), "shift": 24, "tint": "#b7a4b8",
                  "tint_material": "Fill"},
                 {"path": CHIBI, "offset": (0.62, 0.45, 0), "shift": 49, "tint": "#a3b5a6",
                  "tint_material": "Fill"},
                 {"path": CHIBI, "offset": (0, 0, 0), "shift": 0, "tint": "#ffe27a",
                  "tint_material": "Fill"},
             ]},
}


# GitHub's page colours. Each loop ships twice, laid on each, and the README picks one with
# <picture>: an opaque loop compresses far better than one with a soft alpha shadow.
GROUNDS = {"light": (255, 255, 255), "dark": (13, 17, 23)}
# Compressed bytes can spell an email address by chance, and a scan of the repository's files for
# email addresses reads binaries too.
EMAILISH = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def save_webp(frames: list[Image.Image], out: Path, fps: int, quality: int = 75) -> None:
    """Save a looping WebP, one quality step lower each time its bytes look like an email."""
    while True:
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
                       loop=0, quality=quality, method=4)
        if not EMAILISH.search(out.read_bytes()):
            return
        quality -= 1


def join_webp(frames: Path, out_dir: Path, name: str, fps: int, width: int) -> list[str]:
    pngs = sorted(frames.glob("f_*.png"))
    if not pngs:
        raise RuntimeError(f"{frames}: Blender wrote no frames")
    images = [Image.open(p).convert("RGBA") for p in pngs]
    height = round(images[0].height * width / images[0].width)
    images = [im.resize((width, height), Image.LANCZOS) for im in images]
    written = []
    for ground, colour in GROUNDS.items():
        laid = []
        for im in images:
            base = Image.new("RGBA", im.size, (*colour, 255))
            base.alpha_composite(im)
            laid.append(base.convert("RGB"))
        out = out_dir / f"{name}-{ground}.webp"
        save_webp(laid, out, fps)
        written.append(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")
    return written


def render(name: str, out_dir: Path, samples: int | None, frames: int | None) -> None:
    spec = dict(SHOTS[name])
    if samples is not None:
        spec["samples"] = samples
    if frames is not None:
        spec["frames"] = frames
    with tempfile.TemporaryDirectory(prefix=f"showcase-{name}-") as tmp:
        shot = Path(tmp) / "shot.json"
        shot.write_text(json.dumps(spec))
        frames = Path(tmp) / "frames"
        frames.mkdir()
        result = runner.run_blender(RENDER, [str(shot), str(frames)], timeout=TIMEOUT)
        if not result.ok:
            sys.exit(f"{name}: the render failed:\n{result.log[-3000:]}")
        written = join_webp(frames, out_dir, name, spec["fps"], spec["width"])
    print(f"{name}: {spec['frames']} frames: {', '.join(written)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("shots", nargs="*", help=f"any of {', '.join(SHOTS)} (default: all)")
    parser.add_argument("--out", type=Path, default=ROOT / "media")
    parser.add_argument("--samples", type=int, help="render samples (default 32, denoised)")
    parser.add_argument("--frames", type=int, help="render only this many frames, for a quick look")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    unknown = [s for s in args.shots if s not in SHOTS]
    if unknown:
        parser.error(f"unknown shot {', '.join(unknown)}; the shots are {', '.join(SHOTS)}")
    for name in args.shots or SHOTS:
        render(name, args.out, args.samples, args.frames)


if __name__ == "__main__":
    main()
