"""The Blender against three.js comparison: `model-kit bake <scene> --stage FILE --compare`.

`compare` renders each stage camera of a baked scene twice: in Blender (Cycles, the stage's lights
at each layer's colour and strength) and in headless Chromium (`js/compare.html`, which loads the
bake through the shipped mixer and lights the live pieces with three.js lights). It writes both
pictures as PNG and returns, per camera, the mean absolute difference of the two in sRGB, 0 to 1,
over the pixels the scene covers in both (`masked_difference`). Neither side draws a background,
so the number measures the scene, not the sky behind it. It prints a number and never judges
one: a difference is where to look, not a failure. The Blender side is Cycles at 64 samples with
no denoise, so its own noise sets a floor under the number.

Needs Blender (`MODEL_KIT_BLENDER`) and Playwright with its Chromium (`uv sync --extra browser`,
then `playwright install chromium` once). `serve` and `browser_page` are the pieces the mixer's
tests use to load the shipped page in the same browser.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import http.server
import json
import struct
import threading
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from model_kit.blender import runner
from model_kit.core.errors import ManifestError
from model_kit.core.manifest import load_stage

JS_DIR = Path(__file__).resolve().parent / "js"
SCRIPT = Path(__file__).resolve().parent / "blender" / "compare_render.py"
BAKE_DIR = "bake"
COMPARE_TIMEOUT = 1800
# Software GL, so the page draws on a machine with no GPU and draws the same on every one.
CHROMIUM_ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                 "--ignore-gpu-blocklist"]
TYPES = {".js": "text/javascript", ".html": "text/html", ".json": "application/json",
         ".glb": "model/gltf-binary", ".exr": "application/octet-stream", ".png": "image/png"}


def read_values(path: Path | None, layers: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """The `--lights` file: {layer name: {"color": [r, g, b], "strength": s}}, colour in linear
    RGB. A layer the file lacks is white at 1. A name that is no layer of the bake raises."""
    if path is None:
        return {}
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path}: cannot read the lights file: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: expected an object of layer name to colour and strength")
    values = {}
    for name, value in raw.items():
        if name not in layers:
            raise ManifestError(f"{path}: {name!r} is no layer of this bake (it has {list(layers)})")
        color = value.get("color", [1, 1, 1]) if isinstance(value, dict) else None
        strength = value.get("strength", 1) if isinstance(value, dict) else None
        ok = (isinstance(color, list) and len(color) == 3
              and all(isinstance(c, (int, float)) and c >= 0 for c in color)
              and isinstance(strength, (int, float)) and strength >= 0)
        if not ok:
            raise ManifestError(f"{path}: layer {name!r} needs {{\"color\": [r, g, b], "
                                "\"strength\": s}} with non-negative numbers")
        values[name] = {"color": [float(c) for c in color], "strength": float(strength)}
    return values


def read_png(data: bytes) -> np.ndarray:
    """An 8-bit RGB or RGBA PNG (not interlaced) as a (height, width, 4) float array, 0 to 1:
    red, green, blue and alpha. An RGB picture reads as alpha 1 everywhere."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ManifestError("not a PNG")
    at, chunks = 8, []
    width = height = channels = 0
    while at < len(data):
        size, kind = struct.unpack(">I4s", data[at:at + 8])
        body = data[at + 8:at + 8 + size]
        at += 12 + size
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or color not in (2, 6) or interlace:
                raise ManifestError("only 8-bit RGB or RGBA PNG pictures without interlace")
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            chunks.append(body)
    raw = zlib.decompress(b"".join(chunks))
    stride = width * channels
    out = np.zeros((height, stride), dtype=np.int64)
    for y in range(height):
        row = raw[y * (stride + 1):(y + 1) * (stride + 1)]
        kind, line = row[0], np.frombuffer(row[1:], dtype=np.uint8).astype(np.int64)
        up = out[y - 1] if y else np.zeros(stride, dtype=np.int64)
        if kind == 0:
            cur = line
        elif kind == 2:
            cur = (line + up) & 255
        else:  # 1 sub, 3 average, 4 paeth need the left pixel: one pixel at a time
            cur = np.zeros(stride, dtype=np.int64)
            for i in range(stride):
                a = cur[i - channels] if i >= channels else 0
                b = up[i]
                c = up[i - channels] if i >= channels else 0
                if kind == 1:
                    guess = a
                elif kind == 3:
                    guess = (a + b) // 2
                elif kind == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    guess = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                else:
                    raise ManifestError(f"PNG filter {kind} unknown")
                cur[i] = (line[i] + guess) & 255
        out[y] = cur
    pixels = out.reshape(height, width, channels) / 255.0
    if channels == 3:
        pixels = np.concatenate([pixels, np.ones((height, width, 1))], axis=2)
    return pixels


@dataclass(frozen=True)
class Difference:
    """One camera's comparison. `mean` is the mean absolute difference of red, green and blue
    (sRGB, 0 to 1) over the pixels both pictures cover fully, or None when they share none.
    `both` is the share of the picture's pixels covered fully in both; `only_one` the share
    covered fully in one picture and not fully in the other."""

    mean: float | None
    both: float
    only_one: float


def masked_difference(a: np.ndarray, b: np.ndarray) -> Difference:
    """Compare two (height, width, 4) pictures (`read_png`) on the pixels both cover fully.

    A pixel counts when its alpha is 1 (255 in the file) in both pictures. Neither side draws a
    background (Blender renders with a transparent film, three.js clears to alpha 0), so this
    drops the background. It also drops the anti-aliased silhouette edges, where each renderer
    blends the edge its own way. A large `only_one` share says the two pictures do not frame the
    same scene: a camera or a placement that differs.
    """
    if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 4:
        raise ManifestError(f"the two pictures differ in size or channels ({a.shape} against "
                            f"{b.shape}); each needs (height, width, 4)")
    full_a, full_b = a[:, :, 3] >= 1.0, b[:, :, 3] >= 1.0
    both = full_a & full_b
    total = both.size
    mean = float(np.abs(a[both, :3] - b[both, :3]).mean()) if both.any() else None
    return Difference(mean=mean, both=float(both.sum()) / total,
                      only_one=float((full_a ^ full_b).sum()) / total)


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Serves each route prefix from its folder; reads nothing else."""

    routes: ClassVar[dict[str, Path]] = {}

    def translate_path(self, path: str) -> str:
        path = path.split("?", 1)[0].split("#", 1)[0]
        for prefix, folder in self.routes.items():
            if path.startswith(prefix):
                target = (folder / path[len(prefix):].lstrip("/")).resolve()
                if folder.resolve() in (target, *target.parents):
                    return str(target)
        return str(Path("/nonexistent"))

    def guess_type(self, path: str) -> str:
        return TYPES.get(Path(path).suffix, "application/octet-stream")

    def log_message(self, *args: Any) -> None:
        pass


@contextlib.contextmanager
def serve(routes: dict[str, Path]) -> Iterator[str]:
    """Serve `{"/js/": folder, ...}` on a free local port for the length of the block; yields
    the base URL (`http://127.0.0.1:<port>`)."""
    handler = type("Handler", (_Handler,), {"routes": {k: Path(v) for k, v in routes.items()}})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ManifestError("the comparison needs Playwright: run `uv sync --extra browser`, then "
                            "`playwright install chromium` once") from exc
    return sync_playwright


@contextlib.contextmanager
def browser_page() -> Iterator[Any]:
    """A page of headless Chromium with software WebGL. A browser that will not start raises
    `ManifestError` naming what to run."""
    sync_playwright = _playwright()
    with sync_playwright() as play:
        try:
            browser = play.chromium.launch(args=CHROMIUM_ARGS)
        except Exception as exc:
            raise ManifestError("Chromium would not start (run `playwright install chromium` "
                                f"once): {str(exc).splitlines()[0]}") from exc
        try:
            yield browser.new_page()
        finally:
            browser.close()


@functools.cache
def chromium_available() -> bool:
    """Whether Playwright is installed and its Chromium starts here."""
    try:
        with browser_page():
            return True
    except ManifestError:
        return False


def _sizes(cameras: dict[str, Any], long_side: int) -> list[dict[str, Any]]:
    out = []
    for cam in cameras.values():
        wide = cam.aspect >= 1
        width = long_side if wide else max(1, round(long_side * cam.aspect))
        height = max(1, round(long_side / cam.aspect)) if wide else long_side
        out.append({"id": cam.id, "position": list(cam.position), "target": list(cam.target),
                    "fov_deg": cam.fov_deg, "aspect": cam.aspect, "width": width,
                    "height": height})
    return out


def _lights(stage) -> list[dict[str, Any]]:
    return [{"id": li.id, "kind": li.kind, "position": list(li.position),
             "direction": None if li.direction is None else list(li.direction),
             "size": None if li.size is None else list(li.size), "cone_deg": li.cone_deg,
             "blend": li.blend, "radius_m": li.radius_m, "sky_ground": li.sky_ground,
             "layer": li.layer} for li in stage.lights.values()]


def compare(scene: Path, stage_path: Path, out: Path, lights: Path | None = None, *,
            long_side: int = 320, samples: int = 64) -> list[tuple[str, Difference]]:
    """Render each camera of the stage in Blender and in Chromium into `out`
    (`blender-<camera>.png`, `three-<camera>.png`) and return `(camera, Difference)` for each
    (`masked_difference`). `lights` is a file of layer values (`read_values`). Prints how many
    of the bake's parts the page gave a light map: fewer than all means the page and the bake
    disagree on part names, and the number then measures parts drawn with no bake."""
    scene, out = Path(scene), Path(out)
    bake_dir = scene / BAKE_DIR
    info_path = bake_dir / "bake.json"
    if not info_path.is_file():
        raise ManifestError(f"{info_path}: no bake there; run `model-kit bake` first")
    info = json.loads(info_path.read_text())
    stage = load_stage(Path(stage_path))
    if stage.layout != info["layout"]:
        raise ManifestError(f"{stage_path} places layout {stage.layout!r}, but the bake is of "
                            f"{info['layout']!r}")
    values = read_values(lights, tuple(info["layers"]))
    cameras = _sizes(stage.cameras, long_side)
    spec = {"glb": str(bake_dir / f"{info['layout']}.glb"), "cameras": cameras,
            "lights": _lights(stage), "values": values, "out": str(out), "samples": samples}
    with browser_page():  # fail before the long Blender render if there is no browser
        pass
    out.mkdir(parents=True, exist_ok=True)
    result = runner.run_blender(SCRIPT, [json.dumps(spec)], timeout=COMPARE_TIMEOUT)
    if not result.ok:
        raise ManifestError(f"the Blender render failed:\n{result.log[-3000:]}")
    config = {"bake": "/bake/bake.json", "values": values, "lights": spec["lights"],
              "cameras": cameras}
    with serve({"/js/": JS_DIR, "/bake/": bake_dir}) as base, browser_page() as page:
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(f"{base}/js/compare.html")
        page.wait_for_function("window.compareReady === true")
        try:
            shot = page.evaluate("config => window.renderCompare(config)", config)
        except Exception as exc:
            raise ManifestError(f"the page failed: {str(exc).splitlines()[0]}; "
                                f"{'; '.join(errors)}") from exc
    for light in shot["skipped"]:
        print(f"COMPARE note: light {light!r} (area) has no live stand-in in the browser; "
              "live pieces get no light from it", flush=True)
    print(f"COMPARE light maps: {shot['applied']} of {shot['baked']} baked parts", flush=True)
    rows = []
    for cam in cameras:
        data = shot["pictures"][cam["id"]].split(",", 1)[1]
        mine = out / f"three-{cam['id']}.png"
        mine.write_bytes(base64.b64decode(data))
        a = read_png((out / f"blender-{cam['id']}.png").read_bytes())
        b = read_png(mine.read_bytes())
        if a.shape != b.shape:
            raise ManifestError(f"camera {cam['id']!r}: the two pictures differ in size "
                                f"({a.shape} against {b.shape})")
        rows.append((cam["id"], masked_difference(a, b)))
    return rows
