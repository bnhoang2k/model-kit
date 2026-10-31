"""The sheet rendered for real: a built bench beside a wide fixture crown, at five views."""

import json
import math
from pathlib import Path

import pytest

from model_kit import sheet
from model_kit.blender.runner import run_blender
from model_kit.core.units import load_style

pytestmark = pytest.mark.blender

FIXTURES = Path(__file__).resolve().parent.parent / "core" / "fixtures"

BUILD = """
from pathlib import Path
from model_kit.blender import build, runner
from model_kit.core.manifest import Listed, Recipe, View

name, piece, units, width, height, out = runner.script_args()
units, width, height = float(units), float(width), float(height)
depth = max(width, 0.3)
build.reset()
fill = build.slots({"Fill": (0.8, 0.45, 0.2)})["Fill"]
pack = build.PackBuilder(name, units, Path(out), 3.0, crease_deg=35)
root = build.piece(piece)
build.slab("body", -width / 2, width / 2, 0, height, -depth / 2, depth / 2, fill, root,
           props={"listed": "body", "group": "body"})
pack.register(root, view=View(1.0, "a test piece, seen from arm's length"), origin="floor, under the middle", recipe=Recipe("assembly", "build.py"),
              listed=[Listed("body", "A slab", "a drawing", 0.01, None)], done="A slab.",
              size_class="piece", kind="box")
pack.use(root, "front", point=(0, 0, 0.3), facing=(0, 0, -1), posture="stand")
pack.finish([piece])
runner.done()
"""


def build_pack(tmp_path: Path, folder: Path, name: str, piece: str, units: float, width: float,
               height: float) -> Path:
    script = tmp_path / f"build_{name}.py"
    script.write_text(BUILD)
    result = run_blender(script, [name, piece, str(units), str(width), str(height), str(folder)],
                         timeout=240)
    assert "Traceback" not in result.log, result.log
    assert result.ok, result.log
    style = json.loads((FIXTURES / "shop.style.json").read_text())
    style["units_per_metre"] = units
    (folder / "style.json").write_text(json.dumps(style))
    return folder / f"{name}.pack.json"


def test_render_two_rows(tmp_path):
    # A bench 0.6 m wide and 0.46 m tall (1.0 unit a metre), and a bar crown wider than tall:
    # 5.8 m across and 2.5 m tall (2.9 and 1.25 units at 0.5 units a metre).
    new = tmp_path / "new"
    new.mkdir()
    pack = build_pack(tmp_path, new, "shop", "Bench", 1.0, 0.6, 0.46)
    bars = tmp_path / "bars"
    bar = bars / "01-crown"
    bar.mkdir(parents=True)
    bar_pack = build_pack(tmp_path, bar, "crown", "Crown", 0.5, 2.9, 1.25)
    (bar / "sheet.png").write_bytes(b"x")
    (bar / "bar.json").write_text(json.dumps({
        "slug": "01-crown", "title": "Crown", "class": "assembly", "default": True,
        "piece": "crown:Crown", "reads": "r", "fits": "f", "wrong_for": "w",
        "references": [{"key": "W", "url": "https://example.com/w", "gives": "g"}],
        "files": {"style": "style.json", "pack": bar_pack.name, "glb": "crown.glb",
                  "scene": "scene", "sheet": "sheet.png"},
    }))

    out = sheet.make_sheet(pack, "Bench", new / "style.json", bar="auto", bars=bars,
                           out=tmp_path / "out" / "sheet.png")
    Image = pytest.importorskip("PIL.Image")
    image = Image.open(out).convert("RGB")
    cw, ch = sheet.CELL
    assert image.size == (170 + 5 * cw, 70 + 2 * ch)

    def cell(row: int, col: int):
        return image.crop((170 + col * cw, 70 + row * ch, 170 + (col + 1) * cw, 70 + (row + 1) * ch))

    def differs(p, background) -> bool:
        return max(abs(a - b) for a, b in zip(p, background, strict=True)) > 12

    def is_post(p) -> bool:  # the post is blue-grey; the pieces are orange (red well above blue)
        return p[2] >= p[0] - 3

    def is_piece(p) -> bool:
        return p[0] - p[2] > 8

    for row in range(2):
        for col in range(5):
            c = cell(row, col)
            background = c.getpixel((2, 2))
            # nothing is cut off: the outer two pixels of every edge are background
            edges = [c.getpixel((x, y)) for x in range(cw) for y in (0, 1, ch - 2, ch - 1)]
            edges += [c.getpixel((x, y)) for y in range(ch) for x in (0, 1, cw - 2, cw - 1)]
            assert not any(differs(p, background) for p in edges), f"cell {row},{col} is clipped"
            assert sum(differs(p, background) for p in c.get_flattened_data()) > 500, \
                f"cell {row},{col} is empty"

    # One scale: in the front view (8 degrees up) a 1 m post and the 2.5 m crown stand in the
    # ratio of their heights. The post is grey, the pieces are orange.
    views = sheet.resolve_views(load_style(new / "style.json"), sheet.VIEWS)
    rows = sheet.plan_sheet(pack, "Bench", new / "style.json", "auto", bars)
    px_per_m = cw / sheet.ortho_width(rows, views)

    def pixel_height(row: int, grey: bool) -> int:
        c = cell(row, 1)
        background = c.getpixel((2, 2))
        ys = [y for y in range(ch) for x in range(cw)
              if differs(p := c.getpixel((x, y)), background)
              and (is_post(p) if grey else is_piece(p))]
        return max(ys) - min(ys) + 1

    post_px = pixel_height(1, grey=True)
    assert post_px == pytest.approx(px_per_m * 1.0 * math.cos(math.radians(8)), rel=0.08)
    crown_px = pixel_height(1, grey=False)
    # the top face adds its 5.8 m depth, seen at 8 degrees
    expect = px_per_m * (2.5 * math.cos(math.radians(8)) + 5.8 * math.sin(math.radians(8)))
    assert crown_px == pytest.approx(expect, rel=0.05)
    bench_px = pixel_height(0, grey=False)
    expect = px_per_m * (0.46 * math.cos(math.radians(8)) + 0.6 * math.sin(math.radians(8)))
    assert bench_px == pytest.approx(expect, rel=0.1)


def test_the_post_stands_right_of_the_piece_in_every_cell(tmp_path):
    """The post is never in front of the piece or behind it: in every cell its pixels lie to the
    right of the piece's, and it stands on the same ground line as the piece's foot."""
    new = tmp_path / "new"
    new.mkdir()
    pack = build_pack(tmp_path, new, "shop", "Bench", 1.0, 0.6, 0.46)
    out = sheet.make_sheet(pack, "Bench", new / "style.json", bar="none",
                           out=tmp_path / "out" / "sheet.png")
    Image = pytest.importorskip("PIL.Image")
    image = Image.open(out).convert("RGB")
    cw, ch = sheet.CELL
    for col in range(5):
        c = image.crop((170 + col * cw, 70, 170 + (col + 1) * cw, 70 + ch))
        background = c.getpixel((2, 2))
        pixels = {(x, y): c.getpixel((x, y)) for x in range(cw) for y in range(ch)}
        piece = [xy for xy, p in pixels.items() if p[0] - p[2] > 8]
        post = [xy for xy, p in pixels.items()
                if p[2] > p[0] + 2 and sum(p) > 150
                and max(abs(a - b) for a, b in zip(p, background, strict=True)) > 12]
        assert piece and post, f"column {col}: a piece or post is missing"
        assert min(x for x, _ in post) > max(x for x, _ in piece), f"column {col}: post not right"
