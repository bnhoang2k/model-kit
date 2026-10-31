"""The sheet: a new piece beside its bar, at the page camera and four more views.

Each row of the sheet is one piece, each column one view. Every cell shares one ortho width and
every piece is scaled to metres, so the rows compare at true relative size. A 1 m post stands
on the ground beside each piece, to the right of it as that view's camera sees it. Blender
renders the cells (`blender/sheet_render.py`, Workbench); this module plans the rows, runs
Blender and lays the cells out with Pillow. Pillow is the optional `sheet` extra, and only
`montage` imports it.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from model_kit.bars import bar_for, bars_root, read_bars
from model_kit.blender import runner
from model_kit.core.errors import ManifestError
from model_kit.core.kinds import piece_class
from model_kit.core.manifest import load_pack
from model_kit.core.units import Style, load_style

# A view is (name, turn, tilt). None takes the value from the style's page camera. Turn is the
# compass bearing in degrees (0 looks from the front, 90 from the side); tilt is the elevation
# above the horizon. These are the contact-sheet views.
View = tuple[str, float | None, float | None]
VIEWS: tuple[View, ...] = (
    ("page", None, None),
    ("front", 0.0, 8.0),
    ("side", 90.0, 8.0),
    ("back", 200.0, 25.0),
    ("top", None, 89.0),
)

CELL = (560, 460)
POST_M = 1.0
POST_GAP = 0.25  # between the piece's footprint and the post's foot, in metres
POST_FOOT_RADIUS = 0.15
MARGIN = 1.3
LABEL_COLUMN = 170
TITLE_BAND = 70
PAPER = (246, 245, 241)
TITLE_INK = (20, 20, 20)
HEAD_INK = (60, 60, 60)
LABEL_INK = (30, 30, 30)
SCRIPT = Path(__file__).resolve().parent / "blender" / "sheet_render.py"
RENDER_TIMEOUT = 600
NEEDS_PILLOW = "model-kit sheet needs the sheet extra: pip install 'model-kit[sheet]'"


@dataclass(frozen=True)
class SheetRow:
    label: str
    glb: Path
    root: str  # the piece's node name in the glb
    metres_per_unit: float
    height_m: float
    tris: int
    bounds_m: tuple[float, float, float, float]  # X0, X1, Z0, Z1 of the piece in metres (glTF)


def _piece_row(label: str, pack_path: Path, glb: Path, style_path: Path, piece: str) -> tuple[
        SheetRow, str]:
    style = load_style(style_path)
    pack = load_pack(pack_path, style)
    if piece not in pack.pieces:
        raise ManifestError(f"{pack_path}: no piece {piece!r}; it holds {list(pack.pieces)}")
    if not glb.is_file():
        raise ManifestError(f"{glb}: no .glb for {piece!r}; build the pack first")
    item = pack.pieces[piece]
    measured = item.measured
    for key in ("heights", "tris", "bounds"):
        if key not in measured:
            raise ManifestError(f"{pack_path}: pieces.{piece}.measured has no {key!r}")
    if "top" not in measured["heights"]:
        raise ManifestError(f"{pack_path}: pieces.{piece}.measured.heights has no 'top'")
    edges = ("X0", "X1", "Z0", "Z1")
    bounds = measured["bounds"]
    if not isinstance(bounds, dict) or any(k not in bounds for k in edges):
        raise ManifestError(
            f"{pack_path}: pieces.{piece}.measured.bounds must hold {list(edges)}, got {bounds!r}")
    mpu = 1.0 / style.units_per_metre
    height_m = float(measured["heights"]["top"]) * mpu
    return (SheetRow(f"{label}\n{height_m:.1f} m tall\n{int(measured['tris'])} tris", glb, piece,
                     mpu, height_m, int(measured["tris"]),
                     tuple(float(bounds[k]) * mpu for k in edges)), piece_class(item))


def _glb_for(pack_path: Path) -> Path:
    name = pack_path.name
    stem = name[: -len(".pack.json")] if name.endswith(".pack.json") else pack_path.stem
    return pack_path.with_name(stem + ".glb")


def plan_sheet(pack_path: Path, piece: str, style_path: Path, bar: str = "auto",
               bars: Path | None = None) -> list[SheetRow]:
    """The rows of a sheet: the piece, then its bar. `bar` is `auto` (the default bar for the
    piece's class), a bar slug, or `none` (the piece alone). A bar's row uses the bar's own
    style for its units."""
    pack_path, style_path = Path(pack_path), Path(style_path)
    row, cls = _piece_row(piece, pack_path, _glb_for(pack_path), style_path, piece)
    if bar == "none":
        return [row]
    all_bars = read_bars(bars_root(bars))
    if bar == "auto":
        chosen = bar_for(cls, all_bars)
    else:
        found = [b for b in all_bars if b.slug == bar]
        if not found:
            raise ManifestError(f"no bar {bar!r}; the bars are {[b.slug for b in all_bars]}")
        chosen = found[0]
    bar_piece = chosen.piece.split(":", 1)[1]
    bar_row, _ = _piece_row(f"bar {chosen.slug}\n{bar_piece}", chosen.dir / chosen.files.pack,
                            chosen.dir / chosen.files.glb, chosen.dir / chosen.files.style,
                            bar_piece)
    return [row, bar_row]


def _right_xz(turn: float) -> tuple[float, float]:
    """The camera's right direction across the ground, as glTF (x, z), for a compass turn."""
    right, _ = _camera_axes(turn, 0.0)
    return float(right[0]), float(right[2])


def _footprint(row: SheetRow, turn: float) -> tuple[tuple[float, float], tuple[float, float],
                                                      float]:
    """The piece's footprint centre (glTF x, z), the camera's right direction, and how far the
    footprint reaches to the right of its centre, all in metres."""
    x0, x1, z0, z1 = row.bounds_m
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    rx, rz = _right_xz(turn)
    half = max(abs((x - cx) * rx + (z - cz) * rz) for x in (x0, x1) for z in (z0, z1))
    return (cx, cz), (rx, rz), half


def post_spot(row: SheetRow, turn: float) -> tuple[float, float]:
    """Where the scale post stands for the camera at compass `turn`, as glTF (x, z) in metres:
    on the ground at the footprint's middle depth, just past its right edge along the camera's
    right direction. So the post is always beside the piece on screen, never in front of it and
    never behind it, in every view."""
    (cx, cz), (rx, rz), half = _footprint(row, turn)
    reach = half + POST_GAP + POST_FOOT_RADIUS
    return cx + rx * reach, cz + rz * reach


def _outline(row: SheetRow, turn: float) -> np.ndarray:
    """The corners (glTF x, y, z, metres) of the boxes round the piece and its post."""
    x0, x1, z0, z1 = row.bounds_m
    px, pz = post_spot(row, turn)
    r = POST_FOOT_RADIUS
    boxes = [(x0, x1, 0.0, row.height_m, z0, z1), (px - r, px + r, 0.0, POST_M, pz - r, pz + r)]
    return np.array([(x, y, z) for bx0, bx1, y0, y1, bz0, bz1 in boxes
                     for x in (bx0, bx1) for y in (y0, y1) for z in (bz0, bz1)])


def target_xz(row: SheetRow, turn: float) -> tuple[float, float]:
    """The ground point a row's cell centres on at compass `turn`: the middle of the piece and
    its post across the screen, at the piece's middle depth."""
    (cx, cz), (rx, rz), _ = _footprint(row, turn)
    # Across the screen the pair runs from -half (the piece's left edge) to half + POST_GAP + 2 r
    # (the post's right edge), so its middle sits half of POST_GAP + 2 r from the piece's middle.
    across = (POST_GAP + 2 * POST_FOOT_RADIUS) / 2
    return cx + rx * across, cz + rz * across


def centre_height(rows: Sequence[SheetRow]) -> float:
    """The height every cell centres on, so the ground lines of all rows agree."""
    return max([POST_M, *(r.height_m for r in rows)]) / 2


def _camera_axes(turn: float, tilt: float) -> tuple[np.ndarray, np.ndarray]:
    """The camera's right and up directions in glTF axes (turn: compass bearing, 0 looks from
    +Z; tilt: elevation above the horizon)."""
    a, e = math.radians(turn), math.radians(tilt)
    toward = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
    right = np.cross(-toward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    return right, np.cross(right, -toward)


def ortho_width(rows: Sequence[SheetRow], views: Sequence[tuple[str, float, float]],
                cell: tuple[int, int] = CELL) -> float:
    """The span one cell covers across its width, in metres: 1.3 times the largest projected
    extent of any row in any view. Every cell shares it, so rows compare at true relative size.

    The extent is the half-size, from the cell's centre, that the piece and its post need
    (piece footprint and height, the post beside it), measured along the camera's right
    direction, and along its up direction scaled to the cell's aspect.
    """
    if not rows:
        raise ManifestError("a sheet needs at least one row")
    if not views:
        raise ManifestError("a sheet needs at least one view")
    ch = centre_height(rows)
    need = 0.0
    for row in rows:
        for _name, turn, tilt in views:
            tx, tz = target_xz(row, turn)
            rel = _outline(row, turn) - np.array([tx, ch, tz])
            right, up = _camera_axes(turn, tilt)
            need = max(need, float(np.abs(rel @ right).max()) * 2,
                       float(np.abs(rel @ up).max()) * 2 * cell[0] / cell[1])
    return MARGIN * need


def resolve_views(style: Style, views: Sequence[View]) -> list[tuple[str, float, float]]:
    """Fill each view's turn and tilt from the style's page camera where the view leaves them
    open. A missing field raises, naming it."""
    out = []
    for name, turn, tilt in views:
        for field, value in (("turn_deg", turn), ("tilt_deg", tilt)):
            if value is None:
                if field not in style.page_camera:
                    raise ManifestError(
                        f"style.page_camera has no {field!r}, which view {name!r} needs")
                value = style.page_camera[field]
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ManifestError(f"style.page_camera.{field} must be a number, got {value!r}")
            if field == "turn_deg":
                turn = float(value)
            else:
                tilt = float(value)
        out.append((name, turn, tilt))
    return out


def montage(cells: list[list[Path]], row_labels: Sequence[str], col_labels: Sequence[str],
            title: str, out: Path) -> Path:
    """Lay the cell images out in a grid with a label column and a title band."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ManifestError(NEEDS_PILLOW) from exc
    if len(cells) != len(row_labels) or any(len(r) != len(col_labels) for r in cells):
        raise ManifestError(
            f"montage: {len(cells)} rows of cells, {len(row_labels)} row labels, "
            f"{len(col_labels)} column labels do not line up")
    images = [[Image.open(p).convert("RGB") for p in row] for row in cells]
    w, h = images[0][0].size
    for path, image in zip((p for row in cells for p in row),
                           (i for row in images for i in row), strict=True):
        if image.size != (w, h):
            raise ManifestError(f"montage: {path} is {image.size}, but the first cell is {(w, h)}")
    sheet = Image.new("RGB", (LABEL_COLUMN + w * len(col_labels), TITLE_BAND + h * len(cells)),
                      PAPER)
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = bold = ImageFont.load_default()
    draw.text((12, 12), title, fill=TITLE_INK, font=bold)
    for c, label in enumerate(col_labels):
        draw.text((LABEL_COLUMN + c * w + 10, TITLE_BAND - 30), label, fill=HEAD_INK, font=font)
    for r, label in enumerate(row_labels):
        for i, line in enumerate(label.split("\n")):
            draw.text((10, TITLE_BAND + r * h + 14 + i * 26), line, fill=LABEL_INK, font=font)
    for r, row in enumerate(images):
        for c, image in enumerate(row):
            sheet.paste(image, (LABEL_COLUMN + c * w, TITLE_BAND + r * h))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def make_sheet(pack_path: Path, piece: str, style_path: Path, *, bar: str = "auto",
               bars: Path | None = None, out: Path = Path("sheet.png"),
               views: Sequence[View] = VIEWS) -> Path:
    """Render the piece (and its bar) at every view and write the sheet to `out`."""
    try:
        import PIL  # noqa: F401  (fail before the render, not after it)
    except ImportError as exc:
        raise ManifestError(NEEDS_PILLOW) from exc
    style = load_style(Path(style_path))
    resolved = resolve_views(style, views)
    rows = plan_sheet(pack_path, piece, style_path, bar, bars)
    width = ortho_width(rows, resolved)
    with tempfile.TemporaryDirectory(prefix="model-kit-sheet-") as tmp:
        spec = {
            "rows": [{"glb": str(r.glb), "root": r.root, "metres_per_unit": r.metres_per_unit,
                      "height_m": r.height_m, "bounds_m": list(r.bounds_m),
                      "posts": [list(post_spot(r, v[1])) for v in resolved],
                      "targets": [list(target_xz(r, v[1])) for v in resolved]}
                     for r in rows],
            "views": [list(v) for v in resolved],
            "width": width,
            "centre_height": centre_height(rows),
            "post_m": POST_M,
            "cell": list(CELL),
            "out_dir": tmp,
        }
        result = runner.run_blender(SCRIPT, [json.dumps(spec)], timeout=RENDER_TIMEOUT)
        if not result.ok:
            raise ManifestError(f"the sheet render failed:\n{result.log[-2000:]}")
        cells = [[Path(tmp) / f"r{i}_v{j}.png" for j in range(len(resolved))]
                 for i in range(len(rows))]
        missing = [p for row in cells for p in row if not p.is_file()]
        if missing:
            raise ManifestError(f"the sheet render wrote no {missing[0].name}")
        title = (f"{piece}" + ("" if bar == "none" else f" beside its bar ({rows[1].root})")
                 + f". Post = {POST_M:g} m. All cells share one scale.")
        return montage(cells, [r.label for r in rows], [v[0] for v in resolved], title, out)
