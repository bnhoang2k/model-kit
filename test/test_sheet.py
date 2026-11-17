"""The sheet's planning and layout. Rendering is tested in test/blender/test_sheet_render.py."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from model_kit import sheet
from model_kit.core.errors import ManifestError
from model_kit.core.units import load_style
from model_kit.sheet import SheetRow, ortho_width, plan_sheet, resolve_views

FIXTURES = Path(__file__).resolve().parent / "core" / "fixtures"


def make_pack(folder: Path, name: str, piece: str, kind: str, top: float, units: float,
              width: float = 0.6, rigged: bool = False) -> Path:
    """A pack.json (from the shop fixture) holding one piece, with its .glb beside it. The
    piece is `width` units wide and deep, centred on the origin."""
    data = json.loads((FIXTURES / "shop.pack.json").read_text())
    one = data["pieces"]["Doll" if rigged else "Bench"]
    one["recipe"] = {"kind": kind, "source": "build.py"}
    half = width / 2
    one["measured"].update(
        heights={"top": top}, tris=48,
        bounds={"X0": -half, "X1": half, "Z0": -half, "Z1": half})
    data.update(pack=name, units_per_metre=units, pieces={piece: one})
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pack.json"
    path.write_text(json.dumps(data))
    (folder / f"{name}.glb").write_bytes(b"glTF")
    style = json.loads((FIXTURES / "shop.style.json").read_text())
    style["units_per_metre"] = units
    (folder / "style.json").write_text(json.dumps(style))
    return path


def make_bar(root: Path, slug: str, cls: str, piece: str, top: float, units: float,
             default: bool = True) -> Path:
    folder = root / slug
    pack = make_pack(folder, "barpack", piece, cls, top, units)
    (folder / "sheet.png").write_bytes(b"x")
    (folder / "bar.json").write_text(json.dumps({
        "slug": slug, "title": f"Bar {slug}", "class": cls, "default": default,
        "piece": f"barpack:{piece}", "reads": "r", "fits": "f", "wrong_for": "w",
        "references": [{"key": "W", "url": "https://example.com/w", "gives": "g"}],
        "files": {"style": "style.json", "pack": pack.name, "glb": "barpack.glb",
                  "scene": "scene", "sheet": "sheet.png"},
    }))
    return folder


@pytest.fixture
def world(tmp_path):
    """A new assembly piece with a style, and a bars folder with an assembly and a branching bar."""
    pack = make_pack(tmp_path / "new", "shop", "Bench", "assembly", 0.92, 0.5)
    bars = tmp_path / "bars"
    make_bar(bars, "01-stool", "assembly", "Stool", 1.0, 0.25)
    make_bar(bars, "02-redbud", "branching", "Redbud", 2.5, 0.5)
    return pack, pack.parent / "style.json", bars


def test_auto_bar_by_class(world, tmp_path):
    pack, style, bars = world
    rows = plan_sheet(pack, "Bench", style, "auto", bars)
    assert [r.root for r in rows] == ["Bench", "Stool"]
    assert rows[0].glb == pack.parent / "shop.glb"
    assert rows[1].glb == bars / "01-stool" / "barpack.glb"
    assert rows[0].height_m == pytest.approx(0.92 * 2)  # 0.5 units per metre
    assert rows[0].metres_per_unit == pytest.approx(2.0)
    assert rows[0].tris == 48
    branching = make_pack(tmp_path / "tree", "shop", "Maple", "branching", 2.0, 0.5)
    rows = plan_sheet(branching, "Maple", branching.parent / "style.json", "auto", bars)
    assert [r.root for r in rows] == ["Maple", "Redbud"]


def test_a_bar_row_uses_the_bars_own_units(world):
    pack, style, bars = world
    stool = plan_sheet(pack, "Bench", style, "01-stool", bars)[1]
    assert stool.metres_per_unit == pytest.approx(4.0)  # the bar's style is 0.25 units a metre
    assert stool.height_m == pytest.approx(4.0)
    assert "01-stool" in stool.label and "4.0 m tall" in stool.label


def test_bar_none_one_row(world):
    pack, style, _ = world
    rows = plan_sheet(pack, "Bench", style, "none", None)  # no bars folder needed
    assert [r.root for r in rows] == ["Bench"]


def test_no_bar_for_class_names_it(world, tmp_path):
    _, _, bars = world
    rigged_like = make_pack(tmp_path / "tree", "shop", "Maple", "branching", 2.0, 0.5)
    shutil.rmtree(bars / "02-redbud")
    with pytest.raises(ManifestError, match="no bar for branching pieces; pass --bar none"):
        plan_sheet(rigged_like, "Maple", rigged_like.parent / "style.json", "auto", bars)


def test_no_bar_for_a_rigged_piece_names_it(world, tmp_path):
    _, _, bars = world
    doll = make_pack(tmp_path / "doll", "shop", "Doll", "assembly", 1.0, 0.5, rigged=True)
    with pytest.raises(ManifestError, match="no bar for rigged pieces; pass --bar none"):
        plan_sheet(doll, "Doll", doll.parent / "style.json", "auto", bars)


def test_a_piece_without_bounds_raises(world):
    pack, style, _ = world
    data = json.loads(pack.read_text())
    del data["pieces"]["Bench"]["measured"]["bounds"]
    pack.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="no 'bounds'"):
        plan_sheet(pack, "Bench", style, "none", None)


def test_unknown_bar_and_unknown_piece_and_missing_glb_raise(world):
    pack, style, bars = world
    with pytest.raises(ManifestError, match="99-nope"):
        plan_sheet(pack, "Bench", style, "99-nope", bars)
    with pytest.raises(ManifestError, match="Chair"):
        plan_sheet(pack, "Chair", style, "none", None)
    (pack.parent / "shop.glb").unlink()
    with pytest.raises(ManifestError, match=r"shop\.glb"):
        plan_sheet(pack, "Bench", style, "none", None)


def row(height_m: float, width_m: float = 0.0) -> SheetRow:
    half = width_m / 2
    return SheetRow("x", Path("x.glb"), "X", 1.0, height_m, 10, (-half, half, -half, half))


FRONT = [("front", 0.0, 8.0)]
PAGE = [("page", 14.0, 52.0)]


def test_ortho_width_shared():
    views = [("front", 0.0, 8.0), ("page", 14.0, 52.0)]
    tall, small = row(5.0, 0.5), row(2.0, 0.5)
    assert ortho_width([tall, small], views) == ortho_width([small, tall], views)
    assert ortho_width([tall, small], views) > ortho_width([small], views)
    # a row of the same shape at twice the size needs twice the span
    assert ortho_width([row(10.0, 1.0)], views) == pytest.approx(2 * ortho_width([tall], views),
                                                                 rel=0.02)
    with pytest.raises(ManifestError, match="at least one row"):
        ortho_width([], views)
    with pytest.raises(ManifestError, match="at least one view"):
        ortho_width([tall], [])


def test_ortho_width_fits_a_tall_piece_in_the_cell_height():
    # Straight on, a 5 m piece needs 5 m of the cell's height: 1.3 x 5 across the height, which
    # is 1.3 x 5 x 560 / 460 across the width.
    got = ortho_width([row(5.0, 0.1)], [("front", 0.0, 0.0)])
    assert got == pytest.approx(1.3 * 5.0 * 560 / 460, rel=0.1)


def test_ortho_width_fits_a_wide_crown_in_every_view():
    # A 2.5 m crown 1.15 times wider than tall: it needs more than 1.3 x its height across the
    # page view (the span the old height-only rule gave, which clipped it).
    crown = row(2.5, 2.875)
    assert ortho_width([crown], PAGE) > 2 * 2.4 * 1.3  # half-extent 2.4 m at the page camera
    assert ortho_width([crown], PAGE) > 1.3 * 2.5 * 1.3
    # the span grows with the footprint at the top view, where only the footprint shows
    assert ortho_width([row(2.5, 6.0)], [("top", 14.0, 89.0)]) > ortho_width(
        [row(2.5, 3.0)], [("top", 14.0, 89.0)])


def _screen_box(corners: np.ndarray, turn: float, tilt: float) -> tuple[float, float, float, float]:
    """The box a set of glTF points covers on the camera's screen: x0, x1, y0, y1 (metres)."""
    right, up = sheet._camera_axes(turn, tilt)
    across, rising = corners @ right, corners @ up
    return across.min(), across.max(), rising.min(), rising.max()


# Footprints that stress the placement: a square, a wide crown, a deep thin piece, and one whose
# footprint sits well off the origin.
SHAPES = [(-0.3, 0.3, -0.3, 0.3), (-2.9, 2.9, -2.5, 2.5), (-0.2, 0.2, -1.5, 1.5),
          (1.0, 3.0, -4.0, -2.0)]


@pytest.mark.parametrize("bounds", SHAPES)
def test_the_post_stands_beside_the_piece_in_every_view(bounds):
    style = load_style(FIXTURES / "shop.style.json")
    piece = SheetRow("x", Path("x.glb"), "X", 1.0, 2.5, 10, bounds)
    for name, turn, tilt in resolve_views(style, sheet.VIEWS):
        px, pz = sheet.post_spot(piece, turn)
        r = sheet.POST_FOOT_RADIUS
        post = np.array([(x, y, z) for x in (px - r, px + r) for y in (0.0, sheet.POST_M)
                         for z in (pz - r, pz + r)])
        x0, x1, z0, z1 = bounds
        body = np.array([(x, y, z) for x in (x0, x1) for y in (0.0, piece.height_m)
                         for z in (z0, z1)])
        p, b = _screen_box(post, turn, tilt), _screen_box(body, turn, tilt)
        apart_across = p[0] > b[1] or p[1] < b[0]
        apart_rising = p[2] > b[3] or p[3] < b[2]
        assert apart_across or apart_rising, f"the post overlaps the piece in view {name}"
        # beside it: to the right, not in front of or behind it
        assert p[0] > b[1], f"the post is not to the piece's right in view {name}"
        # at the piece's middle depth: no offset along the camera's line of sight
        a = np.radians(turn)
        depth = (px - (x0 + x1) / 2) * np.sin(a) + (pz - (z0 + z1) / 2) * np.cos(a)
        assert depth == pytest.approx(0.0, abs=1e-9)
        # on the ground, and inside the shared ortho width
        outline = sheet._outline(piece, turn)
        assert outline[:, 1].min() == 0.0
        tx, tz = sheet.target_xz(piece, turn)
        right, _ = sheet._camera_axes(turn, tilt)
        span = np.abs((outline - np.array([tx, 0.0, tz])) @ right).max() * 2
        assert span * sheet.MARGIN <= sheet.ortho_width([piece], [(name, turn, tilt)]) + 1e-9


def test_views_read_the_style_and_a_missing_field_raises(tmp_path):
    style = load_style(FIXTURES / "shop.style.json")
    got = {name: (turn, tilt) for name, turn, tilt in resolve_views(style, sheet.VIEWS)}
    assert got == {"page": (14.0, 52.0), "front": (0.0, 8.0), "side": (90.0, 8.0),
                   "back": (200.0, 25.0), "top": (14.0, 89.0)}
    bare = json.loads((FIXTURES / "shop.style.json").read_text())
    del bare["page_camera"]["tilt_deg"]
    (tmp_path / "bare.json").write_text(json.dumps(bare))
    with pytest.raises(ManifestError, match="tilt_deg"):
        resolve_views(load_style(tmp_path / "bare.json"), sheet.VIEWS)
    del bare["page_camera"]["turn_deg"]
    (tmp_path / "bare.json").write_text(json.dumps(bare))
    with pytest.raises(ManifestError, match="turn_deg"):
        resolve_views(load_style(tmp_path / "bare.json"), [("top", None, 89.0)])


def test_montage_layout(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    cells = []
    for r in range(2):
        row_paths = []
        for c in range(2):
            path = tmp_path / f"c{r}{c}.png"
            Image.new("RGB", (10, 8), (200 + r * 20, 10, 10 + c * 40)).save(path)
            row_paths.append(path)
        cells.append(row_paths)
    out = sheet.montage(cells, ["a", "b"], ["x", "y"], "Title", tmp_path / "out" / "sheet.png")
    image = Image.open(out)
    assert image.size == (170 + 20, 70 + 16)
    assert image.getpixel((170 + 5, 70 + 4)) == (200, 10, 10)  # cell (0, 0)
    assert image.getpixel((170 + 15, 70 + 12)) == (220, 10, 50)  # cell (1, 1)
    assert image.getpixel((185, 60)) == (246, 245, 241)  # the paper between title and cells


def test_montage_cells_must_share_a_size(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    Image.new("RGB", (10, 8)).save(tmp_path / "a.png")
    Image.new("RGB", (10, 9)).save(tmp_path / "b.png")
    with pytest.raises(ManifestError, match=r"b\.png.*\(10, 9\)"):
        sheet.montage([[tmp_path / "a.png", tmp_path / "b.png"]], ["r"], ["x", "y"], "t",
                      tmp_path / "o.png")


def test_montage_without_pillow_says_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "PIL.Image", raising=False)
    with pytest.raises(ManifestError, match=r"pip install 'model-kit\[sheet\]'"):
        sheet.montage([[tmp_path / "a.png"]], ["a"], ["x"], "t", tmp_path / "o.png")


def test_eevee_engine_takes_whichever_name_this_blender_has():
    assert sheet.eevee_engine(["BLENDER_WORKBENCH", "BLENDER_EEVEE", "CYCLES"]) == "BLENDER_EEVEE"
    assert sheet.eevee_engine(["BLENDER_EEVEE_NEXT", "CYCLES"]) == "BLENDER_EEVEE_NEXT"
    with pytest.raises(ManifestError, match="no EEVEE render engine"):
        sheet.eevee_engine(["BLENDER_WORKBENCH", "CYCLES"])
