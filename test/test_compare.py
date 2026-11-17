"""The comparison's plain-Python parts: the lights file, the PNG reader and the CLI flags."""

import json
import struct
import zlib

import numpy as np
import pytest

from model_kit import cli, compare
from model_kit.core.errors import ManifestError


def test_lights_file_reads_layer_values(tmp_path):
    path = tmp_path / "lights.json"
    path.write_text(json.dumps({"lamp": {"color": [1, 0.5, 0], "strength": 2}, "sun": {}}))
    assert compare.read_values(path, ("lamp", "sun")) == {
        "lamp": {"color": [1.0, 0.5, 0.0], "strength": 2.0},
        "sun": {"color": [1.0, 1.0, 1.0], "strength": 1.0}}
    assert compare.read_values(None, ("lamp",)) == {}


@pytest.mark.parametrize("text, words", [
    ('{"moon": {"strength": 1}}', "no layer"),
    ('{"lamp": {"color": [1, 1], "strength": 1}}', "non-negative"),
    ('{"lamp": {"strength": -1}}', "non-negative"),
    ("[1]", "object"),
    ("{", "cannot read"),
])
def test_lights_file_refuses_a_bad_file(tmp_path, text, words):
    path = tmp_path / "lights.json"
    path.write_text(text)
    with pytest.raises(ManifestError, match=words):
        compare.read_values(path, ("lamp",))


def png(rows: np.ndarray, kinds: list[int]) -> bytes:
    """An 8-bit RGB or RGBA PNG of `rows` (h, w, 3 or 4 uint8), each row written with the given
    filter."""
    height, width, n = rows.shape
    raw, prev = b"", np.zeros(width * n, dtype=np.int64)
    for line, kind in zip(rows.reshape(height, -1).astype(np.int64), kinds, strict=True):
        left = np.concatenate([np.zeros(n, dtype=np.int64), line[:-n]])
        corner = np.concatenate([np.zeros(n, dtype=np.int64), prev[:-n]])
        guess = {0: 0 * line, 1: left, 2: prev, 3: (left + prev) // 2}.get(kind)
        if kind == 4:
            p = left + prev - corner
            pa, pb, pc = abs(p - left), abs(p - prev), abs(p - corner)
            guess = np.where((pa <= pb) & (pa <= pc), left, np.where(pb <= pc, prev, corner))
        raw += bytes([kind]) + ((line - guess) & 255).astype(np.uint8).tobytes()
        prev = line

    def chunk(name, body):
        return (struct.pack(">I", len(body)) + name + body
                + struct.pack(">I", zlib.crc32(name + body)))
    head = struct.pack(">IIBBBBB", width, height, 8, 2 if n == 3 else 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", head)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.mark.parametrize("channels", [3, 4])
def test_png_reader_undoes_every_filter(channels):
    rows = np.random.default_rng(1).integers(0, 256, (5, 4, channels), dtype=np.uint8)
    got = compare.read_png(png(rows, [0, 1, 2, 3, 4]))
    assert got.shape == (5, 4, 4)
    assert np.allclose(got[:, :, :channels], rows / 255.0)
    if channels == 3:
        assert (got[:, :, 3] == 1).all()  # an RGB picture covers every pixel


def picture(rgb, alpha) -> np.ndarray:
    """A (h, w, 4) picture of one colour per pixel and an alpha mask (h, w), 0 to 1."""
    return np.concatenate([np.asarray(rgb, float), np.asarray(alpha, float)[..., None]], axis=2)


def test_difference_measures_only_pixels_covered_in_both():
    """The background differs wildly (white sky against nothing) and the scene matches to 0.1:
    the number is the scene's 0.1, not the background's. A half-covered edge pixel is dropped."""
    h, w = 4, 5
    alpha_a = np.zeros((h, w))
    alpha_a[1:3, 1:4] = 1.0  # the scene: 6 pixels
    alpha_b = alpha_a.copy()
    alpha_b[1, 4] = 1.0  # one pixel only the browser covers
    alpha_b[2, 3] = 0.5  # an anti-aliased edge: not covered fully, so dropped
    rgb_a = np.ones((h, w, 3))  # Blender: white sky everywhere ...
    rgb_a[1:3, 1:4] = 0.5  # ... and a grey scene
    rgb_b = np.zeros((h, w, 3))  # the browser: black where nothing is drawn ...
    rgb_b[1:3, 1:4] = 0.4  # ... and the scene 0.1 darker
    rgb_b[2, 3] = 0.0  # the edge pixel is far off, but dropped
    got = compare.masked_difference(picture(rgb_a, alpha_a), picture(rgb_b, alpha_b))
    assert got.mean == pytest.approx(0.1)
    assert got.both == pytest.approx(5 / 20)
    assert got.only_one == pytest.approx(2 / 20)  # (1, 4) in the browser, (2, 3) in Blender


def test_difference_with_nothing_covered_in_both():
    alpha = np.zeros((2, 2))
    alpha[0, 0] = 1.0
    got = compare.masked_difference(picture(np.zeros((2, 2, 3)), alpha),
                                    picture(np.zeros((2, 2, 3)), np.zeros((2, 2))))
    assert got.mean is None and got.both == 0 and got.only_one == pytest.approx(0.25)


def test_difference_refuses_pictures_of_different_sizes():
    with pytest.raises(ManifestError, match="differ in size"):
        compare.masked_difference(np.zeros((2, 2, 4)), np.zeros((2, 3, 4)))


def test_lights_flags_need_compare(tmp_path):
    assert cli.main(["bake", str(tmp_path), "--stage", "s.json", "--lights", "l.json"]) == 2
