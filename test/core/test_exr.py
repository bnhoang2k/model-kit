"""The EXR reader (`core.exr`), on small files built here byte by byte.

`encode` writes the one kind of file the bake writes: single-part scanline, half-float channels
B G R (the name order), ZIP compression in blocks of 16 rows. The reader's agreement with a file
Blender wrote is in `test/blender/test_bake_exr.py`.
"""

import struct
import zlib

import numpy as np
import pytest

from model_kit.core import exr
from model_kit.core.errors import ManifestError


def attr(name: str, kind: str, value: bytes) -> bytes:
    return name.encode() + b"\0" + kind.encode() + b"\0" + struct.pack("<I", len(value)) + value


def channels(kind: int = 1, sampling: int = 1) -> bytes:
    tail = struct.pack("<iB3xii", kind, 0, sampling, sampling)
    return b"".join(n.encode() + b"\0" + tail for n in "BGR") + b"\0"


def zip_block(raw: bytes) -> bytes:
    """Split a block's bytes into evens then odds, delta-code them (+128) and deflate."""
    data = np.frombuffer(raw, dtype=np.uint8)
    split = np.concatenate([data[0::2], data[1::2]]).astype(np.int64)
    delta = np.empty_like(split)
    delta[0] = split[0]
    delta[1:] = (split[1:] - split[:-1] + 128) % 256
    return zlib.compress(delta.astype(np.uint8).tobytes())


def encode(image: np.ndarray, *, compression: int = 3, flags: int = 0, kind: int = 1,
           store_raw: bool = False) -> bytes:
    """`image` (H, W, 3) as R, G, B float16 -> an EXR file's bytes."""
    height, width, _ = image.shape
    header = (attr("channels", "chlist", channels(kind))
              + attr("compression", "compression", bytes([compression]))
              + attr("dataWindow", "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1))
              + attr("displayWindow", "box2i", struct.pack("<4i", 0, 0, width - 1, height - 1))
              + attr("lineOrder", "lineOrder", b"\0") + b"\0")
    blocks = []
    for y in range(0, height, 16):
        rows = image[y:y + 16].astype("<f2")
        # per scanline: the B row, then G, then R
        raw = b"".join(rows[r, :, c].tobytes() for r in range(len(rows)) for c in (2, 1, 0))
        body = raw if store_raw else zip_block(raw)
        blocks.append(struct.pack("<iI", y, len(body)) + body)
    start = 8 + len(header) + 8 * len(blocks)
    offsets, at = [], start
    for block in blocks:
        offsets.append(at)
        at += len(block)
    return (b"\x76\x2f\x31\x01" + struct.pack("<I", 2 | flags) + header
            + struct.pack(f"<{len(blocks)}Q", *offsets) + b"".join(blocks))


@pytest.fixture
def picture():
    rng = np.random.default_rng(3)
    return (rng.random((20, 5, 3)) * 8).astype(np.float16)  # two blocks: 16 rows, then 4


def test_reads_what_it_is_given_bit_for_bit(tmp_path, picture):
    path = tmp_path / "a.exr"
    path.write_bytes(encode(picture))
    got = exr.read_exr(path)
    assert got.dtype == np.float32 and got.shape == (20, 5, 3)
    assert np.array_equal(got, picture.astype(np.float32))


def test_reads_a_block_stored_raw(tmp_path, picture):
    path = tmp_path / "raw.exr"
    path.write_bytes(encode(picture, store_raw=True))
    assert np.array_equal(exr.read_exr(path), picture.astype(np.float32))


@pytest.mark.parametrize("make, why", [
    (lambda p: encode(p, compression=0), "not ZIP"),
    (lambda p: encode(p, flags=0x200), "tiled"),
    (lambda p: encode(p, kind=2), "not half-float"),
    (lambda p: b"nope" + b"\0" * 20, "not an OpenEXR"),
])
def test_what_it_does_not_read_raises_naming_the_file(tmp_path, picture, make, why):
    path = tmp_path / "other.exr"
    path.write_bytes(make(picture))
    with pytest.raises(ManifestError, match="other.exr"):
        exr.read_exr(path)


def test_a_cut_short_file_raises(tmp_path, picture):
    path = tmp_path / "cut.exr"
    path.write_bytes(encode(picture)[:-30])
    with pytest.raises(ManifestError, match="cut.exr"):
        exr.read_exr(path)
