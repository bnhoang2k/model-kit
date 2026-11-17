"""A small OpenEXR reader for the pictures the bake writes, with numpy and zlib only.

It reads one kind of file: a single-part scanline image, half-float channels, ZIP compression
(16 scanlines a block). That is what `model-kit bake` writes. Any other file (tiled, deep,
multi-part, another compression or pixel type, a sub-sampled channel) raises a `ManifestError`
naming the file, so a picture the reader does not know is never read as garbage.

A ZIP block holds, for each scanline, each channel's row one after another (channels in name
order). The block's bytes are split into two halves (the even bytes, then the odd bytes), then
delta-coded: each byte is stored as its difference from the one before, plus 128. `read_exr`
undoes both and returns the image as (height, width, 3) float32 in R, G, B order, top row first.
A block that did not shrink under compression is stored raw; that is read too.

Format: https://openexr.com/en/latest/OpenEXRFileLayout.html
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np

from model_kit.core.errors import ManifestError

_MAGIC = b"\x76\x2f\x31\x01"
_ZIP = 3
_HALF = 1
_BLOCK_ROWS = 16
_NOT_PLAIN = 0x200 | 0x800 | 0x1000  # version flags: tiled, deep data, multi-part


def _header(path: Path, data: bytes) -> tuple[dict[str, bytes], int]:
    """The header's attributes by name, and where the offset table starts."""
    if data[:4] != _MAGIC or len(data) < 9:
        raise ManifestError(f"{path}: not an OpenEXR file")
    flags = int.from_bytes(data[4:8], "little")
    if flags & _NOT_PLAIN:
        raise ManifestError(f"{path}: a tiled, deep or multi-part EXR; the reader takes a "
                            "single-part scanline file")
    at, attrs = 8, {}
    try:
        while data[at] != 0:
            end = data.index(b"\0", at)
            name = data[at:end].decode()
            at = data.index(b"\0", end + 1) + 1
            size = int.from_bytes(data[at:at + 4], "little")
            attrs[name] = data[at + 4:at + 4 + size]
            at += 4 + size
    except (ValueError, IndexError):
        raise ManifestError(f"{path}: the EXR header is cut short") from None
    return attrs, at + 1


def _channels(path: Path, raw: bytes) -> list[str]:
    names, i = [], 0
    try:
        while raw[i] != 0:
            end = raw.index(b"\0", i)
            names.append(raw[i:end].decode())
            kind = int.from_bytes(raw[end + 1:end + 5], "little")
            x_sampling = int.from_bytes(raw[end + 9:end + 13], "little")
            y_sampling = int.from_bytes(raw[end + 13:end + 17], "little")
            if kind != _HALF:
                raise ManifestError(f"{path}: channel {names[-1]!r} is not half-float; the "
                                    "reader takes half-float only")
            if (x_sampling, y_sampling) != (1, 1):
                raise ManifestError(f"{path}: channel {names[-1]!r} is sub-sampled")
            i = end + 17
    except (ValueError, IndexError):
        raise ManifestError(f"{path}: the EXR channel list is cut short") from None
    return names


def _undo_zip(packed: bytes) -> np.ndarray:
    """One block's bytes with the delta coding and the even/odd split undone."""
    raw = np.frombuffer(zlib.decompress(packed), dtype=np.uint8)
    deltas = raw.astype(np.int64) - 128
    deltas[0] = raw[0]
    flat = (np.cumsum(deltas) % 256).astype(np.uint8)
    half = (len(flat) + 1) // 2
    joined = np.empty(len(flat), dtype=np.uint8)
    joined[0::2], joined[1::2] = flat[:half], flat[half:]
    return joined


def read_exr(path) -> np.ndarray:
    """The picture in `path` as (height, width, 3) float32, R, G, B, top row first."""
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"{path}: cannot read the EXR: {exc}") from exc
    attrs, at = _header(path, data)
    for need in ("channels", "compression", "dataWindow"):
        if need not in attrs:
            raise ManifestError(f"{path}: the EXR header has no {need}")
    if attrs["compression"][0] != _ZIP:
        raise ManifestError(f"{path}: compression {attrs['compression'][0]} is not ZIP (3); "
                            "the reader takes ZIP only")
    if "tiles" in attrs:
        raise ManifestError(f"{path}: a tiled EXR; the reader takes scanlines only")
    names = _channels(path, attrs["channels"])
    missing = [c for c in "RGB" if c not in names]
    if missing:
        raise ManifestError(f"{path}: no channel {missing} (it has {names})")
    box = np.frombuffer(attrs["dataWindow"], dtype="<i4")
    width, height = int(box[2] - box[0] + 1), int(box[3] - box[1] + 1)
    blocks = -(-height // _BLOCK_ROWS)
    table = np.frombuffer(data[at:at + 8 * blocks], dtype="<u8")
    if len(table) != blocks:
        raise ManifestError(f"{path}: the EXR offset table is cut short")
    out = np.zeros((height, width, len(names)), dtype=np.float32)
    for off in table:
        off = int(off)
        y = int.from_bytes(data[off:off + 4], "little", signed=True) - int(box[1])
        length = int.from_bytes(data[off + 4:off + 8], "little")
        body = data[off + 8:off + 8 + length]
        rows = min(_BLOCK_ROWS, height - y)
        expect = rows * len(names) * width * 2
        if len(body) != length or not 0 <= y < height:
            raise ManifestError(f"{path}: an EXR block is cut short")
        try:
            joined = (np.frombuffer(body, dtype=np.uint8)
                      if length == expect else _undo_zip(body))
        except zlib.error as exc:
            raise ManifestError(f"{path}: an EXR block does not inflate: {exc}") from exc
        if len(joined) != expect:
            raise ManifestError(f"{path}: an EXR block holds {len(joined)} bytes, not {expect}")
        pixels = joined.view("<f2").astype(np.float32).reshape(rows, len(names), width)
        out[y:y + rows] = pixels.transpose(0, 2, 1)
    return out[:, :, [names.index(c) for c in "RGB"]]
