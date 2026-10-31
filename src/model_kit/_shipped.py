"""Which files under bars/ ship in the wheel. Standard library only.

The build hook (hatch_build.py) loads this file by path, and the hygiene test imports it, so the
two read one rule: what ships is exactly what the hygiene scan reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

TEXT_SUFFIXES = (".json", ".md", ".py")  # the hygiene scan reads every one of these
SHIPPED_SUFFIXES = (*TEXT_SUFFIXES, ".glb", ".png")  # .glb: scanned by its JSON chunk; .png: pixels
SCENE_DIR = "scene"  # a bar's Blender scene folder: never ships
PRIVATE_BAR_FILE = "bar.private.json"  # never ships; load_bar refuses a bar folder that holds one


def ships(rel: Path) -> bool:
    """Whether the file at this path (relative to bars/) goes into the wheel."""
    parts = rel.parts
    if SCENE_DIR in parts[:-1] or any(p.startswith(".") or p == "__pycache__" for p in parts):
        return False
    if ".private." in rel.name:
        return False
    return rel.suffix in SHIPPED_SUFFIXES


def shipped_files(root: Path) -> Iterator[Path]:
    """Every file under root that ships, as a path relative to root, sorted."""
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if path.is_file() and ships(path.relative_to(root)):
            yield path.relative_to(root)
