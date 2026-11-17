"""Scenes of a piece with a part that rests on several supports, for the floaters and bearing
tests. A helper module.

Two pieces, both `kit:Thing`, built with `piece_scenes.make_ctx`:

- a EUR pallet. EPAL gives 1200 x 800 x 144 mm and 22 mm boards (the EPAL page on the EUR 1
  pallet). Every other size is a fixture choice: board widths and gaps, the 0.145 m stringers and
  blocks, the 0.100 m block depth. Five top boards rest on three stringers; each stringer rests
  on three blocks; each block stands on a bottom board.
- a bench: a seat on two trestles, with a stretcher between the trestles that touches neither the
  seat nor anything above y 0.04 m. All its sizes are fixture choices.

Sizes are in metres, y up. Both pieces are seen from `NEAREST_M`, 3 m: a pallet passed in an aisle
or a bench seen across a room (a fixture choice: a seat-height piece is looked at from a few steps away).
`SUPPORT_TOUCH` is the contact length that view gives.
"""

import dataclasses

import numpy as np

from model_kit.core import sight
from model_kit.core.checks import pieces
from model_kit.core.manifest import View

from ..shapes import box
from .piece_scenes import make_ctx

NEAREST_M = 3.0
WHY = "seen across a warehouse aisle or a room"


def shortened(tris, by):
    """The box with this footprint and bottom and its top `by` lower."""
    pts = tris.reshape(-1, 3)
    return box(pts.min(axis=0), pts.max(axis=0) - np.array([0.0, by, 0.0]))


def raised(tris, by):
    """The box with this footprint and bottom and its top `by` higher."""
    pts = tris.reshape(-1, 3)
    return box(pts.min(axis=0), pts.max(axis=0) + np.array([0.0, by, 0.0]))


def _with_view(ctx):
    piece = ctx.packs["kit"].pieces["Thing"]
    view = View(NEAREST_M, WHY, (), "")
    return pieces.with_piece(ctx, "kit", "Thing", dataclasses.replace(piece, view=view))


def _specs(defaults, specs):
    merged = {name: dict(spec) for name, spec in defaults.items()}
    for name, spec in (specs or {}).items():
        merged[name] = {**merged.get(name, {}), **spec}
    return merged


# --- the pallet -------------------------------------------------------------------------------

_WIDTHS = (0.145, 0.100, 0.145, 0.100, 0.145)
_STRINGER_X = (-0.5275, 0.0, 0.5275)
_BLOCK_Z = (-0.35, 0.0, 0.35)


def pallet_parts() -> dict[str, np.ndarray]:
    parts = {}
    gap = (0.8 - sum(_WIDTHS)) / (len(_WIDTHS) - 1)
    z = -0.4
    for k, width in enumerate(_WIDTHS, 1):
        parts[f"top_{k}"] = box((-0.6, 0.122, z), (0.6, 0.144, z + width))
        z += width + gap
    for k, x in enumerate(_STRINGER_X, 1):
        parts[f"stringer_{k}"] = box((x - 0.0725, 0.100, -0.4), (x + 0.0725, 0.122, 0.4))
    for k, x in enumerate(_STRINGER_X, 1):
        for j, cz in enumerate(_BLOCK_Z, 1):
            parts[f"block_{k}{j}"] = box((x - 0.0725, 0.022, cz - 0.05),
                                         (x + 0.0725, 0.100, cz + 0.05))
    for j, cz in enumerate(_BLOCK_Z, 1):
        parts[f"bottom_{j}"] = box((-0.6, 0.0, cz - 0.05), (0.6, 0.022, cz + 0.05))
    return parts


def _pallet_specs():
    specs = {f"top_{k}": {"rests_on": [f"stringer_{s}" for s in (1, 2, 3)]}
             for k in range(1, 6)}
    specs |= {f"stringer_{k}": {"rests_on": [f"block_{k}{j}" for j in (1, 2, 3)]}
              for k in (1, 2, 3)}
    specs |= {f"block_{k}{j}": {"on": f"bottom_{j}"} for k in (1, 2, 3) for j in (1, 2, 3)}
    return specs


def pallet_ctx(parts=None, specs=None, hosts=None):
    """The pallet piece `kit:Thing`; `parts` and `specs` merge over the defaults."""
    meshes = {**pallet_parts(), **(parts or {})}
    return _with_view(make_ctx(meshes, _specs(_pallet_specs(), specs), hosts=hosts))


# --- the bench --------------------------------------------------------------------------------


def bench_parts() -> dict[str, np.ndarray]:
    return {
        "seat": box((-0.9, 0.42, -0.15), (0.9, 0.46, 0.15)),
        "trestle_l": box((-0.74, 0.0, -0.15), (-0.66, 0.42, 0.15)),
        "trestle_r": box((0.66, 0.0, -0.15), (0.74, 0.42, 0.15)),
        "stretcher": box((-0.66, 0.0, -0.03), (0.66, 0.04, 0.03)),
    }


def bench_ctx(parts=None, specs=None, hosts=None):
    """The bench piece `kit:Thing`; `parts` and `specs` merge over the defaults."""
    meshes = {**bench_parts(), **(parts or {})}
    defaults = {"seat": {"rests_on": ["trestle_l", "trestle_r"]}}
    return _with_view(make_ctx(meshes, _specs(defaults, specs), hosts=hosts))


BRIDGES = {"pallet": (pallet_ctx, pallet_parts), "bench": (bench_ctx, bench_parts)}

SUPPORT_TOUCH = sight.contact_length(bench_ctx().packs["kit"].pieces["Thing"], 1.0)
