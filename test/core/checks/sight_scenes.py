"""Scenes with a declared viewing distance, for the tests of the seen floor and of contact. A
helper module."""

import dataclasses

from model_kit.core import sight
from model_kit.core.checks import pieces
from model_kit.core.manifest import View


def view_json(floor, units_per_metre=1.0):
    """The `view` entry of a pack file for a piece whose seen floor (and so its contact length) is
    `floor`, in the pack's units: the distance at which one arcminute covers `floor`."""
    return {"nearest_m": sight.seen_distance(floor, units_per_metre), "why": "a test distance"}


def viewed(ctx, floor, hidden=()):
    """The scene with every piece declared from the distance at which one arcminute covers
    `floor` (in the pack's units), with these sides never seen."""
    for pack_name, pack in ctx.packs.items():
        distance = sight.seen_distance(floor, pack.units_per_metre)
        for name, piece in pack.pieces.items():
            view = View(distance, "a test distance", tuple(hidden), "a test side" if hidden else "")
            ctx = pieces.with_piece(ctx, pack_name, name, dataclasses.replace(piece, view=view))
    return ctx


def unviewed(ctx):
    """The scene with no piece declaring a view: no contact length anywhere."""
    for pack_name, pack in ctx.packs.items():
        for name, piece in pack.pieces.items():
            ctx = pieces.with_piece(ctx, pack_name, name, dataclasses.replace(piece, view=None))
    return ctx
