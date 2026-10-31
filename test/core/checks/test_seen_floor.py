"""The shared seen floor: the length one arcminute covers at a piece's nearest viewing distance.

The three visible-detail checks (detail, markings-flush, inset-seat) read it through
`core.sight.seen_floor`; near-crease rule B, floaters and clip-hygiene use it to drop a finding too
small to see. The tests of those drops sit beside each check's own tests.
"""

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import checks, units
from model_kit.core.checks import pieces
from model_kit.core.checks.bearing import Bearing
from model_kit.core.checks.clearance import Clearance
from model_kit.core.checks.connectors import Connectors
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.frames import Frames
from model_kit.core.checks.inset_seat import InsetSeat
from model_kit.core.checks.mirror_pair import MirrorPair
from model_kit.core.checks.receiver import Receiver
from model_kit.core.checks.trace import Trace
from model_kit.core.manifest import View
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import PieceGeom
from model_kit.core.sight import Sight, contact_length, seen_distance, seen_floor
from model_kit.core.tolerances import EYE_RESOLUTION_ARCMIN

from ..shapes import box
from . import (
    assembly_scenes,
    bearing_scenes,
    manifest_scenes,
    piece_scenes,
    test_clearance,
    test_connectors,
    test_frames,
    test_inset_seat,
    test_mirror_pair,
    test_receiver,
    test_trace,
)
from .piece_scenes import make_ctx
from .sight_scenes import unviewed

ONE_ARCMIN = math.tan(math.radians(EYE_RESOLUTION_ARCMIN / 60.0))


def piece_at(nearest_m):
    ctx = make_ctx({"slab": box((0, 0, 0), (1, 1, 1))})
    piece = ctx.packs["kit"].pieces["Thing"]
    return dataclasses.replace(piece, view=None if nearest_m is None
                               else View(nearest_m, "a test distance"))


def test_the_floor_is_one_arcminute_at_the_nearest_distance_in_pack_units():
    assert seen_floor(piece_at(1.0), 1.0) == pytest.approx(ONE_ARCMIN, rel=1e-12)
    assert seen_floor(piece_at(2.0), 0.5) == pytest.approx(ONE_ARCMIN, rel=1e-12)
    assert 0.29e-3 < seen_floor(piece_at(1.0), 1.0) < 0.2910e-3


def test_a_piece_with_no_view_has_no_floor():
    assert seen_floor(piece_at(None), 1.0) is None


def test_the_distance_is_the_inverse_of_the_floor():
    length = seen_floor(piece_at(7.5), 0.747)
    assert seen_distance(length, 0.747) == pytest.approx(7.5, rel=1e-12)


def test_the_coverage_line_names_the_count_the_distance_and_the_piece():
    piece = piece_at(0.5)
    sight = pieces.sight_of(piece, 1.0)
    sight.under = {"a", "b", "c"}
    assert pieces.under_floor_line("some-check", sight, piece, "kit:Thing") == (
        "some-check: 3 under 1 arcmin at 0.5 m: kit:Thing")
    assert pieces.hidden_line("some-check", sight, piece, "kit:Thing", "floats") == []
    sight.unseen = {"d"}
    assert pieces.hidden_line("some-check", sight, piece, "kit:Thing", "floats") == [
        "some-check: 1 floats on hidden sides (): kit:Thing"]


# --- the contact length: one length per piece, read by every contact rule -------------------------


def test_sight_is_frozen_and_the_contact_length_is_the_seen_floor():
    seen = Sight(gap=0.002, crease_deg=35.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        seen.gap = 0.1  # type: ignore[misc]
    piece = piece_at(1.5)
    assert contact_length(piece, 0.747) == seen_floor(piece, 0.747)
    assert contact_length(piece_at(None), 0.747) is None


def _post_lifted():
    return piece_scenes.make_ctx(piece_scenes.post_on_base(lift=0.1), {"post": {"on": "base"}})


def _trace_drifted():
    ctx = test_trace.make_ctx()
    return test_trace.retarget(ctx, baked=tuple(test_trace.baked_of(ctx)
                                                + [0.0, 10 * test_trace.TOL, 0.0]))


# (the check, a scene on which it finds a fault when the pieces declare a view)
CONSUMERS = {
    "floaters": (Floaters, _post_lifted),
    "bearing": (Bearing, lambda: bearing_scenes.stack(x=0.35)),
    "clearance": (Clearance, lambda: test_clearance.stacked(10 * test_clearance.TOUCH)),
    "inset-seat": (InsetSeat, lambda: test_inset_seat.scene(
        {"mark": test_inset_seat.patch(y=test_inset_seat.HOST_TOP + 0.05)},
        test_inset_seat.MARK_SPEC)),
    "frames": (Frames, lambda: test_frames.lifted(manifest_scenes.make_ctx(), "hold_R",
                                                  10 * manifest_scenes.LIMIT)),
    "mirror-pair": (MirrorPair, lambda: test_mirror_pair.moved_left_hold(
        manifest_scenes.make_ctx(), 3 * manifest_scenes.LIMIT)),
    "receiver": (Receiver, lambda: test_receiver.lamp_lifted(10 * assembly_scenes.END)),
    "connectors": (Connectors, lambda: test_connectors.end_moved(
        assembly_scenes.make_ctx(), 0, 10 * assembly_scenes.END)),
    "trace": (Trace, _trace_drifted),
}


@pytest.mark.parametrize("name", sorted(CONSUMERS))
def test_contact_rules_unmeasured_without_view(name):
    """A piece with no `view` has no contact length: every contact rule reads Unmeasured for it,
    judges nothing on it, plants nothing for it, and the run is partial, not passed."""
    check, build = CONSUMERS[name]
    seen = build()
    assert check().run(seen), f"{name}: the scene must fail while its pieces declare a view"
    bare = unviewed(seen)
    assert check().run(bare) == []
    lines = [x for x in check().coverage(bare) if isinstance(x, Unmeasured)]
    assert lines, f"{name}: no Unmeasured line"
    assert run_gate([check()], bare, None).status == "partial"


def test_no_check_reads_a_pack_base():
    """No check reads a pack-wide thickness base: not `pack_thickness_base` (a removed
    name, which must stay gone), not `thickness_base` outside the branching checks (the
    one tube-size dial that still uses it), and not as a `.base` of a `PieceView`. (`clip_hygiene`'s
    `c.base` is a clip's base bone and branching's `sk.base` a skeleton's root: other things.)"""
    root = Path(checks.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        branching = "branching" in path.relative_to(root).parts
        for node in ast.walk(tree):
            for banned in ("pack_thickness_base", *(() if branching else ("thickness_base",))):
                if isinstance(node, ast.Name) and node.id == banned:
                    found.append(f"{path.name}:{node.lineno} reads {banned}")
                if isinstance(node, ast.Attribute) and node.attr == banned:
                    found.append(f"{path.name}:{node.lineno} reads units.{banned}")
                if isinstance(node, ast.alias) and node.name == banned:
                    found.append(f"{path.name}:{node.lineno} imports {banned}")
            if (isinstance(node, ast.Attribute) and node.attr == "base"
                    and isinstance(node.value, ast.Name) and node.value.id in {"view", "v"}
                    and path.parent == root and path.name != "clip_hygiene.py"):
                found.append(f"{path.name}:{node.lineno} reads {node.value.id}.base")
    assert found == []
    assert "base" not in {f.name for f in dataclasses.fields(pieces.PieceView)}
    assert not hasattr(units, "pack_thickness_base")


def _many_boxes(inside: bool):
    """A part of a hundred boxes, 99 of them far out of the big box's reach and one inside it (or,
    with `inside` off, outside it). The one is the third box: its twelve triangles fall between
    the 64 evenly spread triangles a sampling probe would take, so only a probe of every island finds it."""
    boxes = [box((5 + 0.5 * i, 5, 5), (5.2 + 0.5 * i, 5.2, 5.2)) for i in range(100)]
    boxes[2] = (box((0.4, 0.4, 0.4), (0.5, 0.5, 0.5)) if inside
                else box((7, 7, 7), (7.2, 7.2, 7.2)))
    return np.concatenate(boxes)


def test_nested_island_is_found():
    """A part of a hundred boxes with one of them inside another part overlaps it. A probe
    that took a few corners from the first triangles of the part would miss the island."""
    big = box((0, 0, 0), (1.0, 1.0, 1.0))
    for inside in (True, False):
        geometry = {**test_clearance.kit_geometry(),
                    "kit:Block": PieceGeom("Block", {"body": _many_boxes(inside)}),
                    "kit:Slab": PieceGeom("Slab", {"top": big})}
        ctx = test_clearance.yard(test_clearance.at("a", "Block"),
                                  test_clearance.at("b", "Slab"), geometry=geometry)
        got = [f.item for f in Clearance().run(ctx)]
        assert got == (["yard a x b"] if inside else []), inside
