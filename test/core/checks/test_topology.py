import dataclasses
import itertools

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import topology
from model_kit.core.checks.topology import Topology
from model_kit.core.errors import BlindCheck
from model_kit.core.geom import is_closed
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    WELD_OF_PIECE_SIZE,
    ZERO_AREA_OF_LONGEST_EDGE_SQUARED,
)

from ..shapes import box
from .furniture_scenes import furniture_ctx
from .piece_scenes import make_ctx, plants_caught, refuses_the_core, tried_and_caught

SLAB = box((0, 0, 0), (0.1, 0.02, 0.1))
SIZE = float(np.linalg.norm([0.1, 0.02, 0.1]))  # the piece size of a piece that is the slab
WELD = WELD_OF_PIECE_SIZE * SIZE
ITEM = "kit:Thing slab"


def findings(parts, specs=None, only=None):
    return Topology().run(make_ctx(parts, specs), only=only)


def flipped(tris, row):
    out = tris.copy()
    out[row] = out[row][[0, 2, 1]]
    return out


def test_the_check_is_registered_and_measures_assembly_pieces():
    assert gate.CHECKS["topology"] is Topology
    assert Topology.kinds == frozenset({"assembly"})


def test_a_closed_outward_box_passes_and_its_plants_are_caught():
    ctx = make_ctx({"slab": SLAB, "cap": box((0, 0.02, 0), (0.05, 0.04, 0.05))})
    assert Topology().run(ctx) == []
    assert plants_caught(Topology(), ctx)
    assert run_gate([Topology()], ctx, None).status == "passed"


def test_a_flipped_face_fails():
    (found,) = findings({"slab": flipped(SLAB, 3)})
    assert (found.check, found.item) == ("topology", ITEM)
    assert "flipped" in found.detail


def test_an_open_mesh_fails():
    (found,) = findings({"slab": SLAB[:-1]})
    assert "open" in found.detail


def test_an_edge_on_three_faces_fails():
    extra = np.array([[SLAB[0, 0], SLAB[0, 1], SLAB[0, 1] + [0.0, 0.5, 0.0]]])
    (found,) = findings({"slab": np.concatenate([SLAB, extra])})
    assert "3 or more faces" in found.detail


def test_a_zero_area_face_fails():
    sliver = np.array([[SLAB[0, 0], SLAB[0, 0], SLAB[0, 1]]])
    (found,) = findings({"slab": np.concatenate([SLAB, sliver])})
    assert "zero-area" in found.detail


def test_an_inside_out_mesh_has_no_positive_volume():
    (found,) = findings({"slab": SLAB[:, ::-1]})
    assert "volume" in found.detail and "flipped" not in found.detail


def test_a_mesh_with_no_triangles_fails():
    (found,) = findings({"slab": np.zeros((0, 3, 3))})
    assert "no triangles" in found.detail


def test_corners_inside_the_weld_still_close_and_corners_outside_it_do_not():
    off_grid = SLAB + 0.37 * WELD  # no corner sits on a cell centre
    near = off_grid.copy()
    near[::2] += CALIBRATION_PASS * WELD  # half of the triangles shifted: their corners differ
    assert findings({"slab": near}) == []
    far = off_grid.copy()
    far[::2] += CALIBRATION_FAIL * WELD
    (found,) = findings({"slab": far})
    assert "open" in found.detail


def test_leaf_cable_and_decal_shapes_and_skip_outline_parts_are_not_measured():
    broken = SLAB[:-1]
    for spec in ({"shape": "leaf"}, {"shape": "cable"}, {"shape": "decal", "role": "decal"},
                 {"role": "skip_outline"}):
        # a body sets the pack's thickness base; a decal part (a printed role) never does
        ctx = make_ctx({"slab": broken, "body": box((0, 0.02, 0), (0.05, 0.04, 0.05))},
                       {"slab": spec})
        assert Topology().run(ctx) == []
        assert any("not applicable" in line and "slab" in line for line in Topology().coverage(ctx))


def test_topology_skips_decals():
    """A decal is an open patch printed on its host: it is not a fault that it has a boundary."""
    ctx = furniture_ctx("Label")
    legend = ctx.pieces["kit:Label"].parts["legend"]
    assert not is_closed(legend)  # the printed legend really is an open patch
    assert Topology().run(ctx) == []
    assert "kit:Label legend" not in Topology().measured(ctx)
    assert "topology: not applicable (decal part): kit:Label legend" in Topology().coverage(ctx)


def test_printed_parts_are_measured():
    # a solid part is there too: the pack's thickness base comes from solid parts only
    parts = {"body": box((0, 0.02, 0), (0.05, 0.04, 0.05)), "slab": SLAB[:-1]}
    assert [f.item for f in findings(parts, {"slab": {"role": "eyes"}})] == [ITEM]


def test_a_scoped_run_measures_only_the_items_it_is_given():
    parts = {"slab": SLAB[:-1], "cap": box((0, 0.02, 0), (0.05, 0.04, 0.05))[:-1]}
    ctx = make_ctx(parts)
    assert {f.item for f in Topology().run(ctx)} == {ITEM, "kit:Thing cap"}
    assert [f.item for f in Topology().run(ctx, only={ITEM})] == [ITEM]


def test_a_plant_flips_one_face_of_a_clean_part():
    ctx = make_ctx({"slab": SLAB})
    (unit,) = Topology().plants(ctx)
    plant = unit.candidates[0]
    assert plant.item == ITEM
    changed = plant.ctx.pieces["kit:Thing"].parts["slab"] != SLAB
    assert changed.any(axis=(1, 2)).sum() == 1


def test_a_part_that_fails_already_still_gets_a_unit_and_the_gate_skips_it():
    ctx = make_ctx({"slab": SLAB[:-1]})
    (unit,) = Topology().plants(ctx)  # built from the triangles, not from the verdict
    assert unit.item == ITEM
    assert run_gate([Topology()], ctx, None).status == "failed"


def test_topology_plants_never_call_the_core(monkeypatch):
    refuses_the_core(monkeypatch, topology, "problems")
    ctx = make_ctx({"slab": SLAB, "cap": box((0, 0.02, 0), (0.05, 0.04, 0.05))})
    assert [u.item for u in Topology().plants(ctx)] == [ITEM, "kit:Thing cap"]


def test_topology_plants_are_caught_on_two_pieces():
    thing = make_ctx({"slab": SLAB, "cap": box((0, 0.02, 0), (0.05, 0.04, 0.05))})
    assert tried_and_caught(Topology(), thing) == [ITEM, "kit:Thing cap"]
    label = tried_and_caught(Topology(), furniture_ctx("Label"))
    assert label and all(item.startswith("kit:Label ") for item in label)


def test_a_part_with_no_triangles_and_the_parts_left_out_get_no_unit():
    skipped = {"slab": {"shape": "leaf"}}
    assert Topology().plants(make_ctx({"slab": SLAB}, skipped)) == []
    assert Topology().plants(make_ctx({"slab": SLAB[:0]})) == []


def test_a_branching_piece_is_left_to_the_branching_checks():
    ctx = make_ctx({"slab": SLAB}, also={"Tree": {"slab": SLAB[:-1]}})
    assert Topology().run(ctx) == []
    (line,) = Topology().coverage(ctx)
    assert line == "topology: not applicable (branching piece): kit:Tree"
    assert not isinstance(line, Unmeasured)  # the branching checks measure it
    assert run_gate([Topology()], ctx, None).status == "passed"


def test_a_piece_with_no_geometry_in_the_scene_is_unmeasured():
    ctx = dataclasses.replace(make_ctx({"slab": SLAB}), pieces={})
    (line,) = Topology().coverage(ctx)
    assert line == "topology: skipped (no geometry): kit:Thing"
    assert isinstance(line, Unmeasured)
    assert run_gate([Topology()], ctx, None).status == "partial"


def test_a_blind_check_stops_the_run():
    class Blind(Topology):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="topology"):
        run_gate([Blind()], make_ctx({"slab": SLAB}), None)


# --- the zero-area test and the weld follow the piece, not the wall thickness ------------------


def tube(radius, length, sides=32, rings=3):
    """A closed tube along z: `rings` rings of `sides` corners, a fan cap at each end whose
    pole is one corner. Ring meshes with pole fans, like a limb of a modelled figure."""
    angle = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    zs = np.linspace(0.0, length, rings)
    tris = []
    for z0, z1 in itertools.pairwise(zs):
        for i in range(sides):
            j = (i + 1) % sides
            a, b = (*ring[i], z0), (*ring[j], z0)
            c, d = (*ring[j], z1), (*ring[i], z1)
            tris += [(a, b, c), (a, c, d)]
    for i in range(sides):
        j = (i + 1) % sides
        tris.append(((0, 0, 0), (*ring[j], 0), (*ring[i], 0)))
        tris.append(((0, 0, length), (*ring[i], length), (*ring[j], length)))
    return np.array(tris, dtype=float)


def test_a_pole_fan_ring_mesh_of_small_faces_passes():
    # every face is small (a fan triangle is 2.4e-8 in area, a ring edge 1e-4 long), as on a
    # figure's limb; the pack's wall is thick next to them
    mesh = tube(radius=0.0005, length=0.1)
    assert findings({"limb": mesh}) == []
    assert Topology().plants(make_ctx({"limb": mesh}))


def test_a_sliver_face_never_opens_an_edge():
    (found,) = findings({"slab": sliver_slab(CALIBRATION_PASS * ZERO_AREA_OF_LONGEST_EDGE_SQUARED)})
    assert "1 zero-area faces" in found.detail
    assert "open" not in found.detail and "3 or more" not in found.detail
    assert "flipped" not in found.detail


def sliver_slab(ratio):
    """The slab with its first face split into three by a corner `d` inside it, near the edge
    `b c`: the face `b c d` has `ratio` as its area over its longest edge squared."""
    a, b, c = SLAB[0]
    height = 2.0 * ratio * np.linalg.norm(c - b)
    inward = (a - (b + c) / 2.0)
    d = (b + c) / 2.0 + height * inward / np.linalg.norm(inward)
    return np.concatenate([[[a, b, d], [b, c, d], [c, a, d]], SLAB[1:]])


def test_a_face_over_the_zero_area_limit_is_not_a_zero_area_face():
    ratio = CALIBRATION_FAIL * ZERO_AREA_OF_LONGEST_EDGE_SQUARED
    assert findings({"slab": sliver_slab(ratio)}) == []


def test_the_zero_area_test_does_not_depend_on_the_size_of_the_piece():
    # the same shapes at 1000 times the size: the verdicts do not change
    for ratio, clean in ((CALIBRATION_FAIL * ZERO_AREA_OF_LONGEST_EDGE_SQUARED, True),
                         (CALIBRATION_PASS * ZERO_AREA_OF_LONGEST_EDGE_SQUARED, False)):
        big = {"slab": 1000.0 * sliver_slab(ratio)}
        assert (findings(big) == []) is clean


def test_three_collinear_corners_are_a_zero_area_face():
    a, _, c = SLAB[0]
    line = np.array([[a, (a + c) / 2.0, c]])
    (found,) = findings({"slab": np.concatenate([line, SLAB])})
    assert "zero-area" in found.detail


def test_a_piece_that_is_a_single_point_has_no_extent():
    (found,) = findings({"slab": np.zeros((1, 3, 3))})
    assert found.detail == "has no extent"
