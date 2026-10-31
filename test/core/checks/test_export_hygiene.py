"""Check export-hygiene: the packed .glb's node table agrees with its pack.

Per pack that holds a node table: no two nodes share a name; each piece root sits on the pack's
`spacing` grid (or, with no spacing, where the pack's layout places the piece); a piece a clip
carries has its root at the origin; every pack part has a node and every mesh node is a part.
Nodes are keyed by their place in the table, never by name, so two nodes of one name are both
seen. Items: `<pack> <node> duplicate-name`, `<pack>:<piece> root-offset`,
`<pack>:<piece> <part> no-node`, `<pack> <node> not-a-part`.

The scene is `assembly_scenes`: three pieces (`Rack`, `Lamp`, `Lead`) whose roots stand a
`SPACING` apart, each with its parts as mesh nodes named `<piece>/<part>`.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import export_hygiene
from model_kit.core.checks.export_hygiene import ExportHygiene
from model_kit.core.errors import BlindCheck
from model_kit.core.glb import GlbNode
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.tolerances import (
    CALIBRATION_FAIL,
    CALIBRATION_PASS,
    FLOAT32_EPS,
    NODE_OF_PIECE_SIZE,
)

from .assembly_scenes import (
    LAMP_AT,
    SPACING,
    carrying,
    make_ctx,
    node,
    node_tol,
    plants_caught,
    table,
    with_nodes,
)
from .piece_scenes import refuses_the_core, tried_and_caught


def findings(ctx, only=None):
    return ExportHygiene().run(ctx, only=only)


def items(ctx):
    return sorted(f.item for f in findings(ctx))


def clean(**kw):
    """A scene on a grid of `SPACING` with the clean node table of its pack."""
    return with_nodes(make_ctx(spacing=SPACING, **kw))


def placed():
    """A pack in its stage frame (no spacing): the layout places each piece."""
    return with_nodes(make_ctx())


def edited(ctx, edit):
    """The scene with its node table changed by `edit(list of nodes)`."""
    nodes = list(ctx.nodes["kit"])
    edit(nodes)
    return dataclasses.replace(ctx, nodes={"kit": nodes})


def index_of(ctx, name):
    return next(i for i, n in enumerate(ctx.nodes["kit"]) if n.name == name)


def shift(ctx, name, dx):
    def edit(nodes):
        i = index_of(ctx, name)
        x, y, z = nodes[i].translation
        nodes[i] = dataclasses.replace(nodes[i], translation=(x + dx, y, z))

    return edited(ctx, edit)


def test_the_check_is_registered_and_is_a_pack_check():
    assert gate.CHECKS["export-hygiene"] is ExportHygiene
    assert ExportHygiene.name == "export-hygiene"
    assert ExportHygiene.kinds == frozenset()


def test_a_clean_table_passes_and_every_plant_is_caught():
    for ctx in (clean(), placed()):
        assert findings(ctx) == []
        assert plants_caught(ExportHygiene(), ctx)
        assert run_gate([ExportHygiene()], ctx, None).status == "passed"


# --- duplicate names ----------------------------------------------------------------------------


def test_two_nodes_of_one_name_are_reported_once():
    ctx = edited(clean(), lambda nodes: nodes.append(nodes[1]))  # a second 'Rack/frame'
    (found,) = findings(ctx)
    assert (found.check, found.item) == ("export-hygiene", "kit Rack/frame duplicate-name")


def test_a_duplicate_root_that_stands_off_its_slot_is_seen_by_index():
    # the table holds two nodes named Lamp; only the second is off the grid
    ctx = edited(clean(), lambda nodes: nodes.append(node("Lamp", None, (5.5, 0.0, 0.0))))
    assert items(ctx) == ["kit Lamp duplicate-name", "kit:Lamp root-offset"]


def test_two_mesh_nodes_of_a_parts_name_do_not_hide_each_other_or_the_part():
    ctx = edited(clean(), lambda nodes: nodes.append(
        node("Lamp/pin", "Lamp", mesh="Lamp/pin")))
    assert items(ctx) == ["kit Lamp/pin duplicate-name"]


# --- the root on its slot -----------------------------------------------------------------------


def test_calibration_half_the_end_limit_off_its_slot_passes_and_double_fails():
    assert findings(shift(clean(), "Lamp", CALIBRATION_PASS * node_tol(clean(), "Lamp"))) == []
    (found,) = findings(shift(clean(), "Lamp", CALIBRATION_FAIL * node_tol(clean(), "Lamp")))
    assert found.item == "kit:Lamp root-offset"


def test_with_a_spacing_a_root_stands_on_a_slot_of_the_grid():
    ctx = clean()
    assert [n.translation[0] for n in ctx.nodes["kit"] if n.parent is None] == [0.0, 1.0, 2.0]
    assert findings(shift(ctx, "Lamp", SPACING)) == []  # another slot of the grid
    assert items(shift(ctx, "Lamp", 0.5 * SPACING)) == ["kit:Lamp root-offset"]  # between slots


def test_a_root_raised_off_the_ground_is_off_its_slot():
    def edit(nodes):
        i = index_of(clean(), "Rack")
        nodes[i] = dataclasses.replace(nodes[i], translation=(0.0, 0.5, 0.0))

    assert items(edited(clean(), edit)) == ["kit:Rack root-offset"]


def test_with_no_spacing_the_root_sits_where_the_layout_places_the_piece():
    assert findings(placed()) == []
    off = shift(placed(), "Lamp", CALIBRATION_FAIL * node_tol(placed(), "Lamp"))
    (found,) = findings(off)
    assert found.item == "kit:Lamp root-offset"
    lamp = next(n for n in placed().nodes["kit"] if n.name == "Lamp")
    assert tuple(lamp.translation) == tuple(LAMP_AT)  # the layout's place, not the origin


def test_a_piece_no_layout_places_and_no_clip_carries_is_unmeasured():
    ctx = dataclasses.replace(placed(), layouts={})
    assert findings(ctx) == []
    lines = ExportHygiene().coverage(ctx)
    unmeasured = [line for line in lines if isinstance(line, Unmeasured)
                  and "no layout places it" in line]
    assert sorted(line.rsplit(": ", 1)[1] for line in unmeasured) == [
        "kit:Lamp", "kit:Lead", "kit:Rack"]
    # no root can be planted on either: the rule judges none of the pieces, so it has no unit
    assert not any(u.item.endswith("root-offset") for u in ExportHygiene().plants(ctx))
    assert run_gate([ExportHygiene()], ctx, None).status == "partial"


# --- a carried piece stands at the origin -----------------------------------------------------------


def test_a_carried_piece_has_its_root_at_the_origin():
    ctx = carrying(with_nodes(make_ctx(spacing=SPACING)), "Rack")
    assert findings(ctx) == []  # the rack is slot 0
    ctx = carrying(with_nodes(make_ctx(spacing=SPACING)), "Lamp")
    assert items(ctx) == ["kit:Lamp root-offset"]  # the lamp is slot 1
    base = with_nodes(make_ctx(spacing=SPACING))
    off = shift(carrying(base, "Rack"), "Rack", CALIBRATION_FAIL * node_tol(base, "Rack"))
    assert items(off) == ["kit:Rack root-offset"]


def test_a_carried_piece_in_a_pack_with_no_spacing_needs_no_layout_place():
    ctx = carrying(dataclasses.replace(placed(), layouts={}), "Lamp")
    nodes = [dataclasses.replace(n, translation=(0.0, 0.0, 0.0)) if n.name == "Lamp" else n
             for n in ctx.nodes["kit"]]
    ctx = dataclasses.replace(ctx, nodes={"kit": nodes})
    assert findings(ctx) == []
    lamp_lines = [line for line in ExportHygiene().coverage(ctx)
                  if isinstance(line, Unmeasured) and "kit:Lamp" in line]
    assert lamp_lines == []


def test_a_missing_root_is_a_root_offset():
    ctx = edited(clean(), lambda nodes: nodes.pop(index_of(clean(), "Lamp")))
    found = [f for f in findings(ctx) if f.item == "kit:Lamp root-offset"]
    assert found and "no root" in found[0].detail


# --- parts and nodes -----------------------------------------------------------------------------


def test_a_part_with_no_node_fails():
    ctx = edited(clean(), lambda nodes: nodes.pop(index_of(clean(), "Lamp/pin")))
    assert items(ctx) == ["kit:Lamp pin no-node"]


def test_a_mesh_node_that_is_no_part_fails():
    ctx = edited(clean(), lambda nodes: nodes.append(node("Lamp/extra", "Lamp", mesh="Lamp/extra")))
    assert items(ctx) == ["kit Lamp/extra not-a-part"]


def test_a_node_without_a_mesh_need_not_be_a_part():
    ctx = edited(clean(), lambda nodes: nodes.append(node("Lamp/pivot", "Lamp")))
    assert findings(ctx) == []


def test_a_node_renamed_after_finish_fails():
    """The packed file's node names differ from the pack's parts."""
    def edit(nodes):
        i = index_of(clean(), "Lamp/shade")
        nodes[i] = dataclasses.replace(nodes[i], name="Lamp/shadow")

    ctx = edited(clean(), edit)
    assert items(ctx) == ["kit Lamp/shadow not-a-part", "kit:Lamp shade no-node"]
    verdict = run_gate([ExportHygiene()], ctx, None)
    assert verdict.status == "failed" and verdict.failed == 2


def test_the_node_name_of_a_part_is_the_piece_slash_the_part():
    ctx = clean()
    names = {n.name for n in ctx.nodes["kit"] if n.mesh is not None}
    assert names == {"Rack/frame", "Lamp/pin", "Lamp/shade", "Lead/cable", "Lead/plug",
                     "Lead/pin"}


# --- the plants, the scope and the gaps --------------------------------------------------------------


def test_each_rule_has_a_plant():
    plants = ExportHygiene().plants(clean())
    kinds = sorted(u.item.rsplit(" ", 1)[1] for u in plants)
    assert kinds == ["duplicate-name", "no-node", "not-a-part", "root-offset"]


def test_the_root_plant_offsets_a_root_twice_the_limit():
    (unit,) = [u for u in ExportHygiene().plants(clean()) if u.item.endswith("root-offset")]
    assert len(unit.candidates) == 3  # x, then y, then z
    plant = unit.candidates[0]
    piece = plant.item.split()[0].split(":")[1]
    before = next(n for n in clean().nodes["kit"] if n.name == piece)
    after = next(n for n in plant.ctx.nodes["kit"] if n.name == piece)
    moved = np.asarray(after.translation) - np.asarray(before.translation)
    assert np.linalg.norm(moved) == pytest.approx(
        CALIBRATION_FAIL * node_tol(clean(), piece))


def test_the_plants_never_call_the_core(monkeypatch):
    refuses_the_core(monkeypatch, export_hygiene, "_pack_findings", "_all_findings")
    assert len(ExportHygiene().plants(clean())) == 4


def test_a_unit_whose_item_already_fails_is_skipped_not_replanted():
    # the unit sits on the first part (Rack frame); its node is gone for real
    ctx = edited(clean(), lambda nodes: nodes.pop(index_of(clean(), "Rack/frame")))
    assert "kit:Rack frame no-node" in {u.item for u in ExportHygiene().plants(ctx)}
    assert "kit:Rack frame no-node" in items(ctx)
    assert run_gate([ExportHygiene()], ctx, None).status == "failed"


def test_the_plants_are_caught_on_two_packs():
    for ctx in (clean(), placed()):
        assert tried_and_caught(ExportHygiene(), ctx)


def test_a_part_whose_node_name_is_shared_is_still_planted():
    # two mesh nodes of the part's name: taking out "the" node must take out both
    ctx = edited(clean(), lambda nodes: nodes.append(node("Rack/frame", "Rack", mesh="Rack/frame")))
    (unit,) = [u for u in ExportHygiene().plants(ctx) if u.item.endswith("no-node")]
    (plant,) = unit.candidates
    assert not [n for n in plant.ctx.nodes["kit"] if n.name == "Rack/frame"]
    assert tried_and_caught(ExportHygiene(), ctx)


def test_a_scoped_run_yields_only_the_items_it_names():
    ctx = shift(edited(clean(), lambda nodes: nodes.pop(index_of(clean(), "Lamp/pin"))), "Rack",
                CALIBRATION_FAIL * node_tol(clean(), "Rack"))
    assert items(ctx) == ["kit:Lamp pin no-node", "kit:Rack root-offset"]
    assert [f.item for f in findings(ctx, only={"kit:Rack root-offset"})] == [
        "kit:Rack root-offset"]


def test_a_blind_check_stops_the_run():
    class Blind(ExportHygiene):
        def run(self, ctx, only=None, samples=frozenset()):
            return []

    with pytest.raises(BlindCheck, match="export-hygiene"):
        run_gate([Blind()], clean(), None)


def test_a_pack_with_no_node_table_is_unmeasured_and_the_run_partial():
    ctx = make_ctx(spacing=SPACING)
    assert findings(ctx) == []
    lines = ExportHygiene().coverage(ctx)
    assert any(isinstance(line, Unmeasured) and "no node table" in line and "kit" in line
               for line in lines), lines
    assert run_gate([ExportHygiene()], ctx, None).status == "partial"


def test_a_table_is_a_list_of_glb_nodes():
    assert all(isinstance(n, GlbNode) for n in table(make_ctx(spacing=SPACING)))


# --- the root tolerance follows the piece, not the stock ------------------------------------------

HELD_BASE = 0.007175  # a pack of thick stock and small pieces


def held():
    """A pack of thick stock whose pieces are small, like drumsticks: a limit that followed
    the stock (66.7 bases, 0.478 here) would be more than three times the
    lamp's size."""
    ctx = carrying(with_nodes(make_ctx(base=HELD_BASE)), "Lamp")
    nodes = [dataclasses.replace(n, translation=(0.0, 0.0, 0.0)) if n.name == "Lamp" else n
             for n in ctx.nodes["kit"]]
    return dataclasses.replace(ctx, nodes={"kit": nodes}, layouts={})


def test_a_carried_root_on_thick_stock_is_held_to_the_piece_size():
    ctx = held()
    assert findings(ctx) == []
    tol = node_tol(ctx, "Lamp")
    assert tol < 1e-4  # a limit that followed the stock (66.7 x HELD_BASE) would be 0.478
    assert findings(shift(ctx, "Lamp", CALIBRATION_PASS * tol)) == []
    assert items(shift(ctx, "Lamp", CALIBRATION_FAIL * tol)) == ["kit:Lamp root-offset"]
    assert items(shift(ctx, "Lamp", 0.4)) == ["kit:Lamp root-offset"]
    assert plants_caught(ExportHygiene(), ctx)


def test_a_piece_with_no_geometry_has_no_size_so_its_root_is_unmeasured():
    ctx = dataclasses.replace(clean(), pieces={k: v for k, v in clean().pieces.items()
                                                if k != "kit:Lamp"})
    assert items(shift(ctx, "Lamp", 5.5)) == []
    lines = ExportHygiene().coverage(ctx)
    assert any(isinstance(line, Unmeasured) and "no geometry" in line and "kit:Lamp" in line
               for line in lines), lines
    assert run_gate([ExportHygiene()], ctx, None).status == "partial"


def piece_size(ctx, piece):
    every = np.concatenate([t.reshape(-1, 3) for t in ctx.pieces[f"kit:{piece}"].parts.values()])
    return float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))


def test_a_root_far_out_on_the_grid_may_carry_its_float32_rounding():
    bare = NODE_OF_PIECE_SIZE * piece_size(clean(), "Lamp")
    far = round(bare / FLOAT32_EPS / SPACING) * SPACING  # here 2 ulps of rounding is twice `bare`
    ctx = shift(clean(), "Lamp", far)
    at = next(n.translation for n in ctx.nodes["kit"] if n.name == "Lamp" and n.parent is None)
    two_ulps = 2 * FLOAT32_EPS * float(np.linalg.norm(at))
    rounded = float(np.float32(at[0] + two_ulps))  # a few ulps off, as the exporter writes it
    assert findings(shift(ctx, "Lamp", rounded - at[0])) == []
    assert items(shift(ctx, "Lamp", CALIBRATION_FAIL * node_tol(ctx, "Lamp"))) == [
        "kit:Lamp root-offset"]
