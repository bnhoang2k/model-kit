"""The `measured` hook: which parts of a scene each piece check looked at.

The coverage check fails an exported part that no check measured, so every piece check says which
parts it measured, as `scene.part_key` strings (`<pack>:<piece> <part>`, the item form of floaters
and topology). A check lists a part only when it could judge it: a part it left out by design, a
piece of another recipe kind and a piece with no geometry are not listed.
"""

import dataclasses

import numpy as np

from model_kit.core import gate
from model_kit.core.checks.branching import BRANCHING_CHECKS
from model_kit.core.checks.branching._pieces import grown as grown_of
from model_kit.core.checks.branching.crossing import CrossingOrganic
from model_kit.core.checks.branching.floor import FloorOrganic
from model_kit.core.checks.branching.limb_seat import LimbSeat
from model_kit.core.checks.branching.near_miss import NearMissOrganic
from model_kit.core.checks.branching.skeleton_match import SkeletonMatch
from model_kit.core.checks.connectors import Connectors
from model_kit.core.checks.crossings import Crossings
from model_kit.core.checks.done import Done
from model_kit.core.checks.floaters import Floaters
from model_kit.core.checks.frames import Frames
from model_kit.core.checks.pieces import measured_pieces
from model_kit.core.checks.receiver import Receiver
from model_kit.core.checks.take_apart import TakeApart
from model_kit.core.checks.topology import Topology
from model_kit.core.grow.mesh import grown_parts, leaf_part_name
from model_kit.core.metagate import Measures
from model_kit.core.scene import PieceGeom, part_key, piece_key

from ..shapes import box
from .branching.branching_scene import PACK, PIECE, grown_ctx, small_tree
from .manifest_scenes import make_ctx as deck_ctx
from .piece_scenes import make_ctx, post_on_base, with_parts


def test_part_key_is_the_item_form_of_floaters_and_topology():
    assert part_key("kit", "Thing", "post") == "kit:Thing post"
    assert part_key("kit", "Thing", "post") == f"{piece_key('kit', 'Thing')} post"


# --- crossings --------------------------------------------------------------------------------


def test_crossings_measures_every_solid_part_of_a_piece_with_two_or_more():
    ctx = make_ctx(post_on_base())
    assert Crossings().measured(ctx) == {"kit:Thing base", "kit:Thing post"}


def test_crossings_leaves_a_lone_solid_part_and_printed_parts_out():
    lone = make_ctx({"base": post_on_base()["base"]})
    assert Crossings().measured(lone) == set()
    printed = make_ctx({**post_on_base(), "mark": box((0, 0, 0), (0.01, 0.01, 0.01))},
                       {"mark": {"role": "decal"}})
    assert Crossings().measured(printed) == {"kit:Thing base", "kit:Thing post"}


def test_crossings_leaves_a_part_with_no_triangles_out():
    ctx = with_parts(make_ctx(post_on_base()), ghost=np.zeros((0, 3, 3)))
    assert "kit:Thing ghost" not in Crossings().measured(ctx)


def test_an_assembly_check_does_not_list_a_branching_piece():
    ctx = make_ctx(post_on_base(), also={"Tree": post_on_base()})
    for check in (Crossings(), TakeApart(), Floaters(), Topology()):
        assert not any(k.startswith("kit:Tree ") for k in check.measured(ctx)), check.name


# --- take-apart -------------------------------------------------------------------------------


def _two_groups():
    return make_ctx(post_on_base(), {"base": {"group": "plate"}, "post": {"group": "stem"}})


def test_take_apart_measures_the_parts_of_a_piece_with_two_groups():
    assert TakeApart().measured(_two_groups()) == {"kit:Thing base", "kit:Thing post"}


def test_take_apart_lists_nothing_for_a_piece_in_one_group():
    assert TakeApart().measured(make_ctx(post_on_base())) == set()


def test_take_apart_leaves_a_part_with_no_group_out():
    ctx = with_parts(_two_groups(), loose=box((0.3, 0, 0), (0.32, 0.02, 0.02)))
    assert TakeApart().measured(ctx) == {"kit:Thing base", "kit:Thing post"}


# --- floaters and topology ----------------------------------------------------------------------


def test_floaters_measures_every_part_with_triangles():
    ctx = with_parts(make_ctx(post_on_base()), ghost=np.zeros((0, 3, 3)))
    assert Floaters().measured(ctx) == {"kit:Thing base", "kit:Thing post"}


def test_topology_measures_every_part_but_leaf_cable_and_skip_outline():
    parts = {**post_on_base(), "lead": box((0.2, 0, 0), (0.4, 0.01, 0.01)),
             "sheet": box((0.5, 0, 0), (0.6, 0.01, 0.1)), "ink": box((0.7, 0, 0), (0.8, 0.01, 0.1))}
    ctx = make_ctx(parts, {"lead": {"shape": "cable"}, "sheet": {"shape": "leaf"},
                           "ink": {"role": "skip_outline"}})
    assert Topology().measured(ctx) == {"kit:Thing base", "kit:Thing post"}


# --- frames and done ----------------------------------------------------------------------------


def test_frames_measures_the_surface_parts_of_holds_and_seats():
    ctx = deck_ctx()
    assert Frames().measured(ctx) == {"kit:Deck top"}


def test_frames_measures_only_the_part_nearest_each_mount():
    ctx = deck_ctx()
    geom = ctx.pieces["kit:Deck"]
    knob = box((-0.01, 0.03, -0.01), (0.01, 0.05, 0.01))  # far from the jack
    far = dataclasses.replace(ctx, pieces={**ctx.pieces, "kit:Deck": PieceGeom(
        "Deck", {**geom.parts, "knob": knob})})
    assert Frames().measured(far) == {"kit:Deck top"}  # the jack stands on the top
    deck = ctx.packs["kit"].pieces["Deck"]
    over_knob = tuple(dataclasses.replace(m, point=(0.0, 0.1, 0.0)) for m in deck.mounts)
    pack = dataclasses.replace(ctx.packs["kit"], pieces={
        **ctx.packs["kit"].pieces, "Deck": dataclasses.replace(deck, mounts=over_knob)})
    moved = dataclasses.replace(far, packs={"kit": pack})
    assert Frames().measured(moved) == {"kit:Deck top", "kit:Deck knob"}
    bare = dataclasses.replace(deck, mounts=())
    pack = dataclasses.replace(ctx.packs["kit"], pieces={**ctx.packs["kit"].pieces, "Deck": bare})
    assert Frames().measured(dataclasses.replace(far, packs={"kit": pack})) == {"kit:Deck top"}


def test_the_three_part_key_builders_agree():
    from model_kit.core.checks.branching._pieces import Grown
    from model_kit.core.checks.pieces import item_of

    sk = small_tree()
    ctx = grown_ctx(sk)
    view = next(iter(measured_pieces(ctx, {"assembly"})))
    assert item_of(view, "post") == part_key(view.pack, view.name, "post")
    g = next(iter(grown_of(ctx)))
    assert isinstance(g, Grown)
    assert g.item("Branches") == part_key(g.pack, g.name, "Branches")


def test_done_reads_the_piece_record_and_has_no_measured_hook():
    assert not hasattr(Done(), "measured")
    assert not isinstance(Done(), Measures)


def test_receiver_measures_the_takes_part_and_the_host_part_of_each_guest_on_a_mount():
    from .assembly_scenes import make_ctx as lamp_ctx

    ctx = lamp_ctx()
    assert Receiver().measured(ctx) == {"kit:Lamp pin", "kit:Rack frame"}
    assert Receiver().measured(lamp_ctx(pieces=False)) == set()  # no geometry to judge


def test_connectors_measure_the_cable_part_and_the_seated_plug_and_host_of_each_connector():
    from .assembly_scenes import make_ctx as lead_ctx

    assert Connectors().measured(lead_ctx()) == {
        "kit:Lead cable", "kit:Lead plug", "kit:Lead pin", "kit:Rack frame"}
    assert Connectors().measured(lead_ctx(pieces=False)) == set()


# --- the branching checks -------------------------------------------------------------------------


def _limb_parts(sk):
    return {part_key(PACK, PIECE, name) for i, name in enumerate(sk.level_names)
            if (sk.level == i).any()}


def _leaf_parts(sk):
    return {part_key(PACK, PIECE, leaf_part_name(shade)) for i, shade in enumerate(sk.shades)
            if (sk.shade == i).any()}


def test_the_limb_checks_measure_the_limb_parts_and_leave_the_leaf_cards_out():
    sk = small_tree()
    ctx = grown_ctx(sk)
    assert _limb_parts(sk) and _leaf_parts(sk)
    assert CrossingOrganic().measured(ctx) == _limb_parts(sk)
    assert NearMissOrganic().measured(ctx) == _limb_parts(sk)


def test_limb_seat_and_floor_measure_limb_and_leaf_parts():
    sk = small_tree()
    ctx = grown_ctx(sk)
    assert LimbSeat().measured(ctx) == _limb_parts(sk) | _leaf_parts(sk)
    assert FloorOrganic().measured(ctx) == _limb_parts(sk) | _leaf_parts(sk)


def test_skeleton_match_measures_every_shipped_part_of_a_piece_with_a_skeleton():
    sk = small_tree()
    ctx = grown_ctx(sk)
    shipped = grown_parts(sk, None)
    assert SkeletonMatch().measured(ctx) == {part_key(PACK, PIECE, p) for p in shipped}
    extra = grown_ctx(sk, parts={**shipped, "Stray": shipped["Leaves_Leaf"]})
    assert part_key(PACK, PIECE, "Stray") in SkeletonMatch().measured(extra)


def test_the_branching_checks_list_nothing_without_a_skeleton_or_for_an_assembly_piece():
    ctx = grown_ctx(None, listed_from=small_tree())  # the tree has geometry but no skeleton
    for check in BRANCHING_CHECKS:
        assert check().measured(ctx) == set(), check.name
    for check in BRANCHING_CHECKS:  # the crate (assembly) is not a branching check's piece
        assert not any(k.startswith("grove:Crate ") for k in check().measured(
            grown_ctx(small_tree()))), check.name


# Checks that declare a kind and read the manifest, not a mesh: they have no hook.
NO_PART_CHECKS = {"done", "keep-off"}


def test_every_check_that_declares_a_kind_has_a_measured_hook_unless_it_measures_no_part():
    for name, check in gate.CHECKS.items():
        if check.kinds and name not in NO_PART_CHECKS:
            assert callable(getattr(check, "measured", None)), name
    for name in NO_PART_CHECKS:
        assert not hasattr(gate.CHECKS[name], "measured"), name


def test_the_measures_protocol_tells_checks_with_the_hook_from_those_without():
    from model_kit.core.checks.reach import Reach

    assert isinstance(Floaters(), Measures)
    assert isinstance(CrossingOrganic(), Measures)
    assert not isinstance(Reach(), Measures)
