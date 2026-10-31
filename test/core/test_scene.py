import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from model_kit.core.checks.clip_hygiene import ClipHygiene
from model_kit.core.checks.near_crease import NearCrease
from model_kit.core.errors import ManifestError
from model_kit.core.glb import GlbNode
from model_kit.core.grow.buds import grow_buds
from model_kit.core.grow.rules import KIT_RULES, load_rules
from model_kit.core.grow.skeleton import Skeleton, load_skeleton, same_skeleton, save_skeleton
from model_kit.core.manifest import (
    Clip,
    FloorGear,
    Lift,
    Pack,
    Recipe,
    Ref,
    View,
    load_clips,
    load_layouts,
    load_pack,
)
from model_kit.core.metagate import Unmeasured, run_gate
from model_kit.core.scene import (
    ClipSamples,
    PieceGeom,
    RigData,
    RigRest,
    SceneCtx,
    _write_data,
    load_scene,
    piece_key,
    rig_spec,
    save_scene,
    stage_manifests,
    write_nodes,
)
from model_kit.core.units import Style, load_style

from .shapes import block, plate
from .test_glb import TREE, write_glb

FIX = Path(__file__).parent / "fixtures"
TINY_SPIRAL = Path(__file__).resolve().parent / "grow" / "fixtures" / "tiny-spiral.rules.json"


def shop_ctx() -> SceneCtx:
    style = load_style(FIX / "shop.style.json")
    pack = load_pack(FIX / "shop.pack.json", style)
    clips = {clip.name: clip for clip in load_clips(FIX / "shop.clips.json")}
    pieces = {
        piece_key("shop", "Bench"): PieceGeom("Bench", {
            "leg_L": block(0.04, 0.45), "leg_R": block(0.04, 0.45), "seat_top": plate(0.6, 0.02),
        }),
        piece_key("shop", "Lamp"): PieceGeom("Lamp", {"stem": block(0.012, 1.2)}),
    }
    rng = np.random.default_rng(7)
    times = np.arange(1.0, 49.0, 0.5)
    s = len(times)
    samples = {
        "sit_reach": ClipSamples(
            clip="sit_reach",
            times=times,
            groups={"hand.R": rng.normal(size=(s, 5, 3)), "foot.R": rng.normal(size=(s, 4, 3))},
            bones={"hand.R": np.tile(np.eye(4), (s, 1, 1)), "body": rng.normal(size=(s, 4, 4))},
            props={"lamp": rng.normal(size=(s, 4, 4))},
        ),
    }
    doll = pack.pieces["Doll"].rig  # the rig file agrees with the pack's rig
    bones = tuple(b.name for b in doll.bones)
    rig = RigData(
        order=bones,
        parent={b.name: b.parent for b in doll.bones},
        rest={b: rng.normal(size=(4, 4)) for b in bones},
        world=rng.normal(size=(4, 4)),
        effector_mid={"hand.R": np.array([0.0, 0.05, 0.0])},
    )
    return SceneCtx(
        style=style, packs={"shop": pack}, clips=clips, pieces=pieces, samples=samples,
        rigs={"shop:Doll": rig},
        rig_of_clip={"sit_reach": "shop:Doll", "sit_idle": "shop:Doll"},
        layouts=load_layouts(FIX / "layouts", {"shop": pack}),
    )


def small_tree() -> Skeleton:
    """A small, quick tree, grown lite."""
    return grow_buds(load_rules(TINY_SPIRAL), piece="Lamp", height_m=1.5, seed=3, lite=True,
                     units_per_metre=0.5)


def branching_ctx() -> SceneCtx:
    """The shop scene with its Lamp made a branching piece that carries a grown skeleton."""
    ctx = shop_ctx()
    pack = ctx.packs["shop"]
    lamp = dataclasses.replace(pack.pieces["Lamp"], recipe=Recipe("branching", "grow.py"))
    pack = Pack(name=pack.name, units_per_metre=pack.units_per_metre,
                pieces={**pack.pieces, "Lamp": lamp}, spacing=pack.spacing)
    return dataclasses.replace(ctx, packs={"shop": pack},
                               skeletons={piece_key("shop", "Lamp"): small_tree()})


def _same_arrays(a: dict, b: dict) -> None:
    assert sorted(a) == sorted(b)
    for key, value in a.items():
        np.testing.assert_array_equal(np.asarray(value), np.asarray(b[key]))


def test_scene_round_trip(tmp_path):
    ctx = shop_ctx()
    out = tmp_path / "scene"
    save_scene(ctx, out)

    # The on-disk layout.
    assert (out / "style.json").is_file()
    assert (out / "packs" / "shop.pack.json").is_file()
    assert (out / "clips.json").is_file()
    assert (out / "pieces" / "shop" / "Bench.npz").is_file()
    assert (out / "samples" / "sit_reach.npz").is_file()
    assert (out / "rigs" / "shop" / "Doll.json").is_file()
    with np.load(out / "samples" / "sit_reach.npz") as npz:
        assert sorted(npz.files) == sorted(
            ["times", "g:hand.R", "g:foot.R", "b:hand.R", "b:body", "p:lamp"])
    with np.load(out / "pieces" / "shop" / "Bench.npz") as npz:
        assert sorted(npz.files) == ["leg_L", "leg_R", "seat_top"]
    pack_json = json.loads((out / "packs" / "shop.pack.json").read_text())
    assert pack_json["pieces"]["Doll"]["rig"]["bind"]["arm_R"]["mode"] == "chain"
    assert "floor_gear" in json.loads((out / "clips.json").read_text())["clips"]["sit_reach"]

    back = load_scene(out)
    assert back.style == ctx.style
    assert back.packs == ctx.packs
    assert back.clips == ctx.clips
    assert back.rig_of_clip == ctx.rig_of_clip
    assert sorted(back.pieces) == sorted(ctx.pieces)
    for key, geom in ctx.pieces.items():
        assert back.pieces[key].name == geom.name
        _same_arrays(back.pieces[key].parts, geom.parts)
    assert sorted(back.samples) == ["sit_reach"]
    got, want = back.samples["sit_reach"], ctx.samples["sit_reach"]
    assert got.clip == want.clip
    np.testing.assert_array_equal(got.times, want.times)
    _same_arrays(got.groups, want.groups)
    _same_arrays(got.bones, want.bones)
    _same_arrays(got.props, want.props)
    rig, want_rig = back.rigs["shop:Doll"], ctx.rigs["shop:Doll"]
    assert rig.order == want_rig.order
    assert rig.parent == want_rig.parent
    _same_arrays(rig.rest, want_rig.rest)
    np.testing.assert_array_equal(rig.world, want_rig.world)
    _same_arrays(rig.effector_mid, want_rig.effector_mid)


def test_crease_angle_survives_a_saved_scene(tmp_path):
    ctx = shop_ctx()
    dials = {**ctx.style.dials,
             "edges": {"outline": "none", "crease_deg": 30, "crease_why": "a 30 degree rim"}}
    style = Style(ctx.style.units_per_metre, ctx.style.page_camera, dials)
    save_scene(dataclasses.replace(ctx, style=style), tmp_path / "scene")
    back = load_scene(tmp_path / "scene").style
    assert (back.crease_deg, back.crease_why) == (30, "a 30 degree rim")
    assert back.draws_outline is False


def test_farthest_past_far_plane_raises_at_scene_load(tmp_path):
    ctx = shop_ctx()
    camera = {"depth_projection": "perspective", "near_m": 0.1, "far_m": 2000,
              "depth_bits": 24, "depth_why": "a test camera"}
    style = Style(ctx.style.units_per_metre, camera, ctx.style.dials)
    pack = ctx.packs["shop"]
    lamp = dataclasses.replace(pack.pieces["Lamp"], view=View(1.0, "x", farthest_m=2000.0,
                                                              farthest_why="the far plane itself"))

    def scene_with(piece, where: str) -> Path:
        packs = {"shop": dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": piece})}
        save_scene(dataclasses.replace(ctx, style=style, packs=packs), tmp_path / where)
        return tmp_path / where

    assert load_scene(scene_with(lamp, "ok")).style.depth_camera.far_m == 2000.0
    far = dataclasses.replace(lamp, view=dataclasses.replace(lamp.view, farthest_m=2000.5))
    with pytest.raises(ManifestError, match=r"Lamp.view.farthest_m: 2000.5 m is past"):
        load_scene(scene_with(far, "far"))
    beyond = dataclasses.replace(lamp, view=View(2500.0, "x"))
    with pytest.raises(ManifestError, match=r"Lamp.view.nearest_m: 2500.0 m is past"):
        load_scene(scene_with(beyond, "beyond"))
    near = dataclasses.replace(lamp, view=View(0.05, "x"))
    with pytest.raises(ManifestError, match=r"Lamp.view.nearest_m: 0.05 m is nearer"):
        load_scene(scene_with(near, "near"))


def test_save_refuses_a_folder_with_files(tmp_path):
    (tmp_path / "stale.txt").write_text("old")
    with pytest.raises(ManifestError, match="not empty"):
        save_scene(shop_ctx(), tmp_path)


def test_sample_with_wrong_length_stops(tmp_path):
    ctx = shop_ctx()
    save_scene(ctx, tmp_path / "scene")
    path = tmp_path / "scene" / "samples" / "sit_reach.npz"
    with np.load(path) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["g:hand.R"] = arrays["g:hand.R"][:-1]
    np.savez(path, **arrays)
    with pytest.raises(ManifestError, match="g:hand.R"):
        load_scene(tmp_path / "scene")


def test_seat_ref_must_name_a_seat(tmp_path):
    ctx = shop_ctx()
    clip = ctx.clips["sit_reach"]
    wrong = dataclasses.replace(clip, seat=Ref.parse("shop:Bench_L.edge_R"))  # a hold
    ctx = dataclasses.replace(ctx, clips={**ctx.clips, "sit_reach": wrong})
    with pytest.raises(ManifestError, match="seat"):
        save_scene(ctx, tmp_path / "a")
    save_scene(shop_ctx(), tmp_path / "b")
    path = tmp_path / "b" / "clips.json"
    data = json.loads(path.read_text())
    data["clips"]["sit_reach"]["seat"] = "shop:Bench_L.edge_R"
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match="seat"):
        load_scene(tmp_path / "b")


def _bad_scenes():
    ctx = shop_ctx()
    s = ctx.samples["sit_reach"]
    short = dataclasses.replace(s, groups={**s.groups, "hand.R": s.groups["hand.R"][:-1]})
    stray = dataclasses.replace(s, clip="no_such_clip")
    yield "sample shape", dataclasses.replace(ctx, samples={"sit_reach": short}), "g:hand.R"
    yield "unknown clip samples", dataclasses.replace(
        ctx, samples={"no_such_clip": stray}), "no_such_clip"
    yield "rig for unknown clip", dataclasses.replace(
        ctx, rig_of_clip={**ctx.rig_of_clip, "no_such_clip": "shop:Doll"}), "no_such_clip"
    yield "geometry for unknown piece", dataclasses.replace(
        ctx, pieces={**ctx.pieces, piece_key("shop", "Ghost"): PieceGeom("Ghost", {})}), "Ghost"
    yield "geometry for unloaded pack", dataclasses.replace(
        ctx, pieces={**ctx.pieces, piece_key("other", "Bench"): PieceGeom("Bench", {})}), "other"
    yield "piece key", dataclasses.replace(
        ctx, pieces={"shop:Bench": PieceGeom("Lamp", {})}), "Lamp"


@pytest.mark.parametrize("what, ctx, names", list(_bad_scenes()), ids=lambda v: str(v)[:24])
def test_save_refuses_before_writing_anything(tmp_path, what, ctx, names):
    out = tmp_path / "scene"
    with pytest.raises(ManifestError, match=names) as err:
        save_scene(ctx, out)
    assert "model-kit-scene-" not in str(err.value)  # never the deleted staging folder
    if "scene/" in str(err.value) or str(tmp_path) in str(err.value):
        assert str(out) in str(err.value)
    assert not out.exists()  # nothing written, so a fixed scene can still be saved here
    save_scene(shop_ctx(), out)
    assert (out / "clips.json").is_file()


def test_load_refuses_what_save_refuses(tmp_path):
    save_scene(shop_ctx(), tmp_path / "a")
    rig_path = tmp_path / "a" / "rigs" / "shop" / "Doll.json"
    rig = json.loads(rig_path.read_text())
    rig["clips"].append("no_such_clip")
    rig_path.write_text(json.dumps(rig))
    with pytest.raises(ManifestError, match="no_such_clip"):
        load_scene(tmp_path / "a")

    save_scene(shop_ctx(), tmp_path / "b")
    (tmp_path / "b" / "pieces" / "other").mkdir()
    np.savez(tmp_path / "b" / "pieces" / "other" / "Bench.npz", top=plate())
    with pytest.raises(ManifestError, match="other"):
        load_scene(tmp_path / "b")


def test_refused_save_names_the_target_folder(tmp_path):
    ctx = shop_ctx()
    s = ctx.samples["sit_reach"]
    stray = dataclasses.replace(ctx, samples={"no_such_clip": dataclasses.replace(
        s, clip="no_such_clip")})
    out = tmp_path / "target"
    with pytest.raises(ManifestError) as err:
        save_scene(stray, out)
    assert str(err.value).startswith(str(out / "samples" / "no_such_clip.npz"))


def test_save_scene_writes_packs_through_write_pack(tmp_path):
    ctx = shop_ctx()
    pack = ctx.packs["shop"]
    bench = dataclasses.replace(pack.pieces["Bench"], measured={"foot": (0.6, 0.3)})
    bad = dataclasses.replace(pack, pieces={**pack.pieces, "Bench": bench})
    ctx = dataclasses.replace(ctx, packs={"shop": bad})
    out = tmp_path / "scene"
    with pytest.raises(ManifestError, match=r"Bench.*measured.*foot") as err:
        save_scene(ctx, out)
    assert str(out) in str(err.value) and "model-kit-scene-" not in str(err.value)
    assert not out.exists()


def _rewrite(path: Path, **changes) -> None:
    with np.load(path) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays.update(changes)
    for key in [k for k, v in changes.items() if v is None]:
        del arrays[key]
    np.savez(path, **arrays)


@pytest.mark.parametrize("times, words", [
    ([1.0, np.nan, 3.0], "finite and strictly increasing"),
    ([1.0, 3.0, 2.0], "finite and strictly increasing"),
    ([1.0, 2.0, 2.0], "finite and strictly increasing"),
])
def test_load_refuses_sample_times_that_are_not_finite_and_increasing(tmp_path, times, words):
    ctx = shop_ctx()
    s = ctx.samples["sit_reach"]
    n = len(times)
    short = ClipSamples("sit_reach", np.arange(1.0, 1.0 + n), {"hand.R": s.groups["hand.R"][:n]},
                        {}, {})
    save_scene(dataclasses.replace(ctx, samples={"sit_reach": short}), tmp_path / "scene")
    path = tmp_path / "scene" / "samples" / "sit_reach.npz"
    _rewrite(path, times=np.array(times))
    with pytest.raises(ManifestError, match=words):
        load_scene(tmp_path / "scene")
    _rewrite(path, times=np.zeros(0), **{"g:hand.R": np.zeros((0, 5, 3))})
    with pytest.raises(ManifestError, match="no times"):
        load_scene(tmp_path / "scene")


def test_rigs_are_keyed_like_pieces(tmp_path):
    save_scene(shop_ctx(), tmp_path / "scene")
    back = load_scene(tmp_path / "scene")
    assert sorted(back.rigs) == ["shop:Doll"]
    assert set(back.rig_of_clip.values()) == {"shop:Doll"}
    assert (tmp_path / "scene" / "rigs" / "shop" / "Doll.json").is_file()


def _two_pack_ctx() -> SceneCtx:
    """Two packs, each with a rigged `Doll`, and one clip posing each."""
    ctx = shop_ctx()
    shop = ctx.packs["shop"]
    guest = dataclasses.replace(shop, name="guest")
    doll = ctx.rigs["shop:Doll"]
    clips = {name: Clip(name=name, frames=(1, 10), fps=24.0, authoring="keyed", seat=None,
                        holds=(), targets=(), floor_gear=FloorGear({}), props={},
                        mirror_of=None) for name in ("a", "b")}
    layout = ctx.layouts["shop"]  # its one stand plays a clip these two clips lack
    stand = dataclasses.replace(layout.stands["bench"], clip="a")
    layouts = {"shop": dataclasses.replace(layout, stands={"bench": stand})}
    return SceneCtx(
        style=ctx.style, packs={"shop": shop, "guest": guest}, clips=clips, pieces={},
        samples={}, rigs={"shop:Doll": doll, "guest:Doll": doll},
        rig_of_clip={"a": "shop:Doll", "b": "guest:Doll"}, layouts=layouts)


def test_two_packs_rig_one_name(tmp_path):
    save_scene(_two_pack_ctx(), tmp_path / "scene")
    assert (tmp_path / "scene" / "rigs" / "shop" / "Doll.json").is_file()
    assert (tmp_path / "scene" / "rigs" / "guest" / "Doll.json").is_file()
    back = load_scene(tmp_path / "scene")
    assert back.rig_of_clip == {"a": "shop:Doll", "b": "guest:Doll"}
    assert sorted(back.rigs) == ["guest:Doll", "shop:Doll"]
    for clip, key in back.rig_of_clip.items():
        assert rig_spec(back, clip) is back.packs[key.partition(":")[0]].pieces["Doll"].rig


def _flat_scene(tmp_path: Path) -> Path:
    save_scene(shop_ctx(), tmp_path / "scene")
    return tmp_path / "scene"


def test_flat_rig_file_raises(tmp_path):
    scene = _flat_scene(tmp_path)
    (scene / "rigs" / "shop" / "Doll.json").rename(scene / "rigs" / "Doll.json")
    with pytest.raises(ManifestError, match=r"rigs/<pack>/<piece>\.json") as err:
        load_scene(scene)
    assert str(scene / "rigs" / "Doll.json") in str(err.value)


@pytest.mark.parametrize("stray", ["notes.txt", "shop/Doll.json.bak", "shop/deeper/Doll.json"])
def test_stray_file_under_rigs_raises(tmp_path, stray):
    scene = _flat_scene(tmp_path)
    path = scene / "rigs" / stray
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ManifestError, match=r"rigs/<pack>/<piece>\.json") as err:
        load_scene(scene)
    assert str(scene / "rigs" / Path(stray).parts[0]) in str(err.value)


def test_hidden_files_under_rigs_are_ignored(tmp_path):
    scene = _flat_scene(tmp_path)
    (scene / "rigs" / ".DS_Store").write_bytes(b"\x00\x01")
    (scene / "rigs" / "shop" / ".DS_Store").write_bytes(b"\x00\x01")
    (scene / "rigs" / ".hidden").mkdir()
    (scene / "rigs" / ".hidden" / "Doll.json").write_text("{}")
    assert sorted(load_scene(scene).rigs) == ["shop:Doll"]
    (scene / "rigs" / "shop" / "notes.txt").write_text("x")  # a visible stray still raises
    with pytest.raises(ManifestError, match=r"notes\.txt"):
        load_scene(scene)


def test_rig_folder_names_unknown_pack_raises(tmp_path):
    scene = _flat_scene(tmp_path)
    (scene / "rigs" / "shop").rename(scene / "rigs" / "ghost")
    with pytest.raises(ManifestError, match="ghost") as err:
        load_scene(scene)
    assert "pack" in str(err.value)


def test_rig_stem_not_rigged_raises(tmp_path):
    scene = _flat_scene(tmp_path)
    (scene / "rigs" / "shop" / "Doll.json").rename(scene / "rigs" / "shop" / "Bench.json")
    with pytest.raises(ManifestError, match="Bench") as err:
        load_scene(scene)
    assert "rigged piece" in str(err.value)
    (scene / "rigs" / "shop" / "Bench.json").rename(scene / "rigs" / "shop" / "Ghost.json")
    with pytest.raises(ManifestError, match="Ghost"):
        load_scene(scene)


def test_save_refuses_a_rig_key_of_an_unloaded_pack(tmp_path):
    ctx = shop_ctx()
    bad = dataclasses.replace(ctx, rigs={**ctx.rigs, "ghost:Doll": ctx.rigs["shop:Doll"]})
    out = tmp_path / "scene"
    with pytest.raises(ManifestError, match="ghost"):
        save_scene(bad, out)
    assert not out.exists()


@pytest.mark.parametrize("edit, words", [
    (lambda rig: rig["parent"].update({"arm.R": "root"}), "arm.R"),
    (lambda rig: (rig["order"].remove("hand.L"), rig["parent"].pop("hand.L"),
                  rig["rest"].pop("hand.L")), "hand.L"),
])
def test_load_refuses_a_rig_file_that_disagrees_with_the_pack_rig(tmp_path, edit, words):
    save_scene(shop_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "rigs" / "shop" / "Doll.json"
    rig = json.loads(path.read_text())
    edit(rig)
    path.write_text(json.dumps(rig))
    with pytest.raises(ManifestError, match=words) as err:
        load_scene(tmp_path / "scene")
    assert "rig" in str(err.value)


def test_load_refuses_geometry_keyed_by_an_object_name(tmp_path):
    save_scene(shop_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "pieces" / "shop" / "Bench.npz"
    _rewrite(path, **{"Bench/seat_top": plate(0.6, 0.02), "seat_top": None})
    with pytest.raises(ManifestError, match=r"'Bench/seat_top' is not a part name"):
        load_scene(tmp_path / "scene")


def test_load_refuses_geometry_that_lacks_a_surface_part(tmp_path):
    save_scene(shop_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "pieces" / "shop" / "Bench.npz"
    _rewrite(path, seat_top=None)
    with pytest.raises(ManifestError, match=r"hold 'edge_R'.*'seat_top'"):
        load_scene(tmp_path / "scene")


# --- layouts, sources and nodes ------------------------------------------------------------

NODES = [
    GlbNode("Lamp", None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0), None),
    GlbNode("Lamp.base", "Lamp", (0.0, 0.1, 0.0), (0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0),
            "Lamp.base"),
]


def test_scene_round_trip_with_layouts_sources_nodes(tmp_path):
    ctx = dataclasses.replace(
        shop_ctx(), sources={"build_shop.py": "print('shop')\n", "build_doll.py": "x = 1\n"},
        nodes={"shop": NODES})
    assert sorted(ctx.layouts) == ["shop"]
    out = tmp_path / "scene"
    save_scene(ctx, out)
    assert (out / "layouts" / "shop.json").is_file()
    assert (out / "sources" / "build_shop.py").read_text() == "print('shop')\n"
    assert json.loads((out / "nodes" / "shop.json").read_text())["nodes"][1]["parent"] == "Lamp"
    back = load_scene(out)
    assert back.layouts == ctx.layouts
    assert back.sources == ctx.sources
    assert back.nodes == ctx.nodes


def test_write_nodes_puts_the_glbs_own_table_in_the_scene(tmp_path):
    ctx = shop_ctx()
    out = tmp_path / "scene"
    save_scene(ctx, out)
    glb = write_glb(tmp_path / "shop.glb", TREE)
    write_nodes(out, "shop", glb)
    back = load_scene(out)
    assert [n.name for n in back.nodes["shop"]] == ["Lamp", "Lamp.base", "Lamp.stem", "Lamp.plug"]
    assert back.nodes["shop"][1].parent == "Lamp"
    assert back.nodes["shop"][2].scale == (1.0, 2.0, 1.0)


def test_write_nodes_refuses_a_pack_name_that_is_no_file_name(tmp_path):
    with pytest.raises(ManifestError, match="nodes pack"):
        write_nodes(tmp_path, "../shop", write_glb(tmp_path / "shop.glb", TREE))


def test_scene_without_the_optional_folders_loads_empty(tmp_path):
    ctx = dataclasses.replace(shop_ctx(), layouts={})
    ctx = dataclasses.replace(ctx, packs={"shop": _without_connectors(ctx.packs["shop"])})
    out = tmp_path / "scene"
    save_scene(ctx, out)
    back = load_scene(out)
    assert back.layouts == {} and back.sources == {} and back.nodes == {}
    assert not (out / "layouts").exists() and not (out / "sources").exists()


def _without_connectors(pack):
    lamp = dataclasses.replace(pack.pieces["Lamp"], connectors=())
    bench = pack.pieces["Bench"]
    return dataclasses.replace(pack, pieces={**pack.pieces, "Lamp": lamp, "Bench": bench})


def test_scene_ctx_new_fields_default():
    ctx = shop_ctx()
    plain = SceneCtx(style=ctx.style, packs=ctx.packs, clips=ctx.clips, pieces=ctx.pieces,
                     samples=ctx.samples, rigs=ctx.rigs, rig_of_clip=ctx.rig_of_clip)
    assert plain.layouts == {} and plain.sources == {} and plain.nodes == {}
    assert plain.folder == ""


def test_load_scene_names_its_folder(tmp_path):
    save_scene(shop_ctx(), tmp_path / "diner")
    assert load_scene(tmp_path / "diner").folder == "diner"


def test_connector_naming_a_layout_the_scene_lacks_stops_save_and_load(tmp_path):
    ctx = dataclasses.replace(shop_ctx(), layouts={})
    with pytest.raises(ManifestError, match=r"lead_run.*layout 'shop'"):
        save_scene(ctx, tmp_path / "scene")
    good = tmp_path / "good"
    save_scene(shop_ctx(), good)
    (good / "layouts" / "shop.json").unlink()
    (good / "layouts").rmdir()
    with pytest.raises(ManifestError, match=r"lead_run.*layout 'shop'"):
        load_scene(good)


def test_nodes_for_an_unloaded_pack_stop_save_and_load(tmp_path):
    ctx = dataclasses.replace(shop_ctx(), nodes={"other": NODES})
    with pytest.raises(ManifestError, match="other"):
        save_scene(ctx, tmp_path / "a")
    save_scene(shop_ctx(), tmp_path / "b")
    (tmp_path / "b" / "nodes").mkdir()
    (tmp_path / "b" / "nodes" / "other.json").write_text(json.dumps({"nodes": []}))
    with pytest.raises(ManifestError, match=r"other\.json.*not loaded"):
        load_scene(tmp_path / "b")


def test_a_source_that_is_not_a_py_file_stops_save(tmp_path):
    ctx = dataclasses.replace(shop_ctx(), sources={"stray.txt": "x"})
    with pytest.raises(ManifestError, match="stray.txt"):
        save_scene(ctx, tmp_path / "scene")


def test_a_layout_keyed_by_another_name_stops_save(tmp_path):
    ctx = shop_ctx()
    ctx = dataclasses.replace(ctx, layouts={"stage": ctx.layouts["shop"]})
    with pytest.raises(ManifestError, match="stage"):
        save_scene(ctx, tmp_path / "scene")


def test_stand_naming_a_clip_the_scene_lacks_stops_save_and_load(tmp_path):
    ctx = shop_ctx()
    layout = ctx.layouts["shop"]
    stand = dataclasses.replace(layout.stands["bench"], clip="no_such_clip")
    ctx = dataclasses.replace(
        ctx, layouts={"shop": dataclasses.replace(layout, stands={"bench": stand})})
    with pytest.raises(ManifestError, match=r"stand 'bench'.*clip 'no_such_clip'"):
        save_scene(ctx, tmp_path / "a")
    save_scene(shop_ctx(), tmp_path / "b")
    path = tmp_path / "b" / "layouts" / "shop.json"
    path.write_text(path.read_text().replace("sit_reach", "no_such_clip"))
    with pytest.raises(ManifestError, match=r"stand 'bench'.*clip 'no_such_clip'"):
        load_scene(tmp_path / "b")


def test_a_hidden_source_name_stops_save(tmp_path):
    ctx = dataclasses.replace(shop_ctx(), sources={".hidden.py": "x = 1\n"})
    with pytest.raises(ManifestError, match=r"\.hidden\.py"):
        save_scene(ctx, tmp_path / "scene")
    assert not (tmp_path / "scene").exists()


def test_sources_survive_byte_for_byte(tmp_path):
    text = "# café → naïve\r\nx = 1\r\n\ry = 'ü'\n"
    ctx = dataclasses.replace(shop_ctx(), sources={"build_shop.py": text})
    save_scene(ctx, tmp_path / "scene")
    assert (tmp_path / "scene" / "sources" / "build_shop.py").read_bytes() == text.encode("utf-8")
    assert load_scene(tmp_path / "scene").sources == {"build_shop.py": text}


# --- body shells ----------------------------------------------------------------------------

DOLL_GROUPS = {"torso": ("body",), "arm_R": ("arm.R", "forearm.R", "hand.R"),
               "foot_R": ("foot.R",)}


def _shell_arrays(s: int, seed: int = 3):
    """Per part of the Doll: (S, 4, 3) points, two faces over 4 vertices, a label per vertex."""
    rng = np.random.default_rng(seed)
    shells = {p: rng.normal(size=(s, 4, 3)).astype(np.float32) for p in DOLL_GROUPS}
    faces = {p: np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32) for p in DOLL_GROUPS}
    labels = {p: (np.arange(4) % len(g)).astype(np.int32) for p, g in DOLL_GROUPS.items()}
    return shells, faces, labels


def shelled_ctx() -> SceneCtx:
    ctx = shop_ctx()
    s = ctx.samples["sit_reach"]
    shells, faces, labels = _shell_arrays(len(s.times))
    rest, rest_faces, rest_labels = _shell_arrays(1, seed=4)
    return dataclasses.replace(
        ctx, samples={"sit_reach": dataclasses.replace(
            s, shells=shells, faces=faces, labels=labels)},
        rests={"shop:Doll": RigRest(rest, rest_faces, rest_labels, dict(DOLL_GROUPS))})


def test_shells_and_rest_round_trip(tmp_path):
    ctx = shelled_ctx()
    out = tmp_path / "scene"
    save_scene(ctx, out)
    with np.load(out / "samples" / "sit_reach.npz") as npz:
        for part in DOLL_GROUPS:
            assert npz[f"v:{part}"].dtype == np.float32 and npz[f"v:{part}"].ndim == 3
            assert npz[f"f:{part}"].dtype == np.int32 and npz[f"f:{part}"].shape == (2, 3)
            assert npz[f"l:{part}"].dtype == np.int32 and npz[f"l:{part}"].shape == (4,)
    rig_json = json.loads((out / "rigs" / "shop" / "Doll.json").read_text())
    assert rig_json["vertex_groups"]["arm_R"] == ["arm.R", "forearm.R", "hand.R"]
    with np.load(out / "rigs" / "shop" / "Doll.rest.npz") as npz:
        assert sorted(npz.files) == sorted(
            [f"{k}:{p}" for k in "vfl" for p in DOLL_GROUPS])
        assert npz["v:torso"].shape == (1, 4, 3)

    back = load_scene(out)
    got, want = back.samples["sit_reach"], ctx.samples["sit_reach"]
    for name in ("shells", "faces", "labels"):
        _same_arrays(getattr(got, name), getattr(want, name))
    assert got.faces["torso"].dtype == np.int32 and got.labels["torso"].dtype == np.int32
    assert got.shells["torso"].dtype == np.float32
    rest, want_rest = back.rests["shop:Doll"], ctx.rests["shop:Doll"]
    _same_arrays(rest.shells, want_rest.shells)
    _same_arrays(rest.faces, want_rest.faces)
    _same_arrays(rest.labels, want_rest.labels)
    assert rest.vertex_groups == DOLL_GROUPS
    assert sorted(back.rigs) == ["shop:Doll"]  # the rest file is no stray under rigs/


def test_load_refuses_faces_out_of_range(tmp_path):
    save_scene(shelled_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "samples" / "sit_reach.npz"
    _rewrite(path, **{"f:torso": np.array([[0, 1, 4]], dtype=np.int32)})
    with pytest.raises(ManifestError, match=r"sit_reach\.npz.*f:torso.*outside"):
        load_scene(tmp_path / "scene")
    _rewrite(path, **{"f:torso": np.array([[0, 1, 2]], dtype=np.int32)})
    rest = tmp_path / "scene" / "rigs" / "shop" / "Doll.rest.npz"
    _rewrite(rest, **{"f:arm_R": np.array([[0, -1, 2]], dtype=np.int32)})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*f:arm_R.*outside"):
        load_scene(tmp_path / "scene")


def test_load_refuses_label_out_of_range(tmp_path):
    save_scene(shelled_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "samples" / "sit_reach.npz"
    _rewrite(path, **{"l:torso": np.array([0, 0, 1, 0], dtype=np.int32)})  # torso has 1 group
    with pytest.raises(ManifestError, match=r"sit_reach\.npz.*l:torso.*vertex group"):
        load_scene(tmp_path / "scene")
    _rewrite(path, **{"l:torso": np.zeros(4, dtype=np.int32)})
    rest = tmp_path / "scene" / "rigs" / "shop" / "Doll.rest.npz"
    _rewrite(rest, **{"l:foot_R": np.array([0, 0, 0, -1], dtype=np.int32)})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*l:foot_R.*vertex group"):
        load_scene(tmp_path / "scene")


@pytest.mark.parametrize("edit, words", [
    ("v-without-f", "f:torso"),
    ("labels-wrong-length", "l:torso"),
    ("shell-vertex-count", "v:torso"),
    ("rig-has-no-groups", "vertex_groups"),
    ("no-rest-file", "Doll.rest.npz"),
    ("stray-rest-file", "Ghost.rest.npz"),
])
def test_load_refuses_inconsistent_shells(tmp_path, edit, words):
    save_scene(shelled_ctx(), tmp_path / "scene")
    scene = tmp_path / "scene"
    samples = scene / "samples" / "sit_reach.npz"
    rig_path = scene / "rigs" / "shop" / "Doll.json"
    if edit == "v-without-f":
        _rewrite(samples, **{"f:torso": None})
    elif edit == "labels-wrong-length":
        _rewrite(samples, **{"l:torso": np.zeros(3, dtype=np.int32)})
    elif edit == "shell-vertex-count":
        _rewrite(samples, **{"v:torso": np.zeros((97, 5, 3), dtype=np.float32)})
    elif edit == "rig-has-no-groups":
        rig = json.loads(rig_path.read_text())
        del rig["vertex_groups"]
        rig_path.write_text(json.dumps(rig))
    elif edit == "no-rest-file":
        (scene / "rigs" / "shop" / "Doll.rest.npz").unlink()
    else:
        folder = scene / "rigs" / "shop"
        shutil.copy(folder / "Doll.rest.npz", folder / "Ghost.rest.npz")
    with pytest.raises(ManifestError, match=words.replace(".", r"\.")):
        load_scene(scene)


def test_save_refuses_shells_the_load_would_refuse(tmp_path):
    ctx = shelled_ctx()
    s = ctx.samples["sit_reach"]
    bad_labels = dict(s.labels, torso=np.array([0, 3, 0, 0], dtype=np.int32))
    bad_s = dataclasses.replace(s, labels=bad_labels)
    bad = dataclasses.replace(ctx, samples={"sit_reach": bad_s})
    out = tmp_path / "scene"
    with pytest.raises(ManifestError, match="l:torso"):
        save_scene(bad, out)
    assert not out.exists()
    stray = dataclasses.replace(ctx, rests={"shop:Ghost": ctx.rests["shop:Doll"]})
    with pytest.raises(ManifestError, match="shop:Ghost"):
        save_scene(stray, out)
    assert not out.exists()


def test_scene_ctx_shell_fields_default():
    ctx = shop_ctx()
    assert ctx.rests == {}
    s = ctx.samples["sit_reach"]
    assert s.shells == {} and s.faces == {} and s.labels == {}


def _doll_samples(ctx, **changes):
    s = ctx.samples["sit_reach"]
    return dataclasses.replace(ctx, samples={"sit_reach": dataclasses.replace(s, **changes)})


def test_clip_without_shells_on_a_rig_with_rest_shells_is_refused(tmp_path):
    ctx = _doll_samples(shelled_ctx(), shells={}, faces={}, labels={})
    with pytest.raises(ManifestError, match=r"sit_reach\.npz.*arm_R.*foot_R.*torso"):
        save_scene(ctx, tmp_path / "a")
    save_scene(shelled_ctx(), tmp_path / "b")
    path = tmp_path / "b" / "samples" / "sit_reach.npz"
    with np.load(path) as npz:
        kept = {k: npz[k] for k in npz.files if k[:2] not in ("v:", "f:", "l:")}
    np.savez(path, **kept)
    with pytest.raises(ManifestError, match=r"sit_reach\.npz.*arm_R"):
        load_scene(tmp_path / "b")


def test_clip_shell_parts_must_equal_the_rests_parts(tmp_path):
    s = shelled_ctx().samples["sit_reach"]
    drop = {"shells": {k: v for k, v in s.shells.items() if k != "foot_R"},
            "faces": {k: v for k, v in s.faces.items() if k != "foot_R"},
            "labels": {k: v for k, v in s.labels.items() if k != "foot_R"}}
    with pytest.raises(ManifestError, match=r"sit_reach\.npz.*foot_R"):
        save_scene(_doll_samples(shelled_ctx(), **drop), tmp_path / "a")
    extra = {"shells": {**s.shells, "ghost": s.shells["torso"]},
             "faces": {**s.faces, "ghost": s.faces["torso"]},
             "labels": {**s.labels, "ghost": s.labels["torso"]}}
    with pytest.raises(ManifestError, match=r"sit_reach\.npz.*ghost"):
        save_scene(_doll_samples(shelled_ctx(), **extra), tmp_path / "b")


@pytest.mark.parametrize("which", ["faces", "labels"])
def test_clip_faces_and_labels_must_equal_the_rests(tmp_path, which):
    ctx = shelled_ctx()
    s = ctx.samples["sit_reach"]
    if which == "faces":
        other = {**s.faces, "torso": np.array([[0, 1, 3], [1, 2, 3]], dtype=np.int32)}
    else:
        other = {**s.labels, "arm_R": np.array([0, 1, 2, 2], dtype=np.int32)}
    part = "torso" if which == "faces" else "arm_R"
    with pytest.raises(ManifestError, match=rf"sit_reach\.npz.*{which[0]}:{part}.*rest"):
        save_scene(_doll_samples(ctx, **{which: other}), tmp_path / "a")
    save_scene(ctx, tmp_path / "b")
    key = f"{which[0]}:{part}"
    _rewrite(tmp_path / "b" / "samples" / "sit_reach.npz", **{key: other[part]})
    with pytest.raises(ManifestError, match=rf"sit_reach\.npz.*{key}.*rest"):
        load_scene(tmp_path / "b")


def test_rest_parts_must_equal_the_pieces_parts(tmp_path):
    ctx = shelled_ctx()
    rest = ctx.rests["shop:Doll"]
    fewer = RigRest(
        {k: v for k, v in rest.shells.items() if k != "foot_R"},
        {k: v for k, v in rest.faces.items() if k != "foot_R"},
        {k: v for k, v in rest.labels.items() if k != "foot_R"},
        {k: v for k, v in rest.vertex_groups.items() if k != "foot_R"})
    s = ctx.samples["sit_reach"]
    short = dataclasses.replace(
        s, shells={k: v for k, v in s.shells.items() if k != "foot_R"},
        faces={k: v for k, v in s.faces.items() if k != "foot_R"},
        labels={k: v for k, v in s.labels.items() if k != "foot_R"})
    bad = dataclasses.replace(ctx, rests={"shop:Doll": fewer}, samples={"sit_reach": short})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*foot_R"):
        save_scene(bad, tmp_path / "a")
    assert not (tmp_path / "a").exists()
    ghost = RigRest(
        {**rest.shells, "ghost": rest.shells["torso"]},
        {**rest.faces, "ghost": rest.faces["torso"]},
        {**rest.labels, "ghost": rest.labels["torso"]},
        {**rest.vertex_groups, "ghost": rest.vertex_groups["torso"]})
    bad = dataclasses.replace(ctx, rests={"shop:Doll": ghost})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*ghost"):
        save_scene(bad, tmp_path / "b")


def _weights(seed: int = 5):
    """Per part of the Doll: (4, 2) weights and (4, 2) indices into the part's groups. The
    second slot of every vertex is padding (weight 0, index 0) on the one-group torso."""
    rng = np.random.default_rng(seed)
    weights, indices = {}, {}
    for part, groups in DOLL_GROUPS.items():
        k = min(2, len(groups))
        w = np.zeros((4, 2), dtype=np.float32)
        j = np.zeros((4, 2), dtype=np.int32)
        w[:, :k] = rng.uniform(0.1, 0.5, size=(4, k))
        j[:, :k] = np.arange(k)
        weights[part], indices[part] = w, j
    return weights, indices


def weighted_ctx() -> SceneCtx:
    ctx = shelled_ctx()
    weights, indices = _weights()
    rest = dataclasses.replace(ctx.rests["shop:Doll"], weights=weights, weight_groups=indices)
    return dataclasses.replace(ctx, rests={"shop:Doll": rest})


def test_rest_weights_round_trip(tmp_path):
    ctx = weighted_ctx()
    out = tmp_path / "scene"
    save_scene(ctx, out)
    with np.load(out / "rigs" / "shop" / "Doll.rest.npz") as npz:
        assert sorted(npz.files) == sorted(
            [f"{k}:{p}" for k in "vflwj" for p in DOLL_GROUPS])
        assert npz["w:arm_R"].dtype == np.float32 and npz["w:arm_R"].shape == (4, 2)
        assert npz["j:arm_R"].dtype == np.int32 and npz["j:arm_R"].shape == (4, 2)
    rest, want = load_scene(out).rests["shop:Doll"], ctx.rests["shop:Doll"]
    _same_arrays(rest.weights, want.weights)
    _same_arrays(rest.weight_groups, want.weight_groups)
    assert rest.weights["torso"].dtype == np.float32
    assert rest.weight_groups["torso"].dtype == np.int32


def test_rest_without_weights_loads(tmp_path):
    save_scene(shelled_ctx(), tmp_path / "scene")  # the layout of every scene on disk today
    with np.load(tmp_path / "scene" / "rigs" / "shop" / "Doll.rest.npz") as npz:
        assert not [k for k in npz.files if k[:2] in ("w:", "j:")]
    rest = load_scene(tmp_path / "scene").rests["shop:Doll"]
    assert rest.weights == {} and rest.weight_groups == {}


def test_weight_index_out_of_range_raises(tmp_path):
    save_scene(weighted_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "rigs" / "shop" / "Doll.rest.npz"
    bad = np.array([[0, 0], [0, 0], [0, 0], [0, 3]], dtype=np.int32)  # arm_R has 3 groups
    _rewrite(path, **{"j:arm_R": bad})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*j:arm_R.*outside"):
        load_scene(tmp_path / "scene")
    ctx = weighted_ctx()
    rest = ctx.rests["shop:Doll"]
    worse = dataclasses.replace(rest, weight_groups={**rest.weight_groups, "arm_R": bad})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*j:arm_R.*outside"):
        save_scene(dataclasses.replace(ctx, rests={"shop:Doll": worse}), tmp_path / "b")


@pytest.mark.parametrize("changes, words", [
    pytest.param({"j:torso": None}, "j:torso.*missing", id="weights-without-indices"),
    pytest.param({"w:torso": np.zeros((3, 2), np.float32)}, "w:torso.*shape", id="short-weights"),
    pytest.param({"w:torso": np.full((4, 2), -0.1, np.float32)}, "w:torso.*finite",
                 id="negative-weight"),
    pytest.param({"w:ghost": np.zeros((4, 1), np.float32),
                  "j:ghost": np.zeros((4, 1), np.int32)}, "ghost", id="weights-for-no-part"),
    pytest.param({"w:foot_R": None, "j:foot_R": None}, "foot_R", id="weights-for-some-parts"),
])
def test_malformed_rest_weights_raise(tmp_path, changes, words):
    save_scene(weighted_ctx(), tmp_path / "scene")
    _rewrite(tmp_path / "scene" / "rigs" / "shop" / "Doll.rest.npz", **changes)
    with pytest.raises(ManifestError, match=rf"Doll\.rest\.npz.*{words}"):
        load_scene(tmp_path / "scene")


def test_unknown_rest_key_still_refused(tmp_path):
    save_scene(weighted_ctx(), tmp_path / "scene")
    _rewrite(tmp_path / "scene" / "rigs" / "shop" / "Doll.rest.npz",
             **{"x:torso": np.zeros((4, 2), np.float32)})
    with pytest.raises(ManifestError, match=r"Doll\.rest\.npz.*unknown rest key 'x:torso'"):
        load_scene(tmp_path / "scene")


def test_a_rig_without_rest_shells_stays_legal(tmp_path):
    ctx = shop_ctx()  # the original layout: no rest file, no vertex_groups, clips without shells
    save_scene(ctx, tmp_path / "scene")
    back = load_scene(tmp_path / "scene")
    assert back.rests == {} and back.samples["sit_reach"].shells == {}


def test_skeleton_round_trip_in_scene(tmp_path):
    ctx = branching_ctx()
    save_scene(ctx, tmp_path / "scene")
    assert (tmp_path / "scene" / "pieces" / "shop" / "Lamp.skeleton.npz").is_file()
    back = load_scene(tmp_path / "scene")
    assert list(back.skeletons) == ["shop:Lamp"]
    assert same_skeleton(back.skeletons["shop:Lamp"], ctx.skeletons["shop:Lamp"])
    # The geometry loop did not take the skeleton file for a piece.
    assert sorted(back.pieces) == sorted(ctx.pieces)


def test_a_scene_without_skeletons_has_none(tmp_path):
    save_scene(shop_ctx(), tmp_path / "scene")
    assert load_scene(tmp_path / "scene").skeletons == {}


def test_skeleton_for_unknown_piece_raises(tmp_path):
    ctx = dataclasses.replace(branching_ctx(), skeletons={"shop:Ghost": small_tree()})
    with pytest.raises(ManifestError, match=r"shop:Ghost.*lacks"):
        save_scene(ctx, tmp_path / "a")
    save_scene(branching_ctx(), tmp_path / "b")
    folder = tmp_path / "b" / "pieces" / "shop"
    (folder / "Lamp.skeleton.npz").rename(folder / "Ghost.skeleton.npz")
    with pytest.raises(ManifestError, match=r"Ghost.*lacks"):
        load_scene(tmp_path / "b")


def test_skeleton_for_assembly_raises(tmp_path):
    ctx = dataclasses.replace(shop_ctx(), skeletons={"shop:Bench": small_tree()})
    with pytest.raises(ManifestError, match=r"shop:Bench.*assembly"):
        save_scene(ctx, tmp_path / "a")
    save_scene(branching_ctx(), tmp_path / "b")
    folder = tmp_path / "b" / "pieces" / "shop"
    (folder / "Lamp.skeleton.npz").rename(folder / "Bench.skeleton.npz")
    with pytest.raises(ManifestError, match=r"Bench.*assembly"):
        load_scene(tmp_path / "b")
    assert not (tmp_path / "a").exists()


def test_skeleton_under_the_wrong_piece_name_raises(tmp_path):
    """A skeleton whose info names another piece than its file is refused on load, not left to
    skeleton-match."""
    save_scene(branching_ctx(), tmp_path / "scene")
    path = tmp_path / "scene" / "pieces" / "shop" / "Lamp.skeleton.npz"
    sk = load_skeleton(path)
    save_skeleton(dataclasses.replace(sk, info={**sk.info, "piece": "Stool"}), path)
    with pytest.raises(ManifestError, match=r"Lamp\.skeleton\.npz.*piece 'Lamp'.*piece 'Stool'"):
        load_scene(tmp_path / "scene")
    save_skeleton(dataclasses.replace(sk, info={k: v for k, v in sk.info.items() if k != "piece"}),
                  path)
    with pytest.raises(ManifestError, match=r"piece 'Lamp'.*piece None"):
        load_scene(tmp_path / "scene")


def test_skeleton_keys_must_be_piece_keys(tmp_path):
    ctx = dataclasses.replace(branching_ctx(), skeletons={"Lamp": small_tree()})
    with pytest.raises(ManifestError, match=r"skeletons\['Lamp'\]"):
        save_scene(ctx, tmp_path / "a")


# --- stage_manifests ------------------------------------------------------------------------


def _fixture_files() -> dict:
    return {"style": FIX / "shop.style.json", "clips": FIX / "shop.clips.json",
            "packs": [FIX / "shop.pack.json"]}


def _files_under(folder: Path) -> list[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


def test_stage_manifests_lays_out_the_folder(tmp_path):
    ctx = shop_ctx()
    out = tmp_path / "scene"
    stage_manifests(out, **_fixture_files())
    assert _files_under(out) == ["clips.json", "packs/shop.pack.json", "style.json"]
    assert (out / "style.json").read_bytes() == (FIX / "shop.style.json").read_bytes()
    assert (out / "clips.json").read_bytes() == (FIX / "shop.clips.json").read_bytes()
    assert (out / "packs" / "shop.pack.json").read_bytes() == (FIX / "shop.pack.json").read_bytes()
    _write_data(ctx, out)  # an exporter fills the rest
    back = load_scene(out)
    assert back.packs == ctx.packs and back.clips == ctx.clips and back.style == ctx.style
    assert sorted(back.pieces) == sorted(ctx.pieces)
    assert sorted(back.samples) == ["sit_reach"]


def test_stage_manifests_refuses_a_bad_pack(tmp_path):
    bad = tmp_path / "bad.pack.json"
    bad.write_text("{not json")
    out = tmp_path / "scene"
    with pytest.raises(ManifestError, match="bad.pack.json"):
        stage_manifests(out, style=FIX / "shop.style.json", clips=FIX / "shop.clips.json",
                        packs=[FIX / "shop.pack.json", bad])
    assert not out.exists() or _files_under(out) == []


def test_stage_manifests_refuses_a_used_folder(tmp_path):
    out = tmp_path / "scene"
    stage_manifests(out, **_fixture_files())
    before = {name: (out / name).read_bytes() for name in _files_under(out)}
    with pytest.raises(ManifestError, match="style.json"):
        stage_manifests(out, **_fixture_files())
    assert {name: (out / name).read_bytes() for name in _files_under(out)} == before
    other = tmp_path / "other"
    (other / "packs").mkdir(parents=True)
    (other / "packs" / "shop.pack.json").write_text("old")
    with pytest.raises(ManifestError, match="shop.pack.json"):
        stage_manifests(other, **_fixture_files())
    assert _files_under(other) == ["packs/shop.pack.json"]  # nothing else was written


def test_stage_manifests_refuses_two_packs_of_one_name(tmp_path):
    out = tmp_path / "scene"
    twin = tmp_path / "twin.pack.json"
    twin.write_bytes((FIX / "shop.pack.json").read_bytes())
    with pytest.raises(ManifestError, match="shop"):
        stage_manifests(out, style=FIX / "shop.style.json", clips=FIX / "shop.clips.json",
                        packs=[FIX / "shop.pack.json", twin])
    assert not out.exists() or _files_under(out) == []


# --- what a clip asks of its rig -----------------------------------------------------------


def _with_doll_rig(ctx: SceneCtx, **changes) -> SceneCtx:
    """The scene with the Doll's rig spec changed (the rig file keeps its bones)."""
    pack = ctx.packs["shop"]
    doll = pack.pieces["Doll"]
    pieces = {**pack.pieces, "Doll": dataclasses.replace(
        doll, rig=dataclasses.replace(doll.rig, **changes))}
    return dataclasses.replace(ctx, packs={"shop": dataclasses.replace(pack, pieces=pieces)})


def _standing_ctx(*lifts: Lift) -> SceneCtx:
    """The shop scene with one standing clip, `walk`, that the Doll's rig poses."""
    ctx = shop_ctx()
    walk = dataclasses.replace(ctx.clips["sit_idle"], name="walk", seat=None, holds=(),
                               floor_gear=FloorGear({}), posture="stand", lifts=lifts)
    return dataclasses.replace(ctx, clips={**ctx.clips, "walk": walk},
                               rig_of_clip={**ctx.rig_of_clip, "walk": "shop:Doll"})


def test_lift_naming_a_support_loads(tmp_path):
    save_scene(_standing_ctx(Lift("foot.R", ((5, 9),), "a step")), tmp_path / "scene")
    assert load_scene(tmp_path / "scene").clips["walk"].lifts[0].support == "foot.R"


def test_lift_names_a_non_support_raises(tmp_path):
    ctx = _standing_ctx(Lift("hand.R", ((5, 9),), "a wave"))  # a contact group, not a support
    with pytest.raises(ManifestError, match=r"walk.*lifts\[0\].*'hand.R'.*not a support"):
        save_scene(ctx, tmp_path / "a")
    save_scene(_standing_ctx(Lift("foot.R", ((5, 9),), "a step")), tmp_path / "b")
    path = tmp_path / "b" / "clips.json"
    data = json.loads(path.read_text())
    data["clips"]["walk"]["lifts"][0]["support"] = "hand.R"
    path.write_text(json.dumps(data))
    with pytest.raises(ManifestError, match=r"walk.*lifts\[0\].*'hand.R'.*not a support"):
        load_scene(tmp_path / "b")


def test_a_lift_on_a_rig_with_no_supports_raises(tmp_path):
    ctx = _with_doll_rig(_standing_ctx(Lift("foot.R", ((5, 9),), "a step")), supports=())
    with pytest.raises(ManifestError, match=r"walk.*not a support"):
        save_scene(ctx, tmp_path / "scene")


def test_mirror_of_on_a_rig_without_mirror_raises(tmp_path):
    ctx = shop_ctx()
    ctx = dataclasses.replace(ctx, rig_of_clip={**ctx.rig_of_clip, "sit_idle_L": "shop:Doll"})
    save_scene(ctx, tmp_path / "ok")  # the Doll's rig has a mirror map
    bare = _with_doll_rig(ctx, mirror=None)
    with pytest.raises(ManifestError, match=r"sit_idle_L.*mirror_of 'sit_idle'.*no mirror map"):
        save_scene(bare, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_seat_on_a_rig_without_sit_raises(tmp_path):
    bare = _with_doll_rig(shop_ctx(), sit=None)
    with pytest.raises(ManifestError, match=r"clip 'sit_\w+': names a seat.*no sit"):
        save_scene(bare, tmp_path / "scene")
    assert not (tmp_path / "scene").exists()


def test_a_rig_without_mirror_or_sit_loads_for_clips_that_need_neither(tmp_path):
    ctx = _with_doll_rig(_standing_ctx(), mirror=None, sit=None)
    seated = ("sit_reach", "sit_idle")
    ctx = dataclasses.replace(
        ctx, clips={k: v for k, v in ctx.clips.items() if k not in (*seated, "sit_idle_L")},
        samples={}, rig_of_clip={"walk": "shop:Doll"},
        layouts={"shop": dataclasses.replace(ctx.layouts["shop"], stands={})})  # no stand plays
    # a seated clip, and this scene has none
    save_scene(ctx, tmp_path / "scene")
    assert list(load_scene(tmp_path / "scene").clips) == ["walk"]


def test_an_older_scene_loads_partial(tmp_path):
    """A scene written before the crease angle and clip posture were recorded has neither. It must load;
    the checks that read those fields then report `partial`, never fail."""
    ctx = shop_ctx()
    dials = {k: v for k, v in ctx.style.dials.items() if k != "edges"}
    style = Style(ctx.style.units_per_metre, ctx.style.page_camera, dials)
    old = dataclasses.replace(ctx, style=style, clips={
        k: dataclasses.replace(c, posture=None, lifts=()) for k, c in ctx.clips.items()})
    save_scene(old, tmp_path / "scene")
    raw = json.loads((tmp_path / "scene" / "clips.json").read_text())
    assert all("posture" not in c for c in raw["clips"].values())
    back = load_scene(tmp_path / "scene")
    assert back.style.crease_deg is None
    assert all(c.posture is None and c.lifts == () for c in back.clips.values())
    # Clip hygiene's stand rule reads posture: each old clip is unmeasured there, so the run is
    # partial, never failed, whatever else the clips do.
    verdict = run_gate([ClipHygiene()], back, None)
    assert verdict.status == "partial"
    lines = [x for x in verdict.coverage if isinstance(x, Unmeasured) and "no posture" in x]
    assert {x.rsplit(": ", 1)[1] for x in lines} == {f"{name} stand" for name in back.clips
                                                      if name in back.samples}
    assert lines
    # Near-crease reads the crease angle: with none it is unmeasured too, so the run is partial.
    verdict = run_gate([NearCrease()], back, None)
    assert verdict.status == "partial"
    assert any(isinstance(x, Unmeasured) and "crease_deg" in x for x in verdict.coverage)


# --- the rules folder -----------------------------------------------------------------------

TINY_BUDS = Path(__file__).parent / "grow" / "fixtures" / "tiny-buds.rules.json"


def rules_ctx() -> SceneCtx:
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    return dataclasses.replace(shop_ctx(), rules={TINY_BUDS.name: rules})


def test_rules_round_trip(tmp_path):
    ctx = rules_ctx()
    save_scene(ctx, tmp_path / "scene")
    copied = tmp_path / "scene" / "rules" / TINY_BUDS.name
    assert copied.read_bytes() == TINY_BUDS.read_bytes()
    back = load_scene(tmp_path / "scene")
    assert list(back.rules) == [TINY_BUDS.name]
    assert back.rules[TINY_BUDS.name].sha256 == ctx.rules[TINY_BUDS.name].sha256
    assert back.rules[TINY_BUDS.name].kit_sha256 == ctx.rules[TINY_BUDS.name].kit_sha256


def test_rules_folder_is_optional(tmp_path):
    save_scene(shop_ctx(), tmp_path / "scene")
    assert not (tmp_path / "scene" / "rules").exists()
    assert load_scene(tmp_path / "scene").rules == {}
    assert shop_ctx().rules == {}


def test_bad_rules_file_names_the_file(tmp_path):
    save_scene(rules_ctx(), tmp_path / "scene")
    bad = tmp_path / "scene" / "rules" / TINY_BUDS.name
    bad.write_text("{not json")
    with pytest.raises(ManifestError, match=r"tiny-buds\.rules\.json"):
        load_scene(tmp_path / "scene")


def test_rules_keys_must_be_rules_file_names(tmp_path):
    rules = load_rules(TINY_BUDS, kit=KIT_RULES)
    ctx = dataclasses.replace(shop_ctx(), rules={"oak": rules})
    with pytest.raises(ManifestError, match=r"rules\['oak'\]"):
        save_scene(ctx, tmp_path / "scene")


def test_stage_manifests_copies_rules(tmp_path):
    out = tmp_path / "scene"
    stage_manifests(out, **_fixture_files(), rules=[TINY_BUDS])
    assert _files_under(out) == ["clips.json", "packs/shop.pack.json",
                                 f"rules/{TINY_BUDS.name}", "style.json"]
    assert (out / "rules" / TINY_BUDS.name).read_bytes() == TINY_BUDS.read_bytes()
    _write_data(shop_ctx(), out)
    assert list(load_scene(out).rules) == [TINY_BUDS.name]


def test_stage_manifests_refuses_a_bad_rules_file(tmp_path):
    bad = tmp_path / "bad.rules.json"
    bad.write_text("{not json")
    out = tmp_path / "scene"
    with pytest.raises(ManifestError, match=r"bad\.rules\.json"):
        stage_manifests(out, **_fixture_files(), rules=[bad])
    assert not out.exists() or _files_under(out) == []
