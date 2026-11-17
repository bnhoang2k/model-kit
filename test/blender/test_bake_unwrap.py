"""The bake's place, see and unwrap steps, run for real in Blender on two tiny scenes.

`box_on_slab`: a box placed twice on a slab cut into 0.25 m cells, one camera straight above.
`shelf`: a plank with a UV set, a board without, and a block behind a wall no camera sees past.
`tube`: a thin smooth tube, 5 mm across and 30 cm long, seen all round by three cameras.
`mixed`: a flat sheet (a disk region) beside a smooth closed ball (a fallback region).
`chip`: a flat tilted 6 mm sheet of 0.2 mm flat cells whose own normals carry float noise.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from bake_fixtures import TUBE, make_scene, write_stage

from model_kit import bake
from model_kit.blender import runner
from model_kit.core.bake_data import load_bake, texels
from model_kit.core.checks.seam_jump import NORMAL_DIGITS, _seams
from model_kit.core.glb import glb_mesh_attributes, glb_nodes

pytestmark = pytest.mark.blender


@pytest.fixture(scope="module")
def box_on_slab(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("slab"), "box_on_slab")


@pytest.fixture(scope="module")
def shelf(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("shelf"), "shelf")


@pytest.fixture(scope="module")
def tube(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("tube"), "tube")


@pytest.fixture(scope="module")
def mixed(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("mixed"), "mixed")


@pytest.fixture(scope="module")
def chip(tmp_path_factory) -> Path:
    return make_scene(tmp_path_factory.mktemp("chip"), "chip")


def run(bar: Path, name: str, doll_live: bool = True, **stage) -> tuple[Path, dict]:
    """Bake fixture `name` (over any earlier bake) and return the bake folder and bake.json. The
    slab scene's rigged `doll` is live, unless `doll_live` is false."""
    if name == "box_on_slab" and doll_live:
        stage["live"] = [*stage.get("live", ()), "doll"]
    stage_file = write_stage(bar / "stage.json", name, **stage)
    plan = bake.plan_bake(bar / "scene", stage_file)
    out = bake.run_bake(plan, replace=True, timeout=300)
    return out, json.loads((out / "bake.json").read_text())


def world_tris(glb: Path, node: str) -> np.ndarray:
    """A node's triangles, (n, 3, 3), in world space; the fixtures only translate."""
    table = {n.name: n for n in glb_nodes(glb)}
    shift = np.zeros(3)
    at = table[node]
    while at is not None:
        shift += at.translation
        at = table.get(at.parent) if at.parent else None
    attrs = glb_mesh_attributes(glb, node)
    return attrs["POSITION"][attrs["indices"]].reshape(-1, 3, 3) + shift


def corner_uv(glb: Path, node: str, channel: int) -> np.ndarray:
    """A node's UV set `channel` per triangle corner, (n, 3, 2)."""
    attrs = glb_mesh_attributes(glb, node)
    return attrs[f"TEXCOORD_{channel}"][attrs["indices"]].reshape(-1, 3, 2)


def uv_areas(uv: np.ndarray) -> np.ndarray:
    a, b, c = uv[:, 0], uv[:, 1], uv[:, 2]
    return np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / 2


def normals(tris: np.ndarray) -> np.ndarray:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def under_box(tris: np.ndarray, boxes: tuple[float, ...]) -> np.ndarray:
    """Which triangles' centres lie inside the footprint (x +-0.25 round each box centre,
    z +-0.25)."""
    c = tris.mean(axis=1)
    return np.any([(abs(c[:, 0] - x) < 0.25) & (abs(c[:, 2]) < 0.25) for x in boxes], axis=0)


def pixels(uv: np.ndarray, size: int) -> set[tuple[int, int]]:
    """The pixels of a size x size picture that triangles (n, 3, 2) cover, sampled finely."""
    steps = np.linspace(0, 1, 40)
    out = set()
    for tri in uv:
        for s in steps:
            for t in steps:
                if s + t <= 1:
                    p = tri[0] + s * (tri[1] - tri[0]) + t * (tri[2] - tri[0])
                    out.add((int(p[0] * size), int(p[1] * size)))
    return out


def channel_of(js: dict, node: str) -> int:
    return js["uv_channel"][node]


def test_unseen_faces_have_no_area(box_on_slab):
    out, js = run(box_on_slab, "box_on_slab")
    glb = out / "box_on_slab.glb"
    pack_glb = box_on_slab / "yard.glb"
    slab = world_tris(glb, "slab/body")
    n = normals(slab)
    area = uv_areas(corner_uv(glb, "slab/body", channel_of(js, "slab/body")))
    top = (n[:, 1] > 0.9) & (abs(slab[:, :, 1] - 0.2).max(axis=1) < 1e-4)
    covered = top & under_box(slab, (-0.5, 0.5))
    seen = top & ~covered
    assert covered.sum() == 16 and seen.sum() == 128 - 16  # eight triangles under each box
    assert (area[covered] == 0).all()
    assert (area[n[:, 1] < -0.9] == 0).all()  # the underside
    assert (area[seen] > 0).all()
    for box in ("Box/body", "box_b/body"):  # a box's top is seen, its bottom is not
        tris = world_tris(glb, box)
        bn = normals(tris)
        barea = uv_areas(corner_uv(glb, box, channel_of(js, box)))
        assert (barea[bn[:, 1] > 0.9] > 0).all()
        assert (barea[bn[:, 1] < -0.9] == 0).all()
    # the exported mesh keeps every triangle, seen or not
    for node, pack_node in (("slab/body", "Slab/body"), ("Box/body", "Box/body")):
        assert (len(glb_mesh_attributes(glb, node)["indices"])
                == len(glb_mesh_attributes(pack_glb, pack_node)["indices"]))
    assert js["size"] % 2 == 0 and js["size"] > 0
    assert js["unseen"] == []


def test_live_placement_has_no_lightmap(box_on_slab):
    out, js = run(box_on_slab, "box_on_slab", live=["box_b"])
    glb = out / "box_on_slab.glb"
    attrs = glb_mesh_attributes(glb, "box_b/body")
    assert not [k for k in attrs if k.startswith("TEXCOORD")]
    assert "box_b/body" not in js["uv_channel"]
    assert js["live"] == ["box_b", "doll"]
    assert "TEXCOORD_0" in glb_mesh_attributes(glb, "Box/body")  # the still box has one


def test_live_placement_does_not_occlude(box_on_slab):
    # A live piece moves, so the slab under it may be seen at another beat: it keeps its area.
    out, js = run(box_on_slab, "box_on_slab", live=["box_b"])
    glb = out / "box_on_slab.glb"
    slab = world_tris(glb, "slab/body")
    area = uv_areas(corner_uv(glb, "slab/body", channel_of(js, "slab/body")))
    top = (normals(slab)[:, 1] > 0.9) & (abs(slab[:, :, 1] - 0.2).max(axis=1) < 1e-4)
    assert (area[top & under_box(slab, (0.5,))] > 0).all()
    assert (area[top & under_box(slab, (-0.5,))] == 0).all()


def test_two_placements_get_two_regions(box_on_slab):
    out, js = run(box_on_slab, "box_on_slab")
    glb = out / "box_on_slab.glb"
    regions = []
    for box in ("Box/body", "box_b/body"):
        uv = corner_uv(glb, box, channel_of(js, box))
        uv = uv[uv_areas(uv) > 0]
        regions.append(pixels(uv, 2 * js["size"]))
    assert regions[0] and regions[1]
    assert not regions[0] & regions[1]


def test_existing_uv_is_kept(shelf):
    out, js = run(shelf, "shelf")
    glb = out / "shelf.glb"
    assert js["uv_channel"]["plank/body"] == 1
    assert js["uv_channel"]["board/body"] == 0
    kept = corner_uv(glb, "plank/body", 0)
    original = corner_uv(shelf / "stock.glb", "Plank/body", 0)
    assert kept.tobytes() == original.tobytes()
    assert (uv_areas(corner_uv(glb, "plank/body", 1)) > 0).any()
    assert "TEXCOORD_1" not in glb_mesh_attributes(glb, "board/body")


def test_unseen_piece_listed(shelf):
    out, js = run(shelf, "shelf")
    assert js["unseen"] == ["hidden/body"]
    assert "hidden/body" not in js["uv_channel"]
    attrs = glb_mesh_attributes(out / "shelf.glb", "hidden/body")
    assert not [k for k in attrs if k.startswith("TEXCOORD")]
    assert len(attrs["indices"]) == 36  # the model keeps its faces


def test_density_follows_view(box_on_slab):
    # texels_per_m 16, seen from 1 m: 16 px a metre shipped, 32 at the doubled size.
    out, js = run(box_on_slab, "box_on_slab", texels_per_m=16)
    glb = out / "box_on_slab.glb"
    for node in ("slab/body", "Box/body", "box_b/body"):
        uv = corner_uv(glb, node, channel_of(js, node))
        tris = world_tris(glb, node)
        lit = uv_areas(uv) > 0
        world = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]),
                               axis=1) / 2
        px = uv_areas(uv)[lit].sum() * (2 * js["size"]) ** 2
        assert px / world[lit].sum() == pytest.approx(32 ** 2, rel=0.02)


def test_rigged_placement_is_placed_live_or_still(box_on_slab):
    # The toy figure's armature sits under its root and is not named `toy/<part>`.
    out, js = run(box_on_slab, "box_on_slab")
    names = {n.name for n in glb_nodes(out / "box_on_slab.glb")}
    assert {"doll", "doll/body", "doll/toy_rig"} <= names
    assert "doll/body" not in js["uv_channel"]
    assert not [k for k in glb_mesh_attributes(out / "box_on_slab.glb", "doll/body")
                if k.startswith("TEXCOORD")]
    out, js = run(box_on_slab, "box_on_slab", doll_live=False)  # still: out of view, so unseen
    assert js["unseen"] == ["doll/arm_L", "doll/arm_R", "doll/body"]
    assert "doll" not in js["live"]


def test_placement_named_like_its_piece(box_on_slab):
    # Placement `Box` of piece `Box`, then `box_b` of the same piece: two imports of one name.
    out, js = run(box_on_slab, "box_on_slab")
    for node in ("Box/body", "box_b/body"):
        assert (uv_areas(corner_uv(out / "box_on_slab.glb", node, channel_of(js, node))) > 0).any()


def test_instanced_parts_get_their_own_regions(shelf):
    out, js = run(shelf, "shelf")
    glb = out / "shelf.glb"
    regions = []
    for node in ("twin/left", "twin/right"):
        assert js["uv_channel"][node] == 0  # no stray second set
        assert "TEXCOORD_1" not in glb_mesh_attributes(glb, node)
        uv = corner_uv(glb, node, 0)
        uv = uv[uv_areas(uv) > 0]
        assert len(uv)
        regions.append(pixels(uv, 2 * js["size"]))
    assert not regions[0] & regions[1]


def test_a_face_narrower_than_the_rays_is_still_seen(shelf):
    # `wire` is 4 mm across and sits between two grid rays (spacing about 11 mm there).
    out, js = run(shelf, "shelf")
    assert "wire/body" not in js["unseen"]
    area = uv_areas(corner_uv(out / "shelf.glb", "wire/body", js["uv_channel"]["wire/body"]))
    assert (area > 0).sum() == 2  # its front face, two triangles


def islands(tris: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Each triangle's lightmap island: triangles joined through an edge whose two ends sit at
    one place in the world (to a millionth of the extent) and at one place in the UV (to a
    thousandth of a texel, `uv` in texels) on both sides."""
    step = 1e-6 * float(np.ptp(tris.reshape(-1, 3), axis=0).max())
    keys = np.round(tris / step).astype(np.int64)
    parent = list(range(len(tris)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    edges: dict[tuple, tuple[int, int, int]] = {}
    for t in range(len(tris)):
        for a in range(3):
            b = (a + 1) % 3
            ka, kb = tuple(keys[t, a]), tuple(keys[t, b])
            key = (min(ka, kb), max(ka, kb))
            if key not in edges:
                edges[key] = (t, a, b)
                continue
            s, sa, sb = edges[key]
            if tuple(keys[s, sa]) != ka:
                sa, sb = sb, sa
            if max(np.abs(uv[s, sa] - uv[t, a]).max(), np.abs(uv[s, sb] - uv[t, b]).max()) < 1e-3:
                parent[find(s)] = find(t)
    return np.array([find(t) for t in range(len(tris))])


def narrowest(points: np.ndarray) -> float:
    """The smallest width of a point set (n, 2): over the edges of its convex hull, the least
    reach of the points along the edge's normal (the narrow side of the tightest box)."""
    pts = sorted(map(tuple, np.unique(points, axis=0)))

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    hull: list = []
    for seq in (pts, pts[::-1]):
        half: list = []
        for p in seq:
            while len(half) >= 2 and turn(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        hull += half[:-1]
    hull = np.array(hull)
    widths = []
    for i in range(len(hull)):
        d = hull[(i + 1) % len(hull)] - hull[i]
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        reach = hull @ n
        widths.append(reach.max() - reach.min())
    return float(min(widths))


TUBE_TEXELS = 512  # texels a metre: the tube's 15.6 mm round is 8 texels


def test_a_thin_tube_unrolls_into_one_strip(tube):
    # Peeled like a label off a can: one lengthwise cut, so the side's island is as wide as the
    # whole round. Smart UV Project cuts it into strips a few faces wide.
    out, js = run(tube, "tube", texels_per_m=TUBE_TEXELS)
    glb = out / "tube.glb"
    tris = world_tris(glb, "tube/body")
    uv = corner_uv(glb, "tube/body", channel_of(js, "tube/body")) * js["size"]  # shipped texels
    side = np.abs(normals(tris)[:, 0]) < 0.5
    area = uv_areas(uv)
    assert side.sum() == 2 * TUBE["TUBE_SIDES"] * TUBE["TUBE_RINGS"]
    assert (area[side] > 0).all()  # every face round the tube is seen and has room
    lit = area > 0
    assert len(set(islands(tris[lit], uv[lit]))) == 1
    sides = TUBE["TUBE_SIDES"]
    round_m = sides * 2 * TUBE["TUBE_RADIUS"] * math.sin(math.pi / sides)  # the mesh's own round
    across = narrowest(uv[lit].reshape(-1, 2))
    # The packer may change an island's area by DENSITY_TOL (1e-3), its width by half of that.
    assert across >= round_m * TUBE_TEXELS * (1 - 5e-4), (across, round_m * TUBE_TEXELS)
    # The seam-jump check finds one flat seam edge a ring: the cut, and nothing else (the caps'
    # seams are at a bend).
    seams, _ = _seams(load_bake(tube / "scene"), "tube/body")
    assert sum(s.flat for s in seams) == TUBE["TUBE_RINGS"]


def test_regions_follow_the_exported_normals(chip):
    # Blender's own normals on the chip's tiny cells differ by float noise (float32 corners 0.8 m
    # out make a 0.2 mm cell's normal noisy by up to about 2.4e-4, more than the exporter's
    # 4-decimal rounding); the seam-jump check reads the rounded ones. The bake joins its smooth
    # regions by the check's own rule (`seam_jump.same_normal`: the rounding plus each face's
    # float32 tilt, `normal_noise`), so the flat chip is one region with no lightmap seam, and
    # none the check calls flat.
    out, js = run(chip, "chip", texels_per_m=2048)
    area = uv_areas(corner_uv(out / "chip.glb", "chip/body", channel_of(js, "chip/body")))
    assert (area > 0).all()  # every cell is seen and has room
    seams, _ = _seams(load_bake(chip / "scene"), "chip/body")
    assert sum(s.flat for s in seams) == 0, f"{sum(s.flat for s in seams)} flat seam edges"
    assert len(seams) == 0, f"{len(seams)} seam edges on a flat sheet"


def test_the_exporter_rounds_normals_as_the_check_assumes(tmp_path):
    # The check's rule for one normal rests on the exporter's rounding; bake_render refuses to
    # start when they differ, and this names the two numbers.
    script = tmp_path / "digits.py"
    script.write_text("from io_scene_gltf2.io.com.constants import ROUNDING_DIGIT\n"
                      "from model_kit.blender import runner\n"
                      "print('DIGITS', ROUNDING_DIGIT)\nrunner.done()\n")
    out = runner.run_blender(script, [], timeout=120)
    got = [line.split()[1] for line in out.log.splitlines() if line.startswith("DIGITS ")]
    assert out.ok and got == [str(NORMAL_DIGITS)], out.log[-2000:]


def test_every_island_is_packed_apart(mixed):
    # The sheet unwraps whole; the closed ball falls back to Smart UV Project. Both kinds of island
    # are packed with the margin: no doubled texel centre lies in two islands, no shipped texel
    # holds two, and no empty shipped texel (the 2 rings the bake fills) touches two.
    out, js = run(mixed, "mixed", texels_per_m=64)
    glb, size = out / "mixed.glb", js["size"]
    claims, shipped = [], np.full(size * size, -1, dtype=np.int64)
    twice = 0
    n_islands = 0
    for node, ch in js["uv_channel"].items():
        uv = corner_uv(glb, node, ch)
        lit = uv_areas(uv) > 0
        isl = islands(world_tris(glb, node)[lit], uv[lit] * size)
        for one in np.unique(isl):
            mine = uv[lit][isl == one]
            claims.append(texels(mine, 2 * size))
            ids = texels(mine, 2 * size, edges=True)
            blocks = np.unique((ids // (2 * size)) // 2 * size + (ids % (2 * size)) // 2)
            twice += int((shipped[blocks] >= 0).sum())
            shipped[blocks] = n_islands
            n_islands += 1
    assert n_islands >= 2
    counts = np.bincount(np.concatenate(claims), minlength=4 * size * size)
    assert (counts <= 1).all(), f"{int((counts > 1).sum())} doubled texels in two islands"
    assert twice == 0, f"{twice} shipped texels hold two islands"
    grid = np.pad(shipped.reshape(size, size), 1, constant_values=-1)
    near = np.stack([grid[1 + dy:1 + dy + size, 1 + dx:1 + dx + size]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx])
    lo = np.where(near >= 0, near, n_islands).min(axis=0)
    hi = np.where(near >= 0, near, -1).max(axis=0)
    empty = shipped.reshape(size, size) < 0
    mixed_ring = empty & (hi >= 0) & (lo != hi)
    assert not mixed_ring.any(), f"{int(mixed_ring.sum())} empty texels touch two islands"
