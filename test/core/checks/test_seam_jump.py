"""Check seam-jump: light does not step across a lightmap seam beyond what the bend explains.

Item: `seam <part> jump`. The scenes are `bake_scenes.seamed`, a flat surface cut into one island
per triangle with a picture made from a light field that is continuous in the world, and
`bake_scenes.bent_strip`, four flat facets round an arc, each its own island, lit by Lambert's law.
"""

import dataclasses

import numpy as np
import pytest

from model_kit.core import gate
from model_kit.core.checks import seam_jump
from model_kit.core.checks.seam_jump import SeamJump
from model_kit.core.metagate import Unmeasured, run_gate

from ..scenes import make_ctx as reach_ctx
from . import assembly_scenes, manifest_scenes
from .bake_scenes import CELL, MARGIN, SEAM_SCENES, SIZE, bent_strip, seamed
from .piece_scenes import make_ctx as piece_ctx
from .piece_scenes import post_on_base, tried_and_caught

CHECK = SeamJump()
ITEMS = {"slab": "seam slab/body jump", "shelf": "seam shelf/body jump",
         "strip": "seam strip/body jump"}
STEPPED = {"slab": lambda: seamed(1, "slab", step=0.3), "shelf": lambda: seamed(2, "shelf", step=0.3),
           "strip": lambda: bent_strip(step=0.5)}


def test_registered_before_coverage_and_cites_its_principle():
    assert gate.CHECKS["seam-jump"] is SeamJump
    assert list(gate.CHECKS)[-1] == "coverage"
    assert CHECK.principle.endswith(".") and ". " not in CHECK.principle
    assert "Lambert" in CHECK.source


@pytest.mark.parametrize("name", SEAM_SCENES)
def test_smooth_seam_quiet(name):
    """The light is continuous in 3D across every flat seam, and steps at each bend of the strip
    only as Lambert's law says: no finding, and the run passes."""
    ctx = SEAM_SCENES[name]()
    assert CHECK.run(ctx) == []
    assert run_gate([SeamJump()], ctx, None).status == "passed"


@pytest.mark.parametrize("name", SEAM_SCENES)
def test_seam_jump_found(name):
    found = CHECK.run(STEPPED[name]())
    assert [f.item for f in found] == [ITEMS[name]]
    assert found[0].check == "seam-jump" and "lamp" in found[0].detail


def test_the_strip_steps_at_every_bend_and_is_judged_there():
    """Every seam of the strip is at a bend (no two facets share a normal), the light really
    steps there, and every seam is read: the bend allowance, not a skip, keeps the run quiet."""
    ctx = bent_strip(grain=False)
    reads = seam_jump._reads(ctx.bake, "strip/body")
    assert len(reads.seams) == 3 and (reads.bends > 0.1).all() and reads.read.all()
    sides = seam_jump._sides(ctx.bake.layers["lamp"].pixels, reads)
    steps = seam_jump._steps(sides)
    assert (steps > 1e-3).all()  # Lambert's step at each bend
    assert (steps <= seam_jump._allowances(sides, reads.bends)).all()


def test_normal_noise_is_the_float32_step_over_the_shortest_corner_gap():
    """A right triangle with 0.2 mm legs, 0.6 m out: k = 3 corners, e = sqrt(3) / 2 x 2^-23 x 0.6,
    over L = 0.2 mm. A quad's diagonal counts as a corner gap; a face with one point has no
    direction and an unbounded noise."""
    tri = np.array([[[0.6, 0.0, 0.0], [0.6002, 0.0, 0.0], [0.6, 0.0002, 0.0]]])
    want = 3 * np.sqrt(3) / 2 * 2.0 ** -23 * 0.6002 / 0.0002
    assert seam_jump.normal_noise(tri) == pytest.approx([want])
    rhombus = np.array([[[0, 0, 0], [1, 0, 0], [1.5, 0.1, 0], [0.5, 0.1, 0]]], float)
    short_diag = np.hypot(0.5, 0.1)  # shorter than the unit sides
    assert seam_jump.normal_noise(rhombus) == pytest.approx(
        [4 * np.sqrt(3) / 2 * 2.0 ** -23 * 1.5 / short_diag])
    assert seam_jump.normal_noise(np.zeros((1, 3, 3))) == [np.inf]


def test_float32_facets_of_one_plane_are_one_surface():
    """Tiny cells of one tilted plane, 0.8 m from the origin, held in float32 as Blender and the
    .glb hold them: their face normals scatter by float32 noise past the exporter's rounding
    alone, and every pair is one surface once each face's float32 tilt is counted. A real bend of
    1e-2 between neighbours is not."""
    n = np.array([0.2, 0.15, 1.0]) / np.linalg.norm([0.2, 0.15, 1.0])
    u = np.cross([0.3, 1.0, 0.0], n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    step, cells = 2e-4, 30
    grid = np.array([[[0.6, 0.4, 0.3] + (i - cells / 2) * step * u + (j - cells / 2) * step * v
                      for j in range(cells + 1)] for i in range(cells + 1)]).astype(np.float32)
    tris = np.array([(grid[i, j], grid[i + 1, j], grid[i + 1, j + 1])
                     for i in range(cells) for j in range(cells)])
    face = np.cross((tris[:, 1] - tris[:, 0]).astype(np.float64),
                    (tris[:, 2] - tris[:, 0]).astype(np.float64))
    face /= np.linalg.norm(face, axis=1, keepdims=True)
    exported = np.round(face, seam_jump.NORMAL_DIGITS)
    exported /= np.linalg.norm(exported, axis=1, keepdims=True)
    noise = seam_jump.normal_noise(tris)
    pairs = [(a, a + 1) for a in range(len(tris) - 1)]
    assert not all(seam_jump.same_normal(exported[a], exported[b]) for a, b in pairs)
    assert all(seam_jump.same_normal(exported[a], exported[b], noise[a], noise[b])
               for a, b in pairs)
    bent = exported[1] + 1e-2 * u
    assert not seam_jump.same_normal(exported[0], bent / np.linalg.norm(bent), noise[0], noise[1])


def test_a_step_under_the_noise_is_quiet():
    """A step well inside the threshold is the bake's own noise."""
    assert CHECK.run(seamed(1, "slab", step=0.01)) == []


def test_a_steep_continuous_gradient_is_quiet():
    """The light is continuous across the seam but climbs 3 x noise per shipped texel (the steepest
    density still calls converged: its loss is 3/4 of the noise), and every texel carries Gaussian
    noise of the recorded level. Reading each side half a texel inside its own triangle would see
    a step of one texel's climb; extrapolating each side to the edge sees none."""
    noise = 0.02
    per_unit = 3 * noise * (CELL - 2 * MARGIN)  # texels per world unit
    across = np.array([1.0, -1.0]) / np.sqrt(2)  # across the slab's diagonal seam

    def field(pos):
        s = 0.5 + per_unit * (pos[..., 0] * across[0] + pos[..., 1] * across[1])
        return np.stack([s, s, s], axis=-1)
    ctx = seamed(1, "slab", noise=noise, field=field)
    assert CHECK.run(ctx) == []
    assert run_gate([SeamJump()], ctx, None).status == "passed"


def flat_light(level: float):
    """A light field of one level everywhere, per channel."""
    return lambda pos: np.full((*pos.shape[:-1], 3), level)


def test_a_step_the_bend_explains_is_quiet_and_a_larger_one_is_found():
    """The odd triangles tilt to (0, 0.6, 0.8): bend |(0, 0, 1) - (0, 0.6, 0.8)| = 0.632. The
    light is 0.5 on the even side and 0.5 + step on the odd, with no grain. The line is
    3 sigma (0.0095 at noise and error 0.002) plus 0.632 x max(0.5, 0.5 + step): a step of 0.3
    is under its line of 0.515 and quiet; with the even side at 0.1 and a step of 1.0 the line
    is 0.705, and the step is found."""
    kw = {"hard": True, "grain": False, "noise": 0.002, "error": 0.002}
    assert CHECK.run(seamed(2, "shelf", step=0.3, field=flat_light(0.5), **kw)) == []
    found = CHECK.run(seamed(2, "shelf", step=1.0, field=flat_light(0.1), **kw))
    assert [f.item for f in found] == [ITEMS["shelf"]] and "bend 0.632" in found[0].detail


def test_a_flat_seam_gets_no_allowance():
    """The same step of 0.3 on the flat shelf (bend 0) is found: the allowance is the bend's."""
    kw = {"grain": False, "noise": 0.002, "error": 0.002}
    assert [f.item for f in CHECK.run(seamed(2, "shelf", step=0.3, field=flat_light(0.5), **kw))
            ] == [ITEMS["shelf"]]


def test_a_corner_never_steps_past_its_allowance():
    """At a 90 degree corner (bend sqrt(2) = 1.41) the allowance is more than the brighter side,
    so no step of real (non-negative) light passes it: from 0 to 10 is quiet."""
    ctx = seamed(1, "slab", step=10.0, field=flat_light(0.0), grain=False, noise=0.002,
                 error=0.002)
    normals = ctx.bake.normals["slab/body"].copy()
    normals[1] = [1.0, 0.0, 0.0]
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(
        ctx.bake, normals={"slab/body": normals}))
    assert seam_jump._reads(ctx.bake, "slab/body").bends == pytest.approx([np.sqrt(2)])
    assert CHECK.run(ctx) == []


def test_normals_one_rounding_apart_are_one_normal():
    """The exporter rounds each component to 4 decimals and normalises: two exported normals of
    one true normal differ by up to sqrt(3) x 1e-4. A true normal is rounded once, then the same
    normal nudged across each component's rounding boundary: the pair is one normal. A pair two
    rounding units apart is not: turned 4e-4 off, it stays more than sqrt(3) x 1e-4 away after
    both are rounded (each rounding moves a normal by at most sqrt(3) / 2 x 1e-4)."""
    true = np.array([0.30005, 0.40005, 0.0])
    true[2] = np.sqrt(1.0 - true[0] ** 2 - true[1] ** 2)

    def exported(n):
        r = np.round(n, seam_jump.NORMAL_DIGITS)
        return r / np.linalg.norm(r)
    a = exported(true - 1e-9)
    b = exported(true + 1e-9)
    assert not np.array_equal(a, b)  # the rounding split them
    assert seam_jump.same_normal(a, b)
    across = np.cross(true, [0.0, 0.0, 1.0])
    turned = true + 4e-4 * across / np.linalg.norm(across)
    assert not seam_jump.same_normal(exported(true), exported(turned / np.linalg.norm(turned)))


def test_unseen_seam_skipped():
    """The second triangle is collapsed (no lightmap): its edges are not sampled, however far its
    texels are from the first's."""
    assert CHECK.run(seamed(1, "slab", step=0.3, collapse_second=True)) == []


def test_only_limits_the_parts():
    ctx = seamed(1, "slab", step=0.3)
    assert [f.item for f in CHECK.run(ctx, only=["seam slab/body jump"])] == [
        "seam slab/body jump"]
    assert CHECK.run(ctx, only=["seam other/body jump"]) == []


def test_the_threshold_is_three_sigma_of_the_extrapolated_noise_and_the_error():
    """A step of 2.5 sigma passes, one of 3.5 sigma fails. Per shipped texel the noise is
    ns = noise / 2; each side's edge value is `2 a - b`, so its noise is sqrt(5) x ns; each side
    also carries the recorded resolution error e. Both sides are one placement, so sigma is
    sqrt(5 (ns^2 + ns^2) + e^2 + e^2). The step is in all three channels (the channel RMS), and
    the picture holds no grain, so only the threshold is measured."""
    noise, error = 0.04, 0.01
    ns = noise / 2
    sigma = np.sqrt(5 * 2 * ns ** 2 + 2 * error ** 2)
    kw = {"noise": noise, "error": error, "grain": False}
    assert CHECK.run(seamed(1, "slab", step=2.5 * sigma, **kw)) == []
    assert len(CHECK.run(seamed(1, "slab", step=3.5 * sigma, **kw))) == 1


def test_a_step_the_picture_cannot_resolve_is_quiet():
    """The bake's noise is tiny but its recorded resolution error is not: a step 10 x over the
    noise-only line, yet under 3 sigma with the error, is the picture's own resolution and is
    quiet (under-resolved light is lightmap-density's finding); a step over the full line is
    found. Line: 3 x sqrt(5 x 2 x 0.001^2 + 2 x 0.05^2) = 0.212; noise-only: 0.0095."""
    kw = {"noise": 0.002, "error": 0.05, "grain": False}
    assert CHECK.run(seamed(1, "slab", step=0.1, **kw)) == []
    assert [f.item for f in CHECK.run(seamed(1, "slab", step=0.3, **kw))] == [ITEMS["slab"]]


@pytest.mark.parametrize("name", ["slab", "shelf"])
def test_a_half_black_margin_is_not_read(name):
    """The islands sit off the texel grid (0.3, 0.6 px) and each one's margin is half black next
    to it and black beyond, as a plain 2 x 2 shrink of a 1-pixel fill leaves it; the light is
    continuous across every seam. The check reads only texels its side's island covers, so it is
    quiet even at a tight line (noise 0.002, no grain), and every seam is read. The reads at 1/2
    and 1 texel inside each triangle, whatever they touched, found a step on both scenes."""
    ctx = seamed({"slab": 1, "shelf": 2}[name], name, offset=(0.3, 0.6), rim=True, noise=0.002,
                 error=0.001, grain=False)
    assert CHECK.run(ctx) == [] and CHECK.coverage(ctx) == []
    assert run_gate([SeamJump()], ctx, None).status == "passed"


def test_a_layer_with_no_picture_is_skipped():
    """bake-files reports the missing picture; seam-jump adds no finding and no coverage line."""
    ctx = seamed(1, "slab", step=0.3)
    layer = dataclasses.replace(ctx.bake.layers["lamp"], pixels=None)
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers={"lamp": layer}))
    assert CHECK.run(ctx) == [] and CHECK.coverage(ctx) == []


def test_a_placement_with_no_noise_record_is_skipped():
    ctx = seamed(1, "slab", step=0.3)
    layer = dataclasses.replace(ctx.bake.layers["lamp"], noise={})
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers={"lamp": layer}))
    assert CHECK.run(ctx) == []


def test_picture_rows_run_with_v_from_the_top():
    """A texel lit in the picture is the one under the UV of the face you lit: row = v * size,
    column = u * size, no flip."""
    pixels = np.zeros((SIZE, SIZE, 3), np.float32)
    pixels[10, 3] = 1.0  # row 10 (v = 10.5 / SIZE), column 3 (u = 3.5 / SIZE)
    at = np.array([3.5, 10.5])  # (u, v) * size
    assert np.allclose(seam_jump._bilinear(pixels, at[None, :]), [[1.0, 1.0, 1.0]])
    assert np.allclose(seam_jump._bilinear(pixels, at[None, ::-1]), [[0.0, 0.0, 0.0]])


def test_bilinear_blends_and_clamps():
    pixels = np.zeros((4, 4, 3), np.float32)
    pixels[1, 1] = 1.0
    assert np.allclose(seam_jump._bilinear(pixels, np.array([[2.0, 2.0]])), 0.25)
    assert np.allclose(seam_jump._bilinear(pixels, np.array([[-5.0, 99.0]])), 0.0)


@pytest.mark.parametrize("name", SEAM_SCENES)
def test_the_plants_are_caught(name):
    assert tried_and_caught(SeamJump(), SEAM_SCENES[name]()) == [ITEMS[name]]
    assert run_gate([SeamJump()], SEAM_SCENES[name](), None).status == "passed"


def test_a_bake_with_no_seam_has_one_not_applicable_unit():
    ctx = seamed(1, "slab")
    uv = {"slab/body": ctx.bake.uv["slab/body"].copy()}
    uv["slab/body"][1] = uv["slab/body"][0][[0, 2, 1]]  # one island: the shared edge keeps its UV
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, uv=uv))
    units = CHECK.plants(ctx)
    assert [u.item for u in units] == ["seam jump"] and units[0].not_applicable
    verdict = run_gate([SeamJump()], ctx, None)
    assert verdict.status == "passed" and any("not applicable" in c for c in verdict.coverage)


def sliver(ctx):
    """Triangle two shrunk to 0.4 px across: it holds no texel centre. The picture around it is
    one uniform light, so the real run is quiet."""
    pix = ctx.bake.layers["lamp"].pixels.copy()
    pix[38:43, 38:43] = 0.3
    uv = ctx.bake.uv["slab/body"].copy() * SIZE
    c = np.array([40.1, 40.1])
    uv[1] = np.array([c, c + [0.4, 0.4], c + [0.0, 0.4]])
    layer = dataclasses.replace(ctx.bake.layers["lamp"], pixels=pix)
    bake = dataclasses.replace(ctx.bake, layers={"lamp": layer}, uv={"slab/body": uv / SIZE})
    return dataclasses.replace(ctx, bake=bake)


def test_a_side_too_thin_to_read_is_unmeasured():
    """Triangle two is its own island and 0.4 px across: it covers no texel, so it holds no read
    with all 4 taps inside it, at any depth. The seam cannot be read: no finding, even with a large
    step, and one Unmeasured coverage line naming the part, so the run is partial. With no seam to
    plant on, the plant is not applicable, and the gate raises no BlindCheck."""
    ctx = sliver(seamed(1, "slab", grain=False, field=lambda pos: np.full(pos.shape, 0.3)))
    pix = ctx.bake.layers["lamp"].pixels.copy()
    pix[38:43, 38:43] = 0.9  # the sliver's texels: a step no read can see
    layer = dataclasses.replace(ctx.bake.layers["lamp"], pixels=pix)
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers={"lamp": layer}))
    assert CHECK.run(ctx) == []
    lines = CHECK.coverage(ctx)
    assert len(lines) == 1 and isinstance(lines[0], Unmeasured)
    assert "slab/body" in lines[0] and "1 of 1 seam edges" in lines[0]
    units = CHECK.plants(ctx)
    assert [u.item for u in units] == ["seam jump"] and units[0].not_applicable
    assert run_gate([SeamJump()], ctx, None).status == "partial"


def test_an_unreadable_corner_seam_is_not_unmeasured():
    """The same sliver, but at a 90 degree corner (bend 1.41): no light could step past its
    allowance, so no read is needed and no coverage line is written."""
    ctx = sliver(seamed(1, "slab", grain=False, field=lambda pos: np.full(pos.shape, 0.3)))
    normals = ctx.bake.normals["slab/body"].copy()
    normals[1] = [1.0, 0.0, 0.0]
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(
        ctx.bake, normals={"slab/body": normals}))
    reads = seam_jump._reads(ctx.bake, "slab/body")
    assert not reads.read.any() and not reads.need_read.any()
    assert CHECK.coverage(ctx) == []


def test_a_side_read_through_its_own_island_is_measured():
    """Triangle two is a sliver, but it shares its island with a wide triangle beyond it (the UV
    runs on across their shared edge): reads that leave the sliver land on the same surface, so
    the seam is read and a step is found."""
    ctx = seamed(1, "slab", step=0.3, grain=False)
    bake = ctx.bake
    uv = bake.uv["slab/body"] * SIZE
    a, b = uv[1, 0], uv[1, 1]  # the seam edge on side two: (32, 2) to (58, 2)
    sliver_tip = (a + b) / 2 + [0.0, 0.4]
    uv[1] = [a, b, sliver_tip]
    # a wide triangle on the sliver's far edge (b to the tip), in the same island
    pos = bake.positions["slab/body"]
    beyond_pos = np.array([pos[1, 1], pos[1, 2], pos[1, 2] + [0.0, 1.0, 0.0]])
    beyond_uv = np.array([b, sliver_tip, sliver_tip + [0.0, 20.0]])
    bake = dataclasses.replace(
        bake, uv={"slab/body": np.concatenate([uv, beyond_uv[None]]) / SIZE},
        positions={"slab/body": np.concatenate([pos, beyond_pos[None]])},
        normals={"slab/body": np.tile([0.0, 0.0, 1.0], (3, 3, 1))})
    ctx = dataclasses.replace(ctx, bake=bake)
    assert CHECK.coverage(ctx) == []
    assert [f.item for f in CHECK.run(ctx)] == [ITEMS["slab"]]


def test_plants_are_built_from_the_picture_alone(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a plant called the check's own core")
    monkeypatch.setattr(seam_jump, "_jumps", refuse)
    assert len(SeamJump().plants(seamed(1, "slab"))) == 1


def test_a_layer_of_other_size_is_skipped_here():
    """bake-files reports a picture that is not size x size; seam-jump adds nothing."""
    ctx = seamed(1, "slab", step=0.3)
    one = ctx.bake.layers["lamp"]
    layer = dataclasses.replace(one, pixels=one.pixels[:-1, :-1])
    ctx = dataclasses.replace(ctx, bake=dataclasses.replace(ctx.bake, layers={"lamp": layer}))
    assert CHECK.run(ctx) == []


@pytest.mark.parametrize("make", [reach_ctx, assembly_scenes.make_ctx, manifest_scenes.make_ctx,
                                  lambda: piece_ctx(post_on_base())])
def test_no_stage_quiet(make):
    ctx = make()
    assert ctx.bake is None
    assert CHECK.run(ctx) == [] and CHECK.plants(ctx) == [] and CHECK.coverage(ctx) == []
    verdict = run_gate([SeamJump()], ctx, None)
    assert verdict.status == "passed" and verdict.coverage == []
