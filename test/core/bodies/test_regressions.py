"""The regression fixtures: real past defects of the bars, each kept as the state just before its
fix, cut from earlier builds of the bar.

This test loads each fixture and measures the defect's signature, the number a later check reads,
so a fixture that stops holding its defect fails here first. It judges nothing itself: the angles
and distances are the ones measured when the defect was fixed.

The chibi fixtures are `.npz` files of plain arrays in glTF axes and the bar's scene units:
`v:<part>` (V, 3) float32 and `f:<part>` (F, 3) int32 per part, and `units_per_metre`. The idle
fixture keeps the samples format's keys. The guitar fixture is a scene folder.
"""

from pathlib import Path

import numpy as np
import pytest

from model_kit.core import geom
from model_kit.core.scene import load_scene

HERE = Path(__file__).resolve().parents[1] / "fixtures" / "regressions"

# The chibi bar's outline crease angle: an edge whose faces turn more than this draws a line.
# Every chibi defect was measured against it.
CREASE_DEG = 25.0
# Two vertices of a mirrored pair are one point when they lie this close (float32 noise).
SAME_POINT = 1e-6


def _npz(name: str) -> dict[str, np.ndarray]:
    with np.load(HERE / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _chibi(name: str) -> dict[str, np.ndarray]:
    """A chibi fixture, in the bar's units."""
    z = _npz(name)
    assert float(z["units_per_metre"]) == 0.747
    return z


def _mesh(z: dict[str, np.ndarray], part: str) -> tuple[np.ndarray, np.ndarray]:
    v, f = z[f"v:{part}"], z[f"f:{part}"]
    assert v.dtype == np.float32 and v.ndim == 2 and v.shape[1] == 3
    assert f.dtype == np.int32 and f.ndim == 2 and f.shape[1] == 3
    assert geom.is_closed(v[f]), part
    return v.astype(float), f


def _edge_angles(v: np.ndarray, f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each edge two faces share, as (E, 2) vertex indices, and the angle in degrees between
    those faces' normals."""
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    face = np.tile(np.arange(len(f)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, face = edges[order], face[order]
    pair = np.nonzero(np.all(edges[1:] == edges[:-1], axis=1))[0]
    cos = np.clip(np.sum(n[face[pair]] * n[face[pair + 1]], axis=1), -1.0, 1.0)
    return edges[pair], np.degrees(np.arccos(cos))


def _mirror_matches(z: dict[str, np.ndarray], src: str, dst: str) -> tuple[int, int, int]:
    """How many of `src`'s vertices, mirrored across x, land on a vertex of `dst`, and how many
    of its triangles, mirrored, are triangles of `dst` (as vertex sets); and the triangle
    count."""
    vs, fs = _mesh(z, src)
    vd, fd = _mesh(z, dst)
    mirrored = vs * np.array([-1.0, 1.0, 1.0])
    d = np.linalg.norm(mirrored[:, None, :] - vd[None, :, :], axis=2)
    to_dst = np.where(d.min(axis=1) < SAME_POINT, d.argmin(axis=1), -1)
    points = int(np.sum(to_dst >= 0))
    dst_tris = {tuple(t) for t in np.sort(fd, axis=1).tolist()}
    tris = sum(tuple(sorted(t)) in dst_tris for t in to_dst[fs].tolist() if min(t) >= 0)
    return points, tris, len(fs)


def _torso_dash() -> None:
    z = _chibi("torso_dash.npz")
    v, f = _mesh(z, "torso")
    # The ring form's `rings` 20 at 40 segments: 19 rings of vertices and two poles.
    levels = np.unique(np.round(v[:, 1], 5))
    assert (len(levels), len(v), len(f)) == (21, 19 * 40 + 2, 1520)
    edges, angle = _edge_angles(v, f)
    # Rule B's signature: the vertex ring at y 0.3867, the step under the top cap, folds just
    # over the crease on part of the ring only (front and back centre), so its line stops
    # mid-surface.
    y = np.round(v[edges, 1], 5)
    ring = (y[:, 0] == y[:, 1]) & (np.abs(y[:, 0] - 0.38667) < 1e-4)
    over = angle[ring] > CREASE_DEG
    assert ring.sum() == 40 and over.sum() == 8
    assert angle[ring][over].min() == pytest.approx(25.09, abs=0.01)
    assert angle[ring][over].max() == pytest.approx(25.87, abs=0.01)
    assert angle[ring][~over].max() == pytest.approx(24.71, abs=0.01)


def _arm_dash() -> None:
    z = _chibi("arm_dash.npz")
    for part, count, worst in (("arm_L", 11, 31.44), ("arm_R", 10, 30.01)):
        v, f = _mesh(z, part)
        thumb = np.zeros(len(v), dtype=bool)
        thumb[z[f"thumb:{part}"]] = True
        edges, angle = _edge_angles(v, f)
        on_thumb = thumb[edges].all(axis=1) & (angle > CREASE_DEG)
        assert on_thumb.sum() == count, part
        assert angle[on_thumb].max() == pytest.approx(worst, abs=0.01), part


def _shin_speck() -> None:
    z = _chibi("shin_speck.npz")
    # Every left shin vertex has its mirror on the right shin, but only 1966 of 3250 triangles
    # do: the two sides split their quads along other diagonals.
    assert _mirror_matches(z, "shin_L", "shin_R") == (1627, 1966, 3250)
    v, f = _mesh(z, "shin_L")
    edges, angle = _edge_angles(v, f)
    over = angle > CREASE_DEG
    assert over.sum() == 1 and angle.max() == pytest.approx(25.83, abs=0.01)
    assert v[edges[over][0]].mean(axis=0) == pytest.approx([0.132, 0.027, -0.042], abs=0.001)
    v, f = _mesh(z, "shin_R")
    assert _edge_angles(v, f)[1].max() == pytest.approx(23.96, abs=0.01)


def _eye_rims() -> None:
    z = _chibi("eye_rims.npz")
    hv, hf = _mesh(z, "head")
    for part in ("eye_L", "eye_R"):
        v, f = _mesh(z, part)
        edges, angle = _edge_angles(v, f)
        rim = edges[angle > CREASE_DEG]
        inside = geom.inside(hv[hf], v[rim].mean(axis=1))
        # The 32 x 16 lens's rim crease ring weaves across the head: 66 of its 104 edges
        # inside, 38 outside.
        assert (len(rim), int(inside.sum()), int((~inside).sum())) == (104, 66, 38), part
        assert angle.max() == pytest.approx(44.3, abs=0.05), part


def _idle_feet() -> None:
    z = _npz("idle_feet.npz")
    times = z["times"]
    assert len(times) == 145 and times[0] == 1.0 and times[-1] == 73.0
    # The lifts below are scene units: 0.0060 units is about 8.0 mm at 0.747 units per metre.
    # The breath is keyed on the root, so the whole body rises: the root bone's height goes from
    # 0 at frame 1 to 0.0060 at frame 37, and every bone under it with it.
    for bone in ("root", "body"):
        height = z[f"b:{bone}"][:, 1, 3]
        assert times[height.argmax()] == 37.0, bone
        assert height.max() - height[0] == pytest.approx(0.0060, abs=1e-6), bone
    assert z["b:root"][0, 1, 3] == 0.0
    for support in ("ankle.L", "ankle.R"):
        points = z[f"g:{support}"]
        assert points.shape == (145, 1627, 3)
        low = points[:, :, 1].min(axis=1)
        # So the feet lift too: each support's lowest point rises by the same 0.0060 units.
        assert low[0] == pytest.approx(0.001904, abs=1e-6)
        assert times[low.argmax()] == 37.0
        assert low.max() - low.min() == pytest.approx(0.0060, abs=1e-5)


def _strat_before_neck_plate() -> None:
    ctx = load_scene(HERE / "strat_before_neck_plate")
    per_mm = ctx.style.units_per_metre / 1000
    parts = ctx.pieces["strat:Strat"].parts

    def gaps_mm(host: str, part: str) -> np.ndarray:
        """Each distinct vertex of `part`, its distance to `host`'s surface in mm."""
        points = np.unique(parts[part].reshape(-1, 3), axis=0)
        return geom.closest_points(parts[host], points).dist / per_mm

    # Bearing, case 1: the neck plate hangs past the body's heel, with 29 of its 50 mm in the
    # air. Its farthest vertex is 29.12 mm off the body, and 37 of its 56 vertices are more than
    # 1 mm off.
    plate = gaps_mm("Body", "NeckPlate")
    assert plate.max() == pytest.approx(29.12, abs=0.01)
    assert (len(plate), int(np.sum(plate > 1.0))) == (56, 37)
    # Two of its four bolts stand in the air; the other two sit 1.6 mm off the body.
    gap = {b: gaps_mm("Body", b).min()
           for b in ("NeckBolt", "NeckBolt_001", "NeckBolt_002", "NeckBolt_003")}
    assert gap["NeckBolt_002"] == pytest.approx(17.89, abs=0.01)
    assert gap["NeckBolt_003"] == pytest.approx(17.89, abs=0.01)
    assert gap["NeckBolt"] == pytest.approx(1.6, abs=0.01)
    assert gap["NeckBolt_001"] == pytest.approx(1.6, abs=0.01)
    # Case 3: the outer saddles pass the bridge plate's ends along y by 2.8 mm each
    # (Saddle_1 at the top end, Saddle_6 at the bottom); saddles 2 to 5 stay inside.
    bridge = parts["BridgePlate"][..., 1]
    y_lo, y_hi = bridge.min(), bridge.max()
    past = {}
    for i in range(1, 7):
        y = parts[f"Saddle_{i}"][..., 1]
        past[i] = (max(y.max() - y_hi, 0.0) / per_mm, max(y_lo - y.min(), 0.0) / per_mm)
    assert past[1] == pytest.approx((2.8, 0.0), abs=0.01)
    assert past[6] == pytest.approx((0.0, 2.8), abs=0.01)
    assert all(past[i] == (0.0, 0.0) for i in range(2, 6))
    # Case 5: the string-tree wing is built over the two highest strings, String_5 and
    # String_6 (the build's string_pt(4) and string_pt(5), named String_{i + 1}), but floats
    # above them. It comes nearest String_4, at its edge, and touches none.
    wing = {s: gaps_mm(s, "TreeWing").min() for s in ("String_4", "String_5", "String_6")}
    assert wing["String_5"] == pytest.approx(2.83, abs=0.01)
    assert wing["String_6"] == pytest.approx(2.64, abs=0.01)
    assert wing["String_4"] == pytest.approx(1.61, abs=0.01)
    # Markings: 10 fret dots, 9 side dots and the plate logo, every one a solid part.
    piece = ctx.packs["strat"].pieces["Strat"]
    marks = [p for p in piece.parts
             if p.name.startswith(("Dot_", "SideDot_")) or p.name == "PlateLogo"]
    assert len(marks) == 20 and {p.role for p in marks} == {"solid"}


@pytest.mark.parametrize("fixture", [_torso_dash, _arm_dash, _shin_speck,
                                     _eye_rims, _idle_feet,
                                     pytest.param(_strat_before_neck_plate, marks=pytest.mark.held(
                                         "test/core/fixtures/regressions/strat_before_neck_plate"))],
                         ids=lambda f: f.__name__.strip("_"))
def test_regression_fixtures_load(fixture):
    fixture()
