"""Soft forms: the blob and ring meshes, the deforms and the form readers."""

import math
from pathlib import Path

import numpy as np
import pytest

from model_kit.core import geom
from model_kit.core.errors import ManifestError
from model_kit.core.forms import apply_deforms, blob_mesh, decal_grid, ring_mesh
from model_kit.core.manifest.fields import _At
from model_kit.core.manifest.forms import (
    BlobForm,
    Bump,
    Cap,
    DecalForm,
    Flatten,
    MetaForm,
    Push,
    RingForm,
    SeatOn,
    Smooth,
    SoftForm,
    Station,
    read_deform,
    read_form,
    read_seat,
)

AT = _At(Path("pack.json"), "pieces.a.parts[0].form")
UP = np.array([0.0, 1.0, 0.0])


def blob(**kw) -> BlobForm:
    base = {"half": (0.3, 0.5, 0.2), "centre": (0.1, 1.0, -0.2), "power_side": 1.0,
            "power_up": 1.0, "power_down": 1.0, "segments": 16, "rings": 8}
    return BlobForm(**{**base, **kw})


def cosine_us(rings: int) -> list[float]:
    return [(1 - math.cos(math.pi * i / rings)) / 2 for i in range(1, rings)]


def flat_stations(hs=0.3, hd=0.2) -> tuple[Station, ...]:
    return (Station(0.0, hs, hd, 0.0, 0.0, 1.0), Station(1.0, hs, hd, 0.0, 0.0, 1.0))


def ring(**kw) -> RingForm:
    base = {"top": (0.0, 2.0, 0.0), "bottom": (0.0, 0.0, 0.0), "rings": 8, "segments": 16,
            "stations": flat_stations()}
    return RingForm(**{**base, **kw})


def tris(verts, faces) -> np.ndarray:
    out = []
    for f in faces:
        for k in range(1, len(f) - 1):
            out.append([verts[f[0]], verts[f[k]], verts[f[k + 1]]])
    return np.array(out)


def test_blob_counts():
    v, f = blob_mesh(blob(segments=40, rings=20))
    assert (len(v), len(f)) == (762, 800)
    v, f = blob_mesh(blob(segments=32, rings=16))
    assert (len(v), len(f)) == (482, 512)


@pytest.mark.parametrize("power", [0.7, 1.0, 1.3])
def test_blob_box_is_centre_plus_half(power):
    form = blob(power_side=power, power_up=power, power_down=power, segments=40, rings=20)
    v, _ = blob_mesh(form)
    centre, half = np.array(form.centre), np.array(form.half)
    assert np.allclose(v.min(axis=0), centre - half, atol=1e-12, rtol=0)
    assert np.allclose(v.max(axis=0), centre + half, atol=1e-12, rtol=0)


def test_blob_power_below_one_pushes_out():
    form = blob(half=(1.0, 1.0, 1.0), centre=(0.0, 0.0, 0.0), power_side=0.5, power_up=0.5,
                power_down=0.5, segments=40, rings=20)
    v, _ = blob_mesh(form)
    diag = np.ones(3) / math.sqrt(3)
    nearest = v[np.argmax(v @ diag / np.linalg.norm(v, axis=1))]
    assert np.linalg.norm(nearest) > 1.0


def test_ring_counts():
    v, f = ring_mesh(ring(rings=72, segments=56))
    assert (len(v), len(f)) == (3978, 4032)
    v, f = ring_mesh(ring(rings=36, segments=28))
    assert (len(v), len(f)) == (982, 1008)


def test_ring_station_exact():
    rings, segments = 12, 16
    us = cosine_us(rings)
    stations = (Station(0.0, 0.1, 0.1, 0.0, 0.0, 1.0),
                *(Station(u, 0.2 + 0.1 * i, 0.15 + 0.05 * i, 0.0, 0.0, 1.0)
                  for i, u in enumerate(us)),
                Station(1.0, 0.1, 0.1, 0.0, 0.0, 1.0))
    form = ring(rings=rings, segments=segments, stations=stations, blend="smooth")
    v, _ = ring_mesh(form)
    for i in range(rings - 1):
        pts = v[1 + i * segments: 1 + (i + 1) * segments]
        st = stations[i + 1]
        assert np.ptp(pts[:, 0]) == pytest.approx(2 * st.half_side, abs=1e-12)
        assert np.ptp(pts[:, 2]) == pytest.approx(2 * st.half_depth, abs=1e-12)


def test_ring_round_caps_make_a_capsule():
    r = 0.25
    form = ring(top=(0.0, 1.0, 0.0), bottom=(0.0, 0.0, 0.0), rings=24, segments=16,
                stations=flat_stations(r, r),
                caps={"top": Cap("round", r, None), "bottom": Cap("round", r, None)})
    v, _ = ring_mesh(form)
    p0, p1 = np.array([0.0, 1.0 - r, 0.0]), np.array([0.0, r, 0.0])
    seg = p1 - p0
    t = np.clip(((v - p0) @ seg) / (seg @ seg), 0.0, 1.0)
    dist = np.linalg.norm(v - (p0 + t[:, None] * seg), axis=1)
    assert np.allclose(dist, r, atol=1e-9, rtol=0)


def test_ring_super_cap():
    # Axis length 1, cap length 0.5, rings 6: the second ring sits at u = 0.25, so d = length / 2.
    length, p, rings, segments = 0.5, 5.0, 6, 16
    form = ring(top=(0.0, 1.0, 0.0), bottom=(0.0, 0.0, 0.0), rings=rings, segments=segments,
                caps={"top": Cap("super", length, p)})
    v, _ = ring_mesh(form)
    pts = v[1 + segments: 1 + 2 * segments]
    assert np.mean(1.0 - pts[:, 1]) == pytest.approx(0.25, abs=1e-12)
    scale = (1 - 0.5**5) ** (1 / 5)
    assert np.ptp(pts[:, 0]) == pytest.approx(2 * 0.3 * scale, abs=1e-12)
    assert np.ptp(pts[:, 2]) == pytest.approx(2 * 0.2 * scale, abs=1e-12)


def test_ring_bump_is_local():
    rings, segments = 12, 24
    us = cosine_us(rings)
    stations = flat_stations(0.2, 0.2)
    u0, i0, j0 = us[5], 5, 6  # ring 6 (index 5), theta = 2 pi * 6 / 24 = pi / 2
    theta0 = 2 * math.pi * j0 / segments
    bump = Bump(u=u0, u_width=0.08, angle=theta0, angle_width=0.4, height=0.05, lift=0.02)
    base = ring(rings=rings, segments=segments, stations=stations)
    bumped = ring(rings=rings, segments=segments, stations=stations, bumps=(bump,))
    v0, _ = ring_mesh(base)
    v1, _ = ring_mesh(bumped)
    delta = v1 - v0
    k = 1 + i0 * segments + j0
    # Section normal at theta = pi / 2 on a circle is +depth (+Z); toward top is +Y here.
    assert np.allclose(delta[k], 0.05 * np.array([0.0, 0.0, 1.0]) + 0.02 * UP, atol=1e-9)
    thetas = 2 * math.pi * np.arange(segments) / segments
    far = []
    for i in range(rings - 1):
        for j in range(segments):
            dth = (thetas[j] - theta0 + math.pi) % (2 * math.pi) - math.pi
            g = math.exp(-(((us[i] - u0) / 0.08) ** 2)) * math.exp(-((dth / 0.4) ** 2))
            if g < 1e-4:
                far.append(1 + i * segments + j)
    assert far
    assert np.abs(delta[far]).max() <= 0.05 * 1e-4 + 1e-12
    assert np.allclose(delta[[0, -1]], 0.0)


def _planar_ring_forms():
    caps = {"top": Cap("round", 0.3, None), "bottom": Cap("round", 0.3, None)}
    bump = Bump(u=0.5, u_width=0.1, angle=0.0, angle_width=0.5, height=0.05, lift=0.02)
    return [
        ring(),
        ring(caps=caps),
        ring(caps=caps, bumps=(bump,)),
        ring(top=(0.0, 0.0, 0.0), bottom=(0.0, 2.0, 0.0), caps=caps),
        ring(top=(0.1, 2.0, 0.0), bottom=(-0.3, 0.0, 0.4)),
    ]


def test_forms_closed_and_outward():
    cases = [blob_mesh(blob()), *(ring_mesh(f) for f in _planar_ring_forms())]
    for v, f in cases:
        t = tris(v, f)
        assert geom.is_closed(t)
        assert geom.signed_volume(t) > 0


def test_smooth_keeps_the_mesh_closed_and_outward():
    form = ring(bumps=(Bump(0.5, 0.1, 0.0, 0.5, 0.08, 0.0),),
                smooth=Smooth(passes=3, factor=0.5, below=None))
    v, f = ring_mesh(form)
    base, _ = ring_mesh(ring(bumps=form.bumps))
    assert not np.allclose(v, base)
    t = tris(v, f)
    assert geom.is_closed(t) and geom.signed_volume(t) > 0


def test_smooth_below_moves_only_low_vertices():
    bump = (Bump(0.5, 0.1, 0.0, 0.5, 0.08, 0.0),)
    plain, _ = ring_mesh(ring(bumps=bump))
    sm, _ = ring_mesh(ring(bumps=bump, smooth=Smooth(passes=2, factor=0.5, below=1.0)))
    high = plain[:, 1] >= 1.0
    assert np.array_equal(sm[high], plain[high])
    assert not np.allclose(sm[~high], plain[~high])


def test_ring_pole_offset():
    stations = (Station(0.0, 0.3, 0.2, 0.0, 0.0, 1.0), Station(1.0, 0.3, 0.2, 0.1, 0.0, 1.0))
    without, _ = ring_mesh(ring(stations=stations))
    withp, _ = ring_mesh(ring(stations=stations, poles={"bottom": (0.06, 0.0)}))
    assert np.allclose(without[-1], [0.1, 0.0, 0.0], atol=1e-12)
    assert np.allclose(withp[-1], [0.16, 0.0, 0.0], atol=1e-12)
    assert np.allclose(without[0], withp[0])


def test_push_amount_and_ramp():
    push = Push(axis="x", amount=((0.0, 0.5), (1.0, 0.5)), along="z", ramp_from=0.2,
                ramp_to=0.6, ramp_max=2.0)
    zs = np.array([0.0, 0.2, 0.6, 0.8, 5.0])
    verts = np.stack([np.zeros(5), np.full(5, 0.5), zs], axis=1)
    out = apply_deforms(verts, (push,))
    assert np.allclose(out[:, 0], [0.0, 0.0, 0.5, 0.75, 1.0], atol=1e-12)
    assert np.array_equal(out[:, 1:], verts[:, 1:])
    zero = Push("x", ((0.0, 0.0), (1.0, 0.0)), "z", 0.2, 0.6, 2.0)
    assert np.array_equal(apply_deforms(verts, (zero,)), verts)
    neg = Push("-x", ((0.0, 0.5), (1.0, 0.5)), "z", 0.2, 0.6, 2.0)
    assert np.allclose(apply_deforms(verts, (neg,))[:, 0], -out[:, 0])


def test_push_table_blends_by_smoothstep_and_holds_ends():
    push = Push("y", ((0.0, 0.0), (1.0, 1.0)), "z", 0.0, 1.0, 1.0)
    ys = np.array([-1.0, 0.25, 0.5, 2.0])
    verts = np.stack([np.zeros(4), ys, np.ones(4)], axis=1)
    out = apply_deforms(verts, (push,))
    want = np.array([0.0, 0.25 * 0 + 0.0625 * (3 - 0.5), 0.5, 1.0])
    assert np.allclose(out[:, 1] - ys, want, atol=1e-12)


def test_flatten_eases_past_start():
    start, end = 0.4, 0.5
    cs = np.array([-1.0, 0.2, 0.4, 0.7, 1.0])
    verts = np.stack([np.zeros(5), cs, np.zeros(5)], axis=1)
    out = apply_deforms(verts, (Flatten("y", start, end),))
    assert np.allclose(out[:3, 1], cs[:3])
    assert out[4, 1] == pytest.approx(end, abs=1e-12)
    t = 0.5
    assert out[3, 1] == pytest.approx(start + (end - start) * (1 - (1 - t) ** 3), abs=1e-12)
    neg = apply_deforms(verts * [1, -1, 1], (Flatten("-y", start, end),))
    assert np.allclose(neg[:, 1], -out[:, 1])
    assert np.array_equal(out[:, [0, 2]], verts[:, [0, 2]])


def test_deforms_apply_in_order():
    a, b = Flatten("y", 0.4, 0.5), Push("x", ((0.0, 1.0), (1.0, 1.0)), "y", 0.0, 1.0, 1.0)
    verts = np.array([[0.0, 0.2, 0.0], [0.0, 1.0, 0.0]])
    step = apply_deforms(apply_deforms(verts, (a,)), (b,))
    assert np.allclose(apply_deforms(verts, (a, b)), step)


# ----- the readers -----


def ring_raw(**extra) -> dict:
    return {"top": [0, 2, 0], "bottom": [0, 0, 0], "rings": 8, "segments": 16,
            "stations": [{"u": 0, "half_side": 0.3, "half_depth": 0.2},
                         {"u": 1, "half_side": 0.3, "half_depth": 0.2}], **extra}


def blob_raw(**extra) -> dict:
    return {"half": [0.3, 0.5, 0.2], "centre": [0, 1, 0], "power_side": 1, "power_up": 1,
            "power_down": 1, "segments": 16, "rings": 8, **extra}


def decal_raw(**extra) -> dict:
    return {"half": [0.05, 0.04], "segments": 16, "rings": 4, "at": [0.1, 0.6, 0.0], **extra}


def meta_raw(**extra) -> dict:
    return {"elements": [{"kind": "ball", "centre": [0, 0, 0], "half": [0.1, 0.1, 0.1]}],
            "resolution": 0.02, "threshold": 0.6,
            "smooth": {"passes": 2, "factor": 0.5}, **extra}


def test_read_blob_ring_and_meta():
    soft = read_form(AT, "blob", blob_raw())
    assert soft == SoftForm(body=blob(half=(0.3, 0.5, 0.2), centre=(0.0, 1.0, 0.0)),
                            mirror_of=None, seat=None, deforms=(), why=None)
    assert hash(soft) == hash(read_form(AT, "blob", blob_raw()))

    raw = ring_raw(blend="smooth", caps={"top": {"kind": "super", "length": 0.3, "power": 4}},
                   poles={"bottom": [0.06, 0]},
                   bumps=[{"u": 0.5, "u_width": 0.1, "angle": 0.3, "angle_width": 0.4,
                           "height": 0.05, "lift": 0.01}],
                   smooth={"passes": 2, "factor": 0.4, "below": 0.5},
                   seat={"on": "torso", "at": [0.0, 0.1], "sink": 0.02},
                   deforms=[{"kind": "flatten", "axis": "-y", "start": 0.1, "end": 0.2},
                            {"kind": "push", "axis": "x", "amount": [[0, 0.1], [1, 0.2]],
                             "along": "z", "ramp_from": 0, "ramp_to": 1, "ramp_max": 2}],
                   why="a thumb")
    body = read_form(AT, "ring", raw)
    assert isinstance(body.body, RingForm)
    assert body.body.blend == "smooth"
    assert body.body.caps == {"top": Cap("super", 0.3, 4.0)}
    assert body.body.poles == {"bottom": (0.06, 0.0)}
    assert body.body.smooth == Smooth(2, 0.4, 0.5)
    assert body.seat == SeatOn("torso", (0.0, 0.1), 0.02)
    assert body.deforms[0] == Flatten("-y", 0.1, 0.2)
    assert isinstance(body.deforms[1], Push)
    assert body.why == "a thumb"

    soft = read_form(AT, "meta", meta_raw(decimate=0.5))
    assert isinstance(soft.body, MetaForm)
    assert soft.body.decimate == 0.5 and soft.body.smooth == Smooth(2, 0.5, None)


def test_read_decal():
    soft = read_form(AT, "decal", decal_raw(why="an eye"))
    assert soft == SoftForm(DecalForm((0.05, 0.04), 16, 4, (0.1, 0.6, 0.0), (0.0, 0.0, -1.0)),
                            None, None, (), "an eye")
    side = read_form(AT, "decal", decal_raw(along=[-1, 0, 0]))
    assert side.body.along == (-1.0, 0.0, 0.0)


def test_decal_grid_is_a_flat_oval_wound_anticlockwise():
    form = DecalForm((0.05, 0.04), 16, 4, (0.0, 0.0, 0.0))
    points, faces = decal_grid(form)
    assert points.shape == (1 + 4 * 16, 2) and len(faces) == 16 * 4
    assert np.allclose(points[0], 0.0)
    rim = points[1 + 3 * 16:]
    assert np.allclose((rim[:, 0] / 0.05) ** 2 + (rim[:, 1] / 0.04) ** 2, 1.0)
    for f in faces:
        p = points[list(f)]
        area = 0.5 * sum(p[k, 0] * p[(k + 1) % len(p), 1] - p[(k + 1) % len(p), 0] * p[k, 1]
                         for k in range(len(p)))
        assert area > 0, f
    # Every inner edge is shared by two faces running it opposite ways; the rim's 16 are open.
    edges = [(f[k], f[(k + 1) % len(f)]) for f in faces for k in range(len(f))]
    assert len(set(edges)) == len(edges)
    assert sum((b, a) not in set(edges) for a, b in edges) == 16


def test_read_mirror():
    soft = read_form(AT, "blob", {"mirror_of": "arm_l", "why": "the other arm"})
    assert soft == SoftForm(None, "arm_l", None, (), "the other arm")


def test_read_seat_and_deform_directly():
    assert read_seat(AT, {"on": "torso", "at": [0.1, 0.2], "sink": 0.0}) == SeatOn(
        "torso", (0.1, 0.2), 0.0)
    assert read_deform(AT, {"kind": "flatten", "axis": "x", "start": 0, "end": 1}) == Flatten(
        "x", 0.0, 1.0)


def _drop(raw: dict, *keys: str) -> dict:
    return {k: v for k, v in raw.items() if k not in keys}


def _stations(*rows) -> dict:
    return {"stations": [{"u": u, "half_side": hs, "half_depth": hd} for u, hs, hd in rows]}


REJECTS = [
    ("unknown key", "ring", ring_raw(colour=1), "form"),
    ("unknown shape", "cube", ring_raw(), "form"),
    ("stations do not start at 0", "ring",
     ring_raw(**_stations((0.1, 0.3, 0.2), (1, 0.3, 0.2))), "stations[0].u"),
    ("stations do not end at 1", "ring",
     ring_raw(**_stations((0, 0.3, 0.2), (0.9, 0.3, 0.2))), "stations[1].u"),
    ("station u does not increase", "ring",
     ring_raw(**_stations((0, 0.3, 0.2), (0.5, 0.3, 0.2), (0.5, 0.3, 0.2), (1, 0.3, 0.2))),
     "stations[2].u"),
    ("negative half", "ring",
     ring_raw(**_stations((0, 0.3, 0.2), (0.5, -0.1, 0.2), (1, 0.3, 0.2))),
     "stations[1].half_side"),
    ("zero half mid-axis", "ring",
     ring_raw(**_stations((0, 0.3, 0.2), (0.5, 0.3, 0.0), (1, 0.3, 0.2))),
     "stations[1].half_depth"),
    ("overlapping caps", "ring",
     ring_raw(caps={"top": {"kind": "round", "length": 1.2},
                    "bottom": {"kind": "round", "length": 1.2}}), "caps"),
    ("super cap with no power", "ring",
     ring_raw(caps={"top": {"kind": "super", "length": 0.3}}), "caps.top"),
    ("round cap with a power", "ring",
     ring_raw(caps={"top": {"kind": "round", "length": 0.3, "power": 3}}), "caps.top.power"),
    ("cap at an unknown end", "ring",
     ring_raw(caps={"middle": {"kind": "round", "length": 0.3}}), "caps"),
    ("rings 2", "ring", ring_raw(rings=2), "form.rings"),
    ("segments 2", "ring", ring_raw(segments=2), "form.segments"),
    ("fractional rings", "ring", ring_raw(rings=8.5), "form.rings"),
    ("axis of zero length", "ring", ring_raw(bottom=[0, 2, 0]), "form"),
    ("axis along x", "ring", ring_raw(top=[0, 0, 0], bottom=[2, 0, 0]), "form"),
    ("bump with zero width", "ring",
     ring_raw(bumps=[{"u": 0.5, "u_width": 0, "angle": 0, "angle_width": 0.4, "height": 0.1,
                      "lift": 0}]), "bumps[0].u_width"),
    ("smooth with no passes", "ring", ring_raw(smooth={"passes": 0, "factor": 0.5}),
     "smooth.passes"),
    ("meta with no elements", "meta", meta_raw(elements=[]), "elements"),
    ("meta decimate 0", "meta", meta_raw(decimate=0), "decimate"),
    ("meta decimate above 1", "meta", meta_raw(decimate=1.5), "decimate"),
    ("ball with unequal halves", "meta",
     meta_raw(elements=[{"kind": "ball", "centre": [0, 0, 0], "half": [0.1, 0.2, 0.1]}]),
     "elements[0].half"),
    ("mirror with shape keys", "blob", {"mirror_of": "arm_l", **blob_raw()}, "form"),
    ("mirror with a seat", "blob",
     {"mirror_of": "arm_l", "seat": {"on": "t", "at": [0, 0], "sink": 0}}, "form"),
    ("mirror with deforms", "blob", {"mirror_of": "arm_l", "deforms": []}, "form"),
    ("bad deform axis", "blob",
     blob_raw(deforms=[{"kind": "flatten", "axis": "w", "start": 0, "end": 1}]),
     "deforms[0].axis"),
    ("bad deform kind", "blob", blob_raw(deforms=[{"kind": "twist"}]), "deforms[0].kind"),
    ("push along a signed axis", "blob",
     blob_raw(deforms=[{"kind": "push", "axis": "x", "amount": [[0, 1]], "along": "-z",
                        "ramp_from": 0, "ramp_to": 1, "ramp_max": 1}]), "deforms[0].along"),
    ("push amounts out of order", "blob",
     blob_raw(deforms=[{"kind": "push", "axis": "x", "amount": [[1, 1], [0, 1]], "along": "z",
                        "ramp_from": 0, "ramp_to": 1, "ramp_max": 1}]), "deforms[0].amount"),
    ("blob power 0", "blob", blob_raw(power_up=0), "power_up"),
    ("blob half 0", "blob", blob_raw(half=[0, 1, 1]), "form.half"),
    ("empty why", "blob", blob_raw(why=" "), "form.why"),
    ("decal half 0", "decal", decal_raw(half=[0, 0.04]), "form.half"),
    ("decal half with 3 values", "decal", decal_raw(half=[0.05, 0.04, 0.01]), "form.half"),
    ("decal rings 0", "decal", decal_raw(rings=0), "form.rings"),
    ("decal along not a unit", "decal", decal_raw(along=[0, 0, -2]), "form.along"),
    ("decal with a seat", "decal",
     decal_raw(seat={"on": "head", "at": [0, 0], "sink": 0}), "form"),
    ("decal with deforms", "decal", decal_raw(deforms=[]), "form"),
]


@pytest.mark.parametrize("shape,raw,where", [r[1:] for r in REJECTS], ids=[r[0] for r in REJECTS])
def test_read_form_rejects(shape, raw, where):
    with pytest.raises(ManifestError) as err:
        read_form(AT, shape, raw)
    assert where in str(err.value)
    assert "pieces.a.parts[0].form" in str(err.value)
