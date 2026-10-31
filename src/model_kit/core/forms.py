"""Soft part geometry in numpy: blob and ring meshes, the decal's flat grid, mesh smoothing and
the deforms.

Vertices are (N, 3) in glTF axes. Faces are index tuples wound outward, so no normal
recalculation is needed. A mirror flips them later.
"""

from __future__ import annotations

import numpy as np

from .manifest.forms import (
    DEFORM_AXES,
    BlobForm,
    DecalForm,
    Falloff,
    Flatten,
    Push,
    RingForm,
    Smooth,
    Station,
)

# Step in radians for the central difference that finds a section's tangent.
_NORMAL_STEP = 1e-6
# Shortest vector that still counts as a direction when the ring frame is built.
_FRAME_MIN = 1e-9

# A deform axis name to (column of the vertex array, sign). Keyed by the loader's own
# vocabulary, so a name the loader accepts always has a row here.
_SIGN = {name: ("xyz".index(name.lstrip("-")), -1.0 if name.startswith("-") else 1.0)
         for name in DEFORM_AXES}


def _signed_power(v: np.ndarray, power: float) -> np.ndarray:
    return np.sign(v) * np.abs(v) ** power


def _pole_grid_faces(rings: int, segments: int, flip: bool) -> list[tuple[int, ...]]:
    """Faces for a top pole (0), (rings - 1) rings of `segments` points, and a bottom pole.

    Angle runs round each ring; rings go from the top pole to the bottom pole. With
    `flip` False the winding is outward when (angle direction x down direction) points out.
    """
    def at(i: int, j: int) -> int:
        return 1 + (i - 1) * segments + j % segments

    last = 1 + (rings - 1) * segments
    faces: list[tuple[int, ...]] = []
    for j in range(segments):
        faces.append((0, at(1, j + 1), at(1, j)))
    for i in range(1, rings - 1):
        for j in range(segments):
            faces.append((at(i, j), at(i, j + 1), at(i + 1, j + 1), at(i + 1, j)))
    for j in range(segments):
        faces.append((at(rings - 1, j), at(rings - 1, j + 1), last))
    if flip:
        faces = [f[::-1] for f in faces]
    return faces


def blob_mesh(form: BlobForm) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """A superellipsoid: `centre` plus `half` at the extremes, squarer for powers below 1."""
    phi = np.pi * np.arange(1, form.rings) / form.rings
    theta = 2 * np.pi * np.arange(form.segments) / form.segments
    cos_phi = np.cos(phi)[:, None]
    power_y = np.where(cos_phi >= 0, form.power_up, form.power_down)
    y = np.sign(cos_phi) * np.abs(cos_phi) ** power_y
    radius = np.sin(phi)[:, None] ** form.power_side
    x = radius * _signed_power(np.cos(theta)[None, :], form.power_side)
    z = radius * _signed_power(np.sin(theta)[None, :], form.power_side)
    y = np.broadcast_to(y, x.shape)
    rim = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    unit = np.vstack([[0.0, 1.0, 0.0], rim, [0.0, -1.0, 0.0]])
    verts = np.array(form.centre) + unit * np.array(form.half)
    return verts, _pole_grid_faces(form.rings, form.segments, flip=False)


def decal_grid(form: DecalForm) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """A decal's oval as a flat polar grid, before it is cast onto its host.

    Returns (N, 2) points (side, up) in the oval's plane, and faces wound anticlockwise seen from
    the front of the plane (side x up points out of the host). Point 0 is the centre; ring i of
    `segments` points sits at i / `rings` of the way to the rim, and the last ring is the rim."""
    theta = 2 * np.pi * np.arange(form.segments) / form.segments
    share = np.arange(1, form.rings + 1)[:, None] / form.rings
    side = share * form.half[0] * np.cos(theta)[None, :]
    up = share * form.half[1] * np.sin(theta)[None, :]
    points = np.vstack([[0.0, 0.0], np.stack([side, up], axis=-1).reshape(-1, 2)])

    def at(i: int, j: int) -> int:
        return 1 + (i - 1) * form.segments + j % form.segments

    faces: list[tuple[int, ...]] = [(0, at(1, j), at(1, j + 1)) for j in range(form.segments)]
    for i in range(1, form.rings):
        for j in range(form.segments):
            faces.append((at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)))
    return points, faces


def _frame(top: np.ndarray, bottom: np.ndarray):
    """The unit axis (top to bottom), side (+X) and depth (+Z) directions, and the axis length."""
    axis = bottom - top
    length = float(np.linalg.norm(axis))
    if length < _FRAME_MIN:
        raise ValueError("ring top and bottom are the same point")
    a = axis / length
    side = np.array([1.0, 0.0, 0.0]) - a[0] * a
    if np.linalg.norm(side) < _FRAME_MIN:
        raise ValueError("the ring axis runs along X, so its side direction is undefined")
    side /= np.linalg.norm(side)
    depth = np.array([0.0, 0.0, 1.0]) - a[2] * a - side[2] * side
    if np.linalg.norm(depth) < _FRAME_MIN:
        raise ValueError("the ring axis runs along Z, so its depth direction is undefined")
    depth /= np.linalg.norm(depth)
    return a, side, depth, length


def _station_table(stations: tuple[Station, ...], u: np.ndarray, blend: str) -> np.ndarray:
    """Station values (half_side, half_depth, off_side, off_depth, inner_power) at each u."""
    us = np.array([s.u for s in stations])
    rows = np.array([[s.half_side, s.half_depth, s.off_side, s.off_depth, s.inner_power]
                     for s in stations])
    k = np.clip(np.searchsorted(us, u, side="right") - 1, 0, len(stations) - 2)
    t = np.clip((u - us[k]) / (us[k + 1] - us[k]), 0.0, 1.0)
    if blend == "smooth":
        t = t * t * (3.0 - 2.0 * t)
    return rows[k] + (rows[k + 1] - rows[k]) * t[:, None]


def _cap_scale(depth_in: np.ndarray, length: float, kind: str, power: float | None) -> np.ndarray:
    """Half-size factor at distance `depth_in` along the axis from the end. 1 beyond the cap."""
    d = length - np.minimum(depth_in, length)
    x = d / length
    if kind == "round":
        return np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    return np.clip(1.0 - x**power, 0.0, None) ** (1.0 / power)


def _section_x(theta: np.ndarray, inner_power: np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    return np.where(c >= 0, c, -(np.abs(c) ** inner_power))


def ring_mesh(form: RingForm) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """A swept section along an axis, with stations, caps, poles, bumps and smoothing."""
    top, bottom = np.array(form.top), np.array(form.bottom)
    a, side, depth, length = _frame(top, bottom)
    n_rings, n_seg = form.rings - 1, form.segments
    u = (1.0 - np.cos(np.pi * np.arange(1, form.rings) / form.rings)) / 2.0
    theta = 2 * np.pi * np.arange(n_seg) / n_seg

    hs, hd, off_s, off_d, ip = _station_table(form.stations, u, form.blend).T
    scale = np.ones(n_rings)
    for end, along in (("top", u * length), ("bottom", (1.0 - u) * length)):
        cap = form.caps.get(end)
        if cap is not None:
            scale = scale * np.where(along < cap.length,
                                     _cap_scale(along, cap.length, cap.kind, cap.power), 1.0)
    hs, hd = hs * scale, hd * scale
    th = theta[None, :]
    sec_s = off_s[:, None] + hs[:, None] * _section_x(th, ip[:, None])
    sec_d = off_d[:, None] + hd[:, None] * np.sin(th)
    ring = (top + u[:, None, None] * (bottom - top)
            + sec_s[..., None] * side + sec_d[..., None] * depth)

    for bump in form.bumps:
        dtheta = (th - bump.angle + np.pi) % (2 * np.pi) - np.pi
        g = (np.exp(-(((u - bump.u) / bump.u_width) ** 2))[:, None]
             * np.exp(-((dtheta / bump.angle_width) ** 2)))
        step = _NORMAL_STEP
        x_span = _section_x(th + step, ip[:, None]) - _section_x(th - step, ip[:, None])
        ts = (hs[:, None] * x_span, hd[:, None] * (np.sin(th + step) - np.sin(th - step)))
        norm = np.hypot(*ts)
        safe = np.where(norm > 0, norm, 1.0)
        n_side, n_depth = ts[1] / safe, -ts[0] / safe
        ring = (ring + (bump.height * g * n_side)[..., None] * side
                + (bump.height * g * n_depth)[..., None] * depth
                - (bump.lift * g)[..., None] * a)

    ends = []
    for end, station, anchor in (("top", form.stations[0], top),
                                 ("bottom", form.stations[-1], bottom)):
        extra_s, extra_d = form.poles.get(end, (0.0, 0.0))
        ends.append(anchor + side * (station.off_side + extra_s)
                    + depth * (station.off_depth + extra_d))
    verts = np.vstack([ends[0], ring.reshape(-1, 3), ends[1]])

    handed = float(np.dot(np.cross(side, depth), a)) > 0
    faces = _pole_grid_faces(form.rings, form.segments, flip=not handed)
    if form.smooth is not None:
        verts = smooth_vertices(verts, faces, form.smooth)
    return verts, faces


def smooth_vertices(verts: np.ndarray, faces, smooth: Smooth) -> np.ndarray:
    """Laplacian smoothing: each pass moves a vertex `factor` of the way to its neighbours' mean.

    With `smooth.below` set, only vertices under that Y (taken before the first pass) move.
    """
    edges = np.array([(f[k], f[(k + 1) % len(f)]) for f in faces for k in range(len(f))])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    both = np.vstack([edges, edges[:, ::-1]])
    degree = np.bincount(both[:, 0], minlength=len(verts)).astype(float)
    free = np.ones(len(verts), bool) if smooth.below is None else verts[:, 1] < smooth.below
    out = np.array(verts, float)
    for _ in range(smooth.passes):
        total = np.zeros_like(out)
        np.add.at(total, both[:, 0], out[both[:, 1]])
        mean = total / np.maximum(degree, 1.0)[:, None]
        out = np.where(free[:, None], out + smooth.factor * (mean - out), out)
    return out


def _table_at(table: tuple[tuple[float, float], ...], y: np.ndarray) -> np.ndarray:
    ys = np.array([row[0] for row in table])
    amounts = np.array([row[1] for row in table])
    if len(table) == 1:
        return np.full(y.shape, amounts[0])
    k = np.clip(np.searchsorted(ys, y, side="right") - 1, 0, len(table) - 2)
    t = np.clip((y - ys[k]) / (ys[k + 1] - ys[k]), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return amounts[k] + (amounts[k + 1] - amounts[k]) * t


def _push(verts: np.ndarray, push: Push) -> np.ndarray:
    col, sign = _SIGN[push.axis]
    along = verts[:, _SIGN[push.along][0]]
    ramp = np.clip((along - push.ramp_from) / (push.ramp_to - push.ramp_from),
                   0.0, push.ramp_max)
    out = verts.copy()
    out[:, col] += sign * _table_at(push.amount, verts[:, 1]) * ramp
    return out


def _flatten(verts: np.ndarray, flat: Flatten) -> np.ndarray:
    col, sign = _SIGN[flat.axis]
    out = verts.copy()
    if len(verts) == 0:
        return out
    c = sign * verts[:, col]
    c_max = float(c.max())
    if c_max <= flat.start:
        return out
    past = c > flat.start
    t = (c[past] - flat.start) / (c_max - flat.start)
    c[past] = flat.start + (flat.end - flat.start) * (1.0 - (1.0 - t) ** 3)
    out[:, col] = sign * c
    return out


def apply_deforms(verts: np.ndarray, deforms) -> np.ndarray:
    """Apply pushes and flattens in order. Returns a new array."""
    out = np.array(verts, float)
    for deform in deforms:
        out = _push(out, deform) if isinstance(deform, Push) else _flatten(out, deform)
    return out


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def chain_weights(coords: np.ndarray, falloff: Falloff, n_bones: int) -> np.ndarray:
    """Bone weights for points bound to a chain: an (N, n_bones) array whose rows sum to 1.

    `coords` is (N, 3). Each point's u is its coordinate along `falloff.along`, taken from
    `falloff.start` to `falloff.end` and clamped to [0, 1]. Band k is where the weight passes
    from bone k to bone k + 1: with s_k the smoothstep of u across that band, bone i weighs
    s_(i-1) - s_i, where s_(-1) is 1 and s_(n-1) is 0.
    """
    points = np.asarray(coords, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3), got {points.shape}")
    if len(falloff.bands) != n_bones - 1:
        raise ValueError(f"{n_bones} bones need {n_bones - 1} falloff bands, "
                         f"got {len(falloff.bands)}")
    column = "xyz".index(falloff.along)
    u = np.clip((points[:, column] - falloff.start) / (falloff.end - falloff.start), 0.0, 1.0)
    s = np.ones((len(u), n_bones + 1))
    s[:, n_bones] = 0.0
    for k, (lo, hi) in enumerate(falloff.bands):
        s[:, k + 1] = _smoothstep((u - lo) / (hi - lo))
    return s[:, :-1] - s[:, 1:]
