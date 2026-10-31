"""Fixture meshes for the geometry tests. A helper module, not a test file."""

import numpy as np


def box(lo, hi) -> np.ndarray:
    """A closed box between two corners, 12 outward triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    quads = [
        [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)],  # +y
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # -y
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # +x
        [(x0, y0, z1), (x0, y1, z1), (x0, y1, z0), (x0, y0, z0)],  # -x
        [(x1, y0, z1), (x1, y1, z1), (x0, y1, z1), (x0, y0, z1)],  # +z
        [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)],  # -z
    ]
    return np.array([t for p in quads for t in ((p[0], p[1], p[2]), (p[0], p[2], p[3]))],
                    dtype=float)


def _box(w: float, t: float) -> np.ndarray:
    h = w / 2
    return box((-h, 0.0, -h), (h, t, h))


def plate(w: float = 0.2, t: float = 0.002) -> np.ndarray:
    """A closed box: x and z in +-w/2, y in 0..t, 12 outward triangles."""
    return _box(w, t)


def block(w: float = 0.2, t: float = 0.08) -> np.ndarray:
    """The same shape as `plate`, thicker."""
    return _box(w, t)


def open_plate() -> np.ndarray:
    """The top face of the default plate only (2 triangles)."""
    return _box(0.2, 0.002)[:2]


def soup(n: int, size: float, seed: int = 1) -> np.ndarray:
    """`n` random triangles of about `size` across, spread over the cube +-1."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1.0, 1.0, size=(n, 1, 3))
    return centers + rng.normal(size=(n, 3, 3)) * size


def blob(rings: int = 20, around: int = 40, bump: float = 0.25) -> np.ndarray:
    """A closed, bumpy, star-shaped surface: a UV sphere whose radius varies with direction."""
    theta = np.linspace(0.0, np.pi, rings + 1)[1:-1]
    phi = np.linspace(0.0, 2 * np.pi, around, endpoint=False)
    t, p = np.meshgrid(theta, phi, indexing="ij")
    r = 1.0 + bump * np.sin(3 * p) * np.cos(2 * t)
    grid = np.stack([r * np.sin(t) * np.cos(p), r * np.cos(t), r * np.sin(t) * np.sin(p)], axis=-1)
    verts = np.concatenate([[[0.0, 1.0, 0.0]], grid.reshape(-1, 3),
                            [[0.0, -1.0, 0.0]]])
    south = len(verts) - 1
    ring = lambda i, j: 1 + i * around + j % around
    faces = [(0, ring(0, j + 1), ring(0, j)) for j in range(around)]
    for i in range(rings - 2):
        for j in range(around):
            faces += [(ring(i, j), ring(i, j + 1), ring(i + 1, j)),
                      (ring(i, j + 1), ring(i + 1, j + 1), ring(i + 1, j))]
    faces += [(south, ring(rings - 2, j), ring(rings - 2, j + 1)) for j in range(around)]
    return verts[np.array(faces)]


def wavy_grid(nx: int, ny: int) -> np.ndarray:
    """A heightfield of 2 * nx * ny triangles over x, z in 0..1, rippled in y."""
    xs, zs = np.linspace(0.0, 1.0, nx + 1), np.linspace(0.0, 1.0, ny + 1)
    x, z = np.meshgrid(xs, zs, indexing="ij")
    verts = np.stack([x, 0.05 * np.sin(9 * x) * np.cos(7 * z), z], axis=-1)
    a, b = verts[:-1, :-1], verts[1:, :-1]
    c, d = verts[1:, 1:], verts[:-1, 1:]
    return np.concatenate([np.stack([a, b, c], axis=2).reshape(-1, 3, 3),
                           np.stack([a, c, d], axis=2).reshape(-1, 3, 3)])


def cylinder(centre, radius: float, y0: float, y1: float, sides: int = 24,
             hole: float = 0.0) -> np.ndarray:
    """A closed upright cylinder around the (x, z) `centre`, from y0 to y1, with outward
    triangles. With `hole` above 0 it is a ring: a tube of that inner radius, a closed mesh too."""
    cx, cz = centre
    angle = 2 * np.pi * np.arange(sides) / sides

    def ring(r: float, y: float) -> np.ndarray:
        return np.stack([cx + r * np.cos(angle), np.full(sides, y), cz + r * np.sin(angle)], 1)

    def faced(tri, away) -> list:
        tri = np.array(tri)
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        return list(tri if normal @ away >= 0 else tri[::-1])

    out = []

    def quad(a, b, c, d, away) -> None:
        out.extend([faced([a, b, c], away), faced([a, c, d], away)])

    up, down = np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])
    inner = max(0.0, hole)
    for i in range(sides):
        j = (i + 1) % sides
        mid = np.array([np.cos(angle[i] + np.pi / sides), 0.0, np.sin(angle[i] + np.pi / sides)])
        o0, o1, p0, p1 = ring(radius, y0), ring(radius, y1), ring(inner, y0), ring(inner, y1)
        quad(o0[i], o0[j], o1[j], o1[i], mid)  # outer wall
        if hole > 0.0:
            quad(p0[i], p0[j], p1[j], p1[i], -mid)  # inner wall, facing the axis
            quad(o1[i], o1[j], p1[j], p1[i], up)
            quad(o0[i], o0[j], p0[j], p0[i], down)
        else:
            out.append(faced([o1[i], o1[j], [cx, y1, cz]], up))
            out.append(faced([o0[i], o0[j], [cx, y0, cz]], down))
    return np.array(out, dtype=float)


def pitted_box(lo, hi, pits, floor: float) -> np.ndarray:
    """A closed box between two corners with square blind pits cut down from its top face,
    outward triangles. Each pit is `((x0, z0), (x1, z1))`, inside the top face and apart from the
    others; its floor lies at y = `floor`. Every face is cut on the grid of the box's and the pits'
    x and z, so the mesh is watertight."""
    (X0, y0, Z0), (X1, y1, Z1) = lo, hi
    xs = sorted({X0, X1, *(p[i][0] for p in pits for i in (0, 1))})
    zs = sorted({Z0, Z1, *(p[i][1] for p in pits for i in (0, 1))})

    def in_pit(i: int, j: int) -> bool:
        cx, cz = (xs[i] + xs[i + 1]) / 2, (zs[j] + zs[j + 1]) / 2
        return any(p[0][0] < cx < p[1][0] and p[0][1] < cz < p[1][1] for p in pits)

    cells = [(i, j) for i in range(len(xs) - 1) for j in range(len(zs) - 1)]
    out: list = []

    def quad(a, b, c, d, away) -> None:
        for tri in ((a, b, c), (a, c, d)):
            t = np.array(tri, dtype=float)
            flip = np.cross(t[1] - t[0], t[2] - t[0]) @ np.asarray(away, dtype=float) < 0
            out.append(t[::-1] if flip else t)

    up, down = (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)
    for i, j in cells:
        x0, x1, z0, z1 = xs[i], xs[i + 1], zs[j], zs[j + 1]
        quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), down)
        pit = in_pit(i, j)
        y = floor if pit else y1
        quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), up)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # the four neighbours
            ni, nj = i + di, j + dj
            edge = ((x1 if di == 1 else x0, x1 if di == 1 else x0) if di else (x0, x1),
                    (z1 if dj == 1 else z0, z1 if dj == 1 else z0) if dj else (z0, z1))
            outside = not (0 <= ni < len(xs) - 1 and 0 <= nj < len(zs) - 1)
            if outside:  # the box's own side
                lo_y, hi_y, away = y0, y1, (di, 0.0, dj)
            elif not pit and in_pit(ni, nj):  # a pit wall, seen from the pit: faces the pit
                lo_y, hi_y, away = floor, y1, (-di, 0.0, -dj)
            else:
                continue
            (ea, eb), (fa, fb) = edge
            quad((ea, lo_y, fa), (eb, lo_y, fb), (eb, hi_y, fb), (ea, hi_y, fa), away)
    return np.array(out, dtype=float)
