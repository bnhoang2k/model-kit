"""Place every part of a machine's parts.py on its host, for one style. Plain Python (no Blender),
so the layout can be checked in a second: `python3 resolve.py <machine folder> real|toy` prints
every seat or collision problem and exits 1 if there is one. The machine folder comes from the
first argument or the MACHINE_DIR environment variable (machine_build.py sets it).

A part stands on one face of its host. Its frame is that face's (u, v, n): the footprint lies in
(u, v), the height runs along n. It sinks SINK into its host (a join rule: a join sinks
0.018 mm, under the 0.03 mm build bar), so it touches and nothing passes deeper.

A part may carry a `tilt` (degrees): it turns in its host face's plane, about the face's outward
normal, which must be horizontal (a side face). It stays flush on that face, so a slab on a cheek's
side becomes a sloped lane or a slanted chute. A part on a tilted part's face takes that face's
frame, so a parcel on a sloped lane rests on the slope. Faces are still named by world direction:
on a tilted host the name picks the face that looks nearest that way (within 45 degrees).

An `arc` is a horizontal annular sector swept round a vertical axis (a curved belt, its pan and
frames). Its `at` is the axis point; R0, R1 its radii; `arc` its start and end angles in degrees
(0 is +x, 90 is +z); `trim` stops both ends that far inside their radial planes, square to them.
Arcs on one axis share one facet grid, so nested walls stay parallel.

Seats on a tilted part or an arc are checked on the turned outline or the sector; collisions use
oriented boxes (separating axes) whenever a part is tilted or curved, and the old world-axis boxes
otherwise, so a machine with neither resolves and builds exactly as before.

The toy is derived from the real resolution: each part keeps its offset on its host face, scaled
by TOY_SCALE along the world axes; sizes scale the same way, controls grow by TOY_CTL instead,
thin parts thicken, every part rounds hard, and `tier: "real"` parts are dropped.
"""

import importlib
import math
import os
import sys

import toy as _toy

if __name__ == "__main__" and len(sys.argv) > 2:
    os.environ["MACHINE_DIR"] = str(os.path.abspath(sys.argv[1]))
if not os.environ.get("MACHINE_DIR"):
    raise SystemExit("resolve.py needs the machine folder: MACHINE_DIR or the first argument")
sys.path.insert(0, os.environ["MACHINE_DIR"])
P = importlib.import_module("parts")


def _knob(name):
    """A toy number: the machine's own if its parts.py sets one, else the kit's (toy.py)."""
    return getattr(P, name, getattr(_toy, name))


TOY_SCALE, TOY_CTL, TOY_ROUND, TOY_RIM, TOY_MIN_SIDE, TOY_MIN_H = (
    _knob(n) for n in ("TOY_SCALE", "TOY_CTL", "TOY_ROUND", "TOY_RIM", "TOY_MIN_SIDE", "TOY_MIN_H"))

SINK = 0.018e-3  # metres: the depth every join sinks into its host
MARGIN = 0.3e-3  # a seat must lie this far inside its host's flat face (float noise is far less)
NEAREST_M = _knob("NEAREST_M")  # view.nearest_m (toy.py says why)
EYE = NEAREST_M * math.tan(math.radians(1 / 60))  # 1.16 mm: what the eye resolves at 4 m
GAP = 0.9 * EYE  # the most a facet may stand off its curve (the facet-gap rule)
MAX_TURN = 30.0  # no smooth facet turns more than this, 5 under the 35 degree crease angle
SWEEP = {"real": 40.0, "toy": 90.0}  # rim sweep: real draws one crease onto the top, toy none
TOY_ROUND_MAX = 0.16  # toy corner radius cap; easing (below) shrinks it where a host must carry parts
TOY_RIM_MAX = 0.07  # toy rim radius cap, eased the same way


def steps(radius, sweep_deg):
    """steps: the fewest facets for an arc whose gap stays under GAP and whose facets
    each turn at most MAX_TURN."""
    theta = 2 * math.acos(1 - GAP / radius) if radius > GAP else math.pi
    theta = min(theta, math.radians(MAX_TURN))
    return max(1, math.ceil(math.radians(sweep_deg) / theta))


# --- small vector helpers ---------------------------------------------------------------------
def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, k):
    return tuple(x * k for x in a)


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def neg(a):
    return tuple(-x for x in a)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def square(vec):
    """True when the vector runs along a world axis (every component 0 or +-1)."""
    return all(abs(x) in (0.0, 1.0) for x in vec)


def axis(vec):
    """The world axis (0, 1, 2) an axis-aligned unit vector runs along."""
    return max(range(3), key=lambda i: abs(vec[i]))


WORLD = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
         "+z": (0, 0, 1), "-z": (0, 0, -1)}


# --- faces --------------------------------------------------------------------------------------
def faces(p):
    """{name: (centre, U, V, N, flat)} of a placed part. `flat` is the face's flat region in face
    coordinates: ("rrect", W, D, R) centred on the face centre, or ("rect", a0, a1, b0, b1)."""
    if p["kind"] in ("tube", "lathe"):
        return {}
    if p["kind"] == "arc":  # only the top and bottom are flat; the flat is the sector, in axis coordinates
        o, u, v, n, H = p["o"], p["u"], p["v"], p["n"], p["H"]
        a0, a1 = p["arc"]
        r0, r1, ytop = p["top_r"]
        return {"top": (add(o, mul(n, ytop)), u, v, n, ("sector", r0, r1, a0, a1, p["trim"])),
                "bottom": (o, u, v, neg(n), ("sector", p["R0"], p["R1"], a0, a1, p["trim"]))}
    o, u, v, n = p["o"], p["u"], p["v"], p["n"]
    w, d, H, r = p["w"], p["d"], p["H"], p["r"]
    it = p["e"] * (1 - math.cos(math.radians(p["sweep"])))
    yt = H - p["e"] * math.sin(math.radians(p["sweep"]))
    ib = p["eb"] * (1 - math.cos(math.radians(p["sweep"]))) if p["eb"] else 0.0
    yb = p["eb"] * math.sin(math.radians(p["sweep"])) if p["eb"] else 0.0
    out = {
        "top": (add(o, mul(n, H)), u, v, n, ("rrect", w - 2 * it, d - 2 * it, max(r - it, 0))),
        "bottom": (o, u, v, neg(n), ("rrect", w - 2 * ib, d - 2 * ib, max(r - ib, 0))),
    }
    mid = add(o, mul(n, H / 2))
    side = ("rect", -(d / 2 - r), d / 2 - r, yb - H / 2, yt - H / 2)
    out["+u"] = (add(mid, mul(u, w / 2)), v, n, u, side)
    out["-u"] = (add(mid, mul(u, -w / 2)), v, n, neg(u), side)
    side = ("rect", -(w / 2 - r), w / 2 - r, yb - H / 2, yt - H / 2)
    out["+v"] = (add(mid, mul(v, d / 2)), u, n, v, side)
    out["-v"] = (add(mid, mul(v, -d / 2)), u, n, neg(v), side)
    if p["kind"] == "tray":
        t = p["t"]
        out["floor"] = (add(o, mul(n, t)), u, v, n, ("rrect", w - 2 * t, d - 2 * t, r - t))
    return out


def face_key(p, name):
    """The host's own name ("top", "+u", ...) for the face that looks along world `name`."""
    if name == "floor":
        return "floor"
    want = WORLD[name]
    for key, f in faces(p).items():
        if key != "floor" and all(abs(a - b) < 1e-9 for a, b in zip(f[3], want)):
            return key
    # a tilted host has no face looking exactly that way: take the one nearest it, within 45 degrees
    near = [(dot(f[3], want), key) for key, f in faces(p).items() if key != "floor"]
    if near and max(near)[0] > math.cos(math.radians(45)):
        return max(near)[1]
    raise ValueError(f"{p['id']} has no face {name}")


def face_of(p, name):
    return faces(p)[face_key(p, name)]


# --- containment --------------------------------------------------------------------------------
def inside(px, py, flat, m=MARGIN):
    if flat[0] == "sector":  # an annular sector round (0, 0): radii, then each end line, trimmed
        _, r0, r1, a0, a1, trim = flat
        a0, a1 = math.radians(a0), math.radians(a1)
        rr = math.hypot(px, py)
        return (r0 + m <= rr <= r1 - m and -px * math.sin(a0) + py * math.cos(a0) >= trim + m
                and px * math.sin(a1) - py * math.cos(a1) >= trim + m)
    if flat[0] == "rect":
        _, a0, a1, b0, b1 = flat
        return a0 + m <= px <= a1 - m and b0 + m <= py <= b1 - m
    _, W, D, R = flat
    W, D, R = W - 2 * m, D - 2 * m, max(R - m, 0.0)
    if abs(px) > W / 2 + 1e-12 or abs(py) > D / 2 + 1e-12:
        return False
    qx, qy = abs(px) - (W / 2 - R), abs(py) - (D / 2 - R)
    if qx <= 0 or qy <= 0:
        return True
    return math.hypot(qx, qy) <= R + 1e-12


def outline(a, b, W, D, R, n=6):
    """Points round a rounded rect centred (a, b)."""
    pts = []
    R = min(R, W / 2, D / 2)
    for cx, cy, a0 in ((W / 2 - R, D / 2 - R, 0), (-W / 2 + R, D / 2 - R, 90),
                       (-W / 2 + R, -D / 2 + R, 180), (W / 2 - R, -D / 2 + R, 270)):
        for k in range(n + 1):
            t = math.radians(a0 + 90 * k / n)
            pts.append((a + cx + R * math.cos(t), b + cy + R * math.sin(t)))
    return pts


# --- derivation ---------------------------------------------------------------------------------
def toy_parts():
    """The toy's list: the real list minus `tier: real`, with `keep` parts moved onto their
    dropped host's host, and each part's `toy` overrides merged."""
    dropped = {q["id"]: q for q in P.PARTS if q.get("tier") == "real"}
    out = []
    for q in P.PARTS:
        if q["id"] in dropped:
            continue
        q = dict(q)
        while q["host"] in dropped:
            if not q.get("keep"):
                raise ValueError(f"{q['id']} stands on {q['host']}, which the toy drops")
            h = dropped[q["host"]]
            q["host"], q["face"] = h["host"], h["face"]
        out.append(q)
    return out


def resolve(style):
    """[placed part] for `style` ("real" or "toy"), in parts.py order."""
    real = _place(P.PARTS, "real", None)
    if style == "real":
        return real
    byid = {p["id"]: p for p in real}
    rel = {}
    for q in toy_parts():  # each toy part's offset on its toy host's face, read off the real build
        p = byid[q["id"]]
        if q["host"] is None or q["host"] == p["host"]:
            rel[q["id"]] = (p.get("a", 0.0), p.get("b", 0.0))
        else:  # moved onto its dropped host's host: measure where it stands on that face
            c, U, V, N, _ = face_of(byid[q["host"]], q["face"])
            rel[q["id"]] = (dot(sub(p["o"], c), U), dot(sub(p["o"], c), V))
    # Round every toy part as hard as TOY_ROUND and TOY_RIM ask, then ease a host's corner and rim
    # radii (never below the real ones) until every part it carries sits wholly on its flat face.
    ease = {}
    for _ in range(80):
        placed = _place(toy_parts(), "toy", rel, ease)
        byp = {p["id"]: p for p in placed}
        tight = set()
        for line in check(placed, seats_only=True):
            child, host = line.split()[1], line.split(" overhangs ")[1].split("'")[0]
            key = face_key(byp[host], byp[child]["face"]) if child in byp and byp[child].get("face") else "top"
            # a part on a top face is squeezed by the host's rim; one on a side or a tray floor, by its corners
            want = "e" if key in ("top", "bottom") else "r"  # a tray floor narrows with its corners
            if key in ("+u", "-u", "+v", "-v") and child in byp:  # on a side: the rim caps its height
                c = byp[child]
                flat = c["flat"]
                if abs(c["a"]) + c["w"] / 2 + MARGIN <= flat[2] and not (flat[3] + MARGIN <= c["b"] - c["d"] / 2 and c["b"] + c["d"] / 2 <= flat[4] - MARGIN):
                    want = "e"
            h = byp[host]
            if want == "e" and h["e"] <= _real_of(host, "e") + 1e-9:
                want = "r"
            elif want == "r" and (h["kind"] not in ("box", "tray") or h["r"] <= _real_of(host, "r") + 1e-9):
                want = "e"
            tight.add((host, want))
        if not tight:
            return placed
        for h, want in tight:
            ease[(h, want)] = ease.get((h, want), 1.0) * 0.8
    return placed


def _real_of(pid, key):
    q = next(q for q in P.PARTS if q["id"] == pid)
    return q.get(key) or 0.0


def _place(spec, style, rel, ease=None):
    ease = ease or {}
    placed = {}
    out = []
    S = TOY_SCALE
    for q in spec:
        p = {k: q.get(k) for k in ("id", "kind", "row", "role", "host", "face", "tier", "ctl")}
        p["held"] = P.ROWS[q["row"]][2]
        toy = q.get("toy", {}) if style == "toy" else {}
        kind = toy.get("kind", q["kind"])
        p["kind"] = kind
        p["sweep"] = SWEEP[style]
        if q["host"] is None:
            o = q["origin"]
            if style == "toy":
                o = tuple(o[i] * S[i] for i in range(3))
            U, V, N = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
            c, a, b = o, 0.0, 0.0
            p["a"] = p["b"] = 0.0
        else:
            host = placed[q["host"]]
            c, U, V, N, flat = face_of(host, q["face"])
            U, V, N = (tuple(float(x) for x in t) for t in (U, V, N))
            if style == "real" or q["id"] not in rel:
                at = q.get("at") or q["path"][0]
                P3 = _on_face(at, c, N)
                a, b = dot(sub(P3, c), U), dot(sub(P3, c), V)
            else:
                a, b = rel[q["id"]]
                a, b = a * S[axis(U)], b * S[axis(V)]
                for ax, key in enumerate(("nudge_x", "nudge_y", "nudge_z")):
                    if key in toy:
                        if axis(U) == ax:
                            a += toy[key] * U[ax]
                        elif axis(V) == ax:
                            b += toy[key] * V[ax]
            p["a"], p["b"], p["flat"], p["face_c"] = a, b, flat, c
            o = sub(add(add(c, mul(U, a)), mul(V, b)), mul(N, SINK))
        p["u"], p["v"], p["n"] = U, V, N
        p["o"] = o
        tilt = toy.get("tilt", q.get("tilt"))
        if tilt:
            if q["host"] is None or kind in ("tube", "arc") or abs(N[1]) > 1e-9:
                raise ValueError(f"{q['id']}: a tilt turns a part in a side face's plane (a hosted box, "
                                 "tray, round or lathe on a face whose normal is horizontal)")
            p["u"], p["v"], p["spin"] = _spin(U, V, N, tilt)
            p["tilt"] = tilt

        if kind == "arc":
            _arc(p, q, toy, style, U, V, N, ease)
            placed[p["id"]] = p
            out.append(p)
            continue

        if kind == "tube":
            _tube(p, q, placed, style, rel)
            placed[p["id"]] = p
            out.append(p)
            continue

        # sizes, in the part's own (w along u, d along v, h along n)
        if kind in ("round", "lathe"):
            R, h = toy.get("R", q.get("R")), toy.get("h", q.get("h"))
            if style == "toy" and "R" not in toy and kind == "round":
                k = TOY_CTL if q.get("ctl") else min(S[axis(U)], S[axis(V)])
                R = R * k
            if style == "toy" and "h" not in toy and kind == "round":
                h = h * (TOY_CTL if q.get("ctl") else S[axis(N)])
            if kind == "lathe":
                R = max(r_ for r_, _ in q["profile"])
                h = max(y for _, y in q["profile"])
                p["profile"] = q["profile"]
            w = d = 2 * R
            r = R
        else:
            size = toy.get("size", q["size"])
            sz = [size[axis(U)], size[axis(V)], size[axis(N)]]
            reach = sz[2] if isinstance(sz[2], str) else None
            if style == "toy" and "size" not in toy:
                k = [TOY_CTL] * 3 if q.get("ctl") else [S[axis(U)], S[axis(V)], S[axis(N)]]
                sz = [s * kk if not isinstance(s, str) else s for s, kk in zip(sz, k)]
                if not q.get("ctl"):
                    sz[0], sz[1] = max(sz[0], TOY_MIN_SIDE), max(sz[1], TOY_MIN_SIDE)
                    if not isinstance(sz[2], str):
                        sz[2] = max(sz[2], TOY_MIN_H)
            w, d, h = sz
            if reach:
                tgt = placed[reach.split(":")[1]]
                tc = face_of(tgt, "+y" if N[1] < 0 else "-y")[0] if abs(N[1]) > 0.5 else None
                h = dot(sub(tc, add(c, mul(U, a))), N)
                p["reach"] = tgt["id"]
            r = toy.get("r", q.get("r"))
        e, eb, t = toy.get("e", q.get("e")), toy.get("eb", q.get("eb")), toy.get("t", q.get("t"))
        if style == "toy":
            lo = min(w, d)
            if kind == "box" or kind == "tray":
                r = toy.get("r", min(max(r, TOY_ROUND * lo), TOY_ROUND_MAX, lo / 2))
            if kind != "lathe" and "e" not in toy:
                cap = min(TOY_RIM_MAX, 0.8 * r, 0.45 * lo, 0.45 * h)
                if t:
                    cap = min(cap, 0.8 * t)
                e = min(max(e, TOY_RIM * min(lo, h)), cap)
            # the parts it carries need more flat face: ease toward the real radii
            if (q["id"], "r") in ease and kind in ("box", "tray"):
                r = min(max(q.get("r"), r * ease[(q["id"], "r")]), lo / 2)
            if (q["id"], "e") in ease:
                e = max(q.get("e") or 0.0, e * ease[(q["id"], "e")])
            if kind != "lathe" and "e" not in toy:
                e = min(e, 0.8 * r) if kind != "round" else e
            if eb and "eb" not in toy:
                eb = min(1.3 * e, 0.8 * r, 0.45 * lo, 0.45 * h - e)
        p.update(w=w, d=d, h=h, r=r, e=e or 0.0, eb=eb or 0.0, t=t, R=w / 2)
        p["H"] = h + SINK + (SINK if p.get("reach") else 0.0)
        placed[p["id"]] = p
        out.append(p)
    _share_grids(out)
    return out


def _on_face(at, c, N):
    """The world point a part's `at` names on a face: the face's centre fills the component that is
    None. On a tilted face (normal off the world axes) that component is solved so the point lies in
    the face's plane, so `at` keeps meaning world x and z on a sloped lane."""
    if square(N):
        return tuple(c[i] if at[i] is None else at[i] for i in range(3))
    j = [i for i in range(3) if at[i] is None]
    if len(j) != 1 or abs(N[j[0]]) < 1e-6:
        raise ValueError(f"at {at} must leave exactly the component along the face's normal None")
    pt = [c[i] if at[i] is None else at[i] for i in range(3)]
    pt[j[0]] -= dot(sub(pt, c), N) / N[j[0]]
    return tuple(pt)


def _spin(U, V, N, deg):
    """U and V turned `deg` degrees about N (right hand), and the turned pair in (U, V) coordinates."""
    cs, sn = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    u2, v2 = (add(mul(x, cs), mul(cross(N, x), sn)) for x in (U, V))
    return u2, v2, ((dot(u2, U), dot(u2, V)), (dot(v2, U), dot(v2, V)))


def _arc(p, q, toy, style, U, V, N, ease):
    """An annular sector round the vertical axis through p["o"]: sizes, rim, profile [(r, y)]."""
    if abs(abs(N[1]) - 1.0) > 1e-9:
        raise ValueError(f"{q['id']}: an arc stands on a top or bottom face (its axis is vertical)")
    S = TOY_SCALE
    a0, a1 = toy.get("arc", q["arc"])
    if not 0 < a1 - a0 <= 180:
        raise ValueError(f"{q['id']}: an arc sweeps more than 0 and at most 180 degrees")
    k, kn = min(S[axis(U)], S[axis(V)]), S[axis(N)]
    sw = math.radians(p["sweep"])
    prof = toy.get("profile", q.get("profile"))
    if prof is not None:  # a written cross-section: (r, y) from the seat, closed, counter-clockwise
        if style == "toy" and "profile" not in toy:
            prof = [(r * k, y * kn) for r, y in prof]
        prof = [(r, y + SINK if y > 0 else 0.0) for r, y in prof]  # the seat sinks SINK, as a box's does
        e = 0.0
        R0, R1, H = min(r for r, _ in prof), max(r for r, _ in prof), max(y for _, y in prof)
        h = H - SINK
        top = [r for r, y in prof if abs(y - H) < 1e-12]
        seat = [r for r, y in prof if y == 0.0]
        top_r, seat_r = (min(top), max(top), H), (min(seat), max(seat))
    else:
        R0, R1, h = (toy.get(n, q.get(n)) for n in ("R0", "R1", "h"))
        if style == "toy":
            R0, R1 = (toy[n] if n in toy else x * k for n, x in (("R0", R0), ("R1", R1)))
            h = toy["h"] if "h" in toy else h * kn
        e = toy.get("e", q.get("e")) or 0.0
        if style == "toy" and "e" not in toy:
            lo = R1 - R0
            e = min(max(e, TOY_RIM * min(lo, h)), TOY_RIM_MAX, 0.45 * lo, 0.45 * h)
            if (q["id"], "e") in ease:
                e = max(q.get("e") or 0.0, e * ease[(q["id"], "e")])
        H = h + SINK
        base = H - e * math.sin(sw)
        prof = [(R0, 0.0), (R1, 0.0)]
        if e:
            m = steps(e, p["sweep"])
            rim = [(e * (1 - math.cos(sw * j / m)), base + e * math.sin(sw * j / m)) for j in range(m + 1)]
            prof += [(R1 - i, y) for i, y in rim] + [(R0 + i, y) for i, y in reversed(rim)]
        else:
            prof += [(R1, H), (R0, H)]
        it = e * (1 - math.cos(sw))
        top_r, seat_r = (R0 + it, R1 - it, H), (R0, R1)
    p.update(R0=R0, R1=R1, arc=(a0, a1), trim=toy.get("trim", q.get("trim", 0.0)), prof=prof,
             top_r=top_r, seat_r=seat_r, w=2 * R1, d=2 * R1, h=h, r=R1, e=e, eb=0.0, t=None, R=R1, H=H)


def _share_grids(placed):
    """Arcs on one axis take one facet count (the most any of them needs), so their walls are
    parallel polygons and a 0.5 mm gap between two of them stays a gap."""
    arcs = [p for p in placed if p["kind"] == "arc"]
    for p in arcs:
        same = [q for q in arcs if _coaxial(p, q)]
        p["steps"] = max(steps(q["R1"], q["arc"][1] - q["arc"][0]) for q in same)


def _coaxial(p, q):
    """Two arcs on one vertical axis, measuring their angles from the same direction."""
    d = sub(q["o"], p["o"])
    off = sub(d, mul(p["n"], dot(d, p["n"])))
    return (abs(abs(dot(p["n"], q["n"])) - 1) < 1e-9 and dot(off, off) < 1e-18
            and dot(p["u"], q["u"]) > 1 - 1e-9 and dot(p["v"], q["v"]) > 1 - 1e-9)


def arc_angles(p, r):
    """The trimmed start and end angle (radians) of arc p at radius r."""
    a0, a1 = (math.radians(x) for x in p["arc"])
    da = math.asin(min(1.0, p["trim"] / r)) if p["trim"] else 0.0
    return a0 + da, a1 - da


def _fillet(pts, R):
    """A polyline with every inner corner rounded to radius R (an arc of `steps` facets)."""
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        A, B, C = pts[i - 1], pts[i], pts[i + 1]
        d1, d2 = sub(A, B), sub(C, B)
        l1, l2 = math.sqrt(dot(d1, d1)), math.sqrt(dot(d2, d2))
        d1, d2 = mul(d1, 1 / l1), mul(d2, 1 / l2)
        th = math.acos(max(-1.0, min(1.0, dot(d1, d2))))
        tl = R / math.tan(th / 2)
        room1 = l1 if i == 1 else l1 / 2  # an end segment serves one corner, an inner one two
        room2 = l2 if i == len(pts) - 2 else l2 / 2
        if tl > min(room1, room2) - 2e-3:  # keep 2 mm straight, so the end cap sits square
            raise ValueError(f"fillet {R} does not fit the corner at {B}")
        T1, T2 = add(B, mul(d1, tl)), add(B, mul(d2, tl))
        bis = add(d1, d2)
        bis = mul(bis, 1 / math.sqrt(dot(bis, bis)))
        C0 = add(B, mul(bis, R / math.sin(th / 2)))
        x0 = sub(T1, C0)
        x1 = sub(T2, C0)
        turn = math.pi - th
        m = steps(R, math.degrees(turn))
        for k in range(m + 1):
            s = k / m
            # slerp between x0 and x1 about C0
            ang = turn * s
            nx = mul(x0, math.sin(turn - ang) / math.sin(turn))
            ny = mul(x1, math.sin(ang) / math.sin(turn))
            out.append(add(C0, add(nx, ny)))
    out.append(pts[-1])
    clean = [out[0]]
    for q in out[1:]:
        if math.dist(q, clean[-1]) > 1e-6:
            clean.append(q)
    return clean


def _tube(p, q, placed, style, rel):
    R = q["R"]
    first = p["o"]
    end = placed[q["end"]]
    ec, eU, eV, eN, eflat = face_of(end, q["end_face"])
    last = q["path"][-1]
    pe = tuple(ec[i] if last[i] is None else last[i] for i in range(3))
    pe = sub(pe, mul(eN, SINK))
    mid = [tuple(x) for x in q["path"][1:-1]]
    pts = _fillet([first] + mid + [pe], q["bend"])
    p.update(R=R, pts=pts, end=q["end"], end_flat=eflat, end_ab=(dot(sub(pe, ec), eU), dot(sub(pe, ec), eV)),
             w=2 * R, d=2 * R, h=2 * R, r=R, e=0.0, eb=0.0, t=None, H=0.0)


# --- checks -------------------------------------------------------------------------------------
def aabbs(p):
    """Boxes round a part: one per segment for a tube (a U-shaped cable has an empty middle)."""
    if p["kind"] == "tube":
        out = []
        for s0, s1 in zip(p["pts"], p["pts"][1:]):
            out.append(([min(a, b) - p["R"] for a, b in zip(s0, s1)],
                        [max(a, b) + p["R"] for a, b in zip(s0, s1)]))
        return out
    return [aabb(p)]


def aabb(p):
    if p["kind"] == "arc":  # the mesh's own corners: both radii at every grid angle, bottom and top
        m = p.get("steps") or steps(p["R1"], p["arc"][1] - p["arc"][0])
        pts = []
        for r in (p["R0"], p["R1"]):
            t0, t1 = arc_angles(p, r)
            a0, a1 = (math.radians(x) for x in p["arc"])
            for t in [t0, t1] + [a0 + (a1 - a0) * k / m for k in range(1, m)]:
                for y in (0.0, p["H"]):
                    pts.append(add(add(p["o"], mul(p["u"], r * math.cos(t))),
                                   add(mul(p["v"], r * math.sin(t)), mul(p["n"], y))))
        cs = list(zip(*pts))
        return [min(c) for c in cs], [max(c) for c in cs]
    if p["kind"] == "tube":
        xs = list(zip(*p["pts"]))
        return [min(c) - p["R"] for c in xs], [max(c) + p["R"] for c in xs]
    corners = []
    for su in (-0.5, 0.5):
        for sv in (-0.5, 0.5):
            for sh in (0.0, 1.0):
                corners.append(add(add(add(p["o"], mul(p["u"], su * p["w"])), mul(p["v"], sv * p["d"])),
                                   mul(p["n"], sh * p["H"])))
    cs = list(zip(*corners))
    return [min(c) for c in cs], [max(c) for c in cs]


def check(placed, seats_only=False):
    """Problems: a seat not wholly on its host's flat face, or two unjoined parts whose boxes
    overlap."""
    bad = []
    for p in placed:
        if p["host"] is None:
            continue
        if p["kind"] == "arc":
            seats = [(p["a"], p["b"], p["flat"], None, None, None, p["host"])]
        elif p["kind"] == "tube":
            seats = [(p["a"], p["b"], p["flat"], 2 * p["R"], 2 * p["R"], p["R"], p["host"]),
                     (*p["end_ab"], p["end_flat"], 2 * p["R"], 2 * p["R"], p["R"], p["end"])]
        elif p["kind"] == "lathe":
            r0 = p["profile"][1][0]
            seats = [(p["a"], p["b"], p["flat"], 2 * r0, 2 * r0, r0, p["host"])]
        else:
            ib = p["eb"] * (1 - math.cos(math.radians(p["sweep"]))) if p["eb"] else 0.0
            seats = [(p["a"], p["b"], p["flat"], p["w"] - 2 * ib, p["d"] - 2 * ib, max(p["r"] - ib, 0), p["host"])]
        for a, b, flat, W, D, R, host in seats:
            if W is None:  # an arc: its seat sector, round its axis at (a, b)
                pts = [(a + r * math.cos(t), b + r * math.sin(t)) for r in p["seat_r"]
                       for t0, t1 in [arc_angles(p, r)] for t in (t0 + (t1 - t0) * j / 24 for j in range(25))]
            else:
                pts = outline(a, b, W, D, R)
                if p.get("spin") and host == p["host"]:  # a tilted part: its outline turns about its centre
                    (uu, uv), (vu, vv) = p["spin"]
                    pts = [(a + (x - a) * uu + (y - b) * vu, b + (x - a) * uv + (y - b) * vv) for x, y in pts]
            out = [pt for pt in pts if not inside(*pt, flat)]
            if out:
                bad.append(f"SEAT {p['id']} overhangs {host}'s face (e.g. {out[0][0]:+.4f}, {out[0][1]:+.4f} "
                           f"vs flat {tuple(round(x, 4) for x in flat[1:])})")
        if seats_only:
            continue
        if p["kind"] == "arc":
            sw = math.radians(p["sweep"])
            if p["top_r"][1] - p["top_r"][0] < 0.4e-3:
                bad.append(f"SHAPE {p['id']} rims {p['e']:.4f} meet across its width")
            if p["e"] * math.sin(sw) > p["H"]:
                bad.append(f"SHAPE {p['id']} rims taller than the part")
            if arc_angles(p, p["R0"])[1] <= arc_angles(p, p["R0"])[0]:
                bad.append(f"SHAPE {p['id']} trim {p['trim']:.4f} eats its inner edge")
        if p["kind"] in ("box", "tray", "round") and p["r"] > min(p["w"], p["d"]) / 2 + 1e-9:
            bad.append(f"SHAPE {p['id']} corner radius over half its side")
        if p["kind"] in ("box", "tray", "round"):
            it = p["e"] * (1 - math.cos(math.radians(p["sweep"])))
            if p["r"] - it < 0.4e-3 and p["kind"] != "round":
                bad.append(f"SHAPE {p['id']} rim {p['e']:.4f} eats its corner radius {p['r']:.4f}")
            if p["e"] * math.sin(math.radians(p["sweep"])) + (p["eb"] * math.sin(math.radians(p["sweep"])) if p["eb"] else 0) > p["H"]:
                bad.append(f"SHAPE {p['id']} rims taller than the part")
            if p["kind"] == "tray" and p["r"] - p["t"] < 4e-3:
                bad.append(f"SHAPE {p['id']} inner corner radius {p['r'] - p['t']:.4f} (corner radius less wall) collapses")
            if p["kind"] == "tray" and it >= p["t"]:
                bad.append(f"SHAPE {p['id']} rim wider than its wall")
    if seats_only:
        return bad
    joined = set()
    for p in placed:
        if p["host"]:
            joined.add(frozenset((p["id"], p["host"])))
        for k in ("reach", "end"):
            if p.get(k):
                joined.add(frozenset((p["id"], p[k])))
    boxes = {p["id"]: aabbs(p) for p in placed}
    byid = {p["id"]: p for p in placed}
    ids = [p["id"] for p in placed]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if frozenset((a, b)) in joined:
                continue
            pa, pb = byid[a], byid[b]
            if not (_square_part(pa) and _square_part(pb)):  # tilted or curved: oriented boxes
                ov = _overlap(pa, pb)
            else:
                ov = max(min(min(ha[k], hb[k]) - max(la[k], lb[k]) for k in range(3))
                         for la, ha in boxes[a] for lb, hb in boxes[b])
            if ov > 1e-4:
                bad.append(f"HIT {a} x {b}: boxes overlap {ov * 1000:.1f} mm")
    return bad


def _square_part(p):
    """True when the old world-axis boxes bound the part exactly: no tilt, no arc."""
    return p["kind"] == "tube" or (p["kind"] != "arc" and all(square(x) for x in (p["u"], p["v"], p["n"])))


def obbs(p):
    """Oriented boxes round a part, each (centre, three unit axes, three half sizes). An arc gets
    one per facet chunk, each holding the chunk's annulus and its chords; a tube one per segment."""
    if p["kind"] == "tube":
        out = []
        for lo, hi in aabbs(p):
            out.append((tuple((x + y) / 2 for x, y in zip(lo, hi)), ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                        tuple((y - x) / 2 for x, y in zip(lo, hi))))
        return out
    o, u, v, n, H = p["o"], p["u"], p["v"], p["n"], p["H"]
    if p["kind"] == "arc":
        a0, a1 = (math.radians(x) for x in p["arc"])
        m = p.get("steps") or steps(p["R1"], p["arc"][1] - p["arc"][0])
        half = (a1 - a0) / m / 2
        rin = p["R0"] * math.cos(half)
        out = []
        for j in range(m):
            t = a0 + (2 * j + 1) * half
            er = add(mul(u, math.cos(t)), mul(v, math.sin(t)))
            et = add(mul(u, -math.sin(t)), mul(v, math.cos(t)))
            c = add(add(o, mul(er, (rin + p["R1"]) / 2)), mul(n, H / 2))
            out.append((c, (er, et, n), ((p["R1"] - rin) / 2, p["R1"] * math.sin(half), H / 2)))
        return out
    return [(add(o, mul(n, H / 2)), (u, v, n), (p["w"] / 2, p["d"] / 2, H / 2))]


def _sat(A, B):
    """How deep two oriented boxes overlap: the least overlap over the separating axes (the 6 face
    normals and the 9 edge crosses). At or under 0 they are apart."""
    (ca, xa, ha), (cb, xb, hb) = A, B
    d = sub(cb, ca)
    axes = list(xa) + list(xb)
    for x in xa:
        for y in xb:
            c = cross(x, y)
            ln = math.sqrt(dot(c, c))
            if ln > 1e-6:
                axes.append(mul(c, 1 / ln))
    best = math.inf
    for L in axes:
        ra = sum(h * abs(dot(x, L)) for x, h in zip(xa, ha))
        rb = sum(h * abs(dot(x, L)) for x, h in zip(xb, hb))
        best = min(best, ra + rb - abs(dot(d, L)))
        if best <= 0:
            return best
    return best


def _overlap(pa, pb):
    """The overlap of two parts, one of them tilted or curved. Two arcs on one axis compare their
    radius, height and angle ranges exactly (their walls share a facet grid); anything else, oriented
    boxes."""
    if pa["kind"] == "arc" and pb["kind"] == "arc" and _coaxial(pa, pb):
        radial = min(pa["R1"], pb["R1"]) - max(pa["R0"], pb["R0"])
        sa, sb = (sorted((p["o"][1], p["o"][1] + p["n"][1] * p["H"])) for p in (pa, pb))
        tall = min(sa[1], sb[1]) - max(sa[0], sb[0])
        turn = math.radians(min(pa["arc"][1], pb["arc"][1]) - max(pa["arc"][0], pb["arc"][0]))
        return min(radial, tall, turn * max(pa["R0"], pb["R0"]))
    return max(_sat(A, B) for A in obbs(pa) for B in obbs(pb))


def size_of(placed):
    lo = [min(aabb(p)[0][k] for p in placed) for k in range(3)]
    hi = [max(aabb(p)[1][k] for p in placed) for k in range(3)]
    return lo, hi


if __name__ == "__main__":
    style = sys.argv[-1] if sys.argv[-1] in ("real", "toy") else "real"
    placed = resolve(style)
    bad = check(placed)
    for line in bad:
        print(line)
    lo, hi = size_of(placed)
    print(f"{style}: {len(placed)} parts, {len(bad)} problems, extent "
          f"x {lo[0]:.3f}..{hi[0]:.3f} y {lo[1]:.3f}..{hi[1]:.3f} z {lo[2]:.3f}..{hi[2]:.3f}")
    sys.exit(1 if bad else 0)
