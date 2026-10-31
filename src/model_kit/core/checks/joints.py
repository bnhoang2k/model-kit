"""The joint helper that `receiver` and `connectors` judge through: does a guest part seat in the
host part a mount names?

The manifest gives a point (`Mount.point`, or an end's point) and the parts (`Mount.part` for the
host, `End.plug` for the guest). The point only locates the joint, within `locate`; it is not the
bearing surface. The helper reads the two meshes, both in one frame, and answers one of three:

- `off_host`: the joint is not where the manifest says, or the guest does not rest on the host
  part. The point lies further than `locate` from the guest, or from the host part (a point inside
  a part's solid is no distance from it), or the two
  have no contact (`bearing.contact_patch`, within `locate`, a part allowed to sit `locate` into
  its host). `locate` is the guest piece's contact length (`sight.contact_length`), so a plug
  pulled out by more than that, its body off the face, is off its host. The `receiver` check's
  touch rule is this one.
- `no_bore`: only for a joint that needs a hole (`bore_axis` given, the mount a `power` or
  `signal` port). A peg seats only where a hole's wall surrounds it. Nothing of the guest reaches
  float noise past the mouth, so it stands on the face; or a peg is not ringed by the host's wall
  along the part of it past the mouth, or lies in the host's material, not in a void. No
  insertion depth is judged: a peg that is a little past the mouth is seated, and a pin
  through the joint point stays seated while it is in the bore. The mouth is the host's
  outermost face near the joint: the highest point along the axis of the host triangles within
  `locate` of the point (the manifest point is only known to `locate`, so it is not the mouth).
- `seated`: neither.

A peg is a piece of the guest that reaches past the mouth: the triangles with a vertex more than
float noise (`WELD_OF_PIECE_SIZE` times the size of the two parts) below it, joined by shared
corners (the blades of a plug, not its body). Each is read on
its own centre line, from the far end of the host (or its own tip) up to the mouth. At a spread of
points along that line the host's wall must ring it (`bearing.ringed`: rays all round, at right
angles to the axis, meet the host within the peg's own radius plus `locate`) and the point must not
lie inside the host's solid. The host must be a closed solid to tell a void from material; the
caller checks that in the host's own frame (a turn blurs the exact corners `geom.is_closed` matches)
and leaves the joint `Unmeasured` when it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model_kit.core import geom
from model_kit.core.checks import bearing
from model_kit.core.checks.pieces import PartMesh
from model_kit.core.scene import SceneCtx
from model_kit.core.tolerances import ENGAGE_SAMPLES, WELD_OF_PIECE_SIZE

SEATED, NO_BORE, OFF_HOST = "seated", "no_bore", "off_host"
PORT_KINDS = ("power", "signal")
"""The mount kinds that take a connector end, and so have a hole round the plug. A `rest` mount
declares that it has none."""

PRINCIPLE = ("A peg seats only where a hole's wall surrounds it, and a guest rests on the part "
             "that holds it.")
SOURCE = (
    "Boothroyd, Dewhurst and Knight, 'Product Design for Manufacture and Assembly', 2nd ed. "
    "(Marcel Dekker, 2002), section 3.12, pp. 106-108 and Fig. 3.24: insertion is modelled as a "
    "peg of diameter d inside a hole of diameter D, with a clearance between the peg and the "
    "hole's wall. The kit's rule (a peg seats only where a hole's wall surrounds it) is that "
    "geometry stated as a test; the book does not state the rule in words. The second half of the principle (a "
    "guest rests on the part that holds it) is the definition of a rest mount, not from the "
    "book.")


@dataclass(frozen=True)
class JointResult:
    status: str  # SEATED, NO_BORE or OFF_HOST
    detail: str

    @property
    def seated(self) -> bool:
        return self.status == SEATED


def _mm(ctx: SceneCtx, length: float) -> str:
    return f"{length * 1000.0 / ctx.style.units_per_metre:.2f} mm"


def _distance(tris: np.ndarray, point: np.ndarray) -> float:
    """How far a point is from a part: 0 inside its solid, else the distance to its surface."""
    if geom.inside(tris, point[None, :])[0]:
        return 0.0
    return float(geom.closest_points(tris, point[None, :]).dist[0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if not n > geom.NO_DIRECTION:
        raise ValueError(f"a bore axis needs a direction, got {tuple(v)}")
    return v / n


def _mouth(host: np.ndarray, point: np.ndarray, axis: np.ndarray, locate: float) -> float | None:
    """Where the host's face is along the axis: the highest point of the host triangles within
    `locate` of the joint point. None when no face is that near (the point lies inside the
    material, deeper than `locate`)."""
    near = host[geom.face_distances(host, point)[0] <= locate]
    return float((near @ axis).max()) if len(near) else None


def _pegs(tris: np.ndarray, axis: np.ndarray, mouth: float, noise: float) -> list[np.ndarray]:
    """The triangles of the guest that reach more than `noise` below the mouth, split into the
    groups that share corners: one array of triangles per peg."""
    deep = tris[(tris @ axis).min(axis=1) < mouth - noise]
    if not len(deep):
        return []
    keys = np.round(deep.reshape(-1, 3) / geom.NO_DIRECTION).astype(np.int64).tolist()
    corner: dict[tuple, int] = {}
    ids = np.array([corner.setdefault(tuple(k), len(corner)) for k in keys]).reshape(-1, 3)
    parent = list(range(len(corner)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in ids:
        parent[find(int(b))] = find(int(a))
        parent[find(int(c))] = find(int(a))
    groups: dict[int, list[int]] = {}
    for n, a in enumerate(ids[:, 0]):
        groups.setdefault(find(int(a)), []).append(n)
    return [deep[rows] for rows in groups.values()]


def _held(peg: np.ndarray, host: PartMesh, axis: np.ndarray, mouth: float, locate: float
          ) -> tuple[int, int, int]:
    """(points ringed in a void, points inside the host's solid, points) along a peg's centre
    line, between the host's far end (or the peg's tip) and the mouth."""
    corners = np.unique(peg.reshape(-1, 3), axis=0)
    centre = corners.mean(axis=0)
    along = corners @ axis
    lateral = corners - np.outer(along, axis)
    line = centre - (centre @ axis) * axis
    radius = float(np.linalg.norm(lateral - line, axis=1).max())
    lo = max(float(along.min()), float((host.tris @ axis).min()))
    hi = min(float(along.max()), mouth)
    if not hi > lo:
        return 0, 0, 0
    at = lo + (np.arange(ENGAGE_SAMPLES) + 0.5) * (hi - lo) / ENGAGE_SAMPLES
    points = line + np.outer(at, axis)
    ring = bearing.ringed(host.tris, points, axis, radius + locate)
    inside = geom.inside(host.tris, points)
    return int((ring & ~inside).sum()), int(inside.sum()), len(points)


def seat(ctx: SceneCtx, *, guest: PartMesh, host: PartMesh, point, locate: float,
         bore_axis) -> JointResult:
    """Whether `guest` seats in `host` at `point`. Both meshes and the point are in one frame.
    `locate` is how far the manifest point may lie from the joint. `bore_axis` is the way the
    mount points out of the host, toward the guest, for a joint that needs a hole round the
    guest; None for a joint with none (a rest). It has no default, so a caller says whether the
    joint needs a hole. A bore is judged against a closed host: the caller
    has checked it."""
    point = np.asarray(point, dtype=float)
    reach = _distance(guest.tris, point)
    if reach > locate:
        return JointResult(OFF_HOST, f"the joint point is {_mm(ctx, reach)} from {guest.name}, "
                                     f"over the {_mm(ctx, locate)} it may be")
    held = _distance(host.tris, point)
    if held > locate:
        return JointResult(OFF_HOST, f"the joint point is {_mm(ctx, held)} from the host part "
                                     f"{host.name}, over the {_mm(ctx, locate)} it may be")
    on_guest, _ = bearing.contact_patch(guest.tris, host.tris, locate, locate)
    if not on_guest.any():
        gap = geom.surface_gap(guest.tris, host.tris, 4 * locate)
        far = (f"{_mm(ctx, gap)} from" if np.isfinite(gap) and gap > locate
               else "in no contact with")
        return JointResult(OFF_HOST, f"{guest.name} is {far} the host part {host.name} (within "
                                     f"{_mm(ctx, locate)}): it does not rest on it")
    if bore_axis is None:
        return JointResult(SEATED, f"{guest.name} rests on {host.name}")

    axis = _unit(bore_axis)
    every = np.concatenate([guest.tris.reshape(-1, 3), host.tris.reshape(-1, 3)])
    noise = WELD_OF_PIECE_SIZE * float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    mouth = _mouth(host.tris, point, axis, locate)
    if mouth is None:
        return JointResult(OFF_HOST, f"the joint point lies inside the material of {host.name}, "
                                     f"further than {_mm(ctx, locate)} from any face of it")
    pegs = _pegs(guest.tris, axis, mouth, noise)
    if not pegs:
        return JointResult(NO_BORE, f"nothing of {guest.name} reaches past the mouth of the "
                                    f"joint: it stands on the face of {host.name}, in no hole")
    worst = (1.0, 0, 0)
    for peg in pegs:
        ok, inside, total = _held(peg, host, axis, mouth, locate)
        share = ok / total if total else 0.0
        if share < worst[0]:
            worst = (share, inside, total)
    if worst[0] < 1.0:
        where = (f"{worst[1]} of {worst[2]} points of a peg lie in the material of {host.name}, "
                 "not in a hole" if worst[1] else
                 f"the wall of {host.name} does not ring a peg of {guest.name} all round")
        return JointResult(NO_BORE, f"{guest.name} stands on a face with no bore round it: {where}")
    return JointResult(SEATED, f"{guest.name} stands in a bore of {host.name}")
