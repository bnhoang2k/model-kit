"""Linear blend skinning in numpy, checked by hand-worked numbers and against saved samples."""

from pathlib import Path

import numpy as np
import pytest

from model_kit.core.errors import ManifestError, Unmeasurable
from model_kit.core.pose import skin
from model_kit.core.rig import RigData, pose_world
from model_kit.core.scene import ClipSamples, RigRest, load_scene
from model_kit.core.tolerances import FLOAT32_EPS, NODE_FLOAT32_ULPS, NODE_OF_PIECE_SIZE

REPO = Path(__file__).resolve().parents[3]

# The scene folders whose rest file holds no skin weights yet, so the saved samples cannot be
# skinned. None: every bar's rest file holds its weights.
AWAITING_WEIGHTS: set[str] = set()


def _translate(x, y, z) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def _rig(rest: dict[str, np.ndarray], parent: dict[str, str | None],
         world: np.ndarray | None = None) -> RigData:
    return RigData(order=tuple(rest), parent=parent, rest=rest,
                   world=np.eye(4) if world is None else world, effector_mid={})


def _body(rig: RigData, shell, groups, w, j, part: str = "p") -> RigRest:
    """A one-part body: `shell` (V, 3) is the rest shell, `w`/`j` the (V, K) skin."""
    shell = np.asarray(shell, dtype=np.float32)[None]
    v = shell.shape[1]
    return RigRest(
        shells={part: shell}, faces={part: np.zeros((0, 3), np.int32)},
        labels={part: np.zeros(v, np.int32)}, vertex_groups={part: tuple(groups)},
        weights={part: np.asarray(w, dtype=np.float32)},
        weight_groups={part: np.asarray(j, dtype=np.int32)})


QUARTER_Z = (0.0, 0.0, np.pi / 2)  # a quarter turn about Z, as pose euler angles
ZERO = (0.0, 0.0, 0.0)


def test_rigid_part_follows_its_bone():
    # The bone's head sits at x = 1; a quarter turn about Z carries (2, 0, 0) to (1, 1, 0).
    rig = _rig({"b": _translate(1, 0, 0)}, {"b": None})
    rest = _body(rig, [[2, 0, 0]], ["b"], [[1.0]], [[0]])
    posed = skin(rest, rig, pose_world(rig, {"b": (QUARTER_Z, ZERO)}))
    np.testing.assert_allclose(posed["p"], [[1, 1, 0]], atol=1e-12)


def test_rigid_part_follows_a_bone_through_the_world_matrix():
    # The world matrix doubles every length. Head at world x = 2, the vertex at world x = 4:
    # the turn about the head carries it to (2, 2, 0).
    rig = _rig({"b": _translate(1, 0, 0)}, {"b": None}, world=np.diag([2.0, 2.0, 2.0, 1.0]))
    rest = _body(rig, [[4, 0, 0]], ["b"], [[1.0]], [[0]])
    posed = skin(rest, rig, pose_world(rig, {"b": (QUARTER_Z, ZERO)}))
    np.testing.assert_allclose(posed["p"], [[2, 2, 0]], atol=1e-12)


def test_rest_pose_gives_the_rest_shell_back():
    # A chain under a moved, turned, scaled world, vertices on every bone and between them.
    turn = np.eye(4)
    turn[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    world = _translate(0.3, -0.2, 0.5) @ turn @ np.diag([1.5, 1.5, 1.5, 1.0])
    rig = _rig({"a": _translate(0, 0, 0.1), "b": _translate(0.2, 0, 0.7) @ turn,
                "c": _translate(0.4, 0.1, 1.2)},
               {"a": None, "b": "a", "c": "b"}, world=world)
    shell = [[0.5, 0.1, 0.3], [-0.2, 0.4, 0.9], [0.7, -0.3, 1.1]]
    w = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.2, 0.3, 0.5]]
    j = [[0, 0, 0], [0, 1, 0], [2, 0, 1]]
    rest = _body(rig, shell, ["a", "b", "c"], w, j)
    posed = skin(rest, rig, pose_world(rig, {}))
    np.testing.assert_allclose(posed["p"], np.asarray(shell, np.float32), rtol=0, atol=1e-9)


def test_child_bone_carries_a_vertex_with_its_parent():
    rig = _rig({"root": _translate(0, 0, 0), "arm": _translate(1, 0, 0)},
               {"root": None, "arm": "root"})
    rest = _body(rig, [[2, 0, 0]], ["arm"], [[1.0]], [[0]])
    # Turning the root a quarter about Z swings the arm's head to (0, 1, 0) and the vertex to
    # (0, 2, 0).
    posed = skin(rest, rig, pose_world(rig, {"root": (QUARTER_Z, ZERO)}))
    np.testing.assert_allclose(posed["p"], [[0, 2, 0]], atol=1e-12)


def test_two_bone_vertex_blends_linearly():
    # Two bones at the origin. `a` slides 4 along x; `b` slides 8 along y. A vertex at (1, 1, 1)
    # weighing 0.25 to a and 0.75 to b lands on the weighted sum of its two images.
    rig = _rig({"a": np.eye(4), "b": np.eye(4)}, {"a": None, "b": None})
    rest = _body(rig, [[1, 1, 1]], ["a", "b"], [[0.25, 0.75]], [[0, 1]])
    pose = {"a": (ZERO, (4, 0, 0)), "b": (ZERO, (0, 8, 0))}
    posed = skin(rest, rig, pose_world(rig, pose))
    np.testing.assert_allclose(posed["p"], [[0.25 * 5 + 0.75 * 1, 0.25 * 1 + 0.75 * 9, 1]],
                               atol=1e-12)


def test_blend_of_a_turn_and_a_slide_is_not_a_turn():
    # `a` turns a quarter about Z, which sends (1, 0, 0) to (0, 1, 0); `b` slides 4 along z.
    # Half and half gives (0.5, 0.5, 2): inside the unit circle, the collapse of linear
    # blending, where a turn of the mixed matrix would stay on the circle.
    rig = _rig({"a": np.eye(4), "b": np.eye(4)}, {"a": None, "b": None})
    rest = _body(rig, [[1, 0, 0]], ["a", "b"], [[0.5, 0.5]], [[0, 1]])
    pose = {"a": (QUARTER_Z, ZERO), "b": (ZERO, (0, 0, 4))}
    posed = skin(rest, rig, pose_world(rig, pose))
    np.testing.assert_allclose(posed["p"], [[0.5, 0.5, 2.0]], atol=1e-12)


def test_weights_normalised():
    rig = _rig({"a": np.eye(4), "b": np.eye(4)}, {"a": None, "b": None})
    pose = pose_world(rig, {"a": (ZERO, (4, 0, 0)), "b": (ZERO, (0, 8, 0))})
    half = skin(_body(rig, [[1, 1, 1]], ["a", "b"], [[0.5, 0.5]], [[0, 1]]), rig, pose)
    small = skin(_body(rig, [[1, 1, 1]], ["a", "b"], [[0.1, 0.1]], [[0, 1]]), rig, pose)
    np.testing.assert_allclose(small["p"], half["p"], atol=1e-12)
    np.testing.assert_allclose(half["p"], [[3.0, 5.0, 1.0]], atol=1e-12)


def test_padding_slots_add_nothing():
    # Bone `a` flies far away. The vertex weighs 1 to `b` and its second slot is padding (weight
    # 0, index 0, which names `a`): it must stay with `b`.
    rig = _rig({"a": np.eye(4), "b": np.eye(4)}, {"a": None, "b": None})
    rest = _body(rig, [[1, 1, 1]], ["a", "b"], [[1.0, 0.0]], [[1, 0]])
    pose = {"a": (ZERO, (100, 0, 0)), "b": (ZERO, (0, 0, 2))}
    posed = skin(rest, rig, pose_world(rig, pose))
    np.testing.assert_allclose(posed["p"], [[1, 1, 3]], atol=1e-12)


def test_small_weight_still_moves_the_vertex():
    # No floor: a 0.0005 share (under the bind step's 0.001 floor) of a bone that moves 1000
    # along x moves the vertex by 0.5, as linear blend skinning does.
    rig = _rig({"a": np.eye(4), "b": np.eye(4)}, {"a": None, "b": None})
    rest = _body(rig, [[0, 0, 0]], ["a", "b"], [[0.9995, 0.0005]], [[0, 1]])
    posed = skin(rest, rig, pose_world(rig, {"b": (ZERO, (1000, 0, 0))}))
    np.testing.assert_allclose(posed["p"], [[0.5, 0, 0]], atol=1e-9)


def test_vertex_with_no_weight_is_a_manifest_problem():
    rig = _rig({"a": np.eye(4)}, {"a": None})
    rest = _body(rig, [[1, 0, 0], [2, 0, 0]], ["a"], [[1.0], [0.0]], [[0], [0]])
    with pytest.raises(ManifestError, match=r"part 'p'.*vertex 1"):
        skin(rest, rig, pose_world(rig, {}))


def test_part_without_weights_is_unmeasurable():
    rig = _rig({"a": np.eye(4)}, {"a": None})
    rest = _body(rig, [[1, 0, 0]], ["a"], [[1.0]], [[0]])
    rest.weights.clear()
    rest.weight_groups.clear()
    with pytest.raises(Unmeasurable, match="'p'"):
        skin(rest, rig, pose_world(rig, {}))
    assert issubclass(Unmeasurable, ManifestError)


def test_one_part_without_weights_names_that_part():
    rig = _rig({"a": np.eye(4)}, {"a": None})
    rest = _body(rig, [[1, 0, 0]], ["a"], [[1.0]], [[0]], part="has")
    rest.shells["bare"] = rest.shells["has"].copy()
    rest.faces["bare"] = rest.faces["has"].copy()
    rest.labels["bare"] = rest.labels["has"].copy()
    rest.vertex_groups["bare"] = ("a",)
    with pytest.raises(Unmeasurable, match="'bare'"):
        skin(rest, rig, pose_world(rig, {}))


def test_posed_bone_missing_from_world_raises_by_name():
    rig = _rig({"a": np.eye(4), "b": np.eye(4)}, {"a": None, "b": None})
    rest = _body(rig, [[1, 0, 0]], ["a", "b"], [[0.5, 0.5]], [[0, 1]])
    world = pose_world(rig, {})
    del world["b"]
    with pytest.raises(ManifestError, match="'b'"):
        skin(rest, rig, world)


# --- saved samples ----------------------------------------------------------------------------


def _violations(rest: RigRest, rig: RigData, samples: ClipSamples) -> list[str]:
    """Where skinning a sample from its `b:` matrices lands farther from the saved shell than
    export hygiene's node limit: `NODE_OF_PIECE_SIZE` times the body's size (the diagonal of the
    box of all its rest shells), plus `NODE_FLOAT32_ULPS` float32 epsilons of the point's
    distance from the origin."""
    every = np.concatenate([s[0] for s in rest.shells.values()]).astype(float)
    size = float(np.linalg.norm(every.max(axis=0) - every.min(axis=0)))
    out = []
    for i in range(len(samples.times)):
        posed = skin(rest, rig, {b: m[i] for b, m in samples.bones.items()})
        for part, shell in posed.items():
            saved = samples.shells[part][i].astype(float)
            limit = (NODE_OF_PIECE_SIZE * size
                     + NODE_FLOAT32_ULPS * FLOAT32_EPS * np.linalg.norm(saved, axis=1))
            off = np.linalg.norm(shell - saved, axis=1)
            out += [f"{samples.clip} sample {i} part {part} vertex {v}: {off[v]:.3g} > "
                    f"{limit[v]:.3g}" for v in np.nonzero(off > limit)[0]]
    return out


def _synthetic_samples(rig: RigData, rest: RigRest, shell: np.ndarray) -> ClipSamples:
    """One sample at the pose `{"b": quarter turn}`, holding the shell the caller worked out."""
    world = pose_world(rig, {"b": (QUARTER_Z, ZERO)})
    return ClipSamples(
        clip="synthetic", times=np.array([1.0]), groups={},
        bones={b: m[None] for b, m in world.items()}, props={},
        shells={"p": np.asarray(shell, np.float32)[None]}, faces=rest.faces, labels=rest.labels)


def test_saved_sample_comparison_passes_right_shells_and_names_wrong_ones():
    rig = _rig({"b": _translate(1, 0, 0)}, {"b": None})
    rest = _body(rig, [[2, 0, 0], [3, 0, 0], [1, 5, 0]], ["b"], [[1.0]] * 3, [[0]] * 3)
    # Worked by hand: a quarter turn about the head (1, 0, 0) sends (2, 0, 0) to (1, 1, 0),
    # (3, 0, 0) to (1, 2, 0) and (1, 5, 0) to (-4, 0, 0).
    right = [[1, 1, 0], [1, 2, 0], [-4, 0, 0]]
    assert _violations(rest, rig, _synthetic_samples(rig, rest, right)) == []
    wrong = [[1, 1, 0], [1, 2.01, 0], [-4, 0, 0]]  # one vertex off by 0.01 of a 5.4 body
    found = _violations(rest, rig, _synthetic_samples(rig, rest, wrong))
    assert len(found) == 1 and "vertex 1" in found[0], found


def test_skin_matches_saved_samples():
    """Every scene folder of the repo whose rest file holds skin weights: skinning each saved
    sample from its bone matrices gives the saved shells. A folder whose rest file has none is
    recorded as awaiting weights, and the awaiting set must equal `AWAITING_WEIGHTS`."""
    awaiting, compared, problems = set(), 0, []
    for folder in sorted((REPO / "bars").glob("*/scene")):
        name = folder.relative_to(REPO).as_posix()
        ctx = load_scene(folder)
        for key, rest in ctx.rests.items():
            if not rest.weights:
                awaiting.add(name)
                continue
            here = 0
            for clip, rig_key in ctx.rig_of_clip.items():
                if rig_key == key and clip in ctx.samples:
                    here += 1
                    problems += [f"{name}: {line}" for line in
                                 _violations(rest, ctx.rigs[key], ctx.samples[clip])]
            if here == 0:
                problems.append(f"{name}: rig {key} has weights but no saved clip was skinned "
                                "and compared")
            compared += here
    print(f"awaiting weights: {sorted(awaiting)}; clips skinned and compared: {compared}")
    assert problems == []
    assert awaiting == AWAITING_WEIGHTS
