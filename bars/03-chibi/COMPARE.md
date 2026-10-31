# Bar 03: the rebuild against an earlier hand-built version

This note records what the chibi bar measures and how it was checked. The bar is built fresh from `chibi.pack.json` by `build.py` (the "rebuild" below). It was compared with an earlier hand-built version of the same figure (the "source" below), which this repository does not hold, so the comparison cannot be rerun from here. The numbers were measured in Blender on both models, with each limit written as a named constant in the measuring code. Every judged number passes. No shape number was tuned to reach it. The eyes are judged on their footprint on the face only, because they are printed on the head here and are lenses in the source (see the last section).

What is compared, and how:

- Both models are read as the evaluated mesh in glTF axes, every mesh skinned to the armature, with the hand prop (`prop.R`) left out of both. The source's outline (SOLIDIFY) modifiers are off. The eyes are their own group, `eyes`, on both sides: lenses in the source, decals in the rebuild. A vertex belongs to its strongest deform group; a triangle belongs to a group when all three corners do.
- The height for the limits is the source's: 0.99805 (rebuild 0.99810). Limits: box 0.00998 (0.01 of height), gap p99 0.00499 (0.005), gap max 0.00998 (0.01).
- Points, source / rebuild: 15202 / 15100. Triangles: 30364 / 30100. The ankle groups (the shins), the body and the eyes differ in vertex count by design (see the first table), and the forearms and hands by one point each where the thumb's split into triangles moved, so shapes are compared, not vertices.
- The source is posed through `pose_arrays` only. The forward-kinematics check (`key_action`, then `export_rig` and `export_samples`, with no `ManifestError`) runs on the rebuild only; the source is not exported.
- Posed comparisons turn the NLA off and reset the pose before each clip, on both sides, and read every 4th frame plus the clip's last frame.

## Rest box drift, per group (limit 0.00998)

| group | drift | limit | points source / rebuild |
|---|---|---|---|
| ankle.L | 0.00012 | 0.00998 | 1397 / 1627 |
| ankle.R | 0.00012 | 0.00998 | 1397 / 1627 |
| arm.L | 0.00000 | 0.00998 | 1681 / 1681 |
| arm.R | 0.00000 | 0.00998 | 1681 / 1681 |
| body | 0.00001 | 0.00998 | 762 / 842 |
| eyes (on the face: x, y) | 0.00095 | 0.00998 | 964 / 322 |
| forearm.L | 0.00315 | 0.00998 | 632 / 633 |
| forearm.R | 0.00315 | 0.00998 | 632 / 633 |
| hand.L | 0.00353 | 0.00998 | 1665 / 1664 |
| hand.R | 0.00353 | 0.00998 | 1665 / 1664 |
| head | 0.00000 | 0.00998 | 762 / 762 |
| knee.L | 0.00000 | 0.00998 | 982 / 982 |
| knee.R | 0.00000 | 0.00998 | 982 / 982 |

## Silhouette overlap (IoU), tilt 30, 512 px, half extent 0.62, centre (0, 0.45, 0)

| turn | IoU | minimum |
|---|---|---|
| 0 | 0.99934 | 0.97 |
| 45 | 0.99948 | 0.97 |
| 90 | 0.99829 | 0.97 |

## Surface gap at rest, per group (the eyes are in the last section)

| group | p99 | p99 limit | max | max limit | triangles source / rebuild |
|---|---|---|---|---|---|
| ankle.L | 0.00044 | 0.00499 | 0.00072 | 0.00998 | 2790 / 3250 |
| ankle.R | 0.00045 | 0.00499 | 0.00068 | 0.00998 | 2790 / 3250 |
| arm.L | 0.00000 | 0.00499 | 0.00001 | 0.00998 | 3304 / 3304 |
| arm.R | 0.00000 | 0.00499 | 0.00001 | 0.00998 | 3304 / 3304 |
| body | 0.00220 | 0.00499 | 0.00332 | 0.00998 | 1520 / 1680 |
| forearm.L | 0.00347 | 0.00499 | 0.00527 | 0.00998 | 1151 / 1153 |
| forearm.R | 0.00347 | 0.00499 | 0.00527 | 0.00998 | 1151 / 1153 |
| hand.L | 0.00357 | 0.00499 | 0.00472 | 0.00998 | 3271 / 3269 |
| hand.R | 0.00357 | 0.00499 | 0.00472 | 0.00998 | 3271 / 3269 |
| head | 0.00000 | 0.00499 | 0.00000 | 0.00998 | 1520 / 1520 |
| knee.L | 0.00000 | 0.00499 | 0.00000 | 0.00998 | 1960 / 1960 |
| knee.R | 0.00000 | 0.00499 | 0.00000 | 0.00998 | 1960 / 1960 |

## Bones

| number | worst | limit |
|---|---|---|
| head and tail position, any bone | 3e-08 | 1e-6 |
| roll (EditBone.roll), any bone | 0 | 1e-6 |
| bone matrix, any bone | 3e-08 | 1e-6 |
| armature object matrix | 0 | 1e-6 |
| names, parents, deform flags that differ | 0 | 0 |

All 16 bones match in name, parent and deform flag.

## Every clip, worst posed box drift (limit 0.00998)

Each of the source's 19 actions is read with `action_keys` (the hand prop's scale left out), keyed onto the rebuild with `key_action`, and exported through `export_rig` and `export_samples`: the forward-kinematics contract holds at every sample of every clip (no `ManifestError`).

| clips | frames read | worst drift, any clip | group | limit |
|---|---|---|---|---|
| 19 of 19 pass | 287 | 0.00470 | hand.R | 0.00998 |

The eyes are judged here on their footprint on the face too: their worst posed drift in x and y is 0.00218.

## Not compared

- Shape keys are not rebuilt: one source clip plays a shape key that fattens an arm. With it playing, that clip's worst posed box drift is 0.03089. The comparisons above have the source's shape keys off, so they judge the bones and the weights only. This line is information, not a limit.
- The hand prop (`prop.R`) is left out of both sides.

## The eyes: a decal against a lens

The source's eyes are lenses: flat ovals 0.028 deep, standing up to about 8 mm off the head. The rebuild's are decals printed on the head (markings are decals, a design call of this kit). So the eyes are judged on their footprint on the face, the box in x and y, and their depth and surface gap are reported here, not judged. They measure the lens's depth, which that call removes.

| eyes number | value | limit it would meet |
|---|---|---|
| box drift in depth (z), at rest | 0.00577 | box 0.00998 |
| box drift in depth (z), worst posed | 0.00972 | box 0.00998 |
| surface gap p99, at rest | 0.01978 | gap p99 0.00499 |
| surface gap max, at rest | 0.02066 | gap max 0.00998 |

## Reproducing

The compare needs the earlier hand-built version, which this repository does not hold. What can be rerun here is the bar itself: build it as README.md describes, then run `uv run model-kit gate` on the scene folder.
