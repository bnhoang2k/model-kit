# Bar 03: the chibi

A small rigged character, rebuilt from shape numbers. It is the bar for any piece that has a rig: it shows what a soft, outlined, posed figure looks like at true size, and what a rig must pass in the gate.

The rebuild was checked against an earlier hand-built version of the figure, which this repository does not hold. The numbers are in COMPARE.md.

## Reads as

A 1.0-unit-tall chibi (about 1.34 m at 0.747 units per metre). The head is the widest part and takes the top 0.58 of the height. The body is soft and rounded. The eyes are two ink ovals printed on the head. The limbs end in mittens and rounded feet. Eight closed parts overlap at the joints, so a runtime outline draws the creases; the eyes are open patches with no outline of their own. The design follows a reference drawing.

## Fits

A comparison bar for rigged pieces: a character, a creature, or any figure with a bone chain, a skin and a clip. Compare a new piece's sheet, rig and clips with this one.

## Wrong for

- Assemblies with no rig, such as furniture and props.
- Branching pieces, such as plants and trees.
- Realistic proportions. This figure is a chibi: a big head, short limbs.

## What is in the folder

- `chibi.pack.json`: the pack. One piece, `chibi`, of 10 parts: head, two eyes, torso, two arms, two thighs, and two shins (each shin and foot is one piece). A left/right pair is one shape and one `mirror_of` record.
- Soft shapes: every soft part's numbers live in its own `form` in `chibi.pack.json`. A `form` is a blob, ring, metaball or decal record, with a `why` for each choice. Nothing is tuned by eye in a script.
- The rig: 16 bones (`root`, `body`, `head`, and per side an arm of three bones, a leg of three bones, and a `prop.R` bone on the right hand). Each bone's `rest` holds its head, tail and roll. Four chains (both arms and both legs) are declared; the two arms bind to the skin as chains, each with a `falloff` (a smooth blend by height in bands), and the other parts bind rigid to one bone.
- `idle`: a keyed loop, frames 1 to 73 at 30 fps, no holds. Its breath lifts the body 6 mm while the feet stay planted. Its keys are in `idle.keys.json`; `clips.json` declares it.
- `build.py`: builds the piece in Blender from `chibi.pack.json` and writes `chibi.glb` and `chibi.blend`. The .blend is not committed.
- `export_scene.py`: exports `scene/` from that .blend.
- `scene/`: the scene folder the gate reads, with no Blender needed: the manifests, the piece, the rig, the samples of `idle` every half frame, the node table of `chibi.glb`, and `sources/build.py`.
- `sheet.png`: the chibi at true size, in every view.
- [COMPARE.md](COMPARE.md): the rebuild's numbers against the earlier hand-built version.

## Rebuild and check

Run from the repository root. To refresh the committed files, use `bars/03-chibi` as the out folder and `bars/03-chibi/scene` as the scene folder (the export script refuses a scene folder not named `scene`, because it deletes it). Blender does not read `PYTHONPATH`, so each Blender step puts the kit's `src` on its path first, as `model_kit.blender.runner.run_blender` does.

    blender --background --factory-startup --python-exit-code 1 \
        --python-expr "import sys; sys.path.insert(0, 'src')" \
        --python bars/03-chibi/build.py -- <out folder>
    blender --background --factory-startup --python-exit-code 1 <out folder>/chibi.blend \
        --python-expr "import sys; sys.path.insert(0, 'src')" \
        --python bars/03-chibi/export_scene.py -- <scene folder>
    uv run model-kit gate <scene folder>
    uv run model-kit sheet bars/03-chibi/chibi.pack.json --piece chibi \
        --style bars/03-chibi/style.json --bar none --out bars/03-chibi/sheet.png

The gate passes. The piece checks (crossings, take-apart, floaters, topology) and the loop seam of `idle` are measured. The clip checks that need holds, key targets, props or a placed piece say `skipped` or `not applicable` in their coverage lines: `idle` is keyed, holds nothing and carries no prop.

## Compare a new rigged piece against it

    uv run model-kit sheet <pack.pack.json> --piece <piece> --style <style.json> --bar auto

`--bar auto` picks this bar for a rigged piece once the bar is registered with a `bar.json`.
