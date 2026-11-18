"""The unwrap's cost must follow the mesh's size, not its size times its number of regions.

A faceted mesh has one smooth region per face. The unwrap once re-read the whole mesh for each
region, so a 60,000-triangle statue never finished. This counts the whole-mesh steps on a sheet of
200 separate quads, so the test does not depend on how fast the machine is.
"""

import json
from pathlib import Path

import pytest

from model_kit import bake
from model_kit.blender import runner

pytestmark = pytest.mark.blender

QUADS = 200

SCRIPT = """
import json
from pathlib import Path

import bmesh
import bpy
import numpy as np

from model_kit.blender import runner

source, quads = runner.script_args()
code = Path(source).read_text().rstrip()
assert code.endswith("main()")
ns = {"__name__": "bake_render_under_test", "__file__": source}
exec(compile(code[: -len("main()")], source, "exec"), ns)

calls = {"next_loops": 0, "uv_area": 0}
for name in calls:
    def counted(*args, _name=name, _real=ns[name], **kw):
        calls[_name] += 1
        return _real(*args, **kw)
    ns[name] = counted

bpy.ops.wm.read_factory_settings(use_empty=True)
bm = bmesh.new()
for i in range(int(quads)):
    x, y = (i % 20) * 0.2, (i // 20) * 0.2
    vs = [bm.verts.new((x + a, y + b, 0.0)) for a, b in ((0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1))]
    bm.faces.new(vs)
me = bpy.data.meshes.new("sheet")
bm.to_mesh(me)
bm.free()
ob = bpy.data.objects.new("sheet", me)
bpy.context.scene.collection.objects.link(ob)
bpy.context.view_layer.update()
seen = np.ones(len(me.polygons), dtype=bool)
ns["unwrap"]({"placements": [{"id": "p", "px_per_unit": 64.0}]}, {"p": [ob]}, {ob.name: seen})
print("COST " + json.dumps(calls), flush=True)
runner.done()
"""


def test_unwrap_reads_each_mesh_a_fixed_number_of_times(tmp_path: Path):
    script = tmp_path / "count_unwrap.py"
    script.write_text(SCRIPT)
    result = runner.run_blender(script, [str(bake.SCRIPT), str(QUADS)], timeout=120)
    assert result.ok, result.log
    assert f"REGIONS sheet: disk {QUADS}" in result.log
    line = next(ln for ln in result.log.splitlines() if ln.startswith("COST "))
    calls = json.loads(line[len("COST "):])
    assert calls["next_loops"] <= 3, calls
    assert calls["uv_area"] <= 8, calls  # one for the total, one per packing attempt
