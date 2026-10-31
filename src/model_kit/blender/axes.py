"""The one axis change from Blender to glTF. numpy only, so it imports outside Blender too.

Blender is Z up with the front at -Y. glTF is Y up with the front at +Z. A Blender point
(x, y, z) lands at glTF (x, z, -y). Every array the exporters write is in glTF axes: points
are pre-multiplied by this matrix, and a rig's `world` matrix carries it.
"""

import numpy as np

BLENDER_TO_GLTF = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
BLENDER_TO_GLTF.flags.writeable = False
