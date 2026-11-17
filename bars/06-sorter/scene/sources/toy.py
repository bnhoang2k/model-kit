"""The toy style's numbers, shared by every machine of the toy style.

A machine's parts.py may override any of these by defining the same name; the defaults are the
letter sorter's, the first machine reference.
"""

TOY_SCALE = (0.88, 0.76, 1.0)  # x, y, z: shorter and squatter, so a panel sits at a chibi's chest
TOY_CTL = 1.8  # controls (buttons, e-stop, lights) grow by this, the chibi bar's oversized look
TOY_ROUND = 0.30  # corner radius as a share of the smaller footprint side
TOY_RIM = 0.30  # rim radius as a share of the smallest side; the rim rounds a full 90 degrees
TOY_MIN_SIDE = 0.03  # no footprint side thinner than this (a rail, a guide)
TOY_MIN_H = 0.012  # no plate thinner than this out of its face (a label)
NEAREST_M = 4.0  # view.nearest_m: the nearest a scene camera gets to a machine (to be measured)
