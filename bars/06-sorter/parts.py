"""The toy letter sorter (bar 06): its parts list, as data. No Blender, no drawing code.

One list builds both versions. `real` builds it as written (real-product detail, the desk
pack's level). `toy` derives from the same list in resolve.py: it drops the parts marked
`tier: "real"`, squashes the proportions by TOY_SCALE, rounds every part hard, oversizes the
parts marked `ctl` (controls), thickens thin parts, and applies the few per-part `toy` overrides
written next to the part they change.

Units are metres. Axes are glTF: x runs along the machine (the feeder end is +x), y is up, z is
the front (the pockets and the operator panel face +z). The machine's origin is the centre of its
footprint on the floor.

How a part is written:
    kind   box | round | tray | lathe | tube
    host   the part it stands on (None for the one root, the plinth)
    face   the host's face it stands on, by world direction: +x -x +y -y +z -z, or `floor`
           (a tray's inner floor)
    at     a world point; its two components in the face's plane place the centre of the
           part's footprint. The component along the face normal is ignored (None).
    size   world extents (sx, sy, sz) for box and tray; the extent along the face normal is the
           part's height out of the face, or "reach:<part>" to run down (or out) until it meets
           that part's facing surface
    R, h   radius and height out of the face, for round and lathe
    r, e   footprint corner radius and top rim radius (a soft Hairline solid, see machine_build.py)
    eb     bottom rim radius (a solid rounded at both ends, so it can stack on its twin)
    t      wall thickness, for a tray
    tier   "real" parts are left out of the toy
    ctl    a control (button, light): the toy oversizes it instead of squashing it
    keep   when the toy drops this part's host, the part moves onto the host's own host
    held   "bonded" (bolted, welded, moulded: the default) or "rests" (a parcel)
"""

ROLES = {  # linear grays; the renderer draws every machine part near-white and lets the lines carry it
    "Steel": (0.62, 0.63, 0.65),
    "Paint": (0.80, 0.80, 0.80),
    "Belt": (0.30, 0.30, 0.30),
    "Rubber": (0.20, 0.20, 0.20),
    "Screen": (0.18, 0.20, 0.22),
    "Control": (0.86, 0.86, 0.86),
    "Lens": (0.90, 0.90, 0.90),
    "Plate": (0.92, 0.92, 0.92),
    "Parcel": (0.78, 0.66, 0.50),
    "Cord": (0.25, 0.25, 0.25),
}

# Toy derivation: the kit's numbers (toy.py in this folder) are this machine's; set one here to override.

# What machine_build.py registers. `done` says, per style, what the piece is when it is finished.
META = {
    "name": "sorter",  # pack sorter_<style>, file sorter-<style>.glb
    "piece": "Sorter",
    "kind": "letter sorter",
    "size_class": "hero",
    "root": "base",
    "done": {
        "real": "A multi-pocket parcel sorter, real-product detail",
        "toy": "A multi-pocket parcel sorter, a toy of the same machine",
        "common": (": a plinth on levelling feet, a spine cabinet carrying a belt between guide rails, a "
                   "feeder magazine with a stack of parcels, a C-frame scanner with a beacon, six pushers, "
                   "six labelled pockets (four holding a parcel), a takeaway belt under the pockets, and an "
                   "infeed cabinet with a door, an operator panel (screen, three buttons, an emergency stop) "
                   "and a stack light."),
    },
}

# ---------------------------------------------------------------------------------------------
# Rows: one per assembly group. (text, source, held). The build fills min_thickness from the
# parts it built. Build order is row order; model-kit's take-apart removes the last built first.
SRC = "No source: a modelling choice for a parcel sorter of about this size (README.md)."
ROWS = {
    "b01": ("Plinth, the welded base frame every module stands on", SRC, "bonded"),
    "b02": ("Eight levelling feet under the plinth: threaded stem, lock nut and pad", SRC, "bonded"),
    "c01": ("Spine cabinet along the back, carrying the transport on top", SRC, "bonded"),
    "c02": ("Infeed cabinet at the feeder end, holding the electrics", SRC, "bonded"),
    "c03": ("Service door on the infeed cabinet, with hinges and a quarter-turn latch", SRC, "bonded"),
    "c04": ("Vent grille and its louvres on the service door", SRC, "bonded"),
    "c05": ("Nameplate and rivets on the infeed cabinet's end", SRC, "bonded"),
    "c06": ("Cable gland and conduit between the infeed cabinet and the spine", SRC, "bonded"),
    "c07": ("Access covers and screws on the spine's front", SRC, "bonded"),
    "t01": ("Transport bed along the spine's top", SRC, "bonded"),
    "t02": ("Transport belt on the bed", SRC, "bonded"),
    "t03": ("Guide rails each side of the belt, open at every pocket", SRC, "bonded"),
    "t04": ("Rail screws on the front guide rail", SRC, "bonded"),
    "t05": ("Belt drive at the tail end: gearbox and motor", SRC, "bonded"),
    "d01": ("Six pushers behind the belt, one per pocket: bracket, air cylinder and pad", SRC, "bonded"),
    "h01": ("Scanner: a C-frame post and hood over the belt, its window, lamps and status light", SRC, "bonded"),
    "h02": ("Beacon on the scanner hood", SRC, "bonded"),
    "m01": ("Feeder magazine at the head of the belt: back plate and two side guides", SRC, "bonded"),
    "p01": ("Operator panel housing on the infeed cabinet", SRC, "bonded"),
    "p02": ("Touch screen and its bezel", SRC, "bonded"),
    "p03": ("Three push buttons in collars, with legend plates", SRC, "bonded"),
    "p04": ("Emergency stop: collar and mushroom head", SRC, "bonded"),
    "p05": ("Key switch and key", SRC, "bonded"),
    "p06": ("Panel screws and the panel's conduit into the cabinet", SRC, "bonded"),
    "s01": ("Stack light on the infeed cabinet: base, three lamp segments and cap", SRC, "bonded"),
    "k01": ("Pocket shelf on the spine's front, on three legs", SRC, "bonded"),
    "k02": ("Six pocket trays, one per labelled tray", SRC, "bonded"),
    "k03": ("Seven pocket dividers", SRC, "bonded"),
    "k04": ("Label plates on the pocket fronts (blank; a label is added per tray)", SRC, "bonded"),
    "k05": ("Label screws and pocket-full lights", SRC, "bonded"),
    "w01": ("Takeaway conveyor under the pockets: bed, belt and legs; it feeds the main belt", SRC, "bonded"),
    "x01": ("Parcels stacked in the feeder magazine", "No source: a parcel size chosen for the scene (README.md).", "rests"),
    "x02": ("Parcels sorted into pockets", "No source: a parcel size chosen for the scene (README.md).", "rests"),
    "x03": ("A sorted parcel on the takeaway belt", "No source: a parcel size chosen for the scene (README.md).", "rests"),
}

PARTS = []


def add(kind, id, row, role, host, face, **kw):
    PARTS.append({"kind": kind, "id": id, "row": row, "role": role, "host": host, "face": face, **kw})


# ---------------------------------------------------------------------------------------------
# Shared numbers
PARCEL = (0.30, 0.12, 0.22)  # x, y, z: one parcel, a small shipping box
POCKET_X = [-1.67, -1.23, -0.79, -0.35, 0.09, 0.53]  # six pockets, 0.44 m apart
OPEN = 0.15  # half the rail opening at each pocket
DIVIDER_X = [-1.89, -1.45, -1.01, -0.57, -0.13, 0.31, 0.75]
RAIL_SEGMENTS = [(-1.93, -1.82), (-1.52, -1.38), (-1.08, -0.94), (-0.64, -0.50),
                 (-0.20, -0.06), (0.24, 0.38), (0.68, 1.93)]
SHELF_Y = (0.64, 0.68)
TRAY_Z = (-0.08, 0.42)

# ---------------------------------------------------------------------------------------------
# Base
add("box", "base", "b01", "Steel", None, None, origin=(0.0, 0.06, 0.0),
    size=(4.00, 0.08, 1.16), r=0.03, e=0.008,
    toy={"size": (3.52, 0.09, 1.16)})
for i, x in enumerate((-1.85, -0.62, 0.62, 1.85)):
    for j, z in enumerate((-0.48, 0.48)):
        # profile from the plinth's underside down: stem, lock nut, stem, pad
        add("lathe", f"foot_{i}{j}", "b02", "Rubber", "base", "-y", at=(x, None, z),
            profile=[(0.0, 0.0), (0.011, 0.0), (0.011, 0.012), (0.019, 0.012), (0.019, 0.022),
                     (0.011, 0.022), (0.011, 0.038), (0.040, 0.044), (0.040, 0.060), (0.0, 0.060)],
            toy={"kind": "round", "R": 0.06, "h": 0.0456, "e": 0.015})  # h: the toy plinth sits at 0.06 x 0.76

# ---------------------------------------------------------------------------------------------
# Cabinets
add("box", "spine", "c01", "Paint", "base", "+y", at=(0.0, None, -0.34),
    size=(3.94, 1.01, 0.45), r=0.02, e=0.012)
add("box", "infeed", "c02", "Paint", "base", "+y", at=(1.425, None, 0.235),
    size=(1.09, 1.01, 0.66), r=0.02, e=0.012)

# service door on the infeed front (+z face at z 0.565)
add("box", "door", "c03", "Paint", "infeed", "+z", at=(1.66, 0.62, None),
    size=(0.50, 0.86, 0.02), r=0.01, e=0.004, toy={"size": (0.34, 0.60, 0.02), "nudge_y": -0.03})
for k, y in enumerate((0.30, 0.62, 0.94)):
    add("box", f"hinge_{k}", "c03", "Steel", "door", "+z", at=(1.876, y, None),
        size=(0.022, 0.07, 0.012), r=0.004, e=0.003, tier="real")
add("round", "latch", "c03", "Steel", "door", "+z", at=(1.45, 0.66, None), R=0.022, h=0.012,
    e=0.004, ctl=True, toy={"nudge_x": 0.10})
add("box", "latch_grip", "c03", "Steel", "latch", "+z", at=(1.45, 0.66, None),
    size=(0.010, 0.032, 0.018), r=0.004, e=0.003, ctl=True)
add("box", "grille", "c04", "Steel", "door", "+z", at=(1.66, 0.34, None),
    size=(0.30, 0.16, 0.004), r=0.008, e=0.002, tier="real")
for k, y in enumerate((0.28, 0.31, 0.34, 0.37, 0.40)):
    add("box", f"louvre_{k}", "c04", "Steel", "grille", "+z", at=(1.66, y, None),
        size=(0.26, 0.012, 0.004), r=0.004, e=0.0015, tier="real")

# nameplate on the infeed end (+x face at x 1.97)
add("box", "nameplate", "c05", "Plate", "infeed", "+x", at=(None, 0.95, 0.30),
    size=(0.003, 0.09, 0.20), r=0.006, e=0.0015, tier="real")
for k, (y, z) in enumerate(((0.917, 0.212), (0.917, 0.388), (0.983, 0.212), (0.983, 0.388))):
    add("round", f"rivet_{k}", "c05", "Steel", "nameplate", "+x", at=(None, y, z), R=0.004,
        h=0.002, e=0.001, tier="real")

# a cable gland on the infeed end and a conduit back into the spine's end
add("round", "gland", "c06", "Steel", "infeed", "+x", at=(None, 0.40, -0.04), R=0.02, h=0.025,
    e=0.004, tier="real")
add("tube", "conduit", "c06", "Cord", "gland", "+x", R=0.010, bend=0.04, tier="real",
    path=[(None, 0.40, -0.04), (2.07, 0.40, -0.04), (2.07, 0.40, -0.25), (1.97, 0.40, -0.25)],
    end="spine", end_face="+x")

# access covers on the spine front (+z face at z -0.115), between the takeaway and the shelf
for k, x in enumerate((-1.30, 0.10)):
    add("box", f"cover_{k}", "c07", "Paint", "spine", "+z", at=(x, 0.51, None),
        size=(0.60, 0.18, 0.004), r=0.012, e=0.002, tier="real")
    for m, (dx, dy) in enumerate(((-0.28, -0.07), (0.28, -0.07), (-0.28, 0.07), (0.28, 0.07))):
        add("round", f"cover_{k}_screw_{m}", "c07", "Steel", f"cover_{k}", "+z",
            at=(x + dx, 0.51 + dy, None), R=0.005, h=0.003, e=0.0012, tier="real")

# ---------------------------------------------------------------------------------------------
# Transport on the spine's top (y 1.15)
add("box", "bed", "t01", "Steel", "spine", "+y", at=(0.0, None, -0.30),
    size=(3.92, 0.05, 0.34), r=0.01, e=0.004, toy={"size": (3.42, 0.04, 0.30), "nudge_z": -0.01})
add("box", "belt", "t02", "Belt", "bed", "+y", at=(0.0, None, -0.30),
    size=(3.90, 0.006, 0.29), r=0.005, e=0.002,
    toy={"size": (3.38, 0.012, 0.21)})
for k, (x0, x1) in enumerate(RAIL_SEGMENTS):
    xc, w = (x0 + x1) / 2, x1 - x0
    add("box", f"rail_back_{k}", "t03", "Steel", "bed", "+y", at=(xc, None, -0.461),
        size=(w, 0.08, 0.014), r=0.004, e=0.003, toy={"nudge_z": 0.031})
    add("box", f"rail_front_{k}", "t03", "Steel", "bed", "+y", at=(xc, None, -0.139),
        size=(w, 0.08, 0.014), r=0.004, e=0.003, toy={"nudge_z": -0.031})
    xs = [xc - w / 2 + 0.04, xc + w / 2 - 0.04] if w < 0.5 else [x0 + 0.06 + 0.29 * i for i in range(5)]
    for m, x in enumerate(xs):
        add("round", f"rail_screw_{k}_{m}", "t04", "Steel", f"rail_front_{k}", "+z",
            at=(x, 1.24, None), R=0.005, h=0.003, e=0.0012, tier="real")
add("box", "gearbox", "t05", "Steel", "spine", "+y", at=(-1.86, None, -0.5175),
    size=(0.12, 0.10, 0.075), r=0.008, e=0.006, toy={"size": (0.106, 0.076, 0.06), "nudge_z": 0.015})
add("round", "motor", "t05", "Paint", "gearbox", "-x", at=(None, 1.20, -0.5175), R=0.026, h=0.14,
    e=0.006)

# pushers behind the belt, one per pocket
for i, x in enumerate(POCKET_X):
    add("box", f"push_bracket_{i}", "d01", "Steel", "spine", "+y", at=(x, None, -0.515),
        size=(0.08, 0.14, 0.07), r=0.008, e=0.005, toy={"size": (0.07, 0.106, 0.05), "nudge_z": 0.01})
    add("round", f"push_cylinder_{i}", "d01", "Steel", f"push_bracket_{i}", "+z",
        at=(x, 1.245, None), R=0.03, h=0.03, e=0.004)
    add("round", f"push_pad_{i}", "d01", "Rubber", f"push_cylinder_{i}", "+z",
        at=(x, 1.245, None), R=0.024, h=0.008, e=0.002)

# scanner: a C-frame post at the back, its hood reaching over the belt
add("box", "scan_post", "h01", "Paint", "spine", "+y", at=(1.06, None, -0.535),
    size=(0.48, 0.50, 0.04), r=0.01, e=0.006, toy={"nudge_z": 0.025})
add("box", "scan_hood", "h01", "Paint", "scan_post", "+z", at=(1.06, 1.51, None),
    size=(0.44, 0.22, 0.40), r=0.04, e=0.012)
add("box", "scan_window", "h01", "Screen", "scan_hood", "+z", at=(1.03, 1.51, None),
    size=(0.30, 0.10, 0.004), r=0.01, e=0.0015)
add("round", "scan_led", "h01", "Lens", "scan_hood", "+z", at=(1.23, 1.51, None), R=0.008,
    h=0.004, e=0.0015, tier="real")
for k, z in enumerate((-0.40, -0.23)):
    add("box", f"scan_lamp_{k}", "h01", "Lens", "scan_hood", "-y", at=(1.06, None, z),
        size=(0.34, 0.02, 0.04), r=0.008, e=0.004, tier="real")
add("round", "beacon_base", "h02", "Steel", "scan_hood", "+y", at=(1.20, None, -0.22), R=0.03,
    h=0.025, e=0.004, ctl=True, toy={"nudge_x": -0.03})
add("round", "beacon_lens", "h02", "Lens", "beacon_base", "+y", at=(1.20, None, -0.22), R=0.026,
    h=0.05, e=0.006, eb=0.006, ctl=True)
add("round", "beacon_cap", "h02", "Steel", "beacon_lens", "+y", at=(1.20, None, -0.22), R=0.02,
    h=0.012, e=0.004, ctl=True)

# feeder magazine at the head of the belt
add("box", "mag_back", "m01", "Steel", "spine", "+y", at=(1.69, None, -0.535),
    size=(0.46, 0.55, 0.03), r=0.006, e=0.004, toy={"nudge_z": 0.025})
for k, x in enumerate((1.48, 1.90)):
    add("box", f"mag_guide_{k}", "m01", "Steel", "mag_back", "+z", at=(x, 1.52, None),
        size=(0.016, 0.32, 0.40), r=0.006, e=0.004, toy={"nudge_x": (0.006, -0.006)[k]})

# ---------------------------------------------------------------------------------------------
# Operator panel on the infeed front (z 0.565)
add("box", "panel", "p01", "Paint", "infeed", "+z", at=(1.13, 0.98, None),
    size=(0.40, 0.30, 0.10), r=0.025, e=0.008, toy={"size": (0.40, 0.26, 0.10), "nudge_x": 0.08, "nudge_y": -0.07})
add("box", "bezel", "p02", "Control", "panel", "+z", at=(1.10, 1.045, None),
    size=(0.22, 0.15, 0.012), r=0.012, e=0.004)
add("box", "screen", "p02", "Screen", "bezel", "+z", at=(1.10, 1.045, None),
    size=(0.19, 0.12, 0.003), r=0.006, e=0.001)
for k, x in enumerate((1.00, 1.065, 1.13)):
    add("round", f"button_collar_{k}", "p03", "Steel", "panel", "+z", at=(x, 0.89, None), R=0.02,
        h=0.006, e=0.002, tier="real")
    add("round", f"button_{k}", "p03", "Control", f"button_collar_{k}", "+z", at=(x, 0.89, None),
        R=0.0145, h=0.014, e=0.004, ctl=True, keep=True,
        toy={"nudge_x": (-0.03, -0.015, 0.0)[k] * 1.0})
    add("box", f"legend_{k}", "p03", "Plate", "panel", "+z", at=(x, 0.855, None),
        size=(0.04, 0.012, 0.002), r=0.002, e=0.0008, tier="real")
add("round", "estop_collar", "p04", "Control", "panel", "+z", at=(1.245, 0.89, None), R=0.036,
    h=0.008, e=0.003, ctl=True, toy={"R": 0.05, "h": 0.012, "nudge_x": 0.035})
add("round", "estop", "p04", "Control", "estop_collar", "+z", at=(1.245, 0.89, None), R=0.028,
    h=0.024, e=0.010, ctl=True, toy={"R": 0.042, "h": 0.036})
add("round", "key_body", "p05", "Steel", "panel", "+z", at=(1.27, 1.06, None), R=0.014, h=0.01,
    e=0.003, tier="real")
add("box", "key", "p05", "Steel", "key_body", "+z", at=(1.27, 1.06, None),
    size=(0.004, 0.018, 0.014), r=0.0015, e=0.001, tier="real")
for k, (x, y) in enumerate(((0.945, 0.845), (1.315, 0.845), (0.945, 1.115), (1.315, 1.115))):
    add("round", f"panel_screw_{k}", "p06", "Steel", "panel", "+z", at=(x, y, None), R=0.005,
        h=0.003, e=0.0012, tier="real")
add("tube", "panel_conduit", "p06", "Cord", "panel", "-y", R=0.012, bend=0.045, tier="real",
    path=[(1.13, None, 0.615), (1.13, 0.70, 0.615), (1.13, 0.70, 0.565)],
    end="infeed", end_face="+z")

# stack light on the infeed's top (y 1.15)
add("round", "stack_base", "s01", "Steel", "infeed", "+y", at=(1.90, None, 0.49), R=0.035, h=0.04,
    e=0.004, ctl=True, toy={"nudge_x": -0.09, "nudge_z": -0.10})
for k in range(3):
    add("round", f"stack_lamp_{k}", "s01", "Lens", "stack_base" if k == 0 else f"stack_lamp_{k - 1}",
        "+y", at=(1.90, None, 0.49), R=0.032, h=0.055, e=0.003, eb=0.006, ctl=True)
add("round", "stack_cap", "s01", "Steel", "stack_lamp_2", "+y", at=(1.90, None, 0.49), R=0.03,
    h=0.012, e=0.004, ctl=True)

# ---------------------------------------------------------------------------------------------
# Pockets on a shelf off the spine's front
add("box", "shelf", "k01", "Steel", "spine", "+z", at=(-0.57, 0.66, None),
    size=(2.74, 0.04, 0.56), r=0.008, e=0.005)
for k, x in enumerate((-1.90, -0.57, 0.76)):
    add("box", f"shelf_leg_{k}", "k01", "Steel", "shelf", "-y", at=(x, None, 0.41),
        size=(0.04, "reach:base", 0.04), r=0.008, e=0.004)
for i, x in enumerate(POCKET_X):
    add("tray", f"pocket_{i}", "k02", "Paint", "shelf", "+y", at=(x, None, 0.17),
        size=(0.42, 0.10, 0.50), r=0.03, e=0.006, t=0.012, toy={"t": 0.025})
    add("box", f"label_{i}", "k04", "Plate", f"pocket_{i}", "+z", at=(x, 0.73, None),
        size=(0.30, 0.05, 0.003), r=0.006, e=0.0015, toy={"size": (0.20, 0.05, 0.012)})
    for m, dx in enumerate((-0.135, 0.135)):
        add("round", f"label_screw_{i}_{m}", "k05", "Steel", f"label_{i}", "+z",
            at=(x + dx, 0.73, None), R=0.004, h=0.002, e=0.001, tier="real")
    add("round", f"pocket_light_{i}", "k05", "Lens", "spine", "+z", at=(x, 1.05, None), R=0.012,
        h=0.008, e=0.003, ctl=True)
for k, x in enumerate(DIVIDER_X):
    add("box", f"divider_{k}", "k03", "Steel", "shelf", "+y", at=(x, None, 0.07),
        size=(0.012, 0.30, 0.30), r=0.004, e=0.004, tier="real")

# ---------------------------------------------------------------------------------------------
# Takeaway conveyor under the pockets: a stadium-ended bed off the spine's front
add("box", "take_bed", "w01", "Steel", "spine", "+z", at=(-0.57, 0.33, None),
    size=(2.74, 0.08, 0.46), r=0.04, e=0.006)
add("box", "take_belt", "w01", "Belt", "take_bed", "+y", at=(-0.57, None, 0.115),
    size=(2.62, 0.006, 0.42), r=0.006, e=0.002, toy={"size": (2.30, 0.012, 0.40)})
for k, x in enumerate((-1.80, -0.57, 0.66)):
    add("box", f"take_leg_{k}", "w01", "Steel", "take_bed", "-y", at=(x, None, 0.30),
        size=(0.04, "reach:base", 0.04), r=0.008, e=0.004)

# ---------------------------------------------------------------------------------------------
# Parcels: a stack of three in the magazine, four sorted into pockets, one on the takeaway
MAG_STACK = [(PARCEL, 0.0, 0.0), ((0.27, 0.12, 0.19), 0.012, -0.010), ((0.24, 0.11, 0.17), 0.004, -0.004)]
for k, (size, dx, dz) in enumerate(MAG_STACK):  # three parcel sizes, each a little smaller
    add("box", f"parcel_mag_{k}", "x01", "Parcel", "belt" if k == 0 else f"parcel_mag_{k - 1}",
        "+y", at=(1.69 + dx, None, -0.30 + dz), size=size, r=0.012, e=0.006, eb=0.010)
for k, (i, dx, dz) in enumerate(((0, -0.02, 0.02), (1, 0.03, 0.0), (3, 0.0, 0.03), (4, -0.03, 0.01))):
    add("box", f"parcel_pocket_{k}", "x02", "Parcel", f"pocket_{i}", "floor",
        at=(POCKET_X[i] + dx, None, 0.17 + dz), size=PARCEL, r=0.012, e=0.006, eb=0.010)
add("box", "parcel_take", "x03", "Parcel", "take_belt", "+y", at=(-1.25, None, 0.115),
    size=PARCEL, r=0.012, e=0.006, eb=0.010)
