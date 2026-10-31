# Bar 05: a desk lamp plugged into a power strip

The reference for hard-surface assemblies that join two pieces. A desk lamp's cable ends in a
2-blade plug, and the plug's blades stand in the bores of one outlet of a power strip. Both rest on
a plain desk top. It is the known-good model for the joint rules (a plug in a socket with real
bores), for pieces that rest on another, and for a printed label on a flat face.

Both products are real and their published sizes are used where they exist: the Anglepoise Type 75
Mini desk lamp and the Tripp Lite PS615 six-outlet power strip. Everything the sources do not
publish is a modelling choice, listed below.

## Reads as

A black desk lamp on a round cone base, 0.31 m tall in one reading pose: a lower arm leaning back,
an upper arm reaching forward, and a bell shade tilted down over the desk. A thin cable leaves the
back of the base, bends once across the desk, rises, and ends in a small plug whose two brass blades
stand in one outlet of a light gray six-outlet power strip lying flat on the desk (the second piece,
`lamp:Strip`, with a red switch and a printed logo oval on its outlet face).

## Fits

Hard-surface assemblies with a joint between two pieces: a cable and plug into a socket with real
bores, pieces that rest on another, a printed label on a flat face, and pieces seen from a desk's
distance.

## Wrong for

- Rigged characters, branching plants, and soft or organic forms.
- Anything that needs a bulb, a lamp switch, the strip's own cord or its switch housing: the bar has
  none. It is one pose of the lamp, not its range of motion. The desk is a plain slab, not a desk.

## What is in the folder

- `bar.json`: the bar's record, with the three source links and a pointer to the choices below.
- `lamp_build.py`: builds the three pieces in Blender (`lamp:Lamp`, `lamp:Strip` and `lamp:Desk`)
  and writes the layout `desk` into the scene.
- `build.py`: runs it and installs the result (`MODEL_KIT_BLENDER=/path/to/blender uv run python
  bars/05-lamp/build.py`, from the repository root).
- `style.json`: 1 unit per metre; the page camera is ortho at turn 14, tilt 52; crease angle 35.
  It also declares the reference camera for depth: a three.js perspective camera, near 0.1 m,
  far 2000 m, 24-bit depth (the 24 bits to confirm; `depth_why` has the sources).
- `lamp.pack.json` and `lamp.glb`: the pack and its packed file.
- `scene/`: the saved scene the gate reads (`uv run model-kit gate bars/05-lamp/scene`). It does
  not ship in the wheel. Its `layouts/desk.json` holds the placements: the desk, the lamp at the
  origin, and the strip lying flat (y 0 is the desk top), turned a quarter about X so its outlet
  face looks up. The lamp and the strip are `rests_on` the desk.
- `sheet.png`: `Lamp` alone at the page camera and four more views.

## The pieces

| Piece | Parts | Notes |
| --- | --- | --- |
| `Lamp` (the bar) | base, arm_lower, arm_upper, shade, cable, plug, blade_hot, blade_neutral | The connector `mains` runs from the mount `cord_exit` on the base to the mount `socket_3` of `Strip`. |
| `Strip` | body, switch, logo | The housing has six outlets cut in: 12 blade bores and 6 ground-pin bores. The `logo` is a printed decal. |
| `Desk` | top | A plain slab, 1.2 m by 0.8 m by 30 mm. |

What each part stands for in the rules:

- Resting: the lamp and the strip are `rests_on` the desk in the layout, so `clearance` checks that
  each touches it. Inside the pieces, the only `rests` row is `switch` on `body`, and it is inert: the
  strip is built with its outlet face toward +Z, so the switch's patch faces sideways and `bearing`
  prints not-applicable. Every other `on` row is `bonded` and is not judged by gravity: the arms are
  held by a spring-balanced pivot, the shade by the arm's joint, the cable and the blades by their
  moulding. So no part of this bar gives `bearing`'s rest rule a live case.
- A printed label: `logo`, shape and role `decal`, on the flat outlet face of `body`, in a row with
  `finish: print`. The strip is modelled with its outlet face toward +Z because a decal cannot be
  cast onto a face that looks straight up; the layout turns it flat.
- The joint: `cord_exit` and `socket_0` to `socket_5` are `power` mounts. The `mains` connector
  names the cable part and both ends. The strip's sockets name `body` as the part they are cut in
  (`Mount.part`) and the connector's layout end names `plug` as the part that seats (`End.plug`);
  the blades rest on it. `connectors` reads the seat from the geometry: the plug rests on `body`
  and each blade ends inside a bore of it: along the part of the blade that lies past the outlet
  face, the bore's wall rings it and it is in no material. No insertion depth is judged, but the
  contact length is the piece's seen floor (0.145 mm at 0.5 m): a plug pulled half way out lifts its
  body off the face and reads off its host. The cable's rule reads it by way of the plug: the cord
  enters the plug's side, so the cable reaches the plug and the plug reaches the socket's point.
  `clearance` does not judge the plug part and its blades against `body`; every other pair of the
  two pieces is judged.

## The modelling choices

Published sizes: shade 5.1 in across and 5.9 in tall, base 6.7 in across, cable 70.9 in, reach 21.6
in out and 28.4 in up (Anglepoise sheet); strip 10.8 x 1.85 x 1.23 in, six outlets 28.5 mm apart
(Tripp Lite datasheet); the NEMA blade and ground-pin sizes (Wikipedia, secondary). These are not
published, so each is a choice:

| Choice | Value | Reason |
| --- | --- | --- |
| Arm segments | 240 mm each | The sheet gives only the reach. Two equal arms plus the 65 mm shade radius stay under the 548.6 mm horizontal reach. |
| Pose | lower arm 20 degrees off vertical, upper arm 10 degrees up, shade 35 degrees down | One reading pose. The build stops if it leaves the 21.6 x 28.4 in reach. |
| Arm tube | 16 mm across | Not published. |
| Base | 30 mm tall, 6 mm straight rim, the cone meeting the flat top at a radius of 32 mm (about 24 degrees of slope), the top edge an arc of 40 mm radius | Not published. A cast iron base is a low weight, so it is a shallow cone; a casting's edges are rounded, so the cone meets the top in an arc. The angle and the arc radius are modelling choices for a shallow cast cone, not published values and not checked against photographs; `detail` judges the arc's facets. |
| Rim height | 6 mm | Taller than the 5 mm cable that leaves through it. |
| Shade | 1 mm wall, neck 14 mm across and 20 mm long, ceiling 35 mm from the arm end | Not published. The neck is narrower than the arm tube (16 mm) because `bearing`'s seat rule needs the shade's seat to lie wholly on the arm's end: the value is driven by that check, not by the object. |
| Cord exit | the back of the base (-z), at the foot of the rim | Not published. The sheet's drawing shows the cord on the right; the back keeps the cord clear of the shade. |
| Cable | 5 mm across; 0.59 m of the 1.80 m drawn | The plug stands 0.30 m right and 0.60 m behind the base so the sheet can show both ends. The rest of the cable is slack off the desk. The connector's cap is the published 70.9 in. |
| Cable entry into the plug | the side face toward the lamp, mid-height (a right-angle entry) | Not published. A low plug that the cord leaves sideways keeps the cable end near the socket. |
| Plug body | 38 x 22 x 19 mm | No source gives a moulded plug body. |
| Blade length | 17.1 mm | The middle of the published 15.9 to 18.3 mm. |
| Outlet bores | 0.12 mm wider than the blade or pin on each side, 20 mm deep | Receptacle slot sizes are not published in the sources read. The gap is driven by the `joints` check, not by the object: a real slot is looser. The check's ring reach is the peg radius plus the contact length (0.145 mm a viewer resolves at 0.5 m), and it reads any bore clearance over that length as "no bore round it". So the gap stays under 0.145 mm and the wall rings the blade all round. |
| Outlet places | first outlet 22 mm from the plain end, the rest 28.5 mm apart | Not published. |
| Switch | 22 x 12 x 4 mm, 30 mm from the cord end; the raised switch housing and the circuit-breaker icon are left out | Not published. The datasheet photograph shows a raised housing; only the rocker is modelled. |
| Logo | an oval 32 x 9 mm, 65 mm from the cord end | Not published. The oval stands for the wordmark; no lettering is drawn. |
| Desk | a slab 1.2 x 0.8 m, 30 mm thick, centred 0.1 m right and 0.25 m behind the lamp | No source. It only gives the two pieces a top to rest on; it is not a product. |
| `view.nearest_m` | 0.5 m for the three pieces | An estimate from use, not a measured study: a person at the desk reaches for the lamp's shade about an arm's length away, leans over the strip's outlets about as far to plug in, and has the desk top under their forearms. To be confirmed. |
| `view.farthest_m` | 1.3 m for the three pieces | An estimate from use, to be confirmed: the chair puts the eyes about 0.5 m from the desk's near edge and the desk is 0.8 m deep, so nothing on it is seen from farther than 1.3 m. |

The datasheet's "hwd" order cannot be right for a strip that lies on a desk, so the strip is read as
length x width x height. No source confirms the misprint; the shape settles it.

## Gate

`uv run model-kit gate bars/05-lamp/scene` reads `passed`. The `clearance` check does not judge a plug
seated in its socket against the part the socket names, and prints a coverage line that says so;
the plug against the strip's switch, or the lamp's base against the strip, would still fail.
