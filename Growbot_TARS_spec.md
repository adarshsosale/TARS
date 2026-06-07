# Growbot — TARS-inspired bipedal robot

A buildable, real-robot reinterpretation of Interstellar's TARS. Faithful to the
"big rectangle + two small rectangle legs" silhouette, but **without** the 4-panel
dismantle/transform mode. Three vertical slabs:

- **Central torso** — the big rectangle. Houses the battery + electronics, and
  doubles as a slightly-raised **rest leg** so the robot is a stable tripod when
  parked/charging.
- **Two outer legs** — the small rectangles. These are the actuated walkers
  (hip-pitch + ankle-pitch), each ending in a foot with a **TPU sole pad**.

## Files
| file | what it is |
|---|---|
| `Growbot_TARS.obj` | the model (mm, +Y up), parts split into named `o` groups |
| `Growbot_TARS.mtl` | materials (body / accent band / TPU / joint / panel) |
| `generate_growbot.py` | parametric generator — edit params at top, re-run |
| `preview_*.png` | quick render checks |

Regenerate after any tweak: `python3 generate_growbot.py`

## Coordinate system
- `+Y` = up, ground plane at `Y = 0`
- `+Z` = forward (walking direction) — feet are long in Z for fore/aft balance
- `±X` = left/right (the lateral axis the legs pivot about)

## Key dimensions (mm)
| | width X | depth Z | height Y | notes |
|---|---|---|---|---|
| Overall | 193 | 97 | **290 (≈11.4 in)** | shortened from 305 to hold mass when torso widened |
| Central torso | 82 | 58 | 270 (Y 20→290) | W 76→82 so the 2 hip servos go coaxial |
| Each leg slab | 44 | 44 | 238 (Y 52→290) | W widened 38→44 for the ankle servo |
| Ankle gap | — | — | Y 34→52 | hinge pin at Y≈43, axis X |
| Each foot | 55 | 95 | 28 (Y 6→34) | + 6 mm TPU pad below |
| TPU sole pad | 57 | 97 | 6 | 1 mm overhang lip |
| Stance | feet centres ±69 → outer base ±96.5 | — | — | base 193×97 mm |
| Hip pivot | — | — | Y=230 | axis = X (sagittal swing) |
| Bed split | — | — | Y=150 | bolted seam in the dark accent band |

Footprint-to-height ratio ≈ 0.6 (X) — fine statically; for **dynamic** walking
keep the CoM low (battery at the bottom of the torso) and the feet long in Z.

## Parts in the OBJ (each is its own `o` group; 25 total)
`central_body`, `nameplate`, `battery`, `controller_stack`, `power_buck`,
`hip_servo_L`/`R`, `leg_left`/`right`, `foot_left`/`right`, `foot_pad_left`/`right`,
`ankle_servo_L`/`R`, `ankle_pivot_L`/`R` (hinge pin), `ankle_crank_L`/`R`,
`ankle_pushrod_L`/`R`, `foot_rocker_L`/`R`, `central_rest_foot`, `central_rest_pad`.

Materials map straight to filament:
- `mat_body` / `mat_panel_line` / `mat_panel` → **ABS or PETG-CF / ABS-CF** (structure + panel grid)
- `mat_accent` / `mat_accent_rib` → the dark grip bands (paint, or a 2nd-color ABS)
- `mat_tpu` / `mat_tpu_tread` → **TPU** (sole pads + central rest pad) — print separately
- `mat_servo` / `mat_horn` / `mat_steel` → the servos, horns/crank/rocker, and the
  hinge pin + pushrod (bought parts: MG996R, metal horns, M3 rod/pins)

## Decorative detailing (faithful-to-TARS, all real geometry)
These are modeled as actual relief (≈0.7 mm proud panels / engraved lines), so they
show up in any viewer and slicer — no textures needed.
- **Panel-line pattern** on every slab face — the signature TARS look: **uniform
  vertical columns + rows of varying height** (rectangles taller than wide, with
  irregular horizontal seams), matching the reference photos. Column width
  (`PANEL_COL`), the row-height rhythm (`ROW_UNIT` × `ROW_PATTERN`), groove width
  and relief are all parameters. Horizontal seams wrap consistently around each
  segment; `phase` offsets the rhythm between stacked segments so they don't align.
- **Gridded end-caps** on top of all three monoliths (finer grid).
- **Ribbed grip bands** — the dark bands carry fine horizontal knurl ribs (the
  textured grip surface from the reference photos), not flat paint.
- **Nameplate placard + indicator-dot row** on the upper front torso (`nameplate`
  group) — the TARS marking area. Leave blank for a decal, or engrave text in CAD.
- **Treaded TPU soles** — cross-bars with grip grooves on every pad underside.

Note: the decals are modeled as overlapping closed solids (each sinks ~0.5 mm into
the core for a clean boolean union). Slicers union them automatically; if you need a
single watertight mesh, run a union/`Make Manifold` in your tool of choice.

## Internal layout & drivetrain (this is now modelled — see the section renders)
The chassis is built as real **3 mm shells** (hollow), with the drivetrain,
battery and boards inside as separate `o` groups. Render the cutaways with
`python3 render_sections.py` →
`section_front.png` (whole-robot frontal section), `section_hip.png`,
`section_ankle.png`, `cutaway_iso.png`, and the labelled explainers
`annotated_ankle.png` / `annotated_layout.png`.

**Drive architecture — direct-drive servos (the earlier worm design was dropped):**
> **Hip:** MG996R in the torso, output spline + horn straight to the leg (axis X, Y=230).
> **Ankle:** MG996R standing in the lower leg → crank → **pushrod** → foot rocker.

Why this scheme:
- Re-estimated light (~1.4 kg, low torque), so the worm's big reduction + self-locking
  weren't needed; an MG996R's internal gearbox is enough and far simpler to build.
- **Hip**: two MG996R in the torso, **coaxial & symmetric** — back-to-back on the same X
  axis at Z=0 (torso widened 76→82 to fit both). Symmetric load + motors = no fore/aft
  asymmetry for the controller to compensate. Spline drives each leg directly at Y=230.
- **Ankle**: the thin foot can't hold a motor coaxial with the ankle axis, *and* a 38 mm
  leg is too narrow for a servo on that axis (its body is 37 mm long) — so the leg is
  **widened to 44 mm**, the servo stands inside it, and a **4-bar pushrod** reaches down
  to a rocker on the foot. The foot pitches about a real **hinge pin** (X, Y≈43); Y 34→52
  is the open ankle gap. Rocker:crank ≈ 1.23 → ~1.2× servo torque with ±~49° travel
  (tune via the crank/rocker radii). The pushrod runs outboard (exposed/serviceable).

Trade-off: direct servos are **not self-locking** (they hold under power, unlike the
worm) and are roughly **back-drivable** through the linkage — the **TPU sole** still
provides shock compliance. For real lateral balance, add **ankle roll** (a 2nd DOF).
See `annotated_ankle.png` (oblique cutaway) for the linkage, and `detail_ankle.png` /
`section_ankle.png` for the servo-in-leg section.

### Battery / electronics (central torso)
Modelled inside the torso shell (`battery`, `pcb_main`, `pcb_driver`):
- 3S/4S LiPo placed **low** (Y≈26–120) = ballast for a low CoM.
- main controller (ESP32 / Teensy / RPi-class) + IMU + 2× motor drivers on the
  walls; charge port / switch on the back **access panel** (`central_access_panel`).
- Usable cavity with 3 mm walls ≈ **70 × 52 mm** cross-section, most of the height.

### DOF summary
4 powered DOF (2 hip-pitch + 2 ankle-pitch): hips direct-servo, ankles servo+pushrod.
**Two hip servos — one per leg** (both legs driven), now **coaxial & symmetric** (back-to-
back on the X axis at Z=0; torso widened to 82) — see `previews/section_hip_top.png` /
`previews/hip_3d.png`. Lateral balance relies on
the wide/long feet + small torso sway; the central rest foot (`central_rest_foot`, 8 mm
clearance, `CF_CLEAR`) is the anti-tip/parking support.

### Drive ratios, wiring, and printed parts (practical / print-ready)
- **Drive ratios.** Hip (main) = **direct 1:1** (the MG996R's ~1:270 internal gearbox is
  the reduction); ~11 kgf·cm vs ≈4 kgf·cm worst-case stance lean → ~2× margin (add a 2:1
  printed stage or a DS3225 for dynamic walking). Ankle = pushrod **≈1.23:1** (torque
  advantage), ~11–12 kgf·cm vs ~6.7 needed → ~1.8× margin, foot sweep ≈ ±49°.
- **All ankle drive parts are 3D-printed** (CF-ABS): crank, pushrod, foot rocker, and the
  Ø8 hinge pin (drop in an M3 bolt for a long-life build). Print the hinge with 0.3–0.4 mm
  clearance; orient the pushrod/rocker so layer lines run along their length (tension path).
  See `annotated_ankle.png` (assembled) and `ankle_exploded.png` (parts).
- **Wiring** (`wiring.png`). Ankle-servo 3-wire leads run up the leg cavity (Z≈15, behind
  the servo), **cross ABOVE the hip servos with a service loop**, down the torso side wall,
  in to the PCA9685 — a path that touches no solid part. Drill ~6 mm grommet holes at the
  leg↔torso crossing; strain-relief both sides.
- **Bed splits + screw joints.** Slabs split at `SPLIT_Y=150` (in the dark band). Printed
  bosses straddle the seam + M3 screws (torso 4, each leg 4); lower half gets a heat-set
  insert, upper half a clearance hole. Gusset bosses to the wall corners in CAD.
- **True-3D renders** live in `previews/` (run `previews/render3d.py`): `iso_cutaway_3d`,
  `hip_3d`, `ankle_3d`, `ankle_exploded_3d`, and `print_parts_3d` (bed-split halves lifted
  apart + a legend of exactly what prints as one piece vs separately).
- **Mass** auto-computed by the generator: **≈1.40 kg** @3 mm walls (≈1.15 @2 mm).

## Filament / structural notes (ABS vs CF)
- Tall thin slabs are the weak axis. **Print legs/torso upright** (long axis = Z
  of the bed isn't possible at 277 mm on most beds — see below) so layer lines run
  along the load path, then the bending load is across-grain → use **CF-filled**
  (PETG-CF / PA-CF / ABS-CF) for the legs and the hip area; plain ABS is OK for the
  torso skin.
- Walls ≥ 3 mm (≥ 5 perimeters) on the legs; gusset the hip/ankle bosses.
- Heat-set brass inserts at every joint and panel screw — don't thread into ABS.

## Printing & assembly
- **Bed fit:** the 277–285 mm slabs exceed a 256 mm bed. Either print on a 300+ mm
  printer **or** split each slab with a dovetail/bolt joint at ~Y=150 (the dark
  accent band is a natural seam — hides the split). The generator's segment
  boundaries are good cut planes.
- Print orientation: slabs vertical (long axis up) for strength; feet flat;
  TPU pads flat in TPU.
- Hardware: 2× hip actuators, 2× ankle actuators (MG996R-class or small BLDC +
  gearbox / cycloidal for the hips), bearings or servo horns at each hub, M3
  bolts + heat-set inserts, foam/lead ballast low in the torso if CoM is high.

## Faithful-to-TARS choices vs. real-robot optimizations
- Kept: three monolith slabs, the wrap-around dark grip bands, high pivot, blocky
  rectilinear look.
- Changed for realism: dropped the 4-panel dismantle; central slab made distinct
  (bigger) as the e-bay + rest leg; feet lengthened in Z for fore/aft stability;
  added explicit hip/ankle joints and TPU soles; torso hollowed for a battery bay.

## Tuning
All geometry is parametric — edit the block at the top of `generate_growbot.py`
(`STAND_H`, slab W/D, `HIP_Y`, foot size, `CF_CLEAR`, joint radii) and re-run.
The script prints the resulting bounding box and stance width.
