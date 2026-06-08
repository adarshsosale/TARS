# Hexabot → Isaac Lab — Setup, Gait Demo & TARS Comparison

An 18-DOF hexapod, built **parametrically from scratch** (same philosophy as the
TARS/Growbot pipeline) and prepped for **NVIDIA Isaac Lab**. The premise: TARS
walked badly because it had only **4 DOF** and was an underpowered inverted
pendulum. A hexapod fixes that at the architecture level — **6 legs × 3 DOF**,
low and wide, with **at least one tripod of 3 feet always planted** (statically
stable, no balancing act).

> Everything in this folder was produced on a Mac. **Running Isaac needs an
> NVIDIA RTX GPU on Linux/Windows** — there is no macOS build. The model prep is
> done; the GPU steps are in [Phase 8](#phase-8--run-it-on-the-gpu-box).

---

## TL;DR

- **Done (this Mac):** a parametric generator, a validated **URDF** (19 links, 18
  revolute joints), per-link **meshes** (metres, Z-up), **analytic inertia**
  (PhysX-valid), an Isaac Lab **ArticulationCfg**, an offline **validator** (all
  checks pass), an open-loop **tripod-gait** script, and local **renders + a gait
  GIF** proving it stands and walks +X *before* any GPU.
- **You do (GPU box):** convert URDF→USD → check the **ArticulationRoot** →
  `tripod_gait.py` to watch it walk → (optional) wrap as an RL task.
- **Start here:** copy the `isaac/` folder to your GPU machine, then
  [Phase 8](#phase-8--run-it-on-the-gpu-box).

---

## What it is (from the reference model)

A hexagonal-body hexapod with claw feet (the print-ready reference is in
`reference model/`, photos in `reference images/`). Each of the 6 legs is a clean
3-DOF chain, the canonical hexapod:

```
base ──coxa (yaw, +Z)──► coxa link ──femur (pitch)──► femur link ──tibia (pitch)──► tibia link (claw)
```

- **coxa** swings the whole leg fore/aft (about vertical),
- **femur** lifts the leg,
- **tibia** flexes the knee / plants the claw.

6 legs at azimuths **±30° / ±90° / ±150°** (60° apart; a *gap* — not a leg —
points dead-forward, matching the antenna gap in the photos). Names: `lf, lm, lr`
(left front/mid/rear) and `rf, rm, rr`. **Tripod groups:** A = `{lf, rm, lr}`,
B = `{rf, lm, rr}`.

Built parametrically (not by rigging the loose STLs) because a hexapod is maximally
regular — looping over 6 identical legs gives exact, symmetric joint axes and
origins, with nothing guessed. All dimensions are **params at the top of
`generate_hexabot.py`**, calibrated to the measured reference parts + photos.

---

## What's in this folder

```
hexabot_model/
├── generate_hexabot.py        # SOURCE OF TRUTH — params → meshes + URDF + report
├── HEXABOT.md                 # this file
├── isaac/
│   ├── hexabot.urdf           # 19 links, 18 DOF, box/cyl collisions, analytic inertia
│   ├── meshes/                # base_link.obj + coxa/femur/tibia.obj (+ hexabot.mtl)
│   ├── hexabot_cfg.py         # Isaac Lab ArticulationCfg (3 actuator groups)
│   ├── tripod_gait.py         # open-loop tripod-gait demo (run on the GPU box)
│   ├── validate_urdf.py       # offline sanity checks (run anytime)
│   └── inertia_report.md      # masses, CoMs, inertia, validation
├── previews/
│   ├── render_hexabot.py      # local renders + gait GIF (runs on this Mac)
│   └── iso.png top.png front.png gait.gif
└── reference model/, reference images/   # the print-ready source + photos
```

**Only 4 unique meshes** exist: `base_link`, `coxa`, `femur`, `tibia`. All 6 legs
are identical in their own local frame, so the 18 leg links share 3 meshes — the
URDF just instantiates them at different joint origins.

**Regenerate** (only if you edit the params):
```bash
python3 hexabot_model/generate_hexabot.py        # rebuild meshes + URDF + report
python3 hexabot_model/isaac/validate_urdf.py     # confirm still valid
python3 hexabot_model/previews/render_hexabot.py # refresh renders + gait.gif
```

---

## Numbers worth knowing (from `inertia_report.md`)

- **Total mass 1.926 kg** — base 910 g (210 g shell + 250 g LiPo + 120 g
  electronics + 6× coxa servo) + 6×(coxa 64 g + femur 76 g + tibia 29 g).
- **18× MG996R = 990 g** of the total (6 in the body, 12 in the legs) — same servo
  as TARS.
- **Standing body height ≈ 72 mm**, whole-body **CoM ≈ 75 mm** up and centred
  (x≈0, y≈0) → low and well inside the foot polygon → statically stable.
- **Stance:** coxa 0°, femur −18°, tibia 64° → feet at **295 mm** radius
  (span ≈ **590 mm**).
- **Kinematic stride ≈ 303 mm/cycle** in the demo gait (tune via `--coxa-amp`,
  `--period`).
- Every inertia tensor is **positive-definite + triangle-valid** (the validator
  checks this — PhysX silently destabilises otherwise).

---

## Hexabot vs. TARS — why this should walk better

| | **TARS (biped)** | **Hexabot** |
|---|---|---|
| Actuated DOF | **4** (2 hip + 2 ankle, all pitch, 1 plane) | **18** (6 legs × coxa/femur/tibia) |
| Links / joints | 5 / 4 | 19 / 18 |
| Standing stability | **Inverted pendulum** — must actively balance | **Statically stable** — a tripod is always planted |
| Lateral DOF | none (no ankle-roll) → marginal sideways | coxa yaw + 6 ground contacts → inherently stable |
| Falls if controller is off? | **Yes**, immediately | **No**, it just stands |
| Walking approach | needs RL/CPG just to not topple | **open-loop tripod gait already walks** (this repo) |
| Mass / servo | 1.44 kg / 4× MG996R | 1.93 kg / 18× MG996R |
| CoM height | 125 mm (of 235 mm stand) | 75 mm (of 72 mm body) — very low |
| Per-stance-leg load | ~½ body on 2 feet, dynamic | ~⅓ body on ≥3 feet, static |

The headline: **TARS needed a learned policy just to stay upright; the hexapod is
statically stable and walks with a scripted gait.** Same servos, fundamentally
easier control problem. RL becomes *polish* (speed, rough terrain), not a
prerequisite for not-falling-over.

Locally verified on the Mac (no GPU): `previews/gait.gif` shows the alternating
tripod advancing **+X** — base travels 0 → 151 → 303 → 454 mm across the cycles
while legs alternate swing/stance.

---

## Phase 8 — run it on the GPU box

Copy the **`isaac/` folder** to your Linux/Windows + RTX machine (keep
`hexabot.urdf` and `meshes/` together — the URDF references meshes by relative
path). Install Isaac Sim + Isaac Lab per the official guide (same as the TARS
`ISAAC_LAB_SETUP.md` Step 0).

### 1 · Convert URDF → USD
```bash
# path varies by version — check your tree, then read --help:
./isaaclab.sh -p scripts/tools/convert_urdf.py --help
./isaaclab.sh -p scripts/tools/convert_urdf.py \
    ~/hexabot/isaac/hexabot.urdf  ~/hexabot/isaac/hexabot.usd \
    --joint-stiffness 12.0 --joint-damping 0.4     # flags vary; --help is truth
```
Import settings that matter (if you use the GUI importer instead): **Articulation**,
**Z-up** stage, **metres / mesh scale 1.0** (meshes are already metres — don't
re-scale by 0.001), **Fixed Base OFF** (floating base for locomotion), **Self
Collision OFF** (legs are radially separated), collision + inertia **from URDF**.

### 2 · Verify the ArticulationRoot (the silent killer)
In the Stage, the **Physics → Articulation Root** API must be on **exactly one**
prim — the top robot prim (or `base_link`), not several, not missing. Otherwise the
robot appears but ignores all joint commands.

### 3 · Smoke test
Add a ground plane + light, press **Play**. It should **not explode** (units +
inertia are pre-validated), should **stand** on its 6 feet (it won't topple — that's
the whole point), and nudging any `coxa_*`/`femur_*`/`tibia_*` in the Articulation
Inspector should move the right joint.

### 4 · Watch it walk (the demo)
```bash
./isaaclab.sh -p ~/hexabot/isaac/tripod_gait.py --usd ~/hexabot/isaac/hexabot.usd
#   --headless to skip the window · --period 1.6 --coxa-amp 0.26 --lift 0.55 to tune
```
It settles into stance, then runs the alternating-tripod gait and **walks +X**.
The console prints base-x progress (mm) each second. This is the direct
counterpart to the TARS sim run — except TARS needed a trained policy to move at
all, and this walks open-loop.

### 5 · (Optional) wrap as an Isaac Lab RL task
Use `hexabot_cfg.py` (`HEXABOT_CFG`) as the robot in a velocity-tracking locomotion
env (copy an Anymal/quadruped manager-based task). Action space = **18** joints;
reward = track a commanded base velocity + penalise energy / joint limits, with
feet-contact terms on `tibia_*`. The open-loop gait makes a great **warm start** or
sanity baseline.

---

## Conventions & where to change things

**Frame:** metres, **+X forward, +Y left, +Z up**. `base_link` at the body centre;
standing height ≈ 0.072 m. Coxa joints carry the azimuth as `rpy=(0,0,θ)`; coxa
axis = `(0,0,1)`, femur/tibia axis = `(0,1,0)`.

| Want to change… | Edit… | Then run… |
|---|---|---|
| Body size, leg lengths, azimuths, stance | PARAMS atop `generate_hexabot.py` | `generate_hexabot.py` → `validate_urdf.py` |
| Joint limits / effort / velocity | `LIM`, `EFFORT`, `VEL` in `generate_hexabot.py` | `generate_hexabot.py` |
| Masses / densities | `*_G`, `ABS_DENSITY`, `PRINT_FILL`, `SV_MASS` | `generate_hexabot.py` |
| Actuator gains in sim | `hexabot_cfg.py` | (in Isaac Lab) |
| Gait amplitude / speed | `tripod_gait.py` flags, or `render_hexabot.py` consts | — |

> **Honest caveat:** leg joint-to-joint lengths are calibrated to *print-oriented*
> reference-STL bounding boxes + photos, not an assembled CAD source. They're all
> params — if you later measure the real assembled robot, update `L_COXA / L_FEMUR
> / L_TIBIA / R_COXA` and re-run. The kinematic *structure* (axes, tree, tripod
> groups) is exact regardless.
