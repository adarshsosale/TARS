# Growbot → Isaac Lab — Setup & Handoff

Everything needed to get the Growbot model simulating in **NVIDIA Isaac Lab**.
I (Claude) finished the model-prep phases (1–7) on this Mac. **Phase 8 (running
Isaac) is yours** because it needs an NVIDIA GPU — see the blocker right below.

---

## ⚠️ Read this first — you cannot run Isaac on this Mac

Isaac Sim / Isaac Lab require an **NVIDIA RTX GPU** and run only on **Linux or
Windows**. There is **no macOS / Apple-Silicon build**. So nothing in "Phase 8"
runs on your MacBook. Your options:

| Option | Notes |
|---|---|
| A Linux/Windows desktop with an RTX GPU (≥ RTX 3070 / 8 GB; 30xx/40xx ideal) | Cheapest long-term if you'll iterate a lot |
| A cloud GPU box (AWS `g5`/`g6`, Lambda, RunPod, Vast.ai, etc.) | Spin up an "Isaac Sim" image or an Ubuntu 22.04 + driver box; pay per hour |
| NVIDIA Omniverse / Isaac on a remote workstation | If your org has one |

You'll copy the **`isaac_lab/` folder** (URDF + meshes) to that machine. The
assets themselves are plain text/OBJ and are already done — they don't care what
made them.

---

## TL;DR

- **Done (this Mac):** mesh cleanup + a real **normals bug fixed**, unit
  conversion to metres, decomposition into 5 links, **box collision** geometry,
  a hand-authored **URDF** with exact joint origins, **analytic inertia
  tensors**, and an offline **validator** (all checks pass).
- **You do (GPU box):** install Isaac → convert URDF to **USD** → verify the
  **ArticulationRoot** → smoke-test → wrap as an Isaac Lab task → train.
- **Start here:** copy `isaac_lab/` to your GPU machine, then follow
  [Phase 8](#phase-8--what-you-do-on-the-gpu-box).

---

## Status of the 8 phases

| # | Phase | Status | Where / how |
|---|---|---|---|
| 1 | Mesh inspection & **normals** | ✅ Done — **found & fixed an inverted-normals bug** | `generate_growbot.py` (cyl_y + link_bar winding); proof in `isaac_lab/inertia_report.md` |
| 2 | **Unit** conversion (mm → m) | ✅ Done — meshes pre-scaled to metres | `isaac_lab/meshes/*.obj` |
| 3 | **Decompose** into rigid links | ✅ Done — 30 OBJ groups → 5 links | `isaac_lab/meshes/` |
| 4 | **Collision** geometry | ✅ Done — **box primitives** (no V-HACD needed) | `<collision>` in `growbot.urdf` |
| 5 | **URDF** authoring (joints) | ✅ Done — 4 revolute joints, exact origins | `isaac_lab/growbot.urdf` |
| 6 | **Inertia / mass** properties | ✅ Done — analytic, validated | `isaac_lab/inertia_report.md` |
| 7 | URDF **validation** | ✅ Done — custom validator passes; `urdfpy` optional | `isaac_lab/validate_urdf.py` |
| 8 | **Isaac import → USD → RL task** | ⏳ **You** (needs GPU) | this document |

Why so much got done: your model is **parametric** (`generate_growbot.py` is the
source of truth). I pulled joint origins, dimensions and per-part volumes
straight from it, so the URDF and inertia are computed, not guessed — and far
more accurate than the "Blender volume estimate" the original task list assumed.

---

## What I produced (in `isaac_lab/`)

```
isaac_lab/
├── growbot.urdf               # 5 links, 4 DOF, box collisions, analytic inertia
├── meshes/
│   ├── base_link.obj          # torso + 2 hip servos + battery + electronics + rest foot
│   ├── leg_left_link.obj      # leg shell + ankle servo + crank/pushrod + wiring + screws
│   ├── leg_right_link.obj
│   ├── foot_left_link.obj     # foot shell + TPU sole + rocker
│   ├── foot_right_link.obj
│   └── Growbot_TARS.mtl       # colours (copied)
├── build_isaac_assets.py      # regenerates everything above from the model
├── validate_urdf.py           # offline sanity checks (run anytime)
├── growbot_cfg_TEMPLATE.py    # Isaac Lab ArticulationCfg starter (edit on GPU box)
└── inertia_report.md          # masses, CoMs, inertia, validation results
```

**Regenerate** (only if you edit the model):
```bash
python3 generate_growbot.py            # rebuild the OBJ from params
python3 isaac_lab/build_isaac_assets.py  # rebuild URDF + meshes + inertia
python3 isaac_lab/validate_urdf.py       # confirm still valid
```

Numbers worth knowing (from the report):

- **Total mass 1.438 kg** (the HANDOFF's 1.401 kg + ~37 g of steel screws/wire
  that its quick estimate left out — verified).
- **Whole-body CoM 125 mm up**, ≈ the HANDOFF's independent ~124 mm estimate —
  a strong cross-check that the mass + inertia + axis transform are right.
- Per-link: base 774 g · each leg 244 g · each foot 88 g.

---

## Key decisions & assumptions (override if you disagree)

1. **Coordinate frame.** The model is Y-up (mm); the URDF is the standard
   **Z-up, X-forward, metres**. Mapping applied: `x_urdf=z_model`,
   `y_urdf=x_model`, `z_urdf=y_model` (a pure rotation — meshes/normals
   unchanged). So in sim: **+X = walking forward, +Y = left, +Z = up**, and the
   hip/ankle pitch axis is **+Y → `axis="0 1 0"`**. Do your import into a **Z-up
   stage** (Isaac's default) or the robot will lie on its side.

2. **`base_link` frame is at hip height** (≈ 0.23 m above the soles), not at the
   geometric bottom. Standing base height ≈ **0.235 m** — a normal positive
   number for RL height rewards.

3. **The ankle is ONE revolute joint.** The real ankle is a crank→pushrod→rocker
   **4-bar linkage** (a *closed* kinematic loop). PhysX articulations are
   **trees** and can't close a loop, so I model the foot pivoting **directly
   about the hinge pin**. The joint limit **±0.855 rad (±49°)** already encodes
   the pushrod's reachable foot range, and the effort/speed already include the
   1.23× ratio. The crank/pushrod/rocker meshes are **visual-only** and will
   appear to *disconnect* as the ankle moves — **this is cosmetic, physics is
   correct**. To hide them: delete those prims in the USD, or tell me and I'll
   regenerate the meshes without them.

4. **Collision = boxes, not V-HACD/CoACD.** Every structural part (torso, legs,
   feet) is essentially a box, so I used **box primitives** with exact
   dimensions. This is faster and far more stable in PhysX than convex hulls,
   and it sidesteps the manifold/overlap issues in the decorative panels (which
   are overlapping solids — fine for *visuals*, bad for *collision*). You only
   need V-HACD if you later want fingers/curved contact surfaces — you don't.

5. **Masses.** Printed parts = volume × density (ABS 1.04, CF 1.10, TPU
   1.20 g/cm³); bought parts = catalog (MG996R 55 g ×4, LiPo 194 g, electronics
   106 g). Steel screws by volume × 7.85.

6. **Joint limits / effort / velocity** come from the MG996R (≈1.08 N·m,
   ≈6.16 rad/s) and the HANDOFF's drivetrain ratios. These are your
   **sim-to-real anchors** — keep them honest.

7. **Inertia** is the analytic sum of box/cylinder tensors (parallel-axis to
   each link's CoM). The validator confirms every tensor is **positive-definite
   and obeys the triangle inequality** — PhysX silently destabilises otherwise.

---

## Phase 8 — what you do (on the GPU box)

### Step 0 · Install Isaac Sim + Isaac Lab

Follow the **official Isaac Lab install guide** (search "Isaac Lab installation
documentation"). The shape of it:

1. Ubuntu 22.04 (or Windows) with a recent **NVIDIA driver**.
2. Install **Isaac Sim** (pip workflow or the Omniverse binary).
3. Clone **Isaac Lab**, run its installer:
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab.git
   cd IsaacLab
   ./isaaclab.sh --install        # Linux  (isaaclab.bat on Windows)
   ```
4. Verify:
   ```bash
   ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
   ```
   If an empty Isaac window opens, you're good.

> **Version match matters.** Use an Isaac Sim version your Isaac Lab release
> supports (the Isaac Lab README states the pairing). Mismatches are the #1
> install headache.

Then copy your `isaac_lab/` folder (this repo's) onto that machine, e.g.
`~/growbot_assets/isaac_lab/`. Keep `growbot.urdf` and `meshes/` **together** —
the URDF references meshes by relative path.

### Step 1 · Convert the URDF → USD

**Recommended: the headless converter** (reproducible, no clicking). Isaac Lab
ships one; the path varies by version — find it and read its help:

```bash
# path is usually one of these — check your tree:
#   scripts/tools/convert_urdf.py      (newer)
#   source/standalone/tools/convert_urdf.py   (older)
./isaaclab.sh -p scripts/tools/convert_urdf.py --help

./isaaclab.sh -p scripts/tools/convert_urdf.py \
    ~/growbot_assets/isaac_lab/growbot.urdf \
    ~/growbot_assets/isaac_lab/growbot.usd \
    --joint-stiffness 10.0 --joint-damping 0.3   # flags vary; --help is truth
```

**Alternative: the GUI URDF Importer** (Isaac Sim → top menu, *Isaac Utils →
Workflows → URDF Importer*, or *File → Import*). Settings that matter:

| Setting | Value | Why |
|---|---|---|
| Import as | **Articulation** | makes it one jointed robot, not loose parts |
| Stage up axis | **Z** | matches our URDF; default in Isaac |
| Distance / stage units | **metres**, mesh scale **1.0** | meshes are already metres — **don't** re-scale by 0.001 |
| **Fixed Base Link** | **OFF** | floating base for locomotion (turn ON only to inspect a pose) |
| Self Collision | **OFF** | legs/torso are laterally separated; faster + stable |
| Collision source | from URDF `<collision>` | uses our **boxes**, not the messy visual mesh |
| Inertia | from URDF `<inertial>` | uses our validated tensors — don't auto-recompute |
| Joint drive | Position (or leave default) | we override gains in Isaac Lab anyway |

After import, **Save As** `growbot.usd`.

### Step 2 · Verify the ArticulationRoot ← the silent killer

Your task note called this out, correctly. If it's wrong, the robot appears in
the scene but **ignores all joint commands**.

1. In the **Stage** panel, select the **top robot prim** (`/World/growbot` or
   similar).
2. In **Properties**, confirm there's a **Physics → Articulation Root** API on
   **exactly one** prim — the top-level robot (or `base_link`), **not** on
   several prims, and **not** missing.
3. Fixes: missing → *Add → Physics → Articulation Root*. Duplicated/on the wrong
   prim → remove the extras so only the root has it.

### Step 3 · Smoke test (before any RL)

1. Add a **ground plane** and a **light**, press **Play**.
2. Check, in order:
   - **It doesn't explode/launch** off-screen → units + inertia are sane (we
     pre-verified; if it *does* explode, see Troubleshooting).
   - It **falls plausibly** under gravity (a passive biped *will* topple — fine).
   - **Feet touch the ground** (not sinking through, not floating 8 mm up — the
     central rest foot *is* meant to sit ~8 mm above ground; the two main feet
     should contact).
3. Open the **Articulation Inspector** (or joint drive sliders) and nudge
   `hip_left`, `ankle_left`, … — confirm each makes a **pitch** (sagittal) motion
   about the lateral axis. That proves the joints, axes and root are correct.

### Step 4 · Wrap it as an Isaac Lab articulation

1. Open `isaac_lab/growbot_cfg_TEMPLATE.py`, set `usd_path` to your
   `growbot.usd` absolute path. It already lists the right link/joint names and
   sensible starting actuator gains.
2. Stand up an environment the easy way: **copy an existing Isaac Lab locomotion
   task** (under `isaaclab_tasks`, e.g. a velocity-tracking flat-terrain biped or
   the `Anymal`/`H1`-style manager-based env) and swap in:
   - `GROWBOT_CFG` as the robot,
   - the **4 joint names** (`hip_left/right`, `ankle_left/right`) into the action
     term and any joint-position observations,
   - `base_link` / `foot_*_link` into the contact-sensor and reward terms.
3. The **action space is 4** (four pitch joints). Observations: base lin/ang
   velocity, projected gravity (orientation), joint pos/vel, last action, and a
   velocity command. Start from the example's defaults and prune what you don't
   have.

### Step 5 · Train the locomotion / CPG policy

- Easiest supported route: **PPO** via the example task's `rsl_rl` or `skrl`
  runner, reward = **track a commanded base velocity**, with penalties on energy,
  joint-limit proximity, and feet air-time/contact. Then:
  ```bash
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
      --task <your-registered-task-id> --headless
  ```
- For a **CPG** approach: drive the 4 joints with coupled oscillators and use RL
  to tune the CPG parameters (frequency, amplitude, phase offsets, and a
  feedback gain from the IMU/base tilt). The 4 pitch DOF give a sagittal
  shuffling gait; **lateral balance is marginal** by design (the HANDOFF notes
  there's *no ankle-roll DOF*). Mitigations already baked in: low CoM (125 mm),
  long/wide feet. If you need real lateral stability, the cleanest model change
  is to add an **ankle-roll joint** per foot (tell me and I'll add it to the
  generator + URDF).

> Your note's tip — "if the gait diverges, suspect inertia first" — is good
> general advice, but **our inertia is analytic and validated**, so for *this*
> model the more likely culprits are, in order: **drive gains** (Step 4),
> **contact params** (friction/offsets), and the **missing lateral DOF**.

---

## Troubleshooting (mapped to the classic failure modes)

| Symptom | Cause | Fix |
|---|---|---|
| Robot **lying on its side** after import | up-axis mismatch | import into a **Z-up** stage (our URDF is Z-up) |
| Robot **explodes / flies off** on first Play | units or bad inertia | meshes are **already metres** → mesh scale **1.0** (don't apply 0.001 twice); we pre-validated inertia, so check the scale first; then raise solver position iterations / lower dt |
| Robot **appears but won't move** to commands | **ArticulationRoot** missing/misplaced, or no drive | redo Step 2; set actuators (Step 4) |
| **Feet jitter / sink / slide** | contact tuning | friction ≈ 0.8–1.0, tune contact/rest offset, raise solver iterations |
| Ankle **linkage visually detaches** | the 4-bar is **visual-only** | expected — hide `foot_rocker_*`/`ankle_crank_*`/`ankle_pushrod_*`, or ask me to drop them |
| Legs **clip the torso** | won't happen (laterally separated); only if you force self-collision | keep **self-collision OFF** |
| Masses look wrong in the Property panel | importer recomputed inertia | re-import with "use URDF inertia"; cross-check against `inertia_report.md` |

---

## Where to change things (it's all parametric)

| Want to change… | Edit… | Then run… |
|---|---|---|
| Geometry, joint positions, sizes | `generate_growbot.py` (params at top) | `generate_growbot.py` → `build_isaac_assets.py` |
| Joint limits / effort / velocity | `JOINT_TUNING` in `build_isaac_assets.py` | `build_isaac_assets.py` |
| Collision boxes | `collisions` block in `build_isaac_assets.py` | `build_isaac_assets.py` |
| Densities / catalog masses | `DENSITY` / `CATALOG_FIXED` in `build_isaac_assets.py` | `build_isaac_assets.py` |
| Actuator gains in sim | `growbot_cfg_TEMPLATE.py` | (in Isaac Lab) |

Always finish with `python3 isaac_lab/validate_urdf.py`.

---

## Appendix · Frames, names, transform

**World:** +X forward · +Y left · +Z up · metres. **`base_link`** at hip height.

```
base_link ──hip_left  (rev, axis +Y, ±90°)──► leg_left_link  ──ankle_left  (rev, axis +Y, ±49°)──► foot_left_link
          └─hip_right (rev, axis +Y, ±90°)──► leg_right_link └─ankle_right (rev, axis +Y, ±49°)──► foot_right_link
```

| Joint | Origin (m, parent frame) | Axis | Limit (rad) | Effort | Vel |
|---|---|---|---|---|---|
| `hip_left`  | (0, −0.069, 0)   | (0,1,0) | ±1.571 | 1.08 | 6.16 |
| `hip_right` | (0, +0.069, 0)   | (0,1,0) | ±1.571 | 1.08 | 6.16 |
| `ankle_left`  | (0, 0, −0.187) | (0,1,0) | ±0.855 | 1.33 | 5.0 |
| `ankle_right` | (0, 0, −0.187) | (0,1,0) | ±0.855 | 1.33 | 5.0 |

Model (mm, Y-up) → URDF (m, Z-up): `x_urdf = z_model·0.001`,
`y_urdf = x_model·0.001`, `z_urdf = y_model·0.001`.

**Optional external validators** (if you want a second opinion on the GPU box):
`pip install yourdfpy` then `yourdfpy growbot.urdf`, or load it in PyBullet
(`p.loadURDF`). Neither is required — `validate_urdf.py` already checks topology,
units, mesh resolution, and inertia validity.
