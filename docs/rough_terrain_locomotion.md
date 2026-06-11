# Hexabot Rough-Terrain Locomotion — Technical Reference

*Milestone 1. How the 18-DOF hexapod walks over uneven ground, in enough detail to
reason about where to improve it. Source of truth: `isaac_lab/tasks/hexabot/`.*

---

## 1. What the system actually is

The rough-terrain walker is **not** a from-scratch policy. It is the flat-ground
locomotion policy (`HexabotEnv` / `HexabotFlatEnvCfg`) **subclassed** and given three
additions and nothing else:

1. a **privileged height-scanner** sensor (teacher-only exteroception),
2. a **terrain-relative ground height** so the inherited reward/death terms work on
   slopes and steps, and
3. a **terrain-level curriculum** that ramps difficulty per-environment.

The action mechanism (a CPG), the frozen `(vx, vy, yaw)` velocity-command interface,
the entire reward family, and the domain randomization are **inherited unchanged**.
This matters for improvement work: most of the behaviour you observe is set by the
*flat* config; the rough layer only adds perception, terrain-relativity, and a
difficulty schedule.

Task id: `Isaac-Velocity-Rough-Hexabot-Direct-v0`
(`hexabot_rough_env.py`, `hexabot_rough_env_cfg.py`, PPO cfg
`HexabotRoughPPORunnerCfg`).

```
   (vx,vy,yaw) command
        │
        ▼
  ┌───────────────┐   action = per-leg [Δfreq, Δcoxa_amp, Δlift] (18)
  │  PPO policy   │ ─────────────────────────────────────────────────┐
  │  (teacher)    │                                                   │
  └───────────────┘                                                   ▼
        ▲                                              ┌──────────────────────────┐
        │ obs = [ proprio(75) | height_scan(63) ]      │  CPG (cpg.py)            │
        │                                              │  zero action == tripod   │
        │                                              └──────────────────────────┘
        │                                                            │ joint targets (18)
   ┌──────────────────────────────────────────────┐                 ▼
   │ height scanner (9×7 rays) + proprio sensors   │◀──── 200 Hz PhysX, 50 Hz control
   └──────────────────────────────────────────────┘
```

---

## 2. Control architecture — the CPG action

The single most important design choice. The policy does **not** emit joint angles.
It **modulates a Central Pattern Generator** (`cpg.py:HexabotCPG`).

- Each leg `i` carries an absolute phase `θ_i ∈ [0, 2π)`, advanced every control step.
- Phases are **seeded at the alternating-tripod offsets** (tripod A = {lf, rm, lr}
  share a phase; tripod B = {rf, lm, rr} lag by half a cycle) and held there by **weak
  phase coupling** (`cpg_coupling_strength = 2.0`). At the tripod equilibrium the
  coupling term is exactly zero, so **zero action reproduces the analytical tripod gait
  bit-for-bit** (verified in `test_cpg.py`).
- The 18-d action is per-leg `[Δfreq, Δcoxa_amp, Δlift]` (leg-major):

  ```
  f_i    = f_base   * (1 + k_f    · Δfreq_i)        # cadence    (k_f=0.5,  ±50%)
  μ_i    = μ_base   · s · (1 + k_μ · Δcoxa_amp_i)   # coxa sweep (k_μ=0.5,  ±50%)
  lift_i = lift_base· s · (1 + k_l · Δlift_i)       # swing lift (k_l=0.6,  ±60%)
  ```

  `f_base=1.0 Hz`, `μ_base=0.26 rad`, `lift_base=0.55 rad`. `s` is `speed_scale`.
- **`speed_scale`** ties stride amplitude to commanded speed:
  `s = max(|vx|/0.30, |yaw|/0.5)` clamped to `[0, 1.2]`. At `s=0` (zero command) the
  coxa sweep and lift collapse to zero → the robot **holds the standing stance**. This
  **structurally prevents belly-crawl** without a long stand-curriculum, and means
  "stop" is a well-defined zero-amplitude state, not a learned behaviour.
- Joint mapping reuses the analytical waveform: in **stance** (`θ/2π < 0.5`) the coxa
  ramps to sweep the body backward (push-off); in **swing** the femur lifts
  (`qf = STANCE_FEMUR − lift·sin(πs_swing)`) and the tibia tucks
  (`qt = STANCE_TIBIA + 0.4·lift·sin(πs_swing)`). Output is **absolute joint targets**,
  then clamped to soft joint limits.

**Why this helps on rough terrain:** the gait *topology* (which feet are up when) is
guaranteed by the seeded CPG and coupling; the policy only retimes and reshapes it. It
cannot produce an incoherent flailing gait, and the `Δlift` channel gives it a direct
knob to raise swing feet over bumps and steps.

The **per-leg phase enters the observation** as `[sin θ, cos θ]` (12 values), so the
policy knows where in the cycle each leg is.

---

## 3. Perception — the privileged height scanner

- `RayCasterCfg` attached to `base_link`, **yaw-aligned** (heading-relative,
  roll/pitch-invariant — like a real terrain estimate), one scan per control step.
- Grid pattern `0.4 × 0.3 m` at `0.05 m` resolution → **9 × 7 = 63 rays**. The
  footprint is scaled to the robot's ~0.59 m foot span.
- Each ray returns a **clearance** value:
  `height = scanner_z − hit_z − offset`, with `offset = 0.072` (nominal standing
  height, so flat ground reads ≈ 0) and clipped to `±0.15 m`. Rays that miss the mesh
  fall back to the scanner height (finite).
- A separate **median over finite hits** gives `self._ground_height` — the
  terrain-relative ground level fed to the reward/death terms (§5).

The scan is appended to the proprioceptive obs, taking `observation_space` from
**75 → 138**.

> **It is PRIVILEGED.** It is teacher-only. The whole point of Milestone 1 is to train
> a policy whose exteroception can later be *distilled away* into a blind student. See §6.

---

## 4. Observation layout (138-d)

```
[ grav(3) ang_vel(3) cmd(3) jpos−def(18) jvel(18) prev_act(18) cpg_phase(12) | height_scan(63) ]
  └──────────────────────── proprioceptive only, 75 ───────────────────────┘   └ privileged ┘
```

Deliberately **no base linear velocity** — it is not measurable on the real robot, so
it is excluded as a hard constraint. The policy infers forward motion from gravity,
angular velocity, joint state, the command, and the CPG clock. Proprioceptive sensors
get IMU/encoder noise injected (§7).

---

## 5. Reward — inherited family, made terrain-relative

All reward terms come from the flat env (`hexabot_env.py:_get_rewards`). On rough
terrain only two things change: height terms become ground-relative, and the
straight-line shaping is softened.

**Primary translation drivers**
| Term | Scale | Notes |
|---|---|---|
| `forward_progress` | **+12.0** | `clamp(min(vx_body, cmd_vx), 0)` — **linear, zero at standstill**. The real driver. |
| `stationary_penalty` | **−15.0** | velocity shortfall × moving-mask — makes standing *actively costly* when commanded to move. |
| `track_lin_vel_xy_exp` | +0.5 | saturating exp tracker, demoted to minor speed-regulation. |
| `track_ang_vel_z_exp` | +1.5 | yaw-rate tracking (turning). |

**The motion gate (critical for rough terrain).** Both exp trackers are multiplied by a
**motion gate** = `clamp(max(0,vx)/0.10 + |yaw|/0.30, max=1)`. A frozen robot earns ~0
from them; a robot tracking its command sees the gate fully open and gets the full
bonus. *Without the gate, on rough terrain the death-risk of stepping makes a guaranteed
standstill the safe optimum* — the saturating exp terms pay a frozen robot nearly full
marks, the curriculum then demotes everyone back to flat, and the policy collapses to
standing. This is the documented "saturating exp term re-creates the static stance"
landmine, tripped specifically by terrain death-risk.

**Stability / posture** (`flat_orientation −2.5`, `base_height_l2 −8.0`,
`belly_clearance −50.0` one-sided, `foot_support +10.0`, `alive +1.0`, `lin_vel_z −1.0`,
`ang_vel_xy −0.10`).

**Gait quality** (all **motion-gated** — only active when commanded to move):
- `tetrapod_contact +3.0` — bell centered on exactly 4 of 6 feet planted.
- `gait_symmetry +1.5` — planted/lifted feet form a left-right mirror.
- `gait_phase +3.0` — net reward for lifting CPG-scheduled-swing feet minus lifting
  scheduled-stance feet (reads the CPG phase, so it's a *phase-locked* gait check; a
  static stance scores 0).
- `foot_clearance +4.0` up to `0.025 m` apex — deliberate swing lifts (anti-skitter).
- `foot_plant +0.6` — stance feet point steeply down so the claw digs in (anti-slip).
- `feet_air_time +2.5` (threshold 0.2 s) — longer swings, bigger strides.
- `foot_slip −0.2`.

**Effort / smoothness** (`dof_torques −2e-5`, `dof_acc −2.5e-7` *(hard limit — raising
breaks the stand phase)*, `action_rate −0.017`, `dof_pos_limits −1.0`,
`undesired_contacts −1.0`).

**Softened on rough terrain** (`hexabot_rough_env_cfg.py`):
- `heading_reward_scale`: −4.0 → **−1.0**
- `lateral_pos_reward_scale`: −2.0 → **−0.5**

Rough ground legitimately yaws and drifts the body crossing slopes and steps, so the
strict straight-line shaping from flat is too harsh and would fight terrain traversal.

**Terrain-relativity.** `self._ground_height` (median of the scan) is subtracted in:
`base_height_l2`, `belly_clearance`, `foot_support`, `foot_clearance`, `foot_plant`, and
the `too_low` death. On flat ground it is 0 (a no-op); on terrain it makes "height" mean
"height above the ground directly under me," so a robot on top of a step isn't punished
for being higher in world-z.

---

## 6. The distillable teacher (`teacher_policy.py`)

The structural piece that makes this a *teacher*, not a final policy. The privileged
height scan enters the actor **only through a narrow latent bottleneck**, kept isolated
from the proprioceptive path:

```
z      = scan_encoder(height_scan[63] → 16)      # the bottleneck (MLP 63→128→64→16)
action = actor_trunk( [proprio(75) | z(16)] → 18 )   # MLP 91→128→128→128→18
```

This is the RMA / teacher-student decomposition. The **critic is privileged** — it
consumes the full 138-d obs directly (fine, since the critic is discarded at deploy).

**Distillation seam for the next milestone:** keep `actor_trunk` **verbatim**, replace
`scan_encoder` with a *proprioceptive-history encoder* trained to regress the same `z`.
Because the trunk only ever sees `[proprio | z]` and never the raw scan, the student is
a clean module swap, not a re-architecture. `play_rough.py` exports the **full** teacher
(encoder + trunk, traced) so `z` is well-defined — the stock rsl_rl exporter would grab
only the trunk.

`HexabotTeacherActorCritic` subclasses rsl_rl's `ActorCritic`, so PPO, symmetry
augmentation, empirical normalizers, and checkpoint plumbing all keep working; only the
actor's internal forward is overridden. It is injected into rsl_rl's runner namespace by
`train_rough.py` so `class_name="HexabotTeacherActorCritic"` resolves.

---

## 7. Domain randomization

DR is **on from the start** (both flat and rough). It is the sim-to-real budget and also
shapes robustness on terrain.

**Physics-side** (`EventCfg` / `RoughEventCfg`, via the event manager):
- Friction spread: static `(0.8, 1.3)`, dynamic `(0.6, 1.0)`, 64 buckets, startup.
- Body mass scale `(0.85, 1.15)`, leg mass scale `(0.9, 1.1)`, startup.
- Actuator gain scale (stiffness & damping) `(0.8, 1.2)` per **reset**.
- **Rough adds:** payload mass `+(−0.1, 0.2) kg` on `base_link`, and a base
  centre-of-mass shift `±0.02 m` (x,y), `±0.01 m` (z) — so the teacher doesn't overfit
  to a nominal inertia on uneven ground.

**Signal-side** (`DomainRandCfg`, applied in the env step loop):
- Actuator latency `0–2` control steps (`0–40 ms @ 50 Hz`), per-env, held per episode.
- Control-rate jitter `±10%` on the CPG-advance dt; `hold_prob = 0.02` (dropped control
  tick → reuse last command).
- IMU/obs Gaussian noise: gravity `0.02`, ang-vel `0.10 rad/s`, joint-pos `0.01 rad`,
  joint-vel `0.15 rad/s`.

---

## 8. The terrain and its curriculum

**Terrain generator** (`rough_terrains.py`), `2.0 × 2.0 m` tiles,
`horizontal_scale=0.025`, `vertical_scale=0.0025`, `10 rows × 10 cols`, `curriculum=True`
(row index == difficulty). Everything is **scaled hard down** for the 72 mm robot
(default Isaac Lab locomotion terrains, sized for ANYmal, would be impassable). All
difficulty knobs are **named constants at the top of `rough_terrains.py`** (`SLOPE_MAX`,
`ROUGH_NOISE_MAX`, etc., with the previous value noted inline) and are quoted at the
**hardest** curriculum level (difficulty 1); they scale linearly to ~flat at level 0.

This revision's headline change: **mixed-feature tiles** that overlay one small height
field on another (bumps on a slope, uneven steps, small boxes on a ramp) are **now the
main difficulty lever** (~0.54 of the mix vs ~0.36 for the pure single-feature tiles;
the generator normalises by the sum). They are custom height-field functions
(`sloped_rough_terrain`, `stairs_rough_terrain`, `sloped_boxes_terrain`) that **sum the
stock Isaac Lab generators' height arrays**, with a difficulty-scaled overlay
(`_difficulty_noise`) and the central spawn pad zeroed (`_clear_center`) so the robot
never spawns on a bump or box edge.

| Sub-terrain | Proportion | Range (hardest level) |
|---|---|---|
| **Pure** — Pyramid slope up / down | 0.05 + 0.05 | slope `0.0–0.20` (was `0.15`) |
| **Pure** — Random rough / noisy ground | 0.10 | bumps `0.005–0.035 m`, downsampled `0.05 m` |
| **Pure** — Random grid boxes (discrete steps) | 0.08 | height `0.0–0.035 m`, grid 0.3 m |
| **Pure** — Pyramid stairs up / down | 0.04 + 0.04 | step `0.008–0.04 m`, tread `0.22 m` (was `0.18`) |
| **Mixed** — Slope + rolling noise up / down | 0.12 + 0.12 | slope `≤0.20` + `±0.02 m` overlay |
| **Mixed** — Stairs + rolling noise up / down | 0.10 + 0.10 | step `≤0.04 m` + `±0.02 m` overlay |
| **Mixed** — Slope + small flat-topped boxes up / down | 0.05 + 0.05 | slope `≤0.20` + `0.025 m` boxes |

Single-feature ranges were also pushed harder (slope `0.15→0.20`, rough `0.03→0.035`,
box `0.025→0.035`, stair `0.03→0.04`, tread `0.18→0.22 m` so a claw foot fits a step;
the mesh stairs keep a small positive step floor `0.008 m` to avoid degenerate trimesh
boxes at near-zero height). The mixed overlay is deliberately **smaller** than the
standalone rough (`±0.02 m` vs `0.035 m`) so the composite stays in the recoverable
envelope.

**Excluded on purpose:** gaps, stepping stones, narrow beams, holes. A blind
proprioceptive student (the distillation target) could never recover those — they need
look-before-you-step foothold precision — so the teacher must not learn to rely on them.
**Every feature here is continuous and crossable** — a wrong step is feelable and
recoverable, never an instant fall; the mixed tiles only *sum* small height fields, so
they stay within that same envelope. This is **blind-feasible** terrain by construction.

**Curriculum driver** (`HexabotRoughEnv._update_terrain_levels`, called from
`_reset_idx`). The direct workflow has no curriculum manager, so the env does it itself.
It previously mirrored `terrain_levels_vel` (promote past half a tile; demote on failing
to cover half the *commanded* distance), but the harder mixed terrain shortens
distance-per-episode, so the old `1.0 m` promote bar left the top levels **unreachable**.
The band was loosened to a **command-independent fraction of the tile size**
(`hexabot_rough_env_cfg.py`):
- Walked **> `terrain_promote_frac` of a tile** (`0.35 · 2.0 = 0.70 m`, was `1.0 m`) → **level up**.
- Walked **< `terrain_demote_frac` of a tile** (`0.15 · 2.0 = 0.30 m`) → **level down**
  (was: covered less than half the commanded distance).
- Everyone starts at `max_init_terrain_level=0` (near-flat).
- **`Curriculum/terrain_level` (mean) is the lead progress metric.** Watch it first in
  TensorBoard. With the loosened band the mean level now climbs to max — rendered clips
  at terrain levels 6 and 8 (`hexabot_model/hexabot_rough_level6.mp4`,
  `hexabot_rough_level8.mp4`) confirm traversal of the hard mixed tiles.

---

## 9. PPO and symmetry

- rsl_rl PPO, `[128,128,128]` ELU actor/critic, `init_noise_std=0.8`,
  `entropy_coef=0.01`, `lr=1e-3` adaptive (`desired_kl=0.01`), `γ=0.99`, `λ=0.95`,
  `num_steps_per_env=24`, 5 epochs × 4 minibatches.
- **Left-right symmetry data augmentation** (`symmetry.py`,
  `RslRlSymmetryCfg(use_data_augmentation=True)`). The hexapod is exactly mirror-
  symmetric and straight walking is a symmetric task, so every transition's left-right
  mirror is valid on-policy data. PPO appends the mirrored copy to each minibatch:
  forces a mirror-symmetric policy (kills left/right drift → straight walking) and
  doubles effective data. **On flat this cut lateral drift ~9× (dy 1.111 → 0.117 m).**
  - The reflection maps `y → −y`: gravity `(gx,−gy,gz)`, ang-vel `(−wx,wy,−wz)`, command
    `(vx,−vy,−yaw)`; joints swap L↔R with a **sign flip on coxa-yaw only** (femur/tibia
    pitch joints swap without a flip); CPG action/phase swap by leg with **no** sign
    flip (CPG params are sign-invariant amplitudes); and on rough terrain the **63
    height-scan rays are permuted to their left-right partners** (`_scan_mirror_idx`,
    built from the actual ray pattern), heights unchanged.
- `max_iterations = 3000` for rough (more than flat's 1000 — the curriculum needs
  headroom).

---

## 10. Run / export / render

```bash
cd external/IsaacLab          # conda env: env_isaaclab

# full pipeline (train -> export best -> render a clip):
bash ../../scripts/milestone1_rough.sh
#   knobs:  NUM_ENVS=4096 MAX_ITER=3000 bash ../../scripts/milestone1_rough.sh
#   plumbing check: SMOKE=1 bash ../../scripts/milestone1_rough.sh

# step by step:
./isaaclab.sh -p ../../isaac_lab/train_rough.py --num_envs 4096 --max_iterations 3000 --headless
./isaaclab.sh -p ../../isaac_lab/play_rough.py --select_best     # -> exported/policy.pt(+onnx)

# render the LATEST checkpoint over rough terrain, any time (even mid-training):
bash ../../scripts/render_latest_rough.sh
#   SECONDS_CLIP=15 CMD_VX=0.25 TERRAIN_LEVEL=8 bash ../../scripts/render_latest_rough.sh
```

Logs → `logs/rsl_rl/hexabot_rough_direct/`. **Watch order:** `Curriculum/terrain_level`
(lead signal) → `Episode_Reward/forward_progress` & `track_lin_vel_xy_exp` → episode
length → `Episode_Termination/died` → `Metrics/stand_when_cmd_frac` (should → 0; a value
near 1.0 means it collapsed to standing).

**Landmines:** the eplen plateau (≈16 for ~300 iters, walking breaks out ~iter 550 with
symmetry) **also holds on rough** — judge only on a full run. Isaac hangs at
`simulation_app.close()` holding GPU memory; `pkill -9` after every run.

---

## 11. Current limitations → where to improve

These are the seams worth thinking about, roughly in order of leverage.

1. **Blind-feasible terrain ceiling.** Gaps, stepping stones, beams, and holes are
   excluded *by design* so the teacher distills into a blind student. If you don't need
   the blind student (or accept a perception-dependent deployment), re-introducing those
   terrains + keeping the exteroceptive scan at deploy unlocks much harder ground. This
   is a strategic fork, not a tuning knob.

2. **Static height-scan footprint.** 9×7 rays over a fixed `0.4×0.3 m` body-frame patch,
   yaw-aligned. It does not look ahead along the velocity vector. A forward-biased or
   velocity-anchored scan (look where you're going, not just under you) would give more
   reaction time for steps and slope changes. Resolution (`0.05 m`) is also coarse
   relative to the 0.22 m stair tread (and to the rolling overlay on the mixed tiles).

3. **`ground_height` is a single median.** Every terrain-relative reward/death uses one
   scalar ground level under the body. On a slope or at a step edge the body spans two
   heights; a per-foot or front/back ground estimate would make `foot_clearance`,
   `foot_support`, and `too_low` far more accurate near discontinuities.

4. **Per-leg swing lift is the only terrain adaptation channel.** The CPG gives the
   policy `Δlift` per leg, but no explicit *foothold targeting* — it can lift higher, not
   place a foot at a chosen `(x,y)`. Adding a small Cartesian foot-target residual (still
   CPG-seeded) would let it actively avoid bad footholds rather than just clearing them.

5. **Gait reward terms assume a 6-foot tetrapod template.** `tetrapod_contact` (bell at
   exactly 4 down), `gait_symmetry`, and `gait_phase` encode a fixed gait shape. On rough
   ground the optimal contact schedule is terrain-dependent; these may over-constrain.
   Worth an ablation: relax the tetrapod bell width or make it contact-count-tolerant.

6. **Reward softening is global, not terrain-conditioned.** `heading` and `lateral_pos`
   penalties are softened uniformly. Slopes need more heading freedom than flat patches.
   Conditioning these on the local scan gradient (steep → softer) could keep straightness
   on flat while freeing the body on slopes.

7. **DR is fixed-range, not curriculum-coupled.** Latency, friction, mass, and noise
   ranges are constant across terrain levels. Coupling DR magnitude to the terrain
   curriculum (harder terrain ↔ wider DR) would harden the policy progressively instead
   of fighting full DR from level 0.

8. **No proprioceptive-history input yet.** The obs is single-step (no stacked frames /
   recurrence). The distillation plan *requires* a history encoder for the student; the
   teacher could also benefit from short history to infer terrain it already stepped on.

9. **Speed is capped at 0.30 m/s** (`cpg_v_ref`) and the command curriculum tops out
   there. If the hardware has headroom, raising `VX_RANGE` + `forward_progress` weight
   (per the flat notes) would push speed, at some cost to the straightness the symmetry
   aug buys.

10. **Validation cost.** Every judgement requires a full ~550+ iter run past the eplen
    plateau. A cheaper early signal (e.g. a held-out fixed-terrain eval episode logged
    every N iters) would speed the iterate-and-improve loop substantially.
