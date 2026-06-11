# Hexabot Two-Layer Control Stack — Milestone 0

A goal → velocity → gait stack for the 18-DOF hexapod, flat-ground now but built so
obstacles and rough terrain drop in later **without re-architecting**.

```
            ┌─────────────────────────┐   (vx, vy, yaw)   ┌──────────────────────────┐
   goal ──▶ │  Navigation (Layer 2)   │ ════════════════▶ │  Locomotion (Layer 1)    │ ──▶ joints
            │  goal-conditioned        │  FROZEN INTERFACE │  PPO + CPG, continuous   │
            └─────────────────────────┘                   └──────────────────────────┘
```

## The frozen interface (the only contract)
`isaac_lab/interfaces/velocity_command.py` — `VelocityCommand(vx, vy, yaw)` + the
canonical `VX/VY/YAW_RANGE`. **Both layers import this; neither reaches across it.**
Stage 0 is straight-line: `VY_RANGE`/`YAW_RANGE` are pinned to 0. Widening them later
turns on lateral/turning **with no change to either policy**.

## Layer 1 — Locomotion (PPO, continuous control)
- **Task:** `Isaac-Velocity-Flat-Hexabot-Direct-v0` (direct workflow, rsl_rl PPO).
- **Observation — proprioceptive ONLY (81-d), no base linear velocity** (not
  measurable on the real robot): projected gravity, base angular velocity, the
  velocity command, joint pos/vel, previous action, CPG phase. A **dormant
  height-scan slot** (`n_height_scan=0`) is appended last — the exteroceptive seam.
- **Action — CPG modulation, NOT joint offsets** (`cpg.py`): per-leg
  `[d_freq, d_coxa_amp, d_lift, d_stance]` (24 total). **Zero action == the
  analytical tripod gait** (`hexabot_model/isaac/tripod_gait.py`) scaled to the
  commanded speed, so the analytical gait is a *reference*, never a residual basis
  (hard constraint #2). `d_stance` (M1.5 Phase E) is a one-sided femur press-down
  that raises the ride height — the rough-terrain "belly up" posture channel; ≤0
  is a no-op so it cannot reopen the belly-crawl.
- **Reward:** dense velocity tracking (primary) + stability/orientation + effort +
  gait-quality terms + an **annealing imitation reward** (deviation from the
  analytical gait, weight → 0 over the curriculum, tied to curriculum progress).
- **Reference path:** BC warm-start (`bc_warmstart.py`, `--bc_warmstart`) +
  the annealing imitation term. No residual RL.
- **Domain randomization ON from the start** (hard constraint #4): friction, mass,
  actuator gains (`EventCfg`); actuator latency, control-rate jitter, IMU/obs noise
  (`DomainRandCfg`). All ranges are config.

### Train / export / record locomotion
```bash
cd external/IsaacLab        # conda env: env_isaaclab
# smoke (shape/plumbing check)
./isaaclab.sh -p ../../isaac_lab/train_hexabot.py --num_envs 256 --max_iterations 5 --headless
# full run WITH behaviour-cloning warm-start
./isaaclab.sh -p ../../isaac_lab/train_hexabot.py \
    --task Isaac-Velocity-Flat-Hexabot-Direct-v0 \
    --num_envs 4096 --max_iterations 1000 --bc_warmstart --headless
# export best checkpoint (scored by forward body velocity)
./isaaclab.sh -p ../../isaac_lab/play_hexabot.py --select_best
# record the trained policy
./isaaclab.sh -p ../../isaac_lab/record_hexabot.py --policy <.../exported/policy.pt>
```
Logs → `logs/rsl_rl/hexabot_flat_direct/`. **NB:** the eplen plateau lasts to
~iter 300 and walking breaks out later (~iter 550 with symmetry aug) — judge runs
only past the breakout, never on short validation runs.

## Layer 2 — Navigation (hand-coded placeholder, real plumbing)
- **Controller:** `nav/go_to_goal.py` — deterministic goal → `VelocityCommand`. It
  computes heading-correcting yaw too; that channel is *dormant* only because the
  interface ranges clamp it to 0 (turn it on by widening `YAW_RANGE`).
- **Goal-conditioned scaffold:** `nav/nav_goal_cfg.py` — the goal-relative
  observation (+ **dormant zeroed lidar slot**, `n_lidar=0`) and dense
  progress-to-goal reward with **inert** collision/path-cost terms (no obstacles
  yet). This is the seam where a goal-conditioned PPO policy + lidar encoder plug in.
- **End-to-end demo** (both layers over the frozen interface):
```bash
./isaaclab.sh -p ../../isaac_lab/run_nav_demo.py --policy <.../exported/policy.pt> --goal_x 2.5
```
- **Trainable waypoint navigation (PPO, frozen loco in the loop):** the nav layer is now
  learnable — `Isaac-Nav-Waypoint-Flat-Hexabot-Direct-v0` (`isaac_lab/tasks/nav/`,
  `train_nav.py` / `play_nav.py`). Potential-based reward, nav(5 Hz)/loco(50 Hz) timing
  split, command clamped to the frozen envelope, dormant obstacle seams. See
  **`isaac_lab/tasks/nav/README.md`**. `run_nav_demo.py --nav_policy <…>` drives the
  full stack with the trained policy instead of `go_to_goal`.

## Where the dormant extensibility seams are
| Future capability | Seam (already wired, inert) |
|---|---|
| Height-scan / terrain perception (locomotion) | **ACTIVATED in Milestone 1** — `n_height_scan` obs slot now carries the 63-ray privileged scan (`hexabot_rough_env.py`) |
| Lidar / obstacle features (navigation) | `NavGoalCfg.n_lidar` slot in `compute_goal_obs` |
| Lateral / turning commands | widen `VY_RANGE`/`YAW_RANGE` in the frozen interface |
| Obstacles / rough terrain (curriculum) | `obstacle_density`, `terrain_roughness` (stage-0 = 0) |
| Imitation hand-off to terrain stages | `imitation_anneal_steps` (weight → 0 over the curriculum) |

Observation/action **shapes stay fixed** across stages — the dormant slots absorb
new inputs, so no policy re-architecture is needed.

## Milestone 1 — Rough-terrain locomotion (privileged teacher)

Extends the flat locomotion policy to rough terrain. The frozen `(vx, vy, yaw)`
interface and the navigation layer are **unchanged** — locomotion still tracks
velocity commands, now over uneven ground. This activates the dormant
height-scan seam from M0 and trains a **distillable teacher**: a future milestone
distills it into a *blind* proprioceptive student (no exteroceptive sensor).

Task id: `Isaac-Velocity-Rough-Hexabot-Direct-v0`
(`hexabot_rough_env.py` / `hexabot_rough_env_cfg.py`, PPO cfg
`HexabotRoughPPORunnerCfg`). It **subclasses** the flat env/cfg — same CPG action,
same reward family, same DR — and adds only what rough terrain needs.

### What changed vs flat
- **Terrain:** `terrain_type="generator"` with a **curriculum** over blind-feasible
  sub-terrains (`rough_terrains.py`): gentle slopes, random rough ground, low
  boxes, low stairs — scaled hard to the 72 mm hexapod. **No** gaps / stepping
  stones / beams (a blind student could never recover those, so the teacher must
  not learn to rely on them). Level 0 ≈ flat; difficulty ramps per-env with the
  terrain-level curriculum.
- **Curriculum driver:** the direct workflow has no curriculum manager, so
  `HexabotRoughEnv._reset_idx` calls `terrain.update_env_origins(...)` itself
  (walk past ½ tile → level up; cover < ½ the commanded distance → level down)
  and logs the **mean terrain level** — the lead progress metric.
- **Privileged height scanner** (`RayCasterCfg`, 13×7 = **91 rays** after the
  M1.5 Phase C forward extension — body-frame x ∈ [−0.20, +0.40] m, yaw-aligned,
  on `base_link`) fills the dormant `n_height_scan` slot → `observation_space
  81 → 172`. It is **teacher-only exteroception**.
- **Distillable teacher** (`teacher_policy.py`, `HexabotTeacherActorCritic`): the
  privileged scan enters ONLY through a **latent bottleneck**, isolated from the
  proprioceptive path:
  `z = scan_encoder(scan[91→16]);  action = actor_trunk([proprio(81) | z(16)])`.
  The critic is privileged (consumes the full 172-d obs; discarded at deploy).
  The proprio/scan split is derived from the cfg (`n_proprio = obs − N_HEIGHT_SCAN`).
- **Terrain-relative rewards:** a ground height from the scanner
  (`self._ground_height`) makes base-height / belly-clearance / foot-clearance /
  `too_low`-death terrain-relative. Heading / lateral-position penalties softened
  (rough ground legitimately yaws the body). DR extends to payload mass + base COM
  shift (`RoughEventCfg`) on top of the inherited friction / actuator / IMU DR.
- **Symmetry aug** extends to mirror the height-scan rays left↔right
  (`_scan_mirror_idx`, built from the ray pattern in `HexabotRoughEnv`).
- **Imitation reward** keeps the M0 anneal (→0 over the near-flat early
  curriculum); it is **not** re-anchored to the flat gait on rough terrain.

### Milestone 1.5 Phase E — belly management + anticipatory ride height
Level-8 diagnosis: the belly touches the ground and the episode ends (the flat
1 N base-contact death), and the policy has no way to walk taller. Three changes,
all inert on flat ground:
- **`d_stance` CPG channel** (action 18 → 24, obs 75 → 81): a one-sided per-leg
  femur press-down (≤ `cpg_kstance`=0.35 rad ≈ +30–40 mm) so the policy can
  *physically* raise its belly. Zero action is still the analytical gait.
- **Graded belly contact instead of binary death:** on rough, base contact kills
  only above `base_contact_force_death` = 50 N (~2.6× bodyweight — a component-
  damage proxy); below that it is priced linearly (`belly_contact_force`, −0.1/N
  above a 2 N free graze allowance). The `too_low` death tolerates 0.5 s of
  transient belly-down (`too_low_grace_steps`=25) — clambering nudges are a
  legitimate optimization now; lying flat still terminates.
- **Anticipatory posture targets:** the scan's q90-above-mid-ground protrusion
  (incl. the +0.40 m lookahead) raises the base-height target + belly-clearance
  floor (`_height_target_offset`, ≤ 30 mm) and the rewarded swing apex
  (`_foot_clearance_offset`, ≤ +25 mm) ~0.7 s *before* the obstacle — the −50
  belly-clearance term then pays the policy to go "belly higher than normal"
  before committing. A `foot_stumble` penalty (−1.0, horizontal-dominated foot
  forces) prices toe-catches on box walls / stair risers directly.
  Watch `Metrics/height_target_offset` in TensorBoard to see the mode engage.

### Train (headless) / render / export
```bash
cd external/IsaacLab          # conda env: env_isaaclab

# full pipeline (train -> export best -> render a clip), one command:
bash ../../scripts/milestone1_rough.sh
#   knobs:  NUM_ENVS=4096 MAX_ITER=3000 bash ../../scripts/milestone1_rough.sh
#   plumbing check: SMOKE=1 bash ../../scripts/milestone1_rough.sh

# or step by step:
./isaaclab.sh -p ../../isaac_lab/train_rough.py --num_envs 4096 --max_iterations 3000 --headless
./isaaclab.sh -p ../../isaac_lab/play_rough.py --select_best        # export best -> exported/policy.pt(+onnx)

# render the LATEST checkpoint over rough terrain, any time (even mid-training):
bash ../../scripts/render_latest_rough.sh
#   SECONDS_CLIP=15 CMD_VX=0.25 TERRAIN_LEVEL=8 bash ../../scripts/render_latest_rough.sh
# -> hexabot_model/hexabot_rough_locomotion.mp4
```
Logs → `logs/rsl_rl/hexabot_rough_direct/`. **Watch (TensorBoard):**
`Curriculum/terrain_level` **first** (mean terrain difficulty — the lead signal),
then `Episode_Reward/track_lin_vel_xy_exp` / `forward_progress`, episode length,
and `Episode_Termination/died`. The eplen-plateau lesson from flat still holds —
judge only on a full run, not a short validation.

### Distillation seam (for the next milestone)
Keep `actor_trunk` **verbatim**; replace `scan_encoder` with a proprioceptive-
**history encoder** trained to regress the same latent `z`. Because the trunk only
ever sees `[proprio | z]` and never the raw scan, the student is a clean module
swap, not a re-architecture. `play_rough.py` already exports the full teacher
(encoder + trunk) so `z` is well-defined.

## Self-tests (no GPU)
```bash
python isaac_lab/tasks/hexabot/test_cpg.py   # zero-action == analytical gait; mirrors are involutions
```
