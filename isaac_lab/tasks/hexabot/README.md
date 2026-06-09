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
- **Observation — proprioceptive ONLY (75-d), no base linear velocity** (not
  measurable on the real robot): projected gravity, base angular velocity, the
  velocity command, joint pos/vel, previous action, CPG phase. A **dormant
  height-scan slot** (`n_height_scan=0`) is appended last — the exteroceptive seam.
- **Action — CPG modulation, NOT joint offsets** (`cpg.py`): per-leg
  `[d_freq, d_coxa_amp, d_lift]`. **Zero action == the analytical tripod gait**
  (`hexabot_model/isaac/tripod_gait.py`) scaled to the commanded speed, so the
  analytical gait is a *reference*, never a residual basis (hard constraint #2).
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
| Height-scan / terrain perception (locomotion) | `n_height_scan` obs slot + the separable obs block in `_get_observations` |
| Lidar / obstacle features (navigation) | `NavGoalCfg.n_lidar` slot in `compute_goal_obs` |
| Lateral / turning commands | widen `VY_RANGE`/`YAW_RANGE` in the frozen interface |
| Obstacles / rough terrain (curriculum) | `obstacle_density`, `terrain_roughness` (stage-0 = 0) |
| Imitation hand-off to terrain stages | `imitation_anneal_steps` (weight → 0 over the curriculum) |

Observation/action **shapes stay fixed** across stages — the dormant slots absorb
new inputs, so no policy re-architecture is needed.

## Self-tests (no GPU)
```bash
python isaac_lab/tasks/hexabot/test_cpg.py   # zero-action == analytical gait; mirrors are involutions
```
