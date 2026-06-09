# Hexabot Waypoint Navigation — Layer 2 (flat ground)

The navigation layer made **trainable**, sitting on top of the **frozen** locomotion
policy and emitting commands through the **frozen `(vx, vy, yaw)` interface**. This is a
deliberately cheap, end-to-end loop-close:

```
 waypoint seq ─▶ Nav policy (PPO) ──(vx,vy,yaw)──▶ frozen Loco policy + CPG ─▶ joints ─▶ motion ─▶ waypoint reached
                 5 Hz                FROZEN ENVELOPE     50 Hz                      200 Hz
```

On flat, obstacle-free ground the optimum is near-trivial (point at the waypoint, drive
forward), so the reward is intentionally minimal. The locomotion policy, CPG, robot,
physics and domain randomization are reused **unmodified** from
`Isaac-Velocity-Flat-Hexabot-Direct-v0`.

- **Task id:** `Isaac-Nav-Waypoint-Flat-Hexabot-Direct-v0` (direct workflow, rsl_rl PPO).
- **Action:** `(vx, vy, yaw)` velocity command (3-d), **clamped to the frozen envelope**.
- **Observation (9-d):** current waypoint relative to the robot `[dx_b, dy_b, dist,
  sin θ, cos θ]` + the nav layer's previously issued command (3) + body yaw rate (1) +
  a **dormant zeroed lidar slot** (last). No base linear velocity — mirrors the loco
  layer's measurability discipline (goal pose assumed known from odometry/SLAM).
- **Reward:** potential-based progress shaping + sparse reach bonus + command-rate reg
  (see below). Collision / path-cost terms present but **inert** (×0).

## 1. Velocity-envelope clipping (layers can't desync)
The nav policy's raw 3-d output is clamped to the canonical command ranges read from the
**frozen interface** (`isaac_lab/interfaces/velocity_command.py`): `VX_RANGE=(0,0.30)`,
`VY_RANGE=(0,0)` (vy dormant → strafing off), `YAW_RANGE=(-0.5,0.5)` (turning on). So the
command the nav policy can ever emit is **exactly the distribution the loco policy was
trained to track** — the two layers cannot drift apart. Widening the envelope later
(e.g. enabling `vy`) turns on new command channels with **no change to either policy**.

## 2. Nav / locomotion frequency split (explicit hierarchical timing)
A nav command is **held across `NavGoalCfg.nav_decimation = 10` locomotion control
steps**. The env `decimation = nav_decimation × loco.decimation = 10 × 4 = 40` physics
steps per nav action, so at 200 Hz physics:

| layer | rate | what it does |
|---|---|---|
| physics | 200 Hz | sim integration |
| locomotion (frozen, in-loop) | 50 Hz | infer loco policy → CPG → joint targets (zero-order hold between control ticks) |
| navigation (trained) | 5 Hz | emit one held `(vx,vy,yaw)` command |

Training **with the frozen loco policy in the loop** (full sim, not an idealized
velocity-tracking model) is the default so the nav policy learns against real tracking
lag — which matters once terrain is rough.

## 3. Potential-based reward (preserves the optimal policy)
`φ(s) = -‖p_robot − p_current_waypoint‖`; shaping term `γ·φ(s') − φ(s)` with **`γ = 0.99`
equal to the PPO discount** (`NavGoalCfg.gamma` must match the agent `gamma`), so the
shaping only reshapes the value landscape and never changes the optimal policy — no
loitering / progress-farming. Plus a **sparse reach bonus** on reaching the current
waypoint within `reach_tol`, then the index advances (the potential baseline is reset to
the new waypoint so only the bonus pays the hand-off jump). A small command-rate
regularizer keeps the velocity commands smooth. Episodes terminate on completing the
sequence, a fall, or timeout.

## 4. Obstacle-stage seams (this isn't throwaway)
Turning these up later activates obstacle avoidance **without reshaping obs/action/the
interface**:

| Future capability | Seam (already wired, inert) |
|---|---|
| Lidar / obstacle features | `NavGoalCfg.n_lidar` slot, appended **last** in `compute_nav_obs` |
| Collision penalty | `collision_reward_scale` (×0 now) |
| Path / proximity cost | `path_cost_reward_scale` (×0 now) |
| Obstacle density curriculum | `NavGoalCfg.obstacle_density` (0 now) |

## Train / export / roll out
```bash
cd external/IsaacLab        # conda env: env_isaaclab
# smoke (shape/plumbing check; auto-uses the newest exported loco policy)
./isaaclab.sh -p ../../isaac_lab/train_nav.py --num_envs 64 --max_iterations 5 --headless
# full run
./isaaclab.sh -p ../../isaac_lab/train_nav.py \
    --task Isaac-Nav-Waypoint-Flat-Hexabot-Direct-v0 \
    --num_envs 1024 --max_iterations 500 --headless
# (optional) pin a specific frozen loco policy:
#   --loco_policy <abs/.../hexabot_flat_direct/<run>/exported/policy.pt>
# export the trained nav policy
./isaaclab.sh -p ../../isaac_lab/play_nav.py
# full-stack rollout video (nav policy -> frozen interface -> loco policy -> motion)
./isaaclab.sh -p ../../isaac_lab/run_nav_demo.py \
    --policy <.../hexabot_flat_direct/<run>/exported/policy.pt> \
    --nav_policy <.../hexabot_nav_flat/<run>/exported/policy.pt> \
    --waypoints '1.5,0;1.5,1.5;0,1.5'
```
Logs → `logs/rsl_rl/hexabot_nav_flat/`.

## Self-test (no GPU)
```bash
python isaac_lab/tasks/nav/test_nav.py   # obs layout, goal-frame, potential shaping, advance/clip
```
