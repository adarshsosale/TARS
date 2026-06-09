# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Direct RL env config for the Hexabot WAYPOINT NAVIGATION task (flat ground).

This is Layer 2 of the two-layer stack made trainable. The navigation policy emits a
`(vx, vy, yaw)` command through the FROZEN interface; the (frozen) locomotion policy
tracks it inside the same sim. The two layers run at different rates — a nav command
is held across `NavGoalCfg.nav_decimation` locomotion control steps (hard constraint
#3). On flat, obstacle-free ground the optimal behaviour is near-trivial (point at the
waypoint, drive forward), so the reward is deliberately simple: potential-based
progress shaping + a sparse reach bonus + a small command-rate regularizer.

Nothing about the locomotion layer is touched: `loco` below is an unmodified
`HexabotFlatEnvCfg`, and the robot / CPG / physics / domain-randomization come
straight from it so the nav env trains the loco policy under exactly the conditions
it was trained in. The dormant obstacle/lidar slot and inert collision/path-cost terms
from Milestone 0 stay present and zeroed — the obstacle stage turns them up WITHOUT
reshaping observations, actions, or the interface.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Reuse the locomotion task's asset, CPG params, physics and domain randomization
# verbatim (the nav env imports the loco layer's config; it does not re-define it).
from hexabot.hexabot_env_cfg import HexabotFlatEnvCfg

# The nav layer PRODUCES the frozen interface; import the canonical envelope so nav
# outputs are clamped to exactly what the loco policy was trained to track.
from isaac_lab.interfaces import VX_RANGE, VY_RANGE, YAW_RANGE
from isaac_lab.nav import NavGoalCfg


@configclass
class HexabotNavEnvCfg(DirectRLEnvCfg):
    # --- the navigation observation/reward/waypoint definition (the nav seam) ---
    nav: NavGoalCfg = NavGoalCfg()

    # --- the FROZEN locomotion layer, unmodified ---
    # Source of the robot articulation, contact sensor, terrain, physics sim and
    # domain randomization. Its CPG params drive the in-loop locomotion controller.
    loco: HexabotFlatEnvCfg = HexabotFlatEnvCfg()

    # path to the exported (TorchScript) locomotion policy.pt that runs in the loop.
    # None -> the env auto-resolves the newest logs/.../hexabot_flat_direct/*/exported/policy.pt
    loco_policy_path: str | None = None
    # add the loco layer's IMU/obs noise when building the in-loop loco observation.
    # Off by default: the loco policy is robust and a clean signal keeps nav training
    # stable; the realistic tracking lag still comes from the held-command hierarchy +
    # physics DR (friction/mass/gains, on via loco.events).
    loco_obs_noise: bool = False

    # --- frozen command envelope (clamp nav outputs to the trackable distribution) ---
    cmd_vx_range = VX_RANGE        # (0.0, 0.30) m/s
    cmd_vy_range = VY_RANGE        # (0.0, 0.0)  m/s  (dormant)
    cmd_yaw_range = YAW_RANGE      # (-0.5, 0.5) rad/s (turn-to-face)

    # --- arena (bounded so cloned envs never overlap; keeps waypoints reachable) ---
    # Each waypoint is sampled inside a +/- arena_radius box around the env origin.
    # env_spacing must exceed 2*arena_radius so neighbouring envs can't collide.
    arena_radius: float = 2.0
    min_leg: float = 0.8           # minimum spacing between consecutive waypoints [m]

    # --- env / timing -------------------------------------------------------------
    # decimation is PHYSICS steps per NAV action = nav_decimation (loco control steps
    # per nav tick) * loco.decimation (physics steps per loco control step). Set in
    # __post_init__ from the two layers so the hierarchical timing is explicit.
    episode_length_s = 25.0
    action_space = 3               # (vx, vy, yaw) through the frozen interface
    observation_space = NavGoalCfg().obs_dim   # 9 (goal 5 + prev_cmd 3 + yaw_rate 1 + lidar 0)
    state_space = 0
    decimation = 40                # placeholder; recomputed in __post_init__

    # scene: fewer envs than loco (each env runs the loco policy in-loop every control
    # step), wider spacing for the roaming waypoint arena.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=5.0, replicate_physics=True)

    def __post_init__(self):
        # Inherit the physics sim, terrain, robot, contact sensor and physics-side
        # domain randomization from the frozen locomotion layer.
        self.sim = self.loco.sim
        self.terrain = self.loco.terrain
        self.robot = self.loco.robot
        self.contact_sensor = self.loco.contact_sensor
        self.events = self.loco.events
        # explicit hierarchical timing: 50 Hz loco control, 5 Hz nav at nav_decimation=10
        self.decimation = self.nav.nav_decimation * self.loco.decimation
        # keep render cadence at the loco control rate
        self.sim.render_interval = self.loco.decimation
        # arena must fit inside the per-env cell with margin
        assert self.scene.env_spacing > 2.0 * self.arena_radius, (
            f"env_spacing ({self.scene.env_spacing}) must exceed 2*arena_radius "
            f"({2.0 * self.arena_radius}) so cloned envs cannot collide"
        )
