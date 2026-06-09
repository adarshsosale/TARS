# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Goal-conditioned navigation observation / reward DEFINITION (the nav-RL seam).

This is the single place that defines the Navigation layer's observation and reward
SHAPE, shared by the trainable env (`isaac_lab/tasks/nav/nav_env.py`), the hand-coded
controller path, and the full-stack demo so none of them can drift apart.

Stage 1 (flat, obstacle-free, waypoint following) is ACTIVE here:

* Observation = current waypoint relative to the robot (goal channels) + the nav
  layer's previously issued command (the measurable "current velocity" proxy) + the
  body yaw rate (gyro) + a DORMANT, zeroed exteroceptive slot sized for lidar later.
  NB the base LINEAR velocity is deliberately absent — the locomotion layer omits it
  as unmeasurable on the real robot, and the nav layer mirrors that discipline (goal
  position is assumed known from odometry/SLAM; raw body lin-vel is not fed in).
* Reward = POTENTIAL-BASED progress shaping (ACTIVE) + a sparse reach bonus + a small
  command-rate regularizer, plus collision and path-cost terms that are present but
  INERT (no obstacles on flat stage 1).
* Output = `(vx, vy, yaw)` through the frozen interface, clamped to its envelope.

The observation/reward shapes here are fixed across curriculum stages — turning up
`obstacle_density` later fills the dormant lidar slot and activates the inert terms
without changing the policy's I/O.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NavGoalCfg:
    """Config for the goal-conditioned navigation layer."""

    # --- observation ------------------------------------------------------------
    n_lidar: int = 0                 # DORMANT exteroceptive slot width (stage 1: empty).
    #   Active goal channels: [goal_dx_body, goal_dy_body, dist, sin_bearing, cos_bearing] = 5
    goal_channels: int = 5
    prev_cmd_channels: int = 3       # the nav layer's own last (vx, vy, yaw) — fully known
    yaw_rate_channels: int = 1       # body yaw rate (gyro-measurable)

    # --- reward (potential-based shaping, NOT raw distance) ---------------------
    # phi(s) = -||p_robot - p_current_waypoint||;  shaping = gamma*phi(s') - phi(s).
    # gamma MUST equal the RL discount so the shaping is policy-invariant (it only
    # reshapes the value landscape, never the optimal policy — prevents loitering /
    # progress-farming). With gamma<1 a stationary robot earns a vanishing (1-gamma)*d
    # residual; it is dominated by the reach bonus and is the standard, harmless artifact.
    shaping_reward_scale: float = 1.0      # weight on the potential shaping term
    gamma: float = 0.99                    # = PPO discount (keep in sync with the agent cfg)
    reach_bonus: float = 10.0              # one-off bonus for reaching the current waypoint
    control_reg_scale: float = -0.02       # penalize command CHANGES (smooth velocity commands)
    heading_reg_scale: float = 0.0         # optional heading-alignment penalty (off by default)
    collision_reward_scale: float = 0.0    # INERT until obstacles exist
    path_cost_reward_scale: float = 0.0    # INERT until obstacles exist (proximity / detour cost)

    # --- episode / arena --------------------------------------------------------
    reach_tol: float = 0.15                # waypoint counted reached within this distance [m]
    goal_radius_range: tuple[float, float] = (1.0, 2.5)  # how far each next waypoint is sampled [m]
    n_waypoints: int = 3                   # waypoints per episode sequence
    nav_decimation: int = 10               # locomotion control steps per navigation tick

    # --- curriculum extensibility (stage 1 = flat, obstacle-free) ---------------
    # INERT now; later stages turn this up to fill the dormant lidar slot and activate
    # the collision/path-cost terms WITHOUT reshaping obs/action/interface.
    obstacle_density: float = 0.0

    @property
    def obs_dim(self) -> int:
        return self.goal_channels + self.prev_cmd_channels + self.yaw_rate_channels + self.n_lidar


def compute_goal_obs(
    robot_pos_xy: torch.Tensor,   # (N, 2) world
    robot_yaw: torch.Tensor,      # (N,)   world heading [rad]
    goal_pos_xy: torch.Tensor,    # (N, 2) world
) -> torch.Tensor:
    """The 5 goal channels: goal offset in the robot frame + distance + bearing sin/cos."""
    delta = goal_pos_xy - robot_pos_xy                       # world-frame offset
    cos_y, sin_y = torch.cos(robot_yaw), torch.sin(robot_yaw)
    # rotate world offset into the robot frame (+x forward, +y left)
    dx_b = cos_y * delta[:, 0] + sin_y * delta[:, 1]
    dy_b = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    dist = torch.linalg.norm(delta, dim=1)
    bearing = torch.atan2(dy_b, dx_b)
    return torch.stack([dx_b, dy_b, dist, torch.sin(bearing), torch.cos(bearing)], dim=1)


def compute_nav_obs(
    robot_pos_xy: torch.Tensor,   # (N, 2) world
    robot_yaw: torch.Tensor,      # (N,)   world heading [rad]
    goal_pos_xy: torch.Tensor,    # (N, 2) world (the CURRENT waypoint)
    prev_cmd: torch.Tensor,       # (N, 3) the nav layer's last (vx, vy, yaw)
    yaw_rate: torch.Tensor,       # (N,)   body yaw rate [rad/s]
    cfg: NavGoalCfg,
    lidar: torch.Tensor | None = None,   # (N, n_lidar) or None -> zeros (dormant)
) -> torch.Tensor:
    """Assemble the full navigation observation.

    Layout (obs_dim):
      [goal 0:5][prev_cmd 5:8][yaw_rate 8:9][lidar 9:9+n_lidar (DORMANT, last)]
    The dormant lidar slot is appended LAST so a learned lidar encoder plugs in at a
    later stage WITHOUT shifting any active channel.
    """
    goal = compute_goal_obs(robot_pos_xy, robot_yaw, goal_pos_xy)   # (N, 5)
    parts = [goal, prev_cmd, yaw_rate.unsqueeze(-1)]
    if cfg.n_lidar > 0:
        slot = lidar if lidar is not None else torch.zeros(goal.shape[0], cfg.n_lidar, device=goal.device)
        parts.append(slot)
    return torch.cat(parts, dim=-1)


def potential_shaping(
    prev_dist: torch.Tensor, dist: torch.Tensor, cfg: NavGoalCfg
) -> torch.Tensor:
    """Potential-based shaping term gamma*phi(s') - phi(s) with phi = -distance.

    Equivalent to `gamma*(-dist) - (-prev_dist) = prev_dist - gamma*dist`. At gamma=1
    this telescopes to pure progress (prev_dist - dist); at gamma<1 it adds the small,
    policy-invariant (1-gamma)*dist residual. The caller maintains `prev_dist` w.r.t.
    the CURRENT waypoint (resetting it when the waypoint index advances) so the term
    stays continuous across a waypoint hand-off and only the reach bonus pays the jump.
    """
    return (prev_dist - cfg.gamma * dist) * cfg.shaping_reward_scale


def progress_reward(prev_dist: torch.Tensor, dist: torch.Tensor, cfg: NavGoalCfg) -> torch.Tensor:
    """Legacy gamma=1 progress reward (kept for the hand-coded path). Prefer
    `potential_shaping`, which is the discount-correct potential-based form."""
    return (prev_dist - dist) * cfg.shaping_reward_scale
