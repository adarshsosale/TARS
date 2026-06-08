# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Goal-conditioned navigation observation / reward DEFINITION (the nav-RL seam).

Milestone-0 drives navigation with a hand-coded controller (`go_to_goal.py`), so
there is no nav policy to train yet. But the goal-conditioning plumbing is REAL and
defined here once, so a future goal-conditioned PPO policy (and a lidar encoder)
drop in WITHOUT reshaping anything:

* Observation = the goal point relative to the robot (ACTIVE) + a dormant, zeroed
  exteroceptive slot sized for lidar to fill later.
* Reward = dense progress-to-goal (ACTIVE) + collision and path-cost terms that are
  present but INERT (no obstacles on flat stage 0).
* Output = `(vx, vy, yaw)` through the frozen interface (produced by whatever
  controller/policy sits behind it).

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

    # --- observation ---
    n_lidar: int = 0                 # DORMANT exteroceptive slot width (stage 0: empty).
    #   Active goal channels: [goal_dx_body, goal_dy_body, dist, sin_bearing, cos_bearing] = 5
    goal_channels: int = 5

    # --- reward weights ---
    progress_reward_scale: float = 1.0     # dense reduction in distance-to-goal (ACTIVE)
    reach_bonus: float = 10.0              # one-off bonus for reaching the goal
    collision_reward_scale: float = 0.0    # INERT until obstacles exist
    path_cost_reward_scale: float = 0.0    # INERT until obstacles exist (e.g. proximity / detour cost)

    # --- episode / arena ---
    reach_tol: float = 0.10                # goal counted reached within this distance [m]
    goal_radius_range: tuple[float, float] = (1.0, 3.0)  # how far goals are sampled [m]
    nav_decimation: int = 10               # locomotion control steps per navigation tick

    @property
    def obs_dim(self) -> int:
        return self.goal_channels + self.n_lidar


def compute_goal_obs(
    robot_pos_xy: torch.Tensor,   # (N, 2) world
    robot_yaw: torch.Tensor,      # (N,)   world heading [rad]
    goal_pos_xy: torch.Tensor,    # (N, 2) world
    cfg: NavGoalCfg,
    lidar: torch.Tensor | None = None,   # (N, n_lidar) or None -> zeros (dormant)
) -> torch.Tensor:
    """Build the goal-conditioned observation (goal-relative + dormant lidar slot)."""
    delta = goal_pos_xy - robot_pos_xy                       # world-frame offset
    cos_y, sin_y = torch.cos(robot_yaw), torch.sin(robot_yaw)
    # rotate world offset into the robot frame (+x forward, +y left)
    dx_b = cos_y * delta[:, 0] + sin_y * delta[:, 1]
    dy_b = -sin_y * delta[:, 0] + cos_y * delta[:, 1]
    dist = torch.linalg.norm(delta, dim=1)
    bearing = torch.atan2(dy_b, dx_b)
    active = torch.stack([dx_b, dy_b, dist, torch.sin(bearing), torch.cos(bearing)], dim=1)
    if cfg.n_lidar > 0:
        slot = lidar if lidar is not None else torch.zeros(active.shape[0], cfg.n_lidar, device=active.device)
        active = torch.cat([active, slot], dim=1)
    return active


def progress_reward(prev_dist: torch.Tensor, dist: torch.Tensor, cfg: NavGoalCfg) -> torch.Tensor:
    """Dense progress-to-goal: positive when the robot closes distance to the goal."""
    return (prev_dist - dist) * cfg.progress_reward_scale
