# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Standalone correctness checks for the navigation obs / reward / waypoint logic
(no Isaac needed).

Run in the conda env (torch only):  python isaac_lab/tasks/nav/test_nav.py

Verifies:
  1. compute_nav_obs layout & dim (goal 5 + prev_cmd 3 + yaw_rate 1 + dormant lidar).
  2. compute_goal_obs rotates the goal offset into the robot frame correctly.
  3. potential_shaping at gamma=1 telescopes to total distance covered (pure progress)
     and is zero for a stationary robot.
  4. The waypoint-advance + envelope-clip logic the env relies on.
"""

import math
import os
import sys

import torch

# repo root on the path so `isaac_lab.nav` resolves
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _ROOT)

from isaac_lab.interfaces import VX_RANGE, VY_RANGE, YAW_RANGE  # noqa: E402
from isaac_lab.nav import NavGoalCfg, compute_goal_obs, compute_nav_obs, potential_shaping  # noqa: E402


def test_obs_layout():
    cfg = NavGoalCfg()
    assert cfg.obs_dim == 9, cfg.obs_dim
    N = 4
    pos = torch.randn(N, 2)
    yaw = torch.randn(N)
    goal = torch.randn(N, 2)
    prev_cmd = torch.randn(N, 3)
    yaw_rate = torch.randn(N)
    obs = compute_nav_obs(pos, yaw, goal, prev_cmd, yaw_rate, cfg)
    assert obs.shape == (N, 9), obs.shape
    # channels 5:8 are the prev command verbatim; 8 is the yaw rate
    assert torch.allclose(obs[:, 5:8], prev_cmd)
    assert torch.allclose(obs[:, 8], yaw_rate)
    print("ok: obs layout (dim=9, prev_cmd & yaw_rate channels)")


def test_goal_frame():
    # robot at origin facing +x (yaw=0); goal straight ahead -> dx_b>0, dy_b=0
    pos = torch.zeros(1, 2)
    goal = torch.tensor([[2.0, 0.0]])
    g = compute_goal_obs(pos, torch.zeros(1), goal)
    assert abs(g[0, 0].item() - 2.0) < 1e-5 and abs(g[0, 1].item()) < 1e-5
    # rotate robot to face +y (yaw=pi/2); a goal at world +x is now to the robot's RIGHT
    # (dy_b<0), dead abeam (dx_b≈0)
    g2 = compute_goal_obs(pos, torch.tensor([math.pi / 2]), goal)
    assert abs(g2[0, 0].item()) < 1e-5, g2
    assert g2[0, 1].item() < -1.9, g2
    assert abs(g2[0, 2].item() - 2.0) < 1e-5  # distance is frame-invariant
    print("ok: goal offset rotates into the robot frame")


def test_potential_shaping():
    cfg = NavGoalCfg(gamma=1.0, shaping_reward_scale=1.0)
    # a straight 3-step approach: distances 3 -> 2 -> 1 -> 0. Sum of shaping == 3 (= total
    # progress), independent of how it's split (telescoping potential).
    dists = [3.0, 2.0, 1.0, 0.0]
    total = 0.0
    prev = torch.tensor([dists[0]])
    for d in dists[1:]:
        total += potential_shaping(prev, torch.tensor([d]), cfg).item()
        prev = torch.tensor([d])
    assert abs(total - 3.0) < 1e-5, total
    # stationary robot earns exactly 0 at gamma=1
    s = potential_shaping(torch.tensor([2.0]), torch.tensor([2.0]), cfg).item()
    assert abs(s) < 1e-6, s
    # gamma<1: stationary earns the small (1-gamma)*d residual, and it is small
    cfg2 = NavGoalCfg(gamma=0.99, shaping_reward_scale=1.0)
    s2 = potential_shaping(torch.tensor([2.0]), torch.tensor([2.0]), cfg2).item()
    assert abs(s2 - 0.01 * 2.0) < 1e-6, s2
    print("ok: potential shaping telescopes to total progress; stationary≈0")


def test_advance_and_clip():
    cfg = NavGoalCfg(n_waypoints=3, reach_tol=0.15)
    N = 3
    wps = torch.tensor([
        [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],   # env 0
        [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],   # env 1
        [[1.0, 0.0], [1.0, 1.0], [2.0, 1.0]],   # env 2
    ])
    idx = torch.tensor([0, 1, 2])
    pos = torch.tensor([[1.0, 0.0], [5.0, 5.0], [2.0, 1.0]])   # env0 at wp0, env1 far, env2 at last wp
    cur = wps[torch.arange(N), idx]
    dist = torch.linalg.norm(pos - cur, dim=1)
    reached = dist < cfg.reach_tol
    is_last = idx >= (cfg.n_waypoints - 1)
    advance = reached & ~is_last
    new_idx = torch.where(advance, idx + 1, idx)
    assert new_idx.tolist() == [1, 1, 2], new_idx.tolist()   # env0 advances, env1 no, env2 stays (last)
    success = (idx >= cfg.n_waypoints - 1) & reached
    assert success.tolist() == [False, False, True]

    # envelope clip: vy pinned to 0, yaw within range, vx clamped to [0, .30]
    lo = torch.tensor([VX_RANGE[0], VY_RANGE[0], YAW_RANGE[0]])
    hi = torch.tensor([VX_RANGE[1], VY_RANGE[1], YAW_RANGE[1]])
    raw = torch.tensor([[0.9, 0.5, 2.0], [-0.5, -0.3, -1.0]])
    clipped = torch.clamp(raw, lo, hi)
    assert torch.allclose(clipped, torch.tensor([[0.30, 0.0, 0.5], [0.0, 0.0, -0.5]]))
    print("ok: waypoint advance/success + envelope clip")


if __name__ == "__main__":
    test_obs_layout()
    test_goal_frame()
    test_potential_shaping()
    test_advance_and_clip()
    print("\nAll nav self-tests passed.")
