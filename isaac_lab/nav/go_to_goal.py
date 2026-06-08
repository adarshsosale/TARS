# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Hand-coded go-to-goal controller — the Navigation-layer placeholder.

Consumes the goal point relative to the robot (+ a dormant lidar slot) and emits a
`VelocityCommand` through the FROZEN interface. On flat, obstacle-free ground this
is the trivial-on-purpose mapping the spec describes; it is a stand-in for a future
goal-conditioned RL policy, but it already speaks the exact same interface so the
locomotion layer cannot tell the difference.

Design notes
------------
* Output goes through `VelocityCommand.clamp()`, so on stage 0 (`VY_RANGE` and
  `YAW_RANGE` pinned to 0) the command is forward-only and straight — matching what
  the locomotion policy was trained on.
* It nonetheless COMPUTES a heading-correcting yaw (and could command lateral
  motion); those channels are dormant only because the interface ranges clamp them
  to 0. Widening the ranges later turns steering on with NO change here — the
  dormant-but-real pattern the milestone asks for.
* `lidar` is accepted and ignored: the obstacle-avoidance seam. A later version
  reads it to bias the command around obstacles.
"""

from __future__ import annotations

import math

from isaac_lab.interfaces import VX_RANGE, VelocityCommand


def go_to_goal(
    goal_rel_robot: tuple[float, float],
    lidar=None,                      # dormant exteroceptive input (obstacle avoidance seam)
    *,
    max_vx: float = VX_RANGE[1],
    slow_radius: float = 0.5,        # start decelerating within this range [m]
    reach_tol: float = 0.05,         # within this distance the goal is "reached" [m]
    yaw_gain: float = 1.0,           # heading P-gain (dormant until YAW_RANGE widens)
) -> VelocityCommand:
    """Map a robot-frame goal offset (dx forward, dy left) to a VelocityCommand."""
    dx, dy = goal_rel_robot                      # dx forward, dy left (robot frame)
    dist = math.hypot(dx, dy)
    if dist < reach_tol:
        return VelocityCommand(vx=0.0, vy=0.0, yaw=0.0).clamp()
    # Forward speed scales with the FORWARD component of the goal offset (not the
    # distance magnitude): full until inside slow_radius, tapering to 0 as the goal
    # comes alongside, and clamped to >=0 so once the goal is BEHIND the robot it
    # STOPS rather than accelerating away. This matters on the yaw-locked stage-0
    # (it cannot turn around); when YAW_RANGE widens, the yaw term below steers it
    # back toward an off-axis or passed goal instead.
    vx = max_vx * max(0.0, min(1.0, dx / slow_radius))
    # heading correction (dormant on flat straight-line: YAW_RANGE clamps to 0)
    yaw = yaw_gain * math.atan2(dy, dx)
    return VelocityCommand(vx=vx, vy=0.0, yaw=yaw).clamp()
