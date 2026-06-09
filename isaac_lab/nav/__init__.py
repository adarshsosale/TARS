# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Navigation layer (goal-conditioned) for the Hexabot stack.

Milestone-0 uses a HAND-CODED go-to-goal controller as a placeholder for a future
goal-conditioned RL policy, but the plumbing is real: the goal-conditioned
observation (with a dormant zeroed lidar slot) and the frozen `(vx,vy,yaw)`
interface are the same ones a learned policy would use. See:

* `go_to_goal.py`  — the deterministic controller (goal -> VelocityCommand).
* `nav_goal_cfg.py` — the goal-conditioned obs/reward DEFINITION (the seam where a
  nav-RL policy and a lidar encoder plug in later).
"""

from .go_to_goal import go_to_goal
from .nav_goal_cfg import (
    NavGoalCfg,
    compute_goal_obs,
    compute_nav_obs,
    potential_shaping,
    progress_reward,
)

__all__ = [
    "go_to_goal",
    "NavGoalCfg",
    "compute_goal_obs",
    "compute_nav_obs",
    "potential_shaping",
    "progress_reward",
]
