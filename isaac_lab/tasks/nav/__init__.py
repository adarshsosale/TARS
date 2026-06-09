# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Hexabot waypoint-navigation task (Layer 2, frozen locomotion in the loop)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Nav-Waypoint-Flat-Hexabot-Direct-v0",
    entry_point=f"{__name__}.nav_env:HexabotNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:HexabotNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexabotNavPPORunnerCfg",
    },
)
