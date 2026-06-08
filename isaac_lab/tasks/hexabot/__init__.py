# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Hexabot 18-DOF hexapod flat-ground walking task (direct workflow)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Hexabot-Direct-v0",
    entry_point=f"{__name__}.hexabot_env:HexabotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexabot_env_cfg:HexabotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexabotFlatPPORunnerCfg",
    },
)
