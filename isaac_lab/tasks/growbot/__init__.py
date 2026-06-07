# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""TARS (Growbot) sagittal-plane flat-ground walking task (direct workflow)."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Growbot-Direct-v0",
    entry_point=f"{__name__}.growbot_env:GrowbotEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.growbot_env_cfg:GrowbotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GrowbotFlatPPORunnerCfg",
    },
)
