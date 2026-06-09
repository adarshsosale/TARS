# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HexabotNavPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO for the goal-conditioned waypoint navigation policy.

    The task is dense and near-trivial on flat ground (point at the waypoint, drive
    forward), so this is plain PPO with the standard entropy bonus — no distributional
    methods, curiosity, or bespoke exploration (hard constraint #5). A small MLP is
    plenty for the 9-d goal-conditioned obs -> 3-d velocity command mapping.

    NB `gamma` here MUST match `NavGoalCfg.gamma` (the potential-shaping discount) so
    the shaping stays policy-invariant.
    """

    num_steps_per_env = 24
    max_iterations = 500
    save_interval = 50
    experiment_name = "hexabot_nav_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,                 # == NavGoalCfg.gamma (potential-shaping discount)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
