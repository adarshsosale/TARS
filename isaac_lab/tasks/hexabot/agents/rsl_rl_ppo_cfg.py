# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..symmetry import compute_symmetric_states_lr


@configclass
class HexabotFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # Baseline hyperparams kept so left-right symmetry augmentation (below) is the ONLY
    # change vs the validated run (runs/2026-06-08_19-58-43: dx=4.774, dy=1.111). NB the
    # eplen~16 plateau lasts to ~iter 300 and walking only breaks out ~iter 500 — that is
    # NORMAL for this reward config, not a failure, so this MUST be judged on a full run.
    num_steps_per_env = 24
    max_iterations = 1000
    save_interval = 50
    experiment_name = "hexabot_flat_direct"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,   # was 1.0: ±0.4 rad explore noise (×action_scale) instead of ±0.5
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,   # was 0.005: keep exploring; noise std collapsed to 0.06 by iter ~16
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        # --- headline change: left-right symmetry data augmentation ---
        # The hexapod is exactly mirror-symmetric and walking straight is a symmetric
        # task, so every transition's left-right mirror is valid on-policy data. Training
        # on it forces a mirror-symmetric policy (kills the left/right drift -> dy -> 0)
        # and doubles effective data for free (steadier convergence). See symmetry.py.
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            data_augmentation_func=compute_symmetric_states_lr,
        ),
    )
