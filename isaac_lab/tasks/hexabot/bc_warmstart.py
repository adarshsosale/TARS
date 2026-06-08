# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Behaviour-cloning warm-start for the Hexabot locomotion policy.

Milestone-0 hard constraint #2 wants the analytical tripod gait used as a
*reference* via a BC warm-start: pretrain the policy to imitate analytical-gait
rollouts, then hand off to PPO.

Because the CPG (cpg.py) is built so that **zero action == the analytical tripod
gait** (scaled to the commanded speed), the analytical rollout is simply the env
driven with `action = 0`, and the BC target action is `0`. Fitting the actor mean
to that target over the observation distribution visited while walking analytically
makes the freshly-initialized policy ride the nominal CPG from PPO iteration 0
(instead of exploring from random joint babble) — the gait emerges immediately and
the annealing imitation reward then refines it.

This operates on the SHARED rsl_rl runner (no checkpoint-format round-trip): call
`bc_pretrain(runner, env, env_cfg)` after building the runner and before
`runner.learn(...)`. See `train_hexabot.py --bc_warmstart`.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def _collect_analytical_obs(env, num_steps: int, device) -> torch.Tensor:
    """Roll the env with the analytical (zero) action; return stacked observations."""
    base = env.unwrapped
    # force full-speed commands so the rollout shows the GAIT, not the stand phase
    saved_step = base._step_count
    base._step_count = 10**9
    obs, _ = env.reset()
    act_dim = base.cfg.action_space
    zero = torch.zeros(base.num_envs, act_dim, device=device)
    buf = []
    for _ in range(num_steps):
        buf.append(obs.clone())
        obs, _, _, _ = env.step(zero)
    base._step_count = saved_step
    return torch.cat(buf, dim=0)


def bc_pretrain(runner, env, env_cfg, num_rollout_steps: int = 150, epochs: int = 5,
                lr: float = 1.0e-3, batch_size: int = 4096, log=print):
    """Pretrain the actor mean to output the analytical (zero) action.

    Args mirror typical BC knobs; defaults are deliberately gentle so the trunk
    is warmed toward the analytical gait without being driven to a degenerate
    constant — PPO + the annealing imitation reward take over from there.
    """
    device = runner.device
    policy = runner.alg.policy  # rsl_rl ActorCritic
    act_dim = env.unwrapped.cfg.action_space

    log(f"[BC] collecting {num_rollout_steps} analytical-gait control steps "
        f"x {env.unwrapped.num_envs} envs ...")
    X = _collect_analytical_obs(env, num_rollout_steps, device)
    Y = torch.zeros(X.shape[0], act_dim, device=device)  # analytical action == 0
    log(f"[BC] dataset: {tuple(X.shape)} obs -> {tuple(Y.shape)} actions")

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            mean = policy.act_inference(X[idx])      # deterministic mean action
            loss = torch.nn.functional.mse_loss(mean, Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        log(f"[BC] epoch {ep + 1}/{epochs}  mse={tot / max(nb, 1):.5f}")
    log("[BC] warm-start complete — handing off to PPO.")
