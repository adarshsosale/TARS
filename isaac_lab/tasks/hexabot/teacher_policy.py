# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Distillable terrain-teacher actor-critic for the Hexabot rough-terrain policy.

This is the structural piece that makes Milestone 1 the teacher of a future blind
proprioceptive student (the distillation seam for the next milestone).

The single policy observation group is laid out as

    [ proprio (75) | height_scan (n_scan, PRIVILEGED) ]

and this module routes the privileged terrain channel through a NARROW LATENT
BOTTLENECK, kept isolated from the proprioceptive path:

    z      = scan_encoder(height_scan)         # n_scan -> latent_dim  (the bottleneck)
    action = actor_trunk( [proprio | z] )      # 75 + latent_dim -> 18

That is the RMA / teacher-student decomposition. To distill into a blind student
next milestone, you KEEP `actor_trunk` verbatim and SWAP `scan_encoder` for a
proprioceptive-history encoder trained to regress the same `z` — a clean module
swap, not a re-architecture, because the trunk only ever sees `[proprio | z]` and
never the raw scan.

The critic is privileged: it consumes the full observation (proprio + raw scan)
directly, which is fine because the critic is discarded at deployment.

Implementation notes
--------------------
* Subclasses rsl_rl's `ActorCritic` so PPO, the symmetry augmentation, the
  empirical normalizers and the runner checkpoint plumbing all keep working. We
  reuse the base machinery (action std, distribution, critic) and only replace the
  actor's internal forward with the encoder+trunk split.
* `get_actor_obs` returns the concatenated `policy` group (proprio + scan), i.e.
  the FULL observation tensor; the split into proprio / scan happens here by index.
* Registered into rsl_rl's runner namespace by `train_rough.py` so the cfg's
  `class_name="HexabotTeacherActorCritic"` resolves.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.networks import MLP


class HexabotTeacherActorCritic(ActorCritic):
    """ActorCritic whose privileged height scan enters via a latent bottleneck."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        n_proprio: int = 75,
        scan_latent_dim: int = 16,
        scan_encoder_dims: tuple[int, ...] | list[int] = (128, 64),
        actor_hidden_dims: tuple[int] | list[int] = (128, 128, 128),
        critic_hidden_dims: tuple[int] | list[int] = (128, 128, 128),
        activation: str = "elu",
        **kwargs,
    ):
        # Build the base ActorCritic (gives us the critic over the FULL obs, the
        # action std parameter, normalizers and the distribution machinery). Its
        # auto-built `self.actor` (full-obs -> actions) is replaced below.
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )

        # full actor-obs width = proprio + scan (single 'policy' group)
        num_actor_obs = self.get_actor_obs(obs).shape[-1]
        self.n_proprio = n_proprio
        self.n_scan = num_actor_obs - n_proprio
        assert self.n_scan > 0, (
            f"teacher expects obs = [proprio({n_proprio}) | scan(>0)], got width {num_actor_obs}"
        )
        self.scan_latent_dim = scan_latent_dim

        # privileged scan -> latent bottleneck (isolated from the proprio path)
        self.scan_encoder = MLP(self.n_scan, scan_latent_dim, list(scan_encoder_dims), activation)
        # actor trunk consumes [proprio | latent]; THIS is what the student reuses
        self.actor_trunk = MLP(n_proprio + scan_latent_dim, num_actions, list(actor_hidden_dims), activation)

        # point the base-class `self.actor` at the trunk for any generic introspection
        # (e.g. exporters reading actor[0].in_features). The real forward is below.
        self.actor = self.actor_trunk

        print(f"Scan encoder: {self.scan_encoder}")
        print(f"Actor trunk : {self.actor_trunk}")

    # -- the bottleneck forward -------------------------------------------------
    def _actor_forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        proprio = actor_obs[..., : self.n_proprio]
        scan = actor_obs[..., self.n_proprio :]
        z = self.scan_encoder(scan)
        return self.actor_trunk(torch.cat([proprio, z], dim=-1))

    # -- override the points where the base class calls self.actor(full_obs) -----
    def _update_distribution(self, obs: torch.Tensor) -> None:
        from torch.distributions import Normal

        mean = self._actor_forward(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        self._update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        return self._actor_forward(actor_obs)
