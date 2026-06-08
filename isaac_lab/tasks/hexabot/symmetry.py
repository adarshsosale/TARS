# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal) symmetry augmentation for the Hexabot.

The hexagonal hexapod is exactly mirror-symmetric across the body x-z (sagittal)
plane, and the task — track a forward +x velocity, hold heading, walk straight —
is itself symmetric under that reflection. So for every transition (s, a) the
mirrored transition (M·s, M·a) is equally valid on-policy data. RSL-RL's PPO
consumes this via `RslRlSymmetryCfg(use_data_augmentation=True, ...)`: it appends
the mirrored copy to each minibatch, which (a) forces a mirror-symmetric policy
-> no left/right bias -> straight walking (dy -> 0), and (b) doubles the effective
data per update for free -> steadier convergence.

The reflection (world/body y -> -y) acts on the observation as:
  * root lin vel  (x, y, z)   -> ( x, -y,  z)
  * root ang vel  (wx,wy,wz)  -> (-wx, wy,-wz)
  * proj. gravity (gx,gy,gz)  -> ( gx,-gy, gz)
  * command (vx, vy, yaw)     -> ( vx,-vy,-yaw)
  * joint pos / vel / actions -> swap each leg with its left-right mirror, and
    flip the sign of the coxa (yaw) joints; femur/tibia are pitch joints whose
    lift is handedness-free, so they swap without a sign flip (same treatment as
    ANYmal's HAA vs HFE/KFE).
  * gait clock (sin, cos)     -> UNCHANGED. The phase schedule depends only on a
    leg's front/mid/rear position, not its side, so a mirror pair shares its
    phase offset and the clock is invariant under the swap.

The per-joint mirror permutation `_jt_mirror_idx` and sign vector `_jt_mirror_sign`
are built once from the live joint names in `HexabotEnv.__init__` (so this stays
correct regardless of the order Isaac assigns the DOFs).

Obs layout (68): [lin_vel 0:3][ang_vel 3:6][grav 6:9][cmd 9:12]
                 [joint_pos 12:30][joint_vel 30:48][actions 48:66][gait_clock 66:68]
"""

from __future__ import annotations

import torch

__all__ = ["compute_symmetric_states_lr"]


def _mirror_joint_vec(x: torch.Tensor, midx: torch.Tensor, msign: torch.Tensor) -> torch.Tensor:
    """mirrored[i] = sign[i] * x[partner(i)] — swap legs then sign-flip coxa joints."""
    return x[..., midx] * msign


def _mirror_policy_obs(obs: torch.Tensor, midx: torch.Tensor, msign: torch.Tensor) -> torch.Tensor:
    obs = obs.clone()
    dev = obs.device
    obs[:, 0:3] = obs[:, 0:3] * torch.tensor([1.0, -1.0, 1.0], device=dev)      # lin vel
    obs[:, 3:6] = obs[:, 3:6] * torch.tensor([-1.0, 1.0, -1.0], device=dev)     # ang vel
    obs[:, 6:9] = obs[:, 6:9] * torch.tensor([1.0, -1.0, 1.0], device=dev)      # proj gravity
    obs[:, 9:12] = obs[:, 9:12] * torch.tensor([1.0, -1.0, -1.0], device=dev)   # cmd vx,vy,yaw
    obs[:, 12:30] = _mirror_joint_vec(obs[:, 12:30], midx, msign)               # joint pos (rel)
    obs[:, 30:48] = _mirror_joint_vec(obs[:, 30:48], midx, msign)               # joint vel
    obs[:, 48:66] = _mirror_joint_vec(obs[:, 48:66], midx, msign)               # last actions
    # 66:68 gait clock — invariant under the left-right swap, left untouched
    return obs


@torch.no_grad()
def compute_symmetric_states_lr(env, obs=None, actions=None):
    """Augment a batch with its left-right mirror (2x: original + mirrored).

    Signature matches `RslRlSymmetryCfg.data_augmentation_func`: `obs` is a
    TensorDict with a "policy" group (or None) and `actions` is a tensor (or None).
    Returns the augmented `(obs, actions)`, each with batch dim multiplied by 2.
    """
    unwrapped = env.unwrapped
    midx = unwrapped._jt_mirror_idx
    msign = unwrapped._jt_mirror_sign

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(obs["policy"], midx, msign)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = _mirror_joint_vec(actions, midx, msign)
    else:
        actions_aug = None

    return obs_aug, actions_aug
