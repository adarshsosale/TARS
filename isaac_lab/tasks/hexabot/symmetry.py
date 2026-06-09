# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Left-right (sagittal) symmetry augmentation for the Hexabot locomotion policy.

The hexagonal hexapod is exactly mirror-symmetric across the body x-z (sagittal)
plane, and the task — track a forward +x velocity, hold heading, walk straight —
is itself symmetric under that reflection. So for every transition (s, a) the
mirrored transition (M·s, M·a) is equally valid on-policy data. RSL-RL's PPO
consumes this via `RslRlSymmetryCfg(use_data_augmentation=True, ...)`: it appends
the mirrored copy to each minibatch, forcing a mirror-symmetric policy (no
left/right drift -> straight walking) and doubling effective data for free.

Reflection (world/body y -> -y) acting on the PROPRIOCEPTIVE obs:
  * proj. gravity (gx,gy,gz)  -> ( gx,-gy, gz)
  * base ang vel  (wx,wy,wz)  -> (-wx, wy,-wz)
  * command (vx, vy, yaw)     -> ( vx,-vy,-yaw)
  * joint pos / vel           -> swap each leg with its L<->R mirror, sign-flip the
                                 coxa (yaw) joints (femur/tibia pitch joints swap
                                 without a flip — ANYmal HAA vs HFE/KFE).
  * previous action (CPG)     -> swap each leg's [d_freq,d_coxa_amp,d_lift] block
                                 with its mirror. NO sign flip: CPG params are
                                 sign-invariant amplitudes/frequencies (the coxa
                                 sweep's handedness lives in sin(azimuth), which is
                                 carried by the leg swap, not the param sign).
  * CPG phase (per-leg sin/cos) -> swap each leg's (sin,cos) block with its mirror;
                                 values UNCHANGED (the reflection maps leg lf's pose
                                 to leg rf's pose, so mirrored phase = partner phase).
  * dormant height-scan block -> left untouched (zeros now).

The base linear velocity is NOT in the obs (proprioceptive-only), so there is no
lin-vel channel to mirror here — unlike the old 68-dim layout.

Obs layout (75 [+ n_height_scan]):
  [grav 0:3][ang_vel 3:6][cmd 6:9][jpos 9:27][jvel 27:45][prev_act 45:63][cpg 63:75]

`_jt_mirror_idx`/`_jt_mirror_sign` (joints), `_act_mirror_idx` (CPG action),
`_cpg_mirror_idx` (CPG phase) are all built once in `HexabotEnv.__init__`.
"""

from __future__ import annotations

import torch

__all__ = ["compute_symmetric_states_lr"]

# obs slice boundaries (proprioceptive, no base linear velocity)
_GRAV = slice(0, 3)
_ANGV = slice(3, 6)
_CMD = slice(6, 9)
_JPOS = slice(9, 27)
_JVEL = slice(27, 45)
_PREVACT = slice(45, 63)
_CPG = slice(63, 75)


def _mirror_joint_vec(x: torch.Tensor, midx: torch.Tensor, msign: torch.Tensor) -> torch.Tensor:
    """mirrored[i] = sign[i] * x[partner(i)] — swap legs then sign-flip coxa joints."""
    return x[..., midx] * msign


def _mirror_policy_obs(obs, midx, msign, act_idx, cpg_idx, scan_idx=None) -> torch.Tensor:
    obs = obs.clone()
    dev = obs.device
    obs[:, _GRAV] = obs[:, _GRAV] * torch.tensor([1.0, -1.0, 1.0], device=dev)
    obs[:, _ANGV] = obs[:, _ANGV] * torch.tensor([-1.0, 1.0, -1.0], device=dev)
    obs[:, _CMD] = obs[:, _CMD] * torch.tensor([1.0, -1.0, -1.0], device=dev)
    obs[:, _JPOS] = _mirror_joint_vec(obs[:, _JPOS], midx, msign)
    obs[:, _JVEL] = _mirror_joint_vec(obs[:, _JVEL], midx, msign)
    obs[:, _PREVACT] = obs[:, _PREVACT][:, act_idx]      # CPG action mirror: leg-swap, no sign
    obs[:, _CPG] = obs[:, _CPG][:, cpg_idx]              # CPG phase: leg-swap, no value change
    # 75:  PRIVILEGED height-scan block (rough terrain). The reflection y -> -y permutes
    # the scan rays to their left-right partners (heights themselves are unchanged).
    if scan_idx is not None and obs.shape[1] > 75:
        scan = obs[:, 75:]
        obs[:, 75:] = scan[:, scan_idx]
    return obs


@torch.no_grad()
def compute_symmetric_states_lr(env, obs=None, actions=None):
    """Augment a batch with its left-right mirror (2x: original + mirrored).

    Signature matches `RslRlSymmetryCfg.data_augmentation_func`.
    """
    unwrapped = env.unwrapped
    midx = unwrapped._jt_mirror_idx
    msign = unwrapped._jt_mirror_sign
    act_idx = unwrapped._act_mirror_idx
    cpg_idx = unwrapped._cpg_mirror_idx
    scan_idx = getattr(unwrapped, "_scan_mirror_idx", None)   # present only on rough terrain

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size:] = _mirror_policy_obs(obs["policy"], midx, msign, act_idx, cpg_idx, scan_idx)
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size:] = actions[:, act_idx]   # CPG action mirror: leg-swap, no sign
    else:
        actions_aug = None

    return obs_aug, actions_aug
