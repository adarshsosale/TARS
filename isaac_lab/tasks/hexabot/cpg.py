# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Central Pattern Generator (CPG) for the Hexabot locomotion layer.

This is the SINGLE SOURCE OF TRUTH for how the policy's action becomes joint
targets. Milestone-0 hard constraint #2: the analytical tripod gait is a
*reference*, not a residual basis — so instead of the policy emitting joint
offsets directly, it MODULATES a per-leg phase oscillator whose nominal
(zero-action) behaviour is exactly the open-loop tripod gait in
`hexabot_model/isaac/tripod_gait.py:gait_pose`.

Design
------
* Per leg i: an absolute phase `theta_i in [0, 2pi)`, advanced each control step.
* Phases are SEEDED at the alternating-tripod offsets (tripod A legs share a
  phase, tripod B legs lag by half a cycle) and held there by weak phase
  COUPLING, so the tripod structure is robust while the policy retimes/reshapes
  it. At the tripod equilibrium the coupling term is exactly zero, so zero action
  reproduces the analytical gait bit-for-bit (verified in tests).
* The policy action is per-leg `[d_freq, d_coxa_amp, d_lift, d_stance]`
  (leg-major, 24 total). Nominal amplitudes come straight from the analytical
  gait defaults (`coxa_amp=0.26`, `lift=0.55`).

      f_i    = f_base    * (1 + kf   * d_freq_i)        # cadence  -> speed
      mu_i   = mu_base   * (1 + kmu  * d_coxa_amp_i)     # coxa sweep -> stride
      lift_i = lift_base * (1 + klift* d_lift_i)         # swing height (terrain later)
      st_i   = kstance   * clamp(d_stance_i, 0, 1)       # femur press-down -> ride height

  `d_stance` is the rough-terrain posture channel: a ONE-SIDED femur press-down
  offset that extends the leg and raises the body ("belly higher than normal"
  mode). One-sided because lowering the body would reopen the belly-crawl
  optimum the CPG exists to forbid. It is a posture, not a gait amplitude, so it
  is NOT scaled by the commanded speed and applies through stance AND swing
  (the whole trajectory shifts down at the foot == body up). At d_stance<=0 it
  is exactly 0, preserving the zero-action contract below.

* Joint mapping reuses the analytical waveform verbatim: stance half sweeps the
  coxa to push the body, swing half lifts femur/tibia. Output is raw joint
  targets (N, 18); the caller clamps to soft joint limits.

`action = 0  =>  exactly the analytical tripod gait` is the contract that makes
BC warm-start and the annealing imitation reward well-defined.
"""

from __future__ import annotations

import math

import torch

# --- analytical-gait constants (kept in sync with tripod_gait.py / generate_hexabot.py) ---
LEG_AZ_DEG = {"lf": 30.0, "lm": 90.0, "lr": 150.0, "rf": -30.0, "rm": -90.0, "rr": -150.0}
TRIPOD_A = {"lf", "rm", "lr"}  # phase offset 0; tripod B = {rf, lm, rr} -> offset 0.5
STANCE_FEMUR = math.radians(-18.0)
STANCE_TIBIA = math.radians(64.0)
# canonical leg order used for the action / phase-obs layout (leg-major)
LEG_ORDER = ["lf", "lm", "lr", "rf", "rm", "rr"]


def _mirror_leg(name: str) -> str:
    """'lf' -> 'rf' (left<->right reflection across the sagittal plane)."""
    side = "r" if name[0] == "l" else "l"
    return side + name[1:]


class HexabotCPG:
    """Tripod-seeded per-leg phase CPG. Vectorized over `num_envs`."""

    ACTION_PER_LEG = 4   # [d_freq, d_coxa_amp, d_lift, d_stance]
    N_LEGS = 6
    ACTION_DIM = N_LEGS * ACTION_PER_LEG          # 24
    PHASE_OBS_DIM = N_LEGS * 2                     # per-leg sin/cos -> 12

    def __init__(self, joint_names: list[str], num_envs: int, device, cfg):
        self.device = device
        self.num_envs = num_envs
        self.f_base = cfg.cpg_f_base
        self.mu_base = cfg.cpg_coxa_amp
        self.lift_base = cfg.cpg_lift
        self.kf = cfg.cpg_kf
        self.kmu = cfg.cpg_kmu
        self.klift = cfg.cpg_klift
        self.kstance = getattr(cfg, "cpg_kstance", 0.0)
        self.coupling = cfg.cpg_coupling_strength

        # per-leg geometry, in LEG_ORDER
        self._sin_az = torch.tensor(
            [math.sin(math.radians(LEG_AZ_DEG[lg])) for lg in LEG_ORDER], device=device
        )  # (6,)
        self._psi = torch.tensor(
            [0.0 if lg in TRIPOD_A else 0.5 for lg in LEG_ORDER], device=device
        )  # tripod offset in cycles (6,)
        # target pairwise phase differences for coupling: psi_j - psi_i  (in rad)
        self._dpsi = (self._psi.view(1, -1) - self._psi.view(-1, 1)) * 2.0 * math.pi  # (6,6)

        # map each leg -> (coxa_idx, femur_idx, tibia_idx) in the live DOF order
        name_to_idx = {nm: i for i, nm in enumerate(joint_names)}
        self._coxa_idx = torch.tensor([name_to_idx[f"coxa_{lg}"] for lg in LEG_ORDER], device=device)
        self._femur_idx = torch.tensor([name_to_idx[f"femur_{lg}"] for lg in LEG_ORDER], device=device)
        self._tibia_idx = torch.tensor([name_to_idx[f"tibia_{lg}"] for lg in LEG_ORDER], device=device)
        self._n_joints = len(joint_names)

        # leg-mirror permutation (over LEG_ORDER) for symmetry augmentation
        self._leg_mirror = torch.tensor(
            [LEG_ORDER.index(_mirror_leg(lg)) for lg in LEG_ORDER], device=device, dtype=torch.long
        )

        # per-env phase state (seeded in reset)
        self.theta = torch.zeros(num_envs, self.N_LEGS, device=device)
        self.reset(torch.arange(num_envs, device=device))

    # ---- state -------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor):
        """Seed phases at a random global offset + the fixed tripod offsets."""
        n = len(env_ids)
        phi0 = torch.rand(n, 1, device=self.device)              # random per episode -> decorrelate envs
        self.theta[env_ids] = ((phi0 + self._psi.view(1, -1)) % 1.0) * 2.0 * math.pi

    def _params(self, action: torch.Tensor, speed_scale: torch.Tensor | float = 1.0):
        """Split the (N,24) action into per-leg (f, mu, lift, stance), each (N,6).

        `speed_scale` (scalar or (N,1)) scales the nominal stride/lift amplitude by
        the commanded speed: at speed_scale=0 the nominal gait collapses to the
        standing stance (coxa sweep 0, no lift) so zero-action + vx=0 == stand; at
        speed_scale=1 it is the full analytical tripod gait. Frequency is NOT
        scaled (phase keeps advancing; flat amplitude just freezes the joints).
        The stance press-down is a POSTURE (ride height), not a gait amplitude, so
        it is one-sided (raise only) and independent of speed_scale.
        """
        a = action.view(-1, self.N_LEGS, self.ACTION_PER_LEG)
        if not torch.is_tensor(speed_scale):
            speed_scale = torch.as_tensor(speed_scale, device=self.device)
        s = speed_scale if speed_scale.ndim == 0 else speed_scale.view(-1, 1)
        f = (self.f_base * (1.0 + self.kf * a[..., 0])).clamp(min=0.0)
        mu = (self.mu_base * s * (1.0 + self.kmu * a[..., 1])).clamp(min=0.0)
        lift = (self.lift_base * s * (1.0 + self.klift * a[..., 2])).clamp(min=0.0)
        stance = self.kstance * a[..., 3].clamp(0.0, 1.0)
        return f, mu, lift, stance

    def step(self, action: torch.Tensor, dt: float):
        """Advance every leg's phase by one control step (freq-modulated + coupled)."""
        f, _, _, _ = self._params(action)                         # (N,6)
        dtheta = 2.0 * math.pi * f                                # base advance
        if self.coupling > 0.0:
            # weak pull toward the tripod phase relationships (zero at equilibrium)
            diff = self.theta.unsqueeze(1) - self.theta.unsqueeze(2) - self._dpsi.unsqueeze(0)  # (N,6,6)
            dtheta = dtheta + self.coupling * torch.sin(diff).mean(dim=2)
        self.theta = (self.theta + dtheta * dt) % (2.0 * math.pi)

    # ---- outputs -----------------------------------------------------------
    def joint_targets(self, action: torch.Tensor, speed_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Map the CURRENT phase + action amplitudes to raw joint targets (N,18).

        At `speed_scale=1`, `action=0` reproduces `tripod_gait.gait_pose` exactly;
        at `speed_scale=0` it holds the standing stance.
        """
        _, mu, lift, press = self._params(action, speed_scale)
        p = self.theta / (2.0 * math.pi)                          # (N,6) in [0,1)
        stance = p < 0.5
        s_stance = p / 0.5
        s_swing = (p - 0.5) / 0.5
        # coxa: +/-1 ramp in stance, reverse ramp in swing -> a smooth sweep
        sweep = torch.where(stance, 2.0 * s_stance - 1.0, 1.0 - 2.0 * s_swing)
        qc = mu * self._sin_az.unsqueeze(0) * sweep               # (N,6)
        # femur/tibia lift only during swing; the stance press-down offset applies
        # to the WHOLE cycle (posture: foot trajectory shifts down == body rides up)
        k = torch.sin(math.pi * s_swing) * (~stance).float()
        qf = STANCE_FEMUR + press - lift * k
        qt = STANCE_TIBIA + 0.4 * lift * k

        targets = torch.zeros(action.shape[0], self._n_joints, device=self.device)
        targets[:, self._coxa_idx] = qc
        targets[:, self._femur_idx] = qf
        targets[:, self._tibia_idx] = qt
        return targets

    def nominal_joint_targets(self, speed_scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Analytical-gait joint targets at the current phase (action == 0).

        Used by the annealing imitation reward as the reference to track.
        """
        zero = torch.zeros(self.num_envs, self.ACTION_DIM, device=self.device)
        return self.joint_targets(zero, speed_scale)

    def phase_obs(self) -> torch.Tensor:
        """Per-leg [sin theta, cos theta] -> (N, 12), leg-major."""
        s = torch.sin(self.theta)
        c = torch.cos(self.theta)
        return torch.stack([s, c], dim=-1).reshape(self.num_envs, self.PHASE_OBS_DIM)

    # ---- symmetry helpers --------------------------------------------------
    def action_mirror_idx(self) -> torch.Tensor:
        """Permutation over the 18 action dims that mirrors L<->R legs.

        CPG params are sign-invariant amplitudes/frequencies, so the mirror is a
        pure leg-block swap with NO sign flip (unlike direct joint offsets, where
        the coxa-yaw flips sign).
        """
        idx = torch.empty(self.ACTION_DIM, device=self.device, dtype=torch.long)
        for i in range(self.N_LEGS):
            j = int(self._leg_mirror[i])
            for k in range(self.ACTION_PER_LEG):
                idx[i * self.ACTION_PER_LEG + k] = j * self.ACTION_PER_LEG + k
        return idx

    def phase_mirror_idx(self) -> torch.Tensor:
        """Permutation over the 12 phase-obs dims mirroring L<->R legs (no sign)."""
        idx = torch.empty(self.PHASE_OBS_DIM, device=self.device, dtype=torch.long)
        for i in range(self.N_LEGS):
            j = int(self._leg_mirror[i])
            idx[i * 2 + 0] = j * 2 + 0   # sin
            idx[i * 2 + 1] = j * 2 + 1   # cos
        return idx
