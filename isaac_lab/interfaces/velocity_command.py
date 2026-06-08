# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""The frozen Navigation -> Locomotion interface: a planar velocity command.

This is the *single, documented boundary* between the two layers of the Hexabot
stack (Milestone-0 hard constraint #3):

    Navigation layer  --(vx, vy, yaw)-->  Locomotion layer

* The Navigation layer (goal-conditioned) PRODUCES a `VelocityCommand`.
* The Locomotion layer (PPO) CONSUMES it as part of its observation.

Neither layer imports the other; both import THIS module. Keeping the contract
in one place means widening the command space later (e.g. enabling lateral or
yaw commands for obstacle avoidance) is a one-line change to the ranges here and
touches neither policy's architecture.

Frame convention (matches the asset, see hexabot_model/generate_hexabot.py):
metres, +X forward, +Y left, +Z up; `vx`,`vy` are body-frame planar linear
velocities [m/s], `yaw` is the body-frame yaw RATE [rad/s].

Flat-ground / straight-line stage 0: only `vx` is commanded; `vy` and `yaw` are
pinned to 0. The ranges below are the canonical authority — later curriculum
stages widen `VY_RANGE`/`YAW_RANGE` WITHOUT changing either layer.
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical command ranges (the one place that defines the command space).
# Stage 1 enables turning (yaw); lateral strafing (vy) stays dormant. Widen further later.
VX_RANGE: tuple[float, float] = (0.0, 0.30)   # forward speed [m/s]
VY_RANGE: tuple[float, float] = (0.0, 0.0)    # lateral speed [m/s]  (dormant until strafing stages)
YAW_RANGE: tuple[float, float] = (-0.5, 0.5)  # yaw rate [rad/s]     (ENABLED stage 1: turning)


@dataclass(frozen=True)
class VelocityCommand:
    """A planar velocity command, the only thing crossing the layer boundary.

    Immutable on purpose — a command is produced once per navigation tick and
    consumed read-only by locomotion.
    """

    vx: float
    vy: float = 0.0
    yaw: float = 0.0

    def clamp(self) -> "VelocityCommand":
        """Return a copy clamped to the canonical command ranges."""
        return VelocityCommand(
            vx=min(max(self.vx, VX_RANGE[0]), VX_RANGE[1]),
            vy=min(max(self.vy, VY_RANGE[0]), VY_RANGE[1]),
            yaw=min(max(self.yaw, YAW_RANGE[0]), YAW_RANGE[1]),
        )

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.vx, self.vy, self.yaw)

    def to_tensor(self, device=None, batch: int | None = None):
        """`(vx, vy, yaw)` as a torch tensor; shape (3,) or (batch, 3).

        Imported lazily so non-sim consumers (tests, the hand-coded navigation
        controller) need not depend on torch.
        """
        import torch

        t = torch.tensor([self.vx, self.vy, self.yaw], dtype=torch.float32, device=device)
        if batch is not None:
            t = t.unsqueeze(0).expand(batch, 3).clone()
        return t
