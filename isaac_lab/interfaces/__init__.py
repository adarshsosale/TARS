# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Frozen inter-layer interfaces for the Hexabot control stack.

The ONLY contract between the Navigation layer and the Locomotion layer is the
velocity command `(vx, vy, yaw)`. It is defined once here so both layers import
the same typed boundary and neither reaches across it. See `velocity_command.py`.
"""

from .velocity_command import (
    VX_RANGE,
    VY_RANGE,
    YAW_RANGE,
    VelocityCommand,
)

__all__ = ["VelocityCommand", "VX_RANGE", "VY_RANGE", "YAW_RANGE"]
