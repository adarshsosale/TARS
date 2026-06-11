# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Body-frame front/middle/rear partition of the height-scan rays (Milestone 1.5 Phase B).

Pure-torch, no Isaac import, so it is shared by `HexabotRoughEnv` (which builds the
masks from the live ray pattern) and `test_cpg.py` (which checks the partition
analytically). The single source of truth for HOW the scan rays are bucketed into
body regions for the per-region ground height.

A ray's body-frame x (already including the sensor's forward offset) decides its
region. Only rays over the BODY footprint (|x| <= `half`) feed ground height; rays
with x > half are forward LOOKAHEAD (Phase C) and excluded from every region. The
body span [-half, +half] is split into three equal thirds -> rear / middle / front,
matching the hexapod's rear / mid / front leg rows.
"""

from __future__ import annotations

import torch


def scan_region_masks(ray_x: torch.Tensor, half: float = 0.20):
    """Boolean (n_rays,) masks (front, mid, rear) over the BODY footprint.

    Args:
        ray_x: body-frame x of each ray (sensor offset already applied).
        half:  body half-span in metres covered by the ground-height regions.
               Rays with x outside [-half, +half] (e.g. forward lookahead) are in
               none of the three masks.

    Returns:
        (front_mask, mid_mask, rear_mask), each bool of shape (n_rays,), disjoint.
    """
    third = 2.0 * half / 3.0
    in_body = (ray_x >= -half - 1e-6) & (ray_x <= half + 1e-6)
    rear = in_body & (ray_x < -half + third)        # x in [-half, -half/3)
    front = in_body & (ray_x > half - third)         # x in (+half/3, +half]
    mid = in_body & ~rear & ~front                   # x in [-half/3, +half/3]
    return front, mid, rear
