# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rough-terrain RL env config for the Hexabot (Milestone 1).

EXTENDS the flat locomotion config (`HexabotFlatEnvCfg`) — same 50 Hz control slot,
same frozen (vx, vy, yaw) interface, same CPG action, same reward family. The only
additions are what rough terrain needs:

  * `terrain_type="generator"` with a curriculum over blind-feasible sub-terrains
    (`rough_terrains.py`), starting near-flat and ramping with the terrain-level
    curriculum driven from `HexabotRoughEnv._reset_idx`.
  * a PRIVILEGED height-scanner (`RayCasterCfg`) that fills the dormant
    `n_height_scan` observation slot M0 reserved. This is teacher-only exteroception:
    it enters the policy ONLY through the latent bottleneck of
    `HexabotTeacherActorCritic` (see teacher_policy.py), kept isolated from the
    proprioceptive path so the teacher distills cleanly into a blind student.
  * terrain-relevant domain randomization (payload mass / COM shift) on top of the
    inherited friction / actuator / IMU randomization.

NOTHING about the flat task, the navigation layer, or the frozen interface changes.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from .hexabot_env_cfg import EventCfg, HexabotFlatEnvCfg
from .rough_terrains import HEXABOT_ROUGH_TERRAINS_CFG

# Height-scanner footprint, scaled to the hexapod (foot span ~0.59 m). A 0.4 x 0.3 m
# grid at 0.05 m gives 9 x 7 = 63 rays — the privileged terrain channel width.
_SCAN_SIZE = (0.4, 0.3)
_SCAN_RES = 0.05
_SCAN_NX = int(round(_SCAN_SIZE[0] / _SCAN_RES)) + 1   # 9
_SCAN_NY = int(round(_SCAN_SIZE[1] / _SCAN_RES)) + 1   # 7
N_HEIGHT_SCAN = _SCAN_NX * _SCAN_NY                     # 63


@configclass
class RoughEventCfg(EventCfg):
    """Flat DR (friction / mass / actuator gains) + terrain-relevant payload & COM.

    Per-sub-terrain friction is already covered: the inherited `physics_material`
    term buckets friction across the robot/terrain contact. Here we add a payload
    mass offset and a base centre-of-mass shift so the teacher does not overfit to a
    nominal inertia on uneven ground.
    """

    # additive payload on the body (a carried mass / build tolerance), kg
    payload_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.1, 0.2),  # kg, added (body ~ part of 1.926 kg)
            "operation": "add",
            "distribution": "uniform",
        },
    )
    # base centre-of-mass shift (payload not perfectly centred), metres
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        },
    )


@configclass
class HexabotRoughEnvCfg(HexabotFlatEnvCfg):
    # --- privileged height scan fills the dormant exteroceptive slot (75 -> 138) ---
    n_height_scan = N_HEIGHT_SCAN
    observation_space = 75 + N_HEIGHT_SCAN

    # --- generator terrain with a curriculum (level 0 ~ flat) -------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HEXABOT_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=0,     # everyone starts on the easiest (near-flat) row
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.1,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # --- PRIVILEGED height scanner (teacher-only exteroception) -----------------
    # Attached to base_link, yaw-aligned (heading-relative, roll/pitch-invariant like a
    # real terrain estimate). Read in HexabotRoughEnv and routed ONLY through the
    # teacher's latent bottleneck. update_period is set to the control dt in the env.
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=_SCAN_RES, size=list(_SCAN_SIZE)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # clip + scale applied to the height scan before it enters the obs (metres).
    # height = scanner_z - hit_z - offset; clipped to a band around the robot height.
    height_scan_offset = 0.072        # subtract nominal standing height -> ~0 on flat
    height_scan_clip = 0.15           # clip |height| to +/- this (m), then it is the obs

    # terrain-relevant DR (adds payload mass / COM shift to the inherited DR)
    events: RoughEventCfg = RoughEventCfg()

    # --- terrain-level curriculum knobs (driven in HexabotRoughEnv._reset_idx) ---
    # Mirrors isaaclab terrain_levels_vel: walk past size/2 -> level up; fail to cover
    # half the commanded distance -> level down. Mean level is the lead progress metric.
    terrain_curriculum_enabled = True

    # On rough terrain the strict straight-line shaping from the flat task is too harsh
    # (the body legitimately yaws/drifts crossing slopes and steps). Soften the
    # heading / lateral-position penalties so velocity tracking over terrain dominates.
    heading_reward_scale = -1.0       # was -4.0
    lateral_pos_reward_scale = -0.5   # was -2.0

    # --- on-demand render camera (off during training; render_rough.py turns it on) -
    enable_render_camera = False
