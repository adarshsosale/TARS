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

# Height-scanner footprint, scaled to the hexapod (foot span ~0.59 m).
# Milestone 1.5 Phase C: EXTEND the grid forward (not shift) so rear-foot coverage —
# now load-bearing for the per-region ground height (Phase B) — is preserved while
# adding ~0.7 s of forward lookahead at the commanded speed. A 0.6 x 0.3 m grid at
# 0.05 m gives 13 x 7 = 91 rays. The grid is centred on the sensor, so a +0.10 m
# forward sensor offset (`_SCAN_OFFSET_X`, applied in the RayCasterCfg below) puts
# the body-frame x span at [-0.20, +0.40] m: rear edge unchanged at -0.20, +0.20 m of
# new lookahead. ray_starts bakes the offset in, so the env reads body-frame x directly.
_SCAN_SIZE = (0.6, 0.3)
_SCAN_RES = 0.05
_SCAN_OFFSET_X = 0.10                                   # forward sensor shift -> x in [-0.20, 0.40]
_SCAN_NX = int(round(_SCAN_SIZE[0] / _SCAN_RES)) + 1   # 13
_SCAN_NY = int(round(_SCAN_SIZE[1] / _SCAN_RES)) + 1   # 7
N_HEIGHT_SCAN = _SCAN_NX * _SCAN_NY                     # 91


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
    # --- privileged height scan fills the dormant exteroceptive slot (81 -> 172) ---
    # proprio width = 57 + action_space(24); keep in sync with HexabotFlatEnvCfg.
    n_height_scan = N_HEIGHT_SCAN
    observation_space = 81 + N_HEIGHT_SCAN

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
        offset=RayCasterCfg.OffsetCfg(pos=(_SCAN_OFFSET_X, 0.0, 20.0)),
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
    # Promote/demote by absolute distance walked, as a fraction of the tile size.
    # The harder mixed terrain (rough_terrains.py) shortens the distance the robot
    # covers per episode, so the old half-tile (size * 0.5 = 1.0 m) promotion bar left
    # the top curriculum levels unreachable. These loosen it so Curriculum/terrain_level
    # can still climb to max: walk past `promote_frac` of a tile -> level up; cover less
    # than `demote_frac` of a tile -> level down. Mean level is the lead progress metric.
    terrain_curriculum_enabled = True
    terrain_promote_frac = 0.35   # was 0.5 (half-tile / 1.0 m); 0.35 -> 0.70 m
    terrain_demote_frac = 0.15    # cover < 0.30 m -> demote (was: < half the commanded distance)

    # --- Phase D (ablation): terrain-conditioned tetrapod relaxation -------------
    # tetrapod_contact (+3) encodes a flat-ground "exactly 4 of 6 feet down" gait
    # template. On uneven steps the terrain-correct support count is terrain-dependent,
    # so at high levels the template can pay reward against terrain-correct contact
    # schedules. This scales the tetrapod_contact weight DOWN per-env with terrain level:
    #     w_eff = w * (1 - tetrapod_relax_frac * level / level_max)
    # 0.0 = no relaxation (the B+C baseline). The D ablation sets 0.5 (via
    # train_rough.py --tetrapod_relax_frac). gait_symmetry / gait_phase are NOT relaxed
    # — gait_phase is the CPG phase-lock that prevents flailing.
    tetrapod_relax_frac = 0.0

    # On rough terrain the strict straight-line shaping from the flat task is too harsh
    # (the body legitimately yaws/drifts crossing slopes and steps). Soften the
    # heading / lateral-position penalties so velocity tracking over terrain dominates.
    heading_reward_scale = -1.0       # was -4.0
    lateral_pos_reward_scale = -0.5   # was -2.0

    # --- Phase E: belly-contact tolerance + anticipatory ride height -------------
    # Level-8 failure mode: the belly touches the ground and the episode ends — the
    # flat-ground 1 N base-contact death treats a graze as a catastrophe, and the
    # policy never learns to raise its body before rough patches. Three changes:
    #
    # (1) GRADED belly contact instead of binary death. Death is reserved for
    #     component-damage impacts (~2.6x the 19 N bodyweight); below that, contact
    #     force above a 2 N free allowance is priced linearly (belly_contact_force,
    #     scale inherited -0.1), so a light nudge while clambering is a usable
    #     optimization, leaning weight on the belly is taxed, slamming it is death.
    #     The too-low death likewise tolerates 0.5 s of transient belly-down (e.g.
    #     straddling a box) before firing — lying flat still terminates.
    base_contact_force_death = 50.0   # N (flat keeps 1.0 = any touch)
    too_low_grace_steps = 25          # 0.5 s @50 Hz (flat keeps 0 = immediate)
    #
    # (2) ANTICIPATORY "belly higher than normal" mode. HexabotRoughEnv converts the
    #     height scan's q90-above-mid-ground protrusion (body + 0.40 m lookahead)
    #     into _height_target_offset, which raises the base-height target and the
    #     belly_clearance floor up to height_raise_max — so the strong -50
    #     belly_clearance penalty starts paying the policy to extend its legs (the
    #     new CPG d_stance channel, cpg.py) BEFORE it commits to an obstacle. The
    #     same protrusion raises the rewarded swing apex (foot_clearance) so bigger
    #     steps pay exactly where the terrain demands them.
    height_raise_gain = 1.0           # ride-height raise per metre of protrusion
    height_raise_max = 0.030          # cap [m] (~the d_stance channel's authority)
    foot_clearance_raise_max = 0.025  # swing-apex raise cap [m] (0.025 flat -> up to 0.05)
    #
    # (3) STUMBLE penalty: a swing toe catching a vertical face (box wall / stair
    #     riser) shows up as a horizontal-dominated foot contact force — the
    #     "mis-step" the eval flagged on stairs. Inert on flat (scale 0 there).
    foot_stumble_reward_scale = -1.0

    # --- on-demand render camera (off during training; render_rough.py turns it on) -
    enable_render_camera = False
