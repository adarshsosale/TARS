# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Direct RL env config for the Hexabot 18-DOF hexapod flat-ground walking task.

The robot is a hexagonal-body hexapod: 6 legs x 3 DOF (coxa yaw, femur pitch,
tibia pitch) = 18 actuated joints. Unlike the TARS biped, a hexapod is
*statically stable* — at least one tripod of three feet is always planted, so
it does not have to balance an inverted pendulum. RL here is polish (track a
commanded forward velocity, walk straight, step cleanly) rather than a
prerequisite for not falling over.

Frame: metres, +X forward, +Y left, +Z up. base_link at the body centre.
Standing body-centre height ~= 0.072 m (feet settle to z~=0).
Tripod groups: A = {lf, rm, lr}, B = {rf, lm, rr}.
"""

import math

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Absolute path to the converted Hexabot USD (URDF -> USD via convert_urdf.py;
# the bash pipeline runs the conversion the first time if this file is missing).
HEXABOT_USD = "/home/adarshsosale/Workspace/Isaac RL Lab/hexabot_model/isaac/hexabot.usd"

# Standing stance baked into generate_hexabot.py (STANCE_* params), in radians.
_STANCE_FEMUR = math.radians(-18.0)   # knee carried slightly high
_STANCE_TIBIA = math.radians(64.0)    # tibia reaches down to plant the claw (needs tibia limit >= 1.117 rad)

# ---------------------------------------------------------------------------
# Robot articulation. Grippy contact material is applied via the EventCfg below
# (the URDF carried no <material>, so the converted USD uses PhysX defaults).
# Three actuator groups so coxa(swing) / femur(lift) / tibia(knee) tune apart.
# ---------------------------------------------------------------------------
HEXABOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=HEXABOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,   # legs are radially separated
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # spawn AT the 72 mm standing height. Spawning higher (was 0.085) makes the
        # body free-fall on reset and the underdamped legs overshoot DOWN to ~0.023 m
        # within ~5 control steps — below the 0.035 m too_low death floor — so every
        # episode died during the spawn transient before the policy could act. The
        # passive robot settles to ~0.070 m, so spawning here removes the drop.
        pos=(0.0, 0.0, 0.072),
        joint_pos={
            "coxa_.*": 0.0,
            "femur_.*": _STANCE_FEMUR,
            "tibia_.*": _STANCE_TIBIA,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # MG996R: ~1.08 N·m stall @6V, ~6.16 rad/s no-load. Stiffness ~ stall/0.1
        # => ~10 N·m/rad; effort headroom bumped modestly above stock so the legs
        # can drive real swings and push-off without saturating.
        "coxa": ImplicitActuatorCfg(
            joint_names_expr=["coxa_.*"],
            effort_limit_sim=1.6, velocity_limit_sim=6.16,
            stiffness=10.0, damping=0.4,
        ),
        "femur": ImplicitActuatorCfg(
            joint_names_expr=["femur_.*"],
            effort_limit_sim=1.6, velocity_limit_sim=6.16,
            stiffness=12.0, damping=0.4,
        ),
        "tibia": ImplicitActuatorCfg(
            joint_names_expr=["tibia_.*"],
            effort_limit_sim=1.6, velocity_limit_sim=6.16,
            stiffness=12.0, damping=0.4,
        ),
    },
)


@configclass
class EventCfg:
    """Startup randomization — chiefly the grippy foot/ground contact material."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.2),
            "dynamic_friction_range": (0.8, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class HexabotFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0
    decimation = 4                 # 200 Hz physics -> 50 Hz control
    action_scale = 0.5             # rad offset from the standing stance
    action_space = 18              # 6 legs x (coxa, femur, tibia)
    observation_space = 66         # 3+3+3 (root) + 3 (cmd) + 18+18 (joint pos/vel) + 18 (actions)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.1,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot + contact sensor
    robot: ArticulationCfg = HEXABOT_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # command ranges (straight-line: lateral & yaw commands are zero)
    cmd_vx_range = (0.10, 0.30)    # forward speed command [m/s] (upper used by curriculum)

    # --- stand-first curriculum ---------------------------------------------
    # The robot otherwise discovers a belly-crawl: lie flat and wiggle for a sliver
    # of forward reward at minimal effort/risk. We starve that optimum by making it
    # learn to STAND TALL before it is ever asked to move: command vx=0 for the
    # first `curriculum_stand_steps` control-steps, then ramp the upper vx command
    # from 0 to cmd_vx_range[1] over `curriculum_ramp_steps`. Counted in control
    # steps (~24 per training iteration).
    curriculum_stand_steps = 2000     # ~83 iters of pure standing
    curriculum_ramp_steps = 4000      # ~166 iters ramping 0 -> full speed

    # nominal upright base-centre height [m] (the hexapod stands ~72 mm)
    target_height = 0.072

    # --- belly / leg-posture geometry ---------------------------------------
    belly_half_thickness = 0.023      # base centre -> underbelly (BODY_H 46 mm / 2)
    belly_clearance_min = 0.045       # underbelly must stay >= 45 mm off the ground
    claw_offset = 0.135               # tibia-local x to the claw tip (= L_TIBIA)
    support_target = 0.045            # belly->foot vertical gap rewarded up to [m]

    # post-reset settling grace: suppress death terminations for this many control
    # steps after a reset so a residual spawn transient can't kill the episode
    # before the policy acts (15 steps = 0.3 s; episodes run to 600 steps).
    settle_steps = 15

    # reward scales
    # --- track a forward base velocity, go straight ---
    lin_vel_reward_scale = 2.0
    forward_progress_reward_scale = 3.0       # linear: 0 when standing -> must actually move
    yaw_rate_reward_scale = 0.5
    lateral_vel_reward_scale = -2.0
    yaw_rate_l2_reward_scale = -1.0
    lateral_pos_reward_scale = -2.0           # stay on the spawn x-axis (straight line)
    z_vel_reward_scale = -1.0
    ang_vel_reward_scale = -0.05
    flat_orientation_reward_scale = -2.5
    base_height_reward_scale = -8.0
    alive_reward_scale = 1.0                   # survival must clearly pay (was 0.5)
    # --- stand tall, don't belly-crawl ---
    belly_clearance_reward_scale = -50.0      # strong one-sided penalty: belly near ground
    foot_support_reward_scale = 20.0          # reward feet planted well below the belly
    # --- effort / smoothness ---
    joint_torque_reward_scale = -2.0e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    joint_limit_reward_scale = -1.0           # discourage slamming joint limits
    # --- stepping behaviour ---
    feet_air_time_reward_scale = 1.0          # reward real, lifted swing steps
    feet_air_time_threshold = 0.2             # min step duration to count [s]
    foot_slip_reward_scale = -0.1             # kill planted-foot scuffing
    undesired_contact_reward_scale = -1.0     # body/coxa/femur must not touch ground
