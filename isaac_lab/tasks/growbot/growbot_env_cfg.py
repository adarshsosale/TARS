# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Direct RL env config for the TARS (Growbot) flat-ground straight-line walking task.

The robot has 4 sagittal pitch DOFs (hip_left/right, ankle_left/right), no knee
and no ankle-roll, so it can only locomote in the sagittal plane. We train a PPO
policy to track a small forward base velocity while staying upright and going
straight (lateral-velocity and yaw-rate are penalised, their commands are 0).
"""

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

# Absolute path to the converted TARS USD (URDF -> USD via convert_urdf.py).
GROWBOT_USD = "/home/adarshsosale/Workspace/Isaac RL Lab/isaac_lab/growbot.usd"

# ---------------------------------------------------------------------------
# Robot articulation. TPU feet -> grippy contact material is applied via the
# EventCfg below (the URDF carried no <material>, so the USD used PhysX
# defaults ~0.5, which made the feet slip — the user's #1 diagnosis).
# ---------------------------------------------------------------------------
GROWBOT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GROWBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.24),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=["hip_left", "hip_right"],
            effort_limit_sim=2.16,     # 2x stock MG996R torque (was 1.08) — drive bigger/faster leg swings
            velocity_limit_sim=6.16,
            stiffness=10.0,
            damping=0.3,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["ankle_left", "ankle_right"],
            effort_limit_sim=2.66,     # 2x stock MG996R torque (was 1.33) — stronger push-off / foot-flat hold
            velocity_limit_sim=5.0,
            stiffness=10.0,
            damping=0.3,
        ),
    },
)


@configclass
class EventCfg:
    """Startup randomization — chiefly the grippy TPU foot/ground material."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.2, 1.4),
            "dynamic_friction_range": (1.0, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


@configclass
class GrowbotFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0
    decimation = 4                 # 200 Hz physics -> 50 Hz control
    action_scale = 1.5             # rad offset from default stance (very wide ROM — freedom to take big leg swings)
    action_space = 4
    observation_space = 24
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.3,
            dynamic_friction=1.1,
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
            static_friction=1.3,
            dynamic_friction=1.1,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot + contact sensor
    robot: ArticulationCfg = GROWBOT_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # command ranges (straight-line: lateral & yaw commands are zero)
    cmd_vx_range = (0.25, 0.55)    # forward speed command [m/s] — pushed up: speed is a top priority

    # nominal upright base height [m]
    target_height = 0.235

    # reward scales
    # --- TOP PRIORITIES: speed + straight line ---
    lin_vel_reward_scale = 3
    forward_progress_reward_scale = 4.0       # linear: 0 when standing -> must actually move (cranked: speed first)
    yaw_rate_reward_scale = 1.0
    lateral_vel_reward_scale = -3.0
    yaw_rate_l2_reward_scale = -2.0       # directly damp turning
    lateral_pos_reward_scale = -5.0       # stay on the spawn x-axis (straight line)
    z_vel_reward_scale = -1.0
    ang_vel_reward_scale = -0.05
    flat_orientation_reward_scale = -5.0
    base_height_reward_scale = -10.0
    alive_reward_scale = 1.0
    joint_torque_reward_scale = -2.5e-5
    # --- jitter control: loosen blanket motion-MAGNITUDE penalties (they trap
    #     the robot in a tiny-shuffle vibration), keep the TARGETED guards that
    #     punish high-frequency reversals & sliding (the actual jitter signature) ---
    joint_accel_reward_scale = -2.0e-7        # loosened: don't suppress big leg swings
    joint_vel_reward_scale = -2.0e-5          # loosened: let the legs move fast through a real stride
    action_rate_reward_scale = -0.04          # KEEP firm: punishes vibrating/reversing commands = jitter
    # --- BEHAVIOURS TO REINFORCE: air time + bigger steps + single-support gait ---
    feet_air_time_reward_scale = 8          # strongly reward real, lifted swing steps
    feet_air_time_threshold = 0.25            # min step duration to count (a buzz can't accumulate this)
    foot_slip_reward_scale = -0.25            # KEEP firm: kill buzz-and-slide scuffing of the planted foot
    undesired_contact_reward_scale = -1.0
    # gait shaping
    single_stance_reward_scale = 1.0          # exactly one foot planted while moving -> alternating gait, kills the both-feet buzz
    gait_symmetry_reward_scale = -1.0         # small: keep L/R anti-phase mirror -> straight, coordinated
    step_stride_reward_scale = 3            # reward WIDE alternating hip excursion -> larger, deliberate steps
    ankle_usage_reward_scale = 4            # reward alternating ankle motion -> ankle keeps the planted foot flat / push-off
