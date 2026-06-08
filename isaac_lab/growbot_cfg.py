# growbot_cfg.py — Isaac Lab ArticulationCfg for the real TARS (Growbot) robot.
#
# Generated from growbot_cfg_TEMPLATE.py with the USD path filled in after the
# URDF -> USD conversion (scripts/tools/convert_urdf.py).
#
# Link (body) names : base_link, leg_left_link, leg_right_link,
#                     foot_left_link, foot_right_link
# Joint names       : hip_left, hip_right    (pitch, +-90 deg)
#                     ankle_left, ankle_right (pitch, +-49 deg)
# All joints pitch about the robot's lateral (URDF +Y) axis. +X = forward,
# +Z = up. base_link frame sits at hip height -> standing base height ~0.235 m.

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_HERE = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(_HERE, "growbot.usd")

GROWBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,   # legs/torso are laterally separated
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.235),               # ~standing; feet settle to z~0
        joint_pos={".*": 0.0},               # neutral stance
        joint_vel={".*": 0.0},
    ),
    actuators={
        # MG996R: ~1.08 N.m stall @6V, ~6.16 rad/s no-load.
        "hips": ImplicitActuatorCfg(
            joint_names_expr=["hip_left", "hip_right"],
            effort_limit=1.08,
            velocity_limit=6.16,
            stiffness=10.0,
            damping=0.3,
        ),
        # ankle pushrod gives ~1.23x torque advantage, ~1/1.23 the speed.
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["ankle_left", "ankle_right"],
            effort_limit=1.33,
            velocity_limit=5.0,
            stiffness=10.0,
            damping=0.3,
        ),
    },
)
