# growbot_cfg_TEMPLATE.py — Isaac Lab ArticulationCfg for Growbot.
#
# THIS IS A TEMPLATE.  It cannot run on this Mac (Isaac Sim needs an NVIDIA RTX
# GPU on Linux/Windows).  Copy it into your Isaac Lab project on the GPU box,
# fix `usd_path`, and import GROWBOT_CFG from your env.
#
# Import paths below use the current `isaaclab.*` namespace.  On Isaac Lab 1.x
# the package was `omni.isaac.lab.*` — adjust if your version is older.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ----------------------------------------------------------------------------
# Link (body) names : base_link, leg_left_link, leg_right_link,
#                     foot_left_link, foot_right_link
# Joint names       : hip_left, hip_right  (pitch, +-90 deg)
#                     ankle_left, ankle_right (pitch, +-49 deg)
# All joints pitch about the robot's lateral (URDF +Y) axis.  +X = forward,
# +Z = up.  base_link frame sits at hip height -> standing base height ~0.23 m.
# ----------------------------------------------------------------------------

GROWBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/ABSOLUTE/PATH/TO/growbot.usd",   # <-- set after URDF->USD
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
        # Stiffness ~ stall_torque / 0.1 rad error => ~10 N.m/rad (full torque
        # at a 6 deg tracking error).  Damping ~ 2*sqrt(K*I) for the leg inertia
        # (~1.4e-3) => ~0.25.  START HERE, then TUNE for your controller rate.
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

# Reminder: the real ankle is a crank->pushrod->rocker 4-bar (a CLOSED loop).
# A PhysX articulation is a TREE, so the ankle is modeled as ONE revolute joint
# (foot about the hinge pin).  The +-0.855 rad limit already encodes the
# pushrod's reachable foot range.  The crank/pushrod/rocker meshes are VISUAL
# ONLY and will appear to separate as the ankle moves — physics is correct.
