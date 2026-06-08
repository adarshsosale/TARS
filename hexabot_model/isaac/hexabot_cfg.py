# hexabot_cfg.py — Isaac Lab ArticulationCfg for the 18-DOF hexapod.
#
# THIS IS A TEMPLATE.  It cannot run on a Mac (Isaac Sim needs an NVIDIA RTX GPU
# on Linux/Windows).  Copy it into your Isaac Lab project on the GPU box, set
# `usd_path` after converting hexabot.urdf -> hexabot.usd, and import HEXABOT_CFG.
#
# Import paths use the current `isaaclab.*` namespace.  On Isaac Lab 1.x the
# package was `omni.isaac.lab.*` — adjust if your version is older.
#
# ---------------------------------------------------------------------------
# Links (19): base_link + {coxa,femur,tibia}_{lf,lm,lr,rf,rm,rr}
# Joints (18, all revolute):
#   coxa_*   — yaw about +Z (vertical): swings the whole leg fore/aft
#   femur_*  — pitch about the leg's tangent (+Y in the leg frame): lifts the leg
#   tibia_*  — pitch about the same tangent: knee flex (drives the claw)
# Frame: metres, +X forward, +Y left, +Z up.  base_link at the body centre.
# Standing body-centre height ≈ 0.072 m (feet settle to z≈0).
# Tripod groups: A = {lf, rm, lr}, B = {rf, lm, rr}.
# ---------------------------------------------------------------------------

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Standing stance baked into generate_hexabot.py (STANCE_* params), in radians.
_STANCE_FEMUR = math.radians(-18.0)   # knee carried slightly high
_STANCE_TIBIA = math.radians(64.0)    # tibia reaches down to plant the claw

HEXABOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/ABSOLUTE/PATH/TO/hexabot.usd",   # <-- set after URDF->USD
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
        # spawn a touch above the 72 mm standing height so the feet settle, not clip
        pos=(0.0, 0.0, 0.085),
        joint_pos={
            "coxa_.*": 0.0,
            "femur_.*": _STANCE_FEMUR,
            "tibia_.*": _STANCE_TIBIA,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # MG996R: ~1.08 N·m stall @6V, ~6.16 rad/s no-load.  Stiffness ~ stall /
        # 0.1 rad error => ~10 N·m/rad; damping ~2·sqrt(K·I) for the small leg
        # inertias.  START HERE, then tune for your control rate.  One group per
        # joint role so you can tune coxa(swing) / femur(lift) / tibia(knee) apart.
        "coxa": ImplicitActuatorCfg(
            joint_names_expr=["coxa_.*"],
            effort_limit=1.08, velocity_limit=6.16,
            stiffness=10.0, damping=0.4,
        ),
        "femur": ImplicitActuatorCfg(
            joint_names_expr=["femur_.*"],
            effort_limit=1.08, velocity_limit=6.16,
            stiffness=12.0, damping=0.4,
        ),
        "tibia": ImplicitActuatorCfg(
            joint_names_expr=["tibia_.*"],
            effort_limit=1.08, velocity_limit=6.16,
            stiffness=12.0, damping=0.4,
        ),
    },
)

# Handy groupings for gait/RL code (joint order is up to your task to resolve by name).
TRIPOD_A = ["lf", "rm", "lr"]
TRIPOD_B = ["rf", "lm", "rr"]
LEGS = ["lf", "lm", "lr", "rf", "rm", "rr"]

# Why a hexapod beats the TARS biped here: 18 DOF (3/leg) vs 4, and at least one
# tripod of 3 feet is always planted -> STATICALLY stable (no inverted-pendulum
# balancing).  A simple open-loop gait already walks (see tripod_gait.py); RL is
# optional polish, not a prerequisite for not-falling-over.
