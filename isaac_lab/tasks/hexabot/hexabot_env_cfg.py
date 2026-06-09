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
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# The locomotion layer CONSUMES the frozen Navigation->Locomotion interface; it
# imports the canonical command ranges from the one place they are defined so the
# two layers can never disagree on the contract (Milestone-0 hard constraint #3).
from isaac_lab.interfaces import VX_RANGE, YAW_RANGE

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
        # damping 0.4 -> 0.5: light smoothing only. (1.0 over-resisted push-off and the
        # robot marched in place; the phase-clock gait reward now handles smoothness/jitter.)
        "coxa": ImplicitActuatorCfg(
            joint_names_expr=["coxa_.*"],
            effort_limit_sim=1.6, velocity_limit_sim=6.16,
            stiffness=10.0, damping=0.5,
        ),
        "femur": ImplicitActuatorCfg(
            joint_names_expr=["femur_.*"],
            effort_limit_sim=1.6, velocity_limit_sim=6.16,
            stiffness=12.0, damping=0.5,
        ),
        "tibia": ImplicitActuatorCfg(
            joint_names_expr=["tibia_.*"],
            effort_limit_sim=1.6, velocity_limit_sim=6.16,
            stiffness=12.0, damping=0.5,
        ),
    },
)


@configclass
class EventCfg:
    """Domain randomization applied by Isaac Lab's event manager.

    This is half of the DR story: the *physics-side* randomization (friction,
    mass, actuator gains) that needs sim hooks. The *signal-side* DR that lives in
    the env step loop (actuator latency, control-rate jitter, IMU/obs noise) is in
    `DomainRandCfg` below. All of it is ON from the start (Milestone-0 constraint
    #4) and every range is config-exposed here.
    """

    # grippy foot/ground contact material + friction spread (sim-to-real)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.3),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    # per-env body mass spread (payload / build tolerance)
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (0.85, 1.15),  # scale
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*coxa.*", ".*femur.*", ".*tibia.*"]),
            "mass_distribution_params": (0.9, 1.1),  # scale
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    # per-episode actuator-gain spread (servo unit-to-unit variation, wear, voltage)
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),  # scale of default stiffness
            "damping_distribution_params": (0.8, 1.2),    # scale of default damping
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class DomainRandCfg:
    """Signal-side domain randomization applied inside the env step loop.

    These have no Isaac event hook, so the env (`hexabot_env.py`) implements them.
    All ranges are config so later stages can widen them. Master switch `enabled`.
    """

    enabled: bool = True
    # actuator latency: the servo acts on a command delayed by k control steps,
    # k ~ randint(low, high) per env, held for the episode. Models comms + servo lag.
    actuator_latency_steps: tuple[int, int] = (0, 2)   # 0..2 steps @50 Hz = 0..40 ms
    # control-rate jitter: the effective control dt fed to the CPG wobbles, and with
    # `hold_prob` the previous command is reused (a dropped control tick).
    control_dt_jitter: float = 0.1                      # +/-10% on the CPG-advance dt
    hold_prob: float = 0.02                             # P(reuse last command this step)
    # IMU / observation Gaussian noise (std), added in _get_observations
    noise_gravity: float = 0.02                         # projected-gravity unit-vector noise
    noise_ang_vel: float = 0.10                         # rad/s
    noise_joint_pos: float = 0.01                       # rad (encoder)
    noise_joint_vel: float = 0.15                       # rad/s


@configclass
class HexabotFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 12.0
    decimation = 4                 # 200 Hz physics -> 50 Hz control
    # The policy action is NOT joint offsets — it is per-leg CPG modulation
    # [d_freq, d_coxa_amp, d_lift] x 6 legs = 18 (see cpg.py). Zero action == the
    # analytical tripod gait, so the analytical baseline is a *reference* the policy
    # can leave, never a residual added on top of its output (hard constraint #2).
    action_space = 18
    # Proprioceptive-ONLY observation — NO base linear velocity (not measurable on
    # the real robot, hard constraint). Layout (75):
    #   grav(3) ang_vel(3) cmd(3) jpos-def(18) jvel(18) prev_action(18) cpg_phase(12)
    # A dormant exteroceptive (height-scan) block of width `n_height_scan` (=0 now)
    # is appended LAST — the seam where a terrain encoder plugs in at a later
    # curriculum stage WITHOUT changing this shape. observation_space tracks it.
    n_height_scan = 0              # dormant exteroceptive slot width (stage 0: empty)
    observation_space = 75 + n_height_scan
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        # GPU contact buffers. Defaults (collision stack 2**26 = 64 MB) overflow on the
        # rough triangle-mesh terrain with 4096 envs spawn-settling at once -> PhysX
        # drops contacts (corrupts the contact sensor that feeds every gait reward).
        # The scene requests ~94 MB; 2**28 = 256 MB leaves headroom for rougher
        # curriculum rows. The found/lost-pairs bumps pre-empt the next buffer to
        # overflow once collisions are processed instead of dropped. Harmless on flat.
        physx=PhysxCfg(
            gpu_collision_stack_size=2**28,                 # 256 MB (was 64 MB)
            gpu_found_lost_pairs_capacity=2**23,            # was 2**21
            gpu_total_aggregate_pairs_capacity=2**23,       # was 2**21
        ),
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

    # events (physics-side DR) + signal-side DR
    events: EventCfg = EventCfg()
    domain_rand: DomainRandCfg = DomainRandCfg()

    # robot + contact sensor
    robot: ArticulationCfg = HEXABOT_CFG
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # command ranges from the FROZEN interface. The curriculum ramps the vx upper
    # bound; yaw turning is introduced after the straight-walking phase. vy stays 0.
    cmd_vx_range = VX_RANGE        # (0.0, 0.30) m/s
    cmd_yaw_range = YAW_RANGE      # (-0.5, 0.5) rad/s — turning (stage 1)

    # --- turning curriculum -------------------------------------------------
    # Learn to walk STRAIGHT first, then introduce yaw commands so turning is
    # built on a solid forward gait. Before yaw_curriculum_steps (after the vx
    # ramp) every command is straight (yaw=0); after it, a fraction of episodes
    # get a random yaw command (the rest stay straight to preserve straight-line
    # quality). Counted in global control steps.
    yaw_curriculum_start_steps = 6000   # ~250 iters: start turning once forward walking is solid
    yaw_curriculum_ramp_steps = 4000    # ramp the yaw command magnitude 0 -> full
    yaw_command_prob = 0.5              # fraction of episodes that get a (non-zero) yaw command
    cpg_yaw_ref = 0.5                   # yaw rate [rad/s] that fully activates the gait (for speed_scale)

    # --- stand-first curriculum ---------------------------------------------
    # The robot otherwise discovers a belly-crawl: lie flat and wiggle for a sliver
    # of forward reward at minimal effort/risk. We starve that optimum by making it
    # learn to STAND TALL before it is ever asked to move: command vx=0 for the
    # first `curriculum_stand_steps` control-steps, then ramp the upper vx command
    # from 0 to cmd_vx_range[1] over `curriculum_ramp_steps`. Counted in control
    # steps (~24 per training iteration).
    # The CPG now STRUCTURALLY prevents belly-crawl (at vx=0 the nominal stride
    # amplitude is gated to 0 -> the robot holds the standing stance), so the long
    # pure-stand phase the old direct-action policy needed is largely redundant.
    # A short stand phase still lets orientation/height settle before motion.
    curriculum_stand_steps = 500      # ~20 iters of pure standing
    curriculum_ramp_steps = 4000      # ~166 iters ramping 0 -> full speed
    cpg_v_ref = 0.30                  # command speed [m/s] at which the nominal CPG amplitude is full

    # --- CPG (central pattern generator) — the action mechanism ---------------
    # The policy modulates these per-leg oscillators; zero action == the analytical
    # alternating-tripod gait of hexabot_model/isaac/tripod_gait.py. See cpg.py.
    cpg_f_base = 1.0                  # nominal stride frequency [Hz] (1.0 s cycle)
    cpg_coxa_amp = 0.26               # nominal coxa sweep half-amplitude [rad] (tripod_gait default)
    cpg_lift = 0.55                   # nominal swing femur-lift amplitude [rad] (tripod_gait default)
    cpg_kf = 0.5                      # policy authority over per-leg frequency (+/-50%)
    cpg_kmu = 0.5                     # policy authority over per-leg coxa amplitude (+/-50%)
    cpg_klift = 0.6                   # policy authority over per-leg swing lift (+/-60%)
    cpg_coupling_strength = 2.0       # weak pull toward the tripod phase offsets (0 = none)

    # --- annealing imitation reward (reference, NOT residual) -----------------
    # Penalize the policy's joint targets deviating from the analytical-gait targets
    # at the current phase. Weight starts at imitation_w0 and ANNEALS to ~0 over the
    # flat-ground curriculum (tied to curriculum progress), so the policy ends up
    # competent but not tethered to a flat gait when terrain stages begin.
    imitation_reward_scale = -8.0
    imitation_anneal_steps = 8000     # control steps over which the weight decays 1 -> 0 (after the stand phase)

    # --- curriculum extensibility (stage 0 = flat, obstacle-free) -------------
    # These are INERT at stage 0; later stages turn them up WITHOUT changing the
    # policy's observation/action shapes (the dormant height-scan / lidar slots
    # absorb the new inputs). Wired now so the seam exists.
    obstacle_density = 0.0            # fraction of arena occupied by obstacles (0 = none)
    terrain_roughness = 0.0           # rough-terrain amplitude [m] (0 = flat plane)

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
    # exp velocity-tracking: SATURATING and farmable by standing. With small commands
    # (avg ~0.15 m/s) a frozen robot scores exp(-0.15^2/0.25)=0.91 of the max, so at the
    # old scale 2.0 this term alone paid ~1.8 -- the single biggest reward -- for standing
    # still and it drowned forward_progress (~0.15 when frozen). Demoted to a minor speed-
    # regulation term so the linear, standstill-zero forward_progress below is the dominant
    # translation driver.
    lin_vel_reward_scale = 0.5
    forward_progress_reward_scale = 12.0      # linear, un-saturated, 0 at standstill -> the real translation driver
    # --- anti-standstill MOTION GATE on the saturating exp trackers ------------
    # The two exp trackers below (track_lin_vel_xy_exp, track_ang_vel_z_exp) SATURATE
    # and pay a FROZEN robot nearly full marks: a straight episode commands yaw=0, a
    # standing robot has yaw_rate=0 -> exp(0)=1 -> it collects the WHOLE 1.5 of
    # track_ang for doing nothing (plus ~0.46 of track_lin). On flat the linear
    # forward_progress (+12) + stationary_penalty (-15) out-weigh that ~1.95 comfort
    # baseline so it walks. On ROUGH terrain, stepping forward risks a fall that ends
    # the episode and forfeits the future comfort stream, so that guaranteed
    # standstill comfort becomes the safe optimum and the policy collapses to standing
    # (the curriculum then demotes everyone back to flat and it stays stuck) — the
    # documented "saturating exp term re-creates the static stance" landmine, now
    # tripped by terrain death-risk. Gate BOTH exp trackers by ACTUAL motion so a
    # frozen robot earns ~0 from them while a robot that tracks its command still
    # earns the full bonus. The gate opens on real forward OR yaw speed (covers
    # turn-in-place) and is exactly 0 at standstill. At convergence a walking policy
    # (>=motion_gate_lin_ref) sees the gate fully open -> identical reward (no
    # regression on the known-good flat run); only the standstill freebie is removed.
    motion_gate_enabled = True
    motion_gate_lin_ref = 0.10        # body fwd speed [m/s] that fully opens the gate
    motion_gate_ang_ref = 0.30        # |yaw rate| [rad/s] that fully opens the gate (turn-in-place)
    # standstill-when-commanded penalty (the structural anti-static-stance fix). forward_progress
    # is a reward that's merely 0 at standstill, so a frozen tall stance still nets the large
    # command-independent comfort baseline (alive + foot_support + saturating trackers ~= +2.6) and
    # any always-on moving-tax flips the policy back to standing. This makes standstill ACTIVELY
    # COSTLY when commanded (slightly > forward_progress so being short is punished harder than
    # progress is paid), giving a wide walk-vs-stand margin that survives stability/smoothness shaping.
    stationary_penalty_reward_scale = -15.0
    yaw_rate_reward_scale = 1.5              # was 0.5: reward tracking the COMMANDED yaw rate (turning). Bumped so turning is worth learning.
    lateral_vel_reward_scale = -2.0          # gated OFF during turning episodes in the env (a curved path has body-y motion)
    yaw_rate_l2_reward_scale = -1.5          # now penalizes (yaw_rate - cmd_yaw)^2 (command-relative), so it damps drift without fighting commanded turns
    lateral_pos_reward_scale = -2.0           # stay on the spawn x-axis — STRAIGHT-line only; gated OFF during turning episodes in the env
    heading_reward_scale = -4.0               # hold +x heading — STRAIGHT-line only; gated OFF during turning episodes (else it would block turns)
    z_vel_reward_scale = -1.0
    ang_vel_reward_scale = -0.10              # was -0.05: GENTLE 2x to damp fore-aft base rocking. NB this is an
                                              # ALWAYS-ON penalty that taxes moving (a stride inherently pitches a
                                              # little) while a static stance pays 0 -> jumping straight to -0.25
                                              # plus other penalties re-created the static-stance optimum
                                              # (dx 3.8m -> 0.03m). Ramp this up only after confirming the walk holds.
    flat_orientation_reward_scale = -2.5      # keep at the tuned value (the -3.5 bump helped tip it back to standing)
    base_height_reward_scale = -8.0
    alive_reward_scale = 1.0                   # survival must clearly pay (was 0.5)
    # --- stand tall, don't belly-crawl ---
    belly_clearance_reward_scale = -50.0      # strong one-sided penalty: belly near ground
    foot_support_reward_scale = 10.0          # was 20.0: trim the standing scaffolding. This is a
                                              # command-independent reward a tall static tuck maxes out
                                              # (~+0.9), propping the static-stance optimum; belly_clearance
                                              # (-50, one-sided) already does the real anti-belly-crawl job.
    # --- effort / smoothness ---
    joint_torque_reward_scale = -2.0e-5
    joint_accel_reward_scale = -2.5e-7        # NB: doubling this to -5e-7 broke the stand phase (robot stopped correcting)
    action_rate_reward_scale = -0.017        # was -0.015: tiny nudge for smoother leg motion (penalize step-to-step
                                              # action jumps). Kept barely above the known-good value because this is
                                              # also an always-on moving-tax; -0.02 alongside the stability bumps
                                              # helped flip it back to standing. Held far below the -0.04 ceiling.
    joint_limit_reward_scale = -1.0           # discourage slamming joint limits
    # --- stepping behaviour ---
    feet_air_time_reward_scale = 2.5          # was 1.0: longer swings -> slower cadence, bigger strides
    feet_air_time_threshold = 0.2             # min step duration to count [s]
    foot_slip_reward_scale = -0.2             # gentle bump from -0.1 (-0.4 always-on slowed standing); foot_plant reward handles grip
    undesired_contact_reward_scale = -1.0     # body/coxa/femur must not touch ground
    # --- coordinated symmetric tetrapod gait (motion-gated) ---
    tetrapod_contact_reward_scale = 3.0       # was 1.5: pin exactly 4 of 6 feet planted. The robot
                                              # had abandoned the stance (tetrapod 0.82->0.17), teetering
                                              # on tucked legs; strengthen alongside the gait_phase
                                              # stance-violation penalty to rebuild the planted tetrapod.
    gait_symmetry_reward_scale = 1.5          # the planted/lifted feet form a left-right mirror
    foot_clearance_reward_scale = 4.0         # big deliberate swing lifts (anti-skitter)
    foot_clearance_target = 0.025             # swing-foot apex height rewarded up to [m] (0.04 too high -> unstable)
    # --- phase-clock periodic gait (now driven by the CPG, not a separate clock) ---
    # The CPG (cpg.py) OWNS the gait phase and frequency; this reward checks that the
    # feet actually lift/plant when the CPG schedules them. A leg's CPG local phase
    # p>=0.5 is its swing half (matches tripod_gait.gait_pose). Still 0 for a static
    # stance (no scheduled-swing feet lifted, no stance violations).
    gait_phase_reward_scale = 3.0             # net term: +correctly LIFTING scheduled-swing feet, -lifting
                                              # scheduled-stance feet (over-lifting).
    # foot plant angle: stance feet point steeply down so the claw digs in (anti-slip)
    foot_plant_reward_scale = 0.6             # reward steep stance feet
    foot_plant_target = 0.85                  # sin(angle below horizontal) target ~= 58-65 deg
