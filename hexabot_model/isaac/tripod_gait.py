#!/usr/bin/env python3
"""
tripod_gait.py — open-loop alternating-tripod gait demo for the hexabot.

Runs on the GPU box (Isaac Sim / Isaac Lab; NOT on a Mac).  It spawns the robot
on a ground plane and drives a simple sinusoidal tripod gait via joint POSITION
targets — no RL.  PhysX + ground friction turn the coxa sweep into real forward
(+X) walking, so this is the apples-to-apples "does it walk?" demo vs TARS.

This is the same gait the local preview animates (previews/render_hexabot.py),
so what you saw on the Mac is what should happen here — but now with contact,
gravity and actuator dynamics actually in the loop.

Usage (from your Isaac Lab repo root, after URDF->USD — see HEXABOT.md):
    ./isaaclab.sh -p /path/to/hexabot_model/isaac/tripod_gait.py \
        --usd /path/to/hexabot.usd
    # add --headless to run without a window; --period / --stride to tune the gait
"""

import argparse, math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hexabot open-loop tripod gait demo.")
parser.add_argument("--usd", type=str, default=None,
                    help="absolute path to hexabot.usd (converted from hexabot.urdf). "
                         "If omitted, set usd_path in hexabot_cfg.py and pass --use-cfg.")
parser.add_argument("--use-cfg", action="store_true",
                    help="import HEXABOT_CFG from hexabot_cfg.py instead of --usd.")
parser.add_argument("--period", type=float, default=1.6, help="gait cycle period (s).")
parser.add_argument("--coxa-amp", type=float, default=0.26, help="coxa sweep half-amplitude (rad).")
parser.add_argument("--lift", type=float, default=0.55, help="femur swing lift (rad).")
parser.add_argument("--settle", type=float, default=1.0, help="seconds to settle into stance first.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- everything below must be imported AFTER the app launches ----------------
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext

# ----- model constants (kept in sync with generate_hexabot.py) ---------------
LEG_AZ = {"lf": 30.0, "lm": 90.0, "lr": 150.0, "rf": -30.0, "rm": -90.0, "rr": -150.0}
TRIPOD_A = {"lf", "rm", "lr"}                      # B = the other three
STANCE_FEMUR = math.radians(-18.0)
STANCE_TIBIA = math.radians(64.0)


def gait_pose(phase, coxa_amp, lift):
    """phase in [0,1) -> {leg: (qc, qf, qt)} — one alternating-tripod cycle.

    Each leg spends half the cycle in STANCE (foot planted, coxa sweeps to push
    the body +X) and half in SWING (femur/tibia lift, coxa returns).  The coxa
    sweep is scaled by sin(azimuth) so every leg pushes the body the same way."""
    pose = {}
    for leg, az in LEG_AZ.items():
        th = math.radians(az)
        ph = phase if leg in TRIPOD_A else (phase + 0.5) % 1.0
        if ph < 0.5:                                  # STANCE
            s = ph / 0.5
            qc = coxa_amp * math.sin(th) * (2 * s - 1)
            qf, qt = STANCE_FEMUR, STANCE_TIBIA
        else:                                         # SWING
            s = (ph - 0.5) / 0.5
            qc = coxa_amp * math.sin(th) * (1 - 2 * s)
            k = math.sin(math.pi * s)
            qf = STANCE_FEMUR - lift * k
            qt = STANCE_TIBIA + 0.4 * lift * k
        pose[leg] = (qc, qf, qt)
    return pose


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 240.0, device=args.device))
    sim.set_camera_view(eye=[0.9, 0.9, 0.6], target=[0.2, 0.0, 0.05])

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    # robot — either the full cfg (preferred) or straight from a USD path
    if args.use_cfg:
        from hexabot_cfg import HEXABOT_CFG
        robot_cfg = HEXABOT_CFG.replace(prim_path="/World/Robot")
    else:
        assert args.usd, "pass --usd /path/to/hexabot.usd  (or --use-cfg)"
        from isaaclab.actuators import ImplicitActuatorCfg
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=args.usd, activate_contact_sensors=True),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.085),
                joint_pos={"coxa_.*": 0.0, "femur_.*": STANCE_FEMUR, "tibia_.*": STANCE_TIBIA},
            ),
            actuators={
                "legs": ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit=1.08,
                                            velocity_limit=6.16, stiffness=12.0, damping=0.4),
            },
        )

    robot = Articulation(robot_cfg)
    sim.reset()

    # map leg -> (coxa_idx, femur_idx, tibia_idx) by joint name
    names = robot.joint_names
    idx = {}
    for leg in LEG_AZ:
        idx[leg] = (names.index(f"coxa_{leg}"), names.index(f"femur_{leg}"), names.index(f"tibia_{leg}"))

    # start from the stance default
    targets = robot.data.default_joint_pos.clone()
    dt = sim.get_physics_dt()
    settle_steps = int(args.settle / dt)
    t = 0.0
    step = 0
    print(f"[hexabot] {len(names)} joints; walking +X, period {args.period}s. Ctrl-C to stop.")

    while simulation_app.is_running():
        if step >= settle_steps:                       # begin walking after settling
            t += dt
            phase = (t / args.period) % 1.0
            pose = gait_pose(phase, args.coxa_amp, args.lift)
            for leg, (qc, qf, qt) in pose.items():
                ci, fi, ti = idx[leg]
                targets[:, ci] = qc
                targets[:, fi] = qf
                targets[:, ti] = qt

        robot.set_joint_position_target(targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        step += 1

        if step % 240 == 0:                            # ~1 Hz progress print
            x = float(robot.data.root_pos_w[0, 0])
            print(f"  t={step*dt:5.1f}s   base x = {x*1000:7.1f} mm   (forward progress)")


if __name__ == "__main__":
    main()
    simulation_app.close()
