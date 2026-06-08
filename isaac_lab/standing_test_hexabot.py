"""Passive standing sanity test for the Hexabot hexapod.

NO policy, NO RL. Spawns one robot, holds the standing stance (default_joint_pos)
the whole time, and reports whether a passive hexapod settles and STAYS standing.
If a passive robot can't hold itself up above the death thresholds, RL has no
chance — fix the asset/actuators/stance before training.

Checks against the exact termination thresholds used in hexabot_env._get_dones:
    too_low : base-centre z < 0.035 m
    tilted  : projected_gravity_b z > -0.5  (tilted past ~60 deg)

Usage (via isaaclab.sh -p):
    standing_test_hexabot.py [--seconds 3.0]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=3.0, help="how long to hold the stance [s]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- rest follows --
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg

# reuse the exact robot articulation cfg used for training
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", "hexabot"),
)
from hexabot_env_cfg import HEXABOT_CFG, HexabotFlatEnvCfg  # noqa: E402

CFG = HexabotFlatEnvCfg()
TARGET_HEIGHT = CFG.target_height        # 0.072
TOO_LOW = 0.035                          # death threshold from _get_dones
TILT_GZ = -0.5                           # projected_gravity_b z death threshold


def main():
    phys_hz = 200.0
    dt = 1.0 / phys_hz
    sim = SimulationContext(SimulationCfg(dt=dt, device="cuda:0"))

    # grippy contact material on ground + robot (matches training friction)
    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.1, dynamic_friction=0.9, restitution=0.0,
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
    )
    sim_utils.GroundPlaneCfg(physics_material=mat).func(
        "/World/ground", sim_utils.GroundPlaneCfg(physics_material=mat)
    )
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    robot = Articulation(HEXABOT_CFG.replace(prim_path="/World/Robot"))
    # bind grippy material to the robot collisions too (matches training)
    mat.func("/World/robotMaterial", mat)
    sim_utils.bind_physics_material("/World/Robot", "/World/robotMaterial")

    sim.reset()

    default_q = robot.data.default_joint_pos.clone()
    print("[INFO] joints:", robot.joint_names, flush=True)
    print(f"[INFO] spawn height = {robot.data.root_pos_w[0, 2].item():.4f} m", flush=True)
    print(f"[INFO] target standing height = {TARGET_HEIGHT:.4f} m", flush=True)
    print(f"[INFO] death thresholds: z < {TOO_LOW} m  OR  proj_grav_z > {TILT_GZ}", flush=True)
    print("", flush=True)

    n_steps = int(args_cli.seconds * phys_hz)
    sample_every = int(0.1 * phys_hz)   # log every 0.1 s
    # settle window: the first ~0.3 s is an unavoidable spawn-settling transient
    # (the PD legs must deflect to build holding torque, so the body briefly sags).
    # Training ignores deaths during this window (cfg.settle_steps); the verdict
    # below judges the robot only AFTER it, matching training reality.
    settle_t = CFG.settle_steps / 50.0   # settle_steps are control steps @ 50 Hz
    settle_phys = int(settle_t * phys_hz)

    min_h = float("inf")               # min height during the settle transient (info only)
    min_h_post = float("inf")          # min height AFTER the settle window (verdict)
    min_gz = 0.0      # most-tilted (closest to 0 / positive) projected-gravity z seen
    ever_died = False                  # crossed a threshold during the settle transient
    died_post = False                  # crossed a threshold AFTER the settle window
    first_death_t = None

    print(f"{'t [s]':>7} {'height [m]':>11} {'proj_grav_z':>12} {'status':>8}", flush=True)
    for i in range(n_steps):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)

        h = robot.data.root_pos_w[0, 2].item()
        gz = robot.data.projected_gravity_b[0, 2].item()
        min_h = min(min_h, h)
        min_gz = max(min_gz, gz)   # gz is ~-1 upright; larger (toward 0/+) = more tilted

        dead = (h < TOO_LOW) or (gz > TILT_GZ)
        if dead and not ever_died:
            ever_died = True
            first_death_t = i / phys_hz
        if i >= settle_phys:
            min_h_post = min(min_h_post, h)
            if dead:
                died_post = True

        if i % sample_every == 0 or i == n_steps - 1:
            status = "DEAD" if dead else "ok"
            print(f"{i / phys_hz:7.2f} {h:11.4f} {gz:12.4f} {status:>8}", flush=True)

    # final summary
    h_final = robot.data.root_pos_w[0, 2].item()
    gz_final = robot.data.projected_gravity_b[0, 2].item()
    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"  final height        : {h_final:.4f} m   (target {TARGET_HEIGHT:.3f}, death < {TOO_LOW})", flush=True)
    print(f"  min height (transient): {min_h:.4f} m   (first {settle_t:.2f}s spawn settle — ignored)", flush=True)
    print(f"  min height (post-settle): {min_h_post:.4f} m   (verdict window)", flush=True)
    print(f"  final proj_grav_z   : {gz_final:.4f}   (upright -1.0, death > {TILT_GZ})", flush=True)
    print(f"  most-tilted gz      : {min_gz:.4f}", flush=True)
    if ever_died:
        print(f"  [note] dipped below a threshold during the spawn transient at t={first_death_t:.2f}s", flush=True)
        print(f"         (recovered; training's {CFG.settle_steps}-step settle grace covers this).", flush=True)
    if died_post:
        print(f"  RESULT: FAIL — robot crossed a death threshold AFTER the {settle_t:.2f}s settle window.", flush=True)
        print("          A passive hexapod cannot hold its own stance. Fix the asset /", flush=True)
        print("          actuator stiffness / stance BEFORE attempting RL.", flush=True)
    elif h_final < TARGET_HEIGHT - 0.02:
        print(f"  RESULT: MARGINAL — survives but sags {(TARGET_HEIGHT - h_final) * 1000:.0f} mm below target.", flush=True)
        print("          Stands, but the legs are soft (low stiffness). RL may still work.", flush=True)
    else:
        print("  RESULT: PASS — passive robot stands stably at ~target height after settling.", flush=True)
        print("          The hardware can stand; the training collapse was the spawn-transient", flush=True)
        print("          death + last-checkpoint export, not the robot itself.", flush=True)
    print("=" * 60, flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
