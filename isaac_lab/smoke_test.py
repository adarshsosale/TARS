"""Smoke test: load the TARS (Growbot) USD as an Isaac Lab Articulation,
verify the articulation root + joints, settle under gravity, and nudge each
joint to confirm it responds. Headless."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- rest follows --
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from growbot_cfg import GROWBOT_CFG


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0, device="cuda:0"))
    sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.2))

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=2500.0))

    robot_cfg = GROWBOT_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    print("\n================ ARTICULATION REPORT ================")
    print("num_instances     :", robot.num_instances)
    print("num_bodies        :", robot.num_bodies)
    print("body_names        :", robot.body_names)
    print("num_joints        :", robot.num_joints)
    print("joint_names       :", robot.joint_names)
    print("is_initialized    :", robot.is_initialized)
    root_pos = robot.data.root_pos_w[0].tolist()
    print("init root pos (w) :", [round(x, 4) for x in root_pos])

    # settle under gravity for ~1.5 s holding neutral stance
    default_q = robot.data.default_joint_pos.clone()
    for _ in range(300):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(sim.get_physics_dt())
    settled = robot.data.root_pos_w[0].tolist()
    print("settled root pos  :", [round(x, 4) for x in settled])

    # nudge each joint and confirm it moves
    print("\n---- per-joint response (target +0.4 rad) ----")
    for j, name in enumerate(robot.joint_names):
        tgt = default_q.clone()
        tgt[0, j] = 0.4
        for _ in range(120):
            robot.set_joint_position_target(tgt)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(sim.get_physics_dt())
        reached = robot.data.joint_pos[0, j].item()
        ok = "OK" if abs(reached - 0.4) < 0.25 else "??"
        print(f"  [{ok}] {name:12s} reached {reached:+.3f} rad")

    print("====================================================\n")
    simulation_app.close()


if __name__ == "__main__":
    main()
