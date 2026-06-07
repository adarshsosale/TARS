"""Record a 10s video of the TARS (Growbot) robot walking with the TRAINED PPO policy.

Loads the TorchScript policy exported by play.py (logs/.../exported/policy.pt),
runs a single robot on flat ground with TPU-grippy contact friction (matching
training), commands a fixed forward velocity, and records a tracking-camera mp4.

Usage (via isaaclab.sh -p):
    record_policy.py --policy <path/to/exported/policy.pt> [--seconds 10] [--cmd_vx 0.3]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, required=True, help="path to exported TorchScript policy.pt")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--cmd_vx", type=float, default=0.3, help="commanded forward velocity [m/s]")
parser.add_argument("--out", type=str, default="/home/adarshsosale/Workspace/Isaac RL Lab/isaac_lab/tars_locomotion.mp4")
parser.add_argument("--settle", type=float, default=0.5)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- rest follows --
import os
import sys

import numpy as np
import torch
import imageio.v2 as imageio

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext, SimulationCfg

# reuse the exact robot articulation cfg used for training
sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks", "growbot"),
)
from growbot_env_cfg import GROWBOT_CFG, GrowbotFlatEnvCfg  # noqa: E402

CFG = GrowbotFlatEnvCfg()
ACTION_SCALE = CFG.action_scale          # 0.6
TARGET_HEIGHT = CFG.target_height
DECIM = CFG.decimation                   # 4 -> 50 Hz control


def main():
    phys_hz = 200.0
    dt = 1.0 / phys_hz
    sim = SimulationContext(SimulationCfg(dt=dt, device="cuda:0"))

    # grippy TPU-like contact material on ground + robot (matches training friction)
    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.3, dynamic_friction=1.1, restitution=0.0,
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
    )
    sim_utils.GroundPlaneCfg(physics_material=mat).func(
        "/World/ground", sim_utils.GroundPlaneCfg(physics_material=mat)
    )
    sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0))
    )
    sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0),
        translation=(2.0, 2.0, 4.0), orientation=(0.86, 0.32, -0.32, 0.0),
    )

    robot = Articulation(GROWBOT_CFG.replace(prim_path="/World/Robot"))
    # bind grippy material to the robot collisions too
    mat.func("/World/robotMaterial", mat)
    sim_utils.bind_physics_material("/World/Robot", "/World/robotMaterial")

    camera = Camera(CameraCfg(
        prim_path="/World/Camera", update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4),
        ),
    ))

    sim.reset()

    policy = torch.jit.load(args_cli.policy, map_location=sim.device).eval()
    print(f"[INFO] loaded policy: {args_cli.policy}", flush=True)
    print("[INFO] joints:", robot.joint_names, flush=True)

    default_q = robot.data.default_joint_pos.clone()
    command = torch.tensor([[args_cli.cmd_vx, 0.0, 0.0]], device=sim.device)
    last_action = torch.zeros(1, 4, device=sim.device)

    def build_obs():
        return torch.cat(
            [
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                command,
                robot.data.joint_pos - robot.data.default_joint_pos,
                robot.data.joint_vel,
                last_action,
            ],
            dim=-1,
        )

    # settle to neutral stance
    for _ in range(int(args_cli.settle * phys_hz)):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
    x0 = robot.data.root_pos_w[0, 0].item()

    n_control = int(args_cli.seconds * phys_hz / DECIM)   # 50 Hz control steps
    frames = []
    for c in range(n_control):
        with torch.inference_mode():
            action = policy(build_obs())
        last_action = action.clone()
        target = ACTION_SCALE * action + default_q
        for _ in range(DECIM):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
        # capture one frame per control step (-> 50 fps)
        fx = robot.data.root_pos_w[0, 0].item()
        fy = robot.data.root_pos_w[0, 1].item()
        eye = torch.tensor([[fx - 1.1, fy - 1.7, 0.9]], device=sim.device)
        tgt = torch.tensor([[fx + 0.1, fy, 0.18]], device=sim.device)
        camera.set_world_poses_from_view(eye, tgt)
        sim.render()
        camera.update(dt)
        rgb = camera.data.output["rgb"][0, ..., :3].clone().cpu().numpy().astype(np.uint8)
        frames.append(rgb)

    xf = robot.data.root_pos_w[0, 0].item()
    yf = robot.data.root_pos_w[0, 1].item()
    hf = robot.data.root_pos_w[0, 2].item()
    print(f"[INFO] forward travel dx={xf - x0:+.3f} m  dy={yf:+.3f} m  final_h={hf:.3f} m", flush=True)
    print(f"[INFO] frames={len(frames)} fps={phys_hz / DECIM:.0f}", flush=True)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    imageio.mimsave(args_cli.out, frames, fps=phys_hz / DECIM, quality=8, macro_block_size=8)
    print(f"[INFO] wrote {args_cli.out}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
