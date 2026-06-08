"""Record a video of the TARS (Growbot) robot's simulated locomotion.

The robot has only 4 sagittal pitch DOFs (hip_left/right, ankle_left/right),
no knee and no lateral/roll DOF, so it can only locomote in the sagittal plane.
This script evaluates several coupled-sinusoid CPG gaits *without rendering*
(fast), scores each on forward travel while staying upright, then renders the
best one to an mp4. Headless.

Usage (via isaaclab.sh -p):
    record_locomotion.py [--seconds 10] [--fps 30] [--out out.mp4]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--out", type=str, default="/home/adarshsosale/Workspace/Isaac RL Lab/isaac_lab/tars_locomotion.mp4")
parser.add_argument("--settle", type=float, default=1.0, help="settle seconds before gait")
parser.add_argument("--eval_seconds", type=float, default=6.0, help="duration per candidate during search")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -- rest follows --
import math
import os
import sys

import numpy as np
import torch
import imageio.v2 as imageio

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext, SimulationCfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from growbot_cfg import GROWBOT_CFG


# Candidate gaits. leg_antiphase=1 -> legs swing in antiphase; 0 -> in phase.
# ankle_phase = ankle lag vs hip. All amplitudes in rad, freq in Hz.
CANDIDATES = [
    # antiphase rowing gait (one foot always grounded) — search the window between
    # "doesn't translate" (small amp) and "topples forward" (large amp / wrong phase).
    dict(name="row_a", freq=1.0, hip_amp=0.25, ank_amp=0.30, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.05),
    dict(name="row_b", freq=1.0, hip_amp=0.30, ank_amp=0.30, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.05),
    dict(name="row_c", freq=1.2, hip_amp=0.30, ank_amp=0.35, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.05),
    dict(name="row_d", freq=1.0, hip_amp=0.35, ank_amp=0.35, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.10),
    dict(name="row_e", freq=1.5, hip_amp=0.30, ank_amp=0.30, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.05),
    dict(name="row_f", freq=1.0, hip_amp=0.25, ank_amp=0.30, ank_phase=+0.785, antiphase=1, hip_bias=0.0, ank_bias=0.05),
    dict(name="row_g", freq=1.0, hip_amp=0.30, ank_amp=0.30, ank_phase=+1.57, antiphase=1, hip_bias=-0.10, ank_bias=0.05),
    dict(name="row_h", freq=1.0, hip_amp=0.30, ank_amp=0.40, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.15),
    dict(name="row_i", freq=1.3, hip_amp=0.35, ank_amp=0.30, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.05),
    dict(name="row_j", freq=1.1, hip_amp=0.28, ank_amp=0.32, ank_phase=+1.20, antiphase=1, hip_bias=-0.05, ank_bias=0.08),
    dict(name="row_k", freq=0.9, hip_amp=0.40, ank_amp=0.35, ank_phase=+1.57, antiphase=1, hip_bias=0.0,  ank_bias=0.10),
    dict(name="row_l", freq=1.0, hip_amp=0.30, ank_amp=0.25, ank_phase=+2.00, antiphase=1, hip_bias=0.0,  ank_bias=0.05),
    # known-upright baselines for reference
    dict(name="base_up", freq=1.0, hip_amp=0.20, ank_amp=0.25, ank_phase=+1.57, antiphase=1, hip_bias=0.0, ank_bias=0.0),
]


def main():
    phys_hz = 200.0
    dt = 1.0 / phys_hz
    sim = SimulationContext(SimulationCfg(dt=dt, device="cuda:0"))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0))
    )
    sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0),
        translation=(2.0, 2.0, 4.0), orientation=(0.86, 0.32, -0.32, 0.0),
    )

    robot = Articulation(GROWBOT_CFG.replace(prim_path="/World/Robot"))
    camera = Camera(CameraCfg(
        prim_path="/World/Camera", update_period=0.0, height=720, width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4),
        ),
    ))

    sim.reset()
    print("[INFO] bodies:", robot.body_names, flush=True)
    print("[INFO] joints:", robot.joint_names, flush=True)

    jn = robot.joint_names
    iLh, iRh = jn.index("hip_left"), jn.index("hip_right")
    iLa, iRa = jn.index("ankle_left"), jn.index("ankle_right")
    default_q = robot.data.default_joint_pos.clone()
    default_root = robot.data.default_root_state.clone()

    settle_steps = int(args_cli.settle * phys_hz)
    gait_steps = int(args_cli.seconds * phys_hz)
    eval_steps = int(args_cli.eval_seconds * phys_hz)
    decim = max(1, round(phys_hz / args_cli.fps))
    eff_fps = phys_hz / decim

    def reset_robot():
        robot.write_root_pose_to_sim(default_root[:, :7])
        robot.write_root_velocity_to_sim(default_root[:, 7:])
        robot.write_joint_state_to_sim(default_q.clone(), torch.zeros_like(default_q))
        robot.reset()
        for _ in range(settle_steps):
            robot.set_joint_position_target(default_q)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)

    def gait_targets(g, t):
        ph = 2.0 * math.pi * g["freq"] * t
        legoff = math.pi if g["antiphase"] else 0.0
        q = default_q.clone()
        q[0, iLh] = g["hip_bias"] + g["hip_amp"] * math.sin(ph)
        q[0, iRh] = g["hip_bias"] + g["hip_amp"] * math.sin(ph + legoff)
        q[0, iLa] = g["ank_bias"] + g["ank_amp"] * math.sin(ph + g["ank_phase"])
        q[0, iRa] = g["ank_bias"] + g["ank_amp"] * math.sin(ph + legoff + g["ank_phase"])
        return q

    # ---------- evaluate candidates (no rendering) ----------
    print("\n[EVAL] scoring gaits (upright = min_h>0.13 and final_h>0.17)\n", flush=True)
    results = []
    for g in CANDIDATES:
        reset_robot()
        x0 = robot.data.root_pos_w[0, 0].item()
        y0 = robot.data.root_pos_w[0, 1].item()
        min_h = 1e9
        for k in range(eval_steps):
            robot.set_joint_position_target(gait_targets(g, k * dt))
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
            min_h = min(min_h, robot.data.root_pos_w[0, 2].item())
        xf = robot.data.root_pos_w[0, 0].item()
        yf = robot.data.root_pos_w[0, 1].item()
        hf = robot.data.root_pos_w[0, 2].item()
        dx = xf - x0
        dy = yf - y0
        upright = (min_h > 0.13) and (hf > 0.17)
        dist = abs(dx)
        score = dist if upright else -1.0 + 0.001 * dist
        results.append((score, dist, dx, dy, hf, min_h, upright, g))
        print(f"  {g['name']:14s} dx={dx:+.3f} dy={dy:+.3f} final_h={hf:.3f} min_h={min_h:.3f} "
              f"upright={'Y' if upright else 'n'} score={score:+.3f}", flush=True)

    results.sort(key=lambda r: r[0], reverse=True)
    best = results[0]
    g = best[7]
    print(f"\n[BEST] {g['name']}  dx={best[2]:+.3f} m  final_h={best[4]:.3f}  upright={best[6]}\n", flush=True)
    if not best[6]:
        print("[WARN] no fully-upright gait found; rendering best-effort (most forward travel).", flush=True)

    # ---------- render the winner ----------
    reset_robot()
    frames = []
    for k in range(gait_steps):
        robot.set_joint_position_target(gait_targets(g, k * dt))
        robot.write_data_to_sim()
        capture = (k % decim == 0)
        sim.step(render=capture)
        robot.update(dt)
        if capture:
            fx = robot.data.root_pos_w[0, 0].item()
            eye = torch.tensor([[fx - 1.1, -1.7, 0.9]], device=sim.device)
            tgt = torch.tensor([[fx + 0.1, 0.0, 0.18]], device=sim.device)
            camera.set_world_poses_from_view(eye, tgt)
            sim.render()
            camera.update(dt)
            rgb = camera.data.output["rgb"][0, ..., :3].clone().cpu().numpy().astype(np.uint8)
            frames.append(rgb)

    print(f"[INFO] rendered {len(frames)} frames @ {eff_fps:.1f} fps", flush=True)
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    imageio.mimsave(args_cli.out, frames, fps=eff_fps, quality=8, macro_block_size=8)
    print(f"[INFO] wrote {args_cli.out}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
