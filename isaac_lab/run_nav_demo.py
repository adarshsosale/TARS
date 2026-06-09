"""End-to-end TWO-LAYER navigation demo for the Hexabot (flat ground).

Exercises the full stack over the FROZEN interface:

    goal  --[Navigation: go_to_goal]-->  (vx,vy,yaw)  --[Locomotion: PPO+CPG]-->  joints

The Navigation layer here is the hand-coded `go_to_goal` controller (Milestone-0
placeholder); the Locomotion layer is the trained PPO policy exported by
`play_hexabot.py`. The two layers communicate ONLY through `VelocityCommand`. A
slower navigation tick produces the command; the locomotion policy tracks it at
50 Hz. Records a tracking-camera mp4 and reports whether the goal was reached.

Usage (via isaaclab.sh -p):
    run_nav_demo.py --policy <exported/policy.pt> [--goal_x 2.0 --goal_y 0.5] [--seconds 20]
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", type=str, required=True, help="path to exported locomotion policy.pt")
parser.add_argument("--nav_policy", type=str, default=None,
                    help="path to exported NAVIGATION policy.pt (Layer 2). If given, it produces the "
                         "velocity command each nav tick instead of the hand-coded go_to_goal controller.")
parser.add_argument("--goal_x", type=float, default=2.5, help="single-goal world x [m] (forward)")
parser.add_argument("--goal_y", type=float, default=0.0, help="single-goal world y [m] (left)")
parser.add_argument("--waypoints", type=str, default=None,
                    help="multi-turn path as 'x1,y1;x2,y2;...' in world coords (overrides --goal_*). "
                         "e.g. '2,0;2,1.5' = go straight 2 m, then turn left and go 1.5 m")
parser.add_argument("--seconds", type=float, default=20.0)
parser.add_argument("--out", type=str, default="/home/adarshsosale/Workspace/Isaac RL Lab/hexabot_model/hexabot_nav.mp4")
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
from isaaclab.utils.math import quat_apply

_HERE = os.path.dirname(os.path.abspath(__file__))            # isaac_lab/
sys.path.insert(0, os.path.join(_HERE, "tasks", "hexabot"))
sys.path.insert(0, os.path.dirname(_HERE))                    # repo root

from hexabot_env_cfg import HEXABOT_CFG, HexabotFlatEnvCfg  # noqa: E402
from cpg import HexabotCPG  # noqa: E402
from isaac_lab.interfaces import VX_RANGE, VY_RANGE, YAW_RANGE  # noqa: E402
from isaac_lab.nav import NavGoalCfg, compute_nav_obs, go_to_goal  # noqa: E402

CFG = HexabotFlatEnvCfg()
NAV = NavGoalCfg()
DECIM = CFG.decimation
N_ACTIONS = CFG.action_space


def main():
    phys_hz = 200.0
    dt = 1.0 / phys_hz
    sim = SimulationContext(SimulationCfg(dt=dt, device="cuda:0"))

    mat = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.1, dynamic_friction=0.9, restitution=0.0,
        friction_combine_mode="multiply", restitution_combine_mode="multiply",
    )
    sim_utils.GroundPlaneCfg(physics_material=mat).func("/World/ground", sim_utils.GroundPlaneCfg(physics_material=mat))
    sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.95, 1.0))
    )
    sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0).func(
        "/World/KeyLight", sim_utils.DistantLightCfg(intensity=2500.0, angle=2.0),
        translation=(2.0, 2.0, 4.0), orientation=(0.86, 0.32, -0.32, 0.0),
    )

    robot = Articulation(HEXABOT_CFG.replace(prim_path="/World/Robot"))
    mat.func("/World/robotMaterial", mat)
    sim_utils.bind_physics_material("/World/Robot", "/World/robotMaterial")
    camera = Camera(CameraCfg(
        prim_path="/World/Camera", update_period=0.0, height=720, width=1280, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4),
        ),
    ))
    sim.reset()

    policy = torch.jit.load(args_cli.policy, map_location=sim.device).eval()
    print(f"[NAV] loaded locomotion policy: {args_cli.policy}", flush=True)

    nav_policy = None
    if args_cli.nav_policy:
        nav_policy = torch.jit.load(args_cli.nav_policy, map_location=sim.device).eval()
        print(f"[NAV] loaded NAVIGATION policy (Layer 2): {args_cli.nav_policy}", flush=True)
    cmd_lo = torch.tensor([VX_RANGE[0], VY_RANGE[0], YAW_RANGE[0]], device=sim.device)
    cmd_hi = torch.tensor([VX_RANGE[1], VY_RANGE[1], YAW_RANGE[1]], device=sim.device)

    cpg = HexabotCPG(robot.joint_names, num_envs=1, device=sim.device, cfg=CFG)
    soft_lo = robot.data.soft_joint_pos_limits[..., 0]
    soft_hi = robot.data.soft_joint_pos_limits[..., 1]
    default_q = robot.data.default_joint_pos.clone()

    if args_cli.waypoints:
        wps = [[float(v) for v in p.split(",")] for p in args_cli.waypoints.split(";")]
    else:
        wps = [[args_cli.goal_x, args_cli.goal_y]]
    waypoints = torch.tensor(wps, device=sim.device)    # (M, 2) world
    wp_idx = 0
    WP_ADVANCE = 0.30                                    # advance to next waypoint within this distance [m]
    print(f"[NAV] following {len(wps)} waypoint(s): {wps}", flush=True)
    command = torch.zeros(1, 3, device=sim.device)      # the frozen interface, refreshed by the nav layer
    last_action = torch.zeros(1, N_ACTIONS, device=sim.device)

    def robot_yaw():
        fwd = quat_apply(robot.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=sim.device).expand(1, 3))
        return torch.atan2(fwd[:, 1], fwd[:, 0])[0]

    def goal_rel_robot(target):
        pos = robot.data.root_pos_w[0, :2]
        d = target - pos
        yaw = robot_yaw()
        c, s = torch.cos(yaw), torch.sin(yaw)
        dx_b = (c * d[0] + s * d[1]).item()
        dy_b = (-s * d[0] + c * d[1]).item()
        return dx_b, dy_b

    def build_obs():
        return torch.cat(
            [
                robot.data.projected_gravity_b,
                robot.data.root_ang_vel_b,
                command,
                robot.data.joint_pos - robot.data.default_joint_pos,
                robot.data.joint_vel,
                last_action,
                cpg.phase_obs(),
            ],
            dim=-1,
        )

    # settle to stance
    for _ in range(int(args_cli.settle * phys_hz)):
        robot.set_joint_position_target(default_q)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)

    n_control = int(args_cli.seconds * phys_hz / DECIM)
    control_dt = DECIM / phys_hz
    frames = []
    obs = build_obs()
    reached_at = None
    for c in range(n_control):
        # --- Navigation tick: refresh the velocity command every nav_decimation steps ---
        if c % NAV.nav_decimation == 0:
            dxb, dyb = goal_rel_robot(waypoints[wp_idx])
            d = (dxb ** 2 + dyb ** 2) ** 0.5
            # advance to the next waypoint once close enough (turning toward it as needed).
            # The trained nav policy uses the same reach_tol the env advances on.
            advance_tol = NAV.reach_tol if nav_policy is not None else WP_ADVANCE
            if d < advance_tol and wp_idx < len(waypoints) - 1:
                wp_idx += 1
                dxb, dyb = goal_rel_robot(waypoints[wp_idx])
                d = (dxb ** 2 + dyb ** 2) ** 0.5
            if nav_policy is not None:
                # Layer 2 (trained): build the goal-conditioned obs and emit the command
                robot_xy = robot.data.root_pos_w[:, :2]                  # (1,2)
                nav_obs = compute_nav_obs(
                    robot_xy, robot_yaw().view(1), waypoints[wp_idx].view(1, 2),
                    command, robot.data.root_ang_vel_b[:, 2], NAV,
                )
                with torch.inference_mode():
                    command = torch.clamp(nav_policy(nav_obs), cmd_lo, cmd_hi)
            else:
                # Layer 2 (hand-coded placeholder)
                cmd = go_to_goal((dxb, dyb), lidar=None, reach_tol=NAV.reach_tol)  # dormant lidar slot
                command = cmd.to_tensor(device=sim.device, batch=1)
            if wp_idx == len(waypoints) - 1 and d < NAV.reach_tol and reached_at is None:
                reached_at = c * control_dt

        # --- Locomotion: track the command through the CPG ---
        with torch.inference_mode():
            action = policy(obs)
        cpg.step(action, control_dt)
        speed_scale = min(max(command[0, 0].item() / CFG.cpg_v_ref, 0.0), 1.2)
        target = torch.clamp(cpg.joint_targets(action, speed_scale), soft_lo, soft_hi)
        for _ in range(DECIM):
            robot.set_joint_position_target(target)
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt)
        last_action = action.clone()
        obs = build_obs()

        # tracking camera
        fx = robot.data.root_pos_w[0, 0].item()
        fy = robot.data.root_pos_w[0, 1].item()
        eye = torch.tensor([[fx - 1.3, fy - 1.8, 0.7]], device=sim.device)
        tgt = torch.tensor([[fx + 0.1, fy, 0.06]], device=sim.device)
        camera.set_world_poses_from_view(eye, tgt)
        sim.render()
        camera.update(dt)
        frames.append(camera.data.output["rgb"][0, ..., :3].clone().cpu().numpy().astype(np.uint8))

    px, py = robot.data.root_pos_w[0, 0].item(), robot.data.root_pos_w[0, 1].item()
    fg = waypoints[-1]
    final_d = ((fg[0].item() - px) ** 2 + (fg[1].item() - py) ** 2) ** 0.5
    print(f"[NAV] final waypoint=({fg[0].item():.2f},{fg[1].item():.2f})  robot=({px:.2f},{py:.2f})  "
          f"reached_waypoints={wp_idx + 1}/{len(waypoints)}  dist_to_final={final_d:.3f} m  "
          f"reached={'yes @%.1fs' % reached_at if reached_at else 'no'}", flush=True)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    imageio.mimsave(args_cli.out, frames, fps=phys_hz / DECIM, quality=8, macro_block_size=8)
    print(f"[NAV] wrote {args_cli.out}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
