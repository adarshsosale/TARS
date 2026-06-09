"""Render a short clip of the Hexabot walking over ROUGH terrain (Milestone 1).

play.py-style on-demand video: builds the real rough-terrain env (so the privileged
height-scan observation is constructed exactly as in training — no drift), loads a
checkpoint via the rsl_rl runner, rolls the teacher policy, and films one robot with
a tracking camera. This is the monitoring/video mechanism for the milestone.

By default it loads the LATEST checkpoint of the latest rough run, so you can watch
training progress at any time.

Usage (from external/IsaacLab, conda env env_isaaclab):
    ./isaaclab.sh -p ../../isaac_lab/render_rough.py                 # latest checkpoint
    ./isaaclab.sh -p ../../isaac_lab/render_rough.py --checkpoint <abs/.../model_XXXX.pt>
    ./isaaclab.sh -p ../../isaac_lab/render_rough.py --terrain_level 7 --seconds 12 --cmd_vx 0.2
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_PROJECT_ROOT)
_RSL_SCRIPTS = os.path.abspath(
    os.path.join(_PROJECT_ROOT, "external", "IsaacLab", "scripts", "reinforcement_learning", "rsl_rl")
)
sys.path.insert(0, _RSL_SCRIPTS)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "isaac_lab", "tasks"))
sys.path.insert(0, _PROJECT_ROOT)

import argparse

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Render the Hexabot rough-terrain policy.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Hexabot-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--num_envs", type=int, default=16, help="Envs spread across terrain (one is filmed).")
parser.add_argument("--terrain_level", type=int, default=6, help="Max terrain difficulty level to spawn on.")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--cmd_vx", type=float, default=0.2, help="commanded forward velocity [m/s]")
parser.add_argument("--settle", type=float, default=0.5)
parser.add_argument(
    "--select_best", action="store_true", default=False,
    help="Evaluate every model_*.pt in the run and render the one that moves FARTHEST from its "
         "spawn origin (peak horizontal displacement), not the latest checkpoint.",
)
parser.add_argument("--eval_secs", type=float, default=6.0, help="Seconds of rollout per checkpoint when --select_best.")
parser.add_argument(
    "--out", type=str,
    default="/home/adarshsosale/Workspace/Isaac RL Lab/hexabot_model/hexabot_rough_locomotion.mp4",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import numpy as np
import torch
import imageio.v2 as imageio

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner

import rsl_rl.runners.on_policy_runner as _opr
from hexabot.teacher_policy import HexabotTeacherActorCritic  # noqa: E402

_opr.HexabotTeacherActorCritic = HexabotTeacherActorCritic

import hexabot  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _eval_displacement_score(env, runner, ckpt_path, eval_steps, device):
    """Roll out one checkpoint; score by mean PEAK horizontal displacement from spawn origin [m].

    Tracks the per-env running max distance from `env_origins` so a robot that walks far and
    then dies/resets is still credited with how far it got. This is the "most movement from the
    origin" metric the renderer selects on (distance-based, unlike play_rough's velocity score).
    """
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)
    base = env.unwrapped
    base._step_count = 10**9  # full-speed forward commands
    obs, _ = env.reset()
    origin_xy = base._terrain.env_origins[:, :2]
    peak = torch.zeros(base.num_envs, device=device)
    for _ in range(eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        disp = torch.linalg.norm(base._robot.data.root_pos_w[:, :2] - origin_xy, dim=1)
        peak = torch.maximum(peak, disp)
    return peak.mean().item()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # turn the on-demand tracking camera on; spawn on harder terrain; freeze curriculum
    env_cfg.enable_render_camera = True
    env_cfg.terrain.max_init_terrain_level = args_cli.terrain_level
    env_cfg.terrain_curriculum_enabled = False

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[RENDER] loading checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    if args_cli.select_best and not args_cli.checkpoint:
        import glob
        import re

        run_dir = os.path.dirname(resume_path)
        ckpts = sorted(
            glob.glob(os.path.join(run_dir, "model_*.pt")),
            key=lambda p: int(re.findall(r"model_(\d+)\.pt", os.path.basename(p))[0]),
        )
        eval_steps = int(args_cli.eval_secs / env.unwrapped.step_dt)
        print(f"[SELECT] scoring {len(ckpts)} checkpoints by peak displacement from origin "
              f"(over {args_cli.num_envs} envs, {args_cli.eval_secs:.0f}s each) in {run_dir}", flush=True)
        best_path, best_d = resume_path, -1.0e9
        for cp in ckpts:
            md = _eval_displacement_score(env, runner, cp, eval_steps, agent_cfg.device)
            is_best = md > best_d
            print(f"[SELECT] {os.path.basename(cp):>16}: mean_peak_disp={md:.3f} m"
                  f"{'   <-- best' if is_best else ''}", flush=True)
            if is_best:
                best_d, best_path = md, cp
        resume_path = best_path
        runner.load(resume_path)
        print(f"[SELECT] BEST = {os.path.basename(resume_path)}  "
              f"(mean_peak_disp={best_d:.3f} m) -- rendering this one", flush=True)

    policy = runner.get_inference_policy(device=agent_cfg.device)

    base = env.unwrapped
    sim = base.sim
    camera = base._render_camera
    assert camera is not None, "render camera was not created (enable_render_camera)"
    dt = base.physics_dt
    control_dt = base.step_dt

    # full-speed forward command; pin terrain so it can't reshuffle mid-clip
    base._step_count = 10**9
    obs, _ = env.reset()
    # film the robot sitting on the hardest terrain in the batch
    film = int(torch.argmax(base._terrain.terrain_levels)) if base._terrain.terrain_origins is not None else 0
    base._commands[:, 0] = args_cli.cmd_vx
    base._commands[:, 1:] = 0.0

    # settle
    for _ in range(int(args_cli.settle / control_dt)):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs) * 0.0)
        base._commands[:, 0] = args_cli.cmd_vx

    frames = []
    n_control = int(args_cli.seconds / control_dt)
    for _ in range(n_control):
        with torch.inference_mode():
            action = policy(obs)
        obs, _, _, _ = env.step(action)
        base._commands[:, 0] = args_cli.cmd_vx
        base._commands[:, 1:] = 0.0
        # tracking camera on the filmed robot
        rp = base._robot.data.root_pos_w[film]
        fx, fy, fz = rp[0].item(), rp[1].item(), rp[2].item()
        eye = torch.tensor([[fx - 1.0, fy - 1.4, fz + 0.6]], device=sim.device)
        tgt = torch.tensor([[fx + 0.1, fy, fz - 0.02]], device=sim.device)
        camera.set_world_poses_from_view(eye, tgt)
        sim.render()
        camera.update(dt)
        rgb = camera.data.output["rgb"][0, ..., :3].clone().cpu().numpy().astype(np.uint8)
        frames.append(rgb)

    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    fps = 1.0 / control_dt
    imageio.mimsave(args_cli.out, frames, fps=fps, quality=8, macro_block_size=8)
    print(f"[RENDER] filmed env {film} at terrain level "
          f"{int(base._terrain.terrain_levels[film]) if base._terrain.terrain_origins is not None else 0}", flush=True)
    print(f"[RENDER] wrote {args_cli.out}  ({len(frames)} frames @ {fps:.0f} fps)", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
