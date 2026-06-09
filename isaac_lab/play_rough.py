"""Export the trained Hexabot rough-terrain TEACHER policy (Milestone 1).

Companion to train_rough.py. Like play_hexabot.py it loads a checkpoint, optionally
selects the best walker, and exports an inference policy — but the teacher's actor is
`scan_encoder` + `actor_trunk` (the privileged height scan enters through a latent
bottleneck), so the stock RSL-RL exporter (which grabs only `policy.actor`) would
drop the encoder. We export a FULL-teacher wrapper instead: it takes the full
observation [proprio(75) | height_scan] and runs encoder + trunk, so policy.pt is a
faithful, deployable teacher (the next-milestone student replaces the scan encoder).

Usage (from external/IsaacLab):
    ./isaaclab.sh -p ../../isaac_lab/play_rough.py --select_best
    ./isaaclab.sh -p ../../isaac_lab/play_rough.py --checkpoint <abs/.../model_XXXX.pt>
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

parser = argparse.ArgumentParser(description="Export the Hexabot rough-terrain teacher policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is enough for export).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Hexabot-Direct-v0", help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--select_best", action="store_true", default=False, help="Export the best walker, not the last ckpt.")
parser.add_argument("--eval_envs", type=int, default=64, help="Parallel envs used when --select_best evaluates.")
parser.add_argument("--eval_secs", type=float, default=6.0, help="Seconds of rollout per checkpoint when --select_best.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import copy

import gymnasium as gym
import torch
import torch.nn as nn

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from rsl_rl.runners import OnPolicyRunner

# inject the teacher class so the runner can rebuild the policy from the cfg
import rsl_rl.runners.on_policy_runner as _opr
from hexabot.teacher_policy import HexabotTeacherActorCritic  # noqa: E402

_opr.HexabotTeacherActorCritic = HexabotTeacherActorCritic

import hexabot  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


class _TeacherExport(nn.Module):
    """Full-teacher inference: full obs [proprio | scan] -> action (encoder + trunk)."""

    def __init__(self, policy, normalizer):
        super().__init__()
        self.normalizer = copy.deepcopy(normalizer) if normalizer is not None else nn.Identity()
        self.scan_encoder = copy.deepcopy(policy.scan_encoder)
        self.actor_trunk = copy.deepcopy(policy.actor_trunk)
        self.n_proprio = int(policy.n_proprio)

    def forward(self, x):
        x = self.normalizer(x)
        proprio = x[..., : self.n_proprio]
        scan = x[..., self.n_proprio :]
        z = self.scan_encoder(scan)
        return self.actor_trunk(torch.cat([proprio, z], dim=-1))


def _eval_forward_score(env, runner, ckpt_path, eval_steps, device):
    """Roll out one checkpoint; score by mean forward body velocity [m/s] (+ mean height)."""
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)
    base = env.unwrapped
    base._step_count = 10**9  # full-speed forward commands
    obs, _ = env.reset()
    vsum = torch.zeros(base.num_envs, device=device)
    hsum = torch.zeros(base.num_envs, device=device)
    for _ in range(eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        vsum += base._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        hsum += base._robot.data.root_pos_w[:, 2] - base._ground_height
    return (vsum / eval_steps).mean().item(), (hsum / eval_steps).mean().item()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    n_envs = args_cli.eval_envs if args_cli.select_best else args_cli.num_envs
    env_cfg.scene.num_envs = n_envs if n_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[EXPORT] experiment dir: {log_root_path}", flush=True)
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[EXPORT] loading checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    if args_cli.select_best:
        import glob
        import re

        run_dir = os.path.dirname(resume_path)
        ckpts = sorted(
            glob.glob(os.path.join(run_dir, "model_*.pt")),
            key=lambda p: int(re.findall(r"model_(\d+)\.pt", os.path.basename(p))[0]),
        )
        eval_steps = int(args_cli.eval_secs * 50)
        print(f"[SELECT] evaluating {len(ckpts)} checkpoints in {run_dir}", flush=True)
        best_path, best_v = resume_path, -1.0e9
        for cp in ckpts:
            mv, mh = _eval_forward_score(env, runner, cp, eval_steps, agent_cfg.device)
            is_best = mv > best_v
            print(
                f"[SELECT] {os.path.basename(cp):>14}: fwd_vel={mv:+.4f} m/s  mean_h={mh:.4f} m"
                f"{'   <-- best' if is_best else ''}",
                flush=True,
            )
            if is_best:
                best_v, best_path = mv, cp
        resume_path = best_path
        print(f"[SELECT] best = {os.path.basename(resume_path)}  (fwd_vel={best_v:+.4f} m/s)", flush=True)
        runner.load(resume_path)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
    if isinstance(normalizer, nn.Identity):
        normalizer = None

    export_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_dir, exist_ok=True)

    wrapper = _TeacherExport(policy_nn, normalizer).to(agent_cfg.device).eval()
    full_dim = policy_nn.n_proprio + policy_nn.n_scan
    dummy = torch.zeros(1, full_dim, device=agent_cfg.device)
    # TorchScript (trace: the forward is a straight MLP path, no control flow)
    with torch.no_grad():
        scripted = torch.jit.trace(wrapper, dummy)
    jit_path = os.path.join(export_dir, "policy.pt")
    scripted.save(jit_path)
    # ONNX
    onnx_path = os.path.join(export_dir, "policy.onnx")
    torch.onnx.export(
        wrapper, dummy, onnx_path, input_names=["obs"], output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}}, opset_version=13,
    )

    env.close()
    print(f"[EXPORT] teacher obs width = {full_dim} (proprio {policy_nn.n_proprio} + scan {policy_nn.n_scan})", flush=True)
    print(f"[EXPORT] wrote {jit_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
