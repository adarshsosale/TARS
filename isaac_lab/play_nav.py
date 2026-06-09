"""Export a trained Hexabot navigation policy to TorchScript/ONNX (Layer 2).

Companion to train_nav.py (mirrors play_hexabot.py). Loads a checkpoint, exports the
inference policy to <run>/exported/policy.pt (+ .onnx), then exits. The full-stack
rollout is recorded by run_nav_demo.py --nav_policy <exported/policy.pt>.

Usage (from any directory):
    cd external/IsaacLab
    ./isaaclab.sh -p ../../isaac_lab/play_nav.py            # exports latest checkpoint of latest run
    ./isaaclab.sh -p ../../isaac_lab/play_nav.py --checkpoint <abs/.../model_XXX.pt>
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

parser = argparse.ArgumentParser(description="Export a trained Hexabot navigation policy.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is enough for export).")
parser.add_argument("--task", type=str, default="Isaac-Nav-Waypoint-Flat-Hexabot-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

from rsl_rl.runners import OnPolicyRunner

import nav  # noqa: F401  — registers Isaac-Nav-Waypoint-Flat-Hexabot-Direct-v0
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
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

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    env.close()
    print(f"[EXPORT] wrote {os.path.join(export_model_dir, 'policy.pt')}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
