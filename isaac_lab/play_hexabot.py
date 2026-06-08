"""Project-level export wrapper for RSL-RL (Hexabot).

Drop-in companion to isaac_lab/train_hexabot.py. Unlike IsaacLab's stock play.py, this:
  1. Registers the Hexabot task (from isaac_lab/tasks/) so the task id resolves.
  2. Routes paths to <project_root>/logs/ (same as train_hexabot.py).
  3. Loads a checkpoint, exports the policy to TorchScript (policy.pt) + ONNX,
     then EXITS — no infinite play loop (record_hexabot.py does the rendering).

Usage (from any directory):
    cd external/IsaacLab
    ./isaaclab.sh -p ../../isaac_lab/play_hexabot.py --checkpoint <abs/path/to/model_XXXX.pt>
    # or, with no --checkpoint, exports the latest checkpoint of the latest run.
"""

import os
import sys

# Route paths to <project_root>/logs/ by making the project root the cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(_PROJECT_ROOT)

# Add IsaacLab's rsl_rl scripts dir so `import cli_args` resolves.
_RSL_SCRIPTS = os.path.abspath(
    os.path.join(_PROJECT_ROOT, "external", "IsaacLab", "scripts", "reinforcement_learning", "rsl_rl")
)
sys.path.insert(0, _RSL_SCRIPTS)

# Add our tasks dir so `import hexabot` resolves, and the repo root so
# `import isaac_lab.interfaces` (the frozen inter-layer interface) resolves.
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "isaac_lab", "tasks"))
sys.path.insert(0, _PROJECT_ROOT)

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export a trained RSL-RL Hexabot policy to TorchScript/ONNX.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (1 is enough for export).")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Hexabot-Direct-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--select_best",
    action="store_true",
    default=False,
    help="Evaluate every model_*.pt in the run and export the best walker (not the last checkpoint).",
)
parser.add_argument("--eval_envs", type=int, default=64, help="Parallel envs used when --select_best evaluates checkpoints.")
parser.add_argument("--eval_secs", type=float, default=6.0, help="Seconds of rollout per checkpoint when --select_best.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# export only needs a headless app
args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

from rsl_rl.runners import OnPolicyRunner

import hexabot  # noqa: F401  — registers Isaac-Velocity-Flat-Hexabot-Direct-v0
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _eval_forward_score(env, runner, ckpt_path, eval_steps, device):
    """Roll out one checkpoint and score it by mean forward body velocity [m/s].

    Velocity-based (not displacement) so it is robust to mid-rollout resets, and it
    cleanly separates a real walker from a belly-crawler. Also returns mean base
    height so a low score can be attributed to falling vs. belly-crawling.
    """
    import torch

    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)
    base = env.unwrapped
    base._step_count = 10**9  # force the curriculum to full-speed forward commands
    obs, _ = env.reset()
    vsum = torch.zeros(base.num_envs, device=device)
    hsum = torch.zeros(base.num_envs, device=device)
    for _ in range(eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        vsum += base._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
        hsum += base._robot.data.root_pos_w[:, 2]
    return (vsum / eval_steps).mean().item(), (hsum / eval_steps).mean().item()


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Load a checkpoint and export the inference policy."""
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

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

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
        eval_steps = int(args_cli.eval_secs * 50)  # 50 Hz control
        print(
            f"[SELECT] evaluating {len(ckpts)} checkpoints in {run_dir} "
            f"({args_cli.eval_envs} envs x {args_cli.eval_secs:.0f}s each)",
            flush=True,
        )
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

    # extract the policy network (try/except keeps backwards compatibility)
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
    # marker line the orchestration script greps for:
    print(f"[EXPORT] wrote {os.path.join(export_model_dir, 'policy.pt')}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
