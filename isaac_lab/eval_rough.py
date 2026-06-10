# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Held-out, fixed-seed evaluation of the Hexabot rough-terrain teacher (Milestone 1.5, Phase A).

Measures per-terrain-FAMILY traversal success at a fixed difficulty, with no
retraining. Modeled on play_rough.py's checkpoint-loading path (including
--select_best), but headless and batch: it isolates ONE terrain family per run
(a generator with only that sub-terrain at proportion 1.0, curriculum off,
difficulty pinned via difficulty_range=(D, D)), fixes the command to straight
ahead, rolls one full episode over many envs, and writes a metrics JSON.

WHY ONE FAMILY PER PROCESS: Isaac's SimulationApp / USD stage is a singleton, so
building seven scenes back-to-back in one process is fragile (prim-path clashes
at /World/ground). Each family therefore runs as its own deterministic process;
`scripts/eval_rough.sh` loops the seven families and merges the per-family JSONs
into the single `<checkpoint>_d{D}.json` the milestone asks for. The measured
outcome is identical to "seven envs sequentially"; only the process boundary moves.

Definition of success (per the milestone spec):
    survives the episode (ends by TIMEOUT, never a death) AND covers
    >= success_frac * tile_size of commanded-direction (world +x) distance.

Determinism: a single --seed fixes torch / numpy / the terrain generator and all
DR draws, so two invocations with identical args produce identical JSON.

Usage (from external/IsaacLab, conda env env_isaaclab):
    ./isaaclab.sh -p ../../isaac_lab/eval_rough.py --family stairs_pure_down --difficulty 0.8
    ./isaaclab.sh -p ../../isaac_lab/eval_rough.py --family random_rough --select_best
    ./isaaclab.sh -p ../../isaac_lab/eval_rough.py --select_best --resolve_only   # print best ckpt and exit
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

# terrain FAMILY -> the sub-terrain key in HEXABOT_ROUGH_TERRAINS_CFG it isolates.
# Reuses the existing sub-terrain functions/cfgs in rough_terrains.py (no duplicated
# height-field code). "mixed" families carry the noise/box overlays; the two pure
# families are the worst-case (stairs down) and the standalone random rough.
FAMILY_TO_SUBTERRAIN = {
    "slope_noise_up": "sloped_rough",       # mixed: bumps on an up-ramp
    "slope_noise_down": "sloped_rough_inv",  # mixed: bumps on a down-ramp
    "stairs_noise_up": "stairs_rough",       # mixed: uneven steps up
    "stairs_noise_down": "stairs_rough_inv",  # mixed: uneven steps down
    "slope_boxes": "sloped_boxes",           # mixed: little boxes on an up-ramp
    "stairs_pure_down": "stairs_inv",        # pure: inverted pyramid stairs (worst case)
    "random_rough": "random_rough",          # pure: standalone random rough ground
}
ALL_FAMILIES = list(FAMILY_TO_SUBTERRAIN.keys())

parser = argparse.ArgumentParser(description="Held-out per-family eval of the Hexabot rough-terrain teacher.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Hexabot-Direct-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--family", type=str, default="random_rough", choices=ALL_FAMILIES,
                    help="Which terrain family to isolate and evaluate.")
parser.add_argument("--difficulty", type=float, default=0.8, help="Fixed terrain difficulty in [0,1].")
parser.add_argument("--seed", type=int, default=42, help="Master seed (env + terrain + DR + commands).")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel episodes for this family.")
parser.add_argument("--cmd_vx", type=float, default=0.25, help="Fixed forward command [m/s] (straight ahead).")
parser.add_argument("--success_frac", type=float, default=0.7,
                    help="Distance success threshold as a fraction of the tile size (0.7 -> 1.4 m on a 2 m tile).")
parser.add_argument("--out_json", type=str, default=None, help="Override the output JSON path.")
parser.add_argument("--select_best", action="store_true", default=False,
                    help="Pick the best walker (mean forward velocity) among model_*.pt, like play_rough.")
parser.add_argument("--resolve_only", action="store_true", default=False,
                    help="With --select_best: print 'RESOLVED_CHECKPOINT=<path>' and exit (no full eval).")
parser.add_argument("--eval_envs", type=int, default=64, help="Envs used for --select_best scoring.")
parser.add_argument("--eval_secs", type=float, default=6.0, help="Seconds per checkpoint for --select_best scoring.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import copy
import json
import random

import numpy as np
import torch

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
from hexabot.rough_terrains import HEXABOT_ROUGH_TERRAINS_CFG
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_family_terrain(family: str, difficulty: float, seed: int):
    """A TerrainGeneratorCfg holding ONLY `family` at proportion 1.0, fixed difficulty."""
    gen = copy.deepcopy(HEXABOT_ROUGH_TERRAINS_CFG)
    key = FAMILY_TO_SUBTERRAIN[family]
    sub = copy.deepcopy(gen.sub_terrains[key])
    sub.proportion = 1.0
    gen.sub_terrains = {key: sub}
    gen.curriculum = False                       # difficulty fixed (not row-indexed)
    gen.difficulty_range = (difficulty, difficulty)
    gen.seed = seed                              # deterministic height fields
    gen.use_cache = False
    return gen


def _eval_forward_score(env, runner, ckpt_path, eval_steps, device):
    """Roll out one checkpoint; score by mean forward body velocity [m/s] (play_rough's metric)."""
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=device)
    base = env.unwrapped
    base._step_count = 10**9
    obs, _ = env.reset()
    vsum = torch.zeros(base.num_envs, device=device)
    for _ in range(eval_steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        vsum += base._robot.data.root_lin_vel_b[:, 0].clamp(min=0.0)
    return (vsum / eval_steps).mean().item()


def _select_best_checkpoint(env, runner, resume_path, device):
    import glob
    import re

    run_dir = os.path.dirname(resume_path)
    ckpts = sorted(
        glob.glob(os.path.join(run_dir, "model_*.pt")),
        key=lambda p: int(re.findall(r"model_(\d+)\.pt", os.path.basename(p))[0]),
    )
    eval_steps = int(args_cli.eval_secs * 50)
    print(f"[SELECT] scoring {len(ckpts)} checkpoints by mean fwd velocity in {run_dir}", flush=True)
    best_path, best_v = resume_path, -1.0e9
    for cp in ckpts:
        mv = _eval_forward_score(env, runner, cp, eval_steps, device)
        is_best = mv > best_v
        print(f"[SELECT] {os.path.basename(cp):>16}: fwd_vel={mv:+.4f} m/s{'   <-- best' if is_best else ''}", flush=True)
        if is_best:
            best_v, best_path = mv, cp
    print(f"[SELECT] best = {os.path.basename(best_path)} (fwd_vel={best_v:+.4f} m/s)", flush=True)
    return best_path


def _install_done_reason_capture(base):
    """Wrap _get_dones so we can attribute each death to too-low vs orientation.

    Replicates the env's own death logic (too_low | tilted | base_contact, gated by
    the settle grace) and stashes the component masks for the most recent step. Kept
    in lock-step with hexabot_env._get_dones; if that logic changes, update here too.
    """
    orig = base._get_dones

    def patched():
        died, time_out = orig()  # rough._get_dones refreshes self._ground_height first
        root_z = base._robot.data.root_pos_w[:, 2]
        too_low = (root_z - base._ground_height) < 0.035
        tilted = base._robot.data.projected_gravity_b[:, 2] > -0.5
        ncf = base._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(
            torch.max(torch.norm(ncf[:, :, base._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        settle_ok = base._steps_since_reset > base.cfg.settle_steps
        base._eval_too_low = too_low & settle_ok
        base._eval_orient = (tilted | base_contact) & settle_ok
        return died, time_out

    base._get_dones = patched


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    device = agent_cfg.device
    seed = args_cli.seed
    _seed_everything(seed)

    # --- isolate one terrain family, fixed difficulty, curriculum off -----------
    n_envs = args_cli.eval_envs if (args_cli.select_best and args_cli.resolve_only) else args_cli.num_envs
    env_cfg.scene.num_envs = n_envs
    env_cfg.seed = seed
    agent_cfg.seed = seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    fam_terrain = _make_family_terrain(args_cli.family, args_cli.difficulty, seed)
    env_cfg.terrain.terrain_generator = fam_terrain
    env_cfg.terrain_curriculum_enabled = False     # frozen terrain, no promote/demote
    # curriculum is off, so EVERY tile is this family at the pinned difficulty; spread
    # envs across all rows x cols tiles (variety) rather than pinning them to row 0.
    env_cfg.terrain.max_init_terrain_level = fam_terrain.num_rows - 1

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

    print(f"[EVAL] loading checkpoint: {resume_path}", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(resume_path)

    if args_cli.select_best:
        resume_path = _select_best_checkpoint(env, runner, resume_path, device)
        runner.load(resume_path)
        if args_cli.resolve_only:
            print(f"RESOLVED_CHECKPOINT={resume_path}", flush=True)
            env.close()
            return

    base = env.unwrapped
    _install_done_reason_capture(base)
    policy = runner.get_inference_policy(device=device)

    # fully-ramped curriculum so mid-episode auto-resets don't re-pin vx to 0
    base._step_count = 10**9
    _seed_everything(seed)  # re-seed right before the rollout for invocation-order independence
    obs, _ = env.reset()

    # one clean, synchronized full episode for every env
    base.episode_length_buf[:] = 0
    base._steps_since_reset[:] = 0
    origin_x = base._terrain.env_origins[:, 0].clone()
    cmd_vx = args_cli.cmd_vx
    base._commands[:] = 0.0
    base._commands[:, 0] = cmd_vx

    n = base.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=device)
    frozen_dist = torch.zeros(n, device=device)
    # reason code per env: 0 too_low, 1 orientation, 2 timeout(alive)
    reason = torch.full((n,), 2, dtype=torch.long, device=device)
    stand_count = torch.zeros(n, device=device)
    steps_alive = torch.zeros(n, device=device)

    max_steps = int(base.max_episode_length)
    for _ in range(max_steps):
        # position/speed at the START of this control step (pre-death, pre-auto-reset)
        disp_x = base._robot.data.root_pos_w[:, 0] - origin_x
        vx_body = base._robot.data.root_lin_vel_b[:, 0]
        stand_count += (alive & (vx_body < 0.02)).float()
        steps_alive += alive.float()

        with torch.inference_mode():
            action = policy(obs)
        obs, _, _, _ = env.step(action)
        # hold the straight-ahead command (override whatever an auto-reset sampled)
        base._commands[:] = 0.0
        base._commands[:, 0] = cmd_vx

        died = base.reset_terminated.clone()
        timed = base.reset_time_outs.clone()
        ended = (died | timed) & alive

        frozen_dist = torch.where(ended, disp_x, frozen_dist)
        too_low = base._eval_too_low
        orient = base._eval_orient
        # died -> too_low(0) else orientation(1); timeout -> 2. Attributed only on the
        # first episode end (alive gate), so later auto-reset deaths are ignored.
        died_reason = torch.where(too_low, torch.zeros_like(reason), torch.ones_like(reason))
        new_reason = torch.where(died, died_reason, torch.full_like(reason, 2))
        reason = torch.where(ended, new_reason, reason)

        alive = alive & ~(died | timed)
        if not bool(alive.any()):
            break

    # --- metrics -------------------------------------------------------------
    tile_size = float(env_cfg.terrain.terrain_generator.size[0])
    dist_thresh = args_cli.success_frac * tile_size
    timeout_alive_mask = reason == 2
    success_mask = timeout_alive_mask & (frozen_dist >= dist_thresh)

    metrics = {
        "family": args_cli.family,
        "sub_terrain": FAMILY_TO_SUBTERRAIN[args_cli.family],
        "difficulty": args_cli.difficulty,
        "seed": seed,
        "num_envs": n,
        "cmd_vx": cmd_vx,
        "tile_size_m": tile_size,
        "success_dist_m": dist_thresh,
        "checkpoint": os.path.basename(resume_path),
        "checkpoint_path": resume_path,
        "success_rate": float(success_mask.float().mean()),
        "mean_distance_m": float(frozen_dist.mean()),
        "p10_distance_m": float(torch.quantile(frozen_dist, 0.10)),
        "died_too_low": float((reason == 0).float().mean()),
        "died_orientation": float((reason == 1).float().mean()),
        "timeout_alive": float(timeout_alive_mask.float().mean()),
        "stand_when_cmd_frac": float((stand_count / steps_alive.clamp(min=1.0)).mean()),
    }

    if args_cli.out_json:
        out_json = args_cli.out_json
    else:
        ckpt_tag = os.path.splitext(os.path.basename(resume_path))[0]
        eval_dir = os.path.join(os.path.dirname(resume_path), "eval", "parts")
        out_json = os.path.join(eval_dir, f"{ckpt_tag}_d{args_cli.difficulty:g}__{args_cli.family}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 64, flush=True)
    print(f"[EVAL] family={args_cli.family}  difficulty={args_cli.difficulty}  n={n}", flush=True)
    print(f"  success_rate        = {metrics['success_rate']:.3f}", flush=True)
    print(f"  mean_distance_m     = {metrics['mean_distance_m']:.3f}", flush=True)
    print(f"  p10_distance_m      = {metrics['p10_distance_m']:.3f}", flush=True)
    print(f"  died_too_low        = {metrics['died_too_low']:.3f}", flush=True)
    print(f"  died_orientation    = {metrics['died_orientation']:.3f}", flush=True)
    print(f"  timeout_alive       = {metrics['timeout_alive']:.3f}", flush=True)
    print(f"  stand_when_cmd_frac = {metrics['stand_when_cmd_frac']:.3f}", flush=True)
    print(f"[EVAL] wrote {out_json}", flush=True)
    print(f"RESOLVED_CHECKPOINT={resume_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
