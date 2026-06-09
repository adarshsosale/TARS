# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Hexabot WAYPOINT NAVIGATION env (Layer 2), with the frozen locomotion policy in
the loop.

The navigation policy's action is a `(vx, vy, yaw)` command (clamped to the frozen
interface envelope). That command is HELD across `nav_decimation` locomotion control
steps while the FROZEN, TorchScript locomotion policy tracks it inside the same sim —
this is the explicit nav(5 Hz) / loco(50 Hz) / physics(200 Hz) timing split (hard
constraint #3). The nav reward is potential-based progress shaping toward the current
waypoint + a sparse reach bonus (then advance the index) + a small command-rate
regularizer; collision and path-cost terms are wired but INERT (no obstacles yet).

Nothing in the locomotion layer is modified: the loco policy is loaded read-only and
the CPG / robot / physics / domain-randomization come from an unmodified
`HexabotFlatEnvCfg`. The dormant lidar slot and inert collision/path terms are the
obstacle-stage seams (turn up `obstacle_density` later — no reshape).
"""

from __future__ import annotations

import glob
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from hexabot.cpg import HexabotCPG

from isaac_lab.nav import NavGoalCfg, compute_nav_obs, potential_shaping

from .nav_env_cfg import HexabotNavEnvCfg

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _resolve_loco_policy(path: str | None) -> str:
    """Return an explicit path, or auto-pick the newest exported locomotion policy."""
    if path:
        return path
    pattern = os.path.join(_PROJECT_ROOT, "logs", "rsl_rl", "hexabot_flat_direct", "*", "exported", "policy.pt")
    cands = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(
            f"No exported locomotion policy found at {pattern}. Train + export a Hexabot "
            f"locomotion policy first (play_hexabot.py), or pass loco_policy_path."
        )
    return max(cands, key=os.path.getmtime)


class HexabotNavEnv(DirectRLEnv):
    cfg: HexabotNavEnvCfg

    def __init__(self, cfg: HexabotNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.nav: NavGoalCfg = cfg.nav

        self._ALL = torch.arange(self.num_envs, device=self.device)

        # --- nav-layer command state (the frozen interface) -----------------------
        self._command = torch.zeros(self.num_envs, 3, device=self.device)        # current nav action
        self._prev_command = torch.zeros(self.num_envs, 3, device=self.device)   # for the rate regularizer
        # per-channel clamp envelope (vy pinned to 0; yaw enabled -> turn-to-face)
        self._cmd_lo = torch.tensor(
            [cfg.cmd_vx_range[0], cfg.cmd_vy_range[0], cfg.cmd_yaw_range[0]], device=self.device
        )
        self._cmd_hi = torch.tensor(
            [cfg.cmd_vx_range[1], cfg.cmd_vy_range[1], cfg.cmd_yaw_range[1]], device=self.device
        )

        # --- waypoint sequence state ---------------------------------------------
        self._waypoints = torch.zeros(self.num_envs, self.nav.n_waypoints, 2, device=self.device)  # world xy
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)   # potential baseline (current wp)

        # --- frozen locomotion layer, in the loop --------------------------------
        loco_path = _resolve_loco_policy(cfg.loco_policy_path)
        self._loco_policy = torch.jit.load(loco_path, map_location=self.device).eval()
        print(f"[NAV-ENV] frozen locomotion policy in loop: {loco_path}", flush=True)
        # the CPG is driven by the loco layer's params (unmodified HexabotFlatEnvCfg)
        self._loco_cpg = HexabotCPG(self._robot.data.joint_names, self.num_envs, self.device, cfg.loco)
        self._loco_prev_actions = torch.zeros(self.num_envs, cfg.loco.action_space, device=self.device)
        self._loco_joint_targets = self._robot.data.default_joint_pos.clone()
        self._loco_control_dt = cfg.loco.decimation * self.physics_dt   # 0.02 s -> 50 Hz
        self._loco_decim = cfg.loco.decimation                          # physics steps per loco control step
        self._phys_substep = 0

        # world +x unit (for reading the robot's world heading)
        self._world_x = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        # death detection (reuse the loco criteria) + post-reset settle grace
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["shaping", "reach_bonus", "control_reg", "collision", "path_cost"]
        }

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ helpers
    def _robot_yaw(self) -> torch.Tensor:
        fwd = quat_apply(self._robot.data.root_quat_w, self._world_x.expand(self.num_envs, 3))
        return torch.atan2(fwd[:, 1], fwd[:, 0])

    def _current_wp(self) -> torch.Tensor:
        return self._waypoints[self._ALL, self._wp_idx]   # (N, 2) world xy

    def _build_loco_obs(self) -> torch.Tensor:
        """The frozen locomotion policy's 75-d proprioceptive obs (must match the
        layout in hexabot_env.py:_get_observations — kept in sync by hand)."""
        loco = self.cfg.loco
        grav = self._robot.data.projected_gravity_b
        angv = self._robot.data.root_ang_vel_b
        jpos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        jvel = self._robot.data.joint_vel
        if self.cfg.loco_obs_noise and loco.domain_rand.enabled:
            dr = loco.domain_rand
            grav = grav + torch.randn_like(grav) * dr.noise_gravity
            angv = angv + torch.randn_like(angv) * dr.noise_ang_vel
            jpos = jpos + torch.randn_like(jpos) * dr.noise_joint_pos
            jvel = jvel + torch.randn_like(jvel) * dr.noise_joint_vel
        parts = [grav, angv, self._command, jpos, jvel, self._loco_prev_actions, self._loco_cpg.phase_obs()]
        if loco.n_height_scan > 0:
            parts.append(torch.zeros(self.num_envs, loco.n_height_scan, device=self.device))
        return torch.cat(parts, dim=-1)

    def _loco_control_step(self):
        """One 50-Hz locomotion control update: infer the frozen policy, advance the
        CPG, recompute clamped joint targets. The command is the held nav action."""
        obs = self._build_loco_obs()
        with torch.inference_mode():
            action = self._loco_policy(obs)
        self._loco_cpg.step(action, self._loco_control_dt)
        vx_act = self._command[:, 0].abs() / self.cfg.loco.cpg_v_ref
        yaw_act = self._command[:, 2].abs() / self.cfg.loco.cpg_yaw_ref
        speed_scale = torch.maximum(vx_act, yaw_act).clamp(0.0, 1.2)
        raw = self._loco_cpg.joint_targets(action, speed_scale)
        lo = self._robot.data.soft_joint_pos_limits[..., 0]
        hi = self._robot.data.soft_joint_pos_limits[..., 1]
        self._loco_joint_targets = torch.clamp(raw, lo, hi)
        self._loco_prev_actions = action.clone()

    # ------------------------------------------------------------------ stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        # NAV tick: clamp the raw policy output to the frozen trackable envelope and
        # hold it for the whole physics block.
        self._prev_command = self._command.clone()
        self._command = torch.clamp(actions, self._cmd_lo, self._cmd_hi)
        self._phys_substep = 0

    def _apply_action(self):
        # Runs every physics step. Re-run the 50-Hz locomotion controller at the loco
        # decimation boundary; otherwise hold the last joint target (zero-order hold).
        if self._phys_substep % self._loco_decim == 0:
            self._loco_control_step()
        self._robot.set_joint_position_target(self._loco_joint_targets)
        self._phys_substep += 1

    def _get_observations(self) -> dict:
        robot_xy = self._robot.data.root_pos_w[:, :2]
        yaw = self._robot_yaw()
        yaw_rate = self._robot.data.root_ang_vel_b[:, 2]
        # prev_cmd channel = the command we just executed (measurable "current velocity")
        obs = compute_nav_obs(robot_xy, yaw, self._current_wp(), self._command, yaw_rate, self.nav)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        robot_xy = self._robot.data.root_pos_w[:, :2]
        dist = torch.linalg.norm(robot_xy - self._current_wp(), dim=1)

        # potential-based progress shaping toward the CURRENT waypoint
        shaping = potential_shaping(self._prev_dist, dist, self.nav)

        # sparse reach bonus, then advance the waypoint index (not past the last)
        reached = dist < self.nav.reach_tol
        is_last = self._wp_idx >= (self.nav.n_waypoints - 1)
        reach_bonus = reached.float() * self.nav.reach_bonus
        advance = reached & ~is_last
        self._wp_idx = torch.where(advance, self._wp_idx + 1, self._wp_idx)
        # reset the potential baseline to the (possibly new) current waypoint so the
        # shaping term stays continuous across the hand-off — only reach_bonus pays the jump
        self._prev_dist = torch.linalg.norm(robot_xy - self._current_wp(), dim=1)

        # small command-rate regularizer -> smooth velocity commands
        control_reg = torch.sum(torch.square(self._command - self._prev_command), dim=1) * self.nav.control_reg_scale

        # INERT obstacle-stage terms (no obstacles on flat stage 1) — wired, contributing 0
        collision = torch.zeros(self.num_envs, device=self.device) * self.nav.collision_reward_scale
        path_cost = torch.zeros(self.num_envs, device=self.device) * self.nav.path_cost_reward_scale

        rewards = {
            "shaping": shaping,
            "reach_bonus": reach_bonus,
            "control_reg": control_reg,
            "collision": collision,
            "path_cost": path_cost,
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps_since_reset += 1
        robot_xy = self._robot.data.root_pos_w[:, :2]
        dist = torch.linalg.norm(robot_xy - self._current_wp(), dim=1)
        # success: the LAST waypoint reached within tolerance (sequence complete)
        success = (self._wp_idx >= (self.nav.n_waypoints - 1)) & (dist < self.nav.reach_tol)

        # fall detection (same criteria as the loco layer)
        too_low = self._robot.data.root_pos_w[:, 2] < 0.035
        tilted = self._robot.data.projected_gravity_b[:, 2] > -0.5
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        fell = (too_low | tilted | base_contact) & (self._steps_since_reset > self.cfg.loco.settle_steps)

        terminated = success | fell
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------ reset
    def _sample_waypoints(self, env_ids: torch.Tensor):
        n = len(env_ids)
        K = self.nav.n_waypoints
        R = self.cfg.arena_radius
        origins = self._terrain.env_origins[env_ids, :2]          # (n, 2) world
        offs = torch.zeros(n, K, 2, device=self.device)
        prev = torch.zeros(n, 2, device=self.device)              # spawn is at the origin (offset 0)
        for k in range(K):
            cand = torch.empty(n, 2, device=self.device).uniform_(-R, R)
            # a few resample tries to keep consecutive waypoints at least min_leg apart
            for _ in range(4):
                too_close = torch.linalg.norm(cand - prev, dim=1) < self.cfg.min_leg
                if not bool(too_close.any()):
                    break
                m = int(too_close.sum())
                cand[too_close] = torch.empty(m, 2, device=self.device).uniform_(-R, R)
            offs[:, k] = cand
            prev = cand
        self._waypoints[env_ids] = origins.unsqueeze(1) + offs
        self._wp_idx[env_ids] = 0

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # reset nav-layer state
        self._command[env_ids] = 0.0
        self._prev_command[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0

        # reset the in-loop locomotion state for these envs
        self._loco_cpg.reset(env_ids)
        self._loco_prev_actions[env_ids] = 0.0
        self._loco_joint_targets[env_ids] = self._robot.data.default_joint_pos[env_ids]

        # robot state: spawn at the env origin in the standing stance (small joint noise)
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.05, 0.05)
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # sample a fresh waypoint sequence and seed the potential baseline
        self._sample_waypoints(env_ids)
        robot_xy = default_root_state[:, :2]
        self._prev_dist[env_ids] = torch.linalg.norm(robot_xy - self._current_wp()[env_ids], dim=1)

        # episode logging (tensorboard): mean reward per term + waypoints reached
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Metrics/waypoints_reached"] = torch.mean(self._wp_idx[env_ids].float())
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        term = dict()
        term["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        term["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(term)
