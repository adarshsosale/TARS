# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from .hexabot_env_cfg import HexabotFlatEnvCfg


class HexabotEnv(DirectRLEnv):
    cfg: HexabotFlatEnvCfg

    def __init__(self, cfg: HexabotFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # commands: [vx, vy, yaw_rate]; vy & yaw are kept at 0 (straight-line)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # control steps elapsed since each env was last reset (for the settle grace)
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # global control-step counter that drives the stand-first curriculum
        self._step_count = 0
        # tibia-local offset to the claw tip (for the foot-below-belly reward)
        self._claw_offset_local = torch.tensor(
            [self.cfg.claw_offset, 0.0, 0.0], device=self.device
        ).view(1, 1, 3)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "forward_progress",
                "track_ang_vel_z_exp",
                "lateral_vel_l2",
                "yaw_rate_l2",
                "lateral_pos_l2",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "flat_orientation_l2",
                "base_height_l2",
                "belly_clearance",
                "foot_support",
                "alive",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "dof_pos_limits",
                "feet_air_time",
                "foot_slip",
                "undesired_contacts",
                "tetrapod_contact",
                "gait_symmetry",
                "foot_clearance",
            ]
        }

        # contact-sensor body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, feet_names = self._contact_sensor.find_bodies("tibia_.*")  # the 6 claw feet

        # left<->right mirror permutation over the 6 feet (in self._feet_ids order),
        # for the symmetric-gait reward. names look like 'tibia_lf' -> mirror 'tibia_rf'.
        def _mirror(nm: str) -> str:
            prefix, sp = nm.rsplit("_", 1)              # 'tibia', 'lf'
            side = "r" if sp[0] == "l" else "l"
            return f"{prefix}_{side}{sp[1:]}"

        _name_to_local = {nm: i for i, nm in enumerate(feet_names)}
        self._feet_mirror_idx = torch.tensor(
            [_name_to_local[_mirror(nm)] for nm in feet_names], device=self.device, dtype=torch.long
        )
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(["base_link", "coxa_.*", "femur_.*"])
        # robot-frame body indices for the feet (for foot-velocity / slip)
        self._feet_body_ids, _ = self._robot.find_bodies("tibia_.*")

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                self._commands,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # forward (x) velocity tracking; vy command is 0
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # linear forward-progress reward: 0 when standing, capped at the command
        forward_progress = torch.clamp(
            torch.minimum(self._robot.data.root_lin_vel_b[:, 0], self._commands[:, 0]), min=0.0
        )
        # yaw-rate tracking (command is 0 -> go straight)
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        # explicit lateral velocity penalty (straight line)
        lateral_vel = torch.square(self._robot.data.root_lin_vel_b[:, 1])
        # explicit yaw-rate penalty (don't turn)
        yaw_rate_l2 = torch.square(self._robot.data.root_ang_vel_b[:, 2])
        # lateral position drift off the spawn x-axis (enforces straight line)
        lateral_pos = torch.square(
            self._robot.data.root_pos_w[:, 1] - self._terrain.env_origins[:, 1]
        )
        # vertical velocity penalty
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # roll/pitch rate penalty
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # upright (projected gravity should be (0,0,-1))
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # base height tracking
        base_height_error = torch.square(self._robot.data.root_pos_w[:, 2] - self.cfg.target_height)
        # underbelly height above ground (base centre minus half the body thickness)
        belly_height = self._robot.data.root_pos_w[:, 2] - self.cfg.belly_half_thickness
        # one-sided penalty: underbelly dropping below the min clearance (anti belly-crawl)
        belly_clearance = torch.clamp(self.cfg.belly_clearance_min - belly_height, min=0.0)
        # foot-below-belly support: claw tip world z via tibia pose + local claw offset
        feet_quat = self._robot.data.body_quat_w[:, self._feet_body_ids, :]
        feet_pos = self._robot.data.body_pos_w[:, self._feet_body_ids, :]
        claw_w = feet_pos + quat_apply(feet_quat, self._claw_offset_local.expand_as(feet_pos))
        foot_z = claw_w[..., 2]
        # reward each foot sitting well below the belly (legs extended, body held high)
        support = torch.clamp(
            belly_height.unsqueeze(1) - foot_z, min=0.0, max=self.cfg.support_target
        )
        foot_support = support.mean(dim=1)
        # alive bonus
        alive = torch.ones(self.num_envs, device=self.device)
        # effort / smoothness
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # joint-limit penalty: amount each joint is past its soft position limit
        soft_limits = self._robot.data.soft_joint_pos_limits
        out_of_limits = -(self._robot.data.joint_pos - soft_limits[..., 0]).clip(max=0.0)
        out_of_limits += (self._robot.data.joint_pos - soft_limits[..., 1]).clip(min=0.0)
        dof_pos_limits = torch.sum(out_of_limits, dim=1)
        # feet air time (encourage real, lifted steps when commanded to move)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - self.cfg.feet_air_time_threshold) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # foot slip: planar foot speed while the foot is in contact
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        feet_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0
        )
        feet_planar_speed = torch.sum(
            torch.square(self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2]), dim=-1
        )
        foot_slip = torch.sum(feet_planar_speed * feet_contact, dim=1)
        # --- coordinated symmetric tetrapod gait (only while commanded to move) ---
        moving = (torch.norm(self._commands[:, :2], dim=1) > 0.1).float()
        n_contact = feet_contact.float().sum(dim=1)
        # bell peaking at exactly 4 feet planted (0.37 at 3 or 5 -> tolerant of swing transitions)
        tetrapod_contact = torch.exp(-torch.square(n_contact - 4.0)) * moving
        # left-right symmetric contact pattern: fraction of feet matching their mirror.
        # require >=1 foot lifted so a frozen all-6-down stance can't farm symmetry.
        sym_match = (feet_contact == feet_contact[:, self._feet_mirror_idx]).float().mean(dim=1)
        gait_symmetry = sym_match * (n_contact < 6).float() * moving
        # swing-foot clearance: reward lifted feet reaching a target apex height so the
        # legs take big, deliberate strides instead of skittering just off the ground.
        swing = (~feet_contact).float()
        foot_clearance = torch.sum(
            torch.clamp(foot_z, min=0.0, max=self.cfg.foot_clearance_target) * swing, dim=1
        ) * moving
        # undesired contacts (body / coxa / femur hitting the ground)
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "forward_progress": forward_progress * self.cfg.forward_progress_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lateral_vel_l2": lateral_vel * self.cfg.lateral_vel_reward_scale * self.step_dt,
            "yaw_rate_l2": yaw_rate_l2 * self.cfg.yaw_rate_l2_reward_scale * self.step_dt,
            "lateral_pos_l2": lateral_pos * self.cfg.lateral_pos_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "base_height_l2": base_height_error * self.cfg.base_height_reward_scale * self.step_dt,
            "belly_clearance": belly_clearance * self.cfg.belly_clearance_reward_scale * self.step_dt,
            "foot_support": foot_support * self.cfg.foot_support_reward_scale * self.step_dt,
            "alive": alive * self.cfg.alive_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_pos_limits": dof_pos_limits * self.cfg.joint_limit_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "foot_slip": foot_slip * self.cfg.foot_slip_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "tetrapod_contact": tetrapod_contact * self.cfg.tetrapod_contact_reward_scale * self.step_dt,
            "gait_symmetry": gait_symmetry * self.cfg.gait_symmetry_reward_scale * self.step_dt,
            "foot_clearance": foot_clearance * self.cfg.foot_clearance_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps_since_reset += 1
        self._step_count += 1
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # fell over: base-centre too low (collapsed) or tilted past ~60 deg
        too_low = self._robot.data.root_pos_w[:, 2] < 0.035
        tilted = self._robot.data.projected_gravity_b[:, 2] > -0.5
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        died = too_low | tilted | base_contact
        # suppress deaths during the post-reset settling transient so the policy
        # gets to act before a spawn wobble can terminate the episode
        died = died & (self._steps_since_reset > self.cfg.settle_steps)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0

        # straight-line forward command (vy=0, yaw=0). Stand-first curriculum:
        # vx=0 until curriculum_stand_steps, then ramp the upper vx command up.
        n = len(env_ids)
        self._commands[env_ids] = 0.0
        if self._step_count < self.cfg.curriculum_stand_steps:
            vx_max = 0.0
        else:
            frac = min(1.0, (self._step_count - self.cfg.curriculum_stand_steps) / self.cfg.curriculum_ramp_steps)
            vx_max = frac * self.cfg.cmd_vx_range[1]
        if vx_max > 0.0:
            self._commands[env_ids, 0] = torch.empty(n, device=self.device).uniform_(0.0, vx_max)

        # reset robot state with small joint noise for exploration
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_pos += torch.empty_like(joint_pos).uniform_(-0.05, 0.05)
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
