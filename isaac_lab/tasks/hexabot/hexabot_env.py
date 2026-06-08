# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply

from .cpg import HexabotCPG
from .hexabot_env_cfg import HexabotFlatEnvCfg


class HexabotEnv(DirectRLEnv):
    """Locomotion layer (PPO, continuous control).

    Observation is PROPRIOCEPTIVE-ONLY (no base linear velocity — not measurable on
    the real robot): projected gravity, base angular velocity, the velocity command
    (the frozen Navigation->Locomotion interface), joint pos/vel, previous action,
    and the CPG phase. A dormant zero-width height-scan block is appended last (the
    exteroceptive seam).

    Action MODULATES a tripod-seeded CPG (see cpg.py) rather than emitting joint
    offsets: per-leg [d_freq, d_coxa_amp, d_lift]. Zero action == the analytical
    tripod gait scaled to the commanded speed; the analytical gait is therefore a
    *reference* (BC warm-start + annealing imitation reward), never a residual basis.

    Domain randomization is ON from the start: friction / mass / actuator-gain
    spread (event manager, see EventCfg) plus actuator latency, control-rate jitter
    and IMU/observation noise applied here in the step loop (see DomainRandCfg).
    """

    cfg: HexabotFlatEnvCfg

    def __init__(self, cfg: HexabotFlatEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # commands: [vx, vy, yaw_rate] — the frozen interface. vy & yaw pinned to 0 (straight-line).
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        # nominal-amplitude speed scale (set each pre-physics from the command)
        self._speed_scale = torch.zeros(self.num_envs, device=self.device)

        # control steps elapsed since each env was last reset (settle grace)
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # global control-step counter driving the stand-first curriculum + imitation anneal
        self._step_count = 0

        # --- CPG (the action mechanism) ---------------------------------------
        self._cpg = HexabotCPG(self._robot.data.joint_names, self.num_envs, self.device, self.cfg)
        # symmetry permutations exposed for RslRlSymmetryCfg (see symmetry.py)
        self._act_mirror_idx = self._cpg.action_mirror_idx()   # 18, leg-swap, no sign
        self._cpg_mirror_idx = self._cpg.phase_mirror_idx()    # 12, leg-swap, no value change

        # tibia-local offset to the claw tip (for the foot-below-belly reward)
        self._claw_offset_local = torch.tensor(
            [self.cfg.claw_offset, 0.0, 0.0], device=self.device
        ).view(1, 1, 3)
        # world +x unit (spawn forward heading) for the heading-hold reward
        self._world_x = torch.tensor([1.0, 0.0, 0.0], device=self.device)

        # --- signal-side domain randomization state ---------------------------
        dr = self.cfg.domain_rand
        self._latency_max = int(dr.actuator_latency_steps[1]) if dr.enabled else 0
        self._last_applied = self._robot.data.default_joint_pos.clone()
        if self._latency_max > 0:
            self._cmd_buf = self._last_applied.unsqueeze(0).repeat(self._latency_max + 1, 1, 1)
            self._buf_ptr = 0
            self._latency_k = torch.randint(
                int(dr.actuator_latency_steps[0]), self._latency_max + 1, (self.num_envs,), device=self.device
            )
        self._processed_actions = self._last_applied.clone()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "forward_progress",
                "stationary_penalty",
                "track_ang_vel_z_exp",
                "lateral_vel_l2",
                "yaw_rate_l2",
                "heading",
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
                "gait_phase",
                "foot_plant",
                "imitation",
            ]
        }

        # contact-sensor body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        self._feet_ids, feet_names = self._contact_sensor.find_bodies("tibia_.*")  # the 6 claw feet

        # left<->right mirror permutation over the 6 feet (in self._feet_ids order),
        # for the symmetric-gait reward.
        def _mirror(nm: str) -> str:
            prefix, sp = nm.rsplit("_", 1)              # 'tibia', 'lf'
            side = "r" if sp[0] == "l" else "l"
            return f"{prefix}_{side}{sp[1:]}"

        _name_to_local = {nm: i for i, nm in enumerate(feet_names)}
        self._feet_mirror_idx = torch.tensor(
            [_name_to_local[_mirror(nm)] for nm in feet_names], device=self.device, dtype=torch.long
        )
        # map each foot (contact-sensor order) -> its index in the CPG's LEG_ORDER,
        # so the CPG per-leg phase can be reindexed to compare with foot contacts.
        from .cpg import LEG_ORDER
        self._feet_to_legorder = torch.tensor(
            [LEG_ORDER.index(nm.rsplit("_", 1)[1]) for nm in feet_names], device=self.device, dtype=torch.long
        )

        # --- left-right (sagittal) joint symmetry, for PPO symmetry augmentation ---
        # _jt_mirror_idx points each joint at its L<->R partner; _jt_mirror_sign flips
        # the coxa (yaw) joints (femur/tibia pitch joints are handedness-free).
        joint_names = self._robot.data.joint_names

        def _jmirror(nm: str) -> str:  # 'coxa_lf' -> 'coxa_rf'
            prefix, leg = nm.rsplit("_", 1)
            side = "r" if leg[0] == "l" else "l"
            return f"{prefix}_{side}{leg[1:]}"

        _jname_to_idx = {nm: i for i, nm in enumerate(joint_names)}
        self._jt_mirror_idx = torch.tensor(
            [_jname_to_idx[_jmirror(nm)] for nm in joint_names], device=self.device, dtype=torch.long
        )
        self._jt_mirror_sign = torch.tensor(
            [-1.0 if nm.startswith("coxa_") else 1.0 for nm in joint_names], device=self.device
        )

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(["base_link", "coxa_.*", "femur_.*"])
        # robot-frame body indices for the feet (for foot-velocity / slip / clearance)
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

    # ------------------------------------------------------------------ helpers
    def _imitation_weight(self) -> float:
        """Annealing weight for the imitation reward (1.0 -> 0.0 over the curriculum)."""
        if self._step_count < self.cfg.curriculum_stand_steps:
            return 1.0
        prog = (self._step_count - self.cfg.curriculum_stand_steps) / self.cfg.imitation_anneal_steps
        return max(0.0, 1.0 - prog)

    def _apply_latency(self, target: torch.Tensor) -> torch.Tensor:
        """Serve a per-env command delayed by k control steps (actuator latency DR)."""
        if self._latency_max == 0:
            return target
        self._cmd_buf[self._buf_ptr] = target
        idx = (self._buf_ptr - self._latency_k) % (self._latency_max + 1)
        delayed = self._cmd_buf[idx, torch.arange(self.num_envs, device=self.device)]
        self._buf_ptr = (self._buf_ptr + 1) % (self._latency_max + 1)
        return delayed

    # ------------------------------------------------------------------ stepping
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        dr = self.cfg.domain_rand

        # control-rate jitter on the CPG-advance dt
        if dr.enabled and dr.control_dt_jitter > 0.0:
            jit = 1.0 + (torch.rand(self.num_envs, 1, device=self.device) * 2.0 - 1.0) * dr.control_dt_jitter
            eff_dt = self.step_dt * jit
        else:
            eff_dt = self.step_dt
        self._cpg.step(self._actions, eff_dt)

        # nominal stride amplitude tracks the command magnitude (0 -> stand). Yaw
        # activates the gait too, so the robot can turn even at zero forward speed.
        vx_act = self._commands[:, 0].abs() / self.cfg.cpg_v_ref
        yaw_act = self._commands[:, 2].abs() / self.cfg.cpg_yaw_ref
        self._speed_scale = torch.maximum(vx_act, yaw_act).clamp(0.0, 1.2)
        raw = self._cpg.joint_targets(self._actions, self._speed_scale)
        # clamp to soft joint limits (CPG output is absolute joint targets)
        lo = self._robot.data.soft_joint_pos_limits[..., 0]
        hi = self._robot.data.soft_joint_pos_limits[..., 1]
        target = torch.clamp(raw, lo, hi)

        # actuator latency, then a small chance of a dropped control tick (hold last)
        target = self._apply_latency(target)
        if dr.enabled and dr.hold_prob > 0.0:
            hold = torch.rand(self.num_envs, 1, device=self.device) < dr.hold_prob
            target = torch.where(hold, self._last_applied, target)
        self._last_applied = target
        self._processed_actions = target

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        grav = self._robot.data.projected_gravity_b
        angv = self._robot.data.root_ang_vel_b
        jpos = self._robot.data.joint_pos - self._robot.data.default_joint_pos
        jvel = self._robot.data.joint_vel
        # IMU / observation noise (DR)
        dr = self.cfg.domain_rand
        if dr.enabled:
            grav = grav + torch.randn_like(grav) * dr.noise_gravity
            angv = angv + torch.randn_like(angv) * dr.noise_ang_vel
            jpos = jpos + torch.randn_like(jpos) * dr.noise_joint_pos
            jvel = jvel + torch.randn_like(jvel) * dr.noise_joint_vel

        parts = [
            grav,                       # 3
            angv,                       # 3
            self._commands,             # 3 (frozen interface)
            jpos,                       # 18
            jvel,                       # 18
            self._actions,              # 18 (previous action)
            self._cpg.phase_obs(),      # 12 (CPG per-leg sin/cos)
        ]
        # dormant exteroceptive seam: a height-scan block plugs in here later
        if self.cfg.n_height_scan > 0:
            parts.append(torch.zeros(self.num_envs, self.cfg.n_height_scan, device=self.device))
        return {"policy": torch.cat(parts, dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # forward (x) velocity tracking; vy command is 0
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)
        # linear forward-progress reward (BODY-frame +x so "forward" follows the
        # robot's heading — required for turning; on straight episodes the heading
        # term below keeps body +x aligned to world +x). 0 at standstill, capped at cmd.
        forward_progress = torch.clamp(
            torch.minimum(self._robot.data.root_lin_vel_b[:, 0], self._commands[:, 0]), min=0.0
        )
        # yaw-rate tracking (command 0 -> go straight)
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)
        lateral_vel = torch.square(self._robot.data.root_lin_vel_b[:, 1])
        # command-relative yaw-rate penalty: damps drift without fighting a commanded turn
        yaw_rate_l2 = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # heading hold: body +x axis should keep pointing along world +x
        fwd_axis_w = quat_apply(self._robot.data.root_quat_w, self._world_x.expand(self.num_envs, 3))
        heading_err = torch.square(fwd_axis_w[:, 1])
        lateral_pos = torch.square(self._robot.data.root_pos_w[:, 1] - self._terrain.env_origins[:, 1])
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        base_height_error = torch.square(self._robot.data.root_pos_w[:, 2] - self.cfg.target_height)
        belly_height = self._robot.data.root_pos_w[:, 2] - self.cfg.belly_half_thickness
        belly_clearance = torch.clamp(self.cfg.belly_clearance_min - belly_height, min=0.0)
        # foot-below-belly support: claw tip world z via tibia pose + local claw offset
        feet_quat = self._robot.data.body_quat_w[:, self._feet_body_ids, :]
        feet_pos = self._robot.data.body_pos_w[:, self._feet_body_ids, :]
        claw_w = feet_pos + quat_apply(feet_quat, self._claw_offset_local.expand_as(feet_pos))
        foot_z = claw_w[..., 2]
        support = torch.clamp(belly_height.unsqueeze(1) - foot_z, min=0.0, max=self.cfg.support_target)
        foot_support = support.mean(dim=1)
        alive = torch.ones(self.num_envs, device=self.device)
        # effort / smoothness
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
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
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        feet_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0
        )
        feet_planar_speed = torch.sum(
            torch.square(self._robot.data.body_lin_vel_w[:, self._feet_body_ids, :2]), dim=-1
        )
        foot_slip = torch.sum(feet_planar_speed * feet_contact, dim=1)
        # --- coordinated symmetric tetrapod gait (only while commanded to move) ---
        # "moving" now includes a yaw command, so turn-in-place episodes still get the
        # gait-shaping rewards. straight_mask flags pure-straight (no-yaw) episodes,
        # where the go-straight terms (heading / lateral) apply.
        moving = ((torch.norm(self._commands[:, :2], dim=1) > 0.1) | (self._commands[:, 2].abs() > 0.05)).float()
        straight_mask = (self._commands[:, 2].abs() < 1e-3).float()
        vel_shortfall = torch.clamp(
            self._commands[:, 0] - self._robot.data.root_lin_vel_b[:, 0], min=0.0
        ) * moving
        n_contact = feet_contact.float().sum(dim=1)
        tetrapod_contact = torch.exp(-torch.square(n_contact - 4.0)) * moving
        sym_match = (feet_contact == feet_contact[:, self._feet_mirror_idx]).float().mean(dim=1)
        gait_symmetry = sym_match * (n_contact < 6).float() * moving
        swing = (~feet_contact).float()
        foot_clearance = torch.sum(
            torch.clamp(foot_z, min=0.0, max=self.cfg.foot_clearance_target) * swing, dim=1
        ) * moving
        # phase schedule comes from the CPG now: a leg's local phase p>=0.5 is its swing
        # half (matches tripod_gait.gait_pose). Reindex CPG phase into foot order.
        p_feet = (self._cpg.theta / (2.0 * torch.pi))[:, self._feet_to_legorder]
        sched_swing = p_feet >= 0.5
        sched_stance = ~sched_swing
        n_sched_swing = sched_swing.float().sum(dim=1).clamp(min=1.0)
        n_sched_stance = sched_stance.float().sum(dim=1).clamp(min=1.0)
        swing_correct = ((~feet_contact) & sched_swing).float().sum(dim=1) / n_sched_swing
        stance_violation = ((~feet_contact) & sched_stance).float().sum(dim=1) / n_sched_stance
        gait_phase = (swing_correct - stance_violation) * moving
        # foot plant angle: stance feet should point steeply DOWN (claw digs in for grip)
        foot_downness = torch.clamp((feet_pos[..., 2] - foot_z) / self.cfg.claw_offset, min=0.0, max=1.0)
        foot_plant = torch.sum(
            torch.clamp(foot_downness, max=self.cfg.foot_plant_target) * feet_contact.float(), dim=1
        ) * moving
        # undesired contacts (body / coxa / femur hitting the ground)
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)

        # --- annealing imitation reward (analytical gait as a reference) ----------
        # Deviation of the policy's CPG joint targets from the zero-action (analytical)
        # targets at the SAME phase + speed scale. 0 during the stand phase (amplitude
        # gated to 0). Weight anneals to 0 over the flat-ground curriculum.
        imit_w = self._imitation_weight()
        cpg_t = self._cpg.joint_targets(self._actions, self._speed_scale)
        nominal_t = self._cpg.nominal_joint_targets(self._speed_scale)
        imitation = torch.sum(torch.square(cpg_t - nominal_t), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "forward_progress": forward_progress * self.cfg.forward_progress_reward_scale * self.step_dt,
            "stationary_penalty": vel_shortfall * self.cfg.stationary_penalty_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lateral_vel_l2": lateral_vel * straight_mask * self.cfg.lateral_vel_reward_scale * self.step_dt,
            "yaw_rate_l2": yaw_rate_l2 * self.cfg.yaw_rate_l2_reward_scale * self.step_dt,
            "heading": heading_err * moving * straight_mask * self.cfg.heading_reward_scale * self.step_dt,
            "lateral_pos_l2": lateral_pos * straight_mask * self.cfg.lateral_pos_reward_scale * self.step_dt,
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
            "gait_phase": gait_phase * self.cfg.gait_phase_reward_scale * self.step_dt,
            "foot_plant": foot_plant * self.cfg.foot_plant_reward_scale * self.step_dt,
            "imitation": imitation * self.cfg.imitation_reward_scale * imit_w * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps_since_reset += 1
        self._step_count += 1
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        too_low = self._robot.data.root_pos_w[:, 2] < 0.035
        tilted = self._robot.data.projected_gravity_b[:, 2] > -0.5
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        )
        died = too_low | tilted | base_contact
        # suppress deaths during the post-reset settling transient
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
        # reset the CPG phases (re-seed at the tripod offsets + a random global phase)
        self._cpg.reset(env_ids)
        # reset signal-side DR state for these envs
        dr = self.cfg.domain_rand
        default_q = self._robot.data.default_joint_pos[env_ids]
        self._last_applied[env_ids] = default_q
        self._processed_actions[env_ids] = default_q
        if self._latency_max > 0:
            self._cmd_buf[:, env_ids] = default_q.unsqueeze(0)
            self._latency_k[env_ids] = torch.randint(
                int(dr.actuator_latency_steps[0]), self._latency_max + 1, (len(env_ids),), device=self.device
            )

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
        # turning curriculum: after the forward gait is solid, a fraction of episodes
        # get a random yaw command (magnitude ramps in); the rest stay straight so
        # straight-line quality is preserved.
        if self._step_count >= self.cfg.yaw_curriculum_start_steps:
            yfrac = min(1.0, (self._step_count - self.cfg.yaw_curriculum_start_steps) / self.cfg.yaw_curriculum_ramp_steps)
            yaw_max = yfrac * self.cfg.cmd_yaw_range[1]
            turn = torch.rand(n, device=self.device) < self.cfg.yaw_command_prob
            yaw_cmd = torch.empty(n, device=self.device).uniform_(-yaw_max, yaw_max)
            self._commands[env_ids, 2] = torch.where(turn, yaw_cmd, torch.zeros_like(yaw_cmd))

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
