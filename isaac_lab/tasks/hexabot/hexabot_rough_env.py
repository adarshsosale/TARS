# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rough-terrain locomotion env (Milestone 1).

Extends `HexabotEnv` with three things and NOTHING else:

  1. a height-scanner sensor (privileged exteroception) added to the scene, read
     each step into the dormant `n_height_scan` observation slot;
  2. a terrain-relative ground height (`self._ground_height`) computed from that
     scanner so the inherited reward / termination terms work on uneven ground;
  3. the terrain-level curriculum (the direct workflow has no curriculum manager,
     so we call `terrain.update_env_origins` ourselves on reset and log the mean
     level — the lead progress metric).

The action mechanism (CPG), the frozen velocity-command interface, the reward
family and the domain randomization are all inherited unchanged.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, RayCaster

from .hexabot_env import HexabotEnv
from .hexabot_rough_env_cfg import HexabotRoughEnvCfg


class HexabotRoughEnv(HexabotEnv):
    cfg: HexabotRoughEnvCfg

    def __init__(self, cfg: HexabotRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Left-right (lateral, body +y) mirror permutation over the height-scan rays,
        # for PPO symmetry augmentation: reflect each ray's y -> -y and find its
        # partner. Built from the actual ray-start pattern so it is independent of the
        # grid's flatten ordering.
        rs = self._height_scanner.ray_starts     # (count, n_rays, 3) after init, else (n_rays, 3)
        rs = rs[0] if rs.dim() == 3 else rs        # all envs share the pattern -> env 0
        xy = rs[:, :2]
        n = xy.shape[0]
        mirror = torch.empty(n, dtype=torch.long, device=self.device)
        tgt = xy.clone()
        tgt[:, 1] = -tgt[:, 1]                       # reflect y
        for i in range(n):
            d = torch.sum((xy - tgt[i]) ** 2, dim=1)
            mirror[i] = torch.argmin(d)
        self._scan_mirror_idx = mirror               # consumed by symmetry.py

        # cache the curriculum promotion threshold (tile size)
        self._terrain_size_x = float(self.cfg.terrain.terrain_generator.size[0])

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        # mirrors HexabotEnv._setup_scene but adds the height scanner (and, when the
        # render flag is set, a tracking camera) BEFORE cloning environments.
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # privileged height scanner (teacher-only exteroception)
        self.cfg.height_scanner.update_period = self.step_dt   # one scan per control step
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        # optional on-demand render camera (off during training)
        self._render_camera = None
        if getattr(self.cfg, "enable_render_camera", False):
            cam_cfg = CameraCfg(
                prim_path="/World/envs/env_0/RenderCam",
                update_period=0.0,
                height=720,
                width=1280,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0, focus_distance=400.0,
                    horizontal_aperture=20.955, clipping_range=(0.05, 1.0e4),
                ),
            )
            # NOTE: do NOT register this in self.scene.sensors. It is a single camera
            # (prim path pinned to env_0 -> num_instances=1), but scene.reset() forwards
            # the full env_ids [0..num_envs-1] to every registered sensor's reset(),
            # which would index its size-1 buffers out of bounds (CUDA device-side
            # assert). The Camera self-initializes via its own timeline PLAY callback,
            # and render_rough.py drives update()/pose manually, so scene registration
            # is unnecessary anyway.
            self._render_camera = Camera(cam_cfg)

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ helpers
    def _measure_ground_height(self) -> torch.Tensor:
        """Median ground height under the robot from the height scanner [m]."""
        hits_z = self._height_scanner.data.ray_hits_w[..., 2]
        # robust to a few rays missing the mesh (inf): use the median of finite hits
        finite = torch.isfinite(hits_z)
        hits_z = torch.where(finite, hits_z, torch.zeros_like(hits_z))
        return torch.median(hits_z, dim=1).values

    def _height_scan_obs(self) -> torch.Tensor:
        """Privileged height-scan observation: per-ray clearance, clipped (N, n_scan)."""
        scanner_z = self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
        hits_z = self._height_scanner.data.ray_hits_w[..., 2]
        hits_z = torch.where(torch.isfinite(hits_z), hits_z, scanner_z)
        scan = scanner_z - hits_z - self.cfg.height_scan_offset
        return torch.clamp(scan, -self.cfg.height_scan_clip, self.cfg.height_scan_clip)

    # ------------------------------------------------------------------ stepping
    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # replace the (zero) dormant tail with the real privileged height scan
        scan = self._height_scan_obs()
        obs["policy"] = torch.cat([obs["policy"][:, : 75], scan], dim=-1)
        return obs

    def _get_dones(self):
        # refresh the terrain-relative ground height before dones/rewards read it
        self._ground_height = self._measure_ground_height()
        return super()._get_dones()

    # ------------------------------------------------------------------ curriculum
    def _update_terrain_levels(self, env_ids: torch.Tensor):
        """Promote/demote terrain level per env by distance walked (terrain_levels_vel)."""
        if not self.cfg.terrain_curriculum_enabled or self._terrain.terrain_origins is None:
            return
        root_xy = self._robot.data.root_pos_w[env_ids, :2]
        origin_xy = self._terrain.env_origins[env_ids, :2]
        distance = torch.norm(root_xy - origin_xy, dim=1)
        # walked past half a tile -> level up
        move_up = distance > self._terrain_size_x / 2.0
        # covered less than half the commanded distance -> level down
        cmd_speed = torch.norm(self._commands[env_ids, :2], dim=1)
        move_down = distance < cmd_speed * self.max_episode_length_s * 0.5
        move_down = move_down & (~move_up)
        self._terrain.update_env_origins(env_ids, move_up, move_down)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            resolved = self._robot._ALL_INDICES
        else:
            resolved = env_ids
        # update terrain levels from the just-finished episode BEFORE the base reset
        # rewrites root pose to the (possibly new) env origin.
        self._update_terrain_levels(resolved)
        super()._reset_idx(env_ids)
        # log the lead progress metric: mean terrain curriculum level
        if self._terrain.terrain_origins is not None and "log" in self.extras:
            self.extras["log"]["Curriculum/terrain_level"] = torch.mean(
                self._terrain.terrain_levels.float()
            ).item()
