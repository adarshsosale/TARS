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

        # --- Phase B: front/middle/rear body regions over the scan rays ----------
        # Derived from the live ray pattern's body-frame x (the sensor forward offset
        # is already baked into ray_starts), so it tracks any change to the scan grid
        # — including Phase C's forward extension, whose lookahead rays (x > body
        # half-span) are excluded from all three regions by scan_region_masks.
        from .regions import scan_region_masks

        ray_x = rs[:, 0]
        self._scan_front_mask, self._scan_mid_mask, self._scan_rear_mask = scan_region_masks(ray_x, half=0.20)
        # each foot's region row (front .f -> 0, mid .m -> 1, rear .r -> 2), in the
        # SAME order as self._feet_body_ids (the claw_w order used in the reward).
        _, foot_body_names = self._robot.find_bodies("tibia_.*")
        _region_of = {"f": 0, "m": 1, "r": 2}
        self._foot_region = torch.tensor(
            [_region_of[nm.rsplit("_", 1)[1][1]] for nm in foot_body_names],
            device=self.device, dtype=torch.long,
        )
        # per-region ground heights, refreshed each step in _measure_region_heights;
        # initialised flat so any pre-first-step read is a no-op.
        self._ground_height_front = torch.zeros(self.num_envs, device=self.device)
        self._ground_height_mid = torch.zeros(self.num_envs, device=self.device)
        self._ground_height_rear = torch.zeros(self.num_envs, device=self.device)

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

    def _measure_region_heights(self):
        """Per-region (front/mid/rear) median ground height from the scanner [m].

        Phase B: a single body-wide median is a fiction where the body spans a step
        edge. Here each foot row reads the ground under ITS region. Robust to rays
        missing the mesh (inf -> excluded); a region with zero finite hits falls back
        to the all-ray median. `self._ground_height` is kept as the MIDDLE region (the
        ground directly under the torso) so the inherited body-wide terms and the
        too-low death stay terrain-relative without per-foot ambiguity.
        """
        hits_z = self._height_scanner.data.ray_hits_w[..., 2]   # (N, n_rays)
        finite = torch.isfinite(hits_z)
        nanv = torch.full_like(hits_z, float("nan"))
        all_med = torch.nan_to_num(torch.nanmedian(torch.where(finite, hits_z, nanv), dim=1).values, nan=0.0)

        def _region(mask: torch.Tensor) -> torch.Tensor:
            v = torch.where(mask.unsqueeze(0) & finite, hits_z, nanv)
            m = torch.nanmedian(v, dim=1).values
            return torch.where(torch.isnan(m), all_med, m)

        self._ground_height_front = _region(self._scan_front_mask)
        self._ground_height_mid = _region(self._scan_mid_mask)
        self._ground_height_rear = _region(self._scan_rear_mask)
        self._ground_height = self._ground_height_mid

        # --- Phase E: local terrain protrusion -> anticipatory posture targets ----
        # q90 of ALL rays (body + the +0.40 m lookahead) above the mid-region ground
        # = the tallest obstacle the belly must clear soon. Sets the ride-height
        # raise (belly_clearance / base-height targets in the base reward) and the
        # swing-apex raise BEFORE the robot reaches the obstacle — the scan's
        # forward extension makes this the "sense the ground before committing"
        # signal. q90 (not max) so a single bad ray can't command a posture jump.
        q90 = torch.nanquantile(torch.where(finite, hits_z, nanv).float(), 0.9, dim=1)
        protrusion = torch.clamp(torch.nan_to_num(q90, nan=0.0) - self._ground_height_mid, min=0.0)
        self._height_target_offset = torch.clamp(
            self.cfg.height_raise_gain * protrusion, max=self.cfg.height_raise_max
        )
        self._foot_clearance_offset = torch.clamp(protrusion, max=self.cfg.foot_clearance_raise_max)

    def _get_foot_ground_height(self) -> torch.Tensor:
        """Per-foot ground height (N, n_feet) = each foot row's region median.

        Overrides the flat broadcast in HexabotEnv so foot_clearance is judged against
        the ground under each foot (front/mid/rear), the Phase B accuracy fix.
        """
        heights = torch.stack(
            [self._ground_height_front, self._ground_height_mid, self._ground_height_rear], dim=1
        )  # (N, 3)
        return heights[:, self._foot_region]   # (N, n_feet)

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
        n_proprio = self.cfg.observation_space - self.cfg.n_height_scan
        obs["policy"] = torch.cat([obs["policy"][:, :n_proprio], scan], dim=-1)
        return obs

    def _get_dones(self):
        # refresh the per-region terrain-relative ground heights (and self._ground_height
        # = middle region) before dones/rewards read them.
        self._measure_region_heights()
        # Phase D: per-env tetrapod_contact weight relaxed with terrain level (no-op at
        # tetrapod_relax_frac=0.0, the B+C baseline). Levels are 0..max_terrain_level-1.
        if self.cfg.tetrapod_relax_frac > 0.0 and self._terrain.terrain_origins is not None:
            lvl = self._terrain.terrain_levels.float()
            lvl_max = max(1, int(self._terrain.max_terrain_level) - 1)
            self._tetrapod_weight_scale = (1.0 - self.cfg.tetrapod_relax_frac * (lvl / lvl_max)).clamp(min=0.0)
        return super()._get_dones()

    # ------------------------------------------------------------------ curriculum
    def _update_terrain_levels(self, env_ids: torch.Tensor):
        """Promote/demote terrain level per env by distance walked.

        Thresholds are a fraction of the tile size (cfg.terrain_promote_frac /
        terrain_demote_frac). The harder mixed terrain shortens distance-per-episode,
        so the old half-tile (1.0 m) promote bar left the top levels unreachable; the
        loosened, command-independent band keeps the mean level climbing to max.
        """
        if not self.cfg.terrain_curriculum_enabled or self._terrain.terrain_origins is None:
            return
        root_xy = self._robot.data.root_pos_w[env_ids, :2]
        origin_xy = self._terrain.env_origins[env_ids, :2]
        distance = torch.norm(root_xy - origin_xy, dim=1)
        # walked past `promote_frac` of a tile -> level up
        move_up = distance > self._terrain_size_x * self.cfg.terrain_promote_frac
        # covered less than `demote_frac` of a tile -> level down
        move_down = (distance < self._terrain_size_x * self.cfg.terrain_demote_frac) & (~move_up)
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
        # Phase E telemetry: is the anticipatory ride-height mode actually engaging?
        # mean commanded raise (m) across all envs right now — should rise with the
        # terrain level once the curriculum reaches protruding obstacles.
        if "log" in self.extras:
            self.extras["log"]["Metrics/height_target_offset"] = torch.mean(
                self._height_target_offset
            ).item()
