# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rough-terrain generator config for the Hexabot (Milestone 1).

Restricted to the terrain a BLIND, reactive proprioceptive policy can ultimately
reproduce, so the privileged teacher trained here distills cleanly into the
proprioceptive student of the next milestone (see hexabot_rough_env_cfg.py):

    slopes  |  random rough/noisy ground  |  modest steps (boxes)  |  low stairs
    + MIXED tiles that overlay these on each other (bumps on a slope, uneven
      steps, small boxes on a ramp) — the main difficulty lever (this revision).

EXCLUDED on purpose (would need look-before-you-step foothold precision a blind
student can never recover): stepping stones across gaps, narrow beams, holes.
Every feature here is CONTINUOUS and crossable — a wrong step is feelable and
recoverable, never an instant fall. The mixed tiles only SUM small height fields
on top of one another, so they stay within the same recoverable envelope.

Everything is scaled DOWN hard for the hexapod, which is tiny: body centre stands
~72 mm, foot span ~0.59 m, top speed ~0.30 m/s. Default Isaac Lab locomotion
terrains (e.g. 5-23 cm stairs) are sized for ANYmal and would be impassable —
heights/slopes here are a fraction of the standing height. `curriculum=True` makes
row index == difficulty level, so level 0 is ~flat and difficulty ramps with the
terrain-level curriculum driven from the env (HexabotRoughEnv._reset_idx).

ALL difficulty knobs are named constants at the top of this file (old values noted
inline) so they are easy to tune. The mixed tiles are custom height-field functions
(`sloped_rough_terrain`, `stairs_rough_terrain`, `sloped_boxes_terrain`) that
compose the stock Isaac Lab height-field generators by summing their height arrays.
"""

import numpy as np
import scipy.interpolate as interpolate

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import hf_terrains
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.utils import configclass

# ============================================================================
# TUNABLE DIFFICULTY PARAMETERS  (edit here; previous values noted as `was ...`)
# All heights/slopes are at the HARDEST curriculum level (difficulty == 1); they
# scale linearly to ~flat at level 0. Keep them blind-feasible (a wrong step must
# be feelable and recoverable, never an instant fall).
# ============================================================================

# ---- slopes (pyramid up / inverted pyramid down) ---------------------------
SLOPE_MAX = 0.20          # was 0.15   rise/run at the hardest level (~11 deg)
SLOPE_PLATFORM = 0.5      # was 0.6    flat top width [m] (smaller -> more slope, also the spawn pad)

# ---- standalone random rough ground ----------------------------------------
ROUGH_NOISE_MAX = 0.035   # was 0.03   peak bump height [m]
ROUGH_NOISE_STEP = 0.005  #            height quantum [m]
ROUGH_DOWNSAMPLE = 0.05   # was None (=cell 0.025); coarser -> rolling/feelable, not micro-spikes

# ---- standalone discrete boxes (mesh grid of low blocks) -------------------
BOX_HEIGHT_MAX = 0.035    # was 0.025  block height at hardest level [m]
BOX_GRID_WIDTH = 0.30     # unchanged  must NOT evenly divide tile size (2.0): border = 2.0 - 6*0.3 = 0.2
BOX_PLATFORM = 0.5        # was 0.5    central flat spawn pad [m]

# ---- stairs (pyramid up / inverted pyramid down) ---------------------------
STAIR_HEIGHT_MAX = 0.04   # was 0.03   step rise at hardest level [m]
STAIR_WIDTH = 0.22        # was 0.18   tread depth [m] (wider so a claw foot fits a step)
STAIR_PLATFORM = 0.6      # was 0.6    central flat spawn pad [m]

# ---- MIXED tiles: small noise overlaid ON TOP of a slope / stairs ----------
# A perturbation, deliberately smaller than the standalone rough so the composite
# stays within the recoverable envelope. Symmetric (+/-), rolling (feelable).
MIXED_NOISE_MAX = 0.02         # +/- overlay bump on the base feature [m]
MIXED_NOISE_STEP = 0.005       # overlay height quantum [m]
MIXED_NOISE_DOWNSAMPLE = 0.06  # overlay cell [m] -> rolling undulation, not spikes

# ---- MIXED tile: small flat-topped boxes overlaid on a slope ---------------
# "small steps on a slope": one-sided (up only) low plateaus on the ramp.
SLOPE_BOX_MAX = 0.025          # box height on the ramp at hardest level [m]
SLOPE_BOX_DOWNSAMPLE = 0.18    # box footprint [m] (~ a foot-ish patch)


# ============================================================================
# Helpers for the custom (mixed) height-field terrains
# ============================================================================
def _difficulty_noise(
    cfg: HfTerrainBaseCfg,
    difficulty: float,
    max_amp: float,
    downsample: float,
    step: float = MIXED_NOISE_STEP,
    smooth: bool = True,
    one_sided: bool = False,
) -> np.ndarray:
    """Difficulty-scaled height noise as an int16 array matching the cfg grid.

    Amplitude scales linearly with `difficulty` (0 -> flat) so it follows the
    curriculum. `smooth=True` spline-upsamples a coarse random grid into rolling,
    feelable undulation; `smooth=False` does a nearest-neighbour upsample giving
    flat-topped boxes. `one_sided` keeps the noise non-negative (boxes go up only).
    Shape matches `hf_terrains.*_terrain.__wrapped__(difficulty, cfg)` so the
    arrays can be summed directly.
    """
    width_px = int(cfg.size[0] / cfg.horizontal_scale)
    length_px = int(cfg.size[1] / cfg.horizontal_scale)
    amp = max(0.0, float(difficulty)) * max_amp
    if amp <= 0.0:
        return np.zeros((width_px, length_px), dtype=np.int16)
    h_max = max(1, int(amp / cfg.vertical_scale))
    h_step = max(1, int(step / cfg.vertical_scale))
    levels = np.arange(0, h_max + h_step, h_step) if one_sided else np.arange(-h_max, h_max + h_step, h_step)
    # coarse random grid, then upsample to the full cell resolution
    wd = max(2, int(cfg.size[0] / downsample))
    ld = max(2, int(cfg.size[1] / downsample))
    grid = np.random.choice(levels, size=(wd, ld)).astype(np.float64)
    if smooth and wd >= 4 and ld >= 4:
        xs = np.linspace(0.0, 1.0, wd)
        ys = np.linspace(0.0, 1.0, ld)
        k = int(min(3, wd - 1, ld - 1))
        spline = interpolate.RectBivariateSpline(xs, ys, grid, kx=k, ky=k)
        z = spline(np.linspace(0.0, 1.0, width_px), np.linspace(0.0, 1.0, length_px))
    else:
        xi = np.linspace(0, wd - 1, width_px).round().astype(int)
        yi = np.linspace(0, ld - 1, length_px).round().astype(int)
        z = grid[np.ix_(xi, yi)]
    return np.rint(z).astype(np.int16)


def _clear_center(hf: np.ndarray, cfg: HfTerrainBaseCfg, platform_width: float) -> np.ndarray:
    """Zero a central square (the base feature's flat platform / spawn pad) so the
    overlay doesn't put a bump or box edge right under the spawned robot."""
    width_px, length_px = hf.shape
    pw = int(platform_width / cfg.horizontal_scale / 2)
    cx, cy = width_px // 2, length_px // 2
    hf[cx - pw : cx + pw, cy - pw : cy + pw] = 0
    return hf


@height_field_to_mesh
def sloped_rough_terrain(difficulty: float, cfg: "HfSlopedRoughTerrainCfg") -> np.ndarray:
    """A pyramid slope with rolling rough noise overlaid (bumps on a ramp)."""
    base = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    noise = _difficulty_noise(cfg, difficulty, cfg.noise_max, cfg.noise_downsample, smooth=True)
    noise = _clear_center(noise, cfg, cfg.platform_width)
    return (base + noise).astype(np.int16)


@height_field_to_mesh
def stairs_rough_terrain(difficulty: float, cfg: "HfStairsRoughTerrainCfg") -> np.ndarray:
    """Pyramid stairs with rolling rough noise overlaid (uneven, feelable steps)."""
    base = hf_terrains.pyramid_stairs_terrain.__wrapped__(difficulty, cfg)
    noise = _difficulty_noise(cfg, difficulty, cfg.noise_max, cfg.noise_downsample, smooth=True)
    noise = _clear_center(noise, cfg, cfg.platform_width)
    return (base + noise).astype(np.int16)


@height_field_to_mesh
def sloped_boxes_terrain(difficulty: float, cfg: "HfSlopedBoxesTerrainCfg") -> np.ndarray:
    """A pyramid slope with small flat-topped boxes overlaid (small steps on a slope)."""
    base = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    boxes = _difficulty_noise(
        cfg, difficulty, cfg.box_max, cfg.box_downsample, smooth=False, one_sided=True
    )
    boxes = _clear_center(boxes, cfg, cfg.platform_width)
    return (base + boxes).astype(np.int16)


# ============================================================================
# Config classes for the custom mixed terrains. They subclass HfTerrainBaseCfg so
# the generator injects horizontal_scale / vertical_scale / slope_threshold / size
# (see TerrainGenerator: it pushes those onto every HfTerrainBaseCfg sub-terrain).
# ============================================================================
@configclass
class HfSlopedRoughTerrainCfg(HfTerrainBaseCfg):
    function = sloped_rough_terrain
    slope_range: tuple[float, float] = (0.0, SLOPE_MAX)
    platform_width: float = SLOPE_PLATFORM
    inverted: bool = False
    noise_max: float = MIXED_NOISE_MAX
    noise_downsample: float = MIXED_NOISE_DOWNSAMPLE


@configclass
class HfStairsRoughTerrainCfg(HfTerrainBaseCfg):
    function = stairs_rough_terrain
    step_height_range: tuple[float, float] = (0.0, STAIR_HEIGHT_MAX)
    step_width: float = STAIR_WIDTH
    platform_width: float = STAIR_PLATFORM
    inverted: bool = False
    noise_max: float = MIXED_NOISE_MAX
    noise_downsample: float = MIXED_NOISE_DOWNSAMPLE


@configclass
class HfSlopedBoxesTerrainCfg(HfTerrainBaseCfg):
    function = sloped_boxes_terrain
    slope_range: tuple[float, float] = (0.0, SLOPE_MAX)
    platform_width: float = SLOPE_PLATFORM
    inverted: bool = False
    box_max: float = SLOPE_BOX_MAX
    box_downsample: float = SLOPE_BOX_DOWNSAMPLE


# ============================================================================
# The terrain generator. Proportions are rebalanced TOWARD the harder mixed tiles
# (mixed ~0.54 vs pure ~0.36); the generator normalises by their sum.
# ============================================================================
# Small tiles: the robot is slow (0.30 m/s x 12 s episode), so a 2 m tile is plenty
# and keeps the per-level walked-distance promotion threshold reachable within an
# episode. NB the harder terrain shortens distance-per-episode, so the curriculum
# promote/demote thresholds were loosened in HexabotRoughEnv._update_terrain_levels
# (cfg.terrain_promote_frac / terrain_demote_frac) so the top levels stay reachable.
HEXABOT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(2.0, 2.0),
    border_width=2.0,
    num_rows=10,          # 10 difficulty levels (curriculum rows)
    num_cols=10,          # 10 variations per level
    horizontal_scale=0.025,   # fine cells (robot footprint is small)
    vertical_scale=0.0025,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,       # row = difficulty; enables the terrain-level curriculum
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # -------- pure single-feature tiles (reduced share, harder ranges) --------
        # gentle->moderate slopes up/down (blind-friendly, just tilt)
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, SLOPE_MAX),  # was prop 0.2, slope max 0.15
            platform_width=SLOPE_PLATFORM, border_width=0.25,
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, SLOPE_MAX),  # was prop 0.2, slope max 0.15
            platform_width=SLOPE_PLATFORM, border_width=0.25,
        ),
        # random rough / noisy ground
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.005, ROUGH_NOISE_MAX),  # was prop 0.3, max 0.03
            noise_step=ROUGH_NOISE_STEP, downsampled_scale=ROUGH_DOWNSAMPLE, border_width=0.25,
        ),
        # modest discrete steps (a grid of low blocks)
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.08, grid_width=BOX_GRID_WIDTH,           # was prop 0.15
            grid_height_range=(0.0, BOX_HEIGHT_MAX),             # was (0.005, 0.025)
            platform_width=BOX_PLATFORM,
        ),
        # low stairs up and down. Keep a small positive step floor (0.008): the MESH
        # stairs build trimesh boxes, and a near-zero step height makes degenerate boxes.
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.04, step_height_range=(0.008, STAIR_HEIGHT_MAX),  # was prop 0.075, (0.01,0.03)
            step_width=STAIR_WIDTH, platform_width=STAIR_PLATFORM, border_width=0.25, holes=False,
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.04, step_height_range=(0.008, STAIR_HEIGHT_MAX),  # was prop 0.075, (0.01,0.03)
            step_width=STAIR_WIDTH, platform_width=STAIR_PLATFORM, border_width=0.25, holes=False,
        ),
        # -------- MIXED-FEATURE tiles (the main difficulty lever; NEW) -------------
        # rolling bumps overlaid on a slope (up / down)
        "sloped_rough": HfSlopedRoughTerrainCfg(
            proportion=0.12, inverted=False, border_width=0.25,
        ),
        "sloped_rough_inv": HfSlopedRoughTerrainCfg(
            proportion=0.12, inverted=True, border_width=0.25,
        ),
        # uneven steps: stairs with rolling noise overlaid (up / down)
        "stairs_rough": HfStairsRoughTerrainCfg(
            proportion=0.10, inverted=False, border_width=0.25,
        ),
        "stairs_rough_inv": HfStairsRoughTerrainCfg(
            proportion=0.10, inverted=True, border_width=0.25,
        ),
        # small flat-topped boxes (little steps) on a slope (up / down)
        "sloped_boxes": HfSlopedBoxesTerrainCfg(
            proportion=0.05, inverted=False, border_width=0.25,
        ),
        "sloped_boxes_inv": HfSlopedBoxesTerrainCfg(
            proportion=0.05, inverted=True, border_width=0.25,
        ),
    },
)
"""Blind-feasible rough terrains for the Hexabot, scaled to the 72 mm hexapod."""
