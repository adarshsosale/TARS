# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Rough-terrain generator config for the Hexabot (Milestone 1).

Restricted to the terrain a BLIND, reactive proprioceptive policy can ultimately
reproduce, so the privileged teacher trained here distills cleanly into the
proprioceptive student of the next milestone (see hexabot_rough_env_cfg.py):

    slopes  |  random rough/noisy ground  |  modest steps (boxes)  |  low stairs

EXCLUDED on purpose (would need look-before-you-step foothold precision a blind
student can never recover): stepping stones across gaps, narrow beams, holes.

Everything is scaled DOWN hard for the hexapod, which is tiny: body centre stands
~72 mm, foot span ~0.59 m, top speed ~0.30 m/s. Default Isaac Lab locomotion
terrains (e.g. 5-23 cm stairs) are sized for ANYmal and would be impassable —
heights/slopes here are a fraction of the standing height. `curriculum=True` makes
row index == difficulty level, so level 0 is ~flat and difficulty ramps with the
terrain-level curriculum driven from the env (HexabotRoughEnv._reset_idx).
"""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains import TerrainGeneratorCfg

# Small tiles: the robot is slow (0.30 m/s x 12 s episode), so a 2 m tile is plenty
# and keeps the per-level walked-distance promotion threshold (size/2 = 1.0 m)
# reachable within an episode.
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
        # gentle slopes up/down — biggest share (blind-friendly, just tilt)
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.15), platform_width=0.6, border_width=0.25
        ),
        "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, 0.15), platform_width=0.6, border_width=0.25
        ),
        # random rough / noisy ground (sub-cm to ~3 cm bumps)
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.005, 0.03), noise_step=0.005, border_width=0.25
        ),
        # modest discrete steps (a grid of low blocks). grid_width must NOT evenly
        # divide size (else the auto border width is 0): 2.0 - floor(2.0/0.3)*0.3 = 0.2.
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.3, grid_height_range=(0.005, 0.025), platform_width=0.5
        ),
        # low stairs up and down
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.075, step_height_range=(0.01, 0.03), step_width=0.18,
            platform_width=0.6, border_width=0.25, holes=False,
        ),
        "stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.075, step_height_range=(0.01, 0.03), step_width=0.18,
            platform_width=0.6, border_width=0.25, holes=False,
        ),
    },
)
"""Blind-feasible rough terrains for the Hexabot, scaled to the 72 mm hexapod."""
