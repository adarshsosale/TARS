# Milestone 1.5 — Phase A baseline eval

Held-out, fixed-seed per-terrain-family traversal eval of the rough-terrain teacher,
produced by `isaac_lab/eval_rough.py` (looped over the 7 families by
`scripts/eval_rough.sh`). This is the reference every later phase (B+C, D) is judged
against.

**Success** = survives the episode (ends by timeout, never a death) AND covers
≥ 0.7 × tile size = **1.4 m** of commanded-direction (world +x) distance.
Command fixed to `vx=0.25, vy=0, yaw=0`; seed 42; 256 envs/family; physics- and
signal-side DR left ON (the deploy distribution). Two invocations with identical
args reproduce the JSON.

## Checkpoint under test

`logs/rsl_rl/hexabot_rough_direct/2026-06-10_00-30-42/model_400.pt`, selected by
`--select_best` (mean forward velocity). NB the policy **peaked at iter ~400 and
degraded afterwards** (model_950 scored worse) — and this run only reached iter 950
of a planned 3000, so it is **undertrained**. The B+C run is a fresh full 3000-iter
run, so part of any B+C gain will be "more training," not only the scan/region
change. Judge B+C deltas with that caveat (the milestone treats < +5 pts as "no
effect" when only one baseline run exists).

## Results

### difficulty 0.8  (mean success 0.066, min 0.000)

| family | success | mean_d (m) | p10_d (m) | stand_when_cmd |
|---|---|---|---|---|
| slope_boxes        | 0.152 | 0.60 | 0.28 | 0.077 |
| slope_noise_down   | 0.145 | 0.90 | 0.62 | 0.118 |
| random_rough       | 0.098 | 0.70 | 0.16 | 0.139 |
| slope_noise_up     | 0.055 | 0.43 | 0.25 | 0.097 |
| stairs_pure_down   | 0.012 | 0.39 | -0.01 | 0.341 |
| stairs_noise_down  | 0.004 | 0.48 | 0.34 | 0.165 |
| stairs_noise_up    | 0.000 | 0.06 | 0.02 | 0.249 |

### difficulty 0.6  (mean success 0.312, min 0.031)

| family | success | mean_d (m) | p10_d (m) | stand_when_cmd |
|---|---|---|---|---|
| slope_noise_down   | 0.730 | 1.52 | 0.73 | 0.090 |
| slope_noise_up     | 0.578 | 1.21 | 0.34 | 0.086 |
| slope_boxes        | 0.492 | 1.19 | 0.37 | 0.076 |
| stairs_noise_down  | 0.152 | 0.77 | 0.51 | 0.123 |
| stairs_pure_down   | 0.102 | 0.83 | 0.53 | 0.143 |
| random_rough       | 0.098 | 0.70 | 0.16 | 0.139 |
| stairs_noise_up    | 0.031 | 0.17 | 0.06 | 0.171 |

## Takeaways (the prior the rest of the milestone acts on)

- **Stairs are the bottleneck**, as predicted. Up-stairs is the worst: at d=0.8 the
  robot barely moves (mean_d 0.06 m) and freezes when commanded (stand 0.25) — an
  anticipation failure that Phase C (forward-extended scan) targets. Down-stairs
  falls (p10 distance goes negative at d=0.8).
- **Slopes are largely solved at d=0.6** (0.49–0.73) and degrade hard by d=0.8.
- **`random_rough` is identical across d=0.6 and d=0.8**: the pure
  `HfRandomUniformTerrainCfg` does not scale its noise with difficulty (only the
  mixed `*_rough`/`*_boxes` tiles do). So treat random_rough as a single
  difficulty-invariant data point, not a difficulty sweep.
- `stand_when_cmd_frac` is well above the 0.05 definition-of-done target everywhere,
  worst on stairs — the undertrained policy stalls rather than commits.
