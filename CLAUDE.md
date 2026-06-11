# Isaac RL Lab — Agent Context

This file is the authoritative context handoff for any agent (Claude Code or otherwise) working in this repo. It covers what has been tried, what broke, and how the system works.

---

## Environment

- **Conda env:** `env_isaaclab` (Isaac Sim + Isaac Lab 2.3.2 / isaaclab 0.54.2, torch cu128)
- **Isaac Lab clone:** `external/IsaacLab` — run Python via `./isaaclab.sh -p <script>` from inside that directory
- **GPU:** RTX PRO 6000, 96 GB
- **`external/`** is reserved for untouched cloned repos — never put project code there

**Gotchas:**
- Isaac processes **hang at `simulation_app.close()`** holding GPU memory — `pkill -9` them after every run
- Python stdout to a redirected file is block-buffered — use `-u` flag
- Use `sim.step(render=False)` for non-recording loops; `sim.step()` renders every frame and is slow headless
- First run with `--enable_cameras` triggers slow RTX shader compile

---

## TARS / Growbot (bipedal)

**Asset:** `isaac_lab/growbot.urdf` → `isaac_lab/growbot.usd`. 5 links: base_link, leg_left/right_link, foot_left/right_link. 4 revolute pitch joints: hip_left/right ±90°, ankle_left/right ±49°. No knee, no ankle-roll — sagittal-plane only. URDF has no foot friction material (PhysX default ~0.5 → feet slip); friction must be set explicitly. Grippy TPU material (static ~1.3) applied to feet + ground.

**Task:** `isaac_lab/tasks/growbot/`, id `Isaac-Velocity-Flat-Growbot-Direct-v0` (direct workflow). Rewards: forward_progress + velocity tracking + upright + base-height + alive. Penalties: lateral-pos drift, yaw rate, foot_slip, action-rate/joint-vel.

**Run:**
```bash
cd external/IsaacLab
./isaaclab.sh -p ../../isaac_lab/train.py \
  --task Isaac-Velocity-Flat-Growbot-Direct-v0 \
  --num_envs 4096 --max_iterations 1500 --headless
```
Logs → `<project_root>/logs/rsl_rl/growbot_flat_direct/`. Export policy with rsl_rl `play.py`. Record: `./isaaclab.sh -p ../../isaac_lab/record_policy.py --policy <policy.pt>`.

**Performance ceiling:** ~0.035 m/s with modest lateral drift. Faster/straighter is bounded by weak MG996R servos + no lateral DOF.

---

## Hexabot (18-DOF hexapod)

**Spec:** 6 legs × coxa/femur/tibia = 18 DOF. 1.926 kg, stands ~72 mm. Single source of truth for the asset: `hexabot_model/generate_hexabot.py` (claw tip = tibia-local x = L_TIBIA = 0.135 m).

**Task:** `isaac_lab/tasks/hexabot/`, id `Isaac-Velocity-Flat-Hexabot-Direct-v0` (direct workflow). Trains straight-line forward walking.

**Run / export / record:**
```bash
cd external/IsaacLab

# Train
./isaaclab.sh -p ../../isaac_lab/train_hexabot.py \
  --task Isaac-Velocity-Flat-Hexabot-Direct-v0 \
  --num_envs 4096 --max_iterations 1000 --headless

# Export BEST checkpoint (scored by mean forward body-velocity)
./isaaclab.sh -p ../../isaac_lab/play_hexabot.py --select_best
# Without --select_best → exports the LATEST checkpoint

# Record video
./isaaclab.sh -p ../../isaac_lab/record_hexabot.py --policy <exported/policy.pt>

# Passive standing sanity test (no policy — confirm physics before blaming RL)
./isaaclab.sh -p ../../isaac_lab/standing_test_hexabot.py
```
Logs → `logs/rsl_rl/hexabot_flat_direct/`.

---

## Hexabot: diagnosed failure modes and fixes

### 1. Every episode died at ~step 5 (never learned)
**Symptom:** mean ep length 5.00, 800+ deaths/iter.  
**Cause:** PD spawn-settling transient dips base z to ~0.023 m (below `too_low < 0.035` death floor) in the first ~0.1 s regardless of spawn height.  
**Fix:** `settle_steps=15` grace in `_get_dones` suppresses deaths for the first 15 control steps after reset. Also spawn at 0.072 m (not 0.085 m). Lowering spawn height alone does NOT remove the dip.

### 2. Belly-crawl (lie flat, wiggle for a sliver of forward reward)
**Fix:** stand-first curriculum (`curriculum_stand_steps=2000`, `curriculum_ramp_steps=4000` — command vx=0 then ramp), `belly_clearance` penalty (-50, one-sided when underbelly < 0.045 m), `foot_support` reward (+20, feet planted below belly via FK on tibia pose), alive 0.5→1.0, entropy 0.005→0.01, init_noise_std 1.0→0.8.

### 3. Uncoordinated motion
**Fix:** motion-gated (`cmd vx > 0.1`) `tetrapod_contact` (+1.5, bell at exactly 4 feet down) + `gait_symmetry` (+1.5, contact pattern matches left-right mirror; mirror map built from body names lf↔rf etc.) → symmetric tetrapod/wave gait.

### 4. Small rapid jitter (first attempt)
**Fix attempt:** `foot_clearance` reward (+4.0, swing feet rewarded for apex up to `foot_clearance_target=0.025` m) + bumped `feet_air_time` 1.0→2.5. These are motion-gated.

### 5. Persistent small/rapid jitter (structural fix needed)
Reward tuning (foot_clearance, air_time, action_rate, damping) could NOT fix it — `track_lin_vel_xy_exp` (exp map) **saturates**, giving no gradient for gait quality, and contact-count gait terms can be satisfied statically.

**Fix = phase-clock periodic gait reward:**
- Per-env gait clock `_gait_phase` advancing at `gait_frequency=1.5` Hz
- Per-leg offsets: rear=0, mid=1/3, front=2/3 → rear→mid→front symmetric wave
- `gait_swing_fraction=0.30`
- `gait_phase` reward (+2.5) = fraction of feet matching their scheduled stance/swing
- Clock added to obs as sin/cos → observation_space 66→68
- **`record_hexabot.py` MUST build the same clocked obs** (it advances its own `gait_phase` at `GAIT_FREQ`)

Jitter can't match a 0.67 s rhythm → dies out.

### 6. Marching in place after adding gait reward
**Symptom:** gait_phase 0.9 but forward_progress 0.04 — robot marched perfectly in place.  
**Root cause:** `track_lin_vel_xy_exp` saturates → a frozen robot scores `exp(-0.15²/0.25)=0.91` of max at scale 2.0, paying ~1.8 for doing nothing.  
**Fix:** `lin_vel_reward_scale` 2.0→0.5 so the standstill-zero linear `forward_progress` (12.0) dominates. Also dropped actuator `damping` 1.0→0.5 (1.0 over-resisted push-off).  
**Result @250 iters:** forward_progress 0.04→1.0, gait_phase→1.5, tetrapod 0.82, symmetry 0.77, eplen 599, 0 deaths — coordinated forward wave gait.

### 7. World-frame regression: froze + stood motionless (the "world-X" change)
**Symptom after switching `forward_progress` to world-frame `root_lin_vel_w[:,0]` + adding `gait_phase`/`heading`/`foot_plant`:** policy collapsed to standing tall & motionless (dx≈0). Train-log: `track_lin_vel_xy_exp≈1.8` (biggest term), `forward_progress≈0.15`.

**Root cause:** `track_lin_vel_xy_exp` was farmable by standing — at avg cmd ~0.15 m/s a frozen robot scores 0.91 of max, which at scale 2.0 drowned the linear `forward_progress`. Previously body-frame `forward_progress` rewarded the circling motion and masked this. Secondary: old `gait_phase = mean(feet_contact==sched_stance)` paid ~70% to a static all-planted stance.

**Fix:** (a) `lin_vel_reward_scale` 2.0→0.5; (b) rewrote `gait_phase` to reward only correctly **lifting** scheduled-swing feet (`(~feet_contact & sched_swing).sum / n_sched_swing`) — static robot scores 0. World-frame `forward_progress` kept.

**LESSON:** A saturating exp velocity-tracking term must NOT outweigh the linear forward driver, or standing becomes optimal. Audit `train.log Episode_Reward/*` breakdown — `forward_progress` should dominate when moving.

---

## Hexabot: RL-algorithm changes (not reward tuning)

### Left-right symmetry data augmentation (RslRlSymmetryCfg) — ADOPTED
The hexapod is exactly mirror-symmetric and straight-line walking is a symmetric task, so
every transition's left-right mirror is valid on-policy data. `symmetry.py:compute_symmetric_states_lr`
mirrors each PPO minibatch (swap l/r legs, sign-flip coxa-yaw joints, negate lat-vel/gravity/cmd,
gait clock invariant); wired via `RslRlSymmetryCfg(use_data_augmentation=True, ...)` in the PPO cfg.
The joint mirror permutation/sign are built from live `joint_names` in `HexabotEnv.__init__`
(`_jt_mirror_idx`, `_jt_mirror_sign`). Coxa flips sign because its axis is world +Z (yaw → −yaw under
reflection); femur/tibia are pitch joints (lift is handedness-free) so they swap WITHOUT a sign flip
— same treatment as ANYmal HAA vs HFE/KFE. Verified the mirror is a correct involution before training.

**Result vs the dx=4.774 baseline (`runs/2026-06-08_19-58-43`), full 1000-iter run:**
- **dy 1.111 → 0.117 m** (~9× straighter — the headline win) and stands taller (final_h 0.053 → 0.069 m).
- **Training far more robust:** smooth monotonic eplen climb, tail settles at **462–529** vs baseline's
  noisy 240–355 that *declines to 167*. Best checkpoint fwd_vel ≈0.29 m/s, steady across iters 550–999.
- **Cost: raw dx 4.774 → 1.6–1.9 m** AND the walking breakout is ~325 iters LATER (iter ~850 vs ~500).
  Hard augmentation over-regularizes the exploration phase. But the baseline's "high dx" is really
  uncommanded over-driving — it hit 0.477 m/s on a 0.2 m/s command while veering off; the symmetric
  policy *tracks* the command and goes straight. Tradeoff accepted: straight+stable over fast+crooked.
- To push speed back up later: try soft `use_mirror_loss` (won't block exploration → earlier breakout),
  or raise the command range / `forward_progress` weight. Untried: obs-normalization + longer horizon.

**LANDMINE (cost me 3 wasted runs):** the eplen≈16 plateau lasts to ~iter 300 and walking only breaks
out ~iter 500 (later with symmetry). This is NORMAL for the current reward config, NOT a failure —
short 150-400 iter validation runs are still in the plateau and look identically "broken." **Judge any
Hexabot change only on a full run past the breakout (~iter 550+).**

---

## Hexabot: known landmines

- **`joint_accel_reward_scale` -2.5e-7 is a hard limit.** Raising to -5e-7 broke the STAND phase (eplen stuck at 16 = settle_steps+1; robot stopped applying corrective accels). `dof_acc` penalty is always-on (not motion-gated) — any change affects standing. Keep at -2.5e-7.
- **`foot_clearance_target=0.04` is too high** (~55% of 72 mm standing height) → destabilizing. 0.025 is stable.
- **`action_rate = -0.04` over-suppresses motion** (robot nearly stops). -0.015 is fine.
- **Validate any reward change with a ~250-iter run** (reaches full curriculum ramp): expect eplen→599, died→0, and verify `forward_progress` doesn't collapse.

---

## Milestone 0 — two-layer control stack (CURRENT architecture)

A goal→velocity→gait stack. **Read `isaac_lab/tasks/hexabot/README.md` first.** This
RE-ARCHITECTED the locomotion layer; several notes above describe the OLD direct-action
policy and are kept as history (the failure-mode *lessons* still hold; the obs/action
*shapes* do not).

**Frozen interface** (`isaac_lab/interfaces/velocity_command.py`): `VelocityCommand(vx,vy,yaw)`
+ canonical ranges. The ONLY contract between the two layers; both import it (the loco
cfg derives `cmd_vx_range` from `VX_RANGE`). Entry-point scripts add the repo root to
`sys.path` so `import isaac_lab.interfaces` resolves.

**Locomotion (Layer 1) — changed vs the old policy:**
- **Obs is proprioceptive-ONLY, 81-d, NO base linear velocity** (not measurable on the real
  robot). Layout: grav(3) ang_vel(3) cmd(3) jpos(18) jvel(18) prev_act(24) cpg_phase(12)
  + dormant `n_height_scan`(0) height-scan seam. `symmetry.py` matches this layout.
  (Was 75-d with prev_act(18) before M1.5 Phase E added the d_stance action channel.)
- **Action MODULATES a CPG** (`cpg.py`), not joint offsets: per-leg
  `[d_freq,d_coxa_amp,d_lift,d_stance]` (24). `d_stance` (Phase E) is a ONE-SIDED femur
  press-down (≤`cpg_kstance`=0.35 rad ≈ +30–40 mm ride height) so the policy can walk
  belly-up on rough terrain; ≤0 is a no-op (can't reopen the belly-crawl).
  **Zero action == the analytical tripod gait** (`tripod_gait.py:gait_pose`) scaled by command
  speed (`cpg_v_ref`); at vx=0 it holds the standing stance, so the CPG STRUCTURALLY prevents
  belly-crawl (stand curriculum shortened to 500 steps). The CPG action mirror is a leg-swap
  with NO sign flip (params are sign-invariant) — unlike the old joint-offset coxa flip.
- **Imitation reward** (annealing): `-(CPG(action)-CPG(0))²`, weight 1→0 over `imitation_anneal_steps`,
  tied to curriculum. + BC warm-start (`bc_warmstart.py`, `train_hexabot.py --bc_warmstart`).
  Reference, NOT residual (hard constraint).
- **Domain randomization ON from start:** friction/mass/actuator-gains (`EventCfg`) +
  actuator latency / control-rate jitter / IMU+obs noise (`DomainRandCfg`, applied in the env step).
- `record_hexabot.py` rebuilds the new obs + imports `cpg.py` so it can't drift from training.

**Navigation (Layer 2) — hand-coded placeholder, real plumbing:** `nav/go_to_goal.py`
(goal→VelocityCommand, computes dormant yaw), `nav/nav_goal_cfg.py` (goal-rel obs + dormant
zeroed lidar slot + dense progress reward + inert collision/path terms), `run_nav_demo.py`
(end-to-end both layers over the interface). No nav RL this milestone.

**Verified (2026-06-09):** `test_cpg.py` passes (zero-action==analytical gait, mirrors are
involutions); Isaac smoke run (256 envs, 5 iters, --bc_warmstart) — actor/critic in_features=75,
action 18, BC mse→2e-5, symmetry loss active, all reward terms log. Full 1000-iter BC+PPO run
launched. The exit-1 on smoke is only the known `simulation_app.close()` hang (pkilled).

---

## Milestone 1 — rough-terrain locomotion (privileged distillable teacher)

Extends the flat locomotion policy to rough terrain. **Frozen `(vx,vy,yaw)`
interface and the navigation layer are UNCHANGED.** New task id
`Isaac-Velocity-Rough-Hexabot-Direct-v0` SUBCLASSES the flat env/cfg (same CPG
action, reward family, DR) and adds only rough-terrain machinery. Read
`isaac_lab/tasks/hexabot/README.md` (Milestone 1 section) first.

**New files:** `hexabot_rough_env.py`, `hexabot_rough_env_cfg.py`,
`rough_terrains.py`, `teacher_policy.py`; PPO cfg `HexabotRoughPPORunnerCfg`;
entry points `train_rough.py` / `play_rough.py` / `render_rough.py`; bash
`scripts/milestone1_rough.sh` (train→export→video) and
`scripts/render_latest_rough.sh`. Edited: `hexabot_env.py` (+`self._ground_height`
makes height terms terrain-relative), `symmetry.py` (+scan mirror), task `__init__`.

**Key design:**
- **Distillable teacher** (`teacher_policy.py:HexabotTeacherActorCritic`, a custom
  rsl_rl ActorCritic injected into the runner namespace by the entry scripts):
  obs is one `policy` group `[proprio(75) | height_scan(63)]`; the privileged scan
  enters ONLY through a latent bottleneck — `z=scan_encoder(63→16)`,
  `action=actor_trunk([proprio|z])`. Critic uses full 138-d (privileged, dropped at
  deploy). **Distillation seam:** keep `actor_trunk`, swap `scan_encoder` for a
  proprio-history encoder regressing the same `z`.
- **Terrain:** `terrain_type="generator"`, curriculum on, blind-feasible only
  (slopes/rough/low boxes/low stairs; NO gaps/stepping-stones/beams), scaled tiny
  for the 72 mm robot. `rough_terrains.py`. **Gotcha:** a sub-terrain's
  `grid_width` must NOT evenly divide tile `size` (auto border width would be 0 →
  RuntimeError); boxes use 0.3 into 2.0.
- **Curriculum driver:** direct workflow has no curriculum manager →
  `HexabotRoughEnv._reset_idx` calls `terrain.update_env_origins` itself and logs
  `Curriculum/terrain_level` (mean) — the **lead metric**.
- **Height scanner:** `RayCasterCfg` 9×7=63 rays on base_link, yaw-aligned; fills
  `n_height_scan` (obs 75→138). PRIVILEGED.
- **obs width 138** → `symmetry.py` mirrors the scan tail via `_scan_mirror_idx`
  (built from the ray pattern). Imitation reward keeps the M0 anneal (→0 over the
  near-flat early curriculum); NOT re-anchored on rough.
- **Export:** the stock rsl_rl exporter grabs only `policy.actor` (the trunk), so
  `play_rough.py` exports a FULL-teacher wrapper (encoder+trunk, traced) instead.

**Verified (2026-06-09, smoke):** `train_rough.py` (64 envs, 5 iters) runs —
terrain generates, obs=138 (actor/critic in_features 138; scan_encoder 63→16;
trunk 91→18), symmetry+scan active, all reward terms + `Curriculum/terrain_level`
log, 0 errors. `play_rough.py --select_best` exports `exported/policy.pt`(+onnx).
The exit-1 is the known `simulation_app.close()` hang (pkilled). **Full
multi-hour run NOT yet done** — run `scripts/milestone1_rough.sh` and judge on
`Curriculum/terrain_level` climbing past the eplen plateau.
(M1.5 Phases B+C later extended the scan to 91 rays and Phase E widened the
action to 24 → current rough obs = 81 + 91 = **172**; see below + the M1.5 memory.)

### Milestone 1.5 Phase E — belly management + anticipatory ride height (2026-06-10)

Level-8 failure was "belly touches ground → instant death" (flat 1 N base-contact
rule) plus no actuator to walk taller. Changes (all inert on flat — flat defaults
keep the original hard rules):
- **`d_stance` CPG channel:** action 18→24 (`[d_freq,d_coxa_amp,d_lift,d_stance]`),
  obs 75→81 (rough 166→172). One-sided femur press-down, `cpg_kstance=0.35` rad.
  Zero-action == analytical gait still holds; CPG/action mirrors stay leg-swap-only
  (stance is handedness-free). `symmetry.py` slices and `teacher_policy.n_proprio`
  (now DERIVED as obs−N_HEIGHT_SCAN) updated; everything else derives shapes.
- **Graded belly contact (rough):** death only above `base_contact_force_death=50 N`
  (~2.6× bodyweight, damage proxy); below that `belly_contact_force` penalty −0.1/N
  above a 2 N free allowance. `too_low` death now needs `too_low_grace_steps=25`
  (0.5 s) consecutive low steps — transient clamber-nudges survive, lying flat dies.
- **Anticipatory posture (rough):** scan q90-above-mid-ground protrusion (incl. the
  +0.40 m lookahead) → `_height_target_offset` (≤30 mm; raises base-height target +
  belly_clearance floor → the −50 belly term pays for raising the belly BEFORE
  obstacles) and `_foot_clearance_offset` (≤+25 mm swing-apex raise). New
  `foot_stumble` penalty (−1.0 rough, 0 flat) prices horizontal-dominated foot hits
  (toe-catch on box walls/stair risers). Watch `Metrics/height_target_offset`.
- **LANDMINE:** action/obs widths changed — pre-Phase-E checkpoints (incl. the
  2026-06-10_09-48-32 B+C run) are NOT loadable/resumable; retrain from scratch.

---

## Checkpoint for other agents

After all fixes, a 60-iter smoke run showed: ep length 5→112 and rising, deaths 820→3/iter. The hardware/stance/actuators are fine — the original collapse was spawn-death + last-checkpoint export, not the robot.

Validated good run (2026-06-08): eplen 599, 0 deaths, forward_progress ~1.0, coordinated forward wave gait.
