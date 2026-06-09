

# Hexabot

**An open-source hexapod robot — body, simulation, learned locomotion, and (eventually) a brain.**

[status](#project-status)
[phase](#roadmap)
[python](#quickstart)
[sim](#simulation--rl)
[license](#license)



---

## The vision — System 1 / System 2

This project is an exploration of the **dual-process thinking** paradigm described in Daniel Kahneman's *Thinking Fast and Slow*, applied to robotics.

The human brain runs two processes in parallel:


| System       | Speed              | Nature                 | Robot equivalent                         |
| ------------ | ------------------ | ---------------------- | ---------------------------------------- |
| **System 1** | Fast (~50 ms loop) | Reflexive, embodied    | Locomotion policy (runs on-device)       |
| **System 2** | Slow (seconds)     | Deliberative, creative | Vision-language model (cloud / edge GPU) |


**System 1** — a ~50 M parameter neural network — learns balance, gait, and locomotion entirely through reinforcement learning in simulation. It runs on a Raspberry Pi Zero 2W at low latency and handles every millisecond-scale motor decision.

**System 2** — a vision-language model — sets goals, interprets the environment, holds memory, and gives the robot personality and purpose.

When they work together they create a general robot that can navigate, reason, communicate, and act — without any of the brittle, hand-written state machines that have historically made robots fragile.

---

## Why we switched: from TARS (biped) to Hexabot

The project started with **TARS**, a *Interstellar*-inspired bipedal robot: three vertical slabs, 4 DOF, MG996R servos. The 3D model, CAD pipeline, and sim assets are all still in the repo (`generate_growbot.py`, `HANDOFF.md`).

**TARS hit a hard physical ceiling.** With 4 pitch-only DOF and no lateral joints, it is an inverted pendulum — it must actively balance *every millisecond* just to stand. The best RL policy achieved ~0.035 m/s with persistent lateral drift. More fundamentally: **navigating uneven terrain with no lateral ankle DOF is mechanically impossible** — the feet can only push backward, and any slope throws the CoM off the support polygon with no recovery path.

The hexapod solves this at the architecture level:


|                               | **TARS (biped)**                | **Hexabot**                                   |
| ----------------------------- | ------------------------------- | --------------------------------------------- |
| Actuated DOF                  | 4 — hip + ankle pitch only      | **18** — 6 legs × coxa/femur/tibia            |
| Standing stability            | Inverted pendulum, must balance | **Statically stable** — tripod always planted |
| Lateral recovery              | None (no ankle-roll)            | Coxa yaw + 6 ground contacts                  |
| Falls if controller glitches? | Immediately                     | It just stands                                |
| Walking without RL            | Impossible                      | **Open-loop tripod gait works already**       |
| Terrain traversal             | Structurally impossible         | Large stepping range per leg                  |
| CoM height                    | 125 mm — high and unstable      | **75 mm** — low, inside foot polygon          |


RL becomes *polish* on the hexapod — speed, terrain, energy efficiency — instead of a prerequisite for not-falling-over.

---

## The robot

An 18-DOF hexapod built parametrically from scratch. Single source of truth: `hexabot_model/generate_hexabot.py`.

```
base ──coxa (yaw, +Z)──► coxa link ──femur (pitch)──► femur link ──tibia (pitch)──► tibia link (claw)
```

Each leg swings fore/aft (coxa), lifts (femur), and plants/flexes (tibia). Six legs at ±30° / ±90° / ±150° azimuths give **tripod groups A = {lf, rm, lr}, B = {rf, lm, rr}** — one tripod always in stance.


|                      |                                               |
| -------------------- | --------------------------------------------- |
| **Mass**             | 1.926 kg                                      |
| **Standing height**  | ~72 mm                                        |
| **Foot span**        | 590 mm                                        |
| **Actuators**        | 18× MG996R                                    |
| **Actuated DOF**     | 18 (6 legs × coxa/femur/tibia)                |
| **CoM**              | ~75 mm up, centred — low, inside foot polygon |
| **Compute (target)** | Raspberry Pi Zero 2W + PCA9685 + MPU-6050 IMU |
| **Power**            | 2S 7.4V LiPo                                  |


---

## Simulation & RL

### Simulator: NVIDIA Isaac Lab

Training runs in **NVIDIA Isaac Lab** (Isaac Sim 2.3.2) on a single RTX PRO 6000. We use **4 096 parallel environments**, which turns what would be weeks of wall-clock time into hours.

Isaac Lab gives us physically-accurate rigid-body dynamics, contact forces, and GPU-parallelised physics — the robot that learns in simulation is the one that transfers to hardware.

### RL algorithm: PPO with symmetry augmentation

We use **Proximal Policy Optimisation (PPO)** from RSL-RL. Two algorithmic choices are central:

**Left-right symmetry data augmentation.** The hexapod is exactly mirror-symmetric and straight-line walking is a symmetric task, so every transition's left-right mirror is valid on-policy data. Each PPO minibatch is doubled by swapping left/right legs and sign-flipping coxa yaw joints. This acts as a strong regulariser: lateral drift dropped **9× (1.1 m → 0.12 m over a 5 m run)** and training became dramatically more stable (monotonic episode-length climb vs. a noisy baseline that declined).

**CPG-modulated actions.** The policy doesn't output raw joint positions. Instead it outputs **per-leg modulations** `[Δfreq, Δcoxa_amp, Δlift]` over a **Central Pattern Generator** (CPG). Zero action = the analytical tripod gait scaled to commanded speed. This structurally prevents belly-crawling (a persistent failure mode of direct joint-offset policies) and means the policy only needs to learn *corrections* to an already-functional gait.

### The two-layer control stack

```
            ┌─────────────────────────┐  VelocityCommand(vx,vy,yaw)  ┌──────────────────────────┐
   goal ──▶ │  Navigation (Layer 2)   │ ════════════════════════════▶ │  Locomotion (Layer 1)    │ ──▶ joints
            │  goal-conditioned        │      FROZEN INTERFACE        │  PPO + CPG modulation    │
            └─────────────────────────┘                               └──────────────────────────┘
```

**Layer 1 — Locomotion (trained):** A PPO policy with a proprioceptive-only 75-d observation (projected gravity, angular velocity, joint positions/velocities, previous action, CPG phase). No base linear velocity in the obs — it's not measurable on the real robot. Domain randomisation (friction, mass, actuator gains, sensor noise, control-rate jitter) is on from the start so sim-to-real gap is baked into training.

**Layer 2 — Navigation (placeholder, real plumbing):** A hand-coded goal→`VelocityCommand` controller for now. The observation scaffold, lidar slot, and reward terms for a learned navigation policy are already wired in — they're inert until we train the policy.

The **frozen interface** (`isaac_lab/interfaces/velocity_command.py`) is the only contract between the two layers. Swapping either layer doesn't touch the other.

### Training results (Milestone 0)

After a full 1 000-iteration run with symmetry augmentation and the CPG action space:

- **Coordinated forward wave gait** — rear → mid → front leg sequencing
- **Lateral drift reduced ~9×** vs. the direct-joint baseline
- **Near-perfect locomotion** — episode length 462–529 steps (target: 599), 0 deaths, stable across iterations 550–999
- **Best checkpoint forward velocity ≈ 0.29 m/s** on a 0.2 m/s command — tracking, not over-driving

---

## Current status


| Area                                             | State       |
| ------------------------------------------------ | ----------- |
| Parametric 3D model (body + drivetrain)          | ✅ Done      |
| Simulation assets (URDF, inertia, collision)     | ✅ Done      |
| Locomotion policy — flat ground, forward walking | ✅ **Done**  |
| Symmetry data augmentation                       | ✅ Done      |
| CPG-modulated action space                       | ✅ Done      |
| Two-layer control stack (frozen interface)       | ✅ Done      |
| Waypoint / multi-goal navigation                 | 🔜 **Next** |
| Rough terrain curriculum                         | 🔜 Next     |
| Lateral / turning commands                       | 🔜 Next     |
| Physical build bring-up                          | ⬜ Planned   |
| System 2 — vision / VLM integration              | ⬜ Planned   |
| Audio I/O (voice in + out)                       | ⬜ Planned   |
| Memory + agentic behavior                        | ⬜ Planned   |


---

## Roadmap

- **TARS biped** — parametric CAD, simulation assets, RL training (ceiling: ~0.035 m/s, lateral instability). History preserved in `generate_growbot.py` / `HANDOFF.md`.
- **Hexabot** — parametric 18-DOF hexapod generator, validated URDF, analytic inertia.
- **Locomotion RL** — PPO + CPG modulation + symmetry augmentation + BC warm-start. Coordinated wave gait, straight walking, domain-randomised.
- **Two-layer stack** — frozen `VelocityCommand` interface; locomotion Layer 1 trained; navigation Layer 2 placeholder with full scaffold.
- **Waypoint navigation** — train a goal-conditioned Layer 2 policy; multi-waypoint path following.
- **Rough terrain** — height-scan obs slot already wired; add terrain curriculum (ramps, steps, rubble).
- **Lateral + turning** — widen `VY_RANGE`/`YAW_RANGE` in the frozen interface; retrain or fine-tune Layer 1.
- **Physical build** — BOM, print profiles, wiring harness, servo calibration, embedded firmware.
- **System 2 integration** — camera(s), VLM inference stack, goal generation.
- **Audio** — speech-to-text in, text-to-speech out. The TARS conversational feel.
- **Memory & agency** — persistent memory, long-horizon autonomous behavior.

---

## What it looks like fully trained and assembled

Once the full stack is running on hardware:

**Example capability 1 — Autonomous waypoint traversal:** You place a waypoint 3 m away on uneven ground. The VLM (System 2) breaks the path into intermediate goals and sends `VelocityCommand` updates over the frozen interface. The locomotion policy (System 1) handles every footfall, adapts to terrain variation in real time via the height-scan, and never falls — because statically stable. The VLM updates the goal if the robot drifts off path. No human in the loop.

**Example capability 2 — Conversational autonomy:** You say *"go check what's behind the couch."* The VLM interprets the instruction, identifies the destination in the camera feed, plans a route, and sequences waypoints. As it navigates, System 1 handles all the low-level motor control. When it arrives, System 2 describes what it sees, stores the observation in persistent memory, and waits for the next instruction.

---

## Repository layout

```
.
├── hexabot_model/
│   ├── generate_hexabot.py     # ★ SOURCE OF TRUTH — params → meshes + URDF
│   ├── isaac/                  # hexabot.urdf, meshes/, hexabot_cfg.py, tripod_gait.py
│   └── HEXABOT.md              # hexabot spec, numbers, TARS comparison
│
├── isaac_lab/
│   ├── tasks/hexabot/          # ★ RL task — env, CPG, symmetry augmentation
│   │   ├── hexabot_env.py      #   direct-workflow env
│   │   ├── hexabot_env_cfg.py  #   reward/obs/action config
│   │   ├── cpg.py              #   Central Pattern Generator (action modulation)
│   │   ├── symmetry.py         #   left-right data augmentation
│   │   ├── bc_warmstart.py     #   behaviour-cloning warm-start
│   │   └── README.md           #   two-layer stack design doc
│   ├── interfaces/
│   │   └── velocity_command.py # ★ FROZEN interface between nav + locomotion layers
│   ├── nav/
│   │   ├── go_to_goal.py       #   Layer 2 placeholder: goal → VelocityCommand
│   │   └── nav_goal_cfg.py     #   goal-conditioned obs scaffold + reward seams
│   ├── train_hexabot.py        # launch training (+ --bc_warmstart)
│   ├── play_hexabot.py         # export best checkpoint
│   ├── record_hexabot.py       # record video of trained policy
│   └── run_nav_demo.py         # end-to-end two-layer demo
│
├── generate_growbot.py         # TARS/Growbot biped generator (history)
├── HANDOFF.md                  # TARS project state capsule
├── CLAUDE.md                   # agent context: failure modes, fixes, lessons learned
└── logs/rsl_rl/hexabot_flat_direct/   # training logs
```

★ = the best places to start.

---

## Quickstart

**Requirements:** Python 3 + NumPy for model generation. Isaac Lab requires an NVIDIA RTX GPU on Linux/Windows.

```bash
# Regenerate the hexabot model (if you change params)
python3 hexabot_model/generate_hexabot.py
python3 hexabot_model/isaac/validate_urdf.py   # should print ALL CHECKS PASSED

# Train the locomotion policy (GPU box, env_isaaclab conda env)
cd external/IsaacLab
./isaaclab.sh -p ../../isaac_lab/train_hexabot.py \
    --task Isaac-Velocity-Flat-Hexabot-Direct-v0 \
    --num_envs 4096 --max_iterations 1000 --bc_warmstart --headless

# Export the best checkpoint (scored by forward velocity)
./isaaclab.sh -p ../../isaac_lab/play_hexabot.py --select_best

# Record a video of the trained policy
./isaaclab.sh -p ../../isaac_lab/record_hexabot.py --policy <path/to/exported/policy.pt>

# Run the two-layer navigation demo
./isaaclab.sh -p ../../isaac_lab/run_nav_demo.py --policy <path/to/exported/policy.pt> --goal_x 2.5
```

**Run the self-tests (no GPU needed):**

```bash
python isaac_lab/tasks/hexabot/test_cpg.py   # zero-action == analytical gait; mirrors are involutions
```

---

## Documentation


| Doc                                                                      | What's in it                                                                          |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| `[hexabot_model/HEXABOT.md](hexabot_model/HEXABOT.md)`                   | Hexabot spec, numbers, Hexabot vs TARS comparison                                     |
| `[isaac_lab/tasks/hexabot/README.md](isaac_lab/tasks/hexabot/README.md)` | Two-layer stack design: frozen interface, CPG action, obs layout, extensibility seams |
| `[CLAUDE.md](CLAUDE.md)`                                                 | Full agent context: every failure mode, fix, and lesson from training                 |
| `[HANDOFF.md](HANDOFF.md)`                                               | TARS biped history: design, BOM, wiring, why we moved on                              |
| `[ISAAC_LAB_SETUP.md](ISAAC_LAB_SETUP.md)`                               | Isaac Sim/Lab install + URDF→USD conversion guide                                     |


---

## Contributing

This is an open project. The most useful next milestones are waypoint navigation training, rough terrain curriculum, and eventually the physical build.

- **Open an issue** to discuss a direction or report a problem.
- **Keep the model parametric.** Geometry changes go through `hexabot_model/generate_hexabot.py` — not hand-edits to the URDF or meshes. Re-run the generator and validator afterward.
- **Read `CLAUDE.md`** before touching the reward function. Several failure modes have hard-won fixes that look optional but aren't (e.g. `joint_accel_reward_scale`, the eplen plateau, the saturating velocity term).

---

## License

**To be finalized.** Recommended:

- **Code** (generators, training, control) → MIT or Apache-2.0
- **Hardware / 3D models** → CERN-OHL-S or CC-BY-SA-4.0
- **Docs** → CC-BY-4.0

Until a `LICENSE` file is added, no explicit license is granted.

---

## Acknowledgements

Inspired by **TARS** from Christopher Nolan's *Interstellar* (Warner Bros. / Legendary) — the monolith silhouette, the personality, the dual-process model of cognition. Hexabot is an independent original design, not affiliated with or derived from the film's assets.

The dual-process framing draws on Daniel Kahneman's *Thinking Fast and Slow*.