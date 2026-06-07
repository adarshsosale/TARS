# Isaac Lab installation guide

End-to-end procedure for setting up Isaac Lab 2.3.2 (on Isaac Sim 5.1.0)
against an NVIDIA Blackwell-class GPU (RTX PRO 6000, compute capability
`sm_120`). Captures the actual sequence and workarounds used on this
workstation — including issues not documented in the upstream NVIDIA
install pages.

If your hardware differs (older GPU, Ampere/Ada), the pinned versions
may still work but the Blackwell-specific driver warnings below don't apply.

---

## 0. Preflight

Run before installing anything. Each one is a hard requirement; do not skip.

| Check | Command | Requirement |
|---|---|---|
| GPU + driver | `nvidia-smi` | Driver **≥ 580.65.06**, < 590. See "Driver gotcha" below. |
| Python 3.11 | `python3.11 --version` | Exactly 3.11.x. 3.10/3.12 will not work. |
| GLIBC | `ldd --version` | ≥ 2.35 (Ubuntu 22.04 and later are fine) |
| RAM | `free -h` | ≥ 32 GB |
| Disk | `df -h ~` | ≥ 50 GB free (install eats ~30 GB, leave headroom for caches/logs) |
| Conda | `conda --version` | Any recent miniconda/anaconda |

### Driver gotcha (Blackwell)

The NVIDIA 580 production branch (Linux: `>= 580.65.06`) is the **only safe
band** for Isaac Sim 5.1 on RTX PRO 6000 Blackwell at the time of writing.

- Drivers `590+` are known to break Isaac Sim's CUDA device detection.
- Even `595.x` is reported to make PhysX silently fall back to CPU on
  Blackwell — the run looks fine but is ~CPU-speed, with no error message.

If you have a `590+` driver installed, downgrade before continuing.
**Do not attempt the install with a too-new driver and hope it works.**

---

## 1. Conda environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install --upgrade pip
```

Python must be **exactly 3.11**. Isaac Sim 5.x wheels are not built for any other version.

---

## 2. Install the pip layer (PyTorch + Isaac Sim)

From the project root:

```bash
pip install -r requirements.txt
```

This pulls:

- `torch==2.7.0+cu128` and `torchvision==0.22.0+cu128` from PyTorch's cu128 index
- `isaacsim[all,extscache]==5.1.0` from `pypi.nvidia.com`
- `setuptools<81` (needed for the flatdict build later — see step 5)
- `py-spy` (diagnostics)

> ⚠️ The PyTorch wheel is ~1.1 GB and the Isaac Sim wheels total another
> ~10 GB. The full pip step takes **20–40 minutes** depending on network.
> Don't kill it — there's no progress output until pip flushes.

### Verify CUDA + Blackwell support

```bash
python -c "
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('device:', torch.cuda.get_device_name(0))
print('compute_capability:', torch.cuda.get_device_capability(0))
print('compiled_archs:', torch.cuda.get_arch_list())
"
```

Expected output:
- `torch: 2.7.0+cu128`
- `cuda_available: True`
- Device name contains `Blackwell`
- `compute_capability: (12, 0)` (this is `sm_120`)
- `sm_120` appears in `compiled_archs` — confirms no JIT fallback

If `cuda_available: False`, the driver is the problem. Do not continue.

---

## 3. Accept the Omniverse EULA

Isaac Sim refuses to bootstrap without explicit EULA acceptance. Without
this, the very first `import isaacsim` will fail with
`Unable to bootstrap inner kit kernel: EOF when reading a line`.

**Read the EULA first**:
https://docs.omniverse.nvidia.com/platform/latest/common/NVIDIA_Omniverse_License_Agreement.html

Then persist consent inside the conda env's activation script so every
shell that activates the env inherits the setting:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/omni_eula.sh" <<'EOF'
#!/usr/bin/env bash
export OMNI_KIT_ACCEPT_EULA=YES
export PRIVACY_CONSENT=N
EOF
```

`PRIVACY_CONSENT=N` declines Omniverse telemetry/data sharing — flip to `Y`
if you want to opt in. Both vars are scoped to this conda env only, not
global.

Reactivate the env so the new script runs:

```bash
conda deactivate && conda activate env_isaaclab
```

Verify:

```bash
python -c "import isaacsim; import omni; print('isaac sim import OK')"
```

`omni.__file__` will print `None` — that's normal, `omni` is a namespace package.

---

## 4. Clone Isaac Lab into `external/`

This repo treats `external/` as vendored third-party code (gitignored).
Keep cloned material there to avoid polluting the project root.

```bash
mkdir -p external
git clone https://github.com/isaac-sim/IsaacLab.git external/IsaacLab
cd external/IsaacLab
git checkout v2.3.2
```

Pin to `v2.3.2` — do not use `main` or the 3.0 beta (Newton physics path).

---

## 5. Pre-install flatdict (workaround)

`flatdict==4.0.1` is a transitive dependency of Isaac Lab. Its packaging
predates PEP 517 and imports `pkg_resources` directly in `setup.py`.
Since setuptools 81 removed `pkg_resources`, modern pip's isolated build
sandbox can't build it.

The fix is to build it once with the env's setuptools<81 visible
(`--no-build-isolation` uses the env instead of a fresh sandbox):

```bash
pip install --no-build-isolation flatdict==4.0.1
```

Once flatdict is on disk, the upcoming `./isaaclab.sh -i` step will see
the requirement is already satisfied and skip the rebuild.

---

## 6. Install Isaac Lab + RL frameworks

```bash
./isaaclab.sh -i
```

From `external/IsaacLab/`. The script:

- detects the active conda env and uses its Python
- iterates `source/*` and runs `pip install -e` on each
- installs all five RL frameworks: `rsl_rl`, `rl_games`, `skrl`,
  `stable_baselines3`, `robomimic`

Takes ~5–10 minutes. You'll see several `pip dependency resolver` warnings
about `psutil`, `click`, `starlette`, and `packaging` version pin conflicts
between Isaac Sim's wheels and Isaac Lab's dependencies. **These are
non-fatal** — the conflicting versions install anyway and don't cause
runtime failures.

### Verify imports

```bash
python -c "
import importlib
for m in ['isaaclab', 'isaaclab_rl', 'isaaclab_mimic', 'rsl_rl',
         'rl_games', 'skrl', 'stable_baselines3', 'robomimic']:
    try:
        importlib.import_module(m)
        print('  OK  ', m)
    except Exception as e:
        print('  FAIL', m, '—', type(e).__name__, e)
"
```

All eight should report OK. (`isaaclab_tasks` and `isaaclab_assets`
will fail with `No module named 'pxr'` from a bare interpreter — that's
expected because `pxr` only becomes available once the Isaac Sim app is
bootstrapped via `./isaaclab.sh -p`. It is not a broken install.)

---

## 7. Smoke test — run a sample RL job

The simplest end-to-end check: train Ant for 50 PPO iterations.

```bash
cd external/IsaacLab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Isaac-Ant-v0 --headless --max_iterations 50
```

Expected outcome on Blackwell:
- `[INFO][AppLauncher]: Using device: cuda:0`
- 50 iterations complete in **~20 seconds**
- Mean reward climbs from ~0 to ~+18
- Throughput around **300k+ env-steps/sec**
- Checkpoints written to `logs/rsl_rl/ant/<timestamp>/`

### Critical check: PhysX must be on GPU

While the run is going (or after), scan the stdout/log for any of:

- `fallback to CPU`
- `GPU pipeline` followed by `fail` or `disabled`
- `PhysX` followed by `CPU`

If any appear, **stop**. That's the Blackwell driver gotcha from step 0 —
PhysX has silently moved to CPU and you're running at ~1% of real speed.

The Kit log at
`$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/`
should contain `Physics using context ..., device 0` — that's PhysX
binding to GPU device 0.

`PYTHONUNBUFFERED=1` matters because Python's stdout is block-buffered
when piped to a file. Without it, you won't see iteration prints until
the process exits.

---

## 8. Replay (optional)

To replay a trained policy and record video:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Isaac-Ant-v0 --num_envs 32 \
  --headless --video --video_length 200
```

`--headless --video` writes an mp4 under
`logs/rsl_rl/ant/<timestamp>/videos/play/`. Requires `ffmpeg` on PATH.

---

## Known soft issues

### `simulation_app.close()` hangs on first cold close
On a fresh install, calling `simulation_app.close()` after a short
one-shot script can hang for tens of minutes in shutdown. The CUDA
cache (`~/.nv/ComputeCache`) being cold for `sm_120` seems to be a
contributor. Long-running training jobs (e.g. `train.py` for 50+ iters)
exit cleanly — the issue is specific to short scripts that open and
immediately close. Wrap one-shot tests with `timeout` if they hang.

### Block-buffered stdout when piping
Python defaults to block-buffered stdout when the destination isn't a
TTY (e.g. piping to a file, or running under tools that capture output).
Always export `PYTHONUNBUFFERED=1` or pass `-u` when you need to watch
progress in real time.

### pip dependency resolver warnings during `./isaaclab.sh -i`
Pin conflicts between Isaac Sim wheels and Isaac Lab dependencies
(`psutil`, `click`, `starlette`, `packaging`). Non-fatal — packages
install successfully and runtime works. Don't try to "fix" them by
pinning versions; you'll cascade into other breakages.

### Training the Growbot task

Use `isaac_lab/train.py` (not IsaacLab's built-in `train.py`) to train Growbot.
It registers the Growbot task and routes logs to `<project_root>/logs/` automatically:

```bash
cd external/IsaacLab
PYTHONUNBUFFERED=1 ./isaaclab.sh -p ../../isaac_lab/train.py \
  --task Isaac-Velocity-Flat-Growbot-Direct-v0 --headless
```

Checkpoints land in `logs/rsl_rl/growbot_flat_direct/<timestamp>/`.
