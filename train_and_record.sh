#!/usr/bin/env bash
#
# train_and_record.sh — one-shot TARS (Growbot) locomotion pipeline.
#
#   1. Train the PPO policy   (real-time progress streamed to your terminal)
#   2. Export the policy to TorchScript (policy.pt)
#   3. Record a cinematic locomotion video for THIS run (unique filename)
#
# Everything for a run (train log, export log, record log, the .mp4, and a
# symlink to the raw rsl_rl log dir) lands in   runs/<RUN_ID>/   — kept separate
# from the raw logs/ tree so each run is self-contained.
#
# Isaac processes are known to HANG at simulation_app.close(); each stage is run
# in its own process group, streamed live, watched for a completion marker, then
# the hung group is killed -9. (See memory: tars-isaac-rl-setup.)
#
# Usage:
#   ./train_and_record.sh [options]
#
# Options:
#   --name NAME        label appended to the run id (e.g. "symmetry-v1")
#   --num-envs N       parallel envs for training        (default 4096)
#   --iters N          training iterations               (default 1500)
#   --cmd-vx V         commanded forward speed for video (default 0.3 m/s)
#   --seconds S        video length in seconds           (default 10)
#   --no-train         skip training; export + record the latest existing run
#   -h | --help        show this help
#
set -uo pipefail

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$PROJECT_ROOT/external/IsaacLab"
ISAACLAB_SH="$ISAACLAB_DIR/isaaclab.sh"
TASK="Isaac-Velocity-Flat-Growbot-Direct-v0"
EXPERIMENT="growbot_flat_direct"
LOGS_BASE="$PROJECT_ROOT/logs/rsl_rl/$EXPERIMENT"
CONDA_ENV="env_isaaclab"

# defaults (match memory: tars-isaac-rl-setup)
NAME=""
NUM_ENVS=4096
ITERS=1500
CMD_VX=0.3
SECONDS_LEN=10
DO_TRAIN=1

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)      NAME="$2"; shift 2 ;;
        --num-envs)  NUM_ENVS="$2"; shift 2 ;;
        --iters)     ITERS="$2"; shift 2 ;;
        --cmd-vx)    CMD_VX="$2"; shift 2 ;;
        --seconds)   SECONDS_LEN="$2"; shift 2 ;;
        --no-train)  DO_TRAIN=0; shift ;;
        -h|--help)   sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
[[ -n "$NAME" ]] && RUN_ID="${RUN_ID}_${NAME}"
RUN_DIR="$PROJECT_ROOT/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "============================================================"
echo "  TARS locomotion pipeline"
echo "  run id     : $RUN_ID"
echo "  output dir : $RUN_DIR"
echo "  train      : $([[ $DO_TRAIN -eq 1 ]] && echo "yes (envs=$NUM_ENVS iters=$ITERS)" || echo "no (use latest run)")"
echo "  video      : ${SECONDS_LEN}s @ cmd_vx=${CMD_VX} m/s"
echo "============================================================"

# ---------------------------------------------------------------------------
# Conda env
# ---------------------------------------------------------------------------
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if command -v conda >/dev/null 2>&1; then
        # shellcheck disable=SC1091
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV" || { echo "✖ could not activate conda env '$CONDA_ENV'"; exit 1; }
    else
        echo "⚠ conda not found; assuming '$CONDA_ENV' deps are already on PATH"
    fi
fi
echo "  conda env  : ${CONDA_DEFAULT_ENV:-<none>}"

[[ -x "$ISAACLAB_SH" ]] || { echo "✖ isaaclab.sh not found/executable at $ISAACLAB_SH"; exit 1; }

# ---------------------------------------------------------------------------
# kill_tree PID — SIGKILL a process and all of its descendants (depth first).
# Used because Isaac hangs at simulation_app.close(); we can't wait on it.
# ---------------------------------------------------------------------------
kill_tree() {
    local p="$1" k
    for k in $(pgrep -P "$p" 2>/dev/null); do kill_tree "$k"; done
    kill -9 "$p" 2>/dev/null
}

# ---------------------------------------------------------------------------
# run_stage NAME LOGFILE MARKER -- <command...>
#
# Runs <command>, tees its output to LOGFILE *and* the terminal (real-time),
# and waits until MARKER appears in the log or the process exits on its own.
# Then kills the (possibly hung-on-close) process tree. Returns 0 only if
# MARKER was seen.
# ---------------------------------------------------------------------------
run_stage() {
    local name="$1" logf="$2" marker="$3"; shift 3
    [[ "$1" == "--" ]] && shift

    echo
    echo "▶ $name — live output below (also saved to $logf)"
    echo "------------------------------------------------------------"

    : > "$logf"
    PYTHONUNBUFFERED=1 "$@" >>"$logf" 2>&1 &
    local pid=$!

    # live stream to terminal; tail dies when the worker pid dies
    tail -n +1 -f --pid="$pid" "$logf" &
    local tpid=$!

    local found=1
    while kill -0 "$pid" 2>/dev/null; do
        if grep -q -- "$marker" "$logf" 2>/dev/null; then found=0; break; fi
        sleep 2
    done
    # let the final lines flush to screen
    sleep 1
    kill "$tpid" 2>/dev/null

    # double-check (covers a clean, fast exit that beat the loop)
    grep -q -- "$marker" "$logf" 2>/dev/null && found=0

    # kill the whole tree (the python child typically hangs at simulation_app.close)
    kill_tree "$pid"
    wait "$pid" 2>/dev/null

    echo "------------------------------------------------------------"
    if [[ $found -ne 0 ]]; then
        echo "✖ $name FAILED — marker '$marker' never appeared. See $logf"
        return 1
    fi
    echo "✔ $name complete"
    return 0
}

cd "$ISAACLAB_DIR" || { echo "✖ cannot cd to $ISAACLAB_DIR"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Train
# ---------------------------------------------------------------------------
if [[ $DO_TRAIN -eq 1 ]]; then
    echo
    echo "  TensorBoard (optional, another terminal):"
    echo "    tensorboard --logdir \"$LOGS_BASE\""
    run_stage "TRAIN" "$RUN_DIR/train.log" "Training time:" -- \
        "$ISAACLAB_SH" -p "$PROJECT_ROOT/isaac_lab/train.py" \
        --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$ITERS" --headless \
        || exit 1
fi

# ---------------------------------------------------------------------------
# Locate the run we just trained (latest dir) + its newest checkpoint
# ---------------------------------------------------------------------------
TRAIN_RUN_DIR="$(ls -dt "$LOGS_BASE"/*/ 2>/dev/null | head -1)"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR%/}"
[[ -d "$TRAIN_RUN_DIR" ]] || { echo "✖ no training run found under $LOGS_BASE"; exit 1; }
CKPT="$(ls -t "$TRAIN_RUN_DIR"/model_*.pt 2>/dev/null | head -1)"
[[ -f "$CKPT" ]] || { echo "✖ no model_*.pt checkpoint in $TRAIN_RUN_DIR"; exit 1; }
echo
echo "  using run dir   : $TRAIN_RUN_DIR"
echo "  using checkpoint: $(basename "$CKPT")"

# link the raw rsl_rl log dir into this run's folder (tensorboard, params, models)
ln -sfn "$TRAIN_RUN_DIR" "$RUN_DIR/log"

# ---------------------------------------------------------------------------
# 2. Export policy -> TorchScript (policy.pt) + ONNX
# ---------------------------------------------------------------------------
run_stage "EXPORT" "$RUN_DIR/export.log" "\[EXPORT\] wrote" -- \
    "$ISAACLAB_SH" -p "$PROJECT_ROOT/isaac_lab/play.py" \
    --task "$TASK" --checkpoint "$CKPT" --headless \
    || exit 1

POLICY="$TRAIN_RUN_DIR/exported/policy.pt"
[[ -f "$POLICY" ]] || { echo "✖ exported policy not found at $POLICY"; exit 1; }

# ---------------------------------------------------------------------------
# 3. Record the locomotion video (unique filename for this run)
# ---------------------------------------------------------------------------
VIDEO="$RUN_DIR/tars_${RUN_ID}.mp4"
run_stage "RECORD" "$RUN_DIR/record.log" "\[INFO\] wrote" -- \
    "$ISAACLAB_SH" -p "$PROJECT_ROOT/isaac_lab/record_policy.py" \
    --policy "$POLICY" --seconds "$SECONDS_LEN" --cmd_vx "$CMD_VX" --out "$VIDEO" \
    || exit 1

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo
echo "============================================================"
echo "  ✔ pipeline complete — $RUN_ID"
echo "------------------------------------------------------------"
echo "  video : $VIDEO"
echo "  logs  : $RUN_DIR/{train,export,record}.log"
echo "  models: $RUN_DIR/log  ->  $TRAIN_RUN_DIR"
echo "============================================================"
[[ -f "$VIDEO" ]] && grep -h "forward travel" "$RUN_DIR/record.log" 2>/dev/null
