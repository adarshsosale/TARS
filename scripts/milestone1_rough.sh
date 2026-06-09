#!/usr/bin/env bash
# ============================================================================
# Milestone 1 — Hexabot rough-terrain locomotion: train -> export -> video.
#
# One-shot pipeline you can run on your own with minimal setup. It:
#   1. trains the privileged height-scan TEACHER on curriculum rough terrain
#      (headless, fast, scalar logging to TensorBoard every iteration),
#   2. exports the best checkpoint to TorchScript/ONNX,
#   3. renders a short clip of it walking over rough terrain.
#
# Prereqs: conda env `env_isaaclab`, the converted USD at
#   hexabot_model/isaac/hexabot.usd (already in the repo), an RTX GPU.
#
# Usage:
#   bash scripts/milestone1_rough.sh                 # full defaults
#   NUM_ENVS=4096 MAX_ITER=3000 bash scripts/milestone1_rough.sh
#   SMOKE=1 bash scripts/milestone1_rough.sh         # tiny plumbing check
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${PROJECT_ROOT}/external/IsaacLab"
USD="${PROJECT_ROOT}/hexabot_model/isaac/hexabot.usd"

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"
SECONDS_CLIP="${SECONDS_CLIP:-12}"
CMD_VX="${CMD_VX:-0.2}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-6}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  NUM_ENVS=64; MAX_ITER=5; SECONDS_CLIP=4; TERRAIN_LEVEL=2
  echo "[milestone1] SMOKE mode: ${NUM_ENVS} envs x ${MAX_ITER} iters"
fi

[[ -f "${USD}" ]] || { echo "ERROR: missing USD ${USD} (run hexabot_model/generate_hexabot.py first)"; exit 1; }
cd "${ISAACLAB_DIR}"

echo "============================================================"
echo "[milestone1] 1/3  TRAIN rough-terrain teacher"
echo "             envs=${NUM_ENVS} iters=${MAX_ITER}"
echo "============================================================"
./isaaclab.sh -p ../../isaac_lab/train_rough.py \
  --task Isaac-Velocity-Rough-Hexabot-Direct-v0 \
  --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITER}" --headless ${EXTRA_TRAIN_ARGS} \
  || pkill -9 -f "isaac" || true

echo "============================================================"
echo "[milestone1] 2/3  EXPORT best checkpoint -> TorchScript/ONNX"
echo "============================================================"
./isaaclab.sh -p ../../isaac_lab/play_rough.py --select_best || pkill -9 -f "isaac" || true

echo "============================================================"
echo "[milestone1] 3/3  RENDER rough-terrain clip (latest checkpoint)"
echo "============================================================"
./isaaclab.sh -p ../../isaac_lab/render_rough.py \
  --seconds "${SECONDS_CLIP}" --cmd_vx "${CMD_VX}" --terrain_level "${TERRAIN_LEVEL}" \
  || pkill -9 -f "isaac" || true

# Isaac processes hang at simulation_app.close() holding the GPU — clean up.
pkill -9 -f "isaac" 2>/dev/null || true

echo "[milestone1] DONE."
echo "  logs (TensorBoard): ${PROJECT_ROOT}/logs/rsl_rl/hexabot_rough_direct/"
echo "  watch:  tensorboard --logdir ${PROJECT_ROOT}/logs/rsl_rl/hexabot_rough_direct"
echo "  lead metric: Curriculum/terrain_level   (mean terrain difficulty)"
echo "  video: ${PROJECT_ROOT}/hexabot_model/hexabot_rough_locomotion.mp4"
