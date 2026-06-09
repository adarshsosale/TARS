#!/usr/bin/env bash
# ============================================================================
# Render a rough-terrain clip of the LATEST Hexabot rough checkpoint, on demand.
# Run this any time (even mid-training) to watch the policy locomote over rough
# terrain. Loads the latest checkpoint of the latest rough run by default.
#
# Usage:
#   bash scripts/render_latest_rough.sh
#   SECONDS_CLIP=15 CMD_VX=0.25 TERRAIN_LEVEL=8 bash scripts/render_latest_rough.sh
#   CHECKPOINT=/abs/path/model_1500.pt bash scripts/render_latest_rough.sh
#   SELECT_BEST=1 bash scripts/render_latest_rough.sh   # render the checkpoint that
#                                                       # moves FARTHEST from origin
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${PROJECT_ROOT}/external/IsaacLab"

SECONDS_CLIP="${SECONDS_CLIP:-12}"
CMD_VX="${CMD_VX:-0.2}"
TERRAIN_LEVEL="${TERRAIN_LEVEL:-6}"
OUT="${OUT:-${PROJECT_ROOT}/hexabot_model/hexabot_rough_locomotion.mp4}"

CKPT_ARG=""
[[ -n "${CHECKPOINT:-}" ]] && CKPT_ARG="--checkpoint ${CHECKPOINT}"

SELECT_ARG=""
[[ -n "${SELECT_BEST:-}" ]] && SELECT_ARG="--select_best"

cd "${ISAACLAB_DIR}"
./isaaclab.sh -p ../../isaac_lab/render_rough.py \
  --seconds "${SECONDS_CLIP}" --cmd_vx "${CMD_VX}" --terrain_level "${TERRAIN_LEVEL}" \
  --out "${OUT}" ${CKPT_ARG} ${SELECT_ARG} \
  || pkill -9 -f "isaac" || true

pkill -9 -f "isaac" 2>/dev/null || true
echo "[render] wrote ${OUT}"
