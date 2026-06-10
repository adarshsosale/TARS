#!/usr/bin/env bash
# ============================================================================
# Milestone 1.5 Phase A — held-out per-family eval of the Hexabot rough teacher.
#
# Loops the seven terrain families (one deterministic Isaac process each, because
# the SimulationApp/stage is a singleton — see eval_rough.py), then merges the
# per-family JSONs into one  <run>/eval/<checkpoint>_d{D}.json  summary.
#
# Defaults: latest run, its latest checkpoint, difficulty 0.8, seed 42, 256 envs.
# Knobs (env vars):
#   CHECKPOINT=/abs/model_2500.pt   evaluate a specific checkpoint
#   DIFFICULTY=0.6                  fixed terrain difficulty
#   SEED=42                         master seed (determinism)
#   NUM_ENVS=256                    episodes per family
#   SELECT_BEST=1                   pick the best walker among model_*.pt first
#
# Usage:
#   bash scripts/eval_rough.sh
#   DIFFICULTY=0.6 bash scripts/eval_rough.sh
#   CHECKPOINT=/abs/.../model_2500.pt DIFFICULTY=0.8 bash scripts/eval_rough.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_DIR="${PROJECT_ROOT}/external/IsaacLab"
EXP_DIR="${PROJECT_ROOT}/logs/rsl_rl/hexabot_rough_direct"

DIFFICULTY="${DIFFICULTY:-0.8}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-256}"

FAMILIES=(slope_noise_up slope_noise_down stairs_noise_up stairs_noise_down slope_boxes stairs_pure_down random_rough)

cd "${ISAACLAB_DIR}"

# --- resolve the checkpoint once so all families evaluate the SAME model --------
if [[ -n "${CHECKPOINT:-}" ]]; then
  CKPT="${CHECKPOINT}"
elif [[ "${SELECT_BEST:-0}" == "1" ]]; then
  echo "[eval] resolving best checkpoint (fwd velocity) ..."
  RESOLVE_OUT="$(./isaaclab.sh -p ../../isaac_lab/eval_rough.py \
      --family random_rough --select_best --resolve_only --seed "${SEED}" 2>&1 || true)"
  echo "${RESOLVE_OUT}" | tail -5
  CKPT="$(echo "${RESOLVE_OUT}" | grep -oP 'RESOLVED_CHECKPOINT=\K.*' | tail -1)"
  pkill -9 -f "isaac" 2>/dev/null || true
  [[ -n "${CKPT}" ]] || { echo "ERROR: could not resolve best checkpoint"; exit 1; }
else
  # latest run, its highest-numbered checkpoint (matches get_checkpoint_path default)
  RUN_DIR="$(ls -td "${EXP_DIR}"/*/ 2>/dev/null | head -1)"
  [[ -n "${RUN_DIR}" ]] || { echo "ERROR: no runs in ${EXP_DIR}"; exit 1; }
  CKPT="$(ls "${RUN_DIR}"model_*.pt 2>/dev/null | sort -t_ -k2 -n | tail -1)"
  [[ -n "${CKPT}" ]] || { echo "ERROR: no model_*.pt in ${RUN_DIR}"; exit 1; }
fi

RUN_DIR="$(dirname "${CKPT}")"
CKPT_TAG="$(basename "${CKPT}" .pt)"
echo "============================================================"
echo "[eval] checkpoint = ${CKPT}"
echo "[eval] difficulty = ${DIFFICULTY}  seed = ${SEED}  num_envs = ${NUM_ENVS}"
echo "============================================================"

for fam in "${FAMILIES[@]}"; do
  echo "------------------------------------------------------------"
  echo "[eval] family: ${fam}"
  echo "------------------------------------------------------------"
  ./isaaclab.sh -p ../../isaac_lab/eval_rough.py \
    --family "${fam}" --difficulty "${DIFFICULTY}" --seed "${SEED}" \
    --num_envs "${NUM_ENVS}" --checkpoint "${CKPT}" \
    || pkill -9 -f "isaac" || true
  # Isaac hangs at simulation_app.close() holding the GPU — clean up between runs.
  pkill -9 -f "isaac" 2>/dev/null || true
  sleep 2
done

# --- merge the per-family parts into one summary JSON (plain python, no Isaac) ---
MERGED="${RUN_DIR}/eval/${CKPT_TAG}_d${DIFFICULTY}.json"
python3 - "$RUN_DIR" "$CKPT_TAG" "$DIFFICULTY" "$MERGED" <<'PY'
import glob, json, os, sys
run_dir, ckpt_tag, diff, merged = sys.argv[1:5]
parts_glob = os.path.join(run_dir, "eval", "parts", f"{ckpt_tag}_d{float(diff):g}__*.json")
parts = sorted(glob.glob(parts_glob))
families = {}
for p in parts:
    with open(p) as f:
        m = json.load(f)
    families[m["family"]] = m
worst = min((m["success_rate"] for m in families.values()), default=0.0)
mean = (sum(m["success_rate"] for m in families.values()) / len(families)) if families else 0.0
max_stand = max((m["stand_when_cmd_frac"] for m in families.values()), default=0.0)
summary = {
    "checkpoint": ckpt_tag,
    "difficulty": float(diff),
    "n_families": len(families),
    "min_success_rate": worst,
    "mean_success_rate": mean,
    "max_stand_when_cmd_frac": max_stand,
    "families": families,
}
os.makedirs(os.path.dirname(merged), exist_ok=True)
with open(merged, "w") as f:
    json.dump(summary, f, indent=2)
print(f"[merge] {len(families)} families -> {merged}")
print(f"[merge] min_success_rate={worst:.3f}  mean_success_rate={mean:.3f}  max_stand={max_stand:.3f}")
for fam, m in families.items():
    print(f"  {fam:>18}: success={m['success_rate']:.3f}  mean_d={m['mean_distance_m']:.2f}m  "
          f"p10_d={m['p10_distance_m']:.2f}m  stand={m['stand_when_cmd_frac']:.3f}")
PY

pkill -9 -f "isaac" 2>/dev/null || true
echo "[eval] DONE -> ${MERGED}"
