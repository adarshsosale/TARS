#!/usr/bin/env bash
# ============================================================================
# Phase E rough-terrain run — self-serve: train -> export -> video -> eval ->
# one paste-able report. Run this YOURSELF in a terminal so you see real-time
# progress (no agent tokens burned); when it finishes, paste the report file
# it prints back into Claude Code.
#
# Default mode RESUMES the interrupted Phase E run (latest run dir that has
# checkpoints) from its newest model_*.pt and trains to TOTAL_ITER total.
# rsl_rl counts max_iterations as ADDITIONAL on resume, so the script computes
# the remainder itself.
#
# Usage (from the project root):
#   bash scripts/run_phaseE.sh                  # resume interrupted run -> 3000
#   FRESH=1 bash scripts/run_phaseE.sh          # ignore old run, start over
#   REPORT_ONLY=1 bash scripts/run_phaseE.sh    # just regenerate the report
#   SKIP_EVAL=1 bash scripts/run_phaseE.sh      # skip the 2x7 eval sweeps
#
# Stages (multi-hour total on the RTX PRO 6000):
#   1. train  (milestone1_rough.sh: train_rough.py, headless, live console)
#   2. export best checkpoint (play_rough.py --select_best)
#   3. render level-6 clip (render_rough.py)
#   4. eval all 7 terrain families at d=0.8 and d=0.6 (eval_rough.sh)
#   5. write logs/phaseE_report_<ts>.md   <-- paste this back
# ============================================================================
set -uo pipefail   # no -e: Isaac exits 1 on the known simulation_app.close() hang

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP_DIR="${PROJECT_ROOT}/logs/rsl_rl/hexabot_rough_direct"
TS="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_LOG="${PROJECT_ROOT}/logs/phaseE_run_${TS}.log"
REPORT="${PROJECT_ROOT}/logs/phaseE_report_${TS}.md"
TOTAL_ITER="${TOTAL_ITER:-3000}"

cleanup() { pkill -9 -f "isaac" 2>/dev/null || true; }
trap cleanup EXIT

# ---------------------------------------------------------------- conda env --
# isaaclab.sh silently falls back to base python without this -> ModuleNotFoundError
if [[ "${CONDA_DEFAULT_ENV:-}" != "env_isaaclab" ]]; then
  source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
  conda activate env_isaaclab || { echo "ERROR: conda activate env_isaaclab failed"; exit 1; }
fi

# ------------------------------------------------------------ stale processes --
if [[ "${REPORT_ONLY:-0}" != "1" ]] && pgrep -f "isaaclab|isaac-sim|omni.kit" >/dev/null 2>&1; then
  echo "ERROR: Isaac processes still running (they hang holding GPU memory)."
  echo "       Kill them first:  pkill -9 -f isaac    then re-run."
  exit 1
fi

# ------------------------------------------------------------------ training --
if [[ "${REPORT_ONLY:-0}" != "1" ]]; then
  if [[ "${FRESH:-0}" == "1" ]]; then
    MODE="fresh"
    MAX_ITER="${TOTAL_ITER}"
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
    echo "[phaseE] FRESH run: ${MAX_ITER} iterations from scratch"
  else
    # latest run dir that actually has checkpoints, and its newest model_*.pt
    RESUME_RUN=""
    for d in $(ls -td "${EXP_DIR}"/*/ 2>/dev/null); do
      if compgen -G "${d}model_*.pt" >/dev/null; then RESUME_RUN="$(basename "$d")"; break; fi
    done
    [[ -n "${RESUME_RUN}" ]] || { echo "ERROR: no resumable run in ${EXP_DIR} (use FRESH=1)"; exit 1; }
    CKPT_FILE="$(ls "${EXP_DIR}/${RESUME_RUN}"/model_*.pt | sort -t_ -k2 -n | tail -1)"
    CKPT_ITER="$(basename "${CKPT_FILE}" .pt | cut -d_ -f2)"
    if (( CKPT_ITER >= TOTAL_ITER )); then
      echo "[phaseE] ${RESUME_RUN} already at iter ${CKPT_ITER} >= ${TOTAL_ITER}; skipping training."
      MODE="already-done"; MAX_ITER=0
    else
      MODE="resume ${RESUME_RUN} @ model_${CKPT_ITER}"
      MAX_ITER=$(( TOTAL_ITER - CKPT_ITER ))   # rsl_rl: max_iterations is ADDITIONAL on resume
      EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-} --resume --load_run ${RESUME_RUN} --checkpoint $(basename "${CKPT_FILE}")"
      echo "[phaseE] RESUME ${RESUME_RUN} from model_${CKPT_ITER}: ${MAX_ITER} more iters -> ${TOTAL_ITER} total"
    fi
  fi

  if [[ "${MAX_ITER}" != "0" ]]; then
    echo "[phaseE] console log tee'd to: ${RUN_LOG}"
    echo "[phaseE] live curves: tensorboard --logdir '${EXP_DIR}'  (lead metric: Curriculum/terrain_level)"
    MAX_ITER="${MAX_ITER}" EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS}" \
      bash "${PROJECT_ROOT}/scripts/milestone1_rough.sh" 2>&1 | tee "${RUN_LOG}"
  else
    # still export + render so the report has fresh artifacts
    ( cd "${PROJECT_ROOT}/external/IsaacLab" \
      && ./isaaclab.sh -p ../../isaac_lab/play_rough.py --select_best 2>&1 | tee "${RUN_LOG}" \
      ;  cleanup \
      ;  ./isaaclab.sh -p ../../isaac_lab/render_rough.py --seconds 12 --cmd_vx 0.2 --terrain_level 6 2>&1 | tee -a "${RUN_LOG}" )
    cleanup
  fi

  # -------------------------------------------------------------------- eval --
  if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
    for D in 0.8 0.6; do
      echo "[phaseE] eval sweep d=${D} (7 families, one Isaac process each)..."
      SELECT_BEST=1 DIFFICULTY="${D}" bash "${PROJECT_ROOT}/scripts/eval_rough.sh" 2>&1 | tee -a "${RUN_LOG}"
      cleanup; sleep 2
    done
  fi
fi

# ------------------------------------------------------------------- report --
# Plain python3, no Isaac. Parses the newest console log + merged eval JSONs,
# compares to the committed model_400 baselines.
LATEST_LOG="$(ls -t "${PROJECT_ROOT}"/logs/phaseE_run_*.log "${PROJECT_ROOT}"/logs/phaseE_train.log 2>/dev/null | head -1)"
python3 - "${PROJECT_ROOT}" "${EXP_DIR}" "${LATEST_LOG:-}" "${REPORT}" "${MODE:-report-only}" <<'PY'
import glob, json, os, re, sys
root, exp_dir, run_log, report, mode = sys.argv[1:6]

KEYS = [
    "Learning iteration", "Mean episode length", "Episode_Termination/died",
    "Curriculum/terrain_level", "Metrics/height_target_offset",
    "Episode_Reward/forward_progress", "Episode_Reward/belly_contact_force",
    "Episode_Reward/foot_stumble", "Episode_Reward/gait_phase",
    "Episode_Reward/foot_clearance", "Mean reward",
]
last = {}
if run_log and os.path.isfile(run_log):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    with open(run_log, errors="replace") as f:
        for line in f:
            line = ansi.sub("", line).strip()
            for k in KEYS:
                if line.startswith(k):
                    last[k] = line.split(":", 1)[-1].strip() if ":" in line else line
            m = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
            if m:
                last["Learning iteration"] = f"{m.group(1)}/{m.group(2)}"

run_dirs = sorted(glob.glob(os.path.join(exp_dir, "*/")), key=os.path.getmtime, reverse=True)
run_dir = run_dirs[0].rstrip("/") if run_dirs else "(none)"

def load_eval(d):
    cands = sorted(glob.glob(os.path.join(exp_dir, "*", "eval", f"*_d{d}.json")),
                   key=os.path.getmtime, reverse=True)
    return (json.load(open(cands[0])), cands[0]) if cands else (None, None)

def load_base(d):
    p = os.path.join(root, "docs", "eval_baselines", f"model_400_d{d}.json")
    return json.load(open(p)) if os.path.isfile(p) else None

lines = [f"# Phase E run report", "",
         f"- mode: {mode}",
         f"- run dir: {run_dir}",
         f"- console log: {run_log or '(none)'}", ""]

lines.append("## Training (final console block)")
if last:
    for k in KEYS:
        if k in last:
            lines.append(f"- {k}: {last[k]}")
else:
    lines.append("- (no metrics found in console log)")
lines.append("")

for d in ("0.8", "0.6"):
    ev, evp = load_eval(d)
    base = load_base(d)
    lines.append(f"## Eval d={d}" + (f"  ({os.path.relpath(evp, root)})" if evp else ""))
    if not ev:
        lines.append("- (no merged eval JSON found)\n")
        continue
    lines.append(f"- checkpoint: {ev['checkpoint']}   min_success={ev['min_success_rate']:.3f}"
                 f"   mean_success={ev['mean_success_rate']:.3f}")
    lines.append("")
    lines.append("| family | success | mean_d (m) | baseline(model_400) | delta |")
    lines.append("|---|---|---|---|---|")
    for fam, m in sorted(ev["families"].items()):
        b = (base or {}).get("families", {}).get(fam, {})
        bs = b.get("success_rate")
        delta = f"{m['success_rate'] - bs:+.3f}" if bs is not None else "n/a"
        lines.append(f"| {fam} | {m['success_rate']:.3f} | {m['mean_distance_m']:.2f} | "
                     f"{bs if bs is not None else 'n/a'} | {delta} |")
    lines.append("")

lines.append("## Artifacts")
for p in ("hexabot_model/hexabot_rough_locomotion.mp4",):
    full = os.path.join(root, p)
    lines.append(f"- {p}: " + ("present" if os.path.isfile(full) else "MISSING"))
exported = glob.glob(os.path.join(run_dir, "exported", "policy.pt"))
lines.append(f"- exported policy: {exported[0] if exported else 'MISSING'}")
lines.append("")
lines.append("_Paste this whole file back into Claude Code for analysis._")

os.makedirs(os.path.dirname(report), exist_ok=True)
with open(report, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\n[phaseE] report written -> {report}")
PY

echo ""
echo "============================================================"
echo "[phaseE] DONE. Paste this file back into Claude Code:"
echo "         ${REPORT}"
echo "============================================================"
