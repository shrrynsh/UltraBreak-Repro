#!/usr/bin/env bash
#SBATCH --job-name=ub_projvar
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Does the Section 3.2 projection help once we have the budget (cf. job 809)?
# ==========================================================================
#
# Job 809 (projection OFF + D11 fix + 3000 steps) reached SafeBench 83.49% but
# AdvBench only 39.04%. The projection's stated purpose is transfer robustness,
# so it MIGHT trade white-box strength for generalisation. Two variants, both at
# 3000 steps so they read directly against 809:
#
#   projfix       projection ON + fix ON.  Model perceives clip(gamma*x+beta) --
#                 the 27% grey band -- consistently at train and eval. This is the
#                 faithful "Reading B" (x is the image, projection compresses it).
#                 Job 683 T1 predicts weaker white-box; this tests AdvBench.
#
#   doublesquash  projection ON + fix OFF.  Exactly job 625's regime (project then
#                 inject raw -> model perceives gamma*clip(gamma*x+beta)+beta, a
#                 doubly-compressed band, mismatched to the singly-compressed saved
#                 image) but at 3000 steps instead of 625's 1300. Tests whether
#                 625's regime, given 809's budget, also climbs.
#
# The fix is toggled via --raw_patch_injection (cfg.normalise_patch); default is
# the fix. Everything else matches 809: 50-row corpus, --seed 0, init unit.
#
# Usage:  sbatch run_proj_variants.sh <projfix|doublesquash> <num_epochs>
#   sbatch run_proj_variants.sh projfix      3001
#   sbatch run_proj_variants.sh doublesquash 3001
# (num_epochs-1) must be a multiple of 20 so the final checkpoint is written.
#
# Compare (v3 judge, clean 315 / 520):
#   809 (proj off, fix, 3000)   83.49% / 39.04%
#   625 (proj on, no fix, 1300) 48.25% / 22.31%
#   authors' ultrabreak.png     78.73% / 67.69%
#   paper                       81.59% / 72.69%

set -euo pipefail

MODE="${1:?usage: sbatch run_proj_variants.sh <projfix|doublesquash> <num_epochs>}"
NUM_EPOCHS="${2:?usage: sbatch run_proj_variants.sh <projfix|doublesquash> <num_epochs>}"
CKPT_STEP=$((NUM_EPOCHS - 1))
if (( CKPT_STEP % 20 != 0 )); then
  echo "[projvar] ERROR: num_epochs-1 (${CKPT_STEP}) must be a multiple of 20" >&2
  exit 1
fi

case "$MODE" in
  projfix)      RAW_FLAG="";                       FIX_STATE="ON"  ;;
  doublesquash) RAW_FLAG="--raw_patch_injection";  FIX_STATE="OFF" ;;
  *) echo "[projvar] ERROR: mode must be 'projfix' or 'doublesquash', got '${MODE}'" >&2; exit 1 ;;
esac

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
EXP_NAME="full50_proj_${MODE}_s${CKPT_STEP}"
TRAIN_CONFIG="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
SEED=0
IMAGE="outputs/${EXP_NAME}/${CKPT_STEP}.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[projvar ${EXP_NAME}]"

cd "$REPO_ROOT"
mkdir -p logs

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv missing - refusing to run under an unpinned env" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} mode=${MODE} projection=ON fix=${FIX_STATE} epochs=${NUM_EPOCHS} ckpt=${CKPT_STEP} seed=${SEED}"
python - "$MODE" <<'PY'
import importlib.metadata as m, sys, inspect
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[projvar] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
n = len(pd.read_csv("train_configs/safebench-tiny_jailbroken_mode_full50.csv"))
assert n == 50, f"expected 50 training rows, got {n}"
print(f"[projvar] training corpus rows: {n}")

sys.path.insert(0, "optimisation")
from utils import AblationConfig
assert AblationConfig().normalise_patch is True, "fix must default ON"
from qwen2_adapter import Qwen2Adapter
src = inspect.getsource(Qwen2Adapter.compute_loss)
assert "if self.cfg.normalise_patch" in src, "normalise toggle missing"
assert src.count("apply_patch(pixel_values") == 1, "expected a single injection path"
print(f"[projvar] toggle present; this run fix={sys.argv[1]!r}")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage A: training (projection ON, fix ${FIX_STATE}) ====="
# No --no_projection => projection is ON. ${RAW_FLAG} toggles the D11 fix off.
python optimisation/optimise_proj.py \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "$EXP_NAME" \
  --num_epochs "$NUM_EPOCHS" \
  --seed "$SEED" \
  ${RAW_FLAG}

if [[ ! -f "$IMAGE" ]]; then
  echo "${LOG_PREFIX} Expected checkpoint not found: ${IMAGE}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Trained image ready: ${IMAGE}"

echo ""
echo "${LOG_PREFIX} ===== Pixel statistics (projection saves a band-confined image) ====="
python - "$IMAGE" <<'PY'
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64) / 255.0
print("  saved patch: std=%.1f  range=[%.0f,%.0f]" % (a.std()*255, a.min()*255, a.max()*255))
print("  reference -- 809 (proj off) std=33.9 full range;  625 (proj on) std=9.8 band")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage B: evaluate step-${CKPT_STEP} on ${MODEL} ====="

SB_CFG="${EXP_NAME}_safebench_eval"
AB_CFG="${EXP_NAME}_advbench_eval"

python create_attack_configs.py --dataset safebench --config-type attack \
  --exclude-train datasets/SafeBench-Tiny.csv --image "$IMAGE" \
  --output "attack_configs/${SB_CFG}.csv"

python create_attack_configs.py --dataset advbench --config-type attack --normalize \
  --image "$IMAGE" --output "attack_configs/${AB_CFG}.csv"

score_qwen() {
  local cfg="$1"; shift
  echo "${LOG_PREFIX} generating: ${cfg}"
  python evaluation/attack.py --model_name "$MODEL" --attack_config "$cfg"
  local result="results/${cfg}/${MODEL}.csv"
  if [[ ! -f "$result" ]]; then
    echo "${LOG_PREFIX} generation missing: ${result}" >&2
    return 1
  fi
  echo "${LOG_PREFIX} judging: ${result}"
  python evaluation/evaluate.py --attack_result "$result" \
    --output_suffix "_harmbench_v3.csv" "$@"
}

score_qwen "$SB_CFG"
score_qwen "$AB_CFG" --behaviors_csv datasets/adv_bench.csv --behaviors_col goal

echo ""
echo "${LOG_PREFIX} ===== RESULT (${EXP_NAME}) ====="
echo "${LOG_PREFIX} Against 809 (proj off): SafeBench 83.49% / AdvBench 39.04%."
echo "${LOG_PREFIX} AdvBench above 39% => projection helps generalisation (keep it)."
echo "${LOG_PREFIX} Both below 809 => projection is harmful post-fix (band confinement, per 683 T1)."
python evaluation/summarise_asr.py --paper --pattern "${EXP_NAME}" || true

echo ""
echo "${LOG_PREFIX} Done"
