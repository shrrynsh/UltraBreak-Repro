#!/usr/bin/env bash
#SBATCH --job-name=ub_nfgrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Follow-up to job 773 (D11 fix confirmed mechanically but ASR REGRESSED to the
# buggy baseline: SafeBench 23.49% / AdvBench 1.35%, text_loss stalled at 0.214).
# ============================================================================
#
# 773 proved the units fix works at the pixel level (saved std 36.2, full range)
# and that train==eval consistency is NOT what closes the gap. Its semantic loss
# plateaued at 0.214 with small oscillation and no net descent -- the same stall
# as the buggy job 622, and WORSE than the projection run 625 (0.189).
#
# Two competing explanations, tested here as ONE-variable changes from 773:
#
#   A  lr 0.01 -> 0.003   The fix multiplied the attack gradient by ~1/gamma
#                         (approx 3.7x) by removing the perceptual compression.
#                         lr=0.01 was tuned for the old squashed regime; the late
#                         oscillation is consistent with it now being too large.
#
#   B  steps 1300 -> 3000 Independent evidence says 1300 is under-trained: job
#                         622 improved SafeBench 30 -> 66 from 1300 -> 5000 steps.
#                         Same lr as 773, longer budget.
#
# Everything else is identical to 773: D11 fix on, projection OFF, --seed 0,
# 50-row corpus, save raw latent. Read as a grid against 773 (lr 0.01, 1300).
#
# Usage:  sbatch run_normfix_grid.sh <lr> <num_epochs>
#   Job A:  sbatch run_normfix_grid.sh 0.003 1301
#   Job B:  sbatch run_normfix_grid.sh 0.01  3001
# num_epochs is (target_step + 1); (num_epochs-1) must be a multiple of 20 so the
# final checkpoint is written by the epoch%20==0 saver.
#
# Compare (v3 judge, clean 315 / 520):
#   773 (fix, lr 0.01, 1300)   23.49% / 1.35%   text_loss 0.214
#   625 (projection)           48.25% / 22.31%  text_loss 0.189
#   authors' ultrabreak.png    78.73% / 67.69%
#   paper                      81.59% / 72.69%

set -euo pipefail

LR="${1:?usage: sbatch run_normfix_grid.sh <lr> <num_epochs>}"
NUM_EPOCHS="${2:?usage: sbatch run_normfix_grid.sh <lr> <num_epochs>}"
CKPT_STEP=$((NUM_EPOCHS - 1))
if (( CKPT_STEP % 20 != 0 )); then
  echo "[nfgrid] ERROR: num_epochs-1 (${CKPT_STEP}) must be a multiple of 20 so the checkpoint is saved" >&2
  exit 1
fi

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
LR_TAG="${LR/./p}"                       # 0.003 -> 0p003, for a filesystem-safe name
EXP_NAME="full50_normfix_lr${LR_TAG}_s${CKPT_STEP}"
TRAIN_CONFIG="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
SEED=0
IMAGE="outputs/${EXP_NAME}/${CKPT_STEP}.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[nfgrid ${EXP_NAME}]"

cd "$REPO_ROOT"
mkdir -p logs

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv missing - refusing to run under an unpinned env" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} lr=${LR} epochs=${NUM_EPOCHS} ckpt=${CKPT_STEP} seed=${SEED}"
echo "${LOG_PREFIX} D11 fix ON, projection OFF, save raw latent"
python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[nfgrid] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
n = len(pd.read_csv("train_configs/safebench-tiny_jailbroken_mode_full50.csv"))
assert n == 50, f"expected 50 training rows, got {n}"
print(f"[nfgrid] training corpus rows: {n}")

import inspect, sys
sys.path.insert(0, "optimisation")
from qwen2_adapter import Qwen2Adapter
src = inspect.getsource(Qwen2Adapter.compute_loss)
assert "normalised_patch = (patch - mean_tensor) / std_tensor" in src, "D11 fix missing!"
assert src.count("apply_patch(pixel_values") == 1, "expected a single (normalised) injection path"
print("[nfgrid] D11 fix verified present in Qwen2Adapter.compute_loss")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage A: training ====="
python optimisation/optimise_proj.py \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "$EXP_NAME" \
  --num_epochs "$NUM_EPOCHS" \
  --no_projection \
  --seed "$SEED" \
  --lr "$LR"

if [[ ! -f "$IMAGE" ]]; then
  echo "${LOG_PREFIX} Expected checkpoint not found: ${IMAGE}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Trained image ready: ${IMAGE}"

echo ""
echo "${LOG_PREFIX} ===== Pixel statistics ====="
python - "$IMAGE" <<'PY'
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64) / 255.0
print("  saved patch: std=%.1f  range=[%.0f,%.0f]" % (a.std()*255, a.min()*255, a.max()*255))
print("  reference -- 773 (fix, lr 0.01, 1300)  std=36.2")
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
echo "${LOG_PREFIX} text_loss dropping below ~0.19 AND ASR rising => this knob was the blocker."
echo "${LOG_PREFIX} still ~0.214 / ~23% => full-contrast retraining is harder than a scale/budget fix."
python evaluation/summarise_asr.py --paper --pattern "${EXP_NAME}" || true

echo ""
echo "${LOG_PREFIX} Done"
