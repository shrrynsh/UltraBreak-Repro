#!/usr/bin/env bash
#SBATCH --job-name=ub_5b_unclmp_img
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Step 5b: projection WITHOUT the raw-latent clamp AND with a corrected init
# (Edit 1 + Edit 2).
#
# Identical to Step 5a except for the initialisation. Once the clamp is gone
# adv_patch is a latent, not the image, so torch.rand() no longer means "a
# random image" -- it starts the run inside the very band 5a/5b free it from:
#
#   --init_space unit  (5a)  x ~ U(0,1)          -> image [0.408, 0.750] std 0.083
#   --init_space image (5b)  x = (p - beta)/gamma -> image [0.000, 1.000] std 0.289
#
# 5a vs 5b therefore isolates the init change on its own.
# See repro_notes/DISCREPANCIES.md O1 (the clamp) and O2 (this init).
#
# NUM_EPOCHS=1301 gives epochs 0..1300, so 1300.png exists and has had exactly
# the same number of steps applied as job 625's 1300.png -- directly comparable.
# Paper Table 4 budget is 1300 iterations.
#
# Compare against:
#   job 622 (no projection, clamp)      29.84% SafeBench / 1.73% AdvBench
#   job 625 (projection + clamp)        48.25% / 22.31%
#   authors' ultrabreak.png             77.71% / 67.69%
#   paper                               81.59% / 72.69%

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
EXP_NAME="full50_proj_unclamped_imginit"
TRAIN_CONFIG="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
NUM_EPOCHS=1301
CKPT_STEP=1300
INIT_SPACE="image"
IMAGE="outputs/${EXP_NAME}/${CKPT_STEP}.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[5b-imginit]"

cd "$REPO_ROOT"
mkdir -p logs

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv missing - refusing to run under an unpinned env" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} exp=${EXP_NAME} corpus=${TRAIN_CONFIG} epochs=${NUM_EPOCHS}"
echo "${LOG_PREFIX} PROJECTION ON, latent clamp OFF, init_space=${INIT_SPACE}"
python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[5b-imginit] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
n = len(pd.read_csv("train_configs/safebench-tiny_jailbroken_mode_full50.csv"))
assert n == 50, f"expected 50 training rows, got {n}"
print(f"[5b-imginit] training corpus rows: {n}")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage A: training (projection, no latent clamp) ====="
python optimisation/optimise_proj.py \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "$EXP_NAME" \
  --num_epochs "$NUM_EPOCHS" \
  --init_space "$INIT_SPACE"

if [[ ! -f "$IMAGE" ]]; then
  echo "${LOG_PREFIX} Expected checkpoint not found: ${IMAGE}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Trained image ready: ${IMAGE}"

echo ""
echo "${LOG_PREFIX} ===== Pixel statistics (expect FULL [0,1] range, unlike job 625) ====="
python - "$IMAGE" <<'PY'
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64) / 255.0
for c, n in enumerate("RGB"):
    ch = a[:, :, c]
    print(f"  {n}: min={ch.min():.4f} max={ch.max():.4f} std={ch.std():.4f}")
print("  reference -- authors' ultrabreak.png R: min=0.0000 max=1.0000 std=0.1649")
print("  reference -- job 625 (clamped)     R: min=0.4780 max=0.7410 std=0.0250")
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
echo "${LOG_PREFIX} ===== RESULT (step-${CKPT_STEP}, unclamped projection, image init) ====="
python evaluation/summarise_asr.py --paper --pattern "${EXP_NAME}" || true

echo ""
echo "${LOG_PREFIX} Done"
