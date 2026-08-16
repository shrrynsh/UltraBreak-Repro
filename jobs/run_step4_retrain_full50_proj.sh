#!/usr/bin/env bash
#SBATCH --job-name=ultrabreak_retrain50proj
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Step 4: projection-enabled ablation of the Defect B retrain.
#
# Identical to Step 3 (run_step3_retrain_full50.sh) — pinned repro env,
# full 50-query SafeBench-Tiny corpus, Qwen2-VL surrogate — EXCEPT it trains
# with optimise_proj.py, which wires in the Section 3.2 projection the released
# code omits:  x_proj = clip(CLIP_STD * x + CLIP_MEAN, 0, 1).
#
# Read alongside Step 3:
#   Step 3 (no projection) vs Step 4 (projection) isolates the projection's
#   effect on the Qwen2-VL surrogate ASR. Prior retrained image: 27% SafeBench /
#   8% AdvBench on Qwen2-VL; authors' image ~78% / ~68%.

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
EXP_NAME="safebench_tiny_full50_proj_repro"
TRAIN_CONFIG="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
NUM_EPOCHS=1400
CKPT_STEP=1300
IMAGE="outputs/${EXP_NAME}/${CKPT_STEP}.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[retrain50proj]"

cd "$REPO_ROOT"
mkdir -p logs

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv missing - refusing to run under an unpinned env" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} exp=${EXP_NAME} corpus=${TRAIN_CONFIG} epochs=${NUM_EPOCHS} (PROJECTION ON)"
python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[retrain50proj] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
n = len(pd.read_csv("train_configs/safebench-tiny_jailbroken_mode_full50.csv"))
assert n == 50, f"expected 50 training rows, got {n}"
print(f"[retrain50proj] training corpus rows: {n}")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage A: training (with Section 3.2 projection) ====="
python optimisation/optimise_proj.py \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "$EXP_NAME" \
  --num_epochs "$NUM_EPOCHS"

if [[ ! -f "$IMAGE" ]]; then
  echo "${LOG_PREFIX} Expected checkpoint not found: ${IMAGE}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Trained image ready: ${IMAGE}"

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
echo "${LOG_PREFIX} ===== RESULT (step-${CKPT_STEP} PROJECTED image, ${MODEL} surrogate) ====="
python evaluation/summarise_asr.py --paper --pattern "${EXP_NAME}" || true

echo ""
echo "${LOG_PREFIX} Done"
