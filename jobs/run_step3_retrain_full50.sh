#!/usr/bin/env bash
#SBATCH --job-name=ultrabreak_retrain50
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Step 3: the decisive test for Defect B (weak retrained image).
#
# Retrains the adversarial image under the AUTHORS' PINNED ENVIRONMENT (repro
# venv: transformers 4.51.3 / qwen-vl-utils 0.0.11 / torch 2.5.1) on the FULL
# 50-query SafeBench-Tiny corpus (all 10 categories). Both differences vs the
# user's earlier run:
#   - earlier training ran under transformers 5.10.0.dev0 (suspected to misalign
#     Qwen2-VL image preprocessing, so the patch failed on its own surrogate);
#   - earlier corpus was 35 rows (3 categories dropped by the config generator).
#
# After training it evaluates the step-1300 image on the Qwen2-VL SURROGATE for
# both benchmarks. The prior retrained image scored 27% SafeBench / 8% AdvBench
# on Qwen2-VL while transferring to LLaVA at ~90%. If retraining under the pinned
# env lifts the Qwen2-VL surrogate toward the authors' ~68-78%, Defect B is the
# environment mismatch.

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
EXP_NAME="safebench_tiny_full50_repro"
TRAIN_CONFIG="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
NUM_EPOCHS=1400                       # saves the step-1300 checkpoint (+ margin)
CKPT_STEP=1300
IMAGE="outputs/${EXP_NAME}/${CKPT_STEP}.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[retrain50]"

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
python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[retrain50] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
n = len(pd.read_csv("train_configs/safebench-tiny_jailbroken_mode_full50.csv"))
assert n == 50, f"expected 50 training rows, got {n}"
print(f"[retrain50] training corpus rows: {n}")
PY

# ---------------------------------------------------------------------------
# Stage A: training  (first epoch doubles as the env preflight — a 4.51.3
# incompatibility would raise here, right after model load)
# ---------------------------------------------------------------------------
echo ""
echo "${LOG_PREFIX} ===== Stage A: training ====="
python optimisation/optimise.py \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "$EXP_NAME" \
  --num_epochs "$NUM_EPOCHS"

if [[ ! -f "$IMAGE" ]]; then
  echo "${LOG_PREFIX} Expected checkpoint not found: ${IMAGE}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Trained image ready: ${IMAGE}"

# ---------------------------------------------------------------------------
# Stage B: evaluate the step-1300 image on the Qwen2-VL surrogate
# (guarded: training output is already saved, so an eval hiccup is recoverable)
# ---------------------------------------------------------------------------
echo ""
echo "${LOG_PREFIX} ===== Stage B: evaluate step-${CKPT_STEP} on ${MODEL} ====="

SB_CFG="${EXP_NAME}_safebench_eval"      # 315-row held-out SafeBench
AB_CFG="${EXP_NAME}_advbench_eval"       # 520-row AdvBench, normalized TPG

# SafeBench: exclude 3 eval categories + SafeBench-Tiny -> 315 rows; already-canonical targets.
python create_attack_configs.py --dataset safebench --config-type attack \
  --exclude-train datasets/SafeBench-Tiny.csv --image "$IMAGE" \
  --output "attack_configs/${SB_CFG}.csv"

# AdvBench: normalized "Steps to ..." TPG prompts (paper method), canonical goals in target.
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
echo "${LOG_PREFIX} ===== RESULT (step-${CKPT_STEP} image, ${MODEL} surrogate) ====="
python evaluation/summarise_asr.py --paper --pattern "${EXP_NAME}" || true

echo ""
echo "${LOG_PREFIX} Done"
