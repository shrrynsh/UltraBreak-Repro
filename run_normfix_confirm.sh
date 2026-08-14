#!/usr/bin/env bash
#SBATCH --job-name=ub_normfix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Confirmatory experiment for DISCREPANCIES D11 (Option A: normalise at injection).
# ================================================================================
#
# THE FIX. qwen2_adapter.py now CLIP-normalises the patch before compositing it
# into `pixel_values` (which the processor has already normalised). Previously
# only the unused patch_only=True branch did this; every training run took the
# raw-patch branch, so the model perceived the patch as a 27%-wide grey band
# while the saved PNG was deployed at full range -- train and eval saw different
# images. Job 683 T1 showed that band alone drops the authors' working patch from
# 67.69% to 8.85% AdvBench. This job tests whether removing the band closes the
# training gap.
#
# WHY --no_projection. The projection (optimise_proj.py's default) was our WRONG
# fix -- it squeezed the patch a SECOND time at the injection point (see D1,
# revised, and D10). With the units bug now fixed at the adapter, the correct
# pipeline is the released code's own path: the optimised variable IS the image,
# clamped to [0,1], saved directly. --no_projection reproduces exactly that, so
# the ONLY change from the authors' released training is the normalisation fix.
#   - saved PNG = raw latent x, full [0,1] range possible (project() is identity)
#   - clamp_latent = True (implied by --no_projection): x stays a valid image
#   - init_space = unit: torch.rand(3,224,224), a random IMAGE again (correct now)
#
# --seed 0 makes the run reproducible (D7) so it can be replicated for error bars.
#
# PREDICTION.
#   AdvBench jumps from ~22% (job 625, best prior retrain) toward the 60s
#       -> D11 confirmed as THE cause; claim C1 reproduced after a one-line fix.
#   AdvBench barely moves
#       -> band was necessary but not sufficient; the generalisation failure
#          (AdvBench DEGRADES over training, see D11 "what it does not explain")
#          becomes the lead -> step budget / EOT schedule.
#
# Compare against (v3 judge, clean 315 / 520):
#   job 622 (no projection, buggy)   29.84% SafeBench / 1.73% AdvBench
#   job 625 (projection + clamp)     48.25% / 22.31%
#   authors' ultrabreak.png          78.73% / 67.69%
#   paper                            81.59% / 72.69%

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
EXP_NAME="full50_normfix_noproj_seed0"
TRAIN_CONFIG="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
NUM_EPOCHS=1301
CKPT_STEP=1300
SEED=0
IMAGE="outputs/${EXP_NAME}/${CKPT_STEP}.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[normfix]"

cd "$REPO_ROOT"
mkdir -p logs

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv missing - refusing to run under an unpinned env" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} exp=${EXP_NAME} corpus=${TRAIN_CONFIG} epochs=${NUM_EPOCHS} seed=${SEED}"
echo "${LOG_PREFIX} FIX: patch CLIP-normalised at injection (D11); projection OFF; save raw latent"
python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[normfix] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
n = len(pd.read_csv("train_configs/safebench-tiny_jailbroken_mode_full50.csv"))
assert n == 50, f"expected 50 training rows, got {n}"
print(f"[normfix] training corpus rows: {n}")

# Guard: confirm the D11 fix is actually present in the adapter this run imports,
# so a stale checkout cannot silently reproduce the bug.
import inspect, sys
sys.path.insert(0, "optimisation")
from qwen2_adapter import Qwen2Adapter
src = inspect.getsource(Qwen2Adapter.compute_loss)
assert "normalised_patch = (patch - mean_tensor) / std_tensor" in src, "D11 fix missing!"
assert src.count("apply_patch(pixel_values") == 1, "expected a single (normalised) injection path"
print("[normfix] D11 fix verified present in Qwen2Adapter.compute_loss")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage A: training (normalised injection, no projection) ====="
python optimisation/optimise_proj.py \
  --train_config "$TRAIN_CONFIG" \
  --exp_name "$EXP_NAME" \
  --num_epochs "$NUM_EPOCHS" \
  --no_projection \
  --seed "$SEED"

if [[ ! -f "$IMAGE" ]]; then
  echo "${LOG_PREFIX} Expected checkpoint not found: ${IMAGE}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Trained image ready: ${IMAGE}"

echo ""
echo "${LOG_PREFIX} ===== Pixel statistics (the fix should let contrast RECOVER) ====="
python - "$IMAGE" <<'PY'
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float64) / 255.0
print("  saved patch: std=%.1f  range=[%.0f,%.0f]" % (a.std()*255, a.min()*255, a.max()*255))
print("  reference -- authors' ultrabreak.png  std=41.7  range=[0,255]")
print("  reference -- job 625 (band-confined)  std= 9.8  range=[105,189]")
print("  a value far above ~10 means the optimiser is no longer trapped in the band")
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
echo "${LOG_PREFIX} ===== RESULT (step-${CKPT_STEP}, D11 fix, no projection) ====="
echo "${LOG_PREFIX} AdvBench climbing toward the 60s => D11 confirmed, C1 reproduced."
echo "${LOG_PREFIX} AdvBench near ~22 => band was necessary but not sufficient; look to step budget / EOT."
python evaluation/summarise_asr.py --paper --pattern "${EXP_NAME}" || true

echo ""
echo "${LOG_PREFIX} Done"
