#!/usr/bin/env bash
#SBATCH --job-name=ub_diag_norm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Diagnostic: is the training gap explained by the missing patch normalisation?
# ============================================================================
#
# THE DEFECT (qwen2_adapter.py:243-251, verbatim in the authors' c4c276d):
# compute_loss normalises the patch ONLY on the patch_only=True branch. Every
# training run takes patch_only=False (optimise.py:78, optimise_proj.py:175),
# which injects the raw patch straight into `pixel_values` -- a tensor the
# Qwen2VLImageProcessor has already CLIP-normalised (preprocessor_config.json:
# image_mean=CLIP_MEAN, image_std=CLIP_STD, do_normalize=True; white.jpeg comes
# out at min 1.653 / max 2.146). llava16_adapter.py:81-85 has the same defect
# with the normalisation commented out and the variable still called
# `normalised_patch`.
#
# So during training the model perceives the patch as an image with pixels
# gamma*v + beta, i.e. confined to [0.449, 0.718] -- 27% of the range. At eval
# the same PNG goes through the real processor and spans [0,1]. Training and
# deployment are not looking at the same image.
#
# Corollary that needs no GPU: with the [0,1] clamp at optimise.py:141, the set
# of images the model can EVER perceive during training is exactly that band.
# ultrabreak.png spans [0,255]. The released training code therefore cannot
# produce the released artifact.
#
# WHAT THIS JOB TESTS. Three pure image transforms, no training. Each is scored
# on SafeBench-315 and AdvBench-520 exactly as the step-5 runs were.
#
#   T1  clip(gamma*ultrabreak + beta, 0, 1)   the authors' patch squeezed INTO
#       the band our optimiser is trapped in. THE KEY TEST -- if a patch that
#       scores 81.59/67.69 collapses toward our retrain levels, the band is
#       fatal and the defect above is the dominant cause.
#
#   T2  clip((x_625 - beta)/gamma, 0, 1)      job 625's washed-out artifact
#       stretched back out to full range. Tests whether our patches carry real
#       structure that is merely mis-scaled when saved.
#
#   T3  clip(gamma*x_622 + beta, 0, 1)        the deployable image job 622
#       implies under the defect. Did the no-projection run produce something
#       useful that we saved wrong?
#
# CAVEAT for the write-up: these probe the band's effect at INFERENCE. The
# defect is about what the model perceives during OPTIMISATION. Related but not
# identical -- only a retrain with normalisation restored proves the fix, and
# that is deliberately a separate sprint.
#
# Reference points (v3 judge, clean 315 / 520):
#   authors' ultrabreak.png   78.73% SafeBench / 67.69% AdvBench
#   job 625 (best retrain)    48.25% / 22.31%
#   job 622 (no projection)   29.84% /  1.73%

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
OUT_DIR="outputs/diag_normalisation"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[diag-norm]"

cd "$REPO_ROOT"
mkdir -p logs "$OUT_DIR"

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv missing - refusing to run under an unpinned env" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

python - <<'PY'
import importlib.metadata as m
pins = ["torch","torchvision","transformers","tokenizers","qwen-vl-utils","accelerate","numpy","pandas","Pillow"]
print("[diag-norm] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
PY

echo ""
echo "${LOG_PREFIX} ===== Stage A: build the three variants ====="
# Constants come from optimisation/utils.py so they cannot drift from training.
python - <<'PY'
import sys
sys.path.insert(0, "optimisation")
import numpy as np
from PIL import Image
from utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD  # noqa: E402

BETA = np.array(OPENAI_CLIP_MEAN)
GAMMA = np.array(OPENAI_CLIP_STD)

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0

def save(arr, path):
    Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).round().astype(np.uint8)).save(path)

def report(tag, arr, note):
    print("  %-22s std=%5.1f  range=[%3.0f,%3.0f]   %s"
          % (tag, arr.std() * 255, arr.min() * 255, arr.max() * 255, note))

SRC = {
    "ultrabreak": "outputs/ultrabreak.png",
    "x_625":      "outputs/safebench_tiny_full50_proj_repro/1300.png",
    "x_622":      "outputs/safebench_tiny_full50_repro/1300.png",
}
for name, path in SRC.items():
    report("source " + name, load(path), path)

print()
# T1: squeeze the authors' patch into the band training is confined to.
t1 = np.clip(GAMMA * load(SRC["ultrabreak"]) + BETA, 0.0, 1.0)
save(t1, "outputs/diag_normalisation/t1_ultrabreak_band.png")
report("T1 ultrabreak_band", t1, "expect std~13, range~[104,191]")

# T2: stretch job 625's artifact back out of the band.
t2 = np.clip((load(SRC["x_625"]) - BETA) / GAMMA, 0.0, 1.0)
save(t2, "outputs/diag_normalisation/t2_625_stretched.png")
report("T2 625_stretched", t2, "expect full range")

# T3: the deployable image job 622 implies under the defect.
t3 = np.clip(GAMMA * load(SRC["x_622"]) + BETA, 0.0, 1.0)
save(t3, "outputs/diag_normalisation/t3_622_projected.png")
report("T3 622_projected", t3, "expect std~10, band-confined")
PY

echo ""
echo "${LOG_PREFIX} ===== Stage B: score each variant on SafeBench-315 + AdvBench-520 ====="

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

for VARIANT in t1_ultrabreak_band t2_625_stretched t3_622_projected; do
  IMAGE="${OUT_DIR}/${VARIANT}.png"
  if [[ ! -f "$IMAGE" ]]; then
    echo "${LOG_PREFIX} missing variant image: ${IMAGE}" >&2
    exit 1
  fi

  SB_CFG="diag_${VARIANT}_safebench_eval"
  AB_CFG="diag_${VARIANT}_advbench_eval"

  echo ""
  echo "${LOG_PREFIX} ---- ${VARIANT} ----"
  python create_attack_configs.py --dataset safebench --config-type attack \
    --exclude-train datasets/SafeBench-Tiny.csv --image "$IMAGE" \
    --output "attack_configs/${SB_CFG}.csv"

  python create_attack_configs.py --dataset advbench --config-type attack --normalize \
    --image "$IMAGE" --output "attack_configs/${AB_CFG}.csv"

  score_qwen "$SB_CFG"
  score_qwen "$AB_CFG" --behaviors_csv datasets/adv_bench.csv --behaviors_col goal
done

echo ""
echo "${LOG_PREFIX} ===== RESULT ====="
echo "${LOG_PREFIX} T1 collapsing toward ~48/~22 => the band is fatal, defect confirmed."
echo "${LOG_PREFIX} T1 holding near 78.73/67.69 => band survivable, look to H2/H3 instead."
python evaluation/summarise_asr.py --paper --pattern diag_ || true

echo ""
echo "${LOG_PREFIX} Done"
