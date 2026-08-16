#!/usr/bin/env bash
#SBATCH --job-name=ultrabreak_step1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Step 1 of the reproduction diagnosis.
#
# Stage A breaks the AdvBench confound. The two existing AdvBench runs differ in
# two variables at once (adversarial image AND prompt normalization), so neither
# isolates anything. This generates the missing cell:
#     AdvBench x ultrabreak.png (authors' image) x normalized "Steps to ..." prompts
# Comparing it against advbench_jailbroken_mode_ultrabreak (9.62%, same image,
# un-normalized prompts) isolates the effect of normalization alone.
#
# Stage B closes the two gaps left by the job 598 re-judge:
#   - AdvBench configs built with --normalize wrote the reworded instruction into
#     'target', so HarmBench scored against text that appears in no benchmark.
#     --behaviors_csv restores the canonical goal.
#   - The GLM extractor required a closing </think>, which 31% of rows never emit
#     because generation hit the 512-token cap mid-reasoning. Those rows shipped
#     the whole reasoning trace to the judge.
#
# Outputs are written with a _harmbench_v3.csv suffix; v1 and v2 are left intact.

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
STAGE_ROOT="${SLURM_TMPDIR:-/tmp}/ultrabreak_step1_${SLURM_JOB_ID:-manual}"
OUTPUT_SUFFIX="_harmbench_v3.csv"
LOG_PREFIX="[step1]"

TARGET_MODEL="Qwen/Qwen2-VL-7B-Instruct"
STEP1_CONFIG="advbench_jailbroken_mode"

cd "$REPO_ROOT"
mkdir -p logs "$STAGE_ROOT"

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
else
  echo "${LOG_PREFIX} repro venv not found - refusing to fall back to an unpinned env" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} Repository : ${REPO_ROOT}"
echo "${LOG_PREFIX} Staging    : ${STAGE_ROOT}"
echo "${LOG_PREFIX} Python     : $(which python)"
python - <<'PY'
import importlib.metadata as m
pins = ["torch", "torchvision", "transformers", "tokenizers",
        "qwen-vl-utils", "accelerate", "numpy", "pandas", "Pillow"]
print("[step1] Versions : " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
PY

# Score one staged copy of a result file, then copy the scored output back.
# Extra args are forwarded to evaluate.py.
score_file() {
  local input_csv="$1"; shift
  local relative_path="${input_csv#results/}"
  local stage_csv="${STAGE_ROOT}/results/${relative_path}"
  local output_csv="${input_csv%.csv}${OUTPUT_SUFFIX}"

  if [[ ! -f "$input_csv" ]]; then
    echo "${LOG_PREFIX} MISSING input, skipping: ${input_csv}" >&2
    return 0
  fi

  mkdir -p "$(dirname "$stage_csv")" "$(dirname "$output_csv")"
  cp "$input_csv" "$stage_csv"

  echo "${LOG_PREFIX} Scoring ${input_csv} $*"
  python evaluation/evaluate.py \
    --attack_result "$stage_csv" \
    --output_suffix "$OUTPUT_SUFFIX" \
    "$@"

  local stage_output="${stage_csv%.csv}${OUTPUT_SUFFIX}"
  if [[ ! -f "$stage_output" ]]; then
    echo "${LOG_PREFIX} Expected output not found: ${stage_output}" >&2
    exit 1
  fi
  cp "$stage_output" "$output_csv"
  echo "${LOG_PREFIX} Wrote ${output_csv}"
}

# ---------------------------------------------------------------------------
# Stage 0: preflight
#
# The pinned repro venv (transformers 4.51.3) has not yet run a generation - the
# existing results were produced under transformers 5.10.0.dev0. Generate three
# rows first so a loading or API incompatibility fails in minutes rather than
# after hours of queueing and decoding.
# ---------------------------------------------------------------------------
echo ""
echo "${LOG_PREFIX} ===== Stage 0: preflight (3 rows) ====="

head -1 "attack_configs/${STEP1_CONFIG}.csv" > attack_configs/_smoke.csv
sed -n '2,4p' "attack_configs/${STEP1_CONFIG}.csv" >> attack_configs/_smoke.csv

python evaluation/attack.py --model_name "$TARGET_MODEL" --attack_config "_smoke"

SMOKE_RESULT="results/_smoke/${TARGET_MODEL}.csv"
python - "$SMOKE_RESULT" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
if len(df) != 3:
    raise SystemExit(f"preflight: expected 3 rows, got {len(df)}")
if df["response"].isna().any():
    raise SystemExit("preflight: null response - generation failed")
if not df["response"].str.contains("\nassistant\n").all():
    raise SystemExit("preflight: chat template changed - the Qwen extractor would mis-parse")
print("[step1] Preflight OK. Sample response:")
print(df["response"].iloc[0].split("\nassistant\n")[-1][:200])
PY

rm -rf results/_smoke attack_configs/_smoke.csv

# ---------------------------------------------------------------------------
# Stage A: generate the missing AdvBench cell
# ---------------------------------------------------------------------------
echo ""
echo "${LOG_PREFIX} ===== Stage A: generation ====="
echo "${LOG_PREFIX} Model  : ${TARGET_MODEL}"
echo "${LOG_PREFIX} Config : ${STEP1_CONFIG} (ultrabreak.png + normalized prompts)"

STEP1_RESULT="results/${STEP1_CONFIG}/${TARGET_MODEL}.csv"
python evaluation/attack.py \
  --model_name "$TARGET_MODEL" \
  --attack_config "$STEP1_CONFIG"

if [[ ! -f "$STEP1_RESULT" ]]; then
  echo "${LOG_PREFIX} Generation did not produce ${STEP1_RESULT}" >&2
  exit 1
fi
echo "${LOG_PREFIX} Generated ${STEP1_RESULT}"

# The regenerated config already carries canonical AdvBench goals in 'target',
# so this one needs no behavior override.
score_file "$STEP1_RESULT"

# ---------------------------------------------------------------------------
# Stage B: close the gaps from job 598
# ---------------------------------------------------------------------------
echo ""
echo "${LOG_PREFIX} ===== Stage B: re-score affected runs ====="

# AdvBench runs whose configs were built with --normalize: restore canonical goals.
for csv in \
  "results/advbench_jailbroken_mode_1300/Qwen/Qwen2-VL-7B-Instruct.csv" \
  "results/advbench_jailbroken_mode_1300/Qwen/Qwen2.5-VL-7B-Instruct.csv" \
  "results/advbench_jailbroken_mode_1300/THUDM/GLM-4.1V-9B-Thinking.csv" \
  "results/advbench_jailbroken_mode_1300/llava-hf/llava-v1.6-mistral-7b-hf.csv" \
  "results/advbench_jailbroken_mode_1300/moonshotai/Kimi-VL-A3B-Instruct.csv" ; do
  score_file "$csv" --behaviors_csv datasets/adv_bench.csv --behaviors_col goal
done

# SafeBench instructions already use the "Steps to ..." form, so normalization is
# a no-op there and 'target' is already canonical. These are re-scored only to
# pick up the corrected GLM reasoning-block handling.
for csv in \
  "results/safebench_jailbroken_mode/THUDM/GLM-4.1V-9B-Thinking.csv" \
  "results/safebench_jailbroken_mode_1300/THUDM/GLM-4.1V-9B-Thinking.csv" \
  "results/safebench_jailbroken_mode_5000/THUDM/GLM-4.1V-9B-Thinking.csv" ; do
  score_file "$csv"
done

echo ""
echo "${LOG_PREFIX} Done"
