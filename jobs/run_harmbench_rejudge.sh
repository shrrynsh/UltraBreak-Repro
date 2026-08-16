#!/usr/bin/env bash
#SBATCH --job-name=ultrabreak_rejudge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
STAGE_ROOT="${SLURM_TMPDIR:-/tmp}/ultrabreak_harmbench_stage_${SLURM_JOB_ID:-manual}"
OUTPUT_SUFFIX="_harmbench_v2.csv"
LOG_PREFIX="[rejudge]"

cd "$REPO_ROOT"
mkdir -p logs "$STAGE_ROOT"

if [[ -f repro/bin/activate ]]; then
  source repro/bin/activate
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ultrabreak
fi

export PYTHONUNBUFFERED=1

mapfile -t INPUT_CSVS < <(
  find results -type f -name '*.csv' \
    ! -name '*_harmbench.csv' \
    ! -name '*_harmbench_v2.csv' \
    ! -name 'attack_success_rate_summary*.csv' \
    | sort
)

if [[ ${#INPUT_CSVS[@]} -eq 0 ]]; then
  echo "${LOG_PREFIX} No input CSVs found under results/"
  exit 0
fi

echo "${LOG_PREFIX} Repository: ${REPO_ROOT}"
echo "${LOG_PREFIX} Staging: ${STAGE_ROOT}"
echo "${LOG_PREFIX} Inputs: ${#INPUT_CSVS[@]}"

for input_csv in "${INPUT_CSVS[@]}"; do
  relative_path="${input_csv#results/}"
  stage_csv="${STAGE_ROOT}/results/${relative_path}"
  stage_dir="$(dirname "$stage_csv")"
  output_csv="${input_csv%.csv}${OUTPUT_SUFFIX}"

  mkdir -p "$stage_dir" "$(dirname "$output_csv")"
  cp "$input_csv" "$stage_csv"

  echo "${LOG_PREFIX} Scoring ${input_csv}"
  python evaluation/evaluate.py --attack_result "$stage_csv" --output_suffix "$OUTPUT_SUFFIX"

  stage_output="${stage_csv%.csv}${OUTPUT_SUFFIX}"
  if [[ ! -f "$stage_output" ]]; then
    echo "${LOG_PREFIX} Expected output not found: ${stage_output}" >&2
    exit 1
  fi

  cp "$stage_output" "$output_csv"
  echo "${LOG_PREFIX} Wrote ${output_csv}"
done

echo "${LOG_PREFIX} Done"
