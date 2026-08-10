#!/bin/bash
#SBATCH --job-name=ub_v1_rescore
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# Score result CSVs with the AUTHORS' ORIGINAL judge (commit c4c276d), unmodified.
#
# Why this exists
# ---------------
# This fork's scorer has been through three passes: v1 (the authors' released
# code), v2 (model-aware response extraction, job 598), v3 (+ canonical judge
# behaviours, + GLM reasoning handling, job 614).  v2/v3 fix real defects -- v1
# anchors on the first literal "assistant", which matches Qwen's system prompt
# "You are a helpful assistant." -- but they are OUR changes, and a
# reproducibility report has to state what the authors' own protocol yields.
#
# It yields their number exactly.  On the canonical 315-query SafeBench set the
# released ultrabreak.png scores 257/315 = 81.59% under v1, matching the paper's
# 81.59 to the last decimal; our corrected v3 scorer gives 78.73%.  This job
# extends that test to normalised AdvBench, where the paper reports 72.69% --
# which is exactly 378/520 (see evaluation/infer_eval_set_size.py).
#
# The scorer is materialised straight from git rather than kept as a second copy
# in the tree, so it cannot silently drift from what the authors released.
#
# Writes *_harmbench.csv beside each input -- the v1 suffix.  Inputs are chosen
# to have no existing v1 file; check before adding to the list.

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
AUTHORS_COMMIT="c4c276d"
STAGE_ROOT="${SLURM_TMPDIR:-/tmp}/ultrabreak_v1_${SLURM_JOB_ID:-manual}"
LOG_PREFIX="[v1-rescore]"

# Result CSVs to score. These are raw generations (no _harmbench* suffix).
INPUTS=(
  # The exact-match test: authors' image, normalised AdvBench prompts.
  "results/advbench_jailbroken_mode/Qwen/Qwen2-VL-7B-Instruct.csv"
  # Our retrains, so the main table can be read under both protocols.
  "results/safebench_tiny_full50_repro_safebench_eval/Qwen/Qwen2-VL-7B-Instruct.csv"
  "results/safebench_tiny_full50_repro_advbench_eval/Qwen/Qwen2-VL-7B-Instruct.csv"
  "results/safebench_tiny_full50_proj_repro_safebench_eval/Qwen/Qwen2-VL-7B-Instruct.csv"
  "results/safebench_tiny_full50_proj_repro_advbench_eval/Qwen/Qwen2-VL-7B-Instruct.csv"
)

cd "$REPO_ROOT"
mkdir -p logs "$STAGE_ROOT"
source repro/bin/activate
export PYTHONUNBUFFERED=1

echo "${LOG_PREFIX} Repository: ${REPO_ROOT}"
echo "${LOG_PREFIX} Authors' judge from commit: ${AUTHORS_COMMIT}"

# Materialise the authors' scorer next to the judge_llms package it imports.
git show "${AUTHORS_COMMIT}:evaluation/evaluate.py" > "${STAGE_ROOT}/evaluate_v1.py"
echo "${LOG_PREFIX} sha256 of authors' evaluate.py: $(sha256sum "${STAGE_ROOT}/evaluate_v1.py" | cut -d' ' -f1)"
cp "${STAGE_ROOT}/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO_ROOT}/evaluation/_evaluate_v1_tmp.py"' EXIT

# Refuse to clobber an existing v1 score rather than silently rewriting history.
for input_csv in "${INPUTS[@]}"; do
  existing="${input_csv%.csv}_harmbench.csv"
  if [[ -f "$existing" ]]; then
    echo "${LOG_PREFIX} ERROR: ${existing} already exists; remove it or drop the input." >&2
    exit 1
  fi
  if [[ ! -f "$input_csv" ]]; then
    echo "${LOG_PREFIX} ERROR: missing input ${input_csv}" >&2
    exit 1
  fi
done

for input_csv in "${INPUTS[@]}"; do
  echo "${LOG_PREFIX} ===================================================="
  echo "${LOG_PREFIX} Scoring ${input_csv}"
  ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../${input_csv}" ) \
    | tail -20
done

echo "${LOG_PREFIX} ===================================================="
echo "${LOG_PREFIX} Summary (v1 column is the authors' unmodified scorer):"
python evaluation/summarise_asr.py --paper --pattern full50
python evaluation/summarise_asr.py --paper --pattern advbench_jailbroken_mode/
echo "${LOG_PREFIX} Done."
