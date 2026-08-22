#!/usr/bin/env bash
#SBATCH --job-name=ub_pipe2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# PHASE-2 self-chaining pipeline — the RUNNABLE backlog items from REPRO_CHECKLIST
# that have working infra (flags/loaders/adapters already exist). Same contract as
# jobs/pipeline.sh: each step validates its inputs, submits the next step, then does
# its heavy work — so a bug halts the chain instead of cascading. Cheap steps first.
#
#   step 1  B15  — affirming-phrase sensitivity: 3 phrases × GLM × SafeBench
#   step 2  A4-MM— our 809 patch × MM-SafetyBench-520 × 5 black-box targets
#   step 3  C2b  — surrogate choice: train a LLaVA-surrogate patch, eval white-box
#
# NOT included (blocked — need prerequisites, cannot be queued as-is):
#   A5 baselines (no code) · D-sr/A7-native SR scorer (no code / API key) ·
#   C2a (2B/32B/72B Qwen weights not cached) · C4/C6 figures (no plotting code) ·
#   A10 timing (per-epoch time is only in training stdout, not losses.csv — compile
#   separately from job logs; no GPU needed).
#
#   Seed:  sbatch jobs/pipeline2.sh 1      Resume at N:  sbatch jobs/pipeline2.sh N
set -uo pipefail
STEP="${1:?usage: sbatch jobs/pipeline2.sh <step-number>}"
TOTAL=3
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
LOG="[pipe2 ${STEP}/${TOTAL}]"
SURR="Qwen/Qwen2-VL-7B-Instruct"
PATCH809="outputs/full50_normfix_lr0p01_s3000/3000.png"
AUTHIMG="outputs/ultrabreak.png"
TARGETS_BB=( "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat"
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
echo "$LOG start $(date)"

submit_next(){ local n=$((STEP+1))
  if (( n>TOTAL )); then
    local j3; j3=$(sbatch --parsable jobs/pipeline3.sh 1 2>&1) \
      && echo "$LOG pipeline2 complete — seeded pipeline3 as job $j3" \
      || echo "$LOG pipeline2 complete — could not seed pipeline3: $j3 (run: sbatch jobs/pipeline3.sh 1)"
    return; fi
  local j; j=$(sbatch --parsable jobs/pipeline2.sh "$n" 2>&1) \
    && echo "$LOG queued next step $n as job $j" || echo "$LOG WARN could not queue step $n: $j"; }

GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

# gen+judge a config across models (resume-guarded, GLM-routed).
score(){ local cfg="$1" ds="$2"; shift 2
  [[ -s "attack_configs/${cfg}.csv" ]] || { echo "$LOG score: missing/empty attack_configs/${cfg}.csv" >&2; return 1; }
  local extra=""; [[ "$ds" == advbench ]] && extra="--behaviors_csv datasets/adv_bench.csv --behaviors_col goal"
  for m in "$@"; do
    local res="results/${cfg}/${m}.csv"
    if [[ -f "$res" ]] && python - "$res" "attack_configs/${cfg}.csv" <<'PY'
import sys,pandas as pd
r,c=pd.read_csv(sys.argv[1]),pd.read_csv(sys.argv[2])
ne=r["response"].astype(str).str.strip().replace("nan","").astype(bool).sum() if "response" in r else 0
sys.exit(0 if (len(r)>=len(c) and ne>=len(c)) else 1)
PY
    then echo "$LOG reuse gen ${cfg}/${m}"; else
      local gpy; gpy="$(py_for_gen "$m")"; echo "$LOG gen ${cfg} on ${m} (${gpy%%/bin/*})"
      "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" || { echo "$LOG gen failed ${m}"; continue; }
    fi
    [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { echo "$LOG reuse judge ${cfg}/${m}"; continue; }
    python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" $extra || echo "$LOG judge failed ${m}"
  done; }

echo "$LOG ============================================================"
case "$STEP" in
  1)  # B15 — affirming-phrase sensitivity: 3 phrases, authors' image, GLM + Qwen2-VL, SafeBench
      echo "$LOG B15: phrase sweep on SafeBench (authors' image)"
      [[ -f "$AUTHIMG" ]] || { echo "$LOG missing $AUTHIMG" >&2; exit 1; }
      submit_next
      # Paper Table 5's exact 3 phrases (GLM SafeBench: 66.03 / 72.06 / 66.98).
      declare -A PH=( [jbm]="[Jailbroken Mode]" [access]="[ACCESS AUTHORISED]" [classified]="[CLASSIFIED INFO]" )
      for tag in jbm access classified; do
        cfg="b15_${tag}_safebench"
        python create_attack_configs.py --dataset safebench --config-type attack \
          --exclude-train datasets/SafeBench-Tiny.csv --phrase "${PH[$tag]}" \
          --image "$AUTHIMG" --output "attack_configs/${cfg}.csv" \
          || { echo "$LOG build failed ${cfg}"; continue; }
        score "$cfg" safebench "THUDM/GLM-4.1V-9B-Thinking" "$SURR"
      done
      ;;
  2)  # A4-MM — our 809 patch transferred to MM-SafetyBench-520 × 5 black-box targets
      echo "$LOG A4-MM: 809 patch × MM-SafetyBench-520 × 5 transfer targets"
      [[ -f "$PATCH809" ]] || { echo "$LOG missing $PATCH809" >&2; exit 1; }
      submit_next
      cfg="a4mm_809_mmsafety"
      python create_attack_configs.py --dataset mm-safetybench --config-type attack \
        --subsample 520 --image "$PATCH809" --output "attack_configs/${cfg}.csv" \
        || { echo "$LOG build failed ${cfg}" >&2; exit 1; }
      score "$cfg" mm-safetybench "$SURR" "${TARGETS_BB[@]}"
      ;;
  3)  # C2b — surrogate choice: train a LLaVA-surrogate patch (D11 fix, no proj, 3000), eval white-box
      echo "$LOG C2b: train LLaVA-surrogate patch + white-box eval"
      submit_next
      name="c2b_llava_surrogate_s3000"
      if [[ -f "outputs/${name}/3000.png" ]]; then echo "$LOG reuse existing ${name}/3000.png"
      else
        python optimisation/optimise_proj.py \
          --train_config "train_configs/safebench-tiny_jailbroken_mode_full50.csv" \
          --exp_name "$name" --surrogates llava --no_projection --seed 0 \
          --num_epochs 3001 --lr 0.01 || echo "$LOG C2b training returned nonzero"
      fi
      img="outputs/${name}/3000.png"
      [[ -f "$img" ]] || { echo "$LOG C2b produced no ${img}" >&2; exit 1; }
      # white-box eval = the LLaVA surrogate itself, on SafeBench
      python create_attack_configs.py --dataset safebench --config-type attack \
        --exclude-train datasets/SafeBench-Tiny.csv --image "$img" \
        --output "attack_configs/${name}_safebench.csv" || exit 1
      score "${name}_safebench" safebench "llava-hf/llava-v1.6-mistral-7b-hf"
      ;;
  *) echo "$LOG unknown step $STEP" >&2; exit 1 ;;
esac
echo "$LOG done $(date)"
