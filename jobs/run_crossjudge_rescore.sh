#!/usr/bin/env bash
#SBATCH --job-name=ub_xjudge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# Cross-judge rescore so the two reproduction tracks can be read under the SAME
# scorer:
#   A) paper-faithful @1300 generations  -> v1 (authors' unmodified judge)
#   B) code-faithful released-image gens -> v3 (our fullest judge)
# Both directions run on GPU (HarmBench-Llama-2-13b classifier). Idempotent:
# existing judge files are skipped, never rewritten.
set -uo pipefail
REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
AUTHORS_COMMIT="c4c276d"
STAGE="${SLURM_TMPDIR:-/tmp}/ultrabreak_xjudge_${SLURM_JOB_ID:-manual}"
LOG="[xjudge]"
cd "$REPO_ROOT"; mkdir -p logs "$STAGE"
source repro/bin/activate
export PYTHONUNBUFFERED=1
echo "$LOG start $(date)"

# ---- A) v1 (authors' judge) on paper-faithful @1300 -------------------------
git show "${AUTHORS_COMMIT}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py"
echo "$LOG authors' evaluate.py sha256: $(sha256sum "$STAGE/evaluate_v1.py" | cut -d' ' -f1)"
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO_ROOT}/evaluation/_evaluate_v1_tmp.py"' EXIT

V1_INPUTS=(
  "results/repro_paper_faithful_1300_safebench_eval/Qwen/Qwen2-VL-7B-Instruct.csv"
  "results/repro_paper_faithful_1300_advbench_eval/Qwen/Qwen2-VL-7B-Instruct.csv"
)
for in_csv in "${V1_INPUTS[@]}"; do
  out="${in_csv%.csv}_harmbench.csv"
  [[ -f "$in_csv" ]] || { echo "$LOG ERROR missing $in_csv" >&2; continue; }
  [[ -f "$out" ]] && { echo "$LOG SKIP v1 (exists): $out"; continue; }
  echo "$LOG === v1 scoring $in_csv"
  ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$in_csv" ) | tail -15
done

# ---- B) v3 (our judge) on code-faithful released image ----------------------
# Same generation files that already carry v1 (_harmbench.csv) and v2, so all
# three judges sit on identical generations for an apples-to-apples delta.
V3_SB="results/safebench_jailbroken_mode_ultrabreak/Qwen/Qwen2-VL-7B-Instruct.csv"
V3_AB="results/advbench_jailbroken_mode_ultrabreak/Qwen/Qwen2-VL-7B-Instruct.csv"

rescore_v3() {  # input_csv [extra judge args...]
  local in_csv="$1"; shift
  local out="${in_csv%.csv}_harmbench_v3.csv"
  [[ -f "$in_csv" ]] || { echo "$LOG ERROR missing $in_csv" >&2; return; }
  [[ -f "$out" ]] && { echo "$LOG SKIP v3 (exists): $out"; return; }
  echo "$LOG === v3 scoring $in_csv (args: $*)"
  python evaluation/evaluate.py --attack_result "$in_csv" --output_suffix "_harmbench_v3.csv" "$@" | tail -15
}
rescore_v3 "$V3_SB"
# AdvBench: try canonical behaviors; on any failure (row-count/order) fall back.
if ! rescore_v3 "$V3_AB" --behaviors_csv datasets/adv_bench.csv --behaviors_col goal; then
  echo "$LOG v3 advbench with behaviors failed — retrying without behaviors"
  rescore_v3 "$V3_AB"
fi

# ---- Summary ----------------------------------------------------------------
echo "$LOG ===================================================="
python - <<'PY'
import pandas as pd, os
def asr(p):
    if not os.path.exists(p): return None
    d=pd.read_csv(p)
    c="attack_success" if "attack_success" in d.columns else None
    return (d[c].astype(bool).mean()*100, len(d)) if c else None
rows=[
 ("PAPER-FAITHFUL @1300 SafeBench","results/repro_paper_faithful_1300_safebench_eval/Qwen/Qwen2-VL-7B-Instruct"),
 ("PAPER-FAITHFUL @1300 AdvBench","results/repro_paper_faithful_1300_advbench_eval/Qwen/Qwen2-VL-7B-Instruct"),
 ("CODE-FAITHFUL (released) SafeBench","results/safebench_jailbroken_mode_ultrabreak/Qwen/Qwen2-VL-7B-Instruct"),
 ("CODE-FAITHFUL (released) AdvBench","results/advbench_jailbroken_mode_ultrabreak/Qwen/Qwen2-VL-7B-Instruct"),
]
print("%-38s %14s %14s %14s"%("cell","v1(_harmbench)","v2","v3"))
for name,base in rows:
    vals=[]
    for suf in ["_harmbench.csv","_harmbench_v2.csv","_harmbench_v3.csv"]:
        r=asr(base+suf); vals.append(f"{r[0]:.2f}% (n={r[1]})" if r else "-")
    print("%-38s %14s %14s %14s"%(name,*vals))
PY
echo "$LOG done $(date)"
