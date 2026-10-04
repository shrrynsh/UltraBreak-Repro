#!/usr/bin/env bash
#SBATCH --job-name=ub_t1nam
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@900
#
# Table 1 gap: No-Attack x MM-SafetyBench (the paper reports this column; we had
# only run No-Attack on SafeBench + AdvBench). Blank white.jpeg, no jailbreak
# phrase, MM-SafetyBench-520 seed 0. 6 models, judged v1 + v3.
# GLM uses the corrected 4096 budget, so these cells need no later re-run.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
AUTH="c4c276d"; L="[t1nam]"; CFG="noattack_mmsafety_matrix"
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" \
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
STAGE="${SLURM_TMPDIR:-/tmp}/t1nam_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE" logs
ATTEMPT="${1:-1}"; MAX=6
echo "$L start $(date) attempt $ATTEMPT/$MAX"
cells(){ local d=0; for m in "${MODELS[@]}"; do
  [[ -f "results/${CFG}/${m}_harmbench.csv" ]] && d=$((d+1))
  [[ -f "results/${CFG}/${m}_harmbench_v3.csv" ]] && d=$((d+1)); done; echo $d; }

# ── chain to the next stage ────────────────────────────────────────────────────
# The cluster allows one running job, so the night's work is a chain: each stage
# self-resumes until its own work is complete, then seeds the next stage once.
# Guarded against double-seeding by checking the queue for the successor's name.
chain_to(){   # script jobname
  local script="$1" name="$2"
  [[ -f "$script" ]] || { echo "$L chain: $script not found" >&2; return; }
  if squeue -u "${USER:-$(id -un)}" -h -o "%j" 2>/dev/null | grep -qx "$name"; then
    echo "$L chain: $name already queued/running - not seeding again"; return
  fi
  local n; n=$(sbatch --parsable "$script" 2>&1) \
    && echo "$L CHAIN -> $script as job $n" \
    || echo "$L chain submit FAILED: $n" >&2
}

SD=$(cells); echo "$L progress $SD/12"
if (( SD >= 12 )); then
  echo "$L COMPLETE"
  chain_to jobs/glm_rerun.sh ub_glmre
  exit 0
fi
RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(cells)
  (( d>=12 )) && { echo "$L COMPLETE"; return; }; (( ATTEMPT>=MAX )) && { echo "$L MAX"; return; }
  [[ "$1" == end ]] && (( d<=SD )) && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable jobs/t1_noattack_mmsafety.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L RESUME $((ATTEMPT+1)) job $n" || echo "$L resubmit failed $n"; }
trap 'echo "$L USR1"; resub timeout; exit 0' USR1

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py"
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

[[ -s "attack_configs/${CFG}.csv" ]] || { echo "$L missing config" >&2; exit 1; }
for m in "${MODELS[@]}"; do
  res="results/${CFG}/${m}.csv"
  if [[ -f "$res" ]] && python - "$res" "attack_configs/${CFG}.csv" <<'PY'
import sys,pandas as pd
r,c=pd.read_csv(sys.argv[1]),pd.read_csv(sys.argv[2])
ne=r["response"].astype(str).str.strip().replace("nan","").astype(bool).sum() if "response" in r else 0
sys.exit(0 if (len(r)>=len(c) and ne>=len(c)) else 1)
PY
  then echo "$L reuse gen $m"; else
    gpy="$(py_for_gen "$m")"; echo "$L gen ${CFG} on $m (${gpy%%/bin/*})"
    "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$CFG" || { echo "$L gen failed $m"; continue; }
  fi
  [[ -f "$res" ]] || continue
  [[ -f "results/${CFG}/${m}_harmbench_v3.csv" ]] || python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" || echo "$L v3 failed $m"
  [[ -f "results/${CFG}/${m}_harmbench.csv" ]] || ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$res" ) | tail -1
done

echo "$L ====== No-Attack x MM-SafetyBench vs paper ======"
python - <<'PY'
import os,pandas as pd
order=["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen-VL-Chat","Qwen/Qwen2.5-VL-7B-Instruct","llava-hf/llava-v1.6-mistral-7b-hf","moonshotai/Kimi-VL-A3B-Instruct","THUDM/GLM-4.1V-9B-Thinking"]
short={"Qwen/Qwen2-VL-7B-Instruct":"Qwen2-VL","Qwen/Qwen-VL-Chat":"Qwen-VL-Chat","Qwen/Qwen2.5-VL-7B-Instruct":"Qwen2.5-VL","llava-hf/llava-v1.6-mistral-7b-hf":"LLaVA-1.6","moonshotai/Kimi-VL-A3B-Instruct":"Kimi-VL","THUDM/GLM-4.1V-9B-Thinking":"GLM-4.1V"}
paper={"Qwen2-VL":26.19,"Qwen-VL-Chat":21.49,"Qwen2.5-VL":33.45,"LLaVA-1.6":35.06,"Kimi-VL":41.79,"GLM-4.1V":43.69}
def a(m,suf):
    p=f"results/noattack_mmsafety_matrix/{m}{suf}"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return d["attack_success"].astype(bool).mean()*100 if "attack_success" in d else None
print(f"{'model':13s} {'paper':>7s} {'ours_v1':>8s} {'ours_v3':>8s}")
for m in order:
    s=short[m]; v1=a(m,"_harmbench.csv"); v3=a(m,"_harmbench_v3.csv")
    print(f"{s:13s} {paper[s]:>7.1f} {(f'{v1:.1f}' if v1 is not None else '-'):>8s} {(f'{v3:.1f}' if v3 is not None else '-'):>8s}")
PY
if (( $(cells) >= 12 )); then
  echo "$L ALL 12 CELLS COMPLETE - handing the GPU to the GLM re-run"
  chain_to jobs/glm_rerun.sh ub_glmre
else
  resub end
fi
echo "$L done $(date)"
