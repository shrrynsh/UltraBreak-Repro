#!/usr/bin/env bash
#SBATCH --job-name=ub_t1mm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@600
#
# Table 1 -- MM-SafetyBench column, BOTH branches.
#   code-faithful patch = outputs/codefaithful_exact_1300/1300.png  (judge v1 = its branch)
#   paper-faithful patch = outputs/repro_paper_faithful_1300/1300.png (judge v3 = its branch)
# Both patches scored on all 6 models (GLM via repro_glm), MM-SafetyBench-520
# (seed 0 -> identical subset to job 994's authors'-image run). Each cell judged
# under BOTH v1 and v3 so the doc can show branch judge + cross-check.
# Self-resuming (resume-guarded) so a wall timeout just continues.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
AUTH="c4c276d"; L="[t1mm]"
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" \
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
STAGE="${SLURM_TMPDIR:-/tmp}/t1mm_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE" logs
echo "$L start $(date)"

# ---- auto-resume (submit successor at end / on pre-wall signal) -------------
ATTEMPT="${1:-1}"; MAX=6
progress(){ python - <<'PY'
import os
tags={'cf':'outputs/codefaithful_exact_1300/1300.png','pf':'outputs/repro_paper_faithful_1300/1300.png'}
models=["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","Qwen/Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf","moonshotai/Kimi-VL-A3B-Instruct","THUDM/GLM-4.1V-9B-Thinking"]
done=exp=0
for t in tags:
  for m in models:
    for suf in ['_harmbench.csv','_harmbench_v3.csv']:
      exp+=1
      if os.path.exists(f"results/t1mm_{t}_mmsafety/{m}{suf}"): done+=1
print(done,exp)
PY
}
read START_DONE EXP < <(progress); echo "$L attempt $ATTEMPT/$MAX -- $START_DONE/$EXP cells"
if (( EXP>0 && START_DONE>=EXP )); then echo "$L COMPLETE"; exit 0; fi
RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d e; read d e < <(progress)
  (( d>=e )) && { echo "$L COMPLETE ($d/$e)"; return; }
  (( ATTEMPT>=MAX )) && { echo "$L MAX_ATTEMPTS"; return; }
  [[ "$1" == end ]] && (( d<=START_DONE )) && { echo "$L no progress $d<=$START_DONE, stop"; return; }
  local n; n=$(sbatch --parsable jobs/t1_mmsafety.sh $((ATTEMPT+1)) 2>&1) && echo "$L AUTO-RESUME $((ATTEMPT+1)) job $n ($d/$e)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 wall"; resub timeout; exit 0' USR1

# ---- v1 judge (authors', from git) -----------------------------------------
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py"
echo "$L v1 sha256: $(sha256sum "$STAGE/evaluate_v1.py"|cut -d' ' -f1)"
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT

GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

declare -A PATCH=( [cf]="outputs/codefaithful_exact_1300/1300.png" [pf]="outputs/repro_paper_faithful_1300/1300.png" )
for tag in cf pf; do
  img="${PATCH[$tag]}"; [[ -f "$img" ]] || { echo "$L missing $img"; continue; }
  cfg="t1mm_${tag}_mmsafety"
  if [[ ! -s "attack_configs/${cfg}.csv" ]]; then
    python create_attack_configs.py --dataset mm-safetybench --config-type attack \
      --subsample 520 --subsample-seed 0 --image "$img" --output "attack_configs/${cfg}.csv" \
      || { echo "$L build failed $cfg"; continue; }
  fi
  for m in "${MODELS[@]}"; do
    res="results/${cfg}/${m}.csv"
    if [[ -f "$res" ]] && python - "$res" "attack_configs/${cfg}.csv" <<'PY'
import sys,pandas as pd
r,c=pd.read_csv(sys.argv[1]),pd.read_csv(sys.argv[2])
ne=r["response"].astype(str).str.strip().replace("nan","").astype(bool).sum() if "response" in r else 0
sys.exit(0 if (len(r)>=len(c) and ne>=len(c)) else 1)
PY
    then echo "$L reuse gen $cfg/$m"; else
      gpy="$(py_for_gen "$m")"; echo "$L gen $cfg on $m (${gpy%%/bin/*})"
      "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" || { echo "$L gen failed $m"; continue; }
    fi
    [[ -f "$res" ]] || continue
    # v3 judge
    [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" || echo "$L v3 judge failed $m"
    # v1 judge (authors')
    [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$res" ) | tail -1
  done
done

echo "$L =============== RESULT: Table 1 MM-SafetyBench (ASR%) ==============="
python - <<'PY'
import os,pandas as pd
tags={"cf":"code-faithful","pf":"paper-faithful"}
order=["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","Qwen/Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf","moonshotai/Kimi-VL-A3B-Instruct","THUDM/GLM-4.1V-9B-Thinking"]
short={"Qwen/Qwen2-VL-7B-Instruct":"Qwen2-VL","Qwen/Qwen2.5-VL-7B-Instruct":"Qwen2.5-VL","Qwen/Qwen-VL-Chat":"Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf":"LLaVA-1.6","moonshotai/Kimi-VL-A3B-Instruct":"Kimi-VL","THUDM/GLM-4.1V-9B-Thinking":"GLM-4.1V"}
def asr(cfg,m,suf):
    p=f"results/{cfg}/{m}{suf}"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return d["attack_success"].astype(bool).mean()*100 if "attack_success" in d else None
for t,name in tags.items():
    print(f"\n### {name} (MM-SafetyBench-520) ###")
    print(f"{'model':13s} {'v1':>7s} {'v3':>7s}")
    for m in order:
        v1=asr(f"t1mm_{t}_mmsafety",m,"_harmbench.csv"); v3=asr(f"t1mm_{t}_mmsafety",m,"_harmbench_v3.csv")
        print(f"{short[m]:13s} {(f'{v1:.1f}' if v1 is not None else '-'):>7s} {(f'{v3:.1f}' if v3 is not None else '-'):>7s}")
PY
resub end
echo "$L done $(date)"
