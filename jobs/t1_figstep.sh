#!/usr/bin/env bash
#SBATCH --job-name=ub_t1fig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@600
#
# Table 1 baseline -- FigStep. Authors' RELEASED SafeBench typography images
# (baselines/figstep/data/images/SafeBench), FigStep prompt_6 template, our clean
# 315-set. SafeBench ONLY (FigStep has no universal trigger), all 6 models,
# judged v1 (authors', matches how the paper scored) + v3 (our corrected).
# Branch-independent: a baseline is the same regardless of how we reproduce UltraBreak.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
AUTH="c4c276d"; L="[t1fig]"; CFG="figstep_safebench"
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" \
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
STAGE="${SLURM_TMPDIR:-/tmp}/t1fig_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE" logs
echo "$L start $(date)"

ATTEMPT="${1:-1}"; MAX=5
done_cells(){ python - <<'PY'
import os,glob
m=["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","Qwen/Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf","moonshotai/Kimi-VL-A3B-Instruct","THUDM/GLM-4.1V-9B-Thinking"]
d=e=0
for x in m:
  for suf in ['_harmbench.csv','_harmbench_v3.csv']:
    e+=1
    if os.path.exists(f"results/figstep_safebench/{x}{suf}"): d+=1
print(d,e)
PY
}
read SD EXP < <(done_cells); echo "$L attempt $ATTEMPT/$MAX -- $SD/$EXP"
(( EXP>0 && SD>=EXP )) && { echo "$L COMPLETE"; exit 0; }
RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d e; read d e < <(done_cells)
  (( d>=e )) && { echo "$L COMPLETE"; return; }; (( ATTEMPT>=MAX )) && { echo "$L MAX"; return; }
  [[ "$1" == end ]] && (( d<=SD )) && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable jobs/t1_figstep.sh $((ATTEMPT+1)) 2>&1) && echo "$L RESUME $((ATTEMPT+1)) job $n" || echo "$L resubmit failed $n"; }
trap 'echo "$L USR1"; resub timeout; exit 0' USR1

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py"
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

[[ -s "attack_configs/${CFG}.csv" ]] || { echo "$L missing attack_configs/${CFG}.csv" >&2; exit 1; }
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
  [[ -f "results/${CFG}/${m}_harmbench_v3.csv" ]] || python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" || echo "$L v3 judge failed $m"
  [[ -f "results/${CFG}/${m}_harmbench.csv" ]] || ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$res" ) | tail -1
done

echo "$L =========== FigStep SafeBench (ASR%) vs paper ==========="
python - <<'PY'
import os,pandas as pd
order=["Qwen/Qwen2-VL-7B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct","Qwen/Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf","moonshotai/Kimi-VL-A3B-Instruct","THUDM/GLM-4.1V-9B-Thinking"]
short={"Qwen/Qwen2-VL-7B-Instruct":"Qwen2-VL","Qwen/Qwen2.5-VL-7B-Instruct":"Qwen2.5-VL","Qwen/Qwen-VL-Chat":"Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf":"LLaVA-1.6","moonshotai/Kimi-VL-A3B-Instruct":"Kimi-VL","THUDM/GLM-4.1V-9B-Thinking":"GLM-4.1V"}
paper={"Qwen2-VL":44.76,"Qwen2.5-VL":53.97,"Qwen-VL-Chat":69.52,"LLaVA-1.6":47.94,"Kimi-VL":73.02,"GLM-4.1V":88.25}
def asr(m,suf):
    p=f"results/figstep_safebench/{m}{suf}"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return d["attack_success"].astype(bool).mean()*100 if "attack_success" in d else None
print(f"{'model':13s} {'paper':>7s} {'ours_v1':>8s} {'ours_v3':>8s}")
for m in order:
    v1=asr(m,"_harmbench.csv"); v3=asr(m,"_harmbench_v3.csv"); s=short[m]
    print(f"{s:13s} {paper[s]:>7.1f} {(f'{v1:.1f}' if v1 is not None else '-'):>8s} {(f'{v3:.1f}' if v3 is not None else '-'):>8s}")
PY
resub end
echo "$L done $(date)"
