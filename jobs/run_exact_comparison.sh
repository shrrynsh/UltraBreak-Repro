#!/usr/bin/env bash
#SBATCH --job-name=ub_exactcmp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# EXACT reproduction comparison, controlled across the eval axes, white-box AND
# black-box (transfer), reporting BOTH ASR and NRR.
#
#   Patches (rows):
#     cf  = CODE-FAITHFUL retrain  : authors' verbatim optimise.py (only 3 I/O
#           constants changed) @1300 -> outputs/codefaithful_exact_1300/1300.png
#           Carries D11 band bug + dead §3.2 projection, exactly as released.
#     pf  = PAPER-FAITHFUL retrain : outputs/repro_paper_faithful_1300/1300.png
#     img = AUTHORS' RELEASED IMAGE (reference only, NOT code-faithful)
#
#   Axes:
#     SafeBench size : 315 (clean, --exclude-train) vs 350 (contaminated)
#     AdvBench form  : raw goal (authors) vs normalized ("Steps to ...", our dev)
#     Judge          : v1 (authors', c4c276d) vs v3 (our corrected)
#     Setting        : white-box (Qwen2-VL surrogate) + black-box (5 targets)
#     Metrics        : ASR (attack_success) + NRR (non-refusal)
#
#   Coverage: FULL CROSS — every eval variant (sb315, sb350, abraw, abnorm) on
#   every model (Qwen2-VL white-box + 5 transfer targets), both judges, both
#   metrics. Idempotent: existing gens/judgements reused.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
AUTH="c4c276d"
WB="Qwen/Qwen2-VL-7B-Instruct"
MODELS_ALL=( "$WB" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" \
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking" )
STAGE="${SLURM_TMPDIR:-/tmp}/exactcmp_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE" logs
L="[exactcmp]"; echo "$L start $(date)"

# ---------------- 1) TRAIN exact code-faithful @1300 -------------------------
CF_PNG="outputs/codefaithful_exact_1300/1300.png"
if [[ -f "$CF_PNG" ]]; then echo "$L code-faithful patch exists, skip training"
else
  echo "$L TRAIN code-faithful (authors' verbatim optimise_exact.py) ..."
  python reproducibility/authors_code/optimisation/optimise_exact.py || echo "$L TRAIN FAILED" >&2
fi
[[ -f "$CF_PNG" ]] || echo "$L WARN no code-faithful patch; cf rows will be blank" >&2

# ---------------- v1 judge (authors', materialized from git) -----------------
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py"
echo "$L v1 judge sha256: $(sha256sum "$STAGE/evaluate_v1.py" | cut -d' ' -f1)"
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT

# GLM-4.1V generates under repro_glm (transformers 5.x); all else + judges on repro.
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

# ---------------- helpers ----------------------------------------------------
build() {  # variant out_cfg patch
  local v="$1" out="attack_configs/$2.csv" img="$3"
  case "$v" in
    sb315) python create_attack_configs.py --dataset safebench --config-type attack \
             --exclude-train datasets/SafeBench-Tiny.csv --image "$img" --output "$out" ;;
    sb350) python create_attack_configs.py --dataset safebench --config-type attack \
             --image "$img" --output "$out" ;;
    abraw) python create_attack_configs.py --dataset advbench --config-type attack \
             --image "$img" --output "$out" ;;
    abnorm) python create_attack_configs.py --dataset advbench --config-type attack \
             --normalize --image "$img" --output "$out" ;;
  esac
  [[ -s "$out" ]] || { echo "$L build failed $out" >&2; return 1; }
}
gen() {  # cfg model   (resume-guarded, GLM-routed)
  local cfg="$1" m="$2" res="results/$1/$2.csv"
  if [[ -f "$res" ]] && python - "$res" "attack_configs/$1.csv" <<'PY'
import sys,pandas as pd
r,c=pd.read_csv(sys.argv[1]),pd.read_csv(sys.argv[2])
ne=r["response"].astype(str).str.strip().replace("nan","").astype(bool).sum() if "response" in r else 0
sys.exit(0 if (len(r)>=len(c) and ne>=len(c)) else 1)
PY
  then echo "$L reuse gen $cfg/$m"; else
    local gpy; gpy="$(py_for_gen "$m")"
    echo "$L gen $cfg on $m (${gpy%%/bin/*})"
    "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" || echo "$L gen failed $cfg/$m"
  fi
}
judge_v1() {  # cfg model
  local res="results/$1/$2.csv" out="results/$1/$2_harmbench.csv"
  [[ -f "$res" ]] || { echo "$L v1 no gen $1/$2"; return; }
  [[ -f "$out" ]] && { echo "$L reuse v1 $1/$2"; return; }
  ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$res" ) | tail -2
}
judge_v3() {  # cfg model [behaviors...]
  local res="results/$1/$2.csv" out="results/$1/$2_harmbench_v3.csv"; local cfg="$1" m="$2"; shift 2
  [[ -f "$res" ]] || { echo "$L v3 no gen $cfg/$m"; return; }
  [[ -f "$out" ]] && { echo "$L reuse v3 $cfg/$m"; return; }
  python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" "$@" \
    || { echo "$L v3 behaviors failed, retry plain"; python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv"; }
}

# ---------------- 2) factorial: patch x variant x model ----------------------
declare -A PATCH=( [cf]="$CF_PNG" [pf]="outputs/repro_paper_faithful_1300/1300.png" [img]="outputs/ultrabreak.png" )
for tag in cf pf img; do
  img="${PATCH[$tag]}"; [[ -f "$img" ]] || { echo "$L skip $tag (missing $img)"; continue; }
  for v in sb315 sb350 abraw abnorm; do
    cfg="cmp_${tag}_${v}"; build "$v" "$cfg" "$img" || continue
    # Full cross: every eval variant on every model (white-box + 5 transfer targets).
    for m in "${MODELS_ALL[@]}"; do
      gen "$cfg" "$m"
      judge_v1 "$cfg" "$m"
      if [[ "$v" == abraw ]]; then judge_v3 "$cfg" "$m" --behaviors_csv datasets/adv_bench.csv --behaviors_col goal
      else judge_v3 "$cfg" "$m"; fi
    done
  done
done

# ---------------- 3) summary: ASR + NRR, white-box + transfer ----------------
echo "$L ===================== RESULT (ASR% / NRR%) ====================="
python - <<'PY'
import os,glob,pandas as pd
tags={"cf":"code-faithful retrain","pf":"paper-faithful retrain","img":"released image (ref)"}
mshort={"Qwen/Qwen2-VL-7B-Instruct":"Qwen2-VL(WB)","Qwen/Qwen2.5-VL-7B-Instruct":"Qwen2.5-VL",
 "Qwen/Qwen-VL-Chat":"Qwen-VL-Chat","llava-hf/llava-v1.6-mistral-7b-hf":"LLaVA-1.6",
 "moonshotai/Kimi-VL-A3B-Instruct":"Kimi-VL","THUDM/GLM-4.1V-9B-Thinking":"GLM-4.1V"}
def m(cfg,model,suf):
    p=f"results/{cfg}/{model}{suf}"
    if not os.path.exists(p): return None
    d=pd.read_csv(p)
    if "attack_success" not in d.columns: return None
    asr=d["attack_success"].astype(bool).mean()*100
    nrr=(100-d["refused"].astype(bool).mean()*100) if "refused" in d.columns else float("nan")
    return asr,nrr,len(d)
rows=[]
for t in tags:
    for v in ["sb315","sb350","abraw","abnorm"]:
        for model in mshort:
            for jn,suf in [("v1","_harmbench.csv"),("v3","_harmbench_v3.csv")]:
                r=m(f"cmp_{t}_{v}",model,suf)
                if r: rows.append(dict(patch=tags[t],variant=v,model=mshort[model],judge=jn,ASR=round(r[0],1),NRR=round(r[1],1),n=r[2]))
df=pd.DataFrame(rows)
if df.empty: print("(no cells judged yet)");
else:
    df.to_csv("results/exact_comparison_summary.csv",index=False)
    print("saved long-form -> results/exact_comparison_summary.csv\n")
    # A) WHITE-BOX confound sweep (Qwen2-VL), all 4 variants
    wb=df[df.model=="Qwen2-VL(WB)"]
    for metric in ["ASR","NRR"]:
        print(f"### WHITE-BOX {metric}% (Qwen2-VL) ###")
        print(wb.pivot_table(index="patch",columns=["variant","judge"],values=metric,aggfunc="first").to_string(),"\n")
    # B) TRANSFER across all models, every variant
    for v,lbl in [("sb315","SafeBench-315"),("sb350","SafeBench-350"),("abraw","AdvBench-raw"),("abnorm","AdvBench-norm")]:
        sub=df[df.variant==v]
        for metric in ["ASR","NRR"]:
            print(f"### TRANSFER {metric}% — {lbl} ###")
            print(sub.pivot_table(index="model",columns=["patch","judge"],values=metric,aggfunc="first").to_string(),"\n")
PY
echo "$L done $(date)"
