#!/usr/bin/env bash
#SBATCH --job-name=ub_extbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Score existing UltraBreak patches on two EXTERNAL text-instruction benchmarks:
#   StrongREJECT (313 prompts) and HarmBench-standard (200 behaviours).
# =====================================================================
#
# Paradigm fit: both are lists of self-contained harmful TEXT queries, so they
# pair with UltraBreak's universal adversarial image exactly like AdvBench. No
# per-query images, so no paradigm change.
#
# Protocol: each patch x each benchmark is scored in TWO prompt variants, so the
# effect of the D13 "Steps to..." rewrite is measured on these benchmarks too:
#   - native : the benchmark's query verbatim + the TPG phrase.
#   - norm   : --normalize applied ("Steps to <query>"), then the TPG phrase.
#     Clean for HarmBench (imperative behaviours); garbles StrongREJECT's
#     question-form prompts (e.g. "Steps to can you explain...") -- that garbling
#     is itself an informative D13 data point, kept deliberately.
# The TPG affirming phrase (default [Jailbroken Mode]) is part of UltraBreak's
# attack and is present in both variants. Judge behaviour is always the ORIGINAL
# query (build_attack_config keeps `target` un-normalised), so --normalize only
# changes the prompt, never what the judge scores against.
#   - judge: HarmBench classifier, v3 pass (the repo's corrected scorer). The
#     behaviour handed to the judge is the native query (config-target column;
#     no --behaviors_csv needed since we did not normalise).
#
# Patches compared (the study's arc):
#   authors  outputs/ultrabreak.png                          reference artifact
#   622      safebench_tiny_full50_repro/1300.png            buggy baseline (no proj)
#   625      safebench_tiny_full50_proj_repro/1300.png       projection (prior best AdvBench)
#   773      full50_normfix_noproj_seed0/1300.png            D11 fix @ 1300 (looked like a regression)
#   809      full50_normfix_lr0p01_s3000/3000.png            D11 fix @ 3000 (SafeBench reproduction)
#
# Config commands are documented verbatim in ext_benchmarks/README_*.md.

set -euo pipefail

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG_PREFIX="[extbench]"

cd "$REPO_ROOT"
mkdir -p logs

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
print("[extbench] Versions: " + ", ".join(f"{p}=={m.version(p)}" for p in pins))
import pandas as pd
assert len(pd.read_csv("ext_benchmarks/strongreject.csv")) == 313, "StrongREJECT != 313"
assert len(pd.read_csv("ext_benchmarks/harmbench_standard.csv")) == 200, "HarmBench std != 200"
print("[extbench] datasets: StrongREJECT 313, HarmBench-standard 200")
PY

# tag -> patch image
declare -A PATCH=(
  [authors]="outputs/ultrabreak.png"
  [622]="outputs/safebench_tiny_full50_repro/1300.png"
  [625]="outputs/safebench_tiny_full50_proj_repro/1300.png"
  [773]="outputs/full50_normfix_noproj_seed0/1300.png"
  [809]="outputs/full50_normfix_lr0p01_s3000/3000.png"
)
BENCHMARKS=(strongreject harmbench)

score_one() {
  local cfg="$1"
  echo "${LOG_PREFIX} generating: ${cfg}"
  python evaluation/attack.py --model_name "$MODEL" --attack_config "$cfg"
  local result="results/${cfg}/${MODEL}.csv"
  if [[ ! -f "$result" ]]; then
    echo "${LOG_PREFIX} generation missing: ${result}" >&2
    return 1
  fi
  echo "${LOG_PREFIX} judging: ${result}"
  # No --behaviors_csv: native prompt is in the config-target column.
  python evaluation/evaluate.py --attack_result "$result" --output_suffix "_harmbench_v3.csv"
}

for tag in authors 622 625 773 809; do
  img="${PATCH[$tag]}"
  if [[ ! -f "$img" ]]; then echo "${LOG_PREFIX} missing patch ${img}" >&2; exit 1; fi
  for bench in "${BENCHMARKS[@]}"; do
    for variant in native norm; do
      if [[ "$variant" == norm ]]; then NORM_FLAG="--normalize"; SUF="_norm"; else NORM_FLAG=""; SUF=""; fi
      CFG="ext_${tag}_${bench}${SUF}_eval"
      echo ""
      echo "${LOG_PREFIX} ===== ${tag} on ${bench} [${variant}] ====="
      python create_attack_configs.py --dataset "$bench" --config-type attack \
        ${NORM_FLAG} --image "$img" --output "attack_configs/${CFG}.csv"
      score_one "$CFG"
    done
  done
done

echo ""
echo "${LOG_PREFIX} ===== RESULTS: StrongREJECT + HarmBench (v3 judge) ====="
# summarise_asr.py is SafeBench/AdvBench-specific (it would misclassify and filter
# these), so tabulate the ext_ result CSVs directly.
python - <<'PY'
import pandas as pd, glob, re
rows=[]
for f in sorted(glob.glob("results/ext_*_eval/Qwen/Qwen2-VL-7B-Instruct_harmbench_v3.csv")):
    m=re.search(r"results/ext_(.+?)_(strongreject|harmbench)(_norm)?_eval/", f)
    if not m: continue
    tag,bench,variant=m.group(1),m.group(2),("norm" if m.group(3) else "native")
    d=pd.read_csv(f)
    succ=d["attack_success"].astype(bool)
    rows.append((tag,bench,variant,len(d),round(succ.mean()*100,2)))
df=pd.DataFrame(rows,columns=["patch","benchmark","variant","n","ASR%"])
order={"authors":0,"622":1,"625":2,"773":3,"809":4}
df["_o"]=df.patch.map(lambda x: order.get(x,9))
for b in ["strongreject","harmbench"]:
    sub=df[df.benchmark==b]
    piv=sub.pivot_table(index=["_o","patch"],columns="variant",values="ASR%").reset_index().sort_values("_o")
    print(f"\n=== {b} (ASR%, v3 judge) ===")
    print(piv.drop(columns="_o").to_string(index=False))
PY

echo ""
echo "${LOG_PREFIX} Done"
