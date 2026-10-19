#!/usr/bin/env bash
#SBATCH --job-name=ub_mm1680
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# Table 1, MM-SafetyBench column — at the PAPER'S eval-set size.
#
# Why. Every MM-SafetyBench number in this study was measured on a seed-0
# stratified **520-row subsample**. The paper uses all **1680**. That is not an
# inference from its text: 1680 is the UNIQUE denominator in [50,2000] consistent
# with all 24 MM-SafetyBench values it reports (the same forensics give SafeBench
# 315 and AdvBench 520, which do match us). A subsample is an unbiased estimator
# but carries ~2.2 points of sampling error, so it cannot settle a reproduction.
#
#   4 methods x 6 models x 1680 rows = 40,320 generations, ~71 GPU-h
#   (GLM at the 4096-token budget is ~54 of those hours)
#
# Ordered so the headline lands first: UltraBreak's released image, then No-Attack,
# then the two baselines; models cheap-first with GLM last. Per-cell resume markers,
# self-chaining across the 24h wall.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[mm1680] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[mm1680]"; AUTH="c4c276d"; MARK="results/.mm1680"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/mm1680_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=10

echo "$L start $(date) attempt $ATTEMPT/$MAX"
python - <<'PY'
import importlib.metadata as m, torch, pandas as pd
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
n = len(pd.read_csv("datasets/mm_safetybench.csv"))
assert n == 1680, f"mm_safetybench.csv has {n} rows, expected the paper's 1680"
print(f"[mm1680] env pins OK; full MM-SafetyBench present ({n} rows)")
PY
[[ $? -eq 0 ]] || exit 1

# cfg : image  ("" = no image, the No-Attack control)
declare -A IMG=(
  [mm1680_authors]="outputs/ultrabreak.png"
  [mm1680_noattack]="images/white.jpeg"
  [mm1680_vajm]="baselines/out/vajm_5000/bad_prompt.png"
  [mm1680_umk]="baselines/out/umk_2000/bad_vlm_prompt.png"
)
ORDER=( mm1680_authors mm1680_noattack mm1680_vajm mm1680_umk )
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" "Qwen/Qwen2.5-VL-7B-Instruct" \
         "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" \
         "THUDM/GLM-4.1V-9B-Thinking" )
TOTAL=$(( ${#ORDER[@]} * ${#MODELS[@]} ))


# ── chain to the next stage ────────────────────────────────────────────────────
# One running job and at most 3 submitted, so the night's work is a chain: each
# stage self-resumes until ITS OWN work is done, then seeds the next stage once.
# Keeping only one queued also leaves submit slots free for that self-resume.
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


# ── wall guard ────────────────────────────────────────────────────────────────
# A bash trap does NOT interrupt a running foreground command - it is deferred
# until that command returns. Our inner python runs for hours, so the USR1
# pre-wall trap never executes and the job is SIGKILLed at the wall with no
# handoff (exactly what killed job 9138 at 5/6 trainings). So check elapsed time
# explicitly BETWEEN stages and hand off before starting one we cannot finish.
WALL_LIMIT=86400          # 24h
RESERVE_TRAIN=19800       # 5.5h - one 1300-iteration training measured ~4.4h
RESERVE_EVAL=5400         # 1.5h - a single eval cell
wall_left(){ echo $(( WALL_LIMIT - SECONDS )); }

# Per-cell runtime estimate. A flat reserve is wrong because GLM at the corrected
# 4096-token budget runs ~29 s/row while every other model runs 1.4-2.5 s/row: a
# GLM cell at 520 rows needs 4.2h and at 1680 rows needs 13.5h. attack.py writes
# its CSV only after the whole config, so a cell killed mid-generation loses ALL
# its rows and repeats identically on the next resume - it would never finish.
# So: start a cell only if the WHOLE cell fits in the remaining wall time.
rate_for(){ case "$1" in
    *GLM-4.1V*)      echo 29.0 ;;
    *Kimi-VL*)       echo 2.5 ;;
    *llava*)         echo 2.2 ;;
    *Qwen-VL-Chat*)  echo 1.4 ;;
    *)               echo 1.6 ;;
  esac; }
cell_fits(){   # cfg model -> 0 if the whole cell fits in what's left
  local cfg="$1" m="$2"
  local rows; rows=$(( $(wc -l < "attack_configs/${cfg}.csv") - 1 ))
  local est; est=$(python -c "print(int($rows*$(rate_for "$m")+600))")   # +10m model load
  if (( SECONDS + est > WALL_LIMIT - 600 )); then
    echo "$L wall: ${cfg}/${m} needs ~$((est/60))m, only $(( (WALL_LIMIT-SECONDS)/60 ))m left - handing off"
    return 1
  fi
  return 0
}

guard(){   # reserve label -> returns 1 if we should stop now
  local need="$1" what="$2"
  if (( SECONDS > WALL_LIMIT - need )); then
    echo "$L wall: ${SECONDS}s used, $(wall_left)s left - not starting $what"
    return 1
  fi
  return 0
}

done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START=$(done_n); RESUB=0
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( d >= TOTAL )) && { echo "$L COMPLETE $d/$TOTAL"; return; }
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d/$TOTAL"; return; }
  [[ "${1:-}" == end ]] && (( d <= START )) && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable jobs/t1_mm1680.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d/$TOTAL)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# Build the four configs on the FULL 1680 — no --subsample.
# No-Attack keeps the paper's control shape: bare query, no TPG phrase.
for cfg in "${ORDER[@]}"; do
  [[ -s "attack_configs/${cfg}.csv" ]] && continue
  img="${IMG[$cfg]}"
  if [[ "$cfg" == mm1680_noattack ]]; then
    python create_attack_configs.py --dataset mm-safetybench --config-type attack \
      --no-attack --image "$img" --output "attack_configs/${cfg}.csv" || exit 1
  else
    [[ -f "$img" ]] || { echo "$L missing image for $cfg: $img" >&2; continue; }
    python create_attack_configs.py --dataset mm-safetybench --config-type attack \
      --image "$img" --output "attack_configs/${cfg}.csv" || exit 1
  fi
done
for cfg in "${ORDER[@]}"; do
  if [[ -s "attack_configs/${cfg}.csv" ]]; then
    n=$(( $(wc -l < "attack_configs/${cfg}.csv") - 1 ))
    (( n == 1680 )) || { echo "$L $cfg has $n rows, expected 1680" >&2; exit 1; }
    echo "$L config $cfg: $n rows"
  fi
done

git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

for m in "${MODELS[@]}"; do
  echo ""; echo "$L ########## $m ##########"
  for cfg in "${ORDER[@]}"; do
    [[ -s "attack_configs/${cfg}.csv" ]] || continue
    MK="$MARK/$(echo "${cfg}_${m}" | tr '/' '_')"; [[ -f "$MK" ]] && { echo "$L skip $cfg/$m"; continue; }
    cell_fits "$cfg" "$m" || { resub; exit 0; }
    RES="results/${cfg}/${m}.csv"
    if [[ ! -f "$RES" ]]; then
      gpy="$(py_for_gen "$m")"; echo "$L gen $cfg on $m (1680 rows)"
      "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" \
        || { echo "$L gen failed $cfg/$m" >&2; continue; }
    fi
    [[ -f "$RES" ]] || continue
    [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || \
      python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" >/dev/null || true
    [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || \
      ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null || true
    [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { touch "$MK"; echo "$L done $cfg/$m ($(done_n)/$TOTAL)"; }
  done
done

echo ""; echo "$L ===== Table 1, MM-SafetyBench at n=1680 (paper's own size) ====="
python - <<'PY'
import os, pandas as pd
M=[("Qwen2-VL","Qwen/Qwen2-VL-7B-Instruct"),("Qwen-VL-Chat","Qwen/Qwen-VL-Chat"),
   ("Qwen2.5-VL","Qwen/Qwen2.5-VL-7B-Instruct"),("LLaVA-1.6","llava-hf/llava-v1.6-mistral-7b-hf"),
   ("Kimi-VL","moonshotai/Kimi-VL-A3B-Instruct"),("GLM-4.1V","THUDM/GLM-4.1V-9B-Thinking")]
paper={"mm1680_noattack":[26.19,21.49,33.45,35.06,41.79,43.69],
       "mm1680_vajm":[5.42,11.73,26.79,30.18,35.36,36.73],
       "mm1680_umk":[54.76,5.48,17.56,21.96,26.67,37.44],
       "mm1680_authors":[57.26,53.10,45.83,71.90,54.58,67.08]}
name={"mm1680_noattack":"No Attack","mm1680_vajm":"VAJM","mm1680_umk":"UMK","mm1680_authors":"UltraBreak"}
old={"mm1680_noattack":"noattack_mmsafety_matrix","mm1680_vajm":"a5_vajm_mmsafety",
     "mm1680_umk":"a5_umk_mmsafety","mm1680_authors":"authors_mmsafety_matrix"}
def a(cfg,m,suf="_harmbench_v3.csv"):
    p=f"results/{cfg}/{m}{suf}"
    if not os.path.exists(p): return None
    d=pd.read_csv(p); return d.attack_success.astype(bool).mean()*100, len(d)
print(f"  {'method':11s} {'model':13s} {'paper':>7s} {'ours n=1680':>12s} {'ours n=520':>11s} {'subsample err':>14s}")
for cfg in ["mm1680_authors","mm1680_noattack","mm1680_vajm","mm1680_umk"]:
    for i,(sh,m) in enumerate(M):
        new=a(cfg,m); o=a(old[cfg],m); pv=paper[cfg][i]
        nv=f"{new[0]:.2f} ({new[1]})" if new else "·"
        ov=f"{o[0]:.2f}" if o else "·"
        err=f"{new[0]-o[0]:+.2f}" if (new and o) else "·"
        print(f"  {name[cfg]:11s} {sh:13s} {pv:>7.2f} {nv:>12s} {ov:>11s} {err:>14s}")
print("\n  'subsample err' = how far the 520-row estimate sat from the full-set value.")
PY

echo ""; echo "$L cells: $(done_n)/$TOTAL"
if (( $(done_n) >= TOTAL )); then
  echo "$L COMPLETE - handing the GPU to the UMK/Qwen2-VL surrogate run"
  chain_to jobs/umk_qwen_surrogate.sh ub_umkq
else
  resub end
fi
echo "$L finished $(date)"
