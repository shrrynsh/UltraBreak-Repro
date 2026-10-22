#!/usr/bin/env bash
#SBATCH --job-name=ub_t36pf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#SBATCH --signal=B:USR1@1200
#
# Tables 3 and 6, PAPER-FAITHFUL.
#
# Why this job exists. Our existing ablation patches (full50_abl_*) were all
# trained at **3001 epochs** — 2.3x the paper's stated budget. Paper Table 4 says
# **1300 iterations**, lr 0.01, TV 0.5, tau 0.5. So the Table 3 / Table 6 numbers we
# had were neither code-faithful nor paper-faithful: they were our own
# best-known-good configuration, and comparing them to the paper's ablations
# compares two different training budgets. Retrained here at the paper's budget.
#
# Everything except the ablated component matches paper Table 4 and the existing
# `repro_paper_faithful_1300` anchor: 1300 iters, lr 0.01, tau 0.5, transforms on,
# D11 normalisation fix on, seed 0, SafeBench-Tiny full-50 corpus.
#
#   Table 3 needs 3 trainings; "Proposed" = repro_paper_faithful_1300 (exists)
#     a_noconstraints_1300   --tv_weight 0 --no_transforms   (TV *and* transforms off)
#     a_ce_1300              --loss ce
#     a_token_1300           --loss_mode token
#   Table 6 needs 3 trainings; TV 0.5 = repro_paper_faithful_1300 (exists)
#     a_tv0_1300 / a_tv0p2_1300 / a_tv1p0_1300   (transforms stay ON — this is a
#     TV sweep, not the no-constraints ablation)
#
# ~5.2 GPU-h per training (14.32 s/epoch measured in A10) + ~15 GPU-h of eval.
# Per-stage resume markers; self-chains across the 24h wall.
set -uo pipefail
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate || { echo "[t36pf] repro venv missing" >&2; exit 1; }
export PYTHONUNBUFFERED=1
L="[t36pf]"; AUTH="c4c276d"; CORPUS="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
MARK="results/.t36pf"; mkdir -p "$MARK" logs
STAGE="${SLURM_TMPDIR:-/tmp}/t36pf_${SLURM_JOB_ID:-manual}"; mkdir -p "$STAGE"
ATTEMPT="${1:-1}"; MAX=8
EP=1301   # (num_epochs-1) must be a multiple of 20 so the saver writes 1300.png
CKPT=1300

echo "$L start $(date) attempt $ATTEMPT/$MAX"
python - <<'PY'
import importlib.metadata as m, torch, inspect, sys
assert torch.__version__.startswith("2.5.1") and m.version("transformers").startswith("4.51")
sys.path.insert(0, "optimisation")
from qwen2_adapter import Qwen2Adapter
assert "normalised_patch = (patch - mean_tensor) / std_tensor" in inspect.getsource(Qwen2Adapter.compute_loss)
print("[t36pf] env pins OK; D11 normalisation fix present")
PY
[[ $? -eq 0 ]] || exit 1


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

RESUB=0; done_n(){ ls "$MARK" 2>/dev/null | wc -l; }
START=$(done_n)
resub(){ [[ $RESUB -eq 1 ]] && return; RESUB=1; local d; d=$(done_n)
  (( ATTEMPT >= MAX )) && { echo "$L MAX attempts at $d"; return; }
  [[ "${1:-}" == end ]] && (( d <= START )) && { echo "$L no progress"; return; }
  local n; n=$(sbatch --parsable jobs/t36_paperfaithful.sh $((ATTEMPT+1)) 2>&1) \
    && echo "$L AUTO-RESUME $((ATTEMPT+1)) as job $n ($d markers)" || echo "$L resubmit failed: $n"; }
trap 'echo "$L USR1 pre-wall handoff"; resub; exit 0' USR1

# ── Stage 1: six paper-budget trainings ────────────────────────────────────
train(){   # exp  extra-flags...
  local exp="$1"; shift
  local out="outputs/${exp}/${CKPT}.png"
  [[ -f "$out" ]] && { echo "$L reuse $exp"; return 0; }
  guard $RESERVE_TRAIN "$exp" || { resub; exit 0; }
  # DO NOT resume a partial training. --base_epoch restores only the patch pixels;
  # `optim.Adam([adv_patch], lr=lr)` is rebuilt fresh, so Adam's moment estimates
  # are lost and the trajectory diverges from an uninterrupted run. The other five
  # ablations were each trained straight through, so a resumed sixth would not be
  # comparable to them - and comparability is the entire point of this job. The
  # wall guard above means a training only starts with 5.5h left against a measured
  # 4.4h, so interruption should be rare; when it happens, pay the 4.4h again.
  if [[ -d "outputs/${exp}" ]] && ls outputs/${exp}/*.png >/dev/null 2>&1; then
    echo "$L $exp has a PARTIAL run and no final checkpoint - discarding it and"
    echo "$L retraining from scratch, so every ablation is one uninterrupted run"
    rm -f outputs/${exp}/*.png outputs/${exp}/losses.csv
  fi
  echo ""; echo "$L --- train $exp (1300 iters, from scratch) : $* ---"
  python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name "$exp" \
    --surrogates qwen --no_projection --seed 0 --num_epochs $EP --lr 0.01 "$@" \
    || { echo "$L TRAIN FAILED $exp" >&2; return 1; }
  [[ -f "$out" ]] || { echo "$L $exp produced no ${CKPT}.png" >&2; return 1; }
  echo "$L trained $exp"
}

# Table 3 variants (TV defaults to 0.5 where not overridden)
train a_noconstraints_1300 --tv_weight 0   --no_transforms
train a_ce_1300            --loss ce
train a_token_1300         --loss_mode token
# Table 6 TV sweep — transforms stay ON
train a_tv0_1300           --tv_weight 0
train a_tv0p2_1300         --tv_weight 0.2
train a_tv1p0_1300         --tv_weight 1.0

# ── Stage 2: evaluation ────────────────────────────────────────────────────
git show "${AUTH}:evaluation/evaluate.py" > "$STAGE/evaluate_v1.py" || exit 1
cp "$STAGE/evaluate_v1.py" evaluation/_evaluate_v1_tmp.py
trap 'rm -f "${REPO}/evaluation/_evaluate_v1_tmp.py"' EXIT
GLM_PY="repro_glm/bin/python"
py_for_gen(){ [[ "$1" == "THUDM/GLM-4.1V-9B-Thinking" && -x "$GLM_PY" ]] && echo "$GLM_PY" || echo python; }

mkcfg(){   # cfg dataset image [--normalize]
  local cfg="$1" ds="$2" img="$3"; shift 3
  [[ -s "attack_configs/${cfg}.csv" ]] && return 0
  if [[ "$ds" == safebench ]]; then
    python create_attack_configs.py --dataset safebench --config-type attack \
      --exclude-train datasets/SafeBench-Tiny.csv --image "$img" --output "attack_configs/${cfg}.csv"
  else
    python create_attack_configs.py --dataset advbench --config-type attack --normalize \
      --image "$img" --output "attack_configs/${cfg}.csv"
  fi
}

score(){   # cfg model
  local cfg="$1" m="$2"
  local MK="$MARK/$(echo "${cfg}_${m}" | tr '/' '_')"; [[ -f "$MK" ]] && return 0
  cell_fits "$cfg" "$m" || { resub; exit 0; }
  local RES="results/${cfg}/${m}.csv"
  if [[ ! -f "$RES" ]]; then
    local gpy; gpy="$(py_for_gen "$m")"; echo "$L gen $cfg on $m"
    "$gpy" evaluation/attack.py --model_name "$m" --attack_config "$cfg" || return 1
  fi
  [[ -f "$RES" ]] || return 1
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] || \
    python evaluation/evaluate.py --attack_result "$RES" --output_suffix "_harmbench_v3.csv" >/dev/null || true
  [[ -f "results/${cfg}/${m}_harmbench.csv" ]] || \
    ( cd evaluation && python _evaluate_v1_tmp.py --attack_result "../$RES" ) >/dev/null || true
  [[ -f "results/${cfg}/${m}_harmbench_v3.csv" ]] && { touch "$MK"; echo "$L cell done $cfg/$m ($(done_n))"; }
}

# cheap models first, GLM last (GLM at 4096 is ~29 s/row and dominates the budget)
MODELS=( "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen-VL-Chat" "Qwen/Qwen2.5-VL-7B-Instruct" \
         "llava-hf/llava-v1.6-mistral-7b-hf" "THUDM/GLM-4.1V-9B-Thinking" )
T6_VICTIMS=( "Qwen/Qwen-VL-Chat" "THUDM/GLM-4.1V-9B-Thinking" )

for v in noconstraints ce token; do
  mkcfg "pf3_${v}_safebench" safebench "outputs/a_${v}_1300/${CKPT}.png" || true
  mkcfg "pf3_${v}_advbench"  advbench  "outputs/a_${v}_1300/${CKPT}.png" || true
done
for v in tv0 tv0p2 tv1p0; do
  mkcfg "pf6_${v}_safebench" safebench "outputs/a_${v}_1300/${CKPT}.png" || true
done

for m in "${MODELS[@]}"; do
  echo ""; echo "$L ########## $m ##########"
  for v in noconstraints ce token; do
    score "pf3_${v}_safebench" "$m"; score "pf3_${v}_advbench" "$m"
  done
  for tv in tv0 tv0p2 tv1p0; do
    for tm in "${T6_VICTIMS[@]}"; do [[ "$tm" == "$m" ]] && score "pf6_${tv}_safebench" "$m"; done
  done
done

echo ""; echo "$L ===== Table 3, PAPER-FAITHFUL (1300 iters), ours v3 vs paper ====="
python - <<'PY'
import os, pandas as pd
M={"Qwen2-VL":"Qwen/Qwen2-VL-7B-Instruct","Qwen-VL-Chat":"Qwen/Qwen-VL-Chat",
   "Qwen2.5-VL":"Qwen/Qwen2.5-VL-7B-Instruct","LLaVA-1.6":"llava-hf/llava-v1.6-mistral-7b-hf",
   "GLM-4.1V":"THUDM/GLM-4.1V-9B-Thinking"}
paper={"SafeBench":{"w/o Jailbreak Image":[15.87,51.11,12.38,92.06,7.62],
        "w/o Constraints":[89.84,49.21,27.30,83.81,47.62],"w/o Semantic Loss":[72.38,56.83,20.32,85.71,60.32],
        "w/o Attention Wgt":[68.25,42.54,37.78,79.37,70.48],"Proposed":[81.59,72.70,60.32,88.25,66.03]},
       "AdvBench":{"w/o Jailbreak Image":[10.19,10.38,0.38,91.15,2.12],
        "w/o Constraints":[63.27,27.69,2.31,80.00,9.42],"w/o Semantic Loss":[57.69,51.54,2.88,88.85,17.31],
        "w/o Attention Wgt":[47.50,42.54,15.77,79.37,29.62],"Proposed":[72.69,71.92,35.77,92.88,30.00]}}
ours={"SafeBench":{"w/o Jailbreak Image":"noattack_safebench_matrix","w/o Constraints":"pf3_noconstraints_safebench",
       "w/o Semantic Loss":"pf3_ce_safebench","w/o Attention Wgt":"pf3_token_safebench","Proposed":"cmp_pf_sb315"},
      "AdvBench":{"w/o Jailbreak Image":"noattack_advbench_matrix","w/o Constraints":"pf3_noconstraints_advbench",
       "w/o Semantic Loss":"pf3_ce_advbench","w/o Attention Wgt":"pf3_token_advbench","Proposed":"cmp_pf_abnorm"}}
def a(cfg,m):
    p=f"results/{cfg}/{m}_harmbench_v3.csv"
    return pd.read_csv(p).attack_success.astype(bool).mean()*100 if os.path.exists(p) else None
for bench in ("SafeBench","AdvBench"):
    print(f"\n  -- {bench} --")
    print(f"  {'variant':20s}"+"".join(f"{k:>16s}" for k in M))
    for var,cfg in ours[bench].items():
        o="  "+f"{var+' (ours)':20s}"; p="  "+f"{'(paper)':20s}"
        for i,(k,m) in enumerate(M.items()):
            v=a(cfg,m); o+=f"{(f'{v:.2f}' if v is not None else '·'):>16s}"
            p+=f"{paper[bench][var][i]:>16.2f}"
        print(o); print(p)
PY

echo ""; echo "$L ===== Table 6, PAPER-FAITHFUL (1300 iters), SafeBench v3 ====="
python - <<'PY'
import os, pandas as pd
V={"Qwen-VL-Chat":"Qwen/Qwen-VL-Chat","GLM-4.1V":"THUDM/GLM-4.1V-9B-Thinking"}
TV=[("0","pf6_tv0_safebench",61.27,70.48),("0.2","pf6_tv0p2_safebench",77.78,66.98),
    ("0.5","cmp_pf_sb315",72.70,66.03),("1.0","pf6_tv1p0_safebench",65.40,60.95)]
def a(cfg,m):
    p=f"results/{cfg}/{m}_harmbench_v3.csv"
    return pd.read_csv(p).attack_success.astype(bool).mean()*100 if os.path.exists(p) else None
print(f"  {'TV':5s} {'QwenChat ours':>14s} {'paper':>8s} {'GLM ours':>11s} {'paper':>8s}")
for tv,cfg,pq,pg in TV:
    q=a(cfg,V["Qwen-VL-Chat"]); g=a(cfg,V["GLM-4.1V"])
    print(f"  {tv:5s} {(f'{q:.2f}' if q is not None else '·'):>14s} {pq:>8.2f} "
          f"{(f'{g:.2f}' if g is not None else '·'):>11s} {pg:>8.2f}")
print("\n  Paper's peak is TV=0.2 on Qwen-VL-Chat (77.78). Does ours peak there too?")
PY

echo ""; echo "$L markers: $(done_n)/36"
TOTAL_CELLS=36   # 5 models x (3 variants x 2 benchmarks) + 3 TV x 2 victims
if (( $(done_n) >= TOTAL_CELLS )); then
  echo "$L COMPLETE - handing the GPU to the MM-SafetyBench n=1680 run"
  chain_to jobs/t1_mm1680.sh ub_mm1680
else
  resub end
fi
echo "$L finished $(date)"
