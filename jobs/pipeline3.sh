#!/usr/bin/env bash
#SBATCH --job-name=ub_pipe3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
#
# PHASE-3 self-chaining pipeline — the previously-blocked analysis experiments,
# now implemented (see repro_notes/paper_specs.md for the exact paper settings).
#
#   step 1  A10  — Table 10 compute timing: CE vs semantic (analysis/timing_table.py)
#   step 2  C6   — Fig 6 first-token distribution at tau=0 / 0.5 / ->inf
#   step 3  C4   — Fig 4 (30x30) + Fig 5 (200x200) loss landscapes for CE + semantic tau 0/0.5/inf
#   step 4  C2a  — Fig 2a model-size transfer: surrogates {2B,7B} -> victims {3B,7B}
#                  (downloads Qwen2-VL-2B, Qwen2.5-VL-3B)
#
#   Seed:  sbatch jobs/pipeline3.sh 1     Resume at N:  sbatch jobs/pipeline3.sh N
set -uo pipefail
STEP="${1:?usage: sbatch jobs/pipeline3.sh <step-number>}"
TOTAL=4
REPO="/home2/home/ankur_d/sm/UltraBreak-Repro"; cd "$REPO"
source repro/bin/activate; export PYTHONUNBUFFERED=1
L="[pipe3 ${STEP}/${TOTAL}]"; echo "$L start $(date)"
CORPUS="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
P7B="outputs/full50_normfix_lr0p01_s3000/3000.png"     # 809 = Qwen2-VL-7B surrogate patch
OUTDIR="analysis/out"; mkdir -p "$OUTDIR" logs

submit_next(){ local n=$((STEP+1)); (( n>TOTAL )) && { echo "$L phase-3 complete."; return; }
  local j; j=$(sbatch --parsable jobs/pipeline3.sh "$n" 2>&1) \
    && echo "$L queued next step $n as job $j" || echo "$L WARN could not queue step $n: $j"; }

# score one config on one victim (gen + v3 judge, resume-guarded, Qwen2.5 victims)
score_one(){ local cfg="$1" m="$2" res="results/$1/$2.csv"
  [[ -f "$res" ]] || python evaluation/attack.py --model_name "$m" --attack_config "$cfg" || { echo "$L gen failed $m"; return; }
  [[ -f "results/$1/$2_harmbench_v3.csv" ]] || python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" || echo "$L judge failed $m"; }

echo "$L ============================================================"
case "$STEP" in
  1)  # A10 — timing table
      echo "$L A10: CE vs semantic s/iter"; submit_next
      python analysis/timing_table.py --iters 5 --warmup 2
      ;;
  2)  # C6 — first-token distribution at tau = 0 / 0.5 / ->inf
      echo "$L C6: first-token distribution"; submit_next
      # tau=0.5 = 809 (semantic attention). tau->0 = token-mode ablation if present.
      # tau->inf = train a short image here if missing.
      declare -A IMG=( [tau0p5]="$P7B" [tau0]="outputs/full50_abl_token/3000.png" )
      TINF="outputs/c6_tau_inf_s1300/1300.png"
      if [[ ! -f "$TINF" ]]; then
        echo "$L C6: training tau->inf image (1300 steps)"
        python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name c6_tau_inf_s1300 \
          --no_projection --seed 0 --num_epochs 1301 --lr 0.01 --loss semantic --loss_mode attention --tau 1000000 \
          || echo "$L tau-inf train nonzero"
      fi
      [[ -f "$TINF" ]] && IMG[tauinf]="$TINF"
      for tag in tau0 tau0p5 tauinf; do
        img="${IMG[$tag]:-}"; [[ -f "$img" ]] || { echo "$L C6 skip $tag (no image)"; continue; }
        python analysis/first_token_dist.py --image "$img" --label "$tag" --out "$OUTDIR/ftok_${tag}"
      done
      ;;
  3)  # C4 — loss landscape as per paper: Fig 4 (100-epoch image, 30x30 over 20x20)
      #      AND Fig 5 (fully-trained image, 200x200). CE + semantic tau 0/0.5/inf.
      echo "$L C4: loss landscapes (Fig 4 30x30 + Fig 5 200x200)"; submit_next
      [[ -f "$P7B" ]] || { echo "$L missing $P7B" >&2; exit 1; }
      IMG100="outputs/full50_normfix_lr0p01_s3000/100.png"   # 100-epoch checkpoint for Fig 4
      [[ -f "$IMG100" ]] || { echo "$L Fig4: no 100.png, falling back to fully-trained"; IMG100="$P7B"; }
      run_land(){ # img grid nrows out  loss [tau]
        local img="$1" grid="$2" nrows="$3" out="$4" loss="$5" tau="${6:-0.5}"
        [[ -f "${out}.npy" ]] && { echo "$L reuse landscape ${out}"; return; }
        local ta=""; [[ "$loss" == semantic ]] && ta="--tau $tau"
        python analysis/loss_landscape.py --image "$img" --loss "$loss" $ta \
          --grid "$grid" --range 20 --nrows "$nrows" --out "$out"; }
      # Fig 4 — 100-epoch image, 30x30, avg 3 rows (cheap)
      run_land "$IMG100" 30 3 "$OUTDIR/fig4_ce"        ce
      run_land "$IMG100" 30 3 "$OUTDIR/fig4_sem_tau0"  semantic 0.000001
      run_land "$IMG100" 30 3 "$OUTDIR/fig4_sem_tau0p5" semantic 0.5
      run_land "$IMG100" 30 3 "$OUTDIR/fig4_sem_tauinf" semantic 1000000
      # Fig 5 — fully-trained image, 200x200, 1 row (heavy; ~2h/panel, panel-resumable)
      run_land "$P7B" 200 1 "$OUTDIR/fig5_ce"        ce
      run_land "$P7B" 200 1 "$OUTDIR/fig5_sem_tau0"  semantic 0.000001
      run_land "$P7B" 200 1 "$OUTDIR/fig5_sem_tau0p5" semantic 0.5
      run_land "$P7B" 200 1 "$OUTDIR/fig5_sem_tauinf" semantic 1000000
      ;;
  4)  # C2a — model-size transfer: surrogates {2B,7B} -> victims {3B,7B}, SafeBench
      echo "$L C2a: model-size transfer"; submit_next
      # 1) weights (7B already cached; pull the missing three)
      for repo in Qwen/Qwen2-VL-2B-Instruct Qwen/Qwen2.5-VL-3B-Instruct; do  # 32B dropped (won't fit 49GB GPU)
        echo "$L C2a: ensuring weights $repo"; hf download "$repo" >/dev/null 2>&1 || echo "$L WARN download issue $repo"
      done
      # 2) surrogate patches: 7B = 809 (exists); 2B = train now
      P2B="outputs/c2a_qwen2b_s3000/3000.png"
      if [[ ! -f "$P2B" ]]; then
        echo "$L C2a: training 2B-surrogate patch (3000 steps)"
        python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name c2a_qwen2b_s3000 \
          --surrogates qwen2b --no_projection --seed 0 --num_epochs 3001 --lr 0.01 || echo "$L 2B train nonzero"
      fi
      # 3) transfer eval: each surrogate patch on each victim (SafeBench-315)
      VICTIMS=( "Qwen/Qwen2.5-VL-3B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" )  # 32B dropped
      declare -A SURRPATCH=( [7B]="$P7B" [2B]="$P2B" )
      for s in 7B 2B; do
        img="${SURRPATCH[$s]}"; [[ -f "$img" ]] || { echo "$L C2a skip surrogate $s (no patch)"; continue; }
        cfg="c2a_surr${s}_safebench"
        [[ -s "attack_configs/${cfg}.csv" ]] || python create_attack_configs.py --dataset safebench --config-type attack \
          --exclude-train datasets/SafeBench-Tiny.csv --image "$img" --output "attack_configs/${cfg}.csv" || continue
        for v in "${VICTIMS[@]}"; do echo "$L C2a: surrogate=$s -> victim=$v"; score_one "$cfg" "$v"; done
      done
      echo "$L C2a: ASR matrix (surrogate x victim, SafeBench-315, v3)"
      python - <<'PY'
import os,pandas as pd
victims=["Qwen/Qwen2.5-VL-3B-Instruct","Qwen/Qwen2.5-VL-7B-Instruct"]
print("%-14s %14s %14s"%("surrogate","victim-3B","victim-7B"))
for s in ["7B","2B"]:
    row=f"Qwen2-VL-{s:9s}"
    for v in victims:
        p=f"results/c2a_surr{s}_safebench/{v}_harmbench_v3.csv"
        if os.path.exists(p):
            d=pd.read_csv(p); row+=f"{d['attack_success'].astype(bool).mean()*100:>13.1f}%"
        else: row+=f"{'—':>14s}"
    print(row)
PY
      ;;
  *) echo "$L unknown step $STEP" >&2; exit 1 ;;
esac
echo "$L done $(date)"
