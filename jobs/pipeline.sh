#!/usr/bin/env bash
#SBATCH --job-name=ub_pipe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Self-chaining reproduction pipeline for the REPRO_CHECKLIST.
# =========================================================
# Cluster limits: MaxJobs=1 (one job runs at a time), MaxSubmit=3 (<=3 queued),
# MaxWall=24h. A self-chaining chain fits this exactly: each step, right after
# its cheap preflight, submits the NEXT step (which then PENDs until this one
# frees the single run slot), then does its heavy work. So:
#   - only ~2 queue slots are ever used (current running + next pending);
#   - a GPU failure or 24h timeout still advances (next is already queued);
#   - a preflight/config error halts the chain (submit_next runs only after the
#     step's inputs validate), so a bug does not cascade through every step.
#
# Steps are checklist-ordered (cheap scoring first, then ablation trainings).
# Edit the `case` below to add/drop/reorder; delete a step's block to skip it.
# Each step writes to results/<cfg>/... and is summarised at its end.
#
#   Seed once:  sbatch jobs/pipeline.sh 1
#   Resume at N: sbatch jobs/pipeline.sh N   (e.g. after a manual stop)

set -uo pipefail   # not -e: a step's failure must not stop the chain mid-work

STEP="${1:?usage: sbatch jobs/pipeline.sh <step-number>}"
TOTAL=13

REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
CORPUS="train_configs/safebench-tiny_jailbroken_mode_full50.csv"
SURR="Qwen/Qwen2-VL-7B-Instruct"
LOG="[pipe ${STEP}/${TOTAL}]"
MODELS_ALL=(
  "Qwen/Qwen2-VL-7B-Instruct" "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat"
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking"
)
# Black-box transfer targets = all models EXCEPT the Qwen2-VL surrogate.
TARGETS_BB=(
  "Qwen/Qwen2.5-VL-7B-Instruct" "Qwen/Qwen-VL-Chat"
  "llava-hf/llava-v1.6-mistral-7b-hf" "moonshotai/Kimi-VL-A3B-Instruct" "THUDM/GLM-4.1V-9B-Thinking"
)
# Transferability experiment arms (patches already trained, D11-correct, 3000 steps):
PATCH_PROJOFF="outputs/full50_normfix_lr0p01_s3000/3000.png"   # 809: projection OFF, white-box 83.49
PATCH_PROJON="outputs/full50_proj_projfix_s3000/3000.png"       # 843: projection ON,  white-box 42.54
CTRL_DIR="outputs/full50_normfix_lr0p01_s3000"                  # 809 checkpoints, for the matched-WB control
CTRL_FILE="outputs/transfer/control_ckpt.txt"                   # written by step 2, read by steps 3-4
CTRL_TARGET_WB=42.5                                             # match 843's white-box SafeBench

cd "$REPO_ROOT"
mkdir -p logs
if [[ -f repro/bin/activate ]]; then source repro/bin/activate; else echo "$LOG no venv" >&2; exit 1; fi
export PYTHONUNBUFFERED=1
echo "$LOG start $(date)"

submit_next() {
  local n=$((STEP + 1))
  if (( n > TOTAL )); then echo "$LOG pipeline complete — no successor."; return; fi
  local jid
  jid=$(sbatch --parsable "jobs/pipeline.sh" "$n" 2>&1) \
    && echo "$LOG queued next step $n as job $jid" \
    || echo "$LOG WARNING: could not queue step $n: $jid (re-run: sbatch jobs/pipeline.sh $n)"
}

# ---- helpers -----------------------------------------------------------------
# Build a per-(patch,dataset) attack config using each benchmark's own recipe.
build_cfg() {  # dataset patch out_cfg
  local ds="$1" img="$2" out="attack_configs/$3.csv"
  case "$ds" in
    safebench) python create_attack_configs.py --dataset safebench --config-type attack \
                 --exclude-train datasets/SafeBench-Tiny.csv --image "$img" --output "$out" ;;
    advbench)  python create_attack_configs.py --dataset advbench --config-type attack \
                 --normalize --image "$img" --output "$out" ;;
    mm-safetybench) python create_attack_configs.py --dataset mm-safetybench --config-type attack \
                 --subsample 520 --image "$img" --output "$out" ;;
    *) echo "$LOG unknown dataset $ds" >&2; return 1 ;;
  esac
}
judge_extra() { [[ "$1" == advbench ]] && echo "--behaviors_csv datasets/adv_bench.csv --behaviors_col goal"; }

# Score one config across a list of models (gen + v3 judge).
score() {  # cfg dataset model...
  local cfg="$1" ds="$2"; shift 2
  local extra; extra="$(judge_extra "$ds")"
  for m in "$@"; do
    echo "$LOG   gen ${cfg} on ${m}"
    python evaluation/attack.py --model_name "$m" --attack_config "$cfg" || { echo "$LOG gen failed $m"; continue; }
    local res="results/${cfg}/${m}.csv"
    [[ -f "$res" ]] || { echo "$LOG missing $res"; continue; }
    python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" $extra || echo "$LOG judge failed $m"
  done
}

# ASR% (v3) of an already-scored result CSV.
asr_of() {  # cfg model
  python - "results/$1/$2_harmbench_v3.csv" <<'PY'
import sys,pandas as pd
try: d=pd.read_csv(sys.argv[1]); print("%.2f"%(d["attack_success"].astype(bool).mean()*100))
except Exception: print("nan")
PY
}

# Transfer one patch to the 5 black-box targets on a benchmark (gen + v3 judge).
transfer_arm() {  # arm_tag patch dataset
  local tag="$1" img="$2" ds="$3"
  [[ -f "$img" ]] || { echo "$LOG transfer: missing patch $img" >&2; return 1; }
  local cfg="xfer_${tag}_${ds}"
  build_cfg "$ds" "$img" "$cfg"
  score "$cfg" "$ds" "${TARGETS_BB[@]}"
  echo "$LOG   [$tag/$ds] black-box ASR per target:"
  for m in "${TARGETS_BB[@]}"; do echo "$LOG     $m  $(asr_of "$cfg" "$m")%"; done
}

# Train an ablation (D11 fix on, no projection, 3000 steps) and eval white-box.
train_abl() {  # exp_name extra_optimise_flags...
  local name="$1"; shift
  python optimisation/optimise_proj.py --train_config "$CORPUS" --exp_name "$name" \
    --num_epochs 3001 --no_projection --seed 0 "$@"
  local img="outputs/${name}/3000.png"
  [[ -f "$img" ]] || { echo "$LOG training produced no ${img}"; return 1; }
  python - "$img" <<'PY'
import sys,numpy as np;from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float);print("  saved std=%.1f range=[%.0f,%.0f]"%(a.std(),a.min(),a.max()))
PY
  build_cfg safebench "$img" "${name}_safebench_eval"
  build_cfg advbench  "$img" "${name}_advbench_eval"
  score "${name}_safebench_eval" safebench "$SURR"
  score "${name}_advbench_eval"  advbench  "$SURR"
  python evaluation/summarise_asr.py --paper --pattern "$name" || true
}

# ---- the checklist steps -----------------------------------------------------
echo "$LOG ============================================================"
case "$STEP" in
  1)  # TRANSFER control-find: 809 checkpoint whose white-box SafeBench ~= 843's (~42.5%)
      # This makes the projection test FAIR: the control is a projection-OFF patch
      # matched to PROJ-ON's white-box strength, so any transfer difference is the
      # projection's doing, not raw strength.
      echo "$LOG TRANSFER control: find 809 checkpoint matching 843 white-box (~${CTRL_TARGET_WB}%)"
      mkdir -p outputs/transfer
      STEPS_SCAN="1000 1300 1600 1900 2200"
      submit_next
      for s in $STEPS_SCAN; do
        img="${CTRL_DIR}/${s}.png"; [[ -f "$img" ]] || { echo "$LOG   no ckpt $img"; continue; }
        cfg="ctrlscan_${s}_safebench"
        build_cfg safebench "$img" "$cfg"
        score "$cfg" safebench "$SURR"
      done
      python - "$CTRL_DIR" "$CTRL_FILE" "$CTRL_TARGET_WB" $STEPS_SCAN <<'PY'
import sys,os,pandas as pd
cdir,cfile,target=sys.argv[1],sys.argv[2],float(sys.argv[3]); steps=sys.argv[4:]
surr="Qwen/Qwen2-VL-7B-Instruct"; best=None
for s in steps:
    p=f"results/ctrlscan_{s}_safebench/{surr}_harmbench_v3.csv"
    if not os.path.exists(p): continue
    asr=pd.read_csv(p)["attack_success"].astype(bool).mean()*100
    print(f"  step {s}: white-box SafeBench {asr:.2f}%")
    if best is None or abs(asr-target)<abs(best[1]-target): best=(s,asr)
if best:
    png=f"{cdir}/{best[0]}.png"; open(cfile,"w").write(png+"\n")
    print(f"  -> CONTROL = step {best[0]} ({best[1]:.2f}% white-box; target {target}%) -> {png}")
else:
    print("  -> NO control checkpoint scored; steps 3-4 will skip the control arm")
PY
      ;;
  2)  # TRANSFER on SafeBench: PROJ-OFF (809) vs PROJ-ON (843) vs CONTROL (matched WB)
      echo "$LOG TRANSFER SafeBench x 5 black-box targets: projoff / projon / control"
      CTRL=""; [[ -f "$CTRL_FILE" ]] && CTRL=$(head -1 "$CTRL_FILE")
      submit_next
      transfer_arm projoff "$PATCH_PROJOFF" safebench
      transfer_arm projon  "$PATCH_PROJON"  safebench
      if [[ -n "$CTRL" && -f "$CTRL" ]]; then transfer_arm control "$CTRL" safebench
      else echo "$LOG   control checkpoint unavailable — skipping control arm (SafeBench)"; fi
      ;;
  3)  # TRANSFER on AdvBench: same three arms
      echo "$LOG TRANSFER AdvBench x 5 black-box targets: projoff / projon / control"
      CTRL=""; [[ -f "$CTRL_FILE" ]] && CTRL=$(head -1 "$CTRL_FILE")
      submit_next
      transfer_arm projoff "$PATCH_PROJOFF" advbench
      transfer_arm projon  "$PATCH_PROJON"  advbench
      if [[ -n "$CTRL" && -f "$CTRL" ]]; then transfer_arm control "$CTRL" advbench
      else echo "$LOG   control checkpoint unavailable — skipping control arm (AdvBench)"; fi
      ;;
  4)  # EXT-BENCHMARKS — StrongREJECT + HarmBench (native + norm) on the 5 patches
      echo "$LOG EXT-BENCHMARKS: running ext_benchmarks/run_ext_benchmarks.sh"
      submit_next
      bash ext_benchmarks/run_ext_benchmarks.sh || echo "$LOG ext-benchmarks returned nonzero"
      ;;
  5)  # REPRO — the paper-faithful reproduction (Table 4, paper-exact 1300 iters)
      echo "$LOG REPRO: running jobs/run_repro_paper_faithful.sh (reproducibility/paper_faithful)"
      submit_next
      bash jobs/run_repro_paper_faithful.sh || echo "$LOG repro run returned nonzero"
      ;;
  6)  # A1/A2 — the 2 missing authors'-image cells
      echo "$LOG A1/A2: complete authors' ultrabreak.png Table-1 cells"
      IMG="outputs/ultrabreak.png"; [[ -f "$IMG" ]] || { echo "$LOG missing $IMG" >&2; exit 1; }
      build_cfg safebench "$IMG" "authors_safebench_matrix"
      build_cfg advbench  "$IMG" "authors_advbench_matrix"
      submit_next
      score authors_safebench_matrix safebench "moonshotai/Kimi-VL-A3B-Instruct"
      score authors_advbench_matrix  advbench  "Qwen/Qwen-VL-Chat"
      ;;
  7)  # A3 — authors' image on MM-SafetyBench (520) x 6 models
      echo "$LOG A3: authors' image x MM-SafetyBench-520 x 6 models"
      IMG="outputs/ultrabreak.png"; [[ -f "$IMG" ]] || { echo "$LOG missing $IMG" >&2; exit 1; }
      build_cfg mm-safetybench "$IMG" "authors_mmsafety_matrix"
      submit_next
      score authors_mmsafety_matrix mm-safetybench "${MODELS_ALL[@]}"
      ;;
  8)  # B11 — No-Attack column x 6 models on SafeBench + AdvBench
      echo "$LOG B11: No-Attack (white.jpeg, no phrase) x 6 models on SafeBench + AdvBench"
      python create_attack_configs.py --dataset safebench --config-type attack --no-attack \
        --image images/white.jpeg --exclude-train datasets/SafeBench-Tiny.csv \
        --output attack_configs/noattack_safebench_matrix.csv
      python create_attack_configs.py --dataset advbench --config-type attack --no-attack \
        --normalize --image images/white.jpeg \
        --output attack_configs/noattack_advbench_matrix.csv
      submit_next
      score noattack_safebench_matrix safebench "${MODELS_ALL[@]}"
      score noattack_advbench_matrix  advbench  "${MODELS_ALL[@]}"
      ;;
  9)  # B13 — Table 3 "w/o semantic loss": CE baseline (also yields A10 timing)
      echo "$LOG B13: CE-loss retrain (--loss ce)"; submit_next
      train_abl full50_abl_ce --loss ce
      ;;
  10) # B14 — Table 3 "w/o attention weighting": token mode (tau->0)
      echo "$LOG B14: token-mode retrain (--loss_mode token)"; submit_next
      train_abl full50_abl_token --loss_mode token
      ;;
  11) # B12 — Table 3 "w/o constraints": no transforms, no TV
      echo "$LOG B12: no-constraints retrain (--no_transforms --tv_weight 0)"; submit_next
      train_abl full50_abl_noconstraints --no_transforms --tv_weight 0
      ;;
  12) # B16a — Table 6 TV sweep: lambda_TV = 0.2 (paper's claimed peak)
      echo "$LOG B16a: TV weight 0.2"; submit_next
      train_abl full50_abl_tv0p2 --tv_weight 0.2
      ;;
  13) # B16b — Table 6 TV sweep: lambda_TV = 1.0
      echo "$LOG B16b: TV weight 1.0"; submit_next
      train_abl full50_abl_tv1p0 --tv_weight 1.0
      ;;
  *) echo "$LOG unknown step $STEP" >&2; exit 1 ;;
esac

echo "$LOG done $(date)"
