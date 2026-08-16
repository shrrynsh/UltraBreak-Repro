#!/usr/bin/env bash
#SBATCH --job-name=ub_repro_pf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Canonical PAPER-FAITHFUL reproduction: Table 4 / §3.2, paper-exact 1300 iters.
# =============================================================================
# Drives ONLY reproducibility/paper_faithful/ (frozen code) — NOT the working
# tree, and NOT part of jobs/pipeline.sh (the ablation chain). Kept separate so
# the reproduction and the ablations cannot affect each other.
#
# Expectation (the documented gap): at 1300 iterations this UNDER-reproduces
# (~job 773, SafeBench ~23%); ~3000 reproduces (~job 809, ~83%). Running 1300 is
# the paper-exact fidelity choice — see reproducibility/README.md.

set -euo pipefail
REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
EXP="repro_paper_faithful_1300"
IMG="outputs/${EXP}/1300.png"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG="[repro-pf]"

cd "$REPO_ROOT"; mkdir -p logs
if [[ -f repro/bin/activate ]]; then source repro/bin/activate; else echo "$LOG no venv" >&2; exit 1; fi
export PYTHONUNBUFFERED=1

echo "$LOG ===== train: paper-faithful, Table 4, 1300 iterations ====="
python reproducibility/paper_faithful/optimise_paper.py
[[ -f "$IMG" ]] || { echo "$LOG expected $IMG not produced" >&2; exit 1; }
python - "$IMG" <<'PY'
import sys,numpy as np;from PIL import Image
a=np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(float)
print("  paper-faithful patch: std=%.1f range=[%.0f,%.0f]  (projection band -> expect low std)"%(a.std(),a.min(),a.max()))
PY

echo "$LOG ===== eval white-box (SafeBench-315 + AdvBench-520, v3 judge) ====="
SB="${EXP}_safebench_eval"; AB="${EXP}_advbench_eval"
python create_attack_configs.py --dataset safebench --config-type attack \
  --exclude-train datasets/SafeBench-Tiny.csv --image "$IMG" --output "attack_configs/${SB}.csv"
python create_attack_configs.py --dataset advbench --config-type attack --normalize \
  --image "$IMG" --output "attack_configs/${AB}.csv"

for cfg in "$SB" "$AB"; do
  python evaluation/attack.py --model_name "$MODEL" --attack_config "$cfg"
  res="results/${cfg}/${MODEL}.csv"
  [[ -f "$res" ]] || { echo "$LOG missing $res" >&2; continue; }
  extra=""; [[ "$cfg" == "$AB" ]] && extra="--behaviors_csv datasets/adv_bench.csv --behaviors_col goal"
  python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" $extra
done

echo "$LOG ===== RESULT (paper-exact 1300; compare 773 @1300 ~23%, 809 @3000 ~83%) ====="
python evaluation/summarise_asr.py --paper --pattern "$EXP" || true
echo "$LOG done"
