#!/usr/bin/env bash
#SBATCH --job-name=ub_cmp1300
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Compare the three projection regimes at the PAPER-EXACT 1300-iteration checkpoint.
# ==============================================================================
# All three trained on Qwen2-VL, 50-row corpus, seed 0; we take each run's
# 1300.png (Table 4's stated budget) and score it white-box.
#
#   809  outputs/full50_normfix_lr0p01_s3000/1300.png   no projection + D11 fix   (full range, std 33)
#   843  outputs/full50_proj_projfix_s3000/1300.png      projection + D11 fix      (1 squash, band, std 12)
#   844  outputs/full50_proj_doublesquash_s3000/1300.png projection, NO fix        (2 squash, band, std 11)
#
# Question: at the paper's own 1300 iterations, how do the projection regimes
# compare? (The 3000-step versions are 809=83.5/39.0, 843=42.5/17.9.)

set -uo pipefail
REPO_ROOT="/home2/home/ankur_d/sm/UltraBreak-Repro"
MODEL="Qwen/Qwen2-VL-7B-Instruct"
LOG="[cmp1300]"

cd "$REPO_ROOT"; mkdir -p logs
if [[ -f repro/bin/activate ]]; then source repro/bin/activate; else echo "$LOG no venv" >&2; exit 1; fi
export PYTHONUNBUFFERED=1

# tag -> 1300 checkpoint
declare -A P=(
  [809_noproj_fix]="outputs/full50_normfix_lr0p01_s3000/1300.png"
  [843_proj_fix]="outputs/full50_proj_projfix_s3000/1300.png"
  [844_proj_nofix]="outputs/full50_proj_doublesquash_s3000/1300.png"
)

score() {  # cfg dataset extra...
  local cfg="$1" ds="$2"; shift 2
  python evaluation/attack.py --model_name "$MODEL" --attack_config "$cfg" || { echo "$LOG gen failed $cfg"; return 1; }
  local res="results/${cfg}/${MODEL}.csv"
  [[ -f "$res" ]] || { echo "$LOG missing $res"; return 1; }
  python evaluation/evaluate.py --attack_result "$res" --output_suffix "_harmbench_v3.csv" "$@"
}

for tag in 809_noproj_fix 843_proj_fix 844_proj_nofix; do
  img="${P[$tag]}"
  [[ -f "$img" ]] || { echo "$LOG MISSING $img" >&2; continue; }
  echo ""; echo "$LOG ===== ${tag} @1300 ====="
  SB="cmp1300_${tag}_safebench"; AB="cmp1300_${tag}_advbench"
  python create_attack_configs.py --dataset safebench --config-type attack \
    --exclude-train datasets/SafeBench-Tiny.csv --image "$img" --output "attack_configs/${SB}.csv"
  python create_attack_configs.py --dataset advbench --config-type attack --normalize \
    --image "$img" --output "attack_configs/${AB}.csv"
  score "$SB" safebench
  score "$AB" advbench --behaviors_csv datasets/adv_bench.csv --behaviors_col goal
done

echo ""; echo "$LOG ===== COMPARISON (v3 judge, 1300 iterations) ====="
python - <<'PY'
import pandas as pd, numpy as np, os
from PIL import Image
rows=[("809_noproj_fix","no-proj + fix","outputs/full50_normfix_lr0p01_s3000/1300.png"),
      ("843_proj_fix","proj + fix (1 squash)","outputs/full50_proj_projfix_s3000/1300.png"),
      ("844_proj_nofix","proj + NO fix (2 squash)","outputs/full50_proj_doublesquash_s3000/1300.png")]
M="Qwen/Qwen2-VL-7B-Instruct"
def asr(cfg):
    p=f"results/{cfg}/{M}_harmbench_v3.csv"
    if not os.path.exists(p): return float("nan")
    return pd.read_csv(p)["attack_success"].astype(bool).mean()*100
print("%-26s %6s %9s %9s"%("regime @1300","std","SafeBench","AdvBench"))
for tag,desc,img in rows:
    a=np.asarray(Image.open(img).convert("RGB")).astype(float)
    print("%-26s %6.1f %8.2f%% %8.2f%%"%(desc,a.std(),asr(f"cmp1300_{tag}_safebench"),asr(f"cmp1300_{tag}_advbench")))
print("\nreference @3000: 809=83.49/39.04, 843=42.54/17.88")
PY
echo "$LOG done"
