#!/usr/bin/env bash
#SBATCH --job-name=ub_glmsmoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
# Smoke-test GLM-4.1V generation under repro_glm (transformers 5.x) on 2 rows
# BEFORE committing the multi-hour full transfer run.
set -uo pipefail
cd /home2/home/ankur_d/sm/UltraBreak-Repro
export PYTHONUNBUFFERED=1
echo "[glmsmoke] start $(date)"
repro_glm/bin/python evaluation/attack.py \
  --model_name "THUDM/GLM-4.1V-9B-Thinking" --attack_config glm_smoke
rc=$?
echo "[glmsmoke] attack.py rc=$rc"
OUT="results/glm_smoke/THUDM/GLM-4.1V-9B-Thinking.csv"
repro_glm/bin/python - "$OUT" <<'PY'
import sys, pandas as pd
try:
    d = pd.read_csv(sys.argv[1])
    n = len(d)
    nonempty = d["response"].astype(str).str.strip().replace("nan","").astype(bool).sum()
    print(f"[glmsmoke] rows={n} nonempty_responses={nonempty}")
    print("[glmsmoke] SAMPLE:", repr(str(d["response"].iloc[0])[:200]))
    print("[glmsmoke] VERDICT:", "PASS" if nonempty>0 else "FAIL")
except Exception as e:
    print("[glmsmoke] VERDICT: FAIL —", type(e).__name__, e)
PY
echo "[glmsmoke] done $(date)"
