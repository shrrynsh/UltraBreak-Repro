#!/usr/bin/env bash
#SBATCH --job-name=ub_glmprobe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err
set -uo pipefail
cd /home2/home/ankur_d/sm/UltraBreak-Repro
export PYTHONUNBUFFERED=1
echo "[glmprobe] start $(date)"
# GLM runs in repro_glm (transformers 5.x has Glm4v)
repro_glm/bin/python analysis/glm_budget_probe.py --ceiling 6144 --per_cfg 4
echo "[glmprobe] done $(date)"
