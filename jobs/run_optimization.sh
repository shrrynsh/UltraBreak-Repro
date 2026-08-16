#!/bin/bash
#SBATCH --job-name=adv_patch_opt
#SBATCH --gres=gpu:1           # Request 1 GPU [cite: 15, 30]
#SBATCH --cpus-per-task=10     # Request 10 CPU threads [cite: 31]
#SBATCH --mem=40G              # Request 40GB RAM [cite: 32]
#SBATCH --time=24:00:00        # Time limit [cite: 29]
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Activate your environment if necessary
cd ~/sm/UltraBreak
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ultrabreak

# Run your python script
python optimisation/optimise.py