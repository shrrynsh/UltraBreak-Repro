#!/bin/bash

# Activate your environment if necessary
cd ~/UltraBreak

# Log file with timestamp
LOG_FILE="logs/run_evaluation_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Evaluation run started at $(date) ====="
echo "Logging to: $LOG_FILE"
echo ""

# Run your python script
echo "----- Evaluating Qwen2.5-VL-7B-Instruct -----"
python evaluation/evaluate.py --attack_result results/safebench_jailbroken_mode/Qwen/Qwen2.5-VL-7B-Instruct.csv

echo "----- Evaluating Qwen2-VL-7B-Instruct -----"
python evaluation/evaluate.py --attack_result results/safebench_jailbroken_mode/Qwen/Qwen2-VL-7B-Instruct.csv

echo "----- Evaluating llava-v1.6-mistral-7b-hf -----"
python evaluation/evaluate.py --attack_result results/safebench_jailbroken_mode/llava-hf/llava-v1.6-mistral-7b-hf.csv

echo "----- Evaluating GLM-4.1V-9B-Thinking -----"
python evaluation/evaluate.py --attack_result results/safebench_jailbroken_mode/THUDM/GLM-4.1V-9B-Thinking.csv


echo ""
echo "===== Evaluation run finished at $(date) ====="
