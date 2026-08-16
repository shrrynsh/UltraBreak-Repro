#!/bin/bash

# Activate your environment if necessary
cd ~/UltraBreak

# Log file with timestamp
LOG_FILE="logs/run_attack_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Redirect all stdout and stderr to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== Attack run started at $(date) ====="
echo "Logging to: $LOG_FILE"
echo ""

# Run your python script
# echo "----- Running GLM-4.1V-9B-Thinking -----"
# python evaluation/attack.py --model_name "THUDM/GLM-4.1V-9B-Thinking" --attack_config "advbench_jailbroken_mode_1300"

# echo "----- Running llava-v1.6-mistral-7b-hf -----"
# python evaluation/attack.py --model_name "llava-hf/llava-v1.6-mistral-7b-hf" --attack_config "advbench_jailbroken_mode_1300"

echo "----- Running Qwen-VL-Chat -----"
python evaluation/attack.py --model_name "Qwen/Qwen-VL-Chat" --attack_config "advbench_jailbroken_mode_1300"

# echo "----- Running Qwen2-VL-7B-Instruct -----"
# python evaluation/attack.py --model_name "Qwen/Qwen2-VL-7B-Instruct" --attack_config "advbench_jailbroken_mode_1300"

# echo "----- Running Qwen2.5-VL-7B-Instruct -----"
# python evaluation/attack.py --model_name "Qwen/Qwen2.5-VL-7B-Instruct" --attack_config "advbench_jailbroken_mode_1300"

echo ""
echo "===== Attack run finished at $(date) ====="