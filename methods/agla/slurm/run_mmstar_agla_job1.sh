#!/bin/bash
#SBATCH --job-name=mmstar-agla1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/mmstar_agla1_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/mmstar_agla1_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

echo "===== AGLA + LLaVA-1.5 on MMStar ====="
python run_mmstar_agla.py \
    --model_type llava \
    --model-path liuhaotian/llava-v1.5-7b \
    --augmented_folder /path/to/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --output_file /path/to/LVLM/lvlm-logs/MMStar/llava_agla.jsonl \
    --use_agla --alpha 2 --beta 0.5 --seed 42
echo "===== LLaVA-1.5 complete ====="

echo "===== AGLA + InstructBLIP on MMStar ====="
python run_mmstar_agla.py \
    --model_type instructblip \
    --augmented_folder /path/to/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --output_file /path/to/LVLM/lvlm-logs/MMStar/instructblip_agla.jsonl \
    --use_agla --alpha 2 --beta 0.5 --seed 42
echo "===== InstructBLIP complete ====="
