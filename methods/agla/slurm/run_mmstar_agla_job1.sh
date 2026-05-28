#!/bin/bash
#SBATCH --job-name=mmstar-agla1
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/mmstar_agla1_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/mmstar_agla1_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

echo "===== AGLA + LLaVA-1.5 on MMStar ====="
python run_mmstar_agla.py \
    --model_type llava \
    --model-path liuhaotian/llava-v1.5-7b \
    --augmented_folder /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMStar/llava_agla.jsonl \
    --use_agla --alpha 2 --beta 0.5 --seed 42
echo "===== LLaVA-1.5 complete ====="

echo "===== AGLA + InstructBLIP on MMStar ====="
python run_mmstar_agla.py \
    --model_type instructblip \
    --augmented_folder /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_mmstar \
    --output_file /scratch/mmarvani/LVLM/lvlm-logs/MMStar/instructblip_agla.jsonl \
    --use_agla --alpha 2 --beta 0.5 --seed 42
echo "===== InstructBLIP complete ====="
