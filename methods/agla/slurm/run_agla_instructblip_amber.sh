#!/bin/bash
#SBATCH --job-name=agla-ib-amb
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_iblip_amber_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_iblip_amber_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

echo "===== AGLA + InstructBLIP on AMBER (1004 images) ====="
python run_instructblip_chair.py \
    --image-folder /scratch/mmarvani/LVLM/datasets/AMBER/image \
    --question-file ../data/AMBER/amber_generative.jsonl \
    --answers-file /scratch/mmarvani/LVLM/lvlm-logs/AGLA/instructblip_amber_agla.jsonl \
    --use_agla \
    --alpha 2 \
    --beta 0.5 \
    --seed 42

echo "===== Complete ====="
