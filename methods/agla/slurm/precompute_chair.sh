#!/bin/bash
#SBATCH --job-name=agla-aug
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/mmarvani/LVLM/lvlm-logs/agla_augment_%j.out
#SBATCH --error=/scratch/mmarvani/LVLM/lvlm-logs/agla_augment_%j.err

module load python
source /scratch/mmarvani/LVLM/envs/agla-env/bin/activate

export HF_HOME=/scratch/mmarvani/.cache/huggingface
export TORCH_HOME=/scratch/mmarvani/.cache/torch
export TMPDIR=/scratch/mmarvani/tmp
export PYTHONPATH=/scratch/mmarvani/LVLM/AGLA:$PYTHONPATH

cd /scratch/mmarvani/LVLM/AGLA/eval

echo "===== Pre-computing CHAIR augmented images ====="
python precompute_augmentation.py \
    --question_file ../data/CHAIR/chair_500.jsonl \
    --image_folder /scratch/mmarvani/LVLM/datasets/coco2014/val2014 \
    --output_dir /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_chair \
    --image_key image \
    --question_key text

echo "===== Pre-computing AMBER augmented images ====="
python precompute_augmentation.py \
    --question_file ../data/AMBER/amber_generative.jsonl \
    --image_folder /scratch/mmarvani/LVLM/datasets/AMBER/image \
    --output_dir /scratch/mmarvani/LVLM/lvlm-logs/AGLA/augmented_amber \
    --image_key image \
    --question_key text

echo "===== Augmentation complete ====="
