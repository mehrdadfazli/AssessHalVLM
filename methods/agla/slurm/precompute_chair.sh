#!/bin/bash
#SBATCH --job-name=agla-aug
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/path/to/LVLM/lvlm-logs/agla_augment_%j.out
#SBATCH --error=/path/to/LVLM/lvlm-logs/agla_augment_%j.err

module load python
source /path/to/LVLM/envs/agla-env/bin/activate

export HF_HOME=/path/to/.cache/huggingface
export TORCH_HOME=/path/to/.cache/torch
export TMPDIR=/path/to/tmp
export PYTHONPATH=/path/to/LVLM/AGLA:$PYTHONPATH

cd /path/to/LVLM/AGLA/eval

echo "===== Pre-computing CHAIR augmented images ====="
python precompute_augmentation.py \
    --question_file ../data/CHAIR/chair_500.jsonl \
    --image_folder /path/to/LVLM/datasets/coco2014/val2014 \
    --output_dir /path/to/LVLM/lvlm-logs/AGLA/augmented_chair \
    --image_key image \
    --question_key text

echo "===== Pre-computing AMBER augmented images ====="
python precompute_augmentation.py \
    --question_file ../data/AMBER/amber_generative.jsonl \
    --image_folder /path/to/LVLM/datasets/AMBER/image \
    --output_dir /path/to/LVLM/lvlm-logs/AGLA/augmented_amber \
    --image_key image \
    --question_key text

echo "===== Augmentation complete ====="
